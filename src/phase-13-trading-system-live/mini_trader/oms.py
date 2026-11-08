"""
OMS 与订单状态机（第 03 课）

职责：
    1. 维护订单生命周期，只允许合法的状态转换
    2. 按 trade_id 对成交去重，持仓只由成交累计
    3. 维护 client_order_id ↔ exchange_order_id 映射
    4. 处理乱序、重复、迟到的回报；无法解释的情况记为异常（anomaly）
"""

from __future__ import annotations

import itertools
import time
from dataclasses import replace
from datetime import datetime
from typing import Callable

from .models import TERMINAL_STATUSES, OrderData, OrderRequest, Status, TradeData
from .positions import PositionBook, position_side

# 状态转换表：key 为当前状态，value 为允许到达的状态
ALLOWED_TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.SUBMITTING: frozenset(
        {Status.NOT_TRADED, Status.PART_TRADED, Status.ALL_TRADED, Status.CANCELLED, Status.REJECTED}
    ),
    Status.NOT_TRADED: frozenset({Status.PART_TRADED, Status.ALL_TRADED, Status.CANCELLED}),
    Status.PART_TRADED: frozenset({Status.PART_TRADED, Status.ALL_TRADED, Status.CANCELLED}),
    Status.ALL_TRADED: frozenset(),
    Status.CANCELLED: frozenset(),
    Status.REJECTED: frozenset(),
}
# 用于识别"过期回报"：数字越大，订单进展越靠后
_STATUS_RANK = {Status.SUBMITTING: 0, Status.NOT_TRADED: 1, Status.PART_TRADED: 2}


class InvalidTransition(Exception):
    """非法状态转换"""


class OverfillError(Exception):
    """成交量超过委托量：数据异常，需要人工介入"""


def check_transition(old: Status, new: Status) -> None:
    if new not in ALLOWED_TRANSITIONS[old]:
        raise InvalidTransition(f"非法状态转换 {old.name} -> {new.name}")


def _rank(status: Status) -> int:
    return _STATUS_RANK.get(status, 3)


class OrderIdGenerator:
    """策略前缀 + 启动时间戳 + 自增序号：进程重启后也不会与之前的订单号冲突"""

    def __init__(self, prefix: str = "S1", boot: int | None = None):
        self.prefix = prefix
        self.boot = int(time.time()) if boot is None else boot
        self._seq = itertools.count(1)

    def next(self) -> str:
        return f"{self.prefix}-{self.boot}-{next(self._seq):04d}"


class OMS:
    def __init__(self, on_anomaly: Callable[[str], None] | None = None):
        self.orders: dict[str, OrderData] = {}
        self.exid_to_coid: dict[str, str] = {}
        self.trades: list[TradeData] = []
        self.seen_trade_ids: set[str] = set()
        self.positions = PositionBook()
        self.cancel_pending: set[str] = set()
        self.unknown_orders: set[str] = set()  # 超时且查询失败：状态未知
        self.orphans: set[str] = set()  # 不是本系统发出（或记录已丢失）的订单
        self.anomalies: list[str] = []
        self._frozen: dict[str, float] = {}  # 每笔卖出 / 平仓委托尚未释放的冻结量
        self.stats = {"duplicate_trades": 0, "stale_reports": 0, "late_trades": 0}
        self._on_anomaly = on_anomaly

    # ---------- 内部工具 ----------

    def _anomaly(self, msg: str) -> None:
        self.anomalies.append(msg)
        if self._on_anomaly:
            self._on_anomaly(msg)

    def _set_status(self, order: OrderData, new: Status) -> None:
        """唯一修改订单状态的入口：先校验，再修改；进入终态时释放冻结"""
        if new == order.status:
            return
        check_transition(order.status, new)
        order.status = new
        order.updated_at = datetime.now()
        if new in TERMINAL_STATUSES:
            self.cancel_pending.discard(order.client_order_id)
            self.unknown_orders.discard(order.client_order_id)
        if new in (Status.CANCELLED, Status.REJECTED):  # 剩余部分不会再成交：释放冻结
            self._release(order, self._frozen.get(order.client_order_id, 0.0))

    def _release(self, order: OrderData, volume: float) -> None:
        left = self._frozen.get(order.client_order_id, 0.0)
        volume = min(volume, left)
        if volume <= 0:
            return
        pos_dir, _ = position_side(order.symbol, order.direction, order.offset)
        self.positions.unfreeze(order.symbol, pos_dir, volume)
        self._frozen[order.client_order_id] = left - volume

    # ---------- 下单 ----------

    def create_order(self, req: OrderRequest) -> OrderData:
        if not req.client_order_id:
            raise ValueError("下单前必须先生成 client_order_id")
        if req.client_order_id in self.orders:
            raise ValueError(f"client_order_id 重复：{req.client_order_id}")
        order = OrderData(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            direction=req.direction,
            price=req.price,
            volume=req.volume,
            offset=req.offset,
            strategy=req.strategy,
        )
        self.orders[order.client_order_id] = order
        pos_dir, closing = position_side(req.symbol, req.direction, req.offset)
        if closing:  # 卖出 / 平仓先冻结持仓，防止多笔委托重复卖出同一份持仓
            self.positions.freeze(req.symbol, pos_dir, req.volume)
            self._frozen[req.client_order_id] = req.volume
        return order

    def request_cancel(self, client_order_id: str) -> bool:
        order = self.orders.get(client_order_id)
        if order is None or not order.is_active:
            return False
        self.cancel_pending.add(client_order_id)
        return True

    def mark_unknown(self, client_order_id: str) -> None:
        self.unknown_orders.add(client_order_id)
        self._anomaly(f"订单 {client_order_id} 状态未知（超时且查询失败），需暂停该标的并告警")

    def mark_not_sent(self, client_order_id: str, reason: str) -> None:
        """请求确定没有发出（如发送前就断线）：本地直接置为拒单"""
        order = self.orders[client_order_id]
        order.reason = reason
        self._set_status(order, Status.REJECTED)

    def mark_missing(self, client_order_id: str, reason: str) -> None:
        """
        交易所查不到这笔未完成订单：
        还在 SUBMITTING → 请求没到达，按拒单处理；已确认过的 → 已失效（如隔日），按撤销处理
        """
        order = self.orders[client_order_id]
        order.reason = reason
        self._set_status(order, Status.REJECTED if order.status is Status.SUBMITTING else Status.CANCELLED)

    # ---------- 回报处理 ----------

    def on_order(self, report: OrderData) -> bool:
        """处理订单回报，返回本地状态是否发生变化"""
        coid = report.client_order_id
        order = self.orders.get(coid)
        if order is None:
            self.adopt(report)
            return True
        if report.exchange_order_id:
            order.exchange_order_id = report.exchange_order_id
            self.exid_to_coid[report.exchange_order_id] = coid
        if report.reason:
            order.reason = report.reason
        if report.status == order.status:
            return False  # 重复回报
        if order.status in TERMINAL_STATUSES:
            if report.status in TERMINAL_STATUSES:
                self._anomaly(
                    f"订单 {coid} 终态冲突：本地 {order.status.name}，回报 {report.status.name}"
                )
            else:
                self.stats["stale_reports"] += 1  # 终态后到达的旧回报：忽略
            return False
        if _rank(report.status) < _rank(order.status):
            self.stats["stale_reports"] += 1  # 乱序到达的旧回报：忽略
            return False
        try:
            self._set_status(order, report.status)
        except InvalidTransition as exc:
            self._anomaly(f"订单 {coid}：{exc}")
            return False
        return True

    def on_trade(self, trade: TradeData) -> bool:
        """处理成交回报，返回是否为新成交（重复回报返回 False）"""
        if trade.trade_id in self.seen_trade_ids:
            self.stats["duplicate_trades"] += 1
            return False
        self.seen_trade_ids.add(trade.trade_id)
        self.trades.append(trade)
        self.positions.on_trade(trade)  # 持仓只由成交推导

        order = self.orders.get(trade.client_order_id)
        if order is None:
            self._anomaly(f"成交 {trade.trade_id} 对应未知订单 {trade.client_order_id}")
            return True
        order.traded += trade.volume
        self._release(order, trade.volume)
        order.updated_at = datetime.now()
        if order.traded > order.volume + 1e-9:
            self._anomaly(f"订单 {order.client_order_id} 成交量 {order.traded} 超过委托量 {order.volume}")
            raise OverfillError(order.client_order_id)
        if order.status in TERMINAL_STATUSES:
            if order.status is not Status.ALL_TRADED:
                self.stats["late_trades"] += 1  # 撤单后才到达的成交：只记量，不改终态
            return True
        filled = abs(order.traded - order.volume) < 1e-9
        self._set_status(order, Status.ALL_TRADED if filled else Status.PART_TRADED)
        return True

    def on_cancel_reject(self, client_order_id: str, reason: str) -> None:
        """撤单被拒：常见原因是撤单途中订单已全部成交，属于正常竞态"""
        self.cancel_pending.discard(client_order_id)
        order = self.orders.get(client_order_id)
        if order is not None:
            order.reason = f"撤单失败：{reason}"

    def adopt(self, report: OrderData) -> OrderData:
        """收编本地没有记录的订单（孤儿订单），纳入管理"""
        order = replace(report)
        # 成交量只由成交回报累计：先按已收到的成交计数，后续成交再逐笔累加
        order.traded = sum(t.volume for t in self.trades if t.client_order_id == order.client_order_id)
        self.orders[order.client_order_id] = order
        if order.exchange_order_id:
            self.exid_to_coid[order.exchange_order_id] = order.client_order_id
        self.orphans.add(order.client_order_id)
        self._anomaly(f"发现本地无记录的订单 {order.client_order_id}（{order.status.name}），已收编")
        return order

    # ---------- 查询 ----------

    def get_order(self, client_order_id: str) -> OrderData | None:
        return self.orders.get(client_order_id)

    def find_by_exchange_id(self, exchange_order_id: str) -> OrderData | None:
        coid = self.exid_to_coid.get(exchange_order_id)
        return self.orders.get(coid) if coid else None

    def active_orders(self, symbol: str | None = None) -> list[OrderData]:
        return [o for o in self.orders.values() if o.is_active and (symbol is None or o.symbol == symbol)]
