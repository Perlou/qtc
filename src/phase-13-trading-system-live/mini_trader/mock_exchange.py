"""
模拟交易所（MockExchange）

在进程内模拟一个"交易所 + 柜台"：撮合限价单、维护订单 / 成交 / 持仓 / 资金的"真相"，
并可以注入故障，让所有实盘才会遇到的问题都能离线复现：

    reject              下一笔委托被拒
    lost_request        请求在途中丢失：交易所没收到，客户端超时
    lost_response       交易所已收到并处理，但响应和回报都丢失：客户端超时
    duplicate_trade     下一笔成交回报推送两次
    trade_before_order  成交回报先于订单确认到达（乱序）
    disconnect()        断线：请求失败、回报丢失，但交易所照常撮合
    reconnect(replay)   重连；replay=True 时重推当日全部回报（类似 CTP 重新登录）
    adjust_position()   交易所侧调整持仓（如强平、送股），本地不会收到成交

撮合规则（简化）：买单价格 ≥ 卖一价时以卖一价成交，卖单价格 ≤ 买一价时以买一价成交，
成交量受该档挂单量限制，不足部分继续挂单，从而产生部分成交。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable

from .contracts import get_spec, required_funds
from .models import (
    AccountData,
    Direction,
    Market,
    Offset,
    OrderData,
    OrderRequest,
    PositionData,
    Status,
    TickData,
    TradeData,
)
from .positions import PositionBook, position_side

Listener = Callable[[str, Any], None]  # (kind, data)，kind ∈ tick / order / trade / cancel_reject

FAULT_TYPES = {
    "reject": "下一笔委托被拒",
    "lost_request": "请求丢失（交易所未收到）",
    "lost_response": "响应丢失（交易所已处理）",
    "duplicate_trade": "成交回报重复推送",
    "trade_before_order": "成交回报先于订单回报到达",
}


class ExchangeError(Exception):
    pass


class DuplicateOrderError(ExchangeError):
    """client_order_id 重复：很多交易所会直接拒绝，这是防止重复下单的最后一道防线"""


class ExchangeTimeout(TimeoutError):
    pass


class ExchangeDisconnected(ConnectionError):
    pass


class MockExchange:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        account_id: str = "MOCK-001",
        commission_rate: float = 0.0,
        clock: Callable[[], datetime] = datetime.now,
        default_depth: float = 1e9,
    ):
        self.account = AccountData(account_id, balance=initial_cash)
        self.positions = PositionBook()
        self.orders: dict[str, OrderData] = {}  # 当日订单（真相）
        self.trades: list[TradeData] = []  # 当日成交（真相）
        self.quotes: dict[str, TickData] = {}
        self.commission_rate = commission_rate
        self.clock = clock
        self.default_depth = default_depth
        self.connected = True
        self.trading_day = 1
        self.dropped_reports = 0
        self.fault_log: list[str] = []
        self._listeners: list[Listener] = []
        self._faults: dict[str, int] = defaultdict(int)
        self._all_coids: set[str] = set()
        self._unit_frozen: dict[str, float] = {}  # 每单位委托冻结的资金
        self._order_seq = 0
        self._trade_seq = 0
        self._buffer: list[tuple[str, Any]] = []
        self._suppress = False

    # ==================== 订阅回报 ====================

    def add_listener(self, fn: Listener) -> None:
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ==================== 故障注入 ====================

    def inject(self, fault: str, count: int = 1) -> None:
        if fault not in FAULT_TYPES:
            raise ValueError(f"未知故障类型：{fault}，可选 {sorted(FAULT_TYPES)}")
        self._faults[fault] += count

    def _take(self, fault: str) -> bool:
        if self._faults[fault] > 0:
            self._faults[fault] -= 1
            self.fault_log.append(f"{fault}：{FAULT_TYPES[fault]}")
            return True
        return False

    def disconnect(self) -> None:
        self.connected = False
        self.fault_log.append("disconnect：连接断开")

    def reconnect(self, replay: bool = False) -> None:
        self.connected = True
        if replay:  # 重推当日全部订单与成交回报
            self._buffer += [("order", replace(o)) for o in self.orders.values()]
            self._buffer += [("trade", replace(t)) for t in self.trades]
            self._flush()

    def adjust_position(self, symbol: str, direction: Direction, delta: float) -> None:
        """交易所侧持仓调整（强平、送股等）：不产生本地能收到的成交回报"""
        p = self.positions.get(symbol, direction)
        p.volume += delta
        p.yd_volume = max(min(p.yd_volume + delta, p.volume), 0.0)
        self.fault_log.append(f"adjust_position：{symbol} {direction.name} {delta:+g}")

    # ==================== 行情 ====================

    def set_market(
        self,
        symbol: str,
        last: float,
        bid: float | None = None,
        ask: float | None = None,
        bid_volume: float | None = None,
        ask_volume: float | None = None,
        limit_up: float = 0.0,
        limit_down: float = 0.0,
    ) -> TickData:
        """更新行情并推送 Tick；随后尝试撮合该标的的挂单"""
        tick_size = get_spec(symbol).price_tick
        tick = TickData(
            symbol=symbol,
            dt=self.clock(),
            last_price=last,
            bid_price_1=bid if bid is not None else round(last - tick_size, 8),
            ask_price_1=ask if ask is not None else round(last + tick_size, 8),
            bid_volume_1=self.default_depth if bid_volume is None else bid_volume,
            ask_volume_1=self.default_depth if ask_volume is None else ask_volume,
            limit_up=limit_up,
            limit_down=limit_down,
        )
        self.quotes[symbol] = tick
        self._buffer.append(("tick", replace(tick)))
        for order in [o for o in self.orders.values() if o.symbol == symbol and o.is_active]:
            self._match(order)
        self._flush()
        return tick

    # ==================== 交易 ====================

    def submit(self, req: OrderRequest) -> OrderData:
        """接收委托，返回受理结果（订单快照）"""
        if not self.connected:
            raise ExchangeDisconnected("与交易所的连接已断开")
        if self._take("lost_request"):
            raise ExchangeTimeout("请求超时（请求未到达交易所）")
        if req.client_order_id in self._all_coids:
            raise DuplicateOrderError(f"重复的 client_order_id：{req.client_order_id}")
        self._all_coids.add(req.client_order_id)

        self._order_seq += 1
        order = OrderData(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            direction=req.direction,
            price=req.price,
            volume=req.volume,
            offset=req.offset,
            strategy=req.strategy,
            exchange_order_id=f"EX{self._order_seq:06d}",
        )
        lost_response = self._take("lost_response")
        self._suppress = lost_response
        try:
            reason = self._validate(req)
            self.orders[order.client_order_id] = order
            if reason:
                order.status, order.reason = Status.REJECTED, reason
                self._buffer.append(("order", replace(order)))
            else:
                order.status = Status.NOT_TRADED
                self._freeze(order)
                self._buffer.append(("order", replace(order)))
                self._match(order)
            snapshot = replace(order)
            self._flush()
        finally:
            self._suppress = False
        if lost_response:
            raise ExchangeTimeout("请求超时（交易所已处理，但响应丢失）")
        return snapshot

    def cancel(self, client_order_id: str) -> bool:
        if not self.connected:
            raise ExchangeDisconnected("与交易所的连接已断开")
        order = self.orders.get(client_order_id)
        if order is None:
            raise ExchangeError(f"订单不存在：{client_order_id}")
        if not order.is_active:
            reason = f"订单已是终态（{order.status.value}）"
            self._buffer.append(("cancel_reject", {"client_order_id": client_order_id, "reason": reason}))
            self._flush()
            return False
        self._release(order, order.remaining)
        order.status = Status.CANCELLED
        order.updated_at = self.clock()
        self._buffer.append(("order", replace(order)))
        self._flush()
        return True

    def settle(self) -> None:
        """日终：未成交挂单失效，今仓转昨仓，进入下一个交易日"""
        for order in self.orders.values():
            if order.is_active:
                self._release(order, order.remaining)
                order.status, order.reason = Status.CANCELLED, "收盘未成交，订单失效"
                self._buffer.append(("order", replace(order)))
        self._flush()
        self.positions.settle()
        self.orders, self.trades = {}, []
        self.trading_day += 1

    # ==================== 查询（交易所是真相来源） ====================

    def _check_conn(self) -> None:
        if not self.connected:
            raise ExchangeDisconnected("与交易所的连接已断开")

    def query_order(self, client_order_id: str) -> OrderData | None:
        self._check_conn()
        order = self.orders.get(client_order_id)
        return replace(order) if order else None

    def query_orders(self) -> list[OrderData]:
        self._check_conn()
        return [replace(o) for o in self.orders.values()]

    def query_open_orders(self) -> list[OrderData]:
        return [o for o in self.query_orders() if o.is_active]

    def query_trades(self, client_order_id: str | None = None) -> list[TradeData]:
        self._check_conn()
        return [replace(t) for t in self.trades if client_order_id in (None, t.client_order_id)]

    def query_positions(self) -> list[PositionData]:
        self._check_conn()
        return [replace(p) for p in self.positions.all()]

    def query_account(self) -> AccountData:
        self._check_conn()
        return replace(self.account)

    # ==================== 内部：校验、冻结、撮合 ====================

    def _validate(self, req: OrderRequest) -> str:
        spec = get_spec(req.symbol)
        if self._take("reject"):
            return "模拟拒单（故障注入）"
        if req.volume <= 0:
            return "委托数量必须为正"
        if spec.market is Market.FUTURES and req.offset is Offset.NONE:
            return "期货委托必须指定开平"
        steps = req.price / spec.price_tick
        if abs(steps - round(steps)) > 1e-6:
            return f"价格 {req.price} 不是最小变动价位 {spec.price_tick} 的整数倍"
        quote = self.quotes.get(req.symbol)
        if quote and quote.limit_up and req.price > quote.limit_up + 1e-9:
            return f"委托价 {req.price} 高于涨停价 {quote.limit_up}"
        if quote and quote.limit_down and req.price < quote.limit_down - 1e-9:
            return f"委托价 {req.price} 低于跌停价 {quote.limit_down}"

        pos_dir, closing = position_side(req.symbol, req.direction, req.offset)
        if closing and spec.market is Market.FUTURES:
            free = self.positions.closable(req.symbol, pos_dir, req.offset)
            if free < req.volume:
                return f"可平仓位不足（{req.offset.value}：可平 {free:g}，委托 {req.volume:g}）"
        elif closing:
            free = self.positions.available(req.symbol)
            if free < req.volume:
                note = "；T+1：今日买入不可卖出" if spec.t_plus_one else ""
                return f"可用持仓不足（可卖 {free:g}，委托 {req.volume:g}{note}）"
        else:
            if spec.market is Market.STOCK:
                lots = req.volume / spec.lot_size
                if abs(lots - round(lots)) > 1e-9:
                    return f"买入数量须为 {spec.lot_size:g} 的整数倍"
            need = required_funds(req.symbol, req.price, req.volume) * (1 + self.commission_rate)
            if need > self.account.available + 1e-6:
                return f"资金不足（需要 {need:,.2f}，可用 {self.account.available:,.2f}）"
        return ""

    def _freeze(self, order: OrderData) -> None:
        pos_dir, closing = position_side(order.symbol, order.direction, order.offset)
        if closing:
            self.positions.freeze(order.symbol, pos_dir, order.volume)
        else:
            unit = required_funds(order.symbol, order.price, 1) * (1 + self.commission_rate)
            self._unit_frozen[order.client_order_id] = unit
            self.account.frozen += unit * order.volume

    def _release(self, order: OrderData, volume: float) -> None:
        pos_dir, closing = position_side(order.symbol, order.direction, order.offset)
        if closing:
            self.positions.unfreeze(order.symbol, pos_dir, volume)
        else:
            unit = self._unit_frozen.get(order.client_order_id, 0.0)
            self.account.frozen = max(self.account.frozen - unit * volume, 0.0)

    def _match(self, order: OrderData) -> None:
        quote = self.quotes.get(order.symbol)
        if quote is None or not order.is_active:
            return
        if order.direction is Direction.LONG and quote.ask_price_1 and order.price >= quote.ask_price_1:
            qty = min(order.remaining, quote.ask_volume_1)
            price = quote.ask_price_1
            quote.ask_volume_1 -= qty
        elif order.direction is Direction.SHORT and quote.bid_price_1 and order.price <= quote.bid_price_1:
            qty = min(order.remaining, quote.bid_volume_1)
            price = quote.bid_price_1
            quote.bid_volume_1 -= qty
        else:
            return
        if qty > 0:
            self._fill(order, qty, price)

    def _fill(self, order: OrderData, qty: float, price: float) -> None:
        spec = get_spec(order.symbol)
        self._trade_seq += 1
        trade = TradeData(
            trade_id=f"T{self.trading_day:02d}{self._trade_seq:06d}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            direction=order.direction,
            price=price,
            volume=qty,
            offset=order.offset,
            exchange_order_id=order.exchange_order_id,
            dt=self.clock(),
        )
        self.trades.append(trade)
        self._release(order, qty)
        order.traded += qty
        order.status = Status.ALL_TRADED if order.remaining <= 1e-9 else Status.PART_TRADED
        order.updated_at = trade.dt

        pnl = self.positions.on_trade(trade)
        notional = price * qty * spec.size
        fee = notional * self.commission_rate
        if spec.market is Market.FUTURES:
            self.account.balance += pnl - fee
            self.account.margin = sum(
                p.volume * p.price * get_spec(p.symbol).size * get_spec(p.symbol).margin_ratio
                for p in self.positions.all()
                if get_spec(p.symbol).market is Market.FUTURES
            )
        elif order.direction is Direction.LONG:
            self.account.balance -= notional + fee
        else:
            self.account.balance += notional - fee
        self._buffer.append(("trade", replace(trade)))
        self._buffer.append(("order", replace(order)))

    def _flush(self) -> None:
        reports, self._buffer = self._buffer, []
        if self._suppress or not self.connected:
            self.dropped_reports += sum(1 for kind, _ in reports if kind != "tick")
            return
        if any(kind == "trade" for kind, _ in reports) and self._take("trade_before_order"):
            reports = [r for r in reports if r[0] == "trade"] + [r for r in reports if r[0] != "trade"]
        for kind, data in reports:
            repeat = 2 if kind == "trade" and self._take("duplicate_trade") else 1
            for _ in range(repeat):
                for fn in list(self._listeners):
                    fn(kind, data)
