"""
风控引擎（第 04 课）

事前（Pre-trade）：每个订单发出前逐条检查，任一规则不通过即拒绝
事中（Intra-day）：持续监控，触发 Kill Switch 后拒绝一切新单，并通知撤单
设计原则：自动暂停可以，恢复必须人工确认，且要先完成对账
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

from .contracts import get_spec, required_funds
from .models import AccountData, Market, OrderData, OrderRequest, Status, TickData
from .positions import PositionBook, position_side


@dataclass
class RiskConfig:
    max_order_volume: float = 1_000  # 单笔数量上限（防"胖手指"）
    max_order_notional: float = 500_000.0  # 单笔金额上限（按合约价值计）
    max_position: float = 5_000  # 单标的单方向持仓上限
    max_price_deviation: float = 0.05  # 限价偏离最新价的上限
    max_orders_per_window: int = 5  # 报单频率：窗口内最多 N 笔
    order_window_sec: float = 1.0
    max_daily_loss: float = 20_000.0  # 日内亏损上限（正数）
    max_consecutive_rejects: int = 3
    max_tick_silence_sec: float = 5.0  # 行情中断判定
    symbol_whitelist: frozenset[str] | None = None


@dataclass
class RiskResult:
    ok: bool
    rule: str = ""
    reason: str = ""


class RiskEngine:
    # 检查顺序：先检查"开关"，再检查便宜的静态规则，最后检查有状态的流控
    RULES = (
        "kill_switch",
        "symbol_halt",
        "whitelist",
        "volume",
        "price_reference",
        "price_limit",
        "price_deviation",
        "notional",
        "position_limit",
        "funds",
        "available_position",
        "order_rate",
    )

    def __init__(self, config: RiskConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = config or RiskConfig()
        self.clock = clock
        self.last_tick: dict[str, TickData] = {}
        self.last_tick_time: dict[str, float] = {}
        self.order_times: deque[float] = deque()
        self.halted = False
        self.halt_reason = ""
        self.halted_symbols: dict[str, str] = {}
        self.consecutive_rejects = 0
        self._rejected_orders: set[str] = set()  # 同一订单的重复回报只计一次
        self.daily_pnl = 0.0
        self.reject_counts: Counter[str] = Counter()
        self.halt_listeners: list[Callable[[str], None]] = []
        self.events: list[str] = []

    # ==================== 输入 ====================

    def update_tick(self, tick: TickData) -> None:
        self.last_tick[tick.symbol] = tick
        self.last_tick_time[tick.symbol] = self.clock()

    # ==================== 事前检查 ====================

    def check_order(
        self, req: OrderRequest, account: AccountData | None = None, positions: PositionBook | None = None
    ) -> RiskResult:
        for rule in self.RULES:
            reason = getattr(self, f"_rule_{rule}")(req, account, positions)
            if reason:
                self.reject_counts[rule] += 1
                return RiskResult(False, rule, reason)
        self.order_times.append(self.clock())
        return RiskResult(True)

    def _rule_kill_switch(self, req, account, positions):
        return f"交易已暂停（Kill Switch）：{self.halt_reason}" if self.halted else ""

    def _rule_symbol_halt(self, req, account, positions):
        reason = self.halted_symbols.get(req.symbol)
        return f"{req.symbol} 已暂停交易：{reason}" if reason else ""

    def _rule_whitelist(self, req, account, positions):
        wl = self.cfg.symbol_whitelist
        return f"{req.symbol} 不在交易白名单中" if wl is not None and req.symbol not in wl else ""

    def _rule_volume(self, req, account, positions):
        if req.volume <= 0 or req.volume > self.cfg.max_order_volume:
            return f"数量 {req.volume:g} 超出范围 (0, {self.cfg.max_order_volume:g}]"
        return ""

    def _rule_price_reference(self, req, account, positions):
        return "" if req.symbol in self.last_tick else "无参考价（没有收到行情），拒绝下单"

    def _rule_price_limit(self, req, account, positions):
        tick = self.last_tick[req.symbol]
        if tick.limit_up and req.price > tick.limit_up:
            return f"价格 {req.price} 高于涨停价 {tick.limit_up}"
        if tick.limit_down and req.price < tick.limit_down:
            return f"价格 {req.price} 低于跌停价 {tick.limit_down}"
        return ""

    def _rule_price_deviation(self, req, account, positions):
        ref = self.last_tick[req.symbol].last_price
        dev = req.price / ref - 1
        if abs(dev) > self.cfg.max_price_deviation:
            return f"价格偏离最新价 {dev:+.2%}，超过 ±{self.cfg.max_price_deviation:.0%}"
        return ""

    def _rule_notional(self, req, account, positions):
        notional = req.price * req.volume * get_spec(req.symbol).size
        if notional > self.cfg.max_order_notional:
            return f"单笔金额 {notional:,.0f} 超过上限 {self.cfg.max_order_notional:,.0f}"
        return ""

    def _rule_position_limit(self, req, account, positions):
        pos_dir, closing = position_side(req.symbol, req.direction, req.offset)
        if closing or positions is None:
            return ""
        after = positions.get(req.symbol, pos_dir).volume + req.volume
        if after > self.cfg.max_position:
            return f"下单后持仓 {after:g} 将超过上限 {self.cfg.max_position:g}"
        return ""

    def _rule_funds(self, req, account, positions):
        _, closing = position_side(req.symbol, req.direction, req.offset)
        if closing or account is None:
            return ""
        need = required_funds(req.symbol, req.price, req.volume)
        if need > account.available:
            return f"资金不足：需要 {need:,.0f}，可用 {account.available:,.0f}"
        return ""

    def _rule_available_position(self, req, account, positions):
        pos_dir, closing = position_side(req.symbol, req.direction, req.offset)
        if not closing or positions is None:
            return ""
        spec = get_spec(req.symbol)
        if spec.market is Market.FUTURES:
            free = positions.closable(req.symbol, pos_dir, req.offset)
        else:
            free = positions.available(req.symbol, pos_dir)
        if free < req.volume:
            note = "（T+1：今日买入部分不可卖）" if spec.t_plus_one else ""
            return f"可用持仓 {free:g} 少于委托量 {req.volume:g}{note}"
        return ""

    def _rule_order_rate(self, req, account, positions):
        now = self.clock()
        while self.order_times and now - self.order_times[0] > self.cfg.order_window_sec:
            self.order_times.popleft()
        if len(self.order_times) >= self.cfg.max_orders_per_window:
            return f"报单频率超限：{self.cfg.order_window_sec:g} 秒内已报 {len(self.order_times)} 笔"
        return ""

    # ==================== 事中监控 ====================

    def on_order_update(self, order: OrderData) -> None:
        """连续拒单往往意味着账户、资金或接口出了问题"""
        if order.status is Status.REJECTED:
            if order.client_order_id in self._rejected_orders:
                return
            self._rejected_orders.add(order.client_order_id)
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= self.cfg.max_consecutive_rejects:
                self.halt(f"连续拒单 {self.consecutive_rejects} 笔（最近原因：{order.reason}）")
        elif order.status in (Status.NOT_TRADED, Status.PART_TRADED, Status.ALL_TRADED):
            self.consecutive_rejects = 0

    def on_pnl(self, daily_pnl: float) -> None:
        self.daily_pnl = daily_pnl
        if daily_pnl <= -self.cfg.max_daily_loss:
            self.halt(f"日内亏损 {daily_pnl:,.0f} 触及上限 -{self.cfg.max_daily_loss:,.0f}")

    def check_market_data(self, symbols: list[str] | None = None, in_session: bool = True) -> list[str]:
        """行情中断检测：单个标的停滞 → 暂停该标的；全部停滞 → 全局 Kill Switch"""
        if not in_session:  # 非交易时段没有行情是正常的
            return []
        now = self.clock()
        watch = symbols if symbols is not None else list(self.last_tick_time)
        stale = [s for s in watch if now - self.last_tick_time.get(s, float("-inf")) > self.cfg.max_tick_silence_sec]
        for s in stale:
            self.halt_symbol(s, f"行情中断超过 {self.cfg.max_tick_silence_sec:g} 秒")
        if watch and len(stale) == len(watch):
            self.halt("全部标的行情中断，疑似网关断线")
        return stale

    def halt_symbol(self, symbol: str, reason: str) -> None:
        if symbol not in self.halted_symbols:
            self.halted_symbols[symbol] = reason
            self.events.append(f"暂停标的 {symbol}：{reason}")

    def halt(self, reason: str, notify: bool = True) -> None:
        """
        全局暂停。notify=True 时通知监听者（撤销全部挂单）；
        启动恢复期间用 notify=False：只禁止新单，不能在状态未核实前动现有挂单。
        """
        if self.halted:
            return
        self.halted, self.halt_reason = True, reason
        self.events.append(f"KILL SWITCH：{reason}")
        if notify:
            for fn in self.halt_listeners:
                fn(reason)

    def resume(self, confirmed_by: str, reconciled: bool) -> bool:
        """人工恢复：必须有确认人，且已完成对账"""
        if not self.halted:
            return True
        if not confirmed_by or not reconciled:
            self.events.append("恢复被拒绝：需要人工确认且完成对账")
            return False
        self.events.append(f"交易已恢复（确认人：{confirmed_by}，原暂停原因：{self.halt_reason}）")
        self.halted, self.halt_reason = False, ""
        self.consecutive_rejects = 0
        self.halted_symbols.clear()
        return True

    def resume_symbol(self, symbol: str, confirmed_by: str) -> None:
        if self.halted_symbols.pop(symbol, None) is not None:
            self.events.append(f"恢复标的 {symbol}（确认人：{confirmed_by}）")
