"""
执行层：把调仓指令交给 Phase 13 的 mini_trader 执行

    to_exchange_symbol   6 位代码 → mini_trader 统一代码（600000 → 600000.SSE）
    build_market         用面板的下一交易日开盘价构造模拟盘口（含涨跌停价、可成交深度、停牌）
    MiniTraderExecutor   TradingSystem（风控 → OMS → 网关）+ MockExchange：
                         指令校验 → 先卖后买 → 部分成交 / 拒单处理 → 日终失效 → 对账

复用 mini_trader 的 MockExchange、TradingSystem（含 RiskEngine、OMS、超时先查后发）、
recovery.reconcile_positions；交易规则与成本复用 Phase 4 bt_core。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _paths  # noqa: F401
from ..contracts import RebalanceOrder

from bt_core import CostModel, limit_prices, round_lot_buy, round_price  # noqa: E402
from ..rules import board_rule
from mini_trader import ContractSpec, Direction, Market, MockExchange, RiskConfig, register_spec  # noqa: E402
from mini_trader.recovery import positions_to_map, reconcile_positions  # noqa: E402
from mini_trader.system import TradingSystem  # noqa: E402


# ==================== 代码映射 ====================


def to_exchange_symbol(code: str) -> str:
    code = str(code)
    if code.startswith(("4", "8", "92")):
        sym = f"{code}.BSE"
    elif code.startswith(("6", "9")):
        sym = f"{code}.SSE"
    else:
        sym = f"{code}.SZSE"
    if code.startswith(("688", "689")):
        # mini_trader 默认 A 股买入须为 100 股整数倍；科创板是"至少 200 股、按 1 股递增"，
        # 注册单独的合约规格（最低 200 股由组合层 board_rule 保证）
        register_spec(ContractSpec(sym, Market.STOCK, lot_size=1, t_plus_one=True))
    return sym


def from_exchange_symbol(symbol: str) -> str:
    return symbol.split(".")[0]


# ==================== 模拟盘口 ====================


@dataclass
class MarketSnapshot:
    symbol: str
    open: float  # 到达价：下一交易日开盘价（不复权）
    pre_close: float  # 除权参考前收盘
    limit_up: float
    limit_down: float
    suspended: bool
    depth: float  # 开盘时买一 / 卖一可成交数量（股）
    note: str = ""


def build_market(source, trade_date, codes, depth_ratio: float = 0.02,
                 overrides: dict[str, dict] | None = None) -> dict[str, MarketSnapshot]:
    """
    depth_ratio：假设开盘时盘口可成交数量 = 当日成交量 × depth_ratio（粗略近似）
    overrides：情景注入，如 {"600000": {"open": 涨停价, "note": "演示：开盘涨停"}}
    """
    p = source.panel
    d = pd.Timestamp(trade_date)
    i = p.dates.get_loc(d)
    prev = p.dates[i - 1]
    out = {}
    for code in codes:
        o = p.open.at[d, code]
        vol = p.volume.at[d, code]
        f_t, f_p = p.adj_factor.at[d, code], p.adj_factor.at[prev, code]
        pc = p.close_ffill.at[prev, code]
        ref = round_price(pc * f_p / f_t) if pd.notna(pc) and pd.notna(f_t) and pd.notna(f_p) else np.nan
        up, down = limit_prices(ref, board_rule(code).limit_pct) if pd.notna(ref) else (np.nan, np.nan)
        suspended = not (pd.notna(o) and pd.notna(vol) and vol > 0)
        depth = 0.0 if suspended else float(max(vol * depth_ratio // 100 * 100, 100))
        snap = MarketSnapshot(code, float(o) if pd.notna(o) else np.nan, ref, up, down, suspended, depth)
        for k, v in (overrides or {}).get(code, {}).items():
            setattr(snap, k, v)
        out[code] = snap
    return out


# ==================== 执行报告 ====================


@dataclass
class ExecutionReport:
    trade_date: pd.Timestamp
    fills: pd.DataFrame  # 每条指令一行
    skipped: list[dict] = field(default_factory=list)  # 指令校验未通过（停牌、涨跌停）
    extra: list[dict] = field(default_factory=list)  # 额外请求（如演示 T+1）的结果
    position_diffs: list[dict] = field(default_factory=list)  # 对账：本地 vs 交易所
    cash_expected: float = 0.0
    cash_actual: float = 0.0
    risk_events: list[str] = field(default_factory=list)
    risk_rejects: dict = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    @property
    def cash_diff(self) -> float:
        return self.cash_actual - self.cash_expected

    @property
    def reconciled(self) -> bool:
        return not self.position_diffs and abs(self.cash_diff) < 0.01

    def summary(self) -> dict:
        f = self.fills
        ordered_value = float((f["ordered"] * f["ref_price"]).sum()) if len(f) else 0.0
        filled_value = float(f["filled_value"].sum()) if len(f) else 0.0
        filled = f[f["filled"] > 0]
        w = filled["filled_value"]
        return {
            "指令数": len(f) + len(self.skipped),
            "发送数": len(f),
            "校验拦截": len(self.skipped),
            "全部成交": int((f["status"] == "全部成交").sum()) if len(f) else 0,
            "部分成交": int(((f["filled"] > 0) & (f["filled"] < f["ordered"])).sum()) if len(f) else 0,
            "拒单": int(f["status"].isin(["拒单", "风控拒绝"]).sum()) if len(f) else 0,
            "成交率(金额)": filled_value / ordered_value if ordered_value > 0 else np.nan,
            "到达价偏差(bp,加权)": float((filled["arrival_bp"] * w).sum() / w.sum()) if w.sum() > 0 else np.nan,
            "执行缺口(bp,加权)": float((filled["shortfall_bp"] * w).sum() / w.sum()) if w.sum() > 0 else np.nan,
            "费用(元)": float(f["fee"].sum()) if len(f) else 0.0,
            "对账": "一致" if self.reconciled else "不一致",
        }


# ==================== 执行器 ====================


class MiniTraderExecutor:
    """
    一个交易日一次 execute()：每天新建 TradingSystem（进程重启也是这样），
    日初以交易所持仓为准初始化本地 OMS，日终撤销未成交挂单并对账。MockExchange 跨日保存"真相"。
    """

    def __init__(self, initial_cash: float, cost_model: CostModel | None = None, price_buffer: float = 0.003,
                 commission_rate: float = 0.00025, risk_config: RiskConfig | None = None):
        self.exchange = MockExchange(initial_cash=initial_cash, commission_rate=commission_rate)
        self.cost = cost_model or CostModel()
        self.price_buffer = price_buffer
        self.commission_rate = commission_rate
        self.initial_cash = initial_cash
        self.risk_config = risk_config
        self.day = 0

    # ---------- 查询（交易所是真相来源） ----------

    def positions(self) -> pd.Series:
        pos = {from_exchange_symbol(p.symbol): int(p.volume) for p in self.exchange.query_positions() if p.volume > 0}
        return pd.Series(pos, dtype="int64")

    def cash(self) -> float:
        return float(self.exchange.query_account().balance)

    def equity(self, prices: pd.Series) -> float:
        pos = self.positions()
        return self.cash() + float((pos * prices.reindex(pos.index)).sum())

    # ---------- 执行 ----------

    def execute(self, orders: list[RebalanceOrder], market: dict[str, MarketSnapshot],
                inject_reject: set[str] | None = None, extra_requests: list[dict] | None = None,
                trade_date=None) -> ExecutionReport:
        return asyncio.run(self._run_day(orders, market, inject_reject or set(), extra_requests or [], trade_date))

    def _risk_config(self, symbols: set[str]) -> RiskConfig:
        if self.risk_config is not None:
            return self.risk_config
        cap = max(self.initial_cash, 1.0)
        return RiskConfig(
            max_order_volume=10_000_000, max_order_notional=cap * 0.25, max_position=100_000_000,
            max_price_deviation=0.05, max_orders_per_window=10_000, order_window_sec=1.0,
            max_daily_loss=cap * 0.05, max_consecutive_rejects=3, symbol_whitelist=frozenset(symbols),
        )

    def _limit_price(self, side: str, snap: MarketSnapshot) -> float:
        if side == "BUY":
            return min(round_price(snap.open * (1 + self.price_buffer)), snap.limit_up)
        return max(round_price(snap.open * (1 - self.price_buffer)), snap.limit_down)

    def _precheck(self, o: RebalanceOrder, snap: MarketSnapshot | None) -> str:
        if snap is None:
            return "无行情"
        if snap.suspended or not (snap.open > 0):
            return "停牌，无法成交"
        if o.side == "BUY" and snap.open >= snap.limit_up - 1e-9:
            return "开盘涨停，买不进（次日重新评估，不追单）"
        if o.side == "SELL" and snap.open <= snap.limit_down + 1e-9:
            return "开盘跌停，卖不出（次日重新评估）"
        return ""

    async def _run_day(self, orders, market, inject_reject, extra_requests, trade_date) -> ExecutionReport:
        self.day += 1
        ex = self.exchange
        symbols = {to_exchange_symbol(o.symbol) for o in orders} | {to_exchange_symbol(r["symbol"]) for r in extra_requests}
        system = TradingSystem(ex, self._risk_config(symbols), prefix=f"MQ{self.day:02d}", timer_interval=None)
        await system.start()
        # 日初：以交易所持仓为准初始化本地持仓（交易所是真相来源）
        system.oms.positions.load(await system.gateway.query_positions())
        start_balance = ex.account.balance
        for sym in symbols:
            await system.gateway.subscribe(sym)
            snap = market.get(from_exchange_symbol(sym))
            if snap is not None and not snap.suspended and snap.open > 0:
                ex.set_market(sym, snap.open, bid_volume=snap.depth, ask_volume=snap.depth,
                              limit_up=snap.limit_up, limit_down=snap.limit_down)
        await system.engine.join()

        sent, skipped = [], []
        for phase in ("SELL", "BUY"):  # 先卖后买
            if phase == "BUY":
                await system.engine.join()
                system.account = await system.gateway.query_account()  # 卖出后刷新可用资金
            for o in [x for x in orders if x.side == phase]:
                snap = market.get(o.symbol)
                why = self._precheck(o, snap)
                if why:
                    skipped.append({"symbol": o.symbol, "side": o.side, "volume": o.volume, "reason": why})
                    continue
                price = self._limit_price(o.side, snap)
                vol, note = o.volume, ""
                if o.side == "BUY":
                    avail = (await system.gateway.query_account()).available
                    need = price * vol * (1 + self.commission_rate)
                    if need > avail:
                        vol = round_lot_buy(avail / (price * (1 + self.commission_rate)), board_rule(o.symbol))
                        note = f"资金不足，缩量 {o.volume} → {vol}"
                        if vol <= 0:
                            skipped.append({"symbol": o.symbol, "side": o.side, "volume": o.volume, "reason": "资金不足"})
                            continue
                sym = to_exchange_symbol(o.symbol)
                if o.symbol in inject_reject:
                    ex.inject("reject")  # 情景注入：交易所拒单
                direction = Direction.LONG if o.side == "BUY" else Direction.SHORT
                req = system.new_request(sym, direction, price, vol)
                res = await system.send_order(req)
                sent.append((o, req, res, snap, note))
            await system.engine.join()

        extra_results = []
        for r in extra_requests:
            sym = to_exchange_symbol(r["symbol"])
            snap = market[r["symbol"]]
            direction = Direction.LONG if r["side"] == "BUY" else Direction.SHORT
            req = system.new_request(sym, direction, self._limit_price(r["side"], snap), r["volume"])
            res = await system.send_order(req)
            extra_results.append({**r, "outcome": res.outcome, "detail": res.detail})
        await system.engine.join()

        ex.settle()  # 日终：未成交挂单失效，今仓转昨仓
        await system.engine.join()

        rows = []
        for o, req, res, snap, note in sent:
            order = system.oms.get_order(req.client_order_id)
            trades = [t for t in system.oms.trades if t.client_order_id == req.client_order_id]
            filled = sum(t.volume for t in trades)
            value = sum(t.volume * t.price for t in trades)
            avg = value / filled if filled else np.nan
            sign = 1 if o.side == "BUY" else -1
            if res.outcome == "risk_rejected":
                status, reason = "风控拒绝", res.detail
            else:
                status, reason = order.status.value, order.reason or note
            rows.append({
                "symbol": o.symbol, "side": o.side, "ordered": o.volume, "sent": req.volume, "limit_price": req.price,
                "filled": int(filled), "status": status, "ref_price": o.ref_price, "arrival": snap.open,
                "avg_price": avg, "filled_value": value,
                "arrival_bp": sign * (avg / snap.open - 1) * 1e4 if filled else np.nan,
                "shortfall_bp": sign * (avg / o.ref_price - 1) * 1e4 if filled else np.nan,
                "fee": self.cost.total(value, is_sell=(o.side == "SELL")) if filled else 0.0,
                "reason": reason, "coid": req.client_order_id,
            })
        fills = pd.DataFrame(rows)

        # 对账：本地 OMS（由成交推导）vs 交易所持仓；资金：按成交推算 vs 交易所余额
        remote = positions_to_map(await system.gateway.query_positions())
        diffs = reconcile_positions(system.oms.positions.to_volume_map(), remote)
        cr = self.commission_rate
        expected = start_balance
        for t in system.oms.trades:
            notional = t.price * t.volume
            expected += -notional * (1 + cr) if t.direction is Direction.LONG else notional * (1 - cr)
        report = ExecutionReport(
            trade_date=pd.Timestamp(trade_date) if trade_date is not None else pd.NaT,
            fills=fills, skipped=skipped, extra=extra_results, position_diffs=diffs,
            cash_expected=expected, cash_actual=float((await system.gateway.query_account()).balance),
            risk_events=list(system.risk.events), risk_rejects=dict(system.risk.reject_counts),
            anomalies=list(system.oms.anomalies),
        )
        await system.stop()
        return report
