"""
从零实现事件驱动回测引擎
========================

学习目标：
    1. 理解事件驱动架构：Event / Queue / DataHandler / Strategy / Portfolio / ExecutionHandler
    2. 从零实现一个支持 A 股约束的引擎：T+1、涨跌停、停牌、100 股整数手、真实成本、除权除息
    3. 与 02 课的向量化回测在同一策略上"对拍"：先在理想条件下做到逐日一致，
       再逐项打开现实约束，量化每一项带来的差异
    4. 学会给回测引擎做"对账"和"拒单审计"

核心概念：
    - 事件循环：MarketEvent → SignalEvent → OrderEvent → FillEvent
    - 结构上无前视：策略只能拿到已经"推送"过来的 K 线
    - 复权与撮合分离：信号用 adj_close；撮合、费用、涨跌停、整数手用不复权价；
      除权除息日按复权因子把分红计入现金（或理想模式下再投资）
    - 对拍（Reconciliation）：两个独立实现在相同假设下结果必须一致，差异必须可解释

前置知识：
    - 02-vectorized-backtest.py、04-transaction-costs-slippage.py、05-a-share-constraints.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据；合成数据无停牌，停牌案例会人为删除一段行情）
"""

import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import (CostModel, add_reference_columns, board_rule, limit_prices, load_bars, performance_report,
                     round_lot_buy, round_price, trading_calendar, vectorized_backtest)
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SYMBOL, START, END = "600519", "20180101", "20241231"
CASE_SYMBOL, CASE_START, CASE_END = "000002", "20150101", "20171231"
CAPITAL = 1_000_000
EPS = 1e-6


# ==================== 第一部分：架构 ====================


def architecture():
    print("=" * 60)
    print("第一部分：事件驱动架构")
    print("=" * 60)

    print("""
    📺 比喻：向量化是"上帝视角看完整部电影"，事件驱动是"直播"——
       你只能看到当前这一帧及之前的内容，结构上就不可能偷看结局。

    ┌────────────┐ MarketEvent ┌─────────────────────────────────────┐
    │ DataHandler│────────────▶│            Event Queue (FIFO)        │
    │ 逐日推送   │             └──┬──────────┬───────────┬──────────┬─┘
    └────────────┘                │ Market   │ Signal    │ Order    │ Fill
                                  ▼          ▼           ▼          ▼
                          ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
                          │ Strategy │ │Portfolio │ │Execution │ │Portfolio │
                          │ 只见历史 │ │目标→订单 │ │撮合/成本 │ │记账/T+1  │
                          └──────────┘ └──────────┘ │A股约束   │ └──────────┘
                                                    └──────────┘
    每个交易日的时序：
      ① 开盘 MarketEvent(open)：Portfolio 刷新可卖数量（T+1）、处理除权除息
                               → ExecutionHandler 用开盘价撮合昨天收盘挂的单 → FillEvent
      ② 收盘 MarketEvent(close)：Portfolio 按收盘价估值 → Strategy 读历史出信号 → SignalEvent
                               → Portfolio 把目标仓位变成订单 → OrderEvent（挂到明天开盘）
      ③ 日终：记录现金、持仓、净值

    ┌──────────────────┬──────────────────────────┬─────────────────────────┐
    │ 组件             │ 职责                     │ 实盘对应（Phase 13）    │
    ├──────────────────┼──────────────────────────┼─────────────────────────┤
    │ DataHandler      │ 按时间顺序推送行情       │ 行情网关                │
    │ Strategy         │ 根据已知信息产生信号     │ 策略引擎                │
    │ Portfolio        │ 现金、持仓、目标→订单    │ 账户 / OMS              │
    │ ExecutionHandler │ 撮合、成本、约束检查     │ 交易网关 + 交易所       │
    │ Event Queue      │ 解耦各组件               │ 事件引擎（asyncio）     │
    └──────────────────┴──────────────────────────┴─────────────────────────┘
    """)


# ==================== 第二部分：引擎实现 ====================

# ---------- 事件 ----------


@dataclass
class MarketEvent:
    date: pd.Timestamp
    phase: str  # "open" 或 "close"
    bars: dict  # symbol -> dict；open 阶段只含开盘可知的字段


@dataclass
class SignalEvent:
    date: pd.Timestamp
    symbol: str
    target_weight: float


@dataclass
class OrderEvent:
    date: pd.Timestamp
    symbol: str
    side: str  # "buy" / "sell"
    target_weight: float  # 目标仓位单：成交数量在撮合时按成交价换算（见 ExecutionHandler 注释）


@dataclass
class FillEvent:
    date: pd.Timestamp
    symbol: str
    qty: float  # > 0 买入，< 0 卖出
    price: float  # 含滑点的成交价（不复权）
    commission: float
    stamp_duty: float
    transfer_fee: float
    slippage_cost: float

    @property
    def fees(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee


# ---------- 配置 ----------


@dataclass
class EngineConfig:
    """
    ideal()：小数股、零成本、无涨跌停/停牌约束、红利再投资 —— 与向量化回测的假设完全相同
    a_share()：100 股整数手、真实成本、涨跌停/停牌约束、现金分红
    """

    use_lots: bool = True
    cost: CostModel = field(default_factory=CostModel)
    enforce_limits: bool = True
    dividend: str = "cash"  # "cash" 现金分红 / "reinvest" 按复权因子增加持股
    band: float = 0.02  # 当前权重与目标相差不超过 band 时不调仓
    st_symbols: frozenset = frozenset()

    @classmethod
    def ideal(cls) -> "EngineConfig":
        return cls(use_lots=False, cost=CostModel.zero(), enforce_limits=False, dividend="reinvest")


# ---------- 数据 ----------


class DataHandler:
    """
    按交易日历逐日推送行情。frames: symbol -> 不复权日线（含 adj_factor/adj_close/pre_close），停牌日无记录。
    策略只能通过 history() 读取已经推送过的 K 线 —— 这就是"结构上无前视"。
    """

    OPEN_FIELDS = ("open", "pre_close", "div_ratio")

    def __init__(self, frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex):
        start = min(f.index[0] for f in frames.values())
        end = max(f.index[-1] for f in frames.values())
        self.calendar = calendar[(calendar >= start) & (calendar <= end)]
        self.symbols = list(frames)
        self._rows = {}
        for sym, f in frames.items():
            f = f.copy()
            f["div_ratio"] = (f["adj_factor"] / f["adj_factor"].shift(1)).fillna(1.0)
            self._rows[sym] = f[["open", "high", "low", "close", "volume", "pre_close", "adj_close",
                                 "div_ratio"]].to_dict("index")
        self._history = defaultdict(list)

    def stream(self):
        for d in self.calendar:
            bars = {s: rows[d] for s, rows in self._rows.items() if d in rows and rows[d]["volume"] > 0}
            open_bars = {s: {k: b[k] for k in self.OPEN_FIELDS} for s, b in bars.items()}
            yield MarketEvent(d, "open", open_bars)
            for s, b in bars.items():
                self._history[s].append(b)
            yield MarketEvent(d, "close", bars)

    def history(self, symbol: str, field_name: str, n: int) -> list[float]:
        return [b[field_name] for b in self._history[symbol][-n:]]


# ---------- 策略 ----------


class MovingAverageStrategy:
    """后复权收盘价站上 n 日均线 → 目标权重 weight，否则 0。停牌日沿用上一次的目标。"""

    def __init__(self, data: DataHandler, n: int = 20, weight: float = 1.0):
        self.data, self.n, self.weight = data, n, weight
        self.targets = {s: 0.0 for s in data.symbols}

    def on_market(self, ev: MarketEvent) -> list[SignalEvent]:
        out = []
        for s in self.data.symbols:
            if s in ev.bars:
                closes = self.data.history(s, "adj_close", self.n)
                if len(closes) == self.n:
                    self.targets[s] = self.weight / len(self.data.symbols) if closes[-1] > np.mean(closes) else 0.0
            out.append(SignalEvent(ev.date, s, self.targets[s]))
        return out


class ScheduledExitStrategy:
    """第一天起满仓持有，从 exit_date 起目标仓位变为 0（模拟一条风控清仓指令）"""

    def __init__(self, data: DataHandler, exit_date: pd.Timestamp):
        self.data, self.exit_date = data, exit_date

    def on_market(self, ev: MarketEvent) -> list[SignalEvent]:
        w = 1.0 if ev.date < self.exit_date else 0.0
        return [SignalEvent(ev.date, s, w / len(self.data.symbols)) for s in self.data.symbols]


# ---------- 账户 ----------


class Portfolio:
    def __init__(self, cash: float, config: EngineConfig):
        self.initial_cash = self.cash = float(cash)
        self.cfg = config
        self.positions: dict[str, float] = defaultdict(float)
        self.sellable: dict[str, float] = defaultdict(float)
        self.last_price: dict[str, float] = {}
        self.history, self.fills, self.dividends = [], [], []

    def equity(self, prices: dict | None = None) -> float:
        prices = prices or {}
        return self.cash + sum(q * prices.get(s, self.last_price.get(s, 0.0)) for s, q in self.positions.items())

    def on_open(self, ev: MarketEvent):
        """开盘：T+1 刷新可卖数量；处理除权除息"""
        self.sellable = defaultdict(float, self.positions)
        for s, b in ev.bars.items():
            r, q = b["div_ratio"], self.positions.get(s, 0.0)
            if abs(r - 1) < 1e-12:
                continue
            if q and self.cfg.dividend == "reinvest":
                self.positions[s] = self.sellable[s] = q * r  # 理想模式：等价于红利再投资（小数股）
            elif q:
                cash_div = q * self.last_price[s] * (1 - 1 / r)  # 价值等价：昨收 − 除权参考价
                self.cash += cash_div
                self.dividends.append((ev.date, s, cash_div))
            if s in self.last_price:
                self.last_price[s] /= r  # 估值基准换成除权参考价

    def on_close(self, ev: MarketEvent):
        for s, b in ev.bars.items():
            self.last_price[s] = b["close"]

    def on_signal(self, sig: SignalEvent) -> OrderEvent | None:
        """目标权重 → 订单。调仓幅度小于 band 或不足一手时不下单"""
        s, w = sig.symbol, sig.target_weight
        q, price = self.positions.get(s, 0.0), self.last_price.get(s)
        if price is None:
            return None
        eq = self.equity()
        cur_w = q * price / eq if eq > 0 else 0.0
        if w <= 0:
            return OrderEvent(sig.date, s, "sell", 0.0) if q > 0 else None
        lot_value = price * board_rule(s).min_lot if self.cfg.use_lots else 0.0
        diff = (w - cur_w) * eq
        if abs(w - cur_w) <= self.cfg.band or abs(diff) < lot_value:
            return None
        return OrderEvent(sig.date, s, "buy" if diff > 0 else "sell", w)

    def on_fill(self, f: FillEvent):
        self.positions[f.symbol] += f.qty
        if f.qty < 0:
            self.sellable[f.symbol] += f.qty
        self.cash -= f.qty * f.price + f.fees
        if abs(self.positions[f.symbol]) < 1e-9:
            del self.positions[f.symbol]
        self.fills.append(f)

    def record(self, date):
        mv = sum(q * self.last_price[s] for s, q in self.positions.items())
        self.history.append({"date": date, "cash": self.cash, "market_value": mv, "equity": self.cash + mv,
                             **{f"qty_{s}": q for s, q in self.positions.items()}})


# ---------- 模拟撮合 ----------


class ExecutionHandler:
    """
    用 t+1 日开盘价撮合 t 日收盘挂的单。
    约束：停牌不成交；开盘涨停不买、开盘跌停不卖（Decimal 计算的涨跌停价，前收盘为除权参考价）；
          T+1 只能卖可卖部分；买入取整到 100 股（科创板 200 股起）；先卖后买，卖出资金当日可用。
    成本：佣金（最低 5 元）+ 印花税（仅卖出）+ 过户费；滑点体现在成交价上。
    简化：订单是"目标仓位单"，股数按开盘成交价换算（相当于在开盘集合竞价按目标金额下单）；
          未成交的订单当日作废，由 Portfolio 在下一个收盘重新生成。
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.pending: deque[OrderEvent] = deque()
        self.rejections: list[tuple] = []

    def submit(self, order: OrderEvent):
        self.pending.append(order)

    def _reject(self, ev, o, reason):
        self.rejections.append((ev.date, o.symbol, o.side, reason))

    def _price(self, px: float, side: str, up: float, down: float) -> float:
        slip = self.cfg.cost.slippage
        p = px * (1 + slip) if side == "buy" else px * (1 - slip)
        if self.cfg.use_lots:
            p = round_price(p)
        if self.cfg.enforce_limits:
            p = min(p, up) if side == "buy" else max(p, down)
        return p

    def match(self, ev: MarketEvent, pf: Portfolio) -> list[FillEvent]:
        fills, cash = [], pf.cash
        opens = {s: b["open"] for s, b in ev.bars.items()}
        eq_open = pf.equity(opens)
        orders = sorted(self.pending, key=lambda o: o.side != "sell")  # 先卖后买
        self.pending.clear()
        cost = self.cfg.cost
        for o in orders:
            bar = ev.bars.get(o.symbol)
            if bar is None:
                self._reject(ev, o, "停牌")
                continue
            rule = board_rule(o.symbol, is_st=o.symbol in self.cfg.st_symbols, date=ev.date)
            up = down = np.nan
            if self.cfg.enforce_limits and pd.notna(bar["pre_close"]):
                up, down = limit_prices(bar["pre_close"], rule.limit_pct)
                if o.side == "buy" and bar["open"] >= up - EPS:
                    self._reject(ev, o, "开盘涨停")
                    continue
                if o.side == "sell" and bar["open"] <= down + EPS:
                    self._reject(ev, o, "开盘跌停")
                    continue
            px = self._price(bar["open"], o.side, up, down)
            held = pf.positions.get(o.symbol, 0.0)
            target_qty = eq_open * o.target_weight / px

            if o.side == "sell":
                qty = held if o.target_weight <= 0 else held - target_qty
                if self.cfg.use_lots and o.target_weight > 0:
                    qty = int(qty // 100 * 100)  # 部分卖出也按整手
                avail = pf.sellable.get(o.symbol, 0.0)
                if avail <= 0:
                    self._reject(ev, o, "T+1：今日无可卖数量")
                    continue
                qty = min(qty, avail)
                if qty <= 0:
                    continue
                f = cost.fees(qty * px, is_sell=True)
                cash += qty * px - f["total"]
                fills.append(FillEvent(ev.date, o.symbol, -qty, px, f["commission"], f["stamp_duty"],
                                       f["transfer_fee"], qty * abs(bar["open"] - px)))
            else:
                qty = target_qty - held
                variable_rate = cost.commission_rate + cost.transfer_fee_rate
                qty = min(qty, cash / (px * (1 + variable_rate)))  # 预留按比例的费用
                if self.cfg.use_lots:
                    qty = round_lot_buy(qty, rule)
                    while qty > 0 and qty * px + cost.total(qty * px, False) > cash + 1e-9:  # 最低佣金可能不够
                        qty = round_lot_buy(qty - rule.step, rule)
                if qty <= 0:
                    self._reject(ev, o, "资金不足一手" if self.cfg.use_lots else "资金不足")
                    continue
                f = cost.fees(qty * px, is_sell=False)
                cash -= qty * px + f["total"]
                fills.append(FillEvent(ev.date, o.symbol, qty, px, f["commission"], 0.0,
                                       f["transfer_fee"], qty * abs(px - bar["open"])))
        return fills


# ---------- 引擎主循环 ----------


class Backtester:
    def __init__(self, frames: dict, calendar: pd.DatetimeIndex, config: EngineConfig,
                 capital: float = CAPITAL, strategy_cls=MovingAverageStrategy, **strategy_kwargs):
        self.data = DataHandler(frames, calendar)
        self.strategy = strategy_cls(self.data, **strategy_kwargs)
        self.portfolio = Portfolio(capital, config)
        self.execution = ExecutionHandler(config)
        self.events: deque = deque()
        self.event_count = Counter()

    def _dispatch(self):
        while self.events:
            ev = self.events.popleft()
            self.event_count[type(ev).__name__] += 1
            if isinstance(ev, MarketEvent) and ev.phase == "open":
                self.portfolio.on_open(ev)
                self.events.extend(self.execution.match(ev, self.portfolio))
            elif isinstance(ev, MarketEvent):
                self.portfolio.on_close(ev)
                self.events.extend(self.strategy.on_market(ev))
            elif isinstance(ev, SignalEvent):
                order = self.portfolio.on_signal(ev)
                if order is not None:
                    self.events.append(order)
            elif isinstance(ev, OrderEvent):
                self.execution.submit(ev)
            elif isinstance(ev, FillEvent):
                self.portfolio.on_fill(ev)

    def run(self) -> pd.DataFrame:
        for ev in self.data.stream():
            self.events.append(ev)
            self._dispatch()
            if ev.phase == "close":
                self.portfolio.record(ev.date)
        return pd.DataFrame(self.portfolio.history).set_index("date")


def engine_demo(frames, calendar):
    print("\n" + "=" * 60)
    print(f"第二部分：运行引擎（{SYMBOL} MA20，初始资金 {CAPITAL:,}，A 股模式）")
    print("=" * 60)

    t0 = time.perf_counter()
    bt = Backtester(frames, calendar, EngineConfig())
    hist = bt.run()
    elapsed = time.perf_counter() - t0
    pf = bt.portfolio
    print(f"   处理事件：{dict(bt.event_count)}，耗时 {elapsed:.2f} 秒")

    fills = pd.DataFrame([vars(f) | {"fees": f.fees} for f in pf.fills])
    print("\n📋 前 6 笔成交（价格为不复权价，含滑点）：")
    show = fills.head(6).copy()
    show["date"] = show["date"].dt.date
    print(show[["date", "symbol", "qty", "price", "commission", "stamp_duty", "transfer_fee", "slippage_cost"]]
          .round(2).to_string(index=False))

    print("\n📋 收盘记账快照（equity = cash + market_value；首笔成交前后 + 最后一天）：")
    i = hist.index.get_loc(pf.fills[0].date)
    print(hist.iloc[[i - 1, i, i + 1, i + 2, len(hist) - 1]].round(2).to_string())
    return bt, hist, fills


# ==================== 第三部分：对账与审计 ====================


def reconcile(bt: Backtester, hist: pd.DataFrame, fills: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：对账与审计")
    print("=" * 60)

    pf = bt.portfolio
    buys, sells = fills[fills["qty"] > 0], fills[fills["qty"] < 0]
    div = sum(d[2] for d in pf.dividends)
    cash_calc = (pf.initial_cash - (buys["qty"] * buys["price"]).sum() + (-sells["qty"] * sells["price"]).sum()
                 - fills["fees"].sum() + div)
    pos_calc = fills.groupby("symbol")["qty"].sum()
    eq_calc = hist["cash"] + hist["market_value"]
    print(f"""
    ① 现金流对账：初始资金 − 买入金额 + 卖出金额 − 费用 + 现金分红
       = {pf.initial_cash:,.2f} − {(buys['qty'] * buys['price']).sum():,.2f} + {(-sells['qty'] * sells['price']).sum():,.2f}
         − {fills['fees'].sum():,.2f} + {div:,.2f}
       = {cash_calc:,.2f}    引擎期末现金 {pf.cash:,.2f}    差异 {abs(cash_calc - pf.cash):.2e}
    ② 持仓对账：成交累计 {dict(pos_calc.round(0))}，引擎期末持仓 {dict(pf.positions)}
    ③ 净值恒等式 equity = cash + Σ 持股 × 收盘价：最大差异 {(eq_calc - hist['equity']).abs().max():.2e}
    ④ 现金从未为负：最低现金 {hist['cash'].min():,.2f}
    ⑤ 整数手：所有买入股数都是 100 的整数倍 → {bool((buys['qty'] % 100 == 0).all())}
    """)

    print(f"    现金分红 {len(pf.dividends)} 次，合计 {div:,.2f} 元；拒单统计：{dict(Counter(r[3] for r in bt.execution.rejections)) or '无'}")
    print(f"""
    💰 成本拆解：佣金 {fills['commission'].sum():,.0f}，印花税 {fills['stamp_duty'].sum():,.0f}，
       过户费 {fills['transfer_fee'].sum():,.0f}，滑点 {fills['slippage_cost'].sum():,.0f}
       合计 {fills['fees'].sum() + fills['slippage_cost'].sum():,.0f} 元，占初始资金 {(fills['fees'].sum() + fills['slippage_cost'].sum()) / pf.initial_cash:.2%}
       成交 {len(fills)} 笔，年化换手 {(fills['qty'].abs() * fills['price']).sum() / hist['equity'].mean() / (len(hist) / 252):.1f} 倍
    """)
    cash_w = hist["cash"] / hist["equity"]
    held_days = hist["market_value"] > 0
    print(f"    持仓日的平均现金占比 {cash_w[held_days].mean():.1%}（100 股整数手 + 高股价造成的闲置资金）")


# ==================== 第四部分：与向量化回测对拍 ====================


def compare_with_vectorized(frames, calendar) -> dict:
    print("\n" + "=" * 60)
    print("第四部分：与 02 课向量化回测对拍（同一策略、同一数据）")
    print("=" * 60)

    df = frames[SYMBOL]
    close, open_ = df["adj_close"], df["open"] * df["adj_factor"]
    signal = (close > close.rolling(20).mean()).astype(float)

    t0 = time.perf_counter()
    v0 = vectorized_backtest(close, signal, 0.0, open_=open_, execution="next_open")
    t_vec = time.perf_counter() - t0
    v1 = vectorized_backtest(close, signal, 0.001, open_=open_, execution="next_open")

    base = EngineConfig.ideal()
    configs = [
        ("E0 理想模式", base),
        ("E1 + 100股整数手 + 现金分红", EngineConfig(use_lots=True, cost=CostModel.zero(), enforce_limits=False)),
        ("E2 + 真实费用 + 5bp 滑点", EngineConfig(use_lots=True, cost=CostModel(), enforce_limits=False)),
        ("E3 + 涨跌停/停牌约束（完整）", EngineConfig()),
    ]
    navs, t_eng = {}, 0.0
    for name, cfg in configs:
        t0 = time.perf_counter()
        bt = Backtester(frames, calendar, cfg)
        hist = bt.run()
        t_eng = time.perf_counter() - t0
        navs[name] = hist["equity"] / CAPITAL
        navs[name + "|rej"] = len(bt.execution.rejections)

    e0 = navs["E0 理想模式"].reindex(v0.index)
    print(f"""
    ✅ 理想模式对拍：E0（引擎） vs V0（向量化 next_open，零成本）
       逐日净值最大差异 {(e0 - v0['nav']).abs().max():.2e}，期末 {e0.iloc[-1]:.6f} vs {v0['nav'].iloc[-1]:.6f}
       → 两个完全独立的实现，在相同假设下结果一致：这是引擎可信的第一步
    """)

    rows = [["V0 向量化，零成本", v0["ret"], "—"]]
    for name, _ in configs:
        nav = navs[name]
        rows.append([name, nav.pct_change().fillna(nav.iloc[0] - 1), navs[name + "|rej"]])
    rows.append(["V1 向量化，单边 10bp", v1["ret"], "—"])
    table = []
    for name, r, rej in rows:
        rep = performance_report(r)
        table.append([name, f"{(1 + r).prod() - 1:+.2%}", f"{rep['年化收益']:+.2%}", f"{rep['Sharpe']:.2f}",
                      f"{rep['最大回撤']:.2%}", rej])
    print(pd.DataFrame(table, columns=["版本", "总收益", "年化", "Sharpe", "最大回撤", "拒单数"]).to_string(index=False))

    print("""
    🔍 差异解释（从上到下逐项打开约束）：
       E0 → E1：整数手留下闲置现金（降低了仓位），现金分红在下次调仓前不再投资
       E1 → E2：佣金（最低 5 元）、卖出印花税、过户费、滑点
       E2 → E3：开盘涨/跌停或停牌导致的拒单 —— 本例蓝筹股触板很少，差异通常很小；
                 小盘股、题材股的差异会大得多（见第五部分停牌案例）
       V1：向量化用一个固定比例近似全部成本，与 E3 的差距说明"粗略成本"与"精确撮合"的差别
    """)
    print(f"    ⏱️ 速度：向量化 {t_vec * 1000:.1f} ms，事件驱动 {t_eng * 1000:.0f} ms，约慢 {t_eng / t_vec:.0f} 倍")
    return {k: v for k, v in navs.items() if not k.endswith("|rej")} | {"V0 向量化": v0["nav"]}


# ==================== 第五部分：停牌与拒单案例 ====================


def suspension_case(calendar_case):
    print("\n" + "=" * 60)
    print(f"第五部分：停牌与拒单案例（{CASE_SYMBOL}，{CASE_START[:4]}-{CASE_END[:4]}）")
    print("=" * 60)

    df = load_bars(CASE_SYMBOL, CASE_START, CASE_END)
    if is_offline():
        print("⚠️ 离线模式：人为删除 120 个交易日的合成行情来模拟停牌")
        df = add_reference_columns(df.drop(df.index[230:350]))
    cal = calendar_case[(calendar_case >= df.index[0]) & (calendar_case <= df.index[-1])]
    halted = ~cal.isin(df.index)
    grp = (halted != np.roll(halted, 1)).cumsum()
    spans = pd.Series(cal[halted]).groupby(grp[halted]).agg(["first", "last", "size"]).sort_values("size")
    h0, h1, n_halt = spans.iloc[-1]
    exit_date = cal[cal.get_loc(h0) + 20]

    print(f"""
    场景：从第一天起满仓持有；最长停牌为 {h0.date()} ~ {h1.date()}（{n_halt} 个交易日）。
         假设风控在停牌期间的 {exit_date.date()} 下达"清仓"指令 —— 引擎会怎么处理？
    """)
    bt = Backtester({CASE_SYMBOL: df}, cal, EngineConfig(), strategy_cls=ScheduledExitStrategy, exit_date=exit_date)
    hist = bt.run()
    rej = pd.DataFrame(bt.execution.rejections, columns=["date", "symbol", "side", "reason"])
    print(f"   拒单统计：{dict(Counter(rej['reason'])) if len(rej) else '无'}")
    if len(rej):
        show = pd.concat([g.iloc[[0, -1]] if len(g) > 1 else g for _, g in rej.groupby("reason", sort=False)])
        show["date"] = show["date"].dt.date
        print("   每种拒单原因的第一次和最后一次：")
        print(show.to_string(index=False))

    sells = [f for f in bt.portfolio.fills if f.qty < 0]
    last_close = df.loc[:h0, "close"].iloc[-1]
    if sells:
        f = sells[0]
        print(f"""
    实际卖出：{f.date.date()}，{-f.qty:.0f} 股 @ {f.price:.2f}（不复权）
    停牌前最后收盘价 {last_close:.2f} → 卖出价相对停牌前 {f.price / last_close - 1:+.2%}
    💡 向量化回测若在 {exit_date.date()} 按 ffill 价格"卖出"，会得到 {last_close:.2f} 的虚假成交价；
       真实情况是：停牌期间无法交易，复牌后若一字跌停仍然卖不出，只能等到打开跌停的那天。
    """)
    print(f"   期末净值 {hist['equity'].iloc[-1] / CAPITAL:.4f}")


# ==================== 第六部分：引擎单元测试 ====================


def unit_tests():
    print("\n" + "=" * 60)
    print("第六部分：引擎单元测试（构造极端行情，逐条验证约束）")
    print("=" * 60)

    cfg = EngineConfig()
    d = pd.Timestamp("2024-03-01")

    def open_event(bars):
        return MarketEvent(d, "open", {s: {"open": o, "pre_close": pc, "div_ratio": 1.0} for s, (o, pc) in bars.items()})

    results = []

    # ① 开盘涨停：前收 10.00，开盘 11.00 = 涨停价 → 买不进
    pf, ex = Portfolio(100_000, cfg), ExecutionHandler(cfg)
    ex.submit(OrderEvent(d, "600000", "buy", 1.0))
    fills = ex.match(open_event({"600000": (11.00, 10.00)}), pf)
    results.append(["开盘涨停买入", not fills and ex.rejections[-1][3] == "开盘涨停"])

    # ② 停牌：当日没有行情 → 订单作废
    ex.submit(OrderEvent(d, "600000", "buy", 1.0))
    fills = ex.match(open_event({}), pf)
    results.append(["停牌日买入", not fills and ex.rejections[-1][3] == "停牌"])

    # ③ T+1：今天开盘买入后，同一天再卖 → 可卖数量为 0
    pf, ex = Portfolio(100_000, cfg), ExecutionHandler(cfg)
    pf.on_open(open_event({"600000": (10.0, 10.0)}))
    ex.submit(OrderEvent(d, "600000", "buy", 1.0))
    for f in ex.match(open_event({"600000": (10.0, 10.0)}), pf):
        pf.on_fill(f)
    bought = pf.positions["600000"]
    ex.submit(OrderEvent(d, "600000", "sell", 0.0))
    fills = ex.match(open_event({"600000": (10.0, 10.0)}), pf)  # 同一天（未调用 on_open 刷新可卖）
    results.append([f"T+1 当日卖出（已买 {bought:.0f} 股）", not fills and ex.rejections[-1][3].startswith("T+1")])

    # ④ 整数手 + 最低佣金：1 万元买 10 元股票 → 最多 900 股（1000 股需 10000 元 + 5 元佣金）
    pf, ex = Portfolio(10_000, EngineConfig(cost=CostModel(slippage=0.0))), None
    ex = ExecutionHandler(pf.cfg)
    ex.submit(OrderEvent(d, "600000", "buy", 1.0))
    fills = ex.match(open_event({"600000": (10.0, 10.0)}), pf)
    results.append([f"整数手与最低佣金（成交 {fills[0].qty:.0f} 股，佣金 {fills[0].commission:.2f}）",
                    fills[0].qty == 900 and fills[0].commission == 5.0])

    # ⑤ 主板 ST 股：2026-07-06 前 ±5%，开盘 +5% 即涨停
    st_cfg = EngineConfig(st_symbols=frozenset({"600001"}))
    for day, expect_reject in [("2026-07-03", True), ("2026-07-06", False)]:
        ex = ExecutionHandler(st_cfg)
        ex.submit(OrderEvent(pd.Timestamp(day), "600001", "buy", 1.0))
        ev = MarketEvent(pd.Timestamp(day), "open", {"600001": {"open": 10.50, "pre_close": 10.00, "div_ratio": 1.0}})
        fills = ex.match(ev, Portfolio(100_000, st_cfg))
        results.append([f"ST 股 {day} 开盘 +5% 买入（{'应拒绝' if expect_reject else '应成交'}）", (not fills) == expect_reject])

    # ⑥ 现金分红：持有 1000 股，昨收 10 元，复权因子比 1.05 → 现金增加 1000×10×(1−1/1.05)
    pf = Portfolio(0, cfg)
    pf.positions["600000"], pf.last_price["600000"] = 1000, 10.0
    pf.on_open(MarketEvent(d, "open", {"600000": {"open": 9.6, "pre_close": 9.52, "div_ratio": 1.05}}))
    expect = 1000 * 10 * (1 - 1 / 1.05)
    results.append([f"除息日现金分红（{pf.cash:.2f} 元）", abs(pf.cash - expect) < 1e-9 and abs(pf.equity() - 10_000) < 1e-9])

    for name, ok in results:
        print(f"   {'✅' if ok else '❌'} {name}")
    print("   💡 把这些测试写进 pytest：引擎的每一条规则都应该有一个能失败的测试")


# ==================== 主函数 ====================


def main():
    architecture()

    calendar = trading_calendar(START, END)
    frames = {SYMBOL: load_bars(SYMBOL, START, END)}
    bt, hist, fills = engine_demo(frames, calendar)
    reconcile(bt, hist, fills)
    navs = compare_with_vectorized(frames, calendar)
    suspension_case(trading_calendar(CASE_START, CASE_END))
    unit_tests()

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ["V0 向量化", "E1 + 100股整数手 + 现金分红", "E2 + 真实费用 + 5bp 滑点", "E3 + 涨跌停/停牌约束（完整）"]:
        ax.plot(navs[name], label=name, ls="--" if name.startswith("V0") else "-")
    ax.set_title(f"{SYMBOL} MA20：向量化 vs 事件驱动（逐项加入 A 股约束）")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"\n📁 图片已保存：{save_figure(fig, '06_event_vs_vectorized.png', subdir='phase-4')}")

    print("""
    📌 这个引擎刻意省略了：未成交订单的部分成交/延续、成交量约束、限价单/止损单、
       红利税与派息到账延迟、多策略资金分配 —— 这些是本阶段实战项目（迷你回测引擎）的任务。
    """)

    print("=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把订单改为"按前一日收盘价计算股数"的固定股数单，并在 Portfolio 中预留 2% 现金缓冲；
       与"目标仓位单"相比，净值和拒单统计有何变化？
    2. 给 ExecutionHandler 加成交量约束：单笔成交不超过当日成交量的 5%（未成交部分次日继续），
       把初始资金调到 10 亿元观察效果
    3. 用三只股票（600519、601088、000002）等权运行引擎，检查先卖后买、资金不足一手的拒单是否合理
    4. 增加一个"收盘止损"：开盘阶段策略看到开盘价跌破成本 5% 时下一张收盘卖单，
       验证 T+1 规则会拒绝"当天买入当天止损"
    """)


if __name__ == "__main__":
    main()
