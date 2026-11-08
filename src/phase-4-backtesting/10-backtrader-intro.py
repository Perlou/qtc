"""
backtrader 入门
===============

学习目标：
    1. 理解 backtrader 的核心概念：Cerebro、Data Feed、Strategy、Broker、Sizer、Analyzer
    2. 实现 A 股佣金类（双向佣金最低 5 元 + 卖出印花税 + 过户费）和 100 股整数手 Sizer
    3. 认识默认写法的陷阱：用收盘价算股数、次日开盘成交 → 资金不足被拒单
    4. 用 cheat_on_open 在"次日开盘"检查涨跌停，并处理除权除息（现金分红）
    5. 与 06 课自研引擎"对拍"：相同假设下逐日净值一致，差异可解释
    6. 把 Analyzer 的结果与自己写的绩效指标对拍

核心概念：
    - next()：t 日收盘时调用，此时下的市价单默认在 t+1 日开盘成交（天然避免同 bar 成交）
    - next_open()：开启 cheat_on_open 后，在 t+1 日开盘、撮合之前调用，可以看到 t+1 日开盘价
    - CommInfoBase._getcommission：自定义费用模型的入口

前置知识：
    - 06-event-driven-engine.py（事件驱动架构与 A 股撮合规则）

环境要求：
    - pip install backtrader pandas akshare（本课在 backtrader 1.9.78 + pandas 3 上验证通过）
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）

⚠️ backtrader 原作者已长期不维护，新版本 Python / pandas 下可能需要自行处理兼容问题。
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import CostModel, board_rule, limit_prices, load_bars, load_lesson, performance_report, round_lot_buy, trading_calendar
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning)  # backtrader 源码中的旧式转义字符串

SYMBOL, START, END = "600519", "20180101", "20241231"
CAPITAL = 1_000_000
EPS = 1e-6


# ==================== 第一部分：backtrader 结构 ====================


def architecture():
    print("=" * 60)
    print("第一部分：backtrader 结构")
    print("=" * 60)

    print(f"""
    backtrader 版本：{bt.__version__}，pandas 版本：{pd.__version__}

    ┌──────────────────────────────────────────────────┐
    │                  Cerebro（大脑）                  │
    │  ┌───────────┐ ┌───────────┐ ┌────────────────┐  │
    │  │ Data Feed │ │ Strategy  │ │ Broker         │  │
    │  │PandasData │ │ next()    │ │ 现金/佣金/撮合 │  │
    │  └───────────┘ └───────────┘ └────────────────┘  │
    │  ┌───────────┐ ┌────────────────────────────┐    │
    │  │ Sizer     │ │ Analyzers（Sharpe/回撤…）  │    │
    │  └───────────┘ └────────────────────────────┘    │
    └──────────────────────────────────────────────────┘

    与 06 课自研引擎的对应：
    ┌──────────────────┬────────────────────────────────────────────┐
    │ 自研引擎         │ backtrader                                 │
    ├──────────────────┼────────────────────────────────────────────┤
    │ DataHandler      │ bt.feeds.PandasData（可扩展自定义 lines）  │
    │ Strategy         │ bt.Strategy.next() / next_open()           │
    │ Portfolio        │ Broker 的现金与持仓 + Sizer                │
    │ ExecutionHandler │ BackBroker 撮合 + CommInfo + 滑点          │
    │ A 股约束         │ 无内置，需要自己写（本课的主要工作）       │
    └──────────────────┴────────────────────────────────────────────┘

    ⚠️ 索引约定：self.data.close[0] 是当前 bar，[-1] 是前一根 —— 与 pandas 的负索引含义不同！
    """)


# ==================== 第二部分：A 股佣金类 ====================


class AShareCommission(bt.CommInfoBase):
    """A 股股票费用：佣金双向（最低 5 元）+ 印花税仅卖出 + 过户费双向"""

    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),  # commission 以小数表示（0.00025 而不是 0.025%）
        ("commission", 0.00025),
        ("min_comm", 5.0),
        ("stamp_duty", 0.0005),
        ("transfer_fee", 0.00001),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        fee = max(value * self.p.commission, self.p.min_comm) + value * self.p.transfer_fee
        if size < 0:  # 卖出加收印花税
            fee += value * self.p.stamp_duty
        return fee


def commission_check():
    print("\n" + "=" * 60)
    print("第二部分：A 股佣金类（CommInfoBase）")
    print("=" * 60)

    ci, model = AShareCommission(), CostModel()
    rows = []
    for size, price in [(100, 15.0), (1000, 15.0), (-1000, 15.0), (100, 1500.0), (-500, 1500.0)]:
        b = ci.getcommission(size, price)
        m = model.total(abs(size) * price, is_sell=size < 0)
        rows.append([size, price, f"{abs(size) * price:,.0f}", round(b, 2), round(m, 2), "✅" if abs(b - m) < 1e-9 else "❌"])
    print(pd.DataFrame(rows, columns=["size(负=卖)", "价格", "成交金额", "backtrader", "bt_core", "一致"]).to_string(index=False))
    print("   💡 第一行：1500 元成交触发最低 5 元佣金；卖出行多了印花税")


# ==================== 第三部分：默认写法与整数手 Sizer ====================


class AShareFeed(bt.feeds.PandasData):
    """不复权 OHLCV + 额外的 adj_close（信号）、pre_close（涨跌停）、adj_factor（分红）"""

    lines = ("adj_close", "pre_close", "adj_factor")
    params = (("adj_close", -1), ("pre_close", -1), ("adj_factor", -1), ("openinterest", None))


class NaiveLotSizer(bt.Sizer):
    """按 t 日收盘价和可用资金比例计算股数，取整到 100 股（CONCEPT 中的写法）"""

    params = (("pct", 0.95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return int(cash * self.p.pct / data.close[0] // 100 * 100)
        return self.broker.getposition(data).size


class SmaCross(bt.Strategy):
    """默认写法：next() 中下市价单 → 下一根 bar 开盘成交"""

    params = (("n", 20),)

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.adj_close, period=self.p.n)
        self.status = Counter()

    def notify_order(self, order):
        if order.status in (order.Completed, order.Margin, order.Rejected, order.Canceled):
            self.status[order.getstatusname()] += 1

    def next(self):
        if not self.position and self.data.adj_close[0] > self.sma[0]:
            self.buy()
        elif self.position and self.data.adj_close[0] < self.sma[0]:
            self.close()


def run_cerebro(df: pd.DataFrame, strategy, sizer=None, cheat_on_open=False, slippage=0.0, **kw):
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=cheat_on_open)
    cerebro.adddata(AShareFeed(dataname=df))
    cerebro.addstrategy(strategy, **kw)
    cerebro.broker.setcash(CAPITAL)
    cerebro.broker.addcommissioninfo(AShareCommission())
    if slippage:
        cerebro.broker.set_slippage_perc(slippage, slip_open=True, slip_limit=True, slip_match=True, slip_out=False)
    if sizer is not None:
        sizer_cls, sizer_kwargs = sizer
        cerebro.addsizer(sizer_cls, **sizer_kwargs)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="tr", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                        annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    strat = cerebro.run()[0]
    tr = pd.Series(strat.analyzers.tr.get_analysis())
    tr.index = pd.to_datetime(tr.index)
    return strat, cerebro.broker.getvalue(), tr


def naive_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：默认写法的陷阱——用收盘价算股数，次日开盘成交")
    print("=" * 60)

    rows = []
    for pct in [1.00, 0.95]:
        strat, value, _ = run_cerebro(df, SmaCross, sizer=(NaiveLotSizer, {"pct": pct}))
        rows.append([f"{pct:.0%}", strat.status.get("Completed", 0), strat.status.get("Margin", 0), f"{value:,.0f}"])
    print(pd.DataFrame(rows, columns=["资金比例", "成交订单", "Margin 拒单", "期末资产"]).to_string(index=False))
    print("""
    🔍 为什么 100% 资金会被拒单：股数按 t 日收盘价算，但在 t+1 日开盘成交 ——
       只要开盘价高于收盘价（或加上佣金），资金就不够，broker 返回 Margin，这笔买入直接作废。
       常见的"修补"是只用 95% 资金，但这会让仓位系统性偏低。
    ⚠️ 这个默认写法还有两个问题：开盘涨停照样买入（没有涨跌停检查）；除息日现金分红丢失
       （用不复权价撮合时，净值会在除息日"假跳水"）。下一部分逐一解决。
    """)


# ==================== 第四部分：A 股完整版策略 ====================


class AShareLotSizer(bt.Sizer):
    """
    目标仓位 → 股数（与 06 课 ExecutionHandler 相同的规则）：
      按开盘成交价（含滑点）换算，预留按比例的费用，取整到 100 股（科创板 200 股起），
      若最低佣金导致资金不足，再减少一手。需要配合 cheat_on_open，在 next_open() 中下单。
    """

    params = (("target_weight", 1.0), ("slippage", 0.0), ("symbol", SYMBOL), ("cost", CostModel()))

    def _getsizing(self, comminfo, cash, data, isbuy):
        held = self.broker.getposition(data).size
        if not isbuy:
            return held
        px = data.open[0] * (1 + self.p.slippage)
        if self.p.slippage:
            px = round(px, 2)
        eq_open = cash + held * data.open[0]
        rule = board_rule(self.p.symbol)
        c = self.p.cost
        qty = min(eq_open * self.p.target_weight / px - held, cash / (px * (1 + c.commission_rate + c.transfer_fee_rate)))
        qty = round_lot_buy(qty, rule)
        while qty > 0 and qty * px + c.total(qty * px, False) > cash + 1e-9:
            qty = round_lot_buy(qty - rule.step, rule)
        return qty


class AShareSma(bt.Strategy):
    """
    t 日收盘（next）：用后复权价算信号，决定"明天要不要调仓"
    t+1 日开盘（next_open，需 cheat_on_open=True）：
      ① 除权除息：按复权因子把现金分红加到账户
      ② 检查开盘涨停（不买）/ 开盘跌停（不卖），用 Decimal 计算的涨跌停价
      ③ 下市价单，以 t+1 日开盘价成交
    停牌日没有 bar，自然不会交易。T+1：买入在开盘，最早下一个交易日才会卖出，天然满足。
    """

    params = (("n", 20), ("band", 0.02), ("symbol", SYMBOL))

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.adj_close, period=self.p.n)
        self.intent = None  # "buy" / "sell" / None
        self.rejections = Counter()
        self.dividends = 0.0
        self.fills = []

    def notify_order(self, order):
        if order.status == order.Completed:
            self.fills.append((pd.Timestamp(bt.num2date(order.executed.dt)).normalize(), order.executed.size, order.executed.price))

    def next(self):
        d, pos = self.data, self.position.size
        target = 1.0 if d.adj_close[0] > self.sma[0] else 0.0
        eq = self.broker.getvalue()
        cur_w = pos * d.close[0] / eq
        lot_value = d.close[0] * board_rule(self.p.symbol).min_lot
        self.intent = None
        if target == 0 and pos > 0:
            self.intent = "sell"
        elif target > 0 and abs(target - cur_w) > self.p.band and (target - cur_w) * eq >= lot_value:
            self.intent = "buy"

    def next_open(self):
        d = self.data
        if len(d) > 1 and self.position.size:  # ① 现金分红
            r = d.adj_factor[0] / d.adj_factor[-1]
            if abs(r - 1) > 1e-12:
                cash_div = self.position.size * d.close[-1] * (1 - 1 / r)
                self.broker.add_cash(cash_div)
                self.broker.getvalue()  # 让 add_cash 立即生效（backtrader 在估值时才入账）
                self.dividends += cash_div
        if self.intent is None or np.isnan(d.pre_close[0]):
            return
        up, down = limit_prices(d.pre_close[0], board_rule(self.p.symbol).limit_pct)  # ②
        if self.intent == "buy":
            if d.open[0] >= up - EPS:
                self.rejections["开盘涨停"] += 1
                return
            self.buy()  # ③ 股数由 AShareLotSizer 决定
        elif self.intent == "sell":
            if d.open[0] <= down + EPS:
                self.rejections["开盘跌停"] += 1
                return
            self.close()


def run_ashare(df: pd.DataFrame, slippage: float = 0.0):
    sizer = (AShareLotSizer, {"slippage": slippage, "cost": CostModel(slippage=slippage)})
    return run_cerebro(df, AShareSma, sizer=sizer, cheat_on_open=True, slippage=slippage)


# ==================== 第五部分：与 06 课自研引擎对拍 ====================


def compare_with_engine(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("第四、五部分：A 股完整版策略，并与 06 课自研引擎对拍")
    print("=" * 60)

    engine = load_lesson("06-event-driven-engine.py")
    calendar = trading_calendar(START, END)
    out, rows = {}, []
    for slip in [0.0, 0.0005]:
        strat, value, tr = run_ashare(df, slippage=slip)
        bt_nav = (1 + tr).cumprod() * CAPITAL
        cfg = engine.EngineConfig(cost=CostModel(slippage=slip))
        eng_bt = engine.Backtester({SYMBOL: df}, calendar, cfg)
        hist = eng_bt.run()
        engine_fills = eng_bt.portfolio.fills
        eng_nav = hist["equity"].reindex(bt_nav.index)
        diff = (bt_nav - eng_nav).abs()
        ta = strat.analyzers.ta.get_analysis()
        rows.append([f"{slip * 1e4:.0f} bp", f"{value:,.2f}", f"{hist['equity'].iloc[-1]:,.2f}",
                     f"{diff.max():,.2f}", f"{(diff / eng_nav).max():.2e}", ta.total.total,
                     dict(strat.rejections) or "无", f"{strat.dividends:,.0f}"])
        out[slip] = (bt_nav, eng_nav, tr)
        # 逐笔比对成交
        bf = pd.DataFrame(strat.fills, columns=["date", "qty", "price"])
        ef = pd.DataFrame([(f.date, f.qty, f.price) for f in engine_fills], columns=["date", "qty", "price"])
        m = bf.merge(ef, on="date", how="outer", suffixes=("_bt", "_eng"))
        out[f"fills_{slip}"] = m
    print(pd.DataFrame(rows, columns=["滑点", "backtrader 期末", "自研引擎期末", "逐日最大差(元)", "最大相对差",
                                      "交易数", "涨跌停拒单", "现金分红"]).to_string(index=False))

    print("\n🔍 逐笔比对成交（按成交日期合并两边的成交记录）：")
    for slip in [0.0, 0.0005]:
        m = out[f"fills_{slip}"]
        px_diff = (m["price_bt"] - m["price_eng"]).abs()
        qty_diff = m[(m["qty_bt"] - m["qty_eng"]).abs() > 0]
        line = (f"   滑点 {slip * 1e4:.0f} bp：成交 {len(m)} 笔，成交价相差超过 0.5 分的 {int((px_diff > 0.005).sum())} 笔"
                f"（最大 {px_diff.max():.2f} 元），股数不同的 {len(qty_diff)} 笔")
        if len(qty_diff):
            r = qty_diff.iloc[0]
            before = m[m["date"] < r["date"]]
            cash_gap = (before["qty_eng"] * before["price_eng"] - before["qty_bt"] * before["price_bt"]).sum()
            line += (f"；第一次出现在 {r['date'].date()}：backtrader {r['qty_bt']:+.0f} 股 vs 自研引擎 {r['qty_eng']:+.0f} 股"
                     f"（此前两边成交金额累计相差 {cash_gap:,.0f} 元，未计费用差）")
        print(line)
    print("""
    🔍 解读：
    - 滑点为 0 时，两个独立实现的逐笔成交、逐日净值都一致（差异仅为浮点误差）—— 对拍通过
    - 加入滑点后：自研引擎把含滑点的成交价四舍五入到分、只用涨跌停价封顶；
      backtrader 的 set_slippage_perc 不取整，滑点后超出当日最高/最低价时按最高/最低价成交（slip_match）
      → 少数几笔成交价不同 → 两边现金略有差别 → 某一次买入恰好跨过"一手"的门槛，多买或少买 100 股
      → 此后持股数不同，净值差异随行情放大。这就是路径依赖：微小的规则差异也会被整数手放大
    - 差异必须能像这样逐条解释，否则就说明某一边有 bug
    """)
    return out


# ==================== 第六部分：Analyzer 与自研指标对拍 ====================


def analyzer_check(df: pd.DataFrame, tr: pd.Series):
    print("\n" + "=" * 60)
    print("第六部分：Analyzer 与自研指标对拍")
    print("=" * 60)

    strat, _, _ = run_ashare(df)
    sr_bt = strat.analyzers.sharpe.get_analysis()["sharperatio"]
    dd_bt = strat.analyzers.dd.get_analysis().max.drawdown / 100
    ta = strat.analyzers.ta.get_analysis()
    rep = performance_report(tr)
    sr_pop = tr.mean() / tr.std(ddof=0) * np.sqrt(252)
    won = ta.won.total if "won" in ta else 0
    closed = ta.total.closed if "closed" in ta.total else 0
    print(f"""
    ┌──────────────────────┬──────────────┬──────────────┐
    │ 指标                 │ backtrader   │ bt_core      │
    ├──────────────────────┼──────────────┼──────────────┤
    │ Sharpe（rf = 0）     │ {sr_bt:>12.4f} │ {rep['Sharpe']:>12.4f} │
    │ 最大回撤             │ {dd_bt:>12.2%} │ {abs(rep['最大回撤']):>12.2%} │
    └──────────────────────┴──────────────┴──────────────┘
    🔍 Sharpe 的微小差异：backtrader 的 SharpeRatio 默认用总体标准差（ddof=0），pandas 默认 ddof=1；
       用 ddof=0 重算得 {sr_pop:.4f}。另外 backtrader 的 riskfreerate 默认是 1%，本课显式设为 0。
    📋 TradeAnalyzer：已平仓 {closed} 笔，盈利 {won} 笔（胜率 {won / closed if closed else float('nan'):.1%}）
    💡 第三方框架的指标口径（年化方式、无风险利率、标准差自由度）都可能与你的不同 —— 先对拍再引用。
    """)


# ==================== 主函数 ====================


def main():
    architecture()
    commission_check()
    df = load_bars(SYMBOL, START, END)
    naive_demo(df)
    out = compare_with_engine(df)
    analyzer_check(df, out[0.0][2])

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    bt_nav, eng_nav, _ = out[0.0005]
    ax.plot(bt_nav / CAPITAL, label="backtrader（A 股版）")
    ax.plot(eng_nav / CAPITAL, label="06 课自研引擎", ls="--")
    ax.set_title(f"{SYMBOL} MA20：backtrader 与自研引擎对拍（5bp 滑点）")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '10_backtrader_vs_engine.png', subdir='phase-4')}")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 在 AShareCommission 中加入按日期切换的印花税（2023-08-28 前 0.1%），需要从 data 中取当前日期
    2. 用 bt.analyzers.Transactions 导出每笔成交，与 06 课引擎的 fills 逐笔比对
    3. 加载 3 只股票（多个 data），实现等权组合；注意多个 data 的日期对齐与停牌（某只股票缺 bar）的处理
    4. 把 cheat_on_open 关掉、在 next() 中直接 buy()，比较拒单与收益的变化，并解释原因
    """)


if __name__ == "__main__":
    main()
