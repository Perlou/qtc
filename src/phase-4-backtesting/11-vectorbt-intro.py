"""
vectorbt 大规模参数回测
=======================

学习目标：
    1. 了解 vectorbt 的定位：向量化 + Numba，几秒钟跑完上千组回测
    2. 搞清 from_signals 的成交时点：默认"信号所在 bar 的收盘价成交"，与 02 课的"收盘成交"假设相同；
       一旦把成交价换成开盘价却不移动信号，就成了未来函数
    3. 用 MA.run_combs 在多只股票上做大规模参数网格（上千组），并用 DSR 修正挑选偏差
    4. 认识两个口径陷阱：年化默认按 365 天；fees 默认不区分买卖方向
    5. 用不复权价 + size_granularity + 方向性费用 + 涨跌停掩码，近似 A 股约束，并与 06 课引擎比较

核心概念：
    - Portfolio.from_signals(close, entries, exits, price=...)：信号驱动的组合模拟
    - 信号后移一根 bar（shift(1)）+ price=open ⇔ "t 日收盘出信号、t+1 日开盘成交"
    - 参数扫描越容易，过拟合越容易：务必记录尝试次数

前置知识：
    - 02-vectorized-backtest.py、06-event-driven-engine.py、08-overfitting-traps.py

环境要求：
    - pip install vectorbt pandas akshare（本课在 vectorbt 1.2.0 + pandas 3 上验证通过；首次运行 Numba 编译需十几秒）
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from dotenv import load_dotenv

from bt_core import (CostModel, board_rule, expected_max_sr, limit_prices, load_bars, load_lesson, performance_report,
                     psr, trading_calendar, vectorized_backtest)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

SYMBOL, START, END = "600519", "20180101", "20241231"
UNIVERSE = ["600519", "601088", "000002", "600036", "000858"]
WINDOWS = np.arange(5, 121, 5)
CAPITAL = 1_000_000
YEAR = "252 days"  # A 股一年约 252 个交易日；vectorbt 默认按 365 天年化


# ==================== 第一部分：vectorbt 简介与年化口径 ====================


def intro(df: pd.DataFrame):
    print("=" * 60)
    print("第一部分：vectorbt 简介与年化口径")
    print("=" * 60)

    print(f"""
    vectorbt 版本：{vbt.__version__}，pandas 版本：{pd.__version__}

    ┌──────────────┬────────────────────────────┬────────────────────────────┐
    │ 维度         │ backtrader（10 课）        │ vectorbt                   │
    ├──────────────┼────────────────────────────┼────────────────────────────┤
    │ 架构         │ 事件驱动                   │ 向量化 + Numba             │
    │ 速度         │ 慢                         │ 很快，适合大规模参数扫描   │
    │ A 股约束     │ 自定义佣金类、Sizer        │ 需在信号/价格层面近似      │
    │ 适合         │ 精确验证、与自研引擎对拍   │ 信号研究、参数扫描、原型   │
    └──────────────┴────────────────────────────┴────────────────────────────┘
    """)
    c = df["adj_close"]
    sig = c > c.rolling(20).mean()
    pf = vbt.Portfolio.from_signals(c, sig, ~sig, freq="1D", init_cash=CAPITAL)
    print(f"""    ⚠️ 年化口径陷阱：同一个组合
       pf.sharpe_ratio()                      = {pf.sharpe_ratio():.4f}   ← 默认 year_freq = 365 天（加密货币口径）
       pf.sharpe_ratio(year_freq="{YEAR}")  = {pf.sharpe_ratio(year_freq=YEAR):.4f}
       bt_core.performance_report             = {performance_report(pf.returns())['Sharpe']:.4f}
       A 股日线应使用 252 天；本课之后所有 vectorbt 指标都显式传入 year_freq
    """)


# ==================== 第二部分：成交时点与未来函数 ====================


def execution_timing(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"第二部分：from_signals 的成交时点（{SYMBOL} MA20，零成本，与 02 课对拍）")
    print("=" * 60)

    c, o = df["adj_close"], df["adj_open"]
    sig = c > c.rolling(20).mean()
    entries, exits = sig, ~sig  # 状态型信号：在持仓中重复的买入信号会被忽略

    a = vbt.Portfolio.from_signals(c, entries, exits, freq="1D", init_cash=CAPITAL)
    bad = vbt.Portfolio.from_signals(c, entries, exits, price=o, freq="1D", init_cash=CAPITAL)
    good = vbt.Portfolio.from_signals(c, entries.shift(1, fill_value=False), exits.shift(1, fill_value=False),
                                      price=o, freq="1D", init_cash=CAPITAL)
    v_close = vectorized_backtest(c, sig.astype(float), 0.0, execution="close")
    v_open = vectorized_backtest(c, sig.astype(float), 0.0, open_=o, execution="next_open")

    rows = [
        ["默认：信号 bar 的收盘价成交", f"{a.total_return():+.2%}", f"{v_close['nav'].iloc[-1] - 1:+.2%}（02 课 收盘成交）",
         f"{abs(a.value() / CAPITAL - v_close['nav']).max():.1e}"],
        ["❌ price=open，信号不后移", f"{bad.total_return():+.2%}", "—（未来函数）", "—"],
        ["✅ price=open，信号后移 1 根", f"{good.total_return():+.2%}", f"{v_open['nav'].iloc[-1] - 1:+.2%}（02 课 次日开盘）",
         f"{abs(good.value() / CAPITAL - v_open['nav']).max():.1e}"],
    ]
    print(pd.DataFrame(rows, columns=["vectorbt 写法", "总收益", "02 课向量化", "逐日净值最大差"]).to_string(index=False))
    print("""
    🔍 解读：
    - 默认写法 = "t 日收盘出信号、t 日收盘价成交"：信号要用到收盘价，又要以同一个收盘价成交，
      严格说需要在收盘前预知收盘价（02 课说过，这是日频低换手策略常用的偏乐观近似）
    - 把成交价换成开盘价却不移动信号 = 用当天收盘才知道的信号，去买当天早上的开盘价 —— 典型未来函数，
      收益好得离谱
    - 正确做法：entries.shift(1) + price=open，与 02 课、06 课引擎的假设完全一致（零成本时逐日对上）
    """)


# ==================== 第三部分：大规模参数网格 ====================


def load_panel() -> pd.DataFrame:
    calendar = trading_calendar(START, END)
    panel = pd.DataFrame({s: load_bars(s, START, END)["adj_close"] for s in UNIVERSE}).reindex(calendar)
    return panel.ffill().bfill()  # 停牌日沿用停牌前价格估值；开头若缺失用首个价格占位（仅影响预热期）


def parameter_grid(panel: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：大规模参数网格（多股票 × 双均线）")
    print("=" * 60)

    t0 = time.perf_counter()
    fast, slow = vbt.MA.run_combs(panel, window=WINDOWS, r=2, short_names=["fast", "slow"])
    entries = fast.ma_crossed_above(slow).shift(1, fill_value=False)  # 次日成交
    exits = fast.ma_crossed_below(slow).shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(panel, entries, exits, fees=0.0005, slippage=0.0005, freq="1D", init_cash=CAPITAL)
    sr = pf.sharpe_ratio(year_freq=YEAR)
    elapsed = time.perf_counter() - t0
    n = len(sr)
    print(f"""
    {len(UNIVERSE)} 只股票 × {n // len(UNIVERSE)} 组 (快线, 慢线) = {n} 次回测，耗时 {elapsed:.1f} 秒（Numba 已在前两部分完成编译）
    成交：信号后移 1 根 bar、收盘价成交（即 02 课的"次日收盘成交"）；单边费用 5bp + 滑点 5bp
    """)
    top = sr.sort_values(ascending=False).head(8)
    print("📋 Sharpe 最高的 8 个组合：")
    print(top.round(3).to_string())
    per_stock = pd.DataFrame({"最优 Sharpe": sr.groupby(level=-1).max(), "中位数 Sharpe": sr.groupby(level=-1).median(),
                              "Sharpe>0 比例": sr.groupby(level=-1).apply(lambda s: (s > 0).mean())})
    print("\n📋 按股票汇总：")
    print(per_stock.round(3).to_string())

    rets = pf.returns()
    daily_sr = (rets.mean() / rets.std()).to_numpy()
    best_col = sr.idxmax()
    dsr = psr(rets[best_col].to_numpy(), expected_max_sr(daily_sr))
    print(f"""
    🏆 全部 {n} 次回测中最优：{best_col[2]} 快线 {int(best_col[0])} / 慢线 {int(best_col[1])}，Sharpe {sr.max():.2f}
       PSR(0) = {psr(rets[best_col].to_numpy(), 0):.3f}，DSR（N = {n}）= {dsr:.3f}
    💡 "尝试的组合数"这个数字必须写进研究日志；挑出来的最优值要用 DSR 修正、用 Walk-Forward 检验（08、09 课）
    """)
    return {"sr": sr, "elapsed": elapsed, "n": n}


def plot_heatmap(res: dict):
    sr = res["sr"].xs(SYMBOL, level=-1)
    grid = sr.unstack("slow_window")
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 7))
    lim = np.nanmax(np.abs(grid.to_numpy()))
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", origin="lower", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(grid.columns)), grid.columns, fontsize=7)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=7)
    ax.set_xlabel("慢线周期")
    ax.set_ylabel("快线周期")
    ax.set_title(f"{SYMBOL} 双均线参数热力图（vectorbt，Sharpe，252 天年化）")
    fig.colorbar(im, ax=ax, label="Sharpe")
    print(f"📁 图片已保存：{save_figure(fig, '11_vectorbt_heatmap.png', subdir='phase-4')}")


# ==================== 第四部分：近似 A 股约束 ====================


def ashare_approx(df: pd.DataFrame, res: dict):
    print("\n" + "=" * 60)
    print(f"第四部分：在 vectorbt 中近似 A 股约束，并与 06 课引擎比较（{SYMBOL} MA20）")
    print("=" * 60)

    m = CostModel()
    up_down = [limit_prices(p, board_rule(SYMBOL).limit_pct) if pd.notna(p) else (np.inf, -np.inf) for p in df["pre_close"]]
    up = pd.Series([u for u, _ in up_down], index=df.index)
    down = pd.Series([d for _, d in up_down], index=df.index)
    can_buy, can_sell = df["open"] < up - 1e-6, df["open"] > down + 1e-6

    c = df["adj_close"]
    target = (c > c.rolling(20).mean()).shift(1, fill_value=False)  # t 日收盘的目标 → t+1 日开盘执行
    entries = target & can_buy  # 状态型信号：被涨停挡住的买单，次日继续尝试
    exits = ~target & can_sell
    buy_fee = m.commission_rate + m.transfer_fee_rate
    fees = pd.Series(np.where(exits, buy_fee + m.stamp_duty_rate, buy_fee), index=df.index)  # 卖出加印花税

    t0 = time.perf_counter()
    pf = vbt.Portfolio.from_signals(df["close"], entries, exits, price=df["open"], fees=fees, slippage=m.slippage,
                                    size=np.inf, size_granularity=100, freq="1D", init_cash=CAPITAL)
    t_vbt = time.perf_counter() - t0

    engine = load_lesson("06-event-driven-engine.py")
    t0 = time.perf_counter()
    eng = engine.Backtester({SYMBOL: df}, trading_calendar(START, END), engine.EngineConfig())
    hist = eng.run()
    t_eng = time.perf_counter() - t0
    div = sum(d[2] for d in eng.portfolio.dividends)
    orders = pf.orders.records_readable
    print(f"""
    做法：
      - 不复权 close/open 撮合 + size_granularity=100（整数手）
      - fees 传入与信号同形状的数组：卖出那天的费率 = 佣金 + 过户费 + 印花税（绕过"费用不分方向"的限制）
      - 开盘涨停的日子屏蔽买入信号、开盘跌停的日子屏蔽卖出信号；用"状态型信号"让被挡住的订单次日重试
      - 做不到的：最低 5 元佣金（fixed_fees 是叠加而非取最大值）、除权除息现金分红

    ┌────────────────────────────┬────────────────┬────────────────┐
    │                            │ vectorbt 近似  │ 06 课引擎      │
    ├────────────────────────────┼────────────────┼────────────────┤
    │ 期末资产                   │ {pf.final_value():>14,.0f} │ {hist['equity'].iloc[-1]:>14,.0f} │
    │ 成交笔数                   │ {len(orders):>14d} │ {len(eng.portfolio.fills):>14d} │
    │ 买入股数是否都是 100 的倍数│ {str(bool((orders['Size'] % 100 == 0).all())):>14s} │ {'True':>14s} │
    └────────────────────────────┴────────────────┴────────────────┘
    引擎记入的现金分红合计 {div:,.0f} 元 —— vectorbt 用不复权价撮合时没有这部分收入，这是两者差距的主要来源之一；
    其余差异来自最低佣金、价格取整、补仓规则等细节。
    两难：用复权价则整数手、涨跌停判断失真；用不复权价则分红丢失 —— 这正是精确验证要回到事件驱动引擎的原因。
    """)
    per_run_eng = t_eng
    print(f"    ⏱️ 速度：第三部分 {res['n']} 次 vectorbt 回测共 {res['elapsed']:.1f} 秒；"
          f"06 课引擎单次 {per_run_eng * 1000:.0f} ms，跑 {res['n']} 次约需 {per_run_eng * res['n']:.0f} 秒")


# ==================== 第五部分：使用 vectorbt 的注意事项 ====================


def cautions():
    print("\n" + "=" * 60)
    print("第五部分：使用 vectorbt 的注意事项")
    print("=" * 60)

    print("""
    ┌──────────────────────────────┬──────────────────────────────────────────────┐
    │ 陷阱                         │ 对策                                         │
    ├──────────────────────────────┼──────────────────────────────────────────────┤
    │ 默认同 bar 收盘成交          │ 信号 shift(1)；需要开盘成交时 price=open     │
    │ price=open 但信号未后移      │ 未来函数！两者必须同时改                     │
    │ 年化默认 365 天              │ 显式传 year_freq="252 days"                  │
    │ fees 不分买卖方向            │ 传入与信号同形状的费率数组                   │
    │ 无最低佣金、无分红           │ 精确验证交给事件驱动引擎                     │
    │ 几秒扫完上千组参数           │ 记录尝试次数、看热力图、DSR、Walk-Forward    │
    │ crossed 型信号被掩码挡掉会丢 │ 改用状态型信号（目标仓位），或自己处理重试   │
    └──────────────────────────────┴──────────────────────────────────────────────┘

    推荐工作流（CONCEPT 第 7 节"漏斗式"）：
      vectorbt 扫描上千组参数 → 留下"参数平原"上的少数候选 → 06 课引擎 / backtrader 精确验证
      → Walk-Forward → 保留区只跑一次
    """)


# ==================== 主函数 ====================


def main():
    df = load_bars(SYMBOL, START, END)
    intro(df)
    execution_timing(df)
    res = parameter_grid(load_panel())
    plot_heatmap(res)
    ashare_approx(df, res)
    cautions()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第三部分的成交改为"次日开盘"（price=开盘价宽表），最优参数和 Sharpe 分布有何变化？
    2. 用 vbt.MA.run_combs 的结果画出 5 只股票各自的热力图，最优参数在不同股票之间是否一致？
       不一致说明什么？
    3. 结合 09 课：对第三部分的网格做滚动 Walk-Forward（每年用过去 3 年选参数），报告样本外 Sharpe
    4. 在第四部分中加入"按日期切换的印花税"（2023-08-28 前 0.1%），观察期末资产的变化
    """)


if __name__ == "__main__":
    main()
