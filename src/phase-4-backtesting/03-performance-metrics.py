"""
绩效指标：Sharpe、Sortino、MDD、Calmar、胜率
==============================================

学习目标：
    1. 手写一套完整的绩效指标，并与 bt_core 的实现对拍
    2. 理解 Sharpe 的年化、无风险利率、t 值，以及它的局限
    3. 正确实现 Sortino（分母用全部样本数），知道常见的错误写法
    4. 计算最大回撤及其起点、谷底、修复日期，理解 Calmar 为什么不能跨区间比较
    5. 计算交易层面指标：胜率、盈亏比、Profit Factor、换手率、成本占比

核心概念：
    - Sharpe = E[R − Rf] / σ(R − Rf) × √252
    - Sortino：分母是下行偏差 √E[min(R − MAR, 0)²]
    - MDD = min(V_t / max_{s≤t} V_s − 1)，Calmar = CAGR / |MDD|
    - 胜率 × 盈亏比 共同决定期望：E = p·W − (1−p)·L

前置知识：
    - 02-vectorized-backtest.py
    - Phase 2 收益率、波动率与回撤；Phase 3 Sharpe 的 t 检验

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import (cost_summary, extract_trades, format_report, performance_report, trade_stats,
                     vectorized_backtest)
from utils.data import get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"
COST = 0.001
N = 252  # 每年交易日数


# ==================== 第一部分：指标全景 ====================


def metrics_overview():
    print("=" * 60)
    print("第一部分：指标全景")
    print("=" * 60)

    print("""
    ┌──────────┬─────────────────────┬──────────────────────────────────┐
    │ 维度     │ 指标                │ 一句话解释                       │
    ├──────────┼─────────────────────┼──────────────────────────────────┤
    │ 收益     │ 年化收益 CAGR       │ 复利意义上每年赚多少             │
    │ 风险     │ 年化波动率          │ 收益的离散程度                   │
    │ 风险     │ 最大回撤 MDD        │ 最惨时从高点亏了多少             │
    │ 风险调整 │ Sharpe              │ 每单位总波动换来多少超额收益     │
    │ 风险调整 │ Sortino             │ 每单位"下行波动"换来多少收益     │
    │ 风险调整 │ Calmar              │ 每单位最大回撤换来多少年化收益   │
    │ 相对基准 │ 超额收益、信息比率  │ 相对基准的表现                   │
    │ 交易     │ 胜率、盈亏比、PF    │ 单笔交易的质量                   │
    │ 交易     │ 换手率、成本占比    │ 成本敏感度、容量                 │
    └──────────┴─────────────────────┴──────────────────────────────────┘

    📌 一个回测报告至少要同时给出：收益、风险、风险调整收益、交易统计、成本 —— 单看一个数会被误导。
    """)


# ==================== 第二部分：收益、波动与 Sharpe ====================


def cagr(ret: pd.Series) -> float:
    """几何年化收益：(期末净值)^(1/年数) − 1"""
    return (1 + ret).prod() ** (N / len(ret)) - 1


def sharpe(ret: pd.Series, rf_annual: float = 0.0, periods: int = N) -> float:
    excess = ret - rf_annual / periods
    return excess.mean() / excess.std() * np.sqrt(periods)


def return_and_sharpe(ret: pd.Series):
    print("\n" + "=" * 60)
    print(f"第二部分：收益、波动与 Sharpe（{SYMBOL} MA20 择时，次日开盘成交，单边成本 0.1%）")
    print("=" * 60)

    arith = ret.mean() * N
    geo = cagr(ret)
    vol = ret.std() * np.sqrt(N)
    print(f"""
    算术年化 = 日均收益 × 252      = {arith:+.2%}
    几何年化 = 净值^(1/年数) − 1   = {geo:+.2%}
    二者之差 ≈ σ²/2 = {vol ** 2 / 2:.2%}（"波动拖累"，Phase 2）—— 报告收益要用几何年化
    年化波动 = 日标准差 × √252     = {vol:.2%}
    """)

    years = len(ret) / N
    sr0, sr2 = sharpe(ret, 0.0), sharpe(ret, 0.02)
    print(f"    Sharpe（Rf = 0）       = {sr0:.3f}")
    print(f"    Sharpe（Rf = 2%/年）   = {sr2:.3f}   ← 无风险利率口径不同，结果不同，报告要写明")
    print(f"    Sharpe t 值 ≈ SR × √年数 = {sr0:.3f} × √{years:.2f} = {sr0 * np.sqrt(years):.2f}"
          f"   （Phase 3：|t| < 2 时难以与运气区分）")

    monthly = (1 + ret).resample("ME").prod() - 1
    print(f"    用月度收益计算再 ×√12   = {sharpe(monthly, periods=12):.3f}   ← 与日频结果不同；"
          f"收益有自相关时 √N 年化会失真")

    print("""
    ┌────────────────────────┬──────────────────────────────────────────┐
    │ 年化 Sharpe（扣成本后）│ 通常的含义（仅供直觉）                   │
    ├────────────────────────┼──────────────────────────────────────────┤
    │ < 0.5                  │ 弱，难以与运气区分                       │
    │ 0.5 – 1.0              │ 中等，许多长期可行的低频策略在这里       │
    │ 1.0 – 2.0              │ 较好                                     │
    │ > 2.0（日线策略）      │ 先怀疑 bug：前视偏差、成本遗漏、幸存者   │
    └────────────────────────┴──────────────────────────────────────────┘
    Sharpe 的局限：把上涨波动也当风险；看不到偏度和肥尾（卖期权类策略爆仓前 Sharpe 往往很高）
    """)


# ==================== 第三部分：Sortino ====================


def sortino(ret: pd.Series, mar: float = 0.0, periods: int = N) -> float:
    """正确写法：下行偏差的分母是全部样本数"""
    downside = np.sqrt((np.minimum(ret - mar, 0) ** 2).mean())
    return (ret - mar).mean() / downside * np.sqrt(periods)


def sortino_wrong(ret: pd.Series, mar: float = 0.0, periods: int = N) -> float:
    """常见错误：只用负收益样本算标准差（分母变成负收益样本数，且还减了均值）"""
    return (ret - mar).mean() / ret[ret < mar].std() * np.sqrt(periods)


def sortino_demo(ret: pd.Series):
    print("\n" + "=" * 60)
    print("第三部分：Sortino——只惩罚'坏的'波动")
    print("=" * 60)

    print("""
    Sortino = E[R − MAR] / DD × √252,   DD = √( E[ min(R − MAR, 0)² ] )
      MAR：最低可接受收益（常取 0）
      DD ：下行偏差 —— 高于 MAR 的样本贡献 0，但仍计入分母的样本数
    """)
    n_down = int((ret < 0).sum())
    print(f"    样本 {len(ret)} 天，其中收益为负 {n_down} 天，为 0（空仓）{int((ret == 0).sum())} 天")
    print(f"    ✅ 正确 Sortino            = {sortino(ret):.3f}")
    print(f"    ❌ 只用负收益样本的 std    = {sortino_wrong(ret):.3f}")
    print("    两种写法差异可以很大；不同库的口径也不完全一致 —— 用之前先和自己的实现对拍")


# ==================== 第四部分：最大回撤与 Calmar ====================


def max_drawdown(nav: pd.Series) -> dict:
    """最大回撤：深度、前高日期、谷底日期、修复日期"""
    running_max = nav.cummax()
    dd = nav / running_max - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    recovered = nav.loc[trough:][nav.loc[trough:] >= nav.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return {"mdd": dd.min(), "peak": peak, "trough": trough, "recovery": recovery, "dd": dd}


def drawdown_demo(ret: pd.Series) -> pd.Series:
    print("\n" + "=" * 60)
    print("第四部分：最大回撤与 Calmar")
    print("=" * 60)

    nav = (1 + ret).cumprod()
    info = max_drawdown(nav)
    n_days = lambda a, b: len(nav.loc[a:b]) - 1  # noqa: E731
    print(f"""
    最大回撤：{info['mdd']:.2%}
      前高：{info['peak'].date()}   谷底：{info['trough'].date()}（下跌 {n_days(info['peak'], info['trough'])} 个交易日）
      修复：{info['recovery'].date() if info['recovery'] is not None else '样本结束时仍未修复'}
    Calmar = CAGR / |MDD| = {cagr(ret):.2%} / {abs(info['mdd']):.2%} = {cagr(ret) / abs(info['mdd']):.3f}
    """)

    # MDD 随样本长度增长：同一个"策略"，回测越长，最大回撤越深
    rng = np.random.default_rng(0)
    mu, sigma = 0.08 / N, 0.20 / np.sqrt(N)  # 年化 8% 收益、20% 波动
    rows = []
    for years in [1, 3, 5, 10]:
        sims = rng.normal(mu, sigma, size=(2000, years * N))
        navs = np.cumprod(1 + sims, axis=1)
        mdds = (navs / np.maximum.accumulate(navs, axis=1) - 1).min(axis=1)
        rows.append([years, f"{np.median(mdds):.1%}", f"{np.percentile(mdds, 5):.1%}"])
    print("    🎲 蒙特卡洛（年化收益 8%、波动 20% 的正态日收益，各 2000 条路径，种子 0）：")
    print(pd.DataFrame(rows, columns=["回测年数", "MDD 中位数", "MDD 5% 分位"]).to_string(index=False))
    print("""
    💡 同一个收益过程，回测越长 MDD 越深 → 不同区间长度的 Calmar 不可直接比较；
       实盘遇到比回测更深的回撤，是常态而不是意外。
    """)
    return info["dd"]


# ==================== 第五部分：交易层面指标 ====================


def trade_level_demo(bt: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：交易层面指标——胜率、盈亏比、换手、成本占比")
    print("=" * 60)

    # 手写：把连续持仓的区间切成一笔笔交易
    pos, ret = bt["pos"], bt["ret"]
    held = pos > 0
    trade_id = (held & ~held.shift(1, fill_value=False)).cumsum().where(held)
    # 平仓日（持仓结束后的第一天）还承担隔夜段收益和卖出成本，归入上一笔交易
    exit_day = (~held) & held.shift(1, fill_value=False)
    trade_id = trade_id.fillna(trade_id.ffill().where(exit_day))
    trades = (1 + ret).groupby(trade_id).prod() - 1

    wins, losses = trades[trades > 0], trades[trades <= 0]
    p, w, l = len(wins) / len(trades), wins.mean(), abs(losses.mean())
    print(f"""
    交易笔数        {len(trades)}
    胜率            {p:.2%}（按"笔"统计；日胜率是另一回事，见下）
    平均盈利 / 亏损 {w:+.2%} / {-l:+.2%}
    盈亏比          {w / l:.2f}
    Profit Factor   {wins.sum() / abs(losses.sum()):.2f}（总盈利 / 总亏损，按每笔收益率相加的近似）
    每笔期望        p·W − (1−p)·L = {p:.2%}×{w:.2%} − {1 - p:.2%}×{l:.2%} = {p * w - (1 - p) * l:+.3%}
    """)
    print("    💡 趋势策略常见'低胜率 + 高盈亏比'；只看胜率会误判策略好坏")

    core = trade_stats(extract_trades(pos, ret))
    print(f"\n    ✅ 与 bt_core.trade_stats 对拍：笔数 {int(core['交易笔数'])}，胜率 {core['胜率']:.2%}，"
          f"盈亏比 {core['盈亏比']:.2f}，平均持仓 {core['平均持仓天数']:.1f} 天")

    days_in = ret[pos > 0]
    print(f"    日胜率（只统计持仓日）{(days_in > 0).mean():.2%}  vs  按笔胜率 {p:.2%} —— 报告时要说明口径")

    cs = cost_summary(bt)
    print(f"""
    年化换手 = Σ|Δw| / 年数                 = {cs['年化换手']:.1f} 倍（买、卖都计入）
    年化成本拖累 ≈ 年化换手 × 单边成本      = {cs['年化成本拖累']:.2%}
    成本占比 = 成本拖累 / 扣费前算术年化收益 = {cs['成本占比']:.1%}
    盈亏平衡成本 = 扣费前年化收益 / 年化换手 = {cs['盈亏平衡成本(bp)']:.1f} bp（单边成本高于它，策略就不赚钱）
    """)


# ==================== 第六部分：完整报告与对拍 ====================


def full_report(bt: pd.DataFrame, bh: pd.Series, bench: pd.Series, dd: pd.Series):
    print("\n" + "=" * 60)
    print("第六部分：完整绩效报告与对拍")
    print("=" * 60)

    ret = bt["ret"]
    hand = {"年化收益": cagr(ret), "Sharpe": sharpe(ret), "Sortino": sortino(ret),
            "最大回撤": max_drawdown((1 + ret).cumprod())["mdd"]}
    core = performance_report(ret)
    diffs = {k: abs(hand[k] - core[k]) for k in hand}
    print("✅ 手写版本 vs bt_core.performance_report：" + "，".join(f"{k} 差 {v:.1e}" for k, v in diffs.items()))

    table = pd.DataFrame({
        "MA20 择时": format_report(performance_report(ret)),
        f"{NAME} 买入持有": format_report(performance_report(bh)),
        "沪深300": format_report(performance_report(bench)),
    })
    print(table.to_string())

    active = ret - bench
    ir = active.mean() / active.std() * np.sqrt(N)
    print(f"\n    相对沪深 300：年化超额（算术）{active.mean() * N:+.2%}，跟踪误差 {active.std() * np.sqrt(N):.2%}，"
          f"信息比率 IR {ir:.2f}")
    print("    （沪深 300 价格指数不含分红，严格比较应使用全收益指数）")

    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot((1 + ret).cumprod(), label="MA20 择时")
    axes[0].plot((1 + bh).cumprod(), label="买入持有", color="gray", ls="--")
    axes[0].set_title(f"{SYMBOL} {NAME}：净值与回撤")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].fill_between(dd.index, dd.to_numpy(), 0, color="tab:red", alpha=0.4)
    axes[1].set_ylabel("回撤")
    axes[1].grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '03_performance_metrics.png', subdir='phase-4')}")


# ==================== 主函数 ====================


def main():
    metrics_overview()

    df = get_stock_daily_adj(SYMBOL, START, END)
    close, open_ = df["adj_close"], df["open"] * df["adj_factor"]
    signal = (close > close.rolling(20).mean()).astype(float)
    bt = vectorized_backtest(close, signal, COST, open_=open_, execution="next_open")
    bh = close.pct_change().fillna(0.0)
    bench = get_index_daily("000300", START, END)["close"].pct_change().reindex(bt.index).fillna(0.0)

    return_and_sharpe(bt["ret"])
    sortino_demo(bt["ret"])
    dd = drawdown_demo(bt["ret"])
    trade_level_demo(bt)
    full_report(bt, bh, bench, dd)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 Rf = 2% 重新计算 Sortino（MAR = 日无风险利率），和 Rf = 0 时相比变化多大？
    2. 实现"最长回撤持续期"（净值低于前高的最长连续天数），它和 MDD 哪个对投资者更难熬？
    3. 构造一个"每天赚 0.05%，每 200 天亏 15%"的收益序列，计算其 Sharpe、Sortino、MDD，
       体会 Sharpe 为什么看不到尾部风险
    4. 按自然年分组输出每年的收益、Sharpe、MDD（分年度表现是回测报告的必备项）
    """)


if __name__ == "__main__":
    main()
