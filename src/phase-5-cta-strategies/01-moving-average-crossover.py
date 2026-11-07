"""
双均线策略
==========

学习目标：
    1. 理解 SMA / EMA 的计算方式与"滞后"的本质
    2. 实现双均线择时策略：信号 shift(1) 次日执行、扣交易成本、与买入持有对比
    3. 用逐笔交易统计理解趋势策略"低胜率、高盈亏比、正偏"的收益结构
    4. 用数据回答学习检查点："趋势策略为什么在震荡市失效"

核心概念：
    - SMA：过去 N 天收盘价的简单平均；EMA：指数加权平均，alpha = 2/(N+1)
    - 金叉 / 死叉：短均线上穿 / 下穿长均线
    - 效率系数 ER（Efficiency Ratio）= |区间净涨跌| / 区间内每日涨跌绝对值之和，
      越接近 1 越"趋势"，越接近 0 越"震荡"

前置知识：
    - Phase 1 第 01 课（信号 → 持仓 → 净值）
    - Phase 4 回测基础（未来函数、交易成本、绩效指标）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import backtest, perf_stats, stats_table, trade_stats
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20100101", "20251231"
COST = 0.0005  # 单边成本示例：5 个基点（佣金 + 滑点），按名义金额计


def load_close() -> pd.Series:
    df = get_index_daily("000300", START, END)
    return df["close"]


# ==================== 第一部分：均线与滞后 ====================


def moving_average_basics(close: pd.Series):
    print("=" * 60)
    print("第一部分：SMA、EMA 与滞后")
    print("=" * 60)

    print("""
    SMA_t(N) = (P_t + P_{t-1} + ... + P_{t-N+1}) / N         —— 每天权重相同
    EMA_t    = α·P_t + (1-α)·EMA_{t-1},  α = 2/(N+1)          —— 越近的价格权重越大

    比喻：长均线是"大船的航向"，短均线是"小艇的航向"。
         小艇已经转向而大船还没转 → 风向可能变了（金叉 / 死叉）。
    """)

    for n in (20, 60):
        alpha = 2 / (n + 1)
        w_recent = sum(alpha * (1 - alpha) ** k for k in range(n))
        print(f"    EMA({n}) 的 α = {alpha:.4f}，最近 {n} 天的权重合计 {w_recent:.1%}（其余权重在更早的历史上）")

    # 滞后：找到使 "均线" 与 "k 天前的价格" 最接近的 k
    sma = close.rolling(60).mean()
    ema = close.ewm(span=60, adjust=False).mean()
    rows = []
    for name, ma in [("SMA(60)", sma), ("EMA(60)", ema)]:
        err = {k: (ma - close.shift(k)).abs().mean() for k in range(0, 61, 5)}
        best = min(err, key=err.get)
        rows.append([name, best, err[0], err[best]])
    lag = pd.DataFrame(rows, columns=["均线", "最接近 k 天前价格的 k", "与当天价格平均偏差", "与 k 天前价格平均偏差"])
    print("\n📊 均线的滞后（沪深 300）：")
    print(lag.round(1).to_string(index=False))
    print("""
    💡 SMA(N) 大致相当于 N/2 天前的价格；EMA 滞后略小，但同样无法消除。
       滞后是趋势策略的本质，不是 bug：它"不抄底、不逃顶"，只吃中间那段鱼身。
    """)


# ==================== 第二部分：双均线回测 ====================


def ma_cross_target(close: pd.Series, fast: int, slow: int, long_only: bool = False) -> pd.Series:
    """收盘后计算目标仓位：快线 > 慢线 → +1，否则 -1（或只做多时为 0）"""
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    sig = np.sign(ma_f - ma_s)
    if long_only:
        sig = sig.clip(lower=0)
    return sig.where(ma_s.notna(), 0.0)


def ma_cross_backtest(close: pd.Series):
    print("\n" + "=" * 60)
    print("第二部分：双均线策略回测（20 / 60 日）")
    print("=" * 60)

    print("""
    规则（第 t 天收盘后计算，第 t+1 天持有）：
        MA20 > MA60 → 做多      MA20 < MA60 → 做空（或空仓）

    回测三件套（每一课都会坚持）：
        ① 信号 shift(1)：今天收盘才知道信号，明天才能持仓
        ② 扣成本：仓位每变化 1 个单位，扣 COST × 1 的成本（-1 → +1 换手为 2）
        ③ 与基准对比：买入持有沪深 300

    ⚠️ A 股现货不能随意做空：只做多版本可以用 ETF 实现；
       多空版本需要用股指期货（IF）实现，这里用指数收益近似（忽略基差与展期）。
    """)

    ret = close.pct_change().fillna(0)
    bt_lo = backtest(ma_cross_target(close, 20, 60, long_only=True), ret, COST)
    bt_ls = backtest(ma_cross_target(close, 20, 60), ret, COST)
    bt_nocost = backtest(ma_cross_target(close, 20, 60), ret, 0.0)

    table = stats_table({
        "买入持有": perf_stats(ret),
        "双均线-只做多": perf_stats(bt_lo["ret"], bt_lo["turnover"], bt_lo["pos"]),
        "双均线-多空": perf_stats(bt_ls["ret"], bt_ls["turnover"], bt_ls["pos"]),
        "多空(不扣成本)": perf_stats(bt_nocost["ret"], bt_nocost["turnover"], bt_nocost["pos"]),
    })
    print(f"📊 沪深 300，{close.index[0].date()} ~ {close.index[-1].date()}，单边成本 {COST:.2%}：")
    print(table.to_string())

    total_cost = bt_ls["cost"].sum()
    n_switch = int((bt_ls["pos"].diff().abs() > 0).sum())
    print(f"""
    🔄 多空版本共调仓 {n_switch} 次，累计成本（简单相加）约 {total_cost:.1%} 的净值
    📌 读表要点：
       - Sharpe 和最大回撤要与买入持有对比，而不是只看总收益
       - "不扣成本"一列只用来衡量成本侵蚀，永远不要拿它当结论
    """)
    return ret, bt_lo, bt_ls


# ==================== 第三部分：收益结构 ====================


def return_structure(ret: pd.Series, bt_ls: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：趋势策略的收益结构（逐笔交易）")
    print("=" * 60)

    ts = trade_stats(bt_ls["pos"], ret)
    trades = ts["_trades"].sort_values(ascending=False)
    top5 = trades.head(5).sum()
    total = trades.sum()
    print(f"""
    📊 多空双均线逐笔统计（一笔 = 连续同方向持仓）：
       交易笔数   {ts['笔数']}
       胜率       {ts['胜率']:.1%}
       盈亏比     {ts['盈亏比']:.2f}（平均盈利 / 平均亏损）
       单笔偏度   {ts['单笔偏度']:.2f}（> 0 表示右尾长：少数大赚）
       平均持有   {ts['平均持有天数']:.1f} 天
       最赚钱的 5 笔收益之和 {top5:+.1%}，全部交易收益之和 {total:+.1%}
    """)
    print("    最赚钱的 3 笔与最亏钱的 3 笔（按开仓日）：")
    show = pd.concat([trades.head(3), trades.tail(3)])
    for d, r in show.items():
        print(f"      {d.date()}  {'做多' if bt_ls['pos'].loc[d] > 0 else '做空'}  {r:+.1%}")
    print("""
    💡 典型的趋势跟踪特征：胜率不高，靠少数大行情的盈利覆盖大量小额亏损。
       如果把最好的几笔交易拿掉，策略可能就不赚钱了 —— 所以"不错过大趋势"比"提高胜率"更重要。
    """)


# ==================== 第四部分：为什么在震荡市失效 ====================


def efficiency_ratio(close: pd.Series) -> float:
    """ER = |期末 - 期初| / Σ|每日涨跌|：1 = 单边直线，0 = 原地打转"""
    path = close.diff().abs().sum()
    return abs(close.iloc[-1] - close.iloc[0]) / path if path > 0 else 0.0


def trend_vs_range(close: pd.Series, ret: pd.Series, bt_ls: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：学习检查点 —— 趋势策略为什么在震荡市失效")
    print("=" * 60)

    print("""
    价格 │      ╱╲    ╱╲    ╱╲        震荡市：每次金叉都在区间上沿买入，
         │ ────╱──╲──╱──╲──╱──╲────   每次死叉都在区间下沿卖出
         │    ╱    ╲╱    ╲╱    ╲      → 系统性地"高买低卖" + 每次都付手续费
         └──────────────────────────▶
    价格 │                    ╱       趋势市：一次信号吃到整段
         │               ╱╲╱
         │     ───╱╲╱╱                少数大赚覆盖所有小亏
         └──────────────────────────▶
    用每年的效率系数 ER 衡量"趋势程度"，看策略表现与 ER 的关系：
    """)

    years = sorted(set(close.index.year))[1:]  # 第一年有均线预热期，跳过
    rows = []
    for y in years:
        c = close[close.index.year == y]
        r = bt_ls["ret"][bt_ls.index.year == y]
        rows.append({
            "年份": y,
            "指数涨跌": c.iloc[-1] / c.iloc[0] - 1,
            "ER": efficiency_ratio(c),
            "策略收益": (1 + r).prod() - 1,
            "调仓次数": int((bt_ls["pos"][bt_ls.index.year == y].diff().abs() > 0).sum()),
        })
    yearly = pd.DataFrame(rows).set_index("年份")
    print(yearly.to_string(formatters={"指数涨跌": "{:+.1%}".format, "ER": "{:.3f}".format,
                                       "策略收益": "{:+.1%}".format}))

    corr = yearly["ER"].corr(yearly["策略收益"])
    corr_turn = yearly["调仓次数"].corr(yearly["策略收益"])
    hi = yearly[yearly["ER"] >= yearly["ER"].median()]["策略收益"].mean()
    lo = yearly[yearly["ER"] < yearly["ER"].median()]["策略收益"].mean()
    print(f"""
    📊 年度 ER 与策略收益的相关系数：{corr:+.2f}
       年度调仓次数与策略收益的相关系数：{corr_turn:+.2f}
       ER 高于中位数的年份，策略平均收益 {hi:+.1%}；低于中位数的年份 {lo:+.1%}
       （年份样本只有 {len(yearly)} 个，相关系数只能作为方向性参考）

    💡 结论（对照上表自行验证）：
       1. 震荡年份 ER 低，均线反复交叉 → 调仓多、频繁止损、成本累积
       2. 趋势年份 ER 高，一次信号可以持有很久
       3. 趋势跟踪像"买期权"：平时持续支付小额"保险费"，换取大行情中的大赚
    ⚠️ 年度 ER 只看首尾两点：一年里先大涨后大跌（如 2015 年）也会得到很低的 ER，
       但期间其实有两段大趋势 —— 读表时要结合价格走势，不要机械套用。
    """)
    return yearly


# ==================== 第五部分：参数敏感性预览与图表 ====================


def param_preview_and_plot(close: pd.Series, ret: pd.Series, bt_lo: pd.DataFrame, bt_ls: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：参数敏感性预览与图表")
    print("=" * 60)

    fasts, slows = [5, 10, 20, 40], [60, 120, 250]
    grid = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    for f in fasts:
        for s in slows:
            grid.loc[f, s] = perf_stats(backtest(ma_cross_target(close, f, s), ret, COST)["ret"])["Sharpe"]
    grid.index.name = "快线\\慢线"
    print("📊 多空双均线 Sharpe（扣成本）：")
    print(grid.round(2).to_string())
    print(f"""
    最好 {grid.max().max():.2f}，最差 {grid.min().min():.2f}，中位数 {np.nanmedian(grid.values):.2f}
    💡 不要只挑最好的那一格：08 课会系统讲"参数平原"，09 课讲稳健性检验。
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(close.index, close, lw=0.8, color="gray", label="沪深300")
    ax.plot(close.index, close.rolling(20).mean(), lw=0.8, label="MA20")
    ax.plot(close.index, close.rolling(60).mean(), lw=0.8, label="MA60")
    ax.set_title("沪深300 与 20/60 日均线")
    ax.legend()
    ax = axes[1]
    ax.plot(close.index, (1 + ret).cumprod(), color="gray", label="买入持有")
    ax.plot(bt_lo.index, bt_lo["equity"], label="双均线-只做多")
    ax.plot(bt_ls.index, bt_ls["equity"], label="双均线-多空")
    ax.set_title(f"净值对比（单边成本 {COST:.2%}）")
    ax.legend()
    path = save_figure(fig, "01_ma_crossover.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    close = load_close()
    moving_average_basics(close)
    ret, bt_lo, bt_ls = ma_cross_backtest(close)
    return_structure(ret, bt_ls)
    trend_vs_range(close, ret, bt_ls)
    param_preview_and_plot(close, ret, bt_lo, bt_ls)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 SMA 换成 EMA（close.ewm(span=N, adjust=False).mean()），比较 Sharpe 与调仓次数
    2. 把单边成本改为 0.1% 和 0.2%，多空版本的 Sharpe 如何变化？成本提高到多少时策略不再优于买入持有？
    3. 在第四部分的表格中挑出策略亏损最多的一年，画出当年指数与两条均线，数一数假信号
    4. 把标的换成中证 1000（000852），结论是否改变？
    """)


if __name__ == "__main__":
    main()
