"""
布林带均值回归
==============

学习目标：
    1. 理解布林带与 Z-score 的关系，以及"偏离 → 回归"的假设
    2. 实现带状态的均值回归策略（入场、出场、止损），并与买入持有对比
    3. 对比同一根布林带的两种相反用法：均值回归 vs 突破跟随
    4. 认识均值回归的致命风险"接飞刀"，学会加止损
    5. 用方差比（Variance Ratio）判断序列在某周期上偏趋势还是偏回归

核心概念：
    - 布林带：Mid = SMA(N)，Upper/Lower = Mid ± k·σ（σ 为过去 N 天收盘价标准差）
    - Z-score：z = (P - Mid) / σ，布林带上下轨就是 z = ±k
    - 方差比 VR(q) = Var(q 期累计收益) / (q · Var(单期收益))：>1 偏趋势，<1 偏回归

前置知识：
    - 01-moving-average-crossover.py
    - Phase 3 统计基础（标准差、正态分布、自相关）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存时读缓存，否则使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import backtest, get_continuous, perf_stats, stats_table, trade_stats
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20100101", "20251231"
COST = 0.0005  # 单边成本示例：5 个基点
N, K = 20, 2.0


def zscore(close: pd.Series, n: int = N) -> pd.Series:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - mid) / sd


# ==================== 第一部分：布林带与 Z-score ====================


def bollinger_basics(close: pd.Series) -> pd.Series:
    print("=" * 60)
    print("第一部分：布林带与 Z-score")
    print("=" * 60)

    print(f"""
    Mid   = SMA(N)            Upper = Mid + k·σ        Lower = Mid - k·σ
    z     = (P - Mid) / σ     → 上轨就是 z = +k，下轨就是 z = -k
    经典参数 N = {N}，k = {K}

    比喻：均线是橡皮筋的固定点，价格被拉得越远，回弹的力越大 ——
         前提是橡皮筋没断（"均值"本身没有移动）。
    """)

    z = zscore(close)
    zz = z.dropna()
    outside = (zz.abs() > K).mean()
    print(f"📊 沪深 300 日线，{N} 日布林带：")
    print(f"   |z| > {K} 的天数占比：{outside:.2%}（若服从正态分布约为 4.55%）")
    print(f"   z 的分位数：5% = {zz.quantile(0.05):+.2f}，50% = {zz.median():+.2f}，95% = {zz.quantile(0.95):+.2f}")
    print(f"   z 的最小 / 最大值：{zz.min():+.2f} / {zz.max():+.2f}")
    print("""
    💡 注意：只有 20 个样本的滚动标准差本身很不稳定，而且价格不是平稳序列，
       所以"跑出 2σ 就是小概率事件"只是一个粗略的直觉，不是严格的统计结论。
    """)
    return z


# ==================== 第二部分：均值回归策略 ====================


def bollinger_mr_target(z: pd.Series, k: float = K, exit_z: float = 0.0, stop_z: float | None = None,
                        max_days: int | None = None, stop_pct: float | None = None,
                        close: pd.Series | None = None, long_only: bool = False) -> pd.Series:
    """
    均值回归状态机（第 t 天收盘后决定目标仓位）：
        空仓时：z < -k 做多，z > k 做空
        持多时：z 回到 -exit_z 以上平仓；以下任一条件触发止损：
            z < -stop_z（z 止损）、持有满 max_days 天（时间止损）、
            收盘价较开仓日收盘价不利变动超过 stop_pct（价格止损，需要传入 close）
        止损后进入冷却：直到 z 回到 [-k, k] 之内才允许再次开仓（否则会立刻再次入场）
    """
    prices = close.reindex(z.index).to_numpy() if close is not None else np.full(len(z), np.nan)
    pos, cur, held, cooldown, entry = [], 0, 0, False, np.nan
    for zi, px in zip(z.to_numpy(), prices):
        if np.isnan(zi):
            pos.append(0)
            continue
        if cooldown and abs(zi) <= k:
            cooldown = False
        if cur == 0 and not cooldown:
            cur = 1 if zi < -k else (-1 if zi > k and not long_only else 0)
            held, entry = 0, px
        elif cur != 0:
            held += 1
            pnl = cur * (px / entry - 1) if np.isfinite(entry) else 0.0
            stop = ((stop_z is not None and cur * zi < -stop_z)
                    or (max_days is not None and held >= max_days)
                    or (stop_pct is not None and pnl < -stop_pct))
            if cur * zi >= -exit_z or stop:
                cur, cooldown = 0, stop
        pos.append(cur)
    return pd.Series(pos, index=z.index, dtype=float)


def ma_cross_target(close: pd.Series, fast: int = 20, slow: int = 60) -> pd.Series:
    ma_s = close.rolling(slow).mean()
    return np.sign(close.rolling(fast).mean() - ma_s).where(ma_s.notna(), 0.0)


def mean_reversion_backtest(close: pd.Series, z: pd.Series):
    print("\n" + "=" * 60)
    print("第二部分：布林带均值回归回测")
    print("=" * 60)

    print(f"""
    规则：z < -{K} 做多，z > +{K} 做空；z 回到 0（中轨）平仓。
    持仓状态依赖上一时刻（路径依赖）→ 用一个简单的 for 循环实现状态机。
    同样：第 t 天收盘后决定仓位，第 t+1 天才持有（backtest 内部 shift(1)），单边成本 {COST:.2%}。
    """)

    ret = close.pct_change().fillna(0)
    bt_ls = backtest(bollinger_mr_target(z), ret, COST)
    bt_lo = backtest(bollinger_mr_target(z, long_only=True), ret, COST)
    bt_ma = backtest(ma_cross_target(close), ret, COST)
    print(stats_table({
        "买入持有": perf_stats(ret),
        "布林回归-只做多": perf_stats(bt_lo["ret"], bt_lo["turnover"], bt_lo["pos"]),
        "布林回归-多空": perf_stats(bt_ls["ret"], bt_ls["turnover"], bt_ls["pos"]),
        "双均线-多空(对照)": perf_stats(bt_ma["ret"], bt_ma["turnover"], bt_ma["pos"]),
    }).to_string())

    ts_mr, ts_ma = trade_stats(bt_ls["pos"], ret), trade_stats(bt_ma["pos"], ret)
    cmp = pd.DataFrame({
        "布林回归(多空)": [ts_mr["笔数"], ts_mr["胜率"], ts_mr["盈亏比"], ts_mr["单笔偏度"], ts_mr["平均持有天数"]],
        "双均线(多空)": [ts_ma["笔数"], ts_ma["胜率"], ts_ma["盈亏比"], ts_ma["单笔偏度"], ts_ma["平均持有天数"]],
    }, index=["笔数", "胜率", "盈亏比", "单笔偏度", "平均持有天数"])
    print("\n📊 逐笔交易对比：")
    print(cmp.T.to_string(formatters={"笔数": "{:.0f}".format, "胜率": "{:.1%}".format, "盈亏比": "{:.2f}".format,
                                      "单笔偏度": "{:+.2f}".format, "平均持有天数": "{:.1f}".format}))
    print("""
    💡 对照 CONCEPT.md 第 1.2 节的表格看实际数字：
       均值回归的典型画像是"胜率高、盈亏比低、负偏（左尾长）"，趋势跟踪正好相反。
       具体到这段沪深 300 数据是否符合，请以上表为准。
    """)
    return ret, bt_ls, ts_mr


# ==================== 第三部分：同一根布林带，两种相反用法 ====================


def breakout_vs_reversion(close: pd.Series, z: pd.Series, ret: pd.Series, bt_mr: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：同一根布林带的两种用法")
    print("=" * 60)

    print("""
    ┌──────────┬──────────────────────────┬──────────────────────┐
    │ 用法     │ 规则                     │ 假设                 │
    ├──────────┼──────────────────────────┼──────────────────────┤
    │ 均值回归 │ 触上轨做空、触下轨做多   │ 偏离是噪声，会回来   │
    │ 突破跟随 │ 突破上轨做多、跌破下轨做空│ 偏离是新信息，会延续 │
    └──────────┴──────────────────────────┴──────────────────────┘
    指标本身不含观点，规则才含观点。哪种有效，要用数据验证。
    """)

    # 突破跟随：方向与均值回归相反，同样在 z 回到 0 时平仓
    bt_bo = backtest(-bollinger_mr_target(z), ret, COST)
    results = {"均值回归": perf_stats(bt_mr["ret"], bt_mr["turnover"], bt_mr["pos"]),
               "突破跟随": perf_stats(bt_bo["ret"], bt_bo["turnover"], bt_bo["pos"])}
    print(stats_table(results).to_string())
    gross_sum = bt_mr["gross"].sum() + bt_bo["gross"].sum()
    better = max(results, key=lambda k: results[k]["Sharpe"])
    print(f"""
    📌 两个策略的仓位恰好相反，所以未扣成本的日收益之和 ≈ 0（这里为 {gross_sum:+.2e}），
       而扣成本后两者都要付费。在这段沪深 300 数据上，Sharpe 更高的是「{better}」。
       可以和第五部分的方差比对照：一般而言 VR > 1（偏趋势）时突破跟随更可能占优，VR < 1 时均值回归更可能占优。
       如果一个规则"正着做"和"反着做"都不赚钱，说明它在这个标的、这个周期上没有可交易的信息。
    """)


# ==================== 第四部分：接飞刀与止损 ====================


def falling_knife(close: pd.Series, z: pd.Series, ret: pd.Series, bt_mr: pd.DataFrame, ts_mr: dict):
    print("\n" + "=" * 60)
    print("第四部分：致命陷阱 —— 单边趋势中\"接飞刀\"")
    print("=" * 60)

    trades = ts_mr["_trades"]
    worst_day = trades.idxmin()
    seg = bt_mr["pos"].loc[worst_day:]
    end_day = seg[seg != seg.iloc[0]].index[0] if (seg != seg.iloc[0]).any() else seg.index[-1]
    print(f"""
    价格单边运行时（持续下跌时拿着多头，或持续上涨时拿着空头），z 迟迟回不到 0，
    均值回归策略一直扛着亏损仓位，亏损不断扩大。
    📉 本次回测最差的一笔：{worst_day.date()} 开仓，{end_day.date()} 平仓，
       方向 {'做多' if seg.iloc[0] > 0 else '做空'}，收益 {trades.min():+.1%}，
       期间 z 的最小 / 最大值 {z.loc[worst_day:end_day].min():+.2f} / {z.loc[worst_day:end_day].max():+.2f}
    """)

    variants = {
        "无止损": bollinger_mr_target(z),
        "z止损(|z|>3)": bollinger_mr_target(z, stop_z=3.0),
        "时间止损(20天)": bollinger_mr_target(z, max_days=20),
        "价格止损(5%)": bollinger_mr_target(z, stop_pct=0.05, close=close),
    }
    rows, worst = {}, {}
    for name, tgt in variants.items():
        bt = backtest(tgt, ret, COST)
        rows[name] = perf_stats(bt["ret"], bt["turnover"], bt["pos"])
        worst[name] = trade_stats(bt["pos"], ret)["_trades"].min()
    table = stats_table(rows)
    table.loc["最差单笔"] = [f"{worst[k]:+.1%}" for k in table.columns]
    print(table.to_string())
    z_max_abs = z.loc[worst_day:end_day].abs().max()
    if z_max_abs < 3.0:
        print(f"""
    🔍 最差那笔交易期间 |z| 最大只有 {z_max_abs:.2f}，z 止损根本没有触发：
       价格单边运行时，滚动标准差 σ 也同步放大，z 反而不会太极端。
       用 z 定义止损有盲区；按价格（或按 ATR）设置止损才能直接限制单笔亏损。""")
    print(f"""
    💡 止损的目的不是提高收益，而是截断左尾 —— 对比上表的"最差单笔"：
       无止损 {worst['无止损']:+.1%}，价格止损 {worst['价格止损(5%)']:+.1%}
       （价格止损的最差单笔可能超过 5%：收盘才判断、次日才执行，中间还可能跳空）。
       但止损也会把一部分"本来会回归"的交易提前砍掉，并增加换手，总收益不一定改善。
       Phase 6 会把均值回归的对象换成"有经济学理由保证平稳的价差"，从根源上降低接飞刀的风险。
    """)


# ==================== 第五部分：方差比诊断 ====================


def variance_ratio(ret: pd.Series, q: int) -> float:
    r = ret.dropna()
    rq = r.rolling(q).sum().dropna()
    return float(rq.var() / (q * r.var()))


def variance_ratio_diagnosis(close: pd.Series, z: pd.Series, bt_mr: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：用方差比判断\"趋势还是回归\"")
    print("=" * 60)

    print("""
    VR(q) = Var(q 天累计收益) / (q × Var(日收益))
      VR ≈ 1：随机游走      VR > 1：正自相关，偏趋势      VR < 1：负自相关，偏回归
    """)

    series = {"沪深300": close.pct_change()}
    try:
        series["中证500"] = get_index_daily("000905", START, END)["close"].pct_change()
    except Exception as e:  # 数据不可用时跳过
        print(f"   ⏭️ 中证500 获取失败（{type(e).__name__}），跳过")
    rb = get_continuous("RB")
    series["螺纹钢(比例调整连续)"] = rb["close"].pct_change()

    qs = [2, 5, 10, 20, 60]
    vr = pd.DataFrame({name: [variance_ratio(r, q) for q in qs] for name, r in series.items()},
                      index=[f"q={q}" for q in qs]).T
    print(vr.round(3).to_string())
    print("""
    💡 这只是诊断工具：VR 偏离 1 多少才算"显著"需要做检验（Lo-MacKinlay 检验），
       而且样本内的 VR 不代表样本外可交易。但它能避免在明显随机游走的序列上硬套策略。
       另外注意：不同周期（q）上的结论可以不同 —— "趋势还是回归"是周期 + 品种 + 时代的属性。
    """)

    setup_chinese_font()
    recent = close.index[-500:]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    mid, sd = close.rolling(N).mean(), close.rolling(N).std()
    ax = axes[0]
    ax.plot(recent, close.loc[recent], color="black", lw=0.8, label="沪深300")
    ax.plot(recent, mid.loc[recent], lw=0.8, label="中轨")
    ax.fill_between(recent, (mid - K * sd).loc[recent], (mid + K * sd).loc[recent], alpha=0.2, label=f"±{K}σ")
    ax.set_title("布林带（最近 500 个交易日）")
    ax.legend()
    axes[1].plot(recent, z.loc[recent], lw=0.8)
    for lvl in (-K, 0, K):
        axes[1].axhline(lvl, color="red" if lvl else "gray", ls="--", lw=0.8)
    axes[1].set_title("Z-score")
    axes[2].plot(bt_mr.index, bt_mr["equity"], label="布林回归-多空")
    axes[2].plot(close.index, close / close.iloc[0], color="gray", label="买入持有")
    axes[2].set_title("全样本净值")
    axes[2].legend()
    fig.tight_layout()
    path = save_figure(fig, "02_bollinger_mr.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    close = get_index_daily("000300", START, END)["close"]
    z = bollinger_basics(close)
    ret, bt_mr, ts_mr = mean_reversion_backtest(close, z)
    breakout_vs_reversion(close, z, ret, bt_mr)
    falling_knife(close, z, ret, bt_mr, ts_mr)
    variance_ratio_diagnosis(close, z, bt_mr)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 k 改为 1.5 和 2.5，出场改为 exit_z = 0.5，比较胜率、盈亏比和交易次数的变化
    2. 只在 VR(20) < 1 的品种上做均值回归、VR(20) > 1 的品种上做突破跟随，这样的"选择"有什么前视风险？
    3. 把均值回归改到 60 分钟或 5 日周期（重采样），观察 VR 和策略表现是否变化
    4. 给"突破跟随"版本加上第四部分的止损，它还需要止损吗？为什么？
    """)


if __name__ == "__main__":
    main()
