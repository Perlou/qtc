"""
因子拥挤与衰减
==============

学习目标：
    1. 理解因子衰减的两个来源：数据挖掘与套利
    2. 用滚动 IC、前后半段对比监控因子是否在失效
    3. 计算估值价差、组内相关性、换手率比、因子波动率等拥挤度指标
    4. 结合小盘股风格切换，理解"隐藏的小市值暴露"为什么危险

核心概念：
    - 因子衰减：样本外、论文发表后收益下降（McLean & Pontiff, 2016）
    - 多重检验：测试的因子越多，t 值门槛应越高（Harvey, Liu, Zhu, 2016）
    - 拥挤（Crowding）：太多资金做同样的交易 → 预期收益被压缩、回撤时踩踏
    - 估值价差：多头组合与空头组合的估值之比，是最常用的拥挤度指标

前置知识：
    - 05-ic-ir-analysis.py、06-quantile-backtest.py
    - Phase 6 的统计套利（2007 Quant Quake）

环境要求：
    - pip install pandas numpy matplotlib
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from factor_data import load_panel
from factor_tools import FACTOR_LABELS, calc_ic, quantile_labels, quantile_returns, research_inputs
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

WATCH = ["size", "rev_1m", "lowturn", "ep"]  # 重点监控的因子
WINDOW = 12  # 滚动 IC 窗口（月）


# ==================== 第一部分：因子为什么会衰减 ====================


def decay_theory(ics: pd.DataFrame):
    print("=" * 60)
    print("第一部分：因子为什么会衰减")
    print("=" * 60)
    t = ics.mean() / ics.std() * np.sqrt(ics.notna().sum())
    print(f"""
    McLean & Pontiff（2016）研究了近百个已发表的异象：
       样本外收益比样本内平均低约 26%，论文发表后平均低约 58%

    两个原因：
       1. 数据挖掘：部分因子本身就是过拟合的产物，样本外自然失效
          Harvey、Liu、Zhu（2016）认为考虑已测试的因子数量，t 值门槛应提高到 3.0 左右
       2. 套利：因子被广泛知晓后，资金涌入，定价偏差被交易掉

    本阶段测试的 {len(t)} 个因子中：|t| > 2 有 {(t.abs() > 2).sum()} 个，|t| > 3 有 {(t.abs() > 3).sum()} 个。
    真实研究要记录测试过的所有因子数量，而不是只报告最好的那个。

    区分来源的意义：风险补偿型因子的回撤可能是"正常的坏时候"，行为型因子的持续低迷可能是永久衰减。
    """)


# ==================== 第二部分：滚动 IC 监控 ====================


def rolling_ic_monitor(ic: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    """滚动 IC 均值与 ICIR：持续下降是衰减信号"""
    mean, std = ic.rolling(window).mean(), ic.rolling(window).std()
    return pd.DataFrame({"rolling_ic": mean, "rolling_icir": mean / std})


def ic_monitoring(ics: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：滚动 IC 监控")
    print("=" * 60)
    half = ics.index[len(ics) // 2]
    rows = {}
    for k in ics.columns:
        a, b = ics.loc[:half, k].dropna(), ics.loc[half:, k].iloc[1:].dropna()
        m = rolling_ic_monitor(ics[k])
        pooled = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows[FACTOR_LABELS[k]] = {"前半段 IC": a.mean(), "后半段 IC": b.mean(), "差异 t 值": (b.mean() - a.mean()) / pooled,
                                  "滚动 ICIR 最新": m["rolling_icir"].iloc[-1], "最低": m["rolling_icir"].min(),
                                  "最高": m["rolling_icir"].max()}
    print(f"📋 以 {half.date()} 为界，前后两段 Rank IC 对比，以及 {WINDOW} 个月滚动 ICIR：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"""
    💡 解读：
       - 滚动 ICIR 在正负之间大幅摆动是常态：{WINDOW} 个月只有 {WINDOW} 个样本，估计误差很大
       - 判断衰减要看"持续下降"，而不是某一段转负；差异 t 值帮助判断前后变化是否超出噪声
       - 单靠 IC 下降无法区分"衰减"和"正常的低谷"，要结合下面的拥挤度指标和经济逻辑
    """)


# ==================== 第三部分：估值价差 ====================


def valuation_spread(factor: pd.DataFrame, bp: pd.DataFrame, q: int = 5) -> pd.Series:
    """每期：因子最高组与最低组的 BP 中位数之比（比值低 = 多头相对空头更贵）"""
    out = {}
    for d in factor.index:
        f = factor.loc[d].dropna()
        x = bp.loc[d, f.index]
        if len(f) < 30:
            continue
        lab = quantile_labels(f, q)
        lo = x[lab == 1].median()
        if lo > 0:
            out[d] = x[lab == q].median() / lo
    return pd.Series(out)


def expanding_percentile(s: pd.Series) -> pd.Series:
    """当前值在历史（含当期，不含未来）中的分位数"""
    return s.expanding(min_periods=12).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def spread_analysis(clean, raw) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：估值价差")
    print("=" * 60)
    print("""
    估值价差 = 多头组合 BP 中位数 / 空头组合 BP 中位数
       BP 越低越贵。比值持续走低 → 多头相对空头越来越贵 → 可能有资金拥挤进多头
       对价值因子本身比值恒大于 1，要看它相对自身历史的位置
       用 expanding 分位数只和过去比较，避免全样本分位数引入未来信息
    """)
    spreads, rows = {}, {}
    for k in WATCH:
        s = valuation_spread(clean[k], raw["bp"])
        p = expanding_percentile(s)
        spreads[k] = s
        rows[FACTOR_LABELS[k]] = {"均值": s.mean(), "最低": s.min(), "最高": s.max(), "最新": s.iloc[-1],
                                  "最新历史分位": p.iloc[-1]}
    df = pd.DataFrame(rows).T
    print("📋 各因子估值价差（最新分位越低 = 多头相对越贵）：")
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n   最新读数中多头相对自身历史最贵的是「{df['最新历史分位'].idxmin()}」，"
          f"分位 {df['最新历史分位'].min():.0%}（历史不到 6 年，分位数只是粗略参考）")
    return spreads


# ==================== 第四部分：其他拥挤度指标 ====================


def crowding_indicators(panel, dates, factor: pd.DataFrame) -> pd.DataFrame:
    """小市值因子的组内相关性、换手率比、因子波动率（按月）"""
    ret, turn = panel.daily_ret, panel.turnover
    rows, daily_ls = {}, []
    for i, d in enumerate(dates):
        f = factor.loc[d].dropna()
        if len(f) < 30:
            continue
        lab = quantile_labels(f, 5)
        top, bot = lab.index[lab == 5], lab.index[lab == 1]
        hist = ret.loc[:d].tail(60)
        c = hist[top].corr().to_numpy()
        pair_corr = c[np.triu_indices(len(top), 1)]
        ratio = turn.loc[:d].tail(20)[top].mean().median() / turn.loc[:d].tail(20)[bot].mean().median()
        rows[d] = {"组内相关性": np.nanmean(pair_corr), "换手率比": ratio}
        if i + 1 < len(dates):
            nxt = ret.loc[d:dates[i + 1]].iloc[1:]
            daily_ls.append(nxt[top].mean(axis=1) - nxt[bot].mean(axis=1))
    df = pd.DataFrame(rows).T
    ls = pd.concat(daily_ls)
    df["因子波动率"] = (ls.rolling(60).std() * np.sqrt(252)).reindex(df.index, method="ffill")
    return df, ls


def other_indicators(panel, dates, clean):
    print("\n" + "=" * 60)
    print("第四部分：其他拥挤度指标（以小市值因子为例）")
    print("=" * 60)
    print("""
    ┌────────────┬──────────────────────────────┬──────────────────────────┐
    │ 指标       │ 计算                         │ 拥挤信号                 │
    ├────────────┼──────────────────────────────┼──────────────────────────┤
    │ 组内相关性 │ 多头组内股票日收益的平均相关 │ 上升：资金在同步买卖     │
    │ 换手率比   │ 多头换手率 / 空头换手率      │ 上升：多头交易异常活跃   │
    │ 因子波动率 │ 多空日收益的滚动 60 日波动   │ 上升：因子开始剧烈摆动   │
    └────────────┴──────────────────────────────┴──────────────────────────┘
    """)
    ind, ls = crowding_indicators(panel, dates, clean["size"])
    pct = ind.apply(expanding_percentile)
    print("📋 各指标的最新值与历史分位：")
    print(pd.DataFrame({"均值": ind.mean(), "最新": ind.iloc[-1], "最新历史分位": pct.iloc[-1]})
          .to_string(float_format=lambda v: f"{v:.3f}"))
    return ind, pct, ls


# ==================== 第五部分：小盘风格切换 ====================


def small_cap_case(panel, dates, clean, ret, ind, pct, spreads):
    print("\n" + "=" * 60)
    print("第五部分：小盘股风格切换")
    print("=" * 60)
    print("""
    A 股的两个典型案例（小市值暴露是很多量化策略隐藏的主要收益和风险来源）：
       - 2017 年：IPO 提速削弱壳价值、监管趋严，大盘蓝筹显著跑赢，依赖小市值暴露的策略遭受重创
       - 2024 年初：大量量化产品在小微盘股上暴露集中，流动性冲击下集中踩踏，中性和指增产品普遍回撤
    面板从 2018 年开始，下面用中证 1000 / 沪深 300 比值观察 2024 年初这一段。
    """)
    if panel.is_synthetic:
        print("    （合成数据：指数和 2024 年初的回撤都是模拟的，只用来演示计算方法）")
    ratio = (panel.bench["zz1000_close"] / panel.bench["hs300_close"]).dropna()
    by_year = ratio.groupby(ratio.index.year).last().pct_change().dropna().loc[:2024]  # 2025 只有两个月，不计入
    print("   中证 1000 相对沪深 300 的年度超额（比值涨跌）：")
    print("   " + "，".join(f"{y} {v:+.1%}" for y, v in by_year.items()))

    win = ratio.loc["2023-07-01":"2024-06-30"]
    peak, trough = win.idxmax(), win.loc[win.idxmax():].idxmin()
    print(f"""
    2023-07 ~ 2024-06 的相对走势：
       比值高点 {peak.date()}，之后的低点 {trough.date()}，相对回撤 {win[trough] / win[peak] - 1:+.1%}
    """)

    grp = quantile_returns(clean["size"], ret, 5)
    case = grp.loc["2023-09-01":"2024-04-30", [5, 1, "多空"]].copy()
    case.columns = ["最小市值组", "最大市值组", "多空"]
    print("📋 本股票池小市值因子的月度收益（持有期从该调仓日的次日开盘起）：")
    print(case.to_string(float_format=lambda v: f"{v:+.2%}"))

    d = dates[dates <= pd.Timestamp("2023-12-31")][-1]
    sp = expanding_percentile(spreads["size"])
    reads = {"估值价差": sp.asof(d), "组内相关性": pct["组内相关性"].asof(d), "换手率比": pct["换手率比"].asof(d),
             "因子波动率": pct["因子波动率"].asof(d)}
    alarms = [k for k, v in reads.items() if (v < 0.2 if k == "估值价差" else v > 0.8)]
    print(f"""
    事前的拥挤度读数（{d.date()}，只用当时已知的数据）：
       """ + "，".join(f"{k} 历史分位 {v:.0%}" for k, v in reads.items()) + f"""
       （估值价差分位低于 20%、其余指标分位高于 80% 视为警报）
       发出警报的指标：{"、".join(alarms) if alarms else "无"}（{len(alarms)}/{len(reads)}）
    """)
    if not panel.is_synthetic:
        print("""    ⚠️ 本股票池只含沪深 300 / 中证 500 成分股，没有真正的微盘股，又有幸存者偏差，
       所以这里的指标和收益都反映不了微盘股当时的拥挤程度；中证 1000 的比值更能体现风格切换的幅度。
       要监控小微盘的拥挤，必须在对应的股票池上计算这些指标。
    """)
    print("""    💡 拥挤度指标不是精确的择时工具：即使在合适的股票池上，它们也只能提示风险在累积，
       无法告诉你何时发生，还可能完全没有事前信号 —— 所以不能只靠监控，
       事前的风格约束和仓位控制同样重要。多个信号同时出现时，应考虑降低权重、收紧风格约束。
       即使每个因子都做了市值中性化，选股后的组合仍可能偏向小盘（10 课的风险分解）。
    """)


# ==================== 第六部分：应对方法 ====================


def responses():
    print("\n" + "=" * 60)
    print("第六部分：应对衰减与拥挤")
    print("=" * 60)
    print("""
    ┌────────────┬────────────────────────────────────────────────┐
    │ 做法       │ 说明                                           │
    ├────────────┼────────────────────────────────────────────────┤
    │ 持续监控   │ 每月跟踪滚动 IC、ICIR、拥挤度指标              │
    │ 因子分散   │ 不依赖少数热门因子；持续研究新数据源           │
    │ 风格约束   │ 用风险模型控制 Size、Beta 等风格暴露           │
    │ 容量评估   │ 估算因子能容纳多少资金而不显著压缩收益         │
    │ 压力测试   │ 模拟"某因子单月回撤 3 倍标准差"时组合的损失    │
    │ 区分来源   │ 风险补偿型的回撤 vs 行为型的永久衰减           │
    └────────────┴────────────────────────────────────────────────┘
    """)


# ==================== 主函数 ====================


def plot_results(panel, ics, ls):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for k in WATCH:
        axes[0].plot(ics.index, ics[k].rolling(WINDOW).mean(), label=FACTOR_LABELS[k])
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_title(f"滚动 {WINDOW} 个月 Rank IC")
    axes[0].legend()
    ratio = (panel.bench["zz1000_close"] / panel.bench["hs300_close"]).dropna()
    axes[1].plot(ratio.index, ratio / ratio.iloc[0], label="中证1000 / 沪深300")
    axes[1].plot(ls.index, (1 + ls.fillna(0)).cumprod(), label="本股票池小市值多空")
    axes[1].axhline(1, color="gray", lw=0.6)
    axes[1].set_title("小盘风格走势")
    axes[1].legend()
    path = save_figure(fig, "11_factor_crowding.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    dates, mask, ret, raw, clean = research_inputs(panel)
    ics = pd.DataFrame({k: calc_ic(v, ret) for k, v in clean.items()})

    decay_theory(ics)
    ic_monitoring(ics)
    spreads = spread_analysis(clean, raw)
    ind, pct, ls = other_indicators(panel, dates, clean)
    small_cap_case(panel, dates, clean, ret, ind, pct, spreads)
    responses()
    plot_results(panel, ics, ls)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把估值价差的估值指标从 BP 换成 EP，比较各因子的读数是否一致
    2. 把几个拥挤度指标的分位数平均成一个综合拥挤度得分，检验它能否预测下一季度的因子收益
    3. 用滚动 ICIR 给因子动态降权（ICIR < 0 时权重减半），与 09 课的等权比较样本外表现
    4. 压力测试：假设小市值因子单月回撤 3 倍标准差，用 10 课的暴露估算组合损失
    """)


if __name__ == "__main__":
    main()
