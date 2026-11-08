"""
IC、Rank IC、IR 与 IC 衰减
==========================

学习目标：
    1. 理解 IC 的定义：因子值与"下一期"收益的截面相关系数
    2. 计算 Pearson IC、Rank IC、ICIR、t 值、IC>0 占比，并会读这些指标
    3. 计算 IC 衰减（不同持有期），理解它如何决定调仓频率
    4. 认识重叠样本对 t 值的影响（Newey-West 调整），以及主动管理基本定律

核心概念：
    - IC_t = corr(f_i,t, r_i,t→t+h)：每期一个数，形成 IC 时间序列
    - Rank IC：先排名再求相关（Spearman），对极端值稳健
    - ICIR = mean(IC) / std(IC)：IC 的"夏普比率"；t = ICIR × √T
    - IC 衰减：预测能力随持有期延长而下降的速度
    - 基本定律：IR ≈ IC × √Breadth

前置知识：
    - 04-factor-preprocessing.py
    - Phase 3 的 03-hypothesis-testing.py（t 检验）

环境要求：
    - pip install pandas numpy statsmodels matplotlib
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
from factor_tools import FACTOR_LABELS, calc_ic, forward_returns, ic_summary, nw_tstat, research_inputs
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

HORIZONS = (1, 5, 10, 20, 40, 60)


# ==================== 第一部分：IC 的定义 ====================


def ic_definition():
    print("=" * 60)
    print("第一部分：IC 是什么")
    print("=" * 60)
    print("""
    IC_t = corr( f_i,t , r_i,t→t+h )     相关系数在"同一时刻的所有股票之间"计算

       f_i,t：t 日收盘后算出的因子值（只用 t 及之前的信息）
       r_i,t→t+h：本课口径 = t+1 日开盘买入 → 持有 h 个交易日后开盘卖出的收益

    ┌──────────┬──────────────────────┬──────────────────────┐
    │          │ IC（Pearson）        │ Rank IC（Spearman）  │
    ├──────────┼──────────────────────┼──────────────────────┤
    │ 衡量     │ 线性关系             │ 单调关系             │
    │ 极端值   │ 敏感                 │ 稳健                 │
    │ 实践     │ 较少单独使用         │ 更常用               │
    └──────────┴──────────────────────┴──────────────────────┘

    🌦️ 比喻：天气预报员每天的预测与实际天气相关性只有 0.05 —— 听起来很低，
       但如果他每天都稳定地比瞎猜好一点点，在几千只股票上同时下注，优势就很可观。

    经验参考（A 股月频，仅供参考）：|Rank IC| 0.02–0.03 一般，0.03–0.05 较好，> 0.08 先查未来函数；
    ICIR（月频，未年化）0.3 一般、0.5 较好；IC>0 占比 55% 一般、60% 较好。
    """)


# ==================== 第二部分：IC 统计表 ====================


def ic_table(clean, ret):
    print("\n" + "=" * 60)
    print("第二部分：单因子 IC 统计（预处理 + 中性化后，月度调仓）")
    print("=" * 60)
    rows, ics = {}, {}
    for k, f in clean.items():
        ic_p, ic_r = calc_ic(f, ret, "pearson"), calc_ic(f, ret, "rank")
        s = ic_summary(ic_r)
        ics[k] = ic_r
        rows[FACTOR_LABELS[k]] = {"IC": ic_p.mean(), "Rank IC": s["IC均值"], "IC标准差": s["IC标准差"],
                                  "ICIR": s["ICIR"], "年化ICIR": s["ICIR"] * np.sqrt(12), "t值": s["t值"],
                                  "IC>0占比": s["IC>0占比"]}
    df = pd.DataFrame(rows).T.sort_values("t值", ascending=False)
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    n = int(ic_summary(next(iter(ics.values())))["期数"])
    best = df.index[0]
    print(f"""
    💡 解读（{n} 期）：
       - t 值 = ICIR × √T。|t| > 2 大致对应 5% 显著性；考虑到同时测试了 {len(df)} 个因子，
         更严格的门槛是 |t| > 3（11 课的多重检验）
       - 本表 t 值最高的是「{best}」（t = {df['t值'].iloc[0]:+.2f}）
       - IC 的标准差通常远大于均值：单期 IC 正负交替是常态，看的是"长期平均是否稳定为正"
    """)
    return pd.DataFrame(ics)


# ==================== 第三部分：IC 衰减 ====================


def ic_decay(panel, dates, clean, mask):
    print("\n" + "=" * 60)
    print("第三部分：IC 衰减 —— 因子能预测多远？")
    print("=" * 60)

    fwd = {h: forward_returns(panel, dates, h) for h in HORIZONS}
    rows, tstats = {}, {}
    for k, f in clean.items():
        rows[FACTOR_LABELS[k]] = {f"{h}日": calc_ic(f, fwd[h]).mean() for h in HORIZONS}
        ic60 = calc_ic(f, fwd[60])
        tstats[FACTOR_LABELS[k]] = {"60日 朴素t": ic_summary(ic60)["t值"], "60日 NW t": nw_tstat(ic60, lags=3)}
    decay = pd.DataFrame(rows).T
    print("📋 不同持有期的 Rank IC 均值（每月末取样）：")
    print(decay.to_string(float_format=lambda v: f"{v:+.4f}"))

    half = {}
    for name, row in decay.iterrows():
        peak = row.abs().idxmax()
        half[name] = peak
    print("\n   各因子 |Rank IC| 最大的持有期：" + "，".join(f"{k} {v}" for k, v in half.items()))
    print("\n📋 重叠样本问题：月末取样、持有 60 个交易日（约 3 个月）→ 相邻 IC 的收益窗口重叠约 2/3")
    print(pd.DataFrame(tstats).T.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("""
    💡 解读：
       - 经验上，短期反转衰减快（只能高频调仓，换手和成本会吃掉收益），价值、低波等衰减慢
         （适合低频调仓、容量更大）。对照上表看本面板是否符合 —— 小样本下单个持有期的 IC 噪声很大
       - 有些因子的 IC 随持有期变长反而上升（预测的是慢变量），这类因子月度甚至季度调仓就够了
       - 持有期 > 调仓间隔时 IC 序列自相关，朴素 t 值会被高估；Newey-West t 值更可信
       - 1 日、5 日的 IC 只在月末取样（72 个点），想精确研究短持有期应改为每日取样
    """)
    return decay


# ==================== 第四部分：IC 的稳定性 ====================


def ic_stability(ics: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：IC 的稳定性 —— 分年度看")
    print("=" * 60)
    by_year = ics.groupby(ics.index.year).mean()
    by_year.columns = [FACTOR_LABELS[c] for c in by_year.columns]
    print("📋 分年度 Rank IC 均值：")
    print(by_year.to_string(float_format=lambda v: f"{v:+.3f}"))
    pos_years = (by_year > 0).sum()
    print("\n   Rank IC 年均值为正的年数（共 {} 年）：".format(len(by_year))
          + "，".join(f"{k} {v}" for k, v in pos_years.items()))
    print("""
    💡 一个因子"全样本 IC 为正"可能只是某一两年贡献的。分年度、分市场状态（牛 / 熊 / 震荡）
       检验稳定性，是排除"偶然有效"的基本步骤。11 课会用滚动 IC 监控因子衰减。
    """)


# ==================== 第五部分：主动管理基本定律 ====================


def fundamental_law(ics: pd.DataFrame, mask: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：主动管理基本定律 IR ≈ IC × √Breadth")
    print("=" * 60)
    n = mask.sum(axis=1).mean()
    ic = ics.mean().abs().median()
    print(f"""
    Breadth（BR）：每年独立下注的次数
       本股票池：平均每期约 {n:.0f} 只可交易股票 × 12 次调仓 = {n * 12:.0f} 次"下注"
       各因子 |Rank IC| 的中位数：{ic:.4f}
       → 若下注相互独立，理论年化 IR ≈ {ic:.4f} × √{n * 12:.0f} = {ic * np.sqrt(n * 12):.2f}

       同样的 IC，全市场约 5000 只股票：{ic:.4f} × √{5000 * 12} = {ic * np.sqrt(5000 * 12):.2f}

    ⚠️ "独立"是很强的假设：同行业、同风格的股票高度相关，实际有效 BR 远小于"股票数 × 调仓次数"。
       但方向是对的：多因子选股就是"用宽度换精度"—— IC 很小，但下注次数很多。
    """)


# ==================== 主函数 ====================


def plot_results(decay: pd.DataFrame, ics: pd.DataFrame):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, row in decay.iterrows():
        axes[0].plot(HORIZONS, row.values, marker="o", label=name)
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("持有期（交易日）")
    axes[0].set_title("IC 衰减（Rank IC 均值）")
    axes[0].legend(fontsize=8, ncol=2)
    cum = ics.fillna(0).cumsum()
    for c in cum.columns:
        axes[1].plot(cum.index, cum[c], label=FACTOR_LABELS[c])
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_title("累计 Rank IC（斜率稳定向上 = 因子持续有效）")
    axes[1].legend(fontsize=8, ncol=2)
    path = save_figure(fig, "05_ic_analysis.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    dates, mask, ret, raw, clean = research_inputs(panel)
    ic_definition()
    ics = ic_table(clean, ret)
    decay = ic_decay(panel, dates, clean, mask)
    ic_stability(ics)
    fundamental_law(ics, mask)
    plot_results(decay, ics)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 改为每周五取样，重新计算 5 日持有期的 IC 与 ICIR，与月末取样比较
    2. 把收益换成"超额收益"（个股收益 − 行业平均收益）再算 IC，与中性化因子的 IC 比较
    3. 计算每个因子 IC 序列的一阶自相关，判断哪些因子"今天有效、下个月也有效"
    4. 构造一个纯随机因子（每期随机打分），重复 100 次，统计 |t| > 2 的比例，体会多重检验
    """)


if __name__ == "__main__":
    main()
