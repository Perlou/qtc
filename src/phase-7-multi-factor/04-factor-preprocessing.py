"""
因子预处理：去极值、标准化、行业市值中性化
==========================================

学习目标：
    1. 掌握 MAD 去极值，理解为什么不用"均值 ± 3σ"
    2. 掌握 Z-score 与排序标准化，避开"全样本标准化"的未来函数
    3. 实现行业 + 市值中性化（截面回归取残差），并量化中性化前后因子与市值的相关性
    4. 理解预处理对 Pearson IC 与 Rank IC 的不同影响

核心概念：
    - MAD（绝对中位差）= median(|x − median(x)|)，1.4826 × MAD ≈ 正态分布的 σ
    - Z-score：(x − 截面均值) / 截面标准差，每个截面单独计算
    - 中性化：f = Σ γ_k·行业哑变量 + δ·ln(市值) + ε，取残差 ε 作为新因子
    - 顺序：去极值 → 标准化 → 中性化 → 再标准化，全部在"同一天的截面"上做

前置知识：
    - 03-factor-construction.py
    - Phase 3 的 06-linear-regression-ols.py（回归残差）

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
from factor_tools import (FACTOR_LABELS, build_factors, calc_ic, ic_summary, neutralize, period_returns,
                          preprocess, rebalance_dates, tradable_mask, winsorize_mad, zscore)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


# ==================== 第一部分：预处理流程 ====================


def pipeline_overview():
    print("=" * 60)
    print("第一部分：为什么要预处理")
    print("=" * 60)
    print("""
    原始因子的三个问题：
       1. 极端值：某只股票换手率是别人的 10 倍，会主导回归和权重
       2. 量纲不同：EP 在 0.0x 量级，ln(市值) 在 20 多，无法直接加总
       3. 混杂暴露：高 EP 股票集中在银行、建筑 —— 因子收益可能只是"买了银行股"

    ┌──────────────────────────────────────────────────────────┐
    │ 每个截面独立执行（绝不跨时间）                           │
    ├──────────────────────────────────────────────────────────┤
    │ ① 去极值（MAD 截断）        防止极端值主导               │
    │ ② 缺失值处理（本课：剔除）  也可用行业中位数填充         │
    │ ③ 标准化（Z-score）         统一量纲                     │
    │ ④ 中性化（行业哑变量 + ln 市值回归，取残差）             │
    │ ⑤ 再次标准化                残差的尺度归一               │
    └──────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：MAD 去极值 ====================


def mad_demo(panel, dates, raw):
    print("\n" + "=" * 60)
    print("第二部分：MAD 去极值")
    print("=" * 60)

    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, 100_000)
    mad = np.median(np.abs(z - np.median(z)))
    print(f"""
    MAD = median(|x − median(x)|)，截断区间 = 中位数 ± n × 1.4826 × MAD（通常 n = 3）
    常数 1.4826 的来历：标准差为 2 的正态样本，MAD = {mad:.4f}，1.4826 × MAD = {1.4826 * mad:.4f} ≈ σ
    """)

    # 被污染的尺子：一个极端值把均值和标准差都拉偏
    x = pd.Series(np.r_[rng.normal(0, 1, 99), 50.0])
    lo, hi = x.mean() - 3 * x.std(), x.mean() + 3 * x.std()
    print(f"""    🧪 99 个标准正态数 + 1 个极端值 50：
       均值 ± 3σ 区间：[{lo:.2f}, {hi:.2f}]  → 极端值 50 被压到 {min(50, hi):.2f}，仍是其他值的数倍
       MAD 区间：      [{winsorize_mad(x).min():.2f}, {winsorize_mad(x).max():.2f}]  → 中位数和 MAD 几乎不受极端值影响
    """)

    d = dates[-1]
    turn = -raw["lowturn"].loc[d].dropna()  # 还原为"换手率"本身，便于理解
    w = winsorize_mad(turn)
    m, s = turn.mean(), turn.std()
    sig = turn.clip(m - 3 * s, m + 3 * s)
    df = pd.DataFrame({
        "原始": [turn.mean(), turn.median(), turn.max(), turn.skew(), 0],
        "均值±3σ 截断": [sig.mean(), sig.median(), sig.max(), sig.skew(), int((turn != sig).sum())],
        "MAD 截断": [w.mean(), w.median(), w.max(), w.skew(), int((turn != w).sum())],
    }, index=["均值", "中位数", "最大值", "偏度", "被截断个数"])
    print(f"📋 {d.date()} 20 日平均换手率（{len(turn)} 只股票）：")
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    print("""
    💡 去极值是"截断"（winsorize）而不是"删除"：极端股票仍保留在样本里，只是不再主导结果。
       换手率这类右偏严重的因子，也常先取对数再去极值。
    """)
    return turn, w


# ==================== 第三部分：标准化 ====================


def standardize_demo(panel, dates, raw, mask):
    print("\n" + "=" * 60)
    print("第三部分：标准化 —— 必须按截面做")
    print("=" * 60)

    d = dates[-1]
    ep = winsorize_mad(raw["ep"].loc[d].where(mask.loc[d]).dropna())
    z = zscore(ep)
    rk = ep.rank(pct=True) - 0.5
    print(f"""
    Z-score：z = (x − 截面均值) / 截面标准差 → 均值 {z.mean():+.3f}，标准差 {z.std():.3f}
    排序标准化：x.rank(pct=True) − 0.5 → 范围 [{rk.min():+.3f}, {rk.max():+.3f}]，对分布形状完全不敏感
    两者的相关系数：{z.corr(rk):.3f}（排序丢掉了"差距大小"的信息，换来稳健性）
    """)

    # 陷阱：全样本（跨时间）标准化
    size = raw["size"].where(mask)
    pooled = (size - size.stack().mean()) / size.stack().std()
    cross = size.sub(size.mean(axis=1), axis=0).div(size.std(axis=1), axis=0)
    by_year = pd.DataFrame({
        "全样本 z 的年均值": pooled.mean(axis=1).groupby(pooled.index.year).mean(),
        "截面 z 的年均值": cross.mean(axis=1).groupby(cross.index.year).mean(),
    })
    print("📋 小市值因子 −ln(市值)：两种标准化的年度均值")
    print(by_year.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    ❌ 全样本标准化用到了整个样本期的均值和标准差（包含未来），而且市场整体市值随时间变化，
       导致某些年份所有股票的因子值整体偏高或偏低 —— 因子值混入了"时间趋势"
    ✅ 截面标准化每期均值都是 0：只保留"同一时刻谁比谁大"，这正是横截面选股需要的信息
    """)


# ==================== 第四部分：行业市值中性化 ====================


def neutralization_demo(panel, dates, raw, mask):
    print("\n" + "=" * 60)
    print("第四部分：行业市值中性化")
    print("=" * 60)
    print("""
    每个截面回归：f_i = Σ γ_k·D_i,k + δ·ln(MV_i) + ε_i，取残差 ε 作为中性化后的因子
       D_i,k：行业哑变量（全部 11 个行业，不加常数项，避免共线）
       残差与行业、ln(市值) 都不相关 —— "在同行业、同规模的股票之间比较"

    🎓 比喻：评价学习方法要比较同一个班、同样基础的学生；直接拿重点班和普通班比，
       测到的是"分班效应"而不是"学习方法效应"。

        X = pd.get_dummies(sector, dtype=float); X["lnmv"] = np.log(mcap)
        beta = np.linalg.lstsq(X, f)[0];  resid = f − X @ beta
    """)

    lnmv = np.log(panel.mcap.reindex(dates))
    sector = panel.info["sector"]
    rows = {}
    for k in [k for k in raw if k != "size"]:
        c_before, c_after, r2 = [], [], []
        for d in dates:
            x = raw[k].loc[d].where(mask.loc[d]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < 30:
                continue
            z = zscore(winsorize_mad(x))
            e = neutralize(z, sector.reindex(z.index), lnmv.loc[d, z.index]).dropna()
            c_before.append(z.corr(lnmv.loc[d, z.index]))
            c_after.append(e.corr(lnmv.loc[d, e.index]))
            r2.append(1 - e.var() / z.loc[e.index].var())
        rows[FACTOR_LABELS[k]] = {"与 ln市值 相关(前)": np.mean(c_before), "与 ln市值 相关(后)": np.mean(c_after),
                                  "行业+市值 解释的方差 R²": np.mean(r2)}
    print("📋 中性化前后（各期平均）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.3f}"))

    before = preprocess(raw["bp"], panel, mask, neutral=False)
    after = preprocess(raw["bp"], panel, mask, neutral=True)
    sec_tab = pd.DataFrame({
        "中性化前": before.stack().groupby(level=1).mean().groupby(sector).mean(),
        "中性化后": after.stack().groupby(level=1).mean().groupby(sector).mean(),
    }).sort_values("中性化前")
    print("\n📋 BP 因子 Z-score 的行业均值（全样本平均）：")
    print(sec_tab.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    💡 解读：
       - 中性化后与 ln(市值) 的相关系数降到 0 附近（线性回归残差的性质）
       - 行业均值被拉到 0：BP 不再等于"押注金融 / 地产"
       - R² 显示原始因子有多大比例只是行业和市值的影子
       ⚠️ 中性化不是越多越好：如果你的 alpha 本来就包含"行业间估值差异"，完全行业中性会丢掉它；
          线性回归也只能去掉线性的市值关系（Barra 另有"非线性市值"因子，10 课）
    """)
    return sec_tab


# ==================== 第五部分：预处理对 IC 的影响 ====================


def ic_impact(panel, dates, raw, mask):
    print("\n" + "=" * 60)
    print("第五部分：预处理对 IC 的影响")
    print("=" * 60)

    ret = period_returns(panel, dates)
    rows = {}
    for k in raw:
        r0 = raw[k].where(mask).replace([np.inf, -np.inf], np.nan)
        r1 = preprocess(raw[k], panel, mask, neutral=False)
        r2 = preprocess(raw[k], panel, mask, neutral=True, size_neutral=(k != "size"))
        rows[FACTOR_LABELS[k]] = {
            "Pearson IC 原始": calc_ic(r0, ret, "pearson").mean(),
            "Pearson IC 去极值+Z": calc_ic(r1, ret, "pearson").mean(),
            "Rank IC 原始": calc_ic(r0, ret).mean(),
            "Rank IC 去极值+Z": calc_ic(r1, ret).mean(),
            "Rank IC 中性化": calc_ic(r2, ret).mean(),
            "t 中性化": ic_summary(calc_ic(r2, ret))["t值"],
        }
    print("📋 各因子月度 IC 均值（可交易股票，持有期 = 到下一调仓日）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("""
    💡 读表要点：
       - 去极值 + Z-score 是单调变换：Rank IC 几乎不变（只有被截断的并列值略有影响），
         Pearson IC 却可能明显变化 —— 这就是 Rank IC 更常用的原因之一
       - 中性化改变的是"你在测什么"：中性化后的 IC 衡量的是"同行业、同市值里的选股能力"
       - 规模因子只做行业中性（对自己做市值中性化只剩噪声）
       - 本股票池只有一百多只股票，|t| < 2 的结果在统计上都不显著，不要过度解读
    """)


# ==================== 主函数 ====================


def plot_results(turn, w, sec_tab):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(turn, bins=40, color="tab:gray")
    axes[0].set_title("原始 20 日换手率")
    axes[1].hist(zscore(w), bins=40, color="tab:blue")
    axes[1].set_title("MAD 去极值 + Z-score")
    sec_tab.plot.barh(ax=axes[2])
    axes[2].axvline(0, color="gray", lw=0.6)
    axes[2].set_title("BP 因子行业均值：中性化前后")
    path = save_figure(fig, "04_factor_preprocessing.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    raw = build_factors(panel, dates)

    pipeline_overview()
    turn, w = mad_demo(panel, dates, raw)
    standardize_demo(panel, dates, raw, mask)
    sec_tab = neutralization_demo(panel, dates, raw, mask)
    ic_impact(panel, dates, raw, mask)
    plot_results(turn, w, sec_tab)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 MAD 的 n 从 3 改为 5 和 2，统计每期被截断的股票数，观察 Pearson IC 的变化
    2. 实现"行业中位数填充缺失值"，与直接剔除比较因子覆盖率和 IC
    3. 用市值加权均值做中心化（Barra 的做法），验证"市值加权组合"的因子暴露为 0
    4. 先中性化、后去极值，比较残差的分布和 IC —— 为什么顺序很重要？
    """)


if __name__ == "__main__":
    main()
