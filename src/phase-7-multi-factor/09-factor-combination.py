"""
因子合成：等权、IC 加权、最大化 IR
==================================

学习目标：
    1. 理解为什么要合成：多个低相关因子合成后 IC 更稳定
    2. 实现等权、IC 加权、ICIR 加权、最大化 IR（带协方差收缩）四种方法
    3. 用滚动的历史 IC 计算权重，弄清"t 日能用到哪一期的 IC"
    4. 量化用全样本 IC 或未实现 IC 算权重带来的虚假提升

核心概念：
    - 合成因子 F_i = Σ w_k·z_i,k（z 已标准化、方向统一）
    - 最大化 IR：max wᵀμ / √(wᵀΣw) ⟹ w* ∝ Σ⁻¹μ，μ、Σ 是各因子 IC 的均值和协方差
    - 收缩（Shrinkage）：Σ ← (1−λ)Σ + λ·diag(Σ)，降低协方差估计噪声
    - IC 的可得时间：t 期 IC 用到了 t 之后的收益，要等持有期结束才知道

前置知识：
    - 05-ic-ir-analysis.py、06-quantile-backtest.py

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
from factor_tools import (FACTOR_LABELS, calc_ic, ic_summary, perf_stats, quantile_returns, research_inputs,
                          zscore)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

LOOKBACK = 24  # 用过去 24 期 IC 估计权重
IC_LAG = 2  # t 日只能用到 t−2 期及更早的 IC（见第三部分）
METHODS = {"equal": "等权", "ic": "IC 加权", "icir": "ICIR 加权", "max_ir": "最大化 IR"}


# ==================== 第一部分：为什么要合成 ====================


def why_combine(clean, ics):
    print("=" * 60)
    print("第一部分：为什么要合成")
    print("=" * 60)
    print("""
    单个因子的 IC 通常只有 0.02–0.05，而且阶段性失效。多个低相关因子合成后更稳定：

       因子 A 的 IC：  ▲ ▲ ▼ ▲ ▼ ▼ ▲ ▲ ▲ ▼     时好时坏
       因子 B 的 IC：  ▼ ▲ ▲ ▲ ▲ ▼ ▼ ▲ ▲ ▲
       合成因子 IC：   ▲ ▲ ▲ ▲ ▲ ▼ ▲ ▲ ▲ ▲     分散后更稳定

    前提：每个因子都已预处理（标准化 + 中性化）且方向统一。
    """)
    names = list(clean)
    total, n = np.zeros((len(names), len(names))), 0
    for d in next(iter(clean.values())).index:
        df = pd.DataFrame({k: clean[k].loc[d] for k in names}).dropna()
        if len(df) >= 30:
            total += df.corr().to_numpy()
            n += 1
    corr = pd.DataFrame(total / n, index=names, columns=names)
    print("📋 中性化后因子的平均截面相关系数：")
    print(corr.round(2).to_string())
    print("\n📋 IC 序列之间的相关系数（因子是否同涨同跌）：")
    print(ics.corr().round(2).to_string())
    off = corr.to_numpy()[np.triu_indices(len(names), 1)]
    pair = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().dropna().sort_values(key=abs)
    print(f"""
    平均两两相关 {off.mean():+.2f}；相关性最高的一对：{pair.index[-1][0]} 与 {pair.index[-1][1]}（{pair.iloc[-1]:+.2f}）
    💡 相关性很高的两个因子，合成几乎没有分散效果，可以先合并成大类（如 EP、BP → 价值）。
    """)


# ==================== 第二部分：四种权重 ====================


def combine_weights(ic_hist: pd.DataFrame, method: str = "max_ir", shrink: float = 0.5) -> pd.Series:
    """ic_hist：过去 L 期各因子的 IC（行 = 期，列 = 因子），只能包含当前时刻已知的 IC"""
    mu = ic_hist.mean()
    if method == "equal":
        w = pd.Series(1.0, index=ic_hist.columns)
    elif method == "ic":
        w = mu
    elif method == "icir":
        w = mu / ic_hist.std()
    else:  # 最大化 IR：对协方差矩阵向对角线收缩，降低估计噪声
        cov = ic_hist.cov().to_numpy()
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        w = pd.Series(np.linalg.solve(cov, mu.to_numpy()), index=ic_hist.columns)
    w = w.clip(lower=0)  # 不允许负权重：方向已由经济逻辑确定，不反向押注噪声
    return w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)


def composite(clean: dict, weights: pd.DataFrame) -> pd.DataFrame:
    """weights：index = 调仓日，columns = 因子。缺失的因子值按 0（截面均值）处理"""
    names = list(weights.columns)
    out = {}
    for d in weights.index:
        z = pd.DataFrame({k: clean[k].loc[d] for k in names}).dropna(how="all")
        out[d] = zscore(z.fillna(0.0) @ weights.loc[d])
    return pd.DataFrame(out).T.reindex(columns=next(iter(clean.values())).columns)


def rolling_weights(ics: pd.DataFrame, method: str, lag: int = IC_LAG, shrink: float = 0.5) -> pd.DataFrame:
    rows = {}
    for i, d in enumerate(ics.index):
        end = i + 1 - lag  # 只取到第 i − lag 行（含）
        if end <= 0:  # ⚠️ 负数切片 iloc[:-1] 会取到"除最后一行外的全部"，把未来数据带进来
            continue
        hist = ics.iloc[:end].tail(LOOKBACK).dropna()
        if len(hist) >= LOOKBACK:
            rows[d] = combine_weights(hist, method, shrink)
    return pd.DataFrame(rows).T


def weights_overview(ics):
    print("\n" + "=" * 60)
    print("第二部分：四种合成方法")
    print("=" * 60)
    print(f"""
    ┌────────────┬─────────────────────────┬──────────────────────┬────────────────────┐
    │ 方法       │ 权重                    │ 优点                 │ 缺点               │
    ├────────────┼─────────────────────────┼──────────────────────┼────────────────────┤
    │ 等权       │ 1/K                     │ 无需估计，最稳健     │ 不区分好坏         │
    │ IC 加权    │ ∝ 过去 L 期 IC 均值     │ 好因子权重大         │ 忽略相关性         │
    │ ICIR 加权  │ ∝ IC 均值 / IC 标准差   │ 兼顾稳定性           │ 忽略相关性         │
    │ 最大化 IR  │ ∝ Σ⁻¹μ（Σ 收缩 50%）    │ 理论最优，考虑相关性 │ 估计不稳，易过拟合 │
    └────────────┴─────────────────────────┴──────────────────────┴────────────────────┘
    本课：L = {LOOKBACK} 期滚动窗口；权重截断为非负后归一化。
    """)
    weights = {m: rolling_weights(ics, m) for m in METHODS}
    last = pd.DataFrame({METHODS[m]: w.iloc[-1] for m, w in weights.items()})
    avg = pd.DataFrame({METHODS[m]: w.mean() for m, w in weights.items()})
    last.index = avg.index = [FACTOR_LABELS[k] for k in last.index]
    print(f"📋 最后一期（{weights['ic'].index[-1].date()}）的权重：")
    print(last.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n📋 全部调仓日的平均权重：")
    print(avg.to_string(float_format=lambda v: f"{v:.3f}"))
    stab = {METHODS[m]: w.diff().abs().sum(axis=1).mean() / 2 for m, w in weights.items()}
    print("\n   月均权重变动（单边）：" + "，".join(f"{k} {v:.1%}" for k, v in stab.items()))
    print("   权重变动大 → 合成因子不稳定 → 额外换手。最大化 IR 通常最敏感。")
    return weights


# ==================== 第三部分：IC 的可得时间 ====================


def ic_timing():
    print("\n" + "=" * 60)
    print("第三部分：t 日能用到哪一期的 IC？")
    print("=" * 60)
    print("""
    IC 序列第 t 行 = corr(t 日因子, t+1 开盘 → 下一调仓日次日开盘 的收益)

       月末 t−2      月末 t−1        月末 t（算权重）   t+1 开盘
          │             │                │                 │
          ├── IC(t−2) ──┼──▶ 已实现      │                 │
          │             ├── IC(t−1) ─────┼────── 未实现 ──▶│ 要到 t+1 开盘才知道
                                         ├──── IC(t) ───────── 完全在未来

    所以在 t 日收盘计算权重时，最新可用的是 IC(t−2)：
       ✅ ics.iloc[: i − 1]     （本课 lag = 2）
       ❌ ics.loc[:t]           包含 IC(t−1) 和 IC(t)，后者用的是"你正要去赚的收益"
    这一条在调仓频率与持有期一致时最容易被忽略。
    """)


# ==================== 第四部分：样本外比较 ====================


def evaluate(factor: pd.DataFrame, ret: pd.DataFrame, ew: pd.Series) -> dict:
    s = ic_summary(calc_ic(factor, ret))
    g = quantile_returns(factor, ret, 5)
    excess = (g[5] - ew.reindex(g.index)).dropna()
    return {"Rank IC": s["IC均值"], "ICIR": s["ICIR"], "t值": s["t值"], "Q5−Q1 年化": g["多空"].mean() * 12,
            "多空夏普": perf_stats(g["多空"])["夏普"], "Top 超额 年化": excess.mean() * 12,
            "Top 超额 t": excess.mean() / excess.std() * np.sqrt(len(excess))}


def out_of_sample(clean, ics, weights, mask, ret):
    print("\n" + "=" * 60)
    print("第四部分：样本外比较（权重只用当时已知的 IC）")
    print("=" * 60)

    dates = weights["ic"].index
    ew = ret.where(mask).mean(axis=1)
    comps = {METHODS[m]: composite(clean, w) for m, w in weights.items()}
    rows = {name: evaluate(f, ret.loc[dates], ew) for name, f in comps.items()}
    singles = {FACTOR_LABELS[k]: evaluate(clean[k].loc[dates], ret.loc[dates], ew) for k in clean}
    df = pd.DataFrame(rows).T
    print(f"📋 评估区间 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 期，未扣交易成本）：")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    sdf = pd.DataFrame(singles).T.sort_values("t值", ascending=False)
    print("\n📋 同一区间的单因子（事后才知道哪个最好，不能作为真实基准）：")
    print(sdf[["Rank IC", "ICIR", "t值", "Top 超额 年化"]].to_string(float_format=lambda v: f"{v:+.3f}"))

    best = df["ICIR"].idxmax()
    print(f"""
    ICIR 最高的合成方法：{best}；单因子 ICIR 的中位数 {sdf['ICIR'].median():+.3f}，合成后中位数 {df['ICIR'].median():+.3f}
    💡 等权往往很难被打败：样本外复杂加权的优势常被估计误差抵消，
       建议以等权作为基准，复杂方法必须在样本外显著更好才采用。
       这里只有 {len(dates)} 期，方法之间的差异大多在统计误差之内。
    """)
    return comps, df


# ==================== 第五部分：前视的代价 ====================


def lookahead_cost(clean, ics, weights, mask, ret, honest):
    print("\n" + "=" * 60)
    print("第五部分：前视的代价")
    print("=" * 60)

    dates = weights["ic"].index
    ew = ret.where(mask).mean(axis=1)
    rows = {"✅ 滚动 IC（lag = 2）": honest.loc["IC 加权"].to_dict()}
    leak = rolling_weights(ics, "ic", lag=0).reindex(dates)
    rows["❌ 滚动 IC（lag = 0，含当期）"] = evaluate(composite(clean, leak), ret.loc[dates], ew)
    full = combine_weights(ics, "ic")
    full_w = pd.DataFrame([full] * len(dates), index=dates)
    rows["❌ 全样本 IC"] = evaluate(composite(clean, full_w), ret.loc[dates], ew)
    df = pd.DataFrame(rows).T
    print("📋 IC 加权，三种权重来源：")
    print(df[["Rank IC", "ICIR", "t值", "Q5−Q1 年化", "Top 超额 年化"]].to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    💡 全样本权重"事先知道哪些因子在整个样本里有效"；lag = 0 的权重包含了当期收益本身。
       比较这两行与第一行的差距 —— 这部分"提升"在实盘里不存在。
       CONCEPT 第 13.2 节的陷阱 5 说的就是这个：合成权重只能用已经实现的 IC。
    """)


# ==================== 主函数 ====================


def plot_results(comps, ret):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, f in comps.items():
        g = quantile_returns(f, ret.loc[f.index], 5)
        ax.plot(g.index, (1 + g["多空"]).cumprod(), label=name)
    ax.axhline(1, color="gray", lw=0.6)
    ax.set_title("合成因子 Q5 − Q1 净值（样本外，未扣成本）")
    ax.legend()
    path = save_figure(fig, "09_factor_combination.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    dates, mask, ret, raw, clean = research_inputs(panel)
    ics = pd.DataFrame({k: calc_ic(v, ret) for k, v in clean.items()})

    why_combine(clean, ics)
    weights = weights_overview(ics)
    ic_timing()
    comps, df = out_of_sample(clean, ics, weights, mask, ret)
    lookahead_cost(clean, ics, weights, mask, ret, df)
    plot_results(comps, ret)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 LOOKBACK 改为 12 和 36，收缩系数改为 0、0.2、0.8，观察最大化 IR 的权重稳定性和样本外 ICIR
    2. 先把 EP、BP 合成为"价值"，lowvol、lowturn 合成为"低风险"，再在大类之间等权，与直接等权比较
    3. 用 Gram-Schmidt 对因子正交化后再等权合成，改变正交顺序，观察结果如何变化
    4. 计算各方法合成因子 Top 组的换手率和扣费后收益（复用 06 课的 turnover_and_cost）
    """)


if __name__ == "__main__":
    main()
