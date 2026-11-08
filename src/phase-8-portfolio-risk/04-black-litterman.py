"""
Black-Litterman 模型
====================

学习目标：
    1. 理解为什么"历史均值 + 均值方差"在实践中几乎不可用
    2. 反向优化：从基准（市场）权重反推隐含均衡收益 Π = δΣw
    3. 用 P、Q、Ω 表达绝对观点和相对观点，计算后验收益 μ_BL 与后验协方差
    4. 理解信心度（Ω 的大小）如何在"相信市场"和"相信观点"之间折中
    5. 把 μ_BL 放进带约束的优化器，得到"在基准附近按观点偏移"的组合

核心概念：
    - 先验：μ ~ N(Π, τΣ)；观点：Pμ = Q + ε，ε ~ N(0, Ω)
    - 后验：μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]  ← 精度加权平均
    - He-Litterman：Ω = diag(P·τΣ·P')；无观点时最优组合 = 基准组合
    - 只有观点涉及的资产会偏离基准（按 1/(1+τ) 缩放后）

前置知识：
    - 01-mean-variance.py、02-covariance-shrinkage.py
    - CONCEPT.md 第 6 节；Phase 6 卡尔曼滤波（同样是精度加权）

环境要求：
    - pip install numpy pandas cvxpy scikit-learn matplotlib
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用示意参数的模拟收益，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.covariance import LedoitWolf

from portfolio_utils import TRADING_DAYS, load_etf_returns
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

DELTA = 2.5   # 风险厌恶系数（常用 2~3）
TAU = 0.05    # 先验不确定性标量（常用 0.025~0.05）
# 基准权重：一个假设的"战略配置"组合（ETF 没有统一的市值口径，这里只是示例）
BENCHMARK = {"沪深300": 0.25, "中证500": 0.10, "创业板": 0.05, "纳指": 0.075, "标普500": 0.075, "国债": 0.35, "黄金": 0.10}


def black_litterman(Sigma, w_mkt, P, Q, delta=DELTA, tau=TAU, Omega=None):
    """返回 (Π, μ_BL, Σ_BL, Ω)"""
    pi = delta * Sigma @ w_mkt
    tS = tau * Sigma
    if Omega is None:  # He-Litterman：观点不确定性与观点组合自身风险成比例
        Omega = np.diag(np.diag(P @ tS @ P.T))
    inv_tS = np.linalg.inv(tS)
    inv_Om = np.linalg.inv(Omega)
    M = np.linalg.inv(inv_tS + P.T @ inv_Om @ P)  # 后验均值的不确定性
    mu_bl = M @ (inv_tS @ pi + P.T @ inv_Om @ Q)
    return pi, mu_bl, Sigma + M, Omega


def mv_long_only(mu: np.ndarray, Sigma: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """max w'μ - δ/2 w'Σw，s.t. Σw = 1, w ≥ 0"""
    w = cp.Variable(len(mu), nonneg=True)
    obj = cp.Maximize(mu @ w - delta / 2 * cp.quad_form(w, cp.psd_wrap(Sigma)))
    cp.Problem(obj, [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


# ==================== 第一部分：历史均值的问题 ====================


def historical_mean_problem(returns: pd.DataFrame):
    print("=" * 60)
    print("第一部分：用历史均值做均值方差 —— 为什么不行")
    print("=" * 60)

    names = returns.columns
    # 这里 n = 7、T 超过 2500，样本协方差本身已足够稳定；而 sklearn 的 LedoitWolf 以"等方差单位阵"为目标，
    # 会把国债（波动约 2~3%）和创业板（30%+）的方差往中间拉，对波动差异很大的大类资产并不合适
    Sigma = returns.cov().to_numpy() * TRADING_DAYS
    lw = LedoitWolf().fit(returns.to_numpy())
    print(f"\n协方差：样本协方差（n/T = {returns.shape[1] / len(returns):.4f}）。对比：LW 单位阵目标会把国债波动从 "
          f"{np.sqrt(Sigma[-2, -2]):.2%} 拉到 {np.sqrt(lw.covariance_[-2, -2] * TRADING_DAYS):.2%}（收缩强度 {lw.shrinkage_:.3f}）")
    mu_hist = returns.mean().to_numpy() * TRADING_DAYS
    w_mkt = np.array([BENCHMARK[c] for c in names])

    half = len(returns) // 2
    mu_1 = returns.iloc[:half].mean().to_numpy() * TRADING_DAYS
    mu_2 = returns.iloc[half:].mean().to_numpy() * TRADING_DAYS
    tbl = pd.DataFrame({
        "基准权重%": w_mkt * 100,
        "前半段均值%": mu_1 * 100,
        "后半段均值%": mu_2 * 100,
        "MV(前半段μ)%": mv_long_only(mu_1, Sigma) * 100,
        "MV(后半段μ)%": mv_long_only(mu_2, Sigma) * 100,
        "MV(全样本μ)%": mv_long_only(mu_hist, Sigma) * 100,
    }, index=names)
    print(f"\n多头约束、δ = {DELTA} 的均值方差（Σ 固定为全样本协方差，只换 μ）：")
    print(tbl.round(1).to_string())
    print(f"""
    前后两段样本：{returns.index[0].date()} ~ {returns.index[half - 1].date()}  vs  {returns.index[half].date()} ~ {returns.index[-1].date()}
    💡 只换了 μ 的估计区间，最优组合就可能从一组资产跳到另一组；而且通常集中在少数资产上，
       与任何"合理"的基准都相去甚远。BL 的思路：不要从零估计 μ，而是先从基准反推。
    """)
    return Sigma, w_mkt, mu_hist


# ==================== 第二部分：反向优化 ====================


def reverse_optimization(names, Sigma, w_mkt, mu_hist):
    print("\n" + "=" * 60)
    print("第二部分：反向优化 —— 基准组合隐含的均衡收益 Π")
    print("=" * 60)
    print(f"""
    若基准 w_mkt 是 max w'μ - δ/2·w'Σw 的解，则一阶条件 μ = δΣw_mkt，记为 Π。
    含义：只有当各资产的预期超额收益恰好等于 Π 时，持有基准才是最优的。
    δ = {DELTA}，τ = {TAU}
    """)
    pi = DELTA * Sigma @ w_mkt
    vol = np.sqrt(np.diag(Sigma))
    sigma_m = np.sqrt(w_mkt @ Sigma @ w_mkt)
    tbl = pd.DataFrame({
        "基准权重%": w_mkt * 100,
        "年化波动%": vol * 100,
        "隐含超额收益 Π%": pi * 100,
        "Π/σ（隐含Sharpe）": pi / vol,
        "历史均值%": mu_hist * 100,
    }, index=names)
    print(tbl.round(2).to_string())
    w_back = np.linalg.solve(DELTA * Sigma, pi)
    print(f"""
    基准组合：年化波动 {sigma_m:.2%}，隐含超额收益 {w_mkt @ pi:.2%}
    ✅ 校验：用 Π 做无约束均值方差 w* = (δΣ)⁻¹Π，与基准的最大差 = {np.abs(w_back - w_mkt).max():.1e}

    💡 Π 是"平滑"的：波动高、与组合相关性高的资产被要求更高的收益，不会出现历史均值那种极端排序。
       历史均值可以作为参考，但 BL 不直接使用它。
    """)
    return pi


# ==================== 第三部分：加入观点 ====================


def build_views(names):
    idx = {c: i for i, c in enumerate(names)}
    n = len(names)
    views = [
        ("黄金 年化超额收益 3%", {"黄金": 1.0}, 0.03),
        ("中证500 跑赢 沪深300 2%", {"中证500": 1.0, "沪深300": -1.0}, 0.02),
        ("纳指 跑输 标普500 1%", {"纳指": 1.0, "标普500": -1.0}, -0.01),
    ]
    P = np.zeros((len(views), n))
    for k, (_, legs, _) in enumerate(views):
        for c, v in legs.items():
            P[k, idx[c]] = v
    Q = np.array([q for _, _, q in views])
    return views, P, Q


def add_views(names, Sigma, w_mkt):
    print("\n" + "=" * 60)
    print("第三部分：加入观点 —— 计算后验收益 μ_BL")
    print("=" * 60)

    views, P, Q = build_views(names)
    print("\n观点（P 的每一行是一个观点组合，Q 是观点收益）：")
    print(pd.DataFrame(P, index=[v[0] for v in views], columns=names).to_string())
    pi, mu_bl, Sigma_bl, Omega = black_litterman(Sigma, w_mkt, P, Q)
    print(f"\nΩ（He-Litterman，对角）= {np.round(np.diag(Omega), 6)}")

    view_tbl = pd.DataFrame({
        "先验 PΠ%": P @ pi * 100,
        "观点 Q%": Q * 100,
        "后验 Pμ_BL%": P @ mu_bl * 100,
    }, index=[v[0] for v in views])
    print("\n观点组合的收益：先验 → 观点 → 后验")
    print(view_tbl.round(2).to_string())
    print("   后验落在先验和观点之间：BL 是先验与观点的\"精度加权平均\"")

    # 无约束最优：w* = (δΣ_BL)⁻¹ μ_BL；He-Litterman 性质：w*(1+τ) - w_mkt = P'Λ
    w_star = np.linalg.solve(DELTA * Sigma_bl, mu_bl)
    tilt = w_star * (1 + TAU) - w_mkt
    tbl = pd.DataFrame({
        "Π%": pi * 100, "μ_BL%": mu_bl * 100, "μ_BL-Π%": (mu_bl - pi) * 100,
        "基准%": w_mkt * 100, "BL最优%": w_star * 100, "(1+τ)w*-基准%": tilt * 100,
    }, index=names)
    print("\n各资产：")
    print(tbl.round(2).to_string())
    in_view = np.abs(P).sum(axis=0) > 0
    print(f"""
    BL 最优组合权重和 = {w_star.sum():.4f}（无预算约束，剩余部分视为现金/借贷）
    观点未涉及的资产 {list(names[~in_view])}：(1+τ)w* - 基准 的最大绝对值 = {np.abs(tilt[~in_view]).max():.1e}
    💡 μ_BL 会让所有相关资产的预期收益都动一点（通过 Σ 传导），但最优组合只在观点涉及的资产上偏离基准
       ——这正是 BL 好用的地方：组合变化可以追溯到具体观点。
    """)
    return pi, mu_bl, Sigma_bl, P, Q


# ==================== 第四部分：信心度 ====================


def confidence_sweep(names, Sigma, w_mkt, P, Q):
    print("\n" + "=" * 60)
    print("第四部分：信心度 —— Ω 越小越相信观点")
    print("=" * 60)

    base_omega = np.diag(np.diag(P @ (TAU * Sigma) @ P.T))
    rows = []
    for c in [100, 10, 1, 0.1, 0.01]:
        _, mu_bl, Sigma_bl, _ = black_litterman(Sigma, w_mkt, P, Q, Omega=base_omega * c)
        pv = P @ mu_bl
        w = np.linalg.solve(DELTA * Sigma_bl, mu_bl)
        rows.append([c] + [f"{x:+.2%}" for x in pv] + [f"{w[list(names).index('黄金')]:.1%}"])
    pi = DELTA * Sigma @ w_mkt
    cols = ["Ω 缩放"] + [f"观点{k + 1} 后验" for k in range(len(Q))] + ["黄金 BL 权重"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print(f"   先验 PΠ = {np.round(P @ pi * 100, 2)}%，观点 Q = {np.round(Q * 100, 2)}%，黄金基准权重 {BENCHMARK['黄金']:.0%}")
    print("""
    💡 Ω 很大（信心很低）→ 后验 ≈ 先验 Π，组合 ≈ 基准；Ω 很小（信心很高）→ 后验 ≈ Q。
       信心越高，黄金权重越极端——BL 并不自动防止极端仓位，仍需要约束。
       只有 τ 与 Ω 的相对大小影响 μ_BL；不同库对 τ、Ω 的参数化不同，读文档时要注意。
       Idzorek（2005）提出用 0~100% 的直观信心度反推 Ω，更便于和投资经理沟通。
    """)


# ==================== 第五部分：带约束的 BL 组合 ====================


def constrained_bl(names, Sigma, w_mkt, mu_hist, pi, mu_bl, Sigma_bl):
    print("\n" + "=" * 60)
    print("第五部分：把 μ_BL 放进带约束的优化器")
    print("=" * 60)

    sols = {
        "基准": w_mkt,
        "MV(历史均值)": mv_long_only(mu_hist, Sigma),
        "MV(Π)": mv_long_only(pi, Sigma),
        "MV(μ_BL)": mv_long_only(mu_bl, Sigma_bl),
    }
    tbl = pd.DataFrame({k: v * 100 for k, v in sols.items()}, index=names)
    print("多头 + 满仓约束下的最优权重 %：")
    print(tbl.round(1).to_string())
    for k, v in sols.items():
        print(f"   {k:12s} 与基准的 L1 偏离 {np.abs(v - w_mkt).sum():.1%}，持有资产数 {(v > 1e-3).sum()}")
    _, P, _ = build_views(names)
    others = [c for c, used in zip(names, np.abs(P).sum(axis=0) > 0) if not used]
    moved = {c: sols["MV(μ_BL)"][list(names).index(c)] - BENCHMARK[c] for c in others}
    print(f"""
    💡 用 Π 做多头满仓优化正好还原基准；用历史均值则得到与基准毫无关系的集中组合。
       用 μ_BL 时，偏移主要来自观点方向；但在"满仓 + 多头"约束下，观点带来的加仓要从别处挪钱，
       所以观点之外的资产也会变化：{ {k: f"{v:+.1%}" for k, v in moved.items()} }
       BL 解决的是 μ 的问题；资产多、样本短时 Σ 仍应使用收缩或因子模型。
    """)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(names))
    width = 0.2
    for k, ((label, w), c) in enumerate(zip(sols.items(), ["#8a8984", "#e34948", "#1baf7a", "#2a78d6"])):
        ax.bar(x + (k - 1.5) * width, w * 100, width, label=label, color=c, edgecolor="white", lw=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("权重 %")
    ax.set_title("基准、历史均值 MV 与 Black-Litterman 组合")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    path = save_figure(fig, "04_black_litterman_weights.png", subdir="phase-8")
    print(f"🖼  权重对比图已保存：{path}")


# ==================== 主函数 ====================


def main():
    returns = load_etf_returns()
    if is_synthetic(returns):
        print("⚠️ 以下结果基于模拟收益，只用于理解方法，不代表真实市场\n")
    names = returns.columns
    Sigma, w_mkt, mu_hist = historical_mean_problem(returns)
    pi = reverse_optimization(names, Sigma, w_mkt, mu_hist)
    _, mu_bl, Sigma_bl, P, Q = add_views(names, Sigma, w_mkt)
    confidence_sweep(names, Sigma, w_mkt, P, Q)
    constrained_bl(names, Sigma, w_mkt, mu_hist, pi, mu_bl, Sigma_bl)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 加一个"篮子观点"：A 股三只等权组合跑赢海外两只等权组合 3%，P 的这一行应该怎么写？
    2. 用 δ = (E[r_m] - r_f) / σ_m² 从基准组合的历史数据估计 δ，与 2.5 比较，Π 怎么变？
    3. 实现 Idzorek 方法：给每个观点一个 0~100% 的信心度，反推 Ω 的对角元素
    4. 把观点换成"量化观点"：用过去 12 个月动量最高的资产减最低的资产，滚动回测 BL 组合与基准的差异
    """)


if __name__ == "__main__":
    main()
