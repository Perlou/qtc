"""
cvxpy 带约束组合优化
====================

学习目标：
    1. 掌握 cvxpy 的建模方式与 DCP 规则（凸优化的"类型系统"）
    2. 搭建一个指数增强优化器：alpha - λ·主动风险 - 成本，并逐步加入约束
       个股上限 → 行业约束 → 跟踪误差约束 → 换手约束
    3. 用因子模型 Σ = BFB' + D 表达风险，理解不展开 n×n 矩阵的计算技巧
    4. 处理不可行（infeasible）：检查状态码，把硬约束改成软约束
    5. 用资产类别约束做大类资产配置

核心概念：
    - 主动权重 a = w - w_b；跟踪误差 TE = √(a'Σa)
    - 单边换手 = ½ Σ|w - w_0|
    - 转移系数 TC：约束越多，最终持仓与原始 alpha 的相关性越低
    - DCP：Minimize(凸) / Maximize(凹) 才合法

前置知识：
    - 01–05 课；Phase 7 因子与 Barra 风险模型
    - CONCEPT.md 第 8 节

环境要求：
    - pip install numpy pandas cvxpy akshare
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用合成 / 模拟数据，并打印 ⚠️
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cvxpy as cp
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from portfolio_utils import TRADING_DAYS, asset_class_of, load_etf_returns, load_stock_returns

load_dotenv()
pd.set_option("display.width", 200)

IC = 0.05            # 假设的信号 IC（用于把 z-score 转成预期收益尺度）
LAMBDA = 1.0         # 主动风险厌恶系数（α、风险均为年化）
COST = 0.002         # 单边交易成本 0.2%（佣金 + 印花税均摊 + 冲击的线性近似）
INDUSTRY_W = {"银行": 0.25, "食品饮料": 0.20, "医药": 0.15, "周期资源": 0.20, "科技制造": 0.20}


# ==================== 数据准备：alpha、风险模型、基准 ====================


def industry_factor_model(ret: pd.DataFrame, industry: pd.Series):
    """
    最简单的行业因子模型：因子收益 = 每天的行业等权平均收益
    B: n×k 行业哑变量；F: k×k 因子协方差；D: 特异方差（年化）
    """
    B = pd.get_dummies(industry).astype(float).loc[ret.columns]
    f = ret.T.groupby(industry).mean().T[B.columns]     # T × k 因子收益
    resid = ret - f @ B.T
    F = f.cov().to_numpy() * TRADING_DAYS
    D = resid.var().to_numpy() * TRADING_DAYS
    return B, F, D


def prepare(ret: pd.DataFrame, industry: pd.Series, end: int):
    """用截至第 end 天（不含）的数据构造 alpha 与风险模型，避免前视"""
    window = ret.iloc[end - 250 : end]
    B, F, D = industry_factor_model(window, industry)
    Sigma = B.to_numpy() @ F @ B.to_numpy().T + np.diag(D)
    # 示例信号：过去 120 天剔除最近 20 天的动量，截面标准化
    mom = (1 + ret.iloc[end - 120 : end - 20]).prod() - 1
    z = ((mom - mom.mean()) / mom.std()).to_numpy()
    alpha = IC * np.sqrt(D) * z          # Grinold：α = IC × 残差波动 × z
    return alpha, B, F, D, Sigma


def benchmark_weights(industry: pd.Series) -> np.ndarray:
    counts = industry.value_counts()
    return np.array([INDUSTRY_W[ind] / counts[ind] for ind in industry])


# ==================== 第一部分：cvxpy 与 DCP ====================


def dcp_basics():
    print("=" * 60)
    print("第一部分：cvxpy 基础与 DCP 规则")
    print("=" * 60)
    print("""
    cvxpy 的三件套：Variable（决策变量）、Objective（目标）、Constraints（约束列表）
        w = cp.Variable(n)
        prob = cp.Problem(cp.Maximize(alpha @ w - lam * cp.quad_form(w, Sigma)), [cp.sum(w) == 1, w >= 0])
        prob.solve()

    DCP 像一个类型系统：每个表达式都有"凸 / 凹 / 仿射"属性，cvxpy 在求解前做检查。
    ┌─────────────────────────────┬────────┬────────────────────────┐
    │ 表达式                      │ 凸性   │ 组合含义               │
    ├─────────────────────────────┼────────┼────────────────────────┤
    │ alpha @ w                   │ 仿射   │ 预期 alpha             │
    │ cp.quad_form(w, Σ)          │ 凸     │ 方差（Σ 半正定）       │
    │ cp.norm1(w - w0)            │ 凸     │ 换手                   │
    │ cp.sum(cp.pos(-R @ w - z))  │ 凸     │ CVaR 的尾部损失        │
    │ 持股数 ≤ K                  │ 非凸   │ 需要整数变量（MIQP）   │
    └─────────────────────────────┴────────┴────────────────────────┘
    """)
    w = cp.Variable(3)
    Sigma = np.eye(3)
    try:
        cp.Problem(cp.Maximize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1]).solve()
    except cp.error.DCPError as e:
        print(f"❌ Maximize(方差) 被拒绝：DCPError —— {str(e).splitlines()[0][:80]}...")
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1])
    print(f"✅ Minimize(方差) 合法：is_dcp = {prob.is_dcp()}，最优值 {prob.solve(solver=cp.CLARABEL):.4f}，w = {np.round(w.value, 4)}")


# ==================== 第二部分：逐步加约束的指数增强 ====================


def solve_step(alpha, B, F, D, w_b, w_prev, industry, step: int, te_max=0.02, turnover_max=0.03):
    n = len(alpha)
    Bm = B.to_numpy()
    w = cp.Variable(n)
    a = w - w_b
    f = Bm.T @ a                                         # 主动因子暴露（k 维）
    risk = cp.quad_form(f, cp.psd_wrap(F)) + cp.sum(cp.multiply(D, cp.square(a)))  # 不展开 n×n
    trade = cp.norm1(w - w_prev)
    obj = alpha @ w - LAMBDA * risk
    cons = [cp.sum(w) == 1]
    if step >= 2:
        cons += [w >= 0, w <= 0.10, cp.abs(a) <= 0.03]
    if step >= 3:
        cons += [cp.abs(Bm.T @ a) <= 0.01]                # 行业主动权重 ±1%
    if step >= 4:
        cons += [risk <= te_max**2]                       # 跟踪误差约束（年化）
    if step >= 5:
        obj = obj - COST * trade
        cons += [0.5 * trade <= turnover_max]             # 单边换手上限
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob, w.value


def describe(w, alpha, Sigma, w_b, w_prev, B):
    a = w - w_b
    sd = np.sqrt(np.diag(Sigma))
    tc = np.corrcoef(a * sd, alpha / sd)[0, 1]
    return {
        "预期超额": f"{alpha @ a:+.2%}",
        "事前TE": f"{np.sqrt(a @ Sigma @ a):.2%}",
        "最大|主动|": f"{np.abs(a).max():.1%}",
        "做空合计": f"{w[w < 0].sum():+.1%}",
        "持股数": int((w > 1e-3).sum()),
        "最大行业偏离": f"{np.abs(B.to_numpy().T @ a).max():.1%}",
        "单边换手": f"{0.5 * np.abs(w - w_prev).sum():.1%}",
        "TC": f"{tc:.2f}",
    }


def index_enhancement(ret: pd.DataFrame, industry: pd.Series):
    print("\n" + "=" * 60)
    print("第二部分：指数增强 —— 逐步加约束")
    print("=" * 60)

    w_b = benchmark_weights(industry)
    T = len(ret)
    # 上期持仓：用 21 天前的数据跑一次"全部约束（不含换手）"的优化
    alpha0, B0, F0, D0, _ = prepare(ret, industry, T - 21)
    _, w_prev = solve_step(alpha0, B0, F0, D0, w_b, w_b, industry, step=4)
    alpha, B, F, D, Sigma = prepare(ret, industry, T)
    print(f"""
    股票池：{len(w_b)} 只，5 个行业；基准权重为假设值（行业权重 {INDUSTRY_W}，行业内等权）
    风险模型：过去 250 天的行业因子模型 Σ = BFB' + D
    alpha：120 日动量（剔除近 20 日）的 z-score，α = IC × 残差波动 × z，IC 假设为 {IC}
          （示例信号，没有经过 Phase 7 的有效性检验，只用来演示优化器）
    目标：max α'w - {LAMBDA}·(w-w_b)'Σ(w-w_b) [- {COST:.1%}·Σ|w-w_0|]
    """)
    labels = {
        1: "① 仅满仓约束",
        2: "② + 多头、个股≤10%、|主动|≤3%",
        3: "③ + 行业偏离≤1%",
        4: "④ + 跟踪误差≤2%",
        5: "⑤ + 换手≤3%、扣成本",
    }
    rows, weights = {}, {}
    for step, label in labels.items():
        prob, w = solve_step(alpha, B, F, D, w_b, w_prev, industry, step)
        if w is None:
            rows[label] = {"状态": prob.status}
            continue
        weights[label] = w
        rows[label] = {"状态": prob.status, **describe(w, alpha, Sigma, w_b, w_prev, B)}
    print(pd.DataFrame(rows).T.to_string())
    print("""
    💡 读表：
       - ① 没有任何约束时，优化器会大量做空、集中下注——它完全相信 alpha 和风险模型
       - 每加一层约束，可行域变小，目标函数值只会变差不会变好，预期超额通常随之下降
       - TC（风险调整后的主动权重与 alpha 的相关性）总体随约束增多而下降，但不一定逐步单调
       - ⑤ 的换手约束把组合"拴"在上期持仓附近，这是控制交易成本最直接的手段
    """)
    last = list(weights)[-1]        # 正常情况下是 ⑤；若后面的步骤不可行，退回最后一个有解的步骤
    final = pd.DataFrame({"行业": industry, "基准%": w_b * 100, "上期%": w_prev * 100,
                          "最终%": weights[last] * 100, "alpha%": alpha * 100}, index=ret.columns)
    final["主动%"] = final["最终%"] - final["基准%"]
    print(f"最终组合（{last}）：")
    print(final.round(2).sort_values(["行业", "主动%"]).to_string())
    print("\n行业汇总（%）：")
    print(final.groupby("行业")[["基准%", "最终%", "主动%"]].sum().round(2).to_string())
    return alpha, B, F, D, Sigma, w_b, w_prev


# ==================== 第三部分：因子模型技巧与不可行处理 ====================


def tricks_and_infeasible(alpha, B, F, D, Sigma, w_b, w_prev, industry):
    print("\n" + "=" * 60)
    print("第三部分：因子模型写法 vs 完整协方差写法；不可行的处理")
    print("=" * 60)

    t0 = time.perf_counter()
    _, w_factor = solve_step(alpha, B, F, D, w_b, w_prev, industry, step=4)
    t_factor = time.perf_counter() - t0
    n = len(alpha)
    w = cp.Variable(n)
    a = w - w_b
    cons = [cp.sum(w) == 1, w >= 0, w <= 0.10, cp.abs(a) <= 0.03,
            cp.abs(B.to_numpy().T @ a) <= 0.01, cp.quad_form(a, cp.psd_wrap(Sigma)) <= 0.02**2]
    t0 = time.perf_counter()
    cp.Problem(cp.Maximize(alpha @ w - LAMBDA * cp.quad_form(a, cp.psd_wrap(Sigma))), cons).solve(solver=cp.CLARABEL)
    t_full = time.perf_counter() - t0
    print(f"""
    同一个问题（步骤④）的两种写法：
       因子模型写法（f = B'a，风险 = f'Ff + Σ D_i a_i²）：{t_factor * 1000:.0f} ms
       完整协方差写法（a'Σa，Σ 为 n×n）：               {t_full * 1000:.0f} ms
       两种写法的最大权重差：{np.abs(w_factor - w.value).max():.1e}
    n = {n} 时差别不大；n = 3000、k = 40 时，因子写法的二次项从 n² 个降到 k² + n 个，速度相差数十倍。
    """)

    print("⚠️ 不可行示例：跟踪误差 ≤ 0.3% 且单边换手 ≤ 2%")
    prob, w_hard = solve_step(alpha, B, F, D, w_b, w_prev, industry, step=5, te_max=0.003, turnover_max=0.02)
    print(f"   prob.status = {prob.status!r}，w.value = {None if w_hard is None else '有值'}")
    te_prev = np.sqrt((w_prev - w_b) @ Sigma @ (w_prev - w_b))
    print(f"   原因：上期持仓的 TE 为 {te_prev:.2%}，只换手 2% 无法把 TE 降到 0.3% 以内")

    # 软约束：把 TE 上限改成目标函数里的惩罚项
    Bm = B.to_numpy()
    w = cp.Variable(n)
    a = w - w_b
    risk = cp.quad_form(Bm.T @ a, cp.psd_wrap(F)) + cp.sum(cp.multiply(D, cp.square(a)))
    trade = cp.norm1(w - w_prev)
    penalty = 1000 * cp.pos(risk - 0.003**2)            # 超出部分重罚
    soft = cp.Problem(cp.Maximize(alpha @ w - LAMBDA * risk - COST * trade - penalty),
                      [cp.sum(w) == 1, w >= 0, w <= 0.10, cp.abs(Bm.T @ a) <= 0.01, 0.5 * trade <= 0.02])
    soft.solve(solver=cp.CLARABEL)
    a_val = w.value - w_b
    print(f"   软约束版本：status = {soft.status!r}，TE = {np.sqrt(a_val @ Sigma @ a_val):.2%}"
          f"（目标 0.3%，尽力靠近），单边换手 {0.5 * np.abs(w.value - w_prev).sum():.1%}")
    print("""
    💡 生产环境的做法：
       1. 每次求解后检查 prob.status，只接受 optimal / optimal_inaccurate
       2. 不可行时按优先级放松：先把次要约束（TE、风格）改成惩罚项，再放松换手
       3. 仍然失败就沿用上期持仓，并报警——绝不能拿 None 去下单
    """)


# ==================== 第四部分：资产类别约束 ====================


def asset_class_allocation():
    print("\n" + "=" * 60)
    print("第四部分：大类资产配置中的资产类别约束")
    print("=" * 60)

    ret = load_etf_returns(verbose=False)
    names = list(ret.columns)
    classes = asset_class_of(names)
    Sigma = ret.cov().to_numpy() * TRADING_DAYS
    mu_hist = ret.mean().to_numpy() * TRADING_DAYS
    mu = 0.5 * mu_hist + 0.5 * mu_hist.mean()   # 向截面均值收缩一半，降低估计误差的影响
    limits = {"A股": (0.10, 0.40), "海外股票": (0.0, 0.20), "债券": (0.20, 0.70), "商品": (0.0, 0.15)}
    target_vol = 0.08

    upper = np.array([0.60 if c == "债券" else 0.25 for c in classes])  # 国债单只可到 60%，其余 25%
    w = cp.Variable(len(names), nonneg=True)
    cons = [cp.sum(w) == 1, w <= upper, cp.quad_form(w, cp.psd_wrap(Sigma)) <= target_vol**2]
    for c, (lo, hi) in limits.items():
        mask = (classes == c).to_numpy().astype(float)
        cons += [mask @ w >= lo, mask @ w <= hi]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        print(f"   ⚠️ 约束组合不可行（status = {prob.status}），请放松类别区间或波动上限")
        return

    w_free = cp.Variable(len(names), nonneg=True)
    cp.Problem(cp.Maximize(mu @ w_free), [cp.sum(w_free) == 1, cp.quad_form(w_free, cp.psd_wrap(Sigma)) <= target_vol**2]).solve(solver=cp.CLARABEL)

    tbl = pd.DataFrame({"类别": classes, "收缩后μ%": mu * 100, "仅波动约束%": w_free.value * 100, "加类别约束%": w.value * 100}, index=names)
    print(f"目标：在年化波动 ≤ {target_vol:.0%} 的前提下最大化预期收益（μ 为历史均值向截面均值收缩 50%）")
    print(f"约束：国债 ≤ 60%、其余单只 ≤ 25%，类别区间 {limits}")
    print(tbl.round(2).to_string())
    agg = tbl.groupby("类别")[["仅波动约束%", "加类别约束%"]].sum().round(1)
    print("\n按类别汇总：")
    print(agg.to_string())
    print(f"   status = {prob.status}，组合波动 {np.sqrt(w.value @ Sigma @ w.value):.2%}，预期收益 {mu @ w.value:.2%}"
          f"（仅波动约束时 {mu @ w_free.value:.2%}）")
    print("""
    💡 类别约束把投资政策（IPS）写进优化器：约束的代价是预期收益略降，换来的是组合"长得像"你愿意持有的样子。
    """)


# ==================== 主函数 ====================


def main():
    dcp_basics()
    ret, industry = load_stock_returns()
    alpha, B, F, D, Sigma, w_b, w_prev = index_enhancement(ret, industry)
    tricks_and_infeasible(alpha, B, F, D, Sigma, w_b, w_prev, industry)
    asset_class_allocation()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 加入"市值暴露"约束：用 log(流通市值) 做风格因子，限制组合相对基准的暴露 |x'a| ≤ 0.1
    2. 把成本改为平方根冲击模型 c·|Δw|^1.5（cp.power(cp.abs(dw), 1.5)），观察换手如何变化
    3. 把第二部分做成月度滚动回测：每月重算 alpha 和风险模型，比较 ③④⑤ 三种约束组合的净值、换手、TE
    4. 加入"持股数 ≤ 10"的约束：需要布尔变量（cp.Variable(n, boolean=True)），本机有哪些求解器支持 MIQP？
    """)


if __name__ == "__main__":
    main()
