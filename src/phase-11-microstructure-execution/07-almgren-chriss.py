"""
Almgren-Chriss 最优执行
=======================

学习目标：
    1. 理解执行中的根本矛盾：执行越快冲击越大，执行越慢价格风险越大
    2. 掌握 Almgren-Chriss 模型的设定：永久冲击、临时冲击、成本的期望与方差
    3. 实现闭式最优轨迹（sinh 形式），并用数值优化与蒙特卡洛两种方法验证
    4. 画出不同风险厌恶系数 λ 下的执行有效前沿，理解 λ → 0 时退化为 TWAP

核心概念：
    - 目标：min E[Cost] + λ·Var[Cost]（与 Phase 8 的 Markowitz 均值-方差优化同一思想）
    - 最优剩余持仓：x_j = X · sinh(κ(T − t_j)) / sinh(κT)
    - κ 由 (2/τ²)(cosh(κτ) − 1) = λσ²/η̃ 决定；1/κ 是订单的"半衰期"
    - 有效前沿：每个 λ 对应一条最优轨迹，在 (成本标准差, 期望成本) 平面上连成一条曲线

前置知识：
    - 06 课（TWAP / VWAP / 实施缺口）；Phase 8 的均值-方差优化；Phase 3 的蒙特卡洛模拟

环境要求：
    - pip install numpy pandas scipy matplotlib
    - 本课不需要网络。参数量级参考 Almgren & Chriss (2000) 原论文的示例
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.linalg import solve_banded

from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

# 原论文示例的量级：5 天内卖出 100 万股，初始价格 50
PARAMS = dict(X=1e6, T=5.0, N=50, sigma=0.95, eta=2.5e-6, gamma=2.5e-7, eps=0.0625)
S0 = 50.0


# ==================== 第一部分：冲击成本与价格风险的权衡 ====================


def tradeoff_intuition():
    print("=" * 60)
    print("第一部分：执行越快越贵，执行越慢越险")
    print("=" * 60)
    print("""
      成本 ▲
           │╲                                ╱  价格风险（方差）随执行时间增长
           │ ╲                            ╱
           │  ╲  冲击成本               ╱
           │   ╲ 随执行时间下降      ╱
           │     ╲              ╱
           │       ╲ ____  ╱  ← "成本 + λ·风险"最小的点取决于风险厌恶 λ
           └──────────────────────────────────▶ 执行时长
             立即全部成交              慢慢均匀成交

    06 课的结果已经出现了这个权衡："一次性"风险最小但冲击最大，TWAP / VWAP 反之。
    Almgren & Chriss (2000) 把它写成均值-方差优化，给出了闭式解。
    """)


# ==================== 第二部分：模型设定 ====================


def model_setup():
    print("\n" + "=" * 60)
    print("第二部分：模型设定")
    print("=" * 60)
    p = PARAMS
    tau = p["T"] / p["N"]
    print(f"""
    在 [0, T] 内卖出 X 股，分成 N 段，每段 τ = T/N。x_k 为第 k 个时刻的剩余持仓（x_0 = X，x_N = 0），
    n_k = x_{{k−1}} − x_k 为第 k 段卖出量，交易速度 v_k = n_k / τ。

    价格（算术随机游走 + 永久冲击）： S_k = S_{{k−1}} + σ√τ·ξ_k − τ·γ·v_k
    成交价（再减临时冲击）：          S̃_k = S_{{k−1}} − ε·sgn(n_k) − η·v_k

    执行成本 = X·S_0 − Σ n_k·S̃_k，其期望与方差为：
        E[Cost]   = ½γX² + ε·Σ|n_k| + (η̃/τ)·Σ n_k²，   η̃ = η − ½γτ
        Var[Cost] = σ²·Σ τ·x_k²

    符号与本课取值：
      X     要卖出的股数                          {p['X']:,.0f}
      T, N  执行天数、分段数                      {p['T']:.0f} 天, {p['N']} 段
      σ     价格波动（元/√天）                    {p['sigma']}
      γ     永久冲击系数                          {p['gamma']:.1e}
      η     临时冲击系数                          {p['eta']:.1e}
      ε     固定成本（约半个价差 + 费用，元/股）  {p['eps']}
    σ = {p['sigma']} 元/√天、股价 {S0:.0f} 元 → 日波动 {p['sigma'] / S0:.1%}，年化约 {p['sigma'] / S0 * np.sqrt(250):.0%}；τ = {tau} 天

    直觉：期望成本里的 Σ n_k² 惩罚"某段卖太多"→ 均匀卖最省；
          方差里的 Σ x_k² 惩罚"长期拿着大头寸"→ 一开始卖完最稳。
    """)


# ==================== 第三部分：闭式解与数值检验 ====================


def ac_kappa(T, N, sigma, eta, gamma, lam):
    tau = T / N
    eta_t = eta - 0.5 * gamma * tau
    if lam == 0:
        return 0.0
    return np.arccosh(1 + 0.5 * lam * sigma**2 / eta_t * tau**2) / tau


def ac_trajectory(X, T, N, sigma, eta, gamma, lam, **_):
    """Almgren-Chriss 离散最优轨迹，返回长度 N+1 的剩余持仓数组"""
    t = np.arange(N + 1) * (T / N)
    if lam == 0:
        return X * (1 - t / T)  # 风险中性：线性（TWAP）
    kappa = ac_kappa(T, N, sigma, eta, gamma, lam)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


def ac_cost(x, T, sigma, eta, gamma, eps=0.0, **_):
    """给定轨迹，返回 (期望成本, 成本方差)"""
    N = len(x) - 1
    tau = T / N
    n = -np.diff(x)
    eta_t = eta - 0.5 * gamma * tau
    E = 0.5 * gamma * x[0] ** 2 + eps * np.abs(n).sum() + eta_t / tau * (n**2).sum()
    V = sigma**2 * tau * (x[1:] ** 2).sum()
    return E, V


def numerical_optimum(X, T, N, sigma, eta, gamma, lam, **_):
    """
    直接最小化 E + λV（对 x_1..x_{N−1} 是二次函数），一阶条件是三对角线性方程组：
        (η̃/τ)(2x_j − x_{j−1} − x_{j+1}) + λσ²τ·x_j = 0
    """
    tau = T / N
    eta_t = eta - 0.5 * gamma * tau
    a = eta_t / tau
    b = lam * sigma**2 * tau
    m = N - 1
    ab = np.zeros((3, m))
    ab[0, 1:] = -a
    ab[1, :] = 2 * a + b
    ab[2, :-1] = -a
    rhs = np.zeros(m)
    rhs[0] = a * X  # x_0 = X 移到右边
    inner = solve_banded((1, 1), ab, rhs)
    return np.concatenate([[X], inner, [0.0]])


def closed_form_check():
    print("\n" + "=" * 60)
    print("第三部分：闭式解（sinh 轨迹）与数值检验")
    print("=" * 60)
    p = PARAMS
    tau = p["T"] / p["N"]
    print("""
    最优剩余持仓：x_j = X · sinh(κ(T − t_j)) / sinh(κT)
    其中 κ 满足 (2/τ²)(cosh(κτ) − 1) = λσ²/η̃，即 κ = arccosh(1 + λσ²τ²/(2η̃)) / τ
    τ → 0 时 κ ≈ √(λσ²/η)
    """)
    rows = []
    for lam in [1e-7, 1e-6, 2e-6, 1e-5]:
        x_cf = ac_trajectory(lam=lam, **p)
        x_num = numerical_optimum(lam=lam, **p)
        kappa = ac_kappa(p["T"], p["N"], p["sigma"], p["eta"], p["gamma"], lam)
        approx = np.sqrt(lam * p["sigma"] ** 2 / p["eta"])
        rows.append([lam, kappa, approx, np.abs(x_cf - x_num).max()])
    print(pd.DataFrame(rows, columns=["λ", "κ（精确离散）", "κ≈√(λσ²/η)", "闭式解 vs 数值解 最大差(股)"])
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("    ✅ 闭式解与直接求解一阶条件的结果一致（差异在浮点误差量级）")

    x_lin = ac_trajectory(lam=0, **p)
    x_small = ac_trajectory(lam=1e-12, **p)
    print(f"    ✅ λ → 0 的极限：λ=1e-12 的轨迹与线性（TWAP）轨迹最大差 {np.abs(x_small - x_lin).max():.2f} 股")

    # 检验扰动：在最优轨迹上随机扰动，目标函数只会变大
    lam = 2e-6
    x_opt = ac_trajectory(lam=lam, **p)
    E0, V0 = ac_cost(x_opt, **p)
    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(200):
        x = x_opt.copy()
        x[1:-1] += rng.normal(0, 5000, p["N"] - 1)
        x[1:-1] = np.clip(x[1:-1], 0, p["X"])
        x[1:-1] = -np.sort(-x[1:-1])  # 保持单调递减（只卖不买）
        E, V = ac_cost(x, **p)
        worse += (E + lam * V) >= (E0 + lam * V0) - 1e-6
    print(f"    ✅ 对 λ={lam:g} 的最优轨迹随机扰动 200 次，目标函数不降低的次数：{worse}/200")
    print(f"    τ = {tau} 天时，精确 κ 与近似式的差别很小；τ 较大（分段少）时要用精确式")


# ==================== 第四部分：蒙特卡洛验证期望与方差 ====================


def simulate_execution(x, T, sigma, eta, gamma, eps, n_paths=20_000, seed=1, **_):
    """按模型的价格动态模拟执行，返回每条路径的执行成本 X·S0 − Σ n_k·S̃_k"""
    rng = np.random.default_rng(seed)
    N = len(x) - 1
    tau = T / N
    n = -np.diff(x)
    xi = rng.standard_normal((n_paths, N))
    S = np.full(n_paths, S0)
    revenue = np.zeros(n_paths)
    for k in range(N):
        v = n[k] / tau
        revenue += n[k] * (S - eps * np.sign(n[k]) - eta * v)  # 成交价 S̃ 用的是上一时刻价格
        S = S + sigma * np.sqrt(tau) * xi[:, k] - tau * gamma * v
    return x[0] * S0 - revenue


def monte_carlo_check():
    print("\n" + "=" * 60)
    print("第四部分：蒙特卡洛验证期望与方差公式")
    print("=" * 60)
    p = PARAMS
    rows = []
    for lam in [0, 2e-6, 2e-5]:
        x = ac_trajectory(lam=lam, **p)
        E, V = ac_cost(x, **p)
        cost = simulate_execution(x, **p)
        se = cost.std() / np.sqrt(len(cost))
        rows.append([lam, E, cost.mean(), se, (cost.mean() - E) / se, np.sqrt(V), cost.std()])
    table = pd.DataFrame(rows, columns=["λ", "公式 E", "模拟均值", "模拟标准误", "偏差/标准误", "公式 Std", "模拟 Std"])
    print(table.to_string(index=False, formatters={
        "λ": "{:.0e}".format, "公式 E": "{:,.0f}".format, "模拟均值": "{:,.0f}".format, "模拟标准误": "{:,.0f}".format,
        "偏差/标准误": "{:+.2f}".format, "公式 Std": "{:,.0f}".format, "模拟 Std": "{:,.0f}".format,
    }))
    ok = (table["偏差/标准误"].abs() < 3).all()
    std_err = (table["模拟 Std"] / table["公式 Std"] - 1).abs().max()
    print(f"""
    {"✅" if ok else "⚠️"} 模拟均值与公式 E 的偏差都在 3 个标准误以内：{ok}；模拟 Std 与公式 Std 的最大相对差 {std_err:.1%}
       （成本以"元"计：卖出 100 万股、股价 50 元，总价值 5000 万元，20,000 条路径）
    """)


# ==================== 第五部分：有效前沿 ====================


def efficient_frontier():
    print("\n" + "=" * 60)
    print("第五部分：不同风险厌恶下的最优轨迹与有效前沿")
    print("=" * 60)
    p = PARAMS
    value = p["X"] * S0
    rows = []
    for lam in [0, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4]:
        x = ac_trajectory(lam=lam, **p)
        E, V = ac_cost(x, **p)
        kappa = ac_kappa(p["T"], p["N"], p["sigma"], p["eta"], p["gamma"], lam)
        rows.append([lam, kappa, 1 / kappa if kappa > 0 else np.inf, 1 - x[p["N"] // 5] / p["X"],
                     E / value * 1e4, np.sqrt(V) / value * 1e4])
    table = pd.DataFrame(rows, columns=["λ", "κ", "半衰期 1/κ(天)", "第1天完成", "E(bp)", "Std(bp)"])
    print(table.to_string(index=False, formatters={
        "λ": "{:.0e}".format, "κ": "{:.3f}".format, "半衰期 1/κ(天)": "{:.2f}".format,
        "第1天完成": "{:.1%}".format, "E(bp)": "{:.2f}".format, "Std(bp)": "{:.2f}".format,
    }))
    print("""
    💡 解读：
       - λ = 0（风险中性）：κ = 0，轨迹是直线 —— 就是 TWAP，期望成本最低但风险最大
       - λ 增大：κ 增大，半衰期缩短，轨迹"前倾"（早卖多卖）→ 期望成本上升、标准差下降
       - 半衰期 1/κ ≪ T 时，大部分交易在早期完成；此时给更长的执行窗口也没用
    """)

    lams = np.logspace(-9, -3.5, 60)
    frontier = []
    for lam in lams:
        x = ac_trajectory(lam=lam, **p)
        E, V = ac_cost(x, **p)
        frontier.append((np.sqrt(V) / value * 1e4, E / value * 1e4))
    frontier = np.array(frontier)
    mono = np.all(np.diff(frontier[:, 0]) < 0) and np.all(np.diff(frontier[:, 1]) > 0)
    print(f"    ✅ 有效前沿检验：λ 从 1e-9 到 3e-4，标准差单调下降、期望成本单调上升：{mono}")

    # 非最优策略落在前沿的右上方
    t = np.arange(p["N"] + 1) * (p["T"] / p["N"])
    others = {}
    for days in [1, 2, 3]:
        x = p["X"] * np.clip(1 - t / days, 0, 1)  # 在前 days 天内匀速卖完
        E, V = ac_cost(x, **p)
        others[f"{days} 天内匀速卖完"] = (np.sqrt(V) / value * 1e4, E / value * 1e4)
    for name, (sd, e) in others.items():
        # 同样风险下，前沿上的最低期望成本
        e_front = np.interp(sd, frontier[::-1, 0], frontier[::-1, 1])
        print(f"    {name}：Std {sd:.2f} bp，E {e:.2f} bp；同风险下有效前沿的 E = {e_front:.2f} bp")
    print("    → 简单的'缩短 TWAP 窗口'不在前沿上：同样的风险，最优轨迹的期望成本更低")
    return frontier, others


# ==================== 可视化 ====================


def plot_all(frontier, others):
    setup_chinese_font()
    p = PARAMS
    value = p["X"] * S0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    t = np.arange(p["N"] + 1) * (p["T"] / p["N"])
    for lam in [0, 1e-6, 2e-6, 1e-5, 1e-4]:
        x = ac_trajectory(lam=lam, **p)
        ax.plot(t, x / 1e4, marker=".", ms=3, label=f"λ = {lam:g}" + ("（TWAP）" if lam == 0 else ""))
    ax.set_title("Almgren-Chriss 最优剩余持仓")
    ax.set_xlabel("时间（天）")
    ax.set_ylabel("剩余持仓（万股）")
    ax.legend()

    ax = axes[1]
    ax.plot(frontier[:, 0], frontier[:, 1], color="tab:blue", label="有效前沿")
    for lam, mk in [(0, "o"), (2e-6, "s"), (1e-5, "^")]:
        E, V = ac_cost(ac_trajectory(lam=lam, **p), **p)
        ax.scatter(np.sqrt(V) / value * 1e4, E / value * 1e4, marker=mk, s=60, zorder=3, label=f"λ = {lam:g}")
    for name, (sd, e) in others.items():
        ax.scatter(sd, e, marker="x", color="gray", s=50, zorder=3)
        ax.annotate(name, (sd, e), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_title("执行的有效前沿")
    ax.set_xlabel("成本标准差（bp）")
    ax.set_ylabel("期望成本（bp）")
    ax.legend()

    path = save_figure(fig, "07_almgren_chriss.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    tradeoff_intuition()
    model_setup()
    closed_form_check()
    monte_carlo_check()
    frontier, others = efficient_frontier()
    plot_all(frontier, others)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把波动率 σ 翻倍，保持 λ 不变，最优轨迹和半衰期如何变化？用 κ ≈ √(λσ²/η) 解释
    2. 改成买入程序（X 为要买入的股数），推导并实现对应的价格动态与成本，验证轨迹形状不变
    3. 用 06 课的模拟市场执行 AC 轨迹（把天换成分钟），与 TWAP / VWAP 比较 IS 的均值与标准差
    4. 线性临时冲击是简化：把 h(v) 换成 η·v^0.5（平方根），用 scipy.optimize 数值求解最优轨迹并与 sinh 解比较
    """)


if __name__ == "__main__":
    main()
