"""
Black-Scholes 定价与二叉树
==========================

学习目标：
    1. 用"复制组合"理解期权定价的核心：无套利、风险中性概率
    2. 实现 CRR 二叉树，为欧式和美式期权定价，理解提前行权
    3. 用 scipy.stats.norm 实现 Black-Scholes 公式，理解 N(d1)、N(d2) 的含义
    4. 实现蒙特卡洛定价，并演示三种方法收敛到同一个价格

核心概念：
    - 复制：股票 + 借贷 → 与期权到期收益完全相同 → 价格必须相同
    - 风险中性定价：价格 = 风险中性期望收益的贴现，μ 不出现在公式里
    - BS：C = S·N(d1) - K·e^(-rT)·N(d2)
    - CRR：u = e^(σ√Δt)，d = 1/u，p = (e^(rΔt) - d)/(u - d)

前置知识：
    - 01-options-basics.py
    - Phase 3：正态分布、蒙特卡洛模拟
    - CONCEPT.md 第 3、4 节（含 BS 公式的完整推导）

环境要求：
    - numpy、pandas、scipy、matplotlib（本课不需要网络）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm

import option_pricing as op
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

S0, K, T, R, SIGMA = 3.0, 3.0, 0.25, 0.02, 0.20


# ==================== 第一部分：一步二叉树与复制 ====================


def one_step_replication():
    """用股票 + 借贷复制期权"""
    print("=" * 60)
    print("第一部分：一步二叉树 —— 用复制给期权定价")
    print("=" * 60)

    S, Su, Sd, k, r = 100.0, 120.0, 90.0, 100.0, 0.05
    Cu, Cd = max(Su - k, 0), max(Sd - k, 0)
    print(f"""
                    ┌── S = {Su:.0f}  →  Call = {Cu:.0f}
          S = {S:.0f} ───┤
                    └── S = {Sd:.0f}   →  Call = {Cd:.0f}
    一年期，无风险利率 {r:.0%}（离散复利），行权价 {k:.0f}

    复制：买 Δ 股、借款 B，要求两种情形下组合价值 = 期权价值
        {Su:.0f}·Δ - {1 + r}·B = {Cu:.0f}
        {Sd:.0f}·Δ  - {1 + r}·B = {Cd:.0f}
    """)
    delta = (Cu - Cd) / (Su - Sd)
    B = (Sd * delta - Cd) / (1 + r)
    C_rep = S * delta - B
    print(f"   解得 Δ = {delta:.4f}，B = {B:.4f} → 期权价格 = 复制成本 = {C_rep:.4f}")

    u, d = Su / S, Sd / S
    p = ((1 + r) - d) / (u - d)
    C_rn = (p * Cu + (1 - p) * Cd) / (1 + r)
    print(f"   风险中性概率 p = ((1+r) - d)/(u - d) = {p:.4f}")
    print(f"   风险中性定价：C = [p·Cu + (1-p)·Cd]/(1+r) = {C_rn:.4f}（与复制法一致）")

    # 真实上涨概率不影响价格
    print("\n   如果你认为真实上涨概率是 0.8 或 0.2，期权价格会变吗？")
    for p_real in [0.2, 0.5, 0.8]:
        exp_payoff = p_real * Cu + (1 - p_real) * Cd
        print(f"     真实概率 {p_real:.1f}：期望收益 = {exp_payoff:.2f}，但复制成本仍是 {C_rep:.4f}")
    print("""
    💡 只要能用标的 + 借贷复制期权，价格就由复制成本决定，与"你认为涨的概率"无关。
       风险中性概率 p 只是计算工具：在这个虚拟世界里所有资产的期望收益都等于 r。
       Δ 就是对冲比例，也就是 03 课的 Delta。
    """)


# ==================== 第二部分：CRR 多步二叉树 ====================


def crr_price(S, K, T, r, sigma, N=200, kind="call", american=False):
    """CRR 二叉树：从到期日逐步倒推；美式期权在每个节点比较"继续持有"与"立即行权\""""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(N + 1)
    ST = S * u ** (N - j) * d**j  # 终端节点价格（从高到低），u·d=1 所以树会"重合"
    V = np.maximum(sign * (ST - K), 0.0)
    for step in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])  # 风险中性期望再贴现
        if american:
            jj = np.arange(step + 1)
            V = np.maximum(V, sign * (S * u ** (step - jj) * d**jj - K))
    return float(V[0])


def crr_tree():
    """多步二叉树：欧式与美式"""
    print("\n" + "=" * 60)
    print("第二部分：CRR 多步二叉树")
    print("=" * 60)

    print("""
    把 [0, T] 分成 N 步，每步 Δt = T/N：
        u = e^(σ√Δt)，d = 1/u，p = (e^(rΔt) - d)/(u - d)
    从到期日的收益开始，每个节点价值 = 两个子节点的风险中性期望 × e^(-rΔt)
    美式期权：每个节点取 max(继续持有价值, 立即行权价值)

                               S·u²
                     S·u ───<
            S ────<            S        ← u·d = 1，N 步只有 N+1 个终端节点
                     S·d ───<
                               S·d²
    """)

    N = 500
    rows = []
    for kind in ("call", "put"):
        for k in (2.7, 3.0, 3.3):
            eu = crr_price(S0, k, T, R, SIGMA, N, kind, american=False)
            am = crr_price(S0, k, T, R, SIGMA, N, kind, american=True)
            rows.append([kind, k, eu, am, am - eu])
    df = pd.DataFrame(rows, columns=["类型", "K", "欧式", "美式", "提前行权溢价"])
    print(f"S={S0}, T={T} 年, r={R:.0%}, σ={SIGMA:.0%}, N={N} 步：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    put_am_high_r = crr_price(S0, 3.3, 1.0, 0.08, SIGMA, N, "put", True)
    put_eu_high_r = crr_price(S0, 3.3, 1.0, 0.08, SIGMA, N, "put", False)
    print(f"\n   换成 1 年期、r=8% 的实值认沽（K=3.3）：欧式 {put_eu_high_r:.5f}，美式 {put_am_high_r:.5f}，"
          f"溢价 {put_am_high_r - put_eu_high_r:.5f}")
    print("""
    💡 观察：
       - 无分红的美式认购 = 欧式认购（提前行权会丢掉时间价值和 K 的利息，永远不划算）
       - 美式认沽 ≥ 欧式认沽：深度实值时提前拿到 K 可以赚利息，利率越高、越实值，溢价越大
       - 国内 ETF 期权是欧式，所以本阶段后面都用 BS 公式；二叉树主要用于美式期权和离散分红
    """)


# ==================== 第三部分：Black-Scholes 公式 ====================


def bs_price(S, K, T, r, sigma, kind="call"):
    """Black-Scholes 欧式期权价格（无分红）"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes():
    """BS 公式：实现、解读与检验"""
    print("\n" + "=" * 60)
    print("第三部分：Black-Scholes 公式")
    print("=" * 60)

    print("""
        C = S·N(d1) - K·e^(-rT)·N(d2)
        P = K·e^(-rT)·N(-d2) - S·N(-d1)
        d1 = [ln(S/K) + (r + σ²/2)·T] / (σ√T)，  d2 = d1 - σ√T

    推导思路（CONCEPT.md 4.3 节）：风险中性下 S_T = S·exp[(r - σ²/2)T + σ√T·Z]，
        C = e^(-rT)·E[(S_T - K)⁺] = e^(-rT)·E[S_T·1{S_T>K}] - K·e^(-rT)·P(S_T>K)
        其中 P(S_T > K) = N(d2)，e^(-rT)·E[S_T·1{S_T>K}] = S·N(d1)

    直觉：持有 N(d1) 份标的 + 借入 K·e^(-rT)·N(d2) 现金 = 连续时间版的"复制组合"
    """)

    d1 = (np.log(S0 / K) + (R + 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)
    c, p = bs_price(S0, K, T, R, SIGMA, "call"), bs_price(S0, K, T, R, SIGMA, "put")
    print(f"S={S0}, K={K}, T={T}, r={R:.0%}, σ={SIGMA:.0%}")
    print(f"   d1 = {d1:.5f}，d2 = {d2:.5f}，N(d1) = {norm.cdf(d1):.5f}，N(d2) = {norm.cdf(d2):.5f}")
    print(f"   Call = {c:.6f}，Put = {p:.6f}")
    print(f"   平价检验：C - P = {c - p:.6f}，S - Ke^(-rT) = {S0 - K * np.exp(-R * T):.6f}")

    # 用模拟验证 N(d2) 是风险中性行权概率
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(1_000_000)
    ST = S0 * np.exp((R - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * Z)
    print(f"\n   模拟 100 万条风险中性路径：P(S_T > K) = {np.mean(ST > K):.5f}  vs  N(d2) = {norm.cdf(d2):.5f}")
    print(f"   e^(-rT)·E[S_T·1(S_T>K)] = {np.exp(-R * T) * np.mean(ST * (ST > K)):.5f}  vs  S·N(d1) = "
          f"{S0 * norm.cdf(d1):.5f}")

    approx = 0.4 * S0 * SIGMA * np.sqrt(T)
    print(f"\n   心算公式（平值、r≈0）：C ≈ 0.4·S·σ·√T = {approx:.5f}，BS 精确值 {c:.5f}")
    print("   （0.4 ≈ 1/√(2π)；r 越大、越偏离平值，近似越差）")

    ref = float(op.bs_price(S0, K, T, R, SIGMA, "call"))
    print(f"\n   与共享模块 option_pricing.bs_price 交叉校验：差 = {abs(c - ref):.2e}")


# ==================== 第四部分：蒙特卡洛定价 ====================


def mc_price(S, K, T, r, sigma, kind="call", n_paths=100_000, seed=0, antithetic=False):
    """风险中性蒙特卡洛：返回 (价格, 标准误)"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
    if antithetic:
        Z = np.concatenate([Z, -Z])  # 对偶变量：Z 与 -Z 成对出现，抵消一部分随机误差
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T) * payoff
    if antithetic:
        half = len(disc) // 2
        pair = 0.5 * (disc[:half] + disc[half:])
        return pair.mean(), pair.std(ddof=1) / np.sqrt(half)
    return disc.mean(), disc.std(ddof=1) / np.sqrt(len(disc))


def monte_carlo():
    """蒙特卡洛：误差按 1/√n 下降"""
    print("\n" + "=" * 60)
    print("第四部分：蒙特卡洛定价")
    print("=" * 60)

    print("""
    步骤：① 在风险中性世界模拟 S_T  ② 计算每条路径的到期收益  ③ 取平均再贴现
    标准误 = 样本标准差 / √n → 精度提高 10 倍需要 100 倍路径
    优点：能处理路径依赖（亚式、障碍）和高维问题；缺点：慢、有随机误差
    """)
    bs = bs_price(S0, K, T, R, SIGMA, "call")
    rows = []
    for i, n in enumerate([1_000, 10_000, 100_000, 1_000_000]):
        for anti in (False, True):
            price, se = mc_price(S0, K, T, R, SIGMA, "call", n, seed=100 + i, antithetic=anti)
            rows.append([n, "是" if anti else "否", price, se, price - bs, abs(price - bs) / se])
    df = pd.DataFrame(rows, columns=["路径数", "对偶变量", "MC 价格", "标准误", "误差", "|误差|/标准误"])
    print(f"BS 精确值 = {bs:.6f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    se_ratio = df.groupby("对偶变量")["标准误"].mean()
    print(f"   对偶变量使平均标准误从 {se_ratio['否']:.6f} 降到 {se_ratio['是']:.6f}（路径数相同）")

    # 单次实验的"误差/标准误"本身是随机的；重复多次才能检验 MC 是否无偏
    n_rep, n_paths = 500, 10_000
    reps = np.array([mc_price(S0, K, T, R, SIGMA, "call", n_paths, seed=s) for s in range(n_rep)])
    z = (reps[:, 0] - bs) / reps[:, 1]
    print(f"\n   重复 {n_rep} 次独立实验（每次 {n_paths:,} 条路径）：")
    print(f"   |误差| < 1.96 倍标准误的比例 = {np.mean(np.abs(z) < 1.96):.1%}（理论值 95%），"
          f"误差/标准误的均值 = {z.mean():+.3f}（理论值 0）")
    print("""
    💡 单次 MC 结果偏离 BS 一两个标准误很正常；重复实验中约 95% 落在 ±1.96 倍标准误以内，
       才说明 MC 估计是无偏的、标准误也算对了。
    """)


# ==================== 第五部分：三种方法收敛到同一价格 ====================


def convergence_demo():
    """二叉树、蒙特卡洛 → BS"""
    print("\n" + "=" * 60)
    print("第五部分：三种方法收敛到同一价格")
    print("=" * 60)

    bs = bs_price(S0, K, T, R, SIGMA, "call")
    Ns = [10, 11, 25, 26, 50, 51, 100, 101, 200, 201, 500, 1000, 2000]
    tree = []
    for N in Ns:
        t0 = time.perf_counter()
        v = crr_price(S0, K, T, R, SIGMA, N)
        tree.append([N, v, v - bs, (time.perf_counter() - t0) * 1000])
    df = pd.DataFrame(tree, columns=["步数 N", "CRR 价格", "误差", "耗时(ms)"])
    print(f"BS = {bs:.6f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("""
    💡 二叉树误差大致按 1/N 下降，并伴随奇偶步数的振荡（平值期权尤其明显：
       行权价落在节点上还是两节点之间会交替变化）。实务中常取相邻两个 N 的平均来消除振荡。
    """)
    avg = 0.5 * (crr_price(S0, K, T, R, SIGMA, 200) + crr_price(S0, K, T, R, SIGMA, 201))
    print(f"   N=200 与 N=201 的平均 = {avg:.6f}，误差 {avg - bs:+.2e}")

    mc_ns = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
    mc = np.array([mc_price(S0, K, T, R, SIGMA, "call", n, seed=7) for n in mc_ns])

    summary = pd.DataFrame({
        "方法": ["Black-Scholes 解析解", "CRR 二叉树 N=2000", "蒙特卡洛 n=1,000,000", "共享模块 crr_price N=2000",
               "共享模块 mc_price(对偶) n=1,000,000"],
        "价格": [bs, crr_price(S0, K, T, R, SIGMA, 2000), mc[-1, 0], op.crr_price(S0, K, T, R, SIGMA, 2000),
               op.mc_price(S0, K, T, R, SIGMA, n_paths=1_000_000, seed=7)[0]],
    })
    summary["与 BS 之差"] = summary["价格"] - bs
    print("\n汇总：")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    n_grid = np.arange(5, 301)
    ax.plot(n_grid, [crr_price(S0, K, T, R, SIGMA, n) for n in n_grid], lw=0.9, label="CRR 二叉树")
    ax.axhline(bs, color="red", ls="--", label=f"BS = {bs:.4f}")
    ax.set_xlabel("步数 N")
    ax.set_ylabel("认购价格")
    ax.set_title("二叉树收敛（注意奇偶振荡）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(mc_ns, mc[:, 0], yerr=1.96 * mc[:, 1], fmt="o-", capsize=4, label="蒙特卡洛 ± 95% 置信区间")
    ax.axhline(bs, color="red", ls="--", label="BS")
    ax.set_xscale("log")
    ax.set_xlabel("路径数 n（对数坐标）")
    ax.set_title("蒙特卡洛收敛：误差 ∝ 1/√n")
    ax.legend()
    ax.grid(alpha=0.3)
    path = save_figure(fig, "02_convergence.png", subdir="phase-12")
    print(f"\n📈 图已保存：{path}")


# ==================== 主函数 ====================


def main():
    one_step_replication()
    crr_tree()
    black_scholes()
    monte_carlo()
    convergence_demo()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 在一步二叉树中把上涨情形改为 130，重新求 Δ、B 和期权价格；再用风险中性概率验证
    2. 给 crr_price 加入连续股息率 q（p = (e^((r-q)Δt) - d)/(u - d)），验证有分红时
       美式认购可能 > 欧式认购
    3. 用蒙特卡洛为"算术平均亚式认购"定价（每天观察一次，取平均价代替 S_T），并与欧式认购比较
    4. 固定 N=200，改变 S 使行权价恰好落在 / 不落在终端节点上，观察二叉树误差的变化
    """)


if __name__ == "__main__":
    main()
