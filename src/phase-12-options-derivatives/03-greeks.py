"""
希腊字母：Delta、Gamma、Vega、Theta
====================================

学习目标：
    1. 实现 BS 希腊字母的解析公式，统一单位（Vega 每 1 个波动率点、Theta 每天）
    2. 用有限差分校验解析公式（工程上最好用的"单元测试"）
    3. 画出各希腊字母随标的价格、剩余期限的变化，建立数值直觉
    4. 用泰勒展开近似期权价格变化，理解 Gamma 与 Theta 的权衡
    5. 计算期权组合的希腊字母（可加性），并做 Delta / Gamma 中性化

核心概念：
    - Delta = ∂V/∂S，Gamma = ∂²V/∂S²，Vega = ∂V/∂σ，Theta = ∂V/∂t，Rho = ∂V/∂r
    - ΔV ≈ Δ·dS + ½·Γ·dS² + Θ·dt + ν·dσ
    - 组合希腊字母 = Σ 张数 × 合约单位 × 单份希腊字母

前置知识：
    - 02-black-scholes.py
    - CONCEPT.md 第 5 节

环境要求：
    - numpy、pandas、scipy、matplotlib（本课不需要网络）
"""

import sys
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

S0, K, T, R, SIGMA = 3.0, 3.0, 30 / 365, 0.02, 0.20
MULTIPLIER = 10000  # 合约单位（示例，以交易所最新规则为准）


def bs_price(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, kind="call"):
    """
    解析希腊字母（实务单位）：
        delta、gamma：原始单位
        vega：σ 变动 1 个波动率点（0.01）时的价格变化
        theta：每过 1 个自然日的价格变化
        rho：r 变动 1%（0.01）时的价格变化
    """
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * sqrt_t)
    vega = S * pdf * sqrt_t / 100
    if kind == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf * sigma / (2 * sqrt_t) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf * sigma / (2 * sqrt_t) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ==================== 第一部分：解析公式 ====================


def analytic_greeks():
    """希腊字母总览与解析值"""
    print("=" * 60)
    print("第一部分：希腊字母 —— 期权价格的仪表盘")
    print("=" * 60)

    print("""
    ┌──────────┬───────────┬──────────────────────────────┬──────────────┐
    │ 希腊字母 │ 定义      │ 含义                         │ 开车类比     │
    ├──────────┼───────────┼──────────────────────────────┼──────────────┤
    │ Delta Δ  │ ∂V/∂S     │ 标的涨 1 元，期权涨多少      │ 速度         │
    │ Gamma Γ  │ ∂²V/∂S²   │ 标的涨 1 元，Delta 变多少    │ 加速度       │
    │ Vega ν   │ ∂V/∂σ     │ 波动率升 1 点，期权涨多少    │ 路况颠簸     │
    │ Theta Θ  │ ∂V/∂t     │ 过 1 天，期权跌多少          │ 每天的油耗   │
    │ Rho ρ    │ ∂V/∂r     │ 利率升 1%，期权涨多少        │ 短期限不重要 │
    └──────────┴───────────┴──────────────────────────────┴──────────────┘

    公式（φ 为标准正态密度）：
        Call Δ = N(d1)，Put Δ = N(d1) - 1
        Γ = φ(d1) / (S·σ·√T)                  （认购认沽相同）
        ν = S·φ(d1)·√T                         （认购认沽相同，σ 变动 1.00）
        Call Θ = -S·φ(d1)·σ/(2√T) - r·K·e^(-rT)·N(d2)      （每年）
        Put  Θ = -S·φ(d1)·σ/(2√T) + r·K·e^(-rT)·N(-d2)

    ⚠️ 单位陷阱：公式里的 Vega 对应 σ 变动 1.00（100 个点），Theta 是"每年"。
       实务报价用"每 1 个波动率点"（÷100）和"每天"（÷365 或 ÷252），不统一会差几百倍。
    """)

    rows = []
    for k in (2.8, 3.0, 3.2):
        for kind in ("call", "put"):
            g = bs_greeks(S0, k, T, R, SIGMA, kind)
            rows.append({"类型": kind, "K": k, "价格": bs_price(S0, k, T, R, SIGMA, kind), **g})
    df = pd.DataFrame(rows)
    print(f"S={S0}, T={T * 365:.0f} 天, r={R:.0%}, σ={SIGMA:.0%}（vega/波动率点，theta/天，rho/1%）：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    gc, gp = bs_greeks(S0, K, T, R, SIGMA, "call"), bs_greeks(S0, K, T, R, SIGMA, "put")
    print(f"\n   同一行权价：Δcall - Δput = {gc['delta'] - gp['delta']:.6f}（平价关系 → 恒为 1）")
    print(f"               Γ、ν 相同：Γ 差 {gc['gamma'] - gp['gamma']:.1e}，ν 差 {gc['vega'] - gp['vega']:.1e}")
    c = bs_price(S0, K, T, R, SIGMA, "call")
    print(f"\n   平值认购每天 Theta = {gc['theta']:.5f}，约占权利金的 {abs(gc['theta']) / c:.1%}/天")
    print(f"   一张合约：Delta 相当于 {gc['delta'] * MULTIPLIER:,.0f} 份 ETF，"
          f"Vega = {gc['vega'] * MULTIPLIER:,.1f} 元/波动率点，Theta = {gc['theta'] * MULTIPLIER:,.1f} 元/天")


# ==================== 第二部分：有限差分校验 ====================


def finite_difference_check():
    """解析公式 vs 有限差分"""
    print("\n" + "=" * 60)
    print("第二部分：有限差分校验解析公式")
    print("=" * 60)

    print("""
    中心差分：f'(x) ≈ [f(x+h) - f(x-h)] / 2h，f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    Theta 用"真实地过一天"：V(T - 1/365) - V(T)
    """)
    rows = []
    for kind in ("call", "put"):
        g = bs_greeks(S0, K, T, R, SIGMA, kind)
        f = lambda s=S0, t=T, r=R, v=SIGMA: bs_price(s, K, t, r, v, kind)  # noqa: E731
        h = 1e-4
        fd = {
            "delta": (f(s=S0 + h) - f(s=S0 - h)) / (2 * h),
            "gamma": (f(s=S0 + h) - 2 * f() + f(s=S0 - h)) / h**2,
            "vega": (f(v=SIGMA + 0.005) - f(v=SIGMA - 0.005)),  # 中心差分 ±0.5 点 = 每 1 点
            "theta": f(t=T - 1 / 365) - f(),
            "rho": (f(r=R + 0.005) - f(r=R - 0.005)),
        }
        for name in fd:
            rows.append([kind, name, g[name], fd[name], abs(fd[name] / g[name] - 1)])
    df = pd.DataFrame(rows, columns=["类型", "希腊字母", "解析", "有限差分", "相对误差"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("""
    💡 Theta 的相对误差最大：解析 Theta 是瞬时导数，有限差分是"真实过一天"的离散变化；
       临近到期时 Theta 变化很快，两者差异会更大。
    """)

    print("步长 h 对 Gamma 数值精度的影响（截断误差 ∝ h²，舍入误差 ∝ ε/h²）：")
    g_exact = bs_greeks(S0, K, T, R, SIGMA)["gamma"]
    rows = []
    for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        fd = (bs_price(S0 + h, K, T, R, SIGMA) - 2 * bs_price(S0, K, T, R, SIGMA) + bs_price(S0 - h, K, T, R, SIGMA)) / h**2
        rows.append([h, fd, abs(fd / g_exact - 1)])
    print(pd.DataFrame(rows, columns=["h", "FD Gamma", "相对误差"]).to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print("   → h 太大有截断误差，太小被浮点舍入误差淹没，二阶导数尤其敏感")

    ref = op.bs_greeks(S0, K, T, R, SIGMA, "call")
    mine = bs_greeks(S0, K, T, R, SIGMA, "call")
    diff = max(abs(ref["delta"] - mine["delta"]), abs(ref["gamma"] - mine["gamma"]),
               abs(ref["vega"] / 100 - mine["vega"]), abs(ref["theta"] / 365 - mine["theta"]))
    print(f"\n   与共享模块 option_pricing.bs_greeks（原始单位）换算后对比：最大差 = {diff:.1e}")


# ==================== 第三部分：希腊字母随 S 和 T 的变化 ====================


def greeks_profiles():
    """画图：希腊字母随标的价格、剩余期限变化"""
    print("\n" + "=" * 60)
    print("第三部分：希腊字母随标的价格与剩余期限的变化")
    print("=" * 60)

    setup_chinese_font()
    S = np.linspace(2.5, 3.5, 301)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for days, color in [(90, "tab:blue"), (30, "tab:orange"), (7, "tab:red")]:
        g = bs_greeks(S, K, days / 365, R, SIGMA, "call")
        for ax, name in zip(axes.ravel(), ["delta", "gamma", "vega", "theta"]):
            ax.plot(S, g[name], color=color, label=f"剩余 {days} 天")
    titles = {"delta": "Delta（认购）", "gamma": "Gamma", "vega": "Vega（每波动率点）", "theta": "Theta（每天，认购）"}
    for ax, name in zip(axes.ravel(), titles):
        ax.axvline(K, color="gray", ls="--", lw=0.8)
        ax.set_title(titles[name])
        ax.set_xlabel("标的价格 S")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"希腊字母随标的价格变化（K={K}, σ={SIGMA:.0%}）")
    fig.tight_layout()
    path1 = save_figure(fig, "03_greeks_vs_spot.png", subdir="phase-12")

    days = np.arange(1, 181)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, label in [(2.8, "实值 K=2.8"), (3.0, "平值 K=3.0"), (3.2, "虚值 K=3.2")]:
        g = bs_greeks(S0, k, days / 365, R, SIGMA, "call")
        for ax, name in zip(axes, ["gamma", "vega", "theta"]):
            ax.plot(days, g[name], label=label)
    for ax, name in zip(axes, ["Gamma", "Vega（每波动率点）", "Theta（每天）"]):
        ax.set_title(f"认购 {name} 随剩余期限变化")
        ax.set_xlabel("剩余天数")
        ax.invert_xaxis()  # 从左到右 = 时间流逝、临近到期
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = save_figure(fig, "03_greeks_vs_time.png", subdir="phase-12")

    rows = []
    for d in [90, 30, 7, 2]:
        g = bs_greeks(S0, K, d / 365, R, SIGMA, "call")
        rows.append([d, g["delta"], g["gamma"], g["vega"], g["theta"]])
    df = pd.DataFrame(rows, columns=["剩余天数", "Delta", "Gamma", "Vega", "Theta/天"])
    print("平值认购的希腊字母随到期临近的变化：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"   2 天 vs 90 天：Gamma 放大 {df['Gamma'].iloc[-1] / df['Gamma'].iloc[0]:.1f} 倍，"
          f"Vega 缩小到 {df['Vega'].iloc[-1] / df['Vega'].iloc[0]:.1%}，"
          f"Theta 放大 {df['Theta/天'].iloc[-1] / df['Theta/天'].iloc[0]:.1f} 倍")
    print("""
    💡 规律（与 CONCEPT.md 5.3 节对应）：
       - Delta：平值附近从 0 陡增到 1，临近到期变成"台阶"
       - Gamma / Theta：集中在平值附近，临近到期时又尖又高 → 平值短期期权"最刺激"
       - Vega：平值、长期限最大，临近到期趋于 0 → 押注波动率变化要用较长期限的期权
    """)
    print(f"📈 图已保存：{path1}")
    print(f"📈 图已保存：{path2}")


# ==================== 第四部分：泰勒展开与 Gamma-Theta 权衡 ====================


def taylor_approximation():
    """ΔV ≈ Δ·dS + ½Γ·dS² + Θ·dt + ν·dσ"""
    print("\n" + "=" * 60)
    print("第四部分：泰勒展开 —— 用希腊字母估算价格变化")
    print("=" * 60)

    g = bs_greeks(S0, K, T, R, SIGMA, "call")
    v0 = bs_price(S0, K, T, R, SIGMA, "call")
    rows = []
    for ds_pct, dvol in [(0.01, 0), (-0.01, 0), (0.03, 0), (-0.03, 0), (0.0, 0.02), (-0.05, 0.05)]:
        dS = S0 * ds_pct
        actual = bs_price(S0 + dS, K, T - 1 / 365, R, SIGMA + dvol, "call") - v0
        delta_only = g["delta"] * dS
        taylor = delta_only + 0.5 * g["gamma"] * dS**2 + g["theta"] + g["vega"] * dvol * 100
        rows.append([f"{ds_pct:+.0%}", f"{dvol * 100:+.0f}", actual, delta_only, taylor, taylor - actual])
    df = pd.DataFrame(rows, columns=["标的变动", "波动率变动(点)", "完全重估", "仅 Delta", "Δ+Γ+Θ+ν", "近似误差"])
    print(f"平值认购、持有 1 天后的价格变化（初始价格 {v0:.5f}）：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    be_move = np.sqrt(-2 * g["theta"] / g["gamma"])
    print(f"""
    Delta 中性时：ΔV ≈ ½·Γ·dS² + Θ·dt
       Gamma 收益 ½·Γ·dS² 总是 ≥ 0（无论涨跌），Theta 每天约 {g['theta']:+.5f}
       盈亏平衡的单日波动：|dS| = √(-2Θ/Γ) = {be_move:.4f} 元 = {be_move / S0:.2%}
       对照：σ/√365 = {SIGMA / np.sqrt(365):.2%}  → 两者几乎相等不是巧合，
       BS 偏微分方程 Θ + ½σ²S²Γ + rSΔ - rV = 0 说明 Theta 就是为 Gamma 支付的"租金"
       07 课会把这个关系扩展为 Delta 对冲盈亏公式。
    """)


# ==================== 第五部分：组合希腊字母 ====================


def portfolio_greeks():
    """组合希腊字母的可加性与中性化"""
    print("\n" + "=" * 60)
    print("第五部分：组合希腊字母可以相加")
    print("=" * 60)

    book = [  # (类型, 行权价, 剩余天数, 张数)：正数为买入，负数为卖出
        ("call", 3.0, 30, +20),
        ("call", 3.2, 30, -30),
        ("put", 2.8, 30, -15),
        ("put", 3.0, 60, +10),
    ]
    rows = []
    for kind, k, d, n in book:
        g = bs_greeks(S0, k, d / 365, R, SIGMA, kind)
        rows.append({"头寸": f"{'买' if n > 0 else '卖'} {abs(n)} 张 {kind} K={k} {d}天",
                     **{name: n * MULTIPLIER * g[name] for name in ("delta", "gamma", "vega", "theta")}})
    df = pd.DataFrame(rows).set_index("头寸")
    df.loc["组合合计"] = df.sum()
    print("组合希腊字母 = Σ 张数 × 合约单位 × 单份希腊字母：")
    print(df.to_string(float_format=lambda x: f"{x:,.2f}"))

    def book_value(s, vol=SIGMA, day_shift=0):
        return sum(n * MULTIPLIER * bs_price(s, k, (d - day_shift) / 365, R, vol, kind) for kind, k, d, n in book)

    h = 1e-3
    fd_delta = (book_value(S0 + h) - book_value(S0 - h)) / (2 * h)
    fd_gamma = (book_value(S0 + h) - 2 * book_value(S0) + book_value(S0 - h)) / h**2
    print(f"\n   对整个组合直接做有限差分：Delta = {fd_delta:,.2f}，Gamma = {fd_gamma:,.2f}（与逐项相加一致）")

    tot = df.loc["组合合计"]
    print(f"""
    风险报告常用换算：
       Delta 金额 = Delta × S = {tot['delta'] * S0:,.0f} 元（相当于持有这么多市值的 ETF）
       1% Gamma  = Γ × S² × 1% = {tot['gamma'] * S0**2 * 0.01:,.0f} 元（标的涨 1% 时 Delta 金额的变化）
       Vega      = {tot['vega']:,.0f} 元/波动率点，Theta = {tot['theta']:,.0f} 元/天
    """)

    # Gamma 中性：用另一个期权抵消 Gamma；再用 ETF 抵消 Delta（ETF 的 Gamma 为 0）
    hedge = bs_greeks(S0, 3.1, 30 / 365, R, SIGMA, "call")
    n_opt = -tot["gamma"] / (hedge["gamma"] * MULTIPLIER)
    delta_after = tot["delta"] + n_opt * MULTIPLIER * hedge["delta"]
    print("   中性化步骤：")
    print(f"   ① 先用 30 天 K=3.1 认购抵消 Gamma：需要 {n_opt:+.2f} 张（正数买入）")
    print(f"   ② 再用 ETF 抵消剩余 Delta {delta_after:,.0f}：需要 {-delta_after:+,.0f} 份 ETF")
    print(f"   ③ 结果：Gamma ≈ 0、Delta ≈ 0，但 Vega = {tot['vega'] + n_opt * MULTIPLIER * hedge['vega']:,.0f}、"
          f"Theta = {tot['theta'] + n_opt * MULTIPLIER * hedge['theta']:,.0f} 仍然存在")
    print("""
    💡 为什么要先中性 Gamma 再中性 Delta？ETF 没有 Gamma，只能调 Delta；
       而用期权调 Gamma 会同时改变 Delta，所以顺序是"先期权调 Gamma，后标的调 Delta"。
       ⚠️ 不同期限的 Vega 直接相加会掩盖期限结构风险，实务中按到期日分桶管理。
    """)


# ==================== 主函数 ====================


def main():
    analytic_greeks()
    finite_difference_check()
    greeks_profiles()
    taylor_approximation()
    portfolio_greeks()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 推导并实现二阶希腊字母 Vanna（∂Δ/∂σ）、Volga（∂ν/∂σ）、Charm（∂Δ/∂t），用有限差分校验
    2. 画出平值认购 Theta 在 Theta/365 与 Theta/252 两种口径下的差别，说明各自适用场景
    3. 修改第五部分的组合，使它同时满足 Delta、Gamma、Vega 中性（需要几种对冲工具？）
    4. 第四部分中，当标的单日下跌 8% 时泰勒展开的误差有多大？为什么大行情下要用完全重估？
    """)


if __name__ == "__main__":
    main()
