"""
隐含波动率求解
==============

学习目标：
    1. 理解隐含波动率（IV）：把期权价格换算成统一的"波动率报价"
    2. 实现牛顿法求 IV，观察二次收敛
    3. 实现二分法兜底，组合成稳健的求解器，并检查无套利边界
    4. 处理深度实值、深度虚值、临近到期等数值难题
    5. 与 scipy.optimize.brentq 对比精度与速度

核心概念：
    - IV：使 BS(σ) = 市场价格 的 σ；Vega > 0 保证解唯一
    - 牛顿法：σ ← σ - (BS(σ) - C_mkt) / Vega(σ)
    - 无套利边界：max(S - K·e^(-rT), 0) < C < S，超出边界无解，应返回 NaN

前置知识：
    - 02-black-scholes.py、03-greeks.py
    - CONCEPT.md 第 6 节

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
from scipy.optimize import brentq
from scipy.stats import norm

import option_pricing as op
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

R = 0.02
SIG_LO, SIG_HI = 1e-4, 5.0


def bs_price(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    """原始单位的 Vega（σ 变动 1.00），牛顿法要用这个"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ==================== 第一部分：什么是隐含波动率 ====================


def what_is_iv():
    """价格 ↔ 波动率"""
    print("=" * 60)
    print("第一部分：什么是隐含波动率")
    print("=" * 60)

    print("""
         BS 公式（正向）：σ ──────────▶ 期权价格
         求解 IV（反向）：σ ◀────────── 期权价格

    S、K、T、r 都能直接观察，只有 σ 观察不到。给定市场价格，反解出的 σ 就是 IV。
    📌 类比：IV 之于期权 ≈ 到期收益率之于债券 —— 把不同合约的价格换算成可比较的统一尺度。
    """)
    S, T = 3.0, 30 / 365
    sig = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    table = pd.DataFrame({f"K={k}": bs_price(S, k, T, R, sig) for k in (2.8, 3.0, 3.2)},
                         index=pd.Index([f"{s:.0%}" for s in sig], name="σ"))
    print(f"认购价格随 σ 单调递增（S={S}, 30 天）：")
    print(table.round(5).to_string())
    print("""
    💡 单调递增 ⇒ 给定价格，IV 最多只有一个解。
       同一天三个行权价的价格 0.21、0.07、0.01 看不出谁"贵"，换算成 IV 就能直接比较。
    """)


# ==================== 第二部分：牛顿法 ====================


def newton_demo():
    """牛顿法的迭代过程与二次收敛"""
    print("\n" + "=" * 60)
    print("第二部分：牛顿法")
    print("=" * 60)

    print("""
    在当前猜测点作切线，切线与目标价格的交点作为下一个猜测：
        σ_{n+1} = σ_n - [BS(σ_n) - C_mkt] / Vega(σ_n)

    初值：Brenner-Subrahmanyam 近似（平值公式 C ≈ 0.4·S·σ·√T 反解）
        σ_0 ≈ √(2π/T) · C_mkt / S
    （平值附近很好用；第三部分会看到虚值 / 实值合约需要更好的初值）
    """)
    S, K, T, true_sigma = 3.0, 3.1, 0.2, 0.235
    price = bs_price(S, K, T, R, true_sigma)
    sigma = np.sqrt(2 * np.pi / T) * price / S
    print(f"目标：S={S}, K={K}, T={T}, 市场价 {price:.6f}（真实 σ = {true_sigma}）")
    rows = []
    for i in range(6):
        diff = bs_price(S, K, T, R, sigma) - price
        vega = bs_vega(S, K, T, R, sigma)
        rows.append([i, sigma, diff, vega, abs(sigma - true_sigma)])
        if abs(diff) < 1e-14:
            break
        sigma -= diff / vega
    df = pd.DataFrame(rows, columns=["迭代", "σ_n", "BS(σ_n) - C", "Vega", "|σ_n - σ*|"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print("""
    💡 二次收敛：误差大致每步平方（1e-2 → 1e-4 → 1e-8 → …），通常 3–5 步达到机器精度。
       代价：需要导数（Vega），且初值不好或 Vega 太小时可能发散。
    """)


# ==================== 第三部分：稳健求解器 ====================


def implied_vol_bisection(price, S, K, T, r, kind="call", tol=1e-10):
    """二分法：价格对 σ 单调，只要区间端点异号就必然收敛（每步区间减半）"""
    lo, hi = SIG_LO, SIG_HI
    n = 0
    while hi - lo > tol:
        n += 1
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), n


def initial_guess(price, S, K, T, r):
    """
    初值 = max(Brenner 近似, 拐点 σ)：
        Brenner：√(2π/T)·price/S，平值附近很准
        拐点（Manaster-Koehler）：√(2|ln(F/K)|/T)，BS 价格关于 σ 在此由凸变凹，
        从拐点出发的牛顿迭代单调收敛 —— 专门解决虚值 / 实值合约从小 σ 出发时 Vega≈0 的问题
    """
    brenner = np.sqrt(2 * np.pi / T) * price / S
    inflection = np.sqrt(2 * abs(np.log(S * np.exp(r * T) / K)) / T)
    return float(np.clip(max(brenner, inflection), 0.05, 3.0))


def implied_vol(price, S, K, T, r, kind="call", tol=1e-12, max_iter=50):
    """牛顿法 + 二分法兜底；返回 (sigma, 方法, 迭代次数)"""
    lower = max(S - K * np.exp(-r * T), 0.0) if kind == "call" else max(K * np.exp(-r * T) - S, 0.0)
    upper = S if kind == "call" else K * np.exp(-r * T)
    if not (lower < price < upper):
        return np.nan, "无解(违反边界)", 0

    sigma = initial_guess(price, S, K, T, r)
    for i in range(1, max_iter + 1):
        diff = bs_price(S, K, T, r, sigma, kind) - price
        if abs(diff) < tol:
            return sigma, "牛顿", i
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:  # Vega 太小，牛顿步长会爆炸
            break
        step = diff / vega
        sigma -= step
        if not (SIG_LO < sigma < SIG_HI):  # 跑出合理区间
            break
        if abs(step) < 1e-12:  # 价格精度已到浮点极限，σ 不再变化
            return sigma, "牛顿", i
    sigma, n = implied_vol_bisection(price, S, K, T, r, kind)
    if min(sigma - SIG_LO, SIG_HI - sigma) < 1e-6:  # 收敛到搜索区间端点：报价异常，视为无解
        return np.nan, "二分兜底→越界", n
    return sigma, "二分兜底", n


def naive_newton(price, S, K, T, r, kind="call", sigma0=0.2, max_iter=50):
    """没有任何保护的牛顿法（反面教材）：固定初值、不检查 Vega、不限制 σ 范围"""
    sigma = sigma0
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            diff = bs_price(S, K, T, r, sigma, kind) - price
            if abs(diff) < 1e-12:
                return sigma
            vega = bs_vega(S, K, T, r, sigma)
            if vega == 0:
                return np.nan  # 除以 0
            sigma -= diff / vega
            if not np.isfinite(sigma) or sigma <= 0:
                return np.nan  # 跳到负波动率，BS 公式无定义
    return np.nan


def robust_solver():
    """边界检查 + 牛顿 + 二分"""
    print("\n" + "=" * 60)
    print("第三部分：稳健求解器 —— 牛顿法 + 二分法兜底")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. 检查无套利边界：下界 < 价格 < 上界，否则返回 NaN          │
    │    认购：max(S - K·e^(-rT), 0) < C < S                       │
    │    认沽：max(K·e^(-rT) - S, 0) < P < K·e^(-rT)               │
    │ 2. 牛顿法：好初值 + 快速收敛；Vega≈0 或 σ 跑出 [0.0001, 5]   │
    │    时转第 3 步                                               │
    │ 3. 二分法：慢（约 35 步）但一定收敛                          │
    └──────────────────────────────────────────────────────────────┘
    """)
    S = 3.0
    cases = [
        ("普通平值", "call", 3.0, 30, 0.20),
        ("深度虚值认购", "call", 3.6, 30, 0.25),
        ("深度实值认购", "call", 2.4, 30, 0.25),
        ("临近到期平值(1天)", "call", 3.0, 1, 0.20),
        ("临近到期虚值(2天)", "put", 2.85, 2, 0.30),
        ("短期虚值+高波动", "call", 3.3, 10, 0.60),
        ("短期深虚认沽+高波动", "put", 2.6, 5, 0.50),
        ("长期限深度虚值认沽", "put", 2.0, 365, 0.35),
        ("报价贴近下界(σ极小)", "call", 3.0, 30, 0.002),
        ("报价贴近上界(σ>5)", "call", 3.0, 30, 6.0),
    ]
    sigma0s = [0.05, 0.2, 3.0]
    rows = []
    for name, kind, k, d, true_sig in cases:
        t = d / 365
        px = bs_price(S, k, t, R, true_sig, kind)
        naive = [naive_newton(px, S, k, t, R, kind, s0) for s0 in sigma0s]
        sig, method, n = implied_vol(px, S, k, t, R, kind)
        rows.append([name, kind, k, d, px, true_sig, *naive, sig, method, n])
    naive_cols = [f"朴素σ0={s0}" for s0 in sigma0s]
    df = pd.DataFrame(rows, columns=["场景", "类型", "K", "天数", "价格", "真实σ", *naive_cols, "稳健求解", "方法", "迭代"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}", na_rep="失败"))
    for col in naive_cols:
        print(f"   {col}：失败 {int(df[col].isna().sum())} / {len(df)} 个场景")
    print(f"   稳健求解器：失败 {int(df['稳健求解'].isna().sum())} 个，最大误差 = "
          f"{np.nanmax(np.abs(df['稳健求解'] - df['真实σ'])):.1e}")
    print("""
    💡 朴素牛顿法的典型死法：初值 σ 太小 → 虚值合约 Vega≈0 → 除以 0；
       或者一步迈太大 → σ 变成负数。没有"万能固定初值"，所以要么选好初值（拐点），要么兜底。
       最后一行：真实 σ=600% 超出搜索区间 [0.0001, 5]。现实中这样的报价几乎一定是错单或
       陈旧报价，二分法收敛到区间端点后返回 NaN —— "宁可无解，也不输出一个假数字"。
    """)

    print("\n违反无套利边界的报价 → 返回 NaN，而不是硬算一个数：")
    for name, px in [("低于内在价值", 0.59), ("高于标的价格", 3.05)]:
        sig, method, _ = implied_vol(px, S, 2.4, 30 / 365, R, "call")
        print(f"   K=2.4 认购报价 {px}（{name}）→ IV = {sig}，{method}")


# ==================== 第四部分：数值难题 ====================


def numerical_pitfalls():
    """报价精度、深度实值与平价转换"""
    print("\n" + "=" * 60)
    print("第四部分：深度实值、虚值与临近到期的数值问题")
    print("=" * 60)

    print("""
    IV 的误差 ≈ 价格误差 / Vega。Vega 越小，同样的报价误差（如 1 个最小报价单位
    0.0001 元）造成的 IV 误差越大 —— 深度虚值、深度实值、临近到期的 Vega 都很小。
    """)
    S, true_sig, tick = 3.0, 0.22, 1e-4
    rows = []
    for k in (2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4):
        for d in (3, 30):
            t = d / 365
            px = bs_price(S, k, t, R, true_sig, "call")
            vega = bs_vega(S, k, t, R, true_sig)
            if px < tick:  # 理论价格不到 1 个 tick：市场上无法报价，跳过
                continue
            iv_up, _, _ = implied_vol(px + tick, S, k, t, R, "call")
            rows.append([k, d, px, px - max(S - k * np.exp(-R * t), 0), vega / 100, (iv_up - true_sig) * 100])
    df = pd.DataFrame(rows, columns=["K", "天数", "认购价格", "时间价值", "Vega/点", "+1 tick 的 IV 误差(点)"])
    print(f"真实 σ={true_sig:.0%}，价格多报 1 个 tick（{tick}）时 IV 的变化：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("""
    💡 同一行权价的认购与认沽 Vega 相同，同样的价格误差带来同样的 IV 误差。
       但实值合约价格高、成交少，买卖价差的绝对值通常更宽 —— 换算成 IV 就差很多。
       实务做法：用平价关系，同一行权价只用虚值一侧（OTM）的合约算 IV，
       06 课处理真实期权链时就是这样拼接微笑曲线的。
    """)
    k, t = 2.7, 30 / 365
    spreads = {"call": 0.0040, "put": 0.0004}  # 假设的买卖价差：实值认购宽、虚值认沽窄（示例值）
    for kind, label in [("call", "实值认购"), ("put", "虚值认沽")]:
        mid = bs_price(S, k, t, R, true_sig, kind)
        bid, ask = mid - spreads[kind] / 2, mid + spreads[kind] / 2
        iv_bid, _, _ = implied_vol(bid, S, k, t, R, kind)
        iv_ask, _, _ = implied_vol(ask, S, k, t, R, kind)
        bid_txt = "无解" if np.isnan(iv_bid) else f"{iv_bid:.2%}"
        print(f"   K={k} {label}：中间价 {mid:.4f}，假设价差 {spreads[kind]:.4f} → "
              f"买价 IV {bid_txt}，卖价 IV {iv_ask:.2%}（真实 {true_sig:.0%}）")

    setup_chinese_font()
    sig_grid = np.linspace(0.05, 0.8, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, d in [(3.0, 30), (3.3, 30), (3.0, 3), (3.15, 3)]:
        axes[0].plot(sig_grid, [bs_price(S, k, d / 365, R, s) for s in sig_grid], label=f"K={k}, {d} 天")
    axes[0].set_xlabel("σ")
    axes[0].set_ylabel("认购价格")
    axes[0].set_title("价格-波动率曲线：越平坦，IV 越难求、越不准")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for d in (3, 30):
        sub = df[df["天数"] == d]
        axes[1].semilogy(sub["K"], sub["+1 tick 的 IV 误差(点)"].abs(), "o-", label=f"{d} 天")
    axes[1].set_xlabel("行权价 K（S=3.0）")
    axes[1].set_ylabel("IV 误差（波动率点，对数坐标）")
    axes[1].set_title("1 个 tick 报价误差造成的 IV 误差")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    path = save_figure(fig, "04_iv_sensitivity.png", subdir="phase-12")
    print(f"\n📈 图已保存：{path}")


# ==================== 第五部分：与 brentq 对比 ====================


def compare_brentq():
    """自写求解器 vs scipy.optimize.brentq"""
    print("\n" + "=" * 60)
    print("第五部分：与 scipy.optimize.brentq 对比")
    print("=" * 60)

    print("""
    brentq：结合二分、割线和反二次插值，不需要导数，只要区间端点异号就保证收敛。
    它是"兜底法"的工业级版本；牛顿法的优势在于给好初值时步数更少。
    """)
    S = 3.0
    rng = np.random.default_rng(0)
    strikes = np.round(np.arange(2.5, 3.55, 0.05), 2)
    days = np.array([3, 7, 14, 30, 60, 90, 180])
    grid = [(k, d, rng.uniform(0.12, 0.45), kind) for k in strikes for d in days for kind in ("call", "put")]

    prices = [bs_price(S, k, d / 365, R, s, kind) for k, d, s, kind in grid]
    valid = [i for i, (k, d, s, kind) in enumerate(grid)
             if prices[i] > 1e-6 + (max(S - k * np.exp(-R * d / 365), 0) if kind == "call"
                                     else max(k * np.exp(-R * d / 365) - S, 0))]
    print(f"测试集：{len(grid)} 个合约（{len(strikes)} 个行权价 × {len(days)} 个期限 × 认购/认沽，σ 随机），"
          f"时间价值 > 1e-6 的有效合约 {len(valid)} 个")

    results = {}
    for name, solver in [
        ("牛顿+二分（本课）", lambda p, k, t, kind: implied_vol(p, S, k, t, R, kind)[0]),
        ("brentq", lambda p, k, t, kind: brentq(lambda s: bs_price(S, k, t, R, s, kind) - p, SIG_LO, SIG_HI,
                                                xtol=1e-12)),
        ("共享模块 op.implied_vol", lambda p, k, t, kind: op.implied_vol(p, S, k, t, R, kind)),
    ]:
        t0 = time.perf_counter()
        est = np.array([solver(prices[i], grid[i][0], grid[i][1] / 365, grid[i][3]) for i in valid])
        elapsed = time.perf_counter() - t0
        truth = np.array([grid[i][2] for i in valid])
        err = np.abs(est - truth)
        results[name] = est
        print(f"   {name:<22} 最大误差 {np.nanmax(err):.2e}，中位数误差 {np.nanmedian(err):.2e}，"
              f"NaN {np.isnan(est).sum()} 个，平均 {elapsed / len(valid) * 1e6:,.0f} 微秒/个")

    methods = [implied_vol(prices[i], S, grid[i][0], grid[i][1] / 365, R, grid[i][3])[1] for i in valid]
    counts = pd.Series(methods).value_counts()
    print(f"\n   本课求解器的方法分布：{counts.to_dict()}")
    diff = np.nanmax(np.abs(results["牛顿+二分（本课）"] - results["brentq"]))
    print(f"   本课求解器与 brentq 的最大差异：{diff:.2e}")
    print("""
    💡 结论（定性）：精度上两者等价；速度取决于 Python 函数调用开销，量级相同。
       真正的批量计算（上千个合约、每秒刷新）应向量化牛顿法，或使用 Jäckel 的
       "Let's Be Rational" 算法（py_vollib 等库的实现）。
    """)


# ==================== 主函数 ====================


def main():
    what_is_iv()
    newton_demo()
    robust_solver()
    numerical_pitfalls()
    compare_brentq()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 numpy 向量化牛顿法：一次性求一整条期权链的 IV，并与逐个求解比较速度
    2. 把拐点初值换成纯 Brenner 初值（裁剪到 [0.05, 2]），在第五部分的测试集上统计
       牛顿法失败、需要二分兜底的比例，以及平均迭代次数
    3. 给定买价 0.0350、卖价 0.0380，分别求买价 IV、卖价 IV、中间价 IV，
       说明"IV 买卖价差"如何随行权价和期限变化
    4. 实现认沽的 IV 求解时，直接用平价把认沽价格换成认购价格再求解，两种方法结果一致吗？
    """)


if __name__ == "__main__":
    main()
