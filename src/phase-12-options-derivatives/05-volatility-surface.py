"""
波动率微笑与曲面
================

学习目标：
    1. 理解为什么不同行权价的 IV 不相同：肥尾、崩盘恐惧、供求
    2. 用跳跃扩散模型"制造"出偏斜，直观看到肥尾如何变成微笑
    3. 从期权链（真实数据优先，离线时用合成数据）拼接虚值一侧的微笑曲线
    4. 用 SVI 参数化拟合每个到期日的微笑，并组合成波动率曲面
    5. 检查蝶式（Durrleman 条件）与日历（总方差随期限不减）无套利约束

核心概念：
    - 对数在值程度 k = ln(K/F)，总隐含方差 w = σ_imp²·T
    - SVI：w(k) = a + b[ρ(k - m) + √((k - m)² + s²)]
    - 蝶式无套利 ⇔ 风险中性密度非负 ⇔ g(k) ≥ 0
    - 日历无套利 ⇔ 同一 k 下 w(k, T) 随 T 不减

前置知识：
    - 04-implied-volatility.py
    - CONCEPT.md 第 7 节

环境要求：
    - numpy、pandas、scipy、matplotlib、akshare
    - 需要网络获取真实期权链；QTC_OFFLINE=1 时使用缓存或合成数据
"""

import sys
from math import factorial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from option_data import RISK_FREE, chain_implied_vols, get_option_chain
from option_pricing import bs_price, fit_svi, implied_vol, svi_butterfly_g, svi_w
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

R = RISK_FREE


# ==================== 第一部分：偏斜从哪里来 ====================


def merton_price(S, K, T, r, sigma, lam, mu_j, delta_j, kind="call", n_terms=40):
    """
    Merton 跳跃扩散模型的欧式期权价格（泊松加权的 BS 价格之和）
        lam：每年跳跃次数；跳跃幅度 ln(1+J) ~ N(mu_j, delta_j²)
    """
    kappa = np.exp(mu_j + 0.5 * delta_j**2) - 1
    lam_p = lam * (1 + kappa)
    price = 0.0
    for n in range(n_terms):
        weight = np.exp(-lam_p * T) * (lam_p * T) ** n / factorial(n)
        sigma_n = np.sqrt(sigma**2 + n * delta_j**2 / T)
        r_n = r - lam * kappa + n * np.log(1 + kappa) / T
        price += weight * bs_price(S, K, T, r_n, sigma_n, kind)
    return price


def where_skew_comes_from():
    """肥尾（负跳跃）→ 偏斜"""
    print("=" * 60)
    print("第一部分：偏斜从哪里来")
    print("=" * 60)

    print("""
    如果 BS 完全正确，同一到期日所有行权价的 IV 应该相同（一条水平线）。现实中：

      IV ▲ 微笑 Smile                     IV ▲ 偏斜 Skew（股指典型形态）
         │●                    ●             │●
         │  ●                ●               │  ●  ●
         │     ●  ●  ●  ●  ●                 │        ●  ●
         └──────────────────────▶ K          └─────────────────●──●──▶ K

    原因：① 肥尾：极端行情比正态分布多  ② 崩盘恐惧：暴跌比暴涨更剧烈
          ③ 供求：机构买虚值认沽做保护  ④ 杠杆效应：价格跌 → 波动率升

    实验：用"偶尔向下跳"的 Merton 跳跃扩散模型给期权定价，再用 BS 反解 IV。
    """)
    S, T = 3.0, 30 / 365
    sigma, lam, mu_j, delta_j = 0.15, 1.0, -0.08, 0.05  # 平均每年 1 次、平均 -8% 的跳跃
    strikes = np.round(np.arange(2.6, 3.41, 0.1), 2)
    rows = []
    for k in strikes:
        kind = "put" if k < S else "call"  # 用虚值一侧，数值更稳
        px = merton_price(S, k, T, R, sigma, lam, mu_j, delta_j, kind)
        rows.append([k, kind, px, implied_vol(px, S, k, T, R, kind)])
    df = pd.DataFrame(rows, columns=["K", "类型", "跳跃模型价格", "BS 隐含波动率"])
    print(f"扩散波动率 σ={sigma:.0%}，跳跃：每年 {lam} 次，平均幅度 {np.exp(mu_j) - 1:.1%}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    left, atm, right = df["BS 隐含波动率"].iloc[0], df.loc[df["K"] == 3.0, "BS 隐含波动率"].iloc[0], \
        df["BS 隐含波动率"].iloc[-1]
    print(f"\n   K={strikes[0]} 的 IV {left:.2%}，平值 {atm:.2%}，K={strikes[-1]} 的 IV {right:.2%}")
    print("""
    💡 生成价格的模型里只有一个扩散波动率，但 BS 反解出的 IV 随行权价变化：
       低行权价的虚值认沽"保护的正是跳跃风险"，BS 只能用更高的 IV 来解释它的价格。
       偏斜是 BS 模型与真实分布之间差异的"指纹"。
    """)
    return df


# ==================== 第二部分：期权链 → 微笑曲线 ====================


def chain_to_smile():
    """真实（或合成）期权链的微笑与期限结构"""
    print("\n" + "=" * 60)
    print("第二部分：从期权链拼接微笑曲线")
    print("=" * 60)

    chain, meta = get_option_chain("510050")
    tag = {"real": "真实行情", "cache": "缓存的真实行情", "synthetic": "合成数据"}[meta["source"]]
    print(f"   数据：{meta['name']}（{meta['underlying']}）期权链，{tag}，"
          f"快照时间 {meta['snapshot']:%Y-%m-%d %H:%M}，标的价 {meta['spot']:.3f}")

    ivs = chain_implied_vols(chain, r=R)
    otm = ivs[ivs["otm"] & ivs["iv"].notna()].copy()
    print("""
    处理流程（与 06 课相同，细节见 06 课）：
       ① 过滤剩余期限 < 3 天、价格 < 0.0005 的合约
       ② 用平价关系估计每个到期日的隐含远期 F
       ③ 求每个合约的 IV；K < F 用认沽、K ≥ F 用认购（只保留虚值一侧）
    """)
    summary = otm.groupby("expiry").agg(天数=("days", "first"), 远期F=("F", "first"), 合约数=("iv", "size"),
                                         最低IV=("iv", "min"), 最高IV=("iv", "max"))
    atm = otm.loc[otm.groupby("expiry")["k"].apply(lambda s: s.abs().idxmin()), ["expiry", "iv"]]
    summary["平值附近IV"] = atm.set_index("expiry")["iv"]
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n   标的现价 {meta['spot']:.3f}；远期 F 与现价的差异反映了分红与资金成本预期")
    return otm, meta


# ==================== 第三部分：SVI 拟合 ====================


def svi_fitting(otm: pd.DataFrame):
    """先在已知参数上验证拟合，再拟合每个到期日"""
    print("\n" + "=" * 60)
    print("第三部分：SVI 参数化拟合")
    print("=" * 60)

    print("""
    SVI（Gatheral）：对每个到期日拟合总方差 w(k) = σ_imp²·T
        w(k) = a + b[ρ(k - m) + √((k - m)² + s²)]
    ┌──────┬──────────────────────────────────────┐
    │ a    │ 整体方差水平                         │
    │ b    │ 两翼斜率（b ≥ 0）                    │
    │ ρ    │ 偏斜方向（-1 < ρ < 1，股指通常为负） │
    │ m    │ 曲线水平平移                         │
    │ s    │ 平值附近的曲率（s > 0）              │
    └──────┴──────────────────────────────────────┘
    """)

    # 1) 先在"答案已知"的数据上检验拟合器
    T = 30 / 365
    true_p = np.array([0.0015, 0.04, -0.5, 0.01, 0.08])
    k = np.linspace(-0.15, 0.10, 13)
    rng = np.random.default_rng(0)
    iv_noisy = np.sqrt(svi_w(k, *true_p) / T) + rng.normal(0, 0.002, k.size)
    est = fit_svi(k, iv_noisy, T)
    print("   拟合器自检（真实参数已知，IV 加 0.2 个波动率点噪声）：")
    print(pd.DataFrame([true_p, est], index=["真实", "拟合"], columns=["a", "b", "rho", "m", "s"]).round(4).to_string())
    print("   参数本身可能不唯一（多组参数给出相近的曲线），评价拟合优劣应看 IV 误差而不是参数")

    # 2) 拟合每个到期日
    fits = {}
    rows = []
    for expiry, sub in otm.groupby("expiry"):
        T = sub["T"].iloc[0]
        if len(sub) < 6:
            continue
        params = fit_svi(sub["k"], sub["iv"], T)
        fit_iv = np.sqrt(np.maximum(svi_w(sub["k"].to_numpy(), *params), 1e-12) / T)
        rmse = np.sqrt(np.mean((fit_iv - sub["iv"]) ** 2)) * 100
        fits[expiry] = (T, params)
        rows.append([expiry.date(), sub["days"].iloc[0], *params, rmse, np.abs(fit_iv - sub["iv"]).max() * 100])
    df = pd.DataFrame(rows, columns=["到期日", "天数", "a", "b", "rho", "m", "s", "RMSE(点)", "最大误差(点)"])
    print("\n   各到期日 SVI 参数：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("   ρ < 0 表示左侧（低行权价）IV 更高；RMSE 以波动率点计（1 点 = 1%）")
    return fits


# ==================== 第四部分：无套利检查 ====================


def arbitrage_checks(fits: dict, otm: pd.DataFrame):
    """蝶式（Durrleman g ≥ 0）与日历（w 随 T 不减）"""
    print("\n" + "=" * 60)
    print("第四部分：无套利约束检查")
    print("=" * 60)

    print("""
    ① 蝶式无套利：同一到期日，认购价格对行权价是凸函数
         ⇔ 隐含风险中性密度 q(K) = e^(rT)·∂²C/∂K² ≥ 0
         ⇔ Durrleman 函数 g(k) = (1 - k·w'/2w)² - (w')²/4·(1/w + 1/4) + w''/2 ≥ 0
    ② 日历无套利：同一 k 下，总方差 w(k, T) 随 T 不减（否则"卖近买远"的日历价差可套利）
    """)
    k_grid = np.linspace(-0.3, 0.3, 121)
    rows = []
    for expiry, (T, p) in fits.items():
        g = svi_butterfly_g(k_grid, p)
        k_obs = otm.loc[otm["expiry"] == expiry, "k"]
        inside = (k_grid >= k_obs.min()) & (k_grid <= k_obs.max())
        rows.append([expiry.date(), f"[{k_obs.min():+.3f}, {k_obs.max():+.3f}]", g[inside].min(),
                     int((g[inside] < 0).sum()), int((g[~inside] < 0).sum())])
    print("   蝶式检查（k 网格 [-0.3, 0.3]，步长 0.005）：")
    print(pd.DataFrame(rows, columns=["到期日", "有报价的 k 区间", "区间内 min g", "区间内 g<0 点数",
                                      "外推区 g<0 点数"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("   外推区（没有报价的深度虚值区域）的违反只说明 SVI 外推不可靠，区间内的违反才说明报价或拟合有问题")

    items = sorted(fits.items())
    cal_rows = []
    for (e1, (t1, p1)), (e2, (t2, p2)) in zip(items[:-1], items[1:]):
        diff = svi_w(k_grid, *p2) - svi_w(k_grid, *p1)
        k1, k2 = otm.loc[otm["expiry"] == e1, "k"], otm.loc[otm["expiry"] == e2, "k"]
        inside = (k_grid >= max(k1.min(), k2.min())) & (k_grid <= min(k1.max(), k2.max()))  # 两个到期日都有报价
        cal_rows.append([f"{e1.date()} → {e2.date()}", diff[inside].min(), int((diff[inside] < 0).sum()),
                         int((diff[~inside] < 0).sum())])
    print("\n   日历检查：w(k, T_远) - w(k, T_近)")
    print(pd.DataFrame(cal_rows, columns=["相邻到期日", "共同报价区间内最小差值", "区间内违反点数", "外推区违反点数"])
          .to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    bad = np.array([0.0005, 0.6, -0.95, 0.0, 0.02])  # 两翼过陡、曲率过小的反例
    g_bad = svi_butterfly_g(k_grid, bad)
    print(f"\n   反例：参数 {bad.tolist()} → min g = {g_bad.min():.3f}，{int((g_bad < 0).sum())} 个格点违反蝶式无套利")
    print("""
    💡 自由拟合的 SVI 不保证无套利。发现违反时可以：加约束重新拟合（如 SSVI 参数化）、
       剔除异常报价，或在违反区域不使用该曲面定价。深度虚值、成交稀少的合约报价最容易出问题。
    """)
    return k_grid


# ==================== 第五部分：画微笑、隐含密度与曲面 ====================


def plot_surface(otm: pd.DataFrame, fits: dict, meta: dict, jump_df: pd.DataFrame):
    """微笑拟合图、隐含密度、波动率曲面"""
    print("\n" + "=" * 60)
    print("第五部分：微笑、隐含密度与波动率曲面")
    print("=" * 60)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    for (expiry, (T, p)), color in zip(sorted(fits.items()), plt.cm.viridis(np.linspace(0, 0.9, len(fits)))):
        sub = otm[otm["expiry"] == expiry]
        kk = np.linspace(sub["k"].min() - 0.02, sub["k"].max() + 0.02, 100)
        ax.scatter(sub["k"], sub["iv"] * 100, color=color, s=18)
        ax.plot(kk, np.sqrt(svi_w(kk, *p) / T) * 100, color=color, label=f"{expiry:%Y-%m-%d}（{T * 365:.0f}天）")
    ax.set_xlabel("k = ln(K/F)")
    ax.set_ylabel("隐含波动率（%）")
    src = "合成数据" if meta["source"] == "synthetic" else "真实行情"
    ax.set_title(f"{meta['name']} 微笑：点=市场 IV，线=SVI（{src}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 隐含风险中性密度：q(K) = e^{rT}·∂²C/∂K²，用 SVI 曲面上的认购价格做数值二阶差分
    ax = axes[1]
    expiry, (T, p) = sorted(fits.items())[min(1, len(fits) - 1)]
    F = otm.loc[otm["expiry"] == expiry, "F"].iloc[0]
    K = np.linspace(F * 0.8, F * 1.2, 400)
    iv_k = np.sqrt(svi_w(np.log(K / F), *p) / T)
    C = bs_price(F * np.exp(-R * T), K, T, R, iv_k, "call")
    dK = K[1] - K[0]
    dens = np.exp(R * T) * (C[2:] - 2 * C[1:-1] + C[:-2]) / dK**2
    atm_iv = float(np.sqrt(svi_w(0.0, *p) / T))
    Km = K[1:-1]
    logn = np.exp(-(np.log(Km / F) + 0.5 * atm_iv**2 * T) ** 2 / (2 * atm_iv**2 * T)) / (Km * atm_iv * np.sqrt(2 * np.pi * T))
    ax.plot(Km, dens, label="SVI 隐含密度")
    ax.plot(Km, logn, "--", label=f"对数正态（平值 IV {atm_iv:.1%}）")
    ax.set_xlabel("到期价格 K")
    ax.set_title(f"隐含风险中性密度（{expiry:%Y-%m-%d}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    left_mass = np.sum(dens[Km < F * 0.9]) * dK
    left_logn = np.sum(logn[Km < F * 0.9]) * dK
    print(f"   {expiry:%Y-%m-%d} 到期：隐含密度在 K < 0.9F 的概率 = {left_mass:.2%}，"
          f"同平值 IV 的对数正态 = {left_logn:.2%}；密度总和 ≈ {np.sum(dens) * dK:.3f}；最小密度 = {dens.min():.3g}")

    ax = axes[2]
    ax.plot(jump_df["K"], jump_df["BS 隐含波动率"] * 100, "o-")
    ax.set_xlabel("行权价 K（S=3.0）")
    ax.set_ylabel("BS 隐含波动率（%）")
    ax.set_title("第一部分：跳跃模型价格反解出的偏斜")
    ax.grid(alpha=0.3)
    path1 = save_figure(fig, "05_smile_svi.png", subdir="phase-12")

    # 曲面：同一 k 下对总方差按 T 线性插值（这种插值保持日历无套利）
    items = sorted(fits.items())
    Ts = np.array([t for _, (t, _) in items])
    k_grid = np.linspace(-0.15, 0.15, 41)
    W = np.array([svi_w(k_grid, *p) for _, (_, p) in items])
    T_grid = np.linspace(Ts[0], Ts[-1], 40)
    W_interp = np.array([[np.interp(t, Ts, W[:, j]) for j in range(len(k_grid))] for t in T_grid])
    IV = np.sqrt(W_interp / T_grid[:, None]) * 100
    KK, TT = np.meshgrid(k_grid, T_grid * 365)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(KK, TT, IV, cmap="viridis", alpha=0.9)
    ax.set_xlabel("k = ln(K/F)")
    ax.set_ylabel("剩余天数")
    ax.set_zlabel("IV（%）")
    ax.set_title(f"{meta['name']} 波动率曲面（SVI + 总方差线性插值，{src}）")
    path2 = save_figure(fig, "05_vol_surface.png", subdir="phase-12")

    atm_term = [(t * 365, np.sqrt(svi_w(0.0, *p) / t)) for _, (t, p) in items]
    print("\n   平值（k=0）期限结构：" + "，".join(f"{d:.0f}天 {v:.2%}" for d, v in atm_term))
    skew = [(t * 365, (np.sqrt(svi_w(-0.05, *p) / t) - np.sqrt(svi_w(0.05, *p) / t)) * 100) for _, (t, p) in items]
    print("   偏斜 IV(k=-0.05) - IV(k=+0.05)（波动率点）：" + "，".join(f"{d:.0f}天 {v:+.2f}" for d, v in skew))
    print("""
    💡 读图要点：
       - 期限结构：平值 IV 随期限的变化（平静时常向上倾斜，恐慌时近月更高）
       - 偏斜：负数表示右侧（高行权价）IV 更高。国内 50ETF 期权有时会出现右翼上翘，
         与 A 股"怕踏空"的看涨需求有关；不同时期形态会变化，不能当作固定规律
       - 曲面不是静态的：标的变动时曲面如何移动（Sticky Strike / Sticky Delta）会影响对冲比例
    """)
    print(f"📈 图已保存：{path1}")
    print(f"📈 图已保存：{path2}")


# ==================== 主函数 ====================


def main():
    jump_df = where_skew_comes_from()
    otm, meta = chain_to_smile()
    fits = svi_fitting(otm)
    arbitrage_checks(fits, otm)
    plot_surface(otm, fits, meta, jump_df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第一部分中把跳跃均值改为 +8%（向上跳），偏斜方向如何变化？把跳跃频率调高呢？
    2. 用 510300（沪深300ETF）期权链重复第二至第四部分，比较两个标的的偏斜与期限结构
    3. 给 SVI 拟合加权（按 Vega 或按买卖价差的倒数），比较拟合误差在平值与两翼的分布
    4. 用拟合好的曲面为一个不在挂牌行权价上的期权（如 K=2.93、剩余 60 天）定价，
       并与"直接用平值 IV"定价的结果比较
    """)


if __name__ == "__main__":
    main()
