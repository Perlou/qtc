"""
Delta 对冲模拟
==============

学习目标：
    1. 动手实现"卖出认购 + 离散 Delta 对冲"，逐步看清现金、持仓与盈亏的变化
    2. 验证对冲盈亏 ≈ Σ ½·Γ·S²·[(ΔS/S)² − σ_imp²·Δt]，即"实现方差 − 隐含方差"
    3. 比较不同对冲频率：误差与 1/√N 的关系、交易成本的权衡
    4. 比较"用隐含波动率"与"用实现波动率"计算 Delta 的差别
    5. 认识跳空对 Delta 对冲的破坏

核心概念：
    - Delta 中性组合：ΔΠ ≈ Θ·Δt + ½·Γ·(ΔS)²
    - BS 偏微分方程（r=0）：Θ = −½·σ_imp²·S²·Γ → Theta 是为 Gamma 支付的"租金"
    - 卖方对冲 = 涨了买、跌了卖（高买低卖），权利金是对这一成本的补偿

前置知识：
    - 03-greeks.py（Gamma-Theta 权衡）
    - CONCEPT.md 第 9 节

环境要求：
    - numpy、pandas、scipy、matplotlib（本课不需要网络，全部为模拟数据，随机种子固定）
    - 为突出核心逻辑，本课取 r = 0、一年 252 个交易日
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from option_pricing import bs_greeks, bs_price
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

S0, K, R = 3.0, 3.0, 0.0
DAYS = 30
T = DAYS / 252
STEPS_PER_DAY = 16  # 最细的模拟粒度：每天 16 个时间点
N_PATHS = 4000


def simulate_paths(sigma_real, n_paths=N_PATHS, steps_per_day=STEPS_PER_DAY, seed=0, jump=None):
    """
    模拟 GBM 价格路径（形状：n_paths × (n_steps + 1)）
    jump=(第几天, 跳幅)：在指定交易日开盘时加入一次跳空，例如 (15, -0.07)
    """
    rng = np.random.default_rng(seed)
    n_steps = DAYS * steps_per_day
    dt = T / n_steps
    log_ret = (-0.5 * sigma_real**2 * dt) + sigma_real * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    if jump is not None:
        day, size = jump
        log_ret[:, day * steps_per_day] += np.log(1 + size)
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1))


def hedge_short_call(paths, sigma_imp, sigma_delta=None, cost_rate=0.0):
    """
    以 sigma_imp 卖出 1 份平值认购，按 sigma_delta（默认 = sigma_imp）计算的 Delta 在每个时间点调仓。
    返回 dict：pnl（每条路径的总盈亏）、approx（Gamma 公式近似）、cost（交易成本）
    """
    sigma_delta = sigma_imp if sigma_delta is None else sigma_delta
    n_steps = paths.shape[1] - 1
    dt = T / n_steps
    tau = T - dt * np.arange(n_steps)  # 每个调仓时点的剩余期限（不含到期时点）
    S = paths[:, :-1]
    dS = np.diff(paths, axis=1)

    delta = bs_greeks(S, K, tau, R, sigma_delta, "call")["delta"]
    premium = float(bs_price(S0, K, T, R, sigma_imp, "call"))
    payoff = np.maximum(paths[:, -1] - K, 0)

    # 卖出认购收权利金；持有 delta 份标的，标的变动带来 delta·dS 的盈亏；到期支付认购的内在价值
    hedge_gain = np.sum(delta * dS, axis=1)
    trades = np.abs(np.diff(np.concatenate([np.zeros((len(paths), 1)), delta, np.zeros((len(paths), 1))], axis=1), axis=1))
    trade_px = np.concatenate([S, paths[:, -1:]], axis=1)
    cost = cost_rate * np.sum(trades * trade_px, axis=1)
    pnl = premium + hedge_gain - payoff - cost

    gamma = bs_greeks(S, K, tau, R, sigma_imp, "call")["gamma"]
    approx = -np.sum(0.5 * gamma * S**2 * ((dS / S) ** 2 - sigma_imp**2 * dt), axis=1)  # 卖方 = 买方取负
    return {"pnl": pnl, "approx": approx, "cost": cost, "premium": premium}


# ==================== 第一部分：直觉与公式 ====================


def intuition():
    """把期权变成纯波动率头寸"""
    print("=" * 60)
    print("第一部分：Delta 对冲的直觉与核心公式")
    print("=" * 60)

    print("""
    卖出认购后，标的上涨就亏钱。按 Delta 买入标的后，小幅涨跌的损益被抵消，
    剩下的风险只和"波动有多大"有关，与方向无关。

    ┌──────────────────────────────────────────────────────────────┐
    │  卖出认购，收权利金 ──▶ 买入 Δ 份标的                        │
    │         ▲                        │                           │
    │   到期结算                  标的价格变动                     │
    │         ▲                        ▼                           │
    │   重复直到到期 ◀── 重算 Δ，调整持仓（涨了加买，跌了卖出）    │
    └──────────────────────────────────────────────────────────────┘

    Delta 中性组合在一小段时间内（泰勒展开 + BS 方程，r = 0）：
        ΔΠ_买方 ≈ ½·Γ·S²·[(ΔS/S)² − σ_imp²·Δt]
                        ↑ 实现方差     ↑ 隐含方差（你为之付费的）
    累计到到期：PnL ≈ Σ ½·Γ_t·S_t²·(σ_real² − σ_imp²)·Δt

    ┌──────────────┬───────┬───────────────┬────────────────────────┐
    │ 头寸         │ Gamma │ Theta         │ 盼望什么               │
    ├──────────────┼───────┼───────────────┼────────────────────────┤
    │ 买期权 + 对冲│  +    │ −（付租金）   │ 实现波动 > 隐含波动    │
    │ 卖期权 + 对冲│  −    │ +（收租金）   │ 实现波动 < 隐含波动    │
    └──────────────┴───────┴───────────────┴────────────────────────┘
    """)


# ==================== 第二部分：单条路径逐步演示 ====================


def single_path_demo():
    """一条路径上的每日对冲流水"""
    print("\n" + "=" * 60)
    print("第二部分：一条路径上的每日对冲流水")
    print("=" * 60)

    sigma_imp, sigma_real = 0.20, 0.20
    path = simulate_paths(sigma_real, n_paths=1, steps_per_day=1, seed=3)[0]
    tau = T - np.arange(DAYS) / 252
    delta = bs_greeks(path[:-1], K, tau, R, sigma_imp, "call")["delta"]
    gamma = bs_greeks(path[:-1], K, tau, R, sigma_imp, "call")["gamma"]
    premium = float(bs_price(S0, K, T, R, sigma_imp, "call"))

    shares_prev = np.concatenate([[0.0], delta[:-1]])
    trade = delta - shares_prev
    cash = premium - np.cumsum(trade * path[:-1])  # 收权利金后，买入标的花掉现金
    log = pd.DataFrame({"天": np.arange(DAYS), "S": path[:-1], "Delta": delta, "买卖份数": trade,
                        "现金": cash, "Gamma": gamma})
    print(f"σ_imp = σ_real = {sigma_imp:.0%}，每天收盘调仓一次；卖出 1 份平值认购，收权利金 {premium:.4f}")
    print(log.head(8).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("   ...")
    final_cash = cash[-1] + delta[-1] * path[-1]  # 到期卖出全部标的
    payoff = max(path[-1] - K, 0)
    print(f"   到期 S_T = {path[-1]:.4f}：卖出持仓后现金 {final_cash:+.4f}，支付认购价值 {payoff:.4f}，"
          f"总盈亏 {final_cash - payoff:+.4f}（权利金的 {(final_cash - payoff) / premium:+.0%}）")
    up = log["买卖份数"].iloc[1:][np.diff(log["S"]) > 0]
    down = log["买卖份数"].iloc[1:][np.diff(log["S"]) < 0]
    print(f"   标的上涨的 {len(up)} 天中买入的有 {int((up > 0).sum())} 天；下跌的 {len(down)} 天中卖出的有 "
          f"{int((down < 0).sum())} 天 → 卖方对冲就是\"追涨杀跌\"")

    rets = np.diff(path) / path[:-1]
    contrib = -0.5 * gamma * path[:-1] ** 2 * (rets**2 - sigma_imp**2 / 252)
    approx_total = contrib.sum()
    print(f"   Gamma 公式逐日累加 = {approx_total:+.4f}（实际 {final_cash - payoff:+.4f}，差异来自每天只调一次仓）")

    setup_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(np.arange(DAYS + 1), path, "o-", ms=3)
    axes[0].axhline(K, color="gray", ls="--")
    axes[0].set_ylabel("标的价格")
    axes[0].set_title("单条路径：卖出平值认购 + 每日 Delta 对冲")
    axes[1].step(np.arange(DAYS), delta, where="post")
    axes[1].set_ylabel("持有标的份数（Delta）")
    axes[2].bar(np.arange(1, DAYS + 1), contrib, color=np.where(contrib > 0, "tab:green", "tab:red"))
    axes[2].set_ylabel("每日 ½ΓS²(σ²Δt−R²)")
    axes[2].set_xlabel("交易日")
    for ax in axes:
        ax.grid(alpha=0.3)
    path_fig = save_figure(fig, "07_single_path.png", subdir="phase-12")
    print("   绿柱：当天波动小于隐含（卖方赚 Theta）；红柱：当天波动大于隐含（卖方亏 Gamma）")
    print(f"📈 图已保存：{path_fig}")


# ==================== 第三部分：验证 Gamma 公式 ====================


def verify_formula():
    """对冲盈亏 vs ½ΓS²(实现方差 − 隐含方差)"""
    print("\n" + "=" * 60)
    print("第三部分：验证 PnL ≈ Σ ½·Γ·S²·(实现方差 − 隐含方差)")
    print("=" * 60)

    rows = []
    scatter = {}
    for sigma_imp, sigma_real in [(0.25, 0.20), (0.20, 0.20), (0.20, 0.25)]:
        paths = simulate_paths(sigma_real, seed=11)
        res = hedge_short_call(paths, sigma_imp)
        pnl, approx = res["pnl"], res["approx"]
        slope = np.cov(pnl, approx)[0, 1] / np.var(approx, ddof=1)
        rows.append([f"{sigma_imp:.0%}", f"{sigma_real:.0%}", res["premium"], pnl.mean(), pnl.std(), approx.mean(),
                     np.corrcoef(pnl, approx)[0, 1], slope])
        scatter[(sigma_imp, sigma_real)] = (approx, pnl)
    df = pd.DataFrame(rows, columns=["σ_imp", "σ_real", "权利金", "PnL均值", "PnL标准差", "公式均值", "相关系数", "回归斜率"])
    print(f"卖出 1 份平值认购、{DAYS} 个交易日、每天对冲 {STEPS_PER_DAY} 次、{N_PATHS} 条路径：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("""
    💡 读表：
       - 相关系数接近 1、回归斜率接近 1：对冲盈亏几乎完全由"实现方差 − 隐含方差"按 Gamma 加权决定
       - 隐含 > 实现：卖方平均盈利；隐含 < 实现：平均亏损；两者相等：平均约为 0
       - 期权卖方赚的是"隐含 − 实现"的波动率差，与标的涨跌方向无关
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, ((si, sr), (approx, pnl)) in zip(axes, scatter.items()):
        ax.scatter(approx, pnl, s=3, alpha=0.3)
        lim = [min(approx.min(), pnl.min()), max(approx.max(), pnl.max())]
        ax.plot(lim, lim, "r--", lw=1, label="y = x")
        ax.set_title(f"σ_imp={si:.0%}, σ_real={sr:.0%}")
        ax.set_xlabel("Gamma 公式近似")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("实际对冲盈亏（每份）")
    path = save_figure(fig, "07_formula_check.png", subdir="phase-12")
    print(f"📈 图已保存：{path}")


# ==================== 第四部分：对冲频率与交易成本 ====================


def hedge_frequency():
    """离散对冲误差 ∝ 1/√N，交易成本 ∝ 调仓量"""
    print("\n" + "=" * 60)
    print("第四部分：对冲频率与交易成本")
    print("=" * 60)

    sigma = 0.20
    fine = simulate_paths(sigma, seed=21)
    cost_rate = 0.0003  # 单边成本示例：佣金 + 半个买卖价差，约万分之三（以实际费率为准）
    rows = []
    for per_day, label in [(0.2, "每 5 天"), (1, "每天"), (4, "每天 4 次"), (16, "每天 16 次")]:
        step = int(round(STEPS_PER_DAY / per_day))
        paths = fine[:, ::step]
        res0 = hedge_short_call(paths, sigma)
        res1 = hedge_short_call(paths, sigma, cost_rate=cost_rate)
        n = paths.shape[1] - 1
        rows.append([label, n, res0["pnl"].mean(), res0["pnl"].std(), res0["pnl"].std() * np.sqrt(n),
                     res1["cost"].mean(), res1["pnl"].mean()])
    df = pd.DataFrame(rows, columns=["频率", "调仓次数N", "PnL均值(无成本)", "PnL标准差", "标准差×√N",
                                     "平均成本", "PnL均值(含成本)"])
    print(f"σ_imp = σ_real = {sigma:.0%}（公平定价），同一组路径，单边成本率 {cost_rate:.2%}：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    premium = float(bs_price(S0, K, T, R, sigma, "call"))
    print(f"""
    💡 读表（权利金 {premium:.4f}）：
       - 无成本时 PnL 均值都约为 0（公平定价），但标准差随频率提高而下降；
         "标准差×√N" 大致稳定 → 离散对冲误差 ∝ 1/√N
       - 成本随调仓次数增加：对冲越勤，风险越小、成本越高 —— 实务中常用 Delta 阈值（带宽）对冲
    """)


# ==================== 第五部分：用哪个波动率算 Delta ====================


def which_vol_for_delta():
    """σ_imp 对冲 vs σ_real 对冲"""
    print("\n" + "=" * 60)
    print("第五部分：用隐含波动率还是实现波动率算 Delta？")
    print("=" * 60)

    sigma_imp, sigma_real = 0.25, 0.20
    paths = simulate_paths(sigma_real, seed=31)
    theory = float(bs_price(S0, K, T, R, sigma_imp) - bs_price(S0, K, T, R, sigma_real))
    rows = []
    for label, sd in [("用 σ_imp 算 Delta", sigma_imp), ("用 σ_real 算 Delta", sigma_real)]:
        pnl = hedge_short_call(paths, sigma_imp, sigma_delta=sd)["pnl"]
        rows.append([label, pnl.mean(), pnl.std(), np.percentile(pnl, 5), np.percentile(pnl, 95)])
    df = pd.DataFrame(rows, columns=["方式", "PnL均值", "PnL标准差", "5%分位", "95%分位"])
    print(f"以 σ_imp={sigma_imp:.0%} 卖出、真实 σ_real={sigma_real:.0%}（假设你准确预测了实现波动率）：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"   理论值：BS(σ_imp) − BS(σ_real) = {theory:+.5f}")
    print("""
    💡 用 σ_real 对冲：最终盈亏接近确定的 BS(σ_imp) − BS(σ_real)，但持仓期间按市价（σ_imp）
       盯市的盈亏会波动；用 σ_imp 对冲：盯市平稳，最终盈亏取决于路径（Γ 大时波动大不大）。
       两者均值接近、风险形态不同 —— 前提是你真的知道 σ_real，而现实中这正是最难的。
    """)


# ==================== 第六部分：跳空与 A 股约束 ====================


def jump_risk():
    """跳空：Delta 对冲来不及调整"""
    print("\n" + "=" * 60)
    print("第六部分：跳空 —— Delta 对冲的盲区")
    print("=" * 60)

    sigma = 0.20
    rows = []
    for label, jump in [("无跳空", None), ("第 15 天跳空 −7%", (15, -0.07)), ("第 15 天跳空 +7%", (15, 0.07))]:
        paths = simulate_paths(sigma, seed=41, jump=jump)
        pnl = hedge_short_call(paths, sigma)["pnl"]
        rows.append([label, pnl.mean(), pnl.std(), np.percentile(pnl, 1), pnl.min()])
    df = pd.DataFrame(rows, columns=["情景", "PnL均值", "标准差", "1%分位", "最差"])
    premium = float(bs_price(S0, K, T, R, sigma, "call"))
    print(f"σ_imp = σ_real = {sigma:.0%}，每天对冲 {STEPS_PER_DAY} 次，权利金 {premium:.4f}：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(f"""
    💡 跳空的那一下，½·Γ·(ΔS)² 一次性兑现，无论对冲多勤都躲不掉 —— 无论涨跌，卖方都亏。
       这正是 08 课"卖方尾部风险"的根源。

    A 股实务约束（CONCEPT.md 9.5 节）：
       - 50ETF、300ETF 等股票类 ETF 实行 T+1：当天买入的对冲仓位次日才能卖出
       - 卖出对冲需要已有持仓或融券；还要考虑最小交易单位（100 份）
       - 涨跌停、隔夜跳空时无法及时调整；实务中也常用股指期货或期权合成头寸对冲
    """)


# ==================== 主函数 ====================


def main():
    intuition()
    single_path_demo()
    verify_formula()
    hedge_frequency()
    which_vol_for_delta()
    jump_risk()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 实现"Delta 带宽对冲"：只有当 |当前 Delta − 持仓| > 0.05 时才调仓，与每天对冲比较成本和标准差
    2. 加入无风险利率 r = 2%（现金账户计息），重新推导并验证 Gamma 公式
    3. 把平值期权换成虚值 10% 的认购，Gamma 公式的拟合效果如何变化？为什么？
    4. 让实现波动率在前 15 天为 30%、后 15 天为 10%（平均方差与 20% 相同），
       比较与"全程 20%"的对冲盈亏 —— 体会"大波动发生在 Gamma 大的时候"的路径依赖
    """)


if __name__ == "__main__":
    main()
