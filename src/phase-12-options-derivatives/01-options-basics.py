"""
期权基础与收益结构
==================

学习目标：
    1. 理解认购（Call）/ 认沽（Put）、买方权利 / 卖方义务等基本概念
    2. 画出四种基本头寸的到期盈亏图，理解期权收益的"不对称性"
    3. 把期权价格拆成内在价值 + 时间价值，认识实值 / 平值 / 虚值
    4. 用数值验证认购-认沽平价（Put-Call Parity），并演示平价被破坏时的套利

核心概念：
    - Call 到期价值 = max(S_T - K, 0)，Put 到期价值 = max(K - S_T, 0)
    - 期权价格 = 内在价值 + 时间价值
    - 认购-认沽平价：C - P = S - K·e^(-rT)

前置知识：
    - Phase 3：正态分布、随机游走
    - 本阶段 CONCEPT.md 第 2 节

环境要求：
    - numpy、pandas、matplotlib、scipy（本课不需要网络）
    - 期权价格由共享模块 option_pricing.bs_price 计算，公式本身在 02 课详细推导
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from option_pricing import bs_price
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

# 本课统一使用的示例参数：类似 50ETF 的价格水平
S0, K, T, R, SIGMA = 3.0, 3.0, 30 / 365, 0.02, 0.20
MULTIPLIER = 10000  # 合约单位：一张合约对应 10000 份 ETF（示例，以交易所最新规则为准）


def payoff(S_T, K, kind="call", position=1, premium=0.0):
    """到期盈亏（每份）；position=+1 买入，-1 卖出；premium 为开仓时的权利金"""
    intrinsic = np.maximum(S_T - K, 0) if kind == "call" else np.maximum(K - S_T, 0)
    return position * (intrinsic - premium)


# ==================== 第一部分：期权是什么 ====================


def what_is_option():
    """期权的定义与术语"""
    print("=" * 60)
    print("第一部分：期权是什么")
    print("=" * 60)

    print("""
    期权 = 一份"权利"合约：买方付权利金（Premium），换取在到期日按行权价（Strike）
           买入或卖出标的的权利；卖方收权利金，承担对应的义务。

    🏠 Call 像买房定金：付 2 万锁定 300 万房价。房价涨到 350 万 → 行权赚 48 万；
                       房价跌到 250 万 → 放弃定金，最多亏 2 万。
    🛡️ Put  像保险：   付保费，资产跌破约定价时获得赔付。

    ┌──────────┬────────────┬──────────────────────────────────────┐
    │ 术语     │ 英文       │ 含义                                 │
    ├──────────┼────────────┼──────────────────────────────────────┤
    │ 认购期权 │ Call       │ 以行权价 K 买入标的的权利            │
    │ 认沽期权 │ Put        │ 以行权价 K 卖出标的的权利            │
    │ 权利金   │ Premium    │ 期权价格，买方付给卖方               │
    │ 到期日   │ Expiry, T  │ 权利最后有效日                       │
    │ 欧式     │ European   │ 只能到期日行权（国内 ETF 期权）      │
    │ 美式     │ American   │ 到期前任意时刻可行权                 │
    │ 合约单位 │ Multiplier │ 一张合约对应的标的数量，ETF 期权一般 │
    │          │            │ 为 10000 份（分红后会调整）          │
    └──────────┴────────────┴──────────────────────────────────────┘

    ⚠️ 合约单位等要素以交易所最新规则为准（06 课会详细讲国内 ETF 期权合约）。
    """)

    c = float(bs_price(S0, K, T, R, SIGMA, "call"))
    print(f"📌 示例：标的 {S0} 元、行权价 {K} 元、{T * 365:.0f} 天到期、波动率 {SIGMA:.0%} 的平值认购")
    print(f"   每份价格 ≈ {c:.4f} 元 → 一张合约权利金 ≈ {c * MULTIPLIER:,.0f} 元，"
          f"控制的标的市值 = {S0 * MULTIPLIER:,.0f} 元")


# ==================== 第二部分：四种基本头寸 ====================


def four_positions():
    """四种基本头寸的到期盈亏"""
    print("\n" + "=" * 60)
    print("第二部分：四种基本头寸的到期盈亏")
    print("=" * 60)

    c = float(bs_price(S0, K, T, R, SIGMA, "call"))
    p = float(bs_price(S0, K, T, R, SIGMA, "put"))
    print(f"\n权利金（BS 公式，02 课推导）：Call = {c:.4f}，Put = {p:.4f}")

    positions = [
        ("买入认购 Long Call", "call", +1, c),
        ("卖出认购 Short Call", "call", -1, c),
        ("买入认沽 Long Put", "put", +1, p),
        ("卖出认沽 Short Put", "put", -1, p),
    ]
    grid = np.array([2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4])
    table = pd.DataFrame({name: payoff(grid, K, kind, pos, prem) for name, kind, pos, prem in positions},
                         index=pd.Index(grid, name="到期价 S_T"))
    print("\n每份到期盈亏（元）：")
    print(table.round(4).to_string())

    rows = []
    for name, kind, pos, prem in positions:
        breakeven = K + prem if kind == "call" else K - prem
        if kind == "call":
            max_gain, max_loss = ("无限", f"{prem:.4f}") if pos > 0 else (f"{prem:.4f}", "无限")
        else:
            max_gain, max_loss = (f"{K - prem:.4f}", f"{prem:.4f}") if pos > 0 else (f"{prem:.4f}", f"{K - prem:.4f}")
        rows.append([name, f"{breakeven:.4f}", max_gain, max_loss])
    print("\n盈亏平衡点与最大盈亏（每份）：")
    print(pd.DataFrame(rows, columns=["头寸", "盈亏平衡点", "最大收益", "最大亏损"]).to_string(index=False))

    print("""
    💡 不对称性是期权的本质：
       - 买方：亏损封顶（= 权利金），收益可能很大 → 像买彩票 / 买保险
       - 卖方：收益封顶（= 权利金），亏损可能很大 → 像卖彩票 / 卖保险
       - 同一合约的买方盈亏 + 卖方盈亏 ≡ 0：期权是零和合约（不计手续费）
    """)

    setup_chinese_font()
    S_T = np.linspace(2.5, 3.5, 201)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (name, kind, pos, prem) in zip(axes.ravel(), positions):
        pnl = payoff(S_T, K, kind, pos, prem)
        ax.plot(S_T, pnl, lw=2, color="tab:blue" if pos > 0 else "tab:red")
        ax.fill_between(S_T, pnl, 0, where=pnl > 0, alpha=0.15, color="green")
        ax.fill_between(S_T, pnl, 0, where=pnl < 0, alpha=0.15, color="red")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(K, color="gray", ls="--", lw=0.8)
        ax.set_title(name)
        ax.set_ylabel("每份盈亏（元）")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("到期标的价格 S_T")
    fig.suptitle(f"四种基本头寸到期盈亏（K={K}，权利金按 BS 计算）")
    path = save_figure(fig, "01_four_positions.png", subdir="phase-12")
    print(f"📈 图已保存：{path}")


# ==================== 第三部分：内在价值与时间价值 ====================


def intrinsic_and_time_value():
    """价格 = 内在价值 + 时间价值"""
    print("\n" + "=" * 60)
    print("第三部分：内在价值与时间价值")
    print("=" * 60)

    print("""
    期权价格 = 内在价值（立即行权能拿到的） + 时间价值（未来变得更值钱的"希望"）
      认购内在价值 = max(S - K, 0)，认沽内在价值 = max(K - S, 0)

    ┌───────────┬──────────┬──────────┬────────────────────────────┐
    │ 状态      │ 认购     │ 认沽     │ 特点                       │
    ├───────────┼──────────┼──────────┼────────────────────────────┤
    │ 实值 ITM  │ S > K    │ S < K    │ 有内在价值，走势接近标的   │
    │ 平值 ATM  │ S ≈ K    │ S ≈ K    │ 时间价值最大               │
    │ 虚值 OTM  │ S < K    │ S > K    │ 只有时间价值，便宜、易归零 │
    └───────────┴──────────┴──────────┴────────────────────────────┘
    """)

    strikes = np.array([2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3])
    rows = []
    for k in strikes:
        for kind in ("call", "put"):
            price = float(bs_price(S0, k, T, R, SIGMA, kind))
            intrinsic = max(S0 - k, 0) if kind == "call" else max(k - S0, 0)
            moneyness = "平值" if abs(k - S0) < 1e-9 else ("实值" if (S0 > k) == (kind == "call") else "虚值")
            rows.append([kind, k, moneyness, price, intrinsic, price - intrinsic])
    df = pd.DataFrame(rows, columns=["类型", "行权价", "状态", "价格", "内在价值", "时间价值"])
    print(f"标的 S={S0}，{T * 365:.0f} 天到期，σ={SIGMA:.0%}：")
    print(df.round(4).to_string(index=False))
    best = df.loc[df["时间价值"].idxmax()]
    print(f"\n👉 时间价值最大的合约：{best['类型']} K={best['行权价']}（{best['状态']}）")

    # 时间价值随剩余期限衰减
    days = np.array([90, 60, 30, 20, 10, 5, 2, 1])
    tv = [float(bs_price(S0, K, d / 365, R, SIGMA, "call")) for d in days]
    print("\n平值认购的时间价值随剩余期限变化（内在价值为 0，价格全是时间价值）：")
    print(pd.DataFrame({"剩余天数": days, "价格": np.round(tv, 4),
                        "相对 90 天": [f"{v / tv[0]:.0%}" for v in tv]}).to_string(index=False))
    print("   价格大致按 √T 缩放：剩余时间减为 1/4，时间价值约减为 1/2 → 临近到期衰减加速")

    # 欧式深度实值认沽：时间价值可以为负
    k_deep, t_long, r_high = 4.0, 1.0, 0.05
    p_deep = float(bs_price(S0, k_deep, t_long, r_high, SIGMA, "put"))
    print(f"\n⚠️ 反例：欧式深度实值认沽（S={S0}, K={k_deep}, T=1 年, r={r_high:.0%}）")
    print(f"   价格 = {p_deep:.4f}，内在价值 = {k_deep - S0:.4f}，时间价值 = {p_deep - (k_deep - S0):+.4f}")
    print("   原因：行权拿到的 K 要等到期才能收到，贴现后不如现在就拿；美式认沽可以提前行权，所以不会出现这种情况")

    setup_chinese_font()
    S = np.linspace(2.4, 3.6, 241)
    fig, ax = plt.subplots(figsize=(9, 5))
    for d in [90, 30, 7]:
        ax.plot(S, bs_price(S, K, d / 365, R, SIGMA, "call"), label=f"剩余 {d} 天")
    ax.plot(S, np.maximum(S - K, 0), "k--", label="内在价值（到期）")
    ax.set_xlabel("标的价格 S")
    ax.set_ylabel("认购价格（元）")
    ax.set_title(f"认购价格 = 内在价值 + 时间价值（K={K}, σ={SIGMA:.0%}）")
    ax.legend()
    ax.grid(alpha=0.3)
    path = save_figure(fig, "01_time_value.png", subdir="phase-12")
    print(f"📈 图已保存：{path}")


# ==================== 第四部分：认购-认沽平价 ====================


def put_call_parity():
    """C - P = S - K·e^(-rT) 的数值验证与套利演示"""
    print("\n" + "=" * 60)
    print("第四部分：认购-认沽平价（Put-Call Parity）")
    print("=" * 60)

    print("""
    组合 A：买 1 份认购 + 存入现金 K·e^(-rT)（到期变成 K）
    组合 B：买 1 份认沽 + 持有 1 份标的
    到期时两者价值都等于 max(S_T, K) → 今天的价格必须相等：

              C + K·e^(-rT) = P + S    ⇔    C - P = S - K·e^(-rT)
    """)

    # 1) 到期价值逐点相等
    S_T = np.array([2.5, 2.8, 3.0, 3.2, 3.5])
    A = np.maximum(S_T - K, 0) + K
    B = np.maximum(K - S_T, 0) + S_T
    print("到期价值对比：")
    print(pd.DataFrame({"S_T": S_T, "组合A": A, "组合B": B, "差": A - B}).to_string(index=False))

    # 2) BS 价格满足平价：多个行权价、多个期限
    rows = []
    for k in [2.7, 3.0, 3.3]:
        for d in [7, 30, 180]:
            t = d / 365
            c, p = float(bs_price(S0, k, t, R, SIGMA, "call")), float(bs_price(S0, k, t, R, SIGMA, "put"))
            rows.append([k, d, c, p, c - p, S0 - k * np.exp(-R * t), (c - p) - (S0 - k * np.exp(-R * t))])
    df = pd.DataFrame(rows, columns=["K", "天数", "C", "P", "C-P", "S-Ke^(-rT)", "残差"])
    print("\nBS 价格的平价检验：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"   最大 |残差| = {df['残差'].abs().max():.2e}（浮点误差量级 → 平价精确成立）")

    # 3) 平价被破坏时的套利
    c_fair = float(bs_price(S0, K, T, R, SIGMA, "call"))
    p_fair = float(bs_price(S0, K, T, R, SIGMA, "put"))
    c_mkt = c_fair + 0.0100  # 假设市场上认购被高估 0.01 元
    print(f"\n💰 套利演示：假设市场认购报价 {c_mkt:.4f}（比平价关系隐含的价格高 0.0100）")
    print("   卖贵买贱：卖出认购 + 买入认沽 + 买入标的 + 借入 K·e^(-rT)")
    cash0 = c_mkt - p_fair - S0 + K * np.exp(-R * T)
    S_T = np.array([2.5, 2.8, 3.0, 3.2, 3.5])
    final = -np.maximum(S_T - K, 0) + np.maximum(K - S_T, 0) + S_T - K  # 到期头寸价值（含还款）
    print(f"   今天净现金流 = {cash0:+.4f}，到期头寸价值 = {final}（与 S_T 无关）")
    print(f"   → 每份锁定无风险利润 {cash0:.4f}，每张合约 {cash0 * MULTIPLIER:,.0f} 元（未计成本）")

    print("""
    💡 实际意义：
       1. 用市场上同一 K 的 C - P 可以反推"隐含远期价格" F = K + (C - P)·e^(rT)，
          它自动包含了分红和融资成本预期，06 课计算隐含波动率时会用到
       2. 现实中平价偏离往往小于交易成本、保证金占用和卖空约束的成本，不是免费午餐
       3. 有连续股息率 q 时：C - P = S·e^(-qT) - K·e^(-rT)
    """)


# ==================== 第五部分：杠杆与不对称 ====================


def leverage_demo():
    """同样本金：买 ETF vs 买认购"""
    print("\n" + "=" * 60)
    print("第五部分：期权的杠杆 —— 同样 3 万元买 ETF vs 买认购")
    print("=" * 60)

    capital = 30_000
    c = float(bs_price(S0, K, T, R, SIGMA, "call"))
    n_etf = capital / S0
    n_call = capital / (c * MULTIPLIER)  # 可买的合约张数（允许小数，便于比较）
    rows = []
    for move in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
        s_t = S0 * (1 + move)
        etf_ret = n_etf * s_t / capital - 1
        call_ret = n_call * MULTIPLIER * max(s_t - K, 0) / capital - 1
        rows.append([f"{move:+.0%}", f"{etf_ret:+.1%}", f"{call_ret:+.1%}"])
    print(f"本金 {capital:,} 元：ETF {n_etf:,.0f} 份；或认购 {n_call:.1f} 张（每张 {c * MULTIPLIER:,.0f} 元）")
    print(pd.DataFrame(rows, columns=["到期标的涨跌", "买 ETF 收益", "买认购收益"]).to_string(index=False))
    print("""
    💡 买认购：标的不涨（哪怕持平）就亏掉全部权利金；标的大涨时收益是 ETF 的数倍。
       这种"小概率大收益"的形态让深度虚值期权看起来像便宜彩票——
       但 08 课会看到：当隐含波动率高于随后的实现波动率时，买方的期望收益为负。
    """)


# ==================== 主函数 ====================


def main():
    what_is_option()
    four_positions()
    intrinsic_and_time_value()
    put_call_parity()
    leverage_demo()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 画出"买入认购 K=2.9 + 卖出认购 K=3.1"（牛市价差）的到期盈亏图，并求最大盈亏与盈亏平衡点
    2. 把 r 改为 0、5%、10%，观察欧式深度实值认沽的时间价值如何变化，解释原因
    3. 给定 C=0.1250、P=0.0980、K=3.0、r=2%、T=0.1，用平价反推隐含远期价格，
       并与"现货 3.03"比较，分析可能的原因（分红？利率？报价误差？）
    4. 第五部分中，标的需要涨多少，买认购的收益才能超过买 ETF？用代码求出临界涨幅
    """)


if __name__ == "__main__":
    main()
