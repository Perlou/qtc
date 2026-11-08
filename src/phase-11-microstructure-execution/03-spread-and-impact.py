"""
买卖价差与市场冲击
==================

学习目标：
    1. 理解价差的三个组成部分：订单处理成本、库存成本、逆向选择成本
    2. 用 Glosten-Milgrom 数值例子理解"逆向选择如何产生价差"
    3. 用 Roll 模型仅凭成交价估计价差；分清有效价差、实现价差与价格冲击
    4. 用回归估计 Kyle λ，用平方根法则粗估冲击成本与策略容量

核心概念：
    - 有效价差 = 实现价差 + 价格冲击（恒等式）
    - Roll 模型：Cov(ΔP_t, ΔP_{t-1}) = −S²/4，价差藏在相邻价格变化的负相关里
    - Kyle λ：ΔP = λ·Q，单位净订单流推动价格的幅度，λ 越大流动性越差
    - 平方根法则：冲击 ≈ Y·σ·√(Q/V)，订单翻 4 倍，冲击只翻 2 倍

前置知识：
    - 01、02 课；Phase 3 的回归（OLS）与协方差；Phase 4 的交易成本

环境要求：
    - pip install numpy pandas statsmodels matplotlib
    - 本课全部使用模拟数据，不需要网络

⚠️ 费率与规则以监管和券商最新标准为准，模拟数据只用于理解机制，不代表真实市场的量级。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

from orderbook import simulate_zi_market
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


# ==================== 第一部分：价差的三个组成部分 ====================


def glosten_milgrom(v_high, v_low, p_high, mu):
    """返回 (bid, ask)；p_high 为 V=v_high 的先验概率，mu 为知情交易者比例"""
    p_buy_h, p_buy_l = mu + (1 - mu) / 2, (1 - mu) / 2
    post_h_buy = p_buy_h * p_high / (p_buy_h * p_high + p_buy_l * (1 - p_high))
    p_sell_h, p_sell_l = (1 - mu) / 2, mu + (1 - mu) / 2
    post_h_sell = p_sell_h * p_high / (p_sell_h * p_high + p_sell_l * (1 - p_high))
    ask = post_h_buy * v_high + (1 - post_h_buy) * v_low
    bid = post_h_sell * v_high + (1 - post_h_sell) * v_low
    return bid, ask


def spread_components():
    print("=" * 60)
    print("第一部分：价差的三个组成部分")
    print("=" * 60)

    print("""
    比喻：机场的货币兑换柜台。买入 7.30、卖出 7.10，中间 0.20 是柜台的收入。为什么要这么多？
      1. 房租和工资            → 订单处理成本（Order Processing）
      2. 手里压着外币怕汇率波动 → 库存成本（Inventory）
      3. 偶尔碰上"内行人"必亏  → 逆向选择成本（Adverse Selection）

    Glosten-Milgrom：做市商不知道谁是内行，只能把报价设成"有人来买 / 卖时，价值的条件期望"
        ask = E[V | 买单]，bid = E[V | 卖单]
    真实价值 V 各以 50% 概率为 101 或 99，知情交易者比例为 μ：
    """)
    rows = []
    for mu in [0.0, 0.1, 0.2, 0.5, 0.9]:
        bid, ask = glosten_milgrom(101, 99, 0.5, mu)
        rows.append([mu, round(bid, 4), round(ask, 4), round(ask - bid, 4)])
    print(pd.DataFrame(rows, columns=["知情比例 μ", "bid", "ask", "价差"]).to_string(index=False))
    print("""
    💡 即使没有任何手续费和库存风险，只要市场上有知情交易者，价差就大于 0；
       知情比例越高，价差越大。每笔成交都在向市场泄露信息 → 这就是"价格发现"。
    """)

    # 数值验证：按这个报价，做市商的期望利润应为 0
    rng = np.random.default_rng(1)
    mu, n = 0.2, 2_000_000
    v = rng.choice([101.0, 99.0], n)
    informed = rng.random(n) < mu
    side = np.where(informed, np.where(v > 100, 1, -1), rng.choice([1, -1], n))  # +1 买，−1 卖
    bid, ask = glosten_milgrom(101, 99, 0.5, mu)
    px = np.where(side > 0, ask, bid)
    pnl = side * (px - v)  # 做市商：卖给买方赚 px−v；从卖方买入赚 v−px
    print(f"    蒙特卡洛验证（μ={mu}）：做市商每笔期望利润 = {pnl.mean():+.5f}（标准误 {pnl.std() / np.sqrt(n):.5f}）→ 应接近 0")
    print(f"    其中与知情者成交每笔 {pnl[informed].mean():+.4f}，与噪声交易者成交每笔 {pnl[~informed].mean():+.4f}"
          "：赚噪声交易者的钱，补贴给知情者")


# ==================== 第二部分：Roll 模型 ====================


def roll_spread(price: np.ndarray) -> float:
    """Roll (1984)：S = 2√(−Cov(ΔP_t, ΔP_{t−1}))；协方差为正时模型不适用，返回 NaN"""
    dp = np.diff(price)
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    return 2 * np.sqrt(-cov) if cov < 0 else np.nan


def simulate_roll_prices(n, spread, rho=0.0, vol=0.003, seed=42):
    """成交价 = 真实价值 + 方向 × S/2。rho：相邻成交方向的相关系数（0 = 独立，Roll 的假设）"""
    rng = np.random.default_rng(seed)
    value = 10 + np.cumsum(rng.normal(0, vol, n))
    p_continue = (1 + rho) / 2  # 方向延续的概率，对应一阶自相关 rho
    flips = rng.random(n) >= p_continue
    side = np.where(flips, -1, 1).cumprod() * rng.choice([-1, 1])
    return value + side * spread / 2


def roll_model():
    print("\n" + "=" * 60)
    print("第二部分：Roll 模型 —— 只用成交价估计价差")
    print("=" * 60)

    print("""
    直觉：即使真实价值不变，成交价也会在买价和卖价之间来回"弹跳"（Bid-Ask Bounce）。
          这种弹跳让相邻两次价格变化负相关，负相关的强度反映价差大小。

    模型：P_t = V_t + (S/2)·q_t，V_t 是随机游走，q_t 等概率取 ±1 且互相独立
    推导：ΔP_t = ΔV_t + (S/2)(q_t − q_{t−1})
          Cov(ΔP_t, ΔP_{t−1}) = (S²/4)·Cov(q_t − q_{t−1}, q_{t−1} − q_{t−2}) = −S²/4
          （中间那项只有 −Var(q_{t−1}) = −1 不为零）
    所以：S = 2√(−Cov(ΔP_t, ΔP_{t−1}))
    """)

    true_spread = 0.02
    ests = [roll_spread(simulate_roll_prices(10_000, true_spread, seed=s)) for s in range(50)]
    print(f"    ✅ 公式检验：真实价差 {true_spread}，50 次模拟（每次 1 万笔）Roll 估计 "
          f"均值 {np.nanmean(ests):.4f}，标准差 {np.nanstd(ests):.4f}")

    print("\n    不同样本量下的估计稳定性（各 200 次模拟）：")
    rows = []
    for n in [100, 1000, 10_000]:
        e = np.array([roll_spread(simulate_roll_prices(n, true_spread, seed=1000 + s)) for s in range(200)])
        rows.append([n, np.nanmean(e), np.nanstd(e), np.isnan(e).mean()])
    table = pd.DataFrame(rows, columns=["样本量", "估计均值", "估计标准差", "协方差为正的比例"])
    print(table.round(4).to_string(index=False))

    print("""
    假设被破坏时会怎样？如果买卖方向自相关（方向自相关系数 ρ > 0，如大单被拆成同方向小单）：
        Cov(q_t − q_{t−1}, q_{t−1} − q_{t−2}) = −(1−ρ)²  →  Roll 估计 ≈ S·(1−ρ)，系统性低估
    """)
    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        e = np.nanmean([roll_spread(simulate_roll_prices(50_000, true_spread, rho=rho, seed=2000 + s)) for s in range(10)])
        rows.append([rho, true_spread * (1 - rho), e])
    rho_table = pd.DataFrame(rows, columns=["方向自相关 ρ", "理论 S(1−ρ)", "模拟估计"])
    print(rho_table.round(4).to_string(index=False))
    print("""
    📌 实战：Roll 适合"没有盘口数据、只有成交价"的场景（如历史日线粗估流动性），
       协方差为正时估计无定义 —— 这本身就说明数据里动量、信息成分更强，不能硬算。
    """)
    return rho_table


# ==================== 第三部分：有效价差与实现价差 ====================


def order_level_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """把同一笔主动订单的多笔成交合并为一行（价格取成交量加权）"""
    g = trades.assign(notional=trades["price"] * trades["qty"]).groupby("taker_oid")
    df = g.agg(time=("time", "first"), qty=("qty", "sum"), notional=("notional", "sum"),
               side=("aggressor_true", "first"), mid_before=("mid_before", "first"), last_px=("price", "last"))
    df["price"] = df["notional"] / df["qty"]
    return df.sort_values("time").reset_index(drop=True)


def spread_decomposition(market: dict):
    print("\n" + "=" * 60)
    print("第三部分：有效价差 = 实现价差 + 价格冲击")
    print("=" * 60)

    print("""
    设成交价 P_t，成交前中间价 M_t，方向 q_t（+1 主动买），Δ 为一段时间后：

      有效价差（Effective）= 2 q_t (P_t − M_t)         主动方实际付出的成本
      实现价差（Realized） = 2 q_t (P_t − M_{t+Δ})     被动方扣除价格不利移动后真正赚到的
      价格冲击（Impact）   = 2 q_t (M_{t+Δ} − M_t)     成交后中间价的持续移动（逆向选择的体现）
    """)
    orders = order_level_trades(market["trades"])
    snaps = market["snapshots"].copy()
    snaps["mid"] = (snaps["bid1"] + snaps["ask1"]) / 2
    snaps = snaps.dropna(subset=["mid"])

    rows = []
    for horizon in [3, 30, 60, 300]:
        lookup = orders[["time"]].assign(t_after=orders["time"] + horizon)
        after = pd.merge_asof(lookup, snaps[["time", "mid"]].rename(columns={"time": "t_after", "mid": "mid_after"}),
                              on="t_after", direction="forward")
        ok = after["mid_after"].notna().to_numpy()
        q, p, m0 = orders["side"].to_numpy()[ok], orders["price"].to_numpy()[ok], orders["mid_before"].to_numpy()[ok]
        m1 = after["mid_after"].to_numpy()[ok]
        eff = 2 * q * (p - m0) / m0 * 1e4
        real = 2 * q * (p - m1) / m0 * 1e4
        impact = 2 * q * (m1 - m0) / m0 * 1e4
        identity_err = np.abs(eff - real - impact).max()
        rows.append([horizon, ok.sum(), eff.mean(), real.mean(), impact.mean(),
                     impact.std() / np.sqrt(ok.sum()), identity_err])
    table = pd.DataFrame(rows, columns=["Δ(秒)", "样本数", "有效价差(bp)", "实现价差(bp)", "价格冲击(bp)", "冲击标准误", "恒等式最大误差"])
    print(table.round(3).to_string(index=False))
    q_spread = ((snaps["ask1"] - snaps["bid1"]) / snaps["mid"] * 1e4).mean()
    print(f"""
    平均报价价差 {q_spread:.2f} bp。✅ 恒等式误差约等于机器精度。

    💡 读表：
       - 有效价差与报价价差的差异，主要来自"吃多档"的主动单（成交均价比一档更差）
       - 这个模拟器里没有知情交易者，订单流不携带方向信息，价格冲击只是市价单吃掉流动性的机械效应，
         所以 Δ 变长后冲击并不累积，被动方能留下大部分价差（注意 Δ 越长、噪声越大，看标准误）
       - 08 课加入知情交易者后，冲击会持续存在，实现价差被大幅压缩 —— 这正是做市的难处
    """)

    # 把 Roll 用到这组模拟的成交价上：一笔主动订单取最后一个成交价
    roll_est = roll_spread(orders["last_px"].to_numpy())
    avg_q = (snaps["ask1"] - snaps["bid1"]).mean()
    print(f"    Roll 估计（基于模拟成交价）= {roll_est:.4f} 元；平均报价价差 = {avg_q:.4f} 元")
    print("    📌 两者不必相等：这里价格变化还包含大单吃多档带来的跳动，已经偏离了 Roll 的简化假设")
    return orders, snaps


# ==================== 第四部分：Kyle λ ====================


def kyle_lambda(market: dict, orders: pd.DataFrame, snaps: pd.DataFrame, bar_seconds: int = 60):
    print("\n" + "=" * 60)
    print("第四部分：Kyle λ —— 用回归度量价格冲击")
    print("=" * 60)

    print("""
    Kyle (1985)：ΔP = λ·Q，Q 是净订单流（买入量 − 卖出量）。
    λ 是市场"深度"的倒数：λ 越大，同样的订单推动价格越多，流动性越差。
    """)

    # 先用已知 λ 的合成数据检验估计方法
    rng = np.random.default_rng(3)
    true_lambda = 2e-5
    q = rng.normal(0, 2000, 2000)
    dp = true_lambda * q + rng.normal(0, 0.01, 2000)
    fit = sm.OLS(dp, sm.add_constant(q)).fit()
    print(f"    ✅ 方法检验：合成数据真实 λ = {true_lambda:.2e}，OLS 估计 = {fit.params[1]:.2e} "
          f"（标准误 {fit.bse[1]:.2e}）")

    # 用零智能市场的数据估计：按 bar_seconds 聚合
    bars = snaps[["time", "mid"]].copy()
    bars["bar"] = (bars["time"] // bar_seconds).astype(int)
    mid_end = bars.groupby("bar")["mid"].last()
    dm = mid_end.diff()
    flow = (orders["side"] * orders["qty"]).groupby((orders["time"] // bar_seconds).astype(int)).sum()
    df = pd.DataFrame({"dm": dm, "flow": flow}).dropna()
    model = sm.OLS(df["dm"], sm.add_constant(df["flow"])).fit()
    lam = model.params["flow"]
    print(f"""
    模拟市场（{bar_seconds} 秒 K 线，{len(df)} 根）回归 ΔM = c + λ·净主动买量：
      λ = {lam:.3e} 元/股（t = {model.tvalues['flow']:.1f}），R² = {model.rsquared:.3f}
      含义：净买入 1 万股，中间价约变动 {lam * 1e4:.4f} 元（{lam * 1e4 / mid_end.mean() * 1e4:.1f} bp）
    """)
    rows = []
    for sec in [30, 60, 300]:
        b = (snaps["time"] // sec).astype(int)
        d = snaps.groupby(b)["mid"].last().diff()
        f = (orders["side"] * orders["qty"]).groupby((orders["time"] // sec).astype(int)).sum()
        tmp = pd.DataFrame({"dm": d, "flow": f}).dropna()
        m = sm.OLS(tmp["dm"], sm.add_constant(tmp["flow"])).fit()
        rows.append([sec, len(tmp), m.params["flow"], m.tvalues["flow"], m.rsquared])
    print("    不同聚合周期：")
    print("    " + pd.DataFrame(rows, columns=["周期(秒)", "样本", "λ", "t 值", "R²"])
          .to_string(index=False, float_format=lambda x: f"{x:.3g}").replace("\n", "\n    "))
    print("""
    📌 实践：真实数据用分钟线 + 主动买卖量（02 课的 Lee-Ready）回归。
       λ 会随时间、股票变化；小盘股 λ 通常更大。
    """)
    return df, lam


# ==================== 第五部分：平方根法则与策略容量 ====================


def sqrt_impact_bp(order_shares, adv_shares, daily_vol, Y=1.0):
    """平方根法则冲击成本（单位：bp）：Y·σ·√(Q/V)"""
    return Y * daily_vol * np.sqrt(order_shares / adv_shares) * 1e4


def sqrt_law_and_capacity():
    print("\n" + "=" * 60)
    print("第五部分：平方根法则、交易成本全景与策略容量")
    print("=" * 60)

    print("""
    大量实证发现：在一段时间内执行完的大单（母单），总冲击成本 ∝ 规模的平方根，而不是线性：
        ΔP / P ≈ Y · σ · √(Q / V)
    σ：日波动率；Q：订单量；V：日成交量；Y：经验常数，量级约为 1（需要用自己的成交数据校准）
    """)
    rows = [[f"{p:.1%}", sqrt_impact_bp(p, 1.0, 0.02), sqrt_impact_bp(p, 1.0, 0.03)] for p in [0.001, 0.01, 0.05, 0.1, 0.2]]
    print(pd.DataFrame(rows, columns=["参与度 Q/V", "σ=2% 冲击(bp)", "σ=3% 冲击(bp)"]).round(1).to_string(index=False))
    print(f"    ✅ 规模翻 4 倍，冲击翻 {sqrt_impact_bp(0.04, 1, 0.02) / sqrt_impact_bp(0.01, 1, 0.02):.1f} 倍")

    print("\n    交易成本冰山 —— 买入后再卖出 1000 万元、ADV 1 亿元、股价 10 元、日波动 2% 的一只股票：")
    notional, adv, price, vol = 1e7, 1e8, 10.0, 0.02
    commission_bp, stamp_bp, transfer_bp = 1.5, 5.0, 0.1  # 示例费率：万 1.5 佣金、卖出印花税 0.05%、过户费 0.001%
    half_spread_bp = 0.5 * 0.01 / price * 1e4
    impact_bp = sqrt_impact_bp(notional, adv, vol)
    costs = pd.DataFrame(
        [
            ["佣金（示例万 1.5，双边）", commission_bp * 2, "显性"],
            ["印花税（卖出 0.05%）", stamp_bp, "显性"],
            ["过户费（示例 0.001%，双边）", transfer_bp * 2, "显性"],
            ["半价差（1 tick，双边）", half_spread_bp * 2, "隐性"],
            ["市场冲击（平方根法则，双边）", impact_bp * 2, "隐性"],
        ],
        columns=["成本项", "bp", "类型"],
    )
    costs["元"] = (costs["bp"] / 1e4 * notional).round(0)
    print(costs.round(2).to_string(index=False))
    print(f"    合计 {costs['bp'].sum():.1f} bp，其中隐性成本占 {costs.loc[costs['类型'] == '隐性', 'bp'].sum() / costs['bp'].sum():.0%}")
    print("    ⚠️ 费率为示例值，以监管和券商最新标准为准")

    print("""
    策略容量（Capacity）：资金越大 → 参与度越高 → 冲击越大 → 吃掉 alpha。
    设单只股票单次调仓占组合权重 w，ADV 为 V（元），单次调仓预期 alpha 为 a（bp），令冲击 = a：
        Y·σ·√(AUM·w / V) · 10⁴ = a   →   AUM = (a / (Y·σ·10⁴))² · V / w
    """)
    w, adv_cap, sigma, rows = 0.02, 2e8, 0.02, []
    for alpha in [10, 20, 30, 50, 100]:
        aum = (alpha / (sigma * 1e4)) ** 2 * adv_cap / w
        check = sqrt_impact_bp(aum * w, adv_cap, sigma)
        rows.append([alpha, aum / 1e8, check])
    print(f"    示例：w = {w:.0%}，ADV = {adv_cap / 1e8:.0f} 亿元，σ = {sigma:.0%}，Y = 1")
    print(pd.DataFrame(rows, columns=["单次 alpha(bp)", "容量 AUM(亿元)", "代回验证冲击(bp)"]).round(2).to_string(index=False))
    print("""
    💡 alpha 翻倍，容量翻 4 倍；这只是把 alpha 全部吃光的上限，真实可用容量要低得多。
       Phase 14 策略尽调会把这个估算做成"资金规模-净收益"曲线。
    """)


# ==================== 可视化 ====================


def plot_all(rho_table: pd.DataFrame, kyle_df: pd.DataFrame, lam: float):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(rho_table["方向自相关 ρ"], rho_table["理论 S(1−ρ)"], "k--", label="理论 S(1−ρ)")
    ax.plot(rho_table["方向自相关 ρ"], rho_table["模拟估计"], "o", color="tab:blue", label="模拟估计")
    ax.axhline(0.02, color="tab:red", ls=":", label="真实价差")
    ax.set_title("Roll 估计：方向自相关导致低估")
    ax.set_xlabel("方向自相关 ρ")
    ax.set_ylabel("估计价差（元）")
    ax.legend()

    ax = axes[1]
    ax.scatter(kyle_df["flow"], kyle_df["dm"], s=10, alpha=0.5)
    xs = np.linspace(kyle_df["flow"].min(), kyle_df["flow"].max(), 50)
    ax.plot(xs, lam * xs, color="tab:red", label=f"λ = {lam:.2e}")
    ax.set_title("Kyle λ：净主动买量 vs 中间价变化（模拟）")
    ax.set_xlabel("净主动买量（股）")
    ax.set_ylabel("ΔM（元）")
    ax.legend()

    ax = axes[2]
    x = np.linspace(0, 0.3, 200)
    sqrt_cost = sqrt_impact_bp(x, 1.0, 0.02)
    ax.plot(x * 100, sqrt_cost, label="平方根法则")
    slope = sqrt_impact_bp(0.01, 1.0, 0.02) / 0.01
    ax.plot(x * 100, slope * x, "--", label="线性外推（从 1% 参与度）")
    ax.set_ylim(0, sqrt_cost.max() * 1.1)
    ax.set_title("冲击成本：平方根 vs 线性")
    ax.set_xlabel("参与度 Q/V（%）")
    ax.set_ylabel("冲击成本（bp）")
    ax.legend()

    path = save_figure(fig, "03_spread_impact.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    spread_components()
    rho_table = roll_model()
    market = simulate_zi_market(seconds=7200, seed=2026)
    orders, snaps = spread_decomposition(market)
    kyle_df, lam = kyle_lambda(market, orders, snaps)
    sqrt_law_and_capacity()
    plot_all(rho_table, kyle_df, lam)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 Phase 1 的 A 股日线数据计算 20 日滚动 Roll 价差，观察它有多少天无定义（协方差为正）
    2. 给 simulate_zi_market 增加"知情交易者"（提前知道下一步价格方向），比较实现价差和价格冲击的变化
    3. 改变 simulate_zi_market 的 limit_rate（流动性），观察 Kyle λ 如何变化
    4. 为你的一个策略估算容量：换手率、持股数、平均 ADV 代入第五部分的公式
    """)


if __name__ == "__main__":
    main()
