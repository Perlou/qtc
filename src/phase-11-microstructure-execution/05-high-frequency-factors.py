"""
高频因子：订单不平衡与成交量分布
================================

学习目标：
    1. 计算三个经典盘口因子：订单簿不平衡（OBI）、微价格（Microprice）、订单流不平衡（OFI）
    2. 在模拟的 L1 快照上验证它们对短期价格变动的预测力，并观察信号随预测期限衰减
    3. 复现 Cont 等人的发现：短期中间价变化与 OFI 近似线性，斜率与深度成反比
    4. 理解 A 股"高频数据 → 日频因子"的低频化思路

核心概念：
    - OBI = (Q^b − Q^a) / (Q^b + Q^a)：静态的盘口状态
    - OFI（Cont, Kukanov & Stoikov, 2014）：相邻快照之间买一 / 卖一价量的变化，动态的盘口压力
    - 微价格 = 中间价 + 半价差 × OBI：比中间价更好的"公允价"估计
    - IC（信息系数）：因子值与未来收益的秩相关系数（Phase 7）

前置知识：
    - 02、04 课（快照数据、OBI 与微价格）；Phase 3 的回归；Phase 7 的 IC

环境要求：
    - pip install numpy pandas scipy statsmodels matplotlib
    - 本课使用 orderbook.simulate_l1_queue 生成的模拟数据，不需要网络
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

from orderbook import intraday_volume_profile, simulate_l1_queue
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TICK = 0.01


def compute_ofi(df: pd.DataFrame) -> np.ndarray:
    """
    Cont 等人的 OFI：e_n = 1{Pb_n ≥ Pb_{n−1}}·qb_n − 1{Pb_n ≤ Pb_{n−1}}·qb_{n−1}
                         − 1{Pa_n ≤ Pa_{n−1}}·qa_n + 1{Pa_n ≥ Pa_{n−1}}·qa_{n−1}
    df 需含 bid_px, bid_sz, ask_px, ask_sz；价格先换成整数 tick 再比较，避免浮点误差。
    返回长度 len(df)−1 的数组（第 n 个元素对应快照 n−1 → n）。
    """
    bp = np.round(df["bid_px"].to_numpy() / TICK).astype(int)
    ap = np.round(df["ask_px"].to_numpy() / TICK).astype(int)
    bq, aq = df["bid_sz"].to_numpy(), df["ask_sz"].to_numpy()
    eb = np.where(bp[1:] >= bp[:-1], bq[1:], 0) - np.where(bp[1:] <= bp[:-1], bq[:-1], 0)
    ea = np.where(ap[1:] <= ap[:-1], aq[1:], 0) - np.where(ap[1:] >= ap[:-1], aq[:-1], 0)
    return eb - ea


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["spread"] = df["ask_px"] - df["bid_px"]
    df["obi"] = (df["bid_sz"] - df["ask_sz"]) / (df["bid_sz"] + df["ask_sz"])
    df["micro"] = (df["ask_px"] * df["bid_sz"] + df["bid_px"] * df["ask_sz"]) / (df["bid_sz"] + df["ask_sz"])
    df["ofi"] = np.concatenate([[np.nan], compute_ofi(df)])  # 快照 n−1 → n 的 OFI，记在 n 上
    df["dmid"] = df["mid"].diff()  # 同一区间的中间价变化
    df["depth"] = (df["bid_sz"] + df["ask_sz"]) / 2
    return df


# ==================== 第一部分：三个盘口因子 ====================


def factor_definitions():
    print("=" * 60)
    print("第一部分：OBI、微价格与 OFI")
    print("=" * 60)
    print("""
    OBI    = (Q^b − Q^a) / (Q^b + Q^a)                  → 静态盘口状态，∈ [−1, 1]
    微价格 = (P^a·Q^b + P^b·Q^a) / (Q^b + Q^a)          → 公允价估计 = 中间价 + 半价差 × OBI
    OFI    = Σ e_n（相邻快照之间买一 / 卖一的价量变化）  → 动态盘口压力

    OFI 逐项解读（买方，卖方对称、符号相反）：
      买一价不变 → q^b_n − q^b_{n−1}：买一量增加 = 买压上升
      买一价上移 → +q^b_n：新价位上的量都是新增买压
      买一价下移 → −q^b_{n−1}：原价位的量全部消失（被吃或撤单）
    """)
    demo = pd.DataFrame({
        "bid_px": [10.00, 10.00, 10.01, 10.01], "bid_sz": [500, 800, 300, 100],
        "ask_px": [10.01, 10.01, 10.02, 10.02], "ask_sz": [400, 400, 600, 900],
    })
    print(demo.to_string())
    print(f"    compute_ofi → {compute_ofi(demo).tolist()}（CONCEPT 示例的答案为 [300, 700, -500]）")


# ==================== 第二部分：模拟数据 ====================


def simulated_data() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：一档队列模型生成的 L1 快照")
    print("=" * 60)
    raw = simulate_l1_queue(n_events=300_000, seed=7, snapshot_every=20)
    df = add_features(raw)
    spread_ticks = (df["spread"] / TICK).round().astype(int)
    print(f"""
    模型（orderbook.simulate_l1_queue）：只跟踪买一 / 卖一两个队列，
      限价单加入队列、撤单和市价单消耗队列；队列被耗尽时价格才跳动一个 tick。
    共 {len(raw) * 20:,} 个事件，每 20 个事件拍一张快照 → {len(df):,} 张
    价差分布（tick）：{spread_ticks.value_counts().sort_index().head(5).to_dict()}
    中间价范围 {df['mid'].min():.3f} ~ {df['mid'].max():.3f}，一档平均深度 {df['depth'].mean():,.0f} 股
    """)
    print(df[["bid_px", "bid_sz", "ask_px", "ask_sz", "obi", "micro", "ofi", "dmid"]].head(6).round(4).to_string())
    print("""
    ⚠️ 这个模型里价格只在队列耗尽时变动，"哪边队列短"天然包含方向信息。
       真实市场还有隐藏单、跨市场信息、知情交易者等，因子效果要用真实数据重新检验。
    """)
    return df


# ==================== 第三部分：OBI 与微价格的预测力 ====================


def obi_predictive_power(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：OBI 与微价格能预测下一步价格吗？")
    print("=" * 60)

    fwd = (df["mid"].shift(-1) - df["mid"]) / TICK  # 下一张快照的中间价变化（tick）
    tmp = pd.DataFrame({"obi": df["obi"], "fwd": fwd}).dropna()
    tmp["bucket"] = pd.qcut(tmp["obi"], 10, labels=False) + 1
    stats = tmp.groupby("bucket").agg(
        OBI均值=("obi", "mean"), 下一步变化_tick=("fwd", "mean"),
        上涨概率=("fwd", lambda x: (x > 0).mean()), 下跌概率=("fwd", lambda x: (x < 0).mean()),
    )
    print("    按 OBI 十分位分组，下一张快照的中间价变化：")
    print("    " + stats.round(3).to_string().replace("\n", "\n    "))
    spread_mono = stats["下一步变化_tick"].is_monotonic_increasing
    print(f"    十组均值是否单调递增：{'是' if spread_mono else '否'}；"
          f"最高组 − 最低组 = {stats['下一步变化_tick'].iloc[-1] - stats['下一步变化_tick'].iloc[0]:.3f} tick")

    print("\n    信号衰减：OBI 与未来 h 张快照中间价变化的 IC（Spearman 秩相关）")
    horizons = [1, 2, 5, 10, 20, 50, 100]
    ic_rows = []
    for h in horizons:
        f = df["mid"].shift(-h) - df["mid"]
        ic_rows.append([h, df["obi"].corr(f, method="spearman"),
                        (df["micro"] - df["mid"]).corr(f, method="pearson")])
    ic = pd.DataFrame(ic_rows, columns=["h（快照）", "OBI 的 IC", "微价格偏移 的相关系数"])
    print("    " + ic.round(4).to_string(index=False).replace("\n", "\n    "))
    print("""
    💡 预测期限越长，未来价格变化里与当前盘口无关的噪声越多，相关性随之下降。
       高频信号"保质期"极短：你算出信号再下单时，优势可能已经没了。
    """)

    print("    用'当前价格'预测 h 步后的中间价，比较中间价与微价格（均方误差，单位 tick²）：")
    rows = []
    for h in [1, 5, 20]:
        target = df["mid"].shift(-h)
        ok = target.notna()
        mse_mid = (((target - df["mid"]) / TICK)[ok] ** 2).mean()
        mse_micro = (((target - df["micro"]) / TICK)[ok] ** 2).mean()
        rows.append([h, mse_mid, mse_micro, 1 - mse_micro / mse_mid])
    mse = pd.DataFrame(rows, columns=["h", "中间价 MSE", "微价格 MSE", "误差降低"])
    mse["误差降低"] = mse["误差降低"].map(lambda x: f"{x:.1%}")
    print("    " + mse.round(4).to_string(index=False).replace("\n", "\n    "))
    print("""
    📌 微价格的修正量被限制在半个价差以内，而 h 步后的价格变化可能远大于此，所以 MSE 的改进幅度有限；
       它的主要价值是作为做市报价中心、执行算法的"参考价"。Stoikov (2018) 给出了更严谨的微价格定义。
    """)
    return stats, ic


# ==================== 第四部分：OFI 与价格变化 ====================


def ofi_analysis(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：OFI —— ΔM ≈ β·OFI")
    print("=" * 60)

    d = df.dropna(subset=["ofi", "dmid"]).copy()
    d["trade_imb"] = d["buy_vol"] - d["sell_vol"]  # 区间内主动买 − 主动卖成交量
    rows = []
    for name, x in [("OFI", "ofi"), ("成交量不平衡（只看成交）", "trade_imb")]:
        m = sm.OLS(d["dmid"] / TICK, sm.add_constant(d[x])).fit()
        rows.append([name, m.params[x], m.tvalues[x], m.rsquared])
    print("    同一区间回归：ΔM（tick）= c + β·X")
    print("    " + pd.DataFrame(rows, columns=["解释变量 X", "β", "t 值", "R²"])
          .to_string(index=False, float_format=lambda v: f"{v:.4g}").replace("\n", "\n    "))
    r2_ofi, r2_trade = rows[0][3], rows[1][3]
    verdict = "更强" if r2_ofi > r2_trade else "并不更强"
    print(f"""
    💡 本次模拟中 OFI 的 R² 是成交量不平衡的 {r2_ofi / r2_trade:.1f} 倍（解释力{verdict}）。
       OFI 同时包含成交、新挂单和撤单，Cont 等人在真实数据上也发现它比只看成交的指标解释力更强。
       注意这是"同期"关系：它解释价格为什么变，而不是提前预测。
    """)

    lag = sm.OLS((d["dmid"].shift(-1) / TICK).iloc[:-1], sm.add_constant(d["ofi"].iloc[:-1])).fit()
    print(f"    预测关系：下一区间 ΔM 对本区间 OFI 回归，R² = {lag.rsquared:.4f}，t = {lag.tvalues['ofi']:.1f}")

    d["depth_bucket"] = pd.qcut(d["depth"].shift(1), 3, labels=["浅", "中", "深"])
    rows = []
    for b, g in d.dropna(subset=["depth_bucket"]).groupby("depth_bucket", observed=True):
        m = sm.OLS(g["dmid"] / TICK, sm.add_constant(g["ofi"])).fit()
        rows.append([b, g["depth"].mean(), m.params["ofi"], m.rsquared])
    depth_tab = pd.DataFrame(rows, columns=["区间开始时的深度", "平均深度(股)", "β", "R²"])
    print("\n    按区间开始时的一档深度分三组，分别估计 β：")
    print("    " + depth_tab.to_string(index=False, float_format=lambda v: f"{v:.4g}").replace("\n", "\n    "))
    mono = depth_tab["β"].is_monotonic_decreasing
    print(f"""
    本次结果中 β 随深度单调下降：{'是' if mono else '否'}
    💡 理论上深度越大，同样的 OFI 推动价格越少（β ∝ 1/深度）。
       与 03 课的 Kyle λ 是同一个思想：流动性越好，冲击越小。
    """)
    return d


# ==================== 第五部分：高频因子低频化 ====================


def daily_factors():
    print("\n" + "=" * 60)
    print("第五部分：A 股的思路 —— 高频数据聚合为日频因子")
    print("=" * 60)
    print("""
    A 股股票 T+1，无法做日内往返。常见做法是把高频数据聚合成日频因子，再做截面选股（Phase 7）：
      高频快照 / 逐笔 ──▶ 日内聚合 ──▶ 日频因子 ──▶ 截面标准化 ──▶ 选股
    """)
    profile = intraday_volume_profile(240)
    rng = np.random.default_rng(5)
    rows = []
    for day in range(20):
        raw = simulate_l1_queue(n_events=24_000, seed=100 + day, snapshot_every=100)  # 240 张 ≈ 每分钟一张
        f = add_features(raw)
        minute_vol = profile * rng.lognormal(0, 0.25, 240)
        minute_vol = minute_vol / minute_vol.sum()
        r = np.log(f["mid"]).diff().dropna()
        rows.append({
            "day": day + 1,
            "日收益(bp)": (f["mid"].iloc[-1] / f["mid"].iloc[0] - 1) * 1e4,
            "已实现波动(bp)": np.sqrt((r**2).sum()) * 1e4,
            "OBI 均值": f["obi"].mean(),
            "OFI/深度": f["ofi"].sum() / f["depth"].mean(),
            "主动买入占比": f["buy_vol"].sum() / (f["buy_vol"].sum() + f["sell_vol"].sum()),
            "开盘30分钟占比": minute_vol[:30].sum(),
            "尾盘30分钟占比": minute_vol[-30:].sum(),
        })
    table = pd.DataFrame(rows).set_index("day")
    print(table.head(8).round(3).to_string())
    corr = table["OFI/深度"].corr(table["日收益(bp)"])
    print(f"""
    同日 OFI/深度 与日收益的相关系数：{corr:.2f}
    ⚠️ 这是"同期"相关（当天的订单流解释当天的价格），不能直接当选股信号；
       真正的检验是：今天的因子能否预测明天及以后的收益 → 用 Phase 7 的 IC / 分组回测框架
    📌 成交量分布类因子：开盘 / 尾盘成交占比、日内 U 型偏离程度、大单净流入、VPIN 等，
       都是"高频数据低频化"的例子；阈值类因子（如"大单"）定义主观，需谨慎解读
    """)
    return table


# ==================== 可视化 ====================


def plot_all(stats, ic, d):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    ax.bar(stats.index, stats["下一步变化_tick"], color=np.where(stats["下一步变化_tick"] > 0, "tab:red", "tab:green"))
    ax.set_title("OBI 十分位 → 下一步中间价变化（模拟）")
    ax.set_xlabel("OBI 分组（1 = 卖盘最厚，10 = 买盘最厚）")
    ax.set_ylabel("平均变化（tick）")

    ax = axes[1]
    ax.plot(ic["h（快照）"], ic["OBI 的 IC"], marker="o")
    ax.set_xscale("log")
    ax.set_title("信号衰减：OBI 的 IC vs 预测期限")
    ax.set_xlabel("预测期限 h（快照数，对数坐标）")
    ax.set_ylabel("IC")

    ax = axes[2]
    sample = d.sample(3000, random_state=0)
    jitter = np.random.default_rng(0).normal(0, 0.05, len(sample))
    ax.scatter(sample["ofi"], sample["dmid"] / TICK + jitter, s=4, alpha=0.3)
    m = sm.OLS(d["dmid"] / TICK, sm.add_constant(d["ofi"])).fit()
    xs = np.linspace(d["ofi"].quantile(0.01), d["ofi"].quantile(0.99), 50)
    ax.plot(xs, m.params["const"] + m.params["ofi"] * xs, color="tab:red", label=f"β = {m.params['ofi']:.2e}")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_title("OFI vs 同期中间价变化（模拟）")
    ax.set_xlabel("OFI（股）")
    ax.set_ylabel("ΔM（tick，加少量抖动便于观察）")
    ax.legend()

    path = save_figure(fig, "05_hf_factors.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    factor_definitions()
    df = simulated_data()
    stats, ic = obi_predictive_power(df)
    d = ofi_analysis(df)
    daily_factors()
    plot_all(stats, ic, d)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 OBI 扩展到前 k 档（需要 02 课的十档快照），比较 k = 1、3、5、10 时的 IC
    2. 计算多期 OFI（把最近 5 个区间的 OFI 加总），检验它对下一区间价格变化的预测力
    3. 修改 simulate_l1_queue 的事件概率（例如提高撤单比例），观察 OBI 预测力的变化，并解释原因
    4. 用 04 课采样的真实 OKX 盘口（至少几百个样本）计算 OBI 与下一步中间价变化的相关性
    """)


if __name__ == "__main__":
    main()
