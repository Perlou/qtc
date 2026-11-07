"""
蒙特卡洛模拟与 Bootstrap
========================

学习目标：
    1. 理解"回测只是无数种可能历史中的一条"，学会用模拟得到结果的分布
    2. 用几何布朗运动（GBM）模拟价格路径，计算分位数、亏损概率
    3. 理解蒙特卡洛误差按 1/√N 缩小，知道模拟次数该怎么选
    4. 用 Bootstrap 估计 Sharpe 的置信区间
    5. 用 Block Bootstrap 保留波动聚集，得到更现实的最大回撤分布

核心概念：
    - 蒙特卡洛：假设一个模型，用随机数生成大量"平行宇宙"（菜谱错了，结果全错）
    - Bootstrap：不假设模型，从历史数据有放回地重抽样（只能用已有的食材）
    - GBM：S_{t+Δt} = S_t · exp[(μ - σ²/2)Δt + σ√Δt · Z]
    - Block Bootstrap：一次抽一段连续数据，保留自相关与波动聚集

前置知识：
    - 01-probability-distributions.py、03-hypothesis-testing.py

环境要求：
    - pip install numpy pandas scipy arch matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据没有波动聚集，第四部分额外用 GARCH 模拟数据演示）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap
from dotenv import load_dotenv
from scipy import stats

from utils.data import get_index_daily, get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRADING_DAYS = 252
START, END = "20150101", "20241231"


def simulate_gbm(s0: float, mu: float, sigma: float, days: int = TRADING_DAYS,
                 n_paths: int = 10_000, seed: int = 42) -> np.ndarray:
    """GBM 价格路径，返回 shape = (days, n_paths)；mu、sigma 为年化参数"""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    z = rng.standard_normal((days, n_paths))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.cumsum(log_ret, axis=0))


def sharpe(r: np.ndarray, axis=None) -> np.ndarray:
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1) * np.sqrt(TRADING_DAYS)


def max_drawdown(r: np.ndarray, axis: int = 0) -> np.ndarray:
    """简单收益 → 最大回撤（负数）。r 可以是 (天数,) 或 (天数, 路径数)"""
    nav = np.cumprod(1 + r, axis=axis)
    return (nav / np.maximum.accumulate(nav, axis=axis) - 1).min(axis=axis)


# ==================== 第一部分：GBM 蒙特卡洛 ====================


def gbm_demo():
    print("=" * 60)
    print("第一部分：用 GBM 模拟茅台一年后的价格")
    print("=" * 60)

    print("""
    🌌 历史只发生了一次。你的回测净值曲线只是"无数种可能历史"中的一条。
       蒙特卡洛 = 按照菜谱（模型）做 10000 次菜；Bootstrap = 把做好的菜打散重新装盘。

    GBM：S_{t+Δt} = S_t · exp[(μ - σ²/2)Δt + σ√Δt · Z]，Z ~ N(0,1)
       μ 年化漂移，σ 年化波动，Δt = 1/252，-σ²/2 是伊藤修正项（Phase 2 的"波动拖累"）
    """)

    df = get_stock_daily_adj("600519", START, END)
    ret = np.log(df["adj_close"]).diff().dropna()
    s0 = float(df["close"].iloc[-1])
    sigma = ret.std() * np.sqrt(TRADING_DAYS)
    mu = 0.08  # 期望收益极难估计（01 课），这里人为设定一个"温和"的假设，而不是用历史均值
    print(f"📐 参数：S0 = {s0:.2f}（最新不复权收盘价），σ = {sigma:.1%}（历史估计），μ = {mu:.0%}（人为假设）")

    paths = simulate_gbm(s0, mu, sigma, n_paths=10_000)
    final = paths[-1]
    q = np.percentile(final, [5, 25, 50, 75, 95])
    print("\n🎲 10,000 条路径，一年后价格分位数：")
    print("   " + "  ".join(f"{p}%: {v:,.0f}" for p, v in zip([5, 25, 50, 75, 95], q)))
    print(f"   一年后亏损的概率 = {(final < s0).mean():.1%}")
    print(f"   一年后跌幅超过 30% 的概率 = {(final < 0.7 * s0).mean():.1%}")
    path_mdd = (paths / np.maximum.accumulate(paths, axis=0) - 1).min(axis=0)
    print(f"   年内最大回撤中位数 = {np.median(path_mdd):.1%}，5% 最坏 = {np.percentile(path_mdd, 5):.1%}")
    print("""
    💡 即使期望收益为正，一年期的结果仍然高度不确定。
       注意：GBM 假设正态、常波动、独立 —— 这三条在真实市场中都不成立，尾部会被低估。
    """)
    return paths, s0


# ==================== 第二部分：模拟次数与蒙特卡洛误差 ====================


def mc_convergence(s0: float):
    print("\n" + "=" * 60)
    print("第二部分：模拟多少次才够？")
    print("=" * 60)

    print("""
    蒙特卡洛估计的是一个"样本均值"，误差同样服从 σ/√N：
       模拟次数 ×100 → 误差 ÷10
    """)
    # 待估量：一年后亏损的概率（GBM 下有解析解，可以对照）
    mu, sigma = 0.08, 0.30
    exact = stats.norm.cdf(-(mu - 0.5 * sigma**2) / sigma)
    rows = []
    for n in [100, 1_000, 10_000, 100_000]:
        ests = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            log_total = rng.normal(mu - 0.5 * sigma**2, sigma, n)  # 一年总对数收益
            ests.append((log_total < 0).mean())
        ests = np.array(ests)
        rows.append([f"{n:,}", f"{ests.mean():.4f}", f"{ests.std():.4f}", f"{np.sqrt(exact * (1 - exact) / n):.4f}"])
    print(f"📏 估计'一年后亏损的概率'（解析解 = {exact:.4f}），每种次数用 20 个不同种子重复：")
    print(pd.DataFrame(rows, columns=["模拟次数", "20 次估计的均值", "估计值的标准差", "理论标准误 √(p(1-p)/N)"]).to_string(index=False))
    print("""
    💡 估概率、均值，几千到一万次通常够用；估 1% 这样的极端分位数需要更多次数。
       报告模拟结果时，务必写明模拟次数和随机种子。
    """)


# ==================== 第三部分：Bootstrap Sharpe 置信区间 ====================


def bootstrap_sharpe_ci(r: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(r)
    idx = rng.integers(0, n, size=(n_boot, n))
    sims = sharpe(r[idx], axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return lo, hi, sims


def bootstrap_demo() -> pd.Series:
    print("\n" + "=" * 60)
    print("第三部分：Bootstrap 估计 Sharpe 的置信区间")
    print("=" * 60)

    close = get_index_daily("000300", START, END)["close"]
    ret = close.pct_change()
    pos = (close > close.rolling(60).mean()).astype(float).shift(1)
    strat = (pos * ret - pos.diff().abs() * 0.001).dropna()

    r = strat.to_numpy()
    sr = sharpe(r)
    lo, hi, sims = bootstrap_sharpe_ci(r)
    years = len(r) / TRADING_DAYS
    se_analytic = np.sqrt((1 + 0.5 * sr**2) / years)  # Lo (2002) 在 iid 假设下的近似
    print(f"📊 沪深 300 60 日均线择时（同 03 课），{years:.1f} 年")
    print(f"   样本 Sharpe = {sr:.2f}")
    print(f"   Bootstrap 95% 置信区间 = [{lo:.2f}, {hi:.2f}]（5000 次重抽样）")
    print(f"   解析近似 95% 区间 = [{sr - 1.96 * se_analytic:.2f}, {sr + 1.96 * se_analytic:.2f}]"
          f"（SE ≈ √((1 + SR²/2)/年数) = {se_analytic:.2f}）")
    print(f"   Bootstrap 中 Sharpe ≤ 0 的比例 = {(sims <= 0).mean():.1%}")
    verdict = "包含 0 → 证据不足以说明策略 Sharpe 为正" if lo <= 0 else "不包含 0 → 在这个口径下 Sharpe 显著为正"
    print(f"   结论：置信区间{verdict}")
    print("""
    💡 Bootstrap 不需要假设分布形状，但普通 Bootstrap 逐日独立抽样，假设了收益"独立同分布"。
    """)
    return strat


# ==================== 第四部分：Block Bootstrap 与最大回撤分布 ====================


def simulate_garch(n: int, seed: int = 2) -> np.ndarray:
    """GARCH(1,1) 模拟收益：用于在离线 / 无波动聚集的数据上演示 Block Bootstrap 的作用"""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.08, 0.91
    r = np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return r + 0.0003


def iid_bootstrap_paths(r: np.ndarray, horizon: int, n_paths: int, seed: int) -> np.ndarray:
    """普通 Bootstrap：逐日独立有放回抽样，返回 shape = (horizon, n_paths)"""
    rng = np.random.default_rng(seed)
    return r[rng.integers(0, len(r), size=(horizon, n_paths))]


def block_bootstrap_paths(r: np.ndarray, horizon: int, n_paths: int, block: int, seed: int) -> np.ndarray:
    """移动块自助法（向量化版）：每条路径由若干长度为 block 的连续片段拼接而成"""
    rng = np.random.default_rng(seed)
    n_blocks = horizon // block + 1
    starts = rng.integers(0, len(r) - block, size=(n_blocks, n_paths))
    idx = (starts[:, None, :] + np.arange(block)[None, :, None]).reshape(n_blocks * block, n_paths)[:horizon]
    return r[idx]


def abs_acf1(x: np.ndarray) -> float:
    """|收益| 的一阶自相关：衡量波动聚集的简单指标"""
    a = np.abs(x) - np.abs(x).mean()
    return float((a[1:] * a[:-1]).sum() / (a * a).sum())


def q20(r: np.ndarray, q: float = 1) -> float:
    """滚动 20 日对数收益的 q% 分位数（r 可以是一维或 (天数, 路径数)）"""
    c = np.cumsum(np.log1p(r), axis=0)
    return float(np.percentile(c[20:] - c[:-20], q))


def block_bootstrap_demo(strat: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：Block Bootstrap —— 保留时间结构")
    print("=" * 60)

    print("""
    普通 Bootstrap 逐日独立抽样，会打散"连续暴跌"这种结构：
       原始序列: [d1 d2 d3 d4 d5 d6 d7 d8 d9 ...]
       普通:      d7 d2 d9 d2 d5 d1 ...            ← 时间结构被打散
       块(长度3): [d4 d5 d6][d1 d2 d3][d7 d8 d9]   ← 块内保留连续性
    """)

    hs300 = get_index_daily("000300", START, END)["close"].pct_change().dropna().to_numpy()
    series = {
        "沪深300 买入持有": hs300,
        "60 日均线择时": strat.to_numpy(),
        "GARCH 模拟(40年)": simulate_garch(40 * TRADING_DAYS),
    }
    horizon, n_paths, block = TRADING_DAYS, 2000, 20
    rows = []
    mdd_for_plot = {}
    for name, r in series.items():
        iid = iid_bootstrap_paths(r, horizon, n_paths, seed=1)
        blk = block_bootstrap_paths(r, horizon, n_paths, block, seed=1)
        mdd_iid, mdd_blk = max_drawdown(iid), max_drawdown(blk)
        acf_iid = np.mean([abs_acf1(iid[:, i]) for i in range(200)])
        acf_blk = np.mean([abs_acf1(blk[:, i]) for i in range(200)])
        rows.append([name, f"{abs_acf1(r):.3f}", f"{acf_iid:.3f}", f"{acf_blk:.3f}",
                     f"{q20(r):.1%}", f"{q20(iid):.1%}", f"{q20(blk):.1%}",
                     f"{np.percentile(mdd_iid, 5):.1%}", f"{np.percentile(mdd_blk, 5):.1%}"])
        mdd_for_plot[name] = (mdd_iid, mdd_blk)
    cols = ["序列", "|r|自相关:原始", "普通", "块", "20日收益1%分位:原始", "普通", "块",
            "一年MDD 5%最坏:普通", "块"]
    print(f"🎲 每个序列生成 {n_paths} 条一年期路径，块长度 {block} 天：")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("""
    💡 解读：
       - "|r|自相关"衡量波动聚集：普通 Bootstrap 把它打成接近 0，块 Bootstrap 保留了大部分
       - 有波动聚集时，坏日子会扎堆出现：对比"20 日收益 1% 分位"，看哪种 Bootstrap 更接近原始序列
       - 一年期 MDD 的尾部（5% 最坏）是资金规划该参考的数字；回测里的历史 MDD 只是分布中的一个点
       - 块长度是需要选择的参数：太短保留不了结构，太长则路径之间差异太小
    """)
    if is_offline():
        print("    ⚠️ 离线合成数据（GBM）没有波动聚集，两种 Bootstrap 的差别主要看'GARCH 模拟'这一行")

    # arch 库提供更完善的实现：StationaryBootstrap（块长度随机，平均为 block）
    bs = StationaryBootstrap(block, hs300, seed=np.random.default_rng(3))
    mdds = np.array([max_drawdown(data[0][0][:horizon]) for data in bs.bootstrap(500)])
    print(f"📦 arch.bootstrap.StationaryBootstrap（平均块长 {block}，500 次）：沪深300 "
          f"一年期 MDD 中位数 {np.median(mdds):.1%}，5% 最坏 {np.percentile(mdds, 5):.1%}")
    return mdd_for_plot


def plot_results(paths: np.ndarray, s0: float, mdd_for_plot: dict):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(paths[:, :100], lw=0.5, alpha=0.4)
    for p, ls in [(5, "--"), (50, "-"), (95, "--")]:
        ax.plot(np.percentile(paths, p, axis=1), color="k", ls=ls, lw=1.5)
    ax.axhline(s0, color="gray", lw=0.8)
    ax.set_title("GBM 模拟 100 条路径（黑线：5% / 50% / 95% 分位）")

    ax = axes[1]
    mdd_iid, mdd_blk = mdd_for_plot["沪深300 买入持有"]
    ax.hist(mdd_iid, bins=50, alpha=0.5, density=True, label="普通 Bootstrap")
    ax.hist(mdd_blk, bins=50, alpha=0.5, density=True, label="块 Bootstrap")
    ax.set_title("沪深300 一年期最大回撤分布")
    ax.legend()
    path = save_figure(fig, "04_monte_carlo.png", subdir="phase-3")
    print(f"\n🖼️  图已保存：{path}")


# ==================== 第五部分：模型选择对尾部的影响 ====================


def tail_model_comparison():
    print("\n" + "=" * 60)
    print("第五部分：换掉正态假设 —— t 分布扰动的蒙特卡洛")
    print("=" * 60)

    rng = np.random.default_rng(11)
    days, n_paths, sigma_d = TRADING_DAYS, 5000, 0.30 / np.sqrt(TRADING_DAYS)
    z_norm = rng.standard_normal((days, n_paths))
    df = 4
    z_t = stats.t.rvs(df, size=(days, n_paths), random_state=rng) / np.sqrt(df / (df - 2))  # 方差标准化为 1
    rows = []
    for name, z in [("正态扰动", z_norm), ("t(4) 扰动", z_t)]:
        r = sigma_d * z
        rows.append([name, f"{r.std() * np.sqrt(TRADING_DAYS):.1%}", f"{(r < -0.07).sum(axis=0).mean():.2f}",
                     f"{np.percentile(r.min(axis=0), 5):.1%}"])
    print(f"🎲 年化波动都是 30%，各模拟 {n_paths} 条一年期路径：")
    print(pd.DataFrame(rows, columns=["模型", "年化波动", "平均每年单日跌超 7% 的次数", "年内最差单日的 5% 分位"]).to_string(index=False))
    print("""
    💡 波动率相同，但尾部完全不同。用 GBM 做压力测试会低估极端日的频率。
       更现实的做法：t 分布扰动、GARCH 模拟（11 课）、或直接用 Bootstrap。
       Bootstrap 也只能"重组"历史，模拟不出历史上从未出现过的情景 —— 压力测试还需要人为设定极端情景。
    """)


# ==================== 主函数 ====================


def main():
    paths, s0 = gbm_demo()
    mc_convergence(s0)
    strat = bootstrap_demo()
    mdd_for_plot = block_bootstrap_demo(strat)
    plot_results(paths, s0, mdd_for_plot)
    tail_model_comparison()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第一部分的 μ 分别设为 0%、8%、20%，一年后亏损概率如何变化？哪个参数对结果影响更大：μ 还是 σ？
    2. 第四部分把块长度改为 5、20、60，MDD 分布如何变化？块长度如何选择？
    3. 实现"t 分布扰动的 GBM"，比较其与正态 GBM 一年期 1% 分位价格的差异
    4. 对你自己设计的一个策略，用块 Bootstrap 给出"未来一年最大回撤"的 95% 区间，并据此设定止损线
    """)


if __name__ == "__main__":
    main()
