"""
常见分布与收益率分布
====================

学习目标：
    1. 建立"分布 = 所有可能结果及其可能性的清单"的直觉
    2. 认识量化中常用的分布：正态、对数正态、Student-t、二项、泊松、均匀
    3. 理解"对数收益正态 → 价格对数正态"，以及价格分布为什么右偏
    4. 用正态分布和 t 分布拟合真实日收益，比较极端事件概率
    5. 用模拟实验看清：收益均值极难估计，波动率相对容易估计

核心概念：
    - 概率密度 f(x)：曲线下的面积才是概率
    - 68-95-99.7 法则：正态分布落在 μ±1σ / 2σ / 3σ 内的比例
    - 对数正态：ln(P_T/P_0) ~ N(Tμ, Tσ²)，价格永远 > 0
    - 标准误：均值估计的不确定性 ≈ σ/√n

前置知识：
    - Phase 2：简单收益与对数收益、年化

环境要求：
    - pip install numpy pandas scipy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据；合成数据是 GBM，没有肥尾）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from utils.data import get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

TRADING_DAYS = 252
SYMBOL, NAME = "600519", "贵州茅台"
START, END = "20180101", "20241231"


def load_log_returns() -> pd.Series:
    """茅台日对数收益（用等比后复权价 adj_close 计算）"""
    df = get_stock_daily_adj(SYMBOL, START, END)
    return np.log(df["adj_close"]).diff().dropna().rename("ret")


# ==================== 第一部分：分布的直觉与常见分布 ====================


def distribution_zoo():
    print("=" * 60)
    print("第一部分：分布的直觉与常见分布")
    print("=" * 60)

    print("""
    🪣 比喻：把茅台过去每天的收益率写在纸条上扔进一个桶里。
       "分布"就是对这个桶的描述 —— 大部分纸条写着 -1%~+1%，极少数写着 ±9%。
       量化建模的很多工作，就是在猜"明天会从什么样的桶里抽纸条"。

    ┌──────────────┬──────────────────┬──────────────────────────────────┐
    │ 分布         │ 形状             │ 量化中的用途                     │
    ├──────────────┼──────────────────┼──────────────────────────────────┤
    │ 正态         │ 对称钟形、尾部薄 │ 最常用的基准假设                 │
    │ 对数正态     │ 右偏、取值 > 0   │ 价格建模（价格不会跌成负数）     │
    │ Student-t    │ 钟形、尾部厚     │ 更贴近真实收益的肥尾、GARCH 残差 │
    │ 二项         │ 离散             │ n 笔交易中盈利的笔数             │
    │ 泊松         │ 离散             │ 单位时间内跳空、订单到达的次数   │
    │ 均匀         │ 平的             │ 随机数生成、随机策略基准         │
    └──────────────┴──────────────────┴──────────────────────────────────┘
    """)

    rng = np.random.default_rng(42)
    n = 100_000
    samples = {
        "正态 N(0,1)": rng.standard_normal(n),
        "t(df=3) 标准化": stats.t.rvs(3, size=n, random_state=rng) / np.sqrt(3 / (3 - 2)),
        "对数正态 (σ=0.5)": rng.lognormal(0, 0.5, n),
        "均匀 U(0,1)": rng.random(n),
    }
    rows = []
    for name, x in samples.items():
        rows.append([name, x.mean(), x.std(), stats.skew(x), stats.kurtosis(x), np.mean(np.abs(x - x.mean()) > 4 * x.std())])
    table = pd.DataFrame(rows, columns=["分布", "均值", "标准差", "偏度", "超额峰度", "超出 ±4σ 比例"])
    print(f"🎲 每个分布各抽 {n:,} 个样本：")
    print(table.round(4).to_string(index=False))
    print("""
    💡 观察：t 分布已经标准化为方差 1，和正态"一样宽"，但超出 ±4σ 的比例明显更高 —— 这就是肥尾。
       对数正态的偏度为正：右边有一条长尾巴。
       t(3) 的理论峰度是无穷大，所以样本峰度会非常大且很不稳定 —— 换个随机种子就会差很多。
    """)

    # 离散分布：二项与泊松
    p_win, n_trades = 0.55, 100
    wins = stats.binom(n_trades, p_win)
    print(f"📌 二项分布：胜率 {p_win:.0%} 的策略做 {n_trades} 笔交易")
    print(f"   盈利笔数期望 = {wins.mean():.0f}，标准差 = {wins.std():.1f}")
    print(f"   盈利笔数 ≤ 50（看起来像'没有优势'）的概率 = {wins.cdf(50):.1%}")
    print("   → 即使策略真有 55% 胜率，100 笔里也有不小的概率看起来平平无奇")

    lam = 2.0
    jumps = stats.poisson(lam)
    print(f"\n📌 泊松分布：假设某股平均每月出现 {lam:.0f} 次大幅跳空")
    for k in [0, 2, 5]:
        print(f"   某月恰好 {k} 次的概率 = {jumps.pmf(k):.1%}")


# ==================== 第二部分：正态分布与 68-95-99.7 法则 ====================


def normal_rule(ret: pd.Series):
    print("\n" + "=" * 60)
    print("第二部分：正态分布与 68-95-99.7 法则")
    print("=" * 60)

    print("""
    正态密度：f(x) = 1/(√(2π)σ) · exp(-(x-μ)² / (2σ²))
       μ 决定中心，σ 决定宽度；曲线下的面积 = 概率
    """)

    rng = np.random.default_rng(0)
    sim = rng.normal(ret.mean(), ret.std(), 100_000)
    mu, sd = ret.mean(), ret.std()
    rows = []
    for k in [1, 2, 3, 4]:
        theory = stats.norm.cdf(k) - stats.norm.cdf(-k)
        rows.append([
            f"μ ± {k}σ",
            f"{theory:.4%}",
            f"{np.mean(np.abs(sim - sim.mean()) <= k * sim.std()):.4%}",
            f"{np.mean(np.abs(ret - mu) <= k * sd):.4%}",
        ])
    print(pd.DataFrame(rows, columns=["区间", "正态理论", "正态模拟", f"{NAME}样本"]).to_string(index=False))

    beyond3 = int((np.abs(ret - mu) > 3 * sd).sum())
    expected3 = len(ret) * (1 - (stats.norm.cdf(3) - stats.norm.cdf(-3)))
    print(f"\n📊 {NAME} 共 {len(ret)} 个交易日：超出 ±3σ 的天数 = {beyond3}，正态分布预期约 {expected3:.1f} 天")
    print("""
    💡 真实收益通常是"中间更尖、两边更厚"：±1σ 内的比例高于 68%，±3σ 外的天数多于正态预期。
       （离线合成数据是正态的 GBM，两列会非常接近 —— 这正好是"正态世界"的样子）
    """)


# ==================== 第三部分：对数正态与价格分布 ====================


def lognormal_prices():
    print("\n" + "=" * 60)
    print("第三部分：对数收益正态 → 价格对数正态")
    print("=" * 60)

    print("""
    若每日对数收益 r_t ~ N(μ, σ²) 且相互独立，则 T 日后：
        ln(P_T / P_0) = Σ r_t ~ N(Tμ, Tσ²)  →  P_T 服从对数正态分布
    这是 GBM 模拟（04 课）和 Black-Scholes（Phase 12）的基础假设。
    """)

    rng = np.random.default_rng(7)
    p0, mu_a, sigma_a, years = 100.0, 0.08, 0.40, 5
    T = years * TRADING_DAYS
    mu_d = (mu_a - 0.5 * sigma_a**2) / TRADING_DAYS
    sigma_d = sigma_a / np.sqrt(TRADING_DAYS)
    n_paths = 20_000
    # 直接抽 T 日对数收益之和（与逐日累加等价，但更快）
    total_log = rng.normal(T * mu_d, np.sqrt(T) * sigma_d, n_paths)
    final = p0 * np.exp(total_log)

    theo_median = p0 * np.exp(T * mu_d)
    theo_mean = p0 * np.exp(T * mu_d + 0.5 * T * sigma_d**2)
    print(f"🎲 模拟 {n_paths:,} 条路径：初始价 {p0:.0f}，年化期望收益（算术）{mu_a:.0%}，年化波动 {sigma_a:.0%}，{years} 年")
    print(f"   {'':12s}{'模拟':>10s}{'理论':>10s}")
    print(f"   {'均值':12s}{final.mean():>10.1f}{theo_mean:>10.1f}")
    print(f"   {'中位数':12s}{np.median(final):>10.1f}{theo_median:>10.1f}")
    print(f"   最低价 = {final.min():.2f}（永远 > 0），偏度 = {stats.skew(final):.2f}")
    print(f"   最终价格低于初始价的比例 = {(final < p0).mean():.1%}")
    print("""
    💡 均值 > 中位数：少数路径涨得非常多，拉高了均值。
       中位数只取决于 μ - σ²/2：本例中 σ²/2 = 0.4²/2 = 8%，恰好抵消了 8% 的期望收益，
       所以虽然"期望"每年赚 8%，却有约一半路径 5 年后是亏的。
       "平均来看赚钱"不代表"大多数路径赚钱" —— 这就是 Phase 2 的波动拖累在分布上的样子。
    """)
    return final


# ==================== 第四部分：拟合真实收益，比较尾部概率 ====================


def fit_real_returns(ret: pd.Series):
    print("\n" + "=" * 60)
    print(f"第四部分：用正态与 t 分布拟合 {NAME} 日收益")
    print("=" * 60)

    mu, sigma = stats.norm.fit(ret)
    df_t, loc_t, scale_t = stats.t.fit(ret)
    print(f"📐 正态拟合：μ = {mu:.5f}，σ = {sigma:.5f}")
    print(f"📐 t 分布拟合：自由度 df = {df_t:.2f}，loc = {loc_t:.5f}，scale = {scale_t:.5f}")
    print("   df 越小尾部越厚；df → ∞ 时 t 分布退化为正态")

    rows = []
    for k in [3, 4, 5]:
        x = mu - k * sigma
        p_norm = stats.norm.cdf(x, mu, sigma)
        p_t = stats.t.cdf(x, df_t, loc_t, scale_t)
        p_emp = (ret < x).mean()
        rows.append([f"跌超 {k}σ ({x:.2%})", f"{p_norm:.2e}", f"{p_t:.2e}", f"{p_emp:.2e}",
                     f"{1 / p_norm / TRADING_DAYS:,.0f} 年" if p_norm > 0 else "∞"])
    print("\n📊 单日大跌的概率：")
    print(pd.DataFrame(rows, columns=["事件", "正态", "t 分布", "经验频率", "正态下平均多久一次"]).to_string(index=False))

    ll_norm = stats.norm.logpdf(ret, mu, sigma).sum()
    ll_t = stats.t.logpdf(ret, df_t, loc_t, scale_t).sum()
    aic_norm, aic_t = 2 * 2 - 2 * ll_norm, 2 * 3 - 2 * ll_t
    better = "t 分布" if aic_t < aic_norm else "正态分布"
    x5 = mu - 5 * sigma
    p5_norm, p5_emp = stats.norm.cdf(x5, mu, sigma), (ret < x5).mean()
    if p5_emp > 0:
        print(f"\n🔍 跌超 5σ：经验频率是正态预测的 {p5_emp / p5_norm:,.0f} 倍")
    else:
        print("\n🔍 跌超 5σ：样本中没有出现过")
    print(f"\n📏 AIC（越小越好）：正态 {aic_norm:.1f}，t 分布 {aic_t:.1f} → 本数据上 {better} 拟合更好")
    if is_synthetic(ret.index):
        print("   ⚠️ 离线合成数据是正态 GBM，没有肥尾：t 分布自由度会很大，两种分布差别不大")
    print("""
    💡 在真实数据上，正态分布给出的极端概率常常小到"千年一遇"，经验频率却可能高出几个数量级。
       用正态分布估计 VaR 会系统性低估极端亏损（02 课详细讨论）。
    """)
    return (mu, sigma), (df_t, loc_t, scale_t)


def plot_fit(ret: pd.Series, norm_params, t_params, final_prices: np.ndarray):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.linspace(ret.min(), ret.max(), 400)

    ax = axes[0]
    ax.hist(ret, bins=100, density=True, alpha=0.5, color="gray", label="真实")
    ax.plot(x, stats.norm.pdf(x, *norm_params), label="正态拟合")
    ax.plot(x, stats.t.pdf(x, *t_params), label="t 分布拟合")
    ax.set_title(f"{NAME} 日对数收益分布")
    ax.legend()

    ax = axes[1]
    hist, edges = np.histogram(ret, bins=60, density=True)
    centers = (edges[1:] + edges[:-1]) / 2
    ax.semilogy(centers[hist > 0], hist[hist > 0], "o", ms=3, color="gray", label="真实")
    ax.semilogy(x, stats.norm.pdf(x, *norm_params), label="正态")
    ax.semilogy(x, stats.t.pdf(x, *t_params), label="t 分布")
    ax.set_ylim(1e-3, None)
    ax.set_title("对数坐标：看清尾部")
    ax.legend()

    ax = axes[2]
    ax.hist(final_prices, bins=100, density=True, alpha=0.6)
    ax.axvline(np.median(final_prices), color="C1", label="中位数")
    ax.axvline(final_prices.mean(), color="C3", label="均值")
    ax.set_xlim(0, np.percentile(final_prices, 99))
    ax.set_title("5 年后价格：对数正态（右偏）")
    ax.legend()

    path = save_figure(fig, "01_distributions.png", subdir="phase-3")
    print(f"\n🖼️  图已保存：{path}")


# ==================== 第五部分：均值难估计，波动率易估计 ====================


def mean_vs_vol_estimation():
    print("\n" + "=" * 60)
    print("第五部分：均值难估计，波动率易估计")
    print("=" * 60)

    print("""
    🧪 实验：假设"上帝"知道真实参数 —— 年化收益 8%、年化波动 30%（典型个股）。
       我们分别用 1 / 5 / 10 / 20 年的日数据去估计，每种长度重复 2000 次，看估计值有多分散。
    """)
    rng = np.random.default_rng(2024)
    mu_a, sigma_a, n_sims = 0.08, 0.30, 2000
    rows = []
    for years in [1, 5, 10, 20]:
        n = years * TRADING_DAYS
        r = rng.normal(mu_a / TRADING_DAYS, sigma_a / np.sqrt(TRADING_DAYS), size=(n_sims, n))
        mu_hat = r.mean(axis=1) * TRADING_DAYS
        vol_hat = r.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        lo_m, hi_m = np.percentile(mu_hat, [2.5, 97.5])
        lo_v, hi_v = np.percentile(vol_hat, [2.5, 97.5])
        rows.append([f"{years} 年", f"[{lo_m:+.1%}, {hi_m:+.1%}]", f"{(mu_hat < 0).mean():.1%}",
                     f"[{lo_v:.1%}, {hi_v:.1%}]", f"{sigma_a / np.sqrt(years):.1%}"])
    print(pd.DataFrame(rows, columns=["样本长度", "年化收益 95% 区间", "估出负收益的比例",
                                      "年化波动 95% 区间", "理论标准误 σ/√年数"]).to_string(index=False))
    print("""
    💡 结论：
       - 收益均值的标准误 ≈ σ/√年数，只能靠"拉长年数"缩小，而且缩得很慢
       - 波动率的估计误差取决于"观测次数"，日数据一年就有 252 个，所以很快就准了
       - 因此：风险模型（波动率、相关性）通常比收益预测可靠得多 —— 03 课会用 t 检验定量说明
    """)


# ==================== 主函数 ====================


def main():
    distribution_zoo()
    ret = load_log_returns()
    print(f"\n📊 数据：{NAME} {ret.index[0].date()} ~ {ret.index[-1].date()}，{len(ret)} 个日收益")
    normal_rule(ret)
    final_prices = lognormal_prices()
    norm_params, t_params = fit_real_returns(ret)
    plot_fit(ret, norm_params, t_params, final_prices)
    mean_vs_vol_estimation()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把标的换成沪深 300 指数（get_index_daily("000300")），t 分布自由度比茅台大还是小？为什么？
    2. 把日收益换成周收益（resample("W")），重新拟合 t 分布，自由度如何变化？这说明了什么？
    3. 修改第三部分：波动率从 40% 改为 80%，均值与中位数的差距如何变化？低于初始价的路径比例呢？
    4. 第五部分中，要让年化收益 95% 区间宽度缩小到 ±2%，大约需要多少年数据？先用公式估算，再用模拟验证
    """)


if __name__ == "__main__":
    main()
