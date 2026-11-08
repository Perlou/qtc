"""
平稳性与 ADF 检验
=================

学习目标：
    1. 用"橡皮筋上的小球 vs 醉汉回家"理解平稳与非平稳
    2. 用模拟看见随机游走的方差随时间线性增长：Var(P_t) = t·σ²
    3. 掌握 ADF 检验（H0：存在单位根 / 非平稳）与 KPSS 检验（H0：平稳），并结合使用
    4. 对价格、收益、协整价差做平稳性检验
    5. 认识 ADF 的局限：检验功效低、结构突变会误导、收益的方差也并不恒定

核心概念：
    - 弱平稳：均值、方差恒定，自协方差只依赖间隔 k
    - I(1)：差分一次才平稳（如价格）；I(0)：本身平稳（如收益）
    - ADF：Δy_t = α + δt + γ·y_{t-1} + Σφ_i·Δy_{t-i} + ε_t，检验 γ = 0
    - p < 0.05 → 拒绝"非平稳"；不拒绝 ≠ 证明了非平稳

前置知识：
    - 06-linear-regression-ols.py（伪回归）、07-correlation-vs-cointegration.py

环境要求：
    - pip install numpy pandas statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据中股票对不协整，价差检验会给出"非平稳"）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from utils.data import get_close_panel, get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

START, END = "20180101", "20241231"


def adf_p(x, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> tuple[float, float]:
    res = adfuller(np.asarray(x), regression=regression, autolag=autolag, maxlag=maxlag, result_object=True)
    return res.statistic, res.pvalue


def kpss_p(x, regression: str = "c") -> tuple[float, float]:
    """KPSS 的 p 值来自查表，只覆盖 [0.01, 0.1]，超出范围时会被截断（这里静默该提示）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        res = kpss(np.asarray(x), regression=regression, nlags="auto", result_object=True)
    return res.statistic, res.pvalue


def verdict(p_adf: float, p_kpss: float) -> str:
    adf_stationary = p_adf < 0.05
    kpss_stationary = p_kpss >= 0.05
    if adf_stationary and kpss_stationary:
        return "✅ 平稳"
    if not adf_stationary and not kpss_stationary:
        return "❌ 非平稳"
    return "⚠️ 两者矛盾"


def simulate_ar1(phi: float, n: int, rng, sigma: float = 1.0) -> np.ndarray:
    eps = rng.normal(0, sigma, n + 200)
    y = np.zeros_like(eps)
    for t in range(1, len(eps)):
        y[t] = phi * y[t - 1] + eps[t]
    return y[200:]


# ==================== 第一部分：平稳与非平稳的直觉 ====================


def stationarity_intuition():
    print("=" * 60)
    print("第一部分：平稳与非平稳的直觉")
    print("=" * 60)

    print("""
    🎈 平稳序列：像被橡皮筋拴在桩子上的小球 —— 会晃，但总会回到桩子附近（收益率、协整价差）
    🍺 非平稳序列：像醉汉回家 —— 每一步都随机，走到哪儿算哪儿，没有"回归"的位置（价格、指数点位）

    弱平稳：E[y_t] = μ，Var(y_t) = σ²，Cov(y_t, y_{t-k}) = γ_k   —— 都不依赖时刻 t

    随机游走 P_t = P_{t-1} + ε_t：Var(P_t) = t·σ²，随时间线性增长 → 非平稳
    它的一阶差分 ΔP_t = ε_t 是平稳的 → 称为 I(1)（一阶单整）

    非平稳的价格 ─── 差分 / 取收益 ───▶ 近似平稳的收益率 ───▶ 可以建模
         └── 直接建模 → 伪回归、虚假的"规律"
    """)

    rng = np.random.default_rng(0)
    n_paths, n = 2000, 1000
    eps = rng.standard_normal((n, n_paths))
    rw = np.cumsum(eps, axis=0)
    ar = np.zeros_like(eps)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + eps[t]
    print(f"🧪 模拟 {n_paths} 条路径，比较不同时刻的'截面方差'（σ = 1）：")
    print(f"   {'时刻 t':>8s}{'随机游走':>12s}{'理论 t·σ²':>12s}{'AR(1) φ=0.9':>14s}{'理论 1/(1-φ²)':>16s}")
    for t in [10, 100, 500, 999]:
        print(f"   {t:>8d}{rw[t].var():>12.1f}{t + 1:>12d}{ar[t].var():>14.2f}{1 / (1 - 0.81):>16.2f}")
    print("""
    💡 随机游走的方差没有上限 —— "过去的均值"对未来毫无约束力；AR(1) 的方差很快稳定下来。
    """)


# ==================== 第二部分：ADF 与 KPSS ====================


def adf_kpss_on_simulated():
    print("\n" + "=" * 60)
    print("第二部分：ADF 与 KPSS 检验")
    print("=" * 60)

    print("""
    ADF 回归式：Δy_t = α + δ·t + γ·y_{t-1} + Σ φ_i·Δy_{t-i} + ε_t
       γ = 0：存在单位根（随机游走，非平稳）；γ < 0：有拉回力（平稳）
       Σ φ_i·Δy_{t-i}：滞后差分项，吸收自相关（"Augmented"的含义）
       regression="c" 含常数项；"ct" 再加时间趋势项

    ┌──────────┬──────────────────────┬────────────────────────────┐
    │ 检验     │ 原假设 H0            │ p < 0.05 意味着            │
    ├──────────┼──────────────────────┼────────────────────────────┤
    │ ADF      │ 存在单位根（非平稳） │ 拒绝非平稳 → 支持平稳      │
    │ KPSS     │ 平稳                 │ 拒绝平稳 → 支持非平稳      │
    └──────────┴──────────────────────┴────────────────────────────┘
    两者结合：都支持平稳 → 平稳；都支持非平稳 → 非平稳；矛盾 → 证据不足（趋势平稳？结构突变？）
    """)

    rng = np.random.default_rng(1)
    n = 1000
    t = np.arange(n)
    series = {
        "白噪声": rng.standard_normal(n),
        "AR(1) φ=0.9": simulate_ar1(0.9, n, rng),
        "随机游走": np.cumsum(rng.standard_normal(n)),
        "带漂移随机游走": np.cumsum(0.1 + rng.standard_normal(n)),
        "趋势平稳(线性趋势+噪声)": 0.05 * t + rng.standard_normal(n) * 3,
    }
    rows = []
    for name, x in series.items():
        s_c, p_c = adf_p(x, "c")
        _, p_ct = adf_p(x, "ct")
        _, k_c = kpss_p(x, "c")
        _, k_ct = kpss_p(x, "ct")
        rows.append([name, f"{s_c:.2f}", f"{p_c:.4f}", f"{k_c:.3f}", verdict(p_c, k_c),
                     f"{p_ct:.4f}", f"{k_ct:.3f}", verdict(p_ct, k_ct)])
    print(pd.DataFrame(rows, columns=["序列", "ADF 统计量(c)", "ADF p(c)", "KPSS p(c)", "结论(c)",
                                      "ADF p(ct)", "KPSS p(ct)", "结论(ct)"]).to_string(index=False))
    print("""
    💡 观察：
       - KPSS 的 p 值来自查表，被截断在 [0.01, 0.1]：显示 0.100 表示"≥ 0.1"，0.010 表示"≤ 0.01"
       - "c" 检验的是"围绕常数平稳"，"ct" 检验的是"围绕一条直线趋势平稳"
       - 趋势平稳序列：只含常数项的检验会把它判为非平稳；加上趋势项（ct）后结论可能改变 —— 回归形式要和数据匹配
    """)


# ==================== 第三部分：真实数据的平稳性 ====================


def real_data_tests():
    print("\n" + "=" * 60)
    print("第三部分：价格、收益、价差的平稳性")
    print("=" * 60)

    maotai = get_stock_daily_adj("600519", START, END)["adj_close"]
    hs300 = get_index_daily("000300", START, END)["close"]
    banks = np.log(get_close_panel(["601398", "601939"], START, END).dropna())
    beta = sm.OLS(banks["601398"], sm.add_constant(banks["601939"])).fit().params.iloc[1]
    spread = banks["601398"] - beta * banks["601939"]

    series = {
        "茅台 对数价格": np.log(maotai),
        "茅台 日对数收益": np.log(maotai).diff().dropna(),
        "沪深300 点位": hs300,
        "沪深300 日收益": hs300.pct_change().dropna(),
        "工行-建行 价差(全样本β)": spread,
    }
    rows = []
    for name, x in series.items():
        s, p = adf_p(x)
        _, k = kpss_p(x)
        rows.append([name, len(x), f"{s:.2f}", f"{p:.4f}", f"{k:.3f}", verdict(p, k)])
    print(pd.DataFrame(rows, columns=["序列", "样本数", "ADF 统计量", "ADF p", "KPSS p", "结论"]).to_string(index=False))
    print("""
    💡 典型结果：价格非平稳、收益平稳 —— 所以要用收益建模；价格水平不能直接作为机器学习特征。
       价差的结论取决于这对股票在样本期内是否协整（07 课）；注意这里的 β 用了全样本数据（前视偏差）。
       每个检验都有约 5% 的第一类错误：偶尔出现一次"两者矛盾"，不一定说明序列有问题。
    """)
    if is_synthetic(banks.index):
        print("    ⚠️ 离线合成数据：两只银行股是独立的 GBM，价差不会平稳")
    return np.log(maotai).diff().dropna()


# ==================== 第四部分：ADF 的检验功效 ====================


def adf_power():
    print("\n" + "=" * 60)
    print("第四部分：ADF 的检验功效 —— 缓慢回归的序列很难被识别")
    print("=" * 60)

    rng = np.random.default_rng(3)
    n_sims = 200
    rows = []
    for phi in [0.90, 0.95, 0.98, 0.99, 1.00]:
        row = [phi, f"{-np.log(2) / np.log(phi):.0f} 天" if phi < 1 else "∞"]
        for n in [250, 1000]:
            rej = np.mean([adf_p(simulate_ar1(phi, n, rng), autolag=None, maxlag=1)[1] < 0.05 for _ in range(n_sims)])
            row.append(f"{rej:.0%}")
        rows.append(row)
    print(f"🧪 AR(1) 序列，每格模拟 {n_sims} 次，ADF 在 5% 水平下拒绝'非平稳'的比例：")
    print(pd.DataFrame(rows, columns=["φ", "半衰期", "n = 250（约 1 年）", "n = 1000（约 4 年）"]).to_string(index=False))
    print("""
    💡 φ = 1.00 那一行是真的随机游走，拒绝率应接近 5%（第一类错误）。
       φ 接近 1 时（半衰期几十天的缓慢均值回归），1 年数据往往无法拒绝单位根。
       → "ADF 不拒绝"不等于"证明了非平稳"，只是证据不足。结合 KPSS、可视化和经济逻辑判断。
    """)


# ==================== 第五部分：结构突变与方差变化 ====================


def breaks_and_variance(log_ret: pd.Series):
    print("\n" + "=" * 60)
    print("第五部分：结构突变与'近似平稳'")
    print("=" * 60)

    rng = np.random.default_rng(5)
    n = 1000
    rows = []
    for jump in [0, 3, 6]:
        x = simulate_ar1(0.5, n, rng)
        x[n // 2:] += jump  # 中途均值突然抬升（例如制度变化、主营业务调整）
        _, p = adf_p(x)
        _, k = kpss_p(x)
        rows.append([f"均值跳升 {jump}σ", f"{p:.4f}", f"{k:.3f}", verdict(p, k)])
    print("🧪 一个本来平稳的 AR(1)（φ=0.5），在中点加入一次均值跳变：")
    print(pd.DataFrame(rows, columns=["情形", "ADF p", "KPSS p", "结论"]).to_string(index=False))
    print("""
    💡 每一段内部都是平稳的，但整体看起来像"没有固定的桩子"，检验可能给出误导性的结论。
       A 股的制度变化（涨跌幅调整、注册制）可能让序列在某个时点"换了一个世界"。
    """)

    vol = log_ret.rolling(60).std() * np.sqrt(252)
    yearly = vol.groupby(vol.index.year).mean()
    print("📊 茅台日收益通过了平稳性检验，但它的 60 日年化波动率（按年平均）：")
    print("   " + "  ".join(f"{y}: {v:.0%}" for y, v in yearly.dropna().items()))
    print("""
    💡 收益的"平稳"只是近似：均值大致稳定，但方差明显随时间变化（波动率聚集，11 课）。
       差分去掉了"记忆"：收益平稳但几乎丢失了价格水平的信息。
       Phase 9 会遇到分数阶差分（Fractional Differentiation），在平稳性和记忆之间折中。
    """)
    if is_synthetic(log_ret.index):
        print("    ⚠️ 离线合成数据是常波动 GBM，各年波动率差别只来自抽样误差")
    return vol


def plot_series(vol: pd.Series):
    setup_chinese_font()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for phi, label in [(0.0, "白噪声"), (0.9, "AR(1) φ=0.9"), (1.0, "随机游走")]:
        x = np.cumsum(rng.standard_normal(1000)) if phi == 1 else simulate_ar1(phi, 1000, rng)
        axes[0].plot(x, lw=0.8, label=label)
    axes[0].set_title("平稳 vs 非平稳")
    axes[0].legend()
    paths = np.cumsum(rng.standard_normal((500, 200)), axis=0)
    axes[1].plot(paths, lw=0.3, alpha=0.4)
    axes[1].plot(2 * np.sqrt(np.arange(1, 501)), "k--")
    axes[1].plot(-2 * np.sqrt(np.arange(1, 501)), "k--")
    axes[1].set_title("200 条随机游走：散开范围 ∝ √t")
    vol.plot(ax=axes[2])
    axes[2].set_title("茅台 60 日年化波动率")
    path = save_figure(fig, "09_stationarity.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    stationarity_intuition()
    adf_kpss_on_simulated()
    log_ret = real_data_tests()
    adf_power()
    vol = breaks_and_variance(log_ret)
    plot_series(vol)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 对 10 只股票的对数价格和日收益批量做 ADF + KPSS，统计各种结论的数量
    2. 把第三部分的价差换成 07 课中扩展窗口 β 的价差，ADF 结论是否改变？
    3. 在第四部分加入 n = 2500（约 10 年），φ = 0.99 的拒绝率是多少？这对配对交易的样本长度有什么启示？
    4. 对沪深300 的 20 日滚动波动率序列做 ADF 检验，它是平稳的吗？这和"波动率可预测"有什么关系？
    """)


if __name__ == "__main__":
    main()
