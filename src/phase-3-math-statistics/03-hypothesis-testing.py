"""
假设检验：策略收益是否显著
==========================

学习目标：
    1. 用"法庭审判"理解原假设、p 值、显著性水平，以及 p 值"不是什么"
    2. 用模拟看见第一类错误（假阳性）和第二类错误（假阴性 / 检验功效）
    3. 用单样本 t 检验和 Newey-West（HAC）标准误检验一个真实择时策略
    4. 掌握 t = SR × √年数，理解"为什么收益很难被证明"
    5. 体验多重检验：试 100 个随机策略，总有几个"显著"

核心概念：
    - H0（原假设）：策略日均收益 = 0（没有真本事）
    - p 值：假设 H0 为真，仅凭运气得到"至少这么极端"结果的概率
    - t = R̄ / (s/√n)，s/√n 为均值的标准误
    - Newey-West：考虑自相关与异方差的稳健标准误

前置知识：
    - 01-probability-distributions.py、02-fat-tails-normality.py

环境要求：
    - pip install numpy pandas scipy statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from scipy import stats
from scipy.signal import lfilter

from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

TRADING_DAYS = 252
START, END = "20100101", "20251231"


def one_sided_p(t_stat: float, p_two: float) -> float:
    """双侧 p 值 → 单侧 p 值（H1：均值 > 0）"""
    return p_two / 2 if t_stat > 0 else 1 - p_two / 2


def newey_west_t(r: np.ndarray, maxlags: int = 5) -> float:
    """对常数项做 OLS 回归 = 检验均值；用 HAC 标准误得到 Newey-West t 值"""
    res = sm.OLS(r, np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0])


# ==================== 第一部分：p 值的直觉 ====================


def p_value_intuition():
    print("=" * 60)
    print("第一部分：p 值的直觉 —— 统计学里的'无罪推定'")
    print("=" * 60)

    print("""
    ⚖️ 法庭比喻：
       H0（原假设）= "被告无罪" = 策略没有真实收益，日均收益为 0
       证据         = 回测数据
       只有证据"强到在无罪情况下很难出现"时，才判"有罪"（拒绝 H0，认为策略有效）

    p 值 = 在"策略其实没本事"的前提下，仅凭运气得到"至少这么好"结果的概率

    ⚠️ p 值不是"策略无效的概率"，也不是"策略有效的概率"
    """)

    # 模拟"运气的分布"：一个真实均值为 0 的策略，3 年回测的 t 统计量分布
    rng = np.random.default_rng(0)
    n_days, n_sims = 3 * TRADING_DAYS, 20_000
    r = rng.normal(0, 0.01, size=(n_sims, n_days))
    t_null = r.mean(axis=1) / (r.std(axis=1, ddof=1) / np.sqrt(n_days))

    observed_t = 2.1
    p_sim = (t_null >= observed_t).mean()
    p_theory = stats.t.sf(observed_t, df=n_days - 1)
    print(f"🧪 模拟 {n_sims:,} 个'毫无本事'的策略，每个回测 3 年，计算 t 统计量：")
    print(f"   t 统计量的均值 = {t_null.mean():+.3f}，标准差 = {t_null.std():.3f}（理论上接近 N(0,1)）")
    print(f"   假如你的策略 t = {observed_t}：")
    print(f"   无本事策略中 t ≥ {observed_t} 的比例 = {p_sim:.4f}（模拟的单侧 p 值）")
    print(f"   t 分布理论单侧 p 值       = {p_theory:.4f}")
    print("""
    💡 p 值就是这张"运气分布"图里，比你的结果更极端的那块面积。
    """)
    return t_null, observed_t


def plot_null(t_null: np.ndarray, observed_t: float):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    dens, edges = np.histogram(t_null, bins=100, density=True)
    colors = np.where(edges[:-1] >= observed_t, "C3", "C0")
    ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color=colors, alpha=0.7,
           label="H0 下的 t 统计量（模拟）；红色 = p 值对应的面积")
    x = np.linspace(-4, 4, 300)
    ax.plot(x, stats.norm.pdf(x), color="k", lw=1, label="N(0,1)")
    ax.axvline(observed_t, color="C3", ls="--")
    ax.set_title("p 值 = 原假设成立时，出现至少这么极端结果的概率")
    ax.legend(loc="upper left", fontsize=8)
    path = save_figure(fig, "03_p_value.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 第二部分：两类错误与检验功效 ====================


def two_error_types():
    print("\n" + "=" * 60)
    print("第二部分：两类错误与检验功效")
    print("=" * 60)

    print("""
    ┌──────────────────────┬────────────────────────┬────────────────────────┐
    │                      │ 真实：策略无效         │ 真实：策略有效         │
    ├──────────────────────┼────────────────────────┼────────────────────────┤
    │ 判断：有效（拒绝H0） │ ❌ 第一类错误（假阳性）│ ✅ 检验功效 1-β        │
    │ 判断：无效           │ ✅                     │ ❌ 第二类错误（假阴性）│
    └──────────────────────┴────────────────────────┴────────────────────────┘
    量化中第一类错误代价更大：把运气当本事上实盘，会真金白银地亏钱。
    """)

    rng = np.random.default_rng(1)
    n_sims, vol_a, alpha = 2000, 0.15, 0.05
    rows = []
    for years in [1, 3, 5, 10]:
        n = years * TRADING_DAYS
        row = [f"{years} 年"]
        for sr in [0.0, 0.5, 1.0, 2.0]:
            mu_d = sr * vol_a / TRADING_DAYS
            r = rng.normal(mu_d, vol_a / np.sqrt(TRADING_DAYS), size=(n_sims, n))
            t = r.mean(axis=1) / (r.std(axis=1, ddof=1) / np.sqrt(n))
            p = stats.t.sf(t, df=n - 1)  # 单侧
            row.append(f"{(p < alpha).mean():.1%}")
        rows.append(row)
    print(f"🧪 每格模拟 {n_sims} 个策略，单侧 t 检验（α = {alpha:.0%}）判为'显著'的比例：")
    print(pd.DataFrame(rows, columns=["回测长度", "真实SR=0（假阳性率）", "真实SR=0.5", "真实SR=1.0", "真实SR=2.0"]).to_string(index=False))
    print("""
    💡 解读：
       - 第一列：策略真没本事时，约 5% 会被误判为有效 —— 这就是 α 的含义，与样本长度无关
       - 其余列是"检验功效"：真有本事的策略被识别出来的概率
       - 年化 Sharpe 0.5 的好策略，3 年回测大多数情况下都"不显著" —— 不显著 ≠ 无效
    """)


# ==================== 第三部分：检验一个真实择时策略 ====================


def test_real_strategy():
    print("\n" + "=" * 60)
    print("第三部分：检验一个真实策略 —— 沪深 300 均线择时")
    print("=" * 60)

    close = get_index_daily("000300", START, END)["close"]
    ret = close.pct_change()
    pos = (close > close.rolling(60).mean()).astype(float).shift(1)
    cost = pos.diff().abs() * 0.001  # 每次换仓 0.1% 成本
    strat = (pos * ret - cost).dropna()
    excess = (strat - ret.loc[strat.index]).dropna()  # 相对买入持有的超额

    rows = []
    for name, r in [("策略收益", strat), ("相对买入持有的超额", excess)]:
        x = r.to_numpy()
        t_stat, p_two = stats.ttest_1samp(x, popmean=0)
        sr_a = x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS)
        years = len(x) / TRADING_DAYS
        rows.append([name, f"{x.mean() * TRADING_DAYS:+.2%}", f"{sr_a:.2f}", f"{years:.1f}",
                     f"{t_stat:.2f}", f"{sr_a * np.sqrt(years):.2f}", f"{one_sided_p(t_stat, p_two):.4f}",
                     f"{newey_west_t(x):.2f}"])
    print(f"📊 {close.index[0].date()} ~ {close.index[-1].date()}，60 日均线择时，单边成本 0.1%")
    print(pd.DataFrame(rows, columns=["序列", "年化均值", "年化SR", "年数", "t 统计量", "SR×√年数",
                                      "单侧 p", "Newey-West t"]).to_string(index=False))
    print("""
    💡 观察：
       - "t 统计量"与"SR×√年数"几乎相等（差异来自 ddof 和年化口径）—— 这就是 t = SR × √Y
       - Newey-West t 考虑了收益的自相关和异方差；两者差异大时，要以稳健的结果为准
       - 判断时请看实际的 p 值；即使 p < 0.05，还要问：这个均线窗口是我试了多少次才选出来的？
    """)
    if is_synthetic(close.index):
        print("    ⚠️ 离线合成数据（GBM）：以上数字只用于跑通流程，不代表真实市场")


# ==================== 第四部分：t = SR × √年数 ====================


def years_needed():
    print("\n" + "=" * 60)
    print("第四部分：t = SR × √年数 —— 收益为什么难被证明")
    print("=" * 60)

    print("""
    t = R̄ / (s/√n) = SR_日 × √n = SR_年 × √(n/N) = SR_年 × √Y      （N=每年交易日，Y=年数）
    要让 t > 2（约 5% 显著性），需要 Y ≈ (2 / SR_年)²
    """)
    rows = []
    for sr in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        rows.append([sr, f"{(2 / sr) ** 2:.1f} 年", f"{(3 / sr) ** 2:.1f} 年"])
    print(pd.DataFrame(rows, columns=["年化 Sharpe", "t>2 所需年数", "t>3 所需年数"]).to_string(index=False))
    print("""
    💡 一个年化 Sharpe 0.5 的策略（已经不算差）需要约 16 年数据才能与运气区分。
       而 01 课说过，波动率用 1 年日数据就能估得比较准 —— 收益难证明，风险易估计。
    """)


# ==================== 第五部分：自相关让普通 t 检验失效 ====================


def autocorrelation_trap():
    print("\n" + "=" * 60)
    print("第五部分：收益自相关 → 普通 t 检验高估显著性")
    print("=" * 60)

    print("""
    t 检验假设各期收益独立。持仓多日的策略、用重叠窗口算的"未来 20 日收益"，都有正自相关，
    有效样本量远小于行数，普通 t 检验会高估显著性。
    🧪 实验：生成均值为 0 的 AR(1) 收益 r_t = φ·r_{t-1} + ε_t，看两种 t 检验的假阳性率。
    """)
    rng = np.random.default_rng(5)
    n, n_sims = 3 * TRADING_DAYS, 600
    rows = []
    for phi in [0.0, 0.2, 0.5]:
        hits_ols = hits_nw = 0
        for _ in range(n_sims):
            eps = rng.normal(0, 0.01, n + 100)
            r = lfilter([1.0], [1.0, -phi], eps)[100:]  # 等价于 r_t = φ·r_{t-1} + ε_t，丢弃前 100 个预热值
            t_ols = r.mean() / (r.std(ddof=1) / np.sqrt(n))
            t_nw = newey_west_t(r, maxlags=10)
            hits_ols += abs(t_ols) > 1.96
            hits_nw += abs(t_nw) > 1.96
        rows.append([phi, f"{hits_ols / n_sims:.1%}", f"{hits_nw / n_sims:.1%}"])
    print(pd.DataFrame(rows, columns=["自相关 φ", "普通 t 检验假阳性率", "Newey-West 假阳性率"]).to_string(index=False))
    print("""
    💡 名义上双侧 5% 的检验，在自相关存在时普通 t 检验的实际假阳性率会明显上升；
       Newey-West 修正后会更接近 5%（有限样本下未必完全达到）。时间序列检验默认用 HAC。
    """)


# ==================== 第六部分：多重检验 ====================


def multiple_testing():
    print("\n" + "=" * 60)
    print("第六部分：多重检验 —— 试 100 个策略，总有'显著'的")
    print("=" * 60)

    close = get_index_daily("000300", START, END)["close"]
    ret = close.pct_change().dropna()
    rng = np.random.default_rng(2024)
    n_strats = 100
    results = []
    for _ in range(n_strats):
        # 完全随机的多空信号：每 20 天随机抛一次硬币决定做多(+1)或做空(-1)
        coin = rng.choice([-1.0, 1.0], size=len(ret) // 20 + 1)
        pos = np.repeat(coin, 20)[: len(ret)]
        r = pos * ret.to_numpy()
        t_stat, p_two = stats.ttest_1samp(r, 0)
        results.append((t_stat, one_sided_p(t_stat, p_two)))
    t_arr = np.array([x[0] for x in results])
    p_arr = np.array([x[1] for x in results])

    print(f"🎲 在沪深 300 数据上测试 {n_strats} 个'抛硬币'策略（信号与行情毫无关系）：")
    print(f"   单侧 p < 0.05 的策略数：{(p_arr < 0.05).sum()} 个（运气期望约 {n_strats * 0.05:.0f} 个）")
    print(f"   最好的那个：t = {t_arr.max():.2f}，p = {p_arr.min():.4f}")
    print(f"   Bonferroni 修正后的门槛 p < 0.05/{n_strats} = {0.05 / n_strats:.4f}，"
          f"通过的策略数：{(p_arr < 0.05 / n_strats).sum()} 个")
    print("""
    💡 如果你只汇报"最好的那个"，它看起来很显著，但它只是 100 次抛硬币里最幸运的一次。
       - Bonferroni：门槛除以尝试次数（简单但保守）
       - 学术界有观点主张新因子 t 值门槛提高到 3 左右（Harvey, Liu & Zhu, 2016）
       - Phase 4 会系统介绍 BH-FDR、Deflated Sharpe Ratio
       - 务必记录所有尝试过的参数和版本 —— 尝试次数本身就是证据的一部分

    📋 其他陷阱：统计显著 ≠ 经济显著（扣成本后可能为负）；p-hacking（反复改样本区间直到 p < 0.05）
    """)


# ==================== 主函数 ====================


def main():
    t_null, observed_t = p_value_intuition()
    plot_null(t_null, observed_t)
    two_error_types()
    test_real_strategy()
    years_needed()
    autocorrelation_trap()
    multiple_testing()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第三部分把均线窗口换成 20、120、250，分别算 t 值。如果只汇报最好的那个，应如何修正 p 值？
    2. 第二部分中，真实 SR = 1 的策略，要让检验功效达到 80% 大约需要几年？用模拟找出答案
    3. 用"未来 20 日收益"（重叠样本）对一个随机信号做 t 检验，比较普通 t 与 Newey-West(maxlags=20)
    4. 判断对错并说明理由："p = 0.03，所以这个策略有 97% 的概率是有效的"
    """)


if __name__ == "__main__":
    main()
