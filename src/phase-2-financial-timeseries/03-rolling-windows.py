"""
滚动窗口统计
============

学习目标：
    1. 掌握三种窗口：rolling（滚动）、expanding（扩展）、ewm（指数加权）
    2. 计算滚动波动率、EWMA 波动率、滚动 Z-score、滚动相关系数
    3. 理解窗口长度的权衡：短窗口反应快但噪声大，长窗口平稳但滞后
    4. 识别滚动计算中的前视偏差：center=True、全样本标准化
    5. 理解 min_periods、ddof、预热期（warm-up）等细节

核心概念：
    - 滚动窗口：每个时点只看最近 w 个观测，等权
    - 扩展窗口：从起点到当前的全部观测
    - 指数加权（EWMA）：EMA_t = α·x_t + (1-α)·EMA_{t-1}，越近权重越大
    - 半衰期：h 期之前观测的权重只剩当前观测的一半

前置知识：
    - 01-returns-simple-vs-log.py、02-annualization-volatility.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_index_daily, get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
TRADING_DAYS = 252
START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"


# ==================== 第一部分：三种窗口 ====================


def three_windows():
    print("=" * 60)
    print("第一部分：三种窗口")
    print("=" * 60)

    print("""
    时间 ──────────────────────────────────────────────────▶
            [──── w=5 ────]                 t 时刻的统计量
               [──── w=5 ────]              t+1 时刻
                  [──── w=5 ────]           t+2 时刻

    ┌────────────┬─────────────────────────┬────────────────────────────────┐
    │ 类型       │ pandas                  │ 特点 / 适用                    │
    ├────────────┼─────────────────────────┼────────────────────────────────┤
    │ 滚动窗口   │ rolling(w)              │ 最近 w 个点等权；移动平均、波动│
    │ 扩展窗口   │ expanding()             │ 起点到当前；"只用历史"的标准化 │
    │ 指数加权   │ ewm(span=/halflife=)    │ 越近权重越大；EMA、EWMA 波动率 │
    └────────────┴─────────────────────────┴────────────────────────────────┘
    """)

    x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8], index=pd.date_range("2024-01-01", periods=8))
    demo = pd.DataFrame({
        "x": x,
        "rolling(3).mean": x.rolling(3).mean(),
        "rolling(3,min_periods=1)": x.rolling(3, min_periods=1).mean(),
        "expanding().mean": x.expanding().mean(),
        "ewm(span=3).mean": x.ewm(span=3, adjust=False).mean(),
    })
    print("🧮 用 1~8 的序列直观感受（注意前两个 NaN 就是\"预热期\"）：")
    print(demo.to_string(float_format=lambda v: f"{v:.3f}"))

    # 指数加权的权重：span 与 halflife 的换算
    print("\n⚖️ 指数加权的权重分布（最近一个观测记为 lag 0）：")
    rows = []
    for label, alpha in [
        ("span=10", 2 / (10 + 1)),
        ("span=20", 2 / (20 + 1)),
        ("halflife=20", 1 - np.exp(-np.log(2) / 20)),
    ]:
        lags = np.array([0, 1, 5, 10, 20, 40])
        w = alpha * (1 - alpha) ** lags
        rows.append([label, alpha] + list(w / w[0]))
    cols = ["参数", "α"] + [f"lag{lag}" for lag in [0, 1, 5, 10, 20, 40]]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("""
    💡 span=s → α = 2/(s+1)；halflife=h → α = 1 - exp(-ln2/h)
       halflife=20 那一行，lag20 的相对权重正好是 0.5 —— 这就是"半衰期"
    """)


# ==================== 第二部分：滚动波动率 ====================


def rolling_volatility() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"第二部分：滚动波动率（{SYMBOL} {NAME}）")
    print("=" * 60)

    close = get_stock_daily_adj(SYMBOL, START, END)["adj_close"]
    ret = close.pct_change()
    ann = np.sqrt(TRADING_DAYS)

    vols = pd.DataFrame({
        "vol_20": ret.rolling(20).std() * ann,
        "vol_60": ret.rolling(60).std() * ann,
        "vol_250": ret.rolling(250).std() * ann,
        "ewm_hl20": ret.ewm(halflife=20, min_periods=20).std() * ann,
    })
    print("\n📊 各种波动率估计的统计特征：")
    summary = pd.DataFrame({
        "首个有效日期": vols.apply(lambda s: s.first_valid_index().date()),
        "NaN 个数(预热)": vols.isna().sum(),
        "均值": vols.mean(),
        "最小": vols.min(),
        "最大": vols.max(),
        "日变化标准差": vols.diff().std(),
    })
    print(summary.to_string(formatters={
        "均值": "{:.1%}".format, "最小": "{:.1%}".format, "最大": "{:.1%}".format,
        "日变化标准差": "{:.3%}".format,
    }))
    full = ret.std() * ann
    peak_day = vols["vol_20"].idxmax()
    lo, hi = vols["vol_20"].min(), vols["vol_20"].max()

    # 对照实验：假设真实波动恒定，只有 20 个样本的估计值自己会跳动多大？
    constant = pd.Series(np.random.default_rng(0).normal(0, ret.std(), len(ret)))
    control = constant.rolling(20).std() * ann
    print(f"""
    📌 全样本年化波动只有一个数：{full:.1%}
       20 日滚动波动在 {lo:.1%} ~ {hi:.1%} 之间变化，峰值出现在 {peak_day.date()}

    🧪 这个变化全是"市场状态"吗？做个对照：
       假设波动恒定为 {full:.1%}，20 日估计也会在 {control.min():.1%} ~ {control.max():.1%} 之间跳动 —— 这部分纯属估计噪声
    """)
    if hi > control.max() * 1.2:
        print("    → 实际峰值明显超出噪声范围：波动率确实随市场状态变化，不只是估计误差")
    else:
        print("    → 实际范围与噪声相当：这份数据看不出状态变化（合成数据的波动本来就是恒定的）")
    print("""
    💡 "日变化标准差"衡量估计值自身的抖动：窗口越短，估计越不稳定
    """)
    vols["close"] = close
    vols["ret"] = ret
    return vols


# ==================== 第三部分：窗口长度的权衡 ====================


def window_tradeoff():
    print("\n" + "=" * 60)
    print("第三部分：窗口长度的权衡（模拟实验）")
    print("=" * 60)

    # 构造"状态切换"：前 500 天年化波动 15%，之后突然变成 45%
    rng = np.random.default_rng(7)
    n, switch = 1000, 500
    true_vol = np.where(np.arange(n) < switch, 0.15, 0.45)
    ret = pd.Series(rng.normal(0, true_vol / np.sqrt(TRADING_DAYS)), index=pd.bdate_range("2020-01-01", periods=n))
    threshold = 0.30  # 波动估计何时"察觉到"状态切换（越过中点）

    rows = []
    for label, est in [
        ("rolling(10)", ret.rolling(10).std()),
        ("rolling(20)", ret.rolling(20).std()),
        ("rolling(60)", ret.rolling(60).std()),
        ("rolling(250)", ret.rolling(250).std()),
        ("ewm(halflife=10)", ret.ewm(halflife=10).std()),
        ("ewm(halflife=30)", ret.ewm(halflife=30).std()),
    ]:
        est = est * np.sqrt(TRADING_DAYS)
        after = est.iloc[switch:]
        crossed = after[after > threshold]
        lag = (ret.index.get_loc(crossed.index[0]) - switch) if len(crossed) else np.nan
        # 噪声：平稳期（第 300~500 天）估计值相对真实值 15% 的误差
        calm = est.iloc[300:switch]
        rows.append([label, lag, (calm - 0.15).abs().mean(), calm.std()])
    df = pd.DataFrame(rows, columns=["估计方法", "察觉切换滞后(天)", "平稳期平均误差", "平稳期估计抖动"])
    print("🧪 真实年化波动在第 500 天从 15% 跳到 45%：")
    print(df.to_string(index=False, formatters={
        "察觉切换滞后(天)": "{:.0f}".format, "平稳期平均误差": "{:.2%}".format, "平稳期估计抖动": "{:.2%}".format,
    }))
    print("""
    💡 没有"正确"的窗口长度：
       ┌──────────────┬──────────────────────┬──────────────────────┐
       │ 窗口         │ 优点                 │ 缺点                 │
       ├──────────────┼──────────────────────┼──────────────────────┤
       │ 短（5–20）   │ 反应快               │ 噪声大、估计误差高   │
       │ 长（120–250）│ 平稳、误差小         │ 滞后严重             │
       └──────────────┴──────────────────────┴──────────────────────┘
       EWMA 是一种折中：既用到较长历史，又让近期数据权重更大。
       ⚠️ 把窗口长度从 5 试到 250 挑回测最好的那个 —— 这就是过拟合（Phase 4）
    """)


# ==================== 第四部分：Z-score 与滚动相关 ====================


def zscore_and_correlation(vols: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：滚动 Z-score 与滚动相关")
    print("=" * 60)

    close, ret = vols["close"], vols["ret"]

    # 滚动 Z-score：当前价格偏离 60 日均值多少个标准差
    z = (close - close.rolling(60).mean()) / close.rolling(60).std()
    print(f"""
    📏 60 日价格 Z-score = (价格 - 60 日均值) / 60 日标准差
       最新值：{z.iloc[-1]:+.2f}
       |z| > 2 的天数占比：{(z.dropna().abs() > 2).mean():.1%}（有效样本 {z.count()} 天，已排除预热期的 NaN）
       最低点：{z.min():+.2f}（{z.idxmin().date()}），最高点：{z.max():+.2f}（{z.idxmax().date()}）
    """)

    # 扩展窗口标准化：只用历史数据
    z_exp = (ret - ret.expanding(min_periods=60).mean()) / ret.expanding(min_periods=60).std()
    z_full = (ret - ret.mean()) / ret.std()
    print("    扩展窗口标准化（无前视） vs 全样本标准化（有前视）在前几个有效日期的差异：")
    cmp_df = pd.DataFrame({"z_expanding": z_exp, "z_full_sample": z_full}).dropna().head(5)
    print(cmp_df.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("    → 两者数值不同：全样本版本在 2018 年就\"知道\"了 2018–2024 的均值和标准差")

    # 滚动相关：个股与沪深 300
    bench = get_index_daily("000300", START, END)["close"].pct_change()
    pair = pd.concat({"stock": ret, "hs300": bench}, axis=1).dropna()
    roll_corr = pair["stock"].rolling(60).corr(pair["hs300"])
    print(f"""
    🔗 {NAME} 与沪深 300 的 60 日滚动相关系数：
       全样本相关：{pair['stock'].corr(pair['hs300']):.2f}
       滚动相关范围：{roll_corr.min():.2f} ~ {roll_corr.max():.2f}，中位数 {roll_corr.median():.2f}
       💡 滚动相关的变化一部分是真实时变，一部分是估计噪声
          （真实相关接近 0 时，60 日样本的标准误约 1/√60 ≈ {1 / np.sqrt(60):.2f}），解读时要把噪声扣掉
    """)
    return roll_corr


# ==================== 第五部分：前视陷阱与细节 ====================


def pitfalls(vols: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：前视陷阱与细节")
    print("=" * 60)

    close = vols["close"]

    # 陷阱 1：center=True —— 修改未来数据，看今天的值会不会变
    t = close.index[500]
    tampered = close.copy()
    tampered.iloc[505] = tampered.iloc[505] * 1.5  # 篡改 5 天后的价格
    normal_before = close.rolling(20).mean().loc[t]
    normal_after = tampered.rolling(20).mean().loc[t]
    center_before = close.rolling(20, center=True).mean().loc[t]
    center_after = tampered.rolling(20, center=True).mean().loc[t]
    print(f"""
    ❌ 陷阱 1：rolling(center=True) 用到未来数据
       实验：把 {close.index[505].date()} 的价格改大 50%，观察 {t.date()}（5 个交易日之前）的 MA20：
       普通 rolling(20)          篡改前 {normal_before:.2f} → 篡改后 {normal_after:.2f}  {'✅ 不受影响' if np.isclose(normal_before, normal_after) else '❌'}
       rolling(20, center=True)  篡改前 {center_before:.2f} → 篡改后 {center_after:.2f}  {'❌ 被未来数据改变' if not np.isclose(center_before, center_after) else ''}
       🔑 通用自检方法：改动未来数据，过去的指标值必须保持不变
    """)

    # 陷阱 2：ddof
    s20 = close.rolling(20).std()
    p20 = close.rolling(20).std(ddof=0)
    print(f"""    ⚠️ 细节 2：rolling().std() 默认 ddof=1（样本标准差）
       很多行情软件的布林带用 ddof=0（总体标准差），两者比值恒为 √(20/19) = {np.sqrt(20 / 19):.4f}
       实测比值：{(s20 / p20).dropna().mean():.4f} —— 与第三方结果对拍前先确认口径
    """)

    # 陷阱 3：min_periods 与预热期
    ret = vols["ret"]
    early = pd.DataFrame({
        "rolling(20)": ret.rolling(20).std(),
        "rolling(20,min_periods=5)": ret.rolling(20, min_periods=5).std(),
    }).iloc[:25] * np.sqrt(TRADING_DAYS)
    print("    ⚠️ 细节 3：min_periods 允许窗口不满时输出，但早期估计噪声很大")
    print(early.iloc[[5, 8, 12, 19, 20, 24]].to_string(float_format=lambda v: f"{v:.1%}"))
    print("""
       预热期规则：回测起点应从所有指标都"热身完毕"的日期开始，
       否则不同参数的策略起点不同，结果不可比
    """)


# ==================== 第六部分：可视化 ====================


def plot_rolling(vols: pd.DataFrame, roll_corr: pd.Series):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(vols.index, vols["close"], lw=1)
    axes[0].plot(vols.index, vols["close"].rolling(60).mean(), lw=1, label="MA60")
    axes[0].set_title(f"{NAME} 后复权收盘价")
    axes[0].legend()

    for col in ["vol_20", "vol_60", "vol_250", "ewm_hl20"]:
        axes[1].plot(vols.index, vols[col], lw=1, label=col)
    axes[1].set_title("滚动年化波动率：窗口越短越敏感")
    axes[1].legend(ncol=4)

    axes[2].plot(roll_corr.index, roll_corr, lw=1, color="tab:purple")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_title("与沪深 300 的 60 日滚动相关系数")

    path = save_figure(fig, "03_rolling_windows.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    three_windows()
    vols = rolling_volatility()
    window_tradeoff()
    roll_corr = zscore_and_correlation(vols)
    pitfalls(vols)
    plot_rolling(vols, roll_corr)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 实现 RiskMetrics 波动率：σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}，λ = 0.94。
       先用 for 循环写，再用 (ret**2).ewm(alpha=1-λ, adjust=False).mean() 向量化，对比结果
       （注意下标：t 时刻的估计只能用到 t-1 的收益）
    2. 计算 600519 的 250 日滚动最大值和最小值，标出"创 250 日新高"的日期
    3. 用 rolling(60).corr() 计算 600519 与 601088 的滚动相关，找出相关性最低的一段时期
    4. 给第三部分的实验加一个"波动从 45% 降回 15%"的切换，比较各估计方法"察觉回落"的滞后，
       与"察觉上升"的滞后是否对称？为什么？
    """)


if __name__ == "__main__":
    main()
