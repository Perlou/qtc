"""
技术指标：MA、EMA、MACD、RSI、ATR、布林带
=========================================

学习目标：
    1. 用 pandas/NumPy 向量化实现六个经典技术指标（不依赖 TA-Lib / pandas-ta）
    2. 理解每个指标"在描述什么"：趋势、动量、波动
    3. 用手算小例子和循环版本"对拍"，验证实现正确
    4. 理解实现细节带来的差异：EMA 的 adjust、RSI 的 Wilder 平滑、布林带的 ddof
    5. 认识指标的陷阱：复权方式、预热期、指标之间高度相关

核心概念：
    - MA / EMA：平滑后的价格，代价是滞后
    - MACD：DIF = EMA12 - EMA26，DEA = EMA9(DIF)，柱 = 2 × (DIF - DEA)（国内口径）
    - RSI：上涨力量占总力量的比例（0~100）
    - ATR：考虑跳空的平均真实波幅（价格单位）
    - 布林带：MA ± k × 标准差

前置知识：
    - 03-rolling-windows.py（rolling / ewm）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
    - TA-Lib 为可选项：已安装时会自动对拍，未安装则跳过
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"


# ==================== 指标实现（全部向量化） ====================


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """递推式 EMA：EMA_t = α·P_t + (1-α)·EMA_{t-1}，α = 2/(n+1)，与多数行情软件一致"""
    return close.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)  # 国内软件乘 2；TradingView 等海外软件一般不乘
    return dif, dea, hist


def rsi(close: pd.Series, n: int = 14, method: str = "wilder") -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)  # 写成 -delta.clip(upper=0) 会产生 -0.0，数值相同但打印难看
    if method == "wilder":  # Wilder 平滑：α = 1/n 的指数平均
        avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    else:  # Cutler's RSI：简单移动平均
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)  # 多数软件用总体标准差
    upper, lower = mid + k * std, mid - k * std
    pct_b = (close - lower) / (upper - lower)  # %B：价格在带内的相对位置
    return mid, upper, lower, pct_b


def load_adjusted_ohlc() -> pd.DataFrame:
    """用等比复权因子把 OHLC 全部换成后复权价，指标计算与收益口径一致"""
    raw = get_stock_daily_adj(SYMBOL, START, END)
    adj = raw[["open", "high", "low", "close"]].mul(raw["adj_factor"], axis=0)
    adj["volume"] = raw["volume"]
    adj["raw_close"] = raw["close"]
    adj["raw_high"], adj["raw_low"] = raw["high"], raw["low"]
    return adj


# ==================== 第一部分：指标全景 ====================


def overview():
    print("=" * 60)
    print("第一部分：指标全景")
    print("=" * 60)
    print("""
    ┌──────────┬────────────┬──────────────────────────────┬──────────┐
    │ 指标     │ 类型       │ 描述的是什么                 │ 典型参数 │
    ├──────────┼────────────┼──────────────────────────────┼──────────┤
    │ MA       │ 趋势       │ 过去 n 天平均价格            │ 5/20/60  │
    │ EMA      │ 趋势       │ 近期权重更大的平均价格       │ 12/26    │
    │ MACD     │ 趋势/动量  │ 快慢 EMA 之差及其平滑        │ 12,26,9  │
    │ RSI      │ 动量       │ 上涨力量占总力量的比例       │ 14       │
    │ ATR      │ 波动       │ 平均每天真实波动幅度（元）   │ 14       │
    │ 布林带   │ 波动/回归  │ 均线 ± k 倍标准差            │ 20, 2    │
    └──────────┴────────────┴──────────────────────────────┴──────────┘

    ⚠️ 本节重点是"正确实现 + 理解含义"。它们能否赚钱是 Phase 5 的问题——
       剧透：单独使用通常不能稳定赚钱。
    """)


# ==================== 第二部分：MA 与 EMA ====================


def ma_ema_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：MA 与 EMA —— 平滑的代价是滞后")
    print("=" * 60)

    close = df["close"]

    # 对拍 1：EMA 向量化 vs 递推循环
    alpha = 2 / (12 + 1)
    loop = np.empty(len(close))
    loop[0] = close.iloc[0]
    for i in range(1, len(close)):
        loop[i] = alpha * close.iloc[i] + (1 - alpha) * loop[i - 1]
    print(f"\n✅ 对拍：ewm(span=12, adjust=False) 与手写递推循环最大差异 {np.abs(ema(close, 12) - loop).max():.2e}")

    # adjust=True vs False：只在序列开头有差异
    e_true = close.ewm(span=12, adjust=True).mean()
    e_false = ema(close, 12)
    rel = ((e_true - e_false).abs() / close)
    print(f"   adjust=True 与 adjust=False 的相对差异：前 5 天平均 {rel.iloc[:5].mean():.3%}，"
          f"第 100 天后最大 {rel.iloc[100:].max():.1e}")
    print("   → 两种口径只在预热期不同；但不同起始日算出的 EMA 也会因此不同")

    # 滞后实验：价格从 100 跳到 110
    step = pd.Series([100.0] * 30 + [110.0] * 60)
    rows = []
    for name, line in [("MA5", sma(step, 5)), ("MA20", sma(step, 20)), ("EMA5", ema(step, 5)), ("EMA20", ema(step, 20))]:
        reached = line[line >= 109.0]  # 追上 90% 跳变幅度
        rows.append([name, reached.index[0] - 30 if len(reached) else np.nan, line.iloc[30], line.iloc[35]])
    lag = pd.DataFrame(rows, columns=["均线", "追上 90% 所需天数", "跳变当天", "跳变后第 5 天"])
    print("\n🧪 滞后实验：价格在第 30 天从 100 跳到 110 并保持：")
    print(lag.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("   💡 MA 在 n 天后完全追上；EMA 一开始反应更快，但要\"无限久\"才能完全追上")


# ==================== 第三部分：MACD ====================


def macd_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：MACD")
    print("=" * 60)

    dif, dea, hist = macd(df["close"])
    golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))  # 金叉：DIF 上穿 DEA
    death = (dif < dea) & (dif.shift(1) >= dea.shift(1))  # 死叉
    print(f"""
    DIF = EMA12 - EMA26   （> 0：短期均价在长期均价之上）
    DEA = EMA9(DIF)       （DIF 的信号线）
    柱  = 2 × (DIF - DEA)

    📊 {NAME}（后复权）：
       最新 DIF {dif.iloc[-1]:.2f}，DEA {dea.iloc[-1]:.2f}，柱 {hist.iloc[-1]:.2f}
       全区间金叉 {golden.sum()} 次、死叉 {death.sum()} 次，平均每年约 {golden.sum() / (len(df) / 252):.1f} 次金叉
       DIF > 0 的天数占比 {(dif > 0).mean():.1%}
    """)
    print("    最近 3 次金叉：" + ", ".join(str(d.date()) for d in golden[golden].index[-3:]))
    print(f"""
    ⚠️ MACD 是价格单位：本例后复权价最新约 {df['close'].iloc[-1]:,.0f}，|DIF| 的中位数约 {dif.abs().median():.2f}；
       换一只价格 10 元的股票，DIF 就只有零点几 —— 不同股票之间不可直接比较。
       跨股票比较时常用 DIF / close 做归一化。
    """)


# ==================== 第四部分：RSI ====================


def rsi_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：RSI —— 上涨力量占比")
    print("=" * 60)

    # 手算小例子：n=3 的 Cutler RSI
    toy = pd.Series([10.0, 11, 10.5, 11.5, 12, 11])
    delta = toy.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    tbl = pd.DataFrame({"价格": toy, "变动": delta, "上涨": gain, "下跌": loss,
                        "3日平均涨": gain.rolling(3).mean(), "3日平均跌": loss.rolling(3).mean(),
                        "RSI(3)": rsi(toy, 3, method="cutler")})
    print("\n🧮 手算小例子（Cutler 版，n=3）：")
    print(tbl.to_string(float_format=lambda v: f"{v:.2f}"))
    print("   例：最后一行 平均涨 0.50、平均跌 0.33 → RS = 1.5 → RSI = 100 - 100/2.5 = 60")

    close = df["close"]
    r_w, r_c = rsi(close, 14), rsi(close, 14, method="cutler")
    warm = r_w.iloc[50:]  # 跳过预热期：开头样本太少，RSI 可能出现 100 这样的极端值
    print(f"""
    📊 {NAME} RSI(14)：
       Wilder 版 最新 {r_w.iloc[-1]:.1f}；Cutler 版 最新 {r_c.iloc[-1]:.1f}
       两者平均绝对差 {(r_w - r_c).abs().mean():.2f} → 和第三方库对拍前，先确认用的是哪种平滑
       预热期（前 50 天）之后：
       Wilder RSI > 70（"超买"）天数占比 {(warm > 70).mean():.1%}，< 30（"超卖"）占比 {(warm < 30).mean():.1%}
       取值范围 [{warm.min():.1f}, {warm.max():.1f}]
    """)


# ==================== 第五部分：ATR ====================


def atr_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：ATR —— 把跳空也算进去的真实波幅")
    print("=" * 60)

    toy = pd.DataFrame({"high": [101.0, 112.0, 109.0], "low": [99.0, 108.0, 103.0], "close": [100.0, 110.0, 104.0]},
                       index=["D1", "D2", "D3"])
    toy["H-L"] = toy["high"] - toy["low"]
    toy["|H-C前|"] = (toy["high"] - toy["close"].shift()).abs()
    toy["|L-C前|"] = (toy["low"] - toy["close"].shift()).abs()
    toy["TR"] = true_range(toy)
    print("\n🧮 手算：D2 跳空高开（昨收 100，今天 108–112）")
    print(toy.to_string(float_format=lambda v: f"{v:.1f}"))
    print("   D2 的日内振幅只有 4，但真实波幅是 12 —— 跳空缺口也是风险")

    a = atr(df)
    atr_pct = a / df["close"]
    raw_df = df[["raw_high", "raw_low", "raw_close"]].set_axis(["high", "low", "close"], axis=1)
    a_raw = atr(raw_df)
    print(f"""
    📊 {NAME} ATR(14)：
       后复权价口径：ATR = {a.iloc[-1]:.2f}（后复权"元"，不是真实价格单位）
       不复权价口径：ATR = {a_raw.iloc[-1]:.2f} 元（真实价格单位，用于止损价、仓位计算）
       ATR / 收盘价：后复权 {atr_pct.iloc[-1]:.2%}，不复权 {(a_raw / df['raw_close']).iloc[-1]:.2%}
       → 两者接近（差异主要来自不复权序列里的除息缺口）：百分比形式的 ATR 基本不受
         等比复权影响，最适合跨时间、跨股票比较
    """)
    print("""    💡 量化意义：ATR 是海龟交易法则和 Phase 5 ATR 仓位管理的核心
       "波动大的少买、波动小的多买"，让每个仓位承担相近的风险
       ⚠️ 不复权 ATR 在除权除息日会把"价格缺口"误算成波动（TR 偏大），等比复权价口径没有这个问题
    """)


# ==================== 第六部分：布林带 ====================


def bollinger_demo(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：布林带")
    print("=" * 60)

    close = df["close"]
    mid, upper, lower, pct_b = bollinger(close)
    valid = mid.notna()
    outside = ((close > upper) | (close < lower))[valid]
    print(f"""
    Mid = MA20，Upper/Lower = Mid ± 2 × std(20, ddof=0)，%B = (P - Lower) / (Upper - Lower)

    📊 {NAME}：
       最新 %B = {pct_b.iloc[-1]:.2f}（> 1 在上轨之上，< 0 在下轨之下）
       收盘价落在带外的天数占比：{outside.mean():.1%}（上轨外 {(close > upper)[valid].mean():.1%}，下轨外 {(close < lower)[valid].mean():.1%}）
       带宽 (Upper-Lower)/Mid：中位数 {((upper - lower) / mid).median():.1%}，范围 {((upper - lower) / mid).min():.1%} ~ {((upper - lower) / mid).max():.1%}

    ⚠️ 常见误解："价格 95% 的时间在 ±2σ 带内"。这依赖正态分布和平稳性，
       而价格既不正态也不平稳（Phase 3），带外比例与 5% 的差距就是证据之一。
    """)


# ==================== 第七部分：指标之间高度相关 ====================


def indicator_correlation(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第七部分：指标之间高度相关")
    print("=" * 60)

    close = df["close"]
    dif, _, hist = macd(close)
    _, _, _, pct_b = bollinger(close)
    feats = pd.DataFrame({
        "close/MA20-1": close / sma(close, 20) - 1,
        "DIF/close": dif / close,
        "MACD柱/close": hist / close,
        "RSI14": rsi(close, 14),
        "布林%B": pct_b,
        "20日动量": close.pct_change(20),
    }).dropna()
    print(f"\n🔗 六个\"不同\"指标的相关系数（{len(feats)} 个交易日）：")
    print(feats.corr().round(2).to_string())
    # 取上三角（不含对角线）；pandas 3 的 stack 不再丢弃 NaN，需要显式 dropna
    c = feats.corr().where(np.triu(np.ones((6, 6), dtype=bool), k=1)).stack().dropna()
    print(f"""
    相关系数 > 0.7 的指标对：{(c > 0.7).sum()} / {len(c)}
    💡 MA 偏离、MACD、RSI、%B、动量本质上都在描述"最近涨了多少"，
       堆砌多个指标并不能带来多少独立信息
    """)


def talib_crosscheck(df: pd.DataFrame):
    try:
        import talib  # noqa: F401
    except ImportError:
        print("ℹ️ 未安装 TA-Lib，跳过第三方库对拍（本课的手算与循环对拍已足够验证正确性）")
        return
    close = df["close"].to_numpy()
    print(f"✅ TA-Lib 对拍 RSI14 最大差异（预热期后）：{np.nanmax(np.abs(talib.RSI(close, 14)[100:] - rsi(df['close']).to_numpy()[100:])):.2e}")


# ==================== 第八部分：可视化 ====================


def plot_indicators(df: pd.DataFrame):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    d = df.loc["2024-01-01":]
    close = df["close"]
    mid, upper, lower, _ = bollinger(close)
    dif, dea, hist = macd(close)
    r = rsi(close)
    a = atr(df) / close
    sl = d.index

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1, 1]})
    axes[0].plot(sl, d["close"], lw=1, color="black", label="后复权收盘")
    axes[0].plot(sl, ema(close, 12).loc[sl], lw=1, label="EMA12")
    axes[0].plot(sl, mid.loc[sl], lw=1, label="MA20 (布林中轨)")
    axes[0].fill_between(sl, lower.loc[sl], upper.loc[sl], alpha=0.15, label="布林带 ±2σ")
    axes[0].set_title(f"{NAME} 技术指标（2024）")
    axes[0].legend(ncol=4, fontsize=8)

    colors = np.where(hist.loc[sl] >= 0, "red", "green")  # A 股习惯：红涨绿跌
    axes[1].bar(sl, hist.loc[sl], color=colors, width=1.0, label="MACD 柱")
    axes[1].plot(sl, dif.loc[sl], lw=1, label="DIF")
    axes[1].plot(sl, dea.loc[sl], lw=1, label="DEA")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(sl, r.loc[sl], lw=1, color="tab:purple")
    axes[2].axhline(70, ls="--", color="gray", lw=0.8)
    axes[2].axhline(30, ls="--", color="gray", lw=0.8)
    axes[2].set_ylabel("RSI14")

    axes[3].plot(sl, a.loc[sl], lw=1, color="tab:brown")
    axes[3].set_ylabel("ATR14 / close")

    path = save_figure(fig, "06_technical_indicators.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    df = load_adjusted_ohlc()
    overview()
    ma_ema_demo(df)
    macd_demo(df)
    rsi_demo(df)
    atr_demo(df)
    bollinger_demo(df)
    indicator_correlation(df)
    talib_crosscheck(df)
    plot_indicators(df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 for 循环实现 Wilder RSI（第一个平均值用前 n 天简单平均作为种子，这是 Wilder 原始定义），
       与本课的 ewm 版本对比：差异出现在哪里？多少天后可以忽略？
    2. 实现 KDJ 指标（RSV = (C - L9) / (H9 - L9) × 100，K/D 为 1/3 平滑），并画在图上
    3. 计算"ATR 通道"：中轨 EMA20，上下轨 ± 2 × ATR14，与布林带比较带宽的变化节奏
    4. 用前复权价（get_stock_daily(..., adjust="qfq")）重算 MA20 和 ATR，
       与本课后复权结果比较 "close > MA20" 信号在多少天不一致，并解释原因
    """)


if __name__ == "__main__":
    main()
