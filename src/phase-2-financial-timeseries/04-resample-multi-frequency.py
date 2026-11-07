"""
重采样与多频率数据
==================

学习目标：
    1. 用 resample 把日线正确聚合成周线、月线（OHLCV 各列聚合方式不同）
    2. 理解 resample 的标签（label）与实际交易日不一致带来的对齐问题
    3. 掌握升采样（低频 → 高频）与"信息可得时间"：按公告日而不是报告期对齐
    4. 把周线信号正确映射回日线，避免多周期分析中的前视偏差
    5. 了解分钟线重采样中的 closed / label 约定与 A 股午休

核心概念：
    - 降采样（Downsampling）：日 → 周/月，多行聚合成一行（类似 GROUP BY）
    - 升采样（Upsampling）：月/季 → 日，一行展开成多行，需要填充
    - 信息可得时间：数据在什么时候真正能被你拿到

前置知识：
    - Phase 1：05-ohlcv-candlestick.py、07-trading-calendar-alignment.py
    - 02-annualization-volatility.py

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

from utils.data import get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"

OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}


def to_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日线 → 周线/月线 K 线；整周/整月休市产生的空行会被删除"""
    return df[list(OHLCV_AGG)].resample(rule).agg(OHLCV_AGG).dropna(subset=["close"])


# ==================== 第一部分：OHLCV 降采样 ====================


def ohlcv_downsample(df: pd.DataFrame):
    print("=" * 60)
    print("第一部分：OHLCV 降采样（日线 → 周线 / 月线）")
    print("=" * 60)

    print("""
    K 线每一列的聚合语义不同，不能一股脑求平均：
    ┌──────────────────┬──────────┬──────────────────────────────────┐
    │ 字段             │ 聚合方式 │ 原因                             │
    ├──────────────────┼──────────┼──────────────────────────────────┤
    │ open             │ first    │ 本周第一个交易日的开盘价         │
    │ high             │ max      │ 本周最高                         │
    │ low              │ min      │ 本周最低                         │
    │ close            │ last     │ 本周最后一个交易日的收盘价       │
    │ volume / amount  │ sum      │ 成交量、成交额可加               │
    └──────────────────┴──────────┴──────────────────────────────────┘
    """)

    raw_weeks = df[list(OHLCV_AGG)].resample("W-FRI").agg(OHLCV_AGG)
    weekly = to_bars(df, "W-FRI")
    monthly = to_bars(df, "ME")
    empty = raw_weeks.index[raw_weeks["close"].isna()]
    print(f"📊 {len(df)} 根日线 → {len(weekly)} 根周线、{len(monthly)} 根月线")
    if len(empty) > 0:
        print(f"   resample 产生了 {len(empty)} 个空周（整周休市，如春节、国庆），需要 dropna 删除：")
        print("   " + ", ".join(str(d.date()) for d in empty[:8]) + (" ..." if len(empty) > 8 else ""))
    else:
        print("   当前数据没有空周（合成数据不含节假日）；真实 A 股的春节、国庆会产生空周，需要 dropna 删除")

    print("\n📋 前 3 根周线（不复权价格，K 线用真实成交价）：")
    print(weekly.head(3).to_string(formatters={
        "volume": "{:,.0f}".format, "amount": "{:,.0f}".format,
        **{c: "{:.2f}".format for c in ["open", "high", "low", "close"]},
    }))

    # 验证：第一周的 high 是否等于该周日线 high 的最大值
    wk_end = weekly.index[0]
    days = df.loc[: wk_end, ["open", "high", "low", "close"]]
    print(f"\n✅ 验证第一周（{days.index[0].date()} ~ {days.index[-1].date()}，{len(days)} 个交易日）：")
    print(f"   日线 high 最大值 {days['high'].max():.2f} = 周线 high {weekly['high'].iloc[0]:.2f}")
    print(f"   第一天 open {days['open'].iloc[0]:.2f} = 周线 open {weekly['open'].iloc[0]:.2f}")

    wrong = df[["open", "high", "low", "close"]].resample("W-FRI").mean().dropna()
    err = (wrong - weekly[["open", "high", "low", "close"]]).abs().mean()
    print("\n❌ 如果直接 .mean()，各列与正确周线的平均绝对误差：")
    print("   " + "  ".join(f"{k}: {v:.2f}" for k, v in err.items()))
    print("   → 周线 high 被低估、low 被高估，K 线形态完全失真")
    return weekly, monthly


# ==================== 第二部分：标签与对齐 ====================


def label_alignment(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：resample 的标签与实际交易日")
    print("=" * 60)

    close = df["adj_close"]
    m_resample = close.resample("ME").last()
    # 正确方法：按月份分组，取每组最后一行 → 索引就是实际的最后一个交易日
    m_actual = close.groupby(close.index.to_period("M")).tail(1)

    in_index = m_resample.index.isin(close.index)
    print(f"""
    resample("ME") 的标签是日历月末，但月末可能是周末或节假日：
       {len(m_resample)} 个月度标签中，有 {(~in_index).sum()} 个不是交易日
    """)
    cmp_df = pd.DataFrame({
        "resample 标签": m_resample.index.date,
        "实际最后交易日": m_actual.index.date,
        "收盘价相同": np.isclose(m_resample.to_numpy(), m_actual.to_numpy()),
    })
    print(cmp_df[~in_index].head(6).to_string(index=False))
    print("   → 数值完全一样，只是\"贴的日期标签\"不同")

    # 后果：按标签把月度信号合并回日线
    # 先 dropna：第一个月没有收益，NaN > 0 会被算成 False，变成一个虚假的"看空"信号
    signal = (m_resample.pct_change().dropna() > 0).astype(int).rename("signal")
    merged = df[["adj_close"]].join(signal, how="left")
    print(f"""
    ❌ 后果：把月度信号按标签 join 回日线
       月度信号 {signal.count()} 个，join 后只匹配上 {merged['signal'].notna().sum()} 个
       → {signal.count() - merged['signal'].notna().sum()} 个信号悄悄丢失，代码不会报任何错

    ✅ 两种修复：
       1. 用实际最后一个交易日作标签：close.groupby(close.index.to_period("M")).tail(1)
       2. 用 reindex(..., method="ffill") 前向填充，再 shift(1) 确保"下一个交易日"才生效
    """)
    sig_daily = signal.reindex(close.index, method="ffill").shift(1)
    print(f"   方法 2 结果：日线上有效信号 {sig_daily.notna().sum()} 天（每个月的信号都延续到下个月）")
    print(f"""   ⚠️ 方法 2 的细节：标签落在非交易日的 {(~in_index).sum()} 个月，ffill 已经把信号推到下月第一个交易日，
      再 shift(1) 就会多晚一天生效。这偏保守、不构成前视，但各月生效时点不一致；方法 1 没有这个问题""")


# ==================== 第三部分：升采样与信息可得时间 ====================


def make_fundamentals(start_year: int = 2018, end_year: int = 2024) -> pd.DataFrame:
    """构造示意用的季度 ROE 数据：报告期 + 公告日（公告日为简化假设）"""
    rng = np.random.default_rng(3)
    rows = []
    for y in range(start_year, end_year + 1):
        for q_end, announce in [
            (f"{y}-03-31", f"{y}-04-28"),
            (f"{y}-06-30", f"{y}-08-28"),
            (f"{y}-09-30", f"{y}-10-28"),
            (f"{y}-12-31", f"{y + 1}-03-30"),
        ]:
            rows.append({"report_period": pd.Timestamp(q_end), "announce_date": pd.Timestamp(announce),
                         "roe": round(rng.normal(0.08, 0.02), 4)})
    return pd.DataFrame(rows)


def upsample_point_in_time(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：升采样与\"信息可得时间\"")
    print("=" * 60)

    print("""
    把季度/月度数据扩展到日频时，必须按"数据公布日"而不是"数据所属期"对齐：

       年报：报告期 2023-12-31  →  公告日通常在次年 3–4 月
       ❌ 按报告期对齐：2024 年 1 月的策略就"知道"了 2023 年年报
       ✅ 按公告日对齐，然后前向填充（ffill）
    """)

    fund = make_fundamentals()
    print("📋 示意数据（ROE 为随机生成，公告日为简化假设）：")
    print(fund.head(5).to_string(index=False))

    # ⚠️ pandas 3 的时间戳可能有不同精度（数据源返回 ms，pd.Timestamp 构造的是 us 等），
    #    merge_asof 要求两边 dtype 完全一致，先统一成 datetime64[ns]
    dates = df.index.astype("datetime64[ns]")
    fund = fund.astype({"report_period": "datetime64[ns]", "announce_date": "datetime64[ns]"})
    raw_dtypes = (df.index.dtype, make_fundamentals()["announce_date"].dtype)
    if raw_dtypes[0] != raw_dtypes[1]:
        print(f"\n⚙️ pandas 3 细节：行情索引 dtype 为 {raw_dtypes[0]}，示意数据为 {raw_dtypes[1]}，不统一会抛 MergeError")
    wrong = fund.set_index("report_period")["roe"].reindex(dates, method="ffill")
    # merge_asof：对每个交易日，找"公告日 ≤ 当天"的最近一条记录 —— 时点对齐的利器
    right = pd.merge_asof(
        pd.DataFrame({"date": dates}),
        fund.sort_values("announce_date")[["announce_date", "report_period", "roe"]],
        left_on="date", right_on="announce_date", direction="backward",
    ).set_index("date")

    differ = (wrong != right["roe"]) & wrong.notna()
    print(f"""
    🔍 对比两种对齐方式（共 {len(dates)} 个交易日）：
       按报告期对齐，有 {differ.sum()} 天（{differ.mean():.0%}）用到的 ROE 与"当天真实可得"的不同
       → 这些天都在使用尚未公布的数据，即前视偏差
    """)
    d = dates[min(dates.searchsorted(pd.Timestamp("2024-01-15")), len(dates) - 1)]
    wrong_src = fund.set_index("report_period").index.asof(d)
    print(f"   例：{d.date()} 这一天")
    print(f"      按报告期：ROE = {wrong.loc[d]:.4f}（来自报告期 {wrong_src.date()}，此时还没公布！）")
    print(f"      按公告日：ROE = {right.loc[d, 'roe']:.4f}（来自报告期 {right.loc[d, 'report_period'].date()}，"
          f"公告于 {right.loc[d, 'announce_date'].date()}）")
    print("""
    💡 更保守的做法：如果公告在盘后发布，当天无法交易，应再 shift(1) 个交易日。
       这是 Phase 7 多因子研究中最常见的前视偏差来源之一，现在就要养成习惯。
    """)


# ==================== 第四部分：多周期信号映射回日线 ====================


def multi_timeframe(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：周线信号映射回日线")
    print("=" * 60)

    close = df["adj_close"]
    ret = close.pct_change().fillna(0)

    # 周线：用实际最后一个交易日作标签
    wk = close.groupby(close.index.to_period("W-FRI")).tail(1)
    wk_signal = (wk > wk.rolling(10).mean()).astype(int)  # 周收盘价在 10 周均线上方 → 看多

    # ❌ 错误：bfill —— 周一到周四使用了本周五收盘后才知道的信号
    pos_wrong = wk_signal.reindex(close.index, method="bfill")
    # ✅ 正确：ffill 到日线后再 shift(1) —— 周五收盘得到信号，下周一才持仓
    pos_right = wk_signal.reindex(close.index, method="ffill").shift(1)

    valid = pos_right.notna() & pos_wrong.notna() & (close.index >= wk.index[10])
    nav_wrong = (1 + (pos_wrong * ret)[valid]).cumprod()
    nav_right = (1 + (pos_right * ret)[valid]).cumprod()
    nav_hold = (1 + ret[valid]).cumprod()
    diff_days = (pos_wrong[valid] != pos_right[valid]).sum()
    print(f"""
    策略：周收盘价在 10 周均线上方时持有 {NAME}，否则空仓（不计成本）

       持仓不同的天数：{diff_days} / {valid.sum()}
       期末净值  买入持有：{nav_hold.iloc[-1]:.3f}
                 ✅ 正确映射（ffill + shift）：{nav_right.iloc[-1]:.3f}
                 ❌ 错误映射（bfill，前视）：  {nav_wrong.iloc[-1]:.3f}
    """)
    better = nav_wrong.iloc[-1] > nav_right.iloc[-1]
    print("    " + ("→ 前视版本净值更高：它在周一就\"知道\"了本周五的收盘价" if better
                   else "→ 本例中前视版本没有更高，但它依然是错的——前视偏差不保证回测更好看，只保证不可复现"))
    print("""
    💡 记忆口诀：低频信号在"它被计算出来的那一刻之后"才能用。
       周线收盘是周五收盘 → 周五盘中的日线逻辑不能用"本周周线"
    """)


# ==================== 第五部分：分钟线重采样与 A 股午休 ====================


def intraday_resample():
    print("\n" + "=" * 60)
    print("第五部分：分钟线重采样（closed / label 约定）")
    print("=" * 60)

    # 构造一天的 1 分钟线：A 股连续竞价 9:30-11:30、13:00-15:00，每根 K 线以"结束时刻"作标签
    am = pd.date_range("2024-06-03 09:31", "2024-06-03 11:30", freq="1min")
    pm = pd.date_range("2024-06-03 13:01", "2024-06-03 15:00", freq="1min")
    idx = am.append(pm)
    rng = np.random.default_rng(5)
    price = 10 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx))))
    minute = pd.DataFrame({"open": price, "high": price * 1.0005, "low": price * 0.9995, "close": price,
                           "volume": rng.integers(100, 1000, len(idx)) * 100}, index=idx)
    print(f"🕐 构造 {len(minute)} 根 1 分钟 K 线（A 股每天 4 小时 = 240 分钟）")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    bars_right = minute.resample("30min", closed="right", label="right").agg(agg)
    bars_default = minute.resample("30min").agg(agg)
    print(f"""
    resample("30min", closed="right", label="right")：
       (09:30, 10:00] 这 30 根分钟线 → 标签 10:00，与行情软件的 30 分钟 K 线一致
       结果共 {len(bars_right)} 行，其中午休产生 {bars_right['close'].isna().sum()} 个空行 → 需要 dropna
    """)
    print(bars_right.dropna().round(3).to_string())
    print(f"""
    ⚠️ 默认参数 resample("30min")（closed="left", label="left"）得到 {len(bars_default.dropna())} 根非空 K 线，
       第一根标签是 {bars_default.dropna().index[0].time()}，最后一根是 {bars_default.dropna().index[-1].time()}
       —— 11:30 和 15:00 这两根分钟线各自被单独切成了一根"K 线"，与行情软件对不上

    📌 其他市场：
       - 期货夜盘（21:00 起）在交易所口径上归属下一个交易日，按自然日 resample 会切错，要按交易日字段分组
       - 加密货币 24 小时交易，"日线"按 UTC 0 点还是北京时间 0 点切分，因数据源而异，务必统一
    """)


# ==================== 第六部分：可视化 ====================


def plot_frequencies(df: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    fig, ax = plt.subplots(figsize=(12, 5))
    sub = slice("2024-01-01", "2024-12-31")
    ax.plot(df.loc[sub].index, df.loc[sub, "close"], lw=0.8, label="日线收盘")
    ax.step(weekly.loc[sub].index, weekly.loc[sub, "close"], where="post", lw=1.2, label="周线收盘（标签=周五）")
    ax.plot(monthly.loc[sub].index, monthly.loc[sub, "close"], "o--", label="月线收盘（标签=日历月末）")
    ax.set_title(f"{NAME} 2024 年：同一价格序列的三种频率（不复权）")
    ax.legend()
    path = save_figure(fig, "04_resample_frequencies.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    df = get_stock_daily_adj(SYMBOL, START, END)
    weekly, monthly = ohlcv_downsample(df)
    label_alignment(df)
    upsample_point_in_time(df)
    multi_timeframe(df)
    intraday_resample()
    plot_frequencies(df, weekly, monthly)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 改写 to_bars：让周线的索引是"该周实际最后一个交易日"而不是周五
       提示：groupby(df.index.to_period("W-FRI")).agg(...)，再用每组最后一个日期作索引
    2. 用 resample("QE") 生成季线，计算 600519 的季度收益，并验证它等于对应三个月收益的连乘
    3. 把第三部分的公告日改为"公告日 + 1 个交易日"（模拟盘后公告），统计前视天数如何变化
    4. 把第四部分的周线策略改成月线（10 个月均线），比较正确与错误映射的差异
    """)


if __name__ == "__main__":
    main()
