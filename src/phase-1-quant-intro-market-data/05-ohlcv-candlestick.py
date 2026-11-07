"""
OHLCV 与 K 线图
===============

学习目标：
    1. 理解 OHLCV 五个字段的含义和它们之间的逻辑关系
    2. 读懂 K 线：实体、上影线、下影线代表的多空博弈
    3. 用 mplfinance 绘制带均线和成交量的 K 线图
    4. 把日线重采样为周线

核心概念：
    - OHLCV：Open / High / Low / Close / Volume
    - K 线（蜡烛图）：一根 K 线浓缩一个周期内的价格走势
    - 重采样（resample）：日线 → 周线 / 月线

前置知识：
    - 03-akshare-basics.py

环境要求：
    - pip install mplfinance pandas
    - 图片保存在 outputs/phase-1/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mplfinance as mpf
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils import OUTPUT_DIR
from utils.data import get_stock_daily
from utils.plotting import setup_chinese_font

load_dotenv()

OUT = OUTPUT_DIR / "phase-1"


# ==================== 第一部分：OHLCV 字段 ====================


def ohlcv_basics(df: pd.DataFrame):
    """OHLCV 字段含义与逻辑校验"""
    print("=" * 60)
    print("第一部分：OHLCV 字段")
    print("=" * 60)

    print("""
    ┌─────────┬──────────────────────────────────────────────┐
    │ 字段    │ 含义                                         │
    ├─────────┼──────────────────────────────────────────────┤
    │ Open    │ 开盘价：开盘集合竞价成交价                   │
    │ High    │ 最高价：周期内最高成交价                     │
    │ Low     │ 最低价：周期内最低成交价                     │
    │ Close   │ 收盘价：收盘集合竞价成交价（最重要）         │
    │ Volume  │ 成交量：成交股数                             │
    │ Amount  │ 成交额：成交金额，≈ 均价 × 成交量            │
    └─────────┴──────────────────────────────────────────────┘

    必然成立的逻辑关系（数据质量检查的基础）：
        Low ≤ Open ≤ High,  Low ≤ Close ≤ High,  Volume ≥ 0
    """)

    checks = {
        "low ≤ open ≤ high": ((df["low"] <= df["open"]) & (df["open"] <= df["high"])).all(),
        "low ≤ close ≤ high": ((df["low"] <= df["close"]) & (df["close"] <= df["high"])).all(),
        "volume ≥ 0": (df["volume"] >= 0).all(),
    }
    for rule, ok in checks.items():
        print(f"   {'✅' if ok else '❌'} {rule}")

    vwap = df["amount"] / df["volume"]
    print(f"\n📌 成交均价 VWAP = 成交额 / 成交量，最近 3 天：")
    print(pd.DataFrame({"close": df["close"], "vwap": vwap}).tail(3).round(2).to_string())
    print("   VWAP 是机构衡量执行质量的基准（Phase 11 会用到）")


# ==================== 第二部分：K 线解读 ====================


def candle_anatomy(df: pd.DataFrame):
    """K 线结构与形态统计"""
    print("\n" + "=" * 60)
    print("第二部分：K 线结构")
    print("=" * 60)

    print("""
          阳线（收 > 开）              阴线（收 < 开）
               │  ← 上影线                  │
             ┌─┴─┐ ← 收盘价               ┌─┴─┐ ← 开盘价
             │   │                        │███│
             │   │ ← 实体（空心/红）      │███│ ← 实体（实心/绿）
             └─┬─┘ ← 开盘价               └─┬─┘ ← 收盘价
               │  ← 下影线                  │

    📌 A 股习惯：红涨绿跌（与欧美相反）
    📌 长上影线：盘中冲高被打回，上方抛压重
       长下影线：盘中下探被拉回，下方承接强
       十字星：  实体很小，多空僵持
    """)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]

    stats = pd.DataFrame(
        {
            "阳线": df["close"] > df["open"],
            "阴线": df["close"] < df["open"],
            "十字星(实体<10%振幅)": body / rng < 0.1,
            "长上影(上影>50%振幅)": upper / rng > 0.5,
            "长下影(下影>50%振幅)": lower / rng > 0.5,
        }
    )
    print("📊 2024 年 K 线形态统计：")
    for name, col in stats.items():
        print(f"   {name:<20s} {col.sum():>4d} 天 ({col.mean():.1%})")

    # 形态之后的次日收益：一个最朴素的"形态是否有预测力"检验
    next_ret = df["close"].pct_change().shift(-1)
    print("\n📊 各形态出现后的次日平均收益（仅用于演示思路，样本太少不能下结论）：")
    for name, col in stats.items():
        print(f"   {name:<20s} {next_ret[col].mean():+.3%}  (n={col.sum()})")
    print("""
    ⚠️ 这里用了 shift(-1)（未来数据）计算"次日收益"作为研究标签，是合理的；
       但如果在交易信号里使用 shift(-1)，就是未来函数。区分"标签"和"特征"非常重要。
    """)


# ==================== 第三部分：绘制 K 线图 ====================


def plot_candlestick(df: pd.DataFrame, title: str, filename: str):
    """用 mplfinance 绘制 K 线 + 均线 + 成交量"""
    font = setup_chinese_font()
    mc = mpf.make_marketcolors(up="red", down="green", edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        rc={"font.family": font} if font else {},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    plot_df = df.rename(columns=str.capitalize)  # mplfinance 需要 Open/High/Low/Close/Volume
    mpf.plot(
        plot_df,
        type="candle",
        style=style,
        mav=(5, 20, 60),
        volume=True,
        title=title,
        figsize=(12, 7),
        savefig=dict(fname=path, dpi=120, bbox_inches="tight"),
    )
    return path


def candlestick_chart(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：绘制 K 线图")
    print("=" * 60)

    path = plot_candlestick(df.tail(120), "600519 贵州茅台 日K线（MA5/20/60）", "05-kline-daily.png")
    print(f"🖼️  日 K 线图已保存：{path}")
    print("   图中三条均线：MA5（短期）、MA20（月线）、MA60（季线）")


# ==================== 第四部分：重采样为周线 ====================


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线：每个字段的聚合方式不同"""
    print("\n" + "=" * 60)
    print("第四部分：日线重采样为周线")
    print("=" * 60)

    rule = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"}
    weekly = df.resample("W-FRI").agg(rule).dropna(subset=["close"])
    print("""
    聚合规则：
      open → 第一天的 open      high → 最高的 high      low → 最低的 low
      close → 最后一天的 close  volume / amount → 求和
    """)
    print(weekly.tail(4).round(2).to_string())
    print(f"\n   日线 {len(df)} 根 → 周线 {len(weekly)} 根")
    print("   ⚠️ 'W-FRI' 把周五作为周的标签。节假日那周只有 3-4 个交易日，但仍是一根周线")
    print(f"   ⚠️ 最后一根周线标签是 {weekly.index[-1].date()}，但数据只到 {df.index[-1].date()}：")
    print("      标签日期可能还没到！回测中若在'标签日'使用这根周线，就用到了不完整/未来的信息")

    path = plot_candlestick(weekly, "600519 贵州茅台 周K线", "05-kline-weekly.png")
    print(f"🖼️  周 K 线图已保存：{path}")
    return weekly


# ==================== 主函数 ====================


def main():
    df = get_stock_daily("600519", "20230101", "20241231")
    df_2024 = df.loc["2024"]
    ohlcv_basics(df_2024)
    candle_anatomy(df_2024)
    candlestick_chart(df_2024)
    resample_weekly(df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 绘制沪深 300 指数 2024 年的月 K 线（resample 规则 'ME'）
    2. 统计"跳空高开"（今开 > 昨日最高）后当天收阳的比例
    3. 实现 Heikin-Ashi（平均 K 线）的计算并绘图，对比与普通 K 线的差异
    """)


if __name__ == "__main__":
    main()
