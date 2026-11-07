"""
交易日历、停牌与多资产对齐
==========================

学习目标：
    1. 使用交易日历，区分自然日、工作日、交易日
    2. 识别停牌，理解停牌对收益计算和回测的影响
    3. 把多只股票对齐到同一个交易日网格上
    4. 掌握缺失值的正确处理方式（不能一律填 0）

核心概念：
    - 交易日历：交易所公布的开市日期，不等于工作日（有节假日、调休）
    - 停牌：股票暂停交易，当天没有行情数据
    - 对齐（alignment）：多资产数据统一到相同的日期索引
    - 前向填充（ffill）：用最近一个已知值填充缺失，只能向前不能向后

前置知识：
    - 03-akshare-basics.py

环境要求：
    - pip install akshare pandas
    - 无网络时使用合成数据和工作日近似交易日
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_stock_daily_adj, is_offline

load_dotenv()


def get_trade_calendar() -> pd.DatetimeIndex:
    """获取 A 股交易日历；离线时用工作日近似"""
    if not is_offline():
        try:
            import akshare as ak

            cal = ak.tool_trade_date_hist_sina()
            return pd.DatetimeIndex(pd.to_datetime(cal["trade_date"]), name="date")
        except Exception as e:
            print(f"⚠️ 获取交易日历失败（{type(e).__name__}），使用工作日近似")
    return pd.bdate_range("1990-12-19", "2026-12-31", name="date")


# ==================== 第一部分：交易日历 ====================


def trading_calendar(cal: pd.DatetimeIndex):
    """自然日、工作日、交易日"""
    print("=" * 60)
    print("第一部分：交易日历")
    print("=" * 60)

    year = cal[(cal >= "2024-01-01") & (cal <= "2024-12-31")]
    bdays = pd.bdate_range("2024-01-01", "2024-12-31")
    holidays = bdays.difference(year)
    print(f"""
    📅 2024 年：
       自然日 366 天
       工作日 {len(bdays)} 天（周一到周五）
       交易日 {len(year)} 天
       工作日中休市 {len(holidays)} 天（春节、国庆等法定节假日）
    """)
    print("   2024 年国庆休市日：", ", ".join(d.strftime("%m-%d") for d in holidays if d.month == 10))

    print("""
    📌 为什么需要交易日历？
       1. "持有 20 个交易日" ≠ "持有 20 天" ≠ "持有 4 周"
       2. 判断停牌：交易日历上有、但个股数据里没有的日期 → 停牌
       3. 计划调仓日：如"每月最后一个交易日"，要用交易日历计算
    """)

    # 每月最后一个交易日（常用的月度调仓日）
    s = pd.Series(year, index=year)
    month_end = s.groupby(s.index.to_period("M")).last()
    print("📌 2024 年每月最后一个交易日：")
    print("   " + ", ".join(d.strftime("%m-%d") for d in month_end))


# ==================== 第二部分：识别停牌 ====================


def detect_suspension(cal: pd.DatetimeIndex):
    """用交易日历识别停牌"""
    print("\n" + "=" * 60)
    print("第二部分：识别停牌")
    print("=" * 60)

    # 000002 万科A 在 2015-12-18 ~ 2016-07-04 长期停牌（宝万之争）
    symbol, start, end = "000002", "20151101", "20160831"
    df = get_stock_daily_adj(symbol, start, end)
    window = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    missing = window.difference(df.index)

    print(f"\n📋 {symbol} 万科A，{start} ~ {end}")
    print(f"   交易日 {len(window)} 天，有行情 {len(df)} 天，缺失 {len(missing)} 天")
    if len(missing) > 0:
        # 把连续缺失的日期合并为停牌区间
        pos = window.get_indexer(missing)
        groups = np.split(missing, np.where(np.diff(pos) != 1)[0] + 1)
        print("   停牌区间：")
        for g in groups:
            print(f"     {g[0].date()} ~ {g[-1].date()}（{len(g)} 个交易日）")

        last_before = df.loc[: missing[0]].iloc[-1]
        first_after = df.loc[missing[-1] :].iloc[0]
        gap_ret = first_after["adj_close"] / last_before["adj_close"] - 1
        print(f"\n   停牌前收盘 {last_before['close']:.2f}，复牌首日收盘 {first_after['close']:.2f}")
        print(f"   复牌首日相对停牌前收益：{gap_ret:+.2%}（跌停，之后又连续多个跌停）")
        print("   停牌期间积累的风险，在复牌后集中释放；停牌期间你无法卖出")
    else:
        print("   （合成数据没有停牌，联网运行可看到真实停牌区间）")

    print("""
    ⚠️ 停牌对回测的影响：
       1. 停牌期间无法买卖 —— 回测引擎必须拒绝停牌日的订单
       2. 停牌期间的收益在复牌日集中释放 —— 不能把它"平摊"或"忽略"
       3. 长期停牌的股票是流动性风险的来源（想卖卖不掉）
    """)


# ==================== 第三部分：多资产对齐 ====================


def align_multi_assets(cal: pd.DatetimeIndex):
    """多只股票对齐到交易日网格"""
    print("\n" + "=" * 60)
    print("第三部分：多资产对齐")
    print("=" * 60)

    # 万科A、中国神华正常交易；603259 药明康德 2018-05-08 才上市
    symbols = {"000002": "万科A", "601088": "中国神华", "603259": "药明康德"}
    start, end = "20180401", "20180630"
    closes = {s: get_stock_daily_adj(s, start, end)["adj_close"] for s in symbols}

    # 方式 1：直接 concat —— 索引是各股票日期的并集
    panel = pd.DataFrame(closes)
    grid = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    # 方式 2：reindex 到交易日历 —— 显式地暴露所有缺失
    panel = panel.reindex(grid)

    print("\n📊 各股票缺失天数：")
    for s, name in symbols.items():
        first = panel[s].first_valid_index()
        print(f"   {s} {name}：缺失 {panel[s].isna().sum():>2d} 天，首个有效日期 {first.date() if first is not None else '-'}")

    print("""
    缺失的原因不同，处理方式也不同：
    ┌──────────────┬─────────────────────────┬──────────────────────────────┐
    │ 原因         │ 价格处理                │ 收益 / 交易处理              │
    ├──────────────┼─────────────────────────┼──────────────────────────────┤
    │ 尚未上市     │ 保持 NaN                │ 不在可投资范围内             │
    │ 停牌         │ ffill（市值按最后价格） │ 当日收益 = 0，禁止交易       │
    │ 已退市       │ 保持 NaN                │ 按退市价格清算，不能消失     │
    │ 数据错误     │ 查明原因，不要猜        │ 先修数据                     │
    └──────────────┴─────────────────────────┴──────────────────────────────┘
    """)


# ==================== 第四部分：缺失值处理的陷阱 ====================


def missing_value_pitfalls():
    """常见的错误处理方式"""
    print("\n" + "=" * 60)
    print("第四部分：缺失值处理的陷阱")
    print("=" * 60)

    dates = pd.bdate_range("2024-01-01", periods=8)
    price = pd.Series([10.0, 10.5, np.nan, np.nan, 9.0, 9.2, 9.5, 9.8], index=dates)  # 第 3-4 天停牌

    df = pd.DataFrame({
        "原始": price,
        "❌ fillna(0)": price.fillna(0),
        "❌ bfill": price.bfill(),
        "❌ interpolate": price.interpolate(),
        "✅ ffill": price.ffill(),
    })
    print("\n📋 价格：")
    print(df.round(2).to_string())

    rets = df.pct_change()
    print("\n📋 对应的日收益：")
    print(rets.round(4).replace([np.inf, -np.inf], np.nan).to_string())

    print("""
    🔍 分析：
    - fillna(0)：价格变成 0 → 收益 -100% 再 +∞，彻底破坏统计
    - bfill：用复牌后的价格填充停牌日 → 停牌日就"知道"了未来价格（未来函数！）
    - interpolate：线性插值同样用到了未来价格，而且凭空制造出平滑的走势
    - ffill：停牌日价格不变、收益为 0，复牌日一次性体现停牌期间的涨跌 ✅

    📌 规则：金融时间序列只能用过去填充未来，绝不能用未来填充过去
    """)

    # pct_change 的默认行为也要注意
    print("📌 注意 pandas 的 pct_change：")
    print(f"   price.pct_change(fill_method=None) 在停牌日附近给 NaN：共 {price.pct_change(fill_method=None).isna().sum()} 个 NaN（含首行）")
    print("   先 ffill 再 pct_change：停牌日收益为 0，复牌日为停牌期间累计收益")


# ==================== 主函数 ====================


def main():
    cal = get_trade_calendar()
    trading_calendar(cal)
    detect_suspension(cal)
    align_multi_assets(cal)
    missing_value_pitfalls()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 写函数 next_trade_date(date, n)，返回 date 之后第 n 个交易日
    2. 统计 2024 年沪深 300 成分股中，停牌超过 5 个交易日的股票
    3. 实现 align_panel(dfs, calendar)，返回对齐后的收盘价宽表和一张
       "状态表"（normal / suspended / not_listed）
    4. 思考：对于多因子选股，停牌股在调仓日应该如何处理？
    """)


if __name__ == "__main__":
    main()
