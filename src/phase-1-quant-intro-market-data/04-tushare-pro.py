"""
Tushare Pro 数据接口
====================

学习目标：
    1. 学会注册并配置 Tushare Pro token
    2. 掌握 daily（日线）、adj_factor（复权因子）、trade_cal（交易日历）三个核心接口
    3. 用 Tushare 与 AKShare 交叉验证数据，建立"数据要互相核对"的习惯

核心概念：
    - Tushare Pro：基于积分权限的金融数据 API，字段规范、接口稳定
    - ts_code：Tushare 的证券代码格式，如 600519.SH、000001.SZ
    - 复权因子：用于把不复权价格转换成复权价格（详见 06-adjusted-prices.py）

前置知识：
    - 03-akshare-basics.py

环境要求：
    - pip install tushare pandas
    - 在 .env 中配置 TUSHARE_TOKEN（https://tushare.pro/register 注册获取）
    - 没有 token 时本课只讲解概念并跳过 API 调用
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from utils.data import get_stock_daily, is_offline

load_dotenv()


def get_pro_api():
    """初始化 Tushare Pro API；没有 token 或离线时返回 None"""
    token = os.getenv("TUSHARE_TOKEN", "")
    if is_offline():
        print("⏭️  离线模式，跳过 Tushare API 调用")
        return None
    if not token or token == "your_tushare_token_here":
        print("""
    ⏭️  未配置 TUSHARE_TOKEN，跳过 API 调用。配置方法：
        1. 访问 https://tushare.pro/register 注册
        2. 在「个人主页 → 接口TOKEN」复制 token
        3. 写入 .env：TUSHARE_TOKEN=你的token
        """)
        return None
    import tushare as ts

    return ts.pro_api(token)


def safe_query(fn, desc: str):
    try:
        return fn()
    except Exception as e:
        print(f"⚠️ {desc} 失败：{str(e)[:120]}")
        print("   可能原因：积分不足 / 调用频率超限 / token 无效")
        return None


# ==================== 第一部分：Tushare 基础 ====================


def tushare_intro():
    """Tushare Pro 介绍"""
    print("=" * 60)
    print("第一部分：Tushare Pro 基础")
    print("=" * 60)

    print("""
    📌 积分制：注册送基础积分，积分越高可调用的接口越多、频率越高
       - 日线 daily、交易日历 trade_cal：基础积分即可
       - 复权因子、财务数据、分钟线等：需要更高积分（以官网说明为准）

    📌 代码格式对照：
    ┌──────────────┬──────────────┬──────────────┐
    │ 股票         │ AKShare      │ Tushare      │
    ├──────────────┼──────────────┼──────────────┤
    │ 贵州茅台     │ 600519       │ 600519.SH    │
    │ 平安银行     │ 000001       │ 000001.SZ    │
    │ 北交所股票   │ 920xxx       │ 920xxx.BJ    │
    └──────────────┴──────────────┴──────────────┘

    📌 Tushare 日线的单位：vol 是"手"，amount 是"千元" ⚠️ 与 AKShare 不同
    """)

    def to_ts_code(symbol: str) -> str:
        if symbol.startswith(("6", "9")) and not symbol.startswith("92"):
            return f"{symbol}.SH"
        if symbol.startswith(("8", "4", "92")):
            return f"{symbol}.BJ"
        return f"{symbol}.SZ"

    for s in ["600519", "000001", "300750", "688981"]:
        print(f"   {s} → {to_ts_code(s)}")


# ==================== 第二部分：日线与复权因子 ====================


def daily_and_adj_factor(pro):
    """日线行情与复权因子"""
    print("\n" + "=" * 60)
    print("第二部分：日线与复权因子")
    print("=" * 60)

    if pro is None:
        print("""
    示例代码（需要 token）：

        df = pro.daily(ts_code="600519.SH", start_date="20240101", end_date="20240131")
        adj = pro.adj_factor(ts_code="600519.SH", start_date="20240101", end_date="20240131")
        df = df.merge(adj, on=["ts_code", "trade_date"])
        df["close_hfq"] = df["close"] * df["adj_factor"]   # 后复权价格
        """)
        return None

    df = safe_query(
        lambda: pro.daily(ts_code="600519.SH", start_date="20240101", end_date="20240131"), "获取日线"
    )
    if df is None:
        return None
    df = df.sort_values("trade_date")
    print("\n📋 pro.daily 返回（注意：默认按日期倒序，这里已改为正序）：")
    print(df.head(3).to_string(index=False))

    adj = safe_query(
        lambda: pro.adj_factor(ts_code="600519.SH", start_date="20240101", end_date="20240131"), "获取复权因子"
    )
    if adj is not None:
        df = df.merge(adj, on=["ts_code", "trade_date"])
        df["close_hfq"] = df["close"] * df["adj_factor"]
        print("\n📋 合并复权因子后：")
        print(df[["trade_date", "close", "adj_factor", "close_hfq"]].head(3).to_string(index=False))
    return df


# ==================== 第三部分：交易日历 ====================


def trade_calendar(pro):
    """交易日历"""
    print("\n" + "=" * 60)
    print("第三部分：交易日历")
    print("=" * 60)

    if pro is None:
        print("   示例：pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20241231')")
        print("   不用 Tushare 时可用 AKShare：ak.tool_trade_date_hist_sina()（见 07 课）")
        return

    cal = safe_query(
        lambda: pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20241231"), "获取交易日历"
    )
    if cal is None:
        return
    open_days = cal[cal["is_open"] == 1]
    print(f"📅 2024 年：自然日 {len(cal)} 天，交易日 {len(open_days)} 天")
    print("   年化时常用 252 个交易日，实际每年在 240~250 之间")


# ==================== 第四部分：交叉验证 ====================


def cross_validate(ts_df):
    """Tushare 与 AKShare 数据交叉验证"""
    print("\n" + "=" * 60)
    print("第四部分：数据源交叉验证")
    print("=" * 60)

    if ts_df is None:
        print("   需要 Tushare 数据才能交叉验证，已跳过")
        return

    ak_df = get_stock_daily("600519", "20240101", "20240131")
    ts = ts_df.assign(date=pd.to_datetime(ts_df["trade_date"])).set_index("date")
    merged = pd.DataFrame(
        {
            "close_ak": ak_df["close"],
            "close_ts": ts["close"],
            "vol_ak(股)": ak_df["volume"],
            "vol_ts(股)": ts["vol"] * 100,  # 手 → 股
        }
    ).dropna()
    merged["close_diff"] = (merged["close_ak"] - merged["close_ts"]).abs()
    merged["vol_diff%"] = (merged["vol_ak(股)"] / merged["vol_ts(股)"] - 1).abs() * 100
    print(merged.head().round(4).to_string())
    print(f"\n✅ 收盘价最大差异：{merged['close_diff'].max():.4f} 元")
    print(f"✅ 成交量最大相对差异：{merged['vol_diff%'].max():.4f}%")
    print("""
    💡 两个数据源一致时可以放心；不一致时要查明原因（单位、复权、数据错误），
       而不是随便挑一个用。这是专业量化研究员的基本素养。
    """)


# ==================== 主函数 ====================


def main():
    tushare_intro()
    pro = get_pro_api()
    ts_df = daily_and_adj_factor(pro)
    trade_calendar(pro)
    cross_validate(ts_df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 Tushare 获取 000001.SZ 的日线和复权因子，计算后复权收盘价
    2. 写一个函数 ts_to_standard(df)，把 Tushare 日线转成课程统一格式
       （index=date，列 open/high/low/close/volume(股)/amount(元)）
    3. 对 5 只股票做 AKShare vs Tushare 交叉验证，汇总差异
    """)


if __name__ == "__main__":
    main()
