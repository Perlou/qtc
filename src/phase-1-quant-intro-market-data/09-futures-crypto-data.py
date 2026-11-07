"""
期货与加密货币数据
==================

学习目标：
    1. 理解期货合约的基本要素：合约乘数、保证金、到期、主力合约
    2. 获取期货单个合约与主力连续数据，识别换月造成的价格跳空
    3. 用 ccxt 获取加密货币 K 线数据
    4. 比较 A 股、期货、加密货币在交易机制和数据上的差异

核心概念：
    - 期货合约：约定未来某日按约定价格交割标的物的标准化合约
    - 杠杆与保证金：只需交合约价值 10% 左右的保证金，盈亏按全部价值计算
    - 主力合约 / 主力连续：成交持仓最大的合约；把不同月份主力合约拼接成的连续序列
    - ccxt：统一了上百家加密货币交易所 API 的开源库

前置知识：
    - 06-adjusted-prices.py（复权思想与换月调整相通）

环境要求：
    - pip install akshare ccxt pandas
    - 访问海外交易所可能需要在 .env 中设置 HTTPS_PROXY
    - 无网络时自动使用合成数据

⚠️ 合规提示：中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动。
   本课仅获取公开行情数据用于学习研究，不涉及任何交易。请遵守所在地法律法规。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_index_daily, is_offline, synthetic_ohlcv

load_dotenv()


# ==================== 第一部分：期货合约基础 ====================


def futures_basics():
    print("=" * 60)
    print("第一部分：期货合约基础")
    print("=" * 60)

    print("""
    📦 期货 = 约定"未来某天，按今天谈好的价格，买卖一定数量的商品"

    国内主要期货交易所：
      上期所 SHFE（螺纹钢、铜、黄金） 大商所 DCE（铁矿石、豆粕）
      郑商所 CZCE（棉花、PTA、白糖）   中金所 CFFEX（股指期货、国债期货）
      广期所 GFEX（工业硅、碳酸锂）   上期能源 INE（原油）

    合约代码：品种 + 到期年月，如 RB2505 = 螺纹钢 2025 年 5 月到期
    """)

    # 合约规格（以交易所最新规定为准；保证金比例由交易所和期货公司设定且会调整）
    specs = pd.DataFrame(
        [
            ["RB", "螺纹钢", "上期所", 10, "吨", 3500, 0.10],
            ["CU", "沪铜", "上期所", 5, "吨", 75000, 0.10],
            ["AU", "沪金", "上期所", 1000, "克", 600, 0.10],
            ["IF", "沪深300股指", "中金所", 300, "元/点", 3900, 0.12],
        ],
        columns=["代码", "品种", "交易所", "合约乘数", "单位", "示例价格", "保证金比例(示例)"],
    )
    specs["合约价值"] = specs["合约乘数"] * specs["示例价格"]
    specs["一手保证金"] = specs["合约价值"] * specs["保证金比例(示例)"]
    specs["杠杆倍数"] = (1 / specs["保证金比例(示例)"]).round(1)
    print(specs.to_string(index=False))

    print("""
    💡 杠杆的含义：保证金 10% → 10 倍杠杆
       价格下跌 5% → 亏损 = 保证金的 50%
       价格下跌 10% → 保证金亏光，期货公司会要求追加保证金或强行平仓

    📌 与 A 股的核心差异：
       T+0（当天可开平仓）、可以做空、有杠杆、合约会到期、有夜盘
    """)


# ==================== 第二部分：主力合约与换月跳空 ====================


def load_futures():
    """获取螺纹钢主力连续与各月合约收盘价"""
    import akshare as ak

    main = ak.futures_main_sina(symbol="RB0", start_date="20240101", end_date="20241231")
    main = main.assign(date=pd.to_datetime(main["日期"])).set_index("date")["收盘价"].astype(float)
    contracts = {}
    for c in ["RB2401", "RB2405", "RB2410", "RB2501", "RB2505"]:
        d = ak.futures_zh_daily_sina(symbol=c)
        contracts[c] = d.assign(date=pd.to_datetime(d["date"])).set_index("date")["close"].astype(float)
    return main, pd.DataFrame(contracts).reindex(main.index)


def main_contract_rollover():
    print("\n" + "=" * 60)
    print("第二部分：主力合约与换月跳空")
    print("=" * 60)

    print("""
    同一时间有多个月份的合约在交易，资金集中在"主力合约"上。
    主力合约会定期切换（螺纹钢一般在 1、5、10 月合约之间轮换）：

    时间 ──────────────────────────────────────────────────►
    RB2405 ████████████████▓▓▓░░░                      （临近到期，资金撤离）
    RB2410            ░░░▓▓▓████████████████▓▓▓░░░
    RB2501                              ░░░▓▓▓████████████
             ↑ 主力连续 = 每天取当时的主力合约价格拼接
    """)

    if is_offline():
        print("⏭️  离线模式：跳过真实期货数据演示")
        return
    try:
        main, contracts = load_futures()
    except Exception as e:
        print(f"⚠️ 获取期货数据失败（{type(e).__name__}），跳过本部分")
        return

    # 识别每天的主力连续对应哪个合约（价格完全相同的那个）
    diff = contracts.sub(main, axis=0).abs()
    held = diff.idxmin(axis=1).where(diff.min(axis=1) < 1e-6)
    switches = held[held != held.shift()].dropna().iloc[1:]  # 第一个是起点，不是换月

    print("📊 2024 年螺纹钢主力合约切换：")
    rows = []
    for day, new in switches.items():
        i = main.index.get_loc(day)
        prev_day = main.index[i - 1]
        old = held.loc[prev_day]
        main_ret = main.loc[day] / main.loc[prev_day] - 1
        real_ret = contracts.loc[day, new] / contracts.loc[prev_day, new] - 1  # 前一天收盘已换到新合约
        rows.append([
            day.date(), f"{old}→{new}", contracts.loc[prev_day, old], contracts.loc[prev_day, new],
            f"{main_ret:+.2%}", f"{real_ret:+.2%}",
        ])
    cols = ["切换日", "合约", "前日旧合约价", "前日新合约价", "主力连续收益", "真实持仓收益"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    naive = main.iloc[-1] / main.iloc[0] - 1
    print(f"""
    🔍 发现：切换日的"主力连续收益"包含了新旧合约的价差，并不是真实可获得的收益。
       2024 年主力连续首尾涨跌：{naive:+.2%}（含换月价差，不能直接当作策略基准）

    📌 解决方法（Phase 5 详细实现）：
       - 价差调整（加减法）：把历史价格整体平移换月价差
       - 比例调整（等比）：把历史价格整体乘以新旧合约价格比 —— 与股票等比复权思想相同
       - 回测时直接用单个合约数据 + 显式的换月交易，最真实
    """)


# ==================== 第三部分：加密货币数据 ====================


def fetch_crypto(symbol: str = "BTC/USDT", timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
    """用 ccxt 获取加密货币 K 线；失败时返回合成数据"""
    if not is_offline():
        try:
            import ccxt

            exchange_id = os.getenv("CCXT_EXCHANGE", "okx")
            proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            config = {"timeout": 15000}
            if proxy:
                config["proxies"] = {"http": proxy, "https": proxy}
            ex = getattr(ccxt, exchange_id)(config)
            bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            print(f"✅ 从 {exchange_id} 获取 {symbol} {timeframe} K 线 {len(df)} 根")
            return df.set_index("date").drop(columns="ts")
        except Exception as e:
            print(f"⚠️ ccxt 获取失败（{type(e).__name__}: {str(e)[:80]}），使用合成数据")
            print("   提示：在 .env 中设置 HTTPS_PROXY=http://127.0.0.1:端口")
    else:
        print("⚠️ 离线模式：加密货币使用合成数据")
    end = pd.Timestamp.today().normalize()
    df = synthetic_ohlcv(symbol, (end - pd.Timedelta(days=limit)).strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                         start_price=60000, annual_vol=0.6)
    return df[["open", "high", "low", "close", "volume"]]


def crypto_data():
    print("\n" + "=" * 60)
    print("第三部分：加密货币数据（ccxt）")
    print("=" * 60)

    print("""
    ccxt 统一接口：换交易所只需改一个名字
        exchange = ccxt.okx()            # 或 ccxt.binance()
        exchange.fetch_ohlcv("BTC/USDT", "1d", limit=365)
        → [[时间戳(毫秒), 开, 高, 低, 收, 量], ...]
    公开行情接口不需要 API Key。
    """)

    btc = fetch_crypto("BTC/USDT", "1d", 365)
    print(btc.tail(3).round(2).to_string())

    gaps = btc.index.to_series().diff().dt.days.value_counts()
    print(f"\n📌 相邻 K 线间隔天数分布：{ {int(k): int(v) for k, v in gaps.items()} } → 7×24 小时交易，没有周末和节假日")
    print("   ⚠️ 加密货币 K 线时间是 UTC，日线在北京时间早上 8 点切换")
    print("   ⚠️ 最后一根 K 线通常还没收盘（成交量偏小），研究时应丢弃：btc.iloc[:-1]")
    print("   ⚠️ 交易所单次返回条数有上限（如 OKX 约 300 根），更长历史需要用 since 参数分页获取")
    return btc.iloc[:-1]


# ==================== 第四部分：三个市场对比 ====================


def market_comparison(btc: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：A 股 / 期货 / 加密货币对比")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
    │              │ A 股             │ 国内期货         │ 加密货币         │
    ├──────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 交易时间     │ 工作日 4 小时    │ 日盘 + 夜盘      │ 7×24 小时        │
    │ T+0 / T+1    │ T+1              │ T+0              │ T+0              │
    │ 做空         │ 融券（受限）     │ 自由做空         │ 合约做空         │
    │ 杠杆         │ 无（融资除外）   │ 约 5-15 倍       │ 可很高（高风险） │
    │ 涨跌停       │ 有               │ 有               │ 无               │
    │ 年化天数     │ ≈ 252            │ ≈ 252            │ 365              │
    │ 主要数据难点 │ 复权、停牌、退市 │ 换月、夜盘归属   │ 交易所差异、时区 │
    └──────────────┴──────────────────┴──────────────────┴──────────────────┘
    """)

    hs300 = get_index_daily("000300", "20240101", "20241231")["close"]
    vol_hs300 = hs300.pct_change().std() * np.sqrt(252)
    btc_ret = btc["close"].pct_change().dropna()
    vol_btc_365 = btc_ret.std() * np.sqrt(365)
    vol_btc_252 = btc_ret.std() * np.sqrt(252)
    print(f"📊 年化波动率：沪深300 {vol_hs300:.1%}，BTC {vol_btc_365:.1%}（√365）")
    print(f"   如果 BTC 错误地用 √252 年化：{vol_btc_252:.1%} —— 低估了 {1 - vol_btc_252 / vol_btc_365:.0%}")
    print("""
    📌 夜盘归属：期货夜盘（如 21:00-23:00）属于"下一个交易日"。
       周五夜盘的数据归属于下周一 —— 合并 A 股和期货数据时要特别注意对齐。
    """)


# ==================== 主函数 ====================


def main():
    futures_basics()
    main_contract_rollover()
    btc = crypto_data()
    market_comparison(btc)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 计算：以 3500 元/吨买入 2 手螺纹钢，保证金 10%，价格跌到 3300 时亏损多少？
       亏损占保证金的比例是多少？
    2. 用第二部分的数据，实现等比调整的螺纹钢连续价格，并与主力连续比较全年收益
    3. 用 ccxt 获取 ETH/USDT 的 1 小时 K 线，计算日内各小时的平均成交量（UTC 与北京时间）
    4. 获取沪深 300 股指期货主力连续（IF0），与沪深 300 指数对比，观察基差
    """)


if __name__ == "__main__":
    main()
