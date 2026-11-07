"""
AKShare 数据获取
================

学习目标：
    1. 了解国内常用金融数据源及其取舍
    2. 用 AKShare 获取股票列表、个股日线、指数、ETF、实时行情、指数成分股
    3. 理解原始数据的字段与单位，并统一成标准格式

核心概念：
    - AKShare：开源免费的金融数据接口库，数据来自东方财富、新浪等公开网站
    - OHLCV：开盘、最高、最低、收盘、成交量
    - 数据标准化：不同数据源的列名、单位、日期格式不同，入库前必须统一

前置知识：
    - 02-a-share-market-rules.py

环境要求：
    - pip install akshare pandas
    - 需要网络；无网络时设置 QTC_OFFLINE=1，涉及 AKShare 直连的部分会跳过
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from utils.data import get_etf_daily, get_index_daily, get_stock_daily, is_offline

load_dotenv()
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)


def safe_call(fn, desc: str):
    """调用 AKShare 接口；离线或失败时返回 None 并打印原因"""
    if is_offline():
        print(f"⏭️  离线模式，跳过：{desc}")
        return None
    try:
        return fn()
    except Exception as e:
        print(f"⚠️ {desc} 失败：{type(e).__name__}: {str(e)[:100]}")
        print("   可尝试 pip install -U akshare，或稍后重试")
        return None


# ==================== 第一部分：数据源概览 ====================


def data_sources_overview():
    """国内常用金融数据源"""
    print("=" * 60)
    print("第一部分：数据源概览")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────┬────────────────────┬──────────────────────┐
    │ 数据源       │ 费用     │ 优点               │ 缺点                 │
    ├──────────────┼──────────┼────────────────────┼──────────────────────┤
    │ AKShare      │ 免费     │ 覆盖广、无需注册   │ 爬虫接口，可能变动   │
    │ Tushare Pro  │ 积分制   │ 接口稳定、字段规范 │ 高级数据需积分       │
    │ BaoStock     │ 免费     │ 历史日线稳定       │ 品种和字段较少       │
    │ 聚宽/米筐    │ 付费     │ 数据干净、带回测   │ 费用、平台绑定       │
    │ Wind/Choice  │ 昂贵     │ 机构级、最全面     │ 个人难以负担         │
    └──────────────┴──────────┴────────────────────┴──────────────────────┘

    💡 本课程主用 AKShare（免费、够用），Tushare 作为补充和交叉验证。
       个人研究的关键不是"最贵的数据"，而是"知道你的数据有什么缺陷"。
    """)


# ==================== 第二部分：股票列表 ====================


def stock_list():
    """获取全部 A 股列表并按板块统计"""
    print("\n" + "=" * 60)
    print("第二部分：A 股股票列表")
    print("=" * 60)

    import akshare as ak

    df = safe_call(ak.stock_info_a_code_name, "获取股票列表")
    if df is None:
        return

    def board_of(code: str) -> str:
        if code.startswith(("688", "689")):
            return "科创板"
        if code.startswith(("300", "301")):
            return "创业板"
        if code.startswith(("8", "4", "92")):
            return "北交所"
        if code.startswith("6"):
            return "沪市主板"
        return "深市主板"

    df["board"] = df["code"].map(board_of)
    print(f"\n📋 当前 A 股共 {len(df)} 只，前 5 行：")
    print(df.head().to_string(index=False))
    print("\n📊 按板块统计：")
    print(df["board"].value_counts().to_string())
    st = df[df["name"].str.contains("ST")]
    print(f"\n⚠️ 其中名称含 ST 的股票 {len(st)} 只（风险警示，选股时通常剔除）")
    print("""
    ⚠️ 注意：这是"今天"的股票列表，不包含已退市的股票。
       用它回测历史会产生幸存者偏差 —— 详见 10-data-biases-quality.py
    """)


# ==================== 第三部分：个股日线 ====================


def stock_daily():
    """个股日线：原始字段与标准化"""
    print("\n" + "=" * 60)
    print("第三部分：个股日线")
    print("=" * 60)

    import akshare as ak

    raw = safe_call(
        lambda: ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20240101", end_date="20240131"),
        "获取贵州茅台日线（原始格式）",
    )
    if raw is not None:
        print("\n📋 AKShare 原始返回（东方财富接口）：")
        print(raw.head(3).to_string(index=False))
        print("""
    字段说明：
      日期 / 开盘 / 收盘 / 最高 / 最低  —— 价格单位：元
      成交量  —— 单位：手（1 手 = 100 股）⚠️ 不同接口单位不同
      成交额  —— 单位：元
      振幅 / 涨跌幅 / 换手率 —— 单位：%（5.2 表示 5.2%，不是 0.052）
        """)

    # 课程统一入口：自动缓存、多数据源、统一格式
    df = get_stock_daily("600519", "20240101", "20241231")
    print("📋 utils.data.get_stock_daily 统一格式（成交量已换算为股）：")
    print(df.head(3).round(2).to_string())
    print(f"\n   形状：{df.shape}，索引类型：{type(df.index).__name__}")
    print(f"   2024 年收盘价区间：{df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"   日均成交额：{df['amount'].mean() / 1e8:.1f} 亿元")


# ==================== 第四部分：指数与 ETF ====================


def index_and_etf():
    """指数与 ETF：比较不同宽基指数的表现"""
    print("\n" + "=" * 60)
    print("第四部分：指数与 ETF")
    print("=" * 60)

    print("""
    指数 vs ETF：
    - 指数（如沪深 300）只是一个计算出来的数字，不能直接买
    - ETF（如 510300 沪深300ETF）是跟踪指数的基金，可以像股票一样交易
    - 指数代表"理论收益"，ETF 代表"实际可获得收益"（有管理费和跟踪误差）
    """)

    indices = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000", "399006": "创业板指"}
    rows = []
    for code, name in indices.items():
        df = get_index_daily(code, "20240101", "20241231")
        ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
        vol = df["close"].pct_change().std() * (252**0.5)
        rows.append([code, name, f"{ret:+.2%}", f"{vol:.2%}"])
    print("📊 2024 年主要指数表现：")
    print(pd.DataFrame(rows, columns=["代码", "名称", "区间收益", "年化波动"]).to_string(index=False))

    idx = get_index_daily("000300", "20240101", "20241231")["close"]
    etf = get_etf_daily("510300", "20240101", "20241231")["close"]
    both = pd.concat({"指数": idx, "ETF": etf}, axis=1).dropna()
    rets = both.iloc[-1] / both.iloc[0] - 1
    print(f"\n📌 沪深300指数 {rets['指数']:+.2%} vs 沪深300ETF(510300) {rets['ETF']:+.2%}")
    print("   ETF 未复权价格不含分红再投资；指数是价格指数也不含分红 —— 比较前要确认口径")


# ==================== 第五部分：实时行情快照 ====================


def realtime_snapshot():
    """全市场实时行情快照"""
    print("\n" + "=" * 60)
    print("第五部分：全市场行情快照")
    print("=" * 60)

    import akshare as ak

    print("⏳ 获取全市场快照（约 5000+ 只，需要十几秒）...")
    spot = safe_call(ak.stock_zh_a_spot_em, "获取实时行情")
    if spot is None:
        return

    spot = spot[["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率"]].dropna(subset=["最新价"])
    print(f"\n📊 有效报价 {len(spot)} 只（非交易时间为最近一个交易日的收盘快照）")
    up, down = (spot["涨跌幅"] > 0).sum(), (spot["涨跌幅"] < 0).sum()
    print(f"   上涨 {up} 只，下跌 {down} 只，涨跌比 {up / max(down, 1):.2f}")
    print(f"   涨幅 ≥ 9.9% 的约 {(spot['涨跌幅'] >= 9.9).sum()} 只（粗略统计，未区分板块）")

    print("\n🔝 成交额前 5：")
    top = spot.nlargest(5, "成交额").copy()
    top["成交额(亿)"] = (top["成交额"] / 1e8).round(1)
    print(top[["代码", "名称", "最新价", "涨跌幅", "成交额(亿)"]].to_string(index=False))
    print("""
    💡 市场宽度（涨跌家数比）是常用的情绪指标；成交额集中度反映资金抱团程度。
       注意：实时快照接口适合盘中监控，不适合做历史研究（没有历史）。
    """)


# ==================== 第六部分：指数成分股 ====================


def index_constituents():
    """沪深 300 成分股"""
    print("\n" + "=" * 60)
    print("第六部分：指数成分股")
    print("=" * 60)

    import akshare as ak

    cons = safe_call(lambda: ak.index_stock_cons_csindex(symbol="000300"), "获取沪深300成分股")
    if cons is None:
        return

    print(f"\n📋 沪深 300 当前成分股 {len(cons)} 只（数据日期：{cons['日期'].iloc[0]}）")
    print(cons[["成分券代码", "成分券名称", "交易所"]].head(8).to_string(index=False))
    print("\n📊 按交易所分布：")
    print(cons["交易所"].value_counts().to_string())
    print("""
    ⚠️ 指数成分每半年调整一次（6 月、12 月）。
       用"今天的成分股"回测 2018 年的沪深 300 选股策略 → 幸存者偏差：
       今天在 300 里的公司，很多是因为过去几年涨得好才被选进来的。
       正确做法：使用每个历史时点的成分股名单（Phase 7 会处理）。
    """)


# ==================== 主函数 ====================


def main():
    data_sources_overview()
    stock_list()
    stock_daily()
    index_and_etf()
    realtime_snapshot()
    index_constituents()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 获取宁德时代（300750）2020-2024 年日线，计算每年的涨跌幅
    2. 比较同一天东方财富接口（stock_zh_a_hist）与新浪接口（stock_zh_a_daily）
       的成交量，验证单位差异
    3. 统计全市场快照中各板块的平均涨跌幅
    4. 找出 5 只黄金、债券、跨境 ETF，获取其 2024 年收益，与沪深 300 比较相关性
    """)


if __name__ == "__main__":
    main()
