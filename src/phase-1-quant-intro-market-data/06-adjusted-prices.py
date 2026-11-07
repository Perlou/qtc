"""
除权除息与复权
==============

学习目标：
    1. 理解分红、送转为什么会让股价"跳水"，以及除权参考价的计算
    2. 手工计算等比复权因子，并与数据源的复权因子核对
    3. 发现"加减法复权"与"等比复权"的差异，知道回测该用哪一种
    4. 掌握"存不复权价 + 复权因子"的工程实践

核心概念：
    - 除权除息：分红送转后，交易所下调股价使总市值不变
    - 前复权（qfq）：以最新价格为基准，向前调整历史价格
    - 后复权（hfq）：以上市首日为基准，向后调整之后的价格
    - 等比复权 vs 加减法复权：调整方式不同，收益率是否准确也不同

前置知识：
    - 03-akshare-basics.py、05-ohlcv-candlestick.py

环境要求：
    - pip install akshare pandas
    - 第二部分起需要网络（离线时只运行第一部分的手算示例）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from utils.data import get_adj_factor, get_stock_daily, get_stock_daily_adj, is_offline

load_dotenv()

SYMBOL, NAME = "601088", "中国神华"  # 高分红股，复权影响明显
START, END = "20220101", "20241231"


# ==================== 第一部分：除权除息原理 ====================


def ex_rights_reference_price(prev_close: float, cash: float = 0.0, bonus_ratio: float = 0.0) -> float:
    """
    除权（息）参考价 = (前收盘价 - 每股现金红利) / (1 + 每股送转股比例)

    Args:
        prev_close: 股权登记日收盘价
        cash: 每股现金红利（元）
        bonus_ratio: 每股送股 + 转增股数量（如 10 送 3 → 0.3）
    """
    return (prev_close - cash) / (1 + bonus_ratio)


def ex_rights_principle():
    """除权除息原理与手算"""
    print("=" * 60)
    print("第一部分：除权除息原理")
    print("=" * 60)

    print("""
    🍕 比喻：一个披萨切成 8 块，每块 10 元。
       - 派息：先从披萨里拿出 1 元现金给你 → 披萨本身只值 9 元/块
       - 送转：把 8 块切成 16 块 → 每块只值 5 元，但你的块数翻倍
       你的总财富没变，但"每块价格"变了 —— 这就是除权除息。

    除权（息）参考价 = (前收盘价 - 每股现金红利) / (1 + 每股送转比例)
    """)

    cases = [
        ("10 派 10 元", 50.0, 1.0, 0.0),
        ("10 送 5 股", 50.0, 0.0, 0.5),
        ("10 转 10 派 5", 50.0, 0.5, 1.0),
    ]
    rows = []
    for name, prev, cash, ratio in cases:
        ref = ex_rights_reference_price(prev, cash, ratio)
        # 持有 1000 股的财富变化：股票市值 + 现金红利
        wealth_before = 1000 * prev
        wealth_after = 1000 * (1 + ratio) * ref + 1000 * cash
        rows.append([name, prev, round(ref, 2), f"{ref / prev - 1:+.1%}", wealth_before, round(wealth_after, 2)])
    print(pd.DataFrame(rows, columns=["方案", "前收盘", "除权参考价", "价格变化", "除权前财富", "除权后财富"]).to_string(index=False))

    print("""
    💡 关键：价格下跌了，但投资者财富没变（暂不考虑红利税）。
       如果直接用不复权价格计算收益，会把每次分红都误认为"亏损"。
    """)


# ==================== 第二部分：真实数据中的价格缺口 ====================


def load_dividends() -> pd.DataFrame | None:
    import akshare as ak

    div = ak.stock_history_dividend_detail(symbol=SYMBOL, indicator="分红")
    div = div[div["进度"] == "实施"].copy()
    div["除权除息日"] = pd.to_datetime(div["除权除息日"])
    div = div[(div["除权除息日"] >= START) & (div["除权除息日"] <= END)]
    div["每股派息"] = div["派息"].astype(float) / 10  # 原始数据为"每 10 股"
    div["每股送转"] = (div["送股"].astype(float) + div["转增"].astype(float)) / 10
    return div.sort_values("除权除息日")[["除权除息日", "每股派息", "每股送转"]]


def price_gaps(raw: pd.DataFrame, div: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"第二部分：真实数据中的价格缺口（{SYMBOL} {NAME}）")
    print("=" * 60)

    print(f"\n📋 {START[:4]}-{END[:4]} 年分红记录：")
    print(div.to_string(index=False))

    rows = []
    for _, d in div.iterrows():
        i = raw.index.get_indexer([d["除权除息日"]])[0]
        if i <= 0:
            continue
        prev_close, close = raw["close"].iloc[i - 1], raw["close"].iloc[i]
        ref = ex_rights_reference_price(prev_close, d["每股派息"], d["每股送转"])
        rows.append([
            d["除权除息日"].date(), prev_close, round(ref, 2), close,
            f"{close / prev_close - 1:+.2%}", f"{close / ref - 1:+.2%}",
        ])
    print("\n📊 除息日价格变化：")
    print(pd.DataFrame(rows, columns=["除息日", "前收盘", "参考价", "当日收盘", "不复权涨跌", "真实涨跌"]).to_string(index=False))
    print("\n💡 不复权价格在除息日大跌，但扣除分红后，投资者的真实收益其实可能是正的。")


# ==================== 第三部分：手工计算等比复权因子 ====================


def manual_adj_factor(raw: pd.DataFrame, div: pd.DataFrame) -> pd.Series:
    """
    等比复权因子：每个除权日乘以 前收盘 / 除权参考价

    因子在除权日"跳一步"，其余日子不变；后复权价 = 不复权价 × 因子
    """
    step = pd.Series(1.0, index=raw.index)
    for _, d in div.iterrows():
        i = raw.index.get_indexer([d["除权除息日"]])[0]
        if i <= 0:
            continue
        prev_close = raw["close"].iloc[i - 1]
        ref = ex_rights_reference_price(prev_close, d["每股派息"], d["每股送转"])
        step.iloc[i] = prev_close / ref
    return step.cumprod()


def verify_adj_factor(raw: pd.DataFrame, div: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：手工计算等比复权因子")
    print("=" * 60)

    manual = manual_adj_factor(raw, div)
    sina = get_adj_factor(SYMBOL)
    sina = sina.reindex(sina.index.union(raw.index)).ffill().reindex(raw.index)

    # 两者基准日不同（手工从 START 开始为 1，新浪从上市日开始为 1），比较"相对变化"
    compare = pd.DataFrame({"手工因子": manual / manual.iloc[0], "新浪因子(归一)": sina / sina.iloc[0]})
    ex_days = compare[compare["手工因子"].diff().abs() > 1e-9]
    print("📊 除权日的因子对比（归一化到起始日 = 1）：")
    print(ex_days.round(6).to_string())
    max_diff = (compare["手工因子"] - compare["新浪因子(归一)"]).abs().max()
    print(f"\n✅ 全区间最大差异：{max_diff:.2e}")
    print("   手工计算与数据源一致 → 我们真正理解了复权因子是怎么来的")


# ==================== 第四部分：加减法复权 vs 等比复权 ====================


def additive_vs_ratio(raw: pd.DataFrame, div: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：加减法复权 vs 等比复权")
    print("=" * 60)

    print("""
    两种复权方式：
    ┌────────────┬───────────────────────────────┬──────────────────────────┐
    │ 方式       │ 计算                          │ 收益率                   │
    ├────────────┼───────────────────────────────┼──────────────────────────┤
    │ 等比复权   │ 历史价格 × 比例因子           │ ✅ 与真实收益一致        │
    │ 加减法复权 │ 历史价格 ± 累计分红金额       │ ❌ 除权日附近有偏差      │
    └────────────┴───────────────────────────────┴──────────────────────────┘
    """)

    em_hfq = get_stock_daily(SYMBOL, START, END, adjust="hfq")["close"]
    em_qfq = get_stock_daily(SYMBOL, START, END, adjust="qfq")["close"]
    ratio_hfq = get_stock_daily_adj(SYMBOL, START, END)["adj_close"]

    # 检验：东方财富复权价 - 不复权价，在除息日的变化量是否正好等于分红金额？
    gap = (em_hfq - raw["close"]).diff()
    rows = []
    for _, d in div.iterrows():
        day = d["除权除息日"]
        if day not in raw.index:
            continue
        i = raw.index.get_loc(day)
        prev_close = raw["close"].iloc[i - 1]
        true_ret = raw["close"].iloc[i] / ex_rights_reference_price(prev_close, d["每股派息"], d["每股送转"]) - 1
        rows.append([
            day.date(), d["每股派息"], round(gap.loc[day], 4),
            f"{true_ret:+.3%}",
            f"{ratio_hfq.pct_change().loc[day]:+.3%}",
            f"{em_hfq.pct_change().loc[day]:+.3%}",
            f"{em_qfq.pct_change().loc[day]:+.3%}",
        ])
    cols = ["除息日", "每股派息", "东财(复权-原价)变化", "真实收益", "等比复权", "东财后复权", "东财前复权"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    print("""
    🔍 发现：
    1. "东财(复权-原价)变化"正好等于每股派息 → 东方财富接口用的是加减法复权
    2. 只有等比复权的除息日收益与真实收益一致
    3. 加减法前复权对高分红股票的长历史可能出现负价格

    ✅ 结论：计算收益用等比复权（utils.data.get_stock_daily_adj 的 adj_close）
    """)

    total = pd.Series({
        "不复权": raw["close"].iloc[-1] / raw["close"].iloc[0] - 1,
        "等比复权": ratio_hfq.iloc[-1] / ratio_hfq.iloc[0] - 1,
        "东财后复权": em_hfq.iloc[-1] / em_hfq.iloc[0] - 1,
        "东财前复权": em_qfq.iloc[-1] / em_qfq.iloc[0] - 1,
    })
    print(f"📈 {START[:4]}-{END[:4]} 区间总收益：")
    for k, v in total.items():
        print(f"   {k:<8s} {v:+.2%}")
    print(f"   不复权少算了 {total['等比复权'] - total['不复权']:.1%} 的分红收益")


# ==================== 第五部分：工程实践 ====================


def best_practice():
    print("\n" + "=" * 60)
    print("第五部分：回测中的复权最佳实践")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  存储：不复权 OHLCV  +  后复权因子（两张表，都只追加不修改）  │
    └──────────────────────────┬──────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
    ┌───────────────────┐               ┌───────────────────────┐
    │ 计算收益 / 信号   │               │ 模拟撮合              │
    │ adj_close         │               │ 不复权价格            │
    │ = close × factor  │               │ 涨跌停、整数手、金额  │
    └───────────────────┘               └───────────────────────┘

    为什么不直接存前复权价格？
    - 前复权价格每次分红后都会整体改变 → 今天跑的回测和下个月跑的结果不同
    - 前复权在 T 日看到的历史价格，包含了 T 日之后的分红信息 → 隐性未来函数
      （例如用前复权价格判断"股价是否跌破 10 元"，历史上那天的真实价格可能是 15 元）

    为什么撮合要用不复权价格？
    - 涨跌停价、最小变动价位、100 股整数手、成交金额都基于真实价格
    """)


# ==================== 主函数 ====================


def main():
    ex_rights_principle()

    if is_offline():
        print("\n⏭️  离线模式：后续部分需要真实分红数据，已跳过")
        return

    try:
        div = load_dividends()
    except Exception as e:
        print(f"\n⚠️ 获取分红数据失败（{type(e).__name__}），后续部分已跳过")
        return

    raw = get_stock_daily(SYMBOL, START, END)
    price_gaps(raw, div)
    verify_adj_factor(raw, div)
    additive_vs_ratio(raw, div)
    best_practice()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 对一只有过送转的股票（如 300750 宁德时代 2020-2023），重复第三部分的验证
    2. 实现 to_qfq(raw_close, factor)：用后复权因子生成等比前复权价格
       提示：qfq = close × factor / factor.iloc[-1]
    3. 找一只长期高分红股票，用东方财富 qfq 获取 2005 年至今数据，看看是否出现负价格
    4. 思考：如果你的数据只有前复权价格，回测"股价低于 5 元买入"会出什么问题？
    """)


if __name__ == "__main__":
    main()
