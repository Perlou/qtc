"""
A 股交易规则
============

学习目标：
    1. 掌握 A 股交易时段、T+1、涨跌停、最小交易单位等核心规则
    2. 能准确计算一笔交易的全部成本（佣金、印花税、过户费）
    3. 理解这些规则对量化策略和回测的影响

核心概念：
    - T+1：当天买入的股票，下一个交易日才能卖出
    - 涨跌停：单日价格涨跌幅限制，触及时可能无法成交
    - 交易成本：佣金（双向，有最低收费）+ 印花税（仅卖出）+ 过户费（双向）

前置知识：
    - 01-quant-trading-overview.py

环境要求：
    - pip install pandas
    - 本课不需要网络

⚠️ 费率和规则会调整，本课数值以编写时（2026 年）的公开规则为准，实盘前请核对交易所与券商最新规定。
"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd


# ==================== 第一部分：交易时段 ====================


def trading_sessions():
    """A 股交易时段"""
    print("=" * 60)
    print("第一部分：交易时段（沪深交易所）")
    print("=" * 60)

    print("""
    09:15 ──── 09:25   开盘集合竞价（09:20 后不能撤单）
    09:25 ──── 09:30   撮合结果公布，可下单但不撮合
    09:30 ──── 11:30   上午连续竞价
    11:30 ──── 13:00   午间休市
    13:00 ──── 14:57   下午连续竞价
    14:57 ──── 15:00   收盘集合竞价（不能撤单）

    📌 量化相关：
    - 日线的"开盘价"来自 09:25 集合竞价，"收盘价"来自 14:57-15:00 集合竞价
    - 回测中"以收盘价成交"意味着要在收盘集合竞价中下单，信号必须在 14:57 前算好
    - 用当日收盘价计算信号、又用同一个收盘价成交，是常见的隐性未来函数
    """)


# ==================== 第二部分：板块与涨跌停 ====================


@dataclass(frozen=True)
class BoardRule:
    name: str
    limit_pct: float  # 涨跌幅限制
    min_buy: int  # 最小买入股数
    step: int  # 买入递增单位


ST_RULE_CHANGE_DATE = "2026-07-06"  # 沪深主板 ST/*ST 涨跌幅由 5% 调整为 10% 的实施日


def get_board(symbol: str, is_st: bool = False, trade_date: str | None = None) -> BoardRule:
    """
    根据股票代码判断板块及其规则

    规则会随时间变化：回测历史时要用"当时"的规则。例如主板 ST 股在
    2026-07-06 之前涨跌幅为 5%，之后为 10%。trade_date 为空时按现行规则。
    """
    if symbol.startswith(("688", "689")):
        return BoardRule("科创板", 0.20, 200, 1)
    if symbol.startswith("30"):
        return BoardRule("创业板", 0.20, 100, 100)
    if symbol.startswith(("8", "4", "92")):
        return BoardRule("北交所", 0.30, 100, 1)
    if is_st:
        if trade_date is not None and pd.Timestamp(trade_date) < pd.Timestamp(ST_RULE_CHANGE_DATE):
            return BoardRule("主板 ST", 0.05, 100, 100)
        return BoardRule("主板 ST", 0.10, 100, 100)
    return BoardRule("主板", 0.10, 100, 100)


def round_price(x: float) -> float:
    """价格四舍五入到分（交易所规则是四舍五入，而不是 Python round 的银行家舍入）"""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_price_exact(prev_close: float, pct: float) -> float:
    """用 Decimal 从原始价格开始计算 prev_close × (1 + pct)，再四舍五入到分，避免浮点误差"""
    value = Decimal(str(prev_close)) * (Decimal(1) + Decimal(str(pct)))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def limit_prices(prev_close: float, limit_pct: float) -> tuple[float, float]:
    """计算涨停价与跌停价"""
    return round_price_exact(prev_close, limit_pct), round_price_exact(prev_close, -limit_pct)


def price_limits_demo():
    """涨跌停规则"""
    print("\n" + "=" * 60)
    print("第二部分：板块与涨跌停")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────┬──────────────────────────────────┐
    │ 板块         │ 涨跌幅   │ 代码特征                         │
    ├──────────────┼──────────┼──────────────────────────────────┤
    │ 沪市主板     │ ±10%     │ 600 / 601 / 603 / 605            │
    │ 深市主板     │ ±10%     │ 000 / 001 / 002 / 003            │
    │ 创业板       │ ±20%     │ 300 / 301 / 302                  │
    │ 科创板       │ ±20%     │ 688 / 689                        │
    │ 北交所       │ ±30%     │ 8 / 4 / 92 开头                  │
    └──────────────┴──────────┴──────────────────────────────────┘
    注册制新股上市前 5 个交易日不设涨跌幅限制（北交所为首日），之后按上表执行。
    主板 ST / *ST 股：2026-07-06 起由 ±5% 调整为 ±10%（回测 2026 年 7 月之前的数据要用 ±5%）。
    """)

    examples = [("600519", 1500.00), ("300750", 200.00), ("688981", 85.55), ("000001", 7.55)]
    rows = []
    for symbol, prev in examples:
        board = get_board(symbol)
        up, down = limit_prices(prev, board.limit_pct)
        rows.append([symbol, board.name, prev, up, down])
    print(pd.DataFrame(rows, columns=["代码", "板块", "前收盘", "涨停价", "跌停价"]).to_string(index=False))

    print(f"""
    🔍 四舍五入细节：前收盘 7.55 的主板股票，涨停价 = 7.55 × 1.1 = 8.305 → 应为 8.31
       浮点计算 7.55 * 1.1 = {7.55 * 1.1!r}（二进制无法精确表示，实际略小于 8.305）
       Python round(7.55 * 1.1, 2) = {round(7.55 * 1.1, 2)}  ❌
       Decimal 精确计算 + 四舍五入 = {round_price_exact(7.55, 0.10)}  ✅
       差 1 分钱就会把涨停误判为"未涨停" —— 回测里要用 Decimal 从原始价格开始计算

    📌 量化相关：
    - 涨停时往往只有卖单排队吃不到 → 回测里"涨停买入"通常不应假设成交
    - 跌停时可能卖不出 → 风险控制不能假设随时能止损
    - 判断涨停：close >= 涨停价（注意用四舍五入后的价格比较）
    """)


# ==================== 第三部分：T+1 与交易单位 ====================


def t1_and_lot_size():
    """T+1 与最小交易单位"""
    print("\n" + "=" * 60)
    print("第三部分：T+1 与交易单位")
    print("=" * 60)

    print("""
    📌 T+1：今天买入的股票，下一个交易日才能卖出
       - 股票：T+1
       - 部分 ETF（跨境 ETF、债券 ETF、黄金 ETF、货币 ETF）：T+0
       - 期货、加密货币：T+0，可以当天开平仓

    📌 交易单位：
       - 主板 / 创业板：买入必须是 100 股（1 手）的整数倍
       - 科创板：最少买 200 股，超出部分可以 1 股递增
       - 卖出：可以一次性卖出不足 1 手的零股
       - 价格最小变动单位：0.01 元
    """)

    def max_buy_shares(cash: float, price: float, board: BoardRule) -> int:
        """给定资金和价格，计算最多能买多少股（未考虑费用）"""
        raw = int(cash // price)
        if raw < board.min_buy:
            return 0
        return board.min_buy + (raw - board.min_buy) // board.step * board.step

    cases = [(100_000, "600519", 1500.0), (100_000, "000001", 10.85), (50_000, "688981", 85.55), (20_000, "300750", 200.0)]
    rows = []
    for cash, symbol, price in cases:
        board = get_board(symbol)
        shares = max_buy_shares(cash, price, board)
        rows.append([f"{cash:,}", symbol, board.name, price, shares, f"{shares * price:,.0f}", f"{1 - shares * price / cash:.1%}"])
    print(pd.DataFrame(rows, columns=["资金", "代码", "板块", "价格", "可买股数", "买入金额", "闲置资金"]).to_string(index=False))

    print("""
    💡 10 万元买不了 1 手茅台（需 15 万）—— 这就是为什么小资金很难精确执行
       多因子组合的目标权重，回测时必须做整数手取整。
    """)


# ==================== 第四部分：交易成本 ====================


@dataclass
class FeeSchedule:
    commission_rate: float = 0.00025  # 佣金：双向，券商协商，常见万 1 ~ 万 3
    min_commission: float = 5.0  # 最低佣金 5 元（部分券商"免五"）
    stamp_duty_rate: float = 0.0005  # 印花税：仅卖出，2023-08-28 起为 0.05%
    transfer_fee_rate: float = 0.00001  # 过户费：双向，0.001%（沪深）


def trade_cost(price: float, shares: int, side: str, fees: FeeSchedule = FeeSchedule()) -> dict:
    """计算一笔 A 股交易的成本明细"""
    amount = price * shares
    commission = max(amount * fees.commission_rate, fees.min_commission)
    stamp = amount * fees.stamp_duty_rate if side == "sell" else 0.0
    transfer = amount * fees.transfer_fee_rate
    total = commission + stamp + transfer
    return {
        "成交金额": round(amount, 2),
        "佣金": round(commission, 2),
        "印花税": round(stamp, 2),
        "过户费": round(transfer, 2),
        "合计": round(total, 2),
        "费率": total / amount,
    }


def transaction_costs_demo():
    """交易成本计算"""
    print("\n" + "=" * 60)
    print("第四部分：交易成本")
    print("=" * 60)

    print("""
    ┌──────────┬──────────┬───────────────────────────────┐
    │ 费用     │ 方向     │ 费率                          │
    ├──────────┼──────────┼───────────────────────────────┤
    │ 佣金     │ 买 + 卖  │ 万 1 ~ 万 3，最低 5 元        │
    │ 印花税   │ 仅卖出   │ 0.05%（2023-08-28 起）        │
    │ 过户费   │ 买 + 卖  │ 0.001%                        │
    └──────────┴──────────┴───────────────────────────────┘
    注：交易所经手费、证管费一般已包含在佣金中。
    """)

    # 不同金额下的成本：最低 5 元佣金对小额交易影响很大
    rows = []
    for amount in [2_000, 10_000, 20_000, 100_000, 1_000_000]:
        price = 10.0
        shares = amount // price // 100 * 100
        buy = trade_cost(price, shares, "buy")
        sell = trade_cost(price, shares, "sell")
        round_trip = (buy["合计"] + sell["合计"]) / buy["成交金额"]
        rows.append([f"{amount:,}", f"{buy['费率']:.4%}", f"{sell['费率']:.4%}", f"{round_trip:.4%}"])
    print(pd.DataFrame(rows, columns=["交易金额", "买入费率", "卖出费率", "买卖一轮"]).to_string(index=False))

    print("""
    💡 观察：
    - 2 万元以下交易，最低 5 元佣金让实际费率远超万 2.5
    - 一买一卖至少约 0.1%（印花税 0.05% + 双向佣金），还没算滑点
    - 一个每周换仓一次的策略，一年 52 轮 × 0.1% ≈ 5.2% 的成本拖累
    """)

    # 盈亏平衡：价格需要上涨多少才能覆盖成本
    price, shares = 20.0, 1000
    buy = trade_cost(price, shares, "buy")
    breakeven = price
    while True:
        sell = trade_cost(breakeven, shares, "sell")
        pnl = (breakeven - price) * shares - buy["合计"] - sell["合计"]
        if pnl >= 0:
            break
        breakeven = round_price(breakeven + 0.01)
    print(f"📌 以 {price} 元买入 {shares} 股，至少涨到 {breakeven} 元（+{breakeven / price - 1:.2%}）卖出才不亏")


# ==================== 第五部分：规则对回测的影响 ====================


def rules_impact_on_backtest():
    """交易规则对回测的影响清单"""
    print("\n" + "=" * 60)
    print("第五部分：交易规则对回测的影响")
    print("=" * 60)

    print("""
    ┌────────────────┬───────────────────────────────┬──────────────────────────┐
    │ 规则           │ 回测中的错误做法              │ 正确处理                 │
    ├────────────────┼───────────────────────────────┼──────────────────────────┤
    │ T+1            │ 当天买入当天止损卖出          │ 记录可卖数量，次日可卖   │
    │ 涨停           │ 涨停价照常买入                │ 涨停（一字板）不能买入   │
    │ 跌停           │ 跌停价照常卖出                │ 跌停不能卖出，顺延       │
    │ 停牌           │ 停牌日用前收盘价成交          │ 停牌日不能交易           │
    │ 整数手         │ 按任意股数买入                │ 向下取整到 100 股        │
    │ 最低佣金       │ 按固定费率计算                │ max(金额×费率, 5)        │
    │ 印花税         │ 买卖都收 / 都不收             │ 只在卖出时收             │
    └────────────────┴───────────────────────────────┴──────────────────────────┘

    这些规则会在 Phase 4（回测系统）中逐一实现。
    """)


# ==================== 主函数 ====================


def main():
    trading_sessions()
    price_limits_demo()
    t1_and_lot_size()
    transaction_costs_demo()
    rules_impact_on_backtest()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 计算：以 35.67 元买入 300 股某创业板股票，次日以 38.20 元卖出，净收益多少？
    2. 实现 is_limit_up(close, prev_close, symbol) 函数，并用 Decimal 处理舍入
    3. 如果你的券商佣金是万 1 免五，重新计算 2 万元交易的买卖一轮费率
    4. 思考：为什么 A 股的 T+1 规则会让"日内反转"策略难以直接在股票上实现？
       哪些品种可以替代？
    """)


if __name__ == "__main__":
    main()
