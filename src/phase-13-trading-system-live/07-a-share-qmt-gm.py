"""
A 股程序化接口：QMT / 掘金
==========================

学习目标：
    1. 了解个人和中小机构做 A 股程序化交易的几类接入方式
    2. 读懂 miniQMT（xtquant）和掘金（gm）的接入示意代码，知道回调如何映射到统一网关
    3. 统一各接口的标的代码与订单状态，体会网关层的价值
    4. 在 MockExchange 上复现 A 股实盘规则：100 股整数倍、涨跌停、T+1、集合竞价不能撤单
    5. 了解程序化交易报告制度（以监管与交易所最新规定为准）

核心概念：
    - QMT / miniQMT：券商提供的程序化交易终端；miniQMT 允许外部 Python 进程通过 xtquant 调用
    - 掘金量化：研究、仿真、实盘一体的平台，提供 gm SDK
    - 先报告、后交易：程序化交易需要按规定通过券商向交易所报告

前置知识：
    - 01、03、04 课
    - Phase 1 第 02 课（A 股交易规则）
    - 本阶段 CONCEPT.md 第 8、13.5 节

环境要求：
    - 不安装 xtquant / gm，示意代码只做语法检查，不执行
    - 接口细节以券商提供的版本文档和掘金官方文档为准
    - 接口的开通条件（资产门槛、费用等）由各券商自行规定且会调整，本课不做任何承诺
"""

import ast
import importlib.util
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import OMS, Direction, MockExchange, OrderRequest, RiskConfig, Status

load_dotenv()


# ==================== 第一部分：接入方式概览 ====================


def access_overview():
    print("=" * 60)
    print("第一部分：A 股程序化接入方式概览")
    print("=" * 60)
    print(pd.DataFrame(
        [
            ["本地终端 + 内置 Python", "QMT", "策略在券商定制终端中运行", "行情、回测、交易一体"],
            ["本地终端 + 外部 Python", "miniQMT + xtquant", "终端常驻，外部进程调用库", "可用自己的 Python 环境与第三方库"],
            ["券商云端托管", "PTrade", "策略上传到券商机房服务器", "无需本地常驻，但环境受限"],
            ["量化平台 + SDK", "掘金量化（gm）", "终端 + Python SDK", "回测、仿真、实盘切换"],
        ],
        columns=["类型", "代表产品", "运行方式", "特点"],
    ).to_string(index=False))
    installed = {m: importlib.util.find_spec(m) is not None for m in ["xtquant", "gm"]}
    print(f"\n   本机安装情况：{installed}（本课不需要安装）")
    print("""
    ⚠️ 重要说明：
    - 个人投资者一般通过券商提供的程序化终端 / 接口交易，而不是直连交易所
    - 这些接口需要向券商申请开通；支持哪些产品、开通条件、是否收费，各券商不同且会调整，
      以券商最新政策为准，本课不做任何承诺
    - 开通后同样要履行程序化交易报告义务（第六部分）
    """)


# ==================== 第二部分：miniQMT / xtquant（示意） ====================

XTQUANT_DEMO = '''
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount

class Callback(XtQuantTraderCallback):
    def on_stock_order(self, order):            # 委托回报
        print("委托:", order.order_id, order.order_status)
    def on_stock_trade(self, trade):            # 成交回报
        print("成交:", trade.stock_code, trade.traded_volume, trade.traded_price)
    def on_disconnected(self):
        print("与终端断开连接 —— 触发告警与重连逻辑")

trader = XtQuantTrader(r"<miniQMT 安装目录下的 userdata_mini 路径>", 123456)  # 第二个参数为会话 ID
trader.register_callback(Callback())
trader.start()
assert trader.connect() == 0, "连接终端失败"
acc = StockAccount("<资金账号>")
trader.subscribe(acc)
order_id = trader.order_stock(acc, "600000.SH", xtconstant.STOCK_BUY, 100,
                              xtconstant.FIX_PRICE, 10.50, "demo_strategy", "S1-0001")
'''

GM_DEMO = '''
from gm.api import (set_token, order_volume, OrderSide_Buy, OrderType_Limit,
                    PositionEffect_Open)

set_token("<你的 token>")
# 掘金的标的代码格式为 "交易所.代码"，如 SHSE.600000、SZSE.000001
order_volume(symbol="SHSE.600000", volume=100, side=OrderSide_Buy,
             order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=10.50)
'''


def show_demo(title: str, code: str):
    print(f"\n📄 {title}（示意代码，不执行）：")
    print(code)
    ast.parse(code)
    print("   ✅ 语法检查通过（ast.parse）")


def xtquant_and_gm():
    print("\n" + "=" * 60)
    print("第二部分：miniQMT（xtquant）与掘金（gm）接入示意")
    print("=" * 60)
    print("""
    xtquant 分为行情模块 xtdata 和交易模块 xttrader。以下代码仅示意调用结构，
    类名、参数与常量以券商提供的版本文档为准。
    """)
    show_demo("miniQMT + xtquant", XTQUANT_DEMO)
    print("""
    📌 读代码要点：
    - 回调类（Callback）里的委托回报 / 成交回报 / 断线通知，正对应网关的 on_order / on_trade / 连接事件
    - order_stock 的最后一个参数可以携带自定义备注（示例里放了 client_order_id），便于回报对应到本地订单
    - 终端需要常驻并登录，外部 Python 进程通过本地连接调用 —— 终端本身的存活也要纳入监控
    """)
    show_demo("掘金 gm SDK", GM_DEMO)
    print("""
    📌 掘金的策略通常组织为 init(context) + 事件回调（on_bar、on_order_status 等）+ run(...)，
       运行模式可在回测、仿真和实盘之间切换，具体用法以官方文档为准。
    """)


# ==================== 第三部分：网关统一：代码与状态 ====================

_QMT_EX = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}
_GM_EX = {"SHSE": "SSE", "SZSE": "SZSE"}


def to_internal(symbol: str, source: str) -> str:
    """各接口的代码格式 → 内部统一格式 <代码>.<交易所>"""
    if source == "qmt":  # 600000.SH
        code, ex = symbol.split(".")
        return f"{code}.{_QMT_EX[ex]}"
    if source == "gm":  # SHSE.600000
        ex, code = symbol.split(".")
        return f"{code}.{_GM_EX[ex]}"
    raise ValueError(f"未知来源 {source}")


def from_internal(symbol: str, target: str) -> str:
    code, ex = symbol.split(".")
    if target == "qmt":
        return f"{code}.{ {v: k for k, v in _QMT_EX.items()}[ex] }"
    if target == "gm":
        return f"{ {v: k for k, v in _GM_EX.items()}[ex] }.{code}"
    raise ValueError(f"未知目标 {target}")


def unify():
    print("\n" + "=" * 60)
    print("第三部分：网关统一 —— 代码格式与订单状态")
    print("=" * 60)
    rows = []
    for sym in ["600000.SSE", "000001.SZSE"]:
        q, g = from_internal(sym, "qmt"), from_internal(sym, "gm")
        rows.append([sym, q, g, to_internal(q, "qmt") == sym and to_internal(g, "gm") == sym])
    print(pd.DataFrame(rows, columns=["内部格式", "QMT 格式", "掘金格式", "往返转换一致"]).to_string(index=False))
    print("""
    订单状态也要统一（左侧为券商终端里常见的中文状态描述，具体枚举值以接口文档为准）：
      未报 / 待报        → SUBMITTING        已报            → NOT_TRADED
      部成              → PART_TRADED       已成            → ALL_TRADED
      已撤 / 部撤        → CANCELLED（部撤 = 部分成交后撤单，traded > 0）
      废单              → REJECTED
      已报待撤 / 部成待撤 → 状态不变，只在 OMS 里标记"撤单中"（cancel_pending）

    💡 有了统一模型，策略、风控、OMS 完全不知道底层是 QMT、掘金还是 CTP；
       换券商只需要换网关。这就是第 01 课"网关是防腐层"在 A 股场景的落地。
    """)


# ==================== 第四部分：A 股实盘规则（MockExchange） ====================

AUCTION_NO_CANCEL = [(time(9, 20), time(9, 25)), (time(14, 57), time(15, 0))]


def cancel_allowed(t: time) -> bool:
    """集合竞价的部分时段不接受撤单（以交易所规则为准）"""
    return not any(start <= t < end for start, end in AUCTION_NO_CANCEL)


def price_cage_ok(direction: Direction, price: float, ref: float, tick: float = 0.01) -> bool:
    """
    价格笼子（示意参数，以交易所最新规则为准）：
    买入申报价不高于基准价的 102% 与基准价上浮 10 个最小价位中的较高者；卖出对称。
    超出笼子的申报不会立即参与撮合。
    """
    if direction is Direction.LONG:
        return price <= max(ref * 1.02, ref + 10 * tick) + 1e-9
    return price >= min(ref * 0.98, ref - 10 * tick) - 1e-9


def a_share_rules():
    print("\n" + "=" * 60)
    print("第四部分：A 股实盘规则（MockExchange 复现）")
    print("=" * 60)
    sym = "600000.SSE"
    ex, oms = MockExchange(), OMS()
    ex.add_listener(lambda k, d: oms.on_order(d) if k == "order" else oms.on_trade(d) if k == "trade" else None)
    rows = []

    def send(label, direction, price, vol):
        req = OrderRequest(sym, direction, price, vol, client_order_id=f"A-{len(rows) + 1}")
        oms.create_order(req)
        o = ex.submit(req)
        rows.append([label, o.status.value, o.reason or "-"])

    ex.set_market(sym, 10.00, limit_up=11.00, limit_down=9.00)
    send("买入 150 股", Direction.LONG, 10.01, 150)
    send("买入 100 股", Direction.LONG, 10.01, 100)
    send("当天卖出刚买的 100 股", Direction.SHORT, 9.99, 100)
    send("以 11.50 买入（高于涨停）", Direction.LONG, 11.50, 100)
    ex.set_market(sym, 11.00, bid=11.00, ask=0, limit_up=11.00, limit_down=9.00)  # 涨停：卖盘为空
    send("涨停时以 11.00 买入", Direction.LONG, 11.00, 100)
    print(pd.DataFrame(rows, columns=["操作", "结果", "原因"]).to_string(index=False))
    print("   → 涨停时委托合法但没有卖盘，只能排队；回测里假设'涨停价成交'会高估收益")

    print("\n   集合竞价期间能否撤单（以交易所规则为准）：")
    print("   " + "，".join(f"{t:%H:%M} {'可撤' if cancel_allowed(t) else '不可撤'}"
                           for t in [time(9, 18), time(9, 22), time(10, 0), time(14, 58)]))

    ref = 10.00
    print(f"\n   价格笼子示意（基准价 {ref:.2f}）：")
    print("   " + "，".join(f"买 {p:.2f} {'✓' if price_cage_ok(Direction.LONG, p, ref) else '✗ 暂不撮合'}"
                           for p in [10.15, 10.20, 10.25]))
    low = 2.00
    upper = max(low * 1.02, low + 10 * 0.01)
    print(f"   低价股（基准价 {low:.2f}）：2% 对应 {low * 0.02:.2f} 元，10 个最小价位对应 0.10 元，"
          f"上限取较高者 {upper:.2f}；买 2.10 → {'✓' if price_cage_ok(Direction.LONG, 2.10, low) else '✗'}")
    print("""
    ┌──────────────┬──────────────────────────────────────────────────┐
    │ 规则         │ 对系统的影响                                     │
    ├──────────────┼──────────────────────────────────────────────────┤
    │ T+1          │ 持仓区分总量与可用量；当日买入不可当日卖出        │
    │ 100 股整数倍 │ 买入数量取整（科创板等另有规定）                  │
    │ 涨跌停       │ 涨停买不进、跌停卖不出；订单可能长时间挂着        │
    │ 价格笼子     │ 偏离基准价过多的限价单不会立即参与撮合            │
    │ 停牌         │ 无法交易，没有行情 → 风控"无参考价"规则会拒单     │
    │ 除权除息     │ 持仓数量与成本价变化，对账时要识别（第 10 课）    │
    │ 集合竞价     │ 部分时段不能撤单，撤单逻辑要知道当前时段          │
    └──────────────┴──────────────────────────────────────────────────┘
    """)


# ==================== 第五部分：程序化交易报告制度 ====================


def compliance():
    print("\n" + "=" * 60)
    print("第五部分：程序化交易报告制度（以监管与交易所最新规定为准）")
    print("=" * 60)
    print("""
    核心原则："先报告、后交易"。

    - 证监会《证券市场程序化交易管理规定（试行）》（2024 年发布并于同年 10 月起施行）
      确立了证券市场程序化交易的报告、交易监测与监管框架，对高频交易有更严格的要求
    - 沪深北交易所的程序化交易管理实施细则，对报告内容、流程、异常交易监控、
      高频交易认定标准与差异化管理等作出具体规定
    - 投资者一般通过其委托的证券公司向交易所报告；券商负责报告信息核查、接入管理和日常监控
    - 期货市场同样建立了程序化交易报告与管理制度，并与穿透式监管配合实施（第 06 课）

    ⚠️ 具体的报告字段、阈值、流程和适用范围会随监管更新而调整，
       以证监会、交易所与券商的最新规定为准；上线前务必通过券商确认报告义务已完成。
    """)
    cfg = RiskConfig()
    rate = cfg.max_orders_per_window / cfg.order_window_sec
    print("📋 报告信息准备清单（概述性条目，实际字段以券商提供的表单为准）：")
    print(pd.DataFrame(
        [
            ["账户基本情况", "资金账号、账户类型", "人工填写"],
            ["资金规模", "计划投入的资金量", "与第 12 课的分阶段放量计划一致"],
            ["策略类型", "如日内 / 隔夜、趋势 / 套利等", "人工填写"],
            ["交易指令执行方式", "程序自动下单 / 人工确认", "与系统设计一致"],
            ["最高申报速率", f"风控配置：每 {cfg.order_window_sec:g} 秒 {cfg.max_orders_per_window} 笔 ≈ {rate:g} 笔/秒",
             "报告值应与风控硬限制一致"],
            ["交易软件", "使用的终端 / 接口名称与版本", "如 miniQMT、掘金"],
        ],
        columns=["条目", "内容", "来源 / 备注"],
    ).to_string(index=False))
    print("""
    💡 工程上的对应关系：报告中的"最高申报速率"不是随便填的数字，
       它应当是风控引擎里报单频率规则的硬上限 —— 系统保证永远不会超过报告值。

    此外，无论是否属于程序化交易，都要避免交易所重点监控的异常交易行为：
    频繁申报撤单、虚假申报、自买自卖（对倒）、拉抬打压价格等。
    第 04 课的报单频率规则（以及该课练习中的撤单次数、自成交检查）正是为此服务的。
    """)


# ==================== 主函数 ====================


def main():
    access_overview()
    xtquant_and_gm()
    unify()
    a_share_rules()
    compliance()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 仿照 MockGateway 写一个 QmtGateway 骨架：把示意代码中的回调转换为 OrderData / TradeData，
       并用一个假的 trader 对象（只记录调用参数）完成单元测试
    2. 给 MockExchange 加上"科创板"规则：688 开头的股票买入至少 200 股、以 1 股为单位递增
    3. 给第 04 课的 RiskEngine 增加"集合竞价不可撤单时段"检查，在这些时段拒绝撤单请求
    4. 查阅你所在券商关于程序化交易报告的最新说明，列出需要准备的材料（不要依赖本课内容）
    """)


if __name__ == "__main__":
    main()
