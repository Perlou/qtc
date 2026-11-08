"""
OMS 与订单状态机
================

学习目标：
    1. 掌握订单生命周期：提交中 → 已报 → 部分成交 → 全部成交 / 已撤 / 拒单
    2. 用"状态转换表"实现状态机，非法转换必须报错
    3. 处理真实世界的回报问题：重复回报、乱序回报、撤单竞态、迟到成交
    4. 理解为什么持仓必须由成交（Trade）推导，以及 A 股 T+1 的可用数量

核心概念：
    - 状态转换表：ALLOWED_TRANSITIONS[当前状态] = 允许到达的状态集合
    - 终态：ALL_TRADED / CANCELLED / REJECTED，进入后不再变化
    - 成交去重：按交易所成交编号 trade_id 去重，持仓更新才是幂等的
    - 冻结：卖出 / 平仓委托挂着时，对应持仓被冻结，防止重复卖出

前置知识：
    - 01、02 两课
    - 本阶段 CONCEPT.md 第 4 节

环境要求：
    - 不需要网络；全部场景在 MockExchange 上同步复现
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import (
    ALLOWED_TRANSITIONS,
    OMS,
    Direction,
    InvalidTransition,
    MockExchange,
    Offset,
    OrderData,
    OrderIdGenerator,
    OrderRequest,
    PositionBook,
    Status,
    TradeData,
    check_transition,
)

load_dotenv()

SYM = "600000.SSE"


def wire(exchange: MockExchange, oms: OMS, log: list | None = None) -> None:
    """把交易所回报接到 OMS（同步版，相当于网关 + 事件引擎的最简替身）"""

    def on_report(kind, data):
        if kind == "order":
            oms.on_order(data)
            if log is not None:
                log.append(f"order  {data.status.value:<5} traded={data.traded:g}")
        elif kind == "trade":
            new = oms.on_trade(data)
            if log is not None:
                log.append(f"trade  {data.trade_id} {data.volume:g}@{data.price}" + ("" if new else "  (重复，已忽略)"))
        elif kind == "cancel_reject":
            oms.on_cancel_reject(data["client_order_id"], data["reason"])
            if log is not None:
                log.append(f"cancel_reject  {data['reason']}")

    exchange.add_listener(on_report)


def send(exchange: MockExchange, oms: OMS, req: OrderRequest) -> OrderData:
    oms.create_order(req)
    exchange.submit(req)
    return oms.orders[req.client_order_id]


# ==================== 第一部分：订单生命周期与转换表 ====================


def lifecycle():
    print("=" * 60)
    print("第一部分：订单生命周期与状态转换表")
    print("=" * 60)
    print("""
                       ┌─────────────┐
             send ───▶ │ SUBMITTING  │ 提交中（已发出，尚未确认）
                       └──────┬──────┘
             ┌────────────────┼─────────────────────┐
             ▼                ▼                     ▼
       ┌──────────┐    ┌────────────┐        ┌────────────┐
       │ REJECTED │    │ NOT_TRADED │ 已报    │ ALL_TRADED │
       │   拒单    │    │  (未成交)   │───────▶│  全部成交   │
       └──────────┘    └─────┬──────┘        └────────────┘
                             │ 部分成交               ▲
                             ▼                       │
                       ┌─────────────┐ 剩余成交       │
                       │ PART_TRADED │───────────────┘
                       └─────┬───────┘ (可多次部分成交)
                             │ 撤单（NOT_TRADED 也可直接撤）
                             ▼
                       ┌─────────────┐
                       │  CANCELLED  │ 已撤（可能已有部分成交）
                       └─────────────┘

    📦 比喻：订单像快递包裹 —— 已下单 → 已揽收 → 部分送达 → 已签收，也可能被拒收或中途取消。
    """)

    states = list(Status)
    matrix = pd.DataFrame(
        [["✓" if new in ALLOWED_TRANSITIONS[old] else "·" for new in states] for old in states],
        index=[f"{s.name}" for s in states],
        columns=[s.value for s in states],
    )
    print("状态转换表（行 = 当前状态，列 = 目标状态，✓ = 允许）：")
    print(matrix.to_string())
    print("""
    📌 读表要点：
    - 三个终态（ALL_TRADED / CANCELLED / REJECTED）所在的行全是 "·"：终态不再变化
    - SUBMITTING 可以直接到任何后续状态：确认回报可能丢失，第一条回报就是成交
    - PART_TRADED → PART_TRADED 合法：一笔订单可以多次部分成交
    - NOT_TRADED → REJECTED 不合法：本课的简化模型认为"已报"之后只会成交或撤销
    """)


# ==================== 第二部分：非法转换要报错 ====================


def illegal_transitions():
    print("\n" + "=" * 60)
    print("第二部分：非法转换要报错")
    print("=" * 60)
    rows = []
    for old, new in [
        (Status.SUBMITTING, Status.NOT_TRADED),
        (Status.PART_TRADED, Status.ALL_TRADED),
        (Status.ALL_TRADED, Status.CANCELLED),
        (Status.CANCELLED, Status.PART_TRADED),
        (Status.PART_TRADED, Status.NOT_TRADED),
    ]:
        try:
            check_transition(old, new)
            rows.append([old.name, new.name, "合法"])
        except InvalidTransition as exc:
            rows.append([old.name, new.name, f"InvalidTransition: {exc}"])
    print(pd.DataFrame(rows, columns=["当前", "目标", "结果"]).to_string(index=False))

    print("""
    状态机本身对非法转换抛异常；但 OMS 收到的回报来自外部，不能因为一条奇怪的回报就崩溃。
    OMS 的策略：
      - 乱序到达的"旧"回报（如部分成交后又收到"已报"）→ 识别为过期回报，忽略并计数
      - 真正无法解释的转换 → 记为异常（anomaly），通过 on_anomaly 回调交给风控 / 告警
    """)
    anomalies = []
    oms = OMS(on_anomaly=anomalies.append)
    oms.create_order(OrderRequest(SYM, Direction.LONG, 10.0, 500, client_order_id="C1"))
    base = dict(symbol=SYM, direction=Direction.LONG, price=10.0, volume=500, exchange_order_id="EX1")
    oms.on_order(OrderData("C1", status=Status.NOT_TRADED, **base))
    oms.on_trade(TradeData("T1", "C1", SYM, Direction.LONG, 10.0, 200))
    oms.on_order(OrderData("C1", status=Status.NOT_TRADED, **base))  # 迟到的"已报"
    oms.on_order(OrderData("C1", status=Status.REJECTED, **base))  # 部分成交后又说拒单
    print(f"   订单状态：{oms.orders['C1'].status.name}，已成交 {oms.orders['C1'].traded:g}")
    print(f"   过期回报计数：{oms.stats['stale_reports']}")
    print(f"   异常记录：{anomalies}")


# ==================== 第三部分：真实世界的回报场景 ====================


def scenario(name: str, setup) -> list:
    exchange, oms, log = MockExchange(), OMS(), []
    wire(exchange, oms, log)
    note = setup(exchange, oms)
    order = oms.orders["C1"]
    pos = oms.positions.get(SYM).volume
    return [name, order.status.value, f"{order.traded:g}", f"{pos:g}", note, log]


def sc_partial(ex, oms):
    ex.set_market(SYM, 10.0, ask=10.01, ask_volume=300)
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.01, 500, client_order_id="C1"))
    ex.set_market(SYM, 10.0, ask=10.01, ask_volume=500)
    return "分两次成交"


def sc_reject(ex, oms):
    ex.set_market(SYM, 10.0)
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.01, 200_000, client_order_id="C1"))
    return oms.orders["C1"].reason


def sc_partial_cancel(ex, oms):
    ex.set_market(SYM, 10.0, ask=10.01, ask_volume=200)
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.01, 500, client_order_id="C1"))
    oms.request_cancel("C1")
    ex.cancel("C1")
    return "已成交部分保留"


def sc_cancel_race(ex, oms):
    ex.set_market(SYM, 10.0, ask=10.05)
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.02, 500, client_order_id="C1"))
    oms.request_cancel("C1")  # 撤单请求发出时 ……
    ex.set_market(SYM, 10.0, ask=10.01)  # …… 行情变化，订单先全部成交了
    ex.cancel("C1")
    return oms.orders["C1"].reason


def sc_duplicate(ex, oms):
    ex.set_market(SYM, 10.0)
    ex.inject("duplicate_trade")
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.01, 500, client_order_id="C1"))
    return f"重复成交 {oms.stats['duplicate_trades']} 次已忽略"


def sc_out_of_order(ex, oms):
    ex.set_market(SYM, 10.0)
    ex.inject("trade_before_order")
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.01, 500, client_order_id="C1"))
    return f"过期回报 {oms.stats['stale_reports']} 条已忽略"


def sc_late_trade(ex, oms):
    ex.set_market(SYM, 10.0, ask=10.05)
    send(ex, oms, OrderRequest(SYM, Direction.LONG, 10.02, 500, client_order_id="C1"))
    ex.cancel("C1")
    # 部分柜台可能在"已撤"之后才推送一笔撤单前发生的成交（这里直接构造这条回报）
    oms.on_trade(TradeData("T-LATE", "C1", SYM, Direction.LONG, 10.02, 100))
    return f"迟到成交 {oms.stats['late_trades']} 笔：只记量，不改终态"


def real_world_scenarios():
    print("\n" + "=" * 60)
    print("第三部分：真实世界的回报场景（MockExchange 复现）")
    print("=" * 60)
    rows = [
        scenario("正常：部分成交 → 全部成交", sc_partial),
        scenario("资金不足 → 拒单", sc_reject),
        scenario("部分成交后撤单", sc_partial_cancel),
        scenario("撤单竞态：撤单时已全部成交", sc_cancel_race),
        scenario("成交回报重复推送", sc_duplicate),
        scenario("乱序：成交先于确认到达", sc_out_of_order),
        scenario("已撤后又收到成交", sc_late_trade),
    ]
    print(pd.DataFrame([r[:5] for r in rows], columns=["场景", "最终状态", "已成交", "持仓", "说明"]).to_string(index=False))

    for idx in (3, 5):
        print(f"\n🔎 「{rows[idx][0]}」的回报序列：")
        for line in rows[idx][5]:
            print("     " + line)
    print("""
    💡 处理原则：
       - 撤单被拒是正常竞态，不是错误：以最终回报为准
       - 成交先到时，直接把订单推进到 PART/ALL_TRADED；随后到达的"已报"是过期回报
       - 重复回报（重连后柜台重推当日回报很常见）靠 trade_id 去重
       - 迟到成交仍要计入持仓：钱已经花了，持仓必须反映事实
    """)


# ==================== 第四部分：订单号映射 ====================


def order_id_mapping():
    print("\n" + "=" * 60)
    print("第四部分：订单号映射")
    print("=" * 60)
    print("""
      策略 ──意图──▶ OMS：生成 client_order_id ──报单（带 coid）──▶ 交易所
                     OMS ◀──确认：exchange_order_id=EX... ─────────┘
                     维护映射 coid ↔ exchange_order_id，之后的回报两种编号都能定位订单
    """)
    gen = OrderIdGenerator("S1")
    exchange, oms = MockExchange(), OMS()
    wire(exchange, oms)
    exchange.set_market(SYM, 10.0, ask=10.05)
    ids = [gen.next() for _ in range(3)]
    for coid in ids:
        send(exchange, oms, OrderRequest(SYM, Direction.LONG, 10.00, 100, client_order_id=coid))
    print(pd.DataFrame(
        [[c, oms.orders[c].exchange_order_id, oms.orders[c].status.value] for c in ids],
        columns=["client_order_id", "exchange_order_id", "状态"],
    ).to_string(index=False))
    exid = oms.orders[ids[1]].exchange_order_id
    print(f"\n   按交易所编号 {exid} 反查本地订单：{oms.find_by_exchange_id(exid).client_order_id}")
    print("""
    📌 client_order_id 格式 = 策略前缀 - 启动时间戳 - 自增序号：进程重启后不会与之前的编号冲突。
       不同接口对订单号的长度和字符集有限制（如 CTP 的 OrderRef 是会话内递增的数字串），要按接口适配。
    """)


# ==================== 第五部分：持仓由成交推导 ====================


def positions_from_trades():
    print("\n" + "=" * 60)
    print("第五部分：持仓由成交推导")
    print("=" * 60)
    book = PositionBook()
    sym = "rb2501.SHFE"
    rows = []
    for tid, direction, offset, price, vol in [
        ("T1", Direction.LONG, Offset.OPEN, 3500, 2),
        ("T2", Direction.LONG, Offset.OPEN, 3520, 2),
        ("T3", Direction.SHORT, Offset.CLOSE_TODAY, 3550, 1),
    ]:
        pnl = book.on_trade(TradeData(tid, "X", sym, direction, price, vol, offset))
        p = book.get(sym, Direction.LONG)
        rows.append([tid, f"{direction.value}{offset.value}", price, vol, f"{p.volume:g}", f"{p.price:.1f}", f"{pnl:,.0f}"])
    print(pd.DataFrame(rows, columns=["成交", "操作", "价格", "手数", "多头持仓", "持仓均价", "本笔已实现盈亏"]).to_string(index=False))
    print("    （螺纹钢合约乘数按 10 计；平仓盈亏 = (平仓价 - 均价) × 手数 × 乘数）")

    print("""
    ❌ 错误做法：只在订单"全部成交"时把委托量加到持仓上
    """)
    exchange, oms = MockExchange(), OMS()
    wire(exchange, oms)
    exchange.set_market(SYM, 10.0, ask=10.01, ask_volume=200)
    send(exchange, oms, OrderRequest(SYM, Direction.LONG, 10.01, 500, client_order_id="C1"))
    exchange.cancel("C1")
    wrong = sum(o.volume for o in oms.orders.values() if o.status is Status.ALL_TRADED)
    print(f"   订单最终状态 {oms.orders['C1'].status.value}，按订单状态推导的持仓 = {wrong:g}")
    print(f"   按成交累计的持仓 = {oms.positions.get(SYM).volume:g}，交易所持仓 = {exchange.positions.get(SYM).volume:g}")
    print("   → 部分成交后撤单、迟到成交，都会让'按订单推导'出错；成交才是事实。")


# ==================== 第六部分：A 股 T+1 与可用数量 ====================


def t_plus_one():
    print("\n" + "=" * 60)
    print("第六部分：A 股 T+1 与可用数量")
    print("=" * 60)
    exchange, oms = MockExchange(), OMS()
    wire(exchange, oms)
    rows = []

    def snap(step):
        p = oms.positions.get(SYM)
        rows.append([step, f"{p.volume:g}", f"{p.yd_volume:g}", f"{p.frozen:g}", f"{oms.positions.available(SYM):g}"])

    exchange.set_market(SYM, 10.0)
    send(exchange, oms, OrderRequest(SYM, Direction.LONG, 10.01, 1000, client_order_id="B1"))
    snap("第 1 天：买入 1000 股")
    o = send(exchange, oms, OrderRequest(SYM, Direction.SHORT, 9.99, 100, client_order_id="S0"))
    snap(f"第 1 天：卖出 100 → {o.status.value}")
    exchange.settle()
    oms.positions.settle()
    snap("日终结算：今仓转昨仓")
    exchange.set_market(SYM, 10.0, bid=9.99, bid_volume=100)
    send(exchange, oms, OrderRequest(SYM, Direction.SHORT, 9.99, 300, client_order_id="S1"))
    snap("第 2 天：卖 300，成交 100")
    exchange.cancel("S1")
    snap("撤销剩余 200")
    print(pd.DataFrame(rows, columns=["步骤", "总持仓", "昨仓(可卖基数)", "冻结", "可卖"]).to_string(index=False))
    print(f"\n   第 1 天卖单的拒单原因：{oms.orders['S0'].reason}")
    print("""
    📌 可卖 = 昨仓 - 冻结。挂着的卖单会冻结持仓，撤单或成交后释放。
       期货没有 T+1，但要区分今仓 / 昨仓（平今、平昨），第 06 课在 MockExchange 上演示。
    """)


# ==================== 主函数 ====================


def main():
    lifecycle()
    illegal_transitions()
    real_world_scenarios()
    order_id_mapping()
    positions_from_trades()
    t_plus_one()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 有些柜台会在"已报"之后再拒单。修改 ALLOWED_TRANSITIONS 允许 NOT_TRADED → REJECTED，
       并补充 tests/test_oms.py 中对应的测试，思考这样做会放过哪些真正的异常
    2. 为 OMS 增加"超时未确认"检测：SUBMITTING 超过 N 秒的订单列入告警清单
    3. 给 PositionBook 加上手续费：开仓、平今、平昨分别使用不同费率，比较日内策略的成本
    4. 构造一个场景：同一订单的两笔成交以相反顺序到达，验证持仓和订单状态都正确
    """)


if __name__ == "__main__":
    main()
