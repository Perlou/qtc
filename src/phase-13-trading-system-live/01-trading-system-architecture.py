"""
交易系统架构：行情、策略、风控、OMS、执行
==========================================

学习目标：
    1. 理解回测与实盘的本质差异，知道实盘系统要多处理哪些问题
    2. 掌握交易系统的组件划分与职责边界（行情 → 策略 → 风控 → OMS → 网关）
    3. 用 dataclass 定义统一数据模型，理解网关作为"防腐层"的意义
    4. 跑通本阶段迷你交易系统 mini_trader 的第一条链路：下单 → 回报 → 持仓

核心概念：
    - 统一数据模型：TickData / BarData / OrderRequest / OrderData / TradeData / PositionData / AccountData
    - 网关（Gateway）：只做接口翻译，屏蔽 CTP / QMT / ccxt 的差异
    - 交易所是真相来源：本地状态只是缓存，要定期与交易所核对
    - MockExchange：本阶段全程使用的模拟交易所，可以注入各种故障

前置知识：
    - Phase 4：事件驱动回测引擎
    - Phase 11：订单类型、订单簿
    - 本阶段 CONCEPT.md 第 1、2 节

环境要求：
    - 只依赖 Python 标准库 + pandas，不需要网络
    - 共享包 mini_trader/ 位于本阶段目录，后续每一课都在它的基础上展开
"""

import asyncio
import inspect
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import (
    OMS,
    AccountData,
    BarData,
    BaseGateway,
    Direction,
    EventEngine,
    MockExchange,
    MockGateway,
    Offset,
    OrderData,
    OrderRequest,
    PositionData,
    Status,
    TickData,
    TradeData,
    from_dict,
    market_of,
    to_dict,
)
from mini_trader.events import EVENT_ORDER, EVENT_TRADE

load_dotenv()


# ==================== 第一部分：回测与实盘的鸿沟 ====================


def backtest_vs_live():
    print("=" * 60)
    print("第一部分：回测与实盘的鸿沟")
    print("=" * 60)

    table = pd.DataFrame(
        [
            ["时间", "循环遍历历史，可以快进", "真实时间流逝，事件随时到达"],
            ["下单", "函数调用，立即成交", "异步请求：延迟、拒单、部分成交"],
            ["数据", "干净、完整、有序", "可能缺失、重复、乱序、延迟"],
            ["状态", "全在内存，跑完即丢", "必须持久化，崩溃后能恢复"],
            ["失败", "抛异常，修代码重跑", "断网、柜台拒单、进程崩溃"],
            ["真相来源", "自己的模拟撮合", "交易所 / 柜台"],
            ["后果", "浪费时间", "亏钱，甚至违规"],
        ],
        columns=["维度", "回测", "实盘"],
    )
    print(table.to_string(index=False))
    print("""
    🔑 Web 后端经验可以直接迁移：
       消息队列 → 事件引擎      支付状态机 → 订单状态机      限流熔断 → 风控
       支付幂等键 → client_order_id      支付对账 → 持仓 / 成交对账

    ⚠️ 真正的差别在于"出错的代价"：重复下单不是返回 500，而是仓位翻倍。
    """)


# ==================== 第二部分：组件全景与职责 ====================


def components():
    print("\n" + "=" * 60)
    print("第二部分：组件全景与职责")
    print("=" * 60)

    print("""
       交易所 / 柜台（CTP、券商、加密交易所）
            │ 行情 / 回报                      ▲ 报单 / 撤单 / 查询
            ▼                                  │
       ┌──────────────────── 网关 Gateway（翻译成统一模型）───────────────┐
       └────┬───────────────────────────────────────────▲─────────────────┘
            │ TickEvent / OrderEvent / TradeEvent        │ OrderRequest
            ▼                                            │
       ╔════════════════════ 事件引擎 EventEngine ═══════╧════════════════╗
       ╚══╤══════════════╤═══════════════╤═══════════════╤════════════════╝
          ▼              ▼               ▼               ▼
       ┌──────┐  意图  ┌──────┐  通过  ┌──────┐       ┌──────────┐
       │ 策略  │ ─────▶ │ 风控  │ ─────▶ │ OMS  │ ────▶ │ 执行算法   │
       └──────┘        └──────┘        └──────┘       └──────────┘
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │ 事件日志/持久化 │ │ 监控 / 告警    │ │ 对账 / 恢复    │   ← 独立于交易主流程
       └──────────────┘ └──────────────┘ └──────────────┘
    """)

    roles = pd.DataFrame(
        [
            ["行情网关", "订阅行情，转为 Tick/Bar", "断线重连、去重、时间戳标准化", "gateway.py", "01"],
            ["事件引擎", "分发所有事件", "异常隔离、跨线程投递、不能阻塞", "engine.py", "02"],
            ["策略", "产生目标持仓 / 订单意图", "不直接调用交易接口", "（各课示例）", "02/05"],
            ["风控", "事前检查 + 事中 Kill Switch", "一票否决、全局暂停", "risk.py", "04"],
            ["OMS", "订单生命周期、持仓", "状态机、去重、单一写者", "oms.py", "03"],
            ["交易网关", "下单 / 撤单 / 查询", "屏蔽接口差异、处理流控", "gateway.py", "01/06-08"],
            ["监控", "健康检查、指标、告警", "业务级心跳", "monitor.py", "09"],
            ["对账恢复", "与交易所核对、重启恢复", "交易所是真相", "recovery.py", "10"],
        ],
        columns=["组件", "职责", "关键设计点", "mini_trader 模块", "课程"],
    )
    print(roles.to_string(index=False))


# ==================== 第三部分：统一数据模型 ====================


def data_models():
    print("\n" + "=" * 60)
    print("第三部分：统一数据模型（dataclass）")
    print("=" * 60)

    print("\n📄 mini_trader/models.py 中 OrderData 的定义：\n")
    print(inspect.getsource(OrderData))

    rows = []
    for cls in (TickData, BarData, OrderRequest, OrderData, TradeData, PositionData, AccountData):
        names = [f.name for f in fields(cls)]
        rows.append([cls.__name__, len(names), ", ".join(names[:6]) + (" ..." if len(names) > 6 else "")])
    print(pd.DataFrame(rows, columns=["模型", "字段数", "主要字段"]).to_string(index=False))

    print("""
    📌 设计要点：
    - 标的代码统一为 <代码>.<交易所>：rb2501.SHFE、600000.SSE、BTC/USDT.BINANCE
    - 方向（Direction）与开平（Offset）分开：股票只用方向，期货两者都要
    - OrderRequest 是"意图"，OrderData 是"订单的当前状态"，TradeData 是"成交事实"
    - client_order_id 在发单前就生成；exchange_order_id 要等交易所确认后才有
    """)

    for sym in ["600000.SSE", "rb2501.SHFE", "BTC/USDT.BINANCE"]:
        print(f"   {sym:<18} → 市场类型：{market_of(sym).value}")

    trade = TradeData("T0001", "S1-0001", "rb2501.SHFE", Direction.LONG, 3500.0, 2, Offset.OPEN, "EX01")
    as_json = json.dumps(to_dict(trade), ensure_ascii=False)
    restored = from_dict(TradeData, json.loads(as_json))
    print(f"\n   序列化（用于事件日志）：{as_json[:110]}...")
    print(f"   反序列化后与原对象相等：{restored == trade}")


# ==================== 第四部分：网关是防腐层 ====================

# 不同接口的原始回报格式（字段名与取值为示意，以各接口官方文档为准）
CTP_RAW_ORDER = {
    "InstrumentID": "rb2501",
    "ExchangeID": "SHFE",
    "Direction": "0",  # '0' 买 / '1' 卖
    "CombOffsetFlag": "0",  # '0' 开 / '1' 平 / '3' 平今 / '4' 平昨
    "LimitPrice": 3500.0,
    "VolumeTotalOriginal": 2,
    "VolumeTraded": 1,
    "OrderStatus": "1",  # '1' 部分成交还在队列中
    "OrderRef": "000001",
    "OrderSysID": "  123456",
}
CCXT_RAW_ORDER = {
    "id": "8891",
    "clientOrderId": "S1-0002",
    "symbol": "BTC/USDT",
    "side": "sell",
    "price": 65000.0,
    "amount": 0.01,
    "filled": 0.0,
    "status": "open",
}

_CTP_STATUS = {"0": Status.ALL_TRADED, "1": Status.PART_TRADED, "3": Status.NOT_TRADED, "5": Status.CANCELLED}
_CTP_OFFSET = {"0": Offset.OPEN, "1": Offset.CLOSE, "3": Offset.CLOSE_TODAY, "4": Offset.CLOSE_YESTERDAY}


def from_ctp(raw: dict, front_id: int = 1, session_id: int = 42) -> OrderData:
    return OrderData(
        client_order_id=f"{front_id}_{session_id}_{raw['OrderRef'].strip()}",  # CTP 用三元组唯一定位订单
        symbol=f"{raw['InstrumentID']}.{raw['ExchangeID']}",
        direction=Direction.LONG if raw["Direction"] == "0" else Direction.SHORT,
        offset=_CTP_OFFSET[raw["CombOffsetFlag"][0]],
        price=raw["LimitPrice"],
        volume=raw["VolumeTotalOriginal"],
        traded=raw["VolumeTraded"],
        status=_CTP_STATUS.get(raw["OrderStatus"], Status.SUBMITTING),
        exchange_order_id=raw["OrderSysID"].strip(),
    )


def from_ccxt(raw: dict, exchange: str = "BINANCE") -> OrderData:
    status = {"closed": Status.ALL_TRADED, "canceled": Status.CANCELLED, "rejected": Status.REJECTED}.get(
        raw["status"], Status.PART_TRADED if raw["filled"] else Status.NOT_TRADED
    )
    return OrderData(
        client_order_id=raw["clientOrderId"],
        symbol=f"{raw['symbol']}.{exchange}",
        direction=Direction.LONG if raw["side"] == "buy" else Direction.SHORT,
        price=raw["price"],
        volume=raw["amount"],
        traded=raw["filled"],
        status=status,
        exchange_order_id=raw["id"],
    )


def anti_corruption_layer():
    print("\n" + "=" * 60)
    print("第四部分：网关是防腐层（Anti-Corruption Layer）")
    print("=" * 60)

    print("""
    CTP 用 '0'/'1' 表示买卖、'3' 表示平今；ccxt 用 "buy"/"sell" 和 "open"/"closed"。
    如果策略代码里出现 if exchange == "binance"，系统会迅速腐化。
    正确做法：每个网关负责把原始格式翻译成统一的 OrderData，上层只认这一种结构。
    """)
    unified = [from_ctp(CTP_RAW_ORDER), from_ccxt(CCXT_RAW_ORDER)]
    df = pd.DataFrame(
        [
            [src, o.client_order_id, o.symbol, o.direction.value, o.offset.value or "-", f"{o.volume:g}", f"{o.traded:g}", o.status.value]
            for src, o in zip(["CTP 回报", "ccxt 回报"], unified)
        ],
        columns=["来源", "client_order_id", "symbol", "方向", "开平", "委托量", "已成交", "状态"],
    )
    print(df.to_string(index=False))

    abstract = sorted(BaseGateway.__abstractmethods__)
    print(f"\n📄 BaseGateway 要求子类实现的方法（共 {len(abstract)} 个）：")
    print("   " + ", ".join(abstract))
    print("""
    - 网关方法都是 async：真实接口的请求和回报都是异步的
    - on_tick / on_order / on_trade 由基类提供：把翻译好的数据推入事件引擎
    - 回测时用"模拟网关"替换真实网关，策略代码不变（Phase 4 的延续）
    """)


# ==================== 第五部分：跑通第一条链路 ====================


async def first_round_trip():
    engine = EventEngine(timer_interval=None)
    exchange = MockExchange(initial_cash=100_000)
    gateway = MockGateway(engine, exchange)
    oms = OMS()
    timeline = []

    def on_order(event):
        oms.on_order(event.data)
        timeline.append(["订单回报", event.data.status.value, f"{event.data.traded:g}", event.data.exchange_order_id])

    def on_trade(event):
        oms.on_trade(event.data)
        timeline.append(["成交回报", f"{event.data.volume:g} @ {event.data.price}", "", event.data.trade_id])

    engine.register(EVENT_ORDER, on_order)
    engine.register(EVENT_TRADE, on_trade)
    engine.start()
    await gateway.connect()
    await gateway.subscribe("600000.SSE")

    # 卖一只有 300 股，买 500 股 → 先成交 300，剩余 200 挂单；行情补充后再成交
    exchange.set_market("600000.SSE", last=10.00, bid=9.99, ask=10.01, ask_volume=300)
    req = OrderRequest("600000.SSE", Direction.LONG, 10.01, 500, client_order_id="S1-0001")
    oms.create_order(req)
    timeline.append(["本地下单", oms.orders["S1-0001"].status.value, "0", ""])
    await gateway.send_order(req)
    await engine.join()
    exchange.set_market("600000.SSE", last=10.01, bid=10.00, ask=10.01, ask_volume=1000)
    await engine.join()

    account = await gateway.query_account()
    remote_pos = await gateway.query_positions()
    await engine.stop()
    return timeline, oms, account, remote_pos


def run_first_round_trip():
    print("\n" + "=" * 60)
    print("第五部分：跑通第一条链路（MockExchange）")
    print("=" * 60)

    timeline, oms, account, remote_pos = asyncio.run(first_round_trip())
    print("\n⏱️ 事件时间线：")
    print(pd.DataFrame(timeline, columns=["事件", "状态 / 成交", "累计成交", "编号"]).to_string(index=False))

    local = oms.positions.get("600000.SSE")
    print(f"""
    📊 本地 OMS 持仓：{local.volume:g} 股，均价 {local.price:.4f}，可卖 {oms.positions.available('600000.SSE'):g} 股
    📊 交易所持仓：  {remote_pos[0].volume:g} 股（交易所是真相来源，两者应一致）
    💰 交易所资金：  余额 {account.balance:,.2f}，冻结 {account.frozen:,.2f}，可用 {account.available:,.2f}

    💡 观察要点：
       1. 一笔委托产生了多条回报：未成交（即已报）→ 部分成交 → 全部成交，每条都是独立事件
       2. 持仓由两笔成交累计得到，而不是由"订单状态 = 全部成交"推导
       3. 今日买入的股票可卖数量为 0：这是 A 股 T+1，第 03 课会详细处理
    """)


# ==================== 第六部分：架构原则 ====================


def principles():
    print("\n" + "=" * 60)
    print("第六部分：架构原则")
    print("=" * 60)
    print("""
    ┌──────────────────┬──────────────────────────────────────────────┐
    │ 原则             │ 说明                                         │
    ├──────────────────┼──────────────────────────────────────────────┤
    │ 策略不碰网关     │ 策略只输出意图，所有订单必须经过风控和 OMS   │
    │ 交易所是真相来源 │ 本地状态可能错，定期以交易所查询结果校正     │
    │ 单一写者         │ 订单与持仓只由 OMS 修改，避免并发写不一致    │
    │ 一切皆事件       │ 行情、回报、定时器、错误都可记录、可回放     │
    │ 回测实盘同构     │ 换一个网关就能从回测切到实盘，策略代码不变   │
    │ 失败默认安全     │ 不确定时宁可停止交易，也不要"猜"状态继续下单 │
    └──────────────────┴──────────────────────────────────────────────┘

    📁 本阶段共享包 mini_trader/ 的结构：
       models.py      统一数据模型            contracts.py  合约规格（乘数、保证金）
       events.py      事件类型                engine.py     asyncio 事件引擎
       oms.py         订单状态机 / OMS        positions.py  持仓簿（T+1、今昨仓）
       risk.py        风控引擎                gateway.py    网关抽象 + MockGateway
       mock_exchange.py  模拟交易所（故障注入）  monitor.py    监控告警
       persistence.py 事件日志                recovery.py   幂等、对账、恢复
       sessions.py    交易日历与时段          system.py     组件装配
    """)


# ==================== 主函数 ====================


def main():
    backtest_vs_live()
    components()
    data_models()
    anti_corruption_layer()
    run_first_round_trip()
    principles()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 TickData 增加 bid/ask 第 2~5 档字段，思考：哪些组件需要修改，哪些不需要？
    2. 仿照 from_ccxt 写一个 from_ccxt_trade()，把 ccxt 的成交结构转换为 TradeData
    3. 把第五部分的卖一挂单量改为 100，委托量改为 1000，观察会产生几笔成交回报
    4. 列出你熟悉的一个支付系统中的组件，逐一对应到交易系统的组件上，找出无法对应的部分
    """)


if __name__ == "__main__":
    main()
