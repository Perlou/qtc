"""
事前 / 事中风控引擎
==================

学习目标：
    1. 理解风控的三道防线：事前检查、事中监控、事后复盘
    2. 实现可插拔的事前检查规则，并逐条触发：数量、金额、价格偏离、涨跌停、持仓上限、
       资金、可用持仓、报单频率
    3. 实现事中 Kill Switch：日内亏损、连续拒单、行情中断，触发后自动撤单并拒绝新单
    4. 理解"自动暂停、人工恢复"的设计原则

核心概念：
    - 事前风控（Pre-trade）：订单发出前同步检查，任何一条不通过就拒绝，订单不会到达交易所
    - 事中风控（Intra-day）：持续监控账户与系统状态，异常时全局暂停（Kill Switch）
    - 一票否决：风控位于策略和 OMS 之间，拥有否决权和全局暂停权

前置知识：
    - 01~03 课
    - 本阶段 CONCEPT.md 第 5 节

环境要求：
    - 不需要网络；风控规则的阈值都是演示值，实盘要按策略和账户规模设定
"""

import asyncio
import inspect
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import (
    AccountData,
    Direction,
    MockExchange,
    Offset,
    OrderRequest,
    PositionBook,
    RiskConfig,
    RiskEngine,
    TickData,
)
from mini_trader.events import EVENT_TICK
from mini_trader.system import TradingSystem

load_dotenv()

SYM = "600000.SSE"


class FakeClock:
    """可控时钟：让频率限制、行情中断等与时间有关的规则可以确定性地复现"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ==================== 第一部分：风控的三道防线 ====================


def three_lines():
    print("=" * 60)
    print("第一部分：风控的三道防线")
    print("=" * 60)
    print("""
    🚗 比喻：事前风控是出发前的车况检查，事中风控是行驶中的 ABS 和安全气囊，事后风控是事故复盘。

    ┌──────────────────────────────────────────────────────────────────┐
    │ 事前（Pre-trade）：每个订单发出前同步检查，不通过直接拒绝            │
    │    数量 / 金额 / 价格偏离 / 涨跌停 / 持仓上限 / 资金 / 可用持仓 / 流控  │
    │ 事中（Intra-day）：持续监控，触发时暂停交易并撤单                    │
    │    日内亏损 / 连续拒单 / 行情中断 / 对账差异 / 网关断线               │
    │ 事后（Post-trade）：盘后分析                                       │
    │    成交质量（TCA）/ 风险暴露 / 异常交易回顾 / 规则调参                │
    │ 柜台和交易所也有风控，但不能依赖它们替你兜底                          │
    └──────────────────────────────────────────────────────────────────┘

    在系统中的位置：策略 ──意图──▶ [风控] ──通过──▶ OMS ──▶ 网关 ──▶ 交易所
                                     └──拒绝──▶ 记录原因，订单不会离开本机
    """)


# ==================== 第二部分：事前检查逐项触发 ====================


def build_risk() -> tuple[RiskEngine, FakeClock, AccountData, PositionBook]:
    clock = FakeClock()
    cfg = RiskConfig(
        max_order_volume=1_000,
        max_order_notional=50_000,
        max_position=5_000,
        max_price_deviation=0.05,
        max_orders_per_window=5,
        order_window_sec=1.0,
        symbol_whitelist=frozenset({SYM, "000001.SZSE", "rb2501.SHFE"}),
    )
    risk = RiskEngine(cfg, clock=clock)
    now = datetime.now()
    risk.update_tick(TickData(SYM, now, 10.00, 9.99, 10.01, limit_up=11.00, limit_down=9.00))
    risk.update_tick(TickData("rb2501.SHFE", now, 3500, 3499, 3501))
    account = AccountData("DEMO", balance=100_000)
    book = PositionBook()
    p = book.get(SYM)
    p.volume, p.yd_volume = 4_800, 300  # 总持仓 4800 股，其中昨仓（可卖）300 股
    return risk, clock, account, book


def pre_trade_checks():
    print("\n" + "=" * 60)
    print("第二部分：事前检查逐项触发")
    print("=" * 60)
    risk, clock, account, book = build_risk()
    print(f"""
    演示配置：单笔 ≤ {risk.cfg.max_order_volume:g} 股/手，单笔金额 ≤ {risk.cfg.max_order_notional:,.0f}，
              单标的持仓 ≤ {risk.cfg.max_position:g}，价格偏离 ≤ ±{risk.cfg.max_price_deviation:.0%}，
              每 {risk.cfg.order_window_sec:g} 秒最多 {risk.cfg.max_orders_per_window} 笔
    当前状态：{SYM} 最新价 10.00（涨停 11.00 / 跌停 9.00），可用资金 {account.available:,.0f}，
              持仓 4800 股（昨仓 300 股可卖）
    """)
    buy = lambda price, vol, sym=SYM: OrderRequest(sym, Direction.LONG, price, vol, client_order_id="X")  # noqa: E731
    rich = AccountData("RICH", balance=10_000_000)
    cases = [
        ("正常订单", buy(10.02, 100), rich),
        ("不在白名单", buy(5.00, 100, "601318.SSE"), rich),
        ("胖手指：多打一个 0", buy(10.02, 10_000), rich),
        ("没有行情", buy(12.00, 100, "000001.SZSE"), rich),
        ("超过涨停价", buy(11.50, 100), rich),
        ("价格偏离过大", buy(10.60, 100), rich),
        ("合约价值超限（期货乘数）", OrderRequest("rb2501.SHFE", Direction.LONG, 3500, 3, Offset.OPEN, "X"), rich),
        ("持仓上限", buy(10.02, 500), rich),
        ("资金不足", buy(10.02, 100), AccountData("POOR", balance=800)),
        ("可卖不足（T+1）", OrderRequest(SYM, Direction.SHORT, 9.99, 500, client_order_id="X"), rich),
    ]
    rows = []
    for name, req, acct in cases:
        res = risk.check_order(req, acct, book)
        rows.append([name, "✅ 通过" if res.ok else "❌ 拒绝", res.rule or "-", res.reason or "-"])

    for i in range(6):  # 同一时刻连发 6 笔
        res = risk.check_order(buy(10.02, 10), rich, book)
    rows.append(["1 秒内连发 6 笔（第 6 笔）", "✅ 通过" if res.ok else "❌ 拒绝", res.rule or "-", res.reason or "-"])
    clock.t += 1.5
    res = risk.check_order(buy(10.02, 10), rich, book)
    rows.append(["1.5 秒后再发", "✅ 通过" if res.ok else "❌ 拒绝", res.rule or "-", res.reason or "-"])
    print(pd.DataFrame(rows, columns=["场景", "结果", "触发规则", "原因"]).to_string(index=False))

    print("\n📄 规则示例（mini_trader/risk.py）：\n")
    print(inspect.getsource(RiskEngine._rule_price_deviation))
    print(f"   规则检查顺序：{' → '.join(RiskEngine.RULES)}")
    print("""
    📌 设计要点：
    - 每条规则是一个返回"拒绝原因"的小函数，新增规则只需加一个方法 —— 可插拔
    - 先查开关和便宜的静态规则，最后查有状态的频率限制（只有通过的订单才计入频率）
    - "没有行情就拒绝"：缺少参考价时无法判断价格是否合理，失败默认安全
    - 合约价值 = 价格 × 数量 × 乘数：期货 3 手螺纹钢的合约价值是 3500×3×10，而不是 3500×3
    """)


# ==================== 第三部分：风控接入下单链路 ====================


async def risk_in_pipeline():
    exchange = MockExchange(initial_cash=100_000)
    system = TradingSystem(exchange, RiskConfig(max_order_volume=1_000, max_order_notional=50_000))
    await system.start()
    await system.gateway.subscribe(SYM)
    exchange.set_market(SYM, 10.00)
    await system.engine.join()
    results = []
    for vol in (500, 5_000):
        req = system.new_request(SYM, Direction.LONG, 10.01, vol)
        res = await system.send_order(req)
        results.append([f"买入 {vol} 股", res.outcome, res.detail or "-"])
    await system.stop()
    return results, len(exchange.query_orders())


def pipeline_demo():
    print("\n" + "=" * 60)
    print("第三部分：风控接入下单链路")
    print("=" * 60)
    print("\n📄 TradingSystem.send_order：风控 → OMS → 网关\n")
    print(inspect.getsource(TradingSystem.send_order))
    results, n_orders = asyncio.run(risk_in_pipeline())
    print(pd.DataFrame(results, columns=["订单", "结果", "说明"]).to_string(index=False))
    print(f"\n   交易所实际收到的订单数：{n_orders}（被风控拒绝的订单从未离开本机）")


# ==================== 第四部分：事中 Kill Switch ====================


async def daily_loss_demo():
    exchange = MockExchange(initial_cash=100_000)
    system = TradingSystem(exchange, RiskConfig(max_daily_loss=500, max_order_volume=5_000, max_order_notional=50_000))
    start_equity = exchange.account.balance
    pnl_track = []

    async def pnl_monitor(event):
        """每笔行情重算日内盈亏 = 资金余额（含挂单冻结部分）+ 持仓市值 - 期初权益"""
        account = await system.gateway.query_account()
        mv = sum(p.volume * system.risk.last_tick[p.symbol].last_price for p in system.oms.positions.all())
        pnl = account.balance + mv - start_equity
        pnl_track.append([event.data.last_price, round(pnl, 2), system.risk.halted])
        system.risk.on_pnl(pnl)

    system.engine.register(EVENT_TICK, pnl_monitor)
    await system.start()
    await system.gateway.subscribe(SYM)
    exchange.set_market(SYM, 10.00)
    await system.engine.join()
    await system.send_order(system.new_request(SYM, Direction.LONG, 10.01, 2_000))
    resting = await system.send_order(system.new_request(SYM, Direction.LONG, 9.60, 500))  # 低位挂单
    for px in (9.90, 9.80, 9.70):
        exchange.set_market(SYM, px)
        await system.engine.join()
    await asyncio.sleep(0.05)
    await system.engine.join()
    after = await system.send_order(system.new_request(SYM, Direction.LONG, 9.71, 100))
    resting_status = system.oms.orders[resting.client_order_id].status.value
    await system.stop()
    return pnl_track, system.risk.events, resting_status, after


async def consecutive_reject_demo():
    exchange = MockExchange(initial_cash=100_000)
    system = TradingSystem(exchange, RiskConfig(max_consecutive_rejects=3))
    await system.start()
    await system.gateway.subscribe(SYM)
    exchange.set_market(SYM, 10.00)
    await system.engine.join()
    exchange.inject("reject", count=5)  # 模拟柜台连续拒单（如账户被限制、资金被冻结）
    rows = []
    for i in range(4):
        res = await system.send_order(system.new_request(SYM, Direction.LONG, 10.01, 100))
        await system.engine.join()
        rows.append([f"第 {i + 1} 笔", res.outcome, res.detail, system.risk.consecutive_rejects])
    await system.stop()
    return rows, system.risk.events


def market_data_silence_demo():
    clock = FakeClock()
    risk = RiskEngine(RiskConfig(max_tick_silence_sec=5), clock=clock)
    rows = []
    syms = [SYM, "000001.SZSE"]
    for step, (dt, updates) in enumerate([(0, syms), (3, ["000001.SZSE"]), (3, []), (6, [])]):
        clock.t += dt
        for s in updates:
            risk.update_tick(TickData(s, datetime.now(), 10.0))
        stale = risk.check_market_data(syms)
        rows.append([f"t+{clock.t - 1000:g}s", ", ".join(updates) or "（无行情）", ", ".join(stale) or "-",
                     ", ".join(risk.halted_symbols) or "-", risk.halted])
    return rows, risk


def kill_switch():
    print("\n" + "=" * 60)
    print("第四部分：事中 Kill Switch")
    print("=" * 60)

    print("\n🔴 场景 A：日内亏损超限（阈值 500 元）")
    pnl_track, events, resting_status, after = asyncio.run(daily_loss_demo())
    print(pd.DataFrame(pnl_track, columns=["最新价", "日内盈亏", "已暂停"]).to_string(index=False))
    for e in events:
        print(f"   📣 {e}")
    print(f"   低位挂单的最终状态：{resting_status}（Kill Switch 触发后自动撤单）")
    print(f"   暂停后再下单：{after.outcome} → {after.detail or '-'}")

    print("\n🔴 场景 B：连续拒单（阈值 3 笔）")
    rows, events = asyncio.run(consecutive_reject_demo())
    print(pd.DataFrame(rows, columns=["订单", "结果", "原因", "连续拒单计数"]).to_string(index=False))
    for e in events:
        print(f"   📣 {e}")

    print("\n🔴 场景 C：行情中断（静默阈值 5 秒）")
    rows, risk = market_data_silence_demo()
    print(pd.DataFrame(rows, columns=["时刻", "本轮收到行情", "停滞标的", "已暂停标的", "全局暂停"]).to_string(index=False))
    for e in risk.events:
        print(f"   📣 {e}")

    print("\n🟢 恢复交易：")
    print(f"   未对账就恢复 → {risk.resume('alice', reconciled=False)}；{risk.events[-1]}")
    print(f"   对账完成并由人工确认 → {risk.resume('alice', reconciled=True)}；{risk.events[-1]}")
    print("""
    💡 设计原则：
       - 自动"暂停"可以，自动"恢复"要谨慎：恢复必须有确认人，且先完成对账
       - Kill Switch 触发后：拒绝新单 + 撤销全部挂单 + P0 告警（第 09 课）
       - 单个标的行情停滞只暂停该标的；全部停滞多半是网关断线，升级为全局暂停
       - 非交易时段没有行情是正常的，行情中断检测要结合交易时段（第 11 课）
    """)


# ==================== 主函数 ====================


def main():
    three_lines()
    pre_trade_checks()
    pipeline_demo()
    kill_switch()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 增加一条"自成交检查"规则：同一标的存在反向挂单且价格可能撮合时拒绝新单
    2. 增加"日内撤单次数上限"：在 RiskEngine 中记录撤单次数，超限后禁止继续撤单并告警
    3. 持仓上限目前只看已成交持仓。改为"已成交 + 在途买单"，并构造一个能暴露原实现漏洞的场景
    4. 把 RiskConfig 改为从 YAML/JSON 文件加载，并记录每次修改的版本号，思考为什么风控参数需要审计
    """)


if __name__ == "__main__":
    main()
