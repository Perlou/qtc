"""
对账、幂等与故障恢复
====================

学习目标：
    1. 理解本地状态与交易所不一致的原因，实现持仓与挂单对账
    2. 解决"下单超时"难题：先查询再决定是否重发，全程复用同一个 client_order_id
    3. 在 MockExchange 上逐一注入故障并恢复：重复回报、回报丢失、孤儿订单、交易所侧调整
    4. 用 append-only 事件日志（JSONL / SQLite）记录并回放状态，理解"先写日志再发请求"
    5. 模拟进程被 kill -9：重启后从事件日志和交易所恢复订单与持仓

核心概念：
    - 交易所是真相来源：本地只是缓存，银行流水才是真相
    - 幂等：同一个意图无论重试多少次，效果都只发生一次
    - Write-Ahead：订单意图在发出之前先落盘
    - 宁停勿错：无法解释的差异 → 保持暂停，P0 告警，等待人工

前置知识：
    - 03 课（OMS）、04 课（Kill Switch）
    - 本阶段 CONCEPT.md 第 11 节

环境要求：
    - 不需要网络；事件日志写到 outputs/phase-13/events/
"""

import asyncio
import inspect
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import OMS, Direction, EventEngine, MockExchange, MockGateway, OrderRequest, RiskConfig
from mini_trader.events import EVENT_ORDER, EVENT_TRADE
from mini_trader.persistence import JsonlEventLog, SqliteEventLog, rebuild_oms, record
from mini_trader.recovery import (
    positions_to_map,
    reconcile_orders,
    reconcile_positions,
    recover_from_exchange,
    send_with_confirm,
)
from mini_trader.system import TradingSystem
from utils import OUTPUT_DIR

load_dotenv()

SYM = "600000.SSE"
EVENT_DIR = OUTPUT_DIR / "phase-13" / "events"


@dataclass
class Env:
    """一套最小环境：交易所 + 网关 + 事件引擎 + OMS"""

    engine: EventEngine
    exchange: MockExchange
    gateway: MockGateway
    oms: OMS


async def make_env(gateway_cls=MockGateway, **gw_kwargs) -> Env:
    engine = EventEngine(timer_interval=None)
    exchange = MockExchange()
    exchange.set_market(SYM, 10.00)
    gateway = gateway_cls(engine, exchange, **gw_kwargs)
    oms = OMS()
    engine.register(EVENT_ORDER, lambda e: oms.on_order(e.data))
    engine.register(EVENT_TRADE, lambda e: oms.on_trade(e.data))
    engine.start()
    await gateway.connect()
    return Env(engine, exchange, gateway, oms)


def buy(coid: str, volume: float = 100, price: float = 10.01) -> OrderRequest:
    return OrderRequest(SYM, Direction.LONG, price, volume, client_order_id=coid)


def pos(exchange: MockExchange) -> float:
    return exchange.positions.get(SYM).volume


# ==================== 第一部分：为什么要对账 ====================


def why_reconcile():
    print("=" * 60)
    print("第一部分：为什么要对账")
    print("=" * 60)
    print("""
    📒 比喻：你的记账本（本地状态）和银行流水（交易所状态）必须定期核对。
       记账本可能漏记、重复记、记错，银行流水才是真相。

    不一致的常见原因：回报丢失、重复处理、崩溃前未持久化、人工在其他终端下单、
    交易所侧的调整（分红送股、强平）。
    """)
    print(pd.DataFrame(
        [
            ["持仓", "盘中定时 + 启动时 + 日终", "暂停相关标的 → 以交易所为准修正 → 告警"],
            ["挂单", "盘中定时 + 启动时", "本地有交易所无 → 查最终状态；交易所有本地无 → 孤儿订单"],
            ["成交", "日终", "按成交编号比对，补录缺失成交"],
            ["资金", "日终", "核对权益、手续费、出入金"],
        ],
        columns=["对账对象", "频率", "差异处理"],
    ).to_string(index=False))

    local = {"rb2501.SHFE|LONG": 3, "600000.SSE|LONG": 1000}
    remote = {"rb2501.SHFE|LONG": 2, "600000.SSE|LONG": 1000, "IF2412.CFFEX|SHORT": 1}
    print("\n   持仓对账示例（remote 为真相）：")
    print(pd.DataFrame(reconcile_positions(local, remote)).to_string(index=False))


# ==================== 第二部分：下单超时 —— 先查询，再决定 ====================


class DropsAfterTimeout(MockGateway):
    """下单超时后网络彻底断开：后续查询也会失败"""

    async def send_order(self, req):
        try:
            return await super().send_order(req)
        except TimeoutError:
            self.exchange.disconnect()
            raise


async def case_naive_retry():
    """❌ 反例：超时后换一个新订单号直接重发"""
    env = await make_env()
    env.exchange.inject("lost_response")
    first = buy("N-1")
    try:
        await env.gateway.send_order(first)
    except TimeoutError:
        await env.gateway.send_order(replace(first, client_order_id="N-2"))
    await env.engine.stop()
    return ["❌ 超时后直接重发（新订单号）", "duplicated", 2, len(env.exchange.orders), pos(env.exchange)]


async def case_confirm(fault: str | None, gateway_cls=MockGateway, **gw) -> list:
    env = await make_env(gateway_cls, **gw)
    if fault:
        env.exchange.inject(fault)
    req = buy("C-1")
    env.oms.create_order(req)
    result = await send_with_confirm(env.gateway, env.oms, req, timeout=0.05)
    if env.exchange.connected:
        await env.engine.join()
    await env.engine.stop()
    return [result.outcome, result.attempts, len(env.exchange.orders), pos(env.exchange)]


def timeout_handling():
    print("\n" + "=" * 60)
    print("第二部分：下单超时 —— 先查询，再决定是否重发")
    print("=" * 60)
    print("""
    经典难题：请求超时了，订单可能 (a) 根本没到交易所；(b) 已到达并成交，只是回报丢了。
    情况 (b) 下直接重试，就是重复下单 —— 仓位翻倍。

          发出报单 ──▶ 超时 / 无响应
                          │
                          ▼
               ┌───────────────────────┐
               │ 用 client_order_id 查询 │
               └──────────┬────────────┘
             ┌────────────┼──────────────┐
             ▼            ▼              ▼
         查到订单       确认不存在       查询也失败
       → 按其状态更新   → 用同一 coid 重发  → 标记"未知"，暂停该标的 + 告警
    """)

    async def run_all():
        return [
            await case_naive_retry(),
            ["lost_request：请求丢失（交易所未收到）", *(await case_confirm("lost_request"))],
            ["lost_response：响应丢失（交易所已处理）", *(await case_confirm("lost_response"))],
            ["真实超时：响应耗时超过 timeout", *(await case_confirm(None, response_latency=0.2))],
            ["响应丢失且随后断网：查询也失败", *(await case_confirm("lost_response", DropsAfterTimeout))],
        ]

    rows = asyncio.run(run_all())
    df = pd.DataFrame(rows, columns=["场景（目标：买入 100 股）", "结果", "发送次数", "交易所订单数", "交易所持仓"])
    print(df.to_string(index=False))
    print("""
    💡 解读：
       - 反例中交易所收到两笔订单，持仓是目标的两倍
       - 三种"先查后发"的场景，交易所持仓都正好等于目标
       - 查询失败时结果是 unknown：既不重发也不假定成功，先暂停该标的，等网络恢复后由第三部分的恢复流程解决

    幂等实现要点：
       1. 每个意图在发出之前生成全局唯一的 client_order_id，并先持久化（Write-Ahead）
       2. 重试复用同一个 client_order_id；很多交易所会拒绝重复订单号（第 08 课）
       3. 成交按 trade_id 去重，持仓更新才是幂等的
    """)


# ==================== 第三部分：故障注入与恢复 ====================


async def fault_replay() -> list:
    """重连后柜台重推当日回报：持仓不能翻倍"""
    env = await make_env()
    req = buy("R-1", 300)
    env.oms.create_order(req)
    await send_with_confirm(env.gateway, env.oms, req)
    await env.engine.join()
    env.exchange.disconnect()
    env.exchange.reconnect(replay=True)
    await env.engine.join()
    await env.engine.stop()
    return ["重连后回报重推", f"去重 {env.oms.stats['duplicate_trades']} 笔成交", f"{env.oms.positions.get(SYM).volume:g}", f"{pos(env.exchange):g}", "-"]


async def fault_lost_reports() -> list:
    """断线期间挂单成交，回报丢失：对账发现差异 → 补录成交"""
    env = await make_env()
    env.exchange.set_market(SYM, 10.00, ask=10.05)
    req = buy("L-1", 300, 10.02)
    env.oms.create_order(req)
    await send_with_confirm(env.gateway, env.oms, req)
    await env.engine.join()
    env.exchange.disconnect()
    env.exchange.set_market(SYM, 10.00, ask=10.01)  # 断线期间成交
    env.exchange.reconnect()
    before = env.oms.positions.get(SYM).volume
    report = await recover_from_exchange(env.oms, env.gateway)
    await env.engine.stop()
    return ["断线期间成交（回报丢失）", f"本地 {before:g} → 补录 {report.trades_applied} 笔成交",
            f"{env.oms.positions.get(SYM).volume:g}", f"{pos(env.exchange):g}", report.safe_to_resume]


async def fault_orphan() -> list:
    """人工在其他终端下单：本地没有记录"""
    env = await make_env()
    env.exchange.set_market(SYM, 10.00, ask=10.05)
    req = buy("O-1", 100, 10.02)
    env.oms.create_order(req)
    await send_with_confirm(env.gateway, env.oms, req)
    await env.engine.join()
    await env.gateway.close()  # 人工单走的是券商终端，本系统收不到它的回报
    env.exchange.submit(buy("MANUAL-1", 200, 10.00))
    check = reconcile_orders(env.oms.active_orders(), env.exchange.query_open_orders())
    report = await recover_from_exchange(env.oms, env.gateway)
    await env.engine.stop()
    return ["孤儿订单（人工下单）", f"发现 {check['orphans']} → 收编 {report.orders_adopted} 笔",
            f"{env.oms.positions.get(SYM).volume:g}", f"{pos(env.exchange):g}", report.safe_to_resume]


async def fault_adjustment() -> tuple[list, list]:
    """交易所侧调整（如强平、送股）：没有成交回报，无法用补录解释"""
    env = await make_env()
    req = buy("A-1", 100)
    env.oms.create_order(req)
    await send_with_confirm(env.gateway, env.oms, req)
    await env.engine.join()
    env.exchange.adjust_position(SYM, Direction.LONG, +10)  # 例如送股
    report = await recover_from_exchange(env.oms, env.gateway)
    await env.engine.stop()
    row = ["交易所侧持仓调整（无成交）", f"无法解释的差异 {len(report.diffs_unexplained)} 项",
           f"{env.oms.positions.get(SYM).volume:g}", f"{pos(env.exchange):g}", report.safe_to_resume]
    return row, report.diffs_unexplained


def fault_injection():
    print("\n" + "=" * 60)
    print("第三部分：故障注入与恢复（把故障当测试用例）")
    print("=" * 60)
    print("\n📄 恢复流程的核心步骤（mini_trader/recovery.py）：\n")
    print(inspect.getdoc(recover_from_exchange))

    async def run_all():
        return [await fault_replay(), await fault_lost_reports(), await fault_orphan(), *(await fault_adjustment())]

    r1, r2, r3, r4, unexplained = asyncio.run(run_all())
    print()
    print(pd.DataFrame([r1, r2, r3, r4], columns=["故障", "处理过程", "本地持仓", "交易所持仓", "可恢复交易"]).to_string(index=False))
    print(f"\n   无法解释的差异明细：{unexplained}")
    print("""
    💡 恢复的分界线：
       - 差异能被补录的成交解释（回报丢失）→ 自动修复，确认后可恢复交易
       - 差异无法解释（送股、强平、人工操作）→ 本地持仓以交易所为准，但 safe_to_resume=False：
         保持暂停 + P0 告警，由人查明原因后再恢复。这就是"宁停勿错"。
    """)


# ==================== 第四部分：事件日志与崩溃重启 ====================


async def run_until_crash(log: JsonlEventLog):
    """第一次运行：交易一段时间后进程被杀"""
    exchange = MockExchange()
    exchange.set_market(SYM, 10.00)
    exchange.submit(buy("OLD-1", 1000))  # 昨日持仓
    exchange.settle()

    system = TradingSystem(exchange, RiskConfig(max_order_notional=100_000), timer_interval=None, event_log=log)
    await system.start()
    await system.gateway.subscribe(SYM)
    snapshot = await system.gateway.query_positions()  # 盘前：以交易所持仓为起点，写入快照
    system.oms.positions.load(snapshot)
    record(log, "positions_snapshot", snapshot)
    exchange.set_market(SYM, 10.00)
    await system.engine.join()

    await system.send_order(system.new_request(SYM, Direction.LONG, 10.01, 500))  # 成交
    resting = await system.send_order(system.new_request(SYM, Direction.LONG, 9.90, 300))  # 挂单
    await system.send_order(system.new_request(SYM, Direction.SHORT, 10.30, 200))  # 挂单
    await system.engine.join()
    wal_req = system.new_request(SYM, Direction.LONG, 10.01, 100)
    record(log, "order_request", wal_req)  # 已写日志，还没来得及发出：进程在这里被杀

    # 💥 kill -9：内存状态全部丢失，没有机会做任何清理
    await system.engine.stop()
    await system.gateway.close()
    return exchange, resting.client_order_id, wal_req.client_order_id


async def restart(log: JsonlEventLog, exchange: MockExchange):
    """第二次运行：从事件日志和交易所恢复"""
    system = TradingSystem(exchange, timer_interval=None, event_log=log)
    system.oms = rebuild_oms(log)  # 步骤 1：回放本地日志（仅作参考）
    local_view = system.oms.positions.to_volume_map()
    system.risk.halt("启动恢复中", notify=False)  # 恢复完成前禁止新单，但不动现有挂单
    await system.start()  # 步骤 2：连接网关
    await system.gateway.subscribe(SYM)
    report = await recover_from_exchange(system.oms, system.gateway, event_log=log)  # 步骤 3-5
    resumed = system.risk.resume("值班人员 alice", reconciled=report.safe_to_resume)  # 步骤 6
    await system.stop()
    return system, report, local_view, resumed


def event_log_and_crash():
    print("\n" + "=" * 60)
    print("第四部分：事件日志与崩溃重启（kill -9 演练）")
    print("=" * 60)
    if EVENT_DIR.exists():
        shutil.rmtree(EVENT_DIR)
    log = JsonlEventLog(EVENT_DIR / "10_events.jsonl")

    exchange, resting, wal = asyncio.run(run_until_crash(log))
    print(f"   💥 进程被杀。事件日志已落盘 {sum(1 for _ in log.replay())} 条，订单 {wal} 只写了日志、没有发出")

    # 进程死亡期间，世界没有停止
    exchange.set_market(SYM, 9.85, bid=9.84, ask=9.89)  # 买入挂单在此成交
    exchange.submit(buy("MANUAL-1", 100, 9.80))  # 值班人员在券商终端手动下单
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 999, "type": "tra')  # 崩溃时写了一半的最后一行
    print(f"   🌍 进程死亡期间：挂单 {resting} 成交；人工下了一笔单 MANUAL-1；日志末尾留下半行残缺记录")

    log = JsonlEventLog(log.path)  # 新进程重新打开日志：截断残缺的最后一行
    system, report, local_view, resumed = asyncio.run(restart(log, exchange))
    remote = positions_to_map(exchange.query_positions())
    print(f"""
    🔄 启动恢复流程：
       1. 打开日志时截断残缺尾部 {log.repaired_bytes} 字节；回放事件日志 → 本地持仓 {local_view}
       2. 连接网关，查询交易所订单 / 成交 / 持仓
       3. 恢复前差异    → {report.diffs_before}
       4. 更新订单 {report.orders_updated} 笔，收编孤儿 {report.orders_adopted} 笔，
          标记失效 {report.orders_missing} 笔，补录成交 {report.trades_applied} 笔
       5. 补录后仍无法解释的差异 → {report.diffs_unexplained or '无'}
       6. 人工确认恢复交易 → {resumed}
    """)
    rows = [
        [o.client_order_id[-12:], o.direction.value, f"{o.volume:g}", f"{o.traded:g}", o.status.value, o.reason or "-"]
        for o in system.oms.orders.values()
    ]
    print(pd.DataFrame(rows, columns=["订单", "方向", "委托", "成交", "恢复后状态", "备注"]).to_string(index=False))
    print(f"\n   本地持仓 {system.oms.positions.to_volume_map()} vs 交易所 {remote}：{'一致 ✅' if system.oms.positions.to_volume_map() == remote else '不一致 ❌'}")
    print(f"   孤儿订单已纳入管理：{sorted(system.oms.orphans)}（下一步：撤销或由人工确认保留）")

    sqlite_path = EVENT_DIR / "10_events.db"
    sq = SqliteEventLog(sqlite_path)
    for rec in log.replay():
        sq.append(rec["type"], rec["payload"])
    counts = sq.db.execute("SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY type").fetchall()
    sq.close()
    print(f"\n   同一批事件写入 SQLite（{sqlite_path.name}），可以直接用 SQL 查询：{dict(counts)}")
    print("   JSONL 适合顺序追加和人工查看；SQLite 适合按类型 / 时间查询。回放时都按写入顺序重建状态。")
    print("""
    💡 关键设计：
       - Write-Ahead：先写 order_request 再发单。崩溃后才知道"我曾经想发哪些单"，
         否则像上面只写了日志的订单，就会变成一个说不清的悬案
       - 日志只追加不修改：可审计、可回放，也是事后排查问题的依据
       - 恢复期间风控保持暂停，恢复完成并确认后才放行
       - 策略状态同样要持久化。更稳健的做法是用"目标持仓"表达意图：
         恢复时只比较"目标 vs 实际"，天然幂等
    """)


# ==================== 第五部分：故障演练清单 ====================


def drill_table():
    print("\n" + "=" * 60)
    print("第五部分：故障演练清单")
    print("=" * 60)
    print(pd.DataFrame(
        [
            ["网络断开 30 秒", "暂停交易 → 自动重连 → 对账 → 恢复", "断开网卡或用防火墙阻断", "第三部分"],
            ["进程被 kill -9", "重启后从交易所恢复持仓与挂单", "运行中直接杀进程", "第四部分"],
            ["回报重复推送", "成交去重，持仓不变", "注入重复事件", "第三部分"],
            ["下单超时", "查询确认，不重复下单", "模拟网关延迟 / 丢包", "第二部分"],
            ["行情停滞", "告警并暂停该标的", "停止行情推送", "第 04、09 课"],
            ["交易所拒单", "进入 REJECTED，风控统计拒单", "下一个超出涨跌停的单", "第 03、04 课"],
            ["磁盘写满", "告警，停止交易（无法持久化就不能交易）", "限制容器磁盘", "练习 4"],
        ],
        columns=["场景", "预期行为", "如何演练", "本课程覆盖"],
    ).to_string(index=False))
    print("""
    📌 "重启后能从交易所状态恢复持仓"和"风控能在异常时自动暂停交易"的验证方法，
       就是把故障当测试用例，主动制造它。tests/test_recovery.py 把这些场景固化成了自动化测试。
    """)


# ==================== 主函数 ====================


def main():
    why_reconcile()
    timeout_handling()
    fault_injection()
    event_log_and_crash()
    drill_table()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 recover_from_exchange 接到定时器上，实现盘中每 N 分钟自动对账；发现差异时暂停相关标的
    2. 增加"日终成交对账"：把本地成交和交易所成交按 trade_id 比对，输出缺失与多余的记录
    3. 给策略增加状态持久化：把目标持仓写入事件日志，重启后计算"目标 vs 实际"并补单
    4. 模拟磁盘写满：让 event_log.append 抛出 OSError，修改 send_order 使其拒绝发单并触发 Kill Switch
    """)


if __name__ == "__main__":
    main()
