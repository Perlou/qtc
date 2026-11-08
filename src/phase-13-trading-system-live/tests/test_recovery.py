"""幂等下单、事件日志回放、从交易所恢复、引擎异常隔离"""

import asyncio

from mini_trader import (
    OMS,
    Direction,
    EventEngine,
    MockExchange,
    MockGateway,
    OrderRequest,
    Status,
)
from mini_trader.persistence import JsonlEventLog, SqliteEventLog, rebuild_oms, record
from mini_trader.recovery import recover_from_exchange, reconcile_positions, send_with_confirm

SYM = "600000.SSE"


def run(coro):
    return asyncio.run(coro)


async def _setup(**gw):
    engine = EventEngine(timer_interval=None)
    ex = MockExchange()
    ex.set_market(SYM, 10.0)
    gateway = MockGateway(engine, ex, **gw)
    oms = OMS()
    engine.register("order", lambda e: oms.on_order(e.data))
    engine.register("trade", lambda e: oms.on_trade(e.data))
    engine.start()
    await gateway.connect()
    return engine, ex, gateway, oms


def _req(coid, volume=100):
    return OrderRequest(SYM, Direction.LONG, 10.01, volume, client_order_id=coid)


def test_lost_request_is_resent_once():
    async def main():
        engine, ex, gw, oms = await _setup()
        ex.inject("lost_request")
        r = _req("C1")
        oms.create_order(r)
        res = await send_with_confirm(gw, oms, r, timeout=0.2)
        await engine.join()
        await engine.stop()
        return res, ex, oms

    res, ex, oms = run(main())
    assert res.outcome == "resent" and res.attempts == 2
    assert len(ex.query_trades("C1")) == 1
    assert oms.positions.get(SYM).volume == 100


def test_lost_response_is_not_resent():
    async def main():
        engine, ex, gw, oms = await _setup()
        ex.inject("lost_response")
        r = _req("C1")
        oms.create_order(r)
        res = await send_with_confirm(gw, oms, r, timeout=0.2)
        await engine.join()
        await engine.stop()
        return res, ex, oms

    res, ex, oms = run(main())
    assert res.outcome == "confirmed_after_timeout"
    assert len(ex.query_orders()) == 1
    assert oms.orders["C1"].status is Status.ALL_TRADED
    assert oms.positions.get(SYM).volume == 100


def test_real_latency_timeout_then_query():
    async def main():
        engine, ex, gw, oms = await _setup(response_latency=0.3)
        r = _req("C1")
        oms.create_order(r)
        res = await send_with_confirm(gw, oms, r, timeout=0.1)
        await engine.join()
        await engine.stop()
        return res, ex

    res, ex = run(main())
    assert res.outcome == "confirmed_after_timeout"
    assert len(ex.query_orders()) == 1


def test_query_failure_marks_unknown():
    async def main():
        engine, ex, gw, oms = await _setup()
        ex.inject("lost_response")
        r = _req("C1")
        oms.create_order(r)

        orig = gw.query_order

        async def broken(coid):
            raise ConnectionError("查询失败")

        gw.query_order = broken
        res = await send_with_confirm(gw, oms, r, timeout=0.2)
        gw.query_order = orig
        await engine.stop()
        return res, oms

    res, oms = run(main())
    assert res.outcome == "unknown"
    assert "C1" in oms.unknown_orders


def test_reconcile_positions():
    diffs = reconcile_positions({"a|LONG": 3, "b|LONG": 1}, {"a|LONG": 2, "b|LONG": 1, "c|SHORT": 1})
    assert [d["position"] for d in diffs] == ["a|LONG", "c|SHORT"]


def test_recover_after_disconnect_with_dropped_reports():
    async def main():
        engine, ex, gw, oms = await _setup()
        ex.set_market(SYM, 10.0, ask=10.05)
        r = _req("C1", 300)
        oms.create_order(r)
        await send_with_confirm(gw, oms, r)
        await engine.join()
        ex.disconnect()
        ex.set_market(SYM, 10.0, ask=10.01)  # 断线期间成交，回报丢失
        await engine.join()
        assert oms.positions.get(SYM).volume == 0
        ex.reconnect()
        report = await recover_from_exchange(oms, gw)
        await engine.stop()
        return report, oms

    report, oms = run(main())
    assert report.diffs_before and report.trades_applied == 1
    assert report.safe_to_resume
    assert oms.orders["C1"].status is Status.ALL_TRADED
    assert oms.positions.get(SYM).volume == 300


def test_recover_detects_unexplained_diff():
    async def main():
        engine, ex, gw, oms = await _setup()
        r = _req("C1", 100)
        oms.create_order(r)
        await send_with_confirm(gw, oms, r)
        await engine.join()
        ex.adjust_position(SYM, Direction.LONG, -100)  # 交易所侧调整，无成交回报
        report = await recover_from_exchange(oms, gw)
        await engine.stop()
        return report, oms

    report, oms = run(main())
    assert not report.safe_to_resume
    assert oms.positions.get(SYM).volume == 0  # 以交易所为准


def test_replay_with_orphan_and_duplicates_is_idempotent():
    async def main():
        engine, ex, gw, oms = await _setup()
        r = _req("C1", 100)
        oms.create_order(r)
        await send_with_confirm(gw, oms, r)
        ex.submit(_req("MANUAL-1", 200))  # 别的终端下的单：孤儿
        await engine.join()
        fresh = OMS()
        rep1 = await recover_from_exchange(fresh, gw)
        rep2 = await recover_from_exchange(fresh, gw)  # 再执行一次结果不变
        await engine.stop()
        return fresh, rep1, rep2

    fresh, rep1, rep2 = run(main())
    assert rep1.orders_adopted == 2 and rep2.trades_applied == 0
    assert fresh.orders["MANUAL-1"].traded == 200
    assert fresh.positions.get(SYM).volume == 300


def test_event_log_roundtrip(tmp_path):
    for log in (JsonlEventLog(tmp_path / "ev.jsonl"), SqliteEventLog(tmp_path / "ev.db")):
        ex = MockExchange()
        live = OMS()

        def on_report(kind, data, log=log, live=live):
            if kind in ("order", "trade"):
                record(log, kind, data)
                getattr(live, f"on_{kind}")(data)

        ex.add_listener(on_report)
        ex.set_market(SYM, 10.0, ask=10.01, ask_volume=200)
        r = _req("C1", 500)
        record(log, "order_request", r)
        live.create_order(r)
        ex.submit(r)
        rebuilt = rebuild_oms(log)
        assert rebuilt.orders["C1"].status is live.orders["C1"].status is Status.PART_TRADED
        assert rebuilt.positions.to_volume_map() == live.positions.to_volume_map()


def test_jsonl_torn_tail_is_repaired_on_open(tmp_path):
    path = tmp_path / "ev.jsonl"
    log = JsonlEventLog(path)
    log.append("order_request", {"a": 1})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "type": "tra')  # 崩溃时写了一半

    reopened = JsonlEventLog(path)
    assert reopened.repaired_bytes > 0
    reopened.append("trade", {"b": 2})  # 下一条记录不能被残缺内容污染
    assert [r["type"] for r in JsonlEventLog(path).replay()] == ["order_request", "trade"]


def test_recovery_marks_missing_acked_order_cancelled_not_crash():
    """已确认的挂单在交易所查不到（如隔夜失效）：不能因为状态机抛异常而中断恢复"""

    async def main():
        engine, ex, gw, oms = await _setup()
        ex.set_market(SYM, 10.0, ask=10.05)
        r = _req("C1", 100)
        oms.create_order(r)
        await send_with_confirm(gw, oms, r)
        await engine.join()
        await gw.close()  # 日终失效的回报收不到
        ex.settle()
        ex.set_market(SYM, 10.0)
        report = await recover_from_exchange(oms, gw)
        await engine.stop()
        return report, oms

    report, oms = run(main())
    assert report.orders_missing == 1
    assert oms.orders["C1"].status is Status.CANCELLED


def test_startup_recovery_does_not_cancel_resting_orders():
    from mini_trader.system import TradingSystem

    async def main():
        ex = MockExchange()
        ex.set_market(SYM, 10.0, ask=10.05)
        ex.submit(_req("C1", 100))  # 活动挂单，属于上一个进程
        system = TradingSystem(ex, timer_interval=None)
        system.risk.halt("启动恢复中", notify=False)
        await system.start()
        report = await recover_from_exchange(system.oms, system.gateway)
        await system.stop()
        return ex, report

    ex, report = run(main())
    assert ex.query_order("C1").status is Status.NOT_TRADED
    assert report.orders_adopted == 1


def test_recovery_written_to_log_survives_second_crash(tmp_path):
    async def main():
        engine, ex, gw, oms = await _setup()
        log = JsonlEventLog(tmp_path / "ev.jsonl")
        ex.set_market(SYM, 10.0, ask=10.05)
        r = _req("C1", 300)
        record(log, "order_request", r)
        oms.create_order(r)
        await send_with_confirm(gw, oms, r)
        await engine.join()
        ex.set_market(SYM, 10.0, ask=10.01)  # 假设这段时间进程已死，回报没有写进日志
        await engine.join()
        crashed = rebuild_oms(log)
        report = await recover_from_exchange(crashed, gw, event_log=log)
        await engine.stop()
        return rebuild_oms(log), crashed, report

    second, first, report = run(main())
    assert report.trades_applied == 1
    assert second.orders["C1"].status is Status.ALL_TRADED
    assert second.positions.to_volume_map() == first.positions.to_volume_map() == {f"{SYM}|LONG": 300.0}


def test_unknown_send_pauses_symbol():
    from mini_trader.system import TradingSystem

    async def main():
        ex = MockExchange()
        system = TradingSystem(ex, timer_interval=None, order_timeout=0.05)
        await system.start()
        await system.gateway.subscribe(SYM)
        ex.set_market(SYM, 10.0)
        await system.engine.join()

        async def broken(coid):
            raise ConnectionError("查询失败")

        system.gateway.query_order = broken
        ex.inject("lost_response")
        first = await system.send_order(system.new_request(SYM, Direction.LONG, 10.01, 100))
        second = await system.send_order(system.new_request(SYM, Direction.LONG, 10.01, 100))
        await system.stop()
        return first, second

    first, second = run(main())
    assert first.outcome == "unknown"
    assert second.outcome == "risk_rejected" and "symbol_halt" in second.detail


def test_engine_isolates_handler_errors():
    async def main():
        engine = EventEngine(timer_interval=None)
        got = []

        def bad(event):
            raise ValueError("boom")

        engine.register("tick", bad)
        engine.register("tick", lambda e: got.append(e.data))
        engine.register("error", lambda e: got.append("error"))
        engine.start()
        engine.put("tick", 1)
        await engine.join()
        await engine.join()
        await engine.stop()
        return got, engine

    got, engine = run(main())
    assert 1 in got and "error" in got and len(engine.errors) == 1


def test_engine_threadsafe_put():
    import threading

    async def main():
        engine = EventEngine(timer_interval=None)
        got = []
        engine.register("tick", lambda e: got.append(e.data))
        engine.start()
        t = threading.Thread(target=lambda: [engine.put_threadsafe("tick", i) for i in range(50)])
        t.start()
        t.join()
        await asyncio.sleep(0.05)
        await engine.join()
        await engine.stop()
        return got

    assert run(main()) == list(range(50))
