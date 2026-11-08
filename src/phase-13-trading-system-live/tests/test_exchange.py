"""模拟交易所：撮合、部分成交、期货开平今昨、故障注入"""

import pytest

from mini_trader import (
    Direction,
    DuplicateOrderError,
    ExchangeDisconnected,
    ExchangeTimeout,
    MockExchange,
    Offset,
    OrderRequest,
    Status,
)

FUT = "rb2501.SHFE"
DCE = "m2501.DCE"


def req(coid, direction=Direction.LONG, price=10.0, volume=100, symbol="600000.SSE", offset=Offset.NONE):
    return OrderRequest(symbol, direction, price, volume, offset, coid)


def test_partial_fill_by_depth():
    ex = MockExchange()
    ex.set_market("600000.SSE", 10.0, ask=10.01, ask_volume=200)
    ack = ex.submit(req("C1", price=10.01, volume=500))
    assert ack.status is Status.PART_TRADED and ack.traded == 200
    ex.set_market("600000.SSE", 10.0, ask=10.01, ask_volume=1000)
    assert ex.query_order("C1").status is Status.ALL_TRADED


def test_cancel_race_after_fill():
    ex = MockExchange()
    events = []
    ex.add_listener(lambda k, d: events.append(k))
    ex.set_market("600000.SSE", 10.0)
    ex.submit(req("C1", price=10.01))
    assert ex.cancel("C1") is False
    assert "cancel_reject" in events


def test_limit_and_funds_rejects():
    ex = MockExchange(initial_cash=1000)
    ex.set_market("600000.SSE", 10.0, limit_up=11.0, limit_down=9.0)
    assert "涨停" in ex.submit(req("C1", price=11.5)).reason
    assert "资金不足" in ex.submit(req("C2", price=10.0, volume=200)).reason
    assert "整数倍" in ex.submit(req("C3", price=10.0, volume=50)).reason


def test_shfe_close_today_vs_yesterday():
    ex = MockExchange()
    ex.set_market(FUT, 3500)
    ex.submit(req("O1", price=3501, volume=2, symbol=FUT, offset=Offset.OPEN))
    ex.settle()  # 2 手变成昨仓
    ex.set_market(FUT, 3500)
    ex.submit(req("O2", price=3501, volume=1, symbol=FUT, offset=Offset.OPEN))  # 1 手今仓
    p = ex.positions.get(FUT, Direction.LONG)
    assert (p.volume, p.yd_volume, p.td_volume) == (3, 2, 1)
    # 平今 2 手：今仓只有 1 手 → 拒单
    bad = ex.submit(req("C1", Direction.SHORT, 3499, 2, FUT, Offset.CLOSE_TODAY))
    assert bad.status is Status.REJECTED
    ok = ex.submit(req("C2", Direction.SHORT, 3499, 1, FUT, Offset.CLOSE_TODAY))
    assert ok.status is Status.ALL_TRADED
    assert (p.volume, p.yd_volume) == (2, 2)
    ex.submit(req("C3", Direction.SHORT, 3499, 2, FUT, Offset.CLOSE_YESTERDAY))
    assert p.volume == 0


def test_non_shfe_close_uses_yesterday_first():
    ex = MockExchange()
    ex.set_market(DCE, 3000)
    ex.submit(req("O1", price=3001, volume=1, symbol=DCE, offset=Offset.OPEN))
    ex.settle()
    ex.set_market(DCE, 3000)
    ex.submit(req("O2", price=3001, volume=1, symbol=DCE, offset=Offset.OPEN))
    ex.submit(req("C1", Direction.SHORT, 2999, 1, DCE, Offset.CLOSE))
    p = ex.positions.get(DCE, Direction.LONG)
    assert (p.volume, p.yd_volume) == (1, 0)


def test_futures_requires_offset():
    ex = MockExchange()
    ex.set_market(FUT, 3500)
    assert "开平" in ex.submit(req("O1", price=3501, volume=1, symbol=FUT)).reason


def test_lost_request_vs_lost_response():
    ex = MockExchange()
    ex.set_market("600000.SSE", 10.0)
    ex.inject("lost_request")
    with pytest.raises(ExchangeTimeout):
        ex.submit(req("C1"))
    assert ex.query_order("C1") is None
    ex.inject("lost_response")
    with pytest.raises(ExchangeTimeout):
        ex.submit(req("C2", price=10.01))
    assert ex.query_order("C2").status is Status.ALL_TRADED


def test_duplicate_coid_rejected_by_exchange():
    ex = MockExchange()
    ex.set_market("600000.SSE", 10.0)
    ex.submit(req("C1"))
    with pytest.raises(DuplicateOrderError):
        ex.submit(req("C1"))


def test_disconnect_drops_reports_but_keeps_matching():
    ex = MockExchange()
    got = []
    ex.add_listener(lambda k, d: got.append(k))
    ex.set_market("600000.SSE", 10.0, ask=10.05)
    ex.submit(req("C1", price=10.02))  # 挂单
    got.clear()
    ex.disconnect()
    with pytest.raises(ExchangeDisconnected):
        ex.submit(req("C2"))
    ex.set_market("600000.SSE", 10.0, ask=10.01)  # 断线期间成交
    assert got == [] and ex.dropped_reports >= 2
    ex.reconnect(replay=True)
    assert "trade" in got
    assert ex.query_order("C1").status is Status.ALL_TRADED
