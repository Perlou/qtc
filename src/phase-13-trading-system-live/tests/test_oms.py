"""订单状态机、成交去重、持仓推导"""

import pytest

from mini_trader import (
    OMS,
    Direction,
    InvalidTransition,
    MockExchange,
    Offset,
    OrderData,
    OrderRequest,
    OverfillError,
    Status,
    TradeData,
    check_transition,
)

SYM = "600000.SSE"


def make_order(coid="C1", volume=500, direction=Direction.LONG, symbol=SYM, offset=Offset.NONE, price=10.0):
    return OrderRequest(symbol, direction, price, volume, offset, coid)


def report(coid, status, volume=500, traded=0, symbol=SYM, direction=Direction.LONG):
    return OrderData(coid, symbol, direction, 10.0, volume, traded=traded, status=status, exchange_order_id="EX1")


def trade(tid, coid="C1", volume=100, price=10.0, direction=Direction.LONG, symbol=SYM, offset=Offset.NONE):
    return TradeData(tid, coid, symbol, direction, price, volume, offset)


@pytest.mark.parametrize(
    "old,new",
    [
        (Status.SUBMITTING, Status.NOT_TRADED),
        (Status.SUBMITTING, Status.REJECTED),
        (Status.NOT_TRADED, Status.PART_TRADED),
        (Status.PART_TRADED, Status.PART_TRADED),
        (Status.PART_TRADED, Status.CANCELLED),
        (Status.NOT_TRADED, Status.ALL_TRADED),
    ],
)
def test_legal_transitions(old, new):
    check_transition(old, new)


@pytest.mark.parametrize(
    "old,new",
    [
        (Status.ALL_TRADED, Status.CANCELLED),
        (Status.CANCELLED, Status.NOT_TRADED),
        (Status.REJECTED, Status.NOT_TRADED),
        (Status.PART_TRADED, Status.NOT_TRADED),
        (Status.NOT_TRADED, Status.SUBMITTING),
        (Status.NOT_TRADED, Status.REJECTED),
    ],
)
def test_illegal_transitions_raise(old, new):
    with pytest.raises(InvalidTransition):
        check_transition(old, new)


def test_partial_fill_then_cancel_keeps_traded():
    oms = OMS()
    oms.create_order(make_order())
    oms.on_order(report("C1", Status.NOT_TRADED))
    oms.on_trade(trade("T1", volume=200))
    assert oms.orders["C1"].status is Status.PART_TRADED
    oms.on_order(report("C1", Status.CANCELLED, traded=200))
    o = oms.orders["C1"]
    assert o.status is Status.CANCELLED and o.traded == 200
    assert oms.positions.get(SYM).volume == 200


def test_duplicate_trade_is_ignored():
    oms = OMS()
    oms.create_order(make_order())
    assert oms.on_trade(trade("T1", volume=200)) is True
    assert oms.on_trade(trade("T1", volume=200)) is False
    assert oms.positions.get(SYM).volume == 200
    assert oms.stats["duplicate_trades"] == 1


def test_trade_before_ack_and_stale_report():
    oms = OMS()
    oms.create_order(make_order())
    oms.on_trade(trade("T1", volume=500))  # 成交先到
    assert oms.orders["C1"].status is Status.ALL_TRADED
    assert oms.on_order(report("C1", Status.NOT_TRADED)) is False  # 迟到的确认被忽略
    assert oms.orders["C1"].status is Status.ALL_TRADED
    assert oms.stats["stale_reports"] == 1


def test_late_trade_after_cancel_updates_position_not_status():
    oms = OMS()
    oms.create_order(make_order())
    oms.on_order(report("C1", Status.NOT_TRADED))
    oms.on_order(report("C1", Status.CANCELLED))
    oms.on_trade(trade("T9", volume=100))
    assert oms.orders["C1"].status is Status.CANCELLED
    assert oms.orders["C1"].traded == 100
    assert oms.positions.get(SYM).volume == 100
    assert oms.stats["late_trades"] == 1


def test_overfill_raises():
    oms = OMS()
    oms.create_order(make_order(volume=100))
    oms.on_trade(trade("T1", volume=100))
    with pytest.raises(OverfillError):
        oms.on_trade(trade("T2", volume=100))


def test_illegal_report_is_recorded_as_anomaly():
    seen = []
    oms = OMS(on_anomaly=seen.append)
    oms.create_order(make_order())
    oms.on_order(report("C1", Status.NOT_TRADED))
    oms.on_order(report("C1", Status.REJECTED))  # 已报后拒单：不在转换表中
    assert oms.orders["C1"].status is Status.NOT_TRADED
    assert seen and "非法状态转换" in seen[0]


def test_duplicate_client_order_id_rejected_locally():
    oms = OMS()
    oms.create_order(make_order())
    with pytest.raises(ValueError):
        oms.create_order(make_order())


def test_t_plus_one_and_freeze():
    ex = MockExchange()
    oms = OMS()
    ex.add_listener(lambda k, d: oms.on_order(d) if k == "order" else oms.on_trade(d) if k == "trade" else None)
    ex.set_market(SYM, 10.0)
    oms.create_order(make_order("B1", volume=1000, price=10.01))
    ex.submit(make_order("B1", volume=1000, price=10.01))
    assert oms.positions.get(SYM).volume == 1000
    assert oms.positions.available(SYM) == 0  # T+1：今日买入不可卖
    sell = OrderRequest(SYM, Direction.SHORT, 9.0, 100, client_order_id="S1")
    assert ex.submit(sell).status is Status.REJECTED
    ex.settle()
    oms.positions.settle()
    assert oms.positions.available(SYM) == 1000
    oms.create_order(OrderRequest(SYM, Direction.SHORT, 20.0, 300, client_order_id="S2"))
    assert oms.positions.available(SYM) == 700  # 挂单冻结
    oms.on_order(report("S2", Status.CANCELLED, volume=300, direction=Direction.SHORT))
    assert oms.positions.available(SYM) == 1000  # 撤单释放


def test_position_average_cost_and_pnl():
    oms = OMS()
    sym = "rb2501.SHFE"
    for coid, d, off, px, v in [
        ("O1", Direction.LONG, Offset.OPEN, 3500, 2),
        ("O2", Direction.LONG, Offset.OPEN, 3520, 2),
    ]:
        oms.create_order(OrderRequest(sym, d, px, v, off, coid))
        oms.on_trade(TradeData("T" + coid, coid, sym, d, px, v, off))
    p = oms.positions.get(sym, Direction.LONG)
    assert p.volume == 4 and p.price == pytest.approx(3510)
