"""风控：事前规则与 Kill Switch"""

from datetime import datetime

import pytest

from mini_trader import (
    AccountData,
    Direction,
    OrderData,
    OrderRequest,
    PositionBook,
    RiskConfig,
    RiskEngine,
    Status,
    TickData,
)

SYM = "600000.SSE"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def risk():
    clock = FakeClock()
    r = RiskEngine(RiskConfig(max_order_volume=1000, max_order_notional=50_000, max_position=2000), clock=clock)
    r.update_tick(TickData(SYM, datetime.now(), 10.0, 9.99, 10.01, limit_up=11.0, limit_down=9.0))
    r.clock_obj = clock
    return r


def buy(price=10.0, volume=100, symbol=SYM):
    return OrderRequest(symbol, Direction.LONG, price, volume, client_order_id="X")


@pytest.mark.parametrize(
    "request_,rule",
    [
        (buy(volume=5000), "volume"),
        (buy(symbol="000001.SZSE"), "price_reference"),
        (buy(price=11.2), "price_limit"),
        (buy(price=10.8), "price_deviation"),
        (buy(price=10.0, volume=900, symbol=SYM), None),
    ],
)
def test_pre_trade_rules(risk, request_, rule):
    res = risk.check_order(request_)
    assert res.rule == (rule or "")
    assert res.ok is (rule is None)


def test_notional_funds_position(risk):
    risk.cfg.max_order_volume = 100_000
    assert risk.check_order(buy(volume=6000)).rule == "notional"
    acct = AccountData("A", balance=1000)
    assert risk.check_order(buy(volume=200), account=acct).rule == "funds"
    book = PositionBook()
    book.get(SYM).volume = 1950
    assert risk.check_order(buy(volume=100), positions=book).rule == "position_limit"


def test_available_position_t_plus_one(risk):
    book = PositionBook()
    p = book.get(SYM)
    p.volume, p.yd_volume = 1000, 0
    sell = OrderRequest(SYM, Direction.SHORT, 10.0, 100, client_order_id="S")
    assert risk.check_order(sell, positions=book).rule == "available_position"


def test_order_rate_window(risk):
    for _ in range(5):
        assert risk.check_order(buy()).ok
    assert risk.check_order(buy()).rule == "order_rate"
    risk.clock_obj.t += 1.5
    assert risk.check_order(buy()).ok


def test_kill_switch_daily_loss_and_manual_resume(risk):
    halted = []
    risk.halt_listeners.append(halted.append)
    risk.on_pnl(-25_000)
    assert risk.halted and halted
    assert risk.check_order(buy()).rule == "kill_switch"
    assert risk.resume("", reconciled=True) is False
    assert risk.resume("alice", reconciled=False) is False
    assert risk.resume("alice", reconciled=True) is True
    assert risk.check_order(buy()).ok


def rejected(coid):
    return OrderData(coid, SYM, Direction.LONG, 10, 100, status=Status.REJECTED, reason="资金不足")


def test_consecutive_rejects(risk):
    ok = OrderData("OK", SYM, Direction.LONG, 10, 100, status=Status.NOT_TRADED)
    risk.on_order_update(rejected("R1"))
    risk.on_order_update(rejected("R2"))
    risk.on_order_update(ok)  # 中间成功一次 → 计数清零
    risk.on_order_update(rejected("R3"))
    assert not risk.halted
    risk.on_order_update(rejected("R4"))
    risk.on_order_update(rejected("R5"))
    assert risk.halted


def test_repeated_reject_report_counts_once(risk):
    for _ in range(5):
        risk.on_order_update(rejected("R1"))
    assert risk.consecutive_rejects == 1 and not risk.halted


def test_halt_without_notify_does_not_trigger_listeners(risk):
    called = []
    risk.halt_listeners.append(called.append)
    risk.halt("启动恢复中", notify=False)
    assert risk.halted and not called
    assert risk.check_order(buy()).rule == "kill_switch"


def test_market_data_silence(risk):
    risk.update_tick(TickData("000001.SZSE", datetime.now(), 12.0))
    risk.clock_obj.t = 3.0
    risk.update_tick(TickData("000001.SZSE", datetime.now(), 12.0))
    risk.clock_obj.t = 6.0
    assert risk.check_market_data() == [SYM]
    assert SYM in risk.halted_symbols and not risk.halted
    assert risk.check_order(buy()).rule == "symbol_halt"
    assert risk.check_market_data(in_session=False) == []
    risk.clock_obj.t = 20.0
    risk.check_market_data()
    assert risk.halted
