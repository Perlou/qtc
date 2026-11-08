"""调仓指令：按交易单位取整、先卖后买、清仓零股、最小金额、资金校验"""

import pandas as pd
import pytest

from bt_core import CostModel
from mini_quant.contracts import PipelineError, validate_orders
from mini_quant.portfolio import LotOrderPlanner, TopNPortfolio, limit_turnover, lot_ok, one_way_turnover


def _planner(**kw):
    return LotOrderPlanner(CostModel(), cash_buffer=0.0, **kw)


def test_main_board_rounds_down_to_100():
    target = pd.Series({"600000": 0.05})
    orders = _planner().plan(target, pd.Series(dtype="int64"), pd.Series({"600000": 10.37}), 1_000_000)
    assert len(orders) == 1
    o = orders[0]
    assert o.side == "BUY" and o.volume == 4800  # 50,000 / 10.37 = 4821.6 → 4800
    assert o.volume % 100 == 0
    assert o.est_value == pytest.approx(4800 * 10.37)
    assert o.est_fee == pytest.approx(CostModel().total(4800 * 10.37, is_sell=False))


def test_chinext_uses_100_lot():
    orders = _planner().plan(pd.Series({"300750": 0.1}), pd.Series(dtype="int64"),
                             pd.Series({"300750": 183.0}), 1_000_000)
    assert orders[0].volume == 500  # 100,000 / 183 = 546 → 500


def test_star_market_min_200_then_step_1():
    prices = pd.Series({"688001": 100.0})
    small = _planner().plan(pd.Series({"688001": 0.015}), pd.Series(dtype="int64"), prices, 1_000_000)
    assert small == []  # 150 股 < 200 股，买不了
    ok = _planner().plan(pd.Series({"688001": 0.0253}), pd.Series(dtype="int64"), prices, 1_000_000)
    assert ok[0].volume == 253  # 科创板超过 200 股后按 1 股递增
    assert lot_ok(ok[0])


def test_exit_sells_all_including_odd_lot():
    current = pd.Series({"600000": 1_150})
    orders = _planner().plan(pd.Series(dtype=float), current, pd.Series({"600000": 10.0}), 1_000_000)
    assert orders[0].side == "SELL" and orders[0].volume == 1_150 and orders[0].reason == "清仓"
    assert lot_ok(orders[0])


def test_partial_reduction_rounds_to_lot():
    current = pd.Series({"600000": 10_000})
    orders = _planner().plan(pd.Series({"600000": 0.0555}), current, pd.Series({"600000": 10.0}), 1_000_000)
    o = orders[0]
    assert o.side == "SELL" and o.target_shares == 5_500 and o.volume == 4_500
    assert o.volume % 100 == 0


def test_sells_come_before_buys():
    current = pd.Series({"600000": 5_000, "000001": 3_000})
    target = pd.Series({"600519": 0.5, "000002": 0.5})
    prices = pd.Series({"600000": 10.0, "000001": 12.0, "600519": 1500.0, "000002": 8.0})
    orders = _planner().plan(target, current, prices, 1_000_000)
    sides = [o.side for o in orders]
    assert sides == sorted(sides, key=lambda s: s != "SELL")
    assert sides[0] == "SELL" and sides[-1] == "BUY"


def test_min_order_value_skips_small_adjustments_but_not_exits():
    current = pd.Series({"600000": 10_000, "000001": 100})
    target = pd.Series({"600000": 0.1025})  # 目标 10,200 股：只需加仓 200 股 = 2,000 元
    prices = pd.Series({"600000": 10.0, "000001": 10.0})
    p = _planner(min_order_value=5_000)
    orders = p.plan(target, current, prices, 1_000_000)
    syms = {o.symbol: o for o in orders}
    assert "000001" in syms and syms["000001"].volume == 100  # 清仓即使只有 1,000 元也要卖
    assert "600000" not in syms  # 2,000 元的微调被跳过
    assert any(s["symbol"] == "600000" for s in p.skipped)


def test_cash_buffer_keeps_orders_affordable():
    target = pd.Series({c: 0.05 for c in [f"6000{i:02d}" for i in range(20)]})
    prices = pd.Series(1.0, index=target.index)
    orders = LotOrderPlanner(CostModel(), cash_buffer=0.003).plan(target, pd.Series(dtype="int64"), prices, 100_000)
    validate_orders(orders, lot_ok, cash=100_000)  # 不抛异常
    without = LotOrderPlanner(CostModel(), cash_buffer=0.0).plan(target, pd.Series(dtype="int64"), prices, 100_000)
    with pytest.raises(PipelineError, match="资金不足"):
        validate_orders(without, lot_ok, cash=100_000)  # 满仓 + 佣金 → 资金不足


def test_validate_orders_catches_bad_lot_and_order():
    target = pd.Series({"600000": 0.05})
    o = _planner().plan(target, pd.Series(dtype="int64"), pd.Series({"600000": 10.0}), 1_000_000)[0]
    from dataclasses import replace

    with pytest.raises(PipelineError, match="交易单位"):
        validate_orders([replace(o, volume=4_950)], lot_ok, cash=1e9)
    sell = replace(o, side="SELL", volume=100, current_shares=100, target_shares=0)
    with pytest.raises(PipelineError, match="先卖后买"):
        validate_orders([o, replace(sell, symbol="000001")], lot_ok, cash=1e9)


def test_limit_turnover_caps_one_way_turnover():
    current = pd.Series(0.1, index=[f"S{i}" for i in range(10)])
    target = pd.Series(0.1, index=[f"S{i}" for i in range(5, 15)])
    w = limit_turnover(target, current, 0.2)
    assert one_way_turnover(w, current) == pytest.approx(0.2)
    assert w.sum() == pytest.approx(1.0)


def test_topn_freezes_unsellable_names():
    scores = pd.Series(range(30), index=[f"S{i:02d}" for i in range(30)], dtype=float)
    current = pd.Series({"S00": 0.06, "S01": 0.94})
    sellable = pd.Series(True, index=scores.index)
    sellable["S00"] = False  # 停牌：不能卖
    w = TopNPortfolio(n_top=20, max_weight=0.1).target_weights(scores, current, sellable=sellable)
    assert w["S00"] == pytest.approx(0.06)  # 冻结
    assert w.sum() == pytest.approx(1.0)
    assert w.max() <= 0.1 + 1e-9


def test_turnover_cap_holds_with_cash_in_account():
    """账户有现金（当前权重之和 < 1）时，满仓口径的换手仍不超过上限"""
    scores = pd.Series(range(40), index=[f"S{i:02d}" for i in range(40)], dtype=float)
    current = pd.Series(0.043, index=[f"S{i:02d}" for i in range(20)])  # 和为 0.86，其余为现金
    port = TopNPortfolio(n_top=20, max_weight=0.1, max_turnover=0.5)
    w = port.target_weights(scores, current)
    assert w.sum() == pytest.approx(1.0)
    assert one_way_turnover(w, current / current.sum()) <= 0.5 + 1e-9
