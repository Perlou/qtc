"""
调仓指令：目标权重 → 按交易单位取整的股数 → 与当前持仓求差 → 先卖后买（实现 contracts.OrderPlanner）

交易规则与成本全部复用 Phase 4 bt_core：
    board_rule      板块规则：主板 / 创业板买入 100 股整数倍；科创板至少 200 股、超出部分按 1 股递增
    round_lot_buy   买入股数向下取整到合法数量
    CostModel       佣金（最低 5 元）+ 印花税（仅卖出）+ 过户费 + 滑点

卖出规则：清仓可以卖出全部（含零股）；部分减仓按交易单位取整。
"""

from __future__ import annotations

import pandas as pd

from .. import _paths  # noqa: F401
from ..contracts import RebalanceOrder

from bt_core import CostModel, round_lot_buy  # noqa: E402
from ..rules import board_rule


def lot_ok(order: RebalanceOrder) -> bool:
    """股数是否符合板块规则（validate_orders 用）"""
    rule = board_rule(order.symbol)
    if order.side == "BUY":
        return order.volume == round_lot_buy(order.volume, rule)
    if order.target_shares == 0:  # 清仓：允许零股一次性卖出
        return order.volume == order.current_shares
    if rule.step == 1:  # 科创板：单笔至少 200 股，超出部分按 1 股递增
        return order.volume >= rule.min_lot
    return order.volume % rule.step == 0


class LotOrderPlanner:
    def __init__(self, cost_model: CostModel | None = None, min_order_value: float = 0.0, cash_buffer: float = 0.003):
        """
        cash_buffer：按 (1 − buffer) × 资金计算目标股数，为买入费用预留现金，
        否则"刚好满仓"的目标在扣除佣金后会资金不足
        """
        self.cost = cost_model or CostModel()
        self.min_order_value = min_order_value
        self.cash_buffer = cash_buffer
        self.skipped: list[dict] = []

    def target_shares(self, target: pd.Series, prices: pd.Series, capital: float) -> pd.Series:
        investable = capital * (1 - self.cash_buffer)
        out = {}
        for code, w in target.items():
            px = prices.get(code)
            if w <= 0 or px is None or not (px > 0):
                out[code] = 0
                continue
            out[code] = round_lot_buy(w * investable / px, board_rule(code))
        return pd.Series(out, dtype="int64")

    def plan(self, target: pd.Series, current_shares: pd.Series, prices: pd.Series,
             capital: float) -> list[RebalanceOrder]:
        self.skipped = []
        tgt_shares = self.target_shares(target, prices, capital)
        codes = tgt_shares.index.union(current_shares[current_shares > 0].index)
        sells, buys = [], []
        for code in codes:
            cur = int(current_shares.get(code, 0))
            tgt = int(tgt_shares.get(code, 0))
            px = float(prices.get(code, float("nan")))
            rule = board_rule(code)
            delta = tgt - cur
            if delta == 0:
                continue
            if not (px > 0):
                self.skipped.append({"symbol": code, "reason": "无有效价格"})
                continue
            if delta > 0:
                side, vol = "BUY", round_lot_buy(delta, rule)
            else:
                side, vol = "SELL", -delta
                if tgt > 0 and rule.step > 1:
                    vol = vol // rule.step * rule.step
                elif tgt > 0 and vol < rule.min_lot:  # 科创板部分减仓也要满足单笔最低数量
                    vol = 0
            if vol <= 0:
                self.skipped.append({"symbol": code, "reason": "变动不足一个交易单位"})
                continue
            value = vol * px
            if tgt > 0 and value < self.min_order_value:  # 清仓不受最小金额限制
                self.skipped.append({"symbol": code, "reason": f"金额 {value:,.0f} < {self.min_order_value:,.0f}"})
                continue
            fee = self.cost.total(value, is_sell=(side == "SELL"))
            order = RebalanceOrder(
                symbol=code, side=side, volume=int(vol), ref_price=px, current_shares=cur,
                target_shares=tgt, target_weight=float(target.get(code, 0.0)), est_value=value, est_fee=fee,
                reason="清仓" if tgt == 0 else ("建仓" if cur == 0 else ("加仓" if side == "BUY" else "减仓")),
            )
            (sells if side == "SELL" else buys).append(order)
        return sells + buys  # 先卖后买：A 股资金约束下，卖出释放的资金才能用于买入


def orders_frame(orders: list[RebalanceOrder]) -> pd.DataFrame:
    cols = ["symbol", "side", "volume", "ref_price", "current_shares", "target_shares", "target_weight",
            "est_value", "est_fee", "reason"]
    return pd.DataFrame([o.to_dict() for o in orders], columns=cols)


def realized_weights(shares: pd.Series, prices: pd.Series) -> pd.Series:
    v = shares * prices.reindex(shares.index)
    return v / v.sum() if v.sum() > 0 else v
