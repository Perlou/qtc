"""
月频组合回测（消费组合层输出）

与执行层消费的是同一个 PortfolioConstructor：回测里每期调用 target_weights，
实盘流水线里也调用同一个方法——"回测的策略"就是"实盘的策略"（CONCEPT 2.1 节）。

每个调仓日 t：
    1. 当前权重 = 上一期目标权重经过持有期收益漂移后的权重
    2. 目标权重 = constructor.target_weights(分数, 当前权重, 可卖出掩码)
    3. 成本 = 买入额 × 买入费率 + 卖出额 × 卖出费率（Phase 4 CostModel；印花税按当时费率）
    4. 本期收益 = Σ 目标权重 × 持有期收益 − 成本

简化：向量化回测按比例费率扣费，不含最低佣金 5 元（逐笔指令的费用见组合层 LotOrderPlanner）；
停牌股按 Phase 7 口径以最后收盘价估值，不能卖出时保持原权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _paths  # noqa: F401
from ..portfolio.construct import drift_weights

from bt_core import CostModel, stamp_duty_rate_at  # noqa: E402


def cost_rates(cost: CostModel, date) -> tuple[float, float]:
    """单边费率：买入 = 佣金 + 过户费 + 滑点；卖出再加印花税（按当时费率）"""
    buy = cost.commission_rate + cost.transfer_fee_rate + cost.slippage
    return buy, buy + stamp_duty_rate_at(date)


@dataclass
class BacktestResult:
    frame: pd.DataFrame  # index = 调仓日；gross / cost / net / turnover / n_hold
    weights: dict = field(default_factory=dict)  # {调仓日: 目标权重}
    trades: dict = field(default_factory=dict)  # {调仓日: 目标 − 漂移后当前（带符号）}

    def returns(self, col: str = "net", start=None, end=None) -> pd.Series:
        s = self.frame[col]
        return s.loc[pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None]


class MonthlyBacktester:
    def __init__(self, period_ret: pd.DataFrame, constructor, cost_model: CostModel | None = None,
                 sellable: pd.DataFrame | None = None, cost_multiplier: float = 1.0,
                 vol: pd.DataFrame | None = None):
        self.period_ret = period_ret
        self.constructor = constructor
        self.cost = cost_model or CostModel()
        self.sellable = sellable
        self.cost_multiplier = cost_multiplier
        self.vol = vol

    def run(self, scores: pd.DataFrame, start=None, end=None) -> BacktestResult:
        dates = scores.index
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        current = pd.Series(dtype=float)
        rows, weights, trades = {}, {}, {}
        for d in dates:
            r = self.period_ret.loc[d]
            if r.notna().sum() == 0:  # 最后一期没有持有期收益
                break
            sell_mask = self.sellable.loc[d] if self.sellable is not None else None
            vol = self.vol.loc[d] if self.vol is not None else None
            target = self.constructor.target_weights(scores.loc[d], current, sellable=sell_mask, vol=vol)
            idx = target.index.union(current.index)
            delta = target.reindex(idx, fill_value=0.0) - current.reindex(idx, fill_value=0.0)
            buy_rate, sell_rate = cost_rates(self.cost, d)
            buy, sell = delta.clip(lower=0).sum(), (-delta).clip(lower=0).sum()
            cost = (buy * buy_rate + sell * sell_rate) * self.cost_multiplier
            rr = r.reindex(target.index).fillna(0.0)
            gross = float((target * rr).sum())
            rows[d] = {"gross": gross, "cost": cost, "net": gross - cost, "turnover": (buy + sell) / 2,
                       "n_hold": int((target > 1e-9).sum())}
            weights[d], trades[d] = target, delta[delta.abs() > 1e-12]
            current = drift_weights(target, rr)
        return BacktestResult(pd.DataFrame(rows).T, weights, trades)
