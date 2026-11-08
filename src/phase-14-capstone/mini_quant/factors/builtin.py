"""
内置因子

1. Phase 7 的 8 个因子：直接复用 factor_tools.build_factors（含 PIT 财务对齐），
   这里只是注册成插件——不重写任何因子逻辑
2. 两个日频价量因子（CONCEPT 3.2 节示例），演示"新增因子只需写一个函数并注册"
"""

from __future__ import annotations

import pandas as pd

from .. import _paths  # noqa: F401
from .registry import REGISTRY, FactorContext, factor

import factor_tools  # noqa: E402

PHASE7_FACTORS = ["ep", "bp", "roe", "mom_12_1", "rev_1m", "lowvol", "lowturn", "size"]


def _phase7_adapter(name: str):
    def fn(ctx: FactorContext) -> pd.DataFrame:
        return ctx.phase7_raw[name]

    fn.__name__ = f"phase7_{name}"
    fn.__doc__ = f"Phase 7 factor_tools.build_factors()['{name}']"
    return fn


for _name in PHASE7_FACTORS:
    if _name not in REGISTRY:
        REGISTRY.register(
            _name,
            category=factor_tools.FACTOR_CATEGORY[_name],
            description=factor_tools.FACTOR_LABELS[_name],
            source="Phase 7 build_factors",
        )(_phase7_adapter(_name))


if "mom_20" not in REGISTRY:

    @factor("mom_20", category="动量", description="20 日动量（跳过最近 1 日）", source="Phase 14")
    def momentum_20(ctx: FactorContext) -> pd.DataFrame:
        px = ctx.panel.adj_close.ffill()
        return ctx.at(px, 1) / ctx.at(px, 21) - 1


if "vol_20" not in REGISTRY:

    @factor("vol_20", category="低波", description="20 日低波动（取负号）", source="Phase 14")
    def volatility_20(ctx: FactorContext) -> pd.DataFrame:
        return -ctx.at(ctx.panel.daily_ret.rolling(20, min_periods=15).std())
