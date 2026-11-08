"""
单因子快速诊断：Rank IC、ICIR、t 值、IC>0 占比、分层多空（全部调用 Phase 7 factor_tools）

Rank IC 的口径：t 日因子 vs t+1 开盘 → 下一调仓日次日开盘的持有期收益（与回测完全一致）。
"""

from __future__ import annotations

import pandas as pd

from .. import _paths  # noqa: F401

import factor_tools  # noqa: E402


def ic_series(clean: dict[str, pd.DataFrame], fwd: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({n: factor_tools.calc_ic(f, fwd) for n, f in clean.items()})


def ic_table(clean: dict[str, pd.DataFrame], fwd: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """每个因子一行：IC 均值、ICIR、t 值、IC>0 占比、期数、分层多空月均收益"""
    rows = {}
    for n, f in clean.items():
        s = factor_tools.ic_summary(factor_tools.calc_ic(f, fwd))
        q = factor_tools.quantile_returns(f, fwd, n_groups=n_groups)
        s["多空月均"] = q["多空"].mean()
        s["单调性"] = q[list(range(1, n_groups + 1))].mean().corr(pd.Series(range(1, n_groups + 1),
                                                                            index=range(1, n_groups + 1)))
        rows[n] = s
    return pd.DataFrame(rows).T


def diagnose(table: pd.DataFrame, min_abs_t: float = 2.0) -> pd.Series:
    """
    快速结论（规则化）：
        |t| ≥ 2 且 IC 与预设方向一致 → 有效；|t| ≥ 2 但方向相反 → 方向反了；否则 → 不显著
    预设方向：所有因子已统一为"越大越好"，所以 IC 应为正。
    """
    def verdict(r):
        if abs(r["t值"]) < min_abs_t:
            return "不显著"
        return "有效" if r["IC均值"] > 0 else "方向与预期相反"

    return table.apply(verdict, axis=1)
