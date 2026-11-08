"""
绩效、多重检验与容量（复用 Phase 4 bt_core 的绩效指标与 PSR / DSR）

    perf_summary     绝对收益指标 + 相对基准的超额指标（月频）
    trial_sharpes    全部试验的 Sharpe（不年化），用于 DSR
    deflated         DSR = PSR(期望最大 Sharpe | N 次试验)
    capacity_curve   平方根法则冲击成本：impact ≈ Y·σ·√(Q/V)（Phase 11 第 03 课）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _paths  # noqa: F401

import bt_core  # noqa: E402

PERIODS = 12  # 月频


def perf_summary(ret: pd.Series, bench: pd.Series | None = None, rf_annual: float = 0.0,
                 periods: int = PERIODS) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {"期数": len(ret)}
    rep = bt_core.performance_report(ret, rf_annual=rf_annual, periods=periods)
    out = {
        "期数": len(ret),
        "年化收益": rep["年化收益"],
        "年化波动": rep["年化波动"],
        "Sharpe": rep["Sharpe"],
        "最大回撤": rep["最大回撤"],
        "回撤起点": rep["回撤起点"],
        "回撤谷底": rep["回撤谷底"],
        "月胜率": float((ret > 0).mean()),
    }
    if bench is not None:
        b = bench.reindex(ret.index)
        ex = (ret - b).dropna()
        te = ex.std() * np.sqrt(periods)
        nav, bnav = (1 + ret).prod(), (1 + b.fillna(0)).prod()
        years = len(ret) / periods
        out.update({
            "基准年化": bnav ** (1 / years) - 1,
            "超额年化(几何)": (nav / bnav) ** (1 / years) - 1,
            "超额月均": ex.mean(),
            "跟踪误差": te,
            "信息比率": ex.mean() * periods / te if te > 0 else np.nan,
            "超额 t 值": ex.mean() / ex.std() * np.sqrt(len(ex)) if ex.std() > 0 else np.nan,
            "超额月胜率": float((ex > 0).mean()),
        })
    return out


def trial_sharpes(trial_returns: list[pd.Series]) -> np.ndarray:
    return np.array([bt_core.sharpe_ratio(r.dropna()) for r in trial_returns])


def deflated(selected: pd.Series, trial_returns: list[pd.Series]) -> dict:
    """返回观测 Sharpe、PSR(0)、期望最大 Sharpe、DSR（都是不年化的月度值，另给年化值便于阅读）"""
    srs = trial_sharpes(trial_returns)
    sel = selected.dropna().to_numpy()
    e_max = bt_core.expected_max_sr(srs) if len(srs) > 1 else 0.0
    sr = bt_core.sharpe_ratio(sel)
    return {
        "试验次数": len(srs),
        "观测 Sharpe(月)": sr,
        "观测 Sharpe(年化)": sr * np.sqrt(PERIODS),
        "试验 Sharpe 标准差(月)": float(srs.std(ddof=1)) if len(srs) > 1 else np.nan,
        "期望最大 Sharpe(月)": e_max,
        "期望最大 Sharpe(年化)": e_max * np.sqrt(PERIODS),
        "PSR(SR>0)": bt_core.psr(sel, 0.0),
        "DSR": bt_core.psr(sel, e_max),
    }


def sqrt_impact_bp(participation, daily_vol, Y: float = 1.0):
    """平方根法则冲击成本（bp）：Y · σ_daily · √(成交额 / 日均成交额)"""
    return 1e4 * Y * np.asarray(daily_vol) * np.sqrt(np.asarray(participation))


def capacity_curve(trades: dict, adv: pd.DataFrame, vol: pd.DataFrame, aums, years: float,
                   Y: float = 1.0) -> pd.DataFrame:
    """
    不同资金规模下的年化冲击成本拖累与最大参与率。
    trades：{调仓日: 权重变化}；adv / vol：调仓日的 20 日平均成交额（元）/ 60 日日波动率
    """
    rows = []
    for aum in aums:
        drag, max_part = 0.0, 0.0
        for d, dw in trades.items():
            a = adv.loc[d].reindex(dw.index)
            s = vol.loc[d].reindex(dw.index).fillna(vol.loc[d].median())
            part = (dw.abs() * aum / a).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            drag += float((dw.abs() * sqrt_impact_bp(part, s, Y) / 1e4).sum())
            max_part = max(max_part, float(part.max()))
        rows.append({"资金规模": aum, "年化冲击拖累": drag / years, "最大单票参与率": max_part})
    return pd.DataFrame(rows)


def capacity_at(curve: pd.DataFrame, alpha_annual: float, share: float = 0.5) -> float:
    """冲击成本吃掉 share 比例超额收益时的资金规模（对数插值）；超额 ≤ 0 时返回 0"""
    if not alpha_annual > 0:
        return 0.0
    limit = share * alpha_annual
    x, y = np.log(curve["资金规模"].to_numpy(float)), curve["年化冲击拖累"].to_numpy(float)
    if y[0] >= limit:
        return float(np.exp(x[0]))
    if y[-1] < limit:
        return float("inf")
    return float(np.exp(np.interp(limit, y, x)))
