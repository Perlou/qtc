"""
截面预处理：MAD 去极值 → Z-score → 行业 + 市值中性化 → 再 Z-score

全部调用 Phase 7 factor_tools 的实现（winsorize_mad / zscore / neutralize / preprocess），
这里只提供"单个截面逐步执行并返回中间结果"的版本，便于在课程里观察每一步的效果。
"""

from __future__ import annotations

import pandas as pd

from .. import _paths  # noqa: F401

import factor_tools  # noqa: E402

winsorize_mad = factor_tools.winsorize_mad
zscore = factor_tools.zscore
neutralize = factor_tools.neutralize


def preprocess_cross_section(x: pd.Series, sector: pd.Series, log_mcap: pd.Series, n_mad: float = 3.0,
                             neutral: bool = True, size_neutral: bool = True) -> dict[str, pd.Series]:
    """单个截面逐步预处理，返回每一步的结果：raw / winsorized / zscore / neutralized"""
    x = x.replace([float("inf"), float("-inf")], float("nan")).dropna()
    steps = {"raw": x}
    w = winsorize_mad(x, n_mad)
    steps["winsorized"] = w
    z = zscore(w)
    steps["zscore"] = z
    if neutral:
        lnmv = log_mcap.reindex(z.index) if size_neutral else log_mcap.reindex(z.index) * 0.0
        steps["neutralized"] = zscore(neutralize(z, sector.reindex(z.index), lnmv).dropna())
    return steps


def preprocess_panel(raw: pd.DataFrame, panel, mask: pd.DataFrame | None, neutral: bool = True,
                     size_neutral: bool = True) -> pd.DataFrame:
    """整个面板逐期预处理（Phase 7 factor_tools.preprocess）"""
    return factor_tools.preprocess(raw, panel, mask, neutral=neutral, size_neutral=size_neutral)
