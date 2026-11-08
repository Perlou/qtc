"""
组合构建：分数 → 目标权重（实现 contracts.PortfolioConstructor）

    TopN 选股 → 加权（等权 / 逆波动，调用 Phase 8 portfolio_utils）→ 换手约束 → 单票上限

停牌股（sellable=False）既不能卖也不能买：保持当前权重（"冻结"），其余资金在可交易股票中分配。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _paths  # noqa: F401

import portfolio_utils  # noqa: E402

EPS = 1e-9


def topn_weights(scores: pd.Series, n: int, weighting: str = "equal", vol: pd.Series | None = None) -> pd.Series:
    """选分数最高的 n 只；equal 等权，inverse_vol 按波动率倒数加权"""
    s = scores.dropna()
    if len(s) < n:
        raise ValueError(f"有效分数只有 {len(s)} 个，不足 {n} 只")
    chosen = s.nlargest(n).index
    if weighting == "equal" or vol is None:
        return pd.Series(1.0 / n, index=chosen)
    if weighting == "inverse_vol":
        v = vol.reindex(chosen)
        v = v.fillna(v.median()).clip(lower=1e-4)
        w = portfolio_utils.inverse_vol_weights(np.diag(v.to_numpy() ** 2))
        return pd.Series(w, index=chosen)
    raise ValueError(f"未知加权方式：{weighting}")


def one_way_turnover(target: pd.Series, current: pd.Series) -> float:
    idx = target.index.union(current.index)
    return float((target.reindex(idx, fill_value=0.0) - current.reindex(idx, fill_value=0.0)).abs().sum() / 2)


def limit_turnover(target: pd.Series, current: pd.Series, max_turnover: float) -> pd.Series:
    """单边换手超过上限时，只朝目标移动一部分：w = current + λ(target − current)（CONCEPT 4.2 节）"""
    idx = target.index.union(current.index)
    t, c = target.reindex(idx, fill_value=0.0), current.reindex(idx, fill_value=0.0)
    turnover = (t - c).abs().sum() / 2
    lam = min(1.0, max_turnover / turnover) if turnover > 0 else 1.0
    w = c + lam * (t - c)
    return w[w > EPS]


def cap_weights(w: pd.Series, max_w: float, fixed: pd.Index | None = None) -> pd.Series:
    """单票上限：超出部分按比例分给未触顶的股票（fixed 中的股票不参与调整）"""
    w = w.copy()
    fixed = pd.Index([]) if fixed is None else fixed.intersection(w.index)
    for _ in range(50):
        free = w.index.difference(fixed)
        over = w[free][w[free] > max_w + EPS]
        if over.empty:
            break
        excess = (over - max_w).sum()
        w[over.index] = max_w
        room = w[free][w[free] < max_w - EPS]
        if room.empty:
            raise ValueError(f"单票上限 {max_w:.1%} 过低，无法满仓")
        w[room.index] += excess * room / room.sum()
    return w


def drift_weights(w: pd.Series, ret: pd.Series) -> pd.Series:
    """持有一期后的权重漂移：w_i (1 + r_i) / Σ w_j (1 + r_j)"""
    if w.empty:
        return w
    v = w * (1 + ret.reindex(w.index).fillna(0.0))
    return v / v.sum()


class TopNPortfolio:
    def __init__(self, n_top: int = 20, max_weight: float = 0.10, max_turnover: float = 1.0,
                 weighting: str = "equal"):
        if n_top * max_weight < 1 - EPS:
            raise ValueError(f"n_top × max_weight = {n_top * max_weight:.2f} < 1，无法满仓")
        self.n_top = n_top
        self.max_weight = max_weight
        self.max_turnover = max_turnover
        self.weighting = weighting

    def target_weights(self, scores: pd.Series, current: pd.Series, sellable: pd.Series | None = None,
                       vol: pd.Series | None = None) -> pd.Series:
        current = current[current > EPS] if len(current) else pd.Series(dtype=float)
        if len(current) and current.sum() > EPS:
            # 换手按"满仓口径"计算：现金按比例视为已投资，否则用现金建仓的部分也会被算作换手、
            # 归一化后实际换手会超过上限
            current = current / current.sum()
        frozen = pd.Series(dtype=float)
        if sellable is not None and len(current):
            ok = sellable.reindex(current.index).fillna(False).astype(bool)
            frozen = current[~ok]
        avail = 1.0 - frozen.sum()
        cand = scores.drop(frozen.index, errors="ignore")
        ideal = topn_weights(cand, self.n_top, self.weighting, vol) * avail
        ideal = pd.concat([ideal, frozen])
        if current.sum() < 0.5:  # 从现金建仓：不受换手上限约束
            target = ideal
        else:
            target = limit_turnover(ideal, current, self.max_turnover)
        target = target / target.sum()
        target = cap_weights(target, self.max_weight, fixed=frozen.index)
        target = target / target.sum()
        if current.sum() >= 0.5:  # 单票上限调整可能让换手略超上限：再按比例收缩一次（两者都满仓，结果仍满仓）
            to = one_way_turnover(target, current)
            if to > self.max_turnover + 1e-12:
                target = limit_turnover(target, current, self.max_turnover)
                target = target / target.sum()
        return target.sort_values(ascending=False).rename("weight")
