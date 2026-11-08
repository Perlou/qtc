"""
监控层：日报核心指标、因子健康、特征漂移、组合暴露、执行质量与告警（CONCEPT 5.2–5.3 节）

所有阈值集中在 THRESHOLDS，告警分级沿用 Phase 13：P1 需要当天处理，P2 需要关注。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

THRESHOLDS = {
    "ic_recent_window": 6,  # 因子健康：最近 6 期
    "psi_watch": 0.10,
    "psi_drift": 0.25,
    "fill_rate_min": 0.95,
    "arrival_bp_max": 30.0,
    "drawdown_warn": -0.15,
    "tracking_error_max": 0.15,
}


@dataclass
class Alert:
    level: str  # P1 / P2
    key: str
    message: str


def nav_metrics(nav: pd.Series, bench: pd.Series, periods: int = 252) -> dict:
    """CONCEPT 5.3 节的日报指标（nav 与 bench 为同频净值）"""
    ret, bret = nav.pct_change().dropna(), bench.pct_change().dropna()
    excess = (ret - bret.reindex(ret.index)).dropna()
    dd = nav / nav.cummax() - 1
    return {
        "累计收益": nav.iloc[-1] / nav.iloc[0] - 1,
        "基准累计": bench.iloc[-1] / bench.iloc[0] - 1,
        "累计超额": (1 + excess).prod() - 1,
        "年化波动": ret.std() * np.sqrt(periods),
        "夏普(年化, rf=0)": ret.mean() / ret.std() * np.sqrt(periods) if ret.std() > 0 else np.nan,
        "当前回撤": dd.iloc[-1],
        "最大回撤": dd.min(),
        "近 20 期跟踪误差(年化)": excess.tail(20).std() * np.sqrt(periods),
    }


def factor_health(ic: pd.DataFrame, recent: int | None = None) -> pd.DataFrame:
    """
    各因子全样本 IC 均值 vs 最近 recent 期 IC 均值：
        正常：近期与长期同号；衰减：长期为正、近期 ≤ 0；反转：长期为负、近期 > 0
    """
    recent = recent or THRESHOLDS["ic_recent_window"]
    rows = {}
    for n, s in ic.items():
        s = s.dropna()
        long_, near = s.mean(), s.tail(recent).mean()
        if long_ > 0 and near <= 0:
            status = "⚠️ 衰减"
        elif long_ <= 0 < near:
            status = "反转"
        else:
            status = "正常"
        rows[n] = {"全样本IC": long_, f"近{recent}期IC": near, "近期IC>0占比": (s.tail(recent) > 0).mean(), "状态": status}
    return pd.DataFrame(rows).T


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """群体稳定性指标 PSI：按 expected 的分位数分箱，Σ (a − e) ln(a / e)"""
    e, a = expected.dropna().to_numpy(), actual.dropna().to_numpy()
    if len(e) < bins or len(a) < 2:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def feature_drift(X_train: pd.DataFrame, X_now: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in X_now.columns:
        v = psi(X_train[c], X_now[c])
        level = "稳定" if v < THRESHOLDS["psi_watch"] else ("关注" if v < THRESHOLDS["psi_drift"] else "⚠️ 漂移")
        rows[c] = {"PSI": v, "状态": level}
    return pd.DataFrame(rows).T


def sector_exposure(weights: pd.Series, sector: pd.Series, universe: pd.Index) -> pd.DataFrame:
    """组合行业权重 vs 股票池等权的行业权重"""
    port = weights.groupby(sector.reindex(weights.index)).sum()
    bench = sector.reindex(universe).value_counts(normalize=True)
    df = pd.DataFrame({"组合": port, "股票池等权": bench}).fillna(0.0)
    df["偏离"] = df["组合"] - df["股票池等权"]
    return df.sort_values("偏离", ascending=False)


def build_alerts(nav_m: dict | None = None, factor_h: pd.DataFrame | None = None,
                 drift: pd.DataFrame | None = None, exec_summary: dict | None = None,
                 pipeline_ok: bool = True) -> list[Alert]:
    t, alerts = THRESHOLDS, []
    if not pipeline_ok:
        alerts.append(Alert("P1", "pipeline.failed", "流水线失败，今日不生成调仓指令"))
    if nav_m:
        if nav_m.get("当前回撤", 0) < t["drawdown_warn"]:
            alerts.append(Alert("P2", "nav.drawdown", f"当前回撤 {nav_m['当前回撤']:.1%} 超过 {t['drawdown_warn']:.0%}"))
        if nav_m.get("近 20 期跟踪误差(年化)", 0) > t["tracking_error_max"]:
            alerts.append(Alert("P2", "nav.te", f"跟踪误差 {nav_m['近 20 期跟踪误差(年化)']:.1%} 偏高"))
    if factor_h is not None:
        for n, r in factor_h.iterrows():
            if "衰减" in str(r["状态"]):
                alerts.append(Alert("P2", f"factor.{n}", f"因子 {n} 近期 IC 转负（可能衰减）"))
    if drift is not None:
        for n, r in drift.iterrows():
            if "漂移" in str(r["状态"]):
                alerts.append(Alert("P2", f"drift.{n}", f"特征 {n} PSI={r['PSI']:.2f}，分布漂移"))
    if exec_summary:
        fr = exec_summary.get("成交率(金额)")
        if fr is not None and fr == fr and fr < t["fill_rate_min"]:
            alerts.append(Alert("P2", "exec.fill_rate", f"成交率 {fr:.1%} 低于 {t['fill_rate_min']:.0%}，未成交部分次日重新评估"))
        ab = exec_summary.get("到达价偏差(bp,加权)")
        if ab is not None and ab == ab and abs(ab) > t["arrival_bp_max"]:
            alerts.append(Alert("P2", "exec.slippage", f"到达价偏差 {ab:.1f}bp 超过 {t['arrival_bp_max']:.0f}bp"))
        if exec_summary.get("对账") == "不一致":
            alerts.append(Alert("P1", "exec.reconcile", "对账不一致：暂停交易，人工核查"))
    return alerts
