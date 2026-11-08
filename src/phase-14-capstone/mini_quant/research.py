"""
研究编排：把各层串成"可重复的实验"，供 03 / 05 / 06 课共用

    ResearchContext   一次加载：调仓日、持有期收益、可交易性、标签、基准、波动率、成交额
    ctx.run(cfg)      因子 → 滚动训练打分 → 组合构建 + 回测 → StrategyRun（同一 cfg 结果完全确定）
    TrialLog          试验日志（JSONL）：每一次尝试都记录，包括失败的——DSR 需要试验次数
    default_grid      参数网格；select_best 只看样本内指标选择
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import OUT_DIR
from .backtest import MonthlyBacktester, perf_summary
from .config import StrategyConfig
from .data import PanelDataSource
from .factors import RegistryFactorEngine
from .models import RollingTrainer, make_labels, make_model
from .portfolio import TopNPortfolio

from bt_core import CostModel  # noqa: E402
import factor_data  # noqa: E402

BENCH_NAMES = {"ew_universe": "股票池等权", "hs300": "沪深300", "zz500": "中证500"}


@dataclass
class StrategyRun:
    cfg: StrategyConfig
    bt: object  # BacktestResult
    scores: pd.DataFrame
    versions: pd.DataFrame
    bench: pd.DataFrame

    @property
    def frame(self) -> pd.DataFrame:
        return self.bt.frame

    def net(self, period: str = "all") -> pd.Series:
        return self._slice(self.frame["net"], period)

    def gross(self, period: str = "all") -> pd.Series:
        return self._slice(self.frame["gross"], period)

    def bench_ret(self, name: str = "ew_universe", period: str = "all") -> pd.Series:
        return self._slice(self.bench[name].reindex(self.frame.index), period)

    def excess(self, name: str = "ew_universe", period: str = "all") -> pd.Series:
        return (self.net(period) - self.bench_ret(name, period)).rename("excess")

    def _slice(self, s: pd.Series, period: str) -> pd.Series:
        c = self.cfg
        if period == "is":
            return s.loc[c.is_start : c.is_end]
        if period == "oos":
            return s.loc[c.oos_start : c.oos_end]
        return s.loc[c.is_start : c.oos_end]

    def summary(self, period: str = "all", bench: str = "ew_universe") -> dict:
        s = perf_summary(self.net(period), self.bench_ret(bench, period), rf_annual=factor_data.RF_ANNUAL)
        f = self._slice(self.frame["turnover"], period)
        s["月均单边换手"] = f.mean()
        s["年化单边换手"] = f.mean() * 12
        s["年化成本"] = self._slice(self.frame["cost"], period).mean() * 12
        return s


class ResearchContext:
    def __init__(self, source: PanelDataSource | None = None, verbose: bool = True):
        self.source = source or PanelDataSource(verbose=verbose)
        src = self.source
        self.dates = src.rebalance_dates()
        self.period_ret = src.period_returns(self.dates)
        self.status = src.tradable(self.dates)
        self.sellable = src.sellable(self.dates)
        self.labels = make_labels(self.period_ret.where(self.status["all"]))
        self.bench = pd.DataFrame({
            "ew_universe": src.ew_universe_returns(self.dates),
            "hs300": src.bench_returns(self.dates, "hs300"),
            "zz500": src.bench_returns(self.dates, "zz500"),
        })
        p = src.panel
        self.vol = p.daily_ret.rolling(60, min_periods=40).std().reindex(self.dates)
        self.adv = p.amount.rolling(20, min_periods=10).mean().reindex(self.dates)
        self._engines: dict = {}
        self._scores: dict = {}

    def engine(self, cfg: StrategyConfig) -> RegistryFactorEngine:
        key = (cfg.factors, cfg.neutralize)
        if key not in self._engines:
            self._engines[key] = RegistryFactorEngine(self.source, cfg.factors, neutralize=cfg.neutralize)
        return self._engines[key]

    def factor_long(self, cfg: StrategyConfig) -> pd.DataFrame:
        return self.engine(cfg).long_frame(self.dates)

    def scores(self, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = cfg.model_key()
        if key not in self._scores:
            trainer = RollingTrainer(lambda: make_model(cfg.model, cfg.seed), cfg.train_window, cfg.min_train,
                                     cfg.retrain_every)
            scores = trainer.run(self.factor_long(cfg), self.labels, self.dates)
            self._scores[key] = (scores, trainer.versions_frame())
        return self._scores[key]

    def backtest(self, cfg: StrategyConfig, scores: pd.DataFrame):
        constructor = TopNPortfolio(cfg.n_top, cfg.max_weight, cfg.max_turnover, cfg.weighting)
        bt = MonthlyBacktester(
            self.period_ret, constructor, CostModel(slippage=cfg.slippage), sellable=self.sellable,
            cost_multiplier=cfg.cost_multiplier, vol=self.vol,
        )
        return bt.run(scores, start=cfg.is_start, end=cfg.oos_end)

    def run(self, cfg: StrategyConfig) -> StrategyRun:
        scores, versions = self.scores(cfg)
        first = self.dates[self.dates >= pd.Timestamp(cfg.is_start)][0]
        if first not in scores.index:  # 所有试验必须从同一天开始比较
            raise ValueError(f"{cfg.model} 在样本内首个调仓日 {first.date()} 还没有分数，请调小 min_train")
        return StrategyRun(cfg, self.backtest(cfg, scores), scores, versions, self.bench)

    def score_ic(self, scores: pd.DataFrame) -> pd.Series:
        import factor_tools

        return factor_tools.calc_ic(scores, self.period_ret)


# ==================== 试验日志与参数网格 ====================


class TrialLog:
    """JSONL 试验日志。fresh=True 时清空旧记录：每次研究会话完整重跑，日志与本次结果一一对应"""

    def __init__(self, path: Path | None = None, fresh: bool = True):
        self.path = path or OUT_DIR / "trials.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()
        self.records: list[dict] = []

    def log(self, cfg: StrategyConfig, metrics: dict, note: str = "") -> None:
        rec = {"trial": len(self.records) + 1, "config": cfg.to_dict(), "label": cfg.short(),
               "metrics": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()},
               "note": note}
        self.records.append(rec)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def __len__(self) -> int:
        return len(self.records)


def default_grid(base: StrategyConfig) -> list[StrategyConfig]:
    """4 种模型 × 3 种持仓数 × 2 种换手上限 = 24 次试验"""
    grid = []
    for model in ("equal", "icw", "ridge", "hgb"):
        for n in (10, 20, 30):
            for to in (0.3, 1.0):
                grid.append(base.with_(model=model, n_top=n, max_turnover=to,
                                       max_weight=max(base.max_weight, 1.0 / n)))
    return grid


def run_grid(ctx: ResearchContext, grid: list[StrategyConfig], log: TrialLog | None = None):
    """返回 (样本内指标表, {label: 样本内超额收益序列}, {label: StrategyRun})。只计算、只记录样本内指标"""
    rows, is_excess, runs = [], {}, {}
    for cfg in grid:
        run = ctx.run(cfg)
        s = run.summary("is")
        m = {"样本内信息比率": s["信息比率"], "样本内超额年化": s["超额年化(几何)"], "样本内年化收益": s["年化收益"],
             "样本内Sharpe": s["Sharpe"], "月均单边换手": s["月均单边换手"]}
        if log is not None:
            log.log(cfg, m)
        rows.append({"label": cfg.short(), "model": cfg.model, "n_top": cfg.n_top, "max_turnover": cfg.max_turnover, **m})
        is_excess[cfg.short()] = run.excess(period="is")
        runs[cfg.short()] = run
    return pd.DataFrame(rows).set_index("label"), is_excess, runs


def select_best(table: pd.DataFrame, metric: str = "样本内信息比率") -> str:
    return str(table[metric].idxmax())


# ==================== 完整研究：选参 → 样本外 → 稳健性 → 成本 → 容量 ====================


@dataclass
class Study:
    base: StrategyConfig
    grid: pd.DataFrame  # 全部试验的样本内指标
    is_excess: dict  # {label: 样本内超额收益序列}
    runs: dict  # {label: StrategyRun}
    best: str
    run: StrategyRun  # 所选配置
    dsr: dict  # 样本内 DSR（选择发生在样本内，所以对样本内结果做多重检验校正）
    oos_psr: float  # 样本外超额 PSR(SR>0)
    cost_table: pd.DataFrame
    capacity: pd.DataFrame
    trial_log: Path | None = None

    @property
    def cfg(self) -> StrategyConfig:
        return self.run.cfg


def yearly_table(run: StrategyRun, bench: str = "ew_universe") -> pd.DataFrame:
    df = pd.DataFrame({"策略(扣费)": run.net(), "基准": run.bench_ret(bench)})
    g = (1 + df).groupby(df.index.year).prod() - 1
    g["超额"] = g["策略(扣费)"] - g["基准"]
    g.index.name = "年份"
    return g


def run_study(ctx: ResearchContext, base: StrategyConfig, log: TrialLog | None = None,
              cost_multipliers=(0.0, 1.0, 2.0, 3.0), aums=None) -> Study:
    """
    1. 参数网格全部跑一遍，只看样本内信息比率选出一个配置（每次尝试都写入试验日志）
    2. 对所选配置：样本内 DSR（试验次数 = 网格大小）、样本外 PSR
    3. 成本敏感性：0 / 1 / 2 / 3 倍成本
    4. 容量：样本外实际调仓 × 平方根冲击模型
    """
    import bt_core

    from .backtest import capacity_curve, deflated

    grid = default_grid(base)
    table, is_excess, runs = run_grid(ctx, grid, log)
    best = select_best(table)
    run = runs[best]
    dsr = deflated(is_excess[best], list(is_excess.values()))
    oos_ex = run.excess(period="oos").dropna()
    oos_psr = bt_core.psr(oos_ex.to_numpy(), 0.0) if len(oos_ex) > 2 else float("nan")

    rows = []
    for m in cost_multipliers:
        r = ctx.run(run.cfg.with_(cost_multiplier=m))
        for period in ("is", "oos"):
            s = r.summary(period)
            rows.append({"成本倍数": m, "区间": "样本内" if period == "is" else "样本外",
                         "年化收益": s["年化收益"], "超额年化(几何)": s["超额年化(几何)"], "信息比率": s["信息比率"]})
    cost_table = pd.DataFrame(rows)

    aums = aums if aums is not None else np.logspace(6, 10, 17)
    lo, hi = pd.Timestamp(run.cfg.oos_start), pd.Timestamp(run.cfg.oos_end)
    oos_trades = {d: w for d, w in run.bt.trades.items() if lo <= d <= hi}
    capacity = capacity_curve(oos_trades, ctx.adv, ctx.vol, aums, years=len(oos_trades) / 12)
    return Study(base, table, is_excess, runs, best, run, dsr, oos_psr, cost_table, capacity,
                 log.path if log is not None else None)
