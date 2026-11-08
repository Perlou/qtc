"""
一键流水线：数据更新 → 因子 → 模型 → 组合 → 调仓指令（CONCEPT 2.5 节）

原则：按顺序执行、每步校验、失败即停。任何一步失败都抛出 PipelineError，
不返回任何"半成品"指令；每一步的耗时与状态都记录在 steps 里，供监控日报使用。

DailyPipeline 只依赖 contracts 中的 Protocol，不依赖任何具体实现——
测试里可以用几行代码写的假组件替换真实组件（见 tests/test_pipeline.py）。
build_daily_pipeline 负责把真实组件"装配"起来（依赖注入）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .contracts import (
    DataSource,
    FactorEngine,
    Model,
    OrderPlanner,
    PipelineError,
    PortfolioConstructor,
    RebalanceOrder,
    check,
    validate_factor_frame,
    validate_orders,
    validate_scores,
    validate_weights,
)
from .portfolio.construct import one_way_turnover
from .portfolio.orders import lot_ok as default_lot_ok


@dataclass
class StepLog:
    step: str
    status: str  # OK / FAIL / SKIP
    seconds: float
    detail: str = ""


@dataclass
class PipelineResult:
    as_of: pd.Timestamp
    orders: list[RebalanceOrder]
    target: pd.Series
    scores: pd.Series
    factors: pd.DataFrame
    current_weights: pd.Series
    prices: pd.Series
    equity: float
    steps: list[StepLog] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def steps_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self.steps])


class DailyPipeline:
    STEPS = ["1.数据更新", "2.因子计算", "3.模型训练与预测", "4.组合构建", "5.生成调仓指令"]

    def __init__(
        self,
        data: DataSource,
        factors: FactorEngine,
        model: Model,
        portfolio: PortfolioConstructor,
        planner: OrderPlanner,
        *,
        n_top: int,
        max_weight: float,
        max_turnover: float = 1.0,
        training_data: Callable | None = None,  # as_of -> (X_train, y_train)
        sellable: Callable | None = None,  # as_of -> Series[bool]
        lot_ok: Callable[[RebalanceOrder], bool] = default_lot_ok,
        min_coverage: float = 0.8,
        price_lookback_days: int = 30,
        log: Callable[[str], None] | None = None,
    ):
        self.data, self.factors, self.model = data, factors, model
        self.portfolio, self.planner = portfolio, planner
        self.n_top, self.max_weight, self.max_turnover = n_top, max_weight, max_turnover
        self.training_data, self.sellable, self.lot_ok = training_data, sellable, lot_ok
        self.min_coverage = min_coverage
        self.price_lookback_days = price_lookback_days
        self.log = log or (lambda msg: None)
        self.steps: list[StepLog] = []

    def _step(self, name: str, fn):
        t0 = time.perf_counter()
        try:
            out, detail = fn()
        except PipelineError as exc:
            self.steps.append(StepLog(name, "FAIL", time.perf_counter() - t0, str(exc)))
            self.log(f"❌ {name} 失败：{exc}")
            for rest in self.STEPS[self.STEPS.index(name) + 1 :]:
                self.steps.append(StepLog(rest, "SKIP", 0.0, "上游失败，未执行"))
            exc.steps = list(self.steps)  # 失败时也把步骤记录交给调用方（写入日报）
            raise
        self.steps.append(StepLog(name, "OK", time.perf_counter() - t0, detail))
        self.log(f"✅ {name}：{detail}")
        return out

    def run(self, date, current_shares: pd.Series | None = None, cash: float = 0.0) -> PipelineResult:
        self.steps = []
        current_shares = (current_shares if current_shares is not None else pd.Series(dtype="int64")).astype("int64")
        current_shares = current_shares[current_shares > 0]
        expected = pd.Timestamp(date)

        def s1():
            as_of = self.data.update(end=str(expected.date()))
            check(as_of == expected, f"数据只更新到 {as_of.date()}，预期 {expected.date()}：可能数据源未更新")
            px = self.data.load_prices(str((as_of - pd.Timedelta(days=self.price_lookback_days)).date()),
                                       str(as_of.date()))
            check(not px.empty and px.index[-1] == as_of, f"{as_of.date()} 无行情数据")
            today = px.iloc[-1]
            cov = today.notna().mean()
            check(cov >= self.min_coverage, f"{as_of.date()} 行情覆盖率 {cov:.0%} 过低")
            prices = px.ffill().iloc[-1]  # 停牌股用最近收盘价估值
            missing = current_shares.index.difference(prices.dropna().index)
            check(missing.empty, f"持仓 {list(missing)} 没有任何可用价格，无法估值")
            return (as_of, prices), f"数据日期 {as_of.date()}，行情覆盖率 {cov:.0%}"

        as_of, prices = self._step("1.数据更新", s1)

        def s2():
            X = self.factors.compute(as_of)
            validate_factor_frame(X.dropna(how="all"), self.min_coverage)
            check(X.dropna(how="all").shape[0] >= self.min_coverage * len(X), "可打分股票占比过低")
            return X, f"{X.shape[0]} 只股票 × {X.shape[1]} 个因子，有效 {X.dropna(how='all').shape[0]} 只"

        X = self._step("2.因子计算", s2)

        def s3():
            detail = "使用已训练模型"
            if self.training_data is not None:
                X_tr, y_tr = self.training_data(as_of)
                check(len(y_tr) > 0, "训练集为空：历史标签不足")
                self.model.fit(X_tr, y_tr)
                dates = X_tr.index.get_level_values(0) if isinstance(X_tr.index, pd.MultiIndex) else []
                span = f"{pd.Timestamp(min(dates)).date()}~{pd.Timestamp(max(dates)).date()}" if len(dates) else ""
                detail = f"训练样本 {len(y_tr)}（标签期 {span}）"
            scores = self.model.predict(X)
            validate_scores(scores, self.n_top)
            return scores, f"{detail}；分数标准差 {scores.std():.3f}"

        scores = self._step("3.模型训练与预测", s3)

        held_value = float((current_shares * prices.reindex(current_shares.index)).sum())
        equity = cash + held_value
        current_w = (current_shares * prices.reindex(current_shares.index) / equity) if equity > 0 else pd.Series(dtype=float)

        def s4():
            check(equity > 0, "账户权益为 0")
            kw = {"sellable": self.sellable(as_of)} if self.sellable is not None else {}
            target = self.portfolio.target_weights(scores, current_w, **kw)
            validate_weights(target, self.max_weight)
            cw = current_w / current_w.sum() if current_w.sum() > 0 else current_w  # 满仓口径，与组合层一致
            to = one_way_turnover(target, cw)
            if current_w.sum() >= 0.5:  # 建仓不受换手约束
                check(to <= self.max_turnover + 1e-6, f"单边换手 {to:.1%} 超过上限 {self.max_turnover:.0%}")
            kind = "单边换手" if current_w.sum() >= 0.5 else "建仓（不受换手上限约束），单边换手"
            return (target, to), f"持仓 {len(target)} 只，最大权重 {target.max():.2%}，{kind} {to:.1%}"

        target, turnover = self._step("4.组合构建", s4)

        def s5():
            orders = self.planner.plan(target, current_shares, prices, equity)
            validate_orders(orders, self.lot_ok, cash)
            n_buy = sum(o.side == "BUY" for o in orders)
            return orders, f"{len(orders)} 条指令（卖 {len(orders) - n_buy} / 买 {n_buy}）"

        orders = self._step("5.生成调仓指令", s5)
        return PipelineResult(
            as_of=as_of, orders=orders, target=target, scores=scores, factors=X, current_weights=current_w,
            prices=prices, equity=equity, steps=list(self.steps),
            skipped=list(getattr(self.planner, "skipped", [])), info={"turnover": turnover},
        )


def build_daily_pipeline(cfg, source=None, log: Callable[[str], None] | None = None) -> DailyPipeline:
    """用真实组件装配流水线：Phase 7 面板 → 因子注册表 → 滚动训练模型 → TopN → 取整指令"""
    from bt_core import CostModel

    from .data import PanelDataSource
    from .factors import RegistryFactorEngine
    from .models import make_labels, make_model, training_dates, training_set
    from .portfolio import LotOrderPlanner, TopNPortfolio

    source = source or PanelDataSource()
    engine = RegistryFactorEngine(source, cfg.factors, neutralize=cfg.neutralize)
    rebal = source.rebalance_dates()

    def training_data(as_of):
        tr = training_dates(rebal, as_of, cfg.train_window)
        check(len(tr) >= cfg.min_train, f"可用标签只有 {len(tr)} 期，少于 {cfg.min_train} 期")
        labels = make_labels(source.period_returns(rebal).where(source.tradable(rebal)["all"]))
        return training_set(engine.long_frame(rebal), labels, tr)

    def sellable(as_of):
        return source.sellable(rebal).loc[as_of]

    return DailyPipeline(
        source, engine, make_model(cfg.model, cfg.seed),
        TopNPortfolio(cfg.n_top, cfg.max_weight, cfg.max_turnover, cfg.weighting),
        LotOrderPlanner(CostModel(slippage=cfg.slippage), cfg.min_order_value),
        n_top=cfg.n_top, max_weight=cfg.max_weight, max_turnover=cfg.max_turnover,
        training_data=training_data if cfg.model != "equal" else None,
        sellable=sellable, log=log,
    )
