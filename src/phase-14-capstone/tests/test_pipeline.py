"""一键流水线：每步校验、失败即停、不产生半成品指令（用假组件测试，体现 Protocol 的可替换性）"""

import numpy as np
import pandas as pd
import pytest

from bt_core import CostModel
from mini_quant.contracts import (
    DataSource,
    FactorEngine,
    Model,
    OrderPlanner,
    PipelineError,
    PortfolioConstructor,
    check,
)
from mini_quant.pipeline import DailyPipeline
from mini_quant.portfolio import LotOrderPlanner, TopNPortfolio

CODES = [f"6000{i:02d}" for i in range(40)]
DATES = pd.bdate_range("2024-11-01", "2024-12-31")
AS_OF = DATES[-1]


class FakeData:
    def __init__(self, last=AS_OF, nan_ratio=0.0):
        self.last = last
        rng = np.random.default_rng(1)
        self.px = pd.DataFrame(rng.uniform(5, 50, (len(DATES), len(CODES))), index=DATES, columns=CODES)
        if nan_ratio:
            self.px.iloc[-1, : int(len(CODES) * nan_ratio)] = np.nan

    def update(self, end):
        return pd.Timestamp(self.last)

    def load_prices(self, start, end):
        return self.px.loc[pd.Timestamp(start) : pd.Timestamp(end)]


class FakeFactors:
    def __init__(self, missing=0.0):
        rng = np.random.default_rng(2)
        self.X = pd.DataFrame(rng.normal(size=(len(CODES), 3)), index=CODES, columns=["f1", "f2", "f3"])
        if missing:
            self.X.iloc[: int(len(CODES) * missing), 0] = np.nan

    def compute(self, date):
        return self.X


class MeanModel:
    def __init__(self, constant=False):
        self.constant = constant
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return pd.Series(1.0, index=X.index) if self.constant else X.mean(axis=1)


class BadPortfolio:
    def target_weights(self, scores, current):
        return pd.Series(0.03, index=scores.nlargest(20).index)  # 和为 0.6


def make_pipeline(data=None, factors=None, model=None, portfolio=None, **kw):
    return DailyPipeline(
        data or FakeData(), factors or FakeFactors(), model or MeanModel(),
        portfolio or TopNPortfolio(n_top=20, max_weight=0.1, max_turnover=0.3),
        LotOrderPlanner(CostModel(), min_order_value=1_000),
        n_top=20, max_weight=0.1, max_turnover=0.3, **kw,
    )


def test_fakes_satisfy_protocols():
    assert isinstance(FakeData(), DataSource)
    assert isinstance(FakeFactors(), FactorEngine)
    assert isinstance(MeanModel(), Model)
    assert isinstance(TopNPortfolio(), PortfolioConstructor)
    assert isinstance(LotOrderPlanner(), OrderPlanner)


def test_check_raises_pipeline_error():
    check(True, "ok")
    with pytest.raises(PipelineError, match="boom"):
        check(False, "boom")


def test_happy_path_from_cash():
    res = make_pipeline().run(AS_OF, pd.Series(dtype="int64"), cash=1_000_000)
    assert [s.status for s in res.steps] == ["OK"] * 5
    assert abs(res.target.sum() - 1) < 1e-9 and len(res.target) == 20
    assert all(o.side == "BUY" for o in res.orders) and len(res.orders) == 20
    assert all(o.volume % 100 == 0 for o in res.orders)
    assert sum(o.est_value + o.est_fee for o in res.orders) <= 1_000_000


def test_training_hook_is_called():
    model = MeanModel()
    idx = pd.MultiIndex.from_product([DATES[:2], CODES], names=["date", "code"])
    X_tr = pd.DataFrame(0.0, index=idx, columns=["f1", "f2", "f3"])
    y_tr = pd.Series(0.0, index=idx)
    res = make_pipeline(model=model, training_data=lambda as_of: (X_tr, y_tr)).run(AS_OF, cash=1_000_000)
    assert model.fitted
    assert "训练样本" in res.steps[2].detail


def test_turnover_cap_respected_when_rebalancing():
    first = make_pipeline().run(AS_OF, cash=1_000_000)
    held = pd.Series({o.symbol: o.volume for o in first.orders})
    cash = 1_000_000 - sum(o.est_value + o.est_fee for o in first.orders)
    flipped = FakeFactors()
    flipped.X = -flipped.X  # 分数完全反转 → 理想换手 100%
    res = make_pipeline(factors=flipped).run(AS_OF, held, cash=cash)
    assert res.info["turnover"] <= 0.3 + 1e-6
    assert [o.side for o in res.orders][0] == "SELL"


def test_stale_data_stops_pipeline():
    pipe = make_pipeline(data=FakeData(last=DATES[-2]))
    with pytest.raises(PipelineError, match="数据只更新到") as e:
        pipe.run(AS_OF, cash=1_000_000)
    statuses = [s.status for s in e.value.steps]
    assert statuses == ["FAIL", "SKIP", "SKIP", "SKIP", "SKIP"]  # 失败即停，后续步骤不执行


def test_low_price_coverage_fails():
    with pytest.raises(PipelineError, match="覆盖率"):
        make_pipeline(data=FakeData(nan_ratio=0.5)).run(AS_OF, cash=1_000_000)


def test_factor_missing_rate_fails():
    with pytest.raises(PipelineError, match="缺失率过高"):
        make_pipeline(factors=FakeFactors(missing=0.4)).run(AS_OF, cash=1_000_000)


def test_degenerate_scores_fail():
    with pytest.raises(PipelineError, match="无区分度"):
        make_pipeline(model=MeanModel(constant=True)).run(AS_OF, cash=1_000_000)


def test_bad_weights_fail_before_orders():
    pipe = make_pipeline(portfolio=BadPortfolio())
    with pytest.raises(PipelineError, match="之和"):
        pipe.run(AS_OF, cash=1_000_000)
    assert [s.status for s in pipe.steps][-2:] == ["FAIL", "SKIP"]  # 不会生成调仓指令


def test_zero_equity_fails():
    with pytest.raises(PipelineError, match="权益"):
        make_pipeline().run(AS_OF, cash=0.0)
