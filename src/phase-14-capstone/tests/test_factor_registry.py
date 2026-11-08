"""因子注册表与截面预处理"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mini_quant.factors import REGISTRY, FactorContext, FactorRegistry, preprocess_cross_section


def _fake_panel(n_days=30, codes=("A", "B", "C")):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    px = pd.DataFrame(np.arange(n_days * len(codes), dtype=float).reshape(n_days, len(codes)) + 1,
                      index=dates, columns=list(codes))
    return SimpleNamespace(dates=dates, codes=pd.Index(codes), close=px)


def test_decorator_registers_and_returns_function():
    reg = FactorRegistry()

    @reg.register("my_factor", category="测试", description="示例")
    def my_factor(ctx):
        return ctx.panel.close

    assert "my_factor" in reg
    assert reg.names() == ["my_factor"]
    assert reg.get("my_factor").fn is my_factor  # 装饰器不改变原函数
    assert reg.table().iloc[0]["类别"] == "测试"


def test_duplicate_name_raises_unless_overwrite():
    reg = FactorRegistry()
    reg.register("f")(lambda ctx: None)
    with pytest.raises(ValueError, match="已注册"):
        reg.register("f")(lambda ctx: None)
    new = lambda ctx: "v2"  # noqa: E731
    reg.register("f", overwrite=True)(new)
    assert reg.get("f").fn is new


@pytest.mark.parametrize("bad", ["", "a-b", "has space", "中文"])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError):
        FactorRegistry().register(bad)


def test_unknown_factor_lists_registered():
    reg = FactorRegistry()
    reg.register("x")(lambda ctx: None)
    with pytest.raises(KeyError, match="x"):
        reg.get("y")


def test_compute_reindexes_and_cleans_inf():
    reg = FactorRegistry()
    panel = _fake_panel()
    dates = panel.dates[[5, 10, 20]]

    @reg.register("ratio")
    def ratio(ctx):
        out = ctx.at(ctx.panel.close)
        out.iloc[0, 0] = np.inf
        return out[["A", "B"]]  # 少返回一列

    df = reg.compute("ratio", FactorContext(panel, dates))
    assert list(df.columns) == ["A", "B", "C"]
    assert df.index.equals(dates)
    assert np.isnan(df.iloc[0, 0])  # inf → NaN
    assert df["C"].isna().all()


def test_compute_requires_dataframe():
    reg = FactorRegistry()
    reg.register("bad")(lambda ctx: [1, 2, 3])
    panel = _fake_panel()
    with pytest.raises(TypeError):
        reg.compute("bad", FactorContext(panel, panel.dates[:2]))


def test_context_at_uses_only_past_data():
    panel = _fake_panel()
    dates = panel.dates[[3, 10]]
    ctx = FactorContext(panel, dates)
    lag0 = ctx.at(panel.close, 0)
    lag5 = ctx.at(panel.close, 5)
    assert lag0.loc[dates[1], "A"] == panel.close.iloc[10]["A"]
    assert lag5.loc[dates[1], "A"] == panel.close.iloc[5]["A"]
    assert lag5.loc[dates[0]].isna().all()  # 往前不够 5 天 → NaN，而不是偷看


def test_context_rejects_non_trading_dates():
    panel = _fake_panel()
    with pytest.raises(ValueError):
        FactorContext(panel, pd.DatetimeIndex(["2030-01-01"]))


def test_builtin_factors_registered():
    for name in ["ep", "bp", "roe", "mom_12_1", "rev_1m", "lowvol", "lowturn", "size", "mom_20", "vol_20"]:
        assert name in REGISTRY
    assert REGISTRY.get("ep").source == "Phase 7 build_factors"


def test_preprocess_cross_section_steps():
    rng = np.random.default_rng(0)
    codes = [f"S{i:02d}" for i in range(60)]
    x = pd.Series(rng.normal(size=60), index=codes)
    x.iloc[0] = 50.0  # 极端值
    sector = pd.Series(np.repeat(["a", "b", "c"], 20), index=codes)
    log_mcap = pd.Series(rng.normal(22, 1, 60), index=codes)
    x = x + log_mcap * 0.5 + sector.map({"a": 1.0, "b": 0.0, "c": -1.0})
    steps = preprocess_cross_section(x, sector, log_mcap)
    assert steps["winsorized"].max() < steps["raw"].max()  # MAD 截断了极端值
    z = steps["zscore"]
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
    n = steps["neutralized"]
    assert n.groupby(sector).mean().abs().max() < 0.2  # 行业均值接近 0
    assert abs(np.corrcoef(n, log_mcap.reindex(n.index))[0, 1]) < 1e-6  # 与市值正交
