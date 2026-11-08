"""
截面打分模型（实现 contracts.Model）

    equal   等权合成：各因子 Z-score 直接平均（无需训练，最稳健的基准）
    icw     IC 加权：按训练窗口内各因子的平均 Rank IC 加权（Phase 7 09 课）
    ridge   岭回归：截面收益排名对因子回归（scikit-learn Ridge）
    hgb     梯度提升树：HistGradientBoostingRegressor（LightGBM 在本环境不可用，用 sklearn 实现；
            参数沿用 Phase 9 "低信噪比保守参数"的思路：浅树、大叶子、强正则）

训练标签：持有期收益的截面百分位排名 − 0.5。用排名而不是原始收益：
去掉市场整体涨跌（与选股无关），并降低极端收益对低信噪比回归的影响。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

MIN_FEATURE_RATIO = 0.5  # 至少一半因子有值才打分


def make_labels(period_ret: pd.DataFrame) -> pd.Series:
    """宽表持有期收益 → 长表标签 (date, code)：截面百分位排名 − 0.5"""
    y = period_ret.rank(axis=1, pct=True) - 0.5
    s = y.stack()
    s.index.names = ["date", "code"]
    return s.rename("label")


def _valid_rows(X: pd.DataFrame) -> pd.Series:
    return X.notna().mean(axis=1) >= MIN_FEATURE_RATIO


class EqualWeightModel:
    name = "equal"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EqualWeightModel":
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.mean(axis=1).where(_valid_rows(X)).rename("score")

    def describe(self) -> dict:
        return {"weights": "等权"}


class ICWeightModel:
    name = "icw"

    def __init__(self):
        self.weights_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ICWeightModel":
        df = X.join(y.rename("__y__"), how="inner")
        ics = df.groupby(level="date").apply(
            lambda g: g.drop(columns="__y__").corrwith(g["__y__"], method="spearman"))
        mean_ic = ics.mean().fillna(0.0)
        total = mean_ic.abs().sum()
        self.weights_ = mean_ic / total if total > 0 else pd.Series(1 / len(mean_ic), index=mean_ic.index)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        w = self.weights_.reindex(X.columns).fillna(0.0)
        return (X.fillna(0.0) @ w).where(_valid_rows(X)).rename("score")

    def describe(self) -> dict:
        return {"weights": self.weights_.round(3).to_dict()}


class SklearnModel:
    """把 sklearn 回归器包装成 Model 接口；Z-score 后的缺失值填 0（= 截面均值，中性）"""

    def __init__(self, name: str, estimator, fill_na: bool = True):
        self.name = name
        self.estimator = estimator
        self.fill_na = fill_na
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnModel":
        df = X.join(y.rename("__y__"), how="inner")
        df = df[_valid_rows(df.drop(columns="__y__")) & df["__y__"].notna()]
        feats = df.drop(columns="__y__")
        self.columns_ = list(feats.columns)
        Xv = feats.fillna(0.0) if self.fill_na else feats
        self.estimator.fit(Xv.to_numpy(dtype=float), df["__y__"].to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        feats = X[self.columns_]
        Xv = feats.fillna(0.0) if self.fill_na else feats
        pred = pd.Series(self.estimator.predict(Xv.to_numpy(dtype=float)), index=X.index, name="score")
        return pred.where(_valid_rows(feats))

    def describe(self) -> dict:
        est = self.estimator
        if hasattr(est, "coef_"):
            return {"coef": dict(zip(self.columns_, np.round(est.coef_, 4)))}
        return {"params": {k: v for k, v in est.get_params().items() if v is not None}}


def make_model(name: str, seed: int = 42):
    if name == "equal":
        return EqualWeightModel()
    if name == "icw":
        return ICWeightModel()
    if name == "ridge":
        return SklearnModel("ridge", Ridge(alpha=50.0))
    if name == "hgb":
        est = HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
            l2_regularization=10.0, random_state=seed,
        )
        return SklearnModel("hgb", est, fill_na=False)  # 原生支持缺失值
    raise ValueError(f"未知模型：{name}（可选 equal / icw / ridge / hgb）")
