"""
滚动训练（Walk-Forward）与模型版本记录

时间线（月频，d_i 为第 i 个调仓日）：

    d_{i-2}            d_{i-1}            d_i
      │ 信号            │                  │ 现在要打分
      └─ 持有期：d_{i-2}+1 开盘 ──▶ d_{i-1}+1 开盘   （d_i 收盘时已知 ✅）
                         └─ 持有期：d_{i-1}+1 开盘 ──▶ d_i+1 开盘   （d_i 收盘时未知 ❌）

所以在 d_i 训练时，最新可用的标签是 d_{i-2} 那一期（label_lag = 2）。
直接用 d_{i-1} 的标签，就是"用明天开盘价训练今天的模型"——典型的前视偏差。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

LABEL_LAG = 2


@dataclass
class ModelVersion:
    version: str
    fit_date: pd.Timestamp  # 在哪个调仓日训练（之后用于打分，直到下一次重训）
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # 最后一期标签所属的调仓日
    n_samples: int
    features: list[str]
    model: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = ",".join(self.features)
        return d


def training_dates(dates: pd.DatetimeIndex, as_of, window: int, label_lag: int = LABEL_LAG) -> pd.DatetimeIndex:
    """as_of 收盘时可用于训练的调仓日（标签已完整实现）"""
    i = dates.get_indexer([pd.Timestamp(as_of)])[0]
    if i < 0:
        raise ValueError(f"{as_of} 不是调仓日")
    last = i - label_lag
    if last < 0:
        return pd.DatetimeIndex([])
    return dates[max(0, last - window + 1) : last + 1]


def training_set(X_long: pd.DataFrame, y_long: pd.Series, train_dates: pd.DatetimeIndex):
    d = X_long.index.get_level_values("date")
    X = X_long[d.isin(train_dates)]
    y = y_long[y_long.index.get_level_values("date").isin(train_dates)]
    return X, y


class RollingTrainer:
    def __init__(self, model_factory: Callable[[], object], window: int = 36, min_train: int = 12,
                 retrain_every: int = 1, label_lag: int = LABEL_LAG):
        self.model_factory = model_factory
        self.window = window
        self.min_train = min_train
        self.retrain_every = retrain_every
        self.label_lag = label_lag
        self.versions: list[ModelVersion] = []
        self.model = None

    def run(self, X_long: pd.DataFrame, y_long: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """逐个调仓日：必要时重训 → 用"当时已训练好"的模型打分。返回分数宽表（index = 调仓日）"""
        scores, last_fit = {}, None
        by_date = dict(tuple(X_long.groupby(level="date")))
        for i, d in enumerate(dates):
            tr = training_dates(dates, d, self.window, self.label_lag)
            if len(tr) < self.min_train:
                continue
            if self.model is None or last_fit is None or i - last_fit >= self.retrain_every:
                X, y = training_set(X_long, y_long, tr)
                if len(y) == 0:
                    continue
                self.model = self.model_factory().fit(X, y)
                last_fit = i
                self.versions.append(ModelVersion(
                    version=f"v{len(self.versions) + 1:03d}", fit_date=d, train_start=tr[0], train_end=tr[-1],
                    n_samples=len(y), features=list(X.columns), model=getattr(self.model, "name", "?"),
                    detail=self.model.describe() if hasattr(self.model, "describe") else {},
                ))
            Xd = by_date[d].droplevel("date")
            scores[d] = self.model.predict(Xd)
        return pd.DataFrame(scores).T.reindex(columns=X_long.index.get_level_values("code").unique())

    def versions_frame(self) -> pd.DataFrame:
        return pd.DataFrame([v.to_dict() for v in self.versions])
