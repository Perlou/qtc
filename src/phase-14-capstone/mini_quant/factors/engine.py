"""
因子引擎：注册表 + 预处理（实现 contracts.FactorEngine）

    engine = RegistryFactorEngine(source, ["ep", "bp", ...])
    X = engine.compute("2024-12-31")          # 单日 FactorFrame：index = 股票，columns = 因子
    panel = engine.panel(dates)               # {因子名: 预处理后宽表}，研究 / 回测用
    X_long = engine.long_frame(dates)         # (date, code) 长表，模型训练用

缓存键包含因子名、调仓日和预处理参数；数据版本由进程内唯一的面板决定（面板更新需重启进程，
实盘中缓存键还应包含数据内容哈希，见 CONCEPT 3.3 节"因子缓存过期"）。
"""

from __future__ import annotations

import pandas as pd

from .registry import REGISTRY, FactorContext, FactorRegistry
from .preprocess import preprocess_panel
from . import builtin  # noqa: F401  （注册内置因子）


class RegistryFactorEngine:
    def __init__(self, source, names: list[str] | tuple[str, ...], neutralize: bool = True,
                 registry: FactorRegistry = REGISTRY, use_tradable_mask: bool = True):
        missing = [n for n in names if n not in registry]
        if missing:
            raise KeyError(f"配置中启用了未注册的因子：{missing}")
        self.source = source
        self.names = list(names)
        self.neutralize = neutralize
        self.registry = registry
        self.use_tradable_mask = use_tradable_mask
        self._raw: dict = {}
        self._clean: dict = {}

    # ---------- 宽表 ----------

    def raw(self, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        key = tuple(dates)
        if key not in self._raw:
            self._raw[key] = {"__ctx__": FactorContext(self.source.panel, dates)}
        store = self._raw[key]
        for n in self.names:
            if n not in store:
                store[n] = self.registry.compute(n, store["__ctx__"])
        return {n: store[n] for n in self.names}

    def panel(self, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """预处理后的因子宽表：只在可交易股票上做截面标准化（不可交易的为 NaN）"""
        dates = pd.DatetimeIndex(dates)
        key = (tuple(dates), self.neutralize)
        store = self._clean.setdefault(key, {})
        todo = [n for n in self.names if n not in store]
        if todo:
            raw = self.raw(dates)
            mask = self.source.tradable(dates)["all"] if self.use_tradable_mask else None
            for n in todo:
                store[n] = preprocess_panel(raw[n], self.source.panel, mask, neutral=self.neutralize,
                                            size_neutral=(n != "size"))
        return {n: store[n] for n in self.names}

    def long_frame(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        clean = self.panel(dates)
        out = pd.concat({n: df.stack(future_stack=True) for n, df in clean.items()}, axis=1)
        out.index.names = ["date", "code"]
        return out

    # ---------- FactorEngine 接口 ----------

    def compute(self, date) -> pd.DataFrame:
        d = pd.Timestamp(date)
        rebal = self.source.rebalance_dates()
        dates = rebal if d in rebal else pd.DatetimeIndex([d])
        clean = self.panel(dates)
        X = pd.DataFrame({n: clean[n].loc[d] for n in self.names})
        X.index.name = "code"
        return X
