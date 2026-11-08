"""
因子注册表：插件式管理因子

    @factor("mom_20", category="动量", description="20 日动量")
    def mom_20(ctx: FactorContext) -> pd.DataFrame: ...

每个因子是一个函数：输入 FactorContext（面板 + 调仓日），输出宽表（index = 调仓日，
columns = 股票代码），方向统一为"值越大，预期收益越高"。注册即可被配置文件启用，
新增因子不需要改动平台的任何其他代码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd

FactorFn = Callable[["FactorContext"], pd.DataFrame]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: FactorFn
    category: str = ""
    description: str = ""
    source: str = ""


class FactorRegistry:
    def __init__(self):
        self._specs: dict[str, FactorSpec] = {}

    def register(self, name: str, *, category: str = "", description: str = "", source: str = "",
                 overwrite: bool = False) -> Callable[[FactorFn], FactorFn]:
        """装饰器：把函数注册为因子。重名默认报错，防止无意中覆盖已有因子"""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            raise ValueError(f"因子名只能包含字母、数字和下划线：{name!r}")

        def deco(fn: FactorFn) -> FactorFn:
            if name in self._specs and not overwrite:
                raise ValueError(f"因子 {name!r} 已注册（来源：{self._specs[name].source or '未知'}）")
            self._specs[name] = FactorSpec(name, fn, category, description, source)
            return fn

        return deco

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def get(self, name: str) -> FactorSpec:
        if name not in self._specs:
            raise KeyError(f"未注册的因子：{name!r}，已注册：{self.names()}")
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[s.name, s.category, s.description, s.source] for s in self._specs.values()],
            columns=["因子", "类别", "说明", "来源"],
        )

    def compute(self, name: str, ctx: "FactorContext") -> pd.DataFrame:
        out = self.get(name).fn(ctx)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"因子 {name} 必须返回 DataFrame，实际为 {type(out).__name__}")
        return out.reindex(index=ctx.dates, columns=ctx.codes).replace([np.inf, -np.inf], np.nan)


REGISTRY = FactorRegistry()
factor = REGISTRY.register  # 全局注册表的装饰器


class FactorContext:
    """
    因子函数的输入：面板 + 调仓日。
    at(df, lag)：取每个调仓日往前 lag 个交易日的截面（只用过去数据，避免前视）
    phase7_raw：Phase 7 build_factors 的结果（含 PIT 财务因子），按需计算一次
    """

    def __init__(self, panel, dates: pd.DatetimeIndex):
        self.panel = panel
        self.dates = pd.DatetimeIndex(dates)
        self.codes = panel.codes
        self._pos = panel.dates.get_indexer(self.dates)
        if (self._pos < 0).any():
            raise ValueError("调仓日必须是面板中的交易日")

    def at(self, df: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
        idx = self._pos - lag
        out = df.iloc[np.clip(idx, 0, None)].set_axis(self.dates)
        return out.where(pd.Series(idx >= 0, index=self.dates), axis=0)

    @cached_property
    def phase7_raw(self) -> dict[str, pd.DataFrame]:
        import factor_tools

        return factor_tools.build_factors(self.panel, self.dates)
