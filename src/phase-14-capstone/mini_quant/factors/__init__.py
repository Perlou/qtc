"""因子层：注册表（装饰器）、预处理（MAD + Z-score + 中性化）、单因子 IC 诊断"""

from . import builtin  # noqa: F401  （导入即注册内置因子）
from .diagnostics import diagnose, ic_series, ic_table
from .engine import RegistryFactorEngine
from .preprocess import preprocess_cross_section, preprocess_panel
from .registry import REGISTRY, FactorContext, FactorRegistry, FactorSpec, factor

__all__ = [
    "REGISTRY", "FactorContext", "FactorRegistry", "FactorSpec", "factor",
    "RegistryFactorEngine", "preprocess_cross_section", "preprocess_panel",
    "ic_series", "ic_table", "diagnose",
]
