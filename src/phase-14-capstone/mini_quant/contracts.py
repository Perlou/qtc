"""
层间接口（Protocol）与数据契约

每一层只依赖这里定义的"接口 + 数据格式"，不依赖其他层的具体实现：
    - Protocol 类似 Go 的 interface / TypeScript 的 interface：只要方法签名对得上就算实现，
      不需要继承（结构化子类型），所以测试里可以随手写一个假实现替换真实组件
    - validate_* 函数是数据契约的"运行时检查"，流水线每一步都调用，失败即停

数据格式约定：
    FactorFrame   单日：index = 股票代码，columns = 因子名（已预处理，方向统一为越大越好）
                  多日：index = (date, code) 的 MultiIndex
    ScoreSeries   index = 股票代码，值 = 预测分数（越大越好）
    WeightSeries  index = 股票代码，值 = 目标权重，和为 1（满仓）
    RebalanceOrder 一条调仓指令（股数已按交易所规则取整）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

FactorFrame = pd.DataFrame
ScoreSeries = pd.Series
WeightSeries = pd.Series


# ==================== 数据契约：调仓指令 ====================


@dataclass(frozen=True)
class RebalanceOrder:
    symbol: str  # 6 位股票代码
    side: str  # "BUY" / "SELL"
    volume: int  # 股数（已取整）
    ref_price: float  # 参考价（信号日不复权收盘价）
    current_shares: int
    target_shares: int
    target_weight: float
    est_value: float  # 预估成交金额 = volume × ref_price
    est_fee: float  # 预估显性费用（Phase 4 CostModel）
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ==================== 各层接口 ====================


@runtime_checkable
class DataSource(Protocol):
    """数据层：更新数据并提供行情（课程里是静态快照，update 只检查数据是否覆盖到 end）"""

    def update(self, end: str) -> pd.Timestamp: ...

    def load_prices(self, start: str, end: str) -> pd.DataFrame: ...


@runtime_checkable
class FactorEngine(Protocol):
    """因子层：计算某日截面的预处理后因子"""

    def compute(self, date) -> FactorFrame: ...


@runtime_checkable
class Model(Protocol):
    """模型层：截面打分模型"""

    def fit(self, X: FactorFrame, y: pd.Series) -> "Model": ...

    def predict(self, X: FactorFrame) -> ScoreSeries: ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    """组合层：分数 → 目标权重"""

    def target_weights(self, scores: ScoreSeries, current: WeightSeries) -> WeightSeries: ...


@runtime_checkable
class OrderPlanner(Protocol):
    """组合层：目标权重 → 调仓指令（取整、先卖后买）"""

    def plan(
        self, target: WeightSeries, current_shares: pd.Series, prices: pd.Series, capital: float
    ) -> list[RebalanceOrder]: ...


@runtime_checkable
class Executor(Protocol):
    """执行层：执行调仓指令，返回执行报告"""

    def execute(self, orders: list[RebalanceOrder], market: dict) -> object: ...


# ==================== 校验：失败即停 ====================


class PipelineError(RuntimeError):
    """流水线校验失败：中止，不生成半成品指令"""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise PipelineError(msg)


def validate_factor_frame(X: FactorFrame, min_coverage: float = 0.8) -> None:
    check(isinstance(X, pd.DataFrame) and not X.empty, "因子矩阵为空")
    check(X.index.is_unique, "因子矩阵的股票代码有重复")
    values = X.to_numpy(dtype=float)
    check(not np.isinf(values).any(), "因子矩阵含 inf")
    coverage = X.notna().mean()
    worst = coverage.idxmin()
    check(coverage.min() >= min_coverage, f"因子缺失率过高：{worst} 覆盖率 {coverage.min():.0%} < {min_coverage:.0%}")


def validate_scores(scores: ScoreSeries, min_names: int) -> None:
    s = scores.dropna()
    check(len(s) >= min_names, f"有效分数只有 {len(s)} 个，少于选股数 {min_names}")
    check(np.isfinite(s.to_numpy(dtype=float)).all(), "预测分数含 inf")
    check(s.std() > 0, "预测分数无区分度（标准差为 0），模型可能异常")


def validate_weights(w: WeightSeries, max_weight: float, tol: float = 1e-6) -> None:
    check(len(w) > 0, "目标权重为空")
    check((w >= -tol).all(), "出现负权重（本策略只做多）")
    check(abs(w.sum() - 1) < tol, f"目标权重之和为 {w.sum():.6f}，不为 1")
    check(w.max() <= max_weight + tol, f"单票权重 {w.max():.2%} 超过上限 {max_weight:.2%}")


def validate_orders(orders: list[RebalanceOrder], lot_ok, cash: float, tol: float = 1e-6) -> None:
    """lot_ok(order) -> bool：判断股数是否符合板块规则；cash：下单前可用资金"""
    symbols = [o.symbol for o in orders]
    check(len(symbols) == len(set(symbols)), "同一标的出现多条指令")
    sides = [o.side for o in orders]
    check(sides == sorted(sides, key=lambda s: s != "SELL"), "指令未按先卖后买排序")
    bad = [o.symbol for o in orders if o.volume <= 0 or not lot_ok(o)]
    check(not bad, f"股数不符合交易单位规则：{bad[:5]}")
    sell_cash = sum(o.est_value - o.est_fee for o in orders if o.side == "SELL")
    buy_cash = sum(o.est_value + o.est_fee for o in orders if o.side == "BUY")
    check(buy_cash <= cash + sell_cash + tol, f"资金不足：买入需 {buy_cash:,.0f}，可用 {cash + sell_cash:,.0f}")
