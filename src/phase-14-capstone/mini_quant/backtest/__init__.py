"""回测层：月频组合回测（消费组合层输出）、绩效、DSR、容量估算"""

from .engine import BacktestResult, MonthlyBacktester, cost_rates
from .metrics import PERIODS, capacity_at, capacity_curve, deflated, perf_summary, sqrt_impact_bp, trial_sharpes

__all__ = [
    "BacktestResult", "MonthlyBacktester", "cost_rates",
    "PERIODS", "capacity_at", "capacity_curve", "deflated", "perf_summary", "sqrt_impact_bp", "trial_sharpes",
]
