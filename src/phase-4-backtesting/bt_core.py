"""
Phase 4 回测公共模块（bt_core）
================================

前几课（02~05）会先把这些函数"手写一遍"，并与本模块的版本对拍；
从 06 课开始直接 `from bt_core import ...` 复用，避免每课重复实现。

内容：
    - 交易规则：板块规则（涨跌幅、最小买入单位）、Decimal 四舍五入、涨跌停价、整数手
    - 成本模型：佣金（最低 5 元）+ 印花税（仅卖出）+ 过户费 + 滑点
    - 数据准备：不复权 OHLC + 复权因子 + 除权参考前收盘、交易日历
    - 向量化回测：收盘成交 / 次日开盘成交两种假设
    - 绩效指标：CAGR、Sharpe、Sortino、最大回撤（含起止日期）、Calmar、交易统计、换手
    - 多重检验：PSR、期望最大 Sharpe、Deflated Sharpe

⚠️ 费率、涨跌幅等参数为编写时（2026 年）的公开规则示例值，实盘前请核对最新规定。
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PHASE_DIR = Path(__file__).resolve().parent
if str(PHASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR.parent))

from utils.data import get_index_daily, get_stock_daily_adj  # noqa: E402

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


# ==================== 交易规则 ====================

ST_RULE_CHANGE_DATE = pd.Timestamp("2026-07-06")  # 沪深主板 ST/*ST 涨跌幅 5% → 10% 的实施日


@dataclass(frozen=True)
class BoardRule:
    name: str
    limit_pct: float  # 涨跌幅限制
    min_lot: int  # 单笔最小买入股数
    step: int  # 超过最小数量后的递增单位


def board_rule(symbol: str, is_st: bool = False, date=None) -> BoardRule:
    """
    根据代码判断板块规则。规则随时间变化，回测要用"当时"的规则：
    主板 ST 股 2026-07-06 之前涨跌幅 5%，之后 10%（date 为空时按现行规则）。
    """
    if symbol.startswith(("688", "689")):
        return BoardRule("科创板", 0.20, 200, 1)
    if symbol.startswith(("300", "301")):
        return BoardRule("创业板", 0.20, 100, 100)
    if symbol.startswith(("8", "4", "92")):
        return BoardRule("北交所", 0.30, 100, 1)
    if is_st:
        if date is not None and pd.Timestamp(date) < ST_RULE_CHANGE_DATE:
            return BoardRule("主板ST", 0.05, 100, 100)
        return BoardRule("主板ST", 0.10, 100, 100)
    return BoardRule("主板", 0.10, 100, 100)


def round_price(x: float, tick: str = "0.01") -> float:
    """四舍五入到最小变动价位（交易所规则是四舍五入，不是 Python round 的银行家舍入）"""
    return float(Decimal(str(x)).quantize(Decimal(tick), rounding=ROUND_HALF_UP))


def limit_prices(pre_close: float, pct: float, tick: str = "0.01") -> tuple[float, float]:
    """涨停价、跌停价：前收盘 × (1 ± 涨跌幅)，用 Decimal 四舍五入到分"""
    pc, p = Decimal(str(pre_close)), Decimal(str(pct))
    up = (pc * (1 + p)).quantize(Decimal(tick), rounding=ROUND_HALF_UP)
    down = (pc * (1 - p)).quantize(Decimal(tick), rounding=ROUND_HALF_UP)
    return float(up), float(down)


def round_lot_buy(shares: float, rule: BoardRule) -> int:
    """买入股数向下取整到合法数量：主板 100 的整数倍；科创板至少 200 股，超出部分按 1 股递增"""
    shares = int(np.floor(shares + 1e-9))
    if shares < rule.min_lot:
        return 0
    return rule.min_lot + (shares - rule.min_lot) // rule.step * rule.step


# ==================== 成本模型 ====================


@dataclass(frozen=True)
class CostModel:
    """A 股股票交易成本。slippage 为单边滑点比例：买入价上浮、卖出价下浮。"""

    commission_rate: float = 0.00025  # 佣金：双向
    min_commission: float = 5.0  # 单笔最低佣金
    stamp_duty_rate: float = 0.0005  # 印花税：仅卖出（2023-08-28 起 0.05%）
    transfer_fee_rate: float = 0.00001  # 过户费：双向 0.001%
    slippage: float = 0.0005  # 单边滑点（5 bp）

    @classmethod
    def proportional(cls, rate: float) -> "CostModel":
        """简化成本：只有固定比例的单边费率，无最低佣金、无印花税、无滑点"""
        return cls(commission_rate=rate, min_commission=0.0, stamp_duty_rate=0.0, transfer_fee_rate=0.0, slippage=0.0)

    @classmethod
    def zero(cls) -> "CostModel":
        return cls.proportional(0.0)

    def fees(self, value: float, is_sell: bool) -> dict:
        """一笔成交的显性费用明细（元）"""
        if value <= 0:
            return {"commission": 0.0, "stamp_duty": 0.0, "transfer_fee": 0.0, "total": 0.0}
        commission = max(value * self.commission_rate, self.min_commission)
        stamp = value * self.stamp_duty_rate if is_sell else 0.0
        transfer = value * self.transfer_fee_rate
        return {"commission": commission, "stamp_duty": stamp, "transfer_fee": transfer,
                "total": commission + stamp + transfer}

    def total(self, value: float, is_sell: bool) -> float:
        return self.fees(value, is_sell)["total"]


def stamp_duty_rate_at(date) -> float:
    """
    A 股印花税（卖方）历史费率的简化表，用于跨年回测"按当时费率"计算：
      2008-09-19 起：仅卖方征收 0.1%
      2023-08-28 起：减半为 0.05%
    （更早的历史费率多次调整且曾双边征收，此处不展开）
    """
    d = pd.Timestamp(date)
    if d >= pd.Timestamp("2023-08-28"):
        return 0.0005
    return 0.001


# ==================== 数据准备 ====================


def add_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 get_stock_daily_adj 的结果上补充：
      adj_open:  后复权开盘价 = open × adj_factor
      pre_close: 前收盘价。除权除息日用"除权参考价"
                 = 上一交易日收盘 × 上一日因子 / 当日因子（与交易所计算涨跌停的基准一致）
    """
    df = df.copy()
    f = df["adj_factor"]
    df["adj_open"] = df["open"] * f
    ref = df["close"].shift(1) * f.shift(1) / f
    df["pre_close"] = ref.map(lambda x: round_price(x) if pd.notna(x) else np.nan)
    return df


def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """不复权 OHLCV + 复权因子 + adj_close/adj_open + 除权参考前收盘"""
    return add_reference_columns(get_stock_daily_adj(symbol, start, end))


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """交易日历：用沪深 300 指数的交易日近似（个股缺失的交易日即停牌日）"""
    return get_index_daily("000300", start, end).index


# ==================== 向量化回测 ====================


def vectorized_backtest(
    close: pd.Series,
    signal: pd.Series,
    cost_rate: float = 0.0,
    open_: pd.Series | None = None,
    execution: str = "close",
) -> pd.DataFrame:
    """
    单资产向量化回测（价格应为后复权价）。

    signal: t 日收盘后算出的目标仓位（0~1）
    execution:
      "close"     t 日收盘价成交，从 t+1 日起承担收益：pos = signal.shift(1)
      "next_open" t+1 日开盘价成交：t+1 日隔夜段由旧仓位承担，日内段由新仓位承担
    cost_rate: 单边成本率，按换手 |Δw| 扣除

    返回列：pos（当日日内持仓）、gross（扣费前收益）、cost、ret（扣费后收益）、nav、turnover
    """
    signal = signal.reindex(close.index).fillna(0.0).astype(float)
    if execution == "close":
        ret = close.pct_change().fillna(0.0)
        pos = signal.shift(1).fillna(0.0)
        turnover = pos.diff().abs()
        turnover.iloc[0] = abs(pos.iloc[0])
        gross = pos * ret
        cost = turnover * cost_rate
        net = gross - cost
    elif execution == "next_open":
        if open_ is None:
            raise ValueError("execution='next_open' 需要提供 open_（后复权开盘价）")
        w_new = signal.shift(1).fillna(0.0)  # t 日开盘后持仓（t-1 日收盘的信号）
        w_old = signal.shift(2).fillna(0.0)  # t 日隔夜持仓（t-2 日收盘的信号）
        r_on = (open_ / close.shift(1) - 1).fillna(0.0)  # 隔夜：昨收 → 今开
        r_id = (close / open_ - 1).fillna(0.0)  # 日内：今开 → 今收
        w_drift = w_old * (1 + r_on) / (1 + w_old * r_on)  # 隔夜涨跌后仓位的自然漂移
        turnover = (w_new - w_drift).abs()
        gross = (1 + w_old * r_on) * (1 + w_new * r_id) - 1
        net = (1 + w_old * r_on) * (1 - cost_rate * turnover) * (1 + w_new * r_id) - 1
        cost = gross - net
        pos = w_new
    else:
        raise ValueError(f"未知的 execution: {execution}")

    return pd.DataFrame({
        "pos": pos, "gross": gross, "cost": cost, "ret": net,
        "nav": (1 + net).cumprod(), "turnover": turnover,
    })


# ==================== 绩效指标 ====================


def drawdown_info(nav: pd.Series) -> dict:
    """最大回撤及其起点（前高）、谷底、修复日期（未修复为 NaT）"""
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    recovered = after[after >= nav.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else pd.NaT
    return {"mdd": float(dd.min()), "peak": peak, "trough": trough, "recovery": recovery}


def performance_report(ret: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> pd.Series:
    """常用绩效指标（ret 为周期简单收益）"""
    ret = ret.dropna()
    rf = rf_annual / periods
    excess = ret - rf
    nav = (1 + ret).cumprod()
    years = len(ret) / periods

    cagr = nav.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(periods)
    sharpe = excess.mean() / excess.std() * np.sqrt(periods) if excess.std() > 0 else np.nan
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())  # 分母是全部样本数
    sortino = excess.mean() / downside * np.sqrt(periods) if downside > 0 else np.nan
    dd = drawdown_info(nav)
    calmar = cagr / abs(dd["mdd"]) if dd["mdd"] < 0 else np.nan

    return pd.Series({
        "年化收益": cagr, "年化波动": vol, "Sharpe": sharpe, "Sortino": sortino,
        "最大回撤": dd["mdd"], "回撤起点": dd["peak"], "回撤谷底": dd["trough"], "回撤修复": dd["recovery"],
        "Calmar": calmar, "日胜率(非零日)": (ret[ret != 0] > 0).mean() if (ret != 0).any() else np.nan,
        "Sharpe t值": sharpe * np.sqrt(years), "样本年数": years,
    })


def extract_trades(pos: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """
    把持仓序列切成一笔笔交易：连续 pos != 0 的区间算一笔。
    交易收益 = 区间内（含平仓后第一天，以计入平仓成本/隔夜段）策略收益的复利。
    """
    held = (pos.abs() > 1e-12).to_numpy()
    rows, i, n = [], 0, len(pos)
    while i < n:
        if not held[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and held[j + 1]:
            j += 1
        k = min(j + 1, n - 1)  # 平仓日（若仍持有到最后则为最后一天）
        r = float((1 + ret.iloc[i : k + 1]).prod() - 1)
        rows.append({"entry": pos.index[i], "exit": pos.index[k], "days": j - i + 1, "ret": r,
                     "closed": j + 1 < n})
        i = j + 1
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "ret", "closed"])


def trade_stats(trades: pd.DataFrame) -> pd.Series:
    """按"笔"统计：胜率、盈亏比、Profit Factor、平均持仓天数"""
    if trades.empty:
        return pd.Series({"交易笔数": 0, "胜率": np.nan, "盈亏比": np.nan, "Profit Factor": np.nan, "平均持仓天数": np.nan})
    r = trades["ret"]
    wins, losses = r[r > 0], r[r <= 0]
    payoff = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() != 0 else np.nan
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return pd.Series({"交易笔数": len(r), "胜率": len(wins) / len(r), "盈亏比": payoff,
                      "Profit Factor": pf, "平均持仓天数": trades["days"].mean()})


def annual_turnover(turnover: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化换手率 = Σ|Δw| / 年数（买和卖都计入）"""
    return float(turnover.sum() / (len(turnover) / periods))


def cost_summary(bt: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """成本相关指标：年化换手、年化成本拖累、成本占扣费前收益的比例、盈亏平衡成本"""
    years = len(bt) / periods
    to = annual_turnover(bt["turnover"], periods)
    drag = bt["cost"].sum() / years
    gross_annual = bt["gross"].sum() / years  # 算术年化（与成本拖累同口径）
    return pd.Series({
        "年化换手": to,
        "年化成本拖累": drag,
        "成本占比": drag / gross_annual if gross_annual > 0 else np.nan,
        "盈亏平衡成本(bp)": gross_annual / to * 1e4 if to > 0 else np.nan,
    })


def format_report(rep: pd.Series) -> pd.Series:
    """把 performance_report 的结果格式化为便于打印的字符串"""
    pct = {"年化收益", "年化波动", "最大回撤", "日胜率(非零日)", "胜率", "年化成本拖累", "成本占比"}
    out = {}
    for k, v in rep.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.date().isoformat()
        elif v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
            out[k] = "—"
        elif k in pct:
            out[k] = f"{v:.2%}"
        elif isinstance(v, (int, np.integer)):
            out[k] = f"{v:d}"
        else:
            out[k] = f"{v:.2f}"
    return pd.Series(out)


def report_table(rets: dict[str, pd.Series], rf_annual: float = 0.0) -> pd.DataFrame:
    """多条收益序列的绩效对比表（已格式化）"""
    return pd.DataFrame({k: format_report(performance_report(v, rf_annual)) for k, v in rets.items()})


# ==================== 多重检验：PSR / DSR ====================


def sharpe_ratio(ret, periods: int = 1) -> float:
    """Sharpe；periods=1 表示不年化（PSR/DSR 公式用不年化的值）"""
    ret = np.asarray(ret, dtype=float)
    sd = ret.std(ddof=1)
    return float(ret.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def psr(ret, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio：观测 Sharpe 显著超过基准（不年化）的概率"""
    ret = np.asarray(ret, dtype=float)
    sr = sharpe_ratio(ret)
    skew = stats.skew(ret)
    kurt = stats.kurtosis(ret, fisher=False)  # 非超额峰度，正态为 3
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(len(ret) - 1) / denom))


def expected_max_sr(trial_srs, n_trials: int | None = None) -> float:
    """N 次独立试验、真实 Sharpe 全为 0 时，最大 Sharpe 的期望（不年化）"""
    trial_srs = np.asarray(trial_srs, dtype=float)
    n = n_trials or len(trial_srs)
    return float(np.sqrt(trial_srs.var(ddof=1)) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e))
    ))


def deflated_sharpe(ret, trial_srs, n_trials: int | None = None) -> float:
    """DSR = PSR(期望最大 Sharpe)"""
    return psr(ret, expected_max_sr(trial_srs, n_trials))


# ==================== 其他 ====================


def load_lesson(filename: str):
    """按文件名导入本阶段的课程模块（文件名以数字开头，不能直接 import）"""
    path = PHASE_DIR / filename
    name = "lesson_" + path.stem.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
