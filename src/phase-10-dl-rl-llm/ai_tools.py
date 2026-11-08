"""
Phase 10 辅助模块：数据入口、随机种子、统计检验与试验日志
==========================================================

本模块不是课程文件，而是 01–10 课共用的工具层：

    路径与数据    load_panel（复用 Phase 7 截面面板）/ load_etf（单资产日线）
                  daily_fields / signal_dates / forward_open_returns / universe_mask
    随机性        set_seed（numpy / random / torch）
    评估          ic_stats / perf_stats / holm / benjamini_hochberg / deflated_sharpe
    记录          TrialLog（JSONL 试验日志：每一次尝试都要记，包括失败的）
    文本          extract_numbers / verify_numbers（数字溯源，07、10 课）

离线约定（与 utils.data 一致）：
    QTC_OFFLINE=1 时先读本地缓存，缓存不存在才用合成数据。
    Phase 7 的 factor_data.load_panel 在离线时直接生成合成面板，这里额外做了一步：
    若 Phase 7 面板缓存已存在，则离线时也读取缓存（不修改 Phase 7 文件）。
    设置 QTC_FORCE_SYNTHETIC=1 可强制使用合成面板（用于测试降级路径）。
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PHASE_DIR = Path(__file__).resolve().parent
SRC_DIR = PHASE_DIR.parent
P7_DIR = SRC_DIR / "phase-7-multi-factor"
for _p in (SRC_DIR, P7_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import factor_data  # noqa: E402
import factor_tools  # noqa: E402
from utils import OUTPUT_DIR  # noqa: E402
from utils.data import CACHE_DIR, get_etf_daily_adj, is_offline, is_synthetic  # noqa: E402

SUBDIR = "phase-10"
P10_CACHE = CACHE_DIR / "phase10"
P10_OUT = OUTPUT_DIR / SUBDIR

# ETF 交易成本示例值：佣金 + 滑点，单边（ETF 免印花税；以券商实际费率为准）
ETF_COST = 0.0005


# ==================== 随机种子 ====================


def set_seed(seed: int = 0, threads: int = 4) -> None:
    """固定 numpy / random / torch 的随机种子；torch 限制线程数，保证 CPU 上可复现、不抢占资源"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(threads)
    except ImportError:
        pass


# ==================== 数据入口 ====================


def load_panel(verbose: bool = True) -> factor_data.FactorPanel:
    """Phase 7 截面面板：离线时优先读缓存（见模块说明）"""
    if factor_data._PANEL is None:
        files = [factor_data.PANEL_FILE, factor_data.INFO_FILE, factor_data.FIN_FILE, factor_data.BENCH_FILE]
        if os.getenv("QTC_FORCE_SYNTHETIC") == "1":
            factor_data.warn_synthetic("多因子面板", "QTC_FORCE_SYNTHETIC=1")
            factor_data._PANEL = factor_data.synthetic_panel()
        elif is_offline() and all(p.exists() for p in files):
            factor_data._PANEL = factor_data._load_cached()
        else:
            return factor_data.load_panel(verbose)
        if verbose:
            p = factor_data._PANEL
            tag = "合成数据" if p.is_synthetic else "真实数据（本地缓存）"
            print(f"📦 面板（{tag}）：{len(p.codes)} 只股票，{p.dates[0].date()} ~ {p.dates[-1].date()}，"
                  f"{len(p.dates)} 个交易日，{p.info['sector'].nunique()} 个行业")
    return factor_data._PANEL


def load_etf(symbol: str = "510300", start: str = "20130101", end: str = "20241231") -> pd.DataFrame:
    """
    ETF 日线（修正分红/拆分）。返回列：open, high, low, close, volume, amount, ret, adj_close, adj_open
    adj_open = open × (adj_close / close)：与 adj_close 同口径的复权开盘价，用于"次日开盘成交"
    """
    if os.getenv("QTC_FORCE_SYNTHETIC") == "1":
        from utils.data import synthetic_ohlcv

        df = synthetic_ohlcv(symbol, start, end, start_price=4.0, annual_vol=0.20)
        df["ret"] = df["close"].pct_change()
        df["adj_close"] = df["close"]
        df.attrs["synthetic"] = True
    else:
        df = get_etf_daily_adj(symbol, start, end)
    df = df.copy()
    df["adj_open"] = df["open"] * df["adj_close"] / df["close"]
    df.attrs["synthetic"] = is_synthetic(df) or df.attrs.get("synthetic", False)
    return df


def daily_fields(panel) -> dict[str, pd.DataFrame]:
    """因子 DSL 可用的日频字段（价格均为后复权口径，保证跨除权日可比）"""
    adj = panel.adj_factor
    vwap = (panel.amount / panel.volume.where(panel.volume > 0)) * adj
    return {
        "open": panel.adj_open,
        "high": panel.high * adj,
        "low": panel.low * adj,
        "close": panel.adj_close,
        "volume": panel.volume,
        "amount": panel.amount,
        "turnover": panel.turnover,
        "vwap": vwap,
        "ret": panel.daily_ret,
    }


def signal_dates(panel, start: str, end: str, every: int = 5) -> pd.DatetimeIndex:
    """每 every 个交易日取一个信号日（t 日收盘后出信号）"""
    cal = panel.dates
    sel = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    return sel[::every]


def forward_open_returns(panel, dates: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """t+1 开盘买入、持有 h 个交易日后开盘卖出的收益（复用 Phase 7 的口径）"""
    return factor_tools.forward_returns(panel, dates, h)


def universe_mask(panel, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """可交易股票池：上市满一年、当日未停牌、非 ST、次日开盘未涨停（Phase 7 口径）"""
    return factor_tools.tradable_status(panel, dates)["all"]


# ==================== 统计 ====================


def ic_stats(ic: pd.Series) -> dict:
    """IC 序列的均值、ICIR、t 值、双侧 p 值（假设各期独立；重叠样本需改用 Newey-West）"""
    ic = pd.Series(ic).dropna()
    n = len(ic)
    if n < 3 or ic.std() == 0:
        return {"ic": np.nan, "icir": np.nan, "t": np.nan, "p": np.nan, "n": n, "pos": np.nan}
    m, s = ic.mean(), ic.std()
    t = m / s * np.sqrt(n)
    return {"ic": float(m), "icir": float(m / s), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), df=n - 1)), "n": n, "pos": float((ic > 0).mean())}


def perf_stats(r: pd.Series, periods: int = 252) -> dict:
    """日收益序列 → 年化收益、年化波动、夏普（未扣无风险利率）、最大回撤"""
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {"年化收益": np.nan, "年化波动": np.nan, "夏普": np.nan, "最大回撤": np.nan}
    nav = (1 + r).cumprod()
    sd = r.std()
    return {
        "年化收益": float(nav.iloc[-1] ** (periods / len(r)) - 1),
        "年化波动": float(sd * np.sqrt(periods)),
        "夏普": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan,
        "最大回撤": float((nav / nav.cummax() - 1).min()),
    }


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni 校正后的 p 值（控制族错误率 FWER）"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg 校正后的 q 值（控制错误发现率 FDR）"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        prev = min(prev, p[i] * m / (k + 1))
        q[i] = prev
    return q


def expected_max_sr(n_trials: int, sr_std: float) -> float:
    """N 次独立零效应试验中最大夏普的期望（Bailey & López de Prado 近似）"""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sr_std * ((1 - g) * z1 + g * z2))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_std: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    紧缩夏普比率 DSR：观察到的（单期）夏普 sr 超过"N 次试验最大值基准"的概率
    sr、sr_std 都用单期（非年化）口径；sr_std 为各次试验夏普的标准差
    """
    sr0 = expected_max_sr(n_trials, sr_std)
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))


# ==================== 试验日志 ====================


@dataclass
class TrialLog:
    """
    试验日志：无论成功、失败、被拒绝，每一次尝试都记录（多重检验的 N 来自这里）
    同时写入 JSONL 文件，便于事后审计
    """

    path: Path
    records: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # 每次运行重新开始

    def add(self, **rec) -> dict:
        rec = {"trial_id": len(self.records) + 1, "ts": time.strftime("%H:%M:%S"), **rec}
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()}
        self.records.append(clean)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
        return clean

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def __len__(self) -> int:
        return len(self.records)


# ==================== 文本：数字溯源 ====================

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numbers(text: str) -> set[float]:
    """抽取文本中所有数字（去掉千分位逗号），用于"数字必须能在原文找到"的校验"""
    out = set()
    for tok in _NUM_RE.findall(text):
        try:
            out.add(round(float(tok.replace(",", "")), 6))
        except ValueError:
            continue
    return out


def verify_numbers(claimed: dict[str, float | None], source_text: str, tol: float = 1e-6) -> list[str]:
    """
    claimed: {字段名: 数字}。返回问题列表（空列表 = 全部可溯源）
    规则：数字必须以相同数值出现在原文中（允许 1.20 与 1.2 这类格式差异）
    """
    pool = extract_numbers(source_text)
    issues = []
    for name, v in claimed.items():
        if v is None:
            continue
        if not any(abs(float(v) - x) <= tol for x in pool):
            issues.append(f"{name} = {v} 未在原文中出现")
    return issues


def fmt_pct(x: float, digits: int = 2) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"
