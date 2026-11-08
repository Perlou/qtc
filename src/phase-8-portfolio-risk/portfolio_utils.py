"""
Phase 8 辅助模块：资产池数据 + 组合计算常用函数
==============================================

本模块不是课程文件，而是 01–10 课共用的工具：

1. 大类资产 ETF 数据
   - load_etf_returns(symbols)    多只 ETF 的"含分红/拆分修正"日收益宽表（列名为中文简称）
   - etf_total_return(raw, hfq)   用"不复权价 + 东方财富后复权价"还原真实日收益
   - etf_events(...)              列出检测到的分红 / 拆分事件（01 课展示用）

   为什么不直接用东方财富的 hfq？
   东方财富 ETF 的复权方式不统一：有的 ETF 是"加减法"复权（hfq = 不复权价 + 累计分红，
   除息日以外的日收益都被压小），有的是"等比"复权（拆分、份额折算处理正确）。
   所以本模块只在"事件日"使用复权信息，其余日子用不复权价格的真实涨跌：
       加减法区间的分红日：  r = close_t / (close_{t-1} - 分红) - 1
       等比区间的事件日：    r = hfq_t / hfq_{t-1} - 1

2. 离线回退（数据来源看 utils.data.is_synthetic，而不是看"是否离线"）
   QTC_OFFLINE=1 时先读本地缓存，此时与正常模式完全相同。缓存缺失（或联网下载失败）时，
   utils 返回的合成行情彼此独立、参数相同，没有相关结构，组合优化课程会失去意义。
   这时改用 simulate_returns()：按"示意性"的大类资产年化收益、波动和相关系数，
   生成多元 t 分布日收益（有肥尾和相关性），并打印 ⚠️。
   load_etf_returns / load_stock_returns 的返回值都带 attrs["synthetic"]，可用 is_synthetic(df) 判断。
   ⚠️ 模拟参数只是教学假设，不能用来得出任何关于真实市场的结论。

3. 组合计算
   - gmv_weights / tangency_weights        闭式解（允许做空）
   - risk_contribution                     风险贡献 RC_i = w_i (Σw)_i / σ_p
   - perf_stats / perf_table               年化收益、波动、Sharpe、最大回撤
   - walk_forward                          滚动估计 + 定期再平衡 + 权重漂移 的样本外回测
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import get_etf_daily, get_stock_daily_adj, is_synthetic  # noqa: E402

TRADING_DAYS = 252
DATA_START, DATA_END = "20130101", "20241231"

# ==================== 资产池 ====================
# 代码 → (简称, 资产类别)
ETF_UNIVERSE: dict[str, tuple[str, str]] = {
    "510300": ("沪深300", "A股"),
    "510500": ("中证500", "A股"),
    "159915": ("创业板", "A股"),
    "513100": ("纳指", "海外股票"),
    "513500": ("标普500", "海外股票"),
    "511010": ("国债", "债券"),
    "518880": ("黄金", "商品"),
    "159985": ("豆粕", "商品"),
}
# 豆粕 ETF 2019 年底才上市，长历史课程默认不含它（见 01 课的起始日期对齐）
CORE_ETFS = ["510300", "510500", "159915", "513100", "513500", "511010", "518880"]

# 颜色跟随资产（固定顺序，不随排名变化）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ASSET_COLORS = {name: PALETTE[i] for i, (name, _) in enumerate(ETF_UNIVERSE.values())}

# 离线模拟用的"示意性"参数：年化收益、年化波动（教学假设，非真实估计）
_SIM_PARAMS = {
    "沪深300": (0.06, 0.22),
    "中证500": (0.07, 0.26),
    "创业板": (0.08, 0.32),
    "纳指": (0.12, 0.24),
    "标普500": (0.10, 0.18),
    "国债": (0.03, 0.025),
    "黄金": (0.07, 0.15),
    "豆粕": (0.04, 0.20),
}
_SIM_CORR = pd.DataFrame(
    [
        # 沪深300 中证500 创业板 纳指  标普  国债   黄金  豆粕
        [1.00, 0.85, 0.80, 0.20, 0.18, -0.15, 0.02, 0.08],
        [0.85, 1.00, 0.85, 0.18, 0.16, -0.15, 0.02, 0.08],
        [0.80, 0.85, 1.00, 0.20, 0.17, -0.12, 0.00, 0.06],
        [0.20, 0.18, 0.20, 1.00, 0.90, -0.05, 0.10, 0.03],
        [0.18, 0.16, 0.17, 0.90, 1.00, -0.05, 0.12, 0.03],
        [-0.15, -0.15, -0.12, -0.05, -0.05, 1.00, 0.10, -0.02],
        [0.02, 0.02, 0.00, 0.10, 0.12, 0.10, 1.00, 0.05],
        [0.08, 0.08, 0.06, 0.03, 0.03, -0.02, 0.05, 1.00],
    ],
    index=list(_SIM_PARAMS),
    columns=list(_SIM_PARAMS),
)

# 个股池：5 个行业 × 4 只（06、09 课使用）
STOCK_UNIVERSE: dict[str, tuple[str, str]] = {
    "601398": ("工商银行", "银行"),
    "600036": ("招商银行", "银行"),
    "601166": ("兴业银行", "银行"),
    "601288": ("农业银行", "银行"),
    "600519": ("贵州茅台", "食品饮料"),
    "000858": ("五粮液", "食品饮料"),
    "600887": ("伊利股份", "食品饮料"),
    "000568": ("泸州老窖", "食品饮料"),
    "600276": ("恒瑞医药", "医药"),
    "603259": ("药明康德", "医药"),
    "000538": ("云南白药", "医药"),
    "300760": ("迈瑞医疗", "医药"),
    "601088": ("中国神华", "周期资源"),
    "600900": ("长江电力", "周期资源"),
    "600028": ("中国石化", "周期资源"),
    "601899": ("紫金矿业", "周期资源"),
    "300750": ("宁德时代", "科技制造"),
    "002594": ("比亚迪", "科技制造"),
    "002415": ("海康威视", "科技制造"),
    "000333": ("美的集团", "科技制造"),
}


def etf_names(symbols: list[str]) -> list[str]:
    return [ETF_UNIVERSE[s][0] for s in symbols]


def asset_class_of(names: list[str]) -> pd.Series:
    """中文简称 → 资产类别"""
    lookup = {n: c for n, c in ETF_UNIVERSE.values()}
    return pd.Series([lookup[n] for n in names], index=names)


# ==================== 复权：还原 ETF 真实日收益 ====================


def etf_total_return(raw_close: pd.Series, hfq_close: pd.Series, tol: float = 0.002) -> tuple[pd.Series, pd.DataFrame]:
    """
    用不复权价 + 东方财富后复权价，还原分红、拆分修正后的真实日收益

    事件日判定：差值 A = hfq - raw 与比值 M = hfq / raw 当天同时跳变
        - 纯加减法区间：A 恒定、M 随价格变 → 平时不会触发
        - 纯等比区间：  M 恒定、A 随价格变 → 平时不会触发
    事件日之后若 A 保持恒定 → 加减法复权，按"现金分红"计算；否则按等比复权的收益。

    Returns:
        (日收益 Series, 事件明细 DataFrame)
    """
    raw = raw_close.astype(float)
    hfq = hfq_close.reindex(raw.index).astype(float)
    r = raw.pct_change()
    r_hfq = hfq.pct_change()

    dA = (hfq - raw).diff()
    dM = (hfq / raw).pct_change()
    is_event = (dA.abs() > tol) & (dM.abs() > tol)

    rows = []
    idx = list(raw.index)
    for day in raw.index[is_event.fillna(False).to_numpy()]:
        i = idx.index(day)
        after = dA.iloc[i + 1 : i + 6]
        additive = len(after) > 0 and bool((after.abs() < 1e-4).all())
        prev = raw.iloc[i - 1]
        if additive and 0 < dA.iloc[i] < 0.2 * prev:
            kind, cash = "现金分红（加减法复权）", dA.iloc[i]
            fixed = raw.iloc[i] / (prev - cash) - 1
        else:
            ratio = raw.iloc[i - 1] / raw.iloc[i] * (1 + r_hfq.iloc[i])
            kind = "拆分/折算（等比复权）" if abs(ratio - 1) > 0.05 else "分红（等比复权）"
            cash, fixed = np.nan, r_hfq.iloc[i]
        rows.append({"date": day, "类型": kind, "每份分红": cash, "不复权收益": r.iloc[i],
                     "东财hfq收益": r_hfq.iloc[i], "修正后收益": fixed})
        r.iloc[i] = fixed
    events = pd.DataFrame(rows, columns=["date", "类型", "每份分红", "不复权收益", "东财hfq收益", "修正后收益"])
    return r, events


def _load_one_etf(symbol: str, start: str, end: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, bool]:
    """返回 (不复权价, 后复权价, 修正后日收益, 事件明细, 是否含合成数据)"""
    raw_df = get_etf_daily(symbol, start, end)
    hfq_df = get_etf_daily(symbol, start, end, adjust="hfq")
    raw, hfq = raw_df["close"], hfq_df["close"]
    synthetic = is_synthetic(raw_df) or is_synthetic(hfq_df)
    if synthetic:  # 合成价格没有公司行为；真实价配合成复权价也会算出虚假事件，所以不做修正
        return raw, hfq, raw.pct_change(), pd.DataFrame(), True
    ret, events = etf_total_return(raw, hfq)
    return raw, hfq, ret, events, False


def etf_data_report(symbols: list[str] | None = None, start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    """每只 ETF 的起始日期、事件数和三种口径的累计收益（01 课展示复权口径用）"""
    symbols = symbols or list(ETF_UNIVERSE)
    rows, all_events = [], []
    for s in symbols:
        raw, hfq, ret, events, synthetic = _load_one_etf(s, start, end)
        rows.append({
            "代码": s, "简称": ETF_UNIVERSE[s][0], "类别": ETF_UNIVERSE[s][1],
            "数据": "合成" if synthetic else "真实",
            "首个交易日": raw.index[0].date(), "交易日数": len(raw), "事件数": len(events),
            "不复权累计": raw.iloc[-1] / raw.iloc[0] - 1,
            "东财hfq累计": hfq.iloc[-1] / hfq.iloc[0] - 1,
            "修正后累计": (1 + ret.fillna(0)).prod() - 1,
        })
        if len(events):
            all_events.append(events.assign(简称=ETF_UNIVERSE[s][0]))
    report = pd.DataFrame(rows)
    report.attrs["events"] = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return report


def simulate_returns(
    names: list[str],
    start: str = DATA_START,
    end: str = DATA_END,
    dof: float = 5.0,
    seed: int = 8,
) -> pd.DataFrame:
    """按示意参数生成相关的多元 t 分布日收益（工作日索引）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), name="date")
    mu = np.array([_SIM_PARAMS[n][0] for n in names]) / TRADING_DAYS
    vol = np.array([_SIM_PARAMS[n][1] for n in names]) / np.sqrt(TRADING_DAYS)
    cov = _SIM_CORR.loc[names, names].to_numpy() * np.outer(vol, vol)
    chol = np.linalg.cholesky(cov * (dof - 2) / dof)  # 缩放后 t 分布的协方差 = cov
    z = rng.standard_normal((len(dates), len(names))) @ chol.T
    chi = rng.chisquare(dof, size=(len(dates), 1))
    return pd.DataFrame(mu + z * np.sqrt(dof / chi), index=dates, columns=names)


def load_etf_returns(
    symbols: list[str] | None = None,
    start: str = DATA_START,
    end: str = DATA_END,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    多只 ETF 的日收益宽表（列 = 中文简称），已做复权修正和起始日期对齐

    对齐规则：从"最晚上市的那只"的第二个交易日开始（第一天没有收益）；
    之后个别 ETF 停牌造成的缺失按 0 收益处理（停牌期间价格不变）。
    """
    symbols = symbols or CORE_ETFS
    names = etf_names(symbols)

    rets, firsts = {}, {}
    for s, n in zip(symbols, names):
        _, _, ret, _, synthetic = _load_one_etf(s, start, end)
        if synthetic:
            # 只要有一只拿不到真实数据就整体改用模拟：真实收益和彼此独立的合成收益混在一起，相关结构是错的
            print(f"⚠️ {n} ETF 没有本地缓存或下载失败：utils 的合成行情彼此独立，组合优化需要相关结构，"
                  "全部资产改用示意参数的多元 t 分布模拟收益（仅用于跑通代码）")
            panel = simulate_returns(names, start, end)
            panel.attrs["synthetic"] = True
            return panel
        rets[n] = ret
        firsts[n] = ret.index[0]
    panel = pd.DataFrame(rets).sort_index()
    common_start = max(firsts.values())
    panel = panel.loc[panel.index > common_start]
    n_missing = int(panel.isna().sum().sum())
    panel = panel.fillna(0.0)
    if verbose:
        latest = max(firsts, key=firsts.get)
        print(f"📅 起始日期对齐：最晚上市的是 {latest}（{common_start.date()}），"
              f"样本 {panel.index[0].date()} ~ {panel.index[-1].date()}，{len(panel)} 个交易日；"
              f"停牌缺失 {n_missing} 个按 0 收益处理")
    panel.attrs["synthetic"] = False
    return panel


def load_stock_returns(start: str = "20210101", end: str = "20241231") -> tuple[pd.DataFrame, pd.Series]:
    """个股池日收益（等比后复权）与行业标签；列名为股票简称"""
    codes = list(STOCK_UNIVERSE)
    frames = {c: get_stock_daily_adj(c, start, end) for c in codes}  # 与 get_close_panel 同口径，但保留数据来源标记
    n_synth = sum(is_synthetic(f) for f in frames.values())
    close = pd.DataFrame({STOCK_UNIVERSE[c][0]: f["adj_close"] for c, f in frames.items()}).sort_index()
    ret = close.pct_change(fill_method=None).iloc[1:]
    ret = ret.dropna(axis=1, how="all").fillna(0.0)
    industry = pd.Series({STOCK_UNIVERSE[c][0]: STOCK_UNIVERSE[c][1] for c in codes}).loc[ret.columns]
    ret.attrs["synthetic"] = n_synth > 0
    if n_synth:
        print(f"⚠️ 个股池 {len(codes)} 只中有 {n_synth} 只使用合成数据（彼此独立，没有行业结构），相关结果不代表真实市场")
    return ret, industry


# ==================== 组合计算 ====================


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """全局最小方差（允许做空）：w = Σ⁻¹1 / 1'Σ⁻¹1"""
    x = np.linalg.solve(Sigma, np.ones(len(Sigma)))
    return x / x.sum()


def tangency_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """切点组合（允许做空）：w ∝ Σ⁻¹(μ - rf)"""
    x = np.linalg.solve(Sigma, np.asarray(mu) - rf)
    return x / x.sum()


def inverse_vol_weights(Sigma: np.ndarray) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(Sigma))
    return iv / iv.sum()


def risk_contribution(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """风险贡献 RC_i = w_i (Σw)_i / σ_p，满足 Σ RC_i = σ_p"""
    w = np.asarray(w, dtype=float)
    sigma_p = np.sqrt(w @ Sigma @ w)
    return w * (Sigma @ w) / sigma_p


def erc_weights(Sigma: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    """风险预算 / 等风险贡献（Spinu 凸化，cvxpy 求解）"""
    import cvxpy as cp

    n = len(Sigma)
    b = np.ones(n) / n if budget is None else np.asarray(budget) / np.sum(budget)
    scale = 1.0 / np.mean(np.diag(Sigma))  # 数值缩放，不改变最优权重
    y = cp.Variable(n, pos=True)
    obj = 0.5 * cp.quad_form(y, cp.psd_wrap(Sigma * scale)) - b @ cp.log(y)
    cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
    w = np.asarray(y.value).ravel()
    return w / w.sum()


def max_drawdown(ret: pd.Series) -> float:
    """最大回撤；峰值从期初净值 1 算起，第一天就下跌也计入"""
    nav = (1 + ret).cumprod().to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    return float((nav / peak - 1).min())


def perf_stats(ret: pd.Series, rf: float = 0.0) -> dict:
    ret = ret.dropna()
    years = len(ret) / TRADING_DAYS
    total = float((1 + ret).prod())
    ann_ret = total ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    return {
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "Sharpe": (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan,
        "最大回撤": max_drawdown(ret),
    }


def perf_table(rets: dict[str, pd.Series], extra: dict[str, dict] | None = None) -> pd.DataFrame:
    """多个收益序列的绩效对比表（已格式化为字符串）"""
    rows = {}
    for name, r in rets.items():
        s = perf_stats(r)
        row = {
            "年化收益": f"{s['年化收益']:+.2%}",
            "年化波动": f"{s['年化波动']:.2%}",
            "Sharpe": f"{s['Sharpe']:.2f}",
            "最大回撤": f"{s['最大回撤']:.2%}",
        }
        if extra and name in extra:
            row.update(extra[name])
        rows[name] = row
    return pd.DataFrame(rows).T


def walk_forward(
    returns: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], np.ndarray],
    lookback: int = 252,
    freq: str = "ME",
) -> tuple[pd.Series, pd.DataFrame, float]:
    """
    样本外滚动回测：每个再平衡日只用"过去 lookback 天"估计权重，持有到下个再平衡日

    持有期内权重随价格漂移；换手按漂移后权重计算（单边 = 0.5 × Σ|Δw|）。

    Returns:
        (组合日收益, 各再平衡日目标权重, 年化单边换手)
    """
    dates = returns.index
    period_end = returns.groupby(dates.to_period(freq[0])).tail(1).index
    rebal_dates = [d for d in period_end if dates.get_loc(d) >= lookback - 1][:-1]

    port, weights, turnovers = [], {}, []
    w_drift = None
    for k, d in enumerate(rebal_dates):
        i = dates.get_loc(d)
        w = np.asarray(weight_fn(returns.iloc[i - lookback + 1 : i + 1]), dtype=float)
        weights[d] = w
        if w_drift is not None:
            turnovers.append(0.5 * np.abs(w - w_drift).sum())
        nxt = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else dates[-1]
        hold = returns.iloc[i + 1 : dates.get_loc(nxt) + 1]
        value = (1 + hold).cumprod().to_numpy() * w  # 各资产市值（期初总值 = 1）
        total = value.sum(axis=1)
        prev_total = np.r_[1.0, total[:-1]]
        port.append(pd.Series(total / prev_total - 1, index=hold.index))
        w_drift = value[-1] / total[-1]
    port_ret = pd.concat(port)
    years = len(port_ret) / TRADING_DAYS
    ann_turnover = float(np.sum(turnovers) / years) if years > 0 else np.nan
    return port_ret, pd.DataFrame(weights, index=returns.columns).T, ann_turnover


def annualize(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """日收益 → 年化均值向量与年化协方差矩阵"""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def fmt_weights(w, index, digits: int = 1) -> pd.Series:
    return pd.Series(np.asarray(w) * 100, index=index).round(digits)
