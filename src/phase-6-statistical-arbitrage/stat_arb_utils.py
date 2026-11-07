"""
Phase 6 辅助模块：统计套利共用工具
==================================

本模块不是课程文件，而是 01–08 课共用的工具：

1. 股票配对数据
   - load_bank_panel()          9 只银行股等比后复权收盘价 + 沪深 300 指数
   - 离线（QTC_OFFLINE=1）或下载失败时，utils 返回的合成行情之间没有协整关系，
     因此这里改用 synthetic_bank_panel() 构造一组"部分协整"的模拟银行股：
     前 4 只（国有大行）共享同一个随机游走因子 + 平稳的个股偏离 → 两两协整；
     其余 5 只的个股偏离是随机游走 → 不协整

2. 配对交易工具
   - half_life / ols_hedge / kalman_hedge / pair_positions / pair_backtest / perf_stats

3. 期货与加密数据（AKShare 新浪接口 / ccxt）
   - get_future_contract(code)   单个期货合约日线（如 RB2501、IF2412）
   - get_if_main()               沪深 300 股指期货主力连续 IF0
   - get_funding_history(...)    永续合约资金费率历史
   - get_crypto_daily(...)       现货 / 永续日线收盘价
   所有下载结果缓存到 data/cache/stat_arb_*.parquet，第二次运行不再请求网络
   （想刷新数据，删除对应缓存文件即可）。离线或下载失败时回退到合成数据。

⚠️ 交易成本、融券费率、合约规格均为示例值，以交易所、券商和期货公司最新规则为准。
⚠️ 合成数据只用于跑通代码，不能得出任何关于真实市场的结论。
"""

from __future__ import annotations

import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import CACHE_DIR, get_index_daily, get_stock_daily_adj, is_offline, synthetic_ohlcv  # noqa: E402

# ==================== 常量 ====================

BANKS = {
    "601398": "工商银行",
    "601939": "建设银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "600036": "招商银行",
    "601166": "兴业银行",
    "600000": "浦发银行",
    "600016": "民生银行",
    "601328": "交通银行",
}
STOCK_START, STOCK_END = "20180101", "20241231"
FORM_END = "2021-12-31"  # 形成期（样本内）结束；之后为交易期（样本外）
TEST_START = "2022-01-01"
PAIR = ("601398", "601939")  # 主讲配对：工商银行 vs 建设银行

# A 股交易成本示例（以交易所和券商最新规则为准）
COMMISSION = 0.00025  # 佣金，双边
SLIPPAGE = 0.0005  # 滑点 / 冲击成本，双边
STAMP = 0.0005  # 印花税，仅卖出（含融券卖出）收取
BORROW_RATE = 0.08  # 融券年化费率示例，实际费率与券源由券商决定

INDEX_START, INDEX_END = "20100101", "20251231"  # 指数统一用这个区间（与已有缓存一致），再按需切片

_warned: set[str] = set()


def warn_synthetic(label: str, reason: str) -> None:
    """同一类数据只提示一次"""
    if label not in _warned:
        print(f"⚠️ 使用合成数据：{label}（{reason}）")
        _warned.add(label)


def _cache(name: str) -> Path:
    return CACHE_DIR / f"stat_arb_{name}.parquet"


# ==================== 股票：真实数据 + 协整合成数据 ====================


def synthetic_bank_panel(symbols: list[str], start: str, end: str, seed: int = 6) -> pd.DataFrame:
    """
    合成"部分协整"的银行股价格 + 市场指数（最后一列 "market"）

    log P_i = a_i + b_i · F_t + u_i,t
        F_t：共同因子（随机游走，代表行业 / 市场）
        u_i,t：前 4 只为 OU 过程（平稳，半衰期约 15–30 天）→ 两两协整，协整系数 b_A / b_B
               其余为随机游走 → 与任何股票都不协整
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), name="date")
    n = len(dates)
    factor = np.cumsum(rng.normal(0.0001, 0.013, n))
    out = {}
    for i, s in enumerate(symbols):
        b = rng.uniform(0.85, 1.2)
        level = np.log(rng.uniform(4, 40))
        if i < 4:
            hl = rng.uniform(15, 30)
            phi = 0.5 ** (1 / hl)
            sd = 0.03
            eps = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
            u = np.zeros(n)
            for t in range(1, n):
                u[t] = phi * u[t - 1] + eps[t]
        else:
            u = np.cumsum(rng.normal(0, 0.009, n))
        out[s] = np.exp(level + b * factor + u)
    out["market"] = 4000 * np.exp(factor)
    return pd.DataFrame(out, index=dates)


def _looks_synthetic(symbol: str, close: pd.Series, start: str, end: str) -> bool:
    """utils 在下载失败时会悄悄返回合成行情：与同参数的 synthetic_ohlcv 对比即可识别"""
    synth = synthetic_ohlcv(symbol, start, end, start_price=100.0)["close"]
    if len(synth) != len(close):
        return False
    return bool(np.allclose(synth.to_numpy(), close.to_numpy()))


def load_bank_panel(symbols: list[str] | None = None, start: str = STOCK_START, end: str = STOCK_END):
    """
    返回 (prices, market, is_synth)
        prices: 等比后复权收盘价宽表（行 = 日期，列 = 股票代码）
        market: 沪深 300 指数收盘价（合成模式下为合成市场因子）
        is_synth: 是否为合成数据
    """
    symbols = symbols or list(BANKS)
    if is_offline():
        warn_synthetic("银行股价格", "离线模式 QTC_OFFLINE=1，改用模拟的协整价格组")
    else:
        closes, bad = {}, []
        for s in symbols:
            df = get_stock_daily_adj(s, start, end)
            if _looks_synthetic(s, df["close"], start, end):
                bad.append(s)
            closes[s] = df["adj_close"]
        if not bad:
            prices = pd.DataFrame(closes).sort_index().dropna()
            idx = get_index_daily("000300", INDEX_START, INDEX_END)["close"]
            return prices, idx.reindex(prices.index).ffill(), False
        warn_synthetic("银行股价格", f"{bad} 下载失败；utils 的合成行情之间没有协整关系，改用模拟的协整价格组")
    panel = synthetic_bank_panel(symbols, start, end)
    return panel[symbols], panel["market"], True


def synthetic_coint_pair(
    n: int = 1500,
    beta: float = 0.9,
    half_life: float = 15.0,
    spread_sd: float = 0.03,
    vol_b: float = 0.015,
    break_at: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    一对协整的对数价格：log B 为随机游走，log A = 0.5 + beta · log B + s_t，s_t 为 OU 过程
    break_at：从第 break_at 天起 s_t 变为随机游走（协整断裂）
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, name="date")
    lb = np.log(20) + np.cumsum(rng.normal(0, vol_b, n))
    phi = 0.5 ** (1 / half_life)
    eps = rng.normal(0, spread_sd * np.sqrt(1 - phi**2), n)
    s = np.zeros(n)
    for t in range(1, n):
        p = 1.0 if (break_at is not None and t >= break_at) else phi
        s[t] = p * s[t - 1] + eps[t] * (2.0 if p == 1.0 else 1.0)
    la = 0.5 + beta * lb + s
    return pd.DataFrame({"A": la, "B": lb}, index=dates)


# ==================== 配对交易工具 ====================


def ols_hedge(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS：y = alpha + beta · x，返回 (alpha, beta)"""
    X = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    return float(coef[0]), float(coef[1])


def half_life(spread: pd.Series) -> float:
    """OU 过程半衰期：回归 Δs_t = a + b · s_{t-1}，h = -ln2 / ln(1+b)；b ≥ 0 时返回 inf"""
    s = pd.Series(spread).dropna()
    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    _, b = ols_hedge(delta, lag)
    if b >= 0 or b <= -1:
        return float("inf")
    return float(-np.log(2) / np.log(1 + b))


def kalman_hedge(y, x, delta: float = 1e-4, ve: float = 1e-3, init=(0.0, 0.0), p0: float = 1.0):
    """
    卡尔曼滤波动态估计 y_t = beta_t · x_t + alpha_t + ε_t（状态随机游走）

    delta: 状态噪声相对大小，Vw = delta / (1 - delta) · I
    ve:    观测噪声方差
    返回 states[t] = [beta_t, alpha_t]（t 日收盘后可得），预测误差 e[t]，预测误差方差 Q[t]
    e[t] 只用到 t-1 及之前的状态 → 无前视
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    Vw = delta / (1 - delta) * np.eye(2)
    b = np.array(init, dtype=float)
    P = p0 * np.eye(2)  # 初始不确定性较大（扩散先验），让滤波器快速"学"到合理的 beta
    states, e, Q = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
    for t in range(n):
        H = np.array([x[t], 1.0])
        R = P + Vw  # 预测：不确定性扩散
        e[t] = y[t] - H @ b  # 用旧状态预测今天的 y
        Q[t] = H @ R @ H + ve
        K = R @ H / Q[t]  # 卡尔曼增益
        b = b + K * e[t]  # 更新
        P = R - np.outer(K, H) @ R
        states[t] = b
    return states, e, Q


def pair_positions(z: pd.Series, entry: float = 2.0, exit: float = 0.5, stop: float = 4.0,
                   max_hold: int | None = None) -> pd.Series:
    """
    Z-score 状态机，返回每日收盘后的目标仓位：+1 做多价差（多 A 空 B），-1 做空价差，0 空仓

    开仓：z < -entry → +1；z > entry → -1
    平仓：多头 z > -exit / 空头 z < exit（回到均值附近）
    止损：|z| > stop 或持仓天数 ≥ max_hold；止损后需等 |z| < exit 才允许再开仓（冷却）
    """
    vals = z.to_numpy(dtype=float)
    pos = np.zeros(len(vals))
    cur, held, cooldown = 0, 0, False
    for i, zi in enumerate(vals):
        if np.isnan(zi):
            cur, held = 0, 0
        elif cur == 0:
            if cooldown and abs(zi) < exit:
                cooldown = False
            if not cooldown and abs(zi) < stop:
                cur = -1 if zi > entry else (1 if zi < -entry else 0)
            held = 0
        else:
            held += 1
            if abs(zi) > stop or (max_hold is not None and held >= max_hold):
                cur, cooldown = 0, True
            elif (cur == 1 and zi > -exit) or (cur == -1 and zi < exit):
                cur = 0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


def pair_backtest(pos: pd.Series, beta, log_a: pd.Series, log_b: pd.Series,
                  cost: float = COMMISSION + SLIPPAGE, stamp: float = STAMP,
                  borrow: float = BORROW_RATE) -> pd.DataFrame:
    """
    配对回测（对数价格口径，资金归一化：两条腿名义金额之和 = 1）

    t 日收盘后目标权重：w_A = pos / (1+|β|)，w_B = -pos · β / (1+|β|)；t+1 日起持有
    成本：换手 × (佣金 + 滑点) + 卖出金额 × 印花税 + 空头名义金额 × 融券年化费率 / 252
    简化：每日按目标权重再平衡（忽略日内权重漂移）
    """
    beta = beta if isinstance(beta, pd.Series) else pd.Series(float(beta), index=pos.index)
    gross_lev = 1 + beta.abs()
    wa = (pos / gross_lev).shift(1).fillna(0.0)
    wb = (-pos * beta / gross_lev).shift(1).fillna(0.0)
    ra = np.expm1(log_a.diff()).fillna(0.0)
    rb = np.expm1(log_b.diff()).fillna(0.0)
    gross = wa * ra + wb * rb
    dwa, dwb = wa.diff().fillna(wa), wb.diff().fillna(wb)
    turnover = dwa.abs() + dwb.abs()
    sells = (-dwa).clip(lower=0) + (-dwb).clip(lower=0)
    short_notional = (-wa).clip(lower=0) + (-wb).clip(lower=0)
    trade_cost = turnover * cost + sells * stamp
    borrow_cost = short_notional * borrow / 252
    return pd.DataFrame({
        "pos": pos.shift(1).fillna(0.0), "gross": gross, "trade_cost": trade_cost,
        "borrow_cost": borrow_cost, "net_no_borrow": gross - trade_cost,
        "net": gross - trade_cost - borrow_cost, "turnover": turnover,
    })


def perf_stats(ret: pd.Series, pos: pd.Series | None = None) -> dict:
    """常用绩效指标（无风险利率按 0）"""
    r = ret.fillna(0.0)
    nav = (1 + r).cumprod()
    years = max(len(r) / 252, 1e-9)
    sd = r.std()
    out = {
        "总收益": nav.iloc[-1] - 1,
        "年化收益": nav.iloc[-1] ** (1 / years) - 1 if nav.iloc[-1] > 0 else -1.0,
        "年化波动": sd * np.sqrt(252),
        "Sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
        "最大回撤": (nav / nav.cummax() - 1).min(),
    }
    if pos is not None:
        p = pos.fillna(0)
        out["开仓次数"] = int(((p != 0) & (p != p.shift(1).fillna(0))).sum())
        out["持仓占比"] = float((p != 0).mean())
    return out


def stats_table(results: dict[str, dict]) -> pd.DataFrame:
    """把多个 perf_stats 结果格式化成表格（行 = 指标，列 = 策略）"""
    df = pd.DataFrame(results)
    fmt = {"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "年化波动": "{:.1%}", "Sharpe": "{:.2f}",
           "最大回撤": "{:.1%}", "开仓次数": "{:.0f}", "持仓占比": "{:.0%}"}
    rows = {k: [fmt.get(k, "{:.3g}").format(v) if pd.notna(v) else "-" for v in df.loc[k]] for k in df.index}
    return pd.DataFrame.from_dict(rows, orient="index", columns=df.columns, dtype=object)


# ==================== 期货：单合约 / 股指主力 ====================

_FUT_SYNTH = {"RB": dict(price=3800, vol=0.22, carry=-0.05), "HC": dict(price=3900, vol=0.22, carry=-0.04)}
_synth_cache: dict[str, object] = {}


def parse_contract(code: str) -> tuple[str, int, int]:
    """'RB2501' → ('RB', 2025, 1)"""
    m = re.fullmatch(r"([A-Za-z]+)(\d{2})(\d{2})", code)
    if not m:
        raise ValueError(f"无法解析合约代码 {code}")
    return m.group(1).upper(), 2000 + int(m.group(2)), int(m.group(3))


def third_friday(year: int, month: int) -> pd.Timestamp:
    """股指期货最后交易日：合约月份第三个周五（遇节假日顺延，这里不处理）"""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 14)


def _hs300() -> pd.Series:
    if "hs300" not in _synth_cache:
        _synth_cache["hs300"] = get_index_daily("000300", INDEX_START, INDEX_END)["close"]
    return _synth_cache["hs300"]


def _dividend_frac(dates: pd.DatetimeIndex, expiry: pd.Timestamp, annual_yield: float = 0.025) -> np.ndarray:
    """合成用：t 到到期日之间预计除息的分红（占指数比例）。A 股分红集中在 5–8 月"""
    weights = {1: 0.01, 2: 0.01, 3: 0.02, 4: 0.03, 5: 0.10, 6: 0.30, 7: 0.35, 8: 0.12, 9: 0.03, 10: 0.01,
               11: 0.01, 12: 0.01}
    days = pd.date_range(dates.min(), expiry, freq="D")
    dens = np.array([weights[d.month] / d.days_in_month for d in days]) * annual_yield
    cum = np.concatenate([[0.0], np.cumsum(dens)])
    pos = days.searchsorted(dates, side="right")  # 当天之后（不含当天）到到期日的累计分红
    return cum[-1] - cum[pos]


def _synth_if(dates: pd.DatetimeIndex, expiries: pd.DatetimeIndex) -> np.ndarray:
    """合成股指期货价格：F = S · exp(r·τ - 分红 - 对冲压力·τ + 噪声)，到期时收敛到 S"""
    s = _hs300().reindex(dates).ffill().to_numpy()
    rng = np.random.default_rng(300)
    tau = np.maximum((expiries - dates).days.to_numpy(), 0) / 365
    pressure = 0.03 + 0.03 * np.sin(np.arange(len(dates)) / 180)  # 对冲需求带来的额外贴水（年化）
    div = np.array([_dividend_frac(pd.DatetimeIndex([d]), e)[0] for d, e in zip(dates, expiries)])
    noise = rng.normal(0, 0.0015, len(dates)) * np.sqrt(np.minimum(tau * 12, 1))
    return s * np.exp(0.02 * tau - div - pressure * tau + noise)


def _synth_commodity(code: str) -> pd.DataFrame:
    prod, year, month = parse_contract(code)
    spec = _FUT_SYNTH.get(prod, dict(price=3000, vol=0.2, carry=-0.03))
    key = f"spot_{prod}"
    if key not in _synth_cache:
        spot = synthetic_ohlcv(f"STATARB_{prod}", "20170101", "20261231", start_price=spec["price"],
                               annual_return=0.0, annual_vol=spec["vol"])["close"]
        rng = np.random.default_rng(zlib.crc32(prod.encode()))
        k = np.zeros(len(spot))  # 年化持有成本率（负 = 近月高于远月），OU 过程
        for t in range(1, len(k)):
            k[t] = k[t - 1] + 0.02 * (spec["carry"] - k[t - 1]) + rng.normal(0, 0.012)
        _synth_cache[key] = (spot, pd.Series(k, index=spot.index))
    spot, k = _synth_cache[key]
    expiry = pd.Timestamp(year=year, month=month, day=15)
    d = spot.index[(spot.index > expiry - pd.Timedelta(days=365)) & (spot.index <= expiry)]
    tau = (expiry - d).days.to_numpy() / 365
    rng = np.random.default_rng(zlib.crc32(code.encode()))
    noise = np.cumsum(rng.normal(0, 0.0008, len(d))) * 0.3
    close = (spot.loc[d].to_numpy() * np.exp(k.loc[d].to_numpy() * tau + noise)).round()
    days_left = (expiry - d).days.to_numpy()
    hold = (1e5 * np.exp(-(((days_left - 150) / 70.0) ** 2)) + 100).round()
    return pd.DataFrame({"close": close, "volume": (hold * 0.8).round(), "hold": hold}, index=d.rename("date"))


def _synth_contract(code: str) -> pd.DataFrame:
    prod, year, month = parse_contract(code)
    if prod != "IF":
        return _synth_commodity(code)
    expiry = third_friday(year, month)
    s = _hs300()
    d = s.index[(s.index > expiry - pd.Timedelta(days=245)) & (s.index <= expiry)]
    close = _synth_if(d, pd.DatetimeIndex([expiry] * len(d)))
    return pd.DataFrame({"close": close.round(1), "volume": 1e4, "hold": 1e4}, index=d.rename("date"))


def get_future_contract(code: str) -> pd.DataFrame:
    """
    单个期货合约日线（新浪），返回列 close, volume, hold(持仓量)；index 为 date
    缓存：data/cache/stat_arb_fut_{code}.parquet
    """
    if is_offline():
        warn_synthetic(f"期货 {parse_contract(code)[0]} 单合约", "离线模式 QTC_OFFLINE=1")
        return _synth_contract(code)
    path = _cache(f"fut_{code}")
    if path.exists():
        return pd.read_parquet(path)
    try:
        import akshare as ak

        raw = ak.futures_zh_daily_sina(symbol=code)
        if raw is None or raw.empty:
            raise ValueError("返回空数据")
        df = raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")[["close", "volume", "hold"]]
        df = df.astype(float)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df
    except Exception as e:  # 网络异常种类繁多，统一回退
        warn_synthetic(f"期货 {code}", f"下载失败 {type(e).__name__}: {str(e)[:60]}")
        return _synth_contract(code)


def get_if_main(start: str = "20190101", end: str = "20251231") -> pd.Series:
    """沪深 300 股指期货主力连续 IF0 收盘价（新浪，未调整换月）"""
    def synth():
        s = _hs300().loc[pd.Timestamp(start):pd.Timestamp(end)]
        exp = []
        for d in s.index:  # 主力近似为当月合约，到期后切到下月
            e = third_friday(d.year, d.month)
            if d >= e:
                nxt = d + pd.offsets.MonthBegin(1)
                e = third_friday(nxt.year, nxt.month)
            exp.append(e)
        return pd.Series(_synth_if(s.index, pd.DatetimeIndex(exp)), index=s.index, name="close")

    if is_offline():
        warn_synthetic("股指期货 IF0", "离线模式 QTC_OFFLINE=1")
        return synth()
    path = _cache(f"if0_{start}_{end}")
    if path.exists():
        return pd.read_parquet(path)["close"]
    try:
        import akshare as ak

        raw = ak.futures_main_sina(symbol="IF0", start_date=start, end_date=end)
        df = raw.rename(columns={"日期": "date", "收盘价": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["close"]].astype(float)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df["close"]
    except Exception as e:
        warn_synthetic("股指期货 IF0", f"下载失败 {type(e).__name__}: {str(e)[:60]}")
        return synth()


# ==================== 加密：资金费率 / 日线（仅公开数据，用于研究） ====================


def _ccxt_exchange(exchange_id: str):
    """同一交易所只创建一个实例并加载一次市场信息（加载失败重试一次）"""
    key = f"ccxt_{exchange_id}"
    if key not in _synth_cache:
        import ccxt

        config = {"timeout": 30000, "enableRateLimit": True}
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if proxy:
            config["proxies"] = {"http": proxy, "https": proxy}
        ex = getattr(ccxt, exchange_id)(config)
        try:
            ex.load_markets()
        except Exception:
            ex.load_markets(reload=True)
        _synth_cache[key] = ex
    return _synth_cache[key]


def _synth_funding(symbol: str, since: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """合成资金费率（每 8 小时）：在"牛市 / 平静 / 熊市"三种状态间切换的 OU 过程"""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    idx = pd.date_range(since.ceil("8h"), end, freq="8h", name="datetime")
    regime_mean = {0: 0.00005, 1: 0.0003, 2: -0.00005}
    state, x, vals = 0, 0.0001, []
    for _ in idx:
        if rng.random() < 1 / 300:  # 平均约 100 天切换一次状态
            state = int(rng.integers(0, 3))
        x = x + 0.1 * (regime_mean[state] - x) + rng.normal(0, 0.00006)
        vals.append(x)
    return pd.Series(vals, index=idx, name="funding_rate")


def get_funding_history(exchange_id: str = "okx", symbol: str = "BTC/USDT:USDT",
                        since: str | None = None, page_limit: int = 100, max_pages: int = 12) -> pd.Series:
    """
    永续合约资金费率历史（每个结算周期一条，小数），index 为 UTC 时间
    since=None 时取最近约 90 天（OKX 公开接口只保留约 3 个月）
    缓存：data/cache/stat_arb_funding_{exchange}_{币种}_{since}.parquet
    """
    end = pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(since, tz="UTC") if since else end - pd.Timedelta(days=90)
    tag = f"{exchange_id}_{symbol.split('/')[0]}_{since or 'recent'}"
    if is_offline():
        warn_synthetic(f"资金费率 {exchange_id} {symbol}", "离线模式 QTC_OFFLINE=1")
        return _synth_funding(tag, start, end)
    path = _cache(f"funding_{tag}")
    if path.exists():
        return pd.read_parquet(path)["funding_rate"]
    try:
        ex = _ccxt_exchange(exchange_id)
        cursor, rows = int(start.timestamp() * 1000), []
        for _ in range(max_pages):
            page = ex.fetch_funding_rate_history(symbol, since=cursor, limit=page_limit)
            page = [p for p in page if p["timestamp"] >= cursor]
            if not page:
                break
            rows.extend(page)
            cursor = page[-1]["timestamp"] + 1
            if len(page) < page_limit // 2:
                break
        if not rows:
            raise ValueError("没有取到资金费率")
        s = pd.Series({pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in rows})
        s.index = s.index.round("min")  # 部分交易所时间戳带毫秒抖动
        s = s[~s.index.duplicated()].sort_index().rename("funding_rate")
        s.index.name = "datetime"
        path.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(path)
        return s
    except Exception as e:
        warn_synthetic(f"资金费率 {exchange_id} {symbol}", f"获取失败 {type(e).__name__}: {str(e)[:60]}")
        return _synth_funding(tag, start, end)


def get_crypto_daily(exchange_id: str = "okx", symbol: str = "BTC/USDT", limit: int = 300) -> pd.Series:
    """日线收盘价（UTC），丢弃最后一根未收盘 K 线；缓存 stat_arb_ohlcv_{exchange}_{symbol}.parquet"""
    safe = symbol.replace("/", "-").replace(":", "_")
    if is_offline():
        warn_synthetic(f"加密日线 {symbol}", "离线模式 QTC_OFFLINE=1")
        return _synth_crypto(symbol, limit)
    path = _cache(f"ohlcv_{exchange_id}_{safe}")
    if path.exists():
        return pd.read_parquet(path)["close"]
    try:
        ex = _ccxt_exchange(exchange_id)
        bars = ex.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("date")[["close"]].astype(float).iloc[:-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df["close"]
    except Exception as e:
        warn_synthetic(f"加密日线 {symbol}", f"获取失败 {type(e).__name__}: {str(e)[:60]}")
        return _synth_crypto(symbol, limit)


def _synth_crypto(symbol: str, limit: int) -> pd.Series:
    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    idx = pd.date_range(end=end, periods=limit, freq="D", name="date")
    rng = np.random.default_rng(2024)  # 现货与永续共用同一条价格路径
    spot = 60000 * np.exp(np.cumsum(rng.normal(0, 0.03, limit)))
    if ":" in symbol:  # 永续：现货 × (1 + 小幅溢价)
        prem = np.zeros(limit)
        r2 = np.random.default_rng(7)
        for t in range(1, limit):
            prem[t] = 0.7 * prem[t - 1] + r2.normal(0.00005, 0.0004)
        spot = spot * (1 + prem)
    return pd.Series(spot, index=idx, name="close")
