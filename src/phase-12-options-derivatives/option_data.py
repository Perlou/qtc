"""
Phase 12 辅助模块：ETF 期权数据（AKShare + 缓存 + 离线回退）
============================================================

本模块不是课程文件，供 05、06、08 课调用。

    from option_data import get_option_chain
    chain, meta = get_option_chain("510050")

数据来源（AKShare 1.20 实测可用的接口组合，函数名随版本可能变化）：
    option_sse_list_sina(symbol="50ETF")                  可交易的到期月份列表
    option_sse_expire_day_sina(trade_date, symbol)        某到期月份的到期日
    option_finance_board(symbol="华夏上证50ETF期权", ...)   上交所某到期月份的整条期权链（最新价、前结算价）
    option_sse_underlying_spot_price_sina("sh510050")     标的 ETF 实时行情
    option_risk_indicator_sse(date)                       上交所官方公布的希腊字母与隐含波动率
    option_sse_spot_price_sina(security_id)               单个合约的五档盘口（买卖价）

缓存（data/cache/，文件名均以 option_ 开头）：
    option_chain_{标的}_{下载日期}.parquet     期权链快照：同一天重复运行不再请求网络
    option_risk_sse_{交易日}.parquet           上交所风险指标
    option_quote_{合约编号}_{下载日期}.parquet 单合约盘口

离线回退：
    QTC_OFFLINE=1 或下载失败 → 使用最近一次缓存的快照 → 没有缓存则生成合成期权链，
    并打印「⚠️ 使用合成数据」。合成数据只用于跑通代码，不能得出任何关于真实市场的结论。

⚠️ 合约单位、行权价间距、到期日规则等合约要素以交易所最新规则为准。
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from option_pricing import bs_price, implied_vol, svi_w  # noqa: E402
from utils.data import CACHE_DIR, is_offline  # noqa: E402

UNDERLYINGS = {
    "510050": dict(name="上证50ETF", sina="50ETF", board="华夏上证50ETF期权", spot_symbol="sh510050", synth_spot=2.90),
    "510300": dict(name="沪深300ETF", sina="300ETF", board="华泰柏瑞沪深300ETF期权", spot_symbol="sh510300",
                   synth_spot=4.50),
}
CODE_PATTERN = re.compile(r"^(?P<und>\d{6})(?P<cp>[CP])(?P<yymm>\d{4})(?P<adj>[A-Z])(?P<strike>\d{5})$")
RISK_FREE = 0.015  # 示例无风险利率（连续复利）；实务可用 Shibor 或短期国债收益率
SYNTH_DATE = pd.Timestamp("2024-11-08")  # 合成数据的固定估值日，保证每次运行结果相同


# ==================== 合约要素（示例，以交易所最新规则为准） ====================


def strike_step(price: float) -> float:
    """上交所 ETF 期权行权价间距（示例规则，以交易所最新规则为准）"""
    for upper, step in [(3, 0.05), (5, 0.1), (10, 0.25), (20, 0.5), (50, 1.0), (100, 2.5)]:
        if price <= upper:
            return step
    return 5.0


def strike_ladder(spot: float, n_each_side: int = 7) -> np.ndarray:
    """以平值为中心、按 strike_step 规则向上下各挂 n 档行权价（与真实合约的挂牌方式类似）"""
    atm = round(spot / strike_step(spot)) * strike_step(spot)
    up, down = [atm], [atm]
    for _ in range(n_each_side):
        up.append(up[-1] + strike_step(up[-1] + 1e-9))
        down.append(down[-1] - strike_step(down[-1] - 1e-9))
    return np.round(np.array(sorted(set(down[1:] + up))), 3)


def fourth_wednesday(year: int, month: int) -> pd.Timestamp:
    """到期月份的第四个星期三（未考虑节假日顺延，以交易所公告为准）"""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (2 - first.weekday()) % 7  # weekday: 周一=0，周三=2
    return first + pd.Timedelta(days=offset + 21)


def parse_code(code: str) -> dict | None:
    """解析交易代码，如 510050C2612M02800 → 标的 510050、认购、2026-12、未调整、行权价 2.800"""
    m = CODE_PATTERN.match(code)
    if m is None:
        return None
    return {
        "underlying": m["und"],
        "kind": "call" if m["cp"] == "C" else "put",
        "yymm": m["yymm"],
        "adjusted": m["adj"] != "M",  # M 为标准合约；分红调整后的合约通常为 A、B…
        "strike_in_code": int(m["strike"]) / 1000,
    }


# ==================== 缓存工具 ====================


def _latest_cache(pattern: str) -> Path | None:
    files = sorted(CACHE_DIR.glob(pattern))
    return files[-1] if files else None


def _today() -> str:
    return date.today().strftime("%Y%m%d")


# ==================== 真实数据 ====================


def _fetch_chain(underlying: str) -> pd.DataFrame:
    import akshare as ak

    info = UNDERLYINGS[underlying]
    spot_raw = ak.option_sse_underlying_spot_price_sina(symbol=info["spot_symbol"])
    spot_map = dict(zip(spot_raw["字段"], spot_raw["值"]))
    spot = float(spot_map["最近成交价"])

    frames = []
    for month in ak.option_sse_list_sina(symbol=info["sina"], exchange="null"):
        expiry, _ = ak.option_sse_expire_day_sina(trade_date=month, symbol=info["sina"], exchange="null")
        board = ak.option_finance_board(symbol=info["board"], end_month=month[-4:])
        if board is None or board.empty:
            continue
        board = board.assign(expiry=pd.Timestamp(expiry))
        frames.append(board)
    raw = pd.concat(frames, ignore_index=True)

    parsed = raw["合约交易代码"].map(parse_code)
    raw = raw[parsed.notna()].reset_index(drop=True)
    meta = pd.DataFrame(list(parsed.dropna()))
    snapshot = pd.to_datetime(raw["日期"].astype(str).str[:14], format="%Y%m%d%H%M%S")
    df = pd.DataFrame({
        "code": raw["合约交易代码"].astype(str),
        "kind": meta["kind"],
        "adjusted": meta["adjusted"],
        "expiry": raw["expiry"],
        "strike": raw["行权价"].astype(float),
        "price": raw["当前价"].astype(float),
        "prev_settle": raw["前结价"].astype(float),
        "spot": spot,
        "snapshot": snapshot,
    })
    df["days"] = (df["expiry"] - df["snapshot"].dt.normalize()).dt.days
    df["source"] = "real"
    return df


# ==================== 合成数据 ====================


def synthetic_chain(underlying: str = "510050", seed: int = 7) -> pd.DataFrame:
    """
    按 SVI 偏斜曲面生成一条合成期权链：当月、下月、随后两个季月共 4 个到期日。
    价格四舍五入到最小报价单位 0.0001，并加入 ±1 跳的报价噪声。
    """
    rng = np.random.default_rng(seed)
    spot = UNDERLYINGS[underlying]["synth_spot"]
    val = SYNTH_DATE

    months = []
    y, m = val.year, val.month
    while len(months) < 2:  # 当月（若已过到期日则跳过）与下月
        exp = fourth_wednesday(y, m)
        if exp > val:
            months.append(exp)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    while len(months) < 4:  # 随后两个季月
        if m in (3, 6, 9, 12):
            months.append(fourth_wednesday(y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    strikes = strike_ladder(spot)
    rows = []
    for exp in months:
        days = (exp - val).days
        T = days / 365
        F = spot * np.exp((RISK_FREE - 0.02) * T)  # 假设 2% 股息率，远期略低于现货
        k = np.log(strikes / F)
        # 近月偏斜更陡、远月更平：参数随 T 平滑变化，总方差随 T 递增
        a, b, rho, mm, s = 0.02 * T, 0.06 * np.sqrt(T), -0.55, 0.01, 0.10
        iv = np.sqrt(svi_w(k, a, b, rho, mm, s) / T)
        for kind in ("call", "put"):
            px = bs_price(F * np.exp(-RISK_FREE * T), strikes, T, RISK_FREE, iv, kind)
            px = np.round(px + rng.integers(-1, 2, size=px.size) * 1e-4, 4)
            for K, p in zip(strikes, px):
                if p >= 1e-4:
                    cp = "C" if kind == "call" else "P"
                    rows.append({
                        "code": f"{underlying}{cp}{exp:%y%m}M{int(round(K * 1000)):05d}",
                        "kind": kind, "adjusted": False, "expiry": exp, "strike": float(K),
                        "price": float(p), "prev_settle": float(p), "spot": spot,
                        "snapshot": val + pd.Timedelta(hours=15), "days": days, "source": "synthetic",
                    })
    return pd.DataFrame(rows)


# ==================== 对外接口 ====================


def get_option_chain(underlying: str = "510050", verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    获取 ETF 期权链快照

    返回 (chain, meta)：
        chain 列：code, kind, adjusted, expiry, strike, price(最新价), prev_settle(前结算价),
                  spot(标的价), snapshot(行情时间), days(剩余自然日), source
        meta：underlying, name, spot, snapshot, source("real" / "cache" / "synthetic")
    """
    info = UNDERLYINGS[underlying]
    today_path = CACHE_DIR / f"option_chain_{underlying}_{_today()}.parquet"
    df, source = None, "real"

    if today_path.exists():
        df = pd.read_parquet(today_path)
        if verbose:
            print(f"📦 读取今日缓存：{today_path.name}")
    elif not is_offline():
        try:
            df = _fetch_chain(underlying)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(today_path)
            if verbose:
                print(f"🌐 已下载 {info['name']} 期权链 {len(df)} 个合约，缓存到 {today_path.name}")
        except Exception as e:  # 数据源异常种类繁多，统一回退
            if verbose:
                print(f"⚠️ 期权链下载失败：{type(e).__name__}: {str(e)[:120]}")

    if df is None:
        cached = _latest_cache(f"option_chain_{underlying}_*.parquet")
        if cached is not None:
            df, source = pd.read_parquet(cached), "cache"
            if verbose:
                tag = "离线模式（QTC_OFFLINE=1）" if is_offline() else "下载失败"
                print(f"📦 {tag}：使用最近一次缓存的真实快照 {cached.name}")
        else:
            df, source = synthetic_chain(underlying), "synthetic"
            if verbose:
                print(f"⚠️ 使用合成数据：{info['name']} 期权链为按 SVI 偏斜曲面生成的模拟报价")

    df = df.sort_values(["expiry", "kind", "strike"]).reset_index(drop=True)
    meta = {
        "underlying": underlying,
        "name": info["name"],
        "spot": float(df["spot"].iloc[0]),
        "snapshot": pd.Timestamp(df["snapshot"].iloc[0]),
        "source": "synthetic" if df["source"].iloc[0] == "synthetic" else source,
    }
    return df, meta


def get_sse_risk_indicator(trade_date: str, verbose: bool = True) -> pd.DataFrame | None:
    """上交所官方希腊字母与隐含波动率（trade_date: YYYYMMDD）；离线且无缓存时返回 None"""
    path = CACHE_DIR / f"option_risk_sse_{trade_date}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if is_offline():
        return None
    try:
        import akshare as ak

        df = ak.option_risk_indicator_sse(date=trade_date)
        if df is None or df.empty:
            return None
        df = df.astype({"SECURITY_ID": str, "CONTRACT_ID": str, "CONTRACT_SYMBOL": str, "TRADE_DATE": str})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df
    except Exception as e:
        if verbose:
            print(f"⚠️ 上交所风险指标下载失败：{type(e).__name__}: {str(e)[:120]}")
        return None


def get_contract_quote(security_id: str, verbose: bool = True) -> dict | None:
    """单个合约的盘口（买一、卖一、最新价等）；离线且无缓存时返回 None"""
    today_path = CACHE_DIR / f"option_quote_{security_id}_{_today()}.parquet"
    if today_path.exists():
        df = pd.read_parquet(today_path)
    elif not is_offline():
        try:
            import akshare as ak

            df = ak.option_sse_spot_price_sina(symbol=security_id).astype(str)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(today_path)
        except Exception as e:
            if verbose:
                print(f"⚠️ 合约 {security_id} 盘口下载失败：{type(e).__name__}: {str(e)[:120]}")
            return None
    else:
        cached = _latest_cache(f"option_quote_{security_id}_*.parquet")
        if cached is None:
            return None
        df = pd.read_parquet(cached)
    return dict(zip(df["字段"], df["值"]))


# ==================== 期权链 → 隐含波动率 ====================


def estimate_forward(sub: pd.DataFrame, r: float, T: float, n_pairs: int = 3) -> float:
    """
    用认购-认沽平价估计隐含远期价格：取 |C - P| 最小（最接近平值）的几个行权价，
    F = K + (C - P)·e^{rT}，取中位数。sub 为同一到期日的期权链。
    """
    wide = sub.pivot_table(index="strike", columns="kind", values="price")
    wide = wide.dropna(subset=["call", "put"]) if {"call", "put"} <= set(wide.columns) else wide.iloc[0:0]
    if wide.empty:
        return float("nan")
    near = wide.assign(diff=(wide["call"] - wide["put"]).abs()).nsmallest(n_pairs, "diff")
    F = near.index.to_numpy() + (near["call"] - near["put"]).to_numpy() * np.exp(r * T)
    return float(np.median(F))


def chain_implied_vols(chain: pd.DataFrame, r: float = RISK_FREE, min_days: int = 3,
                       min_price: float = 5e-4) -> pd.DataFrame:
    """
    整条期权链求 IV（按 06 课的清洗流程）：
        1. 过滤剩余期限 < min_days 天、价格 < min_price 的合约
        2. 每个到期日用平价关系估计隐含远期 F
        3. 以 S_eff = F·e^{-rT} 代入 BS 求 IV（等价于 Black-76，自动包含分红与融资成本预期）
        4. 标记虚值一侧：K < F 用认沽，K ≥ F 用认购（otm=True）

    返回列：expiry, days, T, kind, strike, price, F, k(=ln(K/F)), iv, otm
    """
    out = []
    for expiry, sub in chain.groupby("expiry"):
        days = int(sub["days"].iloc[0])
        if days < min_days:
            continue
        T = days / 365
        F = estimate_forward(sub, r, T)
        if not np.isfinite(F):
            continue
        sub = sub[sub["price"] >= min_price].copy()
        s_eff = F * np.exp(-r * T)
        sub["iv"] = [implied_vol(p, s_eff, k, T, r, kd) for p, k, kd in zip(sub["price"], sub["strike"], sub["kind"])]
        sub["expiry"], sub["days"], sub["T"], sub["F"] = expiry, days, T, F
        sub["k"] = np.log(sub["strike"] / F)
        sub["otm"] = np.where(sub["kind"] == "put", sub["strike"] < F, sub["strike"] >= F)
        out.append(sub[["expiry", "days", "T", "kind", "strike", "price", "F", "k", "iv", "otm"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
