"""
行情数据获取工具
================

统一的数据入口，供各阶段课程调用：

    from utils.data import get_stock_daily
    df = get_stock_daily("600519", "20230101", "20231231", adjust="qfq")

特性：
    1. 本地缓存：首次从 AKShare 下载后存为 Parquet，之后直接读缓存
    2. 多数据源：东方财富接口失败时尝试新浪接口
    3. 离线回退：网络失败或设置 QTC_OFFLINE=1 时，生成固定随机种子的合成行情
       并打印「⚠️ 使用合成数据」，保证课程在任何环境下都能跑通

返回的 DataFrame 统一格式：
    index:   DatetimeIndex（名为 date）
    columns: open, high, low, close, volume, amount（成交量单位：股）
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from utils import DATA_DIR

CACHE_DIR = DATA_DIR / "cache"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]

# 东方财富接口的中文列名 → 统一英文列名
_EM_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def is_offline() -> bool:
    """是否强制离线模式"""
    return os.getenv("QTC_OFFLINE", "0") == "1"


# ==================== 合成数据 ====================


def synthetic_ohlcv(
    symbol: str,
    start: str,
    end: str,
    start_price: float = 20.0,
    annual_return: float = 0.08,
    annual_vol: float = 0.30,
) -> pd.DataFrame:
    """
    生成合成日线行情（几何布朗运动），同一 symbol 每次结果相同。

    用工作日近似交易日（不含节假日），仅用于在无网络时跑通课程代码，
    不能用来得出任何关于真实市场的结论。
    """
    seed = zlib.crc32(symbol.encode())
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), name="date")
    n = len(dates)

    dt = 1 / 252
    log_ret = rng.normal((annual_return - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), n)
    close = start_price * np.exp(np.cumsum(log_ret))

    # 开盘价 = 前收盘 × 隔夜跳空；最高/最低包住开盘与收盘
    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, annual_vol * np.sqrt(dt) * 0.3, n))
    spread = np.abs(rng.normal(0, annual_vol * np.sqrt(dt) * 0.5, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)

    # 成交量按 100 股取整，波动大的日子成交量更高
    base_volume = 5e6 * (1 + 20 * np.abs(log_ret))
    volume = (base_volume * rng.lognormal(0, 0.3, n) // 100 * 100).astype("int64")
    amount = volume * (open_ + close) / 2

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "amount": amount},
        index=dates,
    )
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].round(2)
    return df


# ==================== 缓存与回退 ====================


def _cache_path(kind: str, symbol: str, start: str, end: str, adjust: str = "") -> Path:
    tag = adjust or "none"
    return CACHE_DIR / f"{kind}_{symbol}_{tag}_{start}_{end}.parquet"


def _load_with_fallback(
    cache_path: Path,
    fetchers: list[Callable[[], pd.DataFrame]],
    synth: Callable[[], pd.DataFrame],
    label: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """读缓存 → 依次尝试各数据源 → 全部失败则使用合成数据"""
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if not is_offline():
        errors = []
        for fetch in fetchers:
            try:
                df = fetch()
                if df is not None and not df.empty:
                    if use_cache:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(cache_path)
                    return df
                errors.append(f"{fetch.__name__}: 返回空数据")
            except Exception as e:  # 数据源异常种类繁多，统一回退
                errors.append(f"{fetch.__name__}: {type(e).__name__}: {e}")
        print(f"⚠️ {label} 下载失败，使用合成数据。原因：")
        for err in errors:
            print(f"    - {err[:150]}")
    else:
        print(f"⚠️ 离线模式（QTC_OFFLINE=1）：{label} 使用合成数据")

    return synth()


def _normalize_em(df: pd.DataFrame) -> pd.DataFrame:
    """东方财富接口返回值 → 统一格式（成交量单位：手 → 股）"""
    df = df.rename(columns=_EM_COLUMN_MAP)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[OHLCV_COLUMNS].astype(float)
    df["volume"] = df["volume"] * 100
    return df


def _normalize_sina(df: pd.DataFrame) -> pd.DataFrame:
    """新浪接口返回值 → 统一格式（成交量单位已是股）"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[OHLCV_COLUMNS].astype(float)


def _exchange_prefix(symbol: str) -> str:
    """根据代码判断交易所前缀：6/9 开头上交所，4/8 开头北交所，其余深交所"""
    if symbol.startswith(("6", "9")):
        return "sh"
    if symbol.startswith(("4", "8")):
        return "bj"
    return "sz"


# ==================== 对外接口 ====================


def get_stock_daily(
    symbol: str = "600519",
    start: str = "20230101",
    end: str = "20231231",
    adjust: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取 A 股个股日线

    Args:
        symbol: 6 位股票代码，如 "600519"（贵州茅台）
        start / end: 日期，格式 YYYYMMDD
        adjust: "" 不复权，"qfq" 前复权，"hfq" 后复权
                ⚠️ 东方财富的 qfq/hfq 是"加减法"复权，除权日收益不准确；
                   计算收益请用 get_stock_daily_adj（等比复权），见 06-adjusted-prices.py
        use_cache: 是否读写本地 Parquet 缓存
    """

    def eastmoney():
        import akshare as ak

        raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust)
        return _normalize_em(raw)

    def sina():
        import akshare as ak

        raw = ak.stock_zh_a_daily(
            symbol=_exchange_prefix(symbol) + symbol, start_date=start, end_date=end, adjust=adjust
        )
        return _normalize_sina(raw)

    return _load_with_fallback(
        _cache_path("stock", symbol, start, end, adjust),
        [eastmoney, sina],
        lambda: synthetic_ohlcv(symbol, start, end, start_price=100.0),
        label=f"股票 {symbol}",
        use_cache=use_cache,
    )


def get_index_daily(
    symbol: str = "000300",
    start: str = "20230101",
    end: str = "20231231",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取指数日线

    常用指数：000300 沪深300、000905 中证500、000852 中证1000、000001 上证指数
    """

    def eastmoney():
        import akshare as ak

        raw = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)
        return _normalize_em(raw)

    def sina():
        import akshare as ak

        prefix = "sz" if symbol.startswith("399") else "sh"
        raw = ak.stock_zh_index_daily(symbol=prefix + symbol)
        raw["amount"] = np.nan  # 新浪指数接口不含成交额
        df = _normalize_sina(raw)
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    return _load_with_fallback(
        _cache_path("index", symbol, start, end),
        [eastmoney, sina],
        lambda: synthetic_ohlcv(symbol, start, end, start_price=4000.0, annual_vol=0.20),
        label=f"指数 {symbol}",
        use_cache=use_cache,
    )


def get_etf_daily(
    symbol: str = "510300",
    start: str = "20230101",
    end: str = "20231231",
    adjust: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取 ETF 日线

    常用 ETF：510300 沪深300ETF、510500 中证500ETF、159915 创业板ETF、518880 黄金ETF
    """

    def eastmoney():
        import akshare as ak

        raw = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust)
        return _normalize_em(raw)

    def sina():
        import akshare as ak

        raw = ak.fund_etf_hist_sina(symbol=_exchange_prefix(symbol) + symbol)
        raw["amount"] = np.nan  # 新浪 ETF 接口不含成交额
        df = _normalize_sina(raw)
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    return _load_with_fallback(
        _cache_path("etf", symbol, start, end, adjust),
        [eastmoney, sina],
        lambda: synthetic_ohlcv(symbol, start, end, start_price=4.0, annual_vol=0.20),
        label=f"ETF {symbol}",
        use_cache=use_cache,
    )


def get_adj_factor(symbol: str = "600519", use_cache: bool = True) -> pd.Series:
    """
    获取后复权因子（新浪，等比复权）

    返回按除权除息日索引的阶梯序列：某日的因子 = 上市以来所有除权事件的累积比例。
    后复权价 = 不复权价 × 因子。离线时返回恒为 1 的序列（即不复权）。
    """

    def sina():
        import akshare as ak

        raw = ak.stock_zh_a_daily(symbol=_exchange_prefix(symbol) + symbol, adjust="hfq-factor")
        s = raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["hfq_factor"].astype(float)
        return s.sort_index().to_frame("adj_factor")

    df = _load_with_fallback(
        CACHE_DIR / f"factor_{symbol}.parquet",
        [sina],
        lambda: pd.DataFrame({"adj_factor": [1.0]}, index=pd.DatetimeIndex(["1900-01-01"], name="date")),
        label=f"复权因子 {symbol}",
        use_cache=use_cache,
    )
    return df["adj_factor"]


def get_stock_daily_adj(
    symbol: str = "600519",
    start: str = "20230101",
    end: str = "20231231",
) -> pd.DataFrame:
    """
    不复权日线 + 等比后复权因子（推荐用于回测）

    额外列：
        adj_factor: 当日后复权因子
        adj_close:  后复权收盘价 = close × adj_factor，用于计算收益
    交易撮合（涨跌停、整数手、成交金额）使用不复权的 open/high/low/close。
    """
    df = get_stock_daily(symbol, start, end, adjust="")
    factor = get_adj_factor(symbol)
    df["adj_factor"] = factor.reindex(factor.index.union(df.index)).ffill().reindex(df.index).fillna(1.0)
    df["adj_close"] = df["close"] * df["adj_factor"]
    return df


def get_close_panel(
    symbols: list[str],
    start: str = "20230101",
    end: str = "20231231",
) -> pd.DataFrame:
    """获取多只股票的等比后复权收盘价宽表：行 = 日期，列 = 股票代码"""
    closes = {s: get_stock_daily_adj(s, start, end)["adj_close"] for s in symbols}
    return pd.DataFrame(closes).sort_index()
