"""
数据获取层

两个实现共享同一接口，存储层和 CLI 不关心数据来自哪里：
    - AkshareFetcher:   真实数据（东方财富为主、新浪为备），带重试与限速
    - SyntheticFetcher: 合成数据，用于测试和离线演示
"""

from __future__ import annotations

import functools
import time
import zlib
from typing import Callable, Protocol

import numpy as np
import pandas as pd

DAILY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]


class Fetcher(Protocol):
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """不复权日线，列为 DAILY_COLUMNS，成交量单位：股"""

    def fetch_adj_factor(self, symbol: str) -> pd.DataFrame:
        """后复权因子，列为 symbol, date, adj_factor（只在除权日有记录）"""

    def fetch_universe(self, name: str) -> list[str]:
        """股票池成分，如 hs300"""


def retry(max_retries: int = 3, base_delay: float = 1.0, sleep: Callable[[float], None] = time.sleep):
    """失败后指数退避重试：1s, 2s, 4s ...，最后一次仍失败则抛出异常"""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    sleep(base_delay * 2**attempt)

        return wrapper

    return decorator


def exchange_prefix(symbol: str) -> str:
    if symbol.startswith(("6", "9")) and not symbol.startswith("92"):
        return "sh"
    if symbol.startswith(("8", "4", "92")):
        return "bj"
    return "sz"


class AkshareFetcher:
    def __init__(self, request_interval: float = 0.5, max_retries: int = 3):
        import akshare as ak

        self.ak = ak
        self.request_interval = request_interval
        self.max_retries = max_retries
        self._last_request = 0.0

    def _throttle(self):
        wait = self.request_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _call(self, fn, *args, **kwargs):
        @retry(self.max_retries)
        def run():
            self._throttle()
            return fn(*args, **kwargs)

        return run()

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            raw = self._call(
                self.ak.stock_zh_a_hist, symbol=symbol, period="daily", start_date=start, end_date=end, adjust=""
            )
            df = raw.rename(
                columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                         "最低": "low", "成交量": "volume", "成交额": "amount"}
            )
            df["volume"] = df["volume"] * 100  # 东方财富：手 → 股
        except Exception:
            raw = self._call(
                self.ak.stock_zh_a_daily, symbol=exchange_prefix(symbol) + symbol, start_date=start, end_date=end
            )
            df = raw.copy()
        if df.empty:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        df[DAILY_COLUMNS[2:]] = df[DAILY_COLUMNS[2:]].astype(float)
        return df[DAILY_COLUMNS]

    def fetch_adj_factor(self, symbol: str) -> pd.DataFrame:
        raw = self._call(self.ak.stock_zh_a_daily, symbol=exchange_prefix(symbol) + symbol, adjust="hfq-factor")
        return pd.DataFrame({
            "symbol": symbol,
            "date": pd.to_datetime(raw["date"]),
            "adj_factor": raw["hfq_factor"].astype(float),
        }).sort_values("date", ignore_index=True)

    def fetch_universe(self, name: str) -> list[str]:
        index_codes = {"hs300": "000300", "zz500": "000905", "zz1000": "000852"}
        if name not in index_codes:
            raise ValueError(f"未知股票池 {name}，可选：{list(index_codes)}")
        cons = self._call(self.ak.index_stock_cons_csindex, symbol=index_codes[name])
        return sorted(cons["成分券代码"].astype(str).str.zfill(6).tolist())


class SyntheticFetcher:
    """固定随机种子的合成数据：同一 symbol 每次结果相同，并在年中模拟一次分红"""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
        if len(dates) == 0:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        # 先生成固定区间（2000-2039）的完整序列再截取：随机数的数量与请求区间无关，
        # 保证不同区间请求的重叠部分完全一致
        full = pd.bdate_range("2000-01-03", "2039-12-30")
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        ret = rng.normal(0.0001, 0.02, len(full))
        close = 20 * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0, 0.005, len(full)))
        spread = np.abs(rng.normal(0, 0.01, len(full)))
        df = pd.DataFrame({
            "symbol": symbol,
            "date": full,
            "open": open_.round(2),
            "high": (np.maximum(open_, close) * (1 + spread)).round(2),
            "low": (np.minimum(open_, close) * (1 - spread)).round(2),
            "close": close.round(2),
            "volume": (rng.lognormal(15, 0.5, len(full)) // 100 * 100),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df["amount"] = df["volume"] * (df["open"] + df["close"]) / 2
        return df[df["date"].isin(dates)].reset_index(drop=True)[DAILY_COLUMNS]

    def fetch_adj_factor(self, symbol: str) -> pd.DataFrame:
        # 每年 7 月第一个工作日除息，因子乘以 1.02
        years = range(2000, pd.Timestamp.today().year + 1)
        dates = [pd.bdate_range(f"{y}-07-01", periods=1)[0] for y in years]
        return pd.DataFrame({
            "symbol": symbol,
            "date": [pd.Timestamp("1900-01-01")] + dates,
            "adj_factor": [1.0] + [1.02 ** (i + 1) for i in range(len(dates))],
        })

    def fetch_universe(self, name: str) -> list[str]:
        return [f"{600000 + i}" for i in range(10)]
