import pandas as pd
import pytest

from fetcher import DAILY_COLUMNS, exchange_prefix, retry


def test_synthetic_daily_schema(fetcher):
    df = fetcher.fetch_daily("600519", "20240101", "20240131")
    assert list(df.columns) == DAILY_COLUMNS
    assert len(df) == len(pd.bdate_range("2024-01-01", "2024-01-31"))
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()


def test_synthetic_daily_is_consistent_across_ranges(fetcher):
    """不同请求区间的重叠部分必须完全一致，否则增量更新会制造假的数据修正"""
    a = fetcher.fetch_daily("600519", "20240101", "20240331").set_index("date")
    b = fetcher.fetch_daily("600519", "20240301", "20240630").set_index("date")
    overlap = a.index.intersection(b.index)
    assert len(overlap) > 0
    pd.testing.assert_frame_equal(a.loc[overlap], b.loc[overlap])


def test_retry_eventually_succeeds():
    calls = {"n": 0}

    @retry(max_retries=3, sleep=lambda s: None)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_max_retries():
    calls = {"n": 0}

    @retry(max_retries=2, sleep=lambda s: None)
    def always_fail():
        calls["n"] += 1
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        always_fail()
    assert calls["n"] == 2


@pytest.mark.parametrize("symbol,prefix", [("600519", "sh"), ("688981", "sh"), ("000001", "sz"),
                                           ("300750", "sz"), ("920001", "bj"), ("830799", "bj")])
def test_exchange_prefix(symbol, prefix):
    assert exchange_prefix(symbol) == prefix
