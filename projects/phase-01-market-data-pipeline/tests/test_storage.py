import pandas as pd

from storage import upsert


def test_upsert_keeps_latest_and_is_idempotent():
    old = pd.DataFrame({"symbol": ["A", "A"], "date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [1.0, 2.0]})
    new = pd.DataFrame({"symbol": ["A", "A"], "date": pd.to_datetime(["2024-01-03", "2024-01-04"]), "close": [2.5, 3.0]})
    merged = upsert(old, new)
    assert merged["close"].tolist() == [1.0, 2.5, 3.0]
    pd.testing.assert_frame_equal(upsert(merged, new), merged)


def test_write_daily_partitions_by_year(fetcher, warehouse):
    df = fetcher.fetch_daily("600519", "20231201", "20240131")
    warehouse.write_daily(df)
    years = sorted(p.name for p in warehouse.daily_dir.iterdir())
    assert years == ["year=2023", "year=2024"]
    back = warehouse.read_daily()
    assert len(back) == len(df)


def test_rewrite_does_not_duplicate(fetcher, warehouse):
    df = fetcher.fetch_daily("600519", "20240101", "20240331")
    warehouse.write_daily(df)
    warehouse.write_daily(df)
    assert len(warehouse.read_daily()) == len(df)


def test_read_daily_filters(fetcher, warehouse):
    for s in ["600519", "000001"]:
        warehouse.write_daily(fetcher.fetch_daily(s, "20230101", "20241231"))
    df = warehouse.read_daily(symbols=["000001"], start="20240601", end="20240630")
    assert set(df["symbol"]) == {"000001"}
    assert df["date"].min() >= pd.Timestamp("2024-06-01")
    assert df["date"].max() <= pd.Timestamp("2024-06-30")


def test_adjusted_close_uses_factor_as_of_date(fetcher, warehouse):
    warehouse.write_daily(fetcher.fetch_daily("600519", "20240601", "20240731"))
    warehouse.write_adj_factor(fetcher.fetch_adj_factor("600519"))
    df = warehouse.read_daily_adjusted().set_index("date")
    ex_date = pd.Timestamp("2024-07-01")  # SyntheticFetcher 在每年 7 月第一个工作日除息
    assert df.loc[ex_date, "adj_factor"] / df.loc[ex_date - pd.offsets.BDay(), "adj_factor"] == 1.02
    assert (df["adj_close"] == df["close"] * df["adj_factor"]).all()


def test_query_and_meta(fetcher, warehouse):
    warehouse.write_daily(fetcher.fetch_daily("600519", "20240101", "20240131"))
    out = warehouse.query("SELECT count(*) AS n FROM stock_daily")
    assert out["n"].iloc[0] == len(pd.bdate_range("2024-01-01", "2024-01-31"))
    meta = warehouse.refresh_meta("synthetic")
    assert meta["symbols"]["600519"]["date_max"] == "2024-01-31"
    assert warehouse.last_date("600519") == pd.Timestamp("2024-01-31")
