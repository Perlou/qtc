import pandas as pd

from main import init_warehouse, main, update_warehouse


def test_init_then_update_is_incremental_and_idempotent(fetcher, warehouse):
    init_warehouse(fetcher, warehouse, ["600519", "000001"], "20240101", "20240630", "synthetic")
    assert warehouse.last_date("600519") == pd.Timestamp("2024-06-28")

    update_warehouse(fetcher, warehouse, "20240930", lookback_days=7, source="synthetic")
    df = warehouse.read_daily()
    expected = len(pd.bdate_range("2024-01-01", "2024-09-30"))
    assert (df.groupby("symbol").size() == expected).all()

    update_warehouse(fetcher, warehouse, "20240930", lookback_days=7, source="synthetic")
    assert len(warehouse.read_daily()) == len(df)


def test_cli_offline_end_to_end(tmp_path, capsys):
    root = str(tmp_path / "wh")
    assert main(["--root", root, "--offline", "init", "--symbols", "600519,300750",
                 "--start", "20240101", "--end", "20240331"]) == 0
    assert main(["--root", root, "query", "SELECT count(DISTINCT symbol) AS n FROM stock_daily"]) == 0
    assert main(["--root", root, "report"]) == 0
    out = capsys.readouterr().out
    assert "写入" in out and "未发现数据质量问题" in out
