import numpy as np
import pandas as pd

from quality import check_quality, limit_pct, write_report


def test_clean_synthetic_data_has_no_issues(fetcher):
    df = fetcher.fetch_daily("600519", "20240101", "20240630")
    assert check_quality(df).empty


def test_detects_injected_errors(fetcher):
    df = fetcher.fetch_daily("600519", "20240101", "20240630")
    df.loc[5, "close"] = np.nan
    df.loc[10, "high"] = df.loc[10, "low"] - 0.1
    df.loc[20, "close"] = df.loc[19, "close"] * 1.25
    df.loc[30, "volume"] = -100
    df = pd.concat([df, df.iloc[[40]]], ignore_index=True)
    df = df.drop(index=range(60, 75))  # 制造 3 周的数据缺口

    checks = set(check_quality(df)["check"])
    assert {"价格缺失", "OHLC 逻辑错误", "涨跌幅越界", "负成交量", "重复主键", "数据缺口"} <= checks


def test_ex_rights_day_is_not_flagged_but_bad_factor_is(fetcher):
    df = fetcher.fetch_daily("600519", "20240601", "20240731").reset_index(drop=True)
    i = df.index[df["date"] == pd.Timestamp("2024-07-01")][0]
    df.loc[i:, ["open", "high", "low", "close"]] *= 0.85  # 除权导致不复权价格下跌 15%
    df["adj_factor"] = 1.0
    df.loc[i:, "adj_factor"] = 1 / 0.85  # 正确的复权因子
    df["adj_close"] = df["close"] * df["adj_factor"]
    assert check_quality(df).empty

    df.loc[i:, "adj_factor"] = 1.0  # 因子缺失 → 复权价格也下跌 15%
    df["adj_close"] = df["close"] * df["adj_factor"]
    assert "复权跳变" in set(check_quality(df)["check"])


def test_limit_pct_by_board():
    assert limit_pct("600519") == 0.10
    assert limit_pct("300750") == 0.20
    assert limit_pct("688981") == 0.20
    assert limit_pct("830799") == 0.30


def test_write_report(fetcher, tmp_path):
    df = fetcher.fetch_daily("600519", "20240101", "20240131")
    path = write_report(df, check_quality(df), tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "数据质量报告" in text and "未发现问题" in text
