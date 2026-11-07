"""
存储层：按年分区的 Parquet 数据仓库 + DuckDB 查询

目录结构：
    warehouse/
    ├── stock_daily/year=2024/data.parquet   不复权日线（主键 symbol + date）
    ├── adj_factor/data.parquet              后复权因子（主键 symbol + date）
    ├── meta.json                            各股票数据范围与更新时间
    └── reports/                             数据质量报告
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

KEYS = ["symbol", "date"]


def upsert(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """合并新旧数据，主键重复时保留新数据；结果按主键排序"""
    if existing is None or existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    return merged.drop_duplicates(subset=KEYS, keep="last").sort_values(KEYS, ignore_index=True)


class Warehouse:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.daily_dir = self.root / "stock_daily"
        self.factor_path = self.root / "adj_factor" / "data.parquet"
        self.meta_path = self.root / "meta.json"
        self.report_dir = self.root / "reports"

    # ---------- 元信息 ----------

    def load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {"symbols": {}}

    def save_meta(self, meta: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 写入 ----------

    def write_daily(self, df: pd.DataFrame) -> int:
        """把日线写入对应年份分区（与已有数据 upsert），返回写入的行数"""
        if df.empty:
            return 0
        df = df.assign(date=pd.to_datetime(df["date"]))
        for year, part in df.groupby(df["date"].dt.year):
            path = self.daily_dir / f"year={year}" / "data.parquet"
            existing = pd.read_parquet(path) if path.exists() else None
            path.parent.mkdir(parents=True, exist_ok=True)
            upsert(existing, part).to_parquet(path, index=False)
        return len(df)

    def write_adj_factor(self, df: pd.DataFrame):
        if df.empty:
            return
        existing = pd.read_parquet(self.factor_path) if self.factor_path.exists() else None
        self.factor_path.parent.mkdir(parents=True, exist_ok=True)
        upsert(existing, df).to_parquet(self.factor_path, index=False)

    # ---------- 读取 ----------

    def read_daily(self, symbols: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """读取日线，自动利用年份分区跳过无关文件"""
        if not self.daily_dir.exists():
            return pd.DataFrame()
        filters = []
        if start:
            filters.append(("year", ">=", pd.Timestamp(start).year))
        if end:
            filters.append(("year", "<=", pd.Timestamp(end).year))
        df = pd.read_parquet(self.daily_dir, filters=filters or None)
        if df.empty:
            return df
        df = df.drop(columns="year")
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.sort_values(KEYS, ignore_index=True)

    def read_adj_factor(self) -> pd.DataFrame:
        if not self.factor_path.exists():
            return pd.DataFrame(columns=["symbol", "date", "adj_factor"])
        return pd.read_parquet(self.factor_path)

    def read_daily_adjusted(self, symbols: list[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """日线 + 当日后复权因子 + 后复权收盘价（等比复权）"""
        daily = self.read_daily(symbols, start, end)
        if daily.empty:
            return daily
        factor = self.read_adj_factor().sort_values("date")
        # 不同来源写入的时间精度可能不同（ns / us），merge_asof 要求一致
        daily["date"] = daily["date"].astype("datetime64[ns]")
        factor["date"] = factor["date"].astype("datetime64[ns]")
        merged = pd.merge_asof(
            daily.sort_values("date"), factor, on="date", by="symbol", direction="backward"
        )
        merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
        merged["adj_close"] = merged["close"] * merged["adj_factor"]
        return merged.sort_values(KEYS, ignore_index=True)

    def last_date(self, symbol: str) -> pd.Timestamp | None:
        info = self.load_meta()["symbols"].get(symbol)
        return pd.Timestamp(info["date_max"]) if info else None

    # ---------- 查询 ----------

    def query(self, sql: str) -> pd.DataFrame:
        """用 DuckDB 执行 SQL，可用的表：stock_daily、adj_factor"""
        con = duckdb.connect()
        try:
            if self.daily_dir.exists():
                con.execute(
                    f"CREATE VIEW stock_daily AS SELECT * FROM "
                    f"read_parquet('{self.daily_dir}/*/*.parquet', hive_partitioning = true)"
                )
            if self.factor_path.exists():
                con.execute(f"CREATE VIEW adj_factor AS SELECT * FROM read_parquet('{self.factor_path}')")
            return con.execute(sql).df()
        finally:
            con.close()

    # ---------- 元信息刷新 ----------

    def refresh_meta(self, source: str):
        """根据仓库中的实际数据刷新每只股票的范围"""
        stats = self.query(
            "SELECT symbol, min(date) AS date_min, max(date) AS date_max, count(*) AS rows "
            "FROM stock_daily GROUP BY symbol ORDER BY symbol"
        )
        meta = self.load_meta()
        now = datetime.now().isoformat(timespec="seconds")
        for row in stats.itertuples():
            meta["symbols"][row.symbol] = {
                "date_min": str(pd.Timestamp(row.date_min).date()),
                "date_max": str(pd.Timestamp(row.date_max).date()),
                "rows": int(row.rows),
            }
        meta["source"] = source
        meta["updated_at"] = now
        self.save_meta(meta)
        return meta
