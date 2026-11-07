"""
数据存储：CSV / Parquet / DuckDB
================================

学习目标：
    1. 比较 CSV、Parquet 的存储体积与读写速度
    2. 设计按年分区的本地行情数据仓库
    3. 用 DuckDB 直接对 Parquet 文件做 SQL 分析
    4. 实现增量更新与去重（upsert），并记录数据元信息

核心概念：
    - 列式存储（Parquet）：按列压缩存储，只读需要的列，适合分析
    - 分区（Partition）：按年/月/品种拆分文件，查询时跳过无关分区
    - DuckDB：嵌入式分析数据库，"分析领域的 SQLite"，可直接查询 Parquet
    - 增量更新：只下载新数据，与已有数据合并去重

前置知识：
    - 07-trading-calendar-alignment.py

环境要求：
    - pip install pandas pyarrow duckdb
    - 本课使用合成数据，不需要网络
"""

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd
from dotenv import load_dotenv

from utils import DATA_DIR
from utils.data import synthetic_ohlcv

load_dotenv()

WORK_DIR = DATA_DIR / "lesson08"


def build_demo_panel(n_stocks: int = 300, start: str = "20150101", end: str = "20241231") -> pd.DataFrame:
    """生成多只股票的长表行情（合成数据，用于性能对比）"""
    frames = []
    for i in range(n_stocks):
        symbol = f"{600000 + i}"
        df = synthetic_ohlcv(symbol, start, end, start_price=10 + i % 50)
        frames.append(df.assign(symbol=symbol).reset_index())
    return pd.concat(frames, ignore_index=True)


def timeit(fn, repeat: int = 3) -> float:
    """返回多次运行的最短耗时（秒）"""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size


# ==================== 第一部分：格式对比 ====================


def format_comparison(panel: pd.DataFrame):
    print("=" * 60)
    print("第一部分：CSV vs Parquet")
    print("=" * 60)

    print(f"\n📊 测试数据：{panel['symbol'].nunique()} 只股票 × 10 年日线 = {len(panel):,} 行")

    csv_path = WORK_DIR / "panel.csv"
    pq_snappy = WORK_DIR / "panel_snappy.parquet"
    pq_zstd = WORK_DIR / "panel_zstd.parquet"

    results = []
    for name, path, write, read, read_cols in [
        ("CSV", csv_path,
         lambda: panel.to_csv(csv_path, index=False),
         lambda: pd.read_csv(csv_path, parse_dates=["date"], dtype={"symbol": str}),
         lambda: pd.read_csv(csv_path, usecols=["date", "symbol", "close"], dtype={"symbol": str})),
        ("Parquet (snappy)", pq_snappy,
         lambda: panel.to_parquet(pq_snappy, compression="snappy"),
         lambda: pd.read_parquet(pq_snappy),
         lambda: pd.read_parquet(pq_snappy, columns=["date", "symbol", "close"])),
        ("Parquet (zstd)", pq_zstd,
         lambda: panel.to_parquet(pq_zstd, compression="zstd"),
         lambda: pd.read_parquet(pq_zstd),
         lambda: pd.read_parquet(pq_zstd, columns=["date", "symbol", "close"])),
    ]:
        t_write = timeit(write, repeat=1)
        t_read = timeit(read)
        t_cols = timeit(read_cols)
        results.append([name, f"{dir_size(path) / 1e6:.1f} MB", f"{t_write:.2f}s", f"{t_read:.3f}s", f"{t_cols:.3f}s"])

    print(pd.DataFrame(results, columns=["格式", "体积", "写入", "全量读取", "只读 3 列"]).to_string(index=False))

    dtypes_kept = pd.read_parquet(pq_snappy).dtypes["date"]
    print(f"""
    💡 结论：
    - Parquet 体积更小、读取快一个数量级，且只读部分列时更快（列式存储）
    - Parquet 保存了数据类型（date 读回来仍是 {dtypes_kept}），CSV 需要每次重新解析
    - CSV 唯一的优势：人类可读、通用 → 适合导出给别人看，不适合做研究数据仓库
    """)


# ==================== 第二部分：分区数据仓库 ====================


def partitioned_warehouse(panel: pd.DataFrame) -> Path:
    print("\n" + "=" * 60)
    print("第二部分：按年分区的数据仓库")
    print("=" * 60)

    wh = WORK_DIR / "warehouse" / "stock_daily"
    if wh.exists():
        shutil.rmtree(wh)
    panel.assign(year=panel["date"].dt.year).to_parquet(wh, partition_cols=["year"], index=False)

    years = sorted(p.name for p in wh.iterdir() if p.is_dir())
    print(f"""
    📁 目录结构（Hive 风格分区）：
    {wh.relative_to(DATA_DIR.parent)}/
    ├── {years[0]}/
    │   └── xxxx.parquet
    ├── ...
    └── {years[-1]}/
        └── xxxx.parquet

    共 {len(years)} 个分区，总大小 {dir_size(wh) / 1e6:.1f} MB
    """)

    t_all = timeit(lambda: pd.read_parquet(wh))
    t_one = timeit(lambda: pd.read_parquet(wh, filters=[("year", "=", 2024)]))
    print(f"   读取全部：{t_all:.3f}s；只读 2024 年（分区过滤）：{t_one:.3f}s")
    print("""
    📌 分区设计原则：
    - 按最常用的过滤条件分区：日线按年，分钟线按日期，tick 按日期 + 品种
    - 分区不要太碎：每个文件至少几 MB，否则文件太多反而变慢
    - 原始数据（raw）和处理后数据（processed）分开存放，原始数据只追加不修改
    """)
    return wh


# ==================== 第三部分：DuckDB 分析 ====================


def duckdb_analysis(wh: Path, panel: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：用 DuckDB 直接查询 Parquet")
    print("=" * 60)

    con = duckdb.connect()
    glob = f"{wh}/**/*.parquet"

    # 每只股票每年的收益率：SQL 窗口函数
    sql = f"""
        WITH yearly AS (
            SELECT symbol, year,
                   arg_min(close, date) AS first_close,
                   arg_max(close, date) AS last_close
            FROM read_parquet('{glob}', hive_partitioning = true)
            GROUP BY symbol, year
        )
        SELECT year,
               count(*)                                   AS n_stocks,
               round(avg(last_close / first_close - 1), 4) AS avg_return,
               round(median(last_close / first_close - 1), 4) AS median_return
        FROM yearly
        GROUP BY year
        ORDER BY year
    """
    t0 = time.perf_counter()
    result = con.execute(sql).df()
    t_duck = time.perf_counter() - t0
    print("\n📊 SQL：每年全市场股票的平均 / 中位数收益（合成数据）")
    print(result.to_string(index=False))

    # 同样的计算用 pandas
    def pandas_version():
        df = panel.assign(year=panel["date"].dt.year).sort_values("date")
        g = df.groupby(["symbol", "year"])["close"].agg(["first", "last"])
        r = (g["last"] / g["first"] - 1).groupby("year").agg(["count", "mean", "median"])
        return r

    t_pd = timeit(pandas_version, repeat=1)
    print(f"\n⏱️  DuckDB（含读文件）：{t_duck:.3f}s；pandas（数据已在内存）：{t_pd:.3f}s")

    # 直接查询内存中的 DataFrame
    top = con.execute("""
        SELECT symbol, round(max(close) / min(close), 2) AS max_min_ratio
        FROM panel GROUP BY symbol ORDER BY max_min_ratio DESC LIMIT 3
    """).df()
    print("\n📊 DuckDB 也能直接查询内存中的 pandas DataFrame（变量名即表名）：")
    print(top.to_string(index=False))
    con.close()

    print("""
    💡 DuckDB 适合：数据比内存大、需要复杂 SQL（窗口函数、多表 JOIN）、
       多个研究脚本共享同一份 Parquet 数据仓库。
       pandas 适合：数据已在内存、需要灵活的逐步变换。两者可以无缝配合。
    """)


# ==================== 第四部分：增量更新 ====================


def upsert(existing: pd.DataFrame, new: pd.DataFrame, keys=("symbol", "date")) -> pd.DataFrame:
    """合并新旧数据：主键重复时保留新数据"""
    merged = pd.concat([existing, new], ignore_index=True)
    return merged.drop_duplicates(subset=list(keys), keep="last").sort_values(list(keys)).reset_index(drop=True)


def incremental_update():
    print("\n" + "=" * 60)
    print("第四部分：增量更新与去重")
    print("=" * 60)

    full = synthetic_ohlcv("600000", "20240101", "20241231").assign(symbol="600000").reset_index()
    store_path = WORK_DIR / "incremental.parquet"
    meta_path = WORK_DIR / "incremental.meta.json"

    # 第 1 次：下载截至 11 月底的数据
    first = full[full["date"] <= "2024-11-30"]
    first.to_parquet(store_path, index=False)
    print(f"   第 1 次下载：{len(first)} 行，截至 {first['date'].max().date()}")

    # 第 2 次：从已有最大日期"往回多取 5 天"再下载 —— 覆盖可能被修正的最近数据
    existing = pd.read_parquet(store_path)
    last_date = existing["date"].max()
    fetch_from = last_date - pd.Timedelta(days=5)
    new = full[full["date"] >= fetch_from].copy()
    new.loc[new.index[0], "close"] += 0.01  # 模拟数据源修正了一条旧数据
    updated = upsert(existing, new)
    updated.to_parquet(store_path, index=False)
    print(f"   第 2 次下载：从 {fetch_from.date()} 开始取 {len(new)} 行（含 {(new['date'] <= last_date).sum()} 行重叠）")
    print(f"   合并去重后：{len(updated)} 行，截至 {updated['date'].max().date()}")
    assert updated["date"].is_unique, "主键必须唯一"

    # 记录元信息：数据从哪来、什么时候下载、覆盖什么范围
    meta = {
        "source": "synthetic",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(updated),
        "date_min": str(updated["date"].min().date()),
        "date_max": str(updated["date"].max().date()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n   📝 元信息 {meta_path.name}：{meta}")

    print("""
    📌 增量更新要点：
    1. 用 (symbol, date) 作为主键，合并后去重，保证幂等（重复运行结果不变）
    2. 往回多取几天：数据源可能修正最近的数据
    3. 记录元信息：来源、下载时间、范围 —— 研究结论要能追溯到具体数据版本
    4. 复权价格不能增量更新（每次分红都会改变历史前复权价）→ 只存不复权价 + 复权因子
    """)


# ==================== 主函数 ====================


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("⏳ 生成测试数据...")
    panel = build_demo_panel()

    format_comparison(panel)
    wh = partitioned_warehouse(panel)
    duckdb_analysis(wh, panel)
    incremental_update()

    print(f"\n🗂️  演示文件位于 {WORK_DIR}（data/ 已被 git 忽略，可随时删除）")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把分区方式改为 year + 板块（主板/创业板/科创板），比较查询"2024 年创业板"的速度
    2. 用 DuckDB 计算每只股票的 20 日滚动波动率（提示：窗口函数 OVER (PARTITION BY ...)）
    3. 把 upsert 改为直接在 DuckDB 中完成（INSERT OR REPLACE / 主键表）
    4. 思考：分钟线数据（每只股票每天 240 行）应该怎样分区？
    """)


if __name__ == "__main__":
    main()
