"""
本地 A 股行情数据仓库 —— 命令行入口

用法：
    python main.py init   --start 20200101                    # 默认股票池全量下载
    python main.py init   --universe hs300 --start 20230101   # 沪深 300 成分股
    python main.py init   --symbols 600519,000001
    python main.py update                                     # 增量更新到今天
    python main.py query  "SELECT symbol, count(*) FROM stock_daily GROUP BY 1"
    python main.py report                                     # 数据质量报告
    python main.py --offline init                             # 使用合成数据（无网络）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import Config
from fetcher import AkshareFetcher, Fetcher, SyntheticFetcher
from quality import check_quality, write_report
from storage import Warehouse


def download(fetcher: Fetcher, wh: Warehouse, symbols: list[str], start: str, end: str) -> dict:
    """下载一批股票的日线与复权因子并写入仓库；单只失败不影响其他股票"""
    frames, factors, failed = [], [], []
    for i, symbol in enumerate(symbols, 1):
        try:
            daily = fetcher.fetch_daily(symbol, start, end)
            frames.append(daily)
            factors.append(fetcher.fetch_adj_factor(symbol))
            print(f"  [{i}/{len(symbols)}] {symbol}: {len(daily)} 行")
        except Exception as e:
            failed.append(symbol)
            print(f"  [{i}/{len(symbols)}] {symbol}: ❌ {type(e).__name__}: {str(e)[:80]}")

    rows = wh.write_daily(pd.concat(frames, ignore_index=True)) if frames else 0
    if factors:
        wh.write_adj_factor(pd.concat(factors, ignore_index=True))
    return {"rows": rows, "failed": failed}


def init_warehouse(fetcher: Fetcher, wh: Warehouse, symbols: list[str], start: str, end: str, source: str) -> dict:
    print(f"📥 全量下载 {len(symbols)} 只股票，{start} ~ {end}")
    result = download(fetcher, wh, symbols, start, end)
    if result["rows"]:
        wh.refresh_meta(source)
    return result


def update_warehouse(fetcher: Fetcher, wh: Warehouse, end: str, lookback_days: int, source: str) -> dict:
    """对仓库中已有的每只股票，从 (最后日期 - lookback_days) 开始增量下载"""
    meta = wh.load_meta()["symbols"]
    if not meta:
        raise RuntimeError("仓库为空，请先运行 init")
    total = {"rows": 0, "failed": []}
    # 按起始日期分组，减少重复逻辑
    by_start: dict[str, list[str]] = {}
    for symbol, info in meta.items():
        start = (pd.Timestamp(info["date_max"]) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
        by_start.setdefault(start, []).append(symbol)
    for start, symbols in sorted(by_start.items()):
        print(f"📥 增量更新 {len(symbols)} 只股票，{start} ~ {end}")
        r = download(fetcher, wh, symbols, start, end)
        total["rows"] += r["rows"]
        total["failed"] += r["failed"]
    wh.refresh_meta(source)
    return total


def run_report(wh: Warehouse) -> tuple[pd.DataFrame, Path]:
    df = wh.read_daily_adjusted()
    issues = check_quality(df)
    path = write_report(df, issues, wh.report_dir)
    return issues, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 A 股行情数据仓库")
    parser.add_argument("--root", type=Path, default=Config().root, help="仓库目录")
    parser.add_argument("--offline", action="store_true", help="使用合成数据（无需网络）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="全量下载")
    p_init.add_argument("--start", default="20200101")
    p_init.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    group = p_init.add_mutually_exclusive_group()
    group.add_argument("--symbols", help="逗号分隔的股票代码")
    group.add_argument("--universe", choices=["hs300", "zz500", "zz1000"], help="指数成分股")

    p_update = sub.add_parser("update", help="增量更新")
    p_update.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))

    p_query = sub.add_parser("query", help="SQL 查询（表：stock_daily, adj_factor）")
    p_query.add_argument("sql")

    sub.add_parser("report", help="数据质量报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(root=args.root)
    wh = Warehouse(cfg.root)
    source = "synthetic" if args.offline else "akshare"

    def make_fetcher() -> Fetcher:
        if args.offline:
            return SyntheticFetcher()
        return AkshareFetcher(cfg.request_interval, cfg.max_retries)

    if args.command == "init":
        fetcher = make_fetcher()
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        elif args.universe:
            symbols = fetcher.fetch_universe(args.universe)
        else:
            symbols = cfg.symbols
        r = init_warehouse(fetcher, wh, symbols, args.start, args.end, source)
        print(f"✅ 写入 {r['rows']:,} 行，失败 {len(r['failed'])} 只 {r['failed'] or ''}")
        print(f"📁 仓库位置：{wh.root}")

    elif args.command == "update":
        r = update_warehouse(make_fetcher(), wh, args.end, cfg.lookback_days, source)
        print(f"✅ 更新 {r['rows']:,} 行（含重叠区间），失败 {len(r['failed'])} 只 {r['failed'] or ''}")

    elif args.command == "query":
        with pd.option_context("display.width", 160, "display.max_rows", 50):
            print(wh.query(args.sql))

    elif args.command == "report":
        issues, path = run_report(wh)
        if issues.empty:
            print("✅ 未发现数据质量问题")
        else:
            print(f"⚠️ 发现 {len(issues)} 个问题：")
            print(issues.groupby("check").size().to_string())
        print(f"📝 报告：{path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
