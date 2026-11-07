"""
数据质量检查

对仓库中的日线做批量检查，输出问题清单与 Markdown 报告。
检查项：
    1. 重复主键          (symbol, date) 出现多次
    2. 价格缺失          OHLC 中存在 NaN
    3. 非正价格/负成交量
    4. OHLC 逻辑错误     不满足 low ≤ open/close ≤ high
    5. 涨跌幅越界        不复权收益超出板块涨跌幅限制，且不是除权日
    6. 复权跳变          后复权收益超出板块涨跌幅限制（复权因子可能有误）
    7. 数据缺口          相邻两条记录间隔超过 15 个自然日（长期停牌或数据缺失）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

TOLERANCE = 0.005  # 涨跌幅容差，覆盖价格四舍五入


def limit_pct(symbol: str) -> float:
    """板块涨跌幅限制（不区分 ST 与新股上市初期，这两类会产生可解释的越界告警）"""
    if symbol.startswith(("688", "689", "300", "301")):
        return 0.20
    if symbol.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


def check_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: 长表，至少包含 symbol, date, open, high, low, close, volume；
            若包含 adj_close / adj_factor，会额外检查复权跳变并排除除权日误报
    Returns:
        问题清单 DataFrame[symbol, date, check, detail]
    """
    columns = ["symbol", "date", "check", "detail"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    issues: list[pd.DataFrame] = []

    def add(mask: pd.Series, check: str, detail: pd.Series | str):
        mask = mask.fillna(False)
        if mask.any():
            part = df.loc[mask, ["symbol", "date"]].copy()
            part["check"] = check
            part["detail"] = detail[mask] if isinstance(detail, pd.Series) else detail
            issues.append(part)

    price_cols = ["open", "high", "low", "close"]
    add(df.duplicated(["symbol", "date"], keep=False), "重复主键", "同一股票同一日期出现多行")
    add(df[price_cols].isna().any(axis=1), "价格缺失", "OHLC 存在 NaN")
    add((df[price_cols] <= 0).any(axis=1), "非正价格", "价格 ≤ 0")
    add(df["volume"] < 0, "负成交量", "volume < 0")

    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    detail = (
        "O=" + df["open"].round(2).astype(str) + " H=" + df["high"].round(2).astype(str)
        + " L=" + df["low"].round(2).astype(str) + " C=" + df["close"].round(2).astype(str)
    )
    add(bad_ohlc, "OHLC 逻辑错误", detail)

    limits = df["symbol"].map(limit_pct)
    same_symbol = df["symbol"] == df["symbol"].shift()
    ret = (df["close"] / df["close"].shift() - 1).where(same_symbol)

    ex_day = pd.Series(False, index=df.index)
    if "adj_factor" in df.columns:
        ex_day = (df["adj_factor"] != df["adj_factor"].shift()) & same_symbol
    add((ret.abs() > limits + TOLERANCE) & ~ex_day, "涨跌幅越界",
        "日收益 " + (ret * 100).round(2).astype(str) + "%")

    if "adj_close" in df.columns:
        adj_ret = (df["adj_close"] / df["adj_close"].shift() - 1).where(same_symbol)
        add(adj_ret.abs() > limits + TOLERANCE, "复权跳变",
            "复权收益 " + (adj_ret * 100).round(2).astype(str) + "%")

    gap = (df["date"] - df["date"].shift()).dt.days.where(same_symbol)
    add(gap > 15, "数据缺口", "距上一条记录 " + gap.fillna(0).astype(int).astype(str) + " 天")

    if not issues:
        return pd.DataFrame(columns=columns)
    return pd.concat(issues, ignore_index=True).sort_values(["symbol", "date"], ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """每只股票的数据覆盖概况"""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("symbol").agg(
        rows=("date", "size"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        zero_volume_days=("volume", lambda v: int((v == 0).sum())),
    ).reset_index().assign(date_min=lambda x: x["date_min"].dt.date, date_max=lambda x: x["date_max"].dt.date)


def to_markdown(df: pd.DataFrame) -> str:
    """DataFrame → Markdown 表格（避免依赖 tabulate）"""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(df: pd.DataFrame, issues: pd.DataFrame, report_dir: Path) -> Path:
    """生成 Markdown 质量报告"""
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = report_dir / f"quality_{now:%Y%m%d_%H%M%S}.md"

    summary = summarize(df)
    lines = [
        "# 数据质量报告",
        "",
        f"- 生成时间：{now:%Y-%m-%d %H:%M:%S}",
        f"- 股票数量：{df['symbol'].nunique() if not df.empty else 0}",
        f"- 总行数：{len(df):,}",
        f"- 问题数量：{len(issues)}",
        "",
        "## 数据覆盖",
        "",
        to_markdown(summary) if not summary.empty else "（无数据）",
        "",
        "## 问题汇总",
        "",
    ]
    if issues.empty:
        lines.append("✅ 未发现问题")
    else:
        counts = issues.groupby("check").size().rename("数量").reset_index()
        lines += [to_markdown(counts), "", "## 问题明细（最多 200 条）", ""]
        lines.append(to_markdown(issues.head(200).assign(date=lambda x: x["date"].dt.date)))
    lines += [
        "",
        "## 说明",
        "",
        "- 涨跌幅越界：ST 股、新股上市初期不设或另设涨跌幅限制，会产生可解释的告警",
        "- 复权跳变：若后复权收益超出涨跌幅限制，复权因子或价格可能有误，需人工复核",
        "- 数据缺口：通常是长期停牌，需与交易所停复牌公告核对",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
