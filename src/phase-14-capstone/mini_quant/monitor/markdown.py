"""Markdown 输出工具（不依赖 tabulate）"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .._paths import OUT_DIR


def fmt(v, kind: str = "auto") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return "—"
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "bp":
        return f"{v:.1f}"
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return f"{v:,d}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame, pct_cols=(), index: bool = True, formats: dict | None = None) -> str:
    """DataFrame → Markdown 表格。pct_cols 中的列按百分比显示；formats 可按列指定格式函数"""
    formats = formats or {}
    cols = list(df.columns)
    header = ([df.index.name or ""] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        cells = [fmt(idx)] if index else []
        for c in cols:
            v = row[c]
            if c in formats:
                cells.append(formats[c](v))
            else:
                cells.append(fmt(v, "pct" if c in pct_cols else "auto"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kv_table(d: dict, pct_keys=(), key: str = "指标", value: str = "数值") -> str:
    df = pd.DataFrame({value: [fmt(v, "pct" if k in pct_keys else "auto") for k, v in d.items()]},
                      index=pd.Index(list(d), name=key))
    return md_table(df)


def write_markdown(text: str, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path
