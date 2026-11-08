"""
事件日志（第 10 课）：追加写（append-only），可审计、可回放

"先写日志，再发请求"（Write-Ahead）：订单意图在发出之前就落盘，
进程崩溃后可以知道"我曾经想发哪些单"，再去交易所核实它们的结局。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

from .models import OrderData, OrderRequest, PositionData, TradeData, from_dict, to_dict
from .oms import OMS


class JsonlEventLog:
    """每行一个 JSON 事件；写入后 fsync，保证断电也不丢已确认的记录"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired_bytes = self._repair_tail()
        self._seq = sum(1 for _ in self.path.open(encoding="utf-8")) if self.path.exists() else 0

    def _repair_tail(self) -> int:
        """
        崩溃可能留下写了一半的最后一行。打开时截断到最后一个换行符，
        否则下一条记录会接在残缺内容后面，连带损坏一条完好的记录。
        """
        if not self.path.exists():
            return 0
        data = self.path.read_bytes()
        keep = data.rfind(b"\n") + 1
        if keep == len(data):
            return 0
        with self.path.open("r+b") as f:
            f.truncate(keep)
        return len(data) - keep

    def append(self, etype: str, payload: dict[str, Any]) -> int:
        self._seq += 1
        line = json.dumps({"seq": self._seq, "ts": time.time(), "type": etype, "payload": payload}, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return self._seq

    def replay(self, etype: str | None = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # 崩溃时写了一半的最后一行：跳过
                    continue
                if etype is None or rec["type"] == etype:
                    yield rec


class SqliteEventLog:
    """SQLite 版本：WAL 模式，适合需要按类型 / 时间查询的场景"""

    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, payload TEXT)"
        )

    def append(self, etype: str, payload: dict[str, Any]) -> int:
        cur = self.db.execute(
            "INSERT INTO events (ts, type, payload) VALUES (?, ?, ?)",
            (time.time(), etype, json.dumps(payload, ensure_ascii=False)),
        )
        self.db.commit()
        return cur.lastrowid

    def replay(self, etype: str | None = None) -> Iterator[dict]:
        sql = "SELECT seq, ts, type, payload FROM events"
        rows = self.db.execute(sql + " WHERE type = ? ORDER BY seq", (etype,)) if etype else self.db.execute(
            sql + " ORDER BY seq"
        )
        for seq, ts, t, payload in rows:
            yield {"seq": seq, "ts": ts, "type": t, "payload": json.loads(payload)}

    def close(self) -> None:
        self.db.close()


def record(log, etype: str, obj: Any) -> int:
    """把模型对象写入事件日志"""
    if etype == "positions_snapshot":
        return log.append(etype, {"positions": [to_dict(p) for p in obj]})
    return log.append(etype, to_dict(obj))


def rebuild_oms(log) -> OMS:
    """按顺序回放事件日志，重建 OMS 状态"""
    oms = OMS()
    for rec in log.replay():
        etype, payload = rec["type"], rec["payload"]
        if etype == "positions_snapshot":
            oms.positions.load([from_dict(PositionData, p) for p in payload["positions"]])
        elif etype == "order_request":
            oms.create_order(from_dict(OrderRequest, payload))
        elif etype == "order":
            oms.on_order(from_dict(OrderData, payload))
        elif etype == "trade":
            oms.on_trade(from_dict(TradeData, payload))
    return oms
