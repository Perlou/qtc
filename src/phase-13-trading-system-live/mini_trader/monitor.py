"""
监控与告警（第 09 课）

    StalenessMonitor   行情断流检测：进程活着不代表行情在更新
    HeartbeatWatchdog  心跳 / Dead Man's Switch：组件长时间没有"报平安"就告警
    Metrics            计数器、仪表、耗时分布
    AlertManager       告警分级（P0-P3）、去重、静默（抑制）
    JsonFormatter      结构化日志：一行一个 JSON，便于检索和聚合
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


class StalenessMonitor:
    def __init__(self, max_silence_sec: float, clock: Callable[[], float] = time.monotonic):
        self.max_silence = max_silence_sec
        self.clock = clock
        self.last_seen: dict[str, float] = {}

    def watch(self, symbol: str) -> None:
        """登记需要监控的标的：从登记时刻起计时，一笔行情都没来也能发现"""
        self.last_seen.setdefault(symbol, self.clock())

    def on_tick(self, symbol: str) -> None:
        self.last_seen[symbol] = self.clock()

    def check(self, in_session: bool = True) -> dict[str, float]:
        """返回 {标的: 已静默秒数}；非交易时段没有行情是正常的"""
        if not in_session:
            return {}
        now = self.clock()
        return {s: now - t for s, t in self.last_seen.items() if now - t > self.max_silence}


class HeartbeatWatchdog:
    def __init__(self, timeout_sec: float, clock: Callable[[], float] = time.monotonic):
        self.timeout = timeout_sec
        self.clock = clock
        self.last_beat: dict[str, float] = {}

    def beat(self, source: str) -> None:
        self.last_beat[source] = self.clock()

    def dead(self) -> list[str]:
        now = self.clock()
        return [s for s, t in self.last_beat.items() if now - t > self.timeout]


class Metrics:
    def __init__(self):
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.timings: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def set(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        self.timings[name].append(value)

    def ratio(self, num: str, den: str) -> float:
        d = self.counters.get(den, 0.0)
        return self.counters.get(num, 0.0) / d if d else 0.0

    def summary(self, name: str) -> dict[str, float]:
        arr = np.asarray(self.timings.get(name, []), dtype=float)
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }


LEVELS = {
    "P0": "致命：电话 + 即时消息，立即处理",
    "P1": "严重：即时消息，尽快处理",
    "P2": "警告：即时消息，当日处理",
    "P3": "信息：邮件 / 日报，知悉即可",
}


@dataclass
class Alert:
    level: str
    key: str  # 去重键：同一个问题用同一个 key
    title: str
    detail: str = ""
    ts: float = 0.0
    repeats: int = 0  # 去重窗口内被合并的次数


@dataclass
class Silence:
    key_prefix: str
    until: float
    reason: str


@dataclass
class AlertManager:
    dedupe_window_sec: float = 300.0
    clock: Callable[[], float] = time.time
    notifier: Callable[[Alert], None] | None = None
    sent: list[Alert] = field(default_factory=list)
    silences: list[Silence] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _last: dict[str, Alert] = field(default_factory=dict)

    def silence(self, key_prefix: str, duration_sec: float, reason: str) -> None:
        """静默（抑制）：计划内维护、已知问题处理中；P0 不受静默影响"""
        self.silences.append(Silence(key_prefix, self.clock() + duration_sec, reason))

    def fire(self, level: str, key: str, title: str, detail: str = "") -> str:
        if level not in LEVELS:
            raise ValueError(f"未知告警级别：{level}")
        now = self.clock()
        if level != "P0" and any(key.startswith(s.key_prefix) and now < s.until for s in self.silences):
            self.stats["silenced"] += 1
            return "silenced"
        last = self._last.get(key)
        if last is not None and now - last.ts < self.dedupe_window_sec:
            last.repeats += 1
            self.stats["deduped"] += 1
            return "deduped"
        alert = Alert(level, key, title, detail, now)
        self._last[key] = alert
        self.sent.append(alert)
        self.stats["sent"] += 1
        if self.notifier:
            self.notifier(alert)
        return "sent"


class JsonFormatter(logging.Formatter):
    """一行一个 JSON；通过 extra={"ctx": {...}} 附加结构化字段"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "ctx", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_json_logger(name: str, path: str | Path | None = None, stream=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    handlers: list[logging.Handler] = []
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w", encoding="utf-8"))
    if stream is not None:
        handlers.append(logging.StreamHandler(stream))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    return logger
