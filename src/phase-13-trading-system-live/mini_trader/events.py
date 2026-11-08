"""事件类型定义（第 02 课）：行情、订单、成交、定时器、错误都是事件"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

EVENT_TICK = "tick"
EVENT_BAR = "bar"
EVENT_ORDER = "order"
EVENT_TRADE = "trade"
EVENT_CANCEL_REJECT = "cancel_reject"
EVENT_TIMER = "timer"
EVENT_ERROR = "error"
EVENT_SIGNAL = "signal"
EVENT_GATEWAY = "gateway"  # 连接状态变化


@dataclass
class Event:
    type: str
    data: Any = None
    ts: float = field(default_factory=time.perf_counter)  # 入队时刻，用于统计排队延迟
