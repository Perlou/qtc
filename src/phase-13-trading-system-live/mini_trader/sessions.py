"""
交易日历与交易时段（第 06、11 课）

- 调度必须基于交易日历，而不是"周一到周五"
- 期货夜盘属于下一个交易日：周五夜盘归属下周一（或节后第一个交易日）
"""

from __future__ import annotations

import bisect
from datetime import date, datetime, time, timedelta
from typing import Iterable

A_SHARE_SESSIONS = [(time(9, 15), time(11, 30)), (time(13, 0), time(15, 0))]  # 含集合竞价
# 期货日盘示意（以 rb 为例）；各品种时段不同，以交易所公布为准
FUTURES_DAY_SESSIONS = [(time(9, 0), time(10, 15)), (time(10, 30), time(11, 30)), (time(13, 30), time(15, 0))]
FUTURES_NIGHT_SESSION = (time(21, 0), time(23, 0))  # 示意：不同品种夜盘结束时间不同


class TradingCalendar:
    def __init__(self, days: Iterable[date]):
        self.days = sorted(set(days))
        self._set = set(self.days)

    def is_trading_day(self, d: date) -> bool:
        return d in self._set

    def next_trading_day(self, d: date) -> date | None:
        """严格晚于 d 的第一个交易日"""
        i = bisect.bisect_right(self.days, d)
        return self.days[i] if i < len(self.days) else None

    def prev_trading_day(self, d: date) -> date | None:
        i = bisect.bisect_left(self.days, d)
        return self.days[i - 1] if i > 0 else None

    def trading_day_of(self, dt: datetime, night_start: time = time(20, 0), night_end: time = time(3, 0)) -> date | None:
        """
        期货交易日归属：
            夜盘（night_start 之后）→ 下一个交易日
            凌晨（night_end 之前，属于前一晚夜盘的延续）→ 前一自然日之后的下一个交易日
            日盘 → 当天（当天不是交易日则返回 None）
        简化假设：节假日前一晚没有夜盘的情况由调用方根据交易所通知处理。
        """
        t, d = dt.time(), dt.date()
        if t >= night_start:
            return self.next_trading_day(d)
        if t < night_end:
            return self.next_trading_day(d - timedelta(days=1))
        return d if self.is_trading_day(d) else None


def in_sessions(t: time, sessions: list[tuple[time, time]]) -> bool:
    return any(start <= t <= end for start, end in sessions)


def in_a_share_session(dt: datetime, calendar: TradingCalendar) -> bool:
    return calendar.is_trading_day(dt.date()) and in_sessions(dt.time(), A_SHARE_SESSIONS)
