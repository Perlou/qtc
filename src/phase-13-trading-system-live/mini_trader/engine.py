"""
asyncio 事件引擎（第 02 课）

单线程事件循环 + 一个队列：所有状态修改都发生在同一线程，
省去了锁，也让事件有了确定的处理顺序。
"""

from __future__ import annotations

import asyncio
import inspect
import time
import traceback
from collections import defaultdict, deque
from typing import Any, Callable

from .events import EVENT_ERROR, EVENT_TIMER, Event

Handler = Callable[[Event], Any]  # 普通函数或 async 函数都可以


class EventEngine:
    def __init__(self, timer_interval: float | None = 1.0):
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._general_handlers: list[Handler] = []  # 接收所有事件，如事件日志
        self._timer_interval = timer_interval
        self._tasks: list[asyncio.Task] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self.processed = 0
        self.errors: list[dict] = []
        self.max_queue_size = 0
        self.wait_times: deque[float] = deque(maxlen=10_000)  # 事件排队耗时（秒）

    def register(self, event_type: str, handler: Handler) -> None:
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unregister(self, event_type: str, handler: Handler) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def register_general(self, handler: Handler) -> None:
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def put(self, event_type: str, data: Any = None) -> None:
        """在事件循环线程内投递"""
        self._queue.put_nowait(Event(event_type, data))
        self.max_queue_size = max(self.max_queue_size, self._queue.qsize())

    def put_threadsafe(self, event_type: str, data: Any = None) -> None:
        """在其他线程投递，例如柜台 C++ API 的回调线程"""
        if self._loop is None:
            raise RuntimeError("事件引擎尚未启动")
        self._loop.call_soon_threadsafe(self.put, event_type, data)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _dispatch(self, event: Event) -> None:
        self.wait_times.append(time.perf_counter() - event.ts)
        for handler in [*self._general_handlers, *self._handlers.get(event.type, [])]:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # 隔离：一个处理器出错不影响其他处理器
                self._on_handler_error(event, handler, exc)

    def _on_handler_error(self, event: Event, handler: Handler, exc: Exception) -> None:
        info = {
            "event_type": event.type,
            "handler": getattr(handler, "__qualname__", repr(handler)),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        self.errors.append(info)
        if event.type != EVENT_ERROR:  # 错误处理器本身出错时不再投递，避免无限循环
            self.put(EVENT_ERROR, info)

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._dispatch(event)
            finally:
                self.processed += 1
                self._queue.task_done()

    async def _timer(self) -> None:
        while True:
            await asyncio.sleep(self._timer_interval)
            self.put(EVENT_TIMER)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._tasks = [asyncio.create_task(self._run(), name="event-loop")]
        if self._timer_interval:
            self._tasks.append(asyncio.create_task(self._timer(), name="event-timer"))

    async def join(self) -> None:
        """等待队列中已有事件全部处理完"""
        await self._queue.join()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
