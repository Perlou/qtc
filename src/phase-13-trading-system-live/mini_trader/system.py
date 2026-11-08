"""
把各组件装配成一个可运行的迷你交易系统

    MockExchange ⇄ MockGateway → EventEngine → (RiskEngine, OMS, 策略)
                                         ↑
    策略意图 → TradingSystem.send_order → 风控 → OMS → 网关（超时先查后发）
"""

from __future__ import annotations

import asyncio
from typing import Callable

from .engine import EventEngine
from .events import EVENT_CANCEL_REJECT, EVENT_ORDER, EVENT_TICK, EVENT_TIMER, EVENT_TRADE, Event
from .gateway import MockGateway
from .mock_exchange import MockExchange
from .models import AccountData, Direction, Offset, OrderRequest
from .oms import OMS, OrderIdGenerator
from .persistence import record
from .recovery import SendResult, send_with_confirm
from .risk import RiskConfig, RiskEngine


class TradingSystem:
    def __init__(
        self,
        exchange: MockExchange,
        risk_config: RiskConfig | None = None,
        *,
        prefix: str = "S1",
        timer_interval: float | None = 0.5,
        order_timeout: float = 1.0,
        gateway_kwargs: dict | None = None,
        event_log=None,
        log: Callable[[str], None] | None = None,
    ):
        self.engine = EventEngine(timer_interval)
        self.gateway = MockGateway(self.engine, exchange, **(gateway_kwargs or {}))
        self.risk = RiskEngine(risk_config)
        self.oms = OMS()
        self.ids = OrderIdGenerator(prefix)
        self.account: AccountData | None = None
        self.order_timeout = order_timeout
        self.event_log = event_log
        self.log = log or (lambda msg: None)
        self._tasks: set[asyncio.Task] = set()

        self.risk.halt_listeners.append(self._on_halt)
        self.engine.register(EVENT_TICK, self._on_tick)
        self.engine.register(EVENT_ORDER, self._on_order)
        self.engine.register(EVENT_TRADE, self._on_trade)
        self.engine.register(EVENT_CANCEL_REJECT, self._on_cancel_reject)
        self.engine.register(EVENT_TIMER, self._on_timer)

    # ---------- 事件处理 ----------

    def _on_tick(self, event: Event) -> None:
        self.risk.update_tick(event.data)

    def _on_order(self, event: Event) -> None:
        if self.event_log is not None:
            record(self.event_log, "order", event.data)
        self.oms.on_order(event.data)
        self.risk.on_order_update(event.data)

    def _on_trade(self, event: Event) -> None:
        if self.event_log is not None:
            record(self.event_log, "trade", event.data)
        self.oms.on_trade(event.data)

    def _on_cancel_reject(self, event: Event) -> None:
        self.oms.on_cancel_reject(event.data["client_order_id"], event.data["reason"])

    async def _on_timer(self, event: Event) -> None:
        try:
            self.account = await self.gateway.query_account()
        except ConnectionError:
            pass  # 断线期间沿用上一次的资金快照，由监控负责告警

    def _on_halt(self, reason: str) -> None:
        self.log(f"🛑 Kill Switch 触发：{reason} → 撤销全部挂单")
        task = asyncio.get_running_loop().create_task(self.cancel_all())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self.engine.start()
        await self.gateway.connect()
        self.account = await self.gateway.query_account()

    async def stop(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.engine.join()
        await self.engine.stop()
        await self.gateway.close()

    # ---------- 交易 ----------

    def new_request(
        self, symbol: str, direction: Direction, price: float, volume: float, offset: Offset = Offset.NONE
    ) -> OrderRequest:
        return OrderRequest(symbol, direction, price, volume, offset, self.ids.next())

    async def send_order(self, req: OrderRequest) -> SendResult:
        if not req.client_order_id:
            req.client_order_id = self.ids.next()
        check = self.risk.check_order(req, self.account, self.oms.positions)
        if not check.ok:
            return SendResult(req.client_order_id, "risk_rejected", 0, f"[{check.rule}] {check.reason}")
        self.oms.create_order(req)
        if self.event_log is not None:  # 先写日志，再发请求（Write-Ahead）
            record(self.event_log, "order_request", req)
        result = await send_with_confirm(self.gateway, self.oms, req, self.order_timeout, log=self.log)
        if result.outcome == "unknown":  # 不确定时不再往这个标的下单
            self.risk.halt_symbol(req.symbol, f"订单 {req.client_order_id} 状态未知")
        return result

    async def cancel_all(self) -> int:
        count = 0
        for order in self.oms.active_orders():
            if self.oms.request_cancel(order.client_order_id):
                try:
                    count += await self.gateway.cancel_order(order.client_order_id)
                except Exception as exc:  # 撤单失败不能让撤单流程中断
                    self.log(f"撤单失败 {order.client_order_id}：{exc!r}")
        return count
