"""
交易网关（第 01 课）

网关只做"翻译"：把柜台 / 交易所的接口和回报转换成统一模型并推送事件，
不包含任何业务逻辑。真实接口（CTP、QMT、ccxt）各写一个子类即可。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .engine import EventEngine
from .events import EVENT_CANCEL_REJECT, EVENT_GATEWAY, EVENT_ORDER, EVENT_TICK, EVENT_TRADE
from .mock_exchange import MockExchange
from .models import AccountData, OrderData, OrderRequest, PositionData, TickData, TradeData


class BaseGateway(ABC):
    """所有交易接口的统一抽象"""

    def __init__(self, event_engine: EventEngine, name: str):
        self.event_engine = event_engine
        self.name = name

    @abstractmethod
    async def connect(self, setting: dict | None = None) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def subscribe(self, symbol: str) -> None: ...

    @abstractmethod
    async def send_order(self, req: OrderRequest) -> OrderData: ...

    @abstractmethod
    async def cancel_order(self, client_order_id: str) -> bool: ...

    @abstractmethod
    async def query_order(self, client_order_id: str) -> OrderData | None: ...

    @abstractmethod
    async def query_orders(self) -> list[OrderData]: ...

    @abstractmethod
    async def query_trades(self, client_order_id: str | None = None) -> list[TradeData]: ...

    @abstractmethod
    async def query_positions(self) -> list[PositionData]: ...

    @abstractmethod
    async def query_account(self) -> AccountData: ...

    # 收到柜台回报后，转换为统一模型并推送事件
    def on_tick(self, tick: TickData) -> None:
        self.event_engine.put(EVENT_TICK, tick)

    def on_order(self, order: OrderData) -> None:
        self.event_engine.put(EVENT_ORDER, order)

    def on_trade(self, trade: TradeData) -> None:
        self.event_engine.put(EVENT_TRADE, trade)


class MockGateway(BaseGateway):
    """
    连接 MockExchange 的网关，可模拟网络延迟：
        request_latency   请求从本地到交易所的耗时（超时发生在这段 → 交易所没收到）
        response_latency  响应从交易所回到本地的耗时（超时发生在这段 → 交易所已处理）
        report_latency    主动推送回报（订单 / 成交）的延迟
    """

    _KIND_TO_EVENT = {
        "tick": EVENT_TICK,
        "order": EVENT_ORDER,
        "trade": EVENT_TRADE,
        "cancel_reject": EVENT_CANCEL_REJECT,
    }

    def __init__(
        self,
        event_engine: EventEngine,
        exchange: MockExchange,
        name: str = "MOCK",
        request_latency: float = 0.0,
        response_latency: float = 0.0,
        report_latency: float = 0.0,
    ):
        super().__init__(event_engine, name)
        self.exchange = exchange
        self.request_latency = request_latency
        self.response_latency = response_latency
        self.report_latency = report_latency
        self.subscribed: set[str] = set()
        self.connected = False

    async def connect(self, setting: dict | None = None) -> None:
        self.exchange.add_listener(self._on_exchange)
        self.connected = True
        self.event_engine.put(EVENT_GATEWAY, {"gateway": self.name, "status": "connected"})

    async def close(self) -> None:
        self.exchange.remove_listener(self._on_exchange)
        self.connected = False

    async def subscribe(self, symbol: str) -> None:
        self.subscribed.add(symbol)

    def _on_exchange(self, kind: str, data: Any) -> None:
        if kind == "tick" and data.symbol not in self.subscribed:
            return
        event_type = self._KIND_TO_EVENT[kind]
        if self.report_latency > 0:
            asyncio.get_running_loop().call_later(self.report_latency, self.event_engine.put, event_type, data)
        else:
            self.event_engine.put(event_type, data)

    async def send_order(self, req: OrderRequest) -> OrderData:
        await asyncio.sleep(self.request_latency)  # 请求在途
        ack = self.exchange.submit(req)
        await asyncio.sleep(self.response_latency)  # 响应在途
        return ack

    async def cancel_order(self, client_order_id: str) -> bool:
        await asyncio.sleep(self.request_latency)
        return self.exchange.cancel(client_order_id)

    async def query_order(self, client_order_id: str) -> OrderData | None:
        await asyncio.sleep(self.request_latency)
        return self.exchange.query_order(client_order_id)

    async def query_orders(self) -> list[OrderData]:
        await asyncio.sleep(self.request_latency)
        return self.exchange.query_orders()

    async def query_trades(self, client_order_id: str | None = None) -> list[TradeData]:
        await asyncio.sleep(self.request_latency)
        return self.exchange.query_trades(client_order_id)

    async def query_positions(self) -> list[PositionData]:
        await asyncio.sleep(self.request_latency)
        return self.exchange.query_positions()

    async def query_account(self) -> AccountData:
        await asyncio.sleep(self.request_latency)
        return self.exchange.query_account()
