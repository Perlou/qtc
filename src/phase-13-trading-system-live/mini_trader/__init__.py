"""
mini_trader：Phase 13 课程用的迷你交易系统骨架

模块与课程的对应关系：
    models / contracts / gateway        01 交易系统架构
    events / engine                     02 asyncio 事件引擎
    oms / positions                     03 OMS 与订单状态机
    risk                                04 风控引擎
    mock_exchange                       全程使用的模拟交易所（可注入故障）
    sessions                            06 夜盘交易日归属、11 按交易日历调度
    monitor                             09 监控与告警
    persistence / recovery              10 对账、幂等与故障恢复
    deploy_templates                    11 生成的调度与健康检查脚本模板
    system                              把以上组件装配在一起
"""

from .contracts import ContractSpec, get_spec, register_spec, required_funds
from .engine import EventEngine
from .events import (
    EVENT_CANCEL_REJECT,
    EVENT_ERROR,
    EVENT_GATEWAY,
    EVENT_ORDER,
    EVENT_SIGNAL,
    EVENT_TICK,
    EVENT_TIMER,
    EVENT_TRADE,
    Event,
)
from .gateway import BaseGateway, MockGateway
from .mock_exchange import (
    DuplicateOrderError,
    ExchangeDisconnected,
    ExchangeError,
    ExchangeTimeout,
    MockExchange,
)
from .models import (
    TERMINAL_STATUSES,
    AccountData,
    BarData,
    Direction,
    Market,
    Offset,
    OrderData,
    OrderRequest,
    PositionData,
    Status,
    TickData,
    TradeData,
    from_dict,
    market_of,
    to_dict,
)
from .oms import ALLOWED_TRANSITIONS, OMS, InvalidTransition, OrderIdGenerator, OverfillError, check_transition
from .positions import PositionBook, position_side
from .risk import RiskConfig, RiskEngine, RiskResult

__all__ = [name for name in dir() if not name.startswith("_")]
