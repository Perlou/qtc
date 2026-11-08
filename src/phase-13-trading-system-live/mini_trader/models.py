"""
统一数据模型（第 01 课）

网关把各家接口的字段都翻译成这里的模型，上层只认这一套结构。
这相当于后端里的 DTO / 防腐层（Anti-Corruption Layer）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(Enum):
    LONG = "多"
    SHORT = "空"


class Offset(Enum):
    """开平标志：期货需要；股票、现货用 NONE"""

    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSE_TODAY = "平今"
    CLOSE_YESTERDAY = "平昨"


class Status(Enum):
    SUBMITTING = "提交中"
    NOT_TRADED = "未成交"  # 即"已报"：交易所已确认，正在挂单
    PART_TRADED = "部分成交"
    ALL_TRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"


class Market(Enum):
    STOCK = "A股"
    FUTURES = "期货"
    CRYPTO = "加密现货"


TERMINAL_STATUSES = frozenset({Status.ALL_TRADED, Status.CANCELLED, Status.REJECTED})

_EXCHANGE_TO_MARKET = {
    "SSE": Market.STOCK,
    "SZSE": Market.STOCK,
    "BSE": Market.STOCK,
    "SHFE": Market.FUTURES,
    "INE": Market.FUTURES,
    "DCE": Market.FUTURES,
    "CZCE": Market.FUTURES,
    "CFFEX": Market.FUTURES,
    "GFEX": Market.FUTURES,
    "BINANCE": Market.CRYPTO,
    "OKX": Market.CRYPTO,
}


def split_symbol(symbol: str) -> tuple[str, str]:
    """把内部统一代码拆成 (代码, 交易所)，如 rb2501.SHFE -> (rb2501, SHFE)"""
    if "." not in symbol:
        raise ValueError(f"内部代码必须是 <代码>.<交易所> 格式：{symbol}")
    code, exchange = symbol.rsplit(".", 1)
    return code, exchange


def market_of(symbol: str) -> Market:
    _, exchange = split_symbol(symbol)
    if exchange not in _EXCHANGE_TO_MARKET:
        raise ValueError(f"未知交易所：{exchange}")
    return _EXCHANGE_TO_MARKET[exchange]


def opposite(direction: Direction) -> Direction:
    return Direction.SHORT if direction is Direction.LONG else Direction.LONG


# ==================== 行情 ====================


@dataclass
class TickData:
    symbol: str  # 统一格式：rb2501.SHFE、600000.SSE、BTC/USDT.BINANCE
    dt: datetime
    last_price: float
    bid_price_1: float = 0.0
    ask_price_1: float = 0.0
    bid_volume_1: float = 0.0
    ask_volume_1: float = 0.0
    limit_up: float = 0.0  # 0 表示该市场没有涨跌停
    limit_down: float = 0.0


@dataclass
class BarData:
    symbol: str
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    interval: str = "1m"


# ==================== 交易 ====================


@dataclass
class OrderRequest:
    symbol: str
    direction: Direction
    price: float
    volume: float
    offset: Offset = Offset.NONE
    client_order_id: str = ""  # 本地生成、全局唯一，用于幂等
    strategy: str = ""


@dataclass
class OrderData:
    client_order_id: str
    symbol: str
    direction: Direction
    price: float
    volume: float
    offset: Offset = Offset.NONE
    traded: float = 0.0
    status: Status = Status.SUBMITTING
    exchange_order_id: str = ""
    reason: str = ""  # 拒单原因等
    strategy: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_active(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    @property
    def remaining(self) -> float:
        return max(self.volume - self.traded, 0.0)


@dataclass
class TradeData:
    trade_id: str  # 交易所成交编号，用于去重
    client_order_id: str
    symbol: str
    direction: Direction
    price: float
    volume: float
    offset: Offset = Offset.NONE
    exchange_order_id: str = ""
    dt: datetime = field(default_factory=datetime.now)


@dataclass
class PositionData:
    symbol: str
    direction: Direction
    volume: float = 0.0  # 总持仓
    yd_volume: float = 0.0  # 昨仓：期货的平昨部分，或 A 股 T+1 下的可卖部分
    frozen: float = 0.0  # 被未完成的卖出 / 平仓委托占用
    price: float = 0.0  # 持仓均价
    realized_pnl: float = 0.0

    @property
    def td_volume(self) -> float:
        return self.volume - self.yd_volume


@dataclass
class AccountData:
    account_id: str
    balance: float  # 资金余额（已含已实现盈亏）
    frozen: float = 0.0  # 挂单冻结
    margin: float = 0.0  # 持仓占用保证金（期货）

    @property
    def available(self) -> float:
        return self.balance - self.frozen - self.margin


# ==================== 序列化：用于事件日志 ====================

_ENUM_FIELDS = {"direction": Direction, "offset": Offset, "status": Status}
_DATETIME_FIELDS = {"dt", "created_at", "updated_at"}


def to_dict(obj: Any) -> dict:
    """dataclass → 可 JSON 序列化的 dict（Enum 存成员名，datetime 存 ISO 字符串）"""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, Enum):
            value = value.name
        elif isinstance(value, datetime):
            value = value.isoformat()
        out[f.name] = value
    return out


def from_dict(cls: type, data: dict) -> Any:
    """to_dict 的逆操作"""
    kwargs = dict(data)
    for name, enum_cls in _ENUM_FIELDS.items():
        if name in kwargs:
            kwargs[name] = enum_cls[kwargs[name]]
    for name in _DATETIME_FIELDS:
        if name in kwargs:
            kwargs[name] = datetime.fromisoformat(kwargs[name])
    return cls(**kwargs)
