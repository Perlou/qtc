"""
合约规格：乘数、最小变动价位、保证金比例、T+1 等

这些参数决定下单数量、资金占用和开平规则。这里的数值只用于离线演示，
实盘必须从交易所、柜台或 load_markets() 获取，不能硬编码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Market, market_of, split_symbol

# 常见期货品种的合约乘数（演示用，以交易所合约规则为准）
_FUTURES_MULTIPLIER = {
    "rb": 10,
    "cu": 5,
    "au": 1000,
    "sc": 1000,
    "i": 100,
    "m": 10,
    "IF": 300,
}
_EXPLICIT_CLOSE_EXCHANGES = {"SHFE", "INE"}  # 平今 / 平昨需要显式指定
DEMO_MARGIN_RATIO = 0.12  # 示意值：实际比例由交易所和期货公司设定


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    market: Market
    size: float = 1.0  # 合约乘数
    price_tick: float = 0.01
    lot_size: float = 1.0  # 买入数量的整数倍要求
    margin_ratio: float = 1.0  # 1.0 表示全额资金
    t_plus_one: bool = False
    explicit_close: bool = False  # True：区分平今 / 平昨，CLOSE 视为平昨


_registry: dict[str, ContractSpec] = {}


def register_spec(spec: ContractSpec) -> None:
    _registry[spec.symbol] = spec


def get_spec(symbol: str) -> ContractSpec:
    if symbol in _registry:
        return _registry[symbol]
    market = market_of(symbol)
    code, exchange = split_symbol(symbol)
    if market is Market.STOCK:
        return ContractSpec(symbol, market, lot_size=100, t_plus_one=True)
    if market is Market.FUTURES:
        product = re.match(r"[A-Za-z]+", code).group()
        return ContractSpec(
            symbol,
            market,
            size=_FUTURES_MULTIPLIER.get(product, 10),
            price_tick=1.0,
            margin_ratio=DEMO_MARGIN_RATIO,
            explicit_close=exchange in _EXPLICIT_CLOSE_EXCHANGES,
        )
    return ContractSpec(symbol, market, lot_size=0.0001)


def required_funds(symbol: str, price: float, volume: float) -> float:
    """买入（全额）或开仓（保证金）需要的资金"""
    spec = get_spec(symbol)
    return price * volume * spec.size * spec.margin_ratio
