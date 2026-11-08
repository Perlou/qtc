"""
持仓簿：只由成交（Trade）累计，不由订单状态推导（第 03 课）

- A 股 / 现货：单向持仓，卖出只减少多头；A 股 T+1 下今日买入不可卖
- 期货：多空分别记录，并区分今仓与昨仓（第 06 课）

OMS（本地视图）和 MockExchange（"真相"）共用这段记账逻辑。
"""

from __future__ import annotations

from .contracts import get_spec
from .models import (
    Direction,
    Market,
    Offset,
    OrderRequest,
    PositionData,
    TradeData,
    opposite,
)


def position_side(symbol: str, direction: Direction, offset: Offset) -> tuple[Direction, bool]:
    """返回 (受影响的持仓方向, 是否减仓)"""
    spec = get_spec(symbol)
    if spec.market is Market.FUTURES:
        if offset is Offset.OPEN:
            return direction, False
        return opposite(direction), True
    return Direction.LONG, direction is Direction.SHORT


def is_closing(req: OrderRequest) -> bool:
    return position_side(req.symbol, req.direction, req.offset)[1]


class PositionBook:
    def __init__(self):
        self._positions: dict[tuple[str, Direction], PositionData] = {}

    def get(self, symbol: str, direction: Direction = Direction.LONG) -> PositionData:
        key = (symbol, direction)
        if key not in self._positions:
            self._positions[key] = PositionData(symbol, direction)
        return self._positions[key]

    def all(self) -> list[PositionData]:
        return [p for p in self._positions.values() if p.volume != 0 or p.frozen != 0]

    def to_volume_map(self) -> dict[str, float]:
        """用于对账：{"rb2501.SHFE|LONG": 3.0, ...}"""
        return {f"{p.symbol}|{p.direction.name}": p.volume for p in self.all() if p.volume != 0}

    def load(self, positions: list[PositionData]) -> None:
        """用交易所查询结果覆盖本地持仓（恢复时使用）"""
        self._positions = {
            (p.symbol, p.direction): PositionData(
                p.symbol, p.direction, p.volume, p.yd_volume, p.frozen, p.price, p.realized_pnl
            )
            for p in positions
        }

    def available(self, symbol: str, direction: Direction = Direction.LONG) -> float:
        """A 股 / 现货的可卖数量"""
        p = self.get(symbol, direction)
        return p.yd_volume - p.frozen

    def closable(self, symbol: str, pos_direction: Direction, offset: Offset) -> float:
        """
        期货可平数量。frozen 没有按今 / 昨拆分，这里保守地用总量上限约束；
        上期所 / 能源中心的 CLOSE 视为平昨（简化规则，以交易所和柜台实际行为为准）。
        """
        p = self.get(symbol, pos_direction)
        spec = get_spec(symbol)
        free = p.volume - p.frozen
        if spec.explicit_close and offset is Offset.CLOSE_TODAY:
            return min(p.td_volume, free)
        if spec.explicit_close:
            return min(p.yd_volume, free)
        return free

    def freeze(self, symbol: str, direction: Direction, volume: float) -> None:
        self.get(symbol, direction).frozen += volume

    def unfreeze(self, symbol: str, direction: Direction, volume: float) -> None:
        p = self.get(symbol, direction)
        p.frozen = max(p.frozen - volume, 0.0)

    def on_trade(self, trade: TradeData) -> float:
        """按一笔成交更新持仓，返回这笔成交实现的盈亏（冻结由调用方按订单释放）"""
        spec = get_spec(trade.symbol)
        pos_direction, closing = position_side(trade.symbol, trade.direction, trade.offset)
        p = self.get(trade.symbol, pos_direction)

        if not closing:
            total = p.volume + trade.volume
            p.price = (p.volume * p.price + trade.volume * trade.price) / total
            p.volume = total
            if not spec.t_plus_one and spec.market is not Market.FUTURES:
                p.yd_volume += trade.volume  # T+0 现货：买入即可卖
            return 0.0

        if spec.explicit_close and trade.offset is Offset.CLOSE_TODAY:
            from_yd = 0.0
        elif spec.explicit_close:
            from_yd = trade.volume
        else:
            from_yd = min(trade.volume, p.yd_volume)  # 先平昨，再平今
        sign = 1.0 if pos_direction is Direction.LONG else -1.0
        pnl = sign * (trade.price - p.price) * trade.volume * spec.size
        p.volume -= trade.volume
        p.yd_volume = max(p.yd_volume - from_yd, 0.0)
        p.realized_pnl += pnl
        if p.volume == 0:
            p.price = 0.0
        return pnl

    def settle(self) -> None:
        """换日：今仓转昨仓，挂单失效后冻结清零"""
        for p in self._positions.values():
            p.yd_volume = p.volume
            p.frozen = 0.0
