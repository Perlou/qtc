"""
Phase 11 公共模块（orderbook）
==============================

01 课会把撮合引擎"从零讲一遍"，之后各课直接 `from orderbook import ...` 复用。

内容：
    - MatchingEngine：价格优先、时间优先的最小撮合引擎
        * 限价单（GTC / IOC / FOK）、市价单（可限制最多吃几档，如"最优五档即时成交剩余撤销"）
        * 部分成交、撤单、成交价取被动方价格
        * 价格用整数 tick 存储，避免浮点数做字典 key 的精度问题
    - simulate_zi_market：零智能（Zero-Intelligence）订单流模拟器
        * 随机到达的限价单 / 市价单 / 撤单驱动撮合引擎
        * 输出"逐笔成交"和"3 秒十档快照"，用来代替付费的 Level-2 数据（02、03 课）
    - simulate_l1_queue：一档队列模型（买一/卖一队列被吃光时价格跳动）
        * 输出 L1 快照序列，用于 OBI / OFI / 微价格研究（05 课）
    - synthetic_depth_snapshots：格式与 ccxt 相同的合成深度快照（04 课离线回退）
    - intraday_volume_profile：A 股日内 U 型成交量分布（05、06 课）

⚠️ 所有模拟器只用于教学：它们能复现微观结构的"定性规律"，但参数不是从真实市场校准的。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

BUY, SELL = "B", "S"


# ==================== 撮合引擎 ====================


@dataclass
class Order:
    oid: int
    side: str  # "B" 买 / "S" 卖
    qty: int  # 原始申报数量
    price: int | None  # 以 tick 为单位的整数价格；市价单为 None
    ts: float = 0.0
    filled: int = 0
    status: str = "NEW"  # NEW / PARTIAL / FILLED / CANCELED / PARTIAL_CANCELED（部分成交后剩余撤销）/ REJECTED

    @property
    def remaining(self) -> int:
        return self.qty - self.filled


@dataclass(frozen=True)
class Trade:
    ts: float
    price: int  # tick 整数价格（= 被动方价格）
    qty: int
    aggressor: str  # 主动方方向 "B" / "S"
    maker_oid: int  # 被动方（挂单方）订单号
    taker_oid: int  # 主动方（吃单方）订单号


class MatchingEngine:
    """
    最小撮合引擎：每个价位一个 FIFO 队列（deque），价格优先 + 时间优先。

    教学取舍：最优价用 max()/min() 查找（O(价位数)）；真实引擎用红黑树、跳表
    或按 tick 索引的数组做到 O(1) / O(log n)。
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick = tick_size
        self.bids: dict[int, deque[Order]] = {}
        self.asks: dict[int, deque[Order]] = {}
        self.orders: dict[int, Order] = {}  # 当前挂在簿上的订单
        self.trades: list[Trade] = []
        self._next_oid = 1

    # ---------- 价格换算 ----------
    def to_ticks(self, price: float) -> int:
        return int(round(price / self.tick))

    def to_price(self, ticks: int) -> float:
        return round(ticks * self.tick, 6)

    # ---------- 盘口查询 ----------
    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2 * self.tick

    def levels(self, side: str, n: int = 5) -> list[tuple[float, int, int]]:
        """返回某一侧前 n 档 [(价格, 总量, 订单笔数), ...]"""
        book = self.bids if side == BUY else self.asks
        prices = sorted(book, reverse=(side == BUY))[:n]
        return [(self.to_price(p), sum(o.remaining for o in book[p]), len(book[p])) for p in prices]

    # ---------- 下单 ----------
    def _new_order(self, side: str, qty: int, price: int | None, ts: float) -> Order:
        if side not in (BUY, SELL):
            raise ValueError(f"side 必须是 'B' 或 'S'，收到 {side!r}")
        order = Order(self._next_oid, side, int(qty), price, ts)
        self._next_oid += 1
        return order

    def _crosses(self, side: str, opp_price: int, limit: int | None) -> bool:
        if limit is None:
            return True
        return opp_price <= limit if side == BUY else opp_price >= limit

    def _available(self, side: str, limit: int | None) -> int:
        """在限价 limit 以内，对手方可成交的总量（FOK 预检用）"""
        opp = self.asks if side == BUY else self.bids
        return sum(
            o.remaining for p, q in opp.items() if self._crosses(side, p, limit) for o in q
        )

    def _match(self, order: Order, limit: int | None) -> list[Trade]:
        opp = self.asks if order.side == BUY else self.bids
        trades = []
        while order.remaining > 0 and opp:
            best = min(opp) if order.side == BUY else max(opp)
            if not self._crosses(order.side, best, limit):
                break
            queue = opp[best]
            resting = queue[0]  # 时间优先：队首先成交
            fill = min(order.remaining, resting.remaining)
            t = Trade(order.ts, best, fill, order.side, resting.oid, order.oid)  # 成交价 = 被动方价格
            trades.append(t)
            order.filled += fill
            resting.filled += fill
            resting.status = "FILLED" if resting.remaining == 0 else "PARTIAL"
            if resting.remaining == 0:
                queue.popleft()
                self.orders.pop(resting.oid, None)
                if not queue:
                    del opp[best]
        self.trades.extend(trades)
        return trades

    def submit_limit(self, side: str, price: float, qty: int, tif: str = "GTC", ts: float = 0.0):
        """
        限价单。tif（Time in Force）：
            GTC 未成交部分挂单；IOC 能成交多少成交多少、剩余撤销；FOK 不能全部成交则全部撤销
        返回 (order, trades)
        """
        order = self._new_order(side, qty, self.to_ticks(price), ts)
        if tif == "FOK" and self._available(side, order.price) < order.qty:
            order.status = "REJECTED"
            return order, []
        trades = self._match(order, order.price)
        if order.remaining > 0 and tif == "GTC":
            mine = self.bids if side == BUY else self.asks
            mine.setdefault(order.price, deque()).append(order)  # 排到同价位队尾
            self.orders[order.oid] = order
            order.status = "PARTIAL" if order.filled else "NEW"
        elif order.remaining > 0:
            order.status = "CANCELED" if order.filled == 0 else "PARTIAL_CANCELED"  # IOC 剩余撤销
        else:
            order.status = "FILLED"
        return order, trades

    def submit_market(self, side: str, qty: int, max_levels: int | None = None, ts: float = 0.0):
        """
        市价单：按对手方最优价逐档成交，未成交部分撤销（不挂单）。
        max_levels=5 近似 A 股"最优五档即时成交剩余撤销"。返回 (order, trades)
        """
        order = self._new_order(side, qty, None, ts)
        limit = None
        if max_levels is not None:
            opp = self.asks if side == BUY else self.bids
            prices = sorted(opp, reverse=(side == SELL))[:max_levels]
            limit = prices[-1] if prices else None
            if limit is None:
                order.status = "CANCELED"
                return order, []
        trades = self._match(order, limit)
        order.status = "FILLED" if order.remaining == 0 else ("PARTIAL_CANCELED" if order.filled else "CANCELED")
        return order, trades

    def cancel(self, oid: int) -> bool:
        """撤单：从所在价位队列中移除，返回是否成功"""
        order = self.orders.pop(oid, None)
        if order is None:
            return False
        book = self.bids if order.side == BUY else self.asks
        queue = book[order.price]
        queue.remove(order)
        if not queue:
            del book[order.price]
        order.status = "CANCELED"
        return True

    # ---------- 展示 ----------
    def depth(self, n: int = 5) -> pd.DataFrame:
        bids, asks = self.levels(BUY, n), self.levels(SELL, n)
        rows = []
        for i in range(n):
            b = bids[i] if i < len(bids) else (np.nan, 0, 0)
            a = asks[i] if i < len(asks) else (np.nan, 0, 0)
            rows.append([i + 1, b[2], b[1], b[0], a[0], a[1], a[2]])
        return pd.DataFrame(rows, columns=["档位", "买笔数", "买量", "买价", "卖价", "卖量", "卖笔数"])

    def render(self, n: int = 5, indent: str = "    ") -> str:
        """ASCII 订单簿：卖盘在上（价格从高到低），买盘在下"""
        lines = [f"{indent}{'档位':<6}{'价格':>10}{'数量':>10}{'订单队列(先→后)':>22}"]
        for i, (p, _, _) in reversed(list(enumerate(self.levels(SELL, n)))):
            q = [o.remaining for o in self.asks[self.to_ticks(p)]]
            lines.append(f"{indent}卖{i + 1:<5}{p:>10.2f}{sum(q):>10}    {q}")
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            lines.append(f"{indent}{'-' * 18} 价差 {self.to_price(ba - bb):.2f}  中间价 {self.mid():.3f} {'-' * 8}")
        else:
            lines.append(f"{indent}{'-' * 18} （一侧为空，价差无定义）{'-' * 8}")
        for i, (p, _, _) in enumerate(self.levels(BUY, n)):
            q = [o.remaining for o in self.bids[self.to_ticks(p)]]
            lines.append(f"{indent}买{i + 1:<5}{p:>10.2f}{sum(q):>10}    {q}")
        return "\n".join(lines)


def trades_to_frame(engine: MatchingEngine, trades: list[Trade] | None = None) -> pd.DataFrame:
    trades = engine.trades if trades is None else trades
    return pd.DataFrame(
        [(t.ts, engine.to_price(t.price), t.qty, t.aggressor, t.maker_oid, t.taker_oid) for t in trades],
        columns=["ts", "price", "qty", "aggressor", "maker_oid", "taker_oid"],
    )


# ==================== 零智能订单流模拟器 ====================


def simulate_zi_market(
    seconds: int = 3600,
    seed: int = 42,
    start_price: float = 10.00,
    tick: float = 0.01,
    limit_rate: float = 3.0,
    market_rate: float = 0.8,
    cancel_rate: float = 0.008,
    snapshot_every: float = 3.0,
    levels: int = 10,
) -> dict:
    """
    零智能（Zero-Intelligence）市场：订单随机到达，驱动 MatchingEngine。

    - 限价单（每秒 limit_rate 笔）：买单价格 = 卖一 − k 个 tick，卖单价格 = 买一 + k 个 tick
      （k ≥ 1，几何分布，越靠近盘口越多）
    - 市价单（每秒 market_rate 笔）：随机方向，最多吃 5 档
    - 撤单：每笔挂单每秒以 cancel_rate 的概率被撤（挂单越多，撤单越多，订单簿规模自然稳定）
    数量都是 100 股的整数倍。事件时间用 Gillespie 方法生成（总速率随挂单数变化）。

    返回 dict：
        trades:    逐笔成交（含真实主动方向 aggressor_true、成交前瞬间的真实中间价 mid_before）
        snapshots: 每 snapshot_every 秒一张十档快照（类似 A 股 Level-2 的 3 秒快照）
        engine:    模拟结束时的撮合引擎
    """
    rng = np.random.default_rng(seed)
    eng = MatchingEngine(tick)
    p0 = eng.to_ticks(start_price)
    lot_choices = np.array([1, 1, 2, 2, 3, 5, 5, 10, 20]) * 100

    # 初始订单簿：两侧各 levels 档
    for k in range(1, levels + 1):
        for _ in range(3):
            eng.submit_limit(BUY, eng.to_price(p0 - k), int(rng.choice(lot_choices)))
            eng.submit_limit(SELL, eng.to_price(p0 + k), int(rng.choice(lot_choices)))

    n_max = int(seconds * (limit_rate + market_rate) * 4) + 1000
    unit_gaps = rng.exponential(1.0, n_max)
    u_kind = rng.random(n_max)
    sides = rng.choice([BUY, SELL], n_max)
    offsets = np.minimum(rng.geometric(0.3, n_max), 15)  # k ≥ 1，近盘口更多
    sizes = rng.choice(lot_choices, n_max)
    mkt_sizes = rng.choice(np.array([1, 2, 3, 5, 10, 20, 30]) * 100, n_max)
    pick = rng.random(n_max)

    trade_rows, snap_rows = [], []
    t, next_snap, cum_vol, last_px = 0.0, 0.0, 0, start_price

    def take_snapshot(ts):
        row = {"time": ts, "last": last_px, "cum_volume": cum_vol}
        for side, name in ((BUY, "bid"), (SELL, "ask")):
            lv = eng.levels(side, levels)
            for j in range(levels):
                px, qty = (lv[j][0], lv[j][1]) if j < len(lv) else (np.nan, 0)
                row[f"{name}{j + 1}"] = px
                row[f"{name}{j + 1}_v"] = qty
        snap_rows.append(row)

    for i in range(n_max):
        cancel_total = cancel_rate * len(eng.orders)
        total = limit_rate + market_rate + cancel_total
        t += unit_gaps[i] / total
        if t > seconds:
            break
        while next_snap <= t:
            take_snapshot(next_snap)
            next_snap += snapshot_every

        bb, ba = eng.best_bid(), eng.best_ask()
        # 某一侧被吃空时补一笔挂单，防止订单簿消失
        if bb is None:
            eng.submit_limit(BUY, eng.to_price((ba or p0) - 1), 500, ts=t)
            continue
        if ba is None:
            eng.submit_limit(SELL, eng.to_price(bb + 1), 500, ts=t)
            continue

        side = str(sides[i])
        u = u_kind[i] * total
        if u < limit_rate:  # 限价单（不主动穿越对手价）
            px = ba - int(offsets[i]) if side == BUY else bb + int(offsets[i])
            eng.submit_limit(side, eng.to_price(px), int(sizes[i]), ts=t)
        elif u < limit_rate + market_rate:  # 市价单
            mid_before = (bb + ba) / 2 * tick
            _, trades = eng.submit_market(side, int(mkt_sizes[i]), max_levels=5, ts=t)
            for tr in trades:
                cum_vol += tr.qty
                last_px = eng.to_price(tr.price)
                trade_rows.append((t, last_px, tr.qty, 1 if side == BUY else -1, mid_before, tr.taker_oid))
        elif eng.orders:  # 撤单：随机挑一笔挂单
            oids = list(eng.orders)
            eng.cancel(oids[int(pick[i] * len(oids))])

    trades = pd.DataFrame(trade_rows, columns=["time", "price", "qty", "aggressor_true", "mid_before", "taker_oid"])
    snapshots = pd.DataFrame(snap_rows)
    return {"trades": trades, "snapshots": snapshots, "engine": eng}


# ==================== 一档队列模型（L1） ====================


def simulate_l1_queue(
    n_events: int = 200_000,
    seed: int = 7,
    snapshot_every: int = 20,
    tick: float = 0.01,
    start_price: float = 10.00,
) -> pd.DataFrame:
    """
    一档队列模型（思想来自 Cont & de Larrard 的排队模型）：
    只跟踪买一、卖一两个队列。事件：限价单加入队列 / 撤单 / 市价单吃队列。
    - 某个队列被吃光（或撤光）→ 该侧价格后退一个 tick，并"露出"下一档的量
    - 价差 > 1 tick 时，新限价单有一定概率挂进价差内部（改善价格）

    价格只在队列耗尽时变化，所以"哪边队列短"天然包含了下一步价格方向的信息 ——
    这正是 OBI / OFI / 微价格能预测短期价格变动的机制。

    返回每 snapshot_every 个事件一张的 L1 快照：bid_px, bid_sz, ask_px, ask_sz（价格单位：元），
    以及区间内主动买/卖成交量 buy_vol / sell_vol。
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.21, 0.21, 0.145, 0.145, 0.145, 0.145])  # 加买、加卖、撤买、撤卖、市价买、市价卖
    kinds = rng.choice(6, n_events, p=probs)
    sizes = rng.choice(np.array([1, 1, 1, 2, 2, 3, 5]) * 100, n_events)
    improve = rng.random(n_events)
    refill_b = rng.integers(5, 40, n_events) * 100  # 队列耗尽后露出的下一档数量
    refill_a = rng.integers(5, 40, n_events) * 100

    bp = int(round(start_price / tick))
    ap = bp + 1
    bq, aq = 2000, 2000
    buy_vol = sell_vol = 0
    rows = []
    for i in range(n_events):
        k, s = kinds[i], int(sizes[i])
        spread = ap - bp
        if k == 0:  # 买方限价单
            if spread > 1 and improve[i] < 0.5:
                bp, bq = bp + 1, s
            else:
                bq += s
        elif k == 1:  # 卖方限价单
            if spread > 1 and improve[i] < 0.5:
                ap, aq = ap - 1, s
            else:
                aq += s
        elif k == 2:  # 撤买
            bq -= s
        elif k == 3:  # 撤卖
            aq -= s
        elif k == 4:  # 市价买：吃卖一
            buy_vol += min(s, aq)
            aq -= s
        else:  # 市价卖：吃买一
            sell_vol += min(s, bq)
            bq -= s
        if bq <= 0:  # 买一耗尽 → 买价后退，露出下一档
            bp, bq = bp - 1, int(refill_b[i])
        if aq <= 0:
            ap, aq = ap + 1, int(refill_a[i])
        if (i + 1) % snapshot_every == 0:
            rows.append((bp * tick, bq, ap * tick, aq, buy_vol, sell_vol))
            buy_vol = sell_vol = 0
    df = pd.DataFrame(rows, columns=["bid_px", "bid_sz", "ask_px", "ask_sz", "buy_vol", "sell_vol"])
    df[["bid_px", "ask_px"]] = df[["bid_px", "ask_px"]].round(6)
    return df


# ==================== 合成深度快照（离线回退） ====================


def synthetic_depth_snapshots(
    n_snapshots: int = 8,
    levels: int = 100,
    mid0: float = 60_000.0,
    tick: float = 0.1,
    seed: int = 11,
) -> list[dict]:
    """
    生成一串合成的深度快照，格式与 ccxt fetch_order_book 相同：
        {"bids": [[价格, 数量], ...], "asks": [...], "timestamp": 毫秒, "nonce": None}
    中间价小幅随机游走；越远离盘口挂单越多。仅用于无网络时跑通代码，不代表真实市场。
    """
    rng = np.random.default_rng(seed)
    mid = mid0
    base_ts = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    out = []
    for i in range(n_snapshots):
        mid += rng.normal(0, 3 * tick)
        half = tick * (0.5 + int(rng.integers(0, 3)))
        book = {}
        for side, sign in (("bids", -1), ("asks", 1)):
            gaps = rng.integers(1, 12, levels)
            prices = mid + sign * (half + np.concatenate([[0], np.cumsum(gaps[:-1])]) * tick)
            sizes = rng.lognormal(mean=-1.5, sigma=1.2, size=levels) * (1 + np.arange(levels) / 30)
            book[side] = [[round(float(p), 1), round(float(q), 6)] for p, q in zip(prices, sizes)]
        book["timestamp"] = base_ts + i * 1000
        book["nonce"] = None
        out.append(book)
    return out


# ==================== 日内成交量分布 ====================


def intraday_volume_profile(n_bins: int = 240) -> np.ndarray:
    """
    A 股典型的日内 U 型成交量分布（240 个交易分钟：上午 120 + 下午 120）。
    开盘最高、午盘前后最低、收盘前回升。返回和为 1 的权重数组。
    形状为示意，并非从某只股票校准。
    """
    x = np.linspace(0, 1, n_bins)
    u = 0.35 + 2.2 * np.exp(-x / 0.04) + 1.2 * (x - 0.5) ** 2 * 4 + 0.9 * np.exp(-(1 - x) / 0.03)
    return u / u.sum()
