# Phase 13: 交易系统工程与实盘 - 核心概念深度解析

> **学习目标**：设计并实现可用于模拟盘和小资金实盘的多市场交易系统
>
> **前置知识**：Phase 4（事件驱动回测引擎）、Phase 11（订单簿、订单类型、执行算法）
>
> **阅读建议**：这是对软件工程师最"友好"的一个阶段——大部分概念你都在 Web 后端见过（事件循环、状态机、幂等、监控、容器化）。真正的难点在于**金融场景下错误的代价**：一个重复下单的 bug 不是返回 500，而是真金白银的亏损。所以本文每一节都会强调"出错时会怎样"。

---

## 目录

1. [引言：从回测到实盘的鸿沟](#1-引言从回测到实盘的鸿沟)
2. [交易系统架构](#2-交易系统架构)
3. [asyncio 事件引擎](#3-asyncio-事件引擎)
4. [OMS 与订单状态机](#4-oms-与订单状态机)
5. [事前 / 事中风控引擎](#5-事前--事中风控引擎)
6. [vn.py 框架入门](#6-vnpy-框架入门)
7. [CTP 期货接口与 SimNow 仿真](#7-ctp-期货接口与-simnow-仿真)
8. [A 股程序化接口：QMT / 掘金等](#8-a-股程序化接口qmt--掘金等)
9. [ccxt 加密货币实盘与测试网](#9-ccxt-加密货币实盘与测试网)
10. [监控与告警](#10-监控与告警)
11. [对账、幂等与故障恢复](#11-对账幂等与故障恢复)
12. [Docker 部署与定时任务](#12-docker-部署与定时任务)
13. [从模拟盘到小资金实盘的流程与合规](#13-从模拟盘到小资金实盘的流程与合规)
14. [最佳实践与常见陷阱](#14-最佳实践与常见陷阱)
15. [总结](#15-总结)

---

## 1. 引言：从回测到实盘的鸿沟

### 1.1 回测世界 vs 真实世界

| 维度 | 回测 | 实盘 |
| --- | --- | --- |
| 时间 | 循环遍历历史数据，可以"快进" | 真实时间流逝，事件随时到达 |
| 下单 | 函数调用，立即返回成交 | 异步请求，回报延迟到达，可能被拒、部分成交 |
| 数据 | 干净、完整、有序 | 可能缺失、重复、乱序、延迟 |
| 状态 | 全在内存，进程结束即丢弃 | 必须持久化，崩溃后能恢复 |
| 失败 | 抛异常，修代码重跑 | 断网、交易所维护、柜台拒单、进程崩溃 |
| 真相来源 | 自己的模拟撮合 | **交易所/柜台**，本地状态只是"缓存" |
| 后果 | 浪费时间 | 亏钱，甚至违规 |

### 1.2 工程师视角：你已经会的和需要新学的

```
┌──────────────────────────────────────────────────────────────────┐
│               Web 后端经验 → 交易系统的映射                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Web 后端                         交易系统                        │
│  ──────────                       ──────────                     │
│  消息队列 / 事件总线       ──▶     事件引擎（行情、订单、成交事件）   │
│  订单/支付状态机          ──▶     OMS 订单状态机                   │
│  限流 / 熔断              ──▶     风控引擎（流控、Kill Switch）     │
│  支付幂等键               ──▶     客户端订单号（Client Order ID）   │
│  对账（支付 vs 银行）      ──▶     持仓 / 资金 / 成交对账            │
│  第三方 SDK 适配层        ──▶     交易网关（CTP / QMT / ccxt）      │
│  可观测性                 ──▶     行情心跳、PnL、拒单率监控          │
│                                                                  │
│  需要新学的：交易所规则、柜台接口特性、金融合规、风控指标             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 本阶段地图

```
第 21 周：系统架构         第 22 周：交易接口           第 23 周：运维与上线
───────────────           ───────────────             ───────────────
01 整体架构               05 vn.py 入门                09 监控告警
02 asyncio 事件引擎        06 CTP + SimNow              10 对账 / 幂等 / 恢复
03 OMS 与订单状态机        07 A 股 QMT / 掘金            11 Docker 与定时任务
04 风控引擎               08 ccxt 实盘与测试网          12 模拟盘 → 小资金实盘
```

---

## 2. 交易系统架构

> 对应课程：`01-trading-system-architecture.py`

### 2.1 全景图

```
┌───────────────────────────────────────────────────────────────────────┐
│                           交易系统全景                                 │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   交易所 / 柜台（CTP、券商、加密交易所）                                 │
│        │ 行情                                   ▲ 报单 / 撤单           │
│        ▼                                        │                      │
│   ┌──────────┐                            ┌──────────┐                 │
│   │ 行情网关  │                            │ 交易网关  │ ← 统一接口适配   │
│   └────┬─────┘                            └────▲─────┘                 │
│        │ TickEvent              Order/TradeEvent│  │ OrderRequest       │
│        ▼                                        │  │                    │
│   ╔═══════════════════ 事件引擎（Event Bus）══════════════════════╗     │
│   ╚═══╤════════════════╤═══════════════╤═══════════════╤══════════╝     │
│       ▼                ▼               ▼               ▼                │
│   ┌────────┐      ┌─────────┐     ┌─────────┐     ┌──────────┐         │
│   │ 策略    │ ───▶ │ 风控     │ ──▶ │  OMS    │ ──▶ │ 执行算法  │         │
│   │ 信号    │ 订单  │ 事前检查  │ 通过 │ 订单管理 │     │ TWAP 等   │         │
│   └────────┘ 意图  └─────────┘     └─────────┘     └──────────┘         │
│                                                                       │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │ 持久化/事件日志 │  │ 监控 / 告警    │  │ 对账 / 恢复    │                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                 │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件职责

| 组件 | 职责 | 关键设计点 |
| --- | --- | --- |
| 行情网关（Market Data Gateway） | 订阅行情，转换为统一的 Tick/Bar 数据结构 | 断线重连、去重、时间戳标准化 |
| 策略引擎（Strategy Engine） | 接收行情，产生"目标持仓"或"订单意图" | 不直接调用交易接口；回测与实盘同一套代码 |
| 风控引擎（Risk Engine） | 每个订单发出前检查；运行中监控账户指标 | 拥有"一票否决"和"全局暂停"权 |
| OMS（Order Management System） | 管理订单生命周期、维护订单与持仓状态 | 状态机、本地/交易所订单号映射 |
| 执行（Execution） | 把大单拆成子单（Phase 11 的 TWAP/VWAP） | 与 OMS 协作，跟踪母单进度 |
| 交易网关（Trading Gateway） | 适配各家接口（CTP、QMT、ccxt），统一下单/撤单/查询 | 屏蔽接口差异，处理流控 |
| 持久化 | 订单、成交、事件日志、策略状态 | 追加写（Append-only），可回放 |
| 监控与对账 | 健康检查、告警、定期与交易所核对状态 | 独立于交易主流程 |

### 2.3 统一数据模型

**为什么重要**：CTP、QMT、ccxt 的字段命名、方向表示、状态枚举完全不同。如果策略代码里出现 `if exchange == "binance"`，系统就会迅速腐化。正确做法是在网关层把一切转换成内部统一模型（类似后端的 DTO / 防腐层 Anti-Corruption Layer）。

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class Direction(Enum):
    LONG = "多"
    SHORT = "空"

class Offset(Enum):          # 期货需要开平；股票、现货可忽略
    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSE_TODAY = "平今"
    CLOSE_YESTERDAY = "平昨"

class Status(Enum):
    SUBMITTING = "提交中"
    NOT_TRADED = "未成交"
    PART_TRADED = "部分成交"
    ALL_TRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"

@dataclass
class TickData:
    symbol: str              # 统一格式，如 "rb2501.SHFE"、"600000.SSE"、"BTC/USDT.BINANCE"
    dt: datetime
    last_price: float
    bid_price_1: float
    ask_price_1: float
    bid_volume_1: float
    ask_volume_1: float

@dataclass
class OrderRequest:
    symbol: str
    direction: Direction
    offset: Offset
    price: float
    volume: float
    client_order_id: str     # 本地生成，全局唯一，用于幂等

@dataclass
class OrderData:
    client_order_id: str
    symbol: str
    direction: Direction
    offset: Offset
    price: float
    volume: float
    traded: float = 0.0
    status: Status = Status.SUBMITTING
    exchange_order_id: str = ""
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradeData:
    trade_id: str            # 交易所成交编号，用于去重
    client_order_id: str
    symbol: str
    direction: Direction
    offset: Offset
    price: float
    volume: float
    dt: datetime
```

### 2.4 网关抽象

```python
from abc import ABC, abstractmethod

class BaseGateway(ABC):
    """所有交易接口的统一抽象。网关只负责"翻译"，不包含业务逻辑。"""

    def __init__(self, event_engine, name: str):
        self.event_engine = event_engine
        self.name = name

    @abstractmethod
    async def connect(self, setting: dict) -> None: ...
    @abstractmethod
    async def subscribe(self, symbol: str) -> None: ...
    @abstractmethod
    async def send_order(self, req: OrderRequest) -> str: ...
    @abstractmethod
    async def cancel_order(self, client_order_id: str) -> None: ...
    @abstractmethod
    async def query_positions(self) -> list: ...
    @abstractmethod
    async def query_open_orders(self) -> list: ...

    # 网关收到柜台回报后，转换为统一模型并推送事件
    def on_tick(self, tick):   self.event_engine.put("tick", tick)
    def on_order(self, order): self.event_engine.put("order", order)
    def on_trade(self, trade): self.event_engine.put("trade", trade)
```

### 2.5 架构原则

| 原则 | 说明 |
| --- | --- |
| 策略不碰网关 | 策略只输出意图，所有订单必须经过风控和 OMS |
| 交易所是真相来源 | 本地状态可能错，定期以交易所查询结果为准校正 |
| 单一写者 | 订单与持仓状态只由 OMS 修改，避免并发写导致不一致 |
| 一切皆事件 | 行情、订单、成交、定时器、错误都是事件，可记录、可回放 |
| 回测实盘同构 | 回测时用"模拟网关"替换真实网关，策略代码不变（Phase 4 事件驱动引擎的延续） |
| 失败默认安全 | 不确定时宁可停止交易，也不要"猜"状态继续下单 |

---

## 3. asyncio 事件引擎

> 对应课程：`02-event-engine-asyncio.py`

### 3.1 为什么用事件驱动

**比喻**：交易系统像机场塔台。飞机（事件）随时到达，塔台不会挨个去"轮询"每架飞机，而是通过无线电频道（事件总线）接收消息，再分派给对应的调度员（事件处理器）。

交易系统的工作负载是典型的 **I/O 密集型**：大部分时间在等待网络（行情推送、订单回报），每个事件的计算量很小。这正是 asyncio（单线程事件循环 + 协程）擅长的场景，与 Node.js 的事件循环模型相似。

| 概念 | asyncio | Node.js 类比 |
| --- | --- | --- |
| 事件循环 | `asyncio` event loop | libuv event loop |
| 协程 | `async def` / `await` | `async function` / `await` |
| 并发任务 | `asyncio.create_task()` | 不 await 的 Promise |
| 队列 | `asyncio.Queue` | EventEmitter / 消息队列 |
| 阻塞陷阱 | 在协程里调用 `time.sleep()`、大计算 | 在回调里同步阻塞 |

### 3.2 实现一个事件引擎

```python
import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable

Handler = Callable[[Any], Awaitable[None]]

class EventEngine:
    def __init__(self, timer_interval: float = 1.0):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._timer_interval = timer_interval
        self._tasks: list[asyncio.Task] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def register(self, event_type: str, handler: Handler) -> None:
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def put(self, event_type: str, data: Any = None) -> None:
        """在事件循环线程内调用"""
        self._queue.put_nowait((event_type, data))

    def put_threadsafe(self, event_type: str, data: Any = None) -> None:
        """在其他线程（如 C++ 柜台 API 的回调线程）中调用"""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (event_type, data))

    async def _run(self) -> None:
        while True:
            event_type, data = await self._queue.get()
            for handler in self._handlers.get(event_type, []):
                try:
                    await handler(data)
                except Exception as e:                 # 隔离：一个处理器出错不影响其他处理器
                    self.put("error", (event_type, repr(e)))
            self._queue.task_done()

    async def _timer(self) -> None:
        while True:
            await asyncio.sleep(self._timer_interval)
            self.put("timer")

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._tasks = [asyncio.create_task(self._run()), asyncio.create_task(self._timer())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

async def main():
    engine = EventEngine(timer_interval=0.2)

    async def on_tick(tick):
        print("策略收到行情:", tick)

    async def on_error(err):
        print("错误事件:", err)

    async def bad_handler(tick):
        raise ValueError("模拟处理器异常")

    engine.register("tick", on_tick)
    engine.register("tick", bad_handler)
    engine.register("error", on_error)
    engine.start()
    for px in [10.00, 10.01, 10.02]:
        engine.put("tick", {"symbol": "600000.SSE", "last": px})
    await asyncio.sleep(0.5)
    await engine.stop()

asyncio.run(main())
```

### 3.3 事件引擎的陷阱

| 陷阱 | 后果 | 解决方案 |
| --- | --- | --- |
| 在处理器中阻塞（`time.sleep`、同步 HTTP、重计算） | 整个事件循环卡住，行情积压，订单回报延迟 | 用 `await asyncio.sleep`；CPU 密集计算放到 `run_in_executor` 或独立进程 |
| 柜台 API 回调在其他线程 | 直接操作 asyncio 对象导致竞态或报错 | 用 `loop.call_soon_threadsafe` 投递到事件循环 |
| 异常被吞掉 | 任务静默退出，系统"假活" | 处理器级 try/except + 错误事件 + 告警 |
| 队列无限增长 | 处理速度跟不上行情，延迟越来越大 | 监控队列长度；行情可只保留每个标的最新一条（合并） |
| 事件顺序假设 | 假设"订单回报一定先于成交回报" | 处理器必须容忍乱序（见第 4 节） |
| 处理器注册顺序依赖 | 风控与策略处理同一事件的先后影响结果 | 关键流程显式调用，不依赖注册顺序 |

---

## 4. OMS 与订单状态机

> 对应课程：`03-oms-order-lifecycle.py`

### 4.1 订单的生命周期

**比喻**：订单像一个快递包裹——"已下单 → 已揽收 → 运输中（部分送达）→ 已签收"，也可能"被拒收"或"中途取消"。每个状态只能按规定路径转换，物流系统必须能处理"取消请求发出后包裹已经送达"这类竞态。

```
                          订单状态机

                      ┌─────────────┐
            send ───▶ │ SUBMITTING  │ 提交中（本地已发出，尚未确认）
                      └──────┬──────┘
               ┌─────────────┼──────────────────────┐
               ▼             ▼                      ▼
        ┌───────────┐  ┌───────────┐         ┌─────────────┐
        │ REJECTED  │  │NOT_TRADED │ 已挂单   │ ALL_TRADED  │ 直接全部成交
        │   拒单     │  │  未成交    │────┐    └─────────────┘
        └───────────┘  └─────┬─────┘    │           ▲
                             │ 部分成交   │ 撤单       │ 剩余全部成交
                             ▼          │           │
                       ┌───────────┐    │           │
                       │PART_TRADED│────┼───────────┘
                       │ 部分成交   │◀─┐ │
                       └─────┬─────┘  │ │ (继续部分成交)
                             │ 撤单    └─┘
                             ▼          ▼
                       ┌──────────────────┐
                       │    CANCELLED     │ 已撤销（可能已有部分成交）
                       └──────────────────┘

   终态：ALL_TRADED / CANCELLED / REJECTED —— 进入终态后不再变化
```

| 状态 | 含义 | 是否终态 | 是否占用资金/持仓 |
| --- | --- | --- | --- |
| SUBMITTING | 已发出请求，等待柜台确认 | 否 | 是（应预先冻结） |
| NOT_TRADED | 已在交易所挂单，未成交 | 否 | 是 |
| PART_TRADED | 部分成交，剩余仍挂单 | 否 | 是 |
| ALL_TRADED | 全部成交 | 是 | 转为持仓 |
| CANCELLED | 已撤销（剩余部分不再成交） | 是 | 释放未成交部分 |
| REJECTED | 被风控/柜台/交易所拒绝 | 是 | 释放 |

### 4.2 状态机实现

```python
from enum import Enum

class Status(Enum):
    SUBMITTING = "提交中"
    NOT_TRADED = "未成交"
    PART_TRADED = "部分成交"
    ALL_TRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"

TERMINAL = {Status.ALL_TRADED, Status.CANCELLED, Status.REJECTED}
ALLOWED = {
    Status.SUBMITTING:  {Status.NOT_TRADED, Status.PART_TRADED, Status.ALL_TRADED,
                         Status.REJECTED, Status.CANCELLED},
    Status.NOT_TRADED:  {Status.PART_TRADED, Status.ALL_TRADED, Status.CANCELLED},
    Status.PART_TRADED: {Status.PART_TRADED, Status.ALL_TRADED, Status.CANCELLED},
}

class Order:
    def __init__(self, coid: str, volume: int):
        self.coid, self.volume = coid, volume
        self.traded = 0
        self.status = Status.SUBMITTING
        self.seen_trades: set[str] = set()

    def on_status(self, new: Status) -> bool:
        if self.status in TERMINAL:
            return False                                  # 终态后忽略迟到的回报
        if new not in ALLOWED[self.status]:
            raise ValueError(f"非法状态转换 {self.status.name} -> {new.name}")
        self.status = new
        return True

    def on_trade(self, trade_id: str, qty: int) -> None:
        if trade_id in self.seen_trades:
            return                                        # 成交回报去重
        self.seen_trades.add(trade_id)
        self.traded += qty
        if self.traded > self.volume:
            raise ValueError("成交量超过委托量：数据异常，需要人工介入")
        if self.status in TERMINAL:
            return                                        # 撤单后才到达的成交：只记量，不改终态
        self.status = Status.ALL_TRADED if self.traded == self.volume else Status.PART_TRADED

o = Order("S1-0001", volume=500)
o.on_status(Status.NOT_TRADED)
o.on_trade("T1", 200)
o.on_trade("T1", 200)            # 重复回报被忽略
print(o.status.name, o.traded)    # PART_TRADED 200
o.on_status(Status.CANCELLED)
print(o.status.name, o.traded)    # CANCELLED 200：撤单成功，已成交部分保留
```

### 4.3 真实世界的复杂情况

| 场景 | 描述 | 处理 |
| --- | --- | --- |
| 撤单竞态 | 撤单请求发出时订单恰好全部成交 | 撤单被拒属于正常情况，以最终回报为准 |
| 乱序回报 | 先收到成交回报，后收到订单确认 | 成交回报到达时若订单状态落后，先推进状态 |
| 迟到成交 | 已标记 CANCELLED 后又收到一笔成交 | 部分柜台可能出现，需按成交更新成交量与持仓 |
| 重复回报 | 重连后柜台重推当日回报 | 按成交编号（trade_id）去重 |
| 状态卡住 | SUBMITTING 很久没有回音 | 超时后主动查询；查询不到时进入"未知"状态并告警，**不要重发** |
| 部分成交后拒单 | 少数接口可能先部分成交再报错 | 以成交回报为准更新持仓 |

### 4.4 订单号映射

```
  策略            OMS                           柜台/交易所
   │  下单意图     │                                │
   │──────────────▶│ 生成 client_order_id=S1-0001   │
   │               │───────── 报单（带 coid）──────▶│
   │               │◀──── 确认：exchange_id=88812 ───│
   │               │  维护映射 coid ↔ exchange_id   │
   │               │◀──── 成交：exchange_id=88812 ───│
   │◀── 成交通知 ───│  通过映射找到本地订单            │
```

- **Client Order ID**：本地生成，发出前就已知。CTP 中对应 `FrontID + SessionID + OrderRef` 的组合；ccxt 中通过 `clientOrderId` 参数传递（各交易所字段名不同，ccxt 负责映射）。
- **Exchange Order ID**：交易所确认后才有。

### 4.5 持仓：由成交推导

持仓不应该由"订单状态"推导，而应该由**成交（Trade）**累计：

```python
from collections import defaultdict

class PositionBook:
    def __init__(self):
        self.qty = defaultdict(float)         # 净持仓：正为多，负为空
        self.cost = defaultdict(float)        # 持仓成本（简化：只记录开仓成本）

    def on_trade(self, symbol: str, side: int, price: float, volume: float):
        """side=+1 买入，-1 卖出"""
        old = self.qty[symbol]
        new = old + side * volume
        if old == 0 or (old > 0) == (side > 0):   # 开仓或加仓：更新均价
            self.cost[symbol] = (abs(old) * self.cost[symbol] + volume * price) / abs(new)
        elif new == 0:
            self.cost[symbol] = 0.0
        elif (new > 0) != (old > 0):              # 反手：剩余部分以成交价为成本
            self.cost[symbol] = price
        self.qty[symbol] = new

pb = PositionBook()
pb.on_trade("rb2501.SHFE", +1, 3500, 2)
pb.on_trade("rb2501.SHFE", +1, 3520, 2)
pb.on_trade("rb2501.SHFE", -1, 3550, 1)
print(pb.qty["rb2501.SHFE"], round(pb.cost["rb2501.SHFE"], 2))   # 3.0 3510.0
```

> **A 股与期货的差异**：A 股 T+1 下需区分"总持仓"与"可用持仓"（今日买入的不可卖）；期货需区分多/空两个方向的持仓，以及今仓与昨仓（见 7.4 节）。

---

## 5. 事前 / 事中风控引擎

> 对应课程：`04-risk-control-engine.py`

### 5.1 风控分层

**比喻**：风控像汽车的安全系统。**事前风控**是出发前的车况检查（订单发出前），**事中风控**是行驶中的 ABS 和安全气囊（运行中实时监控），**事后风控**是事故复盘（盘后分析）。

```
┌──────────────────────────────────────────────────────────────────┐
│                          风控三道防线                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  事前（Pre-trade）：每个订单发出前同步检查，不通过直接拒绝            │
│     数量 / 金额 / 价格偏离 / 持仓上限 / 资金 / 流控 / 自成交           │
│                                                                  │
│  事中（Intra-day）：持续监控账户与系统状态，触发时暂停交易            │
│     日内亏损 / 回撤 / 拒单率 / 撤单率 / 行情中断 / 对账差异            │
│                                                                  │
│  事后（Post-trade）：盘后分析与报告                                 │
│     成交质量（TCA）/ 风险暴露 / 异常交易回顾 / 规则调参                │
│                                                                  │
│  另外：柜台与交易所也有自己的风控，但不能依赖它们替你兜底              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 事前检查清单

| 检查项 | 规则示例 | 防范的问题 |
| --- | --- | --- |
| 单笔数量上限 | 每笔不超过 N 手/股 | "胖手指"（多打一个 0） |
| 单笔金额上限 | 每笔不超过 X 元 | 价格单位错误 |
| 价格偏离 | 限价不能偏离最新价超过 Y% | 价格错误、追高杀跌 |
| 持仓上限 | 单标的、单方向、总持仓上限 | 策略 bug 持续加仓 |
| 资金检查 | 下单所需资金/保证金 ≤ 可用资金 | 柜台拒单、穿仓 |
| 报单频率（流控） | 每秒/每分钟报单数上限 | 死循环下单、异常交易监控 |
| 撤单次数 | 日内撤单笔数上限 | 交易所对频繁撤单的监管 |
| 自成交检查 | 不与自己的反向挂单成交 | 自成交属于需要避免的异常交易行为 |
| 标的白名单 | 只能交易配置内的标的 | 交易了不该交易的品种 |
| 交易时段 | 非交易时段不下单 | 无效报单 |
| A 股规则 | 涨跌停、价格笼子、100 股、可用持仓 | 无效报单 |

### 5.3 代码：可插拔规则 + Kill Switch

```python
import time
from collections import deque
from dataclasses import dataclass

@dataclass
class Req:
    symbol: str
    side: int          # +1 买 / -1 卖
    price: float
    volume: int

class RiskEngine:
    def __init__(self, max_volume=100, max_notional=1e6, max_dev=0.05,
                 max_orders_per_sec=5, max_daily_loss=-50_000):
        self.max_volume, self.max_notional, self.max_dev = max_volume, max_notional, max_dev
        self.max_orders_per_sec, self.max_daily_loss = max_orders_per_sec, max_daily_loss
        self.last_price: dict[str, float] = {}
        self.order_times: deque = deque()
        self.halted = False                      # Kill Switch：一旦触发，拒绝所有新单
        self.halt_reason = ""

    def check(self, r: Req) -> tuple[bool, str]:
        if self.halted:
            return False, f"交易已暂停：{self.halt_reason}"
        if r.volume <= 0 or r.volume > self.max_volume:
            return False, f"数量 {r.volume} 超限"
        if r.price * r.volume > self.max_notional:
            return False, "金额超限"
        ref = self.last_price.get(r.symbol)
        if ref is None:
            return False, "无参考价（行情缺失），拒绝下单"
        if abs(r.price / ref - 1) > self.max_dev:
            return False, f"价格偏离 {r.price / ref - 1:+.1%} 超限"
        now = time.monotonic()
        while self.order_times and now - self.order_times[0] > 1.0:
            self.order_times.popleft()
        if len(self.order_times) >= self.max_orders_per_sec:
            return False, "报单频率超限"
        self.order_times.append(now)
        return True, "OK"

    def on_pnl(self, daily_pnl: float) -> None:   # 事中：由定时器或成交事件触发
        if daily_pnl <= self.max_daily_loss and not self.halted:
            self.halt(f"日内亏损 {daily_pnl:,.0f} 触及上限")

    def halt(self, reason: str) -> None:
        self.halted, self.halt_reason = True, reason
        print(f"[KILL SWITCH] {reason} —— 应同时撤销全部挂单并发出 P0 告警")

risk = RiskEngine()
risk.last_price["600000.SSE"] = 10.0
print(risk.check(Req("600000.SSE", +1, 10.02, 50)))
print(risk.check(Req("600000.SSE", +1, 11.00, 50)))    # 价格偏离
print(risk.check(Req("600000.SSE", +1, 10.02, 5000)))  # 胖手指
risk.on_pnl(-60_000)
print(risk.check(Req("600000.SSE", +1, 10.02, 50)))    # 已暂停
```

### 5.4 事中风控：何时自动暂停

| 触发条件 | 动作 |
| --- | --- |
| 日内亏损 / 回撤超过阈值 | 暂停开新仓，撤销挂单，告警 |
| 行情中断（N 秒无 Tick） | 暂停该标的交易，告警 |
| 拒单率异常（如 1 分钟内连续拒单） | 暂停交易，可能是账户、资金或接口问题 |
| 对账差异 | 暂停交易，等待人工确认 |
| 网关断线 | 暂停交易，自动重连后先对账再恢复 |
| 持仓与目标严重偏离 | 告警，人工判断 |

> **设计原则**：自动"暂停"可以，自动"恢复"要谨慎。触发 Kill Switch 后，恢复交易应需要人工确认，或至少在完成对账之后。

---

## 6. vn.py 框架入门

> 对应课程：`05-vnpy-intro.py`

### 6.1 vn.py 是什么

[vn.py](https://www.vnpy.com/) 是国内使用最广泛的开源 Python 量化交易框架之一，覆盖期货（CTP 等）、股票、期权等多种接口，采用模块化设计：核心框架 `vnpy` 加上按需安装的接口与应用插件（如 `vnpy_ctp`、`vnpy_ctastrategy`、`vnpy_ctabacktester`、`vnpy_datamanager` 等）。

### 6.2 核心架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         vn.py 架构                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                       MainEngine（主引擎）                         │
│        ┌──────────────────┼───────────────────┐                  │
│        ▼                  ▼                   ▼                  │
│   Gateway（接口）       App（应用）           OmsEngine 等         │
│   CtpGateway 等        CtaStrategy           订单/持仓/账户缓存     │
│        │               CtaBacktester                              │
│        │               DataManager ...                            │
│        ▼                  ▲                                       │
│   ═══════════════ EventEngine（事件引擎）═══════════════            │
│   事件类型：EVENT_TICK / ORDER / TRADE / POSITION / ACCOUNT / TIMER │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

它的设计与本文第 2–4 节的架构高度一致：网关推送事件 → 事件引擎分发 → OMS 缓存状态 → 策略应用处理。学习 vn.py 源码是理解交易系统工程的好途径。

### 6.3 CTA 策略模板

`vnpy_ctastrategy` 中的策略继承 `CtaTemplate`，通过回调函数响应事件（以下为结构示意，方法与参数以所安装版本的源码为准）：

```python
from vnpy_ctastrategy import CtaTemplate, BarGenerator, ArrayManager

class DoubleMaStrategy(CtaTemplate):
    author = "qtc"
    fast_window = 10
    slow_window = 30
    fixed_size = 1
    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)        # Tick 合成 1 分钟 K 线
        self.am = ArrayManager()                   # 缓存 K 线序列，计算指标
        self.fast_ma = self.slow_ma = 0.0

    def on_init(self):
        self.load_bar(10)                          # 加载历史数据预热指标

    def on_tick(self, tick):
        self.bg.update_tick(tick)

    def on_bar(self, bar):
        self.cancel_all()
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        self.fast_ma = self.am.sma(self.fast_window)
        self.slow_ma = self.am.sma(self.slow_window)
        if self.fast_ma > self.slow_ma and self.pos <= 0:
            if self.pos < 0:
                self.cover(bar.close_price, abs(self.pos))   # 平空
            self.buy(bar.close_price, self.fixed_size)       # 开多
        elif self.fast_ma < self.slow_ma and self.pos >= 0:
            if self.pos > 0:
                self.sell(bar.close_price, abs(self.pos))    # 平多
            self.short(bar.close_price, self.fixed_size)     # 开空
        self.put_event()

    def on_order(self, order):
        pass

    def on_trade(self, trade):
        self.put_event()
```

- `buy / sell / short / cover` 分别对应 开多 / 平多 / 开空 / 平空，这是期货"开平仓"语义的直接体现。
- `self.pos` 是策略内部记录的持仓，**不等于**账户真实持仓（多个策略可能交易同一合约），这是对账时的常见困惑点。

### 6.4 自研 vs 使用框架

| 维度 | 使用 vn.py | 自研（本课程项目） |
| --- | --- | --- |
| 上手速度 | 快，接口与 GUI 现成 | 慢 |
| 接口覆盖 | 广（期货、股票、期权多种柜台） | 只实现需要的 |
| 理解深度 | 容易停留在"会用" | 理解每一层的设计取舍 |
| 定制性 | 需要理解框架内部 | 完全掌控 |
| 推荐做法 | **生产中可用 vn.py 的网关层**，上层风控、监控、对账按需自建 | 学习阶段自研核心组件 |

---

## 7. CTP 期货接口与 SimNow 仿真

> 对应课程：`06-ctp-simnow.py`

### 7.1 CTP 简介

**CTP（Comprehensive Transaction Platform，综合交易平台）** 由上海期货交易所旗下的上海期货信息技术有限公司（上期技术）开发，是国内期货公司使用最广泛的交易柜台系统之一。程序化交易者通过 CTP 提供的 C++ API 接入：

| API | 功能 |
| --- | --- |
| MdApi（行情） | 订阅合约行情，推送 Tick（期货行情通常为每秒 2 次快照） |
| TraderApi（交易） | 认证、登录、报单、撤单、查询持仓/资金/合约、接收订单与成交回报 |

CTP API 是**异步回调**模式：调用 `ReqXxx` 发出请求，结果通过 `OnRspXxx`（请求响应）或 `OnRtnXxx`（主动推送，如订单/成交回报）回调返回，回调发生在 API 内部线程中。Python 通常通过 vn.py 等项目封装的接口来使用。

### 7.2 登录与交易流程

```
  客户端                                         CTP 前置
    │──── 连接交易前置（地址：tcp://ip:port）──────▶│
    │◀──────────── OnFrontConnected ──────────────│
    │──── ReqAuthenticate（AppID + AuthCode）─────▶│  穿透式监管认证
    │◀──────────── OnRspAuthenticate ─────────────│
    │──── ReqUserLogin（BrokerID + 账号 + 密码）───▶│
    │◀──────────── OnRspUserLogin ────────────────│  返回 FrontID、SessionID、交易日
    │──── ReqSettlementInfoConfirm ──────────────▶│  确认结算单后才能交易
    │◀──────────── OnRspSettlementInfoConfirm ────│
    │──── 查询合约 / 资金 / 持仓（注意查询流控）───▶│
    │──── ReqOrderInsert ────────────────────────▶│
    │◀──── OnRtnOrder（状态变化）/ OnRtnTrade（成交）│
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| BrokerID | 期货公司代码 |
| UserID / Password | 投资者账号与密码 |
| AppID / AuthCode | 穿透式监管要求的程序标识与认证码，需向期货公司申请 |
| 交易前置 / 行情前置地址 | 由期货公司（或 SimNow）提供 |

**穿透式监管**：自 2019 年起，期货市场全面实施穿透式监管，程序化交易软件需要通过期货公司的认证（AppID/AuthCode），并采集终端信息上报，使监管能够识别到具体的交易终端。

### 7.3 期货的开平仓与今昨仓

与股票"买入/卖出"不同，期货需要同时指定**方向**和**开平**：

| 操作 | 方向 | 开平 | vn.py 方法 |
| --- | --- | --- | --- |
| 开多 | 买 | 开仓 | `buy` |
| 平多 | 卖 | 平仓 | `sell` |
| 开空 | 卖 | 开仓 | `short` |
| 平空 | 买 | 平仓 | `cover` |

- 期货账户可以**同时**持有同一合约的多头和空头（"锁仓"），所以持仓按方向分别记录。
- **上期所和能源中心**区分"平今仓"和"平昨仓"，平仓指令必须指定；其他交易所一般由系统按规则处理。不少品种的平今手续费与平昨不同，这会影响日内策略的成本。
- **夜盘与交易日**：有夜盘的品种，夜盘时段属于**下一个交易日**，结算、日内统计和"今仓"的界定都要按交易日而非自然日处理。

### 7.4 SimNow 仿真

[SimNow](https://www.simnow.com.cn/) 是上期技术提供的 CTP 仿真环境，可以免费注册获得仿真账号，是学习期货程序化交易的标准起点：

- 提供与实盘交易时段一致的仿真环境，以及用于开发调试的 7×24 环境（具体环境、前置地址与可用时间以 SimNow 官网最新说明为准）。
- 仿真撮合与真实市场有差异（成交规则、流动性），**仿真盘的成交结果不能代表实盘滑点**。
- 仿真环境会不定期维护或重置，系统要能正确处理"连不上"的情况。

### 7.5 用 vn.py 连接 CTP（示意）

```python
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway
from vnpy_ctastrategy import CtaStrategyApp

# 配置字段以所安装的 vnpy_ctp 版本的 default_setting 为准；地址等从 SimNow 官网获取
setting = {
    "用户名": "你的 SimNow 投资者代码",
    "密码": "你的密码",
    "经纪商代码": "9999",
    "交易服务器": "tcp://<交易前置地址>",
    "行情服务器": "tcp://<行情前置地址>",
    "产品名称": "simnow_client_test",
    "授权编码": "0000000000000000",
}

event_engine = EventEngine()
main_engine = MainEngine(event_engine)
main_engine.add_gateway(CtpGateway)
cta_engine = main_engine.add_app(CtaStrategyApp)
main_engine.connect(setting, "CTP")
```

### 7.6 CTP 接入的陷阱

| 陷阱 | 说明 |
| --- | --- |
| 查询流控 | CTP 对查询类请求有频率限制（通常约每秒 1 次），连续查询会返回流控错误，需要排队 |
| 报单流控 | 期货公司对报单/撤单频率也有限制 |
| 回调线程 | 回调在 API 线程中，不能在回调里做耗时操作 |
| 交易日切换 | 夜盘开始即进入新交易日，日志与统计按交易日归档 |
| 重连后的回报重推 | 重新登录后可能收到当日订单/成交回报的重推，必须去重 |
| 平今平昨 | 上期所/能源中心品种平仓指令选错会被拒单 |
| 平台兼容 | CTP 官方 API 主要面向 Windows 与 Linux（对 macOS 的支持以官方最新发布为准），macOS 上开发常借助 Docker 或虚拟机 |

---

## 8. A 股程序化接口：QMT / 掘金等

> 对应课程：`07-a-share-qmt-gm.py`

### 8.1 接入方式概览

个人和中小机构的 A 股程序化交易，通常通过**券商提供的程序化交易终端/接口**进行，而不是直接连接交易所。

| 类型 | 代表产品 | 运行方式 | 特点 |
| --- | --- | --- | --- |
| 本地终端 + 内置 Python | QMT（迅投） | 策略在券商定制的终端中运行 | 功能完整，含行情、回测、交易 |
| 本地终端 + 外部 Python 调用 | miniQMT + `xtquant` 库 | 终端常驻，外部 Python 进程通过库调用 | 可使用自己的 Python 环境和第三方库，灵活度高 |
| 券商云端托管 | PTrade（恒生） | 策略上传到券商机房的服务器运行 | 延迟较低、无需本地常驻，但环境受限 |
| 量化平台 + SDK | 掘金量化（`gm` SDK） | 终端 + Python SDK，支持仿真与券商实盘对接 | 研究、仿真、实盘一体 |

**重要说明**：

- 这些接口一般需要**向券商申请开通**，不同券商支持的产品不同，开通条件（资产要求、费率、是否收费）各不相同且会调整，**以券商的最新政策为准**，本课程不做承诺。
- 开通后同样需要遵守程序化交易报告要求（见 13.5 节）。

### 8.2 miniQMT / xtquant 接入示意

`xtquant` 分为行情模块 `xtdata` 和交易模块 `xttrader`。以下代码仅示意调用结构，**接口细节以券商提供的版本文档为准**：

```python
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount

class Callback(XtQuantTraderCallback):
    def on_stock_order(self, order):            # 委托回报
        print("委托:", order.order_id, order.order_status)
    def on_stock_trade(self, trade):            # 成交回报
        print("成交:", trade.stock_code, trade.traded_volume, trade.traded_price)
    def on_disconnected(self):
        print("与终端断开连接 —— 触发告警与重连逻辑")

trader = XtQuantTrader(r"<miniQMT 安装目录下的 userdata_mini 路径>", 123456)   # 第二个参数为会话 ID
trader.register_callback(Callback())
trader.start()
assert trader.connect() == 0, "连接终端失败"
acc = StockAccount("<资金账号>")
trader.subscribe(acc)

order_id = trader.order_stock(acc, "600000.SH", xtconstant.STOCK_BUY, 100,
                              xtconstant.FIX_PRICE, 10.50, "demo_strategy", "S1-0001")
```

### 8.3 掘金量化 SDK 接入示意

```python
from gm.api import (set_token, order_volume, OrderSide_Buy, OrderType_Limit,
                    PositionEffect_Open)

set_token("<你的 token>")
# 掘金的标的代码格式为 "交易所.代码"，如 SHSE.600000、SZSE.000001
order_volume(symbol="SHSE.600000", volume=100, side=OrderSide_Buy,
             order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=10.50)
```

> 掘金的策略通常以 `init(context)` + 事件回调（`on_bar`、`on_order_status` 等）+ `run(...)` 的形式组织，运行模式可在回测、仿真和实盘之间切换，具体用法以官方文档为准。

### 8.4 A 股实盘的特殊规则

| 规则 | 对系统的影响 |
| --- | --- |
| T+1 | 持仓需区分总量与可用量；当日买入不可当日卖出 |
| 100 股整数倍 | 买入数量取整（科创板等另有规定） |
| 涨跌停 | 涨停买不进、跌停卖不出；订单可能长时间挂单不成交 |
| 价格笼子 | 偏离基准价过多的限价单不会立即参与撮合 |
| 停牌 | 停牌标的无法交易，目标持仓需顺延 |
| 除权除息 | 持仓数量与成本价变化，对账时要识别 |
| 集合竞价 | 9:20–9:25、14:57–15:00 不能撤单 |

### 8.5 网关统一的价值

把 QMT、掘金、CTP、ccxt 都封装成第 2.4 节的 `BaseGateway`，策略层只看到统一的 `OrderRequest`。标的代码也要统一：

```python
def to_internal(symbol: str, source: str) -> str:
    """把各接口的代码格式转为内部统一格式 <代码>.<交易所>"""
    if source == "qmt":                         # 600000.SH / 000001.SZ
        code, ex = symbol.split(".")
        return f"{code}.{ {'SH': 'SSE', 'SZ': 'SZSE'}[ex] }"
    if source == "gm":                          # SHSE.600000 / SZSE.000001
        ex, code = symbol.split(".")
        return f"{code}.{ {'SHSE': 'SSE', 'SZSE': 'SZSE'}[ex] }"
    raise ValueError(f"未知来源 {source}")

print(to_internal("600000.SH", "qmt"), to_internal("SZSE.000001", "gm"))
```

---

## 9. ccxt 加密货币实盘与测试网

> 对应课程：`08-ccxt-live-trading.py`

> **合规提示**：根据中国人民银行等部门 2021 年发布的相关通知，虚拟货币相关业务活动在中国内地属于非法金融活动，境外虚拟货币交易所通过互联网向境内居民提供服务同样属于非法金融活动。本课程的加密货币内容仅用于学习交易系统技术（推荐只使用**测试网**），读者须自行了解并遵守所在地的法律法规。

### 9.1 ccxt 统一接口

| 功能 | 方法 |
| --- | --- |
| 加载市场信息（精度、最小量） | `load_markets()` |
| 下单 / 撤单 | `create_order()` / `cancel_order()` |
| 查询订单 | `fetch_order()` / `fetch_open_orders()` |
| 查询成交 | `fetch_my_trades()` |
| 查询余额 | `fetch_balance()` |
| WebSocket 订单/成交推送 | `watch_orders()` / `watch_my_trades()`（`ccxt.pro`） |

### 9.2 测试网（Sandbox）

测试网使用虚拟资产，接口与实盘基本一致，是加密实盘前的必经环节。ccxt 通过 `set_sandbox_mode(True)` 切换到交易所提供的测试环境（前提是该交易所在 ccxt 中配置了测试网地址）：

```python
import os
import ccxt

ex = ccxt.binance({
    "apiKey": os.environ["TESTNET_API_KEY"],      # 密钥从环境变量读取，绝不写入代码
    "secret": os.environ["TESTNET_SECRET"],
    "enableRateLimit": True,
})
ex.set_sandbox_mode(True)                         # 必须在任何请求之前调用
markets = ex.load_markets()

symbol = "BTC/USDT"
m = markets[symbol]
print("最小下单量:", m["limits"]["amount"]["min"], "价格精度:", m["precision"]["price"])

ticker = ex.fetch_ticker(symbol)
price = ex.price_to_precision(symbol, ticker["bid"] * 0.98)   # 挂一个不会立即成交的买单
amount = ex.amount_to_precision(symbol, 0.001)
order = ex.create_order(symbol, "limit", "buy", amount, price,
                        params={"clientOrderId": "S1-0001"})   # 统一参数，ccxt 负责映射
print(order["id"], order["status"])
ex.cancel_order(order["id"], symbol)
```

### 9.3 加密实盘的工程要点

| 要点 | 说明 |
| --- | --- |
| 精度与最小量 | 使用 `price_to_precision` / `amount_to_precision`，否则被拒 |
| 频率限制 | 各交易所按权重计算请求配额，超限会被临时封禁 |
| 7×24 运行 | 没有收盘，监控和运维要按"永不停机"设计 |
| 交易所维护 | 会有计划内维护与突发宕机，订阅交易所状态公告 |
| WebSocket 断线 | 连接会被服务端定期断开，需自动重连并重新同步订单状态 |
| 时间同步 | 签名请求依赖时间戳，本机时钟偏差过大会被拒 |
| API 密钥安全 | 只开交易权限、**禁止提现权限**、绑定 IP 白名单、密钥放环境变量或密钥管理服务 |

---

## 10. 监控与告警

> 对应课程：`09-monitoring-alerting.py`

### 10.1 监控什么

```
┌──────────────────────────────────────────────────────────────────┐
│                          监控四层模型                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  账户层   PnL、回撤、保证金占用率、持仓与目标偏离                    │
│  ───────────────────────────────────────────────────────────     │
│  交易层   报单数、拒单率、撤单率、成交率、滑点、订单滞留时间           │
│  ───────────────────────────────────────────────────────────     │
│  连接层   网关连接状态、最后一笔行情时间、回报延迟、重连次数            │
│  ───────────────────────────────────────────────────────────     │
│  系统层   进程存活、CPU/内存、磁盘、事件队列长度、时钟偏差              │
│                                                                  │
│  越往上越贴近"钱"，越往下越贴近"机器"——两者都要看                    │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 心跳与行情断流检测

最隐蔽的故障是"系统在运行，但行情已经不更新了"——进程存活检查发现不了。

```python
import time

class StalenessMonitor:
    def __init__(self, max_silence_sec: float):
        self.max_silence = max_silence_sec
        self.last_seen: dict[str, float] = {}

    def on_tick(self, symbol: str) -> None:
        self.last_seen[symbol] = time.monotonic()

    def check(self, in_trading_session: bool) -> list[str]:
        if not in_trading_session:                 # 非交易时段没有行情是正常的
            return []
        now = time.monotonic()
        return [s for s, t in self.last_seen.items() if now - t > self.max_silence]

mon = StalenessMonitor(max_silence_sec=0.1)
mon.on_tick("rb2501.SHFE")
time.sleep(0.15)
print("行情停滞:", mon.check(in_trading_session=True))
```

### 10.3 告警分级

| 级别 | 示例 | 通知方式 | 响应要求 |
| --- | --- | --- | --- |
| P0 致命 | Kill Switch 触发、对账差异、持仓异常 | 电话 + 即时消息 | 立即处理 |
| P1 严重 | 网关断线超过 N 分钟、连续拒单 | 即时消息（企业微信/钉钉/飞书机器人等） | 尽快处理 |
| P2 警告 | 滑点异常、行情延迟升高 | 即时消息 | 当日处理 |
| P3 信息 | 每日启动/收盘报告、日终 PnL | 邮件/消息 | 知悉 |

**告警设计原则**：

- **去重与抑制**：同一问题 5 分钟内只告警一次，避免"告警风暴"。
- **可操作**：每条告警要说明"发生了什么、影响是什么、建议做什么"。
- **告警疲劳是真实风险**：每天几十条无用告警，会让人忽略真正的 P0。
- **心跳告警（Dead Man's Switch）**：系统定期发送"我还活着"，如果监控方超时未收到，反向告警——用来发现"监控系统自己挂了"。

### 10.4 结构化日志与指标

```python
import json
import logging
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "ctx", {}))
        return json.dumps(payload, ensure_ascii=False)

log = logging.getLogger("trading")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log.addHandler(handler)
log.setLevel(logging.INFO)

log.info("order_sent", extra={"ctx": {"coid": "S1-0001", "symbol": "rb2501.SHFE",
                                       "side": "BUY", "price": 3500, "volume": 1}})
log.warning("order_rejected", extra={"ctx": {"coid": "S1-0002", "reason": "资金不足"}})
```

- 每条订单相关日志都带 `client_order_id`，像分布式系统里的 trace id 一样串起完整链路。
- 指标可以用 `prometheus_client` 暴露，用 Grafana 做看板；个人项目也可以先用 SQLite + 简单网页看板起步。

---

## 11. 对账、幂等与故障恢复

> 对应课程：`10-reconciliation-recovery.py`

### 11.1 为什么要对账

**比喻**：你的记账本（本地状态）和银行流水（交易所状态）必须定期核对。记账本可能因为漏记、重复记、记错而不准，**银行流水才是真相**。

本地状态与交易所不一致的常见原因：回报丢失、重复处理、进程崩溃前未持久化、人工在其他终端下单、交易所侧的调整（如分红送股、强平）。

| 对账对象 | 频率 | 差异处理 |
| --- | --- | --- |
| 持仓 | 盘中定时（如每 1–5 分钟）+ 启动时 + 日终 | 暂停相关标的交易，以交易所为准修正，告警 |
| 挂单 | 盘中定时 + 启动时 | 本地有、交易所无 → 查询最终状态；交易所有、本地无 → "孤儿订单"，撤销或纳入管理 |
| 成交 | 日终 | 按成交编号比对，补录缺失成交 |
| 资金 | 日终 | 核对权益、手续费、出入金 |

### 11.2 代码：持仓对账

```python
def reconcile_positions(local: dict[str, float], remote: dict[str, float],
                        tolerance: float = 1e-9) -> list[dict]:
    """返回差异列表；remote（交易所查询结果）为真相"""
    diffs = []
    for sym in sorted(set(local) | set(remote)):
        l, r = local.get(sym, 0.0), remote.get(sym, 0.0)
        if abs(l - r) > tolerance:
            diffs.append({"symbol": sym, "local": l, "remote": r, "diff": r - l})
    return diffs

local = {"rb2501.SHFE": 3, "600000.SSE": 1000}
remote = {"rb2501.SHFE": 2, "600000.SSE": 1000, "IF2412.CFFEX": -1}
for d in reconcile_positions(local, remote):
    print("持仓差异:", d)
# 有差异时：暂停交易 → 告警 → 以 remote 修正本地 → 人工确认原因后恢复
```

### 11.3 幂等：网络超时后到底下单没有？

**经典难题**：你发出下单请求，网络超时了。订单可能：(a) 根本没到达交易所；(b) 到达并已挂单/成交，只是回报丢了。**如果直接重试，情况 (b) 会导致重复下单。**

```
      发出报单 ──▶ 超时 / 无响应
                      │
                      ▼
           ┌───────────────────────┐
           │ 用 client_order_id 查询 │
           └──────────┬────────────┘
         ┌────────────┼──────────────┐
         ▼            ▼              ▼
     查到订单       确认不存在       查询也失败
   → 按其状态更新   → 可以安全重发    → 标记为"未知"
     本地记录       （仍用同一 coid） → 暂停该标的下单 + 告警
                                     → 等待网络恢复后再查
```

**幂等的实现要点**：

1. 每个订单意图在发出**之前**生成全局唯一的 `client_order_id`，并先持久化（"先写日志，再发请求"，Write-Ahead）。
2. 重试必须复用同一个 `client_order_id`。许多加密交易所会拒绝重复的客户端订单号，从而天然防止重复下单；对不支持的接口，必须"先查后发"。
3. 成交处理按 `trade_id` 去重（4.2 节），持仓更新才是幂等的。

```python
import itertools
import time

class OrderIdGenerator:
    """策略前缀 + 启动时间戳 + 自增序号：进程重启后也不会与之前的订单号冲突"""
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.boot = int(time.time())
        self.seq = itertools.count(1)

    def next(self) -> str:
        return f"{self.prefix}-{self.boot}-{next(self.seq):06d}"

gen = OrderIdGenerator("S1")
print(gen.next(), gen.next())
```

> 注意不同接口对订单号的长度和字符集有限制（如 CTP 的 OrderRef 为数字字符串且需在会话内递增），生成规则要按接口适配。

### 11.4 故障恢复：重启后从交易所恢复状态

```
┌──────────────────────────────────────────────────────────────────┐
│                       启动 / 恢复流程                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 加载本地持久化状态（订单、成交、策略状态）—— 仅作参考            │
│  2. 连接网关，登录                                                 │
│  3. 查询交易所：账户资金、持仓、挂单、当日成交                       │
│  4. 以交易所为准重建 OMS 状态：                                    │
│       • 补录本地缺失的成交，更新持仓                                │
│       • 本地"未完成"订单按交易所结果更新终态                         │
│       • 识别孤儿挂单（不是本系统下的或已丢失记录）                    │
│  5. 对账：若存在无法解释的差异 → 保持暂停，P0 告警，等待人工          │
│  6. 恢复策略内部状态（如目标持仓、计时器），与实际持仓对齐             │
│  7. 风控确认通过 → 恢复交易                                        │
│                                                                  │
│  核心原则：恢复时"宁停勿错"，不确定就不交易                          │
└──────────────────────────────────────────────────────────────────┘
```

**策略状态的持久化**：不仅订单要持久化，策略自身的状态（如"当前处于第几次加仓"、"止损价"）也要保存。更稳健的设计是让策略**以目标持仓表达意图**（"我希望持有 3 手"），而不是以动作表达（"再买 1 手"），这样恢复时只需比较"目标 vs 实际"即可，天然幂等。

### 11.5 事件日志（Event Sourcing）

```python
import json
import sqlite3

class EventLog:
    """追加写的事件日志：可用于审计、故障排查和状态回放"""
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS events ("
                        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, payload TEXT)")

    def append(self, ts: float, etype: str, payload: dict) -> None:
        self.db.execute("INSERT INTO events (ts, type, payload) VALUES (?, ?, ?)",
                        (ts, etype, json.dumps(payload, ensure_ascii=False)))
        self.db.commit()

    def replay(self, etype: str | None = None):
        sql = "SELECT ts, type, payload FROM events"
        rows = self.db.execute(sql + (" WHERE type = ?" if etype else "") + " ORDER BY id",
                               (etype,) if etype else ())
        for ts, t, p in rows:
            yield ts, t, json.loads(p)

log = EventLog()
log.append(1.0, "order_sent", {"coid": "S1-1", "volume": 2})
log.append(2.0, "trade", {"coid": "S1-1", "trade_id": "T9", "volume": 2})
print(list(log.replay("trade")))
```

### 11.6 故障场景演练表

| 场景 | 预期行为 | 如何演练 |
| --- | --- | --- |
| 网络断开 30 秒 | 暂停交易 → 自动重连 → 对账 → 恢复 | 在测试环境断开网卡或用防火墙阻断 |
| 进程被 kill -9 | 重启后从交易所恢复持仓与挂单 | 运行中直接杀进程 |
| 回报重复推送 | 成交去重，持仓不变 | 注入重复事件 |
| 下单超时 | 查询确认，不重复下单 | 模拟网关延迟/丢包 |
| 行情停滞 | 告警并暂停该标的 | 停止行情推送 |
| 交易所拒单 | 订单进入 REJECTED，风控统计拒单率 | 下一个超出涨跌停的单 |
| 磁盘写满 | 告警，停止交易（无法持久化就不能交易） | 限制容器磁盘 |

> 这就是学习检查中"重启后能从交易所状态恢复持仓"与"风控能在异常时自动暂停交易"的验证方法：**把故障当测试用例，主动制造它。**

---

## 12. Docker 部署与定时任务

> 对应课程：`11-docker-deployment.py`

### 12.1 为什么用 Docker

| 问题 | Docker 的作用 |
| --- | --- |
| "在我机器上能跑" | 镜像固定 Python 版本与依赖 |
| CTP 等库只有 Linux/Windows 版 | 在 macOS 上也能运行 Linux 容器（Apple Silicon 需注意 `linux/amd64` 平台与性能） |
| 进程崩溃 | `restart: unless-stopped` 自动拉起 |
| 多组件部署 | docker compose 一次编排交易进程、数据库、监控 |
| 迁移服务器 | 镜像 + 数据卷，快速迁移 |

### 12.2 示例配置

```dockerfile
FROM python:3.11-slim
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m trader && chown -R trader /app
USER trader
CMD ["python", "-m", "trading.main"]
```

```yaml
# docker-compose.yml
services:
  trader:
    build: .
    restart: unless-stopped
    env_file: .env                 # 密钥放 .env，且 .env 必须在 .gitignore 中
    volumes:
      - ./data:/app/data           # 事件日志、数据库持久化到宿主机
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "python", "-m", "trading.healthcheck"]
      interval: 30s
      timeout: 5s
      retries: 3
```

**关键细节**：

- **时区**：容器默认 UTC，交易时段判断全部会错 8 小时。统一设置 `TZ=Asia/Shanghai`，代码中使用带时区的时间。
- **时钟同步**：宿主机开启 NTP（如 chrony），时钟偏差会影响日志排查、签名请求和延迟统计。
- **健康检查要检查业务**：不仅检查进程存活，还要检查"最近 N 秒有行情"、"网关已连接"。

### 12.3 定时任务：按交易日历调度

普通 cron 不知道节假日，交易系统的调度必须结合**交易日历**（Phase 1）：

| 时间点（示例） | 任务 |
| --- | --- |
| 盘前（如 8:30 / 20:30 夜盘前） | 启动、登录、对账、加载策略 |
| 盘中 | 定时对账、心跳、风控检查 |
| 收盘后 | 撤销未成交单、日终对账、生成日报 |
| 夜间 | 数据更新、因子计算、生成次日目标持仓（Phase 14 的平台） |

```python
from datetime import date, datetime, time as dtime

TRADING_DAYS = {date(2026, 11, 9), date(2026, 11, 10)}       # 实际应从交易日历数据加载

SESSIONS = [(dtime(9, 15), dtime(11, 30)), (dtime(13, 0), dtime(15, 0))]   # A 股（含集合竞价）

def in_session(now: datetime) -> bool:
    if now.date() not in TRADING_DAYS:
        return False
    return any(start <= now.time() <= end for start, end in SESSIONS)

print(in_session(datetime(2026, 11, 9, 10, 0)))    # True
print(in_session(datetime(2026, 11, 9, 12, 0)))    # False：午休
print(in_session(datetime(2026, 11, 8, 10, 0)))    # False：非交易日
```

> 期货夜盘跨越自然日（部分品种夜盘到次日凌晨），夜盘时段要按"所属交易日"判断，不能简单用当天日期。

### 12.4 部署位置的选择

| 位置 | 延迟 | 稳定性 | 适合 |
| --- | --- | --- | --- |
| 家用电脑 | 高，网络不稳 | 低（断电、休眠、系统更新） | 仅开发调试 |
| 云服务器 | 中 | 高 | 个人中低频策略的推荐选择 |
| 券商/期货公司机房托管 | 低 | 高 | 对延迟敏感的策略，需单独申请 |

---

## 13. 从模拟盘到小资金实盘的流程与合规

> 对应课程：`12-paper-to-live.py`

### 13.1 分阶段上线流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                        策略上线的五个阶段                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 回测          ② 纸面交易         ③ 仿真盘           ④ 小资金实盘    ⑤ 逐步放大 │
│  历史数据         实时行情 +          SimNow /          真实账户        按规则  │
│  验证逻辑         本地模拟撮合         测试网             最小规模        加资金  │
│     │               │                  │                  │             │    │
│     └──▶ 样本外 ──▶ └──▶ 信号一致 ──▶ └──▶ ≥ 4 周稳定 ──▶ └──▶ 偏差可解释 ──▶ │
│                                                                      │
│  每一阶段都有准入标准和退出（回退）条件                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.2 各阶段准入与退出标准

| 阶段 | 准入标准 | 重点验证 | 回退条件 |
| --- | --- | --- | --- |
| 纸面交易 | 回测通过样本外检验（Phase 4/9） | 实时信号与回测信号一致 | 信号不一致且无法解释 |
| 仿真盘 | 纸面交易无逻辑错误 | 订单状态机、风控、对账、重启恢复、告警 | 出现未处理的异常状态 |
| 小资金实盘 | **仿真盘连续运行至少 4 周**，故障演练通过 | 真实滑点、手续费、成交率 | 实盘与回测偏差超出预设范围 |
| 放大资金 | 小资金实盘达到预设观察期 | 容量与冲击成本（Phase 11） | 回撤超过预设阈值 |

### 13.3 实盘与回测的偏差分析

实盘收益与同期回测收益的差异，应能被拆解解释：

$$
R_{\text{回测}} - R_{\text{实盘}} = \underbrace{\Delta_{\text{信号}}}_{\text{数据/时点差异}} + \underbrace{\Delta_{\text{延迟}}}_{\text{下单时价格已变}} + \underbrace{\Delta_{\text{冲击与价差}}}_{\text{成交价劣于假设}} + \underbrace{\Delta_{\text{未成交}}}_{\text{机会成本}} + \underbrace{\Delta_{\text{费用}}}_{\text{手续费等}}
$$

```python
def shortfall_breakdown(trades):
    """trades: 每笔成交的 dict，含 side(+1/-1)、qty、signal_px（回测假设价）、
    arrival_px（下单时价格）、fill_px（实际成交价）、fee"""
    delay = sum(t["side"] * (t["arrival_px"] - t["signal_px"]) * t["qty"] for t in trades)
    impact = sum(t["side"] * (t["fill_px"] - t["arrival_px"]) * t["qty"] for t in trades)
    fees = sum(t["fee"] for t in trades)
    return {"延迟成本": delay, "冲击与价差": impact, "费用": fees, "合计": delay + impact + fees}

trades = [
    {"side": +1, "qty": 1000, "signal_px": 10.00, "arrival_px": 10.02, "fill_px": 10.03, "fee": 5.0},
    {"side": -1, "qty": 1000, "signal_px": 10.50, "arrival_px": 10.49, "fill_px": 10.48, "fee": 10.2},
]
print(shortfall_breakdown(trades))
```

**通用验收标准**（LEARNING_PLAN）："实盘前至少 4 周模拟盘运行，且模拟盘与回测的偏差可解释"——上面的分解就是"可解释"的具体做法。

### 13.4 资金管理规则

- **小资金的含义**：亏完也不影响生活、能完整覆盖交易成本结构（如能买到最小交易单位）的金额。
- **预先写下规则**：放大条件（如运行满 N 周且回撤 < X%）和停止条件（如亏损达到 Y%），并且严格执行，不在亏损时临时改规则。
- **一次只改变一个变量**：放大资金时不同时上线新策略，否则出问题无法归因。

### 13.5 合规：程序化交易报告制度

国内对程序化交易实行**报告制度**，核心原则是"先报告、后交易"：

- **证监会《证券市场程序化交易管理规定（试行）》**：2024 年发布并于同年 10 月起施行，确立了证券市场程序化交易的报告、交易监测与监管框架，对高频交易有更严格的要求。
- **沪深北交易所程序化交易管理实施细则**：对报告内容、报告流程、异常交易监控、高频交易认定标准与差异化管理等作出具体规定。
- **报告的一般要求**（概述）：投资者通过其委托的证券公司向交易所报告账户基本情况、资金规模、策略类型、交易指令执行方式、最高申报速率、使用的交易软件等信息；信息变更需及时更新；高频交易需要报告更多信息（如服务器位置、系统测试与应急方案等）。
- **券商的角色**：证券公司负责对客户的程序化交易进行报告信息核查、接入管理和日常监控，开通程序化交易接口时券商会要求完成相应的报告与评估流程。
- **期货市场**：同样建立了程序化交易报告与管理制度，并与穿透式监管配合实施（见 7.2 节）。

> 具体的报告字段、阈值、流程和适用范围会随监管更新而调整，**以证监会、交易所与券商的最新规定为准**。上线前务必通过券商/期货公司确认自己的报告义务已完成。

此外，无论是否属于程序化交易，都要避免交易所明确禁止或重点监控的**异常交易行为**，如频繁申报撤单、虚假申报、自买自卖（对倒）、拉抬打压价格等。风控引擎中的报单频率、撤单次数和自成交检查（第 5.2 节）正是为此服务的。

### 13.6 上线前检查清单

```
┌──────────────────────────────────────────────────────────────────┐
│                        实盘上线检查清单                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  策略                                                             │
│  □ 回测含成本、样本外结果、参数尝试次数已记录                        │
│  □ 仿真盘 ≥ 4 周，与回测偏差可解释                                 │
│                                                                  │
│  系统                                                             │
│  □ 订单状态机覆盖拒单、部分成交、撤单、乱序与重复回报                 │
│  □ kill -9 后重启能从交易所恢复持仓与挂单                           │
│  □ 风控规则已配置并测试（胖手指、频率、日内亏损 Kill Switch）          │
│  □ 告警链路实测可达（手机能收到 P0）                                │
│  □ 日志、事件日志持久化，磁盘空间监控                                │
│                                                                  │
│  账户与合规                                                        │
│  □ 程序化交易报告已按券商/期货公司要求完成                           │
│  □ API 密钥最小权限、禁止提现、IP 白名单                            │
│  □ 资金规模、放大与停止规则已书面确定                                │
│                                                                  │
│  人                                                               │
│  □ 知道如何手动一键撤单、平仓（不依赖本系统的应急通道）                │
│  □ 交易时段内能及时响应告警                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 14. 最佳实践与常见陷阱

### 14.1 最佳实践

| 领域 | 实践 |
| --- | --- |
| 架构 | 策略只表达意图；所有订单经过风控和 OMS；网关只做翻译 |
| 状态 | 交易所是真相来源；订单/成交先持久化再处理；定时对账 |
| 订单 | 全局唯一 client_order_id；超时先查后发；成交按 trade_id 去重 |
| 风控 | 事前硬限制 + 事中 Kill Switch；自动暂停、人工恢复 |
| 运维 | 容器化 + 自动重启 + 业务级健康检查；统一时区与 NTP |
| 监控 | 四层监控；告警分级、去重；Dead Man's Switch |
| 测试 | 把断网、杀进程、重复回报、超时写成可重复的故障演练 |
| 上线 | 纸面 → 仿真 ≥ 4 周 → 小资金 → 按书面规则放大 |

### 14.2 常见陷阱

| 陷阱 | 后果 | 解决方案 |
| --- | --- | --- |
| 超时后直接重试下单 | 重复下单，仓位翻倍 | 同一 coid + 先查询确认 |
| 用订单状态推导持仓 | 部分成交、迟到成交导致持仓错误 | 持仓只由成交累计 |
| 信任本地状态 | 重启或丢回报后与交易所不一致 | 启动与定时对账 |
| 事件处理器阻塞 | 行情积压，订单回报延迟 | 重计算移出事件循环 |
| 时区错误 | 交易时段判断偏 8 小时 | 统一 `Asia/Shanghai` + 带时区时间 |
| 忽视夜盘交易日 | 统计与"今仓"判断错误 | 按交易日而非自然日 |
| 只监控进程存活 | 行情停滞而系统"假活" | 业务级心跳与行情新鲜度检查 |
| 告警太多 | 真正的 P0 被淹没 | 分级、去重、可操作 |
| 密钥写进代码仓库 | 资产被盗 | 环境变量/密钥管理，最小权限，禁止提现 |
| 自动恢复交易 | 未查明原因的故障反复触发 | Kill Switch 后人工确认恢复 |
| 仿真结果当实盘 | 仿真成交过于理想 | 小资金实盘验证真实滑点 |
| 忽视合规报告 | 违规风险 | 上线前通过券商/期货公司完成报告 |

---

## 15. 总结

### 15.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                          核心概念回顾                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  架构                                                             │
│  ├── 行情 → 策略 → 风控 → OMS → 执行 → 网关 → 交易所                │
│  ├── 统一数据模型 + 网关抽象（防腐层）                              │
│  └── 事件引擎：asyncio 单线程循环，跨线程用 call_soon_threadsafe     │
│                                                                  │
│  订单与风控                                                       │
│  ├── 状态机：提交中 → 未成交 → 部分成交 → 全成/撤销/拒单             │
│  ├── 持仓由成交推导；成交按 trade_id 去重                           │
│  └── 事前检查（胖手指/频率/资金）+ 事中 Kill Switch                  │
│                                                                  │
│  接口                                                             │
│  ├── vn.py：EventEngine + MainEngine + Gateway + App               │
│  ├── CTP：认证 → 登录 → 结算确认；开平今昨；SimNow 仿真              │
│  ├── A 股：QMT/miniQMT、PTrade、掘金等，经券商开通                  │
│  └── ccxt：统一接口，测试网 set_sandbox_mode(True)                  │
│                                                                  │
│  运维                                                             │
│  ├── 四层监控 + 分级告警                                           │
│  ├── 对账：交易所是真相；幂等：client_order_id + 先查后发            │
│  ├── 恢复：查询交易所 → 重建状态 → 对账 → 人工确认 → 恢复             │
│  └── Docker + 交易日历调度 + 时区/NTP                               │
│                                                                  │
│  上线与合规                                                       │
│  ├── 回测 → 纸面 → 仿真 ≥ 4 周 → 小资金 → 逐步放大                   │
│  └── 程序化交易报告制度：先报告后交易，以最新规定为准                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 15.2 学习检查对照

| 学习检查项 | 对应章节 |
| --- | --- |
| 订单状态机覆盖拒单、部分成交、撤单 | 第 4.1–4.3 节 |
| 重启后能从交易所状态恢复持仓 | 第 11.4、11.6 节 |
| 风控能在异常时自动暂停交易 | 第 5.3、5.4 节 |
| 了解 A 股程序化交易报备要求 | 第 13.5 节 |

### 15.3 与下一阶段的衔接

```
Phase 13 交易系统工程与实盘              Phase 14 综合项目
──────────────────────────              ─────────────────
交易系统（执行、风控、监控）   ───────▶   mini-quant 平台的执行层与监控层
网关抽象与统一数据模型        ───────▶   平台各层之间的接口设计
上线流程与检查清单            ───────▶   策略尽调与上线评审
4 周仿真盘运行记录           ───────▶   研究报告中的"实盘/模拟盘验证"章节
```

Phase 14 将把前 13 个阶段的所有模块——数据、因子、模型、组合、回测、执行、监控——整合成一个端到端的个人量化平台。

**推荐资源**：

- [vn.py 官网与社区](https://www.vnpy.com/)：源码是学习交易系统架构的好材料
- [SimNow](https://www.simnow.com.cn/)：CTP 仿真环境
- [ccxt 文档](https://docs.ccxt.com/)：统一交易接口与测试网说明
- 证监会、沪深北交易所官网：程序化交易相关规定与实施细则的最新版本
