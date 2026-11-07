# Phase 11: 市场微观结构与执行 - 核心概念深度解析

> **学习目标**：理解订单簿与交易成本的来源，掌握执行算法与做市基础
>
> **前置知识**：Phase 4（回测与交易成本建模）、Phase 3（概率统计、随机游走）
>
> **阅读建议**：每个概念按「直觉/比喻 → 公式 → 代码 → 实际意义与陷阱」展开。公式看不懂时先看比喻和代码，代码跑通后再回头看公式。

---

## 目录

1. [引言：为什么要研究微观结构](#1-引言为什么要研究微观结构)
2. [订单簿与撮合规则](#2-订单簿与撮合规则)
3. [Tick 与 Level-2 数据](#3-tick-与-level-2-数据)
4. [买卖价差：三个组成部分](#4-买卖价差三个组成部分)
5. [市场冲击与交易成本全景](#5-市场冲击与交易成本全景)
6. [实时订单簿：ccxt 与本地簿维护](#6-实时订单簿ccxt-与本地簿维护)
7. [高频因子：订单不平衡与成交量分布](#7-高频因子订单不平衡与成交量分布)
8. [执行算法：TWAP / VWAP / POV](#8-执行算法twap--vwap--pov)
9. [Almgren-Chriss 最优执行](#9-almgren-chriss-最优执行)
10. [做市原理：库存风险与逆向选择](#10-做市原理库存风险与逆向选择)
11. [Avellaneda-Stoikov 做市模型](#11-avellaneda-stoikov-做市模型)
12. [高频交易基础设施与延迟](#12-高频交易基础设施与延迟)
13. [最佳实践与常见陷阱](#13-最佳实践与常见陷阱)
14. [总结](#14-总结)

---

## 1. 引言：为什么要研究微观结构

### 1.1 回测里的"成交"是一个谎言

在 Phase 4 的回测里，我们经常写这样一行：`以收盘价成交 N 股`。但真实世界没有"收盘价成交"这个按钮。你下的每一笔单都要进入交易所的**订单簿（Order Book）**，和别人的订单撮合，而你的订单本身会改变价格。

```
┌─────────────────────────────────────────────────────────────────┐
│                    回测假设 vs 真实执行                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   回测：  信号 ──▶ 以 close 价成交全部数量 ──▶ 收益               │
│                                                                 │
│   真实：  信号 ──▶ 拆单 ──▶ 排队 ──▶ 部分成交 ──▶ 价格被推高      │
│                     │        │          │            │          │
│                     ▼        ▼          ▼            ▼          │
│                   延迟     队列位置   未成交风险    市场冲击       │
│                                                                 │
│   回测收益 − 真实收益 = 实施缺口（Implementation Shortfall）      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**市场微观结构（Market Microstructure）** 研究的就是：价格是如何在一笔笔订单的交互中形成的，交易成本从哪里来，以及如何聪明地下单。

### 1.2 软件工程师视角的类比

| 微观结构概念 | 工程类比 |
| --- | --- |
| 订单簿 | 一个按价格分桶、桶内 FIFO 的优先队列 |
| 撮合引擎 | 单线程状态机，按事件顺序处理 `submit/cancel` |
| 限价单 | 挂在队列里的"预约请求" |
| 市价单 | "立即执行，不管价格"的请求，吃掉队列里的流动性 |
| 价差 | 中间商（做市商）的服务费 |
| 市场冲击 | 大请求把缓存打穿后的"雪崩"代价 |
| 执行算法 | 限流 + 调度器，把大任务切成小任务平滑发送 |
| 延迟 | 网络 RTT，决定你能否抢到队列前面的位置 |

### 1.3 本阶段地图

```
┌──────────────────────────────────────────────────────────────────┐
│                       Phase 11 知识地图                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  第 18 周：微观结构（理解成本从哪来）                              │
│    01 订单簿与撮合 ─▶ 02 Tick/L2 数据 ─▶ 03 价差与冲击             │
│                     ─▶ 04 ccxt 实时订单簿 ─▶ 05 高频因子           │
│                                                                  │
│  第 19 周：执行与做市（如何降低成本 / 如何赚价差）                  │
│    06 TWAP/VWAP/POV ─▶ 07 Almgren-Chriss                         │
│    08 做市基础 ─▶ 09 Avellaneda-Stoikov ─▶ 10 HFT 基础设施         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 订单簿与撮合规则

> 对应课程：`01-order-book-basics.py`

### 2.1 订单簿是什么

**直觉**：订单簿就是一个"挂单墙"。左边是想买的人（Bid，买盘），右边是想卖的人（Ask/Offer，卖盘）。每个人在墙上写下"我愿意以 X 价格买/卖 Y 股"。

```
                    某股票订单簿（五档）

        买盘 Bid                     卖盘 Ask
  ┌──────────┬────────┐      ┌──────────┬────────┐
  │  价格     │ 数量    │      │  价格     │ 数量    │
  ├──────────┼────────┤      ├──────────┼────────┤
  │  10.00   │  500   │ 买一  │  10.01   │  300   │ 卖一
  │   9.99   │  1200  │ 买二  │  10.02   │  800   │ 卖二
  │   9.98   │  2000  │ 买三  │  10.03   │  1500  │ 卖三
  │   9.97   │  900   │ 买四  │  10.04   │  600   │ 卖四
  │   9.96   │  3000  │ 买五  │  10.05   │  2200  │ 卖五
  └──────────┴────────┘      └──────────┴────────┘

  最优买价 best bid = 10.00     最优卖价 best ask = 10.01
  中间价 mid = (10.00 + 10.01) / 2 = 10.005
  价差 spread = 10.01 − 10.00 = 0.01（一个最小变动单位 tick）
```

几个关键定义：

- **Best Bid / Best Ask**：最高买价 $P^b$ 和最低卖价 $P^a$，统称 **BBO**（Best Bid and Offer）。
- **中间价（Mid Price）**：$M = \dfrac{P^a + P^b}{2}$，常被当作资产"当前公允价值"的近似。
- **报价价差（Quoted Spread）**：$S = P^a - P^b$。相对价差 $S/M$ 常用基点（bp，1bp = 0.01%）表示。
- **深度（Depth）**：某个价位或某几档上的挂单总量。深度越大，大单越不容易把价格推开。
- **最小变动单位（Tick Size）**：A 股股票为 0.01 元，ETF 等基金为 0.001 元。

### 2.2 撮合规则：价格优先、时间优先

交易所的连续竞价按 **价格优先、时间优先（Price-Time Priority）** 撮合：

1. **价格优先**：买单价格越高越优先，卖单价格越低越优先。
2. **时间优先**：同一价格上，先到的订单先成交（FIFO）。
3. **成交价**：新来的订单与簿上已有订单成交时，成交价为**簿上已有订单（被动方）的价格**。

**比喻**：餐厅排队取号。出价高的客人去 VIP 通道（价格优先），同一通道内先到先得（时间优先）。

```
例：卖一 10.01 上依次挂着  A(100) → B(200)   （A 先到）
    现在来了一笔 "买 250 股 @10.02" 的限价单

    撮合过程：
      1. 对手方最优价 10.01 ≤ 10.02，可成交
      2. 先与 A 成交 100 股 @10.01（被动方价格）
      3. 再与 B 成交 150 股 @10.01
      4. B 剩余 50 股继续挂在 10.01
      5. 买单全部成交，不会进入簿

    注意：买方报价 10.02，但成交价是 10.01 ——"价格改善"归主动方。
```

**主动方 vs 被动方**：

| 角色 | 英文 | 行为 | 成本 |
| --- | --- | --- | --- |
| 主动方 / 吃单方 | Taker / Aggressor | 用可立即成交的价格下单，消耗流动性 | 付出半个价差，加密市场通常付更高手续费 |
| 被动方 / 挂单方 | Maker / Passive | 挂在簿上等待别人来吃，提供流动性 | 赚取半个价差，但可能不成交或被"聪明钱"吃掉 |

### 2.3 订单类型

| 订单类型 | 英文 | 含义 | 典型用途 |
| --- | --- | --- | --- |
| 限价单 | Limit Order | 指定价格，只在该价或更优价成交，未成交部分挂单 | 最常用；控制价格 |
| 市价单 | Market Order | 不指定价格，按对手方最优价逐档成交 | 追求确定成交 |
| 立即成交剩余撤销 | IOC (Immediate-or-Cancel) | 能成交多少成交多少，剩余立即撤销 | 扫单而不留挂单 |
| 全额成交或撤销 | FOK (Fill-or-Kill) | 必须全部成交，否则全部撤销 | 需要整笔成交的套利腿 |
| 只做 Maker | Post-Only | 若会立即成交则拒绝/撤销，保证只挂单 | 做市，保证 maker 费率 |
| 止损单 | Stop Order | 价格触及触发价后转为市价/限价单 | 止损（A 股交易所层面一般不支持，常由券商/客户端条件单实现） |
| 冰山单 | Iceberg | 只显示一部分数量，成交后自动补充 | 隐藏大单意图（多见于期货与加密市场） |

**A 股市价申报的具体类型**（交易所规则，细节以沪深交易所最新业务规则为准）：

| 交易所 | 支持的市价申报类型（连续竞价阶段） |
| --- | --- |
| 上交所 | 最优五档即时成交剩余撤销、最优五档即时成交剩余转限价；科创板等板块还支持本方最优、对手方最优等 |
| 深交所 | 对手方最优价格、本方最优价格、即时成交剩余撤销、最优五档即时成交剩余撤销、全额成交或撤销 |

**A 股的几条特殊约束**：

- **涨跌停**：超出涨跌停价格的申报为无效申报。主板普通股票一般为 ±10%，科创板、创业板为 ±20%，ST 等股票另有规定。
- **有效申报价格范围（"价格笼子"）**：连续竞价阶段，限价买入申报价格不得过度高于"买入基准价格"，卖出申报不得过度低于"卖出基准价格"（例如以基准价的 ±2% 为界，低价股等另有规定）。超出范围的申报不会立即进入撮合，具体规则以交易所最新规定为准。
- **申报数量**：买入一般须为 100 股整数倍（科创板另有规定），卖出零股可一次性申报。
- **T+1**：当日买入的股票次日才能卖出——这直接限制了 A 股股票上的日内高频策略。

### 2.4 A 股交易时段与集合竞价

```
                       A 股（沪深主板）一个交易日
  ─────────────────────────────────────────────────────────────────▶ 时间
  9:15        9:20       9:25    9:30           11:30  13:00        14:57    15:00
   │ 开盘集合竞价 │          │       │   连续竞价      │ 午休 │  连续竞价    │ 收盘集合 │
   │  可撤单      │ 不可撤单  │ 等待  │   (撮合)       │      │   (撮合)    │ 竞价     │
   └─────────────┴──────────┴───────┴────────────────┘      └─────────────┴─────────┘
                             ▲
                     9:25 一次性撮合，产生开盘价
```

- **9:15–9:25 开盘集合竞价**：投资者提交申报，但不连续撮合；9:20–9:25 期间交易所不接受撤单。9:25 统一撮合一次，产生开盘价。
- **9:30–11:30、13:00–14:57 连续竞价**：按价格优先、时间优先逐笔撮合。
- **14:57–15:00 收盘集合竞价**：再次集中撮合，产生收盘价（期间不接受撤单）。
- 科创板、创业板还有盘后固定价格交易等安排。具体时段以交易所规则为准。

**集合竞价的定价原则**（以上交所规则为例）：选择满足以下条件的价格作为成交价：

1. 在该价格下**成交量最大**；
2. 高于该价格的买入申报与低于该价格的卖出申报全部成交；
3. 与该价格相同的买方或卖方至少有一方全部成交。

若有多个价格同时满足，上交所取使未成交量最小的价格，仍有多个时取中间价；深交所则取距前收盘价最近的价格。

**直觉**：集合竞价像一场"拍卖"：把所有人的买卖意愿画成供给曲线和需求曲线，交点就是开盘价。

```
数量
 ▲
 │╲  累计买量（价格 ≥ p 的买单总量，随 p 升高而减少）
 │ ╲             ╱ 累计卖量（价格 ≤ p 的卖单总量，随 p 升高而增加）
 │  ╲          ╱
 │   ╲       ╱
 │    ╲    ╱
 │     ╲ ╱   ← 交点附近：min(买量, 卖量) 最大 → 集合竞价成交价
 │     ╱ ╲
 │   ╱     ╲
 └──────────────────────▶ 价格 p
```

代码示例（简化版，平局规则采用"距前收盘价最近"）：

```python
def call_auction_price(bids, asks, prev_close):
    """bids/asks: [(price, qty), ...]，返回 (成交价, 成交量)"""
    prices = sorted({p for p, _ in bids} | {p for p, _ in asks})
    best = None
    for p in prices:
        buy = sum(q for bp, q in bids if bp >= p)   # 愿意以 ≥p 买的总量
        sell = sum(q for ap, q in asks if ap <= p)  # 愿意以 ≤p 卖的总量
        vol = min(buy, sell)
        # 排序键：成交量最大 > 未成交量最小 > 离前收盘最近
        key = (vol, -abs(buy - sell), -abs(p - prev_close))
        if best is None or key > best[0]:
            best = (key, p, vol)
    return best[1], best[2]

bids = [(10.05, 300), (10.02, 500), (10.00, 800)]
asks = [(9.98, 400), (10.01, 600), (10.04, 700)]
print(call_auction_price(bids, asks, prev_close=10.00))  # (10.01, 800)
# 10.01 与 10.02 成交量、未成交量都相同，按"离前收盘最近"取 10.01
```

**实际意义**：

- 开盘和收盘价来自集合竞价，很多指数基金和组合会在收盘集合竞价中交易（"按收盘价成交"实际上需要参与收盘集合竞价）。
- 集合竞价期间的虚拟成交价与未匹配量是重要的盘前信号，但 9:15–9:20 可撤单阶段的挂单可能是"虚假意图"。

### 2.5 代码：最小撮合引擎

用整数 tick 表示价格（避免浮点数作为字典 key 的精度问题），每个价位一个 FIFO 队列：

```python
from collections import deque
from dataclasses import dataclass

@dataclass
class Order:
    oid: int
    side: str   # "B" 买 / "S" 卖
    price: int  # 以 tick 为单位的整数价格，如 1001 表示 10.01 元
    qty: int

class OrderBook:
    def __init__(self):
        self.bids: dict[int, deque] = {}
        self.asks: dict[int, deque] = {}

    def submit(self, o: Order) -> list[tuple]:
        """限价单撮合，返回成交列表 (被动方id, 主动方id, 价格, 数量)"""
        trades = []
        opp = self.asks if o.side == "B" else self.bids
        mine = self.bids if o.side == "B" else self.asks
        while o.qty > 0 and opp:
            best = min(opp) if o.side == "B" else max(opp)
            crosses = best <= o.price if o.side == "B" else best >= o.price
            if not crosses:
                break
            queue = opp[best]
            resting = queue[0]                      # 时间优先：队首先成交
            fill = min(o.qty, resting.qty)
            trades.append((resting.oid, o.oid, best, fill))  # 成交价 = 被动方价格
            o.qty -= fill
            resting.qty -= fill
            if resting.qty == 0:
                queue.popleft()
                if not queue:
                    del opp[best]
        if o.qty > 0:                               # 剩余部分挂单，排到同价位队尾
            mine.setdefault(o.price, deque()).append(o)
        return trades

book = OrderBook()
book.submit(Order(1, "S", 1001, 100))
book.submit(Order(2, "S", 1001, 200))
book.submit(Order(3, "S", 1002, 500))
print(book.submit(Order(4, "B", 1002, 450)))
# [(1, 4, 1001, 100), (2, 4, 1001, 200), (3, 4, 1002, 150)]
```

> **工程提示**：真实撮合引擎用"价格 → 队列"的有序映射（红黑树、跳表或按 tick 索引的数组）来做到 $O(1)$ 或 $O(\log n)$ 的最优价查找，上面 `min(opp)` 是 $O(n)$ 的教学写法。

---

## 3. Tick 与 Level-2 数据

> 对应课程：`02-tick-level2-data.py`

### 3.1 数据层级

| 数据类型 | 内容 | 频率（A 股典型） | 用途 |
| --- | --- | --- | --- |
| 日线 / 分钟线 | OHLCV 聚合 | 日 / 分钟 | 中低频策略 |
| Level-1 快照 | 最新价、成交量、五档买卖价量 | 约每 3 秒一次 | 价差、简单盘口因子 |
| Level-2 快照 | 十档买卖价量、委托笔数等 | 约每 3 秒一次 | 盘口深度因子 |
| 逐笔成交（Tick-by-Tick Trades） | 每一笔成交的价格、数量、买卖方向/双方订单号 | 事件驱动 | 主动买卖、成交量分布 |
| 逐笔委托（Tick-by-Tick Orders） | 每一笔新申报（及撤单）的价格、数量、方向 | 事件驱动 | 订单流重建、OFI |
| 委托队列 | 买一/卖一价位上前若干笔挂单的数量 | 随快照 | 队列位置估计 |

**关键认知**：A 股的"Tick"数据通常指 **3 秒快照**，而不是每笔事件。3 秒内可能发生了几十笔成交和撤单，快照只是"拍照"。真正的事件级数据是 Level-2 的逐笔委托与逐笔成交，需要从券商或数据商付费获取。

```
时间轴上的事件流 vs 快照

逐笔事件： ─●─●●──●─●●●──●─●───●●─●──●●●─●──●─●●─▶
快照：     ──────────📷──────────📷──────────📷──────▶
                     3s          3s          3s

快照之间发生的事情（新单、撤单、成交顺序）在快照中不可见
```

### 3.2 主动买卖判别

逐笔成交中最重要的信息之一是**这笔成交是买方主动还是卖方主动**。部分交易所数据会直接给出买卖方向标志；没有时常用以下算法推断：

- **Tick Rule**：成交价高于上一笔成交价 → 主动买；低于 → 主动卖；相同 → 沿用上一笔方向。
- **Quote Rule**：成交价高于当时中间价 → 主动买；低于 → 主动卖。
- **Lee-Ready 算法**：先用 Quote Rule，成交价恰好等于中间价时再用 Tick Rule。

```python
import numpy as np

def lee_ready(trade_px, mid_px):
    """trade_px: 成交价数组；mid_px: 成交时刻的中间价数组。返回 +1 主动买 / -1 主动卖"""
    side = np.sign(trade_px - mid_px)
    for i in range(len(side)):
        if side[i] == 0:                          # 恰在中间价：用 tick rule
            if i > 0 and trade_px[i] != trade_px[i - 1]:
                side[i] = np.sign(trade_px[i] - trade_px[i - 1])
            else:
                side[i] = side[i - 1] if i > 0 else 1
    return side.astype(int)

px  = np.array([10.01, 10.01, 10.00, 10.005, 10.01])
mid = np.array([10.005, 10.005, 10.005, 10.005, 10.005])
print(lee_ready(px, mid))  # [ 1  1 -1  1  1]，第 4 笔在中间价，比上一笔高 → 主动买
```

### 3.3 从快照计算常用盘口指标

```python
import pandas as pd

snap = pd.DataFrame({
    "time":   pd.to_datetime(["09:30:00", "09:30:03", "09:30:06"]),
    "bid1":   [10.00, 10.00, 10.01],
    "ask1":   [10.01, 10.02, 10.02],
    "bid1_v": [500, 300, 800],
    "ask1_v": [300, 900, 400],
})
snap["mid"] = (snap.bid1 + snap.ask1) / 2
snap["spread_bp"] = (snap.ask1 - snap.bid1) / snap.mid * 1e4
snap["imbalance"] = (snap.bid1_v - snap.ask1_v) / (snap.bid1_v + snap.ask1_v)
print(snap[["time", "mid", "spread_bp", "imbalance"]])
```

### 3.4 Tick 数据处理的陷阱

| 陷阱 | 说明 | 处理 |
| --- | --- | --- |
| 累计量 vs 增量 | 快照中的成交量常为当日累计值 | `diff()` 得到区间成交量，注意开盘第一条 |
| 集合竞价数据混入 | 9:25 的成交是集合竞价一次性成交 | 单独处理开盘/收盘竞价数据 |
| 时间戳含义 | 交易所时间、行情服务器接收时间、本地接收时间不同 | 明确使用哪个时钟，回测只能用"当时已收到"的数据 |
| 沪深格式差异 | 两个交易所逐笔数据的字段、撤单表示方式不同 | 先做统一的数据模型（Phase 13 网关思想） |
| 涨跌停时的盘口 | 涨停时卖盘为空，价差无定义 | 过滤或特殊标记 |
| 数据量巨大 | 全市场逐笔一天可达数十 GB | Parquet 分区 + 列式查询（Phase 1） |

---

## 4. 买卖价差：三个组成部分

> 对应课程：`03-spread-and-impact.py`

### 4.1 直觉：价差是谁的收入

**比喻**：机场的货币兑换柜台。买入美元 7.30，卖出美元 7.10，中间 0.20 就是柜台的收入。柜台为什么要收这么多？因为它：

1. 要付房租和员工工资（**订单处理成本**）；
2. 手里压着一堆外币，汇率波动会让它亏钱（**库存成本**）；
3. 偶尔会碰上知道明天汇率要大变的"内行人"来换汇，跟他们交易必然亏钱（**逆向选择成本**）。

这正是经典微观结构理论对价差的三分法：

$$
\text{Spread} = \underbrace{\text{订单处理成本}}_{\text{Order Processing}} + \underbrace{\text{库存成本}}_{\text{Inventory}} + \underbrace{\text{逆向选择成本}}_{\text{Adverse Selection}}
$$

| 组成部分 | 来源 | 何时变大 | 代表理论 |
| --- | --- | --- | --- |
| 订单处理成本 | 手续费、系统、人力等固定/变动成本 | 交易所费用高、竞争少 | Roll (1984) |
| 库存成本 | 做市商持有头寸承担价格波动风险 | 波动率高、做市商风险厌恶高、头寸大 | Ho & Stoll (1981) |
| 逆向选择成本 | 与掌握私有信息的交易者交易会系统性亏损 | 信息不对称强（财报前、小盘股、新闻时） | Glosten-Milgrom (1985)、Kyle (1985) |

### 4.2 逆向选择：Glosten-Milgrom 数值例子

假设股票真实价值 $V$ 以各 50% 概率为 101 或 99。市场上有比例 $\mu = 0.2$ 的**知情交易者**（知道 $V$），其余是随机买卖的**噪声交易者**。做市商不知道谁是谁，但必须给出报价使得自己不亏钱（竞争下期望利润为零）。

做市商的卖价 $P^a$ 应等于"有人来买时" $V$ 的条件期望：

$$
P^a = E[V \mid \text{买单}], \qquad P^b = E[V \mid \text{卖单}]
$$

计算：

- 若 $V = 101$，知情者全部买入，噪声交易者一半买入：$P(\text{买} \mid V=101) = \mu + \frac{1-\mu}{2} = 0.6$
- 若 $V = 99$，只有噪声交易者买入：$P(\text{买} \mid V=99) = \frac{1-\mu}{2} = 0.4$
- 由贝叶斯公式：$P(V=101 \mid \text{买}) = \dfrac{0.6 \times 0.5}{0.6 \times 0.5 + 0.4 \times 0.5} = 0.6$
- $P^a = 0.6 \times 101 + 0.4 \times 99 = 100.2$，对称地 $P^b = 99.8$

即使没有任何手续费和库存风险，价差也有 0.4。**知情交易者比例越高，价差越大**。每笔成交都在向市场"泄露"信息，价格随之更新——这就是"价格发现"。

```python
def glosten_milgrom(v_high, v_low, p_high, mu):
    """返回 (bid, ask)；p_high 为 V=v_high 的先验概率，mu 为知情交易者比例"""
    p_buy_h, p_buy_l = mu + (1 - mu) / 2, (1 - mu) / 2
    post_h_buy = p_buy_h * p_high / (p_buy_h * p_high + p_buy_l * (1 - p_high))
    p_sell_h, p_sell_l = (1 - mu) / 2, mu + (1 - mu) / 2
    post_h_sell = p_sell_h * p_high / (p_sell_h * p_high + p_sell_l * (1 - p_high))
    ask = post_h_buy * v_high + (1 - post_h_buy) * v_low
    bid = post_h_sell * v_high + (1 - post_h_sell) * v_low
    return bid, ask

for mu in [0.0, 0.2, 0.5]:
    bid, ask = glosten_milgrom(101, 99, 0.5, mu)
    print(f"知情比例 {mu:.1f}: bid={bid:.2f} ask={ask:.2f} spread={ask - bid:.2f}")
```

### 4.3 价差的几种度量

报价价差只告诉你"挂出来的价格"，真正的成本要看成交。设第 $t$ 笔成交价为 $P_t$，成交前中间价为 $M_t$，方向 $q_t \in \{+1, -1\}$（+1 主动买），$\Delta$ 为一段时间后（如 5 分钟）：

$$
\text{有效价差（Effective Spread）} = 2 q_t (P_t - M_t)
$$

$$
\text{实现价差（Realized Spread）} = 2 q_t (P_t - M_{t+\Delta})
$$

$$
\text{价格冲击（Price Impact）} = 2 q_t (M_{t+\Delta} - M_t)
$$

三者满足恒等式：**有效价差 = 实现价差 + 价格冲击**。

- **有效价差**：主动方实际付出的成本（相对成交时的中间价）。
- **实现价差**：被动方（做市商）扣除"价格随后朝不利方向移动"之后真正赚到的钱，对应订单处理与库存补偿。
- **价格冲击**：成交后中间价永久性的移动，对应**逆向选择**部分——如果你买完价格就涨，说明你的买单携带了信息（或被认为携带了信息）。

```python
import numpy as np

def spread_decomposition(trade_px, side, mid_before, mid_after):
    eff = 2 * side * (trade_px - mid_before)
    real = 2 * side * (trade_px - mid_after)
    impact = 2 * side * (mid_after - mid_before)
    return eff.mean(), real.mean(), impact.mean()

trade_px   = np.array([10.01, 10.00, 10.02, 10.01])
side       = np.array([1, -1, 1, 1])
mid_before = np.array([10.005, 10.005, 10.015, 10.005])
mid_after  = np.array([10.010, 10.000, 10.020, 10.010])
eff, real, impact = spread_decomposition(trade_px, side, mid_before, mid_after)
print(f"有效={eff:.4f} 实现={real:.4f} 冲击={impact:.4f}")  # 有效 = 实现 + 冲击
```

### 4.4 Roll 模型：只用成交价估计价差

**直觉**：即使真实价值不变，成交价也会在买价和卖价之间来回"弹跳"（Bid-Ask Bounce）。这种弹跳让相邻两次价格变化**负相关**，负相关的强度就反映了价差大小。

Roll (1984) 模型假设成交价 $P_t = V_t + \frac{S}{2} q_t$，其中 $V_t$ 是随机游走的真实价值，$q_t$ 等概率取 ±1 且相互独立。可以推出：

$$
\text{Cov}(\Delta P_t, \Delta P_{t-1}) = -\frac{S^2}{4} \quad \Longrightarrow \quad S = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}
$$

- $\Delta P_t = P_t - P_{t-1}$：相邻成交价变化
- $S$：价差

```python
import numpy as np

rng = np.random.default_rng(42)
n, true_spread = 10_000, 0.02
value = 10 + np.cumsum(rng.normal(0, 0.003, n))      # 真实价值：随机游走
side = rng.choice([-1, 1], n)                         # 随机的买卖方向
price = value + side * true_spread / 2                # 成交价在 bid/ask 之间弹跳

dp = np.diff(price)
cov = np.cov(dp[1:], dp[:-1])[0, 1]
roll = 2 * np.sqrt(-cov) if cov < 0 else np.nan       # 协方差为正时模型不适用
print(f"真实价差 {true_spread:.4f}，Roll 估计 {roll:.4f}")
```

**陷阱**：真实数据中协方差可能为正（有动量或信息驱动的连续交易），此时 Roll 估计无定义。它更适合"没有盘口数据、只有成交价"的场景（如历史日线粗估流动性）。

---

## 5. 市场冲击与交易成本全景

> 对应课程：`03-spread-and-impact.py`

### 5.1 临时冲击 vs 永久冲击

**比喻**：你跳进游泳池，水面立刻涌起一个浪（临时冲击），浪很快退去，但整个池子的水位因为你的体积永久上升了一点（永久冲击）。

```
价格
 ▲
 │            ┌──╮  ← 临时冲击：大单吃掉多档流动性，价格瞬间被推高
 │            │   ╲
 │            │    ╲___________  ← 永久冲击：市场"学习"到有人在买，价格中枢上移
 │            │
 │────────────┘  ← 下单前价格
 └──────────────────────────────▶ 时间
              ▲
            你的大买单
```

| 类型 | 原因 | 持续性 | 影响谁 |
| --- | --- | --- | --- |
| 临时冲击（Temporary Impact） | 吃掉对手盘多个价位的流动性 | 短暂，盘口随后恢复 | 只影响当前这笔交易 |
| 永久冲击（Permanent Impact） | 订单流携带信息，市场更新对价值的估计 | 长期 | 影响后续所有交易 |

### 5.2 Kyle 的 λ：线性冲击

Kyle (1985) 模型中，价格变化与净订单流呈线性关系：

$$
\Delta P = \lambda \cdot Q
$$

- $\Delta P$：价格变化
- $Q$：净订单流（买入量 − 卖出量）
- $\lambda$：**Kyle's Lambda**，衡量市场"深度"的倒数。$\lambda$ 越大，同样的订单推动价格越多，流动性越差。

实践中可以用分钟级数据回归 $\Delta P_t = \lambda \cdot \text{净主动买量}_t + \varepsilon_t$ 来估计。

### 5.3 平方根法则（Square-Root Law）

大量实证研究发现，对于在一段时间内执行完毕的大单（"母单" Meta-order），总冲击成本与订单规模的**平方根**成正比，而非线性：

$$
\frac{\Delta P}{P} \approx Y \cdot \sigma \cdot \sqrt{\frac{Q}{V}}
$$

- $\Delta P / P$：相对冲击成本（如以 bp 计）
- $\sigma$：资产的**日**波动率（如 2% 写作 0.02）
- $Q$：订单总量（股数）
- $V$：日成交量（股数），$Q/V$ 称为**参与度**
- $Y$：经验常数，数量级为 1（不同市场和研究中常见取值在 0.5 到 1 之间，需要用自己的成交数据校准）

**直觉**：订单规模翻 4 倍，冲击成本只翻 2 倍。这是因为市场会对持续的订单流"补充流动性"——有其他交易者看到价格偏离后进来提供对手盘。

**数值例子**：日波动率 2%，你要买入日成交量的 1%，取 $Y = 1$：

$$
\frac{\Delta P}{P} \approx 1 \times 0.02 \times \sqrt{0.01} = 0.002 = 20 \text{ bp}
$$

如果买入日成交量的 10%，冲击约为 $0.02 \times \sqrt{0.1} \approx 63$ bp——这已经远大于很多 alpha 信号的预期收益。

```python
import numpy as np

def sqrt_impact_bp(order_shares, adv_shares, daily_vol, Y=1.0):
    """平方根法则冲击成本（单位：bp）"""
    return Y * daily_vol * np.sqrt(order_shares / adv_shares) * 1e4

for pct in [0.001, 0.01, 0.05, 0.1, 0.2]:
    print(f"参与度 {pct:>5.1%}: 冲击约 {sqrt_impact_bp(pct, 1.0, 0.02):5.1f} bp")
```

**实际意义——策略容量（Capacity）**：策略资金越大，单笔订单的参与度越高，冲击成本越大，最终吃掉全部 alpha。用平方根法则可以粗估"资金规模多大时策略净收益归零"，这是 Phase 14 策略尽调的必答题。

### 5.4 交易成本全景

```
┌──────────────────────────────────────────────────────────────────┐
│                      交易成本"冰山"                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   显性成本（看得见，水面以上）                                    │
│     • 佣金（券商协商）                                            │
│     • 印花税（A 股卖出单边征收，2023-08-28 起为 0.05%）            │
│     • 过户费、交易所规费等                                        │
│  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  │
│   隐性成本（看不见，水面以下，往往更大）                          │
│     • 买卖价差：主动成交付出半个价差                              │
│     • 市场冲击：大单推动价格                                      │
│     • 延迟成本：从决策到下单期间价格的漂移                        │
│     • 机会成本：未成交部分错过的收益                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

> 费率会调整，回测与实盘请以券商与监管的最新标准为准。

### 5.5 实施缺口（Implementation Shortfall）

Perold (1988) 提出用**实施缺口**统一度量执行质量：以**决策时刻的价格**（Decision Price / Arrival Price）为基准，比较"纸面组合"和"实际组合"的收益差。

对买单，平均成交价为 $\bar{P}_{\text{exec}}$，到达价为 $P_0$：

$$
\text{IS (bp)} = \frac{\bar{P}_{\text{exec}} - P_0}{P_0} \times 10^4 \;+\; \text{未成交部分的机会成本} \;+\; \text{显性费用}
$$

卖单符号相反。IS 是机构评价执行算法最常用的指标，因为它同时惩罚"冲击太大"和"拖太久没成交"。

---

## 6. 实时订单簿：ccxt 与本地簿维护

> 对应课程：`04-realtime-orderbook-ccxt.py`

加密货币交易所 7×24 运行、API 公开免费，是学习实时订单簿的最佳场所。

### 6.1 REST 快照 vs WebSocket 增量

| 方式 | 原理 | 优点 | 缺点 |
| --- | --- | --- | --- |
| REST 轮询 `fetch_order_book` | 每次请求拿一份完整快照 | 简单、无状态 | 有频率限制，延迟高，两次之间信息丢失 |
| WebSocket 增量 | 先拿快照，再持续接收价位变化（diff） | 低延迟、信息完整 | 需要自己维护本地订单簿，处理丢包与重连 |

### 6.2 本地订单簿维护的通用流程

```
┌────────────────────────────────────────────────────────────────┐
│               本地订单簿同步流程（通用思路）                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 订阅 WebSocket 增量流，先把消息缓存起来                      │
│  2. 通过 REST 获取深度快照，记下快照的序列号 lastUpdateId        │
│  3. 丢弃缓存中序列号 ≤ lastUpdateId 的增量                       │
│  4. 从第一条"覆盖 lastUpdateId+1"的增量开始依次应用               │
│  5. 每条增量：数量为 0 → 删除该价位；否则 → 覆盖该价位数量        │
│  6. 检查序列号连续性：一旦出现缺口 → 丢弃本地簿，回到第 1 步       │
│                                                                │
│  具体字段与校验规则因交易所而异，以交易所 API 文档为准            │
└────────────────────────────────────────────────────────────────┘
```

**注意**：增量消息里的数量是该价位的**新总量**（覆盖），不是变化量（相加）。这是新手最常见的 bug 之一。

```python
class LocalBook:
    def __init__(self, snapshot_bids, snapshot_asks, last_id):
        self.bids = {p: q for p, q in snapshot_bids}
        self.asks = {p: q for p, q in snapshot_asks}
        self.last_id = last_id

    def apply(self, first_id, final_id, bid_updates, ask_updates):
        if final_id <= self.last_id:
            return                                  # 过期消息，丢弃
        if first_id > self.last_id + 1:
            raise RuntimeError("序列号缺口，需要重新同步快照")
        for side, updates in ((self.bids, bid_updates), (self.asks, ask_updates)):
            for p, q in updates:
                if q == 0:
                    side.pop(p, None)               # 数量为 0 表示删除价位
                else:
                    side[p] = q                     # 覆盖，不是累加
        self.last_id = final_id

    def bbo(self):
        return max(self.bids), min(self.asks)
```

### 6.3 用 ccxt 获取订单簿

[ccxt](https://github.com/ccxt/ccxt) 用统一接口封装了上百家加密交易所。REST 接口在 `ccxt` 中，WebSocket 接口在 `ccxt.pro` 中（新版本已合并在同一个包里）：

```python
import ccxt

ex = ccxt.binance()
ob = ex.fetch_order_book("BTC/USDT", limit=10)
best_bid, best_ask = ob["bids"][0][0], ob["asks"][0][0]
print(f"bid={best_bid} ask={best_ask} spread_bp={(best_ask - best_bid) / best_bid * 1e4:.2f}")
```

```python
import asyncio
import ccxt.pro as ccxtpro

async def stream(symbol="BTC/USDT", n=5):
    ex = ccxtpro.binance()
    try:
        for _ in range(n):
            ob = await ex.watch_order_book(symbol)   # ccxt 内部维护本地簿
            print(ob["datetime"], ob["bids"][0], ob["asks"][0])
    finally:
        await ex.close()

asyncio.run(stream())
```

**陷阱**：

- 交易所 API 有频率限制（Rate Limit），超限会被封 IP，ccxt 中设置 `enableRateLimit=True`（默认开启）。
- 部分地区网络无法直连境外交易所，课程代码应提供离线快照回放模式。
- 不同交易所的 `limit` 可选值不同，深度档位也不同。

---

## 7. 高频因子：订单不平衡与成交量分布

> 对应课程：`05-high-frequency-factors.py`

### 7.1 订单簿不平衡（Order Book Imbalance, OBI）

**直觉**：如果买一挂了 10000 股、卖一只有 500 股，卖一很快会被吃光，价格大概率上移。就像一个天平，哪边重价格就往另一边"倾斜"。

$$
\text{OBI} = \frac{Q^b - Q^a}{Q^b + Q^a} \in [-1, 1]
$$

- $Q^b$：买一（或前 $k$ 档）的挂单量
- $Q^a$：卖一（或前 $k$ 档）的挂单量
- OBI 接近 +1 表示买盘厚，价格倾向上涨

### 7.2 微价格（Microprice）

中间价忽略了量的信息。**微价格**用对面的挂单量加权，更好地预测下一个成交价：

$$
P^{\text{micro}} = \frac{P^a \cdot Q^b + P^b \cdot Q^a}{Q^b + Q^a} = M + \frac{S}{2} \cdot \text{OBI}
$$

- 买盘越厚（$Q^b$ 大），微价格越靠近卖价 $P^a$。
- 第二个等号说明：微价格 = 中间价 + 半价差 × 不平衡度。

### 7.3 订单流不平衡（Order Flow Imbalance, OFI）

OBI 是"静态"的盘口状态，**OFI** 是"动态"的盘口变化。Cont、Kukanov 和 Stoikov (2014) 定义：相邻两次快照 $n-1 \to n$ 之间，

$$
e_n = \mathbb{1}_{\{P^b_n \ge P^b_{n-1}\}} q^b_n - \mathbb{1}_{\{P^b_n \le P^b_{n-1}\}} q^b_{n-1} - \mathbb{1}_{\{P^a_n \le P^a_{n-1}\}} q^a_n + \mathbb{1}_{\{P^a_n \ge P^a_{n-1}\}} q^a_{n-1}
$$

- $P^b_n, q^b_n$：第 $n$ 次快照的买一价与买一量；$P^a_n, q^a_n$ 同理为卖一
- $\mathbb{1}_{\{\cdot\}}$：示性函数，条件成立取 1，否则取 0

**逐项解读**（以买方为例）：

| 情况 | 买方贡献 | 含义 |
| --- | --- | --- |
| 买一价不变 | $q^b_n - q^b_{n-1}$ | 买一量增加 = 买压上升 |
| 买一价上移 | $+q^b_n$ | 新价位的全部量都是新增买压 |
| 买一价下移 | $-q^b_{n-1}$ | 原价位的量全部消失（被吃或撤单）= 买压下降 |

卖方对称，符号相反。一个时间窗口内 $\text{OFI} = \sum_n e_n$。该论文发现短时间内中间价变化与 OFI 近似线性：

$$
\Delta M \approx \beta \cdot \text{OFI}
$$

$\beta$ 与市场深度成反比（深度越大，同样的 OFI 推动价格越少）。

```python
import numpy as np
import pandas as pd

def compute_ofi(df: pd.DataFrame) -> np.ndarray:
    """df 需含 bid_px, bid_sz, ask_px, ask_sz 四列（L1 快照序列）"""
    bp, bq = df.bid_px.to_numpy(), df.bid_sz.to_numpy()
    ap, aq = df.ask_px.to_numpy(), df.ask_sz.to_numpy()
    eb = np.where(bp[1:] >= bp[:-1], bq[1:], 0) - np.where(bp[1:] <= bp[:-1], bq[:-1], 0)
    ea = np.where(ap[1:] <= ap[:-1], aq[1:], 0) - np.where(ap[1:] >= ap[:-1], aq[:-1], 0)
    return eb - ea

df = pd.DataFrame({
    "bid_px": [10.00, 10.00, 10.01, 10.01],
    "bid_sz": [500, 800, 300, 100],
    "ask_px": [10.01, 10.01, 10.02, 10.02],
    "ask_sz": [400, 400, 600, 900],
})
print(compute_ofi(df))   # [300 700 -500]：买量增加(+300)、买价上移、买量减少且卖量增加
```

### 7.4 成交量分布类因子

| 因子 | 计算 | 直觉 |
| --- | --- | --- |
| 主动买入占比 | 主动买成交额 / 总成交额 | 资金主动性 |
| 大单净流入 | 按单笔金额阈值划分的大单主买 − 主卖 | "聪明钱"代理（阈值主观，谨慎解读） |
| 日内成交量分布 | 各时段成交量占全天比例（A 股通常呈 U 型：开盘和收盘量大） | VWAP 算法的基础；偏离常态的分布可能有信息 |
| 尾盘成交占比 | 14:30 以后成交量 / 全天 | 机构调仓行为代理 |
| 已实现波动率 | 日内高频收益平方和 $\sum r_i^2$ | 比日线更精确的波动估计 |
| VPIN | 按成交量分桶后的买卖量不平衡均值 | "订单流毒性"，衡量知情交易概率 |

### 7.5 A 股高频因子的特殊性

```
┌──────────────────────────────────────────────────────────────────┐
│                A 股：高频数据 → 低频因子                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  美股/期货/加密：高频信号 ──▶ 秒级/分钟级交易                      │
│                                                                  │
│  A 股股票（T+1 限制）：                                           │
│     高频数据 ──▶ 日内聚合 ──▶ 日频因子 ──▶ 截面选股（Phase 7）      │
│     例：每日 OFI 均值、尾盘主买占比、日内已实现偏度                 │
│                                                                  │
│  这类"高频因子低频化"是国内量化私募 alpha 的重要来源之一            │
└──────────────────────────────────────────────────────────────────┘
```

**陷阱**：

- 高频信号衰减极快（秒到分钟级），在快照数据上算出的"预测力"可能在你能下单时已经消失。
- 用快照计算的 OFI 会漏掉快照之间的事件；逐笔数据更准确。
- 小盘股盘口薄、噪声大，因子值需要截面标准化后再使用（Phase 7 的预处理）。

---

## 8. 执行算法：TWAP / VWAP / POV

> 对应课程：`06-twap-vwap.py`

### 8.1 为什么要拆单

**比喻**：你要把一卡车货物运过一座窄桥。一次全开过去会压垮桥（市场冲击），分 100 趟慢慢运又可能遇上暴雨（价格风险）。执行算法就是"调度器"，决定每趟运多少、什么时候运。

```
         大单（母单 Parent Order）：今天买入 100 万股
                          │
                          ▼
              ┌───────────────────────┐
              │      执行算法          │  ← 决定 何时 / 多少 / 什么价格
              └───────────────────────┘
               │    │    │    │    │
               ▼    ▼    ▼    ▼    ▼
         子单（Child Orders）：每笔几千股，分布在全天
```

### 8.2 TWAP：时间加权平均价格

**思想**：把订单均匀分布在时间上。总量 $X$，分 $N$ 个时间片，每片下 $X/N$。

$$
\text{TWAP} = \frac{1}{N}\sum_{i=1}^{N} P_i
$$

- $P_i$：第 $i$ 个时间片的价格
- 优点：简单、可预测；缺点：不考虑成交量分布，在成交清淡的时段冲击偏大，而且固定节奏容易被识别。

### 8.3 VWAP：成交量加权平均价格

**思想**：市场成交多的时候多下，成交少的时候少下，让自己的成交"混入"市场流动性。

$$
\text{VWAP} = \frac{\sum_i P_i V_i}{\sum_i V_i}
$$

- $P_i, V_i$：第 $i$ 笔（或第 $i$ 个时间段）的成交价与成交量

**VWAP 算法**：用历史日内成交量分布（Volume Profile）预测今天各时段的成交占比 $w_i$，在时段 $i$ 下单 $X \cdot w_i$。目标是让自己的成交均价接近当日市场 VWAP。

```python
import numpy as np

def volume_profile(hist_volumes: np.ndarray) -> np.ndarray:
    """hist_volumes: (天数, 时段数) 历史成交量矩阵，返回各时段平均占比"""
    frac = hist_volumes / hist_volumes.sum(axis=1, keepdims=True)
    return frac.mean(axis=0)

def vwap_schedule(total_qty: int, profile: np.ndarray, lot: int = 100) -> np.ndarray:
    """按成交量分布切分母单，并取整到最小交易单位（A 股 100 股）"""
    raw = total_qty * profile / lot
    lots = np.floor(raw).astype(int)
    remainder = total_qty // lot - lots.sum()
    lots[np.argsort(raw - lots)[::-1][:remainder]] += 1   # 最大余数法分配零头
    return lots * lot

rng = np.random.default_rng(0)
u_shape = np.array([12, 8, 6, 5, 5, 6, 7, 9, 11, 14, 17], dtype=float)  # 11 个时段，U 型
hist = u_shape * rng.uniform(0.8, 1.2, size=(20, 11))                   # 20 天历史
plan = vwap_schedule(100_000, volume_profile(hist))
print(plan, plan.sum())
```

### 8.4 POV：成交量百分比

**思想**：实时跟随市场成交量，始终保持自己占市场成交的固定比例 $\rho$（如 10%）。

$$
\text{子单量}_t = \rho \cdot V^{\text{market}}_{t-1}
$$

- $\rho$：参与率（Participation Rate）
- $V^{\text{market}}_{t-1}$：上一个时间片的市场成交量

```python
def pov_child_qty(rate, last_mkt_volume, remaining, lot=100):
    qty = int(rate * last_mkt_volume // lot * lot)
    return min(qty, remaining)
```

### 8.5 三种算法对比

| 算法 | 跟踪基准 | 完成时间 | 适用场景 | 主要风险 |
| --- | --- | --- | --- | --- |
| TWAP | 时间均价 | 确定 | 流动性好、日内分布平稳的标的；加密市场 | 节奏固定易被识别；忽视量能 |
| VWAP | 市场 VWAP | 确定 | 机构日常调仓，最常用的基准 | 当日成交量分布偏离历史 |
| POV | 参与率 | 不确定（取决于市场成交） | 控制冲击优先，不急于完成 | 市场清淡时可能当天完不成 |
| IS / Arrival Price | 到达价 | 由优化决定，通常前倾 | 信号有时效性、需快速建仓 | 冲击成本较高 |

### 8.6 执行质量评估

```python
def slippage_bp(avg_fill_px, benchmark_px, side):
    """side=+1 买入、-1 卖出；正数表示比基准差（成本）"""
    return side * (avg_fill_px - benchmark_px) / benchmark_px * 1e4

print(slippage_bp(10.032, 10.000, +1))   # 买贵了 32bp
print(slippage_bp(9.990, 10.000, -1))    # 卖便宜了 10bp
```

**陷阱**：VWAP 算法"打败 VWAP"并不代表执行好——如果你的大单本身推高了全天价格，VWAP 基准也被你推高了。机构更看重相对**到达价**的实施缺口。

---

## 9. Almgren-Chriss 最优执行

> 对应课程：`07-almgren-chriss.py`

### 9.1 直觉：冲击成本与时间风险的权衡

执行大单时有一对根本矛盾：

- **执行越快** → 每个时间片下单量越大 → **冲击成本越高**
- **执行越慢** → 暴露在价格波动中的时间越长 → **价格风险越大**

```
               冲击成本 vs 价格风险

  成本 ▲
       │╲                                   ╱  价格风险（方差）
       │ ╲                               ╱      随执行时间增长
       │  ╲                           ╱
       │   ╲   冲击成本            ╱
       │    ╲  随执行时间下降   ╱
       │     ╲             ╱
       │       ╲ ____ ╱  ← 总"成本 + λ·风险"最小的点取决于风险厌恶 λ
       └────────────────────────────────────────▶ 执行时长
         立即全部成交                     慢慢均匀成交
```

Almgren 和 Chriss (2000) 把这个问题形式化为**均值-方差优化**，这和 Phase 8 的 Markowitz 组合优化是同一个思想。

### 9.2 模型设定

- 在时间 $[0, T]$ 内卖出 $X$ 股，分成 $N$ 个时间片，每片长度 $\tau = T/N$
- $x_k$：第 $k$ 个时刻**剩余**持仓，$x_0 = X$，$x_N = 0$
- $n_k = x_{k-1} - x_k$：第 $k$ 个时间片卖出的数量；交易速度 $v_k = n_k / \tau$
- 价格动态（算术随机游走 + 永久冲击）：

$$
S_k = S_{k-1} + \sigma \sqrt{\tau}\, \xi_k - \tau\, g(v_k), \qquad g(v) = \gamma v
$$

- 实际成交价（再减去临时冲击）：

$$
\tilde{S}_k = S_{k-1} - h(v_k), \qquad h(v) = \epsilon\, \text{sgn}(n_k) + \eta v
$$

符号含义：

| 符号 | 含义 |
| --- | --- |
| $\sigma$ | 价格波动率（价格单位 / $\sqrt{\text{时间}}$） |
| $\xi_k$ | 独立标准正态随机变量 |
| $\gamma$ | 永久冲击系数：每单位交易速度对价格的永久影响 |
| $\epsilon$ | 固定成本，约等于半个价差加手续费 |
| $\eta$ | 临时冲击系数：交易越快，单位成交价越差 |

### 9.3 成本的期望与方差

相对"以 $S_0$ 全部成交"的执行成本，其期望和方差为：

$$
E[\text{Cost}] = \frac{1}{2}\gamma X^2 + \epsilon \sum_{k=1}^{N} |n_k| + \frac{\tilde{\eta}}{\tau} \sum_{k=1}^{N} n_k^2, \qquad \tilde{\eta} = \eta - \frac{1}{2}\gamma\tau
$$

$$
\text{Var}[\text{Cost}] = \sigma^2 \sum_{k=1}^{N} \tau\, x_k^2
$$

- 期望成本中第一项（永久冲击）与路径无关；第三项惩罚"某个时间片卖得太多"——**均匀卖出时最小**。
- 方差惩罚"手里长时间拿着大量头寸"——**一开始就卖完时最小**。

目标：最小化 $E[\text{Cost}] + \lambda \cdot \text{Var}[\text{Cost}]$，$\lambda$ 是风险厌恶系数。

### 9.4 最优解：双曲正弦轨迹

最优剩余持仓轨迹有闭式解：

$$
x_j = X \cdot \frac{\sinh\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)}, \qquad t_j = j\tau
$$

其中 $\kappa$ 满足 $\dfrac{2}{\tau^2}\big(\cosh(\kappa\tau) - 1\big) = \dfrac{\lambda \sigma^2}{\tilde{\eta}}$。当 $\tau \to 0$ 时近似为：

$$
\kappa \approx \sqrt{\frac{\lambda \sigma^2}{\eta}}
$$

**解读**：

- $\lambda \to 0$（风险中性）：$\kappa \to 0$，轨迹退化为直线——**就是 TWAP**。
- $\lambda$ 越大：$\kappa$ 越大，轨迹越"前倾"——开头卖得多，尽快降低风险暴露。
- $1/\kappa$ 被称为订单的"半衰期"（Half-life），量纲是时间：若 $1/\kappa \ll T$，大部分交易会在早期完成。

```
剩余持仓 x(t)
 X ▲●
   │ ●●
   │ ●  ●●                     ── λ=0：直线（TWAP）
   │  ●    ●●
   │  ●       ●●               ●● λ 中等：温和前倾
   │   ●         ●●
   │    ●           ●●         ●  λ 很大：迅速卖出
   │      ●            ●●
   │         ●●           ●●
 0 └──────────────●●●●●●●●●●●●●▶ t
   0                          T
```

### 9.5 代码实现

```python
import numpy as np

def ac_trajectory(X, T, N, sigma, eta, gamma, lam):
    """Almgren-Chriss 离散最优轨迹，返回长度 N+1 的剩余持仓数组"""
    tau = T / N
    t = np.arange(N + 1) * tau
    if lam == 0:
        return X * (1 - t / T)                       # 风险中性：线性（TWAP）
    eta_t = eta - 0.5 * gamma * tau
    kappa = np.arccosh(1 + 0.5 * lam * sigma**2 / eta_t * tau**2) / tau
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

def ac_cost(x, T, sigma, eta, gamma, eps=0.0):
    """给定轨迹，返回 (期望成本, 成本方差)"""
    N = len(x) - 1
    tau = T / N
    n = -np.diff(x)
    eta_t = eta - 0.5 * gamma * tau
    E = 0.5 * gamma * x[0] ** 2 + eps * np.abs(n).sum() + eta_t / tau * (n**2).sum()
    V = sigma**2 * tau * (x[1:] ** 2).sum()
    return E, V

# 参数量级参考 Almgren-Chriss 原论文示例（5 天卖出 100 万股）
X, T, N, sigma, eta, gamma = 1e6, 5, 50, 0.95, 2.5e-6, 2.5e-7
for lam in [0, 1e-7, 1e-6, 1e-5]:
    x = ac_trajectory(X, T, N, sigma, eta, gamma, lam)
    E, V = ac_cost(x, T, sigma, eta, gamma)
    print(f"λ={lam:.0e}  第1天完成 {1 - x[N // 5] / X:5.1%}  E={E:>12,.0f}  Std={np.sqrt(V):>12,.0f}")
```

输出中可以看到：$\lambda$ 增大 → 第一天完成比例上升 → 期望成本上升、成本标准差下降。把不同 $\lambda$ 下的 $(\text{Std}, E)$ 画出来，就得到**执行的有效前沿（Efficient Frontier）**。

### 9.6 实际意义与陷阱

- **参数估计是难点**：$\eta$、$\gamma$ 需要用自己的历史成交数据回归，公开数据很难得到可靠估计。
- **线性冲击是简化**：实证上冲击更接近平方根（5.3 节），后续文献有非线性冲击的推广。
- **它给出的是"计划"**：真实执行中还需要根据盘口实时决定每个子单是挂单还是吃单（这一层叫"下单逻辑"或 Order Placement）。
- **工程上的价值**：AC 框架让"执行要多快"从拍脑袋变成一个可以讨论风险偏好的参数。

---

## 10. 做市原理：库存风险与逆向选择

> 对应课程：`08-market-making-basics.py`

### 10.1 做市商赚什么钱

**比喻**：做市商像二手车商。以 9 万收车、以 10 万卖车，赚的是"周转差价"，而不是赌车价涨跌。理想状态是收一辆马上卖一辆，库存永远接近零。

```
┌────────────────────────────────────────────────────────────────┐
│                      做市商的一天                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   同时挂出：  买 @ 99.95          卖 @ 100.05                   │
│                  │                    │                        │
│        有人卖给我 ▼                    ▼ 有人从我这买              │
│              库存 +1               库存 −1                       │
│                                                                │
│   理想：买卖成交次数相当 → 每对成交赚 0.10，库存回到 0             │
│                                                                │
│   现实风险：                                                    │
│     ① 库存风险：只成交了一边，库存堆积，价格朝不利方向走            │
│     ② 逆向选择：和"知道价格要动"的人成交，成交后立刻浮亏           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

做市 PnL 可以分解为：

$$
\text{PnL} = \underbrace{\sum \text{半价差收入}}_{\text{Spread Capture}} + \underbrace{\sum q_t \Delta S_t}_{\text{库存损益}} - \text{手续费} + \text{返佣（Rebate）}
$$

- $q_t$：时刻 $t$ 的库存（正为多头）
- $\Delta S_t$：中间价变化
- 部分交易所（尤其是加密市场和某些期货/期权做市项目）对 maker 给予手续费返还

### 10.2 库存风险

库存损益 $\sum q_t \Delta S_t$ 的期望为零（若价格是随机游走），但方差随库存规模增长：$\text{Var} \propto q^2 \sigma^2$。做市商控制库存的方法：

| 方法 | 做法 |
| --- | --- |
| 报价偏移（Skew） | 多头库存过大时，同时降低买价和卖价，让卖单更容易成交、买单更难成交 |
| 单边报价 | 库存到达上限时只挂减仓方向 |
| 对冲 | 用相关性高的资产（期货、ETF、其他交易所）对冲库存 |
| 收盘平仓 | 日终将库存降到零，避免隔夜风险 |

### 10.3 逆向选择：做市商的头号敌人

回到 4.2 节：每次和知情交易者成交，做市商都会亏钱。在高频语境下，"知情"不一定是内幕消息，更多是**速度优势**——别的交易者比你更早看到另一个交易所的价格变动，赶在你撤单前吃掉你的旧报价（"stale quote sniping"）。

**度量逆向选择**：统计成交后 $\Delta$ 时间内中间价的不利移动（Markout）：

$$
\text{Markout}(\Delta) = -q^{\text{fill}} \cdot (M_{t+\Delta} - P^{\text{fill}})
$$

- $q^{\text{fill}}$：做市商这笔成交的方向（做市商买入为 +1）
- 若 Markout 系统性为正（例如做市商买入后价格继续下跌），说明你正在被逆向选择

### 10.4 代码：最简单的对称做市模拟

```python
import numpy as np

def simulate_symmetric_mm(n_steps=2000, half_spread=0.05, sigma=0.08,
                          arrival_prob=0.5, informed=0.0, rng=None):
    """informed：到达的对手方中知情交易者的比例（他们知道下一步价格方向）"""
    mid, q, cash = 100.0, 0, 0.0
    for _ in range(n_steps):
        move = sigma * rng.choice([-1, 1])            # 下一步价格变化
        if rng.random() < arrival_prob:
            if rng.random() < informed:
                side = -int(np.sign(move))            # 知情者：价格要涨就来买我的卖单
            else:
                side = rng.choice([1, -1])            # 噪声交易者：随机
            px = mid - side * half_spread             # side=+1 我买入@bid，-1 我卖出@ask
            q += side
            cash -= side * px
        mid += move
    return cash + q * mid, q

rng = np.random.default_rng(7)
for inf in [0.0, 0.3, 0.6]:
    res = np.array([simulate_symmetric_mm(informed=inf, rng=rng) for _ in range(200)])
    print(f"知情比例 {inf:.1f}: 平均PnL={res[:, 0].mean():7.2f}  PnL标准差={res[:, 0].std():6.2f}"
          f"  |库存|均值={np.abs(res[:, 1]).mean():5.1f}")
```

可以观察到：知情交易者比例越高，平均 PnL 越低——每笔知情成交赚到半个价差（0.05），却在随后的价格变动（0.08）中亏得更多。同时由于没有库存控制，库存像随机游走一样漂移，PnL 的标准差很大。下一节的 Avellaneda-Stoikov 模型正是为了系统地解决库存问题。

---

## 11. Avellaneda-Stoikov 做市模型

> 对应课程：`09-avellaneda-stoikov.py`

### 11.1 直觉

Avellaneda 和 Stoikov (2008) 回答了两个问题：

1. **报价中心放在哪？** 不是中间价，而是根据库存调整后的**保留价格（Reservation Price）**：持有多头库存时，你心里的"公允价"会低于市场中间价，因为你急于卖出。
2. **价差设多宽？** 价差太窄 → 成交多但每笔赚得少、风险大；价差太宽 → 每笔赚得多但成交少。最优价差取决于风险厌恶与订单到达的强度。

```
                 无库存 q = 0                    多头库存 q > 0

   ask  ─────── s + δ                  ask ─────── r + δ   ← 卖价更有吸引力
                                                            （更容易卖出）
   mid  ─────── s ← 保留价格 r = s       mid ─────── s
                                            ─────── r      ← 保留价格下移
   bid  ─────── s − δ                  bid ─────── r − δ   ← 买价更低（更难买入）
```

### 11.2 模型设定

- 中间价服从算术布朗运动：$dS_t = \sigma\, dW_t$
- 做市商在 $S_t - \delta^b$ 挂买单、在 $S_t + \delta^a$ 挂卖单，$\delta$ 是报价离中间价的距离
- 市价单到达并成交你挂单的强度随距离指数衰减：

$$
\lambda(\delta) = A\, e^{-k\delta}
$$

- $A$：订单到达的基础强度（单位时间内的成交次数量级）
- $k$：衰减速度——$k$ 越大，挂远一点成交概率下降越快（盘口越"密集"）
- 做市商最大化期末财富的指数效用 $E[-e^{-\gamma W_T}]$，$\gamma$ 为风险厌恶系数，$T$ 为交易时段终点

### 11.3 核心公式

**保留价格**：

$$
r(s, q, t) = s - q\, \gamma\, \sigma^2 (T - t)
$$

- $q$：当前库存（多头为正）
- $\gamma \sigma^2 (T - t)$：持有一单位库存到期末的风险"代价"
- 多头时 $r < s$，空头时 $r > s$；离收盘越近（$T - t$ 越小），调整越小

**最优总价差**（买卖报价之间的距离 $\delta^a + \delta^b$）：

$$
\delta^a + \delta^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right)
$$

- 第一项：风险补偿，波动率越高、离收盘越远，价差越宽
- 第二项：与订单流"密度" $k$ 相关的垄断利润项，$k$ 越大（挂远了就没人成交），价差越窄

最终报价为 $r \pm \frac{1}{2}(\delta^a + \delta^b)$。

### 11.4 代码：AS 策略 vs 对称策略

```python
import numpy as np

def as_quotes(s, q, t, T, sigma, gamma, k):
    r = s - q * gamma * sigma**2 * (T - t)
    spread = gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + gamma / k)
    return r - spread / 2, r + spread / 2

def simulate(strategy, n_paths=300, s0=100.0, T=1.0, dt=0.005,
             sigma=2.0, gamma=0.1, k=1.5, A=140, seed=0):
    rng = np.random.default_rng(seed)
    pnls, invs = [], []
    for _ in range(n_paths):
        s, q, cash = s0, 0, 0.0
        for i in range(int(T / dt)):
            t = i * dt
            if strategy == "as":
                bid, ask = as_quotes(s, q, t, T, sigma, gamma, k)
            else:                                    # 对称：同样宽度，但以中间价为中心
                half = as_quotes(s, 0, t, T, sigma, gamma, k)[1] - s
                bid, ask = s - half, s + half
            if rng.random() < A * np.exp(-k * (s - bid)) * dt:
                q, cash = q + 1, cash - bid
            if rng.random() < A * np.exp(-k * (ask - s)) * dt:
                q, cash = q - 1, cash + ask
            s += sigma * np.sqrt(dt) * rng.choice([-1, 1])
        pnls.append(cash + q * s)
        invs.append(q)
    return np.mean(pnls), np.std(pnls), np.std(invs)

for st in ["as", "sym"]:
    m, sd, qs = simulate(st)
    print(f"{st:>4}: PnL 均值={m:6.2f}  PnL 标准差={sd:6.2f}  期末库存标准差={qs:5.2f}")
```

参数取自原论文的数值实验（$s_0=100, T=1, \sigma=2, \gamma=0.1, k=1.5, A=140$）。典型结果：两种策略平均 PnL 接近，但 AS 策略的 PnL 标准差和库存波动显著更小——**AS 用一点点期望收益换来了大幅降低的风险**。

### 11.5 实际应用与陷阱

| 问题 | 说明 |
| --- | --- |
| 参数校准 | $A, k$ 需要从历史成交与盘口数据拟合"距离 → 成交概率"曲线 |
| $T$ 的含义 | 加密市场 7×24 无收盘，常用滚动窗口或无限期限版本（Guéant 等人的推广） |
| 忽略逆向选择 | 模型假设价格独立于订单流，真实市场中需叠加 OFI/微价格等短期 alpha 来调整报价中心 |
| tick size | 报价必须对齐最小变动单位，A 股 0.01 元的 tick 对低价股相当于很宽的价差 |
| 排队位置 | 模型假设"挂在 δ 就有 λ(δ) 的成交率"，真实中还取决于你在队列中的位置 |
| 手续费 | 必须扣除 maker 费率；在 taker 费率高的市场，价差收入可能被吃光 |

---

## 12. 高频交易基础设施与延迟

> 对应课程：`10-hft-infrastructure.py`

### 12.1 延迟链路：Tick-to-Trade

**Tick-to-Trade 延迟**：从交易所发出行情到你的订单到达交易所的总时间。

```
┌──────────────────────────────────────────────────────────────────┐
│                       Tick-to-Trade 链路                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  交易所撮合 ──▶ 行情发布 ──网络──▶ 网卡 ──▶ 内核/协议栈 ──▶ 解码    │
│                                                         │        │
│                                                         ▼        │
│  交易所接收 ◀──网络── 网卡 ◀── 内核/协议栈 ◀── 风控检查 ◀── 策略决策 │
│                                                                  │
│  每一段都可以被优化：                                              │
│    网络：机房托管（Colocation），缩短物理距离                       │
│    内核：Kernel Bypass（如 DPDK、Solarflare/ef_vi、Onload）        │
│    解码 + 策略 + 风控：C++ 无锁编程，或直接用 FPGA 硬件实现          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 延迟的数量级

| 操作 | 数量级 | 备注 |
| --- | --- | --- |
| CPU L1 缓存访问 | ~1 ns | |
| 主内存访问 | ~100 ns | |
| 光在光纤中传播 1 km | ~5 μs | 光纤中光速约为真空的 2/3 |
| FPGA tick-to-trade | 亚微秒级 | 顶级 HFT 竞争领域 |
| C++ 软件 tick-to-trade（优化后） | 微秒级 | |
| Python 处理一条行情（含解析） | 数十 μs 到 ms 级 | 取决于实现 |
| 同城机房间网络往返 | 百 μs 到 ms 级 | |
| 家用宽带到交易所 | 数 ms 到数十 ms | |
| 跨国网络往返 | 数十到数百 ms | |

**结论**：Python + 家用网络在延迟竞赛中没有任何机会。个人量化应选择**延迟不敏感**的策略（分钟级以上），把精力放在 alpha 和风控上。

### 12.3 技术栈一览

| 层次 | 技术 | 作用 |
| --- | --- | --- |
| 物理位置 | Colocation（机房托管）、微波/毫米波链路 | 缩短距离，微波在空气中比光纤快 |
| 网络 | 低延迟交换机、Kernel Bypass | 绕过操作系统协议栈 |
| 时钟 | PTP（精确时间协议）、GPS 授时 | 纳秒级时间戳，用于延迟度量与审计 |
| 计算 | C++/Rust、CPU 绑核、大页内存、无锁队列 | 消除抖动（Jitter） |
| 硬件 | FPGA、智能网卡 | 行情解码与下单硬件化 |
| 行情 | 交易所组播行情、Level-2 逐笔 | 最快、最全的数据源 |

**比喻**：如果说中低频量化是"选对方向开车"，HFT 就是 F1 赛车——车（硬件）、赛道位置（机房）和毫秒级反应缺一不可，而且成本极高、赢家通吃。

### 12.4 延迟测量：中位数不够，要看尾部

```python
import time
import numpy as np

def measure(fn, n=10_000):
    lat = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        lat[i] = time.perf_counter_ns() - t0
    p50, p99, p999 = np.percentile(lat, [50, 99, 99.9])
    print(f"p50={p50/1e3:.1f}μs  p99={p99/1e3:.1f}μs  p99.9={p999/1e3:.1f}μs  max={lat.max()/1e3:.1f}μs")

msg = '{"bid":[["10.00","500"]],"ask":[["10.01","300"]]}'
import json
measure(lambda: json.loads(msg))
```

**关键认知**：低延迟系统关心的是**尾延迟**（p99、p99.9）和**抖动**，而不是平均值。一次 GC 停顿或一次缓存未命中，就可能让你在最关键的行情跳动时慢一步。

### 12.5 国内高频交易的监管环境

- 证监会《证券市场程序化交易管理规定（试行）》及沪深北交易所的程序化交易实施细则，对程序化交易实行报告制度，并对**高频交易**有额外的报告与监管要求。交易所细则中对高频交易的认定标准包括单账户每秒申报、撤单笔数或单日申报、撤单笔数达到一定阈值（如每秒 300 笔以上或单日 20000 笔以上）等。
- 交易所对异常交易行为（如频繁申报撤单、瞬时大额申报、虚假申报等）有实时监控与处置措施，还可能对高频交易收取差异化费用。
- 期货市场也有程序化交易报告和穿透式监管要求。

> 以上内容随监管更新而变化，**具体以监管与交易所最新规定为准**。Phase 13 会进一步讨论合规流程。

---

## 13. 最佳实践与常见陷阱

### 13.1 最佳实践

```
┌──────────────────────────────────────────────────────────────────┐
│                         最佳实践清单                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📐 成本建模                                                      │
│     □ 回测中同时包含显性成本、半价差和冲击成本                     │
│     □ 用平方根法则估计策略容量，报告"资金规模-净收益"曲线           │
│     □ 冲击参数尽量用自己的真实成交数据校准                         │
│                                                                  │
│  📊 数据                                                          │
│     □ 明确区分快照数据与逐笔数据                                   │
│     □ 价格用整数 tick 存储与比较                                   │
│     □ 统一沪深/各交易所的数据模型                                  │
│                                                                  │
│  ⚙️ 执行                                                          │
│     □ 以到达价（实施缺口）为主要评估基准                           │
│     □ 子单量取整到最小交易单位，并处理涨跌停、价格笼子              │
│     □ 对执行结果做事后分析（TCA，Transaction Cost Analysis）        │
│                                                                  │
│  🏦 做市                                                          │
│     □ 设置硬性库存上限与亏损上限                                   │
│     □ 监控成交后 Markout，识别逆向选择                             │
│     □ 回放测试时模拟排队位置，不要假设"挂了就成交"                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 13.2 常见陷阱

| 陷阱 | 表现 | 解决方案 |
| --- | --- | --- |
| 中间价成交假设 | 回测在 mid 成交，实盘每笔多付半个价差 | 主动单按对手价成交，被动单模拟排队 |
| 忽视冲击成本 | 小资金回测漂亮，放大资金后收益消失 | 平方根法则 + 容量分析 |
| 挂单必成交 | 做市回测中挂单价位被触及即视为成交 | 价格"穿越"才算成交，或估计队列位置 |
| 快照当逐笔 | 用 3 秒快照计算的因子信息不完整 | 了解数据粒度，必要时使用逐笔数据 |
| 本地订单簿漂移 | WebSocket 增量丢包导致本地簿与交易所不一致 | 序列号校验 + 定期快照校正 |
| 增量当变化量 | 把价位的新总量累加到旧量上 | 增量消息是覆盖语义 |
| 打败 VWAP 的错觉 | 自己的大单推高了 VWAP 基准 | 用到达价基准，看实施缺口 |
| 盲目追求低延迟 | 个人投入大量精力优化毫秒 | 选择延迟不敏感策略，延迟预算够用即可 |
| 忽视 A 股制度 | T+1、涨跌停、价格笼子导致子单无法成交 | 在执行模拟器中实现这些规则 |

---

## 14. 总结

### 14.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                          核心概念回顾                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  订单簿与撮合                                                     │
│  ├── 价格优先、时间优先；成交价取被动方价格                        │
│  ├── 订单类型：限价、市价、IOC、FOK、Post-Only                    │
│  └── A 股：集合竞价（成交量最大原则）、涨跌停、价格笼子、T+1        │
│                                                                  │
│  交易成本                                                         │
│  ├── 价差 = 订单处理 + 库存 + 逆向选择                             │
│  ├── 有效价差 = 实现价差 + 价格冲击                                │
│  ├── 平方根法则：冲击 ≈ Y·σ·√(Q/V)                                │
│  └── 实施缺口：以到达价为基准的总执行成本                          │
│                                                                  │
│  高频因子                                                         │
│  ├── OBI / 微价格：静态盘口不平衡                                  │
│  ├── OFI：盘口变化，ΔM ≈ β·OFI                                    │
│  └── A 股：高频数据聚合为日频因子                                  │
│                                                                  │
│  执行算法                                                         │
│  ├── TWAP（时间均匀）/ VWAP（跟随量能）/ POV（固定参与率）          │
│  └── Almgren-Chriss：最小化 E[成本] + λ·Var[成本]，sinh 轨迹        │
│                                                                  │
│  做市                                                             │
│  ├── 收益来自价差，风险来自库存与逆向选择                          │
│  └── Avellaneda-Stoikov：保留价格 r = s − qγσ²(T−t) + 最优价差      │
│                                                                  │
│  基础设施                                                         │
│  └── Tick-to-Trade 延迟、尾延迟；个人量化应避开延迟竞赛            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 学习检查对照

| 学习检查项 | 对应章节 |
| --- | --- |
| 能解释价差的三个组成部分 | 第 4 节 |
| 能实现 VWAP 执行算法 | 第 8.3 节 |
| 理解做市商的库存风险与逆向选择 | 第 10、11 节 |
| 能从订单簿计算订单不平衡因子 | 第 7 节 |

### 14.3 与下一阶段的衔接

```
Phase 11 微观结构与执行                 Phase 12 期权与衍生品
──────────────────────                  ──────────────────────
做市商的库存风险          ───────▶      期权做市商的 Delta 对冲
AS 模型中的 σ 与风险厌恶   ───────▶      波动率是期权定价的核心输入
"用对冲控制风险"          ───────▶      Delta / Gamma 对冲与 Gamma-Theta 权衡
交易成本                  ───────▶      对冲频率与交易成本的权衡

Phase 11 的执行与订单簿知识，也会在 Phase 13 中落地为 OMS 与执行模块。
```

**推荐阅读**：

- Larry Harris《Trading and Exchanges》：微观结构的百科全书，非数学化，适合入门
- Cartea, Jaimungal, Penalva《Algorithmic and High-Frequency Trading》：执行与做市的数学模型
- Almgren & Chriss (2000), *Optimal Execution of Portfolio Transactions*
- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*
- Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*
