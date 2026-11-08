"""
asyncio 事件引擎
================

学习目标：
    1. 理解交易系统为什么采用事件驱动 + 单线程事件循环
    2. 读懂 mini_trader 事件引擎的实现：订阅 / 发布、通用处理器、排队延迟统计
    3. 掌握三个工程要点：异常隔离、跨线程投递（call_soon_threadsafe）、定时器事件
    4. 亲手测量"在处理器里阻塞"会让系统延迟多少，并用 to_thread 修复
    5. 跑通"行情 → 策略 → 下单 → 成交"的完整事件链路

核心概念：
    - 事件（Event）：type + data + 入队时间戳
    - 处理器（Handler）：普通函数或 async 函数，按注册顺序执行
    - 异常隔离：一个处理器抛异常，不影响同一事件的其他处理器，错误转为 error 事件
    - 跨线程投递：柜台 API 的回调在它自己的线程里，必须用 loop.call_soon_threadsafe 投递

前置知识：
    - 01-trading-system-architecture.py
    - Python asyncio 基础（async / await、create_task）

环境要求：
    - Python 3.11+ 标准库 + numpy / pandas，不需要网络
    - 所有演示都会自行结束，不会无限循环
"""

import asyncio
import inspect
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mini_trader import OMS, Direction, EventEngine, MockExchange, MockGateway, Offset, OrderRequest
from mini_trader.models import opposite
from mini_trader.events import EVENT_ERROR, EVENT_ORDER, EVENT_SIGNAL, EVENT_TICK, EVENT_TIMER, EVENT_TRADE

load_dotenv()


def wait_stats(engine: EventEngine) -> str:
    w = np.array(engine.wait_times) * 1000
    return f"事件数 {w.size}，排队耗时 p50 {np.median(w):.2f} ms，max {w.max():.2f} ms"


# ==================== 第一部分：为什么用事件驱动 ====================


def why_event_driven():
    print("=" * 60)
    print("第一部分：为什么用事件驱动 + asyncio")
    print("=" * 60)
    print("""
    🛫 比喻：交易系统像机场塔台。飞机（事件）随时到达，塔台不会挨个"轮询"每架飞机，
       而是通过无线电频道（事件总线）接收消息，再分派给对应的调度员（处理器）。

    交易系统是典型的 I/O 密集型：大部分时间在等网络（行情推送、订单回报），
    每个事件的计算量很小 —— 这正是 asyncio（单线程事件循环 + 协程）擅长的场景。
    """)
    print(pd.DataFrame(
        [
            ["事件循环", "asyncio event loop", "libuv event loop"],
            ["协程", "async def / await", "async function / await"],
            ["并发任务", "asyncio.create_task()", "不 await 的 Promise"],
            ["队列", "asyncio.Queue", "EventEmitter / 消息队列"],
            ["阻塞陷阱", "协程里 time.sleep()、重计算", "回调里同步阻塞"],
        ],
        columns=["概念", "Python asyncio", "Node.js 类比"],
    ).to_string(index=False))
    print("""
    📌 单线程的好处：所有订单 / 持仓状态只在一个线程里修改，不需要锁，
       事件处理顺序确定，出了问题可以按事件日志精确回放。
    """)


# ==================== 第二部分：订阅与发布 ====================


async def pub_sub_demo():
    engine = EventEngine(timer_interval=None)
    log = []
    engine.register_general(lambda e: log.append(f"[通用] 记录 {e.type}"))
    engine.register(EVENT_TICK, lambda e: log.append(f"[策略] 收到行情 {e.data['last']}"))

    async def slow_writer(e):  # async 处理器同样支持
        await asyncio.sleep(0)
        log.append(f"[落盘] 写入行情 {e.data['last']}")

    engine.register(EVENT_TICK, slow_writer)
    engine.start()
    for px in [10.00, 10.01]:
        engine.put(EVENT_TICK, {"symbol": "600000.SSE", "last": px})
    await engine.join()
    await engine.stop()
    return log


def pub_sub():
    print("\n" + "=" * 60)
    print("第二部分：订阅与发布")
    print("=" * 60)
    print("\n📄 mini_trader/engine.py 的核心：取出事件 → 依次调用处理器\n")
    print(inspect.getsource(EventEngine._dispatch))
    for line in asyncio.run(pub_sub_demo()):
        print("   " + line)
    print("""
    💡 处理顺序：通用处理器（如事件日志）先执行，然后按注册顺序执行各类型处理器。
       关键流程（如"风控必须在策略之前更新价格"）不要依赖注册顺序，应显式调用。
    """)


# ==================== 第三部分：异常隔离 ====================


async def isolation_demo():
    engine = EventEngine(timer_interval=None)
    got = []

    def buggy_strategy(e):
        if e.data["last"] > 10.005:
            raise ZeroDivisionError("策略 bug：除以零")
        got.append(f"buggy_strategy 处理 {e.data['last']}")

    engine.register(EVENT_TICK, buggy_strategy)
    engine.register(EVENT_TICK, lambda e: got.append(f"risk_monitor 处理 {e.data['last']}"))
    engine.register(EVENT_ERROR, lambda e: got.append(f"⚠️ error 事件：{e.data['handler']} → {e.data['error']}"))
    engine.start()
    for px in [10.00, 10.01, 10.02]:
        engine.put(EVENT_TICK, {"last": px})
    await engine.join()
    await engine.join()  # error 事件是处理过程中新投递的，再等一轮
    await engine.stop()
    return got, engine


def exception_isolation():
    print("\n" + "=" * 60)
    print("第三部分：异常隔离")
    print("=" * 60)
    got, engine = asyncio.run(isolation_demo())
    for line in got:
        print("   " + line)
    print(f"""
    📊 引擎共处理 {engine.processed} 个事件，捕获 {len(engine.errors)} 个处理器异常，引擎没有退出。

    ⚠️ 如果没有 try/except：第一个异常就会让 _run 协程结束，之后所有事件都没人处理，
       而进程仍然"活着" —— 这就是最危险的"假活"。
       异常被捕获后还要转成 error 事件，交给告警（第 09 课），不能静默吞掉。
    """)


# ==================== 第四部分：跨线程投递 ====================


async def threadsafe_demo():
    engine = EventEngine(timer_interval=None)
    seen = []
    engine.register(EVENT_TICK, lambda e: seen.append((e.data["seq"], e.data["from"], threading.current_thread().name)))
    engine.start()

    def ctp_callback_thread():
        """模拟柜台 C++ API：在自己的线程里回调 OnRtnDepthMarketData"""
        for i in range(5):
            engine.put_threadsafe(EVENT_TICK, {"seq": i, "from": threading.current_thread().name})
            time.sleep(0.01)

    t = threading.Thread(target=ctp_callback_thread, name="ctp-md-thread")
    t.start()
    await asyncio.to_thread(t.join)  # 等待线程结束，同时不阻塞事件循环
    await engine.join()
    await engine.stop()
    return seen


def cross_thread():
    print("\n" + "=" * 60)
    print("第四部分：跨线程投递（call_soon_threadsafe）")
    print("=" * 60)
    print("\n📄 put_threadsafe 的实现：\n")
    print(inspect.getsource(EventEngine.put_threadsafe))
    seen = asyncio.run(threadsafe_demo())
    print(pd.DataFrame(seen, columns=["序号", "投递线程", "处理线程"]).to_string(index=False))
    print("""
    💡 回调发生在 ctp-md-thread，处理却在 MainThread（事件循环线程）里完成。
       为什么不能在回调线程里直接 queue.put_nowait()？
         1. asyncio.Queue 不是线程安全的，并发修改可能破坏内部状态
         2. 事件循环可能正阻塞在 select() 上，不知道有新数据，要等下一次唤醒
       call_soon_threadsafe 会把调用排进循环的待办列表，并主动唤醒事件循环。
       另外：回调线程里不能做耗时操作（柜台 API 的回调线程被占住，后续回报都会延迟）。
    """)


# ==================== 第五部分：定时器与阻塞陷阱 ====================


MODES = {
    "block": "直接调用（time.sleep 阻塞）",
    "await": "await asyncio.to_thread(...)",
    "task": "create_task(to_thread) + 完成后投递事件",
}


async def blocking_demo(mode: str):
    engine = EventEngine(timer_interval=0.05)
    tick_latency, timer_marks, loop_lags = [], [], []
    background: set[asyncio.Task] = set()
    engine.register(EVENT_TICK, lambda e: tick_latency.append((time.perf_counter() - e.data) * 1000))
    engine.register(EVENT_TIMER, lambda e: timer_marks.append(time.perf_counter()))

    def heavy_compute():
        time.sleep(0.2)  # 模拟一次 200 ms 的重活（如同步 HTTP 请求、读大文件）

    async def on_bar(e):
        if mode == "block":
            heavy_compute()  # ❌ 整个事件循环停住
        elif mode == "await":
            await asyncio.to_thread(heavy_compute)  # ⚠️ 循环没停，但引擎在等这个处理器
        else:
            task = asyncio.create_task(asyncio.to_thread(heavy_compute))  # ✅ 后台执行
            background.add(task)
            task.add_done_callback(lambda t: (background.discard(t), engine.put("bar_done")))

    async def loop_probe():
        """独立于引擎的探针：测量事件循环本身被卡住多久"""
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            loop_lags.append((time.perf_counter() - t - 0.01) * 1000)

    def md_thread():
        """行情线程：每 20 ms 推一笔 tick，携带产生时刻，用来测端到端延迟"""
        for i in range(25):
            engine.put_threadsafe(EVENT_TICK, time.perf_counter())
            if i == 5:
                engine.put_threadsafe("bar", None)
            time.sleep(0.02)

    engine.register("bar", on_bar)
    engine.start()
    probe = asyncio.create_task(loop_probe())
    t0 = time.perf_counter()
    feeder = threading.Thread(target=md_thread, name="md-thread")
    feeder.start()
    await asyncio.to_thread(feeder.join)
    await asyncio.sleep(0.25)  # 等后台任务结束
    await engine.join()
    probe.cancel()
    await asyncio.gather(probe, return_exceptions=True)
    await engine.stop()
    gaps = np.diff([t0] + timer_marks) * 1000
    return max(loop_lags), max(tick_latency), gaps.max()


def timer_and_blocking():
    print("\n" + "=" * 60)
    print("第五部分：定时器事件与阻塞陷阱")
    print("=" * 60)
    print("""
    定时器事件（timer）每隔固定时间投递一次，用来驱动"定时对账、心跳、风控巡检"。
    实验设置：50 ms 定时器 + 行情线程每 20 ms 推一笔 tick；第 6 笔行情后触发一次 200 ms 的重活。
    用三个指标观察三种写法：
      - 事件循环卡顿：独立探针测得的循环停顿（循环被卡住时，任何协程都无法运行）
      - tick 端到端延迟：行情线程产生 tick → 处理器拿到 tick
      - 定时器最大间隔：timer 处理器两次执行之间的最大间隔
    """)
    rows = []
    for mode, label in MODES.items():
        lag, tick_lat, gap = asyncio.run(blocking_demo(mode))
        rows.append([label, f"{lag:.0f}", f"{tick_lat:.0f}", f"{gap:.0f}"])
    print(pd.DataFrame(rows, columns=["写法", "循环卡顿max(ms)", "tick延迟max(ms)", "定时器间隔max(ms)"]).to_string(index=False))
    print("""
    💡 解读（对照上表的实际数字）：
       1. 直接阻塞：事件循环整个停住，三个指标都被拉长到重活的量级
       2. await to_thread：循环本身没停（探针正常），但引擎按顺序处理事件，
          正在 await 的处理器会让后面的 tick、timer 一起排队 —— 这是很隐蔽的坑
       3. 后台任务：处理器立即返回，重活在线程池里跑，完成后再投递 bar_done 事件
       规则：处理器里只做轻量计算；重活放后台任务，结果以事件形式回到引擎；
       CPU 密集型任务（受 GIL 限制）放到独立进程。
    """)


# ==================== 第六部分：行情 → 策略 → 下单 ====================

SYMBOL = "m2501.DCE"  # 豆粕期货（非上期所，平仓不区分今昨，便于演示）


class MaCrossStrategy:
    """双均线策略：只输出"目标持仓"（-1 / 0 / +1），不直接下单"""

    def __init__(self, engine: EventEngine, fast: int = 5, slow: int = 20):
        self.engine, self.fast, self.slow = engine, fast, slow
        self.prices: list[float] = []
        self.target = 0

    def on_tick(self, event):
        self.prices.append(event.data.last_price)
        if len(self.prices) < self.slow:
            return
        fast_ma = np.mean(self.prices[-self.fast:])
        slow_ma = np.mean(self.prices[-self.slow:])
        target = 1 if fast_ma > slow_ma else -1
        if target != self.target:
            self.target = target
            self.engine.put(EVENT_SIGNAL, {"symbol": SYMBOL, "target": target, "tick": event.data, "tick_ts": event.ts})


class TargetExecutor:
    """把"目标持仓"翻译成具体订单：先平反向仓，再开新仓"""

    SLIP_TICKS = 2  # 可成交限价单：比对手价多让 2 个价位，降低"行情已变化导致不成交"的概率

    def __init__(self, gateway: MockGateway, oms: OMS):
        self.gateway, self.oms = gateway, oms
        self.seq = 0
        self.latency_ms: list[float] = []

    def net_position(self) -> float:
        """净持仓 = 已成交持仓 + 在途订单（尚未成交的部分也要算上，否则会重复下单）"""
        filled = (
            self.oms.positions.get(SYMBOL, Direction.LONG).volume
            - self.oms.positions.get(SYMBOL, Direction.SHORT).volume
        )
        pending = sum(
            (1 if o.direction is Direction.LONG else -1) * o.remaining for o in self.oms.active_orders(SYMBOL)
        )
        return filled + pending

    async def on_signal(self, event):
        sig, tick = event.data, event.data["tick"]
        delta = sig["target"] - self.net_position()
        if delta == 0:
            return
        direction = Direction.LONG if delta > 0 else Direction.SHORT
        # 可平数量已扣除在途平仓单的冻结量
        close_qty = min(abs(delta), self.oms.positions.closable(SYMBOL, opposite(direction), Offset.CLOSE))
        for offset, volume in [(Offset.CLOSE, close_qty), (Offset.OPEN, abs(delta) - close_qty)]:
            if volume <= 0:
                continue
            self.seq += 1
            if direction is Direction.LONG:
                price = tick.ask_price_1 + self.SLIP_TICKS
            else:
                price = tick.bid_price_1 - self.SLIP_TICKS
            req = OrderRequest(SYMBOL, direction, price, volume, offset, f"MA-{self.seq:04d}")
            self.oms.create_order(req)
            self.latency_ms.append((time.perf_counter() - sig["tick_ts"]) * 1000)
            await self.gateway.send_order(req)


async def full_pipeline():
    engine = EventEngine(timer_interval=None)
    exchange = MockExchange(initial_cash=200_000)
    gateway = MockGateway(engine, exchange)
    oms = OMS()
    strategy = MaCrossStrategy(engine)
    executor = TargetExecutor(gateway, oms)
    trades = []

    engine.register(EVENT_TICK, strategy.on_tick)
    engine.register(EVENT_SIGNAL, executor.on_signal)
    engine.register(EVENT_ORDER, lambda e: oms.on_order(e.data))
    engine.register(EVENT_TRADE, lambda e: (oms.on_trade(e.data), trades.append(e.data)))
    engine.start()
    await gateway.connect()
    await gateway.subscribe(SYMBOL)

    rng = np.random.default_rng(13)
    price = 3000.0
    for _ in range(300):  # 300 笔模拟行情
        price = round(price + rng.normal(0, 3))
        exchange.set_market(SYMBOL, last=price)
        await asyncio.sleep(0)  # 让出控制权，事件循环处理行情
    await engine.join()
    await engine.join()
    await engine.stop()
    return oms, trades, executor, engine, exchange, strategy


def pipeline():
    print("\n" + "=" * 60)
    print("第六部分：行情 → 策略 → 下单 → 成交")
    print("=" * 60)
    print("""
      MockExchange.set_market ──tick──▶ MaCrossStrategy ──signal(目标持仓)──▶ TargetExecutor
            ▲                                                                   │
            └──────────── send_order ◀──────── OMS.create_order ◀───────────────┘
            │ order / trade 回报 ──▶ OMS（更新订单与持仓）
    """)
    oms, trades, executor, engine, exchange, strategy = asyncio.run(full_pipeline())
    df = pd.DataFrame(
        [[t.trade_id, t.client_order_id, t.direction.value, t.offset.value, t.price, f"{t.volume:g}"] for t in trades],
        columns=["成交号", "订单号", "方向", "开平", "价格", "手数"],
    )
    print(f"共 {len(trades)} 笔成交，前 8 笔：")
    print(df.head(8).to_string(index=False))

    order_seq = [t.client_order_id for t in trades]
    late = [c for i, c in enumerate(order_seq) if any(prev > c for prev in order_seq[:i])]
    print(f"\n   成交顺序 ≠ 下单顺序的订单：{sorted(set(late)) or '无'}")
    print("   （发单时行情已经变化，限价单先挂着，之后才成交 —— 信号与成交之间存在时间差）")

    long_p = oms.positions.get(SYMBOL, Direction.LONG)
    short_p = oms.positions.get(SYMBOL, Direction.SHORT)
    realized = long_p.realized_pnl + short_p.realized_pnl
    lat = np.array(executor.latency_ms)
    print(f"""
    📊 最终持仓：多 {long_p.volume:g} 手 / 空 {short_p.volume:g} 手（净 {long_p.volume - short_p.volume:+g}，策略最后的目标 {strategy.target:+d}）
       已实现盈亏 {realized:,.0f} 元（合约乘数 10，未计手续费）
    📊 交易所侧持仓：{exchange.positions.to_volume_map()}
    ⏱️ 信号产生 → 订单发出：p50 {np.median(lat):.3f} ms，max {lat.max():.3f} ms
    ⏱️ {wait_stats(engine)}

    💡 要点：
       1. 策略只发布 signal 事件，从不碰网关 —— 换成实盘网关策略代码无需修改
       2. 执行器按"目标 - (已成交持仓 + 在途订单)"计算订单：重复收到同一个目标不会多下单
       3. 信号处理时行情可能已经变了，所以用"可成交限价单"（多让几个价位，Phase 11）
       4. 本例省略了风控，第 04 课会在 OMS 之前插入风控检查
    """)


# ==================== 主函数 ====================


def main():
    why_event_driven()
    pub_sub()
    exception_isolation()
    cross_thread()
    timer_and_blocking()
    pipeline()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 EventEngine 加一个"行情合并"功能：同一标的的 tick 在队列中只保留最新一条，
       比较行情洪峰时开启前后的排队延迟
    2. 第五部分中把 time.sleep 换成纯 CPU 计算（如大矩阵求逆），to_thread 还能解决问题吗？为什么？
    3. 第六部分中，开仓单还在途时收到反向信号，执行器会直接反向开仓，形成"锁仓"（多空同时持有）。
       修改执行器：先撤掉在途订单、等撤单回报后再按目标下单，比较两种做法的成交笔数和手续费
    4. 写一个处理器统计每类事件的数量，在引擎停止时打印出来（提示：register_general）
    """)


if __name__ == "__main__":
    main()
