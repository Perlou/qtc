"""
监控与告警
==========

学习目标：
    1. 掌握交易系统的监控四层模型：账户层、交易层、连接层、系统层
    2. 实现业务级心跳：行情断流检测、组件心跳（Dead Man's Switch）
    3. 采集核心指标：报单数、拒单率、下单往返延迟、事件队列长度
    4. 实现告警分级（P0-P3）、去重和抑制，避免告警风暴
    5. 用 logging 输出 JSON 结构化日志，用 client_order_id 串起一笔订单的完整链路

核心概念：
    - 假活：进程还在，但行情已经不更新了 —— 进程存活检查发现不了
    - Dead Man's Switch：系统定期报平安，监控方超时未收到就反向告警
    - 告警疲劳：每天几十条无用告警，会让人忽略真正的 P0
    - 结构化日志：一行一个 JSON，便于检索、聚合和回放

前置知识：
    - 02 课（事件引擎、定时器）、04 课（Kill Switch）
    - 本阶段 CONCEPT.md 第 10 节

环境要求：
    - 不需要网络；日志写到 outputs/phase-13/logs/，图表写到 outputs/phase-13/
    - 演示时间已按比例压缩（用 0.15 秒代表实盘里的几十秒静默），阈值都是演示值
"""

import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from mini_trader import Direction, MockExchange, RiskConfig
from mini_trader.events import EVENT_ERROR, EVENT_ORDER, EVENT_TICK, EVENT_TIMER, EVENT_TRADE
from mini_trader.monitor import (
    LEVELS,
    AlertManager,
    HeartbeatWatchdog,
    JsonFormatter,
    Metrics,
    StalenessMonitor,
    get_json_logger,
)
from mini_trader.system import TradingSystem
from utils import OUTPUT_DIR
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SYM_A, SYM_B = "600000.SSE", "000001.SZSE"
LOG_PATH = OUTPUT_DIR / "phase-13" / "logs" / "09_trading.jsonl"


# ==================== 第一部分：监控四层模型 ====================


def four_layers():
    print("=" * 60)
    print("第一部分：监控四层模型")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────────────────────┐
    │ 账户层   PnL、回撤、保证金占用率、持仓与目标偏离                    │
    │ 交易层   报单数、拒单率、撤单率、成交率、滑点、订单滞留时间           │
    │ 连接层   网关连接状态、最后一笔行情时间、回报延迟、重连次数            │
    │ 系统层   进程存活、CPU/内存、磁盘、事件队列长度、时钟偏差              │
    │                                                                  │
    │ 越往上越贴近"钱"，越往下越贴近"机器" —— 两者都要看                  │
    └──────────────────────────────────────────────────────────────────┘

    ⚠️ 最隐蔽的故障是"系统在运行，但行情已经不更新了"。
       进程存活检查（ps、容器 healthcheck 只看进程）发现不了，策略会基于过期价格继续决策。
    """)


# ==================== 第二、三部分：运行一个被监控的系统 ====================


class MonitorService:
    """挂在事件引擎上的监控组件：只读事件，不影响交易主流程"""

    def __init__(self, system: TradingSystem, alerts: AlertManager, logger, t0: float):
        self.system, self.alerts, self.logger, self.t0 = system, alerts, logger, t0
        self.metrics = Metrics()
        self.stale = StalenessMonitor(max_silence_sec=0.15)
        self.watchdog = HeartbeatWatchdog(timeout_sec=0.25)
        self.tick_times: list[tuple[float, str]] = []
        self.alert_times: list[tuple[float, str, str]] = []
        for sym in (SYM_A, SYM_B):
            self.stale.watch(sym)
        self.watchdog.beat("md_gateway")
        engine = system.engine
        engine.register(EVENT_TICK, self.on_tick)
        engine.register(EVENT_ORDER, self.on_order)
        engine.register(EVENT_TRADE, lambda e: self.metrics.inc("trades"))
        engine.register(EVENT_ERROR, lambda e: self.fire("P1", "engine.error", "事件处理器异常", e.data["error"]))
        engine.register(EVENT_TIMER, self.on_timer)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def fire(self, level: str, key: str, title: str, detail: str = "") -> None:
        outcome = self.alerts.fire(level, key, title, detail)
        if outcome == "sent":
            self.alert_times.append((self.now(), level, key))

    def on_tick(self, event) -> None:
        self.stale.on_tick(event.data.symbol)
        self.watchdog.beat("md_gateway")
        self.metrics.inc("ticks")
        self.tick_times.append((self.now(), event.data.symbol))

    def on_order(self, event) -> None:
        order = event.data
        if order.status.name == "REJECTED":
            self.metrics.inc("orders_rejected")
            self.logger.warning("order_rejected", extra={"ctx": {"coid": order.client_order_id, "reason": order.reason}})
        elif order.status.name == "ALL_TRADED":
            self.logger.info("order_filled", extra={"ctx": {"coid": order.client_order_id, "traded": order.traded}})

    def on_timer(self, event) -> None:
        engine = self.system.engine
        self.metrics.set("queue_size", engine.queue_size)
        stale = self.stale.check(in_session=True)
        everything_stale = len(stale) == len(self.stale.last_seen)
        for sym, silent in stale.items():
            level = "P1" if everything_stale else "P2"
            self.fire(level, f"md.stale.{sym}", f"行情停滞 {sym}", f"已静默 {silent:.2f}s")
        for source in self.watchdog.dead():
            self.fire("P1", f"heartbeat.{source}", f"组件心跳丢失：{source}", "可能已断线或假活")
        if self.metrics.ratio("orders_rejected", "orders_sent") > 0.3 and self.metrics.counters["orders_sent"] >= 5:
            self.fire("P1", "trade.reject_rate", "拒单率异常", f"{self.metrics.ratio('orders_rejected', 'orders_sent'):.0%}")


async def monitored_session(logger):
    exchange = MockExchange(initial_cash=1_000_000)
    system = TradingSystem(
        exchange,
        RiskConfig(max_orders_per_window=100, max_order_notional=1e7),
        timer_interval=0.05,
        gateway_kwargs={"request_latency": 0.002, "response_latency": 0.002},
    )
    t0 = time.perf_counter()
    alerts = AlertManager(dedupe_window_sec=0.4, clock=time.perf_counter)
    monitor = MonitorService(system, alerts, logger, t0)
    await system.start()
    for sym in (SYM_A, SYM_B):
        await system.gateway.subscribe(sym)
    px = {SYM_A: 10.0, SYM_B: 12.0}
    exchange.inject("reject", count=2)
    for step in range(70):  # 约 1.4 秒
        t = monitor.now()
        if t < 0.5:  # 阶段 1：两个标的都有行情
            active = (SYM_A, SYM_B)
        elif t < 0.9:  # 阶段 2：SYM_B 行情停滞
            active = (SYM_A,)
        else:  # 阶段 3：行情全部中断
            active = ()
        for sym in active:
            exchange.set_market(sym, px[sym])
        await system.engine.join()  # 确保风控已收到最新行情
        if step % 8 == 0 and active:
            req = system.new_request(SYM_A, Direction.LONG, 10.01, 100)
            logger.info("order_sent", extra={"ctx": {"coid": req.client_order_id, "symbol": req.symbol, "price": req.price, "volume": req.volume}})
            start = time.perf_counter()
            result = await system.send_order(req)
            monitor.metrics.observe("order_rtt_ms", (time.perf_counter() - start) * 1000)
            monitor.metrics.inc("orders_sent")
            logger.info("order_ack", extra={"ctx": {"coid": req.client_order_id, "outcome": result.outcome}})
        await asyncio.sleep(0.02)
    await system.stop()
    return monitor, alerts


def heartbeat_and_metrics():
    print("\n" + "=" * 60)
    print("第二部分：心跳与行情断流检测（运行一个被监控的系统）")
    print("=" * 60)
    print("\n📄 mini_trader/monitor.py 中的行情断流检测：\n")
    print(inspect.getsource(StalenessMonitor.check))
    print("""
    实验分三个阶段（时间已压缩）：
      阶段 1  两个标的都有行情，正常下单（前 2 笔注入拒单）
      阶段 2  000001.SZSE 行情停滞（单标的问题）→ P2
      阶段 3  全部行情中断（疑似网关断线）→ P1
    """)
    logger = get_json_logger("trading", LOG_PATH)
    monitor, alerts = asyncio.run(monitored_session(logger))
    logger.info("session_end", extra={"ctx": monitor.metrics.counters})
    for h in logger.handlers:
        h.flush()

    rows = [[f"{t:.2f}s", lvl, key] for t, lvl, key in monitor.alert_times]
    print("🔔 实际发出的告警：")
    print(pd.DataFrame(rows, columns=["时刻", "级别", "告警键"]).to_string(index=False))
    print(f"\n   告警统计：发出 {alerts.stats['sent']} 条，被去重合并 {alerts.stats['deduped']} 条"
          f"（定时器每 50 ms 检查一次，没有去重就是告警风暴）")

    print(f"\n   非交易时段的检查结果：{monitor.stale.check(in_session=False)}（休市没有行情是正常的，不告警）")
    print("   进程本身一直存活 → 只看进程存活的监控在整个过程中都认为系统健康")

    print("\n" + "=" * 60)
    print("第三部分：指标采集")
    print("=" * 60)
    m = monitor.metrics
    rtt = m.summary("order_rtt_ms")
    print(pd.DataFrame(
        [
            ["连接层", "收到行情笔数", f"{m.counters['ticks']:.0f}"],
            ["交易层", "报单数", f"{m.counters['orders_sent']:.0f}"],
            ["交易层", "拒单数", f"{m.counters['orders_rejected']:.0f}"],
            ["交易层", "拒单率", f"{m.ratio('orders_rejected', 'orders_sent'):.1%}"],
            ["交易层", "成交笔数", f"{m.counters['trades']:.0f}"],
            ["连接层", "下单往返 p50 / p95 / max", f"{rtt['p50']:.1f} / {rtt['p95']:.1f} / {rtt['max']:.1f} ms"],
            ["系统层", "事件引擎处理事件数", f"{monitor.system.engine.processed}"],
            ["系统层", "事件队列历史最大长度", f"{monitor.system.engine.max_queue_size}"],
        ],
        columns=["层", "指标", "值"],
    ).to_string(index=False))
    print("""
    📌 指标可以用 prometheus_client 暴露，用 Grafana 做看板；个人项目也可以先写入 SQLite，
       配一个简单网页看板起步。关键是"有历史数据可看"，出问题时才能判断是什么时候开始变坏的。
    """)
    plot_timeline(monitor)
    return monitor


def plot_timeline(monitor: MonitorService):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    series = {SYM_A: (1, "#2a78d6"), SYM_B: (0, "#1baf7a")}  # 分类色：标的身份
    df = pd.DataFrame(monitor.tick_times, columns=["t", "sym"])
    for sym, (y, color) in series.items():
        t = df.loc[df["sym"] == sym, "t"]
        ax.plot(t, [y] * len(t), "|", ms=12, mew=2, color=color, label=f"{sym} 行情")
    status = {"P1": ("#d03b3b", "--"), "P2": ("#ec835a", ":")}  # 状态色：告警级别，配合线型区分
    seen = set()
    for t, level, _ in monitor.alert_times:
        color, ls = status.get(level, ("#52514e", "-."))
        ax.axvline(t, color=color, ls=ls, lw=1.5, label=None if level in seen else f"{level} 告警")
        seen.add(level)
    ax.set_yticks([0, 1], [SYM_B, SYM_A])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("时间（秒）")
    ax.set_title("行情到达与告警时间线", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8, frameon=False)
    path = save_figure(fig, "09_monitor_timeline.png", "phase-13")
    print(f"📈 时间线图已保存：{path}")


# ==================== 第四部分：告警分级、去重与抑制 ====================


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def alert_policy():
    print("\n" + "=" * 60)
    print("第四部分：告警分级、去重与抑制")
    print("=" * 60)
    print(pd.DataFrame([[k, v] for k, v in LEVELS.items()], columns=["级别", "含义与响应"]).to_string(index=False))

    clock = FakeClock()
    delivered = []
    mgr = AlertManager(dedupe_window_sec=300, clock=clock, notifier=lambda a: delivered.append(a))
    script = [
        (0, "P1", "gw.disconnect", "网关断线", "首次出现"),
        (60, "P1", "gw.disconnect", "网关断线", "1 分钟后再次触发"),
        (240, "P1", "gw.disconnect", "网关断线", "4 分钟后再次触发"),
        (310, "P1", "gw.disconnect", "网关断线", "超过 5 分钟的去重窗口"),
        (320, "P2", "md.stale.rb2501", "行情延迟", "同时出现另一个问题"),
    ]
    rows = []
    for t, level, key, title, note in script:
        clock.t = t
        rows.append([f"t={t}s", level, key, mgr.fire(level, key, title), note])

    clock.t = 400
    mgr.silence("md.", duration_sec=1800, reason="计划内维护：行情服务器升级")
    rows.append(["t=400s", "P2", "md.stale.rb2501", mgr.fire("P2", "md.stale.rb2501", "行情延迟"), "维护窗口内：被静默"])
    rows.append(["t=400s", "P0", "md.kill_switch", mgr.fire("P0", "md.kill_switch", "Kill Switch 触发"), "P0 不受静默影响"])
    clock.t = 2300
    rows.append(["t=2300s", "P2", "md.stale.rb2501", mgr.fire("P2", "md.stale.rb2501", "行情延迟"), "静默结束后恢复告警"])
    print()
    print(pd.DataFrame(rows, columns=["时间", "级别", "告警键", "处理结果", "说明"]).to_string(index=False))
    print(f"\n   共触发 {len(rows)} 次，实际通知 {len(delivered)} 次；被合并的重复次数记录在告警上："
          f"gw.disconnect 首条 repeats={delivered[0].repeats}")
    print("""
    📌 告警设计原则：
    - 去重与抑制：同一问题 5 分钟内只告警一次；计划内维护时静默
    - 可操作：每条告警说明"发生了什么、影响是什么、建议做什么"
    - 告警疲劳是真实风险：无用告警太多，真正的 P0 会被忽略
    - P0 永不静默；告警链路本身也要定期实测（手机真的能收到吗？）

    🐕 Dead Man's Switch：监控系统自己挂了怎么办？
       让系统定期向外部服务发送"我还活着"，外部服务超时未收到就反向告警。
       本课的 HeartbeatWatchdog 演示了同样的思路：组件停止 beat() → 被 dead() 发现。
    """)


# ==================== 第五部分：结构化日志 ====================


def structured_logging():
    print("\n" + "=" * 60)
    print("第五部分：结构化日志")
    print("=" * 60)
    print("\n📄 JsonFormatter（用标准库 logging 输出 JSON）：\n")
    print(inspect.getsource(JsonFormatter.format))
    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    print(f"   日志文件：{LOG_PATH}（共 {len(records)} 行），前 3 行：")
    for rec in records[:3]:
        print("   " + json.dumps(rec, ensure_ascii=False))

    df = pd.DataFrame(records)
    print("\n   按 msg 统计（结构化日志可以像数据一样分析）：")
    print(df.groupby(["level", "msg"]).size().rename("count").reset_index().to_string(index=False))
    rejected = df.loc[df["msg"] == "order_rejected", "coid"].iloc[0]
    print(f"\n   用 coid 串起订单 {rejected} 的完整链路（类似分布式系统的 trace id）：")
    print(df.loc[df["coid"] == rejected, ["ts", "level", "msg", "outcome", "reason"]].fillna("-").to_string(index=False))
    print("""
    📌 每条订单相关日志都带 client_order_id；日志里绝不能出现密钥和密码。
    """)


# ==================== 主函数 ====================


def main():
    four_layers()
    heartbeat_and_metrics()
    alert_policy()
    structured_logging()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 MonitorService 接到第 04 课的 RiskEngine：全部行情中断时自动触发 Kill Switch，并发 P0 告警
    2. 给告警增加"升级"：P2 持续 30 分钟未恢复时自动升级为 P1
    3. 增加"恢复通知"：行情恢复后发送一条 resolved 告警，并计算故障持续时间
    4. 把 Metrics 改成 Prometheus 文本格式输出（不需要安装 prometheus_client），思考哪些指标应该用 counter，哪些用 gauge
    """)


if __name__ == "__main__":
    main()
