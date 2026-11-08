"""
vn.py 框架入门
==============

学习目标：
    1. 了解 vn.py 的模块化架构：MainEngine、EventEngine、Gateway、App、OmsEngine
    2. 读懂 CtaTemplate 策略的写法：回调函数、buy / sell / short / cover、self.pos
    3. 在 mini_trader 上实现一个 CtaTemplate 风格的适配层，用同样的写法跑通双均线策略
    4. 建立"自研骨架 ↔ vn.py"的概念对应，知道生产中哪些部分该用框架、哪些该自建

核心概念：
    - vn.py：国内广泛使用的开源 Python 量化交易框架，核心 + 按需安装的接口 / 应用插件
    - CtaTemplate：CTA 策略模板，通过 on_init / on_tick / on_bar / on_order / on_trade 响应事件
    - self.pos：策略内部记录的持仓，不等于账户真实持仓（多个策略可能交易同一合约）

前置知识：
    - 01~04 课
    - 本阶段 CONCEPT.md 第 6 节

环境要求：
    - 本课不安装、不导入 vn.py；vn.py 代码以字符串形式展示，只做语法检查，不执行
    - vn.py 的类名、方法与参数以所安装版本的源码和官方文档为准
"""

import ast
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from mini_trader import OMS, BarData, Direction, MockExchange, Offset, OrderRequest
from utils.data import synthetic_ohlcv

load_dotenv()


# ==================== 第一部分：vn.py 是什么 ====================


def what_is_vnpy():
    print("=" * 60)
    print("第一部分：vn.py 是什么")
    print("=" * 60)
    installed = {m: importlib.util.find_spec(m) is not None for m in ["vnpy", "vnpy_ctp", "vnpy_ctastrategy"]}
    print(f"\n   本机安装情况：{installed}（本课不需要安装）")
    print("""
    vn.py 采用"核心框架 + 插件"的模块化设计（模块名以官方文档为准）：

    ┌──────────────────┬───────────────────────────────────────────┐
    │ 模块             │ 作用                                      │
    ├──────────────────┼───────────────────────────────────────────┤
    │ vnpy             │ 核心：事件引擎、主引擎、数据结构、GUI      │
    │ vnpy_ctp 等      │ 交易接口（Gateway），每个柜台一个包        │
    │ vnpy_ctastrategy │ CTA 策略应用：CtaTemplate、实盘运行        │
    │ vnpy_ctabacktester│ CTA 回测应用                              │
    │ vnpy_datamanager │ 历史数据管理                              │
    └──────────────────┴───────────────────────────────────────────┘

                          MainEngine（主引擎）
               ┌───────────────────┼────────────────────┐
               ▼                   ▼                    ▼
        Gateway（接口）        App（应用）          OmsEngine 等
        CtpGateway ...       CtaStrategyApp       订单 / 持仓 / 账户缓存
               │                   ▲
               ▼                   │
        ═══════════════ EventEngine（事件引擎）═══════════════
        事件类型：EVENT_TICK / ORDER / TRADE / POSITION / ACCOUNT / TIMER

    💡 与第 01~04 课的架构几乎一一对应：网关推送事件 → 事件引擎分发 → OMS 缓存状态 → 应用处理。
       区别之一：vn.py 的事件引擎基于"线程 + 队列"实现，本课程用的是 asyncio（以 vn.py 源码为准）。
    """)


# ==================== 第二部分：CtaTemplate 写法（示意代码） ====================

VNPY_CTA_STRATEGY = '''
from vnpy_ctastrategy import CtaTemplate, BarGenerator, ArrayManager

class DoubleMaStrategy(CtaTemplate):
    author = "qtc"
    fast_window = 10
    slow_window = 30
    fixed_size = 1
    parameters = ["fast_window", "slow_window", "fixed_size"]   # 可在界面上配置的参数
    variables = ["fast_ma", "slow_ma"]                         # 运行时显示的变量

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)      # Tick 合成 1 分钟 K 线
        self.am = ArrayManager()                 # 缓存 K 线序列，计算指标
        self.fast_ma = self.slow_ma = 0.0

    def on_init(self):
        self.load_bar(10)                        # 加载历史数据预热指标

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
'''


def cta_template():
    print("\n" + "=" * 60)
    print("第二部分：CtaTemplate 写法（示意代码，不执行）")
    print("=" * 60)
    print(VNPY_CTA_STRATEGY)
    tree = ast.parse(VNPY_CTA_STRATEGY)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    calls = sorted({
        n.func.attr for n in ast.walk(cls)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
    })
    print(f"   ✅ 语法检查通过（ast.parse），类 {cls.name} 定义的回调：{methods}")
    print(f"   策略调用的模板方法：{calls}")
    print("""
    📌 关键点：
    - buy / sell / short / cover = 开多 / 平多 / 开空 / 平空 —— 期货"方向 + 开平"语义的直接体现
    - on_bar 开头先 cancel_all()：上一根 K 线没成交的单撤掉，按最新信号重新下单
    - self.pos 是策略自己记录的持仓：多个策略交易同一合约时，它不等于账户持仓（第四部分演示）
    """)


# ==================== 第三部分：在 mini_trader 上实现同样的写法 ====================


class MiniCtaTemplate:
    """CtaTemplate 风格的适配层：策略写法与 vn.py 相同，底层换成 mini_trader"""

    def __init__(self, engine: "MiniCtaEngine", name: str, symbol: str, setting: dict | None = None):
        self.engine, self.strategy_name, self.symbol = engine, name, symbol
        self.pos = 0.0  # 策略自己的持仓：只由本策略的成交更新
        for k, v in (setting or {}).items():
            setattr(self, k, v)

    def buy(self, price, volume):
        return self.engine.send(self, Direction.LONG, Offset.OPEN, price, volume)

    def sell(self, price, volume):
        return self.engine.send(self, Direction.SHORT, Offset.CLOSE, price, volume)

    def short(self, price, volume):
        return self.engine.send(self, Direction.SHORT, Offset.OPEN, price, volume)

    def cover(self, price, volume):
        return self.engine.send(self, Direction.LONG, Offset.CLOSE, price, volume)

    def cancel_all(self):
        self.engine.cancel_all(self)

    def on_bar(self, bar: BarData): ...

    def on_trade(self, trade): ...


class MiniDoubleMa(MiniCtaTemplate):
    fast_window, slow_window, fixed_size = 10, 30, 1
    slip = 5  # 可成交限价：对手价再让 5 个价位

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closes: list[float] = []

    def on_bar(self, bar):
        self.cancel_all()
        self.closes.append(bar.close)
        if len(self.closes) < self.slow_window:
            return
        fast = np.mean(self.closes[-self.fast_window:])
        slow = np.mean(self.closes[-self.slow_window:])
        if fast > slow and self.pos <= 0:
            if self.pos < 0:
                self.cover(bar.close + self.slip, abs(self.pos))
            self.buy(bar.close + self.slip, self.fixed_size)
        elif fast < slow and self.pos >= 0:
            if self.pos > 0:
                self.sell(bar.close - self.slip, abs(self.pos))
            self.short(bar.close - self.slip, self.fixed_size)


class MiniCtaEngine:
    """相当于 vn.py 的 CtaEngine：管理多个策略，把订单路由到网关，把成交分发回策略"""

    def __init__(self, exchange: MockExchange):
        self.exchange = exchange
        self.now = None
        exchange.clock = lambda: self.now  # 成交时间使用 K 线时间
        self.oms = OMS()
        self.strategies: list[MiniCtaTemplate] = []
        self.order_owner: dict[str, MiniCtaTemplate] = {}
        self.seq = 0
        exchange.add_listener(self._on_report)

    def add_strategy(self, cls, name: str, symbol: str, setting: dict | None = None):
        s = cls(self, name, symbol, setting)
        self.strategies.append(s)
        return s

    def send(self, strategy, direction, offset, price, volume) -> str:
        self.seq += 1
        req = OrderRequest(strategy.symbol, direction, price, volume, offset, f"{strategy.strategy_name}-{self.seq:04d}")
        self.order_owner[req.client_order_id] = strategy
        self.oms.create_order(req)
        self.exchange.submit(req)
        return req.client_order_id

    def cancel_all(self, strategy):
        for o in self.oms.active_orders(strategy.symbol):
            if self.order_owner.get(o.client_order_id) is strategy:
                self.exchange.cancel(o.client_order_id)

    def _on_report(self, kind, data):
        if kind == "order":
            self.oms.on_order(data)
        elif kind == "trade" and self.oms.on_trade(data):
            s = self.order_owner[data.client_order_id]
            s.pos += data.volume if data.direction is Direction.LONG else -data.volume
            s.on_trade(data)

    def run_bars(self, bars: list[BarData]):
        """日线：每根 K 线是一个交易日，收盘后结算（今仓转昨仓）"""
        for bar in bars:
            self.now = bar.dt
            self.exchange.set_market(bar.symbol, last=bar.close)
            for s in self.strategies:
                if s.symbol == bar.symbol:
                    s.on_bar(bar)
            self.exchange.settle()
            self.oms.positions.settle()


def make_bars(symbol: str) -> list[BarData]:
    """合成日线（几何布朗运动），价格取整到 1 元最小变动价位；仅用于演示"""
    df = synthetic_ohlcv("rb_demo", "20240101", "20241231", start_price=3500, annual_vol=0.25)
    return [
        BarData(symbol, ts.to_pydatetime(), round(r.open), round(r.high), round(r.low), round(r.close), r.volume, "1d")
        for ts, r in df.iterrows()
    ]


def mini_cta_demo():
    print("\n" + "=" * 60)
    print("第三部分：用 mini_trader 跑同样写法的双均线策略")
    print("=" * 60)
    symbol = "rb2501.SHFE"
    bars = make_bars(symbol)
    exchange = MockExchange(initial_cash=1_000_000)
    engine = MiniCtaEngine(exchange)
    strat = engine.add_strategy(MiniDoubleMa, "DMA", symbol)
    engine.run_bars(bars)

    trades = engine.oms.trades
    df = pd.DataFrame(
        [[t.dt.date(), t.client_order_id, f"{t.direction.value}{t.offset.value}", t.price, f"{t.volume:g}"] for t in trades],
        columns=["日期", "订单", "操作", "价格", "手数"],
    )
    print(f"   合成日线 {len(bars)} 根（{bars[0].dt.date()} ~ {bars[-1].dt.date()}，仅用于演示），成交 {len(trades)} 笔，前 6 笔：")
    print(df.head(6).to_string(index=False))
    long_p = exchange.positions.get(symbol, Direction.LONG)
    short_p = exchange.positions.get(symbol, Direction.SHORT)
    rejected = [o for o in engine.oms.orders.values() if o.status.name == "REJECTED"]
    print(f"""
    📊 策略 self.pos = {strat.pos:+g}；交易所持仓：多 {long_p.volume:g} / 空 {short_p.volume:g}；被拒订单 {len(rejected)} 笔
    📊 已实现盈亏（未计手续费，乘数 10）：{long_p.realized_pnl + short_p.realized_pnl:,.0f} 元
    """)

    # 对比实验：同一交易日内开仓、再用 CLOSE 平仓（上期所 CLOSE 视为平昨）
    ex2 = MockExchange()
    ex2.set_market(symbol, 3500)
    ex2.submit(OrderRequest(symbol, Direction.LONG, 3505, 1, Offset.OPEN, "D-1"))
    bad = ex2.submit(OrderRequest(symbol, Direction.SHORT, 3495, 1, Offset.CLOSE, "D-2"))
    good = ex2.submit(OrderRequest(symbol, Direction.SHORT, 3495, 1, Offset.CLOSE_TODAY, "D-3"))
    print(f"""    🔬 对比实验（上期所合约，同一交易日内开仓后平仓）：
       用 CLOSE 平仓       → {bad.status.value}：{bad.reason}
       用 CLOSE_TODAY 平仓 → {good.status.value}

    💡 日线策略的持仓都是之前交易日开的（昨仓），CLOSE 可以成交；
       日内策略在上期所 / 能源中心必须显式平今。vn.py 的 sell / cover 默认发 CLOSE，
       接口层的 OffsetConverter 负责按今昨仓拆分（第 06 课详细讲解）。
    """)


# ==================== 第四部分：self.pos ≠ 账户持仓 ====================


def pos_vs_account():
    print("\n" + "=" * 60)
    print("第四部分：self.pos ≠ 账户持仓")
    print("=" * 60)
    symbol = "m2501.DCE"
    bars = [BarData(symbol, b.dt, b.open, b.high, b.low, b.close, b.volume, "1d") for b in make_bars(symbol)]
    exchange = MockExchange(initial_cash=1_000_000)
    engine = MiniCtaEngine(exchange)
    fast = engine.add_strategy(MiniDoubleMa, "FAST", symbol, {"fast_window": 5, "slow_window": 20})
    slow = engine.add_strategy(MiniDoubleMa, "SLOW", symbol, {"fast_window": 20, "slow_window": 60, "fixed_size": 2})
    snapshots = []
    for i, bar in enumerate(bars):
        engine.run_bars([bar])
        if i % 50 == 49:
            lp = exchange.positions.get(symbol, Direction.LONG).volume
            sp = exchange.positions.get(symbol, Direction.SHORT).volume
            snapshots.append([bar.dt.date(), f"{fast.pos:+g}", f"{slow.pos:+g}", f"{fast.pos + slow.pos:+g}", f"多{lp:g}/空{sp:g}"])
    print(pd.DataFrame(snapshots, columns=["日期", "FAST.pos", "SLOW.pos", "两策略合计", "账户持仓"]).to_string(index=False))
    print("""
    💡 两个策略交易同一合约：每个策略只看到自己的 pos，账户里是两者的叠加（可能同时有多有空）。
       对账时要比较的是"所有策略 pos 之和"与"账户持仓"，而不是单个策略的 pos。
       大连商品交易所的平仓不区分今昨（MockExchange 按先平昨再平今处理），所以本例不会因平今被拒。
    """)


# ==================== 第五部分：概念对应与选型 ====================


def mapping():
    print("\n" + "=" * 60)
    print("第五部分：vn.py ↔ mini_trader 概念对应")
    print("=" * 60)
    table = pd.DataFrame(
        [
            ["EventEngine（线程 + 队列）", "engine.EventEngine（asyncio）", "02"],
            ["MainEngine", "system.TradingSystem", "04"],
            ["BaseGateway / CtpGateway", "gateway.BaseGateway / MockGateway", "01"],
            ["OmsEngine", "oms.OMS", "03"],
            ["TickData / OrderData / TradeData ...", "models.*（字段做了精简）", "01"],
            ["vt_symbol（如 rb2501.SHFE）", "symbol（同样的 <代码>.<交易所>）", "01"],
            ["vt_orderid", "client_order_id", "03"],
            ["CtaTemplate", "MiniCtaTemplate（本课）", "05"],
            ["风控类应用（如 vnpy_riskmanager）", "risk.RiskEngine", "04"],
            ["OffsetConverter（今昨仓拆分）", "positions.PositionBook.closable", "06"],
            ["CtaBacktester", "Phase 4 事件驱动回测", "P4"],
        ],
        columns=["vn.py（以官方源码为准）", "mini_trader", "课程"],
    )
    print(table.to_string(index=False))
    print("""
    ┌──────────┬──────────────────────────┬──────────────────────────┐
    │ 维度     │ 使用 vn.py               │ 自研（本课程项目）        │
    ├──────────┼──────────────────────────┼──────────────────────────┤
    │ 上手速度 │ 快，接口与 GUI 现成       │ 慢                       │
    │ 接口覆盖 │ 广（期货、股票、期权）     │ 只实现需要的             │
    │ 理解深度 │ 容易停留在"会用"          │ 理解每一层的取舍          │
    │ 定制性   │ 需要读懂框架内部          │ 完全掌控                 │
    └──────────┴──────────────────────────┴──────────────────────────┘

    📌 推荐做法：生产中可以复用 vn.py 的网关层（柜台接入最费时间），
       上层的风控、监控、对账、恢复按自己的需求建设 —— 本阶段的 mini_trader 就是这些上层组件的雏形。
    """)


# ==================== 主函数 ====================


def main():
    what_is_vnpy()
    cta_template()
    mini_cta_demo()
    pos_vs_account()
    mapping()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 MiniCtaTemplate 加上 on_init + load_bar：用前 N 根 K 线预热，预热期间不允许下单
    2. 第三部分对比实验中，上期所合约日内用 CLOSE 平仓被拒。参考 vn.py 的 OffsetConverter 思路，
       让 sell / cover 根据今昨仓自动拆分成 CLOSE_TODAY 和 CLOSE_YESTERDAY 两笔订单
    3. 把 MiniCtaEngine 改成异步版本：订阅第 02 课的 EventEngine，而不是直接监听 MockExchange
    4. 安装 vn.py 后（可选，在独立虚拟环境中），对照源码找出 CtaTemplate.buy 最终调用了哪个网关方法
    """)


if __name__ == "__main__":
    main()
