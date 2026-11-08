"""
CTP 期货接口与 SimNow 仿真
==========================

学习目标：
    1. 了解 CTP 的 MdApi / TraderApi 与"请求 → 异步回调"模式
    2. 掌握登录流程：连接前置 → 穿透式监管认证 → 登录 → 结算单确认
    3. 在 MockExchange 上演示期货的开平仓、平今 / 平昨、锁仓，以及平仓指令拆分
    4. 理解夜盘的交易日归属，以及它对日内统计和"今仓"判断的影响
    5. 知道查询流控、回报重推等 CTP 接入陷阱，以及 SimNow 仿真的用途和局限

核心概念：
    - CTP：上期技术开发的期货柜台系统，程序化交易通过其 C++ API 接入
    - 穿透式监管：程序化交易软件需通过期货公司认证（AppID / AuthCode），并上报终端信息
    - 开平：期货下单要同时指定方向和开平；上期所 / 能源中心还要区分平今、平昨
    - 交易日：夜盘属于下一个交易日

前置知识：
    - 02 课（跨线程投递）、03 课（订单状态机）
    - 本阶段 CONCEPT.md 第 7 节

环境要求：
    - 不安装 vn.py / CTP，不连接 SimNow；CTP 的回调时序用模拟对象演示
    - 文中的接口名、参数、前置地址均为示意，以 CTP 官方文档、期货公司和 SimNow 官网为准
"""

import ast
import asyncio
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

from mini_trader import Direction, EventEngine, MockExchange, Offset, OrderRequest, get_spec
from mini_trader.sessions import TradingCalendar

load_dotenv()


# ==================== 第一部分：CTP 简介 ====================


def ctp_intro():
    print("=" * 60)
    print("第一部分：CTP 简介")
    print("=" * 60)
    print("""
    CTP（Comprehensive Transaction Platform，综合交易平台）是国内期货公司广泛使用的柜台系统。
    程序通过它提供的 C++ API 接入，Python 通常借助 vn.py 等项目的封装：

    ┌────────────────┬──────────────────────────────────────────────────┐
    │ API            │ 功能                                             │
    ├────────────────┼──────────────────────────────────────────────────┤
    │ MdApi（行情）   │ 订阅合约行情，推送 Tick 快照                      │
    │ TraderApi（交易）│ 认证、登录、报单、撤单、查询，接收订单 / 成交回报    │
    └────────────────┴──────────────────────────────────────────────────┘

    调用模式：ReqXxx 发出请求 → 结果通过回调返回
      OnRspXxx：对某个请求的响应（如 OnRspUserLogin）
      OnRtnXxx：柜台主动推送（如 OnRtnOrder 订单回报、OnRtnTrade 成交回报）
    ⚠️ 回调发生在 API 内部线程 → 必须用第 02 课的 put_threadsafe 投递到事件循环，回调里不做耗时操作。
    """)


# ==================== 第二部分：登录流程 ====================


class MockCtpTraderApi:
    """模拟 TraderApi 的异步回调时序：请求立即返回，响应在 API 线程里稍后回调（仅为示意）"""

    def __init__(self, spi, auth_ok: bool = True, latency: float = 0.02):
        self.spi, self.auth_ok, self.latency = spi, auth_ok, latency

    def _later(self, fn, *args):
        t = threading.Timer(self.latency, fn, args)
        t.name = "ctp-trader-thread"
        t.start()

    def Init(self):
        self._later(self.spi.OnFrontConnected)

    def ReqAuthenticate(self, req: dict):
        self._later(self.spi.OnRspAuthenticate, {"ok": self.auth_ok, "msg": "" if self.auth_ok else "认证失败（示意）"})

    def ReqUserLogin(self, req: dict):
        self._later(self.spi.OnRspUserLogin, {"ok": True, "FrontID": 1, "SessionID": 1234, "TradingDay": "20261109"})

    def ReqSettlementInfoConfirm(self, req: dict):
        self._later(self.spi.OnRspSettlementInfoConfirm, {"ok": True})


class CtpSpiBridge:
    """回调对象：把 API 线程里的回调转成事件，交给 asyncio 事件循环处理"""

    def __init__(self, engine: EventEngine):
        self.engine = engine

    def _emit(self, name, data=None):
        self.engine.put_threadsafe("ctp", (name, data or {}, threading.current_thread().name))

    def OnFrontConnected(self):
        self._emit("OnFrontConnected")

    def OnRspAuthenticate(self, rsp):
        self._emit("OnRspAuthenticate", rsp)

    def OnRspUserLogin(self, rsp):
        self._emit("OnRspUserLogin", rsp)

    def OnRspSettlementInfoConfirm(self, rsp):
        self._emit("OnRspSettlementInfoConfirm", rsp)


async def ctp_login(auth_ok: bool) -> list:
    engine = EventEngine(timer_interval=None)
    inbox: asyncio.Queue = asyncio.Queue()
    engine.register("ctp", lambda e: inbox.put_nowait(e.data))
    engine.start()
    api = MockCtpTraderApi(CtpSpiBridge(engine), auth_ok)
    timeline = []

    async def expect(name):
        got, data, thread = await asyncio.wait_for(inbox.get(), timeout=1.0)
        assert got == name, f"期望 {name}，收到 {got}"
        timeline.append([f"← {got}", thread, threading.current_thread().name, str(data) if data else ""])
        return data

    steps = [
        ("Init（连接交易前置）", api.Init, None, "OnFrontConnected"),
        ("ReqAuthenticate（AppID + AuthCode）", api.ReqAuthenticate, {"AppID": "<AppID>", "AuthCode": "<AuthCode>"}, "OnRspAuthenticate"),
        ("ReqUserLogin（BrokerID + 账号 + 密码）", api.ReqUserLogin, {"BrokerID": "<BrokerID>"}, "OnRspUserLogin"),
        ("ReqSettlementInfoConfirm（确认结算单）", api.ReqSettlementInfoConfirm, {}, "OnRspSettlementInfoConfirm"),
    ]
    for label, fn, req, rsp_name in steps:
        timeline.append([f"→ {label}", "", threading.current_thread().name, ""])
        fn() if req is None else fn(req)
        data = await expect(rsp_name)
        if data.get("ok") is False:
            timeline.append(["✖ 登录中止", "", "", "认证失败时不能继续登录，需检查 AppID/AuthCode 并告警"])
            break
    else:
        timeline.append(["✔ 就绪", "", "", "可以查询合约 / 资金 / 持仓并开始交易"])
    await engine.stop()
    return timeline


VNPY_CTP_CONNECT = '''
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_ctp import CtpGateway

# 字段名以所安装 vnpy_ctp 版本的 default_setting 为准；地址等从期货公司或 SimNow 官网获取
setting = {
    "用户名": "<投资者代码>",
    "密码": "<密码>",               # 实际使用时从环境变量读取，不要写进代码
    "经纪商代码": "<BrokerID>",
    "交易服务器": "tcp://<交易前置地址>",
    "行情服务器": "tcp://<行情前置地址>",
    "产品名称": "<AppID>",
    "授权编码": "<AuthCode>",
}
main_engine = MainEngine(EventEngine())
main_engine.add_gateway(CtpGateway)
main_engine.connect(setting, "CTP")
'''


def login_flow():
    print("\n" + "=" * 60)
    print("第二部分：登录流程（模拟回调时序）")
    print("=" * 60)
    for ok in (True, False):
        print(f"\n🔐 {'正常登录' if ok else '认证失败'}：")
        rows = asyncio.run(ctp_login(ok))
        print(pd.DataFrame(rows, columns=["步骤", "回调线程", "处理线程", "数据"]).to_string(index=False))
    print("""
    📌 关键参数：
       BrokerID（期货公司代码）、UserID / Password、AppID / AuthCode（穿透式监管认证）、交易 / 行情前置地址
    📌 穿透式监管：自 2019 年起期货市场全面实施，程序化交易软件需通过期货公司认证，
       并采集终端信息上报，使监管能识别到具体的交易终端。AppID / AuthCode 需向期货公司申请。
    📌 登录成功返回 FrontID、SessionID 和交易日：FrontID + SessionID + OrderRef 唯一定位一笔订单。
    """)
    print("📄 用 vn.py 连接 CTP 的示意代码（不执行）：")
    print(VNPY_CTP_CONNECT)
    ast.parse(VNPY_CTP_CONNECT)
    print("   ✅ 语法检查通过（ast.parse）")


# ==================== 第三部分：开平仓与今昨仓 ====================


def split_close(exchange: MockExchange, symbol: str, pos_dir: Direction, volume: float) -> list[tuple[Offset, float]]:
    """平仓指令拆分（思路类似 vn.py 的 OffsetConverter）：上期所 / 能源中心先平昨、再平今"""
    if not get_spec(symbol).explicit_close:
        return [(Offset.CLOSE, volume)]
    book = exchange.positions
    yd = min(volume, book.closable(symbol, pos_dir, Offset.CLOSE_YESTERDAY))
    td = min(volume - yd, book.closable(symbol, pos_dir, Offset.CLOSE_TODAY))
    return [(o, v) for o, v in [(Offset.CLOSE_YESTERDAY, yd), (Offset.CLOSE_TODAY, td)] if v > 0]


def open_close_demo():
    print("\n" + "=" * 60)
    print("第三部分：开平仓与今昨仓（MockExchange 演示）")
    print("=" * 60)
    print("""
    ┌──────┬──────┬──────┬────────────────┐
    │ 操作 │ 方向 │ 开平 │ vn.py 方法      │
    ├──────┼──────┼──────┼────────────────┤
    │ 开多 │ 买   │ 开仓 │ buy            │
    │ 平多 │ 卖   │ 平仓 │ sell           │
    │ 开空 │ 卖   │ 开仓 │ short          │
    │ 平空 │ 买   │ 平仓 │ cover          │
    └──────┴──────┴──────┴────────────────┘
    """)
    sym = "rb2501.SHFE"
    ex = MockExchange()
    rows, seq = [], [0]

    def order(direction, offset, vol, label):
        seq[0] += 1
        price = 3510 if direction is Direction.LONG else 3490
        o = ex.submit(OrderRequest(sym, direction, price, vol, offset, f"F-{seq[0]}"))
        lp, sp = ex.positions.get(sym, Direction.LONG), ex.positions.get(sym, Direction.SHORT)
        rows.append([label, o.status.value, o.reason or "-",
                     f"{lp.volume:g}（昨{lp.yd_volume:g}/今{lp.td_volume:g}）", f"{sp.volume:g}"])

    ex.set_market(sym, 3500)
    order(Direction.LONG, Offset.OPEN, 3, "第 1 天：开多 3 手")
    ex.settle()
    rows.append(["—— 日终结算：今仓转昨仓 ——", "", "", "", ""])
    ex.set_market(sym, 3500)
    order(Direction.LONG, Offset.OPEN, 2, "第 2 天：开多 2 手")
    order(Direction.SHORT, Offset.CLOSE, 4, "平多 4 手（CLOSE = 平昨）")
    order(Direction.SHORT, Offset.CLOSE_TODAY, 3, "平今 3 手")
    order(Direction.SHORT, Offset.OPEN, 1, "开空 1 手（锁仓）")
    print(f"{sym}（上期所，乘数 {get_spec(sym).size:g}）：")
    print(pd.DataFrame(rows, columns=["操作", "结果", "拒单原因", "多头", "空头"]).to_string(index=False))

    plan = split_close(ex, sym, Direction.LONG, 4)
    print(f"\n   想平多 4 手 → 拆分为：{[(o.value, v) for o, v in plan]}")
    for i, (offset, vol) in enumerate(plan):
        o = ex.submit(OrderRequest(sym, Direction.SHORT, 3490, vol, offset, f"SPLIT-{i}"))
        print(f"     {offset.value} {vol:g} 手 → {o.status.value}")
    lp = ex.positions.get(sym, Direction.LONG)
    print(f"   拆分后多头剩余 {lp.volume:g} 手；空头 {ex.positions.get(sym, Direction.SHORT).volume:g} 手（锁仓仍在）")
    print(f"   非上期所合约 m2501.DCE 的拆分结果：{[(o.value, v) for o, v in split_close(ex, 'm2501.DCE', Direction.LONG, 4)]}")
    print("""
    📌 要点：
    - 期货账户可以同时持有同一合约的多头和空头（锁仓），持仓按方向分别记录
    - 上期所 / 能源中心区分平今和平昨，指令选错会被拒；其他交易所一般由系统按规则处理
      （MockExchange 的规则是简化版，实际以交易所和期货公司为准）
    - 不少品种的平今手续费与平昨不同，日内策略的成本要按"平今"计算
    """)


# ==================== 第四部分：夜盘与交易日归属 ====================


def night_session():
    print("\n" + "=" * 60)
    print("第四部分：夜盘与交易日归属")
    print("=" * 60)
    days = pd.bdate_range("2026-11-02", "2026-11-30").date  # 用工作日近似交易日（示意）
    cal = TradingCalendar(days)
    samples = [
        datetime(2026, 11, 12, 21, 30),  # 周四夜盘
        datetime(2026, 11, 13, 10, 0),  # 周五日盘
        datetime(2026, 11, 13, 21, 30),  # 周五夜盘
        datetime(2026, 11, 14, 0, 30),  # 周六凌晨（部分品种夜盘延续到凌晨）
        datetime(2026, 11, 15, 10, 0),  # 周日
        datetime(2026, 11, 16, 9, 30),  # 周一日盘
    ]
    weekday = "一二三四五六日"
    rows = [[f"{dt:%m-%d %H:%M} 周{weekday[dt.weekday()]}", dt.date(), cal.trading_day_of(dt) or "非交易时间"] for dt in samples]
    print(pd.DataFrame(rows, columns=["时间", "自然日", "所属交易日"]).to_string(index=False))

    trades = pd.DataFrame(
        {
            "dt": samples[:4] + [samples[5]],
            "pnl": [1200, -300, -800, 500, 400],
        }
    )
    trades["自然日"] = trades["dt"].dt.date
    trades["交易日"] = [cal.trading_day_of(d.to_pydatetime()) for d in trades["dt"]]
    print("\n   同一批成交盈亏，按自然日 vs 按交易日汇总：")
    for col in ("自然日", "交易日"):
        summary = {str(k): int(v) for k, v in trades.groupby(col)["pnl"].sum().items()}
        print(f"   按{col}：{summary}")
    print("""
    💡 影响：
    - 日内亏损风控、日报、结算都应按交易日统计，否则周五夜盘的亏损会算进"周五"而不是"下周一"
    - "今仓"按交易日界定：周五夜盘开的仓，到下周一日盘仍是今仓，平仓要用平今
    - 节假日前通常没有夜盘，具体以交易所通知为准；本例用工作日近似交易日，没有考虑节假日
    """)


# ==================== 第五部分：查询流控与其他陷阱 ====================


class QueryThrottle:
    """查询排队：保证两次查询间隔不小于 interval（CTP 查询流控常见为约每秒 1 次，以期货公司实际限制为准）"""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def run(self, name: str, t0: float, log: list):
        async with self._lock:
            wait = self._last + self.interval - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.perf_counter()
            log.append([name, f"{(self._last - t0) * 1000:.0f} ms"])


async def throttle_demo():
    throttle, log, t0 = QueryThrottle(interval=0.2), [], time.perf_counter()  # 演示用 0.2 秒代替 1 秒
    await asyncio.gather(*(throttle.run(n, t0, log) for n in ["查询合约", "查询资金", "查询持仓", "查询委托"]))
    return log


def pitfalls():
    print("\n" + "=" * 60)
    print("第五部分：查询流控与其他接入陷阱")
    print("=" * 60)
    log = asyncio.run(throttle_demo())
    print("   4 个查询同时发起，经过流控排队后的实际发出时刻（演示间隔 0.2 秒）：")
    print(pd.DataFrame(log, columns=["查询", "发出时刻"]).to_string(index=False))
    print(pd.DataFrame(
        [
            ["查询流控", "连续查询会返回流控错误，需要排队；启动时的批量查询尤其要注意"],
            ["报单流控", "期货公司对报单 / 撤单频率也有限制"],
            ["回调线程", "回调在 API 线程中，不能做耗时操作"],
            ["交易日切换", "夜盘开始即进入新交易日，日志与统计按交易日归档"],
            ["回报重推", "重新登录后可能收到当日订单 / 成交回报重推，必须按成交编号去重（第 03 课）"],
            ["平今平昨", "上期所 / 能源中心品种平仓指令选错会被拒"],
            ["平台兼容", "CTP API 主要面向 Windows 与 Linux，macOS 开发常借助 Docker 或虚拟机（以官方发布为准）"],
        ],
        columns=["陷阱", "说明"],
    ).to_string(index=False))
    print("""
    🧪 SimNow 仿真（以 SimNow 官网最新说明为准）：
    - 上期技术提供的 CTP 仿真环境，可免费注册仿真账号，是学习期货程序化交易的标准起点
    - 一般提供与实盘时段一致的仿真环境，以及便于开发调试的环境（具体环境与地址以官网为准）
    - 仿真撮合与真实市场有差异：仿真盘的成交结果不能代表实盘滑点
    - 仿真环境会不定期维护或重置：系统要能正确处理"连不上"，这本身就是很好的故障演练
    """)


# ==================== 主函数 ====================


def main():
    ctp_intro()
    login_flow()
    open_close_demo()
    night_session()
    pitfalls()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给登录流程加上"断线重连"：模拟 OnFrontDisconnected 后自动重新走完整的认证与登录流程，
       并在重连成功后调用第 10 课的 recover_from_exchange 对账
    2. 修改 split_close，增加参数 prefer="today"/"yesterday"，按平今、平昨手续费高低选择拆分顺序
    3. 用真实交易日历（Phase 1 第 07 课的 akshare 交易日历）替换工作日近似，找出一个节假日前后
       的例子，验证 trading_day_of 的结果
    4. 注册 SimNow 仿真账号（可选），在独立环境中安装 vn.py，用第二部分的示意代码完成一次登录
    """)


if __name__ == "__main__":
    main()
