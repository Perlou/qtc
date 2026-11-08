"""
ccxt 加密货币实盘与测试网
========================

⚠️ 合规提示：
    根据中国人民银行等部门 2021 年发布的相关通知，中国内地监管部门已明确虚拟货币相关业务活动
    属于非法金融活动，境外虚拟货币交易所通过互联网向境内居民提供服务同样属于非法金融活动。
    本课不涉及任何虚拟货币交易，只借助 ccxt 讲解交易接口的工程问题（测试网、精度、幂等、密钥管理），
    全程不下单。读者须自行了解并遵守所在地的法律法规。

学习目标：
    1. 了解 ccxt 的统一接口，以及它在网关层扮演的角色
    2. 掌握测试网（sandbox）模式的切换方式和注意事项
    3. 用 amount_to_precision / price_to_precision 处理精度，校验最小下单量和最小成交额
    4. 理解 clientOrderId 幂等：超时后先查询，重发时复用同一个订单号
    5. 掌握 API 密钥的安全管理：只从环境变量读取、最小权限、禁止提现

核心概念：
    - set_sandbox_mode(True)：切换到交易所的测试环境，必须在任何请求之前调用
    - 精度（precision）与限制（limits）：不满足就会被交易所拒单
    - clientOrderId：客户端生成的订单号，许多交易所会拒绝重复值，是防重复下单的最后一道防线

前置知识：
    - 03 课（订单号映射）、10 课会深入讲解超时与恢复
    - 本阶段 CONCEPT.md 第 9 节

环境要求：
    - 已安装 ccxt；load_markets() / fetch_time() 需要网络，离线（QTC_OFFLINE=1）或请求失败时
      使用本地构造的示例市场（精度数值为示意）
    - 本课不读取、不打印任何真实密钥，不调用任何下单接口
"""

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ccxt
import pandas as pd
from dotenv import load_dotenv

from mini_trader import OMS, Direction, EventEngine, MockExchange, MockGateway, OrderIdGenerator, OrderRequest
from mini_trader.recovery import send_with_confirm
from utils.data import is_offline

load_dotenv()

SYMBOL = "BTC/USDT"


# ==================== 第一部分：合规提示与 ccxt 统一接口 ====================


def intro():
    print("=" * 60)
    print("第一部分：合规提示与 ccxt 统一接口")
    print("=" * 60)
    print("""
    ⚠️ 合规提示：中国内地监管部门已明确，虚拟货币相关业务活动属于非法金融活动，
       境外虚拟货币交易所通过互联网向境内居民提供服务同样属于非法金融活动。
       本课不涉及交易，只把 ccxt 当作"统一交易接口"的工程案例来学习，全程不下单。
    """)
    print(f"   ccxt 版本：{ccxt.__version__}，支持的交易所数量：{len(ccxt.exchanges)}")
    print(pd.DataFrame(
        [
            ["加载市场信息（精度、最小量）", "load_markets()"],
            ["下单 / 撤单", "create_order() / cancel_order()"],
            ["查询订单", "fetch_order() / fetch_open_orders()"],
            ["查询成交", "fetch_my_trades()"],
            ["查询余额", "fetch_balance()"],
            ["WebSocket 推送", "watch_orders() / watch_my_trades()（ccxt.pro）"],
        ],
        columns=["功能", "统一方法"],
    ).to_string(index=False))
    print("""
    💡 ccxt 本身就是一个"网关"：它把上百家交易所的接口翻译成统一的方法和数据结构，
       与第 01 课 BaseGateway 的思路完全一致。实际项目中，ccxt 之上还要再包一层自己的网关，
       把 ccxt 的订单结构转换成内部 OrderData（第 01 课的 from_ccxt）。
    """)


# ==================== 第二部分：测试网（sandbox） ====================


def sandbox():
    print("\n" + "=" * 60)
    print("第二部分：测试网（sandbox）")
    print("=" * 60)
    ex = ccxt.binance({"enableRateLimit": True})
    before = ex.urls["api"]["public"]
    ex.set_sandbox_mode(True)  # 必须在任何请求之前调用
    after = ex.urls["api"]["public"]
    print(f"   binance 公共接口地址：{before}")
    print(f"   切换 sandbox 之后：  {after}")

    rows = []
    for eid in ["binance", "okx", "bybit", "bitget", "gate"]:
        rows.append([eid, "test" in getattr(ccxt, eid)().urls])
    print("\n   ccxt 中配置了测试网地址的交易所（仅检查配置，不发请求）：")
    print(pd.DataFrame(rows, columns=["交易所", "urls 中有 test"]).to_string(index=False))
    print("""
    📌 注意事项：
    - 测试网使用虚拟资产，接口与正式环境基本一致，但流动性、撮合、可用产品都可能不同
    - 不同交易所的测试网覆盖范围不同（如只支持现货或只支持部分合约），以交易所文档为准
    - 测试网账号和密钥与正式环境是分开的；代码里要显式区分环境，避免"以为在测试网，其实在正式环境"
    """)
    return ex


# ==================== 第三部分：API 密钥管理 ====================


def mask(secret: str) -> str:
    return secret[:3] + "*" * 6 + secret[-2:] if len(secret) > 8 else "*" * len(secret)


def credentials():
    print("\n" + "=" * 60)
    print("第三部分：API 密钥管理")
    print("=" * 60)
    key, secret = os.getenv("TESTNET_API_KEY"), os.getenv("TESTNET_SECRET")
    if key and secret:
        print(f"   已从环境变量读取测试网密钥：TESTNET_API_KEY = {mask(key)}（只显示掩码）")
        print("   本课不会使用它调用任何私有接口")
    else:
        print("   未配置 TESTNET_API_KEY / TESTNET_SECRET：需要密钥的部分只展示代码，不执行")

    pattern = re.compile(r"""(apiKey|secret)["']\s*:\s*["'][A-Za-z0-9]{16,}["']""")
    here = Path(__file__).resolve().parent
    leaks = [p.name for p in here.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    print(f"   扫描本阶段代码中疑似硬编码的密钥：{leaks or '未发现'}")
    print("""
    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ 原则                 │ 做法                                         │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ 不进代码仓库         │ 从环境变量 / 密钥服务读取；.env 加入 .gitignore │
    │ 最小权限             │ 只开读取和交易权限，禁止提现权限                │
    │ 限制来源             │ 绑定服务器 IP 白名单                           │
    │ 不进日志             │ 日志、异常信息、告警里都不能出现密钥            │
    │ 可轮换               │ 定期更换；怀疑泄露时立即作废                    │
    └──────────────────────┴──────────────────────────────────────────────┘
    """)


# ==================== 第四部分：精度与最小量 ====================

OFFLINE_MARKET = {  # 离线示例市场：精度与限制为示意数值，以 load_markets() 的实际返回为准
    "id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT", "baseId": "BTC", "quoteId": "USDT",
    "type": "spot", "spot": True, "active": True,
    "precision": {"amount": 0.00001, "price": 0.01},
    "limits": {"amount": {"min": 0.00001, "max": 9000}, "price": {"min": 0.01, "max": 1e6}, "cost": {"min": 5}},
}


def load_market(ex) -> tuple[dict, float | None, str]:
    if not is_offline():
        try:
            ex.timeout = 10_000
            ex.options["fetchMarkets"] = {"types": ["spot"]}
            markets = ex.load_markets()
            last = ex.fetch_ticker(SYMBOL).get("last")
            return markets[SYMBOL], last, "测试网 load_markets()"
        except Exception as exc:  # 网络或交易所异常：回退到示例市场
            print(f"   ⚠️ 加载测试网市场失败（{type(exc).__name__}），使用离线示例市场")
    else:
        print("   ⚠️ 离线模式（QTC_OFFLINE=1）：使用离线示例市场，精度数值为示意")
    ex.set_markets([OFFLINE_MARKET])
    return ex.markets[SYMBOL], None, "离线示例市场（示意）"


def prepare_order(ex, symbol: str, amount: float, price: float) -> tuple[str, str]:
    """下单前的本地校验：精度取整 + 最小量 + 最小成交额。不满足就在本地拒绝，不要发给交易所"""
    m = ex.market(symbol)
    amt = ex.amount_to_precision(symbol, amount)  # 数量默认截断，避免超出可用余额
    px = ex.price_to_precision(symbol, price)
    min_cost = (m["limits"].get("cost") or {}).get("min")
    if min_cost and float(amt) * float(px) < min_cost:
        raise ccxt.InvalidOrder(f"成交额 {float(amt) * float(px):.4f} 低于最小成交额 {min_cost}")
    return amt, px


def precision(ex):
    print("\n" + "=" * 60)
    print("第四部分：精度与最小下单量")
    print("=" * 60)
    m, last, source = load_market(ex)
    print(f"   数据来源：{source}")
    print(f"   {SYMBOL} 精度：{m['precision']}")
    print(f"   {SYMBOL} 限制：amount={m['limits']['amount']}，cost={m['limits'].get('cost')}")
    ref = last or 65_000.0
    print(f"   参考价格：{ref}（{'测试网最新价' if last else '示意价格'}）")

    rows = []
    min_amt = m["limits"]["amount"]["min"]
    cases = [(0.0012345678, ref * 0.98 + 0.001234), (min_amt / 10, ref), (min_amt * 2, ref), (0.01, ref)]
    for amount, price in cases:
        try:
            amt, px = prepare_order(ex, SYMBOL, amount, price)
            rows.append([f"{amount:.10g}", f"{price:.6f}", amt, px, "✅ 可提交"])
        except ccxt.BaseError as exc:
            rows.append([f"{amount:.10g}", f"{price:.6f}", "-", "-", f"❌ {type(exc).__name__}: {str(exc)[:90]}"])
    print(pd.DataFrame(rows, columns=["原始数量", "原始价格", "取整后数量", "取整后价格", "结果"]).to_string(index=False))
    mode_name = {ccxt.TICK_SIZE: "TICK_SIZE", ccxt.DECIMAL_PLACES: "DECIMAL_PLACES"}.get(ex.precisionMode, "其他")
    print(f"""
    📌 要点：
    - 精度模式：binance 的 precisionMode={ex.precisionMode}（{mode_name}）；TICK_SIZE 模式下
      precision 里的数字是"最小变动单位"，DECIMAL_PLACES 模式下才是小数位数
    - amount_to_precision 默认截断（不会向上取整导致余额不足），price_to_precision 默认四舍五入
    - 最小成交额（cost.min）要自己检查：数量和价格都合法，乘积仍可能太小
    - 精度和限制会被交易所调整：每次启动都应重新 load_markets()，不要硬编码
    """)


# ==================== 第五部分：clientOrderId 幂等 ====================

TESTNET_ORDER_CODE = '''
# ⚠️ 示意代码，本课不执行。下单前务必确认处于测试网，且已了解相关法律法规。
ex = ccxt.binance({
    "apiKey": os.environ["TESTNET_API_KEY"],      # 密钥只从环境变量读取
    "secret": os.environ["TESTNET_SECRET"],
    "enableRateLimit": True,
})
ex.set_sandbox_mode(True)                         # 必须在任何请求之前调用
ex.load_markets()
coid = "S1-1794070000-0001"                       # 发单前生成并先写入事件日志
amount, price = prepare_order(ex, "BTC/USDT", 0.001, 50000)
try:
    order = ex.create_order("BTC/USDT", "limit", "buy", amount, price,
                            params={"clientOrderId": coid})   # 统一参数，ccxt 负责映射字段名
except ccxt.RequestTimeout:
    # 超时：不要直接重发！先按 clientOrderId 查询（具体查询方式以交易所和 ccxt 文档为准）
    ...
except ccxt.DuplicateOrderId:
    # 交易所已有这个订单号：说明之前的请求其实到达了，改为查询同步
    ...
'''


async def idempotency_demo():
    """在 MockExchange 上演示：超时 → 查询 → 不重发；重复订单号 → 交易所拒绝"""
    engine = EventEngine(timer_interval=None)
    exchange = MockExchange(initial_cash=100_000)
    gateway = MockGateway(engine, exchange)
    oms = OMS()
    engine.register("order", lambda e: oms.on_order(e.data))
    engine.register("trade", lambda e: oms.on_trade(e.data))
    engine.start()
    await gateway.connect()
    sym = f"{SYMBOL}.BINANCE"
    exchange.set_market(sym, 50_000.00)
    gen = OrderIdGenerator("S1")
    logs = []

    req = OrderRequest(sym, Direction.LONG, 50_000.01, 0.001, client_order_id=gen.next())
    oms.create_order(req)
    exchange.inject("lost_response")  # 交易所已处理，但响应丢失
    result = await send_with_confirm(gateway, oms, req, timeout=0.2, log=logs.append)
    await engine.join()

    try:  # 如果有人绕过 OMS 直接重发同一个订单号
        await gateway.send_order(req)
        dup = "交易所接受了重复订单（危险）"
    except Exception as exc:
        dup = f"{type(exc).__name__}: {exc}"
    await engine.stop()
    return result, logs, dup, len(exchange.query_orders()), req.client_order_id


def idempotency():
    print("\n" + "=" * 60)
    print("第五部分：clientOrderId 幂等")
    print("=" * 60)
    print("📄 测试网下单的示意代码（不执行）：")
    print(TESTNET_ORDER_CODE)
    result, logs, dup, n_orders, coid = asyncio.run(idempotency_demo())
    print(f"   订单号：{coid}（长度 {len(coid)}；各交易所对长度和字符集有不同限制，以文档为准）")
    print("\n🔬 MockExchange 演示：注入'响应丢失'故障")
    for line in logs:
        print(f"     {line}")
    print(f"   结果：{result.outcome}，订单状态 {result.order.status.value}，交易所订单数 {n_orders}")
    print(f"   绕过 OMS 再次发送同一订单号 → {dup}")
    print(f"   ccxt 中对应的异常类：{' → '.join(c.__name__ for c in ccxt.DuplicateOrderId.__mro__[:4])}")
    print("""
    💡 幂等三要点：
       1. 订单号在发单前生成，并先持久化（Write-Ahead，第 10 课）
       2. 超时后先查询，再决定是否重发；重发必须复用同一个订单号
       3. 交易所拒绝重复订单号是最后一道防线，但不是所有接口都支持 —— 不能依赖它
    """)


# ==================== 第六部分：其他工程要点 ====================


def engineering(ex):
    print("\n" + "=" * 60)
    print("第六部分：其他工程要点")
    print("=" * 60)
    print(f"   enableRateLimit={ex.enableRateLimit}，ccxt 内置的请求间隔 rateLimit={ex.rateLimit} ms")
    if not is_offline():
        try:
            offset = ex.fetch_time() - ex.milliseconds()
            print(f"   测试网服务器时间 - 本机时间 = {offset} ms（含网络往返，签名请求对时钟偏差敏感）")
        except Exception as exc:
            print(f"   获取服务器时间失败（{type(exc).__name__}），跳过时钟偏差检查")
    else:
        print("   离线模式：跳过服务器时间对比")
    print(pd.DataFrame(
        [
            ["频率限制", "各交易所按权重计算请求配额，超限会被临时封禁；开启 enableRateLimit"],
            ["7×24 运行", "没有收盘，监控和运维要按'永不停机'设计"],
            ["交易所维护", "有计划内维护与突发宕机，关注交易所公告，系统要能处理连不上"],
            ["WebSocket 断线", "连接会被服务端定期断开，需自动重连并重新同步订单状态"],
            ["时间同步", "签名请求依赖时间戳，本机时钟偏差过大会被拒；服务器开启 NTP"],
            ["精度与最小量", "每次启动重新 load_markets()，下单前本地校验"],
        ],
        columns=["要点", "说明"],
    ).to_string(index=False))


# ==================== 主函数 ====================


def main():
    intro()
    ex = sandbox()
    credentials()
    precision(ex)
    idempotency()
    engineering(ex)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 写一个 CcxtGateway(BaseGateway) 骨架：把 ccxt 的订单 / 成交结构转换为 OrderData / TradeData，
       用第四部分的离线示例市场和 unittest.mock 完成单元测试（不发真实请求）
    2. 扩展 prepare_order：检查 amount 的最大值，并在余额不足时本地拒绝
    3. 给订单号生成器加上"按交易所规则清洗"的功能：限制长度和字符集，并保证清洗后仍全局唯一
    4. 阅读 ccxt 文档中关于 sandbox 的说明，列出你关注的交易所测试网支持哪些产品类型
    """)


if __name__ == "__main__":
    main()
