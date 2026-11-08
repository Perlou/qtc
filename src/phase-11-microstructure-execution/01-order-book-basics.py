"""
订单簿、撮合规则与订单类型
==========================

学习目标：
    1. 理解订单簿（Order Book）的结构：买盘、卖盘、BBO、中间价、价差、深度
    2. 亲手实现一个"价格优先、时间优先"的最小撮合引擎：限价单、市价单、部分成交、撤单
    3. 掌握常见订单类型（GTC / IOC / FOK / 市价最优五档）的行为差异
    4. 了解 A 股集合竞价与连续竞价、申报类型、涨跌停与"价格笼子"等约束

核心概念：
    - 价格优先、时间优先（Price-Time Priority）：价格更优先成交，同价位先到先得（FIFO）
    - 成交价取被动方（挂单方）价格：主动方报价更激进时，"价格改善"归主动方
    - Maker / Taker：挂单提供流动性 / 吃单消耗流动性
    - 集合竞价：所有申报一次性撮合，取"成交量最大"的价格

前置知识：
    - Phase 1 的 A 股交易规则（02-a-share-market-rules.py）
    - Phase 4 的交易成本与滑点

环境要求：
    - pip install numpy pandas matplotlib
    - 本课不需要网络；撮合引擎在同目录 orderbook.py 中，后续课程会复用

⚠️ A 股交易规则会调整，本课涉及的申报类型、价格笼子、涨跌幅等以交易所最新业务规则为准。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from orderbook import BUY, SELL, MatchingEngine, trades_to_frame
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def show_trades(eng: MatchingEngine, trades, title: str = "成交回报"):
    if not trades:
        print(f"    {title}：无成交")
        return
    df = trades_to_frame(eng, trades)[["maker_oid", "taker_oid", "price", "qty", "aggressor"]]
    df.columns = ["被动方订单", "主动方订单", "成交价", "数量", "主动方向"]
    print(f"    {title}：")
    print("    " + df.to_string(index=False).replace("\n", "\n    "))


# ==================== 第一部分：订单簿是什么 ====================


def build_sample_book() -> MatchingEngine:
    """搭建 CONCEPT.md 2.1 节的五档示例订单簿"""
    eng = MatchingEngine(tick_size=0.01)
    for px, qty in [(10.00, 500), (9.99, 1200), (9.98, 2000), (9.97, 900), (9.96, 3000)]:
        eng.submit_limit(BUY, px, qty)
    for px, qty in [(10.01, 300), (10.02, 800), (10.03, 1500), (10.04, 600), (10.05, 2200)]:
        eng.submit_limit(SELL, px, qty)
    return eng


def order_book_anatomy():
    print("=" * 60)
    print("第一部分：订单簿是什么")
    print("=" * 60)

    print("""
    订单簿 = 一个"挂单墙"：按价格分桶、桶内按到达顺序排队（FIFO）

    工程视角：
        bids: dict[价格 → deque[订单]]   买盘，价格越高越优先
        asks: dict[价格 → deque[订单]]   卖盘，价格越低越优先
        价格用"整数 tick"存储：10.01 元 → 1001，避免浮点数当字典 key 的精度问题
    """)

    eng = build_sample_book()
    print(eng.render(5))
    bb, ba = eng.to_price(eng.best_bid()), eng.to_price(eng.best_ask())
    mid = eng.mid()
    spread = ba - bb
    print(f"""
    best bid = {bb:.2f}    best ask = {ba:.2f}    （合称 BBO）
    中间价 mid = (bid + ask) / 2 = {mid:.3f}
    报价价差 spread = ask − bid = {spread:.2f}，相对价差 = {spread / mid * 1e4:.2f} bp
    买盘五档深度 = {sum(q for _, q, _ in eng.levels(BUY, 5)):,} 股，卖盘五档深度 = {sum(q for _, q, _ in eng.levels(SELL, 5)):,} 股
    """)
    print("    深度表（depth）：")
    print("    " + eng.depth(5).to_string(index=False).replace("\n", "\n    "))
    print("""
    💡 tick size 的含义：价差至少是 1 个 tick。股价 10 元时 1 tick = 0.01 元 ≈ 10 bp；
       股价 2 元时 1 tick ≈ 50 bp —— 低价股的最小价差在相对意义上非常宽。
    """)
    return eng


# ==================== 第二部分：撮合规则 ====================


def matching_rules():
    print("\n" + "=" * 60)
    print("第二部分：撮合规则 —— 价格优先、时间优先")
    print("=" * 60)

    eng = MatchingEngine()
    a, _ = eng.submit_limit(SELL, 10.01, 100)  # A 先到
    b, _ = eng.submit_limit(SELL, 10.01, 200)  # B 后到
    c, _ = eng.submit_limit(SELL, 10.02, 500)
    eng.submit_limit(BUY, 9.99, 400)
    print(f"    初始状态：卖一 10.01 上依次是 A(#{a.oid}, 100) → B(#{b.oid}, 200)，卖二 10.02 上 C(#{c.oid}, 500)")
    print(eng.render(3))

    print("\n    ▶ 来了一笔 买 250 股 @10.02 的限价单：")
    order, trades = eng.submit_limit(BUY, 10.02, 250)
    show_trades(eng, trades)
    print(f"    订单状态：{order.status}，已成交 {order.filled} 股")
    print("""
    逐步解读：
      1. 对手方最优价 10.01 ≤ 10.02 → 可成交；价格优先：先吃 10.01，而不是 10.02
      2. 时间优先：10.01 上 A 先到，先与 A 成交 100 股
      3. 再与 B 成交 150 股，B 剩 50 股继续排在 10.01
      4. 成交价是被动方价格 10.01 —— 买方愿意出 10.02，却按 10.01 成交（价格改善）
    """)
    print(eng.render(3))

    print("\n    ▶ 再来一笔 买 600 股 @10.01（部分成交，剩余挂单）：")
    order, trades = eng.submit_limit(BUY, 10.01, 600)
    show_trades(eng, trades)
    print(f"    订单状态：{order.status}，已成交 {order.filled} 股，剩余 {order.remaining} 股挂在买一 10.01")
    print(eng.render(3))

    print("\n    ▶ 撤单：撤掉刚才剩余的买单")
    ok = eng.cancel(order.oid)
    print(f"    cancel(#{order.oid}) → {ok}；再撤一次 → {eng.cancel(order.oid)}（已不在簿上）")
    print(eng.render(3))
    print("""
    📌 被动方 vs 主动方：
       Maker（被动方 / 挂单）：挂在簿上等待成交 → 赚半个价差，但可能不成交、可能被逆向选择
       Taker（主动方 / 吃单）：以可立即成交的价格下单 → 付半个价差，大单还要"吃多档"
       本部分中两笔买单是 Taker，A、B 两笔卖单是 Maker
    """)


# ==================== 第三部分：订单类型 ====================


def order_types():
    print("\n" + "=" * 60)
    print("第三部分：订单类型 —— 市价单、IOC、FOK、最优五档")
    print("=" * 60)

    eng = build_sample_book()
    mid0 = eng.mid()
    print("    使用第一部分的五档订单簿（卖一 10.01×300、卖二 10.02×800、卖三 10.03×1500 ...）")

    print("\n    ▶ 市价买 2000 股（逐档吃掉对手盘）：")
    order, trades = eng.submit_market(BUY, 2000)
    show_trades(eng, trades)
    df = trades_to_frame(eng, trades)
    avg_px = (df.price * df.qty).sum() / df.qty.sum()
    print(f"    成交均价 {avg_px:.4f}，相对下单前中间价 {mid0:.3f} 的滑点 = {(avg_px - mid0) / mid0 * 1e4:.1f} bp")
    print(f"    下单后卖一变为 {eng.to_price(eng.best_ask()):.2f}：大单'吃穿'多档 → 这就是市场冲击的最直观来源")

    eng = build_sample_book()
    print("\n    ▶ IOC 买 1500 股 @10.02（能成交多少成交多少，剩余撤销）：")
    order, trades = eng.submit_limit(BUY, 10.02, 1500, tif="IOC")
    show_trades(eng, trades)
    print(f"    状态 {order.status}：成交 {order.filled}，撤销 {order.remaining}（剩余部分不进入订单簿）")

    eng = build_sample_book()
    print("\n    ▶ FOK 买 1500 股 @10.02（10.02 以内只有 300+800=1100 股）：")
    order, trades = eng.submit_limit(BUY, 10.02, 1500, tif="FOK")
    print(f"    状态 {order.status}：成交 {order.filled} 股 —— 不能全部成交就一股都不成交")

    eng = build_sample_book()
    print("\n    ▶ 市价买 8000 股，限定最多吃 5 档（近似'最优五档即时成交剩余撤销'）：")
    order, trades = eng.submit_market(BUY, 8000, max_levels=5)
    print(f"    五档卖盘合计 {300 + 800 + 1500 + 600 + 2200} 股 → 成交 {order.filled}，撤销 {order.remaining}，状态 {order.status}")

    types = pd.DataFrame(
        [
            ["限价单 GTC", "指定价格，未成交部分挂单", "最常用，控制价格"],
            ["市价单", "按对手方最优价逐档成交", "追求确定成交"],
            ["IOC", "立即成交，剩余撤销", "扫单而不留挂单"],
            ["FOK", "全部成交，否则全部撤销", "需要整笔成交的套利腿"],
            ["Post-Only", "会立即成交则拒绝，保证只做 Maker", "做市（见练习 1）"],
            ["冰山单 Iceberg", "只显示一部分数量", "隐藏大单意图（期货/加密常见）"],
        ],
        columns=["订单类型", "行为", "典型用途"],
    )
    print("\n" + types.to_string(index=False))
    print("""
    📌 A 股市价申报类型（连续竞价阶段，细节以沪深交易所最新业务规则为准）：
       上交所：最优五档即时成交剩余撤销、最优五档即时成交剩余转限价；
               科创板等板块还支持本方最优、对手方最优等
       深交所：对手方最优价格、本方最优价格、即时成交剩余撤销、
               最优五档即时成交剩余撤销、全额成交或撤销
       A 股交易所层面一般不支持止损单，常见的"条件单"由券商或客户端实现。
    """)


# ==================== 第四部分：集合竞价 ====================


def call_auction_price(bids, asks, prev_close):
    """
    集合竞价定价（简化版）：
        1. 成交量最大；2. 未成交量（|累计买 − 累计卖|）最小；3. 仍并列时取离前收盘最近的价格（深交所做法）
    bids / asks: [(价格, 数量), ...]，返回 (成交价, 成交量, 明细表)
    """
    prices = sorted({p for p, _ in bids} | {p for p, _ in asks})
    rows, best = [], None
    for p in prices:
        buy = sum(q for bp, q in bids if bp >= p)  # 愿意以 ≥ p 买入的总量
        sell = sum(q for ap, q in asks if ap <= p)  # 愿意以 ≤ p 卖出的总量
        vol = min(buy, sell)
        rows.append([p, buy, sell, vol, abs(buy - sell)])
        key = (vol, -abs(buy - sell), -abs(p - prev_close))
        if best is None or key > best[0]:
            best = (key, p, vol)
    table = pd.DataFrame(rows, columns=["价格", "累计买量(≥p)", "累计卖量(≤p)", "可成交量", "未成交量"])
    return best[1], best[2], table


def call_auction():
    print("\n" + "=" * 60)
    print("第四部分：A 股交易时段与集合竞价")
    print("=" * 60)

    print("""
                       A 股（沪深主板）一个交易日（以交易所规则为准）
    9:15 ─ 9:20 ─ 9:25 ─ 9:30 ──────── 11:30   13:00 ──────── 14:57 ─ 15:00
    │开盘集合竞价│     │    │  连续竞价    │ 午休 │  连续竞价    │收盘集合│
    │ 可撤单 │不可撤│等待│ 价格优先时间优先│      │             │竞价不可撤│
                  ▲ 9:25 一次性撮合产生开盘价                   ▲ 15:00 产生收盘价

    集合竞价 = 一场拍卖：把所有买卖意愿画成两条累计曲线，取成交量最大的价格
    """)

    bids = [(10.05, 300), (10.02, 500), (10.00, 800)]
    asks = [(9.98, 400), (10.01, 600), (10.04, 700)]
    px, vol, table = call_auction_price(bids, asks, prev_close=10.00)
    print("    示例申报：买 " + str(bids) + "\n              卖 " + str(asks))
    print("    " + table.to_string(index=False).replace("\n", "\n    "))
    tied = table[table["可成交量"] == table["可成交量"].max()]
    print(f"\n    成交量最大的价格：{tied['价格'].tolist()} → 按规则选出成交价 {px:.2f}，成交 {vol} 股")
    print("""
    📌 并列时的处理（以交易所规则为准）：上交所取使未成交量最小的价格，仍并列取中间价；
       深交所取距前收盘价最近的价格。本函数采用后一种。
    📌 9:15–9:20 可以撤单，这段时间的挂单可能是"虚假意图"；9:20 之后的虚拟成交价更有参考价值。
    """)

    rng = np.random.default_rng(11)
    rand_bids = [(round(10 + rng.integers(-8, 6) * 0.01, 2), int(rng.integers(1, 30)) * 100) for _ in range(40)]
    rand_asks = [(round(10 + rng.integers(-5, 9) * 0.01, 2), int(rng.integers(1, 30)) * 100) for _ in range(40)]
    px2, vol2, table2 = call_auction_price(rand_bids, rand_asks, prev_close=10.00)
    print(f"    随机 40 买 40 卖的集合竞价：成交价 {px2:.2f}，成交量 {vol2:,} 股（曲线见图）")
    return table2, px2


# ==================== 第五部分：A 股申报约束 ====================


def validate_a_share_order(side, price, qty, prev_close, ref_price, limit_pct=0.10, tick=0.01):
    """
    A 股主板限价申报的简化校验（教学用，以交易所最新规则为准）：
        - 价格必须在涨跌停范围内，且是 tick 的整数倍
        - 买入数量为 100 股整数倍
        - 价格笼子（连续竞价）：买价 ≤ max(基准价 × 102%, 基准价 + 10 个 tick)，卖价对称
    返回 (是否有效, 原因)
    """
    up = round(prev_close * (1 + limit_pct) + 1e-9, 2)
    down = round(prev_close * (1 - limit_pct) + 1e-9, 2)
    if abs(price / tick - round(price / tick)) > 1e-6:
        return False, "价格不是 0.01 元的整数倍"
    if price > up or price < down:
        return False, f"超出涨跌停 [{down:.2f}, {up:.2f}]，无效申报"
    if side == BUY and qty % 100 != 0:
        return False, "买入数量须为 100 股整数倍"
    if side == BUY and price > max(ref_price * 1.02, ref_price + 10 * tick) + 1e-9:
        return False, "超出价格笼子上限：不能立即参与撮合"
    if side == SELL and price < min(ref_price * 0.98, ref_price - 10 * tick) - 1e-9:
        return False, "低于价格笼子下限：不能立即参与撮合"
    return True, "有效"


def a_share_constraints():
    print("\n" + "=" * 60)
    print("第五部分：A 股申报约束（涨跌停、价格笼子、整数手）")
    print("=" * 60)

    prev_close, ref_buy, ref_sell = 10.00, 10.20, 10.19  # 买入基准价（如卖一）与卖出基准价（如买一），示例值
    cases = [
        (BUY, 10.30, 1000), (BUY, 10.45, 500), (BUY, 11.20, 100),
        (BUY, 10.25, 150), (SELL, 9.90, 300), (SELL, 10.123, 100),
    ]
    rows = []
    for side, px, qty in cases:
        ref = ref_buy if side == BUY else ref_sell
        ok, reason = validate_a_share_order(side, px, qty, prev_close, ref)
        rows.append(["买" if side == BUY else "卖", px, qty, "✅" if ok else "❌", reason])
    print(f"    前收盘 {prev_close:.2f}（涨跌停 ±10%），买入基准价 {ref_buy:.2f}，卖出基准价 {ref_sell:.2f}")
    print(f"    价格笼子：买价上限 = max({ref_buy}×102%, {ref_buy}+10 tick) = {max(ref_buy * 1.02, ref_buy + 0.1):.3f}；"
          f"卖价下限 = min({ref_sell}×98%, {ref_sell}−10 tick) = {min(ref_sell * 0.98, ref_sell - 0.1):.3f}")
    print(pd.DataFrame(rows, columns=["方向", "价格", "数量", "结果", "原因"]).to_string(index=False))
    print("""
    📌 对量化执行的影响：
       - 子单量要取整到 100 股（06 课会用到）
       - 激进限价单可能落在价格笼子外 → "以为下了单，其实没进撮合"
       - T+1：当日买入次日才能卖，A 股股票上无法做日内往返的高频策略
    """)

    # 撮合引擎吞吐量：顺便感受 Python 的速度（10 课会系统测量延迟）
    rng = np.random.default_rng(0)
    eng = MatchingEngine()
    n = 20_000
    sides = rng.choice([BUY, SELL], n)
    prices = 10 + rng.integers(-10, 11, n) * 0.01
    qtys = rng.integers(1, 20, n) * 100
    t0 = time.perf_counter()
    for s, p, q in zip(sides, prices, qtys):
        eng.submit_limit(str(s), float(p), int(q))
    elapsed = time.perf_counter() - t0
    print(f"    ⏱️ 本 Python 撮合引擎处理 {n:,} 笔随机限价单用时 {elapsed:.2f} 秒，"
          f"约 {n / elapsed:,.0f} 笔/秒，产生 {len(eng.trades):,} 笔成交")
    print("       交易所撮合系统用 C/C++ 与专用硬件实现，吞吐量与延迟不在一个数量级。")


# ==================== 可视化 ====================


def plot_book_and_auction(eng: MatchingEngine, table: pd.DataFrame, auction_px: float):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    bids = eng.levels(BUY, 5)
    asks = eng.levels(SELL, 5)
    bp, bq = [p for p, _, _ in bids], np.cumsum([q for _, q, _ in bids])
    ap, aq = [p for p, _, _ in asks], np.cumsum([q for _, q, _ in asks])
    ax.step(bp, bq, where="post", color="tab:green", label="买盘累计深度")
    ax.fill_between(bp, bq, step="post", color="tab:green", alpha=0.25)
    ax.step(ap, aq, where="post", color="tab:red", label="卖盘累计深度")
    ax.fill_between(ap, aq, step="post", color="tab:red", alpha=0.25)
    ax.axvline(eng.mid(), color="gray", ls="--", label=f"中间价 {eng.mid():.3f}")
    ax.set_title("深度图（Depth Chart）")
    ax.set_xlabel("价格")
    ax.set_ylabel("累计数量（股）")
    ax.legend()

    ax = axes[1]
    ax.plot(table["价格"], table["累计买量(≥p)"], "g-o", ms=3, label="累计买量（价格 ≥ p）")
    ax.plot(table["价格"], table["累计卖量(≤p)"], "r-o", ms=3, label="累计卖量（价格 ≤ p）")
    ax.plot(table["价格"], table["可成交量"], "k--", label="可成交量 min(买, 卖)")
    ax.axvline(auction_px, color="tab:blue", ls=":", label=f"集合竞价成交价 {auction_px:.2f}")
    ax.set_title("集合竞价：成交量最大原则")
    ax.set_xlabel("价格 p")
    ax.legend(fontsize=8)

    path = save_figure(fig, "01_orderbook_auction.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    eng = order_book_anatomy()
    matching_rules()
    order_types()
    table, px = call_auction()
    a_share_constraints()
    plot_book_and_auction(eng, table, px)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 MatchingEngine 增加 Post-Only 订单：若会立即成交则拒绝，并写测试验证
    2. 实现深交所的"对手方最优价格"和"本方最优价格"市价申报，与限价单比较成交结果
    3. 把 min(opp) / max(opp) 换成有序结构（bisect 维护的价格列表或堆），
       用第五部分的 20,000 笔订单对比前后吞吐量
    4. 按上交所规则修改 call_auction_price 的并列处理（未成交量最小 → 取中间价），
       构造一个两种规则给出不同开盘价的例子
    """)


if __name__ == "__main__":
    main()
