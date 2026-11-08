"""
ccxt 实时订单簿
===============

学习目标：
    1. 用 ccxt 读取加密交易所的公开订单簿，理解 REST 快照与 WebSocket 增量的取舍
    2. 从真实盘口计算价差、深度、订单簿不平衡（OBI）与微价格（Microprice）
    3. 用"吃单模拟"把盘口深度换算成不同规模订单的滑点
    4. 掌握本地订单簿维护的核心逻辑：覆盖语义、序列号校验、缺口重建

核心概念：
    - 价差与深度：决定小单的成本（价差）和大单的成本（深度）
    - OBI = (Q^b − Q^a) / (Q^b + Q^a)：买卖一档挂单量的不平衡
    - 微价格 = (P^a·Q^b + P^b·Q^a) / (Q^b + Q^a) = 中间价 + 半价差 × OBI
    - 增量消息里的数量是该价位的"新总量"（覆盖），不是变化量（累加）

前置知识：
    - 01-order-book-basics.py；Phase 1 的 09-futures-crypto-data.py（ccxt 基础）

环境要求：
    - pip install ccxt numpy pandas matplotlib
    - 访问海外交易所可能需要在 .env 中设置 HTTPS_PROXY
    - 无网络、访问失败或 QTC_OFFLINE=1 时，自动回退到合成订单簿

⚠️ 合规提示：中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动。
   本课只读取公开行情用于学习研究，不注册账户、不下单、不做任何交易。请遵守所在地法律法规。
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from orderbook import synthetic_depth_snapshots
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SYMBOL = "BTC/USDT"


# ==================== 第一部分：REST 与 WebSocket ====================


def rest_vs_websocket():
    print("=" * 60)
    print("第一部分：获取订单簿的两种方式")
    print("=" * 60)
    print("""
    ⚠️ 合规提示：中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动。
       本课只读取公开行情用于研究，不交易。选择加密市场做演示，只因其公开接口便于学习订单簿。
    """)
    table = pd.DataFrame(
        [
            ["REST 轮询 fetch_order_book", "每次请求拿完整快照", "简单、无状态", "有频率限制，两次之间信息丢失"],
            ["WebSocket 增量", "先拿快照，再接收价位变化", "低延迟、信息完整", "要自己维护本地簿，处理丢包重连"],
        ],
        columns=["方式", "原理", "优点", "缺点"],
    )
    print(table.to_string(index=False))
    print("""
    本课用 REST 连续采样几次（不依赖 websocket 库）。ccxt 的统一格式：
        ob = exchange.fetch_order_book("BTC/USDT", limit=100)
        ob["bids"] = [[价格, 数量, ...], ...]   价格从高到低
        ob["asks"] = [[价格, 数量, ...], ...]   价格从低到高
        ob["timestamp"] = 交易所时间戳（毫秒）
    """)


# ==================== 第二部分：连续采样 ====================


def fetch_snapshots(symbol=SYMBOL, n=8, interval=1.0, limit=400) -> tuple[list[dict], str]:
    """REST 连续采样 n 次；访问失败或离线模式时回退到合成数据。返回 (快照列表, 数据来源)"""
    if is_offline():
        print("⚠️ 离线模式（QTC_OFFLINE=1）：使用合成订单簿")
        return synthetic_depth_snapshots(n), "合成数据"
    snapshots = []
    try:
        import ccxt

        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        config = {"timeout": 15000}
        if proxy:
            config["proxies"] = {"http": proxy, "https": proxy}
        ex = ccxt.okx(config)

        def fetch_with_retry(retries=3):
            for k in range(retries):
                try:
                    return ex.fetch_order_book(symbol, limit=limit)
                except ccxt.NetworkError:  # 网络抖动：稍等重试；其他异常直接抛出
                    if k == retries - 1:
                        raise
                    time.sleep(1.0)

        for i in range(n):
            t0 = time.perf_counter()
            ob = fetch_with_retry()
            ob["rtt_ms"] = (time.perf_counter() - t0) * 1000
            ob["local_ms"] = ex.milliseconds()
            snapshots.append(ob)
            if i < n - 1:
                time.sleep(interval)
        print(f"✅ 从 OKX 获取 {symbol} 订单簿 {len(snapshots)} 次，每次 {len(snapshots[-1]['bids'])} 档")
        return snapshots, "OKX 实时数据"
    except Exception as e:
        print(f"⚠️ ccxt 获取失败（{type(e).__name__}: {str(e)[:100]}）")
    if len(snapshots) >= 2:  # 部分成功：有几张用几张
        print(f"   已取得 {len(snapshots)} 张快照，继续使用")
        return snapshots, "OKX 实时数据（采样提前结束）"
    print("   回退到合成订单簿（提示：可在 .env 中设置 HTTPS_PROXY）")
    return synthetic_depth_snapshots(n), "合成数据"


def validate_book(ob: dict) -> list[str]:
    """基本校验：真实系统收到行情后第一件事就是检查它是否自洽"""
    problems = []
    bids, asks = np.array([b[:2] for b in ob["bids"]], float), np.array([a[:2] for a in ob["asks"]], float)
    if len(bids) == 0 or len(asks) == 0:
        return ["某一侧为空"]
    if not np.all(np.diff(bids[:, 0]) < 0):
        problems.append("买盘价格不是严格降序")
    if not np.all(np.diff(asks[:, 0]) > 0):
        problems.append("卖盘价格不是严格升序")
    if bids[0, 0] >= asks[0, 0]:
        problems.append("盘口交叉：买一 ≥ 卖一")
    if (bids[:, 1] <= 0).any() or (asks[:, 1] <= 0).any():
        problems.append("存在非正数量")
    return problems


def sample_order_book():
    print("\n" + "=" * 60)
    print(f"第二部分：连续采样 {SYMBOL} 订单簿")
    print("=" * 60)

    snapshots, source = fetch_snapshots()
    print(f"📌 数据来源：{source}")
    latest = snapshots[-1]
    problems = [p for ob in snapshots for p in validate_book(ob)]
    print(f"校验 {len(snapshots)} 张快照：{'✅ 全部通过（排序、无交叉、数量为正）' if not problems else problems}")

    rows = []
    for i in range(5):
        bp, bq = latest["bids"][i][:2]
        ap, aq = latest["asks"][i][:2]
        rows.append([i + 1, bq, bp, ap, aq])
    print("\n最新一张快照的前五档：")
    print(pd.DataFrame(rows, columns=["档位", "买量", "买价", "卖价", "卖量"]).to_string(index=False))

    if "rtt_ms" in latest:
        rtts = [ob["rtt_ms"] for ob in snapshots]
        skews = [ob["local_ms"] - ob["timestamp"] for ob in snapshots if ob.get("timestamp")]
        print(f"\n⏱️ REST 往返耗时（含代理）：首次 {rtts[0]:.0f} ms（需要先加载市场列表），"
              f"之后中位数 {np.median(rtts[1:]):.0f} ms")
        if skews:
            print(f"   本地接收时间 − 交易所时间戳：中位数 {np.median(skews):.0f} ms（包含网络延迟和两边时钟偏差，不能当作单向延迟）")
    return snapshots, source


# ==================== 第三部分：盘口指标 ====================


def book_metrics(ob: dict, depth_levels=(5, 20)) -> dict:
    bids = np.array([b[:2] for b in ob["bids"]], float)
    asks = np.array([a[:2] for a in ob["asks"]], float)
    pb, qb, pa, qa = bids[0, 0], bids[0, 1], asks[0, 0], asks[0, 1]
    mid = (pa + pb) / 2
    spread = pa - pb
    obi = (qb - qa) / (qb + qa)
    micro = (pa * qb + pb * qa) / (qb + qa)
    m = {
        "best_bid": pb, "best_ask": pa, "mid": mid, "spread": spread, "spread_bp": spread / mid * 1e4,
        "qty_bid1": qb, "qty_ask1": qa, "obi_l1": obi, "microprice": micro,
        "micro_check": mid + spread / 2 * obi,  # 用于检验恒等式
    }
    for k in depth_levels:
        qb_k, qa_k = bids[:k, 1].sum(), asks[:k, 1].sum()
        m[f"depth_bid_{k}"] = qb_k
        m[f"depth_ask_{k}"] = qa_k
        m[f"obi_l{k}"] = (qb_k - qa_k) / (qb_k + qa_k)
    # 已取档位覆盖的价格范围；超出范围的深度统计只是下限
    m["bid_cover_bp"] = (mid - bids[-1, 0]) / mid * 1e4
    m["ask_cover_bp"] = (asks[-1, 0] - mid) / mid * 1e4
    # 距中间价 X bp 以内的深度（以计价货币 USDT 计）
    for bp_range in (5, 20):
        lo, hi = mid * (1 - bp_range / 1e4), mid * (1 + bp_range / 1e4)
        m[f"bid_usdt_{bp_range}bp"] = (bids[bids[:, 0] >= lo, 0] * bids[bids[:, 0] >= lo, 1]).sum()
        m[f"ask_usdt_{bp_range}bp"] = (asks[asks[:, 0] <= hi, 0] * asks[asks[:, 0] <= hi, 1]).sum()
    return m


def cover_note(m, bp_range) -> str:
    covered = m["bid_cover_bp"] >= bp_range and m["ask_cover_bp"] >= bp_range
    return "" if covered else "（超出已取档位范围，只是下限）"


def book_indicators(snapshots: list[dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：价差、深度、OBI 与微价格")
    print("=" * 60)
    print("""
    OBI = (Q^b − Q^a) / (Q^b + Q^a) ∈ [−1, 1]：天平，哪边重价格就往另一边倾斜
    微价格 = (P^a·Q^b + P^b·Q^a) / (Q^b + Q^a)：用"对面"的量加权
             买盘越厚（Q^b 大）→ 微价格越靠近卖价 → 预示价格更可能上行
    恒等式：微价格 = 中间价 + 半价差 × OBI
    """)
    df = pd.DataFrame([book_metrics(ob) for ob in snapshots])
    err = (df["microprice"] - df["micro_check"]).abs().max()
    print(f"✅ 微价格恒等式数值检验：最大误差 {err:.2e}")
    cols = ["best_bid", "best_ask", "spread", "spread_bp", "qty_bid1", "qty_ask1", "obi_l1", "microprice"]
    print(df[cols].round({"spread": 2, "spread_bp": 4, "qty_bid1": 4, "qty_ask1": 4, "obi_l1": 3, "microprice": 3}).to_string())
    last = df.iloc[-1]
    print(f"""
    最新盘口：相对价差 {last['spread_bp']:.4f} bp，一档 OBI {last['obi_l1']:+.3f}，
      微价格相对中间价偏移 {last['microprice'] - last['mid']:+.3f}（价差只有 {last['spread']:.2f} 时，偏移被限制在 ±{last['spread'] / 2:.2f} 以内）
    深度：前 5 档 买 {last['depth_bid_5']:.3f} / 卖 {last['depth_ask_5']:.3f} BTC，前 20 档 买 {last['depth_bid_20']:.3f} / 卖 {last['depth_ask_20']:.3f} BTC
    已取档位覆盖：买方到中间价下方 {last['bid_cover_bp']:.1f} bp，卖方到上方 {last['ask_cover_bp']:.1f} bp
    距中间价 ±5 bp 内：买 {last['bid_usdt_5bp']:,.0f} / 卖 {last['ask_usdt_5bp']:,.0f} USDT{cover_note(last, 5)}
    距中间价 ±20 bp 内：买 {last['bid_usdt_20bp']:,.0f} / 卖 {last['ask_usdt_20bp']:,.0f} USDT{cover_note(last, 20)}
    """)
    changed = sum(
        snapshots[i]["bids"][:5] != snapshots[i - 1]["bids"][:5] or snapshots[i]["asks"][:5] != snapshots[i - 1]["asks"][:5]
        for i in range(1, len(snapshots))
    )
    print(f"    相邻 {len(snapshots) - 1} 对快照中，前五档发生变化的有 {changed} 对 —— REST 只看到采样时刻，中间的事件全部丢失")
    return df


# ==================== 第四部分：吃单模拟 ====================


def walk_the_book(levels: list, notional: float) -> dict:
    """模拟市价单吃掉盘口：给定计价货币金额，逐档成交，返回成交均价与吃掉的档数"""
    remaining, base_qty, used = notional, 0.0, 0
    for row in levels:
        px, qty = float(row[0]), float(row[1])
        cost = px * qty
        take = min(remaining, cost)
        base_qty += take / px
        remaining -= take
        used += 1
        if remaining <= 1e-9:
            break
    filled = notional - remaining
    return {"filled": filled, "avg_px": filled / base_qty if base_qty else np.nan, "levels": used, "complete": remaining <= 1e-9}


def slippage_table(snapshot: dict):
    print("\n" + "=" * 60)
    print("第四部分：吃单模拟 —— 把深度换算成滑点")
    print("=" * 60)
    m = book_metrics(snapshot)
    rows = []
    for notional in [1e4, 1e5, 5e5, 1e6, 5e6]:
        buy = walk_the_book(snapshot["asks"], notional)
        sell = walk_the_book(snapshot["bids"], notional)
        buy_bp = (buy["avg_px"] - m["mid"]) / m["mid"] * 1e4
        sell_bp = (m["mid"] - sell["avg_px"]) / m["mid"] * 1e4
        rows.append([f"{notional:,.0f}",
                     buy["levels"], f"{buy_bp:.3f}" if buy["complete"] else "已取档位不够",
                     sell["levels"], f"{sell_bp:.3f}" if sell["complete"] else "已取档位不够"])
    print(f"    已取 {len(snapshot['bids'])} 档；相对中间价 {m['mid']:,.2f} 的滑点：")
    print(pd.DataFrame(rows, columns=["金额(USDT)", "买吃档数", "买滑点(bp)", "卖吃档数", "卖滑点(bp)"]).to_string(index=False))
    print("""
    💡 小单几乎只付半个价差，大单要吃多档，滑点随规模上升 —— 这就是 03 课的"临时冲击"。
       注意这是静态估算：真实吃单时，其他参与者会撤单或补单，实际成本可能更高或更低。
    """)


# ==================== 第五部分：本地订单簿维护 ====================


class LocalBook:
    """按序列号维护的本地订单簿（WebSocket 增量流的核心逻辑，具体字段以交易所 API 文档为准）"""

    def __init__(self, bids, asks, last_id):
        self.bids = {p: q for p, q in bids}
        self.asks = {p: q for p, q in asks}
        self.last_id = last_id

    def apply(self, first_id, final_id, bid_updates, ask_updates):
        if final_id <= self.last_id:
            return "过期，丢弃"
        if first_id > self.last_id + 1:
            raise RuntimeError(f"序列号缺口：期望 {self.last_id + 1}，收到 {first_id}，需要重新同步快照")
        for side, updates in ((self.bids, bid_updates), (self.asks, ask_updates)):
            for p, q in updates:
                if q == 0:
                    side.pop(p, None)  # 数量为 0 表示删除价位
                else:
                    side[p] = q  # 覆盖，不是累加
        self.last_id = final_id
        return "已应用"

    def bbo(self):
        return max(self.bids), min(self.asks)


def local_book_demo():
    print("\n" + "=" * 60)
    print("第五部分：本地订单簿维护（WebSocket 增量的核心逻辑）")
    print("=" * 60)
    print("""
    通用流程（具体字段以交易所 API 文档为准）：
      1. 订阅增量流并缓存  2. REST 拉快照，记下序列号  3. 丢弃序列号 ≤ 快照的增量
      4. 依次应用增量      5. 数量 0 → 删价位，否则覆盖  6. 出现序列号缺口 → 丢弃本地簿，回到第 1 步
    """)
    book = LocalBook([(100.0, 5.0), (99.5, 3.0)], [(100.5, 2.0), (101.0, 4.0)], last_id=10)
    print(f"    初始快照 id=10：买 {book.bids}，卖 {book.asks}")
    msgs = [
        (8, 9, [(100.0, 9.0)], []),  # 过期
        (11, 12, [(100.0, 6.0)], [(100.5, 0.0)]),  # 覆盖买一数量 + 删除卖一
        (13, 13, [], [(100.2, 1.0)]),  # 新价位
        (20, 21, [(100.0, 1.0)], []),  # 缺口
    ]
    for first, final, bu, au in msgs:
        try:
            result = book.apply(first, final, bu, au)
            print(f"    增量 [{first},{final}] 买更新 {bu} 卖更新 {au} → {result}；BBO = {book.bbo()}")
        except RuntimeError as e:
            print(f"    增量 [{first},{final}] → ❌ {e}")

    print("""
    ❌ 新手最常见 bug：把"新总量"累加到旧量上
    """)
    true_q, buggy_q, updates = 5.0, 5.0, [6.0, 4.0, 7.0]  # 交易所依次推送该价位的新总量
    for q in updates:
        true_q = q
        buggy_q += q
    print(f"    某价位初始 5.0，依次收到新总量 {updates}：正确值 {true_q}，累加 bug 得到 {buggy_q} → OBI、微价格全部跑偏")
    print("    📌 除了序列号校验，还要定期用 REST 快照校正本地簿，防止静默漂移。")


# ==================== 可视化 ====================


def plot_book(snapshot: dict, metrics: pd.DataFrame, source: str):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bids = np.array([b[:2] for b in snapshot["bids"]], float)
    asks = np.array([a[:2] for a in snapshot["asks"]], float)

    ax = axes[0]
    ax.step(bids[:, 0], np.cumsum(bids[:, 1]), where="post", color="tab:green", label="买盘累计")
    ax.fill_between(bids[:, 0], np.cumsum(bids[:, 1]), step="post", color="tab:green", alpha=0.25)
    ax.step(asks[:, 0], np.cumsum(asks[:, 1]), where="post", color="tab:red", label="卖盘累计")
    ax.fill_between(asks[:, 0], np.cumsum(asks[:, 1]), step="post", color="tab:red", alpha=0.25)
    ax.set_title(f"{SYMBOL} 深度图（{source}）")
    ax.set_xlabel("价格（USDT）")
    ax.set_ylabel("累计数量（BTC）")
    ax.ticklabel_format(useOffset=False, axis="x")
    ax.legend()

    ax = axes[1]
    x = np.arange(len(metrics))
    ax.plot(x, metrics["best_ask"], "r--", marker="v", label="卖一")
    ax.plot(x, metrics["best_bid"], "g--", marker="^", label="买一")
    ax.plot(x, metrics["mid"], color="gray", marker=".", label="中间价")
    ax.plot(x, metrics["microprice"], color="tab:blue", marker="o", label="微价格")
    ax.set_title("连续采样：微价格 vs 中间价")
    ax.set_xlabel("采样序号")
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.legend()

    path = save_figure(fig, "04_realtime_orderbook.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    rest_vs_websocket()
    snapshots, source = sample_order_book()
    metrics = book_indicators(snapshots)
    slippage_table(snapshots[-1])
    local_book_demo()
    plot_book(snapshots[-1], metrics, source)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把采样间隔改为 0.2 秒，统计买一 / 卖一的变化频率，注意遵守交易所的频率限制
    2. 用 ETH/USDT 重复第三、四部分，比较两者的相对价差和 ±20 bp 深度
    3. 连续采样 10 分钟保存为 Parquet，计算相邻样本的 OBI 与下一步中间价变化的相关系数（05 课会用模拟数据系统研究）
    4. 给 LocalBook 增加校验函数：每隔一段时间拉一次 REST 快照，比较前 N 档，差异过大时重建
    """)


if __name__ == "__main__":
    main()
