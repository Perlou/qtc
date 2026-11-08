"""
Tick 与 Level-2 数据处理
========================

学习目标：
    1. 分清日线、L1 快照、L2 快照、逐笔成交、逐笔委托几种数据的粒度与用途
    2. 理解 A 股"3 秒快照"与"逐笔事件"的区别：快照之间发生的事情是看不见的
    3. 实现 Tick Rule、Quote Rule 与 Lee-Ready 算法，判断每笔成交是主动买还是主动卖
    4. 从快照计算价差、盘口不平衡等常用指标，并避开累计量、集合竞价等数据陷阱

核心概念：
    - 快照（Snapshot）：每隔约 3 秒对订单簿"拍一张照"，含十档买卖价量
    - 逐笔成交（Tick-by-Tick Trade）：每一笔成交的时间、价格、数量
    - 主动方向：成交是由买方吃卖单（主动买）还是卖方砸买单（主动卖）触发
    - Lee-Ready：先比较成交价与中间价（Quote Rule），恰好等于中间价时再看价格变化（Tick Rule）

前置知识：
    - 01-order-book-basics.py（订单簿与撮合）

环境要求：
    - pip install numpy pandas matplotlib akshare
    - A 股 Level-2 数据需要付费，本课用 orderbook.py 的零智能模拟器生成"逐笔 + 3 秒十档快照"
    - 联网时额外探测 AKShare 的免费分时成交接口；QTC_OFFLINE=1 时跳过
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from orderbook import simulate_zi_market
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


# ==================== 第一部分：数据层级 ====================


def data_hierarchy():
    print("=" * 60)
    print("第一部分：从日线到逐笔 —— 数据层级")
    print("=" * 60)

    table = pd.DataFrame(
        [
            ["日线 / 分钟线", "OHLCV 聚合", "日 / 分钟", "中低频策略"],
            ["Level-1 快照", "最新价、成交量、五档买卖价量", "约 3 秒", "价差、简单盘口因子"],
            ["Level-2 快照", "十档买卖价量、委托笔数等", "约 3 秒", "盘口深度因子"],
            ["逐笔成交", "每笔成交的价格、数量、买卖方订单号", "事件驱动", "主动买卖、成交分布"],
            ["逐笔委托", "每笔新申报 / 撤单", "事件驱动", "订单流重建、OFI"],
        ],
        columns=["数据类型", "内容", "频率（A 股典型）", "用途"],
    )
    print(table.to_string(index=False))
    print("""
    时间轴上的事件流 vs 快照：

    逐笔事件： ─●─●●──●─●●●──●─●───●●─●──●●●─●──●─●●─▶
    快照：     ──────────📷──────────📷──────────📷──────▶
                         3s          3s          3s

    关键认知：A 股常说的"Tick 数据"多数是 3 秒快照。3 秒内可能有几十笔成交和撤单，
    快照只记录了"拍照那一刻"的盘口。真正的事件级数据（Level-2 逐笔）需要向券商或数据商付费获取。
    """)


# ==================== 第二部分：模拟逐笔与快照 ====================


def simulated_level2():
    print("\n" + "=" * 60)
    print("第二部分：逐笔成交与 3 秒十档快照（模拟数据）")
    print("=" * 60)

    sim = simulate_zi_market(seconds=3600, seed=2026)
    trades, snaps = sim["trades"], sim["snapshots"]
    print(f"""
    用 orderbook.simulate_zi_market 模拟 1 小时连续竞价：
      随机到达的限价单 / 市价单 / 撤单驱动 01 课的撮合引擎，
      同时记录"逐笔成交"和"每 3 秒一张的十档快照"（结构与 A 股 Level-2 相同）
    逐笔成交 {len(trades):,} 笔（来自 {trades['taker_oid'].nunique():,} 笔主动订单），快照 {len(snaps):,} 张
    """)

    show = trades.head(8).copy()
    show["time"] = show["time"].round(3)
    print("    逐笔成交（前 8 笔；aggressor_true 是模拟器记录的真实主动方向，真实数据里往往没有）：")
    print("    " + show.to_string(index=False).replace("\n", "\n    "))

    snap = snaps.iloc[600]
    rows = [[f"买{i}", snap[f"bid{i}"], int(snap[f"bid{i}_v"]), f"卖{i}", snap[f"ask{i}"], int(snap[f"ask{i}_v"])]
            for i in range(1, 11)]
    print(f"\n    第 {snap['time']:.0f} 秒的十档快照（最新价 {snap['last']:.2f}，累计成交 {int(snap['cum_volume']):,} 股）：")
    book = pd.DataFrame(rows, columns=["买档", "买价", "买量", "卖档", "卖价", "卖量"])
    print("    " + book.to_string(index=False).replace("\n", "\n    "))

    # 一个快照区间里发生了多少事
    bins = np.floor(trades["time"] / 3).astype(int)
    per_snap = trades.groupby(bins)["taker_oid"].nunique().reindex(range(len(snaps)), fill_value=0)
    print(f"""
    每 3 秒区间内的主动订单数：均值 {per_snap.mean():.2f}，最多 {per_snap.max()}，
    有 {(per_snap >= 2).mean():.1%} 的区间发生了 2 笔以上 → 这些事件的先后顺序在快照里看不到
    """)

    # 快照指标
    df = snaps.copy()
    df["mid"] = (df["bid1"] + df["ask1"]) / 2
    df["spread_bp"] = (df["ask1"] - df["bid1"]) / df["mid"] * 1e4
    df["imb_l1"] = (df["bid1_v"] - df["ask1_v"]) / (df["bid1_v"] + df["ask1_v"])
    bid5 = df[[f"bid{i}_v" for i in range(1, 6)]].sum(axis=1)
    ask5 = df[[f"ask{i}_v" for i in range(1, 6)]].sum(axis=1)
    df["imb_l5"] = (bid5 - ask5) / (bid5 + ask5)
    df["volume"] = df["cum_volume"].diff().fillna(df["cum_volume"].iloc[0])  # 累计量 → 区间量
    print("    从快照计算的常用指标：")
    print("    " + df[["time", "mid", "spread_bp", "imb_l1", "imb_l5", "cum_volume", "volume"]]
          .iloc[600:606].round(4).to_string(index=False).replace("\n", "\n    "))
    print(f"""
    价差统计：均值 {df['spread_bp'].mean():.2f} bp，{(df['spread_bp'] < 10.5).mean():.1%} 的快照价差为 1 个 tick
    ⚠️ cum_volume 是当日累计量，区间成交量要 diff()；第一条快照要单独处理
    """)
    return trades, df


# ==================== 第三部分：主动买卖判别 ====================


def tick_rule(price: np.ndarray) -> np.ndarray:
    """价格上升 → +1（主动买），下降 → −1，不变 → 沿用上一笔；第一笔无法判断记为 0"""
    side = np.zeros(len(price), dtype=int)
    for i in range(1, len(price)):
        d = price[i] - price[i - 1]
        side[i] = 1 if d > 1e-9 else (-1 if d < -1e-9 else side[i - 1])
    return side


def lee_ready(price: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Quote Rule 优先：成交价 > 中间价 → 主动买，< → 主动卖；恰好等于中间价时用 Tick Rule"""
    side = np.sign(np.round(price - mid, 9)).astype(int)
    tr = tick_rule(price)
    at_mid = side == 0
    side[at_mid] = tr[at_mid]
    return side


def classify_trades(trades: pd.DataFrame, snaps: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：主动买卖判别 —— Tick Rule / Quote Rule / Lee-Ready")
    print("=" * 60)

    px = np.array([10.01, 10.01, 10.00, 10.005, 10.01])
    mid = np.full(5, 10.005)
    print(f"""
    CONCEPT 示例：成交价 {px.tolist()}，中间价恒为 10.005
      Lee-Ready → {lee_ready(px, mid).tolist()}（第 4 笔恰在中间价，比上一笔高 → 主动买）
    """)

    t = trades.copy()
    price = t["price"].to_numpy()
    truth = t["aggressor_true"].to_numpy()

    # 真实场景：只有 3 秒快照 → 用"成交时刻之前最近一张快照"的中间价
    snap_mid = snaps[["time", "mid"]].rename(columns={"mid": "snap_mid"})
    t = pd.merge_asof(t.sort_values("time"), snap_mid, on="time", direction="backward")
    stale_mid = t["snap_mid"].to_numpy()

    methods = {
        "Tick Rule（只用成交价）": tick_rule(price),
        "Quote Rule（成交瞬间的真实盘口）": np.sign(np.round(price - t["mid_before"].to_numpy(), 9)).astype(int),
        "Quote Rule（最近一张 3 秒快照）": np.sign(np.round(price - stale_mid, 9)).astype(int),
        "Lee-Ready（最近一张 3 秒快照）": lee_ready(price, stale_mid),
    }
    rows = []
    for name, pred in methods.items():
        rows.append([name, f"{(pred == truth).mean():.1%}", f"{(pred == 0).mean():.1%}"])
    print(pd.DataFrame(rows, columns=["方法", "与真实方向一致", "无法判断(0)"]).to_string(index=False))
    print("""
    💡 解读：
       - 有"成交瞬间"的盘口时，Quote Rule 几乎完美（市价单总是在买一/卖一成交）
       - 只有 3 秒快照时，盘口已经过时：成交价可能恰好等于旧中间价，或落在旧盘口的"错误一侧"
       - Lee-Ready 用 Tick Rule 补上"恰在中间价"的部分，比单独使用任何一个规则更稳健
       - 真实数据里没有标准答案：交易所逐笔数据带买卖标志时优先用标志，没有时才用算法推断
    """)
    t["side_lr"] = methods["Lee-Ready（最近一张 3 秒快照）"]
    buy_ratio = t.loc[t["side_lr"] > 0, "qty"].sum() / t["qty"].sum()
    print(f"    按 Lee-Ready 推断的主动买入占比：{buy_ratio:.1%}"
          f"（真实值 {t.loc[truth > 0, 'qty'].sum() / t['qty'].sum():.1%}）")
    return t


# ==================== 第四部分：真实数据探测（AKShare） ====================


def fetch_intraday(symbol: str = "000001") -> pd.DataFrame | None:
    """尝试获取最近一个交易日的分时成交明细，统一为 time / price / lots / label 四列"""
    import akshare as ak

    errors = []
    try:
        raw = ak.stock_intraday_em(symbol=symbol)
        df = raw.rename(columns={"时间": "time", "成交价": "price", "手数": "lots", "买卖盘性质": "label"})
        print(f"    ✅ stock_intraday_em({symbol}) 获取 {len(df):,} 条")
        return df[["time", "price", "lots", "label"]]
    except Exception as e:
        errors.append(f"stock_intraday_em: {type(e).__name__}: {str(e)[:80]}")
    try:
        prefix = "sh" if symbol.startswith("6") else "sz"
        raw = ak.stock_zh_a_tick_tx_js(symbol=prefix + symbol)
        df = raw.rename(columns={"成交时间": "time", "成交价格": "price", "成交量": "lots", "性质": "label"})
        print(f"    ✅ stock_zh_a_tick_tx_js({prefix}{symbol}) 获取 {len(df):,} 条")
        return df[["time", "price", "lots", "label"]]
    except Exception as e:
        errors.append(f"stock_zh_a_tick_tx_js: {type(e).__name__}: {str(e)[:80]}")
    for err in errors:
        print(f"    ⚠️ {err}")
    return None


def real_data_probe(symbol: str = "000001"):
    print("\n" + "=" * 60)
    print(f"第四部分：真实数据探测 —— AKShare 免费分时成交（{symbol}）")
    print("=" * 60)

    if is_offline():
        print("    ⏭️ 离线模式（QTC_OFFLINE=1）：跳过真实数据探测")
        return None
    try:
        df = fetch_intraday(symbol)
    except Exception as e:  # akshare 未安装或其他异常
        print(f"    ⚠️ 获取失败（{type(e).__name__}），跳过本部分")
        return None
    if df is None or df.empty:
        print("    ⚠️ 没有拿到数据，跳过本部分")
        return None

    df = df.copy()
    df["time"] = df["time"].astype(str)
    hhmmss = df["time"]
    phase = np.select(
        [hhmmss < "09:25:00", hhmmss == "09:25:00", hhmmss < "15:00:00", hhmmss == "15:00:00"],
        ["开盘集合竞价(虚拟撮合)", "开盘集合竞价成交", "连续竞价", "收盘集合竞价成交"],
        default="15:00 之后",
    )
    df["phase"] = phase
    summary = df.groupby("phase", sort=False).agg(条数=("lots", "size"), 手数合计=("lots", "sum"))
    print("    按交易阶段拆分：")
    print("    " + summary.to_string().replace("\n", "\n    "))

    # 对账：真实成交量 = 09:25 ~ 15:00 的记录；09:25 之前的虚拟匹配量不计入
    real_lots = df.loc[(hhmmss >= "09:25:00") & (hhmmss <= "15:00:00"), "lots"].sum()
    print(f"\n    09:25~15:00 手数合计 {real_lots:,} 手 = {real_lots * 100:,} 股")
    try:
        import akshare as ak

        end = pd.Timestamp.today()
        prefix = "sh" if symbol.startswith("6") else "sz"
        daily = ak.stock_zh_a_daily(
            symbol=prefix + symbol,
            start_date=(end - pd.Timedelta(days=10)).strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        row = daily.iloc[-1]
        print(f"    新浪日线 {row['date']} 成交量 {row['volume']:,.0f} 股（盘中运行时日线可能尚未更新，两者应在收盘后接近）")
    except Exception as e:
        print(f"    ⚠️ 日线对账跳过（{type(e).__name__}）")
    print("    📌 09:25 之前的记录不是真实成交（集合竞价虚拟匹配量），直接加总会高估全天成交量")

    cont = df[df["phase"] == "连续竞价"].copy()
    ts = pd.to_datetime(cont["time"], format="%H:%M:%S")
    gaps = ts.diff().dt.total_seconds().dropna()
    print(f"""
    连续竞价阶段相邻记录的时间间隔：中位数 {gaps.median():.0f} 秒，{(gaps == 3).mean():.1%} 恰好是 3 秒
    → 免费接口的"分时成交"是按快照间隔聚合的数据，不等同于交易所 Level-2 逐笔成交
    买卖盘性质分布：{cont['label'].value_counts().to_dict()}
    """)

    tr = tick_rule(cont["price"].to_numpy())
    label = cont["label"].map({"买盘": 1, "卖盘": -1}).fillna(0).to_numpy()
    known = (label != 0) & (tr != 0)
    print(f"    Tick Rule 与数据源自带的买卖盘标签一致率：{(tr[known] == label[known]).mean():.1%}"
          f"（共 {known.sum():,} 条可比较）")
    print("    （数据源标签的生成方式未公开，这里只看两者的一致程度，而不是谁对谁错）")

    buy = cont.loc[cont["label"] == "买盘", "lots"].sum()
    print(f"    主动买入手数占比（按数据源标签）：{buy / cont['lots'].sum():.1%}")

    minutes = ts.dt.hour * 60 + ts.dt.minute
    bucket = np.where(minutes < 12 * 60, np.minimum((minutes - 570) // 30, 3), 4 + (minutes - 780) // 30)  # 11:30:00 归入上午最后一段
    vol = cont.groupby(bucket)["lots"].sum()
    vol = vol[(vol.index >= 0) & (vol.index <= 7)]
    labels = ["09:30", "10:00", "10:30", "11:00", "13:00", "13:30", "14:00", "14:30"]
    vol.index = [labels[int(i)] for i in vol.index]
    share = vol / vol.sum()
    print("\n    连续竞价成交量按 30 分钟分布（日内成交量分布，06 课 VWAP 的基础）：")
    for k, v in share.items():
        print(f"      {k} {'█' * int(v * 100):<30} {v:.1%}")
    print("    ⚠️ 只看了一只股票一天的数据，形状会因股票和日期而变化；VWAP 需要用多日平均")
    return share


# ==================== 第五部分：数据陷阱与可视化 ====================


def pitfalls_and_plot(trades: pd.DataFrame, snaps: pd.DataFrame, real_share):
    print("\n" + "=" * 60)
    print("第五部分：Tick 数据处理的陷阱")
    print("=" * 60)
    pitfalls = pd.DataFrame(
        [
            ["累计量 vs 增量", "快照成交量常为当日累计值", "diff() 得到区间量，注意第一条"],
            ["集合竞价数据混入", "9:25 与 15:00 是一次性集中成交", "单独处理开盘 / 收盘竞价"],
            ["时间戳含义", "交易所时间、行情服务器时间、本地接收时间不同", "回测只能用当时已收到的数据"],
            ["沪深格式差异", "逐笔字段、撤单表示方式不同", "先统一数据模型"],
            ["涨跌停时的盘口", "涨停时卖盘为空，价差无定义", "过滤或特殊标记"],
            ["数据量巨大", "全市场逐笔一天可达数十 GB", "Parquet 分区 + 列式查询"],
        ],
        columns=["陷阱", "说明", "处理"],
    )
    print(pitfalls.to_string(index=False))

    setup_chinese_font()
    n_plots = 3 if real_share is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))
    window = snaps[(snaps["time"] >= 1800) & (snaps["time"] <= 1920)]
    tw = trades[(trades["time"] >= 1800) & (trades["time"] <= 1920)]
    ax = axes[0]
    ax.step(window["time"], window["bid1"], where="post", color="tab:green", label="买一（快照）")
    ax.step(window["time"], window["ask1"], where="post", color="tab:red", label="卖一（快照）")
    buys, sells = tw[tw["aggressor_true"] > 0], tw[tw["aggressor_true"] < 0]
    ax.scatter(buys["time"], buys["price"], marker="^", color="red", s=20, label="主动买成交")
    ax.scatter(sells["time"], sells["price"], marker="v", color="green", s=20, label="主动卖成交")
    ax.set_title("2 分钟内的快照盘口与逐笔成交（模拟）")
    ax.set_xlabel("秒")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(snaps["imb_l1"].dropna(), bins=40, alpha=0.6, label="一档不平衡")
    ax.hist(snaps["imb_l5"].dropna(), bins=40, alpha=0.6, label="五档不平衡")
    ax.set_title("盘口不平衡分布（模拟快照）")
    ax.legend()

    if real_share is not None:
        ax = axes[2]
        ax.bar(real_share.index, real_share.values, color="tab:blue")
        ax.set_title("真实数据：日内成交量 30 分钟分布")
        ax.tick_params(axis="x", rotation=45)

    path = save_figure(fig, "02_tick_level2.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    data_hierarchy()
    trades, snaps = simulated_level2()
    classify_trades(trades, snaps)
    real_share = real_data_probe("000001")
    pitfalls_and_plot(trades, snaps, real_share)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第三部分的快照间隔从 3 秒改为 1 秒、10 秒（simulate_zi_market 的 snapshot_every），
       观察 Lee-Ready 准确率如何变化
    2. 只用快照数据（不看逐笔），用相邻快照的累计成交量与盘口变化估计区间主动买卖量
    3. 用第四部分的真实数据，把同一只股票的 1 分钟成交量与 1 分钟价格变动画散点图
    4. 设计一个"统一逐笔数据模型"（dataclass），同时容纳沪深两种格式和加密交易所的成交推送
    """)


if __name__ == "__main__":
    main()
