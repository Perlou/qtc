"""
通道突破：Donchian 与 Dual Thrust
=================================

学习目标：
    1. 实现 Donchian 通道突破（N 日新高做多、新低做空，反向突破短周期极值离场）
    2. 亲手复现"通道包含当天"这一最常见的未来函数，并理解为什么必须 shift(1)
    3. 在股指和期货（螺纹钢主力连续）上回测，计入交易成本与换月成本
    4. 理解 Dual Thrust 的区间定义，认识用日线回测日内策略的失真

核心概念：
    - Donchian 通道：Upper_t = max(H_{t-N..t-1})，Lower_t = min(L_{t-N..t-1})（不含当天）
    - 路径依赖：持仓状态取决于上一时刻状态（入场 → 持有 → 离场），难以完全向量化
    - Dual Thrust：Range = max(HH - LC, HC - LL)，BuyLine = Open + K1·Range，SellLine = Open - K2·Range

前置知识：
    - 01-moving-average-crossover.py
    - cta_utils.py 中的 get_continuous（06 课会详细讲连续合约的构造）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存时读缓存，否则使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import SPECS, backtest, cost_rate, get_continuous, perf_stats, stats_table, trade_stats
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

IDX_COST = 0.0005  # 股指单边成本示例


def donchian_target(df: pd.DataFrame, n_entry: int = 20, n_exit: int = 10, include_today: bool = False) -> pd.Series:
    """
    Donchian 突破状态机（第 t 天收盘后决定目标仓位）：
        空仓：收盘价 > 过去 n_entry 日最高价 → 做多；< 过去 n_entry 日最低价 → 做空
        持多：收盘价 < 过去 n_exit 日最低价 → 平仓；持空同理
    include_today=True 是错误示范：通道窗口包含当天
    """
    sh = 0 if include_today else 1
    up = df["high"].rolling(n_entry).max().shift(sh)
    dn = df["low"].rolling(n_entry).min().shift(sh)
    ex_long = df["low"].rolling(n_exit).min().shift(sh)
    ex_short = df["high"].rolling(n_exit).max().shift(sh)

    pos, cur = [], 0
    for c, u, d, el, es in zip(df["close"], up, dn, ex_long, ex_short):
        if np.isnan(u):
            pos.append(0)
            continue
        if cur == 0:
            cur = 1 if c > u else (-1 if c < d else 0)
        elif cur == 1 and c < el:
            cur = 0
        elif cur == -1 and c > es:
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=df.index, dtype=float)


# ==================== 第一部分：Donchian 通道与未来函数 ====================


def donchian_basics(idx: pd.DataFrame):
    print("=" * 60)
    print("第一部分：Donchian 通道 —— 最纯粹的趋势定义")
    print("=" * 60)

    print("""
    "创 N 日新高就做多，创 N 日新低就做空"（Richard Donchian）

        Upper_t = max(H_{t-N}, ..., H_{t-1})       ← 窗口是 t-N 到 t-1，不包含当天！
        Lower_t = min(L_{t-N}, ..., L_{t-1})

    ❌ 陷阱：upper = high.rolling(20).max()           （包含当天）
             close > upper  →  永远不成立（当天收盘价 ≤ 当天最高价 ≤ 包含当天的通道上轨）
    ✅ 正确：upper = high.rolling(20).max().shift(1)
    """)

    up_wrong = idx["high"].rolling(20).max()
    up_right = idx["high"].rolling(20).max().shift(1)
    n_wrong = int((idx["close"] > up_wrong).sum())
    n_right = int((idx["close"] > up_right).sum())
    n_high = int((idx["high"] > up_right).sum())
    print(f"📊 沪深 300，{idx.index[0].date()} ~ {idx.index[-1].date()}：")
    print(f"   收盘价 > 含当天的 20 日最高价：{n_wrong} 天（逻辑上不可能）")
    print(f"   收盘价 > 不含当天的 20 日最高价：{n_right} 天")
    print(f"   最高价 > 不含当天的 20 日最高价：{n_high} 天（盘中触及突破，用于止损单式入场）")

    tgt_wrong = donchian_target(idx, include_today=True)
    print(f"\n   含当天版本的非零仓位天数：{int((tgt_wrong != 0).sum())} —— 策略一笔交易都开不出来")
    print("""
    💡 另一种更隐蔽的错法：用"最高价 > 含当天通道"判断突破 —— 每当今天创新高时永远成立，
       相当于"用今天的最高价确认今天突破"，回测会好得离谱。
    """)


# ==================== 第二部分：股指上的 Donchian ====================


def donchian_on_index(idx: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：沪深 300 上的 Donchian 突破")
    print("=" * 60)

    ret = idx["close"].pct_change().fillna(0)
    rows = {"买入持有": perf_stats(ret)}
    for ne, nx in [(20, 10), (55, 20), (120, 60)]:
        bt = backtest(donchian_target(idx, ne, nx), ret, IDX_COST)
        rows[f"{ne}/{nx}"] = perf_stats(bt["ret"], bt["turnover"], bt["pos"])
    print(f"多空版本（入场 N / 离场 N），单边成本 {IDX_COST:.2%}：")
    print(stats_table(rows).to_string())
    print("""
    📌 与双均线不同，Donchian 有"空仓"状态：离场通道比入场通道短，平仓后要等新的突破才再入场。
       所以持仓占比明显低于 100%，这也是它在震荡市中少亏一些的原因之一。
    """)


# ==================== 第三部分：期货上的 Donchian（含换月成本） ====================


def donchian_on_futures(symbol: str = "RB"):
    print("\n" + "=" * 60)
    print(f"第三部分：期货 {SPECS[symbol]['name']}（{symbol}）主力连续上的 Donchian")
    print("=" * 60)

    df = get_continuous(symbol)
    ret = df["close"].pct_change().fillna(0)  # 比例调整后的收益 = 实际持有合约的收益
    cost = cost_rate(symbol, df["raw_close"])
    print(f"""
    期货回测的三个细节（06 课详细展开）：
      ① 信号与收益用"比例调整"后的连续价，消除换月跳空（{int(df['roll'].sum())} 次换月）
      ② 成本 = 手续费（示例 万分之一）+ 1 tick 滑点（{symbol} 示例 tick = {SPECS[symbol]['tick']}），
         按真实合约价格换算，平均约 {cost.mean():.3%}（单边）
      ③ 换月日"平旧开新"：持仓时额外计两次成本
    数据区间：{df.index[0].date()} ~ {df.index[-1].date()}
    """)
    rows = {"买入持有(1x多头)": perf_stats(ret)}
    bts = {}
    for ne, nx in [(20, 10), (55, 20)]:
        bt = backtest(donchian_target(df, ne, nx), ret, cost, roll=df["roll"])
        bts[f"{ne}/{nx}"] = bt
        rows[f"Donchian {ne}/{nx}"] = perf_stats(bt["ret"], bt["turnover"], bt["pos"])
    bt_raw = backtest(donchian_target(df.assign(high=df["raw_high"], low=df["raw_low"], close=df["raw_close"]), 20, 10),
                      df["raw_close"].pct_change().fillna(0), cost, roll=df["roll"])
    rows["20/10 用未调整价(❌)"] = perf_stats(bt_raw["ret"], bt_raw["turnover"], bt_raw["pos"])
    print(stats_table(rows).to_string())

    ts = trade_stats(bts["20/10"]["pos"], ret)
    roll_cost = (bts["20/10"]["cost"] * df["roll"]).sum()
    print(f"""
    📊 Donchian 20/10 逐笔：{ts['笔数']} 笔，胜率 {ts['胜率']:.1%}，盈亏比 {ts['盈亏比']:.2f}，
       平均持有 {ts['平均持有天数']:.1f} 天；其中换月日产生的成本合计约 {roll_cost:.2%} 的净值
    ⚠️ "未调整价"一行：信号和收益都被换月跳空污染 —— 跳空日被当成真实涨跌，
       既制造假突破，又把新旧合约的价差算进收益。两行之间的差异就是数据处理错误的代价。
    """)
    return df, bts["20/10"]


# ==================== 第四部分：Dual Thrust ====================


def dual_thrust(symbol: str = "RB", n: int = 4, k1: float = 0.5, k2: float = 0.5):
    print("\n" + "=" * 60)
    print("第四部分：Dual Thrust —— 开盘区间突破（日内策略）")
    print("=" * 60)

    print(f"""
    Range    = max(HH - LC, HC - LL)          （过去 N={n} 日：HH 最高价最大值，LL 最低价最小值，
                                                HC 收盘价最大值，LC 收盘价最小值）
    BuyLine  = Open_t + K1 × Range            K1 = {k1}
    SellLine = Open_t - K2 × Range            K2 = {k2}
    盘中上破 BuyLine 做多，下破 SellLine 做空，收盘平仓（不隔夜）。

    逻辑："今天的波动如果显著超过近期正常波幅，说明有新信息驱动，跟上"。
    这里的 Range 只用过去 N 天（shift(1)），Open_t 在开盘时已知 → 两条线在盘中触发前就确定了，
    不存在未来函数；但 Dual Thrust 是日内策略，"次日执行"在这里体现为"线先定好，价格后触发"。
    """)

    df = get_continuous(symbol)
    hh = df["high"].rolling(n).max().shift(1)
    ll = df["low"].rolling(n).min().shift(1)
    hc = df["close"].rolling(n).max().shift(1)
    lc = df["close"].rolling(n).min().shift(1)
    rng = np.maximum(hh - lc, hc - ll)
    buy = df["open"] + k1 * rng
    sell = df["open"] - k2 * rng

    hit_up = df["high"] >= buy
    hit_dn = df["low"] <= sell
    both = hit_up & hit_dn
    valid = rng.notna()
    print(f"📊 {SPECS[symbol]['name']} 日线近似（{int(valid.sum())} 个交易日）：")
    print(f"   只触发 BuyLine {int((hit_up & ~hit_dn & valid).sum())} 天，只触发 SellLine {int((hit_dn & ~hit_up & valid).sum())} 天，"
          f"两条都触发 {int((both & valid).sum())} 天")

    cost = cost_rate(symbol, df["raw_close"])
    long_r = (df["close"] / np.maximum(buy, df["open"]) - 1).where(hit_up & ~hit_dn, 0.0)
    short_r = (1 - df["close"] / np.minimum(sell, df["open"])).where(hit_dn & ~hit_up, 0.0)
    trade_day = (hit_up ^ hit_dn) & valid
    base = (long_r + short_r - 2 * cost * trade_day).where(valid, 0.0)

    # 两条线都触发时：日线无法判断先后。按 Dual Thrust 的"反手"规则，两种顺序的结果不同：
    #   先多后空：在 BuyLine 买入 → 跌到 SellLine 反手做空 → 收盘平仓
    #   先空后多：在 SellLine 卖空 → 涨到 BuyLine 反手做多 → 收盘平仓
    # 每种顺序都有 4 次单边成交（开、反手 2 次、平）
    both_v = both & valid
    b_fill, s_fill = np.maximum(buy, df["open"]), np.minimum(sell, df["open"])
    up_first = (sell / b_fill - 1) + (1 - df["close"] / sell)
    dn_first = (1 - buy / s_fill) + (df["close"] / buy - 1)
    optimistic = base + (np.maximum(up_first, dn_first) - 4 * cost).where(both_v, 0.0)
    pessimistic = base + (np.minimum(up_first, dn_first) - 4 * cost).where(both_v, 0.0)

    rows = {
        "跳过双触发日": perf_stats(base),
        "双触发-按有利顺序": perf_stats(optimistic),
        "双触发-按不利顺序": perf_stats(pessimistic),
    }
    print(stats_table(rows).to_string())
    gap = (np.maximum(up_first, dn_first) - np.minimum(up_first, dn_first))[both_v]
    print(f"""
    📌 双触发日只有 {int(both_v.sum())} 天，但两种顺序的单日收益平均相差 {gap.mean():.2%}；
       而且无论哪种顺序，第一笔都会在反向线上被迫反手，必然亏损 —— 这正是日内策略怕"来回扫"的地方。
       注意"跳过双触发日"在实盘中根本做不到：第一条线被触发时你已经成交了。""")
    print("""
    ⚠️ 同一个策略，仅仅因为"无法判断盘中先后顺序"，回测结果就会出现差异；
       此外日线也无法模拟盘中滑点、涨跌停和夜盘。日内策略必须用分钟线回测，
       并正确处理国内期货的夜盘归属（夜盘属于下一个交易日，"开盘价"是夜盘开盘价）。
       这里的数字只用于说明失真，不代表策略好坏。
    """)


# ==================== 第五部分：图表 ====================


def plot_donchian(df: pd.DataFrame, bt: pd.DataFrame, symbol: str = "RB"):
    setup_chinese_font()
    recent = df.index[-400:]
    up = df["high"].rolling(20).max().shift(1)
    dn = df["low"].rolling(20).min().shift(1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    ax = axes[0]
    ax.plot(recent, df["close"].loc[recent], color="black", lw=0.8, label="收盘价(比例调整)")
    ax.plot(recent, up.loc[recent], color="red", lw=0.8, label="20 日上轨")
    ax.plot(recent, dn.loc[recent], color="green", lw=0.8, label="20 日下轨")
    pos = bt["pos"].loc[recent]
    ax.fill_between(recent, df["close"].loc[recent].min(), df["close"].loc[recent].max(),
                    where=pos > 0, color="red", alpha=0.08, label="持多")
    ax.fill_between(recent, df["close"].loc[recent].min(), df["close"].loc[recent].max(),
                    where=pos < 0, color="green", alpha=0.08, label="持空")
    ax.set_title(f"{SPECS[symbol]['name']} Donchian 20/10（最近 400 个交易日）")
    ax.legend(loc="upper left", fontsize=8)
    axes[1].plot(bt.index, bt["equity"], label="Donchian 20/10")
    axes[1].plot(df.index, df["close"] / df["close"].iloc[0], color="gray", label="买入持有")
    axes[1].set_title("净值（扣手续费、滑点与换月成本）")
    axes[1].legend()
    fig.tight_layout()
    path = save_figure(fig, "04_channel_breakout.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    idx = get_index_daily("000300", "20100101", "20251231")
    donchian_basics(idx)
    donchian_on_index(idx)
    df, bt = donchian_on_futures("RB")
    dual_thrust("RB")
    plot_donchian(df, bt, "RB")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把入场改为"盘中最高价突破上轨即以上轨价格成交"（止损单式入场），与"收盘确认"比较，哪个更接近实盘？
    2. 在 cta_utils.UNIVERSE 的其他品种上运行 Donchian 20/10 和 55/20，统计 Sharpe > 0 的品种数量
    3. 把 Dual Thrust 的 K1 设为 0.4、K2 设为 0.6（偏多），说明它表达了怎样的市场观点
    4. 给 Donchian 加一个 ATR 止损（入场价 - 2×ATR），观察最大回撤和交易笔数（05 课的海龟会用到）
    """)


if __name__ == "__main__":
    main()
