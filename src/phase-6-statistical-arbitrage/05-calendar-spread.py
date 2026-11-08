"""
期货跨期套利
============

学习目标：
    1. 理解跨期价差：同一品种、不同到期月份的两个合约之间的价差
    2. 用持有成本模型（资金 + 仓储 - 便利收益）理解价差的"锚"，区分 Contango 与 Backwardation
    3. 用 AKShare 单合约数据构造螺纹钢 01-05 跨期价差，按合约对分段研究
    4. 实现统计视角的跨期价差交易（Z-score），计入两条腿的手续费与滑点，并在进入交割月前强制平仓
    5. 认识跨期套利的特有风险：交割月限制、流动性枯竭、逼仓、腿风险

核心概念：
    - Spread_t = F_t(T1) - F_t(T2)，T1 < T2（本课：近月 - 远月）
    - 持有成本：F(T) = S · e^{(r + u - y)(T - t)}；r 资金成本、u 仓储、y 便利收益
    - Contango：远月 > 近月（库存充裕）；价差有"上限"（超过就可以买近、接货、存储、卖远交割）
    - Backwardation：近月 > 远月（现货紧张）；理论上没有明确上限

前置知识：
    - Phase 1 第 09 课（期货合约基础、换月）；Phase 5（期货连续合约）；本阶段 03 课（Z-score）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 首次运行下载 14 个单合约（之后读缓存 data/cache/stat_arb_fut_*.parquet）
    - 无网络时可用 QTC_OFFLINE=1 运行：优先读本地缓存，缓存不存在才使用合成合约数据
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stat_arb_utils import get_future_contract, pair_positions, used_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

YEARS = range(2020, 2027)  # 近月合约 RB2001 ... RB2601
MULT = 10  # 螺纹钢合约乘数：10 吨/手（以交易所最新规则为准）
TICK = 1.0  # 最小变动价位：1 元/吨
COMMISSION = 0.0001  # 单边手续费示例：成交金额的万分之一
R, STORAGE = 0.03, 0.15  # 资金成本（年化）与仓储费（元/吨/天），均为示例值
LOOKBACK, ENTRY, EXIT, STOP = 40, 2.0, 0.5, 4.0


# ==================== 第一部分：跨期价差与持有成本 ====================


def carry_model():
    print("=" * 60)
    print("第一部分：跨期价差与持有成本模型")
    print("=" * 60)

    print("""
    持有成本（Cost of Carry）：今天买现货存到 T 交割 ≈ 今天买 T 月期货
        F(T) = S · e^{(r + u - y)(T - t)}
        r 资金成本  u 仓储成本  y 便利收益（库存紧张时持有现货的好处）

    远近月理论价差：F(T2) - F(T1) ≈ F(T1) · [e^{(r + u - y)(T2 - T1)} - 1]

    ┌────────────────┬──────────────────────┬────────────────────────────────────┐
    │ 期限结构       │ 形态                 │ 价差边界                           │
    ├────────────────┼──────────────────────┼────────────────────────────────────┤
    │ Contango       │ 远月 > 近月          │ 有上限：远 - 近 ≤ 资金 + 仓储 + 交割 │
    │ （库存充裕）   │                      │ 超过就"买近、接货、存储、卖远交割"   │
    │ Backwardation  │ 近月 > 远月          │ 理论上没有明确上限（现货紧张）       │
    │ （现货紧张）   │                      │                                    │
    └────────────────┴──────────────────────┴────────────────────────────────────┘
    """)
    rows = []
    for price in [3000, 3500, 4000]:
        days = 120  # 01 → 05 约 4 个月
        fin = price * R * days / 365
        sto = STORAGE * days
        rows.append([price, round(fin, 1), round(sto, 1), round(fin + sto, 1)])
    df = pd.DataFrame(rows, columns=["近月价格(元/吨)", "资金成本", "仓储成本", "远-近 上限(元/吨)"])
    print(f"📐 01→05 合约约 120 天的完全持有成本（r = {R:.0%}，仓储 {STORAGE} 元/吨/天，均为示例值）：")
    print(df.to_string(index=False))
    print("   实际还要加上交割手续费、质检、出入库、增值税、仓单有效期等，以交易所和交割库最新规则为准。")


# ==================== 第二部分：构造 01-05 跨期价差 ====================


def load_spreads() -> dict:
    """每一组 (RB{yy}01, RB{yy}05)：从远月上市 1 个月后开始，到近月交割月前最后一个交易日"""
    segs = {}
    for y in YEARS:
        near_code, far_code = f"RB{y % 100:02d}01", f"RB{y % 100:02d}05"
        near, far = get_future_contract(near_code), get_future_contract(far_code)
        df = pd.concat({"near": near["close"], "far": far["close"], "near_vol": near["volume"],
                        "far_vol": far["volume"]}, axis=1).dropna()
        start = far.index[0] + pd.Timedelta(days=30)
        end = pd.Timestamp(year=y - 1, month=12, day=31)  # 进入交割月（1 月）前必须平仓
        win = df.loc[start:end].copy()
        if len(win) < LOOKBACK + 20:
            continue
        win["spread"] = win["near"] - win["far"]
        segs[f"{near_code}-{far_code}"] = dict(data=win, near_full=near)
    return segs


def spread_stats(segs: dict):
    print("\n" + "=" * 60)
    print("第二部分：螺纹钢 01-05 跨期价差（近月 - 远月）")
    print("=" * 60)

    rows = []
    for name, seg in segs.items():
        d = seg["data"]
        bound = d["near"] * R * 120 / 365 + STORAGE * 120  # 远 - 近 的持有成本上限
        rows.append([name, f"{d.index[0].date()}~{d.index[-1].date()}", len(d), d["spread"].mean(),
                     d["spread"].std(), d["spread"].min(), d["spread"].max(), (d["spread"] > 0).mean(),
                     int(((d["far"] - d["near"]) > bound).sum())])
    cols = ["合约对", "区间", "天数", "均值", "标准差", "最小", "最大", "近>远占比", "远-近超上限天数"]
    df = pd.DataFrame(rows, columns=cols)
    print("   价差单位：元/吨；\"近>远占比\" = Backwardation 天数占比")
    print(df.round(1).to_string(index=False))
    print(f"""
    💡 观察要点：
       - 价差的均值和波动在不同合约对之间差异很大：每组合约对都是一段独立的历史，
         不能简单拼接成一条长序列来估计"长期均值"
       - "近>远占比"高说明这段时间以 Backwardation 为主；持有成本只约束 Contango 一侧
       - "远-近超上限天数"> 0 时，理论上存在"买近卖远 + 交割"的正向机会（未计全部交割成本）
    """)
    return df


# ==================== 第三部分：统计视角的跨期价差交易 ====================


def spread_strategy(segs: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：统计视角的跨期价差交易（Z-score）")
    print("=" * 60)

    print(f"""
    规则（参数事先固定，不做优化）：
      z = (价差 - {LOOKBACK} 日滚动均值) / {LOOKBACK} 日滚动标准差（只在同一合约对内计算）
      z > {ENTRY}：卖近买远（押注近月相对走弱）；z < -{ENTRY}：买近卖远；|z| < {EXIT} 平仓；|z| > {STOP} 止损
      进入交割月前最后一个交易日强制平仓；每组 1 手近月 + 1 手远月（{MULT} 吨/手）
      成本：每条腿每次成交 = 手续费 {COMMISSION:.2%} × 价格 + 滑点 1 跳（{TICK:.0f} 元/吨）
    """)
    rows, pnl_all = [], []
    for name, seg in segs.items():
        d = seg["data"]
        s = d["spread"]
        z = (s - s.rolling(LOOKBACK).mean()) / s.rolling(LOOKBACK).std()
        z.iloc[-1] = np.nan  # 最后一天强制平仓
        pos = pair_positions(z, ENTRY, EXIT, STOP)  # +1 = 买近卖远（做多价差）
        held = pos.shift(1).fillna(0)
        gross = held * s.diff().fillna(0) * MULT
        trades = pos.diff().abs().fillna(pos.abs())
        cost = trades * ((d["near"] + d["far"]) * COMMISSION + 2 * TICK) * MULT
        net = gross - cost
        pnl_all.append(net)
        n_open = int(((pos != 0) & (pos.shift(1).fillna(0) != pos)).sum())
        rows.append([name, n_open, gross.sum(), cost.sum(), net.sum(), (net.cumsum() - net.cumsum().cummax()).min()])
    df = pd.DataFrame(rows, columns=["合约对", "开仓次数", "毛盈亏(元)", "成本(元)", "净盈亏(元)", "最大回撤(元)"])
    print(df.round(0).to_string(index=False))
    total = pd.concat(pnl_all)
    print(f"\n   合计：开仓 {df['开仓次数'].sum()} 次，净盈亏 {df['净盈亏(元)'].sum():,.0f} 元，"
          f"盈利的合约对 {int((df['净盈亏(元)'] > 0).sum())} / {len(df)}")
    margin = 3500 * MULT * 0.10 * 2  # 示例：两条腿各按 10% 保证金（部分交易所对套利持仓有保证金优惠）
    print(f"   参考：两条腿各 1 手、价格约 3500 元/吨、保证金 10% 时，占用保证金约 {margin:,.0f} 元")
    print("""
    💡 观察要点：
       - 每组合约对只有几个月，扣掉 40 天预热后可交易的时间更短，交易次数很少，统计上很难下结论
       - 跨期价差没有"到期必然收敛"的保证：两个合约到期日不同，价差可以长期单边移动
       - 结果以上表的实际数字为准；参数换一组，结论可能完全不同（练习 2）
    """)
    return df


# ==================== 第四部分：跨期套利的特有风险 ====================


def calendar_risks(segs: dict):
    print("\n" + "=" * 60)
    print("第四部分：跨期套利的特有风险")
    print("=" * 60)

    rows = []
    for name, seg in segs.items():
        d, near = seg["data"], seg["near_full"]
        last_month = near.index[-1].to_period("M")
        vol_deliv = near.loc[near.index.to_period("M") == last_month, "volume"].mean()
        vol_before = near.loc[d.index[-60]:d.index[-1], "volume"].mean()
        rows.append([name, d["spread"].diff().abs().max(), d["spread"].diff().abs().max() / d["spread"].std(),
                     vol_before, vol_deliv, vol_deliv / vol_before if vol_before > 0 else np.nan])
    cols = ["合约对", "价差最大单日变动", "相当于几倍标准差", "平仓前60天近月日均成交", "交割月日均成交", "成交量比"]
    df = pd.DataFrame(rows, columns=cols)
    ratio_median = df["成交量比"].median()
    df["成交量比"] = df["成交量比"].map("{:.2%}".format)
    print("   成交量单位：手（以数据源口径为准）")
    print(df.round(2).to_string(index=False))
    print("""
    ┌──────────────┬──────────────────────────────────────────────────────────────┐
    │ 风险         │ 说明（以交易所最新规则为准）                                 │
    ├──────────────┼──────────────────────────────────────────────────────────────┤
    │ 临近交割     │ 近月进入交割月后流动性骤降、保证金提高；多数品种不允许       │
    │              │ 自然人客户持仓进入交割月，必须提前平仓或移仓                 │
    │ 逼仓         │ 可交割货源紧张时近月大幅上涨，价差远离"理论值"               │
    │ 腿风险       │ 一条腿涨停、另一条没有，无法同时成交；名义金额也不完全相等   │
    │ 合约对切换   │ 01-05 到期后要切换到下一组，历史价差需按合约对分段研究       │
    │ 套利指令     │ 部分交易所提供跨期套利指令和保证金优惠，可降低成本与腿风险   │
    └──────────────┴──────────────────────────────────────────────────────────────┘

    """)
    print(f"    💡 交割月日均成交只有平仓前的 {ratio_median:.2%}（各合约对中位数）：进入交割月后近月合约流动性"
          f"大幅萎缩，\n       这是必须提前平仓或移仓的原因之一。")


def plot_spreads(segs: dict):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, seg in segs.items():
        d = seg["data"]
        days_left = (d.index[-1] - d.index).days
        ax.plot(days_left, d["spread"], label=name, lw=1)
    ax.invert_xaxis()
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("距平仓日的天数"), ax.set_ylabel("近月 - 远月（元/吨）")
    ax.set_title(f"螺纹钢 01-05 跨期价差{'【含合成数据】' if used_synthetic() else ''}")
    ax.legend(fontsize=8, ncol=2)
    print(f"\n🖼️  图已保存：{save_figure(fig, '05_calendar_spread.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    carry_model()
    segs = load_spreads()
    spread_stats(segs)
    spread_strategy(segs)
    calendar_risks(segs)
    plot_spreads(segs)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 构造 05-10 合约对（RB2405-RB2410 等），与 01-05 对比价差均值和 Backwardation 占比
    2. 把 LOOKBACK 改为 20、60，ENTRY 改为 1.5、2.5，结果稳定吗？哪组参数"最好"是否可信？
    3. 用热卷（HC）重复本课；再构造"螺纹 - 热卷"同月合约的跨品种价差，比较二者的稳定性
    4. 估算：若某天 RB01 涨停而 RB05 未涨停，你的价差头寸会面临什么风险？如何在下单逻辑中处理？
    """)


if __name__ == "__main__":
    main()
