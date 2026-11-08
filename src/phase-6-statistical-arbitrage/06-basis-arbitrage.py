"""
期现套利与基差
==============

学习目标：
    1. 理解基差的定义与符号约定，以及"到期收敛"这一期现套利唯一的制度保证
    2. 掌握股指期货持有成本模型 F* = S·e^{(r-q)T} 与无套利区间 [F* - C, F* + C]
    3. 用沪深 300 股指期货（IF）与沪深 300 指数计算基差、年化基差，观察到期收敛
    4. 发现 A 股股指期货的两个现实特征：长期贴水、分红季节性
    5. 理解 A 股反向期现套利为什么难，以及"期货替代现货""多头 + 期货对冲"的成本含义

核心概念：
    - 基差：商品期货常用 Basis = S - F；股指期货交易员常说"期货相对指数升 / 贴水多少点"（F - S）
      本课统一用 F - S（负数 = 贴水），读研报时先确认符号约定
    - 年化基差 = ln(F / S) × 365 / 剩余天数
    - 无套利区间：考虑交易成本 C 后，F 在 [F* - C, F* + C] 内不存在套利机会

前置知识：
    - Phase 1 第 09 课（期货基础）；本阶段 05 课（持有成本）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 首次运行下载 IF0 主力连续 + 24 个季月合约（之后读缓存 data/cache/stat_arb_*.parquet）
    - 无网络时可用 QTC_OFFLINE=1 运行：优先读本地缓存，缓存不存在才使用合成数据
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

from stat_arb_utils import (INDEX_END, INDEX_START, expected_dividend_frac, get_future_contract, get_if_main,
                            used_synthetic, warn_synthetic)
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

START, END = "20190101", "20251231"
QUARTERS = [f"IF{y:02d}{m:02d}" for y in range(20, 26) for m in (3, 6, 9, 12)]
R = 0.02  # 资金成本示例（年化）
DIV_YIELD = 0.025  # 沪深 300 年分红率示例假设
COST_RATE = 0.003  # 完成一次期现套利的总成本示例：指数点位的 0.3%（现货篮子、冲击、跟踪误差等）


def index_futures_arb(S, F, r, q, days, cost_points):
    """持有成本模型：理论价、无套利区间、年化升贴水与操作建议"""
    T = days / 365
    fair = S * np.exp((r - q) * T)
    upper, lower = fair + cost_points, fair - cost_points
    if F > upper:
        action = "正向套利：买现货、卖期货"
    elif F < lower:
        action = "反向套利：卖现货、买期货"
    else:
        action = "无套利机会"
    return fair, lower, upper, (F - S) / S * 365 / days, action


# ==================== 第一部分：基差与持有成本模型 ====================


def basis_concept():
    print("=" * 60)
    print("第一部分：基差、持有成本模型与无套利区间")
    print("=" * 60)

    print("""
    股指期货的持有成本模型（现金交割、无仓储，持有现货能拿到分红）：
        F* = S · e^{(r - q)(T - t)}       r 资金成本、q 剩余期限内的分红率（年化）

    考虑成本 C（现货篮子买卖、期货开平、冲击、跟踪误差、资金占用，折算为指数点）：
        F > F* + C → 正向期现套利：买现货（ETF / 股票篮子）+ 卖期货，持有到期
        F < F* - C → 反向期现套利：卖现货 + 买期货（A 股需要融券，很难执行）

    到期收敛：交割时期货必须收敛到现货（IF 以最后交易日指数最后 2 小时的算术平均价交割），
    这是期现套利唯一的"制度保证"。
    """)
    rows = []
    for F, q, days in [(3780, 0.02, 30), (3830, 0.0, 30), (3740, 0.0, 60), (3740, 0.12, 60)]:
        fair, lo, up, ann, act = index_futures_arb(3800, F, R, q, days, cost_points=8)
        rows.append([3800, F, f"{q:.0%}", days, round(fair, 1), f"[{lo:.1f}, {up:.1f}]", f"{ann:+.2%}", act])
    cols = ["指数S", "期货F", "分红率q(年化)", "剩余天数", "理论价F*", "无套利区间(C=8点)", "年化升贴水", "结论"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("""
    💡 后两行期货价格相同，只是剩余期限内的分红率不同（年化 12% × 60 天 ≈ 指数的 2%，
       相当于分红集中落在这 60 天里）：同样的贴水，扣除分红后就落在无套利区间之内。
       不考虑分红，就会把"分红导致的贴水"误认为"期货便宜"。
    """)


# ==================== 第二部分：IF 主力连续 vs 沪深 300 ====================


def main_contract_basis(spot: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：IF 主力连续 vs 沪深 300 指数")
    print("=" * 60)

    fut = get_if_main(START, END)
    df = pd.concat({"F": fut, "S": spot}, axis=1).dropna()
    df["basis"] = df["F"] - df["S"]
    df["basis_pct"] = df["basis"] / df["S"]
    g = df.groupby(df.index.year)
    table = pd.DataFrame({
        "天数": g.size(),
        "平均基差(点)": g["basis"].mean(),
        "平均基差(%)": g["basis_pct"].mean() * 100,
        "贴水天数占比": g["basis"].apply(lambda s: (s < 0).mean()),
        "最深贴水(%)": g["basis_pct"].min() * 100,
    })
    print(f"   {df.index[0].date()} ~ {df.index[-1].date()}，基差 = F - S（负数 = 贴水）")
    print(table.round(2).to_string())
    print(f"\n   全区间贴水天数占比：{(df['basis'] < 0).mean():.1%}")
    print("""
    ⚠️ 主力连续在换月时从近月切到远月，基差会跳变（剩余期限突然变长）：
       研究基差的期限结构和到期收敛，要用单个合约（第三部分）。
    """)
    return df


# ==================== 第三部分：单合约 —— 到期收敛与分红季节性 ====================


def load_quarterlies(spot: pd.Series) -> pd.DataFrame:
    """季月合约长表：date, contract, F, S, days(剩余自然日), ann_basis(年化基差)"""
    parts = []
    for code in QUARTERS:
        c = get_future_contract(code)
        expiry = c.index[-1]  # 数据最后一天 = 最后交易日
        d = pd.DataFrame({"F": c["close"], "S": spot.reindex(c.index)}).dropna()
        d["contract"], d["expiry"] = code, expiry
        d["days"] = (expiry - d.index).days
        parts.append(d)
    q = pd.concat(parts).rename_axis("date").reset_index()
    q["ann_basis"] = np.log(q["F"] / q["S"]) * 365 / q["days"].clip(lower=1)
    return q


def convergence_and_seasonality(q: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：到期收敛与分红季节性（季月合约）")
    print("=" * 60)

    rows = []
    for code, g in q.groupby("contract"):
        g = g.set_index("days")["F"].sub(g.set_index("days")["S"]) / g.set_index("days")["S"] * 100
        pick = lambda k: g[g.index >= k].sort_index().iloc[0] if (g.index >= k).any() else np.nan  # noqa: E731
        rows.append([code, pick(120), pick(60), pick(20), pick(5), g.sort_index().iloc[0]])
    conv = pd.DataFrame(rows, columns=["合约", "T-120天", "T-60天", "T-20天", "T-5天", "到期日"])
    print("📋 基差 (F-S)/S（%）随剩余期限的变化，最近 8 个季月合约：")
    print(conv.tail(8).round(2).to_string(index=False))
    print(f"   到期日收盘基差的绝对值平均：{conv['到期日'].abs().mean():.3f}%"
          "（交割价是最后 2 小时均价，所以收盘价不必精确等于指数）")

    # 每天取剩余 20 天以上、到期最近的季月合约 → 年化基差的月度分布
    near = q[q["days"] >= 20].sort_values("days").groupby("date").first()
    by_month = near.groupby(near.index.month)["ann_basis"].mean() * 100
    print(f"\n📋 近季合约年化基差（%）按日历月份平均（{near.index[0].date()} ~ {near.index[-1].date()}）：")
    print("   " + "  ".join(f"{m}月 {v:+.1f}" for m, v in by_month.items()))
    deepest = by_month.nsmallest(3).index.tolist()
    print(f"   年化贴水最深的 3 个月份：{deepest}")
    print("""
    💡 A 股分红集中在年中（除息多在 5–8 月）。在此之前、到期日覆盖分红期的合约，
       理论价格要扣除这段时间的分红，表现为季节性的"深贴水"。
       对照上面年化贴水最深的月份，判断它是否与分红期吻合 —— 不考虑分红会误判贴水"便宜"。
    """)
    return near


# ==================== 第四部分：无套利区间 ====================


def no_arbitrage_band(near: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：无套利区间 —— 有没有真正的套利机会？")
    print("=" * 60)

    div = np.array([expected_dividend_frac(pd.DatetimeIndex([d]), e, DIV_YIELD)[0]
                    for d, e in zip(near.index, near["expiry"])])
    tau = near["days"].to_numpy() / 365
    S, F = near["S"].to_numpy(), near["F"].to_numpy()
    cost = S * COST_RATE
    rows = []
    for label, fair in [("不考虑分红：F* = S·e^{rT}", S * np.exp(R * tau)),
                        ("扣除预期分红：F* = S·e^{rT - D}", S * np.exp(R * tau - div))]:
        rows.append([label, np.mean(F > fair + cost), np.mean(F < fair - cost),
                     np.mean((F - fair) / S) * 100])
    df = pd.DataFrame(rows, columns=["理论价", "F > 上界 占比", "F < 下界 占比", "F 偏离 F* 平均(%)"])
    print(f"   假设：r = {R:.0%}，沪深 300 年分红率 {DIV_YIELD:.1%}（按月份集中分布，示例假设），"
          f"成本 C = 指数的 {COST_RATE:.1%}")
    print(df.round(3).to_string(index=False))
    print("""
    💡 观察要点：
       1. 扣除预期分红后，"F 低于下界"的比例通常会下降 —— 一部分贴水只是分红的体现
       2. 剩下的、越过下界的贴水，理论上是反向套利机会（卖现货 + 买期货），
          但在 A 股要卖空现货篮子需要融券，现实中很难执行 —— 这正是贴水能长期存在的原因之一
       3. 理论价对 r、q、C 的假设很敏感；真实研究应使用成分股分红预测和实际成交成本
    """)


# ==================== 第五部分：A 股现实 ====================


def a_share_reality(near: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：A 股股指期货的现实 —— 贴水是谁的成本、谁的收益")
    print("=" * 60)

    ann = near.groupby(near.index.year)["ann_basis"].mean() * 100
    print("📋 近季合约平均年化基差（%）：" + "，".join(f"{y} {v:+.2f}" for y, v in ann.items()))
    avg = near["ann_basis"].mean()
    print(f"   全区间平均：{avg:+.2%} / 年")
    print(f"""
    ┌──────────────────────────────┬───────────────────────────────────────────────┐
    │ 参与者                       │ 年化基差 {avg:+.2%} 意味着                      │
    ├──────────────────────────────┼───────────────────────────────────────────────┤
    │ 市场中性 / 指数增强对冲      │ 持有空头期货：贴水时，期货向现货收敛带来亏损    │
    │ （多股票 + 空期货）          │ → 贴水是对冲成本                               │
    │ 期货替代现货                 │ 持有多头期货：贴水收敛带来额外收益              │
    │ （卖现货、买贴水期货）       │ → 贴水是收益（反向套利的"机构版"）              │
    └──────────────────────────────┴───────────────────────────────────────────────┘

    其他现实约束：
      - 反向套利主要由已持有现货的机构完成（卖出手中的现货、买入期货替代，到期换回）
      - 现货复制有跟踪误差；成分股停牌、涨跌停会让篮子无法按计划建仓或平仓
      - 当月、下月、两个季月合约的年化基差不同，可以选择对冲成本最低的合约
      - 合约规则、保证金、交割方式以中金所最新规则为准
    """)


def plot_basis(df: pd.DataFrame, near: pd.DataFrame):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(df.index, df["basis_pct"] * 100, lw=0.8)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title(f"IF 主力连续基差 (F-S)/S（%）{'【含合成数据】' if used_synthetic() else ''}")
    by_month = near.groupby(near.index.month)["ann_basis"].mean() * 100
    axes[1].bar(by_month.index, by_month.values)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("近季合约年化基差（%）按日历月份平均")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '06_basis.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    idx = get_index_daily("000300", INDEX_START, INDEX_END)
    if is_synthetic(idx):
        warn_synthetic("沪深 300 指数", "utils 返回合成行情")
    spot = idx["close"]
    basis_concept()
    df = main_contract_basis(spot.loc[START:END])
    q = load_quarterlies(spot)
    near = convergence_and_seasonality(q)
    no_arbitrage_band(near)
    a_share_reality(near)
    plot_basis(df, near)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用中证 500 股指期货（IC0 / 指数 000905）重复第二部分，比较 IC 与 IF 的贴水深度
    2. 把分红率假设改为 2% 和 3%，第四部分"F < 下界"的比例如何变化？结论对假设敏感吗？
    3. 计算一个"多沪深300ETF + 空 IF 近季合约"的对冲组合，每季度移仓，统计基差带来的年化成本
    4. 思考：为什么正向套利（买现货、卖期货）在 A 股比反向套利容易执行？它在什么市场环境下会出现？
    """)


if __name__ == "__main__":
    main()
