"""
分层回测与多空组合
==================

学习目标：
    1. 实现按因子值分 5 组的分层回测，判断单调性
    2. 计算换手率和交易成本，看因子扣费后还剩多少收益
    3. 处理 A 股"空头腿难以实现"的问题：纯多头超额、用指数作为空头
    4. 把"涨停买不进、跌停 / 停牌卖不出"放进组合模拟，量化它们的影响

核心概念：
    - 分层回测：每个调仓日把股票等分 N 组，每组等权持有到下一调仓日
    - 单调性：组号与平均收益的 Spearman 相关，越接近 1 越好
    - 多空收益：第 5 组 − 第 1 组；理论检验工具，A 股难以直接实现
    - 换手率：每期被替换的持仓比例；成本 = 换手 × (买入费率 + 卖出费率)

前置知识：
    - 05-ic-ir-analysis.py
    - Phase 4 的 04-transaction-costs-slippage.py、05-a-share-constraints.py

环境要求：
    - pip install pandas numpy scipy matplotlib
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）

⚠️ 费率为示例值，以券商和交易所最新规则为准；股票池有幸存者偏差，收益绝对水平偏高。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

from factor_data import load_panel
from factor_tools import (COMMISSION, FACTOR_LABELS, SLIPPAGE, bench_period_returns, calc_ic, infer_st, limit_pct,
                          perf_stats, preprocess, quantile_labels, quantile_returns, research_inputs, stamp_tax)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

MAIN_FACTOR = "lowturn"  # 本课主讲因子：低换手
N_GROUPS = 5


# ==================== 第一部分：分层回测 ====================


def quantile_backtest(clean, ret):
    print("=" * 60)
    print(f"第一部分：分层回测（{FACTOR_LABELS[MAIN_FACTOR]}，{N_GROUPS} 组，月度调仓）")
    print("=" * 60)
    print("""
    IC 告诉你"相关"，但不告诉你能赚多少钱、收益是否单调。分层回测更直观：
       1. 每个调仓日（t 日收盘）按因子值把可交易股票等分为 5 组
       2. t+1 日开盘买入，每组等权，持有到下一调仓日的次日开盘
       3. 比较各组收益。好因子的三个特征：
          ① 单调：Q1 < Q2 < Q3 < Q4 < Q5
          ② 多空（Q5 − Q1）稳定向上
          ③ 收益不只来自某一头（例如只有 Q1 特别差）
    """)
    grp = quantile_returns(clean[MAIN_FACTOR], ret, N_GROUPS)
    stats = grp.apply(perf_stats)
    print(f"📋 各组表现（{len(grp)} 期，未扣成本）：")
    print(stats.T.to_string(float_format=lambda v: f"{v:+.3f}"))
    means = grp[list(range(1, N_GROUPS + 1))].mean()
    rho = spearmanr(np.arange(1, N_GROUPS + 1), means.to_numpy(dtype=float)).statistic
    t = grp["多空"].mean() / grp["多空"].std() * np.sqrt(len(grp))
    print(f"\n   单调性（组号与月均收益的 Spearman 相关）：{rho:+.2f}")
    print(f"   多空月均 {grp['多空'].mean():+.2%}，t 值 {t:+.2f}")

    # 组内"平均"与"中位数"的差异：少数暴涨股会拉高算术平均
    med = {}
    for d in grp.index:
        f, r = clean[MAIN_FACTOR].loc[d], ret.loc[d]
        ok = f.notna() & r.notna()
        med[d] = r[ok].groupby(quantile_labels(f[ok], N_GROUPS)).median()
    med = pd.DataFrame(med).T.mean()
    ic = calc_ic(clean[MAIN_FACTOR], ret).mean()
    print("\n   各组月收益：" + "  ".join(
        f"Q{i} 均值 {means[i]:+.2%} / 中位数 {med[i]:+.2%}" for i in range(1, N_GROUPS + 1)))
    print(f"""
    💡 同一个因子的 Rank IC 均值是 {ic:+.4f}，而多空月均收益是 {grp['多空'].mean():+.2%}。
       两者不一定同号：Rank IC 看的是全部股票的排序，对极端值不敏感；
       分组的算术平均却会被组内少数暴涨股拉高（比较上面每组的均值和中位数）。
       Q1 与 Q5 的"均值差"和"中位数差"分别是 {means[N_GROUPS] - means[1]:+.2%}、{med[N_GROUPS] - med[1]:+.2%} ——
       两个口径不一致时，要看收益是不是靠少数几只股票撑起来的。
    """)
    return grp


def all_factors_table(panel, clean, ret):
    print("\n📋 全部因子的分层结果（月均收益 × 12）：")
    rows = {}
    for k, f in clean.items():
        g = quantile_returns(f, ret, N_GROUPS)
        m = g[list(range(1, N_GROUPS + 1))].mean() * 12
        rows[FACTOR_LABELS[k]] = {**{f"Q{i}": m[i] for i in m.index}, "Q5−Q1": g["多空"].mean() * 12,
                                  "t值": g["多空"].mean() / g["多空"].std() * np.sqrt(len(g)),
                                  "单调性": spearmanr(np.arange(1, N_GROUPS + 1), m.to_numpy(dtype=float)).statistic}
    df = pd.DataFrame(rows).T.sort_values("t值", ascending=False)
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    if not panel.is_synthetic:
        print("""
    💡 各组收益普遍很高是股票池的幸存者偏差造成的，请比较"组间差异"，不要看绝对水平。""")
    print("""
    💡 每组只有二十多只股票，单调性很容易被个别股票打破。
    """)


# ==================== 第二部分：换手率与成本 ====================


def group_members(factor: pd.DataFrame, g: int) -> dict:
    out = {}
    for d in factor.index:
        f = factor.loc[d].dropna()
        if len(f) < 30:
            continue
        lab = quantile_labels(f, N_GROUPS)
        out[d] = set(lab.index[lab == g])
    return out


def turnover_and_cost(members: dict) -> pd.DataFrame:
    """单边换手 = 本期新买入的比例；成本 = 换手 × (买入费率 + 卖出费率)，首期只有买入"""
    rows, prev = {}, None
    for d, cur in members.items():
        buy_cost = COMMISSION + SLIPPAGE
        sell_cost = COMMISSION + SLIPPAGE + stamp_tax(d)
        if prev is None:
            rows[d] = {"换手": 1.0, "成本": buy_cost}
        else:
            turn = len(cur - prev) / len(cur)
            rows[d] = {"换手": turn, "成本": turn * (buy_cost + sell_cost)}
        prev = cur
    return pd.DataFrame(rows).T


def cost_analysis(clean, grp):
    print("\n" + "=" * 60)
    print("第二部分：换手率与交易成本")
    print("=" * 60)
    print(f"""
    费率假设（示例值）：佣金 {COMMISSION:.3%}（双边）、滑点 / 冲击 {SLIPPAGE:.2%}（双边）、
    印花税仅卖出：2023-08-28 前 0.1%，之后 0.05%
    """)
    rows, costs = {}, {}
    for k, f in clean.items():
        c = turnover_and_cost(group_members(f, N_GROUPS))
        costs[k] = c["成本"]
        rows[FACTOR_LABELS[k]] = {"月均单边换手": c["换手"].iloc[1:].mean(), "年化成本": c["成本"].iloc[1:].mean() * 12}
    print("📋 各因子 Top 组的换手与成本：")
    print(pd.DataFrame(rows).T.sort_values("月均单边换手").to_string(float_format=lambda v: f"{v:.3f}"))

    top_gross = grp[N_GROUPS]
    top_net = top_gross - costs[MAIN_FACTOR].reindex(top_gross.index).fillna(0)
    bottom_cost = turnover_and_cost(group_members(clean[MAIN_FACTOR], 1))["成本"].reindex(grp.index).fillna(0)
    ls_net = grp["多空"] - costs[MAIN_FACTOR].reindex(grp.index).fillna(0) - bottom_cost
    df = pd.DataFrame({"Top组 毛": perf_stats(top_gross), "Top组 净": perf_stats(top_net),
                       "多空 毛": perf_stats(grp["多空"]), "多空 净": perf_stats(ls_net)})
    print(f"\n📋 {FACTOR_LABELS[MAIN_FACTOR]} 扣费前后：")
    print(df.T.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    💡 短期反转这类信号变化快的因子，换手明显更高 —— 看 IC 时很漂亮的因子，扣费后可能所剩无几。
       多空组合两条腿都要交易，成本大约翻倍（还没算融券费用）。
    """)
    return top_net


# ==================== 第三部分：A 股的多空现实 ====================


def long_only_reality(panel, dates, mask, ret, grp, top_net):
    print("\n" + "=" * 60)
    print("第三部分：A 股的多空现实 —— 用指数作为空头")
    print("=" * 60)
    print("""
    A 股融券受限，第 1 组的空头腿基本无法实现。实践中的替代：
    ┌────────────┬────────────────────────────────┬──────────────────────────────┐
    │ 实现方式   │ 做法                           │ 注意                         │
    ├────────────┼────────────────────────────────┼──────────────────────────────┤
    │ 纯多头超额 │ Top 组 − 基准指数 / 等权股票池 │ 最贴近"选股能力"             │
    │ 指数增强   │ 在指数成分内选股，控制跟踪误差 │ 行业、风格偏离受限           │
    │ 市场中性   │ 多头股票 + 做空股指期货        │ 对冲成本受基差影响（Phase 6）│
    └────────────┴────────────────────────────────┴──────────────────────────────┘
    """)
    idx = grp.index
    ew = ret.where(mask).mean(axis=1).reindex(idx)
    hs = bench_period_returns(panel, dates, "hs300").reindex(idx)
    zz = bench_period_returns(panel, dates, "zz500").reindex(idx)
    legs = {"Top − 等权股票池": top_net - ew, "Top − 沪深300": top_net - hs, "Top − 中证500": top_net - zz}
    df = pd.DataFrame({k: perf_stats(v) for k, v in legs.items()}).T
    df["t值"] = [v.mean() / v.std() * np.sqrt(len(v)) for v in legs.values()]
    print("📋 Top 组（已扣费）相对不同空头的表现：")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))

    long_part = (grp[N_GROUPS] - ew).mean() * 12
    short_part = (ew - grp[1]).mean() * 12
    index_note = ("""⚠️ 相对指数的超额要打折看：① 指数是价格指数，不含分红；② 股票池有幸存者偏差；
       ③ 用股指期货对冲时还要承担基差成本（IC、IM 贴水时尤其明显）。
       "Top − 等权股票池"不受 ① 影响、受 ② 的影响也小得多，是本课最可信的一行。"""
                  if not panel.is_synthetic else
                  "⚠️ 用股指期货对冲时还要承担基差成本（IC、IM 贴水时尤其明显），这里没有计入。")
    if long_part * short_part > 0:
        share_txt = f"多头贡献占比：{long_part / (long_part + short_part):.0%}"
    else:
        share_txt = "两头符号相反：多空收益不能简单按比例拆分，要分别看每一头是否有效"
    print(f"""
    多空收益分解（毛收益，月均 × 12）：
       多头贡献 Q5 − 等权池：{long_part:+.2%}
       空头贡献 等权池 − Q1：{short_part:+.2%}
       {share_txt}

    ⚠️ 如果多空收益主要来自"空头组特别差"，而多头组相对基准没有超额，
       这个因子在做不了空的 A 股多头策略里几乎没有用。
    {index_note}
    """)


# ==================== 第四部分：买不进与卖不出 ====================


def sellable_status(panel, dates) -> pd.DataFrame:
    """调仓日 t 的持仓能否在 t+1 开盘卖出：未停牌，且开盘价高于跌停价"""
    cal = panel.dates
    pos = cal.get_indexer(dates)
    nxt = np.minimum(pos + 1, len(cal) - 1)
    lim = limit_pct(panel, infer_st(panel))
    prev = panel.close_ffill.iloc[pos].to_numpy()
    dn = np.round(prev * (1 - lim.iloc[nxt].to_numpy()), 2)
    op = panel.open.iloc[nxt].to_numpy()
    with np.errstate(invalid="ignore"):
        return pd.DataFrame(~np.isnan(op) & (op > dn + 0.001), index=dates, columns=panel.codes)


def execution_simulation(panel, dates, mask, ret, raw, clean):
    print("\n" + "=" * 60)
    print("第四部分：把\"买不进、卖不出\"放进组合模拟")
    print("=" * 60)

    sellable = sellable_status(panel, dates)
    ideal_factor = preprocess(raw[MAIN_FACTOR], panel, None)  # 不过滤：假设什么都能成交
    ideal = pd.Series({d: ret.loc[d, list(m)].mean() for d, m in group_members(ideal_factor, N_GROUPS).items()})

    real, free, held, events = {}, {}, set(), []
    for d, target in group_members(clean[MAIN_FACTOR], N_GROUPS).items():
        stuck = {c for c in held - target if not sellable.loc[d, c]}  # 想卖卖不掉，继续持有
        held = target | stuck
        events += [(d, c, ret.loc[d, c]) for c in stuck]
        real[d] = ret.loc[d, list(held)].mean()
        free[d] = ret.loc[d, list(target)].mean()
    real, free = pd.Series(real), pd.Series(free)
    n_stuck = len(events)

    df = pd.DataFrame({"不过滤、都能成交": perf_stats(ideal), "过滤、假设都能卖出": perf_stats(free),
                       "过滤 + 卖不出继续持有": perf_stats(real)})
    print("📋 Top 组毛收益：")
    print(df.T.to_string(float_format=lambda v: f"{v:+.3f}"))
    if events:
        print("\n   卖不出的持仓（被迫多持有一期）：")
        for d, c, r in events[:8]:
            print(f"     {d.date()} {c} {panel.info.loc[c, 'name']}：被迫持有期间收益 {r:+.1%}")
    print(f"""
    模拟规则：
       - 目标持仓只从可交易股票中选（剔除次新、ST、停牌、次日开盘涨停）
       - 想卖出的持仓如果 t+1 停牌或开盘跌停 → 卖不出，继续持有到下一期
       - 卖不出的情况共发生 {n_stuck} 次

    💡 差异的方向不是固定的：过滤会让你错过一些涨停股（少赚），也会让你躲开 ST 和次新股；
       被迫持有的股票可能继续下跌，也可能复牌后大涨 —— 只有几次事件就能明显改变年化收益，
       这说明个别极端事件对小组合的影响很大，不能把这部分收益当成因子的贡献。
       本模拟未处理"已持有的股票开盘涨停"：它被排除出目标后会被卖出，实盘可以继续持有。
    """)


# ==================== 主函数 ====================


def plot_results(panel, dates, grp):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    nav = (1 + grp[list(range(1, N_GROUPS + 1))]).cumprod()
    for c in nav.columns:
        axes[0].plot(nav.index, nav[c], label=f"Q{c}")
    axes[0].set_title(f"{FACTOR_LABELS[MAIN_FACTOR]}：分组净值（未扣费）")
    axes[0].legend()
    hs = bench_period_returns(panel, dates, "hs300").reindex(grp.index)
    axes[1].plot(grp.index, (1 + grp["多空"]).cumprod(), label="多空 Q5 − Q1")
    axes[1].plot(grp.index, (1 + grp[N_GROUPS] - hs).cumprod(), label="Q5 − 沪深300")
    axes[1].axhline(1, color="gray", lw=0.6)
    axes[1].set_title("多空净值 vs 以指数作为空头")
    axes[1].legend()
    path = save_figure(fig, "06_quantile_backtest.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    dates, mask, ret, raw, clean = research_inputs(panel)
    grp = quantile_backtest(clean, ret)
    all_factors_table(panel, clean, ret)
    top_net = cost_analysis(clean, grp)
    long_only_reality(panel, dates, mask, ret, grp, top_net)
    execution_simulation(panel, dates, mask, ret, raw, clean)
    plot_results(panel, dates, grp)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把分组数改为 3 和 10，观察单调性和多空收益的稳定性（每组股票数够吗？）
    2. 把组内等权改为市值加权，比较 Top 组的收益与换手，思考等权隐含的小盘倾斜
    3. 改为双月调仓，比较短期反转和低换手因子的扣费后收益
    4. 完善第四部分：已持有且仍排名靠前、但次日开盘涨停的股票应继续持有，修改模拟逻辑
    """)


if __name__ == "__main__":
    main()
