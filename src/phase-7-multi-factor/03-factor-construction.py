"""
因子构建：价值、动量、质量、低波、规模
======================================

学习目标：
    1. 认识常见因子的定义、计算方式和 A 股经验方向
    2. 在截面面板上计算 8 个因子，并把方向统一为"越大越好"
    3. 理解为什么用 EP 而不是 PE、为什么很多因子是"小市值的马甲"
    4. 实现可交易性过滤（停牌、涨停买不进、ST、次新股），认识幸存者偏差

核心概念：
    - 截面面板：index = 日期，columns = 股票，每个月末给每只股票打一个分
    - 方向统一：把"越小越好"的因子取负号，后续合成时不必再记方向
    - 可交易性过滤：t 日收盘算信号，t+1 日开盘执行，买不进的股票不能算进组合
    - 幸存者偏差：用"今天的成分股"回测过去，会系统性高估收益

前置知识：
    - 01-capm-beta.py、02-fama-french.py
    - Phase 1 的 07-trading-calendar-alignment.py、10-data-biases-quality.py

环境要求：
    - pip install pandas numpy akshare
    - 数据来自本阶段共享面板 factor_data.py；第六部分需要联网下载退市股票样本
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from factor_data import load_delisted_sample, load_panel
from factor_tools import (FACTOR_CATEGORY, FACTOR_LABELS, build_factors, infer_st, period_returns,
                          rebalance_dates, tradable_status)

load_dotenv()


# ==================== 第一部分：因子一览 ====================


def factor_zoo():
    print("=" * 60)
    print("第一部分：常见因子一览")
    print("=" * 60)
    print("""
    ┌────────┬────────────┬──────────────────────────────┬─────────────────────┐
    │ 大类   │ 因子       │ 计算（本课实现）             │ 方向（A 股经验）    │
    ├────────┼────────────┼──────────────────────────────┼─────────────────────┤
    │ 价值   │ EP         │ 归母净利润 TTM / 总市值      │ 越高越好            │
    │        │ BP         │ 净资产 / 总市值              │ 越高越好            │
    │ 质量   │ ROE        │ 净利润 TTM / 净资产          │ 越高越好            │
    │ 动量   │ 动量 12-1  │ 过去 12 个月收益，剔除近 1 月│ A 股较弱、不稳定    │
    │ 反转   │ 短期反转   │ −(过去 21 个交易日收益)      │ 过去跌得多的更好    │
    │ 低波   │ 低波动     │ −(过去 60 日收益标准差)      │ 波动低的更好        │
    │ 流动性 │ 低换手     │ −(过去 20 日平均换手率)      │ 换手低的更好        │
    │ 规模   │ 小市值     │ −ln(流通市值)                │ 历史上越小越好，    │
    │        │            │                              │ 但阶段性剧烈反转    │
    └────────┴────────────┴──────────────────────────────┴─────────────────────┘

    📌 方向统一：表中带负号的因子已经取反，所有因子都是"值越大，预期收益越高"。
       ⚠️ 方向来自经济逻辑和历史经验，不是真理 —— 规模因子在 A 股就多次"变脸"（11 课）。

    🔗 财务因子（EP、BP、ROE）必须按"公告日"对齐，详见 08 课；本课直接调用对齐好的数据。
    """)


# ==================== 第二部分：股票池与数据 ====================


def universe_overview(panel):
    print("\n" + "=" * 60)
    print("第二部分：股票池与数据")
    print("=" * 60)

    info = panel.info
    last = panel.mcap.iloc[-1] / 1e8
    print(f"""
    股票池：{len(info)} 只（{info['sector'].nunique()} 个行业）
       来源：{'合成数据' if panel.is_synthetic else '沪深 300 + 中证 500 当前成分股，按行业均匀抽样'}
       板块：{info['board'].value_counts().to_dict()}
       指数：{info['index'].value_counts().to_dict()}
       期末流通市值（亿元）：最小 {last.min():.0f}，中位数 {last.median():.0f}，最大 {last.max():.0f}
    """)
    tab = pd.crosstab(info["sector"], info["index"])
    print("   各行业股票数：")
    print("   " + tab.to_string().replace("\n", "\n   "))
    if not panel.is_synthetic:
        print("""
    ⚠️ 这个股票池有两个已知缺陷（课程为了控制下载量而做的取舍）：
       1. 用今天的指数成分股回测 2019–2024 → 幸存者偏差 + 成分股前视（第六部分）
       2. 行业分类是今天的分类，而公司的行业归属会调整 → 应使用历史行业分类
       真实研究应该使用"每个调仓日当时"的全市场股票、当时的行业和成分股。
    """)


# ==================== 第三部分：计算因子 ====================


def compute_factors(panel):
    print("\n" + "=" * 60)
    print("第三部分：在截面面板上计算因子")
    print("=" * 60)
    print("""
    核心代码（factor_tools.build_factors，宽表运算，一行一个因子）：

        px = panel.adj_close.ffill()                     # 后复权价，停牌日沿用最后价格
        "mom_12_1": px.shift(21) / px.shift(252) - 1     # 跳过最近 1 个月
        "rev_1m":   -(px / px.shift(21) - 1)              # 取负号：跌得多 → 值大
        "lowvol":   -ret.rolling(60).std()
        "lowturn":  -turnover.rolling(20).mean()
        "size":     -np.log(float_mcap)
        "ep":       net_profit_ttm / total_mcap          # 财务数据已按公告日对齐
        "bp":       equity / total_mcap
        "roe":      net_profit_ttm / equity（净资产 ≤ 0 的不计算）

    为什么动量要"剔除最近 1 个月"？最近 1 个月的收益在 A 股表现为反转，
    如果混在一起，动量和反转会互相抵消。
    """)

    dates = rebalance_dates(panel)
    raw = build_factors(panel, dates)
    d = dates[-1]
    rows = {}
    for k, v in raw.items():
        x = v.loc[d].replace([np.inf, -np.inf], np.nan)
        rows[FACTOR_LABELS[k]] = {"大类": FACTOR_CATEGORY[k], "覆盖率": x.notna().mean(), "P10": x.quantile(0.1),
                                  "中位数": x.median(), "P90": x.quantile(0.9), "偏度": x.skew()}
    print(f"📋 {d.date()} 截面统计（原始值，未预处理）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))
    print("""
    💡 量纲天差地别：EP 在 0.0x 量级、小市值在 −20 多、换手率在 0.0x —— 这就是 04 课要做标准化的原因；
       偏度大的因子（如换手）有明显的极端值 —— 04 课的去极值。
    """)

    ep = raw["ep"].loc[d]
    top = ep.nlargest(5)
    print("   EP 最高的 5 只股票：" + "，".join(
        f"{panel.info.loc[c, 'name']}({panel.info.loc[c, 'sector']}) {v:.3f}" for c, v in top.items()))
    print("   → 高 EP 往往集中在某几个行业，因子收益可能只是行业押注（04 课中性化）")
    return dates, raw


def ep_vs_pe(panel, dates, raw):
    print("\n   📌 为什么用 EP 而不是 PE？")
    ep = raw["ep"]
    neg = (ep < 0).sum().sum()
    total = ep.notna().sum().sum()
    print(f"   全样本 {total} 个股票-月中，有 {neg} 个净利润 TTM 为负（{neg / total:.1%}）")
    d = dates[-1]
    x = ep.loc[d].dropna().sort_values()
    picks = list(dict.fromkeys([x.index[0], x.index[len(x) // 2], x.index[-1], x[x > 0].index[0]]))
    rows = []
    for c in picks:
        e = x[c]
        rows.append({"股票": panel.info.loc[c, "name"], "EP": e, "PE": 1 / e if e != 0 else np.nan})
    df = pd.DataFrame(rows).set_index("股票")
    df["按 EP 排名（大=便宜）"] = df["EP"].rank(ascending=False).astype(int)
    df["按 PE 排名（小=便宜）"] = df["PE"].rank().astype(int)
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    print("""
   - PE 在利润接近 0 时趋于无穷，利润为负时 PE 为负 —— 亏损股按 PE 排序会被当成"最便宜"
   - EP 是连续的，亏损公司自然排在最后。构建因子时尽量让"分母不会接近零"
    """)


# ==================== 第四部分：可交易性过滤 ====================


def tradability(panel, dates):
    print("\n" + "=" * 60)
    print("第四部分：可交易性过滤")
    print("=" * 60)
    print("""
    时间线（每个调仓日）：
        t 日收盘 ──▶ 用 t 日及之前的数据计算因子、排序、生成目标持仓
        t+1 开盘 ──▶ 下单成交。此时才发现：停牌买不了、开盘涨停买不进
        持有到下一个调仓日 t' 的次日开盘

    ┌──────────────┬──────────────────────────────────────────────┐
    │ 过滤条件     │ 本课实现                                     │
    ├──────────────┼──────────────────────────────────────────────┤
    │ 次新股       │ 上市不满 365 个自然日剔除                    │
    │ 停牌         │ t 日没有成交剔除                             │
    │ ST           │ 没有历史 ST 标记 → 用"恰好收在 ±5% 涨跌停"   │
    │              │ 反推（主板 ST 涨跌幅 2026-07-06 前为 ±5%）   │
    │ 涨停买不进   │ t+1 日停牌，或开盘价 ≥ 涨停价                │
    │              │ （主板 10%，创业板 2020-08-24 起 20%，科创板 20%）│
    └──────────────┴──────────────────────────────────────────────┘
    💡 "t+1 开盘涨停"用到了 t+1 的价格，但它只用来模拟"成交失败"，不参与因子计算和排序，
       所以不是前视偏差。持仓股在调仓日跌停或停牌，应模拟为"卖不出、继续持有"（06 课）。
    """)

    st = tradable_status(panel, dates)
    labels = {"listed": "次新股（上市不满 1 年）", "trading": "t 日停牌", "not_st": "ST（推断）",
              "buyable": "t+1 停牌或开盘涨停", "all": "合计（任一条件）"}
    rows = {labels[k]: {"平均每期剔除": (~v).sum(axis=1).mean(), "最多一期剔除": (~v).sum(axis=1).max(),
                        "剔除的股票-月": int((~v).sum().sum())} for k, v in st.items()}
    print("📋 剔除统计（可能同时满足多个条件，所以合计 ≤ 各项之和）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.1f}"))

    worst = (~st["buyable"]).sum(axis=1)
    d = worst.idxmax()
    nxt = panel.dates[panel.dates.get_loc(d) + 1]
    print(f"\n   买不进最多的调仓日：{d.date()}，次日 {nxt.date()} 有 {worst.max()} 只股票停牌或开盘涨停")
    if not panel.is_synthetic and d == pd.Timestamp("2024-09-30"):
        print("   （2024 年国庆假期前后市场急涨，节后首个交易日 10 月 8 日大量股票开盘即涨停）")

    flags = infer_st(panel)
    hit = flags.sum()
    for c in hit[hit > 0].index:
        days = flags.index[flags[c]]
        print(f"   推断为 ST：{c} {panel.info.loc[c, 'name']}，{days[0].date()} ~ {days[-1].date()}（{len(days)} 个交易日）")
    return st["all"]


# ==================== 第五部分：因子相关性 ====================


def factor_correlation(panel, dates, raw, mask):
    print("\n" + "=" * 60)
    print("第五部分：因子相关性 —— 谁是小市值的马甲？")
    print("=" * 60)

    names = list(raw)
    corr_sum = np.zeros((len(names), len(names)))
    n = 0
    for d in dates:
        df = pd.DataFrame({k: raw[k].loc[d] for k in names})[mask.loc[d]].dropna()
        if len(df) < 30:
            continue
        corr_sum += df.rank().corr().to_numpy()
        n += 1
    corr = pd.DataFrame(corr_sum / n, index=names, columns=names)
    print(f"\n📋 平均截面 Spearman 相关系数（{n} 期）：")
    print(corr.round(2).to_string())
    s = corr["size"].drop("size").sort_values(key=abs, ascending=False)
    print(f"""
    与"小市值"相关性最强的因子：{"，".join(f"{k} {v:+.2f}" for k, v in s.head(3).items())}

    💡 A 股中小市值股票往往同时是高换手、高波动的股票 —— 不做中性化，
       很多"新因子"测到的其实是小市值效应。EP 与 BP 同属价值，相关性也较高，
       合成时可以先合并为"价值"大类（09 课）。
    """)


# ==================== 第六部分：幸存者偏差 ====================


def survivorship(panel, dates):
    print("\n" + "=" * 60)
    print("第六部分：幸存者偏差")
    print("=" * 60)

    ret = period_returns(panel, dates)
    nav = (1 + ret.fillna(0)).prod() - 1
    nav = nav[ret.notna().sum() > 60]
    if panel.is_synthetic:
        print("""
    合成面板没有退市机制，也就没有幸存者偏差，本部分只讲原理。
    联网运行可以看到今天的成分股与真实退市股票的收益对比。
    """)
        df = pd.DataFrame()
    else:
        print(f"""
    本股票池（今天的沪深 300 / 中证 500 成分股）在 {dates[0].date()} ~ {dates[-1].date()} 的总收益：
       中位数 {nav.median():+.0%}，亏损超过 50% 的股票占比 {(nav < -0.5).mean():.1%}
    """)
        df = load_delisted_sample()
        if df.empty:
            print("    （退市股票样本下载失败，已跳过）")
    if not df.empty:
        print(f"""    同期（2019-07 ~ 2024-12）沪深主板退市股票共 {int(df['n_delisted_total'].iloc[0])} 只，
    均匀抽取 {len(df)} 只下载行情，从 2019 年起（或上市首日）到最后一个交易日的价格涨跌幅：
       中位数 {df['total_return'].median():+.0%}，亏损超过 50% 的占比 {(df['total_return'] < -0.5).mean():.0%}
    """)
        show = df.sort_values("delist_date").head(6)
        for _, r in show.iterrows():
            print(f"       {r['code']} {r['name']}：最后交易日 {pd.Timestamp(r['last_date']).date()}，"
                  f"区间涨跌 {r['total_return']:+.0%}")
    print("""
    💡 退市股票永远不会出现在"今天的成分股"里。用今天的股票池回测，
       相当于提前知道了哪些公司能活下来 —— 小市值、低价、高波动类因子尤其受影响，
       因为它们更容易选中"后来退市"的股票。
       正确做法：每个调仓日使用当时上市、当时属于指数的全部股票（含后来退市的），
       退市股票按退市前最后价格（或退市整理期价格）清算，损失必须计入。
    """)


# ==================== 主函数 ====================


def main():
    panel = load_panel()
    factor_zoo()
    universe_overview(panel)
    dates, raw = compute_factors(panel)
    ep_vs_pe(panel, dates, raw)
    mask = tradability(panel, dates)
    factor_correlation(panel, dates, raw, mask)
    survivorship(panel, dates)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 增加两个因子：SP（营业收入 TTM / 总市值）和 Amihud 非流动性（|日收益| / 成交额 的 20 日均值），
       计算它们与已有因子的相关性（营业收入在 panel.fin["revenue"]，同样是累计值口径）
    2. 把动量改为 6-1、3-1 个月，比较与 12-1 动量、短期反转的相关性
    3. 修改 tradable_status：把次新股门槛从 365 天改为 180 天，统计每期多出多少只股票
    4. 思考：如果 t 日停牌的股票在 t+1 复牌并涨停，你的回测会怎么处理？应该怎么处理？
    """)


if __name__ == "__main__":
    main()
