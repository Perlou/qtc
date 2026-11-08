"""
财报 Point-in-Time 对齐
=======================

学习目标：
    1. 区分报告期与公告日，理解按报告期对齐为什么是未来函数
    2. 掌握累计值 → 单季度 → TTM 的换算
    3. 用 merge_asof 按公告日对齐，处理"公告当日不用""报告期倒退"两个细节
    4. 用真实数据量化：按报告期对齐与按公告日对齐的 IC 差异

核心概念：
    - 报告期（end_date）：财报描述的期间；公告日（ann_date）：财报实际公开披露的日期
    - PIT（Point-in-Time）：回测的每一天，只使用当天真实可见的数据
    - TTM = 本期累计 + 上年年报 − 上年同期累计
    - 法定披露截止日：一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30

前置知识：
    - 03-factor-construction.py、05-ic-ir-analysis.py
    - Phase 1 的 10-data-biases-quality.py（前视偏差）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from factor_data import load_panel, report_deadline
from factor_tools import (add_fin_derived, calc_ic, ic_summary, period_returns, pit_merge, preprocess,
                          rebalance_dates, tradable_mask)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

REPORT_NAME = {3: "一季报", 6: "半年报", 9: "三季报", 12: "年报"}


# ==================== 第一部分：报告期 vs 公告日 ====================


def timeline():
    print("=" * 60)
    print("第一部分：报告期 vs 公告日")
    print("=" * 60)
    print("""
     2023-12-31          2024-03-28           2024-04-30
         │                   │                    │
      报告期结束          A 公司发布年报        年报披露截止日
      （数据还不存在）     （从此刻起才可用）

      ❌ 在 2024-01-01 就使用 2023 年报（按报告期对齐）→ 提前近 3 个月"知道"全年利润
      ✅ 从公告日之后的第一个交易日起才使用

    ┌────────┬────────────┬────────────┐
    │ 报告   │ 报告期截止 │ 披露截止   │
    ├────────┼────────────┼────────────┤
    │ 一季报 │ 3 月 31 日 │ 4 月 30 日 │
    │ 半年报 │ 6 月 30 日 │ 8 月 31 日 │
    │ 三季报 │ 9 月 30 日 │ 10 月 31 日│
    │ 年报   │ 12 月 31 日│ 次年 4/30  │
    └────────┴────────────┴────────────┘
    年报和一季报的截止日都是 4 月 30 日，很多公司同一天发布两份报告。
    """)


# ==================== 第二部分：真实的披露时滞 ====================


def disclosure_lag(panel):
    print("\n" + "=" * 60)
    print("第二部分：真实的披露时滞与数据陷阱")
    print("=" * 60)

    fin = panel.fin.copy()
    fin["lag"] = (fin["ann_date"] - fin["end_date"]).dt.days
    fin["类型"] = fin["end_date"].dt.month.map(REPORT_NAME)
    fin["last_week"] = (report_deadline(fin["end_date"]) - fin["ann_date"]).dt.days.between(0, 6)
    tab = fin.groupby("类型").agg(样本数=("lag", "size"), 中位数=("lag", "median"), P90=("lag", lambda s: s.quantile(0.9)),
                                最大=("lag", "max"), 截止前一周内占比=("last_week", "mean"))
    print("📋 公告日相对报告期的时滞（天）：")
    print(tab.reindex(list(REPORT_NAME.values())).to_string(float_format=lambda v: f"{v:.2f}"))
    n_fill = (fin["ann_source"] != "实际披露").sum()
    print(f"\n   公告日来源：{(fin['ann_source'] == '实际披露').mean():.1%} 为实际披露时间，"
          f"{n_fill} 条缺失、按法定截止日保守补齐")

    gap = (fin["latest_ann_date"] - fin["ann_date"]).dt.days
    print(f"""
    ⚠️ 陷阱：东方财富业绩报表 stock_yjbb_em 的"最新公告日期"不是首次公告日！
       它与实际披露时间不同的记录占 {(gap != 0).mean():.0%}，差距中位数 {gap[gap != 0].median():.0f} 天，
       通常是下一年同期报告披露时被更新。本阶段用 stock_yysj_em 的"实际披露时间"作为公告日。
    """)
    c = fin["code"].iloc[0]
    show = fin[fin["code"] == c].tail(5)[["end_date", "ann_date", "latest_ann_date"]].copy()
    show.columns = ["报告期", "实际披露时间（yysj）", "最新公告日期（yjbb）"]
    print(f"   示例 {c} {panel.info.loc[c, 'name']}：")
    print("   " + show.to_string(index=False).replace("\n", "\n   "))
    return fin


# ==================== 第三部分：累计值、单季度与 TTM ====================


def ttm_demo(panel, fin_d):
    print("\n" + "=" * 60)
    print("第三部分：累计值、单季度与 TTM")
    print("=" * 60)
    print("""
    A 股利润表是"年初至今累计值"：一季报 1–3 月，半年报 1–6 月，三季报 1–9 月，年报全年。
       单季度：Q3 单季 = 三季报累计 − 半年报累计
       TTM：   本期累计 + 上年年报 − 上年同期累计
    资产负债表（净资产）是时点值，不需要 TTM。
    """)
    counts = fin_d.dropna(subset=["net_profit_ttm"]).groupby("code").size()
    c = counts.idxmax()
    df = fin_d[(fin_d["code"] == c) & (fin_d["end_date"] >= "2022-03-31") & (fin_d["end_date"] <= "2023-12-31")]
    show = df[["end_date", "net_profit", "net_profit_q", "net_profit_ttm"]].copy()
    show[["net_profit", "net_profit_q", "net_profit_ttm"]] /= 1e8
    show.columns = ["报告期", "累计净利润(亿)", "单季度(亿)", "TTM(亿)"]
    print(f"📋 {c} {panel.info.loc[c, 'name']}：")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    row = df.set_index("end_date")
    a, b, y = pd.Timestamp("2023-03-31"), pd.Timestamp("2022-12-31"), pd.Timestamp("2022-03-31")
    if {a, b, y} <= set(row.index):
        calc = row.loc[a, "net_profit"] + row.loc[b, "net_profit"] - row.loc[y, "net_profit"]
        print(f"\n   验算 2023Q1 TTM = 2023Q1 累计 + 2022 年报 − 2022Q1 累计 = {calc / 1e8:.2f} 亿")
    print("   ❌ 常见错误：把四个报告期的累计值直接相加 —— 累计值之间有重复计算")


# ==================== 第四部分：对齐方式与 IC ====================


def aligned_panels(panel, dates, how: str, fin_d: pd.DataFrame) -> dict:
    pit = pit_merge(panel, dates, ["net_profit_ttm", "np_q_yoy"], how=how, fin=fin_d)
    tmc = panel.total_mcap.reindex(dates)
    age = pit["end_date"].apply(lambda col: (pd.Series(dates, index=dates) - col).dt.days)
    return {"ep": pit["net_profit_ttm"] / tmc, "growth": pit["np_q_yoy"], "end_date": pit["end_date"], "age": age}


def alignment_comparison(panel, fin_d):
    print("\n" + "=" * 60)
    print("第四部分：不同对齐方式的 IC 差异")
    print("=" * 60)

    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    ret = period_returns(panel, dates)
    methods = {
        "❌ 按报告期": ("report", fin_d),
        "✅ 按实际公告日": ("ann", fin_d),
        "🟡 按法定截止日": ("deadline", fin_d),
        "⚠️ 按 yjbb 最新公告日期": ("ann", fin_d.assign(ann_date=fin_d["latest_ann_date"])),
    }
    rows, ics, panels = {}, {}, {}
    for name, (how, f) in methods.items():
        p = aligned_panels(panel, dates, how, f)
        panels[name] = p
        ep = preprocess(p["ep"], panel, mask)
        gr = preprocess(p["growth"], panel, mask)
        s_ep, s_gr = ic_summary(calc_ic(ep, ret)), ic_summary(calc_ic(gr, ret))
        ics[name] = calc_ic(gr, ret)
        rows[name] = {"数据平均年龄(天)": p["age"].stack().mean(), "EP Rank IC": s_ep["IC均值"], "EP t值": s_ep["t值"],
                      "增速 Rank IC": s_gr["IC均值"], "增速 ICIR": s_gr["ICIR"], "增速 t值": s_gr["t值"]}
    df = pd.DataFrame(rows).T
    print("📋 同样的因子，不同的对齐方式（增速 = 单季度净利润同比，行业市值中性化后）：")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))

    wrong, right = panels["❌ 按报告期"]["end_date"], panels["✅ 按实际公告日"]["end_date"]
    ahead = (wrong > right).sum().sum() / right.notna().sum().sum()
    inflate = df.loc["❌ 按报告期", "增速 Rank IC"] - df.loc["✅ 按实际公告日", "增速 Rank IC"]
    inflate_ep = df.loc["❌ 按报告期", "EP Rank IC"] - df.loc["✅ 按实际公告日", "EP Rank IC"]
    diff = (ics["❌ 按报告期"] - ics["✅ 按实际公告日"]).dropna()
    t_diff = diff.mean() / diff.std() * np.sqrt(len(diff))
    verdict = ("差异超出了噪声范围：虚高的 IC 来自未来函数，不是预测能力。" if t_diff > 2 else
               "本样本中这个差异还在噪声范围内，但这不说明按报告期对齐是安全的：它仍然用了当时不存在的数据。")
    print(f"""
    🔍 按报告期对齐时，{ahead:.0%} 的股票-月用到了当时还没有公布的财报。
       增速因子的 Rank IC 因此变化 {inflate:+.4f}（逐期配对差的 t 值 {t_diff:+.2f}），EP 变化 {inflate_ep:+.4f}。
       {verdict}
       机制：财报公布时股价才反应业绩，按报告期对齐等于"公告前就知道了业绩"，
       在财报发布前买入业绩好的公司，当然"预测"得准 —— 这部分 IC 实盘拿不到。

    💡 读表要点：
       - 增速因子的信息集中在"最新一期财报"，对未来函数更敏感
       - EP 的 TTM 每次只更新四分之一，变化慢，受影响通常更小 —— 未来函数不一定一眼能看出来
       - 按法定截止日对齐是没有公告日数据时的保守替代：数据更旧，但不会偷看未来
       - 用 yjbb 的"最新公告日期"对齐不是未来函数，但数据被大幅延后，白白损失信息
    """)

    by_month = pd.DataFrame({k: v.groupby(v.index.month).mean() for k, v in ics.items()
                             if k in ("❌ 按报告期", "✅ 按实际公告日")})
    by_month.index.name = "调仓月份"
    print("📋 增速因子 Rank IC，按调仓月份平均：")
    print(by_month.T.to_string(float_format=lambda v: f"{v:+.3f}"))
    same = by_month.index[(by_month.iloc[:, 0] - by_month.iloc[:, 1]).abs() < 1e-9].tolist()
    print(f"""
    💡 两行完全相同的月份：{same if same else "无"} —— 此时上一期财报都已公布，两种对齐取到的数据一样。
       差异全部来自"报告期已结束、财报还没公布"的窗口：未来函数只在这些月份起作用。
    """)
    return df


# ==================== 第五部分：merge_asof 的细节 ====================


def merge_asof_details():
    print("\n" + "=" * 60)
    print("第五部分：merge_asof 的三个细节")
    print("=" * 60)

    daily = pd.DataFrame({"trade_date": pd.to_datetime(["2024-04-29", "2024-04-30", "2024-05-06", "2024-05-13"]),
                          "stock": "A"})
    fin = pd.DataFrame({
        "stock": "A",
        "ann_date": pd.to_datetime(["2024-04-30", "2024-04-30", "2024-05-10"]),
        "end_date": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
        "eps": [1.00, 0.30, 0.95],
        "备注": ["2023 年报", "2024 一季报", "年报更正（旧报告期晚到）"],
    }).sort_values(["ann_date", "end_date"])

    def asof(f, exact):
        m = pd.merge_asof(daily, f, left_on="trade_date", right_on="ann_date", by="stock",
                          direction="backward", allow_exact_matches=exact)
        return m[["trade_date", "end_date", "eps", "备注"]]

    print("\n📋 细节 1：allow_exact_matches=True（公告当日就用）")
    print(asof(fin, True).to_string(index=False))
    print("   （NaT / NaN 表示这一天还没有任何已公布的财报可用 —— 这正是 PIT 要的结果）")
    print("\n📋 allow_exact_matches=False（公告当日不用；A 股公告常在盘后或晚间发布）")
    print(asof(fin, False).to_string(index=False))

    clean = fin[fin["end_date"] >= fin.groupby("stock")["end_date"].cummax()]
    print("\n📋 细节 2：5 月 10 日的年报更正把一季报覆盖了；剔除报告期倒退的记录后：")
    print(asof(clean, False).to_string(index=False))
    print("""
    细节 2 的处理：按公告日排序后，只保留"报告期不倒退"的记录（end_date ≥ 累计最大值）。
       简化取舍：这样也丢弃了更正内容本身，严格做法是更新对应报告期后重算 TTM。

    细节 3：财报更正与"原始版本"
       数据库常提供最新修订版。回测中直接使用，等于过去就知道了后来才更正的数字。
       ⚠️ 本课的 yjbb 数值可能是修订后的版本（免费数据没有版本历史），
          严格研究应使用"首次披露版本 + 每次更正的公告日"。

    其他需要 PIT 的数据：
       - 行业分类：公司行业归属会调整（本课用的是当前分类）
       - 指数成分股：用当时的成分（本课用的是当前成分）
       - 股票池：必须含已退市股票（03 课）
       - ST 状态：用当时的状态（本课用涨跌停幅度反推）
       - 业绩预告 / 快报：早于正式财报，同样按其公告日对齐
    """)


# ==================== 主函数 ====================


def plot_results(fin: pd.DataFrame, df: pd.DataFrame):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in REPORT_NAME.values():
        axes[0].hist(fin.loc[fin["类型"] == name, "lag"], bins=30, alpha=0.6, label=name)
    axes[0].set_xlabel("公告日 − 报告期（天）")
    axes[0].set_title("财报披露时滞分布")
    axes[0].legend()
    axes[1].barh([s[2:] for s in df.index], df["增速 Rank IC"], color="tab:blue")
    axes[1].axvline(0, color="gray", lw=0.6)
    axes[1].set_title("增速因子 Rank IC：不同对齐方式")
    path = save_figure(fig, "08_point_in_time.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    fin_d = add_fin_derived(panel.fin)
    timeline()
    fin = disclosure_lag(panel)
    ttm_demo(panel, fin_d)
    df = alignment_comparison(panel, fin_d)
    merge_asof_details()
    plot_results(fin, df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 ak.stock_profit_sheet_by_report_em("SH600519") 取 NOTICE_DATE，核对它与
       stock_yysj_em 的实际披露时间是否一致，再看 UPDATE_DATE 与 yjbb"最新公告日期"的关系
    2. 把 allow_exact_matches 改为 True，比较增速因子的 IC 变化；思考为什么这种做法不够保守
    3. 用 ak.stock_yjkb_em（业绩快报）或 stock_yjyg_em（业绩预告）补充更早的盈利信息，
       同样按其公告日对齐，观察增速因子的 IC 是否提升
    4. 实现 SUE（标准化预期外盈余）：(本季 − 去年同季) / 过去 8 季差值的标准差，比较与增速因子的 IC
    """)


if __name__ == "__main__":
    main()
