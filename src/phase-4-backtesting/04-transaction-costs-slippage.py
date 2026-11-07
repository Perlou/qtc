"""
交易成本与滑点建模
==================

学习目标：
    1. 掌握 A 股交易成本的构成：佣金（最低 5 元）、印花税（仅卖出）、过户费
    2. 认识"最低佣金陷阱"：小额交易的实际费率远高于名义费率
    3. 理解滑点与冲击成本模型：固定比例、成交量参与率、平方根冲击模型
    4. 计算换手率、成本拖累与盈亏平衡成本
    5. 完成成本敏感性分析 —— 回测报告的必备内容

核心概念：
    - 年化成本拖累 ≈ 年化换手率 × 单边平均成本率
    - 盈亏平衡成本 c* = 扣成本前年化收益 / 年化换手率
    - 平方根冲击：Impact ≈ k·σ·√(Q/V)

前置知识：
    - 03-performance-metrics.py
    - Phase 1 02-a-share-market-rules.py（费率规则）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）

⚠️ 费率以编写时（2026 年）的公开规则为示例，佣金与券商协商确定，实盘前请核对最新规定。
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import CostModel, cost_summary, performance_report, stamp_duty_rate_at, vectorized_backtest
from utils.data import get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
COSTS_BP = [0, 5, 10, 15, 20, 30, 50]
SYMBOL_STRAT = "600519"  # 成本敏感性用的标的
SYMBOL_IMPACT = "601088"  # 冲击成本 / 精确成本用的标的（股价较低，便于观察小账户）


# ==================== 第一部分：A 股成本构成 ====================


def a_share_cost(value: float, is_sell: bool, comm_rate=0.00025, min_comm=5.0,
                 stamp_rate=0.0005, transfer_rate=0.00001) -> float:
    """A 股单笔交易成本（元）"""
    commission = max(value * comm_rate, min_comm)
    transfer = value * transfer_rate
    stamp = value * stamp_rate if is_sell else 0.0
    return commission + transfer + stamp


def cost_components():
    print("=" * 60)
    print("第一部分：A 股股票交易成本构成")
    print("=" * 60)

    print("""
    🚗 比喻：交易成本是"过路费"。偶尔开一次车无所谓；每天往返几十次（高换手），过路费比油钱还贵。

    ┌──────────┬──────────┬────────────────────────────────────────────┐
    │ 成本项   │ 方向     │ 量级（示例值，以最新规则为准）             │
    ├──────────┼──────────┼────────────────────────────────────────────┤
    │ 佣金     │ 买 + 卖  │ 万 1 ~ 万 3，单笔最低 5 元（部分券商免五） │
    │ 印花税   │ 仅卖出   │ 0.05%（2023-08-28 起；此前为 0.1%）        │
    │ 过户费   │ 买 + 卖  │ 0.001%                                     │
    │ 滑点     │ 隐性     │ 取决于流动性、订单大小、执行方式           │
    │ 冲击成本 │ 隐性     │ 大单推动价格向不利方向移动                 │
    └──────────┴──────────┴────────────────────────────────────────────┘
    """)

    model = CostModel()
    rows = []
    for value in [50_000, 200_000]:
        for side, is_sell in [("买入", False), ("卖出", True)]:
            hand = a_share_cost(value, is_sell)
            f = model.fees(value, is_sell)
            rows.append([f"{value:,}", side, round(f["commission"], 2), round(f["stamp_duty"], 2),
                         round(f["transfer_fee"], 2), round(hand, 2), f"{hand / value * 1e4:.2f}",
                         "✅" if abs(hand - f["total"]) < 1e-9 else "❌"])
    print(pd.DataFrame(rows, columns=["成交金额", "方向", "佣金", "印花税", "过户费", "合计", "费率(bp)",
                                      "与bt_core一致"]).to_string(index=False))

    print("\n📅 跨年回测要按'当时'的费率：bt_core.stamp_duty_rate_at(date)")
    for d in ["2020-06-01", "2023-08-25", "2023-08-28", "2025-01-02"]:
        print(f"   {d} 卖出印花税 = {stamp_duty_rate_at(d):.2%}")


# ==================== 第二部分：最低佣金陷阱 ====================


def min_commission_trap():
    print("\n" + "=" * 60)
    print("第二部分：最低佣金陷阱")
    print("=" * 60)

    rows = []
    for value in [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 1_000_000]:
        buy = a_share_cost(value, False)
        sell = a_share_cost(value, True)
        rows.append([f"{value:,}", f"{buy / value * 1e4:.1f}", f"{sell / value * 1e4:.1f}",
                     f"{(buy + sell) / value * 1e4:.1f}"])
    print(pd.DataFrame(rows, columns=["单笔金额", "买入(bp)", "卖出(bp)", "买卖一轮(bp)"]).to_string(index=False))
    print(f"\n   佣金 万2.5 时，单笔金额低于 5 / 0.00025 = {5 / 0.00025:,.0f} 元，就会触发最低 5 元")

    print("\n📋 10 万元账户等权持有 N 只股票，每只每次调仓的单边实际费率：")
    rows = []
    for n in [1, 5, 10, 20, 50]:
        value = 100_000 / n
        rows.append([n, f"{value:,.0f}", f"{a_share_cost(value, False) / value * 1e4:.1f}",
                     f"{a_share_cost(value, True) / value * 1e4:.1f}"])
    print(pd.DataFrame(rows, columns=["持股数", "每只金额", "买入(bp)", "卖出(bp)"]).to_string(index=False))
    print("   💡 小资金 + 多股票 → 最低佣金让费率成倍上升，小资金回测必须建模这一点")


# ==================== 第三部分：滑点与冲击成本 ====================


def slippage_models(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：滑点与冲击成本模型")
    print("=" * 60)

    print("""
    ┌────────────────────┬──────────────────────────────────┬──────────────────┐
    │ 模型               │ 公式                             │ 适用             │
    ├────────────────────┼──────────────────────────────────┼──────────────────┤
    │ 固定比例           │ 成交价 = 参考价 × (1 ± s)        │ 简单、保守       │
    │ 半价差             │ 成交价 = 中间价 ± 价差/2         │ 有盘口数据时     │
    │ 成交量参与率限制   │ 单日成交 ≤ 当日成交量 × p%       │ 防止无限流动性   │
    │ 平方根冲击         │ Impact ≈ k·σ·√(Q/V)              │ 大资金（Phase 11）│
    └────────────────────┴──────────────────────────────────┴──────────────────┘
    σ：日波动率；Q：订单量；V：日成交量；k：经验系数，需用真实成交数据校准
    """)

    sigma = df["close"].pct_change().tail(60).std()
    adv = df["amount"].tail(60).mean()  # 近 60 日日均成交额（元）
    print(f"📊 {SYMBOL_IMPACT}：近 60 日日波动 σ = {sigma:.2%}，日均成交额 = {adv / 1e8:.2f} 亿元")
    k = 1.0  # 示意值，真实系数需校准
    rows = []
    for order in [1e5, 1e6, 1e7, 1e8, 5e8]:
        part = order / adv
        impact = k * sigma * np.sqrt(part)
        rows.append([f"{order / 1e4:,.0f} 万", f"{part:.3%}", f"{impact * 1e4:.1f}",
                     "⚠️ 超过 10% 参与率" if part > 0.10 else ""])
    print(pd.DataFrame(rows, columns=["订单金额", "参与率 Q/V", "冲击(bp, k=1)", "备注"]).to_string(index=False))
    print("""
    💡 冲击随订单规模"开根号"增长：订单翻 4 倍，单位冲击约翻 2 倍
    💡 回测中至少要加一条成交量约束（如单日成交 ≤ 成交量 × 10%），否则大资金回测会"无限流动性"
    """)



# ==================== 第四部分：换手、成本拖累与盈亏平衡成本 ====================


def make_strategies(df: pd.DataFrame) -> dict:
    close = df["adj_close"]
    ma = (close > close.rolling(20).mean()).astype(float)  # 低换手：均线择时
    rev = (close.pct_change() < 0).astype(float)  # 高换手：今天跌了就买入，持有 1 天（短期反转）
    return {"MA20 择时": ma, "1日反转": rev}


def turnover_and_breakeven(df: pd.DataFrame, strategies: dict):
    print("\n" + "=" * 60)
    print(f"第四部分：换手、成本拖累与盈亏平衡成本（{SYMBOL_STRAT}，收盘成交）")
    print("=" * 60)

    print("""
    年化成本拖累 ≈ 年化换手 × 单边成本
    盈亏平衡成本 c* = 扣成本前年化收益 / 年化换手   ← c* 只比真实成本高一点点 = 没有安全边际
    """)
    close = df["adj_close"]
    rows = []
    for name, sig in strategies.items():
        bt = vectorized_backtest(close, sig, 0.001)
        cs = cost_summary(bt)
        gross_annual = bt["gross"].mean() * 252
        rows.append([name, f"{gross_annual:+.2%}", f"{cs['年化换手']:.0f}", f"{cs['年化成本拖累']:.2%}",
                     f"{cs['盈亏平衡成本(bp)']:.1f}" if gross_annual > 0 else "扣费前已亏损"])
    print(pd.DataFrame(rows, columns=["策略", "扣费前算术年化", "年化换手(倍)", "成本拖累(10bp)",
                                      "盈亏平衡成本(bp)"]).to_string(index=False))



# ==================== 第五部分：成本敏感性分析 ====================


def cost_sensitivity(df: pd.DataFrame, strategies: dict):
    print("\n" + "=" * 60)
    print("第五部分：成本敏感性分析（回测报告必备）")
    print("=" * 60)

    m = CostModel()
    realistic = m.commission_rate + m.transfer_fee_rate + m.stamp_duty_rate / 2 + m.slippage
    print(f"""
    先估计"真实"单边成本（大额交易，未触发最低佣金）：
      佣金 {m.commission_rate * 1e4:.1f} bp + 过户费 {m.transfer_fee_rate * 1e4:.1f} bp
      + 印花税摊到单边 {m.stamp_duty_rate / 2 * 1e4:.1f} bp + 滑点 {m.slippage * 1e4:.1f} bp
      ≈ {realistic * 1e4:.1f} bp
    """)

    close = df["adj_close"]
    results = {}
    for name, sig in strategies.items():
        rows = []
        for c in COSTS_BP:
            rep = performance_report(vectorized_backtest(close, sig, c / 1e4)["ret"])
            rows.append({"单边成本(bp)": c, "年化收益": rep["年化收益"], "Sharpe": rep["Sharpe"]})
        results[name] = pd.DataFrame(rows).set_index("单边成本(bp)")

    table = pd.concat({k: v for k, v in results.items()}, axis=1)
    fmt = table.copy().astype(object)
    for col in table.columns:
        fmt[col] = table[col].map(lambda x: f"{x:+.2%}" if col[1] == "年化收益" else f"{x:.2f}")
    print(fmt.to_string())
    print("""
    💡 解读：
    - 曲线越平缓，策略对成本越不敏感（稳健）；成本稍高就转负 → 脆弱
    - 报告中必须标出"你认为的真实成本"在曲线上的位置
    """)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res in results.items():
        ax.plot(res.index, res["Sharpe"], marker="o", label=name)
    ax.axvline(realistic * 1e4, color="tab:red", ls="--", label=f"估计真实成本 ≈ {realistic * 1e4:.1f} bp")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("单边成本 (bp)")
    ax.set_ylabel("Sharpe")
    ax.set_title(f"{SYMBOL_STRAT} 成本敏感性：Sharpe vs 单边成本")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '04_cost_sensitivity.png', subdir='phase-4')}")


# ==================== 第六部分：精确成本 vs 比例近似 ====================


def precise_cost_nav(df: pd.DataFrame, signal: pd.Series, capital: float, model: CostModel,
                     historical_stamp: bool = False) -> pd.Series:
    """
    次日开盘成交的逐日记账：每次调仓按真实金额计算佣金（最低 5 元）、印花税（仅卖出）、过户费、滑点。
    （为突出成本，本函数不做整数手取整，整数手见 05、06 课）
    """
    close, open_ = df["adj_close"], df["adj_open"]
    w_new = signal.shift(1).fillna(0.0).to_numpy()
    w_old = signal.shift(2).fillna(0.0).to_numpy()
    r_on = (open_ / close.shift(1) - 1).fillna(0.0).to_numpy()
    r_id = (close / open_ - 1).fillna(0.0).to_numpy()
    equity, out = capital, []
    for i, d in enumerate(df.index):
        equity *= 1 + w_old[i] * r_on[i]
        w_drift = w_old[i] * (1 + r_on[i]) / (1 + w_old[i] * r_on[i])
        traded = equity * abs(w_new[i] - w_drift)
        if traded > 0:
            is_sell = w_new[i] < w_drift
            m = replace(model, stamp_duty_rate=stamp_duty_rate_at(d)) if historical_stamp else model
            equity -= m.total(traded, is_sell) + traded * m.slippage
        equity *= 1 + w_new[i] * r_id[i]
        out.append(equity)
    return pd.Series(out, index=df.index)


def precise_vs_proportional(df: pd.DataFrame, signal: pd.Series):
    print("\n" + "=" * 60)
    print(f"第六部分：精确成本 vs 比例近似（{SYMBOL_IMPACT} MA20，次日开盘成交）")
    print("=" * 60)

    model = CostModel()
    rows = []
    for capital in [20_000, 100_000, 1_000_000, 10_000_000]:
        nav = precise_cost_nav(df, signal, capital, model)
        ret = nav / nav.shift(1, fill_value=capital) - 1
        rows.append([f"{capital:,}", f"{performance_report(ret)['年化收益']:+.2%}"])
    prop = vectorized_backtest(df["adj_close"], signal, 0.0010, open_=df["adj_open"], execution="next_open")
    print(pd.DataFrame(rows, columns=["账户资金", "精确成本下年化收益"]).to_string(index=False))
    print(f"   比例近似（单边 10 bp）年化收益：{performance_report(prop['ret'])['年化收益']:+.2%}")
    print("   💡 资金越小，最低 5 元佣金占比越高；比例近似会低估小账户的成本")

    nav_now = precise_cost_nav(df, signal, 1_000_000, model)
    nav_hist = precise_cost_nav(df, signal, 1_000_000, model, historical_stamp=True)
    a_now = nav_now.iloc[-1] / 1e6 - 1
    a_hist = nav_hist.iloc[-1] / 1e6 - 1
    print(f"\n📅 按现行印花税 0.05% 计算全区间总收益 {a_now:+.2%}；按历史费率（2023-08-28 前 0.1%）{a_hist:+.2%}")
    print("   跨期费率不变 → 低估早期成本（常见陷阱 #13）")


# ==================== 主函数 ====================


def main():
    cost_components()
    min_commission_trap()

    shenhua = get_stock_daily_adj(SYMBOL_IMPACT, START, END)
    shenhua["adj_open"] = shenhua["open"] * shenhua["adj_factor"]
    slippage_models(shenhua)

    maotai = get_stock_daily_adj(SYMBOL_STRAT, START, END)
    strategies = make_strategies(maotai)
    turnover_and_breakeven(maotai, strategies)
    cost_sensitivity(maotai, strategies)

    sig = (shenhua["adj_close"] > shenhua["adj_close"].rolling(20).mean()).astype(float)
    precise_vs_proportional(shenhua, sig)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 你的券商是"万 1 免五"，重新计算第二部分的两张表，并重画第五部分的真实成本线
    2. 在 precise_cost_nav 中加入成交量约束：单日成交金额不超过当日成交额的 5%，
       超出部分顺延到下一日。1 亿元资金时，结果会怎样变化？
    3. 用 1 日反转策略找到它的盈亏平衡成本，并解释为什么高换手策略几乎都死在成本上
    4. 把滑点改为"与当日振幅 (high − low)/close 成正比"，比较与固定 5 bp 滑点的差异
    """)


if __name__ == "__main__":
    main()
