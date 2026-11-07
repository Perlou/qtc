"""
向量化回测实现
==============

学习目标：
    1. 理解向量化回测"把整段历史当矩阵一次算完"的思想
    2. 彻底搞懂 shift 的含义：信号在 t 日收盘产生，收益从哪一刻开始算
    3. 实现两种成交价假设：t 日收盘成交、t+1 日开盘成交，并比较结果
    4. 实现多资产截面动量组合的向量化回测，计算换手与成本
    5. 知道向量化回测的盲区（停牌、涨跌停、整数手、T+1）

核心概念：
    - 仓位滞后：w_t = f(信号_{t-1})，R_strat_t = w_t·R_t − c·|w_t − w_{t−1}|
    - 复权与撮合分离：信号和收益用等比后复权价 adj_close；撮合（06 课）用不复权价
    - 换手（Turnover）：Σ|Δw|，决定成本拖累

前置知识：
    - 01-backtest-overview.py
    - Phase 1 06-adjusted-prices.py（等比复权）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import performance_report, report_table, vectorized_backtest as core_vectorized_backtest
from utils.data import get_close_panel, get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"
UNIVERSE = ["600519", "601088", "000002", "600036", "601318", "000858", "600900", "000333"]
COST = 0.001  # 单边成本率（粗略）


# ==================== 第一部分：向量化的思想 ====================


def vectorization_idea():
    print("=" * 60)
    print("第一部分：向量化的思想")
    print("=" * 60)

    print("""
    向量化回测把整个历史当作矩阵，一次性算出所有日期的信号、仓位、收益：

             close   signal   pos=signal.shift(1)   ret      pos × ret
    t-1      100       1            ?                ...        ...
    t        102       0            1               +2%        +2%   ← 昨天收盘决定今天持仓
    t+1      101       0            0               -1%         0
    t+2      103       1            0               +2%         0
    t+3      104       1            1               +1%        +1%

    核心公式：
        w_t        = f(信号_{t-1})                    # 仓位必须滞后
        R_strat_t  = w_t × R_t − c × |w_t − w_{t−1}|  # 收益 − 换手 × 成本

    🎬 比喻：向量化像"看整部电影的剧本"。剧本就摊在桌上 —— 一个 shift 写错，就偷看了结局。
    """)


# ==================== 第二部分：手写向量化回测 ====================


def vectorized_backtest(close: pd.Series, signal: pd.Series, cost_rate: float = 0.0) -> pd.DataFrame:
    """
    最小可用的向量化回测（收盘价成交假设）

    signal: t 日收盘后算出的目标仓位（0/1 或 0~1）
    假设：在 t 日收盘附近（收盘集合竞价）以收盘价成交，从 t+1 日起承担收益
    """
    ret = close.pct_change().fillna(0.0)
    pos = signal.shift(1).fillna(0.0)  # 关键：信号滞后一期
    turnover = pos.diff().abs().fillna(pos.abs())  # 首日建仓也算换手
    gross = pos * ret
    cost = turnover * cost_rate
    net = gross - cost
    return pd.DataFrame({"pos": pos, "gross": gross, "cost": cost, "ret": net,
                         "nav": (1 + net).cumprod(), "turnover": turnover})


def handwritten_backtest(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"第二部分：手写向量化回测（{SYMBOL} {NAME} MA20 择时）")
    print("=" * 60)

    close = df["adj_close"]  # 信号与收益：等比后复权价
    signal = (close > close.rolling(20).mean()).astype(float)
    bt = vectorized_backtest(close, signal, COST)

    print("📋 换仓前后几天的明细（cost 出现在 pos 变化的那一天）：")
    first_flip = bt.index[bt["turnover"] > 0][1]
    i = bt.index.get_loc(first_flip)
    print(bt.iloc[i - 2 : i + 3].round(5).to_string())

    core = core_vectorized_backtest(close, signal, COST, execution="close")
    diff = (bt["nav"] - core["nav"]).abs().max()
    print(f"\n✅ 与 bt_core.vectorized_backtest 对拍：净值最大差异 {diff:.2e}（后续课程直接复用 bt_core）")

    # 为什么要用复权价：不复权价在除息日"假跳水"
    raw_bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    adj_bh = df["adj_close"].iloc[-1] / df["adj_close"].iloc[0] - 1
    print(f"""
    📌 复权与撮合分离：
       买入持有总收益：不复权 {raw_bh:+.2%}，等比后复权 {adj_bh:+.2%}
       → 用不复权价算收益，会把每次分红都当成亏损；
       → 但撮合（涨跌停价、100 股、成交金额）必须用不复权价（05、06 课）
    """)
    return bt


# ==================== 第三部分：成交价假设 ====================


def execution_assumptions(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：成交价假设——shift(1) 还是 shift(2)？")
    print("=" * 60)

    close, open_ = df["adj_close"], df["adj_open"]
    signal = (close > close.rolling(20).mean()).astype(float)

    print("""
    时间线（t 日收盘出信号）：

      t 日 15:00          t+1 日 09:30                 t+1 日 15:00     t+2 日 09:30
         │ 算出信号          │ 开盘成交                      │                 │
         ├───────────────────┼───────────────────────────────┼─────────────────┤
         │← 隔夜：旧仓位承担 →│←──── 日内：新仓位承担 ──────→│← 隔夜：新仓位 →│

    ┌──────────────────────────┬─────────────────────────────────────────┐
    │ 假设                     │ 实现                                    │
    ├──────────────────────────┼─────────────────────────────────────────┤
    │ A. t 日收盘价成交        │ signal.shift(1) × close.pct_change()    │
    │ B. t+1 日开盘价成交      │ signal.shift(2) × open.pct_change()     │
    │    （开盘到开盘计收益）  │ 持有区间 open(t+1)→open(t+2) 在 t+2 行  │
    │ C. t+1 日开盘价成交      │ 隔夜段用 shift(2)，日内段用 shift(1)    │
    │    （按收盘估值，06 课   │ 与事件驱动引擎逐日记账的方式完全一致    │
    │     引擎用这种）         │                                         │
    └──────────────────────────┴─────────────────────────────────────────┘
    B 与 C 是同一笔交易，只是"在开盘还是收盘给净值拍照"，整体收益几乎相同。
    """)

    a = core_vectorized_backtest(close, signal, COST, execution="close")
    open_ret = open_.pct_change().fillna(0.0)
    pos_b = signal.shift(2).fillna(0.0)
    b_ret = pos_b * open_ret - pos_b.diff().abs().fillna(pos_b.abs()) * COST
    c = core_vectorized_backtest(close, signal, COST, open_=open_, execution="next_open")
    bh = close.pct_change().fillna(0.0)

    rets = {"A 收盘成交": a["ret"], "B 次日开盘(开盘估值)": b_ret, "C 次日开盘(收盘估值)": c["ret"], "买入持有": bh}
    table = report_table(rets).loc[["年化收益", "年化波动", "Sharpe", "最大回撤"]]
    print(table.to_string())

    print(f"""
    💡 B 与 C 的总收益：{(1 + b_ret).prod() - 1:+.4%} vs {(1 + c['ret']).prod() - 1:+.4%}
       差异来自首尾两天的估值时点不同，以及成本扣除方式（相减 vs 相乘）的二阶差别。
    💡 建议两种成交假设都跑：如果策略只在"收盘成交"下赚钱，在"次日开盘"下不赚钱，
       说明收益可能来自一个你实际拿不到的价格。
    """)

    r_on = (open_ / close.shift(1)).fillna(1.0).prod() - 1
    r_id = (close / open_).prod() - 1
    print(f"    🔍 把买入持有拆成两段：隔夜段（昨收→今开）累计 {r_on:+.2%}，日内段（今开→今收）累计 {r_id:+.2%}")
    print("       A 股不少股票的隔夜段长期偏弱（有研究认为与 T+1 等制度有关），"
          "所以'收盘成交'还是'次日开盘成交'会显著影响择时策略的结果")
    return {"A 收盘成交": a, "C 次日开盘": c, "bh": bh}


# ==================== 第四部分：多资产截面动量 ====================


def cross_sectional_momentum() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第四部分：多资产向量化——截面动量组合")
    print("=" * 60)

    panel = get_close_panel(UNIVERSE, START, END)
    calendar = get_index_daily("000300", START, END).index
    panel = panel.reindex(calendar)
    print(f"📋 股票池 {len(UNIVERSE)} 只，{len(panel)} 个交易日；停牌造成的缺失值：{int(panel.isna().sum().sum())} 个")

    # ⚠️ 收益计算：停牌日没有价格。先 ffill 再算收益 → 停牌日收益为 0，复牌日承担全部跳空
    #    pandas 3 中 pct_change 默认不再自动填充，直接对含 NaN 的价格算收益会丢掉复牌日的跳空
    rets = panel.ffill().pct_change().fillna(0.0)

    # 每 20 个交易日调仓：选过去 20 日动量最高的 3 只，等权
    mom = panel / panel.shift(20) - 1
    rank = mom.rank(axis=1, ascending=False)
    target = (rank <= 3).astype(float)
    target = target.div(target.sum(axis=1), axis=0).fillna(0.0)
    rebalance_days = panel.index[20::20]
    weights = target.loc[rebalance_days].reindex(panel.index).ffill().fillna(0.0)

    pos = weights.shift(1).fillna(0.0)  # 调仓日收盘出权重，次日起持有
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    gross = (pos * rets).sum(axis=1)
    net = gross - turnover * COST
    ew = rets.mean(axis=1)  # 基准：8 只等权（每日再平衡）

    years = len(net) / 252
    print(f"   调仓 {len(rebalance_days)} 次，年化换手 {turnover.sum() / years:.2f} 倍，"
          f"年化成本拖累 {turnover.sum() * COST / years:.2%}")
    table = report_table({"动量Top3(扣费)": net, "动量Top3(扣费前)": gross, "8只等权": ew})
    print(table.loc[["年化收益", "年化波动", "Sharpe", "最大回撤", "Calmar"]].to_string())

    print("""
    ⚠️ 这段代码还隐含了几个简化，事件驱动引擎（06 课）会逐一处理：
       1. 两次调仓之间权重"不漂移"——等价于每天把仓位再平衡回目标权重，却没收这部分成本
       2. 选中的股票如果当天停牌或涨停，实际上买不进
       3. 100 股整数手：小资金买不起茅台，权重无法精确实现
       4. 成本是固定比例，没有最低 5 元佣金、卖出印花税
    """)
    return pd.DataFrame({"动量Top3(扣费)": net, "8只等权": ew})


# ==================== 第五部分：向量化回测的优势与局限 ====================


def pros_and_cons(exec_res: dict, mom: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：向量化回测的优势与局限")
    print("=" * 60)

    print("""
    ┌────────────────────────────────────┬────────────────────────────────────┐
    │ 优势                               │ 局限                               │
    ├────────────────────────────────────┼────────────────────────────────────┤
    │ 极快：上千组参数几秒跑完（11 课）  │ 难以表达路径依赖（止损、资金约束） │
    │ 代码短，研究迭代快                 │ 难以表达涨跌停、T+1、整数手        │
    │ 适合截面因子研究、参数扫描         │ 一个 shift 写错就是前视偏差        │
    │                                    │ 研究代码无法直接复用到实盘         │
    └────────────────────────────────────┴────────────────────────────────────┘
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(exec_res["A 收盘成交"]["nav"], label="MA20 收盘成交")
    ax.plot(exec_res["C 次日开盘"]["nav"], label="MA20 次日开盘成交")
    ax.plot((1 + exec_res["bh"]).cumprod(), label="买入持有", color="gray", ls="--")
    ax.set_title(f"{SYMBOL} {NAME}：成交价假设对比")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    for col in mom.columns:
        ax.plot((1 + mom[col]).cumprod(), label=col)
    ax.set_title("截面动量 Top3 vs 等权组合")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '02_vectorized_backtest.png', subdir='phase-4')}")


# ==================== 主函数 ====================


def main():
    vectorization_idea()
    df = get_stock_daily_adj(SYMBOL, START, END)
    df["adj_open"] = df["open"] * df["adj_factor"]
    handwritten_backtest(df)
    exec_res = execution_assumptions(df)
    mom = cross_sectional_momentum()
    pros_and_cons(exec_res, mom)

    perf = performance_report(exec_res["C 次日开盘"]["ret"])
    print(f"\n📌 本课 MA20（次日开盘成交）的结果将在 06 课与事件驱动引擎对拍：年化 {perf['年化收益']:+.2%}，"
          f"Sharpe {perf['Sharpe']:.2f}")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第二部分的 signal.shift(1) 去掉，Sharpe 变成多少？用一句话解释为什么会这样
    2. 第四部分改为"每 5 天调仓"和"每 60 天调仓"，比较扣费前/扣费后的 Sharpe 和年化换手
    3. 实现权重漂移：两次调仓之间按个股涨跌更新权重 w_i ← w_i(1+r_i)/(1+Σw_j r_j)，
       与"权重不漂移"的版本比较收益差异
    4. 用 vectorized_backtest 做一个"多空"版本（signal 取 -1/+1），思考 A 股个股为什么很难这样做
    """)


if __name__ == "__main__":
    main()
