"""
回测概览：作用、流程与局限
==========================

学习目标：
    1. 理解回测的正确定位：证伪工具，而不是证明工具
    2. 掌握标准回测流程的 8 个步骤，以及每一步可能引入的错误
    3. 跑通第一个"信号 → 仓位 → 收益 → 净值"回测，并做两个必备的健全性检查
    4. 直观感受"回测收益"是如何被前视偏差、成本、成交价假设一步步挤掉水分的

核心概念：
    - 回测（Backtesting）：用历史数据模拟"过去按这套规则交易会怎样"
    - 健全性检查（Sanity Check）：一直满仓 = 买入持有；随机信号扣成本后应亏钱
    - 回测与实盘的差距：前视偏差、成本、滑点、无法成交、过拟合……

前置知识：
    - Phase 1 01-quant-trading-overview.py（信号 → 持仓 → 净值）
    - Phase 2 收益率与回撤

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
COST = 0.001  # 单边成本 0.1%（本课粗略假设，04 课会精细建模）


def annual_return(ret: pd.Series) -> float:
    """几何年化收益"""
    return (1 + ret).prod() ** (252 / len(ret)) - 1


def sharpe(ret: pd.Series) -> float:
    return ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan


# ==================== 第一部分：回测是什么、不是什么 ====================


def what_is_backtest():
    print("=" * 60)
    print("第一部分：回测是什么、不是什么")
    print("=" * 60)

    print("""
    ✈️ 比喻：回测像飞行模拟器。
       - 能发现明显的设计错误（飞机根本飞不起来）
       - 但模拟器里飞得好，不代表真飞机上安全
       - 更危险的是：模拟器本身有 bug（重力设错了），你会成为"模拟器王牌"，然后在现实中坠机

    ┌──────────────────────────────────┬──────────────────────────────────┐
    │ ✅ 回测能做的                    │ ❌ 回测不能做的                  │
    ├──────────────────────────────────┼──────────────────────────────────┤
    │ 否定一个想法（历史上就不赚钱）   │ 证明策略在未来会赚钱             │
    │ 估计策略的历史风险特征           │ 精确预测实盘收益                 │
    │ 暴露实现层面的错误               │ 模拟大资金对市场的冲击           │
    │ 比较不同方案的相对优劣           │ 模拟历史上没出现过的情景         │
    │ 估计成本、换手对收益的侵蚀       │ 消除研究员自己的选择偏差         │
    └──────────────────────────────────┴──────────────────────────────────┘

    🔑 本阶段的核心信念：一个回测结果默认是错的，直到你证明它没错。
    """)


# ==================== 第二部分：标准回测流程 ====================


def backtest_pipeline():
    print("\n" + "=" * 60)
    print("第二部分：标准回测流程")
    print("=" * 60)

    print("""
    ① 研究假设   "A 股存在短期反转"——先有逻辑，再找数据
         ▼
    ② 数据准备   复权、交易日历、停牌、ST、退市、成分股历史（Phase 1）
         ▼
    ③ 信号生成   只能用 t 时刻及之前的信息          ← 07 课：前视偏差
         ▼
    ④ 组合构建   信号 → 目标仓位 → 订单
         ▼
    ⑤ 撮合模拟   成交价、成本、滑点、A 股约束      ← 04、05、06 课
         ▼
    ⑥ 绩效评估   Sharpe、MDD、换手、成本敏感性     ← 03 课
         ▼
    ⑦ 稳健性检验 样本外、Walk-Forward、多重检验    ← 08、09 课
         ▼
    ⑧ 模拟盘 → 小资金实盘（Phase 13）

    📌 软件工程视角：回测引擎就是一个"模拟器 + 记账系统"。
       它的 bug 不会抛异常，只会让你的收益变高 —— 所以要像对待生产代码一样写测试。
    """)


# ==================== 第三部分：第一个回测与健全性检查 ====================


def first_backtest(close: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：第一个回测与健全性检查")
    print("=" * 60)

    ret = close.pct_change().fillna(0.0)
    signal = (close > close.rolling(20).mean()).astype(float)  # t 日收盘后才知道
    pos = signal.shift(1).fillna(0.0)  # t 日的信号，从 t+1 日开始持有
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turnover * COST
    nav = (1 + strat).cumprod()

    print("""
    策略：沪深 300 指数收盘价站上 20 日均线 → 满仓，跌破 → 空仓
    三行核心代码：
        signal = (close > close.rolling(20).mean())     # t 日收盘后算出
        pos    = signal.shift(1)                         # t+1 日才开始持有
        strat  = pos * ret - |Δpos| × 成本
    """)
    print(pd.DataFrame({
        "close": close, "MA20": close.rolling(20).mean(), "signal": signal,
        "pos": pos, "ret": ret, "strat": strat,
    }).iloc[19:27].round(4).to_string())
    print("    ↑ 注意 pos 总是比 signal 晚一行：这一行之差，就是'能不能赚到钱'的分界线")

    # 健全性检查 1：一直满仓 = 买入持有
    always_in = pd.Series(1.0, index=close.index).shift(1).fillna(0.0) * ret
    bh_total = close.iloc[-1] / close.iloc[0] - 1
    bt_total = (1 + always_in).prod() - 1
    print(f"\n🔍 检查 1：'一直满仓'回测总收益 {bt_total:+.6%}，指数区间涨跌 {bh_total:+.6%}，"
          f"差异 {abs(bt_total - bh_total):.2e}")
    print("   两者必须一致；不一致说明收益计算或对齐有 bug")

    # 健全性检查 2：随机信号扣成本后应该亏钱
    rng = np.random.default_rng(42)
    rand_ann = []
    for _ in range(200):
        rsig = pd.Series(rng.integers(0, 2, len(close)), index=close.index, dtype=float)
        rpos = rsig.shift(1).fillna(0.0)
        r = rpos * ret - rpos.diff().abs().fillna(rpos.abs()) * COST
        rand_ann.append(annual_return(r))
    rand_ann = np.array(rand_ann)
    print(f"\n🔍 检查 2：200 组随机信号（每天抛硬币决定是否持仓，种子 42）")
    print(f"   年化收益均值 {rand_ann.mean():+.2%}，中位数 {np.median(rand_ann):+.2%}，"
          f"为正的比例 {(rand_ann > 0).mean():.0%}")
    print("   随机信号几乎每天换仓，成本会把收益压到明显为负；"
          "如果随机信号也稳定赚钱，说明框架本身有 bug（比如成交价用了未来价格）")

    print(f"\n📈 MA20 策略：年化 {annual_return(strat):+.2%}，Sharpe {sharpe(strat):.2f}；"
          f"买入持有：年化 {annual_return(ret):+.2%}，Sharpe {sharpe(ret):.2f}")
    return {"ret": ret, "signal": signal, "nav": nav}


# ==================== 第四部分：回测收益是如何被"挤水分"的 ====================


def squeeze_the_water(close: pd.Series, open_: pd.Series, ctx: dict):
    print("\n" + "=" * 60)
    print("第四部分：回测收益是如何被'挤水分'的")
    print("=" * 60)

    ret, signal = ctx["ret"], ctx["signal"]

    def run(pos: pd.Series, r: pd.Series, cost: float) -> pd.Series:
        return pos * r - pos.diff().abs().fillna(pos.abs()) * cost

    # ① 未来函数：t 日收盘的信号吃到了 t 日全天的收益
    v1 = run(signal, ret, 0.0)
    # ② 正确滞后，收盘价成交，不计成本
    v2 = run(signal.shift(1).fillna(0.0), ret, 0.0)
    # ③ 加上成本
    v3 = run(signal.shift(1).fillna(0.0), ret, COST)
    # ④ 更保守的成交价：t+1 日开盘成交，持有 open(t+1) → open(t+2)，收益在 open.pct_change() 的 t+2 行
    open_ret = open_.pct_change().fillna(0.0)
    v4 = run(signal.shift(2).fillna(0.0), open_ret, COST)

    rows = []
    for name, r in [("① 信号未滞后（未来函数）", v1), ("② 正确滞后，无成本", v2),
                    ("③ + 单边 0.1% 成本", v3), ("④ + 次日开盘成交", v4)]:
        rows.append([name, f"{annual_return(r):+.2%}", f"{sharpe(r):.2f}"])
    print(pd.DataFrame(rows, columns=["版本", "年化收益", "Sharpe"]).to_string(index=False))

    print("""
    💡 解读：
    - ① 与 ② 的差距全部来自"偷看未来"——代码不报错，结果却好得离谱
    - ② → ③ 是交易成本；③ → ④ 是成交价假设。数值的增减取决于具体行情，要逐项报告
    - 这还没算：A 股涨跌停买卖不得、停牌、100 股整数手、过拟合……

          回测收益（看起来很美）
        ┌──────────────────────────────┐
        │  − 前视偏差   ← 07 课        │
        │  − 交易成本   ← 04 课        │
        │  − 无法成交   ← 05、06 课    │
        │  − 过拟合     ← 08、09 课    │
        └──────────────────────────────┘
          实盘收益（常常显著更低）
    """)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in [("① 未来函数", v1), ("② 无成本", v2), ("③ 含成本", v3), ("④ 次日开盘成交", v4)]:
        ax.plot((1 + r).cumprod(), label=name)
    ax.plot((1 + ret).cumprod(), label="买入持有", color="gray", ls="--")
    ax.set_title("沪深 300 MA20 择时：同一策略的四个回测版本")
    ax.set_ylabel("净值")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '01_squeeze_the_water.png', subdir='phase-4')}")


# ==================== 第五部分：两种回测架构预览 ====================


def architectures_preview():
    print("\n" + "=" * 60)
    print("第五部分：两种回测架构预览")
    print("=" * 60)

    print("""
    🎬 比喻：
       向量化回测 = 看整部电影的剧本：一次看到全部情节，快，但容易"偷看结局"
       事件驱动回测 = 一帧一帧看直播：慢，但结构上不可能提前知道结局

    ┌──────────────┬────────────────────────┬────────────────────────┐
    │ 维度         │ 向量化（02 课）        │ 事件驱动（06 课）      │
    ├──────────────┼────────────────────────┼────────────────────────┤
    │ 速度         │ 极快                   │ 慢                     │
    │ 前视偏差风险 │ 高（一个 shift 写错）  │ 低（只能看到过去）     │
    │ A 股约束     │ 别扭                   │ 自然                   │
    │ 典型用途     │ 因子研究、参数扫描     │ 精确验证、对接实盘     │
    └──────────────┴────────────────────────┴────────────────────────┘

    推荐工作流（漏斗式）：向量化初筛 → 事件驱动精确验证 → Walk-Forward → 模拟盘
    """)


# ==================== 主函数 ====================


def main():
    what_is_backtest()
    backtest_pipeline()

    idx = get_index_daily("000300", START, END)
    ctx = first_backtest(idx["close"])
    squeeze_the_water(idx["close"], idx["open"], ctx)
    architectures_preview()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第三部分的均线周期从 20 改为 60，重新运行"挤水分"表格，哪一步的差距变化最大？为什么？
    2. 把随机信号改为"每 20 天才抛一次硬币"（换手更低），随机策略的平均年化收益如何变化？
    3. 写一个函数 sanity_check(backtest_func, close)，自动执行本课的两个健全性检查，
       失败时抛出 AssertionError —— 以后每写一个回测都先跑一遍
    4. 思考：沪深 300 指数本身不能直接交易，用它回测择时策略与用 510300 ETF 回测，会有哪些差异？
    """)


if __name__ == "__main__":
    main()
