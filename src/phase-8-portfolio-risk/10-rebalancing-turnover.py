"""
再平衡与换手控制
================

学习目标：
    1. 理解权重漂移：不调仓，组合会慢慢"变形"
    2. 比较不同再平衡频率（日 / 周 / 月 / 季 / 年 / 从不）的收益、风险、换手与成本
    3. 比较阈值再平衡与无交易区间（no-trade band），以及部分调仓
    4. 做成本敏感性分析：交易成本多高时，频繁再平衡就不划算了
    5. 了解平方根冲击模型与"主动管理基本定律 + 转移系数"

核心概念：
    - 漂移权重 w_drift = w·(1+r) / (1 + w'r)
    - 单边换手 = ½ Σ|w_new - w_drift|（用漂移后权重，不是上期目标权重）
    - 阈值再平衡：偏离超过阈值才调回目标；无交易区间：只调到区间边界
    - 冲击成本 ≈ c·σ·√(Q/V)；IR ≈ TC × IC × √BR

前置知识：
    - Phase 4 交易成本、03-risk-parity.py
    - CONCEPT.md 第 12 节

环境要求：
    - pip install numpy pandas matplotlib akshare
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用示意参数的模拟收益，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter
import pandas as pd
from dotenv import load_dotenv

from portfolio_utils import TRADING_DAYS, load_etf_returns, perf_stats
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

TARGET = {"沪深300": 0.25, "中证500": 0.10, "创业板": 0.05, "纳指": 0.075, "标普500": 0.075, "国债": 0.35, "黄金": 0.10}
COST = 0.001  # 单边成本 10bp：ETF 免印花税，主要是佣金（约万 1~万 3）+ 买卖价差 / 冲击（示例值）


# ==================== 回测引擎 ====================


def simulate(returns: pd.DataFrame, target: np.ndarray, rule: str = "periodic", freq: str | None = "M",
             band: float = 0.0, eta: float = 1.0, cost: float = COST) -> dict:
    """
    逐日模拟：收盘后按规则决定是否交易，成本在交易日从组合价值中扣除

    rule:
        "never"     买入后不再调仓
        "periodic"  每个周期最后一个交易日调回目标（freq: D/W/M/Q/Y）；eta < 1 为部分调仓
        "threshold" 任一资产 |偏离| > band 时全部调回目标
        "band"      无交易区间：只把超出 [目标 ± band] 的资产调到边界，再归一化
    """
    R = returns.to_numpy()
    T, n = R.shape
    if rule == "periodic":
        if freq == "D":
            is_end = np.ones(T, dtype=bool)
        else:
            period = returns.index.to_period(freq)
            is_end = np.r_[period[1:] != period[:-1], True]
    w = target.copy()
    net = np.empty(T)
    turnover = np.zeros(T)
    deviation = np.empty(T)
    for t in range(T):
        gross = w @ R[t]
        w_drift = w * (1 + R[t]) / (1 + gross)
        deviation[t] = np.abs(w_drift - target).max()
        w_new = w_drift
        if rule == "periodic" and is_end[t]:
            w_new = w_drift + eta * (target - w_drift)
        elif rule == "threshold" and deviation[t] > band:
            w_new = target
        elif rule == "band" and deviation[t] > band:
            w_new = np.clip(w_drift, target - band, target + band)
            w_new = w_new / w_new.sum()
        traded = np.abs(w_new - w_drift).sum()
        turnover[t] = 0.5 * traded
        net[t] = (1 + gross) * (1 - cost * traded) - 1
        w = w_new
    years = T / TRADING_DAYS
    return {
        "ret": pd.Series(net, index=returns.index),
        "ann_turnover": turnover.sum() / years,
        "n_trades": int((turnover > 1e-12).sum()),
        "cost_drag": cost * 2 * turnover.sum() / years,
        "avg_dev": deviation.mean(),
    }


def summarize(results: dict, ideal: pd.Series) -> pd.DataFrame:
    rows = {}
    for label, res in results.items():
        s = perf_stats(res["ret"])
        te = (res["ret"] - ideal).std() * np.sqrt(TRADING_DAYS)
        rows[label] = {
            "年化收益(扣费)": f"{s['年化收益']:+.2%}",
            "年化波动": f"{s['年化波动']:.2%}",
            "Sharpe": f"{s['Sharpe']:.2f}",
            "最大回撤": f"{s['最大回撤']:.2%}",
            "年化单边换手": f"{res['ann_turnover']:.0%}",
            "交易天数": res["n_trades"],
            "年化成本": f"{res['cost_drag']:.3%}",
            "平均最大偏离": f"{res['avg_dev']:.2%}",
            "相对理想TE": f"{te:.2%}",
        }
    return pd.DataFrame(rows).T


# ==================== 第一部分：权重漂移 ====================


def weight_drift(returns: pd.DataFrame, target: np.ndarray):
    print("=" * 60)
    print("第一部分：权重漂移 —— 不调仓，组合会\"变形\"")
    print("=" * 60)
    value = (1 + returns).cumprod() * target
    weights = value.div(value.sum(axis=1), axis=0)
    yearly = weights.groupby(weights.index.year).tail(1)
    yearly.index = yearly.index.year
    print(f"目标权重：{TARGET}")
    print("\n买入持有（从不再平衡）时，每年年末的实际权重 %：")
    print((yearly * 100).round(1).to_string())
    eq = ["沪深300", "中证500", "创业板", "纳指", "标普500"]
    print(f"""
    权益类合计：目标 {target[[returns.columns.get_loc(c) for c in eq]].sum():.0%}，
               期末 {weights[eq].iloc[-1].sum():.1%}，期间最高 {weights[eq].sum(axis=1).max():.1%}、最低 {weights[eq].sum(axis=1).min():.1%}
    💡 涨得多的资产权重越来越大——不再平衡等于"追涨"，组合的风险特征会偏离当初的设计。
    """)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(weights.index, weights[eq].sum(axis=1) * 100, color="#2a78d6", lw=1.6, label="权益类合计（买入持有）")
    ax.axhline(target[[returns.columns.get_loc(c) for c in eq]].sum() * 100, color="#8a8984", ls="--", lw=1, label="目标")
    ax.set_ylabel("权重 %")
    ax.set_title("不再平衡时权益类资产的权重漂移")
    ax.grid(alpha=0.3)
    ax.legend()
    path = save_figure(fig, "10_weight_drift.png", subdir="phase-8")
    print(f"🖼  权重漂移图已保存：{path}")


# ==================== 第二部分：再平衡频率 ====================


def rebalance_frequency(returns: pd.DataFrame, target: np.ndarray):
    print("\n" + "=" * 60)
    print("第二部分：再平衡频率")
    print("=" * 60)
    ideal = simulate(returns, target, "periodic", "D", cost=0.0)["ret"]
    results = {
        "每日": simulate(returns, target, "periodic", "D"),
        "每周": simulate(returns, target, "periodic", "W"),
        "每月": simulate(returns, target, "periodic", "M"),
        "每季": simulate(returns, target, "periodic", "Q"),
        "每年": simulate(returns, target, "periodic", "Y"),
        "从不": simulate(returns, target, "never"),
    }
    print(f"单边成本 {COST:.2%}；\"相对理想TE\" = 与\"每日再平衡且零成本\"组合的收益差的年化波动")
    print(summarize(results, ideal).to_string())
    print("""
    💡 读表：
       - 频率越高，换手和成本越高，但越贴近目标配置（偏离、TE 越小）
       - "从不"再平衡时波动和回撤取决于漂移方向：涨得多的高波动资产占比变大，风险通常随之上升
       - 再平衡是否带来"再平衡收益"取决于资产是否均值回复，不是必然的——看扣费后的年化收益列
    """)
    return ideal, results


# ==================== 第三部分：阈值、无交易区间与部分调仓 ====================


def bands_and_partial(returns: pd.DataFrame, target: np.ndarray, ideal: pd.Series, freq_results: dict):
    print("\n" + "=" * 60)
    print("第三部分：阈值再平衡、无交易区间与部分调仓")
    print("=" * 60)
    print("""
         目标 ──────●────────           阈值再平衡：偏离超过 band → 一次调回中心 ●
              |←band→|←band→|           无交易区间：只拉回到最近的边界 |，交易量更小
                                         部分调仓：每次只走 η 比例的距离
    """)
    results = {}
    for b in [0.01, 0.02, 0.03, 0.05]:
        results[f"阈值 ±{b:.0%}"] = simulate(returns, target, "threshold", band=b)
        results[f"无交易区间 ±{b:.0%}"] = simulate(returns, target, "band", band=b)
    for eta in [0.5, 0.25]:
        results[f"每月部分调仓 η={eta}"] = simulate(returns, target, "periodic", "M", eta=eta)
    results["每月（对照）"] = freq_results["每月"]
    print(summarize(results, ideal).to_string())
    print("""
    💡 读表：
       - 同样的 band 下，无交易区间的换手通常低于阈值再平衡（只调到边界），代价是平均偏离更大
       - 但无交易区间的"交易天数"多：频繁的小额交易在有最低佣金（如每笔 5 元）时并不划算，
         实践中常加一个最小交易金额
       - 理论上在比例成本下，最优策略就是一个无交易区间（Davis & Norman 1990 等）
       - 比较时要看"换手 vs 跟踪误差"的权衡，而不是只看收益：下图每个点是一种规则
       ⚠️ 本例的无交易区间在调到边界后做了归一化，会让少数资产略微越界，这是简化处理
    """)

    all_res = {**{k: v for k, v in freq_results.items() if k != "从不"}, **results}
    all_res.pop("每月（对照）")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    groups = [("定期", "#2a78d6", lambda k: k in freq_results), ("阈值", "#eb6834", lambda k: k.startswith("阈值")),
              ("无交易区间", "#1baf7a", lambda k: k.startswith("无交易")), ("部分调仓", "#4a3aa7", lambda k: "部分" in k)]
    for legend, color, belongs in groups:
        pts = [(k, v) for k, v in all_res.items() if belongs(k)]
        xs = [v["ann_turnover"] * 100 for _, v in pts]
        ys = [(v["ret"] - ideal).std() * np.sqrt(TRADING_DAYS) * 100 for _, v in pts]
        ax.scatter(xs, ys, s=60, color=color, label=legend, edgecolor="white", lw=1.5, zorder=3)
        left = legend == "无交易区间"   # 无交易区间的标签放在点的左侧，避免和阈值、部分调仓的标签重叠
        for (k, _), x, y in zip(pts, xs, ys):
            ax.annotate(k, (x, y), xytext=(-6 if left else 6, 3), textcoords="offset points",
                        fontsize=8, ha="right" if left else "left")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 5, 10, 20, 50, 100])
    ax.set_xlim(left=min(v["ann_turnover"] for v in all_res.values()) * 100 / 2.2)
    ax.set_xlabel("年化单边换手 %（对数轴）")
    ax.set_ylabel("相对理想组合的跟踪误差 %")
    ax.set_title("换手与跟踪误差的权衡")
    ax.grid(alpha=0.3)
    ax.legend()
    path = save_figure(fig, "10_turnover_vs_te.png", subdir="phase-8")
    print(f"🖼  换手-跟踪误差权衡图已保存：{path}")


# ==================== 第四部分：成本敏感性 ====================


def cost_sensitivity(returns: pd.DataFrame, target: np.ndarray):
    print("\n" + "=" * 60)
    print("第四部分：成本敏感性 —— 成本多高时频繁调仓就不划算")
    print("=" * 60)
    rules = {
        "每日": dict(rule="periodic", freq="D"),
        "每月": dict(rule="periodic", freq="M"),
        "每季": dict(rule="periodic", freq="Q"),
        "每年": dict(rule="periodic", freq="Y"),
        "阈值±3%": dict(rule="threshold", band=0.03),
        "无交易区间±3%": dict(rule="band", band=0.03),
    }
    costs_bp = [0, 5, 10, 30, 50, 100]
    grid = pd.DataFrame(index=[f"{c}bp" for c in costs_bp], columns=list(rules), dtype=float)
    for c in costs_bp:
        for label, kw in rules.items():
            grid.loc[f"{c}bp", label] = perf_stats(simulate(returns, target, cost=c / 1e4, **kw)["ret"])["年化收益"]
    print("扣费后年化收益（行 = 单边成本，列 = 再平衡规则）：")
    print(grid.map(lambda x: f"{x:+.2%}").to_string())
    best = grid.idxmax(axis=1)
    print("\n每个成本水平下收益最高的规则：" + "，".join(f"{k}→{v}" for k, v in best.items()))
    print("""
    💡 成本越高，低换手的规则越有优势；零成本时各规则的差异只来自"再平衡本身"的收益影响。
       ETF 的成本较低（免印花税），个股组合（卖出印花税 0.05% + 冲击）的结论会更偏向低频。
    """)


# ==================== 第五部分：冲击成本与转移系数 ====================


def impact_and_tc():
    print("\n" + "=" * 60)
    print("第五部分：平方根冲击模型与转移系数")
    print("=" * 60)
    sigma_daily = 0.02
    print(f"冲击成本 ≈ c · σ_daily · √(Q/V)，取 c = 1、σ_daily = {sigma_daily:.0%}：")
    rows = []
    for qv in [0.001, 0.01, 0.04, 0.10, 0.25]:
        rows.append([f"{qv:.1%}", f"{sigma_daily * np.sqrt(qv):.3%}", f"{sigma_daily * np.sqrt(qv) * 1e4:.0f}bp"])
    print(pd.DataFrame(rows, columns=["交易量/日均成交量 Q/V", "单位冲击成本", "约合"]).to_string(index=False))
    print("   交易量翻 4 倍，单位成本翻 2 倍 → 冲击成本决定了策略容量（Phase 11 详述）")

    ic, br = 0.05, 500
    print(f"\n主动管理基本定律：IR ≈ TC × IC × √BR（IC = {ic}，每年独立下注 BR = {br}）")
    rows = [[f"{tc:.1f}", f"{tc * ic * np.sqrt(br):.2f}"] for tc in [1.0, 0.8, 0.6, 0.4, 0.3]]
    print(pd.DataFrame(rows, columns=["转移系数 TC", "理论 IR"]).to_string(index=False))
    print("""
    💡 换手约束、多头约束、行业中性都会降低 TC（见第 06 课的 TC 列）：
       控制换手省下的成本，要和 TC 下降损失的 alpha 放在一起权衡。
       信号衰减快（如短期反转）需要高换手才能兑现；信号慢（如价值）可以低换手——
       用 Phase 7 的 IC 衰减曲线来决定调仓频率。
    """)


# ==================== 主函数 ====================


def main():
    returns = load_etf_returns()
    if is_synthetic(returns):
        print("⚠️ 以下结果基于模拟收益，只用于理解方法，不代表真实市场\n")
    target = np.array([TARGET[c] for c in returns.columns])
    weight_drift(returns, target)
    ideal, freq_results = rebalance_frequency(returns, target)
    bands_and_partial(returns, target, ideal, freq_results)
    cost_sensitivity(returns, target)
    impact_and_tc()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把无交易区间改为"相对区间"（目标权重的 ±20%），低权重资产和高权重资产的交易频率怎么变？
    2. 实现"带现金缓冲"的无交易区间：调到边界后不归一化，多出 / 不足的部分记为现金
    3. 把目标组合换成第 03 课的 ERC（每月重新估计），目标本身在变时，换手主要来自漂移还是目标变化？
    4. 用 cvxpy 把再平衡写成优化问题：min TE²(w) + κ·Σ|w - w_drift|，扫描 κ 画出"换手-TE"前沿，
       与第三部分的规则比较
    """)


if __name__ == "__main__":
    main()
