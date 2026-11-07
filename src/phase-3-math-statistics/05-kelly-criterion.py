"""
期望值、胜率、盈亏比与凯利公式
==============================

学习目标：
    1. 理解"高胜率 ≠ 赚钱"：期望值由胜率和盈亏比共同决定
    2. 从真实交易记录中统计胜率、盈亏比、每笔期望
    3. 推导并模拟凯利公式：下注太少增长慢，下注太多会被"玩死"
    4. 理解半凯利的价值：估计误差让全凯利变成过度下注
    5. 认识连续凯利 f* = (μ - r_f)/σ²，以及它为什么常常给出离谱的杠杆

核心概念：
    - 每笔期望 E = p·W - q·L = L·(p·b - q)，b = W/L 为盈亏比
    - 盈亏平衡胜率 p* = 1/(1+b)
    - 对数增长率 g(f) = p·ln(1+f·b) + q·ln(1-f)
    - 凯利比例 f* = p - q/b；连续版 f* = (μ - r_f)/σ²

前置知识：
    - 03-hypothesis-testing.py、04-monte-carlo-simulation.py

环境要求：
    - pip install numpy pandas matplotlib akshare
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

TRADING_DAYS = 252
START, END = "20100101", "20251231"


def growth_rate(f: np.ndarray, p: float, b: float) -> np.ndarray:
    """每次下注的期望对数增长率 g(f)"""
    return p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)


def kelly_fraction(p: float, b: float) -> float:
    return p - (1 - p) / b


# ==================== 第一部分：期望值 —— 高胜率不等于赚钱 ====================


def expectancy_basics():
    print("=" * 60)
    print("第一部分：期望值 —— 高胜率不等于赚钱")
    print("=" * 60)

    print("""
    每笔期望 E = p·W - q·L = L·(p·b - q)
       p 胜率，q = 1-p，W 平均盈利，L 平均亏损（取正数），b = W/L 盈亏比
    盈亏平衡胜率：令 E = 0 → p* = 1/(1+b)
    """)

    cases = [
        ("高胜率·小赚大亏", 0.90, 1.0, 20.0),
        ("抛硬币·对称", 0.50, 1.0, 1.0),
        ("趋势跟踪·小亏大赚", 0.35, 3.0, 1.0),
        ("均值回归·常见形态", 0.60, 1.0, 1.2),
    ]
    rows = []
    for name, p, w, l in cases:
        b = w / l
        rows.append([name, f"{p:.0%}", w, l, f"{b:.2f}", f"{1 / (1 + b):.1%}", f"{p * w - (1 - p) * l:+.2f}"])
    print(pd.DataFrame(rows, columns=["策略", "胜率", "平均盈利W", "平均亏损L", "盈亏比b", "盈亏平衡胜率", "每笔期望"]).to_string(index=False))

    rng = np.random.default_rng(0)
    n_trades = 1000
    print(f"\n🎲 每个策略模拟 {n_trades} 笔交易（每笔固定盈亏，单位：元）：")
    for name, p, w, l in cases:
        wins = rng.random(n_trades) < p
        pnl = np.where(wins, w, -l)
        print(f"   {name:<14s} 实际胜率 {wins.mean():.1%}，累计盈亏 {pnl.sum():+8.1f}")
    print("""
    💡 胜率和盈亏比必须一起看，单独报告任何一个都没有意义。
       "胜率 90%"的策略可能长期必亏；"胜率 35%"的策略可能长期赚钱。
    """)


# ==================== 第二部分：从真实交易统计胜率与盈亏比 ====================


def trade_stats():
    print("\n" + "=" * 60)
    print("第二部分：从真实交易统计胜率与盈亏比")
    print("=" * 60)

    close = get_index_daily("000300", START, END)["close"]
    rows = []
    for window in [20, 60, 120]:
        pos = (close > close.rolling(window).mean()).astype(int).shift(1).fillna(0)
        ret = close.pct_change().fillna(0)
        # 每段连续持仓 = 一笔交易；单笔收益 = 持仓期间的累计收益，扣除买卖各 0.1% 成本
        trade_id = (pos.diff() == 1).cumsum()
        held = pos == 1
        trade_ret = (1 + ret[held]).groupby(trade_id[held]).prod() - 1 - 0.002
        win = trade_ret[trade_ret > 0]
        loss = trade_ret[trade_ret <= 0]
        p = len(win) / len(trade_ret)
        b = win.mean() / -loss.mean()
        rows.append([f"MA{window}", len(trade_ret), f"{p:.1%}", f"{win.mean():+.2%}", f"{loss.mean():+.2%}",
                     f"{b:.2f}", f"{1 / (1 + b):.1%}", f"{trade_ret.mean():+.2%}"])
    print(f"📊 沪深 300 均线择时，{close.index[0].date()} ~ {close.index[-1].date()}，每笔扣 0.2% 往返成本：")
    print(pd.DataFrame(rows, columns=["策略", "交易笔数", "胜率", "平均盈利", "平均亏损", "盈亏比",
                                      "盈亏平衡胜率", "每笔期望"]).to_string(index=False))
    print("""
    💡 均线择时是典型的趋势跟踪：胜率通常不高，靠少数大赚覆盖多数小亏。
       判断是否赚钱，要比较"胜率"和"盈亏平衡胜率"，而不是看胜率是否超过 50%。
       ⚠️ 交易笔数只有几十到一两百笔时，胜率和盈亏比本身的估计误差就很大（回忆 03 课）。
    """)


# ==================== 第三部分：凯利公式 ====================


def kelly_curve():
    print("\n" + "=" * 60)
    print("第三部分：凯利公式 —— 每次该押多少")
    print("=" * 60)

    print("""
    每次押上当前本金的比例 f：赢了本金 ×(1 + f·b)，输了 ×(1 - f)
    长期复利增长由"对数增长率"决定：g(f) = p·ln(1 + f·b) + q·ln(1 - f)
    对 f 求导令其为 0 → 凯利比例 f* = p - q/b = (p·b - q)/b

       增长率 g(f)
         ▲
         │          ● f*（最优）
         │       ╱     ╲
         │     ╱         ╲
        0┼──────────────────●──────────▶ f
         │                ≈2f*  ╲ 负增长区：过度下注
    """)

    p, b = 0.55, 1.0
    f_star = kelly_fraction(p, b)
    f = np.linspace(0.001, 0.4, 400)
    g = growth_rate(f, p, b)
    zero_cross = f[1:][(g[:-1] > 0) & (g[1:] <= 0)]
    print(f"📐 p = {p}, b = {b} → f* = {f_star:.1%}，最大增长率 g(f*) = {growth_rate(f_star, p, b):.4%} / 次")
    if len(zero_cross):
        print(f"   增长率回到 0 的位置 f ≈ {zero_cross[0]:.1%}（约 {zero_cross[0] / f_star:.1f} 倍 f*）")

    rng = np.random.default_rng(1)
    n_bets, n_paths = 1000, 2000
    wins = rng.random((n_bets, n_paths)) < p
    rows = []
    for mult in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        frac = mult * f_star
        log_w = np.where(wins, np.log(1 + frac * b), np.log(1 - frac)).cumsum(axis=0)
        wealth = np.exp(log_w)
        final = wealth[-1]
        mdd = (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(axis=0)
        rows.append([f"{mult:g} × f*", f"{frac:.1%}", f"{growth_rate(frac, p, b) / growth_rate(f_star, p, b):.0%}",
                     f"{np.median(final):,.2f}", f"{(final < 1).mean():.1%}", f"{np.median(mdd):.1%}",
                     f"{(mdd < -0.5).mean():.1%}"])
    print(f"\n🎲 模拟 {n_paths} 条路径 × {n_bets} 次下注（初始财富 1）：")
    print(pd.DataFrame(rows, columns=["下注比例", "f", "理论增长率/最优", "财富中位数", "最终亏损概率",
                                      "MDD 中位数", "曾回撤超 50% 的概率"]).to_string(index=False))
    print("""
    💡 观察：
       - f* 处中位数财富最高；超过 2f* 后增长率转负 —— 正期望的机会也会被下注过大"玩死"
       - 半凯利（0.5 × f*）的增长率仍有最优值的约 75%，但回撤明显更小
       - 曲线左右不对称：少押一点损失很小，多押一点损失迅速增加 → 宁可低估
    """)
    return p, b, f, g


# ==================== 第四部分：估计误差与半凯利 ====================


def estimation_error(p_true: float, b: float):
    print("\n" + "=" * 60)
    print("第四部分：估计误差 —— 为什么实践中常用半凯利")
    print("=" * 60)

    print(f"""
    真实胜率 p = {p_true}，但你只能从有限笔历史交易中估计 p̂，再按 p̂ 计算凯利比例。
    🧪 实验：分别用 50 / 200 / 1000 笔历史交易估计胜率，各重复 5000 次
    """)
    rng = np.random.default_rng(2)
    f_true = kelly_fraction(p_true, b)
    g_best = growth_rate(f_true, p_true, b)
    rows = []
    for n_hist in [50, 200, 1000]:
        p_hat = rng.binomial(n_hist, p_true, 5000) / n_hist
        for label, scale in [("全凯利", 1.0), ("半凯利", 0.5)]:
            f_hat = np.clip(scale * kelly_fraction(p_hat, b), 0, 0.99)
            g_real = growth_rate(f_hat, p_true, b)  # 用"真实"胜率评估按估计值下注的效果
            rows.append([n_hist, label, f"[{np.percentile(p_hat, 5):.1%}, {np.percentile(p_hat, 95):.1%}]",
                         f"{np.mean(g_real) / g_best:.0%}", f"{(g_real < 0).mean():.1%}"])
    print(pd.DataFrame(rows, columns=["历史笔数", "下注方式", "p̂ 的 90% 区间", "平均增长率/理论最优",
                                      "负增长的概率"]).to_string(index=False))
    print("""
    💡 解读：
       - 样本少时 p̂ 波动很大：高估胜率 → 过度下注 → 增长率骤降甚至为负
       - 对比同一历史笔数下的两行：样本少时，半凯利的平均增长率反而可能高于全凯利；
         只有当参数估计得足够准时，全凯利才体现出优势
       - 真实交易中，胜率、盈亏比还会随市场状态漂移，比这个实验更难估 → 实践中常用半凯利甚至更低
    """)


# ==================== 第五部分：连续凯利与杠杆 ====================


def continuous_kelly():
    print("\n" + "=" * 60)
    print("第五部分：连续凯利 f* = (μ - r_f)/σ²")
    print("=" * 60)

    close = get_index_daily("000300", START, END)["close"]
    ret = close.pct_change().dropna()
    rf = 0.02
    mu = ret.mean() * TRADING_DAYS
    sigma = ret.std() * np.sqrt(TRADING_DAYS)
    f_star = (mu - rf) / sigma**2
    print(f"📐 沪深 300（{close.index[0].year}-{close.index[-1].year}）：算术年化 μ = {mu:.2%}，σ = {sigma:.1%}，r_f = {rf:.0%}")
    print(f"   → 连续凯利 f* = {f_star:.2f}")
    print(f"   示例：若年化超额收益 10%、波动 20%，f* = 0.10 / 0.04 = {0.10 / 0.04:.1f} 倍杠杆")

    print("\n🔍 用历史数据检验：不同固定杠杆（每日再平衡）下的实际结果")
    rows = []
    cagrs = {}
    rf_d = rf / TRADING_DAYS
    for lev in [0.25, 0.5, 1.0, 2.0, 3.0]:
        r = rf_d + lev * (ret - rf_d)
        nav = (1 + r).cumprod()
        years = len(r) / TRADING_DAYS
        cagr = nav.iloc[-1] ** (1 / years) - 1 if nav.iloc[-1] > 0 else -1.0
        mdd = (nav / nav.cummax() - 1).min()
        cagrs[lev] = cagr
        rows.append([f"{lev:g} 倍", f"{cagr:+.2%}", f"{r.std() * np.sqrt(TRADING_DAYS):.1%}", f"{mdd:.1%}"])
    print(pd.DataFrame(rows, columns=["杠杆", "年化复合收益", "年化波动", "最大回撤"]).to_string(index=False))
    best = max(cagrs, key=cagrs.get)
    print(f"   本样本中复合收益最高的杠杆档位：{best:g} 倍（事后用全样本 μ、σ 算出的 f* = {f_star:.2f}）")
    print("""
    💡 μ 的估计误差极大（01、03 课）：μ 稍有高估，凯利杠杆就会成倍放大。
       凯利还假设每次下注独立、分布已知不变 —— 真实市场的相关性、状态切换都会破坏这一点。
       在 Phase 5 和 Phase 8，你会更多使用"波动率目标"和风险预算来决定仓位。
    """)


def plot_kelly(p: float, b: float, f: np.ndarray, g: np.ndarray):
    setup_chinese_font()
    f_star = kelly_fraction(p, b)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(f, g * 100, lw=2)
    ax.axhline(0, color="gray", lw=0.8)
    for mult, ls in [(0.5, ":"), (1.0, "--"), (2.0, "-.")]:
        ax.axvline(mult * f_star, color="C3", ls=ls, label=f"{mult:g} × f* = {mult * f_star:.0%}")
    ax.set_xlabel("每次下注比例 f")
    ax.set_ylabel("每次期望对数增长率（%）")
    ax.set_title(f"凯利增长曲线（p = {p}, b = {b}）")
    ax.legend()
    path = save_figure(fig, "05_kelly_curve.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    expectancy_basics()
    trade_stats()
    p, b, f, g = kelly_curve()
    plot_kelly(p, b, f, g)
    estimation_error(p, b)
    continuous_kelly()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 一个策略胜率 40%、盈亏比 2，计算每笔期望和凯利比例；如果胜率被高估了 5 个百分点，按高估值下注会怎样？
    2. 用第二部分 MA60 的实际胜率和盈亏比计算凯利比例。这个数字能直接用于实盘吗？为什么？
    3. 推导：当 f = 2f* 时，对于 b = 1 的情形，g(f) 是否近似为 0？（提示：对 ln(1±x) 做二阶泰勒展开）
    4. 实现"波动率目标"仓位：仓位 = 目标波动 15% / 过去 60 日年化波动，与固定 1 倍杠杆比较回撤
    """)


if __name__ == "__main__":
    main()
