"""
Avellaneda-Stoikov 做市模型
===========================

学习目标：
    1. 理解 AS 模型回答的两个问题：报价中心放在哪（保留价格）、价差设多宽（最优价差）
    2. 实现保留价格与最优价差公式，并用数值方法验证价差公式中的对数项
    3. 在泊松到达的模拟环境中比较 AS 策略与对称报价策略的 PnL 与库存分布
    4. 理解风险厌恶系数 γ 的作用，以及模型落地时的参数校准与市场细节问题

核心概念：
    - 保留价格（Reservation Price）：r = s − q·γ·σ²·(T − t)，持有多头时心里的"公允价"低于中间价
    - 最优总价差：δ^a + δ^b = γσ²(T − t) + (2/γ)·ln(1 + γ/k)
    - 成交强度：λ(δ) = A·e^(−kδ)，报价离中间价越远，被成交的概率越低
    - 用一点期望收益，换取大幅降低的库存风险与 PnL 波动

前置知识：
    - 08 课（做市 PnL、库存风险）；Phase 3 的泊松过程与随机游走

环境要求：
    - pip install numpy pandas scipy matplotlib
    - 本课全部使用模拟数据，不需要网络；参数取自 Avellaneda & Stoikov (2008) 的数值实验
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize_scalar

from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

# 原论文数值实验的参数
P = dict(s0=100.0, T=1.0, dt=0.005, sigma=2.0, k=1.5, A=140.0)


# ==================== 第一部分：直觉 ====================


def as_intuition():
    print("=" * 60)
    print("第一部分：AS 模型的两个问题")
    print("=" * 60)
    print("""
    08 课的对称做市没有库存控制，库存像随机游走一样漂移。AS 模型回答：

    1. 报价中心放在哪？ → 保留价格 r（Reservation Price）
       持有多头时急于卖出，心里的"公允价"低于中间价；空头相反
    2. 价差设多宽？   → 最优价差
       太窄：成交多但每笔赚得少、风险大；太宽：每笔赚得多但成交少

                 无库存 q = 0                        多头库存 q > 0
       ask ─────── s + δ                    ask ─────── r + δ   ← 卖价更有吸引力（更易卖出）
       mid ─────── s ← 保留价格 r = s        mid ─────── s
                                                ─────── r       ← 保留价格下移
       bid ─────── s − δ                    bid ─────── r − δ   ← 买价更低（更难买入）

    模型设定：
      中间价 dS = σ dW（算术布朗运动）
      在 S − δ^b 挂买单、S + δ^a 挂卖单；被成交的强度 λ(δ) = A·e^(−kδ)
        A：基础到达强度；k：衰减速度，k 越大，挂远一点成交概率下降越快
      做市商最大化期末财富的指数效用 E[−e^(−γW_T)]，γ 为风险厌恶系数
    """)


# ==================== 第二部分：公式与数值检验 ====================


def reservation_price(s, q, t, T, sigma, gamma):
    return s - q * gamma * sigma**2 * (T - t)


def optimal_spread(t, T, sigma, gamma, k):
    """最优总价差 δ^a + δ^b"""
    return gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + gamma / k)


def as_quotes(s, q, t, T, sigma, gamma, k):
    r = reservation_price(s, q, t, T, sigma, gamma)
    spread = optimal_spread(t, T, sigma, gamma, k)
    return r - spread / 2, r + spread / 2


def formulas_and_checks():
    print("\n" + "=" * 60)
    print("第二部分：核心公式与数值检验")
    print("=" * 60)
    sigma, k, T = P["sigma"], P["k"], P["T"]
    print("""
    保留价格：r(s, q, t) = s − q·γ·σ²·(T − t)
      γσ²(T − t) 是"持有一单位库存到期末"的风险代价；离收盘越近，调整越小

    最优总价差：δ^a + δ^b = γσ²(T − t) + (2/γ)·ln(1 + γ/k)
      第一项：风险补偿，波动越大、离收盘越远，价差越宽
      第二项：与订单流"密度" k 有关，k 越大（挂远了就没人成交），价差越窄
    最终报价：r ± (δ^a + δ^b)/2
    """)

    # 检验 1：对数项来自"单次报价"的最优化
    print("    检验 1：(1/γ)·ln(1 + γ/k) 是使 λ(δ)·(1 − e^(−γδ))/γ 最大的报价距离 δ")
    print("           （成交强度 × 一次成交带来的指数效用增益；对 δ 求导令其为 0 即得）")
    rows = []
    for gamma in [0.01, 0.1, 0.5, 2.0]:
        f = lambda d: -np.exp(-k * d) * (1 - np.exp(-gamma * d)) / gamma
        num = minimize_scalar(f, bounds=(1e-6, 20), method="bounded", options={"xatol": 1e-10}).x
        rows.append([gamma, num, np.log(1 + gamma / k) / gamma, 1 / k])
    print("    " + pd.DataFrame(rows, columns=["γ", "数值最优 δ", "公式 (1/γ)ln(1+γ/k)", "γ→0 极限 1/k"])
          .round(6).to_string(index=False).replace("\n", "\n    "))

    gamma = 0.1
    print(f"\n    检验 2：报价随库存变化（γ = {gamma}，t = 0，s = 100）")
    rows = []
    for q in [-4, -2, 0, 2, 4]:
        bid, ask = as_quotes(100.0, q, 0.0, T, sigma, gamma, k)
        r = reservation_price(100.0, q, 0.0, T, sigma, gamma)
        rows.append([q, r, bid, ask, ask - bid, 100 - bid, ask - 100])
    tab = pd.DataFrame(rows, columns=["库存 q", "保留价格 r", "bid", "ask", "总价差", "δ^b = s − bid", "δ^a = ask − s"])
    print("    " + tab.round(4).to_string(index=False).replace("\n", "\n    "))
    print(f"""
    ✅ 总价差与库存无关（都是 {tab['总价差'].iloc[0]:.4f}），库存只平移报价中心：
       每多 1 单位多头，报价整体下移 γσ²(T−t) = {gamma * sigma**2 * T:.2f}
       q > 0 时 δ^a < δ^b：卖价离中间价更近，更容易卖出、更难买入 —— 自动去库存
    ⚠️ |q| 较大时 δ 可能变成负数（如 q = {tab['库存 q'].iloc[-1]} 时 ask 低于中间价）：实盘中这样的报价会直接与对手盘成交，
       变成主动去库存；所以落地时通常还要加库存上限和报价边界
    """)


# ==================== 第三部分：模拟比较 ====================


def simulate(strategy, gamma, n_paths=1000, seed=0, s0=100.0, T=1.0, dt=0.005, sigma=2.0, k=1.5, A=140.0):
    """
    strategy: "as" 用 AS 报价；"sym" 用同样宽度（q=0 时的 AS 价差），但以中间价为中心
    所有路径向量化，逐步推进。返回 PnL、期末库存、库存路径与成交统计
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    s = np.full(n_paths, s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    inv_path = np.zeros((n_paths, n_steps + 1))
    delta_log, fill_log = [], []
    spreads = []
    for i in range(n_steps):
        t = i * dt
        if strategy == "as":
            bid, ask = as_quotes(s, q, t, T, sigma, gamma, k)
        else:
            half = optimal_spread(t, T, sigma, gamma, k) / 2
            bid, ask = s - half, s + half
        db, da = s - bid, ask - s
        pb = np.clip(A * np.exp(-k * db) * dt, 0, 1)
        pa = np.clip(A * np.exp(-k * da) * dt, 0, 1)
        hit_b = rng.random(n_paths) < pb  # 有人卖给我（我的买单成交）
        hit_a = rng.random(n_paths) < pa  # 有人从我这买（我的卖单成交）
        q += hit_b.astype(float) - hit_a
        cash += np.where(hit_a, ask, 0.0) - np.where(hit_b, bid, 0.0)
        s = s + sigma * np.sqrt(dt) * rng.choice([-1.0, 1.0], n_paths)
        inv_path[:, i + 1] = q
        delta_log.append(np.concatenate([db, da]))
        fill_log.append(np.concatenate([hit_b, hit_a]))
        spreads.append((ask - bid).mean())
    return {
        "pnl": cash + q * s, "q": q, "inv_path": inv_path,
        "delta": np.concatenate(delta_log), "fill": np.concatenate(fill_log),
        "avg_spread": float(np.mean(spreads)),
    }


def compare_strategies():
    print("\n" + "=" * 60)
    print("第三部分：泊松到达的模拟 —— AS vs 对称报价")
    print("=" * 60)
    print(f"""
    参数（原论文数值实验）：s0 = {P['s0']:.0f}，T = {P['T']:.0f}，dt = {P['dt']}，σ = {P['sigma']}，k = {P['k']}，A = {P['A']:.0f}
    对称策略：每一时刻使用与 AS 相同的价差宽度（q = 0 时的值），但以中间价为中心、不随库存平移
    每组 1000 条路径，两种策略使用相同的随机种子
    """)
    rows, results = [], {}
    for gamma in [0.01, 0.1, 0.5]:
        for strat, name in [("as", "AS（库存调整）"), ("sym", "对称报价")]:
            r = simulate(strat, gamma, **P)
            results[(gamma, strat)] = r
            rows.append([gamma, name, r["avg_spread"], r["pnl"].mean(), r["pnl"].std(), r["q"].mean(), r["q"].std(),
                         r["fill"].sum() / len(r["pnl"])])
    table = pd.DataFrame(rows, columns=["γ", "策略", "平均价差", "PnL均值", "PnL标准差", "期末库存均值", "期末库存标准差", "平均成交笔数"])
    print(table.round(2).to_string(index=False))

    print("\n    逐 γ 对比（AS 相对对称报价）：")
    for gamma in [0.01, 0.1, 0.5]:
        a, b = results[(gamma, "as")], results[(gamma, "sym")]
        print(f"      γ = {gamma:<5} PnL 均值 {a['pnl'].mean() - b['pnl'].mean():+6.2f}，"
              f"PnL 标准差 {a['pnl'].std() / b['pnl'].std():.0%}，期末库存标准差 {a['q'].std() / b['q'].std():.0%}")
    print("""
    💡 解读：
       - AS 的 PnL 标准差和库存波动都明显更小：报价随库存平移，自动把库存拉回 0 附近
       - 代价是 PnL 均值通常略低：为了去库存，有时以更差的价格成交
       - γ 越小越接近对称报价（γ → 0 时保留价格调整消失）；γ 越大，价差越宽、成交越少、库存越受控
       原论文的数值实验给出了相同的定性结论：AS 用少量期望收益换来了大幅降低的风险
    """)
    return results


# ==================== 第四部分：检验成交强度假设 ====================


def intensity_check(results):
    print("\n" + "=" * 60)
    print("第四部分：检验模拟中的成交强度 λ(δ) = A·e^(−kδ)")
    print("=" * 60)
    r = results[(0.1, "as")]
    d, f = r["delta"], r["fill"]
    bins = np.quantile(d, np.linspace(0, 1, 7))
    idx = np.clip(np.digitize(d, bins) - 1, 0, 5)
    rows = []
    for b in range(6):
        m = idx == b
        rate = f[m].mean() / P["dt"]
        rows.append([d[m].mean(), rate, P["A"] * np.exp(-P["k"] * d[m].mean()), m.sum()])
    tab = pd.DataFrame(rows, columns=["报价距离 δ（组均值）", "实测成交强度", "A·e^(−kδ)", "样本数"])
    print("    " + tab.round(3).to_string(index=False).replace("\n", "\n    "))
    slope, intercept = np.polyfit(tab["报价距离 δ（组均值）"], np.log(tab["实测成交强度"]), 1)
    print(f"""
    对 ln(实测强度) 与 δ 做线性回归：斜率 = {slope:.3f}（理论 −k = {-P['k']}），截距 e^{intercept:.3f} = {np.exp(intercept):.1f}（理论 A = {P['A']:.0f}）
    （组内 δ 不完全相同、e^(−kδ) 是凸函数，组均值代入会有小偏差）
    📌 实盘校准 A、k 的做法正是这样：统计"报价离中间价 δ 时，单位时间被成交的次数"，再拟合指数曲线
    """)


# ==================== 第五部分：落地问题 ====================


def practical_issues():
    print("\n" + "=" * 60)
    print("第五部分：从模型到实盘")
    print("=" * 60)
    table = pd.DataFrame(
        [
            ["参数校准", "A、k 要从历史成交与盘口拟合；σ 用近期已实现波动率"],
            ["T 的含义", "加密市场 7×24 无收盘，常用滚动窗口或无限期限版本（Guéant 等人的推广）"],
            ["逆向选择", "模型假设价格与订单流无关；实盘要叠加 OFI / 微价格等短期信号调整报价中心"],
            ["tick size", "报价必须对齐最小变动单位；A 股 0.01 元的 tick 对低价股相当于很宽的价差"],
            ["排队位置", "模型假设挂在 δ 就有 λ(δ) 的成交率，真实中还取决于你在队列中的位置"],
            ["手续费", "必须扣除 maker 费率；taker 费率高的市场，价差收入可能被吃光"],
            ["风控", "硬性库存上限、亏损上限、异常行情时撤单（08 课的库存上限仍然需要）"],
        ],
        columns=["问题", "说明"],
    )
    print(table.to_string(index=False))
    print("""
    📌 A 股股票受 T+1 约束，普通投资者很难像模型里那样做日内双边做市（持有底仓做日内回转也受库存与规则限制）；
       国内的做市实践主要是交易所认可的做市商业务（如期权、ETF、科创板股票做市）。
       加密市场的公开盘口常被用于研究，但中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动：
       本阶段项目实战"加密订单簿做市模拟器"只做公开数据回放模拟，不进行任何真实交易。
    """)


# ==================== 可视化 ====================


def plot_all(results):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    gamma = 0.1
    a, b = results[(gamma, "as")], results[(gamma, "sym")]
    ax = axes[0]
    bins = np.arange(min(a["q"].min(), b["q"].min()) - 0.5, max(a["q"].max(), b["q"].max()) + 1.5)
    ax.hist(b["q"], bins=bins, alpha=0.5, label="对称报价")
    ax.hist(a["q"], bins=bins, alpha=0.6, label="AS")
    ax.set_title(f"期末库存分布（γ = {gamma}）")
    ax.set_xlabel("期末库存 q")
    ax.legend()

    ax = axes[1]
    ax.hist(b["pnl"], bins=40, alpha=0.5, label="对称报价")
    ax.hist(a["pnl"], bins=40, alpha=0.6, label="AS")
    ax.set_title(f"PnL 分布（γ = {gamma}）")
    ax.set_xlabel("PnL")
    ax.legend()

    ax = axes[2]
    t = np.linspace(0, P["T"], a["inv_path"].shape[1])
    ax.plot(t, b["inv_path"][0], color="tab:blue", alpha=0.7, label="对称报价")
    ax.plot(t, a["inv_path"][0], color="tab:orange", label="AS")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("同一条路径上的库存变化")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("库存 q")
    ax.legend()

    path = save_figure(fig, "09_avellaneda_stoikov.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    as_intuition()
    formulas_and_checks()
    results = compare_strategies()
    intensity_check(results)
    practical_issues()
    plot_all(results)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 σ 翻倍，比较 AS 与对称报价的 PnL 标准差之比，解释为什么波动越大库存控制越重要
    2. 给报价加上 tick 约束（四舍五入到 0.01 的整数倍），观察 PnL 和库存分布的变化
    3. 在价格过程中加入漂移（例如 dS = μdt + σdW），AS 策略会怎样？如何用短期信号修正保留价格？
    4. 把 08 课的"知情交易者"加入本模拟，观察 AS 策略的 PnL 是否仍为正，并用 Markout 诊断
    """)


if __name__ == "__main__":
    main()
