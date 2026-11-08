"""
做市原理与库存风险
==================

学习目标：
    1. 理解做市商的收入来源（价差）与两大风险（库存风险、逆向选择）
    2. 用模拟撮合比较不同知情交易者比例下对称报价做市的 PnL
    3. 把 PnL 精确分解为"价差收入 + 库存损益"，并用 Markout 度量逆向选择
    4. 理解价差宽度的权衡，以及库存上限这类最朴素的库存控制手段

核心概念：
    - 价差收入（Spread Capture）：每笔成交相对中间价赚到的半个价差
    - 库存损益（Inventory PnL）：Σ q_t·ΔM_t，持仓随中间价波动带来的盈亏
    - 逆向选择（Adverse Selection）：和"知道价格要动"的人成交，成交后价格朝不利方向走
    - Markout(Δ) = −q_fill·(M_{t+Δ} − P_fill)：成交后 Δ 时刻的不利移动，正数 = 被逆向选择

前置知识：
    - 03 课（价差三分法、Glosten-Milgrom）；Phase 3 的随机游走

环境要求：
    - pip install numpy pandas matplotlib
    - 本课全部使用模拟数据，不需要网络
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

S0 = 100.0
SIGMA = 0.08  # 每步中间价变动 ±0.08
HALF_SPREAD = 0.05
N_STEPS = 2000
N_PATHS = 500


# ==================== 第一部分：做市商赚什么钱 ====================


def mm_intro():
    print("=" * 60)
    print("第一部分：做市商赚什么钱")
    print("=" * 60)
    print(f"""
    比喻：做市商像二手车商，9 万收车、10 万卖车，赚的是周转差价，而不是赌车价涨跌。

      同时挂出： 买 @ 中间价 − h          卖 @ 中间价 + h
                    │                         │
         有人卖给我 ▼                         ▼ 有人从我这买
                 库存 +1                    库存 −1

      理想：买卖成交次数相当，每对成交赚 2h，库存回到 0
      风险 ①库存风险：只成交一边，库存堆积，价格朝不利方向走
           ②逆向选择：和"知道价格要动"的人成交，成交后立刻浮亏

    PnL 恒等式（本课会数值验证）：
      PnL = Σ 半价差收入 + Σ q_t·ΔM_t（库存损益） − 手续费 + 返佣

    本课模拟设定：中间价每步 ±{SIGMA}（等概率），做市商始终在中间价 ±{HALF_SPREAD} 双边报价，
    每步以一定概率来一笔市价单：知情交易者知道下一步价格方向，噪声交易者随机买卖。
    """)


# ==================== 模拟器 ====================


def simulate_mm(half_spread=HALF_SPREAD, sigma=SIGMA, p_arrive=0.5, informed=0.0,
                n_paths=N_PATHS, n_steps=N_STEPS, inv_cap=None, seed=7):
    """
    对称报价做市模拟（所有路径向量化，逐步推进）。
    mm_side：做市商这笔成交的方向，+1 = 做市商买入（对手方主动卖），−1 = 做市商卖出
    inv_cap：库存上限，|库存| 达到上限后停止挂会继续加仓那一侧的报价
    """
    rng = np.random.default_rng(seed)
    moves = sigma * rng.choice([-1.0, 1.0], (n_paths, n_steps))
    arrive = rng.random((n_paths, n_steps)) < p_arrive
    is_inf = rng.random((n_paths, n_steps)) < informed
    noise_side = rng.choice([-1, 1], (n_paths, n_steps))
    # 知情交易者：价格要涨就来买（做市商卖出，mm_side = −1）
    want_side = np.where(is_inf, -np.sign(moves).astype(int), noise_side)

    mids = S0 + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(moves, axis=1)], axis=1)  # 长度 n_steps+1
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spread_pnl = np.zeros(n_paths)
    inv_pnl = np.zeros(n_paths)
    side = np.zeros((n_paths, n_steps), dtype=int)
    inv_path = np.zeros((n_paths, n_steps + 1))
    for t in range(n_steps):
        s = np.where(arrive[:, t], want_side[:, t], 0)
        if inv_cap is not None:
            s = np.where((q + s > inv_cap) | (q + s < -inv_cap), 0, s)  # 这一侧已不挂单
        px = mids[:, t] - s * half_spread  # 做市商买在 bid = M − h，卖在 ask = M + h
        q += s
        cash -= s * px
        spread_pnl += np.abs(s) * half_spread
        inv_pnl += q * moves[:, t]  # 成交后持有 q，经历这一步的价格变化
        side[:, t] = s
        inv_path[:, t + 1] = q
    pnl = cash + q * mids[:, -1]
    return {"pnl": pnl, "spread": spread_pnl, "inventory": inv_pnl, "q_final": q, "side": side,
            "is_inf": is_inf, "mids": mids, "inv_path": inv_path, "half_spread": half_spread}


def markout(res, horizon):
    """每笔成交的 Markout(Δ) = −side·(M_{t+Δ} − P_fill)；正数 = 成交后价格朝不利方向走"""
    side, mids = res["side"], res["mids"]
    n_steps = side.shape[1]
    t_idx = np.arange(n_steps - horizon)
    s = side[:, t_idx]
    fill_px = mids[:, t_idx] - s * res["half_spread"]
    m_after = mids[:, t_idx + horizon]
    mo = -s * (m_after - fill_px)
    mask = s != 0
    return mo[mask], res["is_inf"][:, t_idx][mask]


# ==================== 第二部分：知情交易者比例与 PnL ====================


def informed_experiment():
    print("\n" + "=" * 60)
    print("第二部分：对称报价做市 —— 知情交易者越多，越难赚钱")
    print("=" * 60)
    rows, results = [], {}
    p_arrive = 0.5
    for mu in [0.0, 0.2, 0.4, 0.6, 0.8]:
        r = simulate_mm(informed=mu, p_arrive=p_arrive)
        results[mu] = r
        identity = np.abs(r["pnl"] - r["spread"] - r["inventory"]).max()
        theory = N_STEPS * p_arrive * (HALF_SPREAD - mu * SIGMA)
        rows.append([mu, r["pnl"].mean(), r["pnl"].std() / np.sqrt(N_PATHS), theory, r["pnl"].std(), r["spread"].mean(),
                     r["inventory"].mean(), np.abs(r["q_final"]).mean(), identity])
    table = pd.DataFrame(rows, columns=["知情比例", "PnL均值", "标准误", "理论期望", "PnL标准差", "价差收入", "库存损益", "|期末库存|均值", "恒等式误差"])
    print(table.to_string(index=False, formatters={
        "知情比例": "{:.1f}".format, "PnL均值": "{:.1f}".format, "标准误": "{:.1f}".format, "理论期望": "{:.1f}".format, "PnL标准差": "{:.1f}".format,
        "价差收入": "{:.1f}".format, "库存损益": "{:.1f}".format, "|期末库存|均值": "{:.1f}".format, "恒等式误差": "{:.1e}".format,
    }))
    print(f"""
    {N_PATHS} 条路径 × {N_STEPS} 步，每步到达概率 {p_arrive}
    ✅ PnL = 价差收入 + 库存损益，恒等式误差在浮点精度量级
    ✅ 模拟均值与理论期望的最大偏差 = {((table['PnL均值'] - table['理论期望']).abs() / table['标准误']).max():.1f} 个标准误

    理论期望怎么来的？每笔成交先赚 h = {HALF_SPREAD}；
      与噪声交易者成交：之后的价格变化与成交方向无关，库存损益期望为 0
      与知情交易者成交：下一步价格必然朝不利方向走 σ = {SIGMA}，期望亏 σ
      → 每步期望 PnL = 到达概率 × (h − μ·σ)，盈亏平衡点 μ* = h/σ = {HALF_SPREAD / SIGMA:.3f}

    💡 价差收入只取决于成交笔数，几乎不随知情比例变化；
       真正决定盈亏的是库存损益 —— 逆向选择通过"成交后的价格移动"吃掉价差收入。
       没有库存控制时，库存像随机游走一样漂移，PnL 标准差很大（09 课的 AS 模型专门解决这个问题）。
    """)
    return results


# ==================== 第三部分：Markout 度量逆向选择 ====================


def markout_analysis(results):
    print("\n" + "=" * 60)
    print("第三部分：Markout —— 成交之后价格往哪走")
    print("=" * 60)
    print("""
    Markout(Δ) = −q_fill·(M_{t+Δ} − P_fill)，q_fill：做市商买入为 +1
      Δ = 0 时 Markout = −h（刚成交，账面赚半个价差）
      若 Markout 随 Δ 上升并变成正数 → 成交后价格系统性朝不利方向走 → 被逆向选择
    """)
    horizons = [0, 1, 5, 20, 100]
    rows = []
    for mu, r in results.items():
        rows.append([mu] + [markout(r, h)[0].mean() for h in horizons])
    table = pd.DataFrame(rows, columns=["知情比例"] + [f"Δ={h}" for h in horizons])
    print("    平均 Markout（元/股，正数 = 不利）：")
    print("    " + table.round(4).to_string(index=False).replace("\n", "\n    "))

    mo, inf = markout(results[0.4], 5)
    print(f"""
    按对手方类型拆开（知情比例 0.4，Δ=5）：
      与噪声交易者成交：平均 Markout {mo[~inf].mean():+.4f}（≈ −h，赚到半个价差）
      与知情交易者成交：平均 Markout {mo[inf].mean():+.4f}（≈ σ − h，被逆向选择）
    📌 实盘中你不知道对手是谁，但可以按时段、对手渠道、订单大小分组看 Markout，
       找出"有毒"的订单流（Toxic Flow），对它们报更宽的价或干脆不报。
    """)
    return table


# ==================== 第四部分：价差宽度的权衡 ====================


def spread_tradeoff():
    print("\n" + "=" * 60)
    print("第四部分：价差该报多宽？")
    print("=" * 60)
    p_i, p0, k = 0.1, 0.6, 15.0
    print(f"""
    更真实的设定：
      噪声交易者对价格敏感：报价越宽来得越少，到达概率 = {p0} × exp(−{k}·h)
      知情交易者每步以 {p_i} 的概率出现，只在有利可图（σ > h）时才成交
    每步期望 PnL = 噪声到达率 × h + 知情到达率 × (h − σ) × 1{{h < σ}}
    """)
    rows = []
    rng = np.random.default_rng(11)
    for h in [0.01, 0.03, 0.05, 0.07, 0.09, 0.12]:
        pn = p0 * np.exp(-k * h)
        theory = pn * h + (p_i * (h - SIGMA) if h < SIGMA else 0.0)
        # 模拟：直接按到达概率抽样（成交后的价格变化：噪声为 ±σ 随机，知情必然不利）
        n = 2_000_000
        noise = rng.random(n) < pn
        inf = (rng.random(n) < p_i) & (h < SIGMA)
        pnl = noise * (h + SIGMA * rng.choice([-1, 1], n)) + inf * (h - SIGMA)
        rows.append([h, pn, theory * N_STEPS, pnl.mean() * N_STEPS])
    table = pd.DataFrame(rows, columns=["半价差 h", "噪声到达率", f"理论 PnL/{N_STEPS}步", "模拟 PnL"])
    print(table.round(3).to_string(index=False))
    best = table.loc[table[f"理论 PnL/{N_STEPS}步"].idxmax(), "半价差 h"]
    print(f"""
    本设定下理论最优的半价差在 h = {best} 附近（网格内）。
    💡 太窄：被知情者吃、单笔赚得少；太宽：噪声交易者不来，成交太少。
       最优价差取决于订单流的"毒性"和对价格的敏感度 —— 09 课的 AS 模型用 A·e^(−kδ) 刻画同样的权衡。
    """)


# ==================== 第五部分：最朴素的库存控制 ====================


def inventory_cap():
    print("\n" + "=" * 60)
    print("第五部分：库存上限 —— 最朴素的库存控制")
    print("=" * 60)
    rows = []
    res = {}
    for cap in [None, 20, 10, 5]:
        r = simulate_mm(informed=0.0, inv_cap=cap, seed=21)
        res[cap] = r
        fills = np.abs(r["side"]).sum(axis=1).mean()
        rows.append(["无上限" if cap is None else cap, r["pnl"].mean(), r["pnl"].std(), fills, r["spread"].mean(),
                     r["inv_path"].std(), np.abs(r["inv_path"]).max()])
    table = pd.DataFrame(rows, columns=["库存上限", "PnL均值", "PnL标准差", "平均成交笔数", "价差收入", "库存标准差", "最大|库存|"])
    print(table.round(2).to_string(index=False))
    sharpe = table["PnL均值"] / table["PnL标准差"]
    print(f"""
    （只有噪声交易者，其他设定同第二部分）
    PnL均值 / PnL标准差：{', '.join(f'{c}: {v:.2f}' for c, v in zip(table['库存上限'], sharpe))}
    💡 库存上限把库存锁在 ±cap 以内，PnL 标准差从 {table['PnL标准差'].iloc[0]:.1f} 降到 {table['PnL标准差'].iloc[-1]:.1f}；
       代价是拒绝了一部分成交，价差收入从 {table['价差收入'].iloc[0]:.1f} 降到 {table['价差收入'].iloc[-1]:.1f}。
       （这里没有知情交易者，库存损益期望为 0，所以 PnL 均值的差异主要是噪声：标准误约 {table['PnL标准差'].iloc[0] / np.sqrt(N_PATHS):.1f}）
       更聪明的做法是"报价偏移（skew）"：多头库存时同时降低买价和卖价，让卖单更容易成交 —— 见 09 课。
    """)
    return res


# ==================== 可视化 ====================


def plot_all(results, markout_table, cap_res):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    for mu in [0.0, 0.4, 0.8]:
        r = results[mu]
        ax.hist(r["pnl"], bins=40, alpha=0.5, label=f"知情比例 {mu}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("对称做市的 PnL 分布（模拟）")
    ax.set_xlabel("PnL（元）")
    ax.legend()

    ax = axes[1]
    hs = [int(c.split("=")[1]) for c in markout_table.columns[1:]]
    for _, row in markout_table.iterrows():
        ax.plot(hs, row.iloc[1:].to_numpy(), marker="o", label=f"知情比例 {row['知情比例']:.1f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(hs) * 1.2)
    ax.set_title("平均 Markout vs 时间间隔")
    ax.set_xlabel("Δ（步）")
    ax.set_ylabel("Markout（正数 = 不利）")
    ax.legend(fontsize=8)

    ax = axes[2]
    for cap, color in [(None, "tab:gray"), (10, "tab:blue")]:
        ax.plot(cap_res[cap]["inv_path"][0], color=color, label="无上限" if cap is None else f"上限 ±{cap}")
    ax.set_title("库存路径（同一条路径的两种做法）")
    ax.set_xlabel("步")
    ax.set_ylabel("库存")
    ax.legend()

    path = save_figure(fig, "08_market_making_basics.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    mm_intro()
    results = informed_experiment()
    markout_table = markout_analysis(results)
    spread_tradeoff()
    cap_res = inventory_cap()
    plot_all(results, markout_table, cap_res)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给模拟加上 maker 手续费（或返佣），重新计算盈亏平衡的知情比例
    2. 让知情交易者只知道"5 步之后"的价格方向，观察 Markout 曲线的形状如何变化
    3. 实现报价偏移：中心价 = M − κ·q，比较不同 κ 下的库存标准差和 PnL（提示：需要让成交概率依赖报价距离）
    4. 用 02 课的零智能市场：在订单簿上挂 Post-Only 的买一 / 卖一，统计你的成交与 Markout
    """)


if __name__ == "__main__":
    main()
