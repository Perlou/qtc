"""
卡尔曼滤波动态对冲比
====================

学习目标：
    1. 理解为什么固定对冲比会"过时"：公司之间的关系会缓慢漂移
    2. 用 numpy 手写卡尔曼滤波（预测 → 预测误差 → 卡尔曼增益 → 更新），不依赖第三方库
    3. 先在"真实 β 已知"的模拟数据上验证实现正确，再用于真实配对
    4. 理解 delta 参数的含义：太大 β 跟着噪声乱跳，太小退化为固定 OLS
    5. 样本外对比：固定 OLS 对冲比 vs 卡尔曼动态对冲比（计入 β 调整带来的换手成本）

核心概念：
    - 状态方程：β_t = β_{t-1} + ω_t，ω ~ N(0, Vw)（β 随机游走）
    - 观测方程：y_t = β_t · x_t + α_t + ε_t，ε ~ N(0, Ve)
    - 预测误差 e_t = y_t - H_t · β̂_{t|t-1}，其方差 Q_t；e_t / √Q_t 就是天然的 Z-score
    - 卡尔曼增益 K：预测越不确定、观测越精确，就越相信新观测（GPS 导航的比喻）

前置知识：
    - 02、03 课；线性代数基础（矩阵乘法、转置）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行：优先读本地缓存；无缓存时 utils 返回合成行情，
      本课检测到后改用模拟的协整价格组（打印 ⚠️ 使用合成数据）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stat_arb_utils import (BANKS, FORM_END, PAIR, TEST_START, half_life, kalman_hedge, load_bank_panel,
                            ols_hedge, pair_backtest, pair_positions, perf_stats, stats_table)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

WARMUP = 60  # 卡尔曼预热期：初始状态不准，前 60 个观测不用于交易
DELTAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]


# ==================== 第一部分：直觉与状态空间模型 ====================


def kalman_intuition():
    print("=" * 60)
    print("第一部分：为什么需要动态对冲比 + 卡尔曼滤波的直觉")
    print("=" * 60)

    print("""
    ┌──────────────┬────────────────────────────┬────────────────────────────────┐
    │ 方法         │ 优点                       │ 缺点                           │
    ├──────────────┼────────────────────────────┼────────────────────────────────┤
    │ 全样本 OLS   │ 简单                       │ 前视偏差，无法适应变化         │
    │ 滚动 OLS     │ 能适应变化                 │ 窗口是额外参数；β 跳动         │
    │ 卡尔曼滤波   │ 每天平滑更新，自带不确定性 │ 要设定噪声参数，理解门槛稍高   │
    └──────────────┴────────────────────────────┴────────────────────────────────┘

    🚗 GPS 导航比喻：车的真实位置（β）看不见，只能通过有噪声的 GPS 信号（价格）间接观测。
       每一刻做两件事：
         1. 预测："β 应该和昨天差不多"，同时承认不确定性变大了一点     R = P + Vw
         2. 更新：收到今天的价格，按"信任比例"修正                      β = β + K·e
       信任比例 = 卡尔曼增益 K = R·Hᵀ / Q：预测越不确定（R 大）、观测越精确（Ve 小），越相信新数据

    递推（每个交易日一次）：
      H_t = [x_t, 1]                  观测矩阵（x 为 B 的对数价格）
      R_t = P_{t-1} + Vw              预测协方差
      e_t = y_t - H_t·β̂_{t-1}         预测误差 —— 这就是"价差偏离"，只用到昨天的状态，无前视
      Q_t = H_t·R_t·H_tᵀ + Ve         预测误差方差，√Q_t 是偏离的"标准差"
      K_t = R_t·H_tᵀ / Q_t            卡尔曼增益
      β̂_t = β̂_{t-1} + K_t·e_t        更新状态
      P_t = R_t - K_t·H_t·R_t         更新协方差
    """)


# ==================== 第二部分：在已知真相的模拟数据上验证 ====================


def fit_kalman(y: np.ndarray, x: np.ndarray, mask: np.ndarray, deltas=DELTAS) -> pd.DataFrame:
    """
    最大似然估计噪声参数：对每个 delta 网格搜索 Ve，目标为高斯对数似然 -0.5·Σ(ln Q + e²/Q)
    （只在 mask 区间内求和，即形成期、去掉预热期）。返回每个 delta 的最优 Ve 与对数似然
    """
    rows = []
    for d in deltas:
        best = (None, -np.inf)
        for ve in np.logspace(-8, -2, 25):
            _, e, Q = kalman_hedge(y, x, delta=d, ve=ve)
            ll = -0.5 * np.sum(np.log(Q[mask]) + e[mask] ** 2 / Q[mask])
            if ll > best[1]:
                best = (ve, ll)
        rows.append({"delta": d, "ve": best[0], "loglik": best[1]})
    return pd.DataFrame(rows)


def verify_on_simulation():
    print("\n" + "=" * 60)
    print("第二部分：先验证 —— 模拟一个 β 随时间漂移的配对")
    print("=" * 60)

    rng = np.random.default_rng(5)
    n = 1200
    x = np.cumsum(rng.normal(0, 0.015, n))  # 对数价格减去起点（log P/P0），减少 β 与截距的混淆
    t = np.arange(n)
    true_beta = 0.8 + 0.4 * t / n + 0.15 * (t >= 800)  # 缓慢上升 + 第 800 天一次跳升
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.005)
    y = 0.3 + true_beta * x + noise
    idx = pd.bdate_range("2018-01-01", periods=n)
    ys, xs = pd.Series(y, index=idx), pd.Series(x, index=idx)

    _, b_full = ols_hedge(ys, xs)
    win = 120
    roll = np.full(n, np.nan)
    for i in range(win, n):
        roll[i] = ols_hedge(ys.iloc[i - win:i], xs.iloc[i - win:i])[1]
    mask = np.arange(n) >= WARMUP
    fit = fit_kalman(y, x, mask)
    bestp = fit.loc[fit["loglik"].idxmax()]
    states, e, Q = kalman_hedge(y, x, delta=bestp["delta"], ve=bestp["ve"])
    kf = states[:, 0]

    ok = slice(WARMUP + win, n)
    rows = [
        ["全样本 OLS（有前视）", np.mean(np.abs(b_full - true_beta[ok]))],
        [f"滚动 OLS（{win} 天）", np.nanmean(np.abs(roll[ok] - true_beta[ok]))],
        [f"卡尔曼（MLE：delta={bestp['delta']:.0e}, Ve={bestp['ve']:.1e}）", np.mean(np.abs(kf[ok] - true_beta[ok]))],
    ]
    print(f"   真实 β 从 {true_beta[0]:.2f} 漂移到 {true_beta[-1]:.2f}，第 800 天跳升 0.15")
    print(pd.DataFrame(rows, columns=["方法", "与真实 β 的平均绝对误差"]).round(4).to_string(index=False))
    after = slice(800, 860)
    print(f"   跳升后 60 天内的平均误差：滚动 OLS {np.nanmean(np.abs(roll[after] - true_beta[after])):.4f}，"
          f"卡尔曼 {np.mean(np.abs(kf[after] - true_beta[after])):.4f}")
    z = e / np.sqrt(Q)
    print(f"   标准化预测误差 e/√Q：均值 {z[WARMUP:].mean():+.3f}，标准差 {z[WARMUP:].std():.3f}"
          f"（噪声参数设定合理时应接近 0 和 1）")
    print("""
    💡 先看两点：卡尔曼的 β 是否跟上了真实 β 的漂移（误差应明显小于全样本 OLS）；
       e/√Q 的标准差是否接近 1（噪声参数是否合理）。
       写任何量化模型，都应该先在"答案已知"的模拟数据上验证，再用到真实数据。
    """)
    return idx, true_beta, kf, roll, b_full


# ==================== 第三部分：delta 的含义与参数估计 ====================


def norm_log(logp: pd.DataFrame) -> pd.DataFrame:
    """对数价格减去起点：log(P_t / P_0)。复权基准不影响结果，也减少 β 与截距的混淆"""
    return logp - logp.iloc[0]


def delta_effect(logp: pd.DataFrame) -> tuple[dict, float]:
    print("\n" + "=" * 60)
    print("第三部分：delta 的含义与参数估计")
    print("=" * 60)

    a, b = PAIR
    nl = norm_log(logp)
    y, x = nl[a].to_numpy(), nl[b].to_numpy()
    form_mask = (logp.index <= pd.Timestamp(FORM_END)) & (np.arange(len(logp)) >= WARMUP)
    fit = fit_kalman(y, x, form_mask)
    rows, results = [], {}
    for _, r in fit.iterrows():
        states, e, Q = kalman_hedge(y, x, delta=r["delta"], ve=r["ve"])
        beta = states[form_mask, 0]
        results[r["delta"]] = (states, e, Q)
        rows.append([f"{r['delta']:.0e}", f"{r['ve']:.1e}", r["loglik"], beta.min(), beta.max(),
                     np.abs(np.diff(beta)).mean() * 252, np.std(e[form_mask]),
                     np.std(e[form_mask] / np.sqrt(Q[form_mask])), half_life(pd.Series(e[form_mask]))])
    cols = ["delta", "Ve(MLE)", "对数似然", "β 最小", "β 最大", "β 年化变动量", "e 标准差", "e/√Q 标准差",
            "e 半衰期(天)"]
    best_d = float(fit.loc[fit["loglik"].idxmax(), "delta"])
    print(f"   {BANKS[a]} ~ {BANKS[b]}，输入为 log(P/P0)；每个 delta 用最大似然选 Ve；统计区间：形成期（去掉预热期）")
    print(pd.DataFrame(rows, columns=cols).round(4).to_string(index=False))
    print(f"   → 对数似然最大的 delta = {best_d:.0e}（只用形成期数据选定，之后不再改动）")
    print("""
    💡 delta 越大，β（和截距）越灵活：价差偏离被状态"吸收"，预测误差变小、看起来更"平稳"，
       但没有交易价值，还会带来更多调仓；delta 太小则 β 几乎不动，退化为固定 OLS。
       注意"e 半衰期"一列：如果模型设定正确，一步预测误差 e 应接近白噪声（不可预测）。
       e 的均值回归越弱，直接拿 e/√Q 做 Z-score（Chan 的做法）就越像在交易噪声 ——
       所以下面同时比较"把卡尔曼 β 当对冲比、再用滚动 Z-score"的做法。
    """)
    return results, best_d


# ==================== 第四部分：样本外对比 ====================


def oos_compare(logp: pd.DataFrame, results: dict, best_d: float):
    print("\n" + "=" * 60)
    print("第四部分：样本外对比 —— 固定 OLS vs 卡尔曼")
    print("=" * 60)

    a, b = PAIR
    idx = logp.index
    nl = norm_log(logp)
    in_test = idx >= pd.Timestamp(TEST_START)
    states, e, Q = results[best_d]
    beta_kf = pd.Series(states[:, 0], index=idx)

    # 1) 固定 OLS：形成期 β + 滚动 Z-score（与 03 课相同）
    form = nl.loc[:FORM_END]
    _, beta_ols = ols_hedge(form[a], form[b])
    spread = nl[a] - beta_ols * nl[b]
    lb = int(np.clip(round(2 * half_life(spread.loc[:FORM_END])), 10, 120))

    def roll_z(s: pd.Series) -> pd.Series:
        return ((s - s.rolling(lb).mean()) / s.rolling(lb).std()).where(in_test)

    bt_ols = pair_backtest(pair_positions(roll_z(spread), 2.0, 0.5, 4.0), beta_ols, logp[a], logp[b])
    # 2) 卡尔曼 β 当对冲比：s_t = y_t - β_{t-1}·x_t（用昨天的 β，避免今天的偏离被今天的更新吸收）
    spread_kf = nl[a] - beta_kf.shift(1) * nl[b]
    bt_kf = pair_backtest(pair_positions(roll_z(spread_kf), 2.0, 0.5, 4.0), beta_kf, logp[a], logp[b])
    # 3) Chan 的做法：z = e/√Q，±1 开仓、回到 0 平仓
    z_chan = pd.Series(e / np.sqrt(Q), index=idx).where(in_test)
    bt_chan = pair_backtest(pair_positions(z_chan, 1.0, 0.0, 4.0), beta_kf, logp[a], logp[b])
    bts = {"固定 OLS β": bt_ols.loc[in_test], "卡尔曼 β+滚动 Z": bt_kf.loc[in_test],
           "卡尔曼 e/√Q（Chan）": bt_chan.loc[in_test]}

    stats = {}
    for k, bt in bts.items():
        stats[f"{k}·未扣成本"] = perf_stats(bt["gross"], bt["pos"])
        stats[f"{k}·扣全部成本"] = perf_stats(bt["net"], bt["pos"])
    first = bts["固定 OLS β"]
    print(f"   delta = {best_d:.0e}；滚动 Z-score 窗口 L = {lb}；交易期 {first.index[0].date()} ~ {first.index[-1].date()}")
    table = stats_table(stats)
    print(table.iloc[:, :2].to_string())
    print(table.iloc[:, 2:4].to_string())
    print(table.iloc[:, 4:].to_string())
    years = len(first) / 252
    print("\n   年化换手（两条腿合计）与其中\"持仓方向不变、仅随 β 调整 B 腿\"的部分：")
    for k, bt in bts.items():
        held = (bt["pos"] != 0) & (bt["pos"] == bt["pos"].shift(1))
        print(f"     {k:<18s} 总换手 {bt['turnover'].sum() / years:5.1f}x，β 调整 {bt.loc[held, 'turnover'].sum() / years:5.2f}x")
    print(f"   交易期卡尔曼 β 范围：{beta_kf[in_test].min():.3f} ~ {beta_kf[in_test].max():.3f}；固定 OLS β = {beta_ols:.3f}")
    print("""
    💡 观察要点（结论以上面的实际数字为准）：
       1. 卡尔曼的优势是"适应缓慢漂移"；β 每天变化会带来额外的调仓换手 —— 一定要比较扣成本后的结果
       2. 三种做法的开平仓阈值不完全相同，直接比较时要意识到这一点
       3. 卡尔曼不是"让不协整的配对变协整"的魔法：基本面突然分化时，它会跟着走，
          把结构性断裂误认为新的均衡（08 课）
    """)
    return beta_kf, beta_ols


def plot_kalman(sim, beta_kf: pd.Series, beta_ols: float, best_d: float, is_synth: bool):
    setup_chinese_font()
    idx, true_beta, kf, roll, b_full = sim
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(idx, true_beta, "k", lw=2, label="真实 β")
    axes[0].plot(idx, roll, label="滚动 OLS（120 天）", alpha=0.8)
    axes[0].plot(idx, kf, label="卡尔曼", alpha=0.8)
    axes[0].axhline(b_full, color="gray", ls="--", label="全样本 OLS")
    axes[0].set_ylim(0.5, 1.6), axes[0].legend(), axes[0].set_title("模拟验证：β 漂移 + 跳升")
    b = beta_kf.iloc[WARMUP:]
    axes[1].plot(b.index, b, label=f"卡尔曼 β（delta={best_d:.0e}）")
    axes[1].axhline(beta_ols, color="gray", ls="--", label="形成期 OLS β")
    axes[1].axvline(pd.Timestamp(TEST_START), color="r", ls=":", lw=1)
    axes[1].legend()
    axes[1].set_title(f"{BANKS[PAIR[0]]} ~ {BANKS[PAIR[1]]} 动态对冲比{'【合成数据】' if is_synth else ''}")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '04_kalman_hedge.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    prices, _, is_synth = load_bank_panel()
    logp = np.log(prices)

    kalman_intuition()
    sim = verify_on_simulation()
    results, best_d = delta_effect(logp)
    beta_kf, beta_ols = oos_compare(logp, results, best_d)
    plot_kalman(sim, beta_kf, beta_ols, best_d, is_synth)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第二部分的跳升从 0.15 改为 0.5，卡尔曼需要多少天才能把误差降到 0.05 以内？delta 越大越快吗？
    2. 把观测噪声 Ve 放大 10 倍、缩小 10 倍，β 的平滑程度如何变化？Ve 与 delta 的作用有什么关系？
    3. 在卡尔曼回测中，每天只在 |β_t - 当前持仓 β| > 0.05 时才调整 B 腿，换手和净收益如何变化？
    4. 用 2018-2019 年数据选 delta，在 2020-2024 年测试，结论与本课一致吗？
    """)


if __name__ == "__main__":
    main()
