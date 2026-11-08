"""
Deflated Sharpe Ratio 与 PBO
============================

学习目标：
    1. 理解多次试验的选择偏差：N 个纯噪声策略中最好的那个，夏普也会相当可观
    2. 计算 PSR（修正样本长度、偏度、峰度）与 DSR（再修正试验次数），并与 Phase 4 bt_core 对拍
    3. 用 CSCV（组合对称交叉验证）估计回测过拟合概率 PBO
    4. 在自己的一组策略试验上（Phase 7 因子的 255 种组合）完整走一遍"去伪"流程

核心概念：
    - PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )，SR 为非年化
    - E[max SR] ≈ √V[SR_n] · ((1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))，γ ≈ 0.5772
    - DSR = PSR(E[max SR])；DSR > 0.95 才能在 5% 水平上排除"选择偏差"
    - PBO = P(样本内最优策略在样本外排名低于中位数)

前置知识：
    - Phase 4 第 08 课（过拟合与多重检验，已实现 PSR / DSR）
    - 05、07 课

环境要求：
    - pip install pandas numpy scipy matplotlib
    - 数据来自 Phase 7 共享面板；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase-4-backtesting"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, rankdata, skew

import bt_core
import ml_tools as mt
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

EULER_GAMMA = 0.5772156649015329
SEED = 42
S_BLOCKS = 16
TOPK = 20


# ==================== 公式实现 ====================


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std(ddof=1))


def psr(r, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio：真实（非年化）夏普超过 sr_star 的概率"""
    r = np.asarray(r, dtype=float)
    sr, g3, g4 = sharpe(r), skew(r), kurtosis(r, fisher=False)
    return float(norm.cdf((sr - sr_star) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)))


def expected_max_sr(sr_var: float, n: int) -> float:
    """N 次独立试验、真实夏普均为 0 时，样本夏普最大值的期望"""
    return float(np.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                    + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def pbo_cscv(M: np.ndarray, S: int = S_BLOCKS) -> tuple[float, np.ndarray, np.ndarray]:
    """
    CSCV 估计 PBO。M: T × N（N 个策略配置的收益）
    返回 (PBO, 每种划分的 logit λ, 每种划分 [IS 最优的 IS 夏普, 它的 OOS 夏普])
    """
    T, N = M.shape
    blocks = np.array_split(np.arange(T), S)
    sums = np.array([M[b].sum(0) for b in blocks])  # 按块预先求和，避免重复计算
    sqs = np.array([(M[b] ** 2).sum(0) for b in blocks])
    cnt = np.array([len(b) for b in blocks])

    def sr(mask):
        n, s, q = cnt[mask].sum(), sums[mask].sum(0), sqs[mask].sum(0)
        mu = s / n
        return mu / np.sqrt((q - n * mu**2) / (n - 1))

    logits, pairs = [], []
    for is_ids in combinations(range(S), S // 2):
        m = np.zeros(S, dtype=bool)
        m[list(is_ids)] = True
        sr_is, sr_oos = sr(m), sr(~m)
        best = int(np.argmax(sr_is))
        omega = rankdata(sr_oos)[best] / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
        pairs.append((sr_is[best], sr_oos[best]))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits, np.array(pairs)


# ==================== 第一部分：选择偏差 ====================


def selection_bias():
    print("=" * 60)
    print("第一部分：试得越多，最好的那个越'好看'")
    print("=" * 60)
    print("""
    🐒 比喻：1000 只猴子各扔 10 次硬币，总有一只连续 10 次正面 —— 它不是"扔硬币天才"。
       你报告的夏普，必须与"同样多次尝试中，最好的猴子能达到的夏普"相比。
    """)
    rng = np.random.default_rng(SEED)
    T, reps = 756, 200  # 3 年日度数据，每个 N 重复 200 次
    rows = []
    for n in [1, 10, 100, 1000]:
        best = []
        for _ in range(reps):
            R = rng.normal(0, 0.01, (T, n))
            best.append((R.mean(0) / R.std(0, ddof=1)).max())
        best = np.array(best)
        theo = expected_max_sr(1 / T, n) if n > 1 else 0.0  # 纯噪声下 V[SR] ≈ 1/T
        rows.append({"试验次数 N": n, "模拟：最大夏普均值(年化)": best.mean() * np.sqrt(252),
                     "公式 E[max SR](年化)": theo * np.sqrt(252)})
    print(f"📋 纯噪声策略（真实夏普 = 0），3 年日度数据，每档重复 {reps} 次（固定种子）：")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("""
    💡 尝试 1000 次，纯靠运气就能得到一个年化夏普相当可观的"策略"。
       公式与模拟吻合 —— 它就是 DSR 的基准 SR₀。
    """)


# ==================== 第二部分：PSR —— 偏度与峰度的影响 ====================


def psr_demo():
    print("\n" + "=" * 60)
    print("第二部分：PSR —— 同样的夏普，可信度不同")
    print("=" * 60)
    print("""
        PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )
        SR 为非年化夏普；γ₃ 偏度；γ₄ 峰度（正态 = 3）。负偏、高峰度 → 分母变大 → PSR 降低
    """)
    rng = np.random.default_rng(SEED)
    T = 756
    mu, sd = 0.0006, 0.01  # 年化夏普约 0.95

    def standardize(x):  # 调成完全相同的均值和波动 → 两者的样本夏普完全相同
        return (x - x.mean()) / x.std(ddof=1) * sd + mu

    normal = standardize(rng.normal(size=T))
    # "卖期权"型：平时小赚，偶尔大亏（负偏、厚尾）
    crash = standardize(np.where(rng.random(T) < 0.02, rng.normal(-0.05, 0.01, T), rng.normal(0.0014, 0.005, T)))
    rows = {}
    for name, r in {"正态收益": normal, "负偏厚尾（卖期权型）": crash}.items():
        rows[name] = {"年化夏普": sharpe(r) * np.sqrt(252), "偏度": skew(r), "峰度": kurtosis(r, fisher=False),
                      "PSR(0)": psr(r), "bt_core.psr(0)": bt_core.psr(r, 0.0)}
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print("   ✅ 本课实现与 Phase 4 bt_core 对拍一致；同样的夏普，负偏厚尾的 PSR 更低")


# ==================== 第三部分：自己的一组试验 + DSR ====================


def build_trials(data):
    """Phase 7 的 8 个因子的全部非空子集（255 种）→ 等权合成 → Top20 → 日度超额收益"""
    panel = data.panel
    clean, p7 = mt.phase7_composite(data)
    names = list(clean)
    dates = data.dates
    cal = panel.dates
    R = panel.daily_ret.fillna(0.0)
    mask = data.mask_wide.reindex(index=dates, columns=panel.codes).fillna(False)
    ew = R.where(panel.daily_ret.notna()).mean(axis=1)
    # 信号在 d 收盘产生，按 d+1 收盘建仓（保守），持仓从 d+2 的日收益开始计入，直到下一个信号生效
    eff = cal[np.minimum(cal.get_indexer(dates) + 2, len(cal) - 1)]
    trials, labels = {}, []
    Z = {k: clean[k].reindex(index=dates, columns=panel.codes) for k in names}
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            comp = sum(Z[k].fillna(0.0) for k in combo) / size
            comp = comp.where(mask)
            top = comp.rank(axis=1, ascending=False, method="first") <= TOPK
            H = top.astype(float).set_axis(eff)
            H = H.reindex(cal).ffill().fillna(0.0)
            port = (H * R).sum(axis=1) / H.sum(axis=1).replace(0, np.nan)
            trials["+".join(combo)] = port - ew
            labels.append(combo)
    M = pd.DataFrame(trials).loc[eff[0]:dates[-1] + pd.Timedelta(days=31)].dropna()
    return M


def dsr_demo(M: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：在自己的一组试验上计算 DSR")
    print("=" * 60)
    print(f"""
    试验设计：Phase 7 的 8 个因子 → 全部 2⁸ − 1 = {M.shape[1]} 种子集 → 等权合成 → 每月 Top{TOPK} 等权
    → 相对全池等权的日度超额收益（{M.index[0].date()} ~ {M.index[-1].date()}，{len(M)} 天，未扣成本）
    这模拟了一个研究员"把因子组合都试一遍，挑最好的报告"。
    """)
    srs = M.mean() / M.std(ddof=1)
    best = srs.idxmax()
    r = M[best].to_numpy()
    n = M.shape[1]
    corr = M.corr().to_numpy()
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    n_eff = int(fcluster(linkage(squareform(dist, checks=False), "average"), t=0.5, criterion="distance").max())
    sr0_n = expected_max_sr(srs.var(ddof=1), n)
    sr0_eff = expected_max_sr(srs.var(ddof=1), max(n_eff, 2))
    print(f"📋 样本内最优组合：{best}")
    print(f"   年化超额夏普 {srs[best] * np.sqrt(252):.2f}；全部试验年化夏普：中位数 {srs.median() * np.sqrt(252):.2f}，"
          f"标准差 {srs.std() * np.sqrt(252):.2f}")
    print(f"   偏度 {skew(r):+.2f}，峰度 {kurtosis(r, fisher=False):.2f}")
    print(f"   试验之间高度相关：平均两两相关 {corr[np.triu_indices(n, 1)].mean():.2f}；"
          f"按相关性聚类（距离阈值 0.5）得到有效试验数 ≈ {n_eff}")
    rows = [
        {"基准": "PSR(0)：不考虑多次试验", "SR₀(年化)": 0.0, "概率": psr(r, 0.0), "bt_core": bt_core.psr(r, 0.0)},
        {"基准": f"DSR：N = {n}", "SR₀(年化)": sr0_n * np.sqrt(252), "概率": psr(r, sr0_n),
         "bt_core": bt_core.deflated_sharpe(r, srs.to_numpy())},
        {"基准": f"DSR：有效 N = {n_eff}", "SR₀(年化)": sr0_eff * np.sqrt(252), "概率": psr(r, sr0_eff),
         "bt_core": bt_core.deflated_sharpe(r, srs.to_numpy(), n_trials=max(n_eff, 2))},
    ]
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    verdict = "仍高于 0.95" if psr(r, sr0_eff) > 0.95 else "低于 0.95，不能排除'选出来的是运气'"
    print(f"""
    💡 不考虑试验次数时 PSR = {psr(r, 0.0):.3f}；用有效试验数修正后 DSR = {psr(r, sr0_eff):.3f}，{verdict}。
       N 取名义次数 {n} 还是有效次数 {n_eff}，结果差别很大 —— 所以要记录每一次尝试，并估计它们的相关性。
       注意：这里只是"选因子组合"这一个维度；真实研究中还有调仓频率、TopK、预处理方式、模型超参……
       全部计入后的 N 只会更大。
    """)
    return srs


# ==================== 第四部分：PBO ====================


def pbo_demo(M: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：PBO —— 样本内最优在样本外还好吗？（CSCV）")
    print("=" * 60)
    print(f"""
    1. 把 T 行收益切成 S = {S_BLOCKS} 块
    2. 枚举 C({S_BLOCKS},{S_BLOCKS // 2}) = 12870 种"一半块做样本内（IS）、另一半做样本外（OOS）"的划分：
         IS 夏普最高的策略 n* → 它在 OOS 中的相对排名 ω ∈ (0,1) → λ = ln(ω / (1−ω))
    3. PBO = P(λ ≤ 0)：样本内最优在样本外跌到中位数以下的比例
       PBO ≈ 0：选择过程有效；PBO ≥ 0.5：挑"最优"和随机挑差不多
    """)
    rng = np.random.default_rng(SEED)
    T, N = M.shape
    noise = rng.normal(0, M.to_numpy().std(), (T, N))
    planted = noise.copy()
    planted[:, 0] += M.to_numpy().std() * 0.15  # 只有第 0 个策略有真实的正期望（日夏普 0.15 ≈ 年化 2.4）
    cases = {"因子组合试验（Phase 7 面板）": M.to_numpy(), "对照 A：全部是纯噪声": noise,
             "对照 B：噪声中埋 1 个强策略": planted}
    rows, saved = [], {}
    for name, mat in cases.items():
        pbo, logits, pairs = pbo_cscv(mat)
        slope = np.polyfit(pairs[:, 0], pairs[:, 1], 1)[0]
        rows.append({"试验集": name, "PBO": pbo, "λ 中位数": np.median(logits),
                     "IS 最优的 IS 夏普(年化)": pairs[:, 0].mean() * np.sqrt(252),
                     "其 OOS 夏普(年化)": pairs[:, 1].mean() * np.sqrt(252), "OOS~IS 斜率": slope})
        saved[name] = (logits, pairs)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"""
    💡 两个对照组校验了方法：纯噪声的 PBO = {rows[1]['PBO']:.3f}（理论上应接近 0.5）；
       埋入一个强策略后 PBO = {rows[2]['PBO']:.3f} —— 明显下降，但在 255 个噪声策略中，强策略也并非每次都能在样本内胜出。
       因子组合试验的 PBO = {rows[0]['PBO']:.3f}：IS 最优组合在 OOS 的平均年化夏普
       {rows[0]['其 OOS 夏普(年化)']:+.2f}（IS 为 {rows[0]['IS 最优的 IS 夏普(年化)']:+.2f}），这个落差就是"选择"带来的乐观偏差。
       {'PBO ≥ 0.5：在这组试验里"挑样本内最优"并不比随机挑更可靠。' if rows[0]['PBO'] >= 0.5 else 'PBO < 0.5：样本内的排序在样本外有一定延续性，但这不代表最优组合的夏普可信 —— 还要看 DSR。'}
       注意 CSCV 假设各块可交换，样本外块可能位于样本内块之前 —— 它衡量的是"挑选过程"的可靠性，不替代 Walk-Forward。
    """)
    return saved


def other_methods():
    print("\n" + "=" * 60)
    print("第五部分：其他应对多重检验的方法")
    print("=" * 60)
    print("""
    ┌───────────────────────────┬──────────────────────────────────────────────┐
    │ 方法                      │ 说明                                         │
    ├───────────────────────────┼──────────────────────────────────────────────┤
    │ 提高 t 值门槛             │ Harvey, Liu & Zhu (2016)：新因子 t ≥ 3.0     │
    │ Bonferroni / Holm / BHY   │ 经典多重检验校正（FWER 或 FDR，Phase 4 第 08 课）│
    │ 保留最终测试集            │ 研究完全结束后只用一次                       │
    │ 预注册假设                │ 先写下假设和检验方法，再看数据               │
    │ 研究日志                  │ 记录每次尝试 → 才能算 N、DSR、PBO            │
    └───────────────────────────┴──────────────────────────────────────────────┘
    Phase 10 的 LLM 因子挖掘一次生成上百个因子 = 上百次试验，必须用本课的工具去伪。
    """)


def plot_results(saved):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (logits, _) in saved.items():
        axes[0].hist(logits, bins=40, alpha=0.5, label=name, density=True)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("CSCV 的 λ 分布（λ ≤ 0 的比例 = PBO）")
    axes[0].legend(fontsize=8)
    _, pairs = saved["因子组合试验（Phase 7 面板）"]
    axes[1].scatter(pairs[:, 0] * np.sqrt(252), pairs[:, 1] * np.sqrt(252), s=3, alpha=0.3)
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xlabel("IS 最优策略的 IS 夏普（年化）")
    axes[1].set_ylabel("同一策略的 OOS 夏普（年化）")
    axes[1].set_title("因子组合试验：样本内越好 ≠ 样本外越好")
    path = save_figure(fig, "08_pbo_cscv.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：第三、四部分的数字只用于跑通流程，不代表真实市场\n")
    selection_bias()
    psr_demo()
    M = build_trials(data)
    dsr_demo(M)
    saved = pbo_demo(M)
    other_methods()
    plot_results(saved)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 05 课 GBDT 的 12 组超参数（num_leaves × min_child_samples × learning_rate）当作试验，计算 DSR 与 PBO
    2. 把 S_BLOCKS 改为 8、10、12，PBO 估计是否稳定？块太少、太多各有什么问题？
    3. 在第三部分中把聚类阈值从 0.5 改为 0.3、0.7，有效试验数与 DSR 如何变化？
    4. 给第三部分的试验加上交易成本（07 课的 backtest_weights），最优组合是否改变？DSR 呢？
    """)


if __name__ == "__main__":
    main()
