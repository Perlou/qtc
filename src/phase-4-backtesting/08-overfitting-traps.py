"""
过拟合与多重检验
================

学习目标：
    1. 理解为什么"试得越多，越容易找到圣杯"：多重检验的数学
    2. 在纯随机游走上做均线参数搜索，亲手"挖"出高 Sharpe 策略，再看它在样本外崩溃
    3. 用 Bonferroni / Holm / BH 修正 p 值，理解 FWER 与 FDR 的区别
    4. 用 Deflated Sharpe Ratio（DSR）扣除"从 N 个策略中挑最好"的选择偏差
    5. 学会用参数热力图识别"参数平原"与"参数尖峰"

核心概念：
    - P(至少一次假阳性) = 1 − (1 − α)^M
    - 参数优化 = 多重检验：每一组参数都是一次检验
    - DSR = PSR(SR₀)，SR₀ 是"纯运气下 N 次试验的期望最大 Sharpe"

前置知识：
    - 03-performance-metrics.py（Sharpe 与 t 值）
    - Phase 3 假设检验与多重检验

环境要求：
    - pip install pandas numpy scipy statsmodels matplotlib akshare
    - 第二~四部分只用随机数（固定种子）；第五部分需要行情数据，无网络时可用 QTC_OFFLINE=1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bt_core import expected_max_sr, psr
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

FAST = np.arange(2, 31, 2)  # 15 个快线周期
SLOW = np.arange(10, 251, 10)  # 25 个慢线周期
COST = 0.0005
N_IS, N_OOS = 1000, 500  # 样本内 / 样本外天数（约 4 年 / 2 年）


# ==================== 第一部分：多重检验的数学 ====================


def multiple_testing_math():
    print("=" * 60)
    print("第一部分：试得越多，越容易'找到'圣杯")
    print("=" * 60)

    print("""
    🎓 比喻：一个学生把历年真题的答案全背下来（ABCDBCA...），在这些真题上次次满分，
       但换一套新题就只能乱猜。过拟合的策略，就是"背答案"的学生。

    每检验一个"无效"策略，在 5% 显著性下有 5% 的概率得到"显著"结果。试 M 次：
        P(至少一次假阳性) = 1 − (1 − α)^M
    """)
    rows = [[m, f"{1 - 0.95 ** m:.1%}"] for m in [1, 10, 20, 100, 375, 1000]]
    print(pd.DataFrame(rows, columns=["尝试次数 M", "至少一次假阳性（α=5%）"]).to_string(index=False))
    print(f"""
    你以为自己"只是优化了均线参数"？快线 {FAST[0]}~{FAST[-1]} × 慢线 {SLOW[0]}~{SLOW[-1]}
    = {sum(1 for f in FAST for s in SLOW if f < s)} 次检验。研究员每一次"改一下再试试"都是一次检验 —— 包括那些被删掉的代码。
    """)


# ==================== 第二部分：从纯噪声中挖出高 Sharpe ====================


def ma_grid_returns(price: np.ndarray, cost: float = COST) -> tuple[np.ndarray, list]:
    """对一条价格序列跑整个均线参数网格，返回 (日收益矩阵 [组合 × 天], 参数列表)"""
    s = pd.Series(price)
    ret = s.pct_change().fillna(0.0).to_numpy()
    windows = sorted(set(FAST) | set(SLOW))
    ma = {w: s.rolling(w).mean().to_numpy() for w in windows}
    combos = [(f, sl) for f in FAST for sl in SLOW if f < sl]
    out = np.empty((len(combos), len(price)))
    for k, (f, sl) in enumerate(combos):
        sig = (ma[f] > ma[sl]).astype(float)  # NaN 比较为 False → 预热期空仓
        pos = np.concatenate([[0.0], sig[:-1]])  # 信号滞后一天
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        out[k] = pos * ret - turnover * cost
    return out, combos


def ann_sharpe(r: np.ndarray, axis=-1) -> np.ndarray:
    sd = r.std(axis=axis, ddof=1)
    return np.where(sd > 0, r.mean(axis=axis) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)


def random_walk(seed: int, n: int) -> np.ndarray:
    """零漂移随机游走：任何择时策略的真实 Sharpe 都是 0"""
    rng = np.random.default_rng(seed)
    return 10 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))


def mine_noise() -> dict:
    print("\n" + "=" * 60)
    print("第二部分：从纯噪声中'挖'出高 Sharpe")
    print("=" * 60)

    # ---- 实验 A：1000 个相互独立的"策略"，日收益全是均值为 0 的噪声 ----
    rng = np.random.default_rng(42)
    n_days, n_strat = 252 * 3, 1000
    noise = rng.normal(0, 0.01, size=(n_strat, n_days))
    sr = ann_sharpe(noise)
    print(f"""
    实验 A：{n_strat} 个"策略"，每个都是 {n_days} 天（3 年）均值为 0 的纯噪声日收益（种子 42，真实 Sharpe = 0）
      最好的年化 Sharpe：{sr.max():.2f}
      Sharpe > 1 的策略：{(sr > 1).sum()} 个；Sharpe > 1.5 的策略：{(sr > 1.5).sum()} 个
      理论参考：3 年样本的 Sharpe 估计标准误 ≈ 1/√3 ≈ {1 / np.sqrt(3):.2f}，1000 次中最大值约在 +3 个标准误附近
    """)

    # ---- 实验 B：100 位研究员，各自在一条随机游走上调均线参数 ----
    best_is, best_oos, best_param, corr = [], [], [], []
    for seed in range(100):
        r, combos = ma_grid_returns(random_walk(seed, N_IS + N_OOS))
        s_is, s_oos = ann_sharpe(r[:, :N_IS]), ann_sharpe(r[:, N_IS:])
        k = int(np.argmax(s_is))
        best_is.append(s_is[k])
        best_oos.append(s_oos[k])
        best_param.append(tuple(int(x) for x in combos[k]))
        corr.append(np.corrcoef(s_is, s_oos)[0, 1])
    best_is, best_oos = np.array(best_is), np.array(best_oos)
    lucky = int(np.argmax(best_is))
    print(f"""    实验 B：100 位研究员，每人拿到一条零漂移随机游走（种子 0~99），前 {N_IS} 天调参、后 {N_OOS} 天检验，
           网格 {len(combos)} 组 (快线, 慢线)，单边成本 {COST * 1e4:.0f} bp
      每人"样本内最优"的 Sharpe：平均 {best_is.mean():.2f}，中位数 {np.median(best_is):.2f}，最高 {best_is.max():.2f}
      同一组参数在样本外：      平均 {best_oos.mean():.2f}，样本外 Sharpe > 0 的比例 {(best_oos > 0).mean():.0%}
      每条路径上 357 组参数的 IS/OOS Sharpe 相关系数：平均 {np.mean(corr):+.2f}

      🏆 最幸运的研究员（种子 {lucky}）：参数 {best_param[lucky]}，样本内 Sharpe {best_is[lucky]:.2f} → 样本外 {best_oos[lucky]:.2f}
         他会拿着这份"Sharpe {best_is[lucky]:.1f}"的回测来找你 —— 而你看不到另外 99 个人的结果。

    💡 真实 Sharpe 全是 0，但"挑最好的"总能挑出好看的数字；到了样本外，它回到运气水平。
       如果你只汇报最好的那个，你汇报的就是运气。
    """)
    return {"noise": noise, "sr": sr}


# ==================== 第三部分：多重检验修正 ====================


def multiple_testing_correction(res: dict):
    print("\n" + "=" * 60)
    print("第三部分：多重检验修正——Bonferroni / Holm / BH（实验 A 的 1000 个独立噪声策略）")
    print("=" * 60)

    r = res["noise"]
    T = r.shape[1]
    t_stat = r.mean(axis=1) / r.std(axis=1, ddof=1) * np.sqrt(T)
    pvals = 1 - stats.t.cdf(t_stat, df=T - 1)  # 单侧检验：H0 均值 ≤ 0
    rej_b, _, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    rej_h, _, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    rej_bh, _, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    print("""
    ┌──────────────┬──────────────────────────────┬──────────────────────────┐
    │ 方法         │ 控制目标                     │ 做法                     │
    ├──────────────┼──────────────────────────────┼──────────────────────────┤
    │ Bonferroni   │ FWER（至少一个假阳性的概率） │ 门槛变为 α / M           │
    │ Holm         │ FWER                         │ p 值从小到大逐步比较     │
    │ BH           │ FDR（显著结果中假阳性比例）  │ p(k) ≤ k/M × α           │
    │ t > 3 经验法 │ —                            │ Harvey, Liu & Zhu (2016) │
    └──────────────┴──────────────────────────────┴──────────────────────────┘
    """)
    M = len(pvals)
    rows = [["未修正 p < 0.05", int((pvals < 0.05).sum())], [f"Bonferroni（门槛 {0.05 / M:.0e}）", int(rej_b.sum())],
            ["Holm", int(rej_h.sum())], ["Benjamini-Hochberg", int(rej_bh.sum())], ["t > 3", int((t_stat > 3).sum())]]
    print(pd.DataFrame(rows, columns=[f"判定方式（{M} 个纯噪声策略）", "'显著'个数"]).to_string(index=False))
    print(f"""
    最优策略：t = {t_stat.max():.2f}，未修正 p = {pvals.min():.5f}（理论上约 5% 即 {M * 0.05:.0f} 个会"显著"）
    💡 这里的 1000 个策略相互独立，正好满足这些方法的假设。
       现实中相邻参数的策略高度相关，"有效独立试验次数"远小于名义次数：
       Bonferroni 会过于保守，严格估计需要聚类等方法（AFML 相关章节）。
    """)


# ==================== 第四部分：Deflated Sharpe Ratio ====================


def deflated_sharpe_demo(res: dict):
    print("\n" + "=" * 60)
    print("第四部分：Deflated Sharpe Ratio——扣除'挑选'效应")
    print("=" * 60)

    print("""
    核心思想（Bailey & López de Prado）：N 次独立试验中，即使真实 Sharpe 全为 0，最大的那个也会显著大于 0。
    所以评判"最好的策略"时，比较基准不应是 0，而应是"纯运气下的期望最大 Sharpe" SR₀：

      SR₀ = √V[SR] × [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]       γ ≈ 0.5772
      PSR(SR*) = Φ( (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )   （SR 不年化，γ₃ 偏度，γ₄ 峰度）
      DSR = PSR(SR₀)
    """)
    r = res["noise"]
    daily_sr = r.mean(axis=1) / r.std(axis=1, ddof=1)
    k = int(np.argmax(daily_sr))
    best = r[k]
    rows = []
    for n_eff in [1, 10, 100, len(daily_sr)]:
        sr0 = 0.0 if n_eff == 1 else expected_max_sr(daily_sr, n_trials=n_eff)
        rows.append([n_eff, f"{sr0 * np.sqrt(252):.2f}", f"{psr(best, sr0):.3f}"])
    print(f"   实验 A 中最好的策略：年化 Sharpe {daily_sr[k] * np.sqrt(252):.2f}，PSR(0) = {psr(best, 0.0):.3f}"
          f"（假装只试了一次时，'真实 Sharpe > 0'的概率看起来很高）")
    print(f"   实际 1000 个策略的最大年化 Sharpe {daily_sr.max() * np.sqrt(252):.2f}，"
          f"公式给出的期望最大值 SR₀ = {expected_max_sr(daily_sr) * np.sqrt(252):.2f}")
    print(pd.DataFrame(rows, columns=["申报的试验次数 N", "SR₀（年化）", "DSR"]).to_string(index=False))
    print("""
    💡 解读：
    - N = 1 时 DSR 就是 PSR(0)
    - N 越大，"运气基准" SR₀ 越高，DSR 越低；如实申报 N = 1000 后，最好的噪声策略不再显著
    - DSR 接近 1（如 > 0.95）才说明策略大概率不是运气 —— 前提是你如实记录了尝试次数
    - Phase 9 08-deflated-sharpe-pbo.py 会进一步讲 PBO（回测过拟合概率）
    """)


# ==================== 第五部分：真实数据上的参数热力图 ====================


def heatmap_real():
    print("\n" + "=" * 60)
    print("第五部分：真实数据上的参数热力图——平原还是尖峰？")
    print("=" * 60)

    close = get_index_daily("000300", "20150101", "20241231")["close"].to_numpy()
    rets, combos = ma_grid_returns(close)
    sr = ann_sharpe(rets)
    daily_sr = rets.mean(axis=1) / rets.std(axis=1, ddof=1)
    grid = pd.DataFrame(np.nan, index=FAST, columns=SLOW)
    for (f, s), v in zip(combos, sr):
        grid.loc[f, s] = v
    k = int(np.argmax(sr))
    bf, bs = (int(x) for x in combos[k])
    i, j = list(FAST).index(bf), list(SLOW).index(bs)
    neigh = grid.iloc[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
    print(f"""
    沪深 300 指数 2015-2024，{len(combos)} 组双均线（单边 {COST * 1e4:.0f} bp）：
      全样本最优 ({bf}, {bs})：Sharpe {sr[k]:.2f}
      它周围 3×3 邻域的平均 Sharpe：{np.nanmean(neigh.to_numpy()):.2f}，最低 {np.nanmin(neigh.to_numpy()):.2f}
      全部组合 Sharpe 中位数 {np.median(sr):.2f}，Sharpe > 0 的比例 {(sr > 0).mean():.0%}
      DSR（N = {len(combos)}）= {psr(rets[k], expected_max_sr(daily_sr)):.3f}，PSR(0) = {psr(rets[k], 0):.3f}

    💡 最优点周围是一片"平原"（邻域 Sharpe 接近）还是"尖峰"（稍微变动就崩），比最优值本身更重要；
       选参数应选平原中心，而不是最高点。是否可信，还要看 09 课的 Walk-Forward。
    """)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=-np.nanmax(np.abs(grid.to_numpy())), vmax=np.nanmax(np.abs(grid.to_numpy())))
    ax.set_xticks(range(len(SLOW)), SLOW, fontsize=8)
    ax.set_yticks(range(len(FAST)), FAST, fontsize=8)
    ax.scatter([j], [i], marker="*", s=200, color="black", label=f"最优 ({bf}, {bs})")
    ax.set_xlabel("慢线周期")
    ax.set_ylabel("快线周期")
    ax.set_title("沪深 300 双均线参数热力图（Sharpe）")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Sharpe")
    print(f"📁 图片已保存：{save_figure(fig, '08_param_heatmap.png', subdir='phase-4')}")


# ==================== 第六部分：识别与对抗过拟合 ====================


def anti_overfitting():
    print("\n" + "=" * 60)
    print("第六部分：识别与对抗过拟合")
    print("=" * 60)

    print("""
    ┌──────────────────────┬───────────────────────────────────────────────┐
    │ 过拟合信号           │ 说明                                          │
    ├──────────────────────┼───────────────────────────────────────────────┤
    │ 参数尖峰             │ 最优参数稍微变动，绩效急剧下降                │
    │ 规则过多             │ 十几个条件和阈值                              │
    │ 样本外大幅衰减       │ 样本内 Sharpe 2.0，样本外 0.3                 │
    │ 只在特定区间有效     │ 收益集中在某一年或某几个月                    │
    │ 无法用经济逻辑解释   │ "每月第三个周二买入 PE 在 13.7~15.2 的股票"   │
    │ 交易次数很少         │ 30 笔交易的胜率和盈亏比没有统计意义           │
    └──────────────────────┴───────────────────────────────────────────────┘

    对抗原则：
    1. 先有假设，后有回测（经济逻辑先行）
    2. 少参数：每多一个参数，就多一个过拟合的自由度
    3. 记录所有尝试（研究日志），报告时披露尝试次数，并用 DSR 等方法修正
    4. 保留最终样本外：研究结束前绝对不碰，只用一次
    5. 选"参数平原"的中心，而不是最高点
    6. 跨市场 / 跨品种验证
    """)


# ==================== 主函数 ====================


def main():
    multiple_testing_math()
    res = mine_noise()
    multiple_testing_correction(res)
    deflated_sharpe_demo(res)
    heatmap_real()
    anti_overfitting()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 在第二部分的网格中再加一个"止损比例"维度（1%~10%），样本内最优 Sharpe 会变高还是变低？样本外呢？
    2. 用组合收益的相关矩阵估计"有效独立试验数"：对相关矩阵做特征分解，
       取解释 95% 方差所需的特征值个数作为 N，重新计算 DSR
    3. 把随机游走换成带微弱正漂移（年化 5%）的序列，多重检验修正后还能发现"真"信号吗？需要多长的样本？
    4. 为你自己的研究建立一个研究日志模板（日期、假设、参数、样本区间、结果、是否看过样本外）
    """)


if __name__ == "__main__":
    main()
