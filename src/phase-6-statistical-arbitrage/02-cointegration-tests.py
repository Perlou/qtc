"""
Engle-Granger 与 Johansen 协整检验
==================================

学习目标：
    1. 分清"相关"与"协整"：前者看收益同步，后者看价格的长期均衡
    2. 用 ADF 检验判断序列是否平稳，理解 I(1) 与 I(0)
    3. 掌握 Engle-Granger 两步法，并用蒙特卡洛验证：
       对 OLS 残差直接用 adfuller 的临界值会"过于乐观"，必须用 coint()（MacKinnon 协整临界值）
    4. 用 Johansen 检验多个资产之间的协整秩，构造平稳的组合

核心概念：
    - 平稳（I(0)）：均值、方差不随时间变化；随机游走价格是 I(1)，差分一次才平稳
    - 协整：P^A、P^B 都是 I(1)，但存在 β 使 P^A - β·P^B 平稳
    - Engle-Granger：OLS 估计 β → 对残差做单位根检验（协整临界值）
    - Johansen：基于 VECM，Π 矩阵的秩 r = 协整关系个数；迹统计量 / 最大特征值统计量

前置知识：
    - 01-pairs-trading-intro.py；Phase 3 ADF 检验与 OLS

环境要求：
    - pip install akshare pandas numpy statsmodels matplotlib
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
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stat_arb_utils import BANKS, FORM_END, PAIR, half_life, load_bank_panel, ols_hedge
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)  # Johansen 特征值偶有极小虚部


# ==================== 第一部分：相关 ≠ 协整 ====================


def corr_vs_coint() -> dict:
    print("=" * 60)
    print("第一部分：相关 ≠ 协整")
    print("=" * 60)

    print("""
    ┌──────────┬────────────────────────────┬────────────────────────────┐
    │          │ 相关性（Correlation）      │ 协整（Cointegration）      │
    ├──────────┼────────────────────────────┼────────────────────────────┤
    │ 衡量对象 │ 收益率的同向变动           │ 价格水平之间的长期均衡     │
    │ 交易含义 │ 今天一起涨跌               │ 走远了会回来               │
    └──────────┴────────────────────────────┴────────────────────────────┘
    下面用两组模拟数据（固定随机种子）说明二者可以完全"错位"：
    """)
    rng = np.random.default_rng(7)
    n = 1000
    # 情形 1：共同冲击很大（日收益高度相关），但各自还有一个随机游走分量和不同的漂移 → 不协整
    common = rng.normal(0, 0.015, n)
    a1 = np.cumsum(common + rng.normal(0.0008, 0.006, n))
    b1 = np.cumsum(common + rng.normal(-0.0008, 0.006, n))
    # 情形 2：B 随机游走，A = B + 平稳噪声（日噪声大）→ 日收益相关性一般，但协整
    b2 = np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.5 * noise[t - 1] + rng.normal(0, 0.02)
    a2 = b2 + noise

    rows = []
    for label, a, b in [("高相关、不协整", a1, b1), ("低相关、协整", a2, b2)]:
        rows.append([label, np.corrcoef(np.diff(a), np.diff(b))[0, 1], coint(a, b)[1], np.std(a - b)])
    print(pd.DataFrame(rows, columns=["情形", "日收益相关系数", "协整检验 p 值", "价差标准差"]).round(4).to_string(index=False))
    print("""
    💡 配对交易需要的是协整：日收益相关再高，价差也可能越走越远；
       反过来，日收益相关性一般的两条序列，价差也可能始终围绕常数波动。
    """)
    return {"a1": a1, "b1": b1, "a2": a2, "b2": b2}


# ==================== 第二部分：平稳性与 ADF 回顾 ====================


def adf_review(logp: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：平稳性与 ADF 检验回顾")
    print("=" * 60)

    print("""
    ADF 回归：Δs_t = α + γ · s_{t-1} + Σ φ_i · Δs_{t-i} + ε_t
      H0: γ = 0（有单位根，不平稳）；p < 0.05 → 拒绝 H0，认为平稳
      γ < 0 表示"偏高之后倾向于下降"，即均值回归
    """)
    rows = []
    for s in ["601398", "601939", "601288", "601988"]:
        rows.append([BANKS[s], adfuller(logp[s].loc[:FORM_END])[1], adfuller(logp[s])[1],
                     adfuller(logp[s].diff().dropna())[1]])
    df = pd.DataFrame(rows, columns=["股票", "对数价格 p（形成期）", "对数价格 p（全样本）", "日收益 p（全样本）"])
    print(df.round(4).to_string(index=False))
    print("""
    💡 价格通常是 I(1)（不平稳），差分一次后是 I(0)（平稳）。
       协整要找的是：两个 I(1) 序列的某个线性组合是 I(0)。""")
    n_form = int((df["对数价格 p（形成期）"] < 0.05).sum())
    if n_form:
        print(f"""
    ⚠️ 注意：形成期内有 {n_form} 只股票的对数价格本身就拒绝了单位根（p < 0.05）。
       有限样本里，一段长期区间震荡的价格可能"看起来平稳"，但换成全样本就不再成立。
       这提醒我们：单位根 / 协整检验的结论强烈依赖样本区间，而且协整检验的前提（各序列为 I(1)）
       在短样本里未必满足 —— 对检验结果要保持怀疑。""")
    print()


# ==================== 第三部分：Engle-Granger 两步法 ====================


def engle_granger(logp: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：Engle-Granger 两步法")
    print("=" * 60)

    a, b = PAIR
    form = logp.loc[:FORM_END]
    alpha, beta = ols_hedge(form[a], form[b])
    resid = form[a] - alpha - beta * form[b]
    t_stat, p_eg, crit = coint(form[a], form[b])
    p_adf = adfuller(resid)[1]
    print(f"""
    第一步：OLS  log P({BANKS[a]}) = α + β · log P({BANKS[b]}) + u
            α = {alpha:.4f}，β = {beta:.4f}
    第二步：对残差 u 做单位根检验
            coint() 统计量 = {t_stat:.3f}，协整临界值 1%/5%/10% = {np.round(crit, 3).tolist()}
            coint() p 值                   = {p_eg:.4f}   ← 正确做法
            adfuller(残差) p 值            = {p_adf:.4f}   ← ❌ 过于乐观
            反方向 coint({BANKS[b]} ~ {BANKS[a]}) p 值 = {coint(form[b], form[a])[1]:.4f}
            残差半衰期 ≈ {half_life(resid):.1f} 个交易日
    """)

    print("🎲 为什么不能直接用 adfuller？蒙特卡洛：300 对独立随机游走（真实情况：都不协整）")
    rng = np.random.default_rng(1994)
    hits_adf = hits_eg = 0
    n_sim, n = 300, 500
    for _ in range(n_sim):
        y = np.cumsum(rng.normal(0, 1, n))
        x = np.cumsum(rng.normal(0, 1, n))
        X = np.column_stack([np.ones(n), x])
        u = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        hits_adf += adfuller(u)[1] < 0.05
        hits_eg += coint(y, x)[1] < 0.05
    print(f"   adfuller(OLS 残差) 拒绝率：{hits_adf / n_sim:.1%}   ← 名义水平 5%，实际假阳性远高于 5%")
    print(f"   coint()            拒绝率：{hits_eg / n_sim:.1%}   ← 接近名义水平 5%")
    print("""
    原因：OLS 已经挑选了"让残差尽可能小"的 β，残差天然看起来比真实情况更平稳，
    所以需要更严格的协整临界值（MacKinnon），statsmodels 的 coint() 已经处理了这一点。
    """)

    print("📋 Engle-Granger 的局限：对样本区间敏感（每 2 年一个窗口）")
    rows = []
    for y0 in range(logp.index[0].year, logp.index[-1].year):
        w = logp.loc[str(y0):str(y0 + 1)]
        _, bw = ols_hedge(w[a], w[b])
        rows.append([f"{y0}-{y0 + 1}", bw, coint(w[a], w[b])[1], coint(w[b], w[a])[1]])
    print(pd.DataFrame(rows, columns=["窗口", "β", f"p({a}~{b})", f"p({b}~{a})"]).round(3).to_string(index=False))

    raw = np.exp(form)
    _, beta_lvl = ols_hedge(raw[a], raw[b])
    print(f"""
    📋 价格 vs 对数价格：价格回归 β = {beta_lvl:.3f}（股数比，受复权基准影响），p = {coint(raw[a], raw[b])[1]:.4f}；
                        对数回归 β = {beta:.3f}（金额比的近似），p = {p_eg:.4f}
       两者结论可能不同，要与实际交易方式（按股数还是按金额配比）保持一致。
    """)


# ==================== 第四部分：Johansen 检验 ====================


def johansen_report(data: pd.DataFrame, title: str):
    res = coint_johansen(data, det_order=0, k_ar_diff=1)  # det_order=0：含常数项
    rows = []
    rank = None
    for r in range(data.shape[1]):
        trace, cv_t = res.lr1[r], res.cvt[r, 1]
        maxe, cv_m = res.lr2[r], res.cvm[r, 1]
        rows.append([f"r ≤ {r}", trace, cv_t, "拒绝" if trace > cv_t else "不拒绝", maxe, cv_m])
        if rank is None and trace <= cv_t:
            rank = r
    rank = data.shape[1] if rank is None else rank
    print(f"\n📋 {title}")
    cols = ["原假设", "迹统计量", "95%临界值", "迹检验", "最大特征值统计量", "95%临界值 "]
    print(pd.DataFrame(rows, columns=cols).round(2).to_string(index=False))
    print(f"   → 迹检验判定的协整秩 r = {rank}")
    if rank == data.shape[1]:
        print("   r = n：各序列本身就被判为平稳，此时谈不上\"协整\"（对照第二部分的 ADF 结果）")
    elif rank == 0:
        print("   r = 0：没有发现协整关系")
    else:
        print(f"   0 < r < n：存在 {rank} 个协整关系")
    return res, rank


def johansen_test(logp: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 60)
    print("第四部分：Johansen 检验 —— 多个资产、几个协整关系")
    print("=" * 60)

    print("""
    VECM：ΔP_t = Π · P_{t-1} + Σ Γ_i · ΔP_{t-i} + ε_t
      Π 的秩 r = 协整关系个数：r = 0 无协整；0 < r < n 有 r 个协整向量
      从 r = 0 开始依次检验，第一个"不拒绝"的 r 就是协整秩
      优点：不依赖谁是因变量，能处理 3 个及以上资产
    """)

    def simulate(seed: int, n: int = 1000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1 = np.cumsum(rng.normal(0, 0.01, n))
        ou = np.zeros(n)
        for t in range(1, n):
            ou[t] = 0.9 * ou[t - 1] + rng.normal(0, 0.01)
        return pd.DataFrame({"X1": x1, "X2": 0.8 * x1 + ou, "X3": np.cumsum(rng.normal(0, 0.01, n))})

    johansen_report(simulate(0), "先验证：模拟 3 个资产（X2 与 X1 协整，X3 独立随机游走），真实协整秩 = 1")
    ranks = []
    for seed in range(50):
        r = coint_johansen(simulate(seed), det_order=0, k_ar_diff=1)
        ranks.append(next((i for i in range(3) if r.lr1[i] <= r.cvt[i, 1]), 3))
    counts = pd.Series(ranks).value_counts().sort_index()
    print("   重复 50 次模拟，判定的协整秩分布：" + "，".join(f"r={k}: {v} 次" for k, v in counts.items()))
    print("   → 检验本身也会犯错：即使真实秩已知，个别样本也会给出别的结论")

    big4 = ["601398", "601939", "601288", "601988"]
    names = "、".join(BANKS[s] for s in big4)
    johansen_report(logp.loc[:FORM_END, big4], f"四大行（{names}）形成期对数价格")
    full = logp[big4]
    res, rank = johansen_report(full, f"四大行（{names}）全样本对数价格")
    w = np.real(res.evec[:, 0] / res.evec[0, 0])  # 第一个（最强）协整向量，归一化使第一个权重为 1
    basket = pd.Series(full.to_numpy() @ w, index=full.index)
    print("\n   第一个协整向量（归一化）：" + "，".join(f"{BANKS[s]} {wi:+.3f}" for s, wi in zip(big4, w)))
    print(f"   组合 ADF p 值 = {adfuller(basket)[1]:.4f}，半衰期 ≈ {half_life(basket):.1f} 天")
    print("   ⚠️ 这里用全样本估计协整向量只是为了演示检验本身；用于交易时必须在形成期估计、交易期使用")
    print("""
    💡 Johansen 适合构造三个以上资产的平稳组合（如一篮子同行业股票、"螺纹 + 铁矿 + 焦炭"利润价差）。
       但资产越多，交易成本越高、过拟合风险越大；协整秩为 0 时，第一个特征向量构造的组合也不保证平稳。
    """)
    return basket


def plot_results(sims: dict, basket: pd.Series, is_synth: bool):
    setup_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    axes[0].plot(sims["a1"], label="A"), axes[0].plot(sims["b1"], label="B")
    axes[0].set_title("高相关、不协整：每天同涨同跌，但差距越来越大")
    axes[1].plot(sims["a2"], label="A", alpha=0.8), axes[1].plot(sims["b2"], label="B", alpha=0.8)
    axes[1].set_title("低相关、协整：各自乱走，但始终拴在一起")
    for ax in axes[:2]:
        ax.legend()
    axes[2].plot(basket.index, basket - basket.mean())
    axes[2].axhline(0, color="gray", lw=1)
    axes[2].set_title(f"四大行 Johansen 组合（全样本估计，去均值）{'【合成数据】' if is_synth else ''}")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '02_cointegration.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    prices, _, is_synth = load_bank_panel()
    logp = np.log(prices)

    sims = corr_vs_coint()
    adf_review(logp)
    engle_granger(logp)
    basket = johansen_test(logp)
    plot_results(sims, basket, is_synth)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把蒙特卡洛的样本长度 n 从 500 改为 100 和 2000，adfuller(残差) 的拒绝率如何变化？
    2. 对 01 课选出的配对，分别用价格和对数价格做 Engle-Granger，结论是否一致？
    3. 在 Johansen 检验中把 det_order 改为 -1（无常数项）和 1（含线性趋势），协整秩会变吗？
    4. 用四大行的交易期（2022 年以后）数据重做 Johansen，形成期的协整向量还能构造平稳组合吗？
    """)


if __name__ == "__main__":
    main()
