"""
协方差矩阵、特征分解与 PCA
==========================

学习目标：
    1. 理解组合风险 σ_p² = wᵀΣw，以及相关系数如何决定分散化效果
    2. 理解特征分解 Σ = VΛVᵀ：特征向量是"方向"（一组权重），特征值是该方向的方差
    3. 用 PCA 从一篮子股票中提取"市场因子"（PC1），并与沪深300 对照
    4. 认识协方差矩阵的估计误差：N 接近 T 时样本协方差噪声极大，Ledoit-Wolf 收缩可缓解
    5. 知道主成分符号任意、需要对齐

核心概念：
    - 协方差矩阵 Σ：对角线是方差 σ_i²，非对角线是 σ_ij = ρ_ij σ_i σ_j
    - PCA：按解释方差从大到小找正交方向；解释比例 = λ_k / Σλ_i
    - PC1 ≈ 市场因子；PC2、PC3 常对应风格或行业轮动
    - 收缩估计（Shrinkage）：把样本协方差向一个结构化目标"拉"一点

前置知识：
    - 06-linear-regression-ols.py、07-correlation-vs-cointegration.py；矩阵乘法基础

环境要求：
    - pip install numpy pandas scikit-learn matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据中股票彼此独立，没有共同因子；第二部分的模拟数据不受影响）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from utils.data import get_close_panel, get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

TRADING_DAYS = 252
START, END = "20190101", "20241231"
STOCKS = {
    "600519": "贵州茅台", "000858": "五粮液", "600887": "伊利股份", "000333": "美的集团", "000651": "格力电器",
    "601888": "中国中免", "600036": "招商银行", "601166": "兴业银行", "601398": "工商银行", "601318": "中国平安",
    "600030": "中信证券", "000002": "万科A", "600900": "长江电力", "601088": "中国神华", "600309": "万华化学",
    "600031": "三一重工", "600276": "恒瑞医药", "002415": "海康威视", "000725": "京东方A", "601012": "隆基绿能",
}


def pca_eig(rets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """对标准化收益（即相关矩阵）做特征分解，返回降序排列的特征值与特征向量"""
    z = (rets - rets.mean()) / rets.std()
    eigvals, eigvecs = np.linalg.eigh(np.cov(z.T))  # eigh：对称矩阵专用，特征值升序
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    # 符号对齐：特征向量乘以 -1 仍是特征向量，统一让权重之和为正
    eigvecs = eigvecs * np.sign(eigvecs.sum(axis=0))
    return eigvals, eigvecs


# ==================== 第一部分：协方差矩阵与组合风险 ====================


def portfolio_risk(rets: pd.DataFrame):
    print("=" * 60)
    print("第一部分：协方差矩阵与组合风险")
    print("=" * 60)

    print("""
    组合方差：σ_p² = wᵀ Σ w
    两资产时：σ_p² = w₁²σ₁² + w₂²σ₂² + 2·w₁·w₂·ρ₁₂·σ₁·σ₂
       → 相关系数 ρ 越低，组合波动越小 —— 这就是"分散化"的数学本质
    """)

    s1, s2, w1 = 0.30, 0.30, 0.5
    print(f"📐 两只年化波动都是 {s1:.0%} 的股票，各 50%：")
    for rho in [1.0, 0.7, 0.3, 0.0, -0.5]:
        vol = np.sqrt(w1**2 * s1**2 + (1 - w1) ** 2 * s2**2 + 2 * w1 * (1 - w1) * rho * s1 * s2)
        print(f"   ρ = {rho:+.1f} → 组合波动 {vol:.1%}")

    cov = rets.cov() * TRADING_DAYS
    n = len(cov)
    w = np.ones(n) / n
    port_vol = np.sqrt(w @ cov.to_numpy() @ w)
    avg_vol = np.sqrt(np.diag(cov)).mean()
    corr = rets.corr().to_numpy()
    avg_corr = corr[~np.eye(n, dtype=bool)].mean()
    print(f"\n📊 {n} 只 A 股等权组合（{rets.index[0].date()} ~ {rets.index[-1].date()}）：")
    print(f"   个股平均年化波动 = {avg_vol:.1%}，两两平均相关 = {avg_corr:.3f}")
    print(f"   等权组合年化波动 = {port_vol:.1%}（wᵀΣw 计算）")
    print(f"   直接用组合日收益验证 = {(rets @ w).std() * np.sqrt(TRADING_DAYS):.1%}")
    # 极限：N → ∞ 时，等权组合方差 → 平均协方差，分散化消除不了共同部分
    floor = np.sqrt(avg_corr) * avg_vol
    print(f"   粗略估计：股票数无穷多时，组合波动下限 ≈ √ρ̄ × σ̄ = {floor:.1%}（分散不掉的'系统性风险'）")
    if is_synthetic(rets.index):
        print("   ⚠️ 离线合成数据中股票彼此独立（ρ ≈ 0），分散化效果会被夸大")


# ==================== 第二部分：特征分解的直觉（模拟一因子市场） ====================


def eigen_intuition():
    print("\n" + "=" * 60)
    print("第二部分：特征分解 —— 找到'主要的运动方向'")
    print("=" * 60)

    print("""
    🎼 比喻：几百只股票像一支乐队。指挥（市场）的手势决定了大部分乐手的动作，
       然后是声部（行业、风格），最后才是每个乐手的即兴发挥。PCA 就是从录音中分离出"指挥"和"声部"。

    Σ = V Λ Vᵀ
       V 的列是特征向量（相互正交的方向，每个都是一组股票权重）
       Λ 对角线是特征值 λ₁ ≥ λ₂ ≥ ...，表示沿该方向的方差
    """)

    # 模拟：12 只股票 = 市场因子 + 两个"板块"因子 + 个股噪声
    rng = np.random.default_rng(0)
    T, n = 1000, 12
    mkt = rng.normal(0, 0.012, T)
    sector_a, sector_b = rng.normal(0, 0.008, T), rng.normal(0, 0.008, T)
    beta = rng.uniform(0.8, 1.2, n)
    load_a = np.r_[np.ones(6), np.zeros(6)]
    load_b = 1 - load_a
    rets = (np.outer(mkt, beta) + np.outer(sector_a, load_a) + np.outer(sector_b, load_b)
            + rng.normal(0, 0.01, (T, n)))
    df = pd.DataFrame(rets, columns=[f"{'A' if i < 6 else 'B'}{i % 6 + 1}" for i in range(n)])

    cov = df.cov().to_numpy()
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    recon_err = np.abs(eigvecs @ np.diag(eigvals) @ eigvecs.T - cov).max()
    print(f"🧪 模拟 12 只股票 = 市场因子 + 板块因子（A 组 6 只、B 组 6 只）+ 个股噪声")
    print(f"   验证 Σ = VΛVᵀ：最大重构误差 = {recon_err:.1e}")
    print(f"   验证正交：|VᵀV - I| 最大值 = {np.abs(eigvecs.T @ eigvecs - np.eye(n)).max():.1e}")
    v1 = eigvecs[:, 0]
    print(f"   验证'沿特征向量方向的组合方差 = 特征值'：v₁ᵀΣv₁ = {v1 @ cov @ v1:.3e}，λ₁ = {eigvals[0]:.3e}")

    eigvals_c, eigvecs_c = pca_eig(df)
    explained = eigvals_c / eigvals_c.sum()
    print(f"\n   前 4 个主成分解释比例：{np.round(explained[:4], 3).tolist()}")
    loadings = pd.DataFrame(eigvecs_c[:, :2], index=df.columns, columns=["PC1", "PC2"]).round(2)
    print("   前两个主成分的权重（载荷）：")
    print("   " + loadings.T.to_string().replace("\n", "\n   "))
    print("""
    💡 PC1 权重全部同号 → 所有股票一起涨跌 = 市场因子
       PC2 权重在 A 组和 B 组符号相反 → "A 板块 vs B 板块"的轮动因子
       PCA 没有被告诉任何分组信息，它是从协方差结构里自己"发现"的
    """)


# ==================== 第三部分：真实 A 股的 PCA ====================


def real_pca(rets: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：真实 A 股的 PCA")
    print("=" * 60)

    eigvals, eigvecs = pca_eig(rets)
    explained = eigvals / eigvals.sum()
    print(f"📊 {rets.shape[1]} 只股票，{rets.shape[0]} 个交易日（先标准化 → 等价于对相关矩阵做 PCA）")
    print("   前 5 个主成分解释比例：" + "  ".join(f"PC{i + 1}: {e:.1%}" for i, e in enumerate(explained[:5])))
    print(f"   前 3 个累计解释：{explained[:3].sum():.1%}；如果股票彼此独立，每个 PC 约解释 1/{len(eigvals)} = {1 / len(eigvals):.1%}")

    z = (rets - rets.mean()) / rets.std()
    pc_scores = z.to_numpy() @ eigvecs[:, :3]
    pc1 = pd.Series(pc_scores[:, 0], index=rets.index)
    hs300 = get_index_daily("000300", START, END)["close"].pct_change().reindex(rets.index)
    same_sign = (np.sign(eigvecs[:, 0]) == np.sign(eigvecs[0, 0])).mean()
    print(f"\n   PC1 权重同号的比例 = {same_sign:.0%}")
    print(f"   PC1 得分序列与沪深300 日收益的相关系数 = {pc1.corr(hs300):.3f}")

    load = pd.DataFrame(eigvecs[:, :3], index=[STOCKS[c] for c in rets.columns], columns=["PC1", "PC2", "PC3"])
    print("\n   PC2 载荷最正 / 最负的各 4 只股票（解读 PC2 代表什么）：")
    pc2 = load["PC2"].sort_values()
    print("   最负：" + "、".join(f"{k}({v:+.2f})" for k, v in pc2.head(4).items()))
    print("   最正：" + "、".join(f"{k}({v:+.2f})" for k, v in pc2.tail(4).items()))

    sk = PCA(n_components=5).fit(z)
    print(f"\n   sklearn PCA 解释比例：{np.round(sk.explained_variance_ratio_, 3).tolist()}（与手算一致）")
    print("""
    💡 量化意义：
       - 风险分解：PC1 解释比例高 → 一个看似分散的组合，本质上主要暴露在"市场 Beta"上
       - 统计因子模型：PC 可以作为"统计风险因子"，与 Phase 7 的基本面因子互补
       - 统计套利：剔除前几个 PC 后的残差更可能有均值回归性（Phase 6）
       - PC2、PC3 的含义需要人去解读（风格？行业？），而且在不同时间窗口可能变化
    """)
    if is_synthetic(rets.index):
        print("    ⚠️ 离线合成数据中股票彼此独立，各 PC 解释比例接近 1/N、PC1 与沪深300 无关；请以第二部分的模拟为准")
    return explained, load


# ==================== 第四部分：协方差估计误差与 Ledoit-Wolf ====================


def min_var_weights(cov: np.ndarray) -> np.ndarray:
    """最小方差组合（允许做空）：w ∝ Σ⁻¹·1"""
    inv = np.linalg.pinv(cov)
    w = inv @ np.ones(len(cov))
    return w / w.sum()


def estimation_error():
    print("\n" + "=" * 60)
    print("第四部分：协方差矩阵的估计误差")
    print("=" * 60)

    print("""
    当股票数 N 接近样本天数 T 时，样本协方差矩阵噪声极大甚至奇异（不可逆）：
       - 样本特征值比真实特征值更"分散"：小特征值被严重低估（T 越小越严重）
       - 优化器会把权重押在"看起来方差很小"的方向上 —— 而那些方向往往只是估计误差
    🧪 实验：真实协方差为单因子结构（N = 100），用 T 天数据估计，比较样本协方差与 Ledoit-Wolf
    """)

    rng = np.random.default_rng(1)
    n = 100
    beta = rng.uniform(0.5, 1.5, n)
    true_cov = np.outer(beta, beta) * 0.01**2 + np.eye(n) * 0.015**2
    chol = np.linalg.cholesky(true_cov)
    true_eig = np.linalg.eigvalsh(true_cov)
    w_true = min_var_weights(true_cov)
    best_vol = np.sqrt(w_true @ true_cov @ w_true) * np.sqrt(TRADING_DAYS)

    rows = []
    for T in [60, 120, 250, 1000]:
        x = rng.standard_normal((T, n)) @ chol.T
        s_cov = np.cov(x.T)
        lw = LedoitWolf().fit(x)
        s_eig = np.linalg.eigvalsh(s_cov)
        res = []
        for c in (s_cov, lw.covariance_):
            w = min_var_weights(c)
            res.append(np.sqrt(w @ true_cov @ w) * np.sqrt(TRADING_DAYS))
        rows.append([T, f"{T / n:.1f}", f"{np.linalg.matrix_rank(s_cov)}", f"{s_eig.max() / true_eig.max():.2f}",
                     f"{max(s_eig.min(), 0) / true_eig.min():.2f}", f"{res[0]:.1%}", f"{res[1]:.1%}", f"{lw.shrinkage_:.2f}"])
    print(pd.DataFrame(rows, columns=["T 天", "T/N", "样本协方差的秩", "最大特征值 估计/真实", "最小特征值 估计/真实",
                                      "最小方差组合真实波动(样本Σ)", "(Ledoit-Wolf)", "收缩强度"]).to_string(index=False))
    print(f"\n   参照：用真实协方差构建的最小方差组合，年化波动 = {best_vol:.1%}")
    print("""
    💡 解读：
       - T ≤ N 时样本协方差的秩 < N（奇异），最小特征值被估成 0 —— 优化器以为找到了"零风险组合"
       - 本例中真实结构是"一个强因子"，最大特征值反而估得相对准；出问题的是那些小特征值
       - 用样本协方差优化出来的组合，真实波动明显高于理论最优；Ledoit-Wolf 收缩通常更接近最优
       - 其他补救：因子模型协方差、指数加权协方差、随机矩阵理论去噪（Phase 8 进阶）
    """)


def plot_pca(explained: np.ndarray, load: pd.DataFrame):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    k = len(explained)
    ax.bar(range(1, k + 1), explained, label="单个 PC")
    ax.plot(range(1, k + 1), np.cumsum(explained), "o-", color="C3", label="累计")
    ax.axhline(1 / k, color="gray", ls="--", label="独立时的期望 1/N")
    ax.set_xlabel("主成分")
    ax.set_title("解释方差比例（碎石图）")
    ax.legend()
    ax = axes[1]
    load[["PC1", "PC2"]].plot.barh(ax=ax)
    ax.set_title("PC1 / PC2 载荷")
    path = save_figure(fig, "08_pca.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    panel = get_close_panel(list(STOCKS), START, END)
    rets = panel.pct_change().dropna()
    print(f"📊 数据：{len(STOCKS)} 只股票，剔除任一股票停牌的日子后剩 {len(rets)} 个交易日\n")
    portfolio_risk(rets)
    eigen_intuition()
    explained, load = real_pca(rets)
    plot_pca(explained, load)
    estimation_error()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 不标准化、直接对协方差矩阵做 PCA，PC1 的权重会被哪些股票主导？为什么要先标准化？
    2. 分别用 2019-2021 和 2022-2024 两段数据做 PCA，PC1 解释比例如何变化？PC2 的含义变了吗？
    3. 实现滚动 250 日 PCA，画出 PC1 解释比例随时间的变化，并与沪深300 的波动率对比
    4. 用第三部分的数据，比较样本协方差与 Ledoit-Wolf 下最小方差组合的样本外波动（前 3 年估计，后 3 年检验）
    """)


if __name__ == "__main__":
    main()
