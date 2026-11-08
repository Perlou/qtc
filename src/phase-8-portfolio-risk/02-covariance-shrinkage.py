"""
协方差估计与 Ledoit-Wolf 收缩
============================

学习目标：
    1. 理解样本协方差在"资产多、样本短"时为什么会病态（条件数、特征值外扩）
    2. 手写 Ledoit-Wolf 收缩（单位阵目标、常相关目标），并与 sklearn 对比
    3. 用模拟实验量化收缩带来的改进（估计误差、GMV 真实风险）
    4. 用真实个股数据做样本外检验：哪个协方差估计出的 GMV 组合下个月波动最低
    5. 了解 EWMA 协方差与因子模型协方差

核心概念：
    - 样本协方差 S 无偏，但 n/T 不小时特征值"大的更大、小的更小"
    - 条件数 κ = λ_max / λ_min，越大求逆越不稳定；n > T 时 S 奇异
    - 收缩：Σ̂ = δF + (1-δ)S，F 是结构简单的目标，δ 由 Ledoit-Wolf 公式给出
    - 检验 Σ̂ 好坏的实用方法：用它构造 GMV，看样本外实现波动

前置知识：
    - 01-mean-variance.py；Phase 3 协方差与 PCA
    - CONCEPT.md 第 3.3、4 节

环境要求：
    - pip install numpy pandas scikit-learn matplotlib akshare
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用合成 / 模拟数据，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.covariance import OAS, LedoitWolf

from portfolio_utils import TRADING_DAYS, gmv_weights, load_etf_returns, load_stock_returns, perf_table, walk_forward
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)


def cond_number(C: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(C)
    return float(eig.max() / eig.min()) if eig.min() > 1e-14 else np.inf


def factor_model_cov(n: int, seed: int = 0) -> np.ndarray:
    """单因子模型的"真实"日协方差：β_i 不同，市场波动 18%，特异波动 20%~40%（年化）"""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    mkt_var = 0.18**2 / TRADING_DAYS
    idio_var = rng.uniform(0.20, 0.40, n) ** 2 / TRADING_DAYS
    return mkt_var * np.outer(beta, beta) + np.diag(idio_var)


# ==================== 第一部分：样本协方差的问题 ====================


def sample_cov_problem():
    print("=" * 60)
    print("第一部分：样本协方差的问题 —— 特征值外扩与条件数")
    print("=" * 60)

    n = 50
    rng = np.random.default_rng(1)
    print(f"""
    实验 A：{n} 个资产的真实相关矩阵是单位阵（所有特征值都 = 1），看样本相关矩阵的特征值范围。
    随机矩阵理论（Marchenko-Pastur）预测纯噪声下特征值落在 [(1-√q)², (1+√q)²]，q = n/T。
    """)
    rows = []
    for T in [60, 120, 250, 1000, 5000]:
        X = rng.standard_normal((T, n))
        eig = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
        q = n / T
        rows.append([T, f"{q:.2f}", f"{eig.min():.3f}", f"{eig.max():.3f}",
                     f"{(1 - np.sqrt(q)) ** 2:.3f}", f"{(1 + np.sqrt(q)) ** 2:.3f}", f"{eig.max() / eig.min():.1f}"])
    print(pd.DataFrame(rows, columns=["T", "q=n/T", "最小特征值", "最大特征值", "MP 下界", "MP 上界", "条件数"]).to_string(index=False))

    Sigma = factor_model_cov(n)
    print(f"\n    实验 B：单因子模型的真实协方差（条件数 {cond_number(Sigma):.0f}），样本协方差的条件数：")
    rows = []
    L = np.linalg.cholesky(Sigma)
    for T in [30, 60, 120, 250, 1000]:
        X = rng.standard_normal((T, n)) @ L.T
        S = np.cov(X, rowvar=False)
        rows.append([T, np.linalg.matrix_rank(S), f"{cond_number(S):.3g}"])
    print(pd.DataFrame(rows, columns=["T", "S 的秩", "条件数"]).to_string(index=False))
    print(f"""
    💡 结论：
       - 真实特征值全是 1，样本特征值却散开到很宽的区间——小特征值被系统性低估
       - Σ⁻¹ 的特征值是倒数，被低估的小特征值取逆后变成巨大的数，优化器就会重仓对应方向
       - T ≤ n 时 S 的秩最多 T-1，矩阵奇异，条件数为 inf，根本无法求逆
         （A 股 5000 只股票、1 年日线约 242 天，就是这种情况）
    """)


# ==================== 第二部分：手写 Ledoit-Wolf ====================


def ledoit_wolf_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004, JMVA)：收缩目标 F = m·I，m = 平均方差

    S = X'X/T（X 已去均值）
    d² = ||S - mI||²_F / n                      ← 样本与目标的距离
    b̄² = (1/T²) Σ_t ||x_t x_t' - S||²_F / n     ← S 本身的估计噪声
    δ = min(b̄², d²) / d²
    """
    T, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    m = np.trace(S) / n
    d2 = np.linalg.norm(S - m * np.eye(n), "fro") ** 2 / n
    b2_bar = sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in Xc) / T**2 / n
    delta = min(b2_bar, d2) / d2
    return delta * m * np.eye(n) + (1 - delta) * S, delta


def ledoit_wolf_const_corr(X: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004, "Honey, I Shrunk the Sample Covariance Matrix")
    收缩目标 F：保留各自方差，相关系数统一取平均相关 r̄
    δ = max(0, min(1, κ/T))，κ = (π - ρ) / γ
    """
    T, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    var = np.diag(S)
    sd = np.sqrt(var)
    rbar = ((S / np.outer(sd, sd)).sum() - n) / (n * (n - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    Y = Xc**2
    pi_mat = Y.T @ Y / T - S**2                    # 每个 s_ij 的渐近方差
    pi_hat = pi_mat.sum()
    theta = (Xc**3).T @ Xc / T - var[:, None] * S
    np.fill_diagonal(theta, 0.0)
    rho_hat = np.trace(pi_mat) + rbar * ((sd[None, :] / sd[:, None]) * theta).sum()
    gamma_hat = np.linalg.norm(S - F, "fro") ** 2  # 目标与样本的距离
    delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / T))
    return delta * F + (1 - delta) * S, delta


def handwritten_vs_sklearn() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：手写 Ledoit-Wolf，并与 sklearn 对比")
    print("=" * 60)

    ret, industry = load_stock_returns()
    if is_synthetic(ret):
        print("⚠️ 本部分和第四部分的个股是彼此独立的合成数据，数字不代表真实市场")
    window = ret.iloc[-120:]
    X = window.to_numpy()
    T, n = X.shape
    print(f"\n数据：{n} 只个股最近 {T} 个交易日（{window.index[0].date()} ~ {window.index[-1].date()}），n/T = {n / T:.2f}")

    S = np.cov(X, rowvar=False, bias=True)
    lw_hand, d_hand = ledoit_wolf_identity(X)
    lw_sk = LedoitWolf().fit(X)
    cc_hand, d_cc = ledoit_wolf_const_corr(X)
    oas = OAS().fit(X)
    print(f"""
    ┌──────────────────────┬────────────┬──────────────────────────────────┐
    │ 估计量               │ 收缩强度 δ │ 与 sklearn 的差异                │
    ├──────────────────────┼────────────┼──────────────────────────────────┤
    │ 手写 LW（单位阵目标）│ {d_hand:10.4f} │ δ 差 {abs(d_hand - lw_sk.shrinkage_):.1e}，矩阵最大差 {np.abs(lw_hand - lw_sk.covariance_).max():.1e} │
    │ sklearn LedoitWolf   │ {lw_sk.shrinkage_:10.4f} │ —                                │
    │ 手写 LW（常相关目标）│ {d_cc:10.4f} │ sklearn 无此实现                 │
    │ sklearn OAS          │ {oas.shrinkage_:10.4f} │ —                                │
    └──────────────────────┴────────────┴──────────────────────────────────┘
    """)
    rows = []
    for name, C in [("样本 S", S), ("LW 单位阵", lw_sk.covariance_), ("LW 常相关", cc_hand), ("OAS", oas.covariance_)]:
        eig = np.linalg.eigvalsh(C)
        w = gmv_weights(C)
        rows.append([name, f"{eig.min() * 1e4:.3f}", f"{eig.max() * 1e4:.3f}", f"{cond_number(C):.1f}",
                     f"{w.min():+.1%}", f"{w.max():+.1%}", f"{np.abs(w).sum():.2f}"])
    print(pd.DataFrame(rows, columns=["估计量", "λ_min(×1e-4)", "λ_max(×1e-4)", "条件数", "GMV 最小权重", "GMV 最大权重", "GMV 总杠杆 Σ|w|"]).to_string(index=False))
    print("""
    💡 收缩把小特征值往上拉、大特征值往下压 → 条件数下降 → GMV 权重不再那么极端
       单位阵目标会把所有相关性往 0 拉；对同涨同跌的股票，常相关目标通常更合适
    """)

    # 特征值谱图
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (name, C), c in zip([("样本 S", S), ("LW 单位阵", lw_sk.covariance_), ("LW 常相关", cc_hand)],
                            ["#8a8984", "#2a78d6", "#eb6834"]):
        ax.plot(np.sort(np.linalg.eigvalsh(C))[::-1] * 1e4, marker="o", ms=4, lw=1.5, color=c, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("特征值序号（从大到小）")
    ax.set_ylabel("特征值 ×1e-4（对数轴）")
    ax.set_title(f"{n} 只股票、{T} 天样本的协方差特征值谱")
    ax.grid(alpha=0.3)
    ax.legend()
    path = save_figure(fig, "02_eigen_spectrum.png", subdir="phase-8")
    print(f"🖼  特征值谱图已保存：{path}")
    return ret


# ==================== 第三部分：模拟实验 ====================


def simulation_experiment(n: int = 50, n_trials: int = 30, seed: int = 7):
    print("\n" + "=" * 60)
    print("第三部分：模拟实验 —— 已知真实 Σ，收缩到底好多少")
    print("=" * 60)

    Sigma = factor_model_cov(n, seed=3)
    L = np.linalg.cholesky(Sigma)
    w_true = gmv_weights(Sigma)
    var_opt = w_true @ Sigma @ w_true
    rng = np.random.default_rng(seed)
    print(f"""
    设定：{n} 个资产，真实 Σ 来自单因子模型。每个 T 重复 {n_trials} 次：
      ① Frobenius 误差 ||Σ̂ - Σ||_F / ||Σ||_F
      ② 用 Σ̂ 构造 GMV，在真实 Σ 下的波动 / 理论最优波动（=1 最好）
    """)
    rows = []
    for T in [30, 60, 120, 250, 1000]:
        res = {k: ([], []) for k in ["样本 S", "LW 单位阵", "LW 常相关"]}
        for _ in range(n_trials):
            X = rng.standard_normal((T, n)) @ L.T
            ests = {
                "样本 S": np.cov(X, rowvar=False, bias=True),
                "LW 单位阵": LedoitWolf().fit(X).covariance_,
                "LW 常相关": ledoit_wolf_const_corr(X)[0],
            }
            for k, C in ests.items():
                res[k][0].append(np.linalg.norm(C - Sigma, "fro") / np.linalg.norm(Sigma, "fro"))
                # T ≤ n 时样本协方差奇异，只能用伪逆
                x = np.linalg.pinv(C) @ np.ones(n) if T <= n and k == "样本 S" else np.linalg.solve(C, np.ones(n))
                w = x / x.sum()
                res[k][1].append(np.sqrt(w @ Sigma @ w / var_opt))
        row = [T]
        for k in res:
            row += [f"{np.mean(res[k][0]):.3f}", f"{np.median(res[k][1]):.2f}"]
        rows.append(row)
    cols = ["T"] + [f"{k}|{m}" for k in ["样本", "LW单位阵", "LW常相关"] for m in ["F误差", "GMV风险比"]]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("""
    💡 读表：样本越短（T 越小），收缩相对样本协方差的优势越明显；T 很长时三者趋于一致。
       T ≤ n 时样本协方差奇异，只能用伪逆（最小范数解）；T 只比 n 略大时 S 虽然可逆，
       但条件数极大，GMV 可能比伪逆还差——"刚好可逆"是最危险的区域。
       真实结构是"单因子"，所以常相关目标比单位阵目标更接近真相——收缩目标要贴合数据结构。
    """)


# ==================== 第四部分：真实数据样本外检验 ====================


def out_of_sample_test(ret: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：样本外检验 —— 用 GMV 的实现波动比较协方差估计")
    print("=" * 60)
    print(f"""
    规则：每月末用过去 lookback 天估计 Σ̂ → 闭式 GMV（允许做空）→ 持有一个月。
    GMV 的目标就是"风险最小"，所以样本外实现波动越低，说明 Σ̂ 越好。
    数据：{ret.shape[1]} 只个股，{ret.index[0].date()} ~ {ret.index[-1].date()}
    """)

    estimators = {
        "样本 S": lambda X: np.cov(X, rowvar=False),
        "LW 单位阵": lambda X: LedoitWolf().fit(X).covariance_,
        "LW 常相关": lambda X: ledoit_wolf_const_corr(X)[0],
    }
    for lookback in [60, 250]:
        rets, extra = {}, {}
        r, _, to = walk_forward(ret, lambda x: np.ones(x.shape[1]) / x.shape[1], lookback=lookback)
        rets["1/N 等权"], extra["1/N 等权"] = r, {"年化换手": f"{to:.0%}", "平均Σ|w|": "1.00"}
        for name, est in estimators.items():
            lev = []

            def fn(x, est=est, lev=lev):
                w = gmv_weights(est(x.to_numpy()))
                lev.append(np.abs(w).sum())
                return w

            r, _, to = walk_forward(ret, fn, lookback=lookback)
            rets[f"GMV-{name}"] = r
            extra[f"GMV-{name}"] = {"年化换手": f"{to:.0%}", "平均Σ|w|": f"{np.mean(lev):.2f}"}
        print(f"\n▶ lookback = {lookback} 天（n/T = {ret.shape[1] / lookback:.2f}）")
        print(perf_table(rets, extra).to_string())
    print("""
    💡 关注"年化波动"一列：它是协方差估计质量的直接体现。
       收缩版本的换手和杠杆（Σ|w|）通常更低——权重更稳定，交易成本也更低。
       lookback 越短（n/T 越大），收缩的作用越大。
    """)


# ==================== 第五部分：EWMA 与因子模型 ====================


def ewma_and_factor():
    print("\n" + "=" * 60)
    print("第五部分：EWMA 协方差与因子模型")
    print("=" * 60)

    ret = load_etf_returns(verbose=False)[["沪深300", "标普500", "国债"]]
    lam = 0.94
    ewm = ret.ewm(alpha=1 - lam, adjust=False)
    ewm_vol = ewm.std() * np.sqrt(TRADING_DAYS)
    roll_vol = ret.rolling(250).std() * np.sqrt(TRADING_DAYS)
    ewm_corr = ewm.corr()
    corr_eq_bond = ewm_corr.xs("沪深300", level=1)["国债"]
    sample_dates = ret.resample("YE").last().index
    sample_dates = [ret.index[ret.index <= d][-1] for d in sample_dates]
    tbl = pd.DataFrame({
        "沪深300 EWMA 波动": ewm_vol.loc[sample_dates, "沪深300"].map("{:.1%}".format),
        "沪深300 250日波动": roll_vol.loc[sample_dates, "沪深300"].map(lambda x: "—" if np.isnan(x) else f"{x:.1%}"),
        "沪深300-国债 EWMA 相关": corr_eq_bond.loc[sample_dates].map("{:+.2f}".format),
    }, index=pd.Index([d.strftime("%Y-%m-%d") for d in sample_dates], name="年末"))
    print(f"\nEWMA（λ = {lam}，半衰期约 {np.log(0.5) / np.log(lam):.0f} 天）vs 250 日滚动窗口：")
    print(tbl.to_string())
    stable = corr_eq_bond.iloc[60:]  # 去掉前 60 天（观测太少，相关系数不稳定）
    print(f"   EWMA 股债相关的范围（去掉前 60 天）：[{stable.min():+.2f}, {stable.max():+.2f}]——相关性本身随时间剧烈变化")
    print("""
    EWMA：Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1} r_{t-1}'，近期数据权重大，能快速反映波动聚集

    因子模型：Σ = B F B' + D（Phase 7 的 Barra 风险模型）
    ┌──────────────┬─────────────────────┬──────────────────────────────┐
    │ 方法         │ 参数个数（n=3000）  │ 特点                         │
    ├──────────────┼─────────────────────┼──────────────────────────────┤
    │ 样本协方差   │ n(n+1)/2 ≈ 450 万   │ n > T 时奇异                 │
    │ 因子模型 k=40│ nk + k(k+1)/2 + n   │ 天然正定、可解释，生产主流   │
    │              │ ≈ 12.4 万           │                              │
    └──────────────┴─────────────────────┴──────────────────────────────┘
    ⚠️ 无论哪种方法，都只能用"估计时点之前"的数据——全样本估计后回测是前视偏差。
    """)


# ==================== 主函数 ====================


def main():
    sample_cov_problem()
    ret = handwritten_vs_sklearn()
    simulation_experiment()
    out_of_sample_test(ret)
    ewma_and_factor()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 实现"单因子目标"的收缩（Ledoit-Wolf 2003）：F = β β' σ_m² + D，比较它与常相关目标的样本外 GMV 波动
    2. 第三部分把真实 Σ 换成"5 个行业、行业内相关 0.6、行业间相关 0.2"的块结构，哪种目标最好？
    3. 用 EWMA 协方差（halflife=20、60、120）做第四部分的 GMV 检验，与 LW 比较
    4. 对 A 股停牌：把某只股票连续 20 天收益设为 NaN，比较"填 0"与"成对删除"两种处理对相关系数的影响
    """)


if __name__ == "__main__":
    main()
