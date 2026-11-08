"""
风险平价 Risk Parity
====================

学习目标：
    1. 理解"等资金 ≠ 等风险"：60/40 组合的风险几乎都来自股票
    2. 手写边际风险贡献 MRC 与风险贡献 RC，并用数值微分验证
    3. 用三种方式求等风险贡献（ERC）组合：Spinu 凸化（cvxpy）、scipy 直接优化、riskfolio-lib
    4. 推广到风险预算（Risk Budgeting），按资产类别分配风险
    5. 回测风险平价，理解它的低波动、杠杆需求与"股债相关性失效"风险

核心概念：
    - σ_p = Σ_i RC_i，RC_i = w_i · (Σw)_i / σ_p（Euler 分解）
    - ERC：RC_i 全部相等；风险预算：RC_i / σ_p = b_i
    - Spinu 凸化：min ½ y'Σy - Σ b_i ln y_i，w = y / Σy
    - 所有资产两两相关系数相同时，ERC = 逆波动率加权

前置知识：
    - 01-mean-variance.py、02-covariance-shrinkage.py
    - CONCEPT.md 第 5 节

环境要求：
    - pip install numpy pandas scipy cvxpy riskfolio-lib scikit-learn matplotlib
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用示意参数的模拟收益，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from portfolio_utils import (
    ASSET_COLORS, TRADING_DAYS, asset_class_of, erc_weights, inverse_vol_weights,
    load_etf_returns, perf_table, risk_contribution, walk_forward,
)
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

RF = 0.02  # 杠杆融资成本（年化，示例值）


def rc_table(w: np.ndarray, Sigma: np.ndarray, names) -> pd.DataFrame:
    """手写风险贡献表：权重、MRC、RC、RC 占比"""
    sigma_p = np.sqrt(w @ Sigma @ w)
    mrc = Sigma @ w / sigma_p
    rc = w * mrc
    return pd.DataFrame({
        "权重%": w * 100, "MRC%": mrc * 100, "RC%": rc * 100, "RC占比%": rc / sigma_p * 100,
    }, index=names)


# ==================== 第一部分：等资金 ≠ 等风险 ====================


def capital_vs_risk(returns: pd.DataFrame):
    print("=" * 60)
    print("第一部分：等资金 ≠ 等风险 —— 拆开 60/40 组合的风险")
    print("=" * 60)

    names = ["沪深300", "国债"]
    Sigma = returns[names].cov().to_numpy() * TRADING_DAYS
    w = np.array([0.6, 0.4])
    tbl = rc_table(w, Sigma, names)
    sigma_p = np.sqrt(w @ Sigma @ w)
    print(f"""
    Euler 分解：σ_p = Σ_i w_i · ∂σ_p/∂w_i，其中
        MRC_i = ∂σ_p/∂w_i = (Σw)_i / σ_p    ← 边际风险贡献：多买一点资产 i，组合波动增加多少
        RC_i  = w_i · MRC_i                  ← 风险贡献：所有 RC 加起来正好等于 σ_p
    """)
    print(tbl.round(3).to_string())
    print(f"   Σ RC = {tbl['RC%'].sum():.4f}%    σ_p = {sigma_p * 100:.4f}%    差 = {abs(tbl['RC%'].sum() - sigma_p * 100):.1e}")

    # 用数值微分验证 MRC 公式
    eps = 1e-6
    num = []
    for i in range(len(w)):
        dw = np.zeros_like(w)
        dw[i] = eps
        num.append((np.sqrt((w + dw) @ Sigma @ (w + dw)) - np.sqrt((w - dw) @ Sigma @ (w - dw))) / (2 * eps))
    print(f"   MRC 解析式 vs 数值微分 最大差：{np.abs(np.array(num) - Sigma @ w / sigma_p).max():.1e}")
    print(f"""
    💡 资金上 60% 股票，风险上 {tbl.loc['沪深300', 'RC占比%']:.1f}% 来自股票，{tbl.loc['国债', 'RC占比%']:.1f}% 来自国债。
       比喻：拔河队里大力士（股票）和普通人（债券）拿同样长的绳子，出力的几乎全是大力士。
       RC 可以为负：与组合负相关的资产（这里股债相关 {Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1]):+.2f}）在"对冲"风险，
       此时其他资产的 RC 占比会超过 100%。
    """)


# ==================== 第二部分：求解 ERC ====================


def erc_scipy(Sigma: np.ndarray) -> np.ndarray:
    """直接优化：最小化 Σ_i Σ_j (RC_i - RC_j)²，约束 w ≥ 0、Σw = 1（非凸问题，需要检查是否收敛）"""
    n = len(Sigma)

    def obj(w):
        rc = risk_contribution(w, Sigma)
        return np.sum((rc[:, None] - rc[None, :]) ** 2) * 1e4  # 乘 1e4 只是数值缩放

    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
    return res.x / res.x.sum()


def erc_riskfolio(returns: pd.DataFrame) -> np.ndarray | None:
    try:
        import riskfolio as rp

        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)
        return None if w is None else w["weights"].to_numpy()
    except Exception as e:  # riskfolio 版本差异较多，失败时跳过即可
        print(f"   ⚠️ riskfolio 求解失败（{type(e).__name__}: {e}），跳过对比")
        return None


def solve_erc(returns: pd.DataFrame) -> np.ndarray:
    print("\n" + "=" * 60)
    print("第二部分：求解等风险贡献（ERC）组合")
    print("=" * 60)
    print("""
    Spinu（2013）凸化：min_y ½ y'Σy - Σ b_i·ln(y_i)，一阶条件 y_i(Σy)_i = b_i，
    恰好是"风险贡献与预算成比例"；对数项保证 y > 0，所以风险平价天然是多头组合。
    """)
    names = returns.columns
    Sigma = returns.cov().to_numpy() * TRADING_DAYS
    n = len(names)

    sols = {
        "1/N": np.ones(n) / n,
        "逆波动率": inverse_vol_weights(Sigma),
        "ERC-cvxpy": erc_weights(Sigma),
        "ERC-scipy": erc_scipy(Sigma),
    }
    w_rf = erc_riskfolio(returns)
    if w_rf is not None:
        sols["ERC-riskfolio"] = w_rf

    w_tbl = pd.DataFrame({k: v * 100 for k, v in sols.items()}, index=names)
    rc_tbl = pd.DataFrame({k: rc_table(v, Sigma, names)["RC占比%"] for k, v in sols.items()})
    print("权重 %：")
    print(w_tbl.round(2).to_string())
    print("\n风险贡献占比 %（手写 RC 公式计算）：")
    print(rc_tbl.round(2).to_string())

    print("\n核对：")
    for k, v in sols.items():
        rc = risk_contribution(v, Sigma)
        sigma_p = np.sqrt(v @ Sigma @ v)
        print(f"   {k:14s} 权重和 {v.sum():.6f}   ΣRC - σ_p = {rc.sum() - sigma_p:+.1e}   "
              f"max|RC占比 - 1/n| = {np.abs(rc / sigma_p - 1 / n).max():.2e}   组合波动 {sigma_p:.2%}")
    print(f"   ERC-cvxpy 与 ERC-scipy 最大权重差：{np.abs(sols['ERC-cvxpy'] - sols['ERC-scipy']).max():.1e}")
    print("""
    💡 观察：
       - 1/N 组合的风险主要来自高波动的权益类资产
       - 逆波动率只看各自波动，忽略相关性，所以 RC 并不完全相等；ERC 用相关性做了进一步修正
       - ERC 给低波动的国债很高的资金权重——"等风险"的代价是低收益，通常要加杠杆
    """)

    # 权重 vs 风险贡献
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, data, title in [(axes[0], w_tbl, "资金权重 %"), (axes[1], rc_tbl, "风险贡献占比 %")]:
        cols = ["1/N", "逆波动率", "ERC-cvxpy"]
        bottom = np.zeros(len(cols))
        for name in names:
            vals = data.loc[name, cols].to_numpy()
            ax.bar(cols, vals, bottom=bottom, color=ASSET_COLORS[name], label=name, edgecolor="white", lw=1.5)
            bottom += vals
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("等资金、逆波动率与等风险贡献")
    path = save_figure(fig, "03_weights_vs_risk.png", subdir="phase-8")
    print(f"🖼  权重与风险贡献对比图已保存：{path}")
    return Sigma


# ==================== 第三部分：特例与风险预算 ====================


def special_case_and_budget(returns: pd.DataFrame, Sigma: np.ndarray):
    print("\n" + "=" * 60)
    print("第三部分：特例验证与风险预算")
    print("=" * 60)

    names = returns.columns
    vol = np.sqrt(np.diag(Sigma))
    rho = 0.3
    corr = np.full((len(vol), len(vol)), rho)
    np.fill_diagonal(corr, 1.0)
    Sigma_eq = np.outer(vol, vol) * corr
    diff = np.abs(erc_weights(Sigma_eq) - inverse_vol_weights(Sigma_eq)).max()
    print(f"\n① 若两两相关系数都等于 {rho}：ERC 与逆波动率的最大权重差 = {diff:.1e} → 两者相同")

    classes = asset_class_of(list(names))
    class_budget = {"A股": 0.30, "海外股票": 0.20, "债券": 0.35, "商品": 0.15}
    counts = classes.value_counts()
    b = np.array([class_budget[c] / counts[c] for c in classes])
    w = erc_weights(Sigma, b)
    tbl = rc_table(w, Sigma, names)
    tbl["预算%"] = b * 100
    tbl["类别"] = classes
    print(f"\n② 风险预算：按类别分配风险 {class_budget}，类别内平均分")
    print(tbl[["类别", "权重%", "预算%", "RC占比%"]].round(2).to_string())
    agg = tbl.groupby("类别")[["权重%", "RC占比%"]].sum().round(1)
    print("\n   按类别汇总：")
    print(agg.to_string())
    print(f"   max|RC占比 - 预算| = {np.abs(tbl['RC占比%'] - tbl['预算%']).max():.2e}%")
    print("""
    💡 风险预算是大类资产配置最常用的语言："股票承担 50% 风险"比"股票 30% 仓位"更能说明组合性质。
       同样的框架也用于多策略、多因子之间的风险分配。
    """)


# ==================== 第四部分：回测与杠杆 ====================


def backtest(returns: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：样本外回测、杠杆与相关性失效")
    print("=" * 60)
    print("规则：每月末用过去 252 天的样本协方差求权重，持有一个月（未扣交易成本）")
    window = returns.iloc[-252:]
    lw = LedoitWolf().fit(window.to_numpy())
    i_bond = list(returns.columns).index("国债")
    print(f"""
    ⚠️ 为什么这里不用第 02 课的 sklearn LedoitWolf？它的收缩目标是"等方差单位阵"，
       会把所有资产的方差往平均值拉。个股波动相近时问题不大，但大类资产波动相差十倍以上：
       最近 252 天国债年化波动 {window['国债'].std() * np.sqrt(TRADING_DAYS):.2%}，收缩（δ = {lw.shrinkage_:.3f}）后变成 {np.sqrt(lw.covariance_[i_bond, i_bond] * TRADING_DAYS):.2%}。
       风险平价的权重直接取决于波动，这样的扭曲不可接受。这里 n/T = {returns.shape[1] / 252:.3f} 很小，样本协方差已经够用；
       如果确实需要收缩，应选择保留各自方差的目标（如第 02 课的常相关目标）。
    """)

    names = list(returns.columns)

    def sample_cov(x):
        return x.cov().to_numpy() * TRADING_DAYS

    fixed_6040 = np.array([0.6 if c == "沪深300" else 0.4 if c == "国债" else 0.0 for c in names])
    strategies = {
        "1/N": lambda x: np.ones(x.shape[1]) / x.shape[1],
        "60/40 股债": lambda x: fixed_6040,
        "逆波动率": lambda x: inverse_vol_weights(sample_cov(x)),
        "ERC": lambda x: erc_weights(sample_cov(x)),
    }
    rets, extra = {}, {}
    for label, fn in strategies.items():
        r, W, to = walk_forward(returns, fn, lookback=252)
        rets[label] = r
        extra[label] = {"年化换手": f"{to:.0%}", "平均国债权重": f"{W['国债'].mean():.0%}"}

    target_vol = rets["1/N"].std()
    lev = target_vol / rets["ERC"].std()
    # 事后按全样本波动定杠杆只是示意（有前视），实盘应使用波动率目标的滚动杠杆
    rets[f"ERC × {lev:.1f} 倍杠杆"] = lev * rets["ERC"] - (lev - 1) * RF / TRADING_DAYS
    extra[f"ERC × {lev:.1f} 倍杠杆"] = {"年化换手": "—", "平均国债权重": "—"}
    print(perf_table(rets, extra).to_string())
    print(f"""
    ⚠️ Sharpe 按 rf=0 计算，会抬高重仓国债的组合；杠杆版本扣了融资成本，所以 Sharpe 低于不加杠杆的 ERC。
    💡 杠杆版本：把 ERC 放大到与 1/N 相同的波动，融资成本按年化 {RF:.0%} 扣除。
       ⚠️ 这里的杠杆倍数用全样本波动事后确定，含前视偏差，只用于说明"风险平价需要杠杆"这件事。
    """)

    # 按年份拆开：股债同跌的年份风险平价表现如何
    yearly = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in rets.items()})
    eq_bond_corr = returns.groupby(returns.index.year).apply(lambda x: x["沪深300"].corr(x["国债"]))
    yearly["沪深300-国债相关"] = eq_bond_corr.reindex(yearly.index)
    fmt = yearly.copy()
    for c in fmt.columns[:-1]:
        fmt[c] = fmt[c].map("{:+.1%}".format)
    fmt["沪深300-国债相关"] = yearly["沪深300-国债相关"].map("{:+.2f}".format)
    print("分年度收益：")
    print(fmt.to_string())
    pos_years = yearly.index[yearly["沪深300-国债相关"] > 0].tolist()
    print(f"""
    💡 风险平价的隐含假设是"股债负相关、互相对冲"。当股债相关性转正、两者同跌时，
       低波动资产（债券）的高权重（加杠杆后更高）就不再提供保护。
       本样本中沪深300 与国债日收益相关为正的年份：{pos_years or '无'}
       注意：2022 年美股美债同跌，是海外风险平价的典型回撤年；本例的国债 ETF 是中国国债，
       股债相关结构因市场而异——风险平价依赖的"对冲关系"必须在你自己的资产上检验。
    """)

    fig, ax = plt.subplots(figsize=(11, 5))
    for (label, r), c in zip(rets.items(), ["#8a8984", "#eda100", "#1baf7a", "#2a78d6", "#eb6834"]):
        ax.plot((1 + r).cumprod(), label=label, color=c, lw=1.6)
    ax.set_title("样本外净值：1/N、60/40、逆波动率与风险平价")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
    path = save_figure(fig, "03_risk_parity_nav.png", subdir="phase-8")
    print(f"🖼  净值图已保存：{path}")


# ==================== 主函数 ====================


def main():
    returns = load_etf_returns()
    if is_synthetic(returns):
        print("⚠️ 以下结果基于模拟收益，只用于理解方法，不代表真实市场\n")
    capital_vs_risk(returns)
    Sigma = solve_erc(returns)
    special_case_and_budget(returns, Sigma)
    backtest(returns)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用"牛顿法 / 循环坐标下降"手写一个不依赖 cvxpy 的 ERC 求解器，并与 erc_weights 对比
    2. 把第四部分的杠杆改为"波动率目标"：每月用过去 60 天波动把组合缩放到 10% 年化波动，杠杆上限 3 倍
    3. 把风险预算改为"A股 50%、海外 30%、债券 10%、商品 10%"，组合波动和回撤怎么变？
    4. 用 2022 年的数据单独估计协方差，计算 ERC 权重；与 2014-2021 估计的权重相比，国债权重怎么变？为什么？
    """)


if __name__ == "__main__":
    main()
