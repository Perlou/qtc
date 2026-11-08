"""
VaR 与 CVaR
===========

学习目标：
    1. 理解 VaR（分位数）与 CVaR / ES（尾部均值）的含义与区别
    2. 实现三类计算方法：历史模拟法、参数法（含 Cornish-Fisher 修正）、蒙特卡洛法（正态 / 多元 t）
    3. 用反例理解"VaR 不满足次可加性，CVaR 满足"
    4. 检验平方根法则在多日 VaR 上的误差
    5. 用 Kupiec POF 与 Christoffersen 独立性检验回测 VaR 模型
    6. 用 Rockafellar-Uryasev 线性化做最小 CVaR 组合

核心概念：
    - 损失 L = -r；VaR_α = L 的 α 分位数；CVaR_α = E[L | L ≥ VaR_α]
    - 参数法：VaR = -(μ + z_{1-α}σ)，CVaR = -μ + σ·φ(z_{1-α})/(1-α)
    - Kupiec：突破次数应接近 (1-α)T，LR_POF ~ χ²(1)，> 3.84 拒绝
    - CVaR(w) = min_ζ ζ + E[(L(w) - ζ)⁺]/(1-α)，最优 ζ 就是 VaR

前置知识：
    - Phase 3 分布、肥尾、蒙特卡洛
    - CONCEPT.md 第 9 节

环境要求：
    - pip install numpy pandas scipy cvxpy matplotlib
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用示意参数的模拟收益，并打印 ⚠️
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from portfolio_utils import TRADING_DAYS, load_etf_returns
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

CAPITAL = 1_000_000  # 组合市值（元）
WEIGHTS = {"沪深300": 0.25, "中证500": 0.10, "创业板": 0.05, "纳指": 0.075, "标普500": 0.075, "国债": 0.35, "黄金": 0.10}


# ==================== VaR / CVaR 计算函数 ====================


def var_cvar_historical(r: np.ndarray, alpha: float = 0.99):
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()


def var_cvar_parametric(r: np.ndarray, alpha: float = 0.99):
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu + z * sigma), -mu + sigma * stats.norm.pdf(z) / (1 - alpha)


def var_cornish_fisher(r: np.ndarray, alpha: float = 0.99) -> float:
    mu, sigma = r.mean(), r.std(ddof=1)
    S, K = stats.skew(r), stats.kurtosis(r)  # kurtosis 默认是超额峰度
    z = stats.norm.ppf(1 - alpha)
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36
    return -(mu + z_cf * sigma)


def simulate_portfolio(w, mu, Sigma, n_sims=100_000, dof=None, seed=0) -> np.ndarray:
    """蒙特卡洛：dof=None 为多元正态，否则为多元 t（缩放后协方差仍为 Σ）"""
    rng = np.random.default_rng(seed)
    if dof is None:
        L = np.linalg.cholesky(Sigma)
        sims = mu + rng.standard_normal((n_sims, len(w))) @ L.T
    else:
        L = np.linalg.cholesky(Sigma * (dof - 2) / dof)
        z = rng.standard_normal((n_sims, len(w))) @ L.T
        chi = rng.chisquare(dof, size=(n_sims, 1))
        sims = mu + z * np.sqrt(dof / chi)
    return sims @ w


# ==================== 第一部分：定义 ====================


def definitions():
    print("=" * 60)
    print("第一部分：VaR 与 CVaR 的定义")
    print("=" * 60)
    print("""
    VaR（Value at Risk）：  "99% 的日子里，单日亏损不会超过 X"  → 损失分布的 99% 分位数
    CVaR / ES：             "最差的 1% 的日子里，平均亏多少"   → 尾部损失的均值

             收益分布                                   比喻：VaR 告诉你洪水线在哪，
                ▲                                           CVaR 告诉你淹了以后平均水多深
                │         ▄▄██▄▄
                │       ▄████████▄
                │     ▄████████████▄
                │  ▁▄██████████████████▄▁
           ─────┴──┼───┼──────────────────────▶ 收益
                   │   └─ -VaR（1% 分位）
                   └─ CVaR = 这一段尾部的平均

    ⚠️ CVaR 满足次可加性（合并组合风险不会变大），VaR 不满足；
       巴塞尔 FRTB 用 97.5% ES 取代了 99% VaR 作为市场风险的核心度量。
    """)


# ==================== 第二部分：三类方法 ====================


def three_methods(returns: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 60)
    print("第二部分：历史法、参数法、蒙特卡洛法")
    print("=" * 60)

    names = list(returns.columns)
    w = np.array([WEIGHTS[c] for c in names])
    port = returns @ w
    print(f"示例组合（{CAPITAL / 1e4:.0f} 万元，每日再平衡到固定权重）：{WEIGHTS}")
    print(f"样本 {returns.index[0].date()} ~ {returns.index[-1].date()}，{len(port)} 天")

    mu, Sigma = returns.mean().to_numpy(), returns.cov().to_numpy()
    sims_n = simulate_portfolio(w, mu, Sigma)
    sims_t = simulate_portfolio(w, mu, Sigma, dof=4)

    for label, r in [("示例组合", port.to_numpy()), ("创业板 ETF（单只）", returns["创业板"].to_numpy())]:
        print(f"\n▶ {label}：日均 {r.mean():+.3%}，日波动 {r.std():.3%}，偏度 {stats.skew(r):+.2f}，超额峰度 {stats.kurtosis(r):.1f}")
        rows = []
        for a in [0.95, 0.99]:
            hv, hc = var_cvar_historical(r, a)
            pv, pc = var_cvar_parametric(r, a)
            cf = var_cornish_fisher(r, a)
            row = {"置信度": f"{a:.0%}", "历史 VaR": hv, "历史 CVaR": hc, "正态 VaR": pv, "正态 CVaR": pc, "CF VaR": cf}
            if label == "示例组合":
                mv, mc = var_cvar_historical(sims_n, a)
                tv, tc = var_cvar_historical(sims_t, a)
                row.update({"MC正态 VaR": mv, "MC正态 CVaR": mc, "MC-t4 VaR": tv, "MC-t4 CVaR": tc})
            rows.append(row)
        tbl = pd.DataFrame(rows).set_index("置信度")
        print(tbl.map(lambda x: f"{x:.2%}").to_string())
        hv99 = tbl.loc["99%", "历史 VaR"]
        print(f"   99% 历史 VaR 对应金额：{hv99 * CAPITAL:,.0f} 元；"
              f"正态参数法 / 历史法 = {tbl.loc['99%', '正态 VaR'] / hv99:.2f}，"
              f"正态 CVaR / 历史 CVaR = {tbl.loc['99%', '正态 CVaR'] / tbl.loc['99%', '历史 CVaR']:.2f}")
    print("""
    💡 读表：
       - 每一行 CVaR ≥ VaR（尾部均值一定不小于尾部起点）
       - 超额峰度 > 0 说明肥尾：正态参数法在 99% 处通常低估风险，CVaR 低估得更多；
         而在 95% 处正态 VaR 反而可能偏高——肥尾分布"中间更尖、尾巴更厚"
       - Cornish-Fisher 用偏度、峰度修正分位数，部分弥补肥尾；多元 t 蒙特卡洛的尾部也比正态更厚
       - 蒙特卡洛的结果取决于你假设的分布——模型风险并没有消失，只是换了位置
    """)
    return port


# ==================== 第三部分：次可加性反例 ====================


def discrete_var_cvar(losses: np.ndarray, probs: np.ndarray, alpha: float):
    """离散分布的 VaR 与 CVaR（尾部按概率精确分配）"""
    order = np.argsort(losses)[::-1]
    L, p = losses[order], probs[order]
    tail = 1 - alpha
    cum = np.cumsum(p)
    var = L[np.searchsorted(cum, tail - 1e-12, side="right")] if cum[0] <= tail else L[0]
    take = np.minimum(p, np.maximum(tail - (cum - p), 0))
    return var, float((L * take).sum() / tail)


def subadditivity():
    print("\n" + "=" * 60)
    print("第三部分：VaR 不满足次可加性 —— 一个经典反例")
    print("=" * 60)
    p_def, loss = 0.04, 100.0
    a = 0.95
    v1, c1 = discrete_var_cvar(np.array([loss, 0.0]), np.array([p_def, 1 - p_def]), a)
    # 两只独立债券：0、1、2 只违约
    probs = np.array([(1 - p_def) ** 2, 2 * p_def * (1 - p_def), p_def**2])
    v2, c2 = discrete_var_cvar(np.array([0.0, loss, 2 * loss]), probs, a)
    print(f"""
    两只独立的债券，各自违约概率 {p_def:.0%}，违约损失 {loss:.0f}，否则损失 0。置信度 {a:.0%}。
    ┌──────────────┬──────────┬──────────┐
    │              │ VaR      │ CVaR     │
    ├──────────────┼──────────┼──────────┤
    │ 债券 A       │ {v1:8.1f} │ {c1:8.1f} │
    │ 债券 B       │ {v1:8.1f} │ {c1:8.1f} │
    │ A + B 之和   │ {2 * v1:8.1f} │ {2 * c1:8.1f} │
    │ 组合 A+B     │ {v2:8.1f} │ {c2:8.1f} │
    └──────────────┴──────────┴──────────┘
    单只违约概率 {p_def:.0%} < {1 - a:.0%}，所以单只 VaR = 0；但组合"至少一只违约"的概率是 {1 - probs[0]:.2%} > {1 - a:.0%}，
    组合 VaR 变成 {v2:.0f} > 0 + 0。分散投资反而让 VaR 变大，与直觉矛盾。
    CVaR：组合 {c2:.1f} ≤ 单只之和 {2 * c1:.1f}，满足次可加性。
    """)


# ==================== 第四部分：平方根法则 ====================


def square_root_rule(returns: pd.DataFrame, port: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：多日 VaR 与平方根法则")
    print("=" * 60)
    rows = []
    for label, r in [("示例组合", port), ("沪深300", returns["沪深300"]), ("创业板", returns["创业板"])]:
        v1, _ = var_cvar_historical(r.to_numpy(), 0.99)
        for h in [5, 10, 20]:
            multi = (np.log1p(r)).rolling(h).sum().dropna().pipe(np.expm1).to_numpy()  # 重叠的 h 日收益
            vh, _ = var_cvar_historical(multi, 0.99)
            rows.append([label, h, f"{v1:.2%}", f"{np.sqrt(h) * v1:.2%}", f"{vh:.2%}", f"{vh / (np.sqrt(h) * v1):.2f}"])
    print(pd.DataFrame(rows, columns=["资产", "持有天数 h", "1日 VaR99", "√h × 1日", "直接用 h 日收益", "比值"]).to_string(index=False))
    print("""
    💡 比值 ≠ 1 说明平方根法则失真：它要求收益独立同分布、均值可忽略。
       波动聚集、趋势（均值不可忽略）、A 股连续跌停都会让它偏离；
       注意重叠窗口样本不独立，h 越大、可用的独立样本越少，直接估计也更不稳定。
    """)


# ==================== 第五部分：VaR 回测 ====================


def kupiec_pof(x: int, T: int, p: float) -> tuple[float, float]:
    """Kupiec POF 检验：返回 (LR, p 值)"""
    def ll(prob):  # 二项对数似然，约定 0·log(0) = 0
        a = 0.0 if x == T else (T - x) * np.log(1 - prob)
        b = 0.0 if x == 0 else x * np.log(prob)
        return a + b
    lr = -2 * (ll(p) - ll(x / T))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen 独立性检验：突破是否聚集"""
    h0, h1 = hits[:-1], hits[1:]
    n00 = int(((h0 == 0) & (h1 == 0)).sum()); n01 = int(((h0 == 0) & (h1 == 1)).sum())
    n10 = int(((h0 == 1) & (h1 == 0)).sum()); n11 = int(((h0 == 1) & (h1 == 1)).sum())

    def xlogy(n, q):
        return 0.0 if n == 0 else n * np.log(q)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (l0 - l1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def var_backtest(returns: pd.DataFrame, window: int = 250):
    print("\n" + "=" * 60)
    print("第五部分：VaR 回测 —— Kupiec 与 Christoffersen 检验")
    print("=" * 60)
    r = returns["沪深300"]
    lam = 0.94
    ewma_var = (r**2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)  # 只用 t-1 及以前的数据
    forecasts = {}
    for a in [0.95, 0.99]:
        z = stats.norm.ppf(1 - a)
        forecasts[("历史法", a)] = -r.rolling(window).quantile(1 - a).shift(1)
        forecasts[("正态参数法", a)] = -(r.rolling(window).mean() + z * r.rolling(window).std()).shift(1)
        forecasts[("EWMA 正态", a)] = -z * np.sqrt(ewma_var)
    test_idx = r.index[window:]
    rows = []
    for (method, a), var in forecasts.items():
        v = var.loc[test_idx]
        hits = (-r.loc[test_idx] > v).astype(int).to_numpy()
        T, x = len(hits), int(hits.sum())
        lr, pv = kupiec_pof(x, T, 1 - a)
        lri, pvi = christoffersen_ind(hits)
        rows.append([method, f"{a:.0%}", T, f"{(1 - a) * T:.0f}", x, f"{x / T:.2%}", f"{lr:.2f}", f"{pv:.3f}",
                     "拒绝" if lr > 3.84 else "不拒绝", f"{lri:.2f}", "拒绝" if lri > 3.84 else "不拒绝"])
    cols = ["方法", "置信度", "天数", "期望突破", "实际突破", "突破率", "LR_POF", "p值", "Kupiec", "LR_ind", "独立性"]
    print(f"标的：沪深300 ETF；滚动窗口 {window} 天；每天只用前一天及以前的数据预测当天 VaR")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("""
    💡 读表：
       - Kupiec 只看"突破次数对不对"：突破太多 → 低估风险；太少 → 过于保守（资本占用太多）
       - 独立性检验看"突破是否扎堆"：滚动窗口的历史法、参数法反应慢，危机中容易连续突破
       - EWMA 对波动变化反应更快，但正态假设在 99% 处往往仍然不够
       - LR > 3.84（5% 显著性下的 χ²(1) 临界值）即拒绝模型
    """)

    v99h = forecasts[("历史法", 0.99)].loc[test_idx]
    v99e = forecasts[("EWMA 正态", 0.99)].loc[test_idx]
    rt = r.loc[test_idx]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rt.index, rt * 100, color="#c3c2b7", lw=0.6, label="日收益")
    ax.plot(v99h.index, -v99h * 100, color="#2a78d6", lw=1.4, label="-VaR99 历史法（250日）")
    ax.plot(v99e.index, -v99e * 100, color="#eb6834", lw=1.2, label="-VaR99 EWMA 正态")
    brk = rt[-rt > v99h]
    ax.scatter(brk.index, brk * 100, color="#e34948", s=14, zorder=3, label=f"历史法突破（{len(brk)} 次）")
    ax.set_ylabel("日收益 %")
    ax.set_title("沪深300 ETF：VaR99 回测")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", ncol=2)
    path = save_figure(fig, "07_var_backtest.png", subdir="phase-8")
    print(f"🖼  VaR 回测图已保存：{path}")


# ==================== 第六部分：最小 CVaR 组合 ====================


def min_cvar_portfolio(returns: pd.DataFrame, alpha: float = 0.95, target: float = 0.08):
    print("\n" + "=" * 60)
    print("第六部分：Rockafellar-Uryasev —— 最小 CVaR 组合")
    print("=" * 60)
    names = list(returns.columns)
    R = returns.to_numpy()
    S, n = R.shape
    mu = R.mean(0)
    Sigma = np.cov(R, rowvar=False)
    t_daily = target / TRADING_DAYS

    w, zeta = cp.Variable(n, nonneg=True), cp.Variable()
    cvar = zeta + cp.sum(cp.pos(-R @ w - zeta)) / ((1 - alpha) * S)
    cp.Problem(cp.Minimize(cvar), [cp.sum(w) == 1, mu @ w >= t_daily]).solve(solver=cp.CLARABEL)
    w_cvar = np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()

    w2 = cp.Variable(n, nonneg=True)
    cp.Problem(cp.Minimize(cp.quad_form(w2, cp.psd_wrap(Sigma * 1e4))), [cp.sum(w2) == 1, mu @ w2 >= t_daily]).solve(solver=cp.CLARABEL)
    w_mv = np.clip(w2.value, 0, None) / np.clip(w2.value, 0, None).sum()

    rows = {}
    for label, wv in [("最小方差", w_mv), ("最小 CVaR", w_cvar)]:
        pr = R @ wv
        v, c = var_cvar_historical(pr, alpha)
        rows[label] = {**{n_: f"{x:.1%}" for n_, x in zip(names, wv)},
                       "年化波动": f"{pr.std() * np.sqrt(TRADING_DAYS):.2%}", f"VaR{alpha:.0%}": f"{v:.3%}",
                       f"CVaR{alpha:.0%}": f"{c:.3%}", "偏度": f"{stats.skew(pr):+.2f}"}
    print(f"约束：多头、满仓、样本年化收益 ≥ {target:.0%}；情景 = {S} 个历史交易日")
    print(pd.DataFrame(rows).T.to_string())
    v_hist, _ = var_cvar_historical(R @ w_cvar, alpha)
    print(f"   最优 ζ = {zeta.value:.3%}，最小 CVaR 组合的历史 VaR = {v_hist:.3%}（ζ 就是 VaR）")
    print("""
    💡 最小方差惩罚的是"所有波动"（包括上涨），最小 CVaR 只惩罚尾部损失。
       两者的差异取决于资产收益的不对称程度；尾部样本少时 CVaR 优化本身也有较大估计误差。
    """)


# ==================== 主函数 ====================


def main():
    returns = load_etf_returns()
    if is_synthetic(returns):
        print("⚠️ 以下结果基于模拟收益（多元 t），只用于理解方法，不代表真实市场\n")
    definitions()
    port = three_methods(returns)
    subadditivity()
    square_root_rule(returns, port)
    var_backtest(returns)
    min_cvar_portfolio(returns)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 GARCH(1,1)（Phase 3 / arch 包）预测条件波动，做 99% VaR 回测，与 EWMA 比较 Kupiec 结果
    2. 实现"过滤历史模拟法"（FHS）：用 EWMA 波动标准化历史收益，再乘以当前波动取分位数
    3. 把第二部分的组合 VaR 分解到各资产（成分 VaR = w_i · ∂VaR/∂w_i，参数法下与第 03 课的 RC 同形）
    4. 在第六部分把 alpha 改为 0.99，最小 CVaR 组合的权重稳定吗？用 Bootstrap 看一看
    """)


if __name__ == "__main__":
    main()
