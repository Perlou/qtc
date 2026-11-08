"""
Markowitz 均值方差与有效前沿
============================

学习目标：
    1. 搞清楚大类资产 ETF 数据的复权口径与起始日期对齐
    2. 理解"组合风险 ≠ 单个资产风险的加权平均"，量化分散化收益
    3. 用闭式解和 cvxpy 两种方式求 GMV、切点组合，并画出有效前沿
    4. 通过 Bootstrap 亲眼看到：最优权重对输入极度敏感（"误差最大化器"）
    5. 用滚动样本外回测，把优化组合和最朴素的 1/N 等权组合放在一起比较

核心概念：
    - 组合收益 μ_p = w'μ，组合方差 σ_p² = w'Σw
    - GMV（全局最小方差）：w = Σ⁻¹1 / 1'Σ⁻¹1，不需要 μ
    - 切点组合（最大 Sharpe）：w ∝ Σ⁻¹(μ - r_f)
    - 有效前沿：每个风险水平下收益最高的组合集合
    - 估计误差：μ 的标准误 ≈ σ / √年数，远比想象中大

前置知识：
    - Phase 2 收益与波动、Phase 3 协方差与蒙特卡洛
    - CONCEPT.md 第 2、3 节

环境要求：
    - pip install numpy pandas cvxpy matplotlib akshare
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

from portfolio_utils import (
    ASSET_COLORS, DATA_END, TRADING_DAYS, annualize, etf_data_report, fmt_weights, gmv_weights,
    load_etf_returns, perf_table, tangency_weights, walk_forward,
)
from utils.data import is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

RF = 0.02  # 年化无风险利率（示例值）


# ==================== 第一部分：资产池与复权口径 ====================


def data_and_adjustment() -> pd.DataFrame:
    print("=" * 60)
    print("第一部分：大类资产 ETF 池与复权口径")
    print("=" * 60)

    print("""
    组合优化的输入是"收益率"，所以先要保证收益率口径正确。ETF 有三件事会让价格跳变：
      ① 现金分红（如 510300 几乎每年分红一次）  ② 份额拆分（如 513100 2022 年 1 拆 5）
      ③ 份额折算 / 合并（如 510500 2015 年）

    ┌──────────────┬──────────────────────────────┬──────────────────────────────┐
    │ 口径         │ 做法                         │ 问题                         │
    ├──────────────┼──────────────────────────────┼──────────────────────────────┤
    │ 不复权       │ 直接用收盘价                 │ 分红日、拆分日出现虚假暴跌   │
    │ 东财 qfq/hfq │ akshare adjust 参数          │ 部分 ETF 是加减法复权，      │
    │              │                              │ 平时的日收益也被压小         │
    │ 本课做法     │ 不复权日收益 + 事件日修正    │ 需要识别事件日（见下表）     │
    └──────────────┴──────────────────────────────┴──────────────────────────────┘
    对多数没有分红、拆分的 ETF（如创业板、黄金、豆粕），三种口径完全一样，直接用不复权价即可。
    """)

    report = etf_data_report()
    show = report.copy()
    for c in ["不复权累计", "东财hfq累计", "修正后累计"]:
        show[c] = show[c].map(lambda x: f"{x:+.1%}")
    print(show.to_string(index=False))

    events = report.attrs["events"]
    if len(events):
        ev = events[["简称", "date", "类型", "每份分红", "不复权收益", "东财hfq收益", "修正后收益"]].copy()
        ev["date"] = ev["date"].dt.date
        for c in ["不复权收益", "东财hfq收益", "修正后收益"]:
            ev[c] = ev[c].map(lambda x: f"{x:+.2%}")
        print("\n🔍 检测到的分红 / 拆分事件：")
        print(ev.to_string(index=False))
    else:
        print("\n（当前数据中没有检测到分红 / 拆分事件）")
    n_synth = int((report["数据"] == "合成").sum())
    if n_synth:
        print(f"\n⚠️ 上表有 {n_synth} 只 ETF 是合成行情（没有本地缓存或下载失败）：合成数据没有公司行为，三种口径相同，下面的读表要点只适用于真实数据")

    print("""
    💡 读表要点：
       - "不复权累计"会把拆分当成暴跌、漏掉现金分红，大类资产里差异最大的是有拆分的 ETF
       - 东财 hfq 对"加减法复权"的 ETF（如 510300）会压低平时的日收益，累计收益偏低
       - 修正规则见 portfolio_utils.etf_total_return：只在事件日使用复权信息
       - 个股请用 utils.data.get_close_panel（新浪等比复权因子），见 Phase 1 第 06 课
       - 纳指、标普 ETF 是 QDII：场内价格可能有溢价，且美股收盘晚于 A 股，
         同日收益的相关性会被低估（Phase 1 第 07 课的日历问题）；本阶段为简化未做调整
    """)

    first_dates = pd.to_datetime(report.set_index("简称")["首个交易日"])
    others_start = first_dates.drop("豆粕").max()
    end = pd.Timestamp(DATA_END)
    print("📅 起始日期对齐：组合需要所有资产同时有数据，样本起点 = 最晚上市的那只。")
    print(f"   不含豆粕时起点为 {others_start.date()}（约 {(end - others_start).days / 365.25:.1f} 年）；"
          f"豆粕 ETF 首个交易日为 {first_dates['豆粕'].date()}，纳入后约 {(end - max(others_start, first_dates['豆粕'])).days / 365.25:.1f} 年。")
    print("   为了保留更长的历史（包含 2015、2016 年等危机），长历史课程默认使用其余 7 只（HRP 一课会单独用上豆粕）。\n")

    returns = load_etf_returns()
    return returns


# ==================== 第二部分：分散化的数学 ====================


def diversification(returns: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：分散化 —— 组合风险不是加权平均")
    print("=" * 60)

    a, b = "沪深300", "国债"
    mu, cov = annualize(returns[[a, b]])
    s1, s2 = np.sqrt(np.diag(cov))
    rho = cov.iloc[0, 1] / (s1 * s2)
    print(f"""
    两资产公式：σ_p² = w₁²σ₁² + w₂²σ₂² + 2·w₁·w₂·ρ·σ₁·σ₂
    {a}：年化波动 {s1:.2%}    {b}：年化波动 {s2:.2%}    相关系数 ρ = {rho:+.2f}
    """)
    rows = []
    for w1 in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]:
        w = np.array([w1, 1 - w1])
        sp = np.sqrt(w @ cov.to_numpy() @ w)
        naive = w1 * s1 + (1 - w1) * s2
        rows.append([f"{w1:.0%}/{1 - w1:.0%}", f"{w @ mu.to_numpy():+.2%}", f"{sp:.2%}", f"{naive:.2%}", f"{naive - sp:.2%}"])
    print(pd.DataFrame(rows, columns=[f"{a}/{b}", "预期收益", "组合波动", "加权平均波动", "分散化收益"]).to_string(index=False))
    print("\n💡 ρ < 1 时组合波动总是小于加权平均波动；ρ 越低，\"免费\"降低的风险越多。")


# ==================== 第三部分：闭式解 ====================


def frontier_closed_form(mu: np.ndarray, Sigma: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """无约束（允许做空）前沿：σ²(m) = (A·m² - 2B·m + C) / D"""
    inv1 = np.linalg.solve(Sigma, np.ones(len(mu)))
    invm = np.linalg.solve(Sigma, mu)
    A, B, C = inv1.sum(), invm.sum(), mu @ invm
    D = A * C - B**2
    return np.sqrt((A * targets**2 - 2 * B * targets + C) / D)


def closed_form_solutions(returns: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：闭式解 —— GMV 与切点组合（允许做空）")
    print("=" * 60)

    mu_s, cov_df = annualize(returns)
    mu, Sigma = mu_s.to_numpy(), cov_df.to_numpy()
    names = returns.columns

    w_gmv = gmv_weights(Sigma)
    w_tan = tangency_weights(mu, Sigma, RF)

    def stats(w):
        r, v = w @ mu, np.sqrt(w @ Sigma @ w)
        return r, v, (r - RF) / v

    table = pd.DataFrame({
        "年化均值": (mu * 100).round(2),
        "年化波动": (np.sqrt(np.diag(Sigma)) * 100).round(2),
        "GMV 权重%": fmt_weights(w_gmv, names),
        "切点权重%": fmt_weights(w_tan, names),
    }, index=names)
    print("全样本估计（单位：%）：")
    print(table.to_string())
    for label, w in [("GMV", w_gmv), ("切点", w_tan)]:
        r, v, sr = stats(w)
        print(f"   {label:4s}: 权重和 = {w.sum():.4f}，收益 {r:+.2%}，波动 {v:.2%}，Sharpe(rf={RF:.0%}) {sr:.2f}，"
              f"做空仓位合计 {w[w < 0].sum():+.1%}")

    # 只看权益类资产：彼此高度相关，Σ 接近病态，闭式解更容易出现一多一空
    eq = [c for c in names if c not in ("国债", "黄金")]
    idx = [list(names).index(c) for c in eq]
    mu_e, S_e = mu[idx], Sigma[np.ix_(idx, idx)]
    w_tan_e = tangency_weights(mu_e, S_e, RF)
    eig = np.linalg.eigvalsh(S_e)
    print(f"\n只含权益类的 {len(eq)} 只 ETF（条件数 κ = {eig.max() / eig.min():.0f}）：")
    print(pd.DataFrame({"切点权重%": fmt_weights(w_tan_e, eq), "GMV 权重%": fmt_weights(gmv_weights(S_e), eq)}).T.to_string())
    print(f"   切点组合：多头合计 {w_tan_e[w_tan_e > 0].sum():+.0%}，空头合计 {w_tan_e[w_tan_e < 0].sum():+.0%}")

    print("""
    💡 观察：
       - GMV 完全不看 μ，自然重仓波动最低的资产
       - 资产池里有一只低波动、低相关的资产（国债）时，它会主导两个组合；
         去掉它之后，高相关的权益资产之间更容易出现"多一只、空一只"的对冲仓位
       - 两个公式都用 np.linalg.solve(Σ, ·) 而不是 inv(Σ)——数值更稳定
    """)
    return mu, Sigma, w_gmv, w_tan


# ==================== 第四部分：cvxpy 有效前沿 ====================


def max_sharpe_long_only(mu: np.ndarray, Sigma: np.ndarray, rf: float = RF) -> np.ndarray | None:
    """
    多头约束下的最大 Sharpe：令 y = w / k，问题变成凸的
        min y'Σy  s.t. (μ - rf)'y = 1, y ≥ 0，最后 w = y / Σy
    """
    if np.all(mu <= rf):
        return None  # 没有资产跑赢无风险利率，切点组合不存在
    y = cp.Variable(len(mu), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))), [(mu - rf) @ y == 1])
    prob.solve(solver=cp.CLARABEL)
    if y.value is None:
        return None
    w = np.clip(y.value, 0, None)
    return w / w.sum()


def min_var_long_only(Sigma: np.ndarray) -> np.ndarray:
    w = cp.Variable(len(Sigma), nonneg=True)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), [cp.sum(w) == 1]).solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


def cvxpy_frontier(returns, mu, Sigma, w_gmv, w_tan):
    print("\n" + "=" * 60)
    print("第四部分：cvxpy 求解多头约束下的有效前沿")
    print("=" * 60)

    n = len(mu)
    w = cp.Variable(n)
    target = cp.Parameter()  # 用 Parameter 包装目标收益，扫描时复用问题结构
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), [cp.sum(w) == 1, mu @ w >= target, w >= 0])
    pts = []
    for t in np.linspace(mu.min(), mu.max(), 30):
        target.value = t
        prob.solve(solver=cp.CLARABEL)
        if prob.status == "optimal":
            pts.append((np.sqrt(w.value @ Sigma @ w.value), mu @ w.value))
    pts = np.array(pts)

    # 用 cvxpy 再解一次"允许做空"的 GMV，与闭式解核对
    w_free = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(w_free, cp.psd_wrap(Sigma))), [cp.sum(w_free) == 1]).solve(solver=cp.CLARABEL)
    print(f"✅ 允许做空的 GMV：cvxpy 与闭式解最大权重差 = {np.abs(w_free.value - w_gmv).max():.2e}")

    w_gmv_lo = min_var_long_only(Sigma)
    w_tan_lo = max_sharpe_long_only(mu, Sigma)
    names = returns.columns
    print("\n多头约束（w ≥ 0）下的解（权重 %）：")
    lo = pd.DataFrame({"GMV(多头)": fmt_weights(w_gmv_lo, names)})
    if w_tan_lo is not None:
        lo["切点(多头)"] = fmt_weights(w_tan_lo, names)
    print(lo.T.to_string())
    print(f"   前沿上共 {len(pts)} 个点；多头切点组合持有 {(w_tan_lo > 1e-3).sum() if w_tan_lo is not None else 0} 只资产"
          "——多头约束本身就是一种\"收缩\"，会砍掉极端仓位")

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    m_grid = np.linspace(min(mu.min(), 0), mu.max() * 1.3, 200)
    ax.plot(frontier_closed_form(mu, Sigma, m_grid), m_grid, color="#8a8984", lw=1.5, ls="--", label="允许做空（闭式解）")
    ax.plot(pts[:, 0], pts[:, 1], color="#2a78d6", lw=2, label="多头约束（cvxpy）")
    for i, name in enumerate(names):
        ax.scatter(np.sqrt(Sigma[i, i]), mu[i], s=50, color=ASSET_COLORS[name], zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(name, (np.sqrt(Sigma[i, i]), mu[i]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    for label, wv, mk in [("GMV", w_gmv, "s"), ("切点", w_tan, "*")]:
        ax.scatter(np.sqrt(wv @ Sigma @ wv), wv @ mu, marker=mk, s=140, color="#0b0b0b", zorder=4)
        ax.annotate(f"{label}（允许做空）", (np.sqrt(wv @ Sigma @ wv), wv @ mu), xytext=(6, -12), textcoords="offset points", fontsize=9)
    if w_tan_lo is not None:
        s_t, m_t = np.sqrt(w_tan_lo @ Sigma @ w_tan_lo), w_tan_lo @ mu
        xs = np.linspace(0, max(np.sqrt(np.diag(Sigma))) * 1.05, 50)
        ax.plot(xs, RF + (m_t - RF) / s_t * xs, color="#eb6834", lw=1.5, label="资本市场线（多头切点）")
        ax.scatter(s_t, m_t, marker="*", s=180, color="#eb6834", zorder=4)
    ax.set_xlim(0, max(np.sqrt(np.diag(Sigma))) * 1.1)
    ax.set_ylim(min(0, mu.min()) - 0.01, mu.max() * 1.35)
    ax.set_xlabel("年化波动 σ")
    ax.set_ylabel("年化预期收益 μ（样本均值）")
    ax.set_title("有效前沿：允许做空 vs 多头约束")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    path = save_figure(fig, "01_efficient_frontier.png", subdir="phase-8")
    print(f"\n🖼  有效前沿图已保存：{path}")


# ==================== 第五部分：估计误差与 Bootstrap ====================


def estimation_error(returns: pd.DataFrame, n_boot: int = 200, seed: int = 42):
    print("\n" + "=" * 60)
    print("第五部分：估计误差 —— 均值方差是\"误差最大化器\"")
    print("=" * 60)

    names = returns.columns
    years = len(returns) / TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    print(f"\n① 均值的标准误 ≈ σ / √年数（样本 {years:.1f} 年）：")
    se = pd.DataFrame({"年化均值%": (returns.mean() * TRADING_DAYS * 100).round(2),
                       "标准误%": (vol / np.sqrt(years) * 100).round(2)})
    se["95%区间%"] = [f"[{m - 1.96 * s:+.1f}, {m + 1.96 * s:+.1f}]" for m, s in zip(se["年化均值%"], se["标准误%"])]
    print(se.to_string())
    cross = se.index[(se["年化均值%"] - 1.96 * se["标准误%"] < 0)].tolist()
    print(f"   → 95% 区间跨过 0 的资产：{cross or '无'}；各资产区间大幅重叠，{years:.0f} 年数据也很难分辨谁的期望收益更高")

    # ② Bootstrap：对日收益有放回重抽样，重新估计 μ、Σ 再优化
    rng = np.random.default_rng(seed)
    X = returns.to_numpy()
    T = len(X)
    boot_w = []
    for _ in range(n_boot):
        sample = X[rng.integers(0, T, T)]
        mu_b = sample.mean(0) * TRADING_DAYS
        cov_b = np.cov(sample, rowvar=False) * TRADING_DAYS
        w = max_sharpe_long_only(mu_b, cov_b)
        if w is not None:
            boot_w.append(w)
    boot_w = np.array(boot_w)
    mu, Sigma = annualize(returns)
    w_full = max_sharpe_long_only(mu.to_numpy(), Sigma.to_numpy())
    summary = pd.DataFrame({
        "全样本%": w_full * 100,
        "均值%": boot_w.mean(0) * 100,
        "标准差%": boot_w.std(0) * 100,
        "5%分位": np.quantile(boot_w, 0.05, axis=0) * 100,
        "95%分位": np.quantile(boot_w, 0.95, axis=0) * 100,
        "权重为0占比": (boot_w < 1e-3).mean(0),
    }, index=names).round(1)
    print(f"\n② Bootstrap {len(boot_w)} 次（同一段历史的\"另一种可能\"），多头最大 Sharpe 权重：")
    print(summary.to_string())
    print("   → 同样的数据只是重新抽样，最优权重的 5%~95% 区间就很宽，不少资产在部分抽样中权重直接为 0")
    print("   → 把各次结果取平均（\"均值%\"列）就是 Michaud 重抽样前沿的思想：权重更分散、更平滑")

    # ③ 闭式解的敏感性：μ 加 1% 噪声
    n_trials = 500
    print(f"\n③ 允许做空的切点组合：给年化 μ 加 N(0, 1%) 噪声，重复 {n_trials} 次，看权重变化的 L1 范数")
    eq = [c for c in names if c not in ("国债", "黄金")]
    for label, cols in [("全部 7 只", list(names)), ("仅权益类", eq)]:
        m0, S0 = mu[cols].to_numpy(), Sigma.loc[cols, cols].to_numpy()
        w0 = tangency_weights(m0, S0, RF)
        l1 = np.array([np.abs(tangency_weights(m0 + rng.normal(0, 0.01, len(m0)), S0, RF) - w0).sum()
                       for _ in range(n_trials)])
        print(f"   {label}：中位数 {np.median(l1):.0%}，90% 分位 {np.quantile(l1, 0.9):.0%}，最大 {l1.max():.0%}")
    print("   （1% 的噪声远小于上面算出的均值标准误；最大值巨大是因为分母 1'Σ⁻¹(μ-rf) 偶尔接近 0）")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([boot_w[:, i] * 100 for i in range(len(names))], tick_labels=list(names), showfliers=False)
    ax.scatter(range(1, len(names) + 1), w_full * 100, color="#eb6834", zorder=3, s=40, label="全样本最优")
    ax.set_ylabel("权重 %")
    ax.set_title(f"Bootstrap {len(boot_w)} 次的多头最大 Sharpe 权重分布")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    path = save_figure(fig, "01_bootstrap_weights.png", subdir="phase-8")
    print(f"🖼  Bootstrap 权重分布图已保存：{path}")


# ==================== 第六部分：样本外对比 1/N ====================


def out_of_sample(returns: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：样本外回测 —— 优化组合 vs 1/N")
    print("=" * 60)
    print("""
    规则：每月末只用"过去 252 个交易日"估计 μ、Σ，求权重，持有到下月末（权重随价格漂移）。
    这样每一期的权重都只用到当时能拿到的数据，没有前视偏差。
    """)

    def equal(x):
        return np.ones(x.shape[1]) / x.shape[1]

    def gmv(x):
        return min_var_long_only(x.cov().to_numpy() * TRADING_DAYS)

    def max_sharpe(x):
        mu_, cov_ = x.mean().to_numpy() * TRADING_DAYS, x.cov().to_numpy() * TRADING_DAYS
        w = max_sharpe_long_only(mu_, cov_)
        return w if w is not None else min_var_long_only(cov_)  # 无资产跑赢 rf 时退回 GMV

    rets, extra = {}, {}
    for label, fn in [("1/N 等权", equal), ("GMV（多头）", gmv), ("最大Sharpe（多头）", max_sharpe)]:
        r, W, to = walk_forward(returns, fn, lookback=252, freq="ME")
        rets[label] = r
        extra[label] = {"年化换手": f"{to:.0%}", "平均最大单仓": f"{W.max(axis=1).mean():.0%}"}
    print(f"样本外区间：{rets['1/N 等权'].index[0].date()} ~ {rets['1/N 等权'].index[-1].date()}（Sharpe 按 rf=0 计算，未扣交易成本）")
    print(perf_table(rets, extra).to_string())
    print("""
    💡 如何读这张表：
       - GMV 只依赖 Σ，波动通常最低；最大 Sharpe 依赖最难估计的 μ，换手通常最高
       - 这里的 Sharpe 没有扣无风险利率，会抬高重仓国债的组合（国债收益本身接近无风险利率），
         比较时要结合收益和波动一起看
       - 最大 Sharpe 的样本外 Sharpe 能否超过 1/N，取决于具体样本——这正是 DeMiguel 等（2009）
         的结论：许多优化组合在样本外并不能稳定击败 1/N
       - 任何新方法都应先和 1/N、GMV 做同样的样本外对比
    """)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#8a8984", "#2a78d6", "#eb6834"]
    for (label, r), c in zip(rets.items(), colors):
        ax.plot((1 + r).cumprod(), label=label, color=c, lw=1.8)
    ax.set_title("样本外净值：1/N vs 均值方差优化（月度再平衡）")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
    path = save_figure(fig, "01_oos_nav.png", subdir="phase-8")
    print(f"🖼  样本外净值图已保存：{path}")


# ==================== 主函数 ====================


def main():
    returns = data_and_adjustment()
    if is_synthetic(returns):
        print("⚠️ 以下结果基于模拟收益，只用于理解方法，不代表真实市场\n")
    diversification(returns)
    mu, Sigma, w_gmv, w_tan = closed_form_solutions(returns)
    cvxpy_frontier(returns, mu, Sigma, w_gmv, w_tan)
    estimation_error(returns)
    out_of_sample(returns)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 RF 改为 0% 和 4%，切点组合怎样变化？为什么允许做空的切点组合在 RF 接近 GMV 收益时会"爆炸"？
       （提示：tangency_weights 分母 1'Σ⁻¹(μ - rf) 接近 0）
    2. 实现 Michaud 重抽样：在第六部分的每个再平衡日做 30 次 Bootstrap 并平均权重，比较样本外表现
    3. 第五部分改用"块 Bootstrap"（每次抽连续 20 天），权重的离散程度会变大还是变小？为什么？
    4. 把 lookback 改为 60 和 756 天，最大 Sharpe 组合的换手率与 Sharpe 怎么变？
    """)


if __name__ == "__main__":
    main()
