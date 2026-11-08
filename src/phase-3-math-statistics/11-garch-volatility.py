"""
波动率聚集与 GARCH
==================

学习目标：
    1. 理解波动率聚集："方向难测，幅度可测"
    2. 理解 GARCH(1,1)：今天的方差 = 长期基准 + 昨天的冲击 + 昨天方差的惯性
    3. 用模拟数据验证 arch 库能还原真实参数，再拟合沪深300
    4. 计算持续性 α+β、冲击半衰期、长期波动率，并预测未来 10 日波动
    5. 比较 GJR-GARCH、EGARCH 与不同误差分布
    6. 做样本外评估：GARCH vs EWMA vs 历史波动；并用于动态 VaR 和波动率目标

核心概念：
    - r_t = μ + ε_t，ε_t = σ_t·z_t；σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²
    - 持续性 α+β < 1；长期方差 ω/(1-α-β)；半衰期 ln0.5 / ln(α+β)
    - 杠杆效应：坏消息比好消息更推高波动（GJR 的 γ > 0）

前置知识：
    - 02-fat-tails-normality.py（VaR）、10-arma-models.py（收益平方的自相关）

环境要求：
    - pip install numpy pandas arch matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据没有波动聚集：真实数据部分的 GARCH 参数没有意义，
      第一、二部分的 GARCH 模拟数据不受影响）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from dotenv import load_dotenv
from scipy import stats

from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

TRADING_DAYS = 252
START, END = "20100101", "20251231"


def simulate_garch(n: int, omega: float, alpha: float, beta: float, df: float | None = None, seed: int = 0) -> np.ndarray:
    """模拟 GARCH(1,1) 收益（百分比单位，均值 0）；df 不为 None 时使用标准化 t 分布扰动"""
    rng = np.random.default_rng(seed)
    if df is None:
        z = rng.standard_normal(n)
    else:
        z = stats.t.rvs(df, size=n, random_state=rng) / np.sqrt(df / (df - 2))
    r = np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


def half_life(persistence: float) -> float:
    return np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf


# ==================== 第一部分：波动率聚集 ====================


def volatility_clustering(r_pct: pd.Series):
    print("=" * 60)
    print("第一部分：波动率聚集 —— 大波动之后往往跟着大波动")
    print("=" * 60)

    print("""
    🌦️ 比喻：天气。暴风雨之后第二天大概率还是阴雨，晴天之后大概率还是晴天。
       你无法预测明天往哪个方向刮风（涨跌），但能较好地预测风有多大（波动幅度）。
    """)

    garch_sim = simulate_garch(len(r_pct), omega=0.02, alpha=0.08, beta=0.90, seed=1)
    iid = np.random.default_rng(2).normal(0, garch_sim.std(), len(r_pct))
    rows = []
    for name, x in [("沪深300 真实", r_pct.to_numpy()), ("GARCH 模拟", garch_sim), ("i.i.d. 正态", iid)]:
        big = np.abs(x) > np.quantile(np.abs(x), 0.9)  # "大波动日"：|r| 位于前 10%
        p_after_big = big[1:][big[:-1]].mean()
        a = np.abs(x) - np.abs(x).mean()
        acf1 = (a[1:] * a[:-1]).sum() / (a * a).sum()
        rows.append([name, f"{acf1:.3f}", f"{p_after_big:.1%}", f"{stats.kurtosis(x):.2f}"])
    print(pd.DataFrame(rows, columns=["序列", "|r| 的 lag1 自相关", "大波动日次日仍是大波动日的概率", "超额峰度"]).to_string(index=False))
    print("""
    💡 若无聚集，"大波动日次日仍是大波动日"的概率应约等于 10%（i.i.d. 那一行）。
       明显高于 10% 就是波动率聚集。注意 GARCH 即使用正态扰动，也会产生肥尾（超额峰度 > 0）。
    """)
    if is_synthetic(r_pct.index):
        print("    ⚠️ 离线合成数据（GBM）没有波动聚集，'沪深300 真实'一行会与 i.i.d. 相近；请看 GARCH 模拟一行")


# ==================== 第二部分：GARCH(1,1) 与参数还原 ====================


def garch_recovery():
    print("\n" + "=" * 60)
    print("第二部分：GARCH(1,1) 模型 —— 先在模拟数据上验证")
    print("=" * 60)

    print("""
    r_t = μ + ε_t，   ε_t = σ_t · z_t，   z_t ~ i.i.d.(0, 1)
    σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²

       ω > 0：常数项      α ≥ 0：对昨日冲击的反应（新闻效应）      β ≥ 0：对昨日方差的记忆（持续性）
       α + β < 1 才能保证方差平稳

    长期平均方差 = ω / (1 - α - β)
    波动冲击半衰期 = ln 0.5 / ln(α + β)
    """)

    true = {"omega": 0.03, "alpha[1]": 0.10, "beta[1]": 0.87, "nu": 6.0}
    r = simulate_garch(4000, true["omega"], true["alpha[1]"], true["beta[1]"], df=true["nu"], seed=3)
    res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    rows = [[k, v, res.params[k], res.std_err[k]] for k, v in true.items()]
    print("🧪 用已知参数模拟 4000 天 GARCH(1,1)-t，再用 arch 拟合：")
    print(pd.DataFrame(rows, columns=["参数", "真实值", "估计值", "标准误"]).round(4).to_string(index=False))
    print("""
    💡 估计值应落在真实值附近（大约 ±2 个标准误内）—— 先在"已知答案"的数据上验证工具，再用于真实数据。
    """)


# ==================== 第三部分：拟合沪深300 ====================


def fit_real(r_pct: pd.Series):
    print("\n" + "=" * 60)
    print("第三部分：拟合沪深300 日收益")
    print("=" * 60)

    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    print("📐 arch_model(r_pct, mean='Constant', vol='GARCH', p=1, q=1, dist='t')")
    print("   ⚠️ 收益放大到百分比（×100），数值优化更稳定；结果也是'百分比'单位\n")
    table = pd.DataFrame({"估计值": res.params, "标准误": res.std_err, "t 值": res.tvalues, "p 值": res.pvalues})
    print(table.round(4).to_string())

    omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    pers = alpha + beta
    long_vol = np.sqrt(omega / (1 - pers) * TRADING_DAYS) if pers < 1 else np.nan
    print(f"\n   持续性 α + β = {pers:.4f}")
    print(f"   冲击半衰期 ≈ {half_life(pers):.1f} 天")
    print(f"   长期年化波动 ≈ √(ω/(1-α-β) × 252) = {long_vol:.1f}%（样本年化波动 {r_pct.std() * np.sqrt(TRADING_DAYS):.1f}%）")
    nu = res.params["nu"]
    tail_note = "标准化残差仍有肥尾" if nu < 30 else "已接近正态，标准化残差几乎没有肥尾"
    print(f"   t 分布自由度 ν = {nu:.2f}（{tail_note}）")
    print("""
    💡 金融日收益拟合的 α+β 通常很接近 1：波动冲击衰减很慢、持续性很强。
    """)

    fc = res.forecast(horizon=10, reindex=False)
    var_path = fc.variance.iloc[-1].to_numpy()
    cond_vol = res.conditional_volatility * np.sqrt(TRADING_DAYS)
    print(f"🔮 最新条件波动（年化）= {cond_vol.iloc[-1]:.1f}%，未来 1~10 日年化波动预测：")
    print("   " + "  ".join(f"{v:.1f}%" for v in np.sqrt(var_path * TRADING_DAYS)))
    print(f"   → 预测会逐步向长期波动 {long_vol:.1f}% 回归（当前高于长期水平则下降，低于则上升）")
    if is_synthetic(r_pct.index):
        print("    ⚠️ 离线合成数据没有波动聚集，以上 α、β 没有经济含义")
    return res


# ==================== 第四部分：GARCH 家族与误差分布 ====================


def garch_family(r_pct: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：GARCH 家族与误差分布")
    print("=" * 60)

    specs = {
        "GARCH-正态": dict(vol="GARCH", p=1, q=1, dist="normal"),
        "GARCH-t": dict(vol="GARCH", p=1, q=1, dist="t"),
        "GARCH-skewt": dict(vol="GARCH", p=1, q=1, dist="skewt"),
        "GJR-GARCH-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
        "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    }
    rows = []
    fits = {}
    for name, kw in specs.items():
        res = arch_model(r_pct, mean="Constant", **kw).fit(disp="off")
        fits[name] = res
        rows.append([name, len(res.params), res.loglikelihood, res.aic, res.bic])
    table = pd.DataFrame(rows, columns=["模型", "参数个数", "对数似然", "AIC", "BIC"])
    print(table.round(1).to_string(index=False))
    print(f"   AIC 最优：{table.loc[table['AIC'].idxmin(), '模型']}；BIC 最优：{table.loc[table['BIC'].idxmin(), '模型']}")

    gjr = fits["GJR-GARCH-t"]
    g, g_p = gjr.params["gamma[1]"], gjr.pvalues["gamma[1]"]
    print(f"\n🔍 GJR-GARCH：σ_t² = ω + (α + γ·1[ε_{{t-1}}<0])·ε_{{t-1}}² + β·σ_{{t-1}}²")
    print(f"   γ = {g:.4f}，p = {g_p:.4f} → {'下跌冲击对波动的影响显著更大（存在杠杆效应）' if g > 0 and g_p < 0.05 else '在 5% 水平下，没有足够证据支持杠杆效应'}")
    print("""
    💡 杠杆效应在美股指数上比较显著；在 A 股上的证据不完全一致 —— 用自己的数据检验 γ，而不是直接假设。
       t / skewt 分布通常比正态分布的似然更高：即使除以条件波动率，标准化残差仍有肥尾。
    """)


# ==================== 第五部分：样本外评估与应用 ====================


def out_of_sample(r_pct: pd.Series):
    print("\n" + "=" * 60)
    print("第五部分：样本外评估 —— GARCH vs EWMA vs 历史波动")
    print("=" * 60)

    split = int(len(r_pct) * 0.6)
    split_date = r_pct.index[split]
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off", last_obs=split_date)  # 只用 split 之前的数据估计参数
    fc = res.forecast(horizon=1, start=split_date, reindex=False)
    garch_var = fc.variance["h.1"].shift(1).dropna()  # t 日收盘做出的预测，对应 t+1 日
    test = r_pct.loc[garch_var.index]

    ewma_var = (r_pct**2).ewm(alpha=1 - 0.94, adjust=False).mean().shift(1).loc[test.index]  # RiskMetrics λ=0.94
    hist_var = (r_pct**2).rolling(20).mean().shift(1).loc[test.index]
    const_var = pd.Series((r_pct.iloc[:split] ** 2).mean(), index=test.index)
    realized = test**2  # 单日收益平方作为"真实方差"的代理（噪声很大）

    rows = []
    for name, v in [("GARCH(1,1)-t", garch_var), ("EWMA λ=0.94", ewma_var), ("20 日历史", hist_var), ("训练期常数", const_var)]:
        mse = np.mean((realized - v) ** 2)
        qlike = np.mean(realized / v - np.log(realized / v) - 1)  # QLIKE 损失：对波动预测更稳健
        rows.append([name, mse, qlike])
    table = pd.DataFrame(rows, columns=["预测方法", "MSE（越小越好）", "QLIKE（越小越好）"])
    print(f"📊 {split_date.date()} 之前估计参数，之后 {len(test)} 天逐日一步预测（GARCH 参数不再更新）：")
    print(table.round(4).to_string(index=False))
    print(f"   QLIKE 最优：{table.loc[table['QLIKE（越小越好）'].idxmin(), '预测方法']}")
    print("""
    💡 GARCH 常常只比 EWMA 好一点点，有时甚至不如它 —— 样本内拟合 ≠ 预测能力。
       用单日收益平方当"真实方差"噪声极大；更好的代理是日内高频数据算的已实现波动率。
    """)

    # 应用一：动态 VaR
    nu = res.params["nu"]
    q01 = stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)  # 标准化 t 分布的 1% 分位数
    mu = res.params["mu"]
    var_garch = -(mu + np.sqrt(garch_var) * q01)
    var_static = -np.quantile(r_pct.iloc[:split], 0.01)
    breach_g = (test < -var_garch).mean()
    breach_s = (test < -var_static).mean()
    print("📌 应用一：99% 动态 VaR（样本外突破率，理论值 1%）")
    print(f"   固定的历史 VaR（训练期 1% 分位）：突破率 {breach_s:.2%}")
    print(f"   GARCH-t 动态 VaR：                 突破率 {breach_g:.2%}")
    hi = garch_var > garch_var.median()
    print(f"   高波动期 / 低波动期的突破率：固定 VaR {((test < -var_static) & hi).sum() / hi.sum():.2%} / "
          f"{((test < -var_static) & ~hi).sum() / (~hi).sum():.2%}，GARCH VaR {((test < -var_garch) & hi).sum() / hi.sum():.2%} / "
          f"{((test < -var_garch) & ~hi).sum() / (~hi).sum():.2%}")
    print("   → 固定 VaR 的突破往往集中在高波动期；动态 VaR 会随波动率自动调整")

    # 应用二：波动率目标
    target = 15.0
    vol_fc = np.sqrt(garch_var * TRADING_DAYS)
    weight = (target / vol_fc).clip(upper=1.5)
    strat = weight * test
    rows = []
    for name, r in [("买入持有", test), (f"波动率目标 {target:.0f}%", strat)]:
        nav = (1 + r / 100).cumprod()
        roll_vol = r.rolling(60).std() * np.sqrt(TRADING_DAYS)
        rows.append([name, f"{r.std() * np.sqrt(TRADING_DAYS):.1f}%", f"{roll_vol.min():.1f}% ~ {roll_vol.max():.1f}%",
                     f"{(nav / nav.cummax() - 1).min():.1%}", f"{r.mean() / r.std() * np.sqrt(TRADING_DAYS):.2f}"])
    print(f"\n📌 应用二：波动率目标（仓位 = {target:.0f}% / GARCH 预测波动，上限 1.5 倍）")
    print(pd.DataFrame(rows, columns=["策略", "年化波动", "60 日滚动波动范围", "最大回撤", "Sharpe"]).to_string(index=False))
    print(f"   平均仓位 = {weight.mean():.2f}，仓位范围 {weight.min():.2f} ~ {weight.max():.2f}")
    print("""
    💡 波动率目标的作用是让组合的"实际波动"更接近目标、高波动期自动减仓（对比滚动波动范围）；
       它本身不保证提高收益或 Sharpe，效果要看具体数据（Phase 5 / Phase 8 会深入）。
       ⚠️ A 股注意：涨跌停会截断收益；停牌后复牌的跳空会被 GARCH 当作一次超大冲击，污染后续估计。
    """)
    return garch_var, ewma_var, test


def plot_garch(res, r_pct: pd.Series, garch_var: pd.Series, ewma_var: pd.Series):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(r_pct.index, r_pct, lw=0.5, color="gray")
    axes[0].set_title("沪深300 日收益（%）：波动率聚集")
    cond = res.conditional_volatility * np.sqrt(TRADING_DAYS)
    axes[1].plot(cond.index, cond, lw=0.8, label="GARCH 条件波动（样本内）")
    axes[1].plot(garch_var.index, np.sqrt(garch_var * TRADING_DAYS), lw=0.8, label="GARCH 样本外预测")
    axes[1].plot(ewma_var.index, np.sqrt(ewma_var * TRADING_DAYS), lw=0.8, alpha=0.7, label="EWMA 样本外")
    axes[1].set_title("年化波动率（%）")
    axes[1].legend()
    path = save_figure(fig, "11_garch.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    close = get_index_daily("000300", START, END)["close"]
    r_pct = np.log(close).diff().dropna() * 100
    print(f"📊 数据：沪深300 {r_pct.index[0].date()} ~ {r_pct.index[-1].date()}，{len(r_pct)} 个日收益（百分比单位）\n")
    volatility_clustering(r_pct)
    garch_recovery()
    res = fit_real(r_pct)
    garch_family(r_pct)
    garch_var, ewma_var, _ = out_of_sample(r_pct)
    plot_garch(res, r_pct, garch_var, ewma_var)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 对茅台和中证1000 分别拟合 GARCH(1,1)-t，比较 α、β、持续性和半衰期，解释差异
    2. 第五部分改为每 250 天重新估计一次参数（扩展窗口），QLIKE 是否改善？
    3. 把 EWMA 的 λ 从 0.94 改为 0.97、0.90，哪个最好？这个"最好"是在什么数据上选出来的？
    4. 用拟合好的 GARCH-t 参数模拟 1000 条一年期路径，与 04 课 GBM 的一年期最大回撤分布比较
    """)


if __name__ == "__main__":
    main()
