"""
OLS 回归与残差诊断
==================

学习目标：
    1. 理解 OLS 的直觉：让所有点到直线的竖直距离平方和最小
    2. 用矩阵公式手算 β̂ = (XᵀX)⁻¹Xᵀy，并与 statsmodels 对照
    3. 读懂 model.summary() 中的关键输出
    4. 做残差诊断：自相关、异方差、共线性、正态性，并用稳健标准误修正
    5. 亲手制造伪回归：两条毫无关系的随机游走也能"高度显著"

核心概念：
    - y = Xβ + ε，β̂ = (XᵀX)⁻¹Xᵀy
    - R² = 1 - Σε̂² / Σ(y - ȳ)²：自变量解释的方差比例
    - HC / HAC 稳健标准误：系数不变，标准误和 t 值变化
    - 伪回归：非平稳序列（价格）之间的回归会制造虚假的显著关系

前置知识：
    - 03-hypothesis-testing.py；Phase 2 的 Beta 概念

环境要求：
    - pip install numpy pandas statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据中各股票彼此独立，回归系数会接近 0）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.data import get_close_panel, get_index_daily, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRADING_DAYS = 252
START, END = "20180101", "20241231"
SECTOR = {"000858": "五粮液", "000568": "泸州老窖", "600809": "山西汾酒"}


def load_data() -> pd.DataFrame:
    """茅台日收益 y、沪深300 日收益 mkt、白酒板块（3 只白酒股等权）日收益 sector"""
    panel = get_close_panel(["600519", *SECTOR], START, END)
    rets = panel.pct_change()
    hs300 = get_index_daily("000300", START, END)["close"].pct_change()
    data = pd.concat(
        [rets["600519"], hs300, rets[list(SECTOR)].mean(axis=1)], axis=1, keys=["y", "mkt", "sector"]
    ).dropna()
    return data


# ==================== 第一部分：OLS 的直觉与手算 ====================


def ols_by_hand(data: pd.DataFrame):
    print("=" * 60)
    print("第一部分：OLS 的直觉与矩阵手算")
    print("=" * 60)

    print("""
    📏 OLS：在散点图里画一条直线，使所有点到直线的"竖直距离平方和"最小。

       单变量：y_t = α + β·x_t + ε_t
       矩阵：  y = Xβ + ε，  β̂ = (XᵀX)⁻¹ Xᵀ y      （X 第一列全为 1，对应截距 α）

    量化中最常见的用途：
       个股 ~ 市场 → Beta、Alpha；收益 ~ 因子 → 因子暴露（Phase 7）；A ~ B → 对冲比例（Phase 6）
    """)

    X = np.column_stack([np.ones(len(data)), data["mkt"].to_numpy()])
    y = data["y"].to_numpy()
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)  # 等价于 inv(XᵀX) @ Xᵀy，但数值更稳定
    resid = y - X @ beta_hat
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

    model = sm.OLS(data["y"], sm.add_constant(data["mkt"])).fit()
    print("📐 茅台日收益 ~ 沪深300 日收益：")
    print(f"   手算：       α = {beta_hat[0]:+.6f}，β = {beta_hat[1]:.4f}，R² = {r2:.4f}")
    print(f"   statsmodels：α = {model.params['const']:+.6f}，β = {model.params['mkt']:.4f}，R² = {model.rsquared:.4f}")
    print(f"   年化 Alpha ≈ α × 252 = {beta_hat[0] * TRADING_DAYS:+.2%}")
    print(f"   验证：β = Cov(y, x) / Var(x) = {np.cov(y, data['mkt'])[0, 1] / data['mkt'].var():.4f}")
    print("""
    ⚠️ 别忘了常数项：sm.OLS(y, X) 不会自动加截距，要用 sm.add_constant(X)。
       不加截距等于强制直线过原点，β 会被扭曲。
    """)
    return model


# ==================== 第二部分：多元回归与 summary 解读 ====================


def multi_regression(data: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：多元回归与 summary() 解读")
    print("=" * 60)

    X = sm.add_constant(data[["mkt", "sector"]])
    model = sm.OLS(data["y"], X).fit()
    single = sm.OLS(data["y"], sm.add_constant(data["mkt"])).fit()
    print("📐 茅台 ~ 沪深300 + 白酒板块（五粮液、泸州老窖、山西汾酒等权）：\n")
    print(model.summary())
    print("""
    ┌──────────────────┬────────────────────────┬──────────────────────────────┐
    │ 输出项           │ 含义                   │ 关注点                       │
    ├──────────────────┼────────────────────────┼──────────────────────────────┤
    │ coef             │ 系数估计               │ 方向和大小是否合理           │
    │ std err          │ 系数标准误             │ 估计的不确定性               │
    │ t, P>|t|         │ 系数是否显著不为 0     │ 注意多重检验（03 课）        │
    │ R-squared        │ 解释的方差比例         │ 收益回归中低 R² 是常态       │
    │ Durbin-Watson    │ 残差一阶自相关         │ 接近 2 表示无一阶自相关      │
    │ Jarque-Bera (JB) │ 残差正态性             │ 金融数据几乎总是拒绝         │
    │ Cond. No.        │ 条件数                 │ 很大则可能存在多重共线性     │
    └──────────────────┴────────────────────────┴──────────────────────────────┘
    """)
    print(f"💡 加入板块因子后 R² 从单因子的 {single.rsquared:.3f} 变为 {model.rsquared:.3f}；"
          f"沪深300 的系数从 {single.params['mkt']:.3f} 变为 {model.params['mkt']:.3f}。")
    print("   当两个自变量相关时，每个系数表示的是'控制另一个变量后'的边际影响，与单变量回归的含义不同。")
    if is_offline():
        print("   ⚠️ 离线合成数据中各序列彼此独立，系数与 R² 接近 0，仅用于跑通流程")
    return model, X


# ==================== 第三部分：残差诊断与稳健标准误 ====================


def residual_diagnostics(model, X: pd.DataFrame, data: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：残差诊断")
    print("=" * 60)

    print("""
    ┌────────────────┬──────────────────────┬──────────────────────┬────────────────────┐
    │ 假设           │ 违反时的后果         │ 诊断                 │ 补救               │
    ├────────────────┼──────────────────────┼──────────────────────┼────────────────────┤
    │ 残差无自相关   │ 标准误偏小，t 虚高   │ DW、Ljung-Box        │ HAC 标准误         │
    │ 残差同方差     │ 标准误不可靠         │ Breusch-Pagan        │ HC / HAC 标准误    │
    │ 自变量低共线   │ 系数不稳定、符号乱跳 │ 相关矩阵、VIF        │ 删变量、正交化     │
    │ 残差近似正态   │ 小样本推断不准       │ JB、Q-Q 图           │ 大样本下影响较小   │
    │ 变量平稳       │ 伪回归               │ ADF（09 课）         │ 用收益而非价格     │
    └────────────────┴──────────────────────┴──────────────────────┴────────────────────┘
    """)

    resid = model.resid
    lb = acorr_ljungbox(resid, lags=[5, 10])
    print("🔍 1) 自相关 —— Ljung-Box（H0：残差前 m 阶自相关全为 0）")
    print(lb.round(4).to_string())

    lm_stat, lm_p, _, _ = het_breuschpagan(resid, X)
    print(f"\n🔍 2) 异方差 —— Breusch-Pagan（H0：残差同方差）：LM = {lm_stat:.2f}，p = {lm_p:.4g}")
    print("      p 小 → 残差方差随自变量变化（例如大盘大涨大跌时，个股残差也更大）")

    print("\n🔍 3) 共线性 —— 自变量相关系数与 VIF（经验上 VIF > 5~10 需要警惕）")
    print(f"      Corr(mkt, sector) = {data['mkt'].corr(data['sector']):.3f}")
    for i, col in enumerate(X.columns):
        if col != "const":
            print(f"      VIF({col}) = {variance_inflation_factor(X.to_numpy(), i):.2f}")

    print("\n🔍 4) 稳健标准误：系数完全相同，变化的是标准误与 t 值")
    fits = {
        "普通 OLS": model,
        "HC3（异方差稳健）": sm.OLS(data["y"], X).fit(cov_type="HC3"),
        "HAC（Newey-West, 5 阶）": sm.OLS(data["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5}),
    }
    rows = []
    for name, f in fits.items():
        for var in X.columns:
            rows.append([name, var, f"{f.params[var]:+.5f}", f"{f.bse[var]:.5f}", f"{f.tvalues[var]:.2f}"])
    print(pd.DataFrame(rows, columns=["标准误类型", "变量", "系数", "标准误", "t 值"]).to_string(index=False))
    print("""
    💡 时间序列回归默认使用 HAC 标准误。常数项（Alpha）的 t 值最值得关注：
       它回答的是"剔除市场和板块后，茅台还有没有显著的超额收益" —— 看实际的 t 值来下结论。
    """)


# ==================== 第四部分：伪回归 ====================


def spurious_regression():
    print("\n" + "=" * 60)
    print("第四部分：伪回归 —— 最危险的陷阱")
    print("=" * 60)

    print("""
    🎲 实验：生成两条完全独立的随机游走 A、B（各 500 步），分别用"价格水平"和"变化量"做回归，
       重复 500 次，看有多少次被判为"显著相关"（|t| > 1.96）。两者真实关系为 0。
    """)
    rng = np.random.default_rng(0)
    n_sims = 500
    hits_level = hits_diff = 0
    r2_level, t_level = [], []
    for _ in range(n_sims):
        a = np.cumsum(rng.standard_normal(500))
        b = np.cumsum(rng.standard_normal(500))
        f1 = sm.OLS(a, sm.add_constant(b)).fit()
        f2 = sm.OLS(np.diff(a), sm.add_constant(np.diff(b))).fit()
        hits_level += abs(f1.tvalues[1]) > 1.96
        hits_diff += abs(f2.tvalues[1]) > 1.96
        r2_level.append(f1.rsquared)
        t_level.append(abs(f1.tvalues[1]))
    print(f"\n🧪 重复 {n_sims} 次（每次 500 步），|t| > 1.96 的比例：")
    print(f"   价格水平回归：{hits_level / n_sims:.1%}（R² 中位数 {np.median(r2_level):.3f}，|t| 中位数 {np.median(t_level):.1f}）")
    print(f"   变化量回归：  {hits_diff / n_sims:.1%}（理论上应接近 5%）")
    print("""
    💡 原因：两个非平稳序列都有"漂"的趋势，回归把共同的漂移误当成关系。
       规则：用收益率（平稳）做回归，不要用价格（非平稳）。
       唯一例外是协整（07 课），但那需要专门检验。
    """)


# ==================== 第五部分：系数会随时间变化 ====================


def rolling_beta(data: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：滚动 Beta —— 系数并不是常数")
    print("=" * 60)

    window = TRADING_DAYS
    cov = data["y"].rolling(window).cov(data["mkt"])
    var = data["mkt"].rolling(window).var()
    beta = (cov / var).dropna()
    yearly = beta.groupby(beta.index.year).mean()
    print(f"📊 茅台对沪深300 的 {window} 日滚动 Beta（按年平均）：")
    print("   " + "  ".join(f"{y}: {v:.2f}" for y, v in yearly.items()))
    print(f"   区间：最小 {beta.min():.2f}，最大 {beta.max():.2f}")
    print("""
    💡 全样本回归给出一个"平均"的 β。回测中如果用全样本 β 做对冲，就用到了未来信息（前视偏差）。
       应使用滚动或扩展窗口估计，只用"当时可得"的数据。
    """)
    return beta


def plot_ols(data: pd.DataFrame, model, beta: pd.Series):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    ax.scatter(data["mkt"], data["y"], s=3, alpha=0.4)
    x = np.linspace(data["mkt"].min(), data["mkt"].max(), 50)
    ax.plot(x, model.params["const"] + model.params["mkt"] * x, color="C3", label=f"β = {model.params['mkt']:.2f}")
    ax.set_xlabel("沪深300 日收益")
    ax.set_ylabel("茅台日收益")
    ax.set_title("OLS 拟合")
    ax.legend()

    ax = axes[1]
    ax.scatter(model.fittedvalues, model.resid, s=3, alpha=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("拟合值")
    ax.set_ylabel("残差")
    ax.set_title("残差图：喇叭形 = 异方差")

    ax = axes[2]
    beta.plot(ax=ax)
    ax.axhline(model.params["mkt"], color="C3", ls="--", label="全样本 β")
    ax.set_title("252 日滚动 Beta")
    ax.legend()
    path = save_figure(fig, "06_ols_diagnostics.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    data = load_data()
    print(f"📊 数据：{data.index[0].date()} ~ {data.index[-1].date()}，{len(data)} 个交易日\n")
    simple = ols_by_hand(data)
    model, X = multi_regression(data)
    residual_diagnostics(model, X, data)
    spurious_regression()
    beta = rolling_beta(data)
    plot_ols(data, simple, beta)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把茅台换成一只银行股（如 600036），对沪深300 回归，比较 β 和 R²，并解释差异
    2. 用 sector 对 mkt 回归，取残差作为"剔除市场后的板块收益"，再做 y ~ mkt + 板块残差，系数如何变化？
    3. 把日收益换成周收益重做第二部分，R² 和 Durbin-Watson 有什么变化？
    4. 在伪回归实验中，把步数从 500 增加到 5000，价格回归的"显著"比例是上升还是下降？为什么？
    """)


if __name__ == "__main__":
    main()
