"""
自相关与 ARMA 模型
==================

学习目标：
    1. 理解 ACF 与 PACF：直接相关 vs 剔除中间期影响后的相关
    2. 用模拟看清 AR(p)、MA(q) 在 ACF/PACF 上的"截尾/拖尾"特征
    3. 用 Ljung-Box 检验收益、收益平方是否存在自相关
    4. 用 statsmodels 拟合 ARMA、用 AIC/BIC 选阶，并做样本外检验
    5. 理解为什么 ARMA 预测收益"不太好用"，而在价差、残差诊断上很有用

核心概念：
    - ACF：ρ_k = Cov(y_t, y_{t-k}) / Var(y_t)
    - PACF：剔除 1..k-1 期影响后，y_t 与 y_{t-k} 的直接相关
    - AR(p)：y_t = c + Σφ_i·y_{t-i} + ε_t；MA(q)：y_t = c + ε_t + Σθ_j·ε_{t-j}
    - Ljung-Box：H0 = 前 m 阶自相关全为 0（白噪声）

前置知识：
    - 09-stationarity-adf.py（ARMA 只适用于平稳序列）

环境要求：
    - pip install numpy pandas statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据是 i.i.d. 正态收益，收益平方也不会有自相关）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

from utils.data import get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

START, END = "20150101", "20241231"


def fit_arima(y: np.ndarray, order: tuple[int, int, int]):
    """ARIMA 拟合（传 numpy 数组避免日期频率警告；部分阶数可能不收敛，静默该警告并在调用处看 AIC）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")
        return ARIMA(y, order=order).fit()


# ==================== 第一部分：ACF 与 PACF ====================


def acf_pacf_patterns():
    print("=" * 60)
    print("第一部分：ACF 与 PACF —— 用模拟看清模型的'指纹'")
    print("=" * 60)

    print("""
    ACF：序列与自身滞后 k 期的相关系数
       ρ_k > 0：今天涨，k 天后倾向于涨（动量）；ρ_k < 0：倾向于跌（反转）
    PACF：剔除中间滞后期影响后的"直接"相关
       👨‍👦‍👴 比喻：ACF 问"你和爷爷像吗"——像，但可能只是因为你像爸爸、爸爸像爷爷；
                  PACF 问"排除爸爸的影响后，你和爷爷还像吗"

    ┌────────────┬──────────────────┬──────────────────┐
    │ 模型       │ ACF              │ PACF             │
    ├────────────┼──────────────────┼──────────────────┤
    │ AR(p)      │ 逐渐衰减（拖尾） │ p 阶后截断       │
    │ MA(q)      │ q 阶后截断       │ 逐渐衰减（拖尾） │
    │ ARMA(p,q)  │ 拖尾             │ 拖尾             │
    │ 白噪声     │ 全部不显著       │ 全部不显著       │
    └────────────┴──────────────────┴──────────────────┘
    ⚠️ 这里的 MA（移动平均模型）与技术分析中的"均线 MA"不是一回事
    """)

    n = 2000
    # ArmaProcess 的约定：ar 参数写成滞后多项式 [1, -φ1, -φ2, ...]，ma 写成 [1, θ1, ...]
    processes = {
        "AR(1) φ=0.7": ArmaProcess(ar=[1, -0.7], ma=[1]),
        "AR(2) φ=(0.5,0.3)": ArmaProcess(ar=[1, -0.5, -0.3], ma=[1]),
        "MA(1) θ=0.7": ArmaProcess(ar=[1], ma=[1, 0.7]),
        "白噪声": ArmaProcess(ar=[1], ma=[1]),
    }
    band = 1.96 / np.sqrt(n)
    sims = {}
    rows = []
    for i, (name, proc) in enumerate(processes.items()):
        y = proc.generate_sample(nsample=n, distrvs=np.random.default_rng(i).standard_normal, burnin=200)
        sims[name] = y
        a = acf(y, nlags=4)[1:]
        p = pacf(y, nlags=4)[1:]
        rows.append([name, "ACF"] + [f"{v:+.2f}{'*' if abs(v) > band else ' '}" for v in a])
        rows.append(["", "PACF"] + [f"{v:+.2f}{'*' if abs(v) > band else ' '}" for v in p])
    print(f"🧪 每个过程模拟 {n} 个点，* 表示超出 ±1.96/√n = ±{band:.3f} 的显著区间：")
    print(pd.DataFrame(rows, columns=["过程", "", "lag1", "lag2", "lag3", "lag4"]).to_string(index=False))
    print("""
    💡 对照上表：AR(1) 的 PACF 在 1 阶后截断，AR(2) 在 2 阶后截断；MA(1) 的 ACF 在 1 阶后截断。
    """)
    return sims


# ==================== 第二部分：真实收益的自相关 ====================


def real_autocorrelation():
    print("\n" + "=" * 60)
    print("第二部分：真实收益的自相关 —— 收益难预测，波动可预测")
    print("=" * 60)

    print("""
    Ljung-Box：Q = n(n+2)·Σ ρ̂_k² / (n-k)，H0：前 m 阶自相关全为 0
       p < 0.05 → 拒绝"无自相关"
    """)
    series = {
        "沪深300": np.log(get_index_daily("000300", START, END)["close"]).diff().dropna(),
        "贵州茅台": np.log(get_stock_daily_adj("600519", START, END)["adj_close"]).diff().dropna(),
    }
    rows = []
    for name, r in series.items():
        a = acf(r, nlags=5)[1:]
        for label, x in [("r", r), ("r²", r**2), ("|r|", r.abs())]:
            lb = acorr_ljungbox(x, lags=[5, 20])
            rows.append([name, label, f"{acf(x, nlags=1)[1]:+.3f}", f"{lb['lb_pvalue'].iloc[0]:.2e}",
                         f"{lb['lb_pvalue'].iloc[1]:.2e}"])
        print(f"📊 {name} 日收益 ACF(1..5)：" + "  ".join(f"{v:+.3f}" for v in a) + f"   （显著区间 ±{1.96 / np.sqrt(len(r)):.3f}）")
    print()
    print(pd.DataFrame(rows, columns=["标的", "序列", "lag1 自相关", "Ljung-Box p (m=5)", "Ljung-Box p (m=20)"]).to_string(index=False))
    print("""
    💡 典型发现：
       - 收益本身的自相关很弱：如果存在明显的线性自相关，早就被套利者交易掉了
       - 收益平方 / 绝对值的自相关通常强得多 —— "方向难测，幅度可测"，这就是波动率聚集（11 课）
       - Ljung-Box 对收益本身有时也会拒绝 H0：一方面大样本下很灵敏，另一方面波动聚集（异方差）
         会让标准 Ljung-Box 过度拒绝。即使"显著"，也要看自相关系数本身是否大到有经济意义
    """)
    if is_synthetic(series["沪深300"].index):
        print("    ⚠️ 离线合成数据是 i.i.d. 正态收益：r² 和 |r| 也不会有显著自相关，请以 11 课的 GARCH 模拟为准")
    return series["沪深300"]


# ==================== 第三部分：ARMA 拟合与选阶 ====================


def fit_and_select(sims: dict):
    print("\n" + "=" * 60)
    print("第三部分：ARMA 拟合与 AIC/BIC 选阶")
    print("=" * 60)

    print("""
    ARMA(p, q)：y_t = c + Σφ_i·y_{t-i} + ε_t + Σθ_j·ε_{t-j}
    ARIMA(p, d, q)：先差分 d 次再拟合 ARMA（对收益序列 d = 0）
    选阶：AIC / BIC 在"拟合优度"和"参数个数"之间折中，越小越好；BIC 惩罚更重，倾向于选更简单的模型
    """)

    true = ArmaProcess(ar=[1, -0.6], ma=[1, 0.3])
    y = true.generate_sample(nsample=2000, distrvs=np.random.default_rng(10).standard_normal, burnin=200)
    res = fit_arima(y, (1, 0, 1))
    print("🧪 模拟 ARMA(1,1)：真实 φ = 0.6，θ = 0.3")
    print(f"   估计：φ̂ = {res.params[1]:.3f}，θ̂ = {res.params[2]:.3f}（params 顺序：const, ar.L1, ma.L1, sigma2）")

    rows = []
    for p in range(3):
        for q in range(3):
            r = fit_arima(y, (p, 0, q))
            rows.append([f"ARMA({p},{q})", r.aic, r.bic])
    grid = pd.DataFrame(rows, columns=["模型", "AIC", "BIC"])
    print("\n   9 个候选模型的信息准则：")
    print("   " + grid.round(1).to_string(index=False).replace("\n", "\n   "))
    print(f"   AIC 选中：{grid.loc[grid['AIC'].idxmin(), '模型']}；BIC 选中：{grid.loc[grid['BIC'].idxmin(), '模型']}")

    resid = res.resid
    lb = acorr_ljungbox(resid, lags=[10], model_df=2)
    print(f"\n   残差诊断：Ljung-Box(10) p = {lb['lb_pvalue'].iloc[0]:.3f}（model_df=2 扣除已估计的 ARMA 参数个数）")
    print("   → p 大说明残差里没有剩余的线性结构，模型把自相关'吃干净'了")


# ==================== 第四部分：ARMA 预测真实收益的样本外表现 ====================


def out_of_sample(r: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：ARMA 预测真实收益 —— 样本外检验")
    print("=" * 60)

    y = r.to_numpy() * 100  # 百分比单位，数值更稳定
    split = int(len(y) * 0.7)
    train, test = y[:split], y[split:]
    rows = []
    for p in range(3):
        for q in range(3):
            res = fit_arima(train, (p, 0, q))
            rows.append([(p, q), res.aic, res.bic])
    grid = pd.DataFrame(rows, columns=["order", "AIC", "BIC"])
    best = grid.loc[grid["AIC"].idxmin(), "order"]
    res = fit_arima(train, (best[0], 0, best[1]))
    print(f"📐 沪深300 日收益，前 70%（{split} 天）训练，AIC 选中 ARMA{best}")
    print("   系数：" + "，".join(f"{k} = {v:+.4f}" for k, v in zip(res.model.param_names, res.params)))
    names = list(res.model.param_names)
    if "ar.L1" in names and "ma.L1" in names:
        phi, theta = res.params[names.index("ar.L1")], res.params[names.index("ma.L1")]
        print(f"   φ₁ + θ₁ = {phi + theta:+.4f}：若接近 0，AR 与 MA 部分几乎相互抵消，模型近似退化为白噪声，")
        print('   这往往是在噪声上"过拟合"的信号（AIC 选中它，并不代表它有预测能力）')

    # 用训练好的参数（不再重新估计）在测试集上做逐日一步预测
    full = res.apply(y)
    pred = full.fittedvalues[split:]
    mse_model = np.mean((test - pred) ** 2)
    mse_mean = np.mean((test - train.mean()) ** 2)
    oos_r2 = 1 - mse_model / mse_mean
    hit = np.mean(np.sign(pred) == np.sign(test))
    print(f"\n   样本外 {len(test)} 天：")
    print(f"   预测值的标准差 = {pred.std():.4f}%，实际收益的标准差 = {test.std():.4f}%")
    print(f"   样本外 R²（相对'用训练集均值预测'）= {oos_r2:+.4%}")
    print(f"   方向准确率 = {hit:.1%}")
    strat = np.sign(pred) * test
    print(f"   按预测方向做多/做空（未计成本）的日均收益 = {strat.mean():+.4f}%，t = {strat.mean() / strat.std() * np.sqrt(len(strat)):.2f}")
    print("""
    💡 预测值的波动远小于实际收益 —— ARMA 对收益的预测基本"贴着 0"，样本外 R² 接近 0 甚至为负，扣除成本后难以盈利。
       这不代表 ARMA 没用：
       - 价差序列（协整对）常有显著的 AR(1) 结构 —— 07 课的半衰期就是 AR(1) 的应用
       - 部分个股、小市值股票、短周期可能存在短期反转（Phase 7 的反转因子），需要严格样本外检验
       - 残差诊断：任何模型拟合后，都用 Ljung-Box 检查残差里还有没有剩余结构
    """)


def plot_acf_figures(sims: dict, r: pd.Series):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    plot_acf(sims["AR(1) φ=0.7"], lags=15, ax=axes[0, 0], title="AR(1) 的 ACF：拖尾")
    plot_pacf(sims["AR(1) φ=0.7"], lags=15, ax=axes[1, 0], title="AR(1) 的 PACF：1 阶截尾", method="ywm")
    plot_acf(sims["MA(1) θ=0.7"], lags=15, ax=axes[0, 1], title="MA(1) 的 ACF：1 阶截尾")
    plot_pacf(sims["MA(1) θ=0.7"], lags=15, ax=axes[1, 1], title="MA(1) 的 PACF：拖尾", method="ywm")
    plot_acf(r, lags=20, ax=axes[0, 2], title="沪深300 收益 r 的 ACF")
    plot_acf(r**2, lags=20, ax=axes[1, 2], title="沪深300 收益平方 r² 的 ACF")
    for ax in axes.flat:
        ax.set_ylim(-0.3, 1.05)
    fig.tight_layout()
    path = save_figure(fig, "10_acf_pacf.png", subdir="phase-3")
    print(f"🖼️  图已保存：{path}（蓝色阴影带外的柱子才算显著）")


# ==================== 主函数 ====================


def main():
    sims = acf_pacf_patterns()
    r = real_autocorrelation()
    fit_and_select(sims)
    out_of_sample(r)
    plot_acf_figures(sims, r)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 对 07 课工行-建行的价差序列拟合 AR(1)，用 φ 计算半衰期 -ln2/ln(φ)，与 07 课的回归法比较
    2. 把沪深300 换成中证1000（000852）或一只小盘股，日收益的 lag1 自相关是正还是负？
    3. 第四部分改为"滚动重新估计"（每 60 天用最近 500 天重新拟合），样本外 R² 会改善吗？
    4. 对周收益重复第二部分：r 和 r² 的自相关结论是否改变？
    """)


if __name__ == "__main__":
    main()
