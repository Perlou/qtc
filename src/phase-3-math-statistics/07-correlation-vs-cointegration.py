"""
相关性与协整
============

学习目标：
    1. 掌握 Pearson 与 Spearman 相关，理解相关性随时间变化、危机时趋于 1
    2. 用"醉汉和狗"理解协整：长期被绳子拴在一起
    3. 用两个模拟反例记住：相关 ≠ 协整，协整 ≠ 高相关
    4. 用 Engle-Granger 两步法（statsmodels.coint）检验真实股票对
    5. 计算价差、z-score 和均值回归半衰期；认识配对筛选中的多重检验

核心概念：
    - Pearson ρ = Cov(X,Y)/(σ_X σ_Y)：线性相关；Spearman：秩相关，对异常值稳健
    - 协整：X、Y 都是 I(1)，但 s_t = Y_t - βX_t 平稳
    - coint 检验 H0：不协整；p < 0.05 → 拒绝"不协整"
    - 半衰期 ≈ -ln2 / λ，λ 来自 Δs_t = λ·s_{t-1} + c + ε_t

前置知识：
    - 06-linear-regression-ols.py（伪回归）；09 课会系统讲平稳性与 ADF

环境要求：
    - pip install numpy pandas statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据中股票彼此独立，真实股票对部分不会出现协整）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller, coint

from utils.data import get_close_panel, get_index_daily, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
NAMES = {
    "600519": "贵州茅台", "000858": "五粮液", "000568": "泸州老窖",
    "601398": "工商银行", "601288": "农业银行", "601988": "中国银行", "601939": "建设银行",
}


def half_life(spread: pd.Series) -> float:
    """Δs_t = λ·s_{t-1} + c + ε_t，半衰期 = -ln2 / λ（λ ≥ 0 时返回 inf，表示没有均值回归）"""
    ds = spread.diff().dropna()
    lag = spread.shift(1).loc[ds.index]
    lam = sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1]
    return -np.log(2) / lam if lam < 0 else np.inf


# ==================== 第一部分：相关性 ====================


def correlation_basics(panel: pd.DataFrame):
    print("=" * 60)
    print("第一部分：相关性 —— 同涨同跌的程度")
    print("=" * 60)

    print("""
    Pearson：ρ = Cov(X,Y) / (σ_X·σ_Y) ∈ [-1, 1]，只衡量线性关系
    Spearman：先把数据转为排名再算 Pearson，对异常值稳健，能捕捉单调非线性关系
              Phase 7 因子研究的 Rank IC 就是 Spearman 相关
    """)

    rets = panel.pct_change().dropna()
    a, b = rets["600519"], rets["000858"]
    print(f"📊 茅台 vs 五粮液 日收益：Pearson = {a.corr(b):.3f}，Spearman = {a.corr(b, method='spearman'):.3f}")

    # 异常值实验：一个极端点就能大幅改变 Pearson
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, 250)
    y = rng.normal(0, 0.02, 250)
    x_out, y_out = x.copy(), y.copy()
    x_out[0], y_out[0] = 0.25, 0.25  # 一个"同时暴涨 25%"的异常点（例如数据错误或复权错误）
    sx, sy, sxo, syo = map(pd.Series, (x, y, x_out, y_out))
    print("\n🧪 两组独立随机收益（250 天），在其中加入 1 个异常点：")
    print(f"   无异常点：Pearson = {sx.corr(sy):+.3f}，Spearman = {sx.corr(sy, method='spearman'):+.3f}")
    print(f"   有异常点：Pearson = {sxo.corr(syo):+.3f}，Spearman = {sxo.corr(syo, method='spearman'):+.3f}")

    # 相关性随时间与市场状态变化
    roll = a.rolling(60).corr(b).dropna()
    print(f"\n📈 60 日滚动相关：最低 {roll.min():.2f}，最高 {roll.max():.2f}，中位数 {roll.median():.2f}")

    hs300 = get_index_daily("000300", START, END)["close"].pct_change().reindex(rets.index)
    vol20 = hs300.rolling(20).std()
    high = vol20 > vol20.quantile(0.8)
    low = vol20 < vol20.quantile(0.2)
    stocks = [c for c in rets.columns if c != "600519"]
    avg_corr = lambda mask: rets.loc[mask, stocks].corr().where(~np.eye(len(stocks), dtype=bool)).stack().mean()
    print(f"🌪️ {len(stocks)} 只股票两两之间的平均相关系数：")
    print(f"   大盘高波动期（20 日波动前 20%）：{avg_corr(high):.3f}")
    print(f"   大盘低波动期（20 日波动后 20%）：{avg_corr(low):.3f}")
    print("""
    💡 真实市场中，危机/高波动期相关性往往上升 —— 平时分散化良好的组合可能同时下跌。
       用全样本相关系数评估分散化效果会过于乐观。
    """)
    if is_offline():
        print("    ⚠️ 离线合成数据中各股票彼此独立，相关系数都接近 0，看不到上述现象")
    return roll


# ==================== 第二部分：协整的直觉与两个反例 ====================


def drunk_and_dog():
    print("\n" + "=" * 60)
    print("第二部分：协整 —— 醉汉和他的狗")
    print("=" * 60)

    print("""
    🍺🐕 醉汉在街上随机乱走，狗也随机乱跑：
       相关性：每一步醉汉往左时，狗是否也往左（短期同步）
       协整：  狗被绳子拴着，不管两者怎么走，距离不会无限扩大（长期均衡）

    ┌────────────┬──────────────────┬──────────────────────────┐
    │            │ 相关             │ 协整                     │
    ├────────────┼──────────────────┼──────────────────────────┤
    │ 作用对象   │ 收益率（变化）   │ 价格水平                 │
    │ 时间尺度   │ 短期同步         │ 长期均衡                 │
    │ 交易含义   │ 分散化、对冲     │ 配对交易（价差会回归）   │
    └────────────┴──────────────────┴──────────────────────────┘
    """)

    rng = np.random.default_rng(42)
    n = 1000
    # 反例 1：收益高度相关，但 B 每天系统性多涨一点 → 价差持续扩大
    common = rng.normal(0, 0.015, n)
    ra = common + rng.normal(0, 0.004, n)
    rb = common + rng.normal(0.0015, 0.004, n)
    la, lb = np.cumsum(ra), np.cumsum(rb)
    # 反例 2：价差是均值回归的 AR(1)，但每日收益相关性不高
    x = np.cumsum(rng.normal(0, 0.01, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.03)
    y = x + s

    rows = []
    for name, p1, p2 in [("反例1：相关但不协整", la, lb), ("反例2：协整但相关性不高", x, y)]:
        corr = np.corrcoef(np.diff(p1), np.diff(p2))[0, 1]
        _, p_coint, _ = coint(p2, p1)
        spread = p2 - p1
        rows.append([name, f"{corr:.3f}", f"{p_coint:.4f}", f"{spread[:n // 2].std():.3f}", f"{spread[n // 2:].std():.3f}",
                     f"{spread[-1] - spread[0]:+.3f}"])
    print(pd.DataFrame(rows, columns=["情形", "收益相关", "coint p 值", "价差前半段标准差", "价差后半段标准差", "价差首尾变化"]).to_string(index=False))
    print("""
    💡 解读（coint 的 H0 是"不协整"）：
       - 反例 1：收益相关很高，但价差一路漂走 → 做价差回归会亏钱
       - 反例 2：每天的涨跌关系并不紧密，但价差始终在一个范围内来回 → 这才是配对交易需要的
    """)
    return (la, lb), (x, y)


# ==================== 第三部分：Engle-Granger 检验真实股票对 ====================


def test_real_pairs(panel: pd.DataFrame) -> tuple[str, str]:
    print("\n" + "=" * 60)
    print("第三部分：Engle-Granger 两步法检验真实股票对")
    print("=" * 60)

    print("""
    步骤 1：OLS 回归 ln(Y) = α + β·ln(X) + ε，估计对冲比例 β
    步骤 2：对残差 ε̂ 做单位根检验 —— 但临界值与普通 ADF 不同（因为 β 是估计出来的）
    ✅ 直接用 statsmodels.tsa.stattools.coint，它封装了两步法并使用正确的临界值
    """)

    logp = np.log(panel)
    pairs = [("000858", "000568"), ("000858", "600519"), ("601398", "601939"), ("601398", "601288"),
             ("601988", "601288"), ("601939", "601988")]
    rows = []
    for y_sym, x_sym in pairs:
        y, x = logp[y_sym], logp[x_sym]
        t_stat, p_coint, _ = coint(y, x)
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        resid = ols.resid
        p_naive = adfuller(resid, result_object=True).pvalue
        rows.append([f"{NAMES[y_sym]} ~ {NAMES[x_sym]}", f"{y.diff().corr(x.diff()):.3f}", f"{ols.params.iloc[1]:.3f}",
                     f"{p_coint:.4f}", f"{p_naive:.4f}", f"{half_life(resid):.0f}"])
    table = pd.DataFrame(rows, columns=["股票对 (Y ~ X)", "收益相关", "β", "coint p 值", "❌ 残差直接 ADF 的 p", "半衰期(天)"])
    print(f"📊 {START[:4]}-{END[:4]} 对数价格：")
    print(table.to_string(index=False))
    best = table.iloc[table["coint p 值"].astype(float).argmin()]
    print(f"""
    💡 解读：
       - "残差直接 ADF 的 p"系统性偏小 → 会把更多股票对误判为协整，所以不要这么做
       - 本样本中 coint p 值最小的是：{best['股票对 (Y ~ X)']}（p = {best['coint p 值']}）
       - 收益相关性高的股票对，不一定协整；要以 coint 结果为准
       - 注意 coint 对"谁做 Y、谁做 X"并不完全对称，交换顺序结果可能不同
       - ⚠️ 从 {len(pairs)} 对里挑 p 值最小的一对，本身就是多重检验（见第五部分），需要样本外验证
    """)
    i = int(table["coint p 值"].astype(float).argmin())
    return pairs[i]


# ==================== 第四部分：价差、z-score 与半衰期 ====================


def spread_and_half_life(panel: pd.DataFrame, pair: tuple[str, str], sim_pair):
    print("\n" + "=" * 60)
    print("第四部分：价差、z-score 与半衰期")
    print("=" * 60)

    print("""
    价差 s_t = ln(Y_t) - β·ln(X_t)
    z-score = (s_t - 滚动均值) / 滚动标准差 → 偏离多少个标准差（Phase 6 的开平仓信号）
    半衰期：对 Δs_t = λ·s_{t-1} + c + ε_t 回归，半衰期 ≈ -ln2/λ（λ < 0 才有均值回归）
    """)

    x_sim, y_sim = sim_pair
    s_sim = pd.Series(y_sim - x_sim)
    print(f"🧪 模拟协整对（价差为 AR(1)，φ = 0.9）：估计半衰期 = {half_life(s_sim):.1f} 天，"
          f"理论值 = -ln2 / ln(0.9) = {-np.log(2) / np.log(0.9):.1f} 天")

    y_sym, x_sym = pair
    logp = np.log(panel[[y_sym, x_sym]])
    beta_full = sm.OLS(logp[y_sym], sm.add_constant(logp[x_sym])).fit().params.iloc[1]
    spread_full = logp[y_sym] - beta_full * logp[x_sym]

    # ✅ 扩展窗口估计 β：只用截至当天的数据（至少 250 天，每 20 天更新一次）
    beta_exp = pd.Series(np.nan, index=logp.index)
    for i in range(250, len(logp), 20):
        hist = logp.iloc[: i + 1]
        beta_exp.iloc[i] = sm.OLS(hist[y_sym], sm.add_constant(hist[x_sym])).fit().params.iloc[1]
    beta_exp = beta_exp.ffill()
    spread = (logp[y_sym] - beta_exp * logp[x_sym]).dropna()
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

    print(f"\n📊 真实股票对 {NAMES[y_sym]} ~ {NAMES[x_sym]}：")
    print(f"   ❌ 全样本 β = {beta_full:.3f}，价差半衰期 = {half_life(spread_full):.0f} 天（用到了未来数据）")
    print(f"   ✅ 扩展窗口 β 范围 [{beta_exp.min():.3f}, {beta_exp.max():.3f}]，价差半衰期 = {half_life(spread):.0f} 天")
    print(f"   扩展窗口价差的 60 日 z-score：|z| > 2 的天数占比 = {(z.abs() > 2).mean():.1%}")
    print("""
    💡 全样本 β 是在同一段数据上"挑出来让残差最平稳"的，所以样本内价差总是看起来最干净。
       换成只用历史数据估计的 β 后，均值回归往往变弱：β 的每次更新还会乘以对数价格水平，
       让价差出现跳变。两种半衰期的差距，就是"样本内看起来能做"和"实盘真能做"之间的差距。
    """)
    print("""
    💡 半衰期太长（几个月）→ 资金占用久、风险暴露长；太短（< 1 天）→ 可能是噪声或难以交易。
       协整关系会断裂（并购、主营变化、政策冲击），Phase 6 会讨论如何监控和止损。
    """)
    return spread, z, (y_sym, x_sym)


# ==================== 第五部分：配对筛选中的多重检验 ====================


def pair_screening_trap():
    print("\n" + "=" * 60)
    print("第五部分：配对筛选 = 多重检验")
    print("=" * 60)

    rng = np.random.default_rng(7)
    n_assets, n_days = 30, 500
    prices = np.cumsum(rng.normal(0, 0.02, size=(n_days, n_assets)), axis=0)
    p_values = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            p_values.append(coint(prices[:, i], prices[:, j], maxlag=1, autolag=None)[1])
    p_values = np.array(p_values)
    print(f"🎲 {n_assets} 条相互独立的随机游走（彼此绝对不协整），两两检验共 {len(p_values)} 对：")
    print(f"   coint p < 0.05 的'协整对'：{(p_values < 0.05).sum()} 对（{(p_values < 0.05).mean():.1%}）")
    print(f"   Bonferroni 门槛 p < {0.05 / len(p_values):.1e}：{(p_values < 0.05 / len(p_values)).sum()} 对")
    print("""
    💡 3000 只股票约有 450 万个配对，5% 显著性下会"找到"大量假协整对。
       对策：先用经济逻辑缩小候选范围（同行业、同产业链），再做样本外验证。
    """)


def plot_pairs(roll: pd.Series, sim1, sim2, spread: pd.Series, z: pd.Series, pair):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(sim1[0], label="A")
    ax.plot(sim1[1], label="B")
    ax.set_title("反例 1：收益高度相关，但不协整（价差持续扩大）")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(sim2[0], label="X（醉汉）")
    ax.plot(sim2[1], label="Y（狗）")
    ax.set_title("反例 2：协整（被绳子拴住）")
    ax.legend()
    ax = axes[1, 0]
    roll.plot(ax=ax)
    ax.set_title("茅台 vs 五粮液 60 日滚动相关")
    ax = axes[1, 1]
    z.plot(ax=ax, lw=0.8)
    for lvl in (-2, 2):
        ax.axhline(lvl, color="C3", ls="--")
    ax.set_title(f"{NAMES[pair[0]]} ~ {NAMES[pair[1]]} 价差 z-score")
    fig.tight_layout()
    path = save_figure(fig, "07_correlation_cointegration.png", subdir="phase-3")
    print(f"\n🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    panel = get_close_panel(list(NAMES), START, END).dropna()
    print(f"📊 数据：{len(NAMES)} 只股票，{panel.index[0].date()} ~ {panel.index[-1].date()}，{len(panel)} 个交易日\n")
    roll = correlation_basics(panel)
    sim1, sim2 = drunk_and_dog()
    pair = test_real_pairs(panel)
    spread, z, pair = spread_and_half_life(panel, pair, sim2)
    pair_screening_trap()
    plot_pairs(roll, sim1, sim2, spread, z, pair)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把样本分成 2018-2020 和 2021-2024 两段，分别做第三部分的协整检验。结论稳定吗？
    2. 交换 coint(y, x) 中 y 和 x 的顺序，p 值和 β 如何变化？为什么不对称？
    3. 修改反例 2 中 AR(1) 的系数 φ（0.5、0.95、0.99），观察 coint p 值与半衰期的变化
    4. 从同一行业（如券商）选 10 只股票做两两协整筛选，p < 0.05 的有几对？其中多少可能是假阳性？
    """)


if __name__ == "__main__":
    main()
