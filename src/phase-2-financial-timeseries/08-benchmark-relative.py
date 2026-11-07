"""
基准、超额收益与 Beta
=====================

学习目标：
    1. 理解为什么要和基准比较，以及如何选择基准
    2. 计算超额收益、跟踪误差（TE）、信息比率（IR）
    3. 区分累计超额收益的两种算法：日度超额连乘 vs 净值相除
    4. 用协方差公式和 OLS 回归两种方法估计 Beta / Alpha，并理解 R²
    5. 计算滚动 Beta，理解 Beta 的时变性
    6. 认识陷阱：高 Beta 冒充超额、价格指数 vs 全收益指数、停牌错配

核心概念：
    - 超额收益：R_e = R_p - R_b
    - 跟踪误差：TE = std(R_e) × √N；信息比率：IR = mean(R_e) × N / TE
    - CAPM 回归：R_i - R_f = α + β(R_m - R_f) + ε
    - Beta：β = Cov(R_i, R_m) / Var(R_m)

前置知识：
    - 01-returns-simple-vs-log.py、02-annualization-volatility.py、03-rolling-windows.py

环境要求：
    - pip install pandas numpy matplotlib statsmodels akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据；合成资产相互独立，Beta 接近 0）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

from utils.data import get_index_daily, get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
TRADING_DAYS = 252
START, END = "20180101", "20241231"
BENCH = "000300"
STOCKS = {"600519": "贵州茅台", "601088": "中国神华", "600036": "招商银行", "000333": "美的集团"}


def load_prices() -> pd.DataFrame:
    """个股（复权）与基准的收盘价宽表；以基准的交易日为主日历，个股 NaN 保留（停牌）"""
    data = {code: get_stock_daily_adj(code, START, END)["adj_close"] for code in STOCKS}
    bench = get_index_daily(BENCH, START, END)["close"]
    return pd.DataFrame(data).reindex(bench.index).assign(**{BENCH: bench})


# ==================== 第一部分：为什么需要基准 ====================


def why_benchmark(rets: pd.DataFrame):
    print("=" * 60)
    print("第一部分：为什么需要基准")
    print("=" * 60)

    print("""
    🚣 比喻：船顺流而下。船相对岸的速度 = 水流速度（市场，Beta）+ 船自己划的速度（Alpha）。
       顺流时人人都快，量化研究要分清"是我划得好，还是水流得急"。

    ┌───────────────────┬──────────────────────────────┐
    │ 策略类型          │ 常见基准                     │
    ├───────────────────┼──────────────────────────────┤
    │ A 股大盘股        │ 沪深 300（000300）           │
    │ A 股中盘 / 小盘   │ 中证 500 / 中证 1000         │
    │ 全市场            │ 中证全指、万得全 A 等        │
    │ 期货 CTA          │ 绝对收益或无风险利率         │
    │ 加密              │ BTC                          │
    └───────────────────┴──────────────────────────────┘

    ⚠️ 常见的沪深 300 是"价格指数"，不含分红再投资。个股用复权价（含分红）、
       基准用价格指数，会系统性高估超额收益。更公平的做法是用全收益指数，或在报告中注明。
    """)

    yearly = (1 + rets.fillna(0)).groupby(rets.index.year).prod() - 1
    yearly = yearly.rename(columns={**STOCKS, BENCH: "沪深300"})
    print("📅 各年收益：")
    print(yearly.to_string(float_format=lambda v: f"{v:+.1%}"))
    excess = yearly.drop(columns="沪深300").sub(yearly["沪深300"], axis=0)
    print("\n📅 各年相对沪深 300 的超额（简单相减）：")
    print(excess.to_string(float_format=lambda v: f"{v:+.1%}"))


# ==================== 第二部分：超额、TE 与 IR ====================


def te_ir(rets: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：超额收益、跟踪误差与信息比率")
    print("=" * 60)

    print("""
    超额收益  R_e = R_p - R_b
    跟踪误差  TE  = std(R_e) × √N            超额收益的年化波动
    信息比率  IR  = mean(R_e) × N / TE       单位跟踪误差的超额收益
    """)

    rows = []
    for code, name in STOCKS.items():
        pair = rets[[code, BENCH]].dropna()
        ex = pair[code] - pair[BENCH]
        ann_ex = ex.mean() * TRADING_DAYS
        te = ex.std() * np.sqrt(TRADING_DAYS)
        rows.append([f"{code} {name}", len(pair), ann_ex, te, ann_ex / te, (ex > 0).mean()])
    df = pd.DataFrame(rows, columns=["股票", "样本天数", "年化超额", "跟踪误差", "IR", "跑赢基准天数占比"])
    print(df.to_string(index=False, formatters={
        "年化超额": "{:+.2%}".format, "跟踪误差": "{:.2%}".format, "IR": "{:.2f}".format,
        "跑赢基准天数占比": "{:.1%}".format,
    }))
    print("""
    💡 单只股票相对指数的 TE 通常很大；真实的指数增强产品 TE 一般要小得多。
       这里的"年化超额"是算术口径（日均 × N），与 02 课的 CAGR 不同。
    """)


# ==================== 第三部分：累计超额的两种算法 ====================


def cumulative_excess(rets: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：累计超额收益的两种算法")
    print("=" * 60)

    code = "600519"
    pair = rets[[code, BENCH]].dropna()
    r, b = pair[code], pair[BENCH]

    nav1 = (1 + (r - b)).cumprod()  # 方法 1：每日超额连乘（"日度再平衡"的多空组合）
    nav2 = (1 + r).cumprod() / (1 + b).cumprod()  # 方法 2：净值相除（买入持有的相对比值）
    simple = (1 + r).prod() - (1 + b).prod()  # 方法 3：总收益直接相减
    print(f"""
    {STOCKS[code]} vs 沪深 300（{pair.index[0].date()} ~ {pair.index[-1].date()}）：
       个股总收益 {(1 + r).prod() - 1:+.2%}，基准总收益 {(1 + b).prod() - 1:+.2%}

       方法 1 每日超额连乘  ∏(1 + R - B)      期末 {nav1.iloc[-1]:.4f}
       方法 2 净值相除      ∏(1+R) / ∏(1+B)   期末 {nav2.iloc[-1]:.4f}
       方法 3 总收益相减    (总收益之差)       {simple:+.2%}

    💡 本例方法 1、2 期末相差 {nav1.iloc[-1] / nav2.iloc[-1] - 1:+.2%}。差距取决于具体路径：
       短期内两者接近，样本越长、超额波动越大，就越可能拉开。
       没有绝对的对错，但报告中必须说明用的是哪一种。
    """)
    return nav1, nav2


# ==================== 第四部分：Beta 与 Alpha ====================


def beta_alpha(rets: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：Beta 与 Alpha")
    print("=" * 60)

    print("""
    CAPM 回归：R_i - R_f = α + β(R_m - R_f) + ε
       β：对市场的敏感度。β = 1.2 → 市场涨 1%，该股平均涨 1.2%
       α：与市场无关的平均收益（真正的"本事"）
       ε：个股特有的波动（特质风险）
    直接公式：β = Cov(R_i, R_m) / Var(R_m)

    日频下 R_f 影响很小，这里忽略（年化 Sharpe 时才需要扣除，取值以实际数据为准）。
    """)

    rows = []
    for code, name in STOCKS.items():
        d = rets[[code, BENCH]].dropna()
        d.columns = ["stock", "mkt"]
        beta_cov = d["stock"].cov(d["mkt"]) / d["mkt"].var()
        model = sm.OLS(d["stock"], sm.add_constant(d["mkt"])).fit()
        alpha_d, beta_ols = model.params["const"], model.params["mkt"]
        rows.append([f"{code} {name}", beta_cov, beta_ols, alpha_d * TRADING_DAYS,
                     model.tvalues["const"], model.rsquared, model.resid.std() * np.sqrt(TRADING_DAYS)])
    df = pd.DataFrame(rows, columns=["股票", "β(协方差)", "β(OLS)", "年化α", "α的t值", "R²", "特质波动(年化)"])
    print(df.to_string(index=False, formatters={
        "β(协方差)": "{:.3f}".format, "β(OLS)": "{:.3f}".format, "年化α": "{:+.2%}".format,
        "α的t值": "{:.2f}".format, "R²": "{:.2f}".format, "特质波动(年化)": "{:.2%}".format,
    }))
    print("""
    💡 读表：
       - 两种方法的 β 完全一致：一元 OLS 的斜率就是 Cov/Var
       - R² = 市场能解释的方差比例，其余是个股特有波动
       - α 的 t 值绝对值小于 2 时，这个 α 在统计上与 0 无法区分（Phase 3 详述）
    """)


# ==================== 第五部分：滚动 Beta ====================


def rolling_beta(rets: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第五部分：滚动 Beta")
    print("=" * 60)

    out = {}
    for code in ["600519", "601088"]:
        d = rets[[code, BENCH]].dropna()
        out[code] = d[code].rolling(120).cov(d[BENCH]) / d[BENCH].rolling(120).var()
    rb = pd.DataFrame(out)

    def full_beta(code: str) -> float:
        d = rets[[code, BENCH]].dropna()
        return d[code].cov(d[BENCH]) / d[BENCH].var()

    summary = pd.DataFrame({
        "全样本β": [full_beta(c) for c in rb.columns],
        "滚动β最小": rb.min().to_numpy(),
        "滚动β最大": rb.max().to_numpy(),
        "最新": rb.ffill().iloc[-1].to_numpy(),
    }, index=[STOCKS[c] for c in rb.columns])
    print("📊 120 日滚动 Beta：")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
    print("""
    💡 滚动 Beta 的变化既有真实时变，也有 120 日样本的估计噪声：
       - 用全样本 Beta 做历史对冲是前视偏差（你在 2019 年不知道 2024 年的数据）
       - 实盘对冲要用"截至当时"的滚动估计，并接受估计误差
    """)
    return rb


# ==================== 第六部分：陷阱实验 ====================


def pitfalls(prices: pd.DataFrame, rets: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：陷阱实验")
    print("=" * 60)

    # 陷阱 1：高 Beta 冒充超额（模拟：α 恒为 0，β = 1.5，牛市样本）
    rng = np.random.default_rng(11)
    n = TRADING_DAYS
    mkt = rng.normal(0.30 / n, 0.20 / np.sqrt(n), n)  # 牛市：年化 +30%
    stock = 1.5 * mkt + rng.normal(0, 0.15 / np.sqrt(n), n)  # 真实 α = 0
    d = pd.DataFrame({"stock": stock, "mkt": mkt})
    model = sm.OLS(d["stock"], sm.add_constant(d["mkt"])).fit()
    print(f"""
    ❌ 陷阱 1："超额收益"可能只是高 Beta
       模拟牛市一年：真实 α = 0、真实 β = 1.5
       市场总收益 {(1 + d['mkt']).prod() - 1:+.1%}，股票总收益 {(1 + d['stock']).prod() - 1:+.1%}
       简单超额（算术年化）：{(d['stock'] - d['mkt']).mean() * n:+.1%}  ← 看起来很厉害
       回归估计：β = {model.params['mkt']:.2f}，年化 α = {model.params['const'] * n:+.1%}，α 的 t 值 {model.tvalues['const']:.2f}
       {'→ α 的 t 值绝对值小于 2：一年样本里，这个 α 与真实值 0 在统计上无法区分' if abs(model.tvalues['const']) < 2 else '→ 这次抽样的 α 显著不为 0，但真实值是 0：这就是抽样误差'}
       → 牛市里高 Beta 股票天然跑赢指数；扣除 Beta 暴露后再看才是真本事
    """)

    # 陷阱 2：停牌错配 —— 比较三种处理方式
    code = prices.isna().sum().idxmax()
    n_na = int(prices[code].isna().sum())
    print(f"    ⚠️ 陷阱 2：停牌日对齐（{code}，价格缺失 {n_na} 天）")
    if n_na > 0:
        def beta_of(stock_ret: pd.Series) -> float:
            d = pd.concat({"s": stock_ret, "m": rets[BENCH]}, axis=1).dropna()
            return d["s"].cov(d["m"]) / d["m"].var()

        ret_raw = rets[code]  # 不填充：停牌日与复牌首日都是 NaN，会被 dropna 排除
        ret_ffill = prices[code].ffill().pct_change()  # ffill：停牌日收益 0，复牌日 = 停牌期累计收益
        resume = prices.index[prices[code].notna() & prices[code].shift(1).isna()]
        resume = resume[resume > prices[code].first_valid_index()]  # 第一天不算复牌
        print(f"""       不填充（停牌日与复牌首日被排除）         β = {beta_of(ret_raw):.3f}
       先 ffill 再算收益（含 0 收益与累计跳变）  β = {beta_of(ret_ffill):.3f}
       复牌日（{len(resume)} 天）的 ffill 收益 vs 指数当天收益：""")
        print(pd.DataFrame({"个股(含停牌期累计)": ret_ffill.loc[resume], "指数当天": rets[BENCH].loc[resume]})
              .to_string(float_format=lambda v: f"{v:+.2%}"))
        print("       复牌日收益是多日累计，却和指数单日收益配对 → 回归里的错配点；两种 β 的差异就来自这里")
    else:
        print("       当前数据没有停牌，无法演示；联网运行时 000333 有真实缺失段")

    print("""
    ❌ 陷阱 3：R² 很低是常态，不是 bug
       个股日收益对指数回归的 R² 往往只有零点几，说明大部分波动是个股特有的
    """)


# ==================== 第七部分：可视化 ====================


def plot_relative(rets: pd.DataFrame, nav1: pd.Series, nav2: pd.Series, rb: pd.DataFrame):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    d = rets[["600519", BENCH]].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot((1 + d).cumprod(), lw=1, label=["贵州茅台", "沪深300"])
    axes[0, 0].set_title("净值对比")
    axes[0, 0].legend()

    axes[0, 1].plot(nav1.index, nav1, lw=1, label="方法1：每日超额连乘")
    axes[0, 1].plot(nav2.index, nav2, lw=1, label="方法2：净值相除")
    axes[0, 1].axhline(1, color="gray", lw=0.5)
    axes[0, 1].set_title("累计超额：两种算法")
    axes[0, 1].legend()

    axes[1, 0].scatter(d[BENCH], d["600519"], s=4, alpha=0.4)
    beta = d["600519"].cov(d[BENCH]) / d[BENCH].var()
    alpha = d["600519"].mean() - beta * d[BENCH].mean()
    xs = np.array([d[BENCH].min(), d[BENCH].max()])
    axes[1, 0].plot(xs, alpha + beta * xs, "r-", lw=1.5, label=f"β = {beta:.2f}")
    axes[1, 0].set_xlabel("沪深300 日收益")
    axes[1, 0].set_ylabel("贵州茅台 日收益")
    axes[1, 0].set_title("CAPM 回归散点图")
    axes[1, 0].legend()

    for c in rb.columns:
        axes[1, 1].plot(rb.index, rb[c], lw=1, label=STOCKS[c])
    axes[1, 1].axhline(1, color="gray", lw=0.5, ls="--")
    axes[1, 1].set_title("120 日滚动 Beta")
    axes[1, 1].legend()

    path = save_figure(fig, "08_benchmark_beta.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成资产相互独立，Beta/R² 接近 0）\n")
    prices = load_prices()
    rets = prices.pct_change(fill_method=None)  # 不填充：停牌日收益保持 NaN
    why_benchmark(rets)
    te_ir(rets)
    nav1, nav2 = cumulative_excess(rets)
    beta_alpha(rets)
    rb = rolling_beta(rets)
    pitfalls(prices, rets)
    plot_relative(rets, nav1, nav2, rb)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把基准换成中证 500（000905），比较 600519 的 β、α、R² 如何变化，说明基准选择的影响
    2. 把基准换成沪深 300 ETF（510300，用 get_etf_daily），它的收盘价含不含分红？
       比较用指数和用 ETF 作基准时的年化超额差异
    3. 构造"Beta 中性"组合：每天持有 1 份 600519，并按前一日 120 日滚动 β 做空指数
       （注意用 shift(1)），计算对冲后的年化波动与 R²
    4. 分别在 2019–2020 和 2021–2024 两段样本上估计 Beta 和 Alpha，
       结果差异说明了什么？
    """)


if __name__ == "__main__":
    main()
