"""
肥尾、偏度、峰度与正态性检验
============================

学习目标：
    1. 理解四阶矩：均值、方差、偏度、峰度分别描述分布的什么特征
    2. 用模拟看清"负偏策略"为什么是隐形炸弹：Sharpe 看不到偏度
    3. 掌握 Jarque-Bera、Shapiro-Wilk、KS / Lilliefors 正态性检验与 Q-Q 图
    4. 计算 VaR 与 CVaR，并用"突破率"检验正态 VaR 是否低估尾部风险

核心概念：
    - 偏度 Skew = E[((X-μ)/σ)³]：< 0 "经常小赚、偶尔大亏"
    - 超额峰度 = Kurt - 3：> 0 即肥尾（scipy / pandas 默认返回的就是超额峰度）
    - 正态性检验的原假设 H0：数据服从正态分布；p 值小 → 拒绝正态
    - VaR_c = -q_{1-c}(R)；CVaR_c = -E[R | R ≤ q_{1-c}(R)]

前置知识：
    - 01-probability-distributions.py

环境要求：
    - pip install numpy pandas scipy statsmodels matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据没有肥尾，课程会同时用 t 分布模拟数据演示）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

from utils.data import get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


def is_synthetic(index) -> bool:
    """utils 的合成数据用工作日近似交易日：出现元旦、国庆前三天这类 A 股休市日，说明拿到的是合成数据。
    （QTC_OFFLINE=1 时若本地已有缓存，utils 仍会读取真实缓存，所以不能只看环境变量）"""
    idx = pd.DatetimeIndex(index)
    return bool((((idx.month == 1) & (idx.day == 1)) | ((idx.month == 10) & (idx.day <= 3))).any())

TRADING_DAYS = 252
START, END = "20180101", "20241231"


def load_returns() -> dict[str, pd.Series]:
    """真实收益（茅台、沪深300）+ 两组模拟参照（正态、t 分布），长度一致"""
    maotai = np.log(get_stock_daily_adj("600519", START, END)["adj_close"]).diff().dropna()
    hs300 = np.log(get_index_daily("000300", START, END)["close"]).diff().dropna()
    n = len(maotai)
    rng = np.random.default_rng(42)
    sd = maotai.std()
    t_df = 4
    t_sample = stats.t.rvs(t_df, size=n, random_state=rng) * sd / np.sqrt(t_df / (t_df - 2))
    return {
        "茅台": maotai,
        "沪深300": hs300,
        "模拟·正态": pd.Series(rng.normal(0, sd, n)),
        "模拟·t(4)": pd.Series(t_sample),
    }


# ==================== 第一部分：四阶矩与偏度的直觉 ====================


def moments_and_skew():
    print("=" * 60)
    print("第一部分：四阶矩与偏度的直觉")
    print("=" * 60)

    print("""
    ┌──────┬──────────────────┬──────────────┬──────────────────┐
    │ 阶   │ 名称             │ 描述         │ 正态分布取值     │
    ├──────┼──────────────────┼──────────────┼──────────────────┤
    │ 一阶 │ 均值 μ           │ 中心位置     │ —                │
    │ 二阶 │ 方差 σ²          │ 离散程度     │ —                │
    │ 三阶 │ 偏度 Skew        │ 对称性       │ 0                │
    │ 四阶 │ 峰度 Kurt        │ 尾部厚度     │ 3（超额峰度 0）  │
    └──────┴──────────────────┴──────────────┴──────────────────┘

    负偏（Skew < 0）："经常小赚，偶尔大亏"  —— 卖期权、网格、无止损均值回归
    正偏（Skew > 0）："经常小亏，偶尔大赚"  —— 趋势跟踪、买期权
    """)

    # 模拟两个"理论年化收益、年化波动都相同"的策略（约 10% / 12%），各模拟 50 年
    rng = np.random.default_rng(1)
    years = 50
    n = TRADING_DAYS * years
    # 负偏：平时稳定小赚，平均约 3 年一次 -12% 的崩盘（类似卖出虚值期权）
    crash = rng.random(n) < 0.0013
    short_vol = np.where(crash, rng.normal(-0.12, 0.02, n), rng.normal(0.00056, 0.0062, n))
    # 正偏：平时小亏，约 2% 的日子大赚（类似趋势跟踪）
    boom = rng.random(n) < 0.02
    trend = np.where(boom, rng.normal(0.04, 0.01, n), rng.normal(-0.000408, 0.005, n))

    rows = []
    sharpes = {}
    for name, r in [("负偏：卖波动率", short_vol), ("正偏：趋势跟踪", trend)]:
        nav = np.cumprod(1 + r)
        mdd = (nav / np.maximum.accumulate(nav) - 1).min()
        sharpes[name] = r.mean() / r.std() * np.sqrt(TRADING_DAYS)
        rows.append([name, f"{r.mean() * TRADING_DAYS:+.1%}", f"{r.std() * np.sqrt(TRADING_DAYS):.1%}",
                     f"{sharpes[name]:.2f}", f"{(r > 0).mean():.0%}",
                     f"{stats.skew(r):+.2f}", f"{stats.kurtosis(r):.1f}", f"{r.min():.1%}", f"{mdd:.1%}"])
    cols = ["策略", "年化收益", "年化波动", "Sharpe", "日胜率", "偏度", "超额峰度", "最差单日", "最大回撤"]
    print(f"🧪 模拟 {years} 年日收益：")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    # 按自然年切开：负偏策略在"没遇到崩盘"的年份里看起来有多美
    yearly = short_vol.reshape(years, TRADING_DAYS)
    has_crash = crash.reshape(years, TRADING_DAYS).any(axis=1)
    yearly_sr = yearly.mean(axis=1) / yearly.std(axis=1) * np.sqrt(TRADING_DAYS)
    print(f"""
    🔍 把负偏策略按年切开：
       {(~has_crash).mean():.0%} 的年份一次崩盘都没遇到，这些年份的平均 Sharpe = {yearly_sr[~has_crash].mean():.2f}
       遇到崩盘的年份，平均 Sharpe = {yearly_sr[has_crash].mean():.2f}
       → 如果你的回测恰好只覆盖 1~2 个"平静年份"，会看到一个极其平滑、Sharpe 很高的净值曲线

    💡 Sharpe 只用均值和方差，看不到偏度：全样本两者 Sharpe 接近（{sharpes['负偏：卖波动率']:.2f} vs {sharpes['正偏：趋势跟踪']:.2f}），
       但"最差单日"和回撤结构完全不同。这就是 Phase 4 要补充 Sortino、Calmar、尾部指标的原因。
    ⚠️ 库的约定：scipy.stats.kurtosis 默认 fisher=True、pandas .kurt() 都返回"超额峰度"。
    """)


# ==================== 第二部分：真实收益的偏度与峰度 ====================


def real_moments(rets: dict[str, pd.Series]):
    print("\n" + "=" * 60)
    print("第二部分：真实收益的偏度与峰度")
    print("=" * 60)

    rows = []
    for name, r in rets.items():
        rows.append([name, len(r), r.std() * np.sqrt(TRADING_DAYS), stats.skew(r), stats.kurtosis(r),
                     r.kurt(), r.min(), r.max()])
    table = pd.DataFrame(rows, columns=["序列", "样本数", "年化波动", "偏度", "超额峰度(scipy)", "超额峰度(pandas)", "最小", "最大"])
    print(table.round(4).to_string(index=False))
    print("""
    💡 scipy 与 pandas 的峰度略有差异：pandas 用了无偏修正，大样本下两者几乎相同。
       "模拟·正态"的超额峰度应接近 0，"模拟·t(4)"明显大于 0 —— 它们是两把参照尺。
    """)
    if is_synthetic(rets["茅台"].index) or is_synthetic(rets["沪深300"].index):
        print("    ⚠️ 离线模式：茅台、沪深300 是正态 GBM 合成数据，峰度会接近 0，看不到真实市场的肥尾；")
        print("       请以'模拟·t(4)'这一行理解肥尾长什么样。")


# ==================== 第三部分：正态性检验 ====================


def normality_tests(rets: dict[str, pd.Series]):
    print("\n" + "=" * 60)
    print("第三部分：正态性检验")
    print("=" * 60)

    print("""
    所有检验的原假设 H0 都是"数据服从正态分布"：
       p < 0.05 → 拒绝正态；p ≥ 0.05 → 证据不足以拒绝（≠ 证明了正态）

    ┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
    │ 检验            │ 原理                         │ 注意                         │
    ├─────────────────┼──────────────────────────────┼──────────────────────────────┤
    │ Jarque-Bera     │ 偏度、峰度是否偏离 0 和 3    │ 大样本下非常灵敏             │
    │ Shapiro-Wilk    │ 排序值与正态期望值的相关     │ n > 5000 时 p 值可能不准     │
    │ KS（参数已估）  │ 经验 CDF 与理论 CDF 最大距离 │ 用同一数据估参数 → p 值偏大  │
    │ Lilliefors      │ KS 的修正版                  │ 参数估计时应该用它           │
    └─────────────────┴──────────────────────────────┴──────────────────────────────┘
    JB = n/6 · (S² + (K-3)²/4)，H0 下近似服从自由度 2 的卡方分布
    """)

    rows = []
    for name, r in rets.items():
        x = r.to_numpy()
        jb_stat, jb_p = stats.jarque_bera(x)
        sw_stat, sw_p = stats.shapiro(x[:5000])
        z = (x - x.mean()) / x.std(ddof=1)
        ks_stat, ks_p = stats.kstest(z, "norm")
        lf_stat, lf_p = lilliefors(x, dist="norm", pvalmethod="table")
        rows.append([name, f"{jb_stat:.1f}", f"{jb_p:.2e}", f"{sw_p:.2e}", f"{ks_p:.2e}", f"{lf_p:.3f}"])
    print(pd.DataFrame(rows, columns=["序列", "JB 统计量", "JB p", "Shapiro p", "KS p(未修正)", "Lilliefors p"]).to_string(index=False))
    print("\n   注：statsmodels 的 Lilliefors p 值查表得到，范围被截断在 [0.001, 0.2]")
    print("   注：即使是真正的正态样本，每个检验也有 5% 的概率误拒（第一类错误，03 课），偶尔一个 p < 0.05 不奇怪")

    # 大样本下 JB 非常灵敏：t(30) 与正态已经非常接近
    rng = np.random.default_rng(3)
    print("\n🔬 大样本灵敏度实验：t(30) 分布（与正态肉眼几乎无法区分）")
    for n in [250, 2500, 25000, 250000]:
        x = stats.t.rvs(30, size=n, random_state=rng)
        print(f"   n = {n:>7,}：超额峰度 = {stats.kurtosis(x):+.3f}，JB p = {stats.jarque_bera(x).pvalue:.3g}")
    print("""
    💡 样本足够大时，任何微小的偏离都会被判为"显著"。
       几乎所有资产的日收益都会拒绝正态 —— 这不是让你放弃正态，而是提醒你：正态在尾部会失效。
       比 p 值更重要的是"偏离有多大、在哪里偏离"（Q-Q 图能直接看到）。
    """)


def plot_qq(rets: dict[str, pd.Series]):
    setup_chinese_font()
    fig, axes = plt.subplots(1, len(rets), figsize=(4.2 * len(rets), 4))
    for ax, (name, r) in zip(axes, rets.items()):
        sm.qqplot(r.to_numpy(), line="s", ax=ax, markersize=2)
        ax.set_title(f"Q-Q 图：{name}")
    fig.suptitle("点偏离直线的'S 形'两端 = 肥尾", y=1.02)
    path = save_figure(fig, "02_qq_plots.png", subdir="phase-3")
    print(f"🖼️  Q-Q 图已保存：{path}")


# ==================== 第四部分：VaR 与 CVaR ====================


def var_cvar(r: pd.Series, c: float) -> dict:
    """返回历史法、正态法、t 分布法的 VaR，以及历史 CVaR（均以正数表示亏损）"""
    q = np.quantile(r, 1 - c)
    var_hist = -q
    cvar_hist = -r[r <= q].mean()
    var_norm = -(r.mean() + stats.norm.ppf(1 - c) * r.std())
    df_t, loc_t, scale_t = stats.t.fit(r)
    var_t = -stats.t.ppf(1 - c, df_t, loc_t, scale_t)
    return {"历史 VaR": var_hist, "正态 VaR": var_norm, "t 分布 VaR": var_t, "历史 CVaR": cvar_hist}


def tail_risk(rets: dict[str, pd.Series]):
    print("\n" + "=" * 60)
    print("第四部分：VaR 与 CVaR")
    print("=" * 60)

    print("""
    VaR（风险价值）：在置信水平 c 下，单日亏损"不超过"多少
    CVaR（条件风险价值 / ES）：一旦亏损超过 VaR，"平均"亏多少

       VaR_c  = -q_{1-c}(R)               q 为收益分布的 1-c 分位数
       CVaR_c = -E[R | R ≤ q_{1-c}(R)]
    """)

    for c in [0.95, 0.99]:
        rows = []
        for name, r in rets.items():
            v = var_cvar(r, c)
            rows.append([name] + [f"{x:.2%}" for x in v.values()])
        print(f"📊 置信水平 {c:.0%}（单日）：")
        print(pd.DataFrame(rows, columns=["序列"] + list(v.keys())).to_string(index=False))
        print()

    # 突破率：实际亏损超过 VaR 的天数占比，应接近 1 - c
    print("🔍 正态 VaR 的全样本突破率（同一段数据估计、同一段数据检验，只反映分布形状）：")
    rows = []
    for name, r in rets.items():
        row = [name]
        for c in [0.95, 0.99, 0.995]:
            var_n = -(r.mean() + stats.norm.ppf(1 - c) * r.std())
            row.append(f"{(r < -var_n).mean():.2%}（理论 {1 - c:.1%}）")
        rows.append(row)
    print(pd.DataFrame(rows, columns=["序列", "95%", "99%", "99.5%"]).to_string(index=False))
    print("""
    💡 解读：突破率高于理论值 → 正态假设低估了该置信水平下的尾部风险。
       肥尾数据的特点是"中间更尖、尾部更厚"，所以正态 VaR 在 95% 处未必低估，越往深尾（99%、99.5%）越容易低估。
    """)

    print("🔍 样本外检验（前半段估计 VaR，后半段数突破天数）：")
    rows = []
    for name in ["茅台", "沪深300"]:
        r = rets[name]
        half = len(r) // 2
        train, test = r.iloc[:half], r.iloc[half:]
        v = var_cvar(train, 0.99)
        rows.append([name, f"{train.std() * np.sqrt(TRADING_DAYS):.1%}", f"{test.std() * np.sqrt(TRADING_DAYS):.1%}",
                     f"{(test < -v['正态 VaR']).mean():.2%}", f"{(test < -v['历史 VaR']).mean():.2%}",
                     f"{(test < -v['t 分布 VaR']).mean():.2%}"])
    print(pd.DataFrame(rows, columns=["序列", "前半段年化波动", "后半段年化波动", "正态 99% VaR 突破率",
                                      "历史 99% VaR 突破率", "t 99% VaR 突破率"]).to_string(index=False))
    print("""
    💡 样本外突破率同时受两件事影响：分布形状（肥尾）和波动率水平的变化。
       如果后半段比前半段平静，所有方法的突破率都会偏低；反之都会偏高。
       固定参数的 VaR 跟不上波动率的变化 —— 这正是 11 课 GARCH 动态 VaR 要解决的问题。
       另外记住：VaR 不是最坏情况，它对超过 VaR 的那些日子会亏多少一无所知，CVaR 更能反映尾部。
    """)


# ==================== 第五部分：A 股涨跌停对分布的"截断" ====================


def price_limit_note(maotai: pd.Series):
    print("\n" + "=" * 60)
    print("第五部分：涨跌停对收益分布的截断")
    print("=" * 60)

    simple = np.exp(maotai) - 1
    print(f"""
    主板个股日涨跌幅一般被限制在 ±10% 左右（具体以交易所规则为准），日收益分布被"截断"：
       样本中茅台最大单日涨幅 {simple.max():+.2%}，最大单日跌幅 {simple.min():+.2%}

    ⚠️ 风险并没有消失，而是转移了：
       - 连续一字跌停时，你根本卖不出去 → 日收益看起来"只跌了 10%"，实际是连跌多日的流动性风险
       - 只看日收益的峰度和 VaR，会低估这种风险 → 可以同时看多日（如 5 日）收益的分布
    """)
    if is_synthetic(maotai.index):
        print("    ⚠️ 离线合成数据没有涨跌停限制，上面的最大涨跌幅不代表真实市场")
    r5 = maotai.rolling(5).sum().dropna().iloc[::5]
    print(f"   日收益超额峰度 {stats.kurtosis(maotai):.2f}；不重叠 5 日收益超额峰度 {stats.kurtosis(r5):.2f}"
          f"（样本数 {len(r5)}，估计误差较大）")


# ==================== 主函数 ====================


def main():
    moments_and_skew()
    rets = load_returns()
    real_moments(rets)
    normality_tests(rets)
    plot_qq(rets)
    tail_risk(rets)
    price_limit_note(rets["茅台"])

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第一部分的负偏策略每天崩盘概率为 0.13%。在只有 2 年（504 天）样本时，
       回测里一次崩盘都看不到的概率是多少？（提示：二项分布，再用模拟验证）
    2. 分别计算沪深300 在 2020 年和 2024 年的偏度、峰度，结论稳定吗？
    3. 实现 cornish_fisher_var(r, c)：用偏度与峰度修正正态分位数，与历史 VaR 比较
    4. 对 10 只股票计算 99% 正态 VaR 的样本外突破率，有多少只超过了 1%？
    """)


if __name__ == "__main__":
    main()
