"""
年化收益与年化波动率
====================

学习目标：
    1. 理解"每年周期数 N"的含义，知道 A 股、周频、月频、加密货币各用多少
    2. 区分几何年化（CAGR）与算术年化，理解两者差异来自波动拖累
    3. 理解 √T 法则：为什么年化波动率要乘 √N 而不是 N
    4. 能正确年化日频、周频、月频数据，并解释三者结果为何不完全相同
    5. 认识年化的陷阱：短样本年化、收益自相关、加密货币用错 N

核心概念：
    - CAGR：(V_T / V_0)^(N/n) - 1
    - 算术年化：R̄ × N（常用于 Sharpe 分子）
    - 年化波动率：σ_period × √N（假设收益独立同分布）

前置知识：
    - 01-returns-simple-vs-log.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_index_daily, get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
TRADING_DAYS = 252  # 全项目统一的每年交易日数（也可以改成 244），不要到处写魔法数字
START, END = "20180101", "20241231"


def annualize_return(ret: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """几何年化收益（CAGR）"""
    ret = ret.dropna()
    total = (1 + ret).prod()
    return total ** (periods_per_year / len(ret)) - 1


def annualize_mean(ret: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """算术年化收益：周期均值 × N"""
    return ret.dropna().mean() * periods_per_year


def annualize_vol(ret: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """年化波动率：周期标准差 × √N"""
    return ret.dropna().std() * np.sqrt(periods_per_year)


# ==================== 第一部分：每年有多少个周期 ====================


def periods_per_year():
    print("=" * 60)
    print("第一部分：每年有多少个周期")
    print("=" * 60)

    print("""
    年化 = 把不同频率、不同长度的数据换算到同一把尺子（一年）上。
    就像把"米/秒"换成"公里/小时"，单位统一才能比较。

    ┌──────────────────┬──────────────┬───────────────────────────────────┐
    │ 数据频率         │ 每年周期数 N │ 说明                              │
    ├──────────────────┼──────────────┼───────────────────────────────────┤
    │ A 股日频         │ 约 240–245   │ 常直接用 252（美股惯例）或 244    │
    │ 周频             │ 52           │                                   │
    │ 月频             │ 12           │                                   │
    │ 加密货币日频     │ 365          │ 7×24 交易，无休市                 │
    │ 加密货币小时频   │ 8760         │ 365 × 24                          │
    └──────────────────┴──────────────┴───────────────────────────────────┘
    """)

    idx = get_index_daily("000300", START, END)
    counts = idx.groupby(idx.index.year).size()
    print("📅 数据中每年的交易日数（沪深 300）：")
    print("   " + "  ".join(f"{y}:{n}" for y, n in counts.items()))
    if counts.median() > 250:
        print("   （每年超过 250 天：这是合成数据，用工作日近似交易日，不含节假日）")
    print(f"\n💡 本课程统一使用 TRADING_DAYS = {TRADING_DAYS}；关键不在于选哪个数，而在于全项目统一")


# ==================== 第二部分：几何年化 vs 算术年化 ====================


def cagr_vs_arithmetic():
    print("\n" + "=" * 60)
    print("第二部分：几何年化（CAGR）vs 算术年化")
    print("=" * 60)

    print("""
    几何年化 CAGR = (期末净值 / 期初净值)^(N / n) - 1   ← 真实的复利增长率（推荐用于报告）
    算术年化      = 日均收益 × N                          ← 常用于 Sharpe Ratio 的分子
    对数年化      = 日均对数收益 × N，CAGR = exp(对数年化) - 1
    算术年化 - 对数年化 ≈ σ²_ann / 2（波动拖累），波动越大差得越多
    """)

    assets = {
        "600519 贵州茅台": get_stock_daily_adj("600519", START, END)["adj_close"],
        "601088 中国神华": get_stock_daily_adj("601088", START, END)["adj_close"],
        "000300 沪深300": get_index_daily("000300", START, END)["close"],
    }
    rows = []
    for name, close in assets.items():
        ret = close.pct_change().dropna()
        cagr = annualize_return(ret)
        arith = annualize_mean(ret)
        log_ann = np.log1p(ret).mean() * TRADING_DAYS  # 对数年化：CAGR = exp(对数年化) - 1
        vol = annualize_vol(ret)
        rows.append([name, len(ret), cagr, arith, log_ann, arith - log_ann, vol ** 2 / 2, vol, arith / vol])
    df = pd.DataFrame(
        rows, columns=["资产", "天数", "CAGR", "算术年化", "对数年化", "拖累", "σ²/2", "年化波动", "Sharpe(Rf=0)"]
    )
    pct = "{:+.2%}".format
    print(df.to_string(index=False, formatters={
        "CAGR": pct, "算术年化": pct, "对数年化": pct, "拖累": pct, "σ²/2": pct,
        "年化波动": "{:.2%}".format, "Sharpe(Rf=0)": "{:.2f}".format,
    }))
    print("""
    💡 读表：
       - 对数年化 = 日均对数收益 × N，CAGR = exp(对数年化) - 1
       - 拖累 = 算术年化 - 对数年化，与 σ²/2 很接近 → 验证了 g ≈ R̄ - σ²/2
       - 注意"算术年化 - CAGR"并不等于 σ²/2：exp() 是非线性的，收益越高偏离越大
       - Sharpe Ratio = 年化超额收益 / 年化波动，这里暂取无风险利率为 0，Phase 4 会正式讲
    """)


# ==================== 第三部分：√T 法则 ====================


def sqrt_t_rule():
    print("\n" + "=" * 60)
    print("第三部分：√T 法则")
    print("=" * 60)

    print("""
    如果每日收益相互独立、方差都是 σ²，那么 T 天收益之和的方差 = T·σ²
    → 标准差 = σ·√T（独立变量的方差可加，标准差不可加）

    🍺 醉汉走路：每步随机左或右，走 100 步后离起点的典型距离约 √100 = 10 步，
       而不是 100 步 —— 很多步相互抵消了。
    """)

    rng = np.random.default_rng(0)
    sigma = 0.02
    steps = rng.normal(0, sigma, size=(20000, 252))  # 2 万条路径，每条 252 天
    paths = steps.cumsum(axis=1)
    rows = []
    for t in [1, 4, 16, 64, 252]:
        realized = paths[:, t - 1].std()
        rows.append([t, realized, sigma * np.sqrt(t), sigma * t])
    df = pd.DataFrame(rows, columns=["持有天数 T", "模拟的累计收益标准差", "σ·√T", "σ·T（错误）"])
    print(f"🧪 模拟：日波动 σ = {sigma:.0%}，2 万条随机路径")
    print(df.to_string(index=False, formatters={c: "{:.2%}".format for c in df.columns[1:]}))
    print("\n💡 模拟结果与 σ·√T 吻合，与 σ·T 相差甚远")


# ==================== 第四部分：日 / 周 / 月频的年化 ====================


def multi_frequency_annualization() -> pd.Series:
    print("\n" + "=" * 60)
    print("第四部分：日频 / 周频 / 月频分别年化")
    print("=" * 60)

    close = get_stock_daily_adj("600519", START, END)["adj_close"]
    daily = close.pct_change()
    weekly = close.resample("W-FRI").last().pct_change()
    monthly = close.resample("ME").last().pct_change()  # pandas ≥ 2.2 月末用 "ME"

    table = pd.DataFrame(
        {
            "样本数": [daily.count(), weekly.count(), monthly.count()],
            "N": [TRADING_DAYS, 52, 12],
            "周期波动": [daily.std(), weekly.std(), monthly.std()],
            "年化波动": [annualize_vol(daily), annualize_vol(weekly, 52), annualize_vol(monthly, 12)],
            "CAGR": [annualize_return(daily), annualize_return(weekly, 52), annualize_return(monthly, 12)],
        },
        index=["日频", "周频", "月频"],
    )
    print("📊 600519 用不同频率计算的年化指标：")
    print(table.to_string(formatters={
        "周期波动": "{:.2%}".format, "年化波动": "{:.2%}".format, "CAGR": "{:+.2%}".format,
    }))
    print("""
    💡 解读：
       - 三个年化波动"大致相近但不完全相同"：低频样本少、估计误差更大
       - 如果差别很大，往往说明收益存在自相关（下面的实验）
    """)
    years = [daily.count() / TRADING_DAYS, weekly.count() / 52, monthly.count() / 12]
    calendar_years = (close.index[-1] - close.index[0]).days / 365.25
    print(f"""    🔍 三个 CAGR 为什么不同？
       1. 起点不同：周/月频的第一个收益从第一个周末/月末才开始算
       2. 推算的"年数" n/N 不同：日频 {years[0]:.2f} 年、周频 {years[1]:.2f} 年、月频 {years[2]:.2f} 年，
          而这段数据实际跨越 {calendar_years:.2f} 个日历年。
    """)
    if years[0] < calendar_years:
        print("       A 股每年实际交易日少于 252，用 252 推算的年数偏少 → 正收益时 CAGR 被高估、负收益时亏损被放大")
    else:
        print("       本数据每年的周期数多于 252（如离线合成数据按工作日生成），推算的年数偏多 → CAGR 被\"摊薄\"")
    print("       N 的选择是有后果的：要么用实际交易日数，要么全项目统一并在报告中注明\n")
    return daily


def autocorrelation_effect():
    """收益自相关会让 √T 法则失效"""
    print("🧪 实验：收益自相关时，√T 年化会失真")
    rng = np.random.default_rng(1)
    n = 252 * 40
    rows = []
    for phi, label in [(-0.3, "负自相关（均值回归）"), (0.0, "独立（i.i.d.）"), (0.3, "正自相关（趋势）")]:
        eps = rng.normal(0, 0.01, n)
        r = np.empty(n)
        r[0] = eps[0]
        for t in range(1, n):  # AR(1)：今天的收益部分延续昨天
            r[t] = phi * r[t - 1] + eps[t]
        s = pd.Series(r, index=pd.bdate_range("1990-01-01", periods=n))
        nav = (1 + s).cumprod()
        m = nav.resample("ME").last().pct_change()
        vd, vm = annualize_vol(s), annualize_vol(m, 12)
        rows.append([label, phi, vd, vm, vm / vd])
    df = pd.DataFrame(rows, columns=["情形", "自相关 φ", "日频年化波动", "月频年化波动", "月/日"])
    print(df.to_string(index=False, formatters={
        "日频年化波动": "{:.2%}".format, "月频年化波动": "{:.2%}".format, "月/日": "{:.2f}".format,
    }))
    print("""
    💡 正自相关（趋势）→ 长周期波动比 √N 推算的更大；负自相关 → 更小。
       现实提醒：持有流动性差资产的基金，月度净值经常被"平滑"——报告出来的收益
       波动偏小且带正自相关，直接用月度数据 × √12 年化会严重低估真实风险。
    """)


# ==================== 第五部分：年化的陷阱 ====================


def annualization_pitfalls(daily: pd.Series):
    print("\n" + "=" * 60)
    print("第五部分：年化的陷阱")
    print("=" * 60)

    # 陷阱 1：短样本年化
    roll20 = (1 + daily).rolling(20).apply(np.prod, raw=True) - 1
    best_end = roll20.idxmax()
    best = roll20.max()
    annualized = (1 + best) ** (TRADING_DAYS / 20) - 1
    print(f"""
    ❌ 陷阱 1：样本太短不要年化
       600519 表现最好的 20 个交易日（截至 {best_end.date()}）：区间收益 {best:+.2%}
       如果"年化"：(1 + {best:.4f})^(252/20) - 1 = {annualized:+,.0%}  ← 毫无意义的数字
       规则：样本不足 1 年时，报告区间收益，不要年化；或至少明确注明
    """)

    # 陷阱 2：加密货币用 252
    daily_vol = 0.035
    print(f"""    ❌ 陷阱 2：加密货币用 252 年化
       假设某加密资产日波动 {daily_vol:.1%}（每天都交易，包括周末）：
       用 √365 年化：{daily_vol * np.sqrt(365):.1%}
       用 √252 年化：{daily_vol * np.sqrt(252):.1%}  ← 低估了真实年化波动
    """)

    # 陷阱 3：用算术平均 × N 报告收益
    yearly = (1 + daily).groupby(daily.index.year).prod() - 1
    print("    ❌ 陷阱 3：各年收益简单平均 ≠ 年化收益（600519）")
    print("       各年收益：" + "  ".join(f"{y}:{r:+.1%}" for y, r in yearly.items()))
    print(f"       简单平均：{yearly.mean():+.2%}    几何平均（≈CAGR）：{(1 + yearly).prod() ** (1 / len(yearly)) - 1:+.2%}")
    print("       报告\"年化收益\"应使用几何口径")


# ==================== 第六部分：可视化 ====================


def plot_sqrt_rule():
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    close = get_stock_daily_adj("600519", START, END)["adj_close"]
    log_ret = np.log(close).diff().dropna()
    horizons = [1, 2, 5, 10, 20, 40, 60, 120]
    # 用不重叠的 h 日对数收益估计标准差，与 σ_1·√h 比较
    realized = [log_ret.groupby(np.arange(len(log_ret)) // h).sum().std() for h in horizons]
    theory = [log_ret.std() * np.sqrt(h) for h in horizons]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(horizons, realized, "o-", label="实际：不重叠 h 日收益标准差")
    ax.plot(horizons, theory, "s--", label="理论：σ_日 × √h")
    ax.set_xlabel("持有期 h（交易日）")
    ax.set_ylabel("收益标准差")
    ax.set_title("600519：波动率随持有期的增长（√T 法则检验）")
    ax.legend()
    path = save_figure(fig, "02_sqrt_t_rule.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")
    print("   提示：长持有期的不重叠样本很少，实际点会围绕理论线跳动")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    periods_per_year()
    cagr_vs_arithmetic()
    sqrt_t_rule()
    daily = multi_frequency_annualization()
    autocorrelation_effect()
    annualization_pitfalls(daily)
    plot_sqrt_rule()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 TRADING_DAYS 改成 244，重新计算第二部分的表格。哪些指标变化大、哪些几乎不变？
    2. 写 annual_stats(ret, freq) 函数：根据 "D"/"W"/"M" 自动选择 N，返回 CAGR、年化波动、Sharpe
    3. 计算 600519 每个自然年的年化波动率（groupby 年份），哪一年波动最大？对照当年行情解释
    4. 对比日频和月频年化波动的比值（月/日），在 000300、600519、601088 上分别计算，
       结合第四部分的自相关实验，推测哪个资产的收益更有"趋势性"
    """)


if __name__ == "__main__":
    main()
