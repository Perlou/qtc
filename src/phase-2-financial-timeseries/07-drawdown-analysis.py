"""
回撤与水下曲线
==============

学习目标：
    1. 理解回撤（Drawdown）与水下曲线（Underwater Curve）的含义
    2. 计算最大回撤（MDD）及其峰值日、谷底日、恢复日
    3. 列出前 N 大回撤区间，分析下跌期、恢复期与持续时间
    4. 理解回撤恢复的不对称性：亏 50% 需要涨 100%
    5. 认识 MDD 的统计性质：路径依赖、样本越长 MDD 越大

核心概念：
    - Peak_t = max_{s≤t} V_s（截至当天的历史最高净值，天然无前视）
    - DD_t = V_t / Peak_t - 1（≤ 0），MDD = min DD_t
    - 回本所需涨幅 = 1 / (1 + DD) - 1
    - Calmar Ratio = 年化收益 / |最大回撤|

前置知识：
    - 01-returns-simple-vs-log.py、02-annualization-volatility.py

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
TRADING_DAYS = 252
START, END = "20180101", "20241231"


def drawdown_analysis(nav: pd.Series) -> dict:
    """最大回撤及其峰值日、谷底日、恢复日"""
    peak = nav.cummax()
    dd = nav / peak - 1  # 水下曲线
    trough_date = dd.idxmin()
    peak_date = nav.loc[:trough_date].idxmax()  # 谷底之前的峰值
    after = nav.loc[trough_date:]
    recovered = after[after >= nav.loc[peak_date]]  # 谷底之后首次回到峰值水平
    recovery_date = recovered.index[0] if len(recovered) else None
    return {
        "max_drawdown": dd.min(),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,  # None 表示至今未恢复
        "underwater": dd,
    }


def top_drawdowns(nav: pd.Series, top: int = 5) -> pd.DataFrame:
    """列出前 N 大回撤区间（按深度排序）"""
    dd = nav / nav.cummax() - 1
    is_under = dd < 0
    # 每次"从水面进入水下"开启一个新区间：用 cumsum 给每段水下期编号
    episode = (is_under & ~is_under.shift(1, fill_value=False)).cumsum()
    episode = episode.where(is_under)
    rows = []
    for _, seg in dd.groupby(episode):
        start_i = nav.index.get_loc(seg.index[0])
        end_i = nav.index.get_loc(seg.index[-1])
        recovered = end_i + 1 < len(nav)
        rows.append({
            "峰值日": nav.index[start_i - 1].date() if start_i > 0 else seg.index[0].date(),
            "谷底日": seg.idxmin().date(),
            "恢复日": nav.index[end_i + 1].date() if recovered else "未恢复",
            "深度": seg.min(),
            "下跌天数": nav.index.get_loc(seg.idxmin()) - start_i + 1,
            "水下总天数": len(seg),
        })
    return pd.DataFrame(rows).nsmallest(top, "深度").reset_index(drop=True)


def load_navs() -> pd.DataFrame:
    closes = {
        "600519 贵州茅台": get_stock_daily_adj("600519", START, END)["adj_close"],
        "601088 中国神华": get_stock_daily_adj("601088", START, END)["adj_close"],
        "000300 沪深300": get_index_daily("000300", START, END)["close"],
    }
    df = pd.DataFrame(closes).dropna()
    return df / df.iloc[0]


# ==================== 第一部分：回撤的直觉 ====================


def intuition():
    print("=" * 60)
    print("第一部分：回撤的直觉")
    print("=" * 60)

    print("""
    回撤回答的问题：如果我在最高点买入，现在亏了多少？

    净值
     ▲          峰值 ●
     │         ╱      ╲                  ● 新高（回撤结束）
     │   ●   ╱          ╲              ╱
     │  ╱ ╲ ╱             ╲          ╱
     │ ╱   ●                ╲      ╱
     │╱                       ● 谷底
     └──────────────────────────────────────▶ 时间
              │← 下跌期 →│← 恢复期 →│
              │←─── 回撤持续期 ────→│

    Peak_t = 截至 t 的最高净值（cummax）     DD_t = V_t / Peak_t - 1
    水下曲线 = 每天的 DD_t：0 表示创新高（浮出水面），负值表示仍在水下
    """)

    dds = np.array([-0.05, -0.10, -0.20, -0.30, -0.50, -0.80, -0.90])
    tbl = pd.DataFrame({"回撤": dds, "回本所需涨幅": 1 / (1 + dds) - 1})
    print("📐 回撤恢复的不对称性：回本所需涨幅 = 1/(1+DD) - 1")
    print(tbl.to_string(index=False, formatters={"回撤": "{:.0%}".format, "回本所需涨幅": "{:+.1%}".format}))
    print("   💡 回撤越深，回本越难，而且是非线性地变难 —— 控制回撤比追求收益更重要")


# ==================== 第二部分：最大回撤及其起止日期 ====================


def max_drawdown_dates(navs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：最大回撤及其起止日期")
    print("=" * 60)

    rows = []
    for name in navs.columns:
        nav = navs[name]
        res = drawdown_analysis(nav)
        idx = nav.index
        p, t, r = res["peak_date"], res["trough_date"], res["recovery_date"]
        rows.append({
            "资产": name,
            "最大回撤": res["max_drawdown"],
            "峰值日": p.date(),
            "谷底日": t.date(),
            "恢复日": r.date() if r is not None else "未恢复",
            "下跌期(交易日)": idx.get_loc(t) - idx.get_loc(p),
            "恢复期(交易日)": idx.get_loc(r) - idx.get_loc(t) if r is not None else "-",
            "当前回撤": res["underwater"].iloc[-1],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={
        "最大回撤": "{:.2%}".format, "当前回撤": "{:.2%}".format,
    }))
    print("""
    💡 读表：
       - 峰值日用 nav.loc[:谷底日].idxmax() 找"谷底之前"的最高点，而不是全样本最高点
       - "未恢复"说明截至样本结束仍没回到前高 —— 这在实际中意味着投资者可能早已离场
       - 当前回撤 = 最后一天的水下深度，是监控实盘策略的常用指标
    """)


# ==================== 第三部分：前 N 大回撤区间 ====================


def top_n_drawdowns(navs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：前 N 大回撤区间")
    print("=" * 60)

    for name in navs.columns[:2]:
        tbl = top_drawdowns(navs[name], top=5)
        print(f"\n📋 {name} 前 5 大回撤：")
        print(tbl.to_string(index=False, formatters={"深度": "{:.2%}".format}))
    print("""
    💡 只看 MDD 一个数字会丢失信息：
       - 回撤是否集中在某一段时间？（比如一次股灾 vs 反复出现）
       - 第 2、3 大回撤有多深？如果都接近 MDD，说明这种深度的回撤是"常态"
    """)


# ==================== 第四部分：水下统计与 Calmar ====================


def underwater_stats(navs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：水下时间统计与 Calmar 比率")
    print("=" * 60)

    rows = []
    for name in navs.columns:
        nav = navs[name]
        dd = nav / nav.cummax() - 1
        under = dd < 0
        run_id = (under != under.shift()).cumsum()
        longest = under.groupby(run_id).sum().max()  # 最长连续水下天数
        years = (len(nav) - 1) / TRADING_DAYS  # n 个净值点只有 n-1 个收益周期
        cagr = nav.iloc[-1] ** (1 / years) - 1
        rows.append([name, under.mean(), int(longest), dd.mean(), cagr, dd.min(), cagr / abs(dd.min())])
    df = pd.DataFrame(rows, columns=["资产", "水下天数占比", "最长水下(交易日)", "平均回撤", "CAGR", "MDD", "Calmar"])
    print(df.to_string(index=False, formatters={
        "水下天数占比": "{:.1%}".format, "平均回撤": "{:.2%}".format, "CAGR": "{:+.2%}".format,
        "MDD": "{:.2%}".format, "Calmar": "{:.2f}".format,
    }))
    print("""
    💡 回撤是投资者真正"感受到"的风险：
       - 波动率对称地惩罚涨和跌，但没人会因为暴涨而赎回基金
       - 水下天数占比高 = 大部分时间投资者都在"亏钱的感觉"里
       - Calmar = CAGR / |MDD|：每承受 1 单位最大回撤换来多少年化收益
    """)


# ==================== 第五部分：向量化 vs 循环 ====================


def mdd_loop(values: np.ndarray) -> float:
    """O(n) 单次遍历：维护历史最高点，更新最大回撤"""
    peak, mdd = values[0], 0.0
    for v in values:
        if v > peak:
            peak = v
        mdd = min(mdd, v / peak - 1)
    return mdd


def vectorized_check(navs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：向量化实现 vs 循环实现（对拍）")
    print("=" * 60)
    for name in navs.columns:
        nav = navs[name]
        vec = (nav / nav.cummax() - 1).min()
        loop = mdd_loop(nav.to_numpy())
        np_ver = (nav.to_numpy() / np.maximum.accumulate(nav.to_numpy()) - 1).min()
        print(f"   {name}: pandas cummax {vec:.6%} | NumPy maximum.accumulate {np_ver:.6%} | 循环 {loop:.6%}")
    print("   ✅ 三者一致。cummax / np.maximum.accumulate 就是\"截至当天的最高点\"的向量化写法")


# ==================== 第六部分：MDD 随样本长度增大 ====================


def mdd_vs_sample_length():
    print("\n" + "=" * 60)
    print("第六部分：MDD 是路径依赖的极值统计量")
    print("=" * 60)

    rng = np.random.default_rng(2024)
    mu, sigma, n_paths = 0.08, 0.25, 2000
    dt = 1 / TRADING_DAYS
    max_years = 10
    log_ret = rng.normal((mu - sigma**2 / 2) * dt, sigma * np.sqrt(dt), size=(n_paths, max_years * TRADING_DAYS))
    nav = np.exp(np.cumsum(log_ret, axis=1))

    rows = []
    for years in [1, 3, 5, 10]:
        sub = nav[:, : years * TRADING_DAYS]
        mdd = (sub / np.maximum.accumulate(sub, axis=1) - 1).min(axis=1)
        rows.append([years, np.median(mdd), np.percentile(mdd, 5), (mdd < -0.3).mean()])
    df = pd.DataFrame(rows, columns=["样本年数", "MDD 中位数", "MDD 5% 分位(较差情形)", "MDD 超过 30% 的概率"])
    print(f"🧪 {n_paths} 条模拟路径，参数完全相同（年化收益 {mu:.0%}、年化波动 {sigma:.0%}）：")
    print(df.to_string(index=False, formatters={
        "MDD 中位数": "{:.1%}".format, "MDD 5% 分位(较差情形)": "{:.1%}".format, "MDD 超过 30% 的概率": "{:.1%}".format,
    }))
    print("""
    💡 同一个"策略"，样本越长 MDD 越大（有更多机会遇到坏事件）：
       - 3 年回测和 10 年回测之间直接比较 MDD 是不公平的
       - 同样长度下 MDD 也有很大的随机性（看中位数与 5% 分位的差距）
       - Phase 3 的蒙特卡洛会给出 MDD 的"分布"，而不是只看一个点
    """)


# ==================== 第七部分：可视化 ====================


def plot_drawdown(navs: pd.DataFrame):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name in navs.columns:
        nav = navs[name]
        axes[0].plot(nav.index, nav, lw=1, label=name)
        axes[1].plot(nav.index, nav / nav.cummax() - 1, lw=1, label=name)

    first = navs.columns[0]
    res = drawdown_analysis(navs[first])
    end = res["recovery_date"] or navs.index[-1]
    axes[0].axvspan(res["peak_date"], end, color="red", alpha=0.08, label=f"{first} 最大回撤区间")
    axes[0].set_yscale("log")
    # 对数轴：数据不足一个数量级时默认只标 "1"，手动给出常用刻度并显示为普通数字
    axes[0].set_yticks([t for t in [0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8] if navs.min().min() <= t <= navs.max().max()])
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].set_title("净值（对数坐标）")
    axes[0].legend(fontsize=8)
    axes[1].set_title("水下曲线（回撤）")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].legend(fontsize=8, loc="lower left")

    path = save_figure(fig, "07_drawdown_underwater.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    intuition()
    navs = load_navs()
    max_drawdown_dates(navs)
    top_n_drawdowns(navs)
    underwater_stats(navs)
    vectorized_check(navs)
    mdd_vs_sample_length()
    plot_drawdown(navs)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 计算 600519 每个自然年的最大回撤（groupby 年份，每年内部重新从 1 开始计算净值）
    2. 实现"滚动 250 日最大回撤"：每天回看过去一年的最大回撤，画出来观察其变化
    3. 给 top_drawdowns 增加"恢复天数"列（谷底到恢复日），并找出恢复最慢的一次回撤
    4. 在 01 课的均线择时策略（Phase 1）上计算 MDD 和 Calmar，与买入持有对比，
       验证"均线择时主要价值在于降低回撤"这句话在你的数据上是否成立
    """)


if __name__ == "__main__":
    main()
