"""
压力测试与情景分析
==================

学习目标：
    1. 理解压力测试与 VaR 的分工：VaR 看"正常时期的分布"，压力测试问"如果发生 X 会怎样"
    2. 历史情景：把当前组合放回 2015 年股灾、2016 年熔断、2020 年 3 月、2022 年等区间，
       情景的起止日期由沪深 300 指数数据自动定位，而不是凭记忆硬编码
    3. 观察危机中相关性的变化：分散化为什么在最需要的时候失效
    4. 因子冲击情景：直接冲击与"条件冲击"（用协方差推断其他因子的联动）
    5. 反向压力测试：亏损 10% 最可能是怎样发生的
    6. A 股特有的流动性压力：连续跌停卖不出去

核心概念：
    - 历史情景：weights @ 区间累计收益（买入持有）
    - 因子传导：Δr_p ≈ x_p' Δf，x_p = B'w 为组合因子暴露
    - 条件情景：E[Δf₂ | Δf₁] = Σ₂₁ Σ₁₁⁻¹ Δf₁
    - 反向压力测试：在 x_p'Δf = -L 的约束下最小化马氏距离 Δf'Σ⁻¹Δf

前置知识：
    - 07-var-cvar.py；Phase 7 因子暴露
    - CONCEPT.md 第 10 节

环境要求：
    - pip install numpy pandas statsmodels matplotlib akshare
    - QTC_OFFLINE=1 时优先读本地缓存；缓存缺失才改用示意参数的模拟收益并打印 ⚠️（此时历史情景没有意义）
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

from portfolio_utils import load_etf_returns, max_drawdown
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 200)

WEIGHTS = {"沪深300": 0.25, "中证500": 0.10, "创业板": 0.05, "纳指": 0.075, "标普500": 0.075, "国债": 0.35, "黄金": 0.10}

# 情景的大致搜索窗口：实际起止日期由 locate_scenarios 在窗口内按沪深 300 指数自动定位
SCENARIO_WINDOWS = {
    "2015 股灾": ("2015-06-01", "2015-09-30"),
    "2016 熔断": ("2015-12-01", "2016-02-29"),
    "2018 贸易摩擦": ("2018-01-01", "2018-12-31"),
    "2020-03 疫情冲击": ("2020-02-01", "2020-04-30"),
    "2022 年初 A股/美股下跌": ("2022-01-01", "2022-05-31"),
    "2024 年初小微盘急跌": ("2024-01-01", "2024-02-29"),
}


def peak_to_trough(close: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """窗口内最大回撤的起止：低点 = 回撤最深的一天，高点 = 低点之前的最高收盘价"""
    drawdown = close / close.cummax() - 1
    trough = drawdown.idxmin()
    return close.loc[:trough].idxmax(), trough


def locate_scenarios(returns: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """在每个搜索窗口内定位高低点，打印实际使用的日期，返回 {情景: (高点日, 低点日)}"""
    index_df = get_index_daily("000300", "20150101", "20241231")
    if is_synthetic(index_df):
        print("⚠️ 沪深300 指数没有本地缓存或下载失败：改用当前收益数据里的沪深300 ETF 净值定位情景日期")
        close, source = (1 + returns["沪深300"]).cumprod(), "沪深300 ETF 净值"
    else:
        close, source = index_df["close"], "沪深300 指数收盘价"

    scenarios, rows = {}, []
    for label, (start, end) in SCENARIO_WINDOWS.items():
        window = close.loc[start:end]
        peak, trough = peak_to_trough(window)
        if peak == trough:  # 窗口内一路上涨，没有回撤，不构成压力情景
            rows.append([label, f"{start} ~ {end}", "—", "—", "—", 0, "窗口内没有回撤，跳过"])
            continue
        scenarios[label] = (peak, trough)
        notes = []
        if peak == window.index[0]:
            notes.append("高点在窗口首日，真实高点可能更早")
        if trough == window.index[-1]:
            notes.append("低点在窗口末日，真实低点可能更晚")
        note = "；".join(notes)
        rows.append([label, f"{start} ~ {end}", peak.strftime("%Y-%m-%d"), trough.strftime("%Y-%m-%d"),
                     f"{close[trough] / close[peak] - 1:+.1%}", int(((window.index > peak) & (window.index <= trough)).sum()), note])
    print(f"情景日期：在大致窗口内，按{source}定位区间最大回撤的高点和低点")
    print(pd.DataFrame(rows, columns=["情景", "搜索窗口", "高点日", "低点日", "回撤", "交易日", "备注"]).to_string(index=False))
    print("   情景收益从高点的下一个交易日算到低点当天（高点当天的上涨不算在内）。")
    print("   其他资产的高低点未必同日，所以下表的区间收益不等于它们各自的最大跌幅。\n")
    return scenarios


def scenario_mask(index: pd.Index, peak: pd.Timestamp, trough: pd.Timestamp) -> np.ndarray:
    return (index > peak) & (index <= trough)


# ==================== 第一部分：历史情景 ====================


def historical_scenarios(returns: pd.DataFrame) -> dict:
    print("=" * 60)
    print("第一部分：历史情景 —— 把当前组合放回危机")
    print("=" * 60)
    scenarios = locate_scenarios(returns)
    print(f"当前组合：{WEIGHTS}（情景内买入持有、不调仓）\n")

    names = list(returns.columns)
    w = pd.Series(WEIGHTS)[names]
    rows = {}
    cums = {}
    for label, (peak, trough) in scenarios.items():
        r = returns[scenario_mask(returns.index, peak, trough)]
        if len(r) == 0:
            continue
        cum = (1 + r).prod() - 1
        cums[label] = cum
        value = (1 + r).cumprod() @ w                     # 买入持有的组合净值
        port_ret = value.pct_change().fillna(value.iloc[0] - 1)
        rows[label] = {**{c: f"{cum[c]:+.1%}" for c in names},
                       "组合": f"{value.iloc[-1] - 1:+.1%}",
                       "组合区间回撤": f"{max_drawdown(port_ret):.1%}",
                       "最差单日": f"{port_ret.min():+.2%}",
                       "交易日": len(r)}
    tbl = pd.DataFrame(rows).T
    print(tbl.to_string())
    def hedge_summary(c: str) -> str:
        vals = pd.Series({k: v[c] for k, v in cums.items()})
        return f"{c}上涨 {(vals > 0).sum()} 个（区间收益 {vals.min():+.1%} ~ {vals.max():+.1%}）"

    n_loss = sum(float(v["组合"].strip("%")) < 0 for v in rows.values())
    key = "2024 年初小微盘急跌"
    size_note = ""
    if key in cums:
        size_note = (f"2024 年初情景：中证500 {cums[key]['中证500']:+.1%}、沪深300 {cums[key]['沪深300']:+.1%}，组合 {rows[key]['组合']}。"
                     "这只是中小盘宽基，微盘股没有包含在本资产池里；\n         隐含小市值暴露的量化选股组合要单独测试（见练习 4）")
    print(f"""
    💡 读表：
       - 同一个组合在不同危机里的损失差异很大：危机的"形状"各不相同（A股独跌、全球同跌、风格踩踏……）
       - {len(rows)} 个情景中，{hedge_summary('国债')}，{hedge_summary('黄金')}，组合亏损 {n_loss} 个：
         避险资产的缓冲大小因情景而异，不能假设每次都够用
       - {size_note}
       ⚠️ 这里只用日收盘价、假设能按收盘价成交；真实危机中还有流动性折价（见第五部分）
    """)

    port_vals = {k: float(v["组合"].strip("%")) for k, v in rows.items()}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = list(port_vals)
    vals = [port_vals[k] for k in labels]
    ax.barh(labels, vals, color=["#e34948" if v < 0 else "#1baf7a" for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.margins(x=0.12)  # 给数值标签留出空间
    ax.set_xlabel("情景区间内组合收益 %")
    ax.set_title("历史情景压力测试")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    path = save_figure(fig, "08_historical_scenarios.png", subdir="phase-8")
    print(f"🖼  历史情景图已保存：{path}")
    return scenarios


# ==================== 第二部分：危机中的相关性 ====================


def crisis_correlation(returns: pd.DataFrame, scenarios: dict):
    print("\n" + "=" * 60)
    print("第二部分：危机中的相关性 —— 分散化何时失效")
    print("=" * 60)

    crisis_mask = pd.Series(False, index=returns.index)
    for peak, trough in scenarios.values():
        crisis_mask |= scenario_mask(returns.index, peak, trough)
    tail_mask = returns["沪深300"] <= returns["沪深300"].quantile(0.05)

    def summary(r: pd.DataFrame) -> dict:
        c = r.corr()
        a_share = ["沪深300", "中证500", "创业板"]
        a_pairs = [c.loc[x, y] for i, x in enumerate(a_share) for y in a_share[i + 1 :]]
        return {
            "天数": len(r),
            "A股内部平均相关": f"{np.mean(a_pairs):+.2f}",
            "沪深300-标普500": f"{c.loc['沪深300', '标普500']:+.2f}",
            "沪深300-国债": f"{c.loc['沪深300', '国债']:+.2f}",
            "沪深300-黄金": f"{c.loc['沪深300', '黄金']:+.2f}",
            "标普500-黄金": f"{c.loc['标普500', '黄金']:+.2f}",
        }

    tbl = pd.DataFrame({
        "非危机区间": summary(returns[~crisis_mask]),
        "危机区间（第一部分合并）": summary(returns[crisis_mask]),
        "沪深300 最差 5% 的日子": summary(returns[tail_mask]),
    }).T
    print(tbl.to_string())
    print("""
    💡 注意"最差 5% 的日子"这一行是按沪深300 收益筛选的条件样本，条件相关会因为截断而失真，
       只能定性看方向。危机中常见的现象：同类资产相关性上升（"千股跌停"）、避险资产关系改变。
       压力测试应使用"危机期间估计的协方差"，正常时期的相关性会低估联动。
    """)
    return crisis_mask


# ==================== 第三部分：因子冲击情景 ====================


def build_factors(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "A股市场": returns["沪深300"],
        "小盘-大盘": returns["中证500"] - returns["沪深300"],
        "海外股票": returns["标普500"],
        "国债": returns["国债"],
        "黄金": returns["黄金"],
    })


def factor_shocks(returns: pd.DataFrame, crisis_mask: pd.Series):
    print("\n" + "=" * 60)
    print("第三部分：因子冲击情景 —— 直接冲击 vs 条件冲击")
    print("=" * 60)

    f = build_factors(returns)
    names = list(returns.columns)
    B = pd.DataFrame(index=names, columns=f.columns, dtype=float)
    r2 = {}
    for c in names:
        res = sm.OLS(returns[c], sm.add_constant(f)).fit()
        B.loc[c] = res.params[f.columns]
        r2[c] = res.rsquared
    w = pd.Series(WEIGHTS)[names]
    x_p = B.T @ w
    print("各资产的因子暴露（时间序列回归，β 越大对该因子越敏感）：")
    print(B.assign(R2=pd.Series(r2)).round(2).to_string())
    print(f"\n组合因子暴露 x_p = B'w：{x_p.round(3).to_dict()}")

    F_normal = f[~crisis_mask].cov() * 21   # 月度协方差
    F_crisis = f[crisis_mask].cov() * 21

    def conditional(shocks: dict, F: pd.DataFrame) -> pd.Series:
        s = list(shocks)
        o = [c for c in F.index if c not in shocks]
        d1 = np.array([shocks[k] for k in s])
        d2 = F.loc[o, s].to_numpy() @ np.linalg.solve(F.loc[s, s].to_numpy(), d1)
        return pd.Series(np.r_[d1, d2], index=s + o)[F.index]

    scenarios = {
        "A股市场 -10%": {"A股市场": -0.10},
        "海外股票 -15%": {"海外股票": -0.15},
        "A股 -10% 且小盘再跑输 5%": {"A股市场": -0.10, "小盘-大盘": -0.05},
        "利率上行：国债 -2%": {"国债": -0.02},
    }
    rows = []
    for label, shocks in scenarios.items():
        direct = pd.Series(0.0, index=f.columns)
        for k, v in shocks.items():
            direct[k] = v
        cn = conditional(shocks, F_normal)
        cc = conditional(shocks, F_crisis)
        rows.append([label, f"{x_p @ direct:+.2%}", f"{x_p @ cn:+.2%}", f"{x_p @ cc:+.2%}",
                     " ".join(f"{k}{v:+.1%}" for k, v in cc.items() if k not in shocks)])
    print("\n组合损益（Δr_p ≈ x_p'Δf，忽略特异收益）：")
    print(pd.DataFrame(rows, columns=["情景", "只冲击指定因子", "条件(正常协方差)", "条件(危机协方差)", "危机协方差推断的其他因子"]).to_string(index=False))
    print("""
    💡 "只冲击指定因子"隐含假设其他因子不动——现实中几乎不可能。
       条件情景用 E[Δf₂|Δf₁] = Σ₂₁Σ₁₁⁻¹Δf₁ 推断联动，用危机期间的协方差更贴近真实的危机传导。
    """)
    return f, x_p, F_normal


# ==================== 第四部分：反向压力测试 ====================


def reverse_stress(x_p: pd.Series, F: pd.DataFrame, loss: float = 0.10):
    print("\n" + "=" * 60)
    print("第四部分：反向压力测试 —— 亏损 10% 最可能怎样发生")
    print("=" * 60)
    x = x_p[F.index].to_numpy()
    Fm = F.to_numpy()
    sigma_p = np.sqrt(x @ Fm @ x)
    df_star = -loss * Fm @ x / (x @ Fm @ x)       # 在 x'Δf = -L 上马氏距离最小的点
    maha = loss / sigma_p
    print(f"""
    问题：min Δf'F⁻¹Δf   s.t. x_p'Δf = -{loss:.0%}
    解：  Δf* = -L · F x / (x'F x)，马氏距离 = L / √(x'Fx) = {maha:.2f}（以月度因子波动为单位）

    "最可能"的 {loss:.0%} 亏损情景（月度）：""")
    for k, v in zip(F.index, df_star):
        print(f"       {k:8s} {v:+.2%}")
    print(f"""
    组合因子月度波动 √(x'Fx) = {sigma_p:.2%}，亏 {loss:.0%} 相当于 {maha:.1f} 个月度标准差的因子冲击
    若因子服从正态，组合因子收益 x'Δf ~ N(0, x'Fx)，单月亏损超过 {loss:.0%} 的概率 = Φ(-{maha:.2f}) ≈ {stats.norm.cdf(-maha):.1e}
    ——但历史告诉我们，危机的频率远高于正态假设（对照第一部分的历史情景）
    💡 反向压力测试最大的价值：告诉你组合最脆弱的方向。冲击幅度最大的因子，就是你真正押注的东西。
    """)


# ==================== 第五部分：A 股流动性压力 ====================


def liquidity_stress():
    print("\n" + "=" * 60)
    print("第五部分：A 股特有压力 —— 跌停卖不出去")
    print("=" * 60)
    rows = []
    for k in range(1, 6):
        rows.append([k, f"{1 - 0.95**k:.1%}", f"{1 - 0.9**k:.1%}", f"{1 - 0.8**k:.1%}"])
    print("连续 k 个一字跌停、期间无法卖出时的累计损失：")
    print(pd.DataFrame(rows, columns=["连续跌停天数", "±5%", "±10%", "±20%"]).to_string(index=False))
    pos = 0.05
    print("""
    涨跌幅限制对应的品种：
    ┌───────┬────────────────────────────────────────────────┐
    │ ±5%   │ 沪深主板 ST/*ST 股票（2026-07-06 之前）        │
    │ ±10%  │ 沪深主板普通股票；ST/*ST 股票（2026-07-06 起） │
    │ ±20%  │ 创业板、科创板                                 │
    └───────┴────────────────────────────────────────────────┘
    ⚠️ 回测和压力测试要按当时的规则：2026-07-06 之前的历史区间，主板 ST/*ST 用 ±5%；
       新规则下这类股票的单日跌幅上限翻倍，同样 k 个跌停的损失更大，历史回撤不能直接外推到今天。
    """)
    print(f"""
    组合中单只股票占 {pos:.0%} 时，若遇 3 个一字跌停（主板），这一只股票就让组合亏 {pos * (1 - 0.9**3):.2%}，
    而且在跌停打开前，止损单、风控减仓指令都无法执行。

    📋 A 股压力测试清单：
    ┌──────────────────┬──────────────────────────────────────────────────┐
    │ 流动性/涨跌停    │ 假设跌停股票无法卖出，损失按连续跌停计           │
    │ 风格崩塌         │ 小盘/微盘因子短期急跌（如 2024 年初）            │
    │ 对冲工具失效     │ 股指期货贴水扩大、融券受限，中性策略对冲成本骤升 │
    │ 相关性趋近 1     │ 所有股票同跌，分散化失效                         │
    │ 政策冲击         │ 交易规则变化（如熔断）、特定交易方式受限         │
    └──────────────────┴──────────────────────────────────────────────────┘
    建议把"前 N 大情景损失"放进每日风险报告，并设置限额（如任一历史情景损失 ≤ 15%）。
    """)


# ==================== 主函数 ====================


def main():
    returns = load_etf_returns()
    if is_synthetic(returns):
        print("⚠️ 模拟收益没有真实的危机事件，历史情景结果没有意义，只用于跑通代码\n")
    scenarios = historical_scenarios(returns)
    crisis_mask = crisis_correlation(returns, scenarios)
    _, x_p, F_normal = factor_shocks(returns, crisis_mask)
    reverse_stress(x_p, F_normal)
    liquidity_stress()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用第 03 课的 ERC 组合、第 01 课的多头最大 Sharpe 组合做同样的历史情景测试，哪个组合最"抗压"？
    2. 把第三部分的因子暴露改为用"危机区间"回归得到的 β，组合的情景损失怎么变？
    3. 在反向压力测试中加入约束"国债不能跌超过 3%"，用 cvxpy 求解最可能的情景
    4. 用 Phase 7 的个股多因子组合，计算它在第一部分"2024 年初"情景区间的损失，并分解到市值因子
    """)


if __name__ == "__main__":
    main()
