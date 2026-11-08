"""
CAPM 与 Beta
============

学习目标：
    1. 理解 CAPM 的核心思想：只有"无法分散的系统性风险"才能获得补偿
    2. 用时间序列回归估计个股的 Alpha、Beta，并读懂 R²、t 值
    3. 认识 Beta 的不稳定性，以及 Blume 收缩为什么有用
    4. 用真实面板检验证券市场线（SML），观察"低 Beta 异象"

核心概念：
    - Beta = Cov(R_i, R_m) / Var(R_m)：个股对市场的敏感度
    - Alpha：CAPM 无法解释的超额收益，总是"相对于某个模型"而言
    - 系统性风险 vs 特质风险：R² 衡量收益波动中有多少来自市场
    - 证券市场线（SML）：CAPM 预测的"Beta → 期望收益"直线

前置知识：
    - Phase 3 的 06-linear-regression-ols.py（OLS、t 值、Newey-West）
    - Phase 2 的 08-benchmark-relative.py

环境要求：
    - pip install pandas numpy statsmodels matplotlib akshare
    - 数据来自本阶段共享面板 factor_data.py（首次运行会下载并缓存，约 2–4 分钟）
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

from factor_data import RF_ANNUAL, load_panel
from factor_tools import bench_period_returns, period_returns, rebalance_dates, tradable_mask
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "2019-01-01", "2024-12-31"
RF_DAILY = RF_ANNUAL / 252


# ==================== 第一部分：CAPM 的直觉 ====================


def capm_intuition():
    print("=" * 60)
    print("第一部分：CAPM 的直觉")
    print("=" * 60)
    print(f"""
    CAPM：E[R_i] − R_f = β_i × (E[R_m] − R_f)

    个股风险 = 系统性风险（跟着大盘波动，无法分散）+ 特质风险（公司自身事件，可分散）
    既然特质风险可以免费分散掉，市场就只为系统性风险支付补偿。

    🚢 比喻：船整体的颠簸谁都躲不掉 → 船票里含"颠簸补贴"；
            座位旁边有人打呼噜，换个座位就行 → 没人会为此给你钱。

    检验方式：时间序列回归
        R_i,t − R_f,t = α_i + β_i × (R_m,t − R_f,t) + ε_i,t

    ┌────────┬───────────────────────────────────────────────┐
    │ 参数   │ 含义                                          │
    ├────────┼───────────────────────────────────────────────┤
    │ β      │ 大盘涨 1%，个股平均涨 β%                      │
    │ α      │ CAPM 解释不了的平均超额收益（CAPM 成立时 = 0）│
    │ R²     │ 收益方差中由市场解释的比例 = 系统性风险占比   │
    │ ε      │ 特质收益，其波动 = 特质风险                   │
    └────────┴───────────────────────────────────────────────┘

    本课约定：市场组合 = 沪深 300，无风险利率 = 年化 {RF_ANNUAL:.0%}（示例值，按 252 天折算为日频）
    """)


# ==================== 第二部分：个股 CAPM 回归 ====================


def capm_regression(stock_ret: pd.Series, mkt_ret: pd.Series) -> dict:
    """日收益回归，返回年化 alpha、beta、t 值、R²（Newey-West 标准误）"""
    data = pd.concat([stock_ret - RF_DAILY, mkt_ret - RF_DAILY], axis=1, keys=["y", "x"]).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "beta": res.params["x"],
        "alpha_ann": res.params["const"] * 252,
        "t_alpha": res.tvalues["const"],
        "R2": res.rsquared,
        "总波动": data["y"].std() * np.sqrt(252),
        "特质波动": res.resid.std() * np.sqrt(252),
        "天数": len(data),
    }


def pick_examples(panel, n: int = 6) -> list[str]:
    """每个行业挑市值最大的一只，取前 n 个行业，保证示例覆盖不同类型的公司"""
    mc = panel.mcap.loc[END:].iloc[0] if len(panel.mcap.loc[END:]) else panel.mcap.iloc[-1]
    df = pd.DataFrame({"mc": mc, "sector": panel.info["sector"]}).dropna()
    top = df.sort_values("mc", ascending=False).groupby("sector").head(1)
    return list(top.index[:n])


def single_stock_capm(panel):
    print("\n" + "=" * 60)
    print("第二部分：个股 CAPM 回归（2019–2024 日收益）")
    print("=" * 60)

    ret = panel.daily_ret.loc[START:END]
    mkt = panel.bench_ret("hs300").loc[START:END]
    rows = []
    for c in pick_examples(panel):
        r = capm_regression(ret[c], mkt)
        rows.append({"代码": c, "名称": panel.info.loc[c, "name"], "行业": panel.info.loc[c, "sector"], **r})
    df = pd.DataFrame(rows).set_index("代码").rename(
        columns={"beta": "Beta", "alpha_ann": "年化α", "t_alpha": "t(α)", "R2": "R²"})
    print("\n📋 各行业市值最大的股票：")
    print(df.to_string(formatters={
        "Beta": "{:.2f}".format, "年化α": "{:+.1%}".format, "t(α)": "{:+.2f}".format,
        "R²": "{:.2f}".format, "总波动": "{:.1%}".format, "特质波动": "{:.1%}".format,
    }))

    # 全部股票的分布
    allr = pd.DataFrame({c: capm_regression(ret[c], mkt) for c in panel.codes if ret[c].notna().sum() > 250}).T
    print(f"""
    📊 全部 {len(allr)} 只股票：
       Beta 中位数 {allr['beta'].median():.2f}（范围 {allr['beta'].min():.2f} ~ {allr['beta'].max():.2f}）
       R² 中位数 {allr['R2'].median():.2f} → 典型个股只有这么多比例的方差来自市场，其余是特质风险
       |t(α)| > 2 的股票：{(allr['t_alpha'].abs() > 2).sum()} 只（{(allr['t_alpha'].abs() > 2).mean():.0%}）

    💡 解读：
       - 特质波动通常比"市场部分"还大，但特质风险可以通过持有多只股票分散
       - 单只股票的 α 即使看起来很大，t 值往往不显著：6 年日数据也不足以"证明" α
       - 如果完全随机，约 5% 的股票会因为运气出现 |t| > 2 —— 这就是多重检验问题
    """)
    print("   各行业 Beta 中位数：")
    by_sec = allr.join(panel.info["sector"]).groupby("sector")["beta"].median().sort_values()
    print("   " + "，".join(f"{k} {v:.2f}" for k, v in by_sec.items()))
    return allr


# ==================== 第三部分：Beta 不稳定 ====================


def rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 250) -> pd.Series:
    cov = stock_ret.rolling(window, min_periods=window // 2).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=window // 2).var()
    return cov / var


def beta_instability(panel):
    print("\n" + "=" * 60)
    print("第三部分：Beta 不稳定与 Blume 收缩")
    print("=" * 60)

    ret = panel.daily_ret
    mkt = panel.bench_ret("hs300")
    first, second = ("2019-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31")

    def beta_in(span):
        r, m = ret.loc[span[0]:span[1]], mkt.loc[span[0]:span[1]]
        return pd.Series({c: capm_regression(r[c], m)["beta"] for c in panel.codes if r[c].notna().sum() > 250})

    b1, b2 = beta_in(first), beta_in(second)
    common = b1.index.intersection(b2.index)
    b1, b2 = b1[common], b2[common]
    blume = 0.67 * b1 + 0.33 * 1.0
    mse_raw = ((b1 - b2) ** 2).mean()
    mse_blume = ((blume - b2) ** 2).mean()
    print(f"""
    用 2019–2021 的 Beta 预测 2022–2024 的 Beta（{len(common)} 只股票）：
       两段 Beta 的截面相关系数：{b1.corr(b2):.2f}
       前一段 Beta 的标准差 {b1.std():.2f}，后一段 {b2.std():.2f}
       直接用历史 Beta 预测的均方误差：      {mse_raw:.4f}
       Blume 收缩 0.67β + 0.33 后的均方误差：{mse_blume:.4f}

    💡 Beta 有"向 1 回归"的倾向：极端 Beta 往往一部分来自估计误差。
       Blume（1971）的经验公式把历史 Beta 向 1 收缩，Barra 等风险模型也会对 Beta 做类似的收缩。
       （收缩是否改善预测取决于样本，上面打印的是本面板的实际结果）
    """)

    rb = pd.DataFrame({c: rolling_beta(ret[c], mkt) for c in [b1.idxmax(), b1.idxmin()]}).loc[START:END]
    for c in rb.columns:
        s = rb[c].dropna()
        print(f"   {c} {panel.info.loc[c, 'name']}：滚动 250 日 Beta 在 {s.min():.2f} ~ {s.max():.2f} 之间变化")
    return rb


# ==================== 第四部分：证券市场线与低 Beta 异象 ====================


def security_market_line(panel):
    print("\n" + "=" * 60)
    print("第四部分：证券市场线（SML）与低 Beta 异象")
    print("=" * 60)

    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    ret_p = period_returns(panel, dates)
    mkt_p = bench_period_returns(panel, dates, "hs300")
    rf_p = RF_ANNUAL / 12

    # 每个月末用过去 250 个交易日估计 Beta（只用过去数据）
    ret, mkt = panel.daily_ret, panel.bench_ret("hs300")
    cov = ret.rolling(250, min_periods=200).cov(mkt)
    pre_beta = cov.div(mkt.rolling(250, min_periods=200).var(), axis=0).reindex(dates).where(mask)

    groups, betas = {}, {}
    for d in dates:
        b, r = pre_beta.loc[d], ret_p.loc[d]
        ok = b.notna() & r.notna()
        if ok.sum() < 50:
            continue
        g = pd.qcut(b[ok].rank(method="first"), 5, labels=range(1, 6)).astype(int)
        groups[d] = r[ok].groupby(g).mean()
        betas[d] = b[ok].groupby(g).mean()
    grp, pb = pd.DataFrame(groups).T, pd.DataFrame(betas).T

    ex_mkt = (mkt_p - rf_p).reindex(grp.index)
    rows = []
    for k in grp.columns:
        ex = grp[k] - rf_p
        res = sm.OLS(ex, sm.add_constant(ex_mkt), missing="drop").fit()
        beta_real = res.params.iloc[1]
        rows.append({
            "组": f"Q{k}", "事前Beta": pb[k].mean(), "事后Beta": beta_real,
            "实际年化超额": ex.mean() * 12, "CAPM预测年化超额": beta_real * ex_mkt.mean() * 12,
            "年化Alpha": res.params.iloc[0] * 12, "t(Alpha)": res.tvalues.iloc[0],
        })
    df = pd.DataFrame(rows).set_index("组")
    print(f"\n📋 按事前 Beta 分 5 组（月度调仓，{len(grp)} 期，Q1 = Beta 最低）：")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    slope = np.polyfit(df["事后Beta"], df["实际年化超额"], 1)[0]
    capm_slope = ex_mkt.mean() * 12
    flatter = "比 CAPM 预测更平坦" if slope < capm_slope else "比 CAPM 预测更陡峭"
    ew = ret_p.where(mask).mean(axis=1)
    bias_note = f"""
    ⚠️ 注意各组 Alpha 的整体水平：股票池等权组合 {ew.mean() * 12:+.2%}，沪深 300 {mkt_p.mean() * 12:+.2%}（月均收益 × 12）
       {"所有分组的 Alpha 都为正，" if (df['年化Alpha'] > 0).all() else ""}这里至少混入了两个与"选股能力"无关的来源：
       1. 幸存者偏差：股票池是"今天的"沪深 300 / 中证 500 成分股 —— 能留到今天的公司，
          过去几年大多表现不错（03 课详细讨论）
       2. 基准口径：沪深 300 是价格指数，不含分红；个股收益用的是复权价（含分红）
       所以这里的 Alpha 绝对水平不可信，只能比较组与组之间的相对差异。
    """ if not panel.is_synthetic else ""
    print(f"""
    CAPM 预测：组合超额收益 = 事后 Beta × 市场超额收益（一条过原点、斜率为市场超额收益的直线）
       CAPM 斜率（沪深 300 年化超额，相对 {RF_ANNUAL:.0%} 无风险利率）：{capm_slope:+.2%}
       实际拟合的 "Beta → 收益" 斜率：{slope:+.2%}（{flatter}）
       Q5 − Q1 的年化 Alpha 差：{df['年化Alpha'].iloc[-1] - df['年化Alpha'].iloc[0]:+.2%}
{bias_note}
    💡 低 Beta 异象：实证中 SML 往往比 CAPM 预测平坦 —— 高 Beta 股票的风险调整后收益偏低
       （Frazzini & Pedersen 的 Betting Against Beta）。行为解释：杠杆受限的投资者通过买高 Beta
       股票获得"杠杆"，推高了它们的价格。本样本只有 6 年、{len(panel.codes)} 只股票，结论需要更大样本检验。
    """)
    return df, ex_mkt


# ==================== 主函数 ====================


def plot_results(rb: pd.DataFrame, sml: pd.DataFrame, ex_mkt: pd.Series, panel):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in rb.columns:
        axes[0].plot(rb.index, rb[c], label=f"{c} {panel.info.loc[c, 'name']}")
    axes[0].axhline(1, color="gray", ls="--", lw=0.8)
    axes[0].set_title("滚动 250 日 Beta（对沪深 300）")
    axes[0].legend()

    axes[1].scatter(sml["事后Beta"], sml["实际年化超额"], s=60, label="实际（5 个 Beta 分组）")
    for name, row in sml.iterrows():
        axes[1].annotate(name, (row["事后Beta"], row["实际年化超额"]), xytext=(4, 4), textcoords="offset points")
    xs = np.linspace(0, sml["事后Beta"].max() * 1.1, 10)
    axes[1].plot(xs, xs * ex_mkt.mean() * 12, "r--", label="CAPM 预测的 SML")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xlabel("事后 Beta")
    axes[1].set_ylabel("年化超额收益")
    axes[1].set_title("证券市场线")
    axes[1].legend()
    path = save_figure(fig, "01_capm_beta.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    capm_intuition()
    single_stock_capm(panel)
    rb = beta_instability(panel)
    sml, ex_mkt = security_market_line(panel)
    plot_results(rb, sml, ex_mkt, panel)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把市场组合换成中证 500（panel.bench_ret("zz500")），比较 Beta 和 Alpha 的变化，
       体会"Alpha 总是相对于某个基准而言"
    2. 改用周收益回归估计 Beta，与日收益结果比较。思考：低流动性股票的日收益
       为什么会低估 Beta？（提示：价格对市场信息的反应有滞后，可查 Dimson Beta）
    3. 实现 Vasicek 收缩：β_shrunk = w·β̂ + (1−w)·β̄，w 由估计方差和截面方差决定，
       与 Blume 收缩比较预测误差
    4. 把第四部分改为按 Beta 分 10 组，并剔除金融行业，观察 SML 斜率是否稳定
    """)


if __name__ == "__main__":
    main()
