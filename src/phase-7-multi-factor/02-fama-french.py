"""
Fama-French 三因子 / 五因子
===========================

学习目标：
    1. 理解从 CAPM 到多因子模型的演进：规模效应、价值效应
    2. 亲手用面板数据做 2×3 排序，构造 SMB、HML（以及 A 股常用的 EP 版价值因子、RMW）
    3. 做时间序列回归，比较 CAPM 与三因子模型的解释力（R²、Alpha）
    4. 用因子模型评估一个"选股策略"：它的 Alpha 是真本事，还是只是风格暴露？

核心概念：
    - SMB（Small Minus Big）：小市值组合 − 大市值组合
    - HML（High Minus Low）：高账面市值比（价值）− 低账面市值比（成长）
    - 2×3 排序：市值按中位数分两组 × 估值按 30%/70% 分三组，组内市值加权
    - 因子载荷（Loading）：资产收益对因子收益的回归系数
    - CH-3（Liu, Stambaugh, Yuan 2019）：A 股版本，剔除最小 30% 市值，用 EP 构造价值因子

前置知识：
    - 01-capm-beta.py

环境要求：
    - pip install pandas numpy statsmodels matplotlib
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
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
from factor_tools import (bench_period_returns, build_factors, period_returns, quantile_labels,
                          rebalance_dates, tradable_mask)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

RF_M = RF_ANNUAL / 12  # 月度无风险利率（示例值）


# ==================== 第一部分：从 CAPM 到三因子 ====================


def theory(panel):
    universe_note = ("股票池只有约 130 只沪深 300 / 中证 500 成分股" if not panel.is_synthetic
                     else f"股票池只有 {len(panel.codes)} 只（合成数据）")
    ch3_note = "本股票池本身已不含小微盘股，不再剔除" if not panel.is_synthetic else "这里不再剔除"
    print("=" * 60)
    print("第一部分：从 CAPM 到 Fama-French")
    print("=" * 60)
    print(f"""
    实证发现两类 CAPM 解释不了的现象：
       - 规模效应：小市值股票长期平均收益更高（Banz, 1981）
       - 价值效应：账面市值比（BM）高的"便宜"股票长期跑赢"贵"的股票

    三因子模型（Fama & French, 1993）：
       R_i − R_f = α_i + β_i·MKT + s_i·SMB + h_i·HML + ε_i
    五因子模型（2015）再加：RMW（高盈利 − 低盈利）、CMA（低投资 − 高投资）

    2×3 排序构造因子收益：
                     BM 低（G）   BM 中（N）   BM 高（V）
                  ┌────────────┬────────────┬────────────┐
      市值小（S）  │     SG     │     SN     │     SV     │
                  ├────────────┼────────────┼────────────┤
      市值大（B）  │     BG     │     BN     │     BV     │
                  └────────────┴────────────┴────────────┘
      SMB = (SG + SN + SV)/3 − (BG + BN + BV)/3   控制价值后的规模效应
      HML = (SV + BV)/2 − (SG + BG)/2             控制规模后的价值效应

    本课的简化（与学术版的差异）：
       - 学术版每年 6 月调仓、用上一年 12 月的 BM；这里每月调仓、用按公告日对齐的最新财报（PIT）
       - {universe_note}，每个格子只有一二十只股票，噪声较大
       - CH-3 要剔除市值最小的 30%（"壳价值"干扰）：{ch3_note}
    """)


# ==================== 第二部分：构造因子收益 ====================


def ff_one_period(df: pd.DataFrame, value_col: str) -> tuple[float, float, pd.Series]:
    """单期 2×3 排序。df 列：mcap、value_col、r（下期收益）。返回 (SMB, HML, 6 个组合收益)"""
    size = np.where(df["mcap"] <= df["mcap"].median(), "S", "B")
    q30, q70 = df[value_col].quantile([0.3, 0.7])
    value = np.select([df[value_col] <= q30, df[value_col] >= q70], ["G", "V"], "N")
    group = pd.Series(np.char.add(size.astype(str), value.astype(str)), index=df.index)
    w = df["mcap"]
    port = (df["r"] * w).groupby(group).sum() / w.groupby(group).sum()
    port = port.reindex(["SG", "SN", "SV", "BG", "BN", "BV"])
    smb = port[["SG", "SN", "SV"]].mean() - port[["BG", "BN", "BV"]].mean()
    hml = port[["SV", "BV"]].mean() - port[["SG", "BG"]].mean()
    return smb, hml, port


def build_ff_factors(panel):
    print("\n" + "=" * 60)
    print("第二部分：用面板构造 MKT、SMB、HML、VMG、RMW（月频）")
    print("=" * 60)

    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    ret = period_returns(panel, dates)
    raw = build_factors(panel, dates)
    tmc = panel.total_mcap.reindex(dates)

    rows, counts = {}, []
    for d in dates:
        df = pd.DataFrame({"mcap": tmc.loc[d], "bp": raw["bp"].loc[d], "ep": raw["ep"].loc[d],
                           "roe": raw["roe"].loc[d], "r": ret.loc[d]})[mask.loc[d]].dropna()
        if len(df) < 40:
            continue
        smb, hml, port = ff_one_period(df, "bp")
        _, vmg, _ = ff_one_period(df, "ep")  # CH-3 思路：用 EP 构造价值因子
        _, rmw, _ = ff_one_period(df, "roe")  # FF5 的盈利因子（这里用 ROE 近似）
        rows[d] = {"SMB": smb, "HML": hml, "VMG": vmg, "RMW": rmw}
        counts.append(len(df) / 6)
    fac = pd.DataFrame(rows).T
    fac.insert(0, "MKT", bench_period_returns(panel, fac.index, "hs300") - RF_M)

    summary = pd.DataFrame({
        "月均收益": fac.mean(),
        "年化(×12)": fac.mean() * 12,
        "月波动": fac.std(),
        "t值": fac.mean() / fac.std() * np.sqrt(len(fac)),
        "正收益月占比": (fac > 0).mean(),
    })
    print(f"\n📋 因子收益统计（{len(fac)} 个月，平均每个格子约 {np.mean(counts):.0f} 只股票）：")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n📋 因子相关系数：")
    print(fac.corr().round(2).to_string())
    print(f"""
    💡 解读：
       - |t| < 2 的因子在本样本中统计上不显著。6 年 × {len(panel.codes)} 只股票的样本太小，
         学术研究通常用全市场、几十年的数据
       - HML（BP）与 VMG（EP）都是"价值"，但口径不同，相关性见上表
       - 因子收益是"多空组合"收益：它告诉你承担这种风格暴露能拿到多少补偿
    """)
    return fac, dates, mask, ret, raw


# ==================== 第三部分：时间序列回归 ====================


def regress(y: pd.Series, X: pd.DataFrame, lags: int = 3):
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(data["y"], sm.add_constant(data[X.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def time_series_tests(panel, fac, mask, ret):
    print("\n" + "=" * 60)
    print("第三部分：时间序列回归 —— CAPM vs 三因子")
    print("=" * 60)

    sector = panel.info["sector"]
    test = {s: ret.where(mask).loc[:, sector.index[sector == s]].mean(axis=1) for s in sector.unique()}
    rows = []
    for s, r in test.items():
        ex = r.reindex(fac.index) - RF_M
        capm = regress(ex, fac[["MKT"]])
        ff3 = regress(ex, fac[["MKT", "SMB", "HML"]])
        rows.append({"行业组合": s, "CAPM α(年化)": capm.params["const"] * 12, "CAPM R²": capm.rsquared,
                     "FF3 α(年化)": ff3.params["const"] * 12, "t(α)": ff3.tvalues["const"],
                     "β_MKT": ff3.params["MKT"], "s_SMB": ff3.params["SMB"], "h_HML": ff3.params["HML"],
                     "FF3 R²": ff3.rsquared})
    df = pd.DataFrame(rows).set_index("行业组合").sort_values("h_HML")
    print("\n📋 11 个行业等权组合（测试资产）：")
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))

    r2 = []
    for c in panel.codes:
        ex = ret.where(mask)[c].reindex(fac.index) - RF_M
        if ex.notna().sum() >= 36:
            r2.append((regress(ex, fac[["MKT"]]).rsquared, regress(ex, fac[["MKT", "SMB", "HML"]]).rsquared))
    r2 = np.array(r2)
    order = df.index.tolist()
    n_pos = (df["FF3 α(年化)"] > 0).sum()
    alpha_note = ('' if panel.is_synthetic else
                  ' —— α 的绝对水平受幸存者偏差污染（今天的成分股），\n         不能当作"行业有超额收益"的证据')
    print(f"""
    📊 个股层面（{len(r2)} 只，月收益）：CAPM 平均 R² {r2[:, 0].mean():.2f} → 三因子平均 R² {r2[:, 1].mean():.2f}

    💡 读表要点：
       - h_HML 最高的行业：{"、".join(order[-3:][::-1])}（偏"价值"）；
         最低的行业：{"、".join(order[:3])}（偏"成长"）
       - s_SMB > 0 表示该组合相对偏小盘；载荷描述的是"风格"，不是好坏
       - 比较 CAPM R² 与 FF3 R²：上升的部分就是 SMB、HML 解释的、CAPM 漏掉的共同波动
       - {n_pos}/{len(df)} 个行业组合的 FF3 α 为正{alpha_note}
    """)


# ==================== 第四部分：用因子模型评估策略 ====================


def strategy_attribution(panel, fac, dates, mask, ret, raw):
    print("\n" + "=" * 60)
    print("第四部分：用因子模型评估一个选股策略")
    print("=" * 60)

    def top_quintile(factor: pd.DataFrame) -> pd.Series:
        out = {}
        for d in fac.index:
            f = factor.loc[d].where(mask.loc[d]).dropna()
            g = quantile_labels(f, 5)
            out[d] = ret.loc[d, g.index[g == 5]].mean()
        return pd.Series(out)

    strategies = {"低换手 Top20%": top_quintile(raw["lowturn"]), "小市值 Top20%": top_quintile(raw["size"]),
                  "高 EP Top20%": top_quintile(raw["ep"])}
    ew = ret.where(mask).mean(axis=1).reindex(fac.index)
    rows = []
    for name, r in strategies.items():
        ex = r - RF_M
        capm, ff3 = regress(ex, fac[["MKT"]]), regress(ex, fac[["MKT", "SMB", "HML"]])
        rel = regress(r - ew, fac[["SMB", "HML"]])
        rows.append({"策略": name, "CAPM α": capm.params["const"] * 12, "t": capm.tvalues["const"],
                     "FF3 α": ff3.params["const"] * 12, "t ": ff3.tvalues["const"],
                     "SMB 载荷": ff3.params["SMB"], "HML 载荷": ff3.params["HML"],
                     "相对等权池 α": rel.params["const"] * 12})
    df = pd.DataFrame(rows).set_index("策略")
    print("\n📋 策略月收益对因子回归（α 已年化，Newey-West t 值）：")
    print(df.to_string(float_format=lambda x: f"{x:+.3f}"))
    alpha_note = ('绝对 α 受幸存者偏差污染，"相对等权池 α"（策略 − 股票池等权）更有参考意义' if not panel.is_synthetic
                  else '"相对等权池 α"（策略 − 股票池等权）衡量的是相对股票池平均水平的超额')
    print(f"""
    💡 读表要点：
       - "小市值 Top20%" 的 SMB 载荷应该明显为正 —— 它本来就是在买小盘股
       - 加入 SMB、HML 后，α 的变化反映了有多少"超额收益"其实是风格暴露
       - {alpha_note}
       - 一个"新策略"如果 FF3 α 不显著、载荷很大，它本质上只是已知风格的组合

    📌 因子收益从哪里来：
    ┌────────┬────────────────────────────┬──────────────────────────────┐
    │ 因子   │ 风险补偿解释               │ 行为偏差解释                 │
    ├────────┼────────────────────────────┼──────────────────────────────┤
    │ 价值   │ 便宜的公司经营 / 破产风险高│ 投资者过度外推成长           │
    │ 规模   │ 流动性差、信息不透明、壳价值│ 关注度低，定价效率低         │
    │ 盈利   │ 部分可用风险解释           │ 对盈利持续性认识不足         │
    └────────┴────────────────────────────┴──────────────────────────────┘
    风险补偿型因子会在"坏时候"回撤；行为型因子在套利资金涌入后衰减（11 课）。
    """)
    return strategies


# ==================== 主函数 ====================


def plot_factors(fac: pd.DataFrame):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in fac.columns:
        ax.plot(fac.index, (1 + fac[c]).cumprod(), label=c)
    ax.axhline(1, color="gray", lw=0.6)
    ax.set_title("因子累计收益（2×3 排序，月度调仓）")
    ax.legend(ncol=5)
    path = save_figure(fig, "02_fama_french_factors.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    theory(panel)
    fac, dates, mask, ret, raw = build_ff_factors(panel)
    time_series_tests(panel, fac, mask, ret)
    strategy_attribution(panel, fac, dates, mask, ret, raw)
    plot_factors(fac)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 2×3 排序的组内市值加权改为等权，比较 SMB、HML 的均值和波动，思考差异来源
    2. 在 Kenneth French 数据库下载美国三因子，与本课 A 股因子比较均值与相关性
    3. 实现完整的 FF5：用"总资产同比增长率"构造 CMA（需要另外下载资产负债表），
       检验加入 RMW、CMA 后行业组合的 R² 是否继续上升
    4. 把一个你在 Phase 5 写过的择时策略收益对 MKT、SMB、HML 回归，看看它暴露了哪些风格
    """)


if __name__ == "__main__":
    main()
