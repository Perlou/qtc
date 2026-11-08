"""
Barra 风格因子与风险模型
========================

学习目标：
    1. 理解结构化风险模型：用少数共同因子解释股票之间的相关性
    2. 构造风格因子暴露（Size、Beta、Momentum、ResVol、NLSize、BP、Liquidity、EarningsYield）
    3. 每天用 WLS 做截面回归，求因子收益和特质收益
    4. 组装 V = XFXᵀ + Δ，验证预测风险，并做组合风险分解

核心概念：
    - 收益分解：r_i = Σ X_i,k·f_k + u_i；X 已知（暴露），f 每天回归得到（因子收益）
    - 截面回归：WLS，权重为市值平方根；国家因子 + 行业因子 + 风格因子
    - 行业约束：行业因子收益的市值加权和为 0，解决国家因子与行业哑变量的共线性
    - 协方差：V = XFXᵀ + Δ，F 是 K×K 因子协方差，Δ 是特质方差对角阵
    - 偏差统计量（Bias Statistic）：标准化收益的标准差，≈1 说明风险预测准确

前置知识：
    - 04-factor-preprocessing.py、09-factor-combination.py
    - Phase 3 的 06-linear-regression-ols.py（加权最小二乘）

环境要求：
    - pip install pandas numpy matplotlib
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）

⚠️ 这是教学简化版：生产级模型还有 Newey-West 调整、特征因子风险调整、
   波动率区间调整、特质风险贝叶斯收缩等工程细节，且需要全市场数据。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from factor_data import load_panel
from factor_tools import build_factors, preprocess_all, rebalance_dates, tradable_mask, tradable_status
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

STYLES = ["Size", "Beta", "Momentum", "ResVol", "NLSize", "BP", "Liquidity", "EarningsYield"]
HALF_LIFE = 90  # EWMA 半衰期（交易日）
WINDOW = 252  # 估计协方差的窗口（交易日）


# ==================== 第一部分：结构化风险模型 ====================


def theory(panel):
    n, k = len(panel.codes), 1 + panel.info["sector"].nunique() + len(STYLES)
    print("=" * 60)
    print("第一部分：为什么需要结构化风险模型")
    print("=" * 60)
    print(f"""
    Alpha 因子预测收益；构建组合还要知道风险：两只股票会不会一起跌？组合波动来自行业还是市值？

    直接估计协方差矩阵的参数个数：
       本股票池 {n} 只：{n * (n + 1) // 2} 个；全市场 3000 只：{3000 * 3001 // 2} 个 —— 样本远远不够
    结构化模型：
       r = X f + u    →    V = X F Xᵀ + Δ
       本模型 K = {k} 个因子：F 有 {k * (k + 1) // 2} 个参数，Δ 有 {n} 个，共 {k * (k + 1) // 2 + n} 个

    🎓 比喻：与其记住全班同学两两之间的关系，不如记住每个人属于哪个兴趣小组、性格内向还是外向。

    ┌────────────────┬────────────────────────────────────────────┐
    │ 风格因子       │ 本课的简化定义                             │
    ├────────────────┼────────────────────────────────────────────┤
    │ Size           │ ln(总市值)                                 │
    │ Beta           │ 过去 250 日对沪深 300 的回归系数           │
    │ Momentum       │ 12-1 月收益                                │
    │ ResVol         │ 过去 60 日（收益 − Beta × 市场）的标准差   │
    │ NLSize         │ Size³ 对 Size 回归的残差（刻画中盘股）     │
    │ BP             │ 净资产 / 总市值                            │
    │ Liquidity      │ ln(过去 20 日平均换手率)                   │
    │ EarningsYield  │ 净利润 TTM / 总市值                        │
    └────────────────┴────────────────────────────────────────────┘
    Barra CNE5 还有 Growth、Leverage 两个风格因子（需要更多财务数据），这里省略。
    """)


# ==================== 第二部分：风格暴露 ====================


def standardize(x: pd.Series, w: pd.Series) -> pd.Series:
    """Barra 口径：市值加权均值为 0、等权标准差为 1，±3 截断，缺失值填 0（即市场平均暴露）"""
    def once(s):
        return (s - (s * w).sum() / w[s.notna()].sum()) / s.std()

    return once(once(x).clip(-3, 3)).fillna(0.0)


def style_exposures(panel, dates) -> dict:
    """返回 {调仓日: DataFrame(股票 × 风格)}。只用 t 日及之前的数据"""
    ret, mkt = panel.daily_ret, panel.bench_ret("hs300")
    beta = ret.rolling(250, min_periods=120).cov(mkt).div(mkt.rolling(250, min_periods=120).var(), axis=0)
    resid = ret - beta.shift(1).mul(mkt, axis=0)
    resvol = resid.rolling(60, min_periods=40).std()
    turn = panel.turnover.rolling(20, min_periods=10).mean()
    raw = build_factors(panel, dates)
    status = tradable_status(panel, dates)
    universe = status["listed"] & status["not_st"]

    out = {}
    for d in dates:
        codes = universe.columns[universe.loc[d] & panel.total_mcap.loc[d].notna()]
        w = panel.mcap.loc[d, codes]
        size = standardize(np.log(panel.total_mcap.loc[d, codes]), w)
        slope = np.polyfit(size, size**3, 1)
        nl = size**3 - np.polyval(slope, size)
        out[d] = pd.DataFrame({
            "Size": size,
            "Beta": standardize(beta.loc[d, codes], w),
            "Momentum": standardize(raw["mom_12_1"].loc[d, codes], w),
            "ResVol": standardize(resvol.loc[d, codes], w),
            "NLSize": standardize(nl, w),
            "BP": standardize(raw["bp"].loc[d, codes], w),
            "Liquidity": standardize(np.log(turn.loc[d, codes]), w),
            "EarningsYield": standardize(raw["ep"].loc[d, codes], w),
        })[STYLES]
    return out


def exposure_overview(panel, expo):
    print("\n" + "=" * 60)
    print("第二部分：风格因子暴露")
    print("=" * 60)
    d = list(expo)[-1]
    X = expo[d]
    w = panel.mcap.loc[d, X.index]
    print(f"📋 {d.date()} 暴露检查（{len(X)} 只股票）：")
    print(pd.DataFrame({"市值加权均值": (X.mul(w, axis=0)).sum() / w.sum(), "等权标准差": X.std(),
                        "最小": X.min(), "最大": X.max()}).T.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\n📋 风格暴露之间的截面相关系数：")
    print(X.corr().round(2).to_string())
    print("""
    💡 市值加权均值 ≈ 0 意味着"市值加权的市场组合"对每个风格的暴露都是 0，
       这样风格因子收益才是"相对市场"的收益。ResVol 与 Beta、Liquidity 常常相关，
       Barra 实际做法是对它们再做正交化，这里保留原值。
    """)


# ==================== 第三部分：截面回归 ====================


def cross_section_regressions(panel, expo):
    print("\n" + "=" * 60)
    print("第三部分：WLS 截面回归求因子收益")
    print("=" * 60)
    print("""
    每个交易日：r_i = f_c + Σ D_i,k·f_k + Σ X_i,s·f_s + u_i，权重 = √流通市值

       - 暴露用上一个月末的值（只用过去信息），收益用当天的 → 不存在前视
       - 国家因子 f_c（所有股票暴露为 1）与全部行业哑变量共线，加约束 Σ w_k·f_k = 0（w_k 为行业市值占比）
       - 实现：把最后一个行业的因子收益用其余行业表示，f_ind = C·g，对 g 做无约束 WLS，再还原
       - 市值平方根加权：大公司特质波动较小、估计更可靠，又不至于被超大市值股票主导
    """)
    sectors = sorted(panel.info["sector"].unique())
    cols = ["Country", *sectors, *STYLES]
    dates = list(expo)
    ret = panel.daily_ret
    fr, resid, r2 = {}, {}, {}
    for k, d0 in enumerate(dates):
        end = dates[k + 1] if k + 1 < len(dates) else None
        days = ret.index[(ret.index > d0) & ((ret.index <= end) if end is not None else (ret.index <= d0))]
        X = expo[d0]
        D = pd.get_dummies(panel.info.loc[X.index, "sector"], dtype=float).reindex(columns=sectors, fill_value=0.0)
        mc = panel.mcap.loc[d0, X.index]
        cap = (D.mul(mc, axis=0)).sum() / mc.sum()
        C = np.vstack([np.eye(len(sectors) - 1), -(cap.iloc[:-1] / cap.iloc[-1]).to_numpy()[None, :]])
        Z = np.column_stack([np.ones(len(X)), D.to_numpy() @ C, X.to_numpy()])
        sw = np.sqrt(np.sqrt(mc.to_numpy()))  # lstsq 对 √权重 缩放，权重本身 = √市值
        for day in days:
            r = ret.loc[day, X.index].to_numpy()
            ok = np.isfinite(r)
            if ok.sum() < 60:
                continue
            g, *_ = np.linalg.lstsq(Z[ok] * sw[ok, None], r[ok] * sw[ok], rcond=None)
            u = r[ok] - Z[ok] @ g
            f = np.r_[g[0], C @ g[1 : len(sectors)], g[len(sectors):]]
            fr[day] = f
            resid[day] = pd.Series(u, index=X.index[ok])
            w = sw[ok] ** 2
            rbar = (w * r[ok]).sum() / w.sum()
            r2[day] = 1 - (w * u**2).sum() / (w * (r[ok] - rbar) ** 2).sum()
    fr = pd.DataFrame(fr, index=cols).T
    resid = pd.DataFrame(resid).T.reindex(columns=panel.codes)
    r2 = pd.Series(r2)

    style = fr[STYLES]
    stats = pd.DataFrame({"年化收益": style.mean() * 252, "年化波动": style.std() * np.sqrt(252),
                          "t值": style.mean() / style.std() * np.sqrt(len(style))})
    print(f"📋 风格因子收益（{fr.index[0].date()} ~ {fr.index[-1].date()}，{len(fr)} 个交易日）：")
    print(stats.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"""
    截面回归加权 R²：均值 {r2.mean():.1%}，中位数 {r2.median():.1%}
    国家因子年化收益 {fr['Country'].mean() * 252:+.1%}、波动 {fr['Country'].std() * np.sqrt(252):.1%}（代表股票池的市场整体）
    行业因子年化波动中位数 {fr[sectors].std().median() * np.sqrt(252):.1%}，风格因子年化波动中位数 {stats['年化波动'].median():.1%}

    💡 这里的因子收益是"纯因子收益"：控制了其他所有因子之后，单位暴露对应的收益。
       Size 为负表示大市值跑输小市值（暴露是 ln 市值，小盘股暴露为负）。
    """)
    return fr, resid, sectors


# ==================== 第四部分：V = XFXᵀ + Δ 与风险预测检验 ====================


def ewma_weights(n: int) -> np.ndarray:
    w = 0.5 ** (np.arange(n)[::-1] / HALF_LIFE)
    return w / w.sum()


def risk_model(fr: pd.DataFrame, resid: pd.DataFrame, d: pd.Timestamp):
    """用 d 日及之前的数据估计 F 与特质方差（EWMA）"""
    win = fr.loc[:d].tail(WINDOW)
    w = ewma_weights(len(win))
    x = win.to_numpy() - (w[:, None] * win.to_numpy()).sum(axis=0)
    F = pd.DataFrame((x * w[:, None]).T @ x, index=win.columns, columns=win.columns)

    u2 = resid.loc[win.index] ** 2
    wu = pd.DataFrame(np.where(u2.notna(), w[:, None], 0.0), index=u2.index, columns=u2.columns)
    spec = (u2.fillna(0.0) * wu).sum() / wu.sum().where(u2.notna().sum() >= 60)
    return F, spec


def exposure_matrix(panel, expo_d: pd.DataFrame, sectors: list) -> pd.DataFrame:
    D = pd.get_dummies(panel.info.loc[expo_d.index, "sector"], dtype=float).reindex(columns=sectors, fill_value=0.0)
    return pd.concat([pd.Series(1.0, index=expo_d.index, name="Country"), D, expo_d], axis=1)


def validate_risk(panel, expo, fr, resid, sectors):
    print("\n" + "=" * 60)
    print("第四部分：组装 V = XFXᵀ + Δ，检验风险预测")
    print("=" * 60)

    dates = [d for d in expo if d >= pd.Timestamp("2021-01-01")]
    ret = panel.daily_ret
    rng = np.random.default_rng(7)
    rows = []
    for k, d in enumerate(dates[:-1]):
        F, spec = risk_model(fr, resid, d)
        X = exposure_matrix(panel, expo[d], sectors)
        hist = ret.loc[:d].tail(WINDOW)
        codes = X.index[spec.reindex(X.index).notna() & (hist[X.index].notna().sum() >= 200)]
        Xc, s = X.loc[codes], spec[codes]
        V = Xc.to_numpy() @ F.to_numpy() @ Xc.to_numpy().T + np.diag(s.to_numpy())
        S = hist[codes].fillna(0.0).cov().to_numpy()
        fwd = ret.loc[d:dates[k + 1]].iloc[1:][codes].fillna(0.0)
        n_days = len(fwd)
        n = len(codes)

        one = np.ones(n)
        small = pd.Series(-expo[d].loc[codes, "Size"]).rank(ascending=False) <= 30
        rand = np.zeros(n)
        rand[rng.choice(n, 30, replace=False)] = 1 / 30
        mv_f = np.linalg.solve(V, one)
        mv_s = np.linalg.solve(S, one)
        ports = {"等权股票池": (one / n, V), "小市值 30 只": (small.to_numpy() / small.sum(), V), "随机 30 只": (rand, V),
                 "最小方差（因子模型 V）": (mv_f / mv_f.sum(), V), "最小方差（样本协方差）": (mv_s / mv_s.sum(), S)}
        for name, (h, cov) in ports.items():
            sigma = np.sqrt(h @ cov @ h * n_days)
            rows.append({"组合": name, "date": d, "预测": sigma, "实现": float((fwd.to_numpy() @ h).sum())})
    df = pd.DataFrame(rows)
    df["z"] = df["实现"] / df["预测"]
    res = df.groupby("组合", sort=False).agg(预测年化波动=("预测", lambda s: s.mean() * np.sqrt(12)),
                                          实现年化波动=("实现", lambda s: s.std() * np.sqrt(12)), 偏差统计量=("z", "std"))
    print(f"📋 样本外检验（{len(dates) - 1} 个月，每月末预测下个月波动）：")
    print(res.to_string(float_format=lambda v: f"{v:.3f}"))
    print("""
    偏差统计量 = std(实现收益 / 预测波动)：≈ 1 准确，> 1 低估风险，< 1 高估风险

    💡 重点看最后两行。最小方差组合专门"寻找协方差矩阵里最小的方向"：
       样本协方差的估计误差会被它放大，预测波动往往明显低于实现波动；
       结构化模型参数少得多，这种"优化器幻觉"更轻。这正是 Phase 8 组合优化要用风险模型的原因。
       每个组合只有几十个月的样本，偏差统计量本身也有不小的误差。
    """)
    return df


# ==================== 第五部分：组合风险分解 ====================


def risk_decomposition(panel, expo, fr, resid, sectors):
    print("\n" + "=" * 60)
    print("第五部分：组合风险分解")
    print("=" * 60)

    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    clean = preprocess_all(build_factors(panel, dates), panel, mask)
    d = list(expo)[-1]
    score = pd.DataFrame({k: v.loc[d] for k, v in clean.items()}).mean(axis=1).dropna()
    F, spec = risk_model(fr, resid, d)
    X = exposure_matrix(panel, expo[d], sectors)
    codes = X.index.intersection(score.index).intersection(spec.dropna().index)
    X, spec = X.loc[codes], spec[codes]
    top = score[codes].nlargest(max(len(codes) // 5, 10)).index
    h = pd.Series(0.0, index=codes)
    h[top] = 1 / len(top)
    b = pd.Series(1 / len(codes), index=codes)
    a = h - b  # 主动权重：相对等权基准

    x = X.T @ a  # 主动暴露
    Fx = F.to_numpy() @ x.to_numpy()
    contrib = pd.Series(x.to_numpy() * Fx, index=x.index) * 252
    spec_var = (a**2 * spec).sum() * 252
    total = contrib.sum() + spec_var
    groups = {"行业": contrib[sectors].sum(), "风格": contrib[STYLES].sum(), "特质": spec_var}
    print(f"""
    组合：{d.date()} 等权合成 8 个因子，买入得分最高的 {len(top)} 只，等权；基准：等权股票池

    主动风险（跟踪误差）预测：年化 {np.sqrt(total):.2%}
       风险来源：""" + "，".join(f"{k} {v / total:.0%}" for k, v in groups.items()))
    print("\n📋 主动风格暴露与风险贡献：")
    tab = pd.DataFrame({"主动暴露": x[STYLES], "占主动方差": contrib[STYLES] / total})
    print(tab.to_string(float_format=lambda v: f"{v:+.3f}"))
    biggest = tab["主动暴露"].abs().idxmax()
    print(f"""
    最大的主动风格暴露：{biggest}（{tab.loc[biggest, '主动暴露']:+.2f}）
    💡 风险贡献 = x_k·(F x)_k：可以为负，表示该暴露对冲了其他风险。
       即使单因子都做了市值中性化，选出来的组合仍可能有明显风格偏离 —— 用风险模型
       看清楚"这份超额收益到底押注了什么"，才能在 Phase 8 里用约束把它控制住。
    """)


# ==================== 主函数 ====================


def plot_results(fr: pd.DataFrame, df: pd.DataFrame):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for c in STYLES:
        axes[0].plot(fr.index, fr[c].cumsum(), label=c)
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_title("风格因子累计收益（纯因子收益，算术累加）")
    axes[0].legend(fontsize=8, ncol=2)
    res = df.groupby("组合", sort=False).agg(pred=("预测", lambda s: s.mean() * np.sqrt(12)),
                                          real=("实现", lambda s: s.std() * np.sqrt(12)))
    pos = np.arange(len(res))
    axes[1].barh(pos - 0.2, res["pred"], height=0.4, label="预测年化波动")
    axes[1].barh(pos + 0.2, res["real"], height=0.4, label="实现年化波动")
    axes[1].set_yticks(pos, res.index)
    axes[1].set_title("风险预测 vs 实现")
    axes[1].legend()
    path = save_figure(fig, "10_barra_risk_model.png", subdir="phase-7")
    print(f"📈 图表已保存：{path}")


def main():
    panel = load_panel()
    theory(panel)
    dates = rebalance_dates(panel)
    expo = style_exposures(panel, dates)
    exposure_overview(panel, expo)
    fr, resid, sectors = cross_section_regressions(panel, expo)
    df = validate_risk(panel, expo, fr, resid, sectors)
    risk_decomposition(panel, expo, fr, resid, sectors)
    plot_results(fr, df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 EWMA 半衰期改为 42 和 252，比较偏差统计量；思考"反应快"和"估计稳"之间的取舍
    2. 增加 Leverage 因子（资产负债率，需要下载资产负债表），比较截面回归 R² 的变化
    3. 因子纯化：把 lowturn 因子对 8 个风格暴露做截面回归，取残差计算 Rank IC，看是否有增量信息
    4. 对第五部分的组合加上"Size 主动暴露绝对值 ≤ 0.3"的约束，手动调整持仓，比较主动风险
    """)


if __name__ == "__main__":
    main()
