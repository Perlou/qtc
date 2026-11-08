"""
模型漂移与再训练
================

学习目标：
    1. 区分协变量漂移、概念漂移、标签漂移，知道金融里最危险的是概念漂移
    2. 搭建监控：滚动 Rank IC（注意标签滞后）、原始特征的 PSI / KS 检验、预测与持仓的分布
    3. 比较再训练策略：固定模型、定期重训（扩展 / 滚动窗口）、触发式重训
    4. 理解"再训练频率和窗口长度本身就是超参数"，也要计入试验次数

核心概念：
    - PSI = Σ (a_b − e_b)·ln(a_b / e_b)：经验阈值 < 0.1 稳定、0.1–0.25 轻度、> 0.25 显著（信用评分经验）
    - KS 检验：两个样本经验分布函数的最大距离
    - 标签滞后：t 期的 IC 要到 t+2 期决策时才知道 → 监控天然慢一拍
    - ⚠️ 对截面 rank 后的特征算 PSI 毫无意义（rank 后永远是均匀分布）

前置知识：
    - 05-lightgbm-stock-prediction.py、06-feature-importance-shap.py

环境要求：
    - pip install pandas numpy scipy scikit-learn matplotlib（lightgbm 可选）
    - 数据来自 Phase 7 共享面板；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ks_2samp

import ml_tools as mt
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

MIN_TRAIN = 24
IC_WINDOW = 6
SEED = 42


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index：分箱边界由基准期（expected）的分位数决定"""
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ==================== 第一部分：漂移的类型 ====================


def drift_types():
    print("=" * 60)
    print("第一部分：漂移的类型")
    print("=" * 60)
    print("""
    ┌────────────────────────┬──────────────┬──────────────────────────────────────────┐
    │ 类型                   │ 什么变了     │ 金融中的例子                             │
    ├────────────────────────┼──────────────┼──────────────────────────────────────────┤
    │ 协变量漂移 Covariate   │ P(X)         │ 全市场换手率上升；注册制后新股结构变化   │
    │ 概念漂移 Concept       │ P(Y | X)     │ 小市值因子长期有效后突然失效；反转变动量 │
    │ 标签漂移 Label         │ P(Y)         │ 牛熊切换，收益分布整体平移               │
    └────────────────────────┴──────────────┴──────────────────────────────────────────┘

    最危险的是概念漂移：特征分布看起来一切正常，"特征 → 收益"的关系却已经变了。
    alpha 衰减是常态 —— 因子被发现、被拥挤（Phase 7 第 11 课）后，有效性会下降。
    """)


# ==================== 第二部分：滚动 IC 监控 ====================


def ic_monitor(data):
    print("\n" + "=" * 60)
    print("第二部分：预测效果监控 —— 滚动 Rank IC")
    print("=" * 60)
    pred = mt.walk_forward(data.X, data.y, lambda: mt.make_regressor(seed=SEED), min_train=MIN_TRAIN,
                           retrain_every=3)
    ic = mt.daily_rank_ic(pred, data.ret)
    known = ic.shift(2)  # t 期的 IC 在 t+2 期决策时才可见
    roll = known.rolling(IC_WINDOW, min_periods=IC_WINDOW).mean()
    alarm = roll < 0
    print(f"""
    标签滞后：t 期预测的 IC 要等持有期结束（t+1 期末的次日开盘）才算得出 → 在 t+2 期决策时才"看得到"。
    监控指标：决策时可见的最近 {IC_WINDOW} 期 IC 均值，跌破 0 报警。
    """)
    yearly = pd.DataFrame({"IC 均值": ic.groupby(ic.index.year).mean(),
                           "IC>0 占比": (ic > 0).groupby(ic.index.year).mean(),
                           "报警月数": alarm.groupby(alarm.index.year).sum().astype(int)})
    print(f"📋 {mt.gbdt_backend()}（每 3 个月重训）样本外 IC 按年统计：")
    print(yearly.to_string(float_format=lambda v: f"{v:+.3f}"))
    runs = (alarm != alarm.shift()).cumsum()[alarm]
    spans = [f"{g.index[0]:%Y-%m}~{g.index[-1]:%Y-%m}" for _, g in alarm[alarm].groupby(runs)]
    print(f"\n   报警区间：{', '.join(spans) if spans else '无'}")
    print("   💡 IC 是最重要但最慢的指标；输入 / 输出分布监控没有滞后，可以更早发现异常（第三部分）")
    return pred, ic, roll


# ==================== 第三部分：分布漂移 —— PSI 与 KS ====================


def distribution_drift(data, pred):
    print("\n" + "=" * 60)
    print("第三部分：输入分布监控 —— PSI 与 KS 检验")
    print("=" * 60)
    print("""
        PSI = Σ_b (a_b − e_b) · ln(a_b / e_b)
        e_b：基准期（训练期）落在第 b 个分箱的比例；a_b：当前期的比例；分箱 = 基准期十分位
    基准期：前 24 个月（第一个训练窗口）；当前期：之后每一年的全部样本
    """)
    dates = data.dates
    lvl = data.date_index
    base = lvl <= dates[MIN_TRAIN - 1]
    cols = ["vol_20", "turn_20", "amihud_20", "log_mcap", "ret_12_1", "ep"]
    years = sorted(set(lvl[~base].year))
    rows = {}
    for c in cols:
        e = data.raw.loc[base, c].to_numpy()
        rows[c] = {y: psi(e, data.raw.loc[(~base) & (lvl.year == y), c].to_numpy()) for y in years}
    tab = pd.DataFrame(rows).T
    print("📋 原始特征（截面标准化之前）相对基准期的 PSI：")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    worst = tab.stack().idxmax()
    print(f"   PSI 最大：{worst[0]} 在 {worst[1]} 年 = {tab.stack().max():.3f}；"
          f"> 0.25 的格子 {(tab > 0.25).sum().sum()} 个，0.1–0.25 的 {((tab > 0.1) & (tab <= 0.25)).sum().sum()} 个")

    y_last = years[-1]
    rank_psi = psi(data.X.loc[base, "vol_20"].to_numpy(), data.X.loc[lvl.year == y_last, "vol_20"].to_numpy())
    ks = ks_2samp(data.raw.loc[base, "vol_20"].dropna(), data.raw.loc[lvl.year == y_last, "vol_20"].dropna())
    print(f"""
    ⚠️ 陷阱：rank 后的 vol_20 在 {y_last} 年的 PSI = {rank_psi:.3f} —— rank 后分布永远接近均匀，PSI 看不出任何漂移。
       PSI 必须算在原始特征上。
    KS 检验（原始 vol_20，基准期 vs {y_last} 年）：统计量 {ks.statistic:.3f}，p 值 {ks.pvalue:.3g}
       注意：样本量很大时，KS 对很小的差异也会"显著"；样本量小时又可能检测不到 → 实践中更多看统计量大小 / PSI。
       另外同一股票不同月份的样本高度相关，KS 的 p 值（假设独立样本）偏乐观。
    """)

    # 输出监控：预测离散度、持仓风格
    disp = pred.groupby(level="datetime").std()
    top = pred.groupby(level="datetime").rank(ascending=False) <= 20
    size_expo = data.X.loc[pred.index, "log_mcap"].where(top).groupby(level="datetime").mean()
    out = pd.DataFrame({"预测截面标准差": disp.groupby(disp.index.year).mean(),
                        "Top20 市值暴露": size_expo.groupby(size_expo.index.year).mean()})
    print("📋 输出监控（无滞后）：预测离散度与 Top20 持仓的市值暴露（log_mcap rank 均值，负 = 偏小盘）")
    print(out.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("   💡 预测离散度骤降 → 模型变得'没主意'；风格暴露突变 → 模型在悄悄换赛道。两者都不需要等标签。")
    return tab


# ==================== 第四部分：再训练策略对比 ====================


def mean_psi(raw: pd.DataFrame, base_mask, cur_mask, bins: int = 5) -> float:
    return float(np.mean([psi(raw.loc[base_mask, c].to_numpy(), raw.loc[cur_mask, c].to_numpy(), bins=bins)
                          for c in mt.FEATURES]))


def triggered_walk_forward(data, min_gap: int = 3):
    """
    触发式重训（间隔 ≥ min_gap 期）：
      ① IC 报警：决策时可见的最近 IC_WINDOW 期 IC 均值 < 0
      ② PSI 报警：当期原始特征的平均 PSI（相对训练集）> 校准阈值
         校准阈值 = 训练集最后 12 个月"各月相对整个训练集"的平均 PSI 的最大值
         —— 每月只有约 120 个样本，PSI 有很大的小样本偏差，不能直接套用 0.1 / 0.25 的经验阈值
    """
    X, y, raw = data.X, data.y, data.raw
    lvl = X.index.get_level_values("datetime")
    dates = lvl.unique().sort_values()
    model, last_fit, train_mask, thresh = None, -999, None, np.inf
    preds, ics, events = [], {}, []
    for k in range(MIN_TRAIN + 1, len(dates)):
        reason = None
        if model is None:
            reason = "初始训练"
        elif k - last_fit >= min_gap:
            realized = pd.Series({d: v for d, v in ics.items() if d <= dates[k - 2]})
            drift = mean_psi(raw, train_mask, lvl == dates[k])
            if len(realized) >= IC_WINDOW and realized.iloc[-IC_WINDOW:].mean() < 0:
                reason = f"IC 报警（近 {IC_WINDOW} 期均值 {realized.iloc[-IC_WINDOW:].mean():+.3f}）"
            elif drift > thresh:
                reason = f"PSI 报警（平均 PSI {drift:.3f} > 阈值 {thresh:.3f}）"
        if reason:
            train_mask = lvl <= dates[k - 2]
            model = mt.make_regressor(seed=SEED).fit(X[train_mask], y[train_mask])
            thresh = max(mean_psi(raw, train_mask, lvl == d) for d in dates[max(0, k - 13):k - 1])
            last_fit = k
            events.append((dates[k], reason))
        te = lvl == dates[k]
        p = pd.Series(model.predict(X[te]), index=X.index[te])
        preds.append(p)
        ics[dates[k]] = mt.daily_rank_ic(p, data.ret[te]).mean()
    return pd.concat(preds), events


def retrain_comparison(data):
    print("\n" + "=" * 60)
    print("第四部分：再训练策略对比（全部为 Walk-Forward，gap = 1 期）")
    print("=" * 60)
    print("""
    ┌──────────────────┬──────────────────────────────┬──────────────────────────────┐
    │ 策略             │ 做法                         │ 权衡                         │
    ├──────────────────┼──────────────────────────────┼──────────────────────────────┤
    │ 固定模型         │ 训练一次，一直用             │ 简单；无法适应漂移           │
    │ 定期·扩展窗口    │ 每 N 月用全部历史重训        │ 数据多；对新规律反应慢       │
    │ 定期·滚动窗口    │ 每 N 月用最近 W 月重训       │ 适应快；丢掉早期罕见情景     │
    │ 触发式           │ 监控报警才重训               │ 省计算；阈值难定、反应滞后   │
    └──────────────────┴──────────────────────────────┴──────────────────────────────┘
    """)
    make = lambda: mt.make_regressor(seed=SEED)  # noqa: E731
    runs = {
        "固定模型（只训练 1 次）": mt.walk_forward(data.X, data.y, make, MIN_TRAIN, retrain_every=10**6,
                                         return_log=True),
        "定期 12 月·扩展窗口": mt.walk_forward(data.X, data.y, make, MIN_TRAIN, retrain_every=12, return_log=True),
        "定期 3 月·扩展窗口": mt.walk_forward(data.X, data.y, make, MIN_TRAIN, retrain_every=3, return_log=True),
        "定期 3 月·滚动 24 月": mt.walk_forward(data.X, data.y, make, MIN_TRAIN, retrain_every=3, window=24,
                                         return_log=True),
    }
    rows = {}
    for name, (p, log) in runs.items():
        rows[name] = {**mt.ic_stats(mt.daily_rank_ic(p, data.ret)), "训练次数": len(log)}
    p_trig, events = triggered_walk_forward(data)
    rows["触发式（IC < 0 或 PSI 超校准阈值）"] = {**mt.ic_stats(mt.daily_rank_ic(p_trig, data.ret)), "训练次数": len(events)}
    df = mt.ic_frame(rows).astype({"训练次数": int})
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n   触发式重训的触发记录：")
    for d, why in events:
        print(f"      {d.date()}  {why}")
    best = df["ICIR"].idxmax()
    print(f"""
    💡 本次 ICIR 最高：{best}。但请注意：
       - 这里只有 {int(df['期数'].iloc[0])} 期样本外，策略之间的差异大多在统计误差之内
       - 再训练频率、窗口长度、报警阈值都是超参数 —— 挑出"最好的那一种"本身就是一次多重检验（08 课）
       - PSI 在每月约 120 个样本上估计，有明显的小样本偏差（分箱越多越大）；本课用训练期各月的 PSI 校准阈值，
         若直接用 0.1，几乎每次检查都会报警，触发式就退化成了定期重训
       - Qlib 提供滚动训练工具（RollingGen 等）和针对概念漂移的研究方法（如 DDG-DA）
    """)
    return {k: v[0] for k, v in runs.items()} | {"触发式": p_trig}


def dashboard():
    print("\n" + "=" * 60)
    print("第五部分：监控仪表板清单")
    print("=" * 60)
    print("""
    预测效果（最重要，但有滞后）  滚动 RankIC / ICIR；TopK 超额与回撤；累计 IC 曲线斜率
    输入分布（无滞后）            原始特征 PSI / KS；缺失率；异常值比例
    输出分布（无滞后）            预测截面离散度；持仓风格暴露（市值、Beta、行业）；换手率
    模型内部                      特征重要性排序的变化（06 课 MDA 按年计算）
    流程                          数据延迟、上游字段变更、停牌 / 新股处理是否正常

    ✅ 原则：先有监控和"降级预案"（回退到线性模型或减仓），再上线 ML 模型。
    """)


def plot_results(ic, roll, preds, data):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(ic.index, ic, width=20, color=np.where(ic > 0, "tab:red", "tab:green"), alpha=0.5, label="当期 IC")
    axes[0].plot(roll.index, roll, color="black", label=f"决策时可见的 {IC_WINDOW} 期均值")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_title("滚动 Rank IC 监控（标签滞后 2 期）")
    axes[0].legend(fontsize=8)
    for name, p in preds.items():
        c = mt.daily_rank_ic(p, data.ret).cumsum()
        axes[1].plot(c.index, c, label=name)
    axes[1].set_title("不同再训练策略的累计 RankIC")
    axes[1].legend(fontsize=8)
    path = save_figure(fig, "10_drift_monitoring.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    drift_types()
    pred, ic, roll = ic_monitor(data)
    distribution_drift(data, pred)
    preds = retrain_comparison(data)
    dashboard()
    plot_results(ic, roll, preds, data)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 bootstrap 估计"每月 120 个样本、5 个分箱"时 PSI 在无漂移情况下的分布，据此校准报警阈值
    2. 实现"样本加权"再训练：全部历史，按半衰期 12 个月指数加权（sample_weight），与扩展 / 滚动窗口比较
    3. 实现模型集成：同时保留 3 个不同训练窗口的模型，预测取平均，比较 ICIR 与稳定性
    4. 在 06 课的 MDA 基础上，按年计算特征重要性排名，计算相邻年份排名的 Spearman 相关作为漂移指标
    """)


if __name__ == "__main__":
    main()
