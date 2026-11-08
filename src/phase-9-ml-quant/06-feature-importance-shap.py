"""
特征重要性：MDI、MDA 与 SHAP
============================

学习目标：
    1. 计算三种特征重要性：MDI（gain，样本内）、MDA（样本外置换，按 RankIC）、SHAP（Shapley 分解）
    2. 看到 MDI 会给纯噪声特征正的重要性，而样本外 MDA 不会
    3. 演示替代效应：两个高度相关的特征分摊重要性；用聚类特征重要性（按簇置换）修正
    4. 学会用特征重要性做"泄漏检测"和"经济合理性检查"

核心概念：
    - MDI：每个特征在所有分裂上带来的损失下降之和（训练集上计算）
    - MDA：在测试集上打乱某个特征，RankIC 下降多少（在每个截面日期内部打乱，保持截面分布）
    - SHAP：f(x) = φ₀ + Σ φ_j(x)，φ_j 为特征 j 在各种"特征联盟"中的平均边际贡献
    - 替代效应：特征 A、B 几乎相同 → 打乱 A 时模型用 B 顶上 → 两者的 MDA 都偏低

前置知识：
    - 05-lightgbm-stock-prediction.py

环境要求：
    - pip install pandas numpy scipy scikit-learn shap matplotlib（lightgbm 可选）
    - 数据来自 Phase 7 共享面板；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import ml_tools as mt
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SEED = 42
N_REPEATS = 5
N_NOISE = 5
SHAP_SAMPLE = 1500


# ==================== 第一部分：三种方法 ====================


def methods_overview():
    print("=" * 60)
    print("第一部分：为什么特征重要性比回测更重要")
    print("=" * 60)
    print("""
    📝 比喻：回测像考试分数，特征重要性像查看解题过程。
       分数高但解题过程是"偷看了答案"（泄漏特征）或"蒙对的"（噪声特征），就不值得信任。

    ┌──────┬───────────────────────────────┬──────────┬──────────────────────┬──────────────────────┐
    │ 方法 │ 原理                          │ 样本     │ 优点                 │ 缺点                 │
    ├──────┼───────────────────────────────┼──────────┼──────────────────────┼──────────────────────┤
    │ MDI  │ 分裂带来的损失下降（gain）之和│ 样本内   │ 快，训练完即得       │ 噪声特征也有正值；   │
    │      │                               │          │                      │ 识别不了过拟合       │
    │ MDA  │ 测试集上打乱特征，得分下降多少│ 样本外   │ 模型无关；直接度量   │ 慢；相关特征互相掩护 │
    │      │                               │          │ 对样本外表现的贡献   │（替代效应）          │
    │ SHAP │ Shapley 值分解每个预测        │ 均可     │ 有方向、可解释单样本 │ 同样受相关特征影响   │
    │ SFI  │ 每次只用一个特征训练并做样本外│ 样本外   │ 完全没有替代效应     │ 忽略交互             │
    └──────┴───────────────────────────────┴──────────┴──────────────────────┴──────────────────────┘
    """)


def make_split(data, extra: dict | None = None):
    """时间切分：前 60% 训练（与测试隔 1 期），后 40% 测试；extra 为额外加入的特征列"""
    X = data.X.copy()
    for k, v in (extra or {}).items():
        X[k] = v
    dates = data.dates
    cut = int(len(dates) * 0.6)
    lvl = data.date_index
    tr, te = lvl <= dates[cut - 2], lvl >= dates[cut]
    return X, tr, te


def mda_rank_ic(model, X_te: pd.DataFrame, y_te: pd.Series, groups: list | None = None,
                n_repeats: int = N_REPEATS, seed: int = SEED) -> pd.Series:
    """样本外置换重要性：在每个截面日期内部打乱（一组特征一起打乱），返回 RankIC 下降"""
    rng = np.random.default_rng(seed)
    base = mt.daily_rank_ic(pd.Series(model.predict(X_te), X_te.index), y_te).mean()
    lvl = X_te.index.get_level_values("datetime")
    blocks = [np.flatnonzero(lvl == d) for d in lvl.unique()]
    groups = groups or [[c] for c in X_te.columns]
    out = {}
    for g in groups:
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.to_numpy().copy()
            cols = [X_te.columns.get_loc(c) for c in g]
            for b in blocks:
                Xp[b[:, None], cols] = Xp[rng.permutation(b)[:, None], cols]  # 同一组特征用同一个置换
            ic = mt.daily_rank_ic(pd.Series(model.predict(pd.DataFrame(Xp, columns=X_te.columns)), X_te.index),
                                  y_te).mean()
            drops.append(base - ic)
        out[" + ".join(g)] = float(np.mean(drops))
    return pd.Series(out)


# ==================== 第二部分：MDI vs MDA ====================


def mdi_vs_mda(data):
    print("\n" + "=" * 60)
    print("第二部分：MDI（样本内）vs MDA（样本外）—— 加入两个纯噪声特征")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    noise = [f"noise_{i}" for i in range(1, N_NOISE + 1)]
    extra = {c: rng.uniform(-0.5, 0.5, len(data.X)) for c in noise}
    X, tr, te = make_split(data, extra)
    model = mt.make_regressor(seed=SEED).fit(X[tr], data.y[tr])
    base = mt.daily_rank_ic(pd.Series(model.predict(X[te]), X.index[te]), data.ret[te]).mean()
    mdi = mt.gain_importance(model, X.columns)
    mda = mda_rank_ic(model, X[te], data.ret[te])
    df = pd.DataFrame({"MDI 占比": mdi, "MDA（RankIC 下降）": mda})
    df["MDI 排名"] = df["MDI 占比"].rank(ascending=False).astype(int)
    df["MDA 排名"] = df["MDA（RankIC 下降）"].rank(ascending=False).astype(int)
    df = df.sort_values("MDA（RankIC 下降）", ascending=False)
    print(f"   加入 {N_NOISE} 个与收益无关的均匀分布噪声特征 noise_1 ~ noise_{N_NOISE}")
    print(f"   训练 {X[tr].index[0][0].date()} ~ {X[tr].index[-1][0].date()}，测试 {X[te].index[0][0].date()} ~ "
          f"{X[te].index[-1][0].date()}；测试集基准 RankIC = {base:+.3f}")
    print(f"   MDA：每个特征在每个截面日期内打乱 {N_REPEATS} 次取平均\n")
    print(df.to_string(float_format=lambda v: f"{v:+.4f}"))
    real = [c for c in X.columns if c not in noise]
    floor = df.loc[noise, "MDA（RankIC 下降）"].max()
    above = [c for c in df.index if c in real and df.loc[c, "MDA（RankIC 下降）"] > floor]
    print(f"""
    💡 噪声特征的 MDI 平均占比 {mdi[noise].mean():.1%}，真实特征平均 {mdi[real].mean():.1%}
       （噪声特征的 MDI 排名：{sorted(df.loc[noise, 'MDI 排名'].tolist())} / {len(df)}）
       → MDI 只看"训练时降了多少损失"，噪声特征照样能被用来拟合噪声，所以拿到正的重要性。
       噪声特征的 MDA：平均 {mda[noise].mean():+.4f}，最大 {floor:+.4f}。
       → MDA 看"打乱后样本外变差多少"；噪声特征的 MDA 应在 0 附近波动，它们的波动幅度就是 MDA 的"噪声地板"。
       测试期只有 {X[te].index.get_level_values(0).nunique()} 期，MDA 本身噪声不小；
       超过噪声特征最大 MDA 的真实特征：{above if above else '无'}。
       ⚠️ 用 MDI 做特征筛选会保留噪声特征；应以样本外 MDA（并参照噪声地板）为准。
    """)
    return model, X, tr, te, df


# ==================== 第三部分：SHAP ====================


def shap_values(model, X_bg: pd.DataFrame, X_explain: pd.DataFrame):
    """优先 TreeExplainer（LightGBM、HGB 均支持）；失败时降级为 Permutation 解释器（小样本）"""
    import shap

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            sv = np.asarray(explainer.shap_values(X_explain))
            base = float(np.ravel(explainer.expected_value)[0])
        return sv, base, "TreeExplainer（精确的 TreeSHAP）"
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️ TreeExplainer 不支持当前模型（{type(e).__name__}），降级为 Permutation 解释器（200 个样本）")
        X_small = X_explain.iloc[:200]
        explainer = shap.explainers.Permutation(model.predict, X_bg.iloc[:100])
        res = explainer(X_small)
        return np.asarray(res.values), float(np.mean(res.base_values)), "Permutation（近似）"


def shap_demo(model, X, tr, te, data):
    print("\n" + "=" * 60)
    print("第三部分：SHAP —— 把每个预测拆成各特征的贡献")
    print("=" * 60)
    print("""
        f(x) = φ₀ + Σ_j φ_j(x)
        φ_j = Σ_{S ⊆ F∖{j}} |S|!(k−|S|−1)!/k! · [f_{S∪{j}}(x) − f_S(x)]

    直觉：特征 j 加入各种"特征联盟"时带来的平均边际贡献。对树模型有精确快速算法 TreeSHAP。
    """)
    X_te = X[te].sample(min(SHAP_SAMPLE, int(te.sum())), random_state=SEED)
    sv, base, how = shap_values(model, X[tr], X_te)
    X_used = X_te.iloc[:len(sv)]
    recon = base + sv.sum(axis=1)
    err = float(np.abs(recon - model.predict(X_used)).max())
    print(f"   解释器：{how}；样本数 {len(sv)}；可加性检查 max|φ₀ + Σφ − f(x)| = {err:.2e}")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    direction = pd.Series([np.corrcoef(X_used[c], sv[:, j])[0, 1] if X_used[c].std() > 0 else np.nan
                           for j, c in enumerate(X.columns)], index=X.columns)
    ic = pd.Series({c: mt.daily_rank_ic(data.X[c][tr], data.ret[tr]).mean() if c in data.X else np.nan
                    for c in X.columns})
    df = pd.DataFrame({"mean|SHAP|": imp, "SHAP 方向": direction, "训练期单特征 IC": ic}).sort_values(
        "mean|SHAP|", ascending=False)
    df.insert(0, "含义", [mt.FEATURE_LABELS.get(c, "纯噪声") for c in df.index])
    print(df.head(10).to_string(float_format=lambda v: f"{v:+.4f}"))
    agree = (np.sign(df["SHAP 方向"]) == np.sign(df["训练期单特征 IC"])).loc[df.index[:10]].dropna()
    print(f"""
    "SHAP 方向" = 特征值与其 SHAP 值的相关：正 → 特征越大预测越高。
    前 10 个特征中，SHAP 方向与训练期单特征 IC 符号一致的有 {int(agree.sum())} / {len(agree)} 个。
    💡 经济合理性检查：方向是否讲得通？如果高换手 → 高预测，就要问"为什么"，而不是直接上线。
    """)
    return imp


# ==================== 第四部分：替代效应与聚类重要性 ====================


def substitution_synthetic():
    """受控实验：已知真实结构的合成截面数据，替代效应一目了然"""
    rng = np.random.default_rng(SEED)
    n_dates, n_stocks = 100, 200
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=n_dates), range(n_stocks)],
                                     names=["datetime", "instrument"])
    n = len(idx)
    X = pd.DataFrame({"signal": rng.normal(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n)}, index=idx)
    y = pd.Series(0.3 * X["signal"] + 0.15 * X["z1"] + rng.normal(size=n), index=idx)
    lvl = idx.get_level_values("datetime")
    tr, te = lvl < lvl.unique()[60], lvl >= lvl.unique()[60]
    out = {}
    for with_copy in [False, True]:
        Xc = X.assign(signal_copy=X["signal"] + rng.normal(0, 0.05, n)) if with_copy else X
        m = mt.make_regressor(seed=SEED).fit(Xc[tr], y[tr])
        groups = [["signal"], ["signal_copy"], ["signal", "signal_copy"]] if with_copy else [["signal"]]
        mda = mda_rank_ic(m, Xc[te], y[te], groups=groups)
        mdi = mt.gain_importance(m, Xc.columns)
        out[with_copy] = (mda, mdi)
    (mda0, mdi0), (mda1, mdi1) = out[False], out[True]
    rows = {
        "无副本：signal": {"MDI 占比": mdi0["signal"], "MDA": mda0.iloc[0]},
        "有副本：signal": {"MDI 占比": mdi1["signal"], "MDA": mda1.iloc[0]},
        "有副本：signal_copy": {"MDI 占比": mdi1["signal_copy"], "MDA": mda1.iloc[1]},
        "有副本：两者一起打乱（簇）": {"MDI 占比": mdi1[["signal", "signal_copy"]].sum(), "MDA": mda1.iloc[2]},
    }
    print("📋 ① 受控实验（合成数据：y = 0.3·signal + 0.15·z1 + 噪声；signal_copy = signal + 很小的噪声）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"""
       MDI：signal 从 {mdi0['signal']:.1%} 变成 {mdi1['signal']:.1%} + {mdi1['signal_copy']:.1%}（被两者分摊）
       MDA：单独打乱 signal / signal_copy 分别为 {mda1.iloc[0]:+.4f} / {mda1.iloc[1]:+.4f}，
            一起打乱为 {mda1.iloc[2]:+.4f}（无副本时 {mda0.iloc[0]:+.4f}）
       → 单独打乱时，模型用另一个顶上，重要性被低估；按簇一起打乱才能看到这组信息的真实价值。
    """)


def substitution_effect(data, base_mda: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：替代效应 —— 两个几乎相同的特征分摊重要性")
    print("=" * 60)
    substitution_synthetic()
    top = base_mda[[c for c in base_mda.index if c in mt.FEATURES]].idxmax()
    rng = np.random.default_rng(SEED + 1)
    dup = data.X[top] + rng.normal(0, 0.02, len(data.X))
    X, tr, te = make_split(data, {f"{top}_copy": dup})
    print(f"📋 ② 真实面板：取第二部分 MDA 最高的 {top}，加入近似副本 {top}_copy"
          f"（相关 {np.corrcoef(dup, data.X[top])[0, 1]:.3f}），各打乱 20 次：")
    X0, tr0, te0 = make_split(data)
    m0 = mt.make_regressor(seed=SEED).fit(X0[tr0], data.y[tr0])
    m1 = mt.make_regressor(seed=SEED).fit(X[tr], data.y[tr])
    pair = [top, f"{top}_copy"]
    mda0 = mda_rank_ic(m0, X0[te0], data.ret[te0], groups=[[top]], n_repeats=20)
    mda1 = mda_rank_ic(m1, X[te], data.ret[te], groups=[[top], [f"{top}_copy"], pair], n_repeats=20)
    mdi0, mdi1 = mt.gain_importance(m0, X0.columns), mt.gain_importance(m1, X.columns)
    rows = {
        f"无副本：{top}": {"MDI 占比": mdi0[top], "MDA": mda0.iloc[0]},
        f"有副本：{top}": {"MDI 占比": mdi1[top], "MDA": mda1.iloc[0]},
        f"有副本：{top}_copy": {"MDI 占比": mdi1[f"{top}_copy"], "MDA": mda1.iloc[1]},
        "有副本：两者一起打乱（簇）": {"MDI 占比": mdi1[pair].sum(), "MDA": mda1.iloc[2]},
    }
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"""
       MDI 的分摊同样明显：{mdi0[top]:.1%} → {mdi1[top]:.1%} + {mdi1[f'{top}_copy']:.1%}。
       真实数据上 MDA 的噪声大（测试期只有 {X[te].index.get_level_values(0).nunique()} 期，信号很弱），簇 MDA 不一定大于单独打乱；
       另外，只打乱两个近似副本中的一个，会制造出"两者取值不一致"的样本 —— 训练时从未见过，
       模型对它们的预测可能很离谱，这也会扭曲单独打乱的 MDA。这是置换重要性在相关特征上的另一个缺陷。
    """)

    # 对全部 17 个特征做相关性聚类，按簇计算 MDA
    corr = data.X[tr0].groupby(level="datetime").corr().groupby(level=1).mean().loc[mt.FEATURES, mt.FEATURES]
    dist = np.sqrt(np.clip((1 - corr.abs().to_numpy()) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.45, criterion="distance")
    clusters = [list(np.array(mt.FEATURES)[lab == c]) for c in np.unique(lab)]
    cmda = mda_rank_ic(m0, X0[te0], data.ret[te0], groups=clusters).sort_values(ascending=False)
    print(f"📋 聚类特征重要性（相关性距离 √((1−|ρ|)/2) 层次聚类，阈值 0.45 → {len(clusters)} 个簇）：")
    for name, v in cmda.items():
        print(f"   {v:+.4f}  [{name}]")


# ==================== 第五部分：泄漏检测 ====================


def leakage_detection(data):
    print("\n" + "=" * 60)
    print("第五部分：用特征重要性做泄漏检测")
    print("=" * 60)
    rng = np.random.default_rng(SEED + 2)
    leak = 0.15 * data.y.fillna(0) + rng.normal(0, 0.3, len(data.X))  # 偷偷混入了一点"标签"
    X, tr, te = make_split(data, {"suspicious": leak})
    m = mt.make_regressor(seed=SEED).fit(X[tr], data.y[tr])
    mdi = mt.gain_importance(m, X.columns).sort_values(ascending=False)
    ic = mt.daily_rank_ic(pd.Series(m.predict(X[te]), X.index[te]), data.ret[te]).mean()
    print(f"""
    构造一个"可疑特征"：0.15 × 标签 + 大量噪声（模拟误用了未来复权因子、公告日期之类的泄漏）。
       它的单特征 RankIC = {mt.daily_rank_ic(X['suspicious'], data.ret).mean():+.3f}
       模型测试集 RankIC = {ic:+.3f}
       MDI 第一名：{mdi.index[0]}（占 {mdi.iloc[0]:.1%}），第二名 {mdi.index[1]}（占 {mdi.iloc[1]:.1%}），约 {mdi.iloc[0] / mdi.iloc[1]:.1f} 倍

    💡 一个特征的重要性远超其他所有特征、IC 好得不像话 → 第一反应应该是查泄漏，而不是庆祝。
       常见泄漏源：未来复权因子、财报按报告期对齐、用了 t+1 的停牌 / 涨跌停信息、标签参与了特征标准化。
    """)


def plot_importance(df: pd.DataFrame, shap_imp: pd.Series):
    setup_chinese_font()
    t = pd.DataFrame({"MDI": df["MDI 占比"], "MDA": df["MDA（RankIC 下降）"], "SHAP": shap_imp})
    t = t / t.abs().max()
    t = t.sort_values("MDA")
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(t))
    for k, (c, off) in enumerate(zip(t.columns, [-0.27, 0, 0.27])):
        ax.barh(y + off, t[c], 0.27, label=c)
    ax.set_yticks(y, t.index, fontsize=8)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_title("特征重要性（各自除以最大绝对值）：注意噪声特征的 MDI")
    ax.legend()
    path = save_figure(fig, "06_feature_importance.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    methods_overview()
    model, X, tr, te, df = mdi_vs_mda(data)
    shap_imp = shap_demo(model, X, tr, te, data)
    substitution_effect(data, df["MDA（RankIC 下降）"])
    leakage_detection(data)
    plot_importance(df, shap_imp)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 实现 SFI（单特征重要性）：每次只用一个特征做 Walk-Forward，与 MDA 排名比较
    2. 把 MDA 改为在 Purged K-Fold（04 课）的每个验证块上计算再平均，排名是否更稳定？
    3. 在不同年份分别计算 MDA，重要性排序随时间的变化可以作为漂移信号（10 课）
    4. 画 SHAP 依赖图（shap.dependence_plot）：log_mcap 与预测的关系是线性的吗？与哪个特征有交互？
    """)


if __name__ == "__main__":
    main()
