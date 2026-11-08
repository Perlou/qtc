"""
LightGBM 截面收益预测
=====================

学习目标：
    1. 搭建月度截面选股的 ML 流程：特征 → 截面 rank 标签 → 滚动训练（Walk-Forward）→ Rank IC 评估
    2. 理解模型工厂：优先 LightGBM，缺 libomp 时回退 sklearn HistGradientBoosting（同为直方图 GBDT）
    3. 用"按时间切出、与训练集隔开"的验证集做早停，理解随机验证集为什么会泄漏
    4. 与 Phase 7 线性多因子（等权、IC 加权）在同一样本外区间公平对比

核心概念：
    - 标签：下月收益的截面 rank（直接对齐"排序"目标，去掉大盘涨跌）
    - 评估：RankIC_t = Spearman(ŷ_·t, r_·t)；ICIR = mean / std；不看 MSE
    - Walk-Forward：在 t 期只用"标签已实现"的样本训练（gap = 1 期）
    - 低信噪比下的正则：min_child_samples 大、num_leaves 适中、学习率小、特征 / 行采样

前置知识：
    - 04-purged-kfold-cv.py；Phase 7 第 05、09 课（IC 分析、因子合成）

环境要求：
    - pip install pandas numpy scikit-learn matplotlib；lightgbm 可选（macOS 需 brew install libomp）
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ml_tools as mt
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

MIN_TRAIN = 24  # 至少 24 个月训练数据
LOOKBACK = 24  # Phase 7 IC 加权的回看期数


# ==================== 第一部分：问题设定与标签 ====================


def problem_setup(data):
    print("=" * 60)
    print("第一部分：截面预测的问题设定")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────────────────────┐
    │ (datetime, instrument) │ feat_1 │ … │ feat_17 │ label            │
    │ 2024-11-29, 000001     │  0.31  │ … │  −0.12  │ 下月收益的截面排名│
    │ 2024-11-29, 000002     │ −0.20  │ … │   0.44  │ …                │
    │ …（每月约 120 只可交易股票）                                    │
    │ 目标：不是预测收益的绝对值，而是预测"同一个月里谁比谁好"          │
    └──────────────────────────────────────────────────────────────────┘

    ┌──────────────┬────────────────────────┬────────────────────────────────┐
    │ 标签         │ 定义                   │ 说明                           │
    ├──────────────┼────────────────────────┼────────────────────────────────┤
    │ 原始收益     │ r_i                    │ 被大盘涨跌主导，浪费模型容量   │
    │ 超额收益     │ r_i − r_bench          │ 去掉大盘，仍有截面尺度差异     │
    │ 截面 rank ✅ │ rank_t(r_i)/N_t − 0.5  │ 直接对齐排序目标，对极值稳健   │
    │ 排序学习     │ LambdaRank（分组排序） │ 直接优化 NDCG，需要离散相关度  │
    └──────────────┴────────────────────────┴────────────────────────────────┘
    """)
    r = data.ret
    date_mean = r.groupby(level="datetime").transform("mean")
    share = float(((date_mean - r.mean()) ** 2).sum() / ((r - r.mean()) ** 2).sum())
    print(f"📋 原始月收益的方差中，{share:.1%} 来自'当月所有股票的共同涨跌'（日期均值），"
          f"其余 {1 - share:.1%} 才是截面差异")
    print("   → 用原始收益当标签，模型会把相当一部分容量花在预测大盘上，而这对选股排序毫无帮助")


# ==================== 第二部分：模型工厂 ====================


def model_factory():
    print("\n" + "=" * 60)
    print("第二部分：模型工厂 —— LightGBM 与 HistGradientBoosting")
    print("=" * 60)
    print(f"""
    当前后端：{mt.gbdt_backend()}（ml_tools.make_regressor 自动选择）

    ┌─────────────────────┬───────────────────────┬───────┬────────────────────────────────┐
    │ LightGBM 参数       │ HistGradientBoosting  │ 本课  │ 作用                           │
    ├─────────────────────┼───────────────────────┼───────┼────────────────────────────────┤
    │ n_estimators        │ max_iter              │ {mt.GBDT_PARAMS['n_estimators']:<5} │ 树的数量                       │
    │ learning_rate       │ learning_rate         │ {mt.GBDT_PARAMS['learning_rate']:<5} │ 小学习率 + 多棵树 → 平滑       │
    │ num_leaves          │ max_leaf_nodes        │ {mt.GBDT_PARAMS['num_leaves']:<5} │ 逐叶生长（leaf-wise）的复杂度  │
    │ min_child_samples   │ min_samples_leaf      │ {mt.GBDT_PARAMS['min_child_samples']:<5} │ 低信噪比下最关键的正则         │
    │ colsample_bytree    │ max_features（逐分裂）│ {mt.GBDT_PARAMS['colsample_bytree']:<5} │ 特征采样，降低对单一特征的依赖 │
    │ subsample           │ （不支持）            │ {mt.GBDT_PARAMS['subsample']:<5} │ 行采样（bagging）              │
    │ reg_lambda          │ l2_regularization     │ {mt.GBDT_PARAMS['reg_lambda']:<5} │ 叶子值的 L2 惩罚               │
    └─────────────────────┴───────────────────────┴───────┴────────────────────────────────┘

    两者都是"特征分箱成直方图（默认 255 个箱）→ 在箱边界上找最佳分裂 → 逐叶生长"，
    LightGBM 另有 GOSS、EFB 等加速技巧，在本课的数据规模下差别不大。

    ⚠️ HistGradientBoosting 的内置早停（early_stopping=True）用 validation_fraction 从训练集里
       **随机**抽验证集 → 与训练样本同月、同股票相邻 → 泄漏。ml_tools 中已强制关闭，改用第四部分的时间切分早停。

    LightGBM 的原生写法（安装 libomp 后可直接使用）：
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    """)


# ==================== 第三部分：Walk-Forward 滚动训练 ====================


def walk_forward_demo(data):
    print("\n" + "=" * 60)
    print("第三部分：Walk-Forward 滚动训练")
    print("=" * 60)
    print("""
    时间口径（与 Phase 7 一致）：t 日收盘出信号 → t+1 开盘买入 → 下个月末 t' 的次日开盘卖出

       月末 t−2        月末 t−1         月末 t（训练 + 预测）
          │──标签 t−2──▶│ 已实现          │
          │             │──标签 t−1──────▶│──▶ 要到 t+1 开盘才实现
                                          │──标签 t ─────────────▶ 未来

    所以在 t 期训练时，最新可用的样本是 t−2 期（gap = 1）。扩展窗口，每月重训。
    """)
    pred, log = mt.walk_forward(data.X, data.y, lambda: mt.make_regressor(), min_train=MIN_TRAIN,
                                retrain_every=1, gap=1, return_log=True)
    dates = data.dates
    pos = {d: i for i, d in enumerate(dates)}
    leak = [r for r in log if pos[r["train_end"]] >= pos[r["test"]] - 1]
    print("📋 前 3 次 / 最后 1 次训练窗口：")
    for r in log[:3] + log[-1:]:
        print(f"   预测 {r['test'].date()}：训练 {r['train_start'].date()} ~ {r['train_end'].date()}，"
              f"{r['n_train']} 个样本")
    print(f"   共训练 {len(log)} 次；检查'训练集最后一期 ≥ 测试期前一期'的违规次数：{len(leak)}")
    print("   截面 rank 特征只用当期截面计算，不需要在训练集上拟合任何预处理参数")
    ic = mt.daily_rank_ic(pred, data.ret)
    print(f"\n📋 {mt.gbdt_backend()} 样本外（{ic.index[0].date()} ~ {ic.index[-1].date()}）：")
    print(mt.ic_frame({"GBDT（17 特征）": mt.ic_stats(ic)}).to_string(float_format=lambda v: f"{v:+.3f}"))
    return pred


# ==================== 第四部分：时间切分的早停 ====================


def staged_predict(model, X, iters):
    """各迭代次数下的预测：LightGBM 用 num_iteration，HGB 用 staged_predict"""
    if mt.HAS_LGB and hasattr(model, "booster_"):
        return {i: model.predict(X, num_iteration=i) for i in iters}
    want = set(iters)
    return {i: p for i, p in enumerate(model.staged_predict(X), start=1) if i in want}


def early_stopping_demo(data):
    print("\n" + "=" * 60)
    print("第四部分：早停 —— 验证集必须按时间切出并隔开")
    print("=" * 60)
    dates = data.dates
    k = MIN_TRAIN + 12  # 假设站在第 36 个调仓日
    lvl = data.date_index
    va_dates = dates[k - 1 - 6:k - 1]  # 最近 6 期（已实现的）做验证
    tr_dates = dates[:k - 1 - 6 - 1]  # 训练与验证之间再空 1 期（purge）
    tr, va = lvl.isin(tr_dates), lvl.isin(va_dates)
    print(f"   站在 {dates[k].date()}：训练 {tr_dates[0].date()} ~ {tr_dates[-1].date()}，"
          f"空 1 期，验证 {va_dates[0].date()} ~ {va_dates[-1].date()}")
    model = mt.make_regressor(n_estimators=600).fit(data.X[tr], data.y[tr])
    iters = [25, 50, 100, 200, 300, 400, 600]
    preds = staged_predict(model, data.X[va], iters)
    curve = {i: mt.daily_rank_ic(pd.Series(p, data.X.index[va]), data.ret[va]).mean() for i, p in preds.items()}
    s = pd.Series(curve, name="验证集 RankIC")
    s.index.name = "迭代次数"
    print(s.to_frame().T.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"""
    验证集 RankIC 最高的迭代次数：{s.idxmax()}。只有 6 期验证数据，这条曲线本身噪声很大；
    实践中常用 Purged K-Fold（04 课）在多个验证块上平均后再选迭代次数，并把"选参"计入试验次数（08 课）。
    早停用的验证集绝不能是测试集 —— 否则测试集就被"用掉"了。
    """)


# ==================== 第五部分：与 Phase 7 线性多因子对比 ====================


def ic_weighted(p7_long: pd.DataFrame, ret: pd.Series) -> pd.Series:
    """Phase 7 的 IC 加权：t 期只用 t−2 期及更早的 IC（滚动 LOOKBACK 期，负权重截为 0）"""
    ics = pd.DataFrame({k: mt.daily_rank_ic(p7_long[k], ret) for k in p7_long})
    dates = ics.index
    out = []
    for i, d in enumerate(dates):
        hist = ics.iloc[max(0, i - 1 - LOOKBACK):max(0, i - 1)].dropna()
        if len(hist) < 12:
            continue
        w = hist.mean().clip(lower=0)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)
        z = p7_long.xs(d, level="datetime", drop_level=False).fillna(0.0)
        out.append(z @ w)
    return pd.concat(out)


def compare_with_linear(data, pred_gbdt):
    print("\n" + "=" * 60)
    print("第五部分：与 Phase 7 线性多因子对比（同一样本外区间）")
    print("=" * 60)
    clean, p7 = mt.phase7_composite(data)
    p7 = p7.fillna(0.0)
    print(f"""
    参赛选手（所有方法都只用当时可得的信息）：
       ① P7 等权：Phase 7 的 8 个因子（去极值、标准化、行业市值中性化、方向统一）等权
       ② P7 IC 加权：过去 {LOOKBACK} 期 IC 均值加权（lag = 2）
       ③ Ridge：17 个 rank 特征的线性回归（StandardScaler 只在训练集上 fit）
       ④ GBDT（P7 因子）：与 ①② 相同的 8 个因子作为输入 → 区分"模型的贡献"
       ⑤ GBDT（17 特征）：第三部分的模型 → "模型 + 特征"的共同贡献
    """)
    ridge = mt.walk_forward(data.X, data.y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
                            min_train=MIN_TRAIN)
    gbdt_p7 = mt.walk_forward(p7, data.y, lambda: mt.make_regressor(), min_train=MIN_TRAIN)
    preds = {
        "① P7 等权": p7.mean(axis=1),
        "② P7 IC 加权": ic_weighted(p7, data.ret),
        "③ Ridge（17 特征）": ridge,
        "④ GBDT（P7 因子）": gbdt_p7,
        "⑤ GBDT（17 特征）": pred_gbdt,
    }
    common = pred_gbdt.index.get_level_values("datetime").unique()
    rows, ics = {}, {}
    for name, p in preds.items():
        p = p[p.index.get_level_values("datetime").isin(common)]
        ic = mt.daily_rank_ic(p, data.ret)
        ics[name] = ic
        ls = mt.long_short_spread(p, data.ret)
        rows[name] = {**mt.ic_stats(ic), "Q5−Q1 年化": ls.mean() * 12}
    df = mt.ic_frame(rows)
    print(f"📋 样本外 {common[0].date()} ~ {common[-1].date()}（{len(common)} 期，未扣成本）：")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    best = df["ICIR"].idxmax()
    corr = pd.DataFrame({k: v for k, v in preds.items()}).loc[pred_gbdt.index]
    c = corr.groupby(level="datetime").corr(method="spearman").groupby(level=1).mean()
    print(f"\n   ICIR 最高：{best}")
    print(f"   GBDT（17 特征）与 P7 等权的平均截面相关：{c.loc['⑤ GBDT（17 特征）', '① P7 等权']:+.2f}")
    diff = (ics["⑤ GBDT（17 特征）"] - ics["① P7 等权"]).dropna()
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    print(f"   GBDT（17 特征）− P7 等权 的 IC 差：均值 {diff.mean():+.3f}，t 值 {t:+.2f}")
    p7_best = df.loc[["① P7 等权", "② P7 IC 加权"], "ICIR"].max()
    ml_best = df.loc[["③ Ridge（17 特征）", "④ GBDT（P7 因子）", "⑤ GBDT（17 特征）"], "ICIR"].max()
    sig = "统计上不显著（|t| < 2）" if abs(t) < 2 else "统计上显著（|t| ≥ 2）"
    if p7_best >= ml_best:
        verdict = """本次样本外，Phase 7 线性合成的 ICIR 不低于所有 ML 方法。可能的原因：
         Phase 7 的因子经过了行业市值中性化和方向统一，这些"人工先验"在小样本上很值钱；
         GBDT 需要从数据里自己学方向和交互，样本少时学不稳"""
    else:
        verdict = """本次样本外，ML 方法的 ICIR 高于 Phase 7 线性合成。
         但要先确认差异是否显著、是否来自风格暴露（如押注小市值），再看扣费后的组合收益"""
    print(f"""
    💡 如实解读：
       - GBDT（17 特征）与 P7 等权的 IC 差异{sig}；只有约 130 只股票、{len(common)} 期样本外，误差很大
       - {verdict}
       - ML 的优势通常在"大量股票 × 大量特征"时才更容易显现（全 A 股 5000 只、Alpha158 的 158 个特征，09 课）
       - 无论谁赢，都要继续看扣费组合（07 课）和多次试验修正（08 课）
    """)
    return ics


def plot_ic(ics):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, ic in ics.items():
        ax.plot(ic.index, ic.cumsum(), label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"样本外累计 RankIC（{mt.gbdt_backend()} vs 线性多因子）")
    ax.legend(fontsize=8)
    path = save_figure(fig, "05_cumulative_ic.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    problem_setup(data)
    model_factory()
    pred = walk_forward_demo(data)
    early_stopping_demo(data)
    ics = compare_with_linear(data, pred)
    plot_ic(ics)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把标签从截面 rank 换成原始收益，重跑第三部分，RankIC 如何变化？
    2. 把扩展窗口改为 36 个月滚动窗口（walk_forward 的 window 参数），比较 ICIR 与 IC 的稳定性
    3. 把 ①P7 等权 的分数作为第 18 个特征加入 GBDT（"线性先验 + 非线性残差"），结果如何？
    4. 安装 libomp 后（brew install libomp），确认程序自动切换到 LightGBM，比较两个后端的 IC 与耗时
    """)


if __name__ == "__main__":
    main()
