"""
Purged K-Fold 与 Embargo
========================

学习目标：
    1. 理解普通 K-Fold 在金融数据上的两种泄漏：标签重叠 + 特征自相关、同一天的截面相关
    2. 亲手实现事件型 Purged K-Fold（AFML 第 7 章）与面板版"按日期切分 + purge + embargo"
    3. 用"纯噪声特征"做对照实验：普通 K-Fold 也能得到虚高分数，Purged K-Fold 回到随机水平
    4. 知道调参用 Purged K-Fold / CPCV、最终评估用 Walk-Forward

核心概念：
    - Purging：删除标签区间 [t0, t1] 与测试集标签区间有交集的训练样本
    - Embargo：测试块之后再删除一段训练样本（序列相关使"之后"的特征仍含测试期信息）
    - 面板数据必须按日期分组：同一天的股票不能同时出现在训练集和测试集

前置知识：
    - 03-labeling-triple-barrier.py（t1 与重叠标签）
    - Phase 4 第 09 课 Walk-Forward

环境要求：
    - pip install pandas numpy scikit-learn matplotlib（lightgbm 可选）
    - 沪深 300 日线 + Phase 7 面板；无网络时可用 QTC_OFFLINE=1 运行
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

import ml_tools as mt
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

H = 20  # 事件型实验的标签期限（交易日）
N_SPLITS = 5
SEEDS = [0, 1, 2]


# ==================== 第一部分：普通 K-Fold 为什么泄漏 ====================


def why_leak():
    print("=" * 60)
    print("第一部分：普通 K-Fold 为什么会泄漏")
    print("=" * 60)
    print(f"""
    时间轴 ───────────────────────────────────────────────────────────▶
      样本 t=100：特征用到第 100 天，标签 = 第 100→{100 + H} 天收益
      样本 t=103：特征用到第 103 天，标签 = 第 103→{103 + H} 天收益

      若 t=100 在训练集、t=103 在测试集：
        → 两个标签共享第 103→{100 + H} 天的收益（{H - 3} 天！）
        → 特征也几乎一样（20 日动量在相邻两天几乎不变）
        → 模型只要"记住邻居"就能答对测试题，与特征是否真有预测力无关

                      ┌──────── 测试集 ────────┐
      训练 训练 训练  │purge│ 测试 测试 测试  │embargo│ 训练 训练
      ████ ████ ███  ░░░░░ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ░░░░░░░ ████ ████
                      ↑                        ↑
          标签区间与测试重叠的样本删除       测试块之后再隔离一段
    """)


# ==================== 第二部分：纯噪声特征的对照实验 ====================


def noise_features(index: pd.DatetimeIndex, seed: int, k: int = 5) -> pd.DataFrame:
    """与收益完全无关、但有自相关的噪声特征（白噪声的 60 日滚动均值）"""
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(rng.normal(size=(len(index), k)), index=index, columns=[f"noise_{i}" for i in range(k)])
    return z.rolling(60).mean()


def cv_auc(X, y, splits, make):
    scores = []
    for tr, te in splits:
        m = make().fit(X.iloc[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(scores))


def noise_experiment():
    print("\n" + "=" * 60)
    print("第二部分：纯噪声特征 —— 普通 K-Fold 也能'预测'市场")
    print("=" * 60)
    idx = get_index_daily("000300", "20100101", "20251231")
    tag = "（⚠️ 合成数据）" if is_synthetic(idx) else ""
    close = idx["close"]
    fwd = close.shift(-H) / close - 1
    print(f"""
    数据：沪深 300 日线{tag}，每天一个样本；标签 y = 1[未来 {H} 日收益 > 0]，t1 = t + {H} 个交易日
    特征：5 列与市场毫无关系的随机数（白噪声的 60 日滚动均值 → 有自相关，像真实特征一样"缓慢变化"）
    模型：{mt.gbdt_backend()} 分类器；指标：{N_SPLITS} 折平均 AUC（0.5 = 随机）
    """)
    pos = np.arange(len(close))
    rows = []
    for seed in SEEDS:
        X = noise_features(close.index, seed)
        ok = X.notna().all(axis=1) & fwd.notna()
        X, y = X[ok], (fwd[ok] > 0).astype(int)
        t1_pos = np.minimum(pos[ok.to_numpy()] + H, len(close) - 1)
        t1 = pd.Series(close.index[t1_pos], index=X.index)

        def make():
            return mt.make_classifier(seed=seed, n_estimators=100, num_leaves=15, min_child_samples=20,
                                      learning_rate=0.1)

        rows.append({
            "噪声种子": seed,
            "KFold(shuffle=True)": cv_auc(X, y, KFold(N_SPLITS, shuffle=True, random_state=seed).split(X), make),
            "KFold(不打乱)": cv_auc(X, y, KFold(N_SPLITS).split(X), make),
            "PurgedKFold(embargo 1%)": cv_auc(X, y, mt.PurgedKFold(N_SPLITS, t1, 0.01).split(X), make),
        })
    df = pd.DataFrame(rows).set_index("噪声种子")
    df.loc["平均"] = df.mean()
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"""
    💡 特征是纯噪声，任何高于 0.5 的 AUC 都是泄漏：
       - 打乱的 K-Fold：平均 AUC {df.loc['平均'].iloc[0]:.3f} —— 测试样本的"邻居"都在训练集里
       - 不打乱的 K-Fold：{df.loc['平均'].iloc[1]:.3f} —— 只在每个测试块的边界泄漏
       - Purged K-Fold：{df.loc['平均'].iloc[2]:.3f} —— 删除重叠样本后回到随机附近
    ⚠️ 这就是"能解释普通 K-Fold 在金融数据上的泄漏问题"这条检查项的答案。
    """)
    return close, X, t1


def purge_anatomy(close, X, t1):
    print("\n" + "=" * 60)
    print("第三部分：Purging 和 Embargo 各删掉了什么")
    print("=" * 60)
    n = len(X)
    rows = []
    for k, (tr, te) in enumerate(mt.PurgedKFold(N_SPLITS, t1, 0.01).split(X)):
        test_t0, test_t1 = t1.index[te[0]], t1.iloc[te].max()
        overlap = (t1.index <= test_t1) & (t1.to_numpy() >= test_t0)
        n_purge = int(overlap.sum() - len(te))
        n_embargo = int(n - len(te) - len(tr) - n_purge)
        rows.append({"折": k + 1, "测试区间": f"{test_t0.date()} ~ {t1.index[te[-1]].date()}",
                     "测试样本": len(te), "训练样本": len(tr), "purge 删除": n_purge, "embargo 删除": n_embargo})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"""
    - purge：测试块前约 {H} 个样本（标签延伸进测试期）+ 测试块后约 {H} 个样本（标签区间与测试期末尾重叠）
    - embargo：再删除测试块之后 1% × {n} ≈ {int(n * 0.01)} 个样本
    - 第 1 折没有"之前"，最后一折没有"之后"，删除数量较少
    """)


# ==================== 第四部分：面板版 —— 按日期切分 ====================


def panel_experiment(data):
    print("\n" + "=" * 60)
    print("第四部分：截面面板 —— 同一天的股票不能拆开")
    print("=" * 60)
    print("""
    截面相关泄漏：同一天所有股票共同受大盘影响。若随机按"行"切分，
    测试集某只股票当天的收益，可以从训练集里同一天其他股票的收益"猜"出来。
    只要模型能识别"这是哪一天"，就能把当天的大盘涨跌背下来。

    实验：特征 = 17 个截面 rank 特征 + 2 个市场状态特征（沪深 300 的 20 日波动、20 日收益，
          同一天所有股票取值相同 → 相当于"日期指纹"）；标签 = 原始持有期收益（未去市场）
    指标：① 混合相关（所有样本放在一起的 Pearson 相关，会被"猜中大盘"抬高）
          ② 逐期 Rank IC 均值（只看同一天的排序：不受大盘整体涨跌影响，
             但也可能被"当天哪一类股票涨"的同日共振抬高 —— 同行业、同风格的股票同一天一起涨跌）
    """)
    b = data.panel.bench["hs300_close"]
    mkt = pd.DataFrame({"mkt_vol_20": b.pct_change().rolling(20).std(), "mkt_ret_20": b.pct_change(20)})
    lvl = data.date_index
    X = data.X.assign(**{c: mkt[c].reindex(lvl).to_numpy() for c in mkt})
    y = data.ret
    groups = lvl.to_numpy()

    def run(splits):
        pred = pd.Series(np.nan, index=X.index)
        for tr, te in splits:
            m = mt.make_regressor(n_estimators=150, num_leaves=15, min_child_samples=50).fit(X.iloc[tr], y.iloc[tr])
            pred.iloc[te] = m.predict(X.iloc[te])
        ok = pred.notna()
        return {"混合相关": float(np.corrcoef(pred[ok], y[ok])[0, 1]),
                "逐期 RankIC": float(mt.daily_rank_ic(pred[ok], y[ok]).mean()),
                "覆盖样本": int(ok.sum())}

    rows = {
        "❌ 随机按行 KFold": run(KFold(N_SPLITS, shuffle=True, random_state=0).split(X)),
        "⚠️ 按日期分组 KFold": run(GroupKFold(N_SPLITS).split(X, groups=groups)),
        "✅ 按日期 + purge 1 期 + embargo 1 期": run(mt.purged_date_splits(groups, N_SPLITS, horizon=1, embargo=1)),
    }
    df = pd.DataFrame(rows).T.astype({"覆盖样本": int})
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    cs_note = ("本次逐期 RankIC 也是随机按行最高：同日同类股票的共振同样被'背'了下来。"
               if df.iloc[0, 1] > df.iloc[1:, 1].max() else
               "本次逐期 RankIC 没有被随机按行切分抬高：这里的泄漏主要体现在'猜大盘'上。")
    print(f"""
    💡 混合相关：随机按行 {df.iloc[0, 0]:+.3f}，按日期分组 {df.iloc[1, 0]:+.3f}，purged {df.iloc[2, 0]:+.3f}。
       逐期 RankIC：随机按行 {df.iloc[0, 1]:+.3f}，按日期分组 {df.iloc[1, 1]:+.3f}，purged {df.iloc[2, 1]:+.3f}。
       随机按行切分时，模型学到的是"那一天大盘涨了多少"，这在实盘中不可能事先知道（混合相关被大幅抬高）。
       {cs_note}
       月度标签期限 = 1 期：t 期标签延伸到 t+1 期，所以测试块前后各 purge 1 期，再 embargo 1 期。
       GroupKFold 不保证测试块在时间上连续（sklearn 按组大小分配），本例只用来演示"按天分组"的效果。
    """)
    return df


# ==================== 第五部分：K-Fold、Walk-Forward 与 CPCV ====================


def cv_choices():
    print("\n" + "=" * 60)
    print("第五部分：Purged K-Fold、Walk-Forward 与 CPCV")
    print("=" * 60)
    print("""
    ┌──────────────────┬──────────────────────────────┬──────────────────────┬────────────────────────┐
    │ 方法             │ 做法                         │ 优点                 │ 缺点                   │
    ├──────────────────┼──────────────────────────────┼──────────────────────┼────────────────────────┤
    │ Walk-Forward     │ 只用过去训练、未来测试       │ 最贴近实盘           │ 只有一条路径；早期     │
    │                  │                              │                      │ 测试段训练数据少       │
    │ Purged K-Fold    │ 测试块可在中间，删除重叠样本 │ 每个样本都被测一次   │ 训练集含"未来"数据，   │
    │                  │                              │                      │ 强非平稳时偏乐观       │
    │ CPCV（AFML 12 章）│ N 组中取 k 组测试，组合出    │ 得到回测表现的分布   │ 计算量大               │
    │                  │ 多条回测路径                 │                      │                        │
    └──────────────────┴──────────────────────────────┴──────────────────────┴────────────────────────┘

    ✅ 实践建议：
       1. 超参数调优：Purged K-Fold 或 CPCV（训练集内部）
       2. 最终评估：严格 Walk-Forward 样本外（05 课）
       3. 保留一段从未碰过的最终测试期，研究结束时只用一次
       4. sklearn 的 TimeSeriesSplit 有 gap 参数，可当作简化版 purge；但它不知道每个样本的 t1
    """)


def plot_splits(t1):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    n = len(t1)
    for k, (tr, te) in enumerate(mt.PurgedKFold(N_SPLITS, t1, 0.01).split(np.zeros(n))):
        state = np.full(n, 0.5)  # 被删除
        state[tr], state[te] = 0.0, 1.0
        ax.scatter(t1.index, np.full(n, k), c=state, cmap="coolwarm", s=2, marker="|", vmin=0, vmax=1)
    ax.set_yticks(range(N_SPLITS), [f"折 {k + 1}" for k in range(N_SPLITS)])
    ax.set_title("Purged K-Fold：蓝 = 训练，红 = 测试，灰 = purge / embargo 删除")
    path = save_figure(fig, "04_purged_kfold.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    why_leak()
    close, X, t1 = noise_experiment()
    purge_anatomy(close, X, t1)
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场")
    panel_experiment(data)
    cv_choices()
    plot_splits(t1)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第二部分的标签期限 H 改为 1、5、60，打乱 K-Fold 的虚高 AUC 如何变化？为什么？
    2. 去掉噪声特征的 60 日平滑（改成白噪声），打乱 K-Fold 还会泄漏吗？说明泄漏需要哪两个条件
    3. 实现 CPCV：把日期分成 6 组、每次取 2 组测试，共 C(6,2) = 15 种划分，拼出 5 条完整的回测路径
    4. 在第四部分中去掉两个市场状态特征，三种切分的"混合相关"差距还剩多少？
    """)


if __name__ == "__main__":
    main()
