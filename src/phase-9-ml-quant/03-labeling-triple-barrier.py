"""
标签：Triple-Barrier 与 Meta-Labeling
=====================================

学习目标：
    1. 认识固定期限标签 sign(r_{t,t+h}) 的三个问题：忽略路径、阈值不随波动调整、0 附近的噪声
    2. 实现 Triple-Barrier：波动率自适应的止盈 / 止损栅栏 + 时间栅栏，并记录每个样本的结束时间 t1
    3. 用 CUSUM 滤波器挑选事件；用"平均唯一度"给重叠样本降权
    4. 实现 Meta-Labeling：主模型定方向，Meta 模型决定"要不要下注、下多大"

核心概念：
    - 上栅栏 p₀(1 + pt·σ_t)，下栅栏 p₀(1 − sl·σ_t)，时间栅栏 t₀ + h；标签 = 最先碰到哪道栅栏
    - t1：标签的结束时间，是 04 课 Purged CV 和样本权重的关键
    - 平均唯一度：ū_i = mean_{t∈[t0,t1]} 1/c_t，c_t 为时刻 t 的并发标签数
    - Meta 标签：y = 1[side × r > 0]；仓位 m = 2Φ(z) − 1，z = (p − 0.5)/√(p(1−p))

前置知识：
    - Phase 5 CTA（均线交叉）、Phase 4 止盈止损
    - 01-ml-quant-overview.py、02-feature-engineering.py

环境要求：
    - pip install pandas numpy scipy scikit-learn matplotlib（lightgbm 可选）
    - 使用沪深 300 日线；QTC_OFFLINE=1 时优先读本地缓存，缓存缺失才用合成数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import ml_tools as mt
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

H = 10  # 时间栅栏：最长持有 10 个交易日
PT, SL = 1.0, 1.0  # 止盈 / 止损倍数（单位：h 日波动 σ_t·√h）
SEED = 42


def daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """EWM 日收益波动率：t 日的值只用到 t 及以前的收益（因果）"""
    return close.pct_change().ewm(span=span, min_periods=span).std()


# ==================== 第一部分：固定期限标签的问题 ====================


def fixed_horizon_problems(close: pd.Series, vol: pd.Series):
    print("=" * 60)
    print("第一部分：固定期限标签的问题")
    print("=" * 60)
    print(f"""
    最常见的标签：y_t = sign(r_(t → t+h))，这里 h = {H} 天。问题：
       ① 忽略路径：中途先跌 8% 再涨回 +1%，标签为正 —— 现实中你早被止损了
       ② 阈值不随波动调整：同样 2% 的涨幅，对低波动时期是大事件，对高波动时期是噪声
       ③ 大量 |r| 很小的样本，方向几乎是随机的
    """)
    c = close.to_numpy()
    n = len(c)
    rows = []
    for i in range(n - H):
        path = c[i + 1:i + H + 1] / c[i] - 1
        rows.append((path[-1], path.min(), path.max()))
    df = pd.DataFrame(rows, columns=["end", "min", "max"], index=close.index[:n - H])
    s = (vol * np.sqrt(H)).reindex(df.index)
    pos = df["end"] > 0
    hit_dn = df["min"] <= -SL * s
    print(f"📋 沪深 300 共 {len(df)} 个样本：")
    print(f"   ① 终点收益为正、但途中先触及 −{SL:.0f}σ 止损线的样本：{(pos & hit_dn).sum()} 个，"
          f"占正标签的 {(pos & hit_dn).sum() / pos.sum():.1%}")
    thr = 0.02
    yr = df.index.year
    share = (df["end"].abs() < thr).groupby(yr).mean()
    vol_y = vol.reindex(df.index).groupby(yr).mean() * np.sqrt(252)
    t = pd.DataFrame({"年化波动": vol_y, f"|r|<{thr:.0%} 的样本占比": share})
    print(f"   ② 固定阈值 {thr:.0%}：落在阈值内（被当作'无信号'）的样本占比按年份变化（每隔一年展示）：")
    print(t.iloc[::2].to_string(float_format=lambda v: f"{v:.1%}"))
    print(f"      各年占比在 {share.min():.1%} ~ {share.max():.1%} 之间：同一个阈值在不同年份的含义并不相同")
    print(f"   ③ |r| < 0.5% 的样本占比：{(df['end'].abs() < 0.005).mean():.1%} —— 这些样本的方向基本是噪声")


# ==================== 第二部分：Triple-Barrier ====================


def triple_barrier(close: pd.Series, events: pd.DatetimeIndex, trgt: pd.Series, side: pd.Series | None = None,
                   pt: float = PT, sl: float = SL, h: int = H) -> pd.DataFrame:
    """
    events: 事件开始时间 t0；trgt: 每个 t0 的栅栏宽度（收益率单位，只用 t0 及以前的数据）
    side:   主模型方向（None = 只做多 / 对称标签）；止盈止损按 side 调整后的收益判断
    返回 index = t0，列 t1（结束时间）、ret（side 调整后的收益）、label、barrier
    """
    c = close.to_numpy()
    idx = close.index
    pos = idx.get_indexer(events)
    rows = []
    for t0, i0 in zip(events, pos):
        w = trgt.get(t0, np.nan)
        if not np.isfinite(w) or i0 + 1 >= len(c):
            continue
        s = 1.0 if side is None else float(side.loc[t0])
        i_end = min(i0 + h, len(c) - 1)
        path = s * (c[i0 + 1:i_end + 1] / c[i0] - 1)
        up = np.flatnonzero(path >= pt * w)
        dn = np.flatnonzero(path <= -sl * w)
        k_up = up[0] if len(up) else np.inf
        k_dn = dn[0] if len(dn) else np.inf
        k = min(k_up, k_dn, len(path) - 1)
        barrier = "pt" if k == k_up else ("sl" if k == k_dn else "vb")
        ret = path[int(k)]
        label = {"pt": 1, "sl": -1}.get(barrier, int(np.sign(ret)))
        rows.append((t0, idx[i0 + 1 + int(k)], ret, label, barrier, w))
    return pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier", "trgt"]).set_index("t0")


def triple_barrier_demo(close: pd.Series, vol: pd.Series):
    print("\n" + "=" * 60)
    print("第二部分：Triple-Barrier（AFML 第 3 章）")
    print("=" * 60)
    print(f"""
      价格
       ▲  ─────────────────────────────── 上栅栏 p₀·(1 + pt·σ_t)   → +1（止盈）
       │            ╱╲      ╱╲
     p₀●──╱╲╱╲──╱  ╲╱╲╱  ╲
       │                     ╲  ╱
       │  ─────────────────────────────── 下栅栏 p₀·(1 − sl·σ_t)   → −1（止损）
       └──────────────────────────────┼──▶ 时间
       t₀                           t₀+h   时间栅栏 → sign(收益)

    本课：σ_t = EWM(20) 日波动 × √h（h 日波动），pt = {PT}，sl = {SL}，h = {H}
    """)
    trgt = vol * np.sqrt(H)
    events = close.index[trgt.notna().to_numpy()][:-1]
    tb = triple_barrier(close, events, trgt)
    fixed = np.sign(close.shift(-H) / close - 1).reindex(tb.index)
    hold = close.index.get_indexer(tb["t1"]) - close.index.get_indexer(tb.index)
    print(f"📋 每天一个事件，共 {len(tb)} 个：")
    print(pd.DataFrame({
        "样本占比": tb["barrier"].value_counts(normalize=True),
        "平均收益": tb.groupby("barrier")["ret"].mean(),
        "平均持有天数": pd.Series(hold, index=tb.index).groupby(tb["barrier"]).mean(),
    }).rename(index={"pt": "止盈 pt", "sl": "止损 sl", "vb": "时间 vb"}).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n   标签分布：{tb['label'].value_counts().sort_index().to_dict()}")
    print(f"   与固定期限标签 sign(r_t→t+{H}) 一致的比例：{(tb['label'] == fixed).mean():.1%}")
    yr = tb.index.year
    bal = pd.DataFrame({"TB 正标签占比": (tb["label"] > 0).groupby(yr).mean(),
                        "触碰水平栅栏占比": (tb["barrier"] != "vb").groupby(yr).mean()})
    print("\n📋 按年份（每隔一年展示）：栅栏宽度随 σ_t 缩放")
    print(bal.iloc[::2].to_string(float_format=lambda v: f"{v:.1%}"))
    print(f"""
    💡 这组参数（对称、1 倍 h 日波动）下，TB 标签与固定期限标签的方向大多一致；
       差别在于：① 路径上先触及止损的样本被标为 −1；② 每个样本有自己的结束时间 t1（持有天数不同）。
       t1 是后面计算样本唯一度、做 Purged CV 的基础 —— 这才是 TB 在方法论上最重要的产出。""")
    return tb


# ==================== 第三部分：样本唯一度 ====================


def avg_uniqueness(index: pd.DatetimeIndex, t1: pd.Series) -> tuple[pd.Series, pd.Series]:
    """并发数 c_t 与平均唯一度 ū_i（AFML 第 4 章）"""
    pos0 = index.get_indexer(t1.index)
    pos1 = index.get_indexer(t1.to_numpy())
    diff = np.zeros(len(index) + 1)
    np.add.at(diff, pos0, 1)
    np.add.at(diff, pos1 + 1, -1)
    conc = np.cumsum(diff)[:-1]
    inv = np.r_[0, np.cumsum(1 / np.where(conc > 0, conc, np.inf))]
    u = (inv[pos1 + 1] - inv[pos0]) / (pos1 - pos0 + 1)
    return pd.Series(conc, index=index), pd.Series(u, index=t1.index)


def uniqueness_demo(close: pd.Series, tb: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：重叠标签与样本权重")
    print("=" * 60)
    print("""
    样本 i 的标签区间 [t0_i, t1_i] 与其他样本重叠时，它们共享信息：

        c_t = Σ_i 1[t ∈ [t0_i, t1_i]]                 时刻 t 有多少个标签"活着"
        ū_i = 1/(t1_i − t0_i + 1) · Σ_t 1/c_t          样本 i 的平均"独占程度" ∈ (0, 1]

    用法：① 作为训练的 sample_weight；② 随机森林 max_samples ≈ mean(ū)；③ 顺序自助法（Sequential Bootstrap）
    """)
    conc, u = avg_uniqueness(close.index, tb["t1"])
    print(f"   平均并发数 c̄ = {conc[conc > 0].mean():.2f}，平均唯一度 ū = {u.mean():.3f}")
    print(f"   {len(tb)} 个名义样本 ≈ {u.sum():.0f} 个'等效独立样本'（Σ ū_i）")
    print(f"   ū 的分位数：10% {u.quantile(0.1):.3f} / 50% {u.median():.3f} / 90% {u.quantile(0.9):.3f}")
    print("   💡 止损很快触发的样本区间短、重叠少 → 唯一度高；拖到时间栅栏的样本 → 唯一度低")
    return u


# ==================== 第四部分：CUSUM 事件 + Meta-Labeling ====================


def cusum_filter(close: pd.Series, h: pd.Series) -> pd.DatetimeIndex:
    """对称 CUSUM 滤波：累计对数收益偏离超过阈值 h_t 时触发事件并清零（h_t 用 t 时的波动）"""
    r = np.log(close).diff().to_numpy()
    hv = h.to_numpy()
    s_pos = s_neg = 0.0
    out = []
    for i in range(1, len(r)):
        if not np.isfinite(hv[i]) or not np.isfinite(r[i]):
            continue
        s_pos, s_neg = max(0.0, s_pos + r[i]), min(0.0, s_neg + r[i])
        if s_pos > hv[i] or s_neg < -hv[i]:
            out.append(close.index[i])
            s_pos = s_neg = 0.0
    return pd.DatetimeIndex(out)


def meta_features(close: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """Meta 模型的特征（均只用 t 及以前的数据）"""
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    r = close.pct_change()
    return pd.DataFrame({
        "ma_spread": (ma20 / ma60 - 1) / (vol * np.sqrt(20)),
        "ret_5": close.pct_change(5),
        "ret_20": close.pct_change(20),
        "vol_20": vol,
        "vol_ratio": r.rolling(20).std() / r.rolling(120).std(),
        "dist_high": close / close.rolling(250).max() - 1,
    })


def bet_size(p: np.ndarray) -> np.ndarray:
    """AFML 第 10 章：m = 2Φ(z) − 1，z = (p − 0.5)/√(p(1−p))"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 2 * norm.cdf((p - 0.5) / np.sqrt(p * (1 - p))) - 1


def meta_labeling(close: pd.Series, vol: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：CUSUM 事件 + Meta-Labeling")
    print("=" * 60)
    print("""
    主模型（侦察兵）：20/60 日均线，side = sign(MA20 − MA60) —— 追求"高召回"，不漏机会
    Meta 模型（指挥官）：看主信号 + 市场状态，预测 P(这笔交易赚钱)，决定是否出击、派多少兵

       ① CUSUM 滤波挑事件（只在价格"有动静"时采样，减少冗余样本）
       ② 按 side 打 Triple-Barrier；Meta 标签 y = 1[side × r > 0]
       ③ Meta 分类器用特征预测 P(y = 1)；训练 / 测试按时间切分并 purge
    """)
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    side = np.sign(ma20 - ma60)
    events = cusum_filter(close, vol * 2)
    feats = meta_features(close, vol)
    ok = side.reindex(events).ne(0) & feats.reindex(events).notna().all(axis=1) & vol.reindex(events).notna()
    events = events[ok.to_numpy()]
    tb = triple_barrier(close, events, vol * np.sqrt(H), side=side, pt=2.0, sl=1.0)
    y = (tb["ret"] > 0).astype(int)
    X = feats.loc[tb.index].assign(side=side.loc[tb.index])
    _, u = avg_uniqueness(close.index, tb["t1"])
    print(f"   CUSUM 事件 {len(tb)} 个；主模型方向：做多 {(X['side'] > 0).sum()}，做空 {(X['side'] < 0).sum()}")
    print(f"   止盈 2σ / 止损 1σ（不对称：顺势单让利润奔跑）")

    split = tb.index[int(len(tb) * 0.6)]
    tr = (tb.index < split) & (tb["t1"] < split).to_numpy()  # purge：训练标签不能延伸进测试期
    te = tb.index >= split
    print(f"   训练 {tr.sum()} 个（< {split.date()}，已删除 {((tb.index < split) & ~tr).sum()} 个标签跨入测试期的样本），"
          f"测试 {te.sum()} 个")
    clf = mt.make_classifier(seed=SEED, n_estimators=150, num_leaves=7, min_child_samples=20, learning_rate=0.03)
    clf.fit(X[tr], y[tr], sample_weight=u[tr].to_numpy())
    p = clf.predict_proba(X[te])[:, 1]
    yt, rt = y[te].to_numpy(), tb.loc[te, "ret"].to_numpy()
    take = p > 0.5
    m = bet_size(p)

    def stats(sel, w=None):
        w = np.ones(sel.sum()) if w is None else w[sel]
        r = rt[sel] * w
        prec = yt[sel].mean() if sel.sum() else np.nan
        rec = (yt[sel] == 1).sum() / max(yt.sum(), 1)
        return {"交易数": int(sel.sum()), "精确率": prec, "召回率": rec,
                "F1": 2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan,
                "平均每笔收益": r.mean() if len(r) else np.nan, "累计收益(求和)": r.sum()}

    rows = {"主模型：全部执行": stats(np.ones(len(yt), dtype=bool)),
            "Meta 过滤：p > 0.5": stats(take),
            "Meta 仓位：m = 2Φ(z)−1": stats(m > 0, m)}
    print(f"\n📋 测试期（{split.date()} 之后），Meta 模型 AUC = {roc_auc_score(yt, p):.3f}：")
    print(pd.DataFrame(rows).T.astype({"交易数": int}).to_string(float_format=lambda v: f"{v:.3f}"))
    auc = roc_auc_score(yt, p)
    verdict = ("本次 AUC 接近（或低于）0.5：这些特征对'这笔交易能否赚钱'几乎没有区分力，过滤不会带来稳定改善"
               if auc < 0.55 else "本次 AUC 高于 0.55：特征有一定区分力，可以继续检验其稳健性")
    print(f"""
    💡 读表：
       - 主模型的精确率 = 测试期里"顺着均线方向交易能赚钱"的比例
       - Meta 过滤的目标是提高精确率（牺牲召回）
       - {verdict}
       - 只有一个指数、几百个事件，测试期样本很少，以上数字的统计误差很大 —— 本节重在方法
       - Meta-Labeling 可以叠加在任何非 ML 的主策略上（Phase 5 的 CTA），用 ML 过滤信号而不是替代策略逻辑
    """)
    return tb, p


def plot_example(close: pd.Series, vol: pd.Series, tb_all: pd.DataFrame):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    seg = close.iloc[-260:]
    ax.plot(seg.index, seg, lw=0.9, color="black")
    sample = tb_all.loc[tb_all.index.isin(seg.index)].iloc[::25]
    colors = {"pt": "tab:red", "sl": "tab:green", "vb": "tab:gray"}
    for t0, row in sample.iterrows():
        p0, w = close.loc[t0], row["trgt"]
        ax.hlines([p0 * (1 + PT * w), p0 * (1 - SL * w)], t0, row["t1"], colors=["tab:red", "tab:green"], lw=0.8)
        ax.scatter([row["t1"]], [close.loc[row["t1"]]], color=colors[row["barrier"]], s=18, zorder=3)
    ax.set_title("Triple-Barrier 示例：红 = 止盈线，绿 = 止损线，点 = 结束时刻（颜色为触碰的栅栏）")
    path = save_figure(fig, "03_triple_barrier.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    idx = get_index_daily("000300", "20100101", "20251231")
    if is_synthetic(idx):
        print("⚠️ 沪深 300 使用合成数据：数字只用于跑通流程\n")
    close = idx["close"]
    vol = daily_vol(close)
    fixed_horizon_problems(close, vol)
    tb = triple_barrier_demo(close, vol)
    uniqueness_demo(close, tb)
    meta_labeling(close, vol)
    plot_example(close, vol, tb)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 pt / sl 改为 (2, 1)、(1, 2)，标签分布和平均持有天数如何变化？对应什么样的交易风格？
    2. 把主模型换成 Phase 5 的唐奇安通道突破，重做 Meta-Labeling，比较精确率提升
    3. 实现 AFML 的顺序自助法（Sequential Bootstrap），比较它与普通自助法抽出样本的平均唯一度
    4. 把 Meta 模型的输出 m 离散化为 0、0.25、0.5、0.75、1 五档，比较仓位变化次数与收益
    """)


if __name__ == "__main__":
    main()
