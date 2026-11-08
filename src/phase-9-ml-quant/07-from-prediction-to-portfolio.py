"""
从预测分数到持仓
================

学习目标：
    1. 把模型分数变成持仓：TopK 等权、TopK-Dropout、分数加权、换手约束、分数平滑
    2. 在月度调仓回测中计入佣金、滑点、印花税（与 Phase 7 第 06 课口径一致），比较换手与扣费收益
    3. 理解 Grinold 公式 α = IC × σ × z：分数如何变成优化器需要的预期收益
    4. 亲眼看到"准确率提高 ≠ 收益提高"：受控实验 + 真实数据上的多模型对比

核心概念：
    - TopK-Dropout（Qlib 默认策略）：每期只替换持仓中最差的 n_drop 只，换手 ≈ n_drop / topk
    - 换手约束：w = w_prev + λ(w* − w_prev)，λ = min(1, 上限 / 实际换手)
    - 分数平滑：s̃_t = λ·s̃_{t−1} + (1−λ)·s_t，牺牲及时性换低换手
    - 评估层级：RankIC → 分组单调性 → 扣费 TopK 组合 → 风格归因

前置知识：
    - 05-lightgbm-stock-prediction.py；Phase 7 第 06 课（分层回测与成本）；Phase 8 第 10 课（换手）

环境要求：
    - pip install pandas numpy scikit-learn matplotlib（lightgbm 可选）
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
from factor_tools import COMMISSION, SLIPPAGE
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TOPK, N_DROP = 20, 4
RETRAIN = 3  # 本课每 3 个月重训一次，控制运行时间
SEED = 42


class ProbaModel:
    """把分类器包装成"predict 返回 P(y=1)"，以便复用 walk_forward"""

    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


# ==================== 第一部分：映射方式与 Grinold 公式 ====================


def mapping_overview(data, pred):
    print("=" * 60)
    print("第一部分：预测分数 → 持仓的几种方式")
    print("=" * 60)
    print(f"""
    ┌──────────────────────────────────────────────────────────────────────┐
    │ ① TopK 等权       选分数最高的 K 只，等权买入（最简单、最常用）      │
    │ ② TopK-Dropout    持有 K 只，每期只替换最差的 n 只（控制换手）       │
    │ ③ 分数加权        Top K 内按排名加权 w ∝ K, K−1, …, 1                │
    │ ④ 多空分位        做多前 10%、做空后 10%（A 股融券受限，常用期货对冲）│
    │ ⑤ 优化器          分数 → 预期收益 α → Phase 8 的约束优化             │
    └──────────────────────────────────────────────────────────────────────┘
    本课：K = {TOPK}（可交易池每期约 {len(data.X) / len(data.dates):.0f} 只），n_drop = {N_DROP}；月度调仓，t+1 开盘成交

    优化器需要"预期收益的尺度"，而不是任意分数。Grinold 公式：
        α_i = IC × σ_i × z_i
        z_i：截面标准化后的分数；σ_i：股票的波动（预测期限对应）；IC：模型历史平均 IC
    """)
    ic = mt.daily_rank_ic(pred, data.ret)
    d = pred.index.get_level_values("datetime").unique()[-1]
    z = pred.xs(d, level="datetime")
    z = (z - z.mean()) / z.std()
    sigma = data.raw.xs(d, level="datetime")["vol_60"].reindex(z.index) * np.sqrt(21)
    ic_hist = ic.loc[:d].iloc[:-2].mean()  # d 与 d−1 期的 IC 在 d 日收盘时尚未实现，只用更早的
    alpha = (ic_hist * sigma * z).dropna()
    print(f"📋 {d.date()}：历史平均 IC = {ic_hist:+.3f}，月度波动 σ 中位数 = {sigma.median():.1%}")
    print(f"   α 分布：最高 {alpha.max():+.2%}，最低 {alpha.min():+.2%}，标准差 {alpha.std():.2%}（月度预期超额）")
    print("   💡 IC 越小，α 越被压缩向 0 —— 优化器不会因为弱信号去冒大风险")


# ==================== 第二部分：组合构建与扣费回测 ====================


def build_targets(pred: pd.Series, method: str, smooth: float | None = None) -> dict:
    dates = pred.index.get_level_values("datetime").unique()
    targets, held, prev_s = {}, [], None
    for d in dates:
        s = pred.xs(d, level="datetime")
        if smooth is not None:  # EMA 平滑：新上榜的股票没有历史分数，直接用当期分数
            s = s.rank(pct=True)
            if prev_s is not None:
                s = smooth * prev_s.reindex(s.index).fillna(s) + (1 - smooth) * s
            prev_s = s
        if method == "dropout":
            held = mt.topk_dropout(s, held, TOPK, N_DROP)
            targets[d] = pd.Series(1.0 / len(held), index=held)
        elif method == "weighted":
            targets[d] = mt.score_weighted(s, TOPK)
        else:
            targets[d] = mt.topk_equal(s, TOPK)
    return targets


def summarize(bt: pd.DataFrame, bench: pd.Series) -> dict:
    net, gross = bt["net"], bt["gross"]
    ex = net - bench.reindex(net.index)
    nav = (1 + net).cumprod()
    return {"年化毛收益": gross.mean() * 12, "年化净收益": net.mean() * 12, "年化净超额": ex.mean() * 12,
            "超额 t": ex.mean() / ex.std() * np.sqrt(len(ex)), "净夏普": net.mean() / net.std() * np.sqrt(12),
            "最大回撤": (nav / nav.cummax() - 1).min(), "月均换手": bt["turnover"].iloc[1:].mean(),
            "年化成本": bt["cost"].iloc[1:].mean() * 12}


def portfolio_backtest(data, pred):
    print("\n" + "=" * 60)
    print("第二部分：组合构建与扣费回测")
    print("=" * 60)
    print(f"""
    成本（示例值，与 Phase 7 一致）：佣金 {COMMISSION:.3%} 双边 + 滑点 {SLIPPAGE:.2%} 双边 + 印花税（仅卖出，
    2023-08-28 前 0.1%、之后 0.05%）；首期建仓只计买入成本；持有期内权重随价格漂移。
    基准：可交易池等权组合（未扣成本，月度再平衡）
    """)
    bench = data.ret_wide.where(data.mask_wide).mean(axis=1)
    strategies = {
        "① TopK 等权": (build_targets(pred, "topk"), None),
        "② TopK-Dropout": (build_targets(pred, "dropout"), None),
        "③ 分数加权": (build_targets(pred, "weighted"), None),
        "④ TopK + 换手上限 25%": (build_targets(pred, "topk"), 0.25),
        "⑤ 分数 EMA 平滑(0.5) + TopK": (build_targets(pred, "topk", smooth=0.5), None),
    }
    results, rows = {}, {}
    for name, (targets, cap) in strategies.items():
        bt = mt.backtest_weights(targets, data.ret_wide, max_turnover=cap)
        results[name] = bt
        rows[name] = summarize(bt, bench)
    df = pd.DataFrame(rows).T
    print(f"📋 {mt.gbdt_backend()} 样本外分数，{len(bt)} 个月：")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    lo, hi = df["月均换手"].idxmin(), df["月均换手"].idxmax()
    print(f"""
    💡 换手最低：{lo}（{df.loc[lo, '月均换手']:.1%}/月），最高：{hi}（{df.loc[hi, '月均换手']:.1%}/月）；
       年化成本从 {df['年化成本'].min():.2%} 到 {df['年化成本'].max():.2%}。
       TopK 等权每月换掉 {df.loc['① TopK 等权', '月均换手']:.0%} 的持仓 —— ML 分数逐月变化大，控制换手是把预测变成收益的关键一步。
       各方法超额 t 值的最大绝对值为 {df['超额 t'].abs().max():.2f}：{len(bt)} 个月、约 130 只股票的池子里，
       方法之间的差别很可能落在统计误差之内，不要据此认定哪种构建方式"最好"。
    """)
    return results, bench


# ==================== 第三部分：准确率 ≠ 收益（受控实验） ====================


def accuracy_vs_return_synthetic():
    print("\n" + "=" * 60)
    print("第三部分：准确率提高 ≠ 收益提高 —— 受控实验")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    T, N = 60, 200
    big = rng.random((T, N)) < 0.05  # 5% 的股票当月有大事件（±20%）
    r = np.where(big, rng.choice([-0.2, 0.2], (T, N)), rng.normal(0, 0.02, (T, N)))
    sign = np.sign(r)

    def model(acc_small, acc_big):
        """在普通股票上方向正确率 acc_small，在大事件股票上 acc_big；分数 = 方向 × |随机置信度|"""
        right = np.where(big, rng.random((T, N)) < acc_big, rng.random((T, N)) < acc_small)
        direction = np.where(right, sign, -sign)
        return direction * rng.uniform(0.5, 1.0, (T, N))

    rows = {}
    for name, (a_s, a_b) in {"模型 A：小涨小跌猜得准，大事件猜反": (0.60, 0.30),
                              "模型 B：小涨小跌接近瞎猜，大事件猜对": (0.51, 0.70)}.items():
        s = model(a_s, a_b)
        acc = np.mean(np.sign(s) == sign)
        top = np.argsort(-s, axis=1)[:, :TOPK]
        bot = np.argsort(s, axis=1)[:, :TOPK]
        ls = np.take_along_axis(r, top, 1).mean(1) - np.take_along_axis(r, bot, 1).mean(1)
        rows[name] = {"方向准确率": acc, f"Top{TOPK}−Bottom{TOPK} 月均": ls.mean(),
                      "多空夏普(年化)": ls.mean() / ls.std() * np.sqrt(12)}
    df = pd.DataFrame(rows).T
    print(f"   合成截面：{T} 个月 × {N} 只股票；95% 的股票月收益 ~ N(0, 2%)，5% 有 ±20% 的大事件（固定种子）\n")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    💡 准确率把"猜对一只涨 0.5% 的股票"和"猜对一只跌 20% 的股票"算作同等功劳；
       组合收益却由少数大幅波动的股票决定。这是 CONCEPT 5.6 节第 1 条原因（忽略幅度）的直接演示。
    """)


# ==================== 第四部分：准确率 ≠ 收益（真实数据） ====================


def accuracy_vs_return_real(data, pred_gbdt, bench):
    print("\n" + "=" * 60)
    print("第四部分：准确率 vs 扣费收益 —— 真实面板上的多模型对比")
    print("=" * 60)
    y_up = (data.y > 0).astype(int)  # 是否跑赢当月截面中位数
    clean, p7 = mt.phase7_composite(data)
    models = {
        "GBDT 回归（rank 标签）": pred_gbdt,
        "GBDT 分类（是否跑赢中位数）": mt.walk_forward(
            data.X, y_up, lambda: ProbaModel(mt.make_classifier(seed=SEED)), retrain_every=RETRAIN),
        "Ridge 回归": mt.walk_forward(data.X, data.y, lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
                                    retrain_every=RETRAIN),
        "P7 等权（线性）": p7.fillna(0.0).mean(axis=1),
        "单因子：1 月反转": -data.X["ret_1m"],
    }
    common = pred_gbdt.index
    rows = {}
    for name, p in models.items():
        p = p.reindex(common)
        hit = (p.groupby(level="datetime").rank(pct=True) > 0.5) == (y_up.reindex(common) == 1)
        bt = mt.backtest_weights(build_targets(p, "topk"), data.ret_wide)
        s = summarize(bt, bench)
        rows[name] = {"上下半区准确率": hit.mean(), "RankIC": mt.daily_rank_ic(p, data.ret).mean(),
                      "Top20 净超额": s["年化净超额"], "月均换手": s["月均换手"],
                      "持仓市值暴露": data.X.loc[common, "log_mcap"].where(
                          p.groupby(level="datetime").rank(ascending=False) <= TOPK).mean()}
    df = pd.DataFrame(rows).T
    print(f"   样本外 {common.get_level_values(0)[0].date()} ~ {common.get_level_values(0)[-1].date()}；"
          f"'持仓市值暴露' = Top20 持仓的 log_mcap 截面 rank 均值（负 = 偏小盘，范围 ±0.5）\n")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    ra, rr = df["上下半区准确率"].rank(ascending=False), df["Top20 净超额"].rank(ascending=False)
    rho = ra.corr(rr, method="spearman")
    print(f"""
    按准确率排名：{' > '.join(ra.sort_values().index)}
    按净超额排名：{' > '.join(rr.sort_values().index)}
    两种排名的 Spearman 相关 = {rho:+.2f}；准确率最高的"{ra.idxmin()}"在净超额上排第 {int(rr[ra.idxmin()])} 名

    💡 准确率只看"有没有分到正确的半区"，而 Top20 组合只关心最顶端 20 只的收益、还要付换手成本；
       另外要看风格暴露：各模型 Top20 持仓的市值暴露在 {df['持仓市值暴露'].min():+.2f} ~ {df['持仓市值暴露'].max():+.2f} 之间，
       负值意味着偏小盘 —— 这部分收益可能是风格 beta 而非 alpha，需要做归因（Phase 8 第 09 课）。
    """)


def checklist():
    print("\n" + "=" * 60)
    print("第五部分：上线前检查清单")
    print("=" * 60)
    print("""
    ☐ 排序前剔除停牌、t+1 开盘涨停（买不进）、ST、上市不满一年的股票（本课的可交易掩码已做）
    ☐ 用 t+1 开盘价或 VWAP 成交，而不是信号当天的收盘价（本课已做）
    ☐ 跌停 / 停牌时卖不出：持仓被动保留（本课简化为可卖出，Phase 7 第 06 课有逐笔模拟）
    ☐ 计入完整成本：佣金、印花税、滑点、冲击成本随资金规模增大
    ☐ 与线性多因子在同一池子、同一成本、同一构建方式下比较
    ☐ 做风格归因（Phase 8 第 09 课）：超额来自 alpha 还是小盘 / 低波等风格暴露
    """)


def plot_nav(results, bench):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, bt in results.items():
        ax.plot(bt.index, (1 + bt["net"]).cumprod(), label=name, lw=1)
    b = bench.reindex(next(iter(results.values())).index)
    ax.plot(b.index, (1 + b).cumprod(), label="可交易池等权（未扣费）", color="black", ls="--", lw=1)
    ax.set_title(f"扣费净值（{mt.gbdt_backend()} 分数，月度调仓）")
    ax.legend(fontsize=8)
    path = save_figure(fig, "07_portfolio_nav.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    pred = mt.walk_forward(data.X, data.y, lambda: mt.make_regressor(seed=SEED), retrain_every=RETRAIN)
    mapping_overview(data, pred)
    results, bench = portfolio_backtest(data, pred)
    accuracy_vs_return_synthetic()
    accuracy_vs_return_real(data, pred, bench)
    checklist()
    plot_nav(results, bench)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 TOPK 改为 10、30、50，N_DROP 改为 2、8，画出"月均换手 vs 年化净超额"的散点图
    2. 把滑点从 0.1% 提高到 0.3%（中小盘更现实），哪种构建方式受影响最大？
    3. 用第一部分的 α = IC·σ·z 作为预期收益，接入 Phase 8 第 06 课的 cvxpy 优化器（换手惩罚 + 行业约束）
    4. 对 Meta-Labeling（03 课）的概率输出做 bet sizing，与本课的 TopK 等权比较
    """)


if __name__ == "__main__":
    main()
