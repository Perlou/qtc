"""
金融机器学习的特殊性
====================

学习目标：
    1. 理解金融 ML 与 CV / NLP 的根本差异：低信噪比、非平稳、非 IID、对抗性
    2. 用数字感受"IC 0.05 已经很好"：完美预测信号部分时，相关性的理论上限
    3. 在真实数据上观察三种非 IID：标签重叠、截面相关、分布漂移
    4. 看到模型容量越大，样本内外差距越大 —— 本阶段所有方法论的出发点

核心概念：
    - 信噪比：r = s + ε，ρ(s, r) = σ_s / √(σ_s² + σ_ε²)
    - 非 IID：重叠标签 → 相邻样本相关；截面相关 → 3000 个样本 ≠ 3000 个独立观测
    - 有效样本数：N_eff ≈ N / (1 + (N−1)·ρ̄)
    - López de Prado："Backtesting is not a research tool. Feature importance is."

前置知识：
    - Phase 4（回测陷阱、Walk-Forward）、Phase 7（因子、IC）
    - 机器学习基础（训练集 / 测试集、过拟合）

环境要求：
    - pip install pandas numpy scikit-learn matplotlib（lightgbm 可选）
    - 数据来自 Phase 7 共享面板（ml_tools.load_ml_data）；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import ml_tools as mt
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SEED = 42


# ==================== 第一部分：你熟悉的 ML vs 金融 ML ====================


def ml_vs_finance():
    print("=" * 60)
    print("第一部分：你熟悉的 ML vs 金融 ML")
    print("=" * 60)
    print("""
    ┌──────────────┬──────────────────────────┬──────────────────────────────────┐
    │ 维度         │ CV / NLP                 │ 金融收益预测                     │
    ├──────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 信噪比       │ 高（猫就是猫）           │ 极低（明天的收益大部分是噪声）   │
    │ 好模型的指标 │ 准确率 95%+              │ 截面 Rank IC 0.03–0.05 就不错    │
    │ 平稳性       │ 分布基本稳定             │ 非平稳：市场结构、政策持续变化   │
    │ 样本独立性   │ 大致 IID                 │ 标签重叠、自相关、截面相关       │
    │ 数据量       │ 可以继续采集、增强       │ 历史只有一条路径，无法重来       │
    │ 对抗性       │ 猫不会因被识别而改变     │ alpha 被发现后会被套利掉         │
    │ 评估         │ 测试集准确率             │ 扣成本后的样本外收益、回撤、容量 │
    └──────────────┴──────────────────────────┴──────────────────────────────────┘

    🎸 比喻：普通 ML 像在安静的房间里听人说话；金融 ML 像在摇滚演唱会现场
       听清 50 米外一个人的耳语 —— 而且他发现你在偷听后，会换一种语言。
    """)


# ==================== 第二部分：低信噪比的量化理解 ====================


def signal_to_noise():
    print("\n" + "=" * 60)
    print("第二部分：低信噪比 —— 为什么 IC 0.05 已经很好")
    print("=" * 60)
    print("""
    设收益 r = s + ε：s 是可预测部分，ε 是不可预测噪声。
    即使你"完美"地预测了 s，预测与实际收益的相关性也只有：

        ρ(s, r) = σ_s / √(σ_s² + σ_ε²)
    """)
    rng = np.random.default_rng(SEED)
    n = 200_000
    rows = []
    for sig_s in [0.0005, 0.001, 0.002, 0.005]:
        s = rng.normal(0, sig_s, n)
        r = s + rng.normal(0, 0.02, n)
        rows.append({
            "σ_s（信号）": f"{sig_s:.2%}", "σ_ε（噪声）": "2.00%",
            "理论 ρ": sig_s / np.sqrt(sig_s**2 + 0.02**2),
            "模拟 corr(s, r)": np.corrcoef(s, r)[0, 1],
            "方向准确率": np.mean(np.sign(s) == np.sign(r)),
        })
    print("📋 用'完美的'信号预测（模拟 20 万个样本，固定种子）：")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("""
    💡 读表：
       - 即使模型 100% 学会了可预测部分，方向准确率也只比 50% 高几个百分点
       - 模型容量稍大，就会去拟合占绝对多数的噪声 ε → 过拟合是金融 ML 的头号敌人
       - 所以本阶段评估指标是 Rank IC / ICIR，而不是准确率或 MSE（05 课）
    """)


# ==================== 第三部分：三种非 IID ====================


def non_iid(data):
    print("\n" + "=" * 60)
    print("第三部分：金融样本为什么不是 IID")
    print("=" * 60)

    # ① 标签重叠
    idx = get_index_daily("000300", "20100101", "20251231")
    tag = "（⚠️ 合成数据）" if is_synthetic(idx) else ""
    ret = idx["close"].pct_change()
    rows = []
    for h in [1, 5, 20]:
        fwd = idx["close"].shift(-h) / idx["close"] - 1  # 每天一个样本，标签 = 未来 h 日收益
        rows.append({"标签期限 h": h, "相邻样本标签自相关": fwd.autocorr(1),
                     "间隔 h 天的自相关": fwd.autocorr(h)})
    n_obs = int(ret.notna().sum())
    print(f"① 标签重叠：沪深 300 日线{tag}，每天一个样本，标签 = 未来 h 日收益")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"""
       h = 20 时，t 和 t+1 的标签共享 19 天收益 → 几乎是同一个样本。
       名义上 {n_obs} 个样本，不重叠的标签区间只有约 {n_obs} / 20 ≈ {n_obs // 20} 个。
       普通 K-Fold 把 t 放进训练、t+1 放进测试 → 等于"偷看答案"（04 课）。
    """)

    # ② 截面相关
    panel = data.panel
    r = panel.daily_ret.loc["2021":"2024"]
    corr = r.corr(min_periods=200).to_numpy()
    rho = np.nanmean(corr[np.triu_indices_from(corr, 1)])
    n = r.shape[1]
    n_eff = n / (1 + (n - 1) * rho)
    print(f"② 截面相关：面板 {n} 只股票 2021–2024 日收益的平均两两相关 ρ̄ = {rho:.3f}")
    print(f"   有效独立股票数 N_eff ≈ N / (1 + (N−1)·ρ̄) = {n_eff:.1f}（名义 {n} 只）")
    print("   → 同一天的股票不能随机分到训练集和测试集，必须按日期切分（04 课）")

    # ③ 非平稳：同一特征的 IC 逐年变化
    ics = {k: mt.daily_rank_ic(data.X[k], data.ret) for k in ["ret_1m", "log_mcap", "vol_20", "ep"]}
    yearly = pd.DataFrame({mt.FEATURE_LABELS[k]: v.groupby(v.index.year).mean() for k, v in ics.items()})
    print("\n③ 非平稳：几个特征的月度 Rank IC，按年份平均")
    print(yearly.to_string(float_format=lambda v: f"{v:+.3f}"))
    flips = {c: int((np.sign(yearly[c]).diff().abs() > 0).sum()) for c in yearly}
    print(f"   年度 IC 符号翻转次数：{flips}")
    print("   → 训练期学到的'规律'在测试期可能反过来；所以要滚动训练并监控漂移（10 课）")


# ==================== 第四部分：模型容量与过拟合 ====================


def capacity_vs_overfit(data):
    print("\n" + "=" * 60)
    print("第四部分：模型越复杂，样本内外差距越大")
    print("=" * 60)
    dates = data.dates
    split = len(dates) // 2
    lvl = data.date_index
    tr = lvl <= dates[split - 2]  # 与测试期之间空一期：最后一期训练标签要到测试首日之后才实现
    te = lvl >= dates[split]
    print(f"   训练：{dates[0].date()} ~ {dates[split - 2].date()}；测试：{dates[split].date()} ~ {dates[-1].date()}")
    print(f"   标签 = 下月收益的截面排名；模型 = {mt.gbdt_backend()}\n")
    configs = [
        ("极简：4 叶 / 叶≥300", dict(num_leaves=4, min_child_samples=300, n_estimators=100)),
        ("保守：15 叶 / 叶≥100", dict(num_leaves=15, min_child_samples=100, n_estimators=200)),
        ("中等：31 叶 / 叶≥30", dict(num_leaves=31, min_child_samples=30, n_estimators=300)),
        ("激进：63 叶 / 叶≥5", dict(num_leaves=63, min_child_samples=5, n_estimators=500, learning_rate=0.1)),
    ]
    rows = []
    for name, kw in configs:
        m = mt.make_regressor(seed=SEED, **kw).fit(data.X[tr], data.y[tr])
        ic_in = mt.daily_rank_ic(pd.Series(m.predict(data.X[tr]), data.X.index[tr]), data.ret[tr]).mean()
        ic_out = mt.daily_rank_ic(pd.Series(m.predict(data.X[te]), data.X.index[te]), data.ret[te]).mean()
        rows.append({"配置": name, "样本内 RankIC": ic_in, "样本外 RankIC": ic_out, "差距": ic_in - ic_out})
    df = pd.DataFrame(rows).set_index("配置")
    print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
    best = df["样本外 RankIC"].idxmax()
    print(f"""
    💡 样本内 IC 随复杂度从 {df['样本内 RankIC'].iloc[0]:+.3f} 升到 {df['样本内 RankIC'].iloc[-1]:+.3f}，
       样本外 IC 却只在 {df['样本外 RankIC'].min():+.3f} ~ {df['样本外 RankIC'].max():+.3f} 之间（样本外最好的是"{best}"）。
       样本内的"进步"几乎都是在记忆噪声；样本内 IC 接近 1 是过拟合的典型信号。
       低信噪比下，正则（大叶子、小学习率、少特征）通常比"更强的模型"更值得先调。
    """)
    return df


# ==================== 第五部分：研究范式与本阶段路线 ====================


def research_paradigm():
    print("\n" + "=" * 60)
    print("第五部分：研究范式与本阶段路线")
    print("=" * 60)
    print("""
    ❌ 西西弗斯范式：一个人从数据做到回测，回测不好就改参数再回测 …… 直到曲线好看
       → 每次回测都是一次"多重检验"，最后挑出来的往往是运气（08 课 DSR / PBO）
    ✅ 元策略范式：数据、特征、标签、验证、组合、监控分成独立环节，每个环节有自己的质检

    ┌────────┬──────────────────────────────┬──────────────────────────────┐
    │ 课     │ 环节                         │ 防的是什么坑                 │
    ├────────┼──────────────────────────────┼──────────────────────────────┤
    │ 02     │ 特征工程、截面标准化、分数差分│ 全样本标准化前视、非平稳     │
    │ 03     │ Triple-Barrier、Meta-Label   │ 固定期限标签忽略路径与波动   │
    │ 04     │ Purged K-Fold + Embargo      │ 重叠标签 / 截面相关造成泄漏  │
    │ 05     │ GBDT 截面预测 + Walk-Forward │ 用 MSE 评估、随机验证集      │
    │ 06     │ MDI / MDA / SHAP             │ 噪声特征、替代效应、泄漏特征 │
    │ 07     │ 预测 → 持仓 → 扣费回测       │ 准确率高 ≠ 赚钱              │
    │ 08     │ DSR / PBO                    │ 多次试验的选择偏差           │
    │ 09     │ Qlib 工作流                  │ 重复造轮子、口径不一致       │
    │ 10     │ 漂移监控与再训练             │ 模型上线后悄悄失效           │
    └────────┴──────────────────────────────┴──────────────────────────────┘

    📜 AFML 名言："Backtesting is not a research tool. Feature importance is."
       回测只告诉你一个数字；特征重要性告诉你模型"为什么"有效。
    """)


def plot_capacity(df):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["样本内 RankIC"], 0.4, label="样本内")
    ax.bar(x + 0.2, df["样本外 RankIC"], 0.4, label="样本外")
    ax.set_xticks(x, df.index, fontsize=8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"模型复杂度 vs 样本内外 RankIC（{mt.gbdt_backend()}）")
    ax.legend()
    path = save_figure(fig, "01_capacity_overfit.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    ml_vs_finance()
    signal_to_noise()
    non_iid(data)
    df = capacity_vs_overfit(data)
    research_paradigm()
    plot_capacity(df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第二部分中，把噪声 σ_ε 改为 1% 和 3%，要达到 ρ = 0.1 分别需要多大的 σ_s？
    2. 把第三部分的截面相关改为"先减去当天截面均值"（去市场）后再算 ρ̄，N_eff 如何变化？说明了什么？
    3. 第四部分中把训练 / 测试对调（用后半段训练、前半段测试），结论是否仍成立？这样做有什么问题？
    4. 计算标签期限 h = 20 时，用 4000 个日度样本训练相当于多少个"独立"样本（提示：AFML 平均唯一度，03 课）
    """)


if __name__ == "__main__":
    main()
