"""
特征工程：价量、基本面与分数阶差分
==================================

学习目标：
    1. 在 Phase 7 面板上构造 13 个价量特征 + 4 个 PIT 基本面特征，组织成 (datetime, instrument) 长表
    2. 用"截断数据重算"的方法自动检查特征有没有用到未来数据
    3. 理解截面标准化（rank / z-score）为什么是截面选股特征工程的核心，以及全样本标准化的前视
    4. 实现 López de Prado 的固定窗口分数阶差分（FFD），找到"刚好平稳"的最小 d

核心概念：
    - 截面 rank：x̃ = rank_t(x) / N_t − 0.5，每天独立计算，天然不含未来信息
    - 全样本 z-score：用了整个样本（含未来）的均值方差 → 前视
    - 分数阶差分：X̃_t = Σ ω_k X_{t−k}，ω_0 = 1，ω_k = −ω_{k−1}(d − k + 1)/k
    - 平稳性与记忆的权衡：d = 1 平稳但丢掉价格水平；d ≈ 0.3–0.5 往往已平稳且保留记忆

前置知识：
    - Phase 3 ADF 检验；Phase 7 第 03 课（因子构建）、第 04 课（预处理）、第 08 课（PIT）
    - 01-ml-quant-overview.py

环境要求：
    - pip install pandas numpy statsmodels scikit-learn matplotlib
    - 数据来自 Phase 7 共享面板；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller

import ml_tools as mt
from utils.data import get_index_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


# ==================== 第一部分：特征分类与构造 ====================


def feature_overview(data):
    print("=" * 60)
    print("第一部分：特征分类与构造")
    print("=" * 60)
    print("""
    ┌──────────┬──────────────────────────────────┬──────────┬────────────────────────────┐
    │ 类别     │ 例子                             │ 更新频率 │ 关键陷阱                   │
    ├──────────┼──────────────────────────────────┼──────────┼────────────────────────────┤
    │ 价量     │ 动量、反转、波动、换手、Amihud   │ 日       │ 复权、停牌、涨跌停         │
    │ 基本面   │ EP、BP、ROE、利润增速            │ 季度     │ PIT：按公告日而非报告期    │
    │ 一致预期 │ 分析师 EPS 预期、评级调整        │ 不定期   │ 小盘股没人覆盖             │
    │ 另类数据 │ 新闻情绪、舆情、供应链（Phase 10）│ 不定     │ 时间戳、样本短、幸存者偏差 │
    │ 市场状态 │ 指数波动率、市场宽度             │ 日       │ 截面上全相同，需与个股交互 │
    └──────────┴──────────────────────────────────┴──────────┴────────────────────────────┘

    本阶段的数据形态与 Qlib 一致：长表，索引 = (datetime, instrument)，每行一个"股票 × 调仓日"样本。
    """)
    raw = data.raw
    rows = []
    for k in mt.FEATURES:
        rows.append({"特征": k, "含义": mt.FEATURE_LABELS[k], "类别": mt.FEATURE_GROUP[k],
                     "缺失率": raw[k].isna().mean(), "中位数": raw[k].median()})
    print(f"📋 {len(mt.FEATURES)} 个特征（调仓日 {len(data.dates)} 个，样本 {len(raw)} 行，"
          f"平均每期 {len(raw) / len(data.dates):.0f} 只可交易股票）：")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    d = data.dates[-1]
    print(f"\n📋 最后一个调仓日 {d.date()} 的前 3 行（原始值）：")
    print(raw.xs(d, level="datetime").head(3).iloc[:, :8].to_string(float_format=lambda v: f"{v:.4f}"))


# ==================== 第二部分：自动检查前视 ====================


def lookahead_check(data):
    print("\n" + "=" * 60)
    print("第二部分：自动检查 —— 特征有没有偷看未来？")
    print("=" * 60)
    print("""
    最可靠的检查方法："假装今天是 t"。
       1. 把面板截断到 t（t 之后的数据全部删掉）
       2. 在截断数据上重算 t 日的特征
       3. 与完整数据上算出的 t 日特征对比 —— 只要有差异，就说明用到了 t 之后的数据
    这个测试应当写进单元测试，每加一个特征都跑一遍。
    """)
    panel = data.panel
    full = mt.daily_price_features(panel)
    rows = []
    for t in [data.dates[20], data.dates[45]]:
        trunc = replace(panel, **{f: getattr(panel, f).loc[:t] for f in
                                  ["open", "high", "low", "close", "adj_factor", "volume", "amount",
                                   "float_shares", "total_shares"]})
        part = mt.daily_price_features(trunc)
        diff = max(float((full[k].loc[t] - part[k].loc[t]).abs().max()) for k in full)
        rows.append({"截断日 t": t.date(), "特征数": len(full), "最大绝对差": diff})
    print(pd.DataFrame(rows).to_string(index=False))

    # 反例：居中滚动均值（center=True）用到了未来
    px = panel.adj_close.ffill()
    bad_full = px.rolling(21, center=True).mean()
    t = data.dates[45]
    bad_part = px.loc[:t].rolling(21, center=True).mean()
    a, b = bad_full.loc[t], bad_part.loc[t]
    n_nan = int((a.notna() & b.isna()).sum())
    n_diff = int(((a - b).abs() > 1e-12).sum())
    print(f"\n   ❌ 反例 rolling(21, center=True)：截断后 t 日变成 NaN 的股票 {n_nan} 只、数值改变的 {n_diff} 只"
          f"（共 {int(a.notna().sum())} 只有值）→ 用了未来")
    print("   💡 基本面特征的 PIT 对齐已在 Phase 7 第 08 课验证：只使用'公告日 < t'的财报。")


# ==================== 第三部分：截面标准化 ====================


def cross_sectional_norm(data):
    print("\n" + "=" * 60)
    print("第三部分：截面标准化")
    print("=" * 60)
    print("""
    截面选股的目标是"同一天里谁比谁好"，所以特征要在每个调仓日内标准化：
        rank：x̃ = rank_t(x) / N_t − 0.5      （本阶段默认，对极值天然稳健）
        z-score：x̃ = (x − median_t) / MAD_t  （保留幅度信息，需先去极值）

    原始特征的"整体水平"随市场环境漂移。下表是每年各特征截面中位数的平均：
    """)
    raw = data.raw
    yr = raw.index.get_level_values("datetime").year
    cols = ["vol_20", "turn_20", "amihud_20", "ep"]
    level = raw[cols].groupby(level="datetime").median().groupby(lambda d: d.year).mean()
    print(level.to_string(float_format=lambda v: f"{v:.4g}"))

    full_z = (raw[cols] - raw[cols].mean()) / raw[cols].std()  # ❌ 全样本统计量
    cs = mt.cs_rank(raw[cols])
    cmp = pd.DataFrame({"全样本 z 的年均值(vol_20)": full_z["vol_20"].groupby(yr).mean(),
                        "截面 rank 的年均值(vol_20)": cs["vol_20"].groupby(yr).mean()})
    print("\n📋 两种标准化后的 vol_20，按年份平均：")
    print(cmp.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    💡 全样本 z-score 有两个问题：
       ① 前视：2019 年的标准化值用到了 2024 年的均值和方差
       ② 漂移：高波动年份所有股票的值都偏高，树模型会把"年份"当成信号
       截面 rank 每期均值恒约为 0（= 1/(2N)，只用当天的截面），两个问题同时消失。
       Rank IC 对每期内的单调变换不变，所以单因子 IC 不受影响；受影响的是跨期混合训练的模型。
    """)

    # 实验：同一模型，原始特征 vs 截面 rank 特征
    rows = {}
    for name, X in [("原始特征（不标准化）", raw.fillna(raw.median())), ("截面 rank", data.X)]:
        pred = mt.walk_forward(X, data.y, lambda: mt.make_regressor(), retrain_every=6)
        rows[name] = mt.ic_stats(mt.daily_rank_ic(pred, data.ret))
    print(f"📋 Walk-Forward 样本外对比（{mt.gbdt_backend()}，每 6 个月重训，标签 = 截面 rank 收益）：")
    print(mt.ic_frame(rows).to_string(float_format=lambda v: f"{v:+.3f}"))
    print("    以上是实际计算结果；两者差距在少量期数下未必显著，但截面标准化在原理上是正确做法。")


# ==================== 第四部分：特征冗余与单特征 IC ====================


def redundancy(data):
    print("\n" + "=" * 60)
    print("第四部分：特征冗余与单特征 Rank IC")
    print("=" * 60)
    corr = data.X.groupby(level="datetime").corr().groupby(level=1).mean()
    corr = corr.loc[mt.FEATURES, mt.FEATURES]
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().dropna()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(6)
    print("📋 平均截面相关系数（rank 后）绝对值最高的 6 对：")
    for (a, b), v in top.items():
        print(f"   {a:>14s} ~ {b:<14s} {v:+.2f}")
    print("   → 高度相关的特征会让特征重要性失真（06 课替代效应），也增加过拟合风险")

    ic = mt.ic_frame({k: mt.ic_stats(mt.daily_rank_ic(data.X[k], data.ret)) for k in mt.FEATURES})
    ic.insert(0, "含义", [mt.FEATURE_LABELS[k] for k in ic.index])
    ic = ic.reindex(ic["t值"].abs().sort_values(ascending=False).index)
    print("\n📋 单特征月度 Rank IC（全样本，仅作描述；方向未统一，负 IC 表示值越大收益越低）：")
    print(ic.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("""
    ⚠️ 这张表用了全样本，属于"事后描述"。用它来挑特征再去回测 = 数据窥探。
       正确做法是在每个训练窗口内部做特征选择（06 课 MDA 在训练窗口内计算）。
    """)
    return corr


# ==================== 第五部分：分数阶差分 ====================


def frac_diff_weights(d: float, thresh: float = 1e-4, max_len: int = 5000) -> np.ndarray:
    """FFD 权重：ω_0 = 1，ω_k = −ω_{k−1}(d − k + 1)/k，|ω_k| < thresh 时截断"""
    w = [1.0]
    for k in range(1, max_len):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w)


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """固定窗口分数阶差分：X̃_t = Σ_k ω_k X_{t−k}，只用 t 及以前的 len(w) 个观测"""
    w = frac_diff_weights(d, thresh)
    x = series.to_numpy(dtype=float)
    if len(x) < len(w):
        return pd.Series(np.nan, index=series.index)
    out = np.convolve(x, w, mode="valid")  # convolve 会翻转 w：out[n] = Σ_k w_k x[n + len(w) − 1 − k]
    return pd.Series(np.r_[np.full(len(w) - 1, np.nan), out], index=series.index)


def fractional_diff():
    print("\n" + "=" * 60)
    print("第五部分：分数阶差分（AFML 第 5 章）")
    print("=" * 60)
    print("""
    价格（对数价格）非平稳；一阶差分（收益率）平稳，但丢掉了几乎所有"记忆"（价格水平）。
    分数阶差分：d ∈ (0, 1)，在通过 ADF 检验的前提下保留尽可能多的记忆。

        X̃_t = Σ_k ω_k X_{t−k}，  ω_0 = 1，  ω_k = −ω_{k−1} · (d − k + 1) / k

        d = 1：ω = (1, −1, 0, 0, …) → 普通差分
        d = 0.4：ω = (1, −0.4, −0.12, −0.064, …) 缓慢衰减 → 保留长期记忆
    FFD（fixed-width window）：|ω_k| < 1e-4 后截断，所有时点使用同样长度的窗口，避免早期样本权重不一致。
    """)
    idx = get_index_daily("000300", "20100101", "20251231")
    tag = "（⚠️ 合成数据）" if is_synthetic(idx) else ""
    logp = np.log(idx["close"])
    print(f"📋 沪深 300 对数收盘价{tag}（{logp.index[0].date()} ~ {logp.index[-1].date()}，{len(logp)} 天）：")
    rows, series = [], {}
    for d in np.round(np.arange(0.0, 1.01, 0.1), 1):
        fd = frac_diff_ffd(logp, d).dropna()
        stat, p, *_ = adfuller(fd, maxlag=1, regression="c", autolag=None, result_object=False)
        rows.append({"d": d, "窗口长度": len(frac_diff_weights(d)), "ADF 统计量": stat, "p 值": p,
                     "与原序列相关": np.corrcoef(fd, logp.loc[fd.index])[0, 1]})
        series[d] = fd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ok = df[df["p 值"] < 0.05]
    d_star = float(ok["d"].min()) if len(ok) else 1.0
    corr_star = float(df.loc[df["d"] == d_star, "与原序列相关"].iloc[0])
    corr_one = float(df.loc[df["d"] == 1.0, "与原序列相关"].iloc[0])
    print(f"""
    ✅ 通过 ADF（p < 0.05）的最小 d = {d_star}：与原序列相关 {corr_star:.3f}；
       对比 d = 1（收益率）：与原序列相关 {corr_one:.3f}。
    💡 分数阶差分在单资产时序模型（期货、加密、指数择时）中更常用；
       截面选股的特征多是比率或收益类，本身大多平稳。
    """)
    return logp, series, d_star


def pitfalls():
    print("\n" + "=" * 60)
    print("第六部分：特征工程的常见陷阱")
    print("=" * 60)
    print("""
    ┌────────────────────────┬──────────────────────────────┬──────────────────────────┐
    │ 陷阱                   │ 表现                         │ 对策                     │
    ├────────────────────────┼──────────────────────────────┼──────────────────────────┤
    │ 全样本标准化           │ 微妙前视；水平漂移           │ 截面标准化 / 滚动统计量  │
    │ 财报按报告期对齐       │ 用到尚未公布的财报           │ 按公告日 PIT（Phase 7）  │
    │ center=True / shift(-n)│ 特征里出现未来               │ 截断重算测试             │
    │ 只用当前存续股票       │ 幸存者偏差                   │ 包含退市股               │
    │ 几百个高度相关特征     │ 重要性失真、过拟合           │ 去冗余、聚类重要性       │
    │ 停牌日填 0 收益        │ 波动率、动量被扭曲           │ 停牌为 NaN，min_periods  │
    └────────────────────────┴──────────────────────────────┴──────────────────────────┘
    """)


def plot_results(corr, logp, series, d_star):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(corr)), corr.index, fontsize=7)
    axes[0].set_title("特征平均截面相关（rank 后）")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    ax = axes[1]
    ax.plot(logp.index, logp - logp.iloc[0], label="对数价格（d = 0）", lw=0.8)
    s = series[d_star]
    ax.plot(s.index, s - s.mean(), label=f"FFD d = {d_star}", lw=0.8)
    ax.plot(series[1.0].index, series[1.0] * 10, label="对数收益率 ×10（d = 1）", lw=0.5, alpha=0.6)
    ax.set_title("沪深 300：分数阶差分保留记忆")
    ax.legend(fontsize=8)
    path = save_figure(fig, "02_features_fracdiff.png", subdir="phase-9")
    print(f"📈 图表已保存：{path}")


def main():
    data = mt.load_ml_data()
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    feature_overview(data)
    lookahead_check(data)
    cross_sectional_norm(data)
    corr = redundancy(data)
    logp, series, d_star = fractional_diff()
    pitfalls()
    plot_results(corr, logp, series, d_star)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 新增特征"20 日收益 / 20 日波动率"（风险调整动量），先通过第二部分的截断测试，再看单特征 Rank IC
    2. 把截面 rank 换成"MAD 去极值 + z-score"（mt.cs_zscore + 去极值），重跑第三部分的 Walk-Forward 对比
    3. 对面板中任意一只股票的对数价格做 FFD，最小 d 与沪深 300 是否相同？把 thresh 改为 1e-5 有何变化？
    4. 构造一个市场状态特征（沪深 300 的 20 日波动率），它在截面上全相同。直接加入模型和与 vol_20 相乘后加入，效果有何不同？
    """)


if __name__ == "__main__":
    main()
