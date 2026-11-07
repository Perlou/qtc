"""
参数优化与参数平原
==================

学习目标：
    1. 对双均线策略做参数网格搜索，画出 Sharpe 热力图
    2. 区分"参数孤峰"与"参数平原"，用邻域平均（平滑）后再选参
    3. 用样本内 / 样本外切分，检验"单点最优"与"平原最优"哪个更可靠
    4. 用置换检验理解"尝试次数越多，最优结果中运气成分越大"
    5. 掌握参数集成（多参数等权）这一 CTA 业内常用做法

核心概念：
    - 参数平原（Parameter Plateau）：最优参数附近一片区域表现都不错 → 规律是结构性的
    - 参数孤峰：最优点邻居都很差 → 大概率是噪声拟合
    - 邻域平均：每个格子替换为 3×3 邻域均值，再取最大值
    - 数据挖掘偏差：在纯噪声上搜索足够多参数，也能找到"漂亮"的最优值

前置知识：
    - 01-moving-average-crossover.py、07-position-sizing-atr.py
    - Phase 4 中的过拟合与样本外检验

环境要求：
    - pip install akshare pandas numpy scipy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.ndimage import uniform_filter

from cta_utils import SPECS, UNIVERSE, cost_rate, load_universe, sharpe
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

FASTS = list(range(5, 55, 5))  # 10 个快线周期
SLOWS = list(range(20, 260, 20))  # 12 个慢线周期
SPLIT = "2023-01-01"  # 样本内 / 样本外分界


def ma_strategy_returns(close: pd.Series, ret: pd.Series, cost: pd.Series, roll: pd.Series,
                        fast: int, slow: int) -> pd.Series:
    """双均线多空（±1），信号 shift(1) 次日持有，扣成本与换月成本"""
    ma_s = close.rolling(slow).mean()
    sig = np.sign(close.rolling(fast).mean() - ma_s).where(ma_s.notna(), 0.0)
    pos = sig.shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(0) + 2 * pos.abs() * roll.astype(float)
    return pos * ret - turnover * cost


def grid_search(df: pd.DataFrame, s: str, periods: dict[str, slice] | None = None) -> dict[str, pd.DataFrame]:
    """返回 {区间名: Sharpe 网格}；指标在全样本上计算（保证预热），Sharpe 在各区间上分别计算"""
    close, ret = df["close"], df["close"].pct_change().fillna(0)
    cost, roll = cost_rate(s, df["raw_close"]), df["roll"]
    periods = periods or {"全样本": slice(None)}
    out = {k: pd.DataFrame(np.nan, index=FASTS, columns=SLOWS) for k in periods}
    for f in FASTS:
        for sl in SLOWS:
            if f >= sl:
                continue
            r = ma_strategy_returns(close, ret, cost, roll, f, sl).iloc[sl:]
            for k, p in periods.items():
                out[k].loc[f, sl] = sharpe(r.loc[p])
    for g in out.values():
        g.index.name, g.columns.name = "快线", "慢线"
    return out


def smooth(grid: pd.DataFrame) -> pd.DataFrame:
    """
    3×3 邻域平均。无效组合（快线 ≥ 慢线）和网格边界外都按全表最小值处理（保守）：
    边界上的参数邻居更少、更不确定，不应该因为"复制自己"而占便宜（mode="nearest" 会这样）
    """
    lo = grid.min().min()
    filled = grid.fillna(lo).to_numpy()
    sm = pd.DataFrame(uniform_filter(filled, size=3, mode="constant", cval=lo), index=grid.index, columns=grid.columns)
    return sm.where(grid.notna())


# ==================== 第一部分：参数网格与热力图 ====================


def heatmap_demo(data: dict, s: str = "RB") -> pd.DataFrame:
    print("=" * 60)
    print("第一部分：参数网格搜索与 Sharpe 热力图")
    print("=" * 60)

    print(f"""
    对 {SPECS[s]['name']} 的双均线多空策略遍历：快线 {FASTS[0]}~{FASTS[-1]}（步长 5），慢线 {SLOWS[0]}~{SLOWS[-1]}（步长 20），
    剔除快线 ≥ 慢线的组合。每个组合都扣手续费、滑点和换月成本。

    比喻：在山区找地盖房子。"孤峰"视野最好，但风一吹就倒；"高原"未必最高，但四周都是平地，地基稳。
    """)
    grid = grid_search(data[s], s)["全样本"]
    print(grid.round(2).to_string(na_rep="  -"))
    v = grid.stack().dropna()
    print(f"""
    📊 {len(v)} 个有效组合的 Sharpe 分布：
       最大 {v.max():.2f}  中位数 {v.median():.2f}  最小 {v.min():.2f}  Sharpe > 0 的占比 {(v > 0).mean():.0%}
    💡 报告参数结果时，应该报告这个"分布"，而不是只报最大值。
    """)
    return grid


# ==================== 第二部分：孤峰 vs 平原 ====================


def plateau_vs_peak(grid: pd.DataFrame, s: str = "RB"):
    print("\n" + "=" * 60)
    print("第二部分：孤峰 vs 平原 —— 平滑后再选参")
    print("=" * 60)

    sm = smooth(grid)
    best_raw = grid.stack().idxmax()
    best_sm = sm.stack().idxmax()
    f, sl = best_raw
    fi, si = FASTS.index(f), SLOWS.index(sl)
    neigh = grid.iloc[max(fi - 1, 0):fi + 2, max(si - 1, 0):si + 2]
    ratio = neigh.stack().mean() / grid.loc[f, sl] if grid.loc[f, sl] > 0 else np.nan
    print(f"""
    单点最优：快线 {best_raw[0]} / 慢线 {best_raw[1]}，Sharpe {grid.loc[best_raw]:.2f}
       它的 3×3 邻居平均 Sharpe {neigh.stack().mean():.2f}，是最优值的 {ratio:.0%}
       （经验参考：邻域平均不低于最优值的 70% 左右，才算处在"平原"上）
    平滑后最优：快线 {best_sm[0]} / 慢线 {best_sm[1]}，原始 Sharpe {grid.loc[best_sm]:.2f}，邻域平均 {sm.loc[best_sm]:.2f}
    """)
    print("📊 3×3 邻域平均后的热力图：")
    print(sm.round(2).to_string(na_rep="  -"))
    print("""
    💡 平滑后的最优点问的是："哪个参数及其邻居平均表现最好？"
       它天然偏向平坦区域，对"未来参数最优点漂移"更不敏感。
    """)
    return sm


# ==================== 第三部分：样本内选参，样本外检验 ====================


def in_out_sample(data: dict):
    print("\n" + "=" * 60)
    print(f"第三部分：样本内选参、样本外检验（分界 {SPLIT}）")
    print("=" * 60)

    print("""
    对每个品种：只用样本内数据画热力图并选参，然后看所选参数在样本外的 Sharpe。
    比较三种选法：① 样本内单点最优  ② 样本内平滑后最优  ③ 参数集成（不选参，等权多组参数）
    """)
    periods = {"样本内": slice(None, pd.Timestamp(SPLIT) - pd.Timedelta(days=1)), "样本外": slice(SPLIT, None)}
    ens_pairs = [(5, 20), (10, 40), (20, 80), (40, 160)]
    rows, rank_corr = [], []
    for s, df in data.items():
        g = grid_search(df, s, periods)
        gi, go = g["样本内"], g["样本外"]
        b_raw = gi.stack().idxmax()
        b_sm = smooth(gi).stack().idxmax()
        close, ret = df["close"], df["close"].pct_change().fillna(0)
        cost, roll = cost_rate(s, df["raw_close"]), df["roll"]
        ens = pd.concat([ma_strategy_returns(close, ret, cost, roll, f, sl) for f, sl in ens_pairs], axis=1).mean(axis=1)
        rank_corr.append(gi.stack().corr(go.stack(), method="spearman"))
        rows.append([s, SPECS[s]["name"], f"{b_raw[0]}/{b_raw[1]}", gi.loc[b_raw], go.loc[b_raw],
                     f"{b_sm[0]}/{b_sm[1]}", gi.loc[b_sm], go.loc[b_sm], sharpe(ens.loc[periods["样本外"]]),
                     go.stack().median()])
    t = pd.DataFrame(rows, columns=["代码", "品种", "单点最优", "IS", "OOS", "平滑最优", "IS ", "OOS ", "集成OOS",
                                    "全网格OOS中位数"])
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    m = t[["IS", "OOS", "IS ", "OOS ", "集成OOS", "全网格OOS中位数"]].mean()
    decay_raw, decay_sm = m["IS"] - m["OOS"], m["IS "] - m["OOS "]
    print(f"""
    📊 各品种平均：单点最优 IS {m['IS']:+.2f} → OOS {m['OOS']:+.2f}（衰减 {decay_raw:.2f}）；
       平滑最优 IS {m['IS ']:+.2f} → OOS {m['OOS ']:+.2f}（衰减 {decay_sm:.2f}）；
       参数集成 OOS {m['集成OOS']:+.2f}；随便选一个参数（全网格 OOS 中位数）{m['全网格OOS中位数']:+.2f}
    📊 样本内与样本外热力图的 Spearman 秩相关（各品种平均）：{np.nanmean(rank_corr):+.2f}
       → 越接近 0（甚至为负），说明样本内"哪里好"对样本外"哪里好"的预测力越弱
    💡 观察要点：
       - 样本内最优的 Sharpe 是被"挑"出来的，样本外几乎总会衰减 —— 衰减幅度就是过拟合的代价
       - 平滑最优不追求样本内最好；参数集成干脆不做选参，结果不依赖"挑选"这一步
       - 如果"精心选出的参数"在样本外并不比"随便选一个"好，说明选参这一步没有带来真实信息
       - 只有 {len(t)} 个品种、一次切分，结论有很大随机性；09 课会用更多维度检验
    """)


# ==================== 第四部分：数据挖掘偏差 —— 置换检验 ====================


def permutation_test(data: dict, s: str = "RB", grid: pd.DataFrame | None = None, n_perm: int = 20):
    print("\n" + "=" * 60)
    print("第四部分：数据挖掘偏差 —— 在\"没有规律\"的数据上也能找到漂亮参数")
    print("=" * 60)

    df = data[s]
    ret = df["close"].pct_change().fillna(0).to_numpy()
    avg_cost = float(cost_rate(s, df["raw_close"]).mean())
    no_roll = pd.Series(False, index=df.index)
    rng = np.random.default_rng(0)
    maxes = []
    for _ in range(n_perm):
        r = pd.Series(rng.permutation(ret), index=df.index)  # 打乱顺序：保留收益分布，破坏时间结构（趋势）
        close = (1 + r).cumprod()
        best = -np.inf
        for f in FASTS:
            for sl in SLOWS:
                if f < sl:
                    best = max(best, sharpe(ma_strategy_returns(close, r, avg_cost, no_roll, f, sl).iloc[sl:]))
        maxes.append(best)
    maxes = np.array(maxes)
    actual = grid.stack().max() if grid is not None else np.nan
    n_combo = int(grid.notna().sum().sum()) if grid is not None else 0
    print(f"""
    做法：把 {SPECS[s]['name']} 的日收益随机打乱 {n_perm} 次（保留收益分布、破坏趋势结构），
          每次都对 {n_combo} 个参数组合做网格搜索，记录"最优 Sharpe"。
    📊 打乱后的最优 Sharpe：中位数 {np.median(maxes):.2f}，最大 {maxes.max():.2f}，最小 {maxes.min():.2f}
       真实数据上的最优 Sharpe：{actual:.2f}，超过了 {(maxes < actual).mean():.0%} 的打乱样本
    💡 即使数据里没有任何趋势，网格最优的 Sharpe 也常常明显大于 0 —— 这就是"尝试次数"带来的运气成分。
       真实最优值只有显著高于打乱样本的分布时，才说明参数背后可能有真实规律。
       按课程规范，研究日志要记录所有尝试次数，Phase 9 会用 Deflated Sharpe 做校正。
    ⚠️ 打乱次数 {n_perm} 次只是演示，严肃的检验通常需要上千次。
    """)


# ==================== 第五部分：选参原则与热力图 ====================


def principles_and_plot(grid: pd.DataFrame, sm: pd.DataFrame, s: str = "RB"):
    print("\n" + "=" * 60)
    print("第五部分：选参原则")
    print("=" * 60)
    print("""
    ┌──────────────┬─────────────────────────────────────────────────────┐
    │ 原则         │ 说明                                                │
    ├──────────────┼─────────────────────────────────────────────────────┤
    │ 参数少       │ 每多一个参数，过拟合空间指数级增大                  │
    │ 有经济含义   │ "20 日 ≈ 一个月"、"250 日 ≈ 一年"，比"37 日"可信    │
    │ 粗网格       │ 用 10、20、40、80 的几何网格，不要 1 步长精搜       │
    │ 看分布       │ 报告所有组合 Sharpe 的中位数、正收益占比            │
    │ 参数集成     │ 同时运行多组参数并等权平均（参数分散化）            │
    │ 记录尝试次数 │ 研究日志记录每一次尝试，后续做多重检验校正          │
    └──────────────┴─────────────────────────────────────────────────────┘
    """)
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, g, title in [(axes[0], grid, "原始 Sharpe"), (axes[1], sm, "3×3 邻域平均")]:
        im = ax.imshow(g.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(SLOWS)), SLOWS)
        ax.set_yticks(range(len(FASTS)), FASTS)
        ax.set_xlabel("慢线周期")
        ax.set_ylabel("快线周期")
        ax.set_title(f"{SPECS[s]['name']} 双均线：{title}")
        fig.colorbar(im, ax=ax)
    path = save_figure(fig, "08_param_heatmap.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    data = load_universe(UNIVERSE)
    s = "RB" if "RB" in data else next(iter(data))
    grid = heatmap_demo(data, s)
    sm = plateau_vs_peak(grid, s)
    in_out_sample(data)
    permutation_test(data, s, grid)
    principles_and_plot(grid, sm, s)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把网格改为几何网格（快线 5,10,20,40；慢线 20,40,80,160,320），热力图是否更平滑？
    2. 用 5×5 邻域代替 3×3 做平滑，平滑最优点如何移动？样本外表现呢？
    3. 把第三部分的单次切分改为 Walk-Forward：每年用过去 3 年选参，下一年检验，滚动推进
    4. 把置换次数提高到 200 次，计算真实最优 Sharpe 的经验 p 值
    """)


if __name__ == "__main__":
    main()
