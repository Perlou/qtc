"""
Walk-Forward 与样本外检验
==========================

学习目标：
    1. 理解样本内（IS）/ 样本外（OOS）划分，以及单次划分的问题
    2. 实现 Walk-Forward 分析：滚动窗口（Rolling）与锚定窗口（Anchored / Expanding）
    3. 拼接所有测试段得到一条完整的样本外净值，并与"全样本最优参数"（事后诸葛亮）对比
    4. 计算 Walk-Forward 效率（WFE）、观察参数稳定性
    5. 用"随机游走对照组"确认 Walk-Forward 能识别出纯噪声策略
    6. 理解最终保留区（Holdout）只能用一次

核心概念：
    - Walk-Forward：用过去一段数据选参数 → 在紧接着的下一段上测试 → 窗口前移，重复
    - 指标预热合法，用测试段数据选参数不合法
    - WFE = 样本外年化收益 / 样本内年化收益

前置知识：
    - 08-overfitting-traps.py
    - 07-lookahead-bias.py（信号必须只用过去数据，Walk-Forward 才有意义）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import performance_report
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20100101", "20241231"
HOLDOUT_START = "2023-01-01"  # 最终保留区：研究全部结束后只用一次
FAST = [5, 10, 20, 30, 40]
SLOW = [20, 40, 60, 90, 120, 160, 200, 250]
COST = 0.0005
TRAIN, TEST = 750, 250  # 训练约 3 年，测试约 1 年


# ==================== 第一部分：为什么需要样本外 ====================


def why_oos():
    print("=" * 60)
    print("第一部分：为什么需要样本外，以及单次划分的问题")
    print("=" * 60)

    print("""
    📝 比喻：样本内是"练习题"，样本外是"考试"。只看练习成绩，你无法知道学生是真懂还是背了答案。

    单次划分：
    |←──────── 样本内 70% ────────→|←── 样本外 30% ──→|
    问题：
    - 只有一次"考试"，结果受那段行情影响很大
    - 看到样本外不好 → 回去改策略再测 → 样本外变成了样本内（"样本外污染"）
    - 没有模拟"定期重新优化参数"的真实过程

    Walk-Forward：
      滚动窗口（训练长度固定）              锚定窗口（训练起点固定，不断扩展）
      [══ 训练 ══][─测试─]                  [══ 训练 ══][─测试─]
           [══ 训练 ══][─测试─]             [══════ 训练 ══════][─测试─]
                [══ 训练 ══][─测试─]        [══════════ 训练 ══════════][─测试─]
      拼接所有"测试"段 → 一条完整的样本外净值曲线

    ┌──────────┬────────────────────────────┬────────────────────────────────┐
    │ 类型     │ 优点                       │ 缺点                           │
    ├──────────┼────────────────────────────┼────────────────────────────────┤
    │ 滚动窗口 │ 适应市场状态变化           │ 训练数据少，参数可能不稳定     │
    │ 锚定窗口 │ 数据越来越多，估计更稳定   │ 远古与近期数据一视同仁，适应慢 │
    └──────────┴────────────────────────────┴────────────────────────────────┘
    """)


# ==================== 第二部分：参数网格的收益矩阵 ====================


def grid_returns(close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    对每组 (快线, 慢线) 计算整段历史的持仓与扣费后日收益。
    均线只用过去数据（07 课的截断测试可验证），所以先在全历史上算好、再按窗口截取是合法的 ——
    这正是"用训练段末尾的数据为测试段预热指标"。
    """
    ret = close.pct_change().fillna(0.0)
    ma = {w: close.rolling(w).mean() for w in sorted(set(FAST) | set(SLOW))}
    pos, rets = {}, {}
    for f in FAST:
        for s in SLOW:
            if f >= s:
                continue
            p = (ma[f] > ma[s]).astype(float).shift(1).fillna(0.0)
            pos[(f, s)] = p
            rets[(f, s)] = p * ret - p.diff().abs().fillna(p.abs()) * COST
    return pd.DataFrame(pos), pd.DataFrame(rets)


def sharpe(r: pd.DataFrame | pd.Series):
    return r.mean() / r.std() * np.sqrt(252)


# ==================== 第三部分：Walk-Forward 实现 ====================


def walk_forward(pos: pd.DataFrame, rets: pd.DataFrame, train: int = TRAIN, test: int = TEST,
                 anchored: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """
    返回 (拼接的样本外日收益, 每个窗口的记录)。
    训练段只用 [start − train, start)（锚定窗口为 [0, start)）的收益选参数，
    测试段 [start, start + test) 使用选中的参数。
    参数切换时，旧参数的持仓要换成新参数的持仓，这笔换仓成本单独计入测试段第一天。
    """
    oos, rows = [], []
    prev_param = None
    start = train
    while start < len(rets):
        tr = rets.iloc[(0 if anchored else start - train) : start]
        te = rets.iloc[start : start + test].copy()
        sr = sharpe(tr)
        best = sr.idxmax()
        if prev_param is not None and prev_param != best:
            d0 = te.index[0]
            i0 = rets.index.get_loc(d0)
            p_new, p_new_prev, p_old_prev = pos[best].iloc[i0], pos[best].iloc[i0 - 1], pos[prev_param].iloc[i0 - 1]
            te.loc[d0, best] += COST * (abs(p_new - p_new_prev) - abs(p_new - p_old_prev))
        oos.append(te[best])
        is_ann = (1 + tr[best]).prod() ** (252 / len(tr)) - 1
        oos_ann = (1 + te[best]).prod() ** (252 / len(te)) - 1
        rows.append({"训练段": f"{tr.index[0].date()}~{tr.index[-1].date()}",
                     "测试段": f"{te.index[0].date()}~{te.index[-1].date()}", "选中参数": tuple(int(x) for x in best),
                     "IS Sharpe": sr[best], "OOS Sharpe": sharpe(te[best]), "IS 年化": is_ann, "OOS 年化": oos_ann})
        prev_param = best
        start += test
    return pd.concat(oos), pd.DataFrame(rows)


def run_walk_forward(close: pd.Series, label: str, verbose: bool = True) -> dict:
    pos, rets = grid_returns(close)
    out = {}
    for name, anchored in [("滚动窗口", False), ("锚定窗口", True)]:
        oos, win = walk_forward(pos, rets, anchored=anchored)
        out[name] = (oos, win)
    # 事后诸葛亮：用全样本挑出的最优参数，在同样的样本外区间上的表现
    oos_index = out["滚动窗口"][0].index
    hindsight = tuple(int(x) for x in sharpe(rets).idxmax())
    out["全样本最优(事后)"] = (rets.loc[oos_index, hindsight], hindsight)
    out["固定参数(20,60)"] = (rets.loc[oos_index, (20, 60)], (20, 60))
    out["买入持有"] = (close.pct_change().fillna(0.0).loc[oos_index], None)
    out["n_combos"] = rets.shape[1]
    return out


def walk_forward_demo(close: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print(f"第二、三部分：Walk-Forward（沪深 300 指数，研究区 {close.index[0].date()} ~ {close.index[-1].date()}）")
    print("=" * 60)

    res = run_walk_forward(close, "沪深300")
    print(f"   参数网格 {res['n_combos']} 组双均线；训练 {TRAIN} 天、测试 {TEST} 天、单边成本 {COST * 1e4:.0f} bp")
    for name in ["滚动窗口", "锚定窗口"]:
        win = res[name][1].copy()
        for c in ["IS Sharpe", "OOS Sharpe"]:
            win[c] = win[c].map(lambda x: "—(全程空仓)" if pd.isna(x) else f"{x:.2f}")
        for c in ["IS 年化", "OOS 年化"]:
            win[c] = win[c].map(lambda x: f"{x:+.1%}")
        print(f"\n📋 {name}：")
        print(win.drop(columns=["训练段"] if name == "锚定窗口" else []).to_string(index=False))
    return res


# ==================== 第四部分：解读 Walk-Forward 结果 ====================


def interpret(res: dict, label: str) -> pd.DataFrame:
    rows = []
    for name in ["滚动窗口", "锚定窗口", "全样本最优(事后)", "固定参数(20,60)", "买入持有"]:
        r = res[name][0]
        rep = performance_report(r)
        rows.append([name, f"{rep['年化收益']:+.2%}", f"{rep['Sharpe']:.2f}", f"{rep['最大回撤']:.2%}"])
    table = pd.DataFrame(rows, columns=["方案（同一段样本外区间）", "年化", "Sharpe", "最大回撤"])
    print(f"\n📊 {label}：样本外拼接结果")
    print(table.to_string(index=False))

    for name in ["滚动窗口", "锚定窗口"]:
        win = res[name][1]
        wfe = win["OOS 年化"].mean() / win["IS 年化"].mean() if win["IS 年化"].mean() > 0 else np.nan
        n_param = win["选中参数"].nunique()
        print(f"   {name}：平均 IS Sharpe {win['IS Sharpe'].mean():.2f} → 平均 OOS Sharpe {win['OOS Sharpe'].fillna(0).mean():.2f}；"
              f"WFE = {wfe:.2f}；{len(win)} 个窗口选出 {n_param} 种不同参数；"
              f"OOS 为正的窗口 {(win['OOS 年化'] > 0).sum()}/{len(win)}")
    print(f"   全样本最优参数（事后才知道）：{res['全样本最优(事后)'][1]}")
    return table


def random_walk_like(close: pd.Series, seed: int) -> pd.Series:
    """与 close 同长度、同日波动、零漂移的随机游走（真实 Sharpe = 0 的对照组）"""
    rng = np.random.default_rng(seed)
    vol = close.pct_change().std()
    return pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, vol, len(close)))), index=close.index)


def interpretation_guide(close: pd.Series, res_real: dict, res_noise: dict):
    print("\n" + "=" * 60)
    print("第四部分：如何解读——与随机游走对照组比较")
    print("=" * 60)

    interpret(res_real, "沪深 300")
    interpret(res_noise, "随机游走对照组（零漂移，种子 7；与沪深 300 同样长度和日波动）")

    # 安慰剂检验：50 条随机游走上的 Walk-Forward 样本外 Sharpe 分布
    placebo = []
    for seed in range(50):
        r = run_walk_forward(random_walk_like(close, 100 + seed), "placebo")
        placebo.append(sharpe(r["滚动窗口"][0]))
    placebo = np.array(placebo)
    real = sharpe(res_real["滚动窗口"][0])
    pct = (placebo < real).mean()
    verdict = "明显好于运气" if pct >= 0.95 else "与运气难以区分"
    print(f"""
    🎲 安慰剂检验：在 50 条零漂移随机游走（种子 100~149）上跑同样的滚动 Walk-Forward
       对照组样本外 Sharpe：中位数 {np.median(placebo):.2f}，95% 分位 {np.percentile(placebo, 95):.2f}
       沪深 300 滚动 Walk-Forward 样本外 Sharpe {real:.2f}，高于 {pct:.0%} 的对照组 → {verdict}
       （注意：随机游走没有趋势也没有均值回复，真实市场的"运气基准"可能不同，这只是一个粗略的参照）
    """)
    print("""    ┌──────────────────────────────┬───────────────────────────────────────────┐
    │ 观察                         │ 解读                                      │
    ├──────────────────────────────┼───────────────────────────────────────────┤
    │ OOS Sharpe 接近 IS           │ 策略较稳健                                │
    │ OOS 远低于 IS                │ 过拟合，或市场状态变化                    │
    │ WFE 很低甚至为负             │ 样本内结果大部分是拟合噪声（无统一阈值）  │
    │ 选中参数在窗口间剧烈跳动     │ 没有稳定的最优值，优化可能只是在拟合噪声  │
    │ OOS 只在少数几段赚钱         │ 收益依赖特定行情                          │
    └──────────────────────────────┴───────────────────────────────────────────┘
    💡 对照组的真实 Sharpe 为 0，它的样本内最优 Sharpe 若为正，完全来自"挑最好的"这一步。
    💡 "全样本最优(事后)"那一行是研究报告里最容易误报的数字：它的参数是看完全部数据才选出来的。
    """)


# ==================== 第五部分：最终保留区与进阶方法 ====================


def holdout_and_advanced(close_all: pd.Series):
    print("\n" + "=" * 60)
    print("第五部分：最终保留区（Holdout）——只能用一次")
    print("=" * 60)

    pos, rets = grid_returns(close_all)
    research = rets.loc[: pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)]
    holdout = rets.loc[HOLDOUT_START:]
    # 研究结束时的决定：用研究区最后 TRAIN 天（与滚动窗口一致）选出最终参数
    final = tuple(int(x) for x in sharpe(research.iloc[-TRAIN:]).idxmax())
    rep_h = performance_report(holdout[final])
    rep_bh = performance_report(close_all.pct_change().loc[HOLDOUT_START:].fillna(0.0))
    print(f"""
    |←──────── 研究区（IS + Walk-Forward） ────────→|←── 保留区 {HOLDOUT_START[:4]} 起 ──→|

    研究区最后 {TRAIN} 天选出的最终参数：{final}
    保留区一次性结果：年化 {rep_h['年化收益']:+.2%}，Sharpe {rep_h['Sharpe']:.2f}，最大回撤 {rep_h['最大回撤']:.2%}
    同期买入持有：    年化 {rep_bh['年化收益']:+.2%}，Sharpe {rep_bh['Sharpe']:.2f}

    ⚠️ 纪律比方法更重要：看完保留区结果后如果回去改策略，保留区就失效了 —— 如实报告即可。
    """)

    print("""    进阶（Phase 9 机器学习中必需）：
    - Purging：标签涉及未来一段时间（如"未来 20 日收益"）时，删掉训练集中与测试集时间重叠的样本
    - Embargo：测试集之后再留一段间隔不用于训练
    - CPCV：把数据切成多块组合成多种训练/测试划分，得到多条样本外路径，而不是 Walk-Forward 的一条
    """)


# ==================== 主函数 ====================


def main():
    why_oos()

    close_all = get_index_daily("000300", START, END)["close"]
    close = close_all.loc[: pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)]  # 研究区
    res_real = walk_forward_demo(close)

    res_noise = run_walk_forward(random_walk_like(close, 7), "随机游走")
    interpretation_guide(close, res_real, res_noise)
    holdout_and_advanced(close_all)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, title in [(axes[0], res_real, "沪深 300 研究区"), (axes[1], res_noise, "随机游走对照组")]:
        for name in ["滚动窗口", "锚定窗口", "全样本最优(事后)", "买入持有"]:
            ax.plot((1 + res[name][0]).cumprod(), label=name, ls="--" if "事后" in name else "-")
        ax.set_title(f"{title}：样本外拼接净值")
        ax.legend()
        ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '09_walk_forward.png', subdir='phase-4')}")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把训练窗口改为 500 天和 1250 天，比较滚动窗口的 OOS Sharpe 与参数稳定性
    2. 把"选 Sharpe 最高的参数"改为"选邻域平均 Sharpe 最高的参数"（参数平原中心），OOS 是否更稳定？
    3. 用 03 课的 performance_report 输出每个测试段的完整指标，找出 OOS 亏损最大的一段，分析当时的行情
    4. 对 601088 个股（用 get_stock_daily_adj 的 adj_close）重复本课实验，结论是否一致？
    """)


if __name__ == "__main__":
    main()
