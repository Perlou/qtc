"""
Z-score 信号与半衰期
====================

学习目标：
    1. 理解 OU 过程是"平稳价差"的数学模型，会用回归估计半衰期
    2. 用半衰期设定 Z-score 窗口和时间止损
    3. 实现完整的 Z-score 配对交易：开仓 / 平仓 / 止损 / 冷却期，信号 shift(1) 次日执行
    4. 严格区分样本内与样本外：形成期估计参数，交易期检验；体会参数网格的过拟合
    5. 计入 A 股真实成本：佣金、滑点、印花税、融券费；以及"借不到券"时的替代方案

核心概念：
    - OU 过程：ds = θ(μ - s)dt + σdW；半衰期 h = ln2 / θ
    - 离散估计：Δs_t = a + b·s_{t-1} + ε，h = -ln2 / ln(1+b)
    - Z-score：z_t = (s_t - 滚动均值) / 滚动标准差；±2 开仓、±0.5 平仓、±4 止损

前置知识：
    - 01、02 课；Phase 4 回测（未来函数、交易成本）

环境要求：
    - pip install akshare pandas numpy statsmodels matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行：优先读本地缓存；无缓存时 utils 返回合成行情，
      本课检测到后改用模拟的协整价格组（打印 ⚠️ 使用合成数据）
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

from stat_arb_utils import (BANKS, FORM_END, PAIR, TEST_START, half_life, load_bank_panel, ols_hedge,
                            pair_backtest, pair_positions, perf_stats, stats_table)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

ENTRY, EXIT, STOP = 2.0, 0.5, 4.0


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """滚动 Z-score：只用 t 及之前的 lookback 天，t 日收盘可得"""
    return (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()


def run_period(logp, a, b, beta, lookback, start, end, entry=ENTRY, max_hold=None, **bt_kw):
    """在 [start, end] 内交易：价差和 Z-score 用全部历史计算（滚动窗口无前视），区间外不持仓"""
    spread = logp[a] - beta * logp[b]
    z = zscore(spread, lookback)
    z = z.where((z.index >= pd.Timestamp(start)) & (z.index <= pd.Timestamp(end)))
    pos = pair_positions(z, entry, EXIT, STOP, max_hold)
    sl = slice(pd.Timestamp(start), pd.Timestamp(end))
    bt = pair_backtest(pos, beta, logp[a], logp[b], **bt_kw).loc[sl]
    return bt, z.loc[sl], spread.loc[sl]


# ==================== 第一部分：OU 过程与半衰期 ====================


def ou_half_life():
    print("=" * 60)
    print("第一部分：OU 过程与半衰期")
    print("=" * 60)

    print("""
    OU 过程：ds_t = θ(μ - s_t)dt + σ dW_t
      θ：回归速度（弹簧硬度）  μ：长期均值  σ：噪声强度（风力）
    偏离按 e^{-θt} 衰减 → 半衰期 h = ln2 / θ

    离散估计：Δs_t = a + b · s_{t-1} + ε_t  （等价于 AR(1)，系数 1+b）
              h = -ln2 / ln(1+b) ≈ -ln2 / b；b ≥ 0 时不回归
    """)
    rng = np.random.default_rng(2024)
    rows = []
    for true_hl in [5, 10, 20, 40, 80]:
        phi = 0.5 ** (1 / true_hl)
        est = []
        for _ in range(100):
            n = 750  # 约 3 年日频
            eps = rng.normal(0, 0.01, n)
            s = np.zeros(n)
            for t in range(1, n):
                s[t] = phi * s[t - 1] + eps[t]
            est.append(half_life(pd.Series(s)))
        est = np.array(est)
        rows.append([true_hl, np.median(est), np.percentile(est, 5), np.percentile(est, 95)])
    df = pd.DataFrame(rows, columns=["真实半衰期", "估计中位数", "5% 分位", "95% 分位"]).round(1)
    print("🎲 模拟 OU 价差（每种 100 次，每次 750 天），检验半衰期估计的准确度：")
    print(df.to_string(index=False))
    print("""
    💡 半衰期越长，估计越不准（区间越宽）：样本里"完整回归"的次数越少，信息越少。
       实际使用时不要把半衰期估计当作精确值，取整数量级（几天 / 几周 / 几个月）即可。
    """)


# ==================== 第二部分：真实配对的半衰期与参数设定 ====================


def pair_parameters(logp: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("第二部分：真实配对的半衰期 → Z-score 窗口与时间止损")
    print("=" * 60)

    a, b = PAIR
    form = logp.loc[:FORM_END]
    _, beta = ols_hedge(form[a], form[b])
    hl = half_life(form[a] - beta * form[b])
    lookback = int(np.clip(round(2 * hl), 10, 120))
    max_hold = int(np.ceil(3 * hl))
    print(f"""
    配对：{BANKS[a]}（A） vs {BANKS[b]}（B），形成期 {form.index[0].date()} ~ {form.index[-1].date()}
    对冲比 β = {beta:.3f}   形成期价差半衰期 ≈ {hl:.1f} 个交易日

    ┌──────────────────┬──────────────────────────────┬───────────────┐
    │ 用途             │ 规则                         │ 本例取值      │
    ├──────────────────┼──────────────────────────────┼───────────────┤
    │ Z-score 窗口 L   │ 约 2 倍半衰期（限制 10–120） │ {lookback:>4d} 天       │
    │ 时间止损         │ 持仓超过 3 倍半衰期          │ {max_hold:>4d} 天       │
    │ 开仓 / 平仓      │ |z| 越过开仓线 / 回到平仓线  │ ±{ENTRY:.1f} / ±{EXIT:.1f}   │
    │ 止损             │ |z| 越过止损线，之后冷却     │ ±{STOP:.1f}          │
    └──────────────────┴──────────────────────────────┴───────────────┘
    """)
    rows = []
    for y in range(logp.index[0].year, logp.index[-1].year + 1):
        w = logp.loc[str(y)]
        s = w[a] - beta * w[b]
        rows.append([y, half_life(s), s.std()])
    print("📋 用形成期 β 构造的价差，每个自然年的半衰期与波动（交易期的数值是事后观察）：")
    print(pd.DataFrame(rows, columns=["年份", "半衰期(天)", "价差标准差"]).round(3).to_string(index=False))
    print("""
    💡 半衰期太长（如 > 60 天）：资金占用久、不确定性大；太短（< 1 天）：多半是噪声或
       买卖价差反弹，扣除成本后无利可图。半衰期逐年明显变长，是关系减弱的预警（08 课）。
    """)
    return dict(beta=beta, hl=hl, lookback=lookback, max_hold=max_hold)


# ==================== 第三部分：Z-score 信号与前视偏差 ====================


def zscore_signals(logp: pd.DataFrame, prm: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：Z-score 信号（样本外交易期）")
    print("=" * 60)

    a, b = PAIR
    end = logp.index[-1]
    bt, z, spread = run_period(logp, a, b, prm["beta"], prm["lookback"], TEST_START, end,
                               max_hold=prm["max_hold"])
    pos = bt["pos"]
    print(f"   交易期 {z.index[0].date()} ~ {z.index[-1].date()}，共 {len(z)} 天")
    print(f"   |z| > {ENTRY} 的天数：{int((z.abs() > ENTRY).sum())}；|z| > {STOP} 的天数：{int((z.abs() > STOP).sum())}")
    print(f"   持仓天数：做多价差 {int((pos > 0).sum())}，做空价差 {int((pos < 0).sum())}，空仓 {int((pos == 0).sum())}")

    # 错误示范：用交易期全样本的均值和标准差（包含未来信息）
    z_cheat = (spread - spread.mean()) / spread.std()
    pos_cheat = pair_positions(z_cheat, ENTRY, EXIT, STOP, prm["max_hold"])
    bt_cheat = pair_backtest(pos_cheat, prm["beta"], logp[a].loc[TEST_START:], logp[b].loc[TEST_START:])
    print("\n📋 滚动 Z-score vs ❌ 全样本 Z-score（未扣成本）：")
    print(stats_table({
        "滚动窗口（正确）": perf_stats(bt["gross"], bt["pos"]),
        "❌ 全样本均值/标准差": perf_stats(bt_cheat["gross"], bt_cheat["pos"]),
    }).to_string())
    print("""
    💡 全样本 Z-score 在 t 日用到了 t 之后价差的均值和标准差 —— 它"知道"价差最终会回到哪里。
       无论它的结果看起来更好还是更差，都不可信。同理，β 也不能用全样本估计（第四部分）。
    """)
    return bt


# ==================== 第四部分：样本内 vs 样本外，参数网格的过拟合 ====================


def in_vs_out_of_sample(logp: pd.DataFrame, prm: dict):
    print("\n" + "=" * 60)
    print("第四部分：样本内 vs 样本外 —— 参数网格的过拟合")
    print("=" * 60)

    a, b = PAIR
    form_start = logp.index[0]
    rows = []
    for entry in [1.0, 1.5, 2.0, 2.5]:
        for mult in [1, 2, 3]:
            lb = int(np.clip(round(mult * prm["hl"]), 10, 120))
            ins, _, _ = run_period(logp, a, b, prm["beta"], lb, form_start, FORM_END, entry, prm["max_hold"])
            oos, _, _ = run_period(logp, a, b, prm["beta"], lb, TEST_START, logp.index[-1], entry, prm["max_hold"])
            rows.append([entry, lb, perf_stats(ins["net"])["Sharpe"], perf_stats(oos["net"])["Sharpe"]])
    grid = pd.DataFrame(rows, columns=["开仓阈值", "窗口 L", "样本内 Sharpe", "样本外 Sharpe"])
    print("📋 12 组参数（已扣全部成本），样本内 = 形成期，样本外 = 交易期：")
    print(grid.round(2).to_string(index=False))
    best = grid.loc[grid["样本内 Sharpe"].idxmax()]
    rank_corr = grid["样本内 Sharpe"].rank().corr(grid["样本外 Sharpe"].rank())
    print(f"""
    样本内最优参数：开仓 {best['开仓阈值']}，窗口 {int(best['窗口 L'])} → 样本内 Sharpe {best['样本内 Sharpe']:.2f}，
                    样本外 Sharpe {best['样本外 Sharpe']:.2f}
    样本外 Sharpe 中位数：{grid['样本外 Sharpe'].median():.2f}；样本内外 Sharpe 排名相关系数：{rank_corr:.2f}

    💡 样本内挑出来的"最优"参数，在样本外通常会打折扣 —— 网格里的最高值部分来自运气。
       另外，形成期的 β 和半衰期本身也是在样本内估计的，样本内回测天然偏乐观。
       可靠的做法：只看样本外，并且关注"参数平原"（相邻参数结果是否稳定），而不是单点最优。
    """)


# ==================== 第五部分：交易成本与 A 股做空约束 ====================


def costs_and_shorting(logp: pd.DataFrame, market: pd.Series, prm: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第五部分：交易成本与 A 股做空约束")
    print("=" * 60)

    a, b = PAIR
    end = logp.index[-1]
    bt, _, _ = run_period(logp, a, b, prm["beta"], prm["lookback"], TEST_START, end, max_hold=prm["max_hold"])
    print("""
    成本假设（示例，以交易所和券商最新规则为准）：
      佣金 0.025%（双边）+ 滑点 0.05%（双边）+ 印花税 0.05%（仅卖出，含融券卖出）
      融券年化费率 8%，按空头名义金额逐日计提
    """)

    # 替代方案 1：借不到券 —— 只持有价差中的多头腿（z 偏低买 A，z 偏高买 B）
    pos = bt["pos"]  # 已 shift，等于当天持有的方向
    ra = np.expm1(logp[a].diff()).reindex(pos.index).fillna(0)
    rb = np.expm1(logp[b].diff()).reindex(pos.index).fillna(0)
    long_w = pd.DataFrame({"A": (pos > 0).astype(float), "B": (pos < 0).astype(float)})
    turn = long_w.diff().abs().sum(axis=1).fillna(long_w.sum(axis=1))
    sells = (-long_w.diff()).clip(lower=0).sum(axis=1).fillna(0)
    long_only = long_w["A"] * ra + long_w["B"] * rb - turn * 0.00075 - sells * 0.0005

    # 替代方案 2：空头腿换成沪深 300（ETF 融券或股指期货），对冲比用形成期对市场的 Beta
    lm = np.log(market)
    form = pd.concat([logp[a].diff(), lm.diff()], axis=1).loc[:FORM_END].dropna()
    beta_m = float(np.cov(form.iloc[:, 0], form.iloc[:, 1])[0, 1] / form.iloc[:, 1].var())
    w_a = (pos > 0).astype(float) / (1 + beta_m)
    rm = np.expm1(lm.diff()).reindex(pos.index).fillna(0)
    hedged = w_a * ra - w_a * beta_m * rm - w_a.diff().abs().fillna(w_a) * (1 + beta_m) * 0.00075

    mret = rm
    table = stats_table({
        "配对·未扣成本": perf_stats(bt["gross"], bt["pos"]),
        "配对·扣交易成本": perf_stats(bt["net_no_borrow"], bt["pos"]),
        "配对·再扣融券费": perf_stats(bt["net"], bt["pos"]),
        "借不到券·只做多": perf_stats(long_only, pos),
        "多A+空沪深300": perf_stats(hedged, pos.clip(lower=0)),
    })
    print(f"📋 交易期 {bt.index[0].date()} ~ {bt.index[-1].date()}（{BANKS[a]} vs {BANKS[b]}）：")
    print(table.to_string())
    corr = pd.Series({
        "配对": bt["net"].corr(mret), "只做多": long_only.corr(mret), "多A+空沪深300": hedged.corr(mret),
    })
    print("\n   与市场日收益的相关系数：" + "，".join(f"{k} {v:.2f}" for k, v in corr.items()))
    print(f"   成本拆分（交易期合计，占资金比例）：交易成本 {bt['trade_cost'].sum():.2%}，"
          f"融券费 {bt['borrow_cost'].sum():.2%}")
    print(f"   形成期 {BANKS[a]} 对市场指数的 Beta = {beta_m:.2f}（沪深 300；无真实数据时为代理）")
    print("""
    💡 观察要点：
       1. 两条腿、四笔交易，加上融券费，成本对配对交易的侵蚀比单边策略更严重
       2. "只做多"不再市场中性：收益里混入了大盘涨跌，风险结构完全变了
       3. "多 A + 空指数"只在信号为"做多价差"时开仓，只对冲了市场 Beta：
          它的收益来源已变成"A 相对大盘"的表现，与配对价差是否修复关系不大，
          表现好坏不能归功于配对信号；用股指期货做空还要承担基差成本（06 课），这里未计入
    """)
    return bt


def plot_backtest(bt: pd.DataFrame, logp: pd.DataFrame, prm: dict, is_synth: bool):
    setup_chinese_font()
    a, b = PAIR
    spread = (logp[a] - prm["beta"] * logp[b])
    z = zscore(spread, prm["lookback"]).loc[bt.index]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(spread.loc[bt.index])
    axes[0].set_title(f"价差 log{BANKS[a]} - {prm['beta']:.2f}·log{BANKS[b]}（交易期）{'【合成数据】' if is_synth else ''}")
    axes[1].plot(z, lw=0.8)
    for lvl, c in [(ENTRY, "r"), (-ENTRY, "r"), (EXIT, "g"), (-EXIT, "g")]:
        axes[1].axhline(lvl, color=c, ls="--", lw=0.8)
    axes[1].fill_between(bt.index, -4, 4, where=bt["pos"] != 0, color="orange", alpha=0.15, label="持仓")
    axes[1].set_ylim(-4.5, 4.5), axes[1].legend(loc="upper left")
    axes[1].set_title(f"滚动 Z-score（L = {prm['lookback']}）")
    for col, label in [("gross", "未扣成本"), ("net_no_borrow", "扣交易成本"), ("net", "再扣融券费")]:
        axes[2].plot((1 + bt[col]).cumprod(), label=label)
    axes[2].legend(), axes[2].set_title("净值")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '03_zscore_backtest.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    prices, market, is_synth = load_bank_panel()
    logp = np.log(prices)

    ou_half_life()
    prm = pair_parameters(logp)
    zscore_signals(logp, prm)
    in_vs_out_of_sample(logp, prm)
    bt = costs_and_shorting(logp, market, prm)
    plot_backtest(bt, logp, prm, is_synth)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 去掉时间止损（max_hold=None）和冷却期，样本外结果如何变化？止损触发了几次？
    2. 实现"滚动推进"（walk-forward）：每 6 个月用过去 2 年重新估计 β 和半衰期，再交易下 6 个月
    3. 把融券费率改为 0、4%、12%，画出样本外净收益随费率变化的曲线，找出盈亏平衡费率
    4. 换成 01 课选出的另一对配对，重复本课全部流程，参数要在形成期确定后就不再改动
    """)


if __name__ == "__main__":
    main()
