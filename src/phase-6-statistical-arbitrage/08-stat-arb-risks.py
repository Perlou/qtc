"""
统计套利的风险：协整断裂与拥挤
==============================

学习目标：
    1. 理解"套利的极限"：价差先扩大再修复时，高杠杆的套利者可能等不到修复就被迫平仓
    2. 用滚动协整检验监控配对关系，并认识滚动检验本身的噪声（误报）
    3. 用模拟数据演示协整断裂：检测有多滞后、止损和时间止损能挽回多少
    4. 用一个简化的多基金模拟重现 2007 年 8 月 Quant Quake 的机制：拥挤 + 杠杆 + 被迫平仓
    5. 掌握统计套利的风控工具箱与压力测试

核心概念：
    - 套利的极限（Shleifer & Vishny, 1997）：用别人的钱、有杠杆、有考核 → 在最差的时刻被迫平仓
    - 协整断裂：协整由基本面维持，基本面一变就会断（并购、转型、政策……）
    - 拥挤：相似策略的资金越多，收益越低、彼此相关性越高、踩踏风险越大
    - Quant Quake：2007 年 8 月量化多空基金集体巨亏，事后解释为拥挤头寸的连锁去杠杆

前置知识：
    - 本阶段 01–04 课

环境要求：
    - pip install akshare pandas numpy statsmodels matplotlib
    - 第二部分用真实配对（无网络时用 QTC_OFFLINE=1 运行，优先读本地缓存；无缓存时改用模拟的协整价格组），
      其余部分均为固定随机种子的模拟
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
from statsmodels.tsa.stattools import coint

from stat_arb_utils import (BANKS, FORM_END, PAIR, half_life, load_bank_panel, ols_hedge, pair_backtest,
                            pair_positions, perf_stats, stats_table, synthetic_coint_pair)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)


def rolling_coint(y: pd.Series, x: pd.Series, window: int = 250, step: int = 20) -> pd.DataFrame:
    """每 step 天，用过去 window 天做一次 Engle-Granger 检验，并估计价差半衰期"""
    out = {}
    for end in range(window, len(y) + 1, step):
        ys, xs = y.iloc[end - window:end], x.iloc[end - window:end]
        _, b = ols_hedge(ys, xs)
        out[y.index[end - 1]] = {"p": coint(ys, xs)[1], "hl": half_life(ys - b * xs), "beta": b}
    return pd.DataFrame(out).T


# ==================== 第一部分：套利的极限 ====================


def limits_of_arbitrage():
    print("=" * 60)
    print("第一部分：套利的极限 —— 对的方向，错的时机")
    print("=" * 60)

    print("""
    "市场保持非理性的时间，可能比你保持偿付能力的时间更长。"（常被归于凯恩斯）

    模拟：在价差偏离 2σ 时开仓，之后价差先继续扩大到约 5σ（第 25 天），再在第 100 天前修复。
    组合回撤超过 25% 时，投资人赎回 / 追加保证金 → 被迫在当天平仓。
    """)
    rng = np.random.default_rng(1997)
    days = 120
    t = np.arange(days + 1)
    path = np.where(t <= 25, 2 + 3 * t / 25, 5 * np.exp(-(t - 25) / 15))  # 价差（以 σ 计）
    path = path + np.concatenate([[0], np.cumsum(rng.normal(0, 0.15, days))]) * (t < 100)
    sigma = 0.02  # 1σ 价差 = 2% 名义金额
    pnl_per_unit = -(path - path[0]) * sigma  # 做空价差：价差扩大即亏损
    rows = []
    for lev in [1, 2, 4, 8]:
        nav = 1 + lev * pnl_per_unit
        dd = nav / np.maximum.accumulate(nav) - 1
        hit = np.where(dd < -0.25)[0]
        if len(hit):
            d = hit[0]
            rows.append([f"{lev}x", f"{dd.min():.1%}", f"第 {d} 天被迫平仓", f"{nav[d] - 1:+.1%}"])
        else:
            rows.append([f"{lev}x", f"{dd.min():.1%}", "坚持到修复", f"{nav[-1] - 1:+.1%}"])
    print(pd.DataFrame(rows, columns=["杠杆", "持有期最大回撤", "结局", "最终收益"]).to_string(index=False))
    print("""
    💡 方向完全正确（价差最终回到了均值），但杠杆决定了你能不能活到那一天。
       被迫平仓的卖盘又会进一步推大价差 —— 这正是第四部分 Quant Quake 的机制。
    """)


# ==================== 第二部分：滚动协整检验（真实配对） ====================


def rolling_monitor(logp: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：滚动协整检验 —— 监控配对关系")
    print("=" * 60)

    a, b = PAIR
    roll = rolling_coint(logp[a], logp[b])
    g = roll.groupby(roll.index.year)
    table = pd.DataFrame({"检验次数": g.size(), "p 值中位数": g["p"].median(),
                          "p<0.05 占比": g["p"].apply(lambda s: (s < 0.05).mean()),
                          "半衰期中位数": g["hl"].median(), "β 中位数": g["beta"].median()})
    print(f"   {BANKS[a]} ~ {BANKS[b]}：每 20 天用过去 250 天做一次 Engle-Granger 检验")
    print(table.round(3).to_string())
    print(f"   全部 {len(roll)} 次检验中 p < 0.05 的占比：{(roll['p'] < 0.05).mean():.0%}；"
          f"滚动 β 范围 {roll['beta'].min():.2f} ~ {roll['beta'].max():.2f}")
    print("""
    💡 预警信号：p 值从长期 < 0.05 变为持续 > 0.2、半衰期显著变长、β 大幅漂移。
       但滚动检验本身也有噪声 —— 下一部分用"真相已知"的模拟数据量化这一点。
    """)
    return roll


# ==================== 第三部分：协整断裂演示 ====================


def cointegration_break():
    print("\n" + "=" * 60)
    print("第三部分：协整断裂演示（模拟数据：第 900 天突发事件 + 之后基本面持续分化）")
    print("=" * 60)

    n, brk = 1500, 900
    kw = dict(n=n, beta=0.9, half_life=10, spread_sd=0.03, seed=8)
    df = synthetic_coint_pair(break_at=brk, break_drift=-0.003, break_jump=-0.08, **kw)
    stable = synthetic_coint_pair(**kw)
    roll = rolling_coint(df["A"], df["B"], window=250, step=10)
    roll_ok = rolling_coint(stable["A"], stable["B"], window=250, step=10)
    brk_date = df.index[brk]
    before = roll[roll.index < brk_date]
    after = roll[roll.index >= brk_date]
    alarm = (roll["p"] > 0.2).rolling(3).sum() >= 3  # 连续 3 次 p > 0.2 才报警，减少误报
    first = alarm[(alarm) & (alarm.index >= brk_date)].index
    lag = int(df.index.get_indexer([first[0]])[0] - brk) if len(first) else None
    false_alarm = int(alarm[alarm.index < brk_date].sum())
    print(f"   关系稳定的对照组：{len(roll_ok)} 次滚动检验中 p > 0.05 的占比 {(roll_ok['p'] > 0.05).mean():.0%}"
          "（关系从未改变，这些都是\"误报\"）")
    print(f"   断裂前：p < 0.05 的占比 {(before['p'] < 0.05).mean():.0%}，半衰期中位数 {before['hl'].median():.1f} 天")
    print(f"   断裂后：p < 0.05 的占比 {(after['p'] < 0.05).mean():.0%}，半衰期中位数 {after['hl'].median():.1f} 天")
    print(f"   报警规则（连续 3 次 p > 0.2）：断裂前误报 {false_alarm} 次；断裂后首次报警在"
          f"{'第 ' + str(lag) + ' 天' if lag is not None else '样本内未触发'}")

    # 协整监控：报警后暂停交易，滚动 p 值回到 0.05 以下再恢复
    state, paused = [], False
    for d, p in roll["p"].items():
        if alarm.loc[d]:
            paused = True
        elif p < 0.05:
            paused = False
        state.append(not paused)
    tradable = pd.Series(state, index=roll.index).reindex(df.index).ffill().fillna(True).astype(bool)
    tradable = tradable.shift(1, fill_value=True)  # 检验结果在当天收盘后才知道

    form = df.iloc[:500]  # 形成期：前 500 天估计 β 与半衰期；之后交易
    _, beta = ols_hedge(form["A"], form["B"])
    hl = half_life(form["A"] - beta * form["B"])
    lb = int(np.clip(round(2 * hl), 10, 120))
    spread = df["A"] - beta * df["B"]
    z = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
    z.iloc[:500] = np.nan
    res = {}
    for label, stop, mh, zz in [("无止损", 1e9, None, z), ("Z止损+时间止损", 4.0, int(3 * hl), z),
                                ("再加协整监控", 4.0, int(3 * hl), z.where(tradable))]:
        bt = pair_backtest(pair_positions(zz, 2.0, 0.5, stop, mh), beta, df["A"], df["B"])
        res[f"{label}·断裂前"] = perf_stats(bt["net"].iloc[500:brk], bt["pos"].iloc[500:brk])
        res[f"{label}·断裂后"] = perf_stats(bt["net"].iloc[brk:], bt["pos"].iloc[brk:])
    table = stats_table(res).loc[["总收益", "Sharpe", "最大回撤", "开仓次数", "持仓占比"]]
    print(f"\n📋 配对交易（形成期 β={beta:.3f}，半衰期 {hl:.1f} 天，窗口 {lb} 天，扣全部成本）：")
    for i in range(0, 6, 2):
        print(table.iloc[:, i:i + 2].to_string())
    print("""
    💡 观察要点：
       1. 检测有滞后：滚动窗口里还装着断裂前的数据，p 值要过一段时间才会升高
       2. 即使关系从未改变，滚动 p 值也会时不时超过 0.05 —— 不要凭一次 p 值上升就判定断裂
       3. 对比前两组：滚动 Z-score 会"适应"漂移的价差，Z 止损未必会被触发 ——
          亏损来自一次次逆着趋势开仓。关系层面的监控（第三组）才直接针对断裂，
          代价是误报时会错过正常的交易机会；结合基本面判断才是根本
    """)
    return df, roll, brk_date


# ==================== 第四部分：Quant Quake 机制模拟 ====================


def simulate_quake(overlap: float, seed: int = 2007, n_funds: int = 10, days: int = 80) -> dict:
    """
    简化模型：所有基金持有同一个"拥挤因子"多空组合（占各自头寸的 overlap 比例）
      - 拥挤因子日收益 = 小正收益 + 噪声 - 价格冲击的变化；冲击由净卖出累积、每天衰减 15%
      - 1 号基金规模是其他基金的 3 倍，第 20–22 天因与本策略无关的亏损被迫清仓
      - 其他基金：回撤超过 8% 即按风控把头寸砍半（之后若仍超过阈值，继续砍半）
    """
    rng = np.random.default_rng(seed)
    lev = 4.0
    capital = np.ones(n_funds)
    capital[0] = 3.0
    peak = capital.copy()
    expo = lev * capital * overlap  # 各基金在拥挤因子上的名义敞口
    lam, decay = 0.01, 0.15  # 冲击系数：卖出 1 单位敞口（≈ 1 只小基金的全部净值）使因子下跌 1%
    impact, cum, cut_count = 0.0, 0.0, np.zeros(n_funds, dtype=int)
    navs, factor_cum = [capital / capital], []
    start_cap = capital.copy()
    for d in range(days):
        sell = np.zeros(n_funds)
        if 20 <= d <= 22:  # 外生冲击：1 号基金在 3 天内清仓
            sell[0] = expo[0] if d == 22 else expo[0] / (23 - d)
        hit = (capital / peak - 1 < -0.08) & (np.arange(n_funds) > 0) & (expo > 1e-6)
        sell[hit] += expo[hit] * 0.5
        cut_count += hit
        expo = expo - sell
        new_impact = (1 - decay) * impact + lam * sell.sum()
        r = 0.0004 + rng.normal(0, 0.003) - (new_impact - impact)
        impact = new_impact
        idio = rng.normal(0, 0.004, n_funds) * lev * (1 - overlap)
        capital = capital + expo * r + capital * idio
        peak = np.maximum(peak, capital)
        cum += r
        factor_cum.append(cum)
        navs.append(capital / start_cap)
    return {"factor_cum": np.array(factor_cum), "navs": np.array(navs), "cuts": cut_count,
            "hold_final": lev * overlap * cum}


def quant_quake():
    print("\n" + "=" * 60)
    print("第四部分：2007 年 8 月 Quant Quake 的机制模拟")
    print("=" * 60)

    print("""
    背景（Khandani & Lo 的事后解释）：次贷危机发酵，一个或多个大型多策略基金因其他业务亏损，
    被迫快速清算股票多空组合；大量量化基金持有相似头寸，抛售引发连锁去杠杆。
    大盘本身波动并不极端 —— 亏损来自"多空价差"的反向移动；之后价差大幅反弹。

    下面的模拟（固定随机种子，数字只说明机制）：10 只基金、4 倍杠杆，
    只改变"彼此头寸重合度"（拥挤程度），比较 1 号基金被迫清仓后的连锁反应。
    """)
    rows, sims = [], {}
    for overlap in [0.2, 0.5, 0.8]:
        sim = simulate_quake(overlap)
        sims[overlap] = sim
        f = sim["factor_cum"]
        navs = sim["navs"][:, 1:]  # 不含被迫清仓的 1 号基金
        trough = int(np.argmin(f[15:40]) + 15)
        rows.append([f"{overlap:.0%}", f"{f[trough] - f[19]:+.1%}", trough, f"{f[-1] - f[trough]:+.1%}",
                     int((sim["cuts"][1:] > 0).sum()), f"{navs.min(axis=0).mean() - 1:+.1%}",
                     f"{navs[-1].mean() - 1:+.1%}", f"{sim['hold_final']:+.1%}"])
    cols = ["头寸重合度", "因子最大跌幅", "谷底(第几天)", "谷底后反弹", "被迫减仓基金数", "其余基金平均最低净值",
            "其余基金平均期末", "不减仓·期末(近似)"]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("""
    💡 教训（对照上表）：
       1. "市场中性"不等于"无风险"：你中性于大盘，但暴露于"同类策略集体平仓"这一隐藏因子
       2. 策略的风险由持有者结构决定：头寸重合度越高，同一次外生抛售引发的跌幅和连锁减仓越严重
       3. 杠杆与风控规则决定能否活下来：价差之后会反弹，被迫在谷底减仓的基金错过了修复，
          比较"其余基金平均期末"与"不减仓·期末"即可看出差别
       4. 1998 年 LTCM 的收敛交易也是类似模式：高杠杆 + 价差持续扩大 → 危机
    """)
    return sims


# ==================== 第五部分：风控工具箱与压力测试 ====================


def risk_toolkit(logp: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：风控工具箱与压力测试")
    print("=" * 60)

    print("""
    ┌──────────────┬──────────────────────────────────────┬──────────────┐
    │ 工具         │ 做法                                 │ 防范         │
    ├──────────────┼──────────────────────────────────────┼──────────────┤
    │ Z-score 止损 │ |z| > 4 强制平仓                     │ 关系断裂     │
    │ 时间止损     │ 持仓 > 2–3 倍半衰期强制平仓          │ 回归变慢     │
    │ 滚动重估     │ 定期重做协整检验，不通过的移出组合   │ 关系漂移     │
    │ 配对分散     │ 大量、跨行业的配对，单对仓位小       │ 个股事件     │
    │ 事件过滤     │ 财报、重组公告、停牌前后不开新仓     │ 跳空         │
    │ 杠杆控制     │ 能承受价差先扩大再修复               │ 被迫平仓     │
    │ 拥挤监控     │ 跟踪与公开因子 / 同业指数的相关性    │ 踩踏         │
    │ 压力测试     │ 所有配对价差同时不利扩大 3σ          │ 尾部风险     │
    └──────────────┴──────────────────────────────────────┴──────────────┘
    """)
    a, b = PAIR
    form = logp.loc[:FORM_END]
    _, beta = ols_hedge(form[a], form[b])
    sd = (form[a] - beta * form[b]).std()
    loss_unit = 3 * sd / (1 + abs(beta))  # 价差不利变动 3σ 时，单位总敞口的亏损
    rows = [[f"{lev}x", f"{lev * loss_unit:.1%}"] for lev in [1, 2, 4, 8]]
    print(f"📋 压力测试：以 {BANKS[a]}~{BANKS[b]} 形成期价差标准差 {sd:.4f}（对数）为代表，"
          "假设所有配对同时向不利方向扩大 3σ：")
    print(pd.DataFrame(rows, columns=["总敞口 / 净值", "单日亏损（占净值）"]).to_string(index=False))
    print("""
    💡 3σ 只是起点：拥挤踩踏时价差可能扩大到 5σ 以上，而且所有配对同时发生（相关性趋于 1）。
       杠杆要以"能扛过历史最坏情况的若干倍"来设定，而不是以"平时的波动"来设定。
    """)


def plot_risks(roll_real: pd.DataFrame, brk_df, roll_brk: pd.DataFrame, brk_date, sims: dict, is_synth: bool):
    setup_chinese_font()
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    axes[0].plot(roll_real.index, roll_real["p"], marker=".", lw=0.8)
    axes[0].axhline(0.05, color="r", ls="--", lw=0.8)
    axes[0].set_title(f"{BANKS[PAIR[0]]}~{BANKS[PAIR[1]]} 滚动协整 p 值（250 天窗口）{'【合成数据】' if is_synth else ''}")
    axes[1].plot(roll_brk.index, roll_brk["p"], marker=".", lw=0.8)
    axes[1].axvline(brk_date, color="k", ls=":", label="断裂点")
    axes[1].axhline(0.05, color="r", ls="--", lw=0.8)
    axes[1].legend(), axes[1].set_title("模拟：协整断裂前后的滚动 p 值")
    for ov, sim in sims.items():
        axes[2].plot(sim["factor_cum"] * 100, label=f"头寸重合度 {ov:.0%}")
    axes[2].axvspan(20, 22, color="orange", alpha=0.3, label="1 号基金被迫清仓")
    axes[2].legend(), axes[2].set_title("模拟：拥挤因子累计收益（%）")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '08_stat_arb_risks.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    prices, _, is_synth = load_bank_panel()
    logp = np.log(prices)

    limits_of_arbitrage()
    roll_real = rolling_monitor(logp)
    brk_df, roll_brk, brk_date = cointegration_break()
    sims = quant_quake()
    risk_toolkit(logp)
    plot_risks(roll_real, brk_df, roll_brk, brk_date, sims, is_synth)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第一部分中，把回撤阈值从 25% 改为 15% 和 40%，各杠杆下的结局如何变化？
    2. 第三部分改用 120 天和 500 天的滚动窗口，检测滞后与误报率如何权衡？
    3. 第四部分把杠杆从 4 倍降到 2 倍、风控阈值从 8% 放宽到 15%，被迫减仓的基金数如何变化？
    4. 对 01 课的全部 36 个银行配对做滚动协整检验，统计"至少有一半窗口 p<0.05"的配对有几个
    """)


if __name__ == "__main__":
    main()
