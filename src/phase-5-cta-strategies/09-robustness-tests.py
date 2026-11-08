"""
稳健性检验：扰动、分段、跨品种
==============================

学习目标：
    1. 对一个"看起来还行"的 CTA 策略做系统的稳健性检验，而不是只看一条净值曲线
    2. 参数维度：参数随机扰动 ±20%，看绩效分布
    3. 时间维度：分年度统计、收益集中度、滚动 Sharpe、简单 Walk-Forward
    4. 截面维度：同一参数跨品种、跨板块
    5. 成本与数据维度：成本 ×2 / ×3、盈亏平衡成本；换主力规则、换复权方式
    6. 输出一份稳健性报告清单

核心概念：
    - 稳健性：策略表现是否依赖某个特定的参数、时期、品种或成本假设？依赖越少越可信
    - 盈亏平衡成本：成本提高到多少倍时 Sharpe 归零
    - Walk-Forward：参数只用过去数据选择，在下一段样本外检验，滚动推进

前置知识：
    - 07-position-sizing-atr.py（波动率目标）、08-parameter-optimization.py（参数平原）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存时读缓存，否则使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import (SPECS, UNIVERSE, build_continuous, cost_rate, get_contracts, load_universe, max_drawdown,
                       perf_stats, sharpe, stats_table)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

BASE_FAST, BASE_SLOW = 20, 100
TARGET_VOL, MAX_LEV = 0.10, 3.0


def strategy(df: pd.DataFrame, s: str, fast: int = BASE_FAST, slow: int = BASE_SLOW, cost_mult: float = 1.0,
             price_col: str = "close") -> pd.Series:
    """
    被检验的策略：双均线方向 × 波动率目标（单品种 10%），信号 shift(1) 次日持有，扣成本与换月成本
    price_col：用哪条价格序列计算信号（收益始终用比例调整价，即实际持仓收益）
    """
    px = df[price_col]
    ret = df["close"].pct_change().fillna(0)
    ma_s = px.rolling(slow).mean()
    sig = np.sign(px.rolling(fast).mean() - ma_s).where(ma_s.notna(), 0.0)
    vol = ret.ewm(span=40, adjust=False, min_periods=20).std() * np.sqrt(252)
    w = (sig * (TARGET_VOL / vol).clip(upper=MAX_LEV)).fillna(0)
    pos = w.shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(0) + 2 * pos.abs() * df["roll"].astype(float)
    return (pos * ret - turnover * cost_rate(s, df["raw_close"]) * cost_mult).iloc[slow:]


def portfolio(data: dict, **kw) -> pd.Series:
    """各品种等权（每个品种已经各自波动率目标），缺失日期按 0 收益处理"""
    return pd.DataFrame({s: strategy(df, s, **kw) for s, df in data.items()}).fillna(0).mean(axis=1)


# ==================== 第一部分：基准策略 ====================


def base_strategy(data: dict) -> pd.Series:
    print("=" * 60)
    print("第一部分：被检验的策略")
    print("=" * 60)

    print(f"""
    策略：每个品种用 MA{BASE_FAST}/MA{BASE_SLOW} 双均线方向（±1）× 波动率目标 {TARGET_VOL:.0%}（杠杆上限 {MAX_LEV:.0f} 倍），
          {len(data)} 个品种等权。信号 shift(1) 次日执行，扣手续费、滑点和换月成本。
    基准：各品种 1 倍做多的等权组合（"买入持有一篮子商品"）

    ┌────────────────────────────────────────────────────────────┐
    │  参数维度 ── 参数扰动 ±20%，绩效是否平滑变化？              │
    │  时间维度 ── 分年度 / 滚动窗口，每段是否都有效？            │
    │  截面维度 ── 换品种 / 换板块，逻辑是否通用？                │
    │  成本维度 ── 成本 ×2、×3，策略是否还活着？                  │
    │  数据维度 ── 换主力规则、换复权方式，结论是否改变？         │
    └────────────────────────────────────────────────────────────┘
    """)
    port = portfolio(data)
    bench = pd.DataFrame({s: df["close"].pct_change() for s, df in data.items()}).fillna(0).mean(axis=1).loc[port.index]
    print(stats_table({"等权买入持有": perf_stats(bench), "趋势组合": perf_stats(port)}).to_string())
    print(f"""
    📌 趋势组合的年化波动只有 {port.std() * np.sqrt(252):.1%}：每个品种各自 {TARGET_VOL:.0%} 的波动在等权平均中被分散掉了
       （10 课会在组合层再整体缩放到目标波动）。比较两者时看 Sharpe 和回撤，而不是总收益。
       买入持有一篮子商品的表现很依赖样本期间的商品大势，这段时间它的 Sharpe 为 {sharpe(bench):.2f}。
    """)
    return port


# ==================== 第二部分：参数扰动 ====================


def perturbation(data: dict, n: int = 200, pct: float = 0.2) -> dict:
    print("\n" + "=" * 60)
    print(f"第二部分：参数扰动（每个参数在 ±{pct:.0%} 内均匀抽样，{n} 次）")
    print("=" * 60)

    rng = np.random.default_rng(42)
    base = sharpe(portfolio(data))
    s0 = "RB" if "RB" in data else next(iter(data))
    base_single = sharpe(strategy(data[s0], s0))
    rows = []
    for _ in range(n):
        f = int(round(BASE_FAST * rng.uniform(1 - pct, 1 + pct)))
        sl = int(round(BASE_SLOW * rng.uniform(1 - pct, 1 + pct)))
        rows.append([f, sl, sharpe(portfolio(data, fast=f, slow=sl)), sharpe(strategy(data[s0], s0, f, sl))])
    t = pd.DataFrame(rows, columns=["fast", "slow", "组合", s0])
    desc = pd.DataFrame({
        "基准参数": [base, base_single],
        "扰动中位数": [t["组合"].median(), t[s0].median()],
        "5%分位": [t["组合"].quantile(0.05), t[s0].quantile(0.05)],
        "95%分位": [t["组合"].quantile(0.95), t[s0].quantile(0.95)],
        "Sharpe>0占比": [(t["组合"] > 0).mean(), (t[s0] > 0).mean()],
    }, index=["组合", SPECS[s0]["name"]])
    print(desc.to_string(formatters={c: "{:+.2f}".format for c in desc.columns[:4]} | {"Sharpe>0占比": "{:.0%}".format}))
    print(f"""
    💡 判断：扰动后 Sharpe 的中位数应接近基准值，且分布不应出现大量负值。
       组合的分布（5%~95% 分位宽度 {desc.loc['组合', '95%分位'] - desc.loc['组合', '5%分位']:.2f}）
       比单品种（{desc.iloc[1]['95%分位'] - desc.iloc[1]['5%分位']:.2f}）{'更窄' if desc.loc['组合', '95%分位'] - desc.loc['组合', '5%分位'] < desc.iloc[1]['95%分位'] - desc.iloc[1]['5%分位'] else '更宽'}
       —— 多品种平均本身就降低了对参数的敏感度。
    """)
    return {"perturb_pos": (t["组合"] > 0).mean(), "perturb": t}


# ==================== 第三部分：分段检验 ====================


def segmentation(data: dict, port: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：分段检验 —— 按年份切开看")
    print("=" * 60)

    g = port.groupby(port.index.year)
    yearly = pd.DataFrame({
        "天数": g.size(),
        "收益": g.apply(lambda x: (1 + x).prod() - 1),
        "年化波动": g.std() * np.sqrt(252),
        "Sharpe": g.apply(sharpe),
        "最大回撤": g.apply(max_drawdown),
    })
    print(yearly.to_string(formatters={"收益": "{:+.1%}".format, "年化波动": "{:.1%}".format,
                                       "Sharpe": "{:+.2f}".format, "最大回撤": "{:.1%}".format}))
    log_y = np.log1p(yearly["收益"])
    total = log_y.sum()
    top_share = log_y.max() / total if total > 0 else np.nan
    print(f"""
    📊 盈利年份 {int((yearly['收益'] > 0).sum())}/{len(yearly)}；
       最好的一年贡献了全部对数收益的 {top_share:.0%}（总对数收益 {total:+.1%}；总收益为负时此比例无意义）
    💡 趋势策略的收益集中在少数大行情年份是正常的，但要意识到这一点，并评估"未来几年没有大行情"的情景。
    """)

    # 简单 Walk-Forward：每年年初用过去 2 年的数据，从候选慢线中选组合 Sharpe 最高的，用于当年
    cands = [40, 60, 100, 160, 250]
    rets = {sl: portfolio(data, slow=sl) for sl in cands}
    panel = pd.DataFrame(rets).fillna(0)
    years = sorted(set(panel.index.year))
    wf, chosen = [], []
    for y in years[2:]:
        past = panel[(panel.index.year >= y - 2) & (panel.index.year < y)]
        best = past.apply(sharpe).idxmax()
        chosen.append(f"{y}:{best}")
        wf.append(panel.loc[panel.index.year == y, best])
    wf = pd.concat(wf)
    fixed = panel.loc[wf.index, BASE_SLOW]
    ens = panel.loc[wf.index].mean(axis=1)
    print(f"📊 Walk-Forward（候选慢线 {cands}，用过去 2 年选参，下一年检验）：")
    print(f"   每年选中的慢线：{'  '.join(chosen)}")
    print(stats_table({"WF选参": perf_stats(wf), f"固定 {BASE_SLOW}": perf_stats(fixed), "候选等权集成": perf_stats(ens)}).to_string())
    print("""
    💡 WF 的样本外结果才是"如果当时用这套选参流程"真正能拿到的。
       如果 WF 选参不比固定参数或等权集成更好，说明"根据近期表现调参"没有带来价值。
    """)
    return {"yearly": yearly, "top_share": top_share, "wf_sharpe": sharpe(wf), "is_sharpe": sharpe(port)}


# ==================== 第四部分：跨品种 / 跨板块 ====================


def cross_section(data: dict) -> dict:
    print("\n" + "=" * 60)
    print("第四部分：跨品种与跨板块")
    print("=" * 60)

    rows = []
    for s, df in data.items():
        r = strategy(df, s)
        rows.append([s, SPECS[s]["name"], SPECS[s]["sector"], sharpe(r), max_drawdown(r)])
    t = pd.DataFrame(rows, columns=["代码", "品种", "板块", "Sharpe", "最大回撤"])
    print(t.to_string(index=False, formatters={"Sharpe": "{:+.2f}".format, "最大回撤": "{:.1%}".format}))
    sec = t.groupby("板块")["Sharpe"].agg(["mean", "count", lambda x: (x > 0).sum()])
    sec.columns = ["平均Sharpe", "品种数", "Sharpe>0"]
    print("\n📊 按板块：")
    print(sec.to_string(formatters={"平均Sharpe": "{:+.2f}".format}))
    pos = (t["Sharpe"] > 0).mean()
    print(f"""
    📊 同一参数下 Sharpe > 0 的品种占比 {pos:.0%}，有 {int((sec['平均Sharpe'] > 0).sum())}/{len(sec)} 个板块平均 Sharpe > 0
    💡 一个有真实逻辑的趋势策略，应当在"多数"品种上有正期望（不要求每个都赚钱）；
       如果只在一个板块有效，要警惕结论其实是"那个板块在这段时间走出了大趋势"。
    """)
    return {"xs_pos": pos}


# ==================== 第五部分：成本与数据维度 ====================


def cost_and_data(data: dict) -> dict:
    print("\n" + "=" * 60)
    print("第五部分：成本敏感性与数据处理方式")
    print("=" * 60)

    mults = [0, 1, 2, 3, 5, 10]
    res = {m: sharpe(portfolio(data, cost_mult=m)) for m in mults}
    print("📊 成本倍数 → 组合 Sharpe：  " + "   ".join(f"×{m}: {v:+.2f}" for m, v in res.items()))
    lo, hi = 0.0, 50.0
    if sharpe(portfolio(data, cost_mult=hi)) > 0:
        breakeven = np.inf
    elif res[0] <= 0:
        breakeven = 0.0
    else:
        for _ in range(12):  # 二分法找 Sharpe = 0 的成本倍数
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sharpe(portfolio(data, cost_mult=mid)) > 0 else (lo, mid)
        breakeven = (lo + hi) / 2
    print(f"   盈亏平衡成本 ≈ 当前假设的 {breakeven:.1f} 倍（成本高于此值，Sharpe ≤ 0）")
    print("""
    💡 如果平衡成本只比假设成本高一点点，说明策略非常脆弱 —— 短周期、高换手的策略尤其常见。
       国内商品期货手续费不高，但滑点往往被低估；换月时还要额外考虑流动性切换的冲击。
    """)

    # 数据维度：① 信号用价差调整价  ② 主力规则改为"按成交量"
    alt = {}
    alt["比例调整(基准)"] = sharpe(portfolio(data))
    alt["信号用价差调整价"] = sharpe(pd.DataFrame(
        {s: strategy(df, s, price_col="close_diff") for s, df in data.items()}).fillna(0).mean(axis=1))
    vol_rule = {}
    for s in data:
        c = get_contracts(s)
        vol_rule[s] = build_continuous(c.assign(hold=c["volume"]))  # 用前一日成交量（而非持仓量）选主力
    alt["主力按成交量切换"] = sharpe(portfolio(vol_rule))
    n_roll_oi = sum(int(df["roll"].sum()) for df in data.values())
    n_roll_vol = sum(int(df["roll"].sum()) for df in vol_rule.values())
    print("📊 数据处理方式 → 组合 Sharpe：  " + "   ".join(f"{k}: {v:+.2f}" for k, v in alt.items()))
    print(f"""   换月总次数：按持仓量 {n_roll_oi} 次，按成交量 {n_roll_vol} 次
       （成交量每天波动比持仓量大，按成交量选主力更容易提前或频繁切换；换月越多，换月成本越高）
    💡 如果换一种合理的数据处理方式，结论方向就变了，说明结论建立在数据细节之上，而不是策略逻辑之上。
    """)
    return {"cost2": res[2], "breakeven": breakeven, "data_alt": alt}


# ==================== 第六部分：稳健性报告 ====================


def report(port: pd.Series, p: dict, seg: dict, xs: dict, cd: dict, data: dict):
    print("\n" + "=" * 60)
    print("第六部分：稳健性报告清单")
    print("=" * 60)

    base = sharpe(port)
    same_dir = all(np.sign(v) == np.sign(base) for v in cd["data_alt"].values())
    items = [
        ("参数扰动", f"±20% 扰动下 Sharpe>0 占比 {p['perturb_pos']:.0%}", p["perturb_pos"] >= 0.8),
        ("分年度", f"最好一年占总对数收益 {seg['top_share']:.0%}" if np.isfinite(seg["top_share"]) else "总收益为负",
         np.isfinite(seg["top_share"]) and seg["top_share"] < 0.5),
        ("样本外", f"Walk-Forward Sharpe {seg['wf_sharpe']:+.2f}（全样本固定参数 {seg['is_sharpe']:+.2f}）", seg["wf_sharpe"] > 0),
        ("跨品种", f"Sharpe>0 的品种占比 {xs['xs_pos']:.0%}", xs["xs_pos"] > 0.5),
        ("成本", f"成本 ×2 后 Sharpe {cd['cost2']:+.2f}", cd["cost2"] > 0),
        ("数据", "换主力规则 / 复权方式后 Sharpe 方向" + ("不变" if same_dir else "改变"), same_dir),
    ]
    print(f"    被检验策略全样本 Sharpe {base:+.2f}\n")
    for name, desc, ok in items:
        print(f"    {'✅' if ok else '⚠️'} {name:<6} {desc}")
    n_ok = sum(ok for *_, ok in items)
    print(f"""
    通过 {n_ok}/{len(items)} 项（通过标准为经验参考，见 CONCEPT.md 8.6 节）。
    💡 稳健性检验不是为了"证明策略好"，而是为了找出它在哪些维度上脆弱。
       ⚠️ 的项目不一定意味着放弃策略，但必须在研究报告中写清楚。
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(p["perturb"]["组合"], bins=30, color="steelblue", alpha=0.8)
    axes[0].axvline(base, color="red", ls="--", label=f"基准参数 {base:.2f}")
    axes[0].set_title("参数扰动 ±20% 下的组合 Sharpe 分布")
    axes[0].legend()
    roll_sr = port.rolling(252).apply(sharpe, raw=False)
    axes[1].plot(roll_sr.index, roll_sr, lw=0.8)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("组合滚动 1 年 Sharpe")
    path = save_figure(fig, "09_robustness.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    data = load_universe(UNIVERSE)
    port = base_strategy(data)
    p = perturbation(data)
    seg = segmentation(data, port)
    xs = cross_section(data)
    cd = cost_and_data(data)
    report(port, p, seg, xs, cd, data)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把参数扰动范围从 ±20% 扩大到 ±50%，Sharpe>0 的占比如何变化？在哪个范围开始明显恶化？
    2. 用 Bootstrap（按月有放回抽样组合收益）估计组合 Sharpe 的 90% 置信区间
    3. 剔除贡献最大的一个品种或一个年份，重新计算组合 Sharpe —— 策略是否依赖单一事件？
    4. 换主力规则时加入"连续 3 天领先才切换"，再跑一遍第五部分
    """)


if __name__ == "__main__":
    main()
