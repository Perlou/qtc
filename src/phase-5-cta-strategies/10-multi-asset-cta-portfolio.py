"""
多品种 CTA 组合
===============

学习目标：
    1. 理解分散化的数学：组合 Sharpe 取决于品种数量和平均相关性，相关性比数量更重要
    2. 搭建覆盖黑色、有色、贵金属、农产品、能化、金融的期货品种池（带本地缓存）
    3. 构建多周期趋势信号，并按"两层风险预算"计算权重：品种波动率缩放 → 板块限额 → 组合波动率目标
    4. 完成回测（次日执行、成本、换月成本），与买入持有、TSMOM 对比，检查实现波动率是否接近目标
    5. 把权重换算成手数，检查取整误差与保证金占用

核心概念：
    - SR_p ≈ SR × sqrt(N / (1 + (N-1)ρ))
    - 两层风险预算：w_i = s_i / σ̂_i（品种层）→ 板块限额 → k = σ_target / sqrt(w'Σw)（组合层）
    - 时间序列动量（TSMOM，Moskowitz-Ooi-Pedersen 2012）：过去 12 个月涨则做多、跌则做空，按波动率缩放
    - Crisis Alpha：危机中常伴随跨资产的持续单边行情，趋势策略可能表现较好

前置知识：
    - 07-position-sizing-atr.py、08-parameter-optimization.py、09-robustness-tests.py
    - Phase 8 将系统讲解协方差估计与风险平价（本课用简单的滚动样本协方差）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 期货数据缓存在 data/cache/futures_*.parquet（首次运行需联网下载，约 1–2 分钟）
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存时读缓存，否则使用合成数据）
    ⚠️ 合约乘数、保证金比例为示例值，以交易所和期货公司最新规定为准
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import SPECS, UNIVERSE, cost_rate, load_universe, perf_stats, stats_table
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TARGET_PORT_VOL = 0.10  # 组合目标年化波动
SECTOR_CAP = 0.35  # 单板块风险占比上限（按 |信号| 计）
MAX_GROSS = 4.0  # 总名义杠杆上限
COV_WINDOW = 120
CAPITAL = 10_000_000


def display_width(text: str) -> int:
    """终端显示宽度：中文等全角字符占 2 格，组合字符（如 σ̂ 的帽子）占 0 格"""
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_box(lines: list[str], width: int = 62):
    """打印右边框对齐的 ASCII 框（内容含动态数值时，按实际显示宽度补空格）"""
    print("\n    ┌" + "─" * width + "┐")
    for line in lines:
        print("    │ " + line + " " * max(width - 1 - display_width(line), 0) + "│")
    print("    └" + "─" * width + "┘\n")


# ==================== 第一部分：分散化的数学 ====================


def diversification_math():
    print("=" * 60)
    print("第一部分：分散化的数学 —— 相关性比数量更重要")
    print("=" * 60)

    print("""
    N 个品种，单品种 Sharpe 都是 SR，两两相关系数平均为 ρ，等权组合：
        σ_p = σ × sqrt(ρ + (1-ρ)/N)          SR_p ≈ SR × sqrt(N / (1 + (N-1)ρ))
    """)
    ns = [1, 5, 10, 20, 40]
    t = pd.DataFrame({f"ρ={rho}": [0.4 * np.sqrt(n / (1 + (n - 1) * rho)) for n in ns] for rho in [0.0, 0.2, 0.5]},
                     index=[f"N={n}" for n in ns])
    print("📊 单品种 SR = 0.4 时的组合 SR：")
    print(t.round(2).to_string())
    print(f"""
    → ρ = 0 时 10 个品种把 Sharpe 提升约 {np.sqrt(10):.1f} 倍；ρ = 0.5 时 N → ∞ 也只能提升到 SR/√0.5 ≈ {0.4/np.sqrt(0.5):.2f}
    → 20 个高度相关的品种 ≈ 少数几个独立品种。选品时板块均衡比堆数量更重要
    """)


# ==================== 第二部分：品种池与相关性 ====================


def universe_and_corr(data: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：品种池与板块相关性")
    print("=" * 60)

    rows = [[s, SPECS[s]["name"], SPECS[s]["sector"], SPECS[s]["exchange"], df.index[0].date(), df.index[-1].date(),
             int(df["roll"].sum())] for s, df in data.items()]
    print(pd.DataFrame(rows, columns=["代码", "品种", "板块", "交易所", "起始", "结束", "换月次数"]).to_string(index=False))
    missing = [s for s in UNIVERSE if s not in data]
    if missing:
        print(f"   ⏭️ 取不到、已跳过的品种：{missing}")

    rets = pd.DataFrame({s: df["close"].pct_change() for s, df in data.items()}).fillna(0)
    corr = rets.corr()
    sec = pd.Series({s: SPECS[s]["sector"] for s in rets.columns})
    same = sec.to_numpy()[:, None] == sec.to_numpy()[None, :]
    off = ~np.eye(len(sec), dtype=bool)
    intra = corr.to_numpy()[same & off].mean() if (same & off).any() else np.nan
    inter = corr.to_numpy()[~same].mean()
    print(f"""
    📊 日收益相关性：同板块品种之间平均 {intra:+.2f}，不同板块之间平均 {inter:+.2f}
       相关性最高的一对：{corr.where(off).stack().idxmax()}（{corr.where(off).stack().max():+.2f}）
    💡 选品原则：流动性优先、剔除上市时间过短或常封板的品种、板块尽量均衡。
       ⚠️ 本课品种池是"今天还在活跃交易"的品种，用它回测过去有幸存者偏差；严谨的做法是品种池随时间动态变化。
    """)
    return rets


# ==================== 第三部分：信号与两层风险预算 ====================


def trend_ensemble(close: pd.Series) -> pd.Series:
    """多周期趋势集成：三组双均线方向的平均，取值 -1, -1/3, 1/3, 1"""
    sigs = []
    for f, s in [(10, 40), (20, 80), (40, 160)]:
        ma_s = close.rolling(s).mean()
        sigs.append(np.sign(close.rolling(f).mean() - ma_s).where(ma_s.notna()))
    return pd.concat(sigs, axis=1).mean(axis=1, skipna=False).fillna(0)


def tsmom_signal(close: pd.Series, lookback: int = 252) -> pd.Series:
    """TSMOM：过去 12 个月（约 252 个交易日）收益的符号"""
    return np.sign(close / close.shift(lookback) - 1).fillna(0)


def cta_weights(signals: pd.DataFrame, rets: pd.DataFrame, target: float = TARGET_PORT_VOL,
                sector_cap: float | None = SECTOR_CAP) -> tuple[pd.DataFrame, pd.Series]:
    """
    返回 (第 t 天收盘后计算的目标权重, 组合层缩放系数 k)；权重 = 名义敞口 / 净值，回测时再 shift(1)
        1) 品种层：w_i = s_i / σ̂_i（EWMA 年化波动）→ 每个品种风险均衡
        2) 板块层：某板块 |s_i| 之和占比超过 sector_cap 时，按比例压缩该板块
        3) 组合层：k = target / sqrt(w' Σ w)，Σ 为过去 COV_WINDOW 天的样本协方差（只用 t 及之前的数据）
        4) 总杠杆上限 MAX_GROSS
    """
    vol = rets.ewm(span=40, adjust=False, min_periods=20).std() * np.sqrt(252)
    raw = (signals / vol).replace([np.inf, -np.inf], np.nan).fillna(0)
    if sector_cap is not None:
        sec = pd.Series({s: SPECS[s]["sector"] for s in signals.columns})
        risk = signals.abs()
        share = risk.T.groupby(sec).sum().T.div(risk.sum(axis=1).replace(0, np.nan), axis=0)
        factor = (sector_cap / share).clip(upper=1.0).fillna(1.0)
        raw = raw * factor[sec.to_numpy()].to_numpy()

    R, W = rets.to_numpy(), raw.to_numpy()
    out, ks = np.zeros_like(W), np.zeros(len(W))
    for i in range(COV_WINDOW, len(W)):
        cov = np.cov(R[i - COV_WINDOW + 1:i + 1].T) * 252
        pv = np.sqrt(max(W[i] @ cov @ W[i], 0.0))
        if pv <= 0:
            continue
        w = W[i] * target / pv
        gross = np.abs(w).sum()
        if gross > MAX_GROSS:
            w *= MAX_GROSS / gross
        out[i], ks[i] = w, target / pv
    return pd.DataFrame(out, index=rets.index, columns=rets.columns), pd.Series(ks, index=rets.index)


def run_portfolio(weights: pd.DataFrame, rets: pd.DataFrame, costs: pd.DataFrame, rolls: pd.DataFrame) -> dict:
    """次日执行：pos_t = w_{t-1}；成本 = Σ|Δpos|×成本 + 换月日 2|pos|×成本"""
    pos = weights.shift(1).fillna(0)
    trade = pos.diff().abs().fillna(pos.abs())
    roll_trade = 2 * pos.abs() * rolls.astype(float)
    cost = ((trade + roll_trade) * costs).sum(axis=1)
    gross_ret = (pos * rets).sum(axis=1)
    net = gross_ret - cost
    return {"ret": net, "pos": pos, "turnover": (trade + roll_trade).sum(axis=1), "cost": cost, "gross": gross_ret}


def build_and_backtest(data: dict, rets: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("第三部分：多周期信号 + 两层风险预算")
    print("=" * 60)

    print_box([
        "信号 s_i ∈ [-1, 1]：MA 10/40、20/80、40/160 三组方向的平均",
        "  × 品种波动率缩放 1/σ̂_i          → 每个品种风险均衡",
        f"  × 板块限额：单板块风险占比 ≤ {SECTOR_CAP:.0%}",
        f"  × 组合缩放 k = {TARGET_PORT_VOL:.0%} / sqrt(w'Σw)（{COV_WINDOW} 日协方差）",
        f"  → 总杠杆 ≤ {MAX_GROSS:.0f} 倍 → 次日执行 → 换算手数 → 检查保证金",
    ])
    close = pd.DataFrame({s: df["close"] for s, df in data.items()}).ffill()
    costs = pd.DataFrame({s: cost_rate(s, df["raw_close"]) for s, df in data.items()}).ffill().bfill()
    rolls = pd.DataFrame({s: df["roll"] for s, df in data.items()}).fillna(False).astype(bool)

    sig_cta = close.apply(trend_ensemble)
    sig_ts = close.apply(tsmom_signal)
    w_cta, k_cta = cta_weights(sig_cta, rets)
    w_ts, _ = cta_weights(sig_ts, rets)
    w_nocap, _ = cta_weights(sig_cta, rets, sector_cap=None)
    res = {
        "CTA(多周期趋势)": run_portfolio(w_cta, rets, costs, rolls),
        "TSMOM(12个月)": run_portfolio(w_ts, rets, costs, rolls),
        "CTA(无板块限额)": run_portfolio(w_nocap, rets, costs, rolls),
    }
    start = rets.index[max(COV_WINDOW, 252) + 1]  # 等所有信号和协方差都预热完成
    bench = rets.mean(axis=1)
    rows = {"等权买入持有(1x)": perf_stats(bench.loc[start:])}
    for k, v in res.items():
        rows[k] = perf_stats(v["ret"].loc[start:], v["turnover"].loc[start:])
    print(f"📊 回测区间 {start.date()} ~ {rets.index[-1].date()}（扣手续费、滑点、换月成本）：")
    print(stats_table(rows).to_string())
    diff_days = ((w_cta - w_nocap).abs().sum(axis=1) > 1e-9).loc[start:].mean()
    print(f"""
    💡 读表要点：
       - 年化换手按"成交名义金额 / 净值"计算，组合层的换手包括信号变化、波动率变化、协方差变化和换月
       - TSMOM 是学术版"最小可行 CTA"，可作为自己策略的基准
       - 板块限额在 {diff_days:.0%} 的交易日起了作用（权重与无限额版本不同）。它是风控约束，
         不保证提高收益，作用是防止组合在某些时期过度押注单一板块
    """)
    return {"res": res, "w": w_cta, "k": k_cta, "start": start, "bench": bench, "sig": sig_cta}


# ==================== 第四部分：检查点 —— 实现波动率 vs 目标 ====================


def vol_check(ctx: dict):
    print("\n" + "=" * 60)
    print("第四部分：检查点 —— 组合实现波动率是否接近目标")
    print("=" * 60)

    r = ctx["res"]["CTA(多周期趋势)"]["ret"].loc[ctx["start"]:]
    pos = ctx["res"]["CTA(多周期趋势)"]["pos"].loc[ctx["start"]:]
    realized = r.std() * np.sqrt(252)
    yearly_vol = r.groupby(r.index.year).std() * np.sqrt(252)
    roll = r.rolling(126).std() * np.sqrt(252)
    gross = pos.abs().sum(axis=1)
    capped = (gross >= MAX_GROSS - 1e-6).mean()
    print(f"""
    目标年化波动 {TARGET_PORT_VOL:.0%}，全样本实现 {realized:.1%}（偏差 {realized / TARGET_PORT_VOL - 1:+.0%}）
    分年度实现波动：{'  '.join(f'{y}: {v:.1%}' for y, v in yearly_vol.items())}
    滚动半年实现波动：最低 {roll.min():.1%}，最高 {roll.max():.1%}
    平均总杠杆 {gross.mean():.2f} 倍，触及杠杆上限 {MAX_GROSS:.0f} 倍的天数占比 {capped:.0%}
    """)
    print("""    偏差较大时的常见原因：
       1. 协方差窗口跟不上相关性变化（危机时相关性飙升）
       2. 杠杆上限或保证金约束频繁触发
       3. 手数取整误差（资金量太小）
       4. 估计误差：用过去 120 天预测未来，本身就有误差
    """)


# ==================== 第五部分：分年度与危机表现 ====================


def crisis_alpha(ctx: dict):
    print("\n" + "=" * 60)
    print("第五部分：分年度表现与\"危机 Alpha\"")
    print("=" * 60)

    r = ctx["res"]["CTA(多周期趋势)"]["ret"].loc[ctx["start"]:]
    try:
        hs = get_index_daily("000300", "20100101", "20251231")["close"].pct_change()
    except Exception as e:  # 指数数据不可用时只做分年度统计
        print(f"   ⏭️ 沪深 300 获取失败（{type(e).__name__}）")
        hs = pd.Series(dtype=float)
    hs = hs.reindex(r.index)
    g = pd.DataFrame({"CTA": r, "沪深300": hs})
    yearly = g.groupby(g.index.year).apply(lambda x: (1 + x.fillna(0)).prod() - 1)
    yearly.loc[:, "沪深300"] = yearly["沪深300"].where(g["沪深300"].groupby(g.index.year).count() > 0)
    print(yearly.to_string(formatters={"CTA": "{:+.1%}".format, "沪深300": "{:+.1%}".format}, na_rep="-"))

    m = g.dropna().resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if len(m) > 12:
        worst = m["沪深300"].nsmallest(max(len(m) // 10, 3))
        print(f"""
    📊 月度相关系数（CTA vs 沪深300）：{m['CTA'].corr(m['沪深300']):+.2f}
       沪深300 最差的 {len(worst)} 个月里，CTA 的平均月收益 {m.loc[worst.index, 'CTA'].mean():+.2%}（沪深300 平均 {worst.mean():+.2%}）
    💡 Crisis Alpha 是一种"倾向"而非保证：它依赖危机期间是否出现持续的单边行情。
       样本只有几年、危机月份很少，这里的统计只能作为参考。
    """)


# ==================== 第六部分：换算手数与保证金 ====================


def lots_and_margin(ctx: dict, data: dict):
    print("\n" + "=" * 60)
    print(f"第六部分：最后一天的目标仓位 → 手数与保证金（资金 {CAPITAL/1e4:,.0f} 万）")
    print("=" * 60)

    w = ctx["w"].iloc[-1]
    last = ctx["w"].index[-1]
    rows = []
    for s, wi in w.items():
        px = data[s]["raw_close"].iloc[-1]
        lot_val = px * SPECS[s]["mult"]
        exact = wi * CAPITAL / lot_val
        lots = int(np.round(exact))
        rows.append([s, SPECS[s]["name"], ctx["sig"][s].iloc[-1], wi, exact, lots, lots * lot_val / CAPITAL,
                     abs(lots) * lot_val * SPECS[s]["margin"]])
    t = pd.DataFrame(rows, columns=["代码", "品种", "信号", "目标权重", "理论手数", "取整手数", "取整后权重", "保证金(元)"])
    k_last = ctx["k"].iloc[-1]
    print(f"   数据日期 {last.date()}；缩放前（w_i = s_i/σ̂_i，相当于每个品种按 100% 年化波动配置）的组合预测波动 "
          f"{TARGET_PORT_VOL / k_last:.0%}，整体乘以 k = {k_last:.3f} 缩放到 {TARGET_PORT_VOL:.0%}"
          if k_last > 0 else f"   数据日期 {last.date()}")
    print(t.to_string(index=False, formatters={"信号": "{:+.2f}".format, "目标权重": "{:+.1%}".format,
                                               "理论手数": "{:+.1f}".format, "取整后权重": "{:+.1%}".format,
                                               "保证金(元)": "{:,.0f}".format}))
    margin = t["保证金(元)"].sum()
    err = (t["取整后权重"] - t["目标权重"]).abs().sum()
    top = t.loc[t["取整后权重"].abs().idxmax()]
    print(f"""
    📊 总名义敞口（取整后）{t['取整后权重'].abs().sum():.0%}，保证金占用 {margin:,.0f} 元 = 净值的 {margin / CAPITAL:.1%}
       取整造成的权重偏差合计 {err:.1%}
       名义敞口最大的是 {top['品种']}（{top['取整后权重']:+.0%}）：低波动品种需要更大的名义敞口才能贡献相同的风险，
       所以"总名义杠杆"不能直接当作风险指标
    💡 通常 CTA 组合的保证金占用控制在净值的 20%–40% 以内，其余资金作为缓冲（保证金比例为示例值）。
       手数用"未调整的真实合约价格"计算，不能用比例调整后的连续价。
    """)


def plot_portfolio(ctx: dict):
    setup_chinese_font()
    start = ctx["start"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for k, v in ctx["res"].items():
        axes[0].plot((1 + v["ret"].loc[start:]).cumprod(), label=k)
    axes[0].plot((1 + ctx["bench"].loc[start:]).cumprod(), color="gray", label="等权买入持有(1x)")
    axes[0].set_title("多品种 CTA 组合净值")
    axes[0].legend()
    r = ctx["res"]["CTA(多周期趋势)"]["ret"].loc[start:]
    axes[1].plot(r.rolling(126).std() * np.sqrt(252), label="滚动半年实现波动")
    axes[1].axhline(TARGET_PORT_VOL, color="red", ls="--", label=f"目标 {TARGET_PORT_VOL:.0%}")
    axes[1].set_title("组合实现波动率 vs 目标")
    axes[1].legend()
    fig.tight_layout()
    path = save_figure(fig, "10_cta_portfolio.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    diversification_math()
    data = load_universe(UNIVERSE)
    rets = universe_and_corr(data)
    ctx = build_and_backtest(data, rets)
    vol_check(ctx)
    crisis_alpha(ctx)
    lots_and_margin(ctx, data)
    plot_portfolio(ctx)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把协方差窗口从 120 天改为 60 天和 250 天，实现波动率与目标的偏差如何变化？
    2. 把等权的趋势信号换成"按板块风险平价"：先在板块内等风险，再在板块间等风险，比较 Sharpe 与回撤
    3. 加入 Carry 信号（06 课的展期收益）与趋势信号等权组合，观察两者的相关性和组合 Sharpe
    4. 把资金改为 100 万，重做第六部分：哪些品种无法交易？实现波动率会偏离目标多少？
    """)


if __name__ == "__main__":
    main()
