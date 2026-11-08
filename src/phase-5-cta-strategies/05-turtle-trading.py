"""
海龟交易法则完整复现
====================

学习目标：
    1. 掌握海龟交易系统的完整规则：N 值、头寸单位、两套入场系统、加仓、止损、离场、组合限额
    2. 在期货主力连续上实现逐 bar 的事件驱动回测：手数取整、按当前净值定仓、换月成本
    3. 实现系统 1 的"上次突破盈利则跳过"过滤规则（需要后台"虚拟交易"）
    4. 理解海龟的价值在于"完整交易系统"而非入场信号

核心概念：
    - N：20 日 ATR（Wilder 平滑），N_t = (19·N_{t-1} + TR_t) / 20
    - Unit：使 1N 的价格波动 = 账户净值的 1%，Unit = 1% × 净值 / (N × 每点价值)
    - 系统 1：20 日突破入场（有过滤规则），10 日反向突破离场
    - 系统 2：55 日突破入场（无过滤），20 日反向突破离场
    - 加仓：每有利移动 0.5N 加 1 Unit，最多 4 Unit；止损：2N，加仓后整体上移

前置知识：
    - 04-channel-breakout.py（Donchian 通道）
    - 06-futures-contracts-rollover.py 中的合约乘数概念（可先看本课第二部分的简要说明）

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

from cta_utils import SPECS, UNIVERSE, cost_rate, get_continuous, perf_stats, stats_table, wilder_atr
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

EQUITY0 = 1_000_000  # 初始资金 100 万元
RISK_PCT = 0.01  # 1N 波动 = 净值的 1%
MAX_UNITS = 4


# ==================== 第一部分：海龟规则总览 ====================


def turtle_rules():
    print("=" * 60)
    print("第一部分：海龟交易法则 —— 第一个广为人知的完整交易系统")
    print("=" * 60)

    print("""
    1983 年，Richard Dennis 与 William Eckhardt 打赌"交易能否被教会"，招募学员（"海龟"）
    传授一套完全机械化的趋势跟踪系统。规则后来由 Curtis Faith 在《海龟交易法则》中公开。

    ┌──────────────┬────────────────────────────────────────────────────────────┐
    │ 模块         │ 规则                                                       │
    ├──────────────┼────────────────────────────────────────────────────────────┤
    │ 市场         │ 流动性好的多品种期货                                       │
    │ 波动度量 N   │ 20 日 ATR（Wilder 平滑）：N = (19 × 前一日N + 当日TR) / 20 │
    │ 头寸单位     │ 1 Unit = 1% × 账户净值 / (N × 每点价值)                    │
    │ 入场 系统1   │ 突破 20 日高/低点；若上一次突破是盈利交易则跳过            │
    │              │ （被跳过时，若出现 55 日突破则入场，避免错过大趋势）        │
    │ 入场 系统2   │ 突破 55 日高/低点，不过滤                                  │
    │ 加仓         │ 每向有利方向移动 0.5N 加 1 Unit，单品种最多 4 Unit         │
    │ 止损         │ 反向 2N；加仓后所有 Unit 的止损上移到"最后成交价 - 2N"     │
    │ 离场         │ 系统1：反向突破 10 日极值；系统2：反向突破 20 日极值       │
    │ 组合限额     │ 单品种 ≤ 4；高度相关 ≤ 6；松散相关 ≤ 10；单方向合计 ≤ 12   │
    └──────────────┴────────────────────────────────────────────────────────────┘

    "突破"指盘中价格超过前 N 日（不含当天）的最高/最低价 —— 海龟用止损单入场，触价即成交。
    """)


# ==================== 第二部分：N 与 Unit ====================


def n_and_unit():
    print("\n" + "=" * 60)
    print("第二部分：N 与 Unit —— 把\"波动\"翻译成\"手数\"")
    print("=" * 60)

    print("""
    TR = max(H - L, |H - 前收|, |L - 前收|)        ← 比 H - L 多考虑了跳空
    N  = TR 的 20 日 Wilder 平滑（alpha = 1/20）
    Unit（手）= 净值 × 1% / (N × 合约乘数)          ← 合约乘数 = 价格变动 1 个单位的每手盈亏

    例：账户 100 万，螺纹钢 N = 50 元/吨，乘数 10 吨/手
        Unit = 1,000,000 × 1% / (50 × 10) = 20 手
        2N 止损 → 每个 Unit 最大预期亏损约为净值的 2%；加满 4 个 Unit 约 8%（跳空时会更多）
    """)
    print(f"    计算验证：int(1_000_000 * 0.01 / (50 * 10)) = {int(1_000_000 * 0.01 / (50 * 10))} 手")

    rows = []
    for s in UNIVERSE:
        df = get_continuous(s)
        n_adj = wilder_atr(df).iloc[-1]
        scale = df["raw_close"].iloc[-1] / df["close"].iloc[-1]  # 最新一天比例调整价 = 真实价，scale = 1
        n_raw = n_adj * scale
        mult = SPECS[s]["mult"]
        unit = int(EQUITY0 * RISK_PCT / (n_raw * mult))
        notional = unit * df["raw_close"].iloc[-1] * mult
        rows.append([s, SPECS[s]["name"], df["raw_close"].iloc[-1], n_raw, n_raw / df["raw_close"].iloc[-1], mult,
                     unit, notional / EQUITY0])
    t = pd.DataFrame(rows, columns=["代码", "品种", "最新价", "N", "N/价格", "乘数", "1 Unit(手)", "名义敞口/净值"])
    print(f"\n📊 各品种最新的 N 与 1 Unit（净值 {EQUITY0/1e4:.0f} 万；乘数为示例值，以交易所最新规定为准）：")
    print(t.to_string(index=False, formatters={"最新价": "{:.2f}".format, "N": "{:.2f}".format,
                                               "N/价格": "{:.2%}".format, "名义敞口/净值": "{:.0%}".format}))
    print("""
    💡 观察"名义敞口/净值"一列：波动小的品种（N/价格 小）分到的名义敞口大，波动大的分到的小，
       每个 Unit 承担的风险（1N 的盈亏）都约等于净值的 1%（取整到整数手后会有偏差）。
       这就是"按波动率定仓位"（07 课展开）。
       1 Unit 为 0 手的品种说明资金不足以按规则交易 —— 小资金做大合约时的"颗粒度"问题。
    """)
    big = t[t["名义敞口/净值"] > 3]
    if len(big):
        names = "、".join(f"{r['品种']}({r['名义敞口/净值']:.0%})" for _, r in big.iterrows())
        print(f"""    ⚠️ 低波动陷阱：{names} 的 1 Unit 名义敞口超过净值的 3 倍。
       按波动率定仓位会给低波动品种很高的杠杆 —— 一旦波动突然放大，损失会远超 1%。
       实践中要设置杠杆上限 / 波动率下限（07 课）。
    """)


# ==================== 第三部分：回测引擎 ====================


def turtle_backtest(df: pd.DataFrame, symbol: str, system: int = 1, use_filter: bool = True) -> dict:
    """
    单品种海龟回测（日线，逐 bar 事件驱动）

    价格空间：信号、止损、加仓价都在"比例调整连续价"上计算；
             盈亏和手数换算成真实价格点数：真实点数 = 调整价点数 × (真实价 / 调整价)
    同一根 K 线内的顺序（日线无法知道先后，采用保守假设）：先检查止损/离场，未离场再检查加仓
    简化：入场当天不检查止损、不加仓；按当前净值计算 Unit，不实现原版"亏损时缩减名义账户"规则
    """
    n_entry, n_exit = (20, 10) if system == 1 else (55, 20)
    o, h, lo, c = (df[k].to_numpy() for k in ["open", "high", "low", "close"])
    scale = (df["raw_close"] / df["close"]).to_numpy()
    N = wilder_atr(df).shift(1).to_numpy()  # 用昨天及之前的 TR
    up = df["high"].rolling(n_entry).max().shift(1).to_numpy()
    dn = df["low"].rolling(n_entry).min().shift(1).to_numpy()
    up55 = df["high"].rolling(55).max().shift(1).to_numpy()
    dn55 = df["low"].rolling(55).min().shift(1).to_numpy()
    ex_l = df["low"].rolling(n_exit).min().shift(1).to_numpy()
    ex_s = df["high"].rolling(n_exit).max().shift(1).to_numpy()
    roll = df["roll"].to_numpy()
    mult = SPECS[symbol]["mult"]
    cost_pt = cost_rate(symbol, df["raw_close"]).to_numpy() * df["raw_close"].to_numpy() * mult  # 每手单边成本(元)

    equity, d, lots, unit_lots, stop, next_add, n_fix, n_units = EQUITY0, 0, 0, 0, 0.0, 0.0, 0.0, 0
    sh_d, sh_entry, sh_stop, last_win = 0, 0.0, 0.0, False  # 系统 1 过滤用的"虚拟交易"
    eq_path, lots_path, traded = [], [], []
    trades, cur = [], None

    for i in range(len(df)):
        pnl, cost_today, trade_lots = 0.0, 0.0, 0
        ready = np.isfinite(N[i]) and np.isfinite(up[i]) and np.isfinite(up55[i])

        # ---------- 1) 虚拟交易（只对系统 1 有意义）：记录"上一次突破"是否盈利 ----------
        filter_win = last_win
        if system == 1 and ready:
            if sh_d != 0:
                lvl = max(sh_stop, ex_l[i]) if sh_d > 0 else min(sh_stop, ex_s[i])
                if (sh_d > 0 and lo[i] <= lvl) or (sh_d < 0 and h[i] >= lvl):
                    px = min(o[i], lvl) if sh_d > 0 else max(o[i], lvl)
                    last_win, sh_d = sh_d * (px - sh_entry) > 0, 0
            elif h[i] > up[i] and not lo[i] < dn[i]:
                sh_d, sh_entry = 1, max(o[i], up[i])
                sh_stop = sh_entry - 2 * N[i]
            elif lo[i] < dn[i] and not h[i] > up[i]:
                sh_d, sh_entry = -1, min(o[i], dn[i])
                sh_stop = sh_entry + 2 * N[i]

        # ---------- 2) 持仓：先止损/离场，再加仓 ----------
        prev_close = c[i - 1] if i > 0 else c[i]
        if lots and ready:
            lvl = max(stop, ex_l[i]) if d > 0 else min(stop, ex_s[i])
            if (d > 0 and lo[i] <= lvl) or (d < 0 and h[i] >= lvl):
                px = min(o[i], lvl) if d > 0 else max(o[i], lvl)
                pnl += d * lots * (px - prev_close) * scale[i] * mult
                cost_today += lots * cost_pt[i]
                trade_lots += lots
                cur.update(exit_date=df.index[i], exit_px=px, reason="止损" if lvl == stop else "离场")
                cur["pnl"] += pnl - cost_today
                trades.append(cur)
                cur, lots, d, n_units = None, 0, 0, 0
            else:
                pnl += d * lots * (c[i] - prev_close) * scale[i] * mult
                while n_units < MAX_UNITS and ((d > 0 and h[i] >= next_add) or (d < 0 and lo[i] <= next_add)):
                    px = max(o[i], next_add) if d > 0 else min(o[i], next_add)
                    pnl += d * unit_lots * (c[i] - px) * scale[i] * mult
                    lots += unit_lots
                    n_units += 1
                    cost_today += unit_lots * cost_pt[i]
                    trade_lots += unit_lots
                    stop = px - d * 2 * n_fix
                    next_add = px + d * 0.5 * n_fix
                    cur["units"] = n_units
                if roll[i]:  # 换月：平旧开新
                    cost_today += 2 * lots * cost_pt[i]
                    trade_lots += 2 * lots
                cur["pnl"] += pnl - cost_today

        # ---------- 3) 空仓：检查入场 ----------
        elif not lots and ready:
            go = 0
            if h[i] > up[i] and not lo[i] < dn[i]:
                go = 1
            elif lo[i] < dn[i] and not h[i] > up[i]:
                go = -1
            tag = f"{n_entry}日突破"
            if system == 1 and use_filter and go != 0 and filter_win:
                go = 0  # 上一次突破盈利 → 跳过；但若同时出现 55 日突破则入场（failsafe）
                if h[i] > up55[i]:
                    go, tag = 1, "55日突破(failsafe)"
                elif lo[i] < dn55[i]:
                    go, tag = -1, "55日突破(failsafe)"
            n_raw = N[i] * scale[i]
            unit_lots = int(equity * RISK_PCT / (n_raw * mult)) if go else 0
            if go and unit_lots >= 1:
                lvl = (up55[i] if "failsafe" in tag else up[i]) if go > 0 else (dn55[i] if "failsafe" in tag else dn[i])
                px = max(o[i], lvl) if go > 0 else min(o[i], lvl)
                d, lots, n_units, n_fix = go, unit_lots, 1, N[i]
                stop, next_add = px - d * 2 * n_fix, px + d * 0.5 * n_fix
                pnl += d * lots * (c[i] - px) * scale[i] * mult
                cost_today += lots * cost_pt[i]
                trade_lots += lots
                cur = dict(entry_date=df.index[i], dir=d, entry_px=px, n=n_fix, unit_lots=unit_lots, units=1,
                           tag=tag, equity0=equity, pnl=pnl - cost_today)

        equity += pnl - cost_today
        eq_path.append(equity)
        lots_path.append(d * lots)
        traded.append(trade_lots * df["raw_close"].iloc[i] * mult)

    eq = pd.Series(eq_path, index=df.index)
    expo = pd.Series(lots_path, index=df.index) * df["raw_close"] * mult / eq.shift(1).fillna(EQUITY0)
    turnover = pd.Series(traded, index=df.index) / eq.shift(1).fillna(EQUITY0)
    return {"equity": eq, "ret": eq.pct_change().fillna(0), "expo": expo, "turnover": turnover,
            "trades": pd.DataFrame(trades)}


def summarize(res: dict) -> dict:
    s = perf_stats(res["ret"], res["turnover"], res["expo"])
    t = res["trades"]
    s["交易笔数"] = len(t)
    s["胜率"] = (t["pnl"] > 0).mean() if len(t) else np.nan
    s["平均名义敞口"] = res["expo"].abs().mean()
    return s


def fmt_table(rows: dict) -> pd.DataFrame:
    table = stats_table({k: {kk: vv for kk, vv in v.items() if kk not in ("交易笔数", "胜率", "平均名义敞口")}
                         for k, v in rows.items()})
    table.loc["交易笔数"] = [f"{v.get('交易笔数', np.nan):.0f}" if "交易笔数" in v else "-" for v in rows.values()]
    table.loc["胜率"] = [f"{v['胜率']:.0%}" if "胜率" in v else "-" for v in rows.values()]
    table.loc["平均名义敞口"] = [f"{v['平均名义敞口']:.0%}" if "平均名义敞口" in v else "100%" for v in rows.values()]
    return table


def turtle_on_rb(symbol: str = "RB") -> dict:
    print("\n" + "=" * 60)
    print(f"第三部分：在{SPECS[symbol]['name']}（{symbol}）上完整回测")
    print("=" * 60)

    df = get_continuous(symbol)
    print(f"""
    数据：{df.index[0].date()} ~ {df.index[-1].date()}，比例调整主力连续（{int(df['roll'].sum())} 次换月）
    初始资金 {EQUITY0/1e4:.0f} 万；合约乘数 {SPECS[symbol]['mult']}（示例值）；
    成本 = 手续费万分之一 + 1 tick 滑点（单边），换月日对持仓"平旧开新"计两次成本
    """)
    res = {
        "系统1(含过滤)": turtle_backtest(df, symbol, 1, True),
        "系统1(无过滤)": turtle_backtest(df, symbol, 1, False),
        "系统2": turtle_backtest(df, symbol, 2),
    }
    bh = df["close"].pct_change().fillna(0)
    rows = {"买入持有(1x)": perf_stats(bh)}
    rows.update({k: summarize(v) for k, v in res.items()})
    print(fmt_table(rows).to_string())
    print("""
    📌 年化换手 = 成交名义金额 / 净值（按年）；平均名义敞口 = |持仓名义价值| / 净值 的平均值。
       海龟的敞口随 N 和加仓动态变化，所以和"1 倍做多"的买入持有比较时，Sharpe 比总收益更有意义。
    """)
    return {"df": df, "res": res, "bh": bh}


# ==================== 第四部分：加仓、止损与过滤规则的细节 ====================


def trade_details(ctx: dict, symbol: str = "RB"):
    print("\n" + "=" * 60)
    print("第四部分：加仓、止损与过滤规则的细节")
    print("=" * 60)

    t = ctx["res"]["系统1(含过滤)"]["trades"]
    t2 = ctx["res"]["系统2"]["trades"]
    if t.empty:
        print("   本样本没有产生交易")
        return
    t = t.assign(ret_on_equity=t["pnl"] / t["equity0"])
    print("📊 系统 1（含过滤）的交易分布：")
    print(f"   按加仓到的 Unit 数：{t['units'].value_counts().sort_index().to_dict()}")
    print(f"   按离场原因：{t['reason'].value_counts().to_dict()}")
    print(f"   按入场类型：{t['tag'].value_counts().to_dict()}")
    print(f"   单笔盈亏 / 入场时净值：最好 {t['ret_on_equity'].max():+.2%}，最差 {t['ret_on_equity'].min():+.2%}，"
          f"中位数 {t['ret_on_equity'].median():+.2%}")
    worst_expected = -RISK_PCT * 2 * MAX_UNITS
    print(f"   规则上满仓 4 Unit 时 2N 止损的预期最大亏损约 {worst_expected:.0%}（跳空、成本会让实际亏损更大）")

    best = t.loc[t["pnl"].idxmax()]
    print(f"""
    🔍 系统 1 最赚钱的一笔：
       {best['entry_date'].date()} {'做多' if best['dir'] > 0 else '做空'}（{best['tag']}），入场价 {best['entry_px']:.1f}（调整价），
       入场 N = {best['n']:.1f}，1 Unit = {best['unit_lots']} 手，最多加到 {best['units']} Unit，
       {best['exit_date'].date()} {best['reason']}，盈利 {best['pnl']/1e4:+.1f} 万元（入场时净值的 {best['ret_on_equity']:+.1%}）
    """)

    n_filter_skip = len(ctx["res"]["系统1(无过滤)"]["trades"]) - len(t)
    print(f"""
    📌 过滤规则："上一次 20 日突破（不论是否真实交易）若是盈利交易，就跳过本次"。
       实现上需要在后台跑一条"虚拟交易"：每次突破都假想入场、按 2N 止损 / 10 日离场判断输赢，
       即使真实账户跳过了这次突破，也要继续记录它的输赢，供下一次判断使用。
       本样本中系统 1 无过滤 {len(ctx['res']['系统1(无过滤)']['trades'])} 笔，含过滤 {len(t)} 笔（差 {n_filter_skip} 笔）；
       系统 2 共 {len(t2)} 笔交易。
    """)


# ==================== 第五部分：跨品种与启示 ====================


def cross_market(ctx: dict):
    print("\n" + "=" * 60)
    print("第五部分：跨品种表现与今天的启示")
    print("=" * 60)

    rows = []
    rets = {}
    for s in UNIVERSE:
        df = get_continuous(s)
        r1 = turtle_backtest(df, s, 1, True)
        r2 = turtle_backtest(df, s, 2)
        rets[s] = r2["ret"]
        rows.append([s, SPECS[s]["name"], SPECS[s]["sector"], perf_stats(r1["ret"])["Sharpe"],
                     perf_stats(r2["ret"])["Sharpe"], len(r2["trades"]), r2["expo"].abs().mean()])
    t = pd.DataFrame(rows, columns=["代码", "品种", "板块", "系统1 Sharpe", "系统2 Sharpe", "系统2 笔数", "系统2 平均敞口"])
    print(t.to_string(index=False, formatters={"系统1 Sharpe": "{:+.2f}".format, "系统2 Sharpe": "{:+.2f}".format,
                                               "系统2 平均敞口": "{:.0%}".format}))
    port = pd.DataFrame(rets).mean(axis=1)
    pos1 = (t["系统1 Sharpe"] > 0).sum()
    pos2 = (t["系统2 Sharpe"] > 0).sum()
    zero = t[t["系统2 笔数"] == 0]["品种"].tolist()
    if zero:
        print(f"\n    ⚠️ {'、'.join(zero)} 没有任何交易：100 万资金按 1% 风险算出的 1 Unit 不足 1 手（合约价值太大）。")
    print(f"""
    📊 系统 1 Sharpe > 0 的品种 {pos1}/{len(t)} 个，系统 2 为 {pos2}/{len(t)} 个
       系统 2 各品种等权平均（每个品种各自 100 万账户的收益取平均）：Sharpe {perf_stats(port)['Sharpe']:.2f}，
       最大回撤 {perf_stats(port)['最大回撤']:.1%}
    ⚠️ 这里没有实现组合层的 Unit 上限（高度相关 ≤ 6 等），也没有共享同一个账户，只是粗略的分散化示意。
       10 课会用波动率目标构建真正的多品种组合。

    💡 海龟的原版信号在今天的市场上表现远不如当年（趋势跟踪变得拥挤、突破被更快的资金提前交易等），
       但它的框架至今仍是 CTA 的标准范式：
         ① 用波动率而不是金额决定仓位   ② 用多品种分散单品种的不确定性
         ③ 用机械规则消除情绪           ④ 止损是系统的一部分，不是事后补丁
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for k, v in ctx["res"].items():
        axes[0].plot(v["equity"].index, v["equity"] / EQUITY0, label=k)
    axes[0].plot(ctx["bh"].index, (1 + ctx["bh"]).cumprod(), color="gray", label="买入持有(1x)")
    axes[0].set_title("螺纹钢：海龟系统净值")
    axes[0].legend()
    axes[1].plot(port.index, (1 + port).cumprod(), label="系统2 等权平均（各品种独立账户）")
    axes[1].set_title(f"系统 2 跨 {len(rets)} 个品种")
    axes[1].legend()
    fig.tight_layout()
    path = save_figure(fig, "05_turtle.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    turtle_rules()
    n_and_unit()
    ctx = turtle_on_rb("RB")
    trade_details(ctx, "RB")
    cross_market(ctx)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 实现原版"亏损时缩减名义账户"规则：净值每比初始资金低 10%，计算 Unit 用的名义账户缩减 20%
    2. 把止损改为 Faith 书中的"双重止损（Whipsaw）"变体：止损 1/2 N、亏损后可再次入场，比较交易笔数与回撤
    3. 把入场当天"不检查止损"改为"保守假设先止损"，结果差多少？这说明日线回测有什么局限？
    4. 在多个品种共享一个账户的前提下，实现"单方向合计 ≤ 12 Unit"的组合限额
    """)


if __name__ == "__main__":
    main()
