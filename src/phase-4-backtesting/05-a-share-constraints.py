"""
A 股约束：T+1、涨跌停、停牌、100 股
====================================

学习目标：
    1. 掌握 A 股四大成交约束，以及它们随板块、随时间变化的规则（如主板 ST 股 2026-07-06 起 5% → 10%）
    2. 用 Decimal 正确计算涨跌停价；除权除息日要用"除权参考价"作为前收盘
    3. 构造"能否成交"的掩码：开盘成交 / 收盘成交时，涨停买不进、跌停卖不出、停牌不能交易
    4. 用真实停牌案例（000002 万科A，2015-12 至 2016-07）看清向量化 ffill 的陷阱
    5. 理解 T+1 对盘中止损的影响，以及 100 股整数手对小资金的影响

核心概念：
    - 涨跌停价 = round_half_up(前收盘 × (1 ± 涨跌幅), 0.01)
    - 可成交掩码：can_buy = 有行情 & 成交价 < 涨停价；can_sell = 有行情 & 成交价 > 跌停价
    - T+1：sellable（可卖数量）每天开盘刷新，当日买入不增加 sellable
    - 复权与撮合分离：收益用 adj_close，约束判断用不复权价

前置知识：
    - 02-vectorized-backtest.py、04-transaction-costs-slippage.py
    - Phase 1 02-a-share-market-rules.py、06-adjusted-prices.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行：合成数据没有停牌，本课会人为删除一段行情来模拟停牌

⚠️ 交易规则会调整，本课以编写时（2026 年）的公开规则为准，实盘前请核对交易所最新规定。
"""

import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import add_reference_columns, board_rule, limit_prices, load_bars, round_lot_buy, trading_calendar
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

CASE_SYMBOL, CASE_START, CASE_END = "000002", "20150101", "20171231"  # 万科A：2015-12-18 起长期停牌
START, END = "20180101", "20241231"
T1_SYMBOLS = ["000002", "601088", "600519"]
LOT_SYMBOLS = ["600519", "601088", "000002", "600036", "000858"]
EPS = 1e-6  # 浮点比较容差（价格都已四舍五入到分）


# ==================== 第一部分：约束一览与板块规则 ====================


def rules_overview():
    print("=" * 60)
    print("第一部分：约束一览与板块规则")
    print("=" * 60)

    print("""
    ┌──────────┬──────────────────────────────────────┬────────────────────────────────┐
    │ 约束     │ 规则要点                             │ 对回测的影响                   │
    ├──────────┼──────────────────────────────────────┼────────────────────────────────┤
    │ T+1      │ 当日买入，下一交易日才能卖出；       │ 当日买入不能当日止损           │
    │          │ 卖出所得资金当日可再买入             │                                │
    │ 涨跌停   │ 主板 ±10%；主板 ST 2026-07-06 前 ±5% │ 涨停往往买不进，跌停往往卖不出 │
    │          │ 之后 ±10%；创业板/科创板 ±20%；      │                                │
    │          │ 北交所 ±30%；新股上市初期另有规定    │                                │
    │ 停牌     │ 停牌期间无法交易，复牌可能大幅跳空   │ 持仓被"锁住"，信号无法执行     │
    │ 整数手   │ 买入 100 股整数倍；科创板 ≥200 股、  │ 资金无法精确按权重分配         │
    │          │ 超出部分可 1 股递增；零股须一次卖出  │                                │
    └──────────┴──────────────────────────────────────┴────────────────────────────────┘
    """)

    cases = [("600519", False, None), ("300750", False, None), ("688981", False, None), ("830799", False, None),
             ("600000", True, "2026-07-03"), ("600000", True, "2026-07-06")]
    rows = []
    for sym, st, date in cases:
        r = board_rule(sym, is_st=st, date=date)
        rows.append([sym + (" (ST)" if st else ""), date or "现行", r.name, f"±{r.limit_pct:.0%}", r.min_lot, r.step])
    print(pd.DataFrame(rows, columns=["代码", "交易日", "板块", "涨跌幅", "最小买入", "递增单位"]).to_string(index=False))

    print("\n📋 买入股数取整（bt_core.round_lot_buy）：")
    rows = []
    for shares in [99, 150, 199, 200, 257, 1234.9]:
        rows.append([shares, round_lot_buy(shares, board_rule("600519")), round_lot_buy(shares, board_rule("688981"))])
    print(pd.DataFrame(rows, columns=["想买股数", "主板", "科创板"]).to_string(index=False))


# ==================== 第二部分：涨跌停价与除权参考价 ====================


def limit_prices_hand(pre_close: float, pct: float) -> tuple[float, float]:
    """手写版：Decimal 四舍五入到分"""
    pc = Decimal(str(pre_close))
    q = Decimal("0.01")
    up = (pc * (Decimal(1) + Decimal(str(pct)))).quantize(q, rounding=ROUND_HALF_UP)
    down = (pc * (Decimal(1) - Decimal(str(pct)))).quantize(q, rounding=ROUND_HALF_UP)
    return float(up), float(down)


def price_limit_demo(bars: dict):
    print("\n" + "=" * 60)
    print("第二部分：涨跌停价与除权参考价")
    print("=" * 60)

    x = 7.55 * 1.1
    print(f"""
    🔍 浮点陷阱：前收盘 7.55，涨停价应为 7.55 × 1.1 = 8.305 → 四舍五入 8.31
       浮点计算 7.55 * 1.1 显示为 {x!r}，实际存储的是 {x:.20f}（二进制无法精确表示）
       Python round(x, 2) = {round(x, 2)}，Decimal 四舍五入 = {limit_prices_hand(7.55, 0.10)[0]}
       用浮点 round 算出的涨停价差 1 分钱 —— 收盘 8.31 的真涨停会被误判为"没涨停、可以买"
    """)
    rows = []
    for pc, pct in [(7.55, 0.10), (5.05, 0.10), (8.88, 0.05), (85.55, 0.20), (1500.00, 0.10)]:
        hand = limit_prices_hand(pc, pct)
        rows.append([pc, f"{pct:.0%}", hand[0], hand[1], "✅" if hand == limit_prices(pc, pct) else "❌"])
    print(pd.DataFrame(rows, columns=["前收盘", "涨跌幅", "涨停价", "跌停价", "与bt_core一致"]).to_string(index=False))

    df = bars["601088"]
    f = df["adj_factor"]
    ex_days = f.index[f.ne(f.shift(1)) & f.shift(1).notna()]
    print(f"\n📋 601088 除息日：前收盘应使用'除权参考价'（= 昨收 × 昨日因子 / 今日因子）")
    rows = []
    for d in ex_days[:5]:
        i = df.index.get_loc(d)
        raw_prev = df["close"].iloc[i - 1]
        ref = df["pre_close"].iloc[i]
        rows.append([d.date(), raw_prev, ref, limit_prices(raw_prev, 0.10)[1], limit_prices(ref, 0.10)[1], df["low"].iloc[i]])
    print(pd.DataFrame(rows, columns=["除息日", "昨收(不复权)", "除权参考价", "跌停价(用昨收,错)",
                                      "跌停价(用参考价)", "当日最低"]).to_string(index=False))
    print("   用错前收盘，涨跌停价会整体偏移一个分红金额 → 除息日的约束判断全错")

    print("\n📊 各股票触及涨跌停的天数（收盘价 = 涨/跌停价；一字板 = 开高低收全等于涨/跌停价）：")
    rows = []
    for sym, d in bars.items():
        up, down = zip(*[limit_prices(p, board_rule(sym, date=t).limit_pct) if pd.notna(p) else (np.nan, np.nan)
                         for t, p in d["pre_close"].items()])
        up, down = np.array(up), np.array(down)
        c_up = np.abs(d["close"] - up) < EPS
        c_dn = np.abs(d["close"] - down) < EPS
        flat = (d["high"] - d["low"]).abs() < EPS
        rows.append([sym, f"{d.index[0].date()}~{d.index[-1].date()}", int(c_up.sum()), int(c_dn.sum()),
                     int((c_up & flat).sum()), int((c_dn & flat).sum())])
    print(pd.DataFrame(rows, columns=["代码", "区间", "收盘涨停", "收盘跌停", "一字涨停", "一字跌停"]).to_string(index=False))
    print("   大盘蓝筹很少触板；小盘股、ST 股、题材股触板频繁 —— 追涨/抄底类策略受影响最大")


# ==================== 第三部分：可成交掩码 ====================


def tradability_masks(df: pd.DataFrame, calendar: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    """在交易日历上构造"能否买/卖"的掩码（停牌日该股票没有行情记录）"""
    d = df.reindex(calendar)
    trading = d["volume"].fillna(0) > 0
    limits = [limit_prices(p, board_rule(symbol, date=t).limit_pct) if pd.notna(p) else (np.nan, np.nan)
              for t, p in d["pre_close"].items()]
    up = pd.Series([u for u, _ in limits], index=calendar)
    down = pd.Series([v for _, v in limits], index=calendar)
    return pd.DataFrame({
        "trading": trading, "up": up, "down": down,
        "buy_open": trading & (d["open"] < up - EPS), "sell_open": trading & (d["open"] > down + EPS),
        "buy_close": trading & (d["close"] < up - EPS), "sell_close": trading & (d["close"] > down + EPS),
    })


def suspension_spans(masks: pd.DataFrame) -> list[tuple]:
    """连续停牌区间 [(起, 止, 天数)]，按天数从长到短排序"""
    halted = ~masks["trading"]
    grp = (halted != halted.shift()).cumsum()
    spans = [(g.index[0], g.index[-1], len(g)) for _, g in halted[halted].groupby(grp[halted])]
    return sorted(spans, key=lambda x: -x[2])


def masks_demo(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"第三部分：可成交掩码（{CASE_SYMBOL}，{CASE_START[:4]}-{CASE_END[:4]}）")
    print("=" * 60)

    print("""
    保守规则：
      开盘成交 → 开盘价 ≥ 涨停价视为买不进，开盘价 ≤ 跌停价视为卖不出
      收盘成交 → 收盘价 ≥ 涨停价视为买不进（封板排队），收盘价 ≤ 跌停价视为卖不出
      停牌（无行情或成交量为 0）→ 不能买也不能卖
    """)
    m = tradability_masks(df, calendar, CASE_SYMBOL)
    n = len(m)
    rows = [["停牌", int((~m["trading"]).sum())],
            ["开盘不能买（停牌或开盘涨停）", int((~m["buy_open"]).sum())],
            ["开盘不能卖（停牌或开盘跌停）", int((~m["sell_open"]).sum())],
            ["收盘不能买（停牌或收盘涨停）", int((~m["buy_close"]).sum())],
            ["收盘不能卖（停牌或收盘跌停）", int((~m["sell_close"]).sum())]]
    print(pd.DataFrame(rows, columns=[f"情形（共 {n} 个交易日）", "天数"]).to_string(index=False))

    print("\n📋 最长的停牌区间：")
    for a, b, k in suspension_spans(m)[:3]:
        print(f"   {a.date()} ~ {b.date()}：{k} 个交易日")
    return m


# ==================== 第四部分：停牌案例——向量化 ffill 的陷阱 ====================


def simulate_constrained(adj_close: pd.Series, adj_open: pd.Series, signal: pd.Series, masks: pd.DataFrame,
                         cost: float = 0.001) -> tuple[pd.Series, list]:
    """
    带约束的逐日模拟（单资产，全仓/空仓，t 日收盘信号 → t+1 日开盘成交）：
      - 停牌、开盘涨停（买）、开盘跌停（卖）时订单不成交，下一个交易日继续尝试
      - 估值用后复权价；停牌期间价格不变，复牌日承担全部跳空
    """
    close = adj_close.ffill()
    open_ = adj_open.where(masks["trading"])
    pos, equity, desired = 0.0, 1.0, 0.0
    nav, log = [], []
    prev_close = np.nan
    for d in masks.index:
        # 开盘：先承担隔夜段，再尝试执行昨天收盘的目标仓位
        trading = bool(masks.at[d, "trading"])
        if trading and pos > 0 and pd.notna(prev_close):
            equity *= open_[d] / prev_close
        base = open_[d] if trading else prev_close
        if desired != pos:
            ok = masks.at[d, "buy_open"] if desired > pos else masks.at[d, "sell_open"]
            if ok:
                equity *= 1 - cost
                pos = desired
                log.append((d.date(), "买入" if pos > 0 else "卖出", "成交", round(open_[d], 2)))
            else:
                reason = "停牌" if not trading else ("开盘涨停" if desired > pos else "开盘跌停")
                log.append((d.date(), "买入" if desired > pos else "卖出", f"未成交：{reason}", np.nan))
        # 收盘：日内段 + 生成明天的目标仓位（停牌日没有新信号，沿用旧目标）
        if trading:
            if pos > 0 and pd.notna(base):
                equity *= close[d] / base
            if pd.notna(signal.get(d, np.nan)):
                desired = float(signal[d])
        prev_close = close[d]
        nav.append(equity)
    return pd.Series(nav, index=masks.index), log


def suspension_case(df: pd.DataFrame, calendar: pd.DatetimeIndex, masks: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：停牌案例——向量化 ffill 的陷阱")
    print("=" * 60)

    adj_close = df["adj_close"].reindex(calendar)
    adj_open = (df["open"] * df["adj_factor"]).reindex(calendar)

    # ❌ 向量化常见写法：先 ffill 价格再算信号与收益 —— 停牌期间价格是一条水平线
    c_ff, o_ff = adj_close.ffill(), adj_open.ffill()
    sig_naive = (c_ff > c_ff.rolling(20).mean()).astype(float)
    pos_new, pos_old = sig_naive.shift(1).fillna(0), sig_naive.shift(2).fillna(0)
    r_on = (o_ff / c_ff.shift(1) - 1).fillna(0)
    r_id = (c_ff / o_ff - 1).fillna(0)
    naive_ret = (1 + pos_old * r_on) * (1 - 0.001 * (pos_new - pos_old).abs()) * (1 + pos_new * r_id) - 1
    naive_nav = (1 + naive_ret).cumprod()

    # ✅ 只在真实交易日上算信号；订单受停牌、涨跌停约束
    traded = df["adj_close"]
    sig_real = (traded > traded.rolling(20).mean()).astype(float)
    real_nav, log = simulate_constrained(adj_close, adj_open, sig_real, masks)

    a, b, _ = suspension_spans(masks)[0]
    i0 = calendar.get_loc(a)
    win = calendar[max(i0 - 3, 0) : min(calendar.get_loc(b) + 6, len(calendar))]
    print(f"📋 最长停牌：{a.date()} ~ {b.date()}；停牌前后的行为对比（向量化 vs 带约束）：")
    show = pd.DataFrame({
        "有行情": masks.loc[win, "trading"], "ffill价": c_ff.loc[win].round(1),
        "向量化持仓": pos_new.loc[win], "向量化净值": naive_nav.loc[win].round(4),
        "带约束净值": real_nav.loc[win].round(4),
    })
    print(show.iloc[np.r_[0:6, -9:0]].to_string())

    log_df = pd.DataFrame(log, columns=["日期", "方向", "结果", "成交价(复权)"])
    log_df = log_df[(pd.to_datetime(log_df["日期"]) >= win[0]) & (pd.to_datetime(log_df["日期"]) <= win[-1])]
    print("\n📋 带约束模拟的订单日志（停牌窗口内）：")
    print(log_df.to_string(index=False) if len(log_df) else "   （窗口内无订单）")

    seg = slice(win[0], win[-1])
    print(f"""
    窗口内收益：向量化 {naive_nav.loc[seg].iloc[-1] / naive_nav.loc[seg].iloc[0] - 1:+.2%}，
               带约束 {real_nav.loc[seg].iloc[-1] / real_nav.loc[seg].iloc[0] - 1:+.2%}
    全区间收益：向量化 {naive_nav.iloc[-1] - 1:+.2%}，带约束 {real_nav.iloc[-1] - 1:+.2%}

    💡 向量化 ffill 的问题：停牌期间价格是水平线，均线慢慢"追上"价格，信号可能翻转，
       回测就在停牌期间"成功卖出"—— 实际上根本不可能。
       带约束模拟中：停牌期间订单无法成交；复牌后若一字跌停也卖不出，只能等到能成交的那天。
    """)
    return naive_nav, real_nav


# ==================== 第五部分：T+1 与盘中止损 ====================


def t1_stop_loss(bars: dict):
    print("\n" + "=" * 60)
    print("第五部分：T+1 与盘中止损")
    print("=" * 60)

    print("""
    日线策略"t 日收盘出信号 → t+1 日开盘买入"，最早 t+2 日才可能卖出 —— T+1 自动满足。
    但一旦加入"盘中止损"（用当日最低价判断是否触发），T+1 立刻成为问题：

         买入日 t+1                         t+2
    ┌───────────────────────┐        ┌──────────────────┐
    │ 09:30 开盘买入        │        │ 开盘才能卖出     │
    │ 盘中跌破止损线 −3%    │ ─────▶ │ 可能继续低开     │
    │ ❌ T+1：今天不能卖    │        │ 甚至跌停卖不出   │
    └───────────────────────┘        └──────────────────┘
    """)
    stop = 0.03
    rows = []
    for sym, d in bars.items():
        sig = (d["adj_close"] > d["adj_close"].rolling(20).mean()).astype(int)
        entry_days = d.index[(sig.diff() == 1).shift(1, fill_value=False)]  # 信号翻多的次日开盘买入
        for t in entry_days:
            i = d.index.get_loc(t)
            if i + 1 >= len(d):
                continue
            o, low = d["open"].iloc[i], d["low"].iloc[i]
            if low > o * (1 - stop):
                continue
            naive = -stop  # 当日按止损价卖出（违反 T+1）
            # T+1：次日开盘卖出；若次日开盘跌停则顺延
            j = i + 1
            while j < len(d) - 1:
                _, down = limit_prices(d["pre_close"].iloc[j], board_rule(sym).limit_pct)
                if d["open"].iloc[j] > down + EPS:
                    break
                j += 1
            # 用复权价计算收益（跨除息日也正确）
            real = d["adj_open"].iloc[j] / d["adj_open"].iloc[i] - 1
            rows.append([sym, t.date(), naive, real])
    ev = pd.DataFrame(rows, columns=["代码", "买入日", "当日止损(违规)", "T+1次日开盘卖出"])
    if ev.empty:
        print("   样本中没有触发盘中止损的买入日")
        return
    print(f"   {len(ev)} 次买入当天盘中跌破 −{stop:.0%}（{', '.join(bars)}，{START[:4]}-{END[:4]}）：")
    print(f"   平均收益：当日止损（违规）{ev['当日止损(违规)'].mean():+.2%}，"
          f"T+1 次日开盘卖出 {ev['T+1次日开盘卖出'].mean():+.2%}")
    print(f"   T+1 版本最差一笔：{ev['T+1次日开盘卖出'].min():+.2%}")
    print("   💡 区别不只在均值：违规版本把亏损'截断'在 −3%，T+1 版本暴露在隔夜跳空风险中")


# ==================== 第六部分：100 股整数手 ====================


def lot_size_demo(bars: dict):
    print("\n" + "=" * 60)
    print("第六部分：100 股整数手对小资金的影响")
    print("=" * 60)

    prices = {s: bars[s]["close"].iloc[-1] for s in LOT_SYMBOLS}
    print("   最新不复权收盘价：" + "，".join(f"{s} {p:.2f}" for s, p in prices.items()))
    print("   ⚠️ 整数手必须用不复权价计算（复权价可能是真实价格的几倍到上百倍）")
    rows = []
    for capital in [100_000, 1_000_000, 10_000_000]:
        target = capital / len(prices)
        shares = {s: round_lot_buy(target / p, board_rule(s)) for s, p in prices.items()}
        values = {s: shares[s] * prices[s] for s in prices}
        weights = {s: v / capital for s, v in values.items()}
        dev = max(abs(w - 1 / len(prices)) for w in weights.values())
        rows.append([f"{capital:,}", " / ".join(f"{weights[s]:.1%}" for s in prices), f"{dev:.1%}",
                     f"{1 - sum(values.values()) / capital:.1%}"])
    print(pd.DataFrame(rows, columns=["资金", "实际权重（目标各 20%）", "最大偏离", "闲置现金"]).to_string(index=False))
    print("   💡 回测里若允许'小数股'，小资金策略的结果会严重失真")


# ==================== 主函数 ====================


def load_case_data() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = load_bars(CASE_SYMBOL, CASE_START, CASE_END)
    calendar = trading_calendar(CASE_START, CASE_END)
    if is_offline():
        # 合成数据没有停牌：人为删除 120 个交易日模拟一次长期停牌
        print("⚠️ 离线模式：人为删除一段合成行情来模拟停牌，以下结果仅演示流程")
        drop = df.index[230:350]
        df = add_reference_columns(df.drop(drop))
    return df, calendar


def main():
    rules_overview()

    bars = {s: load_bars(s, START, END) for s in sorted(set(T1_SYMBOLS + LOT_SYMBOLS))}
    case_df, calendar = load_case_data()
    price_limit_demo({"601088": bars["601088"], "600519": bars["600519"], f"{CASE_SYMBOL}": case_df})
    masks = masks_demo(case_df, calendar)
    naive_nav, real_nav = suspension_case(case_df, calendar, masks)
    t1_stop_loss({s: bars[s] for s in T1_SYMBOLS})
    lot_size_demo(bars)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(naive_nav, label="向量化（ffill 价格）")
    ax.plot(real_nav, label="带约束（停牌/涨跌停不能成交）")
    ax.fill_between(masks.index, 0, 1, where=~masks["trading"].to_numpy(), color="gray", alpha=0.2,
                    transform=ax.get_xaxis_transform(), label="停牌")
    ax.set_title(f"{CASE_SYMBOL} MA20 择时：停牌处理对净值的影响")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"\n📁 图片已保存：{save_figure(fig, '05_suspension_case.png', subdir='phase-4')}")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 simulate_constrained 加上"收盘成交"模式（用 buy_close / sell_close 掩码），
       与开盘成交模式比较停牌案例的结果
    2. 实现 is_st 判断（可用股票名称是否含 "ST"），回测 2026-07-06 前后的 ST 股时自动切换 5% / 10%
    3. 第五部分改为止损 −5% 和 −2%，触发次数与两种处理方式的差异如何变化？
    4. 思考：一只股票复权因子在停牌期间发生变化（停牌期间除权），复牌首日的涨跌停价应如何计算？
    """)


if __name__ == "__main__":
    main()
