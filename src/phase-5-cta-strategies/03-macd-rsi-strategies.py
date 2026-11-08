"""
MACD 与 RSI 策略
================

学习目标：
    1. 正确计算 MACD（DIF / DEA / 柱）与 RSI（Wilder 平滑），了解常见实现差异
    2. 回测 MACD 趋势策略，并认识到 MACD、双均线本质上是同一类信号
    3. 回测 RSI 的两种用法：超买超卖反转 vs 趋势中的回调买入
    4. 用策略收益相关性理解"趋势信号 + 反转信号"组合的分散化效果

核心概念：
    - DIF = EMA12 - EMA26；DEA = EMA9(DIF)；MACD 柱 = 2 × (DIF - DEA)（国内软件惯例乘 2）
    - RSI = 100 - 100 / (1 + 平均涨幅 / 平均跌幅)，Wilder 平滑 alpha = 1/N（不是 2/(N+1)）
    - ewm(adjust=False)：递推算法，与行情软件一致；pandas 默认 adjust=True 在序列开头不同

前置知识：
    - 01-moving-average-crossover.py、02-bollinger-mean-reversion.py

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

from cta_utils import backtest, perf_stats, stats_table, trade_stats
from utils.data import get_index_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20100101", "20251231"
COST = 0.0005  # 单边成本示例：5 个基点


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, adjust: bool = False) -> pd.DataFrame:
    dif = close.ewm(span=fast, adjust=adjust).mean() - close.ewm(span=slow, adjust=adjust).mean()
    dea = dif.ewm(span=signal, adjust=adjust).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": 2 * (dif - dea)})


def rsi(close: pd.Series, n: int = 14, method: str = "wilder") -> pd.Series:
    """method: wilder（alpha=1/n，原始定义）/ ema（alpha=2/(n+1)）/ sma（简单平均）"""
    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    if method == "wilder":
        ag, al = gain.ewm(alpha=1 / n, adjust=False).mean(), loss.ewm(alpha=1 / n, adjust=False).mean()
    elif method == "ema":
        ag, al = gain.ewm(span=n, adjust=False).mean(), loss.ewm(span=n, adjust=False).mean()
    else:
        ag, al = gain.rolling(n).mean(), loss.rolling(n).mean()
    return 100 - 100 / (1 + ag / al)


# ==================== 第一部分：MACD 的计算细节 ====================


def macd_basics(close: pd.Series) -> pd.DataFrame:
    print("=" * 60)
    print("第一部分：MACD 的计算细节")
    print("=" * 60)

    print("""
    DIF  = EMA(12) - EMA(26)          ← 快慢 EMA 之差：趋势方向与强度
    DEA  = EMA(DIF, 9)                ← DIF 的再平滑：信号线
    柱   = 2 × (DIF - DEA)            ← 国内软件乘 2，海外软件（histogram）通常不乘
    """)

    m = macd(close)
    m_adj = macd(close, adjust=True)
    head_gap = (m["dif"] - m_adj["dif"]).abs().iloc[:30].mean()
    tail_gap = (m["dif"] - m_adj["dif"]).abs().iloc[-250:].mean()
    print(f"📊 adjust=False（递推）与 adjust=True（pandas 默认）的 DIF 平均差异：")
    print(f"   前 30 根 K 线：{head_gap:.3f} 点      最近 250 根：{tail_gap:.2e} 点")
    print("   → 差异只出现在序列开头。做研究统一用 adjust=False，并丢弃前 3×26 ≈ 80 根\"预热期\"数据")

    print("\n📊 最近 5 个交易日的 MACD（沪深 300）：")
    print(m.tail().round(2).to_string())
    return m


# ==================== 第二部分：MACD 策略 ====================


def macd_strategies(close: pd.Series, m: pd.DataFrame, ret: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("第二部分：MACD 趋势策略")
    print("=" * 60)

    warm = pd.Series(np.arange(len(close)) >= 80, index=close.index)  # 丢弃预热期
    targets = {
        "DIF上穿/下穿DEA": np.sign(m["dif"] - m["dea"]).where(warm, 0.0),
        "DIF 符号(=EMA12/26)": np.sign(m["dif"]).where(warm, 0.0),
        "SMA20/60(01课)": np.sign(close.rolling(20).mean() - close.rolling(60).mean()).fillna(0),
    }
    bts = {k: backtest(v, ret, COST) for k, v in targets.items()}
    rows = {"买入持有": perf_stats(ret)}
    rows.update({k: perf_stats(b["ret"], b["turnover"], b["pos"]) for k, b in bts.items()})
    print(f"多空版本，单边成本 {COST:.2%}：")
    print(stats_table(rows).to_string())

    pos = pd.DataFrame({k: b["pos"] for k, b in bts.items()})
    agree = (np.sign(pos["DIF 符号(=EMA12/26)"]) == np.sign(pos["SMA20/60(01课)"])).mean()
    print(f"""
    📌 "DIF 符号"策略与 01 课 SMA20/60 双均线在 {agree:.0%} 的交易日方向相同。
       数学上，均线差是过去收益的加权和 → 双均线、MACD、时间序列动量本质是同一类信号，
       只是对过去收益的加权方式不同。"组合了三个趋势指标" ≠ 获得了三份独立信息。

    📌 "DIF 穿 DEA"比"DIF 符号"反应更快（对 DIF 再求一次差），换手更高 —— 比较上表的年化换手。
    """)
    return bts


# ==================== 第三部分：RSI 的计算与分布 ====================


def rsi_basics(close: pd.Series) -> pd.Series:
    print("\n" + "=" * 60)
    print("第三部分：RSI 的计算与分布")
    print("=" * 60)

    print("""
    RS  = 平均涨幅 / 平均跌幅（N 日）        RSI = 100 - 100 / (1 + RS)
    Wilder 原始定义的"平均"是 alpha = 1/N 的递推平均（Wilder smoothing / RMA），
    很多代码误用 alpha = 2/(N+1) 的 EMA 或简单平均，算出来的数值会不同。
    """)

    variants = {m: rsi(close, 14, m) for m in ["wilder", "ema", "sma"]}
    r = variants["wilder"]
    print("📊 同一天三种实现的 RSI(14)（最近 5 天）：")
    print(pd.DataFrame(variants).tail().round(1).to_string())

    rr = r.dropna()
    above = rr > 70
    streak = above.groupby((~above).cumsum()).sum().max()
    print(f"""
    📊 沪深 300 RSI(14)（Wilder）分布：
       RSI > 70 的天数占比 {above.mean():.1%}，RSI < 30 的天数占比 {(rr < 30).mean():.1%}
       RSI 连续停留在 70 以上最长 {int(streak)} 天
    💡 RSI 被限制在 0–100，看起来像"归一化"，但强趋势中它可以长时间停留在 70 以上 ——
       "超买"之后还可以更超买。
    """)
    return r


# ==================== 第四部分：RSI 的两种用法 ====================


def rsi_reversal_target(r: pd.Series, lo: float = 30, hi: float = 70, exit_level: float = 50) -> pd.Series:
    """超买超卖反转：RSI < lo 做多、> hi 做空；回到 exit_level 平仓"""
    pos, cur = [], 0
    for x in r.to_numpy():
        if np.isnan(x):
            pos.append(0)
            continue
        if cur == 0:
            cur = 1 if x < lo else (-1 if x > hi else 0)
        elif (cur == 1 and x >= exit_level) or (cur == -1 and x <= exit_level):
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=r.index, dtype=float)


def trend_pullback_target(close: pd.Series, r: pd.Series, ma_n: int = 120, entry: float = 40,
                          exit_level: float = 60) -> pd.Series:
    """趋势中的回调买入（只做多）：收盘在 MA120 之上且 RSI < entry 时买入；RSI > exit_level 或跌破 MA120 离场"""
    up = (close > close.rolling(ma_n).mean()).to_numpy()
    pos, cur = [], 0
    for x, u in zip(r.to_numpy(), up):
        if np.isnan(x):
            pos.append(0)
            continue
        if cur == 0 and u and x < entry:
            cur = 1
        elif cur == 1 and (x > exit_level or not u):
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=r.index, dtype=float)


def rsi_strategies(close: pd.Series, r: pd.Series, ret: pd.Series) -> dict:
    print("\n" + "=" * 60)
    print("第四部分：RSI 的两种用法 —— 反转 vs 趋势中的回调")
    print("=" * 60)

    print("""
    A. 经典反转：RSI < 30 超卖做多，RSI > 70 超买做空，回到 50 平仓
    B. 趋势过滤 + 回调入场（只做多）：
       价格在 120 日均线之上（大趋势向上）且 RSI < 40（短期回调）时买入，
       RSI > 60 或跌破 120 日均线时离场
    """)
    targets = {
        "A.RSI反转-多空": rsi_reversal_target(r),
        "A.RSI反转-只做多": rsi_reversal_target(r).clip(lower=0),
        "B.趋势+RSI回调": trend_pullback_target(close, r),
    }
    bts = {k: backtest(v, ret, COST) for k, v in targets.items()}
    rows = {"买入持有": perf_stats(ret)}
    rows.update({k: perf_stats(b["ret"], b["turnover"], b["pos"]) for k, b in bts.items()})
    table = stats_table(rows)
    ts = {k: trade_stats(b["pos"], ret) for k, b in bts.items()}
    table.loc["交易笔数"] = ["-"] + [str(ts[k]["笔数"]) for k in bts]
    table.loc["胜率"] = ["-"] + [f"{ts[k]['胜率']:.0%}" if ts[k]["笔数"] else "-" for k in bts]
    print(table.to_string())
    best = max(bts, key=lambda k: rows[k]["Sharpe"])
    print(f"\n    在这段数据上 Sharpe 最高的 RSI 方案是「{best}」（{rows[best]['Sharpe']:.2f}，买入持有为 {rows['买入持有']['Sharpe']:.2f}）")
    print("""
    💡 读表要点：
       - 反转规则要逆势扛单，做空一侧在单边上涨市中尤其危险 —— 对比"多空"与"只做多"两列
       - B 方案把 RSI 当作"趋势中的入场择机"，而不是独立的方向判断，这是 RSI 在趋势框架中的常见用法
       - 这些方案的持仓占比都不高：评估时也要看"资金闲置"的机会成本
       - 单一标的、单一参数的结果不能直接推广，08、09 课会讲如何检验
    """)
    return bts


# ==================== 第五部分：信号相关性与组合 ====================


def signal_correlation(close: pd.Series, ret: pd.Series, m: pd.DataFrame, r: pd.Series, macd_bts: dict,
                       rsi_bts: dict):
    print("\n" + "=" * 60)
    print("第五部分：趋势信号与反转信号的相关性")
    print("=" * 60)

    rets = pd.DataFrame({
        "SMA20/60": macd_bts["SMA20/60(01课)"]["ret"],
        "MACD(DIF穿DEA)": macd_bts["DIF上穿/下穿DEA"]["ret"],
        "RSI反转": rsi_bts["A.RSI反转-多空"]["ret"],
        "趋势+RSI回调": rsi_bts["B.趋势+RSI回调"]["ret"],
    }).iloc[250:]
    print("📊 策略日收益相关系数：")
    print(rets.corr().round(2).to_string())

    combo = (rets["SMA20/60"] + rets["RSI反转"]) / 2
    print("\n📊 等权组合 SMA20/60（趋势）+ RSI 反转（回归）：")
    print(stats_table({
        "SMA20/60": perf_stats(rets["SMA20/60"]),
        "RSI反转": perf_stats(rets["RSI反转"]),
        "50/50 组合": perf_stats(combo),
    }).to_string())
    c = rets["SMA20/60"].corr(rets["RSI反转"])
    print(f"""
    📌 趋势策略与反转策略的收益相关系数为 {c:+.2f}。
       相关性越低（甚至为负），组合的波动越小 —— 但组合 Sharpe 能否超过单个策略，
       还取决于每个策略本身是否赚钱。两个负 Sharpe 的策略组合在一起，仍然不赚钱。
    """)

    setup_chinese_font()
    recent = close.index[-500:]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(recent, close.loc[recent], color="black", lw=0.8)
    axes[0].set_title("沪深300（最近 500 个交易日）")
    axes[1].plot(recent, m["dif"].loc[recent], lw=0.8, label="DIF")
    axes[1].plot(recent, m["dea"].loc[recent], lw=0.8, label="DEA")
    axes[1].bar(recent, m["hist"].loc[recent], color=np.where(m["hist"].loc[recent] > 0, "red", "green"),
                width=1.0, alpha=0.5, label="MACD 柱(×2)")
    axes[1].legend()
    axes[1].set_title("MACD")
    axes[2].plot(recent, r.loc[recent], lw=0.8)
    for lvl in (30, 50, 70):
        axes[2].axhline(lvl, color="gray", ls="--", lw=0.8)
    axes[2].set_title("RSI(14)（Wilder）")
    fig.tight_layout()
    path = save_figure(fig, "03_macd_rsi.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    close = get_index_daily("000300", START, END)["close"]
    ret = close.pct_change().fillna(0)
    m = macd_basics(close)
    macd_bts = macd_strategies(close, m, ret)
    r = rsi_basics(close)
    rsi_bts = rsi_strategies(close, r, ret)
    signal_correlation(close, ret, m, r, macd_bts, rsi_bts)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 adjust=True 重新计算 MACD 并回测，结果差多少？差异集中在哪一段？
    2. 把 RSI 反转的阈值改为 20/80、出场改为 40/60，交易次数和最大回撤如何变化？
    3. 用"MACD 柱由负转正"作为入场信号，与"DIF 上穿 DEA"比较 —— 它们其实是同一个条件，为什么？
    4. 把 B 方案的趋势过滤从 MA120 改为 MA250，并允许在跌破 MA250 且 RSI > 60 时做空，结果如何？
    """)


if __name__ == "__main__":
    main()
