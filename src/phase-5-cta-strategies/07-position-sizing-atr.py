"""
ATR 仓位与波动率目标
====================

学习目标：
    1. 理解为什么"每个品种 1 手"的组合风险完全失衡，仓位要按波动率分配
    2. 掌握 ATR 仓位公式（止损触发时亏损为净值固定比例）
    3. 实现波动率目标（Volatility Targeting）：EWMA 波动率估计、杠杆上限、次日生效
    4. 检验"实现波动率是否接近目标"，并认识低波动陷阱、估计滞后、调仓频率与取整误差

核心概念：
    - ATR 仓位：手数 = 净值 × 单笔风险比例 / (k × ATR × 合约乘数)
    - 波动率目标：名义敞口权重 w = σ_target / σ̂，手数 = 净值 × w / (价格 × 乘数)
    - EWMA 波动率：σ̂²_t = λ σ̂²_{t-1} + (1-λ) r²_t，年化 × √252
    - 风险贡献：组合方差中由每个品种"贡献"的份额

前置知识：
    - 05-turtle-trading.py（N 与 Unit）、06-futures-contracts-rollover.py（合约乘数、连续合约）
    - Phase 3 中的波动率聚集（GARCH）

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

from cta_utils import SPECS, UNIVERSE, backtest, cost_rate, load_universe, perf_stats, stats_table, wilder_atr
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TARGET_VOL = 0.10  # 单品种目标年化波动率
MAX_LEV = 3.0  # 单品种名义杠杆上限


def trend_signal(close: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """本课统一使用的趋势信号：双均线方向 ±1（只为演示仓位，不追求最优）"""
    ma_s = close.rolling(slow).mean()
    return np.sign(close.rolling(fast).mean() - ma_s).where(ma_s.notna(), 0.0)


def ewma_vol(ret: pd.Series, span: int = 40) -> pd.Series:
    """年化 EWMA 波动率：第 t 天的值只用 t 及之前的收益，可用于决定 t+1 的仓位"""
    return ret.ewm(span=span, adjust=False, min_periods=20).std() * np.sqrt(252)


# ==================== 第一部分：为什么不能"每个品种 1 手" ====================


def one_lot_each(data: dict):
    print("=" * 60)
    print("第一部分：为什么不能\"每个品种 1 手\"")
    print("=" * 60)

    print("""
    比喻：组队拔河时不应该"每人一根绳"，而要按每个人的力气分配位置 —— 波动率就是每个品种的"力气"。
    实验：所有品种用同一个趋势信号，分别按"每品种 1 手"与"波动率目标"持仓，比较每个品种的风险贡献。
    风险贡献_i = Cov(品种 i 的盈亏, 组合盈亏) / Var(组合盈亏)，所有品种合计 100%
    """)
    pnl_lot, pnl_vt = {}, {}
    for s, df in data.items():
        r = df["close"].pct_change().fillna(0)
        sig = trend_signal(df["close"]).shift(1).fillna(0)
        notional = df["raw_close"].shift(1) * SPECS[s]["mult"]  # 1 手的名义价值（真实价格）
        pnl_lot[s] = sig * r * notional
        w = (TARGET_VOL / ewma_vol(r)).clip(upper=MAX_LEV).shift(1)
        pnl_vt[s] = sig * w * r * 1e6  # 每个品种分配 100 万净值
    rows = []
    for name, pnl in [("每品种1手", pd.DataFrame(pnl_lot)), ("波动率目标", pd.DataFrame(pnl_vt))]:
        pnl = pnl.dropna().iloc[120:]
        total = pnl.sum(axis=1)
        contrib = pnl.apply(lambda x: x.cov(total)) / total.var()
        rows.append(contrib.rename(name))
    t = pd.concat(rows, axis=1)
    t.insert(0, "品种", [SPECS[s]["name"] for s in t.index])
    lot_vol = pd.DataFrame(pnl_lot).std() * np.sqrt(252)
    t.insert(1, "1手年化盈亏波动(元)", lot_vol)
    print(t.to_string(formatters={"1手年化盈亏波动(元)": "{:,.0f}".format, "每品种1手": "{:.1%}".format,
                                  "波动率目标": "{:.1%}".format}))
    top = t["每品种1手"].idxmax()
    print(f"""
    📊 "每品种 1 手"时，{SPECS[top]['name']} 一个品种贡献了 {t.loc[top, '每品种1手']:.0%} 的组合风险；
       按波动率目标分配后，最大的单品种风险贡献为 {t['波动率目标'].max():.0%}（{len(t)} 个品种平均为 {1/len(t):.0%}）。
       波动率目标只让每个品种"单独的"波动相等，品种之间的相关性仍会让风险贡献不完全相等（10 课处理）。
    """)


# ==================== 第二部分：ATR 仓位 ====================


def atr_sizing(data: dict, equity: float = 10_000_000, risk: float = 0.005, k: float = 2.0):
    print("\n" + "=" * 60)
    print("第二部分：ATR 仓位 —— 止损触发时亏损为净值的固定比例")
    print("=" * 60)

    print(f"""
    手数 = 净值 × 单笔风险比例 / (k × ATR × 合约乘数)
    参数：净值 {equity/1e4:,.0f} 万，单笔风险 {risk:.1%}，止损距离 k = {k} 倍 ATR（海龟用 2N）
    含义：如果价格反向走了 k 个 ATR 触发止损，亏损约为净值的 {risk:.1%}
    """)
    rows = []
    for s, df in data.items():
        atr_raw = wilder_atr(df).iloc[-1] * df["raw_close"].iloc[-1] / df["close"].iloc[-1]  # 最新一天 scale = 1
        mult = SPECS[s]["mult"]
        exact = equity * risk / (k * atr_raw * mult)
        lots = int(exact)
        notional = lots * df["raw_close"].iloc[-1] * mult
        rows.append([s, SPECS[s]["name"], atr_raw, mult, exact, lots, notional / equity,
                     lots * k * atr_raw * mult / equity])
    t = pd.DataFrame(rows, columns=["代码", "品种", "ATR(20)", "乘数", "理论手数", "取整手数", "名义敞口/净值", "止损亏损/净值"])
    print(t.to_string(index=False, formatters={"ATR(20)": "{:.2f}".format, "理论手数": "{:.1f}".format,
                                               "名义敞口/净值": "{:.0%}".format, "止损亏损/净值": "{:.2%}".format}))
    print(f"""
    💡 "止损亏损/净值"一列在取整后都接近（但不超过）{risk:.1%}。取整误差最大的是理论手数很小的品种 ——
       资金越少、合约越大，按规则下单就越"粗糙"。
    ⚠️ ATR 的点数必须换算成真实合约价格：比例调整价的点数在历史上被缩放过（06 课）。
    """)


# ==================== 第三部分：波动率目标与实现波动率 ====================


def vol_target_backtest(df: pd.DataFrame, s: str, target: float = TARGET_VOL, cap: float = MAX_LEV,
                        span: int = 40) -> dict:
    r = df["close"].pct_change().fillna(0)
    sig = trend_signal(df["close"])
    vol = ewma_vol(r, span)
    w_vt = sig * (target / vol).clip(upper=cap)
    cost = cost_rate(s, df["raw_close"])
    return {
        "ret": r,
        "vol": vol,
        "fixed": backtest(sig, r, cost, roll=df["roll"]),
        "vt": backtest(w_vt.fillna(0), r, cost, roll=df["roll"]),
        "w": w_vt,
    }


def vol_targeting(data: dict):
    print("\n" + "=" * 60)
    print("第三部分：波动率目标 —— 让每个品种的年化波动都接近目标")
    print("=" * 60)

    print(f"""
    w_t = 信号_t × min(σ_target / σ̂_t, 杠杆上限)      σ_target = {TARGET_VOL:.0%}，上限 {MAX_LEV:.0f} 倍
    σ̂_t：EWMA（span = 40）年化波动率，只用 t 及之前的收益；w_t 在 t+1 才生效（backtest 内部 shift(1)）
    ❌ 错误：w = target / ret.std()  —— 全样本标准差用到了未来数据
    """)
    s0 = "RB" if "RB" in data else next(iter(data))
    res = vol_target_backtest(data[s0], s0)
    print(f"📊 {SPECS[s0]['name']}：同一个趋势信号，固定 1 倍 vs 波动率目标（均扣成本与换月成本）")
    print(stats_table({
        "买入持有(1x)": perf_stats(res["ret"]),
        "趋势-固定1x": perf_stats(res["fixed"]["ret"], res["fixed"]["turnover"], res["fixed"]["pos"]),
        "趋势-波动率目标": perf_stats(res["vt"]["ret"], res["vt"]["turnover"], res["vt"]["pos"]),
    }).to_string())

    rows = []
    for s, df in data.items():
        rs = vol_target_backtest(df, s)
        f, v = rs["fixed"]["ret"].iloc[120:], rs["vt"]["ret"].iloc[120:]
        roll_vol = v.rolling(252).std().dropna() * np.sqrt(252)
        rows.append([s, SPECS[s]["name"], f.std() * np.sqrt(252), v.std() * np.sqrt(252),
                     roll_vol.min(), roll_vol.max(), rs["w"].abs().iloc[120:].mean(), perf_stats(f)["Sharpe"],
                     perf_stats(v)["Sharpe"]])
    t = pd.DataFrame(rows, columns=["代码", "品种", "固定1x波动", "目标后波动", "滚动1年波动最低", "滚动1年波动最高",
                                    "平均杠杆", "固定1x Sharpe", "目标后 Sharpe"])
    print(f"\n📊 全部品种（目标 {TARGET_VOL:.0%}）：")
    print(t.to_string(index=False, formatters={**{c: "{:.1%}".format for c in t.columns[2:6]},
                                               "平均杠杆": "{:.2f}x".format, "固定1x Sharpe": "{:+.2f}".format,
                                               "目标后 Sharpe": "{:+.2f}".format}))
    dev = (t["目标后波动"] / TARGET_VOL - 1).abs()
    print(f"""
    📊 固定 1 倍时各品种年化波动从 {t['固定1x波动'].min():.1%} 到 {t['固定1x波动'].max():.1%}；
       波动率目标后从 {t['目标后波动'].min():.1%} 到 {t['目标后波动'].max():.1%}，与目标的平均偏离 {dev.mean():.0%}
    💡 实现波动率不会正好等于目标：EWMA 估计有误差和滞后；杠杆上限会让低波动品种达不到目标
       （平均杠杆接近上限的品种就是这种情况）。Sharpe 的变化因品种而异 —— 波动率聚集越明显，改善越可能出现。
    """)
    return res, s0


# ==================== 第四部分：低波动陷阱与估计滞后 ====================


def pitfalls(data: dict, res: dict, s0: str):
    print("\n" + "=" * 60)
    print("第四部分：陷阱 —— 低波动、估计滞后")
    print("=" * 60)

    low = min(data, key=lambda s: data[s]["close"].pct_change().std())
    df = data[low]
    r = df["close"].pct_change().fillna(0)
    raw_w = TARGET_VOL / ewma_vol(r)
    print(f"""
    ① 低波动陷阱：波动率越低，w = σ_target / σ̂ 越大
       样本中日波动最低的品种是 {SPECS[low]['name']}：不设上限时杠杆中位数 {raw_w.median():.1f} 倍、最高 {raw_w.max():.1f} 倍
       → 若随后波动突然放大，损失会远超目标。对策：杠杆上限（本课 {MAX_LEV:.0f} 倍）或波动率下限
    """)
    caps = {}
    for cap in [1.0, 3.0, 5.0, np.inf]:
        rs = vol_target_backtest(df, low, cap=cap)
        v = rs["vt"]["ret"].iloc[120:]
        caps["无上限" if cap == np.inf else f"上限 {cap:.0f}x"] = {
            "实现波动": f"{v.std() * np.sqrt(252):.2%}", "最差单日": f"{v.min():.2%}",
            "平均杠杆": f"{rs['w'].abs().iloc[120:].mean():.2f}x"}
    print(pd.DataFrame(caps).to_string())

    # 估计滞后：找出波动率目标版本最差的一天，看当天"预测"的波动是多少
    vt = res["vt"]
    worst = vt["ret"].idxmin()
    pred = res["vol"].shift(1).loc[worst]
    z = res["ret"].loc[worst] / (pred / np.sqrt(252))
    print(f"""
    ② 估计滞后：{SPECS[s0]['name']} 波动率目标版本最差的一天是 {worst.date()}，策略亏损 {vt['ret'].loc[worst]:.2%}
       前一天的预测年化波动 {pred:.1%}，当天价格变动 {res['ret'].loc[worst]:+.2%}，相当于 {abs(z):.1f} 个"预测日波动"
       → EWMA 对突发事件反应慢，"黑天鹅第一天"仍是按平静时期的仓位持有。接受这一点，配合止损和组合层风控
    """)


# ==================== 第五部分：调仓阈值与取整误差 ====================


def banded(target_lots: pd.Series, band: float = 0.1) -> pd.Series:
    """只有当目标手数与当前手数相差超过 band（相对）时才调仓；方向改变时总是调仓"""
    out, cur = [], 0.0
    for x in target_lots.fillna(0).to_numpy():
        if np.sign(x) != np.sign(cur) or (cur != 0 and abs(x - cur) / abs(cur) > band):
            cur = x
        out.append(cur)
    return pd.Series(out, index=target_lots.index)


def rebalance_and_rounding(data: dict):
    print("\n" + "=" * 60)
    print("第五部分：调仓阈值与取整误差")
    print("=" * 60)

    print("""
    波动率每天都在变 → 目标手数每天都变 → 每天都要小额调仓，交易成本上升
    对策：设置调仓阈值，例如目标手数变化超过 10% 才调。另外，手数必须是整数：资金越小，误差越大。
    """)
    rows = []
    for s in [x for x in ["RB", "CU", "IF"] if x in data]:
        df = data[s]
        r = df["close"].pct_change().fillna(0)
        w = (trend_signal(df["close"]) * (TARGET_VOL / ewma_vol(r)).clip(upper=MAX_LEV)).fillna(0)
        lot_value = df["raw_close"] * SPECS[s]["mult"]
        cost = cost_rate(s, df["raw_close"])
        for capital in [1e6, 1e7]:
            ideal = w * capital / lot_value
            for name, lots in [("每日调仓", ideal.round()), ("10%阈值", banded(ideal).round())]:
                real_w = lots * lot_value / capital
                bt = backtest(real_w, r, cost, roll=df["roll"])
                ideal_bt = backtest(w, r, cost, roll=df["roll"])
                te = (bt["ret"] - ideal_bt["ret"]).std() * np.sqrt(252)
                rows.append([SPECS[s]["name"], f"{capital/1e4:,.0f}万", name, bt["turnover"].sum() / (len(bt) / 252),
                             (lots == 0).mean(), te, perf_stats(bt["ret"])["Sharpe"]])
    t = pd.DataFrame(rows, columns=["品种", "资金", "调仓方式", "年化换手", "0手天数占比", "与理想仓位的跟踪误差", "Sharpe"])
    print(t.to_string(index=False, formatters={"年化换手": "{:.1f}x".format, "0手天数占比": "{:.0%}".format,
                                               "与理想仓位的跟踪误差": "{:.2%}".format, "Sharpe": "{:+.2f}".format}))
    print("""
    💡 读表要点：
       - "0 手天数"包含均线预热期（前 100 天信号为 0）；超出预热期的部分才是取整造成的
       - 资金越小、单手价值越大（如股指期货），"0 手天数"和跟踪误差越大 —— 先评估最小可行资金量
       - 调仓阈值的目的是降低换手（对比"年化换手"列），代价是实际仓位偏离目标、跟踪误差变大
    """)


def plot_vol(data: dict):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    for s, df in data.items():
        v = vol_target_backtest(df, s)["vt"]["ret"].iloc[120:]
        ax.plot(v.rolling(252).std() * np.sqrt(252), lw=0.8, label=SPECS[s]["name"])
    ax.axhline(TARGET_VOL, color="black", ls="--", label=f"目标 {TARGET_VOL:.0%}")
    ax.set_title("波动率目标后，各品种滚动 1 年实现波动率")
    ax.legend(ncol=4, fontsize=8)
    path = save_figure(fig, "07_vol_target.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    data = load_universe(UNIVERSE)
    one_lot_each(data)
    atr_sizing(data)
    res, s0 = vol_targeting(data)
    pitfalls(data, res, s0)
    rebalance_and_rounding(data)
    plot_vol(data)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 EWMA 的 span 改为 20 和 120，实现波动率与目标的偏离如何变化？换手呢？
    2. 用 λ = 0.94 的 RiskMetrics 公式手写 EWMA 方差递推，与 ewm(span=...) 的结果对比（span 与 λ 的换算：λ = 1 - 2/(span+1)）
    3. 用 ATR 仓位（第二部分）代替波动率目标做一遍第三部分的回测，两种方法的结果差多少？
    4. 计算本课品种池"每个品种至少能交易 1 手"所需的最小资金量
    """)


if __name__ == "__main__":
    main()
