"""
TWAP / VWAP / POV 执行算法
==========================

学习目标：
    1. 理解为什么要把大单（母单）拆成小单（子单）执行
    2. 实现三种基础执行算法：TWAP（按时间均匀）、VWAP（按成交量分布）、POV（固定参与率）
    3. 用历史日内成交量分布（U 型）生成 VWAP 计划，子单按 100 股取整
    4. 用实施缺口（Implementation Shortfall）和相对 VWAP 的滑点评估执行质量

核心概念：
    - 母单 / 子单（Parent / Child Order）：执行算法决定"何时、下多少"
    - 参与率（Participation Rate）：自己的成交量 / 市场成交量
    - 实施缺口 IS：以决策时刻的到达价（Arrival Price）为基准的全部执行成本，含未成交部分的机会成本
    - "打败 VWAP"的错觉：自己的大单可能把 VWAP 基准本身推高

前置知识：
    - 03 课（冲击成本、平方根法则）、05 课（日内成交量分布）
    - Phase 4 的交易成本建模

环境要求：
    - pip install numpy pandas matplotlib
    - 本课使用模拟的日内行情（U 型成交量分布 + 随机游走价格 + 冲击模型），不需要网络
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from orderbook import intraday_volume_profile
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

N_MIN = 240  # A 股连续竞价 240 分钟（上午 120 + 下午 120）
LOT = 100
ADV = 10_000_000  # 日均成交量（股）
S0 = 10.0
DAILY_VOL = 0.02
HALF_SPREAD_BP = 5.0  # 10 元股票 1 tick = 10 bp 价差，主动成交付半个价差
TEMP_COEF_BP = 20.0  # 临时冲击：每分钟 κ·√(参与率)
PERM_BP_PER_PCT_ADV = 2.0  # 永久冲击：每成交 1% ADV，中间价永久移动 2 bp


def minute_labels():
    am = pd.date_range("09:30", periods=120, freq="min")
    pm = pd.date_range("13:00", periods=120, freq="min")
    return [t.strftime("%H:%M") for t in am.append(pm)]


# ==================== 第一部分：为什么要拆单 ====================


def why_split():
    print("=" * 60)
    print("第一部分：为什么要拆单")
    print("=" * 60)
    print("""
    比喻：一卡车货物要过一座窄桥。一次全开过去会压垮桥（市场冲击），
          分 100 趟慢慢运又可能遇上暴雨（价格风险）。执行算法就是"调度器"。

              母单：今天买入 50 万股
                       │
              ┌────────▼────────┐
              │    执行算法      │  ← 决定 何时 / 多少 / 什么价格
              └─┬───┬───┬───┬──┘
                ▼   ▼   ▼   ▼
              子单：每笔几百到几千股，分布在全天

    ┌──────┬────────────┬────────────────────────┬────────────────────────┐
    │ 算法 │ 跟踪基准   │ 子单量                 │ 主要风险               │
    ├──────┼────────────┼────────────────────────┼────────────────────────┤
    │ TWAP │ 时间均价   │ X / N                  │ 忽视量能，节奏易被识别 │
    │ VWAP │ 市场 VWAP  │ X · 历史成交量占比 w_i │ 当日分布偏离历史       │
    │ POV  │ 参与率 ρ   │ ρ · 上一分钟市场成交量 │ 完成时间不确定         │
    └──────┴────────────┴────────────────────────┴────────────────────────┘
    """)


# ==================== 第二部分：模拟市场与历史成交量分布 ====================


def simulate_market_volumes(n_days: int, seed: int) -> np.ndarray:
    """(天数, 240) 的分钟成交量：U 型分布 × 日间波动 × 分钟噪声"""
    rng = np.random.default_rng(seed)
    profile = intraday_volume_profile(N_MIN)
    day_factor = rng.lognormal(0, 0.25, (n_days, 1))
    noise = rng.lognormal(0, 0.35, (n_days, N_MIN))
    return ADV * day_factor * profile * noise


def volume_profile(hist_volumes: np.ndarray) -> np.ndarray:
    """历史成交量矩阵 → 各分钟平均占比（先按天归一化，再取平均）"""
    frac = hist_volumes / hist_volumes.sum(axis=1, keepdims=True)
    return frac.mean(axis=0)


def lots_schedule(total_qty: int, weights: np.ndarray, lot: int = LOT) -> np.ndarray:
    """按权重切分母单，取整到 100 股；零头用"最大余数法"分配，保证总量不变"""
    raw = total_qty * weights / weights.sum() / lot
    lots = np.floor(raw).astype(int)
    remainder = total_qty // lot - lots.sum()
    lots[np.argsort(raw - lots)[::-1][:remainder]] += 1
    return lots * lot


def history_profile():
    print("\n" + "=" * 60)
    print("第二部分：历史日内成交量分布（U 型）")
    print("=" * 60)
    hist = simulate_market_volumes(20, seed=0)
    est = volume_profile(hist)
    true = intraday_volume_profile(N_MIN)
    labels = minute_labels()
    buckets = np.arange(N_MIN) // 30
    est_30 = pd.Series(est).groupby(buckets).sum()
    print("    用过去 20 天的分钟成交量估计日内分布（按 30 分钟汇总）：")
    for b, v in est_30.items():
        print(f"      {labels[b * 30]} {'█' * int(v * 100):<25} {v:.1%}")
    err = np.abs(est - true).sum() / 2
    print(f"""
    估计分布与真实分布的总变差距离：{err:.1%}（历史天数越多越准，但市场结构也会变）
    第一分钟占全天 {est[0]:.2%}，午盘前后最低的分钟只占 {est.min():.2%}
    📌 真实 VWAP 系统还会考虑：收盘集合竞价的成交量、特殊日子（指数调整日、月末）、个股事件
    """)
    return est


# ==================== 第三部分：三种算法的子单计划 ====================


def twap_schedule(total_qty):
    return lots_schedule(total_qty, np.ones(N_MIN))


def vwap_schedule(total_qty, profile):
    return lots_schedule(total_qty, profile)


def pov_child_qty(rate, last_mkt_volume, remaining, lot=LOT):
    qty = (rate * last_mkt_volume // lot * lot).astype(int)
    return np.minimum(qty, remaining)


def show_schedules(Q, profile):
    print("\n" + "=" * 60)
    print("第三部分：子单计划（母单 = 买入 {:,} 股，约 {:.0%} ADV）".format(Q, Q / ADV))
    print("=" * 60)
    twap, vwap = twap_schedule(Q), vwap_schedule(Q, profile)
    print(f"    ✅ 取整检验：TWAP 合计 {twap.sum():,}，VWAP 合计 {vwap.sum():,}，"
          f"全部是 100 的倍数：{bool((twap % LOT == 0).all() and (vwap % LOT == 0).all())}")
    labels = minute_labels()
    buckets = np.arange(N_MIN) // 30
    df = pd.DataFrame({"TWAP": pd.Series(twap).groupby(buckets).sum(), "VWAP": pd.Series(vwap).groupby(buckets).sum()})
    df.index = [labels[b * 30] for b in df.index]
    print("    按 30 分钟汇总：")
    print("    " + df.to_string().replace("\n", "\n    "))
    print(f"    TWAP 每分钟 {twap.min()}~{twap.max()} 股；VWAP 每分钟 {vwap.min()}~{vwap.max()} 股")
    print("    POV 没有事先的计划：每分钟按'上一分钟市场成交量 × ρ'实时决定（见第四部分）")
    return twap, vwap


# ==================== 第四部分：模拟执行与评估 ====================


def execute(day_vol: np.ndarray, z: np.ndarray, algo: str, Q: int, plan=None, rate=None):
    """
    在 n_days 个模拟交易日上执行同一个买单，返回各项指标（每个数组长度 = 天数）。
    价格：分钟随机游走 + 我们成交带来的永久冲击；我们的成交价 = 中间价 × (1 + 半价差 + 临时冲击)
    """
    n_days = day_vol.shape[0]
    sigma_min = DAILY_VOL / np.sqrt(N_MIN)
    mid = np.full(n_days, S0)
    remaining = np.full(n_days, Q)
    cash = np.zeros(n_days)
    filled = np.zeros(n_days)
    mkt_notional = np.zeros(n_days)
    mkt_volume = np.zeros(n_days)
    max_part = np.zeros(n_days)
    for i in range(N_MIN):
        mid = mid * (1 + sigma_min * z[:, i])
        v = day_vol[:, i]
        if algo == "POV":
            last = day_vol[:, i - 1] if i > 0 else v * 0  # 第一分钟还没有"上一分钟"
            q = pov_child_qty(rate, last, remaining)
        elif algo == "一次性":
            q = remaining if i == 0 else np.zeros(n_days, dtype=int)
        else:
            q = np.minimum(np.broadcast_to(plan[i], (n_days,)), remaining)
        part = q / v
        px = mid * (1 + (HALF_SPREAD_BP + TEMP_COEF_BP * np.sqrt(part)) / 1e4)
        cash += q * px
        filled += q
        remaining = remaining - q
        max_part = np.maximum(max_part, q / (v + q))
        mkt_notional += v * mid + q * px  # 市场 VWAP 包含我们自己的成交
        mkt_volume += v + q
        mid = mid * (1 + PERM_BP_PER_PCT_ADV * (q / ADV * 100) / 1e4)  # 永久冲击
    close = mid
    avg_px = np.where(filled > 0, cash / np.maximum(filled, 1), np.nan)
    vwap = mkt_notional / mkt_volume
    # 实施缺口（买单）：已成交部分相对到达价的成本 + 未成交部分的机会成本
    is_bp = ((cash - filled * S0) + remaining * (close - S0)) / (Q * S0) * 1e4
    return {
        "IS(bp)": is_bp,
        "相对VWAP(bp)": (avg_px - vwap) / vwap * 1e4,
        "完成率": filled / Q,
        "最大参与率": max_part,
    }


def market_vwap(day_vol, z, plan):
    """只计算市场成交（不含我们自己）的 VWAP：用来看我们的永久冲击把基准推高了多少"""
    sigma_min = DAILY_VOL / np.sqrt(N_MIN)
    mid = np.full(day_vol.shape[0], S0)
    notional = np.zeros(day_vol.shape[0])
    for i in range(N_MIN):
        mid = mid * (1 + sigma_min * z[:, i])
        notional += day_vol[:, i] * mid
        mid = mid * (1 + PERM_BP_PER_PCT_ADV * (plan[i] / ADV * 100) / 1e4)
    return notional / day_vol.sum(axis=1)


def run_backtest(Q, profile, twap, vwap, n_days=500):
    print("\n" + "=" * 60)
    print(f"第四部分：在 {n_days} 个模拟交易日上执行，比较执行质量")
    print("=" * 60)
    print(f"""
    市场模型：日波动 {DAILY_VOL:.0%}；主动成交付半价差 {HALF_SPREAD_BP} bp；
              每分钟临时冲击 = {TEMP_COEF_BP} bp × √(参与率)；永久冲击 = 每 1% ADV 推动 {PERM_BP_PER_PCT_ADV} bp
    所有算法在同样的 {n_days} 条价格 / 成交量路径上执行（共同随机数，比较更公平）
    """)
    day_vol = simulate_market_volumes(n_days, seed=123)
    z = np.random.default_rng(456).standard_normal((n_days, N_MIN))
    algos = {
        "TWAP": dict(algo="TWAP", plan=twap),
        "VWAP": dict(algo="VWAP", plan=vwap),
        "POV 5%": dict(algo="POV", rate=0.05),
        "POV 10%": dict(algo="POV", rate=0.10),
        "一次性（开盘全部买入）": dict(algo="一次性"),
    }
    results = {name: execute(day_vol, z, Q=Q, **kw) for name, kw in algos.items()}
    # 同样的成交量路径、但价格没有随机波动 → 只剩冲击成本（价差 + 临时冲击 + 永久冲击）
    z0 = np.zeros_like(z)
    impact_only = {name: execute(day_vol, z0, Q=Q, **kw) for name, kw in algos.items()}
    rows = []
    for name, r in results.items():
        rows.append([name, np.nanmean(r["IS(bp)"]), np.nanstd(r["IS(bp)"]), np.nanmean(impact_only[name]["IS(bp)"]),
                     np.nanmean(r["相对VWAP(bp)"]), np.nanstd(r["相对VWAP(bp)"]),
                     r["完成率"].mean(), (r["完成率"] < 1).mean(), np.median(r["最大参与率"])])
    cols = ["算法", "IS均值", "IS标准差", "冲击成本", "相对VWAP均值", "相对VWAP标准差", "完成率", "未完成天数", "最大参与率中位数"]
    table = pd.DataFrame(rows, columns=cols)
    fmt2 = "{:.2f}".format
    print(table.to_string(index=False, formatters={
        "IS均值": fmt2, "IS标准差": fmt2, "冲击成本": fmt2, "相对VWAP均值": "{:+.2f}".format, "相对VWAP标准差": fmt2,
        "完成率": "{:.1%}".format, "未完成天数": "{:.1%}".format, "最大参与率中位数": "{:.1%}".format,
    }))
    diff = results["TWAP"]["IS(bp)"] - results["VWAP"]["IS(bp)"]
    t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))
    imp = table.set_index("算法")["冲击成本"]
    track = table.loc[table["相对VWAP标准差"].idxmin(), "算法"]
    low_risk = table.loc[table["IS标准差"].idxmin(), "算法"]
    twap_part = np.median(results["TWAP"]["最大参与率"])
    print(f"""
    （单位都是 bp，正数 = 成本；"冲击成本"= 同样的成交量路径、去掉价格随机波动后的 IS，只反映算法本身造成的成本）

    💡 读表要点：
       1. IS 标准差远大于各算法均值的差异 —— 单日执行结果主要由价格漂移（运气）决定。
          配对比较同一天的 TWAP − VWAP：IS 差 {diff.mean():+.2f} bp，约 {t_stat:.1f} 个标准误
          {"→ 差异显著" if abs(t_stat) > 2 else "→ 500 天里仍不显著：两者冲击成本只差 " + f"{imp['TWAP'] - imp['VWAP']:.2f}" + " bp，被价格噪声淹没"}
       2. 冲击成本：TWAP {imp['TWAP']:.2f} vs VWAP {imp['VWAP']:.2f} bp。TWAP 在午盘前后的清淡时段照样下单，
          最大参与率中位数 {twap_part:.0%}；VWAP 跟随成交量，参与率更均匀
       3. IS 标准差最小的是「{low_risk}」：几乎不承担价格风险，但冲击成本 {imp[low_risk]:.2f} bp{"，所有算法中最高" if imp[low_risk] == imp.max() else ""}
          —— 07 课 Almgren-Chriss 正是在"冲击"与"风险"之间权衡
       4. POV 5% 平均完成率 {results['POV 5%']['完成率'].mean():.1%}：它的冲击成本低，部分原因是少买了，
          没买到的部分要按收盘价计机会成本（价格上涨时代价很大）
       5. 相对 VWAP 标准差最小的是「{track}」：成交时间分布和市场一致，"跟得最像"；
          其它算法的成交时间偏离市场，相对 VWAP 的结果也混入了价格漂移
    """)

    # VWAP 基准被自己"污染"了多少
    vwap_with = market_vwap(day_vol, z, plan=vwap)
    vwap_without = market_vwap(day_vol, z, plan=np.zeros(N_MIN, dtype=int))
    pollution = (vwap_with / vwap_without - 1) * 1e4
    print(f"    「打败 VWAP」的错觉：执行 VWAP 母单后，当日市场 VWAP 平均被我们自己推高 {pollution.mean():.2f} bp")
    print("    （永久冲击抬高了之后所有人的成交价，基准跟着变高，我们的相对成绩因此显得更好）")
    return results, day_vol


# ==================== 第五部分：执行质量评估方法 ====================


def evaluation_notes():
    print("\n" + "=" * 60)
    print("第五部分：执行质量评估（TCA）")
    print("=" * 60)

    def slippage_bp(avg_fill_px, benchmark_px, side):
        """side=+1 买入、−1 卖出；正数表示比基准差（成本）"""
        return side * (avg_fill_px - benchmark_px) / benchmark_px * 1e4

    print(f"    买入均价 10.032 vs 基准 10.000 → {slippage_bp(10.032, 10.0, 1):.1f} bp（买贵了）")
    print(f"    卖出均价  9.990 vs 基准 10.000 → {slippage_bp(9.990, 10.0, -1):.1f} bp（卖便宜了）")

    Q, p0 = 100_000, 10.00
    fills = [(30_000, 10.01), (40_000, 10.03), (20_000, 10.05)]
    close = 10.10
    filled = sum(q for q, _ in fills)
    exec_cost = sum(q * (p - p0) for q, p in fills)
    opp_cost = (Q - filled) * (close - p0)
    fee = sum(q * p for q, p in fills) * 1.5e-4  # 示例：万 1.5 佣金
    total = exec_cost + opp_cost + fee
    print(f"""
    实施缺口分解示例（买入 {Q:,} 股，到达价 {p0:.2f}，收盘 {close:.2f}）：
      成交 {filled:,} 股：执行成本 {exec_cost:,.0f} 元
      未成交 {Q - filled:,} 股：机会成本 {opp_cost:,.0f} 元（收盘价 − 到达价）
      显性费用（示例佣金万 1.5）：{fee:,.0f} 元
      合计 {total:,.0f} 元 = {total / (Q * p0) * 1e4:.1f} bp
    """)
    print("""
    ┌──────────────────┬────────────────────────────────────────────┐
    │ 基准             │ 适用与陷阱                                 │
    ├──────────────────┼────────────────────────────────────────────┤
    │ 到达价（IS）     │ 最全面：同时惩罚冲击和拖延，机构首选       │
    │ 市场 VWAP        │ 常用但可被"污染"：大单推高的价格也在基准里 │
    │ 收盘价           │ 适合跟踪指数的被动资金                     │
    └──────────────────┴────────────────────────────────────────────┘
    📌 A 股执行还要处理：涨跌停时无法成交、价格笼子、T+1（卖出只能用昨日持仓）、午间休市
    """)


# ==================== 可视化 ====================


def plot_all(profile, twap, vwap, results, day_vol):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    fig.subplots_adjust(wspace=0.4)
    x = np.arange(N_MIN)
    ax = axes[0]
    ax.bar(x, profile * 100, color="lightgray", label="历史成交量占比（%）")
    ax2 = ax.twinx()
    ax2.plot(x, twap, label="TWAP 子单", color="tab:blue")
    ax2.plot(x, vwap, label="VWAP 子单", color="tab:red")
    ax.set_title("U 型成交量分布与子单计划")
    ax.set_xlabel("交易分钟（0 = 09:30，120 = 13:00）")
    ax.set_ylabel("成交量占比（%）")
    ax2.set_ylabel("子单（股）")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    names = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, names, fontsize=8, loc="upper center")

    ax = axes[1]
    for name, r in results.items():
        ax.hist(r["IS(bp)"], bins=40, histtype="step", label=name, linewidth=1.5)
    ax.set_title("实施缺口分布（模拟）")
    ax.set_xlabel("IS（bp，正数 = 成本）")
    ax.legend(fontsize=8)

    ax = axes[2]
    for name, r in results.items():
        ax.scatter(np.nanstd(r["IS(bp)"]), np.nanmean(r["IS(bp)"]), s=60, label=name)
    ax.set_title("成本均值 vs 风险（标准差）")
    ax.set_xlabel("IS 标准差（bp）")
    ax.set_ylabel("IS 均值（bp）")
    ax.legend(fontsize=8)

    path = save_figure(fig, "06_twap_vwap_pov.png", subdir="phase-11")
    print(f"\n📈 图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    why_split()
    profile = history_profile()
    Q = 500_000
    twap, vwap = show_schedules(Q, profile)
    results, day_vol = run_backtest(Q, profile, twap, vwap)
    evaluation_notes()
    plot_all(profile, twap, vwap, results, day_vol)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把母单改为 2% 和 15% ADV，比较各算法 IS 的变化，验证冲击成本随规模的增长是否接近平方根
    2. 实现"自适应 VWAP"：每分钟根据当天已实现成交量调整剩余计划（当天成交量偏离历史时更稳健）
    3. 给 POV 增加"截止时间"约束：14:30 后若完成率不足，提高参与率追赶，比较 IS 与风险
    4. 加入 A 股涨停约束：价格触及涨停后买单无法成交，统计各算法的未完成比例
    """)


if __name__ == "__main__":
    main()
