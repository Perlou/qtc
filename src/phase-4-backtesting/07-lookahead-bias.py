"""
前视偏差与未来函数
==================

学习目标：
    1. 认识前视偏差（Look-ahead Bias）的 12 种常见来源
    2. 在真实数据上演示多种未来函数，量化它们带来的"虚假收益"
    3. 实现两个自动化检测工具：
       - 截断一致性测试：截断数据后，历史信号不应改变
       - 扰动测试：把 t 日及之后的数据换成噪声，t 日的持仓不应改变（专抓"信号未滞后"）
    4. 用随机化检验发现"成交价偷看未来"这类框架级 bug

核心概念：
    - 未来函数：在 t 时刻使用了 t 时刻还无法获得的信息
    - 信号 vs 持仓：信号 s_t 可以用 t 日收盘数据；持有于 t 日的仓位 w_t 只能用 t−1 日及之前的数据
    - 截断一致性：signal(data[:T])[:T] == signal(data)[:T]

前置知识：
    - 02-vectorized-backtest.py（shift 的含义）
    - Phase 1 06-adjusted-prices.py（前复权为什么是隐性未来函数）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（合成数据没有分红，前复权一项会自动退化为通过）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bt_core import performance_report, vectorized_backtest
from utils.data import get_index_daily, get_stock_daily_adj
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20150101", "20241231"
COST = 0.001


# ==================== 第一部分：未来函数清单 ====================


def lookahead_catalog():
    print("=" * 60)
    print("第一部分：前视偏差的常见来源")
    print("=" * 60)

    print("""
    📰 比喻：拿着明天的报纸去买今天的彩票。回测里你是"中奖专业户"，实盘里你没有明天的报纸。

    ┌────┬──────────────────────┬────────────────────────────────────┬──────────────────────┐
    │ #  │ 来源                 │ 示例                               │ 修复                 │
    ├────┼──────────────────────┼────────────────────────────────────┼──────────────────────┤
    │ 1  │ 信号未滞后           │ signal * ret                       │ signal.shift(1)      │
    │ 2  │ 同一根 K 线高低点    │ 当日 high 止盈、low 止损都"先止盈" │ 保守假设/分钟数据    │
    │ 3  │ 全样本统计量         │ 全样本均值/标准差做 Z-score        │ rolling / expanding  │
    │ 4  │ 全样本参数估计       │ 全样本回归的对冲比例 β             │ 滚动估计             │
    │ 5  │ 前复权价格           │ 前复权价 < 15 元就买入             │ 收益后复权/判断不复权│
    │ 6  │ 财报按报告期对齐     │ 1 月就用上年年报                   │ 按公告日对齐         │
    │ 7  │ 当前成分股           │ 用今天的沪深 300 回测十年          │ 历史时点成分股       │
    │ 8  │ bfill / 插值         │ 缺失值用后面的值填充               │ 只允许 ffill         │
    │ 9  │ rolling(center=True) │ 窗口中心对齐                       │ 默认右对齐           │
    │ 10 │ 重采样标签错位       │ 周线信号对齐回日线没 shift         │ shift 到下一周期     │
    │ 11 │ ML 数据泄露          │ 随机打乱划分训练/测试集            │ 时间序列切分（P9）   │
    │ 12 │ 修订后的数据         │ 宏观/财务数据事后修订              │ point-in-time 数据   │
    └────┴──────────────────────┴────────────────────────────────────┴──────────────────────┘

    为什么它最致命：① 能把毫无价值的策略变成"圣杯"；② 代码不报错；③ 结果越好，人越不愿意怀疑。
    """)


# ==================== 第二部分：未来函数的"虚假收益" ====================


def ma_signal(df: pd.DataFrame) -> pd.Series:
    """✅ 正确：t 日收盘价 vs 截至 t 日的 20 日均线"""
    c = df["close"]
    return (c > c.rolling(20).mean()).astype(float).where(c.rolling(20).count() == 20)


def zscore_full(df: pd.DataFrame) -> pd.Series:
    """❌ 全样本 Z-score：均值和标准差用了整个样本（包括未来）"""
    c = df["close"]
    return ((c - c.mean()) / c.std() < -0.5).astype(float)


def zscore_expanding(df: pd.DataFrame) -> pd.Series:
    """✅ expanding Z-score：只用截至 t 日的数据（至少 250 天）"""
    c = df["close"]
    z = (c - c.expanding(250).mean()) / c.expanding(250).std()
    return (z < -0.5).astype(float).where(z.notna())


def ma_center(df: pd.DataFrame) -> pd.Series:
    """❌ rolling(center=True) 均线向上 → 持有：窗口以 t 为中心，均线的斜率包含了未来 10 天的走势"""
    m = df["close"].rolling(20, center=True).mean()
    return (m > m.shift(1)).astype(float).where(m.shift(1).notna())


def ma_slope(df: pd.DataFrame) -> pd.Series:
    """✅ 右对齐的 20 日均线向上 → 持有"""
    m = df["close"].rolling(20).mean()
    return (m > m.shift(1)).astype(float).where(m.shift(1).notna())


def run_signal(df: pd.DataFrame, sig: pd.Series, lag: bool = True) -> pd.Series:
    """收盘成交；lag=False 表示"信号未滞后"的错误写法"""
    sig = sig.fillna(0.0)
    if lag:
        return vectorized_backtest(df["close"], sig, COST)["ret"]
    ret = df["close"].pct_change().fillna(0.0)
    turnover = sig.diff().abs().fillna(sig.abs())
    return sig * ret - turnover * COST


def intrabar_stop(df: pd.DataFrame, tp: float = 0.015, sl: float = 0.015) -> tuple[pd.Series, pd.Series, int]:
    """
    每天开盘买入，盘中触及 +tp 止盈、−sl 止损，否则收盘卖出（仅演示 K 线内部顺序问题：
    A 股股票有 T+1，日内平仓不可行，可以想象为股指期货）。
    日线无法知道 high 和 low 谁先出现：乐观假设"先止盈"，保守假设"先止损"。
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    hit_tp, hit_sl = h >= o * (1 + tp), l <= o * (1 - sl)
    base = c / o - 1
    optimistic = np.where(hit_tp, tp, np.where(hit_sl, -sl, base))
    conservative = np.where(hit_sl, -sl, np.where(hit_tp, tp, base))
    cost = 2 * COST
    return (pd.Series(optimistic, index=df.index) - cost, pd.Series(conservative, index=df.index) - cost,
            int((hit_tp & hit_sl).sum()))


def fake_returns(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"第二部分：未来函数的'虚假收益'（沪深 300 指数 {START[:4]}-{END[:4]}，单边成本 0.1%）")
    print("=" * 60)

    opt, cons, both = intrabar_stop(df)
    cases = [
        ("1 信号未滞后", run_signal(df, ma_signal(df), lag=False), run_signal(df, ma_signal(df))),
        ("3 全样本 Z-score", run_signal(df, zscore_full(df)), run_signal(df, zscore_expanding(df))),
        ("9 均线斜率 center=True", run_signal(df, ma_center(df)), run_signal(df, ma_slope(df))),
        ("2 同K线先止盈 vs 先止损", opt, cons),
    ]
    rows = []
    for name, bad, good in cases:
        b, g = performance_report(bad), performance_report(good)
        rows.append([name, f"{b['年化收益']:+.2%}", f"{b['Sharpe']:.2f}", f"{g['年化收益']:+.2%}", f"{g['Sharpe']:.2f}"])
    print(pd.DataFrame(rows, columns=["未来函数", "❌年化", "❌Sharpe", "✅年化", "✅Sharpe"]).to_string(index=False))
    print(f"""
    🔍 第 2 项：{len(df)} 个交易日中有 {both} 天同时触及止盈线和止损线 —— 这些天"先发生哪个"决定了结果，
       日线数据根本无法回答。保守做法是一律按止损处理，或使用分钟数据。
    🔍 第 3 项：全样本 Z-score 的"低位"是用 {END[:4]} 年才知道的均值定义的 —— 等于事后知道了整个十年的价格区间。
    🔍 第 9 项：center=True 的均线包含未来 10 天的价格，它的斜率"预知"了接下来的趋势。

    💡 共同特征：代码完全不报错，回测结果被系统性抬高。抬高的幅度因情况而异 ——
       "好得离谱"的（如第 1 项）容易被发现；只"好一点"的（如全样本统计量）更危险，因为看起来很合理。
       Sharpe > 2 的日线策略，第一反应应该是查 bug；但 Sharpe 正常也不代表没有未来函数 —— 要靠第三部分的自动化测试。
    """)
    return cases


# ==================== 第三部分：截断一致性测试与扰动测试 ====================


def first_mismatch(a: pd.Series, b: pd.Series):
    """两条序列在共同索引上第一个不一致的日期（NaN 与 NaN 视为一致）"""
    a, b = a.align(b, join="inner")
    same = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
    return None if same.all() else a.index[~same][0]


def truncation_test(func, df: pd.DataFrame, cutoffs) -> pd.Timestamp | None:
    """截断一致性测试：用截至 cut 的数据算出的信号，必须与全量数据算出的信号在 cut 之前完全一致"""
    full = func(df)
    for cut in cutoffs:
        bad = first_mismatch(full.loc[:cut], func(df.loc[:cut]))
        if bad is not None:
            return bad
    return None


def perturbation_test(pos_func, df: pd.DataFrame, cutoffs, seed: int = 0) -> pd.Timestamp | None:
    """
    扰动测试（检查持仓是否滞后）：把 cut 日及之后的行情替换为随机噪声，
    cut 日（含）之前的持仓必须不变 —— 因为持有于 cut 日的仓位只能由 cut−1 日收盘前的信息决定。
    """
    rng = np.random.default_rng(seed)
    full = pos_func(df)
    for cut in cutoffs:
        noisy = df.copy()
        mask = noisy.index >= cut
        for col in ["open", "high", "low", "close"]:
            noisy.loc[mask, col] = noisy.loc[mask, col].to_numpy() * rng.uniform(0.5, 1.5, mask.sum())
        bad = first_mismatch(full.loc[:cut], pos_func(noisy).loc[:cut])
        if bad is not None:
            return bad
    return None


def with_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """人为制造缺失值（固定种子），用于演示 bfill / 插值的泄露"""
    rng = np.random.default_rng(7)
    out = df.copy()
    idx = rng.choice(len(out), size=len(out) // 20, replace=False)
    out.iloc[idx, out.columns.get_loc("close")] = np.nan
    return out


def bfill_signal(df):
    """❌ 缺失值用后一天的价格填充"""
    c = df["close"].bfill()
    return (c > c.rolling(20).mean()).astype(float)


def ffill_signal(df):
    """✅ 缺失值只用前一天的价格填充"""
    c = df["close"].ffill()
    return (c > c.rolling(20).mean()).astype(float).where(c.rolling(20).count() == 20)


def interp_signal(df):
    """❌ 线性插值：缺失日的值由前后两端决定，用到了后一端"""
    c = df["close"].interpolate()
    return (c > c.rolling(20).mean()).astype(float)


PRICE_THRESHOLD = 20.0  # "股价低于 20 元就买入"这类绝对价格规则


def qfq_signal_factory(adj_factor: pd.Series):
    """
    ❌ 前复权价 < 20 元：前复权价 = 不复权价 × 因子 / 最新因子，"最新因子"随数据延长（新的分红）而改变，
    历史上某天的前复权价会被未来的分红改写。
    注意：如果信号只用价格的"比例"（收益率、均线比值），前复权与后复权只差一个常数倍，不会泄露；
    泄露发生在绝对价格判断上（价格阈值、整数手、涨跌停）。
    """
    def qfq_signal(df):
        f = adj_factor.reindex(df.index)
        return (df["close"] * f / f.iloc[-1] < PRICE_THRESHOLD).astype(float)
    return qfq_signal


def raw_price_signal(df):
    """✅ 不复权价 < 20 元：那一天真实看到的价格"""
    return (df["close"] < PRICE_THRESHOLD).astype(float)


def detection_tests(idx: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：自动化检测——截断一致性测试与扰动测试")
    print("=" * 60)

    print("""
    方法一：截断一致性测试（Truncation Test）
      如果信号只依赖过去，"截至 T 的数据算出的信号"与"全量数据算出的信号"在 T 之前必须完全一致。
          full    = signal(data)
          partial = signal(data[:T])
          assert full[:T] == partial

    方法二：扰动测试（Perturbation Test）—— 截断测试抓不到"信号未滞后"！
      signal_t 本身只用到 t 日数据，截断测试会通过；错误在于把 signal_t 当成 t 日的持仓。
      检查办法：把 t 日及之后的行情换成噪声，持有于 t 日的仓位 w_t 不应改变。
    """)

    shenhua = get_stock_daily_adj("601088", "20180101", END)
    sh = shenhua[["open", "high", "low", "close"]]
    factor = shenhua["adj_factor"]
    gapped = with_gaps(idx)

    # 截断点越密，测试越灵敏：bfill / 插值只有在截断点恰好落在缺失值上、且信号因此翻转时才会暴露，
    # 所以对含缺失值的数据每 2 天截断一次（截断点稀疏时这两项可能漏检，可以自己改成 [250::25] 试试）
    cut_idx = idx.index[250::25]
    cut_gap = gapped.index[250::2]
    cut_sh = sh.index[100::20]

    lag = lambda f: (lambda d: f(d).shift(1))  # noqa: E731  持仓 = 信号滞后一天
    tests = [
        ("MA20 信号", ma_signal, idx, cut_idx),
        ("全样本 Z-score", zscore_full, idx, cut_idx),
        ("expanding Z-score", zscore_expanding, idx, cut_idx),
        ("均线斜率 center=True", ma_center, idx, cut_idx),
        ("均线斜率 右对齐", ma_slope, idx, cut_idx),
        ("缺失值 bfill", bfill_signal, gapped, cut_gap),
        ("缺失值 interpolate", interp_signal, gapped, cut_gap),
        ("缺失值 ffill", ffill_signal, gapped, cut_gap),
        ("601088 前复权价 < 20 元", qfq_signal_factory(factor), sh, cut_sh),
        ("601088 不复权价 < 20 元", raw_price_signal, sh, cut_sh),
    ]
    rows = []
    for name, func, data, cuts in tests:
        bad = truncation_test(func, data, cuts)
        rows.append([name, "✅ 通过" if bad is None else "❌ 失败", "" if bad is None else bad.date()])
    print("📋 截断一致性测试（对信号函数）：")
    print(pd.DataFrame(rows, columns=["信号函数", "结果", "首个不一致日期"]).to_string(index=False))
    print("   （'首个不一致日期'是第一个暴露问题的截断窗口里最早的差异，不一定是全样本中最早的泄露点）")

    rows = []
    for name, pos_func in [("持仓 = 信号（未滞后）", ma_signal), ("持仓 = 信号.shift(1)", lag(ma_signal))]:
        t_bad = truncation_test(pos_func, idx, cut_idx)
        p_bad = perturbation_test(pos_func, idx, cut_idx)
        rows.append([name, "✅ 通过" if t_bad is None else "❌ 失败", "✅ 通过" if p_bad is None else f"❌ 失败（{p_bad.date()}）"])
    print("\n📋 对'持仓函数'同时跑两种测试：")
    print(pd.DataFrame(rows, columns=["持仓函数", "截断测试", "扰动测试"]).to_string(index=False))

    if factor.nunique() <= 1:
        print("\n   ⚠️ 当前数据没有复权事件（离线合成数据），前复权一项无法暴露问题")
    print("""
    💡 建议：把这两个测试写进 pytest，每个信号函数 / 持仓函数都跑一遍。
       它们能抓住全样本统计量、center=True、bfill、插值、前复权、全样本回归、信号未滞后等大部分问题。
    """)


# ==================== 第四部分：随机化检验——框架级 bug ====================


def random_signal_test(df: pd.DataFrame, n_sims: int = 200):
    print("\n" + "=" * 60)
    print("第四部分：随机化检验——成交价偷看未来")
    print("=" * 60)

    print("""
    思路：随机信号与价格无关，在正确的框架里扣成本后应该亏钱（收益 ≈ −成本）。
         如果随机信号也稳定赚钱，说明框架本身有 bug —— 例如撮合时用了"当天最有利的价格"。
    演示：❌ 有 bug 的撮合：买入按当天最低价成交，卖出按当天最高价成交
          ✅ 正确的撮合：t 日收盘出信号，t+1 日开盘价成交
    """)
    o, h, l = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    rng = np.random.default_rng(42)
    res = {"✅ 次日开盘成交": [], "❌ 买最低/卖最高": []}
    years = len(df) / 252
    for _ in range(n_sims):
        sig = rng.integers(0, 2, len(df))
        sig[-2:] = 0  # 结束前清仓
        change = np.diff(np.concatenate([[0], sig]))
        entries, exits = np.where(change == 1)[0] + 1, np.where(change == -1)[0] + 1  # t 日信号 → t+1 日成交
        entries, exits = entries[entries < len(df)], exits[exits < len(df)]
        n = min(len(entries), len(exits))
        entries, exits = entries[:n], exits[:n]
        good = o[exits] / o[entries] * (1 - COST) ** 2
        bad = h[exits] / l[entries] * (1 - COST) ** 2
        res["✅ 次日开盘成交"].append(np.prod(good) ** (1 / years) - 1)
        res["❌ 买最低/卖最高"].append(np.prod(bad) ** (1 / years) - 1)
    rows = [[k, f"{np.mean(v):+.2%}", f"{np.mean(np.array(v) > 0):.0%}"] for k, v in res.items()]
    print(f"📋 {n_sims} 组随机信号（每天抛硬币，种子 42）的年化收益：")
    print(pd.DataFrame(rows, columns=["撮合方式", "平均年化", "为正的比例"]).to_string(index=False))
    print("""
    💡 随机信号在 bug 框架里"稳赚" → 收益来自框架，而不是策略。
       每个新引擎上线前都应该跑这个测试，配合 01 课的"一直满仓 = 买入持有"检查。
    """)
    return res


# ==================== 第五部分：逐笔审计清单 ====================


def audit_checklist(cases):
    print("\n" + "=" * 60)
    print("第五部分：逐笔审计清单")
    print("=" * 60)

    print("""
    挑 5~10 笔交易，手工回答：
    ☐ 信号在哪一刻产生？（收盘后？盘中？）
    ☐ 用到了哪些数据？那些数据在那一刻是否已经公布？（财报看公告日、指数成分看当时名单）
    ☐ 成交价是否可实现？（涨停时能买到吗？用的是开盘价还是当天最低价？）
    ☐ 复权方式？价格判断用的是不复权价吗？
    ☐ 收益最大的几笔交易，是否集中在信号当天？（信号当天就大赚 → 怀疑未对齐）

    "好得不真实"检查：
    ☐ 日线策略扣成本后 Sharpe > 2～3？
    ☐ 胜率 > 70% 且盈亏比 > 1？
    ☐ 回撤小得不像股票策略？
    """)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, bad, good in cases[:3]:
        ax.plot((1 + bad).cumprod(), label=f"❌ {name}")
    ax.plot((1 + cases[0][2]).cumprod(), label="✅ MA20 正确滞后", color="black", lw=2)
    ax.set_yscale("log")
    ax.set_title("未来函数制造的'圣杯'（对数坐标）")
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"📁 图片已保存：{save_figure(fig, '07_lookahead_bias.png', subdir='phase-4')}")


# ==================== 主函数 ====================


def main():
    lookahead_catalog()
    idx = get_index_daily("000300", START, END)[["open", "high", "low", "close"]]
    cases = fake_returns(idx)
    detection_tests(idx)
    random_signal_test(idx)
    audit_checklist(cases)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 写一个"周线信号"：每周五收盘价 > 周线 10 周均线时下周持有。先故意不 shift 对齐回日线，
       再用扰动测试找出问题，最后修正
    2. 用全样本 OLS 估计 601088 对沪深 300 的 β 并构造对冲组合，再改为 250 日滚动 β，
       用截断一致性测试证明前者是未来函数
    3. 把 truncation_test 和 perturbation_test 改写成 pytest 测试，对本课所有信号函数参数化运行
    4. 找一个你在网上看到的"高胜率"策略代码，逐行审计它是否有未来函数
    """)


if __name__ == "__main__":
    main()
