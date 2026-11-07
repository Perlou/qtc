"""
简单收益与对数收益
==================

学习目标：
    1. 理解简单收益（Simple Return）与对数收益（Log Return）的定义和换算
    2. 掌握"横向（组合）用简单收益、纵向（多期）用对数收益"的经验法则
    3. 用真实数据验证：两种收益都能还原同一条净值曲线
    4. 理解波动拖累（Volatility Drag）：为什么算术平均收益会"骗人"
    5. 认识收益计算中的常见陷阱：未复权、长表串味

核心概念：
    - 简单收益：R_t = P_t / P_{t-1} - 1，多期连乘，截面可加权求和
    - 对数收益：r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)，多期直接相加
    - 波动拖累：几何平均 ≈ 算术平均 - σ² / 2

前置知识：
    - Phase 1：06-adjusted-prices.py（等比复权）

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"


# ==================== 第一部分：直觉与定义 ====================


def intuition():
    """100 → 110 → 100：两种收益的直观对比"""
    print("=" * 60)
    print("第一部分：直觉与定义")
    print("=" * 60)

    prices = pd.Series([100.0, 110.0, 100.0], index=["第 1 天", "第 2 天", "第 3 天"])
    simple = prices.pct_change()
    log = np.log(prices / prices.shift(1))
    table = pd.DataFrame({"价格": prices, "简单收益": simple, "对数收益": log})
    print("\n📊 先涨后跌，回到原点：")
    print(table.to_string(float_format=lambda x: f"{x:+.4%}" if abs(x) < 1 else f"{x:.2f}"))
    print(f"\n   简单收益之和：{simple.sum():+.4%}（不为 0，却回到了原点）")
    print(f"   对数收益之和：{log.sum():+.4%}（正好为 0）")

    print("""
    📐 定义：
       简单收益  R_t = P_t / P_{t-1} - 1
       对数收益  r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)
       反向换算  R_t = exp(r_t) - 1

    🏔️ 比喻：简单收益像"打折"——先涨 10% 再跌 10% 回不到原价；
             对数收益像"海拔变化"——上山 100 米再下山 100 米正好回原地。

    ┌──────────────┬──────────────────────┬──────────────────────┐
    │ 性质         │ 简单收益 R           │ 对数收益 r           │
    ├──────────────┼──────────────────────┼──────────────────────┤
    │ 多期累计     │ 连乘 ∏(1+R) - 1      │ 直接求和 Σr          │
    │ 组合（截面） │ ✅ 加权求和          │ ❌ 加权求和不成立    │
    │ 下界         │ -100%                │ -∞                   │
    │ 典型用途     │ 组合收益、净值、报告 │ 统计建模、多期累加   │
    └──────────────┴──────────────────────┴──────────────────────┘
    """)

    # 收益越大，两者差距越大
    rs = np.array([-0.5, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.5, 1.0])
    gap = pd.DataFrame({"简单收益 R": rs, "对数收益 r": np.log1p(rs), "差值 R - r": rs - np.log1p(rs)})
    print("📊 两者差距随收益幅度变化：")
    print(gap.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n💡 |R| 很小时 R ≈ r（泰勒展开 ln(1+R) ≈ R - R²/2）；幅度越大差距越大，且 r 总是 ≤ R")


# ==================== 第二部分：真实数据计算 ====================


def real_data_returns() -> pd.DataFrame:
    """用复权收盘价计算两种收益，并用两种方式还原净值"""
    print("\n" + "=" * 60)
    print(f"第二部分：真实数据计算（{SYMBOL} {NAME}）")
    print("=" * 60)

    df = get_stock_daily_adj(SYMBOL, START, END)
    close = df["adj_close"]
    print(f"\n📊 数据：{close.index[0].date()} ~ {close.index[-1].date()}，共 {len(close)} 个交易日")

    ret = close.pct_change()  # 简单收益
    log_ret = np.log(close / close.shift(1))  # 对数收益（等价于 np.log1p(ret)）
    assert np.allclose(log_ret.dropna(), np.log1p(ret.dropna()))

    stats = pd.DataFrame(
        {
            "简单收益": [ret.mean(), ret.std(), ret.min(), ret.max()],
            "对数收益": [log_ret.mean(), log_ret.std(), log_ret.min(), log_ret.max()],
        },
        index=["均值", "标准差", "最小值", "最大值"],
    )
    print("\n📈 日收益描述统计：")
    print(stats.to_string(float_format=lambda x: f"{x:+.4%}"))
    diff = (ret - log_ret).dropna()
    print(f"\n   日度差值 R - r：平均 {diff.mean():.6f}，最大 {diff.max():.6f}（日频下差异很小）")

    # 两种方式还原净值：连乘 vs exp(累加)
    nav_simple = (1 + ret.fillna(0)).cumprod()
    nav_log = np.exp(log_ret.fillna(0).cumsum())
    assert np.allclose(nav_simple, nav_log)
    total_direct = close.iloc[-1] / close.iloc[0] - 1
    print(f"""
    🔁 三种方式计算区间总收益（应完全一致）：
       价格直接相除         P_T / P_0 - 1        = {total_direct:+.4%}
       简单收益连乘         ∏(1 + R_t) - 1       = {nav_simple.iloc[-1] - 1:+.4%}
       对数收益求和再取指数 exp(Σ r_t) - 1       = {nav_log.iloc[-1] - 1:+.4%}
       对数收益之和 Σ r_t（这是"对数总收益"）   = {log_ret.sum():+.4f}
    """)
    print("    ⚠️ 常见错误：把简单收益直接相加当作总收益")
    print(f"       Σ R_t = {ret.sum():+.4%}，与真实总收益 {total_direct:+.4%} 相差 {ret.sum() - total_direct:+.2%}")

    return pd.DataFrame({"close": close, "ret": ret, "log_ret": log_ret, "raw_close": df["close"]})


# ==================== 第三部分：时间可加 vs 截面可加 ====================


def additivity_demo():
    """组合收益必须用简单收益加权；多期累计用对数收益更方便"""
    print("\n" + "=" * 60)
    print("第三部分：时间可加 vs 截面可加")
    print("=" * 60)

    a = get_stock_daily_adj("600519", START, END)["adj_close"]
    b = get_stock_daily_adj("601088", START, END)["adj_close"]
    prices = pd.concat({"600519": a, "601088": b}, axis=1).dropna()
    w = np.array([0.5, 0.5])

    # 期初各投 50 万，按"买入持有"计算一天后的真实组合收益
    day0, day1 = prices.index[0], prices.index[1]
    p0, p1 = prices.loc[day0].to_numpy(), prices.loc[day1].to_numpy()
    true_port = (w * p1 / p0).sum() - 1
    simple = p1 / p0 - 1
    logr = np.log(p1 / p0)
    print(f"\n🧮 {day1.date()} 单日，两只股票各 50%：")
    print(f"   个股简单收益：{simple[0]:+.4%}、{simple[1]:+.4%}")
    print(f"   真实组合收益（按市值计算）    = {true_port:+.6%}")
    print(f"   简单收益加权和 Σ w·R          = {(w * simple).sum():+.6%}  ✅ 完全相等")
    print(f"   对数收益加权和再换算 exp(Σw·r)-1 = {np.exp((w * logr).sum()) - 1:+.6%}  ❌ 有偏差")

    # 多天：误差会累积
    rets = prices.pct_change().dropna()
    log_rets = np.log1p(rets)
    port_simple = (rets * w).sum(axis=1)  # 每日再平衡到 50/50 的组合
    port_logw = np.expm1((log_rets * w).sum(axis=1))
    nav_true = (1 + port_simple).cumprod()
    nav_wrong = (1 + port_logw).cumprod()
    print(f"\n📈 全区间（每日再平衡）期末净值：")
    print(f"   简单收益加权（正确）：{nav_true.iloc[-1]:.4f}")
    print(f"   对数收益加权（错误）：{nav_wrong.iloc[-1]:.4f}，误差 {nav_wrong.iloc[-1] / nav_true.iloc[-1] - 1:+.2%}")

    # 时间维度：对数收益分组求和即得区间收益
    yearly_log = np.log1p(rets["600519"]).groupby(rets.index.year).sum()
    yearly_simple = np.expm1(yearly_log)
    check = (1 + rets["600519"]).groupby(rets.index.year).prod() - 1
    assert np.allclose(yearly_simple, check)
    print("\n📅 用对数收益按年 groupby().sum() 再 expm1，即得各年简单收益（600519）：")
    print(pd.DataFrame({"对数收益和": yearly_log, "年度收益": yearly_simple}).to_string(
        float_format=lambda x: f"{x:+.2%}"))
    print("""
    💡 经验法则：
       横向（多资产组合）→ 用简单收益，Σ w_i·R_i 精确成立
       纵向（单资产多期）→ 用对数收益，groupby + sum 就能得到任意区间收益
    """)


# ==================== 第四部分：波动拖累 ====================


def volatility_drag(data: pd.DataFrame):
    """算术平均收益高估了真实复利增长"""
    print("\n" + "=" * 60)
    print("第四部分：波动拖累（Volatility Drag）")
    print("=" * 60)

    print("""
    一个策略日收益交替 +10% / -10%：算术平均 0%，但 1.1 × 0.9 = 0.99，两天亏 1%。
    近似关系：几何平均 g ≈ 算术平均 R̄ - σ² / 2
    """)

    # 模拟：算术平均相同、波动不同的三个"策略"
    rng = np.random.default_rng(42)
    n = 252 * 10
    rows = []
    for sigma in [0.005, 0.015, 0.03, 0.05]:
        r = rng.normal(0.0005, sigma, n)
        r = r - r.mean() + 0.0005  # 强制样本算术均值完全相同
        geo = np.prod(1 + r) ** (1 / n) - 1
        rows.append([f"{sigma:.1%}", r.mean(), geo, r.mean() - r.var() / 2, np.prod(1 + r)])
    sim = pd.DataFrame(rows, columns=["日波动", "算术平均", "几何平均", "R̄-σ²/2 近似", "10 年终值"])
    print("🧪 模拟 10 年：日均收益都是 0.05%，只有波动不同")
    print(sim.to_string(index=False, formatters={
        "算术平均": "{:+.4%}".format, "几何平均": "{:+.4%}".format,
        "R̄-σ²/2 近似": "{:+.4%}".format, "10 年终值": "{:.2f}".format,
    }))
    print("\n💡 算术平均一模一样，波动越大，复利终值越低 —— 这就是\"波动是收益的敌人\"")

    ret = data["ret"].dropna()
    arith = ret.mean()
    geo = (1 + ret).prod() ** (1 / len(ret)) - 1
    print(f"""
    📊 {NAME} 实际数据：
       算术平均日收益     {arith:+.4%}
       几何平均日收益     {geo:+.4%}
       近似 R̄ - σ²/2      {arith - ret.var() / 2:+.4%}
       差值（拖累）       {arith - geo:.4%} / 天
    """)


# ==================== 第五部分：常见陷阱 ====================


def pitfalls(data: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：常见陷阱")
    print("=" * 60)

    # 陷阱 1：未复权价格算收益
    raw_ret = data["raw_close"].pct_change()
    gap = (data["ret"] - raw_ret).dropna()
    ex_days = gap[gap.abs() > 1e-4]
    print("\n❌ 陷阱 1：用不复权价格计算收益")
    if len(ex_days) == 0:
        print("   （当前数据没有复权因子变化——离线合成数据的因子恒为 1，无法演示，联网后再运行）")
    else:
        show = pd.DataFrame({"不复权收益": raw_ret.loc[ex_days.index], "复权收益": data["ret"].loc[ex_days.index]})
        print(f"   找到 {len(ex_days)} 个除权除息日，不复权收益在这些天被系统性低估：")
        print(show.to_string(float_format=lambda x: f"{x:+.3%}"))
        total_raw = data["raw_close"].iloc[-1] / data["raw_close"].iloc[0] - 1
        total_adj = data["close"].iloc[-1] / data["close"].iloc[0] - 1
        print(f"   区间总收益：不复权 {total_raw:+.2%} vs 复权 {total_adj:+.2%}（差额来自分红再投资）")

    # 陷阱 2：长表直接 pct_change
    print("\n❌ 陷阱 2：长表不分组直接 pct_change —— 股票之间\"串味\"")
    long_df = pd.DataFrame({
        "code": ["600519"] * 3 + ["601088"] * 3,
        "date": list(pd.date_range("2024-01-02", periods=3)) * 2,
        "close": [1700.0, 1690.0, 1710.0, 30.0, 30.5, 30.2],
    })
    long_df["ret_wrong"] = long_df["close"].pct_change()
    long_df["ret_right"] = long_df.groupby("code")["close"].pct_change()
    print(long_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.1f}"))
    print("   第 4 行的 ret_wrong 把 601088 的第一天和 600519 的最后一天相除，得到 -98%！")

    # 陷阱 3：对数收益的下界
    print("\n❌ 陷阱 3：亏光时对数收益为 -∞")
    with np.errstate(divide="ignore"):
        print(f"   R = -100% → r = ln(0) = {np.log1p(-1.0)}；杠杆/期货回测中要注意这种极端情况")

    print("""
    ✅ 小结：
       1. 收益一律基于复权价（本课程用 get_stock_daily_adj 的 adj_close）
       2. 长表上的时间序列运算一律 groupby("code")
       3. 组合收益用简单收益加权，多期累计用对数收益求和
    """)


# ==================== 第六部分：可视化 ====================


def plot_returns(data: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：可视化")
    print("=" * 60)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ret, log_ret = data["ret"].dropna(), data["log_ret"].dropna()
    axes[0].scatter(ret, ret - log_ret, s=4, alpha=0.5)
    xs = np.linspace(ret.min(), ret.max(), 200)
    axes[0].plot(xs, xs - np.log1p(xs), "r-", lw=1, label="理论曲线 R - ln(1+R)")
    axes[0].set_title("简单收益与对数收益的差值")
    axes[0].set_xlabel("简单收益 R")
    axes[0].set_ylabel("R - r")
    axes[0].legend()

    nav = (1 + data["ret"].fillna(0)).cumprod()
    naive = 1 + data["ret"].fillna(0).cumsum()
    axes[1].plot(nav.index, nav, label="正确：∏(1+R)")
    axes[1].plot(naive.index, naive, "--", label="错误：1 + ΣR")
    axes[1].set_title(f"{NAME} 净值：连乘 vs 简单相加")
    axes[1].legend()

    path = save_figure(fig, "01_returns_simple_vs_log.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：使用合成数据，数值仅用于演示代码，不代表真实市场\n")
    intuition()
    data = real_data_returns()
    additivity_demo()
    volatility_drag(data)
    pitfalls(data)
    plot_returns(data)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 计算 600519 每个月的简单收益：分别用"对数收益 groupby 月份求和再 expm1"
       和"月末价格 pct_change"两种方法，验证结果一致
    2. 构造一个 3 只股票、权重 50%/30%/20% 的买入持有组合（不再平衡），
       计算其每日收益。提示：权重会随价格漂移，每日权重 = 前一日市值占比
    3. 模拟一个 3 倍杠杆 ETF：每日收益 = 3 × 沪深 300 日收益。比较 3 倍杠杆的
       区间总收益与"3 × 指数区间总收益"，用波动拖累解释差异
    4. 写一个 safe_log_return(price) 函数：检查价格非正、重复日期、未排序等问题后再计算
    """)


if __name__ == "__main__":
    main()
