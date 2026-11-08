"""
alphalens 因子报告
==================

学习目标：
    1. 把面板数据整理成 alphalens 需要的格式（MultiIndex 因子 + 价格宽表）
    2. 正确对齐价格：t 日收盘算因子，t+1 日开盘成交，避免前视
    3. 用 alphalens 的核心函数得到 IC、分组收益、换手率、自相关，并与 05、06 课手写结果核对
    4. 生成标准因子报告（tear sheet）并保存图片

核心概念：
    - factor：MultiIndex (date, asset) 的 Series；prices：index = 日期、columns = 股票的宽表
    - get_clean_factor_and_forward_returns：对齐因子、计算前向收益、分组，是所有分析的入口
    - max_loss：因缺失数据丢弃样本的上限；报错时先查数据对齐，不要直接调大
    - 价格对齐：alphalens 把"因子日当天的价格"当作买入价

前置知识：
    - 05-ic-ir-analysis.py、06-quantile-backtest.py

环境要求：
    - pip install alphalens-reloaded（导入名仍为 alphalens）
    - 已在 alphalens-reloaded 0.5.1 + pandas 3.1 上验证；未安装或调用失败时跳过相应部分
    - 数据来自本阶段共享面板 factor_data.py；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from factor_data import load_panel
from factor_tools import FACTOR_LABELS, calc_ic, exec_price, forward_returns, research_inputs
from utils.plotting import save_figure, setup_chinese_font

try:
    import alphalens as al
except ImportError:  # 优雅降级：没有 alphalens 时只讲解概念
    al = None

load_dotenv()

MAIN_FACTOR = "lowturn"
PERIODS = (1, 5, 20)


# ==================== 第一部分：数据格式与价格对齐 ====================


def to_alphalens_factor(factor: pd.DataFrame) -> pd.Series:
    """宽表 → MultiIndex (date, asset) 的 Series"""
    s = factor.stack().dropna()
    s.index.names = ["date", "asset"]
    return s


def build_prices(panel) -> dict[str, pd.DataFrame]:
    """
    三种价格口径。alphalens 用 prices[t + N] / prices[t] − 1 计算前向收益：
        next_open  第 t 行放 t+1 日开盘价 → 收益从 t+1 开盘算起（正确）
        close      第 t 行放 t 日收盘价   → 隐含"收盘算完因子、立刻以收盘价成交"（略乐观）
        lagged     第 t 行放 t−1 日收盘价 → 收益窗口包含 t 日本身（错位，严重前视）
    """
    px = exec_price(panel)
    close = panel.adj_close.ffill()
    return {"next_open": px.shift(-1).iloc[:-1], "close": close, "lagged": close.shift(1).iloc[1:]}


def data_format(panel):
    print("=" * 60)
    print("第一部分：数据格式与价格对齐")
    print("=" * 60)
    print("""
    alphalens 需要两份输入：

        factor（Series）                     prices（DataFrame）
        date        asset                    date        000001  000002  ...
        2024-01-31  000001   0.53            2024-01-31   10.12   12.30
                    000002  -1.20            2024-02-01   10.25   12.11

    ⚠️ 最容易出错的地方：alphalens 把"因子日当天的价格"当作买入价。

      因子在 t 日收盘后才算出来，而 prices 第 t 行如果放 t 日收盘价，
      就隐含了"收盘那一刻算完因子并以收盘价成交"。更稳妥的做法：
      把 t+1 日开盘价"向前挪一行"放到第 t 行：prices = adj_open.shift(-1)

      时间线：   t−1 收盘      t 收盘       t+1 开盘     t+2 开盘
                   │    t 日收益   │ 算因子     │ 买入        │
      错位 lagged: └─买入─────────┘ ❌ 收益窗口包含了算因子用过的数据
      收盘 close:                  └─买入 ─────┘ 隐含立刻成交
      正确 next_open:                           └─买入 ──────┘ ✅
    """)


# ==================== 第二部分：生成 factor_data ====================


def clean_factor_data(factor: pd.Series, prices: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """filter_zscore=None：不按 z 值过滤收益，便于与手写结果逐项核对"""
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return al.utils.get_clean_factor_and_forward_returns(
            factor, prices, quantiles=5, periods=PERIODS, filter_zscore=None, max_loss=0.35)


def build_factor_data(panel, clean, prices):
    print("\n" + "=" * 60)
    print(f"第二部分：get_clean_factor_and_forward_returns（{FACTOR_LABELS[MAIN_FACTOR]}）")
    print("=" * 60)
    factor = to_alphalens_factor(clean[MAIN_FACTOR])
    fd = clean_factor_data(factor, prices["next_open"])
    print(f"\n因子样本 {len(factor)} 条 → 清洗后 {len(fd)} 条，列：{list(fd.columns)}")
    print(fd.head(3).to_string(float_format=lambda v: f"{v:.4f}"))
    print("""
    💡 列名里的 1D / 5D / 20D 是前向收益的持有期（按 prices 的行数计，即交易日）。
       factor_quantile 是 1–5 的分组。"Dropped x%" 是因缺失而丢弃的比例：
       如果超过 max_loss 而报错，先检查日期和代码是否对齐，而不是把 max_loss 调大。
       本课的因子已经过可交易性过滤和预处理，alphalens 只负责分析。
    """)
    return fd


# ==================== 第三部分：核心分析函数 ====================


def core_analysis(panel, dates, clean, fd):
    print("\n" + "=" * 60)
    print("第三部分：核心分析函数，并与手写结果核对")
    print("=" * 60)

    cols = al.utils.get_forward_returns_columns(fd.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic = al.performance.factor_information_coefficient(fd)
        mean_q, _ = al.performance.mean_return_by_quantile(fd)
        ls = al.performance.factor_returns(fd)
        turn = {q: al.performance.quantile_turnover(fd["factor_quantile"], q, period=1) for q in (1, 5)}
        auto = al.performance.factor_rank_autocorrelation(fd, period=1)

    print("📋 IC（alphalens，Spearman）：")
    print(pd.DataFrame({"IC均值": ic.mean(), "IC标准差": ic.std(), "ICIR": ic.mean() / ic.std()}).T
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n📋 分组平均收益（相对全部股票均值，单位 bp / 持有期）：")
    print((mean_q * 1e4).to_string(float_format=lambda v: f"{v:+.1f}"))

    print("\n📋 因子加权多空组合（按因子值加权，零美元净敞口）每期平均收益：")
    print(ls.mean().to_string(float_format=lambda v: f"{v:+.4%}"))

    print(f"""
    📋 换手与自相关（相邻两个调仓日之间）：
       Q5 平均换手 {turn[5].mean():.3f}，Q1 平均换手 {turn[1].mean():.3f}
       因子排名自相关 {auto.mean():.3f} → 越接近 1，组合越稳定、换手越低
    """)

    # 核对：alphalens 的 20D IC 应与手写的"t+1 开盘持有 20 日"IC 一致
    mine = calc_ic(clean[MAIN_FACTOR], forward_returns(panel, dates, 20))
    theirs = ic[cols[-1]]
    both = pd.concat([mine, theirs], axis=1, keys=["手写", "alphalens"]).dropna()
    diff = (both["手写"] - both["alphalens"]).abs().max()
    print(f"""    🔍 核对：手写 Rank IC（20 日）均值 {both['手写'].mean():+.5f}，alphalens {both['alphalens'].mean():+.5f}，
       逐期最大差异 {diff:.2e}（{len(both)} 期）
       两者口径相同才能一致 —— 这也是检验自己代码的好办法。

    ⚠️ alphalens 的 20D 是"持有 20 个交易日"，06 课的持有期是"到下一个调仓日"，两者略有差异；
       这里的因子每月才有一期，所以 1D、5D 的结果只反映月末附近的表现。
    """)


# ==================== 第四部分：价格错位的后果 ====================


def alignment_experiment(clean, prices):
    print("\n" + "=" * 60)
    print("第四部分：价格错位的后果")
    print("=" * 60)

    factor = to_alphalens_factor(-clean["rev_1m"])  # 1 月动量：过去一个月涨得多的值大
    rows = {}
    labels = {"next_open": "✅ t+1 开盘", "close": "⚠️ t 收盘", "lagged": "❌ t−1 收盘（错位）"}
    for k, label in labels.items():
        fd = clean_factor_data(factor, prices[k], quiet=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = al.performance.factor_information_coefficient(fd).mean()
        rows[label] = ic.to_dict()
    print("📋 同一个因子（1 月动量），三种价格口径的 Rank IC 均值：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:+.4f}"))
    print("""
    💡 错位时，"未来收益"窗口包含了 t 日的收益，而 t 日的收益本身就是因子的一部分 ——
       这是机械的相关，不是预测能力。持有期越短，这一天在窗口中的占比越大，假象越强。
       如果一个因子的 1 日 IC 高得不合常理，先查价格和因子的对齐。
       "t 收盘"与"t+1 开盘"的差别则反映隔夜跳空：这部分收益实际很难拿到。
    """)


# ==================== 第五部分：Tear Sheet ====================


def tear_sheet(fd):
    print("\n" + "=" * 60)
    print("第五部分：生成因子报告（summary tear sheet）")
    print("=" * 60)

    setup_chinese_font()
    saved = []
    original_show = plt.show

    def capture(*args, **kwargs):
        """tear sheet 内部调用 plt.show() 后马上关闭图，这里把它改为"保存到 outputs" """
        for num in plt.get_fignums():
            fig = plt.figure(num)
            drawn = any(ax.has_data() for ax in fig.axes)
            if drawn:  # 跳过 alphalens 内部创建、但没有画任何内容的空白图
                saved.append(save_figure(fig, f"07_alphalens_{len(saved) + 1}.png", subdir="phase-7"))
            else:
                plt.close(fig)

    plt.show = capture
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            al.tears.create_summary_tear_sheet(fd, long_short=True)
    except Exception as e:  # 库版本兼容问题不应该中断课程
        print(f"⚠️ tear sheet 生成失败（{type(e).__name__}: {str(e)[:100]}），第三部分的函数结果仍然有效")
    finally:
        plt.show = original_show
    for p in saved:
        print(f"📈 图表已保存：{p}")
    print("""
    💡 读报告的顺序：
       1. Quantiles Statistics：每组股票数是否均衡
       2. Returns Analysis：分组收益是否单调（Ann. alpha / beta 按因子日频率折算，月频因子要谨慎）
       3. Information Analysis：IC 均值、Risk-Adjusted IC（即 ICIR）、t 值
       4. Turnover Analysis：换手和自相关
       ⚠️ 因子是月频时，Turnover Analysis 的列名会误导：它按"因子日期的行数"位移，
          这里的 1D 实际是相邻 1 个调仓日（1 个月），5D 是相隔 5 个调仓日，不是 5 个交易日
       完整版用 al.tears.create_full_tear_sheet(fd)，内容更多但也更慢。
       注意：报告里的收益不含交易成本和涨跌停约束，也不等于真实回测。
    """)


# ==================== 主函数 ====================


def main():
    panel = load_panel()
    dates, mask, ret, raw, clean = research_inputs(panel)
    data_format(panel)
    if al is None:
        print("⚠️ 未安装 alphalens-reloaded，跳过后续部分（05、06 课已手写了相同的分析）")
        return
    prices = build_prices(panel)
    fd = build_factor_data(panel, clean, prices)
    core_analysis(panel, dates, clean, fd)
    alignment_experiment(clean, prices)
    tear_sheet(fd)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 groupby 参数传入行业，运行 group_neutral 分析，比较行业内分组与全市场分组的结果
    2. 把 periods 改为 (1, 10, 40)，用 al.plotting.plot_quantile_returns_bar 画分组收益图
    3. 把因子从月频改为周频（每周五），比较 IC、换手率和因子自相关
    4. 故意把 max_loss 设为 0.01 触发报错，读懂错误信息中丢弃数据的各项来源
    """)


if __name__ == "__main__":
    main()
