"""
数据偏差与数据质量
==================

学习目标：
    1. 理解幸存者偏差、前视偏差、选择偏差等常见数据偏差
    2. 用实验量化幸存者偏差对回测结果的影响
    3. 实现一套可复用的数据质量检查函数
    4. 建立"先审计数据，再研究策略"的习惯

核心概念：
    - 幸存者偏差（Survivorship Bias）：只用今天还存在的股票做历史研究
    - 前视偏差（Look-ahead Bias）：使用了在当时还无法获得的信息
    - 数据质量检查：重复、缺失、逻辑错误、异常跳变、涨跌停越界

前置知识：
    - 02-a-share-market-rules.py、06-adjusted-prices.py、07-trading-calendar-alignment.py

环境要求：
    - pip install pandas numpy
    - 第一、二、四部分使用合成数据，第三部分需要网络（离线时自动使用合成数据）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_stock_daily

load_dotenv()


# ==================== 第一部分：常见数据偏差 ====================


def biases_overview():
    print("=" * 60)
    print("第一部分：常见数据偏差")
    print("=" * 60)

    print("""
    ┌──────────────┬────────────────────────────────┬────────────────────────────┐
    │ 偏差         │ 典型错误                       │ 正确做法                   │
    ├──────────────┼────────────────────────────────┼────────────────────────────┤
    │ 幸存者偏差   │ 用今天的股票列表回测 10 年前   │ 使用含退市股的历史股票池   │
    │ 前视偏差     │ 用 3 月 31 日"报告期"的财报    │ 按实际公告日期对齐         │
    │              │ 用当日收盘价算信号并以收盘成交 │ 信号次日执行               │
    │              │ 用前复权价格做价格阈值判断     │ 用当时的真实价格           │
    │ 选择偏差     │ 只研究"大家都在讨论"的股票     │ 预先定义股票池规则         │
    │ 数据窥探     │ 反复试参数直到回测好看         │ 记录尝试次数、样本外检验   │
    │ 成分股偏差   │ 用今天的沪深 300 成分回测      │ 使用历史时点成分股         │
    └──────────────┴────────────────────────────────┴────────────────────────────┘

    这些偏差的共同点：让回测"看起来很好"，实盘却赚不到钱。
    """)


# ==================== 第二部分：幸存者偏差实验 ====================


def survivorship_experiment(n_stocks: int = 1000, n_years: int = 10, seed: int = 42):
    """
    模拟实验：所有股票期望收益相同，但跌得多的股票会退市。
    比较"全部股票"与"只看幸存者"的平均收益。
    """
    print("\n" + "=" * 60)
    print("第二部分：幸存者偏差实验")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    days = n_years * 252
    daily = rng.normal(0.0002, 0.03, size=(days, n_stocks))  # 所有股票真实期望收益相同
    nav = np.cumprod(1 + daily, axis=0)

    # 退市规则（简化）：净值跌破 0.2（即亏损 80%）后退市，之后按退市价清算，收益冻结
    delisted_at = np.full(n_stocks, days)
    for j in range(n_stocks):
        hit = np.where(nav[:, j] < 0.2)[0]
        if len(hit) > 0:
            delisted_at[j] = hit[0]
            nav[hit[0]:, j] = nav[hit[0], j]

    survivors = delisted_at == days
    final = nav[-1]
    print(f"""
    📊 模拟设置：{n_stocks} 只股票，{n_years} 年，所有股票真实期望收益完全相同
       退市规则：累计下跌 80% 即退市

    结果：
       退市股票：{(~survivors).sum()} 只（{(~survivors).mean():.1%}）
       ✅ 全部股票（含退市）等权平均总收益：{final.mean() - 1:+.1%}
       ❌ 只看幸存者的平均总收益：        {final[survivors].mean() - 1:+.1%}
       偏差：{final[survivors].mean() - final.mean():+.1%}
    """)
    print("""
    💡 幸存者偏差凭空制造了收益：股票本身没有任何区别，
       只是"今天还活着"这个条件，就筛掉了历史上最差的那批。

    📌 A 股的特殊之处：
       - 2020 年后退市制度改革，退市数量明显增加 → 偏差在近年更严重
       - 小市值、低价股策略对幸存者偏差尤其敏感（它们更容易退市）
       - 免费数据源的"股票列表"通常不含退市股，需要单独获取退市名单
    """)


# ==================== 第三部分：数据质量检查函数 ====================


def check_ohlcv_quality(df: pd.DataFrame, limit_pct: float = 0.10) -> pd.DataFrame:
    """
    对单只股票的不复权日线做质量检查，返回问题清单

    Args:
        df: index 为日期，包含 open/high/low/close/volume 列
        limit_pct: 涨跌幅限制（用于检查越界）
    Returns:
        DataFrame[date, check, detail]
    """
    issues = []

    def add(mask: pd.Series, check: str, detail_fn):
        for d in df.index[mask.fillna(False).to_numpy()]:
            issues.append({"date": d, "check": check, "detail": detail_fn(d)})

    # 1. 重复日期
    dup = df.index.duplicated(keep=False)
    add(pd.Series(dup, index=df.index), "重复日期", lambda d: "同一日期出现多行")

    # 2. 缺失值
    add(df[["open", "high", "low", "close"]].isna().any(axis=1), "价格缺失", lambda d: "OHLC 存在 NaN")

    # 3. 非正价格 / 负成交量
    add((df[["open", "high", "low", "close"]] <= 0).any(axis=1), "非正价格", lambda d: f"close={df.loc[d, 'close']}")
    add(df["volume"] < 0, "负成交量", lambda d: f"volume={df.loc[d, 'volume']}")

    # 4. OHLC 逻辑：low ≤ open/close ≤ high
    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    add(bad_ohlc, "OHLC 逻辑错误", lambda d: f"O={df.loc[d, 'open']:.2f} H={df.loc[d, 'high']:.2f} L={df.loc[d, 'low']:.2f} C={df.loc[d, 'close']:.2f}")

    # 5. 涨跌幅越界（留 0.5% 容差覆盖四舍五入；除权日会误报，需要结合分红数据复核）
    ret = df["close"].pct_change()
    add(ret.abs() > limit_pct + 0.005, "涨跌幅越界", lambda d: f"日收益 {ret.loc[d]:+.2%}")

    # 6. 价格不变但有成交量为 0（可能是停牌日被填充）
    flat = (df["volume"] == 0) & (df["close"] == df["close"].shift())
    add(flat, "疑似停牌填充", lambda d: "成交量为 0 且价格不变")

    return pd.DataFrame(issues, columns=["date", "check", "detail"])


def quality_checks_demo():
    print("\n" + "=" * 60)
    print("第三部分：数据质量检查")
    print("=" * 60)

    # 先在一份干净数据上人为注入各种错误，验证检查函数能抓到
    df = get_stock_daily("600000", "20240101", "20240630")
    dirty = df.copy()
    d = dirty.index
    dirty.loc[d[5], "close"] = np.nan  # 缺失
    dirty.loc[d[10], "high"] = dirty.loc[d[10], "low"] - 0.1  # high < low
    dirty.loc[d[20], "close"] = dirty.loc[d[19], "close"] * 1.25  # 主板涨 25%（越界）
    dirty.loc[d[30], "volume"] = -100  # 负成交量
    dirty = pd.concat([dirty, dirty.iloc[[40]]]).sort_index()  # 重复日期

    issues = check_ohlcv_quality(dirty)
    print(f"\n🧪 注入 5 类错误后的检查结果（共 {len(issues)} 条）：")
    print(issues.to_string(index=False))
    print("\n📊 按类型汇总：")
    print(issues["check"].value_counts().to_string())

    clean_issues = check_ohlcv_quality(df)
    print(f"\n✅ 原始数据检查结果：{len(clean_issues)} 条问题")
    if len(clean_issues) > 0:
        print(clean_issues.head().to_string(index=False))
    print("""
    💡 注意：
    - 一个错误常常触发多条检查（缺失的收盘价会让前后两天的涨跌幅都异常）
    - "涨跌幅越界"在除权除息日会误报 —— 检查函数只负责发现，判断需要结合分红数据
    - 发现问题后：记录 → 查明原因 → 修复或剔除 → 在研究报告中披露
    """)


# ==================== 第四部分：前视偏差的隐蔽形式 ====================


def lookahead_examples():
    print("\n" + "=" * 60)
    print("第四部分：前视偏差的隐蔽形式")
    print("=" * 60)

    def strat_ret(signal: pd.Series, ret: pd.Series) -> float:
        pos = (signal < -1).astype(int).shift(1)  # z < -1 时买入（均值回归），次日执行
        return float((1 + pos * ret).prod() - 1)

    rng = np.random.default_rng(0)
    n_paths, n_days = 500, 250
    full, expanding = [], []
    for _ in range(n_paths):
        price = pd.Series(10 * np.cumprod(1 + rng.normal(0, 0.02, n_days)))
        ret = price.pct_change()
        # ❌ 错误：用全样本均值和标准差做标准化（包含了未来数据）
        z_full = (price - price.mean()) / price.std()
        # ✅ 正确：只用截至当天的数据（扩展窗口）
        z_exp = (price - price.expanding(20).mean()) / price.expanding(20).std()
        full.append(strat_ret(z_full, ret))
        expanding.append(strat_ret(z_exp, ret))
    full, expanding = np.array(full), np.array(expanding)

    print(f"""
    🧪 实验：在 {n_paths} 条随机游走价格上做 z-score 均值回归（理论上不应有收益）

                                平均收益     盈利路径占比
       ❌ 全样本 z-score        {full.mean():+7.2%}       {(full > 0).mean():.0%}
       ✅ 扩展窗口 z-score      {expanding.mean():+7.2%}       {(expanding > 0).mean():.0%}

    ❌ 全样本均值"知道"整段行情的平均水平：价格低于它时，"未来一定会回到均值附近"
       在随机数据上也能制造稳定的虚假收益。注意：单看一条路径可能看不出来，
       要用大量模拟才能看清偏差 —— 这也是研究方法的一课。
    """)

    print("""
    📋 其他隐蔽的前视偏差：
    1. 用全样本数据拟合标准化器 / PCA / 机器学习模型，再在同一段数据上回测
    2. 财报按报告期（如 2024-03-31）而不是公告日（如 2024-04-28）对齐
    3. 用当前的行业分类、ST 状态、指数成分去筛选历史股票
    4. 用前复权价格做"股价 < 5 元"之类的绝对价格判断
    5. 日线 resample 成周线后，在周一就使用了"本周"的周线数据
    6. 用到了数据供应商事后修正过的数据（而当时发布的是另一个值）
    """)


# ==================== 主函数 ====================


def main():
    biases_overview()
    survivorship_experiment()
    quality_checks_demo()
    lookahead_examples()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 修改幸存者偏差实验：把退市阈值改为亏损 50% / 90%，观察偏差如何变化
    2. 扩展 check_ohlcv_quality：根据股票代码自动选择涨跌幅限制（复用 02 课的 get_board）
    3. 对沪深 300 当前成分股批量运行质量检查，汇总问题最多的 10 只股票并查明原因
    4. 用 ak.stock_info_sh_delist() / ak.stock_info_sz_delist() 获取退市股票名单，
       统计近 5 年每年的退市数量
    """)


if __name__ == "__main__":
    main()
