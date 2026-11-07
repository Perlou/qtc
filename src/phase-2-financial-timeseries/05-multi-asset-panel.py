"""
多资产面板数据：宽表与长表
==========================

学习目标：
    1. 理解宽表（Wide）与长表（Long / Tidy）两种形态及各自适用场景
    2. 熟练使用 pivot / melt / stack / unstack 互相转换（注意 pandas 3 的 stack 行为变化）
    3. 区分时间序列运算（沿 axis=0）与截面运算（沿 axis=1）
    4. 在长表上用 groupby 正确做分组时间序列运算，避免"串味"
    5. 认识面板中的 NaN：停牌、未上市、退市，以及"可交易状态"表

核心概念：
    - 宽表：index = 日期，columns = 代码，适合向量化与矩阵运算
    - 长表：每行一条 (日期, 代码, 字段...) 记录，适合存储与分组
    - 截面（Cross-section）：同一天所有股票之间的比较，如排名、去均值

前置知识：
    - 01-returns-simple-vs-log.py、03-rolling-windows.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import get_close_panel, get_stock_daily_adj, is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 多序列按固定顺序取色
START, END = "20180101", "20241231"
STOCKS = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "600036": "招商银行",
    "601318": "中国平安",
    "000333": "美的集团",
    "601088": "中国神华",
}


# ==================== 第一部分：获取宽表并认识 NaN ====================


def load_panel() -> pd.DataFrame:
    print("=" * 60)
    print("第一部分：宽表与面板中的 NaN")
    print("=" * 60)

    print("""
    宽表（Wide）                              长表（Long / Tidy）
                 600519   000858   ...        date        code    close
    2024-01-02   1685.0   140.2               2024-01-02  600519  1685.0
    2024-01-03   1680.1   139.5               2024-01-02  000858  140.2
    ...                                       ...

    ┌──────────┬──────────────────────────────┬──────────────────────────────┐
    │ 维度     │ 宽表                         │ 长表                         │
    ├──────────┼──────────────────────────────┼──────────────────────────────┤
    │ 结构     │ index=日期, columns=代码     │ 每行 (日期, 代码, 字段...)   │
    │ 适合     │ 向量化、截面运算、矩阵乘法   │ 存储、多字段、groupby        │
    │ 缺失     │ 显式 NaN（停牌、未上市）     │ 表现为"没有这一行"           │
    └──────────┴──────────────────────────────┴──────────────────────────────┘
    """)

    wide = get_close_panel(list(STOCKS), START, END)
    print(f"📊 后复权收盘价宽表：{wide.shape[0]} 个交易日 × {wide.shape[1]} 只股票")
    print(wide.tail(3).round(2).to_string())

    na = wide.isna().sum()
    print("\n🕳️ 各股票的 NaN 个数（停牌 / 未上市 / 数据缺失）：")
    print("   " + "  ".join(f"{c}:{n}" for c, n in na.items()))
    if na.sum() > 0:
        code = na.idxmax()
        # 把连续的 NaN 切成"段"：值发生变化时 cumsum + 1，同一段内编号相同（游程编码的常用技巧）
        is_na = wide[code].isna()
        seg_id = (is_na != is_na.shift()).cumsum()[is_na]
        print(f"   {code} {STOCKS[code]} 的缺失段：")
        segments = [seg for _, seg in seg_id.groupby(seg_id)]
        for seg in segments[:5]:
            print(f"      {seg.index[0].date()} ~ {seg.index[-1].date()}（{len(seg)} 个交易日）")
        if len(segments) > 5:
            print(f"      ……共 {len(segments)} 段")
        print("   → 宽表按日期外连接对齐，某只股票当天没有数据就是 NaN；连续多日缺失通常是停牌")
    else:
        print("   当前数据没有缺失（离线合成数据各股票日期完全相同）")
    return wide


# ==================== 第二部分：宽表 ⇄ 长表 ====================


def reshape_demo(wide: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：宽表 ⇄ 长表互相转换")
    print("=" * 60)

    # 宽 → 长：melt
    long_df = wide.reset_index().melt(id_vars="date", var_name="code", value_name="close")
    print(f"\n🔄 melt：宽表 {wide.shape} → 长表 {long_df.shape}")
    print(long_df.head(3).to_string(index=False))
    print(f"   dtypes：{dict(long_df.dtypes.astype(str))}")
    print("   （pandas 3 中字符串列默认是 str dtype，不再是 object）")

    # 长 → 宽：pivot
    back = long_df.pivot(index="date", columns="code", values="close")
    same = back.equals(wide[back.columns]) or np.allclose(back.fillna(-1), wide[back.columns].fillna(-1))
    print(f"\n🔄 pivot：长表 → 宽表 {back.shape}，与原宽表一致：{same}")

    # MultiIndex：stack / unstack
    stacked = wide.stack()
    print(f"""
    🔄 stack / unstack（MultiIndex 方式）：
       wide.stack() 得到 {len(stacked)} 行，其中 NaN {stacked.isna().sum()} 行
       pandas 3 的 stack 不再自动丢弃 NaN（旧版默认 dropna=True 会悄悄删掉停牌日）
       如需旧行为，显式写：wide.stack().dropna() → {len(stacked.dropna())} 行
    """)
    restored = stacked.unstack()
    print(f"   unstack 还原：{restored.shape}，与原宽表一致：{restored.equals(wide)}")

    # 长表存多个字段更自然
    fields = []
    for code in list(STOCKS)[:2]:
        d = get_stock_daily_adj(code, START, END)[["close", "adj_close", "volume"]].assign(code=code)
        fields.append(d.reset_index())
    multi = pd.concat(fields, ignore_index=True)
    print("\n📦 多字段长表（每行一个 日期×股票，多列字段）：")
    print(multi.head(3).to_string(index=False))
    wide_vol = multi.pivot(index="date", columns="code", values="volume")
    print(f"   取出其中一个字段转宽表：pivot(values='volume') → {wide_vol.shape}")
    print("   也可以 pivot(values=['close','volume']) 得到 MultiIndex 列的宽表")
    return long_df


# ==================== 第三部分：时间序列运算 vs 截面运算 ====================


def ts_vs_cross_section(wide: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：时间序列运算 vs 截面运算")
    print("=" * 60)

    print("""
                  时间序列运算（沿 axis=0，每只股票独立）
                  ─────────────────────────────────────▶
                  ┃  600519  000858  600036 ...
      截面运算    ┃   ...
     （沿 axis=1，┃   ...
      每一天）    ▼
    """)

    rets = wide.pct_change(fill_method=None)  # 不自动填充 NaN，停牌日收益保持 NaN
    mom_20 = wide / wide.shift(20) - 1  # 时间序列：20 日动量
    mom_rank = mom_20.rank(axis=1, pct=True)  # 截面：每天的百分位排名
    mom_demean = mom_20.sub(mom_20.mean(axis=1), axis=0)  # 截面：去均值（相对强弱）

    last = mom_20.index[-1]
    snap = pd.DataFrame({
        "名称": pd.Series(STOCKS),
        "20日动量": mom_20.loc[last],
        "截面排名(pct)": mom_rank.loc[last],
        "去均值动量": mom_demean.loc[last],
    }).sort_values("截面排名(pct)", ascending=False)
    print(f"📊 {last.date()} 的截面快照：")
    print(snap.to_string(formatters={
        "20日动量": "{:+.2%}".format, "截面排名(pct)": "{:.2f}".format, "去均值动量": "{:+.2%}".format,
    }))
    print("   💡 rank(axis=1, pct=True) 是 Phase 7 因子研究最常用的操作：把因子值变成 0~1 的相对位置")

    # 等权组合：NaN 自动跳过
    ew = rets.mean(axis=1)
    nav = (1 + ew.fillna(0)).cumprod()
    print(f"""
    📈 等权组合（每日再平衡）：rets.mean(axis=1)
       NaN 自动跳过 → 停牌股当天不参与平均
       期末净值：{nav.iloc[-1]:.3f}（{nav.index[0].date()} 起）
       各股票期末净值：{', '.join(f'{c}:{v:.2f}' for c, v in (wide.iloc[-1] / wide.bfill().iloc[0]).items())}
    """)

    corr = rets.corr()
    print("🔗 日收益相关系数矩阵：")
    print(corr.round(2).to_string())
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()  # 上三角，去掉 NaN
    print(f"   相关最高：{tri.idxmax()} = {tri.max():.2f}；最低：{tri.idxmin()} = {tri.min():.2f}")
    return rets, corr


# ==================== 第四部分：长表上的分组运算 ====================


def groupby_on_long(long_df: pd.DataFrame, wide: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：长表上的分组运算")
    print("=" * 60)

    long_df = long_df.sort_values(["code", "date"]).reset_index(drop=True)

    wrong = long_df["close"].pct_change(fill_method=None)
    right = long_df.groupby("code")["close"].pct_change(fill_method=None)
    # 注意：NaN 与任何数比较都是 False，"(wrong - right).abs() > 0" 会漏掉 right 为 NaN 的行
    same = np.isclose(wrong, right) | (wrong.isna() & right.isna())
    bad = ~same
    first_rows = long_df.groupby("code").head(1).index
    print(f"""
    ❌ long_df["close"].pct_change()：
       有 {bad.sum()} 行与正确结果不同，全部是每只股票的第一行：{bad[first_rows].sum() == bad.sum()}
       例：第 {first_rows[1]} 行（{long_df.loc[first_rows[1], 'code']} 的第一天）
           错误值 {wrong.iloc[first_rows[1]]:+.2%} ← 用的是上一只股票最后一天的价格
    ✅ long_df.groupby("code")["close"].pct_change()
    """)

    # 分组滚动：transform 保持原长度
    long_df["ma20"] = long_df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean())
    wide_ma20 = wide.rolling(20).mean()
    check = long_df.pivot(index="date", columns="code", values="ma20")
    print(f"   groupby + transform 计算的 MA20 与宽表 rolling(20) 一致："
          f"{np.allclose(check.fillna(0), wide_ma20[check.columns].fillna(0))}")

    # 性能对比
    def timeit(fn, n=5):
        best = np.inf
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_wide = timeit(lambda: wide.rolling(20).mean())
    t_long = timeit(lambda: long_df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean()))
    t_long2 = timeit(lambda: long_df.groupby("code")["close"].rolling(20).mean())
    print(f"""
    ⏱️ 计算 MA20 的耗时（{wide.shape[1]} 只股票，取 5 次最快）：
       宽表 wide.rolling(20).mean()                    {t_wide * 1000:7.2f} ms
       长表 groupby().transform(lambda: rolling)       {t_long * 1000:7.2f} ms
       长表 groupby().rolling(20).mean()               {t_long2 * 1000:7.2f} ms
    💡 股票越多，宽表的优势越明显；计算用宽表，存储用长表
    """)


# ==================== 第五部分：停牌、填充与可交易状态 ====================


def tradable_mask(wide: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：停牌、填充与可交易状态")
    print("=" * 60)

    print("""
    宽表里的 NaN 可能有三种完全不同的含义：
    ┌──────────┬──────────────────────────┬──────────────────────────────┐
    │ 情形     │ 含义                     │ 处理                         │
    ├──────────┼──────────────────────────┼──────────────────────────────┤
    │ 未上市   │ 那时还不存在             │ 保持 NaN，不能参与任何计算   │
    │ 停牌     │ 存在但不能交易           │ 估值可 ffill，交易必须禁止   │
    │ 退市     │ 不再存在                 │ 按退市价清算，之后保持 NaN   │
    └──────────┴──────────────────────────┴──────────────────────────────┘
    """)

    tradable = wide.notna()  # 最简单的"可交易状态"表：当天有行情 = 可交易
    filled = wide.ffill()
    rets_raw = wide.pct_change(fill_method=None)
    rets_ffill = filled.pct_change()
    code = wide.isna().sum().idxmax()
    n_na = int(wide[code].isna().sum())
    print(f"📋 以 {code} 为例（NaN {n_na} 天）：")
    # 复牌日 = 今天有价、昨天没价，且不是这只股票的第一天
    resumes = wide.index[wide[code].notna() & wide[code].shift(1).isna()]
    resumes = resumes[resumes > wide[code].first_valid_index()]
    if len(resumes) > 0:
        resume = resumes[-1]
        print(f"""   ffill 后：停牌期间价格不变、收益为 0 → 净值估值合理
   但复牌日 {resume.date()} 的收益：
      不填充：{rets_raw.loc[resume, code]}（前一天是 NaN，无法计算）
      ffill 后：{rets_ffill.loc[resume, code]:+.2%}（= 停牌期间的累计收益，集中在这一天体现）
   ⚠️ 如果策略假设停牌期间也能按 ffill 的价格买卖，就会严重失真 → 必须配合 tradable 表使用
        """)
    else:
        print("   当前数据没有停牌后复牌的情形，无法演示；联网运行可看到真实停牌数据")

    print(f"   可交易状态表 tradable = wide.notna()：形状 {tradable.shape}，"
          f"整体可交易比例 {tradable.to_numpy().mean():.2%}")
    print("""
    ⚠️ 幸存者偏差的种子：本课的股票列表是"今天还在、今天很有名"的公司，
       宽表从一开始就排除了历史上被剔除和退市的股票（Phase 4 详述）
    """)


# ==================== 第六部分：可视化 ====================


def plot_panel(wide: pd.DataFrame, corr: pd.DataFrame):
    setup_chinese_font()
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SERIES_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})

    norm = wide / wide.bfill().iloc[0]
    for code in norm.columns:
        axes[0].plot(norm.index, norm[code], lw=1, label=f"{code} {STOCKS[code]}")
    axes[0].set_yscale("log")
    axes[0].set_title("归一化净值（对数坐标）")
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [STOCKS[c] for c in corr.columns]
    axes[1].set_xticks(range(len(labels)), labels, rotation=45)
    axes[1].set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].set_title("日收益相关系数")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    path = save_figure(fig, "05_multi_asset_panel.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    wide = load_panel()
    long_df = reshape_demo(wide)
    _, corr = ts_vs_cross_section(wide)
    groupby_on_long(long_df, wide)
    tradable_mask(wide)
    plot_panel(wide, corr)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 每月最后一个交易日，按 20 日动量截面排名，买入排名前 2 的股票、等权持有一个月，
       计算这个"迷你动量策略"的净值（注意 shift，月末算信号、下个交易日才持仓）
    2. 把宽表收益 rets 转成长表，用 groupby("code") 计算每只股票的年化波动率，
       再用宽表 rets.std() * √252 验证结果一致
    3. 构造一个"可交易状态"表：停牌（NaN）或当日涨跌幅绝对值 ≥ 9.8% 视为不可交易，
       统计每只股票的不可交易天数
    4. 把宽表保存为 Parquet 长表（melt 后），再用 DuckDB 查询"每只股票每年的收益"
    """)


if __name__ == "__main__":
    main()
