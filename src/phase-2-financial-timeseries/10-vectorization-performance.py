"""
向量化与性能优化
================

学习目标：
    1. 亲手测出 iterrows、for + iloc、rolling.apply、rolling、NumPy、Numba 的速度差距
    2. 掌握三种"把循环变成数组运算"的写法：累加和、滑动窗口视图、np.select
    3. 知道面板数据为什么要在宽表上算
    4. 理解向量化的边界：路径依赖逻辑（移动止损）与 Numba
    5. 警惕"快但错"：用打乱收益的实验检测前视偏差

核心概念：
    - 向量化：把循环下推到 C 层，对整个数组一次性计算
    - 路径依赖：当前状态依赖上一步的结果，无法简单向量化
    - 前视偏差自检：在打乱后的数据上，策略不应该还有优势

前置知识：
    - 03-rolling-windows.py、05-multi-asset-panel.py、06-technical-indicators.py

环境要求：
    - pip install pandas numpy matplotlib akshare
    - Numba 为可选项：未安装时跳过 JIT 对比，其余内容照常运行
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
    - 本课打印的耗时取决于你的机器，每次运行都会有波动
"""

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.lib.stride_tricks import sliding_window_view

from utils.data import get_stock_daily_adj, is_offline, synthetic_ohlcv
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

try:
    from numba import njit
except ImportError:
    njit = None

TRADING_DAYS = 252
START, END = "20180101", "20241231"
WINDOW = 20


def bench(fn, repeat: int = 3) -> float:
    """取多次运行的最快耗时（秒）：最快值受系统噪声影响最小"""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def timing_table(rows: list[tuple[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["写法", "耗时(ms)"])
    df["耗时(ms)"] = df["耗时(ms)"] * 1000
    df["相对最快"] = df["耗时(ms)"] / df["耗时(ms)"].min()
    return df.sort_values("耗时(ms)", ascending=False)


def print_timing(rows: list[tuple[str, float]]):
    print(timing_table(rows).to_string(index=False, formatters={
        "耗时(ms)": "{:,.3f}".format, "相对最快": "{:,.1f}x".format,
    }))


# ==================== 第一部分：同一个 MA20 的多种写法 ====================


def ma_iterrows(df: pd.DataFrame, n: int) -> np.ndarray:
    out, buf = [], deque(maxlen=n)
    for _, row in df.iterrows():
        buf.append(row["close"])
        out.append(sum(buf) / n if len(buf) == n else np.nan)
    return np.array(out)


def ma_iloc(close: pd.Series, n: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(n - 1, len(close)):
        out[i] = close.iloc[i - n + 1 : i + 1].mean()
    return out


def ma_cumsum(x: np.ndarray, n: int) -> np.ndarray:
    """累加和技巧：窗口和 = 两个前缀和之差，O(n) 且不依赖 pandas"""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ma_sliding(x: np.ndarray, n: int) -> np.ndarray:
    """滑动窗口视图：不复制数据，把一维数组看成 (len-n+1, n) 的二维数组"""
    out = np.full(len(x), np.nan)
    out[n - 1 :] = sliding_window_view(x, n).mean(axis=1)
    return out


def ma_benchmark() -> pd.DataFrame:
    print("=" * 60)
    print("第一部分：同一个 MA20 的多种写法")
    print("=" * 60)

    print("""
    🛒 for 循环像收银员一件一件扫码；向量化像把整车货物开过 RFID 门，一次扫完。
       Python 循环每次迭代都要经过解释器，NumPy/pandas 把循环下推到 C 层。
    """)

    # 性能实验用固定长度的合成数据，结果不受网络影响
    df = synthetic_ohlcv("PERF", "20000101", "20241231")
    close = df["close"]
    x = close.to_numpy()
    print(f"📊 测试数据：{len(close)} 个交易日（合成），计算 MA{WINDOW}\n")

    expected = close.rolling(WINDOW).mean().to_numpy()
    candidates = {
        "df.iterrows()": lambda: ma_iterrows(df, WINDOW),
        "for + .iloc[].mean()": lambda: ma_iloc(close, WINDOW),
        "rolling.apply(np.mean)": lambda: close.rolling(WINDOW).apply(np.mean).to_numpy(),
        "rolling.apply(raw=True)": lambda: close.rolling(WINDOW).apply(np.mean, raw=True).to_numpy(),
        "rolling().mean()": lambda: close.rolling(WINDOW).mean().to_numpy(),
        "NumPy 累加和": lambda: ma_cumsum(x, WINDOW),
        "NumPy 滑动窗口视图": lambda: ma_sliding(x, WINDOW),
    }
    rows = []
    for name, fn in candidates.items():
        assert np.allclose(fn(), expected, equal_nan=True), name  # 先对拍，再比速度
        slow = name in ("df.iterrows()", "for + .iloc[].mean()", "rolling.apply(np.mean)")
        rows.append((name, bench(fn, repeat=1 if slow else 5)))
    print_timing(rows)
    print("""
    ✅ 上面所有写法都先通过了结果一致性断言：优化的前提是结果不变
    💡 优先级：pandas 内置方法（可读性好且够快）→ NumPy → Numba。
       iterrows 与 for + iloc 谁更慢取决于循环体：这里 iterrows 版只做轻量累加，
       而 iloc 版每步都切片并调用 mean。两者都比内置方法慢几个数量级，这才是重点。
       iterrows 还有一个隐患：每行都被转成 Series，混合 dtype 时类型可能被悄悄改变。
    """)
    return df


# ==================== 第二部分：rolling.apply 与滑动窗口 ====================


def rolling_slope(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：没有内置方法时 —— 滚动回归斜率")
    print("=" * 60)

    close = df["close"].iloc[:3000]
    x = close.to_numpy()
    t = np.arange(WINDOW)

    def slope_series(s: pd.Series) -> float:
        return np.polyfit(t, s.to_numpy(), 1)[0]

    def slope_raw(a: np.ndarray) -> float:
        return np.polyfit(t, a, 1)[0]

    # 最小二乘斜率 = Σ(t - t̄)·y / Σ(t - t̄)²，对窗口来说就是一次点积
    weights = (t - t.mean()) / ((t - t.mean()) ** 2).sum()

    def slope_vector() -> np.ndarray:
        out = np.full(len(x), np.nan)
        out[WINDOW - 1 :] = sliding_window_view(x, WINDOW) @ weights
        return out

    a = close.rolling(WINDOW).apply(slope_series, raw=False).to_numpy()
    b = close.rolling(WINDOW).apply(slope_raw, raw=True).to_numpy()
    c = slope_vector()
    assert np.allclose(a, b, equal_nan=True) and np.allclose(a, c, equal_nan=True)

    print(f"计算 {len(close)} 天的 {WINDOW} 日滚动线性回归斜率：\n")
    print_timing([
        ("rolling.apply(raw=False)", bench(lambda: close.rolling(WINDOW).apply(slope_series, raw=False), 1)),
        ("rolling.apply(raw=True)", bench(lambda: close.rolling(WINDOW).apply(slope_raw, raw=True), 1)),
        ("sliding_window_view @ 权重", bench(slope_vector, 5)),
    ])
    print("""
    💡 raw=True 传 ndarray 而不是 Series，省去每个窗口构造 Series 的开销。
       最快的写法来自一点数学：窗口内的线性运算都能写成与固定权重的点积。
    """)


# ==================== 第三部分：用 np.select 代替 if-else ====================


def conditional_logic(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：用 np.where / np.select 代替逐行 if-else")
    print("=" * 60)

    close = df["close"]
    ma = close.rolling(WINDOW).mean()

    def signal_loop() -> np.ndarray:
        c, m = close.to_numpy(), ma.to_numpy()
        out = np.zeros(len(c))
        for i in range(len(c)):
            if c[i] > m[i] * 1.02:
                out[i] = 1
            elif c[i] < m[i] * 0.98:
                out[i] = -1
        return out

    def signal_select() -> np.ndarray:
        return np.select([close > ma * 1.02, close < ma * 0.98], [1.0, -1.0], default=0.0)

    assert np.array_equal(signal_loop(), signal_select())
    print("三态信号：高于均线 2% → +1，低于均线 2% → -1，否则 0\n")
    print_timing([("for + if/elif", bench(signal_loop, 3)), ("np.select", bench(signal_select, 5))])
    print("\n    💡 np.select 按条件顺序取第一个命中的分支，语义与 if/elif 相同")


# ==================== 第四部分：宽表 vs 长表 ====================


def panel_performance():
    print("\n" + "=" * 60)
    print("第四部分：面板数据 —— 宽表 vs 长表")
    print("=" * 60)

    rng = np.random.default_rng(10)
    n_days, n_stocks = 1500, 300
    dates = pd.bdate_range("2019-01-01", periods=n_days, name="date")
    codes = [f"S{i:04d}" for i in range(n_stocks)]
    wide = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_stocks)), axis=0)), index=dates, columns=codes)
    long_df = wide.reset_index().melt(id_vars="date", var_name="code", value_name="close").sort_values(["code", "date"])
    print(f"📊 {n_days} 天 × {n_stocks} 只股票 = {len(long_df):,} 行长表（模拟数据）\n")

    print("任务 1：每只股票的 MA20")
    print_timing([
        ("长表 groupby + transform(lambda)", bench(lambda: long_df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean()), 2)),
        ("长表 groupby().rolling()", bench(lambda: long_df.groupby("code")["close"].rolling(20).mean(), 2)),
        ("宽表 rolling()", bench(lambda: wide.rolling(20).mean(), 3)),
    ])
    print("\n任务 2：每天的截面排名")
    print_timing([
        ("长表 groupby('date').rank()", bench(lambda: long_df.groupby("date")["close"].rank(pct=True), 2)),
        ("宽表 rank(axis=1)", bench(lambda: wide.rank(axis=1, pct=True), 3)),
    ])

    wide32 = wide.astype("float32")
    nav64 = (1 + wide.pct_change().fillna(0)).cumprod()
    nav32 = (1 + wide32.pct_change().fillna(0)).cumprod()
    rel_err = ((nav32.astype("float64") - nav64).abs() / nav64).to_numpy().max()
    print(f"""
    💾 数据类型：float64 占 {wide.memory_usage().sum() / 1e6:.1f} MB，float32 占 {wide32.memory_usage().sum() / 1e6:.1f} MB
       代价是精度：float32 单步相对精度约 {np.finfo(np.float32).eps:.1e}，累乘 {n_days} 天后最大相对误差 {rel_err:.2e}
       累计运算（cumprod、cumsum）会把舍入误差累积起来：存储可以省，核心计算建议保持 float64
    """)


# ==================== 第五部分：向量化的边界 ====================


def trailing_stop_py(close: np.ndarray, entry: np.ndarray, stop_pct: float) -> np.ndarray:
    """entry[i] 为 True 表示 i 日收盘出现入场信号；返回每日持仓 0/1"""
    pos = np.zeros(len(close))
    holding, peak = False, 0.0
    for i in range(len(close)):
        if holding:
            peak = max(peak, close[i])
            if close[i] < peak * (1 - stop_pct):
                holding = False
        elif entry[i]:
            holding, peak = True, close[i]
        pos[i] = 1.0 if holding else 0.0
    return pos


def path_dependence(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：向量化的边界 —— 路径依赖与 Numba")
    print("=" * 60)

    print("""
    移动止损：从买入后的最高价回落 8% 就卖出。
       是否止损 ← 取决于持仓后的最高价 ← 取决于何时入场 ← 取决于之前是否止损过
    当前状态依赖上一步结果，这种逻辑无法用 shift / cummax 简单表达。
    """)

    close_s = df["close"]
    close = close_s.to_numpy()
    entry = (close_s > close_s.rolling(WINDOW).mean()).to_numpy()
    stop = 0.08

    pos = trailing_stop_py(close, entry, stop)

    # 看起来很像、但语义不对的向量化写法：cummax 从第一天开始算，不知道"何时入场"
    naive = (entry & (close >= np.maximum.accumulate(close) * (1 - stop))).astype(float)
    print(f"""    📊 持仓天数：路径版 {int(pos.sum())}，cummax 向量化版 {int(naive.sum())}
       两者持仓不同的天数：{int((pos != naive).sum())} / {len(pos)}
       ❌ 向量化版的"最高价"是全历史最高，而不是"本次持仓以来的最高"，语义已经变了
    """)

    if njit is None:
        print("    ℹ️ 未安装 Numba，跳过 JIT 对比（pip install numba 后可运行）")
        return

    jitted = njit(trailing_stop_py)
    t0 = time.perf_counter()
    pos_jit = jitted(close, entry, stop)  # 第一次调用包含编译
    compile_s = time.perf_counter() - t0
    assert np.array_equal(pos, pos_jit)

    print_timing([
        ("纯 Python 循环", bench(lambda: trailing_stop_py(close, entry, stop), 3)),
        ("Numba @njit（已编译）", bench(lambda: jitted(close, entry, stop), 5)),
    ])
    print(f"""
    ⏱️ 首次调用含编译，耗时 {compile_s:.2f} 秒：只调用一次时 JIT 不一定划算，
       参数扫描、蒙特卡洛这类反复调用的场景收益才大。
    💡 这正是 Phase 4"向量化回测 vs 事件驱动回测"之争的缩影：
       向量化快，但表达不了复杂的路径依赖逻辑
    """)


# ==================== 第六部分：快但错 ====================


def fast_but_wrong() -> dict:
    print("\n" + "=" * 60)
    print("第六部分：快但错 —— 用打乱实验检测前视偏差")
    print("=" * 60)

    print("""
    向量化最大的风险不是性能，而是一个 shift 方向错误导致前视偏差：
    它不会报错，只会让回测"好得不真实"。

       signal = (close > ma20).astype(int)
       ❌ signal * ret            今天收盘才知道信号，却吃到了今天全天的涨幅
       ✅ signal.shift(1) * ret   今天的信号，明天才持仓
    """)

    def strategies(price: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """对价格宽表（每列一条路径）计算三种年化收益（算术口径）"""
        ret = price.pct_change()
        signal = (price > price.rolling(WINDOW).mean()).astype(float)
        ann = lambda r: r.iloc[WINDOW:].mean() * TRADING_DAYS  # noqa: E731
        return ann(ret), ann(signal.shift(1) * ret), ann(signal * ret)

    close = get_stock_daily_adj("600519", START, END)["adj_close"]
    hold, right, wrong = strategies(close.to_frame("real"))

    # 打乱：把日收益随机重排后重建价格，时间结构被破坏，均线策略不应再有优势
    rets = close.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(2024)
    n_paths = 300
    shuffled = rng.permuted(np.tile(rets, (n_paths, 1)), axis=1).T
    paths = pd.DataFrame(np.vstack([np.ones(n_paths), np.cumprod(1 + shuffled, axis=0)]))
    t0 = time.perf_counter()
    s_hold, s_right, s_wrong = strategies(paths)
    elapsed = time.perf_counter() - t0

    table = pd.DataFrame({
        "真实序列": [hold.iloc[0], right.iloc[0], wrong.iloc[0]],
        f"打乱 {n_paths} 次的均值": [s_hold.mean(), s_right.mean(), s_wrong.mean()],
        "打乱后跑赢买入持有的比例": [np.nan, (s_right > s_hold).mean(), (s_wrong > s_hold).mean()],
    }, index=["买入持有", "✅ signal.shift(1) × ret", "❌ signal × ret"])
    print("📊 均线策略年化收益（算术口径，不计成本）：")
    print(table.to_string(na_rep="-", formatters={
        "真实序列": "{:+.1%}".format, f"打乱 {n_paths} 次的均值": "{:+.1%}".format,
        "打乱后跑赢买入持有的比例": "{:.0%}".format,
    }))
    print(f"""
    🔍 怎么读：
       打乱收益后，历史走势里不再有任何可利用的规律。
       - 正确版本在打乱数据上的优势应该消失（只持有部分时间，收益通常低于买入持有）
       - 如果一个策略在打乱数据上仍然大幅跑赢，几乎可以断定代码有前视偏差
    ⏱️ 这个实验本身就是向量化的成果：{n_paths} 条路径一次性算完，耗时 {elapsed * 1000:.0f} ms
    """)
    return {"right": s_right, "wrong": s_wrong, "hold": float(s_hold.mean())}


# ==================== 第七部分：可视化 ====================


def plot_shuffle(result: dict):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(min(result["right"].min(), result["wrong"].min()), result["wrong"].max(), 50)
    ax.hist(result["right"], bins=bins, alpha=0.75, color="#2a78d6", label="正确：signal.shift(1) × ret")
    ax.hist(result["wrong"], bins=bins, alpha=0.75, color="#eb6834", label="错误：signal × ret（前视）")
    ax.axvline(result["hold"], color="#52514e", lw=1.5, label="买入持有（打乱后均值）")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("打乱收益后的策略年化收益分布")
    ax.set_xlabel("年化收益（算术口径）")
    ax.set_ylabel("路径数")
    ax.legend()
    path = save_figure(fig, "10_shuffle_lookahead_test.png", subdir="phase-2")
    print(f"\n🖼️ 图片已保存：{path}")


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    df = ma_benchmark()
    rolling_slope(df)
    conditional_logic(df)
    panel_performance()
    path_dependence(df)
    result = fast_but_wrong()
    plot_shuffle(result)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 sliding_window_view 实现滚动标准差（ddof=1），与 rolling().std() 对拍并比较速度
    2. 给 trailing_stop_py 增加"止损后 5 天内不再入场"的冷静期，再用 Numba 加速
    3. 用 Wilder 递推公式手写 ATR：纯 Python、Numba、ewm 三个版本，对拍并计时
    4. 给第六部分的实验换一个带前视的信号：把 close.rolling(20, center=True).mean() 当均线，
       加上 shift(1)，观察打乱实验能否把它查出来
    """)


if __name__ == "__main__":
    main()
