"""
Plotly 交互式图表
=================

学习目标：
    1. 理解什么时候用 matplotlib（静态报告）、什么时候用 Plotly（探索与仪表板）
    2. 画出可缩放、可悬停的净值对比图，理解对数坐标的意义
    3. 用 make_subplots 搭建"K 线 + 均线 + 布林带 / 成交量 / MACD"三层联动仪表板
    4. 解决 K 线图的非交易日空白问题（rangebreaks 与 category 轴）
    5. 控制 HTML 体积，认识"图表会骗人"的常见方式

核心概念：
    - go.Candlestick / go.Scatter / go.Bar：K 线、折线、柱状图
    - make_subplots(shared_xaxes=True)：共享 X 轴，缩放一个，其他联动
    - rangebreaks / type="category"：跳过非交易日
    - write_html(include_plotlyjs=...)：生成可离线打开的 HTML

前置知识：
    - 06-technical-indicators.py、07-drawdown-analysis.py

环境要求：
    - pip install plotly pandas numpy akshare
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存读缓存，否则用合成数据）
    - 输出的 HTML 在 outputs/phase-2/ 下，用浏览器打开（plotly.min.js 在同一目录，不需要联网）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils import OUTPUT_DIR
from utils.data import get_index_daily, get_stock_daily_adj, is_offline

load_dotenv()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

START, END = "20180101", "20241231"
SYMBOL, NAME = "600519", "贵州茅台"
OUT_DIR = OUTPUT_DIR / "phase-2"

# 多序列按固定顺序取色（已做色觉障碍校验）；红/绿只用于 A 股涨跌语义
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
UP_COLOR, DOWN_COLOR = "#e34948", "#008300"
FONT = "PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif"


def save_html(fig, filename: str) -> Path:
    """
    保存交互图。include_plotlyjs="directory" 让所有 HTML 共用同目录下一份 plotly.min.js：
    既能离线打开，又不用在每个文件里重复嵌入约 4.8 MB 的 JS。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.write_html(path, include_plotlyjs="directory")
    print(f"🌐 已保存：{path}（{path.stat().st_size / 1024:.0f} KB）")
    return path


def load_ohlc() -> pd.DataFrame:
    """后复权 OHLC：与 06 课一致，指标与收益用同一口径"""
    raw = get_stock_daily_adj(SYMBOL, START, END)
    df = raw[["open", "high", "low", "close"]].mul(raw["adj_factor"], axis=0)
    df["volume"] = raw["volume"]
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在完整历史上算指标，之后再截取展示区间，避开预热期"""
    close = df["close"]
    out = df.copy()
    out["ma20"] = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    out["upper"], out["lower"] = out["ma20"] + 2 * std, out["ma20"] - 2 * std
    out["dif"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["dea"] = out["dif"].ewm(span=9, adjust=False).mean()
    out["hist"] = 2 * (out["dif"] - out["dea"])
    out["pct"] = close.pct_change()
    return out


# ==================== 第一部分：为什么需要交互式图表 ====================


def why_interactive():
    print("=" * 60)
    print("第一部分：为什么需要交互式图表")
    print("=" * 60)
    print("""
    ┌──────────────┬──────────────────────────┬──────────────────────────┐
    │ 维度         │ matplotlib               │ Plotly                   │
    ├──────────────┼──────────────────────────┼──────────────────────────┤
    │ 输出         │ PNG / PDF（静态）        │ HTML（可交互）           │
    │ 操作         │ 看                       │ 缩放、悬停、开关图例     │
    │ 适合         │ 研究报告、论文           │ 探索分析、仪表板         │
    │ 数据量       │ 大数据量也只是一张图     │ 数据全部进浏览器，要控量 │
    └──────────────┴──────────────────────────┴──────────────────────────┘

    打开 HTML 后可以试试：
      - 拖动框选 → 放大；双击 → 复原
      - 单击图例 → 隐藏该序列；双击图例 → 只看该序列
      - 悬停 → 查看当天数值
    """)


# ==================== 第二部分：净值对比与对数坐标 ====================


def nav_chart():
    print("\n" + "=" * 60)
    print("第二部分：净值对比与对数坐标")
    print("=" * 60)

    closes = pd.DataFrame({
        "贵州茅台": get_stock_daily_adj("600519", START, END)["adj_close"],
        "中国神华": get_stock_daily_adj("601088", START, END)["adj_close"],
        "沪深300": get_index_daily("000300", START, END)["close"],
    }).dropna()
    nav = closes / closes.iloc[0]

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, subplot_titles=["线性坐标", "对数坐标"])
    for i, name in enumerate(nav.columns):
        for col in (1, 2):
            fig.add_trace(
                go.Scatter(
                    x=nav.index, y=nav[name], name=name, legendgroup=name, showlegend=col == 1,
                    line=dict(width=2, color=SERIES_COLORS[i]),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + name + " 净值 %{y:.3f}<extra></extra>",
                ),
                row=1, col=col,
            )
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(title="净值对比（起点 = 1）", template="plotly_white", height=500, font=dict(family=FONT))
    save_html(fig, "09_nav_compare.html")

    # 用数字解释对数坐标：同样的涨幅，线性坐标上的高度差完全不同
    first = nav.iloc[:, 0]
    lo, hi = first.min(), first.max()
    print(f"""
    📐 对数坐标下，等距离 = 等百分比变化：
       {nav.columns[0]} 净值从 1 → 2 和从 {hi / 2:.2f} → {hi:.2f} 都是翻倍：
       线性坐标上的高度分别是 1.00 和 {hi / 2:.2f}，对数坐标上完全相同
       （区间净值范围 {lo:.2f} ~ {hi:.2f}）
    💡 长周期净值应该用对数坐标，否则早期的波动被压扁，看起来"后期涨得特别猛"
    """)


# ==================== 第三部分：三层联动仪表板 ====================


def build_dashboard(d: pd.DataFrame, title: str, x=None):
    """K 线 + MA20 + 布林带 / 成交量 / MACD；x 不传时用日期轴"""
    x = d.index if x is None else x
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=["K 线（后复权）", "成交量（股）", "MACD"],
    )

    # 布林带先画，让 K 线压在上面；fill="tonexty" 填充上下轨之间
    fig.add_trace(go.Scatter(x=x, y=d["upper"], name="布林上轨", line=dict(width=1, color="#9aa0a6")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=d["lower"], name="布林下轨", line=dict(width=1, color="#9aa0a6"),
                   fill="tonexty", fillcolor="rgba(154,160,166,0.12)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name=NAME,
            # A 股习惯红涨绿跌；阳线空心、阴线实心，色弱读者靠形状也能分辨涨跌
            increasing=dict(line=dict(color=UP_COLOR, width=1), fillcolor="white"),
            decreasing=dict(line=dict(color=DOWN_COLOR, width=1), fillcolor=DOWN_COLOR),
            hovertext=[f"涨跌幅 {v:+.2%}" for v in d["pct"]],
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=x, y=d["ma20"], name="MA20", line=dict(width=2, color=SERIES_COLORS[0])), row=1, col=1)

    up = d["close"] >= d["open"]
    fig.add_trace(
        go.Bar(x=x, y=d["volume"], name="成交量", marker_color=np.where(up, UP_COLOR, DOWN_COLOR), showlegend=False),
        row=2, col=1,
    )

    hist_color = np.where(d["hist"] >= 0, UP_COLOR, DOWN_COLOR)
    fig.add_trace(go.Bar(x=x, y=d["hist"], name="MACD 柱", marker_color=hist_color, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["dif"], name="DIF", line=dict(width=2, color=SERIES_COLORS[0])), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["dea"], name="DEA", line=dict(width=2, color=SERIES_COLORS[1])), row=3, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(title=title, template="plotly_white", height=850, hovermode="x unified",
                      font=dict(family=FONT), legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
    return fig


def dashboard(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：三层联动仪表板")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────────┐
    │  K 线 (Candlestick) + MA20 + 布林带      高度 60%    │
    ├──────────────────────────────────────────────────────┤
    │  成交量 (Bar)                            高度 20%    │
    ├──────────────────────────────────────────────────────┤
    │  MACD                                    高度 20%    │
    └──────────────────────────────────────────────────────┘
              ↑ 共享 X 轴：缩放一个，其他联动
    """)

    full = add_indicators(df)
    d = full.tail(250)  # 指标在完整历史上算好再截取，展示区间内没有预热期的 NaN
    assert d[["ma20", "upper", "dif"]].notna().all().all()
    print(f"📊 展示区间：{d.index[0].date()} ~ {d.index[-1].date()}，{len(d)} 根 K 线")

    x_cat = d.index.strftime("%Y-%m-%d")  # category 轴：只显示有数据的日期，A 股最省事
    fig = build_dashboard(d, f"{SYMBOL} {NAME} 技术分析仪表板", x=x_cat)
    fig.update_xaxes(type="category", nticks=12)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    save_html(fig, "09_stock_dashboard.html")
    print("""
    💡 这个 build_dashboard 函数就是本阶段实战项目（股票分析仪表板）的雏形。
       颜色约定：A 股红涨绿跌，海外绿涨红跌，给不同受众的报告要注意。
       柱状成交量未做复权：送转股后股数会跳变，长周期比较时用成交额或换手率更稳妥。
    """)
    return d


# ==================== 第四部分：非交易日空白 ====================


def trading_day_gaps(d: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：非交易日空白")
    print("=" * 60)

    all_bdays = pd.bdate_range(d.index[0], d.index[-1])
    holidays = all_bdays.difference(d.index)  # 工作日但没有行情 = 节假日（或全市场休市）
    calendar_days = (d.index[-1] - d.index[0]).days + 1
    print(f"""
    用日期作 X 轴时，Plotly 按真实时间排布，周末和节假日会留出空白：
       展示区间跨 {calendar_days} 个自然日，只有 {len(d)} 个交易日
       其中工作日 {len(all_bdays)} 天，节假日休市 {len(holidays)} 天
    """)
    if len(holidays) == 0:
        print("    （离线合成数据按工作日生成，没有节假日，所以只需跳过周末）\n")

    fig = make_subplots(
        rows=3, cols=1, vertical_spacing=0.09,
        subplot_titles=["不处理：周末和节假日都有空白", "只跳过周末：节假日仍有空白", "跳过周末 + 节假日列表：连续"],
    )
    for row in (1, 2, 3):
        fig.add_trace(
            go.Candlestick(
                x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], showlegend=False,
                increasing=dict(line=dict(color=UP_COLOR, width=1), fillcolor="white"),
                decreasing=dict(line=dict(color=DOWN_COLOR, width=1), fillcolor=DOWN_COLOR),
            ),
            row=row, col=1,
        )
    weekend = dict(bounds=["sat", "mon"])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[weekend], row=2, col=1)
    fig.update_xaxes(rangebreaks=[weekend, dict(values=holidays.strftime("%Y-%m-%d").tolist())], row=3, col=1)
    fig.update_layout(title="非交易日空白的三种处理", template="plotly_white", height=900, font=dict(family=FONT))
    save_html(fig, "09_gap_compare.html")
    print("""
    ┌────────────────────────────┬───────────────────────────────────────┐
    │ 方法                       │ 特点                                  │
    ├────────────────────────────┼───────────────────────────────────────┤
    │ rangebreaks(bounds=周末)   │ 保留日期轴；节假日要额外给列表        │
    │ type="category"            │ 最省事；但轴上的距离不再代表时间长度  │
    └────────────────────────────┴───────────────────────────────────────┘
    """)


# ==================== 第五部分：控制数据量 ====================


def data_volume():
    print("\n" + "=" * 60)
    print("第五部分：控制数据量")
    print("=" * 60)

    rng = np.random.default_rng(9)
    rows = []
    for n in [1_700, 50_000, 500_000]:
        idx = pd.date_range("2015-01-05 09:31", periods=n, freq="min")
        y = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
        size = len(go.Figure(go.Scatter(x=idx, y=y)).to_json())
        rows.append([f"{n:,}", size / 1024 / 1024])
    df = pd.DataFrame(rows, columns=["数据点数", "图表 JSON 体积(MB)"])
    print("📦 一条折线的数据进入浏览器时的体积：")
    print(df.to_string(index=False, formatters={"图表 JSON 体积(MB)": "{:.2f}".format}))

    js = OUT_DIR / "plotly.min.js"
    print(f"""
    🌐 write_html 的 include_plotlyjs 参数：
       True         每个 HTML 嵌入一份 JS（本机这份 plotly.min.js 为 {js.stat().st_size / 1024 / 1024:.1f} MB），单文件可分发
       "cdn"        HTML 很小，但打开时需要联网
       "directory"  同目录共用一份 JS，离线可用（本课采用）

    💡 几十万根分钟 K 线直接塞进浏览器会卡死：先 resample，或只画最近一段；
       折线点数很多时可以用 go.Scattergl（WebGL 渲染）
    """)


# ==================== 第六部分：图表会骗人 ====================


def charts_can_lie(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：图表会骗人")
    print("=" * 60)

    close = df["close"]
    ends = close.index[-1]
    # 起点选在区间内最低点和最高点，展示"挑起点"的威力
    candidates = {
        "数据起点": close.index[0],
        "区间最低点": close.idxmin(),
        "区间最高点": close.idxmax(),
    }
    rows = []
    for label, day in candidates.items():
        if day >= ends:
            continue
        rows.append([label, day.date(), close.iloc[-1] / close.loc[day] - 1])
    tbl = pd.DataFrame(rows, columns=["起点选择", "起点日期", f"到 {ends.date()} 的收益"])
    print(f"📊 同一只股票、同一个终点，只改起点（{NAME}）：")
    print(tbl.to_string(index=False, formatters={tbl.columns[2]: "{:+.1%}".format}))

    print("""
    ⚠️ 常见的"图表话术"：
       1. 挑起止日期：同一段行情可以讲出完全相反的故事（见上表）
       2. 截断 Y 轴：不从 0 开始的柱状图会放大差异
       3. 线性坐标画长周期净值：让近期波动显得惊心动魄
       4. 把两个量纲不同的序列画在双 Y 轴上：两轴的比例是任意选的，
          "相关性"可以被画出来 → 改用归一化到同一起点，或者拆成两张图
    ✅ 研究报告要固定口径：统一起止日期、统一坐标类型、注明数据来源与复权方式
    """)


# ==================== 主函数 ====================


def main():
    if is_offline():
        print("⚠️ 离线模式：有本地缓存的数据读缓存，其余用合成数据（合成数值不代表真实市场）\n")
    if go is None:
        print("⚠️ 未安装 plotly，无法生成交互图表。请运行 pip install plotly 后重试")
        return

    why_interactive()
    nav_chart()
    df = load_ohlc()
    d = dashboard(df)
    trading_day_gaps(d)
    data_volume()
    charts_can_lie(df)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给仪表板增加第四层 RSI(14)，在 70 和 30 处用 fig.add_hline 画参考线
    2. 用 07 课的 top_drawdowns 找出最大回撤区间，用 fig.add_vrect 在净值图上标出来
    3. 把 build_dashboard 改成接受股票代码的函数，加上与沪深 300 的归一化对比图，
       这就是实战项目"股票分析仪表板"的主体
    4. 用 plotly 的 updatemenus 或 rangeselector 加上 1M / 6M / 1Y / 全部 的时间范围按钮
    """)


if __name__ == "__main__":
    main()
