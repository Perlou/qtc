# Phase 2: 金融时间序列与 pandas - 核心概念深度解析

> 本文是 Phase 2 的配套概念文档。建议先通读一遍建立直觉，再逐个运行 `01-*.py` ~ `10-*.py`。
> 读者假设：编程能力强，但对金融和数学不熟。每个概念按「直觉/比喻 → 公式 → 代码 → 量化中的意义与陷阱」展开。

---

## 目录

1. [引言：为什么金融时间序列"不一样"](#1-引言为什么金融时间序列不一样)
2. [收益率：简单收益 vs 对数收益](#2-收益率简单收益-vs-对数收益)
3. [年化：收益与波动率的时间尺度换算](#3-年化收益与波动率的时间尺度换算)
4. [滚动窗口统计](#4-滚动窗口统计)
5. [重采样与多频率数据](#5-重采样与多频率数据)
6. [多资产面板数据：宽表与长表](#6-多资产面板数据宽表与长表)
7. [技术指标：MA、EMA、MACD、RSI、ATR、布林带](#7-技术指标maemamacdrsiatr布林带)
8. [回撤与水下曲线](#8-回撤与水下曲线)
9. [基准、超额收益与 Beta](#9-基准超额收益与-beta)
10. [Plotly 交互式图表](#10-plotly-交互式图表)
11. [向量化与性能优化](#11-向量化与性能优化)
12. [最佳实践与常见陷阱](#12-最佳实践与常见陷阱)
13. [总结](#13-总结)

---

## 1. 引言：为什么金融时间序列"不一样"

### 1.1 你熟悉的时间序列 vs 金融时间序列

作为工程师，你可能处理过服务器监控指标、日志 QPS、IoT 传感器数据。金融时间序列看起来很像，但有几个关键差异：

| 维度 | 监控指标（如 QPS） | 金融价格序列 |
| --- | --- | --- |
| **时间轴** | 连续、均匀采样 | 只在**交易日**有数据，节假日、周末、停牌都会"缺席" |
| **关注对象** | 数值本身 | **变化率**（收益）比价格本身重要得多 |
| **平稳性** | 常在某个水平附近波动 | 价格不平稳（会"漂走"），收益近似平稳 |
| **信噪比** | 较高 | **极低**：日收益中"可预测部分"通常远小于噪声 |
| **数据修订** | 极少 | 除权除息导致历史价格需要**复权**，财报数据会修订 |
| **因果约束** | 事后分析无所谓 | 任何计算都必须确保**只用到当时可得的信息** |

最后一条最重要，贯穿整个课程：**写金融时间序列代码时，每一行都要问自己"在 t 时刻我真的知道这个值吗？"** 这个问题在 Phase 4 会被系统化为"前视偏差（Look-ahead Bias）"。

### 1.2 本阶段在课程中的位置

```
┌──────────────────────────────────────────────────────────────────┐
│                   Phase 1 → Phase 2 → Phase 3/4                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: 数据获取与清洗          Phase 2: 数据变成"度量"          │
│  ┌──────────────────┐            ┌──────────────────────────┐     │
│  │ OHLCV / 复权     │            │ 收益、波动、回撤、Beta    │     │
│  │ 交易日历 / 停牌  │ ─────────▶ │ 滚动统计、重采样、面板    │     │
│  │ Parquet/DuckDB   │            │ 技术指标、向量化          │     │
│  └──────────────────┘            └────────────┬─────────────┘     │
│                                               │                   │
│                       ┌───────────────────────┴──────────┐        │
│                       ▼                                  ▼        │
│            Phase 3: 统计推断                  Phase 4: 回测         │
│            （这些度量靠谱吗？）               （这些度量能赚钱吗？）│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

本阶段的产出是一套**可靠的度量工具箱**。后面所有阶段的策略评估、因子分析、风险控制都建立在它之上——这里错一个 `shift`，后面全盘皆错。

### 1.3 本文使用的示例数据约定

```python
import numpy as np
import pandas as pd

# 假设 df 是贵州茅台（600519）的日线数据，已做后复权，index 为交易日
# 列：open, high, low, close, volume
# 数据获取见 Phase 1（AKShare / Tushare）
close = df["close"]
```

> 💡 **复权提醒**：计算收益率要用**复权价**，否则除息日会出现一个虚假的"下跌"。但判断涨跌停、计算可买股数等需要**真实成交价（不复权）**。这个区别在 Phase 4 会反复出现。

---

## 2. 收益率：简单收益 vs 对数收益

### 2.1 直觉

假设你 100 元买入茅台，第二天涨到 110 元，第三天跌回 100 元。

- 简单收益：+10%，然后 -9.09%。两个数字加起来不是 0，但你确实回到了原点。
- 对数收益：+9.53%，然后 -9.53%。加起来正好是 0。

**比喻**：简单收益像"打折"——先涨 10% 再跌 10% 并不回到原价（商家的"先提价再打折"套路）；对数收益像"海拔高度变化"——上山 100 米再下山 100 米，正好回到原地，可以直接相加。

### 2.2 公式

**简单收益（Simple Return / Arithmetic Return）**：

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

- $P_t$：第 $t$ 日的（复权）收盘价
- $R_t$：第 $t$ 日的简单收益率

**对数收益（Log Return / Continuously Compounded Return）**：

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(1 + R_t)
$$

- $\ln$：自然对数
- $r_t$：第 $t$ 日的对数收益率

**多期累计**：

$$
1 + R_{0 \to T} = \prod_{t=1}^{T} (1 + R_t), \qquad r_{0 \to T} = \sum_{t=1}^{T} r_t
$$

简单收益多期要**连乘**，对数收益多期直接**相加**——这就是对数收益在时间维度上的优势。

**多资产组合**：组合在某一天的简单收益等于各资产简单收益的加权和：

$$
R_{p,t} = \sum_{i=1}^{N} w_i R_{i,t}
$$

- $w_i$：资产 $i$ 在期初的权重，$\sum w_i = 1$
- $N$：资产数量

但对数收益**没有**这个性质：$r_{p,t} \neq \sum_i w_i r_{i,t}$。

### 2.3 两者对照

| 性质 | 简单收益 $R$ | 对数收益 $r$ |
| --- | --- | --- |
| 时间上累加 | ❌ 需要连乘 | ✅ 直接求和 |
| 截面（资产间）加权 | ✅ 组合收益 = 加权和 | ❌ 不成立 |
| 下界 | $-100\%$ | $-\infty$ |
| 对称性 | 涨 10% 跌 10% 不回原点 | +x 和 -x 正好抵消 |
| 小收益时 | $R \approx r$（日频差异通常很小） | 同左 |
| 典型用途 | 组合收益、回测净值、绩效报告 | 统计建模、多期累加、波动率估计 |

**经验法则**：

- **横向（多资产组合）用简单收益**
- **纵向（单资产多期、建模）用对数收益**
- 对外报告收益时一般用简单收益（投资者理解的"赚了百分之几"）

### 2.4 代码

```python
# 简单收益
ret = close.pct_change()

# 对数收益
log_ret = np.log(close / close.shift(1))
# 等价：np.log1p(ret)

# 净值曲线（从 1 开始）
nav = (1 + ret.fillna(0)).cumprod()

# 用对数收益还原净值：exp(累计和)
nav2 = np.exp(log_ret.fillna(0).cumsum())
assert np.allclose(nav, nav2)

# 区间总收益
total_return = close.iloc[-1] / close.iloc[0] - 1
```

### 2.5 波动拖累（Volatility Drag）：为什么算术平均会"骗人"

一个策略日收益交替为 +10% 和 -10%，算术平均是 0%，但两天后净值是 $1.1 \times 0.9 = 0.99$，亏了 1%。

近似关系（$\sigma$ 不太大时）：

$$
\bar{g} \approx \bar{R} - \frac{\sigma^2}{2}
$$

- $\bar{g}$：几何平均收益（真实的复利增长率）
- $\bar{R}$：简单收益的算术平均
- $\sigma^2$：收益的方差

**量化意义**：

- 两个算术平均收益相同的策略，波动大的那个长期复利更低。这就是"波动是收益的敌人"的数学含义。
- 杠杆 ETF 长期持有会损耗，原因之一正是波动拖累。
- 回测报告中"年化收益"应该用几何方式计算（见第 3 节），而不是日均收益 × 252。

### 2.6 陷阱

```python
# ❌ 陷阱1：在未复权价格上算收益，除息日出现假下跌
ret_wrong = raw_close.pct_change()

# ❌ 陷阱2：长表里直接 pct_change，第二只股票的第一天和第一只股票最后一天相减
long_df["ret"] = long_df["close"].pct_change()          # 错！
long_df["ret"] = long_df.groupby("code")["close"].pct_change()  # 对

# ❌ 陷阱3：停牌日被前向填充后，复牌日的收益是"停牌期间累计收益"
#    这本身没错，但如果你的策略假设停牌期间能交易，就错了（见 Phase 4）

# ❌ 陷阱4：对数收益加权求组合收益
port_log_ret_wrong = (log_rets * weights).sum(axis=1)  # 近似，不精确
port_ret = (simple_rets * weights).sum(axis=1)          # 正确（当日权重）
```

---

## 3. 年化：收益与波动率的时间尺度换算

### 3.1 直觉

日收益 0.05% 好不好？日波动 1.5% 大不大？这些数字没有直观感受。年化就是把不同频率、不同长度的数据**换算到同一把尺子（一年）**上，方便比较。

**比喻**：就像把"每秒米数"换成"每小时公里数"——单位统一了才能比较。

### 3.2 每年多少个周期？

| 数据频率 | 每年周期数 $N$（常用值） | 说明 |
| --- | --- | --- |
| A 股日频 | 约 240–245 | A 股每年交易日一般在这个范围，具体看当年日历；很多代码直接用 252（沿用美股惯例）或 244/250，**关键是全项目统一** |
| 周频 | 52 | |
| 月频 | 12 | |
| 期货日频 | 同 A 股交易日 | 夜盘不额外计为一天 |
| 加密货币日频 | 365 | 7×24 小时交易，没有休市 |
| 加密货币小时频 | 365 × 24 = 8760 | |

> 本文代码统一用 `N = 252`，你也可以改成 244。请在项目里定义成常量，而不是到处写魔法数字。

### 3.3 年化收益

**几何年化（CAGR，Compound Annual Growth Rate）——推荐**：

$$
R_{\text{ann}} = \left(\frac{V_T}{V_0}\right)^{N / n} - 1
$$

- $V_0, V_T$：期初、期末净值
- $n$：样本中的周期数（如交易日数）
- $N$：每年周期数

**算术年化**（简单乘法，常用于 Sharpe Ratio 分子）：

$$
\mu_{\text{ann}} = \bar{R} \times N
$$

- $\bar{R}$：周期收益的算术平均

两者差异就是上一节的波动拖累。波动越大，差得越多。

### 3.4 年化波动率与 $\sqrt{T}$ 法则

$$
\sigma_{\text{ann}} = \sigma_{\text{period}} \times \sqrt{N}
$$

- $\sigma_{\text{period}}$：周期收益的标准差（如日收益标准差）

**为什么是 $\sqrt{N}$ 而不是 $N$？** 如果每日收益相互独立、方差都是 $\sigma^2$，那么 $N$ 天收益之和的方差是 $N\sigma^2$（独立变量方差可加），标准差就是 $\sigma\sqrt{N}$。

**比喻**：醉汉走路。每一步随机向左或向右，走 100 步后离起点的典型距离不是 100 步，而是约 $\sqrt{100} = 10$ 步——因为很多步相互抵消了。

### 3.5 代码

```python
N = 252  # 每年交易日数，项目内统一

def annualize_return(ret: pd.Series, periods_per_year: int = N) -> float:
    """几何年化收益（CAGR）"""
    ret = ret.dropna()
    total = (1 + ret).prod()
    return total ** (periods_per_year / len(ret)) - 1

def annualize_vol(ret: pd.Series, periods_per_year: int = N) -> float:
    """年化波动率"""
    return ret.dropna().std() * np.sqrt(periods_per_year)

daily = close.pct_change()
weekly = close.resample("W-FRI").last().pct_change()
monthly = close.resample("ME").last().pct_change()   # pandas >= 2.2 用 "ME"，旧版用 "M"

print(annualize_vol(daily, 252), annualize_vol(weekly, 52), annualize_vol(monthly, 12))
```

三个数字应该**大致相近但不完全相同**。如果差别很大，说明收益存在显著的自相关（见下文陷阱）。

### 3.6 量化意义与陷阱

- **$\sqrt{T}$ 法则假设收益独立同分布（i.i.d.）**。如果收益有正自相关（趋势），长周期波动会比 $\sigma\sqrt{N}$ 更大；负自相关（均值回归）则更小。对于对冲基金这类持有流动性差资产的产品，月度收益平滑会让 $\sqrt{12}$ 年化严重低估真实风险。
- **样本太短不要年化**。一个跑了 20 天、收益 15% 的策略，年化后是一个天文数字，毫无意义。
- **加密货币用 365**。用 252 会低估加密资产的年化波动。
- **日内数据年化**：例如 A 股 5 分钟线，每天 48 根（4 小时交易），每年约 $48 \times 252$ 根。注意隔夜跳空收益要不要算进去——这是一个容易被忽略的建模选择。

---

## 4. 滚动窗口统计

### 4.1 直觉

全样本的平均和标准差只给你一个数，但市场的状态是变化的：2015 年股灾前后、2020 年疫情期间，茅台的波动率完全不同。**滚动窗口（Rolling Window）**就是拿一个固定长度的"滑动镜头"沿时间轴移动，每个时点只看过去 $w$ 天。

```
时间 ──────────────────────────────────────────────────▶
        [──── w=20 ────]                 t=20 的统计量
           [──── w=20 ────]              t=21 的统计量
              [──── w=20 ────]           t=22 的统计量
```

### 4.2 三种窗口

| 类型 | pandas | 特点 | 适用 |
| --- | --- | --- | --- |
| **滚动窗口** | `rolling(w)` | 只看最近 $w$ 个点，等权 | 移动平均、滚动波动率 |
| **扩展窗口** | `expanding()` | 从起点到当前所有点 | 累计统计、"只用历史"的标准化 |
| **指数加权** | `ewm(span=..)` / `ewm(halflife=..)` | 越近权重越大，平滑衰减 | EMA、EWMA 波动率（RiskMetrics） |

**指数加权移动平均（EWMA）**：

$$
\text{EMA}_t = \alpha x_t + (1 - \alpha)\,\text{EMA}_{t-1}
$$

- $x_t$：当前观测值
- $\alpha \in (0,1]$：平滑系数；`span=s` 时 $\alpha = 2/(s+1)$；`halflife=h` 时 $\alpha = 1 - e^{-\ln 2 / h}$

**半衰期（Half-life）** 的直觉：$h$ 期之前的观测，权重只剩当前观测的一半。

### 4.3 代码

```python
ret = close.pct_change()

# 20 日滚动年化波动率
vol_20 = ret.rolling(20).std() * np.sqrt(252)

# min_periods：允许窗口不满时也输出（慎用，早期值噪声很大）
vol_20_early = ret.rolling(20, min_periods=10).std() * np.sqrt(252)

# EWMA 波动率（半衰期 20 天）
ewm_vol = ret.ewm(halflife=20).std() * np.sqrt(252)

# 滚动 Z-score：当前价格偏离 60 日均值多少个标准差
z = (close - close.rolling(60).mean()) / close.rolling(60).std()

# 扩展窗口标准化（无前视）
z_expanding = (ret - ret.expanding(min_periods=60).mean()) / ret.expanding(min_periods=60).std()

# 滚动相关：茅台与沪深 300（hs300_ret 见第 9 节）
roll_corr = ret.rolling(60).corr(hs300_ret)
```

### 4.4 窗口长度的权衡

| 窗口 | 优点 | 缺点 |
| --- | --- | --- |
| 短（5–20 日） | 反应快，捕捉状态切换 | 噪声大，估计误差高 |
| 长（120–250 日） | 平稳、估计误差小 | 滞后严重，状态变了还在用旧数据 |

**没有"正确"的窗口长度**，它本质上是一个参数。在 Phase 4 你会看到：对窗口长度做大量尝试并挑最好的，就是过拟合的典型来源。

### 4.5 陷阱

```python
# ❌ 致命陷阱：center=True 用到了未来数据
ma_wrong = close.rolling(20, center=True).mean()   # t 时刻用了 t+10 的数据！

# ❌ 全样本标准化 = 前视偏差
z_wrong = (ret - ret.mean()) / ret.std()            # ret.mean() 包含未来

# ❌ 默认 rolling(20).std() 是样本标准差 (ddof=1)，
#    而很多软件的布林带用总体标准差 (ddof=0)，结果会有细微差别
std_pop = close.rolling(20).std(ddof=0)

# ⚠️ 预热期（warm-up）：前 w-1 个值是 NaN。
#    回测起点要从所有指标都"热身完毕"的日期开始，否则不同参数的策略起点不同、不可比
```

---

## 5. 重采样与多频率数据

### 5.1 直觉

**重采样（Resample）** 就是改变数据的时间粒度：

- **降采样（Downsampling）**：日线 → 周线/月线，多行聚合成一行（类似 SQL 的 `GROUP BY 周`）
- **升采样（Upsampling）**：月度财务数据 → 日频，一行展开成多行，需要填充

### 5.2 OHLCV 的正确聚合方式

K 线的每一列有不同的聚合语义，不能一股脑求平均：

| 字段 | 聚合方式 | 原因 |
| --- | --- | --- |
| open | `first` | 周线的开盘价 = 本周第一个交易日开盘价 |
| high | `max` | 本周最高 |
| low | `min` | 本周最低 |
| close | `last` | 本周最后一个交易日收盘价 |
| volume / amount | `sum` | 成交量可加 |

```python
weekly = df.resample("W-FRI").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}).dropna(subset=["close"])   # 整周休市（如春节、国庆）会产生空行
```

### 5.3 标签与对齐：一个隐蔽的坑

`resample("ME")` 的标签是**日历月末**（如 2024-03-31），但 A 股 3 月最后一个交易日可能是 3 月 29 日。如果你把月度信号按标签日期合并回日线：

```
月度信号标签：2024-03-31（周日）
日线数据中没有这一天 → merge 后信号丢失或被错误地对到 4 月
```

**解决思路**：

```python
# 方法1：用实际最后一个交易日作为标签
month_end_dates = df.groupby(df.index.to_period("M")).apply(lambda g: g.index[-1])
monthly_close = df.loc[month_end_dates.values, "close"]

# 方法2：合并回日线时显式 shift，确保信号在"下一个交易日"才生效
monthly_signal = monthly_signal.reindex(df.index, method="ffill").shift(1)
```

### 5.4 升采样与"信息可得时间"

把低频数据扩展到日频时，必须用 **"数据公布日"** 而不是 **"数据所属期"**：

```
茅台 2023 年年报：
  报告期（所属期）：2023-12-31
  公告日：通常在次年 3–4 月
  
❌ 错误：按 2023-12-31 对齐 → 1 月份的策略就"知道"了年报数据
✅ 正确：按公告日对齐，然后 ffill
```

```python
# fundamentals: index=公告日, columns=[roe, ...]
fund_daily = fundamentals.reindex(df.index, method="ffill")
```

这是 Phase 7 多因子研究中最常见的前视偏差来源之一，现在就要养成习惯。

### 5.5 量化意义

- **多周期共振**：用周线判断趋势、日线找入场点，需要把周线信号对齐回日线——注意周线收盘是周五，周五盘中的日线信号不能用"本周周线"。
- **降低噪声**：月频收益的信噪比通常高于日频，适合做低频资产配置研究。
- **期货夜盘**：期货夜盘（如 21:00 开始）在交易所口径上归属**下一个交易日**。按自然日 resample 会把夜盘和日盘切错，需要按交易日字段分组。
- **加密货币**：24 小时交易，"日线"的切分时点（UTC 0 点还是北京时间 0 点）因数据源而异，要统一。

---

## 6. 多资产面板数据：宽表与长表

### 6.1 两种形态

**比喻**：宽表像 Excel 透视表（行是日期、列是股票），适合"人看"和矩阵运算；长表像数据库的事实表（每行一条记录），适合存储和分组运算。

```
宽表（Wide）                              长表（Long / Tidy）
             600519   000858   000001     date        code    close
2024-01-02   1685.0   140.2    9.39       2024-01-02  600519  1685.0
2024-01-03   1680.1   139.5    9.20       2024-01-02  000858  140.2
...                                       2024-01-02  000001  9.39
                                          2024-01-03  600519  1680.1
                                          ...
```

| 维度 | 宽表 | 长表 |
| --- | --- | --- |
| 结构 | index=日期，columns=代码 | 每行 (日期, 代码, 字段...) |
| 适合 | 向量化计算、截面运算、矩阵乘法 | 存储（Parquet/DuckDB）、多字段、groupby |
| 多字段 | 每个字段一张表，或 MultiIndex 列 | 天然支持 |
| 缺失值 | 显式为 NaN（停牌、未上市） | 缺失表现为"没有这一行" |

### 6.2 互相转换

```python
# 长 → 宽
wide_close = long_df.pivot(index="date", columns="code", values="close")

# 宽 → 长
long_back = wide_close.reset_index().melt(id_vars="date", var_name="code", value_name="close")

# MultiIndex 方式
panel = long_df.set_index(["date", "code"]).sort_index()
wide = panel["close"].unstack("code")     # 长 → 宽
long = wide.stack().rename("close")       # 宽 → 长（会丢掉 NaN）
```

### 6.3 两类运算：时间序列 vs 截面

```
              时间序列运算（沿 axis=0，对每只股票独立）
              ──────────────────────────────────────▶
              ┃ 600519  000858  000001
  截面运算    ┃  ...
  （沿 axis=1，┃  ...
  对每一天）  ▼
```

```python
rets = wide_close.pct_change()

# 时间序列：每只股票的 20 日动量
mom_20 = wide_close / wide_close.shift(20) - 1

# 截面：每天对所有股票的动量做百分位排名（Phase 7 因子研究的基础操作）
mom_rank = mom_20.rank(axis=1, pct=True)

# 截面去均值（相对强弱）
mom_demean = mom_20.sub(mom_20.mean(axis=1), axis=0)

# 等权组合日收益（NaN 自动跳过 = 停牌股不参与当天平均）
ew_ret = rets.mean(axis=1)

# 长表上的分组时间序列运算
long_df["ret"] = long_df.groupby("code")["close"].pct_change()
long_df["ma20"] = long_df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean())
```

### 6.4 陷阱

- **停牌与 NaN**：宽表中停牌日是 NaN。`ffill()` 后价格不变、收益为 0，这对净值估值是合理的，但对"能否交易"必须另外标记（Phase 4）。
- **未上市 vs 停牌 vs 退市**：三者在宽表里都是 NaN，但含义完全不同。建议额外维护一张同形状的"可交易状态"布尔表。
- **幸存者偏差的种子**：如果你的股票列表是"今天的沪深 300 成分股"，宽表从一开始就排除了历史上被剔除和退市的股票（Phase 4 详述）。
- **`stack()` 默认丢弃 NaN**：转长表后"停牌"这一信息消失，如果下游需要它，要先 `fillna` 一个标记值或在新版 pandas 中注意 `stack` 的参数行为。

---

## 7. 技术指标：MA、EMA、MACD、RSI、ATR、布林带

> 本节重点是**用向量化方式正确实现**这些指标并理解它们在描述什么。它们能否赚钱是 Phase 5 的问题——剧透一下：单独使用通常不能稳定赚钱。

### 7.1 指标全景

| 指标 | 类型 | 描述的是什么 | 典型参数 |
| --- | --- | --- | --- |
| MA（简单移动平均） | 趋势 | 过去 $n$ 天平均价格 | 5/10/20/60 |
| EMA（指数移动平均） | 趋势 | 近期权重更大的平均价格 | 12/26 |
| MACD | 趋势/动量 | 快慢 EMA 之差及其平滑 | 12, 26, 9 |
| RSI | 动量/超买超卖 | 上涨力量占总力量的比例 | 14 |
| ATR | 波动 | 平均每天的真实波动幅度（价格单位） | 14 |
| 布林带 | 波动/均值回归 | 均线 ± k 倍标准差 | 20, 2 |

### 7.2 MA 与 EMA

$$
\text{MA}_t(n) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}
$$

- $P_{t-i}$：$i$ 天前的收盘价；$n$：窗口长度

EMA 见 4.2 节公式，`span=n` 对应 $\alpha = 2/(n+1)$。

```python
ma20 = close.rolling(20).mean()
ema12 = close.ewm(span=12, adjust=False).mean()
```

> `adjust=False` 使用递推公式 $\text{EMA}_t = \alpha P_t + (1-\alpha)\text{EMA}_{t-1}$，与大多数行情软件一致；`adjust=True`（pandas 默认）在序列开头用另一种归一化方式，早期数值会不同。

**直觉**：均线是"平滑后的价格"，代价是**滞后**。$n$ 越大越平滑、越滞后。EMA 对最新价格更敏感，滞后小一些。

### 7.3 MACD

$$
\text{DIF}_t = \text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t
$$

$$
\text{DEA}_t = \text{EMA}_{9}(\text{DIF})_t
$$

$$
\text{MACD 柱}_t = 2 \times (\text{DIF}_t - \text{DEA}_t)
$$

- DIF：快慢均线之差，>0 说明短期均价在长期均价之上
- DEA：DIF 的平滑（信号线）
- 柱状图：国内行情软件通常乘以 2，海外（如 TradingView）一般不乘——对比时注意

```python
def macd(close: pd.Series, fast=12, slow=26, signal=9):
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist
```

### 7.4 RSI（相对强弱指数）

**直觉**：过去 14 天里，"上涨的力量"占"总力量"的百分比。100 表示只涨不跌，0 表示只跌不涨。

$$
\text{RS}_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}
$$

- $\text{AvgGain}_t$：过去 $n$ 期上涨幅度的平均（下跌日记为 0）
- $\text{AvgLoss}_t$：过去 $n$ 期下跌幅度绝对值的平均（上涨日记为 0）
- Wilder 原始定义用的是 $\alpha = 1/n$ 的指数平滑

```python
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()   # Wilder 平滑
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)
```

> 用 `rolling(n).mean()` 代替 Wilder 平滑是另一种常见实现（Cutler's RSI），数值会不同。和第三方库对拍前先确认用的是哪种。

### 7.5 ATR（平均真实波幅）

**直觉**：一天"真实"波动了多少元。之所以叫"真实"，是因为要把**跳空缺口**也算进去——昨天收 100、今天开盘直接 110、日内 108–112 波动，真实波幅是 12 而不是 4。

$$
\text{TR}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)
$$

$$
\text{ATR}_t = \text{Wilder 平滑}(\text{TR}, n)
$$

- $H_t, L_t$：当日最高、最低价
- $C_{t-1}$：前一日收盘价

```python
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()
```

**量化意义**：ATR 是**价格单位**的波动度量，是海龟交易法则与 Phase 5 ATR 仓位管理的核心：波动大的品种少买、波动小的多买，让每个仓位承担相近的风险。

### 7.6 布林带（Bollinger Bands）

$$
\text{Mid}_t = \text{MA}_t(n), \quad \text{Upper}_t = \text{Mid}_t + k\,\sigma_t(n), \quad \text{Lower}_t = \text{Mid}_t - k\,\sigma_t(n)
$$

- $\sigma_t(n)$：过去 $n$ 日收盘价的标准差
- $k$：带宽倍数，常用 2

```python
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)   # 多数软件用总体标准差
    upper, lower = mid + k * std, mid - k * std
    pct_b = (close - lower) / (upper - lower)   # %B：价格在带内的相对位置
    return mid, upper, lower, pct_b
```

**常见误解**：「价格 95% 的时间在 ±2σ 带内」——这依赖正态分布和平稳性，而价格既不正态也不平稳（Phase 3），所以这只是一个粗略说法。

### 7.7 技术指标的陷阱

| 陷阱 | 说明 |
| --- | --- |
| **用前复权价算指标** | 前复权价会随着未来的分红送转"改写历史"，你今天看到的 2018 年 MA 和 2018 年当时看到的不一样。研究信号通常用后复权价或基于收益率构造 |
| **信号与成交在同一根 K 线** | 用 t 日收盘价算出信号，又假设以 t 日收盘价成交——Phase 4 的头号前视偏差 |
| **预热期不一致** | EMA 理论上依赖无限历史，前几十个值不稳定。不同起始日期算出的 EMA 不同 |
| **参数挖掘** | MA 窗口从 2 试到 200，挑出最好的——几乎一定过拟合 |
| **指标高度相关** | MA 交叉、MACD、动量本质上都在描述同一件事（趋势），堆砌多个并不能带来多少独立信息 |

---

## 8. 回撤与水下曲线

### 8.1 直觉

**回撤（Drawdown）** 回答的问题是：「如果我在最高点买入，现在亏了多少？」

**比喻**：登山。你的海拔（净值）不断变化，"回撤"是你当前位置距离**此前到过的最高峰**往下掉了多少。**水下曲线（Underwater Curve）** 就是把每天的回撤画出来——0 表示正在创新高（浮出水面），负值表示还在水下。

```
净值
 ▲          峰值 ●
 │         ╱      ╲                  ● 新高（回撤结束）
 │   ●   ╱          ╲              ╱
 │  ╱ ╲ ╱             ╲          ╱
 │ ╱   ●                ╲      ╱
 │╱                       ● 谷底
 └──────────────────────────────────────▶ 时间
          │← 下跌期 →│← 恢复期 →│
          │←─── 回撤持续期 ────→│
```

### 8.2 公式

$$
\text{Peak}_t = \max_{s \le t} V_s, \qquad \text{DD}_t = \frac{V_t}{\text{Peak}_t} - 1, \qquad \text{MDD} = \min_t \text{DD}_t
$$

- $V_t$：$t$ 时刻净值
- $\text{Peak}_t$：截至 $t$ 的历史最高净值（只看过去，天然无前视）
- $\text{DD}_t$：$t$ 时刻回撤（≤ 0）
- $\text{MDD}$：最大回撤（Maximum Drawdown）

### 8.3 代码：最大回撤及其起止日期

```python
def drawdown_analysis(nav: pd.Series) -> dict:
    peak = nav.cummax()
    dd = nav / peak - 1                     # 水下曲线

    trough_date = dd.idxmin()               # 最大回撤谷底
    peak_date = nav.loc[:trough_date].idxmax()   # 谷底之前的峰值
    # 恢复日：谷底之后首次回到峰值水平
    after = nav.loc[trough_date:]
    recovered = after[after >= nav.loc[peak_date]]
    recovery_date = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": dd.min(),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,          # None 表示至今未恢复
        "underwater": dd,
    }

res = drawdown_analysis(nav)
print(f"最大回撤 {res['max_drawdown']:.2%}，{res['peak_date']:%Y-%m-%d} → {res['trough_date']:%Y-%m-%d}")
```

**列出前 N 大回撤区间**（检查回撤是否集中在某一段时间）：

```python
def top_drawdowns(nav: pd.Series, top: int = 5) -> pd.DataFrame:
    dd = nav / nav.cummax() - 1
    is_under = dd < 0
    # 每次"从水面进入水下"开启一个新区间
    episode = (is_under & ~is_under.shift(1, fill_value=False)).cumsum()
    episode = episode.where(is_under)
    rows = []
    for _, seg in dd.groupby(episode):
        rows.append({
            "start": seg.index[0], "trough": seg.idxmin(),
            "end": seg.index[-1], "depth": seg.min(), "days": len(seg),
        })
    return pd.DataFrame(rows).nsmallest(top, "depth")
```

### 8.4 量化意义

- **回撤是投资者真正"感受到"的风险**。波动率对称地惩罚涨和跌，但没人会因为暴涨而赎回基金。
- **回撤恢复的不对称性**：亏 50% 需要涨 100% 才能回本。

$$
\text{回本所需涨幅} = \frac{1}{1 + \text{DD}} - 1
$$

| 回撤 | 回本所需涨幅 |
| --- | --- |
| -10% | +11.1% |
| -20% | +25.0% |
| -50% | +100% |
| -80% | +400% |

- **回撤持续时间**同样关键：一个最大回撤 20% 但花了 3 年才恢复的策略，在实际中很可能早就被叫停了。
- **MDD 是路径依赖的单一极值统计量**，样本越长，MDD 天然越大（有更多机会遇到坏事件）。所以不同长度的回测之间直接比 MDD 不公平，Phase 3 的蒙特卡洛可以给出 MDD 的分布而不是一个点。
- 茅台这样的长期大牛股，历史上也出现过超过 50% 的回撤（例如 2008 年前后）。"好公司"不等于"低回撤"。

---

## 9. 基准、超额收益与 Beta

### 9.1 直觉

茅台今年涨了 10%，好不好？如果同期沪深 300 涨了 25%，那其实是**跑输**了。**基准（Benchmark）** 就是比较的参照物。

**比喻**：船在河里顺流而下。船相对岸边的速度 = 水流速度（市场，Beta 部分）+ 船自己划的速度（Alpha 部分）。顺流时人人都快，量化研究要分清"是我划得好，还是水流得急"。

### 9.2 基准选择

| 策略类型 | 常见基准 |
| --- | --- |
| A 股大盘股 | 沪深 300（000300） |
| A 股中盘 / 小盘 | 中证 500（000905）/ 中证 1000（000852） |
| 全市场 | 中证全指、万得全 A 等 |
| 期货 CTA | 往往以绝对收益或无风险利率为基准 |
| 加密 | BTC |

> **价格指数 vs 全收益指数**：常见的沪深 300 指数是**价格指数**，不含分红再投资。你的个股用复权价（含分红），基准却用价格指数，会系统性高估超额收益。更公平的做法是使用**全收益指数**（中证指数公司发布了沪深 300 全收益指数等），或至少在报告中注明。

### 9.3 超额收益、跟踪误差与信息比率

$$
R^{e}_t = R_{p,t} - R_{b,t}
$$

$$
\text{TE} = \text{std}(R^{e}_t) \times \sqrt{N}, \qquad \text{IR} = \frac{\overline{R^{e}} \times N}{\text{TE}}
$$

- $R_{p,t}$：组合（或个股）收益；$R_{b,t}$：基准收益
- $R^{e}_t$：超额收益（Active Return）
- TE：跟踪误差（Tracking Error），超额收益的年化波动
- IR：信息比率（Information Ratio），单位跟踪误差的超额收益

**注意**：累计超额收益有两种算法，结果不同：

```python
# 方法1：每日超额收益累计（"日度再平衡"的多空组合）
excess_nav_1 = (1 + (ret - bench_ret)).cumprod()

# 方法2：净值相除（"买入持有"策略相对基准的比值）
excess_nav_2 = (1 + ret).cumprod() / (1 + bench_ret).cumprod()
```

两者在短期内接近，长期可能差别很大。报告中要说明用的是哪一种。

### 9.4 Beta 与 Alpha

用 **CAPM 回归** 把个股收益分解为"跟随市场的部分"和"独立部分"：

$$
R_{i,t} - R_{f} = \alpha + \beta\,(R_{m,t} - R_f) + \varepsilon_t
$$

- $R_{i,t}$：股票 $i$ 的收益；$R_{m,t}$：市场（基准）收益；$R_f$：无风险利率
- $\beta$：对市场的敏感度。$\beta = 1.2$ 表示市场涨 1%，该股平均涨 1.2%
- $\alpha$：截距，与市场无关的平均收益（真正的"本事"）
- $\varepsilon_t$：残差，个股特有的波动（特质风险）

Beta 的直接计算公式：

$$
\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}
$$

```python
# hs300: 沪深 300 指数日线收盘价
hs300_ret = hs300.pct_change()
data = pd.concat([ret, hs300_ret], axis=1, keys=["stock", "mkt"]).dropna()

# 方法1：协方差公式
beta = data["stock"].cov(data["mkt"]) / data["mkt"].var()

# 方法2：OLS 回归（同时得到 alpha 与 t 统计量，Phase 3 详述）
import statsmodels.api as sm
model = sm.OLS(data["stock"], sm.add_constant(data["mkt"])).fit()
alpha_daily, beta_ols = model.params
print(f"beta={beta_ols:.2f}, 年化 alpha≈{alpha_daily * 252:.2%}, R²={model.rsquared:.2f}")

# 滚动 Beta（120 日）
roll_beta = data["stock"].rolling(120).cov(data["mkt"]) / data["mkt"].rolling(120).var()
```

### 9.5 量化意义与陷阱

- **"超额收益"可能只是高 Beta**：牛市里高 Beta 股票天然跑赢指数。真正的 Alpha 要扣除 Beta 暴露后看。
- **Beta 不稳定**：滚动 Beta 会随时间显著变化。用全样本 Beta 做历史对冲，是前视偏差。
- **R² 很低是常态**：个股日收益对指数回归的 R² 往往只有零点几，说明大部分波动是个股特有的。
- **对齐问题**：个股停牌日没有收益，指数有。`dropna()` 会去掉这些日子；但复牌日的收益是停牌期累计收益，会与指数当天收益错配，产生异常点。
- **无风险利率**：日频分析中 $R_f$ 影响很小，常直接忽略；年化 Sharpe Ratio 时应扣除（A 股研究常用一年期国债收益率或 Shibor 等作为参照，取值以实际数据为准）。

---

## 10. Plotly 交互式图表

### 10.1 为什么需要交互式图表

matplotlib 适合出静态报告图；但研究时你经常需要：放大某段行情、悬停看具体数值、同时对比 K 线和指标。**Plotly** 生成的 HTML 图表支持缩放、悬停、图例开关，非常适合探索性分析和仪表板（本阶段实战项目）。

### 10.2 典型布局：K 线 + 均线 + 成交量 + 指标

```
┌──────────────────────────────────────────────────────┐
│  K 线 (Candlestick) + MA20 + 布林带      高度 60%    │
├──────────────────────────────────────────────────────┤
│  成交量 (Bar)                            高度 20%    │
├──────────────────────────────────────────────────────┤
│  MACD / RSI                              高度 20%    │
└──────────────────────────────────────────────────────┘
          ↑ 共享 X 轴：缩放一个，其他联动
```

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)

fig.add_trace(go.Candlestick(
    x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    increasing_line_color="red", decreasing_line_color="green",   # A 股习惯：红涨绿跌
    name="600519"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=ma20, name="MA20", line=dict(width=1)), row=1, col=1)

vol_color = np.where(df["close"] >= df["open"], "red", "green")
fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_color, name="成交量"), row=2, col=1)

dif, dea, hist = macd(df["close"])
fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD柱"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=dif, name="DIF"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=dea, name="DEA"), row=3, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=800, hovermode="x unified")
fig.write_html("outputs/600519_dashboard.html")
```

### 10.3 非交易日空白问题

用日期作为 X 轴时，Plotly 会为周末和节假日留出空白，K 线图看起来"断断续续"。两种解决方式：

```python
# 方法1：跳过周末（但节假日仍有空白，需要额外提供节假日列表）
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

# 方法2：把 X 轴设为类别轴，只显示有数据的日期（A 股更省事）
fig.update_xaxes(type="category")
```

### 10.4 可视化陷阱

- **对数坐标**：长周期净值曲线（如茅台 20 年）应该用对数 Y 轴（`fig.update_yaxes(type="log")`），否则早期的波动被压扁，看起来"后期涨得特别猛"。对数坐标下，等距离代表等百分比变化。
- **数据量过大**：几十万根分钟 K 线直接塞进浏览器会卡死。先 resample 或只画最近一段。
- **颜色约定**：A 股红涨绿跌，海外绿涨红跌。给不同受众的报告要注意。
- **图表会骗人**：起止日期的选择、Y 轴范围都能显著改变观感。研究报告中要固定口径。

---

## 11. 向量化与性能优化

### 11.1 直觉

Python 的 `for` 循环每次迭代都要经过解释器；而 NumPy/pandas 的向量化操作把循环下推到 C 层，对整个数组一次性计算。

**比喻**：for 循环像收银员一件一件扫码；向量化像把整车货物开过 RFID 门，一次扫完。

### 11.2 性能层级

| 方式 | 典型相对速度 | 适用 |
| --- | --- | --- |
| `df.iterrows()` | 最慢（可能慢数百倍以上） | 几乎永远不要用 |
| `df.apply(func, axis=1)` | 慢 | 逻辑复杂且数据量小时 |
| Python `for` + `.iloc` | 慢 | 原型验证 |
| pandas 内置向量化（`rolling`、`shift`、`ewm`） | 快 | **首选** |
| NumPy 数组运算 | 很快 | 需要极致性能、逻辑可向量化 |
| Numba `@njit` | 接近 C | **路径依赖**、无法向量化的循环 |

> 具体倍数与数据规模和机器有关，请在 `10-vectorization-performance.py` 中用 `%timeit` 或 `time.perf_counter` 自己测。

### 11.3 示例：同一个指标的三种写法

```python
# ❌ 循环版：20 日均线
def ma_loop(close: pd.Series, n: int = 20) -> pd.Series:
    out = [np.nan] * len(close)
    for i in range(n - 1, len(close)):
        out[i] = close.iloc[i - n + 1: i + 1].mean()
    return pd.Series(out, index=close.index)

# ✅ pandas 向量化
ma_vec = close.rolling(20).mean()

# ✅ NumPy 累加和技巧（O(n)，不依赖 pandas）
def ma_cumsum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out
```

### 11.4 向量化的边界：路径依赖

有些逻辑**当前状态依赖于上一步的结果**，无法简单向量化。最典型的是**移动止损（Trailing Stop）**：是否止损取决于持仓后的最高价，而是否持仓又取决于之前是否止损过。

```python
from numba import njit

@njit
def trailing_stop(close: np.ndarray, entry: np.ndarray, stop_pct: float) -> np.ndarray:
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

pos = trailing_stop(close.to_numpy(), signal.to_numpy(), 0.08)
```

这正是 Phase 4 中**向量化回测 vs 事件驱动回测**之争的缩影：向量化快，但表达不了复杂的路径依赖逻辑。

### 11.5 其他性能技巧

```python
# 1. rolling().apply 慢，加 raw=True 传 ndarray 而不是 Series，可明显提速
slope = close.rolling(20).apply(lambda a: np.polyfit(np.arange(20), a, 1)[0], raw=True)

# 2. 用 np.where / np.select 代替逐行 if-else
signal = np.where(close > ma20, 1, 0)

# 3. 面板数据：宽表上的整体运算远快于长表 groupby + apply
#    rets = wide_close.pct_change() 比 long.groupby('code').apply(...) 快得多

# 4. 数据类型：大面板可考虑 float32 节省一半内存（注意累计运算的精度损失）

# 5. 存储：Parquet + 列裁剪读取（Phase 1），只读需要的列和日期范围
```

### 11.6 陷阱：快但错

向量化最大的风险不是性能，而是**一个 `shift` 方向错误导致的前视偏差**，而且这种错误不会报错，只会让回测"好得不真实"。

```python
# 信号：今天收盘价上穿 MA20
signal = (close > ma20).astype(int)

# ❌ 用今天的信号吃今天的收益（今天收盘才知道信号，却赚了今天全天的涨幅）
strat_ret_wrong = signal * ret

# ✅ 今天的信号，明天才持仓
strat_ret = signal.shift(1) * ret
```

**自检方法**：把收益序列随机打乱（或用未来数据替换），如果策略表现几乎不变或仍然很好，多半有 bug。

---

## 12. 最佳实践与常见陷阱

### 12.1 最佳实践

```
┌──────────────────────────────────────────────────────────────────┐
│                      金融时间序列编码规范                          │
├──────────────────────────────────────────────────────────────────┤
│  ✅ 全项目统一常量：TRADING_DAYS = 252（或 244），写在一个地方      │
│  ✅ 收益率基于复权价；交易约束判断基于不复权价                     │
│  ✅ index 为 DatetimeIndex，已排序、无重复（加断言）               │
│  ✅ 每个派生列都能回答："t 时刻是否可得？"                         │
│  ✅ 信号 → 持仓之间显式 shift(1)，并写注释说明                     │
│  ✅ 长表运算一律 groupby('code')                                   │
│  ✅ 低频数据按"公告日/可得日"对齐，而不是"所属期"                  │
│  ✅ 指标实现与第三方库（如 TA-Lib / pandas-ta）对拍                │
│  ✅ 优先向量化；路径依赖逻辑用 Numba                               │
└──────────────────────────────────────────────────────────────────┘
```

**数据健康检查**（每次加载数据后跑一遍）：

```python
def sanity_check(df: pd.DataFrame) -> None:
    assert df.index.is_monotonic_increasing, "index 未排序"
    assert not df.index.duplicated().any(), "存在重复日期"
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9).all(), "high 异常"
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9).all(), "low 异常"
    big_moves = df["close"].pct_change().abs() > 0.25
    if big_moves.any():
        print("⚠️ 单日变动超过 25%，检查是否复权错误或为新股/特殊日：")
        print(df.loc[big_moves].index.tolist())
```

> A 股主板日涨跌幅限制一般为 ±10%，创业板/科创板一般为 ±20%（具体以交易所最新规则为准，新股上市初期等情形另有规定）。复权后的收益若出现明显超出限制的单日变动，大概率是数据或复权问题。

### 12.2 常见陷阱速查表

| # | 陷阱 | 症状 | 修复 |
| --- | --- | --- | --- |
| 1 | 未复权算收益 | 除息日出现假下跌 | 用复权价 |
| 2 | 长表直接 `pct_change` | 股票之间"串味" | `groupby('code')` |
| 3 | `rolling(center=True)` | 回测极好 | 删掉 `center=True` |
| 4 | 全样本 mean/std 标准化 | 回测好、实盘差 | 用 `expanding` 或 `rolling` |
| 5 | 信号未 shift | 回测好得离谱 | `signal.shift(1)` |
| 6 | 日历月末标签对不上交易日 | 月度信号丢失/错位 | 用实际交易日作标签 |
| 7 | 财报按报告期对齐 | 提前"知道"业绩 | 按公告日对齐 |
| 8 | 短样本年化 | 年化收益几百% | 样本不足 1 年不年化或注明 |
| 9 | 基准用价格指数 | 超额收益虚高 | 用全收益指数或说明 |
| 10 | 加密用 252 年化 | 低估波动 | 用 365 |

---

## 13. 总结

### 13.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                     Phase 2 核心概念地图                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📌 收益率                                                        │
│     ├── 简单收益：截面可加（组合）                                │
│     ├── 对数收益：时间可加（多期、建模）                          │
│     └── 波动拖累：几何均值 ≈ 算术均值 − σ²/2                      │
│                                                                   │
│  📌 年化                                                          │
│     ├── 收益：CAGR（几何）                                        │
│     ├── 波动：σ × √N（假设 i.i.d.）                               │
│     └── N：A 股 ~242–252 / 加密 365，全项目统一                   │
│                                                                   │
│  📌 数据操作                                                      │
│     ├── rolling / expanding / ewm（禁止 center=True）             │
│     ├── resample：OHLCV 分列聚合，注意标签对齐                    │
│     └── 宽表（计算）⇄ 长表（存储），截面 vs 时间序列运算          │
│                                                                   │
│  📌 技术指标：MA/EMA/MACD/RSI/ATR/布林带 —— 描述工具，不是圣杯    │
│                                                                   │
│  📌 风险与相对表现                                                │
│     ├── 回撤 / 水下曲线 / MDD 起止日期 / 恢复期                   │
│     └── 超额收益、跟踪误差、IR、Beta、Alpha                       │
│                                                                   │
│  📌 工程                                                          │
│     ├── 向量化优先，路径依赖用 Numba                              │
│     └── 每一行都问："t 时刻我真的知道这个值吗？"                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 13.2 学习检查对照

| README 学习检查 | 对应章节 |
| --- | --- |
| 知道什么时候用简单收益、什么时候用对数收益 | 2.3 |
| 能正确年化日频、周频、月频数据 | 3.2–3.5 |
| 能计算最大回撤及其起止日期 | 8.3 |
| 能用向量化代替循环实现技术指标 | 7、11 |

### 13.3 与下一阶段的衔接

本阶段你计算了大量统计量：年化收益、波动率、Beta、相关系数、最大回撤……但还有一个根本问题没有回答：

> **这些数字靠谱吗？**

- 茅台过去 5 年的年化收益是 X%，这个估计的**误差有多大**？
- 一个策略回测年化 15%，这是**真本事还是运气**？
- 日收益真的服从正态分布吗？用 $\pm 2\sigma$ 估计风险会不会严重低估极端行情？
- 两只股票相关系数 0.8，能用来做配对交易吗？

这些正是 **Phase 3：量化数学与统计** 要回答的问题——分布与肥尾、假设检验、蒙特卡洛、回归、协整、平稳性与 GARCH。Phase 2 给了你"测量工具"，Phase 3 教你"判断测量结果是否可信"。

---

> 💡 **提示**：本文中涉及 A 股交易日数量、涨跌幅限制等规则性数字仅用于帮助理解，具体以交易所最新规则为准。
