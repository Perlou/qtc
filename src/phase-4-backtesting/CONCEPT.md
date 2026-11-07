# Phase 4: 回测系统原理 - 核心概念深度解析

> 本文是 Phase 4 的配套概念文档。建议先通读一遍建立直觉，再逐个运行 `01-*.py` ~ `11-*.py`。
> 读者假设：编程能力强，已完成 Phase 2（收益/回撤/向量化）和 Phase 3（假设检验/多重检验/平稳性）。
> 本阶段的核心信念：**一个回测结果默认是错的，直到你证明它没错。**

---

## 目录

1. [引言：回测是什么、不是什么](#1-引言回测是什么不是什么)
2. [向量化回测](#2-向量化回测)
3. [绩效指标：Sharpe、Sortino、MDD、Calmar、胜率](#3-绩效指标sharpesortinomddcalmar胜率)
4. [交易成本与滑点建模](#4-交易成本与滑点建模)
5. [A 股约束：T+1、涨跌停、停牌、100 股](#5-a-股约束t1涨跌停停牌100-股)
6. [事件驱动回测引擎](#6-事件驱动回测引擎)
7. [向量化 vs 事件驱动：架构对比](#7-向量化-vs-事件驱动架构对比)
8. [前视偏差与未来函数](#8-前视偏差与未来函数)
9. [幸存者偏差与其他数据偏差](#9-幸存者偏差与其他数据偏差)
10. [过拟合与多重检验](#10-过拟合与多重检验)
11. [Walk-Forward 与样本外检验](#11-walk-forward-与样本外检验)
12. [开源框架：backtrader 与 vectorbt](#12-开源框架backtrader-与-vectorbt)
13. [最佳实践与常见陷阱](#13-最佳实践与常见陷阱)
14. [总结](#14-总结)

---

## 1. 引言：回测是什么、不是什么

### 1.1 定义

**回测（Backtesting）**：用历史数据模拟"如果过去按这套规则交易，会得到什么结果"。

**比喻**：回测像**飞行模拟器**。它能帮你发现明显的设计错误（飞机根本飞不起来），训练你对各种情况的反应，但模拟器里飞得好，不代表真飞机上就安全——模拟器对天气、机械故障、人的紧张都只是近似。更危险的是，如果模拟器本身有 bug（比如重力设错了），你会在模拟器里成为"王牌飞行员"，然后在现实中坠机。

### 1.2 回测的正确定位：证伪工具，而非证明工具

| 回测能做的 | 回测不能做的 |
| --- | --- |
| ✅ 否定一个想法（"这个在历史上就不赚钱"） | ❌ 证明一个策略在未来会赚钱 |
| ✅ 估计策略在历史上的风险特征 | ❌ 精确预测实盘收益 |
| ✅ 暴露实现层面的错误 | ❌ 模拟你对市场的冲击（大资金时） |
| ✅ 比较不同方案的相对优劣 | ❌ 模拟历史上没出现过的情景 |
| ✅ 估计成本、换手对收益的侵蚀 | ❌ 消除你作为研究员的选择偏差 |

> 引用一句业内常说的话：**"我从没见过一个不好看的回测。"** 因为不好看的回测都没被拿出来展示——这本身就是一种幸存者偏差。

### 1.3 回测流程

```
┌───────────────────────────────────────────────────────────────────────┐
│                           标准回测流程                                 │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ① 研究假设          "A 股存在短期反转效应"                            │
│        │              （先有逻辑，再找数据；不是反过来）                │
│        ▼                                                               │
│  ② 数据准备          复权、交易日历、停牌、ST、退市、成分股历史         │
│        │              （Phase 1：数据质量决定回测上限）                │
│        ▼                                                               │
│  ③ 信号生成          只用 t 时刻及之前可得的信息 ← 前视偏差（第 8 节）  │
│        │                                                               │
│        ▼                                                               │
│  ④ 组合构建          信号 → 目标仓位 → 订单                            │
│        │                                                               │
│        ▼                                                               │
│  ⑤ 撮合模拟          成交价、成本、滑点、A 股约束（第 4、5 节）         │
│        │                                                               │
│        ▼                                                               │
│  ⑥ 绩效评估          Sharpe、MDD、换手、成本敏感性（第 3 节）          │
│        │                                                               │
│        ▼                                                               │
│  ⑦ 稳健性检验        样本外、Walk-Forward、参数扰动、多重检验修正       │
│        │              （第 10、11 节）                                 │
│        ▼                                                               │
│  ⑧ 模拟盘 → 小资金实盘（Phase 13）                                     │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.4 回测与实盘的差距从哪里来

```
          回测收益（看起来很美）
    ┌──────────────────────────────┐
    │  − 前视偏差带来的虚假收益     │  ← 代码 bug，最致命
    │  − 幸存者偏差                 │  ← 数据问题
    │  − 过拟合 / 多重检验          │  ← 研究流程问题
    │  − 交易成本（佣金/印花税）    │  ← 可精确建模
    │  − 滑点与冲击成本             │  ← 只能近似建模
    │  − 无法成交（涨跌停/停牌）    │  ← A 股特有，可建模
    │  − 策略容量与拥挤             │  ← 很难建模
    │  − 执行延迟与系统故障         │  ← Phase 13
    └──────────────────────────────┘
          实盘收益（常常显著更低）
```

本阶段的目标就是逐项把这些"虚假收益"挤掉。

---

## 2. 向量化回测

### 2.1 直觉

**向量化回测（Vectorized Backtest）**：把整个历史当作一个矩阵，用 pandas/NumPy 一次性算出所有时点的信号、仓位和收益，没有显式的逐日循环。

**比喻**：向量化回测像**看整部电影的剧本**——你一次看到全部情节，处理很快；事件驱动回测像**一帧一帧地看电影**——慢，但你永远不会提前知道结局。向量化的危险恰恰在于：剧本就摊在桌上，一不小心就"偷看了结局"。

### 2.2 核心公式

$$
w_t = f(\text{信号}_{t-1}), \qquad R^{\text{strat}}_t = w_t \cdot R_t - c \cdot |w_t - w_{t-1}|
$$

- $w_t$：$t$ 日持有的仓位（如 0 或 1；多资产时为权重向量）
- $\text{信号}_{t-1}$：$t-1$ 日收盘后计算出的信号——**必须滞后**
- $R_t$：资产在 $t$ 日的收益
- $c$：单边交易成本率
- $|w_t - w_{t-1}|$：换手（仓位变化量）

### 2.3 代码：最小可用的向量化回测

```python
import numpy as np
import pandas as pd

def vectorized_backtest(close: pd.Series, signal: pd.Series, cost_rate: float = 0.001):
    """
    signal: t 日收盘后计算的目标仓位（0/1 或 -1~1）
    假设：在 t 日收盘附近以收盘价成交（例如参与收盘集合竞价），从 t+1 日起承担收益
    """
    ret = close.pct_change().fillna(0)
    pos = signal.shift(1).fillna(0)                 # 关键：信号滞后一期
    turnover = pos.diff().abs().fillna(pos.abs())   # 首日建仓也算换手
    strat_ret = pos * ret - turnover * cost_rate
    nav = (1 + strat_ret).cumprod()
    return pd.DataFrame({"pos": pos, "ret": strat_ret, "nav": nav, "turnover": turnover})

# 例：茅台 20 日均线择时（close 为后复权收盘价）
signal = (close > close.rolling(20).mean()).astype(float)
bt = vectorized_backtest(close, signal, cost_rate=0.001)
```

### 2.4 成交价假设：shift(1) 还是 shift(2)？

"信号滞后一期"背后其实隐含了一个成交价假设，必须想清楚：

| 假设 | 实现 | 现实性 |
| --- | --- | --- |
| t 日收盘出信号、**t 日收盘价**成交 | `signal.shift(1) * close.pct_change()` | 偏乐观：严格来说要在收盘前用"接近收盘"的价格算信号。A 股有收盘集合竞价，日频低换手策略常用此近似 |
| t 日收盘出信号、**t+1 日开盘价**成交 | `signal.shift(2) * open.pct_change()` | 更保守：完全不需要预知收盘价 |
| t 日收盘出信号、t+1 日 VWAP 成交 | 需要分钟数据或 VWAP 字段 | 接近机构的实际执行 |

第二行的推导：$t$ 日收盘的信号在 $t+1$ 日开盘执行，持有区间是 $\text{open}_{t+1} \to \text{open}_{t+2}$，这段收益在 `open.pct_change()` 中位于 $t+2$ 行，所以信号要 `shift(2)`。

> 💡 **建议**：两种都跑一遍。如果策略只在"收盘价成交"假设下赚钱、在"次日开盘成交"假设下不赚钱，说明它的收益可能来自一个你无法真正获取的价格。

### 2.5 多资产向量化

```python
# wide_close: 日期 × 股票 的后复权收盘价宽表
rets = wide_close.pct_change()

# 截面信号：每天选 20 日动量最高的 10 只，等权
mom = wide_close / wide_close.shift(20) - 1
rank = mom.rank(axis=1, ascending=False)
weights = (rank <= 10).astype(float)
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

pos = weights.shift(1).fillna(0)
turnover = pos.diff().abs().sum(axis=1)
port_ret = (pos * rets.fillna(0)).sum(axis=1) - turnover * 0.001
```

> ⚠️ 这段代码还没有处理：停牌股票无法买卖、涨停股票买不进、整数手、T+1。在向量化框架中加入这些约束会非常别扭（见第 5、7 节）——这正是事件驱动架构存在的理由。

### 2.6 向量化回测的优势与局限

| 优势 | 局限 |
| --- | --- |
| 极快：几千个参数组合几秒钟跑完 | 难以表达路径依赖（止损、加减仓、资金约束） |
| 代码短，研究迭代快 | 难以表达 A 股成交约束（涨跌停、T+1、整手） |
| 适合截面因子研究、大规模参数扫描 | 容易因 `shift` 错误引入前视偏差 |
| | 研究代码无法直接复用到实盘 |

---

## 3. 绩效指标：Sharpe、Sortino、MDD、Calmar、胜率

### 3.1 指标全景

| 维度 | 指标 | 一句话解释 |
| --- | --- | --- |
| 收益 | 年化收益（CAGR） | 复利意义上每年赚多少 |
| 风险 | 年化波动率 | 收益的离散程度 |
| 风险 | 最大回撤（MDD） | 最惨时从高点亏了多少 |
| 风险调整 | **Sharpe Ratio** | 每单位总波动换来多少超额收益 |
| 风险调整 | **Sortino Ratio** | 每单位"下行波动"换来多少超额收益 |
| 风险调整 | **Calmar Ratio** | 每单位最大回撤换来多少年化收益 |
| 相对基准 | 超额收益、信息比率 IR | 相对基准的表现（Phase 2） |
| 交易 | 胜率、盈亏比、Profit Factor | 单笔交易的质量 |
| 交易 | 换手率、平均持仓期 | 成本敏感度、容量 |

### 3.2 Sharpe Ratio

$$
\text{SR} = \frac{\mathbb{E}[R_t - R_f]}{\sigma(R_t - R_f)} \times \sqrt{N}
$$

- $R_t$：策略周期收益；$R_f$：同周期无风险利率
- $\sigma(\cdot)$：标准差
- $N$：每年周期数（日频常用 252，见 Phase 2）

**直觉**：开车的"油耗"——每承担一单位波动，换来多少收益。

**经验量级**（仅供直觉，依资产与市场而异）：

| 年化 Sharpe（扣成本后） | 通常的含义 |
| --- | --- |
| < 0.5 | 弱，难以与运气区分（Phase 3：需要十几年数据） |
| 0.5 – 1.0 | 中等，许多长期可行的低频策略在这个区间 |
| 1.0 – 2.0 | 较好 |
| > 2.0（低频日线策略） | **先怀疑 bug**：前视偏差、成本遗漏、幸存者偏差 |
| > 3.0 | 通常只在高频/做市等特殊领域可持续，且容量有限 |

**Sharpe 的局限**：

- 把上涨波动也当作风险
- 看不到偏度和肥尾：卖期权类负偏策略在爆仓前 Sharpe 往往很高（Phase 3 第 3.6 节）
- 依赖 i.i.d. 假设年化；收益自相关时 $\sqrt{N}$ 年化会失真

### 3.3 Sortino Ratio

只惩罚"坏的"波动：

$$
\text{Sortino} = \frac{\mathbb{E}[R_t - \text{MAR}]}{\text{DD}} \times \sqrt{N}, \qquad \text{DD} = \sqrt{\mathbb{E}\left[\min(R_t - \text{MAR},\, 0)^2\right]}
$$

- MAR（Minimum Acceptable Return）：最低可接受收益，常取 0 或无风险利率
- DD（Downside Deviation）：下行偏差，只统计低于 MAR 的部分。注意分母是**全部样本数**而不是只用负收益样本数（后者是常见的实现错误）

### 3.4 最大回撤与 Calmar Ratio

$$
\text{MDD} = \min_t\left(\frac{V_t}{\max_{s \le t} V_s} - 1\right), \qquad \text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}
$$

- $V_t$：净值；CAGR：几何年化收益

Calmar 对投资者很直观："为了每年赚 X%，我得忍受多深的坑"。但 MDD 是单一极值，样本越长 MDD 越大，所以**不同回测区间的 Calmar 不可直接比较**。

### 3.5 交易层面指标

| 指标 | 公式 | 说明 |
| --- | --- | --- |
| 胜率 | 盈利交易笔数 / 总笔数 | 按"笔"还是按"天"统计要说明 |
| 盈亏比 | 平均盈利 / 平均亏损 | 与胜率一起看（Phase 3 第 6 节） |
| Profit Factor | 总盈利 / 总亏损 | > 1 才赚钱；扣成本后 1.2 以下通常很脆弱 |
| 换手率 | $\sum_t \sum_i |w_{i,t} - w_{i,t-1}|$ / 年 | 决定成本拖累，见第 4 节 |
| 平均持仓天数 | | 影响容量和成本敏感度 |

### 3.6 代码：绩效报告函数

```python
def performance_report(ret: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> pd.Series:
    ret = ret.dropna()
    rf = rf_annual / periods
    excess = ret - rf
    nav = (1 + ret).cumprod()

    cagr = nav.iloc[-1] ** (periods / len(ret)) - 1
    vol = ret.std() * np.sqrt(periods)
    sharpe = excess.mean() / excess.std() * np.sqrt(periods)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(periods)
    sortino = excess.mean() * periods / downside
    mdd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    t_stat = sharpe * np.sqrt(len(ret) / periods)      # Phase 3：t ≈ SR × √年数

    return pd.Series({
        "年化收益": cagr, "年化波动": vol, "Sharpe": sharpe, "Sortino": sortino,
        "最大回撤": mdd, "Calmar": calmar, "日胜率": (ret > 0).mean(),
        "Sharpe t 值": t_stat, "样本年数": len(ret) / periods,
    })
```

> 生产中可以使用 `quantstats`、`empyrical` 等库生成完整报告，但**至少手写一次**，并用自己的实现与库对拍。不同库对年化、Sortino 分母、无风险利率的处理不一致。

---

## 4. 交易成本与滑点建模

### 4.1 直觉

**比喻**：交易成本像**过路费**。偶尔开一次车，过路费无所谓；如果你每天往返几十次（高换手策略），过路费可能比油钱还贵。许多回测中看起来很美的短线策略，在扣完"过路费"后直接变成亏损。

### 4.2 A 股股票交易成本构成

| 成本项 | 收取方式 | 量级（以交易所/券商最新规则为准） |
| --- | --- | --- |
| **佣金** | 买卖双向，按成交金额比例 | 与券商协商，监管设有上限；常见万分之几，不少券商有单笔最低收费（如 5 元） |
| **印花税** | **仅卖出**收取 | 自 2023 年 8 月 28 日起减半为成交金额的 0.05%（以最新政策为准） |
| **过户费** | 买卖双向 | 成交金额的十万分之一量级（以中国结算最新标准为准） |
| **滑点** | 隐性成本 | 取决于流动性、订单大小、执行方式 |
| **冲击成本** | 隐性成本 | 大单推动价格向不利方向移动 |

> ⚠️ 这些费率历史上调整过多次。回测跨越多年时，严格做法是**按当时的费率**计算（例如 2023 年 8 月之前印花税为 0.1%）。

期货、加密货币的成本结构不同：期货按手或按金额收手续费（各品种、各交易所不同，部分品种平今仓有特殊费率），无印花税；加密交易所常区分 Maker/Taker 费率。

### 4.3 显性成本建模

```python
def a_share_cost(value: float, is_sell: bool,
                 comm_rate=0.00025, min_comm=5.0,
                 stamp_rate=0.0005, transfer_rate=0.00001) -> float:
    """A 股单笔交易成本（元）。费率为示例参数，请按券商合同与最新规则设置。"""
    commission = max(value * comm_rate, min_comm)
    transfer = value * transfer_rate
    stamp = value * stamp_rate if is_sell else 0.0
    return commission + transfer + stamp

# 最低佣金的影响：小额交易的实际费率远高于名义费率
for v in [2_000, 10_000, 100_000]:
    print(v, f"{a_share_cost(v, False) / v:.4%}")
```

**最低佣金陷阱**：账户资金小、持仓股票多时，单笔金额小，5 元最低佣金会让实际费率成倍上升。小资金回测必须建模这一点。

### 4.4 滑点与冲击成本模型

**滑点（Slippage）**：你看到的价格与实际成交价之间的差。

| 模型 | 公式 | 适用 |
| --- | --- | --- |
| 固定比例 | 成交价 = 参考价 × (1 ± s)，如 s = 5–20 bp | 简单、保守估计 |
| 半价差 | 成交价 = 中间价 ± 价差/2 | 有盘口数据时 |
| 成交量参与率限制 | 单日成交 ≤ 当日成交量 × p%（如 5%–10%） | 防止回测中"无限流动性" |
| **平方根冲击模型** | $\text{Impact} \approx k\,\sigma\,\sqrt{Q/V}$ | 大资金、机构执行（Phase 11） |

平方根模型中：

- $\sigma$：资产日波动率
- $Q$：订单数量；$V$：日成交量（$Q/V$ 即参与率）
- $k$：经验系数，需用实际成交数据校准

**直觉**：冲击成本随订单规模增长，但增长速度是"开根号"——订单翻 4 倍，单位冲击约翻 2 倍。这是一个被广泛观察到的经验规律，具体系数因市场和股票而异。

### 4.5 换手与成本拖累

$$
\text{年化成本拖累} \approx \text{年化换手率} \times \text{单边平均成本率}, \qquad \text{年化换手率} = \sum_{t \in \text{一年}} \sum_i |w_{i,t} - w_{i,t-1}|
$$

- 换手率把买入和卖出都计入（卖出 100% 仓位再买入 100% 新仓位，计为 2）
- 单边平均成本率：每成交 1 元平均付出的成本（佣金、印花税摊销、滑点等）

例如：一个每天把全部仓位换成新股票的策略，每天换手为 2（卖 1 + 买 1），一年约 252 × 2 = 504 倍；单边平均成本 0.1% 时，年化成本拖累约 50%。

**盈亏平衡成本（Break-even Cost）**：

$$
c^* = \frac{\text{扣成本前年化收益}}{\text{年化换手率}}
$$

如果 $c^*$ 只比你的真实成本高一点点，策略在实盘中几乎没有安全边际。

### 4.6 成本敏感性分析（README 学习检查要求）

```python
costs_bp = [0, 5, 10, 15, 20, 30, 50]
rows = []
for c in costs_bp:
    bt = vectorized_backtest(close, signal, cost_rate=c / 10_000)
    rep = performance_report(bt["ret"])
    rows.append({"单边成本(bp)": c, "年化收益": rep["年化收益"], "Sharpe": rep["Sharpe"]})
sens = pd.DataFrame(rows)
print(sens.to_string(index=False))
```

```
Sharpe
  ▲
2 ┤●
  │  ●
1 ┤     ●                        ← 稳健策略：曲线平缓
  │        ●      ●       ●
0 ┼──────────●────────────────▶ 单边成本 (bp)
  │              ●                ← 脆弱策略：成本稍高就转负
-1┤                  ●
```

**回测报告必须包含这张表/图**，并标明你认为的"真实成本"在曲线上的位置。

---

## 5. A 股约束：T+1、涨跌停、停牌、100 股

> 以下规则描述用于帮助建模，交易规则会调整，**具体以交易所最新规则为准**。

### 5.1 约束一览

| 约束 | 规则要点 | 对回测的影响 |
| --- | --- | --- |
| **T+1** | 股票当日买入，下一交易日才能卖出；卖出所得资金当日可用于买入 | 当日买入的仓位不能当日止损；日内反转策略不可行 |
| **涨跌停** | 主板一般 ±10%，主板 ST 股 2026-07-06 前为 ±5%、之后为 ±10%，创业板/科创板一般 ±20%，北交所一般 ±30%；新股上市初期等有特殊规定 | 涨停时往往买不进，跌停时往往卖不出 |
| **停牌** | 停牌期间无法交易，复牌时可能大幅跳空 | 持仓被"锁住"；信号无法执行 |
| **整数手** | 买入一般须为 100 股整数倍（科创板单笔申报一般不少于 200 股，超出部分可按 1 股递增）；卖出时不足 100 股的零股须一次性卖出 | 资金无法精确按权重分配；高价股（如茅台）对小资金影响显著 |

期货是 T+0、可做空，涨跌停幅度因品种而异；加密货币一般 7×24 交易、无涨跌停（但交易所可能有价格保护机制）。

### 5.2 T+1 建模

```
     t 日                     t+1 日
┌──────────────┐        ┌──────────────┐
│ 09:30 买入   │        │ 可以卖出     │
│ 14:00 暴跌   │        │              │
│ ❌ 不能卖出  │ ─────▶ │ 开盘可能继续 │
│              │        │ 跌停卖不出   │
└──────────────┘        └──────────────┘
```

在事件驱动引擎中，维护两个持仓量：

- `positions[code]`：总持仓
- `sellable[code]`：可卖持仓。每个交易日开始时 `sellable = positions`；当日买入只增加 `positions`，不增加 `sellable`

**日线策略的一个巧合**：如果你在 t 日收盘出信号、t+1 日开盘成交，那么任何一笔买入最早也要在 t+2 日开盘才能卖出——T+1 在日线级别"自动满足"。但只要你引入**盘中止损**（用当日最低价判断触发），T+1 就会立刻成为问题。

### 5.3 涨跌停建模

**涨跌停价计算**：

$$
P_{\text{up}} = \text{round}\big(P_{\text{pre}} \times (1 + L)\big), \qquad P_{\text{down}} = \text{round}\big(P_{\text{pre}} \times (1 - L)\big)
$$

- $P_{\text{pre}}$：前收盘价（除权除息日为交易所公布的**除权除息参考价**）
- $L$：涨跌幅限制比例
- round：四舍五入到价格最小变动单位（A 股为 0.01 元）

> ⚠️ **必须用不复权价格判断涨跌停**。复权价格只用于计算收益。用复权价算涨跌停价，在除权除息日附近会得到错误结果。

```python
from decimal import Decimal, ROUND_HALF_UP

def limit_prices(pre_close: float, pct: float = 0.10) -> tuple[float, float]:
    """计算涨跌停价（四舍五入到分；用 Decimal 避免浮点舍入误差）"""
    pc = Decimal(str(pre_close))
    up = (pc * Decimal(str(1 + pct))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    down = (pc * Decimal(str(1 - pct))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(up), float(down)
```

**日线数据下判断"能否成交"的近似规则**：

| 情形 | 判断 | 买入 | 卖出 |
| --- | --- | --- | --- |
| 一字涨停 | open = high = low = close = 涨停价 | ❌ 几乎不可能 | ✅ |
| 一字跌停 | open = high = low = close = 跌停价 | ✅ | ❌ 几乎不可能 |
| 开盘涨停、盘中打开 | open = 涨停价, low < 涨停价 | ⚠️ 有可能，但通常排队 | ✅ |
| 收盘涨停 | close = 涨停价 | 以收盘价买入**不现实**（封板排队） | ✅ |

保守做法：**按收盘价成交时，收盘涨停视为买不进、收盘跌停视为卖不出；按开盘价成交时，开盘涨停视为买不进、开盘跌停视为卖不出**。有分钟/Level-2 数据时可以做得更精细。

**为什么这很重要**：许多 A 股短线策略（如"追涨停""打板"、超跌反弹）的回测收益很大一部分来自"在涨停价买入"或"在跌停价卖出"——这在实盘中恰恰是最难成交的。不处理涨跌停，回测会系统性高估这类策略。

### 5.4 停牌建模

- **识别**：当日无行情记录，或成交量为 0
- **处理**：
  - 停牌期间不能买入或卖出；挂单作废或延续（需明确规则）
  - 持仓按停牌前最后价格估值，但要意识到复牌后价格可能大幅跳空
  - 长期停牌股票会让组合被"锁住"一部分资金
- **向量化回测中常见错误**：对停牌期价格 `ffill()` 后计算信号，策略在停牌期间"成功卖出"——实际上根本不可能。

### 5.5 整数手建模

```python
def round_lot(target_value: float, price: float, lot: int = 100) -> int:
    """按 100 股整数倍向下取整"""
    return int(target_value // (price * lot)) * lot

# 茅台股价在一两千元量级时，一手（100 股）需要十几万到二十万元
# 一个 10 万元的账户：
print(round_lot(100_000, 1700))   # 0 —— 一股也买不了
```

**影响**：

- 小资金无法持有高价股，或权重严重偏离目标
- 多股票等权组合中，资金越小、股票越多，偏差越大
- 回测时如果使用"小数股"，小资金策略的回测结果会严重失真

### 5.6 其他值得注意的 A 股因素

| 因素 | 说明 |
| --- | --- |
| ST / *ST | 涨跌幅限制不同，退市风险高；很多策略选择剔除 |
| 新股 | 上市初期涨跌幅规则特殊，且一般投资者很难按开盘价大量买到 |
| 退市整理期 | 价格行为极端，回测数据可能缺失 |
| 融券限制 | A 股做空受限，股票多空策略在实盘中往往只能做"多头 + 股指期货对冲" |
| 集合竞价 | 开盘价、收盘价由集合竞价产生，大单在集合竞价中也有冲击 |

---

## 6. 事件驱动回测引擎

### 6.1 直觉

**事件驱动（Event-Driven）回测**：模拟真实交易系统的运行方式——行情一根一根地"到达"，策略只能看到已经到达的数据，产生订单，订单交给模拟券商撮合，成交回报更新账户。

**比喻**：向量化是"上帝视角看完整部电影"，事件驱动是"直播"——你只能看到当前这一帧以及之前的内容，**结构上**就不可能偷看结局。

### 6.2 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                       事件驱动回测引擎架构                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌────────────┐  MarketEvent   ┌──────────────────────────────┐      │
│   │  DataFeed  │ ─────────────▶ │          Event Queue          │      │
│   │ (逐日推送) │                │  (FIFO, 主循环不断取事件分发)  │      │
│   └────────────┘                └──┬──────────┬──────────┬─────┘      │
│                                    │          │          │            │
│                      MarketEvent   │  Order   │   Fill   │            │
│                                    ▼          ▼          ▼            │
│   ┌────────────┐            ┌────────────┐          ┌────────────┐    │
│   │  Strategy  │──Order───▶ │   Broker   │──Fill──▶ │ Portfolio  │    │
│   │ (只见历史) │            │ (撮合/成本 │          │ (现金/持仓 │    │
│   │            │◀──读取持仓─│  A股约束)  │          │  T+1/净值) │    │
│   └────────────┘            └────────────┘          └────────────┘    │
│                                                                       │
│   每个交易日的时序：                                                   │
│   ① 开盘：Portfolio 刷新可卖数量（T+1）→ Broker 撮合昨日挂单 → Fill   │
│   ② 收盘：Strategy 读取当日收盘数据 → 生成 Order → 挂到明日撮合        │
│   ③ 日终：Portfolio 按收盘价估值，记录净值                             │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

| 组件 | 职责 | 对应实盘系统中的组件（Phase 13） |
| --- | --- | --- |
| DataFeed | 按时间顺序推送行情 | 行情网关 |
| Strategy | 根据已知信息生成订单 | 策略引擎 |
| Broker | 撮合、成本、约束检查 | 交易网关 + 交易所 |
| Portfolio | 现金、持仓、净值 | 账户/持仓管理、OMS |
| Event Queue | 解耦各组件 | 事件引擎（Phase 13 用 asyncio 实现） |

### 6.3 代码：一个支持 A 股约束的最小引擎

```python
import queue
from dataclasses import dataclass

# ---------- 事件 ----------
@dataclass
class MarketEvent:
    date: pd.Timestamp
    bars: dict          # code -> {open, high, low, close, pre_close, volume}（不复权）
    phase: str          # "open" 或 "close"

@dataclass
class OrderEvent:
    date: pd.Timestamp
    code: str
    qty: int            # > 0 买入，< 0 卖出

@dataclass
class FillEvent:
    date: pd.Timestamp
    code: str
    qty: int
    price: float
    cost: float

# ---------- 数据 ----------
def daily_feed(frames: dict[str, pd.DataFrame]):
    """frames: code -> 不复权日线 DataFrame（index 为交易日）。停牌日无记录。"""
    dates = sorted(set().union(*(f.index for f in frames.values())))
    for d in dates:
        yield d, {c: f.loc[d].to_dict() for c, f in frames.items() if d in f.index}

# ---------- 账户 ----------
class Portfolio:
    def __init__(self, cash: float):
        self.cash = cash
        self.positions: dict[str, int] = {}
        self.sellable: dict[str, int] = {}
        self.last_price: dict[str, float] = {}
        self.history = []

    def on_new_day(self):
        self.sellable = dict(self.positions)            # T+1：昨日及之前的持仓今日可卖

    def on_fill(self, f: FillEvent):
        self.positions[f.code] = self.positions.get(f.code, 0) + f.qty
        if f.qty < 0:
            self.sellable[f.code] = self.sellable.get(f.code, 0) + f.qty
        self.cash -= f.qty * f.price + f.cost

    def mark_to_market(self, date, bars):
        for c, b in bars.items():
            self.last_price[c] = b["close"]             # 停牌股沿用最后价格
        equity = self.cash + sum(q * self.last_price[c] for c, q in self.positions.items())
        self.history.append((date, equity))

# ---------- 模拟券商 ----------
class Broker:
    def __init__(self, limit_pct=0.10, slippage=0.0005):
        self.pending: list[OrderEvent] = []
        self.limit_pct, self.slippage = limit_pct, slippage

    def submit(self, order: OrderEvent):
        self.pending.append(order)

    def match(self, ev: MarketEvent, pf: Portfolio) -> list[FillEvent]:
        fills, cash = [], pf.cash
        for o in sorted(self.pending, key=lambda o: o.qty):   # 先卖后买：卖出资金当日可用
            bar = ev.bars.get(o.code)
            if bar is None or bar["volume"] == 0:              # 停牌：订单作废
                continue
            up, down = limit_prices(bar["pre_close"], self.limit_pct)
            px = bar["open"]
            if o.qty > 0:
                if px >= up:                                   # 开盘涨停：视为买不进
                    continue
                px = min(px * (1 + self.slippage), up)
                qty = min(o.qty, int(cash / (px * 1.001)))     # 预留少量费用
                qty = qty // 100 * 100                         # 100 股整数手
            else:
                if px <= down:                                 # 开盘跌停：视为卖不出
                    continue
                px = max(px * (1 - self.slippage), down)
                qty = -min(-o.qty, pf.sellable.get(o.code, 0)) # T+1：只能卖可卖部分
            if qty == 0:
                continue
            cost = a_share_cost(abs(qty) * px, is_sell=qty < 0)
            cash -= qty * px + cost
            fills.append(FillEvent(ev.date, o.code, qty, px, cost))
        self.pending = []                                      # 简化：未成交订单当日作废
        return fills

# ---------- 策略 ----------
class MACrossStrategy:
    def __init__(self, code: str, n: int = 20, alloc: float = 0.95):
        self.code, self.n, self.alloc = code, n, alloc
        self.closes: list[float] = []                          # 策略自己积累历史：只见过去

    def on_bar(self, ev: MarketEvent, pf: Portfolio) -> list[OrderEvent]:
        bar = ev.bars.get(self.code)
        if bar is None or bar["volume"] == 0:
            return []
        self.closes.append(bar["close"])
        if len(self.closes) < self.n:
            return []
        ma = sum(self.closes[-self.n:]) / self.n
        held = pf.positions.get(self.code, 0)
        if bar["close"] > ma and held == 0:
            return [OrderEvent(ev.date, self.code, int(pf.cash * self.alloc / bar["close"]))]
        if bar["close"] < ma and held > 0:
            return [OrderEvent(ev.date, self.code, -held)]
        return []

# ---------- 引擎主循环 ----------
class Backtester:
    def __init__(self, frames, strategy, broker, portfolio):
        self.frames, self.strategy, self.broker, self.pf = frames, strategy, broker, portfolio
        self.events: queue.Queue = queue.Queue()

    def _dispatch(self, first_event):
        self.events.put(first_event)
        while not self.events.empty():
            ev = self.events.get()
            if isinstance(ev, MarketEvent) and ev.phase == "open":
                self.pf.on_new_day()
                for f in self.broker.match(ev, self.pf):
                    self.events.put(f)
            elif isinstance(ev, MarketEvent) and ev.phase == "close":
                for o in self.strategy.on_bar(ev, self.pf):
                    self.events.put(o)
            elif isinstance(ev, OrderEvent):
                self.broker.submit(ev)                         # 挂到下一交易日开盘撮合
            elif isinstance(ev, FillEvent):
                self.pf.on_fill(ev)

    def run(self) -> pd.Series:
        for date, bars in daily_feed(self.frames):
            self._dispatch(MarketEvent(date, bars, "open"))
            self._dispatch(MarketEvent(date, bars, "close"))
            self.pf.mark_to_market(date, bars)
        return pd.Series(dict(self.pf.history), name="equity")

# 用法：frames = {"600519": maotai_raw_df}（需含 open/high/low/close/pre_close/volume）
# equity = Backtester(frames, MACrossStrategy("600519"), Broker(), Portfolio(1_000_000)).run()
```

### 6.4 这个最小引擎刻意省略了什么

| 省略项 | 真实引擎需要 |
| --- | --- |
| 除权除息 | 不复权价格下，分红要增加现金、送转要增加持股，否则净值在除权日出现假跳水 |
| 未成交订单 | 可选择延续到下一日、按比例部分成交 |
| 成交量约束 | 成交量不超过当日成交量的一定比例 |
| 订单类型 | 限价单、止损单、收盘单 |
| 多策略/多账户 | 资金分配、风控 |
| 日志与审计 | 每笔订单的拒绝原因（涨停、停牌、资金不足、T+1）都要记录，便于对拍 |

**本阶段实战项目**就是在此基础上补齐这些功能，并与 backtrader 的结果**对拍**（相同输入，比较每日持仓与净值，差异必须可解释）。

---

## 7. 向量化 vs 事件驱动：架构对比

### 7.1 对比表

| 维度 | 向量化回测 | 事件驱动回测 |
| --- | --- | --- |
| **执行方式** | 整个矩阵一次计算 | 按时间逐事件推进 |
| **速度** | 极快（适合万级参数组合） | 慢（Python 实现可能慢几十到上百倍） |
| **前视偏差风险** | 高：一个 `shift` 错误就会泄露未来 | 低：结构上只能看到过去 |
| **路径依赖** | 难（止损、加仓、资金约束） | 自然支持 |
| **A 股约束** | 别扭，需要大量掩码矩阵 | 自然地写在 Broker 里 |
| **订单/成交细节** | 只有仓位和收益 | 完整订单、成交、拒单记录 |
| **与实盘代码复用** | 低 | 高：同一套 Strategy 可以接实盘网关 |
| **典型用途** | 因子研究、信号初筛、参数扫描 | 策略精确验证、上线前回测、实盘对照 |
| **代表框架** | pandas 手写、vectorbt | backtrader、vn.py、自研引擎 |

### 7.2 推荐工作流：漏斗式

```
                   想法 / 因子 / 参数组合（成百上千个）
                   ┌──────────────────────────────┐
                   │   向量化回测：快速初筛        │  ← 粗糙但快
                   │   （简化成本，统一假设）       │
                   └──────────────┬───────────────┘
                                  ▼  留下少数候选
                   ┌──────────────────────────────┐
                   │   事件驱动回测：精确验证       │  ← A 股约束、精确成本
                   │   （涨跌停/T+1/整手/停牌）     │
                   └──────────────┬───────────────┘
                                  ▼  结果差异可解释
                   ┌──────────────────────────────┐
                   │   Walk-Forward / 样本外        │
                   └──────────────┬───────────────┘
                                  ▼
                           模拟盘 → 实盘
```

**两个引擎的结果之间的差异本身就是信息**：如果向量化回测年化 30%，事件驱动只有 8%，差额就是 A 股约束和成本吞掉的部分——这说明策略可能严重依赖于无法成交的价格。

### 7.3 混合方案

实践中常见的折中：

- **向量化 + Numba 状态机**：信号向量化计算，持仓状态（止损、T+1）用 Numba 循环（Phase 2 第 11 节）。vectorbt 本质上就是这个思路
- **事件驱动 + 预计算指标**：指标在事件循环外向量化计算好（注意必须只用过去数据），事件循环只做决策和撮合，提速明显

---

## 8. 前视偏差与未来函数

### 8.1 定义

**前视偏差（Look-ahead Bias）**：在回测的 $t$ 时刻，使用了 $t$ 时刻**实际上还无法获得**的信息。国内常把导致前视偏差的代码/指标叫作**"未来函数"**。

**比喻**：拿着明天的报纸去买今天的彩票。回测里你是"中奖专业户"，实盘里你连明天的报纸都没有。

前视偏差是**最致命**的回测错误，因为：

1. 它能把一个毫无价值的策略变成"圣杯"
2. 代码不会报错
3. 结果越好，人越不愿意怀疑

### 8.2 前视偏差的常见来源

| # | 来源 | 示例 | 修复 |
| --- | --- | --- | --- |
| 1 | **信号未滞后** | `signal * ret`，t 日收盘的信号吃 t 日收益 | `signal.shift(1) * ret` |
| 2 | **同一根 K 线的高低点** | 用当日 high 触发止盈、当日 low 触发止损，同时假设两者都按有利顺序发生 | 日线无法知道 high/low 先后，取保守假设或用分钟数据 |
| 3 | **全样本统计量** | 用全样本均值/标准差做 Z-score；全样本分位数做阈值 | `rolling` / `expanding` |
| 4 | **全样本参数估计** | 用全样本回归得到配对交易对冲比例 β | 滚动估计（Phase 3 第 8.5 节） |
| 5 | **前复权价格** | 前复权会随着未来的分红送转改写历史价格；用价格水平做判断（如价格 > 100 元、整手计算、涨跌停）就泄露了未来 | 收益用后复权；价格判断用不复权 |
| 6 | **财务数据按报告期对齐** | 用 2023-12-31 的年报数据在 2024 年 1 月交易 | 按**公告日**对齐（Phase 2 第 5.4 节） |
| 7 | **当前成分股** | 用今天的沪深 300 成分股回测过去十年 | 用历史时点成分股（见第 9 节） |
| 8 | **`bfill` / 插值** | 缺失值用后面的值填充 | 只允许 `ffill` |
| 9 | **`rolling(center=True)`** | 窗口中心对齐，用到了未来一半数据 | 默认右对齐 |
| 10 | **重采样标签错位** | 周线/月线信号对齐回日线时没有 shift | 显式 shift 到下一周期首日 |
| 11 | **机器学习数据泄露** | 随机打乱划分训练/测试集；用全样本做标准化/特征选择 | 时间序列切分、Purged CV（Phase 9） |
| 12 | **修订后的数据** | 宏观数据、财务数据事后修订，回测用的是修订后的值 | 使用 point-in-time 数据库 |

### 8.3 代码示例：三个经典未来函数

```python
# ❌ 未来函数 1：信号与收益同期
ret = close.pct_change()
sig = (close > close.rolling(20).mean()).astype(int)
wrong = (sig * ret)                 # t 日收盘才知道 sig_t，却赚了 t 日全天收益
right = (sig.shift(1) * ret)

# ❌ 未来函数 2：全样本标准化
z_wrong = (close - close.mean()) / close.std()
z_right = (close - close.expanding(250).mean()) / close.expanding(250).std()

# ❌ 未来函数 3：日线止损止盈同时触发时假设先止盈
#    当日 high >= 止盈价 且 low <= 止损价，你不知道哪个先发生
both_hit = (df["high"] >= take_profit) & (df["low"] <= stop_loss)
# 保守处理：both_hit 时按止损计算
```

### 8.4 如何系统地检测前视偏差

**方法一：截断一致性测试（Truncation Test）** —— 最有效的自动化手段

思路：如果信号只依赖过去，那么用"截至 T 的数据"算出的信号，与用"全部数据"算出的信号在 T 之前应该**完全一致**。

```python
def assert_no_lookahead(signal_func, df: pd.DataFrame, cutoffs) -> None:
    """signal_func: DataFrame -> Series。任何截断点之前的信号都不应因后续数据而改变。"""
    full = signal_func(df)
    for cut in cutoffs:
        partial = signal_func(df.loc[:cut])
        pd.testing.assert_series_equal(
            full.loc[:cut], partial, check_names=False,
            obj=f"截断于 {cut} 的信号与全量信号不一致 → 疑似未来函数",
        )

assert_no_lookahead(
    lambda d: (d["close"] > d["close"].rolling(20).mean()).astype(int),
    df, cutoffs=df.index[[300, 800, 1500]],
)
```

这个测试能抓住全样本标准化、`center=True`、`bfill`、全样本回归等大部分问题。**建议写进 pytest，每个信号函数都跑一遍。**

**方法二：好得不真实检查**

- 日线策略扣成本后 Sharpe > 3？先查 bug
- 胜率 > 70% 且盈亏比 > 1？先查 bug
- 收益集中在信号当天？检查是否信号与收益同期

**方法三：随机化检验**

把信号随机打乱或换成随机信号，策略收益应该接近 0（扣成本后为负）。如果随机信号也赚钱，说明收益来自框架本身的 bug（例如成交价使用了未来价格）。

**方法四：逐笔审计**

挑 5–10 笔交易，手工核对：信号在哪一刻产生？用到了哪些数据？那些数据当时是否已经公布？成交价是否可实现？

---

## 9. 幸存者偏差与其他数据偏差

### 9.1 幸存者偏差（Survivorship Bias）

**比喻**：二战时研究返航飞机的弹孔分布，想在弹孔多的地方加装甲。统计学家 Abraham Wald 指出：应该加固**没有弹孔的地方**——因为被打中那些部位的飞机根本没飞回来。只研究"幸存者"，会得出系统性错误的结论。

**A 股中的典型表现**：

| 场景 | 偏差 |
| --- | --- |
| 用**今天的**沪深 300 成分股回测 2015 年至今 | 今天的成分股是"后来变好、市值变大"的公司；被剔除的公司不在样本里 → 高估收益 |
| 股票列表只包含**当前上市**股票 | 退市股票（往往是暴跌后退市）被排除 → 高估收益，尤其高估"低价股""小市值""困境反转"类策略 |
| 只研究"至今仍在运作的基金" | 业绩差而清盘的基金消失了 |
| 期货只用当前活跃品种 | 已停止交易或流动性枯竭的品种被忽略 |

**修复**：

- 使用**历史时点（Point-in-time）成分股**：每个调仓日只能从当时的成分股中选股
- 股票池包含**已退市股票**的完整历史数据（Phase 1 数据仓库要支持这一点；很多免费数据接口默认只返回当前上市股票的列表，需要额外处理）
- 在回测中正确处理退市：退市日之后持仓如何处置（按最后价格、按退市整理期价格、或计为大幅损失），要有明确假设

```python
# ❌ 幸存者偏差
universe = get_current_hs300_constituents()            # 今天的成分股
# ✅ 历史时点成分股
def universe_at(date):
    h = hs300_history      # 列：code, in_date, out_date（仍在指数中则 out_date 为 NaT）
    mask = (h["in_date"] <= date) & (h["out_date"].isna() | (h["out_date"] > date))
    return h.loc[mask, "code"]
```

### 9.2 其他数据偏差

| 偏差 | 说明 | 对策 |
| --- | --- | --- |
| **数据窥探（Data Snooping）** | 在同一份数据上反复研究、反复修改，最终"找到"的规律可能只是这份数据的特征 | 预留从未碰过的最终样本外；记录所有尝试 |
| **回填偏差（Backfill Bias）** | 数据库在某个时点才收录某只证券/基金，但把它之前的历史也补了进来 | 注意数据的"入库时间" |
| **时间段选择偏差** | 只展示策略表现好的区间 | 报告完整区间；按年分段展示 |
| **新股不可得** | 回测中"买入"上市初期的新股，实际中很难按该价格获得 | 剔除上市初期一段时间的股票 |
| **流动性偏差** | 回测中买入成交额极小的股票 | 设置成交额/流通市值门槛 |
| **"故事偏差"** | 先看到结果，再编一个漂亮的经济学解释 | 先写假设，再跑回测 |

---

## 10. 过拟合与多重检验

### 10.1 直觉

**过拟合（Overfitting）**：策略记住了历史数据中的**噪声**，而不是学到了可重复的**规律**。

**比喻**：一个学生把历年考试真题的答案全背下来（ABCDBCA...），在这些真题上次次满分。但他什么都没学会，换一套新题就只能乱猜。回测中的过拟合策略，就是"背答案"的学生。

**为什么量化中过拟合特别严重**：Phase 3 告诉我们，金融数据信噪比极低。在低信噪比环境中，**噪声的"规律"远比真实规律多、也更容易被找到**。

### 10.2 多重检验：试得越多，越容易"找到"圣杯

Phase 3 讲过：在 5% 显著性下，每检验一个无效策略，有 5% 的概率得到"显著"结果。试 $M$ 次，至少一次"显著"的概率为：

$$
P(\text{至少一次假阳性}) = 1 - (1 - \alpha)^M
$$

- $\alpha$：单次检验的显著性水平（如 0.05）
- $M$：独立检验次数

| 尝试次数 $M$ | 至少一次假阳性的概率（$\alpha = 0.05$） |
| --- | --- |
| 1 | 5% |
| 10 | 40% |
| 20 | 64% |
| 100 | 99.4% |

**代码演示：从纯噪声中"挖"出高 Sharpe 策略**

```python
rng = np.random.default_rng(42)
n_days, n_strategies = 252 * 3, 1000

# 1000 个"策略"，日收益全是均值为 0 的纯噪声（真实 Sharpe = 0）
noise = rng.normal(0, 0.01, size=(n_days, n_strategies))
sharpes = noise.mean(axis=0) / noise.std(axis=0) * np.sqrt(252)

print(f"最好的 Sharpe: {sharpes.max():.2f}")              # 通常在 2 附近甚至更高
print(f"Sharpe > 1 的策略数: {(sharpes > 1).sum()}")
```

如果你只汇报"最好的那个"，你就在汇报运气。

### 10.3 参数优化就是多重检验

你可能觉得"我没有测试 1000 个策略，只是优化了均线参数"。但：

```
快线 5~60 × 慢线 10~250 × 止损 1%~10% × 持仓周期 ...
= 成千上万次"隐性检验"
```

每一组参数都是一次检验。**研究员的每一次"改一下再试试"，都是一次检验**——包括那些失败后被删掉的代码。这就是课程通用验收标准要求"每次参数尝试都记录在研究日志中，报告时披露尝试次数"的原因。

### 10.4 多重检验修正方法

| 方法 | 控制目标 | 做法 | 特点 |
| --- | --- | --- | --- |
| **Bonferroni** | 族错误率 FWER（至少一个假阳性的概率） | 显著性门槛变为 $\alpha / M$ | 简单、非常保守 |
| **Holm** | FWER | p 值从小到大逐步比较 | 比 Bonferroni 略宽松，仍严格 |
| **Benjamini-Hochberg (BH)** | 错误发现率 FDR（被判显著的结果中假阳性的比例） | 排序后比较 $p_{(k)} \le \frac{k}{M}\alpha$ | 适合大量因子筛选 |
| **提高 t 值门槛** | — | 如要求 t > 3（Harvey, Liu & Zhu, 2016） | 经验法则 |
| **Deflated Sharpe Ratio** | 考虑试验次数、偏度、峰度 | 见 10.5 | 专为策略选择设计 |

```python
from statsmodels.stats.multitest import multipletests

# pvals: 每个策略/因子 t 检验的 p 值
reject_bonf, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
reject_holm, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
print(f"未修正显著: {(np.array(pvals) < 0.05).sum()}, "
      f"Bonferroni: {reject_bonf.sum()}, Holm: {reject_holm.sum()}, BH: {reject_bh.sum()}")
```

> 注意：这些方法通常假设检验之间独立或特定的相关结构。参数相近的策略高度相关，"有效独立试验次数"小于名义次数，严格估计需要聚类等方法（参见《Advances in Financial Machine Learning》相关章节）。

### 10.5 Deflated Sharpe Ratio（DSR）

Bailey 与 López de Prado 提出的思路：**在 $N$ 次独立试验中，即使所有策略真实 Sharpe 都为 0，最大的那个 Sharpe 也会显著大于 0**。所以评判"最好的策略"时，比较基准不应是 0，而应是"纯运气下预期的最大 Sharpe"。

**纯运气下的期望最大 Sharpe**（近似）：

$$
\text{SR}_0 = \sqrt{V[\widehat{\text{SR}}]}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{N e}\right)\right]
$$

- $N$：独立试验次数
- $V[\widehat{\text{SR}}]$：各次试验 Sharpe 估计值的方差
- $\gamma \approx 0.5772$：欧拉-马歇罗尼常数
- $\Phi^{-1}$：标准正态分布的分位数函数；$e$：自然常数

**Probabilistic Sharpe Ratio（PSR）**——观测 Sharpe 显著超过基准 $\text{SR}^*$ 的概率：

$$
\text{PSR}(\text{SR}^*) = \Phi\left(\frac{(\widehat{\text{SR}} - \text{SR}^*)\sqrt{T - 1}}{\sqrt{1 - \hat{\gamma}_3\,\widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}}\right)
$$

- $\widehat{\text{SR}}$：观测到的**未年化**（与 $T$ 同频率）Sharpe
- $T$：收益样本数
- $\hat{\gamma}_3$：收益偏度；$\hat{\gamma}_4$：收益峰度（**非超额**，正态为 3）
- 负偏、肥尾会增大分母 → 降低 PSR

**DSR = PSR(SR₀)**：把基准设为"纯运气下的期望最大 Sharpe"。DSR 接近 1 才说明策略大概率不是运气。

```python
from scipy import stats

def psr(ret: np.ndarray, sr_benchmark: float = 0.0) -> float:
    sr = ret.mean() / ret.std(ddof=1)                     # 未年化
    skew = stats.skew(ret)
    kurt = stats.kurtosis(ret, fisher=False)              # 非超额峰度
    T = len(ret)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return stats.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom)

def expected_max_sr(trial_srs: np.ndarray) -> float:
    """trial_srs: 所有试验的未年化 Sharpe"""
    N, g = len(trial_srs), 0.5772156649
    return np.sqrt(trial_srs.var(ddof=1)) * (
        (1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e))
    )

daily_srs = noise.mean(axis=0) / noise.std(axis=0, ddof=1)
best = noise[:, daily_srs.argmax()]
print("PSR(0):", psr(best))                               # 看起来很显著
print("DSR   :", psr(best, expected_max_sr(daily_srs)))   # 扣除"挑选"效应后，不再显著
```

这部分内容在 Phase 9 的 `08-deflated-sharpe-pbo.py` 中会更深入展开，包括 **PBO（Probability of Backtest Overfitting，回测过拟合概率）**：通过把样本切成多块、组合成多种样本内/外划分，统计"样本内最优的参数在样本外排名低于中位数"的频率。

### 10.6 过拟合的识别信号

| 信号 | 说明 |
| --- | --- |
| 参数"尖峰" | 最优参数周围稍微变动，绩效急剧下降（Phase 5 会讲"参数平原"） |
| 规则过多 | 一个策略有十几个条件和阈值 |
| 样本外大幅衰减 | 样本内 Sharpe 2.0，样本外 0.3 |
| 只在特定区间有效 | 收益集中在某一年或某几个月 |
| 无法用经济逻辑解释 | "每月第三个周二买入市盈率在 13.7 到 15.2 之间的股票" |
| 交易次数很少 | 30 笔交易的胜率和盈亏比没有统计意义 |

### 10.7 对抗过拟合的原则

1. **先有假设，后有回测**：经济逻辑先行
2. **少参数**：每多一个参数，就多一个过拟合的自由度
3. **记录所有尝试**：披露尝试次数，用 DSR 等方法修正
4. **保留最终样本外**：一段数据在研究结束前**绝对不碰**，只用一次
5. **参数稳健性**：选"参数平原"的中心，而不是最高点
6. **跨市场/跨品种验证**：在 A 股有效的逻辑，在其他相似市场、其他品种上是否也大致成立

---

## 11. Walk-Forward 与样本外检验

### 11.1 直觉

**样本内（In-Sample, IS）**：用于开发和优化策略的数据。
**样本外（Out-of-Sample, OOS）**：策略开发过程中**没有看过**的数据，用于检验。

**比喻**：样本内是"练习题"，样本外是"考试"。只看练习成绩的学生，你无法知道他是真懂还是背了答案。

### 11.2 简单划分的问题

```
|←──────── 样本内 70% ────────→|←── 样本外 30% ──→|
2010                          2019              2024
```

问题：

- 只有一次"考试"，结果受该时段市场状态影响很大
- 研究员看到样本外结果不好后，**回去改策略再测**——样本外就变成了样本内（"样本外污染"）
- 没有模拟"定期重新优化参数"的真实过程

### 11.3 Walk-Forward 分析

**Walk-Forward（滚动前推）**：模拟真实的研究-部署循环——用过去一段数据优化参数，在紧接着的下一段数据上测试，然后窗口前移，重复。

```
滚动窗口（Rolling）：训练窗口长度固定
  窗口1: [══ 训练 ══][─测试─]
  窗口2:      [══ 训练 ══][─测试─]
  窗口3:           [══ 训练 ══][─测试─]
  窗口4:                [══ 训练 ══][─测试─]
         ──────────────────────────────────────▶ 时间
  拼接所有"测试"段 → 一条完整的样本外净值曲线

锚定窗口（Anchored / Expanding）：训练窗口起点固定，不断扩展
  窗口1: [══ 训练 ══][─测试─]
  窗口2: [══════ 训练 ══════][─测试─]
  窗口3: [══════════ 训练 ══════════][─测试─]
```

| 类型 | 优点 | 缺点 |
| --- | --- | --- |
| 滚动窗口 | 适应市场状态变化 | 训练数据较少，参数可能不稳定 |
| 锚定窗口 | 训练数据越来越多，估计更稳定 | 对远古数据和近期数据一视同仁，适应慢 |

### 11.4 代码

```python
def walk_forward(close: pd.Series, param_grid, train_days=500, test_days=120, cost=0.001):
    """对均线参数做 Walk-Forward，返回拼接的样本外收益与每段选中的参数"""
    oos_rets, chosen = [], []
    start = train_days
    while start + test_days <= len(close):
        # 训练段：只用 [start - train_days, start) 的数据选参数
        train = close.iloc[start - train_days: start]
        best_p, best_sr = None, -np.inf
        for n in param_grid:
            sig = (train > train.rolling(n).mean()).astype(float)
            r = vectorized_backtest(train, sig, cost)["ret"].iloc[n:]
            sr = r.mean() / r.std() if r.std() > 0 else -np.inf
            if sr > best_sr:
                best_p, best_sr = n, sr

        # 测试段：用训练段末尾的数据预热指标，避免测试段开头为 NaN
        warm = close.iloc[start - best_p: start + test_days]
        sig = (warm > warm.rolling(best_p).mean()).astype(float)
        test_ret = vectorized_backtest(warm, sig, cost)["ret"].iloc[best_p:]
        oos_rets.append(test_ret)
        chosen.append((close.index[start], best_p))
        start += test_days

    return pd.concat(oos_rets), pd.DataFrame(chosen, columns=["测试段起点", "选中参数"])

oos_ret, params = walk_forward(close, param_grid=range(10, 121, 10))
print(performance_report(oos_ret))
print(params)     # 参数是否在各段之间剧烈跳动？跳动越大越可疑
```

> 测试段使用训练段末尾的数据为指标"预热"是合法的——这些数据在测试段开始前确实已经存在。不合法的是用测试段的数据来**选择参数**。

### 11.5 如何解读 Walk-Forward 结果

| 观察 | 解读 |
| --- | --- |
| 拼接的样本外 Sharpe 接近样本内 | 策略较稳健 |
| 样本外远低于样本内 | 过拟合，或市场状态变化 |
| **Walk-Forward 效率**（WFE = 样本外年化收益 / 样本内年化收益） | 经验上过低说明样本内结果大部分是过拟合；没有统一标准阈值 |
| 选中参数在各段间剧烈跳动 | 参数没有稳定的最优值 → 优化可能只是在拟合噪声 |
| 样本外只在少数几段赚钱 | 收益依赖特定行情 |

### 11.6 进阶：Purging、Embargo 与组合交叉验证

当标签涉及**未来一段时间**（如"未来 20 日收益"）时，训练集末尾的样本标签会与测试集开头的时间重叠，产生信息泄露。《Advances in Financial Machine Learning》提出：

- **Purging（清洗）**：从训练集中删除标签时间区间与测试集重叠的样本
- **Embargo（禁运期）**：在测试集之后再额外留出一段间隔，不用于训练
- **CPCV（Combinatorial Purged Cross-Validation）**：把数据切成多块并组合成多种训练/测试划分，得到**多条**样本外路径，而不是 Walk-Forward 的一条，从而得到绩效指标的分布

这些方法在 Phase 9（机器学习选股）中是必需的，这里先建立概念。

### 11.7 最终样本外：只能用一次

```
|←──────── 研究区（IS + Walk-Forward） ────────→|←── 保留区 ──→|
                                                   研究全部结束后
                                                   只跑一次，结果如实报告
```

如果在保留区跑完后又回去改策略，保留区就失效了。**纪律比方法更重要。**

---

## 12. 开源框架：backtrader 与 vectorbt

### 12.1 框架对比

| 维度 | backtrader | vectorbt（开源版） |
| --- | --- | --- |
| 架构 | 事件驱动 | 向量化（底层大量使用 Numba） |
| 速度 | 慢 | 很快，适合大规模参数扫描 |
| 学习曲线 | 中等，概念多（Cerebro、Strategy、Broker、Sizer、Analyzer） | 较陡，API 风格独特 |
| A 股约束 | 无内置，需自定义佣金类、Sizer 和下单前检查 | 无内置，需在信号层面处理 |
| 维护状态 | 原作者已长期不活跃，新版本 Python/依赖下可能需要处理兼容问题 | 开源版以维护为主，作者另有商业版 vectorbt PRO |
| 适合 | 学习事件驱动架构、单策略精确回测、与自研引擎对拍 | 参数扫描、信号研究、快速原型 |

> 国内也有 vn.py（Phase 13 会用于实盘）、聚宽/米筐等平台内置的回测框架，各有对 A 股规则的内置支持程度，使用前请阅读其文档确认撮合规则。

### 12.2 backtrader 基本结构

```
┌──────────────────────────────────────────────┐
│                  Cerebro（大脑）              │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Data Feed│ │ Strategy │ │ Broker       │  │
│  │ PandasData│ │ next()   │ │ 现金/佣金/撮合│  │
│  └──────────┘ └──────────┘ └──────────────┘  │
│  ┌──────────┐ ┌──────────────────────────┐   │
│  │ Sizer    │ │ Analyzers（Sharpe/回撤…） │   │
│  └──────────┘ └──────────────────────────┘   │
└──────────────────────────────────────────────┘
```

```python
import backtrader as bt

class AShareCommission(bt.CommInfoBase):
    """A 股佣金：双向佣金（含最低收费）+ 卖出印花税。费率为示例值。"""
    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),            # commission 以小数表示（0.00025 而不是 0.025%）
        ("commission", 0.00025),
        ("stamp_duty", 0.0005),
        ("min_comm", 5.0),
    )

    def _getcommission(self, size, price, pseudoexec):
        value = abs(size) * price
        comm = max(value * self.p.commission, self.p.min_comm)
        if size < 0:                  # 卖出加收印花税
            comm += value * self.p.stamp_duty
        return comm

class LotSizer(bt.Sizer):
    """按可用资金比例买入，取整到 100 股"""
    params = (("pct", 0.95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return int(cash * self.p.pct / data.close[0] // 100 * 100)
        return self.broker.getposition(data).size

class SmaCross(bt.Strategy):
    params = (("n", 20),)

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=self.p.n)

    def next(self):
        # data.close[0] 是当前 bar，[-1] 是前一根 —— 注意与 pandas 的负索引含义不同
        if not self.position and self.data.close[0] > self.sma[0]:
            self.buy()                # 默认市价单：在下一根 bar 开盘成交
        elif self.position and self.data.close[0] < self.sma[0]:
            self.close()

cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=df))   # df: DatetimeIndex + open/high/low/close/volume
cerebro.addstrategy(SmaCross)
cerebro.broker.setcash(1_000_000)
cerebro.broker.addcommissioninfo(AShareCommission())
cerebro.addsizer(LotSizer)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
result = cerebro.run()[0]
print(result.analyzers.sharpe.get_analysis(), result.analyzers.dd.get_analysis().max.drawdown)
```

**backtrader 使用要点**：

- **默认成交时点**：在 `next()` 中下的市价单，默认在**下一根 bar 的开盘价**成交——这天然避免了"同 bar 成交"的前视偏差。`cheat_on_close` / `cheat_on_open` 等选项会改变这一行为，开启前要明白其含义
- **涨跌停与停牌**：没有内置，需要在 `next()` 中判断（例如下一根 bar 开盘涨停则撤单），或自定义撮合逻辑。注意判断时同样要用不复权价格
- **T+1**：如上文 5.2 所述，日线 + 次日开盘成交时基本自动满足；若使用日内数据需自行限制
- **分析器结果与自研实现对拍**：Sharpe 的年化方式、无风险利率默认值可能与你的实现不同

### 12.3 vectorbt：大规模参数扫描

```python
import vectorbt as vbt

# close: 后复权收盘价 Series
windows = np.arange(5, 121, 5)
fast_ma, slow_ma = vbt.MA.run_combs(close, window=windows, r=2, short_names=["fast", "slow"])
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# ⚠️ from_signals 默认在信号出现的同一根 bar 以 close 成交
#    更保守：把信号整体后移一根 bar，相当于"次日收盘成交"
entries = entries.shift(1).fillna(False).astype(bool)
exits = exits.shift(1).fillna(False).astype(bool)

pf = vbt.Portfolio.from_signals(close, entries, exits,
                                fees=0.0005, slippage=0.001, freq="1D", init_cash=1_000_000)

sharpe = pf.sharpe_ratio()                  # 每个 (fast, slow) 组合一个 Sharpe
print(sharpe.sort_values(ascending=False).head(10))
print("尝试的组合数:", len(sharpe))          # ← 这个数字要写进研究日志！
```

**用 vectorbt 时最需要警惕的**：

- **扫描几百个参数组合只需几秒——这恰恰让过拟合变得前所未有地容易**。扫描完后，应该：
  - 画出参数热力图，看最优点周围是"平原"还是"尖峰"
  - 用 Walk-Forward 而不是全样本最优参数评估
  - 用 DSR 修正"从 N 个组合中挑最优"的选择偏差
- **A 股约束不在默认模型中**：涨跌停、T+1、整手需要通过信号掩码或自定义模拟函数处理
- **`from_signals` 的 `fees` 不区分买卖方向**，无法直接表达"只在卖出收印花税"，需要自行近似（例如把印花税平摊到双边）或使用更底层的自定义订单接口

---

## 13. 最佳实践与常见陷阱

### 13.1 回测报告必须包含的内容

对应课程通用验收标准：

```
┌──────────────────────────────────────────────────────────────────┐
│                       回测报告检查清单                             │
├──────────────────────────────────────────────────────────────────┤
│  ☐ 样本区间（起止日期）、股票池定义（是否历史时点成分股）         │
│  ☐ 数据来源与复权方式；是否包含退市股票                           │
│  ☐ 成交价假设（收盘/次日开盘/VWAP）                               │
│  ☐ 交易成本假设（佣金、印花税、过户费、滑点），并标注执行时费率   │
│  ☐ A 股约束处理：T+1、涨跌停、停牌、整数手                       │
│  ☐ 基准对比（沪深 300 / 中证 500 等，说明是否为全收益指数）       │
│  ☐ 绩效指标：CAGR、波动、Sharpe（含 t 值）、Sortino、MDD、Calmar  │
│  ☐ 换手率与成本敏感性分析（不同成本下的 Sharpe）                  │
│  ☐ 样本外 / Walk-Forward 结果                                     │
│  ☐ 参数尝试次数，以及多重检验修正（如 DSR）                       │
│  ☐ 分年度表现、最大回撤区间                                       │
│  ☐ 向量化与事件驱动结果的差异及解释                               │
└──────────────────────────────────────────────────────────────────┘
```

### 13.2 最佳实践

- **把回测引擎当作生产代码对待**：单元测试、截断一致性测试、与第三方框架对拍
- **先写"必然亏钱"的测试**：随机信号扣成本后应该亏钱；"买入持有"的回测结果应该与指数/个股收益一致
- **结果越好越要怀疑**：Sharpe 异常高时，第一反应是查 bug，而不是庆祝
- **研究日志**：每次尝试（包括失败的）记录日期、假设、参数、结果
- **成本宁可高估**：实盘成本几乎总是比回测假设更高
- **分清研究代码与回测框架**：信号函数应是纯函数（输入历史数据，输出信号），便于做前视测试与在不同引擎间复用

### 13.3 常见陷阱速查表

| # | 陷阱 | 症状 | 对策 |
| --- | --- | --- | --- |
| 1 | 信号未滞后 | Sharpe 异常高 | `shift(1)`；截断一致性测试 |
| 2 | 同 bar 高低点顺序假设 | 止盈止损策略胜率极高 | 保守假设或用分钟数据 |
| 3 | 前复权做价格判断 | 整手/涨跌停/价格阈值出错 | 收益用后复权，判断用不复权 |
| 4 | 当前成分股回测 | 收益虚高 | 历史时点成分股 + 退市股票 |
| 5 | 忽略涨跌停 | 追涨/抄底策略收益虚高 | 涨停不买、跌停不卖 |
| 6 | 忽略停牌 | 停牌期间"成功交易" | 停牌订单作废/延续 |
| 7 | 小数股 | 小资金结果失真 | 100 股整数手 |
| 8 | 忽略最低佣金 | 小资金多股票成本低估 | 建模最低收费 |
| 9 | 成本只按一个值 | 不知道策略对成本多敏感 | 成本敏感性分析 |
| 10 | 全样本参数优化 | 样本外崩溃 | Walk-Forward |
| 11 | 不披露尝试次数 | 无法评估过拟合 | 研究日志 + DSR |
| 12 | 反复使用样本外 | 样本外变样本内 | 保留区只用一次 |
| 13 | 跨期费率不变 | 早期成本低估 | 按历史时点费率 |
| 14 | 交易次数太少 | 指标无统计意义 | 报告交易次数与 t 值 |

---

## 14. 总结

### 14.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                     Phase 4 核心概念地图                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📌 回测定位：证伪工具，不是证明工具                              │
│                                                                   │
│  📌 两种架构                                                      │
│     ├── 向量化：快、适合初筛；shift 错误 = 前视偏差               │
│     └── 事件驱动：慢、真实；结构上无法偷看未来；可复用到实盘      │
│                                                                   │
│  📌 绩效指标                                                      │
│     └── Sharpe(+t 值) / Sortino / MDD / Calmar / 胜率×盈亏比 / 换手│
│                                                                   │
│  📌 现实约束                                                      │
│     ├── 成本：佣金(最低收费) + 印花税(卖出) + 过户费 + 滑点 + 冲击 │
│     ├── 成本敏感性 + 盈亏平衡成本                                 │
│     └── A 股：T+1、涨跌停买卖不得、停牌、100 股整数手             │
│                                                                   │
│  📌 回测陷阱                                                      │
│     ├── 前视偏差：截断一致性测试是最佳自动化武器                  │
│     ├── 幸存者偏差：历史时点成分股 + 退市股票                     │
│     ├── 过拟合 & 多重检验：Bonferroni/Holm/BH、DSR、记录尝试次数  │
│     └── Walk-Forward：模拟"优化-部署"循环，拼接样本外曲线         │
│                                                                   │
│  📌 框架：backtrader（事件驱动/对拍） vectorbt（参数扫描/警惕过拟合）│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 学习检查对照

| README 学习检查 | 对应章节 |
| --- | --- |
| 能手写一个事件驱动回测循环 | 6.2–6.3 |
| 能找出一段代码中的未来函数 | 8.2–8.4 |
| 回测报告包含成本敏感性分析 | 4.5–4.6、13.1 |
| 会用 Walk-Forward 评估策略 | 11.3–11.5 |

### 14.3 与下一阶段的衔接

到这里，课程的**第一部分"基础能力"** 就结束了：

```
Phase 1 数据  →  Phase 2 度量  →  Phase 3 统计  →  Phase 4 回测
"拿到干净数据"   "算对指标"       "判断真假"       "可信地模拟交易"
```

**Phase 5：择时与 CTA 策略** 开始进入**策略研究**。你将用本阶段的回测引擎检验双均线、布林带、MACD/RSI、通道突破、海龟交易法则等经典策略，并在期货上构建多品种 CTA 组合。届时本阶段的每一个概念都会被实际用到：

- **期货的不同约束**：T+0、可做空、保证金与杠杆、主力合约**展期**——你的回测引擎需要扩展以支持这些
- **ATR 仓位与波动率目标**：把 Phase 2 的 ATR、Phase 3 的 GARCH 用于仓位管理
- **参数优化与参数平原**：本阶段的过拟合与多重检验知识的直接应用
- **稳健性检验**：参数扰动、分段检验、跨品种检验——Walk-Forward 思想的延伸

Phase 4 教你"如何不被回测欺骗"；从 Phase 5 开始，你要用这套方法去寻找真正经得起检验的策略。

---

> 💡 **提示**：本文中涉及的 A 股交易规则（T+1、涨跌幅限制、申报数量）与费率（佣金、印花税、过户费）仅用于帮助建模，规则和费率会调整，**请以交易所、中国结算与券商的最新规则为准**。代码中的费率参数均为示例值。
