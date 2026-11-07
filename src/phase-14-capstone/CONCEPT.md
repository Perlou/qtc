# Phase 14: 综合项目 - 核心概念深度解析

> **学习目标**：整合全部知识，构建个人量化平台，完成职业化输出
>
> **前置知识**：Phase 1–13 全部内容。本阶段不引入新的金融理论，重点是**系统整合、研究表达与职业准备**。
>
> **阅读建议**：前半部分（第 2–5 节）是工程设计，后半部分（第 6–8 节）是研究写作、尽调和面试。建议先画出自己的平台架构图，再对照本文检查遗漏。

---

## 目录

1. [引言：从课程到平台](#1-引言从课程到平台)
2. [mini-quant 平台架构设计](#2-mini-quant-平台架构设计)
3. [数据层与因子层](#3-数据层与因子层)
4. [模型层与组合层](#4-模型层与组合层)
5. [执行层与监控面板](#5-执行层与监控面板)
6. [量化研究报告写作](#6-量化研究报告写作)
7. [策略尽调与上线评审](#7-策略尽调与上线评审)
8. [量化岗位与面试](#8-量化岗位与面试)
9. [最佳实践与常见陷阱](#9-最佳实践与常见陷阱)
10. [总结](#10-总结)

---

## 1. 引言：从课程到平台

### 1.1 前 13 个阶段在平台中的位置

```
┌──────────────────────────────────────────────────────────────────┐
│               mini-quant：把 13 个阶段串成一条流水线               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   数据层  ◀── Phase 1 数据获取/复权/存储、Phase 2 时间序列          │
│     │                                                            │
│   因子层  ◀── Phase 7 因子构建与预处理、Phase 11 高频因子低频化      │
│     │                                                            │
│   模型层  ◀── Phase 9 机器学习、Phase 10 深度学习/LLM              │
│     │                                                            │
│   组合层  ◀── Phase 8 组合优化与风险模型                           │
│     │                                                            │
│   回测层  ◀── Phase 4 回测引擎、Phase 3 统计检验                   │
│     │                                                            │
│   执行层  ◀── Phase 11 执行算法、Phase 13 网关/OMS/风控             │
│     │                                                            │
│   监控层  ◀── Phase 13 监控告警、对账                              │
│                                                                  │
│   策略库：Phase 5 CTA、Phase 6 统计套利、Phase 12 期权（可选插件）   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 本阶段的三个交付物

| 交付物 | 验收标准 | 对应章节 |
| --- | --- | --- |
| mini-quant 平台 | 一键从数据更新跑到生成调仓指令 | 第 2–5 节 |
| 策略研究报告 | 可复现，能用 5 分钟讲清收益来源与失效条件 | 第 6–7 节 |
| 面试准备 | 完成 20 道量化面试题 | 第 8 节 |

---

## 2. mini-quant 平台架构设计

> 对应课程：`01-platform-architecture.py`

### 2.1 分层架构

**比喻**：量化平台像一条汽车生产线。每个工位（层）只关心"接收什么零件、交付什么零件"，零件规格（数据契约）是固定的。你可以替换某个工位的机器（如把线性模型换成 LightGBM），而不影响其他工位。

```
┌────────────────────────────────────────────────────────────────────┐
│                        mini-quant 分层架构                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────┐  行情/财务   ┌──────────┐  因子矩阵   ┌──────────┐      │
│  │  数据层   │ ──────────▶ │  因子层   │ ─────────▶ │  模型层   │      │
│  │ Data     │  (PIT 对齐)  │ Factor   │ (预处理后)  │ Model    │      │
│  └──────────┘             └──────────┘            └────┬─────┘      │
│                                                        │ 预测分数    │
│                                                        ▼            │
│  ┌──────────┐  绩效/归因   ┌──────────┐  目标权重   ┌──────────┐      │
│  │  回测层   │ ◀────────── │  组合层   │ ◀───────── │          │      │
│  │ Backtest │             │ Portfolio│            └──────────┘      │
│  └──────────┘             └────┬─────┘                              │
│                                │ 目标权重 → 调仓指令                  │
│                                ▼                                    │
│                          ┌──────────┐  订单/成交   ┌──────────┐      │
│                          │  执行层   │ ──────────▶ │  监控层   │      │
│                          │ Execution│             │ Monitor  │      │
│                          └──────────┘             └──────────┘      │
│                                                                    │
│  横切关注点：配置（YAML）· 日志 · 交易日历 · 版本与可复现 · 测试        │
└────────────────────────────────────────────────────────────────────┘
```

**关键设计**：回测层与执行层**消费同一份组合层输出**。回测用模拟撮合，实盘用 Phase 13 的网关——这保证了"回测的策略"就是"实盘的策略"。

### 2.2 设计原则

| 原则 | 说明 | 违反时的后果 |
| --- | --- | --- |
| 数据契约优先 | 层间只通过约定格式的 DataFrame/Parquet 交互 | 层间耦合，改一处坏一片 |
| Point-in-Time | 每个数据点都带"可获得时间"，任何层只能使用当时可得的数据 | 前视偏差（Phase 4） |
| 可复现 | 记录数据版本、代码版本、配置、随机种子 | 报告结果无法复现，无法尽调 |
| 配置驱动 | 因子列表、模型参数、约束、调仓频率写在配置文件中 | 每次实验都改代码，难以追溯 |
| 每层可独立运行 | 每层都能单独执行与测试，中间结果落盘 | 调试一次要跑完整流水线 |
| 失败可见 | 任何一层失败都中止流水线并告警，不生成"半成品"指令 | 用过期数据生成错误调仓 |

### 2.3 目录结构示例

```
mini-quant/
├── config/
│   ├── universe.yaml          # 股票池、基准
│   ├── factors.yaml           # 启用的因子与参数
│   └── strategy.yaml          # 模型、组合约束、调仓频率
├── mq/
│   ├── data/                  # 数据层：下载、清洗、复权、存储
│   ├── factors/               # 因子层：因子注册表、预处理
│   ├── models/                # 模型层：线性合成、LightGBM
│   ├── portfolio/             # 组合层：权重生成、约束优化
│   ├── backtest/              # 回测层：事件驱动/向量化回测、绩效
│   ├── execution/             # 执行层：调仓指令、网关适配
│   ├── monitor/               # 监控层：日报、面板、告警
│   └── pipeline.py            # 流水线编排
├── store/                     # Parquet 数据与中间结果（不入 git）
├── reports/                   # 研究报告与回测输出
└── tests/
```

### 2.4 模块接口设计

用 `typing.Protocol` 定义每层的接口（类似 Go 的 interface / TypeScript 的 interface），具体实现可以自由替换：

```python
from typing import Protocol
import pandas as pd

# 约定：长表格式，索引为 (date, symbol)
FactorFrame = pd.DataFrame        # 列 = 因子名
ScoreSeries = pd.Series           # 预测分数
WeightSeries = pd.Series          # 目标权重，和为 1（或给定仓位）

class DataSource(Protocol):
    def update(self, end: str) -> None: ...
    def load_prices(self, start: str, end: str) -> pd.DataFrame: ...

class FactorEngine(Protocol):
    def compute(self, date: str) -> FactorFrame: ...

class Model(Protocol):
    def fit(self, X: FactorFrame, y: pd.Series) -> None: ...
    def predict(self, X: FactorFrame) -> ScoreSeries: ...

class PortfolioConstructor(Protocol):
    def target_weights(self, scores: ScoreSeries, current: WeightSeries) -> WeightSeries: ...

class Executor(Protocol):
    def rebalance(self, target: WeightSeries, capital: float) -> list[dict]: ...
```

### 2.5 一键流水线

学习检查要求"平台能一键从数据更新跑到生成调仓指令"。流水线编排的核心是**按顺序执行、每步校验、失败即停**：

```python
import logging
import pandas as pd

log = logging.getLogger("pipeline")

class PipelineError(RuntimeError):
    pass

def check(cond: bool, msg: str) -> None:
    if not cond:
        raise PipelineError(msg)

def run_daily(date: str, data, factors, model, portfolio, executor,
              current_weights: pd.Series, capital: float) -> list[dict]:
    data.update(end=date)                                     # 1. 数据更新
    prices = data.load_prices(start=date, end=date)
    check(not prices.empty, f"{date} 无行情数据，可能数据源未更新")

    X = factors.compute(date)                                 # 2. 因子计算
    check(X.notna().mean().min() > 0.8, "因子缺失率过高")

    scores = model.predict(X)                                 # 3. 模型预测
    check(scores.std() > 0, "预测分数无区分度，模型可能异常")

    target = portfolio.target_weights(scores, current_weights)  # 4. 组合构建
    check(abs(target.sum() - 1) < 1e-6, "目标权重之和不为 1")
    check(target.max() <= 0.05 + 1e-9, "单票权重超限")

    orders = executor.rebalance(target, capital)              # 5. 调仓指令
    log.info("生成 %d 条调仓指令", len(orders))
    return orders
```

> 运行方式：Phase 13 的定时任务在每个交易日收盘后调用 `run_daily`，生成次日调仓指令；指令先人工复核（或经风控检查），再交给执行层。

---

## 3. 数据层与因子层

> 对应课程：`02-data-and-factor-layer.py`

### 3.1 数据层职责

| 职责 | 要点 | 来自 |
| --- | --- | --- |
| 增量下载 | 只拉取上次更新之后的数据，失败可重试 | Phase 1 |
| 复权 | 保存不复权价格 + 复权因子，按需计算前/后复权 | Phase 1 |
| Point-in-Time 财务数据 | 以公告日而非报告期对齐 | Phase 7 |
| 股票池 | 每日成分（含已退市股票），避免幸存者偏差 | Phase 1/4 |
| 交易状态 | 停牌、涨跌停、ST 标记，供回测与执行使用 | Phase 4 |
| 质量检查 | 缺失、重复、价格跳变、成交量为 0 | Phase 1 |
| 存储 | Parquet 分区 + DuckDB 查询 | Phase 1 |

### 3.2 因子层：注册表模式

**比喻**：因子注册表像插件市场。每个因子是一个独立插件，通过装饰器"注册"到平台，配置文件决定启用哪些。新增因子不需要改动平台代码。

```python
import numpy as np
import pandas as pd

FACTORS: dict = {}

def factor(name: str):
    def deco(fn):
        FACTORS[name] = fn
        return fn
    return deco

@factor("mom_20")
def momentum_20(px: pd.DataFrame) -> pd.DataFrame:
    """20 日动量（跳过最近 1 日以减弱短期反转）；px：行=日期，列=股票"""
    return px.shift(1) / px.shift(21) - 1

@factor("vol_20")
def volatility_20(px: pd.DataFrame) -> pd.DataFrame:
    return -px.pct_change().rolling(20).std()        # 取负号：低波动得分高

def preprocess(cross_section: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """截面预处理：MAD 去极值 + Z-score 标准化（Phase 7）"""
    med = cross_section.median()
    mad = (cross_section - med).abs().median()
    clipped = cross_section.clip(med - n_mad * 1.4826 * mad, med + n_mad * 1.4826 * mad)
    return (clipped - clipped.mean()) / clipped.std()

rng = np.random.default_rng(0)
dates = pd.bdate_range("2026-01-01", periods=60)
px = pd.DataFrame(np.exp(rng.normal(0, 0.02, (60, 5)).cumsum(axis=0)) * 10,
                  index=dates, columns=[f"S{i}" for i in range(5)])
enabled = ["mom_20", "vol_20"]                       # 实际从 factors.yaml 读取
last = {name: preprocess(FACTORS[name](px).iloc[-1]) for name in enabled}
print(pd.DataFrame(last).round(3))
```

### 3.3 数据与因子层的陷阱

| 陷阱 | 说明 | 预防 |
| --- | --- | --- |
| 前视偏差 | 财报用报告期对齐、用当日收盘价计算当日可交易的信号 | PIT 数据 + 信号统一 `shift(1)` 后再交易 |
| 幸存者偏差 | 股票池只含当前上市股票 | 使用历史成分与退市股票 |
| 复权口径混用 | 因子用前复权、回测用不复权 | 文档明确每列数据的口径 |
| 因子缓存过期 | 数据更新后因子缓存未失效 | 缓存键包含数据版本（内容哈希） |
| 行业中性化遗漏 | 因子暴露大量行业风格 | 预处理流水线中显式包含中性化步骤 |

---

## 4. 模型层与组合层

> 对应课程：`03-model-and-portfolio-layer.py`

### 4.1 模型层

| 方案 | 做法 | 优点 | 风险 |
| --- | --- | --- | --- |
| 等权合成 | 各因子 Z-score 直接平均 | 稳健、无需训练 | 忽略因子强弱 |
| IC 加权 | 按历史滚动 IC 加权（Phase 7） | 简单自适应 | IC 估计噪声大 |
| 线性回归 | 截面回归预测下期收益 | 可解释 | 共线性 |
| LightGBM | 非线性截面模型（Phase 9） | 捕捉交互效应 | 过拟合、需 Purged CV |

**工程要点**：模型必须**滚动训练**（每月/每季度用截至当时的数据重新训练），并保存每个版本的模型文件、训练区间与特征列表。回测中每个调仓日只能使用"当时已训练好"的模型。

### 4.2 组合层：从分数到权重再到指令

```
 预测分数 ──▶ 选股（Top N / 分位） ──▶ 权重（等权 / 优化器） ──▶ 约束 ──▶ 目标权重
                                                          │
                       行业偏离、单票上限、换手上限、流动性 ◀──┘
                                                          │
 目标权重 × 资金 ÷ 价格 ──▶ 目标股数（取整 100 股）──▶ 与当前持仓求差 ──▶ 调仓指令
```

```python
import numpy as np
import pandas as pd

def topn_weights(scores: pd.Series, n: int, max_w: float) -> pd.Series:
    """选分数最高的 n 只等权，并检查单票上限"""
    chosen = scores.nlargest(n).index
    w = pd.Series(1.0 / n, index=chosen)
    assert w.max() <= max_w + 1e-12, "n 太小导致单票权重超限"
    return w

def limit_turnover(target: pd.Series, current: pd.Series, max_turnover: float) -> pd.Series:
    """若单边换手超过上限，则只朝目标移动一部分：w = current + λ (target - current)"""
    idx = target.index.union(current.index)
    t, c = target.reindex(idx, fill_value=0.0), current.reindex(idx, fill_value=0.0)
    turnover = (t - c).abs().sum() / 2
    lam = min(1.0, max_turnover / turnover) if turnover > 0 else 1.0
    return (c + lam * (t - c))[lambda s: s > 1e-9]

def to_orders(target_w: pd.Series, current_shares: pd.Series, prices: pd.Series,
              capital: float, lot: int = 100) -> list[dict]:
    idx = target_w.index.union(current_shares.index)
    tgt_shares = (target_w.reindex(idx, fill_value=0) * capital / prices.reindex(idx)
                  // lot * lot).astype(int)
    delta = tgt_shares - current_shares.reindex(idx, fill_value=0)
    return [{"symbol": s, "side": "BUY" if d > 0 else "SELL", "volume": int(abs(d))}
            for s, d in delta.items() if d != 0]

scores = pd.Series(np.random.default_rng(1).normal(size=30), index=[f"S{i:02d}" for i in range(30)])
current = pd.Series(0.1, index=[f"S{i:02d}" for i in range(10)])
target = limit_turnover(topn_weights(scores, n=20, max_w=0.05), current, max_turnover=0.3)
prices = pd.Series(10.0, index=scores.index)
orders = to_orders(target, (current * 1_000_000 / 10).astype(int), prices, capital=1_000_000)
print(f"目标持仓 {len(target)} 只，权重和 {target.sum():.3f}，指令 {len(orders)} 条")
```

> 注意：取整到 100 股后实际权重会偏离目标，小资金时偏离更明显；卖出时还要检查 T+1 可用持仓、停牌和跌停（Phase 4、13）。

---

## 5. 执行层与监控面板

> 对应课程：`04-execution-and-monitoring.py`

### 5.1 执行层

执行层把调仓指令交给 Phase 13 的交易系统，复用其中的风控、OMS 与网关：

| 环节 | 做法 |
| --- | --- |
| 指令校验 | 停牌、涨跌停、可用持仓、资金是否足够 |
| 先卖后买 | A 股资金约束下，先卖出释放资金再买入 |
| 执行算法 | 小资金可用开盘/收盘集合竞价或 TWAP；资金较大时用 VWAP（Phase 11） |
| 未完成处理 | 当日未成交部分（如涨停买不进）次日重新评估，而不是盲目追单 |
| 执行记录 | 每笔成交的到达价、成交价、费用，用于偏差分析（Phase 13 的 13.3 节） |

### 5.2 监控面板

| 面板 | 指标 | 作用 |
| --- | --- | --- |
| 净值 | 策略净值 vs 基准、超额收益、回撤 | 总体表现 |
| 实盘 vs 回测 | 同期回测净值与实盘净值的差异及分解 | 发现实现偏差 |
| 因子健康 | 各因子滚动 IC、因子收益 | 发现因子衰减（Phase 7） |
| 模型健康 | 预测分数分布、特征分布漂移（PSI 等） | 发现模型漂移（Phase 9） |
| 组合暴露 | 行业、市值、风格暴露 | 风险是否符合预期（Phase 8） |
| 交易 | 换手率、成交率、滑点、费用 | 执行质量（Phase 11） |
| 系统 | 流水线各步骤耗时与状态、数据更新时间 | 运维 |

工具选择：个人项目可用 Streamlit 或 Dash 快速搭建交互式面板，指标型监控可沿用 Phase 13 的 Prometheus + Grafana 方案。

### 5.3 代码：日报核心指标

```python
import numpy as np
import pandas as pd

def daily_report(nav: pd.Series, bench: pd.Series) -> dict:
    ret, bret = nav.pct_change().dropna(), bench.pct_change().dropna()
    excess = ret - bret
    dd = nav / nav.cummax() - 1
    return {
        "累计收益": nav.iloc[-1] / nav.iloc[0] - 1,
        "累计超额": (1 + excess).prod() - 1,
        "年化波动": ret.std() * np.sqrt(252),
        "夏普(年化, rf=0)": ret.mean() / ret.std() * np.sqrt(252),
        "当前回撤": dd.iloc[-1],
        "最大回撤": dd.min(),
        "近 20 日跟踪误差": excess.tail(20).std() * np.sqrt(252),
    }

rng = np.random.default_rng(3)
idx = pd.bdate_range("2026-01-01", periods=200)
bench = pd.Series((1 + rng.normal(0.0003, 0.012, 200)).cumprod(), index=idx)
nav = pd.Series((1 + rng.normal(0.0006, 0.013, 200)).cumprod(), index=idx)
print({k: round(float(v), 4) for k, v in daily_report(nav, bench).items()})
```

---

## 6. 量化研究报告写作

> 对应课程：`05-research-report-writing.py`

### 6.1 为什么要写报告

研究报告是量化研究员最核心的"产品"。它的读者（投资经理、风控、面试官）通常只有 5–10 分钟，需要快速回答三个问题：**钱从哪里来？证据可信吗？什么时候会失效？**

### 6.2 报告结构

| 章节 | 内容 | 常见缺陷 |
| --- | --- | --- |
| 1. 摘要 | 一段话：策略做什么、核心结果（样本外、扣费后）、主要风险 | 只报样本内最好结果 |
| 2. 假设与经济逻辑 | 为什么这个信号能赚钱：风险补偿、行为偏差还是结构性原因 | 只有数据挖掘，没有逻辑 |
| 3. 数据 | 数据来源、样本区间、股票池、PIT 处理、清洗规则 | 未说明幸存者偏差处理 |
| 4. 方法 | 信号构建、预处理、模型、组合构建、调仓规则 | 细节不足以复现 |
| 5. 结果 | 样本内/样本外绩效、与基准对比、分年度表现 | 没有基准、只看总收益 |
| 6. 稳健性 | 参数扰动、分段、分市场环境、不同股票池 | 只展示一组参数 |
| 7. 成本与容量 | 成本假设、换手、冲击成本估计、容量上限（Phase 11 平方根法则） | 不扣成本或成本过低 |
| 8. 风险与失效条件 | 最大回撤期分析、拥挤度、何种市场环境会失效 | 缺失 |
| 9. 结论与建议 | 是否建议上线、建议的资金规模与监控指标 | 结论超出证据 |
| 附录 | 参数尝试次数、全部实验记录、代码与数据版本 | 隐瞒尝试次数 |

### 6.3 必须披露的五项（课程通用验收标准）

```
┌──────────────────────────────────────────────────────────────────┐
│                    每个策略结论都必须包含                          │
├──────────────────────────────────────────────────────────────────┤
│  □ 样本区间（含样本内/样本外的划分）                               │
│  □ 交易成本假设（佣金、印花税、滑点、冲击）                         │
│  □ 基准对比（指数、等权组合或简单替代策略）                         │
│  □ 样本外结果                                                      │
│  □ 参数尝试次数（用于评估过拟合，见 Phase 9 Deflated Sharpe）        │
└──────────────────────────────────────────────────────────────────┘
```

### 6.4 5 分钟讲清一个策略

学习检查要求"能用 5 分钟讲清楚一个策略的收益来源与失效条件"。一个可用的框架：

| 时间 | 内容 | 示例（以 A 股低波动因子为例） |
| --- | --- | --- |
| 30 秒 | 一句话策略 | 每月买入过去一段时间波动率最低的一批股票 |
| 90 秒 | 收益来源 | 低波动异象：投资者偏好高波动"彩票型"股票、杠杆约束使部分投资者追逐高 Beta |
| 90 秒 | 证据 | 样本外扣费后的超额收益与夏普、分年度稳定性、参数扰动结果 |
| 60 秒 | 失效条件 | 题材炒作、风格急剧切换的行情中可能显著跑输；策略拥挤时超额收益衰减 |
| 30 秒 | 风险控制与建议 | 行业中性约束、拥挤度监控、建议的资金规模 |

### 6.5 可复现性

```python
import hashlib
import json
import platform
import subprocess
import sys

def run_manifest(config: dict, data_files: list[str]) -> dict:
    """记录一次研究运行的全部版本信息，附在报告附录中"""
    def sha(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    return {
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
        "data_hashes": {p: sha(p) for p in data_files},
    }

print(json.dumps(run_manifest({"seed": 42, "factors": ["mom_20", "vol_20"]}, []),
                 ensure_ascii=False, indent=2))
```

---

## 7. 策略尽调与上线评审

> 对应课程：`06-strategy-due-diligence.py`

### 7.1 尽调：像审计师一样质疑自己的策略

**比喻**：策略尽调像代码评审 + 安全审计。作者天然会对自己的策略"自信"，尽调的目的是在真金白银投入前，系统性地找出所有可能的漏洞。

| 维度 | 关键问题 |
| --- | --- |
| 收益来源 | 赚的是谁的钱？风险溢价、行为偏差还是流动性提供？能否用一句话说清？ |
| 经济逻辑 | 如果这个规律真实存在，为什么还没被套利掉？ |
| 数据 | 是否存在前视、幸存者偏差？财务数据是否 PIT？ |
| 过拟合 | 尝试了多少参数和变体？Deflated Sharpe 是多少？参数平原是否宽？ |
| 样本外 | 样本外与样本内表现是否一致？是否在样本外多次"偷看"后调参？ |
| 成本 | 成本假设是否保守？双倍成本下还能盈利吗？ |
| 容量 | 资金放大到多少时冲击成本吃掉一半超额？ |
| 风险 | 最大回撤发生在什么时候、为什么？与已有策略的相关性？尾部风险？ |
| 拥挤度 | 是否是市场上的常见因子？拥挤时表现如何？ |
| 运营 | 数据依赖是否稳定？执行是否可实现（涨跌停、停牌、T+1）？ |

### 7.2 红旗信号

| 红旗 | 可能原因 |
| --- | --- |
| 夏普比率过高（如中低频策略回测夏普 > 3） | 前视偏差、成本遗漏、过拟合 |
| 收益集中在少数几天或几只股票 | 偶然性，或数据错误 |
| 参数稍微变化，收益大幅下降 | 过拟合，处在参数"尖峰"上 |
| 样本外显著差于样本内 | 过拟合或市场结构变化 |
| 换手极高但回测成本很低 | 成本假设不现实 |
| 小市值、低流动性股票贡献大部分收益 | 实际无法成交，容量极小 |
| 回测曲线异常平滑 | 前视偏差或未来数据泄漏 |
| 无法说清收益来源 | 数据挖掘结果，样本外大概率失效 |

### 7.3 上线评审清单

```
┌──────────────────────────────────────────────────────────────────┐
│                         上线评审清单                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  研究                                                             │
│  □ 研究报告完整，五项披露齐全                                      │
│  □ 独立复现：另一个人（或另一套代码）能复现核心结果                   │
│  □ 双倍成本、参数扰动、分段测试通过                                 │
│                                                                  │
│  模拟盘                                                           │
│  □ 模拟盘 ≥ 4 周，与同期回测偏差可解释（Phase 13）                   │
│                                                                  │
│  风险                                                             │
│  □ 设定最大回撤止损线与策略停用条件                                 │
│  □ 与现有策略的相关性与组合层面影响已评估                            │
│                                                                  │
│  运营                                                             │
│  □ 监控指标与告警已配置；失效判定标准已书面化                        │
│  □ 合规：程序化交易报告等要求已完成                                  │
│                                                                  │
│  结论：□ 通过   □ 有条件通过（限定资金/观察期）   □ 不通过            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.4 策略失效的判定

上线前就要写好"什么情况下停用策略"，避免事后找理由：

- **回撤**：超过回测最大回撤的 1.5 倍（或预设阈值）
- **统计**：滚动窗口内实盘收益显著低于回测预期（如实盘夏普的置信区间不再包含回测值）
- **逻辑**：收益来源的前提条件消失（如市场规则变化、因子极度拥挤）

---

## 8. 量化岗位与面试

> 对应课程：`07-quant-career-interview.py`

### 8.1 岗位分类

| 岗位 | 主要工作 | 核心能力 | 适合背景 |
| --- | --- | --- | --- |
| 量化研究员 Quant Researcher | 因子挖掘、模型研究、策略开发 | 统计、机器学习、金融直觉 | 数理统计、CS、物理 |
| 量化交易员 Quant Trader | 管理策略运行、盘中决策、做市/套利 | 概率直觉、快速决策、风险意识 | 数理背景，反应快 |
| 量化开发 Quant Developer | 交易系统、回测平台、数据基础设施 | 系统设计、C++/Python、性能优化 | **软件工程（你的优势）** |
| 风控 / 模型验证 | 风险度量、模型审核、限额管理 | 统计、风险模型、严谨 | 金融工程、统计 |
| 数据工程 / 另类数据 | 数据采集、清洗、另类数据产品化 | 数据工程、爬虫、NLP | 软件工程 |
| 投资经理 PM | 组合管理、策略配置、对投资结果负责 | 综合能力与业绩记录 | 资深研究员晋升 |

**国内常见机构类型**：量化私募、公募基金量化/指数增强团队、券商自营与金融工程研究、期货公司及 CTA 机构、外资与做市机构等。不同机构对岗位的划分和要求差异较大。

**对资深软件工程师的建议**：从**量化开发**切入是最自然的路径；同时用本课程的研究报告证明自己具备研究能力，可以争取"研究型开发"或研究员岗位。

### 8.2 面试结构

```
┌──────────────────────────────────────────────────────────────────┐
│                         典型面试流程                              │
├──────────────────────────────────────────────────────────────────┤
│  简历筛选 ──▶ 笔试/在线测评 ──▶ 技术面（1–3 轮）──▶ 主管面 ──▶ Offer  │
│                                                                  │
│  技术面常见题型：                                                  │
│    ① 概率脑筋急转弯（考直觉与严谨）                                 │
│    ② 统计与计量（考对数据的理解）                                   │
│    ③ 编程（考实现能力与复杂度意识）                                 │
│    ④ 策略/研究（考研究方法论与金融理解）                             │
│    ⑤ 项目深挖（考你简历上的每一个数字）                              │
└──────────────────────────────────────────────────────────────────┘
```

### 8.3 概率题

**题 1：连续两次正面**。反复掷一枚均匀硬币，直到连续出现两次正面（HH），期望需要掷多少次？

> **思路**：状态法（马尔可夫链）。设 $E_0$ 为从头开始的期望次数，$E_1$ 为"上一次是 H"时的期望剩余次数：
>
> $$E_0 = 1 + \tfrac{1}{2}E_1 + \tfrac{1}{2}E_0, \qquad E_1 = 1 + \tfrac{1}{2}\cdot 0 + \tfrac{1}{2}E_0$$
>
> 解得 $E_0 = 6$。追问：等到 HT 的期望是 4（因为出现 H 后，再出 H 不会"前功尽弃"）。

**题 2：可以重掷一次的骰子**。掷一个骰子，你可以选择接受点数（获得等额奖金），或重掷一次并接受第二次的结果。最优策略下期望收益是多少？

> **思路**：第二次掷的期望是 3.5，所以第一次掷出 4、5、6 就接受，掷出 1、2、3 就重掷。
>
> $$E = \frac{4+5+6}{6} + \frac{3}{6}\times 3.5 = 2.5 + 1.75 = 4.25$$
>
> 追问：可以重掷两次呢？（先算出"还剩一次重掷"的价值 4.25，第一次掷出 5、6 接受，否则重掷：$E = \frac{5+6}{6} + \frac{4}{6}\times 4.25 \approx 4.67$）

**题 3：折断木棍**。在单位长度木棍上独立均匀地随机选两个点折断，三段能构成三角形的概率？

> **思路**：三段能构成三角形 ⇔ 每段都小于 1/2。在单位正方形 $(x, y)$ 中画出满足条件的区域，面积为 $1/4$。

**题 4：集齐骰子六面**。掷骰子直到六个点数都至少出现一次，期望次数？

> **思路**：已集齐 $k$ 种后，出现新点数的概率为 $\frac{6-k}{6}$，等待次数服从几何分布，期望 $\frac{6}{6-k}$。
>
> $$E = \sum_{k=0}^{5}\frac{6}{6-k} = 6\left(1 + \tfrac{1}{2} + \cdots + \tfrac{1}{6}\right) = 14.7$$

**题 5：生日问题**。房间里至少要有多少人，才能使"至少两人同一天生日"的概率超过 50%？

> **思路**：计算"所有人生日都不同"的概率 $\prod_{i=0}^{n-1}\frac{365-i}{365}$，$n = 23$ 时首次低于 50%。

可以用蒙特卡洛快速验证答案——这也是面试后自查的好习惯：

```python
import numpy as np

rng = np.random.default_rng(0)

def flips_until(pattern: str, trials: int = 100_000) -> float:
    total = 0
    for _ in range(trials):
        seq = ""
        while not seq.endswith(pattern):
            seq += "H" if rng.random() < 0.5 else "T"
        total += len(seq)
    return total / trials

def coupon_dice(trials: int = 100_000) -> float:
    counts = []
    for _ in range(trials):
        seen, n = set(), 0
        while len(seen) < 6:
            seen.add(int(rng.integers(1, 7)))
            n += 1
        counts.append(n)
    return float(np.mean(counts))

x, y = rng.random(1_000_000), rng.random(1_000_000)
a, b = np.minimum(x, y), np.maximum(x, y)
triangle = np.mean((a < 0.5) & (b - a < 0.5) & (1 - b < 0.5))

print(f"HH: {flips_until('HH'):.2f}  HT: {flips_until('HT'):.2f}")
print(f"集齐六面: {coupon_dice():.2f}  折棍成三角形: {triangle:.4f}")
```

### 8.4 统计题

**题 6**：$X \sim N(0, 1)$，$X$ 与 $X^2$ 相关吗？独立吗？

> $\text{Cov}(X, X^2) = E[X^3] - E[X]E[X^2] = 0$，所以**不相关**；但 $X^2$ 完全由 $X$ 决定，所以**不独立**。考点：相关系数只度量线性关系。

**题 7**：$y$ 对 $x$ 回归的斜率是 $\beta_1$，$x$ 对 $y$ 回归的斜率是 $\beta_2$，二者有什么关系？

> $\beta_1 = \rho\frac{\sigma_y}{\sigma_x}$，$\beta_2 = \rho\frac{\sigma_x}{\sigma_y}$，所以 $\beta_1\beta_2 = \rho^2 \le 1$。考点：回归不对称，$\beta_2 \ne 1/\beta_1$（除非完全相关）。

**题 8**：一个策略的日均收益除以日收益标准差为 0.1，年化夏普约是多少？

> 假设日收益独立同分布，$0.1 \times \sqrt{252} \approx 1.59$。追问：若日收益存在正自相关，简单 $\sqrt{252}$ 年化会**高估**夏普（Lo, 2002）。

**题 9**：你测试了 100 个策略，最好的那个 t 统计量为 2.5，可信吗？

> 不一定。即使 100 个策略都无效，在 5% 显著性水平下也期望有约 5 个"显著"。需要多重检验校正（如 Bonferroni：显著性水平调整为 $0.05/100$，对应 t 约 3.5 以上），或使用 Deflated Sharpe（Phase 9）。

**题 10**：用 3 年日度数据估计出年化夏普为 1.0，这个估计的标准误大约是多少？

> 在收益独立同分布的近似下，夏普估计的标准误约为 $\sqrt{(1 + \frac{1}{2}SR^2)/T}$，$T$ 以年为单位：$\sqrt{1.5/3} \approx 0.71$。也就是说，95% 置信区间大约是 $1.0 \pm 1.4$——**3 年数据几乎无法区分夏普 1.0 和 0**。

### 8.5 编程题

**题 11：最大回撤**。给定净值序列，$O(n)$ 计算最大回撤。

```python
def max_drawdown(nav: list[float]) -> float:
    peak, mdd = nav[0], 0.0
    for x in nav:
        peak = max(peak, x)
        mdd = max(mdd, 1 - x / peak)
    return mdd

print(max_drawdown([1.0, 1.2, 0.9, 1.1, 0.8, 1.3]))   # 0.3333：从 1.2 跌到 0.8
```

**题 12：滑动窗口最大值**。$O(n)$ 计算长度为 $k$ 的滑动窗口最大值（如计算 Donchian 通道上轨，Phase 5）。

```python
from collections import deque

def sliding_max(xs: list[float], k: int) -> list[float]:
    dq, out = deque(), []                      # dq 保存下标，对应值单调递减
    for i, x in enumerate(xs):
        while dq and xs[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(xs[dq[0]])
    return out

print(sliding_max([1, 3, 2, 5, 4, 1, 6], 3))    # [3, 5, 5, 5, 6]
```

**其他常见编程题**：实现限价订单簿（Phase 11 第 2.5 节）、水塘抽样（流式数据中均匀抽样）、实现 LRU 缓存、用 pandas 计算分组截面排名、多线程/异步下的生产者消费者（Phase 13 事件引擎）。

### 8.6 策略与研究题

**题 13**：你发现一个回测夏普为 3 的日频选股因子，接下来做什么？

> **回答框架**：先怀疑再验证。① 检查前视偏差（信号时间、财报 PIT、复权方式）；② 检查成本与可交易性（涨跌停、停牌、小市值）；③ 样本外与分段测试；④ 与已知因子（市值、动量、反转、低波动等）做回归，看是否只是已知因子的重新包装；⑤ 披露尝试次数，计算 Deflated Sharpe；⑥ 估算容量。参考第 7.2 节红旗信号。

**题 14**：为什么动量效应可能存在？什么时候会失效？

> **收益来源**：投资者对信息反应不足、羊群效应、处置效应等行为偏差。**失效条件**：市场急剧反转时（"动量崩溃"，如熊市底部快速反弹）；拥挤导致的踩踏。**A 股特点**：中短期常表现出反转效应，长期动量的表现也与美股不同——回答时应结合数据而不是背结论。

**题 15**：如何判断一个策略是否已经失效，而不只是处于正常回撤？

> **回答框架**：① 事先设定的统计标准（回撤超出历史分布的多少分位、滚动收益置信区间）；② 检查收益来源的前提是否仍成立（因子 IC、拥挤度、市场规则变化）；③ 区分"市场环境不利"与"结构性失效"——前者在类似历史环境中也出现过；④ 用小资金或降低仓位的方式持续观察，而不是全有或全无。

### 8.7 20 题练习清单

学习检查要求完成 20 道面试题。上文 15 道之外，建议补充以下 5 道并写出完整解答：

| # | 类型 | 题目 |
| --- | --- | --- |
| 16 | 概率 | 三门问题：换门后获胜概率为什么是 2/3？用条件概率和模拟两种方式说明 |
| 17 | 概率 | 两个独立的 $U(0,1)$ 随机变量，其最大值的期望是多少？（答案 2/3） |
| 18 | 统计 | 什么是 p-hacking？在因子研究中有哪些表现形式？ |
| 19 | 编程 | 用 pandas 对面板数据按日期分组计算截面 Rank IC |
| 20 | 策略 | 用平方根法则估算一个日频策略在给定换手率下的容量 |

---

## 9. 最佳实践与常见陷阱

### 9.1 最佳实践

```
┌──────────────────────────────────────────────────────────────────┐
│                          最佳实践清单                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  平台                                                             │
│  □ 先定义层间数据契约，再写实现                                     │
│  □ 回测与实盘消费同一份组合层输出                                   │
│  □ 每次运行生成 manifest（代码、数据、配置版本）                     │
│  □ 流水线每步校验，失败即停并告警                                   │
│                                                                  │
│  研究                                                             │
│  □ 研究日志记录每一次尝试（含失败的）                               │
│  □ 先写"收益来源与失效条件"，再看回测结果                            │
│  □ 报告五项披露齐全                                                │
│                                                                  │
│  职业                                                             │
│  □ 把 mini-quant 平台和研究报告整理成作品集                         │
│  □ 面试中每个数字都能解释来源与局限                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 常见陷阱

| 陷阱 | 表现 | 解决方案 |
| --- | --- | --- |
| 平台过度设计 | 花大量时间做框架，没有一个可用策略 | 先跑通最简单的端到端流水线，再逐层增强 |
| 层间隐式耦合 | 模型层直接读数据库，绕过因子层 | 通过 Protocol 和数据契约强制分层 |
| 结果不可复现 | 报告中的数字再也跑不出来 | manifest + 固定随机种子 + 数据快照 |
| 报告只讲优点 | 评审者无法判断风险 | 单独的"风险与失效条件"章节 |
| 面试背答案 | 追问一层就答不上来 | 理解推导过程，用代码验证 |
| 简历数字经不起追问 | 回测夏普很高但说不清成本与样本外 | 只写自己能完整解释的结果 |

---

## 10. 总结

### 10.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                          核心概念回顾                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  mini-quant 平台                                                  │
│  ├── 数据 → 因子 → 模型 → 组合 → 回测 / 执行 → 监控                 │
│  ├── Protocol 接口 + 数据契约 + 配置驱动                            │
│  └── 一键流水线：每步校验，失败即停                                 │
│                                                                  │
│  研究与评审                                                       │
│  ├── 报告结构：逻辑 → 数据 → 方法 → 结果 → 稳健性 → 成本容量 → 风险  │
│  ├── 五项披露：区间、成本、基准、样本外、尝试次数                    │
│  └── 尽调：收益来源、过拟合、成本容量、失效条件                      │
│                                                                  │
│  职业                                                             │
│  ├── 岗位：研究员、交易员、开发、风控、数据                          │
│  └── 面试：概率、统计、编程、策略、项目深挖                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 学习检查对照

| 学习检查项 | 对应章节 |
| --- | --- |
| 平台能一键从数据更新跑到生成调仓指令 | 第 2.5、4.2 节 |
| 完成一份可复现的策略研究报告 | 第 6.2、6.5 节 |
| 能用 5 分钟讲清楚一个策略的收益来源与失效条件 | 第 6.4 节 |
| 完成 20 道量化面试题 | 第 8.3–8.7 节 |

### 10.3 全课程回顾与下一步

```
  第一部分 基础能力         第二部分 策略研究          第三部分 AI 量化     第四部分 系统与实盘
  ────────────────         ────────────────         ──────────────      ────────────────
  P1 市场与数据             P5 CTA                   P9  机器学习        P11 微观结构与执行
  P2 金融时间序列           P6 统计套利               P10 DL/RL/LLM      P12 期权与衍生品
  P3 数学与统计             P7 多因子                                    P13 交易系统与实盘
  P4 回测系统               P8 组合与风险                                P14 综合项目 ◀ 你在这里
```

完成本课程后，可以沿以下方向继续深入：

- **研究深度**：持续维护 mini-quant，每月新增并检验因子，积累研究日志
- **实盘经验**：按 Phase 13 的流程，以小资金长期运行一个经过尽调的策略
- **专业方向**：高频与做市（Phase 11 延伸）、期权波动率交易（Phase 12 延伸）、AI 量化（Phase 10 延伸，可结合 llm-course）
- **职业输出**：把平台、研究报告和面试题解整理为作品集

**推荐阅读**：

- Ernest Chan《Quantitative Trading》：个人量化交易者的完整实践指南
- Xinfeng Zhou《A Practical Guide to Quantitative Finance Interviews》：量化面试题集
- Marcos López de Prado《Advances in Financial Machine Learning》：研究方法论与回测陷阱
