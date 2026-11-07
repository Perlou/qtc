# Phase 10: 深度学习、RL 与 LLM 量化 - 核心概念深度解析

> **学习目标**：理解深度学习、强化学习和大语言模型在量化中**能做什么、不能做什么**；掌握无前视的数据与环境设计；把 llm-course 中学到的 RAG、Agent 能力迁移到量化研究
> **前置知识**：Phase 9（金融 ML 方法论：Purged CV、RankIC、DSR）；你已掌握 PyTorch、Transformer、RAG、Agent、LangGraph

---

## 目录

1. [引言：AI 量化的能力边界](#1-引言ai-量化的能力边界)
2. [LSTM 时序预测](#2-lstm-时序预测)
3. [Transformer 时序模型](#3-transformer-时序模型)
4. [Qlib 深度模型库](#4-qlib-深度模型库)
5. [强化学习交易环境设计](#5-强化学习交易环境设计)
6. [PPO 交易智能体](#6-ppo-交易智能体)
7. [新闻与公告情绪因子](#7-新闻与公告情绪因子)
8. [LLM 研报与财报分析](#8-llm-研报与财报分析)
9. [LLM 因子挖掘](#9-llm-因子挖掘)
10. [量化研究 Agent](#10-量化研究-agent)
11. [AI 量化的陷阱：前视、过拟合与幻觉](#11-ai-量化的陷阱前视过拟合与幻觉)
12. [最佳实践清单](#12-最佳实践清单)
13. [总结](#13-总结)

---

## 1. 引言：AI 量化的能力边界

### 1.1 三类技术、三种角色

```
┌──────────────────────────────────────────────────────────────────────┐
│                     AI 技术在量化流水线中的位置                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  数据 ──▶ 特征/因子 ──▶ 预测模型 ──▶ 组合/决策 ──▶ 执行 ──▶ 研究管理    │
│              ▲             ▲             ▲           ▲         ▲      │
│              │             │             │           │         │      │
│        LLM 文本因子    深度学习        强化学习     RL 执行    LLM Agent │
│        LLM 因子挖掘   (LSTM/TF)     (仓位决策)   (拆单)    (研究助手)  │
│                                                                      │
│   深度学习：替代/增强 Phase 9 的 GBDT，处理序列与原始数据               │
│   强化学习：把"预测 + 决策"合并为直接学习策略                           │
│   LLM：     打开非结构化数据（新闻、研报）+ 自动化研究流程               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 期望管理：一张诚实的表

| 技术 | 擅长 | 不擅长 / 主要风险 |
|---|---|---|
| **LSTM / Transformer** | 原始序列、高频数据、多模态、大样本 | 日频低信噪比数据上常不如 GBDT；易过拟合；调参成本高 |
| **强化学习** | 执行、做市、对冲等**有明确动态和成本结构**的问题 | 用于"选股/择时"时奖励噪声极大、样本只有一条历史路径，极易过拟合 |
| **LLM 文本分析** | 把新闻、公告、研报变成结构化信号；跨语言；零样本 | 训练数据截止带来的前视偏差、幻觉、成本、可复现性 |
| **LLM Agent** | 自动化研究流程：提假设、写代码、跑回测、写报告 | 隐性多重检验（大量试验）、生成未来函数、错误累积 |

> **比喻**：如果说 Phase 9 的 GBDT 是一把趁手的瑞士军刀，那么深度学习是一台精密机床（数据够多够干净时威力巨大，否则不如军刀），RL 是一个需要大量试错的学徒（只有在模拟器足够真实时才学得好），LLM 是一个博览群书但偶尔信口开河、而且"读过未来报纸"的研究助理。

---

## 2. LSTM 时序预测

### 2.1 快速回顾

你已经熟悉 LSTM：通过遗忘门、输入门、输出门控制细胞状态 $c_t$，缓解 RNN 的梯度消失。

$$
h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1}), \qquad \hat y_t = W h_t + b
$$

- $x_t$：第 $t$ 步输入特征；$h_t$：隐状态；$c_t$：细胞状态；$\hat y_t$：输出

在量化中，输入通常是过去 $L$ 天的特征序列（如 Alpha360 的 60 天 × 6 字段），输出是未来收益（或其截面排名）。

### 2.2 经典误区：预测价格而不是收益

网上大量"LSTM 预测股价准确率 99%"的教程都犯了同一个错误：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     "完美预测" 的价格曲线为什么是假的                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   价格  ▲      实际 ───   预测 - - -                                  │
│        │         ╱╲      ╱╲                                          │
│        │    ╱╲  ╱  ╲ - -╱ -╲╱╲    预测曲线 ≈ 实际曲线向右平移一天        │
│        │ ╱╲╱  ╲╱ - -╲╱    - ╲ ╲                                       │
│        │╱  - -╱                                                      │
│        └──────────────────────────▶ 时间                              │
│                                                                      │
│   价格序列近似随机游走：p_{t+1} ≈ p_t。                                 │
│   模型只要学会"输出昨天的价格"，R² 就能达到 0.99。                      │
│   但这对交易毫无价值——交易需要的是 p_{t+1} - p_t 的方向和大小。          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**正确做法**：预测收益（或超额收益、截面排名），并永远和**朴素基线**比较：

| 基线 | 定义 |
|---|---|
| 持续性（persistence） | $\hat p_{t+1} = p_t$，即预测收益为 0 |
| 历史均值 | $\hat r_{t+1} = \bar r$ |
| 线性模型 / GBDT | Phase 7 线性多因子、Phase 9 LightGBM |

### 2.3 代码：无泄漏的窗口构造与模型

```python
import numpy as np
import torch
import torch.nn as nn

def make_windows(feat: np.ndarray, fwd_ret: np.ndarray, L: int = 60):
    """feat: (T, F)，第 t 行只含 t 日收盘前可得信息
    fwd_ret: (T,)，fwd_ret[t] = t → t+1 的收益（标签，未来值）
    返回 X: (N, L, F)，X[k] = feat[t-L+1 : t+1]；y[k] = fwd_ret[t]"""
    X, y = [], []
    for t in range(L - 1, len(feat) - 1):
        X.append(feat[t - L + 1:t + 1])
        y.append(fwd_ret[t])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)

class LSTMReg(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (B, L, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)   # 取最后一步的隐状态

# 关键：
# 1) 标准化参数（均值/方差）只用训练集计算，或对每个窗口做自身归一化
# 2) 训练/验证/测试按时间切分，相邻段之间留出 ≥ 标签期限的间隔（Phase 9 Purging）
# 3) 损失可用 MSE，但评估用 IC / 方向准确率 / 扣费策略收益，并与基线比较
```

### 2.4 截面化：从"单股票时序"到"多股票截面"

单只股票的 LSTM 择时样本量太小（10 年约 2400 个日样本，还高度自相关）。更有效的做法是**所有股票共享一个模型**：每个样本是（股票，日期）的历史窗口，标签做截面标准化——这正是 Qlib 深度模型的设定（第 4 节）。样本量从几千跳到数百万，而且学的是"截面排序"规律，比"绝对涨跌"更稳定。

---

## 3. Transformer 时序模型

### 3.1 自注意力用在时间维度

把 $L$ 个时间步看作 $L$ 个 token，自注意力让每一步直接关注任意历史时刻：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

- $Q, K, V$：由输入序列线性投影得到的查询、键、值；$d_k$：键向量维度
- $M$：掩码矩阵。对**编码器读取历史窗口**做单点预测时，窗口内全是过去数据，不需要因果掩码；若做逐步预测（每个位置都输出未来值），必须用因果掩码 $M_{ij} = -\infty\ (j > i)$ 防止"偷看"后面的时间步

### 3.2 时序 Transformer 的研究进展（简述）

| 方向 | 代表工作 | 要点 |
|---|---|---|
| 长序列效率 | Informer（2021）、Autoformer（2021） | 稀疏注意力、序列分解 |
| **对 Transformer 的质疑** | Zeng et al.,《Are Transformers Effective for Time Series Forecasting?》（AAAI 2023），提出 DLinear | 在常用长期预测基准上，一个简单的线性模型就能匹敌或超过许多 Transformer 变体 |
| Patch 化 | PatchTST（ICLR 2023） | 把连续若干时间步切成一个 patch 作为 token；各变量通道独立建模 |
| 时序基础模型 | TimesFM（Google）、Chronos（Amazon）、Moirai（Salesforce）等，2024 年起 | 大规模预训练、零样本预测 |

这些基准大多是电力、交通、天气等**高信噪比、强周期性**的数据。金融收益几乎没有稳定周期，信噪比极低，**通用时序模型的优势很难直接迁移**。此外，时序基础模型的预训练语料可能包含公开金融序列，用它们在历史区间做回测时，存在与 LLM 类似的前视风险（第 11 节）。

### 3.3 金融场景更有意义的注意力用法

- **截面注意力**：在同一天的股票之间做注意力，学习股票间的关联（行业、供应链、联动），例如 Qlib 中的 GATs、HIST 等图/关系模型
- **时间 + 截面双重注意力**：先在每只股票的时间维度编码，再在截面维度交互
- **多模态融合**：价量序列 + 新闻文本 embedding

### 3.4 代码：一个最小的时序 Transformer

```python
import torch
import torch.nn as nn

class TSTransformer(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, L: int = 60, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))      # 可学习位置编码
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                      # x: (B, L, F)，全部是历史数据
        h = self.encoder(self.inp(x) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)  # 用最后一个时间步的表示预测
```

### 3.5 深度模型在金融数据上的局限（学习检查项）

```
┌──────────────────────────────────────────────────────────────────────┐
│                 为什么深度模型在日频金融数据上常常不如 GBDT              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 信噪比低：可预测部分只占收益方差的极小比例，大容量模型优先拟合噪声    │
│  ② 有效样本少：样本高度相关（时间重叠 + 截面相关），名义样本数虚高       │
│  ③ 非平稳：训练期学到的模式在测试期可能已失效，深模型"记得更牢"反而更糟  │
│  ④ 表格数据特性：手工因子已是高度提炼的特征，树模型在表格上是强基线      │
│  ⑤ 训练不稳定：随机种子不同，IC 可能差异显著 → 需多种子集成与报告方差     │
│  ⑥ 调参即试验：大量超参数搜索 = 大量隐性试验（Phase 9 的 DSR）           │
│                                                                      │
│  深度模型更可能有价值的场景：                                          │
│  · 高频/Tick/订单簿原始数据（Phase 11），样本量大、结构丰富              │
│  · 文本、图像等非结构化数据                                            │
│  · 与 GBDT 集成，提供低相关的预测                                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Qlib 深度模型库

### 4.1 概览

Qlib 在 `qlib.contrib.model` 下实现了大量 PyTorch 模型，并在 `examples/benchmarks/` 中提供了统一配置。部分模型：

| 类别 | 模型 | 说明 |
|---|---|---|
| 循环网络 | LSTM、GRU、ALSTM（带注意力的 LSTM） | 经典序列模型 |
| 卷积 | TCN | 时间卷积网络 |
| 注意力 | Transformer、Localformer、TFT | 时间维度注意力 |
| 图 / 关系 | GATs、HIST、IGMTF | 利用股票间关系 |
| 表格 | MLP、TabNet | 不建模序列 |
| 漂移/鲁棒 | TRA、ADARNN、TCTS、DoubleEnsemble | 应对分布漂移、样本选择 |

### 4.2 Alpha158 vs Alpha360 与数据集类型

| 组合 | 适用 | 说明 |
|---|---|---|
| Alpha158 + `DatasetH` | GBDT、MLP | 每个样本是一行手工特征 |
| Alpha360 + `DatasetH` | 把 360 维当作 60×6 序列 reshape 的模型 | 原始价量，让模型自己学特征 |
| Alpha158 + `TSDatasetH` | 名称带 `_ts` 的时序模型 | `TSDatasetH` 按 `step_len` 为每个样本拼出历史窗口 |

```python
# 时序数据集配置片段（结构与 Qlib benchmark 配置一致）
dataset_cfg = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {...},          # Alpha158，通常配合 RobustZScoreNorm、Fillna 等处理器
        "segments": {...},
        "step_len": 20,            # 每个样本包含过去 20 个交易日
    },
}
```

### 4.3 如何阅读官方 benchmark

Qlib README 的 benchmark 表格给出了各模型在 CSI300 上的 IC、ICIR、年化超额、信息比率、最大回撤等。阅读时注意：

- 在官方结果中，**深度模型并未全面胜过 LightGBM / DoubleEnsemble 等树模型**；不同模型之间的差距也常常小于多次随机种子之间的波动（以最新官方表格为准）
- 结果基于特定数据源、特定时段和调好的超参数，换数据或时段后排序可能改变
- 正确姿势：把 benchmark 当作**可复现的起点**，在自己的数据和 Purged 验证框架下重新评估，并报告多种子的均值与标准差

---

## 5. 强化学习交易环境设计

### 5.1 把交易表述为 MDP

| MDP 要素 | 交易中的对应 | 设计要点 |
|---|---|---|
| **状态 $s_t$** | 截至 $t$ 的市场特征 + 当前持仓 + 账户信息 | 只能包含 $t$ 时刻**已经可得**的信息 |
| **动作 $a_t$** | 目标仓位（离散 {空仓, 半仓, 满仓} 或连续权重） | A 股个股只能做多；考虑 100 股整数手 |
| **转移 $P(s_{t+1} \mid s_t, a_t)$** | 市场按历史数据推进；持仓按动作更新 | 若假设"不影响市场"，环境就是历史回放 |
| **奖励 $r_t$** | 扣费后的收益、风险调整收益 | 奖励设计决定智能体学到什么 |
| **折扣 $\gamma$** | 对未来奖励的重视程度 | 交易中常取接近 1 |

> **比喻**：搭建交易环境就像给飞行员搭模拟器。如果模拟器里"窗外的景色比飞机提前一秒出现"（前视），飞行员在模拟器里会是王牌，上了真飞机就坠毁。**环境的真实性比算法的先进性重要得多。**

### 5.2 时序：避免前视的关键

```
  t 日                               t+1 日                          t+2 日
  ├── 开盘 ── 盘中 ── 收盘 ──┤        ├── 开盘 ── 盘中 ── 收盘 ──┤      ├── 开盘 ...
                        │             │                                 │
                  观察 s_t（用到 t 日收盘数据）                           │
                  智能体决策 a_t                                         │
                              以 t+1 开盘价执行 ─┐                        │
                                                 └─ 持有至 t+2 开盘 ──────┘
                                                    奖励 r_t = 仓位 × (open_{t+2}/open_{t+1} - 1) - 成本

  错误示范：观察用 t 日收盘数据，却按 t 日收盘价成交（或更糟：按 t 日开盘价）
```

常见前视来源：

| 来源 | 例子 | 对策 |
|---|---|---|
| 成交价 | 用当天收盘价算信号，又以当天收盘价成交 | 次日开盘 / VWAP 成交 |
| 特征归一化 | 用全样本均值方差标准化状态 | 只用训练期统计量；或滚动标准化 |
| 观察窗口 | 状态切片写成 `X[t-w+1 : t+2]`（多取一天） | 单元测试：打乱 $t+1$ 之后的数据，状态不应变化 |
| 奖励泄漏进状态 | 把"上一步奖励"之外的未来收益放进 info 再被策略读到 | 严格区分 obs 与 info |
| 复权 | 前复权价格包含未来除权信息（Phase 1） | 用后复权或当时可得的复权因子 |
| A 股规则 | 忽略涨停买不进、跌停卖不出、T+1 | 环境中模拟成交约束 |

### 5.3 奖励设计

| 奖励 | 公式 | 特点 |
|---|---|---|
| 扣费收益 | $r_t = w_t R_{t+1} - c\lvert w_t - w_{t-1}\rvert$ | 最直接；智能体可能学会承担极端风险 |
| 风险惩罚 | $r_t = w_t R_{t+1} - c\lvert\Delta w\rvert - \kappa (w_t R_{t+1})^2$ | 近似均值方差效用 |
| 回撤惩罚 | 当前回撤超过阈值时额外扣分 | 控制路径风险 |
| 差分夏普 | 见下式 | 在线优化夏普比率 |

- $w_t$：持仓权重；$R_{t+1}$：持有期资产收益；$c$：单位交易成本；$\kappa$：风险厌恶系数

**差分夏普比率（Differential Sharpe Ratio，Moody & Saffell 2001）**：用指数移动估计的一阶、二阶矩 $A_t, B_t$，夏普对最新收益 $R_t$ 的增量为：

$$
D_t = \frac{B_{t-1}\Delta A_t - \frac{1}{2}A_{t-1}\Delta B_t}{\left(B_{t-1} - A_{t-1}^2\right)^{3/2}}, \quad
\Delta A_t = R_t - A_{t-1},\ \ \Delta B_t = R_t^2 - B_{t-1}
$$

- $A_t = A_{t-1} + \eta\Delta A_t$，$B_t = B_{t-1} + \eta\Delta B_t$，$\eta$ 为适应速率

### 5.4 代码：最小可用的 Gymnasium 交易环境

```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SingleAssetEnv(gym.Env):
    """单资产多头环境：t 日收盘后观察，t+1 开盘执行，持有到 t+2 开盘"""
    metadata = {"render_modes": []}
    POSITIONS = (0.0, 0.5, 1.0)

    def __init__(self, features: np.ndarray, open_px: np.ndarray,
                 window: int = 20, cost: float = 0.001, episode_len: int = 250):
        super().__init__()
        assert len(features) == len(open_px)
        self.X = features.astype(np.float32)        # 已用训练期统计量标准化
        self.open = open_px.astype(np.float64)
        self.window, self.cost, self.episode_len = window, cost, episode_len
        n_feat = self.X.shape[1]
        self.action_space = spaces.Discrete(len(self.POSITIONS))
        self.observation_space = spaces.Box(-np.inf, np.inf, (window * n_feat + 1,), np.float32)

    def _obs(self):
        win = self.X[self.t - self.window + 1:self.t + 1]   # 只到 t（含）
        return np.append(win.ravel(), self.pos).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        last_start = len(self.X) - self.episode_len - 2
        self.t = int(self.np_random.integers(self.window - 1, last_start))  # 随机起点
        self.t_end = self.t + self.episode_len
        self.pos = 0.0
        return self._obs(), {}

    def step(self, action):
        target = self.POSITIONS[int(action)]
        trade_cost = self.cost * abs(target - self.pos)
        hold_ret = self.open[self.t + 2] / self.open[self.t + 1] - 1   # 揭示于 t+2，不进入 obs
        reward = target * hold_ret - trade_cost
        self.pos = target
        self.t += 1
        terminated = self.t >= self.t_end
        return self._obs(), float(reward), terminated, False, {"hold_ret": hold_ret}
```

要点：

- 遵循 Gymnasium API：`reset` 返回 `(obs, info)`，`step` 返回 `(obs, reward, terminated, truncated, info)`
- **随机起点**让智能体看到不同的市场片段，降低对单一路径的记忆
- 训练环境与测试环境用**不重叠的时间段**分别构造
- 可以写一个"前视测试"：修改 `features[t+1:]`，确认 `_obs()` 在 $t$ 时刻不变

---

## 6. PPO 交易智能体

### 6.1 PPO 的核心思想

PPO（Proximal Policy Optimization，Schulman et al. 2017）是策略梯度方法。直觉：**每次更新策略时不要迈太大步**，否则一次糟糕的更新可能让策略崩溃。

> **比喻**：调整交易系统时，你不会一夜之间把策略完全推翻，而是"在原策略附近小步改进"。PPO 用数学方式强制这种保守。

### 6.2 公式

概率比：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

截断的代理目标：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat A_t,\ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\right)\right]
$$

广义优势估计（GAE）：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
$$

总目标（最大化）：

$$
L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2\,\mathcal{H}[\pi_\theta](s_t)
$$

- $\pi_\theta$：当前策略；$\pi_{\theta_{\text{old}}}$：采样数据时的策略
- $\hat A_t$：优势估计，"这个动作比平均好多少"
- $\epsilon$：截断范围（常取 0.2）；$\gamma$：折扣；$\lambda$：GAE 参数（常取 0.95）
- $V$：价值函数（critic）；$L^{\text{VF}}$：价值函数的平方误差；$\mathcal{H}$：策略熵，鼓励探索；$c_1, c_2$：权重

### 6.3 代码：Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

train_env = SingleAssetEnv(X_train, open_train)
check_env(train_env)                                    # 检查 API 是否符合规范

model = PPO("MlpPolicy", train_env, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            learning_rate=3e-4, seed=0, verbose=0)
model.learn(total_timesteps=300_000)

def evaluate(model, env):
    """在测试期从头到尾跑一遍（测试环境应设置固定起点、覆盖整个测试期）"""
    obs, _ = env.reset(seed=0)
    rewards, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(int(action))
        rewards.append(r)
        done = terminated or truncated
    return np.array(rewards)
```

如果使用 `VecNormalize` 做观察/奖励归一化，评估时要加载训练时的统计量并设置 `training=False`、`norm_reward=False`，否则测试期数据会"更新"归一化参数，造成前视。

### 6.4 为什么 RL 用于择时/选股特别难

| 问题 | 说明 |
|---|---|
| **奖励噪声** | 单步奖励几乎全是噪声（第 1.2 节的信噪比问题），信用分配极难 |
| **只有一条历史** | 无法像游戏那样无限重开；多 episode 只是同一段历史的不同切片 |
| **动作不影响环境** | 小资金不影响价格时，问题退化为"上下文老虎机"（contextual bandit），RL 的序贯决策优势用不上，且比监督学习更低效 |
| **超参数敏感** | 不同随机种子结果差异大；必须报告多个种子 |
| **评估** | 必须与买入持有、Phase 9 监督学习 + TopK 规则对比，扣费后比较 |

**RL 更适合的金融问题**：动作会改变后续状态和成本的问题——

- **最优执行**：大单拆分，动作影响冲击成本（Phase 11 的 Almgren-Chriss 是解析基准）。Qlib 提供了面向订单执行的 RL 框架（`qlib.rl`）
- **做市**：库存影响后续报价风险（Phase 11 的 Avellaneda-Stoikov）
- **期权对冲**：在成本存在时学习对冲频率与比例（Phase 12）

参考项目：FinRL（AI4Finance）提供了多种交易环境与 DRL 基线，适合学习，但其示例回测结果不应直接视为可实盘的证据。

---

## 7. 新闻与公告情绪因子

### 7.1 从文本到因子的流水线

```
┌──────────────────────────────────────────────────────────────────────┐
│                         文本因子流水线                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 采集      新闻、公告、股吧/社交媒体、研报                             │
│     │        记录【首次发布时间】而不是抓取时间或入库时间                  │
│  ② 去重      同一事件被多家媒体转载 → 聚类去重，保留最早时间               │
│     │                                                                │
│  ③ 实体链接  文本 → 股票代码（简称、全称、曾用名、子公司）                 │
│     │                                                                │
│  ④ 打分      词典法 / FinBERT 类模型 / LLM 结构化输出                    │
│     │        情绪、事件类型、置信度、相关度                               │
│  ⑤ 时间对齐  发布时间 → 可交易的信号日（盘后公告 → 次日）                 │
│     │                                                                │
│  ⑥ 聚合      每只股票每天聚合，按时间衰减                                │
│     │                                                                │
│  ⑦ 预处理    截面标准化、市值/行业中性化（Phase 7）                       │
│     │        注意：新闻覆盖度本身与市值强相关                             │
│  ⑧ 检验      RankIC、IC 衰减、分组回测、与已有因子的相关性                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 情绪打分方法对比

| 方法 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| 词典法 | 正/负面词计数（如 Loughran-McDonald 金融词典，英文） | 透明、快、无前视 | 不理解否定、语境；中文需自建词典 |
| 微调小模型 | FinBERT（英文金融情绪 BERT）或中文金融 BERT 类模型 | 便宜、可本地部署、可复现 | 需要标注数据；模型本身的预训练语料也有截止时间 |
| LLM 零样本 / 少样本 | 用 prompt 让 LLM 输出结构化判断 | 理解复杂语境、事件类型、不需标注 | 成本高；**前视偏差**；输出可能不稳定 |

Lopez-Lira & Tang（2023，*Can ChatGPT Forecast Stock Price Movements?*）用 ChatGPT 对新闻标题打分，发现其与次日收益有显著相关性；值得注意的是，他们特意使用了**模型训练截止日期之后**的新闻来评估，以规避前视偏差——这正是第 11 节要讲的关键点。

### 7.3 代码：LLM 结构化打分 + 时间对齐 + 衰减聚合

```python
from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

class NewsSignal(BaseModel):
    stock_code: str
    sentiment: float = Field(ge=-1, le=1, description="对该公司股价的影响方向与强度")
    event_type: Literal["业绩", "并购重组", "增持回购", "减持", "监管处罚", "诉讼",
                        "订单合同", "人事变动", "其他"]
    relevance: float = Field(ge=0, le=1, description="新闻与该公司的相关程度")

PROMPT = """你是 A 股研究员。阅读以下新闻，判断其对【{code}】股价的短期影响。
只依据新闻文本本身，不要使用你对该公司后续股价走势的任何记忆。
以 JSON 输出，字段：stock_code, sentiment(-1~1), event_type, relevance(0~1)。
新闻：{text}"""

def score_news(llm_json, code: str, text: str) -> NewsSignal:
    """llm_json: 你在 llm-course 中封装的"调用 LLM 并返回 JSON dict"的函数"""
    raw = llm_json(PROMPT.format(code=code, text=text))
    return NewsSignal.model_validate(raw)                 # 校验失败会抛异常 → 重试或丢弃

def to_signal_date(ts: pd.Timestamp, trade_days: pd.DatetimeIndex) -> pd.Timestamp:
    """返回第一个"收盘时刻（15:00）晚于发布时间"的交易日。
    信号在该日收盘后形成，次日开盘交易；盘后/周末公告自然归到下一交易日。"""
    closes = trade_days + pd.Timedelta(hours=15)
    i = closes.searchsorted(ts, side="right")
    return trade_days[i] if i < len(trade_days) else pd.NaT

def aggregate_sentiment(news: pd.DataFrame, trade_days: pd.DatetimeIndex,
                        halflife: float = 3.0) -> pd.DataFrame:
    """news 列：signal_date, stock_code, sentiment, relevance
    返回 (date × stock) 的衰减加权情绪因子"""
    news = news.assign(s=news.sentiment * news.relevance)
    daily = news.pivot_table(index="signal_date", columns="stock_code",
                             values="s", aggfunc="sum")
    daily = daily.reindex(trade_days).fillna(0.0)
    return daily.ewm(halflife=halflife).mean()            # 只用过去，天然无前视
```

然后用 Phase 7/9 的方法检验：因子截面标准化、市值行业中性化，计算与未来 1/5/20 日收益的 RankIC 与 IC 衰减。

### 7.4 文本因子的特有陷阱

| 陷阱 | 说明 |
|---|---|
| **时间戳** | 抓取时间 ≠ 发布时间；新闻被修改后时间戳更新；A 股公告大量在盘后/晚间发布 |
| **覆盖偏差** | 大盘股新闻多、小盘股少，"情绪因子"可能只是市值或关注度的代理 → 必须中性化 |
| **无新闻 ≠ 中性** | 缺失值填 0 会混淆"没有消息"与"中性消息"；可另设"新闻热度"因子 |
| **已反映在价格中** | 消息发布当天价格已大幅反应，因子在次日开盘后的 IC 才是可交易的 |
| **提示注入** | 新闻文本里可能出现"忽略之前的指令……"之类内容（llm-course 讲过的 prompt injection），批量处理外部文本时要隔离指令与数据 |
| **成本** | 全市场每天数万条文本，LLM 调用成本可观 → 先用便宜模型/规则过滤，再用强模型精判（llm-course 的成本优化） |

---

## 8. LLM 研报与财报分析

### 8.1 应用场景

| 场景 | 输入 | 输出 |
|---|---|---|
| 研报结构化 | 券商研报 PDF | 评级、目标价、盈利预测、核心逻辑、风险点 |
| 财报解读 | 年报/季报 | 关键指标、管理层表述变化、会计异常提示 |
| 业绩电话会/调研纪要 | 文字记录 | 语气变化、指引上调/下调 |
| 跨文档问答 | 多份研报 + 公告 | "这家公司过去一年的主要分歧是什么？" |

### 8.2 与 llm-course RAG 知识的衔接

你在 llm-course Phase 4/5 学的 RAG 技术几乎可以全部复用，但金融场景要加上**时间维度**：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Point-in-Time RAG（时点一致的 RAG）                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  普通 RAG：   query ──▶ 检索全部文档 ──▶ LLM 生成                       │
│                                                                      │
│  PIT RAG：    (query, as_of) ──▶ 检索 publish_ts ≤ as_of 的文档         │
│                                 ──▶ LLM 生成（并要求引用来源）           │
│                                                                      │
│  回测 2020-06-30 的信号时，绝不能检索到 2020-07-01 之后发布的研报          │
│  —— 这就是 Phase 7 "Point-in-Time 财报对齐" 在 RAG 中的翻版              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

| llm-course 技术 | 在研报分析中的用法 |
|---|---|
| 文档加载与分块（Phase 4） | PDF 研报中表格（盈利预测表）需单独解析，不能被切碎 |
| Self-Query 检索（Phase 5） | 由查询自动生成元数据过滤：股票代码、券商、发布时间 ≤ as_of |
| 混合检索 + 重排序（Phase 5） | 股票代码、财务科目名等精确词用 BM25，语义用向量 |
| 结构化输出（Phase 2） | Pydantic schema 约束抽取结果 |
| 忠实度评估（Phase 10） | 检查抽取的数字是否能在原文找到 |

### 8.3 代码：PIT 检索 + 结构化抽取 + 数字溯源

```python
from pydantic import BaseModel, Field

class ReportSummary(BaseModel):
    stock_code: str
    broker: str
    publish_date: str
    rating: str = Field(description="买入/增持/中性/减持/卖出，原文未提及则为 '未提及'")
    target_price: float | None
    eps_forecast: dict[str, float] = Field(description="年份 -> EPS 预测")
    key_points: list[str]
    risks: list[str]
    evidence: list[str] = Field(description="支撑上述数字的原文片段")

def pit_retrieve(collection, query: str, code: str, as_of_ts: int, k: int = 5):
    """以 Chroma 为例：元数据中 publish_ts 存为整数时间戳"""
    return collection.query(
        query_texts=[query], n_results=k,
        where={"$and": [{"stock_code": code}, {"publish_ts": {"$lte": as_of_ts}}]},
    )

def verify_numbers(summary: ReportSummary, source_text: str) -> list[str]:
    """反幻觉：抽取出的数字必须能在原文中找到（简单字符串匹配，生产中需处理格式差异）"""
    issues = []
    if summary.target_price is not None and f"{summary.target_price:g}" not in source_text:
        issues.append(f"目标价 {summary.target_price} 未在原文出现")
    for year, eps in summary.eps_forecast.items():
        if f"{eps:g}" not in source_text:
            issues.append(f"{year} EPS {eps} 未在原文出现")
    return issues
```

### 8.4 从研报到因子

LLM 抽取的结构化数据可以构造经典的**分析师因子**（这些在学术和业界都有长期研究）：

| 因子 | 构造 |
|---|---|
| 评级上调/下调 | 最近 N 天评级变化的净数量 |
| 盈利预测修正 | $\frac{\text{EPS}_{t}^{\text{forecast}} - \text{EPS}_{t-k}^{\text{forecast}}}{\lvert\text{EPS}_{t-k}^{\text{forecast}}\rvert}$ |
| 业绩超预期（SUE 类） | $\frac{\text{实际 EPS} - \text{一致预期 EPS}}{\sigma}$，$\sigma$ 为预期分歧或历史意外的标准差 |
| 文本语气变化 | 本期与上期管理层讨论部分的语气差异 |

LLM 的增量价值主要在于**降低从非结构化文本中抽取这些信息的成本**，而不是"让 LLM 直接判断该买哪只股票"。

---

## 9. LLM 因子挖掘

### 9.1 从遗传规划到 LLM

传统的公式化因子挖掘用**遗传规划（GP）**在算子空间里搜索表达式（如 WorldQuant《101 Formulaic Alphas》风格的表达式）。后来出现了用强化学习生成因子集合的方法（如 AlphaGen，KDD 2023），以及用 LLM 生成因子的方法（如 Alpha-GPT 等）。微软的 RD-Agent 项目也包含面向量化的"因子与模型联合迭代"场景。

LLM 的优势：能从**经济直觉**出发提出假设（"高换手伴随价格不涨，可能是分歧加大"），并把它翻译成表达式；GP 只能盲目组合算子。

### 9.2 闭环框架

```
┌──────────────────────────────────────────────────────────────────────┐
│                          LLM 因子挖掘闭环                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐   假设 + 表达式    ┌─────────────┐                  │
│   │  LLM 生成器  │ ───────────────▶ │ 语法/安全校验 │── 不合法 ──┐      │
│   └──────▲──────┘                   └──────┬──────┘            │      │
│          │                                 │ 合法               │      │
│          │ 反馈：IC、相关性、               ▼                    │      │
│          │ 失败原因                  ┌─────────────┐            │      │
│          │                          │ 计算因子值   │            │      │
│          │                          │ (仅训练期)   │            │      │
│          │                          └──────┬──────┘            │      │
│          │                                 ▼                   │      │
│          │                          ┌─────────────┐            │      │
│          └───────────────────────── │ 评估与筛选   │ ◀──────────┘      │
│                                     │ IC/IR/相关性 │                   │
│                                     └──────┬──────┘                   │
│                                            ▼                          │
│                            因子库 + 试验日志（记录每一次尝试）           │
│                                            ▼                          │
│                      研究结束后：样本外测试一次 + DSR（Phase 9）         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.3 安全的表达式求值器

**永远不要 `eval()` LLM 生成的代码**。用 AST 白名单解析一个受限 DSL，并禁止未来函数：

```python
import ast
import operator as op
import numpy as np
import pandas as pd

def _win(n) -> int:
    n = int(n)
    if n < 1:                                   # 负窗口 = 向未来取值 = 未来函数
        raise ValueError(f"窗口必须 ≥ 1，得到 {n}")
    return n

FUNCS = {   # 输入均为宽表 DataFrame（index=日期, columns=股票）
    "delay":   lambda x, n: x.shift(_win(n)),
    "delta":   lambda x, n: x - x.shift(_win(n)),
    "ts_mean": lambda x, n: x.rolling(_win(n)).mean(),
    "ts_std":  lambda x, n: x.rolling(_win(n)).std(),
    "ts_corr": lambda x, y, n: x.rolling(_win(n)).corr(y),
    "rank":    lambda x: x.rank(axis=1, pct=True),          # 截面排名
    "log":     lambda x: np.log(x.where(x > 0)),                # 非正值置 NaN
}
BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

def evaluate(expr: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            return BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
            return FUNCS[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in data:
            return data[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不允许的语法: {ast.dump(node)}")
    return ev(ast.parse(expr, mode="eval"))

# 例：LLM 提出"放量但不涨 → 分歧加大 → 未来收益偏弱"
# factor = evaluate("-rank(ts_corr(close, volume, 10))", {"close": close, "volume": volume})
```

> 白名单之外的任何语法（属性访问、下标、import、lambda 等）都会被拒绝。若使用 Qlib 表达式作为 DSL，同样要拒绝 `Ref(x, -n)` 这类负偏移。

### 9.4 生成与筛选循环

```python
import json

GEN_PROMPT = """你是量化研究员。基于以下可用字段与算子，提出 {k} 个新的日频选股因子。
字段: close, open, high, low, volume, amount, turnover
算子: delay, delta, ts_mean, ts_std, ts_corr, rank, + - * /（窗口必须为正整数）
已有因子及其训练期 RankIC（避免重复）: {library}
上一轮失败的尝试及原因: {failures}
输出 JSON 列表，每项含 "hypothesis"（经济逻辑）与 "expr"（表达式）。"""

def mining_round(llm_json, data_train, fwd_ret_train, library, failures, trial_log, k=10):
    proposals = llm_json(GEN_PROMPT.format(k=k, library=json.dumps(library, ensure_ascii=False),
                                           failures=json.dumps(failures[-10:], ensure_ascii=False)))
    for p in proposals:
        record = {"expr": p["expr"], "hypothesis": p["hypothesis"]}
        try:
            f = evaluate(p["expr"], data_train)
            ic = f.rank(axis=1).corrwith(fwd_ret_train.rank(axis=1), axis=1)   # 日度 RankIC
            record.update(ic_mean=float(ic.mean()), icir=float(ic.mean() / ic.std()))
        except Exception as e:
            record["error"] = str(e)
            failures.append(record)
        trial_log.append(record)            # 无论成败都记录 → DSR 的 N
    return trial_log
```

筛选标准（全部在**训练期**计算）：

- $\lvert\text{RankIC}\rvert$ 与 ICIR 超过阈值
- 与因子库中已有因子的截面相关性 < 0.5–0.7（避免换汤不换药）
- 经济逻辑可解释；换手不过高
- 研究结束后才在**封存的样本外期**测试一次，并用全部试验的夏普分布计算 DSR

---

## 10. 量化研究 Agent

### 10.1 从 llm-course 的 Agent 到研究 Agent

llm-course Phase 6 讲了工具调用、ReAct、MCP；Phase 7 讲了 LangGraph、Supervisor、人在回路。研究 Agent 就是把这些能力组装成一条"假设 → 实现 → 回测 → 分析 → 报告"的流水线。

| llm-course 概念 | 研究 Agent 中的对应 |
|---|---|
| 工具（Tool） | 数据查询、因子计算、回测、绘图、写文件 |
| ReAct 循环 | 提出假设 → 跑实验 → 观察结果 → 修正假设 |
| 记忆（Memory） | 已尝试因子与结果的日志（同时是 DSR 的试验计数） |
| MCP Server | 把行情库、回测引擎包装成标准化的工具服务 |
| LangGraph 状态图 | 显式的研究阶段与条件分支 |
| 人在回路（HITL） | 关键决策（进入样本外测试、上线模拟盘）需人工批准 |
| 评估（Phase 10） | 用 LLM-as-Judge 检查报告是否忠实于实验结果 |

### 10.2 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         量化研究 Agent 架构                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│     │ propose  │──▶│implement │──▶│ backtest │──▶│ analyze  │     │
│     │ 提出假设  │    │ 生成表达式│    │ 训练期回测 │    │ 分析结果  │     │
│     └────▲─────┘    └──────────┘    └──────────┘    └────┬─────┘     │
│          │                                               │           │
│          └────────────── 继续迭代（未达标且预算未用完）─────┤           │
│                                                          │ 达标/预算耗尽│
│                                                          ▼           │
│                                                    ┌──────────┐      │
│                                                    │ human    │      │
│                                                    │ review   │      │
│                                                    └────┬─────┘      │
│                                                         ▼            │
│                                                    ┌──────────┐      │
│                                                    │ report   │      │
│                                                    └──────────┘      │
│                                                                      │
│   工具层（强制的护栏）：                                               │
│   · 数据工具只返回训练期数据；样本外数据不对 Agent 开放                   │
│   · 表达式经 AST 白名单校验；代码在沙箱中执行，有超时与资源限制           │
│   · 每次回测自动写入试验日志                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**设计原则**：护栏放在**工具层**，而不是寄希望于 prompt 里"请不要使用未来数据"。这和 llm-course 中"最小权限原则"一致——Agent 根本拿不到样本外数据，就不可能（有意或无意地）用它调参。

### 10.3 代码：LangGraph 骨架

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    goal: str
    hypothesis: str
    expr: str
    metrics: dict
    trial_log: list
    iteration: int
    report: str

def propose(state: ResearchState) -> dict:
    # 调用 LLM：根据 goal 与 trial_log 中的历史结果提出新假设
    ...
    return {"hypothesis": "...", "iteration": state["iteration"] + 1}

def implement(state: ResearchState) -> dict:
    # 调用 LLM 把假设翻译为 DSL 表达式；用 evaluate() 的白名单校验
    ...
    return {"expr": "..."}

def backtest(state: ResearchState) -> dict:
    # 工具：只在训练期计算 RankIC、分组收益、换手；自动追加到 trial_log
    metrics = {...}
    return {"metrics": metrics, "trial_log": state["trial_log"] + [{**metrics, "expr": state["expr"]}]}

def analyze(state: ResearchState) -> dict:
    # 调用 LLM 解读结果，总结失败原因供下一轮参考
    ...
    return {}

def should_continue(state: ResearchState) -> str:
    good = state["metrics"].get("icir", 0) > 0.5
    return "done" if good or state["iteration"] >= 20 else "continue"

def write_report(state: ResearchState) -> dict:
    # 报告必须披露：试验总次数、样本区间、成本假设、失败的尝试
    ...
    return {"report": "..."}

g = StateGraph(ResearchState)
for name, fn in [("propose", propose), ("implement", implement), ("backtest", backtest),
                 ("analyze", analyze), ("report", write_report)]:
    g.add_node(name, fn)
g.set_entry_point("propose")
g.add_edge("propose", "implement")
g.add_edge("implement", "backtest")
g.add_edge("backtest", "analyze")
g.add_conditional_edges("analyze", should_continue, {"continue": "propose", "done": "report"})
g.add_edge("report", END)
app = g.compile()
# result = app.invoke({"goal": "寻找低相关的量价因子", "trial_log": [], "iteration": 0, ...})
```

在 `analyze` 与 `report` 之间插入人工审核节点，可以用 LangGraph 的 `interrupt_before` 配合 checkpointer 实现（llm-course Phase 7 的人在回路）。

### 10.4 研究 Agent 的评估

| 维度 | 检查方法 |
|---|---|
| 正确性 | 生成的因子代码是否计算正确？与人工实现对拍 |
| 无前视 | 自动化单元测试：截断未来数据后因子值不变 |
| 忠实性 | 报告中的数字是否与实验日志一致（LLM-as-Judge + 程序化比对） |
| 研究质量 | 样本外存活率：Agent 发现的因子有多少在封存期仍有效？ |
| 效率 | 每个有效因子的 token 成本与计算成本 |

---

## 11. AI 量化的陷阱：前视、过拟合与幻觉

### 11.1 LLM 训练数据截止带来的前视偏差（核心）

这是 LLM 量化中**最隐蔽、也最致命**的陷阱，也是本阶段的学习检查项。

> **比喻**：你请一位"穿越者"帮你回测 2020 年的新闻情绪策略。他读到"某疫苗公司启动三期临床"的新闻时，打出了"强烈利好"——不是因为新闻本身，而是因为他**记得**这只股票后来涨了 5 倍。他的回测表现无与伦比，但他对 2026 年的新闻不会有这种"记忆"。

```
┌──────────────────────────────────────────────────────────────────────┐
│                       LLM 前视偏差的机制                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  时间轴 ───────────────────────────────────────────────────────────▶  │
│                                                                      │
│     回测区间 [2018, 2023]          模型训练截止 2024        实盘 2026   │
│     ├────────────────────┤             │                    │         │
│     │  模型"见过"这段时期的             │                    │         │
│     │  新闻、股价、事后分析、           │                    │         │
│     │  "XX 公司为何暴涨"的文章          │                    │         │
│                                        │                    │         │
│     → 回测中的情绪判断混入了事后知识      │      → 实盘中没有这种    │
│     → IC 虚高                           │        "记忆"，IC 回落    │
│                                                                      │
│  泄漏渠道：                                                           │
│   · 公司名/股票代码 → 唤起对该公司后续走势的记忆                        │
│   · 日期 → 唤起对宏观事件结果的记忆（疫情、政策、危机走向）              │
│   · 独特事件描述 → 唤起对事件结局的记忆                                 │
│   · 学术文献 → 模型"知道"哪些因子后来被发表为有效                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**缓解方法**：

| 方法 | 做法 | 局限 |
|---|---|---|
| **只在截止日之后评估** | 只用模型训练截止日期之后的数据做样本外测试 | 可用样本期短；需要准确知道截止日期（厂商公布的日期可能只是近似） |
| **实体匿名化** | 把公司名、代码、人名替换为"公司A"，日期替换为相对时间 | 文本中的细节（产品、地名、金额）仍可能泄露身份；匿名化也会损失有用信息 |
| **对照实验** | 比较匿名化前后的 IC；若原文 IC 显著更高，说明存在记忆效应 | 只能检测，不能完全消除 |
| **记忆探针** | 直接问模型"X 公司 2021 年股价表现如何"，看它是否知道 | 模型知道 ≠ 一定在打分中使用了 |
| **使用时点一致的模型** | 使用训练截止于回测期之前的模型（如旧版开源模型）| 旧模型能力弱；可用版本有限 |
| **只做信息抽取** | 让 LLM 抽取客观事实（"公告宣布回购 5 亿元"），而不是给出主观判断 | 减少但不能消除记忆影响 |

学术界已有专门研究，例如 Glasserman & Lin（2023）讨论了 GPT 情绪分析中的前视偏差，并使用匿名化公司名称作为对照。

```python
import re

def anonymize(text: str, entities: dict[str, str]) -> str:
    """entities: {"贵州茅台": "公司A", "600519": "代码A", ...}；按长度降序替换避免部分匹配"""
    for name in sorted(entities, key=len, reverse=True):
        text = text.replace(name, entities[name])
    text = re.sub(r"20\d{2}年", "某年", text)                # 粗粒度的日期遮蔽
    return text

def lookahead_check(ic_raw: pd.Series, ic_anon: pd.Series, cutoff: str) -> pd.DataFrame:
    """对比截止日前后、匿名化前后的平均 IC"""
    def split(s):
        return pd.Series({"截止前": s[:cutoff].mean(), "截止后": s[cutoff:].mean()})
    return pd.DataFrame({"原文": split(ic_raw), "匿名化": split(ic_anon)})

# 危险信号：原文 IC 在截止前显著高于截止后，且匿名化后差距明显缩小
```

> 同样的问题也存在于**时序基础模型**和**基于大语料预训练的文本 embedding 模型**：只要预训练数据覆盖了你的回测区间，就要考虑记忆效应。

### 11.2 过拟合：AI 让试验变得太便宜

| 来源 | 表现 | 对策 |
|---|---|---|
| **LLM/Agent 批量生成因子** | 一晚上试 1000 个因子，挑最好的 10 个 | 记录全部试验；DSR / PBO；封存样本外 |
| **深度模型超参数搜索** | 网格搜索 200 组配置 | 搜索本身计入试验次数；多种子报告方差 |
| **反复"看一眼"测试集** | 每次在测试集上看结果后调整 prompt | 测试集只用一次；用验证集迭代 |
| **重新发现已知因子** | LLM 生成的"新因子"是已发表异象的变体，而这些异象发表后常常衰减 | 与已知因子库做相关性检查；关注发表后的有效性 |
| **RL 记住历史路径** | 训练收益完美、测试崩溃 | 随机起点、不重叠的测试期、与简单基线对比 |

### 11.3 幻觉

| 类型 | 量化中的例子 | 对策 |
|---|---|---|
| 事实幻觉 | 编造研报中的目标价、EPS 预测、财务数字 | 数字溯源验证（第 8.3 节）；要求附带原文引用 |
| 代码幻觉 | 调用不存在的 API、参数；或代码能跑但逻辑错（如把 `shift(-1)` 当作滞后） | 沙箱执行 + 单元测试 + 前视检测测试 |
| 推理幻觉 | 对回测结果给出看似合理但错误的解释 | 报告中的结论必须由实验数字支撑；LLM-as-Judge 检查忠实性 |
| 时间幻觉 | 混淆事件先后，把后来的信息当作当时已知 | PIT 检索；在 prompt 中明确 as-of 日期并只提供该日期前的材料 |

### 11.4 工程与运营风险

- **不可复现**：闭源 API 模型版本更新、temperature > 0 都会导致同一输入输出不同 → 固定模型版本、temperature=0、缓存所有 LLM 输出（作为数据的一部分版本化）
- **成本与延迟**：全市场文本实时打分的成本与时延，决定了策略的可行频率
- **数据合规**：新闻、研报的版权与使用许可；不得将未公开信息输入外部 API
- **提示注入**：外部文本中的恶意指令（第 7.4 节）
- **模型供应商风险**：API 下线或政策变化 → 保留可本地部署的替代方案

---

## 12. 最佳实践清单

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AI 量化最佳实践清单                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  深度学习                                                             │
│  □ 预测收益/截面排名，不预测价格；始终与朴素基线和 LightGBM 对比          │
│  □ 标准化只用训练期统计量；按时间切分并 Purge                            │
│  □ 报告多个随机种子的均值与标准差                                       │
│                                                                      │
│  强化学习                                                             │
│  □ 观察只含 t 时刻可得信息；次日开盘执行；成本与 A 股规则进入环境         │
│  □ 写"前视单元测试"：篡改未来数据，观察不应变化                          │
│  □ 与买入持有、监督学习 + 规则策略对比；优先用于执行/做市/对冲            │
│                                                                      │
│  LLM 文本因子                                                         │
│  □ 记录首次发布时间；盘后信息归到下一交易日                              │
│  □ 截止日之后的样本外评估 + 匿名化对照实验                               │
│  □ 市值/行业/关注度中性化；缺失 ≠ 中性                                  │
│  □ 固定模型版本、temperature=0、缓存输出                                │
│                                                                      │
│  LLM 因子挖掘与 Agent                                                  │
│  □ 只用 AST 白名单 DSL，禁止 eval 与负偏移                              │
│  □ 护栏放在工具层：Agent 拿不到样本外数据                                │
│  □ 记录全部试验，用 DSR/PBO 修正                                        │
│  □ 关键步骤人工审核；报告披露试验次数与失败尝试                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 13. 总结

### 13.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────────┐
│                            核心概念回顾                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  深度学习                                                             │
│  ├── LSTM：预测收益而非价格；"完美价格曲线"只是滞后一天的复制             │
│  ├── Transformer：DLinear 的质疑；PatchTST；金融上截面注意力更有意义      │
│  ├── Qlib 深度模型：Alpha360 + TSDatasetH；benchmark 中未全面胜过 GBDT   │
│  └── 局限：低信噪比、有效样本少、非平稳、训练不稳定                      │
│                                                                      │
│  强化学习                                                             │
│  ├── MDP：状态（只含可得信息）、动作（目标仓位）、奖励（扣费收益/差分夏普）│
│  ├── 前视：t 收盘观察 → t+1 开盘执行；归一化只用训练期统计量             │
│  ├── PPO：截断概率比 + GAE + 价值损失 + 熵                              │
│  └── 更适合执行、做市、对冲，而非直接择时选股                             │
│                                                                      │
│  LLM                                                                  │
│  ├── 情绪因子：发布时间对齐 → 结构化打分 → 衰减聚合 → 中性化 → IC        │
│  ├── 研报分析：PIT RAG（publish_ts ≤ as_of）+ 结构化抽取 + 数字溯源      │
│  ├── 因子挖掘：LLM 提假设 → 白名单 DSL → 训练期 IC → 去相关 → DSR       │
│  ├── 研究 Agent：LangGraph 状态图 + 工具层护栏 + 人在回路                │
│  └── 陷阱：训练截止前视（核心）、隐性多重检验、幻觉、不可复现            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.2 与 llm-course 的知识映射

| llm-course | 本阶段 |
|---|---|
| Phase 2 结构化输出 | 情绪打分、研报抽取的 Pydantic schema |
| Phase 4-5 RAG / Self-Query / 重排序 | Point-in-Time 研报检索 |
| Phase 6 工具调用 / MCP | 数据、因子计算、回测作为工具 |
| Phase 7 LangGraph / 人在回路 | 研究 Agent 状态图与人工审核 |
| Phase 9 部署与成本优化 | 全市场文本打分的成本控制、缓存 |
| Phase 10 评估 / 忠实度 | 报告忠实性检查、因子样本外存活率 |

### 13.3 与下一阶段的衔接

Phase 11 进入**市场微观结构与执行**。本阶段的若干内容会在那里落地：

- **RL 的主场**：最优执行与做市是 RL 在金融中最有希望的方向，Phase 11 会先用 Almgren-Chriss 与 Avellaneda-Stoikov 给出解析基准——任何 RL 方案都应先打败这些基准
- **深度学习的主场**：Tick 与 Level-2 订单簿数据样本量大、结构丰富，比日频数据更适合深度模型；Phase 11 的高频因子（订单不平衡等）可以作为深度模型的输入
- **成本的真相**：本阶段所有策略的"扣费收益"都依赖成本模型；Phase 11 会解释价差与市场冲击从何而来，以及策略容量的上限

> **一句话总结**：AI 让量化研究**更快**，但不会自动让它**更对**。模型越强大、试验越便宜，Phase 4 与 Phase 9 建立的那套"不骗自己"的验证纪律就越重要。
