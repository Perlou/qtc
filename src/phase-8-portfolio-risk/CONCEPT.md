# Phase 8: 组合优化与风险管理 - 核心概念深度解析

> **学习目标**：理解"从 alpha 信号到最终持仓"的组合构建过程，掌握主流组合优化方法（MV / Shrinkage / RP / BL / HRP / 约束优化）、风险度量（VaR / CVaR / 压力测试）、绩效归因与换手控制
> **前置知识**：Phase 2（收益与波动）、Phase 3（协方差、PCA、蒙特卡洛）、Phase 7（因子、IC、Barra 风险模型）

---

## 目录

1. [引言：为什么需要组合优化](#1-引言为什么需要组合优化)
2. [Markowitz 均值方差与有效前沿](#2-markowitz-均值方差与有效前沿)
3. [估计误差：均值方差的阿喀琉斯之踵](#3-估计误差均值方差的阿喀琉斯之踵)
4. [协方差估计与 Ledoit-Wolf 收缩](#4-协方差估计与-ledoit-wolf-收缩)
5. [风险平价 Risk Parity](#5-风险平价-risk-parity)
6. [Black-Litterman 模型](#6-black-litterman-模型)
7. [层次风险平价 HRP](#7-层次风险平价-hrp)
8. [cvxpy 带约束组合优化](#8-cvxpy-带约束组合优化)
9. [VaR 与 CVaR](#9-var-与-cvar)
10. [压力测试与情景分析](#10-压力测试与情景分析)
11. [绩效归因：Brinson 与因子归因](#11-绩效归因brinson-与因子归因)
12. [再平衡与换手控制](#12-再平衡与换手控制)
13. [最佳实践与常见陷阱](#13-最佳实践与常见陷阱)
14. [总结](#14-总结)

---

## 1. 引言：为什么需要组合优化

### 1.1 从信号到持仓

Phase 7 你得到了一个因子打分（例如每只股票的合成 alpha 分数）。但"分数高的股票多买一点"距离一个可交易的组合还差很远：

- 买多少只？每只买多少？
- 两只高度相关的股票都得分很高，是不是等于把同一个赌注下了两次？
- 组合会不会不知不觉地押注了某个行业、或者全仓小盘股？
- 每次调仓要换掉多少仓位？交易成本会吃掉多少 alpha？
- 最坏情况下一天会亏多少？

```
┌──────────────────────────────────────────────────────────────────────┐
│                     量化投资流水线中的 Phase 8                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   数据 ──▶ 因子/模型 ──▶ alpha 预期 ──▶【组合构建】──▶ 执行 ──▶ 归因   │
│  (P1-2)   (P7, P9-10)      μ̂          (本阶段)       (P11)   (本阶段) │
│                                         ▲                            │
│                                         │                            │
│                              【风险模型】 Σ̂ (P7 Barra / 本阶段收缩)   │
│                                                                      │
│   组合构建 = 在 "预期收益 - 风险 - 成本" 之间、在约束下找最优权重       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

> **比喻**：alpha 模型是"选菜的厨师"，告诉你哪些食材好；组合优化是"配菜师"，决定每样放多少、荤素搭配、不超预算；风险管理是"食品安全员"，检查最坏情况下会不会吃坏肚子。

### 1.2 本阶段方法地图

| 方法 | 需要的输入 | 核心思想 | 典型用途 |
|---|---|---|---|
| **均值方差 MV** | $\mu$、$\Sigma$ | 给定风险下收益最大 | 理论基准；配合约束做指数增强 |
| **Ledoit-Wolf 收缩** | 历史收益 | 改进 $\Sigma$ 的估计 | 所有需要 $\Sigma$ 的方法 |
| **风险平价 RP** | 只要 $\Sigma$ | 每个资产贡献相同风险 | 大类资产配置（股债商品） |
| **Black-Litterman** | 市场权重、$\Sigma$、观点 | 市场均衡 + 主观观点的贝叶斯融合 | 有观点的战术配置 |
| **HRP** | 只要 $\Sigma$ | 层次聚类 + 递归二分 | 资产多、协方差病态时 |
| **约束优化 cvxpy** | $\alpha$、风险模型、约束 | 凸优化通用框架 | 实际生产中的股票组合 |

---

## 2. Markowitz 均值方差与有效前沿

### 2.1 直觉：不要把鸡蛋放在同一个篮子里，但要算清楚篮子之间的关系

Markowitz（1952，《Portfolio Selection》）的洞见是：**组合的风险不是单个资产风险的加权平均**，因为资产之间不完全相关。两只波动 30% 但相关性为 0 的股票各买一半，组合波动只有约 21%。

> **比喻**：两个醉汉各自随机走路（波动大），但如果你把他们用绳子绑在一起且他们走向互不相关，绳子中点的晃动幅度会小得多。相关性越低，"互相抵消"越多。

### 2.2 公式

设有 $n$ 个资产，权重向量 $w \in \mathbb{R}^n$，预期收益向量 $\mu$，协方差矩阵 $\Sigma$：

$$
\mu_p = w^\top \mu, \qquad \sigma_p^2 = w^\top \Sigma w = \sum_{i}\sum_{j} w_i w_j \sigma_{ij}
$$

- $\mu_p$：组合预期收益
- $\sigma_p^2$：组合方差；$\sigma_{ij}$ 是资产 $i$ 与 $j$ 收益的协方差，$\sigma_{ii}=\sigma_i^2$ 是方差
- 两资产情形：$\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho\sigma_1\sigma_2$，$\rho$ 是相关系数。$\rho<1$ 时交叉项变小，这就是分散化收益

均值方差优化的三种等价写法：

$$
\text{(a) 最小方差：}\ \min_w\ w^\top\Sigma w \quad \text{s.t. } w^\top\mu \ge \mu^*,\ \mathbf{1}^\top w = 1
$$

$$
\text{(b) 效用最大化：}\ \max_w\ w^\top\mu - \frac{\lambda}{2} w^\top \Sigma w \quad \text{s.t. } \mathbf{1}^\top w = 1
$$

$$
\text{(c) 最大夏普：}\ \max_w\ \frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma w}}
$$

- $\mu^*$：目标收益；$\lambda$：风险厌恶系数（越大越保守）；$r_f$：无风险利率；$\mathbf{1}$：全 1 向量

无约束（允许做空）时有闭式解：

$$
w_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}, \qquad
w_{\text{Tangency}} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu - r_f\mathbf{1})}
$$

- GMV（Global Minimum Variance）：全局最小方差组合，**完全不需要 $\mu$**
- Tangency：切点组合（最大夏普组合）

注意两个公式里都有 $\Sigma^{-1}$——这是后面所有麻烦的根源。

### 2.3 有效前沿与资本市场线

```
   预期收益 μ
      ▲
      │                         ● 切点组合（最大 Sharpe）
      │               ....--''  ╱
      │          ..-''        ╱  ← 资本市场线 CML：
      │       .-'   有效前沿 ╱       无风险资产 + 切点组合的连线
      │     .'      (上半支)╱
      │    :             ╱
      │    ● GMV       ╱
      │    :         ╱
      │     '.     ╱      × 单个资产（都在前沿内侧）
      │       '-.╱  ×
  r_f ●───────╱'--..   ×
      │           无效部分(下半支)
      └────────────────────────────────▶ 风险 σ
```

- **有效前沿（Efficient Frontier）**：每个风险水平下收益最高的组合的集合
- **GMV**：前沿最左端
- **切点组合**：从 $r_f$ 出发与前沿相切的点，夏普比率最大

### 2.4 代码：闭式解与带约束的有效前沿

```python
import numpy as np
import cvxpy as cp

def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """全局最小方差（允许做空）"""
    x = np.linalg.solve(Sigma, np.ones(len(Sigma)))  # 用 solve 而不是 inv，数值更稳
    return x / x.sum()

def tangency_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """最大夏普（允许做空；要求 x.sum() > 0 才有意义）"""
    x = np.linalg.solve(Sigma, mu - rf)
    return x / x.sum()

def efficient_frontier(mu, Sigma, n_points=30, long_only=True):
    """用 cvxpy 逐个目标收益求最小方差，得到前沿上的点"""
    n = len(mu)
    w = cp.Variable(n)
    target = cp.Parameter()
    cons = [cp.sum(w) == 1, mu @ w >= target]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)

    frontier = []
    for t in np.linspace(mu.min(), mu.max(), n_points):
        target.value = t
        prob.solve()
        if w.value is not None:
            frontier.append((np.sqrt(w.value @ Sigma @ w.value), t, w.value.copy()))
    return frontier
```

> 用 `cp.Parameter` 包装目标收益，cvxpy 可以复用问题结构，扫描前沿时更快。

### 2.5 量化中的实际意义

- 在股票多因子组合里，$\mu$ 一般**不是历史均值**，而是 alpha 模型的预测（Phase 7/9 的分数经过缩放）。
- 纯 MV 在实盘中很少裸用，通常是"MV 目标函数 + 大量约束 + 稳健的风险模型"（见第 8 节）。
- GMV 不依赖 $\mu$，在实证中往往比最大夏普组合更稳健——因为 $\mu$ 是最难估计的量。

---

## 3. 估计误差：均值方差的阿喀琉斯之踵

### 3.1 直觉："误差最大化器"

Michaud（1989）把均值方差优化称为 **"error maximizer"（误差最大化器）**：优化器会系统性地超配那些**预期收益被高估、风险被低估**的资产——而这些恰恰是估计误差最大的资产。

> **比喻**：优化器像一个极其聪明但轻信的投资者。你给他的数据里有个笔误把某只股票的预期收益多写了 2%，他会毫不犹豫地重仓这只股票。优化越"精确"，越会放大输入里的噪声。

### 3.2 为什么均值这么难估计

年化收益估计的标准误：

$$
\text{SE}(\hat\mu_{\text{ann}}) \approx \frac{\sigma_{\text{ann}}}{\sqrt{Y}}
$$

- $\sigma_{\text{ann}}$：年化波动率；$Y$：样本年数

一只年化波动 30% 的股票，用 10 年数据估计均值，标准误约 $30\%/\sqrt{10}\approx 9.5\%$。也就是说你估计"年化 12%"，真实值在 $[-7\%, 31\%]$ 内都很正常。**增加数据频率（日→分钟）不能缩小均值的标准误**，只有拉长时间跨度才行；而方差估计则能从高频数据中受益。

Chopra & Ziemba（1993）的研究表明，在典型风险厌恶水平下，**均值估计误差对组合效用的影响约是方差误差的 10 倍量级，方差误差又比协方差误差更重要**。

### 3.3 为什么 $\Sigma^{-1}$ 会放大误差

$\Sigma^{-1}$ 的特征值是 $\Sigma$ 特征值的倒数。样本协方差的**最小特征值会被系统性低估**（样本数 $T$ 与资产数 $n$ 接近时尤其严重），取逆后变成巨大的数，对应的特征向量方向（通常是"多一只、空一只相似股票"的组合）会被赋予极大权重。

条件数（condition number）衡量这种病态程度：

$$
\kappa(\Sigma) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

- $\lambda_{\max}, \lambda_{\min}$：最大与最小特征值。$\kappa$ 越大，求逆时对输入的微小扰动越敏感
- 当 $n > T$ 时样本协方差矩阵**奇异**（秩最多为 $T-1$），根本无法求逆。A 股 5000 只股票、用 1 年日线（约 242 个交易日）就是这种情况

### 3.4 代码：亲眼看看敏感性

```python
rng = np.random.default_rng(0)
n = 10
vol = rng.uniform(0.15, 0.35, n)
corr = 0.6 * np.ones((n, n)) + 0.4 * np.eye(n)         # 高相关资产（像同行业股票）
Sigma = np.outer(vol, vol) * corr
mu = rng.uniform(0.05, 0.12, n)

w0 = tangency_weights(mu, Sigma)
w1 = tangency_weights(mu + rng.normal(0, 0.01, n), Sigma)  # 预期收益加 1% 噪声

print(np.round(w0, 2))
print(np.round(w1, 2))
print("权重变化 L1:", np.abs(w1 - w0).sum().round(2))
# 典型结果：权重从 +0.8 跳到 -0.3 之类，L1 变化 > 100%
```

### 3.5 缓解手段一览

| 手段 | 做法 | 针对的问题 |
|---|---|---|
| 收缩协方差 | Ledoit-Wolf、因子模型 | $\Sigma$ 病态 |
| 收缩均值 | 向截面均值/零收缩；BL 先验 | $\mu$ 噪声大 |
| 不用 $\mu$ | GMV、风险平价、HRP | 彻底回避均值估计 |
| 加约束 | 权重上下限、行业中性、换手上限 | 防止极端头寸（约束本身等价于一种收缩，Jagannathan & Ma 2003） |
| 重抽样 | Michaud Resampled Frontier：bootstrap 多次求解再平均 | 平滑权重 |
| 正则化 | 目标函数加 $\gamma\lVert w\rVert_2^2$ | 权重集中 |

> **基准提醒**：DeMiguel, Garlappi & Uppal（2009）发现，在多数数据集上，许多"优化"组合的样本外表现并不能稳定地击败简单的 1/N 等权组合。任何优化方法都应该与 1/N 对比。

---

## 4. 协方差估计与 Ledoit-Wolf 收缩

### 4.1 样本协方差的问题

$$
S = \frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar r)(r_t - \bar r)^\top
$$

- $r_t \in \mathbb{R}^n$：第 $t$ 期的收益向量；$\bar r$：样本均值；$T$：样本期数

$S$ 是无偏的，但需要估计 $n(n+1)/2$ 个参数：300 只股票就是 45150 个参数。当 $n/T$ 不小时，$S$ 的特征值分布会严重"外扩"：大的更大，小的更小（随机矩阵理论中的 Marchenko-Pastur 分布描述了纯噪声下特征值的分布范围）。

### 4.2 收缩的直觉

> **比喻**：你想估计一个新开的餐厅的评分，只有 3 条评价。与其完全相信这 3 条（样本估计，方差大），不如把它往"全城餐厅的平均分"拉一拉（先验/目标，有偏但稳定）。收缩就是在"相信数据"和"相信结构"之间按最优比例折中——即偏差-方差权衡。

$$
\hat\Sigma_{\text{shrink}} = \delta F + (1-\delta) S, \qquad \delta \in [0, 1]
$$

- $F$：收缩目标（shrinkage target），一个结构简单、估计误差小的矩阵
- $\delta$：收缩强度。Ledoit & Wolf 的贡献是给出了**最小化期望 Frobenius 范数误差 $\mathbb{E}\lVert\hat\Sigma - \Sigma\rVert_F^2$ 的 $\delta$ 的一致估计量**，无需交叉验证

常见目标：

| 收缩目标 $F$ | 出处 | 特点 |
|---|---|---|
| 缩放单位阵 $\bar\sigma^2 I$ | Ledoit & Wolf (2004, JMVA) | sklearn `LedoitWolf` 的默认；把特征值向均值拉拢 |
| 常相关矩阵（保留各自方差，相关系数取平均） | Ledoit & Wolf (2004, "Honey, I Shrunk the Sample Covariance Matrix") | 适合股票 |
| 单因子（市场模型）协方差 | Ledoit & Wolf (2003) | 结构接近 CAPM |

### 4.3 代码

```python
import numpy as np
from sklearn.covariance import LedoitWolf, OAS

def compare_cov(returns: np.ndarray):
    """returns: T × n 的收益矩阵"""
    S = np.cov(returns, rowvar=False)
    lw = LedoitWolf().fit(returns)
    oas = OAS().fit(returns)
    for name, C in [("sample", S), ("ledoit-wolf", lw.covariance_), ("OAS", oas.covariance_)]:
        eig = np.linalg.eigvalsh(C)
        print(f"{name:12s} 条件数={eig.max() / eig.min():10.1f}")
    print("LW 收缩强度 δ =", round(lw.shrinkage_, 3))
    return lw.covariance_
```

样本外检验协方差好坏的实用方法：**用估计出的 $\hat\Sigma$ 构造 GMV 组合，看下一期组合的实现波动率**。实现波动越低，说明 $\hat\Sigma$ 越好。

### 4.4 其他协方差估计方法

**EWMA（指数加权）**：近期数据权重更大，适应波动率聚集（Phase 3 的 GARCH 思想）。

$$
\Sigma_t = \lambda \Sigma_{t-1} + (1-\lambda) r_{t-1} r_{t-1}^\top
$$

- $\lambda$：衰减因子。RiskMetrics 对日频数据采用 $\lambda = 0.94$，半衰期约 11 天

**因子模型协方差（Phase 7 的 Barra）**：

$$
\Sigma = B F B^\top + D
$$

- $B$：$n\times k$ 因子暴露矩阵（行业、风格）；$F$：$k\times k$ 因子收益协方差；$D$：对角的特异方差矩阵
- 参数从 $n(n+1)/2$ 降到约 $nk + k(k+1)/2 + n$，且天然正定、可解释。**股票组合生产环境的主流做法**

```python
import pandas as pd
# EWMA 协方差：pandas 内置
ewm_cov = returns_df.ewm(halflife=60).cov().loc[returns_df.index[-1]]
```

### 4.5 量化中的意义与陷阱

- 收缩不是只用于 MV：风险平价、HRP、风险预算、VaR 都需要 $\Sigma$，都应使用稳健估计。
- **对 A 股要注意停牌**：停牌期间收益为 0 会人为压低波动与相关性，计算前应把停牌日设为 NaN，再做成对（pairwise）或填补处理；但成对估计可能导致矩阵非正定，需修正（如特征值截断）。
- 风险模型要用**估计时点之前**的数据，不能用全样本估计后回测——这是常见的前视偏差。

---

## 5. 风险平价 Risk Parity

### 5.1 直觉：等权不等于等风险

60/40 股债组合"资金上"是 60% 股票，但由于股票波动大约是债券的 3-4 倍，**风险上往往 90% 以上来自股票**。风险平价的思想是：让每个资产对组合总风险的贡献相等。

> **比喻**：一支拔河队里有大力士和普通人。"等资金"是每人分配同样长度的绳子；"等风险"是让每个人实际出的力一样——大力士少拉一点绳，普通人多拉一点。

### 5.2 风险贡献公式（核心）

组合波动 $\sigma_p(w) = \sqrt{w^\top\Sigma w}$ 是 $w$ 的一次齐次函数，由 Euler 定理：

$$
\sigma_p = \sum_{i=1}^{n} w_i \frac{\partial \sigma_p}{\partial w_i} = \sum_{i=1}^{n} \underbrace{w_i \frac{(\Sigma w)_i}{\sigma_p}}_{RC_i}
$$

- $\frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$：边际风险贡献（MRC, Marginal Risk Contribution），即资产 $i$ 加一点仓位，组合波动增加多少
- $RC_i = w_i \cdot MRC_i$：资产 $i$ 的**风险贡献**（Risk Contribution），所有 $RC_i$ 相加恰好等于 $\sigma_p$
- $RC_i/\sigma_p$：百分比风险贡献，和为 1

**等风险贡献（ERC）组合**：$RC_i = RC_j,\ \forall i,j$。推广的**风险预算（Risk Budgeting）**：$RC_i/\sigma_p = b_i$，$b_i$ 是预设的风险预算。

特例：若所有资产两两相关系数相同，ERC 退化为**逆波动率加权** $w_i \propto 1/\sigma_i$。

### 5.3 求解：Spinu 凸化技巧

ERC 的方程组本身不是凸问题，但 Spinu（2013）给出了等价的凸优化：

$$
\min_{y > 0}\ \frac{1}{2}y^\top\Sigma y - \sum_{i=1}^{n} b_i \ln y_i, \qquad w = \frac{y}{\mathbf{1}^\top y}
$$

一阶条件 $(\Sigma y)_i = b_i / y_i$，即 $y_i(\Sigma y)_i = b_i$，恰好是"风险贡献与预算成比例"。对数障碍项保证 $y>0$（风险平价天然是多头组合）。

```python
import numpy as np
import cvxpy as cp

def risk_contribution(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    sigma_p = np.sqrt(w @ Sigma @ w)
    mrc = Sigma @ w / sigma_p          # 边际风险贡献
    return w * mrc                     # 风险贡献，sum == sigma_p

def risk_budget_weights(Sigma: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    n = len(Sigma)
    b = np.ones(n) / n if budget is None else budget / budget.sum()
    y = cp.Variable(n)
    obj = 0.5 * cp.quad_form(y, Sigma) - b @ cp.log(y)
    cp.Problem(cp.Minimize(obj)).solve()
    w = y.value / y.value.sum()
    return w

# 验证
w = risk_budget_weights(Sigma)
rc = risk_contribution(w, Sigma)
print(np.round(rc / rc.sum(), 4))   # 应接近 1/n
```

### 5.4 量化中的意义与陷阱

| 维度 | 说明 |
|---|---|
| **优点** | 不需要 $\mu$；对 $\Sigma$ 的误差比 MV 稳健得多；权重平滑 |
| **适用** | 大类资产配置（股、债、商品、黄金）；多策略/多因子之间的风险分配；CTA 多品种（Phase 5 的波动率目标就是它的近亲） |
| **杠杆问题** | 风险平价组合低波动资产（债券）权重很高，整体波动低，要达到目标收益通常需加杠杆（如桥水 All Weather 的思路）。杠杆带来融资成本与利率风险 |
| **隐含假设** | 等价于假设各资产夏普比率相同、且相关结构特定——这是一个信念，不是中立 |
| **相关性突变** | 2022 年全球股债同跌，经典风险平价组合大幅回撤：风险平价假设的"股债负相关"会失效 |
| **个股层面** | 对上千只个股做 ERC 意义不大（个股风险大部分是特异风险），常在**因子或资产类别层面**做风险预算 |

---

## 6. Black-Litterman 模型

### 6.1 直觉：先相信市场，再按信心加入自己的观点

MV 的问题是 $\mu$ 难估计且优化器极端敏感。Black & Litterman（1990/1992，高盛）提出：

1. **先验**：假设市场是均衡的，从市值权重**反推**市场隐含的预期收益 $\Pi$
2. **观点**：你可以有若干观点（"A 比 B 多涨 2%"），每个观点有信心度
3. **后验**：用贝叶斯公式融合先验与观点，得到 $\mu_{BL}$，再做 MV 优化

> **比喻**：GPS 导航（先验 = 市场共识路线）+ 你对本地交通的了解（观点）。你越确定"这条路在修"，就越偏离 GPS；没有观点时，就老老实实跟着 GPS 走。结果是一个"在市场组合附近按观点偏移"的组合，而不是 MV 那种极端的角点解。

如果你学过卡尔曼滤波（Phase 6），BL 的后验公式与卡尔曼的更新步骤是同一个结构：**精度加权平均**。

### 6.2 反向优化：市场隐含收益

若市场组合 $w_{\text{mkt}}$ 是效用最大化 $\max w^\top\mu - \frac{\delta}{2}w^\top\Sigma w$ 的解，则一阶条件给出：

$$
\Pi = \delta\, \Sigma\, w_{\text{mkt}}
$$

- $\Pi$：隐含均衡超额收益向量
- $\delta$：市场风险厌恶系数，可用 $\delta = \frac{\mathbb{E}[r_m] - r_f}{\sigma_m^2}$ 估计，常取 2-3 左右
- $w_{\text{mkt}}$：市值权重（A 股可用沪深 300 或中证 800 的成分权重）

### 6.3 观点的表达

$$
P\mu = Q + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Omega)
$$

- $P$：$k\times n$ 观点矩阵，每行是一个观点涉及的资产组合
- $Q$：$k$ 维观点收益
- $\Omega$：$k\times k$ 观点不确定性（对角阵，值越小信心越强）

例：3 个资产，观点 1"资产 1 绝对收益 5%"，观点 2"资产 2 比资产 3 多 2%"：

$$
P = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \end{pmatrix},\quad Q = \begin{pmatrix} 0.05 \\ 0.02 \end{pmatrix}
$$

### 6.4 后验公式

先验 $\mu \sim \mathcal{N}(\Pi, \tau\Sigma)$，融合观点后：

$$
\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\right]
$$

$$
\Sigma_{BL} = \Sigma + \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1}
$$

- $\tau$：标量，表示"对均衡收益估计的不确定性"，常取 0.025–0.05（也有取 $1/T$ 的做法）
- $(\tau\Sigma)^{-1}$ 与 $P^\top\Omega^{-1}P$：分别是先验与观点的**精度**（方差的逆）
- $\mu_{BL}$ 就是"按精度加权"的先验与观点的平均
- He & Litterman（1999）建议 $\Omega = \text{diag}(P\,\tau\Sigma\,P^\top)$，使观点不确定性与资产本身风险成比例；Idzorek（2005）提出用 0–100% 的直观信心度来反推 $\Omega$

### 6.5 代码

```python
import numpy as np

def black_litterman(Sigma, w_mkt, P, Q, delta=2.5, tau=0.05, Omega=None):
    pi = delta * Sigma @ w_mkt                         # 市场隐含收益
    tS = tau * Sigma
    if Omega is None:                                  # He-Litterman 默认
        Omega = np.diag(np.diag(P @ tS @ P.T))
    inv_tS = np.linalg.inv(tS)
    inv_Om = np.linalg.inv(Omega)
    M = np.linalg.inv(inv_tS + P.T @ inv_Om @ P)       # 后验均值的协方差
    mu_bl = M @ (inv_tS @ pi + P.T @ inv_Om @ Q)
    Sigma_bl = Sigma + M
    return pi, mu_bl, Sigma_bl

def unconstrained_mv(mu, Sigma, delta=2.5):
    return np.linalg.solve(delta * Sigma, mu)         # w* = (δΣ)^{-1} μ

# 无观点时：mu_bl == pi，最优权重 == w_mkt（可以用这个做单元测试）
```

一个漂亮的性质：**无观点时，BL 的最优组合正好是市场组合**；有观点时，只有观点涉及的资产权重偏离市场组合。

### 6.6 量化中的意义与陷阱

- **观点从哪来**：量化场景下，观点可以是因子模型的预测（例如"高动量行业组合比低动量行业组合多 1%/月"），$\Omega$ 由该因子的历史 IC 稳定性决定。
- **基准选择**：A 股市值权重高度集中在金融、消费等大盘股，用不同指数作为 $w_{\text{mkt}}$ 会显著改变先验。
- **$\tau$ 与 $\Omega$ 的尺度**：只有 $\tau$ 与 $\Omega$ 的相对大小影响 $\mu_{BL}$；文献中参数化方式不统一，阅读不同实现（如 PyPortfolioOpt、riskfolio-lib）时要注意。
- BL 解决的是 **$\mu$ 的问题**，不解决 $\Sigma$ 的问题，$\Sigma$ 仍应使用收缩或因子模型。

---

## 7. 层次风险平价 HRP

### 7.1 直觉：先分组，再在组间分钱

López de Prado（2016，"Building Diversified Portfolios that Outperform Out of Sample"）指出，MV 类方法把所有资产视为一张"全连接图"，任何一个资产都可以替代其他资产，导致 $\Sigma^{-1}$ 不稳定。HRP 的做法：

1. **树聚类**：按相关性把资产组织成一棵层次树（像行业 → 子行业 → 个股）
2. **准对角化**：重排协方差矩阵，让相似资产相邻
3. **递归二分**：自顶向下，在每次二分的两个子簇之间按**逆方差**分配权重

> **比喻**：分家产。不是让每个亲戚直接和所有其他亲戚比较（MV），而是先按家族分支分（大房、二房），再在每一房内部往下分。每一层只比较两个"分支"的风险，永远不需要对整个大矩阵求逆。

### 7.2 算法步骤与公式

**Step 1：相关性距离**

$$
d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})}
$$

- $\rho_{ij}$：相关系数。$\rho=1 \Rightarrow d=0$，$\rho=-1 \Rightarrow d=1$。这是一个合法的距离度量

原文还计算了"距离的距离" $\tilde d_{ij} = \sqrt{\sum_k (d_{ki} - d_{kj})^2}$（把每个资产的距离向量当作特征），再用 single linkage 层次聚类。

**Step 2：准对角化（Quasi-Diagonalization）**：按聚类树的叶子顺序重排资产，使大的协方差集中在对角线附近。

**Step 3：递归二分（Recursive Bisection）**：对当前排序列表一分为二，记左右子簇为 $L$、$R$：

$$
\tilde V_L = \tilde w_L^\top \Sigma_L \tilde w_L, \quad \tilde w_L \propto \frac{1}{\text{diag}(\Sigma_L)}, \qquad
\alpha = 1 - \frac{\tilde V_L}{\tilde V_L + \tilde V_R}
$$

- $\tilde V_L$：子簇 $L$ 内部用逆方差加权得到的簇方差
- 左子簇整体权重乘 $\alpha$，右子簇乘 $1-\alpha$；风险越小的子簇分到越多
- 递归直到每个子簇只剩一个资产

### 7.3 代码

```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)

def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0.0, None))      # 防止浮点误差出现负数
    # 简化版：直接对距离矩阵做 single linkage（原文用的是"距离的距离"）
    link = linkage(squareform(dist.values, checks=False), method="single")
    order = cov.index[leaves_list(link)].tolist()              # 准对角化顺序

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        # 每个长度 > 1 的簇一分为二
        clusters = [c[s:e] for c in clusters if len(c) > 1
                    for s, e in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.reindex(cov.index)
```

### 7.4 方法对比

| 方法 | 需要 $\mu$ | 需要 $\Sigma^{-1}$ | 权重稳定性 | 样本内最优 | 典型问题 |
|---|---|---|---|---|---|
| MV / 最大夏普 | 是 | 是 | 差 | 是 | 极端权重 |
| GMV | 否 | 是 | 中 | 是（方差） | 集中于低波资产 |
| 逆波动率 | 否 | 否 | 好 | 否 | 忽略相关性 |
| ERC | 否 | 否（迭代求解） | 好 | 否 | 低波资产权重高 |
| HRP | 否 | 否 | 好 | 否 | 结果依赖聚类方法；二分点按列表中点切而非按树切 |

### 7.5 量化中的意义与陷阱

- HRP 在**资产数多、样本短、协方差接近奇异**时最有优势，例如上百个 ETF、CTA 品种、多策略组合。
- 原版"按列表中点二分"会切断聚类树的自然分支，后续有改进版本（如按树的结构二分的 HERC 等），riskfolio-lib 中有实现。
- HRP 只是风险分配方法，不含 alpha。股票选股组合中常见用法是"先选股（alpha），再用 HRP/ERC 分配权重"。
- 原论文的"样本外优于 MV"结论主要来自蒙特卡洛实验，实际效果要在你的数据上用滚动样本外检验确认。

---

## 8. cvxpy 带约束组合优化

### 8.1 为什么生产环境用约束优化

真实的股票组合（尤其是**指数增强**）需要同时满足一堆限制：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     一个指数增强组合的典型约束                         │
├──────────────────────────────────────────────────────────────────────┤
│  目标：max  α'w  -  λ·风险  -  成本                                    │
│                                                                      │
│  预算约束    Σw = 1，全额投资                                          │
│  多头约束    w ≥ 0（A 股融券困难，基本只能做多）                         │
│  个股约束    w_i - w_b,i ≤ 1%（相对基准的主动权重上限）                  │
│  行业中性    每个行业的主动权重 = 0（或 |·| ≤ 0.5%）                     │
│  风格约束    市值、Beta 等风格因子主动暴露 |·| ≤ 0.1σ                    │
│  跟踪误差    年化 TE ≤ 5%                                              │
│  换手约束    单边换手 ≤ 20%/期                                         │
│  成分股约束  成分股权重之和 ≥ 80%                                       │
│  交易约束    停牌、涨停（买不进）、跌停（卖不出）的股票不能交易          │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 凸优化与 DCP

cvxpy 基于 **DCP（Disciplined Convex Programming）** 规则：只要你按规则组合凸/凹函数，cvxpy 就能自动验证问题是凸的，并转换给求解器（OSQP、Clarabel、ECOS、SCS，商业的 MOSEK、Gurobi 等）。

> 对软件工程师的类比：DCP 像一个**类型系统**。`quad_form(w, Σ)` 是"凸"类型，`log(x)` 是"凹"类型；`Minimize(凸)` 合法，`Minimize(凹)` 会被 cvxpy 拒绝（`DCPError`），就像编译器拒绝类型错误。

常见元素的凸性：

| 表达式 | 凸性 | 组合含义 |
|---|---|---|
| `w @ alpha` | 仿射 | 预期 alpha |
| `cp.quad_form(w, Sigma)` | 凸（$\Sigma$ 半正定） | 方差 |
| `cp.norm1(w - w0)` | 凸 | 换手 |
| `cp.sum_squares(B.T @ w)` | 凸 | 因子暴露惩罚 |
| `cp.pos(-r @ w - z)` | 凸 | CVaR 中的尾部损失 |
| 持股数量 $\le K$ | **非凸** | 需要整数变量（MIQP），求解难得多 |

### 8.3 目标函数

$$
\max_{w}\ \underbrace{\alpha^\top w}_{\text{预期 alpha}} - \lambda\,\underbrace{(w - w_b)^\top \Sigma (w - w_b)}_{\text{主动风险（跟踪误差平方）}} - \underbrace{c^\top |w - w_0|}_{\text{线性交易成本}}
$$

- $w_b$：基准权重；$w - w_b$：**主动权重**（active weight）
- $w_0$：当前持仓；$c$：每只股票的单边成本率（佣金 + 印花税 + 冲击的线性近似）
- 若 $\Sigma = BFB^\top + D$（因子模型），主动风险 $= a^\top B F B^\top a + a^\top D a$，$a = w - w_b$

**计算技巧**：不要把 $BFB^\top + D$ 展开成 $n\times n$ 矩阵，而是引入因子暴露 $f = B^\top a$（$k$ 维），风险写成 $f^\top F f + \sum_i D_i a_i^2$。$n=3000$、$k=40$ 时速度相差数十倍。

### 8.4 代码：指数增强优化器

```python
import numpy as np
import cvxpy as cp

def index_enhance_opt(
    alpha, B, F, D, w_bench, w_prev, industry,      # industry: n×m 的行业哑变量矩阵
    style_idx, lam=50.0, cost=0.002, max_active=0.01,
    style_tol=0.1, max_turnover=0.2, te_annual=0.05, tradable=None,
):
    n = len(alpha)
    w = cp.Variable(n)
    a = w - w_bench                                 # 主动权重
    f = B.T @ a                                     # 主动因子暴露（k 维）
    risk = cp.quad_form(f, cp.psd_wrap(F)) + cp.sum(cp.multiply(D, cp.square(a)))
    trade = cp.norm1(w - w_prev)

    objective = cp.Maximize(alpha @ w - lam * risk - cost * trade)
    cons = [
        cp.sum(w) == 1,
        w >= 0,
        a <= max_active,
        industry.T @ a == 0,                        # 行业中性
        cp.abs(B[:, style_idx].T @ a) <= style_tol, # 风格暴露限制
        trade <= 2 * max_turnover,                  # 单边换手 = 0.5 * L1
        risk <= (te_annual ** 2) / 252,             # 日度 TE 约束（F、D 为日度时）
    ]
    if tradable is not None:                        # 不可交易股票保持原仓位
        frozen = ~tradable
        cons.append(w[frozen] == w_prev[frozen])

    prob = cp.Problem(objective, cons)
    prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"优化失败: {prob.status}")
    return w.value
```

要点：

- `cp.psd_wrap(F)` 告诉 cvxpy "$F$ 是半正定的"，避免因浮点误差导致的 PSD 检查失败
- `risk <= ...` 是二次约束（QCQP），cvxpy 会转为 SOCP
- `prob.status` 必须检查：约束太紧时会 `infeasible`

### 8.5 量化中的意义与陷阱

| 陷阱 | 说明与对策 |
|---|---|
| **不可行（infeasible）** | 例如行业中性 + 换手 20% + 某行业股票全部停牌 → 无解。对策：把硬约束改成目标函数中的惩罚项（软约束），或分级放松 |
| **alpha 尺度** | $\alpha$ 与风险项量纲必须匹配。如果直接把 z-score 当 $\alpha$，$\lambda$ 的含义就失去意义。Grinold 的经验公式：$\alpha_i = \text{IC}\cdot\sigma_i\cdot z_i$（$\sigma_i$ 为残差波动，$z_i$ 为标准化分数） |
| **优化器只是放大器** | 优化器会利用风险模型的漏洞：风险模型没覆盖的风险（如某个主题、拥挤交易），优化器会以为"免费"而重仓 |
| **transfer coefficient** | 约束越多，最终持仓与原始 alpha 的相关性（TC）越低，见 12.4 |
| **数值稳定** | 预期收益按日度、风险按日度，单位统一；数值太小时可整体放大（如乘以 1e4，以基点计） |

---

## 9. VaR 与 CVaR

### 9.1 直觉

- **VaR（Value at Risk）**："在 99% 的日子里，单日亏损不会超过 X"——即损失分布的 99% 分位数
- **CVaR / ES（Conditional VaR / Expected Shortfall）**："在最差的 1% 的日子里，平均亏多少"——尾部损失的均值

> **比喻**：VaR 告诉你"洪水线在哪"，但不告诉你超过洪水线后水有多深；CVaR 告诉你"一旦淹了，平均水深多少"。肥尾资产（Phase 3）的 VaR 可能看起来还行，CVaR 却很吓人。

### 9.2 定义

记损失 $L = -r$（亏损为正），置信水平 $\alpha$（如 0.99）：

$$
\text{VaR}_\alpha = \inf\{\ell : P(L \le \ell) \ge \alpha\}, \qquad
\text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]
$$

- $\text{VaR}_\alpha$：损失分布的 $\alpha$ 分位数
- $\text{CVaR}_\alpha$：超过 VaR 的损失的条件期望（连续分布下的定义），始终 $\ge \text{VaR}_\alpha$

**一致性风险度量（Artzner et al., 1999）**：CVaR 满足次可加性 $\rho(A+B) \le \rho(A) + \rho(B)$（合并组合风险不会变大），VaR 不满足——两个各自 VaR 很小的头寸合在一起，VaR 可能变大，这与"分散化降低风险"的直觉矛盾。巴塞尔协议在市场风险新框架（FRTB）中用 97.5% ES 取代了 99% VaR 作为核心度量。

### 9.3 三种计算方法

| 方法 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **历史模拟法** | 直接用过去 N 天组合收益的经验分位数 | 无分布假设，保留肥尾和非线性 | 受样本窗口限制；尾部样本少；"没见过的风险"算不出来 |
| **参数法（方差-协方差法）** | 假设正态，$\text{VaR} = -(\mu + z_{1-\alpha}\sigma)$ | 快、可解析分解 | 正态假设低估肥尾；不适合期权等非线性头寸 |
| **蒙特卡洛法** | 从拟合的分布（多元 t、GARCH、copula）模拟大量情景 | 灵活，可处理非线性与路径依赖 | 结果依赖模型设定；计算量大 |

参数法公式（正态）：

$$
\text{VaR}_\alpha = -(\mu + z_{1-\alpha}\,\sigma), \qquad
\text{CVaR}_\alpha = -\mu + \sigma\frac{\phi(z_{1-\alpha})}{1-\alpha}
$$

- $z_{1-\alpha} = \Phi^{-1}(1-\alpha)$：标准正态分位数（$\alpha = 0.99$ 时约为 $-2.326$）
- $\phi$：标准正态密度；组合层面 $\sigma = \sqrt{w^\top\Sigma w}$

**Cornish-Fisher 修正**：用偏度 $S$ 和超额峰度 $K$ 修正分位数，部分考虑肥尾：

$$
z_{cf} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36}
$$

### 9.4 代码

```python
import numpy as np
from scipy import stats

def var_cvar_historical(r: np.ndarray, alpha=0.99):
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()

def var_cvar_parametric(r: np.ndarray, alpha=0.99):
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu + z * sigma), -mu + sigma * stats.norm.pdf(z) / (1 - alpha)

def var_cornish_fisher(r: np.ndarray, alpha=0.99):
    mu, sigma = r.mean(), r.std(ddof=1)
    S, K = stats.skew(r), stats.kurtosis(r)          # kurtosis 默认返回超额峰度
    z = stats.norm.ppf(1 - alpha)
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36
    return -(mu + z_cf * sigma)

def var_cvar_monte_carlo(w, mu, Sigma, alpha=0.99, n_sims=100_000, dof=4, seed=0):
    """多元 t 分布模拟：比正态更肥尾"""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma * (dof - 2) / dof)   # 缩放使模拟协方差 = Sigma
    z = rng.standard_normal((n_sims, len(w))) @ L.T
    chi = rng.chisquare(dof, size=(n_sims, 1))
    sims = mu + z * np.sqrt(dof / chi)
    port = sims @ w
    return var_cvar_historical(port, alpha)
```

### 9.5 时间缩放与 VaR 回测

**平方根法则**：$\text{VaR}_{h\text{天}} \approx \sqrt{h}\cdot\text{VaR}_{1\text{天}}$，仅在收益独立同分布、均值可忽略时成立。存在波动聚集、自相关或趋势时会失真；A 股涨跌停导致连续跌停时，多日损失远超平方根推算。

**VaR 回测（Kupiec POF 检验）**：如果 99% VaR 模型正确，突破（实际损失 > VaR）的比例应接近 1%。

$$
LR_{POF} = -2\ln\left[(1-p)^{T-x}p^{x}\right] + 2\ln\left[\left(1-\tfrac{x}{T}\right)^{T-x}\left(\tfrac{x}{T}\right)^{x}\right] \sim \chi^2(1)
$$

- $p = 1-\alpha$：理论突破率；$T$：观测天数；$x$：实际突破次数
- $LR > 3.84$（5% 显著性）则拒绝模型。还应检查突破是否**聚集**（Christoffersen 独立性检验）

### 9.6 CVaR 优化（Rockafellar-Uryasev）

CVaR 可以写成一个线性规划，因此能直接作为优化目标或约束：

$$
\text{CVaR}_\alpha(w) = \min_{\zeta}\ \zeta + \frac{1}{(1-\alpha)S}\sum_{s=1}^{S}\max(-r_s^\top w - \zeta,\ 0)
$$

- $r_s$：第 $s$ 个情景（历史或模拟）的收益向量；$S$：情景数
- 最优的 $\zeta$ 恰好是 VaR

```python
def min_cvar_portfolio(R: np.ndarray, alpha=0.95):
    """R: S × n 情景收益矩阵"""
    S, n = R.shape
    w, zeta = cp.Variable(n), cp.Variable()
    cvar = zeta + cp.sum(cp.pos(-R @ w - zeta)) / ((1 - alpha) * S)
    cp.Problem(cp.Minimize(cvar), [cp.sum(w) == 1, w >= 0]).solve()
    return w.value, cvar.value
```

---

## 10. 压力测试与情景分析

### 10.1 为什么 VaR 不够

VaR/CVaR 依赖历史分布或模型假设。真正的危机往往是**分布本身变了**：相关性飙升趋近 1、流动性消失、涨跌停无法成交。压力测试回答的是："如果发生 X，我会亏多少？"

> **比喻**：VaR 是根据过去天气统计算出的"正常年份最大降雨量"；压力测试是直接问"如果再来一次 1998 年的特大洪水，大坝撑得住吗？"

### 10.2 三类压力测试

```
┌──────────────────────────────────────────────────────────────────────┐
│                          压力测试的三种方式                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 历史情景      把当前持仓放回历史危机区间，计算损失                   │
│                  例：2015 年 6-8 月股灾、2016 年 1 月熔断、              │
│                      2020 年 2 月初疫情开盘、2024 年初小微盘股急跌        │
│                                                                      │
│  ② 假设情景      人为设定因子冲击："市场 -10%、小盘因子 -5%、            │
│                  波动率翻倍"，通过因子暴露传导到组合                     │
│                                                                      │
│  ③ 反向压力测试  倒推："亏损 20% 需要发生什么？" 找出组合最脆弱的方向     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.3 因子冲击传导

组合收益的因子分解：$r_p = x_p^\top f + u_p$，其中 $x_p = B^\top w$ 是组合的因子暴露。对因子施加冲击 $\Delta f$：

$$
\Delta r_p \approx x_p^\top \Delta f
$$

问题是：你只设定了部分因子（如市场 -10%），其他因子会怎么动？**条件情景**用因子协方差推断其余因子的条件期望：

$$
\mathbb{E}[\Delta f_2 \mid \Delta f_1] = \Sigma_{21}\Sigma_{11}^{-1}\Delta f_1
$$

- $\Delta f_1$：被设定冲击的因子；$\Delta f_2$：其余因子
- $\Sigma_{21}, \Sigma_{11}$：因子协方差矩阵的分块。危机时应使用**危机期间估计的协方差**，正常时期的相关性会低估联动

```python
import numpy as np
import pandas as pd

def conditional_stress(x_p: pd.Series, F: pd.DataFrame, shocks: dict) -> float:
    """x_p: 组合因子暴露；F: 因子协方差；shocks: {因子名: 冲击幅度}"""
    s = list(shocks)
    o = [c for c in F.index if c not in shocks]
    d1 = np.array([shocks[k] for k in s])
    d2 = F.loc[o, s].values @ np.linalg.solve(F.loc[s, s].values, d1)
    df = pd.Series(np.r_[d1, d2], index=s + o)
    return float(x_p.reindex(df.index).fillna(0) @ df)

def historical_scenario(weights: pd.Series, returns: pd.DataFrame, start: str, end: str) -> float:
    """当前持仓在历史区间的累计收益（简单持有，不调仓）"""
    cum = (1 + returns.loc[start:end, weights.index].fillna(0)).prod() - 1
    return float(weights @ cum)
```

### 10.4 A 股特有的压力维度

| 维度 | 说明 |
|---|---|
| **流动性/涨跌停** | 股灾期间大量个股连续跌停，**有价无市**。压力测试应假设跌停股票无法卖出，损失按连续跌停计 |
| **小盘/微盘风格崩塌** | 很多量化选股组合隐含小市值暴露；2024 年初小微盘股快速下跌，导致大量中性策略产品出现较大回撤 |
| **对冲工具失效** | 股指期货贴水扩大、融券受限会让"市场中性"产品的对冲成本骤升 |
| **相关性趋近 1** | 危机中"千股跌停"，分散化失效 |
| **政策冲击** | 交易规则变化（如熔断机制）、监管对特定交易方式的限制 |

### 10.5 实践建议

- 压力测试结果要配合**限额**：例如"任一历史情景下损失不得超过 15%"，并作为优化约束或事前风控检查。
- 把压力测试纳入每日风险报告：VaR、CVaR、最大因子暴露、前 5 大情景损失。
- 反向压力测试最有价值：它往往揭示你没意识到的集中暴露（"原来我的组合 70% 风险来自小市值"）。

---

## 11. 绩效归因：Brinson 与因子归因

### 11.1 为什么要归因

组合跑赢基准 3%，是因为**选对了行业**、**选对了个股**，还是**意外押中了小盘风格**？归因回答"钱从哪里赚的"，用于：

- 验证 alpha 来源是否符合策略设计（声称选股能力，结果全是风格 beta？）
- 发现未预期的风险暴露
- 向投资者解释业绩

### 11.2 Brinson 归因（BHB 1986）

按行业（或板块）$i$ 拆分。记 $w_{p,i}, w_{b,i}$ 为组合、基准在行业 $i$ 的权重，$R_{p,i}, R_{b,i}$ 为组合、基准在行业 $i$ 内的收益：

$$
R_p - R_b = \sum_i \Big[ \underbrace{(w_{p,i} - w_{b,i})R_{b,i}}_{\text{配置效应 AR}} + \underbrace{w_{b,i}(R_{p,i} - R_{b,i})}_{\text{选择效应 SR}} + \underbrace{(w_{p,i} - w_{b,i})(R_{p,i} - R_{b,i})}_{\text{交互效应 IR}} \Big]
$$

- **配置效应（Allocation）**：超配了表现好的行业吗？
- **选择效应（Selection）**：在行业内选出了跑赢行业的股票吗？
- **交互效应（Interaction）**：在选股好的行业上也超配了吗？

**Brinson-Fachler（BF 1985）改进**：配置效应改为 $(w_{p,i} - w_{b,i})(R_{b,i} - R_b)$。区别：超配一个**跑输整体基准**的行业，即使该行业收益为正，也应记为负贡献。实践中 BF 更常用。

```
                    基准行业收益 R_b,i      组合行业收益 R_p,i
                 ┌────────────────────┬────────────────────┐
  基准权重 w_b,i │  Q1 = 基准收益      │  Q2 = 选择后         │
                 ├────────────────────┼────────────────────┤
  组合权重 w_p,i │  Q3 = 配置后        │  Q4 = 组合收益       │
                 └────────────────────┴────────────────────┘
     配置 = Q3 - Q1     选择 = Q2 - Q1     交互 = Q4 - Q3 - Q2 + Q1
```

```python
import pandas as pd

def brinson(df: pd.DataFrame, fachler: bool = True) -> pd.DataFrame:
    """df 列：industry, w_p, w_b, r_p, r_b（行业内收益）"""
    d = df.set_index("industry")
    Rb = (d.w_b * d.r_b).sum()
    base = d.r_b - Rb if fachler else d.r_b
    out = pd.DataFrame({
        "allocation": (d.w_p - d.w_b) * base,
        "selection": d.w_b * (d.r_p - d.r_b),
        "interaction": (d.w_p - d.w_b) * (d.r_p - d.r_b),
    })
    out.loc["合计"] = out.sum()
    # 校验：合计三项之和 == R_p - R_b
    return out
```

注意：组合没有持有某行业时 $R_{p,i}$ 未定义，常用基准行业收益填充（选择效应记 0）。

**多期链接**：单期归因可加，但多期复利收益不可简单相加（$\prod(1+R_p) - \prod(1+R_b) \ne \sum(R_{p,t} - R_{b,t})$）。常用 Carino 或 Menchero 平滑系数做多期链接。

### 11.3 因子归因（基于 Barra 风险模型）

Brinson 只能回答"行业维度"，而量化组合更关心风格因子。利用 Phase 7 的截面回归得到的因子收益 $f_t$：

$$
r_{p,t} - r_{b,t} = \underbrace{(x_p - x_b)^\top f_t}_{\text{因子贡献}} + \underbrace{(u_{p,t} - u_{b,t})}_{\text{特异贡献（选股 alpha）}}
$$

- $x_p - x_b$：组合相对基准的**主动因子暴露**（由 $B^\top(w_p - w_b)$ 计算）
- $f_t$：第 $t$ 期因子收益（行业因子 + 风格因子如 Size、Beta、Momentum、Value 等）
- 特异贡献 $=$ 主动收益 $-$ 因子贡献：这才是"真正的选股能力"

**风险归因**同理：主动风险 $\sigma_A^2 = a^\top B F B^\top a + a^\top D a$，可以把跟踪误差拆成"多少来自风格、多少来自行业、多少来自特异"，并用第 5 节的风险贡献公式逐因子拆解。

```python
def factor_attribution(w_p, w_b, B: pd.DataFrame, f: pd.Series, r: pd.Series) -> pd.Series:
    """单期：B 为期初暴露（n×k），f 为当期因子收益（k），r 为当期个股收益（n）"""
    a = w_p - w_b
    x_active = B.T @ a                         # 主动暴露
    factor_contrib = x_active * f              # 每个因子的贡献
    total_active = a @ r
    out = factor_contrib.copy()
    out["特异收益"] = total_active - factor_contrib.sum()
    out["主动收益合计"] = total_active
    return out
```

### 11.4 量化中的意义与陷阱

- **暴露用期初值**：归因必须用期初（持仓形成时）的暴露，用期末暴露是前视。
- **特异收益稳定为正**才说明选股有效；如果主动收益主要来自 Size 因子，策略本质上是"小盘 beta"，风格反转时会大幅回撤（见第 10 节 2024 年初的例子）。
- 归因结果依赖所用的风险模型；换一套因子定义，"alpha"可能被重新解释为某个因子的贡献。

---

## 12. 再平衡与换手控制

### 12.1 为什么要关心换手

alpha 模型每天都在变，理论最优持仓也每天在变。但每次交易都有成本：

$$
\text{净 alpha} = \text{毛 alpha} - \text{换手率} \times \text{单位交易成本}
$$

一个月度调仓、单边换手 50%、单边成本 0.2% 的策略，年化成本约 $12\times 0.5 \times 2 \times 0.2\% = 2.4\%$（买入、卖出各计一次）。对于年化超额 5% 的指数增强，这已经是一半的 alpha。

**A 股交易成本构成**（以普通股票为例，具体以券商和交易所现行规定为准）：

| 项目 | 方向 | 量级 |
|---|---|---|
| 印花税 | 仅卖出 | 0.05%（2023 年 8 月 28 日起由 0.1% 减半） |
| 佣金 | 双向 | 约万分之 1 – 万分之 3，常有最低 5 元 |
| 过户费 | 双向 | 很小 |
| 冲击成本 + 滑点 | 双向 | 依赖规模和流动性，小盘股常是最大头 |

### 12.2 换手的定义

$$
\text{单边换手率} = \frac{1}{2}\sum_i |w_{i,\text{new}} - w_{i,\text{old}}^{\text{drift}}|
$$

- $w^{\text{drift}}_{\text{old}}$：上期持仓经过本期价格变动**漂移后**的权重（不是上期目标权重）
- 也有人用 $\sum|\Delta w|$（双边），比较不同报告时要确认口径

### 12.3 再平衡策略

| 策略 | 做法 | 优缺点 |
|---|---|---|
| **定期再平衡** | 每月/每周调回目标 | 简单；可能在不需要时交易 |
| **阈值再平衡** | 偏离目标超过阈值才调 | 减少无效交易；阈值需调参 |
| **无交易区间（No-Trade Region）** | 在目标周围设一个区间，只把超出区间的部分调到区间**边界**，而不是调回中心 | 理论上在比例成本下最优（Davis & Norman 等的结论） |
| **部分调仓** | 每次只向目标移动一部分：$w_{\text{new}} = w_0 + \eta(w^* - w_0)$ | 平滑交易；等价于对 alpha 做时间平滑 |
| **优化内控制** | 在目标函数中加换手惩罚/约束（第 8 节） | 最灵活，可与 alpha、风险统一权衡 |

```python
import numpy as np

def drift_weights(w: np.ndarray, period_ret: np.ndarray) -> np.ndarray:
    """持仓随价格漂移"""
    v = w * (1 + period_ret)
    return v / v.sum()

def no_trade_band(w_cur: np.ndarray, w_target: np.ndarray, band: float = 0.005) -> np.ndarray:
    """只把超出 ±band 的部分调到边界；之后需重新归一化或用现金吸收"""
    lo, hi = w_target - band, w_target + band
    w_new = np.clip(w_cur, lo, hi)
    return w_new / w_new.sum()

def turnover(w_new, w_old_drift) -> float:
    return 0.5 * np.abs(w_new - w_old_drift).sum()
```

### 12.4 交易成本模型与市场冲击

线性成本只是近似。大额交易的**市场冲击**通常遵循"平方根法则"（Phase 11 详述）：

$$
\text{冲击成本（比例）} \approx c\cdot\sigma_{\text{daily}}\sqrt{\frac{Q}{V}}
$$

- $Q$：交易量；$V$：日均成交量（ADV）；$\sigma_{\text{daily}}$：日波动率；$c$：经验常数，量级约为 1
- 含义：交易量翻 4 倍，单位成本翻 2 倍；**策略容量**由此决定

在优化中，冲击成本 $\propto |\Delta w|^{1.5}$ 仍是凸的，cvxpy 可用 `cp.power(cp.abs(dw), 1.5)` 表达。

### 12.5 主动管理基本定律与转移系数

Grinold（1989）的**主动管理基本定律**：

$$
IR \approx IC \times \sqrt{BR}
$$

- $IR$：信息比率（年化主动收益 / 跟踪误差）
- $IC$：预测与实际收益的相关性（Phase 7）
- $BR$（Breadth）：每年独立下注的次数

Clarke, de Silva & Thorley（2002）加入**转移系数 TC**：

$$
IR \approx TC \times IC \times \sqrt{BR}
$$

- $TC$：最终持仓（风险调整后的主动权重）与原始 alpha 的相关系数，$0 < TC \le 1$
- 约束（多头、行业中性、换手上限）越多，$TC$ 越低。纯多头 A 股组合的 TC 往往显著小于 1

**实践含义**：降低换手（降低成本）与保持 TC（兑现 alpha）是一对权衡。高频衰减的 alpha（如短期反转）需要高换手才能兑现；慢变量 alpha（如价值）可以低换手。观察 **IC 衰减曲线**（Phase 7）可以帮你决定调仓频率。

---

## 13. 最佳实践与常见陷阱

### 13.1 最佳实践

```
┌──────────────────────────────────────────────────────────────────────┐
│                         组合构建最佳实践清单                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  □ 所有估计（Σ、μ、因子暴露）只用 t 时刻之前的数据，滚动计算            │
│  □ 协方差用收缩或因子模型，永远不直接对样本协方差求逆                   │
│  □ 每个优化方法都与 1/N 等权、逆波动率做样本外对比                      │
│  □ alpha 先转换为预期收益尺度（IC × σ × z），再进入优化器               │
│  □ 交易成本放进目标函数，而不是回测完再扣                               │
│  □ 停牌/涨跌停股票冻结仓位，不作为可交易资产                            │
│  □ 检查优化状态码；不可行时有降级方案（放松约束/沿用上期持仓）          │
│  □ 风险报告每日输出：TE、VaR/CVaR、因子暴露、前 N 大情景损失             │
│  □ 定期做归因：特异收益是否稳定为正？风格暴露是否在预期内？              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.2 常见陷阱

| 陷阱 | 症状 | 对策 |
|---|---|---|
| 用历史均值当 $\mu$ | 权重极端、每期剧烈跳变 | 用 alpha 模型或 BL；或只用 GMV/RP/HRP |
| 全样本估计 $\Sigma$ 后回测 | 回测波动异常低、夏普虚高 | 滚动/扩展窗口估计 |
| $n > T$ 的样本协方差 | 矩阵奇异，求解失败或权重爆炸 | 因子模型 / 收缩 |
| 忽略权重漂移 | 换手率与成本被低估 | 用漂移后权重计算交易量 |
| 只看 VaR 不看 CVaR | 肥尾策略（卖期权、做空波动）看起来很安全 | 同时报告 CVaR 与压力测试 |
| 正态参数法 VaR | 尾部风险被系统性低估 | 历史法、Cornish-Fisher、t 分布蒙特卡洛 |
| 平方根法则外推长期 VaR | 低估多日风险（跌停、波动聚集） | 直接用多日收益或路径模拟 |
| 归因用期末暴露 | "alpha" 来源被错误归类 | 用期初暴露 |
| 风险模型盲区 | 优化器重仓风险模型未覆盖的主题 | 加入主题/拥挤度约束，个股权重上限 |
| 过度约束 | TC 很低，alpha 无法兑现 | 监控 TC，删除冗余约束 |

---

## 14. 总结

### 14.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────────┐
│                            核心概念回顾                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  组合优化                                                             │
│  ├── 均值方差：σ_p² = w'Σw；有效前沿；GMV 与切点组合                   │
│  ├── 估计误差：MV 是"误差最大化器"，μ 误差影响远大于 Σ                  │
│  ├── 协方差：Ledoit-Wolf 收缩 δF+(1-δ)S；EWMA；因子模型 BFB'+D         │
│  ├── 风险平价：RC_i = w_i(Σw)_i/σ_p，ΣRC_i = σ_p；Spinu 凸化求解       │
│  ├── Black-Litterman：Π = δΣw_mkt；先验 + 观点的精度加权               │
│  ├── HRP：相关距离聚类 → 准对角化 → 递归二分，无需求逆                  │
│  └── cvxpy：DCP 规则；指数增强 = alpha - λ·TE² - 成本 + 约束            │
│                                                                      │
│  风险管理                                                             │
│  ├── VaR：分位数；CVaR：尾部均值，满足次可加性                          │
│  ├── 历史 / 参数 / 蒙特卡洛三种方法；Kupiec 回测                        │
│  └── 压力测试：历史情景、因子冲击（条件传导）、反向压力测试              │
│                                                                      │
│  归因与交易                                                           │
│  ├── Brinson：配置 + 选择 + 交互；BF 版本相对基准总收益                  │
│  ├── 因子归因：主动收益 = 主动暴露 × 因子收益 + 特异收益                  │
│  └── 换手：漂移权重、无交易区间、冲击平方根法则、IR ≈ TC·IC·√BR         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 14.2 方法选择速查

| 场景 | 推荐方法 |
|---|---|
| 股债商品大类配置 | 风险平价 / 风险预算 |
| 有明确的少量主观/量化观点 | Black-Litterman |
| 资产多、历史短、相关结构复杂 | HRP |
| A 股指数增强、行业/风格中性 | cvxpy 约束优化 + 因子风险模型 |
| 尾部风险敏感（如期权、杠杆） | CVaR 优化 + 压力测试约束 |
| 不确定用什么 | 先跑 1/N 和逆波动率当基准 |

### 14.3 与下一阶段的衔接

Phase 9 将用机器学习（LightGBM 等）替代 Phase 7 的线性因子合成，产生新的 alpha 分数。本阶段的工具链在下一阶段直接复用：

- **Phase 9 的 `07-from-prediction-to-portfolio.py`**：ML 预测分数 → 预期收益尺度 → 本阶段的 cvxpy 优化器 / TopK 持仓
- **换手与成本**：ML 模型的预测往往比线性因子更"跳"，第 12 节的换手控制会变得更关键
- **归因**：用第 11 节的因子归因检验 ML 模型的超额收益究竟是新 alpha，还是已知风格因子（如小市值、反转）的非线性重组
- **过拟合检验**：Phase 9 的 Deflated Sharpe Ratio 与 PBO 同样适用于"在多个组合优化参数（λ、约束阈值）中挑最好的"这一过程

> **一句话总结**：alpha 决定你能赚多少，组合构建决定你能保住多少，风险管理决定你能活多久。
