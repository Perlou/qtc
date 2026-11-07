# Phase 12: 期权与衍生品 - 核心概念深度解析

> **学习目标**：掌握期权定价、希腊字母与波动率交易基础
>
> **前置知识**：Phase 3（正态分布、蒙特卡洛、随机游走）、Phase 11（做市与对冲思想）
>
> **阅读建议**：期权是整个课程中数学最密集的部分。每个概念都按「直觉/比喻 → 公式 → 代码 → 实际意义与陷阱」展开。推导部分第一次读可以跳过，先把代码跑通、建立数值直觉。

---

## 目录

1. [引言：为什么量化要学期权](#1-引言为什么量化要学期权)
2. [期权基础与收益结构](#2-期权基础与收益结构)
3. [定价直觉：无套利、复制与二叉树](#3-定价直觉无套利复制与二叉树)
4. [Black-Scholes 模型](#4-black-scholes-模型)
5. [希腊字母](#5-希腊字母)
6. [隐含波动率求解](#6-隐含波动率求解)
7. [波动率微笑与曲面](#7-波动率微笑与曲面)
8. [国内 ETF 期权](#8-国内-etf-期权)
9. [Delta 对冲与 Gamma-Theta 权衡](#9-delta-对冲与-gamma-theta-权衡)
10. [波动率策略与卖方尾部风险](#10-波动率策略与卖方尾部风险)
11. [最佳实践与常见陷阱](#11-最佳实践与常见陷阱)
12. [总结](#12-总结)

---

## 1. 引言：为什么量化要学期权

### 1.1 期权是"交易波动率"的工具

股票和期货让你表达**方向**观点（涨或跌）。期权则让你表达更丰富的观点：

| 你的观点 | 股票/期货能做吗 | 期权的做法 |
| --- | --- | --- |
| 会涨 | 买入 | 买认购 |
| 会涨，但最多亏 X 元 | 需要止损单，可能跳空 | 买认购，最大亏损 = 权利金 |
| 会大幅波动，但不知道方向 | 不能 | 买跨式（Straddle） |
| 会横盘不动 | 不能 | 卖跨式 / 铁鹰（Iron Condor） |
| 市场高估了未来波动 | 不能 | 卖期权并对冲 Delta |
| 持有股票但想防范暴跌 | 只能卖出 | 买认沽（保护性认沽） |

**核心认知**：期权价格里最重要的未知量是**波动率**。期权交易员常说"我们交易的不是价格，是波动率"。

### 1.2 本阶段地图

```
┌──────────────────────────────────────────────────────────────────┐
│                       Phase 12 知识地图                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  定价                                                             │
│    01 期权基础 ──▶ 02 BS 与二叉树 ──▶ 03 希腊字母                  │
│                         │                                        │
│  波动率                  ▼                                        │
│    04 隐含波动率 ──▶ 05 微笑与曲面 ──▶ 06 国内 ETF 期权数据         │
│                         │                                        │
│  交易                    ▼                                        │
│    07 Delta 对冲（Gamma-Theta 权衡）──▶ 08 波动率策略与尾部风险     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 期权基础与收益结构

> 对应课程：`01-options-basics.py`

### 2.1 定义

**期权（Option）** 是一份合约：买方支付**权利金（Premium）**，获得在未来某个日期以约定价格买入或卖出某资产的**权利**（而非义务）；卖方收取权利金，承担相应的**义务**。

**比喻**：

- **认购期权（Call）** 像买房的"定金"：付 2 万定金锁定 300 万的房价。三个月后房价涨到 350 万，你行使权利，以 300 万买入，赚 50 万减去 2 万定金；房价跌到 250 万，你放弃，最多损失 2 万定金。
- **认沽期权（Put）** 像"保险"：付保费，资产价值跌破约定价时获得赔付。

| 术语 | 英文 | 含义 |
| --- | --- | --- |
| 认购期权 | Call | 以行权价**买入**标的的权利 |
| 认沽期权 | Put | 以行权价**卖出**标的的权利 |
| 行权价 | Strike, $K$ | 约定的买卖价格 |
| 到期日 | Expiry, $T$ | 权利的最后有效日 |
| 权利金 | Premium | 期权的价格，由买方支付给卖方 |
| 标的资产 | Underlying, $S$ | 期权对应的资产，如 50ETF |
| 欧式期权 | European | 只能在到期日行权（国内 ETF 期权属于欧式） |
| 美式期权 | American | 到期前任意时间可行权（国内部分商品期权为美式） |
| 合约乘数 | Multiplier | 一张合约对应的标的数量，国内 ETF 期权一般为 10000 份 |

### 2.2 四种基本头寸的到期收益

到期日 $T$ 时，标的价格为 $S_T$，行权价为 $K$：

$$
\text{Call 到期价值} = \max(S_T - K, 0), \qquad \text{Put 到期价值} = \max(K - S_T, 0)
$$

扣除权利金 $c$ 或 $p$ 后得到盈亏：

```
  买入认购 Long Call            卖出认购 Short Call
  盈亏                          盈亏
   ▲            ╱               ▲
   │          ╱                 │ ───────┐  +c（最大收益）
   │        ╱                   │        │╲
 0 ┼──────K╱──────▶ S_T       0 ┼────────K─╲─────▶ S_T
   │──────┘  −c（最大亏损）       │            ╲   （亏损无限）
   │                            │              ╲

  买入认沽 Long Put             卖出认沽 Short Put
  盈亏                          盈亏
   ▲╲                           ▲
   │  ╲                         │        ┌──────  +c（最大收益）
   │    ╲                       │       ╱
 0 ┼─────╲K──────▶ S_T        0 ┼──────╱K────────▶ S_T
   │       └────── −p           │    ╱
   │                            │  ╱  （最大亏损 = K − p，标的归零时）
```

**不对称性**是期权的本质：买方亏损有限、收益可能很大；卖方收益有限、亏损可能很大。

```python
import numpy as np

def payoff(S_T, K, kind="call", position=1, premium=0.0):
    """到期盈亏；position=+1 买入，-1 卖出"""
    intrinsic = np.maximum(S_T - K, 0) if kind == "call" else np.maximum(K - S_T, 0)
    return position * (intrinsic - premium)

S_T = np.linspace(2.5, 3.5, 5)
print(payoff(S_T, K=3.0, kind="call", position=+1, premium=0.08))
print(payoff(S_T, K=3.0, kind="put",  position=-1, premium=0.06))
```

### 2.3 内在价值、时间价值与虚实程度

$$
\text{期权价格} = \underbrace{\text{内在价值}}_{\text{立即行权能得到的}} + \underbrace{\text{时间价值}}_{\text{未来可能变得更值钱的"希望"}}
$$

- 认购的内在价值 $= \max(S - K, 0)$，认沽 $= \max(K - S, 0)$
- 时间价值通常为正（美式期权恒 ≥ 0；欧式深度实值认沽在正利率下可能略为负），并随到期临近衰减到 0

| 状态 | 认购 | 认沽 | 特点 |
| --- | --- | --- | --- |
| 实值 ITM（In-the-Money） | $S > K$ | $S < K$ | 有内在价值，表现接近标的 |
| 平值 ATM（At-the-Money） | $S \approx K$ | $S \approx K$ | 时间价值最大，Gamma 最大 |
| 虚值 OTM（Out-of-the-Money） | $S < K$ | $S > K$ | 只有时间价值，便宜、杠杆高、大概率归零 |

### 2.4 认购-认沽平价（Put-Call Parity）

**直觉**：构造两个组合，到期时价值完全相同：

- 组合 A：买 1 份认购 + 存入现金 $K e^{-rT}$（到期变成 $K$）
- 组合 B：买 1 份认沽 + 持有 1 份标的

到期时：若 $S_T > K$，A 值 $S_T - K + K = S_T$，B 值 $0 + S_T = S_T$；若 $S_T \le K$，A 值 $K$，B 值 $K - S_T + S_T = K$。既然到期价值永远相同，**今天的价格也必须相同**，否则就有无风险套利：

$$
C - P = S - K e^{-rT}
$$

- $C, P$：同一标的、同一行权价、同一到期日的欧式认购与认沽价格
- $r$：连续复利无风险利率；$T$：剩余期限（年）
- 若标的有分红（连续股息率 $q$），则为 $C - P = S e^{-qT} - K e^{-rT}$

**实际意义**：

1. 由市场上的 $C - P$ 可以反推"隐含远期价格" $F = K + (C - P)e^{rT}$，这在计算隐含波动率时比直接用 ETF 现价更准确（自动包含分红和融资成本预期）。
2. 平价关系明显被破坏时，可能存在套利机会（但要扣除交易成本、保证金占用与卖空约束）。

```python
import numpy as np

def implied_forward(call_px, put_px, K, r, T):
    return K + (call_px - put_px) * np.exp(r * T)

print(implied_forward(call_px=0.1250, put_px=0.0980, K=3.0, r=0.02, T=0.1))
```

---

## 3. 定价直觉：无套利、复制与二叉树

> 对应课程：`02-black-scholes.py`

### 3.1 一步二叉树：用复制给期权定价

**问题**：股票现价 100，一年后要么涨到 120，要么跌到 90。无风险利率 5%（简化为单利/离散复利）。行权价 100 的一年期认购值多少？

**关键思想**：我们不需要知道涨跌的概率！因为可以用"股票 + 借钱"**复制**出和期权一模一样的收益。

```
            今天              一年后
                        ┌── S = 120  →  Call = 20
          S = 100 ──────┤
                        └── S = 90   →  Call = 0
```

设买 $\Delta$ 股股票、借款 $B$（到期还 $B \times 1.05$），要求两种情形下组合价值等于期权价值：

$$
120\Delta - 1.05B = 20, \qquad 90\Delta - 1.05B = 0
$$

解得 $\Delta = \frac{20 - 0}{120 - 90} = \frac{2}{3}$，$B = \frac{90 \times 2/3}{1.05} \approx 57.14$。

$$
C = 100\Delta - B = 66.67 - 57.14 = 9.52
$$

**这就是期权定价的全部秘密**：如果能用标的复制期权，期权价格就等于复制成本，否则存在套利。这里的 $\Delta$ 就是后面要讲的 **Delta**——也是对冲比例。

### 3.2 风险中性概率

把上面的结果改写成"贴现的期望"：

$$
C = \frac{1}{1+r}\big[p \cdot C_u + (1-p) \cdot C_d\big], \qquad p = \frac{(1+r) - d}{u - d}
$$

- $u = 1.2,\ d = 0.9$：上涨和下跌倍数
- $C_u = 20,\ C_d = 0$：两种情形下的期权价值
- $p = \frac{1.05 - 0.9}{1.2 - 0.9} = 0.5$：**风险中性概率**（Risk-Neutral Probability）

验算：$C = \frac{0.5 \times 20}{1.05} = 9.52$，与复制法一致。

**关键认知**：$p$ 不是"真实的上涨概率"，而是一个计算工具——在这个"虚拟世界"中，所有资产的期望收益都等于无风险利率。股票的真实期望收益 $\mu$ 不出现在定价公式里，因为它的影响已经被"可以对冲"这件事抵消了。

### 3.3 多步二叉树：CRR 模型

把 $[0, T]$ 分成 $N$ 步，每步 $\Delta t = T/N$。Cox-Ross-Rubinstein (1979) 取：

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}, \qquad p = \frac{e^{r\Delta t} - d}{u - d}
$$

- $\sigma$：年化波动率
- 从到期日的收益开始，逐步向前"倒推"：每个节点价值 = 下一步两个子节点价值的风险中性期望再贴现
- 美式期权：每个节点取 $\max(\text{继续持有的价值}, \text{立即行权的价值})$

```
                                    S·u³
                           S·u² ──<
                  S·u ───<          S·u
         S ────<           S  ───<
                  S·d ───<          S·d
                           S·d² ──<
                                    S·d³
  t=0          Δt          2Δt          3Δt = T
  （由于 u·d = 1，树是"重合"的，N 步只有 N+1 个终端节点）
```

```python
import numpy as np

def crr_price(S, K, T, r, sigma, N=200, kind="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)   # 终端价格，从高到低
    V = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    for step in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            S_nodes = S * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
            exercise = np.maximum(S_nodes - K, 0) if kind == "call" else np.maximum(K - S_nodes, 0)
            V = np.maximum(V, exercise)
    return V[0]

print(f"欧式认沽: {crr_price(3.0, 3.0, 0.5, 0.03, 0.2, kind='put'):.5f}")
print(f"美式认沽: {crr_price(3.0, 3.0, 0.5, 0.03, 0.2, kind='put', american=True):.5f}")
```

**实际意义**：二叉树直观、能处理美式期权和离散分红；当 $N \to \infty$ 时欧式期权价格收敛到 Black-Scholes 公式（4.5 节会验证）。

---

## 4. Black-Scholes 模型

> 对应课程：`02-black-scholes.py`

### 4.1 模型假设

| 假设 | 含义 | 现实 |
| --- | --- | --- |
| 标的服从几何布朗运动 | 对数收益正态分布，波动率 $\sigma$ 为常数 | 真实收益肥尾、波动率随时间变化（Phase 3 GARCH） |
| 无风险利率 $r$ 为常数 | 可以以 $r$ 自由借贷 | 借贷利率不同 |
| 无交易成本、可连续交易 | 可以随时无成本调整对冲 | 只能离散对冲，有成本 |
| 无套利、可卖空 | | 存在卖空限制 |
| 无分红（或连续股息率 $q$） | | ETF 有分红，需调整 |

**几何布朗运动（GBM）**：

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t
$$

- $\mu$：漂移（期望收益率）；$\sigma$：波动率；$W_t$：布朗运动，$dW_t \sim N(0, dt)$
- 直觉：价格的"百分比变化" = 确定性趋势 + 随机冲击。这保证价格永远为正。
- 解为 $S_T = S_0 \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right]$，$Z \sim N(0,1)$

### 4.2 Black-Scholes 公式

欧式认购与认沽：

$$
C = S\, N(d_1) - K e^{-rT} N(d_2)
$$

$$
P = K e^{-rT} N(-d_2) - S\, N(-d_1)
$$

$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

| 符号 | 含义 |
| --- | --- |
| $S$ | 标的当前价格 |
| $K$ | 行权价 |
| $T$ | 剩余期限，**以年为单位**（如 30 个自然日 ≈ 30/365） |
| $r$ | 连续复利无风险利率 |
| $\sigma$ | 年化波动率（对数收益的标准差） |
| $N(\cdot)$ | 标准正态分布的累积分布函数（CDF） |

**公式的直觉解读**：

$$
C = \underbrace{S\, N(d_1)}_{\text{到期"拿到股票"的现值}} - \underbrace{K e^{-rT}}_{\text{行权价的现值}} \times \underbrace{N(d_2)}_{\text{风险中性下到期行权的概率}}
$$

- $N(d_2)$：在风险中性世界里 $S_T > K$ 的概率
- $N(d_1)$：认购期权的 Delta，即复制组合中应持有的股票数量（3.1 节中的 $\Delta$）
- 所以 BS 公式就是 3.1 节复制组合的连续时间版本：**持有 $N(d_1)$ 份股票，借入 $K e^{-rT} N(d_2)$ 现金**

### 4.3 推导：风险中性期望

> 学习检查要求"能推导 BS 公式"，这里给出最直接的推导路径。

**第一步**：在风险中性测度下，标的期望收益等于 $r$，因此

$$
S_T = S \exp\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\, Z\right], \quad Z \sim N(0, 1)
$$

**第二步**：期权价格 = 到期收益的风险中性期望再贴现：

$$
C = e^{-rT}\, E\big[(S_T - K)^+\big] = e^{-rT} E\big[S_T \mathbb{1}_{\{S_T > K\}}\big] - K e^{-rT} P(S_T > K)
$$

**第三步**：计算行权概率。$S_T > K$ 等价于

$$
Z > \frac{\ln(K/S) - (r - \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}} = -d_2 \quad \Longrightarrow \quad P(S_T > K) = N(d_2)
$$

**第四步**：计算 $E[S_T \mathbb{1}_{\{Z > -d_2\}}]$。记 $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ 为标准正态密度：

$$
E\big[e^{\sigma\sqrt{T}Z}\mathbb{1}_{\{Z > -d_2\}}\big] = \int_{-d_2}^{\infty} e^{\sigma\sqrt{T}z}\phi(z)\,dz = e^{\frac{1}{2}\sigma^2 T}\int_{-d_2}^{\infty}\phi(z - \sigma\sqrt{T})\,dz = e^{\frac{1}{2}\sigma^2 T} N(d_2 + \sigma\sqrt{T})
$$

（第二个等号是"配方"：$\sigma\sqrt{T}z - \frac{z^2}{2} = -\frac{(z - \sigma\sqrt{T})^2}{2} + \frac{\sigma^2 T}{2}$）

于是 $E[S_T \mathbb{1}_{\{S_T>K\}}] = S e^{(r - \frac{1}{2}\sigma^2)T} \cdot e^{\frac{1}{2}\sigma^2 T} N(d_1) = S e^{rT} N(d_1)$，其中 $d_1 = d_2 + \sigma\sqrt{T}$。

**第五步**：代回得到 $C = S N(d_1) - K e^{-rT} N(d_2)$。认沽由平价关系得到。∎

另一条推导路径是构造"期权 − Δ 份标的"的无风险组合，得到 **Black-Scholes 偏微分方程**：

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

- $V(S, t)$：期权价值；用第 5 节的希腊字母记号即 $\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + r S \Delta - r V = 0$
- 这个方程在第 9 节会被用来解释 **Gamma-Theta 权衡**

### 4.4 代码实现与验证

```python
import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes(-Merton) 欧式期权价格；q 为连续股息率"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

S, K, T, r, sigma = 3.0, 3.0, 0.25, 0.02, 0.20
c, p = bs_price(S, K, T, r, sigma, "call"), bs_price(S, K, T, r, sigma, "put")
print(f"Call={c:.5f}  Put={p:.5f}")
print(f"平价检验 C-P={c - p:.6f}  S-Ke^(-rT)={S - K * np.exp(-r * T):.6f}")

# 蒙特卡洛验证：风险中性期望
rng = np.random.default_rng(0)
Z = rng.standard_normal(1_000_000)
ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
print(f"MC Call={np.exp(-r * T) * np.maximum(ST - K, 0).mean():.5f}")
```

把 3.3 节的 `crr_price` 与 `bs_price` 对比，可以观察到二叉树随步数增加收敛（并带有奇偶步数的振荡）：

```python
# 需先运行 3.3 节的 crr_price 与上面的 bs_price
for N in [10, 50, 200, 1000]:
    print(N, round(crr_price(3.0, 3.0, 0.25, 0.02, 0.20, N=N), 6))
print("BS", round(bs_price(3.0, 3.0, 0.25, 0.02, 0.20), 6))
```

### 4.5 一个经验公式：平值期权的快速估算

当 $S = K$、$r \approx 0$ 时，$C \approx P \approx 0.4\, S \sigma \sqrt{T}$（因为 $\frac{1}{\sqrt{2\pi}} \approx 0.4$）。

例：50ETF 价格 3 元、波动率 20%、一个月到期：$0.4 \times 3 \times 0.2 \times \sqrt{1/12} \approx 0.069$ 元，一张合约（10000 份）约 690 元。这个公式在面试和盘中心算时非常有用。

### 4.6 陷阱

| 陷阱 | 说明 |
| --- | --- |
| 时间单位 | $T$ 必须用年；用交易日（/252）还是自然日（/365）要与波动率的年化口径一致 |
| 波动率口径 | $\sigma$ 是对数收益年化标准差，20% 写作 0.20 而不是 20 |
| 分红 | ETF 分红会降低认购价值，应使用含 $q$ 的版本或用隐含远期价格 |
| 利率 | 国内实务常用 Shibor 或国债收益率近似，短期限期权对 $r$ 不敏感 |
| 模型不是真理 | BS 的价值在于提供"价格 ↔ 波动率"的统一换算语言，而不在于其假设正确 |

---

## 5. 希腊字母

> 对应课程：`03-greeks.py`

### 5.1 总览：期权价格的"仪表盘"

**比喻**：开车时，速度表告诉你位置变化多快（Delta），加速度表告诉你速度变化多快（Gamma），油表告诉你每天消耗多少（Theta），路况告诉你颠簸对你的影响（Vega）。希腊字母就是期权价格对各个输入变量的**敏感度**（偏导数）。

| 希腊字母 | 定义 | 含义 | 工程类比 |
| --- | --- | --- | --- |
| Delta $\Delta$ | $\partial V / \partial S$ | 标的变动 1 元，期权变动多少 | 一阶导数 / 速度 |
| Gamma $\Gamma$ | $\partial^2 V / \partial S^2$ | 标的变动 1 元，Delta 变动多少 | 二阶导数 / 加速度 |
| Vega $\nu$ | $\partial V / \partial \sigma$ | 波动率变动 1 个单位，期权变动多少 | 对"不确定性"的敏感度 |
| Theta $\Theta$ | $\partial V / \partial t$ | 时间流逝一天，期权变动多少 | 每日"租金"/衰减 |
| Rho $\rho$ | $\partial V / \partial r$ | 利率变动，期权变动多少 | 短期期权中通常不重要 |

### 5.2 公式

记 $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ 为标准正态密度，无分红情形：

| 希腊字母 | 认购 Call | 认沽 Put |
| --- | --- | --- |
| Delta | $N(d_1)$ | $N(d_1) - 1$ |
| Gamma | $\dfrac{\phi(d_1)}{S\sigma\sqrt{T}}$ | 同左 |
| Vega | $S\phi(d_1)\sqrt{T}$ | 同左 |
| Theta | $-\dfrac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$ | $-\dfrac{S\phi(d_1)\sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2)$ |
| Rho | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

**单位约定**（务必统一，否则数值差几百倍）：

- Vega 公式对应 $\sigma$ 变动 1.00（即 100 个波动率点），实务常报"每 1 个波动率点"，需除以 100
- Theta 公式是"每年"，实务常报"每天"，需除以 365（自然日）或 252（交易日）

### 5.3 逐个理解

**Delta**：

- 认购 Delta ∈ [0, 1]，认沽 Delta ∈ [−1, 0]
- 平值期权 Delta 约 ±0.5；深度实值趋向 ±1（像标的本身），深度虚值趋向 0
- 也是**对冲比例**：卖出 1 张 Delta=0.6 的认购，需买入 0.6 × 合约乘数份标的来对冲
- 常被粗略地当作"到期成为实值的概率"，但严格来说风险中性行权概率是 $N(d_2)$

**Gamma**：

- 买入期权（无论认购认沽）Gamma 为正，卖出为负
- 平值、临近到期时 Gamma 最大——这时 Delta 对标的价格极其敏感
- **Long Gamma**：标的涨时 Delta 变大（越涨越多头），跌时 Delta 变小（越跌越空头）→ 天然"追涨杀跌"，波动越大越赚
- **Short Gamma**：相反，大幅波动对你不利

**Vega**：

- 买入期权 Vega 为正：隐含波动率上升，期权变贵
- 平值、长期限期权 Vega 最大

**Theta**：

- 买入期权 Theta 通常为负：每过一天，时间价值流失
- 平值期权在临近到期时 Theta 衰减最快

```
       Gamma / Theta 随标的价格             时间价值随剩余期限
         （平值附近最大）                     （临近到期加速衰减）
  Γ ▲                                    价值 ▲
    │        ╱╲                              │●
    │       ╱  ╲   ← 短期限：尖              │  ●●
    │      ╱    ╲                            │     ●●
    │   ╱‾‾      ‾‾╲  ← 长期限：平           │        ●●●
    │ ╱              ╲                       │            ●●
    └──────────K──────────▶ S                │              ●
                                             └──────────────────●▶ 到期
                                              90天     30天    0天
```

### 5.4 代码：解析希腊字母与数值验证

```python
import numpy as np
from scipy.stats import norm

def bs_greeks(S, K, T, r, sigma, kind="call"):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT / 100                          # 每 1 个波动率点
    if kind == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf * sigma / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf * sigma / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

S, K, T, r, sig = 3.0, 3.0, 30 / 365, 0.02, 0.20
g = bs_greeks(S, K, T, r, sig)
h = 1e-4
fd_delta = (bs_call(S + h, K, T, r, sig) - bs_call(S - h, K, T, r, sig)) / (2 * h)
fd_gamma = (bs_call(S + h, K, T, r, sig) - 2 * bs_call(S, K, T, r, sig) + bs_call(S - h, K, T, r, sig)) / h**2
fd_theta = bs_call(S, K, T - 1 / 365, r, sig) - bs_call(S, K, T, r, sig)
print({k: round(v, 6) for k, v in g.items()})
print(f"有限差分: delta={fd_delta:.6f} gamma={fd_gamma:.6f} theta(1天)={fd_theta:.6f}")
```

> **工程提示**：用有限差分校验解析公式是非常好的单元测试方法。

### 5.5 组合的希腊字母可以相加

一个期权组合的希腊字母 = 各头寸希腊字母 × 数量 × 合约乘数之和：

$$
\Delta_{\text{组合}} = \sum_i n_i \cdot m_i \cdot \Delta_i
$$

- $n_i$：第 $i$ 个头寸的张数（卖出为负）
- $m_i$：合约乘数（如 10000）
- 风险管理中常把 Delta 换算为"等价标的金额" $\Delta_{\text{组合}} \times S$，把 Gamma 换算为"标的涨 1% 时 Delta 金额的变化"，便于跨品种汇总

---

## 6. 隐含波动率求解

> 对应课程：`04-implied-volatility.py`

### 6.1 直觉：从价格反推波动率

BS 公式中，$S, K, T, r$ 都能直接观察，只有 $\sigma$ 不能。反过来，给定市场价格 $C_{\text{mkt}}$，求使 $\text{BS}(\sigma) = C_{\text{mkt}}$ 的 $\sigma$，就是**隐含波动率（Implied Volatility, IV）**。

**比喻**：IV 就像债券的"到期收益率"。不同期限、不同票息的债券价格无法直接比较，但换算成收益率就可以。同样，不同行权价、不同期限的期权价格无法直接比较，换算成 IV 后就有了统一的"贵贱"度量。

```
         BS 公式（正向）
  σ ─────────────────────────▶ 期权价格
  σ ◀───────────────────────── 期权价格
        求解隐含波动率（反向）
```

因为认购价格对 $\sigma$ 单调递增（Vega > 0），解存在时唯一。

### 6.2 牛顿法

**思想**：在当前猜测点用切线近似函数，切线与目标值的交点作为新的猜测。BS 价格对 $\sigma$ 的导数正好是 Vega：

$$
\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - C_{\text{mkt}}}{\text{Vega}(\sigma_n)}
$$

- $\sigma_n$：第 $n$ 次迭代的波动率猜测
- $\text{Vega}(\sigma_n) = S\phi(d_1)\sqrt{T}$（这里用"每 1.00 波动率"的原始单位）

```
  价格 ▲
       │                        ╱ BS(σ)
  C_mkt┼ - - - - - - - - - - -●
       │                   ╱  ↑ 真解
       │               ╱ ╱
       │           ╱ ╱   ← 切线（斜率 = Vega）
       │       ╱  ●
       │   ╱      σ₀
       └───────────────────────────▶ σ
               σ₀    σ₁ → 迅速收敛到真解
```

**初始值**：可用 Brenner-Subrahmanyam 近似（由 4.5 节的平值公式反解）：

$$
\sigma_0 \approx \sqrt{\frac{2\pi}{T}} \cdot \frac{C_{\text{mkt}}}{S}
$$

### 6.3 代码：牛顿法 + 二分法兜底

```python
import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(price, S, K, T, r, tol=1e-8, max_iter=50):
    lower = max(S - K * np.exp(-r * T), 0.0)            # 无套利下界
    if not (lower < price < S):
        return np.nan                                    # 价格违反无套利边界，无解
    sigma = np.clip(np.sqrt(2 * np.pi / T) * price / S, 0.05, 2.0)
    for _ in range(max_iter):
        diff = bs_call(S, K, T, r, sigma) - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-8:                                     # Vega 太小，牛顿法不稳定
            break
        sigma = sigma - diff / v
        if not (1e-4 < sigma < 5.0):
            break
    lo, hi = 1e-4, 5.0                                   # 兜底：二分法，慢但稳
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

true_sigma = 0.235
px = bs_call(3.0, 3.1, 0.2, 0.02, true_sigma)
print(f"价格={px:.6f}  反解 IV={implied_vol(px, 3.0, 3.1, 0.2, 0.02):.6f}")
```

### 6.4 实务陷阱

| 问题 | 说明 | 处理 |
| --- | --- | --- |
| 深度虚值 / 临近到期 | Vega 极小，牛顿法发散 | 二分法或 Brent 法兜底（`scipy.optimize.brentq`） |
| 价格违反无套利边界 | 报价陈旧或买卖价差大，IV 无解 | 返回 NaN 并记录，不要强行求解 |
| 用最新成交价 | 深度虚值合约可能很久没成交 | 优先用买卖中间价，并检查价差宽度 |
| 认购/认沽 IV 不一致 | 远期价格估计不准（分红、利率） | 用平价关系反推隐含远期价格 $F$ |
| 批量计算性能 | 期权链有成百上千个合约 | 向量化牛顿法，或使用 `py_vollib` 等成熟库（基于 Jäckel 的高精度算法） |

---

## 7. 波动率微笑与曲面

> 对应课程：`05-volatility-surface.py`

### 7.1 微笑与偏斜

如果 BS 模型完全正确，同一标的、同一到期日的所有期权应该有**相同的** IV。但实际画出 IV 对行权价的曲线，会看到：

```
  IV ▲                                IV ▲
     │●                       ●          │●
     │  ●                   ●            │  ●
     │    ●               ●              │    ●
     │      ●  ●  ●  ● ●                 │      ●
     │                                   │        ●  ●
     │                                   │              ●  ●  ●
     └──────────────────────────▶ K      └──────────────────────────▶ K
            波动率微笑（Smile）                   波动率偏斜（Skew）
       外汇、部分商品期权常见                 股票指数期权的典型形态：
                                            低行权价（虚值认沽）IV 更高
```

**为什么会有偏斜？**

1. **肥尾**：真实收益分布比正态分布有更厚的尾部（Phase 3），极端行情的概率被 BS 低估，深度虚值期权需要更高的 IV 才能反映真实价格。
2. **崩盘恐惧**：股票指数暴跌往往比暴涨更剧烈。1987 年美股"黑色星期一"之后，股指期权的偏斜变得非常显著且长期存在。
3. **供求关系**：机构大量买入虚值认沽做组合保护，推高了低行权价期权的 IV。
4. **杠杆效应**：股价下跌 → 公司杠杆上升 → 波动率上升，价格与波动率负相关。

### 7.2 期限结构

同一行权价（如平值）、不同到期日的 IV 曲线叫**波动率期限结构（Term Structure）**：

- 平静市场中通常向上倾斜（远期 IV > 近期 IV）
- 恐慌时常倒挂（近期 IV 飙升，高于远期）
- 已知事件（如重要数据发布、财报）会让覆盖该事件的到期月 IV 抬升

### 7.3 波动率曲面

把行权价（或在值程度）和到期期限两个维度合起来，就是**波动率曲面（Volatility Surface）**：

$$
\sigma_{\text{imp}} = \Sigma(k, T), \qquad k = \ln(K / F)
$$

- $k$：对数在值程度（Log-Moneyness），$F$ 为对应到期日的远期价格。用 $k$ 而不是 $K$ 可以让不同期限、不同标的价格水平的曲线可比
- 实务中也常用 Delta（如 25 Delta 认沽、平值、25 Delta 认购）作为横坐标

```
                        波动率曲面示意
     IV
      ▲
      │ ╲
      │  ╲ ╲            ← 近月：偏斜更陡
      │   ╲  ╲ ╲
      │    ‾‾╲  ╲ ╲
      │        ‾‾─╲─╲─────  ← 远月：更平坦
      │              ‾‾‾───────
      └──────────────────────────────▶ k = ln(K/F)
     ╱
    ╱ T（到期期限）
```

### 7.4 参数化：SVI 模型

原始的 IV 点稀疏、有噪声，需要拟合成平滑曲面。Gatheral 提出的 **SVI（Stochastic Volatility Inspired）** 参数化，对每个到期日拟合总隐含方差 $w = \sigma_{\text{imp}}^2 T$：

$$
w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + s^2}\right]
$$

| 参数 | 含义 |
| --- | --- |
| $a$ | 整体方差水平 |
| $b$ | 两翼斜率（$b \ge 0$） |
| $\rho$ | 偏斜方向（$-1 < \rho < 1$，股指通常为负） |
| $m$ | 曲线水平平移 |
| $s$ | 平值附近曲率（$s > 0$，原文献中记为 $\sigma$，这里换个字母避免与波动率混淆） |

```python
import numpy as np
from scipy.optimize import least_squares

def svi_w(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))

def fit_svi(k, iv, T):
    w_mkt = iv**2 * T
    def resid(p):
        return svi_w(k, *p) - w_mkt
    x0 = [w_mkt.min(), 0.1, -0.5, 0.0, 0.1]
    bounds = ([-1, 0, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 2])
    return least_squares(resid, x0, bounds=bounds).x

T = 30 / 365
k = np.linspace(-0.15, 0.10, 11)
iv_mkt = 0.20 - 0.35 * k + 1.5 * k**2 + np.random.default_rng(0).normal(0, 0.003, k.size)
params = fit_svi(k, iv_mkt, T)
iv_fit = np.sqrt(svi_w(k, *params) / T)
print("SVI 参数:", np.round(params, 4))
print("最大拟合误差(vol点):", np.round(np.abs(iv_fit - iv_mkt).max() * 100, 3))
```

### 7.5 无套利约束

拟合出来的曲面必须满足无套利条件，否则用它定价会产生"可套利"的价格：

- **日历价差无套利**：同一 $k$ 下，总方差 $w(k, T)$ 随 $T$ 不减
- **蝶式价差无套利**：同一到期日，认购价格对行权价是凸函数（等价于隐含的风险中性密度非负）

### 7.6 实际意义

- **相对价值**：哪些合约的 IV 相对曲面偏高/偏低？
- **风险管理**：Vega 风险要按期限和在值程度分桶，而不是一个数
- **"粘性"假设**：标的价格变动时曲面如何移动（Sticky Strike：按行权价不变；Sticky Delta / Sticky Moneyness：按在值程度不变），会影响 Delta 对冲比例的计算

---

## 8. 国内 ETF 期权

> 对应课程：`06-etf-options-data.py`

### 8.1 品种概览

| 品种 | 交易所 | 标的代码 | 上市时间 |
| --- | --- | --- | --- |
| 上证 50ETF 期权 | 上交所 | 510050 | 2015-02-09（国内首个场内股票期权） |
| 沪深 300ETF 期权 | 上交所 | 510300 | 2019-12-23 |
| 沪深 300ETF 期权 | 深交所 | 159919 | 2019-12-23 |
| 中证 500ETF 期权 | 上交所 | 510500 | 2022-09-19 |
| 中证 500ETF 期权 | 深交所 | 159922 | 2022-09-19 |
| 创业板 ETF 期权 | 深交所 | 159915 | 2022-09-19 |
| 深证 100ETF 期权 | 深交所 | 159901 | 2022-12-12 |
| 科创 50ETF 期权 | 上交所 | 588000、588080 | 2023-06-05 |

此外，中金所有沪深 300、中证 1000、上证 50 **股指期权**（现金交割），大商所、郑商所、上期所、广期所等有多种**商品期权**。品种会持续增加，以交易所公告为准。

### 8.2 合约要素（以上交所 ETF 期权为例）

| 要素 | 内容 |
| --- | --- |
| 合约类型 | 认购期权、认沽期权 |
| 行权方式 | 欧式，到期日行权 |
| 交割方式 | 实物交割（交割 ETF 份额） |
| 合约单位 | 一般为 10000 份（分红后会调整） |
| 到期月份 | 当月、下月及随后两个季月 |
| 到期日 | 到期月份的第四个星期三（遇法定节假日顺延） |
| 最小报价单位 | 0.0001 元 |
| 行权价 | 按标的价格区间设定间距，随标的价格变动加挂新行权价 |
| 交易时间 | 与股票类似，含开盘/收盘集合竞价；**T+0** 交易 |

> 合约细则会修订，开发数据与交易系统前请以交易所最新合约规格为准。

**与股票的关键区别**：

- 期权可以 T+0 买卖，也可以"卖出开仓"（成为义务方），需缴纳保证金
- 实物交割：行权后得到/交付 ETF，行权日需要准备足额资金或 ETF
- 分红调整：ETF 分红时，期权的合约单位和行权价会做相应调整，调整后的合约在合约简称中通常带有 "A" 等标识。数据处理时**不能**简单假设合约单位恒为 10000

### 8.3 保证金：卖方的门票与紧箍咒

卖出开仓需缴纳保证金，交易所标准的计算思路是"期权价格 + 标的价格的一定比例 − 虚值额（并设下限）"。以上交所认购期权义务仓的开仓保证金为例：

$$
\text{保证金} = \left[\text{合约前结算价} + \max\big(12\% \times \text{标的前收盘价} - \text{认购虚值额},\ 7\% \times \text{标的前收盘价}\big)\right] \times \text{合约单位}
$$

- 认购虚值额 $= \max(K - S_{\text{前收}}, 0)$，即行权价高于标的前收盘价的部分（按每份计）
- 维持保证金每日根据结算价重新计算，行情不利时需追加，不足会被强行平仓

> 上式仅用于建立直觉，券商通常在交易所标准上加收，**以交易所与券商的最新标准为准**。

### 8.4 投资者适当性

个人投资者参与股票期权交易需满足交易所和券商的适当性管理要求，通常包括资产门槛、交易经历、期权知识测试、模拟交易经历和风险承受能力评估等，并按**分级权限**开放：一级通常只能做备兑开仓和保护性认沽，二级可买入开仓，三级才可卖出开仓（保证金交易）。具体条件**以交易所与券商最新规定为准**。

### 8.5 数据获取与清洗流程

```
┌──────────────────────────────────────────────────────────────────┐
│                     ETF 期权链处理流程                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 获取合约列表：代码、认购/认沽、行权价、到期日、合约单位          │
│        （AKShare 等开源库提供上交所/深交所期权接口，                │
│          函数名随版本变化，以其文档为准）                          │
│  2. 获取行情：买一/卖一价、最新价、成交量、持仓量                   │
│  3. 计算剩余期限 T（注意自然日/交易日口径）                        │
│  4. 过滤：成交量为 0、价差过宽、剩余期限过短（如 < 3 天）的合约      │
│  5. 用平价关系为每个到期日估计隐含远期价格 F                        │
│  6. 对每个合约求 IV（用中间价），选虚值一侧（OTM）拼接成微笑         │
│  7. 计算希腊字母，按到期日 × 在值程度汇总                          │
│  8. 拟合曲面（SVI 等），检查无套利                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

```python
import numpy as np
import pandas as pd

def estimate_forward(chain: pd.DataFrame, r: float, T: float) -> float:
    """chain 含 strike, call_mid, put_mid；取 |C-P| 最小的几个行权价估计远期"""
    c = chain.assign(diff=(chain.call_mid - chain.put_mid).abs()).nsmallest(3, "diff")
    F = c.strike + (c.call_mid - c.put_mid) * np.exp(r * T)
    return float(F.median())

chain = pd.DataFrame({
    "strike":   [2.9, 3.0, 3.1],
    "call_mid": [0.1320, 0.0650, 0.0240],
    "put_mid":  [0.0280, 0.0600, 0.1180],
})
print(f"隐含远期 F ≈ {estimate_forward(chain, r=0.02, T=30 / 365):.4f}")
```

**为什么选 OTM 一侧**：虚值期权只有时间价值，流动性通常更好；实值期权的价格大部分是内在价值，对 IV 不敏感，买卖价差带来的 IV 误差大。

---

## 9. Delta 对冲与 Gamma-Theta 权衡

> 对应课程：`07-delta-hedging.py`

### 9.1 直觉：把期权变成纯波动率头寸

卖出一张认购后，标的上涨你就亏钱。如果同时按 Delta 买入标的，小幅价格变动带来的损益就被抵消了，剩下的风险只与**波动有多大**有关，与方向无关。

**比喻**：Delta 对冲像骑自行车时不断微调车把。车把（标的持仓）要跟着车身倾斜（Delta 变化）不断修正，修正的频率和幅度取决于路面颠簸程度（Gamma 和实际波动率）。

```
┌──────────────────────────────────────────────────────────────────┐
│                   卖出认购 + Delta 对冲的循环                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   卖出认购，收权利金 ──▶ 买入 Δ 份标的                             │
│          ▲                     │                                 │
│          │                     ▼                                 │
│     到期结算              标的价格变动                             │
│          ▲                     │                                 │
│          │                     ▼                                 │
│   重复直到到期 ◀── 重新计算 Δ，调整标的持仓（涨了加买，跌了卖出）    │
│                                                                  │
│   卖方对冲：涨了买、跌了卖 = "高买低卖"，这就是 Short Gamma 的成本    │
│   收到的权利金（Theta）就是对这一成本的补偿                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 Gamma-Theta 权衡

对 Delta 中性的组合，在一小段时间 $\Delta t$ 内做泰勒展开：

$$
\Delta \Pi \approx \Theta\, \Delta t + \frac{1}{2}\Gamma (\Delta S)^2
$$

- $\Theta \Delta t$：时间流逝的损益（买方为负）
- $\frac{1}{2}\Gamma (\Delta S)^2$：价格波动的凸性损益（买方为正，无论涨跌）
- Delta 项因为已对冲而消失

由 BS 偏微分方程（4.3 节），在 $r \approx 0$ 时 $\Theta \approx -\frac{1}{2}\sigma_{\text{imp}}^2 S^2 \Gamma$。代入：

$$
\Delta \Pi \approx \frac{1}{2}\Gamma S^2 \left[\left(\frac{\Delta S}{S}\right)^2 - \sigma_{\text{imp}}^2 \Delta t\right]
$$

**这是期权交易最重要的一个公式**：

- $\left(\frac{\Delta S}{S}\right)^2$：这段时间**实际**发生的波动（实现方差）
- $\sigma_{\text{imp}}^2 \Delta t$：期权价格中**隐含**的波动（你为之付费的方差）
- **买入期权并对冲（Long Gamma）**：实现波动 > 隐含波动 → 赚钱；否则 Theta 损失大于 Gamma 收益
- **卖出期权并对冲（Short Gamma）**：相反

累计到到期，对冲后的总损益近似为：

$$
\text{PnL} \approx \sum_t \frac{1}{2}\Gamma_t S_t^2 \left(\sigma_{\text{real}}^2 - \sigma_{\text{imp}}^2\right)\Delta t
$$

| 头寸 | Gamma | Theta | 盼望什么 |
| --- | --- | --- | --- |
| 买入期权 + 对冲 | + | −（每天付"租金"） | 实现波动率 > 隐含波动率，大波动来得越早越好 |
| 卖出期权 + 对冲 | − | +（每天收"租金"） | 实现波动率 < 隐含波动率，市场平静 |

**比喻**：Long Gamma 像买保险——每天付保费（Theta），等待事故（大波动）；Short Gamma 像卖保险——每天收保费，但一次大事故就可能赔掉很多天的收入。

### 9.3 代码：Delta 对冲模拟

```python
import numpy as np
from scipy.stats import norm

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def call_price(S, K, T, r, sigma):
    x = d1(S, K, T, r, sigma)
    return S * norm.cdf(x) - K * np.exp(-r * T) * norm.cdf(x - sigma * np.sqrt(T))

def hedge_short_call(sigma_imp, sigma_real, S0=3.0, K=3.0, T=30 / 252,
                     r=0.0, n_steps=30, n_paths=2000, seed=0):
    """以 sigma_imp 卖出平值认购，并按 sigma_imp 计算的 Delta 离散对冲；返回每份期权的 PnL"""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    pnl = np.empty(n_paths)
    for p in range(n_paths):
        S = S0
        delta = norm.cdf(d1(S, K, T, r, sigma_imp))
        cash = call_price(S, K, T, r, sigma_imp) - delta * S   # 收权利金，买入 Δ 份标的
        for i in range(1, n_steps + 1):
            S *= np.exp(-0.5 * sigma_real**2 * dt + sigma_real * np.sqrt(dt) * rng.standard_normal())
            tau = T - i * dt
            new_delta = norm.cdf(d1(S, K, tau, r, sigma_imp)) if tau > 1e-12 else float(S > K)
            cash -= (new_delta - delta) * S                     # 调整对冲仓位
            delta = new_delta
        pnl[p] = cash + delta * S - max(S - K, 0.0)             # 平掉标的，支付期权到期价值
    return pnl

for s_imp, s_real in [(0.25, 0.20), (0.20, 0.20), (0.20, 0.25)]:
    for n in [10, 30, 120]:
        pnl = hedge_short_call(s_imp, s_real, n_steps=n)
        print(f"隐含={s_imp:.2f} 实现={s_real:.2f} 对冲次数={n:>3}: "
              f"PnL 均值={pnl.mean():+.4f}  标准差={pnl.std():.4f}")
```

典型结果解读：

- 隐含 > 实现时卖方平均盈利，隐含 < 实现时平均亏损，隐含 = 实现时平均约为零——**期权卖方赚的是"隐含 − 实现"的波动率差**
- 对冲越频繁，PnL 标准差越小（离散对冲误差下降，大致与 $1/\sqrt{\text{对冲次数}}$ 成比例），但均值不变
- 平值期权的权利金约 0.10 元，平均 PnL 约 0.02 元，说明波动率判断对了也只赚权利金的一小部分

### 9.4 对冲频率与交易成本

更频繁对冲 → 误差更小，但交易成本更高。实务中常见的对冲规则：

| 规则 | 做法 | 特点 |
| --- | --- | --- |
| 固定时间 | 每 N 分钟/每日收盘对冲 | 简单，但大行情中可能来不及 |
| Delta 阈值（带宽） | 组合 Delta 偏离超过阈值才对冲 | 节省成本，常用 |
| 价格阈值 | 标的价格每变动 X% 对冲一次 | 与 Gamma 风险更匹配 |
| 最优带宽 | Whalley-Wilmott 等模型给出与成本、Gamma 相关的最优带宽 | 理论最优，参数需估计 |

### 9.5 陷阱

- **用哪个波动率算 Delta**：用隐含波动率还是预测的实现波动率算 Delta，会得到不同的 PnL 路径（均值相同，路径方差不同）。
- **Gamma 在到期附近爆炸**：平值期权临近到期时 Gamma 极大，Delta 在 0 和 1 之间剧烈跳动，对冲成本急剧上升（"Pin Risk"）。
- **跳空无法对冲**：Delta 对冲假设价格连续变动，隔夜跳空或涨跌停时无法及时调整。
- **A 股约束**：50ETF、300ETF 等股票类 ETF 实行 T+1（跨境、债券、黄金等部分 ETF 可 T+0），当日买入的对冲仓位次日才能卖出；**卖出对冲**需要已有持仓或融券，还要考虑最小交易单位。实务中也常用股指期货或期权合成头寸做对冲。

---

## 10. 波动率策略与卖方尾部风险

> 对应课程：`08-volatility-strategies.py`

### 10.1 策略地图

```
                       方向观点
              看跌          中性          看涨
           ┌──────────┬──────────────┬──────────┐
  看涨波动 │ 买认沽     │ 买跨式        │ 买认购    │
  (Long    │           │ 买宽跨式      │           │
   Vol)    │           │              │           │
           ├──────────┼──────────────┼──────────┤
  中性波动 │ 熊市价差   │ 日历价差      │ 牛市价差   │
           ├──────────┼──────────────┼──────────┤
  看跌波动 │ 卖认购     │ 卖跨式/宽跨式  │ 卖认沽    │
  (Short   │ (备兑)    │ 铁鹰 / 蝶式   │           │
   Vol)    │           │              │           │
           └──────────┴──────────────┴──────────┘
```

### 10.2 常见策略

| 策略 | 构成 | 最大收益 | 最大亏损 | 观点 |
| --- | --- | --- | --- | --- |
| 买入跨式 Long Straddle | 买同一 K 的认购 + 认沽 | 理论无限 | 两份权利金 | 大波动，方向不确定 |
| 买入宽跨式 Long Strangle | 买虚值认购 + 虚值认沽 | 理论无限 | 两份权利金（更便宜） | 更大波动 |
| 卖出跨式 Short Straddle | 卖同一 K 的认购 + 认沽 | 两份权利金 | **理论无限** | 横盘 |
| 卖出宽跨式 Short Strangle | 卖虚值认购 + 虚值认沽 | 两份权利金 | **理论无限** | 区间震荡 |
| 牛市价差 Bull Call Spread | 买低 K 认购 + 卖高 K 认购 | 两 K 之差 − 净权利金 | 净权利金 | 温和上涨 |
| 铁鹰 Iron Condor | 卖宽跨式 + 买更虚值的宽跨式保护 | 净权利金 | 翼宽 − 净权利金（**有限**） | 区间震荡，限定风险 |
| 备兑认购 Covered Call | 持有标的 + 卖虚值认购 | 有限 | 标的下跌 | 增强收益，放弃大涨 |
| 保护性认沽 Protective Put | 持有标的 + 买认沽 | 理论无限 | 有限 | 给持仓买保险 |
| 日历价差 Calendar Spread | 卖近月 + 买远月，同一 K | 有限 | 有限 | 近月 Theta 衰减快 |

```python
import numpy as np

def strategy_payoff(S_T, legs):
    """legs: [(kind, K, position, premium), ...]，position=+1 买入，-1 卖出"""
    total = np.zeros_like(S_T, dtype=float)
    for kind, K, pos, prem in legs:
        intrinsic = np.maximum(S_T - K, 0) if kind == "call" else np.maximum(K - S_T, 0)
        total += pos * (intrinsic - prem)
    return total

S_T = np.array([2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4])
short_strangle = [("put", 2.9, -1, 0.03), ("call", 3.1, -1, 0.03)]
iron_condor = short_strangle + [("put", 2.8, +1, 0.012), ("call", 3.2, +1, 0.012)]
print("卖宽跨:", strategy_payoff(S_T, short_strangle))
print("铁  鹰:", strategy_payoff(S_T, iron_condor))   # 两翼亏损被封顶
```

### 10.3 波动率风险溢价（VRP）

大量实证研究发现，指数期权的隐含波动率**平均而言**高于随后的实现波动率，两者之差称为**波动率风险溢价（Volatility Risk Premium, VRP）**：

$$
\text{VRP} = \sigma_{\text{imp}} - \sigma_{\text{real}} \quad (\text{平均为正，但会在危机中剧烈变负})
$$

**为什么存在**：投资者愿意为"保险"支付溢价，期权卖方承担了崩盘风险，需要补偿。这就是卖方策略长期"看起来赚钱"的原因——但这份收益是**承担尾部风险的报酬**，不是免费午餐。

### 10.4 卖方尾部风险：压路机前捡硬币

```
          卖方策略的典型收益分布（负偏、肥尾）

  频率 ▲
       │                                  ███
       │                                 ████
       │                                █████   ← 绝大多数时间：小幅盈利
       │                               ██████
       │                              ███████
       │  ▪           ▪    ▪▪  ▪  ▪▪▪████████
       └──┼───────────────────────────┼───────────▶ 收益
         ↑                            0
     偶尔一次：巨亏（可抹掉多年收益）
```

**比喻**："在压路机前捡硬币"——大多数时候捡到硬币，偶尔一次被压路机碾过。

**真实案例**（简述，细节请查阅公开报道）：

- **2018 年 2 月 "Volmageddon"**：美股波动率指数 VIX 单日暴涨，做空波动率的交易所交易产品 XIV 净值单日暴跌约 90% 以上，随后被清盘。
- **2018 年 LJM Partners**：一家长期卖出标普 500 期货期权的基金，在 2018 年 2 月的波动中短时间内损失大部分资产。
- **2018 年 11 月 OptionSellers.com**：卖出天然气期权的资管机构，在天然气价格暴涨时客户账户大面积穿仓。
- **国内 ETF 期权**：临近到期的虚值合约在标的大幅波动时，可能出现单日数十倍甚至上百倍的涨幅（市场俗称"末日轮"行情）。对于卖出这些合约的一方，就是"一天亏掉数十倍权利金"。

### 10.5 代码：用带跳跃的模型看卖方收益分布

```python
import numpy as np

def simulate_short_strangle(n_months=600, S0=3.0, sigma=0.15, jump_prob=0.03,
                            jump_size=-0.15, k_put=0.92, k_call=1.08, premium=0.010, seed=0):
    """每月卖一次宽跨式并持有到期；标的月收益 = 正态波动 + 小概率跳跃（暴跌）"""
    rng = np.random.default_rng(seed)
    T = 1 / 12
    ret = rng.normal(-0.5 * sigma**2 * T, sigma * np.sqrt(T), n_months)
    ret += np.where(rng.random(n_months) < jump_prob, jump_size, 0.0)
    S_T = S0 * np.exp(ret)
    loss = np.maximum(k_put * S0 - S_T, 0) + np.maximum(S_T - k_call * S0, 0)
    pnl = 2 * premium - loss                                   # 每月每份的盈亏
    return pnl

premium = 0.010
pnl = simulate_short_strangle(premium=premium)
print(f"胜率={np.mean(pnl > 0):.1%}  月均盈亏={pnl.mean():+.4f}")
print(f"最好一月={pnl.max():+.4f}  最差一月={pnl.min():+.4f}")
print(f"偏度={((pnl - pnl.mean())**3).mean() / pnl.std()**3:.2f}")
print(f"最差一月相当于 {abs(pnl.min()) / (2 * premium):.1f} 个月的权利金收入")
```

在这组参数下，典型输出为：胜率超过 90%、月均盈亏为小幅正值、偏度显著为负，而最差的一个月亏掉了约 28 个月的权利金收入。改变 `premium`、`jump_prob` 会看到月均盈亏的正负对参数极其敏感——**卖方策略的回测结果高度依赖样本期是否包含尾部事件**。

### 10.6 卖方风控清单

```
┌──────────────────────────────────────────────────────────────────┐
│                       期权卖方风控清单                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  □ 优先使用有限风险结构（铁鹰、价差），而不是裸卖                   │
│  □ 保证金占用率上限（如不超过账户权益的 30%–50%，按自身情况设定）    │
│  □ 压力测试：标的 ±10%/±20% 跳空 + IV 翻倍时的组合损益              │
│  □ 限制组合的 Short Gamma 与 Short Vega 总量                       │
│  □ 临近到期的平值附近空头头寸提前处理（Gamma 风险与行权风险）        │
│  □ 回测样本必须包含极端行情（如 2015 年股灾、2020 年初疫情冲击）      │
│  □ 不要因为"连续盈利"而加大仓位——这正是尾部风险累积的过程           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 11. 最佳实践与常见陷阱

### 11.1 最佳实践

| 领域 | 实践 |
| --- | --- |
| 定价代码 | BS、二叉树、蒙特卡洛三种方法互相校验；希腊字母用有限差分校验 |
| 单位 | 统一 $T$（年）、$\sigma$（小数）、Vega（每 1 vol 点）、Theta（每天）的口径，并写在函数文档中 |
| IV 计算 | 用买卖中间价、隐含远期价格、OTM 合约；对无解情况返回 NaN |
| 曲面 | 拟合后检查日历与蝶式无套利 |
| 对冲 | 回测中计入每次对冲的交易成本和滑点；模拟跳空 |
| 风控 | 按希腊字母、情景压力测试、保证金三个维度同时设限 |

### 11.2 常见陷阱

| 陷阱 | 表现 | 解决方案 |
| --- | --- | --- |
| 把期权当便宜彩票 | 大量买入深度虚值期权，持续亏损 | 理解时间价值衰减与行权概率 |
| 卖方回测样本期太短 | 回测几年都是平静市，夏普极高 | 必须覆盖危机期，做尾部压力测试 |
| 合约单位假设为常数 | 分红调整后的合约计算错误 | 使用合约列表中的实际合约单位 |
| 用成交价算 IV | 不活跃合约价格陈旧，IV 异常 | 用中间价，过滤价差过宽的合约 |
| 混淆 Delta 与概率 | 用 Delta 估计行权概率 | 风险中性行权概率是 $N(d_2)$，且风险中性概率 ≠ 真实概率 |
| 忽视到期日行权风险 | 到期日平值附近不确定是否被行权 | 到期前主动平仓或展期 |
| Vega 单一汇总 | 不同期限的 Vega 简单相加 | 按期限分桶管理 Vega |
| 忽视保证金动态 | 行情不利时保证金飙升，被迫在最差价位平仓 | 预留充足保证金，压力测试保证金需求 |

---

## 12. 总结

### 12.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                          核心概念回顾                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  期权基础                                                         │
│  ├── Call / Put，买方权利、卖方义务，收益非对称                     │
│  ├── 价格 = 内在价值 + 时间价值                                    │
│  └── 平价关系：C − P = S − K·e^(−rT)                              │
│                                                                  │
│  定价                                                             │
│  ├── 复制与无套利 → 风险中性定价                                   │
│  ├── 二叉树（CRR）：可处理美式期权                                 │
│  └── BS：C = S·N(d1) − K·e^(−rT)·N(d2)                           │
│                                                                  │
│  希腊字母                                                         │
│  ├── Delta（方向）Gamma（凸性）Vega（波动率）Theta（时间）           │
│  └── 组合希腊字母可加，按乘数换算                                  │
│                                                                  │
│  波动率                                                           │
│  ├── IV：牛顿法 σ ← σ − (BS(σ) − C)/Vega，二分法兜底               │
│  ├── 微笑/偏斜：肥尾、崩盘恐惧、供求                               │
│  └── 曲面：SVI 参数化 + 无套利约束                                 │
│                                                                  │
│  交易                                                             │
│  ├── Delta 对冲后 PnL ≈ ½·Γ·S²·(σ_real² − σ_imp²)·Δt              │
│  ├── Long Gamma 付 Theta，Short Gamma 收 Theta                    │
│  └── 卖方：赚 VRP，承担尾部风险，必须有限风险结构 + 压力测试          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 学习检查对照

| 学习检查项 | 对应章节 |
| --- | --- |
| 能推导并实现 BS 公式 | 第 4.3、4.4 节 |
| 能用牛顿法求隐含波动率 | 第 6.2、6.3 节 |
| 理解 Gamma 与 Theta 的权衡 | 第 9.2 节 |
| 知道卖期权策略的尾部风险 | 第 10.4–10.6 节 |

### 12.3 与下一阶段的衔接

```
Phase 12 期权与衍生品                   Phase 13 交易系统工程与实盘
──────────────────────                  ──────────────────────────
Delta 对冲需要实时调仓      ───────▶     事件驱动引擎 + OMS 自动下单
希腊字母与保证金限额        ───────▶     事前 / 事中风控引擎
期权链数据拉取与清洗        ───────▶     行情网关与统一数据模型
卖方尾部风险               ───────▶     异常时自动暂停交易、熔断机制
```

至此，课程的"策略与定价"知识已经完整。Phase 13 将把这些策略放进一个真正可以连接交易所、7×24 稳定运行的交易系统中。

**推荐阅读**：

- John Hull《期权、期货及其他衍生产品》：衍生品的标准教材
- Sheldon Natenberg《Option Volatility and Pricing》：交易员视角的波动率与策略
- Jim Gatheral《The Volatility Surface》：波动率曲面与 SVI
- 上交所、深交所期权业务规则与投资者教育专栏
