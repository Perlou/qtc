# Phase 3: 量化数学与统计 - 核心概念深度解析

> 本文是 Phase 3 的配套概念文档。建议先通读一遍建立直觉，再逐个运行 `01-*.py` ~ `11-*.py`。
> 读者假设：编程能力强，但概率统计已经"还给老师"。每个概念按「直觉/比喻 → 公式 → 代码 → 量化中的意义与陷阱」展开。
> 本文不追求数学严谨的证明，而是让你**知道每个工具在回答什么问题、什么时候会失效**。

---

## 目录

1. [引言：量化为什么离不开统计](#1-引言量化为什么离不开统计)
2. [概率分布与收益率分布](#2-概率分布与收益率分布)
3. [肥尾、偏度、峰度与正态性检验](#3-肥尾偏度峰度与正态性检验)
4. [假设检验：策略收益是否显著](#4-假设检验策略收益是否显著)
5. [蒙特卡洛模拟与 Bootstrap](#5-蒙特卡洛模拟与-bootstrap)
6. [期望值、胜率、盈亏比与凯利公式](#6-期望值胜率盈亏比与凯利公式)
7. [OLS 回归与残差诊断](#7-ols-回归与残差诊断)
8. [相关性与协整](#8-相关性与协整)
9. [协方差矩阵、特征分解与 PCA](#9-协方差矩阵特征分解与-pca)
10. [平稳性与 ADF 检验](#10-平稳性与-adf-检验)
11. [自相关与 ARMA 模型](#11-自相关与-arma-模型)
12. [波动率聚集与 GARCH](#12-波动率聚集与-garch)
13. [最佳实践与常见陷阱](#13-最佳实践与常见陷阱)
14. [总结](#14-总结)

---

## 1. 引言：量化为什么离不开统计

### 1.1 金融数据的本质：极低信噪比

在工程领域，你测一个接口的延迟，100 次测量就能得到很稳定的平均值。金融数据完全不同：

```
茅台日收益 = 很小的"可预测部分"（信号） + 巨大的随机波动（噪声）

  信号：可能每天只有 0.0x% 量级
  噪声：日波动通常在 1%–3% 量级
```

**信噪比极低**意味着：

- 你在历史数据中看到的"规律"，大部分可能是噪声
- 需要**统计工具**来区分"真规律"和"巧合"
- 需要**概率思维**：没有确定的预测，只有概率上的优势

### 1.2 本阶段的工具箱

| 问题 | 工具 | 章节 |
| --- | --- | --- |
| 收益率长什么样？极端行情有多常见？ | 分布、偏度、峰度、正态性检验 | 2、3 |
| 这个策略是真的赚钱还是运气？ | 假设检验、t 检验 | 4 |
| 未来可能有哪些情景？回撤可能多深？ | 蒙特卡洛、Bootstrap | 5 |
| 每次该下多大注？ | 期望值、凯利公式 | 6 |
| A 和 B 是什么关系？ | OLS 回归、相关、协整 | 7、8 |
| 几百只股票的共同驱动因素是什么？ | 协方差矩阵、PCA | 9 |
| 这个序列能不能直接建模？ | 平稳性、ADF | 10 |
| 过去能预测未来吗？ | 自相关、ARMA | 11 |
| 明天波动会有多大？ | GARCH | 12 |

### 1.3 推荐依赖

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from arch import arch_model   # pip install arch
```

---

## 2. 概率分布与收益率分布

### 2.1 直觉：分布是"所有可能结果及其可能性"的清单

**比喻**：把茅台过去 10 年每天的收益率写在纸条上扔进一个桶里。**分布**就是对这个桶的描述——大部分纸条写着 -1% 到 +1%，极少数写着 ±9%。量化建模的很多工作，就是在猜"明天会从什么样的桶里抽纸条"。

### 2.2 量化中常见的分布

| 分布 | 形状 | 量化中的用途 |
| --- | --- | --- |
| **正态分布** $N(\mu, \sigma^2)$ | 对称钟形，尾部很薄 | 最常用的基准假设；很多经典模型的前提 |
| **对数正态分布** | 右偏，取值 > 0 | 价格建模：若对数收益正态，则价格对数正态（不会跌到负数） |
| **Student-t 分布** | 钟形，尾部更厚 | 更贴近真实收益率的肥尾；GARCH 残差 |
| **二项分布** | 离散 | $n$ 笔交易中盈利笔数 |
| **泊松分布** | 离散 | 单位时间内事件（如跳空、订单到达）次数 |
| **均匀分布** | 平的 | 随机数生成、随机策略基准 |

### 2.3 正态分布：所有模型的起点

$$
f(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- $\mu$：均值（分布中心）
- $\sigma$：标准差（分布宽度）
- $f(x)$：概率密度，曲线下面积为概率

**68-95-99.7 法则**：约 68% 的值落在 $\mu \pm \sigma$，95% 在 $\pm 2\sigma$，99.7% 在 $\pm 3\sigma$。

### 2.4 从对数收益正态到价格对数正态

如果每日对数收益 $r_t \sim N(\mu, \sigma^2)$ 且相互独立，那么 $T$ 日后：

$$
\ln\frac{P_T}{P_0} = \sum_{t=1}^{T} r_t \sim N(T\mu,\ T\sigma^2)
$$

- $P_0, P_T$：期初、期末价格

即 $P_T$ 服从**对数正态分布**。这是第 5 节几何布朗运动（GBM）模拟和 Phase 12 Black-Scholes 期权定价的基础假设。

### 2.5 代码：拟合并对比

```python
ret = np.log(close).diff().dropna()     # 茅台日对数收益

# 拟合正态分布
mu, sigma = stats.norm.fit(ret)

# 拟合 Student-t 分布：df 越小尾部越厚，df→∞ 退化为正态
df_t, loc_t, scale_t = stats.t.fit(ret)
print(f"t 分布自由度 ≈ {df_t:.1f}")   # 日收益常见 3~6 左右，具体以你的数据为准

# 比较极端事件概率：单日跌幅超过 5 个标准差
x = mu - 5 * sigma
p_norm = stats.norm.cdf(x, mu, sigma)
p_t = stats.t.cdf(x, df_t, loc_t, scale_t)
p_emp = (ret < x).mean()
print(f"正态: {p_norm:.2e}, t 分布: {p_t:.2e}, 经验频率: {p_emp:.2e}")
```

你通常会看到：正态分布给出的概率小到"宇宙年龄内都不该发生"，而经验频率显示它在十年里发生过好几次。

### 2.6 量化意义

- **分布假设决定风险估计**。用正态分布估计 VaR（风险价值），会系统性低估极端亏损。
- **收益率的期望（$\mu$）极难估计，波动率（$\sigma$）相对容易**。这是一个非常重要的结论，第 4 节会定量说明：估计均值所需的样本长度远超估计波动率。所以量化实践中，风险模型往往比收益预测更可靠。

---

## 3. 肥尾、偏度、峰度与正态性检验

### 3.1 直觉

**肥尾（Fat Tails）**：极端事件（暴涨暴跌）出现的频率远高于正态分布的预测。

**比喻**：正态分布假设世界像"人的身高"——最高的人也不过平均身高的 1.5 倍左右；而金融市场更像"人的财富"——偶尔会冒出一个比平均值大几千倍的个体。

### 3.2 四阶矩

| 矩 | 名称 | 描述 | 正态分布取值 |
| --- | --- | --- | --- |
| 一阶 | 均值 $\mu$ | 中心位置 | — |
| 二阶 | 方差 $\sigma^2$ | 离散程度 | — |
| 三阶 | **偏度（Skewness）** | 对称性 | 0 |
| 四阶 | **峰度（Kurtosis）** | 尾部厚度 | 3（超额峰度为 0） |

$$
\text{Skew} = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right], \qquad
\text{Kurt} = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]
$$

- $\mathbb{E}[\cdot]$：期望（平均）
- **超额峰度（Excess Kurtosis）** = Kurt − 3，正态分布为 0；> 0 即肥尾

**偏度的直觉**：

```
负偏（Skew < 0）               正偏（Skew > 0）
"经常小赚，偶尔大亏"          "经常小亏，偶尔大赚"
     ┃                              ┃
    ┃┃┃                            ┃┃┃
   ┃┃┃┃                            ┃┃┃┃
 ┃┃┃┃┃┃                          ┃┃┃┃┃┃┃
━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━
←长尾                                     长尾→
典型：卖期权、网格交易           典型：趋势跟踪、买期权
```

> ⚠️ 注意库的约定：`scipy.stats.kurtosis` 默认 `fisher=True` 返回**超额峰度**；pandas 的 `.kurt()` 也返回超额峰度。不要和"正态 = 3"的定义混用。

### 3.3 正态性检验

| 检验 | 原理 | 原假设 $H_0$ | 注意 |
| --- | --- | --- | --- |
| **Jarque-Bera** | 偏度和峰度是否偏离 0 和 3 | 数据服从正态 | 大样本下非常灵敏 |
| **Shapiro-Wilk** | 排序后与正态期望值的相关 | 数据服从正态 | scipy 对 n > 5000 会警告 p 值可能不准 |
| **Kolmogorov-Smirnov** | 经验 CDF 与理论 CDF 的最大距离 | 数据服从指定分布 | 若参数由同一数据估计，p 值偏大，应用 Lilliefors 修正 |
| **Q-Q 图** | 分位数对分位数作图 | — | 不是检验，但最直观 |

Jarque-Bera 统计量：

$$
\text{JB} = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)
$$

- $n$：样本量；$S$：样本偏度；$K$：样本峰度（非超额）
- 在 $H_0$ 下近似服从自由度为 2 的卡方分布

### 3.4 代码

```python
print("偏度:", stats.skew(ret))
print("超额峰度:", stats.kurtosis(ret))          # fisher=True 默认

jb_stat, jb_p = stats.jarque_bera(ret)
print(f"JB={jb_stat:.1f}, p={jb_p:.2e}")          # p 极小 → 拒绝正态

# Q-Q 图：点偏离直线的"S 形"两端 = 肥尾
import matplotlib.pyplot as plt
sm.qqplot(ret, line="s")
plt.show()
```

### 3.5 尾部风险度量：VaR 与 CVaR

**VaR（Value at Risk，风险价值）**：在给定置信水平 $c$（如 95%）下，单日亏损**不超过**多少。

**CVaR / ES（Conditional VaR / Expected Shortfall，条件风险价值）**：一旦亏损超过 VaR，**平均**亏多少。

$$
\text{VaR}_c = -q_{1-c}(R), \qquad \text{CVaR}_c = -\mathbb{E}\left[R \mid R \le q_{1-c}(R)\right]
$$

- $q_{1-c}(R)$：收益分布的 $1-c$ 分位数（如 5% 分位数）
- 取负号是为了把亏损表示为正数

```python
c = 0.95
var_hist = -np.quantile(ret, 1 - c)                     # 历史模拟法
cvar_hist = -ret[ret <= -var_hist].mean()
var_norm = -(ret.mean() + stats.norm.ppf(1 - c) * ret.std())   # 参数法（正态假设）
print(f"历史 VaR {var_hist:.2%}, 正态 VaR {var_norm:.2%}, CVaR {cvar_hist:.2%}")
```

### 3.6 量化意义与陷阱

- **几乎所有资产的日收益都"拒绝正态"**。这不是要你放弃正态分布，而是要知道它在尾部会失效。
- **负偏策略是"隐形炸弹"**：卖出虚值期权、高频网格、无止损的均值回归，平时收益曲线非常平滑（Sharpe 很高），然后某一天亏掉多年利润。Sharpe Ratio 只看均值和方差，**看不到偏度**——这是 Phase 4 要补充 Sortino、Calmar、尾部指标的原因。
- **VaR 不是最坏情况**。VaR 只告诉你"95% 的日子亏不超过 X"，对剩下 5% 的日子会亏多少一无所知。CVaR 更能反映尾部。
- **A 股的涨跌停会"截断"日收益分布**：单日收益被限制在一定范围内（主板一般 ±10%），但风险并没有消失，而是转移成连续多日跌停、无法卖出的**流动性风险**。只看日收益分布会低估这种风险。

---

## 4. 假设检验：策略收益是否显著

### 4.1 直觉：统计学里的"无罪推定"

**比喻**：法庭审判。原假设 $H_0$ 是"被告无罪"（策略没有真实收益，平均收益为 0）。我们收集证据（回测数据），只有当证据**强到不太可能在"无罪"情况下出现**时，才判"有罪"（拒绝 $H_0$，认为策略有效）。

- **p 值**：假设 $H_0$ 为真（策略其实没本事），仅凭运气得到"至少这么好"的结果的概率
- p 值小（如 < 0.05）→ "运气这么好的可能性很小" → 拒绝 $H_0$

> ⚠️ p 值**不是**"策略无效的概率"，也**不是**"策略有效的概率"。它是在"策略无效"这个前提下，观测数据有多极端。

### 4.2 两类错误

| | 真实：策略无效 | 真实：策略有效 |
| --- | --- | --- |
| **判断：有效**（拒绝 $H_0$） | ❌ 第一类错误（假阳性），概率 $\alpha$ | ✅ |
| **判断：无效**（不拒绝 $H_0$） | ✅ | ❌ 第二类错误（假阴性），概率 $\beta$ |

在量化中，**第一类错误代价更大**：把一个靠运气的策略当成真策略上实盘，会真金白银地亏钱。

### 4.3 单样本 t 检验

检验策略日收益的均值是否显著大于 0：

$$
t = \frac{\bar{R}}{s / \sqrt{n}}
$$

- $\bar{R}$：样本平均日收益
- $s$：样本标准差
- $n$：样本天数
- $s/\sqrt{n}$：均值的**标准误（Standard Error）**——均值估计本身的不确定性

### 4.4 t 统计量与 Sharpe Ratio 的关系

把上式变形：

$$
t = \frac{\bar{R}}{s}\sqrt{n} = \text{SR}_{\text{daily}}\sqrt{n} = \text{SR}_{\text{annual}}\sqrt{\frac{n}{N}} = \text{SR}_{\text{annual}}\sqrt{Y}
$$

- $\text{SR}_{\text{daily}}$：日频 Sharpe（未年化，此处忽略无风险利率）
- $N$：每年交易日数；$Y = n/N$：样本年数

这个公式非常有用。要让 $t > 2$（约 5% 显著性）：

| 年化 Sharpe | 需要的样本年数 $Y \approx (2/\text{SR})^2$ |
| --- | --- |
| 0.5 | 16 年 |
| 1.0 | 4 年 |
| 1.5 | 约 1.8 年 |
| 2.0 | 1 年 |

**结论**：一个年化 Sharpe 0.5 的策略（已经不算差），需要十几年数据才能在统计上与"运气"区分开。这就是为什么**收益很难被证明，而波动率很容易被估计**。

### 4.5 代码

```python
# strat_ret: 策略日收益（已扣成本）
t_stat, p_two = stats.ttest_1samp(strat_ret.dropna(), popmean=0)
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2      # 单侧：均值 > 0
print(f"t={t_stat:.2f}, 单侧 p={p_one:.4f}")

# 更稳健：考虑自相关和异方差的 Newey-West 标准误
y = strat_ret.dropna().to_numpy()
res = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})   # 只对常数项回归 = 检验均值
print(f"Newey-West t={res.tvalues[0]:.2f}")
```

### 4.6 量化中的陷阱

1. **收益不独立**：t 检验假设各期独立。策略收益往往有自相关（持仓多日、波动聚集），普通 t 检验会高估显著性。用 **Newey-West（HAC）标准误** 修正。
2. **重叠样本**：用"未来 20 日收益"做研究时，相邻样本共享 19 天数据，有效样本量远小于行数。
3. **多重检验（Multiple Testing）**：这是最致命的问题。如果你试了 100 个策略，即使全都无效，在 5% 显著性下也平均会有约 5 个"显著"。Phase 4 会系统讨论 Bonferroni、BH-FDR、Deflated Sharpe Ratio 等修正方法。学术界也有观点主张新因子的 t 值门槛应提高到 3 左右（Harvey, Liu & Zhu, 2016）。
4. **统计显著 ≠ 经济显著**：日均 0.01% 的超额收益在大样本下可能"显著"，但扣掉交易成本后可能为负。
5. **p-hacking**：反复修改样本区间、剔除"异常"时段直到 p < 0.05。所有尝试都应记录在研究日志中（见课程通用验收标准）。

---

## 5. 蒙特卡洛模拟与 Bootstrap

### 5.1 直觉

历史只发生了一次。你的回测净值曲线只是"无数种可能历史"中的一条。

- **蒙特卡洛（Monte Carlo）**：假设一个模型（如 GBM），用随机数生成成千上万条"平行宇宙"路径
- **Bootstrap（自助法）**：不假设模型，从历史数据中**有放回地重抽样**，构造新的"可能历史"

**比喻**：蒙特卡洛是"按照菜谱做 10000 次菜"，菜谱（模型）错了，结果全错；Bootstrap 是"把已经做好的菜打散重新装盘"，不需要菜谱，但只能用已有的食材（历史上没出现过的极端情况它也模拟不出来）。

### 5.2 几何布朗运动（GBM）

$$
S_{t+\Delta t} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right], \quad Z \sim N(0,1)
$$

- $S_t$：$t$ 时刻价格
- $\mu$：年化漂移（期望收益率）；$\sigma$：年化波动率
- $\Delta t$：时间步长（日频取 $1/252$）
- $Z$：标准正态随机数
- $-\sigma^2/2$：伊藤修正项，正是 Phase 2 提到的"波动拖累"

```python
def simulate_gbm(s0, mu, sigma, days=252, n_paths=10_000, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal((days, n_paths))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.cumsum(log_ret, axis=0))      # shape: (days, n_paths)

paths = simulate_gbm(s0=1700, mu=0.08, sigma=0.30)
final = paths[-1]
print("一年后价格 5%/50%/95% 分位:", np.percentile(final, [5, 50, 95]).round(0))
```

### 5.3 Bootstrap：估计 Sharpe Ratio 的置信区间

```python
def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)

def bootstrap_sharpe(r: np.ndarray, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r)
    sims = np.array([sharpe(r[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return np.percentile(sims, [2.5, 97.5])

lo, hi = bootstrap_sharpe(strat_ret.dropna().to_numpy())
print(f"Sharpe 95% 置信区间: [{lo:.2f}, {hi:.2f}]")   # 若包含 0，说明证据不足
```

### 5.4 Block Bootstrap：保留时间结构

普通 Bootstrap 逐日独立抽样，**破坏了自相关和波动聚集**（"连续暴跌"这种结构消失了），会低估回撤。**块自助法（Block Bootstrap）** 一次抽取连续的一段：

```
原始序列: [d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 ...]
普通:      d7 d2 d9 d2 d5 d10 ...           ← 时间结构被打散
块(长度3): [d4 d5 d6][d1 d2 d3][d7 d8 d9]  ← 块内保留连续性
```

```python
def block_bootstrap(r: np.ndarray, block=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r)
    starts = rng.integers(0, n - block, size=n // block + 1)
    return np.concatenate([r[s:s + block] for s in starts])[:n]

# 模拟 2000 条路径的最大回撤分布
mdds = []
for i in range(2000):
    sim = block_bootstrap(strat_ret.dropna().to_numpy(), block=20, seed=i)
    nav = np.cumprod(1 + sim)
    mdds.append((nav / np.maximum.accumulate(nav) - 1).min())
print("MDD 中位数 / 5% 最坏:", np.median(mdds), np.percentile(mdds, 5))
```

> `arch` 库提供了更完善的实现：`arch.bootstrap.StationaryBootstrap`、`CircularBlockBootstrap`。

### 5.5 量化意义与陷阱

- **回测的 MDD 只是一个样本点**。Bootstrap 给出 MDD 的分布，实盘资金规划应参考分布的尾部（如 5% 最坏情形），而不是回测中的那一个数。
- **GBM 的假设（正态、常波动、独立）在真实市场中都不成立**。用 GBM 做压力测试会低估尾部；可以换成 t 分布扰动或 GARCH 模拟（第 12 节）。
- **Bootstrap 只能"重组"历史**，模拟不出历史上从未出现过的情景（例如一个从未经历过熔断的样本）。压力测试需要补充人为设定的极端情景。
- **随机种子**：结果要可复现，固定 `seed` 并在报告中注明模拟次数。

---

## 6. 期望值、胜率、盈亏比与凯利公式

### 6.1 直觉：高胜率 ≠ 赚钱

一个策略胜率 90%，听起来很厉害？如果每次赚 1 元、每次亏 20 元：

$$
\mathbb{E} = 0.9 \times 1 - 0.1 \times 20 = -1.1 \text{ 元/笔}
$$

长期必亏。反过来，趋势跟踪策略胜率常常只有 30%–40%，但靠少数大赚覆盖多数小亏，长期为正。

### 6.2 期望值公式

$$
\mathbb{E}[\text{每笔盈亏}] = p \cdot W - q \cdot L = L\,(p\,b - q)
$$

- $p$：胜率；$q = 1 - p$：败率
- $W$：平均盈利；$L$：平均亏损（取正数）
- $b = W / L$：**盈亏比（Payoff Ratio）**

**盈亏平衡胜率**：令期望为 0，得 $p^* = \dfrac{1}{1+b}$

| 盈亏比 $b$ | 盈亏平衡胜率 |
| --- | --- |
| 0.25 | 80% |
| 0.5 | 66.7% |
| 1 | 50% |
| 2 | 33.3% |
| 3 | 25% |

**胜率和盈亏比必须一起看**，单独报告任何一个都没有意义。

### 6.3 凯利公式：每次该下多大注

**问题**：你有一个正期望的机会，每次应该押上本金的多大比例 $f$，才能让**长期复利增长最快**？

- 押太少：增长慢
- 押太多：一次大亏就伤筋动骨，甚至破产（还记得"亏 50% 要涨 100% 才回本"吗）

**每次下注的对数增长率**：

$$
g(f) = p\ln(1 + f b) + q\ln(1 - f)
$$

- $f$：每次下注占当前本金的比例
- 赢了本金变为 $(1+fb)$ 倍，输了变为 $(1-f)$ 倍

对 $f$ 求导令其为 0，得**凯利比例**：

$$
f^* = p - \frac{q}{b} = \frac{p\,b - q}{b}
$$

**连续版本**（适用于收益连续分布的策略/资产）：

$$
f^* = \frac{\mu - r_f}{\sigma^2}
$$

- $\mu$：策略期望收益；$r_f$：无风险利率；$\sigma^2$：收益方差

### 6.4 代码：凯利增长曲线

```python
p, b = 0.55, 1.0
f_star = p - (1 - p) / b
print(f"凯利比例 f* = {f_star:.2%}")            # 10%

f = np.linspace(0.001, 0.4, 400)
g = p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)
# g 在 f* 处最大；f ≈ 2f* 时增长率回到约 0；继续加大则为负 —— 正期望也会被下注过大"玩死"

# 模拟验证
rng = np.random.default_rng(1)
wins = rng.random(1000) < p
for frac in [0.5 * f_star, f_star, 2 * f_star, 3 * f_star]:
    wealth = np.prod(np.where(wins, 1 + frac * b, 1 - frac))
    print(f"f={frac:.2%}: 1000 次后财富 = {wealth:.2f}")
```

```
增长率 g(f)
  ▲
  │          ● f* (最优)
  │       ╱     ╲
  │     ╱         ╲
  │   ╱             ╲
 0┼──────────────────●─────────▶ f
  │                 ≈2f*  ╲
  │                         ╲ 负增长区：过度下注
```

### 6.5 量化意义与陷阱

- **半凯利（Half Kelly）是实践中的常见选择**：增长率仍有最优值的约 75%，但波动和回撤大幅下降。原因是 $p$、$b$、$\mu$、$\sigma$ 都是**估计值**，估计误差会让"全凯利"实际上变成"过度下注"。
- **高估胜率的代价不对称**：在 $f^*$ 左侧，少押一点损失很小；在右侧，多押一点损失迅速增加。宁可低估。
- **连续凯利 $\mu/\sigma^2$ 往往给出很高的杠杆**（例如年化超额 10%、波动 20% 时 $f^* = 2.5$），这正说明了全凯利的激进。
- **凯利假设每次下注独立、分布已知不变**。真实市场中相关性、状态切换都会破坏这一假设。在 Phase 5 和 Phase 8，你会更多使用"波动率目标"和风险预算来决定仓位。

---

## 7. OLS 回归与残差诊断

### 7.1 直觉

**OLS（Ordinary Least Squares，普通最小二乘）**：在散点图里画一条直线，使所有点到直线的**竖直距离平方和**最小。

在量化中最常见的用途：

- 个股对市场回归 → Beta、Alpha（Phase 2 已见）
- 收益对多个因子回归 → 因子暴露（Phase 7）
- 一只股票对另一只回归 → 配对交易的对冲比例（Phase 6）

### 7.2 公式

单变量：

$$
y_t = \alpha + \beta x_t + \varepsilon_t
$$

多变量（矩阵形式）：

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

- $\mathbf{y}$：$n \times 1$ 因变量向量（如茅台日收益）
- $\mathbf{X}$：$n \times k$ 自变量矩阵（第一列全为 1 对应截距）
- $\hat{\boldsymbol{\beta}}$：系数估计
- $\boldsymbol{\varepsilon}$：残差

**$R^2$（拟合优度）**：

$$
R^2 = 1 - \frac{\sum_t \hat{\varepsilon}_t^2}{\sum_t (y_t - \bar{y})^2}
$$

自变量能"解释"的因变量方差比例。

### 7.3 代码：茅台对沪深 300 和白酒板块回归

```python
data = pd.concat([maotai_ret, hs300_ret, baijiu_ret], axis=1,
                 keys=["y", "mkt", "sector"]).dropna()
X = sm.add_constant(data[["mkt", "sector"]])      # 别忘了加常数项！
model = sm.OLS(data["y"], X).fit()
print(model.summary())
```

`summary()` 中你需要看懂的部分：

| 输出项 | 含义 | 关注点 |
| --- | --- | --- |
| `coef` | 系数估计 | 方向和大小是否合理 |
| `std err` | 系数标准误 | 估计的不确定性 |
| `t`, `P>|t|` | 系数是否显著不为 0 | 注意第 4 节的多重检验 |
| `R-squared` | 解释的方差比例 | 收益回归中低 R² 是常态 |
| `Durbin-Watson` | 残差一阶自相关 | 接近 2 表示无自相关 |
| `Jarque-Bera` | 残差正态性 | 金融数据几乎总是拒绝 |
| `Cond. No.` | 条件数 | 很大则可能存在多重共线性 |

### 7.4 OLS 的假设与残差诊断

OLS 的系数估计和 t 检验依赖若干假设。**残差诊断**就是检查这些假设有没有被严重违反：

| 假设 | 违反时的后果 | 诊断 | 补救 |
| --- | --- | --- | --- |
| 残差无自相关 | 标准误偏小，t 值虚高 | Durbin-Watson、Ljung-Box | Newey-West (HAC) 标准误 |
| 残差同方差 | 标准误不可靠 | Breusch-Pagan、残差图 | 稳健标准误（HC/HAC） |
| 自变量不高度共线 | 系数不稳定、符号乱跳 | 相关矩阵、VIF、条件数 | 删变量、正交化、正则化 |
| 残差近似正态 | 小样本推断不准 | JB、Q-Q 图 | 大样本下影响较小 |
| 变量平稳 | **伪回归** | ADF（第 10 节） | 用收益而非价格回归 |

```python
resid = model.resid

# 自相关
print(acorr_ljungbox(resid, lags=[5, 10]))

# 异方差：Breusch-Pagan 检验
lm_stat, lm_p, _, _ = het_breuschpagan(resid, X)
print(f"BP 检验 p={lm_p:.4f}")      # p 小 → 存在异方差

# 稳健标准误重新估计（系数不变，标准误和 t 值变化）
robust = sm.OLS(data["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(robust.summary().tables[1])
```

### 7.5 伪回归（Spurious Regression）：最危险的陷阱

两只**毫无关系**的随机游走序列，直接用价格做回归，经常得到 $R^2$ 很高、t 值很显著的结果。

```python
rng = np.random.default_rng(0)
a = np.cumsum(rng.standard_normal(1000))   # 随机游走 A
b = np.cumsum(rng.standard_normal(1000))   # 独立的随机游走 B
fake = sm.OLS(a, sm.add_constant(b)).fit()
print(f"R²={fake.rsquared:.2f}, t={fake.tvalues[1]:.1f}")   # 常常"高度显著"
```

**原因**：两个非平稳序列都有"趋势"，回归把共同的漂移误当成关系。

**规则**：**用收益率（平稳）做回归**，不要用价格（非平稳）。唯一的例外是**协整**（下一节），但那需要专门检验。

---

## 8. 相关性与协整

### 8.1 相关性：同涨同跌的程度

**Pearson 相关系数**：

$$
\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X\,\sigma_Y} \in [-1, 1]
$$

- $\text{Cov}(X,Y)$：协方差；$\sigma_X, \sigma_Y$：标准差
- 只衡量**线性**关系

**Spearman 秩相关**：先把数据转为排名再算 Pearson，对异常值稳健，能捕捉单调非线性关系。在 Phase 7 因子研究中，**Rank IC** 就是因子值与下期收益的 Spearman 相关。

```python
corr_pearson = maotai_ret.corr(wly_ret)                       # 茅台 vs 五粮液
corr_spearman = maotai_ret.corr(wly_ret, method="spearman")
roll_corr = maotai_ret.rolling(60).corr(wly_ret)              # 相关性随时间变化
```

### 8.2 协整：长期"拴在一起"

**比喻：醉汉和他的狗**。醉汉在街上随机乱走（随机游走），狗也随机乱跑。

- **相关性**描述的是：每一步醉汉往左时，狗是否也往左（短期同步程度）
- **协整**描述的是：狗被**绳子**拴着，不管两者怎么乱走，它们之间的距离不会无限扩大（长期均衡关系）

```
价格
 ▲        醉汉 ────╲    ╱──╲
 │    ╱╲  ╱        ╲╱      ╲    ╱
 │   ╱  ╲╱    狗 ·····  ···  ╲╱
 │  ╱ ·····  ···     ···  ···
 │ ╱··     ··           ··
 └──────────────────────────────▶ 时间
 两者都"漂"，但距离（价差）始终在一定范围内来回 → 协整
```

**关键区别**：

| | 相关 | 协整 |
| --- | --- | --- |
| 作用对象 | 收益率（变化） | 价格水平 |
| 时间尺度 | 短期同步 | 长期均衡 |
| 高相关 → ? | 不一定协整 | |
| 协整 → ? | | 短期相关可以不高 |
| 交易含义 | 分散化、对冲 | **配对交易 / 统计套利**（价差偏离后会回归） |

**两个反例帮你记住区别**：

1. 两只股票每天收益高度相关，但其中一只每年系统性多涨一点 → 价差持续扩大 → **相关但不协整**，做价差回归会亏钱
2. 两个价格每天的涨跌关系很弱，但价差长期围绕某个值波动 → **协整但相关性不高**

### 8.3 协整的定义与 Engle-Granger 两步法

如果 $X_t$ 和 $Y_t$ 都是非平稳的 I(1) 序列（差分一次才平稳，例如价格），但存在 $\beta$ 使得

$$
s_t = Y_t - \beta X_t
$$

是**平稳**的，就称 $X$ 与 $Y$ **协整**，$\beta$ 是对冲比例，$s_t$ 是价差（Spread）。

**Engle-Granger 两步法**：

1. 用 OLS 回归 $Y_t = \alpha + \beta X_t + \varepsilon_t$ 估计 $\beta$
2. 对残差 $\hat{\varepsilon}_t$ 做单位根检验（类似 ADF，但临界值不同）

```python
# 例：两只同行业银行股的对数价格（示意）
y, x = np.log(price_a), np.log(price_b)

# statsmodels 的 coint 封装了两步法，并使用正确的临界值
t_stat, p_value, crit = coint(y, x)
print(f"协整检验 p={p_value:.4f}")    # p < 0.05 → 拒绝"不协整"

# 估计对冲比例与价差
ols = sm.OLS(y, sm.add_constant(x)).fit()
beta = ols.params.iloc[1]
spread = y - beta * x
zscore = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
```

> ⚠️ 不要在第一步残差上直接用 `adfuller` 的 p 值判断——因为 $\beta$ 是估计出来的，临界值与普通 ADF 不同。直接用 `coint`。

### 8.4 均值回归的半衰期

价差偏离后，平均多久回到一半？对价差做 AR(1) 型回归：

$$
\Delta s_t = \lambda\, s_{t-1} + c + \varepsilon_t, \qquad \text{半衰期} \approx -\frac{\ln 2}{\lambda}
$$

- $\Delta s_t = s_t - s_{t-1}$：价差的变化
- $\lambda < 0$ 表示均值回归，越负回归越快

```python
ds = spread.diff().dropna()
lag = spread.shift(1).dropna().loc[ds.index]
lam = sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1]
print(f"半衰期 ≈ {-np.log(2) / lam:.1f} 天")
```

半衰期太长（几个月）意味着资金占用久、风险暴露长；太短（不到 1 天）可能是数据噪声或难以交易。

### 8.5 陷阱

- **危机时相关性趋于 1**：平时分散化良好的组合，在股灾中可能同时下跌。用全样本相关系数评估分散化效果会过于乐观。
- **协整关系会断裂**：公司基本面变化（并购、主营业务调整、政策冲击）会让历史上的协整失效。Phase 6 会讨论如何监控和止损。
- **在大量股票对中筛选协整对是多重检验**：3000 只股票有约 450 万个配对，5% 显著性下会"找到"大量假协整对。
- **对冲比例 $\beta$ 用全样本估计 = 前视偏差**。回测时要用滚动或扩展窗口估计。

---

## 9. 协方差矩阵、特征分解与 PCA

### 9.1 为什么需要协方差矩阵

组合的风险不仅取决于每只股票的波动，还取决于它们**如何一起运动**。

$$
\sigma_p^2 = \mathbf{w}^\top \boldsymbol{\Sigma}\, \mathbf{w}
$$

- $\mathbf{w}$：$N \times 1$ 权重向量
- $\boldsymbol{\Sigma}$：$N \times N$ 协方差矩阵，对角线是各资产方差 $\sigma_i^2$，非对角线是协方差 $\sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$
- $\sigma_p^2$：组合方差

**两资产时的直觉**：

$$
\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\rho_{12}\sigma_1\sigma_2
$$

相关系数 $\rho_{12}$ 越低，组合波动越小——这就是"分散化"的数学本质，也是 Phase 8 组合优化的起点。

```python
rets = wide_close.pct_change().dropna()       # 宽表：日期 × 股票
cov = rets.cov() * 252                          # 年化协方差矩阵
w = np.ones(len(cov)) / len(cov)                # 等权
port_vol = np.sqrt(w @ cov.values @ w)
```

### 9.2 特征分解：找到"主要的运动方向"

协方差矩阵是对称半正定矩阵，可以分解为：

$$
\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top
$$

- $\mathbf{V}$：列向量为**特征向量**（相互正交的"方向"），每个特征向量是一组股票权重
- $\boldsymbol{\Lambda}$：对角矩阵，对角元素是**特征值** $\lambda_1 \ge \lambda_2 \ge \dots$，表示沿该方向的方差

### 9.3 PCA：用少数"主成分"解释大部分波动

**比喻**：几百只股票每天的涨跌看起来杂乱无章，但其实像一支乐队——**指挥（市场整体）** 的手势决定了大部分乐手的动作，然后是**声部（行业、风格）**，最后才是每个乐手的即兴发挥。PCA 就是从录音中把"指挥"和"声部"分离出来。

**主成分（Principal Component，PC）**：

- **PC1**：解释方差最多的方向。在股票市场中，PC1 的权重通常几乎全为同号 → 它就是"**市场因子**"
- **PC2、PC3**：常常对应大盘 vs 小盘、行业轮动（如消费 vs 周期）等
- 第 $k$ 个主成分的**解释方差比例** $= \lambda_k / \sum_i \lambda_i$

```python
# 先标准化（用相关矩阵），否则高波动股票会主导 PC
z = (rets - rets.mean()) / rets.std()
eigvals, eigvecs = np.linalg.eigh(np.cov(z.T))   # eigh: 对称矩阵专用，特征值升序
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

explained = eigvals / eigvals.sum()
print("前 5 个主成分解释比例:", explained[:5].round(3))

pc1_weights = pd.Series(eigvecs[:, 0], index=rets.columns)
print("PC1 权重是否同号:", (np.sign(pc1_weights) == np.sign(pc1_weights.iloc[0])).mean())

# 或者直接用 sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=5).fit(z)
print(pca.explained_variance_ratio_)
```

### 9.4 量化意义

- **风险分解**：一个看似分散的 50 只股票组合，如果 PC1 解释了大部分方差，它本质上就是一个"市场 Beta"组合。
- **统计因子模型**：PCA 提取的成分可以作为"统计风险因子"，与 Phase 7 的基本面因子（市值、价值、动量）互补。
- **统计套利**：把股票收益中被前几个 PC 解释的部分剔除，剩下的残差更可能具有均值回归性（Phase 6）。
- **降维**：机器学习特征过多时，PCA 可压缩维度（Phase 9）。

### 9.5 陷阱：协方差矩阵估计误差

当股票数 $N$ 接近或超过样本天数 $T$ 时（例如 500 只股票、250 天数据），样本协方差矩阵**噪声极大甚至奇异**（不可逆），优化器会把权重押在"估计误差"上。

| 问题 | 补救 |
| --- | --- |
| $N$ 相对 $T$ 太大 | 延长样本、降低频率要谨慎（会减少 T） |
| 小特征值被严重低估 | **收缩估计（Shrinkage）**，如 Ledoit-Wolf |
| 结构不稳定 | 用指数加权协方差、因子模型协方差 |
| 噪声特征值难区分 | 随机矩阵理论（RMT）去噪（进阶） |

```python
from sklearn.covariance import LedoitWolf
lw_cov = LedoitWolf().fit(rets.values).covariance_ * 252
```

> 主成分的**符号是任意的**（特征向量乘以 -1 仍是特征向量）。不同时间窗口算出的 PC1 可能符号相反，滚动 PCA 时要做符号对齐。

---

## 10. 平稳性与 ADF 检验

### 10.1 直觉

**平稳（Stationary）**：序列的统计性质（均值、方差、自相关结构）**不随时间改变**。

**比喻**：

- **平稳序列**像一只被橡皮筋拴在桩子上的小球——会晃，但总会回到桩子附近。例如：日收益率、协整价差。
- **非平稳序列**像醉汉回家——每一步都是随机的，走到哪儿算哪儿，没有"回归"的位置。例如：股票价格、指数点位。

### 10.2 为什么平稳性重要

几乎所有统计模型（回归、ARMA、相关性、机器学习）都隐含一个假设：**过去学到的规律在未来仍然成立**。这要求数据的统计性质稳定——也就是平稳。

```
非平稳的价格 ─── 差分/取收益 ───▶ 近似平稳的收益率 ───▶ 可以建模
     ↓
  直接建模 → 伪回归、虚假的"规律"
```

### 10.3 弱平稳的定义

$$
\mathbb{E}[y_t] = \mu,\qquad \text{Var}(y_t) = \sigma^2,\qquad \text{Cov}(y_t, y_{t-k}) = \gamma_k
$$

- 均值 $\mu$、方差 $\sigma^2$ 都是常数，不依赖 $t$
- 自协方差 $\gamma_k$ 只依赖间隔 $k$，不依赖具体时刻 $t$

**随机游走**（价格的经典模型）：$P_t = P_{t-1} + \varepsilon_t$，其方差随时间线性增长（$\text{Var}(P_t) = t\sigma^2$），因此**非平稳**。它的一阶差分 $\Delta P_t = \varepsilon_t$ 是平稳的。这种"差分一次才平稳"的序列称为 **I(1)**（一阶单整）。

### 10.4 ADF 检验（Augmented Dickey-Fuller）

ADF 检验的回归式：

$$
\Delta y_t = \alpha + \delta t + \gamma\, y_{t-1} + \sum_{i=1}^{p} \phi_i \Delta y_{t-i} + \varepsilon_t
$$

- $\Delta y_t = y_t - y_{t-1}$
- $\alpha$：常数项；$\delta t$：时间趋势项（可选）
- $\gamma$：**关键参数**。$\gamma = 0$ 表示存在单位根（随机游走，非平稳）；$\gamma < 0$ 表示有拉回力（平稳）
- $\sum \phi_i \Delta y_{t-i}$：滞后差分项，用来吸收自相关（"Augmented"的含义）

**假设**：

- $H_0$：$\gamma = 0$，**存在单位根，序列非平稳**
- $H_1$：$\gamma < 0$，序列平稳

**p < 0.05 → 拒绝非平稳 → 认为平稳**。

**KPSS 检验**的原假设恰好相反（$H_0$：平稳）。两者结合使用更稳妥：

| ADF 结果 | KPSS 结果 | 结论 |
| --- | --- | --- |
| 拒绝 $H_0$（平稳） | 不拒绝 $H_0$（平稳） | ✅ 平稳 |
| 不拒绝（非平稳） | 拒绝（非平稳） | ❌ 非平稳 |
| 两者矛盾 | | 证据不足，需进一步分析（如趋势平稳、结构突变） |

### 10.5 代码

```python
def adf_report(series: pd.Series, name: str):
    stat, p, used_lag, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    print(f"{name:12s} ADF={stat:7.3f}  p={p:.4f}  1%临界值={crit['1%']:.2f}  "
          f"→ {'平稳' if p < 0.05 else '非平稳'}")

adf_report(close, "茅台价格")              # 通常：非平稳
adf_report(np.log(close).diff(), "对数收益")   # 通常：平稳
adf_report(spread, "协整价差")              # 若协整：平稳

# KPSS 交叉验证（原假设：平稳）
kpss_stat, kpss_p, _, _ = kpss(np.log(close).diff().dropna(), regression="c", nlags="auto")
print(f"KPSS p={kpss_p:.3f}")   # statsmodels 的 KPSS p 值有上下限截断，会给出提示
```

### 10.6 量化意义与陷阱

- **价格非平稳 → 不要直接用价格作为模型特征**。在 Phase 9 机器学习中，用价格水平作特征是典型错误。
- **差分去掉了"记忆"**：收益率平稳但几乎丢失了价格水平的信息。《Advances in Financial Machine Learning》提出**分数阶差分（Fractional Differentiation）**，在平稳性和记忆之间折中（Phase 9 会遇到）。
- **ADF 检验功效（Power）较低**：对"缓慢均值回归"的序列，ADF 常常无法拒绝单位根。不拒绝 $H_0$ 不等于证明了非平稳。
- **结构突变**：A 股制度变化（如涨跌幅制度调整、注册制改革）可能让序列在某个时点"换了一个世界"。ADF 可能把带突变的平稳序列误判为非平稳。
- **收益率"平稳"也只是近似**：均值大致稳定，但**方差明显随时间变化**（波动聚集，第 12 节）。严格来说，收益率不满足"方差恒定"。

---

## 11. 自相关与 ARMA 模型

### 11.1 自相关：今天和过去的关系

**自相关函数（ACF，Autocorrelation Function）**：序列与自身滞后 $k$ 期的相关系数。

$$
\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}
$$

- $\rho_k$：滞后 $k$ 期的自相关系数
- $\rho_k > 0$：今天涨，$k$ 天后倾向于涨（动量/趋势）
- $\rho_k < 0$：今天涨，$k$ 天后倾向于跌（反转/均值回归）

**偏自相关函数（PACF）**：剔除中间滞后期影响后，$y_t$ 与 $y_{t-k}$ 的"直接"相关。

**比喻**：ACF 像问"你和你爷爷长得像吗"——像，但可能只是因为你像你爸、你爸像你爷爷；PACF 问的是"排除你爸的影响后，你和你爷爷还像吗"。

### 11.2 ARMA 模型

**AR(p)，自回归模型**：今天的值由过去 $p$ 天的值线性决定。

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t
$$

**MA(q)，移动平均模型**（注意：与技术分析中的"均线 MA"不是一回事）：今天的值受过去 $q$ 期**随机冲击**影响。

$$
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}
$$

**ARMA(p, q)**：两者结合。**ARIMA(p, d, q)**：先差分 $d$ 次再拟合 ARMA。

- $\phi_i$：AR 系数；$\theta_j$：MA 系数
- $\varepsilon_t$：白噪声（不可预测的新信息）

### 11.3 用 ACF/PACF 识别模型阶数

| 模型 | ACF | PACF |
| --- | --- | --- |
| AR(p) | 逐渐衰减（拖尾） | p 阶后截断 |
| MA(q) | q 阶后截断 | 逐渐衰减（拖尾） |
| ARMA(p,q) | 拖尾 | 拖尾 |
| 白噪声 | 全部不显著 | 全部不显著 |

实践中更常用 **AIC/BIC 信息准则** 自动选阶：在拟合优度和参数个数之间折中，越小越好。

### 11.4 Ljung-Box 检验：整体有没有自相关

$$
Q = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}
$$

- $n$：样本量；$m$：检验的最大滞后阶数；$\hat{\rho}_k$：样本自相关
- $H_0$：前 $m$ 阶自相关全为 0（白噪声）

### 11.5 代码

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

r = np.log(close).diff().dropna()

plot_acf(r, lags=20)          # 蓝色阴影带外的柱子才算显著
plot_pacf(r, lags=20)

print(acorr_ljungbox(r, lags=[5, 10, 20]))       # 收益本身：自相关通常很弱
print(acorr_ljungbox(r**2, lags=[5, 10, 20]))    # 收益平方：通常高度显著 → 波动聚集

# 拟合 ARMA(1,1)（对收益序列，d=0）
model = ARIMA(r, order=(1, 0, 1)).fit()
print(model.summary())
print("AIC:", model.aic)
forecast = model.forecast(steps=5)
```

### 11.6 量化意义：为什么 ARMA 在收益预测上"不太好用"

- **日收益的自相关通常非常弱**（接近 0）。如果存在明显的线性自相关，早就被套利者交易掉了——这是市场有效性的一种体现。
- 对收益序列拟合 ARMA，得到的预测值往往**接近 0**，$R^2$ 极低，扣除成本后难以盈利。
- 弱自相关不代表毫无用处：
  - 在**部分个股、小市值股票或短周期**上，可能存在短期反转效应（A 股的短期反转因子在 Phase 7 中会涉及）。这类效应需要严格的样本外检验。
  - **价差序列**（协整对）常常有显著的 AR(1) 结构，这正是均值回归策略的基础（8.4 节的半衰期就是 AR(1) 的应用）。
  - **残差诊断**：模型（回归、GARCH）拟合后，用 Ljung-Box 检查残差是否还有剩余结构。
- **收益平方/绝对值的自相关却很强**——这揭示了"收益难预测，但波动可预测"，引出下一节的 GARCH。

---

## 12. 波动率聚集与 GARCH

### 12.1 直觉：波动率聚集（Volatility Clustering）

> "大波动之后往往跟着大波动，小波动之后往往跟着小波动。" —— Mandelbrot

**比喻**：天气。暴风雨天之后，第二天大概率还是阴雨；晴天之后，大概率还是晴天。你无法预测明天**往哪个方向**刮风（涨跌方向），但能较好地预测明天**风有多大**（波动幅度）。

```
日收益
 ▲
 │  ┃     ┃┃┃┃ ┃                         ┃┃┃
 │ ┃┃┃ ┃ ┃┃┃┃┃┃┃┃ ┃     ┃  ┃          ┃┃┃┃┃┃┃
─┼──────────────────────────────────────────────▶
 │ ┃┃┃  ┃┃┃┃┃┃┃┃┃┃  ┃ ┃  ┃            ┃┃┃┃┃┃┃┃
 │  ┃     ┃┃┃ ┃┃                         ┃┃┃
     ↑ 高波动期      ↑ 低波动期            ↑ 高波动期
```

### 12.2 GARCH(1,1) 模型

**GARCH（Generalized Autoregressive Conditional Heteroskedasticity，广义自回归条件异方差）**：

$$
r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim \text{i.i.d.}(0, 1)
$$

$$
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2
$$

- $r_t$：收益；$\mu$：均值
- $\sigma_t^2$：**条件方差**——基于 $t-1$ 及之前信息对 $t$ 日方差的预测
- $z_t$：标准化残差（可设为正态或 t 分布）
- $\omega > 0$：常数项
- $\alpha \ge 0$：对**昨日冲击**的反应（"新闻效应"）
- $\beta \ge 0$：对**昨日方差**的记忆（"持续性"）
- 要求 $\alpha + \beta < 1$ 才能保证方差平稳

**解读**：今天的方差 = 长期基准 + 昨天的意外冲击 + 昨天方差的惯性。

**两个重要衍生量**：

$$
\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta} \qquad \text{（长期平均方差）}
$$

$$
\text{波动冲击半衰期} = \frac{\ln 0.5}{\ln(\alpha + \beta)} \qquad \text{（冲击衰减一半所需天数）}
$$

金融日收益拟合的 $\alpha + \beta$ 通常很接近 1（如 0.95–0.99），说明波动冲击衰减很慢、持续性很强。

### 12.3 代码：用 arch 库拟合与预测

```python
r_pct = np.log(close).diff().dropna() * 100     # arch 推荐把收益放大到百分比，数值更稳定

am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
res = am.fit(disp="off")
print(res.summary())

omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
persistence = alpha + beta
print(f"α+β={persistence:.3f}, 半衰期≈{np.log(0.5) / np.log(persistence):.1f} 天")
print(f"长期年化波动≈{np.sqrt(omega / (1 - persistence) * 252):.1f}%")

# 条件波动率（样本内拟合值，年化）
cond_vol = res.conditional_volatility * np.sqrt(252)

# 预测未来 10 天的方差
fc = res.forecast(horizon=10)
next_10d_var = fc.variance.iloc[-1]                 # 每一天的方差预测（百分比平方）
print("未来 10 日年化波动预测:", (np.sqrt(next_10d_var * 252)).round(1).tolist())
```

### 12.4 GARCH 家族

| 模型 | 改进点 | `arch_model` 参数 |
| --- | --- | --- |
| GARCH(1,1) | 基准模型 | `vol="GARCH", p=1, q=1` |
| **GJR-GARCH** | 下跌冲击对波动影响更大（杠杆效应） | `vol="GARCH", p=1, o=1, q=1` |
| EGARCH | 对数方差建模，自动保证正值、允许非对称 | `vol="EGARCH"` |
| t 分布误差 | 刻画残差的肥尾 | `dist="t"` 或 `dist="skewt"` |

> **杠杆效应**（坏消息比好消息更推高波动）在美股指数上比较显著；在 A 股上的证据不完全一致，建议用 GJR-GARCH 在你自己的数据上检验 `gamma` 参数是否显著，而不是直接假设。

### 12.5 量化中的应用

```
┌─────────────────────────────────────────────────────────────────┐
│                     GARCH 波动率预测的用途                       │
├─────────────────────────────────────────────────────────────────┤
│  📌 波动率目标（Volatility Targeting）                          │
│     仓位 = 目标波动 / 预测波动 → 高波动期自动减仓（Phase 5/8）  │
│                                                                  │
│  📌 动态 VaR / 风险预算                                          │
│     VaR_t = −(μ + σ_t × 分位数) → 比固定 σ 更及时（Phase 8）     │
│                                                                  │
│  📌 期权定价与交易                                               │
│     比较 GARCH 预测波动 vs 隐含波动率（Phase 12）               │
│                                                                  │
│  📌 蒙特卡洛模拟                                                 │
│     用 GARCH 生成带波动聚集的情景，比 GBM 更真实（第 5 节）      │
│                                                                  │
│  📌 状态识别                                                     │
│     条件波动率作为"市场状态"特征（Phase 9）                     │
└─────────────────────────────────────────────────────────────────┘
```

### 12.6 陷阱

- **样本内拟合 ≠ 预测能力**。`conditional_volatility` 是样本内拟合值；评估预测能力要做**滚动样本外预测**，并与简单基准（如 20 日历史波动、EWMA）比较。很多时候 GARCH 只比 EWMA 好一点点。
- **"真实波动"不可观测**：评估波动预测时，常用"已实现波动率"（如日内高频收益平方和）作为代理。用单日收益平方作为"真值"噪声极大。
- **涨跌停与停牌**：A 股的涨跌停会截断收益，连续一字板期间的收益分布被扭曲；停牌后复牌的巨大跳空会被 GARCH 当作一次超大冲击，污染后续估计。必要时剔除或单独处理。
- **单位**：记得收益放大 100 倍后，预测结果也是"百分比"单位，换算回年化波动时注意。

---

## 13. 最佳实践与常见陷阱

### 13.1 最佳实践

```
┌──────────────────────────────────────────────────────────────────┐
│                       量化统计分析规范                             │
├──────────────────────────────────────────────────────────────────┤
│  ✅ 先画图再检验：直方图、Q-Q 图、ACF 图、滚动统计量               │
│  ✅ 用收益率（平稳）建模，不用价格（非平稳）                       │
│  ✅ 报告点估计的同时报告不确定性（标准误、置信区间、Bootstrap）    │
│  ✅ 时间序列回归默认使用 HAC 稳健标准误                            │
│  ✅ 默认假设肥尾：风险度量用历史法/t 分布，而非正态                │
│  ✅ 记录所有尝试过的假设与参数，披露尝试次数                       │
│  ✅ 所有参数估计（β、均值、协方差）都只用"当时可得"的数据         │
│  ✅ 固定随机种子，模拟结果可复现                                   │
│  ✅ 下注比例宁小勿大：半凯利或更低                                 │
└──────────────────────────────────────────────────────────────────┘
```

### 13.2 常见陷阱速查表

| # | 陷阱 | 后果 | 对策 |
| --- | --- | --- | --- |
| 1 | 假设收益正态 | 严重低估尾部风险 | t 分布、历史模拟、CVaR |
| 2 | 只看胜率 | 高胜率负期望策略 | 同时看盈亏比与期望值 |
| 3 | 误读 p 值 | 把 p=0.04 当"96% 有效" | 理解 p 值定义 |
| 4 | 多重检验 | 大量假阳性 | 修正方法、提高门槛、样本外验证 |
| 5 | 价格对价格回归 | 伪回归 | 用收益；或做协整检验 |
| 6 | 相关当协整 | 配对价差持续发散 | `coint` 检验 + 半衰期 |
| 7 | 全样本估计参数 | 前视偏差 | 滚动/扩展窗口 |
| 8 | 普通 t 检验忽略自相关 | t 值虚高 | Newey-West |
| 9 | 小样本协方差矩阵 | 组合优化结果极端 | Ledoit-Wolf 收缩 |
| 10 | GARCH 只看样本内 | 高估预测能力 | 滚动样本外 vs EWMA 基准 |
| 11 | 全凯利下注 | 估计误差导致过度下注 | 半凯利、波动率目标 |
| 12 | ADF 不拒绝就断言"非平稳" | 忽略检验功效低 | 结合 KPSS、可视化 |

---

## 14. 总结

### 14.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────┐
│                     Phase 3 核心概念地图                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📌 分布：收益率肥尾、常见负偏 → 正态假设低估极端风险             │
│     └── 偏度、超额峰度、JB 检验、Q-Q 图、VaR / CVaR                │
│                                                                   │
│  📌 推断：t = SR × √年数 → 收益难证明，波动易估计                 │
│     └── p 值、两类错误、Newey-West、多重检验                      │
│                                                                   │
│  📌 模拟：蒙特卡洛（模型驱动）vs Bootstrap（数据驱动）            │
│     └── Block Bootstrap 保留时间结构，给出 MDD/Sharpe 的分布       │
│                                                                   │
│  📌 决策：期望 = p·W − q·L；凯利 f* = p − q/b；实践用半凯利       │
│                                                                   │
│  📌 回归与关系                                                    │
│     ├── OLS + 残差诊断；警惕伪回归                                │
│     ├── 相关（短期同步） vs 协整（长期拴绳）                      │
│     └── 协方差矩阵 → 特征分解 → PCA（PC1 ≈ 市场因子）             │
│                                                                   │
│  📌 时间序列                                                      │
│     ├── 平稳性：价格 I(1)，收益 ≈ I(0)；ADF + KPSS                │
│     ├── ACF/PACF/ARMA：收益自相关弱，价差 AR(1) 有用              │
│     └── GARCH：方向难测，波动可测（α+β 接近 1）                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 学习检查对照

| README 学习检查 | 对应章节 |
| --- | --- |
| 能解释为什么高胜率不等于正期望 | 6.1–6.2 |
| 能用 t 检验判断策略收益是否显著 | 4.3–4.6 |
| 能区分相关与协整 | 8.2 |
| 能用 GARCH 预测波动率 | 12.2–12.3 |

### 14.3 与下一阶段的衔接

Phase 3 给了你一套"判断真假"的工具，但我们一直在分析**给定的收益序列**。下一个问题是：

> **策略的收益序列是怎么来的？它可信吗？**

**Phase 4：回测系统原理** 将回答：

- 如何从"信号"生成"收益序列"——向量化回测与事件驱动回测
- 如何在回测中加入交易成本、滑点、T+1、涨跌停、停牌、100 股整数手等现实约束
- 本阶段学到的统计工具如何变成回测陷阱的"侦探工具"：
  - **t 检验 / Sharpe 显著性** → 判断策略是否只是运气
  - **多重检验** → 理解为什么参数优化会制造"假圣杯"，引出 Deflated Sharpe Ratio
  - **Bootstrap** → 估计回测绩效指标的置信区间
  - **平稳性** → 理解为什么样本内规律在样本外会失效，引出 Walk-Forward

统计学告诉你"数据说了什么"；回测要确保"数据说的是真话"。

---

> 💡 **提示**：本文中涉及 A 股涨跌幅限制等交易规则的描述仅用于帮助理解，具体以交易所最新规则为准。文中的典型数值（如 t 分布自由度、α+β 范围）是经验量级，请以你在自己数据上的计算结果为准。
