# Phase 9: 机器学习量化 - 核心概念深度解析

> **学习目标**：掌握金融机器学习的正确方法论——标签怎么打、交叉验证怎么做、特征重要性怎么看、预测怎么变成持仓、回测结果怎么去伪存真
> **前置知识**：Phase 4（回测陷阱、Walk-Forward）、Phase 7（因子、IC、预处理）、Phase 8（组合优化、换手）；机器学习基础（你已经熟悉 PyTorch 与常规 ML 流程）
> **主要参考**：López de Prado《Advances in Financial Machine Learning》（下文简称 AFML）

---

## 目录

1. [引言：金融 ML 为什么特殊](#1-引言金融-ml-为什么特殊)
2. [特征工程：价量、基本面与另类特征](#2-特征工程价量基本面与另类特征)
3. [标签：Triple-Barrier 与 Meta-Labeling](#3-标签triple-barrier-与-meta-labeling)
4. [交叉验证：Purged K-Fold 与 Embargo](#4-交叉验证purged-k-fold-与-embargo)
5. [LightGBM 截面收益预测](#5-lightgbm-截面收益预测)
6. [特征重要性：MDI、MDA 与 SHAP](#6-特征重要性mdimda-与-shap)
7. [从预测分数到持仓](#7-从预测分数到持仓)
8. [回测过拟合：Deflated Sharpe Ratio 与 PBO](#8-回测过拟合deflated-sharpe-ratio-与-pbo)
9. [Qlib 完整工作流](#9-qlib-完整工作流)
10. [模型漂移与再训练](#10-模型漂移与再训练)
11. [最佳实践与常见陷阱](#11-最佳实践与常见陷阱)
12. [总结](#12-总结)

---

## 1. 引言：金融 ML 为什么特殊

### 1.1 你熟悉的 ML 与金融 ML 的差异

你在图像分类、NLP 中积累的直觉，在金融数据上很多会失效：

| 维度 | CV / NLP | 金融收益预测 |
|---|---|---|
| **信噪比** | 高：猫就是猫，标签几乎无噪声 | 极低：明天的收益绝大部分是噪声 |
| **好模型的指标** | 准确率 95%+ | 日度截面 Rank IC 0.05 已经算不错；方向准确率 52% 可能就能赚钱 |
| **平稳性** | 数据分布基本稳定 | 非平稳：市场结构、参与者、政策持续变化 |
| **样本独立性** | IID 假设大致成立 | 样本重叠、自相关、截面相关，**非 IID** |
| **数据量** | 可以无限采集、增强 | 历史只有一条路径；A 股约 20-30 年、几千只股票 |
| **对抗性** | 猫不会因为你识别它而改变 | 市场会**适应**：alpha 被发现后会被套利掉 |
| **评估** | 测试集准确率 | 扣除成本后的样本外收益、回撤、容量 |

> **比喻**：普通 ML 像在安静的房间里听人说话；金融 ML 像在摇滚演唱会现场，试图听清 50 米外一个人的耳语——而且这个人发现你在偷听后，会换一种语言。

### 1.2 低信噪比的量化理解

假设一只股票日收益 $r = s + \epsilon$，可预测部分 $s$ 的标准差为 0.1%，噪声 $\epsilon$ 的标准差为 2%。即使你**完美**预测了 $s$，预测与实际收益的相关性也只有：

$$
\rho(s, r) = \frac{\sigma_s}{\sqrt{\sigma_s^2 + \sigma_\epsilon^2}} = \frac{0.1}{\sqrt{0.01 + 4}} \approx 0.05
$$

- $\sigma_s$：可预测信号的标准差；$\sigma_\epsilon$：不可预测噪声的标准差

这就是为什么 IC 0.05 已经很好。也意味着：**模型容量稍大就会去拟合噪声**，深模型、多参数在金融上极易过拟合。

### 1.3 非 IID 的三个来源

```
┌──────────────────────────────────────────────────────────────────────┐
│                       金融样本为什么不是 IID                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 标签重叠（时间维度）                                                │
│     t=1 的标签 = 未来 5 日收益 [1, 6]                                   │
│     t=2 的标签 = 未来 5 日收益 [2, 7]   ← 与 t=1 共享 4 天               │
│     → 相邻样本高度相关，"有效样本数"远小于名义样本数                      │
│                                                                      │
│  ② 截面相关（股票维度）                                                │
│     同一天 3000 只股票共同受大盘、行业影响                               │
│     → 3000 个样本 ≠ 3000 个独立观测                                     │
│                                                                      │
│  ③ 非平稳（分布漂移）                                                  │
│     2015 年牛熊、2017 年白马行情、2021 年小盘行情……                      │
│     → 训练集与测试集来自不同分布                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

后果：普通的随机 K-Fold 会严重高估模型表现（第 4 节），基于 IID 假设的显著性检验会给出过于乐观的 p 值（第 8 节）。

### 1.4 研究范式：López de Prado 的建议

AFML 第 1 章批评"西西弗斯范式"（每个研究员独自从数据到回测做完全流程，反复回测直到找到好看的结果），提倡流水线式的"元策略范式"：数据、特征、建模、回测、部署由不同环节专业化分工。对个人研究者的启示是：

- **把回测当作验证而不是研究工具**。AFML 第 8 章的名言：*"Backtesting is not a research tool. Feature importance is."*
- 记录每一次尝试（第 8 节会用到尝试次数 $N$）

---

## 2. 特征工程：价量、基本面与另类特征

### 2.1 特征分类

| 类别 | 例子 | 更新频率 | 关键陷阱 |
|---|---|---|---|
| **价量** | 动量、反转、波动率、换手率、Amihud 非流动性、量价相关性 | 日/分钟 | 复权方式、停牌、涨跌停 |
| **基本面** | ROE、毛利率、PE、PB、营收增速、应计项 | 季度 | **Point-in-Time**：必须用公告日而不是报告期（Phase 7） |
| **一致预期** | 分析师 EPS 预期、评级调整 | 不定期 | 覆盖偏差（小盘股没人覆盖） |
| **另类数据** | 新闻情绪、舆情、供应链、招聘、卫星图像 | 不定 | 时间戳、样本历史短、幸存者偏差（Phase 10） |
| **市场状态** | 指数波动率、市场宽度、资金流 | 日 | 对所有股票相同，在截面模型中需与个股特征交互才有用 |

### 2.2 截面预处理：金融特征工程的核心

截面选股的目标是**同一天内对股票排序**，因此特征要在每个交易日内做标准化，消除市场整体水平的漂移：

$$
\tilde x_{i,t} = \frac{x_{i,t} - \text{median}_t(x)}{\text{MAD}_t(x)} \quad\text{或}\quad \tilde x_{i,t} = \frac{\text{rank}_t(x_{i,t})}{N_t} - 0.5
$$

- $x_{i,t}$：股票 $i$ 在日期 $t$ 的原始特征
- $\text{median}_t, \text{MAD}_t$：当天截面的中位数与绝对中位差（对极值稳健）
- $\text{rank}_t$：当天截面排名；$N_t$：当天有效股票数

```python
import numpy as np
import pandas as pd

def make_features(px: pd.DataFrame) -> pd.DataFrame:
    """px: MultiIndex (datetime, instrument)，列含 close, volume, turnover, amount
    返回同索引的特征。所有计算只用 t 及之前的数据。"""
    g = px.groupby(level="instrument", group_keys=False)
    ret = g["close"].pct_change()
    f = pd.DataFrame(index=px.index)
    f["mom_20"] = g["close"].pct_change(20)
    f["rev_5"] = -g["close"].pct_change(5)
    f["vol_20"] = ret.groupby(level="instrument").transform(lambda s: s.rolling(20).std())
    f["turn_20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    f["illiq_20"] = (ret.abs() / px["amount"]).groupby(level="instrument") \
        .transform(lambda s: s.rolling(20).mean())          # Amihud 非流动性
    return f

def cs_rank(f: pd.DataFrame) -> pd.DataFrame:
    """按交易日截面排名，映射到 [-0.5, 0.5]"""
    return f.groupby(level="datetime").rank(pct=True) - 0.5
```

### 2.3 平稳性与记忆的权衡：分数阶差分

价格序列非平稳（Phase 3 的 ADF 检验），一阶差分（收益率）平稳但**丢掉了几乎所有记忆**（价格水平信息）。AFML 第 5 章提出**分数阶差分（Fractional Differentiation）**：差分阶数 $d \in (0, 1)$，在"刚好平稳"的前提下保留最多记忆。

$$
\tilde X_t = \sum_{k=0}^{\infty} \omega_k X_{t-k}, \qquad \omega_0 = 1,\quad \omega_k = -\omega_{k-1}\frac{d - k + 1}{k}
$$

- $d = 1$ 时权重为 $(1, -1, 0, 0, \dots)$，就是普通差分
- $d = 0.4$ 时权重缓慢衰减，序列保留长期记忆
- 实践中用固定窗口截断权重（权重绝对值小于阈值就丢弃），找使 ADF 检验通过的最小 $d$

```python
def frac_diff_weights(d: float, thresh: float = 1e-4, max_len: int = 1000) -> np.ndarray:
    w = [1.0]
    for k in range(1, max_len):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w)

def frac_diff(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    w = frac_diff_weights(d, thresh)
    return series.rolling(len(w)).apply(lambda x: np.dot(w, x[::-1]), raw=True)
```

> 在截面选股中，大多数特征本身就是比率或收益类，平稳性问题不如时序择时突出；分数阶差分在单资产时序模型（期货、加密）中更常用。

### 2.4 采样方式：不只有时间 K 线

AFML 第 2 章指出，按固定时间采样的 K 线在交易活跃时信息过少、清淡时信息过多。替代方案：

| 采样方式 | 每根 bar 的定义 | 性质 |
|---|---|---|
| 时间 bar | 固定时间（1 分钟、1 天） | 最常用；收益分布异方差明显 |
| Tick bar | 固定成交笔数 | 跟随交易活跃度 |
| Volume bar | 固定成交量 | 收益分布更接近正态 |
| Dollar bar | 固定成交额 | 对价格大幅变化更稳健 |
| 事件采样（CUSUM 滤波） | 累计变化超过阈值才采样 | 只在"有事发生"时生成样本，减少冗余 |

日频截面选股一般仍用时间 bar；高频和单资产策略可尝试其他 bar。

### 2.5 特征工程的陷阱

- **用全样本统计量标准化**：例如用 2010-2023 全部数据的均值方差做 z-score，测试期信息泄漏进训练期。截面标准化（按天）天然避免这一点；时序标准化必须用滚动窗口。
- **财务数据前视**：用报告期（如 3 月 31 日）对齐一季报，但一季报 4 月底才公布。
- **幸存者偏差**：只用当前存续的股票训练，忽略退市股。
- **特征太多**：几百个高度相关的特征会让特征重要性失真（第 6 节），也加剧过拟合。

---

## 3. 标签：Triple-Barrier 与 Meta-Labeling

### 3.1 固定期限标签的问题

最常见的标签是"未来 $h$ 天收益的符号"：

$$
y_t = \text{sign}(r_{t, t+h})
$$

问题：

1. **忽略路径**：未来 5 天先跌 15% 再涨回 +1%，标签为正——但现实中你早就被止损了
2. **固定阈值不适应波动**：同样 2% 的涨幅，对低波动的银行股是大事件，对高波动的题材股是噪声
3. **大量 0 附近的标签**是纯噪声

### 3.2 Triple-Barrier 方法（AFML 第 3 章）

> **比喻**：每笔交易都设置止盈、止损和最长持有时间三道"关卡"。标签不是"5 天后涨没涨"，而是"**最先碰到哪道关卡**"——这和真实交易员的决策方式一致。

```
  价格
   ▲
   │  ─────────────────────────────────── 上边界：p₀·(1 + pt·σ_t)  → 标签 +1
   │                    ╱╲
   │        ╱╲      ╱╲╱  ╲
 p₀●───╱╲╱  ╲╱╲╱          ╲
   │                        ╲    ╱
   │                         ╲╱╲╱
   │  ─────────────────────────────────── 下边界：p₀·(1 - sl·σ_t)  → 标签 -1
   │                                    │
   └────────────────────────────────────┼──▶ 时间
   t₀                                  t₀+h  垂直边界 → 标签 sign(收益) 或 0
```

- **上边界（止盈）**：$p_0(1 + pt\cdot\sigma_{t_0})$
- **下边界（止损）**：$p_0(1 - sl\cdot\sigma_{t_0})$
- **垂直边界（最长持有）**：$t_0 + h$
- $\sigma_{t_0}$：$t_0$ 时刻的波动率估计（例如日收益的 EWM 标准差），使边界随波动自适应
- $pt, sl$：止盈止损倍数，可以不对称

每个样本除了标签，还记录**结束时间 $t_1$**（首次触碰边界的时间）。$t_1$ 是后面 Purged CV 和样本权重的关键。

### 3.3 代码

```python
import numpy as np
import pandas as pd

def daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    return close.pct_change().ewm(span=span).std()

def triple_barrier(close: pd.Series, events: pd.DatetimeIndex,
                   pt: float = 1.0, sl: float = 1.0, max_hold: int = 10) -> pd.DataFrame:
    vol = daily_vol(close)
    idx = close.index
    rows = []
    for t0 in events:
        if pd.isna(vol.get(t0)):
            continue
        i0 = idx.get_loc(t0)
        i_end = min(i0 + max_hold, len(idx) - 1)
        path = close.iloc[i0 + 1:i_end + 1] / close.iloc[i0] - 1
        up, dn = pt * vol[t0], -sl * vol[t0]
        hits = pd.Series({
            "pt": path.index[path >= up].min(),     # 空则为 NaT
            "sl": path.index[path <= dn].min(),
            "vb": idx[i_end],                        # 垂直边界
        }).dropna().sort_values(kind="stable")      # 同时触碰时优先 pt/sl
        barrier, t1 = hits.index[0], hits.iloc[0]
        ret = close[t1] / close[t0] - 1
        label = {"pt": 1, "sl": -1}.get(barrier, int(np.sign(ret)))
        rows.append((t0, t1, ret, label, barrier))
    return pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier"]).set_index("t0")
```

注意：`vol[t0]` 只用到 $t_0$ 及之前的数据（EWM 是因果的），满足无前视。`events` 可以是每天，也可以是 CUSUM 滤波器挑出的事件日。

### 3.4 Meta-Labeling：把"方向"和"是否下注/下多大"分开

> **比喻**：主模型是"侦察兵"，负责发现机会并判断方向（做多还是做空）；Meta 模型是"指挥官"，看着侦察兵的报告和当前环境，决定"这次要不要真的出击、派多少兵"。

步骤：

1. **主模型**（可以是简单规则如均线交叉、或一个 ML 模型、或人的判断）给出方向 $s_t \in \{-1, +1\}$
2. 按方向应用 Triple-Barrier，**Meta 标签**为 $y^{meta}_t = \mathbb{1}[s_t \cdot r_{t,t_1} > 0]$（这笔交易赚没赚）
3. **Meta 模型**（二分类）用特征预测 $P(y^{meta}=1)$
4. 只在概率高时下注，并按概率决定仓位大小

```python
from sklearn.ensemble import RandomForestClassifier

# 1) 主模型：20/60 均线交叉给出方向
side = np.sign(close.rolling(20).mean() - close.rolling(60).mean())
events = side.diff().fillna(0).ne(0)          # 方向变化日作为事件
events = close.index[events & side.ne(0)]

# 2) 打标签（这里只需用 ret 与 side 判断盈亏）
tb = triple_barrier(close, events, pt=2.0, sl=1.0, max_hold=20)
meta_y = (side.loc[tb.index] * tb["ret"] > 0).astype(int)

# 3) Meta 模型（X_meta 为事件日的特征：波动率、市场状态、主信号强度等）
clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features=0.5)
# clf.fit(X_meta_train, meta_y_train)  —— 务必用第 4 节的 Purged CV 训练/验证
# prob = clf.predict_proba(X_meta_test)[:, 1]
```

Meta-Labeling 的好处：

- 主模型可以追求**高召回**（尽量不漏机会），Meta 模型负责提高**精确率**（过滤假信号）
- 天然输出概率，可直接用于**仓位管理**（第 7.4 节）
- 可以叠加在**非 ML 的主策略**上（比如 Phase 5 的 CTA 策略），用 ML 过滤信号而不是替代策略逻辑

### 3.5 样本权重：处理标签重叠（AFML 第 4 章）

标签区间 $[t_0, t_1]$ 重叠时，样本之间共享信息。AFML 提出用**平均唯一度（average uniqueness）**度量：

$$
c_t = \sum_i \mathbb{1}[t \in [t_{0,i}, t_{1,i}]], \qquad \bar u_i = \frac{1}{t_{1,i} - t_{0,i} + 1}\sum_{t = t_{0,i}}^{t_{1,i}} \frac{1}{c_t}
$$

- $c_t$：时刻 $t$ 有多少个样本的标签区间覆盖它（并发数）
- $\bar u_i$：样本 $i$ 在其生命期内的平均"独占程度"，$\in (0, 1]$

用法：作为训练样本权重；或在随机森林中用 `max_samples` ≈ 平均唯一度，降低 bagging 中样本冗余；AFML 还提出**顺序自助法（Sequential Bootstrap）**，在抽样时降低重叠样本被同时抽中的概率。

---

## 4. 交叉验证：Purged K-Fold 与 Embargo

### 4.1 普通 K-Fold 为什么会泄漏

普通 K-Fold 随机打乱后划分。在金融数据上至少有两种泄漏：

```
时间轴 ──────────────────────────────────────────────────────────▶

  样本 t=100：特征用到第 100 天，标签 = 第 100-105 天收益
  样本 t=103：特征用到第 103 天，标签 = 第 103-108 天收益

  若 t=100 在训练集，t=103 在测试集：
  → 训练标签包含了第 103-105 天的收益，而这正是测试标签的一部分
  → 模型"记住"了测试期的部分答案

  此外特征本身有自相关（20 日动量在相邻两天几乎一样），
  打乱后训练集里到处是测试样本的"近亲"
```

结果：交叉验证分数虚高，上线后表现崩塌。**"能解释普通 K-Fold 在金融数据上的泄漏问题"**是本阶段的核心检查项。

### 4.2 Purging 与 Embargo

**Purging（清洗）**：从训练集中删除所有**标签区间与测试集标签区间有重叠**的样本：

$$
\text{删除训练样本 } i \quad \text{若}\quad [t_{0,i}, t_{1,i}] \cap [t_0^{test}, t_1^{test}] \neq \emptyset
$$

**Embargo（禁运）**：在每个测试块**之后**再额外删除一段训练样本（通常为总样本的 1% 左右，或与标签期限相当）。原因：测试期之后的训练样本的**特征**（如滚动均值）仍包含测试期信息，序列相关会造成泄漏。

```
                  ┌────────── 测试集 ──────────┐
  训练  训练  训练 │purge│  测试  测试  测试   │embargo│ 训练  训练
  ████  ████  ███ ░░░░░ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ░░░░░░░ ████  ████
                  ↑                          ↑
       标签区间与测试重叠的样本被删除    测试后一段时间也删除
```

### 4.3 代码：事件型（AFML 风格）

```python
import numpy as np
import pandas as pd

class PurgedKFold:
    """t1: pd.Series，index 为样本开始时间 t0（已排序），值为标签结束时间 t1"""
    def __init__(self, n_splits=5, t1: pd.Series = None, embargo_pct=0.01):
        self.n_splits, self.t1, self.embargo_pct = n_splits, t1, embargo_pct

    def split(self, X):
        n = len(X)
        t0 = self.t1.index.values
        t1 = self.t1.values
        idx = np.arange(n)
        embargo = int(n * self.embargo_pct)
        for test_idx in np.array_split(idx, self.n_splits):
            start, end = test_idx[0], test_idx[-1]
            test_t0, test_t1 = t0[start], t1[test_idx].max()
            overlap = (t0 <= test_t1) & (t1 >= test_t0)        # purge
            train_mask = ~overlap
            train_mask[end + 1:end + 1 + embargo] = False      # embargo
            train_mask[test_idx] = False
            yield idx[train_mask], test_idx
```

### 4.4 代码：截面面板（按交易日划分）

截面选股中，同一天有几千个样本，必须**按日期**划分，否则同一天的股票会同时出现在训练和测试中（截面相关泄漏）：

```python
def purged_date_splits(dates: np.ndarray, n_splits=5, horizon=5, embargo=5):
    """dates: 每个样本的交易日；horizon: 标签期限（天）"""
    udates = np.sort(np.unique(dates))
    m = len(udates)
    for block in np.array_split(np.arange(m), n_splits):
        lo, hi = block[0], block[-1]
        drop = np.zeros(m, dtype=bool)
        drop[max(0, lo - horizon):min(m, hi + 1 + horizon + embargo)] = True
        train_days, test_days = udates[~drop], udates[lo:hi + 1]
        yield np.flatnonzero(np.isin(dates, train_days)), np.flatnonzero(np.isin(dates, test_days))
```

- 测试块前 `horizon` 天：其标签区间会延伸进测试期 → purge
- 测试块后 `horizon` 天：其标签区间与测试期末尾样本的标签区间重叠 → purge
- 再加 `embargo` 天缓冲

### 4.5 K-Fold、Walk-Forward 与 CPCV

| 方法 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **Walk-Forward**（Phase 4） | 只用过去训练、未来测试，滚动前进 | 最贴近实盘 | 只有一条回测路径；早期测试段训练数据少 |
| **Purged K-Fold** | 允许用"未来"数据训练（测试块在中间），但清洗重叠 | 每个样本都被测试一次，数据利用率高 | 训练集包含测试期之后的数据，对强非平稳性有乐观偏差 |
| **CPCV**（AFML 第 12 章） | 把数据分为 $N$ 组，每次取 $k$ 组做测试，组合出多条回测路径 | 得到**回测表现的分布**而非单一值 | 计算量大 |

实践建议：**超参数调优用 Purged K-Fold 或 CPCV，最终评估用严格的 Walk-Forward 样本外**，并保留一段从未碰过的最终测试期。

---

## 5. LightGBM 截面收益预测

### 5.1 问题设定

```
┌──────────────────────────────────────────────────────────────────────┐
│                         截面预测的数据形态                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   (datetime, instrument) │ feat_1 │ feat_2 │ ... │ feat_k │ label     │
│   ───────────────────────┼────────┼────────┼─────┼────────┼────────── │
│   2020-01-02, 600000     │  0.31  │ -0.12  │ ... │  0.05  │ 未来收益   │
│   2020-01-02, 600004     │ -0.20  │  0.44  │ ... │ -0.33  │ 未来收益   │
│   ...（当天约 3000-5000 只）                                          │
│   2020-01-03, 600000     │  ...                                       │
│                                                                      │
│   目标：不是预测收益的绝对值，而是预测"同一天里谁比谁好"                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 为什么是 GBDT

- 表格数据上，梯度提升树（LightGBM、XGBoost、CatBoost）通常是强基线，往往优于或持平深度模型
- 自动处理非线性和特征交互（如"低估值**且**高质量"）
- 对特征的单调变换不敏感，对缺失值有原生处理
- 训练快，便于做大量滚动训练

### 5.3 标签的选择

| 标签 | 定义 | 说明 |
|---|---|---|
| 原始收益 | $r_{i,t+1:t+h}$ | 受市场整体涨跌主导，模型浪费容量去拟合市场 |
| 超额收益 | $r_i - r_{\text{benchmark}}$ | 去除市场，但仍有截面尺度差异 |
| **截面标准化收益** | 当天截面 z-score 或 rank | **最常用**：直接对齐"排序"目标 |
| 排序学习 | LambdaRank / 分组排序目标 | 直接优化排序指标（如 NDCG），需要离散相关度标签和分组 |

### 5.4 评估：不看 MSE，看 IC

MSE 对截面选股几乎没有意义（预测全为 0 也能得到很低的 MSE）。应看：

$$
\text{RankIC}_t = \text{Spearman}\left(\hat y_{\cdot, t},\ r_{\cdot, t+1:t+h}\right), \qquad \text{ICIR} = \frac{\text{mean}(\text{RankIC}_t)}{\text{std}(\text{RankIC}_t)}
$$

- $\text{RankIC}_t$：第 $t$ 天预测值与实际未来收益的截面 Spearman 相关
- ICIR：IC 的稳定性，类似夏普比率
- 还要看**分组回测**（Phase 7）：按预测值分 10 组，各组收益是否单调

### 5.5 代码

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

params = dict(
    objective="regression",
    learning_rate=0.02,
    num_leaves=63,
    min_data_in_leaf=500,          # 每个叶子至少 500 个样本：低信噪比下的关键正则
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=42,
)

def train_lgb(X_tr, y_tr, X_va, y_va):
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    return lgb.train(
        params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

def daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    df = pd.DataFrame({"pred": pred, "label": label}).dropna()
    return df.groupby(level="datetime").apply(
        lambda g: g["pred"].corr(g["label"], method="spearman"))

# 标签：未来 5 日收益的截面排名
# label = fwd_ret_5.groupby(level="datetime").rank(pct=True) - 0.5
# 训练/验证/测试按时间切分，验证集与训练集之间留出 ≥ horizon 天的间隔
```

超参数经验（低信噪比场景）：

| 参数 | 方向 | 原因 |
|---|---|---|
| `min_data_in_leaf` | 调大（数百到数千） | 防止叶子拟合个别噪声样本 |
| `num_leaves` / `max_depth` | 适中 | 交互不需要太深 |
| `learning_rate` | 小（0.01-0.05） + 早停 | 平滑学习 |
| `feature_fraction` / `bagging` | 0.5-0.8 | 降低方差，减少对单个强特征的依赖 |
| 早停验证集 | **按时间切分、与训练集隔开** | 随机验证集会泄漏 |

### 5.6 "准确率提高 ≠ 收益提高"

这是本阶段另一个核心检查项。原因：

1. **准确率忽略幅度**：模型对 99 只小涨的股票判断正确、对 1 只暴跌 50% 的股票判断错误，准确率 99%，但可能亏钱
2. **截面排序才是赚钱的东西**：多空组合只关心头尾，中间股票的预测准确与否无关紧要
3. **成本**：更"准"的模型若预测变化更快，换手更高，扣费后可能更差
4. **可交易性**：模型可能在涨停（买不进）、停牌、极小市值股票上最"准"
5. **风格暴露**：IC 提升可能只是因为模型学会了押注小市值，承担了风格风险

**正确的评估层级**：RankIC/ICIR → 分组收益单调性 → 扣费 TopK 组合收益 → 归因（Phase 8）看特异收益。

---

## 6. 特征重要性：MDI、MDA 与 SHAP

### 6.1 为什么特征重要性比回测更重要

AFML 的观点：回测只给你一个数字（夏普），反复回测会导致过拟合；而特征重要性告诉你**模型为什么有效**，让你能判断这个理由在经济上是否站得住。

> **比喻**：回测像考试分数，特征重要性像查看学生的解题过程。分数高但解题过程是"偷看了答案"（泄漏特征）或"蒙对的"（噪声特征），就不值得信任。

### 6.2 三种方法

| 方法 | 原理 | 样本内/外 | 优点 | 缺点 |
|---|---|---|---|---|
| **MDI**（Mean Decrease Impurity） | 每个特征在所有分裂中带来的不纯度下降（或 gain）之和 | 样本内 | 快，训练完即得 | 样本内，会给噪声特征正的重要性；偏向取值多的连续特征；无法识别过拟合 |
| **MDA**（Mean Decrease Accuracy，置换重要性） | 在验证集上打乱某特征，看得分下降多少 | 样本外 | 模型无关；直接度量对 OOS 表现的贡献 | 慢；相关特征会互相"掩护"（替代效应） |
| **SHAP** | 基于 Shapley 值，把每个预测分解为各特征的贡献之和 | 两者皆可 | 有理论基础；可给出单样本解释与方向 | 计算量较大（TreeSHAP 对树模型很快）；同样受相关特征影响 |

另外 AFML 还提出 **SFI（Single Feature Importance）**：每次只用一个特征训练并做样本外评估，完全不受替代效应影响，但忽略交互。

### 6.3 SHAP 的公式

$$
\hat f(x) = \phi_0 + \sum_{j=1}^{k}\phi_j(x), \qquad
\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(k - |S| - 1)!}{k!}\left[f_{S\cup\{j\}}(x_{S\cup\{j\}}) - f_S(x_S)\right]
$$

- $\phi_0$：基准值（平均预测）；$\phi_j(x)$：特征 $j$ 对样本 $x$ 的贡献
- $F$：全部特征集合，$k = |F|$；$S$：不含 $j$ 的特征子集
- $f_S$：只知道子集 $S$ 中特征时的模型期望输出
- 直觉：特征 $j$ 加入各种"特征联盟"时带来的平均边际贡献

### 6.4 代码

```python
import numpy as np
import pandas as pd
import shap

# --- MDI：LightGBM 原生 gain 重要性 ---
mdi = pd.Series(model.feature_importance(importance_type="gain"),
                index=model.feature_name()).sort_values(ascending=False)

# --- MDA：在样本外按 RankIC 计算的置换重要性 ---
def mda_rank_ic(model, X_te: pd.DataFrame, y_te: pd.Series, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = daily_rank_ic(pd.Series(model.predict(X_te), X_te.index), y_te).mean()
    out = {}
    for col in X_te.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            # 在每个交易日内部打乱，保持截面分布不变
            Xp[col] = Xp.groupby(level="datetime")[col].transform(
                lambda s: rng.permutation(s.values))
            ic = daily_rank_ic(pd.Series(model.predict(Xp), Xp.index), y_te).mean()
            drops.append(base - ic)
        out[col] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)

# --- SHAP ---
X_sample = X_te.sample(20_000, random_state=0)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_sample)                     # (n_samples, n_features)
shap_global = pd.Series(np.abs(sv).mean(axis=0), index=X_sample.columns)
# shap.summary_plot(sv, X_sample)  # 蜂群图：看方向与非线性

# LightGBM 也能直接输出 SHAP 贡献（最后一列是 bias）
contrib = model.predict(X_sample, pred_contrib=True)
```

### 6.5 替代效应与聚类特征重要性

如果 `mom_20` 和 `mom_21` 几乎相同，MDA 打乱其中一个，模型可以用另一个替代，两者重要性都显得很低；MDI 则会把重要性在两者间随机分摊。

对策：

- 先对特征做**相关性聚类**（与 Phase 8 HRP 的聚类方法类似），以簇为单位计算重要性——López de Prado 后续提出的 **Clustered Feature Importance**
- 或用 PCA 正交化后再看重要性（牺牲可解释性）

### 6.6 量化中的用法

- **泄漏检测**：某个特征重要性异常高（比如"公告日期"或"未来复权因子"），首先怀疑泄漏
- **经济合理性检查**：重要特征是否有合理故事？SHAP 方向是否符合直觉（如高换手 → 低未来收益）？
- **特征筛选**：去掉 MDA 为负或接近 0 的特征（在 Purged CV 上计算）
- **监控**：重要性排序随时间剧变，是模型漂移的信号（第 10 节）

---

## 7. 从预测分数到持仓

### 7.1 映射方式

```
┌──────────────────────────────────────────────────────────────────────┐
│                      预测分数 → 持仓 的几种方式                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ① TopK 等权       选分数最高的 K 只，等权买入（最简单、最常用）        │
│   ② TopK-Dropout    持有 K 只，每期只替换最差的 n 只（控制换手）        │
│   ③ 分数加权         w ∝ max(score, 0) 或 w ∝ rank                    │
│   ④ 多空分位        做多前 10%、做空后 10%（A 股做空受限，常用期货对冲）  │
│   ⑤ 优化器          score → 预期收益 α → Phase 8 的 cvxpy 约束优化      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 分数到预期收益

优化器需要的是**预期收益的尺度**，不是任意分数。Grinold 的经验公式：

$$
\alpha_{i} = \text{IC} \times \sigma_i \times z_i
$$

- $z_i$：截面标准化后的预测分数（均值 0、标准差 1）
- $\sigma_i$：股票 $i$ 的残差波动率（预测期限对应的）
- IC：模型历史上的平均 IC

直觉：预测能力越弱（IC 小），alpha 越被压缩向 0；波动大的股票，同样的分数对应更大的收益幅度。

### 7.3 TopK-Dropout（Qlib 默认策略）

Qlib 的 `TopkDropoutStrategy` 的逻辑：持有约 `topk` 只股票，每个调仓日把当前持仓中分数最低的 `n_drop` 只卖出，从未持仓股票中买入分数最高的 `n_drop` 只。其直觉是**只替换"最不想要"的那部分**，换手率被 `n_drop / topk` 粗略限制。

```python
def topk_dropout(scores: pd.Series, holdings: list, topk=50, n_drop=5) -> list:
    """scores: 当天全市场预测分数（已剔除不可交易股票）"""
    held = [s for s in holdings if s in scores.index]
    if len(held) < topk:                                # 建仓/补仓
        candidates = scores.drop(held).nlargest(topk - len(held)).index.tolist()
        return held + candidates
    sell = scores[held].nsmallest(n_drop).index.tolist()
    buy = scores.drop(held).nlargest(n_drop).index.tolist()
    return [s for s in held if s not in sell] + buy
```

> 真实 Qlib 实现还处理了涨跌停/停牌无法成交、只卖出排名确实被新股票超过的持仓等细节，这里是简化版。

### 7.4 用概率做仓位（Bet Sizing，AFML 第 10 章）

Meta-Labeling 输出的概率 $p = P(\text{盈利})$ 可以转为仓位。对二分类，检验"$p$ 显著大于 0.5"的统计量：

$$
z = \frac{p - 0.5}{\sqrt{p(1-p)}}, \qquad m = 2\Phi(z) - 1
$$

- $m \in [-1, 1]$：仓位大小（乘以主模型方向得到最终仓位）
- $\Phi$：标准正态 CDF
- $p = 0.5$ 时 $m = 0$；$p$ 越接近 1，仓位越接近满仓

AFML 还建议对仓位做**离散化**（如按 0.1 步长）并对同时活跃的多个信号取平均，避免仓位因概率的微小波动而频繁调整。

### 7.5 平滑与换手

ML 预测往往每天剧烈变化。常见处理：

- **分数平滑**：$\tilde s_t = \lambda \tilde s_{t-1} + (1-\lambda)s_t$（EMA），牺牲一点及时性换取更低换手
- **降低调仓频率**：根据 IC 衰减曲线选择周度/双周
- **换手惩罚**：放进 Phase 8 的优化目标

### 7.6 检查清单

- 剔除当天**停牌、涨停（买不进）、ST、上市不满 N 天**的股票再排序
- 用**次日开盘价或 VWAP** 成交，而不是信号当天收盘价（信号用了当天收盘数据）
- 计入 Phase 8 第 12 节的完整成本
- 对比 Phase 7 线性多因子：ML 的超额究竟来自哪里？做因子归因

---

## 8. 回测过拟合：Deflated Sharpe Ratio 与 PBO

### 8.1 问题：多次试验的选择偏差

如果你尝试了 100 个策略变体，每个都是**纯噪声**（真实夏普为 0），最好的那个在样本内的夏普依然会相当可观。这就是 Phase 4 讲过的多重检验问题。

> **比喻**：让 1000 只猴子扔硬币，每只扔 10 次。总会有一只连续 10 次正面——但它不是"扔硬币天才"。你报告的夏普，必须与"在同样多次尝试中，最好的猴子能达到的夏普"相比。

### 8.2 Probabilistic Sharpe Ratio（PSR）

夏普估计量本身有抽样误差，且受偏度、峰度影响。Bailey & López de Prado（2012）给出：真实夏普超过某基准 $SR^*$ 的概率：

$$
\widehat{PSR}(SR^*) = \Phi\left(\frac{(\widehat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat\gamma_3\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right)
$$

- $\widehat{SR}$：**非年化**的样本夏普（与收益同频率，例如日夏普）
- $T$：收益观测数
- $\hat\gamma_3$：收益偏度；$\hat\gamma_4$：收益峰度（**非超额**，正态分布为 3）
- 负偏度、高峰度会增大分母，降低 PSR——"卖期权"类策略的夏普更不可信

### 8.3 Deflated Sharpe Ratio（DSR）

Bailey & López de Prado（2014）：把 PSR 的基准 $SR^*$ 设为"$N$ 次独立试验中，纯噪声下最大夏普的期望"：

$$
SR_0 = \sqrt{V[\widehat{SR}_n]}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)\right]
$$

$$
DSR = \widehat{PSR}(SR_0)
$$

- $N$：独立试验次数（参数组合、特征集、模型……的有效数量）
- $V[\widehat{SR}_n]$：所有试验的夏普估计值的方差
- $\gamma \approx 0.5772$：Euler-Mascheroni 常数；$e$：自然常数
- DSR > 0.95 才能在 5% 显著性下认为策略的夏普不是选择偏差的产物

关键难点：**$N$ 是多少？** 相关的试验不算独立，可以对试验收益做聚类，以簇数作为有效 $N$ 的估计。这就是为什么要**记录每一次尝试**（LEARNING_PLAN 通用验收标准第 2 条）。

```python
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER_GAMMA = 0.5772156649

def psr(sr, T, g3, g4, sr_star=0.0):
    return norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))

def expected_max_sr(sr_var, N):
    return np.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / N)
                              + EULER_GAMMA * norm.ppf(1 - 1 / (N * np.e)))

def deflated_sharpe(best_returns: np.ndarray, trial_srs: np.ndarray) -> float:
    """best_returns: 选中策略的日收益；trial_srs: 所有试验的日夏普（非年化）"""
    sr = best_returns.mean() / best_returns.std(ddof=1)
    sr0 = expected_max_sr(np.var(trial_srs, ddof=1), len(trial_srs))
    return psr(sr, len(best_returns), skew(best_returns),
               kurtosis(best_returns, fisher=False), sr0)

# 示例：200 次尝试，最好的策略年化夏普 1.5，3 年日度数据
# 若 trial_srs 的标准差对应年化 0.5 左右，DSR 往往远低于 0.95
```

### 8.4 PBO：回测过拟合概率（CSCV）

Bailey, Borwein, López de Prado & Zhu（2017）提出用**组合对称交叉验证（CSCV）**估计"样本内最优策略在样本外表现低于中位数"的概率：

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CSCV 计算 PBO                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入：T × N 矩阵 M（N 个策略配置在 T 期的收益）                       │
│                                                                      │
│  1. 把 T 行切成 S 个等长块（S 为偶数，如 16）                           │
│  2. 枚举所有 C(S, S/2) 种"选一半块作样本内"的组合：                     │
│       a. 样本内（IS）各策略夏普 → 选出最优策略 n*                        │
│       b. 样本外（OOS）各策略夏普 → n* 在 OOS 中的相对排名 ω ∈ (0,1)       │
│       c. λ = ln(ω / (1-ω))                                           │
│  3. PBO = P(λ ≤ 0) = 样本内最优在样本外跌到中位数以下的比例              │
│                                                                      │
│  PBO 接近 0：选择过程有效；PBO ≥ 0.5：选最优和随机选差不多甚至更差       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

```python
from itertools import combinations
from scipy.stats import rankdata

def sharpe_cols(R):
    return R.mean(axis=0) / R.std(axis=0, ddof=1)

def pbo_cscv(M: np.ndarray, S: int = 16) -> float:
    T, N = M.shape
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for is_ids in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in is_ids])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in is_ids])
        best = np.argmax(sharpe_cols(M[is_idx]))
        omega = rankdata(sharpe_cols(M[oos_idx]))[best] / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0))
```

$S=16$ 时有 $\binom{16}{8} = 12870$ 种组合，计算量可接受。

### 8.5 其他应对多重检验的方法

| 方法 | 说明 |
|---|---|
| 提高 t 统计量门槛 | Harvey, Liu & Zhu（2016）建议新因子的 t 值至少 3.0 而非 2.0 |
| Bonferroni / Holm / BHY | 经典多重检验校正（控制 FWER 或 FDR） |
| 保留最终测试集 | 只在研究完全结束后**使用一次** |
| 预注册假设 | 先写下假设和检验方法，再看数据 |

---

## 9. Qlib 完整工作流

### 9.1 Qlib 是什么

**Qlib** 是微软开源的 AI 量化研究平台（论文：*Qlib: An AI-oriented Quantitative Investment Platform*, 2020）。它把截面选股研究的全流程标准化：

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Qlib 工作流                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  数据层         ── 二进制存储的行情；表达式引擎 $close、Ref、Mean...    │
│     │                                                                │
│  DataHandler    ── Alpha158 / Alpha360 特征 + 标签 + 预处理器          │
│     │              （Processor：Fillna、CSZScoreNorm、DropnaLabel…）   │
│  Dataset        ── DatasetH / TSDatasetH，按 segments 切 train/valid/test│
│     │                                                                │
│  Model          ── LGBModel、XGBModel、各种 PyTorch 模型（Phase 10）    │
│     │                                                                │
│  Recorder       ── SignalRecord（预测）→ SigAnaRecord（IC 分析）        │
│     │              → PortAnaRecord（回测 + 风险分析）                   │
│  Strategy       ── TopkDropoutStrategy 等                              │
│     │                                                                │
│  Executor       ── 模拟撮合（涨跌停、成本、成交价）                      │
│                                                                      │
│  实验管理：基于 MLflow 的 R（Recorder）记录参数、模型、指标             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

> 对 LLM 应用工程师的类比：Qlib 之于量化研究，大致相当于 LangChain 之于 LLM 应用——提供标准化组件与可组合的配置，让你专注于"换模型、换特征"而不是重复写管道。

### 9.2 表达式引擎

Qlib 的特征用字符串表达式定义，例如：

| 表达式 | 含义 |
|---|---|
| `$close / Ref($close, 5) - 1` | 5 日收益 |
| `Mean($close, 20) / $close` | 20 日均线相对价格 |
| `Std($close / Ref($close, 1) - 1, 20)` | 20 日波动率 |
| `Corr($close, Log($volume + 1), 10)` | 10 日量价相关 |
| `Ref($close, -2) / Ref($close, -1) - 1` | **Alpha158 的默认标签**：$t+1$ 收盘买入、$t+2$ 收盘卖出的收益 |

注意 `Ref(x, -n)` 是**向未来**取值——只能出现在标签里，出现在特征里就是未来函数。Phase 10 的 LLM 因子挖掘会生成这类表达式，必须做校验。

**Alpha158**：158 个手工设计的价量特征（K 线形态、动量、波动、量价相关等）。**Alpha360**：过去 60 天的 6 个原始字段（开高低收、VWAP、成交量，按最新值归一化），共 360 维，适合深度模型自己学特征。

### 9.3 代码：用 Python 跑一个完整实验

以下结构与 Qlib 官方 `workflow_by_code` 示例一致（数据路径与日期需按你本地数据调整）：

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

market, benchmark = "csi300", "SH000300"
segments = {"train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test":  ("2017-01-01", "2020-08-01")}

task = {
    "model": {
        "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "learning_rate": 0.05, "num_leaves": 210,
                   "colsample_bytree": 0.8879, "subsample": 0.8789,
                   "lambda_l1": 205.6999, "lambda_l2": 580.9768,
                   "max_depth": 8, "num_threads": 8},
    },
    "dataset": {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": "2008-01-01", "end_time": "2020-08-01",
                           "fit_start_time": "2008-01-01", "fit_end_time": "2014-12-31",
                           "instruments": market},
            },
            "segments": segments,
        },
    },
}

model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

port_analysis_config = {
    "executor": {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                 "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
    "strategy": {"class": "TopkDropoutStrategy",
                 "module_path": "qlib.contrib.strategy.signal_strategy",
                 "kwargs": {"signal": (model, dataset), "topk": 50, "n_drop": 5}},
    "backtest": {"start_time": "2017-01-01", "end_time": "2020-08-01",
                 "account": 100_000_000, "benchmark": benchmark,
                 "exchange_kwargs": {"freq": "day", "limit_threshold": 0.095,
                                     "deal_price": "close", "open_cost": 0.0005,
                                     "close_cost": 0.0015, "min_cost": 5}},
}

with R.start(experiment_name="lgb_alpha158"):
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rec = R.get_recorder()
    SignalRecord(model, dataset, rec).generate()      # 保存测试期预测
    SigAnaRecord(rec).generate()                      # IC / RankIC / ICIR
    PortAnaRecord(rec, port_analysis_config, "day").generate()  # 回测与风险指标
```

同样的配置也可以写成 YAML，用命令行 `qrun config.yaml` 运行；官方 `examples/benchmarks/` 下有各模型的配置。

### 9.4 使用 Qlib 的注意事项

| 注意点 | 说明 |
|---|---|
| **数据质量** | 官方示例数据来自公开数据源，存在缺失和误差；Qlib 官方数据集曾因数据政策原因暂停提供，社区有替代数据源——以 Qlib README 的最新说明为准。生产研究应接入自己清洗过的数据（Phase 1） |
| **成交价假设** | 示例用 `deal_price="close"`。Qlib 的 `TopkDropoutStrategy` 在 $t+1$ 日交易时使用 $t$ 日的预测（信号自动滞后一期），按 $t+1$ 日收盘价成交，与 Alpha158 标签 $t+1 \to t+2$ 的口径一致。但用收盘价成交仍假设能在收盘集合竞价中全部成交，大资金或流动性差的股票应改用 VWAP 或加入冲击成本 |
| **成本参数** | 示例 `close_cost=0.0015` 是旧印花税时代的设定；按现行费率和你的冲击成本估计调整 |
| **涨跌停** | `limit_threshold=0.095` 适用于主板 10% 涨跌幅；创业板/科创板为 20%，北交所为 30%，主板 ST 在 2026-07-06 前为 5%、之后为 10%，需要更精细的处理 |
| **超参数** | 官方 benchmark 的超参数是在特定数据和时段调出来的，直接沿用就是一种隐性的数据窥探 |
| **对比基准** | 与 Phase 7 线性多因子在同一数据、同一成本、同一策略下比较 |

---

## 10. 模型漂移与再训练

### 10.1 漂移的类型

| 类型 | 定义 | 金融中的例子 |
|---|---|---|
| **协变量漂移（Covariate Shift）** | $P(X)$ 变了 | 市场整体换手率显著上升；注册制后新股结构变化 |
| **概念漂移（Concept Drift）** | $P(Y \mid X)$ 变了 | 小市值因子长期有效后突然失效；反转效应在某段时期变成动量 |
| **标签漂移** | $P(Y)$ 变了 | 牛熊市切换，收益分布整体变化 |

金融中最危险的是**概念漂移**：特征分布看起来正常，但"特征 → 收益"的关系已经变了。而且**alpha 衰减是常态**——因子被发现、被拥挤（Phase 7 的因子拥挤）后，有效性会下降。

### 10.2 监控什么

```
┌──────────────────────────────────────────────────────────────────────┐
│                          模型监控仪表板                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  预测效果（最重要，但有滞后：需等标签实现）                              │
│    · 滚动 20/60 日 RankIC、ICIR                                       │
│    · TopK 组合超额收益、回撤                                           │
│    · IC 累计曲线的斜率变化                                             │
│                                                                      │
│  输入分布（无滞后）                                                    │
│    · 原始特征（截面标准化之前）的 PSI                                   │
│    · 缺失率、异常值比例                                                │
│                                                                      │
│  输出分布（无滞后）                                                    │
│    · 预测值的截面离散度                                                │
│    · 持仓的风格暴露（市值、Beta）是否突变                               │
│    · 换手率                                                           │
│                                                                      │
│  模型内部                                                             │
│    · 特征重要性排序的变化（第 6 节）                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.3 PSI（Population Stability Index）

$$
\text{PSI} = \sum_{b=1}^{B}(a_b - e_b)\ln\frac{a_b}{e_b}
$$

- $e_b$：基准期（训练期）落在第 $b$ 个分箱的比例；$a_b$：当前期的比例；$B$：分箱数
- 常用经验阈值：< 0.1 稳定，0.1–0.25 轻度变化，> 0.25 显著变化（这是信用评分领域的经验值，在量化中仅作参考）

```python
import numpy as np
import pandas as pd

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def ic_alarm(daily_ic: pd.Series, window=60, threshold=0.0) -> pd.Series:
    """滚动均值 IC 跌破阈值时报警"""
    return daily_ic.rolling(window).mean() < threshold
```

> **陷阱**：对**已截面排名**的特征算 PSI 毫无意义——排名后分布永远是均匀的。PSI 应该算在原始特征上。

### 10.4 再训练策略

| 策略 | 做法 | 权衡 |
|---|---|---|
| **固定模型** | 训练一次，一直用 | 简单；无法适应漂移 |
| **滚动窗口** | 每月/季度用最近 N 年数据重训 | 适应变化；丢掉早期数据（可能包含罕见情景） |
| **扩展窗口** | 每次用全部历史重训 | 数据多；对新规律反应慢 |
| **样本加权** | 全部历史，近期样本权重更大 | 介于二者之间 |
| **触发式重训** | 监控指标报警时才重训 | 节省计算；报警阈值难定 |
| **模型集成** | 多个不同训练窗口的模型平均 | 稳健；复杂度高 |

Qlib 提供了滚动训练相关的工具（如 `RollingGen` 任务生成），以及针对概念漂移的研究型方法（如 DDG-DA）。

**重要原则**：再训练的**频率和窗口长度本身是超参数**，应在回测中按 Walk-Forward 方式模拟（每个时点只用当时可得的数据重训），并计入试验次数。

---

## 11. 最佳实践与常见陷阱

### 11.1 金融 ML 研究流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                       金融 ML 研究流程（推荐）                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 写下假设：为什么这个特征/模型应该有效？经济逻辑是什么？               │
│  2. 数据：PIT 对齐、包含退市股、复权正确、记录数据版本                    │
│  3. 标签：截面 rank / Triple-Barrier；记录每个样本的 t1                 │
│  4. 切分：训练 / 验证 / 测试按时间；最终测试集封存                       │
│  5. 验证：Purged K-Fold + Embargo（或 CPCV）调参                       │
│  6. 解释：MDA/SHAP 检查泄漏与经济合理性                                │
│  7. 组合：TopK 或优化器，含真实成本与交易约束                            │
│  8. 评估：IC → 分组 → 扣费组合 → 归因；与线性基线对比                    │
│  9. 去伪：DSR / PBO，披露尝试次数                                      │
│ 10. 上线：模拟盘 + 漂移监控 + 再训练计划                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 常见陷阱

| 陷阱 | 表现 | 对策 |
|---|---|---|
| 随机 K-Fold / 随机验证集 | CV 分数虚高，样本外崩塌 | 按时间切分 + Purge + Embargo |
| 同一天的股票分到训练和测试 | 截面相关泄漏 | 按日期分组切分 |
| 标准化用全样本统计量 | 微妙的前视 | 截面标准化或滚动统计量 |
| 财务数据按报告期对齐 | 未来信息泄漏 | 按公告日 PIT 对齐 |
| 早停用测试集 | 测试集被"用掉" | 早停只用验证集 |
| 只看 MSE/准确率 | 模型"好"但不赚钱 | IC、分组收益、扣费组合收益 |
| 不记录尝试次数 | 无法评估过拟合 | 研究日志；DSR |
| 特征高度共线 | 重要性失真 | 聚类特征重要性；去冗余 |
| 用 MDI 做特征筛选 | 噪声特征被保留 | 用样本外 MDA / SFI |
| 回测成交价用信号当日收盘 | 收益高估 | 次日开盘/VWAP 成交 |
| 忽略不可交易股票 | 模型最"准"的股票买不到 | 排序前剔除停牌/涨停/ST |
| 盲目照搬 benchmark 超参 | 隐性数据窥探 | 自己用 Purged CV 调参 |

---

## 12. 总结

### 12.1 核心概念回顾

```
┌──────────────────────────────────────────────────────────────────────┐
│                            核心概念回顾                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  金融 ML 的特殊性                                                      │
│  ├── 低信噪比：IC 0.05 就不错，模型容量越大越易拟合噪声                 │
│  ├── 非平稳：分布持续变化，alpha 会衰减                                 │
│  └── 非 IID：标签重叠、截面相关 → 普通 CV 与显著性检验失效              │
│                                                                      │
│  数据与标签                                                           │
│  ├── 截面标准化特征；PIT 基本面；分数阶差分平衡平稳与记忆                 │
│  ├── Triple-Barrier：止盈 / 止损 / 最长持有，边界随波动自适应            │
│  └── Meta-Labeling：主模型定方向，Meta 模型定是否下注与仓位              │
│                                                                      │
│  验证与模型                                                           │
│  ├── Purging：删除标签区间与测试重叠的训练样本                           │
│  ├── Embargo：测试块之后再隔离一段                                      │
│  ├── LightGBM 截面预测：rank 标签、强正则、按 RankIC 评估                │
│  └── 准确率提高 ≠ 收益提高                                              │
│                                                                      │
│  解释、组合与评估                                                      │
│  ├── MDI（样本内）/ MDA（样本外置换）/ SHAP（Shapley 分解）              │
│  ├── 分数 → α = IC·σ·z → 优化器；TopK-Dropout 控制换手                  │
│  ├── PSR / DSR：修正偏度峰度与多次试验                                  │
│  ├── PBO：CSCV 估计样本内最优在样本外失效的概率                          │
│  ├── Qlib：Handler → Dataset → Model → Record → Strategy → Executor    │
│  └── 漂移：监控 IC、PSI、暴露；再训练频率也是超参数                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 12.2 与 Phase 7、Phase 8 的关系

| 环节 | Phase 7（线性多因子） | Phase 9（ML） |
|---|---|---|
| 因子合成 | 等权 / IC 加权 / 最大化 IR | GBDT 自动学习非线性与交互 |
| 验证 | 分层回测、IC 分析 | + Purged CV、DSR、PBO |
| 可解释性 | 天然可解释 | 需要 SHAP / MDA |
| 组合构建 | Phase 8 优化器 | 同样接入 Phase 8 优化器 |
| 风险 | 线性暴露清晰 | 可能隐藏非线性风格暴露，需归因 |

### 12.3 与下一阶段的衔接

Phase 10 将把模型从 GBDT 扩展到深度学习、强化学习与 LLM。本阶段的方法论是 Phase 10 的**护栏**：

- **LSTM / Transformer**：在 Qlib 中以 Alpha360 + `TSDatasetH` 训练，用本阶段的 RankIC 与 Purged 验证评估；深度模型参数更多，低信噪比下更容易过拟合，DSR 更加重要
- **强化学习**：RL 交易环境的"避免前视"与本阶段 Purging 的思想同源——智能体在任何时刻都只能看到当时可得的信息
- **LLM 因子挖掘**：LLM 一次生成上百个因子表达式，等价于做了上百次试验——必须用第 8 节的 DSR/PBO 去伪，并用第 9.2 节的思路校验表达式中没有 `Ref(x, -n)` 这类未来函数
- **LLM 文本因子**：新闻情绪因子同样用截面 RankIC 检验，与本阶段的特征处理一致；此外还要面对 LLM 训练数据截止日期带来的新型前视偏差

> **一句话总结**：在金融 ML 中，模型的复杂度从来不是瓶颈，**验证方法的严谨性才是**。先学会不骗自己，再去追求更强的模型。
