# Phase 7: 多因子模型 - 核心概念深度解析

> **学习目标**：理解"因子"是什么、因子收益从哪里来，掌握 A 股多因子选股的完整研究流程——因子构建、预处理（去极值 / 标准化 / 中性化）、单因子检验（IC / Rank IC / IR / 分层回测）、财报 Point-in-Time 对齐、因子合成、Barra 风险模型，以及因子拥挤与衰减；并能正确处理 A 股特有的小市值、反转、ST、涨跌停、停牌等情况。

---

## 目录

1. [引言：从择时到选股](#1-引言从择时到选股)
2. [CAPM 与 Beta](#2-capm-与-beta)
3. [Fama-French 三因子 / 五因子与因子收益的来源](#3-fama-french-三因子--五因子与因子收益的来源)
4. [因子构建：价值、动量、质量、低波、规模](#4-因子构建价值动量质量低波规模)
5. [因子预处理：去极值、标准化、中性化](#5-因子预处理去极值标准化中性化)
6. [单因子检验：IC、Rank IC、IR 与 IC 衰减](#6-单因子检验icrank-icir-与-ic-衰减)
7. [分层回测、多空组合与 alphalens](#7-分层回测多空组合与-alphalens)
8. [财报 Point-in-Time 对齐](#8-财报-point-in-time-对齐)
9. [因子合成：等权、IC 加权、最大化 IR](#9-因子合成等权ic-加权最大化-ir)
10. [Barra 风格因子与风险模型](#10-barra-风格因子与风险模型)
11. [因子拥挤与衰减](#11-因子拥挤与衰减)
12. [A 股特有情况汇总](#12-a-股特有情况汇总)
13. [最佳实践与常见陷阱](#13-最佳实践与常见陷阱)
14. [总结](#14-总结)

---

## 1. 引言：从择时到选股

### 1.1 时间序列 vs 横截面

Phase 5 的 CTA 问的是"**这个品种**现在该做多还是做空"（时间序列）；本阶段的多因子模型问的是"**在同一时刻**，几千只股票中哪些会相对更好"（横截面，Cross-Section）。

| 维度 | 时间序列（择时 / CTA）| 横截面（多因子选股）|
| --- | --- | --- |
| 比较对象 | 同一资产的过去 vs 现在 | 同一时刻的不同资产 |
| 信号含义 | 绝对方向（涨 / 跌）| 相对排序（谁比谁好）|
| 典型收益来源 | 趋势、动量 | 风险溢价、定价偏差 |
| 样本量 | 少（几十个品种）| 多（A 股 5000+ 只股票）|
| 大盘的影响 | 是收益的主要来源 | 被对冲掉或作为基准 |
| 主要风险 | 市场方向判断错误 | 风格暴露、因子失效 |

### 1.2 什么是因子

**因子（Factor）** 是一个能够**解释或预测股票收益横截面差异**的特征。每个月给每只股票打一个分（比如市盈率的倒数），如果分数高的股票系统性地比分数低的股票涨得多，这个分数就是一个有效因子。

> 比喻：因子像**体检指标**。医生不会只看一个指标就判断健康状况，而是综合血压、血糖、胆固醇等多个指标打分。多因子模型就是给每只股票做"体检"：估值是否便宜（价值）、近期表现如何（动量）、公司赚钱能力怎样（质量）、波动是否剧烈（低波）、规模多大（市值）……然后综合打分，买入"最健康"的一批。

### 1.3 多因子研究流程

```
┌────────────────────────────────────────────────────────────────────┐
│                      多因子研究流水线                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  原始数据（行情 + 财报 PIT + 行业分类）                             │
│     │  08 财报按公告日对齐                                          │
│     ▼                                                              │
│  股票池过滤（剔除 ST、停牌、次新股、涨跌停不可交易）  ← 12 节        │
│     │                                                              │
│     ▼                                                              │
│  03 因子构建：EP、BP、动量、反转、ROE、波动率、市值……              │
│     │                                                              │
│     ▼                                                              │
│  04 预处理：去极值 → 补缺失 → 标准化 → 行业市值中性化               │
│     │                                                              │
│     ▼                                                              │
│  单因子检验：05 IC / IR / 衰减   06 分层回测   07 alphalens         │
│     │                                                              │
│     ▼                                                              │
│  09 因子合成：等权 / IC 加权 / 最大化 IR                            │
│     │                                                              │
│     ▼                                                              │
│  10 风险模型（Barra）→ 组合优化（Phase 8）→ 回测                    │
│     │                                                              │
│     ▼                                                              │
│  11 监控：因子拥挤、衰减                                            │
│                                                                    │
│  理论基础：01 CAPM → 02 Fama-French（因子从哪来、为什么有效）        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. CAPM 与 Beta

### 2.1 直觉（`01-capm-beta.py`）

**CAPM（Capital Asset Pricing Model，资本资产定价模型）** 由 Sharpe、Lintner 等人在 1960 年代提出，是"单因子模型"的鼻祖。它的核心思想是：

> **你只会因为承担"无法分散的风险"而获得补偿。**

个股的风险分两部分：
- **系统性风险**：跟着大盘一起波动的部分，无法通过持有更多股票消除
- **特质风险**：公司自身的事件（新产品、管理层变动），持有足够多股票就能分散掉

既然特质风险可以免费分散，市场就不会为它支付补偿。

> 比喻：你买的是一艘船上的座位。船整体的颠簸（系统性风险）谁都躲不掉，所以船票里包含了"颠簸补贴"；你座位旁边有没有人打呼噜（特质风险），换个座位就能解决，没人会为此给你钱。

### 2.2 公式

$$
E[R_i] - R_f = \beta_i \left(E[R_m] - R_f\right)
$$

- $E[R_i]$：股票 $i$ 的期望收益
- $R_f$：无风险利率
- $E[R_m] - R_f$：**市场风险溢价**（Market Risk Premium）
- $\beta_i$：股票 $i$ 对市场的敏感度

$$
\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}
$$

- $\beta = 1$：和大盘同步；$\beta = 1.5$：大盘涨 1%，平均涨 1.5%；$\beta = 0.5$：防御型

**时间序列回归**（实证检验 CAPM 的方式）：

$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \varepsilon_{i,t}
$$

- $\alpha_i$：**Alpha**，CAPM 无法解释的超额收益。CAPM 成立时 $\alpha_i = 0$
- $\varepsilon_{i,t}$：特质收益

### 2.3 代码

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def capm_regression(stock_ret: pd.Series, mkt_ret: pd.Series, rf_daily: pd.Series):
    """日收益率回归，返回年化 alpha、beta 和 alpha 的 t 值"""
    data = pd.concat([stock_ret - rf_daily, mkt_ret - rf_daily], axis=1,
                     keys=["ex_stock", "ex_mkt"]).dropna()
    res = sm.OLS(data["ex_stock"], sm.add_constant(data["ex_mkt"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})          # Newey-West 稳健标准误
    alpha_ann = res.params["const"] * 252
    return alpha_ann, res.params["ex_mkt"], res.tvalues["const"]

# 滚动 Beta：过去 250 个交易日
def rolling_beta(stock_ret, mkt_ret, window=250):
    cov = stock_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var
```

### 2.4 量化中的实际意义与陷阱

| 要点 | 说明 |
| --- | --- |
| **市场组合怎么选** | A 股常用沪深 300、中证全指或全市场市值加权指数。选择不同，Beta 和 Alpha 都会变 |
| **无风险利率怎么选** | 常用 1 年期国债收益率、Shibor 或定期存款利率折算为日频；选定后保持一致 |
| **Beta 不稳定** | 个股 Beta 随时间漂移，估计误差大；Barra 等风险模型会对 Beta 做收缩处理 |
| **低 Beta 异象** | 实证中低 Beta 股票的风险调整后收益往往**高于** CAPM 预测，高 Beta 股票则偏低（Frazzini & Pedersen 的 Betting Against Beta）。这说明 CAPM 并不完全成立，也是"低波因子"的理论来源之一 |
| **Alpha 的含义** | 一个策略的"Alpha"总是**相对于某个模型**而言。相对 CAPM 的 Alpha，可能只是相对 Fama-French 的 Beta——这就是下一节的内容 |

---

## 3. Fama-French 三因子 / 五因子与因子收益的来源

### 3.1 从单因子到多因子（`02-fama-french.py`）

实证研究发现，有些股票特征能稳定解释 CAPM 无法解释的收益差异：

- **规模效应**：小市值股票长期平均收益高于大市值（Banz, 1981）
- **价值效应**：账面市值比（BM）高的"便宜"股票长期跑赢 BM 低的"贵"股票

Fama & French（1993）把这两个效应做成因子，加入 CAPM，形成**三因子模型**：

$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_i\,\text{MKT}_t + s_i\,\text{SMB}_t + h_i\,\text{HML}_t + \varepsilon_{i,t}
$$

- $\text{MKT}$：市场超额收益 $R_m - R_f$
- $\text{SMB}$（Small Minus Big）：小市值组合收益 − 大市值组合收益
- $\text{HML}$（High Minus Low）：高 BM（价值股）组合收益 − 低 BM（成长股）组合收益
- $s_i, h_i$：股票 $i$ 对规模、价值因子的**暴露（载荷）**

**五因子模型**（Fama & French, 2015）再加入两个因子：

- $\text{RMW}$（Robust Minus Weak）：高盈利能力 − 低盈利能力
- $\text{CMA}$（Conservative Minus Aggressive）：低投资（资产扩张慢）− 高投资

另一个常见扩展是 Carhart（1997）加入的**动量因子** UMD（Up Minus Down）。

### 3.2 因子收益的构造：2×3 排序

以 SMB 和 HML 为例：每年按市值中位数分成小（S）、大（B）两组，按 BM 的 30% / 70% 分位数分成成长（G）、中性（N）、价值（V）三组，交叉得到 6 个组合，组内市值加权：

```
                 BM 低（G）   BM 中（N）   BM 高（V）
              ┌────────────┬────────────┬────────────┐
  市值小（S）  │     SG     │     SN     │     SV     │
              ├────────────┼────────────┼────────────┤
  市值大（B）  │     BG     │     BN     │     BV     │
              └────────────┴────────────┴────────────┘

  SMB = (SG + SN + SV)/3 − (BG + BN + BV)/3     → 控制住价值后的规模效应
  HML = (SV + BV)/2 − (SG + BG)/2               → 控制住规模后的价值效应
```

```python
def ff_factors_one_period(df: pd.DataFrame):
    """df：某个调仓日的股票截面，列 mcap（市值）、bm（账面市值比）、ret_next（下期收益）"""
    size = pd.Series(np.where(df["mcap"] <= df["mcap"].median(), "S", "B"), index=df.index)
    q30, q70 = df["bm"].quantile([0.3, 0.7])
    value = pd.Series(np.select([df["bm"] <= q30, df["bm"] >= q70], ["G", "V"], "N"),
                      index=df.index)
    group = size + value                                     # 如 "SV"、"BG"
    w = df["mcap"]
    port = (df["ret_next"] * w).groupby(group).sum() / w.groupby(group).sum()
    smb = port[["SG", "SN", "SV"]].mean() - port[["BG", "BN", "BV"]].mean()
    hml = port[["SV", "BV"]].mean() - port[["SG", "BG"]].mean()
    return smb, hml
```

**用因子模型评估策略**：把策略收益对 FF 因子回归，看 Alpha 是否显著。

```python
def factor_regression(strat_ex_ret: pd.Series, factors: pd.DataFrame):
    """factors 列如 ['MKT', 'SMB', 'HML', 'RMW', 'CMA']，与策略超额收益同频对齐"""
    data = pd.concat([strat_ex_ret.rename("y"), factors], axis=1).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data[factors.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6})
    return res.params, res.tvalues, res.rsquared

# 如果一个"选股策略"的收益几乎全被 SMB 解释（SMB 载荷大、Alpha 不显著），
# 说明它本质上只是在买小盘股，而不是什么新发现
```

### 3.3 A 股的 Fama-French：CH-3 模型

直接把美股方法搬到 A 股会遇到问题。Liu、Stambaugh、Yuan（2019）的 *Size and Value in China* 提出了针对 A 股的 **CH-3** 模型，两个关键修改：

1. **剔除市值最小的 30% 股票**：在 IPO 核准制时代，上市资格稀缺，小市值公司往往具有"**壳价值**"（被借壳上市的期权价值），其股价反映的是壳价值而非经营基本面
2. **用 EP（盈利市值比）替代 BM** 构造价值因子：在 A 股中 EP 对横截面收益的解释力更强

他们还提出了加入换手率相关情绪因子的 **CH-4** 模型，反映 A 股散户交易占比高的特点。

> 💡 **启示**：因子模型不是放之四海皆准的公式，而是对某个市场**制度和参与者结构**的刻画。A 股的注册制改革、退市制度改革都会改变壳价值，进而改变小市值因子的表现。

### 3.4 因子收益从哪里来：风险补偿 vs 行为偏差

这是本阶段的学习检查点之一。一个因子长期有正收益，只有两种解释：

| 解释 | 逻辑 | 推论 |
| --- | --- | --- |
| **风险补偿** | 因子暴露高的股票在"坏时候"表现更差，投资者要求更高回报作为补偿 | 收益应该**持续存在**；因子会在危机 / 经济下行时大幅回撤 |
| **行为偏差（错误定价）** | 投资者的系统性认知偏差导致定价错误，因子在利用别人的错误 | 收益会因套利资金进入而**衰减**；在难以套利的股票（小盘、流动性差、做空难）中更强 |

| 因子 | 风险补偿解释 | 行为偏差解释 |
| --- | --- | --- |
| **价值** | 便宜的公司往往陷入困境，有更高的破产 / 经营风险 | 投资者过度外推历史增长，把成长股炒得过贵、把价值股抛得过低 |
| **动量** | 动量股票在市场反转时崩溃（动量崩溃风险）| 投资者对新信息**反应不足**，价格缓慢调整 |
| **短期反转**（A 股显著）| 为急于交易者提供流动性的补偿 | 散户**过度反应**，短期涨多跌多后回调 |
| **规模** | 小公司流动性差、信息不透明；A 股还有壳价值 | 关注度低导致定价效率低 |
| **低波 / 低 Beta** | ——（与"高风险高收益"相反，难以用风险解释）| 杠杆约束使投资者偏好高 Beta；散户偏好"彩票型"股票，推高高波动股价格 |
| **质量 / 盈利** | 部分可用风险解释 | 投资者对盈利能力的持续性认识不足 |

> 实践中两种解释往往同时存在。理解来源的意义在于**预判因子在什么环境下会失效**：风险补偿型因子在危机中回撤，行为型因子在拥挤后衰减。

---

## 4. 因子构建：价值、动量、质量、低波、规模

### 4.1 常见因子一览（`03-factor-construction.py`）

| 大类 | 因子 | 计算 | 方向（A 股经验）|
| --- | --- | --- | --- |
| **价值** | EP | 净利润（TTM）/ 总市值 | 越高越好 |
| | BP | 净资产 / 总市值 | 越高越好 |
| | SP | 营业收入（TTM）/ 总市值 | 越高越好 |
| | 股息率 | 过去 12 个月分红 / 总市值 | 越高越好 |
| **动量 / 反转** | 中期动量 | 过去 12 个月收益，剔除最近 1 个月（12-1）| A 股中较弱、不稳定 |
| | 短期反转 | 过去 1 个月（约 20 个交易日）收益 | 越低越好（反转）|
| **质量** | ROE | 净利润（TTM）/ 净资产 | 越高越好 |
| | 毛利率 / 资产 | 毛利润 / 总资产 | 越高越好 |
| | 应计项目 | （净利润 − 经营现金流）/ 总资产 | 越低越好 |
| **低波** | 历史波动率 | 过去 N 日收益标准差 | 越低越好 |
| | 特质波动率 | FF 三因子回归残差的标准差 | 越低越好 |
| **规模** | 对数市值 | $\ln(\text{总市值或流通市值})$ | 历史上越小越好，但阶段性剧烈反转 |
| **流动性** | 换手率 | 过去 N 日平均换手率 | 越低越好（A 股显著）|

> **为什么用 EP 而不是 PE**：PE = 市值 / 利润，利润接近 0 时 PE 趋于无穷，利润为负时 PE 为负且排序含义颠倒。取倒数后 EP 是连续的，亏损公司自然排在最后。**因子构建时尽量让"分母不会接近零"**。

### 4.2 代码：从面板数据计算因子

数据结构约定：宽表 `DataFrame`，`index = 交易日`，`columns = 股票代码`。

```python
def build_factors(close: pd.DataFrame, mcap: pd.DataFrame, turnover: pd.DataFrame,
                  net_profit_ttm: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """所有输入都已对齐到交易日（财务数据已按公告日 PIT 对齐，见第 8 节）"""
    ret = close.pct_change()
    factors = {
        "ep": net_profit_ttm / mcap,
        "bp": equity / mcap,
        "roe": net_profit_ttm / equity.where(equity > 0),        # 净资产为负的不计算
        "mom_12_1": close.shift(21) / close.shift(252) - 1,       # 12-1 动量
        "rev_1m": -(close / close.shift(21) - 1),                 # 取负号：统一为"越大越好"
        "vol_60d": -ret.rolling(60).std(),                        # 低波：取负号
        "turn_20d": -turnover.rolling(20).mean(),                 # 低换手：取负号
        "size": -np.log(mcap),                                    # 小市值：取负号
    }
    return factors
```

**方向统一**：把所有因子调整为"值越大预期收益越高"，后续合成时就不用再记每个因子的方向。

> ⚠️ **方向要基于样本内检验和经济逻辑确定**，而且要意识到方向可能随时代改变（例如 A 股规模因子在 2017 年前后的表现截然不同）。

### 4.3 A 股因子的特殊性

1. **小市值效应长期显著但波动巨大**：A 股小市值因子在很长时间内收益可观，但经历过多次剧烈回撤，例如 2017 年大盘蓝筹显著跑赢小盘、2024 年初微盘股大幅回撤。小市值暴露是很多 A 股量化策略**隐藏的主要收益来源和主要风险来源**。
2. **短期反转强于动量**：美股中 12-1 动量是经典因子；A 股中月度级别的短期反转更显著，中期动量相对较弱且不稳定。一般认为这与 A 股散户交易占比高、过度反应较多有关。
3. **换手率与波动率因子有效**：低换手、低特质波动率的股票长期表现较好，与"彩票偏好"、投机交易有关。
4. **因子之间高度相关**：小市值股票往往同时是高换手、高波动、低价股。不做中性化，很多"新因子"只是小市值的马甲。

---

## 5. 因子预处理：去极值、标准化、中性化

### 5.1 为什么要预处理（`04-factor-preprocessing.py`）

原始因子值存在三个问题：

1. **极端值**：某只股票 EP = 5（利润异常大或市值极小），会主导回归、扭曲权重
2. **量纲不同**：EP 在 0.05 附近，市值在 $10^9$ 量级，无法直接加总
3. **混杂暴露**：EP 高的股票集中在银行、钢铁等行业，而且多为大市值——EP 因子的收益可能只是"买了银行股"

```
┌────────────────────────────────────────────────────────────────┐
│               因子预处理标准流程（每个截面独立执行）              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  原始因子                                                       │
│     │                                                          │
│     ▼                                                          │
│  ① 去极值（MAD / 分位数截断）    ── 防止极端值主导               │
│     │                                                          │
│     ▼                                                          │
│  ② 缺失值处理（行业中位数填充 / 剔除）                           │
│     │                                                          │
│     ▼                                                          │
│  ③ 标准化（Z-score / 排序）       ── 统一量纲                   │
│     │                                                          │
│     ▼                                                          │
│  ④ 中性化（对行业哑变量 + 对数市值回归，取残差）                 │
│     │                                                          │
│     ▼                                                          │
│  ⑤ 再次标准化                     ── 残差的尺度归一              │
│                                                                │
│  关键：每一步都在"同一天的截面"上做，绝不跨时间                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 去极值：MAD 法

**MAD（Median Absolute Deviation，绝对中位差）**：

$$
\text{MAD} = \text{median}\left(\,|x_i - \text{median}(x)|\,\right)
$$

截断边界：

$$
\left[\ \tilde{x} - n \cdot 1.4826 \cdot \text{MAD},\ \ \tilde{x} + n \cdot 1.4826 \cdot \text{MAD}\ \right]
$$

- $\tilde{x}$：截面中位数
- $1.4826$：常数，使正态分布下 $1.4826 \times \text{MAD}$ 等于标准差 $\sigma$
- $n$：通常取 3（即约 3 倍标准差）；也有实践直接用 $n \times \text{MAD}$ 不乘常数，各家约定不同

**为什么不用均值 ± 3σ**：均值和标准差本身就会被极端值拉偏，用"被污染的尺子"去量污染物不可靠。中位数和 MAD 对极端值稳健。

```python
def winsorize_mad(x: pd.Series, n: float = 3.0) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    bound = n * 1.4826 * mad
    return x.clip(med - bound, med + bound)    # 截断（winsorize），而不是删除
```

### 5.3 标准化

**Z-score 标准化**：

$$
z_i = \frac{x_i - \bar{x}}{\sigma_x}
$$

- $\bar{x}, \sigma_x$：当日截面的均值与标准差

**排序标准化**：把因子值换成截面排名再缩放到 $[0, 1]$ 或标准正态分位数。对分布形状完全不敏感，代价是丢失"差距大小"的信息。

```python
def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()

def rank_normalize(x: pd.Series) -> pd.Series:
    return x.rank(pct=True) - 0.5
```

> 也可以用市值加权均值做中心化（Barra 的做法），使"市值加权的基准组合"对该因子的暴露为 0。

### 5.4 行业市值中性化

这是本阶段的学习检查点之一：**为什么要做行业市值中性化？**

#### 直觉

假设你发现"低 PB 因子"在过去收益很好。仔细一看，低 PB 的股票 70% 都是银行和地产。那么你的"因子收益"到底来自"低估值"，还是来自"那几年银行地产涨得好"？如果是后者，这不是选股能力，而是**行业押注**——行业轮动有它自己的逻辑和风险，不应该混在估值因子里。

同理，很多因子与市值高度相关（换手率、波动率、价格水平……），不剔除市值影响，你的因子很可能只是"小市值因子的马甲"。

> 比喻：评价学生的学习方法好不好，要比较**同一个班、同样基础**的学生。如果直接拿重点班和普通班比，你测到的是"分班效应"，而不是"学习方法效应"。中性化就是"在同班、同基础的条件下比较"。

#### 公式

在每个截面上做回归：

$$
f_i = \sum_{k=1}^{K} \gamma_k D_{i,k} + \delta \ln(\text{MV}_i) + \varepsilon_i
$$

- $f_i$：股票 $i$ 标准化后的因子值
- $D_{i,k}$：行业哑变量，股票 $i$ 属于行业 $k$ 时为 1，否则为 0
- $\ln(\text{MV}_i)$：对数市值
- $\varepsilon_i$：**残差，即中性化后的因子值**——它与行业和市值都不相关

```python
def neutralize(factor: pd.Series, industry: pd.Series, log_mcap: pd.Series) -> pd.Series:
    """单个截面的行业市值中性化。三个输入以股票代码为 index"""
    df = pd.concat([factor.rename("f"), industry.rename("ind"),
                    log_mcap.rename("lnmv")], axis=1).dropna()
    X = pd.get_dummies(df["ind"], dtype=float)       # 全部行业哑变量（不加常数项，避免共线）
    X["lnmv"] = df["lnmv"]
    resid = sm.OLS(df["f"], X).fit().resid
    return resid.reindex(factor.index)

def preprocess_panel(raw: pd.DataFrame, industry: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    """raw/industry/mcap：index=日期, columns=股票。逐日处理"""
    out = {}
    for d in raw.index:
        x = raw.loc[d].dropna()
        if len(x) < 100:
            continue
        x = zscore(winsorize_mad(x))
        x = neutralize(x, industry.loc[d], np.log(mcap.loc[d]))
        out[d] = zscore(x.dropna())
    return pd.DataFrame(out).T
```

### 5.5 实际意义与陷阱

| 要点 | 说明 |
| --- | --- |
| **行业分类** | 常用申万行业、中信行业（一级约 30 个行业）。行业分类会调整，回测时要用**当时的**分类 |
| **中性化不是越多越好** | 如果你的 alpha 本来就来自"行业内估值差异 + 行业间估值差异"，完全行业中性会丢掉一部分收益。中性化的目的是**让收益来源清晰、可控**，是否中性取决于你想承担什么风险 |
| **市值中性化 ≠ 消除所有市值风险** | 线性回归只能去除线性关系；小市值效应可能是非线性的（Barra 中有"非线性市值"因子）|
| **顺序** | 先去极值再标准化再中性化。如果先中性化，极端值会扭曲回归系数 |
| **截面而非时序** | 预处理只能用当天截面数据，用全样本的均值 / 标准差会引入未来信息 |

---

## 6. 单因子检验：IC、Rank IC、IR 与 IC 衰减

### 6.1 IC：因子的"预测准确度"（`05-ic-ir-analysis.py`）

**IC（Information Coefficient，信息系数）**：在每个截面上，因子值与**下一期**收益的相关系数。

$$
\text{IC}_t = \text{corr}\left(f_{i,t},\ r_{i,t \to t+h}\right)
$$

- $f_{i,t}$：股票 $i$ 在 $t$ 时刻的因子值（只用 $t$ 时刻及之前的信息）
- $r_{i,t \to t+h}$：股票 $i$ 从 $t$ 到 $t+h$ 的收益
- 相关系数在**同一时刻的所有股票之间**计算，得到一个数；每期一个，形成 IC 时间序列

**Rank IC**：先把因子值和收益都换成截面排名，再算相关系数（即 Spearman 秩相关）。

| | IC（Pearson）| Rank IC（Spearman）|
| --- | --- | --- |
| 衡量 | 线性关系 | 单调关系 |
| 对极端值 | 敏感 | 稳健 |
| 实践中 | 较少单独使用 | **更常用** |

> 比喻：IC 像天气预报员的"准确率"。一个预报员每天的预测和实际天气的相关性只有 0.05，听起来很低，但如果他每天都稳定地比瞎猜好一点点，坚持一年（或在几千只股票上同时下注），积累起来的优势就很可观。

### 6.2 IR：IC 的稳定性

只看 IC 均值不够——一个因子可能某些月份 IC = 0.2，某些月份 IC = −0.15，均值看起来不错但极不稳定。

$$
\text{IR} = \frac{\overline{\text{IC}}}{\sigma(\text{IC})}
$$

- $\overline{\text{IC}}$：IC 时间序列的均值
- $\sigma(\text{IC})$：IC 时间序列的标准差
- 这里的 IR 也常写作 **ICIR**，本质是"IC 的夏普比率"

**IC 的显著性检验**：

$$
t = \frac{\overline{\text{IC}}}{\sigma(\text{IC}) / \sqrt{T}} = \text{IR} \times \sqrt{T}
$$

- $T$：IC 序列的期数。注意若 $h$ 大于调仓间隔（重叠持有期），IC 序列存在自相关，$t$ 值会被高估，需要用 Newey-West 调整或使用不重叠样本

**经验参考值**（A 股月频，仅供参考，不同股票池和时期差异很大）：

| 指标 | 一般 | 较好 | 很强（需警惕是否有偏差）|
| --- | --- | --- | --- |
| $\lvert$Rank IC 均值$\rvert$ | 0.02–0.03 | 0.03–0.05 | > 0.08 |
| ICIR（月频，未年化）| 0.3 | 0.5 | > 1.0 |
| IC > 0 的比例 | 55% | 60% | > 70% |

> ⚠️ 如果一个因子 Rank IC 均值达到 0.15，第一反应应该是**检查有没有未来函数**（财报没按公告日对齐、收益与因子时间错位、幸存者偏差等），而不是庆祝。

### 6.3 代码

```python
def calc_ic(factor: pd.DataFrame, fwd_ret: pd.DataFrame, method: str = "rank") -> pd.Series:
    """factor、fwd_ret：index=日期, columns=股票。fwd_ret 在 t 行存放 t→t+h 的收益"""
    mask = factor.notna() & fwd_ret.notna()           # 两者都有值的股票才参与
    f, r = factor.where(mask), fwd_ret.where(mask)
    if method == "rank":
        f, r = f.rank(axis=1), r.rank(axis=1)          # 先排名 → Spearman
    return f.corrwith(r, axis=1)

def ic_summary(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    return pd.Series({
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "ICIR": ic.mean() / ic.std(),
        "t值": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC>0占比": (ic > 0).mean(),
    })

# 前向收益：t 日收盘计算因子，t+1 日开盘买入更贴近现实
# fwd_ret = open_.shift(-1 - h) / open_.shift(-1) - 1
fwd_ret_20d = close.shift(-20) / close - 1             # 简化版：t 收盘到 t+20 收盘
```

### 6.4 IC 衰减

**IC 衰减（IC Decay）**：计算因子与未来不同时间跨度收益的 IC，观察预测能力随时间如何衰减。

```python
def ic_decay(factor, close, horizons=(1, 5, 10, 20, 40, 60)):
    out = {}
    for h in horizons:
        fwd = close.shift(-h) / close - 1
        out[h] = calc_ic(factor, fwd).mean()
    return pd.Series(out)
```

```
Rank IC
 0.06 │●
      │  ●                        短期反转因子：衰减快
 0.04 │    ●                      → 需要高频调仓，换手高、成本高
      │       ●
 0.02 │            ●
      │                  ●  ●
 0.00 └──────────────────────────▶ 持有期（天）
       1   5  10  20  40  60

 0.04 │ ●  ●  ●   ●    ●     ●    价值因子：衰减慢
      │                            → 月度调仓即可，换手低
 0.00 └──────────────────────────▶
```

**实际意义**：IC 衰减决定**调仓频率**。衰减快的因子要频繁调仓，但换手带来的成本可能吃掉全部收益；衰减慢的因子适合低频调仓，容量更大。

### 6.5 主动管理基本定律

Grinold 提出的**主动管理基本定律（Fundamental Law of Active Management）**：

$$
\text{IR}_{组合} \approx \text{IC} \times \sqrt{\text{BR}}
$$

- $\text{IR}_{组合}$：组合的信息比率（超额收益 / 跟踪误差）
- $\text{IC}$：预测能力
- $\text{BR}$（Breadth）：每年**独立**下注的次数

**含义**：IC 只有 0.03 的因子，在 3000 只股票上每月调仓（且假设下注相互独立），也能产生可观的 IR。这正是多因子选股"宽度换精度"的逻辑。但"独立"是一个很强的假设——同行业、同风格的股票并不独立，所以实际 BR 远小于"股票数 × 调仓次数"。

---

## 7. 分层回测、多空组合与 alphalens

### 7.1 分层回测（`06-quantile-backtest.py`）

IC 告诉你因子和收益"相关"，但不告诉你**能赚多少钱**、收益是否**单调**。分层回测更直观：

1. 每个调仓日，按因子值把股票等分为 $N$ 组（通常 5 或 10 组）
2. 每组等权（或市值加权）持有到下个调仓日
3. 比较各组的累计收益

```python
def quantile_backtest(factor: pd.DataFrame, period_ret: pd.DataFrame,
                      tradable: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    """factor：调仓日的因子值；period_ret：t 行为 t 到下一调仓日的收益；
    tradable：t 日是否可交易（剔除 ST、停牌、涨停买不进等，见第 12 节）"""
    rows = {}
    for d in factor.index:
        f = factor.loc[d].where(tradable.loc[d])
        r = period_ret.loc[d]
        ok = f.notna() & r.notna()
        if ok.sum() < n_groups * 20:
            continue
        g = pd.qcut(f[ok].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
        rows[d] = r[ok].groupby(g, observed=True).mean()
    grp = pd.DataFrame(rows).T
    grp["多空"] = grp[n_groups] - grp[1]
    return grp

grp = quantile_backtest(factor_m, ret_m, tradable_m)
nav = (1 + grp).cumprod()          # 各组净值曲线
```

**好因子的分层回测**：

```
累计收益
  │                                  ── 第 5 组（因子值最高）
  │                              ╱
  │                          ╱       ── 第 4 组
  │                      ╱  ╱
  │                  ╱  ╱  ╱         ── 第 3 组
  │              ╱  ╱  ╱  ╱
  │          ╱  ╱  ╱  ╱  ╱           ── 第 2 组
  │      ╱──╱──╱──╱──╱
  │  ───╱                            ── 第 1 组（因子值最低）
  └────────────────────────────────▶ 时间

  看三点：① 单调性：1 < 2 < 3 < 4 < 5
          ② 多空收益（5 − 1）稳定向上
          ③ 收益不只来自某一头（例如只有第 1 组特别差）
```

### 7.2 多空组合与 A 股的现实

**多空组合**（Top − Bottom）是检验因子的标准做法，但在 A 股中**空头腿通常无法实现**（融券受限）。实践中的替代：

| 实现方式 | 做法 | 注意 |
| --- | --- | --- |
| **纯多头超额** | 第 $N$ 组 − 基准指数（或全市场等权）| 衡量"选股能力"更贴近实际 |
| **指数增强** | 在沪深 300 / 中证 500 / 中证 1000 成分股内选股，控制跟踪误差 | 行业、风格偏离受限 |
| **市场中性** | 多头股票 + 做空股指期货 | 对冲成本受**基差**影响（Phase 6）；IC、IM 贴水时对冲有成本 |

> ⚠️ 如果因子的多空收益主要来自"空头组特别差"（例如第 1 组都是即将暴雷、ST 的股票），而多头组相对基准没有超额，那么这个因子在 A 股多头策略中**几乎没有用**。

**换手率**：每期组合中新换入股票的比例。高换手意味着高成本，A 股卖出时有印花税（税率以最新规定为准），单边总成本通常在千分之一到千分之几之间，月频调仓、换手 80% 时，一年成本就可能吃掉数个百分点的收益。

### 7.3 alphalens 因子报告（`07-alphalens-tearsheet.py`）

`alphalens`（社区维护版本为 `alphalens-reloaded`，导入名仍为 `alphalens`）把上述检验打包为标准报告。

```python
import alphalens as al

# factor：MultiIndex (date, asset) 的 Series
factor_series = factor_panel.stack()
factor_series.index.names = ["date", "asset"]

# prices：宽表，index=日期, columns=股票；日期需覆盖到最后一个因子日之后 max(periods) 天
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor_series, prices, quantiles=5, periods=(1, 5, 20), max_loss=0.35)

al.tears.create_full_tear_sheet(factor_data, long_short=True)
```

报告包含：分组收益、多空累计收益、IC 时间序列与分布、IC 衰减、换手率、因子自相关等。

> ⚠️ **alphalens 的 prices 用的是"因子日当天的价格"作为起点计算前向收益**。如果因子是用 $t$ 日收盘数据算的，又把 $t$ 日收盘价作为 prices 传进去，就隐含了"收盘时算完因子立即以收盘价成交"的假设。更稳妥的做法是传入 **$t+1$ 日开盘价**（把开盘价序列向前对齐一天），或者把因子日期整体后移一天。`max_loss` 控制因缺失数据丢弃样本的上限，如果报错说丢弃过多，先检查数据对齐而不是直接调大它。

---

## 8. 财报 Point-in-Time 对齐

### 8.1 报告期 vs 公告日（`08-point-in-time-fundamentals.py`）

这是本阶段的学习检查点之一，也是 A 股基本面因子研究中**最常见、最隐蔽**的未来函数来源。

- **报告期（Report Period / end_date）**：财报所描述的期间，例如 2023 年报的报告期是 2023-12-31
- **公告日（Announcement Date / ann_date）**：财报实际**公开披露**的日期，例如 2024-03-28

```
时间线：

 2023-12-31          2024-03-28           2024-04-30
     │                   │                    │
     ▼                   ▼                    ▼
  报告期结束          A 公司发布年报        年报披露截止日
  （数据还不存在）     （从此刻起才可用）

  ❌ 错误：在 2024-01-01 就使用 2023 年报数据（按报告期对齐）
     → 提前近 3 个月"知道"了全年利润 → 严重的前视偏差
  ✅ 正确：从公告日之后的第一个交易日起才使用
```

**A 股定期报告披露截止时间**（以证监会和交易所最新规定为准）：

| 报告 | 报告期截止 | 披露截止 |
| --- | --- | --- |
| 一季报 | 3 月 31 日 | 4 月 30 日 |
| 半年报 | 6 月 30 日 | 8 月 31 日 |
| 三季报 | 9 月 30 日 | 10 月 31 日 |
| 年报 | 12 月 31 日 | 次年 4 月 30 日 |

注意：**年报和一季报的截止日都是 4 月 30 日**，很多公司同一天发布两份报告。

### 8.2 PIT 对齐的三个细节

#### 细节 1：用 merge_asof 按公告日对齐

```python
def pit_align(daily: pd.DataFrame, fin: pd.DataFrame, cols: list) -> pd.DataFrame:
    """daily：列 [trade_date, stock]；fin：列 [stock, ann_date, end_date] + cols
    对每个交易日，取该日之前已公告的、报告期最新的一条财务数据"""
    fin = fin.sort_values(["ann_date", "end_date"])           # 同日公告时，报告期新的排后面
    daily = daily.sort_values("trade_date")
    merged = pd.merge_asof(
        daily, fin[["stock", "ann_date", "end_date"] + cols],
        left_on="trade_date", right_on="ann_date", by="stock",
        direction="backward",
        allow_exact_matches=False)     # 公告当日不用：盘后公告当天无法交易
    return merged
```

`allow_exact_matches=False` 的含义：公告日当天不使用，从下一个交易日起才使用。A 股公告常在盘后或晚间发布，这样处理更保守。

#### 细节 2：旧报告期的晚到数据

`merge_asof` 取的是"**公告日最近**"的一条，但这条不一定是"**报告期最新**"的。例如某公司 4 月 30 日发布了一季报，5 月 10 日又发布了对**去年年报**的更正公告——按公告日取最近一条，会拿到更正后的去年年报，覆盖了更新的一季报。

处理方法：对齐后，对每只股票按交易日检查，**只保留报告期不倒退的记录**（例如对 `end_date` 做累计最大值，并取该报告期对应的数据）。

#### 细节 3：财报更正与"原始版本"

上市公司可能事后更正财报。很多数据库默认提供**最新修订版**数据，如果回测中直接使用，相当于在过去就知道了后来才更正的数字。正确做法是使用数据库提供的"**首次披露版本 + 每次更正的公告日**"，严格按当时可见的版本回测。

### 8.3 累计值、单季度与 TTM

A 股利润表、现金流量表的数据是**年初至今累计值**（一季报是 1–3 月，半年报是 1–6 月，三季报是 1–9 月，年报是全年）。

**单季度值**：

$$
\text{Q3 单季} = \text{三季报累计} - \text{半年报累计}
$$

**TTM（Trailing Twelve Months，滚动 12 个月）**：

$$
\text{TTM} = \text{本期累计} + \text{上年年报} - \text{上年同期累计}
$$

例如在 2024 年一季报发布后：$\text{TTM} = \text{2024Q1 累计} + \text{2023 年报} - \text{2023Q1 累计}$。

```python
def ttm_from_cumulative(cum: pd.Series) -> pd.Series:
    """cum：单只股票按报告期排序的累计值，index 为报告期（Timestamp，季末日）"""
    out = {}
    for end, val in cum.items():
        if end.month == 12:
            out[end] = val
            continue
        last_annual = pd.Timestamp(year=end.year - 1, month=12, day=31)
        same_last_year = end - pd.DateOffset(years=1)
        if last_annual in cum.index and same_last_year in cum.index:
            out[end] = val + cum[last_annual] - cum[same_last_year]
    return pd.Series(out)
```

> 资产负债表科目（净资产、总资产）是**时点值**，不需要做 TTM，直接用最新一期。计算 ROE 时分子用 TTM 利润，分母常用期初期末平均净资产。

### 8.4 其他 PIT 问题

| 数据 | PIT 问题 |
| --- | --- |
| **行业分类** | 公司行业归属会调整，要用当时的分类 |
| **指数成分股** | 回测"中证 500 内选股"要用历史成分股，而不是今天的成分股 |
| **股票池** | 必须包含已退市股票（幸存者偏差）|
| **ST 状态** | 用当时的 ST 状态，而不是今天的 |
| **分析师预期** | 用预期发布时可见的数据 |
| **业绩预告 / 快报** | 早于正式财报披露，可以作为更及时的信息源，但同样按其公告日对齐 |

---

## 9. 因子合成：等权、IC 加权、最大化 IR

### 9.1 为什么要合成（`09-factor-combination.py`）

单个因子的 IC 通常只有 0.02–0.05，而且会阶段性失效。多个**低相关**因子合成后，IC 更稳定（与 Phase 5 第 9 节多品种分散同理）。

```
因子 A 的 IC：  ▲ ▲ ▼ ▲ ▼ ▼ ▲ ▲ ▲ ▼        单因子：时好时坏
因子 B 的 IC：  ▼ ▲ ▲ ▲ ▲ ▼ ▼ ▲ ▲ ▲
因子 C 的 IC：  ▲ ▼ ▲ ▼ ▲ ▲ ▲ ▼ ▲ ▲
──────────────────────────────────────
合成因子 IC：   ▲ ▲ ▲ ▲ ▲ ▼ ▲ ▲ ▲ ▲        低相关因子合成：更稳定
```

合成前提：每个因子都已经预处理（标准化 + 中性化）且方向统一。

### 9.2 三种合成方法

设有 $K$ 个因子，标准化后的因子值为 $z_{i,k}$，合成因子为：

$$
F_i = \sum_{k=1}^{K} w_k\, z_{i,k}
$$

| 方法 | 权重 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **等权** | $w_k = 1/K$ | 无需估计，最稳健 | 不区分好坏因子 |
| **IC 加权** | $w_k \propto \overline{\text{IC}}_k$（过去 $L$ 期）| 好因子权重大 | 忽略因子间相关性；IC 估计有噪声 |
| **ICIR 加权** | $w_k \propto \overline{\text{IC}}_k / \sigma(\text{IC}_k)$ | 兼顾稳定性 | 同样忽略相关性 |
| **最大化 IR** | $\mathbf{w} \propto \Sigma_{IC}^{-1}\, \boldsymbol{\mu}_{IC}$ | 理论最优，考虑相关性 | 协方差矩阵估计不稳定，易过拟合 |

**最大化 IR 的推导**：把每个因子的 IC 序列看作"资产收益"，求使合成因子 IR 最大的权重：

$$
\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu}_{IC}}{\sqrt{\mathbf{w}^T \Sigma_{IC}\, \mathbf{w}}} \implies \mathbf{w}^* \propto \Sigma_{IC}^{-1} \boldsymbol{\mu}_{IC}
$$

- $\boldsymbol{\mu}_{IC}$：各因子 IC 均值组成的向量
- $\Sigma_{IC}$：各因子 IC 序列的协方差矩阵

这与 Phase 8 的"最大夏普组合"形式完全一样。

### 9.3 代码

```python
def combine_weights(ic_hist: pd.DataFrame, method: str = "max_ir",
                    shrink: float = 0.5) -> pd.Series:
    """ic_hist：过去 L 期各因子的 IC（行=期，列=因子），只能包含当前时刻已知的 IC"""
    mu = ic_hist.mean()
    if method == "equal":
        w = pd.Series(1.0, index=ic_hist.columns)
    elif method == "ic":
        w = mu
    elif method == "icir":
        w = mu / ic_hist.std()
    else:  # max_ir：对协方差矩阵做收缩，降低估计噪声
        cov = ic_hist.cov().values
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        w = pd.Series(np.linalg.solve(cov, mu.values), index=ic_hist.columns)
    w = w.clip(lower=0)                      # 不允许负权重（可选，防止反向押注噪声）
    return w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)

def composite_factor(z_factors: dict, w: pd.Series) -> pd.Series:
    """z_factors：{因子名: 当期截面 Series}，均已标准化、方向统一"""
    return sum(w[k] * z_factors[k].fillna(0) for k in w.index)
```

> ⚠️ **IC 的可得时间**：$t$ 日的 IC 用到了 $t \to t+h$ 的收益，在 $t+h$ 日才知道。在 $t$ 日计算合成权重时，只能使用因子日期 $\le t-h$ 的 IC。如果直接用 `ic.loc[:t]`，就引入了未来信息。正确写法类似 `ic.loc[:t].iloc[:-h_periods]`（$h$ 换算成 IC 序列的期数），或者事先将 IC 序列 `shift(h_periods)`。

### 9.4 实际意义

- **等权往往很难被打败**：在样本外，复杂加权方法的优势经常被估计误差抵消。建议以等权作为基准，复杂方法必须在样本外显著更好才采用。
- **先看因子相关性**：两个因子相关系数 0.8，合成几乎没有分散效果；可以先合并为一个大类因子（如把 EP、BP、SP 合成"价值"大类），再在大类之间合成。
- **正交化**：可以用 Gram-Schmidt 或回归残差使因子相互正交，但正交顺序会影响结果，解释性变差。
- **机器学习**：Phase 9 将用 LightGBM 做非线性合成，并与本节的线性合成对比。

---

## 10. Barra 风格因子与风险模型

### 10.1 从 Alpha 模型到风险模型（`10-barra-risk-model.py`）

到目前为止的因子都用来**预测收益**（Alpha 因子）。但构建组合还需要知道**风险**：两只股票会不会一起跌？组合的波动有多少来自行业、多少来自市值？

直接估计 3000 只股票的协方差矩阵需要估计约 450 万个参数，而样本只有几年的日数据——完全不可行。**结构化风险模型**用少数共同因子来解释股票之间的相关性：

> 比喻：与其记住全班 50 个同学两两之间的关系（1225 对），不如记住"每个人属于哪个兴趣小组、性格内向还是外向"——同组、同性格的人自然关系更近。Barra 模型就是用"行业 + 风格"给每只股票画像。

### 10.2 模型结构

**收益分解**：

$$
r_i = \sum_{k} X_{i,k}\, f_k + u_i
$$

- $r_i$：股票 $i$ 的收益
- $X_{i,k}$：股票 $i$ 在因子 $k$ 上的**暴露**（行业哑变量、标准化的风格因子值），是已知的
- $f_k$：因子 $k$ 的**因子收益**，每期通过横截面回归估计
- $u_i$：特质收益

**横截面回归**：每个交易日，用当天所有股票的收益对暴露矩阵做**加权最小二乘**（WLS），权重通常为市值的平方根——大公司特质波动较小、估计更可靠，同时避免权重过度集中于超大市值。Barra 的完整模型还包含一个**国家因子**（所有股票暴露为 1），并加入"行业因子收益的市值加权和为 0"的约束以解决共线性。

**协方差矩阵**：

$$
V = X F X^T + \Delta
$$

- $V$：$N \times N$ 股票收益协方差矩阵
- $X$：$N \times K$ 因子暴露矩阵
- $F$：$K \times K$ 因子收益协方差矩阵（只需估计几十个因子之间的协方差）
- $\Delta$：对角矩阵，元素为各股票的特质方差

组合风险：

$$
\sigma_p^2 = \mathbf{h}^T V \mathbf{h} = \underbrace{\mathbf{h}^T X F X^T \mathbf{h}}_{\text{因子风险}} + \underbrace{\mathbf{h}^T \Delta\, \mathbf{h}}_{\text{特质风险}}
$$

- $\mathbf{h}$：组合中各股票的权重向量

### 10.3 Barra CNE5 风格因子

MSCI Barra 的中国股票模型 CNE5 包含 10 个风格因子（以及行业因子和国家因子）：

| 风格因子 | 含义 | 构成示例 |
| --- | --- | --- |
| **Size** | 规模 | 对数总市值 |
| **Beta** | 市场敏感度 | 对市场收益回归的 Beta |
| **Momentum** | 动量 | 剔除最近一段时间的长期相对强度 |
| **Residual Volatility** | 残差波动率 | 日收益波动、累计区间、残差波动 |
| **Non-linear Size** | 非线性规模 | 市值三次方对市值正交化后的残差（刻画中盘股）|
| **Book-to-Price** | 账面市值比 | BP |
| **Liquidity** | 流动性 | 月、季、年换手率 |
| **Earnings Yield** | 盈利收益率 | EP、预期 EP、现金流 EP |
| **Growth** | 成长 | 盈利与营收增长率 |
| **Leverage** | 杠杆 | 市场杠杆、账面杠杆、资产负债率 |

后续版本（如 CNE6）对因子做了扩充和细分。

### 10.4 代码：一个简化的横截面回归

```python
def barra_cross_section(ret: pd.Series, style: pd.DataFrame,
                        industry: pd.Series, mcap: pd.Series):
    """单日横截面回归（简化版：未加国家因子与行业约束）
    ret：当日股票收益；style：标准化后的风格因子暴露；industry：行业；mcap：市值"""
    X = pd.concat([pd.get_dummies(industry, dtype=float), style], axis=1)
    data = pd.concat([ret.rename("r"), X, np.sqrt(mcap).rename("w")], axis=1).dropna()
    res = sm.WLS(data["r"], data[X.columns], weights=data["w"]).fit()
    return res.params, res.resid        # 因子收益 f_k，特质收益 u_i

def portfolio_risk(h: pd.Series, X: pd.DataFrame, F: pd.DataFrame, spec_var: pd.Series):
    """h：组合权重；X：暴露矩阵（股票×因子）；F：因子协方差；spec_var：特质方差"""
    exposure = X.T @ h                                  # 组合在每个因子上的暴露
    factor_var = exposure @ F @ exposure
    specific_var = (h ** 2 * spec_var).sum()
    return np.sqrt(factor_var + specific_var), exposure
```

> 生产级 Barra 模型还包含大量工程细节：因子协方差的 EWMA 估计、Newey-West 自相关调整、特征因子（Eigenfactor）风险调整、波动率区间调整、特质风险的贝叶斯收缩等。理解结构即可，实践中常直接使用商业风险模型或开源实现。

### 10.5 风险模型的用途

| 用途 | 说明 |
| --- | --- |
| **风险预测** | 估计组合的波动率和跟踪误差 |
| **风险分解** | 组合风险中多少来自行业、多少来自 Size、多少来自特质 |
| **风格控制** | 在组合优化中约束风格暴露（例如 Size 暴露在 ±0.3 以内），防止策略变成"小盘股基金" |
| **因子纯化** | 检验一个 Alpha 因子相对 Barra 风格是否有增量信息（对风格因子回归后看残差的 IC）|
| **业绩归因** | 把收益分解为因子收益贡献和特质收益贡献（Phase 8 详解）|

---

## 11. 因子拥挤与衰减

### 11.1 因子衰减（`11-factor-crowding-decay.py`）

McLean & Pontiff（2016）研究了近百个学术文献中发表的异象，发现：因子在样本外的收益比样本内平均低约 26%，**论文发表后**平均低约 58%。

两个原因：

1. **数据挖掘**：部分因子本身就是过拟合的产物，样本外自然失效。Harvey、Liu、Zhu（2016）认为考虑到已测试的因子数量，新因子的 t 值门槛应提高到 3.0 左右而不是传统的 2.0
2. **套利**：因子被广泛知晓后，资金涌入，定价偏差被交易掉

### 11.2 因子拥挤

**拥挤（Crowding）**：太多资金在做同样的因子交易。后果：

- 预期收益被压缩（买的人多了，价格提前反映）
- 回撤时更剧烈（同时平仓 → 踩踏，回顾 Phase 6 的 2007 Quant Quake）

**常用拥挤度指标**：

| 指标 | 计算 | 拥挤信号 |
| --- | --- | --- |
| **估值价差** | 多头组合的估值（如 PB 中位数）/ 空头组合的估值 | 对价值以外的因子，多头相对空头**越来越贵** |
| **配对相关性** | 多头组合内部股票收益的平均相关性 | 相关性上升，说明有资金在同步买卖 |
| **换手率比** | 多头组合换手率 / 空头组合换手率 | 多头交易异常活跃 |
| **因子波动率** | 因子收益的滚动波动率 | 波动率显著上升 |
| **因子收益自相关 / 动量** | 因子近期累计收益 | 短期涨幅过大，可能透支 |
| **融券余额 / 空头仓位** | 空头组合的做空比例 | 空头端拥挤 |

```python
def valuation_spread(factor: pd.Series, bp: pd.Series, q: float = 0.2) -> float:
    """单期：因子最高组与最低组的 BP 中位数之比（>1 表示多头更便宜）"""
    hi = factor >= factor.quantile(1 - q)
    lo = factor <= factor.quantile(q)
    return bp[hi].median() / bp[lo].median()

def rolling_ic_monitor(ic: pd.Series, window: int = 12) -> pd.DataFrame:
    """滚动 IC 均值与 ICIR：持续下降是衰减信号"""
    return pd.DataFrame({
        "rolling_ic": ic.rolling(window).mean(),
        "rolling_icir": ic.rolling(window).mean() / ic.rolling(window).std(),
    })

# 时间序列上看：估值价差处于历史极端分位、因子内部相关性上升、
# 滚动 ICIR 下降 —— 多个信号同时出现时，应考虑降低该因子权重
```

### 11.3 A 股的拥挤案例

- **2017 年小市值因子大幅回撤**：IPO 发行提速削弱了壳价值、监管趋严、资金偏好大盘蓝筹，长期依赖小市值暴露的策略遭受重创
- **2024 年初微盘股大幅回撤**：大量量化产品（包括部分 DMA 等杠杆产品）在小微盘股上暴露过度集中，市场流动性冲击下出现集中踩踏，量化指数增强和中性产品普遍回撤
- **共同教训**：在 A 股，"多因子"策略最常见的隐藏风险是**未加控制的小市值暴露**。即使每个因子都做了市值中性化，选股后的组合仍可能在市值上严重偏离基准——这就是为什么需要第 10 节的风险模型和 Phase 8 的约束优化

### 11.4 应对衰减与拥挤

| 做法 | 说明 |
| --- | --- |
| **持续监控** | 每月跟踪滚动 IC、ICIR、拥挤度指标 |
| **因子分散** | 不依赖少数热门因子；持续研究新数据源（Phase 10 的文本、另类数据）|
| **风格约束** | 用风险模型控制 Size、Beta 等风格暴露 |
| **容量评估** | 估算因子能容纳多少资金而不显著压缩收益 |
| **压力测试** | 模拟"某因子单月回撤 3 倍标准差"时组合的损失 |
| **区分来源** | 风险补偿型因子的回撤可能是"正常的坏时候"，行为型因子的持续低迷可能是"永久衰减" |

---

## 12. A 股特有情况汇总

### 12.1 交易制度对回测的影响

| 制度 | 内容（以交易所最新规则为准）| 回测处理 |
| --- | --- | --- |
| **T+1** | 当日买入的股票次日才能卖出 | 日内反转类信号无法实现 |
| **涨跌停** | 主板 ±10%；创业板、科创板 ±20%；北交所 ±30%；主板 ST 股票 2026-07-06 前为 ±5%，之后调整为 ±10%（回测须按日期使用当时规则）| 涨停（尤其一字涨停）买不进，跌停卖不出 |
| **新股** | 注册制下新股上市初期若干交易日不设涨跌幅限制 | 剔除上市不足一定时间（如 6–12 个月）的次新股 |
| **停牌** | 停牌期间无法交易，复牌时可能一次性补涨补跌 | 停牌股不能买入；已持有的停牌股无法卖出，按停牌前价格估值，复牌后计入收益 |
| **ST / \*ST** | 财务异常或其他风险警示，有退市风险 | 通常整体剔除 |
| **退市** | 退市制度趋严（如 2024 年新"国九条"发布后）| 股票池必须包含已退市股票；持有到退市的股票按退市价格计损失 |
| **做空** | 融券受限 | 以多头超额 / 期货对冲实现 |
| **交易成本** | 佣金、印花税（卖出单边）、过户费 | 按当时的税费标准计入，标准调整过多次 |

### 12.2 可交易性过滤

```python
def tradable_mask(is_st: pd.DataFrame, is_suspended: pd.DataFrame,
                  list_days: pd.DataFrame, limit_up_next_open: pd.DataFrame,
                  min_list_days: int = 250) -> pd.DataFrame:
    """所有输入：index=调仓日, columns=股票，布尔值或天数
    limit_up_next_open：调仓日次日开盘即涨停（买不进）"""
    return (~is_st) & (~is_suspended) & (list_days >= min_list_days) & (~limit_up_next_open)
```

> 💡 `limit_up_next_open` 用到了 $t+1$ 日的开盘价，这不是前视偏差——它模拟的是**执行层面的成交失败**（你在 $t$ 日决定买入，$t+1$ 日开盘时发现涨停买不进），与信号计算无关。但要确保这个信息**只用于限制成交**，不能反过来影响因子值或选股排序。同理，持仓股票在调仓日跌停，应当模拟为"卖不出、继续持有"。

### 12.3 A 股常见效应与处理建议

| 效应 | 特点 | 研究建议 |
| --- | --- | --- |
| **小市值效应** | 长期显著，但与壳价值、流动性、制度改革密切相关，阶段性剧烈回撤 | 所有因子都要检查与市值的相关性；报告中披露组合的市值暴露；用 CH-3 思路剔除最小市值股票做对比 |
| **短期反转** | 月度反转显著，强于中期动量 | 反转因子换手高，必须严格计成本；注意涨跌停导致的"假反转"（涨停后次日买不进）|
| **换手率 / 波动率** | 低换手、低特质波动长期占优 | 与小市值高度相关，必须做中性化后再评估 |
| **ST 效应** | ST 股票波动和事件风险巨大 | 默认剔除；若研究"摘帽"等事件需单独处理 |
| **涨跌停** | 封板影响成交、影响收益计算 | 区分"能否成交"与"收益多少"，分别建模 |
| **停牌** | 历史上曾出现大量长期停牌（如 2015 年股市异常波动期间）| 停牌股票的收益不能用"停牌期间为 0、复牌后不计"来简化 |

---

## 13. 最佳实践与常见陷阱

### 13.1 最佳实践

```
┌─────────────────────────────────────────────────────────────────┐
│                     多因子研究最佳实践                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 数据                                                        │
│     □ 财报按公告日对齐（merge_asof + 公告当日不用）              │
│     □ 使用首次披露版本；处理旧报告期晚到数据                     │
│     □ 股票池含退市股；行业、成分股、ST 状态都用历史值            │
│                                                                 │
│  🔧 因子                                                        │
│     □ 方向统一为"越大越好"；分母避免接近零                       │
│     □ 去极值 → 标准化 → 行业市值中性化，全部按截面执行           │
│     □ 检查新因子与已有因子、与市值的相关性                       │
│                                                                 │
│  🔬 检验                                                        │
│     □ Rank IC、ICIR、t 值、IC 衰减、分层单调性、多头超额          │
│     □ 计入换手和交易成本；剔除不可交易股票                       │
│     □ 分年度、分市场状态看 IC 稳定性                             │
│                                                                 │
│  ⚖️ 组合                                                        │
│     □ 以等权合成为基准；复杂方法须样本外胜出                     │
│     □ 合成权重只用当时已知的 IC                                  │
│     □ 用风险模型监控 Size、Beta 等风格暴露                       │
│                                                                 │
│  📈 监控                                                        │
│     □ 滚动 IC、拥挤度指标、风格暴露                              │
│     □ 记录所有测试过的因子数量（多重检验）                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 13.2 常见陷阱与解决方案

```python
# ❌ 陷阱 1：按报告期对齐财报
fin_daily = fin.set_index("end_date").reindex(trade_dates, method="ffill")
# ✅ 改进：按公告日 merge_asof，公告当天不用
merged = pd.merge_asof(daily, fin, left_on="trade_date", right_on="ann_date",
                       by="stock", allow_exact_matches=False)

# ❌ 陷阱 2：全样本标准化（跨时间）
z = (factor - factor.stack().mean()) / factor.stack().std()
# ✅ 改进：每个截面单独标准化
z = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

# ❌ 陷阱 3：因子与收益错位（用 t 日收益评价 t 日因子）
ic = factor.corrwith(close.pct_change(), axis=1)        # 同期收益 → 未来函数
# ✅ 改进：用未来收益
ic = factor.corrwith(close.shift(-20) / close - 1, axis=1)

# ❌ 陷阱 4：用今天的股票列表回测过去
universe = get_current_stock_list()                     # 幸存者偏差
# ✅ 改进：每个调仓日使用当时上市、未退市的股票

# ❌ 陷阱 5：合成权重使用了尚未实现的 IC
w = combine_weights(ic.loc[:t])
# ✅ 改进：剔除最近 h 期尚未实现的 IC
w = combine_weights(ic.loc[:t].iloc[:-h_periods])
```

| 陷阱 | 后果 | 解决方案 |
| --- | --- | --- |
| 不做中性化就宣称发现新因子 | 新因子只是小市值 / 行业的马甲 | 中性化后检验；对 Barra 风格回归看增量 |
| 只看多空收益 | 收益可能来自无法做空的空头腿 | 同时报告多头相对基准的超额 |
| 忽略涨跌停与停牌 | 回测买入了实际买不进的涨停股 | 可交易性过滤 + 卖不出时继续持有 |
| 测试上百个因子只报告最好的 | 多重检验导致虚假发现 | 记录测试数量，提高 t 值门槛，样本外验证 |
| 因子 IC 高得离谱 | 往往是未来函数 | 先排查数据对齐，再相信结果 |
| 合成后组合市值暴露失控 | 风格切换时大幅回撤 | 风险模型 + 约束优化（Phase 8）|
| 用 TTM 时直接把累计值相加 | 重复计算 | 按"本期累计 + 上年年报 − 上年同期累计"计算 |

---

## 14. 总结

### 14.1 核心概念回顾

```
┌─────────────────────────────────────────────────────────────────────┐
│                          核心概念回顾                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   资产定价                                                          │
│   ├── CAPM：E[R] − Rf = β(E[Rm] − Rf)，只补偿系统性风险              │
│   ├── FF3：MKT + SMB + HML；FF5：+ RMW + CMA                         │
│   ├── A 股 CH-3：剔除最小 30% 市值（壳价值），用 EP 构造价值          │
│   └── 因子收益来源：风险补偿（危机回撤）vs 行为偏差（拥挤衰减）       │
│                                                                     │
│   因子研究流程                                                       │
│   ├── 构建：价值、动量/反转、质量、低波、规模、换手；方向统一          │
│   ├── 预处理：MAD 去极值 → Z-score → 行业市值中性化（取残差）         │
│   ├── 检验：IC / Rank IC / ICIR / t 值 / IC 衰减                     │
│   ├── 分层回测：单调性、多空、多头超额、换手                           │
│   └── PIT：按公告日对齐，累计值 → 单季 / TTM，使用原始披露版本         │
│                                                                     │
│   组合与风险                                                         │
│   ├── 合成：等权（基准）/ IC 加权 / ICIR 加权 / 最大化 IR（Σ⁻¹μ）     │
│   ├── 基本定律：IR ≈ IC × sqrt(Breadth)                               │
│   ├── Barra：r = Xf + u；V = XFXᵀ + Δ；CNE5 十大风格因子               │
│   └── 拥挤与衰减：发表后收益下降；估值价差、相关性、换手监控            │
│                                                                     │
│   A 股特有                                                          │
│   └── 小市值、短期反转、ST、涨跌停、停牌、新股、退市、做空受限          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 与下一阶段的衔接

```
┌─────────────────────────────────────────────────────────────────────┐
│             Phase 7  ──▶  Phase 8 组合优化与风险管理                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   本阶段的输出是"每只股票的合成因子得分"和"风险模型"；               │
│   但"得分最高的 50 只等权买入"并不是最优组合：                       │
│                                                                     │
│   ├── 如何把得分转化为预期收益？           → 均值方差 / Black-Litterman│
│   ├── 如何用 Barra 协方差 V = XFXᵀ + Δ？   → 约束优化（cvxpy）       │
│   ├── 如何控制风格暴露、行业偏离、换手？   → 带约束的组合优化         │
│   ├── 如何估计更稳健的协方差？             → Ledoit-Wolf 收缩         │
│   └── 收益来自哪里？                       → Brinson 与因子归因       │
│                                                                     │
│   Phase 9 还会用 LightGBM 替代线性因子合成，并与本阶段结果对比。      │
│                                                                     │
│   需要带过去的工具：                                                 │
│   □ 预处理与中性化流水线    □ IC 检验框架与分层回测                  │
│   □ PIT 财务数据表          □ 风险模型暴露矩阵与因子协方差            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**推荐阅读**：

- 石川、刘洋溢、连祥斌《因子投资：方法与实践》——中文因子投资的系统性参考
- Fama & French. *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics, 1993
- Fama & French. *A Five-Factor Asset Pricing Model*. Journal of Financial Economics, 2015
- Liu, Stambaugh, Yuan. *Size and Value in China*. Journal of Financial Economics, 2019
- McLean & Pontiff. *Does Academic Research Destroy Stock Return Predictability?* Journal of Finance, 2016
- Grinold & Kahn《Active Portfolio Management》——主动管理基本定律与风险模型
- alphalens-reloaded 文档：<https://github.com/stefan-jansen/alphalens-reloaded>

---

**文档版本**: 1.0
**适用课程**: 量化交易课程 Phase 7（第 10–11 周）
