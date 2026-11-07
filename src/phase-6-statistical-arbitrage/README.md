# Phase 6: 统计套利

> **学习目标**：掌握配对交易与各类套利策略的原理、实现与风险
> **预计时长**：1 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

#### 1. 配对交易

- `01-pairs-trading-intro.py` - 配对交易原理与选对
- `02-cointegration-tests.py` - Engle-Granger 与 Johansen 检验
- `03-zscore-signals.py` - Z-score 信号与半衰期
- `04-kalman-filter-hedge.py` - 卡尔曼滤波动态对冲比

#### 2. 套利策略

- `05-calendar-spread.py` - 期货跨期套利
- `06-basis-arbitrage.py` - 期现套利与基差
- `07-crypto-funding-rate-arb.py` - 加密永续合约资金费率套利
- `08-stat-arb-risks.py` - 统计套利的风险：协整断裂与拥挤

## 🎯 实战项目

**配对交易系统**（[`projects/phase-06-pairs-trading/`](../../projects/README.md)）：在同行业股票或 ETF 中自动筛选协整对，用卡尔曼滤波对冲，并做滚动样本外回测。

## 📖 参考资料

- [Gatev et al. Pairs Trading (2006)](https://doi.org/10.1093/rfs/hhj020)
- [Ernest Chan《Algorithmic Trading》](https://www.wiley.com/en-us/Algorithmic+Trading-p-9781118460146)

## ✅ 学习检查

- [ ] 能用 ADF 检验价差平稳性
- [ ] 能计算均值回归半衰期
- [ ] 理解套利不等于无风险
- [ ] 能实现卡尔曼滤波对冲比
