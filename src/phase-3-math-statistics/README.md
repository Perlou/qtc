# Phase 3: 量化数学与统计

> **学习目标**：用代码补齐量化所需的概率统计、回归与时间序列知识
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 3 周：概率与统计推断

#### 1. 概率基础

- `01-probability-distributions.py` - 常见分布与收益率分布
- `02-fat-tails-normality.py` - 肥尾、偏度、峰度与正态性检验
- `03-hypothesis-testing.py` - 假设检验：策略收益是否显著

#### 2. 模拟与决策

- `04-monte-carlo-simulation.py` - 蒙特卡洛模拟与 Bootstrap
- `05-kelly-criterion.py` - 期望值、胜率、盈亏比与凯利公式

### 第 4 周：回归与时间序列

#### 3. 回归与相关

- `06-linear-regression-ols.py` - OLS 回归与残差诊断
- `07-correlation-vs-cointegration.py` - 相关性与协整
- `08-linear-algebra-pca.py` - 协方差矩阵、特征分解与 PCA

#### 4. 时间序列模型

- `09-stationarity-adf.py` - 平稳性与 ADF 检验
- `10-arma-models.py` - 自相关与 ARMA 模型
- `11-garch-volatility.py` - 波动率聚集与 GARCH

## 🎯 实战项目

**风险收益统计报告生成器**（[`projects/phase-03-stats-report/`](../../projects/README.md)）：对任意资产生成统计报告：分布检验、尾部风险、自相关、GARCH 波动率预测与蒙特卡洛情景。

## 📖 参考资料

- [《统计学习导论》ISLR](https://www.statlearning.com/)
- [statsmodels 文档](https://www.statsmodels.org/)
- [arch 库文档](https://arch.readthedocs.io/)

## ✅ 学习检查

- [ ] 能解释为什么高胜率不等于正期望
- [ ] 能用 t 检验判断策略收益是否显著
- [ ] 能区分相关与协整
- [ ] 能用 GARCH 预测波动率
