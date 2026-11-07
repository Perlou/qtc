# Phase 8: 组合优化与风险管理

> **学习目标**：掌握主流组合优化方法和风险度量、归因技术
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 12 周：组合优化

#### 1. 经典方法

- `01-mean-variance.py` - Markowitz 均值方差与有效前沿
- `02-covariance-shrinkage.py` - 协方差估计与 Ledoit-Wolf 收缩
- `03-risk-parity.py` - 风险平价
- `04-black-litterman.py` - Black-Litterman 模型
- `05-hrp.py` - 层次风险平价 HRP

### 第 13 周：风险管理

#### 2. 约束优化

- `06-constrained-optimization-cvxpy.py` - cvxpy 带约束组合优化

#### 3. 风险度量与归因

- `07-var-cvar.py` - VaR 与 CVaR
- `08-stress-testing.py` - 压力测试与情景分析
- `09-performance-attribution.py` - Brinson 与因子归因
- `10-rebalancing-turnover.py` - 再平衡与换手控制

## 🎯 实战项目

**组合优化与风控模块**（[`projects/phase-08-portfolio-optimizer/`](../../projects/README.md)）：实现可插拔的组合优化器（MV/RP/BL/HRP）与风险报告，接入 Phase 7 的因子信号。

## 📖 参考资料

- [riskfolio-lib](https://riskfolio-lib.readthedocs.io/)
- [cvxpy](https://www.cvxpy.org/)
- [《Active Portfolio Management》Grinold & Kahn](https://www.mheducation.com/)

## ✅ 学习检查

- [ ] 理解均值方差对输入误差敏感的原因
- [ ] 能用 cvxpy 写出带行业、换手约束的优化
- [ ] 能计算历史法和参数法 VaR
- [ ] 能把组合收益拆解为因子贡献与特异贡献
