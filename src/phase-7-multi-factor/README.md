# Phase 7: 多因子模型

> **学习目标**：掌握 A 股多因子选股的完整研究流程
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 10 周：因子构建与检验

#### 1. 资产定价

- `01-capm-beta.py` - CAPM 与 Beta
- `02-fama-french.py` - Fama-French 三因子 / 五因子

#### 2. 因子研究

- `03-factor-construction.py` - 因子构建：价值、动量、质量、低波、规模
- `04-factor-preprocessing.py` - 去极值、标准化、行业市值中性化
- `05-ic-ir-analysis.py` - IC、Rank IC、IR 与 IC 衰减

### 第 11 周：因子组合与风险模型

#### 3. 因子评估

- `06-quantile-backtest.py` - 分层回测与多空组合
- `07-alphalens-tearsheet.py` - alphalens 因子报告
- `08-point-in-time-fundamentals.py` - 财报 Point-in-Time 对齐

#### 4. 组合与风险

- `09-factor-combination.py` - 因子合成：等权、IC 加权、最大化 IR
- `10-barra-risk-model.py` - Barra 风格因子与风险模型
- `11-factor-crowding-decay.py` - 因子拥挤与衰减

## 🎯 实战项目

**A 股多因子选股**（[`projects/phase-07-multi-factor-stock-selection/`](../../projects/README.md)）：构建 10+ 个因子，完成预处理、单因子检验、因子合成与月度调仓回测，输出研究报告。

## 📖 参考资料

- [《因子投资：方法与实践》石川等](https://book.douban.com/subject/35165432/)
- [alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded)
- [Fama-French 数据库](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## ✅ 学习检查

- [ ] 能独立完成单因子检验流程
- [ ] 理解为什么要做行业市值中性化
- [ ] 财报数据按公告日而非报告期对齐
- [ ] 能解释因子收益的来源（风险补偿或行为偏差）
