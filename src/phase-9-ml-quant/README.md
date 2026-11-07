# Phase 9: 机器学习量化

> **学习目标**：掌握金融机器学习的正确方法论，避免常见的数据泄漏与过拟合
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 14 周：金融 ML 方法论

#### 1. 数据与标签

- `01-ml-quant-overview.py` - 金融 ML 的特殊性
- `02-feature-engineering.py` - 特征工程：价量、基本面、另类特征
- `03-labeling-triple-barrier.py` - Triple-Barrier 与 Meta-Labeling

#### 2. 验证与模型

- `04-purged-kfold-cv.py` - Purged K-Fold 与 Embargo
- `05-lightgbm-stock-prediction.py` - LightGBM 截面收益预测

### 第 15 周：从预测到交易

#### 3. 模型解释与组合

- `06-feature-importance-shap.py` - MDI、MDA、SHAP 特征重要性
- `07-from-prediction-to-portfolio.py` - 从预测分数到持仓

#### 4. 评估与工程

- `08-deflated-sharpe-pbo.py` - Deflated Sharpe Ratio 与 PBO
- `09-qlib-workflow.py` - Qlib 完整工作流
- `10-model-drift-monitoring.py` - 模型漂移与再训练

## 🎯 实战项目

**ML 选股模型**（[`projects/phase-09-ml-stock-selection/`](../../projects/README.md)）：基于 Qlib/LightGBM 的截面选股模型，Purged CV 验证，与 Phase 7 线性多因子对比。

## 📖 参考资料

- [《Advances in Financial Machine Learning》López de Prado](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [《Machine Learning for Algorithmic Trading》Stefan Jansen](https://github.com/stefan-jansen/machine-learning-for-trading)
- [Microsoft Qlib](https://github.com/microsoft/qlib)

## ✅ 学习检查

- [ ] 能解释普通 K-Fold 在金融数据上的泄漏问题
- [ ] 能实现 Triple-Barrier 标签
- [ ] 知道准确率提高不等于收益提高
- [ ] 能用 Deflated Sharpe 修正多次试验的影响
