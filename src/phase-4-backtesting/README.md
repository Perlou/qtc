# Phase 4: 回测系统原理

> **学习目标**：理解并亲手实现回测引擎，能识别回测中的各类陷阱
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 5 周：向量化回测

#### 1. 回测基础

- `01-backtest-overview.py` - 回测的作用、流程与局限
- `02-vectorized-backtest.py` - 向量化回测实现
- `03-performance-metrics.py` - 绩效指标：Sharpe、Sortino、MDD、Calmar、胜率

#### 2. 现实约束

- `04-transaction-costs-slippage.py` - 交易成本与滑点建模
- `05-a-share-constraints.py` - A 股约束：T+1、涨跌停、停牌、100 股

### 第 6 周：事件驱动与陷阱

#### 3. 事件驱动

- `06-event-driven-engine.py` - 从零实现事件驱动回测引擎

#### 4. 回测陷阱

- `07-lookahead-bias.py` - 前视偏差与未来函数
- `08-overfitting-traps.py` - 过拟合与多重检验
- `09-walk-forward.py` - Walk-Forward 与样本外检验

#### 5. 开源框架

- `10-backtrader-intro.py` - backtrader 入门
- `11-vectorbt-intro.py` - vectorbt 大规模参数回测

## 🎯 实战项目

**迷你回测引擎**（[`projects/phase-04-mini-backtester/`](../../projects/README.md)）：实现支持 A 股约束、成本模型、多标的的事件驱动回测引擎，与 backtrader 结果对拍。

## 📖 参考资料

- [backtrader 文档](https://www.backtrader.com/docu/)
- [vectorbt 文档](https://vectorbt.dev/)
- [《Advances in Financial Machine Learning》第 11-14 章](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

## ✅ 学习检查

- [ ] 能手写一个事件驱动回测循环
- [ ] 能找出一段代码中的未来函数
- [ ] 回测报告包含成本敏感性分析
- [ ] 会用 Walk-Forward 评估策略
