# Phase 11: 市场微观结构与执行

> **学习目标**：理解订单簿与交易成本的来源，掌握执行算法与做市基础
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 18 周：微观结构

#### 1. 订单簿

- `01-order-book-basics.py` - 订单簿、撮合规则与订单类型
- `02-tick-level2-data.py` - Tick 与 Level-2 数据处理
- `03-spread-and-impact.py` - 买卖价差与市场冲击
- `04-realtime-orderbook-ccxt.py` - ccxt 实时订单簿
- `05-high-frequency-factors.py` - 高频因子：订单不平衡、成交量分布

### 第 19 周：执行与做市

#### 2. 执行算法

- `06-twap-vwap.py` - TWAP / VWAP / POV
- `07-almgren-chriss.py` - Almgren-Chriss 最优执行

#### 3. 做市

- `08-market-making-basics.py` - 做市原理与库存风险
- `09-avellaneda-stoikov.py` - Avellaneda-Stoikov 做市模型
- `10-hft-infrastructure.py` - 高频交易基础设施与延迟

## 🎯 实战项目

**加密订单簿做市模拟器**（[`projects/phase-11-market-making-sim/`](../../projects/README.md)）：基于真实订单簿快照回放，实现 AS 做市策略与库存控制，评估 PnL 与风险。

## 📖 参考资料

- [《Algorithmic and High-Frequency Trading》Cartea et al.](https://www.cambridge.org/9781107091146)
- [《Trading and Exchanges》Larry Harris](https://global.oup.com/academic/product/trading-and-exchanges-9780195144703)

## ✅ 学习检查

- [ ] 能解释价差的三个组成部分
- [ ] 能实现 VWAP 执行算法
- [ ] 理解做市商的库存风险与逆向选择
- [ ] 能从订单簿计算订单不平衡因子
