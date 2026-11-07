# Phase 12: 期权与衍生品

> **学习目标**：掌握期权定价、希腊字母与波动率交易基础
> **预计时长**：1 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

#### 1. 定价

- `01-options-basics.py` - 期权基础与收益结构
- `02-black-scholes.py` - Black-Scholes 定价与二叉树
- `03-greeks.py` - 希腊字母：Delta、Gamma、Vega、Theta

#### 2. 波动率

- `04-implied-volatility.py` - 隐含波动率求解
- `05-volatility-surface.py` - 波动率微笑与曲面
- `06-etf-options-data.py` - 50ETF / 300ETF 期权数据

#### 3. 交易

- `07-delta-hedging.py` - Delta 对冲模拟
- `08-volatility-strategies.py` - 波动率策略：跨式、价差、卖方策略风险

## 🎯 实战项目

**期权定价与波动率曲面工具**（[`projects/phase-12-options-toolkit/`](../../projects/README.md)）：拉取 ETF 期权链，计算隐含波动率与希腊字母，绘制波动率曲面，并模拟 Delta 对冲。

## 📖 参考资料

- [《期权、期货及其他衍生产品》John Hull](https://book.douban.com/subject/35003939/)
- [《Option Volatility and Pricing》Natenberg](https://www.mheducation.com/)

## ✅ 学习检查

- [ ] 能推导并实现 BS 公式
- [ ] 能用牛顿法求隐含波动率
- [ ] 理解 Gamma 与 Theta 的权衡
- [ ] 知道卖期权策略的尾部风险
