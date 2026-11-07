# Phase 5: 择时与 CTA 策略

> **学习目标**：掌握经典趋势与均值回归策略，能在期货上构建多品种 CTA 组合
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 7 周：经典择时策略

#### 1. 趋势与反转

- `01-moving-average-crossover.py` - 双均线策略
- `02-bollinger-mean-reversion.py` - 布林带均值回归
- `03-macd-rsi-strategies.py` - MACD 与 RSI 策略
- `04-channel-breakout.py` - 通道突破（Donchian / Dual Thrust）
- `05-turtle-trading.py` - 海龟交易法则完整复现

### 第 8 周：期货 CTA

#### 2. 期货与仓位

- `06-futures-contracts-rollover.py` - 期货合约、保证金与展期
- `07-position-sizing-atr.py` - ATR 仓位与波动率目标

#### 3. 优化与组合

- `08-parameter-optimization.py` - 参数优化与参数平原
- `09-robustness-tests.py` - 稳健性检验：扰动、分段、跨品种
- `10-multi-asset-cta-portfolio.py` - 多品种 CTA 组合

## 🎯 实战项目

**期货 CTA 多品种组合**（[`projects/phase-05-cta-portfolio/`](../../projects/README.md)）：在 10+ 个期货品种上构建趋势跟踪组合，包含展期处理、波动率目标仓位与稳健性报告。

## 📖 参考资料

- [《海龟交易法则》](https://book.douban.com/subject/1858389/)
- [Moskowitz et al. Time Series Momentum (2012)](https://doi.org/10.1016/j.jfineco.2011.11.003)

## ✅ 学习检查

- [ ] 能解释趋势策略在震荡市失效的原因
- [ ] 能正确处理期货换月
- [ ] 参数选择基于参数平原而非单点最优
- [ ] 组合的波动率接近设定目标
