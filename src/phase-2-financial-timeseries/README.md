# Phase 2: 金融时间序列与 pandas

> **学习目标**：熟练用 pandas/NumPy 处理金融时间序列，掌握收益、波动、回撤等基本度量
> **预计时长**：1 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

#### 1. 收益与波动

- `01-returns-simple-vs-log.py` - 简单收益与对数收益
- `02-annualization-volatility.py` - 年化收益与年化波动率
- `03-rolling-windows.py` - 滚动窗口统计

#### 2. 数据操作

- `04-resample-multi-frequency.py` - 重采样与多频率数据
- `05-multi-asset-panel.py` - 多资产面板数据（宽表与长表）
- `06-technical-indicators.py` - 技术指标：MA、EMA、MACD、RSI、ATR、布林带

#### 3. 分析与可视化

- `07-drawdown-analysis.py` - 回撤与水下曲线
- `08-benchmark-relative.py` - 基准、超额收益与 Beta
- `09-interactive-charts.py` - Plotly 交互式图表
- `10-vectorization-performance.py` - 向量化与性能优化

## 🎯 实战项目

**股票分析仪表板**（[`projects/phase-02-stock-dashboard/`](../../projects/README.md)）：输入股票代码，输出收益/波动/回撤统计、技术指标图和与沪深 300 的对比报告。

## 📖 参考资料

- [pandas 时间序列文档](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [《Python for Data Analysis》第 3 版](https://wesmckinney.com/book/)

## ✅ 学习检查

- [ ] 知道什么时候用简单收益、什么时候用对数收益
- [ ] 能正确年化日频、周频、月频数据
- [ ] 能计算最大回撤及其起止日期
- [ ] 能用向量化代替循环实现技术指标
