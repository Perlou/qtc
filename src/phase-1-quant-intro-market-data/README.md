# Phase 1: 量化入门与市场数据

> **学习目标**：了解量化行业与 A 股交易规则，能获取、清洗、存储多市场行情数据
> **预计时长**：1 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

#### 1. 量化与市场

- `01-quant-trading-overview.py` - 量化交易全景：岗位、策略分类、研究流程
- `02-a-share-market-rules.py` - A 股交易规则：T+1、涨跌停、交易成本

#### 2. 数据获取

- `03-akshare-basics.py` - AKShare：股票、指数、ETF 行情
- `04-tushare-pro.py` - Tushare Pro：日线、复权因子、交易日历
- `05-ohlcv-candlestick.py` - OHLCV 与 K 线图

#### 3. 数据处理

- `06-adjusted-prices.py` - 除权除息与前/后复权
- `07-trading-calendar-alignment.py` - 交易日历、停牌与多资产对齐
- `08-data-storage.py` - CSV / Parquet / DuckDB 数据存储
- `09-futures-crypto-data.py` - 期货主力连续与加密货币数据
- `10-data-biases-quality.py` - 数据偏差与数据质量检查

## 🎯 实战项目

**本地行情数据仓库**（[`projects/phase-01-market-data-pipeline/`](../../projects/phase-01-market-data-pipeline/)）：增量下载 A 股日线与复权因子，按年分区存为 Parquet，用 DuckDB 查询，并生成数据质量报告。

## 📖 参考资料

- [AKShare 文档](https://akshare.akfamily.xyz/)
- [Tushare Pro 文档](https://tushare.pro/document/2)
- [ccxt 文档](https://docs.ccxt.com/)
- [上交所交易规则](https://www.sse.com.cn/lawandrules/sserules/trading/)

## ✅ 学习检查

- [ ] 能说出量化研究员、量化交易员、量化开发的分工
- [ ] 能手算一笔 A 股买卖的全部交易成本
- [ ] 理解前复权与后复权的区别，以及回测应该用哪种
- [ ] 能从 AKShare / ccxt 获取数据并存成 Parquet
- [ ] 能列举至少 5 种数据质量问题和偏差
