# Phase 13: 交易系统工程与实盘

> **学习目标**：设计并实现可用于模拟盘和小资金实盘的多市场交易系统
> **预计时长**：3 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 21 周：系统架构

#### 1. 核心组件

- `01-trading-system-architecture.py` - 交易系统架构：行情、策略、风控、OMS、执行
- `02-event-engine-asyncio.py` - asyncio 事件引擎
- `03-oms-order-lifecycle.py` - OMS 与订单状态机
- `04-risk-control-engine.py` - 事前 / 事中风控引擎

### 第 22 周：交易接口

#### 2. 接口接入

- `05-vnpy-intro.py` - vn.py 框架入门
- `06-ctp-simnow.py` - CTP 期货接口与 SimNow 仿真
- `07-a-share-qmt-gm.py` - A 股程序化接口：QMT / 掘金
- `08-ccxt-live-trading.py` - ccxt 加密货币实盘与测试网

### 第 23 周：运维与上线

#### 3. 生产运维

- `09-monitoring-alerting.py` - 监控与告警
- `10-reconciliation-recovery.py` - 对账、幂等与故障恢复
- `11-docker-deployment.py` - Docker 部署与定时任务
- `12-paper-to-live.py` - 从模拟盘到小资金实盘的流程与合规

## 🎯 实战项目

**多市场实盘交易系统**（[`projects/phase-13-live-trading-system/`](../../projects/README.md)）：统一网关接入 CTP 仿真与加密测试网，包含 OMS、风控、对账、监控告警，连续运行至少 4 周模拟盘。

## 📖 参考资料

- [vn.py](https://www.vnpy.com/)
- [SimNow 仿真](https://www.simnow.com.cn/)
- [证监会《证券市场程序化交易管理规定（试行）》](http://www.csrc.gov.cn/)

## ✅ 学习检查

- [ ] 订单状态机覆盖拒单、部分成交、撤单
- [ ] 重启后能从交易所状态恢复持仓
- [ ] 风控能在异常时自动暂停交易
- [ ] 了解 A 股程序化交易报备要求
