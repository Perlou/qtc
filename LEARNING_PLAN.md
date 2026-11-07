# 量化交易课程完整学习计划

> **定制对象**：资深工程师（编程强、金融/数学基础弱）转型量化策略研究员 / 量化交易员  
> **市场范围**：A 股为主，兼顾期货与加密货币  
> **学习方式**：每课一个可运行的 Python 文件，理论在代码中讲解  
> **预计时长**：26 周（每周投入 10-15 小时）  
> **当前进度**：🔄 准备开始第 1 阶段

---

## 📊 学习目标

1. 熟悉 A 股、期货、加密市场的交易机制与数据
2. 掌握量化所需的统计、时间序列与优化方法
3. 能搭建可信的回测系统并识别各类偏差
4. 能独立完成择时、套利、多因子、机器学习策略研究
5. 掌握组合优化、风险管理与绩效归因
6. 能设计并运维多市场实盘交易系统

## 🗓️ 每周节奏建议

| 环节 | 时间 | 内容 |
|---|---|---|
| 跑课程代码 | 4-5 小时 | 逐个运行本周 `.py`，读懂每一段输出 |
| 练习 | 3-4 小时 | 完成每课末尾的练习，代码放 `ex/` |
| 实战项目 | 3-5 小时 | 推进本阶段项目 |
| 复盘笔记 | 1 小时 | 在 `docs/` 记录本周收获、疑问与错误 |

---

## 🗺️ 详细学习计划

## 第一部分：基础能力（W1-6）

### 阶段 1：量化入门与市场数据 (1 周)

> **目标**：了解量化行业与 A 股交易规则，能获取、清洗、存储多市场行情数据  
> **目录**：[`src/phase-1-quant-intro-market-data/`](src/phase-1-quant-intro-market-data/README.md) · [核心概念](src/phase-1-quant-intro-market-data/CONCEPT.md)

#### 第 1 周：市场与数据

- [ ] **量化与市场**
  - 创建 `01-quant-trading-overview.py`：量化交易全景：岗位、策略分类、研究流程
  - 创建 `02-a-share-market-rules.py`：A 股交易规则：T+1、涨跌停、交易成本

- [ ] **数据获取**
  - 创建 `03-akshare-basics.py`：AKShare：股票、指数、ETF 行情
  - 创建 `04-tushare-pro.py`：Tushare Pro：日线、复权因子、交易日历
  - 创建 `05-ohlcv-candlestick.py`：OHLCV 与 K 线图

- [ ] **数据处理**
  - 创建 `06-adjusted-prices.py`：除权除息与前/后复权
  - 创建 `07-trading-calendar-alignment.py`：交易日历、停牌与多资产对齐
  - 创建 `08-data-storage.py`：CSV / Parquet / DuckDB 数据存储
  - 创建 `09-futures-crypto-data.py`：期货主力连续与加密货币数据
  - 创建 `10-data-biases-quality.py`：数据偏差与数据质量检查

- [ ] **实战项目**：本地行情数据仓库（`projects/phase-01-market-data-pipeline/`）
  - 增量下载 A 股日线与复权因子，按年分区存为 Parquet，用 DuckDB 查询，并生成数据质量报告。

---

### 阶段 2：金融时间序列与 pandas (1 周)

> **目标**：熟练用 pandas/NumPy 处理金融时间序列，掌握收益、波动、回撤等基本度量  
> **目录**：[`src/phase-2-financial-timeseries/`](src/phase-2-financial-timeseries/README.md) · [核心概念](src/phase-2-financial-timeseries/CONCEPT.md)

#### 第 2 周：收益与风险度量

- [ ] **收益与波动**
  - 创建 `01-returns-simple-vs-log.py`：简单收益与对数收益
  - 创建 `02-annualization-volatility.py`：年化收益与年化波动率
  - 创建 `03-rolling-windows.py`：滚动窗口统计

- [ ] **数据操作**
  - 创建 `04-resample-multi-frequency.py`：重采样与多频率数据
  - 创建 `05-multi-asset-panel.py`：多资产面板数据（宽表与长表）
  - 创建 `06-technical-indicators.py`：技术指标：MA、EMA、MACD、RSI、ATR、布林带

- [ ] **分析与可视化**
  - 创建 `07-drawdown-analysis.py`：回撤与水下曲线
  - 创建 `08-benchmark-relative.py`：基准、超额收益与 Beta
  - 创建 `09-interactive-charts.py`：Plotly 交互式图表
  - 创建 `10-vectorization-performance.py`：向量化与性能优化

- [ ] **实战项目**：股票分析仪表板（`projects/phase-02-stock-dashboard/`）
  - 输入股票代码，输出收益/波动/回撤统计、技术指标图和与沪深 300 的对比报告。

---

### 阶段 3：量化数学与统计 (2 周)

> **目标**：用代码补齐量化所需的概率统计、回归与时间序列知识  
> **目录**：[`src/phase-3-math-statistics/`](src/phase-3-math-statistics/README.md) · [核心概念](src/phase-3-math-statistics/CONCEPT.md)

#### 第 3 周：概率与统计推断

- [ ] **概率基础**
  - 创建 `01-probability-distributions.py`：常见分布与收益率分布
  - 创建 `02-fat-tails-normality.py`：肥尾、偏度、峰度与正态性检验
  - 创建 `03-hypothesis-testing.py`：假设检验：策略收益是否显著

- [ ] **模拟与决策**
  - 创建 `04-monte-carlo-simulation.py`：蒙特卡洛模拟与 Bootstrap
  - 创建 `05-kelly-criterion.py`：期望值、胜率、盈亏比与凯利公式

#### 第 4 周：回归与时间序列

- [ ] **回归与相关**
  - 创建 `06-linear-regression-ols.py`：OLS 回归与残差诊断
  - 创建 `07-correlation-vs-cointegration.py`：相关性与协整
  - 创建 `08-linear-algebra-pca.py`：协方差矩阵、特征分解与 PCA

- [ ] **时间序列模型**
  - 创建 `09-stationarity-adf.py`：平稳性与 ADF 检验
  - 创建 `10-arma-models.py`：自相关与 ARMA 模型
  - 创建 `11-garch-volatility.py`：波动率聚集与 GARCH

- [ ] **实战项目**：风险收益统计报告生成器（`projects/phase-03-stats-report/`）
  - 对任意资产生成统计报告：分布检验、尾部风险、自相关、GARCH 波动率预测与蒙特卡洛情景。

---

### 阶段 4：回测系统原理 (2 周)

> **目标**：理解并亲手实现回测引擎，能识别回测中的各类陷阱  
> **目录**：[`src/phase-4-backtesting/`](src/phase-4-backtesting/README.md) · [核心概念](src/phase-4-backtesting/CONCEPT.md)

#### 第 5 周：向量化回测

- [ ] **回测基础**
  - 创建 `01-backtest-overview.py`：回测的作用、流程与局限
  - 创建 `02-vectorized-backtest.py`：向量化回测实现
  - 创建 `03-performance-metrics.py`：绩效指标：Sharpe、Sortino、MDD、Calmar、胜率

- [ ] **现实约束**
  - 创建 `04-transaction-costs-slippage.py`：交易成本与滑点建模
  - 创建 `05-a-share-constraints.py`：A 股约束：T+1、涨跌停、停牌、100 股

#### 第 6 周：事件驱动与陷阱

- [ ] **事件驱动**
  - 创建 `06-event-driven-engine.py`：从零实现事件驱动回测引擎

- [ ] **回测陷阱**
  - 创建 `07-lookahead-bias.py`：前视偏差与未来函数
  - 创建 `08-overfitting-traps.py`：过拟合与多重检验
  - 创建 `09-walk-forward.py`：Walk-Forward 与样本外检验

- [ ] **开源框架**
  - 创建 `10-backtrader-intro.py`：backtrader 入门
  - 创建 `11-vectorbt-intro.py`：vectorbt 大规模参数回测

- [ ] **实战项目**：迷你回测引擎（`projects/phase-04-mini-backtester/`）
  - 实现支持 A 股约束、成本模型、多标的的事件驱动回测引擎，与 backtrader 结果对拍。

---

## 第二部分：策略研究（W7-13）

### 阶段 5：择时与 CTA 策略 (2 周)

> **目标**：掌握经典趋势与均值回归策略，能在期货上构建多品种 CTA 组合  
> **目录**：[`src/phase-5-cta-strategies/`](src/phase-5-cta-strategies/README.md) · [核心概念](src/phase-5-cta-strategies/CONCEPT.md)

#### 第 7 周：经典择时策略

- [ ] **趋势与反转**
  - 创建 `01-moving-average-crossover.py`：双均线策略
  - 创建 `02-bollinger-mean-reversion.py`：布林带均值回归
  - 创建 `03-macd-rsi-strategies.py`：MACD 与 RSI 策略
  - 创建 `04-channel-breakout.py`：通道突破（Donchian / Dual Thrust）
  - 创建 `05-turtle-trading.py`：海龟交易法则完整复现

#### 第 8 周：期货 CTA

- [ ] **期货与仓位**
  - 创建 `06-futures-contracts-rollover.py`：期货合约、保证金与展期
  - 创建 `07-position-sizing-atr.py`：ATR 仓位与波动率目标

- [ ] **优化与组合**
  - 创建 `08-parameter-optimization.py`：参数优化与参数平原
  - 创建 `09-robustness-tests.py`：稳健性检验：扰动、分段、跨品种
  - 创建 `10-multi-asset-cta-portfolio.py`：多品种 CTA 组合

- [ ] **实战项目**：期货 CTA 多品种组合（`projects/phase-05-cta-portfolio/`）
  - 在 10+ 个期货品种上构建趋势跟踪组合，包含展期处理、波动率目标仓位与稳健性报告。

---

### 阶段 6：统计套利 (1 周)

> **目标**：掌握配对交易与各类套利策略的原理、实现与风险  
> **目录**：[`src/phase-6-statistical-arbitrage/`](src/phase-6-statistical-arbitrage/README.md) · [核心概念](src/phase-6-statistical-arbitrage/CONCEPT.md)

#### 第 9 周：统计套利

- [ ] **配对交易**
  - 创建 `01-pairs-trading-intro.py`：配对交易原理与选对
  - 创建 `02-cointegration-tests.py`：Engle-Granger 与 Johansen 检验
  - 创建 `03-zscore-signals.py`：Z-score 信号与半衰期
  - 创建 `04-kalman-filter-hedge.py`：卡尔曼滤波动态对冲比

- [ ] **套利策略**
  - 创建 `05-calendar-spread.py`：期货跨期套利
  - 创建 `06-basis-arbitrage.py`：期现套利与基差
  - 创建 `07-crypto-funding-rate-arb.py`：加密永续合约资金费率套利
  - 创建 `08-stat-arb-risks.py`：统计套利的风险：协整断裂与拥挤

- [ ] **实战项目**：配对交易系统（`projects/phase-06-pairs-trading/`）
  - 在同行业股票或 ETF 中自动筛选协整对，用卡尔曼滤波对冲，并做滚动样本外回测。

---

### 阶段 7：多因子模型 (2 周)

> **目标**：掌握 A 股多因子选股的完整研究流程  
> **目录**：[`src/phase-7-multi-factor/`](src/phase-7-multi-factor/README.md) · [核心概念](src/phase-7-multi-factor/CONCEPT.md)

#### 第 10 周：因子构建与检验

- [ ] **资产定价**
  - 创建 `01-capm-beta.py`：CAPM 与 Beta
  - 创建 `02-fama-french.py`：Fama-French 三因子 / 五因子

- [ ] **因子研究**
  - 创建 `03-factor-construction.py`：因子构建：价值、动量、质量、低波、规模
  - 创建 `04-factor-preprocessing.py`：去极值、标准化、行业市值中性化
  - 创建 `05-ic-ir-analysis.py`：IC、Rank IC、IR 与 IC 衰减

#### 第 11 周：因子组合与风险模型

- [ ] **因子评估**
  - 创建 `06-quantile-backtest.py`：分层回测与多空组合
  - 创建 `07-alphalens-tearsheet.py`：alphalens 因子报告
  - 创建 `08-point-in-time-fundamentals.py`：财报 Point-in-Time 对齐

- [ ] **组合与风险**
  - 创建 `09-factor-combination.py`：因子合成：等权、IC 加权、最大化 IR
  - 创建 `10-barra-risk-model.py`：Barra 风格因子与风险模型
  - 创建 `11-factor-crowding-decay.py`：因子拥挤与衰减

- [ ] **实战项目**：A 股多因子选股（`projects/phase-07-multi-factor-stock-selection/`）
  - 构建 10+ 个因子，完成预处理、单因子检验、因子合成与月度调仓回测，输出研究报告。

---

### 阶段 8：组合优化与风险管理 (2 周)

> **目标**：掌握主流组合优化方法和风险度量、归因技术  
> **目录**：[`src/phase-8-portfolio-risk/`](src/phase-8-portfolio-risk/README.md) · [核心概念](src/phase-8-portfolio-risk/CONCEPT.md)

#### 第 12 周：组合优化

- [ ] **经典方法**
  - 创建 `01-mean-variance.py`：Markowitz 均值方差与有效前沿
  - 创建 `02-covariance-shrinkage.py`：协方差估计与 Ledoit-Wolf 收缩
  - 创建 `03-risk-parity.py`：风险平价
  - 创建 `04-black-litterman.py`：Black-Litterman 模型
  - 创建 `05-hrp.py`：层次风险平价 HRP

#### 第 13 周：风险管理

- [ ] **约束优化**
  - 创建 `06-constrained-optimization-cvxpy.py`：cvxpy 带约束组合优化

- [ ] **风险度量与归因**
  - 创建 `07-var-cvar.py`：VaR 与 CVaR
  - 创建 `08-stress-testing.py`：压力测试与情景分析
  - 创建 `09-performance-attribution.py`：Brinson 与因子归因
  - 创建 `10-rebalancing-turnover.py`：再平衡与换手控制

- [ ] **实战项目**：组合优化与风控模块（`projects/phase-08-portfolio-optimizer/`）
  - 实现可插拔的组合优化器（MV/RP/BL/HRP）与风险报告，接入 Phase 7 的因子信号。

---

## 第三部分：AI 量化（W14-17）

### 阶段 9：机器学习量化 (2 周)

> **目标**：掌握金融机器学习的正确方法论，避免常见的数据泄漏与过拟合  
> **目录**：[`src/phase-9-ml-quant/`](src/phase-9-ml-quant/README.md) · [核心概念](src/phase-9-ml-quant/CONCEPT.md)

#### 第 14 周：金融 ML 方法论

- [ ] **数据与标签**
  - 创建 `01-ml-quant-overview.py`：金融 ML 的特殊性
  - 创建 `02-feature-engineering.py`：特征工程：价量、基本面、另类特征
  - 创建 `03-labeling-triple-barrier.py`：Triple-Barrier 与 Meta-Labeling

- [ ] **验证与模型**
  - 创建 `04-purged-kfold-cv.py`：Purged K-Fold 与 Embargo
  - 创建 `05-lightgbm-stock-prediction.py`：LightGBM 截面收益预测

#### 第 15 周：从预测到交易

- [ ] **模型解释与组合**
  - 创建 `06-feature-importance-shap.py`：MDI、MDA、SHAP 特征重要性
  - 创建 `07-from-prediction-to-portfolio.py`：从预测分数到持仓

- [ ] **评估与工程**
  - 创建 `08-deflated-sharpe-pbo.py`：Deflated Sharpe Ratio 与 PBO
  - 创建 `09-qlib-workflow.py`：Qlib 完整工作流
  - 创建 `10-model-drift-monitoring.py`：模型漂移与再训练

- [ ] **实战项目**：ML 选股模型（`projects/phase-09-ml-stock-selection/`）
  - 基于 Qlib/LightGBM 的截面选股模型，Purged CV 验证，与 Phase 7 线性多因子对比。

---

### 阶段 10：深度学习、RL 与 LLM 量化 (2 周)

> **目标**：了解深度学习、强化学习和大模型在量化中的应用与边界  
> **目录**：[`src/phase-10-dl-rl-llm/`](src/phase-10-dl-rl-llm/README.md) · [核心概念](src/phase-10-dl-rl-llm/CONCEPT.md)

#### 第 16 周：深度学习与强化学习

- [ ] **深度学习**
  - 创建 `01-lstm-price-prediction.py`：LSTM 时序预测
  - 创建 `02-transformer-timeseries.py`：Transformer 时序模型
  - 创建 `03-qlib-deep-models.py`：Qlib 深度模型库

- [ ] **强化学习**
  - 创建 `04-rl-trading-env.py`：构建 Gym 交易环境
  - 创建 `05-rl-agents-ppo.py`：PPO 交易智能体

#### 第 17 周：LLM 量化

- [ ] **文本数据**
  - 创建 `06-news-sentiment.py`：新闻与公告情绪因子
  - 创建 `07-llm-research-report-analysis.py`：LLM 研报与财报分析

- [ ] **LLM Agent**
  - 创建 `08-llm-factor-mining.py`：LLM 生成与筛选因子表达式
  - 创建 `09-alpha-research-agent.py`：量化研究 Agent
  - 创建 `10-ai-quant-pitfalls.py`：AI 量化的陷阱：前视、过拟合与幻觉

- [ ] **实战项目**：LLM 因子研究助手（`projects/phase-10-llm-factor-agent/`）
  - 用 LLM Agent 自动提出因子假设、生成代码、调用回测并撰写研究摘要（衔接 llm-course）。

---

## 第四部分：交易系统与实盘（W18-26）

### 阶段 11：市场微观结构与执行 (2 周)

> **目标**：理解订单簿与交易成本的来源，掌握执行算法与做市基础  
> **目录**：[`src/phase-11-microstructure-execution/`](src/phase-11-microstructure-execution/README.md) · [核心概念](src/phase-11-microstructure-execution/CONCEPT.md)

#### 第 18 周：微观结构

- [ ] **订单簿**
  - 创建 `01-order-book-basics.py`：订单簿、撮合规则与订单类型
  - 创建 `02-tick-level2-data.py`：Tick 与 Level-2 数据处理
  - 创建 `03-spread-and-impact.py`：买卖价差与市场冲击
  - 创建 `04-realtime-orderbook-ccxt.py`：ccxt 实时订单簿
  - 创建 `05-high-frequency-factors.py`：高频因子：订单不平衡、成交量分布

#### 第 19 周：执行与做市

- [ ] **执行算法**
  - 创建 `06-twap-vwap.py`：TWAP / VWAP / POV
  - 创建 `07-almgren-chriss.py`：Almgren-Chriss 最优执行

- [ ] **做市**
  - 创建 `08-market-making-basics.py`：做市原理与库存风险
  - 创建 `09-avellaneda-stoikov.py`：Avellaneda-Stoikov 做市模型
  - 创建 `10-hft-infrastructure.py`：高频交易基础设施与延迟

- [ ] **实战项目**：加密订单簿做市模拟器（`projects/phase-11-market-making-sim/`）
  - 基于真实订单簿快照回放，实现 AS 做市策略与库存控制，评估 PnL 与风险。

---

### 阶段 12：期权与衍生品 (1 周)

> **目标**：掌握期权定价、希腊字母与波动率交易基础  
> **目录**：[`src/phase-12-options-derivatives/`](src/phase-12-options-derivatives/README.md) · [核心概念](src/phase-12-options-derivatives/CONCEPT.md)

#### 第 20 周：期权

- [ ] **定价**
  - 创建 `01-options-basics.py`：期权基础与收益结构
  - 创建 `02-black-scholes.py`：Black-Scholes 定价与二叉树
  - 创建 `03-greeks.py`：希腊字母：Delta、Gamma、Vega、Theta

- [ ] **波动率**
  - 创建 `04-implied-volatility.py`：隐含波动率求解
  - 创建 `05-volatility-surface.py`：波动率微笑与曲面
  - 创建 `06-etf-options-data.py`：50ETF / 300ETF 期权数据

- [ ] **交易**
  - 创建 `07-delta-hedging.py`：Delta 对冲模拟
  - 创建 `08-volatility-strategies.py`：波动率策略：跨式、价差、卖方策略风险

- [ ] **实战项目**：期权定价与波动率曲面工具（`projects/phase-12-options-toolkit/`）
  - 拉取 ETF 期权链，计算隐含波动率与希腊字母，绘制波动率曲面，并模拟 Delta 对冲。

---

### 阶段 13：交易系统工程与实盘 (3 周)

> **目标**：设计并实现可用于模拟盘和小资金实盘的多市场交易系统  
> **目录**：[`src/phase-13-trading-system-live/`](src/phase-13-trading-system-live/README.md) · [核心概念](src/phase-13-trading-system-live/CONCEPT.md)

#### 第 21 周：系统架构

- [ ] **核心组件**
  - 创建 `01-trading-system-architecture.py`：交易系统架构：行情、策略、风控、OMS、执行
  - 创建 `02-event-engine-asyncio.py`：asyncio 事件引擎
  - 创建 `03-oms-order-lifecycle.py`：OMS 与订单状态机
  - 创建 `04-risk-control-engine.py`：事前 / 事中风控引擎

#### 第 22 周：交易接口

- [ ] **接口接入**
  - 创建 `05-vnpy-intro.py`：vn.py 框架入门
  - 创建 `06-ctp-simnow.py`：CTP 期货接口与 SimNow 仿真
  - 创建 `07-a-share-qmt-gm.py`：A 股程序化接口：QMT / 掘金
  - 创建 `08-ccxt-live-trading.py`：ccxt 加密货币实盘与测试网

#### 第 23 周：运维与上线

- [ ] **生产运维**
  - 创建 `09-monitoring-alerting.py`：监控与告警
  - 创建 `10-reconciliation-recovery.py`：对账、幂等与故障恢复
  - 创建 `11-docker-deployment.py`：Docker 部署与定时任务
  - 创建 `12-paper-to-live.py`：从模拟盘到小资金实盘的流程与合规

- [ ] **实战项目**：多市场实盘交易系统（`projects/phase-13-live-trading-system/`）
  - 统一网关接入 CTP 仿真与加密测试网，包含 OMS、风控、对账、监控告警，连续运行至少 4 周模拟盘。

---

### 阶段 14：综合项目 (3 周)

> **目标**：整合全部知识，构建个人量化平台，完成职业化输出  
> **目录**：[`src/phase-14-capstone/`](src/phase-14-capstone/README.md) · [核心概念](src/phase-14-capstone/CONCEPT.md)

#### 第 24 周：平台设计

- [ ] **架构**
  - 创建 `01-platform-architecture.py`：mini-quant 平台架构设计
  - 创建 `02-data-and-factor-layer.py`：数据层与因子层实现

#### 第 25 周：平台实现

- [ ] **核心模块**
  - 创建 `03-model-and-portfolio-layer.py`：模型层与组合层实现
  - 创建 `04-execution-and-monitoring.py`：执行层与监控面板

#### 第 26 周：职业输出

- [ ] **研究与求职**
  - 创建 `05-research-report-writing.py`：量化研究报告写作
  - 创建 `06-strategy-due-diligence.py`：策略尽调与上线评审
  - 创建 `07-quant-career-interview.py`：量化岗位面试：概率题、编程题、策略题

- [ ] **实战项目**：mini-quant 量化平台（`projects/capstone-mini-quant/`）
  - 数据 → 因子 → 模型 → 组合 → 回测 → 模拟/实盘 → 监控的端到端平台，附完整研究报告。

---

## ✅ 通用验收标准

1. 每个策略结论都必须包含：样本区间、交易成本假设、基准对比、样本外结果
2. 每次参数尝试都记录在研究日志中，报告时披露尝试次数
3. 回测代码中的每个信号都能说明"在当时能否获得"
4. 实盘前至少 4 周模拟盘运行，且模拟盘与回测的偏差可解释
5. 每阶段结束写复盘：最重要的收获、踩过的坑、仍不确定的问题
