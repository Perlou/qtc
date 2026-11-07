# Phase 10: 深度学习、RL 与 LLM 量化

> **学习目标**：了解深度学习、强化学习和大模型在量化中的应用与边界
> **预计时长**：2 周

📖 **核心概念详解**：[CONCEPT.md](./CONCEPT.md)（建议先通读，再运行代码）

## 📚 本阶段内容

### 第 16 周：深度学习与强化学习

#### 1. 深度学习

- `01-lstm-price-prediction.py` - LSTM 时序预测
- `02-transformer-timeseries.py` - Transformer 时序模型
- `03-qlib-deep-models.py` - Qlib 深度模型库

#### 2. 强化学习

- `04-rl-trading-env.py` - 构建 Gym 交易环境
- `05-rl-agents-ppo.py` - PPO 交易智能体

### 第 17 周：LLM 量化

#### 3. 文本数据

- `06-news-sentiment.py` - 新闻与公告情绪因子
- `07-llm-research-report-analysis.py` - LLM 研报与财报分析

#### 4. LLM Agent

- `08-llm-factor-mining.py` - LLM 生成与筛选因子表达式
- `09-alpha-research-agent.py` - 量化研究 Agent
- `10-ai-quant-pitfalls.py` - AI 量化的陷阱：前视、过拟合与幻觉

## 🎯 实战项目

**LLM 因子研究助手**（[`projects/phase-10-llm-factor-agent/`](../../projects/README.md)）：用 LLM Agent 自动提出因子假设、生成代码、调用回测并撰写研究摘要（衔接 llm-course）。

## 📖 参考资料

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [RD-Agent (Microsoft)](https://github.com/microsoft/RD-Agent)

## ✅ 学习检查

- [ ] 能说明深度模型在低信噪比数据上的局限
- [ ] 能构建无前视偏差的 RL 交易环境
- [ ] 能用 LLM 构建文本因子并检验 IC
- [ ] 理解 LLM 训练数据截止日期带来的前视风险
