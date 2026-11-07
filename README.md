# 📈 量化交易全栈课程 (Quantitative Trading Course)

> **定制对象**：资深工程师（编程强、金融/数学基础弱）向量化策略研究员 / 量化交易员转型  
> **学习方式**：每课一个可运行的 Python 文件，金融与数学知识"边写代码边补"  
> **市场范围**：A 股为主，兼顾期货与加密货币  
> **预计时长**：26 周（每周投入 10-15 小时）

---

## 🚀 快速开始

### 1. 环境设置

```bash
cd qtc

# 创建虚拟环境（推荐 Python 3.11 / 3.12，部分量化库尚未适配 3.13+）
python3.12 -m venv venv
# 或者使用 uv：uv venv --python 3.12 venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（Tushare token 等，可选）
cp .env.example .env
```

### 2. 运行课程

```bash
source venv/bin/activate
python src/phase-1-quant-intro-market-data/01-quant-trading-overview.py

# 没有网络或数据源不可用时，使用离线合成数据运行
QTC_OFFLINE=1 python src/phase-1-quant-intro-market-data/03-akshare-basics.py
```

### 3. 常见问题

<details>
<summary><b>❌ AKShare 接口报错 / 返回空数据</b></summary>

AKShare 抓取的是公开网页数据，接口偶尔会随数据源变化而失效。

```bash
# 1. 升级到最新版本
pip install -U akshare

# 2. 仍失败时，临时使用离线合成数据继续学习
QTC_OFFLINE=1 python src/phase-1-quant-intro-market-data/xx.py
```

课程的 `src/utils/data.py` 在网络失败时会自动回退到合成数据，并打印 `⚠️ 使用合成数据` 提示。

</details>

<details>
<summary><b>❌ ccxt 访问交易所超时</b></summary>

国内网络访问海外交易所通常需要代理，在 `.env` 中设置 `HTTPS_PROXY=http://127.0.0.1:7890`（按你的代理端口修改）。

</details>

<details>
<summary><b>❌ mplfinance 图片中文乱码</b></summary>

课程的 `src/utils/plotting.py` 会自动选择系统中文字体（macOS: PingFang SC / Heiti）。图片保存在 `outputs/` 目录。

</details>

---

## 📂 项目结构

```
qtc/
├── README.md                    # 快速开始指南
├── ROADMAP.md                   # 学习路线图
├── LEARNING_PLAN.md             # 详细学习计划（逐周）
├── CONCEPTS.md                  # 核心概念速查
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── src/                         # 课程代码（按阶段组织）
│   ├── utils/                   # 公共工具：数据获取（缓存 + 离线回退）、绘图
│   ├── phase-1-quant-intro-market-data/
│   ├── phase-2-financial-timeseries/
│   └── ...
├── projects/                    # 每阶段配套实战项目
├── notebooks/                   # Jupyter 研究笔记
├── docs/                        # 学习笔记
├── data/                        # 行情缓存（git 忽略）
└── outputs/                     # 图表与报告输出（git 忽略）
```

**重要文档**：

- 🗺️ [ROADMAP.md](./ROADMAP.md) - 可视化学习路线
- 📝 [LEARNING_PLAN.md](./LEARNING_PLAN.md) - 逐周学习计划
- 📖 [CONCEPTS.md](./CONCEPTS.md) - 核心概念速查
- 🛠️ [projects/README.md](./projects/README.md) - 实战项目说明

---

## 🧭 课程设计原则

1. **代码优先**：每个知识点都落到一个能跑的 `.py` 文件，理论在 `print` 和注释里讲清楚
2. **数学按需**：不单独花几周补数学，用到协整就学协整，用到优化就学凸优化
3. **从研究到实盘**：覆盖 数据 → 因子/策略 → 回测 → 组合 → 执行 → 实盘运维 全链路
4. **对过拟合保持警惕**：前视偏差、幸存者偏差、多重检验、样本外验证贯穿全程
5. **项目驱动**：每阶段一个实战项目，最后整合成自己的量化平台 `mini-quant`

---

## 🛠️ 技术栈

- **Python 3.11/3.12** - 核心开发语言
- **数据**：AKShare、Tushare Pro、ccxt
- **分析**：NumPy、pandas、SciPy、statsmodels
- **存储**：Parquet、DuckDB
- **回测**：自研引擎、backtrader、vectorbt
- **因子与组合**：alphalens、cvxpy、riskfolio-lib
- **AI 量化**：LightGBM、PyTorch、Qlib、Stable-Baselines3、LLM
- **实盘**：vn.py + CTP、QMT / 掘金、ccxt

---

## ⚠️ 风险提示

本课程用于学习研究。历史回测不代表未来收益，实盘交易可能损失全部本金。进入 Phase 13 实盘环节前，请先完成至少 4 周模拟盘运行，并只使用可以承受损失的小额资金。

---

**Good luck! 🚀**

有任何问题随时记录在 `docs/` 中，养成写研究笔记的习惯——这是量化研究员最重要的职业习惯之一。
