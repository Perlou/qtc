# 本地 A 股行情数据仓库

> **Phase 1 实战项目** - 构建后续所有阶段都会用到的数据基础设施

## 项目简介

从 AKShare 增量下载 A 股不复权日线和后复权因子，按年分区存成 Parquet，用 DuckDB 做 SQL 查询，并生成数据质量报告。

后续阶段的回测、因子研究都可以直接读这个仓库，不必每次都重新下载。

## 功能特性

- ✅ 全量下载与增量更新（主键去重，重复运行结果不变）
- ✅ 东方财富为主、新浪为备的数据源切换，自动重试和限速
- ✅ 存储不复权价格和等比后复权因子，按需计算复权价格
- ✅ 按年分区的 Parquet 仓库，DuckDB 可直接用 SQL 查询
- ✅ 7 类数据质量检查，输出 Markdown 报告
- ✅ 离线模式（合成数据），测试不依赖网络

## 项目结构

```
phase-01-market-data-pipeline/
├── README.md          # 项目说明
├── requirements.txt   # 依赖列表
├── main.py            # CLI 入口：init / update / query / report
├── config.py          # 配置（仓库路径、限速、重试、默认股票池）
├── fetcher.py         # 数据获取：AkshareFetcher / SyntheticFetcher
├── storage.py         # 存储：分区写入、upsert、复权、DuckDB 查询
├── quality.py         # 数据质量检查与报告
├── warehouse/         # 数据仓库（运行后生成，git 忽略）
└── tests/             # pytest 测试（使用合成数据）
```

## 快速开始

### 1. 安装依赖

```bash
# 在课程根目录的虚拟环境中（已包含全部依赖）
source ../../venv/bin/activate
# 或单独安装
pip install -r requirements.txt
```

### 2. 全量下载

```bash
# 默认股票池（茅台、平安银行、宁德时代、中芯国际、中国神华）
python main.py init --start 20200101

# 沪深 300 成分股（约 300 只，需要几分钟）
python main.py init --universe hs300 --start 20200101

# 指定股票
python main.py init --symbols 600519,000001 --start 20150101
```

### 3. 增量更新

```bash
python main.py update     # 从每只股票的最后日期往回多取 7 天，更新到今天
```

### 4. 查询与报告

```bash
python main.py query "SELECT symbol, count(*) AS n, max(date) AS last FROM stock_daily GROUP BY 1"
python main.py report     # 生成 warehouse/reports/quality_*.md
```

### 5. 运行测试

```bash
pytest -q
```

## 使用示例

```
$ python main.py init --start 20220101 --end 20241231
📥 全量下载 5 只股票，20220101 ~ 20241231
  [1/5] 600519: 726 行
  [2/5] 000001: 726 行
  ...
✅ 写入 3,630 行，失败 0 只

$ python main.py update --end 20250131
📥 增量更新 5 只股票，20241224 ~ 20250131
✅ 更新 120 行（含重叠区间），失败 0 只

$ python main.py report
✅ 未发现数据质量问题
📝 报告：warehouse/reports/quality_20261107_210857.md
```

在 Python 中使用（后续阶段）：

```python
from storage import Warehouse

wh = Warehouse("projects/phase-01-market-data-pipeline/warehouse")
df = wh.read_daily_adjusted(symbols=["600519"], start="20230101")
# 列：symbol, date, open, high, low, close, volume, amount, adj_factor, adj_close
```

## 设计要点

| 决策 | 原因 |
|---|---|
| 存不复权价格 + 复权因子 | 前复权价格每次分红都会整体变化，无法增量更新；撮合需要真实价格（见 06 课） |
| 等比复权（新浪因子） | 东方财富的复权是加减法，除权日收益不准确（见 06 课） |
| 按年分区 | 日线最常见的查询是按时间范围，分区过滤可以跳过无关文件 |
| 主键 upsert + 往回多取 7 天 | 重复运行结果不变；能覆盖数据源对近期数据的修正 |
| Fetcher 接口 + 合成数据实现 | 测试不依赖网络；以后更换数据源（Tushare 等）只需新增一个实现 |

## 扩展练习

- [ ] 新增 `TushareFetcher`，并与 AkshareFetcher 做交叉验证
- [ ] 下载退市股票名单与历史数据，消除幸存者偏差
- [ ] 存储每日 ST 状态，让涨跌幅检查区分 ST 股（注意 2026-07-06 主板 ST 涨跌幅规则变化）
- [ ] 增加指数日线和交易日历表，检查"交易日历上有、但数据缺失"的日期
- [ ] 用 `concurrent.futures` 并发下载（注意数据源限流）
