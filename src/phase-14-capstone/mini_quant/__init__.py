"""
mini-quant：课程综合项目的个人量化平台

    数据层 data/       Phase 7 截面面板适配器（PIT 财务、可交易性、基准）+ 数据质量检查
    因子层 factors/    因子注册表（装饰器）+ MAD / Z-score / 中性化 + 单因子 IC 诊断
    模型层 models/     等权 / IC 加权 / Ridge / HistGradientBoosting + 滚动训练与版本记录
    组合层 portfolio/  TopN + 换手约束 + 单票上限 + 按交易单位取整的调仓指令
    回测层 backtest/   月频组合回测（消费组合层输出）+ 绩效 / DSR / 容量
    执行层 execution/  调仓指令 → Phase 13 mini_trader（风控、OMS、MockExchange）+ 对账
    监控层 monitor/    日报指标、因子健康、特征漂移、组合暴露、执行质量、告警
    pipeline.py        一键流水线：每步校验，失败即停
    research.py        研究编排：试验日志、参数网格、样本内选择

层与层之间只通过 contracts.py 中的 Protocol 与数据契约交互。
"""

from . import _paths  # noqa: F401  （必须最先导入：把前面各阶段目录加入 sys.path）
from ._paths import OUT_DIR, PHASE_DIR, SUBDIR
from .config import StrategyConfig, load_config, manifest_json, run_manifest
from .contracts import PipelineError, RebalanceOrder

__version__ = "0.1.0"

LAYERS = ["data", "factors", "models", "portfolio", "backtest", "execution", "monitor"]

__all__ = ["OUT_DIR", "PHASE_DIR", "SUBDIR", "StrategyConfig", "load_config", "manifest_json", "run_manifest",
           "PipelineError", "RebalanceOrder", "LAYERS", "__version__"]
