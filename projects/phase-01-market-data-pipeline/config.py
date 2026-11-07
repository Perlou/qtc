"""项目配置"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# 默认股票池：覆盖沪市主板、深市主板、创业板、科创板，以及一只高分红股
DEFAULT_SYMBOLS = ["600519", "000001", "300750", "688981", "601088"]


@dataclass
class Config:
    root: Path = PROJECT_DIR / "warehouse"  # 数据仓库根目录
    request_interval: float = 0.5  # 两次请求的最小间隔（秒），避免被数据源限流
    max_retries: int = 3  # 单次请求最大重试次数
    lookback_days: int = 7  # 增量更新时往回多取的自然日，覆盖数据源对近期数据的修正
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
