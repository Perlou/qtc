"""
课程公共工具包

- data:     行情数据获取（Parquet 缓存 + 离线合成数据回退）
- plotting: 中文字体配置与图片输出
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
