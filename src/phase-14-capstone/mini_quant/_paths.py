"""
路径与跨阶段导入

mini-quant 的原则是"复用前面阶段的模块，不重写"。前面各阶段的模块放在各自目录下，
文件夹名带连字符（如 phase-7-multi-factor），不能直接 import，这里统一把它们加入 sys.path：

    src/                          utils.data / utils.plotting
    phase-4-backtesting/          bt_core（交易规则、成本模型、绩效、PSR/DSR）
    phase-7-multi-factor/         factor_data / factor_tools（截面面板、PIT、预处理、IC）
    phase-8-portfolio-risk/       portfolio_utils（组合权重、风险贡献）
    phase-13-trading-system-live/ mini_trader（OMS、风控、MockExchange、对账）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# HistGradientBoosting 在小数据上线程过多反而更慢（Phase 9 的经验），导入 sklearn 前先限制
os.environ.setdefault("OMP_NUM_THREADS", "4")

PACKAGE_DIR = Path(__file__).resolve().parent
PHASE_DIR = PACKAGE_DIR.parent
SRC_DIR = PHASE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

PHASE_DIRS = {
    "phase4": SRC_DIR / "phase-4-backtesting",
    "phase7": SRC_DIR / "phase-7-multi-factor",
    "phase8": SRC_DIR / "phase-8-portfolio-risk",
    "phase13": SRC_DIR / "phase-13-trading-system-live",
}

for _p in (SRC_DIR, *PHASE_DIRS.values()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SUBDIR = "phase-14"
OUT_DIR = PROJECT_ROOT / "outputs" / SUBDIR
CONFIG_DIR = PHASE_DIR / "config"
