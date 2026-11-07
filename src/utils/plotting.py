"""
绘图工具
========

- setup_chinese_font(): 自动选择系统可用的中文字体，解决图表中文乱码
- save_figure(): 统一把图片保存到 outputs/ 目录
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端：课程脚本只保存图片，不弹窗

import matplotlib.pyplot as plt
from matplotlib import font_manager

from utils import OUTPUT_DIR

_CANDIDATE_FONTS = [
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
]


def setup_chinese_font() -> str | None:
    """配置 matplotlib 中文字体，返回选中的字体名（找不到时返回 None）"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATE_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
            return name
    return None


def save_figure(fig, filename: str, subdir: str = "") -> Path:
    """保存图片到 outputs/{subdir}/{filename}，返回保存路径"""
    out_dir = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
