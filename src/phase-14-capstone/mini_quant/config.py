"""
配置与可复现（横切关注点）

    StrategyConfig   策略的全部可调参数：因子列表、模型、组合约束、成本、样本划分
    load_config      从 config/strategy.yaml 读取并覆盖默认值（配置驱动：做实验改 YAML，不改代码）
    run_manifest     记录一次运行的代码 / 数据 / 配置版本，附在报告附录（CONCEPT 6.5 节）
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml

from ._paths import CONFIG_DIR, PACKAGE_DIR

DEFAULT_FACTORS = ("ep", "bp", "roe", "mom_12_1", "rev_1m", "lowvol", "lowturn", "size")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "mq_multifactor"
    # 因子层
    factors: tuple[str, ...] = DEFAULT_FACTORS
    neutralize: bool = True  # 行业 + 市值中性化
    # 模型层
    model: str = "ridge"  # equal / icw / ridge / hgb
    train_window: int = 36  # 滚动训练窗口（期数，月频即月数）
    min_train: int = 10  # 至少多少期标签才开始训练（10 期 → 2020 年 1 月起所有模型都有分数）
    retrain_every: int = 1  # 每隔几期重新训练
    # 组合层
    n_top: int = 20
    weighting: str = "equal"  # equal / inverse_vol
    max_weight: float = 0.10  # 单票权重硬上限（风控）
    max_turnover: float = 0.5  # 每次调仓单边换手上限
    # 执行与成本
    capital: float = 1_000_000.0
    min_order_value: float = 3_000.0  # 小于该金额的调仓不做（避免最低佣金 5 元吞噬收益）
    slippage: float = 0.001  # 单边滑点 / 冲击（中小盘取 10bp，与 Phase 7 一致）
    cost_multiplier: float = 1.0  # 成本敏感性测试用
    # 样本划分（参数只在样本内选择；样本外只评估一次）
    is_start: str = "2020-01-01"
    is_end: str = "2022-12-31"
    oos_start: str = "2023-01-01"
    oos_end: str = "2024-12-31"
    seed: int = 42

    def with_(self, **kw) -> "StrategyConfig":
        if "factors" in kw:
            kw["factors"] = tuple(kw["factors"])
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = list(self.factors)
        return d

    def model_key(self) -> tuple:
        """决定"预测分数"的参数（组合层参数变化时可复用同一份分数）"""
        return (self.model, self.factors, self.neutralize, self.train_window, self.min_train, self.retrain_every)

    def short(self) -> str:
        return f"{self.model}|N{self.n_top}|{self.weighting}|TO{self.max_turnover:g}"


def load_config(path: str | Path | None = None, **overrides) -> StrategyConfig:
    path = Path(path) if path else CONFIG_DIR / "strategy.yaml"
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(StrategyConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"配置文件含未知字段：{sorted(unknown)}")
    data.update(overrides)
    if "factors" in data:
        data["factors"] = tuple(data["factors"])
    return StrategyConfig(**data)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def code_fingerprint() -> str:
    """mini_quant 包全部源码的哈希：项目不在 git 仓库时，用它代替 commit id"""
    h = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_manifest(config: StrategyConfig | dict, data_files: list[Path] | None = None, extra: dict | None = None) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=PACKAGE_DIR
        ).strip()
    except Exception:  # 不是 git 仓库或没装 git
        commit = "unknown（非 git 仓库，见 code_sha256）"
    import numpy
    import pandas
    import sklearn

    cfg = config.to_dict() if isinstance(config, StrategyConfig) else dict(config)
    return {
        "git_commit": commit,
        "code_sha256": code_fingerprint(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"pandas": pandas.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__},
        "config": cfg,
        "data_hashes": {Path(p).name: _sha(Path(p)) for p in (data_files or []) if Path(p).exists()},
        **(extra or {}),
    }


def manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, default=str)


__all__ = ["DEFAULT_FACTORS", "StrategyConfig", "load_config", "run_manifest", "manifest_json", "code_fingerprint"]
