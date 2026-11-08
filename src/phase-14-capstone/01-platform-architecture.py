"""
mini-quant 平台架构设计
======================

学习目标：
    1. 理解量化平台的分层架构：数据 → 因子 → 模型 → 组合 → 回测 / 执行 → 监控
    2. 用 typing.Protocol 定义层间接口，理解"结构化子类型"带来的可替换性与可测试性
    3. 看懂 mini_quant 的目录结构，以及它如何复用 Phase 4 / 7 / 8 / 13 的模块
    4. 自动分析源码 import，打印依赖关系图，并检查"层间不得隐式耦合"的规则
    5. 运行一个最小的端到端流程：一键从数据更新跑到生成调仓指令；再演示"失败即停"

核心概念：
    - 数据契约优先：层间只通过约定格式的 DataFrame / 数据类交互
    - 依赖注入：流水线只依赖接口，真实组件在 build_daily_pipeline 中装配
    - 回测与实盘消费同一份组合层输出

前置知识：
    - Phase 1–13 全部内容（本阶段不引入新的金融理论）
    - 本阶段 CONCEPT.md 第 1–2 节

环境要求：
    - pandas、numpy、scikit-learn、pyyaml（课程环境已安装）
    - 无网络时可用 QTC_OFFLINE=1 运行（先读本地缓存，缓存不存在才用合成数据）
"""

import ast
import inspect
import sys
from collections import defaultdict
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR.parent))
sys.path.insert(0, str(PHASE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import mini_quant  # noqa: E402  （导入时把前面各阶段目录加入 sys.path）
from mini_quant import contracts, load_config, manifest_json, run_manifest  # noqa: E402
from mini_quant.contracts import PipelineError  # noqa: E402
from mini_quant.pipeline import build_daily_pipeline  # noqa: E402
from mini_quant.portfolio import orders_frame  # noqa: E402
from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()

PKG_DIR = PHASE_DIR / "mini_quant"

# 前面阶段的模块 → 所属阶段（用于依赖分析）
EXTERNAL = {
    "utils": "src/utils",
    "bt_core": "Phase 4",
    "factor_data": "Phase 7",
    "factor_tools": "Phase 7",
    "portfolio_utils": "Phase 8",
    "mini_trader": "Phase 13",
}


# ==================== 第一部分：从课程到平台 ====================


def course_to_platform():
    print("=" * 60)
    print("第一部分：从 13 个阶段到一条流水线")
    print("=" * 60)
    rows = [
        ["数据层 data/", "Phase 1 数据/复权、Phase 7 截面面板与 PIT 财务", "factor_data、factor_tools"],
        ["因子层 factors/", "Phase 7 因子构建与预处理", "factor_tools.build_factors / preprocess"],
        ["模型层 models/", "Phase 7 因子合成、Phase 9 机器学习思路", "scikit-learn（Ridge、HistGB）"],
        ["组合层 portfolio/", "Phase 8 组合构建、Phase 4 交易规则与成本", "portfolio_utils、bt_core（rules.py 适配）"],
        ["回测层 backtest/", "Phase 4 绩效与 DSR、Phase 11 平方根冲击", "bt_core"],
        ["执行层 execution/", "Phase 13 风控、OMS、MockExchange、对账", "mini_trader"],
        ["监控层 monitor/", "Phase 13 监控告警、Phase 7 因子衰减", "（本阶段实现）"],
    ]
    print(pd.DataFrame(rows, columns=["平台层", "来自哪些阶段", "复用的模块"]).to_string(index=False))
    print("""
    🔑 本阶段的原则：能复用就不重写。前面阶段已经验证过的逻辑（PIT 对齐、涨跌停判断、
       成本模型、DSR、订单状态机……）在平台里只写薄薄的适配器（adapter）。
       重写一遍不仅浪费时间，还会引入"两份实现口径不一致"的隐患。
    """)


# ==================== 第二部分：分层与 Protocol ====================


class ConstantModel:
    """一个没有继承任何基类的"模型"：只要方法签名对得上，就满足 Model 协议"""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return pd.Series(0.0, index=X.index)


def layers_and_protocols():
    print("\n" + "=" * 60)
    print("第二部分：分层架构与 Protocol 接口")
    print("=" * 60)
    print("""
    ┌──────────┐ 行情/财务  ┌──────────┐ 因子矩阵  ┌──────────┐ 预测分数  ┌──────────┐
    │  数据层   │ ────────▶ │  因子层   │ ───────▶ │  模型层   │ ───────▶ │  组合层   │
    └──────────┘ (PIT)      └──────────┘          └──────────┘          └────┬─────┘
                                                              目标权重 / 调仓指令 │
                                       ┌───────────────────────────────┬──────┘
                                       ▼                               ▼
                                 ┌──────────┐                    ┌──────────┐ 成交/对账 ┌──────────┐
                                 │  回测层   │                    │  执行层   │ ───────▶ │  监控层   │
                                 └──────────┘                    └──────────┘          └──────────┘
       横切：配置（config/strategy.yaml）· 校验（失败即停）· manifest（可复现）· 测试（tests/）
    """)
    print("    📄 mini_quant/contracts.py 中的接口定义：\n")
    for name in ["DataSource", "FactorEngine", "Model", "PortfolioConstructor", "OrderPlanner", "Executor"]:
        proto = getattr(contracts, name)
        methods = [m for m in proto.__dict__ if not m.startswith("_") and callable(proto.__dict__[m])]
        sigs = [f"{m}{inspect.signature(proto.__dict__[m])}".replace("'", "") for m in methods]
        print(f"    class {name}(Protocol):")
        for s in sigs:
            print(f"        def {s}: ...")
    print("""
    Protocol 是"结构化子类型"：不需要 class X(Model) 显式继承，只要有同名同参数的方法即可，
    类似 Go 的 interface、TypeScript 的 interface。好处：
      ① 实现可以自由替换（Ridge → HistGradientBoosting，MockExchange → 真实柜台）
      ② 测试里几行代码就能写一个假实现（tests/test_pipeline.py 就是这样测流水线的）
      ③ 平台核心不依赖任何第三方模型库
    """)
    from mini_quant.data import PanelDataSource
    from mini_quant.execution import MiniTraderExecutor
    from mini_quant.factors import RegistryFactorEngine
    from mini_quant.models import make_model
    from mini_quant.portfolio import LotOrderPlanner, TopNPortfolio

    checks = [
        ("PanelDataSource", PanelDataSource, contracts.DataSource),
        ("RegistryFactorEngine", RegistryFactorEngine, contracts.FactorEngine),
        ("make_model('ridge')", type(make_model("ridge")), contracts.Model),
        ("make_model('hgb')", type(make_model("hgb")), contracts.Model),
        ("ConstantModel（本文件，未继承）", ConstantModel, contracts.Model),
        ("TopNPortfolio", TopNPortfolio, contracts.PortfolioConstructor),
        ("LotOrderPlanner", LotOrderPlanner, contracts.OrderPlanner),
        ("MiniTraderExecutor", MiniTraderExecutor, contracts.Executor),
    ]
    rows = []
    for label, cls, proto in checks:
        missing = [m for m in proto.__dict__ if not m.startswith("_") and callable(proto.__dict__[m]) and not hasattr(cls, m)]
        rows.append([label, proto.__name__, "✅" if not missing else f"❌ 缺少 {missing}"])
    print(pd.DataFrame(rows, columns=["实现", "协议", "满足"]).to_string(index=False))
    print("""
    ⚠️ runtime_checkable 的 isinstance 只检查"方法名是否存在"，不检查参数和返回类型；
       真正的类型检查交给 mypy / pyright，运行时的数据检查交给 validate_* 函数（第五部分）。
    """)


# ==================== 第三部分：目录结构 ====================


def directory_layout():
    print("\n" + "=" * 60)
    print("第三部分：目录结构")
    print("=" * 60)
    total = 0
    lines = ["    src/phase-14-capstone/",
             "    ├── config/strategy.yaml            策略配置（因子、模型、约束、成本、样本划分）",
             "    ├── mini_quant/"]
    top = sorted(p for p in PKG_DIR.iterdir() if p.suffix == ".py")
    subs = sorted(p for p in PKG_DIR.iterdir() if p.is_dir() and not p.name.startswith("__"))
    for path in top:
        n = len(path.read_text(encoding="utf-8").splitlines())
        total += n
        lines.append(f"    │   ├── {path.name:<26}{n:>5} 行")
    for d in subs:
        files = sorted(d.glob("*.py"))
        n_sub = 0
        for f in files:
            n_sub += len(f.read_text(encoding="utf-8").splitlines())
        total += n_sub
        names = ", ".join(f.stem for f in files if f.name != "__init__.py")
        lines.append(f"    │   ├── {d.name + '/':<26}{n_sub:>5} 行   {names}")
    lines.append("    ├── tests/                        pytest：因子注册、调仓取整、流水线校验、面试编程题")
    lines.append("    └── 01–07 课程文件")
    print("\n".join(lines))
    print(f"\n    mini_quant 共 {total} 行（不含前面阶段复用的模块）")
    print("""
    与 CONCEPT 2.3 节示例目录的对应：config/ ↔ config/，mini_quant/ ↔ mq/，
    store/ 与 reports/ 的角色由 data/cache（Phase 1–7 的 Parquet 缓存）和 outputs/phase-14/ 承担。
    """)


# ==================== 第四部分：依赖关系图 ====================


def module_layer(path: Path) -> str:
    rel = path.relative_to(PKG_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def scan_imports() -> tuple[dict, dict]:
    """解析 mini_quant 每个模块的 import 语句：返回 (层→内部依赖层, 层→外部依赖模块)"""
    internal, external = defaultdict(set), defaultdict(set)
    for path in PKG_DIR.rglob("*.py"):
        layer = module_layer(path)
        pkg_parts = path.relative_to(PKG_DIR).parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level > 0:  # 相对导入：解析到 mini_quant 内部的层
                    base = list(pkg_parts[: len(pkg_parts) - node.level + 1])
                    target = (base + mod.split(".")) if mod else base
                    target = [t for t in target if t]
                    if not target:  # from .. import X
                        names = [a.name for a in node.names]
                        target = [names[0]]
                    tgt = target[0]
                    if tgt != layer and tgt not in ("_paths", "__init__"):
                        internal[layer].add(tgt)
                elif mod.split(".")[0] in EXTERNAL:
                    external[layer].add(mod.split(".")[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in EXTERNAL:
                        external[layer].add(a.name.split(".")[0])
    return internal, external


def dependency_graph():
    print("\n" + "=" * 60)
    print("第四部分：依赖关系图（自动分析源码 import）")
    print("=" * 60)
    internal, external = scan_imports()
    order = ["contracts", "config", "rules", "data", "factors", "models", "portfolio", "backtest", "execution", "monitor",
             "pipeline", "research"]
    print("\n    内部依赖（A ──▶ B 表示 A import 了 B）：")
    for layer in order:
        deps = sorted(internal.get(layer, set()), key=lambda x: order.index(x) if x in order else 99)
        ext = sorted(external.get(layer, set()))
        ext_s = "、".join(f"{m}（{EXTERNAL[m]}）" for m in ext)
        print(f"      {layer:<10} ──▶ {', '.join(deps) if deps else '（无）':<40} 复用：{ext_s or '—'}")

    # 架构规则：七个层之间只能依赖 contracts / config，以及"下游依赖上游的纯函数"白名单
    layers = set(mini_quant.LAYERS)
    allowed = {("backtest", "portfolio"), ("execution", "contracts"), ("portfolio", "contracts"), ("monitor", "_paths")}
    violations = []
    for a in layers:
        for b in internal.get(a, set()):
            if b in layers and (a, b) not in allowed:
                violations.append(f"{a} ──▶ {b}")
    print("\n    🔍 规则检查：七个功能层之间不得互相 import（只能依赖 contracts / config / rules；"
          "白名单：backtest 复用 portfolio.drift_weights，保证回测与组合层口径一致）")
    print(f"      {'✅ 无违规依赖' if not violations else '❌ 违规：' + '、'.join(violations)}")
    print("      编排层 pipeline.py / research.py 负责把各层装配起来，只有它们可以依赖所有层")

    # 依赖矩阵图
    cols = order + sorted(EXTERNAL)
    mat = np.zeros((len(order), len(cols)))
    for i, a in enumerate(order):
        for b in internal.get(a, set()):
            if b in cols:
                mat[i, cols.index(b)] = 1
        for b in external.get(a, set()):
            mat[i, cols.index(b)] = 2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(mat, cmap=plt.matplotlib.colors.ListedColormap(["#f4f4f4", "#2a78d6", "#eb6834"]), vmin=0, vmax=2)
    ax.set_xticks(range(len(cols)), [c if c not in EXTERNAL else f"{c}\n({EXTERNAL[c]})" for c in cols],
                  rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(len(order) - 0.5, color="#888", lw=1)
    ax.set_title("mini_quant 依赖矩阵（行 import 列）：蓝 = 平台内部，橙 = 复用前面阶段")
    path = save_figure(fig, "01_dependency_matrix.png", subdir=mini_quant.SUBDIR)
    print(f"\n    📊 依赖矩阵已保存：{path}")


# ==================== 第五部分：最小端到端流程 ====================


def minimal_end_to_end():
    print("\n" + "=" * 60)
    print("第五部分：一键流水线——从数据更新到调仓指令")
    print("=" * 60)
    cfg = load_config()
    print(f"\n    📄 配置 config/strategy.yaml：模型 {cfg.model}，因子 {len(cfg.factors)} 个，"
          f"Top {cfg.n_top}，单票上限 {cfg.max_weight:.0%}，换手上限 {cfg.max_turnover:.0%}，资金 {cfg.capital:,.0f}")
    pipe = build_daily_pipeline(cfg)
    src = pipe.data
    print(f"    数据：{src.label()}")
    as_of = src.rebalance_dates()[-1]
    print(f"\n    ▶ run_daily({as_of.date()})：从空仓（现金 {cfg.capital:,.0f}）开始")
    res = pipe.run(as_of, pd.Series(dtype="int64"), cash=cfg.capital)
    steps = res.steps_frame()
    steps["seconds"] = steps["seconds"].round(3)
    print(steps.to_string(index=False))
    of = orders_frame(res.orders)
    print(f"\n    前 5 条调仓指令（共 {len(of)} 条，次日开盘执行）：")
    print(of.head().assign(target_weight=lambda d: d["target_weight"].map("{:.2%}".format),
                           est_fee=lambda d: d["est_fee"].round(2)).to_string(index=False))
    invested = float(of["est_value"].sum())
    print(f"\n    预估投入 {invested:,.0f} 元（{invested / cfg.capital:.1%}），预估费用 {of['est_fee'].sum():,.2f} 元，"
          f"剩余现金 {cfg.capital - invested - of['est_fee'].sum():,.0f} 元（取整 100 股 + 费用预留）")

    print("\n    ▶ 失败即停：要求生成数据之后的日期（模拟'数据源没有更新'）")
    late = src.calendar[-1] + pd.offsets.BDay(20)
    try:
        pipe.run(late, cash=cfg.capital)
    except PipelineError as exc:
        print(f"      ❌ PipelineError：{exc}")
        print("      " + " → ".join(f"{s.step}[{s.status}]" for s in exc.steps))
        print("      没有返回任何调仓指令：宁可今天不调仓，也不用过期数据生成错误指令")

    m = run_manifest(cfg, extra={"as_of": str(res.as_of.date()), "data": src.label()})
    print("\n    📋 本次运行的 manifest（可复现：代码哈希、包版本、配置）：")
    brief = {k: m[k] for k in ["git_commit", "code_sha256", "python", "packages", "as_of", "data"]}
    print("    " + manifest_json(brief).replace("\n", "\n    "))


def main():
    course_to_platform()
    layers_and_protocols()
    directory_layout()
    dependency_graph()
    minimal_end_to_end()
    print("""
    ✅ 小结：
      1. 七层 + 编排层；层间只通过 contracts.py 的 Protocol 与数据契约交互
      2. 复用 Phase 4 / 7 / 8 / 13 的模块，平台只写适配器
      3. 一键流水线每步校验、失败即停，失败时也留下步骤记录供日报使用
    下一课：数据层与因子层——因子注册表、预处理与单因子快速诊断。
    """)


if __name__ == "__main__":
    main()
