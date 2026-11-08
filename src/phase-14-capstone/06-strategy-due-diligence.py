"""
策略尽调与上线评审
==================

学习目标：
    1. 像审计师一样质疑自己的策略：把 CONCEPT 7.3 节的上线评审清单写成可自动执行的代码
    2. 自动检查：样本外长度、样本外超额显著性、DSR、成本敏感性、换手、容量、最大回撤、
       参数稳健性（参数平原 + 样本内外一致性），以及红旗信号扫描
    3. 每一项给出"通过 / 不通过 / 证据不足"及证据数字，汇总出评审结论
    4. 区分"能自动检查的"与"必须人工确认的"（独立复现、模拟盘、合规……）

核心概念：
    - 阈值必须事先写好（THRESHOLDS），评审时只代入证据，不临时调整
    - "证据不足"不是"差一点通过"：它意味着现在的数据无法支持上线决策
    - 尽调对象与 05 课研究报告完全一致（同一个 run_study、同一个选定配置）

前置知识：
    - 05 课（研究报告）、Phase 4（DSR）、Phase 11（容量）、Phase 13（模拟盘到实盘）
    - 本阶段 CONCEPT.md 第 7 节

环境要求：
    - 无网络时可用 QTC_OFFLINE=1 运行（先读本地缓存，缓存不存在才用合成数据）
"""

import sys
from dataclasses import dataclass
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR.parent))
sys.path.insert(0, str(PHASE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import mini_quant  # noqa: E402
from mini_quant import load_config  # noqa: E402
from mini_quant.backtest import capacity_at  # noqa: E402
from mini_quant.monitor import md_table, write_markdown  # noqa: E402
from mini_quant.research import ResearchContext, TrialLog, run_study  # noqa: E402

from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 170)

PASS, FAIL, INSUFFICIENT = "通过", "不通过", "证据不足"
ICON = {PASS: "✅", FAIL: "❌", INSUFFICIENT: "⚠️"}

# 事先写好的评审阈值（评审时不得修改）
THRESHOLDS = {
    "oos_months_pass": 24,  # 样本外 ≥ 24 个月
    "oos_months_min": 12,  # < 12 个月直接不通过
    "oos_t_pass": 2.0,  # 样本外超额 t ≥ 2
    "dsr_pass": 0.95,
    "dsr_min": 0.50,
    "cost_mult": 2.0,  # 双倍成本下样本外超额仍 > 0
    "annual_turnover_max": 6.0,  # 年化单边换手 ≤ 600%
    "cost_share_max": 0.5,  # 年化成本 ≤ 扣费前超额的 50%
    "capacity_mult": 10.0,  # 容量 ≥ 计划资金的 10 倍
    "mdd_abs_max": -0.30,  # 样本外最大回撤不深于 −30%
    "excess_mdd_ratio": 1.5,  # 样本外超额回撤不超过样本内的 1.5 倍
    "plateau_ratio": 0.5,  # 邻近参数的样本内 IR ≥ 所选的 50%
    "plateau_share": 0.8,  # 且至少 80% 的邻近参数满足
    "rho_fail": -0.3,  # 样本内外 IR 秩相关 ≤ −0.3 → 不通过
    "sharpe_red_flag": 3.0,  # 中低频策略回测 Sharpe > 3 → 红旗
}


@dataclass
class CheckResult:
    category: str
    item: str
    status: str
    evidence: str
    rule: str


# ==================== 收集证据 ====================


def collect_evidence(ctx: ResearchContext) -> dict:
    study = run_study(ctx, load_config(), TrialLog(mini_quant.OUT_DIR / "trials_due_diligence.jsonl"))
    run, cfg = study.run, study.cfg
    IS, OOS = run.summary("is"), run.summary("oos")
    gross_oos = run.gross("oos")
    bench_oos = run.bench_ret(period="oos")
    gross_ex = (1 + gross_oos).prod() ** (12 / len(gross_oos)) / (1 + bench_oos).prod() ** (12 / len(bench_oos)) - 1

    ct = study.cost_table[study.cost_table["区间"] == "样本外"].set_index("成本倍数")["超额年化(几何)"]
    slope, intercept = np.polyfit(ct.index.to_numpy(float), ct.to_numpy(float), 1)
    breakeven = -intercept / slope if slope < 0 else np.inf

    def excess_mdd(period):
        ex = (1 + run.excess(period=period)).cumprod()
        return float((ex / ex.cummax() - 1).min())

    model = cfg.model
    neighbors = study.grid[(study.grid["model"] == model) & (study.grid.index != study.best)]
    oos_ir = pd.Series({k: r.summary("oos")["信息比率"] for k, r in study.runs.items()})
    ex_is = run.excess(period="is")
    top3 = ex_is.nlargest(3)
    return {
        "study": study, "cfg": cfg, "IS": IS, "OOS": OOS, "data_label": ctx.source.label(),
        "oos_months": int(OOS["期数"]),
        "oos_excess": OOS["超额年化(几何)"], "oos_mean_excess": OOS["超额月均"], "oos_t": OOS["超额 t 值"],
        "oos_ir": OOS["信息比率"],
        "dsr": study.dsr["DSR"], "n_trials": study.dsr["试验次数"], "oos_psr": study.oos_psr,
        "cost_table": ct, "breakeven_mult": breakeven,
        "annual_turnover": OOS["年化单边换手"], "annual_cost": OOS["年化成本"], "gross_excess": gross_ex,
        "capacity": capacity_at(study.capacity, min(OOS["超额年化(几何)"], OOS["超额月均"] * 12)),
        "capital": cfg.capital,
        "oos_mdd": OOS["最大回撤"], "excess_mdd_is": excess_mdd("is"), "excess_mdd_oos": excess_mdd("oos"),
        "selected_is_ir": IS["信息比率"], "neighbors": neighbors,
        "is_oos_rho": float(study.grid["样本内信息比率"].corr(oos_ir, method="spearman")),
        "is_sharpe": IS["Sharpe"], "top3_share": float(top3.sum() / ex_is.sum()) if ex_is.sum() > 0 else np.nan,
        "ex_without_top3": float(ex_is.drop(top3.index).mean()),
        "avg_hold": float(run.frame["n_hold"].mean()),
    }


# ==================== 自动检查 ====================


def checklist(e: dict) -> list[CheckResult]:
    T = THRESHOLDS
    out = []

    m = e["oos_months"]
    s = PASS if m >= T["oos_months_pass"] else (INSUFFICIENT if m >= T["oos_months_min"] else FAIL)
    out.append(CheckResult("样本外", "样本外长度", s, f"{m} 个月", f"≥ {T['oos_months_pass']} 通过；< {T['oos_months_min']} 不通过"))

    ex, t, mex = e["oos_excess"], e["oos_t"], e["oos_mean_excess"]
    beat = ex > 0 and mex > 0  # 几何超额与算术月均超额都为正才算跑赢
    s = FAIL if not beat else (PASS if t >= T["oos_t_pass"] else INSUFFICIENT)
    out.append(CheckResult("样本外", "样本外超额显著性", s,
                           f"几何超额年化 {ex:.2%}，算术月均超额 {mex:.3%}，t = {t:.2f}，PSR = {e['oos_psr']:.3f}",
                           f"两种超额都 > 0 且 t ≥ {T['oos_t_pass']:g} 通过；任一 ≤ 0 不通过"))

    d = e["dsr"]
    s = PASS if d >= T["dsr_pass"] else (INSUFFICIENT if d >= T["dsr_min"] else FAIL)
    out.append(CheckResult("过拟合", "Deflated Sharpe", s, f"DSR = {d:.3f}（{e['n_trials']} 次试验）",
                           f"≥ {T['dsr_pass']} 通过；< {T['dsr_min']} 不通过"))

    ct = e["cost_table"]
    ex2 = float(ct.get(T["cost_mult"], np.nan))
    s = PASS if beat and ex2 > 0 else FAIL
    be = e["breakeven_mult"]
    out.append(CheckResult("成本", "成本敏感性", s,
                           f"{T['cost_mult']:g} 倍成本样本外超额 {ex2:.2%}；盈亏平衡约 {be:.1f} 倍成本" if np.isfinite(be)
                           else f"{T['cost_mult']:g} 倍成本样本外超额 {ex2:.2%}",
                           f"{T['cost_mult']:g} 倍成本下样本外超额 > 0"))

    to, cost, gex = e["annual_turnover"], e["annual_cost"], e["gross_excess"]
    share = cost / gex if gex > 0 else np.nan
    if to > T["annual_turnover_max"] or (share == share and share > T["cost_share_max"]):
        s = FAIL
    elif not gex > 0:
        s = INSUFFICIENT
    else:
        s = PASS
    out.append(CheckResult("成本", "换手与成本占比", s,
                           f"年化单边换手 {to:.0%}，年化成本 {cost:.2%}，占扣费前超额 {share:.0%}" if share == share
                           else f"年化单边换手 {to:.0%}，扣费前超额 ≤ 0",
                           f"换手 ≤ {T['annual_turnover_max']:.0%} 且成本 ≤ 扣费前超额的 {T['cost_share_max']:.0%}"))

    cap, capital = e["capacity"], e["capital"]
    need = T["capacity_mult"] * capital
    s = PASS if cap >= need else (INSUFFICIENT if cap >= capital else FAIL)
    cap_s = "∞" if np.isinf(cap) else f"{cap / 1e8:.2f} 亿元"
    out.append(CheckResult("容量", "容量估算（冲击吃掉一半超额）", s, f"{cap_s}；计划资金 {capital / 1e4:,.0f} 万元",
                           f"≥ {T['capacity_mult']:g} 倍计划资金通过；< 计划资金不通过（超额 ≤ 0 时容量记为 0）"))

    mdd, r_is, r_oos = e["oos_mdd"], e["excess_mdd_is"], e["excess_mdd_oos"]
    ok_abs = mdd >= T["mdd_abs_max"]
    ok_rel = r_oos >= T["excess_mdd_ratio"] * r_is
    out.append(CheckResult("风险", "最大回撤", PASS if ok_abs and ok_rel else FAIL,
                           f"样本外最大回撤 {mdd:.1%}；超额回撤 样本外 {r_oos:.1%} vs 样本内 {r_is:.1%}",
                           f"样本外回撤不深于 {T['mdd_abs_max']:.0%}，超额回撤不超过样本内的 {T['excess_mdd_ratio']:g} 倍"))

    nb = e["neighbors"]
    ratio = nb["样本内信息比率"] / e["selected_is_ir"] if e["selected_is_ir"] > 0 else pd.Series(0.0, index=nb.index)
    share_ok = float((ratio >= T["plateau_ratio"]).mean()) if len(nb) else np.nan
    out.append(CheckResult("过拟合", "参数平原（样本内邻近参数）", PASS if share_ok >= T["plateau_share"] else FAIL,
                           f"同模型 {len(nb)} 个邻近参数中 {share_ok:.0%} 的样本内 IR ≥ 所选的 {T['plateau_ratio']:.0%}"
                           f"（邻近 IR {nb['样本内信息比率'].min():.2f} ~ {nb['样本内信息比率'].max():.2f}，所选 {e['selected_is_ir']:.2f}）",
                           f"≥ {T['plateau_share']:.0%} 的邻近参数满足"))

    rho = e["is_oos_rho"]
    s = PASS if rho > 0 else (INSUFFICIENT if rho > T["rho_fail"] else FAIL)
    out.append(CheckResult("过拟合", "样本内外一致性（全部试验）", s, f"样本内 IR 与样本外 IR 的秩相关 ρ = {rho:.2f}",
                           f"ρ > 0 通过；ρ ≤ {T['rho_fail']} 不通过"))
    return out


def red_flags(e: dict) -> list[str]:
    flags = []
    if e["is_sharpe"] > THRESHOLDS["sharpe_red_flag"]:
        flags.append(f"样本内 Sharpe {e['is_sharpe']:.2f} > 3：检查前视偏差、成本遗漏")
    if e["top3_share"] == e["top3_share"] and e["top3_share"] > 0.5:
        flags.append(f"样本内超额高度集中：最好的 3 个月贡献了 {e['top3_share']:.0%} 的累计超额"
                     f"（去掉后月均超额 {e['ex_without_top3']:.2%}）")
    if e["selected_is_ir"] > 0 and e["oos_ir"] < 0.5 * e["selected_is_ir"]:
        flags.append(f"样本外显著差于样本内：IR {e['oos_ir']:.2f} vs {e['selected_is_ir']:.2f}")
    if e["is_oos_rho"] <= THRESHOLDS["rho_fail"]:
        flags.append(f"参数选择失效：样本内越好的配置样本外越差（ρ = {e['is_oos_rho']:.2f}）")
    if e["avg_hold"] < 15:
        flags.append(f"持仓集中：平均只持有 {e['avg_hold']:.1f} 只，个股事件风险大")
    return flags


MANUAL = [
    ("研究", "独立复现：另一个人（或另一套代码）能复现核心结果"),
    ("研究", "已与 Barra 等风格因子做归因，超额不是已知风格暴露"),
    ("模拟盘", "模拟盘 ≥ 4 周，与同期回测的偏差可解释（Phase 13）"),
    ("风险", "最大回撤止损线与停用条件已书面化（见研究报告第 9 节）"),
    ("风险", "与现有策略的相关性及组合层面影响已评估"),
    ("运营", "监控指标与告警已配置（04 课日报）；数据依赖稳定"),
    ("运营", "合规：程序化交易报告等要求已完成"),
]


def verdict(results: list[CheckResult]) -> str:
    statuses = [r.status for r in results]
    if FAIL in statuses:
        return FAIL
    if INSUFFICIENT in statuses:
        return INSUFFICIENT
    return PASS


# ==================== 课程主体 ====================


def explain():
    print("=" * 60)
    print("第一部分：尽调——像审计师一样质疑自己的策略")
    print("=" * 60)
    print("""
    作者天然对自己的策略"自信"。尽调的目的：在真金白银投入前，系统性地找出所有漏洞。

    评审对象：05 课研究报告中选定的配置（同一个 run_study，结果完全一致）
    评审方式：事先写好的阈值 THRESHOLDS → 代入证据 → 每项 通过 / 不通过 / 证据不足
    汇总规则：任一项不通过 → 不通过；否则任一项证据不足 → 证据不足；全部通过 → 通过（再进入人工项）
    """)
    print(pd.Series(THRESHOLDS).to_string())


def run_checks(e: dict):
    print("\n" + "=" * 60)
    print("第二部分：自动检查")
    print("=" * 60)
    print(f"    评审对象：{e['study'].best}（{e['n_trials']} 次试验中样本内信息比率最高）；数据：{e['data_label']}\n")
    results = checklist(e)
    for r in results:
        print(f"    {ICON[r.status]} [{r.category}] {r.item}：{r.status}")
        print(f"         证据：{r.evidence}")
        print(f"         规则：{r.rule}")
    flags = red_flags(e)
    print("\n    🚩 红旗信号扫描（CONCEPT 7.2 节）：")
    for f in flags or ["未发现"]:
        print(f"      - {f}")
    return results, flags


def evidence_horizon(e: dict):
    print("\n" + "=" * 60)
    print("第三部分：还需要多少证据？")
    print("=" * 60)
    ir = e["oos_ir"]
    if ir > 0:
        years = (THRESHOLDS["oos_t_pass"] / ir) ** 2
        print(f"""
    超额 t 值 ≈ IR × √年数。按当前样本外 IR = {ir:.2f}，要让 t ≥ 2 需要约 {years:.0f} 年的样本外记录。
    反过来，24 个月的样本外要达到 t ≥ 2，真实 IR 需要 ≥ {2 / np.sqrt(e['oos_months'] / 12):.2f}。
    → 低 IR 的策略几乎不可能"靠时间证明自己"；更有效的做法是扩大截面（更多股票、更多市场）、
      降低噪声（更好的数据、去掉幸存者偏差），或寻找经济逻辑更强的信号。
    """)
    else:
        print(f"\n    样本外 IR = {ir:.2f} ≤ 0：更长的样本外不会改变结论，需要重新研究信号。")


def manual_items():
    print("\n" + "=" * 60)
    print("第四部分：必须人工确认的项目")
    print("=" * 60)
    for cat, item in MANUAL:
        print(f"    □ [{cat}] {item}")
    print("""
    这些项目无法从回测数据自动判断，必须由人签字确认。自动检查全部通过只是"可以进入人工评审"，
    而不是"可以上线"。
    """)


def write_dd_report(e: dict, results, flags, final: str) -> Path:
    df = pd.DataFrame([r.__dict__ for r in results])
    df["status"] = df["status"].map(lambda s: f"{ICON[s]} {s}")
    df = df.rename(columns={"category": "维度", "item": "检查项", "status": "结论", "evidence": "证据", "rule": "规则"})
    L = [f"# 上线评审报告：{e['study'].best}", "",
         f"> 由 `06-strategy-due-diligence.py` 自动生成 · 数据：{e['data_label']} · 与 research_report.md 评审同一配置", "",
         f"## 结论：{ICON[final]} {final}", "",
         "## 自动检查", "", md_table(df.set_index("维度")), "",
         "## 红旗信号", ""] + [f"- {f}" for f in flags or ["未发现"]] + [
         "", "## 人工确认项（未勾选）", ""] + [f"- [ ] [{c}] {i}" for c, i in MANUAL] + [
         "", "## 评审阈值（事先确定）", "", "```", pd.Series(THRESHOLDS).to_string(), "```", ""]
    return write_markdown("\n".join(L), "due_diligence.md")


def plot(e: dict):
    st = e["study"]
    oos_ir = pd.Series({k: r.summary("oos")["信息比率"] for k, r in st.runs.items()})
    is_ir = st.grid["样本内信息比率"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"equal": "#888", "icw": "#2a78d6", "ridge": "#1baf7a", "hgb": "#eb6834"}
    for m, c in colors.items():
        idx = st.grid.index[st.grid["model"] == m]
        axes[0].scatter(is_ir[idx], oos_ir[idx], color=c, label=m)
    axes[0].scatter([is_ir[st.best]], [oos_ir[st.best]], s=160, facecolors="none", edgecolors="k", label="所选配置")
    axes[0].axhline(0, color="#aaa", lw=0.8)
    axes[0].axvline(0, color="#aaa", lw=0.8)
    axes[0].set_xlabel("样本内信息比率")
    axes[0].set_ylabel("样本外信息比率")
    axes[0].set_title(f"24 次试验：样本内 vs 样本外（秩相关 {e['is_oos_rho']:.2f}）")
    axes[0].legend(fontsize=8)
    ct = e["cost_table"]
    axes[1].plot(ct.index, ct.values * 100, marker="o", color="#2a78d6")
    axes[1].axhline(0, color="#aaa", lw=0.8)
    axes[1].set_xlabel("成本倍数")
    axes[1].set_ylabel("样本外超额年化 (%)")
    axes[1].set_title("成本敏感性（样本外）")
    return save_figure(fig, "06_due_diligence.png", subdir=mini_quant.SUBDIR)


def main():
    explain()
    ctx = ResearchContext()
    e = collect_evidence(ctx)
    results, flags = run_checks(e)
    evidence_horizon(e)
    manual_items()
    final = verdict(results)
    counts = pd.Series([r.status for r in results]).value_counts().to_dict()
    print("=" * 60)
    print(f"评审结论：{ICON[final]} {final}（自动检查 {len(results)} 项：{counts}）")
    print("=" * 60)
    path = write_dd_report(e, results, flags, final)
    fig = plot(e)
    print(f"    📝 评审报告：{path}")
    print(f"    📊 图表：{fig}")
    print("""
    ✅ 小结：
      1. 阈值事先写好，评审时只代入证据；结论由规则汇总
      2. "证据不足"与"不通过"都意味着不能上线；前者需要更多 / 更好的数据，后者需要重新研究
      3. 自动检查通过后，仍有独立复现、模拟盘、合规等人工项
    下一课：量化岗位面试——概率、统计、编程、策略题。
    """)


if __name__ == "__main__":
    main()
