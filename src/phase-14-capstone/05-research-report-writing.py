"""
量化研究报告写作
================

学习目标：
    1. 按"摘要 → 假设 → 数据 → 方法 → 成本 → 结果 → 稳健性 → 试验次数 → 容量 → 风险 → 局限 → 结论"
       的结构，自动生成一份可复现的策略研究报告（outputs/phase-14/research_report.md）
    2. 遵守研究纪律：先写假设；参数网格的每一次尝试都记入试验日志；只用样本内指标选参；
       样本外只评估一次；用 Deflated Sharpe 校正试验次数
    3. 报告中的每一个数字都来自本次运行的计算，结论由规则从证据推出——
       样本外不赚钱就如实写"不建议上线"
    4. 练习"5 分钟讲清一个策略"：收益来源、证据、失效条件、建议

核心概念：
    - 五项必须披露：样本区间、成本假设、基准对比、样本外结果、试验次数
    - 结论不能超出证据：24 个月的样本外很难提供强证据（07 课题 10）
    - 可复现：manifest 记录代码哈希、数据哈希、包版本与完整配置

前置知识：
    - 01–04 课（mini-quant 平台）、Phase 4（DSR）、Phase 11（平方根冲击）
    - 本阶段 CONCEPT.md 第 6 节

环境要求：
    - 无网络时可用 QTC_OFFLINE=1 运行（先读本地缓存，缓存不存在才用合成数据）
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR.parent))
sys.path.insert(0, str(PHASE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import mini_quant  # noqa: E402
from mini_quant import load_config, manifest_json, run_manifest  # noqa: E402
from mini_quant.backtest import capacity_at, cost_rates, perf_summary  # noqa: E402
from mini_quant.data import PANEL_FILES  # noqa: E402
from mini_quant.factors import ic_series  # noqa: E402
from mini_quant.monitor import md_table, sector_exposure, write_markdown  # noqa: E402
from mini_quant.research import BENCH_NAMES, ResearchContext, TrialLog, run_study, yearly_table  # noqa: E402

from bt_core import CostModel  # noqa: E402
import factor_data  # noqa: E402
import factor_tools  # noqa: E402
from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 170)

REPORT_FILE = "research_report.md"
FIG_FILE = "05_research_nav.png"

# 事前写下的假设（在看回测结果之前）：因子 → (类别, 经济逻辑, 收益来源类型)
HYPOTHESES = {
    "ep": ("价值", "盈利收益率高 = 相对盈利便宜；投资者对差公司过度悲观、对热门公司过度外推", "行为偏差 / 风险补偿"),
    "bp": ("价值", "账面市值比高的公司困境风险更高，要求更高回报", "风险补偿"),
    "roe": ("质量", "高盈利能力的公司被系统性低估（分析师与投资者对持续性反应不足）", "行为偏差"),
    "mom_12_1": ("动量", "信息逐步扩散、反应不足；跳过最近 1 个月以避开短期反转", "行为偏差"),
    "rev_1m": ("反转", "A 股散户占比高，短期过度反应后回归；也包含流动性提供的补偿", "行为偏差 / 流动性提供"),
    "lowvol": ("低波", "彩票偏好与杠杆约束使高波动股票被高估（低波动异象）", "行为偏差 / 结构性约束"),
    "lowturn": ("流动性", "高换手常伴随过度交易与关注度溢价，低换手股票被忽视", "行为偏差"),
    "size": ("规模", "小市值的流动性与信息风险补偿；A 股还有壳价值（2017 年后减弱）", "风险补偿 / 结构性"),
}


def pct(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}%}"


def num(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def rho_text(rho: float) -> str:
    if rho <= -0.3:
        return "样本内越好的配置，样本外反而越差——典型的过拟合 / 选择偏差信号（也可能是风格在两段之间切换）"
    if rho < 0.3:
        return "样本内排序对样本外几乎没有预测力，选参主要在拟合噪声"
    return "样本内排序对样本外有一定预测力，但仍需考虑试验次数"


def date(x) -> str:
    return "—" if x is None or x is pd.NaT or (isinstance(x, float) and np.isnan(x)) else pd.Timestamp(x).date().isoformat()


# ==================== 计算：所有数字的唯一来源 ====================


def compute(ctx: ResearchContext):
    base = load_config()
    log = TrialLog()
    study = run_study(ctx, base, log)
    run, cfg = study.run, study.cfg
    out = {"study": study, "run": run, "cfg": cfg, "trial_log": log}
    out["is"], out["oos"] = run.summary("is"), run.summary("oos")
    rows = {}
    for period, label in (("is", "样本内"), ("oos", "样本外")):
        rows[(label, "策略（扣费）")] = perf_summary(run.net(period), run.bench_ret("ew_universe", period), factor_data.RF_ANNUAL)
        rows[(label, "策略（扣费前）")] = perf_summary(run.gross(period), run.bench_ret("ew_universe", period), factor_data.RF_ANNUAL)
        for b in ("ew_universe", "hs300", "zz500"):
            rows[(label, BENCH_NAMES[b])] = perf_summary(run.bench_ret(b, period), rf_annual=factor_data.RF_ANNUAL)
    out["bench_table"] = rows
    yearly = yearly_table(run)
    yearly.index = yearly.index.astype(str)
    out["yearly"] = yearly
    oos_ir = pd.Series({k: r.summary("oos")["信息比率"] for k, r in study.runs.items()})
    is_ir = study.grid["样本内信息比率"]
    out["oos_rank"] = int(oos_ir.rank(ascending=False)[study.best])
    out["oos_best"] = (oos_ir.idxmax(), oos_ir.max(), is_ir[oos_ir.idxmax()])
    out["is_oos_rho"] = float(is_ir.corr(oos_ir, method="spearman"))
    clean = ctx.engine(cfg).panel(ctx.dates)
    ics = ic_series(clean, ctx.period_ret)
    out["factor_ic"] = pd.DataFrame({"样本内IC": ics.loc[cfg.is_start : cfg.is_end].mean(),
                                     "样本外IC": ics.loc[cfg.oos_start : cfg.oos_end].mean()})
    out["score_ic"] = {p: ctx.score_ic(run.scores).loc[a:b].mean()
                       for p, a, b in (("is", cfg.is_start, cfg.is_end), ("oos", cfg.oos_start, cfg.oos_end))}
    # 容量按更保守的超额口径：几何超额与算术月均超额×12 取小
    alpha = min(out["oos"]["超额年化(几何)"], out["oos"]["超额月均"] * 12)
    out["capacity_half"] = capacity_at(study.capacity, alpha)
    ex_oos = run.excess(period="oos")
    out["excess_mdd_oos"] = float(((1 + ex_oos).cumprod() / (1 + ex_oos).cumprod().cummax() - 1).min())
    ex_is = run.excess(period="is")
    out["excess_mdd_is"] = float(((1 + ex_is).cumprod() / (1 + ex_is).cumprod().cummax() - 1).min())
    out["worst_months"] = run.net().nsmallest(3)
    last = run.bt.weights[max(run.bt.weights)]
    out["exposure"] = sector_exposure(last, ctx.source.sector, ctx.source.codes)
    out["avg_hold"] = float(run.frame["n_hold"].mean())
    out["manifest"] = run_manifest(cfg, data_files=PANEL_FILES, extra={
        "data": ctx.source.label(), "trial_log": str(log.path.name), "n_trials": len(log),
        "selected": study.best})
    out["conclusion"] = conclude(out)
    return out


def conclude(o: dict) -> dict:
    """由证据推出结论（规则事先写好，而不是看完结果再决定怎么写）"""
    oos, dsr = o["oos"], o["study"].dsr["DSR"]
    ex, t, mean_ex = oos["超额年化(几何)"], oos["超额 t 值"], oos["超额月均"]
    rf = factor_data.RF_ANNUAL
    reasons = []
    if not (ex > 0 and mean_ex > 0):  # 几何超额与算术超额（IR、t 值的口径）都为正才算跑赢
        verdict = "不建议上线"
        reasons.append(f"样本外扣费后没有稳定跑赢股票池等权基准（几何超额年化 {pct(ex)}，"
                       f"算术月均超额 {pct(mean_ex, 3)}，t = {num(t)}）")
    elif dsr >= 0.95 and t >= 2:
        verdict = "建议进入模拟盘观察（≥ 4 周），通过上线评审后小资金试运行"
        reasons.append(f"样本外超额 {pct(ex)}，t = {num(t)}；DSR = {num(dsr, 3)} ≥ 0.95")
    else:
        verdict = "不建议上线（证据不足）"
        reasons.append(f"样本外超额为正（{pct(ex)}），但统计上不显著：t = {num(t)}，样本外 PSR = {num(o['study'].oos_psr, 3)}")
        if dsr < 0.95:
            reasons.append(f"考虑 {o['study'].dsr['试验次数']} 次试验后，样本内 DSR = {num(dsr, 3)} < 0.95")
    if oos["年化收益"] < rf:
        reasons.append(f"样本外绝对年化收益 {pct(oos['年化收益'])} 低于无风险利率假设 {pct(rf)}")
    decay = oos["信息比率"] / o["is"]["信息比率"] if o["is"]["信息比率"] > 0 else np.nan
    if oos["信息比率"] <= 0 < o["is"]["信息比率"]:
        reasons.append(f"样本外信息比率 {num(oos['信息比率'])} ≤ 0（样本内 {num(o['is']['信息比率'])}），疑似过拟合或市场结构变化")
    elif decay == decay and decay < 0.5:
        reasons.append(f"样本外信息比率只有样本内的 {pct(decay, 0)}（{num(oos['信息比率'])} vs {num(o['is']['信息比率'])}），疑似过拟合或市场结构变化")
    return {"verdict": verdict, "reasons": reasons, "ir_decay": decay}


# ==================== 图表 ====================


def make_figure(o: dict) -> Path:
    run, cfg = o["run"], o["cfg"]
    nav = (1 + run.net()).cumprod()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for b, color in (("ew_universe", "#888"), ("hs300", "#eb6834")):
        axes[0].plot((1 + run.bench_ret(b)).cumprod(), color=color, ls="--", label=BENCH_NAMES[b])
    axes[0].plot(nav, color="#2a78d6", lw=2, label=f"策略（扣费）{cfg.short()}")
    for ax in axes:
        ax.axvspan(pd.Timestamp(cfg.oos_start), nav.index[-1], color="#f0f0f0")
    axes[0].text(pd.Timestamp(cfg.oos_start), axes[0].get_ylim()[1] * 0.95, " 样本外", color="#555")
    axes[0].set_title("净值（月频，横轴为信号日）")
    axes[0].legend(fontsize=8)
    ex = (1 + run.excess()).cumprod()
    axes[1].plot(ex, color="#1baf7a")
    axes[1].axhline(1, color="#888", lw=0.8)
    axes[1].set_title("相对股票池等权的累计超额")
    return save_figure(fig, FIG_FILE, subdir=mini_quant.SUBDIR)


# ==================== 报告 ====================


def bench_md(o: dict) -> str:
    cols = ["年化收益", "年化波动", "Sharpe", "最大回撤", "超额年化(几何)", "信息比率", "超额 t 值"]
    rows = []
    for (period, name), s in o["bench_table"].items():
        rows.append([period, name] + [s.get(c, np.nan) for c in cols])
    df = pd.DataFrame(rows, columns=["区间", "组合"] + cols).set_index("区间")
    f = {c: pct for c in ["年化收益", "年化波动", "最大回撤", "超额年化(几何)"]}
    f.update({c: num for c in ["Sharpe", "信息比率", "超额 t 值"]})
    return md_table(df, formats=f)


def grid_md(study) -> str:
    g = study.grid.copy()
    oos_ir = {k: r.summary("oos")["信息比率"] for k, r in study.runs.items()}
    g["样本外信息比率（事后检查，未用于选择）"] = pd.Series(oos_ir)
    g = g[["model", "n_top", "max_turnover", "样本内信息比率", "样本内超额年化", "月均单边换手",
           "样本外信息比率（事后检查，未用于选择）"]]
    g.index.name = "试验"
    f = {"样本内信息比率": num, "样本外信息比率（事后检查，未用于选择）": num, "样本内超额年化": pct,
         "月均单边换手": pct, "max_turnover": lambda v: f"{v:.0%}"}
    return md_table(g, formats=f)


def write_report(ctx: ResearchContext, o: dict, fig_path: Path) -> Path:
    st, run, cfg = o["study"], o["run"], o["cfg"]
    src, p = ctx.source, ctx.source.panel
    IS, OOS, dsr, concl = o["is"], o["oos"], st.dsr, o["conclusion"]
    buy, sell_old = cost_rates(CostModel(slippage=cfg.slippage), pd.Timestamp("2023-01-31"))
    _, sell_new = cost_rates(CostModel(slippage=cfg.slippage), pd.Timestamp("2023-09-30"))
    n_is, n_oos = int(IS["期数"]), int(OOS["期数"])
    cap = o["capacity_half"]
    cap_s = "无法估算（样本外超额 ≤ 0，容量记为 0）" if cap == 0 else ("> 100 亿元（超出测算范围）" if np.isinf(cap) else f"约 {cap / 1e8:.2f} 亿元")
    capacity = st.capacity.iloc[::2].copy()
    capacity["资金规模"] = capacity["资金规模"].map(lambda v: f"{v / 1e4:,.0f} 万")
    capacity = capacity.set_index("资金规模")
    cost = st.cost_table.copy()
    cost["成本倍数"] = cost["成本倍数"].map(lambda v: f"{v:g}×")
    cost = cost.set_index("成本倍数")
    yearly = o["yearly"].loc[str(pd.Timestamp(cfg.is_start).year) : str(pd.Timestamp(cfg.oos_end).year)]
    fic = o["factor_ic"].copy()
    fic["方向是否一致"] = np.where(np.sign(fic["样本内IC"]) == np.sign(fic["样本外IC"]), "一致", "反转")
    dd_oos = run.summary("oos")
    worst = "；".join(f"{date(d)} 信号期 {pct(v)}" for d, v in o["worst_months"].items())
    synth_warn = ("\n> ⚠️ **本报告基于合成数据**，只用于演示报告生成流程，不能得出任何关于真实市场的结论。\n"
                  if src.is_synthetic else "")

    L = []
    L += [f"# 策略研究报告：A 股多因子月度选股（{cfg.short()}）", "",
          f"> 由 `src/phase-14-capstone/05-research-report-writing.py` 自动生成 · 数据：{src.label()} · "
          f"代码哈希 `{o['manifest']['code_sha256']}` · 试验日志 `outputs/phase-14/{o['manifest']['trial_log']}`",
          synth_warn,
          "## 摘要", "",
          f"在 {len(src.codes)} 只 A 股（当前沪深 300 + 中证 500 成分股按行业抽样）上，用 {len(cfg.factors)} 个经行业与市值中性化的因子、"
          f"滚动训练的 `{cfg.model}` 模型打分，每月末选分数最高的 {cfg.n_top} 只等权持有（单边换手上限 {cfg.max_turnover:.0%}）。"
          f"该配置是 {dsr['试验次数']} 次参数试验中**样本内**信息比率最高的一个。",
          "",
          f"- **样本内**（{cfg.is_start[:7]} ~ {cfg.is_end[:7]}，{n_is} 期）：扣费后年化 {pct(IS['年化收益'])}，"
          f"相对股票池等权超额年化 {pct(IS['超额年化(几何)'])}，信息比率 {num(IS['信息比率'])}",
          f"- **样本外**（{cfg.oos_start[:7]} ~ {cfg.oos_end[:7]}，{n_oos} 期）：扣费后年化 {pct(OOS['年化收益'])}，"
          f"超额年化 {pct(OOS['超额年化(几何)'])}，信息比率 {num(OOS['信息比率'])}，超额 t 值 {num(OOS['超额 t 值'])}",
          f"- **多重检验**：{dsr['试验次数']} 次试验下样本内 Deflated Sharpe = {num(dsr['DSR'], 3)}；样本外超额 PSR(>0) = {num(st.oos_psr, 3)}",
          f"- **结论：{concl['verdict']}**。主要原因：{'；'.join(concl['reasons'])}。",
          "",
          f"![净值]({fig_path.name})", ""]

    L += ["## 1. 假设与经济逻辑", "",
          "以下假设在查看任何回测结果之前写下（因子方向统一为\"值越大，预期收益越高\"）：", "",
          "| 因子 | 类别 | 经济逻辑 | 收益来源 |", "|---|---|---|---|"]
    L += [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in HYPOTHESES.items() if k in cfg.factors]
    L += ["", "组合逻辑：单因子信号弱且不稳定（02 课诊断中没有一个因子 |t| ≥ 2），合成多个低相关的异象以分散单一因子失效的风险；"
          "模型层只决定各因子的权重，不引入新的信息来源。", "",
          "**为什么还没被套利掉**：价值、低波、反转等异象在文献中长期存在，但收益并不稳定，有较长的失效期（风格切换），"
          "且部分收益是对风险或流动性的补偿；A 股散户占比高、做空受限，也使部分错误定价持续更久。这些都是假设，"
          "需要样本外证据支持。", ""]

    L += ["## 2. 数据与样本区间", "",
          "| 项目 | 说明 |", "|---|---|",
          f"| 来源 | Phase 7 截面面板（AKShare：行情、股本、财报、实际披露日）；{src.label()} |",
          f"| 股票池 | {len(src.codes)} 只，{src.info['sector'].nunique()} 个行业，板块 {src.info['board'].value_counts().to_dict()} |",
          f"| 行情区间 | {p.dates[0].date()} ~ {p.dates[-1].date()}（{len(p.dates)} 个交易日） |",
          f"| 信号日 | 每月最后一个交易日，{ctx.dates[0].date()} ~ {ctx.dates[-1].date()}（{len(ctx.dates)} 期） |",
          f"| 训练 | 2019 年起的标签用于滚动训练（窗口 {cfg.train_window} 期，至少 {cfg.min_train} 期，标签滞后 2 期） |",
          f"| 样本内 | {cfg.is_start} ~ {cfg.is_end}（信号日，{n_is} 期）：只在这里比较与选择参数 |",
          f"| 样本外 | {cfg.oos_start} ~ {cfg.oos_end}（信号日，{n_oos} 期）：选定参数后只评估一次 |",
          "| PIT | 财报按首次实际披露日对齐，公告当天不用（`factor_tools.pit_merge`） |",
          "| 可交易性 | 剔除上市不满一年、信号日停牌、推断为 ST、次日开盘涨停（买不进）的股票；停牌持仓保持原权重 |",
          "| 已知偏差 | 股票池为**当前**指数成分股 → 幸存者偏差与成分股前视偏差；行业为当前分类 |", "",
          "数据质量检查（2019–2024）：", "",
          md_table(src.quality_report("2019-01-01", "2024-12-31").set_index("检查项")), ""]

    L += ["## 3. 方法", "",
          f"1. **因子**：{', '.join(cfg.factors)}（Phase 7 `build_factors`，经 mini-quant 因子注册表启用）",
          "2. **预处理**：逐期截面 MAD 去极值（3 倍）→ Z-score → 对行业哑变量 + 对数流通市值回归取残差 → 再 Z-score（size 只做行业中性）",
          f"3. **模型**：`{cfg.model}`；标签 = 持有期收益的截面百分位排名 − 0.5；每 {cfg.retrain_every} 期用最近 {cfg.train_window} 期已实现标签重训",
          f"4. **组合**：分数最高的 {cfg.n_top} 只等权；单边换手上限 {cfg.max_turnover:.0%}（w = 当前 + λ(目标 − 当前)）；单票上限 {cfg.max_weight:.0%}",
          "5. **执行假设**：t 日收盘出信号，t+1 日开盘成交，持有到下一信号日的次日开盘；持有期内权重随价格漂移",
          "6. **基准**：股票池等权（每期对可交易股票等权，**不扣费**，衡量选股能力）；另列沪深 300、中证 500 供参考", ""]

    L += ["## 4. 成本假设", "",
          "| 项目 | 取值 |", "|---|---|",
          "| 佣金 | 0.025%，双边（向量化回测不含 5 元最低佣金；逐笔指令的预估费用含最低佣金） |",
          "| 过户费 | 0.001%，双边 |",
          "| 印花税 | 卖出单边：2023-08-28 前 0.1%，之后 0.05%（按当时费率） |",
          f"| 滑点 / 冲击 | 单边 {cfg.slippage:.2%}（中小盘取 10bp） |",
          f"| 合计单边费率 | 买入 {buy:.3%}；卖出 {sell_old:.3%}（2023-08-28 前）/ {sell_new:.3%}（之后） |",
          f"| 结果 | 样本内年化成本 {pct(IS['年化成本'])}、样本外 {pct(OOS['年化成本'])}；月均单边换手 {pct(IS['月均单边换手'])} / {pct(OOS['月均单边换手'])} |",
          "", "资金规模带来的额外冲击成本见第 8 节。", ""]

    L += ["## 5. 结果", "", "### 5.1 基准对比", "",
          "超额相关指标均相对\"股票池等权\"；Sharpe 已扣除 2% 无风险利率假设。", "", bench_md(o), "",
          "### 5.2 分年度（按信号日所在年份；持有期为信号日次月）", "",
          md_table(yearly, formats={c: pct for c in yearly.columns}), "",
          "### 5.3 样本内 vs 样本外", "",
          "| 指标 | 样本内 | 样本外 |", "|---|---|---|",
          f"| 模型分数 Rank IC 均值 | {num(o['score_ic']['is'], 4)} | {num(o['score_ic']['oos'], 4)} |",
          f"| 超额年化（几何） | {pct(IS['超额年化(几何)'])} | {pct(OOS['超额年化(几何)'])} |",
          f"| 超额月均（算术，IR 与 t 值的口径） | {pct(IS['超额月均'], 3)} | {pct(OOS['超额月均'], 3)} |",
          f"| 信息比率 | {num(IS['信息比率'])} | {num(OOS['信息比率'])} |",
          f"| 超额月胜率 | {pct(IS['超额月胜率'], 1)} | {pct(OOS['超额月胜率'], 1)} |",
          f"| 平均持仓数 | {o['avg_hold']:.1f}（全区间） | |", "",
          f"注：目标是 Top {cfg.n_top}，但换手上限只允许每期朝目标移动一部分，新旧持仓并存，所以实际平均持仓数多于 {cfg.n_top}。", "",
          "几何超额 = 策略与基准各自复利后的年化差，受两者波动率不同的影响（波动拖累）；算术月均超额是 IR 和 t 值的口径。"
          "两者符号不一致时，说明超额不稳定，不能算\"跑赢\"。", ""]

    L += ["## 6. 稳健性", "", "### 6.1 参数网格（全部试验）", "",
          "选择规则（事先确定）：样本内信息比率最高者。样本外一列只在选择完成后用于检查，未参与选择。", "",
          grid_md(st), "",
          f"样本内信息比率 > 0 的试验占 {pct((st.grid['样本内信息比率'] > 0).mean(), 0)}。"
          f"所选配置的样本外信息比率在 {len(st.grid)} 个试验中排第 {o['oos_rank']}；样本外最好的是 `{o['oos_best'][0]}`"
          f"（样本外 IR {num(o['oos_best'][1])}，但样本内 IR 只有 {num(o['oos_best'][2])}）。"
          f"全部试验的样本内与样本外信息比率的秩相关为 {num(o['is_oos_rho'])}："
          f"{rho_text(o['is_oos_rho'])}。**不能据此改选样本外最好的配置**——那等于把样本外当成了样本内。", "",
          "### 6.2 成本敏感性", "", md_table(cost, formats={"年化收益": pct, "超额年化(几何)": pct, "信息比率": num}), "",
          "### 6.3 单因子在样本内外的表现（Rank IC）", "",
          md_table(fic, formats={"样本内IC": lambda v: num(v, 4), "样本外IC": lambda v: num(v, 4)}), ""]

    L += ["## 7. 试验次数与 Deflated Sharpe", "",
          "| 指标 | 数值 |", "|---|---|",
          f"| 记录的试验次数 | {dsr['试验次数']}（4 种模型 × 3 种持仓数 × 2 种换手上限，见附录 A） |",
          f"| 所选试验样本内超额 Sharpe（月 / 年化） | {num(dsr['观测 Sharpe(月)'], 3)} / {num(dsr['观测 Sharpe(年化)'])} |",
          f"| 试验间 Sharpe 标准差（月） | {num(dsr['试验 Sharpe 标准差(月)'], 3)} |",
          f"| 无技能时的期望最大 Sharpe（月 / 年化） | {num(dsr['期望最大 Sharpe(月)'], 3)} / {num(dsr['期望最大 Sharpe(年化)'])} |",
          f"| PSR（样本内 SR > 0） | {num(dsr['PSR(SR>0)'], 3)} |",
          f"| **DSR（样本内 SR > 期望最大 SR）** | **{num(dsr['DSR'], 3)}** |",
          f"| 样本外超额 PSR（SR > 0） | {num(st.oos_psr, 3)} |", "",
          "说明：DSR 衡量的是收益相对\"股票池等权\"的超额序列；选择发生在样本内，所以对样本内结果做多重检验校正。"
          "01–04 课的探索（如 03 课的模型与换手比较）也看过样本内结果，**实际尝试次数不少于上表**，DSR 只会更低。", ""]

    L += ["## 8. 容量（平方根冲击模型）", "",
          "冲击成本 ≈ σ_日 × √(单票成交额 / 20 日均成交额)，Y = 1；按样本外实际调仓的权重变化、假设每次调仓在一天内完成计算，"
          "未计入执行算法拆单的改善。超额取几何超额与算术月均超额 × 12 中较小者。"
          f"冲击成本吃掉一半样本外超额的规模：**{cap_s}**。", "",
          md_table(capacity, formats={"年化冲击拖累": pct, "最大单票参与率": pct}), ""]

    dd = run.summary("all")
    L += ["## 9. 风险与失效条件", "",
          "| 指标 | 数值 |", "|---|---|",
          f"| 全区间最大回撤 | {pct(dd['最大回撤'])}（{date(dd['回撤起点'])} → {date(dd['回撤谷底'])}，信号日） |",
          f"| 样本外最大回撤 | {pct(dd_oos['最大回撤'])}（{date(dd_oos['回撤起点'])} → {date(dd_oos['回撤谷底'])}） |",
          f"| 超额最大回撤（样本内 / 样本外） | {pct(o['excess_mdd_is'])} / {pct(o['excess_mdd_oos'])} |",
          f"| 最差 3 个月 | {worst} |",
          f"| 跟踪误差（样本内 / 样本外） | {pct(IS['跟踪误差'])} / {pct(OOS['跟踪误差'])} |", "",
          "最新一期组合的行业暴露（相对股票池等权）：", "",
          md_table(o["exposure"], pct_cols={"组合", "股票池等权", "偏离"}), "",
          "**可能失效的环境**：",
          "- 风格急剧切换（如小盘 / 微盘行情或题材炒作），价值、低波类因子会显著跑输；",
          "- 单因子 IC 在样本内外方向反转（见 6.3），合成权重依赖近期 IC，切换时会滞后；",
          "- 因子拥挤时超额衰减，且回撤更深；持仓数少（集中度高）时个股事件风险大。", "",
          "**事先写好的停用条件**（上线后执行，避免事后找理由）：",
          f"- 实盘回撤超过回测最大回撤的 1.5 倍（{pct(1.5 * dd['最大回撤'])}）；",
          "- 滚动 12 个月超额 Sharpe 的 95% 置信区间不再包含回测值；",
          "- 合成分数的 Rank IC 连续 6 个月为负，或市场规则发生实质变化。", ""]

    L += ["## 10. 局限性", "",
          "1. **幸存者偏差**：股票池是当前成分股，历史上被剔除或退市的股票不在样本中，收益很可能被高估；",
          f"2. **样本小**：{len(src.codes)} 只股票、Top {cfg.n_top} 持仓，截面分散度有限；样本外只有 {n_oos} 个月（夏普的标准误约 ±0.7，见 07 课题 10）；",
          "3. **成本简化**：向量化回测按比例扣费，未含最低佣金；冲击成本只在容量测算中估计；",
          "4. **可交易性近似**：ST 用涨跌停幅度反推；停牌股按最后收盘价估值；未考虑跌停卖不出；",
          "5. **执行模拟**：MockExchange 用开盘价和\"成交量 × 2%\"的深度近似，不代表真实盘口；",
          "6. **行业分类**为当前分类，历史行业变更未体现；财报数值可能是事后修订版；",
          "7. **未与已知风险模型做归因**（如 Barra 风格因子），超额可能部分来自风格暴露。", ""]

    L += ["## 11. 结论与建议", "",
          f"**{concl['verdict']}。**", ""]
    L += [f"- {r}" for r in concl["reasons"]]
    L += ["", "建议的下一步：",
          "- 不改动参数，继续用新数据做样本外跟踪（滚动更新本报告），积累至少 12 个月新的样本外记录；",
          "- 使用含退市股票的历史成分股重建股票池，消除幸存者偏差后重新评估；",
          "- 与 Barra 风格因子做归因，确认超额不是已知风格暴露；",
          "- 通过 06 课上线评审清单后，再考虑模拟盘（≥ 4 周）与小资金试运行。", ""]

    trials = pd.DataFrame([{"#": r["trial"], "配置": r["label"], **{k: v for k, v in r["metrics"].items()}}
                           for r in o["trial_log"].records]).set_index("#")
    L += ["## 附录 A：试验记录（样本内指标）", "",
          md_table(trials, formats={"样本内信息比率": num, "样本内超额年化": pct, "样本内年化收益": pct,
                                    "样本内Sharpe": num, "月均单边换手": pct}), "",
          "## 附录 B：运行清单（manifest）", "", "```json", manifest_json(o["manifest"]), "```", "",
          "## 附录 C：复现方式", "", "```bash",
          "cd src/phase-14-capstone",
          "python 05-research-report-writing.py            # 联网 / 有缓存时使用真实数据",
          "QTC_OFFLINE=1 python 05-research-report-writing.py   # 离线：先读本地缓存，缓存不存在才用合成数据",
          "```", ""]
    return write_markdown("\n".join(L), REPORT_FILE)


# ==================== 课程讲解 ====================


def explain_structure():
    print("=" * 60)
    print("第一部分：研究报告的结构与研究纪律")
    print("=" * 60)
    print("""
    报告读者（投资经理、风控、面试官）只有 5–10 分钟，要快速回答三个问题：
        钱从哪里来？证据可信吗？什么时候会失效？

    本课的流程（研究纪律写进代码）：
      ① 事先写下假设（HYPOTHESES）与选择规则（样本内信息比率最高）
      ② 参数网格：4 种模型 × 3 种持仓数 × 2 种换手上限 = 24 次试验，全部写入 trials.jsonl
      ③ 只按样本内指标选一个配置 → 样本外评估一次
      ④ DSR 校正试验次数；成本 0–3 倍敏感性；平方根冲击估算容量
      ⑤ 结论由 conclude() 的规则推出：样本外不赚钱 → 不建议上线；赚钱但不显著 → 证据不足
    """)


def print_results(o: dict):
    print("\n" + "=" * 60)
    print("第二部分：本次研究的核心结果（全部来自计算）")
    print("=" * 60)
    st, IS, OOS = o["study"], o["is"], o["oos"]
    top = st.grid.sort_values("样本内信息比率", ascending=False).head(5)
    print(f"    {len(st.grid)} 次试验中样本内信息比率最高的 5 个：")
    print(top[["样本内信息比率", "样本内超额年化", "月均单边换手"]].round(3).to_string())
    print(f"\n    ▶ 选定：{st.best}")
    rows = [["年化收益（扣费）", pct(IS["年化收益"]), pct(OOS["年化收益"])],
            ["股票池等权年化", pct(IS["基准年化"]), pct(OOS["基准年化"])],
            ["超额年化（几何）", pct(IS["超额年化(几何)"]), pct(OOS["超额年化(几何)"])],
            ["信息比率", num(IS["信息比率"]), num(OOS["信息比率"])],
            ["超额 t 值", num(IS["超额 t 值"]), num(OOS["超额 t 值"])],
            ["最大回撤", pct(IS["最大回撤"]), pct(OOS["最大回撤"])],
            ["月均单边换手", pct(IS["月均单边换手"]), pct(OOS["月均单边换手"])]]
    print(pd.DataFrame(rows, columns=["指标", "样本内", "样本外"]).to_string(index=False))
    print(f"\n    样本内排序 vs 样本外排序：所选配置样本外排第 {o['oos_rank']}/{len(st.grid)}；"
          f"样本外最好的是 {o['oos_best'][0]}（样本内 IR {o['oos_best'][2]:.2f}）；秩相关 {o['is_oos_rho']:.2f}")
    print(f"\n    DSR（{st.dsr['试验次数']} 次试验）= {st.dsr['DSR']:.3f}；样本外超额 PSR = {st.oos_psr:.3f}")
    print(f"    结论：{o['conclusion']['verdict']}")
    for r in o["conclusion"]["reasons"]:
        print(f"      - {r}")


def five_minute_pitch(o: dict):
    print("\n" + "=" * 60)
    print("第三部分：5 分钟讲清这个策略（CONCEPT 6.4 节框架）")
    print("=" * 60)
    cfg, IS, OOS, st = o["cfg"], o["is"], o["oos"], o["study"]
    flips = o["factor_ic"].pipe(lambda d: d.index[np.sign(d["样本内IC"]) != np.sign(d["样本外IC"])].tolist())
    print(f"""
    [30 秒] 一句话：每月末在 {len(o['run'].scores.columns)} 只 A 股中，用 {len(cfg.factors)} 个中性化因子的 {cfg.model} 合成分数，
            买入最高的 {cfg.n_top} 只等权持有。
    [90 秒] 收益来源：价值、质量、低波、反转、低换手等异象——行为偏差（过度反应、彩票偏好）
            与风险 / 流动性补偿；多因子合成用于分散单因子失效。
    [90 秒] 证据：样本内超额 {pct(IS['超额年化(几何)'])}、IR {num(IS['信息比率'])}；但样本外超额 {pct(OOS['超额年化(几何)'])}、
            IR {num(OOS['信息比率'])}、t = {num(OOS['超额 t 值'])}；{st.dsr['试验次数']} 次试验后 DSR = {num(st.dsr['DSR'], 3)}。
            双倍成本下样本外超额 {pct(st.cost_table.query('成本倍数 == 2 and 区间 == "样本外"')['超额年化(几何)'].iloc[0])}。
    [60 秒] 失效条件：风格急切换；样本内外 IC 方向不一致的因子：{flips or "无"}（因子权重依赖历史 IC，切换时滞后）；
            股票池存在幸存者偏差，真实表现可能更差。
    [30 秒] 建议：{o['conclusion']['verdict']}。
    """)


def main():
    explain_structure()
    ctx = ResearchContext()
    o = compute(ctx)
    fig = make_figure(o)
    path = write_report(ctx, o, fig)
    print_results(o)
    five_minute_pitch(o)
    text = path.read_text(encoding="utf-8")
    required = ["假设与经济逻辑", "数据与样本区间", "成本假设", "基准对比", "样本内 vs 样本外", "试验次数与 Deflated Sharpe",
                "风险与失效条件", "局限性", "结论与建议"]
    missing = [s for s in required if s not in text]
    print("=" * 60)
    print("第四部分：输出与自检")
    print("=" * 60)
    print(f"    📝 研究报告：{path}（{len(text.splitlines())} 行）")
    print(f"    📊 净值图：{fig}")
    print(f"    🧾 试验日志：{o['trial_log'].path}（{len(o['trial_log'])} 条）")
    print(f"    ✅ 必备章节检查：{'全部齐全' if not missing else '缺少 ' + str(missing)}")
    print("""
    ✅ 小结：
      1. 报告结构 = 逻辑 → 数据 → 方法 → 成本 → 结果 → 稳健性 → 试验次数 → 容量 → 风险 → 局限 → 结论
      2. 每个数字都由代码算出；结论由事先写好的规则推出，不由作者的期望决定
      3. 一个"样本内很漂亮、样本外不显著"的策略，诚实的结论就是"不建议上线"
    下一课：把上线评审清单写成代码，对这个策略做一次尽调。
    """)


if __name__ == "__main__":
    main()
