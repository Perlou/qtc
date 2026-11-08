"""
量化研究 Agent：纯 Python 的 Agent 循环与工具层护栏
================================================

学习目标：
    1. 把 llm-course 的 Agent 概念（工具、ReAct、记忆、人在回路）映射到量化研究流程
    2. 用纯 Python 实现 Agent 循环：提出假设 → 生成表达式 → 计算 IC → 记录 → 决策
    3. 把护栏放在工具层：试验预算、样本外数据不对 Agent 开放、表达式白名单、人工审核点
    4. 自动生成研究报告，并用程序检查"报告中的数字是否都来自试验日志"（忠实性）

核心概念：
    - 护栏在工具层，而不是在 prompt 里："请不要用未来数据"不是护栏，"根本拿不到未来数据"才是
    - 最小权限：Agent 的数据工具只返回训练期；样本外测试需要人工批准，且每个想法只能测一次
    - 试验日志 = 记忆 = 多重检验的 N：Agent 让试验变便宜，所以更要记录每一次尝试
    - 决策策略：有密钥时由 LLM 决定下一步（JSON 输出）；无密钥时用规则策略代替

前置知识：
    - 08-llm-factor-mining.py（DSL、试验日志、多重检验）
    - llm-course Phase 6：02-react-agent.py、04-tool-basics.py、10-agent-memory.py；
      Phase 7：02-langgraph-basics.py、08-human-in-the-loop.py；Phase 10：03-llm-as-judge.py

环境要求：
    - 纯 Python 实现（不依赖 langgraph；思路与 CONCEPT.md 第 10.3 节的 LangGraph 状态图一一对应）
    - 可选：OPENAI_API_KEY（最多 5 次 LLM 决策，结果缓存）；无密钥时使用规则策略
    - 数据来自 Phase 7 面板；支持 QTC_OFFLINE=1
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import factor_dsl as dsl
from ai_tools import (P10_OUT, SUBDIR, TrialLog, benjamini_hochberg, daily_fields, extract_numbers,
                      forward_open_returns, ic_stats, load_panel, signal_dates, universe_mask)
from llm_client import LLMClient
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRAIN = ("2019-07-01", "2022-12-31")
HOLDOUT = ("2023-01-01", "2025-02-14")
HORIZON = 5
BUDGET = 14          # 训练期回测次数上限（含被拒绝的表达式）
HOLDOUT_LIMIT = 1    # 整个研究最多做几次样本外测试
GOAL = "寻找与短期反转低相关、可解释的日频量价因子（持有 5 个交易日）"


# ==================== 第一部分：从 llm-course 的 Agent 到研究 Agent ====================


def overview():
    print("=" * 60)
    print("第一部分：从 llm-course 的 Agent 到研究 Agent")
    print("=" * 60)
    print("""
    ┌────────────────────┬───────────────────────────────────────────────┐
    │ llm-course 概念    │ 研究 Agent 中的对应                           │
    ├────────────────────┼───────────────────────────────────────────────┤
    │ 工具 Tool          │ 数据查询、表达式校验、训练期回测、样本外测试  │
    │ ReAct 循环         │ 提出假设 → 跑实验 → 观察结果 → 修正假设       │
    │ 记忆 Memory        │ 试验日志（同时是多重检验的 N）                │
    │ LangGraph 状态图   │ 本课用 while 循环 + 状态对象实现同样的流程    │
    │ 人在回路 HITL      │ 进入样本外测试前的人工审核                    │
    │ LLM-as-Judge       │ 报告忠实性检查（本课用程序化比对）            │
    └────────────────────┴───────────────────────────────────────────────┘

      propose ──▶ implement ──▶ backtest ──▶ analyze ──┬─▶ 继续（未达标且预算未用完）──▶ propose
                                                       └─▶ 达标 / 预算耗尽 ──▶ human review ──▶ holdout ──▶ report

    工具层强制护栏：数据只到训练期末 · 表达式走 AST 白名单 · 每次回测自动记日志并扣预算 ·
                    样本外需要人工批准且次数有限
    """)


# ==================== 第二部分：工具层（护栏所在） ====================


class GuardrailError(RuntimeError):
    """护栏拦截"""


class ResearchTools:
    """Agent 只能通过这些方法接触数据；样本外数据保存在工具内部，不经由任何工具返回给 Agent"""

    def __init__(self, panel, log: TrialLog, budget: int = BUDGET):
        full = daily_fields(panel)
        dates = signal_dates(panel, TRAIN[0], HOLDOUT[1], every=HORIZON)
        self._fwd = forward_open_returns(panel, dates, HORIZON)
        self._mask = universe_mask(panel, dates)
        self._full = full
        cut = pd.Timestamp(TRAIN[1])
        self.train_dates = dates[dates <= cut][:-1]  # 最后一期的持有期跨入封存期 → 去掉
        self._holdout_dates = dates[dates >= pd.Timestamp(HOLDOUT[0])]
        self._train_data = {k: v.loc[:cut] for k, v in full.items()}  # Agent 能看到的全部数据
        self.log, self.budget, self.used = log, budget, 0
        self.holdout_used: list[str] = []

    def get_data(self, end: str | None = None) -> dict:
        if end is not None and pd.Timestamp(end) > pd.Timestamp(TRAIN[1]):
            raise GuardrailError(f"数据工具只开放到训练期末 {TRAIN[1]}，请求 {end} 被拒绝")
        return self._train_data

    def backtest(self, expr: str, hypothesis: str, source: str) -> dict:
        if self.used >= self.budget:
            raise GuardrailError(f"试验预算 {self.budget} 次已用完")
        self.used += 1
        t0 = time.time()
        rec = {"source": source, "hypothesis": hypothesis, "expr": expr}
        try:
            dsl.validate(expr)
            ic = dsl.factor_ic(expr, self._train_data, self.train_dates, self._fwd, self._mask)
            h = len(ic) // 2
            st, a, b = ic_stats(ic), ic_stats(ic.iloc[:h]), ic_stats(ic.iloc[h:])
            rec.update(status="ok", ic=st["ic"], icir=st["icir"], t=st["t"], p=st["p"], n=st["n"],
                       ic_h1=a["ic"], ic_h2=b["ic"])
        except dsl.DSLError as e:
            rec.update(status="rejected", error=str(e))
        rec["seconds"] = round(time.time() - t0, 2)
        return self.log.add(**rec)

    def holdout_test(self, expr: str, approval: "Approval") -> dict:
        if not (isinstance(approval, Approval) and approval.approved and approval.expr == expr):
            raise GuardrailError("样本外测试需要针对该表达式的人工批准")
        if len(self.holdout_used) >= HOLDOUT_LIMIT or expr in self.holdout_used:
            raise GuardrailError(f"样本外测试次数已达上限（{HOLDOUT_LIMIT} 次）或该表达式已测过")
        self.holdout_used.append(expr)
        ic = dsl.factor_ic(expr, self._full, self._holdout_dates, self._fwd, self._mask)
        st = ic_stats(ic)
        return self.log.add(source="holdout", expr=expr, status="holdout", reviewer=approval.reviewer,
                            ic=st["ic"], icir=st["icir"], t=st["t"], p=st["p"], n=st["n"])


# ==================== 第三部分：Agent 状态与决策策略 ====================


@dataclass
class State:
    goal: str
    iteration: int = 0
    history: list = field(default_factory=list)
    tried: set = field(default_factory=set)
    candidates: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    done: bool = False


FAMILIES = [
    ("量价背离：放量不涨意味着分歧加大，未来收益偏弱", "-corr(close, volume, {n})", (10, 20, 5)),
    ("低波动：低波动股票被低估", "-ts_std(ret, {n})", (20, 60)),
    ("低换手：高换手代表过度交易", "-ts_mean(turnover, {n})", (20, 60)),
    ("非流动性（Amihud）：单位成交额的价格冲击越大，要求的补偿越高", "ts_mean(abs(ret) / amount, {n})", (20,)),
    ("日内反转：日内涨幅大的股票随后回落", "-ts_mean(close / open - 1, {n})", (5, 10)),
    ("隔夜动量：隔夜收益反映知情交易", "ts_mean(open / delay(close, 1) - 1, {n})", (20,)),
]


class RulePolicy:
    """无密钥时的规则策略：按假设族逐个试；显著且两半样本同号 → 收录并换族；显著但为负 → 反向一次"""

    name = "规则策略"

    def __init__(self):
        self.fam, self.k, self.flip_next = 0, 0, None

    def next_action(self, state: State) -> dict:
        if self.flip_next:
            expr, hyp = self.flip_next
            self.flip_next = None
            return {"action": "test", "expr": f"-({expr})", "hypothesis": f"反向：{hyp}",
                    "thought": "上一个表达式显著为负，按反方向重新检验（这也算一次新的试验）"}
        if len(state.candidates) >= 2 and not any("rank(" in h["expr"] for h in state.history):
            a, b = state.candidates[0]["expr"], state.candidates[1]["expr"]
            return {"action": "test", "expr": f"rank({a}) + rank({b})", "hypothesis": "两个候选的等权组合",
                    "thought": "已有两个稳定候选，尝试截面排名等权组合"}
        while self.fam < len(FAMILIES):
            hyp, tpl, ns = FAMILIES[self.fam]
            if self.k < len(ns):
                expr = tpl.format(n=ns[self.k])
                self.k += 1
                if expr not in state.tried:
                    return {"action": "test", "expr": expr, "hypothesis": hyp,
                            "thought": f"尝试假设族 {self.fam + 1}，窗口 {ns[self.k - 1]}"}
            self.fam, self.k = self.fam + 1, 0
        return {"action": "stop", "thought": "所有假设族已尝试"}

    def observe(self, rec: dict, state: State):
        if rec["status"] != "ok":
            state.notes.append(f"表达式被拒绝：{rec['error']}")
            return
        stable = np.sign(rec["ic_h1"]) == np.sign(rec["ic_h2"])
        if rec["t"] >= 2.5 and stable:
            state.candidates.append(rec)
            self.fam, self.k = self.fam + 1, 0  # 收录后换下一个假设族，避免在同一族里过度调参
        elif rec["t"] <= -2.5 and stable and not rec["hypothesis"].startswith("反向"):
            self.flip_next = (rec["expr"], rec["hypothesis"])
        elif abs(rec["t"]) < 1:
            self.fam, self.k = self.fam + 1, 0  # 毫无信号 → 放弃该族


AGENT_PROMPT = """你是量化研究 Agent。研究目标：{goal}
可用字段：open, high, low, close, volume, amount, turnover, vwap, ret。
可用算子：rank, zscore, abs, log, sign, delay, delta, ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_rank, corr, + - * /（窗口为正整数）。
剩余试验预算：{left} 次。已完成的试验（训练期，持有 5 日，t 为 IC 的 t 值）：
{history}
请决定下一步。输出 JSON：{{"thought": "...", "action": "test" 或 "stop", "hypothesis": "...", "expr": "..."}}
要求：每个表达式必须有经济逻辑；不要只做参数微调；如果已有足够稳定的候选，可以选择 stop。"""


class LLMPolicy(RulePolicy):
    """有密钥时由 LLM 决策；LLM 不可用或输出不合规时，该步退回规则策略"""

    name = "LLM 策略（失败时退回规则）"

    def __init__(self, llm: LLMClient):
        super().__init__()
        self.llm = llm

    def next_action(self, state: State) -> dict:
        hist = "\n".join(f"- {h['expr']} | 状态 {h['status']} | t={h.get('t', float('nan')):.2f}" for h in state.history[-8:]) or "（无）"
        raw = self.llm.json(AGENT_PROMPT.format(goal=state.goal, left=BUDGET - len(state.history), history=hist))
        if isinstance(raw, dict) and raw.get("action") in ("test", "stop") and (raw["action"] == "stop" or raw.get("expr")):
            raw["source"] = "LLM"
            return raw
        return super().next_action(state)


# ==================== 第四部分：Agent 循环 ====================


def run_agent(tools: ResearchTools, policy, state: State):
    print("\n" + "=" * 60)
    print(f"第三、四部分：运行 Agent 循环（{policy.name}，预算 {tools.budget} 次）")
    print("=" * 60)
    print(f"\n    目标：{state.goal}")
    print(f"    Agent 可见数据：{TRAIN[0]} ~ {TRAIN[1]}（样本外 {HOLDOUT[0]} 之后对 Agent 不可见）\n")
    while not state.done:
        state.iteration += 1
        act = policy.next_action(state)                                   # propose + implement
        if act["action"] == "stop":
            print(f"    [{state.iteration:02d}] 决策：停止 —— {act.get('thought', '')}")
            break
        try:
            rec = tools.backtest(act["expr"], act.get("hypothesis", ""), act.get("source", "规则"))   # backtest
        except GuardrailError as e:
            print(f"    [{state.iteration:02d}] 🛑 护栏：{e}")
            break
        state.history.append(rec)
        state.tried.add(act["expr"])
        policy.observe(rec, state)                                        # analyze
        if rec["status"] == "ok":
            print(f"    [{state.iteration:02d}] {act['expr'][:46]:<46} IC={rec['ic']:+.4f} t={rec['t']:+.2f} "
                  f"前/后半 {rec['ic_h1']:+.3f}/{rec['ic_h2']:+.3f}")
        else:
            print(f"    [{state.iteration:02d}] {act['expr'][:46]:<46} ❌ {rec['error'][:30]}")
        state.done = tools.used >= tools.budget                           # decide
    print(f"\n    结束：Agent 共 {len(state.history)} 次试验，候选 {len(state.candidates)} 个；"
          f"预算使用 {tools.used}/{tools.budget}（含护栏演示中被拒绝的 1 次）")
    return state


# ==================== 第五部分：护栏演示 ====================


def guardrail_demo(tools: ResearchTools):
    print("\n" + "=" * 60)
    print("第二部分（演示）：工具层护栏会拦住什么")
    print("=" * 60)
    attempts = [("请求 2024 年的数据", lambda: tools.get_data(end="2024-06-30")),
                ("未批准就做样本外测试", lambda: tools.holdout_test("-ts_std(ret, 20)", approval=None))]
    for name, fn in attempts:
        try:
            fn()
            print(f"    {name}：未被拦截 ❌")
        except GuardrailError as e:
            print(f"    {name}：🛑 {e}")
    rec = tools.backtest("delay(close, -5) / close - 1", "（演示）偷看未来", "演示")
    print(f"    提交含负窗口的表达式：🛑 {rec['error']}（这次尝试同样计入日志与预算）")
    print(f"    训练数据最后一天：{tools.get_data()['close'].index[-1].date()}")
    print("""
    生产环境中，样本外数据应放在独立服务或独立权限的存储中；本课用同一进程的私有属性模拟，
    关键是 Agent 能调用的工具里，没有任何一个会返回封存期数据。
    """)


# ==================== 第六部分：人工审核、样本外与报告 ====================


@dataclass
class Approval:
    expr: str
    approved: bool
    reviewer: str
    reasons: list


def human_review(state: State, log: TrialLog) -> Approval | None:
    """人工审核点（课程中用规则模拟审核人的检查清单；实际中这里应暂停并等待人工确认）"""
    ok = log.frame().query("status == 'ok'")
    if not state.candidates:
        print("\n    审核：没有候选进入审核 → 不做样本外测试")
        return None
    q = pd.Series(benjamini_hochberg(ok["p"].to_numpy()), index=ok["expr"].to_numpy())
    rows = []
    for c in state.candidates:
        checks = {"有经济逻辑": bool(c["hypothesis"]), "BH q < 0.10（全部试验）": q[c["expr"]] < 0.10,
                  "两半样本同号": np.sign(c["ic_h1"]) == np.sign(c["ic_h2"]),
                  "非组合因子（更易解释）": "rank(" not in c["expr"]}
        rows.append((c, checks, sum(checks.values())))
    print("\n    审核清单（模拟审核人）：")
    for c, checks, s in rows:
        print(f"      {c['expr'][:44]:<44} " + " ".join(f"{k}{'✅' if v else '❌'}" for k, v in checks.items()))
    passed = [(c, s) for c, checks, s in rows if all(checks.values())]
    if not passed:
        print("    审核结论：没有候选通过全部检查 → 拒绝进入样本外")
        return None
    best = max(passed, key=lambda x: x[0]["t"])[0]
    print(f"    审核结论：批准 1 个候选进入样本外测试 → {best['expr']}")
    return Approval(best["expr"], True, "模拟审核人", ["通过全部检查"])


def write_report(state: State, log: TrialLog, holdout: dict | None) -> str:
    df = log.frame()
    ok = df[df["status"] == "ok"]
    lines = [f"研究目标：{state.goal}",
             f"样本：训练期 {TRAIN[0]} 至 {TRAIN[1]}，封存样本外 {HOLDOUT[0]} 至 {HOLDOUT[1]}，持有 {HORIZON} 个交易日",
             f"试验总数 {len(df[df['source'] != 'holdout'])} 次，其中被拒绝 {int((df['status'] == 'rejected').sum())} 次",
             "训练期结果（按 t 值排序）："]
    for _, r in ok.sort_values("t", ascending=False).head(4).iterrows():
        lines.append(f"  {r['expr']}：IC {r['ic']:.4f}，t {r['t']:.2f}")
    if holdout:
        lines.append(f"样本外（仅测试一次）：{holdout['expr']}：IC {holdout['ic']:.4f}，t {holdout['t']:.2f}，期数 {holdout['n']}")
    else:
        lines.append("样本外：未进行（无候选通过审核）")
    lines.append("局限：股票池约 130 只、样本期短、未计交易成本与容量；所有结论需在更大样本上复核")
    return "\n".join(lines)


def faithfulness_check(report: str, log: TrialLog) -> list[str]:
    """报告中的每个数字都必须出现在试验日志或研究配置中（四舍五入到报告精度）"""
    allowed = set()
    for rec in log.records:
        for v in rec.values():
            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                allowed |= {round(float(v), 4), round(float(v), 2), float(int(v)) if float(v).is_integer() else None}
            elif isinstance(v, str):
                allowed |= extract_numbers(v)
    allowed |= extract_numbers(" ".join([*TRAIN, *HOLDOUT, str(HORIZON), "130"]))
    allowed.add(float(len([r for r in log.records if r.get("source") != "holdout"])))
    allowed.add(float(sum(r.get("status") == "rejected" for r in log.records)))
    return [f"数字 {x:g} 在试验日志中找不到来源" for x in sorted(extract_numbers(report)) if x not in allowed]


def review_holdout_report(tools, state, log):
    print("\n" + "=" * 60)
    print("第五部分：人工审核 → 一次性样本外测试 → 报告与忠实性检查")
    print("=" * 60)
    approval = human_review(state, log)
    holdout = tools.holdout_test(approval.expr, approval) if approval else None
    if holdout:
        print(f"    样本外：IC = {holdout['ic']:.4f}，t = {holdout['t']:.2f}（{holdout['n']} 期）")
        try:
            tools.holdout_test(approval.expr, approval)
        except GuardrailError as e:
            print(f"    再测一次？🛑 {e}")
    report = write_report(state, log, holdout)
    print("\n    —— 自动生成的研究报告 ——")
    print("\n".join("    " + line for line in report.splitlines()))
    issues = faithfulness_check(report, log)
    print(f"\n    忠实性检查（程序生成的报告）：{'✅ 所有数字均可溯源' if not issues else '❌ ' + '；'.join(issues)}")
    best = log.frame().query("status == 'ok'").sort_values("t").iloc[-1]
    fake = report + f"\n结论：最佳因子训练期 IC 高达 {best['ic'] * 1.5:.4f}，样本外胜率 68%，建议立即上线。"
    issues = faithfulness_check(fake, log)
    print("    模拟 LLM 改写后的报告（夸大 IC、编造胜率）：" + ("✅" if not issues else "❌ " + "；".join(issues)))
    print("""
    报告必须披露：试验总次数、被拒绝的尝试、样本区间、成本假设、局限。
    LLM 擅长把报告写得"有说服力"，所以数字一律从日志生成，再用程序（或 LLM-as-Judge）核对。
    """)
    return holdout


def plot_trace(log: TrialLog):
    df = log.frame()
    ok = df[df["status"] == "ok"]
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ok["trial_id"].astype(str), ok["t"], color=["tab:green" if t > 2.5 else "tab:gray" for t in ok["t"]])
    ax.axhline(2.5, color="red", ls="--", lw=1, label="规则策略的收录阈值 t = 2.5")
    ax.set_xlabel("试验编号")
    ax.set_ylabel("训练期 IC 的 t 值")
    ax.set_title("Agent 的试验轨迹（每一根柱子都是一次计入 N 的试验）")
    ax.legend(fontsize=8)
    path = save_figure(fig, "09_agent_trial_trace.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    overview()
    panel = load_panel()
    log = TrialLog(P10_OUT / "09_agent_trial_log.jsonl")
    tools = ResearchTools(panel, log)
    guardrail_demo(tools)
    llm = LLMClient(max_calls=5)
    print(f"    {llm.describe()}")
    policy = LLMPolicy(llm) if llm.available else RulePolicy()
    state = run_agent(tools, policy, State(goal=GOAL))
    review_holdout_report(tools, state, log)
    plot_trace(log)
    print(f"    试验日志：{log.path}")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 human_review 改成真正的人工确认（input() 或写入待审文件后退出），并保存 Agent 状态以便恢复
    2. 安装 langgraph 后，用 StateGraph 重写本课循环，用 interrupt_before=["holdout"] 实现人在回路
    3. 给 ResearchTools 增加单次回测的超时与内存限制（提示：在子进程中执行 dsl.evaluate）
    4. 让规则策略在收录候选时额外检查与已有候选的截面相关性（> 0.7 则拒绝），观察试验轨迹的变化
    """)


if __name__ == "__main__":
    main()
