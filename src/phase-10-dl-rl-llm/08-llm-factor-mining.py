"""
LLM 因子挖掘：安全 DSL、训练期 IC、试验日志与多重检验
====================================================

学习目标：
    1. 理解从遗传规划（GP）到 LLM 的因子挖掘演进，以及"LLM 生成 → 校验 → 评估 → 反馈"的闭环
    2. 实现安全的因子表达式 DSL：ast 解析 + 白名单，拒绝属性访问、未知函数、负窗口（未来函数）
    3. 在 Phase 7 面板上计算候选因子的训练期 Rank IC，并把每一次尝试（含失败）写入试验日志
    4. 用 Holm / Benjamini-Hochberg / 紧缩夏普（DSR）修正多重检验，最后只做一次样本外检验

核心概念：
    - 永远不要 eval() LLM 生成的代码：白名单 DSL 让"不安全"和"未来函数"在解析阶段就被拒绝
    - 前视检测：截断未来数据后重新计算，历史因子值必须完全不变
    - 隐性多重检验：试了 N 个表达式、挑出最好的，最好那个的 t 值天然偏高
    - 封存样本外：研究期间只用训练期；研究结束后才在封存期测试一次

前置知识：
    - Phase 7：因子构建、Rank IC（05-ic-ir-analysis.py）
    - Phase 9：多重检验、DSR
    - llm-course Phase 2：JSON 输出；Phase 10：10-adversarial-prompting（不可信输入）

环境要求：
    - pip install pandas numpy scipy openai
    - 可选：OPENAI_API_KEY（有密钥时由 LLM 额外生成 8 个候选，调用 1 次并缓存）；无密钥时使用预置候选列表模拟
    - 数据来自 Phase 7 面板；支持 QTC_OFFLINE=1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import factor_dsl as dsl
from ai_tools import (P10_OUT, SUBDIR, TrialLog, benjamini_hochberg, daily_fields, deflated_sharpe, expected_max_sr,
                      forward_open_returns, holm, ic_stats, load_panel, signal_dates, universe_mask)
from llm_client import LLMClient
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRAIN = ("2019-07-01", "2022-12-31")
HOLDOUT = ("2023-01-01", "2025-02-14")
HORIZON = 5

# 预置候选（模拟"LLM 第一轮生成"）：假设 + 表达式；其中混入了典型的不合法输出
CANDIDATES = [
    ("短期反转：一周跌多的股票下周倾向反弹", "-(close / delay(close, 5) - 1)"),
    ("中期反转：一个月涨幅过大的股票回吐", "-(close / delay(close, 20) - 1)"),
    ("动量 12-1：跳过最近一个月的年度动量", "delay(close, 21) / delay(close, 250) - 1"),
    ("低波动异象：低波动股票风险调整后收益更高", "-ts_std(ret, 20)"),
    ("低换手：换手率高代表过度交易与情绪", "-ts_mean(turnover, 20)"),
    ("量价背离：放量但价格不涨意味着分歧", "-corr(close, volume, 10)"),
    ("放量下跌后反弹", "rank(-ts_mean(ret, 5)) - rank(ts_mean(turnover, 5) / ts_mean(turnover, 60))"),
    ("日内反转：日内涨幅大的股票之后回落", "-ts_mean(close / open - 1, 5)"),
    ("隔夜动量：隔夜收益反映知情交易", "ts_mean(open / delay(close, 1) - 1, 20)"),
    ("低振幅：日内振幅小的股票更稳健", "-ts_mean((high - low) / close, 20)"),
    ("收盘位置：收在日内高位代表买盘强", "ts_mean((close - low) / (high - low), 10)"),
    ("VWAP 偏离：收盘高于均价后回落", "-(close / vwap - 1)"),
    ("彩票偏好（MAX）：近期单日暴涨的股票被高估", "-ts_max(ret, 20)"),
    ("均线偏离：偏离 20 日均线过远会回归", "-(close / ts_mean(close, 20) - 1)"),
    ("小市值代理：成交额小的股票溢价", "-log(ts_mean(amount, 20))"),
    ("时序位置：处于近期高位的股票回落", "-ts_rank(close, 20)"),
    # ---- 典型的不合法输出 ----
    ("（未来函数）明天涨的股票今天买", "delay(close, -1) / close - 1"),
    ("（属性访问）pandas 写法", "close.shift(-1) / close"),
    ("（注入）执行系统命令", "__import__('os').system('ls')"),
    ("（未知字段）市盈率", "rank(pe_ratio)"),
    ("（关键字参数）", "ts_mean(close, n=5)"),
    ("（条件表达式）", "close if close > open else open"),
]
# 模拟"让 LLM 继续改进"时最常见的输出：同一想法的参数微调 —— 每一个都是一次新的试验
SWEEPS = [("短期反转（窗口微调）", "-(close / delay(close, {n}) - 1)", (3, 10, 15, 30, 40, 60)),
          ("低波动（窗口微调）", "-ts_std(ret, {n})", (5, 10, 40, 60, 120))]

GEN_PROMPT = """你是量化研究员。基于以下字段与算子，提出 {k} 个新的日频 A 股选股因子（预测未来一周收益）。
字段: open, high, low, close, volume, amount, turnover, vwap, ret（价格为后复权）
算子: rank, zscore, abs, log, sign, delay(x,n), delta(x,n), ts_mean(x,n), ts_std(x,n), ts_sum(x,n),
      ts_max(x,n), ts_min(x,n), ts_rank(x,n), corr(x,y,n), + - * /。窗口 n 必须是 1~250 的正整数常量。
已有因子（避免重复）: {library}
输出 JSON 列表，每项含 "hypothesis"（一句话经济逻辑）与 "expr"（表达式）。"""


# ==================== 第一部分：从遗传规划到 LLM ====================


def overview():
    print("=" * 60)
    print("第一部分：从遗传规划到 LLM 因子挖掘")
    print("=" * 60)
    print("""
    ┌────────────────┬────────────────────────────────┬──────────────────────────────────┐
    │ 方法           │ 代表                           │ 特点                             │
    ├────────────────┼────────────────────────────────┼──────────────────────────────────┤
    │ 遗传规划 GP    │ 101 Formulaic Alphas 风格搜索  │ 盲目组合算子，搜索空间巨大       │
    │ 强化学习       │ AlphaGen（KDD 2023）           │ 以因子组合的 IC 为奖励生成表达式 │
    │ LLM            │ Alpha-GPT、RD-Agent 等         │ 从经济直觉出发提出假设并翻译     │
    └────────────────┴────────────────────────────────┴──────────────────────────────────┘

    闭环：LLM 生成(假设+表达式) → AST 白名单校验 → 训练期计算 IC → 去相关筛选
          → 反馈给 LLM（IC、失败原因）→ ……  → 研究结束：封存样本外测试一次 + 多重检验修正
                              ↑ 每一次尝试（包括失败）都写入试验日志 = 多重检验的 N
    """)


# ==================== 第二部分：安全 DSL 与前视检测 ====================


def dsl_safety(data):
    print("\n" + "=" * 60)
    print("第二部分：安全的表达式 DSL —— 白名单校验与前视检测")
    print("=" * 60)
    tests = ["-rank(corr(close, volume, 10))", "delay(close, -1) / close - 1", "close.shift(-1) / close",
             "__import__('os').system('ls')", "ts_mean(close, 5.5)", "rank(pe_ratio)", "[c for c in close]",
             "close > open", "ts_mean(close, 1000)", "lambda x: x"]
    print(f"\n    支持字段：{', '.join(dsl.FIELDS)}")
    print(f"    支持算子：{', '.join(dsl.FUNCS)}\n")
    for e in tests:
        try:
            info = dsl.validate(e)
            print(f"      ✅ {e:<34} 回看 {info.lookback} 天，{info.n_nodes} 个节点")
        except dsl.DSLError as err:
            print(f"      ❌ {e:<34} {err}")
    cut = data["close"].index[-300]
    good = "-(close / ts_mean(close, 20) - 1)"
    bad_fn = lambda d: d["close"].shift(-1) / d["close"] - 1  # noqa: E731  手写 pandas 代码里的未来函数
    print(f"\n    前视检测（截断到 {cut.date()} 后重算，比较截断点之前最后 20 天）：")
    print(f"      DSL 表达式 {good:<36} → {'无前视 ✅' if dsl.lookahead_test(good, data, cut) else '有前视 ❌'}")
    print(f"      手写代码   close.shift(-1)/close - 1             → "
          f"{'无前视 ✅' if dsl.lookahead_test(bad_fn, data, cut) else '有前视 ❌'}")
    print("""
    白名单的意义：不是"检查 LLM 有没有写坏代码"，而是让坏代码根本无法表达。
    负窗口在解析阶段就被拒绝；属性访问、下标、import、lambda 都不在语法白名单里。
    若用 Qlib 表达式作为 DSL，同样要拒绝 Ref($close, -1) 这类负偏移。
    """)


# ==================== 第三部分：生成候选并在训练期评估 ====================


def build_inputs(panel):
    data = daily_fields(panel)
    dates = signal_dates(panel, TRAIN[0], HOLDOUT[1], every=HORIZON)
    fwd = forward_open_returns(panel, dates, HORIZON)
    mask = universe_mask(panel, dates)
    tr = dates[dates <= pd.Timestamp(TRAIN[1])][:-1]  # 训练期最后一个信号日的持有期跨入封存期 → 去掉
    ho = dates[dates >= pd.Timestamp(HOLDOUT[0])]
    return data, fwd, mask, tr, ho


def generate_candidates(llm: LLMClient) -> list[tuple[str, str, str]]:
    cands = [(h, e, "预置") for h, e in CANDIDATES]
    cands += [(h, t.format(n=n), "参数微调") for h, t, ns in SWEEPS for n in ns]
    raw = llm.json(GEN_PROMPT.format(k=8, library=[e for _, e in CANDIDATES[:6]]))
    if isinstance(raw, list):
        for item in raw[:8]:
            if isinstance(item, dict):
                cands.append((str(item.get("hypothesis", ""))[:60], str(item.get("expr", "")), "LLM"))
    return cands


def mining_round(cands, data, fwd, mask, train_dates, log: TrialLog):
    print("\n" + "=" * 60)
    print("第三部分：候选表达式 → 校验 → 训练期 Rank IC → 写入试验日志")
    print("=" * 60)
    n_llm = sum(src == "LLM" for *_, src in cands)
    print(f"\n    候选来源：预置 {sum(s == '预置' for *_, s in cands)} 个、参数微调 {sum(s == '参数微调' for *_, s in cands)} 个、"
          f"LLM 生成 {n_llm} 个" + ("（无密钥：用预置列表模拟 LLM 输出）" if n_llm == 0 else ""))
    print(f"    训练期 {train_dates[0].date()} ~ {train_dates[-1].date()}，每 {HORIZON} 个交易日一期（{len(train_dates)} 期，"
          f"持有期不重叠），标签 = t+1 开盘 → t+{HORIZON + 1} 开盘收益\n")
    for hyp, expr, src in cands:
        rec = {"source": src, "hypothesis": hyp, "expr": expr}
        try:
            info = dsl.validate(expr)
            ic = dsl.factor_ic(expr, data, train_dates, fwd, mask)
            st = ic_stats(ic)
            rec.update(status="ok", lookback=info.lookback, **{k: st[k] for k in ("ic", "icir", "t", "p", "n")})
        except dsl.DSLError as e:
            rec.update(status="rejected", error=str(e))
        except Exception as e:  # 数值问题等
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        log.add(**rec)
    df = log.frame()
    ok = df[df["status"] == "ok"].sort_values("t", key=np.abs, ascending=False)
    print(ok[["trial_id", "source", "expr", "ic", "icir", "t", "p"]].head(14).to_string(
        index=False, float_format=lambda v: f"{v:.4f}", max_colwidth=46))
    bad = df[df["status"] != "ok"]
    print(f"\n    被拒绝 / 出错 {len(bad)} 个：")
    for _, r in bad.iterrows():
        print(f"      #{r['trial_id']:<3} {r['expr'][:36]:<36} → {r['error'][:50]}")
    print(f"\n    试验日志已写入：{log.path}（共 {len(log)} 条，失败的尝试也计入）")
    return ok


# ==================== 第四部分：多重检验修正 ====================


def multiple_testing(ok: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：多重检验 —— 试了这么多次，最好的那个还显著吗？")
    print("=" * 60)
    ok = ok.copy()
    ok["p_holm"] = holm(ok["p"].to_numpy())
    ok["q_bh"] = benjamini_hochberg(ok["p"].to_numpy())
    n = len(ok)
    print(f"\n    有效试验 N = {n}")
    print(f"      原始 p < 0.05      ：{int((ok['p'] < 0.05).sum())} 个")
    print(f"      Holm 校正 p < 0.05 ：{int((ok['p_holm'] < 0.05).sum())} 个（控制族错误率 FWER，最严格）")
    print(f"      BH 校正 q < 0.10   ：{int((ok['q_bh'] < 0.10).sum())} 个（控制错误发现率 FDR）")
    best = ok.iloc[0]
    sr_std = ok["icir"].std()
    dsr = deflated_sharpe(abs(best["icir"]), int(best["n"]), n, sr_std)
    print(f"""
    最显著的因子：{best['expr']}（IC = {best['ic']:.4f}，ICIR = {best['icir']:.3f}，t = {best['t']:.2f}）
    把 IC 序列看作一条"收益流"，ICIR 就是它的单期夏普：
      N 次零效应试验中最大 ICIR 的期望 ≈ {expected_max_sr(n, sr_std):.3f}（各试验 ICIR 标准差 {sr_std:.3f}）
      紧缩夏普 DSR = {dsr:.3f}（> 0.95 才算在多重试验后仍显著）
    注意：参数微调得到的变体彼此高度相关，并不是独立试验，Holm 偏保守、"有效试验数"小于 N；
    但反过来，研究者脑中"想过但没写进日志"的试验也应该计入 —— 这正是要强制记录日志的原因。
    """)
    return ok


def decorrelate(ok, data, mask, train_dates, thresh: float = 0.7, max_keep: int = 5):
    """按 |t| 从高到低贪心选择；与已选因子的平均截面秩相关 |ρ| ≥ thresh 则跳过"""
    pool = ok[ok["q_bh"] < 0.10]
    kept, vals = [], {}
    sub = train_dates[::4]
    for _, r in pool.iterrows():
        f = dsl.evaluate(r["expr"], data).reindex(sub).where(mask.reindex(sub).astype(bool)).rank(axis=1)
        corr = {k: f.corrwith(v, axis=1).mean() for k, v in vals.items()}
        if all(abs(c) < thresh for c in corr.values()):
            kept.append((r, corr))
            vals[r["expr"]] = f
        if len(kept) >= max_keep:
            break
    return kept


# ==================== 第五部分：去相关筛选与一次性样本外检验 ====================


def selection_and_holdout(ok, data, fwd, mask, train_dates, holdout_dates, log: TrialLog):
    print("\n" + "=" * 60)
    print("第五部分：去相关筛选 → 封存样本外测试（只做一次）")
    print("=" * 60)
    kept = decorrelate(ok, data, mask, train_dates)
    if not kept:
        print("\n    没有因子通过 BH 校正（q < 0.10）→ 本轮研究没有可以进入样本外测试的候选。")
        print("    这是完全正常的结果：大多数想法在训练期就应该被否定。")
        return pd.DataFrame()
    print(f"\n    通过 BH(q<0.10) 且与已选因子 |ρ| < 0.7 的因子：{len(kept)} 个")
    rows = []
    for r, corr in kept:
        sign = np.sign(r["ic"])
        ho = ic_stats(dsl.factor_ic(r["expr"], data, holdout_dates, fwd, mask) * sign)  # 方向按训练期确定
        rows.append({"expr": r["expr"], "训练 IC": r["ic"] * sign, "训练 t": r["t"] * sign, "样本外 IC": ho["ic"],
                     "样本外 t": ho["t"], "样本外期数": ho["n"],
                     "与已选最大|ρ|": max((abs(c) for c in corr.values()), default=0.0)})
        log.add(source="holdout", expr=r["expr"], status="holdout", ic=ho["ic"], t=ho["t"], n=ho["n"])
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}", max_colwidth=44))
    alive = int(((out["样本外 IC"] > 0) & (out["样本外 t"] > 2)).sum())
    print(f"""
    方向已按训练期 IC 统一为"越大越好"。样本外 t > 2 且 IC > 0 的因子：{alive}/{len(out)} 个。
    样本外期数只有 {int(out['样本外期数'].iloc[0])} 期，标准误约 1/√期数 —— 样本外检验本身的功效也很有限。
    本课的候选列表本身大多是已发表异象（反转、低波、低换手、量价相关）的变体 —— 真实的 LLM 也很擅长
    "重新发现"已发表的因子，而已发表异象在发表后常有衰减。下一步应与 Phase 7 的因子库做相关性检查。
    ⚠️ 样本外只看这一次。若看完结果再回去改表达式，封存期就变成了训练期。
    """)
    return out


def plot_trials(ok, holdout):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"预置": "tab:blue", "参数微调": "tab:orange", "LLM": "tab:green"}
    for src, g in ok.groupby("source"):
        ax.scatter(range(len(g)), g["t"].abs().sort_values(ascending=False), label=src, s=18, color=colors.get(src, "gray"))
    ax.axhline(1.96, color="red", ls="--", lw=1, label="|t| = 1.96（单次检验）")
    ax.set_title("训练期各试验 |t| 值：试得越多，越容易出现'显著'的结果")
    ax.set_xlabel("排名")
    ax.legend(fontsize=8)
    path = save_figure(fig, "08_factor_mining_trials.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    overview()
    panel = load_panel()
    data, fwd, mask, tr, ho = build_inputs(panel)
    dsl_safety(data)
    llm = LLMClient()
    print(f"\n    {llm.describe()}")
    log = TrialLog(P10_OUT / "08_trial_log.jsonl")
    ok = mining_round(generate_candidates(llm), data, fwd, mask, tr, log)
    ok = multiple_testing(ok)
    holdout = selection_and_holdout(ok, data, fwd, mask, tr, ho, log)
    plot_trials(ok, holdout)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给 DSL 增加 ts_skew(x, n) 与 if_else(cond, a, b)，并为每个新算子写前视单元测试
    2. 计算入选因子与 Phase 7 build_factors 中 8 个经典因子的截面相关，有多少是"换汤不换药"？
    3. 把 SWEEPS 的窗口扩展到 3~120 的全部整数，观察原始 p < 0.05 的数量与 Holm 后的数量
    4. 有密钥时：把第三部分的 IC 结果与失败原因写进 GEN_PROMPT，让 LLM 做第二轮生成，比较两轮的命中率
    """)


if __name__ == "__main__":
    main()
