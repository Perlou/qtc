"""
AI 量化的陷阱：前视、过拟合与幻觉
================================

学习目标：
    1. 理解 LLM 训练数据截止日期带来的前视偏差，并用模拟看到"知道结果的模型"在回测中如何虚高
    2. 掌握缓解方法：只在截止日之后评估、实体匿名化对照实验、记忆探针
    3. 识别三类幻觉：事实幻觉、代码幻觉（未来函数 / 不存在的 API）、推理幻觉，并用程序化方法检测
    4. 量化"AI 让试验变便宜"带来的过拟合：纯噪声因子试得越多，最好的那个越"显著"

核心概念：
    - 训练截止前视：回测区间在模型训练语料覆盖范围内 → 模型可能"记得"后来发生了什么
    - 匿名化对照：公司名、代码、日期替换为占位符；原文 IC 显著高于匿名化 IC → 存在记忆效应
    - 截止日前后对比：截止前 IC 高、截止后 IC 回落 → 危险信号
    - 最大值偏差：N 个零效应试验中最大 |t| 的期望随 √(2 ln N) 增长

前置知识：
    - 06–09 课；Phase 9 的多重检验与 DSR
    - llm-course Phase 10：06-faithfulness.py（忠实度）

环境要求：
    - pip install numpy pandas scipy matplotlib
    - 第二、三、五部分为模拟实验（参数为演示设定，不代表任何真实模型）；
      第四部分需要 Phase 7 面板（支持 QTC_OFFLINE=1）；有 OPENAI_API_KEY 时会做 1 次真实的记忆探针
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

import factor_dsl as dsl
from ai_tools import SUBDIR, daily_fields, ic_stats, load_panel
from llm_client import LLMClient
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

CUTOFF = pd.Timestamp("2024-06-30")  # 模拟中的"模型训练截止日"（演示设定）


# ==================== 第一部分：训练截止前视的机制 ====================


def mechanism():
    print("=" * 60)
    print("第一部分：LLM 训练数据截止带来的前视偏差")
    print("=" * 60)
    print("""
    时间轴 ─────────────────────────────────────────────────────────────▶
         回测区间 [2018, 2024]            模型训练截止          实盘
         ├──────────────────────────┤          │                  │
         │ 模型"见过"这段时期的新闻、股价、    │                  │
         │ 事后分析、"XX 为何暴涨"的文章       │                  │
         → 回测中的判断混入了事后知识 → IC 虚高 │   → 实盘没有这种"记忆"，IC 回落

    比喻：你请一位"穿越者"帮你回测 2020 年的新闻情绪策略。他读到"某疫苗公司启动三期临床"时
    打出"强烈利好"——不是因为新闻本身，而是因为他记得这只股票后来涨了 5 倍。

    泄漏渠道：公司名 / 代码 → 唤起对后续走势的记忆 · 日期 → 唤起对宏观事件结局的记忆 ·
              独特事件描述 → 唤起对事件结局的记忆 · 学术文献 → "知道"哪些因子后来被发表为有效
    同样的问题也存在于时序基础模型与大语料预训练的文本 embedding 模型。
    """)


# ==================== 第二部分：模拟"知道结果的模型" ====================


def simulate_lookahead(seed: int = 0):
    print("\n" + "=" * 60)
    print("第二部分：模拟 —— '知道结果的模型'在历史回测中的虚高表现")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-31", "2026-06-30", freq="ME")
    n = 300
    a = 0.03            # 新闻文本中真实可用的信息：收益 = a × 信号 + 噪声
    kappa = 0.3         # 记忆强度：模型认出公司时，分数里混入的"已知结果"成分（标准化收益的系数）
    recog = {"原文": 0.8, "匿名化": 0.25}  # 认出公司的概率（匿名化后仍可能通过产品、地名等细节认出）
    rows = []
    for m in months:
        s = rng.standard_normal(n)
        r = a * s + rng.standard_normal(n)
        read = s + rng.standard_normal(n)  # 模型对文本本身的理解（有误差）
        knows = m <= CUTOFF
        rec = {"month": m, "诚实模型（无记忆）": stats.spearmanr(read, r).statistic}
        for k, p in recog.items():
            mem = (rng.random(n) < p) & knows
            score = read + kappa * np.where(mem, (r - r.mean()) / r.std(), 0.0)
            rec[f"LLM·{k}"] = stats.spearmanr(score, r).statistic
        rows.append(rec)
    ic = pd.DataFrame(rows).set_index("month")
    yearly = ic.groupby(ic.index.year).mean()
    print(f"\n    设定：{n} 只股票 × {len(months)} 个月；模拟训练截止日 {CUTOFF.date()}；"
          f"真实文本信号强度 a = {a}（演示参数）\n")
    print("    各年平均 Rank IC：\n")
    print(yearly.to_string(float_format=lambda v: f"{v:+.3f}"))
    return ic


def lookahead_check(ic_raw: pd.Series, ic_anon: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    """对比截止日前后、匿名化前后的平均 IC（CONCEPT.md 第 11.1 节）"""
    def split(s):
        return pd.Series({"截止前": s[s.index <= cutoff].mean(), "截止后": s[s.index > cutoff].mean()})
    return pd.DataFrame({"原文": split(ic_raw), "匿名化": split(ic_anon)})


def diagnose(ic: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第三部分：诊断 —— 截止日前后对比 + 匿名化对照")
    print("=" * 60)
    tab = lookahead_check(ic["LLM·原文"], ic["LLM·匿名化"], CUTOFF)
    tab["诚实模型"] = [ic.loc[ic.index <= CUTOFF, "诚实模型（无记忆）"].mean(), ic.loc[ic.index > CUTOFF, "诚实模型（无记忆）"].mean()]
    print("\n" + tab.to_string(float_format=lambda v: f"{v:+.4f}"))
    post = ic[ic.index > CUTOFF]
    t_post = ic_stats(post["LLM·原文"])["t"]
    print(f"""
    危险信号（对照上表的本次模拟结果）：
      · 原文 IC 截止前 {tab.loc['截止前', '原文']:+.3f} → 截止后 {tab.loc['截止后', '原文']:+.3f}
      · 截止前：原文 {tab.loc['截止前', '原文']:+.3f} vs 匿名化 {tab.loc['截止前', '匿名化']:+.3f}，匿名化后差距明显缩小
      · 截止后只有 {len(post)} 个月，原文 IC 的 t 值 = {t_post:.2f} —— 真正可信的样本外区间往往很短
    匿名化只能"缓解"：本模拟中匿名化后仍有 25% 的公司被认出，截止前 IC 仍高于诚实模型。

    ┌────────────────────┬───────────────────────────────────┬──────────────────────────────┐
    │ 方法               │ 做法                              │ 局限                         │
    ├────────────────────┼───────────────────────────────────┼──────────────────────────────┤
    │ 只在截止日后评估   │ 只用截止日之后的数据做样本外      │ 样本期短；截止日可能不精确   │
    │ 实体匿名化         │ 公司名、代码、日期 → 占位符       │ 细节仍可能泄露身份           │
    │ 对照实验           │ 比较匿名化前后、截止前后的 IC     │ 只能检测，不能消除           │
    │ 记忆探针           │ 直接问模型"X 公司 2021 年表现"    │ 知道 ≠ 一定在打分中使用了    │
    │ 时点一致的模型     │ 用训练截止于回测期之前的旧模型    │ 旧模型能力弱                 │
    │ 只做信息抽取       │ 抽取客观事实，而非主观判断        │ 减少但不能消除记忆影响       │
    └────────────────────┴───────────────────────────────────┴──────────────────────────────┘
    参考：Lopez-Lira & Tang（2023）使用截止日之后的新闻评估；Glasserman & Lin（2023）用匿名化公司名做对照。
    """)


def anonymize(text: str, entities: dict[str, str]) -> str:
    """entities 按长度降序替换，避免"茅台"先于"贵州茅台"被替换；日期粗粒度遮蔽"""
    for name in sorted(entities, key=len, reverse=True):
        text = text.replace(name, entities[name])
    text = re.sub(r"20\d{2}年", "某年", text)
    return re.sub(r"\d{1,2}月\d{1,2}日", "某日", text)


def anonymize_demo():
    text = "贵州茅台（600519）2021年3月12日公告：茅台酒出厂价上调，董事长在遵义调研。"
    ents = {"贵州茅台": "公司A", "600519": "代码A", "茅台": "公司A"}
    out = anonymize(text, ents)
    print("    匿名化示例：")
    print(f"      原文  ：{text}")
    print(f"      匿名化：{out}")
    left = [w for w in ("酒", "出厂价", "遵义") if w in out]
    print(f"      残留线索：{left} —— 产品与地名仍足以让模型猜出公司，匿名化需要配合人工审查")


# ==================== 第四部分：幻觉检测 ====================


def hallucination_checks(llm: LLMClient):
    print("\n" + "=" * 60)
    print("第四部分：幻觉检测 —— 代码、事实与记忆探针")
    print("=" * 60)
    panel = load_panel(verbose=False)
    data = daily_fields(panel)
    cut = data["close"].index[-250]
    generated = {  # 模拟 LLM 生成的因子代码
        "动量（正确）": lambda d: d["close"] / d["close"].shift(20) - 1,
        "动量（把 shift(-20) 当作滞后）": lambda d: d["close"] / d["close"].shift(-20) - 1,
        "居中滚动均值（center=True）": lambda d: d["close"].rolling(21, center=True).mean() / d["close"] - 1,
        "全样本标准化": lambda d: (d["ret"] - d["ret"].mean()) / d["ret"].std(),
        "调用不存在的 API": lambda d: d["close"].rolling_zscore(20),
    }
    print(f"\n    代码幻觉：对 LLM 生成的因子函数做沙箱执行 + 前视检测（截断到 {cut.date()} 重算）\n")
    for name, fn in generated.items():
        try:
            ok = dsl.lookahead_test(fn, data, cut)
            print(f"      {name:<30} {'✅ 无前视' if ok else '❌ 截断未来数据后历史值改变 → 未来函数'}")
        except Exception as e:
            print(f"      {name:<30} ❌ 执行失败：{type(e).__name__}: {str(e)[:50]}")
    print("""
    "全样本标准化"在单列上看似无害，却用到了未来的均值和方差 —— 前视测试同样能抓住它。
    这正是 08 课坚持用白名单 DSL 的原因：DSL 里根本写不出负偏移和居中窗口。
    """)
    print("    事实幻觉：抽取的数字必须能在原文中找到（07 课 verify_numbers）")
    print("    推理幻觉：报告中的结论必须由实验日志支撑（09 课 faithfulness_check）")
    print("    时间幻觉：混淆事件先后 → PIT 检索 + 在 prompt 中明确 as-of 日期\n")
    probe = llm.json('只输出 JSON：{"stock": "贵州茅台", "year": 2021, "direction": "上涨/下跌/不知道", "approx_return_pct": 数字或 null}。'
                     "问题：贵州茅台 A 股股价在 2021 年全年是上涨还是下跌，大约多少？")
    if probe is None:
        print("    记忆探针：（无密钥 / 离线，未调用）做法是直接询问模型某公司在回测期内的股价表现，")
        print("    若回答准确，说明模型“记得”结果，基于原文的回测打分就必须做匿名化或只用截止日后的数据。")
    else:
        print(f"    记忆探针（真实 LLM 回答）：{probe}")
        print("    → 把回答与真实行情比对；模型若能答对，说明它记得回测期的结果（知道 ≠ 一定在打分中使用）")


# ==================== 第五部分：过拟合 —— AI 让试验变得太便宜 ====================


def overfitting_sim(seed: int = 1):
    print("\n" + "=" * 60)
    print("第五部分：过拟合 —— 纯噪声因子试得越多，最好的那个越'显著'")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    T_is, T_oos, sd = 120, 60, 0.08  # 每期 IC 的标准差（约等于 1/√股票数）
    rows = []
    for N in (1, 10, 100, 1000, 5000):
        ic_is = rng.normal(0, sd, (N, T_is))      # 真实 IC 全部为 0
        t = ic_is.mean(1) / ic_is.std(1, ddof=1) * np.sqrt(T_is)
        best = int(np.argmax(t))
        ic_oos = rng.normal(0, sd, T_oos)          # 最优者在样本外：仍然是噪声
        rows.append([N, t.max(), np.sqrt(2 * np.log(N)) if N > 1 else 0.0, (np.abs(t) > 1.96).sum(),
                     ic_is[best].mean(), ic_oos.mean(), ic_oos.mean() / ic_oos.std() * np.sqrt(T_oos)])
    tab = pd.DataFrame(rows, columns=["试验数 N", "最大 t", "√(2lnN) 近似", "|t|>1.96 个数",
                                      "最优者样本内 IC", "最优者样本外 IC", "样本外 t"])
    print(f"\n    设定：每个因子的真实 IC = 0；样本内 {T_is} 期、样本外 {T_oos} 期\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    big = tab.iloc[-1]
    print(f"""
    本次模拟：N = {int(big['试验数 N'])} 时最大 t = {big['最大 t']:.2f}，有 {int(big['|t|>1.96 个数'])} 个纯噪声因子"显著"；
    而最优者的样本外 t = {big['样本外 t']:.2f}。一个 LLM Agent 一晚上就能跑几千次试验 ——
    只报告最好的那个，等于把运气当作发现。
    ┌──────────────────────────┬────────────────────────────────────┬────────────────────────────┐
    │ 来源                     │ 表现                               │ 对策                       │
    ├──────────────────────────┼────────────────────────────────────┼────────────────────────────┤
    │ LLM / Agent 批量生成因子 │ 试 1000 个、挑最好的 10 个          │ 记录全部试验；DSR / PBO    │
    │ 深度模型超参数搜索       │ 网格搜索 200 组配置                │ 搜索本身计入试验次数       │
    │ 反复"看一眼"测试集       │ 看完结果再调 prompt                 │ 测试集只用一次             │
    │ 重新发现已知因子         │ "新因子"是已发表异象的变体          │ 与已知因子库做相关性检查   │
    │ RL 记住历史路径          │ 训练完美、测试崩溃（05 课）         │ 随机起点、简单基线对比     │
    └──────────────────────────┴────────────────────────────────────┴────────────────────────────┘
    """)
    return tab


# ==================== 第六部分：工程风险与最佳实践 ====================


def best_practices(ic: pd.DataFrame, tab: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第六部分：工程风险与最佳实践清单")
    print("=" * 60)
    print("""
    工程与运营风险：
      · 不可复现：模型版本更新、temperature > 0 → 固定版本、temperature=0、缓存全部输出（llm_client.py）
      · 成本与延迟：全市场文本实时打分的成本决定策略可行的频率
      · 数据合规：新闻 / 研报的版权与许可；不得把未公开信息输入外部 API
      · 提示注入：外部文本中的恶意指令（06 课）
      · 供应商风险：API 下线或政策变化 → 保留可本地部署的替代方案

    □ 深度学习：预测收益而非价格；和朴素基线、线性模型比；标准化只用训练期；多种子
    □ 强化学习：t 收盘观察 → t+1 开盘执行；成本进环境；前视单元测试；和买入持有比
    □ LLM 文本：记录首次发布时间；截止日后评估 + 匿名化对照；中性化；缓存输出
    □ 因子挖掘 / Agent：白名单 DSL；护栏在工具层；记录全部试验并做多重检验；人工审核
    """)
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    roll = ic.rolling(6).mean()
    for col in roll:
        axes[0].plot(roll.index, roll[col], label=col, lw=1.3)
    axes[0].axvline(CUTOFF, color="red", ls="--", lw=1, label="模拟训练截止日")
    axes[0].set_title("模拟：'知道结果的模型'的 IC（6 个月滚动平均）")
    axes[0].legend(fontsize=8)
    axes[1].plot(tab["试验数 N"], tab["最大 t"], marker="o", label="最大 t（模拟）")
    axes[1].plot(tab["试验数 N"], tab["√(2lnN) 近似"], ls="--", label="√(2 ln N)")
    axes[1].axhline(1.96, color="gray", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_title("纯噪声因子：试验数越多，最大 t 越高")
    axes[1].set_xlabel("试验数 N")
    axes[1].legend(fontsize=8)
    path = save_figure(fig, "10_ai_quant_pitfalls.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    mechanism()
    ic = simulate_lookahead()
    diagnose(ic)
    anonymize_demo()
    llm = LLMClient(max_calls=1)
    print(f"\n    {llm.describe()}")
    hallucination_checks(llm)
    tab = overfitting_sim()
    best_practices(ic, tab)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 在第二部分的模拟中把记忆强度 kappa 调小到 0.05，截止日前后的 IC 差异还能被 t 检验发现吗？需要多少个月？
    2. 用 06 课的真实新闻，写一个实体字典并做匿名化，人工检查有多少条仍能猜出公司
    3. 为第四部分的"代码幻觉"列表再添加 3 种常见错误（如 bfill、使用当日收盘价成交），验证前视测试能否发现
    4. 用第五部分的模拟估计：若要让"最优者"在 N = 1000 时仍有 5% 的显著性，单次检验的 t 阈值应设为多少？
    """)


if __name__ == "__main__":
    main()
