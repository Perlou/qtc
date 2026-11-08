"""
LLM 研报与财报分析：Point-in-Time RAG、结构化抽取与数字溯源
==========================================================

学习目标：
    1. 把 llm-course 的 RAG 技术迁移到金融场景，并加上时间维度：Point-in-Time（PIT）检索
    2. 用实验看到"普通 RAG"在回测中如何检索到未来发布的研报
    3. 用 schema 约束研报抽取结果（评级、目标价、EPS 预测、证据片段）
    4. 实现数字溯源：LLM 输出中的每个数字都必须能在原文中找到，否则标记为疑似幻觉
    5. 把抽取结果变成分析师因子（盈利预测修正、评级变化），全程按 as_of 时点计算

核心概念：
    - PIT RAG：检索条件 = (query, as_of)，只返回 publish_ts ≤ as_of 的文档
    - 结构化抽取：pydantic schema 约束字段、类型、取值范围
    - 数字溯源（faithfulness 的程序化版本）：抽取的数字 ∈ 原文数字集合
    - 分析师因子：EPS 预测修正 = (EPS_t − EPS_{t−k}) / |EPS_{t−k}|，必须用当时可得的最新预测

前置知识：
    - Phase 7 的 08-point-in-time-fundamentals.py（财报 PIT 对齐）
    - llm-course：Phase 2 的 08-structured-extraction.py；Phase 4 的 06-chroma-basics.py、08-retrieval-basics.py；
      Phase 5 的 01-hybrid-search.py、06-self-query-retrieval.py；Phase 10 的 06-faithfulness.py

环境要求：
    - pip install scikit-learn pydantic pandas openai
    - 可选：OPENAI_API_KEY（有密钥时对 2 篇研报做真实 LLM 抽取，并缓存）；无密钥时用规则抽取 + 模拟 LLM 响应
    - 本课的研报语料是【虚构的模拟文本】（版权原因，课程不分发真实研报），只用于演示流程
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
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer

from ai_tools import SUBDIR, verify_numbers
from llm_client import LLMClient
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

# ---- 虚构的研报语料：两家虚构公司，多家虚构券商，2023–2024 年 ----
DOCS = [
    ("R01", "900001", "甲电池", "华东证券", "2023-03-15 19:30",
     "甲电池2022年报点评：出货量高增，维持买入评级。公司2022年实现营收812.4亿元，同比增长61%。"
     "我们预计公司2023-2024年EPS分别为2.15元、2.68元，给予目标价58.0元，维持买入评级。风险提示：原材料价格波动、需求不及预期。"),
    ("R02", "900001", "甲电池", "北方证券", "2023-08-25 20:10",
     "甲电池中报点评：单位盈利修复。上半年归母净利润同比增长18%，毛利率回升至22.3%。"
     "预计2023-2024年EPS为2.05元、2.51元，下调目标价至49.5元，评级由买入调整为增持。风险：价格战加剧。"),
    ("R03", "900001", "甲电池", "华东证券", "2023-10-30 21:00",
     "甲电池三季报点评：利润略低于预期。前三季度营收同比增长9%，第三季度毛利率21.1%。"
     "下调盈利预测，预计2023-2024年EPS为1.92元、2.33元，目标价45.0元，维持增持评级。风险：海外政策变化。"),
    ("R04", "900001", "甲电池", "南方证券", "2024-03-20 18:45",
     "甲电池2023年报点评：盈利触底，海外业务提速。2023年营收同比增长3%，海外收入占比提升至31%。"
     "预计2024-2025年EPS为2.10元、2.55元，目标价52.0元，首次覆盖给予买入评级。风险：贸易摩擦。"),
    ("R05", "900001", "甲电池", "北方证券", "2024-08-28 19:20",
     "甲电池中报点评：量利齐升。上半年归母净利润同比增长25%，毛利率23.8%。"
     "上调盈利预测，预计2024-2025年EPS为2.36元、2.90元，目标价60.0元，评级上调为买入。风险：产能过剩。"),
    ("R06", "900002", "乙医药", "华东证券", "2023-04-10 20:00",
     "乙医药年报点评：创新药放量。2022年营收同比增长14%，研发费用率18.5%。"
     "预计2023-2024年EPS为1.10元、1.32元，目标价36.0元，给予增持评级。风险：集采降价、研发失败。"),
    ("R07", "900002", "乙医药", "南方证券", "2023-11-02 19:00",
     "乙医药三季报点评：集采影响超预期。前三季度归母净利润同比下降12%。"
     "下调预测，预计2023-2024年EPS为0.92元、1.05元，目标价28.0元，评级下调为中性。风险：医保谈判。"),
    ("R08", "900002", "乙医药", "华东证券", "2024-04-15 20:30",
     "乙医药年报点评：新品获批，拐点初现。2023年营收同比下降4%，两款新药获批上市。"
     "预计2024-2025年EPS为1.02元、1.25元，目标价33.0元，维持增持评级。风险：销售不及预期。"),
    ("R09", "900002", "乙医药", "北方证券", "2024-09-05 18:50",
     "乙医药中报点评：新品放量超预期。上半年归母净利润同比增长31%。"
     "上调预测，预计2024-2025年EPS为1.18元、1.46元，目标价40.0元，评级上调为买入。风险：竞争加剧。"),
]
RATING_SCORE = {"买入": 2, "增持": 1, "中性": 0, "减持": -1, "卖出": -2}


def corpus() -> pd.DataFrame:
    df = pd.DataFrame(DOCS, columns=["doc_id", "stock_code", "name", "broker", "publish_ts", "text"])
    df["publish_ts"] = pd.to_datetime(df["publish_ts"])
    return df


# ==================== 第一部分：场景与 llm-course 的衔接 ====================


def overview():
    print("=" * 60)
    print("第一部分：LLM 研报 / 财报分析的场景")
    print("=" * 60)
    print("""
    ┌──────────────────┬──────────────────────┬────────────────────────────────────────┐
    │ 场景             │ 输入                 │ 输出                                   │
    ├──────────────────┼──────────────────────┼────────────────────────────────────────┤
    │ 研报结构化       │ 券商研报 PDF         │ 评级、目标价、盈利预测、核心逻辑、风险 │
    │ 财报解读         │ 年报 / 季报          │ 关键指标、管理层表述变化、会计异常     │
    │ 调研纪要         │ 文字记录             │ 语气变化、指引上调 / 下调              │
    │ 跨文档问答       │ 多份研报 + 公告      │ "过去一年的主要分歧是什么？"           │
    └──────────────────┴──────────────────────┴────────────────────────────────────────┘

    与 llm-course 的对应：
      Phase 2  08-structured-extraction.py  → 研报字段抽取的 schema
      Phase 4  06-chroma-basics.py          → 元数据中存 stock_code、publish_ts
      Phase 5  06-self-query-retrieval.py   → 由查询生成元数据过滤（代码、publish_ts ≤ as_of）
      Phase 5  01-hybrid-search.py          → 股票代码、财务科目等精确词用 BM25
      Phase 10 06-faithfulness.py           → 本课的"数字溯源"是忠实度检查的程序化版本

    ⚠️ 本课语料为虚构公司（甲电池、乙医药）的模拟研报，代码为 9000xx，仅用于演示。
    """)


# ==================== 第二部分：普通 RAG vs Point-in-Time RAG ====================


class Retriever:
    """字符 n-gram TF-IDF 检索（中文无需分词；生产中可换成向量库 + BM25 混合检索）"""

    def __init__(self, docs: pd.DataFrame):
        self.docs = docs.reset_index(drop=True)
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        self.M = self.vec.fit_transform(self.docs["text"])

    def search(self, query: str, code: str | None = None, as_of: pd.Timestamp | None = None, k: int = 3):
        sim = (self.M @ self.vec.transform([query]).T).toarray().ravel()
        ok = np.ones(len(self.docs), dtype=bool)
        if code is not None:
            ok &= (self.docs["stock_code"] == code).to_numpy()
        if as_of is not None:  # PIT 过滤：发布时间必须早于"决策时刻"
            ok &= (self.docs["publish_ts"] <= as_of).to_numpy()
        sim = np.where(ok, sim, -np.inf)
        idx = [i for i in np.argsort(-sim)[:k] if np.isfinite(sim[i])]
        return self.docs.iloc[idx].assign(score=sim[idx])


def pit_rag_demo(docs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：普通 RAG vs Point-in-Time RAG")
    print("=" * 60)
    r = Retriever(docs)
    query = "甲电池 盈利预测 目标价 评级"
    as_of = pd.Timestamp("2023-09-30 15:00")
    print(f"\n    回测场景：在 {as_of:%Y-%m-%d %H:%M} 收盘后，为 900001 生成信号。查询：\"{query}\"\n")
    for label, kw in (("普通 RAG（不过滤时间）", {}), ("PIT RAG（publish_ts ≤ as_of）", {"as_of": as_of})):
        hits = r.search(query, code="900001", k=3, **kw)
        print(f"    {label}：")
        for _, h in hits.iterrows():
            flag = "❌ 未来文档" if h["publish_ts"] > as_of else "✅"
            print(f"      {h['doc_id']}  {h['publish_ts']:%Y-%m-%d}  {h['broker']}  相似度 {h['score']:.3f}  {flag}")
    # 系统性统计：每个月末做一次检索，普通 RAG 的 top-3 里有多少未来文档
    checks = pd.date_range("2023-04-30", "2024-08-31", freq="ME") + pd.Timedelta(hours=15)
    leak = []
    for t in checks:
        for code in ("900001", "900002"):
            hits = r.search("盈利预测 目标价 评级", code=code, k=3)
            leak.append((hits["publish_ts"] > t).mean())
    print(f"\n    在 {len(checks)} 个月末 × 2 只股票做检索：普通 RAG 的 top-3 结果中，"
          f"平均 {np.mean(leak):.0%} 是决策时刻之后才发布的研报")
    print("""
    对应到向量库（以 Chroma 为例），元数据里把 publish_ts 存成整数时间戳：
        collection.query(query_texts=[q], n_results=5,
                         where={"$and": [{"stock_code": code}, {"publish_ts": {"$lte": as_of_ts}}]})
    这就是 Phase 7 "财报按公告日对齐" 在 RAG 中的翻版。注意 as_of 要精确到时刻：
    晚上 19:30 发布的研报，不能用于当天 15:00 收盘时形成的信号。
    """)
    return r


# ==================== 第三部分：结构化抽取 ====================


class ReportSummary(BaseModel):
    doc_id: str
    stock_code: str
    rating: str = Field(description="买入/增持/中性/减持/卖出；原文未提及则为 '未提及'")
    target_price: float | None = Field(default=None, gt=0)
    eps_forecast: dict[str, float] = Field(description="年份 -> EPS 预测（元）")
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list, description="支撑上述数字的原文片段")


def rule_extract(doc: pd.Series) -> ReportSummary:
    """规则抽取（无密钥时的降级路径）：正则匹配固定句式"""
    t = doc["text"]
    rating = next((r for r in ("买入", "增持", "中性", "减持", "卖出")
                   if re.search(rf"(?:给予|维持|调整为|上调为|下调为)\s*{r}", t)), "未提及")
    tp = re.search(r"目标价(?:至)?\s*(\d+(?:\.\d+)?)\s*元", t)
    eps = {}
    m = re.search(r"(\d{4})-(\d{4})年EPS(?:分别)?为\s*(\d+\.\d+)元、\s*(\d+\.\d+)元", t)
    if m:
        eps = {m.group(1): float(m.group(3)), m.group(2): float(m.group(4))}
    risks = re.split(r"[、，]", re.split(r"风险(?:提示)?[:：]", t)[-1].rstrip("。"))
    evidence = [s for s in re.split(r"[。]", t) if "EPS" in s or "目标价" in s]
    return ReportSummary(doc_id=doc["doc_id"], stock_code=doc["stock_code"], rating=rating,
                         target_price=float(tp.group(1)) if tp else None, eps_forecast=eps,
                         risks=[x for x in risks if x], evidence=evidence)


EXTRACT_PROMPT = """从下面的研报片段中抽取信息，输出 JSON，字段：
doc_id, stock_code, rating(买入/增持/中性/减持/卖出/未提及), target_price(数字或 null),
eps_forecast(对象，键为年份字符串，值为 EPS 数字), risks(字符串列表), evidence(原文片段列表，必须逐字摘抄)。
只能使用原文中出现的数字，不要推算或补充。
doc_id={doc_id}  stock_code={code}
<report>{text}</report>"""


def extraction(docs: pd.DataFrame, llm: LLMClient):
    print("\n" + "=" * 60)
    print("第三部分：结构化抽取 —— 规则 / LLM，统一用 schema 校验")
    print("=" * 60)
    print(f"\n    {llm.describe()}")
    rows = [rule_extract(d) for _, d in docs.iterrows()]
    table = pd.DataFrame([{"doc_id": s.doc_id, "评级": s.rating, "目标价": s.target_price,
                           "EPS 预测": ", ".join(f"{k}:{v}" for k, v in s.eps_forecast.items()),
                           "风险": "、".join(s.risks)[:16]} for s in rows])
    print("\n    规则抽取结果（全部 9 篇）：\n")
    print(table.to_string(index=False))
    llm_rows = {}
    for _, d in docs.iloc[[2, 6]].iterrows():  # 有密钥时只调用 2 次，结果缓存
        raw = llm.json(EXTRACT_PROMPT.format(doc_id=d["doc_id"], code=d["stock_code"], text=d["text"]))
        if raw is None:
            continue
        try:
            llm_rows[d["doc_id"]] = ReportSummary.model_validate(raw)
        except ValidationError as e:
            print(f"    {d['doc_id']}：LLM 输出未通过 schema 校验（{e.error_count()} 处错误）→ 丢弃")
    if llm_rows:
        print("\n    LLM 抽取结果（与规则抽取对比）：")
        for k, s in llm_rows.items():
            rule = next(r for r in rows if r.doc_id == k)
            same = (s.rating == rule.rating, s.target_price == rule.target_price, s.eps_forecast == rule.eps_forecast)
            print(f"      {k}：评级 {s.rating}，目标价 {s.target_price}，EPS {s.eps_forecast} → "
                  f"与规则一致：评级 {same[0]}、目标价 {same[1]}、EPS {same[2]}")
    else:
        print("\n    （未调用 LLM：规则抽取足以处理固定句式；真实研报句式多变，LLM 的价值在这里）")
    print("""
    规则 vs LLM：规则透明、零成本，但遇到"上调至""较前次下调"这类变体就失效；
    LLM 能处理变体，但会出错、会编造 → 无论哪种来源，都要走同一个 schema 校验与数字溯源。
    研报中的盈利预测表在 PDF 里常是表格，切块时要整表保留（llm-course Phase 4 的文档加载与分块）。
    """)
    return rows, llm_rows


# ==================== 第四部分：数字溯源（反幻觉） ====================


def number_grounding(docs: pd.DataFrame, rule_rows, llm_rows):
    print("\n" + "=" * 60)
    print("第四部分：数字溯源 —— LLM 引用的数字必须在原文中找到")
    print("=" * 60)
    doc = docs.set_index("doc_id").loc["R03"]
    # 模拟的 LLM 响应：目标价被"记忆"中的另一篇研报污染，2024 年 EPS 被自行推算
    simulated = {"doc_id": "R03", "stock_code": "900001", "rating": "增持", "target_price": 49.5,
                 "eps_forecast": {"2023": 1.92, "2024": 2.40}, "risks": ["海外政策变化"],
                 "evidence": ["预计2023-2024年EPS为1.92元、2.33元"]}
    cases = {"规则抽取（R03）": next(r for r in rule_rows if r.doc_id == "R03"),
             "模拟 LLM 响应（含 2 处幻觉）": ReportSummary.model_validate(simulated)}
    cases.update({f"真实 LLM（{k}）": v for k, v in llm_rows.items()})
    texts = docs.set_index("doc_id")["text"]
    print(f"\n    原文（R03）：{doc['text']}\n")
    for name, s in cases.items():
        claims = {"target_price": s.target_price, **{f"EPS {y}": v for y, v in s.eps_forecast.items()}}
        issues = verify_numbers(claims, texts[s.doc_id])
        ev_bad = [e for e in s.evidence if e not in texts[s.doc_id]]
        status = "✅ 全部可溯源" if not issues and not ev_bad else "❌ " + "；".join(issues + [f"证据未逐字出现：{e[:20]}" for e in ev_bad])
        print(f"    {name:<22} {status}")
    print("""
    要点：
      · 校验规则很朴素："数字必须以相同数值出现在原文中"，却能拦住最常见的两类幻觉：
        串文档（把另一篇研报的目标价写进来）与自行推算（原文没有的数字）
      · evidence 字段要求逐字摘抄，便于人工复核，也能做字符串级检查
      · 生产中还要处理单位与格式差异（亿元 / 万元、1.2 与 1.20、全角数字）
      · 校验失败 → 丢弃或重试，绝不能"差不多就用"
    """)


# ==================== 第五部分：从研报到分析师因子（PIT） ====================


def analyst_factors(rule_rows, docs: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：从抽取结果到分析师因子（按 as_of 时点计算）")
    print("=" * 60)
    meta = docs.set_index("doc_id")
    ext = pd.DataFrame([{"doc_id": s.doc_id, "code": s.stock_code, "broker": meta.loc[s.doc_id, "broker"],
                         "ts": meta.loc[s.doc_id, "publish_ts"], "rating": RATING_SCORE.get(s.rating, np.nan),
                         "tp": s.target_price, **{f"eps_{y}": v for y, v in s.eps_forecast.items()}}
                        for s in rule_rows]).sort_values("ts")
    as_ofs = pd.date_range("2023-06-30", "2024-12-31", freq="QE") + pd.Timedelta(hours=15)
    rows = []
    for t in as_ofs:
        fy = str(t.year)  # 当年 EPS 预测
        for code, g in ext[ext["ts"] <= t].groupby("code"):  # 只用 as_of 之前发布的研报
            latest = g.sort_values("ts").groupby("broker").tail(1)  # 每家券商最新一份
            eps_now = latest[f"eps_{fy}"].mean() if f"eps_{fy}" in latest else np.nan
            prev = ext[(ext["code"] == code) & (ext["ts"] <= t - pd.DateOffset(months=3))]
            prev = prev.sort_values("ts").groupby("broker").tail(1)
            eps_prev = prev[f"eps_{fy}"].mean() if f"eps_{fy}" in prev and len(prev) else np.nan
            rows.append({"as_of": t.date(), "code": code, "覆盖券商数": len(latest), f"一致 EPS(当年)": eps_now,
                         "3 个月 EPS 修正": (eps_now - eps_prev) / abs(eps_prev) if eps_prev else np.nan,
                         "平均评级分": latest["rating"].mean()})
    out = pd.DataFrame(rows)
    print("\n    评级分：买入 2、增持 1、中性 0、减持 −1、卖出 −2；EPS 修正 = 当前一致预期相对 3 个月前的变化\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("""
    注意：
      · 每个 as_of 只使用此前发布的研报（PIT）；每家券商只取最新一份，避免重复计数
      · 年份切换时"当年 EPS"的口径会变（2023 → 2024），真实研究中常用 FY1/FY2 滚动口径
      · LLM 的增量价值在于降低抽取成本，而不是让 LLM 直接判断"该买哪只股票"
    """)
    return ext


def plot_revisions(ext):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 4))
    for code, g in ext.groupby("code"):
        ax.step(g["ts"], g["tp"], where="post", marker="o", label=f"{code} 目标价")
    ax.axvline(pd.Timestamp("2023-09-30"), color="red", ls="--", lw=1, label="as_of 示例：2023-09-30")
    ax.set_title("模拟研报的目标价变化：as_of 右侧的信息在回测当时不可用")
    ax.legend(fontsize=8)
    path = save_figure(fig, "07_report_target_price_pit.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    docs = corpus()
    overview()
    pit_rag_demo(docs)
    rule_rows, llm_rows = extraction(docs, LLMClient())
    number_grounding(docs, rule_rows, llm_rows)
    ext = analyst_factors(rule_rows, docs)
    plot_revisions(ext)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 llm-course Phase 4 的 Chroma 重写 Retriever，把 publish_ts 存为整数并用 where 过滤
    2. 给 verify_numbers 增加单位换算（"812.4 亿元" 与 "81240000000"）和全角数字支持
    3. 新增一篇句式不同的研报（如"目标价由 45 元上调至 52 元"），比较规则抽取与 LLM 抽取的结果
    4. 实现"评级变化"因子：最近 90 天内上调次数 − 下调次数，并说明它在什么时刻可以被交易
    """)


if __name__ == "__main__":
    main()
