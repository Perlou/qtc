"""
配对交易原理与选对
==================

学习目标：
    1. 理解配对交易"赚相对价格偏离修复的钱"，以及它为什么对大盘方向不敏感
    2. 掌握两种主流选对方法：距离法（SSD）与协整法
    3. 亲手验证多重检验陷阱：候选对越多，"通过检验"的假阳性越多
    4. 用样本外数据检验"选出来的配对"是否还有效
    5. 了解 A 股做空受限的现实，以及 ETF / 股指期货 / 商品期货等替代方案

核心概念：
    - 价差（Spread）：s_t = log P^A_t - β · log P^B_t
    - 距离法：标准化价格的偏差平方和（SSD）越小，两只股票"走得越近"
    - 协整法：用 Engle-Granger 检验价差是否平稳（02 课详解）
    - 多重检验：检验 N 次、显著性 5%，即使全是噪声也会有约 5% · N 次"显著"

前置知识：
    - Phase 3 假设检验、ADF 检验；Phase 4 样本内 / 样本外

环境要求：
    - pip install akshare pandas numpy statsmodels matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行：优先读本地缓存；无缓存时 utils 返回合成行情，
      本课检测到后改用模拟的协整价格组（打印 ⚠️ 使用合成数据）
"""

import itertools
import sys
import warnings
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tsa.stattools import coint

from stat_arb_utils import BANKS, FORM_END, STOCK_END, STOCK_START, TEST_START, load_bank_panel, ols_hedge
from utils.data import get_etf_daily, is_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)


def name(code: str) -> str:
    return BANKS.get(code, code)


# ==================== 第一部分：配对交易原理 ====================


def pairs_principle(logp: pd.DataFrame, market: pd.Series):
    print("=" * 60)
    print("第一部分：配对交易原理 —— 赚的是相对价格修复的钱")
    print("=" * 60)

    print("""
    醉汉遛狗：醉汉（股票 A）和狗（股票 B）各自乱走，但被一根绳子拴着。
    配对交易就是在"绳子被拉得很长"时押注它会收回来。

    价差 s_t = log P^A_t - β · log P^B_t
       │
    +2σ ─────────●──────────────────●──────  ← 做空价差（空 A、多 β 份 B）
       │       ╱   ╲              ╱  ╲
    均值 ──╲──╱─────╲─────╱╲─────╱────╲────  ← 平仓
       │    ╲╱       ╲   ╱  ╲   ╱
    -2σ ─────●────────●──────●─────────────  ← 做多价差（多 A、空 β 份 B）
       └─────────────────────────────────────▶ 时间

    为什么用对数价格？我们用的是等比后复权价，它的绝对水平取决于复权基准日，
    没有经济含义；取对数后，复权基准只影响截距，β 可以理解为"金额比"的近似。
    """)

    a, b = "601398", "601939"
    form = logp.loc[:FORM_END]
    _, beta = ols_hedge(form[a], form[b])
    ret = logp[[a, b]].diff().dropna()
    pair_ret = (ret[a] - beta * ret[b]) / (1 + abs(beta))  # 多 A 空 βB，两腿金额之和 = 1
    mret = np.log(market).diff().reindex(ret.index)

    def mkt_beta(r: pd.Series) -> float:
        d = pd.concat([r, mret], axis=1).dropna()
        return float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 1].var())

    rows = [
        [f"{name(a)}（单只）", mkt_beta(ret[a]), ret[a].corr(mret), ret[a].std() * np.sqrt(252)],
        [f"{name(b)}（单只）", mkt_beta(ret[b]), ret[b].corr(mret), ret[b].std() * np.sqrt(252)],
        [f"配对组合（多{name(a)} 空{beta:.2f}份{name(b)}）", mkt_beta(pair_ret), pair_ret.corr(mret),
         pair_ret.std() * np.sqrt(252)],
    ]
    df = pd.DataFrame(rows, columns=["资产", "对市场 Beta", "与市场相关系数", "年化波动"]).round(3)
    print(f"📊 对冲比 β（形成期 OLS）= {beta:.3f}；市场指数：沪深 300（无真实数据时为代理，见上方 ⚠️ 提示）")
    print(df.to_string(index=False))
    print("""
    💡 配对组合的市场 Beta 明显小于单只股票：大盘涨跌被多空两条腿大致抵消，
       组合只暴露于"A 相对 B"的表现 —— 这就是"市场中性"。
       但"中性于大盘"不等于"没有风险"：两只股票之间的关系本身可能改变（08 课）。
    """)


# ==================== 第二部分：选对 —— 基本面过滤 + 距离法 ====================


def distance_method(prices: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第二部分：选对 —— 基本面过滤 + 距离法（SSD）")
    print("=" * 60)

    print("""
    统计检验之前，先用经济逻辑缩小范围：
    ┌────────────┬──────────────────────────┬──────────────────────────┐
    │ 维度       │ 好的配对                 │ 差的配对                 │
    ├────────────┼──────────────────────────┼──────────────────────────┤
    │ 行业       │ 同一细分行业（两家大行） │ 跨行业（统计上偶然相关） │
    │ 商业模式   │ 收入、成本结构相似       │ 一家正在转型             │
    │ 流动性     │ 都足够大、可交易         │ 一只流动性很差           │
    │ 可做空     │ 两条腿都能交易           │ 空头腿借不到券           │
    └────────────┴──────────────────────────┴──────────────────────────┘

    本课的候选池：9 只 A 股上市银行（同一行业、大市值、流动性好）
    距离法（Gatev 等 2006）：形成期内把价格标准化为"起点 = 1"的累计收益指数，
    计算每对的偏差平方和 SSD = Σ (P̃A - P̃B)²，SSD 越小越"像"。
    """)
    form = prices.loc[:FORM_END]
    norm = form / form.iloc[0]
    rows = []
    for a, b in itertools.combinations(prices.columns, 2):
        ssd = float(((norm[a] - norm[b]) ** 2).sum())
        rows.append([a, b, f"{name(a)}-{name(b)}", ssd])
    ssd = pd.DataFrame(rows, columns=["A", "B", "配对", "SSD"]).sort_values("SSD").reset_index(drop=True)
    print(f"📋 形成期 {form.index[0].date()} ~ {form.index[-1].date()}，共 {len(ssd)} 个候选对（C(9,2)）")
    print("   SSD 最小的 5 对：")
    print(ssd.head(5)[["配对", "SSD"]].round(3).to_string(index=False))
    print("\n   ⚠️ SSD 小只说明“过去走得近”，不代表“走远了会回来”。")
    return ssd


# ==================== 第三部分：协整法 ====================


def coint_method(logp: pd.DataFrame, ssd: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("第三部分：协整法 —— 价差是否平稳")
    print("=" * 60)

    form = logp.loc[:FORM_END]
    rows = []
    for a, b in itertools.combinations(logp.columns, 2):
        p_ab = coint(form[a], form[b])[1]  # 用协整临界值（MacKinnon），不是 adfuller 的临界值
        p_ba = coint(form[b], form[a])[1]
        rows.append([a, b, f"{name(a)}-{name(b)}", p_ab, p_ba, max(p_ab, p_ba)])
    res = pd.DataFrame(rows, columns=["A", "B", "配对", "p(A~B)", "p(B~A)", "p_max"])
    res = res.sort_values("p_max").reset_index(drop=True)
    print("📋 Engle-Granger 检验（两个方向都做，取较大的 p 值更保守）")
    print(res.head(8)[["配对", "p(A~B)", "p(B~A)", "p_max"]].round(4).to_string(index=False))
    n_pass = int((res["p_max"] < 0.05).sum())
    print(f"\n   两个方向都 p < 0.05 的配对：{n_pass} / {len(res)}")

    merged = res.merge(ssd[["A", "B", "SSD"]], on=["A", "B"])
    rank_corr = merged["p_max"].rank().corr(merged["SSD"].rank())
    overlap = set(res.head(5)["配对"]) & set(ssd.head(5)["配对"])
    print(f"   SSD 排名与协整 p 值排名的相关系数：{rank_corr:.2f}")
    print(f"   两种方法的前 5 名中重合的配对：{sorted(overlap) if overlap else '无'}")
    print("""
    💡 两种方法回答的问题不同：距离法偏好"整体走势像"的配对，
       协整法关心"价差会不会回来"，并且能直接给出对冲比 β。
    """)
    return res


# ==================== 第四部分：多重检验陷阱 ====================


def multiple_testing(res: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：多重检验陷阱 —— 候选对越多，假阳性越多")
    print("=" * 60)

    for n in [9, 30, 100, 300]:
        k = comb(n, 2)
        print(f"   {n:>4d} 只股票 → {k:>6d} 个配对 → 全是噪声时，5% 水平下预计约 {k * 0.05:>7.1f} 个“显著”")

    print("\n🎲 蒙特卡洛：435 对（= C(30,2)）相互独立的随机游走，每对之间一定不协整，逐对做协整检验")
    rng = np.random.default_rng(1)
    m, n_days = comb(30, 2), 750
    pvals = np.array([
        coint(np.cumsum(rng.normal(0, 0.02, n_days)), np.cumsum(rng.normal(0, 0.02, n_days)))[1]
        for _ in range(m)
    ])
    print(f"   配对数 {m}，p < 0.05 的有 {(pvals < 0.05).sum()} 个（{(pvals < 0.05).mean():.1%}，"
          f"理论约 5%），全部是假阳性")
    print("   注：真实选对时同一只股票出现在多个配对里，p 值彼此相关，假阳性个数会围绕期望值大幅波动")
    print(f"   Bonferroni 校正（阈值 0.05 / {m} = {0.05 / m:.2e}）后仍“显著”：{(pvals < 0.05 / m).sum()} 个")

    def bh_reject(p: np.ndarray, q: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg：控制错误发现率（FDR）"""
        order = np.argsort(p)
        thresh = q * np.arange(1, len(p) + 1) / len(p)
        passed = p[order] <= thresh
        k = passed.nonzero()[0].max() + 1 if passed.any() else 0
        out = np.zeros(len(p), dtype=bool)
        out[order[:k]] = True
        return out

    print(f"   BH-FDR 校正后仍“显著”：{bh_reject(pvals).sum()} 个")

    bank_p = res["p_max"].to_numpy()
    print(f"\n📋 回到银行股：{len(bank_p)} 个候选对中 {int((bank_p < 0.05).sum())} 个 p < 0.05，"
          f"纯噪声下预计约 {len(bank_p) * 0.05:.1f} 个")
    print(f"   Bonferroni 后 {int((bank_p < 0.05 / len(bank_p)).sum())} 个，BH-FDR 后 {int(bh_reject(bank_p).sum())} 个")
    print("""
    💡 对策：
       1. 先用行业 / 基本面过滤，把候选对数从几千降到几十
       2. 更严格的显著性水平，或 Bonferroni / FDR 校正
       3. 最重要：样本内选对，样本外检验（下一部分）
    """)


# ==================== 第五部分：样本外检验 ====================


def out_of_sample(logp: pd.DataFrame, res: pd.DataFrame, ssd: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：样本外检验 —— 选出来的配对还有效吗？")
    print("=" * 60)

    test = logp.loc[TEST_START:]
    form = logp.loc[:FORM_END]
    chosen = res[res["p_max"] < 0.10].head(6)
    if chosen.empty:
        chosen = res.head(3)
    rows = []
    for _, r in chosen.iterrows():
        _, beta = ols_hedge(form[r["A"]], form[r["B"]])
        s_form = form[r["A"]] - beta * form[r["B"]]
        s_test = test[r["A"]] - beta * test[r["B"]]
        drift = (s_test.mean() - s_form.mean()) / s_form.std()  # 样本外均值漂移了几个样本内标准差
        rows.append([r["配对"], r["p_max"], max(coint(test[r["A"]], test[r["B"]])[1],
                                                coint(test[r["B"]], test[r["A"]])[1]),
                     drift, s_test.std() / s_form.std()])
    df = pd.DataFrame(rows, columns=["配对", "形成期 p", "交易期 p（重新检验）", "价差均值漂移(σ)", "波动比(外/内)"])
    print(f"   形成期 {form.index[0].date()} ~ {form.index[-1].date()}；交易期 {test.index[0].date()} ~ {test.index[-1].date()}")
    print("   用形成期估计的 β 构造价差，看它在交易期是否还“围绕原来的均值波动”：")
    print(df.round(3).to_string(index=False))
    still = int((df["交易期 p（重新检验）"] < 0.05).sum())
    print(f"\n   形成期入选的 {len(df)} 对中，交易期重新检验仍 p < 0.05 的：{still} 对")
    print("""
    💡 观察要点：
       - "价差均值漂移"绝对值大于 1–2 个标准差，意味着用旧均值做 Z-score 会持续给出单边信号
       - 协整关系对样本区间很敏感：同一对股票换一段时间，结论可能完全不同
       - 这就是为什么回测必须"形成期估计参数，交易期使用"，并且定期重新检验
    """)
    return chosen


# ==================== 第六部分：A 股做空约束与替代方案 ====================


def a_share_constraints():
    print("\n" + "=" * 60)
    print("第六部分：A 股做空约束与替代方案")
    print("=" * 60)

    print("""
    配对交易需要一条空头腿。A 股个股做空只能通过融券，现实约束很多：
    ┌──────────────────────┬────────────────────────────────────────────────┐
    │ 约束                 │ 说明（以监管和券商最新规则为准）               │
    ├──────────────────────┼────────────────────────────────────────────────┤
    │ 券源有限             │ 想借的券经常借不到，尤其是中小市值股票         │
    │ 融券费率高           │ 年化费率通常远高于融资利率，且随时可能调整     │
    │ 监管收紧             │ 近年融券及转融券业务明显收紧                   │
    │ 可能被召回           │ 出借方可提前收回，空头腿被迫平仓               │
    └──────────────────────┴────────────────────────────────────────────────┘

    可行的替代方案：
      1. ETF 配对：跟踪相近指数的 ETF（部分 ETF 可融券，或用股指期货替代空头腿）
      2. 期货品种间配对：螺纹 vs 热卷、豆油 vs 棕榈油，做空无障碍（05 课跨期套利）
      3. "多头 + 股指期货对冲"：只做多偏低的一边，用 IF/IC 对冲市场风险（06 课、Phase 7）
      4. 港股、美股等做空便利的市场
    """)

    etfs = {}
    for code, label in [("510300", "华泰柏瑞沪深300ETF"), ("159919", "嘉实沪深300ETF")]:
        df = get_etf_daily(code, STOCK_START, STOCK_END, adjust="hfq")  # 先读缓存，离线且无缓存才合成
        if is_synthetic(df):
            print("⏭️  ETF 为合成数据（两条独立合成序列之间的配对没有意义），跳过演示")
            return
        etfs[f"{code} {label}"] = np.log(df["close"])
    pair = pd.DataFrame(etfs).dropna()
    a, b = pair.columns
    _, beta = ols_hedge(pair[a], pair[b])
    spread = pair[a] - beta * pair[b]
    cost_bp = 4 * (0.0001 + 0.0003) * 1e4  # 4 笔（两腿开 + 平）× (佣金 + 滑点) 示例，ETF 免印花税
    print("📊 两只跟踪同一指数（沪深 300）的 ETF：")
    print(f"   日收益相关系数：{pair[a].diff().corr(pair[b].diff()):.4f}")
    print(f"   协整检验 p 值：{coint(pair[a], pair[b])[1]:.4f}")
    print(f"   价差（β={beta:.3f}）标准差：{spread.std() * 1e4:.1f} bp；日变化标准差：{spread.diff().std() * 1e4:.1f} bp")
    print(f"   一次完整往返的示例成本：约 {cost_bp:.0f} bp")
    print("""
    💡 观察要点：
       - 日收益相关系数接近 1，但协整检验结论要看上面实际的 p 值 ——
         管理费差异、分红时点与金额、跟踪误差、复权方式都会让价差缓慢漂移，"高相关 ≠ 协整"
       - 价差的波动幅度要和交易成本放在一起比较：协整只是必要条件，扣除成本后有利可图才是目标
    """)


def plot_pair(prices: pd.DataFrame, ssd: pd.DataFrame, res: pd.DataFrame, is_synth: bool):
    setup_chinese_font()
    a, b = ssd.iloc[0][["A", "B"]]
    c, d = res.iloc[0][["A", "B"]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, (x, y), title in [(axes[0], (a, b), "SSD 最小"), (axes[1], (c, d), "协整 p 值最小")]:
        norm = prices[[x, y]] / prices[[x, y]].iloc[0]
        ax.plot(norm.index, norm[x], label=name(x))
        ax.plot(norm.index, norm[y], label=name(y))
        ax.axvline(pd.Timestamp(TEST_START), color="gray", ls="--", lw=1)
        ax.set_title(f"{title}的配对（标准化价格，虚线右侧为样本外）{'【合成数据】' if is_synth else ''}")
        ax.legend()
    fig.tight_layout()
    path = save_figure(fig, "01_pair_candidates.png", subdir="phase-6")
    print(f"\n🖼️  图已保存：{path}")


# ==================== 主函数 ====================


def main():
    prices, market, is_synth = load_bank_panel()
    logp = np.log(prices)
    print(f"📊 银行股等比后复权价格：{len(prices)} 个交易日，{prices.shape[1]} 只"
          f"{'（合成数据）' if is_synth else ''}\n")

    pairs_principle(logp, market)
    ssd = distance_method(prices)
    res = coint_method(logp, ssd)
    multiple_testing(res)
    out_of_sample(logp, res, ssd)
    a_share_constraints()
    plot_pair(prices, ssd, res, is_synth)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把候选池换成白酒股（600519、000858、000568、600809、002304），重复距离法与协整法选对
    2. 把形成期改为 2019-2022、交易期改为 2023-2024，入选配对会变吗？样本外仍有效的有几对？
    3. 修改第四部分的蒙特卡洛：资产数从 30 增加到 60，假阳性个数大约变成多少？与 C(60,2)×5% 比较
    4. 思考：如果某个配对在形成期 p=0.001，但两家公司刚宣布合并，你还会交易它吗？为什么？
    """)


if __name__ == "__main__":
    main()
