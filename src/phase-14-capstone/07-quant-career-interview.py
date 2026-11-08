"""
量化岗位面试：概率题、统计题、编程题、策略题
============================================

学习目标：
    1. 了解量化岗位分类与典型面试流程
    2. 完成 20 道面试题：概率题用蒙特卡洛验证解析解，统计题用模拟验证结论
    3. 编程题写成函数，并用暴力解法对拍测试（tests/test_interview.py 也会调用这些函数）
    4. 策略题按"回答框架"组织要点，并结合本课程的平台输出举例

核心概念：
    - 面试官考的是推导过程与严谨性，不是背答案：每道题都要能"用代码验证"
    - 解析解与模拟结果对不上 → 要么推导错了，要么模拟写错了，两者都值得深究

前置知识：
    - Phase 3（概率统计、蒙特卡洛）、Phase 4（Sharpe、DSR）、Phase 7（Rank IC）、Phase 11（平方根法则）
    - 本阶段 CONCEPT.md 第 8 节

环境要求：
    - 不需要网络；只用 numpy / pandas / scipy
"""

import math
import random
import sys
from collections import OrderedDict, deque
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

SEED = 2026
RESULTS: list[dict] = []  # 每道题的"解析解 vs 模拟"记录，最后汇总


def record(no: int, kind: str, title: str, analytic=None, simulated=None, tol=None, note: str = "") -> None:
    ok = None
    if analytic is not None and simulated is not None and tol is not None:
        ok = abs(float(analytic) - float(simulated)) <= tol
    RESULTS.append({"题号": no, "类型": kind, "题目": title, "解析解": analytic, "模拟/验证": simulated,
                    "一致": {True: "✅", False: "❌", None: "—"}[ok], "备注": note})


def header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==================== 第一部分：岗位与面试流程 ====================


def careers_overview():
    header("第一部分：量化岗位与面试流程")
    roles = pd.DataFrame(
        [
            ["量化研究员 QR", "因子挖掘、模型研究、策略开发", "统计、机器学习、金融直觉"],
            ["量化交易员 QT", "策略运行、盘中决策、做市/套利", "概率直觉、快速决策、风险意识"],
            ["量化开发 QD", "交易系统、回测平台、数据基础设施", "系统设计、C++/Python、性能优化"],
            ["风控/模型验证", "风险度量、模型审核、限额管理", "统计、风险模型、严谨"],
            ["数据工程/另类数据", "数据采集、清洗、产品化", "数据工程、爬虫、NLP"],
        ],
        columns=["岗位", "主要工作", "核心能力"],
    )
    print(roles.to_string(index=False))
    print("""
    典型流程：简历筛选 → 笔试/在线测评 → 技术面（1–3 轮）→ 主管面 → Offer
    技术面题型：① 概率 ② 统计与计量 ③ 编程 ④ 策略/研究 ⑤ 项目深挖

    💡 资深全栈工程师的切入点：量化开发（QD）。用 mini-quant 平台证明工程能力，
       用研究报告（05 课）和尽调结论（06 课）证明你"知道回测数字的局限"——
       项目深挖环节最常被问倒的，恰恰是简历上那个"回测夏普 2.5"。
    """)


# ==================== 第二部分：概率题（蒙特卡洛验证） ====================


def expected_flips_analytic(pattern: str) -> int:
    """
    等到某个 H/T 模式出现的期望次数：Σ 2^k，对所有"前缀 = 后缀"的长度 k 求和（Conway 自相关）。
    HH：前缀 H = 后缀 H、HH = HH → 2 + 4 = 6；HT：只有整体 → 4
    """
    n = len(pattern)
    return sum(2**k for k in range(1, n + 1) if pattern[:k] == pattern[-k:])


def flips_until(pattern: str, trials: int, rng: random.Random) -> float:
    total = 0
    n = len(pattern)
    for _ in range(trials):
        seq = ""
        while seq[-n:] != pattern:
            seq += "H" if rng.random() < 0.5 else "T"
        total += len(seq)
    return total / trials


def dice_reroll_value(rerolls: int) -> Fraction:
    """可以重掷 k 次的骰子的最优期望（倒推：第一次点数 > 继续的价值才接受）"""
    v = Fraction(7, 2)  # 没有重掷机会：期望 3.5
    for _ in range(rerolls):
        v = sum(max(Fraction(x), v) for x in range(1, 7)) / 6
    return v


def q01_hh_ht():
    header("题 1（概率）：连续两次正面 HH 的期望次数？HT 呢？")
    rng = random.Random(SEED)
    sims = {p: flips_until(p, 40_000, rng) for p in ("HH", "HT")}
    for p in ("HH", "HT"):
        print(f"    {p}: 解析解 {expected_flips_analytic(p)}，模拟 {sims[p]:.3f}")
    print("""
    推导（状态法）：E0 = 1 + E1/2 + E0/2，E1 = 1 + 0/2 + E0/2 → E0 = 6
    HT 只要 4：出现 H 之后再出 H 不会"前功尽弃"（仍处在"上一次是 H"的状态）
    """)
    record(1, "概率", "等到 HH 的期望次数", 6, sims["HH"], 0.15)
    record(1, "概率", "等到 HT 的期望次数", 4, sims["HT"], 0.1)


def q02_dice_reroll():
    header("题 2（概率）：可以重掷一次的骰子，最优期望收益？重掷两次呢？")
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    a, b, c = rng.integers(1, 7, (3, n))
    one = np.where(a >= 4, a, b).mean()  # 第一次 ≥ 4 接受，否则重掷
    two = np.where(a >= 5, a, np.where(b >= 4, b, c)).mean()  # 先用 4.25 的门槛，再用 3.5
    v1, v2 = dice_reroll_value(1), dice_reroll_value(2)
    print(f"    重掷一次：解析解 {v1} = {float(v1):.4f}，模拟 {one:.4f}")
    print(f"    重掷两次：解析解 {v2} = {float(v2):.4f}，模拟 {two:.4f}")
    print("    思路：从最后一次往前倒推，'继续的价值'就是接受门槛（与美式期权提前行权同构）")
    record(2, "概率", "重掷一次的骰子期望", float(v1), one, 0.01)
    record(2, "概率", "重掷两次的骰子期望", float(v2), two, 0.01)


def q03_broken_stick():
    header("题 3（概率）：木棍随机折成三段，能构成三角形的概率？")
    rng = np.random.default_rng(SEED)
    x, y = rng.random((2, 2_000_000))
    a, b = np.minimum(x, y), np.maximum(x, y)
    p = np.mean((a < 0.5) & (b - a < 0.5) & (1 - b < 0.5))
    print(f"    解析解 1/4，模拟 {p:.4f}")
    print("    思路：构成三角形 ⇔ 每段都 < 1/2；在单位正方形里画出可行区域，面积为 1/4")
    record(3, "概率", "折棍成三角形", 0.25, p, 0.002)


def q04_coupon():
    header("题 4（概率）：掷骰子直到六个点数都出现，期望次数？")
    rng = np.random.default_rng(SEED)
    # 向量化：集齐第 k+1 种之前的等待次数 ~ 几何分布(p = (6−k)/6)
    waits = sum(rng.geometric((6 - k) / 6, 500_000) for k in range(6))
    analytic = 6 * sum(1 / i for i in range(1, 7))
    print(f"    解析解 6·(1 + 1/2 + … + 1/6) = {analytic:.4f}，模拟 {waits.mean():.4f}")
    record(4, "概率", "集齐骰子六面", analytic, waits.mean(), 0.05)


def q05_birthday():
    header("题 5（概率）：至少多少人，'有人同一天生日'的概率才超过 50%？")
    p_distinct, n = 1.0, 0
    while 1 - p_distinct <= 0.5:
        p_distinct *= (365 - n) / 365
        n += 1
    rng = np.random.default_rng(SEED)
    bdays = rng.integers(0, 365, (200_000, n))
    srt = np.sort(bdays, axis=1)
    sim = np.mean((np.diff(srt, axis=1) == 0).any(axis=1))
    print(f"    解析解 n = {n}（P = {1 - p_distinct:.4f}），模拟 n={n} 时 P = {sim:.4f}")
    print("    启发：365 天里只要 23 人就 > 50%——'巧合'在大样本里非常常见（联想多重检验）")
    record(5, "概率", "生日问题：23 人时的碰撞概率", 1 - p_distinct, sim, 0.005, note=f"n = {n}")


def q16_monty_hall():
    header("题 16（概率）：三门问题，换门后获胜概率为什么是 2/3？")
    rng = np.random.default_rng(SEED)
    n = 1_000_000
    car, pick = rng.integers(0, 3, (2, n))
    # 主持人打开一扇"非你所选、且没有车"的门；换门 ⇔ 选另一扇未开的门
    # 若首次选中了车（概率 1/3），换门必输；否则换门必赢
    switch_win = np.mean(car != pick)
    stay_win = np.mean(car == pick)
    print(f"    换门胜率：解析解 2/3，模拟 {switch_win:.4f}；不换：{stay_win:.4f}")
    print("""
    条件概率：设你选 1 号门，主持人开 3 号门（H3）。
      P(H3 | 车在1) = 1/2，P(H3 | 车在2) = 1，P(H3 | 车在3) = 0
      P(车在2 | H3) = (1 × 1/3) / (1/2 × 1/3 + 1 × 1/3) = 2/3
    关键：主持人的行为依赖于车的位置，他开门这件事本身带有信息。
    """)
    record(16, "概率", "三门问题换门胜率", 2 / 3, switch_win, 0.002)


def q17_max_uniform():
    header("题 17（概率）：两个独立 U(0,1) 的最大值的期望？")
    rng = np.random.default_rng(SEED)
    m = rng.random((2, 2_000_000)).max(axis=0).mean()
    print(f"    解析解 2/3，模拟 {m:.4f}")
    print("    推导：P(max ≤ x) = x²，密度 2x，E = ∫ x·2x dx = 2/3；n 个时 E = n/(n+1)")
    record(17, "概率", "max(U1, U2) 的期望", 2 / 3, m, 0.002)


# ==================== 第三部分：统计题 ====================


def q06_uncorrelated_not_independent():
    header("题 6（统计）：X ~ N(0,1)，X 与 X² 相关吗？独立吗？")
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(1_000_000)
    rho = np.corrcoef(x, x**2)[0, 1]
    # 独立性检验：若独立，则 |X| 大小不应影响 X² 的分布——显然不成立
    p_big = np.mean(x**2 > 1)
    p_big_given = np.mean((x**2 > 1)[np.abs(x) > 1])
    print(f"    corr(X, X²) = {rho:+.4f}（理论 0：Cov = E[X³] − E[X]E[X²] = 0）")
    print(f"    P(X² > 1) = {p_big:.3f}，但 P(X² > 1 | |X| > 1) = {p_big_given:.3f} → 不独立")
    print("    考点：相关系数只度量线性关系。因子研究中的 Pearson IC 也一样——U 形关系会被漏掉")
    record(6, "统计", "corr(X, X²)", 0.0, rho, 0.005)


def q07_regression_asymmetry():
    header("题 7（统计）：y 对 x 回归斜率 β1，x 对 y 回归斜率 β2，二者关系？")
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 2, 200_000)
    y = 0.5 * x + rng.normal(0, 1.5, 200_000)
    b1 = np.cov(x, y)[0, 1] / x.var(ddof=1)
    b2 = np.cov(x, y)[0, 1] / y.var(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    print(f"    β1 = {b1:.4f}，β2 = {b2:.4f}，β1·β2 = {b1 * b2:.4f}，ρ² = {rho**2:.4f}，1/β1 = {1 / b1:.4f}")
    print("    结论：β1·β2 = ρ² ≤ 1，β2 ≠ 1/β1（除非完全相关）。配对交易里用哪个做因变量会改变对冲比例")
    record(7, "统计", "β1·β2 = ρ²", rho**2, b1 * b2, 1e-9)


def q08_sharpe_annualization():
    header("题 8（统计）：日均收益 / 日标准差 = 0.1，年化夏普约多少？有自相关呢？")
    print(f"    独立同分布：0.1 × √252 = {0.1 * math.sqrt(252):.3f}")
    rng = np.random.default_rng(SEED)
    phi, n_days, n_paths = 0.2, 252, 4000
    eps = rng.normal(0, 0.01, (n_paths, n_days + 50))
    r = np.zeros_like(eps)
    for t in range(1, eps.shape[1]):
        r[:, t] = phi * r[:, t - 1] + eps[:, t]
    r = r[:, 50:] + 0.001  # 每日均值 0.1%
    naive = r.mean(axis=1) / r.std(axis=1, ddof=1) * math.sqrt(252)
    true_annual = (r.sum(axis=1).mean()) / r.sum(axis=1).std(ddof=1)  # 年收益的均值 / 年收益标准差
    lo_factor = math.sqrt((1 + phi) / (1 - phi))  # 大样本下年度方差放大倍数的平方根（近似）
    print(f"    AR(1) φ = {phi}：简单 √252 年化 = {naive.mean():.3f}，按年收益计算的真实夏普 ≈ {true_annual:.3f}")
    print(f"    比值 ≈ {naive.mean() / true_annual:.3f}，近似公式 √((1+φ)/(1−φ)) = {lo_factor:.3f}（Lo, 2002）")
    print("    结论：正自相关时 √252 年化会高估夏普；平滑估值的资产（私募、非流动资产）尤其明显")
    record(8, "统计", "正自相关下 √252 年化高估倍数", lo_factor, naive.mean() / true_annual, 0.05)


def q09_multiple_testing():
    header("题 9（统计）：测试了 100 个策略，最好的 t = 2.5，可信吗？")
    rng = np.random.default_rng(SEED)
    n_strat, n_obs, n_rep = 100, 756, 400
    best_t = np.empty(n_rep)
    n_sig = np.empty(n_rep)
    for k in range(n_rep):
        r = rng.normal(0, 0.01, (n_strat, n_obs))  # 100 个真实收益为 0 的策略
        t = r.mean(axis=1) / r.std(axis=1, ddof=1) * math.sqrt(n_obs)
        best_t[k], n_sig[k] = t.max(), (np.abs(t) > 1.96).sum()
    bonf = stats.norm.ppf(1 - 0.05 / 100 / 2)
    print(f"    100 个无效策略中：平均 {n_sig.mean():.1f} 个 |t| > 1.96（理论 5 个）")
    print(f"    最好的 t 中位数 {np.median(best_t):.2f}，P(最大 t ≥ 2.5) = {np.mean(best_t >= 2.5):.0%}")
    print(f"    Bonferroni 校正：双侧 α = 0.05/100 → t 门槛 {bonf:.2f}")
    print("    结论：t = 2.5 在 100 次尝试后很可能只是运气；报告试验次数并计算 DSR（Phase 4、本阶段 05 课）")
    record(9, "统计", "100 个无效策略中显著的个数", 5.0, n_sig.mean(), 0.5)


def q10_sharpe_se():
    header("题 10（统计）：3 年日度数据估计出年化夏普 1.0，标准误大约多少？")
    sr, years, n = 1.0, 3, 252 * 3
    sr_d = sr / math.sqrt(252)  # 日夏普
    se_daily = math.sqrt((1 + 0.5 * sr_d**2) / n) * math.sqrt(252)  # 日度观测：公式里用"日"夏普
    se_annual_obs = math.sqrt((1 + 0.5 * sr**2) / years)  # 年度观测（只有 3 个数据点）
    rng = np.random.default_rng(SEED)
    r = rng.normal(sr_d * 0.01, 0.01, (20_000, n))
    est = r.mean(axis=1) / r.std(axis=1, ddof=1) * math.sqrt(252)
    print(f"""
    渐近公式（Lo, 2002，收益独立同分布）：Var(SR̂) ≈ (1 + SR²/2) / n，SR 与 n 必须是同一频率
      日度数据：SR_日 = {sr_d:.4f}，n = {n} → 年化标准误 = √((1 + SR_日²/2)/n) × √252 = {se_daily:.3f}
      模拟（2 万条 3 年日收益路径）：年化夏普估计值的标准差 = {est.std():.3f}
      若把年化夏普直接代入 √((1 + SR²/2)/T)（T 以年计）= {se_annual_obs:.3f}：
        这对应"只有 3 个年度观测"的情形；日度数据下 SR²/2 项应使用日夏普，几乎可以忽略
    结论：无论哪种口径，95% 置信区间约为 1.0 ± {1.96 * se_daily:.1f}（日度）~ ± {1.96 * se_annual_obs:.1f}（年度）
          → 3 年数据很难区分夏普 1.0 和 0
    联系：本课程样本外只有 24 个月（05、06 课），这正是尽调结论常为"证据不足"的原因
    """)
    record(10, "统计", "夏普估计标准误（3 年日度，SR=1）", se_daily, est.std(), 0.02,
           note=f"年度观测口径 {se_annual_obs:.2f}")


def q18_p_hacking():
    header("题 18（统计）：什么是 p-hacking？在因子研究中有哪些表现？")
    print("""
    定义：反复尝试不同的数据处理和检验方式，只报告"显著"的那一个，使 p 值失去意义。

    因子研究中的表现：
      ① 换回看窗口（20/40/60/120 日）、换去极值阈值、换中性化方式，直到 IC 显著
      ② 挑样本区间：去掉"表现不好的年份"，或在样本外不理想后回头改参数（"偷看"样本外）
      ③ 挑股票池：全 A 不显著就换中证 500、再换中证 1000
      ④ 只报告多空组合中好看的那一头，或只报告扣费前
      ⑤ 事后编故事：先找到规律，再补经济逻辑

    对策：研究日志记录每一次尝试（本平台的 TrialLog）、预先写下假设与样本划分、
          报告试验次数并计算 DSR、样本外只评估一次
    """)
    record(18, "统计", "p-hacking 的表现与对策", note="要点题")


# ==================== 第四部分：编程题（函数 + 对拍测试） ====================


def max_drawdown(nav: list[float]) -> float:
    """题 11：O(n) 最大回撤（返回正数）"""
    if not nav:
        raise ValueError("空序列")
    peak, mdd = nav[0], 0.0
    for x in nav:
        peak = max(peak, x)
        mdd = max(mdd, 1 - x / peak)
    return mdd


def max_drawdown_bruteforce(nav: list[float]) -> float:
    return max([0.0] + [1 - nav[j] / nav[i] for i in range(len(nav)) for j in range(i, len(nav))])


def sliding_max(xs: list[float], k: int) -> list[float]:
    """题 12：O(n) 滑动窗口最大值（单调队列，dq 保存下标，对应值单调递减）"""
    if k <= 0:
        raise ValueError("k 必须为正")
    dq, out = deque(), []
    for i, x in enumerate(xs):
        while dq and xs[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(xs[dq[0]])
    return out


def sliding_max_bruteforce(xs: list[float], k: int) -> list[float]:
    return [max(xs[i - k + 1 : i + 1]) for i in range(k - 1, len(xs))]


def rank_ic_by_date(panel: pd.DataFrame, factor: str = "factor", ret: str = "fwd_ret",
                    date: str = "date", min_obs: int = 3) -> pd.Series:
    """题 19：长表面板按日期分组计算截面 Rank IC（Spearman = 排名的 Pearson）"""
    def one(g: pd.DataFrame) -> float:
        g = g[[factor, ret]].dropna()
        if len(g) < min_obs:
            return np.nan
        return g[factor].rank().corr(g[ret].rank())

    return panel.groupby(date)[[factor, ret]].apply(one).rename("rank_ic")


def reservoir_sample(stream, k: int, rng: random.Random) -> list:
    """附加题：水塘抽样——单遍扫描、O(k) 内存，从未知长度的流中等概率抽 k 个"""
    sample = []
    for i, x in enumerate(stream):
        if i < k:
            sample.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = x
    return sample


class LRUCache:
    """附加题：LRU 缓存（行情快照缓存、因子缓存的常见实现）"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        if key not in self._d:
            return default
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


def run_coding_tests() -> list[tuple[str, bool]]:
    """对拍：随机输入下与暴力解法比较；返回 [(测试名, 是否通过)]"""
    rng = np.random.default_rng(SEED)
    results = []

    ok = abs(max_drawdown([1.0, 1.2, 0.9, 1.1, 0.8, 1.3]) - 1 / 3) < 1e-12
    for _ in range(200):
        nav = list(np.cumprod(1 + rng.normal(0, 0.03, rng.integers(1, 40))))
        ok &= abs(max_drawdown(nav) - max_drawdown_bruteforce(nav)) < 1e-12
    results.append(("max_drawdown：示例 + 200 组随机对拍", bool(ok)))

    ok = sliding_max([1, 3, 2, 5, 4, 1, 6], 3) == [3, 5, 5, 5, 6]
    for _ in range(200):
        xs = list(rng.integers(0, 10, rng.integers(1, 30)))
        k = int(rng.integers(1, len(xs) + 1))
        ok &= sliding_max(xs, k) == sliding_max_bruteforce(xs, k)
    results.append(("sliding_max：示例 + 200 组随机对拍（含重复值）", bool(ok)))

    dates = np.repeat(pd.date_range("2024-01-31", periods=4, freq="ME"), 50)
    f = rng.normal(size=200)
    df = pd.DataFrame({"date": dates, "factor": f, "fwd_ret": 0.3 * f + rng.normal(size=200)})
    ic = rank_ic_by_date(df)
    ref = [stats.spearmanr(g["factor"], g["fwd_ret"])[0] for _, g in df.groupby("date")]
    results.append(("rank_ic_by_date：与 scipy.stats.spearmanr 一致", bool(np.allclose(ic.to_numpy(), ref))))

    r = random.Random(SEED)
    counts = np.zeros(20)
    for _ in range(20_000):
        for x in reservoir_sample(range(20), 5, r):
            counts[x] += 1
    freq = counts / 20_000  # 每个元素被抽中的概率应为 5/20
    results.append((f"reservoir_sample：每个元素被抽中的频率 ∈ [{freq.min():.3f}, {freq.max():.3f}]，理论 0.25",
                    bool(np.abs(freq - 0.25).max() < 0.015)))

    cache = LRUCache(2)
    cache.put("a", 1), cache.put("b", 2), cache.get("a"), cache.put("c", 3)
    results.append(("LRUCache：容量满时淘汰最久未使用的 b", cache.get("b") is None and cache.get("a") == 1))
    return results


def coding_questions():
    header("题 11 / 12 / 19（编程）：最大回撤、滑动窗口最大值、截面 Rank IC")
    print(f"    题 11 示例：max_drawdown([1.0, 1.2, 0.9, 1.1, 0.8, 1.3]) = "
          f"{max_drawdown([1.0, 1.2, 0.9, 1.1, 0.8, 1.3]):.4f}（从 1.2 跌到 0.8）")
    print(f"    题 12 示例：sliding_max([1, 3, 2, 5, 4, 1, 6], 3) = {sliding_max([1, 3, 2, 5, 4, 1, 6], 3)}")
    print("    复杂度：两者都是 O(n)。单调队列里每个下标最多进出一次（均摊 O(1)）")
    print("\n    🧪 对拍测试（与暴力解法 / 参考实现比较）：")
    tests = run_coding_tests()
    for name, ok in tests:
        print(f"      {'✅' if ok else '❌'} {name}")
    record(11, "编程", "O(n) 最大回撤", note="对拍通过" if tests[0][1] else "对拍失败")
    record(12, "编程", "O(n) 滑动窗口最大值", note="对拍通过" if tests[1][1] else "对拍失败")
    record(19, "编程", "pandas 分组计算截面 Rank IC", note="与 scipy 一致" if tests[2][1] else "不一致")
    print("\n    同样的测试也写在 tests/test_interview.py 中，用 pytest 运行")


# ==================== 第五部分：策略与研究题 ====================


def strategy_questions():
    header("题 13–15（策略）：回答框架")
    print("""
    题 13：你发现一个回测夏普为 3 的日频选股因子，接下来做什么？
      先怀疑，再验证：
        ① 前视偏差：信号时间戳、财报是否 PIT（Phase 7 08 课）、复权口径
        ② 可交易性与成本：涨跌停、停牌、T+1、小市值；双倍成本还剩多少
        ③ 样本外与分段：不同年份、不同市场环境、不同股票池
        ④ 与已知因子回归（市值、动量、反转、低波……）：是否只是旧因子换了包装
        ⑤ 披露试验次数，计算 DSR；⑥ 估算容量
      → 本平台的 06 课把这些写成了可自动执行的清单

    题 14：为什么动量效应可能存在？什么时候会失效？
      收益来源：信息反应不足、羊群效应、处置效应（行为偏差）
      失效：市场急剧反转（动量崩溃，熊市底部快速反弹）；拥挤导致踩踏
      A 股：中短期更常见的是反转效应——回答要结合数据，而不是背结论

    题 15：如何判断策略已经失效，而不只是正常回撤？
      ① 上线前写好的统计标准：回撤超过历史分布的某个分位、滚动收益置信区间
      ② 收益来源的前提是否仍成立：因子 IC、拥挤度、市场规则变化
      ③ 区分"环境不利"与"结构性失效"：前者在历史类似环境中出现过
      ④ 降低仓位持续观察，而不是全有或全无
    """)
    for no, t in [(13, "回测夏普 3 的因子怎么验证"), (14, "动量效应的来源与失效"), (15, "失效 vs 正常回撤")]:
        record(no, "策略", t, note="要点题")


def q20_capacity():
    header("题 20（策略）：用平方根法则估算日频策略的容量")
    sigma, adv, n_names, turnover, alpha_bp, Y = 0.02, 2e8, 50, 0.5, 15.0, 1.0
    print(f"""
    设定：持仓 {n_names} 只等权、每日单边换手 {turnover:.0%}、单票日均成交额 {adv / 1e8:.0f} 亿元、
          日波动 {sigma:.0%}、扣费前日均超额 {alpha_bp:.0f}bp、冲击系数 Y = {Y}
    冲击成本（每次交易，相对成交额）：impact = Y·σ·√(Q/V)
    每日冲击拖累 ≈ 2 × 换手 × impact（买卖两边）
    """)
    aums = np.array([1e7, 5e7, 1e8, 3e8, 1e9, 3e9])
    rows = []
    for aum in aums:
        q = aum * turnover / n_names  # 单票单边日成交额
        impact = Y * sigma * np.sqrt(q / adv)
        drag_bp = 2 * turnover * impact * 1e4
        rows.append([f"{aum / 1e8:g} 亿", f"{q / adv:.2%}", f"{impact * 1e4:.1f}", f"{drag_bp:.1f}", f"{alpha_bp - drag_bp:.1f}"])
    print(pd.DataFrame(rows, columns=["资金规模", "参与率", "单次冲击(bp)", "日拖累(bp)", "剩余超额(bp)"]).to_string(index=False))
    # 冲击吃掉一半超额时的规模：2·TO·Y·σ·√(AUM·TO/(N·V)) = α/2 → 解析求解
    half = (alpha_bp / 2 / 1e4 / (2 * turnover * Y * sigma)) ** 2 * n_names * adv / turnover
    print(f"\n    解析解：冲击吃掉一半超额时 AUM = (α/(4·TO·Y·σ))² · N·V/TO ≈ {half / 1e8:.2f} 亿元")
    print("    要点：冲击 ∝ √规模，所以容量对换手极其敏感（换手翻倍，容量约缩为 1/8）")
    from scipy.optimize import brentq

    def drag(aum):
        return 2 * turnover * Y * sigma * np.sqrt(aum * turnover / n_names / adv) * 1e4

    numeric = brentq(lambda a: drag(a) - alpha_bp / 2, 1e5, 1e11)
    print(f"    数值求根验证：{numeric / 1e8:.4f} 亿元")
    record(20, "策略", "平方根法则容量：半数超额被吃掉的规模(亿)", half / 1e8, numeric / 1e8, 1e-4)


# ==================== 第六部分：汇总 ====================


def summary():
    header("第六部分：20 题汇总")
    df = pd.DataFrame(RESULTS)

    def f(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{v:.4f}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)

    df["解析解"] = [f(v) for v in df["解析解"].astype(object)]
    df["模拟/验证"] = [f(v) for v in df["模拟/验证"].astype(object)]
    print(df.to_string(index=False))
    n_q = df["题号"].nunique()
    n_checked = (df["一致"] != "—").sum()
    n_ok = (df["一致"] == "✅").sum()
    print(f"\n    共 {n_q} 道题；其中 {n_checked} 个数值结论做了模拟验证，{n_ok} 个与解析解一致（容差内）")
    print("""
    📌 面试建议：
      1. 先说思路与假设，再算数；被追问时能回到推导，而不是背答案
      2. 概率题答完主动给出"如何用模拟验证"——这是工程背景的加分项
      3. 项目深挖：只写自己能完整解释的数字（样本区间、成本、基准、样本外、试验次数）
    """)


def main():
    careers_overview()
    q01_hh_ht()
    q02_dice_reroll()
    q03_broken_stick()
    q04_coupon()
    q05_birthday()
    q16_monty_hall()
    q17_max_uniform()
    q06_uncorrelated_not_independent()
    q07_regression_asymmetry()
    q08_sharpe_annualization()
    q09_multiple_testing()
    q10_sharpe_se()
    q18_p_hacking()
    coding_questions()
    strategy_questions()
    q20_capacity()
    summary()


if __name__ == "__main__":
    main()
