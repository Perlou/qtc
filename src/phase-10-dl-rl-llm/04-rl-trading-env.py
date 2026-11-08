"""
构建 Gym 交易环境：时序、成本、奖励与前视测试
============================================

学习目标：
    1. 把交易表述为 MDP：状态、动作、转移、奖励、折扣各对应什么
    2. 实现"t 日收盘观察、t+1 开盘执行、t+2 开盘结算"的 Gymnasium 环境，计入交易成本
    3. 用 gymnasium.utils.env_checker.check_env 检查 API；写"篡改未来数据"的前视单元测试
    4. 比较两种奖励：扣费收益（PnL）与差分夏普比率（DSR）；看一个有泄漏的环境如何制造神话

核心概念：
    - 环境的真实性比算法的先进性重要：模拟器里能"提前一秒看到窗外"，飞行员上了真飞机就坠毁
    - Gymnasium API：reset → (obs, info)；step → (obs, reward, terminated, truncated, info)
    - 差分夏普 D_t = (B_{t-1}ΔA_t − ½A_{t-1}ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}
    - 前视单元测试：修改 t+1 之后的数据，t 时刻的观察必须完全不变

前置知识：
    - Phase 4（回测时序与成本）；01 课（无泄漏窗口）
    - 强化学习基础：MDP、策略、回报

环境要求：
    - pip install gymnasium pandas numpy matplotlib
    - 数据：沪深 300 ETF（510300）；支持 QTC_OFFLINE=1
"""

import inspect
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gymnasium.utils.env_checker import check_env

from ai_tools import ETF_COST, SUBDIR, load_etf, perf_stats
from trading_env import FEATURES, TradingEnv, build_env_data, run_policy
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRAIN_END, TEST_START = "2020-12-31", "2021-01-01"
WINDOW = 10


class LeakyEnv(TradingEnv):
    """❌ 错误示范：观察窗口多取了一天（切片写成 t+2），t+1 日的特征混进了 t 时刻的观察"""

    def _obs(self):
        win = self.X[self.t - self.window + 2:self.t + 2]
        return np.append(win.ravel(), self.pos).astype(np.float32)


# ==================== 第一部分：把交易表述为 MDP ====================


def mdp_overview():
    print("=" * 60)
    print("第一部分：把交易表述为 MDP")
    print("=" * 60)
    print("""
    ┌────────────┬──────────────────────────────────┬────────────────────────────────────┐
    │ MDP 要素   │ 本课环境中的对应                 │ 设计要点                           │
    ├────────────┼──────────────────────────────────┼────────────────────────────────────┤
    │ 状态 s_t   │ 过去 10 天 × 8 个特征 + 当前仓位 │ 只含 t 日收盘时已经可得的信息      │
    │ 动作 a_t   │ 目标仓位 {0, 0.5, 1}（离散）     │ A 股 ETF 只做多；不考虑整手约束    │
    │ 转移       │ 按历史数据推进一天               │ 假设小资金不影响价格 = 历史回放    │
    │ 奖励 r_t   │ 扣费收益 或 差分夏普             │ 奖励决定智能体学到什么             │
    │ 折扣 γ     │ 05 课取 0.99                     │ 交易中常取接近 1                   │
    └────────────┴──────────────────────────────────┴────────────────────────────────────┘

    时序（避免前视的关键）：

      t 日                        t+1 日                       t+2 日
      ├─ 开盘 ─ 盘中 ─ 收盘 ─┤    ├─ 开盘 ─ 盘中 ─ 收盘 ─┤    ├─ 开盘 ...
                        │          │                            │
                 观察 s_t、决策 a_t │                            │
                               以 t+1 开盘价调仓 ─── 持有 ───────┘
                               奖励 r_t = a_t × (open_{t+2}/open_{t+1} − 1) − c × |a_t − a_{t−1}|

    常见前视来源：成交价用了信号同一时刻的价格 · 全样本标准化 · 窗口切片多取一天 ·
                  未来收益放进 obs · 前复权价格 · 忽略涨跌停 / T+1 等成交约束
    """)


# ==================== 第二部分：环境实现与 API 检查 ====================


def build_and_check(df):
    print("\n" + "=" * 60)
    print("第二部分：环境实现与 check_env")
    print("=" * 60)
    (Xtr, Otr, Dtr), (Xte, Ote, Dte), scaler = build_env_data(df, TRAIN_END, TEST_START)
    tag = "⚠️ 合成数据" if df.attrs.get("synthetic") else "真实数据"
    print(f"\n    数据：510300（{tag}）；训练段 {Dtr[0].date()} ~ {Dtr[-1].date()}（{len(Dtr)} 天），"
          f"测试段 {Dte[0].date()} ~ {Dte[-1].date()}（{len(Dte)} 天）")
    print(f"    特征 {FEATURES}，标准化参数只用训练段拟合，并截断到 ±5")
    print("\n    核心代码（trading_env.TradingEnv）：\n")
    for fn in (TradingEnv._obs, TradingEnv.step):
        src = inspect.getsource(fn)
        print("\n".join("      " + line for line in src.rstrip().splitlines()))
        print()
    env = TradingEnv(Xtr, Otr, Dtr, window=WINDOW, cost=ETF_COST, episode_len=250)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    msgs = sorted({str(w.message).split("WARN: ")[-1].strip("\x1b[0m") for w in caught})
    print(f"    check_env：通过 ✅（警告 {len(msgs)} 条）" + ("".join(f"\n      - {m[:90]}" for m in msgs)))
    print(f"    observation_space = {env.observation_space}")
    print(f"    action_space      = {env.action_space}（0 → 空仓，1 → 半仓，2 → 满仓）")
    print("""
    check_env 会检查：reset/step 的返回格式、obs 是否落在 observation_space 内、dtype、
    相同 seed 下 reset 是否可复现等。它只检查"接口"，不检查"有没有前视" —— 后者要自己写测试。
    """)
    return (Xtr, Otr, Dtr), (Xte, Ote, Dte), scaler


# ==================== 第三部分：前视单元测试 ====================


def lookahead_unit_test(train):
    print("\n" + "=" * 60)
    print("第三部分：前视单元测试 —— 篡改未来，观察不应变化")
    print("=" * 60)
    X, O, D = train
    rng = np.random.default_rng(0)

    def check(env_cls) -> tuple[int, int]:
        bad, ts = 0, rng.integers(WINDOW, len(X) - 3, 200)
        for t in ts:
            X2 = X.copy()
            X2[t + 1:] = rng.normal(size=X2[t + 1:].shape)  # 把 t+1 之后的特征全部换成噪声
            e1, e2 = env_cls(X, O, D, window=WINDOW), env_cls(X2, O, D, window=WINDOW)
            for e in (e1, e2):
                e.reset(seed=0)
                e.t, e.pos = int(t), 0.0
            bad += int(not np.array_equal(e1._obs(), e2._obs()))
        return bad, len(ts)

    for name, cls in (("TradingEnv（正确）", TradingEnv), ("LeakyEnv（窗口多取一天）", LeakyEnv)):
        bad, n = check(cls)
        print(f"    {name:<24}：{n} 个随机时点中，观察被未来数据改变的有 {bad} 个 → {'通过 ✅' if bad == 0 else '失败 ❌'}")
    print("""
    这类测试应该放进 CI：环境代码任何改动都要重新跑。它同样适用于因子函数（08、10 课）。
    """)


# ==================== 第四部分：奖励设计与基准策略 ====================


def reward_design(test, scaler):
    print("\n" + "=" * 60)
    print("第四部分：奖励设计 —— 扣费收益 vs 差分夏普")
    print("=" * 60)
    X, O, D = test
    ma_idx = FEATURES.index("ma_gap")
    ma_zero = float((0 - scaler.mean_["ma_gap"]) / scaler.std_["ma_gap"])  # 原始 ma_gap = 0 对应的标准化值
    policies = {
        "始终满仓（买入持有）": lambda obs, env: 2,
        "始终半仓": lambda obs, env: 1,
        "始终空仓": lambda obs, env: 0,
        "随机": lambda obs, env: int(env.np_random.integers(0, 3)),  # 用环境的随机数发生器 → 两种奖励下路径相同
        "均线规则（价在 20 日线上方满仓）": lambda obs, env: 2 if obs[(WINDOW - 1) * len(FEATURES) + ma_idx] > ma_zero else 0,
    }
    rows, navs = [], {}
    for name, pol in policies.items():
        env = TradingEnv(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=None, random_start=False, reward="pnl")
        r = run_policy(env, pol, seed=0)
        ps = perf_stats(r["net"])
        rows.append([name, ps["年化收益"], ps["夏普"], ps["最大回撤"], r["cost"].sum(), (r["pos"].diff().abs() > 0).sum(),
                     r["reward"].sum()])
        navs[name] = (1 + r["net"]).cumprod()
    table = pd.DataFrame(rows, columns=["策略", "年化收益", "夏普", "最大回撤", "累计成本", "调仓次数", "Σ PnL 奖励"])
    print(f"\n    测试段全程（固定起点、覆盖整个测试期），单边成本 {ETF_COST:.2%}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rand_cost = table.set_index("策略").loc["随机", "累计成本"]
    print(f"""
    · Σ PnL 奖励 = 扣费收益之和 × 100（放大只为便于网络学习，不改变最优策略）
    · "随机"策略累计支付成本 {rand_cost:.2%}：频繁换仓本身就是确定的亏损来源，环境必须计入成本
    · 这些基准就是 05 课 PPO 必须打败的对象（尤其是"买入持有"和简单规则）
    """)

    # 差分夏普：给定当前的均值/方差估计，看单步奖励如何随收益 R 变化
    A, sd = 0.0005, 0.01  # 当前估计：日均收益 0.05%，日波动 1%
    B = A**2 + sd**2
    rows = []
    for R in (-0.03, -0.01, 0.0, 0.01, 0.03):
        env = TradingEnv(X[:5], O[:5], D[:5], window=1, reward="dsr")
        env.reset(seed=0)
        env.A, env.B = A, B
        rows.append([f"{R:+.0%}", R * 100, env._dsr(R)])
    tab = pd.DataFrame(rows, columns=["本步收益 R", "PnL 奖励（×100）", "DSR 奖励 D_t"])
    print(f"    差分夏普的单步奖励（当前估计：A = 日均 {A:.2%}，波动 {sd:.0%}）：\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    up, dn = tab.iloc[-1, 2], tab.iloc[0, 2]
    print(f"""
    D_t = [B·(R − A) − ½·A·(R² − B)] / (B − A²)^1.5：线性项奖励收益，−½·A·R² 项惩罚波动。
    本例中 +3% 得到 {up:.3f}，−3% 得到 {dn:.3f}：大幅波动（无论涨跌）都被额外扣分，
    这就是"在线优化夏普"的含义。注意：
      · Σ D_t 近似是"滑动夏普估计的变化 ÷ η"，会随路径终点而变，不能当作全期夏普
      · A、B 在起步阶段估计不稳，常需要用一段历史预热
    """)
    print("""
    ┌──────────────┬───────────────────────────────────┬──────────────────────────────────┐
    │ 奖励         │ 公式                              │ 特点                             │
    ├──────────────┼───────────────────────────────────┼──────────────────────────────────┤
    │ 扣费收益     │ w_t·R_{t+1} − c·|Δw|               │ 最直接；可能学会承担极端风险     │
    │ 风险惩罚     │ 再减 κ·(w_t·R_{t+1})²              │ 近似均值-方差效用                │
    │ 差分夏普     │ D_t（Moody & Saffell 2001）        │ 在线优化夏普；早期估计不稳定     │
    └──────────────┴───────────────────────────────────┴──────────────────────────────────┘
    """)
    return navs


# ==================== 第五部分：有泄漏的环境如何制造神话 ====================


def leaky_env_demo(test):
    print("\n" + "=" * 60)
    print("第五部分：有泄漏的环境如何制造神话")
    print("=" * 60)
    X, O, D = test
    k = FEATURES.index("intraday")
    last = (WINDOW - 1) * len(FEATURES) + k  # 观察中"最后一天"的日内收益特征
    peek = lambda obs, env: 2 if obs[last] > 0 else 0  # noqa: E731  同一条规则：最后一天日内上涨就满仓
    out = {}
    for name, cls in (("正确环境（最后一天 = t）", TradingEnv), ("泄漏环境（最后一天 = t+1）", LeakyEnv)):
        env = cls(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=None, random_start=False)
        r = run_policy(env, peek, seed=0)
        out[name] = perf_stats(r["net"])
    print("\n    同一条简单规则（窗口最后一天日内收益 > 0 则满仓）在两个环境中的测试段表现：\n")
    print(pd.DataFrame(out).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("""
    泄漏环境里"最后一天"其实是 t+1：t+1 的日内收益正是持有期 open_{t+1} → open_{t+2} 的主要部分，
    规则等于提前知道了答案。任何 RL 算法在这样的环境里都会学到这条"捷径"，训练曲线漂亮得惊人，
    而实盘中这个信息根本不存在。第三部分的前视单元测试能在训练之前就抓住这个 bug。
    """)


def plot_navs(navs):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, nav in navs.items():
        ax.plot(nav.index, nav.to_numpy(), label=name, lw=1.2)
    ax.set_title("测试段净值：基准策略（扣费）")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    path = save_figure(fig, "04_env_baseline_policies.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    df = load_etf("510300", "20130101", "20241231")
    mdp_overview()
    train, test, scaler = build_and_check(df)
    lookahead_unit_test(train)
    navs = reward_design(test, scaler)
    leaky_env_demo(test)
    plot_navs(navs)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 给环境加上 A 股约束：t+1 开盘涨停时不能买入、跌停时不能卖出（提示：需要不复权价格与涨跌停价）
    2. 实现"风险惩罚"奖励 r = net − κ·net²，比较 κ = 0、10、50 时各基准策略的奖励排序
    3. 把动作改为连续仓位（spaces.Box(0, 1, (1,))），check_env 需要做哪些修改？
    4. 为 make_features 写一个前视单元测试：修改 t+1 之后的价格，t 日的特征必须不变
    """)


if __name__ == "__main__":
    main()
