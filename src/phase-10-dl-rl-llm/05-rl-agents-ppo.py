"""
PPO 交易智能体：训练、样本外评估与 VecNormalize 陷阱
====================================================

学习目标：
    1. 理解 PPO 的核心：截断概率比、GAE 优势估计、价值损失与熵正则
    2. 用 Stable-Baselines3 在 04 课的环境上训练 PPO（小步数，约 1 分钟内完成）
    3. 在样本外测试期与买入持有、随机策略、均线规则比较，并报告多个随机种子
    4. 掌握 VecNormalize 的正确用法：评估时冻结训练期的统计量（training=False）

核心概念：
    - 概率比 ρ_t = π_θ(a|s) / π_θold(a|s)；L^CLIP = E[min(ρ_t·Â_t, clip(ρ_t, 1−ε, 1+ε)·Â_t)]
    - GAE：δ_t = r_t + γV(s_{t+1}) − V(s_t)，Â_t = Σ (γλ)^l δ_{t+l}
    - 只有一条历史：多个 episode 只是同一段历史的不同切片 → 极易过拟合
    - VecNormalize：在线估计观察的均值方差；评估时若继续更新，就用到了测试期的统计信息

前置知识：
    - 04-rl-trading-env.py（环境、时序、奖励）
    - 强化学习基础：策略梯度、Actor-Critic

环境要求：
    - pip install stable-baselines3 gymnasium torch（CPU，单次运行约 1 分钟）
    - 数据：沪深 300 ETF（510300）；支持 QTC_OFFLINE=1
"""

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai_tools import ETF_COST, SUBDIR, load_etf, perf_stats, set_seed
from trading_env import FEATURES, TradingEnv, build_env_data, run_policy
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

TRAIN_END, TEST_START = "2020-12-31", "2021-01-01"
WINDOW = 10
TIMESTEPS = 40_000
SEEDS = (0, 1, 2)


# ==================== 第一部分：PPO 回顾 ====================


def ppo_recap():
    print("=" * 60)
    print("第一部分：PPO 的核心思想")
    print("=" * 60)
    print("""
    每次更新策略时不要迈太大步 —— 就像调整交易系统时不会一夜之间推翻原策略。

        ρ_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)
        L^CLIP = E_t[ min( ρ_t·Â_t , clip(ρ_t, 1−ε, 1+ε)·Â_t ) ]
        总目标 = L^CLIP − c1·L^VF + c2·H[π_θ]          （价值损失、熵奖励）

    ┌──────────────┬──────────┬────────────────────────────────────────┐
    │ 超参数       │ 本课取值 │ 含义                                   │
    ├──────────────┼──────────┼────────────────────────────────────────┤
    │ n_steps      │ 512      │ 每次更新前采样的步数                   │
    │ batch_size   │ 128      │ 小批量大小                             │
    │ n_epochs     │ 5        │ 每批数据重复优化的轮数                 │
    │ gamma / λ    │ 0.99/0.95│ 折扣、GAE 参数                         │
    │ clip_range ε │ 0.2      │ 概率比截断范围                         │
    │ ent_coef     │ 0.01     │ 熵奖励，鼓励探索                       │
    └──────────────┴──────────┴────────────────────────────────────────┘
    """)


# ==================== 第二部分：训练 ====================


def make_train_env(train):
    X, O, D = train
    env = TradingEnv(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=250, random_start=True, reward="pnl")
    return VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=True, clip_obs=5.0)


def train_agents(train):
    print("\n" + "=" * 60)
    print("第二部分：训练 PPO（训练段，随机起点的 250 天 episode）")
    print("=" * 60)
    agents = {}
    for s in SEEDS:
        set_seed(s)
        venv = make_train_env(train)
        venv.seed(s)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=128, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01, learning_rate=3e-4, policy_kwargs={"net_arch": [64, 64]},
                    seed=s, device="cpu", verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=TIMESTEPS)
        agents[s] = (model, venv)
        print(f"    seed={s}：{TIMESTEPS:,} 步，用时 {time.time() - t0:.1f}s")
    print(f"""
    {TIMESTEPS:,} 步 ≈ {TIMESTEPS // 250} 个 episode，但训练段只有约 {len(train[0])} 天 ——
    智能体反复"重玩"同一段历史的不同切片。游戏 RL 动辄上亿步，这里的步数远不足以说明问题，
    但更多步数在单一历史路径上往往只会带来更多过拟合。
    """)
    return agents


# ==================== 第三部分：评估（冻结 VecNormalize） ====================


def evaluate(model, train_venv, data, frozen: bool = True) -> pd.DataFrame:
    """
    frozen=True：复制训练期的 obs_rms，training=False、norm_reward=False（正确）
    frozen=False：在测试数据上新建并持续更新 VecNormalize（错误：用到了测试期统计量）
    """
    X, O, D = data
    env = TradingEnv(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=None, random_start=False)
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=not frozen, clip_obs=5.0,
                        training=not frozen)
    if frozen:
        venv.obs_rms = copy.deepcopy(train_venv.obs_rms)
    obs = venv.reset()
    rows = []
    for _ in range(env.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(action)
        rows.append({k: info[0][k] for k in ("date", "pos", "net", "cost")})
        if done[0]:
            break
    out = pd.DataFrame(rows).set_index("date")
    out.attrs["obs_rms_mean"] = venv.obs_rms.mean.copy()
    return out


def baselines(data, scaler) -> dict:
    X, O, D = data
    ma = (WINDOW - 1) * len(FEATURES) + FEATURES.index("ma_gap")
    ma_zero = float(-scaler.mean_["ma_gap"] / scaler.std_["ma_gap"])
    pols = {"买入持有": lambda o, e: 2, "空仓": lambda o, e: 0,
            "均线规则": lambda o, e: 2 if o[ma] > ma_zero else 0}
    out = {}
    for name, pol in pols.items():
        env = TradingEnv(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=None, random_start=False)
        out[name] = run_policy(env, pol)
    rand = []
    for s in range(20):
        env = TradingEnv(X, O, D, window=WINDOW, cost=ETF_COST, episode_len=None, random_start=False)
        rand.append(run_policy(env, lambda o, e: int(e.np_random.integers(0, 3)), seed=s))
    out["随机（20 次平均）"] = rand
    return out


def compare(agents, train, test, scaler):
    print("\n" + "=" * 60)
    print("第三部分：样本内 vs 样本外，与简单基线比较")
    print("=" * 60)
    rows, navs, results = [], {}, {}
    for seg_name, data in (("训练段", train), ("测试段", test)):
        base = baselines(data, scaler)
        for name, res in base.items():
            if isinstance(res, list):
                st = pd.DataFrame([perf_stats(r["net"]) for r in res]).mean()
                rows.append([seg_name, name, st["年化收益"], st["夏普"], st["最大回撤"], np.nan, np.nan])
            else:
                st = perf_stats(res["net"])
                rows.append([seg_name, name, st["年化收益"], st["夏普"], st["最大回撤"], res["pos"].mean(),
                             (res["pos"].diff().abs() > 0).sum()])
                if seg_name == "测试段":
                    navs[name] = (1 + res["net"]).cumprod()
        for s, (model, venv) in agents.items():
            res = evaluate(model, venv, data, frozen=True)
            results[(seg_name, s)] = res
            st = perf_stats(res["net"])
            rows.append([seg_name, f"PPO seed={s}", st["年化收益"], st["夏普"], st["最大回撤"], res["pos"].mean(),
                         (res["pos"].diff().abs() > 0).sum()])
            if seg_name == "测试段":
                navs[f"PPO seed={s}"] = (1 + res["net"]).cumprod()
    table = pd.DataFrame(rows, columns=["区间", "策略", "年化收益", "夏普", "最大回撤", "平均仓位", "调仓次数"])
    print(f"\n    单边成本 {ETF_COST:.2%}；评估为确定性策略（deterministic=True），全程固定起点\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    te = table[table["区间"] == "测试段"].set_index("策略")
    tr = table[table["区间"] == "训练段"].set_index("策略")
    ppo = [f"PPO seed={s}" for s in SEEDS]
    beat_bh = sum(te.loc[p, "夏普"] > te.loc["买入持有", "夏普"] for p in ppo)
    beat_ma = sum(te.loc[p, "夏普"] > te.loc["均线规则", "夏普"] for p in ppo)
    gap = (tr.loc[ppo, "夏普"] - te.loc[ppo, "夏普"]).mean()
    degen = [p for p in ppo if not 0.1 < te.loc[p, "平均仓位"] < 0.9]
    degen_txt = (f"{', '.join(degen)} 的平均仓位接近 0 或 1，策略基本退化成了一直空仓/满仓" if degen
                 else "各种子平均仓位都不在 0 或 1 附近：策略确实在择时，但择时没有带来样本外优势")
    print(f"""
    本次结果（测试段夏普）：{len(SEEDS)} 个种子中，有 {beat_bh} 个超过买入持有、{beat_ma} 个超过均线规则；
    PPO 的夏普从训练段到测试段平均变化 {-gap:+.2f}（负数 = 样本外变差）。
    种子之间的差异（见上表）提醒我们：单个种子的好结果不能当作证据。
    {degen_txt}。
    """)
    return navs, results


# ==================== 第四部分：VecNormalize 陷阱 ====================


def vecnormalize_trap(agents, test):
    print("\n" + "=" * 60)
    print("第四部分：VecNormalize 陷阱 —— 评估时必须冻结统计量")
    print("=" * 60)
    model, venv = agents[SEEDS[0]]
    right = evaluate(model, venv, test, frozen=True)
    wrong = evaluate(model, venv, test, frozen=False)
    m_train = venv.obs_rms.mean[: len(FEATURES)]
    m_wrong = wrong.attrs["obs_rms_mean"][: len(FEATURES)]
    print("\n    观察均值估计（第一天窗口的 8 个特征）：")
    print(pd.DataFrame({"训练期统计（冻结）": m_train, "在测试期上继续更新": m_wrong}, index=FEATURES).T.to_string(
        float_format=lambda v: f"{v:+.3f}"))
    same = (right["pos"].to_numpy() == wrong["pos"].to_numpy()).mean()
    r1, r2 = perf_stats(right["net"]), perf_stats(wrong["net"])
    print(f"""
    同一个模型（seed={SEEDS[0]}）、同一段测试数据：
      冻结统计量    ：年化 {r1['年化收益']:+.2%}，夏普 {r1['夏普']:.2f}
      继续更新统计量：年化 {r2['年化收益']:+.2%}，夏普 {r2['夏普']:.2f}
      两种评估下动作一致的天数占比：{same:.1%}
    本例中错误做法的结果{'更好' if r2['夏普'] > r1['夏普'] else '更差'} —— 无论好坏，它都不是实盘中能复现的数字。

    错误做法的问题：测试期的观察会被"用测试期自己的均值方差"归一化 —— 这是前视；
    而且评估开始时统计量从零开始，前几十天的输入分布与训练时完全不同。
    正确做法：
        venv.save("vecnormalize.pkl")                         # 训练结束后保存
        eval_env = VecNormalize.load("vecnormalize.pkl", DummyVecEnv([make_test_env]))
        eval_env.training = False                             # 不再更新均值方差
        eval_env.norm_reward = False                          # 评估看真实奖励
    """)


# ==================== 第五部分：RL 在金融中的定位 ====================


def rl_positioning(navs):
    print("\n" + "=" * 60)
    print("第五部分：为什么 RL 用于择时/选股特别难，RL 更适合做什么")
    print("=" * 60)
    print("""
    ┌──────────────────┬────────────────────────────────────────────────────────────┐
    │ 问题             │ 说明                                                       │
    ├──────────────────┼────────────────────────────────────────────────────────────┤
    │ 奖励噪声         │ 单步奖励几乎全是噪声，信用分配极难                         │
    │ 只有一条历史     │ 无法像游戏那样无限重开                                     │
    │ 动作不影响环境   │ 小资金时退化为上下文老虎机，比监督学习更低效               │
    │ 超参数敏感       │ 不同种子结果差异大，必须报告多个种子                       │
    │ 评估             │ 必须与买入持有、监督学习 + 规则策略对比，扣费后比较         │
    └──────────────────┴────────────────────────────────────────────────────────────┘

    RL 更适合"动作会改变后续状态和成本"的问题：
      · 最优执行（大单拆分，Phase 11 的 Almgren-Chriss 是解析基准；Qlib 提供 qlib.rl 执行框架）
      · 做市（库存影响报价风险，Phase 11 的 Avellaneda-Stoikov）
      · 期权对冲（有成本时学习对冲频率与比例，Phase 12）
    FinRL 等项目适合学习，但其示例回测结果不应直接视为可实盘的证据。
    """)
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, nav in navs.items():
        style = "-" if name.startswith("PPO") else "--"
        ax.plot(nav.index, nav.to_numpy(), style, label=name, lw=1.2)
    ax.set_title("测试段净值：PPO（多个种子）vs 简单基线（扣费）")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    path = save_figure(fig, "05_ppo_vs_baselines.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    torch.set_num_threads(4)
    df = load_etf("510300", "20130101", "20241231")
    if df.attrs.get("synthetic"):
        print("⚠️ 当前为合成数据，结果只用于跑通流程")
    train, test, scaler = build_env_data(df, TRAIN_END, TEST_START)
    ppo_recap()
    agents = train_agents(train)
    navs, _ = compare(agents, train, test, scaler)
    vecnormalize_trap(agents, test)
    rl_positioning(navs)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把奖励换成差分夏普（reward="dsr"），比较测试段夏普与调仓次数
    2. 把 TIMESTEPS 增加到 200,000，观察训练段与测试段表现的差距是否扩大
    3. 用 Phase 9 的监督学习（如逻辑回归预测涨跌 + "预测上涨则持有"规则）作为基线，与 PPO 比较
    4. 用 5 个种子 × 3 个 ent_coef 做网格，统计"在测试段打败买入持有"的比例，并思考这是多少次试验
    """)


if __name__ == "__main__":
    main()
