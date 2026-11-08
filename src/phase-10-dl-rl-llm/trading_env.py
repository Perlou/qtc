"""
Phase 10 辅助模块：单资产交易环境（Gymnasium）
==============================================

04 课讲解设计，05 课用 PPO 训练。时序约定（避免前视的关键）：

    t 日收盘后  观察 s_t（特征只用到 t 日收盘及之前的数据）→ 智能体给出目标仓位 a_t
    t+1 日开盘  以开盘价调仓到 a_t，支付交易成本 cost × |a_t − 当前仓位|
    t+2 日开盘  持有期结束，奖励 r_t = a_t × (open_{t+2} / open_{t+1} − 1) − 成本
               （r_t 在 t+2 开盘才"揭晓"，绝不进入 s_t）

奖励模式：
    "pnl"  扣费后的单步收益（放大 100 倍，便于神经网络学习，不改变最优策略）
    "dsr"  差分夏普比率（Moody & Saffell 2001），在线优化夏普
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

FEATURES = ["r1", "r5", "r20", "vol20", "ma_gap", "intraday", "gap", "vol_z"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """由 t 日收盘及之前的数据构造特征（第 t 行在 t 日收盘后才可得）"""
    c, o = df["adj_close"], df["adj_open"]
    r1 = c.pct_change()
    feat = pd.DataFrame({
        "r1": r1,                                           # 当日收益（收盘到收盘）
        "r5": c / c.shift(5) - 1,                           # 5 日动量
        "r20": c / c.shift(20) - 1,                         # 20 日动量
        "vol20": r1.rolling(20).std(),                      # 20 日波动率
        "ma_gap": c / c.rolling(20).mean() - 1,             # 偏离 20 日均线
        "intraday": c / o - 1,                              # 日内收益（开盘到收盘）
        "gap": o / c.shift(1) - 1,                          # 隔夜跳空
        "vol_z": np.log(df["volume"] / df["volume"].rolling(20).mean()),  # 相对成交量
    }, index=df.index)
    return feat


class FeatureScaler:
    """只用训练期拟合的标准化器（全样本标准化 = 把未来的均值方差泄漏给过去）"""

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.mean_, self.std_ = X.mean(), X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return ((X - self.mean_) / self.std_).clip(-5, 5)


class TradingEnv(gym.Env):
    """单资产多头环境：动作 = 目标仓位 {0, 0.5, 1}"""

    metadata = {"render_modes": []}
    POSITIONS = (0.0, 0.5, 1.0)

    def __init__(self, features: np.ndarray, open_px: np.ndarray, dates=None, window: int = 10,
                 cost: float = 0.0005, episode_len: int | None = 250, random_start: bool = True,
                 reward: str = "pnl", eta: float = 0.02, reward_scale: float = 100.0):
        super().__init__()
        assert len(features) == len(open_px), "特征与价格长度必须一致"
        assert reward in ("pnl", "dsr")
        self.X = np.clip(np.asarray(features, dtype=np.float32), -5.0, 5.0)
        self.open = np.asarray(open_px, dtype=np.float64)
        self.dates = pd.DatetimeIndex(dates) if dates is not None else pd.RangeIndex(len(self.X))
        self.window, self.cost, self.reward_mode = window, cost, reward
        self.eta, self.reward_scale = eta, reward_scale
        self.random_start = random_start
        self.last_t = len(self.X) - 3  # 需要 open[t+2]
        full = self.last_t - (window - 1) + 1
        self.episode_len = full if episode_len is None else min(episode_len, full)
        n_feat = self.X.shape[1]
        self.action_space = spaces.Discrete(len(self.POSITIONS))
        # 特征已用训练期统计量标准化并截断到 ±5；最后一维是当前仓位（0~1）
        self.observation_space = spaces.Box(-5.0, 5.0, (window * n_feat + 1,), np.float32)

    def _obs(self) -> np.ndarray:
        win = self.X[self.t - self.window + 1:self.t + 1]  # 只到 t（含），不含 t+1
        return np.append(win.ravel(), self.pos).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        first = self.window - 1
        if self.random_start:
            latest = self.last_t - self.episode_len + 1
            self.t = int(self.np_random.integers(first, latest + 1))
        else:
            self.t = first
        self.t_end = self.t + self.episode_len - 1
        self.pos = 0.0
        self.A, self.B = 0.0, 1e-4  # 差分夏普的一阶、二阶矩（指数移动估计）
        return self._obs(), {}

    def _dsr(self, R: float) -> float:
        dA, dB = R - self.A, R**2 - self.B
        var = self.B - self.A**2
        D = (self.B * dA - 0.5 * self.A * dB) / var**1.5 if var > 1e-10 else 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB
        return float(np.clip(D, -5, 5))

    def step(self, action):
        target = self.POSITIONS[int(action)]
        trade_cost = self.cost * abs(target - self.pos)
        hold_ret = self.open[self.t + 2] / self.open[self.t + 1] - 1  # t+2 开盘才揭晓，不进入观察
        net = target * hold_ret - trade_cost
        reward = net * self.reward_scale if self.reward_mode == "pnl" else self._dsr(net)
        info = {"date": self.dates[self.t], "pos": target, "hold_ret": hold_ret, "cost": trade_cost, "net": net}
        self.pos = target
        terminated = self.t >= self.t_end
        if not terminated:
            self.t += 1
        return self._obs(), float(reward), terminated, False, info


def build_env_data(df: pd.DataFrame, train_end: str, test_start: str):
    """特征 → 训练期拟合标准化 → 切分为训练段与测试段（各自含 open 价格与日期）"""
    feat = make_features(df).dropna()
    px = df["adj_open"].reindex(feat.index)
    train = feat.index <= pd.Timestamp(train_end)
    test = feat.index >= pd.Timestamp(test_start)
    scaler = FeatureScaler().fit(feat[train])
    Z = scaler.transform(feat)
    pack = lambda m: (Z[m].to_numpy(), px[m].to_numpy(), feat.index[m])  # noqa: E731
    return pack(train), pack(test), scaler


def run_policy(env: TradingEnv, policy, seed: int = 0) -> pd.DataFrame:
    """在环境中跑一个完整 episode；policy(obs, env) -> action"""
    obs, _ = env.reset(seed=seed)
    rows, done = [], False
    while not done:
        a = policy(obs, env)
        obs, r, terminated, truncated, info = env.step(a)
        rows.append({**info, "reward": r})
        done = terminated or truncated
    return pd.DataFrame(rows).set_index("date")
