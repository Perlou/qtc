"""
Phase 12 辅助模块：期权定价工具箱
==================================

本模块不是课程文件，而是 05–08 课（以及 02–04 课的交叉校验）共用的工具。
02、03、04 课会在课程内部从零实现同样的函数并逐行讲解，这里是整理后的"正式版"。

函数一览（所有函数都支持 numpy 向量化，除非特别说明）：
    bs_d1_d2(S, K, T, r, sigma, q)          d1、d2
    bs_price(S, K, T, r, sigma, kind, q)    Black-Scholes-Merton 欧式期权价格
    bs_greeks(S, K, T, r, sigma, kind, q)   解析希腊字母（单位见下）
    crr_price(...)                          CRR 二叉树（欧式 / 美式，标量）
    mc_price(...)                           蒙特卡洛定价（标量，返回价格与标准误）
    price_bounds(...)                       无套利价格上下界
    implied_vol(price, ...)                 隐含波动率：牛顿法 + 二分法兜底（标量）
    implied_vol_chain(prices, ...)          对一组合约逐个求 IV
    implied_forward(C, P, K, r, T)          由认购-认沽平价反推远期价格
    svi_w / svi_derivatives / fit_svi       SVI 总方差参数化与拟合
    svi_butterfly_g(k, params)              Durrleman 蝶式无套利检验函数 g(k)

单位约定（务必统一）：
    T      年（自然日 / 365）
    sigma  小数（20% 写作 0.20）
    vega   σ 变动 1.00 时的价格变化 → 每 1 个波动率点 = vega / 100
    theta  每年 → 每个自然日 = theta / 365
    rho    r 变动 1.00 时的价格变化 → 每 1% = rho / 100
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

SIGMA_MIN, SIGMA_MAX = 1e-4, 5.0


# ==================== Black-Scholes-Merton ====================


def _is_call(kind) -> np.ndarray:
    return np.asarray(kind) == "call"


def bs_d1_d2(S, K, T, r, sigma, q=0.0):
    """返回 (d1, d2)；对 T→0、sigma→0 做了数值保护"""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol_sqrt_t = np.maximum(sigma * np.sqrt(np.maximum(T, 0.0)), 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton 欧式期权价格；q 为连续股息率"""
    d1, d2 = bs_d1_d2(S, K, T, r, sigma, q)
    disc_s = np.asarray(S, dtype=float) * np.exp(-q * np.asarray(T, dtype=float))
    disc_k = np.asarray(K, dtype=float) * np.exp(-r * np.asarray(T, dtype=float))
    call = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    put = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    return np.where(_is_call(kind), call, put)[()]


def bs_greeks(S, K, T, r, sigma, kind="call", q=0.0) -> dict:
    """解析希腊字母：delta, gamma, vega(每1.00), theta(每年), rho(每1.00)"""
    S, K, T = (np.asarray(x, dtype=float) for x in (S, K, T))
    d1, d2 = bs_d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(np.maximum(T, 1e-12))
    pdf = norm.pdf(d1)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    is_call = _is_call(kind)

    gamma = eq * pdf / (S * sigma * sqrt_t)
    vega = S * eq * pdf * sqrt_t
    common_theta = -S * eq * pdf * sigma / (2 * sqrt_t)
    theta_c = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
    theta_p = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
    return {
        "delta": np.where(is_call, eq * norm.cdf(d1), eq * (norm.cdf(d1) - 1))[()],
        "gamma": gamma[()],
        "vega": vega[()],
        "theta": np.where(is_call, theta_c, theta_p)[()],
        "rho": np.where(is_call, K * T * er * norm.cdf(d2), -K * T * er * norm.cdf(-d2))[()],
    }


# ==================== 二叉树与蒙特卡洛 ====================


def crr_price(S, K, T, r, sigma, N=500, kind="call", american=False, q=0.0) -> float:
    """Cox-Ross-Rubinstein 二叉树；american=True 时每个节点比较提前行权价值"""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(N + 1)
    ST = S * u ** (N - j) * d**j  # 终端价格，从高到低
    V = np.maximum(sign * (ST - K), 0.0)
    for step in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            jj = np.arange(step + 1)
            S_nodes = S * u ** (step - jj) * d**jj
            V = np.maximum(V, sign * (S_nodes - K))
    return float(V[0])


def mc_price(S, K, T, r, sigma, kind="call", n_paths=200_000, seed=0, q=0.0, antithetic=True):
    """风险中性蒙特卡洛：返回 (价格, 标准误)"""
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(half)
    if antithetic:
        Z = np.concatenate([Z, -Z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    disc_payoff = np.exp(-r * T) * payoff
    if antithetic:  # 对偶变量：成对平均后再算标准误
        pair = 0.5 * (disc_payoff[:half] + disc_payoff[half:])
        return float(pair.mean()), float(pair.std(ddof=1) / np.sqrt(half))
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))


# ==================== 隐含波动率 ====================


def price_bounds(S, K, T, r, kind="call", q=0.0):
    """欧式期权无套利价格区间 (下界, 上界)"""
    fs, pk = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return max(fs - pk, 0.0), fs
    return max(pk - fs, 0.0), pk


def initial_guess(price, S, K, T, r, q=0.0) -> float:
    """
    牛顿法初值：取 Brenner-Subrahmanyam 近似与 Manaster-Koehler 拐点中的较大者
        Brenner：σ ≈ √(2π/T)·price/S（平值附近准确）
        拐点：σ = √(2|ln(F/K)|/T)，BS 价格在此处由凸变凹，从这里出发的牛顿迭代单调收敛
    """
    brenner = np.sqrt(2 * np.pi / T) * price / S
    inflection = np.sqrt(2 * abs(np.log(S * np.exp((r - q) * T) / K)) / T)
    return float(np.clip(max(brenner, inflection), 0.05, 3.0))


def implied_vol(price, S, K, T, r, kind="call", q=0.0, tol=1e-12, max_iter=50, return_info=False):
    """
    隐含波动率（标量）：牛顿法为主，失败时二分法兜底

    返回 sigma；价格违反无套利边界时返回 nan。
    return_info=True 时返回 (sigma, method, iterations)，method ∈ {"newton", "bisection", "invalid"}
    """
    lower, upper = price_bounds(S, K, T, r, kind, q)
    if not (np.isfinite(price) and T > 0 and lower < price < upper):
        return (np.nan, "invalid", 0) if return_info else np.nan

    sigma = initial_guess(price, S, K, T, r, q)
    for i in range(1, max_iter + 1):
        diff = float(bs_price(S, K, T, r, sigma, kind, q)) - price
        if abs(diff) < tol:
            return (sigma, "newton", i) if return_info else sigma
        vega = float(bs_greeks(S, K, T, r, sigma, kind, q)["vega"])
        if vega < 1e-12:  # Vega 太小：牛顿步长会爆炸
            break
        step = diff / vega
        sigma -= step
        if not (SIGMA_MIN < sigma < SIGMA_MAX):
            break
        if abs(step) < 1e-12:
            return (sigma, "newton", i) if return_info else sigma

    lo, hi = SIGMA_MIN, SIGMA_MAX  # 二分法：价格对 σ 单调递增，必然收敛
    n = 0
    for n in range(1, 200):
        mid = 0.5 * (lo + hi)
        if float(bs_price(S, K, T, r, mid, kind, q)) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10:
            break
    sigma = 0.5 * (lo + hi)
    if min(sigma - SIGMA_MIN, SIGMA_MAX - sigma) < 1e-6:  # 收敛到搜索区间端点：报价异常
        return (np.nan, "invalid", n) if return_info else np.nan
    return (sigma, "bisection", n) if return_info else sigma


def implied_vol_chain(prices, S, K, T, r, kind, q=0.0) -> np.ndarray:
    """对一组合约逐个求 IV（S、K、T、kind 可以是数组或标量）"""
    prices = np.asarray(prices, dtype=float)
    S, K, T, kind = (np.broadcast_to(np.asarray(x), prices.shape) for x in (S, K, T, kind))
    return np.array(
        [implied_vol(p, s, k, t, r, kd, q) for p, s, k, t, kd in zip(prices.ravel(), S.ravel(), K.ravel(),
                                                                      T.ravel(), kind.ravel())]
    ).reshape(prices.shape)


def implied_forward(call_px, put_px, K, r, T):
    """认购-认沽平价：F = K + (C - P)·e^{rT}"""
    return K + (np.asarray(call_px) - np.asarray(put_px)) * np.exp(r * T)


# ==================== SVI ====================


def svi_w(k, a, b, rho, m, s):
    """SVI 总隐含方差 w(k) = a + b[ρ(k-m) + sqrt((k-m)^2 + s^2)]"""
    x = np.asarray(k) - m
    return a + b * (rho * x + np.sqrt(x**2 + s**2))


def svi_derivatives(k, a, b, rho, m, s):
    """返回 (w, w', w'')"""
    x = np.asarray(k) - m
    root = np.sqrt(x**2 + s**2)
    return a + b * (rho * x + root), b * (rho + x / root), b * s**2 / root**3


def fit_svi(k, iv, T, weights=None):
    """对单个到期日拟合 SVI，返回参数数组 [a, b, rho, m, s]"""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w_mkt = iv**2 * T
    wts = np.ones_like(w_mkt) if weights is None else np.asarray(weights, float)

    def resid(p):
        return (svi_w(k, *p) - w_mkt) * wts

    x0 = [max(w_mkt.min() * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1]
    bounds = ([-1.0, 0.0, -0.999, -1.0, 1e-4], [1.0, 5.0, 0.999, 1.0, 2.0])
    res = least_squares(resid, x0, bounds=bounds)
    return res.x


def svi_butterfly_g(k, params):
    """
    Durrleman 条件：g(k) = (1 - k w'/(2w))^2 - (w')^2/4 · (1/w + 1/4) + w''/2
    g(k) ≥ 0 处处成立 ⇔ 该到期日不存在蝶式套利（隐含风险中性密度非负）
    """
    w, w1, w2 = svi_derivatives(k, *params)
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
