"""
波动率策略：跨式、价差、卖方策略风险
====================================

学习目标：
    1. 掌握常见期权组合（跨式、宽跨式、价差、铁鹰、蝶式）的到期收益与希腊字母
    2. 理解"买波动率"与"卖波动率"的盈亏取决于实现波动率与隐含波动率之差
    3. 用带跳跃的模拟路径展示卖方策略的收益分布：平时赚小钱、偶尔大亏
    4. 对比不同波动率风险溢价（VRP）水平下卖方策略的收益与风险，理解有限风险结构的作用

核心概念：
    - 跨式 Straddle：同一 K 的认购 + 认沽；宽跨式 Strangle：虚值认购 + 虚值认沽
    - 铁鹰 Iron Condor：卖宽跨式 + 买更虚值的宽跨式保护，亏损有上限
    - VRP = σ_imp − σ_real：平均为正，但会在危机中剧烈变负
    - 卖方收益分布：高胜率、负偏、肥尾（"压路机前捡硬币"）

前置知识：
    - 03-greeks.py、07-delta-hedging.py
    - CONCEPT.md 第 10 节

环境要求：
    - numpy、pandas、scipy、matplotlib、akshare
    - 第一至第四部分为模拟（不需要网络）；第五部分的市场快照需要网络，
      QTC_OFFLINE=1 时使用缓存或合成数据
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import skew

from option_data import RISK_FREE, chain_implied_vols, get_option_chain
from option_pricing import bs_greeks, bs_price
from utils.data import get_etf_daily
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

S0, T, R, IV = 3.0, 30 / 365, 0.0, 0.18
MULTIPLIER = 10000  # 合约单位（示例，以交易所最新规则为准）


def leg(kind, K, pos, T=T, iv=IV):
    """一条腿：(类型, 行权价, 头寸, 开仓价格)；pos=+1 买入，-1 卖出"""
    return (kind, K, pos, float(bs_price(S0, K, T, R, iv, kind)))


def strategy_payoff(S_T, legs):
    """到期盈亏（每份）"""
    total = np.zeros_like(np.asarray(S_T, dtype=float))
    for kind, K, pos, prem in legs:
        intrinsic = np.maximum(S_T - K, 0) if kind == "call" else np.maximum(K - S_T, 0)
        total = total + pos * (intrinsic - prem)
    return total


def strategy_value(S, legs, tau, iv=IV):
    """剩余期限 tau 时的组合盈亏（按 BS 重估）"""
    total = np.zeros_like(np.asarray(S, dtype=float))
    for kind, K, pos, prem in legs:
        total = total + pos * (bs_price(S, K, tau, R, iv, kind) - prem)
    return total


STRATEGIES = {
    "买入跨式": [leg("call", 3.0, +1), leg("put", 3.0, +1)],
    "买入宽跨式": [leg("call", 3.2, +1), leg("put", 2.8, +1)],
    "牛市认购价差": [leg("call", 2.9, +1), leg("call", 3.1, -1)],
    "卖出跨式": [leg("call", 3.0, -1), leg("put", 3.0, -1)],
    "卖出宽跨式": [leg("call", 3.2, -1), leg("put", 2.8, -1)],
    "铁鹰": [leg("call", 3.2, -1), leg("put", 2.8, -1), leg("call", 3.3, +1), leg("put", 2.7, +1)],
}


# ==================== 第一部分：策略地图 ====================


def strategy_map():
    """方向 × 波动率观点"""
    print("=" * 60)
    print("第一部分：策略地图")
    print("=" * 60)
    print("""
                       方向观点
              看跌          中性            看涨
           ┌──────────┬──────────────┬──────────┐
  看涨波动 │ 买认沽    │ 买跨式/宽跨式 │ 买认购    │
           ├──────────┼──────────────┼──────────┤
  中性波动 │ 熊市价差  │ 日历价差      │ 牛市价差  │
           ├──────────┼──────────────┼──────────┤
  看跌波动 │ 卖认购    │ 卖跨式/宽跨式 │ 卖认沽    │
           │ (备兑)    │ 铁鹰 / 蝶式   │           │
           └──────────┴──────────────┴──────────┘

    关键：买方押注"实现波动 > 隐含波动"，卖方押注"实现波动 < 隐含波动"。
    """)


# ==================== 第二部分：组合收益与希腊字母 ====================


def strategy_profiles():
    """各策略的盈亏特征与希腊字母"""
    print("\n" + "=" * 60)
    print("第二部分：常见组合的收益结构与希腊字母")
    print("=" * 60)

    strikes = sorted({K for legs in STRATEGIES.values() for _, K, _, _ in legs})
    grid = np.union1d(np.linspace(0.01, 6.0, 6000), strikes)  # 网格包含所有行权价，极值才准确
    rows = []
    for name, legs in STRATEGIES.items():
        pnl = strategy_payoff(grid, legs)
        net = -sum(pos * prem for _, _, pos, prem in legs)  # 正数 = 净收入
        sign_change = np.where(np.diff(np.sign(pnl)) != 0)[0]
        breakeven = ", ".join(f"{grid[i]:.3f}" for i in sign_change)
        right_slope = pnl[-1] - pnl[-2]  # 标的无限上涨时盈亏的斜率：>0 收益无上限，<0 亏损无上限
        gain_txt = "无限" if right_slope > 1e-12 else f"{pnl.max():+.4f}"
        loss_txt = "无限(上涨方向)" if right_slope < -1e-12 else f"{pnl.min():+.4f}"
        g = {k: sum(pos * bs_greeks(S0, K, T, R, IV, kind)[k] for kind, K, pos, _ in legs) for k in
             ("delta", "gamma", "vega", "theta")}
        rows.append([name, net, gain_txt, loss_txt, breakeven, g["delta"], g["gamma"], g["vega"] / 100,
                     g["theta"] / 365])
    df = pd.DataFrame(rows, columns=["策略", "净权利金(+收)", "最大收益", "最大亏损", "盈亏平衡点", "Delta",
                                     "Gamma", "Vega/点", "Theta/天"])
    print(f"S={S0}, {T * 365:.0f} 天, IV={IV:.0%}（每份；一张合约 × {MULTIPLIER}）：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("""
    💡 规律：
       - 买跨式 / 宽跨式：Gamma > 0、Vega > 0、Theta < 0 → 付"租金"等大行情
       - 卖跨式 / 宽跨式：Gamma < 0、Theta > 0 → 每天收租，大行情时亏损理论上无上限
       - 铁鹰：用两条更虚值的买入腿把亏损封顶，代价是少收一部分权利金
       - 牛市价差：主要是方向性头寸，Vega 和 Theta 都较小
    """)

    setup_chinese_font()
    S = np.linspace(2.5, 3.5, 301)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (name, legs) in zip(axes.ravel(), STRATEGIES.items()):
        ax.plot(S, strategy_payoff(S, legs), lw=2, label="到期")
        ax.plot(S, strategy_value(S, legs, T * 0.5), "--", label=f"剩余 {T * 365 / 2:.0f} 天")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(S0, color="gray", ls=":", lw=0.8)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("标的价格")
    fig.suptitle(f"常见期权组合盈亏（每份，IV={IV:.0%}）")
    fig.tight_layout()
    path = save_figure(fig, "08_strategy_payoffs.png", subdir="phase-12")
    print(f"📈 图已保存：{path}")


# ==================== 第三部分：买波动率 vs 卖波动率 ====================


def long_vs_short_vol():
    """跨式盈亏取决于实现波动 vs 隐含波动"""
    print("\n" + "=" * 60)
    print("第三部分：买跨式的期望盈亏 = f(实现波动率 − 隐含波动率)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    Z = rng.standard_normal(200_000)
    legs = STRATEGIES["买入跨式"]
    cost = sum(prem for _, _, _, prem in legs)
    rows = []
    for real in [0.10, 0.14, 0.18, 0.22, 0.26]:
        S_T = S0 * np.exp(-0.5 * real**2 * T + real * np.sqrt(T) * Z)
        pnl = strategy_payoff(S_T, legs)
        rows.append([f"{real:.0%}", f"{IV - real:+.0%}", pnl.mean(), pnl.mean() / cost, np.mean(pnl > 0)])
    df = pd.DataFrame(rows, columns=["实现波动率", "VRP(隐含−实现)", "平均盈亏", "平均收益率", "盈利概率"])
    print(f"以 IV={IV:.0%} 买入平值跨式（成本 {cost:.4f}），持有到期，标的无漂移：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("""
    💡 隐含 = 实现时买方期望盈亏约为 0；VRP > 0 时买方期望为负、卖方期望为正。
       这就是 01 课说的："当隐含波动率高于随后的实现波动率时，买方的期望收益为负"。
       但"期望为正"不等于"值得做" —— 下一部分看卖方的收益分布。
    """)


# ==================== 第四部分：卖方尾部风险 ====================


def simulate_monthly(n_months, sigma, jump_prob, jump_size, seed):
    """每月标的收益 = 正态扩散 + 小概率跳跃（暴跌）；返回月末价格（每月从 S0 重新开始）"""
    rng = np.random.default_rng(seed)
    t = 1 / 12
    ret = rng.normal(-0.5 * sigma**2 * t, sigma * np.sqrt(t), n_months)
    jumps = rng.random(n_months) < jump_prob
    ret = ret + np.where(jumps, np.log(1 + jump_size), 0.0)
    return S0 * np.exp(ret), jumps


def monthly_pnl(S_T, iv, wings=False, k_put=0.92, k_call=1.08, wing=0.06):
    """每月卖一次宽跨式（或铁鹰）并持有到期；权利金按 BS(iv) 计算"""
    t = 1 / 12
    kp, kc = k_put * S0, k_call * S0
    premium = float(bs_price(S0, kp, t, R, iv, "put") + bs_price(S0, kc, t, R, iv, "call"))
    loss = np.maximum(kp - S_T, 0) + np.maximum(S_T - kc, 0)
    if wings:
        kp2, kc2 = (k_put - wing) * S0, (k_call + wing) * S0
        premium -= float(bs_price(S0, kp2, t, R, iv, "put") + bs_price(S0, kc2, t, R, iv, "call"))
        loss -= np.maximum(kp2 - S_T, 0) + np.maximum(S_T - kc2, 0)
    return premium - loss, premium


def stats(pnl, premium):
    cum = np.cumsum(pnl)
    dd = (cum - np.maximum.accumulate(np.concatenate([[0], cum]))[1:]).min()
    return {
        "胜率": np.mean(pnl > 0),
        "月均盈亏": pnl.mean(),
        "月标准差": pnl.std(),
        "年化夏普": pnl.mean() / pnl.std() * np.sqrt(12),
        "偏度": skew(pnl),
        "最差一月": pnl.min(),
        "最差月/月权利金": pnl.min() / premium,
        "最大回撤(月权利金)": dd / premium,
    }


def tail_risk():
    """带跳跃的模拟：卖方收益分布"""
    print("\n" + "=" * 60)
    print("第四部分：卖方尾部风险 —— 压路机前捡硬币")
    print("=" * 60)

    sigma, jump_prob, jump_size, n_months = 0.15, 0.03, -0.15, 600
    sell_iv = 0.21  # 卖出时的隐含波动率：比扩散波动率高 6 个点（假设市场给了较厚的 VRP）
    S_T, jumps = simulate_monthly(n_months, sigma, jump_prob, jump_size, seed=0)
    print(f"模拟 {n_months} 个月（50 年）：扩散波动率 {sigma:.0%}，每月 {jump_prob:.0%} 概率暴跌 {jump_size:.0%}，"
          f"实际发生 {jumps.sum()} 次")
    print(f"每月按 IV={sell_iv:.0%} 卖出 1 个月期宽跨式（K = 92% / 108% 现价），持有到期\n")

    results, curves = {}, {}
    for name, wings in [("卖出宽跨式", False), ("铁鹰(翼宽6%)", True)]:
        pnl, prem = monthly_pnl(S_T, sell_iv, wings=wings)
        results[name] = stats(pnl, prem)
        results[name]["月权利金"] = prem
        curves[name] = (pnl, prem)
    df = pd.DataFrame(results)
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    pnl, prem = curves["卖出宽跨式"]
    worst = np.argsort(pnl)[:5]
    print(f"\n   最差的 5 个月：{', '.join(f'第{i}月 {pnl[i]:+.4f}' for i in sorted(worst))}")
    print(f"   其中发生跳跃的有 {int(jumps[worst].sum())} 个月")

    # 回测窗口是否包含尾部事件：所有 5 年滚动窗口的夏普
    window = 60
    sharpe = np.array([pnl[i:i + window].mean() / pnl[i:i + window].std() * np.sqrt(12)
                       for i in range(n_months - window + 1)])
    has_jump = np.array([jumps[i:i + window].any() for i in range(n_months - window + 1)])
    print(f"\n   所有 5 年滚动窗口（{len(sharpe)} 个）的卖宽跨式夏普：")
    print(f"     不含跳跃的窗口 {int((~has_jump).sum())} 个，夏普中位数 "
          f"{np.median(sharpe[~has_jump]) if (~has_jump).any() else float('nan'):+.2f}")
    print(f"     含跳跃的窗口 {int(has_jump.sum())} 个，夏普中位数 {np.median(sharpe[has_jump]):+.2f}")
    print(f"     全样本夏普 {results['卖出宽跨式']['年化夏普']:+.2f}")
    r = results["卖出宽跨式"]
    mean_txt = "小幅正值" if r["月均盈亏"] > 0 else "负值"
    print(f"""
    💡 读表：
       - 卖宽跨式胜率 {r['胜率']:.1%}、月均盈亏为{mean_txt}（{r['月均盈亏']:+.4f}），偏度 {r['偏度']:+.2f}：
         盈利月很多，亏损月很少但很大
       - 最差一个月亏掉 {abs(r['最差月/月权利金']):.0f} 个月的权利金；铁鹰把单月亏损封顶，代价是平时少收权利金
         （本模拟中所有行权价按同一 IV 定价，保护腿显得便宜；真实市场虚值认沽 IV 更高（05 课），保护更贵）
       - 回测结果高度依赖样本期是否包含尾部事件：没有跳跃的 5 年窗口夏普明显更高，看起来"稳赚"
    """)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].hist(pnl / prem, bins=80, color="tab:red", alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("月盈亏（以当月权利金为单位）")
    axes[0].set_ylabel("月数（对数坐标）")
    axes[0].set_title("卖出宽跨式的月收益分布：负偏、肥尾")
    for name, (p, pr) in curves.items():
        axes[1].plot(np.cumsum(p) / pr, label=name)
    for i in np.where(jumps)[0]:
        axes[1].axvline(i, color="gray", alpha=0.3, lw=0.8)
    axes[1].set_xlabel("月份（灰线 = 跳跃月）")
    axes[1].set_ylabel("累计盈亏（月权利金）")
    axes[1].set_title("累计盈亏")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    path = save_figure(fig, "08_short_vol_tail.png", subdir="phase-12")
    print(f"📈 图已保存：{path}")
    return S_T, jumps


def vrp_comparison(S_T_jump):
    """不同 VRP 水平：有无跳跃"""
    print("\n" + "=" * 60)
    print("第五部分：波动率风险溢价（VRP）对卖方的意义")
    print("=" * 60)

    sigma = 0.15
    S_T_nojump, _ = simulate_monthly(len(S_T_jump), sigma, 0.0, 0.0, seed=0)
    rows = []
    for iv in [0.13, 0.15, 0.17, 0.19, 0.21]:
        for label, S_T in [("无跳跃", S_T_nojump), ("有跳跃", S_T_jump)]:
            pnl, prem = monthly_pnl(S_T, iv)
            s = stats(pnl, prem)
            rows.append([f"{iv:.0%}", f"{iv - sigma:+.0%}", label, prem, s["月均盈亏"], s["年化夏普"], s["偏度"],
                         s["最差月/月权利金"]])
    df = pd.DataFrame(rows, columns=["卖出IV", "IV−扩散σ", "情景", "月权利金", "月均盈亏", "年化夏普", "偏度",
                                     "最差月/权利金"])
    print(f"扩散波动率 {sigma:.0%}；\"有跳跃\"情景与第四部分相同（同一组随机数）：")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # 盈亏平衡 IV：让有跳跃情景下的月均盈亏 = 0
    grid = np.linspace(0.10, 0.35, 251)
    means = np.array([monthly_pnl(S_T_jump, iv)[0].mean() for iv in grid])
    be = grid[np.argmax(means > 0)] if (means > 0).any() else np.nan
    print(f"\n   有跳跃时，卖宽跨式的盈亏平衡 IV ≈ {be:.1%}（扩散波动率仅 {sigma:.0%}）")
    print("""
    💡 跳跃风险本身就"值"一部分波动率溢价：只看平静时期的实现波动率，会以为 VRP 很厚，
       其实其中相当一部分是对尾部风险的补偿。VRP 是承担尾部风险的报酬，不是免费午餐。
    """)


# ==================== 第六部分：市场快照 ====================


def market_snapshot():
    """近月平值 IV vs 标的历史实现波动率"""
    print("\n" + "=" * 60)
    print("第六部分：市场快照 —— 隐含波动率 vs 历史实现波动率")
    print("=" * 60)

    chain, meta = get_option_chain("510050", verbose=True)
    ivs = chain_implied_vols(chain, r=RISK_FREE)
    otm = ivs[ivs["otm"] & ivs["iv"].notna()]
    near = otm[otm["expiry"] == otm["expiry"].min()]
    atm = near.loc[near["k"].abs().idxmin()]

    end = meta["snapshot"]
    start = end - pd.Timedelta(days=200)
    px = get_etf_daily("510050", f"{start:%Y%m%d}", f"{end:%Y%m%d}", adjust="hfq")["close"]
    ret = np.log(px).diff().dropna()
    rv20, rv60 = ret.tail(20).std() * np.sqrt(252), ret.tail(60).std() * np.sqrt(252)
    print(f"   {meta['name']}，快照 {end:%Y-%m-%d}（{'合成数据' if meta['source'] == 'synthetic' else '真实行情'}）")
    print(f"   近月（{int(atm['days'])} 天）平值附近 IV = {atm['iv']:.2%}")
    print(f"   过去 20 日实现波动率 = {rv20:.2%}，过去 60 日 = {rv60:.2%}（后复权收盘价，年化 √252）")
    print(f"   IV − 20 日实现 = {(atm['iv'] - rv20) * 100:+.2f} 点，IV − 60 日实现 = {(atm['iv'] - rv60) * 100:+.2f} 点")
    print("""
    ⚠️ 这只是一个时点的对比，而且用的是"过去"的实现波动率；VRP 的严格定义是
       今天的 IV 减去"未来"到期前的实现波动率。要研究 VRP，需要长期的 IV 历史序列。
    """)


# ==================== 主函数 ====================


def main():
    strategy_map()
    strategy_profiles()
    long_vs_short_vol()
    S_T_jump, _ = tail_risk()
    vrp_comparison(S_T_jump)
    market_snapshot()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 第四部分中把跳跃概率改为 1%、5%，或把跳跃幅度改为 −25%，卖方的夏普和最差月如何变化？
    2. 实现"保证金约束"：每月可卖张数 = 账户权益 × 30% / 单张保证金（用 06 课的公式），
       计算账户净值曲线与最大回撤，看看杠杆如何放大尾部风险
    3. 给卖宽跨式加止损：单月亏损超过 3 倍权利金就平仓（需要按日模拟路径），止损能消除尾部风险吗？
    4. 用 STRATEGIES 的写法实现蝶式（买 2.9 认购、卖 2 份 3.0 认购、买 3.1 认购）和日历价差，
       分析它们的希腊字母
    """)


if __name__ == "__main__":
    main()
