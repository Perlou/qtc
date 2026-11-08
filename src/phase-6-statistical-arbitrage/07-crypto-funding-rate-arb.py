"""
加密永续合约资金费率套利
========================

⚠️ 合规提示：中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动。
   本课仅使用交易所公开行情数据（资金费率、K 线）做学习研究，不涉及任何交易、开户或资金操作，
   不构成任何投资建议。请遵守所在地法律法规。

学习目标：
    1. 理解永续合约没有到期日，如何用资金费率机制把合约价格"拴"在现货附近
    2. 用 ccxt 获取永续合约资金费率历史，计算年化、正费率占比，并观察费率随市场状态切换
    3. 拆解 Cash-and-Carry（现货多 + 永续空）的收益与成本：费率收入 vs 4 笔手续费
    4. 量化"套利 ≠ 无风险"：空头腿强平、开平仓基差、交易所与稳定币风险

核心概念：
    - 资金费率 F = 溢价指数 P + clamp(利息 I - P, ±0.05%)（常见设计，以交易所最新规则为准）
    - 费率为正：多头付给空头；为负：空头付给多头
    - 年化 ≈ 每期费率 × 每天结算次数 × 365

前置知识：
    - Phase 1 第 09 课（ccxt 基础）；本阶段 06 课（期现套利、持有成本）

环境要求：
    - pip install ccxt pandas numpy matplotlib
    - 访问海外交易所公开接口可能需要代理（.env 中设置 HTTPS_PROXY）
    - 结果缓存到 data/cache/stat_arb_*.parquet；QTC_OFFLINE=1 时优先读缓存，
      缓存不存在或获取失败时自动使用合成数据
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from stat_arb_utils import get_crypto_daily, get_funding_history, used_synthetic
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

FEE_PER_LEG = 0.0005  # 单边手续费示例 0.05%（吃单），以交易所最新费率为准
LONG_SINCE = "2021-01-01"


def compliance_notice():
    print("=" * 60)
    print("⚠️ 合规提示")
    print("=" * 60)
    print("""
    中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动。
    本课仅使用交易所公开行情数据（资金费率、K 线）做学习研究，
    不涉及任何交易、开户或资金操作，不构成任何投资建议。请遵守所在地法律法规。
    """)


def periods_per_day(s: pd.Series) -> float:
    """根据相邻记录的时间间隔推断每天结算次数"""
    hours = s.index.to_series().diff().dt.total_seconds().median() / 3600
    return 24 / hours


# ==================== 第一部分：永续合约与资金费率机制 ====================


def funding_mechanism():
    print("\n" + "=" * 60)
    print("第一部分：永续合约与资金费率机制")
    print("=" * 60)

    print("""
    永续合约（Perpetual Swap）：可以加杠杆、双向交易，但没有到期日 → 没有"交割收敛"

    永续价格 > 现货（市场偏多）→ 资金费率为正 → 多头定期付钱给空头 → 做多变贵，价格被压回
    永续价格 < 现货（市场偏空）→ 资金费率为负 → 空头付钱给多头

    常见设计（参数、结算周期、上下限因交易所和合约而异，以交易所最新规则为准）：
        F = P + clamp(I - P, -0.05%, +0.05%)     P 溢价指数，I 利息部分（常见每 8 小时 0.01%）
        资金费用 = 持仓名义价值 × F；常见每 8 小时结算一次（也有 4 小时、1 小时）

    ┌────────────────────────────────────────────────────────────────┐
    │  Cash-and-Carry（Delta 中性）                                  │
    │    现货：买入 1 BTC          Delta = +1                        │
    │    永续：做空 1 BTC          Delta = -1    → 组合 Delta ≈ 0    │
    │    收益：费率为正时，空头每期收取资金费                        │
    │    成本：开平仓 4 笔手续费、滑点、资金占用                     │
    └────────────────────────────────────────────────────────────────┘
    """)
    for rate in [0.0001, 0.0003, -0.0001]:
        print(f"   每 8 小时 {rate:+.2%} → 年化约 {rate * 3 * 365:+.2%}")


# ==================== 第二部分：近期资金费率（OKX） ====================


def summarize(s: pd.Series, label: str) -> dict:
    ppd = periods_per_day(s)
    last30 = s[s.index >= s.index[-1] - pd.Timedelta(days=30)]
    return {
        "标的": label, "起": str(s.index[0].date()), "止": str(s.index[-1].date()), "期数": len(s),
        "每天结算": round(ppd, 1), "平均每期": f"{s.mean():+.4%}", "年化": f"{s.mean() * ppd * 365:+.2%}",
        "近30天年化": f"{last30.mean() * ppd * 365:+.2%}", "正费率占比": f"{(s > 0).mean():.0%}",
        "最高": f"{s.max():+.4%}", "最低": f"{s.min():+.4%}",
    }


def recent_funding() -> pd.Series:
    print("\n" + "=" * 60)
    print("第二部分：近期资金费率（OKX 公开接口，约最近 3 个月）")
    print("=" * 60)

    print("""
    ccxt 统一接口（公开数据，不需要 API Key）：
        ex = ccxt.okx()
        ex.fetch_funding_rate_history("BTC/USDT:USDT", since=毫秒时间戳, limit=100)
        → [{"timestamp": ..., "fundingRate": 0.0001, ...}, ...]
    "BTC/USDT:USDT" 表示以 USDT 结算的 BTC 永续合约；OKX 公开接口只保留约 3 个月历史。
    """)
    rows, data = [], {}
    for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        s = get_funding_history("okx", sym)
        data[sym] = s
        rows.append(summarize(s, sym))
    print(pd.DataFrame(rows).to_string(index=False))
    return data["BTC/USDT:USDT"]


# ==================== 第三部分：长周期 —— 费率随市场状态切换 ====================


def long_history() -> pd.Series:
    print("\n" + "=" * 60)
    print(f"第三部分：长周期资金费率（Binance USDⓈ-M BTC 永续，{LONG_SINCE} 至今）")
    print("=" * 60)

    s = get_funding_history("binanceusdm", "BTC/USDT:USDT", since=LONG_SINCE, page_limit=1000)
    ppd = periods_per_day(s)
    g = s.groupby(s.index.year)
    table = pd.DataFrame({
        "期数": g.size(),
        "年化": g.mean() * ppd * 365,
        "正费率占比": g.apply(lambda x: (x > 0).mean()),
        "最高每期": g.max(),
        "最低每期": g.min(),
    })
    fmt = table.copy()
    for c in ["年化", "正费率占比"]:
        fmt[c] = table[c].map("{:+.2%}".format)
    for c in ["最高每期", "最低每期"]:
        fmt[c] = table[c].map("{:+.4%}".format)
    print(fmt.to_string())
    roll = s.rolling(int(30 * ppd)).mean() * ppd * 365
    print(f"\n   滚动 30 天年化费率：最高 {roll.max():+.2%}（{roll.idxmax().date()}），"
          f"最低 {roll.min():+.2%}（{roll.idxmin().date()}）")
    print("""
    💡 观察要点：
       - 年化费率在不同年份之间差异很大：牛市情绪高涨时费率高，熊市或恐慌时可能长期为负
       - 用某一段高费率时期外推"年化收益"是典型的营销话术，要看包含不同市场状态的长周期分布
       - 资金费率套利已被大量资金规模化执行，费率中枢可能被压低（拥挤）
    """)
    return s


# ==================== 第四部分：Cash-and-Carry 收益拆解 ====================


def carry_backtest(s: pd.Series):
    print("\n" + "=" * 60)
    print("第四部分：Cash-and-Carry 收益拆解 —— 费率收入 vs 手续费")
    print("=" * 60)

    ppd = periods_per_day(s)
    rt_cost = 4 * FEE_PER_LEG  # 现货买 + 卖、永续开 + 平
    print(f"   假设：单边手续费 {FEE_PER_LEG:.2%}，一次完整开平 4 笔 = {rt_cost:.2%}；"
          "资金费按固定名义金额计算（忽略价格变动对名义金额的影响）")
    rows = []
    for days in [7, 30, 90, 365]:
        gross = s.mean() * ppd * days
        rows.append([days, f"{gross:+.2%}", f"{rt_cost:.2%}", f"{gross - rt_cost:+.2%}",
                     f"{(gross - rt_cost) / days * 365:+.2%}"])
    print("\n📋 按长周期平均费率估算，持有不同天数（开平各一次）：")
    print(pd.DataFrame(rows, columns=["持有天数", "费率收入", "开平成本", "净收益", "净年化"]).to_string(index=False))

    # 规则对比：一直持有 vs 按费率正负开关（滞后 1 期决策，避免前视）
    signal_raw = s.rolling(3).mean().shift(1)
    results = {}
    for label, on in [("一直持有", pd.Series(1.0, index=s.index)),
                      ("近 3 期平均 > 0 才持有", (signal_raw > 0).astype(float)),
                      ("近 3 期平均 > 0.01% 才持有", (signal_raw > 0.0001).astype(float))]:
        on = on.fillna(0.0)
        switches = on.diff().abs().fillna(on.iloc[0]).sum() + on.iloc[-1]  # 期末仍持有则计一次平仓
        income = (on * s).sum()
        cost = switches / 2 * rt_cost  # 每次"开 + 平"一个来回 = 4 笔
        years = len(s) / ppd / 365
        results[label] = {"持有占比": f"{on.mean():.0%}", "开平次数": int(switches),
                          "费率收入": f"{income:+.2%}", "手续费": f"{cost:.2%}",
                          "净收益": f"{income - cost:+.2%}", "净年化": f"{(income - cost) / years:+.2%}"}
    print(f"\n📋 {s.index[0].date()} ~ {s.index[-1].date()} 不同持有规则（单位：占名义金额）：")
    print(pd.DataFrame(results).to_string())
    print("""
    💡 观察要点：
       1. 持有期越短，开平成本占比越大；频繁根据费率正负来回切换，手续费会吃掉大量收益
       2. 这里的收益是相对"名义金额"的。现货腿需要全额资金，永续空头还要额外保证金，
          按总占用资金计算的收益率更低
       3. 表中只算了费率和手续费，没有算基差、强平、交易所风险（第五部分）
    """)


# ==================== 第五部分：风险 —— 远比看起来多 ====================


def carry_risks():
    print("\n" + "=" * 60)
    print("第五部分：风险 —— 套利 ≠ 无风险")
    print("=" * 60)

    spot = get_crypto_daily("okx", "BTC/USDT")
    perp = get_crypto_daily("okx", "BTC/USDT:USDT")
    df = pd.concat({"spot": spot, "perp": perp}, axis=1).dropna()
    prem = (df["perp"] / df["spot"] - 1) * 1e4
    print(f"   OKX BTC 日线（UTC 收盘）{df.index[0].date()} ~ {df.index[-1].date()}，{len(df)} 天")
    print(f"   永续相对现货溢价（bp）：均值 {prem.mean():+.1f}，标准差 {prem.std():.1f}，"
          f"最低 {prem.min():+.1f}，最高 {prem.max():+.1f}")
    print("   → 开仓时溢价、平仓时折价，会产生一笔与费率无关的盈亏（基差风险）")

    # 空头腿强平：现货与永续分属不同账户时，现货的浮盈不能补充永续保证金
    window, mm = 30, 0.005  # 观察窗口 30 天；维持保证金率示例 0.5%
    px = df["spot"].to_numpy()
    max_rise = np.array([px[i:i + window + 1].max() / px[i] - 1 for i in range(len(px) - window)])
    rows = []
    for lev in [2, 3, 5, 10, 20]:
        threshold = 1 / lev - mm  # 价格上涨超过约 (1/杠杆 - 维持保证金率) 时空头被强平（近似）
        rows.append([f"{lev}x", f"{threshold:.1%}", f"{(max_rise >= threshold).mean():.1%}"])
    print(f"\n📋 永续空头腿在 {window} 天内被强平的历史频率（不追加保证金，按每天开仓统计；近似计算）：")
    print(pd.DataFrame(rows, columns=["空头杠杆", "强平所需涨幅≈", "被强平的开仓日占比"]).to_string(index=False))
    print(f"   {window} 天内最大涨幅：中位数 {np.median(max_rise):.1%}，最大 {max_rise.max():.1%}")
    print("""
    ┌──────────────────┬───────────────────────────────────────────────────────────┐
    │ 风险             │ 说明                                                      │
    ├──────────────────┼───────────────────────────────────────────────────────────┤
    │ 费率反转         │ 情绪逆转后费率转负且持续，空头从收钱变付钱                │
    │ 空头腿强平       │ 暴涨时永续空头浮亏，现货盈利在另一个账户无法抵充保证金，  │
    │                  │ 空头被强平后"套利"变成裸多                                │
    │ 基差风险         │ 开平仓时永续相对现货的溢价变化                            │
    │ 自动减仓（ADL）  │ 极端行情下交易所可能强制减少盈利方仓位，对冲被打破        │
    │ 交易所风险       │ 倒闭、冻结提币、被盗（例如 2022 年 FTX 破产）             │
    │ 稳定币风险       │ 计价稳定币可能脱锚（例如 2023 年 3 月 USDC 曾短暂脱锚）   │
    │ 拥挤             │ 资金规模化执行后费率中枢被压低，历史高费率不可外推        │
    │ 合规风险         │ 各司法辖区监管差异很大；中国内地相关业务属于非法金融活动  │
    └──────────────────┴───────────────────────────────────────────────────────────┘

    💡 资金费率套利是"承担尾部风险、换取稳定小收益"的策略：平时像收房租，
       极端行情一次就可能抹掉数月甚至数年的收益。"无风险年化 X%"是营销话术。
    """)


def plot_funding(recent: pd.Series, long: pd.Series):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    ppd = periods_per_day(long)
    roll = long.rolling(int(30 * ppd)).mean() * ppd * 365 * 100
    axes[0].plot(roll.index, roll, lw=1)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title(f"BTC 永续资金费率：滚动 30 天年化（%）{'【含合成数据】' if used_synthetic() else ''}")
    axes[1].bar(recent.index, recent * 100, width=0.25)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("OKX BTC 永续近期每期资金费率（%）")
    fig.tight_layout()
    print(f"\n🖼️  图已保存：{save_figure(fig, '07_funding_rate.png', subdir='phase-6')}")


# ==================== 主函数 ====================


def main():
    compliance_notice()
    funding_mechanism()
    recent = recent_funding()
    long = long_history()
    carry_backtest(long)
    carry_risks()
    plot_funding(recent, long)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 对 ETH、SOL 永续重复第三部分，哪个币种的费率更高？费率高是否意味着风险也更高？
    2. 把第四部分的单边手续费改为 0.02%（挂单）和 0.1%，"按费率开关"规则的净收益如何变化？
    3. 在第五部分加入"每天把现货盈利划转到合约账户"的假设，重新计算被强平的比例
    4. 思考：如果市场上所有人都在做资金费率套利，费率会怎样变化？这和 08 课的"拥挤"有什么联系？
    """)


if __name__ == "__main__":
    main()
