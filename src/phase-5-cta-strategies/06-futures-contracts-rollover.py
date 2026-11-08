"""
期货合约、保证金与展期
======================

学习目标：
    1. 掌握合约乘数、最小变动价位、合约价值、保证金与杠杆的计算
    2. 用真实结算价模拟"逐日盯市"，理解追加保证金与强平的触发过程
    3. 实现主力合约选择规则（前一日持仓量最大 + 只向远月切换），并与新浪主力连续对比
    4. 构造价差调整与比例调整两种连续合约，验证"比例调整收益 = 实际持仓收益"
    5. 理解期限结构（Contango / Backwardation）与展期收益

核心概念：
    - 合约价值 = 价格 × 合约乘数；占用保证金 = 合约价值 × 手数 × 保证金比例
    - 逐日盯市：每天按结算价（不是收盘价）划转盈亏
    - 主力连续：把不同时期的主力合约拼接成一条长序列；换月处有虚假跳空
    - 价差调整（保点数）vs 比例调整（保收益率），都是"向后调整"：最新价格不变，修改历史价格
    - 展期收益：期货总收益 ≈ 价格变化 + 展期收益

前置知识：
    - Phase 1 第 06 课（复权）、第 09 课（期货数据与换月跳空）

环境要求：
    - pip install akshare pandas numpy matplotlib
    - 无网络时可用 QTC_OFFLINE=1 运行（有本地缓存时读缓存，否则使用合成数据）
    ⚠️ 合约乘数、保证金比例等为示例值，以交易所和期货公司最新规定为准
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from cta_utils import (SPECS, UNIVERSE, wide_panel, build_continuous, get_contracts, get_continuous, get_main_sina,
                       select_main)
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()


# ==================== 第一部分：合约要素、保证金与杠杆 ====================


def contract_specs():
    print("=" * 60)
    print("第一部分：合约要素、保证金与杠杆")
    print("=" * 60)

    print("""
    期货像"预售房合同"：今天约定未来的价格，只交一点定金（保证金）；
    但合约标准化、可随时转让，而且每天都按市价结算盈亏。

    合约价值 = 价格 × 合约乘数          一手保证金 = 合约价值 × 保证金比例
    名义杠杆 ≈ 1 / 保证金比例           1 tick 价值 = 最小变动价位 × 合约乘数
    """)
    rows = []
    for s in UNIVERSE:
        px = get_continuous(s)["raw_close"].iloc[-1]
        sp = SPECS[s]
        value = px * sp["mult"]
        rows.append([s, sp["name"], sp["exchange"], f"{sp['mult']} {sp['unit']}" if sp["unit"] != "点" else f"{sp['mult']} 元/点",
                     px, value, sp["margin"], value * sp["margin"], 1 / sp["margin"], sp["tick"] * sp["mult"]])
    t = pd.DataFrame(rows, columns=["代码", "品种", "交易所", "合约乘数", "最新价", "合约价值", "保证金比例(示例)",
                                    "一手保证金", "名义杠杆", "1tick价值"])
    print(t.to_string(index=False, formatters={
        "最新价": "{:.2f}".format, "合约价值": "{:,.0f}".format, "保证金比例(示例)": "{:.0%}".format,
        "一手保证金": "{:,.0f}".format, "名义杠杆": "{:.0f}x".format, "1tick价值": "{:g}".format}))
    print("""
    ⚠️ 乘数、tick、保证金比例均为示例值，以交易所和期货公司最新规定为准。
       保证金比例 = 交易所标准 + 期货公司加收，节假日前、行情剧烈时、临近交割月都可能上调。
    💡 杠杆是资金效率工具，风险由"名义敞口"决定：CTA 研究中用"名义敞口 / 净值"衡量风险，
       而不是保证金占用。国债期货 T 的保证金比例低、杠杆高，但它的价格波动也小得多。
    """)


# ==================== 第二部分：逐日盯市与追加保证金 ====================


def mark_to_market(symbol: str = "RB", lots: int = 5, days: int = 15):
    print("\n" + "=" * 60)
    print("第二部分：逐日盯市与追加保证金")
    print("=" * 60)

    cont = get_continuous(symbol)
    raw = get_contracts(symbol)
    sp = SPECS[symbol]
    # 选历史上主力合约跌幅最大的 15 日窗口，模拟一个做多账户
    chg = cont["raw_close"].shift(-days) / cont["raw_close"] - 1
    same = cont["contract"].shift(-days) == cont["contract"]
    start = chg.where(same).idxmin()
    code = cont.at[start, "contract"]
    c = raw[raw["contract"] == code].set_index("date").sort_index()
    window = c.loc[start:].iloc[: days + 1]

    entry = window["close"].iloc[0]
    margin0 = entry * sp["mult"] * lots * sp["margin"]
    deposit = margin0 * 1.5  # 入金 = 初始保证金 × 1.5
    print(f"""
    情景：{start.date()} 以收盘价 {entry:.0f} 买入 {lots} 手 {code}，保证金比例 {sp['margin']:.0%}（示例）
          初始保证金 {margin0:,.0f} 元，账户入金 {deposit:,.0f} 元（保证金的 1.5 倍）
    规则：每天按"结算价"计算盈亏并划转；权益 < 持仓保证金 → 追加保证金通知（简化：不区分维持保证金）
    """)
    rows, equity, prev, first_call = [], deposit, entry, None
    for d, r in window.iterrows():
        settle = r["settle"] if np.isfinite(r["settle"]) and r["settle"] > 0 else r["close"]
        pnl = (settle - prev) * sp["mult"] * lots
        equity += pnl
        req = settle * sp["mult"] * lots * sp["margin"]
        call = equity < req
        if call and first_call is None:
            first_call = d
        rows.append([d.date(), r["close"], settle, pnl, equity, req, "⚠️ 追保" if call else ""])
        prev = settle
    t = pd.DataFrame(rows, columns=["日期", "收盘价", "结算价", "当日盈亏", "账户权益", "所需保证金", "状态"])
    print(t.to_string(index=False, formatters={"当日盈亏": "{:+,.0f}".format, "账户权益": "{:,.0f}".format,
                                               "所需保证金": "{:,.0f}".format}))
    n_call = int((t["状态"] != "").sum())
    loss = deposit - equity
    print(f"""
    📊 第一行是开仓当天：盈亏按"结算价 - 开仓价"计算，之后每天按"今日结算价 - 昨日结算价"计算
       {days} 个交易日结算价变动 {prev / entry - 1:+.1%}，账户亏损 {loss:,.0f} 元，
       相当于初始保证金的 {loss / margin0:.0%}；共有 {n_call} 天权益低于所需保证金""")
    if first_call is not None:
        print(f"""       首次追保发生在 {first_call.date()}：现实中若次日开盘前不补足，期货公司会强行平仓，
       表中之后的"负权益"只是假设没有被强平时的账面数字（穿仓风险由客户承担）""")
    print("""
    💡 结算价通常是某个时段的成交量加权均价（各交易所、各品种算法不同），所以按收盘价回测会与实际盈亏有小差异。
       回测中如果仓位过重，"理论上能扛过去"的回撤在实盘中可能已经被强平。
    """)


# ==================== 第三部分：主力合约选择 ====================


def main_contract_selection(symbol: str = "RB", year: int = 2024):
    print("\n" + "=" * 60)
    print("第三部分：主力合约选择 —— 不能用\"上帝视角\"")
    print("=" * 60)

    print("""
    规则：① 今天的主力 = 昨天收盘后持仓量最大的合约（今天开盘前就能知道）
          ② 只向更远月切换，不回切（避免在两个合约之间来回跳）
    ❌ 常见错误：用"今天"的持仓量决定"今天"的主力 —— 今天收盘前你不可能知道
    """)
    raw = get_contracts(symbol)
    oi, close = wide_panel(raw, "hold"), wide_panel(raw, "close")
    main = select_main(oi, close)
    cheat = oi.where(close.notna()).fillna(-1).idxmax(axis=1)  # 未来函数版：当天持仓量，且可回切

    y = main[main.index.year == year]
    sw = y[y != y.shift()].iloc[1:]
    print(f"📊 {year} 年 {symbol} 主力切换（本课规则）：")
    for d, c in sw.items():
        prev_d = main.index[main.index.get_loc(d) - 1]
        old = main.loc[prev_d]
        print(f"   {d.date()}  {old} → {c}   前一日持仓量：{old} {oi.at[prev_d, old]:,.0f}  {c} {oi.at[prev_d, c]:,.0f}")

    sina = get_main_sina(symbol)
    cont = build_continuous(raw)
    j = pd.concat([cont["raw_close"], sina["close"]], axis=1, keys=["本课", "新浪"]).dropna()
    same = np.isclose(j["本课"], j["新浪"]).mean()
    n_flip = int((cheat != cheat.shift()).sum() - 1)
    n_main = int((main != main.shift()).sum() - 1)
    common = main.index.intersection(cheat.index)
    n_diff = int((main.loc[common] != cheat.loc[common]).sum())
    print(f"""
    📊 与新浪主力连续（{symbol}0）对比：收盘价完全相同的交易日占 {same:.1%}
       差异来自主力切换规则不同（新浪的具体规则未公开，切换日可能早几天或晚几天）
    📊 "当天持仓量 + 允许回切"的错误版本：切换 {n_flip} 次（本课规则 {n_main} 次），
       有 {n_diff} 个交易日选出的主力与本课规则不同
       → 错误版本在切换日"提前一天"换到了新主力：它用到了当天收盘后才公布的持仓量，
         在回测里看起来换月时点更"完美"；持仓量接近的品种上，它还可能在两个合约之间来回跳
    """)
    return cont, sina


# ==================== 第四部分：价差调整 vs 比例调整 ====================


def back_adjustment(cont: pd.DataFrame, sina: pd.DataFrame, symbol: str = "RB"):
    print("\n" + "=" * 60)
    print("第四部分：价差调整 vs 比例调整")
    print("=" * 60)

    print("""
    设换月日为 d，换月前一日新、旧合约收盘价分别为 P_new、P_old（向后调整：d 之前的价格全部修改）
       价差调整：P̃_t = P_t + (P_new - P_old)          保留点数差，百分比收益失真，历史价格可能为负
       比例调整：P̃_t = P_t × (P_new / P_old)          保留百分比收益，点数被扭曲
    多次换月时调整从最近一次往前累积；最新价格保持真实。
    """)
    rolls = cont[cont["roll"]]
    print(f"📊 {symbol} 最近 5 次换月：")
    show = rolls.tail(5)[["contract", "raw_close", "roll_gap", "roll_ratio"]].copy()
    show.columns = ["新主力", "换月日真实收盘", "新-旧价差(前一日)", "新/旧比例(前一日)"]
    print(show.to_string(formatters={"新/旧比例(前一日)": "{:.4f}".format}))

    # 验证：比例调整连续价在换月日的收益 = 新合约的真实收益（前一日已按新合约计）
    raw = get_contracts(symbol)
    close_w = wide_panel(raw, "close")
    errs = []
    for d in rolls.index:
        i = cont.index.get_loc(d)
        if i == 0:
            continue
        p = cont.index[i - 1]
        real = close_w.at[d, cont.at[d, "contract"]] / close_w.at[p, cont.at[d, "contract"]] - 1
        errs.append(abs(cont["close"].iloc[i] / cont["close"].iloc[i - 1] - 1 - real))
    r_raw = cont["raw_close"].pct_change()
    r_ratio = cont["close"].pct_change()
    r_diff = cont["close_diff"].pct_change()
    print(f"""
    ✅ 验证：{len(errs)} 个换月日上，"比例调整价收益"与"新合约真实收益"的最大差异 = {max(errs) if errs else 0:.2e}
       → 比例调整后的连续价收益，就是"每天持有当天主力合约"的真实收益

    📊 换月日的收益对比（{len(errs)} 个换月日的平均绝对值）：
       未调整拼接：{r_raw[cont['roll']].abs().mean():.2%}    比例调整：{r_ratio[cont['roll']].abs().mean():.2%}
       非换月日平均绝对收益：{r_ratio[~cont['roll']].abs().mean():.2%}（作为参照）
    📊 全样本总收益：未调整拼接 {cont['raw_close'].iloc[-1] / cont['raw_close'].iloc[0] - 1:+.1%}，
       比例调整 {cont['close'].iloc[-1] / cont['close'].iloc[0] - 1:+.1%}，
       价差调整（按百分比算，❌失真）{(1 + r_diff.fillna(0)).prod() - 1:+.1%}
    """)

    neg = []
    for s in UNIVERSE:
        c = get_continuous(s)
        if c["close_diff"].min() <= 0:
            neg.append(f"{SPECS[s]['name']}({c['close_diff'].min():.0f})")
    if neg:
        print(f"    ⚠️ 价差调整后历史价格出现 ≤ 0 的品种：{'、'.join(neg)} —— 此时百分比收益完全没有意义")
    else:
        print("    本样本中没有品种的价差调整价格跌到 0 以下，但样本越长、累积价差越大，越可能出现")
    print("""
    ┌────────────────────────────┬──────────────────────────────────┐
    │ 用途                       │ 用哪个价格                       │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 信号（均线、突破、ATR）    │ 比例调整连续价（更通用）         │
    │ 收益 / 盈亏                │ 比例调整收益 = 实际持有合约收益  │
    │ 手数 / 保证金 / 合约价值   │ 未调整的真实合约价格             │
    │ 换月成本                   │ 换月日平旧开新，计两次成本       │
    └────────────────────────────┴──────────────────────────────────┘
    """)


# ==================== 第五部分：期限结构与展期收益 ====================


def term_structure(symbol: str = "CU"):
    print("\n" + "=" * 60)
    print("第五部分：期限结构与展期收益")
    print("=" * 60)

    print("""
    Contango（远期升水）：远月 > 近月 → 多头展期"卖便宜买贵"，持续损失展期收益
    Backwardation（远期贴水）：远月 < 近月 → 多头展期"卖贵买便宜"，获得展期收益
    """)
    raw = get_contracts(symbol)
    last = raw["date"].max()
    snap = raw[(raw["date"] == last) & (raw["hold"] > 0)].sort_values("contract")
    snap = snap.assign(**{"相对最近月": snap["close"] / snap["close"].iloc[0] - 1})
    print(f"📊 {SPECS[symbol]['name']} 期限结构（{last.date()}）：")
    print(snap[["contract", "close", "hold", "相对最近月"]].rename(
        columns={"contract": "合约", "close": "收盘价", "hold": "持仓量"}).to_string(
        index=False, formatters={"相对最近月": "{:+.2%}".format, "持仓量": "{:,.0f}".format}))
    slope = snap["close"].iloc[-1] - snap["close"].iloc[0]
    print(f"   → 最远月 - 最近月 = {slope:+.0f}，当前呈 {'Contango（远月升水）' if slope > 0 else 'Backwardation（远月贴水）'}")

    print("""
    把期货总收益拆开：比例调整收益 ≈ 主力价格变化 + 展期收益
       展期收益 = -Σ ln(换月前一日 新合约价 / 旧合约价)  —— 新合约越贵（Contango），多头展期损失越大
    """)
    rows = []
    for s in UNIVERSE:
        c = get_continuous(s)
        total = np.log(c["close"].iloc[-1] / c["close"].iloc[0])
        price = np.log(c["raw_close"].iloc[-1] / c["raw_close"].iloc[0])
        roll_y = -np.log(c["roll_ratio"].iloc[1:]).sum()
        years = len(c) / 252
        rows.append([s, SPECS[s]["name"], price, roll_y, total, roll_y / years])
    t = pd.DataFrame(rows, columns=["代码", "品种", "主力价格变化(对数)", "展期收益(对数)", "多头总收益(对数)", "年化展期收益"])
    print(t.to_string(index=False, formatters={c: "{:+.1%}".format for c in t.columns[2:]}))
    print("""
    💡 读表：两列相加 = 多头总收益（对数收益可加）。展期收益为正的品种，历史上多数时间远月贴水，
       多头"不涨不跌也能赚"；为负则相反。这部分收益只有用比例调整连续价才能正确反映。
       期限结构本身就是著名的 Carry 因子：做多贴水最深、做空升水最深的品种。
    """)


# ==================== 第六部分：图表 ====================


def plot_continuous(cont: pd.DataFrame, sina: pd.DataFrame, symbol: str = "RB"):
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    ax.plot(sina.index, sina["close"], lw=0.8, color="gray", label="新浪主力连续（未调整）")
    ax.plot(cont.index, cont["close_diff"], lw=0.8, label="价差调整")
    ax.plot(cont.index, cont["close"], lw=0.8, label="比例调整")
    for d in cont.index[cont["roll"]]:
        ax.axvline(d, color="red", lw=0.3, alpha=0.4)
    ax.set_title(f"{SPECS[symbol]['name']}：三种连续合约（红线 = 本课规则的换月日）")
    ax.legend()
    ax = axes[1]
    ax.plot(cont.index, (1 + cont["raw_close"].pct_change().fillna(0)).cumprod(), label="未调整拼接（含虚假跳空）")
    ax.plot(cont.index, cont["close"] / cont["close"].iloc[0], label="比例调整（真实持仓收益）")
    ax.set_title("1 倍做多的净值")
    ax.legend()
    fig.tight_layout()
    path = save_figure(fig, "06_futures_rollover.png", subdir="phase-5")
    print(f"🖼️  图表已保存：{path}")


# ==================== 主函数 ====================


def main():
    contract_specs()
    mark_to_market("RB")
    cont, sina = main_contract_selection("RB", 2024)
    back_adjustment(cont, sina, "RB")
    term_structure("CU")
    plot_continuous(cont, sina, "RB")

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 在 select_main 中加入"连续 3 天持仓量领先才切换"的确认条件，换月次数和换月日如何变化？
    2. 用价差调整价和比例调整价分别计算 20 日 ATR，在样本最早一年两者相差多少？这对 ATR 止损有什么影响？
    3. 假设保证金比例从 10% 上调到 15%，第二部分的账户哪一天会触发追保？
    4. 用第五部分的"年化展期收益"做一个简单的 Carry 策略：做多展期收益最高的 3 个品种、做空最低的 3 个
    """)


if __name__ == "__main__":
    main()
