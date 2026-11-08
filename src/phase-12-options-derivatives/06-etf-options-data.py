"""
50ETF / 300ETF 期权数据
=======================

学习目标：
    1. 了解国内场内 ETF 期权的品种与合约要素（合约单位、行权价间距、到期日规则）
    2. 探测 AKShare 中可用的期权接口，拉取上证 50ETF 期权的整条期权链
    3. 按"过滤 → 隐含远期 → 求 IV → 虚值拼接"的流程计算整条链的隐含波动率
    4. 与上交所官方公布的隐含波动率对比，画出微笑曲线与期限结构
    5. 认识盘口买卖价差对 IV 的影响，以及卖方保证金的计算思路

核心概念：
    - 合约代码：510050C2612M02800 = 标的 510050、认购、2026 年 12 月到期、标准合约、行权价 2.800
    - 隐含远期 F = K + (C - P)·e^(rT)：自动包含分红与融资成本预期
    - 只用虚值一侧（OTM）拼接微笑：K < F 用认沽，K ≥ F 用认购

前置知识：
    - 04-implied-volatility.py、05-volatility-surface.py
    - CONCEPT.md 第 8 节

环境要求：
    - akshare（需要网络）；QTC_OFFLINE=1 时使用最近一次缓存的快照，没有缓存则使用合成期权链
    - 数据缓存在 data/cache/option_*.parquet，同一天重复运行不会重复请求

⚠️ 合约单位、行权价间距、到期日、保证金等规则以交易所最新规则为准。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from option_data import (RISK_FREE, chain_implied_vols, estimate_forward, fourth_wednesday, get_contract_quote,
                         get_option_chain, get_sse_risk_indicator, parse_code, strike_step)
from option_pricing import bs_greeks, implied_vol
from utils.data import is_offline
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

UNDERLYING = "510050"
R = RISK_FREE
MULTIPLIER = 10000  # 标准合约的合约单位（以交易所最新规则为准；分红调整后的合约不同）


# ==================== 第一部分：品种与合约要素 ====================


def contract_specs():
    """国内 ETF 期权概览与合约要素"""
    print("=" * 60)
    print("第一部分：国内 ETF 期权品种与合约要素")
    print("=" * 60)

    print("""
    ┌──────────────────┬────────┬────────────────┬────────────────────────┐
    │ 品种             │ 交易所 │ 标的代码       │ 上市时间               │
    ├──────────────────┼────────┼────────────────┼────────────────────────┤
    │ 上证 50ETF 期权  │ 上交所 │ 510050         │ 2015-02-09（首个）     │
    │ 沪深 300ETF 期权 │ 上交所 │ 510300         │ 2019-12-23             │
    │ 沪深 300ETF 期权 │ 深交所 │ 159919         │ 2019-12-23             │
    │ 中证 500ETF 期权 │ 沪/深  │ 510500/159922  │ 2022-09-19             │
    │ 创业板 ETF 期权  │ 深交所 │ 159915         │ 2022-09-19             │
    │ 科创 50ETF 期权  │ 上交所 │ 588000/588080  │ 2023-06-05             │
    └──────────────────┴────────┴────────────────┴────────────────────────┘

    上交所 ETF 期权合约要素（以交易所最新规则为准）：
      - 欧式、实物交割（交割 ETF 份额）；T+0 交易，可卖出开仓（需保证金）
      - 合约单位：一般为 10000 份；ETF 分红后会调整合约单位和行权价
      - 到期月份：当月、下月及随后两个季月（共 4 个）
      - 到期日：到期月份的第四个星期三（遇法定节假日顺延）
      - 最小报价单位：0.0001 元
      - 行权价间距按标的价格分档；标的价格变动后会加挂新行权价
    """)

    print("行权价间距示例（strike_step，以交易所最新规则为准）：")
    for price in [2.5, 3.0, 4.0, 6.0, 15.0]:
        print(f"   标的价格 {price:>5.1f} 元 → 间距 {strike_step(price)} 元")

    print("\n到期日规则示例（第四个星期三，未考虑节假日顺延）：")
    print("   " + "，".join(f"{y}-{m:02d} → {fourth_wednesday(y, m):%m-%d}" for y, m in
                          [(2026, 11), (2026, 12), (2027, 3), (2027, 6)]))

    print("\n合约代码解析：")
    for code in ["510050C2612M02800", "510300P2703M04500", "510050C2606A02750"]:
        info = parse_code(code)
        adj = "分红调整后的合约（合约单位 ≠ 10000）" if info["adjusted"] else "标准合约"
        print(f"   {code}：标的 {info['underlying']}，{'认购' if info['kind'] == 'call' else '认沽'}，"
              f"20{info['yymm'][:2]} 年 {info['yymm'][2:]} 月到期，{adj}，行权价 {info['strike_in_code']:.3f}")
    print("   ⚠️ 分红调整后的合约行权价和合约单位都会变，数据处理时不能假设合约单位恒为 10000")


# ==================== 第二部分：AKShare 接口探测 ====================


def probe_interfaces():
    """检查已安装的 AKShare 中有哪些期权接口"""
    print("\n" + "=" * 60)
    print("第二部分：AKShare 期权接口探测")
    print("=" * 60)

    import akshare as ak

    candidates = {
        "option_sse_list_sina": "到期月份列表",
        "option_sse_expire_day_sina": "到期日与剩余天数",
        "option_sse_codes_sina": "某月份的合约编号列表",
        "option_sse_spot_price_sina": "单个合约的五档盘口",
        "option_sse_underlying_spot_price_sina": "标的 ETF 行情",
        "option_sse_daily_sina": "单个合约的日线历史",
        "option_finance_board": "某到期月份的整条期权链（一次请求）",
        "option_risk_indicator_sse": "上交所官方希腊字母与 IV",
        "option_current_em": "东方财富全市场期权行情",
    }
    used = {"option_sse_list_sina", "option_sse_expire_day_sina", "option_sse_spot_price_sina",
            "option_sse_underlying_spot_price_sina", "option_finance_board", "option_risk_indicator_sse"}
    print(f"AKShare 版本 {ak.__version__}：")
    for name, desc in candidates.items():
        exists = "✅" if hasattr(ak, name) else "❌"
        mark = "  ← 本课使用" if name in used else ""
        print(f"   {exists} {name:<40}{desc}{mark}")
    print("""
    💡 本课的组合：option_finance_board 一次请求拿到一个到期月份的整条链（最新价、前结算价），
       比逐个合约请求 option_sse_spot_price_sina 少几十倍请求量；盘口只对少数合约单独请求。
       AKShare 的函数名和返回格式会随版本变化，接口失效时先查官方文档。
    """)


# ==================== 第三部分：拉取期权链 ====================


def load_chain():
    """拉取并概览期权链"""
    print("\n" + "=" * 60)
    print("第三部分：拉取上证 50ETF 期权链")
    print("=" * 60)

    chain, meta = get_option_chain(UNDERLYING)
    tag = {"real": "真实行情", "cache": "缓存的真实行情", "synthetic": "合成数据"}[meta["source"]]
    print(f"   {meta['name']} 期权链：{len(chain)} 个合约，{tag}，快照 {meta['snapshot']:%Y-%m-%d %H:%M}，"
          f"标的价 {meta['spot']:.3f}")
    print("   （收盘后的快照中，期权价格为最新成交价；盘中运行时期权与标的的时间戳可能不完全同步）")

    summary = chain.groupby("expiry").agg(
        剩余天数=("days", "first"), 合约数=("code", "size"), 最低行权价=("strike", "min"),
        最高行权价=("strike", "max"), 调整合约数=("adjusted", "sum"))
    print(summary.to_string())

    near = chain[chain["expiry"] == chain["expiry"].min()]
    wide = near.pivot_table(index="strike", columns="kind", values="price")
    print(f"\n   最近到期月（{near['expiry'].iloc[0]:%Y-%m-%d}）T 型报价（最新价）：")
    print(wide.rename(columns={"call": "认购", "put": "认沽"}).to_string(float_format=lambda x: f"{x:.4f}"))
    strikes = np.sort(near["strike"].unique())
    steps = sorted(set(np.round(np.diff(strikes), 3).tolist()))
    print(f"   行权价间距：{steps}（与标的价格分档规则对照）")
    return chain, meta


# ==================== 第四部分：整条链的隐含波动率 ====================


def chain_iv(chain: pd.DataFrame, meta: dict):
    """隐含远期 → IV → 与上交所官方 IV 对比"""
    print("\n" + "=" * 60)
    print("第四部分：计算整条期权链的隐含波动率")
    print("=" * 60)

    print(f"""
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. 过滤：剩余期限 < 3 天、价格 < 0.0005（5 个 tick）的合约    │
    │ 2. 隐含远期：取 |C - P| 最小的 3 个行权价，F = K + (C-P)e^(rT)│
    │ 3. 求 IV：用 S_eff = F·e^(-rT) 代入 BS（r = {R:.1%}，示例值）   │
    │ 4. 拼接：K < F 用认沽，K ≥ F 用认购                           │
    └──────────────────────────────────────────────────────────────┘
    """)
    rows = []
    for expiry, sub in chain.groupby("expiry"):
        T = sub["days"].iloc[0] / 365
        F = estimate_forward(sub, R, T)
        implied_q = -np.log(F / meta["spot"]) / T + R  # F = S·e^{(r-q)T} 反推的"隐含股息率"
        rows.append([expiry.date(), sub["days"].iloc[0], F, F - meta["spot"], implied_q])
    print(pd.DataFrame(rows, columns=["到期日", "天数", "隐含远期 F", "F - 现价", "隐含股息率 q"])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("   隐含股息率 = 让 F = S·e^((r-q)T) 成立的 q；包含了预期分红，也包含卖空约束等因素")

    ivs = chain_implied_vols(chain, r=R)
    # 对照组：直接用 ETF 现价、不做分红调整（很多行情软件的口径）
    ivs["iv_spot"] = [implied_vol(p, meta["spot"], k, t, R, kd)
                      for p, k, t, kd in zip(ivs["price"], ivs["strike"], ivs["T"], ivs["kind"])]
    bad = ivs[ivs["iv"].isna()]
    print(f"\n   通过过滤的合约 {len(ivs)} 个，IV 无解（价格违反无套利边界）{len(bad)} 个" + ("：" if len(bad) else ""))
    for _, row in bad.iterrows():
        disc_k, disc_f = row["strike"] * np.exp(-R * row["T"]), row["F"] * np.exp(-R * row["T"])
        lower = max(disc_f - disc_k, 0) if row["kind"] == "call" else max(disc_k - disc_f, 0)
        print(f"     {row['expiry']:%Y-%m} {row['kind']} K={row['strike']}：最新价 {row['price']:.4f} ≤ 无套利下界 "
              f"{lower:.4f}（多见于成交稀少的深度实值合约，最新价可能是很久之前的成交）")

    # 同一行权价：认购 IV vs 认沽 IV
    print("\n   同一行权价认购 IV − 认沽 IV（波动率点）：")
    gaps = {}
    for col, label in [("iv_spot", "用现价 S、不调分红"), ("iv", "用隐含远期 F")]:
        both = ivs.pivot_table(index=["expiry", "strike"], columns="kind", values=col).dropna()
        gap = (both["call"] - both["put"]) * 100
        gaps[col] = gap.median()
        print(f"     {label}：中位数 {gap.median():+.2f} 点，绝对值中位数 {gap.abs().median():.2f} 点（{len(both)} 对）")
    side = "低于" if gaps["iv_spot"] < 0 else "高于"
    print(f"""   💡 用现价算时，认购 IV 中位数{side}认沽 IV {abs(gaps['iv_spot']):.2f} 点。原因不是市场"错了"，
      而是现价口径忽略了远期与现价的差异（预期分红让远期低于现价 → 认购更便宜、认沽更贵）。
      改用隐含远期后，平价关系被"吸收"，剩余差异来自成交价不同步与报价精度。""")

    if meta["source"] == "synthetic":
        print("\n   ⚠️ 使用合成数据，跳过与上交所官方 IV 的对比")
        return ivs
    risk = get_sse_risk_indicator(f"{meta['snapshot']:%Y%m%d}")
    if risk is None or risk.empty:
        print("\n   ⚠️ 未取得上交所风险指标（离线且无缓存），跳过对比")
        return ivs
    official = risk[risk["CONTRACT_ID"].str.startswith(UNDERLYING)][["CONTRACT_ID", "IMPLC_VOLATLTY"]]
    merged = chain[["code", "kind", "expiry", "strike"]].merge(official, left_on="code", right_on="CONTRACT_ID")
    merged = merged.merge(ivs[["expiry", "strike", "kind", "iv", "iv_spot"]], on=["expiry", "strike", "kind"])
    merged = merged[(merged["IMPLC_VOLATLTY"] > 0) & merged["iv"].notna() & merged["iv_spot"].notna()]
    print(f"\n   与上交所官方公布的 IV 对比（{len(merged)} 个官方 IV > 0 的合约）：")
    for col, label in [("iv_spot", "本课·现价口径"), ("iv", "本课·远期口径")]:
        diff = (merged[col] - merged["IMPLC_VOLATLTY"]) * 100
        print(f"     {label}：绝对差中位数 {diff.abs().median():.2f} 点，"
              f"相关系数 {merged[col].corr(merged['IMPLC_VOLATLTY']):.3f}")
    err = {col: ((merged[col] - merged["IMPLC_VOLATLTY"]) * 100).abs().median() for col in ("iv_spot", "iv")}
    closer = "现价口径" if err["iv_spot"] < err["iv"] else "远期口径"
    print(f"""   官方 IV 与"{closer}"更接近。官方计算口径（价格类型、标的价格、利率、期限计法、是否调整分红）
      未完全公开，拿第三方 IV 做研究前，先用这种方法反推它的口径。做波动率研究时推荐远期口径。""")
    return ivs


# ==================== 第五部分：微笑曲线、期限结构与希腊字母 ====================


def plot_smile(ivs: pd.DataFrame, meta: dict):
    """画微笑曲线与期限结构，汇总近月希腊字母"""
    print("\n" + "=" * 60)
    print("第五部分：微笑曲线、期限结构与希腊字母")
    print("=" * 60)

    otm = ivs[ivs["otm"] & ivs["iv"].notna()]
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, otm["expiry"].nunique()))
    for (expiry, sub), color in zip(otm.groupby("expiry"), colors):
        axes[0].plot(sub["strike"], sub["iv"] * 100, "o-", color=color, ms=4,
                     label=f"{expiry:%Y-%m-%d}（{sub['days'].iloc[0]}天）")
        axes[0].axvline(sub["F"].iloc[0], color=color, ls=":", lw=0.8)
    axes[0].set_xlabel("行权价 K（虚线 = 各到期日隐含远期）")
    axes[0].set_ylabel("隐含波动率（%）")
    src = "合成数据" if meta["source"] == "synthetic" else "真实行情"
    axes[0].set_title(f"{meta['name']} 期权微笑（虚值拼接，{src}）")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    near = ivs[ivs["expiry"] == ivs["expiry"].min()].dropna(subset=["iv"])
    for kind, marker in [("call", "o"), ("put", "s")]:
        sub = near[near["kind"] == kind]
        axes[1].plot(sub["strike"], sub["iv"] * 100, marker=marker, ls="-", ms=4, label="认购" if kind == "call" else "认沽")
    axes[1].set_xlabel("行权价 K")
    axes[1].set_title(f"近月认购 vs 认沽 IV（{near['days'].iloc[0]} 天）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    atm = otm.loc[otm.groupby("expiry")["k"].apply(lambda s: s.abs().idxmin())]
    axes[2].plot(atm["days"], atm["iv"] * 100, "o-")
    axes[2].set_xlabel("剩余天数")
    axes[2].set_ylabel("平值附近 IV（%）")
    axes[2].set_title("期限结构")
    axes[2].grid(alpha=0.3)
    path = save_figure(fig, "06_etf_option_smile.png", subdir="phase-12")
    print("   平值附近 IV 期限结构：" + "，".join(f"{d}天 {v:.2%}" for d, v in zip(atm["days"], atm["iv"])))

    # 近月虚值合约的希腊字母（每张合约，按标准合约单位）
    sub = otm[otm["expiry"] == otm["expiry"].min()].copy()
    T, F = sub["T"].iloc[0], sub["F"].iloc[0]
    g = bs_greeks(F * np.exp(-R * T), sub["strike"].to_numpy(), T, R, sub["iv"].to_numpy(), sub["kind"].to_numpy())
    sub["Delta"] = g["delta"]
    sub["Gamma"] = g["gamma"]
    sub["Vega/点/张"] = g["vega"] / 100 * MULTIPLIER
    sub["Theta/天/张"] = g["theta"] / 365 * MULTIPLIER
    print(f"\n   近月（{sub['days'].iloc[0]} 天）虚值合约的希腊字母：")
    print(sub[["kind", "strike", "price", "iv", "Delta", "Gamma", "Vega/点/张", "Theta/天/张"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n📈 图已保存：{path}")


# ==================== 第六部分：盘口与保证金 ====================


def quotes_and_margin(chain: pd.DataFrame, ivs: pd.DataFrame, meta: dict):
    """平值合约的买卖价差 → IV 区间；卖出开仓保证金"""
    print("\n" + "=" * 60)
    print("第六部分：盘口价差与卖方保证金")
    print("=" * 60)

    near = ivs[(ivs["expiry"] == ivs["expiry"].min()) & ivs["iv"].notna()]
    T, F = near["T"].iloc[0], near["F"].iloc[0]
    atm_k = near.loc[(near["strike"] - F).abs().idxmin(), "strike"]
    s_eff = F * np.exp(-R * T)

    risk = None if meta["source"] == "synthetic" else get_sse_risk_indicator(f"{meta['snapshot']:%Y%m%d}", verbose=False)
    if risk is None:
        msg = "合成数据" if meta["source"] == "synthetic" else ("离线且无缓存" if is_offline() else "风险指标不可用")
        print(f"   ⚠️ {msg}：跳过盘口查询，下面用假设的价差演示")
        quotes = {}
        for kind in ("call", "put"):
            px = near.loc[(near["strike"] == atm_k) & (near["kind"] == kind), "price"].iloc[0]
            quotes[kind] = {"买价": px - 0.0005, "卖价": px + 0.0005, "最新价": px}
    else:
        codes = chain[(chain["expiry"] == near["expiry"].iloc[0]) & (chain["strike"] == atm_k)]
        id_map = dict(zip(risk["CONTRACT_ID"], risk["SECURITY_ID"]))
        quotes = {}
        for _, row in codes.iterrows():
            q = get_contract_quote(id_map.get(row["code"], ""), verbose=True) if row["code"] in id_map else None
            if q:
                quotes[row["kind"]] = {"买价": float(q["买价"]), "卖价": float(q["卖价"]), "最新价": float(q["最新价"])}
    rows = []
    for kind, q in quotes.items():
        mid = 0.5 * (q["买价"] + q["卖价"])
        ivx = {name: implied_vol(px, s_eff, atm_k, T, R, kind) for name, px in [("买价", q["买价"]), ("中间价", mid),
                                                                              ("卖价", q["卖价"]), ("最新价", q["最新价"])]}
        rows.append([kind, atm_k, q["买价"], q["卖价"], q["卖价"] - q["买价"], *[ivx[n] for n in ["买价", "中间价", "卖价", "最新价"]]])
    if rows:
        df = pd.DataFrame(rows, columns=["类型", "K", "买价", "卖价", "价差", "买价IV", "中间价IV", "卖价IV", "最新价IV"])
        print(f"   近月平值合约（K={atm_k}）盘口：")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("   价差换算成 IV 区间后，就能看出\"一个 tick\"对波动率交易意味着多少成本")
    else:
        print("   ⚠️ 盘口数据不可用，跳过")

    # 卖出开仓保证金（上交所认购义务仓开仓保证金公式，以交易所与券商最新标准为准）
    call_px = near.loc[(near["strike"] == atm_k) & (near["kind"] == "call"), "price"].iloc[0]
    S = meta["spot"]
    rows = []
    for k in sorted(near["strike"].unique()):
        sub = near[(near["strike"] == k) & (near["kind"] == "call")]
        if sub.empty:
            continue
        px = sub["price"].iloc[0]
        otm_amt = max(k - S, 0)
        margin = (px + max(0.12 * S - otm_amt, 0.07 * S)) * MULTIPLIER
        rows.append([k, px, otm_amt, margin, px * MULTIPLIER, px * MULTIPLIER / margin])
    df = pd.DataFrame(rows, columns=["K", "认购价", "虚值额", "开仓保证金/张", "权利金/张", "权利金/保证金"])
    df["开仓保证金/张"] = df["开仓保证金/张"].round(0)
    df["权利金/张"] = df["权利金/张"].round(0)
    df["权利金/保证金"] = df["权利金/保证金"].map(lambda x: f"{x:.1%}")
    print("""
    认购义务仓开仓保证金（每张）≈ [合约前结算价 + max(12%×标的前收 − 认购虚值额, 7%×标的前收)] × 合约单位
    （下表用最新价近似前结算价、用现价近似标的前收；以交易所与券商最新标准为准）""")
    print(df.to_string(index=False, float_format=lambda x: f"{x:g}"))
    print(f"   平值认购每张权利金约 {call_px * MULTIPLIER:,.0f} 元，保证金是权利金的数倍 → 卖方收益率看起来不高，"
          "且行情不利时保证金还会上升")


# ==================== 主函数 ====================


def main():
    contract_specs()
    probe_interfaces()
    chain, meta = load_chain()
    ivs = chain_iv(chain, meta)
    plot_smile(ivs, meta)
    quotes_and_margin(chain, ivs, meta)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 UNDERLYING 改成 "510300"，对比 50ETF 与 300ETF 期权的平值 IV、偏斜与隐含股息率
    2. 用 option_sse_daily_sina 下载一个平值合约的日线，画出它临近到期时的价格衰减（注意请求量与缓存）
    3. 把无风险利率从 1.5% 改为 0% 和 3%，隐含远期 F 和 IV 各变化多少？为什么 IV 几乎不变？
    4. 用最近一次快照计算全链的组合希腊字母：假设卖出 10 张近月平值宽跨式，Delta、Gamma、Vega 是多少？
    """)


if __name__ == "__main__":
    main()
