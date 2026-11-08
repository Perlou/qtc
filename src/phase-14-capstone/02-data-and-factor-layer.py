"""
数据层与因子层实现
==================

学习目标：
    1. 数据层：用 PanelDataSource 封装 Phase 7 截面面板，跑一遍数据质量检查
    2. 用实际数据演示 Point-in-Time 的重要性：财报按"报告期"对齐 vs 按"公告日"对齐
    3. 因子注册表：用装饰器注册新因子；重名保护；配置文件决定启用哪些因子
    4. 截面预处理：MAD 去极值 → Z-score → 行业 + 市值中性化，逐步观察每一步的效果
    5. 单因子快速诊断：Rank IC、ICIR、t 值、分层多空，并按样本内 / 样本外拆开看稳定性

核心概念：
    - 注册表模式：因子是插件，新增因子不改平台代码
    - 因子方向统一为"越大越好"，IC 应为正
    - 诊断 ≠ 选因子：一次看 11 个因子就是 11 次检验，要警惕多重检验

前置知识：
    - Phase 7（因子构建、预处理、IC、PIT 财务）
    - 本阶段 CONCEPT.md 第 3 节

环境要求：
    - 无网络时可用 QTC_OFFLINE=1 运行（先读本地缓存，缓存不存在才用合成数据）
"""

import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE_DIR.parent))
sys.path.insert(0, str(PHASE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy import stats  # noqa: E402

import mini_quant  # noqa: E402
from mini_quant import load_config  # noqa: E402
from mini_quant.data import PanelDataSource  # noqa: E402
from mini_quant.factors import (  # noqa: E402
    REGISTRY,
    FactorContext,
    RegistryFactorEngine,
    diagnose,
    factor,
    ic_series,
    ic_table,
    preprocess_cross_section,
)

import factor_tools  # noqa: E402  （Phase 7，经 mini_quant 加入路径）
from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 160)


# ==================== 第一部分：数据层 ====================


def data_layer(src: PanelDataSource):
    print("=" * 60)
    print("第一部分：数据层——封装而不是重写")
    print("=" * 60)
    p = src.panel
    print(f"""
    PanelDataSource 是 Phase 7 截面面板的薄适配器，实现 DataSource 协议：
      update(end)          → 返回 ≤ end 的最新交易日（实盘在这里做增量下载 + 重试）
      load_prices(s, e)    → 不复权收盘价（计算股数、成交金额用）
      rebalance_dates()    → 月末调仓日（factor_tools.rebalance_dates）
      tradable(dates)      → 上市满一年 / 未停牌 / 非 ST / 次日开盘未涨停（factor_tools.tradable_status）
      period_returns(dates)→ t+1 开盘买入 → 下一调仓日次日开盘卖出（factor_tools.period_returns）

    数据：{src.label()}
    股票池 {len(p.codes)} 只（{p.info['sector'].nunique()} 个行业，板块：{p.info['board'].value_counts().to_dict()}）
    行情 {p.dates[0].date()} ~ {p.dates[-1].date()}，{len(p.dates)} 个交易日；
    财报 {len(p.fin)} 条（含首次公告日 ann_date）
    """)
    print("    🔍 数据质量检查（2019–2024）：")
    print(src.quality_report("2019-01-01", "2024-12-31").to_string(index=False))
    print("""
    "异常跳变"如果被标出：逐条核对（停牌复牌、更名换码、板块规则识别错误都可能触发），
    确认是真实行情还是数据错误。本平台在核对中发现 bt_core.board_rule 把创业板新代码段 302xxx
    识别为主板（10%），因此在 mini_quant/rules.py 中做了适配，上表已使用适配后的规则。

    ⚠️ 已知偏差（必须写进研究报告）：股票池是"当前"沪深 300 + 中证 500 成分股，
       回测过去存在幸存者偏差和成分股前视偏差；行业是当前分类（Phase 7 第 03 课）。
    """)


def pit_demo(src: PanelDataSource, dates: pd.DatetimeIndex, fwd: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第二部分：Point-in-Time——财报按什么日期对齐？")
    print("=" * 60)
    if src.is_synthetic:
        print("    合成数据的财报公告日是模拟的，下面的对比只演示流程")
    mask = src.tradable(dates)["all"]
    rows = []
    for how, label in [("ann", "按实际公告日（正确）"), ("report", "按报告期（前视偏差！）")]:
        raw = factor_tools.build_factors(src.panel, dates, how=how)
        for name in ("ep", "roe"):
            clean = factor_tools.preprocess(raw[name], src.panel, mask)
            s = factor_tools.ic_summary(factor_tools.calc_ic(clean, fwd))
            rows.append([label, name, s["IC均值"], s["ICIR"], s["t值"]])
    df = pd.DataFrame(rows, columns=["对齐方式", "因子", "IC均值", "ICIR", "t值"])
    print(df.round(4).to_string(index=False))
    gap = df.pivot(index="因子", columns="对齐方式", values="IC均值")
    print(f"""
    按报告期对齐 = 报告期一结束（如 3 月 31 日）就"知道"了一季报，但实际要到 4 月底才公布。
    上表中两种口径的 IC 差异：{', '.join(f'{k} {v:+.4f}' for k, v in (gap.iloc[:, 1] - gap.iloc[:, 0]).items())}
    （报告期口径 − 公告日口径）。差异的方向和大小取决于数据，但只有公告日口径才是回测能用的。
    平台约定：数据层只提供 PIT 数据，因子层永远拿不到"未来的财报"。
    """)


# ==================== 第三部分：因子注册表 ====================


def registry_demo():
    print("\n" + "=" * 60)
    print("第三部分：因子注册表——用装饰器注册插件")
    print("=" * 60)
    print("    已注册的因子（Phase 7 的 8 个因子直接复用 build_factors，另有 2 个日频价量因子）：")
    print(REGISTRY.table().to_string(index=False))

    print("""
    新增一个因子只需要写一个函数并加上装饰器（不改平台任何代码）：

        @factor("illiq_20", category="流动性", description="Amihud 非流动性 20D")
        def amihud_illiquidity(ctx):
            illiq = ctx.panel.daily_ret.abs() / ctx.panel.amount   # 每元成交额带来的价格变动
            return np.log(ctx.at(illiq.rolling(20, min_periods=15).mean()) * 1e9)
    """)
    if "illiq_20" not in REGISTRY:

        @factor("illiq_20", category="流动性", description="Amihud 非流动性 20D（取对数）", source="02 课新增")
        def amihud_illiquidity(ctx: FactorContext) -> pd.DataFrame:
            illiq = ctx.panel.daily_ret.abs() / ctx.panel.amount.replace(0, np.nan)
            return np.log(ctx.at(illiq.rolling(20, min_periods=15).mean()) * 1e9)

    print(f"    ✅ 注册成功：现在共有 {len(REGISTRY)} 个因子：{REGISTRY.names()}")
    try:
        factor("ep")(lambda ctx: None)
    except ValueError as exc:
        print(f"    🛡️ 重名保护：再次注册 ep → ValueError：{exc}")
    try:
        RegistryFactorEngine(None, ["ep", "not_exist"])
    except KeyError as exc:
        print(f"    🛡️ 配置校验：启用未注册的因子 → KeyError：{exc}")
    cfg = load_config()
    print(f"\n    config/strategy.yaml 当前启用：{list(cfg.factors)}（实验新因子时改配置，不改代码）")


# ==================== 第四部分：截面预处理 ====================


def preprocess_demo(src: PanelDataSource, dates: pd.DatetimeIndex):
    print("\n" + "=" * 60)
    print("第四部分：截面预处理——MAD 去极值 → Z-score → 行业 + 市值中性化")
    print("=" * 60)
    d = dates[-1]
    ctx = FactorContext(src.panel, pd.DatetimeIndex([d]))
    mask = src.tradable(dates)["all"].loc[d]
    sector, lnmv = src.sector, src.log_mcap.loc[d]
    rows = []
    for name in ("bp", "illiq_20"):
        raw = REGISTRY.compute(name, ctx).loc[d].where(mask)
        steps = preprocess_cross_section(raw, sector, lnmv)
        for step, x in steps.items():
            idx = x.dropna().index
            sec_eta = x.groupby(sector.reindex(idx)).mean().var() / x.var() if x.var() > 0 else np.nan
            rows.append([name, step, len(idx), x.skew(), (x - x.mean()).abs().max() / x.std(), sec_eta,
                         np.corrcoef(x.loc[idx], lnmv.loc[idx])[0, 1]])
    df = pd.DataFrame(rows, columns=["因子", "步骤", "样本", "偏度", "最大偏离(σ)", "行业均值离散度", "与对数市值相关"])
    print(f"    截面日期 {d.date()}（只在当日可交易的股票上做截面）：")
    print(df.round(3).to_string(index=False))
    print("""
    读表：
      winsorized   极端值被截断到 中位数 ± 3 × 1.4826 × MAD → 偏度与"最大偏离(σ)"下降
      zscore       只是平移缩放：偏度、最大偏离、相关性都不变
      neutralized  对行业哑变量 + 对数市值回归取残差 → 与市值相关 ≈ 0、行业均值离散度 ≈ 0
    illiq_20（非流动性）天然与市值强负相关：不做市值中性化，它基本就是"小市值因子"的复制品。
    全部计算调用 Phase 7 的 winsorize_mad / zscore / neutralize，口径与 Phase 7 一致。
    """)


# ==================== 第五部分：单因子快速诊断 ====================


def ic_diagnosis(src: PanelDataSource, dates: pd.DatetimeIndex, fwd: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第五部分：单因子快速诊断")
    print("=" * 60)
    cfg = load_config()
    engine = RegistryFactorEngine(src, REGISTRY.names(), neutralize=cfg.neutralize)
    clean = engine.panel(dates)
    table = ic_table(clean, fwd)
    table["结论"] = diagnose(table)
    ics = ic_series(clean, fwd)
    is_ic = ics.loc[cfg.is_start : cfg.is_end].mean()
    oos_ic = ics.loc[cfg.oos_start : cfg.oos_end].mean()
    table["样本内IC"], table["样本外IC"] = is_ic, oos_ic
    cols = ["IC均值", "ICIR", "t值", "IC>0占比", "期数", "多空月均", "单调性", "样本内IC", "样本外IC", "结论"]
    show = table[cols].sort_values("t值", ascending=False)
    fmt = show.copy()
    for c in ["IC均值", "ICIR", "t值", "IC>0占比", "多空月均", "单调性", "样本内IC", "样本外IC"]:
        fmt[c] = show[c].astype(float).round(3)
    fmt["期数"] = show["期数"].astype(int)
    print(f"    {dates[0].date()} ~ {dates[-1].date()} 月频，{len(clean)} 个因子（预处理 + 行业市值中性化后）：")
    print(fmt.to_string())
    n_sig = int((show["t值"].abs() >= 2).sum())
    n_flip = int((np.sign(is_ic) != np.sign(oos_ic)).sum())
    p_any = 1 - (1 - 0.05) ** len(clean)
    print(f"""
    读表：
      |t| ≥ 2 的因子 {n_sig} 个；样本内与样本外 IC 符号相反的因子 {n_flip} 个
      同时检验 {len(clean)} 个因子：即使全部无效，至少一个"显著"的概率也有 1 − 0.95^{len(clean)} = {p_any:.0%}
      → 诊断用于"排除明显无效 / 方向错误"的因子，不能用来在样本外反复挑因子
      多空月均 = 第 5 组 − 第 1 组的月均收益；单调性 = 各组平均收益与组号的相关系数
    """)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    order = show.index
    colors = ["#2a78d6" if v > 0 else "#eb6834" for v in show["IC均值"]]
    axes[0].barh(order[::-1], show["IC均值"][::-1], color=colors[::-1])
    axes[0].axvline(0, color="#666", lw=0.8)
    axes[0].set_title("各因子 Rank IC 均值（蓝 > 0，橙 < 0）")
    for name in order[:4]:
        axes[1].plot(ics[name].fillna(0).cumsum(), label=name)
    axes[1].axvline(pd.Timestamp(cfg.oos_start), color="#888", ls="--", lw=1)
    axes[1].text(pd.Timestamp(cfg.oos_start), axes[1].get_ylim()[1] * 0.9, " 样本外", color="#666")
    axes[1].set_title("t 值最高的 4 个因子：累计 Rank IC")
    axes[1].legend(fontsize=8)
    path = save_figure(fig, "02_factor_ic.png", subdir=mini_quant.SUBDIR)
    print(f"    📊 已保存：{path}")


def main():
    src = PanelDataSource()
    dates = src.rebalance_dates()
    fwd = src.period_returns(dates)
    data_layer(src)
    pit_demo(src, dates, fwd)
    registry_demo()
    preprocess_demo(src, dates)
    ic_diagnosis(src, dates, fwd)
    print("""
    ✅ 小结：
      1. 数据层只做封装与质量检查，PIT、可交易性、持有期收益全部复用 Phase 7
      2. 因子是插件：@factor 注册，配置启用；重名与未注册都会被拦截
      3. 预处理三步各有作用，中性化决定了因子是不是"换了名字的市值因子"
      4. 诊断表要同时看全样本、样本内、样本外，并记住多重检验
    下一课：模型层与组合层——滚动训练、TopN + 换手约束、按 100 股取整的调仓指令。
    """)


if __name__ == "__main__":
    main()
