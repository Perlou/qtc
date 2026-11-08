"""
模型层与组合层实现
==================

学习目标：
    1. 构造训练标签，并理解为什么滚动训练时标签要滞后 2 期（否则就是前视偏差）
    2. 实现滚动训练（Walk-Forward）：每月只用"当时已实现"的标签重训，并记录模型版本
    3. 对比 4 种模型：等权合成、IC 加权、Ridge、HistGradientBoosting（只看样本内）
    4. 组合层：TopN 选股 + 换手约束 + 单票上限，观察换手上限对成本与收益的影响
    5. 从目标权重生成调仓指令：按 A 股交易单位取整、先卖后买，复用 Phase 4 的规则与成本模型

核心概念：
    - 回测中每个调仓日只能使用"当时已训练好"的模型
    - 样本外只在 05 课评估一次：本课所有比较都限定在样本内
    - 取整 100 股会让实际权重偏离目标，小资金时偏离更明显

前置知识：
    - Phase 4（交易规则、成本）、Phase 7（因子合成）、Phase 8（组合构建、换手）
    - 本阶段 CONCEPT.md 第 4 节

环境要求：
    - scikit-learn（LightGBM 在本环境不可用，用 HistGradientBoostingRegressor 代替）
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

import mini_quant  # noqa: E402
from mini_quant import load_config  # noqa: E402
from mini_quant.backtest import cost_rates  # noqa: E402
from mini_quant.models import LABEL_LAG, RollingTrainer, make_model, training_dates  # noqa: E402
from mini_quant.portfolio import LotOrderPlanner, TopNPortfolio, orders_frame  # noqa: E402
from mini_quant.research import ResearchContext  # noqa: E402
from mini_quant.rules import board_rule  # noqa: E402

from bt_core import CostModel  # noqa: E402
import factor_tools  # noqa: E402
from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 160)

MODELS = ["equal", "icw", "ridge", "hgb"]
MODEL_LABELS = {"equal": "等权合成", "icw": "IC 加权", "ridge": "Ridge", "hgb": "HistGB"}


# ==================== 第一部分：标签与滚动训练 ====================


def labels_and_rolling(ctx: ResearchContext, cfg):
    print("=" * 60)
    print("第一部分：训练标签与滚动训练")
    print("=" * 60)
    dates = ctx.dates
    d = pd.Timestamp(cfg.is_start) + pd.offsets.MonthEnd(0)
    d = dates[dates >= d][0]
    tr = training_dates(dates, d, cfg.train_window)
    i = dates.get_loc(d)
    print(f"""
    标签：持有期收益（t+1 开盘 → 下一调仓日次日开盘）的截面百分位排名 − 0.5
          用排名而不是收益本身：去掉市场整体涨跌，降低极端收益的影响

    时间线（以 {d.date()} 调仓日为例）：
      {dates[i - 1].date()} 的标签 = 持有到 {d.date()} 的次日开盘 → 今天收盘时还不知道 ❌
      {dates[i - 2].date()} 的标签 = 持有到 {dates[i - 1].date()} 的次日开盘 → 已实现 ✅
    所以训练集最晚只能用到 {tr[-1].date()}（LABEL_LAG = {LABEL_LAG}），本例训练期 {tr[0].date()} ~ {tr[-1].date()}，共 {len(tr)} 期
    """)

    # 前视偏差实验：label_lag = 1（偷看一天开盘价）vs 2（正确）
    X = ctx.factor_long(cfg)
    rows = []
    for lag in (2, 1):
        trainer = RollingTrainer(lambda: make_model("ridge"), cfg.train_window, cfg.min_train, label_lag=lag)
        scores = trainer.run(X, ctx.labels, dates)
        ic = factor_tools.calc_ic(scores, ctx.period_ret).loc[cfg.is_start : cfg.is_end]
        rows.append([f"label_lag = {lag}" + ("（正确）" if lag == 2 else "（多用了一期未实现的标签）"),
                     ic.mean(), ic.mean() / ic.std(), len(trainer.versions)])
    print(pd.DataFrame(rows, columns=["设置", "样本内 Rank IC", "ICIR", "模型版本数"]).round(4).to_string(index=False))
    diff = rows[1][1] - rows[0][1]
    print(f"""
    两种设置的 IC 差 {diff:+.4f}。差异可大可小，但 label_lag = 1 的结果在实盘中无法复现：
    下单那一刻，最新一期标签对应的卖出价还没有发生。流水线 build_daily_pipeline 使用同一个
    training_dates()，保证回测与实盘的训练窗口一致。
    """)

    scores, versions = ctx.scores(cfg)
    print(f"    滚动训练记录（{cfg.model}，每 {cfg.retrain_every} 期重训，窗口 {cfg.train_window} 期）：共 {len(versions)} 个模型版本")
    cols = ["version", "fit_date", "train_start", "train_end", "n_samples", "model"]
    v = versions[cols].copy()
    for c in ["fit_date", "train_start", "train_end"]:
        v[c] = pd.to_datetime(v[c]).dt.date
    print(pd.concat([v.head(3), v.tail(2)]).to_string(index=False))
    coef_first = versions.iloc[0]["detail"].get("coef", {})
    coef_last = versions.iloc[-1]["detail"].get("coef", {})
    if coef_first:
        cf = pd.DataFrame({"首个版本": coef_first, "最新版本": coef_last}).T
        print("\n    Ridge 系数（因子权重）随时间的变化——模型会'学到'近期哪些因子有效：")
        print(cf.round(4).to_string())
    return scores


# ==================== 第二部分：模型对比（只看样本内） ====================


def compare_models(ctx: ResearchContext, cfg):
    print("\n" + "=" * 60)
    print("第二部分：四种模型对比（样本内 2020–2022）")
    print("=" * 60)
    rows, navs = [], {}
    for m in MODELS:
        c = cfg.with_(model=m)
        run = ctx.run(c)
        ic = ctx.score_ic(run.scores).loc[cfg.is_start : cfg.is_end]
        s = run.summary("is")
        rows.append([MODEL_LABELS[m], ic.mean(), ic.mean() / ic.std(), s["年化收益"], s["超额年化(几何)"],
                     s["信息比率"], s["月均单边换手"], s["年化成本"]])
        navs[MODEL_LABELS[m]] = (1 + run.net("is")).cumprod()
    bench = (1 + ctx.bench["ew_universe"].loc[cfg.is_start : cfg.is_end]).cumprod()
    df = pd.DataFrame(rows, columns=["模型", "Rank IC", "ICIR", "年化收益", "超额年化", "信息比率", "月均换手", "年化成本"])
    print(df.set_index("模型").round(3).to_string())
    print(f"""
    组合设置：Top {cfg.n_top}、等权、单票上限 {cfg.max_weight:.0%}、单边换手上限 {cfg.max_turnover:.0%}；
    基准为"股票池等权"（每月对可交易股票等权，不扣费）。
    ⚠️ 这里只是"一组参数下的样本内比较"。05 课会把模型、持仓数、换手上限组成网格，
       全部试验记入日志、只按样本内指标选一个，再在样本外评估一次，并用 DSR 校正试验次数。
    """)
    return navs, bench


# ==================== 第三部分：换手约束 ====================


def turnover_tradeoff(ctx: ResearchContext, cfg):
    print("\n" + "=" * 60)
    print("第三部分：组合层——换手上限的取舍（样本内）")
    print("=" * 60)
    buy, sell = cost_rates(CostModel(slippage=cfg.slippage), pd.Timestamp(cfg.is_end))
    print(f"""
    成本（Phase 4 CostModel）：买入 = 佣金 0.025% + 过户费 0.001% + 滑点 {cfg.slippage:.2%} = {buy:.4%}
                               卖出 = 买入 + 印花税（2023-08-28 前 0.1%，之后 0.05%，按当时费率）
    换手约束：w = 当前 + λ(目标 − 当前)，λ = min(1, 上限 / 理想换手)（CONCEPT 4.2 节）
    """)
    rows = []
    for to in (0.1, 0.2, 0.3, 0.5, 1.0):
        s = ctx.run(cfg.with_(max_turnover=to)).summary("is")
        rows.append([f"{to:.0%}", s["月均单边换手"], s["年化成本"], s["年化收益"], s["超额年化(几何)"], s["信息比率"]])
    df = pd.DataFrame(rows, columns=["换手上限", "月均单边换手", "年化成本", "年化收益", "超额年化", "信息比率"])
    print(df.set_index("换手上限").round(4).to_string())
    best = df.loc[df["信息比率"].idxmax(), "换手上限"]
    print(f"""
    读表：上限越紧，成本越低，但组合跟不上最新信号（信号"打折"执行）；上限过松，成本吃掉超额。
    本数据下样本内信息比率最高的是 {best}；50% 与 100% 几乎相同，说明理想换手本来就很少超过 50%。
    最优值取决于信号衰减速度和成本，需要在样本内选择，并在 06 课检查相邻取值是否同样可用（参数平原）。
    """)
    return df


# ==================== 第四部分：从权重到调仓指令 ====================


def orders_demo(ctx: ResearchContext, cfg, scores: pd.DataFrame):
    print("\n" + "=" * 60)
    print("第四部分：从目标权重到调仓指令（按交易单位取整）")
    print("=" * 60)
    src = ctx.source
    dates = scores.index
    d0, d1 = dates[-2], dates[-1]
    port = TopNPortfolio(cfg.n_top, cfg.max_weight, cfg.max_turnover, cfg.weighting)
    planner = LotOrderPlanner(CostModel(slippage=cfg.slippage), cfg.min_order_value)

    print("    ① 资金规模对取整误差的影响（从空仓建仓）：")
    w0 = port.target_weights(scores.loc[d0], pd.Series(dtype=float), sellable=ctx.sellable.loc[d0])
    px0 = src.price_at(d0)
    rows, detail = [], {}
    for capital in (5_000_000, 1_000_000, 300_000, 100_000):
        orders = planner.plan(w0, pd.Series(dtype="int64"), px0, capital)
        shares = pd.Series({o.symbol: o.target_shares for o in orders}, dtype="int64")
        shares = shares[shares > 0]
        real = (shares * px0.reindex(shares.index) / capital).reindex(w0.index, fill_value=0.0)  # 占总资金的权重
        invested = float((shares * px0.reindex(shares.index)).sum())
        fees = sum(o.est_fee for o in orders)
        n_min = sum(1 for o in orders if abs(o.est_fee - CostModel().total(o.est_value, False)) < 1e-9
                    and o.est_value * CostModel().commission_rate < CostModel().min_commission)
        skipped = pd.Series([x["reason"].split(" ")[0] for x in planner.skipped]).value_counts().to_dict()
        rows.append([f"{capital:,.0f}", len(shares), len(w0) - len(shares), invested / capital,
                     (real - w0).abs().sum() / 2, fees / max(invested, 1) * 1e4, n_min])
        detail[capital] = (real, list(planner.skipped), skipped)
    df = pd.DataFrame(rows, columns=["资金", "实际持仓数", "未买入", "资金使用率", "权重偏离(单边)",
                                     "显性费用(bp)", "触发最低佣金笔数"])
    print(df.round(4).to_string(index=False))
    real, skipped_list, _ = detail[100_000]
    bought = real[real > 0]
    worst = (bought - w0.reindex(bought.index)).idxmin()  # 已买入股票中取整偏离最大的
    print(f"""
    显性费用 = 佣金 + 过户费（滑点体现在成交价上，不计入这一列）。资金 10 万元时：
      单票目标 {w0[worst]:.1%}（约 {w0[worst] * 100_000:,.0f} 元），{worst} 股价 {px0[worst]:.2f} 元 → 取整后实际权重 {real[worst]:.1%}
      未买入的股票：{[f"{x['symbol']}（{x['reason']}）" for x in skipped_list] or '无'}
      即：取整到 100 股后，有的仓位不到目标的一半，有的金额低于最小委托金额 {cfg.min_order_value:,.0f} 元被跳过；
      每笔委托不足 2 万元时佣金按 5 元收取，实际费率远高于 0.025%。
    → 小资金应减少持仓数、放宽最小金额，或改用 ETF；资金规模本身就是策略参数。
    """)

    print(f"    ② 次月调仓：{d0.date()} 按 100 万建仓 → {d1.date()} 按新分数调仓（先卖后买、换手上限 {cfg.max_turnover:.0%}）")
    first = planner.plan(w0, pd.Series(dtype="int64"), px0, cfg.capital)
    held = pd.Series({o.symbol: o.target_shares for o in first}, dtype="int64")
    cash = cfg.capital - sum(o.est_value + o.est_fee for o in first)
    px1 = src.price_at(d1)
    equity = cash + float((held * px1.reindex(held.index)).sum())
    cur_w = held * px1.reindex(held.index) / equity
    w1 = port.target_weights(scores.loc[d1], cur_w, sellable=ctx.sellable.loc[d1])
    orders = planner.plan(w1, held, px1, equity)
    of = orders_frame(orders)
    print(f"      账户权益 {equity:,.0f}（现金 {cash:,.0f}），目标持仓 {len(w1)} 只，生成 {len(of)} 条指令，"
          f"卖 {int((of['side'] == 'SELL').sum())} / 买 {int((of['side'] == 'BUY').sum())}")
    show = of[["symbol", "side", "volume", "ref_price", "current_shares", "target_shares", "est_value", "est_fee", "reason"]]
    print(show.head(8).round(2).to_string(index=False))
    if planner.skipped:
        print(f"      跳过 {len(planner.skipped)} 条：{planner.skipped[:3]}")
    lots = {c: board_rule(c).name for c in of["symbol"]}
    print(f"""
      取整规则（Phase 4 board_rule，经 mini_quant/rules.py 适配）：本次涉及
      {pd.Series(lots).value_counts().to_dict()}；主板 / 创业板 100 股整数倍，科创板至少 200 股、超出部分按 1 股递增。
      清仓允许卖出零股；部分减仓按交易单位向下取整。指令按"先卖后买"排序，卖出释放的资金才能用于买入。
    """)


def plot(navs: dict, bench: pd.Series, to_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, v in navs.items():
        axes[0].plot(v.index, v.values, label=k)
    axes[0].plot(bench.index, bench.values, color="#888", ls="--", label="股票池等权（基准）")
    axes[0].set_title("样本内净值（扣费，2020–2022；横轴为信号日）")
    axes[0].legend(fontsize=8)
    x = np.arange(len(to_df))
    axes[1].bar(x - 0.2, to_df["年化成本"] * 100, width=0.4, label="年化成本 (%)", color="#eb6834")
    axes[1].bar(x + 0.2, to_df["超额年化"] * 100, width=0.4, label="超额年化 (%)", color="#2a78d6")
    axes[1].set_xticks(x, to_df["换手上限"])
    axes[1].set_xlabel("单边换手上限")
    axes[1].set_title("换手上限：成本 vs 超额（样本内）")
    axes[1].legend(fontsize=8)
    path = save_figure(fig, "03_models_turnover.png", subdir=mini_quant.SUBDIR)
    print(f"    📊 已保存：{path}")


def main():
    cfg = load_config()
    ctx = ResearchContext()
    print(f"    数据：{ctx.source.label()}；样本内 {cfg.is_start} ~ {cfg.is_end}（样本外留到 05 课）\n")
    scores = labels_and_rolling(ctx, cfg)
    navs, bench = compare_models(ctx, cfg)
    to_df = turnover_tradeoff(ctx, cfg)
    orders_demo(ctx, cfg, scores)
    plot(navs, bench, to_df)
    print("""
    ✅ 小结：
      1. 标签滞后 2 期；滚动训练记录每个模型版本的训练区间与特征
      2. 模型比较只在样本内进行；样本外留给最终评估
      3. 换手上限是成本与信号时效之间的取舍
      4. 调仓指令按交易单位取整、先卖后买；小资金的取整偏离和最低佣金不可忽略
    下一课：执行层与监控——把调仓指令交给 mini_trader 执行，并生成监控日报。
    """)


if __name__ == "__main__":
    main()
