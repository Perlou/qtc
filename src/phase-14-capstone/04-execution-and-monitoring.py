"""
执行层与监控面板
================

学习目标：
    1. 把流水线生成的调仓指令交给 Phase 13 的 mini_trader 执行：风控 → OMS → MockExchange
    2. 处理真实执行中的"不完美"：开盘涨停买不进、盘口深度不足导致部分成交、交易所拒单、T+1 卖不出
    3. 日终对账：本地 OMS（由成交推导）vs 交易所持仓与资金；并演示对账发现差异
    4. 连续两个调仓日：第二天从交易所查询持仓与资金，重新跑流水线（未成交部分次日重新评估）
    5. 生成 Markdown 监控日报：净值、模拟盘 vs 回测、因子健康、特征漂移、组合暴露、交易、系统、告警

核心概念：
    - 交易所是真相来源：每天开盘前以交易所持仓为准初始化本地状态
    - 未成交部分次日重新评估，而不是盲目追单
    - 失败可见：流水线失败也要出日报，并升级为 P1 告警

前置知识：
    - Phase 13（OMS、风控、MockExchange、对账）、Phase 11（执行质量）
    - 本阶段 CONCEPT.md 第 5 节

环境要求：
    - 不需要网络（行情来自本地面板缓存或合成数据）；日报写到 outputs/phase-14/
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
from mini_quant.contracts import PipelineError  # noqa: E402
from mini_quant.execution import MiniTraderExecutor, build_market, to_exchange_symbol  # noqa: E402
from mini_quant.factors import ic_series  # noqa: E402
from mini_quant.models import training_dates  # noqa: E402
from mini_quant.monitor import (  # noqa: E402
    build_alerts,
    factor_health,
    feature_drift,
    kv_table,
    md_table,
    nav_metrics,
    sector_exposure,
    write_markdown,
)
from mini_quant.pipeline import build_daily_pipeline  # noqa: E402
from mini_quant.portfolio import orders_frame  # noqa: E402
from mini_quant.research import ResearchContext  # noqa: E402

from bt_core import CostModel  # noqa: E402
from mini_trader import Direction  # noqa: E402
from mini_trader.recovery import positions_to_map, reconcile_positions  # noqa: E402
from utils.plotting import save_figure, setup_chinese_font  # noqa: E402

load_dotenv()
setup_chinese_font()
pd.set_option("display.width", 170)

FILL_COLS = ["symbol", "side", "ordered", "sent", "filled", "status", "limit_price", "arrival", "avg_price",
             "arrival_bp", "shortfall_bp", "reason"]


def show_fills(rep) -> None:
    f = rep.fills[FILL_COLS].copy()
    for c in ["avg_price", "arrival_bp", "shortfall_bp"]:
        f[c] = f[c].astype(float).round(2)
    f["reason"] = f["reason"].str.slice(0, 28)
    print(f.to_string(index=False))
    if rep.skipped:
        print("    执行前校验拦截（不发单）：")
        for s in rep.skipped:
            print(f"      ⛔ {s['symbol']} {s['side']} {s['volume']}：{s['reason']}")
    for e in rep.extra:
        print(f"    额外请求 {e['symbol']} {e['side']} {e['volume']} → {e['outcome']}：{e['detail']}")


# ==================== 第一部分：执行层职责 ====================


def execution_overview():
    print("=" * 60)
    print("第一部分：执行层——复用 Phase 13 的交易系统")
    print("=" * 60)
    print("""
    调仓指令（RebalanceOrder） ──▶ MiniTraderExecutor.execute()
        ① 执行前校验：停牌、开盘涨停（买）/ 跌停（卖）→ 不发单，次日重新评估
        ② 先卖后买：卖出成交后刷新可用资金，再发买单；资金不足时按交易单位缩量
        ③ 限价 = 开盘价 ×(1 ± 0.3%)，不超过涨跌停价（避免市价单在流动性差时成交在极端价）
        ④ TradingSystem.send_order：RiskEngine 事前风控 → OMS 建单 → 网关（超时先查后发）
        ⑤ MockExchange 撮合：成交量受盘口深度限制 → 部分成交；日终未成交部分失效
        ⑥ 对账：本地 OMS 持仓（只由成交累计）vs 交易所持仓；按成交推算资金 vs 交易所余额

    模拟盘口：开盘价 = 面板中下一交易日的真实开盘价；可成交深度 = 当日成交量 × 2%（粗略假设）
    ⚠️ MockExchange 只收比例佣金（0.025%），不收印花税和最低佣金；日报中的"费用"按 Phase 4
       CostModel 另行估算。到达价偏差 = 成交均价相对开盘价，执行缺口 = 相对信号日收盘价（含隔夜跳空）。
    """)


# ==================== 第二部分：第 1 天——建仓（含故障情景） ====================


def day_one(pipe, executor, cfg, d1):
    print("\n" + "=" * 60)
    print("第二部分：第 1 个调仓日——从空仓建仓，并注入执行中的常见问题")
    print("=" * 60)
    src = pipe.data
    res = pipe.run(d1, executor.positions(), cash=executor.cash())
    t1 = src.next_trading_day(d1)
    codes = [o.symbol for o in res.orders]
    base = build_market(src, t1, codes)
    s_limit, s_reject, s_part = codes[0], codes[1], codes[2]
    part_order = next(o for o in res.orders if o.symbol == s_part)
    overrides = {
        s_limit: {"open": base[s_limit].limit_up, "note": "情景：开盘一字涨停"},
        s_part: {"depth": float(max(part_order.volume // 2 // 100 * 100, 100)), "note": "情景：盘口深度只有委托量的一半"},
    }
    market = build_market(src, t1, codes, overrides=overrides)
    print(f"    信号日 {d1.date()} 收盘后生成 {len(res.orders)} 条买单，{t1.date()} 开盘执行。注入情景：")
    print(f"      {s_limit}：开盘价设为涨停价 {base[s_limit].limit_up}（买不进）")
    print(f"      {s_reject}：交易所拒单（MockExchange.inject('reject')）")
    print(f"      {s_part}：盘口深度 {overrides[s_part]['depth']:.0f} 股 < 委托 {part_order.volume} 股（部分成交）")
    first_buy = next(o for o in res.orders if o.symbol not in (s_limit, s_reject, s_part))
    extra = [{"symbol": first_buy.symbol, "side": "SELL", "volume": 100, "note": "T+1 测试"}]
    print(f"      {first_buy.symbol}：买入成交后立刻卖出 100 股（A 股 T+1，应被风控拒绝）\n")
    rep = executor.execute(res.orders, market, inject_reject={s_reject}, extra_requests=extra, trade_date=t1)
    show_fills(rep)
    print(f"\n    执行摘要：{rep.summary()}")
    print(f"    风控拦截统计：{rep.risk_rejects}；OMS 异常：{rep.anomalies or '无'}")
    print(f"    对账：持仓差异 {rep.position_diffs or '无'}；资金差异 {rep.cash_diff:+.2f} 元 → "
          f"{'✅ 一致' if rep.reconciled else '❌ 不一致'}")
    return res, rep


# ==================== 第三部分：第 2 天——从交易所状态继续 ====================


def day_two(pipe, executor, cfg, d1, d2, rep1):
    print("\n" + "=" * 60)
    print("第三部分：第 2 个调仓日——以交易所持仓为准，重新跑流水线")
    print("=" * 60)
    src = pipe.data
    shares, cash = executor.positions(), executor.cash()
    missed = sorted({s["symbol"] for s in rep1.skipped} | set(rep1.fills.loc[rep1.fills["filled"] < rep1.fills["ordered"], "symbol"]))
    print(f"    交易所查询：持仓 {len(shares)} 只，现金 {cash:,.0f} 元；第 1 天未完成的标的：{missed}")
    res = pipe.run(d2, shares, cash=cash)
    of = orders_frame(res.orders)
    again = sorted(set(missed) & set(of.loc[of["side"] == "BUY", "symbol"]))
    print(f"    {d2.date()} 生成 {len(of)} 条指令（卖 {int((of['side'] == 'SELL').sum())} / 买 {int((of['side'] == 'BUY').sum())}），"
          f"单边换手 {res.info['turnover']:.1%}（上限 {cfg.max_turnover:.0%}）")
    print(f"    第 1 天未完成、今天按最新分数重新评估后仍要买入的：{again or '无'}（不是'昨天没买到所以今天追'）")
    t2 = src.next_trading_day(d2)
    # 模拟盘估值：第 2 天开盘、调仓之前（与回测"持有到下一调仓日次日开盘"的窗口一致）
    open_t2 = src.panel.open.loc[t2].fillna(src.panel.close_ffill.loc[d2])
    equity_before = executor.equity(open_t2)
    market = build_market(src, t2, [o.symbol for o in res.orders])
    rep = executor.execute(res.orders, market, trade_date=t2)
    print(f"\n    {t2.date()} 执行结果：")
    show_fills(rep)
    print(f"\n    执行摘要：{rep.summary()}")
    return res, rep, t2, equity_before


def reconcile_demo(executor, rep_list):
    print("\n" + "=" * 60)
    print("第四部分：对账——发现交易所侧的持仓变化")
    print("=" * 60)
    ledger: dict[str, float] = {}
    for rep in rep_list:  # 平台自己的成交台账：只由成交累计
        for _, r in rep.fills.iterrows():
            key = f"{to_exchange_symbol(r['symbol'])}|LONG"
            ledger[key] = ledger.get(key, 0.0) + (r["filled"] if r["side"] == "BUY" else -r["filled"])
    ledger = {k: v for k, v in ledger.items() if v != 0}
    remote = positions_to_map(executor.exchange.query_positions())
    print(f"    平台成交台账 vs 交易所持仓：差异 {reconcile_positions(ledger, remote) or '无'}")
    sym = sorted(remote)[0].split("|")[0]
    executor.exchange.adjust_position(sym, Direction.LONG, -100)  # 交易所侧调整（如司法划转、强制卖出）
    diffs = reconcile_positions(ledger, positions_to_map(executor.exchange.query_positions()))
    print(f"    交易所侧调整 {sym} −100 股（本地不会收到成交回报）后再对账：")
    for d in diffs:
        print(f"      ❌ {d}")
    print("    → 无法由成交解释的差异：暂停交易（Kill Switch），P1 告警，人工确认后以交易所为准（Phase 13 第 10 课）")
    executor.exchange.adjust_position(sym, Direction.LONG, +100)  # 演示结束，恢复
    return diffs


# ==================== 第五部分：监控日报 ====================


def build_daily_report(cfg, ctx, pipe_res, rep, t2, paper, steps_df, extra_alerts=None, failed: str = "") -> str:
    src = ctx.source
    run = ctx.run(cfg)
    nav = (1 + run.net()).cumprod()
    bench = (1 + run.bench_ret()).cumprod()
    nav_m = nav_metrics(pd.concat([pd.Series([1.0], index=[nav.index[0] - pd.offsets.MonthEnd(1)]), nav]),
                        pd.concat([pd.Series([1.0], index=[bench.index[0] - pd.offsets.MonthEnd(1)]), bench]), periods=12)
    clean = ctx.engine(cfg).panel(ctx.dates)
    ics = ic_series(clean, ctx.period_ret)
    fh = factor_health(ics)
    tr = training_dates(ctx.dates, pipe_res.as_of, cfg.train_window)
    X_long = ctx.factor_long(cfg)
    X_train = X_long[X_long.index.get_level_values("date").isin(tr)]
    drift = feature_drift(X_train, pipe_res.factors)
    expo = sector_exposure(pipe_res.target, src.sector, src.codes)
    summ = rep.summary() if rep is not None else None
    alerts = build_alerts(nav_m, fh, drift, summ, pipeline_ok=not failed) + list(extra_alerts or [])

    pct = {"累计收益", "基准累计", "累计超额", "年化波动", "当前回撤", "最大回撤", "近 20 期跟踪误差(年化)"}
    lines = [f"# mini-quant 监控日报：{t2.date()}", "",
             f"> 策略 `{cfg.name}`（{cfg.short()}）· 数据：{src.label()} · 生成方式：04-execution-and-monitoring.py",
             "> 本日报中的回测净值为月频（每期 = 一个调仓周期），模拟盘为 MockExchange 上的两日演示。", ""]
    if failed:
        lines += ["## ❌ 流水线失败", "", f"失败原因：{failed}", "", "今日不生成调仓指令。", ""]
    lines += ["## 1. 告警", ""]
    lines += [f"- **{a.level}** `{a.key}`：{a.message}" for a in alerts] or ["- 无"]
    lines += ["", "## 2. 净值（回测，扣费，月频）", "", kv_table(nav_m, pct_keys=pct), "",
              "## 3. 模拟盘 vs 回测（同期）", "", kv_table(paper, pct_keys=set(paper)), "",
              "> 同一窗口：首个执行日开盘 → 第二个执行日开盘（调仓前）。差异来自取整、现金拖累、未成交和成本口径。", "",
              "## 4. 因子健康（Rank IC）", "",
              md_table(fh, formats={c: (lambda v: f"{float(v):.3f}") for c in fh.columns if c != "状态"}), "",
              "## 5. 模型健康：特征分布漂移（PSI，当前截面 vs 训练窗口）", "",
              md_table(drift, formats={"PSI": lambda v: f"{float(v):.3f}"}), "",
              f"预测分数：均值 {pipe_res.scores.mean():.4f}，标准差 {pipe_res.scores.std():.4f}，有效 {pipe_res.scores.notna().sum()} 只", "",
              "## 6. 组合暴露（行业权重 vs 股票池等权）", "", md_table(expo, pct_cols={"组合", "股票池等权", "偏离"}), ""]
    if rep is not None:
        f = rep.fills
        lines += ["## 7. 交易", "", kv_table(summ, pct_keys={"成交率(金额)"}), "",
                  md_table(f[["symbol", "side", "ordered", "filled", "status", "avg_price", "arrival_bp", "fee"]].set_index("symbol"),
                           formats={"avg_price": lambda v: f"{v:.2f}", "arrival_bp": lambda v: "—" if v != v else f"{v:.1f}",
                                    "fee": lambda v: f"{v:.2f}"}), ""]
    lines += ["## 8. 系统（流水线各步骤）", "", md_table(steps_df.set_index("step"),
                                                    formats={"seconds": lambda v: f"{float(v):.3f}"}), ""]
    return "\n".join(lines), alerts, nav, bench


def monitoring(cfg, ctx, pipe, executor, res2, rep2, t2, d1, rep1, equity_before):
    print("\n" + "=" * 60)
    print("第五部分：监控日报（Markdown）")
    print("=" * 60)
    src = ctx.source
    paper_ret = equity_before / cfg.capital - 1
    bt_ret = float(ctx.run(cfg).frame["net"].get(d1, np.nan))
    paper = {"模拟盘收益": paper_ret, "同期回测收益": bt_ret, "差异": paper_ret - bt_ret}
    print(f"    模拟盘权益 {equity_before:,.0f} 元（{t2.date()} 开盘、调仓前估值），收益 {paper_ret:+.2%}；"
          f"同期回测（{d1.date()} 信号期）{bt_ret:+.2%}，差异 {paper_ret - bt_ret:+.2%}")
    print("    同一窗口下的差异来源：取整与现金拖累、未成交（涨停 / 拒单 / 部分成交）、成交价（盘口一档）、成本口径")
    text, alerts, nav, bench = build_daily_report(cfg, ctx, res2, rep2, t2, paper, res2.steps_frame())
    path = write_markdown(text, f"daily_report_{t2:%Y%m%d}.md")
    print(f"\n    📝 日报已写入：{path}（{len(text.splitlines())} 行）")
    print("    告警：")
    for a in alerts:
        print(f"      [{a.level}] {a.key}：{a.message}")
    if not alerts:
        print("      无")

    print("\n    ▶ 失败可见：数据未更新时，流水线失败也要出日报")
    try:
        pipe.run(src.calendar[-1] + pd.offsets.BDay(10), executor.positions(), cash=executor.cash())
    except PipelineError as exc:
        steps = pd.DataFrame([s.__dict__ for s in exc.steps])
        text_f, alerts_f, _, _ = build_daily_report(cfg, ctx, res2, None, t2, paper, steps, failed=str(exc))
        path_f = write_markdown(text_f, "daily_report_failed_example.md")
        print(f"      ❌ {exc}")
        print(f"      失败日报：{path_f}；告警：{[f'{a.level} {a.key}' for a in alerts_f if a.level == 'P1']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(nav.index, nav.values, label="策略（扣费）", color="#2a78d6")
    axes[0].plot(bench.index, bench.values, label="股票池等权", color="#888", ls="--")
    axes[0].axvline(pd.Timestamp(cfg.oos_start), color="#bbb", ls=":")
    axes[0].set_title(f"回测净值（{cfg.short()}，横轴为信号日）")
    axes[0].legend(fontsize=8)
    f = pd.concat([rep1.fills.assign(day="第1天"), rep2.fills.assign(day="第2天")])
    rate = (f["filled"] / f["ordered"]).to_numpy()
    colors = ["#2a78d6" if r >= 1 else ("#eda100" if r > 0 else "#eb6834") for r in rate]
    axes[1].bar(range(len(f)), rate, color=colors)
    axes[1].axvline(len(rep1.fills) - 0.5, color="#888", lw=1)
    axes[1].set_title("每条指令的成交比例（蓝 = 全部，黄 = 部分，橙 = 未成交）")
    axes[1].set_xlabel("指令序号（竖线左侧为第 1 天）")
    path_fig = save_figure(fig, "04_execution_monitor.png", subdir=mini_quant.SUBDIR)
    print(f"\n    📊 已保存：{path_fig}")


def main():
    cfg = load_config()
    ctx = ResearchContext()
    pipe = build_daily_pipeline(cfg, source=ctx.source)
    executor = MiniTraderExecutor(cfg.capital, cost_model=CostModel(slippage=cfg.slippage))
    dates = ctx.source.rebalance_dates()
    d1, d2 = dates[-2], dates[-1]
    print(f"    数据：{ctx.source.label()}；演示调仓日 {d1.date()}、{d2.date()}，资金 {cfg.capital:,.0f}\n")
    execution_overview()
    _, rep1 = day_one(pipe, executor, cfg, d1)
    res2, rep2, t2, equity_before = day_two(pipe, executor, cfg, d1, d2, rep1)
    reconcile_demo(executor, [rep1, rep2])
    monitoring(cfg, ctx, pipe, executor, res2, rep2, t2, d1, rep1, equity_before)
    print("""
    ✅ 小结：
      1. 执行层只做适配：指令校验、先卖后买、限价，交易全部交给 mini_trader
      2. 涨停、拒单、部分成交、T+1 都会让实际持仓偏离目标——次日按最新分数重新评估
      3. 每天以交易所为真相来源；对账差异无法解释时暂停交易
      4. 日报覆盖净值、模拟盘 vs 回测、因子、模型、暴露、交易、系统七个面板，失败也要出日报
    下一课：把整个研究写成一份可复现的策略研究报告。
    """)


if __name__ == "__main__":
    main()
