"""
从模拟盘到小资金实盘的流程与合规
================================

学习目标：
    1. 掌握策略上线的五个阶段，以及每个阶段的准入、验证重点和回退条件
    2. 用代码打印上线检查清单，能自动验证的项目直接运行测试
    3. 掌握偏差分解：把"回测 vs 模拟盘 vs 实盘"的差异拆成信号、延迟、冲击、未成交、费用
    4. 把分阶段放量规则写成代码：预先写下规则，用数据判断放大、保持还是回退
    5. 了解上线前的合规事项（以监管与交易所最新规定为准）

核心概念：
    - 可解释的偏差：差异能被逐项拆解，才算验证通过；赚钱但解释不清同样不合格
    - 一次只改变一个变量：放大资金时不同时上线新策略，否则出问题无法归因
    - 小资金：亏完也不影响生活，且能完整覆盖交易成本结构的金额

前置知识：
    - Phase 11（冲击成本、执行）
    - 03、04、09、10 课
    - 本阶段 CONCEPT.md 第 13 节

环境要求：
    - 不需要网络；偏差归因使用合成数据（固定随机种子），结果不代表任何真实策略
    - 图表写到 outputs/phase-13/
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.data import synthetic_ohlcv
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

PHASE_DIR = Path(__file__).resolve().parent
CAPITAL = 200_000.0
QTY = 1_000
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"


# ==================== 第一部分：上线的五个阶段 ====================


def stages():
    print("=" * 60)
    print("第一部分：策略上线的五个阶段")
    print("=" * 60)
    print("""
      ① 回测       ② 纸面交易      ③ 仿真盘        ④ 小资金实盘    ⑤ 逐步放大
      历史数据     实时行情 +      SimNow /        真实账户        按规则
      验证逻辑     本地模拟撮合    测试网          最小规模        加资金
         └─ 样本外 ─▶└─ 信号一致 ─▶└─ ≥ 4 周稳定 ─▶└─ 偏差可解释 ─▶
    """)
    print(pd.DataFrame(
        [
            ["纸面交易", "回测通过样本外检验", "实时信号与回测信号一致", "信号不一致且无法解释"],
            ["仿真盘", "纸面交易无逻辑错误", "状态机、风控、对账、重启恢复、告警", "出现未处理的异常状态"],
            ["小资金实盘", "仿真盘连续运行至少 4 周，故障演练通过", "真实滑点、手续费、成交率", "实盘与回测偏差超出预设范围"],
            ["放大资金", "小资金实盘达到预设观察期", "容量与冲击成本（Phase 11）", "回撤超过预设阈值"],
        ],
        columns=["阶段", "准入标准", "重点验证", "回退条件"],
    ).to_string(index=False))
    print("""
    ⚠️ 仿真撮合过于理想：仿真盘验证的是系统，不是收益；真实滑点只能靠小资金实盘验证。
    """)


# ==================== 第二部分：上线检查清单 ====================


def run_tests(*args: str) -> str:
    """运行 pytest 并返回一行结果：能自动验证的项目不靠人工打勾"""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider", *args],
        cwd=PHASE_DIR, capture_output=True, text=True, timeout=60,
    )
    summary = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()[:80]
    return f"{'✅' if out.returncode == 0 else '❌'} {summary}"


CHECKLIST = [
    ("策略", "回测含成本，样本外结果、参数尝试次数已记录", None),
    ("策略", "仿真盘 ≥ 4 周，与回测偏差可解释", None),
    ("系统", "订单状态机覆盖拒单、部分成交、撤单、乱序与重复回报", ("tests/test_oms.py", "tests/test_exchange.py")),
    ("系统", "下单超时先查后发，不重复下单", ("tests/test_recovery.py", "-k", "lost or latency or unknown")),
    ("系统", "kill -9 后重启能从交易所恢复持仓与挂单", ("tests/test_recovery.py", "-k", "recover or replay or torn")),
    ("系统", "风控规则已配置并测试（胖手指、频率、日内亏损 Kill Switch）", ("tests/test_risk.py",)),
    ("系统", "告警链路实测可达（手机能收到 P0）", None),
    ("系统", "日志、事件日志持久化，磁盘空间监控", None),
    ("账户与合规", "程序化交易报告已按券商 / 期货公司要求完成", None),
    ("账户与合规", "API 密钥最小权限、禁止提现、IP 白名单", None),
    ("账户与合规", "资金规模、放大与停止规则已书面确定", None),
    ("人", "知道如何手动一键撤单、平仓（不依赖本系统的应急通道）", None),
    ("人", "交易时段内能及时响应告警", None),
]


def checklist():
    print("\n" + "=" * 60)
    print("第二部分：上线检查清单（代码打印 + 自动验证）")
    print("=" * 60)
    group = ""
    for category, item, test_args in CHECKLIST:
        if category != group:
            group = category
            print(f"\n   【{group}】")
        print(f"   □ {item}")
        print(f"       {run_tests(*test_args) if test_args else '⬜ 人工确认'}")
    print("""
    📌 自动验证只说明 mini_trader 这份课程代码的测试通过；
       接入真实接口后，每一项都要在你自己的系统和仿真环境上重新验证。
    """)


# ==================== 第三部分：偏差分解 ====================


@dataclass
class Fill:
    side: int  # +1 买 / -1 卖
    target: float  # 回测里的成交量
    filled: float  # 实际成交量；信号差异时为 0
    signal_px: float  # 回测假设的成交价
    arrival_px: float  # 订单到达时的价格
    fill_px: float  # 实际成交价
    fee_bt: float  # 回测假设的费用
    fee: float  # 实际费用
    day: int
    missed_signal: bool = False


def decompose(fills: list[Fill], mark: float) -> dict[str, float]:
    """
    按最终价格 mark 给每笔交易估值，把差异拆成可加的五项（正数 = 使实盘少赚）：
        信号     回测有、实盘没有的交易，其回测盈亏全部归入此项
        延迟     side × filled × (arrival - signal)
        冲击价差 side × filled × (fill - arrival)
        未成交   side × (target - filled) × (mark - signal)，错过的机会成本
        费用     实际费用 - 回测假设费用
    """
    out = dict.fromkeys(["信号差异", "延迟", "冲击与价差", "未成交", "费用"], 0.0)
    for f in fills:
        if f.missed_signal:
            out["信号差异"] += f.side * f.target * (mark - f.signal_px) - f.fee_bt
            continue
        out["延迟"] += f.side * f.filled * (f.arrival_px - f.signal_px)
        out["冲击与价差"] += f.side * f.filled * (f.fill_px - f.arrival_px)
        out["未成交"] += f.side * (f.target - f.filled) * (mark - f.signal_px)
        out["费用"] += f.fee - f.fee_bt
    return out


def pnl_backtest(fills: list[Fill], mark: float) -> float:
    return sum(f.side * f.target * (mark - f.signal_px) - f.fee_bt for f in fills)


def pnl_actual(fills: list[Fill], mark: float) -> float:
    return sum(f.side * f.filled * (mark - f.fill_px) - f.fee for f in fills if not f.missed_signal)


def hand_example():
    print("\n" + "=" * 60)
    print("第三部分：偏差分解")
    print("=" * 60)
    print("""
    R(回测) - R(实盘) = Δ信号 + Δ延迟 + Δ冲击与价差 + Δ未成交 + Δ费用

      信号价 ──延迟──▶ 到达价 ──冲击与价差──▶ 成交价
      （回测假设）     （下单时的价格）       （实际成交）

    先用两笔交易手算验证：
    """)
    fills = [
        Fill(+1, 1000, 1000, 10.00, 10.02, 10.03, 0.0, 5.0, 0),
        Fill(-1, 1000, 1000, 10.50, 10.49, 10.48, 0.0, 10.2, 1),
    ]
    parts = decompose(fills, mark=10.50)
    bt, live = pnl_backtest(fills, 10.50), pnl_actual(fills, 10.50)
    print(f"   回测盈亏 {bt:.2f}，实际盈亏 {live:.2f}，差异 {bt - live:.2f}")
    print("   分解：" + "，".join(f"{k} {v:.2f}" for k, v in parts.items()) + f"；合计 {sum(parts.values()):.2f}")


def build_trades() -> tuple[list[tuple], pd.Series]:
    """用合成行情生成双均线多空切换信号（只用于演示归因，不代表策略有效）"""
    close = synthetic_ohlcv("paper_live_b", "20250101", "20260630", start_price=20.0, annual_vol=0.28)["close"]
    pos = (close.rolling(5).mean() > close.rolling(20).mean()).astype(int)
    trades = []
    for i, change in enumerate(pos.diff().fillna(0).to_numpy()):
        if change != 0:
            trades.append((i, int(np.sign(change)), float(close.iloc[i])))
    return trades, close


def simulate(trades, *, seed, delay_bps, cost_bps, fill_rate, miss_rate, fee_rate) -> list[Fill]:
    """简化：每笔交易相互独立，部分未成交的数量不在下一笔补回"""
    rng = np.random.default_rng(seed)
    fee_bt_rate = 0.0003
    fills = []
    for day, side, px in trades:
        target = float(QTY)
        fee_bt = target * px * fee_bt_rate
        if rng.random() < miss_rate:
            fills.append(Fill(side, target, 0.0, px, px, px, fee_bt, 0.0, day, missed_signal=True))
            continue
        arrival = px * (1 + side * rng.normal(delay_bps, delay_bps) / 1e4)
        fill_px = arrival * (1 + side * abs(rng.normal(cost_bps, cost_bps / 2)) / 1e4)
        filled = target if rng.random() > 1 - fill_rate else float(rng.integers(3, 9) * 100)
        fills.append(Fill(side, target, filled, px, arrival, fill_px, fee_bt, filled * fill_px * fee_rate, day))
    return fills


def cumulative_pnl(fills: list[Fill], close: pd.Series, actual: bool) -> pd.Series:
    """每日按收盘价给累计持仓估值"""
    n = len(close)
    qty, cash = np.zeros(n), np.zeros(n)
    for f in fills:
        if actual and f.missed_signal:
            continue
        q, px, fee = (f.filled, f.fill_px, f.fee) if actual else (f.target, f.signal_px, f.fee_bt)
        qty[f.day] += f.side * q
        cash[f.day] -= f.side * q * px + fee
    return pd.Series(np.cumsum(cash) + np.cumsum(qty) * close.to_numpy(), index=close.index)


def attribution():
    trades, close = build_trades()
    mark = float(close.iloc[-1])
    # 模拟盘：本地撮合，成交率 100%，几乎没有冲击，费用与回测假设相同
    paper = simulate(trades, seed=1, delay_bps=3, cost_bps=0.5, fill_rate=1.0, miss_rate=0.0, fee_rate=0.0003)
    # 实盘：延迟更大、有真实冲击、约 10% 的订单部分成交、约 8% 的信号因停牌 / 风控拦截等没有发出
    live = simulate(trades, seed=7, delay_bps=4, cost_bps=6, fill_rate=0.9, miss_rate=0.08, fee_rate=0.0005)

    p_parts, l_parts = decompose(paper, mark), decompose(live, mark)
    bt = pnl_backtest(live, mark)
    p_pnl, l_pnl = pnl_actual(paper, mark), pnl_actual(live, mark)
    assert abs((bt - p_pnl) - sum(p_parts.values())) < 1e-6
    assert abs((bt - l_pnl) - sum(l_parts.values())) < 1e-6

    print(f"\n   合成行情 {len(close)} 个交易日，信号切换 {len(trades)} 次，每次 {QTY} 股，资金 {CAPITAL:,.0f}（模拟数据）")
    print(pd.DataFrame(
        [["回测", f"{bt:,.0f}", f"{bt / CAPITAL:+.2%}"],
         ["模拟盘", f"{p_pnl:,.0f}", f"{p_pnl / CAPITAL:+.2%}"],
         ["实盘", f"{l_pnl:,.0f}", f"{l_pnl / CAPITAL:+.2%}"]],
        columns=["阶段", "总盈亏", "收益率"],
    ).to_string(index=False))

    table = pd.DataFrame({"回测 → 模拟盘": p_parts, "回测 → 实盘": l_parts})
    table.loc["合计"] = table.sum()
    print("\n   偏差分解（元，正数表示使收益减少；负数表示这一项反而有利）：")
    print(table.round(0).to_string())
    print(f"\n   恒等式校验：实盘差异 {bt - l_pnl:,.2f} = 五项合计 {sum(l_parts.values()):,.2f} ✅")

    biggest = max(l_parts, key=lambda k: abs(l_parts[k]))
    print(f"""
    💡 如何使用：
       - 本例实盘偏差中绝对值最大的一项是"{biggest}"，下一步就针对这一项查原因
       - 这里是模拟数据，五项能完全解释差异；真实数据通常还有残差，残差本身就是要追查的问题
       - 模拟盘只暴露了延迟，看不到真实的冲击、未成交和漏单 —— 这正是必须上小资金实盘的原因
       - 未成交是机会成本，可正可负：错过的交易如果事后亏钱，这一项为负，但这不是"执行得好"
    """)
    plot_attribution(close, live, paper, l_parts)


def plot_attribution(close, live, paper, parts):
    setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[3, 2])
    series = [
        ("回测", cumulative_pnl(live, close, actual=False), BLUE),
        ("模拟盘", cumulative_pnl(paper, close, actual=True), AQUA),
        ("实盘", cumulative_pnl(live, close, actual=True), ORANGE),
    ]
    for name, s, color in series:
        ax1.plot(s.index, s.to_numpy(), color=color, lw=2, label=name)
    end_labels = [("回测 ≈ 模拟盘", series[0][1]), ("实盘", series[2][1])]  # 前两条几乎重合，合并标注
    for text, s in end_labels:
        ax1.annotate(text, (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
    ax1.set_title("累计盈亏（模拟数据）", loc="left", fontsize=11)
    ax1.set_ylabel("元")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.tick_params(axis="x", labelrotation=30)

    names = list(parts)[::-1]
    values = [parts[k] for k in names]
    ax2.barh(names, values, color=BLUE, height=0.55)
    ax2.axvline(0, color=MUTED, lw=1)
    span = max(abs(v) for v in values) or 1
    for i, v in enumerate(values):
        ax2.annotate(f"{v:,.0f}", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                     va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax2.set_xlim(-span * 1.4, span * 1.4)
    ax2.set_title("回测 → 实盘 偏差分解（元）", loc="left", fontsize=11)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    print(f"   📈 图表已保存：{save_figure(fig, '12_deviation_attribution.png', 'phase-13')}")


# ==================== 第四部分：分阶段放量 ====================

SCALE_LADDER = ["10%", "25%", "50%", "100%"]
RULES = {  # 示意阈值：要在上线前结合策略写下来，不在亏损时临时修改
    "运行周数": (lambda m: m["weeks"] >= 4, "≥ 4 周"),
    "未解决故障": (lambda m: m["incidents"] == 0, "= 0"),
    "最大回撤": (lambda m: m["drawdown"] <= 0.05, "≤ 5%"),
    "偏差可解释比例": (lambda m: m["explained"] >= 0.9, "≥ 90%"),
}
STOP_DRAWDOWN = 0.08  # 超过即回退


def evaluate_stage(stage: int, metrics: dict) -> tuple[str, list]:
    rows = [[name, desc, "✅" if rule(metrics) else "❌"] for name, (rule, desc) in RULES.items()]
    if metrics["drawdown"] > STOP_DRAWDOWN or metrics["incidents"] > 0:
        target = SCALE_LADDER[max(stage - 1, 0)]
        return f"回退到 {target} 并复盘", rows
    if all(r[2] == "✅" for r in rows):
        return f"放大到 {SCALE_LADDER[min(stage + 1, len(SCALE_LADDER) - 1)]}", rows
    return f"保持 {SCALE_LADDER[stage]}，继续观察", rows


def scale_up():
    print("\n" + "=" * 60)
    print("第四部分：分阶段放量标准")
    print("=" * 60)
    print(f"   放量阶梯（占计划资金）：{' → '.join(SCALE_LADDER)}；阈值为示意值，上线前按策略书面确定")
    print(pd.DataFrame([[k, v[1]] for k, v in RULES.items()], columns=["条件", "标准"]).to_string(index=False))
    print(f"   回退条件：回撤超过 {STOP_DRAWDOWN:.0%}，或存在未解决故障")
    cases = [
        ("运行平稳", 1, {"weeks": 6, "incidents": 0, "drawdown": 0.03, "explained": 0.93}),
        ("时间还不够", 1, {"weeks": 3, "incidents": 0, "drawdown": 0.02, "explained": 0.95}),
        ("偏差解释不清", 2, {"weeks": 8, "incidents": 0, "drawdown": 0.04, "explained": 0.70}),
        ("回撤超限", 2, {"weeks": 8, "incidents": 0, "drawdown": 0.09, "explained": 0.92}),
        ("有未解决故障", 1, {"weeks": 6, "incidents": 1, "drawdown": 0.01, "explained": 0.95}),
    ]
    rows = []
    for name, stage, m in cases:
        decision, _ = evaluate_stage(stage, m)
        rows.append([name, SCALE_LADDER[stage], m["weeks"], m["incidents"], f"{m['drawdown']:.0%}", f"{m['explained']:.0%}", decision])
    print()
    print(pd.DataFrame(rows, columns=["情形", "当前规模", "周数", "故障", "回撤", "可解释", "决策"]).to_string(index=False))
    print("""
    📌 资金管理规则：
    - 小资金：亏完也不影响生活，并且能买到最小交易单位、完整覆盖交易成本结构的金额
    - 预先写下放大条件和停止条件，严格执行，不在亏损时临时改规则
    - 一次只改变一个变量：放大资金时不同时上线新策略，否则出问题无法归因
    - "偏差解释不清"即使在赚钱也不放大：你不知道赚的是哪部分钱
    """)


# ==================== 第五部分：合规 ====================


def compliance():
    print("\n" + "=" * 60)
    print("第五部分：上线前的合规事项")
    print("=" * 60)
    print("""
    ⚠️ 以下为概述，具体要求以证监会、交易所、期货业协会和券商 / 期货公司的最新规定为准：

    - 程序化交易实行报告制度，核心原则是"先报告、后交易"（详见第 07 课）
    - 期货程序化交易：穿透式监管认证（第 06 课）+ 程序化交易报告与管理制度
    - 高频交易有更严格的差异化管理要求，认定标准以交易所实施细则为准
    - 无论是否属于程序化交易，都要避免异常交易行为：频繁申报撤单、虚假申报、
      自买自卖（对倒）、拉抬打压价格等
    - 中国内地监管部门已明确虚拟货币相关业务活动属于非法金融活动（第 08 课），本课程不涉及此类交易

    上线前务必通过券商 / 期货公司确认自己的报告义务已完成；不确定时不要上线。
    本课程内容仅用于学习系统工程，不构成任何投资建议。
    """)


# ==================== 主函数 ====================


def main():
    stages()
    checklist()
    hand_example()
    attribution()
    scale_up()
    compliance()

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 decompose 接到第 10 课的事件日志上：从 trade 事件和策略信号自动生成每周偏差报告
    2. 增加"信号差异"的反向情形：实盘有、回测没有的交易，修改 decompose 并保持恒等式成立
    3. 调高 simulate 的 cost_bps，观察偏差占比变化，估算这个策略能承受的最大单边成本
    4. 为你自己的策略写一份上线计划：资金规模、放量阶梯、停止条件、应急手动平仓步骤
    """)


if __name__ == "__main__":
    main()
