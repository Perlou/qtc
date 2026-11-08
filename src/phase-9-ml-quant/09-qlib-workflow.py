"""
Qlib 完整工作流
===============

学习目标：
    1. 理解 Qlib 的架构：数据层 → 表达式引擎 → DataHandler（Alpha158）→ Dataset → Model → Record → Strategy → Executor
    2. 不下载官方数据：把 Phase 7 面板转换成 Qlib 二进制格式，在本地"迷你数据集"上真实运行 Qlib
    3. 用表达式引擎计算因子，并用 pandas 实现 Alpha158 中的代表性因子、与 Qlib 结果对拍
    4. 跑通 DatasetH(Alpha158) → 自定义模型 → IC 分析 → TopkDropoutStrategy 回测，并导出 workflow 配置

核心概念：
    - 表达式：$close / Ref($close, 5) - 1；Ref(x, -n) 向未来取值，只能出现在标签里
    - Alpha158：158 个价量特征；默认标签 Ref($close, -2) / Ref($close, -1) - 1（t+1 收盘 → t+2 收盘）
    - Qlib 数据格式：calendars/day.txt、instruments/all.txt、features/<代码>/<字段>.day.bin（float32，首元素为起始日历下标）
    - 自定义模型：继承 qlib.model.base.Model，实现 fit(dataset) / predict(dataset, segment)

前置知识：
    - 05、07 课（截面预测、TopK-Dropout）

环境要求：
    - pip install pyqlib（本课用 0.9.7 验证）；lightgbm 可选
    - 不需要下载 Qlib 官方数据；导入 qlib 失败时自动降级为纯 pandas 版本
    - 数据来自 Phase 7 共享面板；无网络时可用 QTC_OFFLINE=1 运行（合成数据）
"""

import contextlib
import io
import logging
import os
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import ml_tools as mt
from utils import OUTPUT_DIR
from utils.data import CACHE_DIR

load_dotenv()

OUT_DIR = OUTPUT_DIR / "phase-9"
SEGMENTS = {"train": ("2019-01-01", "2021-12-24"), "valid": ("2022-01-04", "2022-12-23"),
            "test": ("2023-01-03", "2024-12-31")}  # 段与段之间空出约一周：Alpha158 标签要用到 t+2 的收盘价
TOPK, N_DROP = 20, 4
SEED = 42

# Alpha158 中的代表性因子（定义与 qlib.contrib.data.loader.Alpha158DL 一致）
ALPHA_SUBSET = {
    "KMID": "($close-$open)/$open",
    "KLEN": "($high-$low)/$open",
    "ROC20": "Ref($close, 20)/$close",
    "MA20": "Mean($close, 20)/$close",
    "STD20": "Std($close, 20)/$close",
    "RSV20": "($close-Min($low, 20))/(Max($high, 20)-Min($low, 20)+1e-12)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "CNTP20": "Mean($close>Ref($close, 1), 20)",
    "SUMP20": "Sum(Greater($close-Ref($close, 1), 0), 20)/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)",
    "VSTD20": "Std($volume, 20)/($volume+1e-12)",
}
LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"


def qlib_code(code: str) -> str:
    return ("SH" if code.startswith("6") else "SZ") + code


@contextlib.contextmanager
def quiet():
    """屏蔽 Qlib 的进度条、日志与第三方提示（只影响这段代码）"""
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), \
            contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# ==================== 第一部分：Qlib 架构 ====================


def architecture():
    print("=" * 60)
    print("第一部分：Qlib 架构")
    print("=" * 60)
    print("""
    Qlib（微软开源，2020）把截面选股研究的全流程标准化：

    数据层        二进制行情（按股票、按字段存储）；表达式引擎 $close、Ref、Mean、Corr…
       │
    DataHandler   Alpha158 / Alpha360 特征 + 标签 + Processor（Fillna、CSZScoreNorm、DropnaLabel…）
       │
    Dataset       DatasetH：按 segments 切 train / valid / test
       │
    Model         LGBModel、XGBModel、各种 PyTorch 模型；也可以自己继承 Model
       │
    Record        SignalRecord（预测）→ SigAnaRecord（IC）→ PortAnaRecord（回测 + 风险指标）
       │
    Strategy      TopkDropoutStrategy 等  ──▶  Executor：模拟撮合（涨跌停、成本、成交价）
    实验管理：基于 MLflow 的 R（Recorder）记录参数、模型、指标

    ┌─────────────────────┬──────────────────────────────┬──────────────────────────┐
    │ Qlib 组件           │ 本阶段对应                   │ 课                       │
    ├─────────────────────┼──────────────────────────────┼──────────────────────────┤
    │ 表达式 + Alpha158   │ ml_tools.daily_price_features│ 02                       │
    │ CSZScoreNorm 等     │ cs_rank（截面标准化）        │ 02                       │
    │ DatasetH segments   │ walk_forward 的训练 / 测试窗口│ 04、05                   │
    │ LGBModel            │ make_regressor               │ 05                       │
    │ SigAnaRecord        │ daily_rank_ic / ic_stats     │ 05                       │
    │ TopkDropoutStrategy │ topk_dropout                 │ 07                       │
    │ RollingGen / DDG-DA │ 定期 / 触发式重训            │ 10                       │
    └─────────────────────┴──────────────────────────────┴──────────────────────────┘
    🔗 类比：Qlib 之于量化研究，大致相当于 LangChain 之于 LLM 应用 —— 标准组件 + 可组合配置。
    """)


# ==================== 第二部分：把面板转成 Qlib 格式 ====================


def dump_qlib_data(panel, root: Path) -> Path:
    """
    Qlib 二进制格式（与官方 scripts/dump_bin.py 的产物一致）：
        calendars/day.txt                 交易日历
        instruments/all.txt               代码 \\t 起始日 \\t 结束日
        features/<代码小写>/<字段>.day.bin  float32 数组，第 1 个元素 = 该股票首日在日历中的下标
    价格为复权价（$close = 不复权收盘 × 复权因子），$factor 为复权因子，$volume 按复权调整。
    """
    cal = panel.dates
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in cal))
    lines = []

    def write(inst: str, fields: dict, start: int):
        d = root / "features" / inst.lower()
        d.mkdir(parents=True, exist_ok=True)
        for k, v in fields.items():
            np.hstack([[start], v.iloc[start:].to_numpy(dtype=float)]).astype("<f4").tofile(d / f"{k}.day.bin")

    for c in panel.codes:
        first = panel.close[c].first_valid_index()
        if first is None:
            continue
        f = panel.adj_factor[c]
        close = panel.close[c] * f
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = panel.amount[c] / panel.volume[c] * f
        fields = {"open": panel.open[c] * f, "high": panel.high[c] * f, "low": panel.low[c] * f, "close": close,
                  "volume": panel.volume[c] / f, "vwap": vwap, "factor": f.where(panel.close[c].notna()),
                  "change": close / close.ffill().shift(1) - 1}
        write(qlib_code(c), fields, cal.get_loc(first))
        lines.append(f"{qlib_code(c)}\t{first.date()}\t{cal[-1].date()}")
    b = panel.bench["hs300_close"]
    one = b * 0 + 1
    write("SH000300", {"open": panel.bench["hs300_open"], "high": b, "low": b, "close": b, "volume": one,
                       "vwap": b, "factor": one, "change": b.pct_change()}, 0)
    (root / "instruments" / "all.txt").write_text("\n".join(lines))
    return root


def init_qlib(root: Path):
    """初始化 Qlib；失败（未安装、版本不兼容等）时返回 None"""
    try:
        with quiet():
            import qlib
            from qlib.constant import REG_CN

            # kernels=1：本机多进程加载在部分 macOS 环境下会卡住；数据量小，单进程足够快
            qlib.init(provider_uri=str(root), region=REG_CN, kernels=1, logging_level=logging.WARNING)
        return qlib
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Qlib 初始化失败（{type(e).__name__}: {str(e)[:100]}），后续使用纯 pandas 版本")
        return None


def prepare_data(panel):
    print("\n" + "=" * 60)
    print("第二部分：不下载官方数据 —— 把 Phase 7 面板转成 Qlib 格式")
    print("=" * 60)
    try:
        with quiet():
            import qlib  # noqa: F401
        version = qlib.__version__
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 无法导入 qlib（{type(e).__name__}）。pip install pyqlib 后可运行完整版；以下使用纯 pandas 版本")
        return None
    tag = "synth" if panel.is_synthetic else "real"
    root = dump_qlib_data(panel, CACHE_DIR / f"phase9_qlib_{tag}")
    n_files = len(list((root / "features").glob("*/*.bin")))
    print(f"""
    pyqlib {version} 已安装。Qlib 官方数据体积大（数 GB），本课不下载，而是把 Phase 7 面板写成 Qlib 格式：
       目录：{root}
       {len(panel.codes)} 只股票 + 基准 SH000300，{len(panel.dates)} 个交易日，共 {n_files} 个 .bin 文件
       字段：$open $high $low $close $volume $vwap $factor $change（价格为复权价）
    """)
    q = init_qlib(root)
    if q is not None:
        print("   ✅ qlib.init(provider_uri=上述目录) 成功 —— 之后的 D.features、Alpha158、回测都运行在这份迷你数据上")
    return q


# ==================== 第三部分：表达式引擎与 Alpha158 ====================


def alpha_pandas(panel) -> pd.DataFrame:
    """用 pandas 实现 ALPHA_SUBSET（与 Qlib 的口径一致：滚动窗口 min_periods=1，按股票、按日历位置计算）"""
    f = panel.adj_factor
    o, h, l, c = panel.open * f, panel.high * f, panel.low * f, panel.close * f
    v = panel.volume / f
    dc = c - c.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "KMID": (c - o) / o,
            "KLEN": (h - l) / o,
            "ROC20": c.shift(20) / c,
            "MA20": c.rolling(20, min_periods=1).mean() / c,
            "STD20": c.rolling(20, min_periods=1).std() / c,
            "RSV20": (c - l.rolling(20, min_periods=1).min())
            / (h.rolling(20, min_periods=1).max() - l.rolling(20, min_periods=1).min() + 1e-12),
            "CORR20": c.rolling(20, min_periods=1).corr(np.log(v + 1)),
            "CNTP20": (c > c.shift(1)).astype(float).rolling(20, min_periods=1).mean(),
            "SUMP20": dc.clip(lower=0).rolling(20, min_periods=1).sum() / (dc.abs().rolling(20, min_periods=1).sum()
                                                                           + 1e-12),
            "VSTD20": v.rolling(20, min_periods=1).std() / (v + 1e-12),
        }
    codes = pd.Index([qlib_code(x) for x in panel.codes])
    frames = {k: x.set_axis(codes, axis=1) for k, x in out.items()}
    return mt.wide_to_long(frames, panel.dates, codes)


def has_future_ref(expr: str) -> bool:
    """检查表达式里有没有 Ref(x, -n)（向未来取值）。Phase 10 的 LLM 因子挖掘必须做这一步校验"""
    return re.search(r"Ref\((?:[^()]|\([^()]*\))*,\s*-\s*\d+\s*\)", expr) is not None


def expression_engine(q, panel):
    print("\n" + "=" * 60)
    print("第三部分：表达式引擎与 Alpha158")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────┬────────────────────────────────────────────┐
    │ 表达式                               │ 含义                                       │
    ├──────────────────────────────────────┼────────────────────────────────────────────┤
    │ $close / Ref($close, 5) - 1          │ 5 日收益                                   │
    │ Mean($close, 20) / $close            │ 20 日均线相对价格                          │
    │ Std($close / Ref($close, 1) - 1, 20) │ 20 日波动率                                │
    │ Corr($close, Log($volume + 1), 10)   │ 10 日量价相关                              │
    │ Ref($close, -2) / Ref($close, -1) - 1│ Alpha158 默认标签：t+1 收盘 → t+2 收盘     │
    └──────────────────────────────────────┴────────────────────────────────────────────┘
    Alpha158 = K 线形态（KMID、KLEN…）+ 价格相对位置 + 5/10/20/30/60 日的 ROC、MA、STD、BETA、RSQR、
               MAX、MIN、QTLU、RSV、CORR、CNTP、SUMP、VMA、VSTD…（158 个）
    Alpha360 = 过去 60 天的开高低收、VWAP、成交量（按最新值归一化），留给深度模型自己学（Phase 10）
    """)
    print("📋 未来函数校验（Ref(x, −n) 只能出现在标签里）：")
    for e in ["Mean($close, 20)/$close", LABEL_EXPR, "Corr($close, Ref($volume, -1), 10)"]:
        print(f"   {'❌ 含未来' if has_future_ref(e) else '✅ 无未来'}  {e}")

    pdf = alpha_pandas(panel)
    if q is None:
        print("\n   （Qlib 不可用）以下为 pandas 实现的代表性因子，最后一个交易日的截面描述：")
        print(pdf.xs(panel.dates[-1], level="datetime").describe().T[["mean", "std"]].to_string(
            float_format=lambda v: f"{v:+.4f}"))
        return pdf
    with quiet():
        from qlib.data import D
    insts = [qlib_code(c) for c in panel.codes[:3]]
    with quiet():
        demo = D.features(insts, ["$close", "$close/Ref($close, 5)-1", "Corr($close, Log($volume+1), 10)",
                                  LABEL_EXPR], start_time="2024-12-24", end_time="2024-12-31")
    demo.columns = ["$close", "5 日收益", "10 日量价相关", "标签(含未来)"]
    print(f"\n📋 D.features 示例（{len(insts)} 只股票，2024-12-24 ~ 12-31；最后一列用到了 t+1、t+2 的收盘价 = 未来信息）：")
    print(demo.to_string(float_format=lambda v: f"{v:+.4f}"))

    s, e = SEGMENTS["train"][0], SEGMENTS["test"][1]
    with quiet():
        qdf = D.features(D.instruments("all"), list(ALPHA_SUBSET.values()), start_time=s, end_time=e)
    qdf.columns = list(ALPHA_SUBSET)
    qdf = qdf.swaplevel().sort_index()
    pdf_s = pdf.loc[pd.Timestamp(s):pd.Timestamp(e)]
    a, b = qdf.align(pdf_s, join="inner")
    rows = {}
    for k in ALPHA_SUBSET:
        ok = a[k].notna() & b[k].notna() & np.isfinite(a[k]) & np.isfinite(b[k])
        rel = ((a[k][ok] - b[k][ok]).abs() / (b[k][ok].abs() + 1e-6))
        rows[k] = {"Qlib 表达式": ALPHA_SUBSET[k][:46], "对比样本": int(ok.sum()),
                   "相对误差中位数": rel.median(), "相对误差 99%": rel.quantile(0.99)}
    print(f"\n📋 Qlib 表达式引擎 vs pandas 实现（{s} ~ {e}，全部股票）：")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1e}"))
    print("""
    💡 Qlib 以 float32 存储，相对误差在 1e-6 量级属正常；99% 分位若明显偏大，通常出现在分母接近 0
       （如成交量很小的 VSTD、价格几乎不变时的 CORR）的样本上。
       口径要点：Qlib 的滚动算子 min_periods = 1（上市初期不足 20 天也会算出值），按"日历位置"而不是自然日计算。
    """)
    return pdf


# ==================== 第四部分：完整工作流 ====================


def make_model_class():
    """把 ml_tools 的模型工厂包装成 Qlib Model（LightGBM 可用时等价于 Qlib 自带的 LGBModel）"""
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.model.base import Model

    class GBDTModel(Model):
        def __init__(self, **params):
            self.params = params
            self.model = None

        @staticmethod
        def _clean(X: pd.DataFrame) -> np.ndarray:
            return X.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

        def fit(self, dataset, **kwargs):
            df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            self.model = mt.make_regressor(seed=SEED, **self.params)
            self.model.fit(self._clean(df["feature"]), df["label"].iloc[:, 0].to_numpy())
            return self

        def predict(self, dataset, segment="test"):
            X = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
            return pd.Series(self.model.predict(self._clean(X)), index=X.index)

    return GBDTModel


def qlib_workflow(q):
    print("\n" + "=" * 60)
    print("第四部分：完整工作流 —— DatasetH(Alpha158) → 模型 → IC → TopkDropout 回测")
    print("=" * 60)
    with quiet():  # 部分子模块导入时会打印第三方提示（如 gym），一并屏蔽
        from qlib.backtest import backtest
        from qlib.contrib.eva.alpha import calc_ic
        from qlib.contrib.evaluate import risk_analysis
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import init_instance_by_config

    dataset_cfg = {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                        "kwargs": {"instruments": "all", "start_time": "2018-01-02",
                                   "end_time": SEGMENTS["test"][1], "fit_start_time": SEGMENTS["train"][0],
                                   "fit_end_time": SEGMENTS["train"][1]}},
            "segments": SEGMENTS,
        },
    }
    with quiet():
        dataset = init_instance_by_config(dataset_cfg)
        train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    print(f"""
    数据集：Alpha158 handler，{train['feature'].shape[1]} 个特征；标签 {LABEL_EXPR}（训练时做截面 z-score）
       segments：{SEGMENTS}
       训练样本 {len(train)} 行（股票 × 交易日）
    模型：自定义 GBDTModel（继承 qlib.model.base.Model，内部调用 ml_tools.make_regressor → {mt.gbdt_backend()}）
    """)
    with quiet():
        Model = make_model_class()
        model = Model(n_estimators=100, num_leaves=31, min_child_samples=500, learning_rate=0.05)
        model.fit(dataset)
    rows = {}
    preds = {}
    for seg in ["valid", "test"]:
        with quiet():
            pred = model.predict(dataset, seg)
            label = dataset.prepare(seg, col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
            ic, ric = calc_ic(pred, label)
        preds[seg] = pred
        rows[seg] = {"IC": ic.mean(), "ICIR": ic.mean() / ic.std(), "RankIC": ric.mean(),
                     "RankICIR": ric.mean() / ric.std(), "交易日": len(ric)}
    print("📋 SigAna（qlib.contrib.eva.alpha.calc_ic，日频截面）：")
    print(pd.DataFrame(rows).T.astype({"交易日": int}).to_string(float_format=lambda v: f"{v:+.4f}"))

    strategy = {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"signal": preds["test"].to_frame("score"), "topk": TOPK, "n_drop": N_DROP}}
    executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    exchange = {"freq": "day", "limit_threshold": 0.095, "deal_price": "open",
                "open_cost": 0.00125, "close_cost": 0.00175, "min_cost": 5}
    cal = pd.DatetimeIndex(preds["test"].index.get_level_values("datetime").unique()).sort_values()
    with quiet():
        pm, _ = backtest(start_time=cal[0], end_time=cal[-2], strategy=strategy, executor=executor,
                         benchmark="SH000300", account=1e8, exchange_kwargs=exchange)
    rep = pm["1day"][0]
    ex_gross = rep["return"] - rep["bench"]
    ex_net = ex_gross - rep["cost"]
    with quiet():
        ra = pd.DataFrame({"超额（未扣费）": risk_analysis(ex_gross, freq="day")["risk"],
                           "超额（扣费）": risk_analysis(ex_net, freq="day")["risk"]})
    print(f"""
    回测：TopkDropoutStrategy(topk={TOPK}, n_drop={N_DROP})，SimulatorExecutor 日频，t 日预测 → t+1 开盘成交
       成本：买 {exchange['open_cost']:.3%}、卖 {exchange['close_cost']:.3%}（佣金 + 滑点 + 印花税，示例值），最低 5 元
       涨跌停：limit_threshold = 0.095（主板口径；池中创业板 / 科创板为 20%，这里是近似）
       基准：沪深 300；{rep.index[0].date()} ~ {rep.index[-1].date()}，{len(rep)} 个交易日，日均换手 {rep['turnover'].mean():.1%}
    """)
    print("📋 risk_analysis（年化指标按 238 个交易日 / 年，Qlib 的默认口径）：")
    print(ra.to_string(float_format=lambda v: f"{v:+.4f}"))
    ric = rows["test"]["RankIC"]
    print(f"""
    💡 如实解读：本次测试期日频 RankIC = {ric:+.4f}（{'为负，模型在测试期没有有效的排序能力' if ric < 0 else '为正'}），
       超额年化：扣费前 {ra.loc['annualized_return', '超额（未扣费）']:+.2%}，扣费后 {ra.loc['annualized_return', '超额（扣费）']:+.2%}
       —— 日频调仓、日均换手 {rep['turnover'].mean():.0%} 时，成本是决定性因素（07 课）。
       只有约 130 只股票、2 年测试期、一组未调参的超参数，结果的统计误差很大；
       这里的目的是跑通 Qlib 的完整流程。在全市场数据上，相同的配置文件可以直接换成官方数据运行（第五部分）。
       注意 Alpha158 标签是 t+1 收盘 → t+2 收盘，而这里按 t+1 开盘成交，二者口径略有差异。
    """)


def pandas_workflow(pdf, panel):
    """Qlib 不可用时的降级：同样的分段、同样的标签，用 pandas 因子 + ml_tools 模型"""
    print("\n" + "=" * 60)
    print("第四部分（降级版）：pandas 因子 + GBDT，按 Qlib 的分段与标签")
    print("=" * 60)
    c = panel.adj_close.set_axis([qlib_code(x) for x in panel.codes], axis=1)
    label = mt.wide_to_long({"y": c.shift(-2) / c.shift(-1) - 1}, panel.dates, c.columns)["y"]
    df = pdf.assign(label=label).replace([np.inf, -np.inf], np.nan)
    lvl = df.index.get_level_values("datetime")

    def seg(name):
        a, b = SEGMENTS[name]
        return df[(lvl >= pd.Timestamp(a)) & (lvl <= pd.Timestamp(b)) & df["label"].notna()]

    tr = seg("train")
    y = tr["label"].groupby(level="datetime").transform(lambda s: (s - s.mean()) / s.std())
    model = mt.make_regressor(seed=SEED, n_estimators=100, min_child_samples=500).fit(tr[list(ALPHA_SUBSET)], y)
    rows = {}
    for name in ["valid", "test"]:
        d = seg(name)
        ic = mt.daily_rank_ic(pd.Series(model.predict(d[list(ALPHA_SUBSET)]), d.index), d["label"])
        rows[name] = mt.ic_stats(ic)
    print(mt.ic_frame(rows).to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"""
    以上为 {len(ALPHA_SUBSET)} 个代表性 Alpha158 因子 + {mt.gbdt_backend()} 的实际结果（日频截面 RankIC）。
    t 值假设各交易日独立；截面相关与 IC 的自相关会让它偏乐观。安装 pyqlib 后本节会自动切换为完整的 Qlib 工作流。
    """)


# ==================== 第五部分：配置文件与官方数据 ====================

OFFICIAL_YAML = """# Qlib 官方风格的 workflow 配置（Phase 9 第 09 课导出）
# 运行：qrun workflow_config_lightgbm_alpha158.yaml   （需要 lightgbm 与官方 / 社区 cn_data）
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn
market: &market csi300
benchmark: &benchmark SH000300
data_handler_config: &data_handler_config
    start_time: 2008-01-01
    end_time: 2020-08-01
    fit_start_time: 2008-01-01
    fit_end_time: 2014-12-31
    instruments: *market
port_analysis_config: &port_analysis_config
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5
    backtest:
        start_time: 2017-01-01
        end_time: 2020-08-01
        account: 100000000
        benchmark: *benchmark
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015      # 旧印花税时代的设定，按现行费率调整
            min_cost: 5
task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs:
            loss: mse
            colsample_bytree: 0.8879
            learning_rate: 0.2
            subsample: 0.8789
            lambda_l1: 205.6999
            lambda_l2: 580.9768
            max_depth: 8
            num_leaves: 210
            num_threads: 20
    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test: [2017-01-01, 2020-08-01]
    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            ana_long_short: False
            ann_scaler: 252
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config: *port_analysis_config
"""


def export_configs(q_root: Path | None):
    print("\n" + "=" * 60)
    print("第五部分：workflow 配置与官方数据")
    print("=" * 60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "workflow_config_lightgbm_alpha158.yaml"
    p1.write_text(OFFICIAL_YAML, encoding="utf-8")
    paths = [p1]
    if q_root is not None:
        local = (OFFICIAL_YAML.replace('"~/.qlib/qlib_data/cn_data"', f'"{q_root}"')
                 .replace("market: &market csi300", "market: &market all")
                 .replace("start_time: 2008-01-01\n    end_time: 2020-08-01", "start_time: 2018-01-02\n    end_time: 2024-12-31")
                 .replace("fit_start_time: 2008-01-01\n    fit_end_time: 2014-12-31",
                          f"fit_start_time: {SEGMENTS['train'][0]}\n    fit_end_time: {SEGMENTS['train'][1]}")
                 .replace("start_time: 2017-01-01\n        end_time: 2020-08-01",
                          f"start_time: {SEGMENTS['test'][0]}\n        end_time: 2024-12-30")
                 .replace("train: [2008-01-01, 2014-12-31]", f"train: [{', '.join(SEGMENTS['train'])}]")
                 .replace("valid: [2015-01-01, 2016-12-31]", f"valid: [{', '.join(SEGMENTS['valid'])}]")
                 .replace("test: [2017-01-01, 2020-08-01]", f"test: [{', '.join(SEGMENTS['test'])}]")
                 .replace("topk: 50", f"topk: {TOPK}").replace("n_drop: 5", f"n_drop: {N_DROP}")
                 .replace("# 运行：qrun workflow_config_lightgbm_alpha158.yaml   （需要 lightgbm 与官方 / 社区 cn_data）",
                          "# 本地迷你数据版（Phase 7 面板转换）。运行：qrun workflow_config_local_mini.yaml（需要 lightgbm）"))
        p2 = OUT_DIR / "workflow_config_local_mini.yaml"
        p2.write_text(local, encoding="utf-8")
        paths.append(p2)
    for p in paths:
        print(f"   📄 已写出：{p}")
    print("""
    官方配置中的 <MODEL>、<DATASET>、<PRED> 是 qrun 的占位符，由框架在运行时替换。
    LGBModel 的超参数来自官方 benchmark（在特定数据和时段上调出），直接沿用是一种隐性的数据窥探。

    下载数据后如何运行（本课没有执行这些命令）：
       1. 安装依赖：brew install libomp（macOS）；pip install pyqlib lightgbm
       2. 准备 cn_data：Qlib 官方数据集曾因数据政策暂停提供，社区有替代数据源（如 GitHub 上的
          chenditc/investment_data 发布的 qlib_bin.tar.gz）—— 以 Qlib README 的最新说明为准。
          解压到 ~/.qlib/qlib_data/cn_data（目录结构与第二部分生成的迷你数据完全相同）
       3. 运行：qrun outputs/phase-9/workflow_config_lightgbm_alpha158.yaml
          结果记录在 ./mlruns（MLflow），用 R.get_recorder() 读取预测、IC 与回测报告
       4. 与 Phase 7 线性多因子在同一数据、同一成本、同一策略下比较；把调参次数计入 DSR（08 课）

    ⚠️ 注意事项：数据质量（公开数据源有缺失和误差）、成交价口径、成本参数、创业板 / 科创板 20% 涨跌幅、
       ST 股票涨跌幅（主板 ST 在 2026-07-06 前为 5%、之后为 10%）都需要按自己的场景调整。
    """)


def main():
    data = mt.load_ml_data()
    panel = data.panel
    if data.is_synthetic:
        print("⚠️ 当前是合成面板：数字只用于跑通流程，不代表真实市场\n")
    mt._warn_fallback()  # 第四部分的模型训练在静默块中进行，提前提示当前 GBDT 后端
    architecture()
    q = prepare_data(panel)
    pdf = expression_engine(q, panel)
    ok = False
    if q is not None:
        try:
            qlib_workflow(q)
            ok = True
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Qlib 工作流运行失败（{type(e).__name__}: {str(e)[:120]}），降级为 pandas 版本")
    if not ok:
        pandas_workflow(pdf, panel)
    tag = "synth" if panel.is_synthetic else "real"
    export_configs(CACHE_DIR / f"phase9_qlib_{tag}" if q is not None else None)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 用 D.features 写出 Alpha158 中的 BETA20、RSQR20、QTLU20，并用 pandas 实现对拍
    2. 把 Alpha158 handler 换成 Alpha360，模型不变，比较 RankIC（提示：Alpha360 更适合深度模型）
    3. 用 qlib.workflow 的 R.start(...) + SignalRecord / SigAnaRecord / PortAnaRecord 改写第四部分，查看 mlruns 中的记录
    4. 把第四部分改成 Qlib 的滚动训练（qlib.workflow.task.gen.RollingGen，step = 60 个交易日），与一次性训练比较
    """)


if __name__ == "__main__":
    main()
