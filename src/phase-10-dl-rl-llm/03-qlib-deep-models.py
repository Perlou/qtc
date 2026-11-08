"""
Qlib 深度模型库：Alpha360 输入与截面深度模型
==========================================

学习目标：
    1. 了解 Qlib 模型库（LSTM、GRU、ALSTM、Transformer、TRA 等）的组织方式与适用的数据集类型
    2. 读懂 Alpha360 的 360 个特征与默认标签，理解 Qlib 的处理器（Processor）流水线
    3. 不下载官方数据，用 Phase 7 面板构造"类 Alpha360"输入，训练小型截面模型（GRU、MLP）
    4. 以 Rank IC / ICIR 评估，并与线性模型、简单反转因子比较

核心概念：
    - 截面化：所有股票共享一个模型，样本 = (股票, 日期) 的历史窗口，标签做截面排名标准化
    - Alpha360：过去 60 天 × 6 个字段（close/open/high/low/vwap 除以今日收盘价，volume 除以今日成交量）
    - RobustZScoreNorm：用训练期的中位数和 MAD 标准化特征；CSRankNorm：标签按当天截面排名
    - Rank IC：每天模型分数与未来收益的 Spearman 相关；ICIR = mean(IC) / std(IC)

前置知识：
    - 01、02 课；Phase 7 的 05-ic-ir-analysis.py；Phase 9 的截面 ML

环境要求：
    - pip install pyqlib torch scikit-learn（本课不调用 qlib.init，不需要下载 Qlib 官方数据）
    - 数据来自 Phase 7 共享面板（factor_data.py）；支持 QTC_OFFLINE=1
    - CPU 单次运行约 1–2 分钟
"""

import contextlib
import copy
import importlib.util
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.linear_model import Ridge

from ai_tools import SUBDIR, forward_open_returns, ic_stats, load_panel, set_seed, signal_dates, universe_mask
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

LOOKBACK, HORIZON = 60, 5
TRAIN_END, VALID_END = "2022-12-31", "2023-12-31"  # 训练 2019–2022，验证 2023，测试 2024 起

MODEL_ZOO = [
    ("循环网络", "LSTM / GRU / ALSTM", ["pytorch_lstm", "pytorch_gru", "pytorch_alstm"], "经典序列模型；ALSTM 加时间注意力"),
    ("卷积", "TCN", ["pytorch_tcn"], "时间卷积网络"),
    ("注意力", "Transformer / Localformer", ["pytorch_transformer", "pytorch_localformer"], "时间维度注意力"),
    ("图 / 关系", "GATs / HIST / IGMTF", ["pytorch_gats", "pytorch_hist", "pytorch_igmtf"], "利用股票间关系"),
    ("表格", "MLP / TabNet", ["pytorch_nn", "pytorch_tabnet"], "不建模序列"),
    ("漂移 / 鲁棒", "TRA / ADARNN / TCTS", ["pytorch_tra", "pytorch_adarnn", "pytorch_tcts"], "应对分布漂移、样本选择"),
    ("树模型", "LightGBM / DoubleEnsemble", ["gbdt", "double_ensemble"], "强基线"),
    ("时序数据集版", "*_ts（LSTM/GRU/ALSTM/TCN…）", ["pytorch_lstm_ts", "pytorch_gru_ts", "pytorch_alstm_ts"], "配合 TSDatasetH"),
]


# ==================== 第一部分：Qlib 模型库 ====================


def model_zoo():
    print("=" * 60)
    print("第一部分：Qlib 模型库（qlib.contrib.model）")
    print("=" * 60)
    import qlib

    root = Path(importlib.util.find_spec("qlib").submodule_search_locations[0]) / "contrib" / "model"
    print(f"\n    已安装 pyqlib {qlib.__version__}；模型源码目录：.../qlib/contrib/model/\n")
    rows = []
    for cat, name, files, note in MODEL_ZOO:
        ok = all((root / f"{f}.py").exists() for f in files)
        rows.append([cat, name, "✅" if ok else "—", note])
    print(pd.DataFrame(rows, columns=["类别", "模型", "源码存在", "说明"]).to_string(index=False))
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # qlib 导入时会打印可选依赖缺失的提示
            importlib.import_module("qlib.contrib.model")
        status = "可以正常导入"
    except Exception as e:  # 常见：lightgbm 动态库缺 libomp（macOS: brew install libomp）
        msg = str(e)
        why = "LightGBM 找不到 libomp 运行库（macOS 可 brew install libomp）" if "libomp" in msg else msg.splitlines()[0][:80]
        status = f"导入失败（{type(e).__name__}：{why}）"
    print(f"\n    qlib.contrib.model 包：{status}")
    print("""    该包的 __init__ 会顺带导入 LightGBM 等依赖；依赖缺失时整个包导入失败。
    本课不依赖它：我们读 Qlib 的特征定义，然后用 PyTorch 自己实现同类小模型，便于看清每一步。

    统一接口（所有模型都一样，换模型 = 换配置）：
        model.fit(dataset)                 # dataset 内含 train / valid 段，早停看 valid
        pred = model.predict(dataset, segment="test")   # 返回 (datetime, instrument) 索引的分数
    """)


# ==================== 第二部分：Alpha360 与处理器 ====================


def alpha360_spec():
    print("\n" + "=" * 60)
    print("第二部分：Alpha360 特征、标签与处理器")
    print("=" * 60)
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL

    fields, names = Alpha360DL.get_feature_config()
    f158, _ = Alpha158DL.get_feature_config()
    groups = pd.Series([n.rstrip("0123456789") for n in names]).value_counts(sort=False)
    print(f"\n    Alpha360DL.get_feature_config()：{len(fields)} 个特征 = "
          + " + ".join(f"{g}×{c}" for g, c in groups.items()))
    for i in (0, 1, 59, 60, 359):
        print(f"      {names[i]:<9} = {fields[i]}")
    print(f"    对比 Alpha158：{len(f158)} 个手工特征，例如 {f158[0]}、{f158[1]}")
    print("""
    默认标签：Ref($close, -2) / Ref($close, -1) - 1
        → t 日收盘出信号，t+1 收盘买、t+2 收盘卖（本课改用 t+1 开盘 → t+6 开盘，与 Phase 7 口径一致）

    Qlib benchmark 中深度模型常用的处理器配置（结构示意）：
        infer_processors = [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]
        learn_processors = [{"class": "DropnaLabel"},
                            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}]

    ┌──────────────────────────┬──────────────────────────────────────────────┐
    │ 组合                     │ 用法                                         │
    ├──────────────────────────┼──────────────────────────────────────────────┤
    │ Alpha158 + DatasetH      │ GBDT、MLP：每个样本一行手工特征              │
    │ Alpha360 + DatasetH      │ 模型内部把 360 维 reshape 成 60×6 序列        │
    │ Alpha158 + TSDatasetH    │ *_ts 模型：按 step_len 拼出历史窗口          │
    └──────────────────────────┴──────────────────────────────────────────────┘
    ⚠️ RobustZScoreNorm 的统计量由 fit_start_time / fit_end_time 指定的区间计算 —— 必须只覆盖训练期。
    """)


# ==================== 第三部分：用 Phase 7 面板构造类 Alpha360 ====================


def build_alpha360(panel):
    print("\n" + "=" * 60)
    print("第三部分：用 Phase 7 面板构造类 Alpha360 输入")
    print("=" * 60)
    adj = panel.adj_factor
    vwap = panel.amount / panel.volume.where(panel.volume > 0) * adj
    cube = np.stack([panel.adj_close, panel.adj_open, panel.high * adj, panel.low * adj, vwap,
                     panel.volume.where(panel.volume > 0)], axis=-1).astype(np.float64)  # (T, N, 6)
    dates = signal_dates(panel, "2019-01-01", "2025-02-14", every=HORIZON)
    fwd = forward_open_returns(panel, dates, HORIZON)
    mask = universe_mask(panel, dates)
    cal = panel.dates
    X, y, D, C = [], [], [], []
    for d in dates:
        t = cal.get_loc(d)
        win = cube[t - LOOKBACK + 1:t + 1]  # (60, N, 6)，最后一行是 t 日
        today = win[-1]
        ok = mask.loc[d].to_numpy() & fwd.loc[d].notna().to_numpy() & np.isfinite(today).all(axis=1)
        if ok.sum() < 20:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = win / np.concatenate([np.repeat(today[:, :1], 5, axis=1), today[:, 5:]], axis=1)[None]
        X.append(ratio[:, ok].transpose(1, 0, 2))  # (n_ok, 60, 6)
        r = fwd.loc[d].to_numpy()[ok]
        y.append((pd.Series(r).rank(pct=True).to_numpy() - 0.5) * 3.46)  # CSRankNorm
        D += [d] * ok.sum()
        C += list(panel.codes[ok])
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)
    D = pd.DatetimeIndex(D)
    raw_ret = fwd.stack().rename("ret")
    ret = raw_ret.reindex(pd.MultiIndex.from_arrays([D, C])).to_numpy()

    ud = D.unique()
    tr_dates = ud[ud <= pd.Timestamp(TRAIN_END)][:-1]  # 段末去掉一个信号日：其标签期跨入下一段（purge）
    va_dates = ud[(ud > pd.Timestamp(TRAIN_END)) & (ud <= pd.Timestamp(VALID_END))][:-1]
    seg = {"train": np.where(D.isin(tr_dates))[0], "valid": np.where(D.isin(va_dates))[0],
           "test": np.where(D > pd.Timestamp(VALID_END))[0]}

    # RobustZScoreNorm：中位数与 MAD 只用训练段；截断到 ±3；Fillna(0)
    flat = X.reshape(len(X), -1)
    med = np.nanmedian(flat[seg["train"]], axis=0)
    mad = np.nanmedian(np.abs(flat[seg["train"]] - med), axis=0) * 1.4826 + 1e-8
    flat = np.nan_to_num(np.clip((flat - med) / mad, -3, 3), nan=0.0)
    X = flat.reshape(X.shape).astype(np.float32)

    tag = "⚠️ 合成数据" if panel.is_synthetic else "真实数据"
    print(f"\n    面板：{len(panel.codes)} 只股票（{tag}）；每 {HORIZON} 个交易日一个信号日，共 {len(ud)} 个")
    print(f"    输入张量 X：{X.shape}（样本 × 60 天 × 6 字段）；标签：未来 {HORIZON} 日收益的截面排名（CSRankNorm）")
    for k, v in seg.items():
        print(f"      {k:<5}：{len(v):>6} 个样本，{D[v].min().date()} ~ {D[v].max().date()}")
    print("""
    构造细节：
      · 价格字段用后复权口径，除以 t 日收盘价 → 跨股票、跨时间可比（这正是 Alpha360 的做法）
      · 只保留当天可交易（上市满一年、未停牌、非 ST、次日未一字涨停）的股票
      · 窗口内停牌日为 NaN → 标准化后填 0（Fillna）
    """)
    return X, y, D, ret, seg


# ==================== 第四部分：训练截面模型 ====================


class GRUNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(6, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(360, hidden), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def daily_rank_ic(score, ret, D) -> pd.Series:
    df = pd.DataFrame({"s": score, "r": ret, "d": D})
    return df.groupby("d").apply(lambda g: g["s"].rank().corr(g["r"].rank()), include_groups=False)


def train_net(model, X, y, D, ret, seg, seed, epochs=12, patience=3, lr=1e-3, batch=512):
    """MSE 训练；早停看验证段的平均 Rank IC（Qlib 很多模型也支持以 IC 作为早停指标）"""
    Xt, yt = torch.from_numpy(X[seg["train"]]), torch.from_numpy(y[seg["train"]])
    Xv = torch.from_numpy(X[seg["valid"]])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, state, wait, ep = -np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy()
        vic = daily_rank_ic(pv, ret[seg["valid"]], D[seg["valid"]]).mean()
        if vic > best + 1e-5:
            best, state, wait = vic, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X[seg["test"]])).numpy(), ep, best


def train_and_evaluate(X, y, D, ret, seg):
    print("\n" + "=" * 60)
    print("第四部分：训练截面模型并用 Rank IC 评估")
    print("=" * 60)
    te = seg["test"]
    # X[:, -6, 0] = 标准化后的 close_{t-5}/close_t：值越大 = 过去 5 天跌得越多 → 反转逻辑下看多
    scores = {"5 日反转因子（无需训练）": X[te, -6, 0].copy()}
    ridge = Ridge(alpha=100.0).fit(X[seg["train"]].reshape(len(seg["train"]), -1), y[seg["train"]])
    scores["Ridge（360 维线性）"] = ridge.predict(X[te].reshape(len(te), -1))
    info = []
    for name, cls in (("MLP", MLPNet), ("GRU", GRUNet)):
        for s in (0, 1):
            set_seed(s)
            p, ep, vic = train_net(cls(), X, y, D, ret, seg, seed=s)
            scores[f"{name} seed={s}"] = p
            info.append(f"{name} seed={s}: {ep} 轮，验证 IC {vic:.4f}")
    print("\n    " + "；".join(info))

    rows, ics = [], {}
    for name, sc in scores.items():
        ic = daily_rank_ic(sc, ret[te], D[te])
        ics[name] = ic
        st = ic_stats(ic)
        q = pd.DataFrame({"s": sc, "r": ret[te], "d": D[te]})
        q["g"] = q.groupby("d")["s"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        spread = (q[q.g == 4].groupby("d")["r"].mean() - q[q.g == 0].groupby("d")["r"].mean()).mean()
        rows.append([name, st["ic"], st["icir"], st["t"], st["pos"], spread])
    table = pd.DataFrame(rows, columns=["模型", "Rank IC", "ICIR", "t 值", "IC>0 占比", "多空 5 日收益"])
    print(f"\n    测试段 {D[te].min().date()} ~ {D[te].max().date()}，{D[te].nunique()} 个信号日\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    gru = (scores["GRU seed=0"] + scores["GRU seed=1"]) / 2
    corr_rev = pd.Series(gru).rank().corr(pd.Series(scores["5 日反转因子（无需训练）"]).rank())
    best = table.iloc[table["Rank IC"].idxmax()]
    rev_note = ("相关性较高：GRU 很大程度上在重新发现一个简单的已知因子" if abs(corr_rev) > 0.5
                else "相关性不高：GRU 学到的并非简单反转（但这不代表它学到的东西有效）")
    print(f"""
    本次结果：测试段 Rank IC 最高的是【{best['模型']}】（{best['Rank IC']:.4f}，t = {best['t 值']:.2f}）。
    GRU 两个种子平均分数与"5 日反转因子"的秩相关 = {corr_rev:.2f} → {rev_note}
    提示：信号日数只有 {D[te].nunique()} 个，|t| < 2 的差异不应解读为"模型更好"；
    样本量（{len(seg['train'])} 个训练样本）远小于 Qlib benchmark（CSI300/500 × 十多年），
    这也是小样本上深度模型难以发挥的原因之一。
    """)
    return ics


# ==================== 第五部分：如何阅读 benchmark ====================


def reading_benchmarks(ics):
    print("\n" + "=" * 60)
    print("第五部分：如何阅读 Qlib 官方 benchmark")
    print("=" * 60)
    print("""
    · 官方结果中，深度模型并未全面胜过 LightGBM / DoubleEnsemble 等树模型；
      不同模型之间的差距常小于多次随机种子之间的波动（以最新官方表格为准）
    · 结果基于特定数据源、时段和调好的超参数；换数据或时段后排序可能改变
    · 正确姿势：把 benchmark 当作可复现的起点，在自己的数据和 Purged 验证框架下
      重新评估，报告多种子均值与标准差
    """)
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, ic in ics.items():
        ax.plot(ic.index, ic.cumsum().to_numpy(), label=name, lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("测试段累计 Rank IC（斜率稳定向上才说明预测力持续）")
    ax.legend(fontsize=8)
    path = save_figure(fig, "03_qlib_style_cumulative_ic.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    torch.set_num_threads(4)
    model_zoo()
    alpha360_spec()
    panel = load_panel()
    X, y, D, ret, seg = build_alpha360(panel)
    ics = train_and_evaluate(X, y, D, ret, seg)
    reading_benchmarks(ics)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把 RobustZScoreNorm 的统计量改成用全部样本计算，比较测试 IC，并说明为什么这仍是错误做法
    2. 实现 ALSTM：在 GRU 输出上加时间注意力池化（softmax 权重加权各时间步），与 GRU 比较
    3. 把标签换成 Qlib 默认的 Ref($close,-2)/Ref($close,-1)-1（次日收盘到后日收盘），IC 有何变化？
    4. 有 Qlib 数据的同学：用 qlib.init + Alpha360 + pytorch_gru 跑官方配置，对照本课的实现
    """)


if __name__ == "__main__":
    main()
