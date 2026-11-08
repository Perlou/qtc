"""
Transformer 时序模型：因果掩码、DLinear 基线与诚实的比较
========================================================

学习目标：
    1. 把自注意力用在时间维度上，理解因果掩码（causal mask）为什么是"防前视"的结构保证
    2. 用实验看到：去掉因果掩码的逐步预测模型会在训练中"抄答案"
    3. 实现 DLinear（序列分解 + 线性层），作为 Transformer 必须先打败的基线
    4. 在多资产日频数据上如实比较 Transformer、DLinear、Ridge 与零预测

核心概念：
    - Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)V，M_ij = −∞ (j > i) 即因果掩码
    - 逐步预测（seq2seq）：每个位置 i 都预测 i 之后一天的收益 —— 不加掩码就能看到答案
    - DLinear（Zeng et al., AAAI 2023）：移动平均分解为趋势 + 残差，各接一个线性层
    - 多资产共享模型：5 只 ETF 合并训练，样本量比单资产大数倍

前置知识：
    - 01-lstm-price-prediction.py（窗口、切分、基线）
    - 深度学习课程中的 Transformer 编码器

环境要求：
    - pip install torch pandas numpy scikit-learn scipy matplotlib（CPU，单次运行约 1 分钟）
    - 数据：510300 / 510500 / 159915 / 518880 / 513100 五只 ETF 日线；支持 QTC_OFFLINE=1
"""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from ai_tools import SUBDIR, fmt_pct, load_etf, set_seed
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

ETFS = {"510300": "沪深300", "510500": "中证500", "159915": "创业板", "518880": "黄金", "513100": "纳指"}
L = 30
TRAIN_END, VALID_END = "2019-12-31", "2020-12-31"
FEATS = ["r1", "gap", "intraday", "range", "vol_z"]


# ==================== 模型 ====================


class TSTransformer(nn.Module):
    """小型时序 Transformer 编码器：每个位置输出"下一天收益"的预测，形状 (B, L)"""

    def __init__(self, n_feat: int, L: int, d_model: int = 32, nhead: int = 4, layers: int = 2,
                 causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, L, d_model) * 0.02)  # 可学习位置编码
        pe, t = torch.zeros(1, L, d_model), torch.arange(L)[:, None]
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100.0) / d_model))
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(t * freq), torch.cos(t * freq)
        self.register_buffer("pe", pe)  # 固定正弦位置编码：让模型容易识别"相邻位置"
        layer = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)
        self.causal = causal
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(L))

    def forward(self, x):  # x: (B, L, F)
        h = self.enc(self.inp(x) + self.pos + self.pe, mask=self.mask if self.causal else None)
        return self.head(h).squeeze(-1)


class DLinear(nn.Module):
    """DLinear：滑动平均分解为趋势 + 残差，各用一个 Linear(L→1)，再跨特征线性组合"""

    def __init__(self, n_feat: int, L: int, k: int = 5):
        super().__init__()
        self.k = k
        self.trend, self.season = nn.Linear(L, 1), nn.Linear(L, 1)
        self.mix = nn.Linear(n_feat, 1)

    def forward(self, x):  # x: (B, L, F) → (B,)
        xt = x.transpose(1, 2)  # (B, F, L)
        pad = (self.k - 1) // 2
        padded = torch.cat([xt[..., :1].repeat(1, 1, pad), xt, xt[..., -1:].repeat(1, 1, pad)], dim=-1)
        trend = F.avg_pool1d(padded, self.k, stride=1)  # 只在输入窗口内平滑，不涉及未来
        out = self.trend(trend) + self.season(xt - trend)  # (B, F, 1)
        return self.mix(out.squeeze(-1)).squeeze(-1)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def fit(model, Xtr, Ytr, Xva, Yva, seq2seq: bool, epochs=12, lr=1e-3, batch=128, patience=3, seed=0,
        es_all: bool = False):
    """seq2seq=True：损失取所有位置；False：只取最后一个位置。
    早停默认看验证集"最后位置"的 MSE（真正要用的预测）；es_all=True 时看所有位置的平均 MSE"""
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    best, state, wait = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            out = model(Xtr[idx])
            loss = F.mse_loss(out, Ytr[idx]) if seq2seq else F.mse_loss(out, Ytr[idx][:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva)
            if es_all and seq2seq:
                vl = F.mse_loss(out, Yva).item()
            else:
                vl = F.mse_loss(out[:, -1] if seq2seq else out, Yva[:, -1]).item()
        if vl < best - 1e-6:
            best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def last_pred(model, X, seq2seq: bool) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(X))
    return (out[:, -1] if seq2seq else out).numpy()


# ==================== 第一部分：时间维度上的自注意力与因果掩码 ====================


def attention_and_mask():
    print("=" * 60)
    print("第一部分：时间维度上的自注意力与因果掩码")
    print("=" * 60)
    m = nn.Transformer.generate_square_subsequent_mask(6)
    print("\n    把窗口内的 L 个交易日看作 L 个 token。因果掩码 M（L = 6 示例，0 = 可见，-inf = 屏蔽）：\n")
    print("              " + "  ".join(f"k={j}" for j in range(6)))
    for i in range(6):
        cells = ["  0 " if v == 0 else "-inf" for v in m[i].tolist()]
        print(f"      查询 q={i}  " + " ".join(cells))
    print("""
    第 i 行只能看到第 0..i 列 —— 位置 i 的输出只依赖 i 及之前的交易日。

    什么时候必须加掩码？
    ┌──────────────────────────────┬──────────────┬──────────────────────────────┐
    │ 任务形式                     │ 需要因果掩码 │ 原因                         │
    ├──────────────────────────────┼──────────────┼──────────────────────────────┤
    │ 读完整窗口，只预测窗口之后   │ 不需要       │ 窗口里全是过去               │
    │ 每个位置都预测"下一天"       │ 必须         │ 否则位置 i 能看到 i+1 的答案 │
    │ 编码器-解码器的解码器端      │ 必须         │ 自回归生成                   │
    └──────────────────────────────┴──────────────┴──────────────────────────────┘
    逐步预测能让一个窗口提供 L 个训练信号（比只训练最后一个位置更省数据），
    这正是本课 Transformer 的训练方式，所以掩码是必需的。
    """)


# ==================== 第二部分：研究进展与 DLinear ====================


def research_landscape():
    print("\n" + "=" * 60)
    print("第二部分：时序 Transformer 的研究进展与 DLinear")
    print("=" * 60)
    print("""
    ┌────────────────────┬─────────────────────────────────┬────────────────────────────────┐
    │ 方向               │ 代表工作                        │ 要点                           │
    ├────────────────────┼─────────────────────────────────┼────────────────────────────────┤
    │ 长序列效率         │ Informer、Autoformer（2021）    │ 稀疏注意力、序列分解           │
    │ 对 Transformer 的  │ DLinear（Zeng et al., AAAI 2023）│ 在常用长期预测基准上，线性模型 │
    │ 质疑               │                                 │ 能匹敌或超过许多 Transformer   │
    │ Patch 化           │ PatchTST（ICLR 2023）           │ 连续时间步切块作为 token        │
    │ 时序基础模型       │ TimesFM、Chronos、Moirai 等     │ 大规模预训练、零样本预测       │
    └────────────────────┴─────────────────────────────────┴────────────────────────────────┘

    DLinear 结构：x ──滑动平均──▶ 趋势 ─Linear(L→1)─┐
                  x ── − 趋势 ──▶ 残差 ─Linear(L→1)─┴─▶ 相加 ─▶ 预测

    这些基准多是电力、交通、天气等高信噪比、强周期数据；金融收益几乎没有稳定周期，
    优势很难直接迁移。另外，时序基础模型的预训练语料可能包含公开金融序列，
    在历史区间回测时存在与 LLM 类似的前视风险（见 10 课）。
    """)


# ==================== 第三部分：多资产数据 ====================


def build_dataset():
    print("\n" + "=" * 60)
    print("第三部分：多资产窗口数据")
    print("=" * 60)
    frames, synth = {}, []
    for code in ETFS:
        df = load_etf(code, "20130101", "20241231")
        synth.append(df.attrs.get("synthetic", False))
        c = df["adj_close"]
        frames[code] = pd.DataFrame({
            "r1": c.pct_change(),
            "gap": df["adj_open"] / c.shift(1) - 1,
            "intraday": c / df["adj_open"] - 1,
            "range": (df["high"] - df["low"]) / df["close"],
            "vol_z": np.log(df["volume"].where(df["volume"] > 0) / df["volume"].rolling(20).mean()),
        }).replace([np.inf, -np.inf], np.nan)
    train_rows = pd.concat([f.loc[:TRAIN_END] for f in frames.values()])
    mu, sd = train_rows.mean(), train_rows.std()  # 只用训练期、跨资产合并拟合
    X, Y, D, A = [], [], [], []
    for code, f in frames.items():
        z = ((f - mu) / sd).clip(-5, 5)
        target = z["r1"].shift(-1)  # 位置 i 的标签 = 第 i+1 天的收益（标准化后）
        arr, tgt, idx = z[FEATS].to_numpy(np.float32), target.to_numpy(np.float32), f.index
        for t in range(L - 1, len(f) - 1):
            w, y = arr[t - L + 1:t + 1], tgt[t - L + 1:t + 1]
            if np.isfinite(w).all() and np.isfinite(y).all():
                X.append(w)
                Y.append(y)
                D.append(idx[t])
                A.append(code)
    X, Y, D, A = np.stack(X), np.stack(Y), pd.DatetimeIndex(D), np.array(A)
    purge = pd.Timedelta(days=10)
    tr = np.where(D <= pd.Timestamp(TRAIN_END) - purge)[0]
    va = np.where((D > pd.Timestamp(TRAIN_END)) & (D <= pd.Timestamp(VALID_END) - purge))[0]
    te = np.where(D > pd.Timestamp(VALID_END))[0]
    print(f"\n    资产：{', '.join(f'{k}({v})' for k, v in ETFS.items())}"
          f"{'（⚠️ 含合成数据）' if any(synth) else ''}")
    print(f"    特征 {len(FEATS)} 个：{FEATS}；窗口 L = {L}；标准化参数只用训练期拟合")
    print("    标签：下一天收盘到收盘收益（为突出因果掩码而选；可交易口径见 01 课的 t+1 开盘 → t+2 开盘）")
    print(f"    样本：训练 {len(tr)}，验证 {len(va)}，测试 {len(te)}（段间留 10 个自然日间隔）")
    return X, Y, D, A, tr, va, te


# ==================== 第四部分：去掉掩码会发生什么 ====================


def mask_leak_experiment(X, Y, tr, va):
    print("\n" + "=" * 60)
    print("第四部分：实验 —— 去掉因果掩码会发生什么")
    print("=" * 60)
    print("\n    两个模型结构相同，只差一个掩码；都按\"验证集所有位置的平均 MSE\"早停（最常见的写法）")
    models = {}
    for causal in (True, False):
        set_seed(0)
        m = TSTransformer(len(FEATS), L, causal=causal)
        models[causal] = fit(m, X[tr], Y[tr], X[va], Y[va], seq2seq=True, epochs=10, seed=0, es_all=True)
    mse = {}
    with torch.no_grad():
        for name, idx in (("train", tr), ("valid", va)):
            xs, ys = torch.from_numpy(X[idx]), torch.from_numpy(Y[idx])
            mse[name] = {c: ((models[c](xs) - ys) ** 2).mean(0).numpy() / Y[idx].var(0) for c in models}
    print("\n    各位置的 MSE / 标签方差（= 1 表示毫无预测力，接近 0 表示几乎完美）：\n")
    rows = []
    for name, lab in (("train", "训练"), ("valid", "验证")):
        rows.append([f"{lab}：位置 0..{L - 2} 平均", mse[name][True][:-1].mean(), mse[name][False][:-1].mean()])
        rows.append([f"{lab}：位置 {L - 1}（真正的预测）", mse[name][True][-1], mse[name][False][-1]])
    print(pd.DataFrame(rows, columns=["位置", "有因果掩码", "无掩码"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    early_gap = mse["valid"][True][:-1].mean() - mse["valid"][False][:-1].mean()
    last_gap = mse["valid"][True][-1] - mse["valid"][False][-1]
    print(f"""
    本次结果：在验证集上，无掩码模型前面各位置的相对 MSE 比有掩码模型低 {early_gap:.3f}，
    而最后一个位置只差 {last_gap:+.3f}（正数 = 无掩码更好）。
    机制：无掩码时位置 i 的注意力能看到位置 i+1 的输入，而 i+1 的 r1 特征正是位置 i 的标签，
    模型学会"抄答案"。窗口最后一个位置之后没有数据可抄，真正需要的那个预测并不会因此变好；
    如果只盯着"所有位置的平均验证损失"，无掩码模型会显得好得多 —— 这就是结构性的前视泄漏。
    """)
    return mse["train"]


# ==================== 第五部分：与 DLinear、Ridge 的诚实比较 ====================


def compare_models(X, Y, D, A, tr, va, te):
    print("\n" + "=" * 60)
    print("第五部分：测试集比较 —— Transformer vs DLinear vs Ridge vs 零预测")
    print("=" * 60)
    preds = {"零预测": np.zeros(len(te))}
    ridge = Ridge(alpha=50.0).fit(X[tr].reshape(len(tr), -1), Y[tr][:, -1])
    preds["Ridge（展平窗口）"] = ridge.predict(X[te].reshape(len(te), -1))
    for s in (0, 1):
        set_seed(s)
        m = DLinear(len(FEATS), L)
        preds[f"DLinear seed={s}"] = last_pred(fit(m, X[tr], Y[tr], X[va], Y[va], False, epochs=30, lr=3e-3, seed=s),
                                               X[te], False)
    for s in (0, 1):  # 与第四部分不同：这里按"最后位置"的验证 MSE 早停，与 DLinear 口径一致
        set_seed(s)
        m = fit(TSTransformer(len(FEATS), L), X[tr], Y[tr], X[va], Y[va], seq2seq=True, seed=s)
        preds[f"Transformer seed={s}"] = last_pred(m, X[te], True)
    print(f"\n    参数量：Transformer {n_params(m):,}，DLinear {n_params(DLinear(len(FEATS), L)):,}，"
          f"Ridge {X.shape[1] * X.shape[2] + 1:,}")

    yt = Y[te][:, -1]
    rows = []
    for name, p in preds.items():
        has = np.std(p) > 0
        per_asset = [spearmanr(p[A[te] == a], yt[A[te] == a]).statistic for a in ETFS] if has else []
        rows.append([name, 1 - np.sum((yt - p) ** 2) / np.sum(yt ** 2),
                     spearmanr(p, yt).statistic if has else np.nan, np.mean(per_asset) if has else np.nan,
                     ((p > 0) == (yt > 0)).mean() if has else np.nan])
    table = pd.DataFrame(rows, columns=["模型", "R²_oos", "合并 IC", "各资产 IC 均值", "方向准确率"])
    print(f"\n    测试期 {D[te].min().date()} ~ {D[te].max().date()}，{len(te)} 个样本"
          f"（上涨占比 {fmt_pct((yt > 0).mean(), 1)}）\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.dropna(subset=["合并 IC"]).sort_values("合并 IC", ascending=False).iloc[0]
    t_best = best["合并 IC"] * np.sqrt(len(te))
    verdict = ("即便是最好的一个，|t| 也小于 2：没有任何模型能与零预测区分开" if abs(t_best) < 2
               else "最好的模型 |t| ≥ 2，但它是从多个模型 × 种子中挑出来的，需要多重检验修正")
    print(f"""
    本次运行合并 IC 最高的是【{best['模型']}】（IC = {best['合并 IC']:.4f}，近似 t = {t_best:.2f}）。
    {verdict}。
    注意：
      · 同一模型换一个随机种子，IC 可能变化很大 —— 对比两行 seed 的差异
      · 样本在时间上重叠、资产之间相关，有效样本数远小于 {len(te)}，上面的 t 值偏乐观
      · 深度模型若没有稳定地超过线性基线，就不值得付出额外的复杂度与调参（=隐性试验）成本
    """)
    return table


# ==================== 第六部分：小结与图表 ====================


def summary_and_plot(mse, table):
    print("\n" + "=" * 60)
    print("第六部分：金融场景中更有意义的注意力用法")
    print("=" * 60)
    print("""
    · 截面注意力：同一天的股票之间做注意力（行业、供应链联动），如 Qlib 的 GATs、HIST
    · 时间 + 截面双重注意力：先编码每只股票的历史，再在截面上交互
    · 多模态：价量序列 + 新闻文本 embedding（06 课的文本因子是一个起点）

    检查清单：逐步预测必须加因果掩码 → 先和 DLinear / Ridge 比 → 多种子 → 有效样本数要打折扣
    """)
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(range(L), mse[True], marker="o", ms=3, label="有因果掩码")
    axes[0].plot(range(L), mse[False], marker="o", ms=3, label="无掩码（泄漏）")
    axes[0].set_xlabel("窗口内位置")
    axes[0].set_ylabel("训练集 MSE / 标签方差")
    axes[0].set_title("去掉掩码：前面的位置在'抄答案'")
    axes[0].legend()
    t = table.dropna(subset=["合并 IC"])
    axes[1].barh(t["模型"], t["合并 IC"], color="tab:gray")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("测试集合并 IC")
    path = save_figure(fig, "02_transformer_mask_and_ic.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    torch.set_num_threads(4)
    attention_and_mask()
    research_landscape()
    X, Y, D, A, tr, va, te = build_dataset()
    mse = mask_leak_experiment(X, Y, tr, va)
    table = compare_models(X, Y, D, A, tr, va, te)
    summary_and_plot(mse, table)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 只训练最后一个位置（seq2seq=False）时去掉掩码，验证此时不再存在泄漏，并比较测试 IC
    2. 实现 PatchTST 的 patch 化：把 30 天切成 6 个 5 天的 patch 作为 token，与本课模型比较
    3. 把标签换成 01 课的"t+1 开盘 → t+2 开盘"收益，重新检查无掩码模型能抄到多少答案
    4. 用 5 个随机种子训练 Transformer 与 DLinear，画出两者 IC 的分布，判断差异是否显著
    """)


if __name__ == "__main__":
    main()
