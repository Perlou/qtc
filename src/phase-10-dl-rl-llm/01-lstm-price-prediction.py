"""
LSTM 时序预测：从"预测价格"的假象到可检验的收益预测
====================================================

学习目标：
    1. 亲手复现"LSTM 预测股价 R² 很高"的假象，并用朴素基线（明天 = 今天）戳穿它
    2. 掌握无泄漏的窗口构造：窗口只含 t 日及之前的数据，标签是 t+1 开盘 → t+2 开盘的收益
    3. 理解"标准化只在训练集上拟合"与"训练/验证/测试之间留间隔（purge）"
    4. 把任务改成预测收益/方向，用 R²_oos、IC、方向准确率、扣费策略收益与基线比较

核心概念：
    - 价格近似随机游走：p_{t+1} ≈ p_t，预测价格的模型只要"抄昨天"就能得到很高的 R²
    - 朴素基线（persistence）：预测价格 = 今天的价格，等价于预测收益 = 0
    - R²_oos = 1 − Σ(y − ŷ)² / Σy²：相对"预测收益为 0"的改进，金融里常常 ≤ 0
    - 多随机种子：深度模型在低信噪比数据上结果波动大，必须报告均值与离散程度

前置知识：
    - 深度学习课程（PyTorch、LSTM）
    - Phase 4（回测与交易成本）、Phase 9（时间序列切分、Purged CV）

环境要求：
    - pip install torch pandas numpy scikit-learn scipy matplotlib（CPU 即可，单次运行约 30 秒）
    - 数据：沪深 300 ETF（510300）日线；无网络时可用 QTC_OFFLINE=1 运行（先读缓存，无缓存用合成数据）
"""

import copy
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from ai_tools import ETF_COST, SUBDIR, fmt_pct, load_etf, perf_stats, set_seed
from trading_env import FeatureScaler, make_features
from utils.plotting import save_figure, setup_chinese_font

load_dotenv()

L = 20  # 窗口长度（交易日）
TRAIN_END, VALID_END = "2019-12-31", "2020-12-31"  # 训练 2013–2019，验证 2020，测试 2021–2024
GAP = 5  # 各段之间丢弃的样本数（≥ 标签期限，Phase 9 的 purging）
SEEDS = (0, 1, 2)


# ==================== 工具：窗口、切分、模型 ====================


def make_windows(feat: np.ndarray, label: np.ndarray, dates: pd.DatetimeIndex, L: int):
    """X[k] = feat[t-L+1 : t+1]（只到 t，含 t）；y[k] = label[t]（t 之后才揭晓的未来值）"""
    X, y, d = [], [], []
    for t in range(L - 1, len(feat)):
        win = feat[t - L + 1:t + 1]
        if np.isfinite(label[t]) and np.isfinite(win).all():
            X.append(win)
            y.append(label[t])
            d.append(dates[t])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(d)


def split_masks(d: pd.DatetimeIndex):
    """按窗口结束日 t 切分；每段末尾丢掉 GAP 个样本，防止标签期跨进下一段"""
    tr = np.where(d <= pd.Timestamp(TRAIN_END))[0]
    va = np.where((d > pd.Timestamp(TRAIN_END)) & (d <= pd.Timestamp(VALID_END)))[0]
    te = np.where(d > pd.Timestamp(VALID_END))[0]
    return tr[:-GAP], va[:-GAP], te


class LSTMReg(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, L, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # 取最后一步的隐状态


def train_model(model, Xtr, ytr, Xva, yva, epochs=40, lr=2e-3, batch=64, patience=6, seed=0):
    """小批量 Adam + 验证集早停，返回验证损失最低时的权重"""
    Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    best, state, wait, ep = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xva), yva).item()
        if vl < best - 1e-7:
            best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return model, ep


def predict(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


# ==================== 第一部分：LSTM 与金融序列 ====================


def lstm_recap(df):
    print("=" * 60)
    print("第一部分：LSTM 回顾与金融序列的特点")
    print("=" * 60)
    tag = "⚠️ 合成数据" if df.attrs.get("synthetic") else "真实数据"
    print(f"\n    数据：沪深 300 ETF（510300），{df.index[0].date()} ~ {df.index[-1].date()}，"
          f"{len(df)} 个交易日（{tag}）")
    print(f"    切分：训练 ≤ {TRAIN_END}，验证 ≤ {VALID_END}，测试之后；段间丢弃 {GAP} 个样本")
    print("""
    h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1}),   ŷ = W·h_L + b    （门控缓解梯度消失）

    ┌──────────────────────┬──────────────────────────┬──────────────────────────┐
    │                      │ NLP / 语音               │ 日频金融收益             │
    ├──────────────────────┼──────────────────────────┼──────────────────────────┤
    │ 信噪比               │ 高（语法有强规律）       │ 极低（可预测部分很小）   │
    │ 样本量               │ 百万~十亿                │ 单资产 10 年约 2400 个   │
    │ 平稳性               │ 语言规律长期稳定         │ 规律会变化、会被套利掉   │
    │ 评估基线             │ 人类水平 / 旧模型        │ "明天 = 今天"、历史均值  │
    └──────────────────────┴──────────────────────────┴──────────────────────────┘

    本课会先犯一次"经典错误"，再一步步改正。
    """)


# ==================== 第二部分：预测价格的假象 ====================


def price_illusion(df):
    print("\n" + "=" * 60)
    print("第二部分：预测价格的假象 —— 与朴素基线比较")
    print("=" * 60)
    price = df["adj_close"].to_numpy()
    dates = df.index
    label = np.append(price[1:], np.nan)  # 标签：明天的收盘价
    tr_mu, tr_sd = price[dates <= pd.Timestamp(TRAIN_END)].mean(), price[dates <= pd.Timestamp(TRAIN_END)].std()
    z = ((price - tr_mu) / tr_sd)[:, None]
    X, y, d = make_windows(z, (label - tr_mu) / tr_sd, dates, L)
    tr, va, te = split_masks(d)

    set_seed(0)
    model, ep = train_model(LSTMReg(1), X[tr], y[tr], X[va], y[va], seed=0)
    pred = predict(model, X[te]) * tr_sd + tr_mu
    actual = y[te] * tr_sd + tr_mu
    today = X[te][:, -1, 0] * tr_sd + tr_mu  # 窗口最后一天 = 今天的价格 = 朴素预测

    def r2(p):
        return 1 - np.sum((actual - p) ** 2) / np.sum((actual - actual.mean()) ** 2)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    dir_true = np.sign(actual - today)
    rows = [
        ["LSTM（预测明天价格）", rmse(pred, actual), r2(pred), (np.sign(pred - today) == dir_true).mean()],
        ["朴素基线：明天 = 今天", rmse(today, actual), r2(today), np.nan],
    ]
    print(f"\n    LSTM 训练 {ep} 轮（早停）；测试集 {len(te)} 个样本\n")
    print(pd.DataFrame(rows, columns=["模型", "RMSE", "R²", "涨跌方向准确率"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n    LSTM 预测值与【今天】价格的 RMSE：{rmse(pred, today):.4f}")
    print(f"    LSTM 预测值与【明天】价格的 RMSE：{rmse(pred, actual):.4f}")
    closer = "今天" if rmse(pred, today) < rmse(pred, actual) else "明天"
    beat = "低于" if rmse(pred, actual) < rmse(today, actual) else "不低于"
    print(f"""
    解读（以上为本次实际计算结果）：
      · 两者 R² 都很高 —— 因为价格本身高度自相关，高 R² 不代表任何预测能力
      · LSTM 的 RMSE {beat}朴素基线；它的预测更接近【{closer}】的价格
      · 判断"预测方向"才是交易真正需要的，上表的方向准确率才值得关注
      · 测试期价格若超出训练期范围，标准化后的输入落在模型没见过的区域（非平稳），
        这也是"预测价格"方案的另一个硬伤

    结论：预测价格的曲线"看起来很准"，本质上只是把今天的价格向右平移一天。
    """)
    return pd.DataFrame({"actual": actual, "lstm": pred, "naive": today}, index=d[te] + pd.offsets.BDay(1))


# ==================== 第三部分：无泄漏的窗口与标准化 ====================


def leak_free_windows(df):
    print("\n" + "=" * 60)
    print("第三部分：无泄漏的窗口构造与标准化")
    print("=" * 60)
    feat = make_features(df)
    o = df["adj_open"]
    label = o.shift(-2) / o.shift(-1) - 1  # t+1 开盘买入 → t+2 开盘卖出
    valid = feat.notna().all(axis=1) & label.notna()
    feat, label = feat[valid], label[valid]
    dates = feat.index

    scaler = FeatureScaler().fit(feat[dates <= pd.Timestamp(TRAIN_END)])
    leaky = FeatureScaler().fit(feat)  # ❌ 错误示范：全样本拟合
    print("\n    标准化参数对比（vol20 = 20 日波动率特征）：")
    print(f"      只用训练期拟合：均值 {scaler.mean_['vol20']:.5f}，标准差 {scaler.std_['vol20']:.5f}")
    print(f"      全样本拟合    ：均值 {leaky.mean_['vol20']:.5f}，标准差 {leaky.std_['vol20']:.5f}")
    print("      全样本拟合时，训练样本的取值里混进了测试期的波动水平 —— 这是信息泄漏，"
          "\n      在非平稳数据上会让验证结果偏乐观且难以察觉。")

    X, y, d = make_windows(scaler.transform(feat).to_numpy(), label.to_numpy(), dates, L)
    tr, va, te = split_masks(d)
    cal = df.index  # 完整交易日历（标签的买卖日在日历上取 t+1、t+2）
    print("\n    测试集前 3 个样本的时间结构：")
    rows = []
    for k in te[:3]:
        t, t0 = cal.get_loc(d[k]), dates.get_loc(d[k]) - L + 1
        rows.append([dates[t0].date(), cal[t].date(), cal[t + 1].date(), cal[t + 2].date()])
    print(pd.DataFrame(rows, columns=["窗口起点", "窗口终点 t", "买入 t+1 开盘", "卖出 t+2 开盘"]).to_string(index=False))
    ok = all(cal[cal.get_loc(t) + 1] > t for t in d)
    print(f"\n    自动检查：所有样本的买入日都晚于窗口终点 → {'通过 ✅' if ok else '失败 ❌'}")
    print(f"    样本数：训练 {len(tr)}，验证 {len(va)}，测试 {len(te)}（段末各丢弃 {GAP} 个做 purge）")
    print("""
    ⚠️ 常见泄漏写法：
      feat[t-L+1 : t+2]          多取了一天（t+1）
      label = close.shift(-1)/close - 1 却用 t 日收盘成交 —— 信号与成交用的是同一个价格
      scaler.fit(全部数据)         如上所示
      train_test_split(shuffle=True)  时间序列被打乱，相邻且高度相关的样本跨越训练/测试
    """)
    return X, y, d, tr, va, te


# ==================== 第四部分：改为预测收益与方向 ====================


def predict_returns(X, y, d, tr, va, te):
    print("\n" + "=" * 60)
    print("第四部分：预测收益 / 方向，并与基线比较")
    print("=" * 60)
    sd = y[tr].std()  # 标签缩放（只用训练集）：让网络输出的量级接近 1
    preds = {"零预测（= 明天价格等于今天）": np.zeros(len(te)), "历史均值": np.full(len(te), y[tr].mean())}

    ridge = Ridge(alpha=10.0).fit(X[tr].reshape(len(tr), -1), y[tr] / sd)
    preds["Ridge（线性基线）"] = ridge.predict(X[te].reshape(len(te), -1)) * sd

    lstm_preds, epochs = [], []
    for s in SEEDS:
        set_seed(s)
        m, ep = train_model(LSTMReg(X.shape[2]), X[tr], y[tr] / sd, X[va], y[va] / sd, seed=s)
        lstm_preds.append(predict(m, X[te]) * sd)
        epochs.append(ep)
    for s, p in zip(SEEDS, lstm_preds):
        preds[f"LSTM seed={s}"] = p
    preds["LSTM 三种子平均"] = np.mean(lstm_preds, axis=0)
    print(f"\n    LSTM 早停轮数：{epochs}")

    yt = y[te]
    rows, navs = [], {}
    for name, p in preds.items():
        pos = (p > 0).astype(float)
        trade = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pd.Series(pos * yt - ETF_COST * trade, index=d[te])
        ps = perf_stats(net)
        ic = spearmanr(p, yt).statistic if np.std(p) > 0 else np.nan
        acc = ((p > 0) == (yt > 0)).mean() if np.std(p) > 0 else np.nan
        rows.append([name, 1 - np.sum((yt - p) ** 2) / np.sum(yt ** 2), ic, acc, ps["年化收益"], ps["夏普"], trade.sum()])
        navs[name] = (1 + net).cumprod()
    bh = pd.Series(yt, index=d[te])
    rows.append(["买入持有", np.nan, np.nan, (yt > 0).mean(), perf_stats(bh)["年化收益"], perf_stats(bh)["夏普"], 1.0])
    navs["买入持有"] = (1 + bh).cumprod()
    table = pd.DataFrame(rows, columns=["模型", "R²_oos", "IC", "方向准确率", "策略年化", "策略夏普", "换手次数"])
    print(f"\n    测试集 {len(te)} 天；策略 = 预测 > 0 时满仓、否则空仓，t+1 开盘成交，单边成本 {ETF_COST:.2%}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lstm_ic = [spearmanr(p, yt).statistic for p in lstm_preds]
    up = (yt > 0).mean()
    ens_ic = spearmanr(preds["LSTM 三种子平均"], yt).statistic
    t_ic = ens_ic * np.sqrt(len(yt))
    verdict = "统计上无法与 0 区分" if abs(t_ic) < 2 else "超过常用阈值 2（仍需多重检验与样本外确认）"
    print(f"\n    LSTM 平均预测的 IC = {ens_ic:.4f}，近似 t 值 = IC × √N = {t_ic:.2f} → {verdict}")
    print(f"""
    读表要点（数字均为本次运行的实际结果）：
      · R²_oos 以"预测收益为 0"为基准；≤ 0 表示模型的 MSE 还不如什么都不预测
      · 方向准确率要和"天天看涨"的比例 {fmt_pct(up, 1)} 比，而不是和 50% 比
      · 三个种子的 IC：{', '.join(f'{v:.4f}' for v in lstm_ic)}；
        种子间离散度（标准差 {np.std(lstm_ic):.4f}）若与 IC 本身同量级，说明结果主要是噪声
      · "历史均值"预测恒为正 → 策略等价于一直持有；零预测恒为 0 → 一直空仓
    """)
    return navs


# ==================== 第五部分：总结与图表 ====================


def summary_and_plot(price_df, navs):
    print("\n" + "=" * 60)
    print("第五部分：小结")
    print("=" * 60)
    print("""
    ┌────────────────────────────┬──────────────────────────────────────────────┐
    │ 做法                       │ 要点                                         │
    ├────────────────────────────┼──────────────────────────────────────────────┤
    │ 预测对象                   │ 收益 / 方向 / 截面排名，而不是价格           │
    │ 基线                       │ 零预测、历史均值、线性模型（再加 GBDT）       │
    │ 数据切分                   │ 按时间；段间 purge；标准化只拟合训练集       │
    │ 评估                       │ R²_oos、IC、方向准确率（对比上涨比例）、扣费 │
    │ 稳定性                     │ 多随机种子，报告均值与离散程度               │
    │ 更有希望的方向             │ 多股票共享模型（截面化，见 03 课）、高频数据 │
    └────────────────────────────┴──────────────────────────────────────────────┘
    """)
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    tail = price_df.iloc[-80:]
    axes[0].plot(tail.index, tail["actual"], label="实际价格", color="black", lw=1.6)
    axes[0].plot(tail.index, tail["lstm"], label="LSTM 预测", color="tab:red", lw=1.2)
    axes[0].plot(tail.index, tail["naive"], label="朴素：明天 = 今天", color="tab:blue", lw=1, ls="--")
    axes[0].set_title("预测价格：曲线贴合 ≈ 向右平移一天")
    axes[0].legend()
    for name in ["买入持有", "Ridge（线性基线）", "LSTM 三种子平均"]:
        axes[1].plot(navs[name].index, navs[name].to_numpy(), label=name)
    axes[1].set_title("测试期净值（扣费）")
    axes[1].legend()
    fig.autofmt_xdate()
    path = save_figure(fig, "01_lstm_price_vs_return.png", subdir=SUBDIR)
    print(f"📊 图表已保存：{path}")


def main():
    df = load_etf("510300", "20130101", "20241231")
    lstm_recap(df)
    price_df = price_illusion(df)
    X, y, d, tr, va, te = leak_free_windows(df)
    navs = predict_returns(X, y, d, tr, va, te)
    summary_and_plot(price_df, navs)

    print("\n" + "=" * 60)
    print("📝 练习")
    print("=" * 60)
    print("""
    1. 把第二部分的标准化改成"全样本 MinMax"（网上教程的常见写法），比较测试期的 RMSE 与 R²，
       并解释为什么泄漏在这里不一定让结果"变好看"，却会让结论不可信
    2. 把 LSTM 改成二分类（BCEWithLogitsLoss 预测涨跌），比较方向准确率与"天天看涨"的基线
    3. 把 SEEDS 扩展到 10 个，画出 IC 的分布；IC 的 95% 区间是否包含 0？
    4. 把窗口长度 L 改为 5、60，观察早停轮数与测试 IC 的变化，并思考"调 L 本身也是一次试验"
    """)


if __name__ == "__main__":
    main()
