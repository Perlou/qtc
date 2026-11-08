"""
Phase 9 辅助模块：机器学习量化工具
==================================

本模块不是课程文件，而是 01–10 课共用的"ML 研究流水线"。每个函数的原理都在对应课程中
讲解，这里集中实现一份，保证各课口径一致：

    模型工厂      make_regressor / make_classifier / gain_importance   （05、06 课）
    数据集        load_ml_data：Phase 7 面板 → 月度截面特征 + 标签     （02、05 课）
    截面工具      cs_rank / daily_rank_ic / ic_stats                    （02、05 课）
    交叉验证      PurgedKFold / purged_date_splits                      （04 课）
    滚动训练      walk_forward                                           （05、10 课）
    组合构建      topk_equal / topk_dropout / score_weighted / backtest_weights（07 课）

统一约定（与 Phase 7 相同）：
    - 长表：MultiIndex (datetime, instrument)，与 Qlib 一致
    - 调仓日 t = 月末最后一个交易日；特征只用 t 日收盘及以前的数据
    - 标签 = t+1 开盘买入 → 下一调仓日 t' 的次日开盘卖出的收益
      ⚠️ 因此 t 期标签要到 t' + 1 日开盘才"实现"：在 t' 收盘训练时，最新可用的标签是 t'' = t' 的上一期
         → walk_forward 的 gap = 1（与 Phase 7 第 09 课 IC_LAG = 2 是同一件事）

LightGBM / XGBoost 在 macOS 上需要 OpenMP 运行库（brew install libomp）。导入失败时自动回退到
sklearn 的 HistGradientBoosting —— 它同样是"直方图 + 逐叶生长"的 GBDT，概念与 LightGBM 一致。
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 小数据上 OpenMP 线程过多反而更慢（10 线程比 4 线程慢数倍），统一限制线程数。
# threadpool_limits 只对"已加载"的 OpenMP 运行库生效，所以先导入 sklearn.ensemble
N_THREADS = min(4, os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
import sklearn.ensemble  # noqa: E402,F401
from threadpoolctl import threadpool_limits  # noqa: E402

_THREAD_LIMITER = threadpool_limits(limits=N_THREADS)

PHASE_DIR = Path(__file__).resolve().parent
PHASE7_DIR = PHASE_DIR.parent / "phase-7-multi-factor"
for _p in (PHASE_DIR.parent, PHASE7_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import factor_data  # noqa: E402
import factor_tools  # noqa: E402

# ==================== 模型工厂 ====================

try:  # macOS 缺 libomp 时抛出的是 OSError 而不是 ImportError，所以捕获 Exception
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:  # noqa: BLE001
    lgb = None
    HAS_LGB = False

_WARNED = {"lgb": False}

# 低信噪比场景的保守参数（LightGBM 命名）；回退时自动映射到 sklearn 参数
GBDT_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=100,  # 每个叶子至少 100 个样本：低信噪比下最重要的正则
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10.0,
)


def gbdt_backend() -> str:
    return "LightGBM" if HAS_LGB else "sklearn HistGradientBoosting"


def _warn_fallback() -> None:
    if not HAS_LGB and not _WARNED["lgb"]:
        print("⚠️ 未检测到 lightgbm（macOS 需 brew install libomp），使用 sklearn HistGradientBoosting 代替")
        _WARNED["lgb"] = True


def _hgb_kwargs(p: dict, seed: int) -> dict:
    """LightGBM 参数 → HistGradientBoosting 参数（HGB 没有行采样 subsample，忽略之）"""
    return dict(
        max_iter=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_leaf_nodes=p["num_leaves"],
        min_samples_leaf=p["min_child_samples"],
        l2_regularization=p["reg_lambda"],
        max_features=p["colsample_bytree"],  # HGB 是"每次分裂"随机选特征，类似 feature_fraction_bynode
        early_stopping=False,  # ⚠️ HGB 内置早停用"随机"切出的验证集 → 金融数据上会泄漏，必须关掉
        random_state=seed,
    )


def make_regressor(seed: int = 42, **overrides):
    """GBDT 回归器：优先 LightGBM，失败时回退 HistGradientBoostingRegressor"""
    p = {**GBDT_PARAMS, **overrides}
    if HAS_LGB:
        return lgb.LGBMRegressor(objective="regression", subsample_freq=1, random_state=seed, verbose=-1, n_jobs=N_THREADS, **p)
    _warn_fallback()
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**_hgb_kwargs(p, seed))


def make_classifier(seed: int = 42, **overrides):
    """GBDT 二分类器：优先 LightGBM，失败时回退 HistGradientBoostingClassifier"""
    p = {**GBDT_PARAMS, **overrides}
    if HAS_LGB:
        return lgb.LGBMClassifier(objective="binary", subsample_freq=1, random_state=seed, verbose=-1, n_jobs=N_THREADS, **p)
    _warn_fallback()
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**_hgb_kwargs(p, seed))


def gain_importance(model, names) -> pd.Series:
    """MDI（gain）重要性：每个特征在所有分裂上带来的损失下降之和，归一化为占比"""
    if HAS_LGB and hasattr(model, "booster_"):
        g = model.booster_.feature_importance(importance_type="gain").astype(float)
    else:
        # HistGradientBoosting 不提供 feature_importances_，从内部树结构累加 split gain
        # （_predictors 是私有属性，sklearn 1.x 一直存在；升级后若失效请改用 MDA）
        g = np.zeros(len(names))
        for iteration in model._predictors:
            for tree in iteration:
                nodes = tree.nodes
                split = nodes["is_leaf"] == 0
                np.add.at(g, nodes["feature_idx"][split], nodes["gain"][split])
    s = pd.Series(g, index=list(names))
    return s / s.sum() if s.sum() > 0 else s


# ==================== 数据集 ====================

FEATURE_LABELS = {
    # 价量
    "ret_1m": "1 月收益（反转）",
    "ret_3m": "3 月收益",
    "ret_12_1": "12-1 动量",
    "vol_20": "20 日波动率",
    "vol_60": "60 日波动率",
    "skew_60": "60 日偏度",
    "max_ret_20": "20 日最大日收益",
    "turn_20": "20 日换手率",
    "turn_ratio": "换手 20/120 比",
    "amihud_20": "Amihud 非流动性",
    "pv_corr_20": "量价相关 20 日",
    "dist_high_250": "距 250 日高点",
    "log_mcap": "对数流通市值",
    # 基本面（PIT，按公告日对齐）
    "ep": "EP（TTM）",
    "bp": "BP",
    "roe": "ROE（TTM）",
    "np_yoy": "单季净利同比",
}
FEATURE_GROUP = {
    "ret_1m": "价量", "ret_3m": "价量", "ret_12_1": "价量", "vol_20": "价量", "vol_60": "价量",
    "skew_60": "价量", "max_ret_20": "价量", "turn_20": "价量", "turn_ratio": "价量", "amihud_20": "价量",
    "pv_corr_20": "价量", "dist_high_250": "价量", "log_mcap": "规模",
    "ep": "基本面", "bp": "基本面", "roe": "基本面", "np_yoy": "基本面",
}
FEATURES = list(FEATURE_LABELS)


def wide_to_long(frames: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """{名字: 宽表(index=日期, columns=代码)} → 长表 MultiIndex(datetime, instrument)"""
    idx = pd.MultiIndex.from_product([dates, codes.astype(str)], names=["datetime", "instrument"])
    return pd.DataFrame({k: v.reindex(index=dates, columns=codes).to_numpy().ravel() for k, v in frames.items()},
                        index=idx)


def daily_price_features(panel) -> dict[str, pd.DataFrame]:
    """日频价量特征宽表（全部是因果的滚动计算：t 日的值只用 t 日及以前的数据）"""
    px = panel.adj_close.ffill()
    r = panel.daily_ret
    turn = panel.turnover
    turn_20 = turn.rolling(20, min_periods=10).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        dlogv = np.log(panel.volume.where(panel.volume > 0)).diff()
        illiq = r.abs() / (panel.amount / 1e8)
    return {
        "ret_1m": px / px.shift(21) - 1,
        "ret_3m": px / px.shift(63) - 1,
        "ret_12_1": px.shift(21) / px.shift(252) - 1,
        "vol_20": r.rolling(20, min_periods=15).std(),
        "vol_60": r.rolling(60, min_periods=40).std(),
        "skew_60": r.rolling(60, min_periods=40).skew(),
        "max_ret_20": r.rolling(20, min_periods=15).max(),
        "turn_20": turn_20,
        "turn_ratio": turn_20 / turn.rolling(120, min_periods=60).mean(),
        "amihud_20": illiq.rolling(20, min_periods=10).mean(),
        "pv_corr_20": r.rolling(20, min_periods=15).corr(dlogv),
        "dist_high_250": px / px.rolling(250, min_periods=120).max() - 1,
        "log_mcap": np.log(panel.mcap),
    }


def cs_rank(df: pd.DataFrame | pd.Series, fill: float | None = 0.0):
    """每个截面（datetime）内排名，映射到 [-0.5, 0.5]；缺失填 0（= 截面中位数）"""
    out = df.groupby(level="datetime").rank(pct=True) - 0.5
    return out.fillna(fill) if fill is not None else out


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level="datetime")
    return (df - g.transform("mean")) / g.transform("std")


@dataclass
class MLData:
    """月度截面数据集（长表只保留"调仓日可交易 且 标签已知"的行）"""

    panel: object
    dates: pd.DatetimeIndex  # 调仓日
    raw: pd.DataFrame  # 原始特征（未标准化）
    X: pd.DataFrame  # 截面 rank 后的特征 [-0.5, 0.5]
    ret: pd.Series  # 持有期收益（标签原值）
    y: pd.Series  # 截面 rank 后的标签 [-0.5, 0.5]
    ret_wide: pd.DataFrame  # 持有期收益宽表（index = 调仓日）
    mask_wide: pd.DataFrame  # 可交易掩码宽表
    daily_feats: dict = field(default_factory=dict, repr=False)

    @property
    def is_synthetic(self) -> bool:
        return bool(self.panel.is_synthetic)

    @property
    def date_index(self) -> pd.Index:
        return self.X.index.get_level_values("datetime")


_CACHE: dict = {}


def load_panel(verbose: bool = True):
    """
    加载 Phase 7 面板。与 utils.data 的约定一致：QTC_OFFLINE=1 时先读本地缓存，缓存缺失才用合成数据。
    （Phase 7 的 factor_data.load_panel 在离线模式下直接使用合成数据，这里在其外层补上"先读缓存"）
    """
    files = [factor_data.PANEL_FILE, factor_data.INFO_FILE, factor_data.FIN_FILE, factor_data.BENCH_FILE]
    if factor_data._PANEL is None and factor_data.is_offline() and all(p.exists() for p in files):
        factor_data._PANEL = factor_data._load_cached()
        if verbose:
            p = factor_data._PANEL
            print(f"📦 面板（离线模式，读取本地缓存的真实数据）：{len(p.codes)} 只股票，"
                  f"{p.dates[0].date()} ~ {p.dates[-1].date()}，{len(p.dates)} 个交易日")
        return factor_data._PANEL
    return factor_data.load_panel(verbose=verbose)


def load_ml_data(verbose: bool = True) -> MLData:
    """Phase 7 面板 → 月度截面特征与标签（同一进程内只构建一次）"""
    if "data" in _CACHE:
        return _CACHE["data"]
    panel = load_panel(verbose=verbose)
    dates = factor_tools.rebalance_dates(panel)
    cal = panel.dates
    pos = cal.get_indexer(dates)

    daily = daily_price_features(panel)
    sampled = {k: v.iloc[pos].set_axis(dates) for k, v in daily.items()}
    # 基本面：按实际公告日对齐（公告日 < t 才可用），见 Phase 7 第 08 课
    pit = factor_tools.pit_merge(panel, dates, ["net_profit_ttm", "equity", "np_q_yoy"], how="ann")
    tmc = panel.total_mcap.iloc[pos].set_axis(dates)
    equity = pit["equity"].where(pit["equity"] > 0)
    sampled["ep"] = pit["net_profit_ttm"] / tmc
    sampled["bp"] = equity / tmc
    sampled["roe"] = pit["net_profit_ttm"] / equity
    sampled["np_yoy"] = pit["np_q_yoy"].clip(-5, 5)

    mask = factor_tools.tradable_mask(panel, dates)
    ret_w = factor_tools.period_returns(panel, dates)
    codes = panel.codes
    raw = wide_to_long({k: sampled[k] for k in FEATURES}, dates, codes).replace([np.inf, -np.inf], np.nan)
    lab = wide_to_long({"ret": ret_w, "mask": mask.astype(float)}, dates, codes)
    keep = (lab["mask"] == 1.0) & lab["ret"].notna()
    raw = raw[keep.to_numpy()]
    ret = lab.loc[keep.to_numpy(), "ret"]
    X = cs_rank(raw)
    y = cs_rank(ret, fill=None)
    data = MLData(panel=panel, dates=dates, raw=raw, X=X, ret=ret, y=y, ret_wide=ret_w, mask_wide=mask,
                  daily_feats=daily)
    _CACHE["data"] = data
    return data


def phase7_composite(data: MLData) -> tuple[dict, pd.DataFrame]:
    """Phase 7 的 8 个预处理（去极值、标准化、行业市值中性化）后的因子，及其长表"""
    if "p7" not in _CACHE:
        dates, mask, ret, raw, clean = factor_tools.research_inputs(data.panel)
        long = wide_to_long(clean, dates, data.panel.codes).reindex(data.X.index)
        _CACHE["p7"] = (clean, long)
    return _CACHE["p7"]


# ==================== 截面评估 ====================


def daily_rank_ic(pred: pd.Series, label: pd.Series, min_obs: int = 20) -> pd.Series:
    """逐期截面 Spearman 相关（Rank IC），返回 index = datetime"""
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    r = df.groupby(level="datetime").rank()
    rc = r - r.groupby(level="datetime").transform("mean")
    g = rc.groupby(level="datetime")
    num = (rc["p"] * rc["y"]).groupby(level="datetime").sum()
    den = np.sqrt((rc["p"] ** 2).groupby(level="datetime").sum() * (rc["y"] ** 2).groupby(level="datetime").sum())
    ic = num / den
    return ic.where(g.size() >= min_obs).dropna()


def ic_stats(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    return pd.Series({
        "RankIC": ic.mean(),
        "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "t值": ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan,
        "IC>0": (ic > 0).mean(),
        "期数": len(ic),
    })


def ic_frame(rows: dict) -> pd.DataFrame:
    """{名字: ic_stats 结果} → 表格（期数显示为整数）"""
    df = pd.DataFrame(rows).T
    if "期数" in df:
        df["期数"] = df["期数"].astype(int)
    return df


def long_short_spread(pred: pd.Series, ret: pd.Series, q: float = 0.2) -> pd.Series:
    """每期：预测最高 q 分位的平均收益 − 最低 q 分位（未扣成本）"""
    df = pd.DataFrame({"p": pred, "r": ret}).dropna()
    pct = df.groupby(level="datetime")["p"].rank(pct=True)
    top = df["r"].where(pct > 1 - q).groupby(level="datetime").mean()
    bot = df["r"].where(pct <= q).groupby(level="datetime").mean()
    return top - bot


# ==================== 交叉验证 ====================


class PurgedKFold:
    """
    事件型 Purged K-Fold（AFML 第 7 章）
    t1: pd.Series，index = 样本开始时间 t0（已排序），值 = 标签结束时间 t1
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None, embargo_pct: float = 0.01):
        self.n_splits, self.t1, self.embargo_pct = n_splits, t1, embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        t0 = self.t1.index.to_numpy()
        t1 = self.t1.to_numpy()
        idx = np.arange(n)
        embargo = int(n * self.embargo_pct)
        for test_idx in np.array_split(idx, self.n_splits):
            start, end = test_idx[0], test_idx[-1]
            test_t0, test_t1 = t0[start], t1[test_idx].max()
            train = ~((t0 <= test_t1) & (t1 >= test_t0))  # purge：标签区间与测试区间有交集
            train[end + 1:end + 1 + embargo] = False  # embargo：测试块之后再隔离一段
            train[test_idx] = False
            yield idx[train], test_idx


def purged_date_splits(dates: np.ndarray, n_splits: int = 5, horizon: int = 1, embargo: int = 1):
    """面板版：按日期分块；测试块前后各删 horizon 个日期（purge），之后再删 embargo 个日期"""
    dates = np.asarray(dates)
    udates = np.sort(np.unique(dates))
    m = len(udates)
    for block in np.array_split(np.arange(m), n_splits):
        lo, hi = block[0], block[-1]
        drop = np.zeros(m, dtype=bool)
        drop[max(0, lo - horizon):min(m, hi + 1 + horizon + embargo)] = True
        train_days, test_days = udates[~drop], udates[lo:hi + 1]
        yield np.flatnonzero(np.isin(dates, train_days)), np.flatnonzero(np.isin(dates, test_days))


# ==================== 滚动训练（Walk-Forward） ====================


def walk_forward(X: pd.DataFrame, y: pd.Series, model_fn, min_train: int = 24, retrain_every: int = 1,
                 gap: int = 1, window: int | None = None, start: int | None = None,
                 retrain_rule=None, return_log: bool = False):
    """
    逐期样本外预测。对第 k 个调仓日 D[k]：
        训练集 = D[k − gap − window : k − gap)（window=None 为扩展窗口），即不含 D[k−1] 及之后
        gap = 1：D[k−1] 的标签要到 D[k]+1 开盘才实现，在 D[k] 收盘时不可用
    retrain_every：每隔多少期重训；retrain_rule(k, dates, log) → bool 可实现触发式重训
    """
    lvl = X.index.get_level_values("datetime")
    dates = lvl.unique().sort_values()
    first = start if start is not None else min_train + gap
    preds, log, model, last_fit = [], [], None, None
    for k in range(first, len(dates)):
        due = model is None or (k - last_fit) >= retrain_every
        if retrain_rule is not None and model is not None:
            due = retrain_rule(k, dates, log)
        if due:
            hi = k - gap
            lo = 0 if window is None else max(0, hi - window)
            tr = (lvl >= dates[lo]) & (lvl <= dates[hi - 1])
            ok = tr & y.notna().to_numpy()
            model = model_fn()
            model.fit(X[ok], y[ok])
            last_fit = k
            log.append({"test": dates[k], "train_start": dates[lo], "train_end": dates[hi - 1],
                        "n_train": int(ok.sum()), "model": model})
        te = lvl == dates[k]
        preds.append(pd.Series(model.predict(X[te]), index=X.index[te]))
    pred = pd.concat(preds)
    return (pred, log) if return_log else pred


# ==================== 组合构建与回测 ====================


def topk_equal(scores: pd.Series, k: int = 20) -> pd.Series:
    top = scores.nlargest(k).index
    return pd.Series(1.0 / len(top), index=top)


def topk_dropout(scores: pd.Series, holdings: list, topk: int = 20, n_drop: int = 4) -> list:
    """Qlib TopkDropoutStrategy 的简化版：每期只卖掉持仓中分数最低的 n_drop 只，买入未持仓中最高的"""
    held = [s for s in holdings if s in scores.index]  # 今天不在可交易池里的持仓视为卖出（简化）
    if len(held) < topk:
        return held + scores.drop(held).nlargest(topk - len(held)).index.tolist()
    sell = scores[held].nsmallest(n_drop).index.tolist()
    buy = scores.drop(held).nlargest(n_drop).index.tolist()
    return [s for s in held if s not in sell] + buy


def score_weighted(scores: pd.Series, k: int = 20) -> pd.Series:
    """Top K 内按分数排名加权：第 1 名权重 k，第 k 名权重 1"""
    top = scores.nlargest(k)
    w = pd.Series(np.arange(len(top), 0, -1, dtype=float), index=top.index)
    return w / w.sum()


def backtest_weights(targets: dict, ret_wide: pd.DataFrame, max_turnover: float | None = None) -> pd.DataFrame:
    """
    月度调仓回测（成本口径与 Phase 7 第 06 课一致）
        targets: {调仓日: 目标权重 Series}；ret_wide: 持有期收益宽表
        max_turnover: 单边换手上限；超过时只向目标权重移动一部分 w = w_prev + λ(w* − w_prev)
    返回 DataFrame[gross, cost, net, turnover]
    """
    COMM, SLIP, stamp = factor_tools.COMMISSION, factor_tools.SLIPPAGE, factor_tools.stamp_tax
    prev = pd.Series(dtype=float)
    rows = {}
    for d, tgt in targets.items():
        codes = tgt.index.union(prev.index)
        w_prev = prev.reindex(codes, fill_value=0.0)
        w = tgt.reindex(codes, fill_value=0.0)
        trade = w - w_prev
        turn = trade.abs().sum() / 2
        if max_turnover is not None and len(prev) and turn > max_turnover:
            w = w_prev + trade * (max_turnover / turn)
            trade = w - w_prev
            turn = max_turnover
        buy, sell = trade.clip(lower=0).sum(), (-trade).clip(lower=0).sum()
        cost = buy * (COMM + SLIP) + sell * (COMM + SLIP + stamp(d))
        r = ret_wide.loc[d].reindex(codes).fillna(0.0)
        gross = float((w * r).sum())
        rows[d] = {"gross": gross, "cost": cost, "net": gross - cost, "turnover": turn}
        w = w[w > 1e-10]
        prev = w * (1 + r.reindex(w.index)) / (1 + gross)  # 持有期内的权重漂移
    return pd.DataFrame(rows).T


def perf_table(rets: dict[str, pd.Series], periods: int = 12) -> pd.DataFrame:
    return pd.DataFrame({k: factor_tools.perf_stats(v, periods) for k, v in rets.items()}).T
