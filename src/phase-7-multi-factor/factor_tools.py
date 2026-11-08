"""
Phase 7 辅助模块：因子研究工具
==============================

本模块不是课程文件，而是 03–11 课共用的"研究流水线"。每个函数的原理都在对应课程中
讲解，这里集中实现一份，保证各课结果口径一致：

    时间口径      rebalance_dates / period_returns / forward_returns   （03、05 课）
    可交易性过滤  infer_st / limit_pct / tradable_status              （03、06 课）
    财报 PIT      add_fin_derived / pit_merge                         （08 课）
    因子构建      build_factors                                       （03 课）
    预处理        winsorize_mad / zscore / neutralize / preprocess    （04 课）
    单因子检验    calc_ic / ic_summary / quantile_returns             （05、06 课）

统一约定：
    - 宽表：index = 日期，columns = 股票代码
    - 因子方向统一为"值越大，预期收益越高"
    - 信号在 t 日收盘后计算，t+1 日开盘执行；持有到下一个调仓日的次日开盘
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from factor_data import FACTOR_END, FACTOR_START, FactorPanel, report_deadline

FACTOR_LABELS = {
    "ep": "EP 盈利收益率",
    "bp": "BP 账面市值比",
    "roe": "ROE (TTM)",
    "mom_12_1": "动量 12-1",
    "rev_1m": "短期反转 1M",
    "lowvol": "低波动 60D",
    "lowturn": "低换手 20D",
    "size": "小市值",
}
FACTOR_CATEGORY = {
    "ep": "价值", "bp": "价值", "roe": "质量", "mom_12_1": "动量",
    "rev_1m": "反转", "lowvol": "低波", "lowturn": "流动性", "size": "规模",
}

# A 股交易成本（示例值，以券商和交易所最新规则为准）
COMMISSION = 0.00025  # 佣金，买卖双边
SLIPPAGE = 0.001  # 冲击成本 / 滑点，买卖双边（中小盘更高）
STAMP_CUT_DATE = pd.Timestamp("2023-08-28")  # 印花税减半：0.1% → 0.05%（仅卖出收取）

CHINEXT_20PCT_DATE = pd.Timestamp("2020-08-24")  # 创业板注册制改革后涨跌幅放宽到 20%
ST_LIMIT_CHANGE_DATE = pd.Timestamp("2026-07-06")  # 主板 ST 涨跌幅 5% → 10%


def stamp_tax(date: pd.Timestamp) -> float:
    return 0.001 if date < STAMP_CUT_DATE else 0.0005


# ==================== 时间口径 ====================


def month_ends(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个交易日"""
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).last().to_numpy())


def rebalance_dates(panel: FactorPanel, start: str = FACTOR_START, end: str = FACTOR_END) -> pd.DatetimeIndex:
    me = month_ends(panel.dates)
    return me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]


def exec_price(panel: FactorPanel) -> pd.DataFrame:
    """成交 / 估值价格（复权）：能交易时用开盘价；停牌时只能按最后收盘价估值"""
    return panel.adj_open.fillna(panel.adj_close.ffill())


def _next_rebalance(panel: FactorPanel, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个调仓日的下一个调仓日（最后一个调仓日之后取日历中的下一个月末）"""
    me = month_ends(panel.dates)
    nxt = list(dates[1:])
    later = me[me > dates[-1]]
    nxt.append(later[0] if len(later) else pd.NaT)
    return pd.DatetimeIndex(nxt)


def period_returns(panel: FactorPanel, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    持有期收益：t 日收盘出信号 → t+1 日开盘买入 → 下一调仓日 t' 的次日开盘卖出
    返回 index = 调仓日 t
    """
    cal, px = panel.dates, exec_price(panel)
    entry = cal.get_indexer(dates) + 1
    nxt = _next_rebalance(panel, dates)
    exit_ = np.array([cal.get_loc(d) + 1 if pd.notna(d) and cal.get_loc(d) + 1 < len(cal) else -1 for d in nxt])
    out = pd.DataFrame(np.nan, index=dates, columns=panel.codes)
    ok = (entry < len(cal)) & (exit_ > 0)
    out.loc[dates[ok]] = px.iloc[exit_[ok]].to_numpy() / px.iloc[entry[ok]].to_numpy() - 1
    return out


def forward_returns(panel: FactorPanel, dates: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    """t+1 日开盘到 t+1+h 日开盘的收益（h 个交易日），用于 IC 衰减"""
    cal, px = panel.dates, exec_price(panel)
    a = cal.get_indexer(dates) + 1
    b = a + h
    out = pd.DataFrame(np.nan, index=dates, columns=panel.codes)
    ok = b < len(cal)
    out.loc[dates[ok]] = px.iloc[b[ok]].to_numpy() / px.iloc[a[ok]].to_numpy() - 1
    return out


def bench_period_returns(panel: FactorPanel, dates: pd.DatetimeIndex, name: str = "hs300") -> pd.Series:
    """基准指数在相同持有期（t+1 开盘 → t'+1 开盘）的收益"""
    cal, o = panel.dates, panel.bench[f"{name}_open"]
    a = cal.get_indexer(dates) + 1
    nxt = _next_rebalance(panel, dates)
    vals = []
    for i, d in zip(a, nxt):
        if pd.isna(d) or cal.get_loc(d) + 1 >= len(cal) or i >= len(cal):
            vals.append(np.nan)
        else:
            vals.append(o.iloc[cal.get_loc(d) + 1] / o.iloc[i] - 1)
    return pd.Series(vals, index=dates, name=name)


# ==================== 可交易性 ====================


def infer_st(panel: FactorPanel, window: int = 60) -> pd.DataFrame:
    """
    免费数据没有历史 ST 标记，这里用"涨跌停幅度"反推（只用过去数据，不含未来信息）：
    主板股票在过去 window 个交易日里，至少 2 次恰好收在 ±5% 的涨跌停价，
    且没有一天的涨跌幅超过 5.25%（除权日除外）→ 视为当时处于 ST 状态。
    这是近似方法：长期不触及涨跌停的 ST 股会漏判。
    """
    close, prev = panel.close, panel.close.ffill().shift(1)
    up5, dn5 = (prev * 1.05).round(2), (prev * 0.95).round(2)
    at_limit = (((close - up5).abs() < 0.001) & (close >= panel.high - 0.001)) | (
        ((close - dn5).abs() < 0.001) & (close <= panel.low + 0.001)
    )
    ex_day = panel.adj_factor.ne(panel.adj_factor.shift(1))
    big = ((close / prev - 1).abs() > 0.0525) & ~ex_day
    hits = at_limit.astype(float).rolling(window, min_periods=1).sum()
    bigs = big.astype(float).rolling(window, min_periods=1).sum()
    main = (panel.info["board"] == "主板").reindex(panel.codes).to_numpy()
    return (hits >= 2) & (bigs == 0) & close.notna() & main


def limit_pct(panel: FactorPanel, st: pd.DataFrame | None = None) -> pd.DataFrame:
    """涨跌幅限制：主板 10%；创业板 2020-08-24 起 20%；科创板 20%；主板 ST 5%（2026-07-06 起 10%）"""
    cal = panel.dates
    board = panel.info["board"].reindex(panel.codes)
    lim = pd.DataFrame(0.10, index=cal, columns=panel.codes)
    lim.loc[:, (board == "科创板").to_numpy()] = 0.20
    lim.loc[cal >= CHINEXT_20PCT_DATE, (board == "创业板").to_numpy()] = 0.20
    if st is not None:  # ST 状态用前一交易日的推断结果（只用过去信息）
        st_lim = pd.Series(np.where(cal < ST_LIMIT_CHANGE_DATE, 0.05, 0.10), index=cal)
        lim = lim.mask(st.shift(1, fill_value=False), st_lim, axis=0)
    return lim


def tradable_status(panel: FactorPanel, dates: pd.DatetimeIndex, min_list_days: int = 365) -> dict:
    """
    调仓日 t 的可交易性（布尔宽表，index = 调仓日）：
        listed   上市满 min_list_days 个自然日（剔除次新股）
        trading  t 日正常交易（未停牌）
        not_st   t 日不是 ST（infer_st 推断）
        buyable  t+1 日未停牌且开盘未涨停（执行层面：买不进）
        all      以上全部满足
    """
    cal = panel.dates
    pos = cal.get_indexer(dates)
    st = infer_st(panel)
    lim = limit_pct(panel, st)
    list_date = panel.info["list_date"].reindex(panel.codes)
    listed = pd.DataFrame(
        (dates.to_numpy()[:, None] - list_date.to_numpy()[None, :]) >= np.timedelta64(min_list_days, "D"),
        index=dates, columns=panel.codes,
    )
    trading = panel.volume.iloc[pos].set_axis(dates).gt(0)
    not_st = ~st.iloc[pos].set_axis(dates)
    nxt = np.minimum(pos + 1, len(cal) - 1)
    close_t = panel.close.iloc[pos].to_numpy()
    up_next = np.round(close_t * (1 + lim.iloc[nxt].to_numpy()), 2)
    open_next = panel.open.iloc[nxt].to_numpy()
    with np.errstate(invalid="ignore"):
        buyable = pd.DataFrame(~np.isnan(open_next) & (open_next < up_next - 0.001), index=dates, columns=panel.codes)
    status = {"listed": listed, "trading": trading, "not_st": not_st, "buyable": buyable}
    status["all"] = listed & trading & not_st & buyable
    return status


def tradable_mask(panel: FactorPanel, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return tradable_status(panel, dates)["all"]


# ==================== 财报 Point-in-Time ====================


def add_fin_derived(fin: pd.DataFrame) -> pd.DataFrame:
    """
    在累计值口径的财报长表上增加：
        net_profit_ttm  TTM = 本期累计 + 上年年报 − 上年同期累计（年报直接用累计值）
        net_profit_q    单季度 = 本期累计 − 上一季度累计（一季报直接用累计值）
        np_q_yoy        单季度净利润同比增速 = (本季 − 去年同季) / |去年同季|
    """
    fin = fin.sort_values(["code", "end_date"]).reset_index(drop=True)
    key = fin.set_index(["code", "end_date"])["net_profit"]

    def lookup(series: pd.Series, dates: pd.Series) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([fin["code"], dates])
        return series.reindex(idx).to_numpy()

    ed = fin["end_date"]
    last_annual = pd.to_datetime(pd.DataFrame({"year": ed.dt.year - 1, "month": 12, "day": 31}))
    same_last = ed - pd.DateOffset(years=1)
    prev_q = ed - pd.offsets.QuarterEnd(1)
    np_ = fin["net_profit"].to_numpy()
    is_annual, is_q1 = (ed.dt.month == 12).to_numpy(), (ed.dt.month == 3).to_numpy()
    fin["net_profit_ttm"] = np.where(is_annual, np_, np_ + lookup(key, last_annual) - lookup(key, same_last))
    fin["net_profit_q"] = np.where(is_q1, np_, np_ - lookup(key, prev_q))
    q_key = fin.set_index(["code", "end_date"])["net_profit_q"]
    q_last = lookup(q_key, same_last)
    with np.errstate(divide="ignore", invalid="ignore"):
        fin["np_q_yoy"] = np.where(np.abs(q_last) > 0, (fin["net_profit_q"] - q_last) / np.abs(q_last), np.nan)
    return fin


def pit_merge(panel: FactorPanel, dates: pd.DatetimeIndex, cols: list[str], how: str = "ann",
              fin: pd.DataFrame | None = None, max_age_days: int = 500) -> dict[str, pd.DataFrame]:
    """
    把财报数据对齐到调仓日，返回 {列名: 宽表}

    how:
        "ann"       按实际公告日（正确）
        "deadline"  按法定披露截止日（保守，没有公告日数据时的替代方案）
        "report"    按报告期（错误！报告期结束第二天就"知道"了财报）
    对齐规则：merge_asof 取"可用日 < t"的最近一条（公告当天不用），
    并剔除"报告期倒退"的晚到旧数据；报告期距今超过 max_age_days 的视为过期。
    """
    fin = add_fin_derived(panel.fin) if fin is None else fin
    avail = {"ann": fin["ann_date"], "deadline": report_deadline(fin["end_date"]), "report": fin["end_date"]}[how]
    f = fin[["code", "end_date", *cols]].assign(avail=pd.to_datetime(avail).astype("datetime64[ns]"))
    f = f.dropna(subset=["avail"]).sort_values(["avail", "end_date"])
    f = f[f["end_date"] >= f.groupby("code")["end_date"].cummax()]  # 只保留报告期不倒退的记录
    f["code"] = f["code"].astype(str)

    codes = panel.codes.astype(str)
    left = pd.DataFrame({"date": np.repeat(dates.astype("datetime64[ns]"), len(codes)),
                         "code": np.tile(codes, len(dates))})
    m = pd.merge_asof(left, f, left_on="date", right_on="avail", by="code",
                      direction="backward", allow_exact_matches=False)
    stale = (m["date"] - m["end_date"]).dt.days > max_age_days
    out = {}
    for c in cols + ["end_date"]:
        vals = m[c].where(~stale)
        out[c] = (pd.DataFrame({"date": m["date"], "code": m["code"], "v": vals})
                  .pivot(index="date", columns="code", values="v")
                  .reindex(index=dates, columns=panel.codes))
    return out


# ==================== 因子构建 ====================


def build_factors(panel: FactorPanel, dates: pd.DatetimeIndex, how: str = "ann") -> dict[str, pd.DataFrame]:
    """在调仓日计算原始因子（方向已统一为"越大越好"，未做预处理）"""
    cal = panel.dates
    pos = cal.get_indexer(dates)
    px = panel.adj_close.ffill()

    def at(df: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
        idx = pos - lag
        out = df.iloc[np.clip(idx, 0, None)].set_axis(dates)
        return out.where(pd.Series(idx >= 0, index=dates), axis=0)

    pit = pit_merge(panel, dates, ["net_profit_ttm", "equity"], how=how)
    tmc = at(panel.total_mcap)
    equity = pit["equity"]
    return {
        "ep": pit["net_profit_ttm"] / tmc,
        "bp": equity.where(equity > 0) / tmc,
        "roe": pit["net_profit_ttm"] / equity.where(equity > 0),
        "mom_12_1": at(px, 21) / at(px, 252) - 1,
        "rev_1m": -(at(px, 0) / at(px, 21) - 1),
        "lowvol": -at(panel.daily_ret.rolling(60, min_periods=40).std()),
        "lowturn": -at(panel.turnover.rolling(20, min_periods=10).mean()),
        "size": -np.log(at(panel.mcap)),
    }


# ==================== 预处理 ====================


def winsorize_mad(x: pd.Series, n: float = 3.0) -> pd.Series:
    """MAD 去极值：截断到 中位数 ± n × 1.4826 × MAD"""
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return x
    bound = n * 1.4826 * mad
    return x.clip(med - bound, med + bound)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def neutralize(f: pd.Series, sector: pd.Series, log_mcap: pd.Series) -> pd.Series:
    """对行业哑变量 + 对数市值做截面回归，返回残差（不加常数项：全部行业哑变量之和就是常数）"""
    df = pd.concat([f.rename("f"), sector.rename("sec"), log_mcap.rename("lnmv")], axis=1).dropna()
    X = pd.get_dummies(df["sec"], dtype=float)
    X["lnmv"] = df["lnmv"]
    beta, *_ = np.linalg.lstsq(X.to_numpy(), df["f"].to_numpy(), rcond=None)
    resid = df["f"] - X.to_numpy() @ beta
    return resid.reindex(f.index)


def preprocess(raw: pd.DataFrame, panel: FactorPanel, mask: pd.DataFrame | None = None,
               neutral: bool = True, n_mad: float = 3.0, min_obs: int = 30, size_neutral: bool = True) -> pd.DataFrame:
    """
    逐个截面：(可交易过滤) → MAD 去极值 → Z-score → (行业市值中性化 → Z-score)
    size_neutral=False：只做行业中性（规模因子本身不能对市值中性化，否则只剩噪声）
    """
    lnmv = np.log(panel.mcap.reindex(raw.index))
    if not size_neutral:
        lnmv = lnmv * 0.0  # 市值列全为 0 → 回归只剩行业哑变量
    sector = panel.info["sector"]
    out = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    for d in raw.index:
        x = raw.loc[d]
        if mask is not None:
            x = x.where(mask.loc[d])
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < min_obs:
            continue
        x = zscore(winsorize_mad(x, n_mad))
        if neutral:
            x = zscore(neutralize(x, sector.reindex(x.index), lnmv.loc[d, x.index]).dropna())
        out.loc[d, x.index] = x.to_numpy()
    return out


def preprocess_all(factors: dict, panel: FactorPanel, mask: pd.DataFrame | None = None,
                   neutral: bool = True) -> dict[str, pd.DataFrame]:
    return {k: preprocess(v, panel, mask, neutral, size_neutral=(k != "size")) for k, v in factors.items()}


# ==================== 单因子检验 ====================


def calc_ic(factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "rank", min_obs: int = 20) -> pd.Series:
    """逐期截面相关系数。method = "rank"（Spearman，Rank IC）或 "pearson"（IC）"""
    f, r = factor.align(fwd, join="inner")
    ok = f.notna() & r.notna()
    f, r = f.where(ok), r.where(ok)
    if method == "rank":
        f, r = f.rank(axis=1), r.rank(axis=1)
    fc, rc = f.sub(f.mean(axis=1), axis=0), r.sub(r.mean(axis=1), axis=0)
    ic = (fc * rc).sum(axis=1) / np.sqrt((fc**2).sum(axis=1) * (rc**2).sum(axis=1))
    return ic.where(ok.sum(axis=1) >= min_obs)


def nw_tstat(x: pd.Series, lags: int) -> float:
    """均值的 Newey-West t 值（重叠样本 / 自相关时使用）"""
    x = x.dropna()
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def ic_summary(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    return pd.Series({
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "ICIR": ic.mean() / ic.std(),
        "t值": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC>0占比": (ic > 0).mean(),
        "期数": len(ic),
    })


def quantile_labels(factor: pd.Series, n_groups: int = 5) -> pd.Series:
    """按因子值等分为 1..n 组（1 = 因子值最低）"""
    f = factor.dropna()
    return pd.qcut(f.rank(method="first"), n_groups, labels=range(1, n_groups + 1)).astype(int)


def quantile_returns(factor: pd.DataFrame, ret: pd.DataFrame, n_groups: int = 5, min_obs: int = 30) -> pd.DataFrame:
    """分层回测：每期各组等权平均收益；列 1..n 与 "多空"（n − 1）"""
    rows = {}
    for d in factor.index:
        f, r = factor.loc[d], ret.loc[d]
        ok = f.notna() & r.notna()
        if ok.sum() < min_obs:
            continue
        g = quantile_labels(f[ok], n_groups)
        rows[d] = r[ok].groupby(g).mean()
    grp = pd.DataFrame(rows).T
    grp["多空"] = grp[n_groups] - grp[1]
    return grp


def perf_stats(r: pd.Series, periods: int = 12) -> pd.Series:
    """年化收益、年化波动、夏普（未扣无风险利率）、最大回撤、胜率"""
    r = r.dropna()
    nav = (1 + r).cumprod()
    return pd.Series({
        "年化收益": nav.iloc[-1] ** (periods / len(r)) - 1,
        "年化波动": r.std() * np.sqrt(periods),
        "夏普": r.mean() / r.std() * np.sqrt(periods),
        "最大回撤": (nav / nav.cummax() - 1).min(),
        "胜率": (r > 0).mean(),
    })


def research_inputs(panel: FactorPanel, neutral: bool = True):
    """常用的一组输入：调仓日、可交易掩码、持有期收益、原始因子、预处理后因子"""
    dates = rebalance_dates(panel)
    mask = tradable_mask(panel, dates)
    ret = period_returns(panel, dates)
    raw = build_factors(panel, dates)
    clean = preprocess_all(raw, panel, mask, neutral=neutral)
    return dates, mask, ret, raw, clean
