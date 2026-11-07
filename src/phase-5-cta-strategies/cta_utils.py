"""
Phase 5 辅助模块：期货数据 + 回测绩效工具
==========================================

本模块不是课程文件，而是 06–10 课（以及 04、05 课的期货部分）共用的工具：

1. 期货数据（AKShare 新浪接口）
   - get_main_sina(symbol)      新浪"主力连续"（未复权，换月处有跳空）
   - get_contracts(symbol)      逐个下载单个合约，长表格式
   - build_continuous(...)      用"前一日持仓量最大 + 只向远月切换"规则选主力，
                                构造价差调整 / 比例调整的连续合约
   - get_continuous(symbol)     上面三步的一站式入口（推荐）
   所有下载结果缓存到 data/cache/futures_*.parquet，第二次运行不再请求网络。

2. 离线回退
   QTC_OFFLINE=1 或网络失败时，用 utils.data.synthetic_ohlcv 生成"现货"路径，
   再按到期日、升贴水和持仓量曲线派生出一组合成合约，整条处理流程保持不变。
   合成数据只用于跑通代码，不能得出任何关于真实市场的结论。

3. 回测与绩效
   - backtest(target, ret, cost, roll)   目标仓位 → shift(1) 次日持有 → 扣成本
   - perf_stats(...)                     年化收益 / 波动 / Sharpe / 最大回撤 / 换手
   - trade_stats(...)                    逐笔交易统计：胜率、盈亏比
   - true_range / wilder_atr             真实波幅与 ATR（Wilder 平滑）

⚠️ 合约乘数、最小变动价位、保证金比例均为示例值，以交易所和期货公司最新规定为准。
⚠️ 新浪单合约数据只覆盖约 2019 年年中之后到期的合约，所以连续合约从 2019-07 开始。
"""

from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import CACHE_DIR, is_offline, synthetic_ohlcv  # noqa: E402

# 期货研究区间：新浪单合约数据最早约到 2019 年年中到期的合约
FUT_START, FUT_END = "20190701", "20260930"

# ==================== 品种规格（示例值，以交易所和期货公司最新规定为准） ====================
# mult: 合约乘数；tick: 最小变动价位；margin: 保证金比例示例（交易所 + 期货公司加收，会调整）
# months: 主力合约通常所在的月份（用于决定下载哪些单合约）
# price / vol: 仅用于离线合成数据的起始价格与年化波动
SPECS: dict[str, dict] = {
    "RB": dict(name="螺纹钢", sector="黑色", exchange="上期所", mult=10, unit="吨", tick=1.0, margin=0.10,
               months=[1, 5, 10], price=3800, vol=0.22),
    "I": dict(name="铁矿石", sector="黑色", exchange="大商所", mult=100, unit="吨", tick=0.5, margin=0.13,
              months=[1, 5, 9], price=750, vol=0.32),
    "CU": dict(name="沪铜", sector="有色", exchange="上期所", mult=5, unit="吨", tick=10.0, margin=0.10,
               months=list(range(1, 13)), price=60000, vol=0.18),
    "AU": dict(name="沪金", sector="贵金属", exchange="上期所", mult=1000, unit="克", tick=0.02, margin=0.10,
               months=[6, 12], price=400, vol=0.14),
    "AG": dict(name="沪银", sector="贵金属", exchange="上期所", mult=15, unit="千克", tick=1.0, margin=0.12,
               months=[6, 12], price=5000, vol=0.25),
    "M": dict(name="豆粕", sector="农产品", exchange="大商所", mult=10, unit="吨", tick=1.0, margin=0.08,
              months=[1, 5, 9], price=3000, vol=0.18),
    "P": dict(name="棕榈油", sector="农产品", exchange="大商所", mult=10, unit="吨", tick=2.0, margin=0.10,
              months=[1, 5, 9], price=7000, vol=0.28),
    "SR": dict(name="白糖", sector="农产品", exchange="郑商所", mult=10, unit="吨", tick=1.0, margin=0.08,
               months=[1, 5, 9], price=5500, vol=0.15),
    "TA": dict(name="PTA", sector="能化", exchange="郑商所", mult=5, unit="吨", tick=2.0, margin=0.08,
               months=[1, 5, 9], price=5000, vol=0.24),
    "MA": dict(name="甲醇", sector="能化", exchange="郑商所", mult=10, unit="吨", tick=1.0, margin=0.09,
               months=[1, 5, 9], price=2500, vol=0.26),
    "IF": dict(name="沪深300股指", sector="金融", exchange="中金所", mult=300, unit="点", tick=0.2, margin=0.12,
               months=list(range(1, 13)), price=4000, vol=0.20),
    "T": dict(name="10年期国债", sector="金融", exchange="中金所", mult=10000, unit="百元面值", tick=0.005,
              margin=0.02, months=[3, 6, 9, 12], price=100, vol=0.04),
}
UNIVERSE = list(SPECS)

COMMISSION = 0.0001  # 单边手续费示例：名义金额的万分之一（各品种实际收费方式不同）
SLIP_TICKS = 1  # 单边滑点示例：1 个最小变动价位


def cost_rate(symbol: str, raw_price: pd.Series) -> pd.Series:
    """单边交易成本（占名义金额比例）= 手续费 + 滑点 ticks × tick / 价格"""
    return COMMISSION + SLIP_TICKS * SPECS[symbol]["tick"] / raw_price


# ==================== 合成数据（离线回退） ====================


def _synthetic_contracts(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    合成一组单合约数据：现货走几何布朗运动，合约价 = 现货 × exp(-基差率 × 剩余期限)，
    基差率随时间缓慢变化（时而升水、时而贴水），持仓量在到期前约 4 个月达到峰值。
    """
    spec = SPECS.get(symbol, dict(months=[1, 5, 9], price=3000, vol=0.2))
    t0 = pd.Timestamp(start) - pd.Timedelta(days=420)
    spot = synthetic_ohlcv(f"FUT_{symbol}", t0.strftime("%Y%m%d"), end, start_price=spec["price"],
                           annual_return=0.0, annual_vol=spec["vol"])
    dates = spot.index
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    phase = rng.uniform(0, 2 * np.pi)
    years = (dates - dates[0]).days / 365.0
    basis = 0.06 * np.sin(2 * np.pi * years / 3.0 + phase)  # 年化基差率：正 = 远月贴水

    rows = []
    for y in range(t0.year, pd.Timestamp(end).year + 2):
        for m in spec["months"]:
            expiry = pd.Timestamp(year=y, month=m, day=15)
            life = (dates > expiry - pd.Timedelta(days=365)) & (dates <= expiry)
            if not life.any():
                continue
            d = dates[life]
            tau = (expiry - d).days.values / 365.0
            factor = np.exp(-basis[life] * tau)
            days_to_exp = (expiry - d).days.values
            oi = 1e5 * np.exp(-(((days_to_exp - 120) / 60.0) ** 2)) + 50
            code = f"{symbol}{y % 100:02d}{m:02d}"
            part = pd.DataFrame(
                {
                    "date": d,
                    "contract": code,
                    "open": spot.loc[d, "open"].values * factor,
                    "high": spot.loc[d, "high"].values * factor,
                    "low": spot.loc[d, "low"].values * factor,
                    "close": spot.loc[d, "close"].values * factor,
                    "volume": (oi * 0.8).round(),
                    "hold": oi.round(),
                    "settle": spot.loc[d, "close"].values * factor,
                }
            )
            rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    return df[(df["date"] >= pd.Timestamp(start) - pd.Timedelta(days=30)) & (df["date"] <= pd.Timestamp(end))]


_warned: set[str] = set()


def _warn_synthetic(label: str, reason: str) -> None:
    if label not in _warned:
        print(f"⚠️ 使用合成数据：{label}（{reason}）")
        _warned.add(label)


# ==================== 真实数据下载 + 缓存 ====================


def get_contracts(symbol: str, start: str = FUT_START, end: str = FUT_END, verbose: bool = True) -> pd.DataFrame:
    """
    下载某品种在 [start, end] 内可能成为主力的所有单合约（长表）

    返回列：date, contract, open, high, low, close, volume, hold(持仓量), settle(结算价)
    缓存：data/cache/futures_contracts_{symbol}_{start}_{end}.parquet
    """
    path = CACHE_DIR / f"futures_contracts_{symbol}_{start}_{end}.parquet"
    if is_offline():
        _warn_synthetic(f"期货 {symbol}", "离线模式 QTC_OFFLINE=1")
        return _synthetic_contracts(symbol, start, end)
    if path.exists():
        return pd.read_parquet(path)

    try:
        import akshare as ak

        codes = [
            f"{symbol}{y % 100:02d}{m:02d}"
            for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2)
            for m in SPECS[symbol]["months"]
        ]
        if verbose:
            print(f"   ⏳ 首次下载 {symbol} 的 {len(codes)} 个候选合约（之后读缓存）...")
        parts = []
        for code in codes:
            try:
                raw = ak.futures_zh_daily_sina(symbol=code)
            except KeyError:  # 合约不存在（尚未上市或超出数据源范围），新浪返回空数据
                continue
            if raw is None or raw.empty:
                continue
            raw = raw.assign(date=pd.to_datetime(raw["date"]), contract=code)
            parts.append(raw[["date", "contract", "open", "high", "low", "close", "volume", "hold", "settle"]])
        if not parts:
            raise ValueError("没有取到任何合约")
        df = pd.concat(parts, ignore_index=True)
        num = ["open", "high", "low", "close", "volume", "hold", "settle"]
        df[num] = df[num].astype(float)
        df = df[(df["date"] >= pd.Timestamp(start) - pd.Timedelta(days=30)) & (df["date"] <= pd.Timestamp(end))]
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df
    except Exception as e:  # 网络异常种类繁多，统一回退
        _warn_synthetic(f"期货 {symbol}", f"下载失败 {type(e).__name__}: {str(e)[:60]}")
        return _synthetic_contracts(symbol, start, end)


def get_main_sina(symbol: str, start: str = FUT_START, end: str = FUT_END) -> pd.DataFrame:
    """
    新浪主力连续（如 RB0），未经复权调整

    返回列：open, high, low, close, volume, oi, settle；index 为 date
    缓存：data/cache/futures_main_{symbol}_{start}_{end}.parquet
    """
    path = CACHE_DIR / f"futures_main_{symbol}_{start}_{end}.parquet"
    if not is_offline():
        if path.exists():
            return pd.read_parquet(path)
        try:
            import akshare as ak

            raw = ak.futures_main_sina(symbol=f"{symbol}0", start_date=start, end_date=end)
            df = raw.rename(columns={"日期": "date", "开盘价": "open", "最高价": "high", "最低价": "low",
                                     "收盘价": "close", "成交量": "volume", "持仓量": "oi", "动态结算价": "settle"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["open", "high", "low", "close", "volume", "oi", "settle"]].astype(float)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return df
        except Exception as e:
            _warn_synthetic(f"期货主力连续 {symbol}0", f"下载失败 {type(e).__name__}")
    else:
        _warn_synthetic(f"期货主力连续 {symbol}0", "离线模式 QTC_OFFLINE=1")
    # 离线：用合成合约的"未调整拼接"模拟数据源的主力连续
    cont = build_continuous(_synthetic_contracts(symbol, start, end))
    out = cont[["raw_open", "raw_high", "raw_low", "raw_close", "volume", "oi"]].copy()
    out.columns = ["open", "high", "low", "close", "volume", "oi"]
    out["settle"] = out["close"]
    return out


# ==================== 主力选择与连续合约构造 ====================


def _wide(contracts: pd.DataFrame, field: str) -> pd.DataFrame:
    """长表 → 宽表（行 = 日期，列 = 合约，按到期先后排序）"""
    w = contracts.pivot_table(index="date", columns="contract", values=field, aggfunc="last")
    order = sorted(w.columns, key=lambda c: int("".join(ch for ch in c if ch.isdigit())))
    return w[order].sort_index()


def select_main(oi: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """
    主力合约规则（无未来函数）：
        1. 今天的主力 = 昨天收盘后持仓量最大的合约（且今天有成交价）
        2. 只允许向更远月切换，不回切
    """
    prev_oi = oi.shift(1).where(close.notna())  # 只看昨天的持仓量；今天必须有价格
    has_any = prev_oi.notna().any(axis=1)
    dominant = prev_oi[has_any].fillna(-1).idxmax(axis=1)
    rank_of = {c: i for i, c in enumerate(oi.columns)}
    rank = dominant.map(rank_of).cummax()  # 只向远月切换
    return rank.map(dict(enumerate(oi.columns))).rename("contract")


def build_continuous(contracts: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    构造连续合约（向后调整：最新价格保持真实，修改历史价格）

    返回列：
        contract           当天持有的主力合约
        open/high/low/close 比例调整后的连续价格（信号、收益用这个）
        close_diff          价差调整后的连续收盘价
        raw_open/.../raw_close  当天主力合约的真实价格（手数、保证金、成本用这个）
        volume, oi          主力合约成交量、持仓量
        roll                当天是否发生换月（持有合约与前一天不同）
    """
    fields = {f: _wide(contracts, f) for f in ["open", "high", "low", "close", "volume", "hold"]}
    close_w = fields["close"]
    main = select_main(fields["hold"], close_w)

    idx = main.index
    cols = close_w.columns.get_indexer(main.values)
    rows = close_w.index.get_indexer(idx)

    def pick(w: pd.DataFrame, row_shift: int = 0) -> np.ndarray:
        arr = w.to_numpy(dtype=float)
        r = rows + row_shift
        out = np.full(len(idx), np.nan)
        ok = r >= 0
        out[ok] = arr[r[ok], cols[ok]]
        return out

    out = pd.DataFrame(index=idx)
    out["contract"] = main.values
    for f in ["open", "high", "low", "close", "volume"]:
        out[f"raw_{f}" if f != "volume" else "volume"] = pick(fields[f])
    out["oi"] = pick(fields["hold"])
    out = out.dropna(subset=["raw_close"])
    for f in ["raw_open", "raw_high", "raw_low"]:
        out[f] = out[f].fillna(out["raw_close"])

    # 换月日 d：持有合约与前一天不同；用 d-1 日新旧两个合约的收盘价计算价差 / 比例
    out["roll"] = out["contract"] != out["contract"].shift(1)
    out.iloc[0, out.columns.get_loc("roll")] = False
    ratio = pd.Series(1.0, index=out.index)
    gap = pd.Series(0.0, index=out.index)
    close_ff = close_w.ffill(limit=3)
    for d in out.index[out["roll"]]:
        i = out.index.get_loc(d)
        prev_d = out.index[i - 1]
        p_new = close_ff.at[prev_d, out.at[d, "contract"]]
        p_old = close_ff.at[prev_d, out.at[prev_d, "contract"]]
        if np.isfinite(p_new) and np.isfinite(p_old) and p_old > 0:
            ratio[d] = p_new / p_old
            gap[d] = p_new - p_old

    # 第 t 天的调整 = t 之后所有换月的累积（乘积 / 求和）
    ratio_factor = ratio[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    gap_total = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    for f in ["open", "high", "low", "close"]:
        out[f] = out[f"raw_{f}"] * ratio_factor
    out["close_diff"] = out["raw_close"] + gap_total
    out["roll_ratio"] = ratio
    out["roll_gap"] = gap

    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    cols_order = ["contract", "open", "high", "low", "close", "close_diff", "raw_open", "raw_high", "raw_low",
                  "raw_close", "volume", "oi", "roll", "roll_ratio", "roll_gap"]
    return out[cols_order]


def get_continuous(symbol: str, start: str = FUT_START, end: str = FUT_END) -> pd.DataFrame:
    """一站式：下载 / 读取单合约 → 选主力 → 比例 & 价差调整连续合约（见 build_continuous）"""
    return build_continuous(get_contracts(symbol, start, end), start, end)


def load_universe(symbols: list[str] | None = None, start: str = FUT_START, end: str = FUT_END) -> dict:
    """批量加载多个品种的连续合约，取不到的品种跳过"""
    data = {}
    for s in symbols or UNIVERSE:
        try:
            df = get_continuous(s, start, end)
            if len(df) > 250:
                data[s] = df
            else:
                print(f"   ⏭️ {s} 数据不足 250 天，跳过")
        except Exception as e:
            print(f"   ⏭️ {s} 构造失败（{type(e).__name__}: {str(e)[:50]}），跳过")
    return data


# ==================== 指标 ====================


def true_range(df: pd.DataFrame, h: str = "high", lo: str = "low", c: str = "close") -> pd.Series:
    """TR = max(H-L, |H-前收|, |L-前收|)"""
    pc = df[c].shift(1)
    return pd.concat([df[h] - df[lo], (df[h] - pc).abs(), (df[lo] - pc).abs()], axis=1).max(axis=1)


def wilder_atr(df: pd.DataFrame, n: int = 20, **kw) -> pd.Series:
    """ATR（Wilder 平滑，alpha = 1/n），即海龟的 N"""
    return true_range(df, **kw).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ==================== 回测与绩效 ====================


def backtest(target: pd.Series, ret: pd.Series, cost, roll: pd.Series | None = None) -> pd.DataFrame:
    """
    向量化回测（名义敞口口径，1 = 100% 净值）

    target: 第 t 天收盘后决定的目标仓位（只能用 t 及之前的信息）
    ret:    资产日收益（期货用比例调整连续价的收益 = 实际持有合约的收益）
    cost:   单边成本（标量或逐日 Series，占名义金额比例）
    roll:   期货换月标记；换月日"平旧开新"计两次成本
    """
    pos = target.reindex(ret.index).shift(1).fillna(0.0)  # 今天收盘出信号，明天才持有
    ret = ret.fillna(0.0)
    trade = pos.diff().abs().fillna(pos.abs())
    roll_trade = (2 * pos.abs() * roll.reindex(ret.index).fillna(False).astype(float)) if roll is not None else 0.0
    turnover = trade + roll_trade
    costs = turnover * cost
    net = pos * ret - costs
    return pd.DataFrame({"pos": pos, "gross": pos * ret, "cost": costs, "ret": net,
                         "turnover": turnover, "equity": (1 + net).cumprod()})


def max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def perf_stats(ret: pd.Series, turnover: pd.Series | None = None, pos: pd.Series | None = None) -> dict:
    """常用绩效指标（无风险利率按 0 处理）"""
    r = ret.fillna(0)
    years = len(r) / 252
    nav = (1 + r).cumprod()
    total = nav.iloc[-1] - 1
    out = {
        "总收益": total,
        "年化收益": nav.iloc[-1] ** (1 / years) - 1 if nav.iloc[-1] > 0 else -1.0,
        "年化波动": r.std() * np.sqrt(252),
        "Sharpe": sharpe(r),
        "最大回撤": max_drawdown(r),
    }
    if turnover is not None:
        out["年化换手"] = turnover.sum() / years
    if pos is not None:
        out["持仓占比"] = (pos.abs() > 1e-9).mean()
    return out


def stats_table(results: dict[str, dict]) -> pd.DataFrame:
    """把多个 perf_stats 结果格式化成表格（行 = 指标，列 = 策略）"""
    df = pd.DataFrame(results)
    fmt = {"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "年化波动": "{:.1%}", "Sharpe": "{:.2f}",
           "最大回撤": "{:.1%}", "年化换手": "{:.1f}x", "持仓占比": "{:.0%}"}
    rows = {}
    for k in df.index:
        f = fmt.get(k, "{:.3g}")
        rows[k] = [f.format(v) if pd.notna(v) else "-" for v in df.loc[k]]
    return pd.DataFrame.from_dict(rows, orient="index", columns=df.columns, dtype=object)


def trade_stats(pos: pd.Series, ret: pd.Series) -> dict:
    """
    逐笔交易统计：连续同方向非零持仓为一笔交易（不区分加减仓）

    返回 笔数、胜率、平均盈利 / 平均亏损（盈亏比）、单笔收益偏度、平均持有天数
    """
    sign = np.sign(pos.fillna(0))
    trade_id = (sign != sign.shift()).cumsum()
    df = pd.DataFrame({"sign": sign, "r": (pos * ret).fillna(0), "id": trade_id})
    df = df[df["sign"] != 0]
    if df.empty:
        return {"笔数": 0, "胜率": np.nan, "盈亏比": np.nan, "单笔偏度": np.nan, "平均持有天数": np.nan}
    g = df.groupby("id")
    tr = g["r"].apply(lambda x: (1 + x).prod() - 1)
    entry = pd.Series(df.index, index=df.index).groupby(df["id"]).first()
    tr.index = pd.DatetimeIndex(entry.loc[tr.index].values, name="开仓日")
    wins, losses = tr[tr > 0], tr[tr <= 0]
    payoff = wins.mean() / -losses.mean() if len(wins) and len(losses) and losses.mean() < 0 else np.nan
    return {
        "笔数": len(tr),
        "胜率": len(wins) / len(tr),
        "盈亏比": payoff,
        "单笔偏度": tr.skew() if len(tr) > 2 else np.nan,
        "平均持有天数": g.size().mean(),
        "_trades": tr,
    }


def offline_note() -> str:
    return "（当前为合成数据，数字只用于跑通流程）" if is_offline() else ""


if __name__ == "__main__":  # 预热缓存：python cta_utils.py
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    for s in UNIVERSE:
        df = get_continuous(s)
        print(s, len(df), df.index[0].date(), df.index[-1].date(), "换月次数", int(df["roll"].sum()))
