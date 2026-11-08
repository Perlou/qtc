"""
Phase 7 辅助模块：多因子截面面板数据
====================================

本模块不是课程文件，而是 01–11 课共用的数据层。它负责构建一个"中等规模、可复用"的
A 股截面面板，并缓存到 data/cache/factor_panel_*.parquet，之后每课直接读缓存。

数据方案（联网模式）：
    股票池   中证指数官网的沪深 300 + 中证 500 **当前**成分股（index_stock_cons_csindex），
             按东方财富行业映射为 11 个大类行业，每个行业在两个指数中各按代码均匀抽取至多
             6 只，共约 130 只；要求 2019 年报已披露（即 2020 年 4 月前后已上市）
             ⚠️ 用"今天的成分股"回测过去 → 幸存者偏差 + 成分股前视偏差（03 课详细讨论）
    行情     utils.data.get_stock_daily_adj：不复权 OHLCV + 等比后复权因子
    股本     stock_value_em：逐日总股本、流通股本 → 总市值、流通市值、换手率
    上市日期 stock_individual_info_em
    行业     stock_yjbb_em 的"所处行业"（东方财富行业，是**当前**分类，不是历史分类）
    财报     stock_yjbb_em(date=报告期)：净利润（归母，年初至今累计）、营业总收入、
             每股收益、每股净资产、ROE（数值可能是事后修订版）
    公告日   stock_yysj_em(date=报告期)："实际披露时间" = 首次披露日
             ⚠️ yjbb 的"最新公告日期"是**最近一次更新**的日期（常常是下一年同期报告
                重述时），不是首次公告日，不能用来做 PIT 对齐（08 课演示）
    基准     沪深 300（000300）、中证 500（000905）、中证 1000（000852）
    时间     行情 2018-01 ~ 2025-02（留出因子回看期和最后一期的持有期）
             因子 / 调仓 2019-01 ~ 2024-12，月末调仓

离线（QTC_OFFLINE=1）且无本地缓存，或下载失败：
    synthetic_panel() 生成结构相同的合成面板，收益由"因子暴露 × 因子收益 + 噪声"生成，
    内置价值、质量、小市值、低波、低换手、短期反转、财报公告跳空等结构，并模拟停牌、
    涨跌停、ST、次新股。合成数据只用于跑通代码，不能得出任何关于真实市场的结论。

想刷新数据：删除 data/cache/factor_panel_*.parquet（以及 factor_fin_* 等中间缓存）。
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import CACHE_DIR, get_index_daily, get_stock_daily_adj, is_offline, synthetic_ohlcv  # noqa: E402

# ==================== 常量 ====================

PRICE_START, PRICE_END = "20180101", "20250228"
FACTOR_START, FACTOR_END = "2019-01-01", "2024-12-31"
INDEX_START, INDEX_END = "20100101", "20251231"  # 与已有指数缓存一致，再按需切片
FIN_PERIODS = pd.date_range("2017-09-30", "2024-12-31", freq="QE")
RF_ANNUAL = 0.02  # 无风险利率示例值（约等于近年 1 年期国债收益率水平），全阶段统一使用
PER_SECTOR_PER_INDEX = 6
MIN_STOCKS = 40  # 真实数据成功下载少于这个数量 → 回退到合成数据

BENCH_CODES = {"hs300": "000300", "zz500": "000905", "zz1000": "000852"}

PANEL_FILE = CACHE_DIR / "factor_panel_daily.parquet"
INFO_FILE = CACHE_DIR / "factor_panel_info.parquet"
FIN_FILE = CACHE_DIR / "factor_panel_fin.parquet"
BENCH_FILE = CACHE_DIR / "factor_panel_bench.parquet"

DAILY_FIELDS = ["open", "high", "low", "close", "adj_factor", "volume", "amount", "float_shares", "total_shares"]

# 东方财富行业（二级）→ 11 个大类行业。只为本课程的小股票池服务：
# 120 只股票如果用 100 多个细分行业做中性化，每个行业只有一两只股票，回归会退化
SECTORS = {
    "金融": "证券Ⅱ 银行Ⅱ 多元金融 保险Ⅱ",
    "地产建筑": "房地产开发 房地产服务 基础建设 专业工程 房屋建设Ⅱ 装修建材 装修装饰Ⅱ 水泥 玻璃玻纤 工程咨询服务Ⅱ",
    "公用交运": "电力 燃气Ⅱ 航运港口 航空机场 铁路公路 物流 环境治理 环保设备Ⅱ 水务及水治理",
    "周期资源": "煤炭开采 焦炭Ⅱ 炼化及贸易 油服工程 油气开采Ⅱ 工业金属 小金属 贵金属 能源金属 "
    "普钢 特钢Ⅱ 冶钢原料 金属新材料",
    "化工材料": "化学制品 化学原料 农化制品 化学纤维 塑料 橡胶 非金属材料Ⅱ 造纸 包装印刷",
    "医药": "化学制药 中药Ⅱ 生物制品 医疗服务 医疗器械 医药商业 医疗美容",
    "消费": "白酒Ⅱ 非白酒 饮料乳品 食品加工 调味发酵品Ⅱ 休闲食品 养殖业 饲料 种植业 农产品加工 "
    "动物保健Ⅱ 白色家电 小家电 黑色家电 家电零部件Ⅱ 厨卫电器 一般零售 旅游零售Ⅱ 互联网电商 "
    "服装家纺 纺织制造 饰品 家居用品 化妆品 个护用品 旅游及景区 酒店餐饮 文娱用品 教育",
    "电力设备": "电池 光伏设备 电网设备 风电设备 其他电源设备Ⅱ 电机Ⅱ",
    "机械军工汽车": "汽车零部件 乘用车 商用车 摩托车及其他 汽车服务 工程机械 通用设备 专用设备 自动化设备 "
    "轨交设备Ⅱ 航空装备Ⅱ 军工电子Ⅱ 航天装备Ⅱ 航海装备Ⅱ 地面兵装Ⅱ 专业服务 综合Ⅱ",
    "电子": "半导体 消费电子 元件 光学光电子 其他电子Ⅱ 电子化学品Ⅱ",
    "TMT": "通信设备 通信服务 软件开发 IT服务Ⅱ 计算机设备 游戏Ⅱ 影视院线 数字媒体 出版 广告营销 电视广播Ⅱ",
}
INDUSTRY_TO_SECTOR = {ind: sec for sec, inds in SECTORS.items() for ind in inds.split()}

_warned: set[str] = set()


def warn_synthetic(label: str, reason: str) -> None:
    """同一类数据只提示一次"""
    if label not in _warned:
        print(f"⚠️ 使用合成数据：{label}（{reason}）")
        _warned.add(label)


def board_of(code: str) -> str:
    """按代码判断板块（决定涨跌幅限制）"""
    if code.startswith("688"):
        return "科创板"
    if code.startswith("30"):  # 300 / 301 / 302（302 为更名换码后的创业板代码段）
        return "创业板"
    return "主板"


def report_deadline(end_date: pd.Series) -> pd.Series:
    """法定披露截止日：一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30"""
    end_date = pd.to_datetime(end_date)
    month = end_date.dt.month
    year = end_date.dt.year
    out = pd.Series(pd.NaT, index=end_date.index, dtype="datetime64[ns]")
    for m, (dy, dm, dd) in {3: (0, 4, 30), 6: (0, 8, 31), 9: (0, 10, 31), 12: (1, 4, 30)}.items():
        sel = month == m
        out[sel] = pd.to_datetime(
            pd.DataFrame({"year": year[sel] + dy, "month": dm, "day": dd}), errors="coerce"
        )
    return out


# ==================== 面板数据结构 ====================


@dataclass
class FactorPanel:
    """
    日频宽表（index = 交易日，columns = 股票代码），停牌 / 未上市为 NaN：
        open / high / low / close   不复权价格（用于涨跌停判断、市值）
        adj_factor                  等比后复权因子（adj_close = close × adj_factor，用于收益）
        volume / amount             成交量（股）/ 成交额（元）
        float_shares / total_shares 流通股本 / 总股本
    info:  index = 股票代码；列 name, sector, industry, index, board, list_date
    fin:   长表；列 code, end_date, ann_date, ann_source, latest_ann_date, net_profit, revenue,
           eps, bvps, roe, total_shares_end, equity（累计值口径，见 08 课）
    bench: 基准指数日线；列 {hs300,zz500,zz1000}_{open,close}
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_factor: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    float_shares: pd.DataFrame
    total_shares: pd.DataFrame
    info: pd.DataFrame
    fin: pd.DataFrame
    bench: pd.DataFrame
    is_synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def codes(self) -> pd.Index:
        return self.close.columns

    @cached_property
    def adj_close(self) -> pd.DataFrame:
        return self.close * self.adj_factor

    @cached_property
    def adj_open(self) -> pd.DataFrame:
        return self.open * self.adj_factor

    @cached_property
    def close_ffill(self) -> pd.DataFrame:
        """停牌日沿用最后收盘价（只向前填充，用于估值，不代表可以成交）"""
        return self.close.ffill()

    @cached_property
    def mcap(self) -> pd.DataFrame:
        """流通市值（元）"""
        return self.close_ffill * self.float_shares

    @cached_property
    def total_mcap(self) -> pd.DataFrame:
        """总市值（元）"""
        return self.close_ffill * self.total_shares

    @cached_property
    def turnover(self) -> pd.DataFrame:
        """换手率 = 成交量 / 流通股本"""
        return self.volume / self.float_shares

    @cached_property
    def daily_ret(self) -> pd.DataFrame:
        """日收益（等比复权）。停牌日为 NaN，复牌日为停牌期间的累计收益"""
        r = self.adj_close.ffill().pct_change(fill_method=None)
        return r.where(self.close.notna())

    def bench_ret(self, name: str = "hs300") -> pd.Series:
        return self.bench[f"{name}_close"].pct_change(fill_method=None)


# ==================== 缓存读写 ====================


def _save(panel: FactorPanel) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = {f: getattr(panel, f).stack() for f in DAILY_FIELDS}
    daily = pd.DataFrame(frames)
    daily.index.names = ["date", "code"]
    daily.to_parquet(PANEL_FILE)
    panel.info.to_parquet(INFO_FILE)
    panel.fin.to_parquet(FIN_FILE)
    panel.bench.to_parquet(BENCH_FILE)


def _load_cached() -> FactorPanel:
    daily = pd.read_parquet(PANEL_FILE)
    info = pd.read_parquet(INFO_FILE)
    info["board"] = [board_of(c) for c in info.index]
    fin = pd.read_parquet(FIN_FILE)
    bench = pd.read_parquet(BENCH_FILE)
    calendar = bench.index
    wide = {}
    for f in DAILY_FIELDS:
        w = daily[f].unstack("code").reindex(index=calendar, columns=info.index)
        w.columns.name = None
        wide[f] = w
    # 股本是"状态"而不是"成交"：停牌日也存在，向前填充
    for f in ["float_shares", "total_shares", "adj_factor"]:
        wide[f] = wide[f].ffill()
    return FactorPanel(**wide, info=info, fin=fin, bench=bench, is_synthetic=False)


# ==================== 真实数据下载 ====================


def _cached_df(name: str, fetch, retries: int = 2) -> pd.DataFrame:
    """单个接口的 parquet 缓存 + 重试"""
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    last = None
    for i in range(retries + 1):
        try:
            df = fetch()
            if df is None or df.empty:
                raise ValueError("返回空数据")
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return df
        except Exception as e:  # 网络异常种类繁多，统一重试
            last = e
            time.sleep(1.0 + i)
    raise RuntimeError(f"{name}: {type(last).__name__}: {str(last)[:120]}")


def _yjbb(d: pd.Timestamp) -> pd.DataFrame:
    import akshare as ak

    def fetch():
        df = ak.stock_yjbb_em(date=d.strftime("%Y%m%d"))
        df["最新公告日期"] = pd.to_datetime(df["最新公告日期"], errors="coerce")
        return df.drop(columns=["序号"]).astype({"股票代码": str})

    return _cached_df(f"factor_fin_yjbb_{d:%Y%m%d}", fetch)


def _yysj(d: pd.Timestamp) -> pd.DataFrame:
    import akshare as ak

    def fetch():
        df = ak.stock_yysj_em(symbol="沪深A股", date=d.strftime("%Y%m%d"))
        for c in ["首次预约时间", "实际披露时间"]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        return df[["股票代码", "股票简称", "首次预约时间", "实际披露时间"]].astype({"股票代码": str})

    return _cached_df(f"factor_fin_yysj_{d:%Y%m%d}", fetch)


def _looks_synthetic(code: str, df: pd.DataFrame) -> bool:
    """utils 下载失败时会返回合成行情：与同参数的 synthetic_ohlcv 对比即可识别"""
    synth = synthetic_ohlcv(code, PRICE_START, PRICE_END, start_price=100.0)["close"]
    return len(synth) == len(df) and bool(np.allclose(synth.to_numpy(), df["close"].to_numpy()))


def _fetch_stock(code: str) -> dict:
    """下载单只股票：行情 + 复权因子（utils）、逐日股本、上市日期"""
    import akshare as ak

    px = get_stock_daily_adj(code, PRICE_START, PRICE_END)
    if _looks_synthetic(code, px) or not (CACHE_DIR / f"factor_{code}.parquet").exists():
        raise RuntimeError("行情或复权因子下载失败")

    def value():
        v = ak.stock_value_em(symbol=code)
        v = v.rename(columns={"数据日期": "date", "总股本": "total_shares", "流通股本": "float_shares"})
        v["date"] = pd.to_datetime(v["date"])
        return v[["date", "total_shares", "float_shares"]].astype({"total_shares": float, "float_shares": float})

    def info():
        df = ak.stock_individual_info_em(symbol=code)
        return pd.DataFrame({"item": df["item"].astype(str), "value": df["value"].astype(str)})

    shares = _cached_df(f"factor_value_{code}", value).set_index("date").sort_index()
    meta = _cached_df(f"factor_info_{code}", info).set_index("item")["value"]
    return {"px": px, "shares": shares, "list_date": pd.to_datetime(meta.get("上市时间"), format="%Y%m%d"),
            "name": meta.get("股票简称", code)}


def _select_universe(latest_yjbb: pd.DataFrame, listed_2019: set[str]) -> pd.DataFrame:
    import akshare as ak

    cons = []
    for label, idx in [("沪深300", "000300"), ("中证500", "000905")]:
        df = _cached_df(f"factor_cons_{idx}", lambda idx=idx: ak.index_stock_cons_csindex(symbol=idx))
        cons.append(pd.DataFrame({"code": df["成分券代码"].astype(str), "name": df["成分券名称"], "index": label}))
    cons = pd.concat(cons).drop_duplicates("code")
    ind = latest_yjbb.drop_duplicates("股票代码").set_index("股票代码")["所处行业"]
    cons["industry"] = cons["code"].map(ind)
    cons["sector"] = cons["industry"].map(INDUSTRY_TO_SECTOR)
    cons = cons[cons["sector"].notna() & cons["code"].isin(listed_2019) & cons["code"].str[0].isin(["0", "3", "6"])]

    picked = []
    for _, g in cons.sort_values("code").groupby(["sector", "index"], sort=True):
        k = min(PER_SECTOR_PER_INDEX, len(g))
        pos = np.unique(np.linspace(0, len(g) - 1, k).round().astype(int))  # 按代码均匀抽样
        picked.append(g.iloc[pos])
    return pd.concat(picked).set_index("code").sort_index()


def build_real_panel() -> FactorPanel:
    """联网构建面板（首次约 2–4 分钟），失败时抛出异常"""
    t0 = time.time()
    print("📥 首次构建 Phase 7 面板数据（之后直接读取缓存）……")

    hs300 = get_index_daily("000300", INDEX_START, INDEX_END)
    if np.allclose(hs300["close"].iloc[:50], synthetic_ohlcv("000300", INDEX_START, INDEX_END, 4000.0, annual_vol=0.20)["close"].iloc[:50]):
        raise RuntimeError("沪深 300 指数下载失败")
    calendar = hs300.loc[pd.Timestamp(PRICE_START) : pd.Timestamp(PRICE_END)].index

    # 1. 财报（按报告期批量下载，4 线程）
    print(f"   [1/3] 财报：{len(FIN_PERIODS)} 个报告期 × (业绩报表 + 披露时间) ……")
    with contextlib.redirect_stderr(io.StringIO()), ThreadPoolExecutor(4) as ex:
        yjbb = dict(zip(FIN_PERIODS, ex.map(_yjbb, FIN_PERIODS)))
        yysj = dict(zip(FIN_PERIODS, ex.map(_yysj, FIN_PERIODS)))
    print(f"         完成，用时 {time.time() - t0:.0f}s")

    # 2. 股票池
    listed_2019 = set(yysj[pd.Timestamp("2019-12-31")]["股票代码"])
    universe = _select_universe(yjbb[FIN_PERIODS[-1]], listed_2019)
    print(f"   [2/3] 股票池：{len(universe)} 只（{universe['sector'].nunique()} 个行业），下载行情与股本 ……")

    # 3. 个股行情、股本、上市日期（4 线程）
    results, failed = {}, {}

    def task(code):
        try:
            return code, _fetch_stock(code), None
        except Exception as e:
            return code, None, f"{type(e).__name__}: {str(e)[:80]}"

    with ThreadPoolExecutor(4) as ex:
        for i, (code, res, err) in enumerate(ex.map(task, universe.index), 1):
            if res is None:
                failed[code] = err
            else:
                results[code] = res
            if i % 30 == 0:
                print(f"         {i}/{len(universe)}，用时 {time.time() - t0:.0f}s")
    if failed:
        print(f"   ⚠️ {len(failed)} 只股票下载失败已跳过：{', '.join(list(failed)[:8])}")
    if len(results) < MIN_STOCKS:
        raise RuntimeError(f"成功下载的股票只有 {len(results)} 只")

    codes = sorted(results)
    wide = {}
    for f in ["open", "high", "low", "close", "adj_factor", "volume", "amount"]:
        wide[f] = pd.DataFrame({c: results[c]["px"][f] for c in codes}).reindex(calendar)
    wide["adj_factor"] = wide["adj_factor"].ffill()
    for f in ["float_shares", "total_shares"]:
        s = pd.DataFrame({c: results[c]["shares"][f] for c in codes})
        wide[f] = s.reindex(s.index.union(calendar)).ffill().reindex(calendar)
    # 未上市 / 停牌：成交量为 0 的行也视为停牌
    no_trade = wide["volume"].isna() | (wide["volume"] <= 0)
    for f in ["open", "high", "low", "close", "volume", "amount"]:
        wide[f] = wide[f].mask(no_trade)

    info = universe.loc[codes].copy()
    info["name"] = [results[c]["name"] for c in codes]
    info["list_date"] = pd.to_datetime([results[c]["list_date"] for c in codes])
    info["board"] = [board_of(c) for c in codes]
    info = info[["name", "sector", "industry", "index", "board", "list_date"]]

    # 4. 财报长表：数值来自 yjbb，首次披露日来自 yysj
    rows = []
    for d in FIN_PERIODS:
        y = yjbb[d].drop_duplicates("股票代码").set_index("股票代码")
        s = yysj[d].drop_duplicates("股票代码").set_index("股票代码")
        y = y[y.index.isin(codes)]
        rows.append(pd.DataFrame({
            "code": y.index,
            "end_date": d,
            "ann_date": s["实际披露时间"].reindex(y.index).to_numpy(),
            "latest_ann_date": y["最新公告日期"].to_numpy(),
            "net_profit": pd.to_numeric(y["净利润-净利润"], errors="coerce").to_numpy(),
            "revenue": pd.to_numeric(y["营业总收入-营业总收入"], errors="coerce").to_numpy(),
            "eps": pd.to_numeric(y["每股收益"], errors="coerce").to_numpy(),
            "bvps": pd.to_numeric(y["每股净资产"], errors="coerce").to_numpy(),
            "roe": pd.to_numeric(y["净资产收益率"], errors="coerce").to_numpy(),
        }))
    fin = pd.concat(rows, ignore_index=True)
    fin["ann_source"] = np.where(fin["ann_date"].notna(), "实际披露", "截止日补齐")
    fin["ann_date"] = fin["ann_date"].fillna(report_deadline(fin["end_date"]))
    fin["total_shares_end"] = [
        results[c]["shares"]["total_shares"].asof(d) if d >= results[c]["shares"].index[0]
        else results[c]["shares"]["total_shares"].iloc[0]
        for c, d in zip(fin["code"], fin["end_date"])
    ]
    fin["equity"] = fin["bvps"] * fin["total_shares_end"]
    fin = fin.sort_values(["code", "end_date"]).reset_index(drop=True)

    # 5. 基准指数
    bench = pd.DataFrame(index=calendar)
    for name, idx in BENCH_CODES.items():
        df = hs300 if idx == "000300" else get_index_daily(idx, INDEX_START, INDEX_END)
        bench[f"{name}_open"] = df["open"].reindex(calendar)
        bench[f"{name}_close"] = df["close"].reindex(calendar)
    bench.index.name = "date"

    panel = FactorPanel(**wide, info=info, fin=fin, bench=bench)
    _save(panel)
    print(f"✅ 面板构建完成：{len(codes)} 只股票 × {len(calendar)} 个交易日，用时 {time.time() - t0:.0f}s")
    return _load_cached()


# ==================== 合成数据 ====================


def synthetic_panel(n: int = 120, seed: int = 7) -> FactorPanel:
    """
    合成截面面板：收益 = Beta × 市场 + 行业 + Σ 风格暴露 × 因子收益 + 特质噪声

    内置结构（只为让每课的方法"有东西可测"，参数并非真实市场估计）：
        价值（EP 高）+、质量（ROE 高）+、小市值 +（2024 年初有一次剧烈回撤）、
        低波 +、低换手 +、短期反转（特质价格中有均值回复成分）、动量 ≈ 0、
        财报公告日按"业绩超预期程度"跳空（用于 08 课演示按报告期对齐的虚假 IC）
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", PRICE_END, name="date")
    T, S = len(dates), list(SECTORS)
    month_id = dates.to_period("M")
    months = month_id.unique()

    # ---- 股票静态属性 ----
    prefixes = ["600", "601", "603", "000", "002", "300", "688"]
    counts = [30, 15, 10, 20, 20, 15, 10]
    codes = []
    for p, c in zip(prefixes, counts):
        codes += [f"{p}{i:03d}" for i in rng.choice(np.arange(1, 999), size=c, replace=False)]
    codes = sorted(codes[:n])
    n = len(codes)
    sector = np.array(S)[rng.permutation(np.arange(n) % len(S))]
    log_mcap0 = rng.normal(np.log(4e10), 0.9, n)
    beta = rng.uniform(0.6, 1.4, n)
    size_rank = pd.Series(-log_mcap0).rank(pct=True).to_numpy()
    sigma = 0.011 + 0.012 * size_rank + rng.uniform(0, 0.006, n)  # 小市值特质波动更大
    float_ratio = rng.uniform(0.5, 1.0, n)
    late = rng.choice(n, 12, replace=False)  # 12 只样本期内上市的次新股
    lo, hi = dates.searchsorted(pd.Timestamp("2019-03-01")), dates.searchsorted(pd.Timestamp("2021-06-30"))
    list_date = pd.DatetimeIndex([dates[rng.integers(lo, hi)] if i in late else pd.Timestamp("2015-01-05")
                                  for i in range(n)])
    st_stocks = rng.choice(np.setdiff1d(np.arange(n), late), 3, replace=False)
    board = np.array([board_of(c) for c in codes])

    # ---- 潜在特征（月度 AR(1)，缓慢变化）----
    def ar1(phi, size):
        x = np.zeros((len(months), size))
        x[0] = rng.normal(size=size)
        for t in range(1, len(months)):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi**2) * rng.normal(size=size)
        return x

    value_z, quality_z, sent_z = ar1(0.95, n), ar1(0.97, n), ar1(0.90, n)
    ep_lat = 0.055 + 0.035 * value_z  # 盈利收益率（可以为负 = 亏损）
    roe_lat = np.clip(0.10 + 0.05 * quality_z, 0.02, None)

    # ---- 财报日程与业绩超预期（先生成，供价格模拟使用）----
    periods = FIN_PERIODS
    windows = {3: ((0, 4, 10), (0, 4, 30)), 6: ((0, 7, 20), (0, 8, 31)),
               9: ((0, 10, 15), (0, 10, 31)), 12: ((1, 3, 1), (1, 4, 30))}
    surprise = rng.normal(size=(len(periods), n))
    ann = np.empty((len(periods), n), dtype="datetime64[ns]")
    for k, d in enumerate(periods):
        (y0, m0, d0), (y1, m1, d1) = windows[d.month]
        lo, hi = pd.Timestamp(d.year + y0, m0, d0), pd.Timestamp(d.year + y1, m1, d1)
        days = pd.bdate_range(lo, hi)
        ann[k] = days[rng.integers(0, len(days), n)].to_numpy()
    jump = np.zeros((T, n))
    for k in range(len(periods)):
        pos = dates.searchsorted(ann[k], side="right")  # 盘后公告 → 下一交易日反应
        ok = pos < T
        jump[pos[ok], np.where(ok)[0]] += 0.05 * surprise[k, ok]

    # ---- 因子收益 ----
    mkt = rng.normal(0.0003, 0.012, T)
    ind_ret = rng.normal(0, 0.005, (T, len(S)))
    size_mu = np.where(dates.year < 2021, 0.0002, 0.0004)
    crash = (dates >= "2024-01-02") & (dates <= "2024-02-07")
    size_mu = np.where(crash, -0.003, size_mu)  # 模拟 2024 年初小盘股踩踏
    size_mu = np.where((dates > "2024-02-07") & (dates <= "2024-03-15"), 0.003, size_mu)
    f = {
        "value": rng.normal(0.00024, 0.002, T),
        "quality": rng.normal(0.00016, 0.0015, T),
        "size": size_mu + rng.normal(0, 0.003, T),
        "vol": rng.normal(-0.00035, 0.002, T),
        "sent": rng.normal(-0.00020, 0.002, T),
    }

    # ---- 逐日模拟（月度更新风格暴露）----
    sec_idx = np.array([S.index(s) for s in sector])
    phi = 0.5 ** (1 / 8)  # 特质均值回复成分，半衰期约 8 天 → 短期反转
    u = np.zeros(n)
    log_p = np.log(20.0) + rng.normal(0, 0.5, n)
    log_mcap = log_mcap0.copy()
    ret = np.zeros((T, n))
    limit = np.where(board == "主板", 0.10, 0.20)
    st_window = {}
    for j, s in enumerate(st_stocks):
        a = dates.get_loc(pd.Timestamp("2020-03-02")) + 120 * j
        st_window[s] = (a, a + 250)

    def z(x):
        return (x - x.mean()) / x.std()

    vol_z = z(sigma)
    for mi, m in enumerate(months):
        rows = np.where(month_id == m)[0]
        size_z = -z(log_mcap)
        expo_mu = (f["value"][rows, None] * value_z[mi] + f["quality"][rows, None] * quality_z[mi]
                   + f["size"][rows, None] * size_z + f["vol"][rows, None] * vol_z
                   + f["sent"][rows, None] * sent_z[mi])
        for k, t in enumerate(rows):
            eta = rng.normal(0, sigma)
            u_new = phi * u + 0.6 * sigma * rng.normal(size=n)
            r = beta * mkt[t] + ind_ret[t, sec_idx] + expo_mu[k] + eta + (u_new - u) + jump[t]
            lim = limit.copy()
            for s, (a, b) in st_window.items():
                if a <= t < b:
                    lim[s] = 0.05
                    r[s] -= 0.002
            if t > 0:
                r = np.clip(r, -lim, lim)
            ret[t] = r
            u = u_new
            log_mcap += np.log1p(r)
    adj_close = np.exp(log_p) * np.cumprod(1 + ret, axis=0)

    # ---- 除息：每年 6–7 月派息一次，后复权因子随之增加 ----
    adj_factor = np.ones((T, n))
    for yr in range(2018, 2025):
        days = np.where((dates.year == yr) & ((dates.month == 6) | (dates.month == 7)))[0]
        ex_day = rng.choice(days, n)
        dy = np.clip(0.015 + 0.01 * value_z[months.get_loc(pd.Period(f"{yr}-06", "M"))], 0.0, 0.05)
        for j in range(n):
            adj_factor[ex_day[j]:, j] /= 1 - dy[j]
    close = np.round(adj_close / adj_factor, 2)

    # ---- 开盘价、高低价（含一字涨停）----
    prev = np.vstack([close[0], close[:-1]])
    lim_arr = np.tile(limit, (T, 1))
    for s, (a, b) in st_window.items():
        lim_arr[a:b, s] = 0.05
    up, dn = np.round(prev * (1 + lim_arr), 2), np.round(prev * (1 - lim_arr), 2)
    open_ = np.clip(np.round(prev * (1 + 0.3 * ret + rng.normal(0, 0.3, (T, n)) * sigma), 2), dn, up)
    yizi = (ret >= lim_arr - 1e-9) & (rng.random((T, n)) < 0.4)
    open_ = np.where(yizi, close, open_)
    spread = np.abs(rng.normal(0, 0.4, (T, n))) * sigma
    high = np.minimum(np.maximum(open_, close) * (1 + spread), np.maximum(up, close))
    low = np.maximum(np.minimum(open_, close) * (1 - spread), np.minimum(dn, close))
    high, low = np.round(high, 2), np.round(low, 2)

    # ---- 股本、换手 ----
    total_shares = np.exp(log_mcap0) / close[0]
    float_shares = total_shares * float_ratio
    base_turn = 0.008 * np.exp(0.6 * size_z_last(log_mcap0) + 0.3 * vol_z)
    turn = base_turn * np.exp(0.35 * sent_z[months.get_indexer(month_id)]) \
        * (1 + 12 * np.abs(ret)) * rng.lognormal(0, 0.25, (T, n))
    volume = np.round(turn * float_shares / 100) * 100
    amount = volume * (open_ + close) / 2

    def frame(x):
        return pd.DataFrame(x, index=dates, columns=codes)

    wide = {"open": frame(open_), "high": frame(high), "low": frame(low), "close": frame(close),
            "adj_factor": frame(adj_factor), "volume": frame(volume), "amount": frame(amount),
            "float_shares": frame(np.tile(float_shares, (T, 1))),
            "total_shares": frame(np.tile(total_shares, (T, 1)))}

    # ---- 停牌、未上市 ----
    gone = pd.DataFrame(False, index=dates, columns=codes)
    for j, c in enumerate(codes):
        gone.loc[: list_date[j] - pd.Timedelta(days=1), c] = True
    for _ in range(20):
        j, a = rng.integers(n), rng.integers(T - 70)
        gone.iloc[a : a + rng.integers(3, 60), j] = True
    for fld in ["open", "high", "low", "close", "volume", "amount"]:
        wide[fld] = wide[fld].mask(gone)

    info = pd.DataFrame({
        "name": [f"合成{c}" for c in codes], "sector": sector, "industry": sector,
        "index": np.where(log_mcap0 >= np.median(log_mcap0), "沪深300", "中证500"),
        "board": board, "list_date": list_date,
    }, index=pd.Index(codes, name="code"))

    # ---- 财报：季度利润 = EP × 季末市值 / 4 × (1 + 超预期) ----
    mcap_daily = pd.DataFrame(np.exp(log_mcap0) * adj_close / adj_close[0], index=dates, columns=codes)
    rows = []
    for k, d in enumerate(periods):
        mi = months.get_loc(pd.Period(min(d, dates[-1]), "M")) if d >= dates[0] else 0
        mc = mcap_daily.asof(d).to_numpy() if d >= dates[0] else np.exp(log_mcap0)
        q_profit = ep_lat[mi] * mc / 4 * (1 + 0.6 * surprise[k])
        equity = np.maximum(ep_lat[mi], 0.01) / roe_lat[mi] * mc
        rows.append(pd.DataFrame({"code": codes, "end_date": d, "ann_date": ann[k], "q_profit": q_profit,
                                  "equity": equity}))
    fin = pd.concat(rows, ignore_index=True).sort_values(["code", "end_date"])
    fin["net_profit"] = fin.groupby(["code", fin["end_date"].dt.year])["q_profit"].cumsum()
    fin["revenue"] = fin["net_profit"] / 0.12
    shares_map = dict(zip(codes, total_shares))
    fin["total_shares_end"] = fin["code"].map(shares_map)
    fin["eps"] = fin["net_profit"] / fin["total_shares_end"]
    fin["bvps"] = fin["equity"] / fin["total_shares_end"]
    fin["roe"] = fin["net_profit"] / fin["equity"] * 100
    revised = rng.random(len(fin)) < 0.3  # 部分报告在一年后随同期报告"重述"
    fin["latest_ann_date"] = np.where(revised, fin["ann_date"] + pd.Timedelta(days=365), fin["ann_date"])
    fin["latest_ann_date"] = pd.to_datetime(fin["latest_ann_date"])
    fin["ann_source"] = "实际披露"
    fin = fin[fin["ann_date"] >= fin["code"].map(dict(zip(codes, list_date)))]  # 上市前没有定期报告
    fin = fin.drop(columns="q_profit").reset_index(drop=True)

    # ---- 基准：沪深 300 ≈ 市场因子；中证 500 / 1000 额外暴露小市值因子 ----
    def index_path(level, beta_m, beta_s):
        r = beta_m * mkt + beta_s * f["size"] + rng.normal(0, 0.002, T)
        c = level * np.cumprod(1 + r)
        o = np.concatenate([[level], c[:-1]]) * (1 + 0.2 * r)
        return o, c

    bench = pd.DataFrame(index=dates)
    for name, (lv, bm, bs) in {"hs300": (3800, 1.0, 0.0), "zz500": (6000, 1.05, 0.5), "zz1000": (6500, 1.1, 0.9)}.items():
        o, c = index_path(lv, bm, bs)
        bench[f"{name}_open"], bench[f"{name}_close"] = o, c

    return FactorPanel(**wide, info=info, fin=fin, bench=bench, is_synthetic=True)


def size_z_last(log_mcap: np.ndarray) -> np.ndarray:
    """小市值暴露（市值越小值越大）"""
    return -(log_mcap - log_mcap.mean()) / log_mcap.std()


# ==================== 对外接口 ====================

_PANEL: FactorPanel | None = None


def load_panel(verbose: bool = True) -> FactorPanel:
    """读缓存 → 联网构建 → 失败则合成。同一进程内只加载一次"""
    global _PANEL
    if _PANEL is not None:
        return _PANEL
    if all(p.exists() for p in [PANEL_FILE, INFO_FILE, FIN_FILE, BENCH_FILE]):
        _PANEL = _load_cached()  # 有缓存时，无论是否离线都读取真实面板（与 utils.data 口径一致）
    elif is_offline():
        warn_synthetic("多因子面板", "离线模式 QTC_OFFLINE=1 且无本地缓存")
        _PANEL = synthetic_panel()
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _PANEL = build_real_panel()
        except Exception as e:
            warn_synthetic("多因子面板", f"真实数据构建失败：{type(e).__name__}: {str(e)[:100]}")
            _PANEL = synthetic_panel()
    if verbose:
        p = _PANEL
        tag = "合成数据" if p.is_synthetic else "真实数据"
        print(f"📦 面板（{tag}）：{len(p.codes)} 只股票，{p.dates[0].date()} ~ {p.dates[-1].date()}，"
              f"{len(p.dates)} 个交易日，{p.info['sector'].nunique()} 个行业")
    return _PANEL


def load_delisted_sample(max_n: int = 20) -> pd.DataFrame:
    """
    2019–2024 年退市的沪深主板股票样本（用于 03 课演示幸存者偏差）

    返回 DataFrame[code, name, delist_date, first_date, last_date, total_return]，
    total_return 为样本期首个交易日到最后交易日的不复权收盘价涨跌幅（粗略，未计分红）。
    离线且无缓存或失败时返回空表。
    """
    path = CACHE_DIR / "factor_delisted_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if is_offline():
        return pd.DataFrame()
    try:
        import akshare as ak

        sh = _cached_df("factor_delist_sh", ak.stock_info_sh_delist)
        sz = _cached_df("factor_delist_sz", lambda: ak.stock_info_sz_delist(symbol="终止上市公司"))
        lst = pd.concat([
            pd.DataFrame({"code": sh["公司代码"].astype(str), "name": sh["公司简称"], "delist_date": sh["暂停上市日期"]}),
            pd.DataFrame({"code": sz["证券代码"].astype(str), "name": sz["证券简称"], "delist_date": sz["终止上市日期"]}),
        ])
        lst["delist_date"] = pd.to_datetime(lst["delist_date"], errors="coerce")
        lst = lst[(lst["delist_date"] >= "2019-07-01") & (lst["delist_date"] <= "2024-12-31")
                  & lst["code"].str[:2].isin(["00", "60"])].sort_values("delist_date")
        pick = lst.iloc[np.unique(np.linspace(0, len(lst) - 1, min(max_n, len(lst))).round().astype(int))]
        out = []
        for _, row in pick.iterrows():
            try:
                raw = ak.stock_zh_a_hist(symbol=row["code"], period="daily", start_date="20190101",
                                         end_date="20241231", adjust="")
                if raw.empty:
                    continue
                c = raw.set_index(pd.to_datetime(raw["日期"]))["收盘"].astype(float)
                out.append({**row.to_dict(), "first_date": c.index[0], "last_date": c.index[-1],
                            "total_return": c.iloc[-1] / c.iloc[0] - 1})
                time.sleep(0.2)
            except Exception:
                continue
        df = pd.DataFrame(out)
        if not df.empty:
            df["n_delisted_total"] = len(lst)
            df.to_parquet(path)
        return df
    except Exception as e:
        print(f"⚠️ 退市股票数据获取失败（{type(e).__name__}），跳过")
        return pd.DataFrame()


if __name__ == "__main__":
    p = load_panel()
    print(p.info["sector"].value_counts().to_string())
    print(p.fin.head())
