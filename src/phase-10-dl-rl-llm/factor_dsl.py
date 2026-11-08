"""
Phase 10 辅助模块：安全的因子表达式 DSL
======================================

永远不要 eval() LLM 生成的代码。本模块用 Python ast 解析一个受限的表达式语言，
只允许白名单内的语法节点、字段与算子，并在**解析阶段**拒绝未来函数（窗口必须是正整数常量）。

语法：
    字段     open high low close volume amount turnover vwap ret   （日频宽表：index=日期，columns=股票）
    常量     整数 / 小数
    运算     + - * /  一元负号
    截面算子 rank(x)  zscore(x)                       （同一天所有股票之间）
    逐元素   abs(x)  log(x)  sign(x)
    时序算子 delay(x, n)  delta(x, n)  ts_mean(x, n)  ts_std(x, n)  ts_sum(x, n)
             ts_max(x, n)  ts_min(x, n)  ts_rank(x, n)  corr(x, y, n)
             n 必须是 1..MAX_WINDOW 的整数常量；delay(x, -1) 之类的负窗口 = 读取未来 → 拒绝

被拒绝的语法（示例）：属性访问 close.shift(-1)、下标 close[0]、关键字参数、lambda、
推导式、比较运算、import / __import__、未知函数与字段、过深或过大的表达式。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "amount", "turnover", "vwap", "ret")
MAX_WINDOW = 250
MAX_NODES = 60
MAX_DEPTH = 10

# 名称 → (序列参数个数, 是否带窗口参数)
FUNCS = {
    "rank": (1, False), "zscore": (1, False), "abs": (1, False), "log": (1, False), "sign": (1, False),
    "delay": (1, True), "delta": (1, True), "ts_mean": (1, True), "ts_std": (1, True), "ts_sum": (1, True),
    "ts_max": (1, True), "ts_min": (1, True), "ts_rank": (1, True), "corr": (2, True),
}
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


class DSLError(ValueError):
    """表达式不合法（语法、白名单或未来函数）"""


@dataclass
class ExprInfo:
    expr: str
    fields: set
    funcs: set
    lookback: int  # 最长回看窗口（粗略：各层窗口之和），决定需要多少历史数据"预热"
    n_nodes: int


def _window(node: ast.AST) -> int:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        raise DSLError("窗口为负数：负偏移 = 读取未来数据（未来函数），禁止")
    if not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        raise DSLError("窗口参数必须是整数常量")
    v = node.value
    if float(v) != int(v):
        raise DSLError(f"窗口必须是整数，得到 {v}")
    v = int(v)
    if v < 1:
        raise DSLError(f"窗口必须 ≥ 1，得到 {v}（0 或负数没有意义或指向未来）")
    if v > MAX_WINDOW:
        raise DSLError(f"窗口 {v} 超过上限 {MAX_WINDOW}")
    return v


def validate(expr: str) -> ExprInfo:
    """解析并做白名单校验；不合法时抛出 DSLError"""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("表达式为空")
    if len(expr) > 400:
        raise DSLError("表达式过长")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"语法错误：{e.msg}") from None
    fields, funcs, count = set(), set(), [0]

    def visit(node, depth) -> int:
        """返回该子树的回看长度"""
        count[0] += 1
        if count[0] > MAX_NODES:
            raise DSLError(f"表达式节点数超过 {MAX_NODES}")
        if depth > MAX_DEPTH:
            raise DSLError(f"嵌套深度超过 {MAX_DEPTH}")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in BINOPS:
                raise DSLError(f"不允许的运算符：{type(node.op).__name__}")
            return max(visit(node.left, depth + 1), visit(node.right, depth + 1))
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise DSLError(f"不允许的一元运算：{type(node.op).__name__}")
            return visit(node.operand, depth + 1)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DSLError(f"只允许数值常量，得到 {node.value!r}")
            return 0
        if isinstance(node, ast.Name):
            if node.id not in FIELDS:
                raise DSLError(f"未知字段：{node.id}（可用：{', '.join(FIELDS)}）")
            fields.add(node.id)
            return 0
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DSLError("只允许直接调用白名单函数（禁止属性访问与方法调用，如 x.shift、os.system）")
            name = node.func.id
            if name not in FUNCS:
                raise DSLError(f"未知函数：{name}")
            if node.keywords:
                raise DSLError("不允许关键字参数")
            n_series, has_win = FUNCS[name]
            expect = n_series + int(has_win)
            if len(node.args) != expect or any(isinstance(a, ast.Starred) for a in node.args):
                raise DSLError(f"{name} 需要 {expect} 个参数，得到 {len(node.args)} 个")
            funcs.add(name)
            inner = max(visit(a, depth + 1) for a in node.args[:n_series])
            if has_win:
                count[0] += 1
                return inner + _window(node.args[-1])
            return inner
        raise DSLError(f"不允许的语法：{type(node).__name__}")

    lookback = visit(tree, 0)
    if not fields:
        raise DSLError("表达式没有引用任何字段")
    return ExprInfo(expr.strip(), fields, funcs, lookback, count[0])


# ==================== 求值 ====================


def _ts(x, n, fn):
    if not isinstance(x, pd.DataFrame):
        raise DSLError("时序算子的输入必须是字段或字段表达式")
    return fn(x.rolling(n, min_periods=n))


def _impl(name: str, args: list):
    x = args[0]
    if name == "rank":
        return x.rank(axis=1, pct=True)
    if name == "zscore":
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
    if name == "abs":
        return x.abs()
    if name == "log":
        return np.log(x.where(x > 0))
    if name == "sign":
        return np.sign(x)
    n = int(args[-1])
    if name == "delay":
        return x.shift(n)
    if name == "delta":
        return x - x.shift(n)
    if name == "ts_mean":
        return _ts(x, n, lambda r: r.mean())
    if name == "ts_std":
        return _ts(x, n, lambda r: r.std())
    if name == "ts_sum":
        return _ts(x, n, lambda r: r.sum())
    if name == "ts_max":
        return _ts(x, n, lambda r: r.max())
    if name == "ts_min":
        return _ts(x, n, lambda r: r.min())
    if name == "ts_rank":
        return _ts(x, n, lambda r: r.rank(pct=True))
    if name == "corr":
        y = args[1]
        return x.rolling(n, min_periods=n).corr(y)
    raise DSLError(name)


def evaluate(expr: str, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """先校验，再在宽表上求值；返回 index=日期、columns=股票的因子宽表（inf → NaN）"""
    info = validate(expr)
    tree = ast.parse(info.expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            op = BINOPS[type(node.op)]
            return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[op]()
        if isinstance(node, ast.UnaryOp):
            return -ev(node.operand)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return data[node.id]
        if isinstance(node, ast.Call):
            name = node.func.id
            n_series, has_win = FUNCS[name]
            args = [ev(a) for a in node.args[:n_series]]
            if not all(isinstance(a, pd.DataFrame) for a in args):
                raise DSLError(f"{name} 的序列参数必须是字段表达式，不能是常量")
            if has_win:
                args.append(node.args[-1].value)
            return _impl(name, args)
        raise DSLError(type(node).__name__)

    out = ev(tree)
    if not isinstance(out, pd.DataFrame):
        raise DSLError("表达式结果必须是因子宽表（不能是单个常量）")
    return out.replace([np.inf, -np.inf], np.nan)


def lookahead_test(expr_or_fn, data: dict[str, pd.DataFrame], cut: pd.Timestamp, n_check: int = 20) -> bool:
    """
    前视检测：只用 cut 及之前的数据计算一次，再用全部数据计算一次，
    比较 cut 之前最后 n_check 天的因子值 —— 无未来函数时两者必须完全一致
    expr_or_fn：DSL 表达式字符串，或 f(data) -> 宽表 的普通函数（用于检测手写代码）
    """
    calc = (lambda d: evaluate(expr_or_fn, d)) if isinstance(expr_or_fn, str) else expr_or_fn
    full = calc(data)
    trunc = calc({k: v.loc[:cut] for k, v in data.items()})
    a = full.loc[:cut].iloc[-n_check:]
    b = trunc.iloc[-n_check:]
    return bool(np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))


def factor_ic(expr: str, data: dict, dates: pd.DatetimeIndex, fwd: pd.DataFrame,
              mask: pd.DataFrame | None = None, min_obs: int = 20) -> pd.Series:
    """表达式 → 在信号日取值 → (可交易过滤) → 与未来收益的逐期 Rank IC"""
    from factor_tools import calc_ic

    f = evaluate(expr, data).reindex(dates)
    if mask is not None:
        f = f.where(mask.reindex(index=dates, columns=f.columns).fillna(False).astype(bool))
    return calc_ic(f, fwd.reindex(dates), method="rank", min_obs=min_obs)
