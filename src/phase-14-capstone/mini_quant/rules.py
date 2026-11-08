"""
交易规则适配器（横切关注点）

板块规则复用 Phase 4 bt_core.board_rule，只补一个缺口：
创业板新代码段 302xxx（如 2024 年更名换码的 302132）在 bt_core 中会被识别为主板（10%），
这里按创业板规则处理（20% 涨跌幅、100 股整数倍）。不修改 Phase 4 文件。
"""

from __future__ import annotations

from . import _paths  # noqa: F401

from bt_core import BoardRule, board_rule as _bt_board_rule  # noqa: E402

CHINEXT_EXTRA_PREFIXES = ("302",)


def board_rule(symbol: str, is_st: bool = False, date=None) -> BoardRule:
    if str(symbol).startswith(CHINEXT_EXTRA_PREFIXES):
        return BoardRule("创业板", 0.20, 100, 100)
    return _bt_board_rule(symbol, is_st=is_st, date=date)
