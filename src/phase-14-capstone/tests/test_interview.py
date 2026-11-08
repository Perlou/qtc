"""面试编程题（07 课）的 pytest 版本：与暴力解法对拍"""

import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "07-quant-career-interview.py"
_spec = importlib.util.spec_from_file_location("lesson_07_interview", _PATH)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)  # 课程主体在 if __name__ == "__main__" 下，导入不会运行


def test_max_drawdown_example():
    assert lesson.max_drawdown([1.0, 1.2, 0.9, 1.1, 0.8, 1.3]) == pytest.approx(1 / 3)
    assert lesson.max_drawdown([1.0, 1.1, 1.2]) == 0.0
    with pytest.raises(ValueError):
        lesson.max_drawdown([])


@pytest.mark.parametrize("seed", range(5))
def test_max_drawdown_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    nav = list(np.cumprod(1 + rng.normal(0, 0.05, 60)))
    assert lesson.max_drawdown(nav) == pytest.approx(lesson.max_drawdown_bruteforce(nav))


@pytest.mark.parametrize("seed", range(5))
def test_sliding_max_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    xs = list(rng.integers(0, 5, 40))  # 大量重复值
    for k in (1, 2, 7, 40):
        assert lesson.sliding_max(xs, k) == lesson.sliding_max_bruteforce(xs, k)


def test_rank_ic_by_date():
    df = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 4,
        "factor": [1, 2, 3, 4, 1, 2, 3, 4],
        "fwd_ret": [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
    })
    ic = lesson.rank_ic_by_date(df)
    assert ic["d1"] == pytest.approx(1.0) and ic["d2"] == pytest.approx(-1.0)


def test_reservoir_sample_size_and_membership():
    s = lesson.reservoir_sample(range(1000), 10, random.Random(0))
    assert len(s) == 10 and len(set(s)) == 10 and all(0 <= x < 1000 for x in s)
    assert lesson.reservoir_sample(range(3), 10, random.Random(0)) == [0, 1, 2]


def test_lru_cache():
    c = lesson.LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None and c.get("c") == 3


def test_analytic_helpers():
    assert lesson.expected_flips_analytic("HH") == 6
    assert lesson.expected_flips_analytic("HT") == 4
    assert lesson.expected_flips_analytic("HTH") == 10
    assert float(lesson.dice_reroll_value(1)) == pytest.approx(4.25)
