import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetcher import SyntheticFetcher  # noqa: E402
from storage import Warehouse  # noqa: E402


@pytest.fixture
def fetcher():
    return SyntheticFetcher()


@pytest.fixture
def warehouse(tmp_path):
    return Warehouse(tmp_path / "warehouse")
