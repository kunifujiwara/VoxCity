import sys
from pathlib import Path

# Make tests/ importable so benchmarks reuse the shared make_city factory
# (tests/conftest.py::make_city). This intentionally duplicates none of its
# logic -- it just extends sys.path so `from conftest import make_city` finds
# the real tests/conftest.py module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import pytest
from conftest import make_city  # tests/conftest.py


@pytest.fixture
def small_city():
    return make_city(shape=(32, 32, 24), meshsize=2.0)
