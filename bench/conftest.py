import os
import sys
from pathlib import Path

# Make tests/ importable so benchmarks reuse the shared make_city factory
# (tests/conftest.py::make_city). This intentionally duplicates none of its
# logic -- it just extends sys.path so `from conftest import make_city` finds
# the real tests/conftest.py module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

# tests/conftest.py sets NUMBA_DISABLE_JIT=1 for fast unit tests. Benchmarks
# exist to measure the JIT-compiled kernels, so restore the caller's setting
# (default: JIT enabled) before numba is first imported by any bench module.
_pre_import_jit = os.environ.get("NUMBA_DISABLE_JIT")

import pytest
from conftest import make_city  # tests/conftest.py

if _pre_import_jit is None:
    os.environ["NUMBA_DISABLE_JIT"] = "0"
else:
    os.environ["NUMBA_DISABLE_JIT"] = _pre_import_jit


@pytest.fixture
def small_city():
    return make_city(shape=(32, 32, 24), meshsize=2.0)
