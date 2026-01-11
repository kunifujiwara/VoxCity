"""Small compatibility module matching `voxcity.simulator.utils`.

VoxCity's `voxcity.simulator` flattens `utils` into the toplevel namespace.
Some user code may rely on these names existing after:

    import simulator_gpu as simulator

This module keeps that behavior without pulling in VoxCity.
"""

from datetime import datetime

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def dummy_function(test_string):
    return test_string


__all__ = ["np", "pd", "datetime", "dummy_function"]
