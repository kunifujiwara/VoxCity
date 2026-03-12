"""Backward-compatible re-exports from :mod:`voxcity.io`.

All I/O functions now live in ``voxcity.io``.  This shim ensures that
existing imports like ``from voxcity.generator.io import save_voxcity``
continue to work.
"""

from ..io import (  # noqa: F401
    load_voxcity,
    save_voxcity,
    save_results_h5,
    load_results_h5,
)
