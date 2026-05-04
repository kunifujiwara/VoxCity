"""Grid orientation helpers.

Contract:
- Core Phase 3 processing uses uv_m/SOUTH_UP: axis 0 = u/north (row 0 = southern origin edge),
  increasing row index moves north. Columns increase eastward: column 0 is
  west/left and indices increase toward the east/right. All processing functions
  accept and return 2D grids in this orientation unless explicitly documented
  otherwise.
- Visualization utilities may flip vertically for display purposes only.
- 3D indexing follows (row, col, z) = (north→south, west→east, ground→up).

Utilities here are intentionally minimal to avoid introducing hidden behavior.
They can be used at I/O boundaries (e.g., when reading rasters with south_up
conventions) to normalize to the internal orientation.

Coordinate-frame vocabulary (see also voxcity.utils.projector):
  uv cell — cell index (i, j) in uv_m/SOUTH_UP (Phase 3); row 0 is south/origin.
  legacy ij_south — same layout as uv_m/SOUTH_UP; kept for backward compatibility.
  ensure_orientation() is the only legitimate place to convert between the two.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

# Public constants to reference orientation in docs and code
ORIENTATION_NORTH_UP: Literal["north_up"] = "north_up"
ORIENTATION_SOUTH_UP: Literal["south_up"] = "south_up"


def ensure_orientation(
    grid: np.ndarray,
    orientation_in: Literal["north_up", "south_up"],
    orientation_out: Literal["north_up", "south_up"] = ORIENTATION_NORTH_UP,
) -> np.ndarray:
    """Return ``grid`` converted from ``orientation_in`` to ``orientation_out``.

    Both orientations are defined for 2D arrays as:
    - north_up: row 0 = north/top, last row = south/bottom
    - south_up: row 0 = south/bottom, last row = north/top

    If orientations match, the input array is returned unchanged. When converting
    between north_up and south_up, a vertical flip is applied using ``np.flipud``.

    Notes
    -----
    - This function does not copy when no conversion is needed.
    - Use at data boundaries (read/write, interop) rather than deep in processing code.
    """
    if orientation_in == orientation_out:
        return grid
    # Only two orientations supported; converting between them is a vertical flip
    return np.flipud(grid)


