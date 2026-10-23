"""
FABDEM v1-2 read without any account.

FABDEM (Hawker et al., 2022) is Copernicus GLO-30 with forests and buildings
removed. The University of Bristol serves it as 10 x 10 degree zip archives
of 1 x 1 degree GeoTIFFs. GDAL reads a single tile out of a remote zip with
HTTP range requests (``/vsizip//vsicurl/``), so only the window around the
area is transferred, not the 1-2 GB archive.

Licence: CC BY-NC-SA 4.0 (non-commercial). Cite Hawker, L. et al. (2022),
"A 30 m global map of elevation with forests and buildings removed",
Environmental Research Letters 17 024016.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from ..utils.logging import get_logger
from .cog import _lat_tag, _lon_tag, fetch_cog_crop, tile_origins_for

__all__ = ["fabdem_paths", "fetch_fabdem", "FABDEM_BASE_URL"]

_logger = get_logger(__name__)

FABDEM_BASE_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"


def _fabdem_zip_name(lat0: int, lon0: int) -> str:
    """Archive holding the 1-degree tile at (lat0, lon0): named by its 10-degree corners."""
    zlat = math.floor(lat0 / 10) * 10
    zlon = math.floor(lon0 / 10) * 10
    return f"{_lat_tag(zlat)}{_lon_tag(zlon)}-{_lat_tag(zlat + 10)}{_lon_tag(zlon + 10)}_FABDEM_V1-2.zip"


def _fabdem_member_name(lat0: int, lon0: int) -> str:
    """GeoTIFF name of the 1-degree tile with south-west corner (lat0, lon0)."""
    return f"{_lat_tag(lat0)}{_lon_tag(lon0)}_FABDEM_V1-2.tif"


def fabdem_paths(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = FABDEM_BASE_URL,
) -> List[str]:
    """GDAL paths of the FABDEM tiles covering the area.

    Not probed: a tile missing from its archive (open ocean) simply fails to
    open and is skipped by fetch_cog_crop.
    """
    base = base_url.rstrip("/")
    return [
        f"/vsizip//vsicurl/{base}/{_fabdem_zip_name(lat, lon)}/{_fabdem_member_name(lat, lon)}"
        for lat, lon in tile_origins_for(rectangle_vertices, 1, "FABDEM")
    ]


def fetch_fabdem(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = FABDEM_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from FABDEM (no login). Nodata cells are
    filled from the nearest valid pixel. Returns None when no tile could be
    read.
    """
    paths = fabdem_paths(rectangle_vertices, base_url=base_url)
    if not paths:
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~120 m of padding at 1 arc-second, as for Copernicus.
    return fetch_cog_crop(paths, rectangle_vertices, out_path, pad_pixels=4, fill_nodata=True)
