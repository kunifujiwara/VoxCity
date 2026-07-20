"""Empirical orientation guard: the permanent, cheap version of the
building-centroid correlation test that caught the voxcity_vwind axis swap.

A building at the NE corner of the AOI MUST land at high-i (axis 0 = north)
and high-j (axis 1 = east) in the grids. If this test fails, an orientation
flip or transpose has been introduced somewhere in the rasterization path —
do NOT fix the test; find the flip.
"""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from voxcity.geoprocessor.raster.buildings import create_building_height_grid_from_gdf_polygon


# Non-square AOI so a transpose changes grid.shape and cannot slip through:
# north/latitude extent 0.0018 (~200 m, ~20 cells) is HALF the east/longitude
# extent 0.0036 (~400 m, ~40 cells). RECT order is [SW, NW, NE, SE].
RECT = [(0.0, 0.0), (0.0, 0.0018), (0.0036, 0.0018), (0.0036, 0.0)]
# NE corner, deliberately asymmetric: deep into the north (high i, ~88% up)
# but only moderately into the east (~69% across), so i-mass and j-mass differ
# and no flip/transpose can map the footprint onto an equivalent cell block.
NE_BUILDING = Polygon([
    (0.0022, 0.0015), (0.0022, 0.0017), (0.0028, 0.0017), (0.0028, 0.0015),
])


def test_ne_corner_building_lands_at_high_i_high_j():
    gdf = gpd.GeoDataFrame({"height": [20.0]}, geometry=[NE_BUILDING], crs="EPSG:4326")
    grid, _min_h, ids, _filtered = create_building_height_grid_from_gdf_polygon(
        gdf, 10.0, RECT
    )
    ii, jj = np.nonzero(np.nan_to_num(grid) > 0)
    assert len(ii) > 0, "building did not rasterize at all"
    n_i, n_j = grid.shape

    # Non-square AOI: north extent is half the east extent, so there must be
    # fewer north rows (axis 0) than east cols (axis 1). A TRANSPOSE — the
    # voxcity_vwind axis-swap — inverts this and is caught here.
    assert n_i < n_j, (
        f"grid shape {grid.shape}: axis 0 must be north (shorter, ~20 rows) and "
        "axis 1 east (longer, ~40 cols); a transpose inverts this ratio"
    )
    # Building sits deep north (high i) — catches a north/south flip (flipud).
    assert ii.mean() > 0.75 * n_i, (
        f"building mass at mean i={ii.mean():.1f} of {n_i} rows — axis 0 is not "
        "pointing north (row 0 must be the south edge)"
    )
    # Building sits moderately east (high j) — catches an east/west flip (fliplr).
    assert jj.mean() > 0.5 * n_j, (
        f"building mass at mean j={jj.mean():.1f} of {n_j} cols — axis 1 is not "
        "pointing east"
    )
