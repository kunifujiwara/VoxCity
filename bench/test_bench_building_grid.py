"""Benchmark the gdf -> building-height-grid path.

Profiling (Batch 4, Task 5) showed this is the hottest CPU stage of
generation after downloads: ``_process_with_geometry_intersection`` in
``geoprocessor/raster/buildings_precise.py`` spends most of its time building
one shapely Polygon per grid cell and intersecting per (building, cell) pair.
This benchmark keeps that cost visible; the vectorization itself is a
follow-up candidate (see the improvement roadmap).
"""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from voxcity.geoprocessor.raster import create_building_height_grid_from_gdf_polygon

RECT = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.01), (0.0, 0.01)]


@pytest.fixture(scope="module")
def synthetic_buildings():
    rng = np.random.default_rng(42)
    n = 1000
    polys = []
    for _ in range(n):
        x = rng.uniform(0.0, 0.009)
        y = rng.uniform(0.0, 0.009)
        d = 0.00015  # ~15 m squares
        polys.append(Polygon([(x, y), (x + d, y), (x + d, y + d), (x, y + d)]))
    return gpd.GeoDataFrame(
        {
            "height": rng.uniform(5, 60, n),
            "min_height": np.zeros(n),
            "id": np.arange(n),
            "is_inner": np.zeros(n, dtype=bool),
        },
        geometry=polys,
        crs="EPSG:4326",
    )


def test_bench_building_height_grid_from_gdf(benchmark, synthetic_buildings):
    # Warm shapely/pyproj caches outside the timed region.
    create_building_height_grid_from_gdf_polygon(synthetic_buildings, 5.0, RECT)
    result = benchmark(
        create_building_height_grid_from_gdf_polygon, synthetic_buildings, 5.0, RECT
    )
    height_grid = result[0]
    assert (height_grid > 0).any()
