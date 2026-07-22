"""Regression test for issue #20: raster-only base source + complementary source.

When building_source is a raster-only source ("Open Building 2.5D Temporal"),
it computes the height grids directly and has no vector `gdf`. If a
complementary source is also set, the old code dereferenced an unassigned
`gdf` and raised UnboundLocalError. The grids computed by the raster-only
base must be returned unchanged, with the complementary source ignored.
"""

import numpy as np

from voxcity.generator import grids as grids_mod
from voxcity.downloader import gee as gee_mod


def test_raster_only_base_with_complementary_does_not_raise(monkeypatch):
    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    meshsize = 5.0
    shape = (4, 4)

    sentinel_height = np.full(shape, 7.0)
    sentinel_min = np.zeros(shape)
    sentinel_id = np.zeros(shape, dtype=int)

    # `initialize_earth_engine` is imported *inside* get_building_height_grid
    # (a local `from ..downloader.gee import ...`), so it must be patched on
    # its defining module rather than on grids_mod. "Open Building 2.5D
    # Temporal" is in ee_required_sources, so without this the test would
    # fail on a real Earth Engine auth error before ever reaching the
    # gdf-related bug this test targets.
    monkeypatch.setattr(gee_mod, "initialize_earth_engine", lambda *a, **k: None)

    def fake_open_building_temporal(meshsize_, rect_, output_dir_):
        import geopandas as gpd
        return sentinel_height, sentinel_min, sentinel_id, gpd.GeoDataFrame()

    monkeypatch.setattr(
        grids_mod,
        "create_building_height_grid_from_open_building_temporal_polygon",
        fake_open_building_temporal,
    )

    def fail_if_called(*a, **k):
        raise AssertionError("complementary gdf-merge path must not run for a raster-only base")

    monkeypatch.setattr(grids_mod, "create_building_height_grid_from_gdf_polygon", fail_if_called)

    # `get_mbfp_gdf` (module-level in grids_mod) is what the buggy code path
    # would call to fetch the complementary source's footprints before
    # dereferencing the unassigned `gdf`. Stub it so the test is
    # network-free and deterministic; once the fix short-circuits on a
    # raster-only base source, this must never be invoked either.
    def fail_if_mbfp_called(*a, **k):
        raise AssertionError("complementary source fetch must not run for a raster-only base")

    monkeypatch.setattr(grids_mod, "get_mbfp_gdf", fail_if_mbfp_called)

    height, min_h, ids, _ = grids_mod.get_building_height_grid(
        rect,
        meshsize,
        source="Open Building 2.5D Temporal",
        output_dir="output",
        building_complementary_source="Microsoft Building Footprints",
        gridvis=False,
        quiet=True,
    )
    assert np.array_equal(height, sentinel_height)
