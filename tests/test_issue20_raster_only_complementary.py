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


def test_temporal_source_with_provided_gdf_uses_the_gdf(monkeypatch):
    """When a caller supplies building_gdf AND labels the source raster-only,
    the temporal-raster branch is skipped (grids are not computed there), so
    the provided gdf must flow through the vector-merge path instead of
    leaving the return grids unassigned (would UnboundLocalError)."""
    import geopandas as gpd

    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    shape = (4, 4)
    merged_height = np.full(shape, 3.0)

    monkeypatch.setattr(gee_mod, "initialize_earth_engine", lambda *a, **k: None)

    # The temporal raster builder must NOT run when a gdf is provided.
    def fail_temporal(*a, **k):
        raise AssertionError("temporal raster builder must not run when building_gdf is provided")

    monkeypatch.setattr(
        grids_mod,
        "create_building_height_grid_from_open_building_temporal_polygon",
        fail_temporal,
    )

    # The provided gdf must be merged via the vector path.
    def fake_from_gdf(gdf_, *a, **k):
        return merged_height, np.zeros(shape), np.zeros(shape, dtype=int), gdf_

    monkeypatch.setattr(grids_mod, "create_building_height_grid_from_gdf_polygon", fake_from_gdf)

    provided = gpd.GeoDataFrame()
    height, _min, _ids, _bld = grids_mod.get_building_height_grid(
        rect,
        5.0,
        source="Open Building 2.5D Temporal",
        output_dir="output",
        building_gdf=provided,
        building_complementary_source="None",
        gridvis=False,
        quiet=True,
    )
    assert np.array_equal(height, merged_height)
