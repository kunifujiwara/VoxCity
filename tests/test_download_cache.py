"""Tests for the on-disk download cache."""

import pickle

import pytest

from voxcity.utils import cache as cache_mod


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXCITY_CACHE_DIR", str(tmp_path))
    yield tmp_path


def test_second_call_uses_cache(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None, floor_height=3.0):
        calls["n"] += 1
        return {"data": calls["n"]}

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    a = fake_download(rect, floor_height=3.0)
    b = fake_download(rect, floor_height=3.0)
    assert calls["n"] == 1
    assert a == b == {"data": 1}


def test_different_args_different_cache(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None, floor_height=3.0):
        calls["n"] += 1
        return calls["n"]

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    fake_download(rect, floor_height=3.0)
    fake_download(rect, floor_height=4.0)
    assert calls["n"] == 2


def test_force_refresh_bypasses(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None):
        calls["n"] += 1
        return calls["n"]

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert fake_download(rect) == 1
    assert fake_download(rect, force_refresh=True) == 2
    assert calls["n"] == 2


def test_use_download_cache_false_disables(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None):
        calls["n"] += 1
        return calls["n"]

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    fake_download(rect, use_download_cache=False)
    fake_download(rect, use_download_cache=False)
    assert calls["n"] == 2


def test_corrupt_cache_falls_back(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None):
        calls["n"] += 1
        return calls["n"]

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    fake_download(rect)
    for p in cache_mod.get_cache_dir().glob("*.pkl"):
        p.write_bytes(b"not a pickle")
    assert fake_download(rect) == 2


def test_output_dir_excluded_from_cache_key(isolated_cache_dir):
    """A different output_dir/download_dir must not poison the cache key --
    two calls that differ only in those directory params should hit the same
    cache entry (matters for mbfp.get_mbfp_gdf, which takes output_dir first)."""
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(output_dir, rectangle_vertices=None):
        calls["n"] += 1
        return calls["n"]

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert fake_download("/tmp/a", rect) == 1
    assert fake_download("/tmp/b", rect) == 1
    assert calls["n"] == 1


def _one_building_payload():
    """Minimal valid Overpass response: one closed building way.

    The production query uses ``out geom;``, so ways carry an inline
    ``geometry`` list of {lat, lon} points (see osm.py's element processing).
    Non-empty on purpose: empty results are deliberately never cached, so the
    caching assertion below needs a payload that yields at least one feature.
    """
    corners = [(0.0002, 0.0002), (0.0006, 0.0002), (0.0006, 0.0006), (0.0002, 0.0006)]
    ring = corners + [corners[0]]
    way = {
        "type": "way",
        "id": 100,
        "nodes": [1, 2, 3, 4, 1],
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in ring],
        "tags": {"building": "yes", "height": "12"},
    }
    return {"elements": [way]}


def test_osm_download_cached_end_to_end(isolated_cache_dir, monkeypatch):
    from voxcity.downloader import osm as osm_mod

    calls = {"n": 0}

    def fake_fetch(query, **kw):
        calls["n"] += 1
        return _one_building_payload()

    monkeypatch.setattr(osm_mod, "_fetch_overpass_with_retry", fake_fetch)
    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    first = osm_mod.load_gdf_from_openstreetmap(rect)
    second = osm_mod.load_gdf_from_openstreetmap(rect)
    assert len(first) > 0, "payload must produce a feature or the cache path is untested"
    assert calls["n"] == 1
    assert len(second) == len(first)


def test_empty_download_result_is_not_cached(isolated_cache_dir):
    calls = {"n": 0}

    @cache_mod.cached_download
    def fake_download(rectangle_vertices=None):
        calls["n"] += 1
        return []  # empty: must not be cached

    rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    fake_download(rect)
    fake_download(rect)
    assert calls["n"] == 2


def test_use_download_cache_and_force_refresh_reach_downloader(isolated_cache_dir, monkeypatch):
    """get_building_height_grid must forward the user-facing
    use_download_cache/force_refresh kwargs down to the (module-level
    imported) downloader function it calls -- see grids.py's _cache_kwargs
    plumbing added alongside the @cached_download decorator."""
    from voxcity.generator import grids

    captured = {}

    def fake_load_gdf_from_openstreetmap(rectangle_vertices, floor_height=3.0, **kwargs):
        captured["use_download_cache"] = kwargs.get("use_download_cache")
        captured["force_refresh"] = kwargs.get("force_refresh")
        import geopandas as gpd
        from shapely.geometry import Polygon
        # A minimal valid (non-empty) building footprint so the rest of the
        # building-height-grid pipeline (which reads gdf['height'] etc.) runs
        # without crashing on a schema-less empty GeoDataFrame.
        return gpd.GeoDataFrame(
            {"height": [10.0], "id": [1]},
            geometry=[Polygon([(0.0, 0.0), (0.0005, 0.0), (0.0005, 0.0005), (0.0, 0.0005)])],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(grids, "load_gdf_from_openstreetmap", fake_load_gdf_from_openstreetmap)

    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    grids.get_building_height_grid(
        rect,
        meshsize=10,
        source="OpenStreetMap",
        output_dir=str(isolated_cache_dir),
        use_download_cache=False,
        force_refresh=True,
        gridvis=False,
    )

    assert captured["use_download_cache"] is False
    assert captured["force_refresh"] is True
