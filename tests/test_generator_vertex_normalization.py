"""Wiring tests: generator entry points canonicalize rectangle_vertices."""
import pytest

grids = pytest.importorskip("voxcity.generator.grids")

SW = (139.75, 35.65)
NW = (139.75, 35.66)
NE = (139.76, 35.66)
SE = (139.76, 35.65)
CANONICAL = [SW, NW, NE, SE]
SHUFFLED = [NE, SE, SW, NW]


def test_normalization_is_wired_into_entry_points():
    """Source-level guard: each public entry normalizes rectangle_vertices."""
    import inspect
    from voxcity.generator import api

    for fn in (api.get_voxcity, api.get_voxcity_CityGML):
        src = inspect.getsource(fn)
        assert "normalize_rectangle_vertices" in src, fn.__name__
    for name in (
        "get_land_cover_grid",
        "get_building_height_grid",
        "get_canopy_height_grid",
        "get_dem_grid",
    ):
        src = inspect.getsource(getattr(grids, name))
        assert "normalize_rectangle_vertices" in src, name


def test_get_dem_grid_normalizes_vertices_before_ee_fallback(tmp_path, monkeypatch):
    """Behavioral check for get_dem_grid: when Earth Engine is unavailable,
    the function falls back to a flat DEM by calling
    ``compute_grid_shape(rectangle_vertices, meshsize)`` (see
    src/voxcity/generator/grids.py, get_dem_grid's else-branch except
    handler). That call happens on the *local* ``rectangle_vertices``
    variable, so it only observes the normalized value if normalization
    actually ran first. This avoids any network/Earth Engine calls by
    forcing ``initialize_earth_engine`` to raise, using the same patch
    target already used in tests/test_generator_gsi_dem.py.
    """
    captured = {}

    def fake_compute_grid_shape(rectangle_vertices, meshsize):
        captured["vertices"] = list(rectangle_vertices)
        return (2, 2)

    def raise_ee_unavailable():
        raise RuntimeError("EE unavailable (simulated, no network)")

    monkeypatch.setattr(grids, "initialize_earth_engine", raise_ee_unavailable)
    monkeypatch.setattr(
        "voxcity.geoprocessor.raster.core.compute_grid_shape",
        fake_compute_grid_shape,
    )

    grid = grids.get_dem_grid(
        SHUFFLED,
        meshsize=10,
        source="USGS 3DEP 1m",
        output_dir=str(tmp_path),
    )

    assert captured["vertices"] == CANONICAL
    assert grid.shape == (2, 2)
