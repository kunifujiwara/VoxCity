"""Unit tests for app.backend.zoning helpers."""
import numpy as np
import pytest

from app.backend.zoning import (
    polygon_lonlat_to_cells,
    points_in_polygon_lonlat,
    stats_from_values,
)


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture
def grid_geom_axis_aligned():
    """A trivial 10x10 axis-aligned grid: cell centres at (i+0.5, j+0.5)."""
    return {
        "origin":    [0.0, 0.0],
        "u_vec":     [1.0, 0.0],
        "v_vec":     [0.0, 1.0],
        "adj_mesh":  [1.0, 1.0],
        "grid_size": [10, 10],
    }


# ---- polygon_lonlat_to_cells ----------------------------------------------

def test_rect_polygon_returns_inside_cells(grid_geom_axis_aligned):
    # Square covering centres (0.5..3.5, 0.5..3.5) -> 4x4 = 16 cells.
    ring = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    cells = polygon_lonlat_to_cells(ring, grid_geom_axis_aligned)
    assert len(cells) == 16
    assert (0, 0) in cells and (3, 3) in cells


def test_polygon_outside_grid_returns_empty(grid_geom_axis_aligned):
    ring = [[100.0, 100.0], [101.0, 100.0], [101.0, 101.0]]
    assert polygon_lonlat_to_cells(ring, grid_geom_axis_aligned) == []


def test_degenerate_polygon_returns_empty(grid_geom_axis_aligned):
    assert polygon_lonlat_to_cells([[0.0, 0.0], [1.0, 0.0]], grid_geom_axis_aligned) == []


# ---- stats_from_values -----------------------------------------------------

def test_stats_finite_values():
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 4
    assert s.mean == pytest.approx(2.5)
    assert s.min == 1.0 and s.max == 4.0
    assert s.std == pytest.approx(np.std(vals))


def test_stats_with_nan_values():
    vals = np.array([1.0, np.nan, 3.0, np.inf])
    s = stats_from_values("z1", cell_count=4, values=vals)
    assert s.cell_count == 4
    assert s.valid_count == 2
    assert s.mean == pytest.approx(2.0)


def test_stats_empty_zone():
    s = stats_from_values("z1", cell_count=0, values=np.array([], dtype=float))
    assert s.cell_count == 0 and s.valid_count == 0
    assert s.mean is None and s.std is None


# ---- points_in_polygon_lonlat ---------------------------------------------

def test_points_in_polygon():
    ring = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    pts = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0]])
    mask = points_in_polygon_lonlat(pts, ring)
    assert mask.tolist() == [True, False, False]


# ---- Endpoint tests --------------------------------------------------------

from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.state import app_state


@pytest.fixture
def client():
    return TestClient(app)


def test_zone_stats_no_model(client, monkeypatch):
    monkeypatch.setattr(app_state, "voxcity", None)
    r = client.post("/api/zones/stats", json={"zones": []})
    assert r.status_code == 400
    assert "No model" in r.json()["detail"]


def test_zone_stats_no_sim(client, monkeypatch):
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", None)
    r = client.post("/api/zones/stats", json={"zones": []})
    assert r.status_code == 400
    assert "simulation" in r.json()["detail"].lower()


def test_zone_stats_ground_basic(client, monkeypatch):
    """A 4x4 ramp grid + 2 zones (one inside, one outside)."""
    grid = np.arange(16, dtype=float).reshape(4, 4)
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "solar")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid", grid)
    monkeypatch.setattr(app_state, "last_sim_mesh", None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "W/m2")
    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "solar": SimulationResultCache(
            sim_type="solar", target="ground", grid=grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="W/m2",
        )
    })
    import app.backend.main as main_mod
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: {"origin": [0.0, 0.0], "u_vec": [1.0, 0.0], "v_vec": [0.0, 1.0],
                 "adj_mesh": [1.0, 1.0], "grid_size": [4, 4]},
        raising=True,
    )
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "z1", "name": "all",     "ring_lonlat": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        {"id": "z2", "name": "outside", "ring_lonlat": [[100, 100], [101, 100], [101, 101]]},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"] == "ground"
    assert body["unit_label"] == "W/m2"
    by_id = {s["zone_id"]: s for s in body["stats"]}
    assert by_id["z1"]["cell_count"] == 16
    assert by_id["z1"]["mean"] == pytest.approx(grid.mean())
    assert by_id["z2"]["cell_count"] == 0
    assert by_id["z2"]["mean"] is None


# ---- Building-surface adapter ---------------------------------------------

from types import SimpleNamespace

from app.backend.zoning import grid_xy_to_lonlat, mesh_face_data


def _make_fake_mesh(value_key: str):
    """Two right triangles sharing an edge, in z=0 plane.

    Tri 0: (0,0,0)-(2,0,0)-(0,2,0) -> centroid (2/3, 2/3), area 2.0
    Tri 1: (2,0,0)-(2,2,0)-(0,2,0) -> centroid (4/3, 4/3), area 2.0
    """
    V = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=float)
    F = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    metadata = {value_key: np.array([10.0, 30.0])}
    return SimpleNamespace(vertices=V, faces=F, metadata=metadata)


def test_mesh_face_data_solar():
    mesh = _make_fake_mesh("global")
    centroids, values, areas = mesh_face_data(mesh, "solar")
    assert centroids.shape == (2, 2)
    assert values.tolist() == [10.0, 30.0]
    assert areas == pytest.approx([2.0, 2.0])
    assert centroids[0] == pytest.approx([2 / 3, 2 / 3])


def test_grid_xy_to_lonlat_axis_aligned():
    # Real VoxCity north-up geometry: u_vec=[0,1] (north/lat), v_vec=[1,0] (east/lon).
    # grid_xy_to_lonlat maps scene_x=east → v_m and scene_y=north → u_m.
    # For origin=[10,20], scene (x=1,y=2) → lon=10+v_m=10+1=11, lat=20+u_m=20+2=22.
    geom = {"origin": [10.0, 20.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
            "adj_mesh": [1.0, 1.0], "grid_size": [4, 4]}
    out = grid_xy_to_lonlat(np.array([[1.0, 2.0], [3.0, 0.0]]), geom)
    assert out.tolist() == [[11.0, 22.0], [13.0, 20.0]]


def test_zone_stats_building_basic(client, monkeypatch):
    """Polygon containing only the second triangle -> area-weighted mean = 30."""
    mesh = _make_fake_mesh("view_factor_values")
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "view")
    monkeypatch.setattr(app_state, "last_sim_target", "building")
    monkeypatch.setattr(app_state, "last_sim_grid", None)
    monkeypatch.setattr(app_state, "last_sim_mesh", mesh)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Factor")
    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "view": SimulationResultCache(
            sim_type="view", target="building", grid=None, mesh=mesh,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Factor",
        )
    })
    import app.backend.main as main_mod
    # Real VoxCity north-up geometry: u_vec=[0,1] (north), v_vec=[1,0] (east).
    # With fixed grid_xy_to_lonlat: scene_x=east→v_m, scene_y=north→u_m.
    # Centroids: Tri0=(2/3,2/3), Tri1=(4/3,4/3) → lon=scene_x, lat=scene_y (symmetric).
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: {"origin": [0.0, 0.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
                 "adj_mesh": [1.0, 1.0], "grid_size": [4, 4]},
        raising=True,
    )
    # Polygon covers (1.1..2, 1.1..2): only tri 1 centroid (4/3, 4/3) is inside.
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "z1", "name": "tri1", "ring_lonlat": [[1.1, 1.1], [2.0, 1.1], [2.0, 2.0], [1.1, 2.0]]},
        {"id": "z2", "name": "all",  "ring_lonlat": [[-1, -1], [3, -1], [3, 3], [-1, 3]]},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    by_id = {s["zone_id"]: s for s in body["stats"]}
    assert by_id["z1"]["cell_count"] == 1
    assert by_id["z1"]["mean"] == pytest.approx(30.0)
    # Area-weighted mean of [10, 30] with equal areas = 20.
    assert by_id["z2"]["cell_count"] == 2
    assert by_id["z2"]["mean"] == pytest.approx(20.0)


# ---- Cross-stack rasterizer parity snapshot --------------------------------
# JS counterpart: app/frontend/src/lib/grid.ts `polygonToCells`.
# Uses ray-casting; matches matplotlib.path.Path.contains_points for polygons
# whose edges do not pass through any cell centre. The diamond below is chosen
# so that no cell centre lies exactly on a boundary edge (vertices at half
# integers, edges on the lines |x-5| + |y-5| = 4.5 — no integer pairs i+j
# satisfy i+j = 4.5).
#
# Manually verified cell count: 2+4+6+8+8+6+4+2 = 40.

def test_polygon_to_cells_snapshot_diamond(grid_geom_axis_aligned):
    ring = [[0.5, 5.0], [5.0, 0.5], [9.5, 5.0], [5.0, 9.5]]
    cells = polygon_lonlat_to_cells(ring, grid_geom_axis_aligned)
    expected = set()
    for i in range(10):
        for j in range(10):
            cx, cy = i + 0.5, j + 0.5
            if abs(cx - 5.0) + abs(cy - 5.0) <= 4.5:
                expected.add((i, j))
    assert len(expected) == 40
    assert set(cells) == expected


# ---- Cross-tab isolation ---------------------------------------------------

def _patch_grid_geom(monkeypatch):
    import app.backend.main as main_mod
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: {"origin": [0.0, 0.0], "u_vec": [1.0, 0.0], "v_vec": [0.0, 1.0],
                 "adj_mesh": [1.0, 1.0], "grid_size": [4, 4]},
        raising=True,
    )


def test_zone_stats_isolation_solar_after_view(client, monkeypatch):
    """When the global latest-sim is 'view', requesting sim_type='solar' must
    return the solar values, not the view values."""
    solar_grid = np.full((4, 4), 100.0)
    view_grid  = np.full((4, 4), 0.5)

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type",   "view")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid",   view_grid)
    monkeypatch.setattr(app_state, "last_sim_mesh",   None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Index")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "solar": SimulationResultCache(
            sim_type="solar", target="ground", grid=solar_grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="W/m2",
        ),
        "view": SimulationResultCache(
            sim_type="view", target="ground", grid=view_grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Index",
        ),
    })
    _patch_grid_geom(monkeypatch)

    r = client.post("/api/zones/stats", json={
        "sim_type": "solar",
        "zones": [{"id": "z1", "name": "all", "ring_lonlat": [[0, 0], [4, 0], [4, 4], [0, 4]]}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sim_type"] == "solar"
    assert body["unit_label"] == "W/m2"
    by_id = {s["zone_id"]: s for s in body["stats"]}
    # Must return the solar mean (100), NOT the view mean (0.5).
    assert by_id["z1"]["mean"] == pytest.approx(100.0)


def test_zone_stats_isolation_view_after_solar(client, monkeypatch):
    """Symmetric: requesting sim_type='view' when global latest is 'solar'
    must return the view values."""
    solar_grid = np.full((4, 4), 100.0)
    view_grid  = np.full((4, 4), 0.5)

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type",   "solar")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid",   solar_grid)
    monkeypatch.setattr(app_state, "last_sim_mesh",   None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "W/m2")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "solar": SimulationResultCache(
            sim_type="solar", target="ground", grid=solar_grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="W/m2",
        ),
        "view": SimulationResultCache(
            sim_type="view", target="ground", grid=view_grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Index",
        ),
    })
    _patch_grid_geom(monkeypatch)

    r = client.post("/api/zones/stats", json={
        "sim_type": "view",
        "zones": [{"id": "z1", "name": "all", "ring_lonlat": [[0, 0], [4, 0], [4, 4], [0, 4]]}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sim_type"] == "view"
    assert body["unit_label"] == "View Index"
    by_id = {s["zone_id"]: s for s in body["stats"]}
    assert by_id["z1"]["mean"] == pytest.approx(0.5)


def test_zone_stats_requested_but_missing(client, monkeypatch):
    """Requesting solar stats when only view has been cached should return 400."""
    view_grid = np.full((4, 4), 0.5)

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type",   "view")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid",   view_grid)
    monkeypatch.setattr(app_state, "last_sim_mesh",   None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Index")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "view": SimulationResultCache(
            sim_type="view", target="ground", grid=view_grid, mesh=None,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Index",
        ),
    })
    _patch_grid_geom(monkeypatch)

    r = client.post("/api/zones/stats", json={
        "sim_type": "solar",
        "zones": [{"id": "z1", "name": "all", "ring_lonlat": [[0, 0], [4, 0], [4, 4], [0, 4]]}],
    })
    assert r.status_code == 400
    assert "solar" in r.json()["detail"].lower()


# ---- Solar geometry endpoint regression ------------------------------------

def test_solar_ground_geometry_endpoint_fails_without_per_type_cache(client, monkeypatch):
    """Regression: legacy last_sim_* fields alone do NOT satisfy the geometry
    endpoint — it must return 400 when sim_results_by_type['solar'] is missing.
    This documents the pre-fix failure mode that occurred when run_solar()
    skipped store_sim_result() for the ground branch.
    """
    solar_grid = np.full((4, 4), 500.0)
    monkeypatch.setattr(app_state, "last_sim_type", "solar")
    monkeypatch.setattr(app_state, "last_sim_target", "ground")
    monkeypatch.setattr(app_state, "last_sim_grid", solar_grid)
    monkeypatch.setattr(app_state, "last_sim_mesh", None)
    monkeypatch.setattr(app_state, "last_colorbar_title", "Cum. Solar Irradiance (Wh/m\u00b2)")
    # Deliberately NOT populating sim_results_by_type["solar"].
    monkeypatch.setattr(app_state, "sim_results_by_type", {})

    r = client.post("/api/sim/solar/geometry", json={"colormap": "viridis"})
    assert r.status_code == 400
    assert "solar" in r.json()["detail"].lower()


def test_solar_ground_geometry_endpoint_returns_200_with_cache(client, monkeypatch):
    """Contract: when sim_results_by_type['solar'] is seeded (as store_sim_result
    does after the solar-ground fix), the geometry endpoint must return 200
    with correct target and sim_type fields.
    """
    solar_grid = np.full((4, 4), 500.0)
    voxcity_grid = np.zeros((4, 4, 2), dtype=np.int32)
    voxcity_grid[:, :, 0] = 1

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "solar": SimulationResultCache(
            sim_type="solar",
            target="ground",
            grid=solar_grid,
            mesh=None,
            voxcity_grid=voxcity_grid,
            view_point_height=1.5,
            colorbar_title="Cum. Solar Irradiance (Wh/m\u00b2)",
        ),
    })

    r = client.post("/api/sim/solar/geometry", json={"colormap": "viridis"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"] == "ground"
    assert body["sim_type"] == "solar"
    assert body["unit_label"] == "Cum. Solar Irradiance (Wh/m\u00b2)"


# ---- /api/buildings/at axis mapping regression ------------------------------

class _MockBuildings:
    def __init__(self, ids):
        self.ids = ids


class _MockVoxels:
    class _Meta:
        meshsize = 1.0
    meta = _Meta()


class _MockVoxcity:
    def __init__(self, bid_grid):
        self.buildings = _MockBuildings(bid_grid)
        self.voxels = _MockVoxels()


def test_buildings_at_returns_correct_building_for_scene_xy(client, monkeypatch):
    """Regression: /api/buildings/at used to swap scene X and Y axes.

    Grid layout (axis 0 = u = north, axis 1 = v = east), meshsize=1.0:
        row 0: [1, 2]   <- u=0: north row
        row 1: [3, 4]   <- u=1

    Scene convention: X = east = v (axis 1), Y = north = u (axis 0).
    So scene (x=0.5, y=0.5) -> u=0, v=0 -> building_id=1
       scene (x=1.5, y=0.5) -> u=0, v=1 -> building_id=2
       scene (x=0.5, y=1.5) -> u=1, v=0 -> building_id=3
       scene (x=1.5, y=1.5) -> u=1, v=1 -> building_id=4

    The old (buggy) code used ci=floor(x) for axis 0 and cj=floor(y) for
    axis 1, which would have returned building_id=1 at (x=1.5, y=0.5) instead
    of the correct building_id=2.
    """
    bid_grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
    vc_mock = _MockVoxcity(bid_grid)
    monkeypatch.setattr(app_state, "voxcity", vc_mock)

    cases = [
        (0.5, 0.5, 1),   # north-west cell
        (1.5, 0.5, 2),   # north-east cell  (was wrong before fix)
        (0.5, 1.5, 3),   # south-west cell  (was wrong before fix)
        (1.5, 1.5, 4),   # south-east cell
    ]
    for x, y, expected_bid in cases:
        r = client.get(f"/api/buildings/at?x={x}&y={y}")
        assert r.status_code == 200, r.text
        assert r.json()["building_id"] == expected_bid, (
            f"At scene (x={x}, y={y}) expected building_id={expected_bid} "
            f"but got {r.json()['building_id']}"
        )


# ---------------------------------------------------------------------------
# Coordinate convention: grid_xy_to_lonlat north-up non-symmetric regression
# ---------------------------------------------------------------------------

def test_grid_xy_to_lonlat_north_up_grid_non_symmetric():
    """For a north-up VoxCity grid, scene_x (east) must map to longitude and
    scene_y (north) to latitude — NOT the other way around.

    The non-symmetric input (scene_x=3, scene_y=1) distinguishes correct from
    swapped behaviour: correct gives lon=3, lat=1; swapped gives lon=1, lat=3.
    """
    # Real VoxCity north-up geometry: u_vec=[0,1] (north/lat), v_vec=[1,0] (east/lon)
    geom = {
        "origin": [0.0, 0.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
        "adj_mesh": [1.0, 1.0], "grid_size": [4, 4],
    }
    # Symmetric case: same result regardless of column order
    out_sym = grid_xy_to_lonlat(np.array([[2.0, 2.0]]), geom)
    assert out_sym[0] == pytest.approx([2.0, 2.0])

    # Non-symmetric: scene (east=3, north=1) → lon=3, lat=1
    out = grid_xy_to_lonlat(np.array([[3.0, 1.0]]), geom)
    assert out[0] == pytest.approx([3.0, 1.0]), (
        f"Expected lon=3.0 lat=1.0 but got {out[0].tolist()} — "
        "columns in grid_xy_to_lonlat may be swapped"
    )


# ---------------------------------------------------------------------------
# Building-ID helpers
# ---------------------------------------------------------------------------

def test_building_ids_in_zone_basic():
    """building_ids_in_zone returns nonzero IDs from rasterised zone cells."""
    from app.backend.zoning import building_ids_in_zone

    grid = np.array(
        [[42, 0, 0, 0],
         [0, 99, 0, 0],
         [0,  0, 0, 0],
         [0,  0, 0, 0]],
        dtype=np.int32,
    )
    # u_vec=[0,1] (north), v_vec=[1,0] (east): cell (i,j) centre at lon=j+0.5, lat=i+0.5
    geom = {
        "origin": [0.0, 0.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
        "adj_mesh": [1.0, 1.0], "grid_size": [4, 4],
    }
    # Zone covers cells (0,0) [lon=0.5,lat=0.5] and (0,1) [lon=1.5,lat=0.5].
    # Cell (1,1) [lon=1.5,lat=1.5] is outside (lat > 0.8).
    zone_ring = [[0.0, 0.0], [2.0, 0.0], [2.0, 0.8], [0.0, 0.8]]
    bids = building_ids_in_zone(zone_ring, grid, geom)
    assert 42 in bids
    assert 99 not in bids
    assert 0 not in bids


def test_building_ids_in_zone_empty_polygon():
    """Empty polygon (degenerate) returns empty set."""
    from app.backend.zoning import building_ids_in_zone

    grid = np.ones((4, 4), dtype=np.int32)
    geom = {
        "origin": [0.0, 0.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
        "adj_mesh": [1.0, 1.0], "grid_size": [4, 4],
    }
    bids = building_ids_in_zone([[0.0, 0.0], [1.0, 0.0]], grid, geom)
    assert bids == set()


# ---------------------------------------------------------------------------
# Building-surface zone stats: ownership gating regression
# ---------------------------------------------------------------------------

# Real VoxCity north-up geometry used by all ownership-gate tests.
# u_vec=[0,1] (north/lat), v_vec=[1,0] (east/lon).
# Cell (i,j) centre: lon=j+0.5, lat=i+0.5.
# grid_xy_to_lonlat: scene_x=east → lon, scene_y=north → lat.
_NORTH_UP_GEOM = {
    "origin": [0.0, 0.0], "u_vec": [0.0, 1.0], "v_vec": [1.0, 0.0],
    "adj_mesh": [1.0, 1.0], "grid_size": [4, 4],
}


def _patch_north_up_geom(monkeypatch):
    import app.backend.main as main_mod
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: dict(_NORTH_UP_GEOM),
        raising=True,
    )


def test_zone_stats_building_no_footprint_shows_no_values(client, monkeypatch):
    """Regression: zone with no building footprint cells must return zero stats.

    Building 42 lives at grid cell (3,3) — far corner.  One simulated face has
    building_id=42 and a centroid at scene (x=2.0, y≈1.83) which projects to
    lon≈2.0, lat≈1.83 — inside the test zone [[1.5,1.5],[2.5,2.5]].
    But cell (3,3) sits at lon=3.5, lat=3.5 which is outside the zone, so
    building_ids_in_zone returns an empty set and the ownership gate suppresses
    all face values.
    """
    from types import SimpleNamespace

    # Face centroid: scene (x=(1.5+2.5+2.0)/3=2.0, y=(1.5+1.5+2.5)/3≈1.833)
    V = np.array([[1.5, 1.5, 0.0], [2.5, 1.5, 0.0], [2.0, 2.5, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=int)
    mesh = SimpleNamespace(
        vertices=V,
        faces=F,
        metadata={
            "view_factor_values": np.array([25.0]),
            "building_id":        np.array([42]),
        },
    )

    building_id_grid = np.zeros((4, 4), dtype=np.int32)
    building_id_grid[3, 3] = 42  # building footprint only at far corner

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "view")
    monkeypatch.setattr(app_state, "last_sim_target", "building")
    monkeypatch.setattr(app_state, "last_sim_grid", None)
    monkeypatch.setattr(app_state, "last_sim_mesh", mesh)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Factor")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "view": SimulationResultCache(
            sim_type="view", target="building", grid=None, mesh=mesh,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Factor",
            building_id_grid=building_id_grid,
        ),
    })
    _patch_north_up_geom(monkeypatch)

    # Zone covers the face centroid (lon≈2, lat≈1.83) but NOT cell (3,3).
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "no_bld", "name": "no_building_zone",
         "ring_lonlat": [[1.4, 1.4], [2.6, 1.4], [2.6, 2.6], [1.4, 2.6]]},
    ]})
    assert r.status_code == 200, r.text
    s = {stat["zone_id"]: stat for stat in r.json()["stats"]}["no_bld"]
    assert s["cell_count"] == 0, (
        "Expected zero cell_count when building footprint is outside zone, "
        f"got cell_count={s['cell_count']} mean={s['mean']}"
    )
    assert s["valid_count"] == 0
    assert s["mean"] is None


def test_zone_stats_building_with_footprint_returns_values(client, monkeypatch):
    """Positive case: when the zone contains building footprint cells AND the
    face centroid, stats must be reported correctly.

    Building 99 at cell (1,1) (lon=1.5, lat=1.5).  One face with value=30
    and centroid at scene (x≈1.5, y≈1.33) → lon≈1.5, lat≈1.33 — inside
    zone [[1.1,1.1],[1.9,1.9]].  Cell (1,1) is also inside that zone.
    """
    from types import SimpleNamespace

    # Face centroid: scene (x=(1.0+2.0+1.5)/3=1.5, y=(1.0+1.0+2.0)/3≈1.333)
    V = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.5, 2.0, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=int)
    mesh = SimpleNamespace(
        vertices=V,
        faces=F,
        metadata={
            "view_factor_values": np.array([30.0]),
            "building_id":        np.array([99]),
        },
    )

    building_id_grid = np.zeros((4, 4), dtype=np.int32)
    building_id_grid[1, 1] = 99  # cell (1,1) → lon=1.5, lat=1.5 → inside zone

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "view")
    monkeypatch.setattr(app_state, "last_sim_target", "building")
    monkeypatch.setattr(app_state, "last_sim_grid", None)
    monkeypatch.setattr(app_state, "last_sim_mesh", mesh)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Factor")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "view": SimulationResultCache(
            sim_type="view", target="building", grid=None, mesh=mesh,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Factor",
            building_id_grid=building_id_grid,
        ),
    })
    _patch_north_up_geom(monkeypatch)

    # Zone covers both the face centroid and cell (1,1).
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "has_bld", "name": "has_building",
         "ring_lonlat": [[1.1, 1.1], [1.9, 1.1], [1.9, 1.9], [1.1, 1.9]]},
    ]})
    assert r.status_code == 200, r.text
    s = {stat["zone_id"]: stat for stat in r.json()["stats"]}["has_bld"]
    assert s["cell_count"] == 1
    assert s["valid_count"] == 1
    assert s["mean"] == pytest.approx(30.0)


def test_zone_stats_building_no_bid_metadata_falls_back_to_centroid(client, monkeypatch):
    """Fallback: when mesh has no building_id metadata and cache has no
    building_id_grid, centroid-only masking is used (backward compat)."""
    mesh = _make_fake_mesh("view_factor_values")
    # _make_fake_mesh does NOT add building_id to metadata.

    from app.backend.state import SimulationResultCache
    monkeypatch.setattr(app_state, "voxcity", object())
    monkeypatch.setattr(app_state, "last_sim_type", "view")
    monkeypatch.setattr(app_state, "last_sim_target", "building")
    monkeypatch.setattr(app_state, "last_sim_grid", None)
    monkeypatch.setattr(app_state, "last_sim_mesh", mesh)
    monkeypatch.setattr(app_state, "last_colorbar_title", "View Factor")
    monkeypatch.setattr(app_state, "sim_results_by_type", {
        "view": SimulationResultCache(
            sim_type="view", target="building", grid=None, mesh=mesh,
            voxcity_grid=None, view_point_height=1.5, colorbar_title="View Factor",
            # building_id_grid intentionally omitted (None)
        ),
    })
    _patch_north_up_geom(monkeypatch)

    # Both centroids inside large zone → both counted (centroid-only fallback).
    r = client.post("/api/zones/stats", json={"zones": [
        {"id": "all", "name": "all", "ring_lonlat": [[-1, -1], [3, -1], [3, 3], [-1, 3]]},
    ]})
    assert r.status_code == 200, r.text
    s = {stat["zone_id"]: stat for stat in r.json()["stats"]}["all"]
    assert s["cell_count"] == 2
    assert s["mean"] == pytest.approx(20.0)  # area-weighted mean of [10, 30]

