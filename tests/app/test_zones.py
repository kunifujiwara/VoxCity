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
    geom = {"origin": [10.0, 20.0], "u_vec": [1.0, 0.0], "v_vec": [0.0, 1.0],
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
    import app.backend.main as main_mod
    # Identity mapping so grid xy == lon/lat for easy reasoning.
    monkeypatch.setattr(
        main_mod, "_grid_geom_for_zoning",
        lambda: {"origin": [0.0, 0.0], "u_vec": [1.0, 0.0], "v_vec": [0.0, 1.0],
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

