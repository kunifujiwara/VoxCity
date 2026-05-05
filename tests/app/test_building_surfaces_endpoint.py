"""Tests for building-surface zone API models and endpoint validation."""
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.state import SimulationResultCache, app_state


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _fake_surface_mesh(building_ids=(1, 1), value_key="view_factor_values", values=(10.0, 30.0)):
    """Return a SimpleNamespace mesh with two triangular faces and metadata."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    metadata = {
        value_key: np.array(values, dtype=float),
        "building_id": np.array(building_ids),
        # One roof normal (Z+) and one east-wall normal (X+)
        "provided_face_normals": np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float
        ),
    }
    return types.SimpleNamespace(vertices=vertices, faces=faces, metadata=metadata)


def _patch_grid_geom(monkeypatch):
    """Monkeypatch _grid_geom_for_zoning to return an axis-aligned 4×4 grid."""
    import app.backend.main as main_mod

    grid_geom = types.SimpleNamespace(
        origin=[0, 0],
        u_vec=[0, 1],
        v_vec=[1, 0],
        adj_mesh=[1, 1],
        grid_size=[4, 4],
    )
    monkeypatch.setattr(main_mod, "_grid_geom_for_zoning", lambda: grid_geom, raising=True)


def _patch_ground_cache(monkeypatch):
    """Set up a valid solar/ground simulation cache."""
    _patch_grid_geom(monkeypatch)
    grid = np.arange(16, dtype=float).reshape(4, 4)
    monkeypatch.setattr(app_state, "voxcity", types.SimpleNamespace())
    monkeypatch.setattr(app_state, "last_sim_type", "solar")
    monkeypatch.setattr(
        app_state,
        "sim_results_by_type",
        {
            "solar": SimulationResultCache(
                sim_type="solar",
                target="ground",
                grid=grid,
                colorbar_title="W/m2",
            )
        },
    )


def _patch_building_cache_with_surface_meta(monkeypatch):
    """Set up a valid view/building simulation cache with surface face metadata."""
    _patch_grid_geom(monkeypatch)
    mesh = _fake_surface_mesh()
    mesh.metadata["surface_face_meta"] = [
        {"face_key": "b1:roof", "building_id": 1, "surface_kind": "roof"},
        {"face_key": "b1:east", "building_id": 1, "surface_kind": "east_wall"},
    ]
    mesh.metadata["surface_face_meta_version"] = 1
    monkeypatch.setattr(app_state, "voxcity", types.SimpleNamespace())
    monkeypatch.setattr(app_state, "last_sim_type", "view")
    monkeypatch.setattr(
        app_state,
        "sim_results_by_type",
        {
            "view": SimulationResultCache(
                sim_type="view",
                target="building",
                mesh=mesh,
                colorbar_title="View Factor",
            )
        },
    )


def _patch_model_state(monkeypatch):
    """Set up a minimal voxcity model state."""
    voxcity = types.SimpleNamespace(
        voxels=types.SimpleNamespace(
            classes=np.zeros((4, 4, 3), dtype=int),
            meta=types.SimpleNamespace(meshsize=1.0),
        ),
        buildings=types.SimpleNamespace(
            ids=np.zeros((4, 4), dtype=int),
        ),
    )
    monkeypatch.setattr(app_state, "voxcity", voxcity)


# ---------------------------------------------------------------------------
# Validation tests (Step 1 — these should fail before model changes)
# ---------------------------------------------------------------------------

def test_zone_stats_rejects_mixed_group_types(client, monkeypatch):
    _patch_ground_cache(monkeypatch)
    payload = {
        "zones": [
            {
                "id": "h1",
                "group_id": "g1",
                "name": "Zone",
                "type": "horizontal",
                "ring_lonlat": [[0, 0], [1, 0], [1, 1]],
            },
            {
                "id": "s1",
                "group_id": "g1",
                "name": "Zone",
                "type": "building_surface",
                "selectors": [],
            },
        ]
    }
    response = client.post("/api/zones/stats", json=payload)
    assert response.status_code == 400
    assert "mixed" in response.json()["detail"].lower()


def test_zone_stats_rejects_duplicate_surface_records_in_group(client, monkeypatch):
    _patch_building_cache_with_surface_meta(monkeypatch)
    payload = {
        "zones": [
            {
                "id": "s1",
                "group_id": "g1",
                "name": "Zone",
                "type": "building_surface",
                "selectors": [],
            },
            {
                "id": "s2",
                "group_id": "g1",
                "name": "Zone",
                "type": "building_surface",
                "selectors": [],
            },
        ]
    }
    response = client.post("/api/zones/stats", json=payload)
    assert response.status_code == 400
    assert "building_surface" in response.json()["detail"]


def test_zone_stats_rejects_malformed_surface_selector(client, monkeypatch):
    _patch_building_cache_with_surface_meta(monkeypatch)
    payload = {
        "zones": [
            {
                "id": "s1",
                "name": "Zone",
                "type": "building_surface",
                "selectors": [{"building_id": 1, "mode": "wall_orientation"}],
            }
        ]
    }
    response = client.post("/api/zones/stats", json=payload)
    assert response.status_code == 400
    assert "orientation" in response.json()["detail"].lower()
