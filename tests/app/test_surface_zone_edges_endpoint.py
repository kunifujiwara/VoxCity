import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.state import app_state


@pytest.fixture
def client():
    return TestClient(app)


def _patch_edge_model(monkeypatch):
    voxcity = types.SimpleNamespace(
        voxels=types.SimpleNamespace(
            classes=np.zeros((3, 3, 2), dtype=int),
            meta=types.SimpleNamespace(meshsize=1.0),
        ),
        buildings=types.SimpleNamespace(ids=np.zeros((3, 3), dtype=int)),
    )
    voxcity.voxels.classes[1, 1, 0] = -3
    voxcity.buildings.ids[1, 1] = 7
    # app_state.meshsize resolves to voxcity.voxels.meta.meshsize
    monkeypatch.setattr(app_state, "voxcity", voxcity)


def test_surface_zone_edges_endpoint_filters_horizontal_zones(client, monkeypatch):
    _patch_edge_model(monkeypatch)
    response = client.post("/api/buildings/surface-zone-edges", json={
        "zones": [{
            "id": "h1",
            "name": "Horizontal",
            "type": "horizontal",
            "ring_lonlat": [[0, 0], [1, 0], [1, 1]],
            "selectors": [],
        }],
    })
    assert response.status_code == 200, response.text
    assert response.json() == {"zones": []}


def test_surface_zone_edges_endpoint_returns_segments(client, monkeypatch):
    _patch_edge_model(monkeypatch)
    response = client.post("/api/buildings/surface-zone-edges", json={
        "zones": [{
            "id": "roof",
            "name": "Roof",
            "type": "building_surface",
            "ring_lonlat": None,
            "selectors": [{"building_id": 7, "mode": "roof", "orientation": None, "face_keys": None}],
        }],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["zones"][0]["id"] == "roof"
    assert len(body["zones"][0]["segments"]) == 4


def test_surface_zone_edges_validates_surface_selectors(client, monkeypatch):
    _patch_edge_model(monkeypatch)
    response = client.post("/api/buildings/surface-zone-edges", json={
        "zones": [{
            "id": "bad",
            "name": "Bad",
            "type": "building_surface",
            "selectors": [{"building_id": 7, "mode": "wall_orientation", "orientation": None, "face_keys": None}],
        }],
    })
    assert response.status_code == 400
    assert "orientation" in response.json()["detail"].lower()


def test_surface_zone_edges_returns_empty_without_model_or_ids(client, monkeypatch):
    monkeypatch.setattr(app_state, "voxcity", None)
    response = client.post("/api/buildings/surface-zone-edges", json={"zones": []})
    assert response.status_code == 200
    assert response.json() == {"zones": []}
