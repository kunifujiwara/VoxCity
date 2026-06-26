"""The rooftop flag must be accepted by the request models and forwarded to the
GPU ground functions for solar / view / landmark."""
import numpy as np
import pytest

from app.backend.models import SolarRequest, ViewRequest, LandmarkRequest


def test_models_accept_include_building_roofs_default_false():
    assert SolarRequest().include_building_roofs is False
    assert ViewRequest().include_building_roofs is False
    assert LandmarkRequest().include_building_roofs is False


def test_models_accept_true():
    assert SolarRequest(include_building_roofs=True).include_building_roofs is True
    assert ViewRequest(include_building_roofs=True).include_building_roofs is True
    assert LandmarkRequest(include_building_roofs=True).include_building_roofs is True


def test_view_endpoint_forwards_flag(monkeypatch):
    from fastapi.testclient import TestClient
    import app.backend.main as main
    from app.backend.state import app_state
    from tests.simulator._roof_helpers import make_voxcity_with_building

    captured = {}

    def fake_get_view_index(voxcity, mode=None, **kwargs):
        captured['include_building_roofs'] = kwargs.get('include_building_roofs')
        nx, ny, _ = voxcity.voxels.classes.shape
        return np.zeros((nx, ny), dtype=float)

    vc = make_voxcity_with_building()
    monkeypatch.setattr(app_state, "voxcity", vc)
    monkeypatch.setattr(app_state, "rectangle_vertices", vc.extras["rectangle_vertices"])
    monkeypatch.setattr(main, "get_view_index", fake_get_view_index)
    monkeypatch.setattr(main, "_make_plotly_json", lambda *a, **k: "{}")

    client = TestClient(main.app)
    r = client.post("/api/view", json={
        "analysis_target": "ground", "view_type": "sky",
        "include_building_roofs": True,
    })
    assert r.status_code == 200, r.text
    assert captured['include_building_roofs'] is True
