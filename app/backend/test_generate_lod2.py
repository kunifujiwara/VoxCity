"""Tests for the PLATEAU LOD2 generation branch (pipeline mocked)."""
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app


# Axis-aligned rectangle near Tokyo ([SW, NW, NE, SE], [lon, lat])
RECT = [
    [139.770, 35.646], [139.770, 35.650],
    [139.775, 35.650], [139.775, 35.646],
]
# Same rectangle rotated (lon of SW != lon of NW)
ROTATED_RECT = [
    [139.770, 35.646], [139.771, 35.650],
    [139.776, 35.649], [139.775, 35.645],
]


def _base_request(**overrides):
    body = {
        "rectangle_vertices": RECT,
        "meshsize": 5.0,
        "mode": "plateau",
        "plateau_lod": "lod2",
    }
    body.update(overrides)
    return body


class _FakeVoxels:
    classes = np.zeros((4, 4, 3), dtype=np.int16)


class _FakeVoxCity:
    voxels = _FakeVoxels()
    extras = {"land_cover_source": "OpenEarthMapJapan"}


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Stub voxcitygml import + downstream app plumbing."""
    calls = {}

    fake_pkg = types.ModuleType("voxcitygml")

    class FakeConfig:
        def __init__(self, **kwargs):
            calls['config'] = kwargs

    def fake_generate(cfg):
        calls['generate_called'] = True
        return _FakeVoxCity()

    fake_pkg.VoxelizerConfig = FakeConfig
    fake_pkg.generate_voxcity = fake_generate
    monkeypatch.setitem(sys.modules, "voxcitygml", fake_pkg)

    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    monkeypatch.setattr(main_mod, "_reset_taichi_and_caches", lambda: None)
    monkeypatch.setattr(main_mod, "_refine_canopy_with_ndsm",
                        lambda *a, **k: False)
    monkeypatch.setattr(main_mod, "_preview_figure_json",
                        lambda *a, **k: "{}")
    monkeypatch.setattr(main_mod, "_preview_disabled_for_shape",
                        lambda shape: False)

    # ``generate_model`` reads ``app_state.voxcity`` right after storing the
    # result, so the stub has to publish the object like the real
    # ``store_generation_result`` does. Registering ``voxcity`` with monkeypatch
    # first makes pytest restore the previous value at teardown.
    monkeypatch.setattr(main_mod.app_state, "voxcity", None)

    def fake_store(**k):
        calls['stored'] = k
        main_mod.app_state.voxcity = k["voxcity_obj"]

    monkeypatch.setattr(main_mod.app_state, "store_generation_result",
                        fake_store)
    return calls


def test_lod2_calls_voxcitygml(stubbed):
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['generate_called']
    cfg = stubbed['config']
    assert cfg['building_lod'] == 2
    assert cfg['rectangle_vertices'] == [tuple(v) for v in RECT]
    assert cfg['save_output'] is False
    assert 'stored' in stubbed


def test_lod2_requires_citygml_path(stubbed, monkeypatch):
    monkeypatch.setattr(main_mod, "CITYGML_PATH", None)
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 400
    assert "CITYGML_PATH" in resp.json()["detail"]


def test_lod2_rejects_rotated_rectangle(stubbed):
    client = TestClient(app)
    resp = client.post("/api/generate",
                       json=_base_request(rectangle_vertices=ROTATED_RECT))
    assert resp.status_code == 400
    assert "rotat" in resp.json()["detail"].lower()


def test_lod2_maps_no_buildings_to_400(stubbed, monkeypatch):
    def raise_no_buildings(cfg):
        raise ValueError("No CityGML buildings found in the selected area.")
    sys.modules["voxcitygml"].generate_voxcity = raise_no_buildings
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 400
    assert "buildings" in resp.json()["detail"]


def test_lod1_default_unaffected(stubbed, monkeypatch):
    """plateau_lod defaults to lod1 → existing cache path, voxcitygml untouched."""
    monkeypatch.setattr(main_mod, "_load_citygml_cache", lambda verts: None)

    def fake_citygml(*args, **kwargs):
        return _FakeVoxCity()
    monkeypatch.setattr(main_mod, "get_voxcity_CityGML", fake_citygml)

    body = _base_request()
    del body["plateau_lod"]
    client = TestClient(app)
    resp = client.post("/api/generate", json=body)
    assert resp.status_code == 200, resp.text
    assert 'generate_called' not in stubbed
