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


def test_lod2_calls_voxcitygml(stubbed, tmp_path):
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['generate_called']

    # Assert the FULL kwarg set, not a subset: VoxelizerConfig is a dataclass,
    # so a renamed/typo'd kwarg is a TypeError in production. A partial
    # assertion would let that drift through green tests.
    cfg = stubbed['config']
    assert cfg == {
        'citygml_path': str(tmp_path),
        'rectangle_vertices': [tuple(v) for v in RECT],
        'meshsize': 5.0,
        'building_lod': 2,
        'land_cover_source': "OpenEarthMapJapan",
        'canopy_height_source': "Static",
        'static_tree_height': 10.0,
        'output_dir': main_mod.os.path.join(main_mod.BASE_OUTPUT_DIR, "test"),
        'save_output': False,
        'gridvis': False,
    }
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


# ---------------------------------------------------------------------------
# _is_axis_aligned_rectangle unit tests
# ---------------------------------------------------------------------------

def _dimension_rectangle(center_lon, center_lat, width_m, height_m,
                         rotation_deg=0.0):
    """Build a rectangle via the real /api/rectangle-from-dimensions endpoint.

    Deliberately not hand-typed: the geodesic construction is precisely what an
    earlier absolute 1e-9 tolerance mis-classified as "rotated".
    """
    resp = TestClient(app).post("/api/rectangle-from-dimensions", json={
        "center_lon": center_lon, "center_lat": center_lat,
        "width_m": width_m, "height_m": height_m,
        "rotation_deg": rotation_deg,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()["vertices"]


def test_axis_aligned_exact_rectangle():
    """Two-click map draw reuses the same sw/ne floats -> exactly aligned."""
    assert main_mod._is_axis_aligned_rectangle(RECT) is True


@pytest.mark.parametrize("center_lat,size_m", [
    (35.65, 100),     # Tokyo, small
    (35.65, 1250),    # Tokyo, the UI's default extent
    (35.65, 20000),   # Tokyo, very large
    (45.52, 1000),    # northern Japan (worst latitude for PLATEAU coverage)
    (45.52, 20000),   # northern Japan, very large -> worst case overall
])
def test_geodesic_rectangle_at_rotation_zero_is_accepted(center_lat, size_m):
    """Regression: geodesic rectangles are NOT lon/lat-aligned even at
    rotation 0, because a given easting maps to a larger delta-lon at the north
    edge than the south edge. An absolute tolerance rejected all of these."""
    verts = _dimension_rectangle(139.77, center_lat, size_m, size_m, 0.0)
    assert verts[0][0] != verts[1][0], "expected a non-trivial geodesic skew"
    assert main_mod._is_axis_aligned_rectangle(verts) is True


@pytest.mark.parametrize("rotation_deg", [0.5, 1.0, 30.0, -30.0, 45.0])
def test_rotated_rectangles_are_rejected(rotation_deg):
    verts = _dimension_rectangle(139.77, 35.65, 1000, 1000, rotation_deg)
    assert main_mod._is_axis_aligned_rectangle(verts) is False


def test_hand_typed_rotated_fixture_is_rejected():
    assert main_mod._is_axis_aligned_rectangle(ROTATED_RECT) is False


@pytest.mark.parametrize("verts", [
    [],
    [[139.0, 35.0]],
    RECT[:3],
    RECT + [[139.775, 35.646]],
])
def test_wrong_vertex_count_is_rejected(verts):
    """Guards the 4-tuple unpack; must not raise."""
    assert main_mod._is_axis_aligned_rectangle(verts) is False


@pytest.mark.parametrize("verts", [RECT[:3], RECT + [[139.775, 35.646]]])
def test_generate_rejects_wrong_vertex_count_with_422(verts):
    """GenerateRequest constrains the list length, so malformed input is a
    validation error rather than a 500 from unpacking downstream."""
    resp = TestClient(app).post("/api/generate",
                                json=_base_request(rectangle_vertices=verts))
    assert resp.status_code == 422
