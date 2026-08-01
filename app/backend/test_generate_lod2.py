"""Tests for the PLATEAU LOD2 generation branch (pipeline mocked)."""
import builtins
import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app

# Captured at import time, *before* any fixture stubs ``sys.modules``.
# ``pytest.importorskip("voxcitygml")`` inside a test using the ``stubbed``
# fixture would resolve to the fake module and make the contract test below
# vacuous, since the fixture patches ``sys.modules['voxcitygml']``.
try:
    from voxcitygml import VoxelizerConfig as _RealVoxelizerConfig
except ImportError:  # package not installed in this env
    _RealVoxelizerConfig = None

try:
    from voxcitygml.grid_utils import compute_grid_params as _real_compute_grid_params
except ImportError:  # package not installed in this env
    _real_compute_grid_params = None


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
    # extras must be per-instance: the backend tags it with building_lod, and a
    # shared class attribute would leak that mutation into other tests.
    def __init__(self):
        self.voxels = _FakeVoxels()
        self.extras = {"land_cover_source": "OpenEarthMapJapan"}


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

    def fake_refine(*a, **k):
        # Recorded, not just stubbed: for LOD2 this must never be reached,
        # because it ends in regenerate_voxels() and would overwrite the mesh
        # voxelization with extruded footprints.
        calls['ndsm_called'] = True
        return False

    monkeypatch.setattr(main_mod, "_refine_canopy_with_ndsm", fake_refine)
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
        # Design decision 3: CityGML bridges are out of scope.
        'include_bridges': False,
    }
    assert 'stored' in stubbed


@pytest.mark.skipif(_RealVoxelizerConfig is None,
                    reason="voxcitygml is not installed in this environment")
def test_lod2_kwargs_construct_a_real_voxelizer_config(stubbed):
    """Bind the cross-repo contract with the real ``VoxelizerConfig``.

    Every other test here builds the config through ``FakeConfig(**kwargs)``,
    which accepts anything. A typo'd kwarg, or a field renamed in VoxCityGML,
    would keep those tests green while production raised ``TypeError`` -> 500.
    ``VoxelizerConfig`` is a dataclass, so unknown kwargs are a hard error:
    feeding it the kwargs the backend actually passed is what catches drift.
    """
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text

    cfg = _RealVoxelizerConfig(**stubbed['config'])

    # Guard against a future dataclass that silently swallows **kwargs.
    assert cfg.building_lod == 2
    assert cfg.meshsize == 5.0
    assert cfg.save_output is False


def test_lod2_skips_ndsm_refinement(stubbed):
    """Regression: _refine_canopy_with_ndsm ends in regenerate_voxels(), which
    rebuilds the voxel grid from the 2.5-D component grids. Running it on an
    LOD2 model silently replaces the true roof/wall geometry with extruded
    footprints — an invisible downgrade to LOD1."""
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert 'ndsm_called' not in stubbed


def test_lod1_still_runs_ndsm_refinement(stubbed, monkeypatch):
    """Pins the counterpart: the LOD2 skip must come from the new gate alone,
    not from some other condition failing for both LODs."""
    monkeypatch.setattr(main_mod, "_load_citygml_cache", lambda verts: None)
    monkeypatch.setattr(main_mod, "get_voxcity_CityGML",
                        lambda *a, **k: _FakeVoxCity())
    body = _base_request()
    del body["plateau_lod"]
    resp = TestClient(app).post("/api/generate", json=body)
    assert resp.status_code == 200, resp.text
    assert stubbed.get('ndsm_called') is True


def test_lod2_tags_extras_with_building_lod(stubbed):
    """The LOD-ness must be recorded so voxel-rebuilding paths can refuse."""
    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['stored']['voxcity_obj'].extras["building_lod"] == 2


def test_lod2_maps_missing_dataset_layout_to_400(stubbed):
    """resolve_citygml_paths raises FileNotFoundError (an OSError, not a
    ValueError) when the directory has no udx/bldg; it must not become a 500."""
    def raise_not_found(cfg):
        raise FileNotFoundError("No udx/bldg directory found")
    sys.modules["voxcitygml"].generate_voxcity = raise_not_found
    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 400
    assert "udx/bldg" in resp.json()["detail"]


def test_lod2_requires_citygml_path(stubbed, monkeypatch):
    monkeypatch.setattr(main_mod, "CITYGML_PATH", None)
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 400
    assert "CITYGML_PATH" in resp.json()["detail"]


def test_lod2_accepts_rotated_rectangle(stubbed):
    """Rotated rectangles are supported since VoxCityGML gained an affine
    grid frame; vertices must be forwarded verbatim."""
    client = TestClient(app)
    resp = client.post("/api/generate",
                       json=_base_request(rectangle_vertices=ROTATED_RECT))
    assert resp.status_code == 200, resp.text
    assert stubbed['generate_called']
    assert stubbed['config']['rectangle_vertices'] == [tuple(v) for v in ROTATED_RECT]


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


def test_lod2_excludes_bridges(stubbed):
    """Bridges are out of scope (design decision 3); voxcitygml clears
    collection.bridges before rasterisation when include_bridges is False."""
    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['config']['include_bridges'] is False


# ---------------------------------------------------------------------------
# LOD2 capability reporting
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_capability_cache():
    """The import probe is cached for the process; reset around each test."""
    main_mod._voxcitygml_import_state.cache_clear()
    yield
    main_mod._voxcitygml_import_state.cache_clear()


def test_health_reports_lod2_available(clear_capability_cache, stubbed):
    """voxcitygml stubbed into sys.modules + CITYGML_PATH set by the fixture."""
    body = TestClient(app).get("/api/health").json()
    assert body["status"] == "ok"
    assert body["capabilities"]["plateau_lod2"] == {"available": True, "reason": ""}


def test_health_reports_lod2_unavailable_without_package(
        clear_capability_cache, monkeypatch, tmp_path):
    """The happy path is the local default, so force the missing-package branch
    by making the import fail rather than only observing what's installed."""
    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "voxcitygml":
            raise ImportError("No module named 'voxcitygml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "voxcitygml", raising=False)

    cap = TestClient(app).get("/api/health").json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "not installed" in cap["reason"]


def test_health_reports_lod2_unavailable_when_package_too_old(
        clear_capability_cache, monkeypatch, tmp_path):
    """An installed-but-stale voxcitygml lacks generate_voxcity()."""
    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    stale = types.ModuleType("voxcitygml")  # no generate_voxcity attribute
    monkeypatch.setitem(sys.modules, "voxcitygml", stale)

    cap = TestClient(app).get("/api/health").json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "0.2.0" in cap["reason"]


@pytest.mark.parametrize("path_value", [None, "", "/definitely/not/a/real/dir"])
def test_health_reports_lod2_unavailable_without_dataset(
        clear_capability_cache, stubbed, monkeypatch, path_value):
    monkeypatch.setattr(main_mod, "CITYGML_PATH", path_value)
    cap = TestClient(app).get("/api/health").json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "CITYGML_PATH" in cap["reason"]


def test_capability_probe_failure_does_not_break_health(
        clear_capability_cache, monkeypatch):
    """A probe that blows up must degrade to unavailable, not 500 /api/health."""
    def boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(main_mod, "_voxcitygml_import_state", boom)
    resp = TestClient(app).get("/api/health")
    assert resp.status_code == 200
    cap = resp.json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "probe exploded" in cap["reason"]


# ---------------------------------------------------------------------------
# LOD2 model protection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extras,expected", [
    ({"building_lod": 2}, True),
    ({"building_lod": 1}, False),
    ({}, False),                                    # LOD1 / normal mode
    ({"building_lod": None}, False),
    ({"building_lod": "2"}, True),                  # tolerate a string
    ({"building_lod": "nonsense"}, False),
])
def test_is_lod2_model(extras, expected):
    vc = _FakeVoxCity()
    vc.extras = extras
    assert main_mod._is_lod2_model(vc) is expected


def test_is_lod2_model_handles_missing_extras():
    assert main_mod._is_lod2_model(object()) is False


@pytest.fixture
def _restore_voxcity(monkeypatch):
    monkeypatch.setattr(main_mod.app_state, "voxcity", None)


def test_apply_edits_rejects_lod2_model(_restore_voxcity):
    """An LOD2 VoxCity cannot be rebuilt from its own 2.5-D component grids, so
    apply_edits' trailing regenerate_voxels() would flatten it. Fail loudly."""
    vc = _FakeVoxCity()
    vc.extras["building_lod"] = 2
    main_mod.app_state.voxcity = vc

    resp = TestClient(app).post("/api/model/apply_edits", json={
        "edits": [{"kind": "paint_lc", "cells": [[0, 0]], "class_index": 5}],
    })
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "LOD2" in detail and "LOD1" in detail


def test_apply_edits_guard_runs_before_any_edit_is_applied(_restore_voxcity,
                                                           monkeypatch):
    """The guard must precede mutation, or a rejected batch still leaves the
    model half-edited."""
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return {"n_changed": 1}

    monkeypatch.setattr(main_mod, "_apply_paint_lc", spy)
    vc = _FakeVoxCity()
    vc.extras["building_lod"] = 2
    main_mod.app_state.voxcity = vc

    resp = TestClient(app).post("/api/model/apply_edits", json={
        "edits": [{"kind": "paint_lc", "cells": [[0, 0]], "class_index": 5}],
    })
    assert resp.status_code == 400
    assert called["n"] == 0, "no edit may be applied before the guard rejects"


# ---------------------------------------------------------------------------
# Rectangle validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_real_compute_grid_params is None,
                    reason="voxcitygml is not installed in this environment")
@pytest.mark.parametrize("verts", [
    pytest.param([[139.77, 35.65]] * 4, id="zero-area"),
    pytest.param([[139.770, 35.646], [139.772, 35.646],
                  [139.774, 35.646], [139.776, 35.646]], id="collinear"),
])
def test_lod2_degenerate_rectangle_yields_400(stubbed, verts):
    """The deleted axis-alignment guard also rejected zero-area input as a side
    effect, because it bailed out when either extent was 0. That protection now
    lives in voxcitygml's ``_check_non_degenerate``, which every grid-params
    call runs. Route the *real* function's exception through the endpoint to
    prove the composition still produces a 400 rather than a 500.

    ``_real_compute_grid_params`` is captured at import time, before the
    ``stubbed`` fixture patches ``sys.modules['voxcitygml']`` — otherwise this
    would resolve to the fake and assert nothing.
    """
    def real_guard(_cfg):
        # FakeConfig only records its kwargs, so read them back from the
        # fixture — which is also the stricter check: these are the exact
        # vertices the endpoint forwarded.
        passed = stubbed['config']
        _real_compute_grid_params(passed['rectangle_vertices'],
                                  passed['meshsize'])
        raise AssertionError(
            "voxcitygml accepted a degenerate rectangle; the app no longer "
            "screens these itself, so this would reach the voxelizer")

    sys.modules["voxcitygml"].generate_voxcity = real_guard
    resp = TestClient(app).post("/api/generate",
                                json=_base_request(rectangle_vertices=verts))
    assert resp.status_code == 400, resp.text
    # Non-vacuity: the 400 must be voxcitygml's degeneracy ValueError, not some
    # unrelated early rejection that would mask a future regression.
    assert "degenerate" in resp.json()["detail"].lower(), resp.text


@pytest.mark.parametrize("verts", [
    RECT[:3],                                       # too few vertices
    RECT + [[139.775, 35.646]],                     # too many (5-point ring)
    [[139.0], [1.0], [2.0], [3.0]],                 # inner pair too short
    [[139.0, 35.0, 1.0]] * 4,                       # inner pair too long
])
def test_generate_rejects_malformed_vertices_with_422(verts):
    """GenerateRequest constrains both the outer list and each [lon, lat] pair,
    so malformed input is a validation error rather than a 500 from unpacking
    or indexing downstream in _vertices_to_tuples."""
    resp = TestClient(app).post("/api/generate",
                                json=_base_request(rectangle_vertices=verts))
    assert resp.status_code == 422
