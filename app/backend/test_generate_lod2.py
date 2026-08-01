"""Tests for the PLATEAU LOD2 generation branch (pipeline mocked)."""
import builtins
import dataclasses
import inspect
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

try:
    from voxcitygml import reapply_canopy as _real_reapply_canopy
except ImportError:  # package predates canopy re-apply, or is not installed
    _real_reapply_canopy = None


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


def _fake_grid_utils(*, rotation_capable: bool):
    """A stub of ``voxcitygml.grid_utils`` with/without the affine grid frame.

    The backend probes ``GridParams`` for the affine fields rather than
    comparing versions: rotation shipped in 0.3.0, but voxcitygml is installed
    editable from a git checkout whose recorded metadata routinely lags the
    code on disk, so ``__version__`` cannot be trusted to separate the two
    builds (nor can ``generate_voxcity()``'s presence — 0.2.0 has it too).
    """
    mod = types.ModuleType("voxcitygml.grid_utils")

    @dataclasses.dataclass
    class _BboxGridParams:
        """Pre-rotation shape: plain min/max bounding-box arithmetic."""
        n_rows: int = 0
        n_cols: int = 0
        min_lon: float = 0.0
        max_lon: float = 0.0
        min_lat: float = 0.0
        max_lat: float = 0.0
        pixel_width: float = 0.0
        pixel_height: float = 0.0

    @dataclasses.dataclass
    class _AffineGridParams(_BboxGridParams):
        """Post-rotation shape: NW origin + column/row basis vectors."""
        origin_lon: float = 0.0
        origin_lat: float = 0.0
        e_col_lon: float = 0.0
        e_col_lat: float = 0.0
        e_row_lon: float = 0.0
        e_row_lat: float = 0.0

    mod.GridParams = _AffineGridParams if rotation_capable else _BboxGridParams
    return mod


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
    # The stub must mirror the *rotation-capable* package, or the backend's
    # capability probe would report every LOD2 test's environment unusable.
    fake_pkg.grid_utils = _fake_grid_utils(rotation_capable=True)
    # Likewise current enough to overlay a canopy onto an LOD2 grid.
    fake_pkg.reapply_canopy = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "voxcitygml", fake_pkg)
    # The probe is process-cached; swapping the package underneath it must not
    # leave a stale verdict behind for (or from) a neighbouring test.
    main_mod._voxcitygml_import_state.cache_clear()
    main_mod._voxcitygml_reapply_state.cache_clear()

    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    monkeypatch.setattr(main_mod, "_reset_taichi_and_caches", lambda: None)

    def fake_refine(*a, **k):
        # Recorded, not just stubbed: the endpoint must reach refinement for
        # both LODs. Which *grid-writing* entrypoint refinement then takes
        # (rebuild vs overlay) is pinned by the unit tests further down.
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
    yield calls
    main_mod._voxcitygml_import_state.cache_clear()
    main_mod._voxcitygml_reapply_state.cache_clear()


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
        # User-controllable, but off unless the request opts in.
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


def test_lod2_runs_ndsm_refinement(stubbed):
    """"Use nDSM for Canopy" must have an effect in LOD2.

    Refinement used to be gated off for LOD2 entirely, because it ended in
    regenerate_voxels() — a rebuild from the 2.5-D component grids that
    replaces true roof/wall geometry with extruded footprints. The checkbox
    rendered, stayed checked, and silently did nothing. The gate is gone: LOD2
    safety now lives *inside* _refine_canopy_with_ndsm, which overlays the
    canopy with reapply_canopy instead of rebuilding.
    """
    client = TestClient(app)
    resp = client.post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed.get('ndsm_called') is True


def test_lod1_still_runs_ndsm_refinement(stubbed, monkeypatch):
    """Pins the counterpart: removing the LOD2 gate must not have been a
    no-op because some *other* condition fails for both LODs."""
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


def test_lod2_generate_refuses_a_package_without_rotation_support(stubbed):
    """/api/health gating the UI is not enough: the endpoint is reachable
    directly, and a pre-rotation voxcitygml returns *wrong* geometry rather
    than raising. Refuse before calling it — for the axis-aligned request too,
    since the whole mode is declared unavailable rather than half-supported."""
    sys.modules["voxcitygml"].grid_utils = _fake_grid_utils(
        rotation_capable=False)
    main_mod._voxcitygml_import_state.cache_clear()
    main_mod._voxcitygml_reapply_state.cache_clear()

    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 500, resp.text
    assert "rotat" in resp.json()["detail"].lower(), resp.text
    assert 'generate_called' not in stubbed


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


def test_lod2_excludes_bridges_by_default(stubbed):
    """Omitting the field must preserve the pre-toggle behaviour exactly.

    voxcitygml's own ``VoxelizerConfig.include_bridges`` defaults to True, so
    the False here has to come from *our* request model — forwarding a missing
    field as "unset" would silently switch bridges on for every existing user.
    """
    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['config']['include_bridges'] is False


def test_lod2_includes_bridges_when_requested(stubbed):
    """Opting in reaches VoxelizerConfig, where voxcitygml keeps
    collection.bridges before building both the 2-D grids and the voxel grid."""
    resp = TestClient(app).post("/api/generate",
                                json=_base_request(include_bridges=True))
    assert resp.status_code == 200, resp.text
    assert stubbed['config']['include_bridges'] is True


# ---------------------------------------------------------------------------
# LOD2 capability reporting
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_capability_cache():
    """The import probe is cached for the process; reset around each test."""
    main_mod._voxcitygml_import_state.cache_clear()
    main_mod._voxcitygml_reapply_state.cache_clear()
    yield
    main_mod._voxcitygml_import_state.cache_clear()
    main_mod._voxcitygml_reapply_state.cache_clear()


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


def test_health_reports_lod2_unavailable_without_rotation_support(
        clear_capability_cache, monkeypatch, tmp_path):
    """An installed voxcitygml that has generate_voxcity() but no rotation.

    Rotation shipped in 0.3.0, but an editable checkout's recorded version
    lags its code, so neither ``__version__`` nor ``generate_voxcity``'s
    presence reliably tells this build from a current one. Such a build does
    not *fail* on a rotated rectangle — it quietly grids the axis-aligned
    bounding box — so the probe has to key on the affine ``GridParams`` fields
    instead.
    """
    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    old = types.ModuleType("voxcitygml")
    old.generate_voxcity = lambda cfg: None
    old.VoxelizerConfig = object
    old.grid_utils = _fake_grid_utils(rotation_capable=False)
    monkeypatch.setitem(sys.modules, "voxcitygml", old)

    cap = TestClient(app).get("/api/health").json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "predates rotated-rectangle" in cap["reason"], cap["reason"]


def test_health_reports_lod2_unavailable_when_grid_utils_is_unreachable(
        clear_capability_cache, monkeypatch, tmp_path):
    """An unreachable ``grid_utils`` must fail closed *and* say so distinctly.

    Two things are pinned here. First: the probe resolves ``grid_utils``
    through the package object, so a stub that does not expose it is reported
    unavailable even though the real ``voxcitygml.grid_utils`` is sitting in
    ``sys.modules`` (imported at the top of this file) with the affine fields
    present — a bare ``importlib.import_module`` would return that real module
    and pass this test vacuously.

    Second: the reason must not be the "predates rotated-rectangle support"
    text. That message names a cause — an outdated install — that is simply
    wrong here, and it would send users chasing a version bump when the actual
    fault is a broken/partial install or a restructured package.
    """
    monkeypatch.setattr(main_mod, "CITYGML_PATH", str(tmp_path))
    headless = types.ModuleType("voxcitygml")  # no grid_utils attribute
    headless.generate_voxcity = lambda cfg: None
    headless.VoxelizerConfig = object
    monkeypatch.setitem(sys.modules, "voxcitygml", headless)

    cap = TestClient(app).get("/api/health").json()["capabilities"]["plateau_lod2"]
    assert cap["available"] is False
    assert "grid_utils" in cap["reason"], cap["reason"]
    assert "predates" not in cap["reason"], cap["reason"]


@pytest.mark.skipif(_RealVoxelizerConfig is None,
                    reason="voxcitygml is not installed in this environment")
def test_rotation_probe_accepts_the_real_package(clear_capability_cache):
    """Non-vacuity for the test above: the probe must not reject reality.

    ``_fake_grid_utils`` only asserts that the probe distinguishes two stubs of
    the app's own making. This pins the other direction against the installed
    package — which does carry the affine frame — so a probe keyed on a field
    name that VoxCityGML later renames fails here instead of silently
    disabling LOD2 for everyone.
    """
    ok, reason = main_mod._voxcitygml_import_state()
    assert ok, reason


@pytest.fixture
def ndsm_cog(monkeypatch, tmp_path):
    """Pretend the ~5 GB nDSM COG is present on this server."""
    cog = tmp_path / "ndsm_cog.tif"
    cog.write_bytes(b"")
    monkeypatch.setattr(main_mod, "NDSM_COG_PATH", str(cog))
    return cog


def test_health_reports_ndsm_canopy_lod2_available(clear_capability_cache,
                                                   stubbed, ndsm_cog):
    """The fixture's stub carries reapply_canopy, like a current install."""
    cap = TestClient(app).get("/api/health").json()["capabilities"]
    assert cap["ndsm_canopy_lod2"] == {"available": True, "reason": ""}
    assert cap["ndsm_canopy"] == {"available": True, "reason": ""}


def test_health_reports_ndsm_canopy_unavailable_without_the_cog(
        clear_capability_cache, stubbed, monkeypatch, tmp_path):
    """The likeliest reason refinement does nothing on a fresh deployment: the
    ~5 GB nDSM raster is not shipped. Without this the checkbox stays ticked
    and silently no-ops — the original symptom, a different cause."""
    monkeypatch.setattr(main_mod, "NDSM_COG_PATH",
                        str(tmp_path / "definitely-absent.tif"))

    cap = TestClient(app).get("/api/health").json()["capabilities"]
    assert cap["ndsm_canopy"]["available"] is False
    assert "nDSM raster" in cap["ndsm_canopy"]["reason"]
    # LOD-independent: this one is about the data file, not the package, so the
    # LOD2-only flag must stay clean and let the UI report the right cause.
    assert cap["ndsm_canopy_lod2"]["available"] is True


def test_ndsm_cog_capability_is_not_cached(clear_capability_cache, stubbed,
                                           monkeypatch, tmp_path):
    """An operator can drop the COG in while the server runs; a cached
    "missing" verdict would outlive the file's arrival."""
    cog = tmp_path / "ndsm_cog.tif"
    monkeypatch.setattr(main_mod, "NDSM_COG_PATH", str(cog))
    client = TestClient(app)

    before = client.get("/api/health").json()["capabilities"]["ndsm_canopy"]
    assert before["available"] is False

    cog.write_bytes(b"")
    after = client.get("/api/health").json()["capabilities"]["ndsm_canopy"]
    assert after["available"] is True, "the COG check must not be memoized"


def test_health_reports_ndsm_canopy_unavailable_without_reapply_canopy(
        clear_capability_cache, stubbed, ndsm_cog):
    """Closes the loop on the original bug: without this flag the user ticks
    "Use nDSM for Canopy", waits out a full LOD2 generation, and gets a model
    where it did nothing — visible only in a server-side print."""
    del sys.modules["voxcitygml"].reapply_canopy
    main_mod._voxcitygml_reapply_state.cache_clear()

    cap = TestClient(app).get("/api/health").json()["capabilities"]
    assert cap["ndsm_canopy_lod2"]["available"] is False
    assert "reapply_canopy" in cap["ndsm_canopy_lod2"]["reason"]


def test_missing_reapply_canopy_does_not_disable_lod2_generation(
        clear_capability_cache, stubbed):
    """The nDSM probe must stay *non-gating*.

    Folding it into _voxcitygml_import_state would be the tempting
    simplification, and it would be wrong: that verdict refuses /api/generate
    with a 500 and flips plateau_lod2 to unavailable. LOD2 generation itself
    works perfectly well on a package without reapply_canopy — only the
    optional canopy refinement is lost.
    """
    del sys.modules["voxcitygml"].reapply_canopy
    main_mod._voxcitygml_reapply_state.cache_clear()

    cap = TestClient(app).get("/api/health").json()["capabilities"]
    assert cap["plateau_lod2"] == {"available": True, "reason": ""}

    resp = TestClient(app).post("/api/generate", json=_base_request())
    assert resp.status_code == 200, resp.text
    assert stubbed['generate_called']


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
# nDSM canopy refinement: which grid-writing entrypoint gets used
# ---------------------------------------------------------------------------

# Index of "Tree" in _LC_CLASSES below. Deliberately not 0: _refine_canopy_with_
# ndsm resolves the id with ``name_to_id.get("Tree") or ...``, so a Tree at
# index 0 would fall through the ``or`` chain to the 4 fallback.
_TREE_ID = 1
_LC_CLASSES = {"bareland": "Bareland", "tree": "Tree"}
# Non-square on purpose: a rows/cols mix-up cannot hide behind it.
_REFINE_SHAPE = (6, 4)


def _asymmetric_land_cover():
    """Tree cells in the northern rows only — so a north<->south mirror shows.

    An all-tree grid (the obvious fixture) is flip-invariant, which is exactly
    why the mirrored-canopy defect survived every earlier test. Row 0 is the
    *southern* edge in voxcity's land-cover frame, so "northern rows" are the
    high row indices here.
    """
    lc = np.zeros(_REFINE_SHAPE, dtype=np.int16)
    lc[_REFINE_SHAPE[0] // 2:, :] = _TREE_ID
    return lc


class _RefineModel:
    """Stand-in VoxCity carrying exactly the grids _refine_canopy_with_ndsm reads."""

    def __init__(self, building_lod: int, land_cover=None):
        self.land_cover = types.SimpleNamespace(
            classes=(_asymmetric_land_cover() if land_cover is None
                     else land_cover))
        self.buildings = types.SimpleNamespace(
            heights=np.zeros(_REFINE_SHAPE, dtype=float))
        self.tree_canopy = types.SimpleNamespace(
            top=np.zeros(_REFINE_SHAPE, dtype=float),
            bottom=np.zeros(_REFINE_SHAPE, dtype=float))
        self.voxels = types.SimpleNamespace(
            classes=np.zeros((*_REFINE_SHAPE, 3), dtype=np.int16))
        self.extras = {"building_lod": building_lod}


@pytest.fixture
def refine_env(monkeypatch):
    """Record which grid-writing entrypoint _refine_canopy_with_ndsm reaches.

    Both are stubbed, so a test can assert on the one that *was not* called —
    which is the whole point: calling the wrong one is the bug. Arrays handed
    to the stub are copied on capture, never held by reference: the real
    reapply_canopy writes through the model's grids, and regenerate_voxels
    rebinds ``city.voxels`` outright, so a retained reference would report
    pre-call state and pass on the very regression it exists to catch.
    """
    calls = {}

    monkeypatch.setattr(main_mod, "get_land_cover_classes",
                        lambda source: _LC_CLASSES)
    monkeypatch.setattr(main_mod, "_load_ndsm_grid",
                        lambda verts, meshsize: np.full(_REFINE_SHAPE, 8.0))

    def fake_regenerate(obj, **kwargs):
        calls['regenerate'] = kwargs

    monkeypatch.setattr(main_mod, "regenerate_voxels", fake_regenerate)

    fake_pkg = types.ModuleType("voxcitygml")

    def fake_reapply(*args, **kwargs):
        # Captured verbatim, positional-vs-keyword included: the signature test
        # replays this against the *real* reapply_canopy, so normalizing the
        # call here would hide exactly the renamed parameter it exists to
        # catch. ndarrays are snapshotted, never held by reference.
        def snap(v):
            return v.copy() if isinstance(v, np.ndarray) else v

        calls['reapply'] = {
            'args': tuple(snap(a) for a in args),
            'kwargs': {k: snap(v) for k, v in kwargs.items()},
        }

    fake_pkg.reapply_canopy = fake_reapply
    monkeypatch.setitem(sys.modules, "voxcitygml", fake_pkg)
    return calls


def _refine(vc):
    return main_mod._refine_canopy_with_ndsm(
        vc, RECT, meshsize=5.0, land_cover_source="OpenEarthMapJapan")


def _overlay_args(calls):
    """The captured reapply_canopy call, resolved to parameter names.

    Accepts either calling convention so a switch between positional and
    keyword in the backend is not a spurious failure here — the *signature*
    test is what pins the call shape against the real function.
    """
    names = ("city", "canopy_top", "canopy_bottom", "trunk_height_ratio")
    resolved = dict(zip(names, calls['reapply']['args']))
    resolved.update(calls['reapply']['kwargs'])
    return resolved


# The trunk ratio _refine_canopy_with_ndsm derives crown bases with.
_TRUNK_RATIO = 11.76 / 19.98
# _load_ndsm_grid is stubbed to this, and the sanitizer leaves it alone: it is
# within [min_tree_height_m, max_tree_height_m] and the model has no buildings,
# so no leakage/outlier rule fires.
_NDSM_HEIGHT = 8.0


def _land_frame_canopy():
    """The canopy as built upstream: _NDSM_HEIGHT at tree cells, south-up.

    Derived from _asymmetric_land_cover() rather than restating its layout, so
    that reverting *only* the fixture to something flip-invariant trips the
    non-vacuity guard in the mirror test — with its explanation — instead of
    failing an unrelated equality first as a bare `assert False`.
    """
    return np.where(_asymmetric_land_cover() == _TREE_ID, _NDSM_HEIGHT, 0.0)


def _expected_overlay_top():
    """The canopy reapply_canopy should receive: tree cells, unreoriented.

    Every grid in the model shares one frame, so the canopy goes over exactly as
    it was built from the land cover.
    """
    return _land_frame_canopy()


def test_refine_lod1_rebuilds_the_voxel_grid(refine_env):
    """LOD1 keeps the rebuild: its voxels *are* extruded footprints, so
    regenerating from the revised component grids is the correct update."""
    vc = _RefineModel(building_lod=1)
    assert _refine(vc) is True
    assert 'regenerate' in refine_env
    assert refine_env['regenerate']['inplace'] is True
    assert refine_env['regenerate']['land_cover_source'] == "OpenEarthMapJapan"
    assert 'reapply' not in refine_env


def test_refine_lod2_overlays_instead_of_rebuilding(refine_env):
    """LOD2 must take reapply_canopy, which clears and rewrites only canopy
    voxels. regenerate_voxels here would discard the mesh-voxelized roof/wall
    geometry the mode exists to produce — measured on real data as roof-slope
    0.1512 -> 0.0965 and 26,088 -> 21,623 building voxels."""
    vc = _RefineModel(building_lod=2)
    assert _refine(vc) is True
    assert 'reapply' in refine_env
    assert 'regenerate' not in refine_env, (
        "regenerate_voxels rebuilds from the 2.5-D grids and flattens LOD2")

    overlay = _overlay_args(refine_env)
    assert overlay['city'] is vc
    # Pin the values, not a re-read of the model's own grids: on this path
    # reapply_canopy writes those grids itself, so comparing against them would
    # only assert that the (stubbed) overlay did nothing.
    assert np.array_equal(overlay['canopy_top'], _expected_overlay_top())
    assert overlay.get('canopy_bottom') is not None, (
        "voxcitygml's own default trunk ratio currently equals ours, so "
        "passing the ratio instead would yield the same array today — passing "
        "the derived bottom keeps that a coincidence we don't depend on")
    assert np.allclose(overlay['canopy_bottom'],
                       _expected_overlay_top() * _TRUNK_RATIO)


def test_refine_lod2_hands_the_canopy_over_unreoriented(refine_env):
    """The canopy must reach reapply_canopy in the frame it was built in.

    It is built from land_cover.classes, which is south-up (row 0 = southern
    edge). voxcitygml returns a model that is south-up throughout — DEM, canopy
    and voxel grid included — so reapply_canopy writes into that same frame and
    a flip here would land the crowns mirrored north<->south.

    The land cover here is asymmetric for a reason — an all-tree grid is
    flip-invariant, which is precisely how the mirrored-canopy defect survived
    earlier tests.
    """
    vc = _RefineModel(building_lod=2)
    assert _refine(vc) is True

    got = _overlay_args(refine_env)['canopy_top']
    land_frame = _land_frame_canopy()

    assert np.array_equal(got, land_frame)
    assert not np.array_equal(got, np.flipud(land_frame)), (
        "the canopy was flipped out of the land-cover frame on the way to "
        "reapply_canopy — it would land mirrored north<->south in the voxel grid")


def test_refine_lod2_hands_over_the_canopy_at_component_resolution(refine_env):
    """The canopy must keep land_cover's shape, not the voxel grid's.

    reapply_canopy resamples a mismatched canopy itself — as it already does
    for the DEM — but it *stores* what the caller passed. Pre-resampling to the
    voxel shape would therefore leave tree_canopy.top at voxel resolution while
    land_cover, dem and buildings stayed at component resolution: the 2.5-D
    grids desync from each other, and a later update_voxcity raises
    "Grid shape mismatch". Resampling is the package's job.
    """
    vc = _RefineModel(building_lod=2)
    voxel_shape = (_REFINE_SHAPE[0] * 2, _REFINE_SHAPE[1] * 2)
    vc.voxels = types.SimpleNamespace(
        classes=np.zeros((*voxel_shape, 3), dtype=np.int16))

    assert _refine(vc) is True
    overlay = _overlay_args(refine_env)
    assert overlay['canopy_top'].shape == _REFINE_SHAPE, (
        "the canopy was pre-resampled to the voxel grid; what reapply_canopy "
        "stores would then desync tree_canopy from the other 2.5-D grids")
    assert overlay['canopy_bottom'].shape == _REFINE_SHAPE
    assert vc.land_cover.classes.shape == _REFINE_SHAPE


def test_refine_lod2_skips_when_reapply_canopy_is_unavailable(refine_env):
    """An older voxcitygml has no reapply_canopy. Refinement is optional and
    non-fatal, so it degrades to a no-op — but it must NOT fall back to
    regenerate_voxels, which would silently destroy the LOD2 geometry that the
    missing entrypoint exists to protect."""
    del sys.modules["voxcitygml"].reapply_canopy
    vc = _RefineModel(building_lod=2)

    assert _refine(vc) is False, "the caller logs the reason when this is False"
    assert 'regenerate' not in refine_env, (
        "no silent fallback: rebuilding is exactly what LOD2 must never do")
    assert 'reapply' not in refine_env
    # Bailing out after the component grids were rewritten would leave them
    # describing crowns the voxel grid does not contain, so the check has to
    # run before any mutation — same rule as the apply_edits guard.
    assert not vc.tree_canopy.top.any(), "model must be left untouched"
    assert not vc.tree_canopy.bottom.any(), "model must be left untouched"


@pytest.mark.parametrize("building_lod", [2, 1])
def test_refine_passes_buildings_to_the_sanitizer_unreoriented(
        refine_env, monkeypatch, building_lod):
    """_sanitize_ndsm_canopy compares tree cells against nearby building
    heights, so both grids must share a frame — and they already do.

    buildings.heights and land_cover.classes (hence canopy and tree_mask) are
    both south-up in every model, LOD1 and LOD2 alike. Reorienting either one
    would compare each tree cell against the buildings on the opposite side of
    the rectangle.
    """
    seen = {}

    def spy(canopy, building_heights, **kwargs):
        seen['heights'] = np.asarray(building_heights).copy()
        return canopy

    monkeypatch.setattr(main_mod, "_sanitize_ndsm_canopy", spy)

    vc = _RefineModel(building_lod=building_lod)
    # Asymmetric, like the land cover, so a stray flip is observable.
    heights = np.zeros(_REFINE_SHAPE)
    heights[:_REFINE_SHAPE[0] // 2, :] = 20.0
    vc.buildings.heights = heights

    _refine(vc)

    assert np.array_equal(seen['heights'], heights)
    assert not np.array_equal(seen['heights'], np.flipud(heights)), (
        "buildings were flipped out of the canopy's frame — each tree cell "
        "would be checked against the buildings on the opposite side")


def test_refine_lod2_leaves_the_model_untouched_when_the_overlay_raises(
        refine_env):
    """reapply_canopy raises ValueError on a missing extras['voxel_min_z'], a
    mismatched mesh_vegetation_mask, or a canopy shape mismatch. The endpoint
    treats refinement failures as non-fatal, so a half-written model would sail
    on with its 2.5-D grids describing crowns the voxel grid does not contain —
    for the rest of the session, reported only in a server log. On this path
    reapply_canopy owns those grids, so the app writes nothing of its own.
    """
    def boom(*a, **k):
        raise ValueError("extras['voxel_min_z'] is missing or None")

    sys.modules["voxcitygml"].reapply_canopy = boom
    vc = _RefineModel(building_lod=2)

    with pytest.raises(ValueError):
        _refine(vc)
    assert not vc.tree_canopy.top.any(), "the canopy grids must be untouched"
    assert not vc.tree_canopy.bottom.any(), "the canopy grids must be untouched"


@pytest.mark.skipif(_real_reapply_canopy is None,
                    reason="voxcitygml with reapply_canopy is not installed")
def test_refine_lod2_call_matches_the_real_reapply_canopy_signature(refine_env):
    """Bind the cross-repo contract, as the VoxelizerConfig test does above.

    ``refine_env`` stubs reapply_canopy with a fake that swallows any argument
    list, so every other assertion here would stay green while production
    raised TypeError. This replays the captured call — verbatim, keywords as
    keywords — against the *real* signature, captured at import time before the
    fixture patched ``sys.modules``.
    """
    vc = _RefineModel(building_lod=2)
    assert _refine(vc) is True
    raw = refine_env['reapply']

    # bind() raises TypeError on an unknown/renamed keyword or a changed arity.
    bound = inspect.signature(_real_reapply_canopy).bind(*raw['args'],
                                                         **raw['kwargs'])
    # bind() alone is not enough: a *positional* argument binds to whatever the
    # parameter is now called, so a rename of canopy_top would pass silently.
    # Assert the names the call actually resolves to.
    assert bound.arguments.keys() == {"city", "canopy_top", "canopy_bottom"}


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

    Caveat on the message: in the real pipeline a degenerate rectangle would
    likely trip "No CityGML buildings found" first, so what is pinned here is
    the *mapping* (ValueError -> 400), not the exact wording a user sees.
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
