"""Unit tests for session_io save / parse helpers."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backend import session_io
from backend.session_io import (
    DEFAULT_MAX_UPLOAD_BYTES,
    SESSION_FORMAT_VERSION,
    SessionLoadError,
    parse_session_zip,
    save_session_to_zip,
)


class _FakeMeshsizeMeta(SimpleNamespace):
    pass


class _FakeVoxels(SimpleNamespace):
    pass


class _FakeVoxCity(SimpleNamespace):
    pass


def _fake_state_with_model(tmp_path: Path) -> Any:
    """Return a SimpleNamespace state that behaves enough like AppState."""
    h5_path = tmp_path / "voxcity.h5"
    h5_path.write_bytes(b"FAKEH5DATA")  # save_voxcity is monkeypatched to write here
    voxcity = _FakeVoxCity(
        voxels=_FakeVoxels(meta=_FakeMeshsizeMeta(meshsize=5.0)),
    )
    return SimpleNamespace(
        voxcity=voxcity,
        rectangle_vertices=[[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]],
        land_cover_source="OpenStreetMap",
    )


@pytest.fixture(autouse=True)
def _patch_save_load_voxcity(monkeypatch, tmp_path):
    """Stub voxcity.io.save_voxcity / load_voxcity so the tests don't need a real H5."""
    sentinel_payload = b"VOXCITY-H5-SENTINEL"

    def fake_save(path: str, _voxcity) -> None:
        Path(path).write_bytes(sentinel_payload)

    def fake_load(path: str):
        data = Path(path).read_bytes()
        assert data == sentinel_payload
        return _FakeVoxCity(
            voxels=_FakeVoxels(meta=_FakeMeshsizeMeta(meshsize=5.0)),
        )

    import voxcity.io as voxcity_io
    monkeypatch.setattr(voxcity_io, "save_voxcity", fake_save, raising=False)
    monkeypatch.setattr(voxcity_io, "load_voxcity", fake_load, raising=False)
    yield


def test_save_then_parse_round_trip_basic(tmp_path: Path) -> None:
    state = _fake_state_with_model(tmp_path)
    buf = save_session_to_zip(state)
    parsed = parse_session_zip(buf)
    assert parsed.meta["format_version"] == SESSION_FORMAT_VERSION
    assert parsed.meta["rectangle_vertices"] == state.rectangle_vertices
    assert parsed.meta["land_cover_source"] == "OpenStreetMap"
    assert parsed.meta["meshsize"] == 5.0
    assert parsed.meta["has_sim_results"] is False
    assert parsed.frontend_state is None
    assert parsed.sim_results is None


def test_save_round_trips_frontend_state_json(tmp_path: Path) -> None:
    state = _fake_state_with_model(tmp_path)
    payload = json.dumps({"zones": [{"id": "z_1"}]})
    buf = save_session_to_zip(state, frontend_state=payload)
    parsed = parse_session_zip(buf)
    assert parsed.frontend_state == payload


def test_save_without_model_raises_value_error(tmp_path: Path) -> None:
    state = SimpleNamespace(
        voxcity=None,
        rectangle_vertices=None,
        land_cover_source="OpenStreetMap",
    )
    with pytest.raises(ValueError, match="no scene has been generated"):
        save_session_to_zip(state)


def test_save_rejects_non_json_frontend_state(tmp_path: Path) -> None:
    state = _fake_state_with_model(tmp_path)
    with pytest.raises(ValueError, match="frontend_state must be valid JSON"):
        save_session_to_zip(state, frontend_state="{not json")


def test_parse_rejects_missing_voxcity_h5(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("meta.json", json.dumps({"format_version": SESSION_FORMAT_VERSION}))
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="voxcity.h5"):
        parse_session_zip(buf)


def test_parse_rejects_missing_meta_json(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("voxcity.h5", b"x")
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="meta.json"):
        parse_session_zip(buf)


def test_parse_rejects_malformed_meta_json(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("voxcity.h5", b"x")
        zf.writestr("meta.json", "{not json")
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="not valid JSON"):
        parse_session_zip(buf)


def test_parse_rejects_newer_format_version(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("voxcity.h5", b"x")
        zf.writestr("meta.json", json.dumps({"format_version": SESSION_FORMAT_VERSION + 1}))
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="newer version"):
        parse_session_zip(buf)


def test_parse_rejects_path_traversal(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escapes.txt", b"x")
        zf.writestr("voxcity.h5", b"x")
        zf.writestr("meta.json", json.dumps({"format_version": SESSION_FORMAT_VERSION}))
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="path traversal"):
        parse_session_zip(buf)


def test_parse_rejects_absolute_paths(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("/etc/passwd", b"x")
        zf.writestr("voxcity.h5", b"x")
        zf.writestr("meta.json", json.dumps({"format_version": SESSION_FORMAT_VERSION}))
    buf.seek(0)
    with pytest.raises(SessionLoadError, match="absolute path"):
        parse_session_zip(buf)


def test_parse_rejects_oversized_upload(tmp_path: Path) -> None:
    state = _fake_state_with_model(tmp_path)
    buf = save_session_to_zip(state)
    with pytest.raises(SessionLoadError, match="exceeds maximum size"):
        parse_session_zip(buf, max_bytes=10)


# ── Task 2: sim_results round-trip ────────────────────────────────────────────

def _state_with_sim_result(tmp_path: Path) -> Any:
    """A state where store_sim_result populated one solar entry."""
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    voxcity_grid = np.ones((3, 4, 2), dtype=np.int8)
    building_id_grid = np.zeros((3, 4), dtype=np.int32)

    entry = SimpleNamespace(
        sim_type="solar",
        target="ground",
        grid=grid,
        mesh=None,
        voxcity_grid=voxcity_grid,
        building_id_grid=building_id_grid,
        view_point_height=1.5,
        colorbar_title="W/m^2",
    )

    state = _fake_state_with_model(tmp_path)
    state.sim_results_by_type = {"solar": entry}
    state.last_sim_type = "solar"
    return state


def test_save_with_sim_results_round_trips_grid(tmp_path: Path) -> None:
    state = _state_with_sim_result(tmp_path)
    buf = save_session_to_zip(state, include_sim_results=True)
    parsed = parse_session_zip(buf)
    assert parsed.meta["has_sim_results"] is True
    assert parsed.sim_results is not None
    entries = parsed.sim_results["entries"]
    assert len(entries) == 1
    assert entries[0]["sim_type"] == "solar"
    np.testing.assert_array_equal(entries[0]["grid"], np.arange(12).reshape(3, 4))


def test_save_without_include_sim_results_omits_them(tmp_path: Path) -> None:
    state = _state_with_sim_result(tmp_path)
    buf = save_session_to_zip(state, include_sim_results=False)
    parsed = parse_session_zip(buf)
    assert parsed.meta["has_sim_results"] is False
    assert parsed.sim_results is None


def test_save_rejects_unsafe_sim_type_name(tmp_path: Path) -> None:
    state = _state_with_sim_result(tmp_path)
    state.sim_results_by_type["../escape"] = state.sim_results_by_type["solar"]
    with pytest.raises(ValueError, match="unsafe sim_type"):
        save_session_to_zip(state, include_sim_results=True)


# ── Task 3: apply_session_to_state ────────────────────────────────────────────

def test_apply_session_restores_state_atomically(tmp_path: Path, monkeypatch) -> None:
    cleared: dict[str, bool] = {"visibility": False, "solar": False}

    def fake_clear_visibility():
        cleared["visibility"] = True

    def fake_clear_solar():
        cleared["solar"] = True

    monkeypatch.setattr(
        "voxcity.simulator_gpu.visibility.integration.clear_visibility_cache",
        fake_clear_visibility,
        raising=False,
    )
    monkeypatch.setattr(
        "voxcity.simulator_gpu.solar.integration.caching.clear_all_caches",
        fake_clear_solar,
        raising=False,
    )

    from backend.state import AppState
    from backend.session_io import apply_session_to_state

    state = AppState()
    state.last_base_fig_json = "STALE"
    state.last_hidden_classes = [1, 2, 3]
    state.raw_data = {"stale": True}

    saved_state = _fake_state_with_model(tmp_path)
    buf = save_session_to_zip(saved_state)
    parsed = parse_session_zip(buf)

    summary = apply_session_to_state(parsed, state)
    assert summary["has_voxcity"] is True
    assert summary["rectangle_vertices"] == saved_state.rectangle_vertices
    assert summary["land_cover_source"] == "OpenStreetMap"
    assert summary["frontend_state"] is None

    assert state.last_base_fig_json is None
    assert state.last_hidden_classes is None
    assert state.raw_data == {}
    assert cleared["visibility"] is True
    assert cleared["solar"] is True
