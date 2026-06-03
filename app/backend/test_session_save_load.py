"""HTTP-level tests for /api/session/save and /api/session/load."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    sentinel = b"H5"

    def fake_save(path, _voxcity):
        Path(path).write_bytes(sentinel)

    def fake_load(path):
        assert Path(path).read_bytes() == sentinel
        return SimpleNamespace(voxels=SimpleNamespace(meta=SimpleNamespace(meshsize=5.0)))

    import voxcity.io as voxcity_io
    monkeypatch.setattr(voxcity_io, "save_voxcity", fake_save, raising=False)
    monkeypatch.setattr(voxcity_io, "load_voxcity", fake_load, raising=False)

    monkeypatch.setattr(
        "voxcity.simulator_gpu.visibility.integration.clear_visibility_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        "voxcity.simulator_gpu.solar.integration.caching.clear_all_caches",
        lambda: None,
        raising=False,
    )

    from backend.main import app, app_state
    app_state.voxcity = SimpleNamespace(voxels=SimpleNamespace(meta=SimpleNamespace(meshsize=5.0)))
    app_state.rectangle_vertices = [[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]]
    app_state.land_cover_source = "OpenStreetMap"
    yield TestClient(app)
    app_state.voxcity = None


def test_save_endpoint_returns_zip(client: TestClient) -> None:
    r = client.post("/api/session/save")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    assert "voxcity.h5" in names
    assert "meta.json" in names


def test_save_endpoint_with_frontend_state_round_trips_via_load(client: TestClient) -> None:
    payload = json.dumps({"zones": [{"id": "z_1"}]})
    save = client.post("/api/session/save", data={"frontend_state": payload})
    assert save.status_code == 200

    load = client.post(
        "/api/session/load",
        files={"file": ("session.zip", save.content, "application/zip")},
    )
    assert load.status_code == 200
    body = load.json()
    assert body["has_voxcity"] is True
    assert body["frontend_state"] == payload


def test_save_without_model_returns_400(client: TestClient) -> None:
    from backend.main import app_state
    app_state.voxcity = None
    r = client.post("/api/session/save")
    assert r.status_code == 400


def test_load_malformed_zip_returns_400(client: TestClient) -> None:
    r = client.post(
        "/api/session/load",
        files={"file": ("session.zip", b"not a zip", "application/zip")},
    )
    assert r.status_code == 400


def test_load_oversized_returns_413(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("VOXCITY_SESSION_MAX_UPLOAD_MB", "0")
    save = client.post("/api/session/save")
    r = client.post(
        "/api/session/load",
        files={"file": ("session.zip", save.content, "application/zip")},
    )
    assert r.status_code == 413
