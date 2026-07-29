"""Tests for share-by-URL: share module helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import config, share


@pytest.fixture(autouse=True)
def _share_dir(tmp_path):
    share.set_share_base_dir(tmp_path / "shares")
    yield
    share.set_share_base_dir(Path(config.SHARE_DIR))


# --- token validation -------------------------------------------------------

def test_is_valid_token_accepts_urlsafe_tokens():
    assert share.is_valid_token("Ab3xK9_qRt2mVw8pLzYh4g")  # 22 chars
    assert share.is_valid_token("a" * 16)
    assert share.is_valid_token("a" * 64)


def test_is_valid_token_rejects_malformed_tokens():
    assert not share.is_valid_token("")
    assert not share.is_valid_token("short")
    assert not share.is_valid_token("a" * 65)
    assert not share.is_valid_token("../" + "a" * 16)
    assert not share.is_valid_token("a" * 16 + "/x")


def test_share_zip_path_rejects_invalid_token_without_touching_fs():
    assert share.share_zip_path("../etc/passwd") is None
    assert share.share_zip_path("short") is None


def test_share_zip_path_none_for_unknown_token():
    assert share.share_zip_path("a" * 20) is None


# --- create_share -----------------------------------------------------------

def _fake_state():
    # save_session_to_zip only needs a state with a scene; monkeypatched below.
    return SimpleNamespace(sim_results_by_type={})


def test_create_share_writes_zip_and_meta(monkeypatch, tmp_path):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"ok": True}))
        buf.seek(0)
        return buf

    monkeypatch.setattr(share, "save_session_to_zip", fake_save)

    token = share.create_share(_fake_state(), frontend_state='{"zones":[]}')
    assert share.is_valid_token(token)

    zip_path = share.share_zip_path(token)
    assert zip_path is not None and zip_path.is_file()

    meta_path = zip_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["has_sim_results"] is False
    assert meta["size_bytes"] > 0
    assert "created_at_utc" in meta


def test_create_share_leaves_no_staging_dir_on_success(monkeypatch, tmp_path):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x", "y")
        buf.seek(0)
        return buf

    monkeypatch.setattr(share, "save_session_to_zip", fake_save)
    share.create_share(_fake_state())
    base = share.share_base_dir()
    assert not list(base.glob(".tmp-*"))


def test_create_share_cleans_up_staging_and_reraises_on_write_failure(monkeypatch):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        buf.write(b"zipbytes")
        buf.seek(0)
        return buf

    monkeypatch.setattr(share, "save_session_to_zip", fake_save)
    # Fail AFTER the staging dir + session.zip are created, while writing meta.json.
    monkeypatch.setattr(share.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        share.create_share(_fake_state())

    base = share.share_base_dir()
    # No staging dir and no committed token dir should remain.
    assert not list(base.glob(".tmp-*"))
    assert not any(p.is_dir() and p.name != "" and not p.name.startswith(".tmp-") for p in base.glob("*"))


# --- endpoints --------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    import io

    def fake_save(state, **kwargs):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"v": 1}))
        buf.seek(0)
        return buf

    # Patch where create_share looks the symbol up.
    monkeypatch.setattr(share, "save_session_to_zip", fake_save)

    from backend.main import app, app_state
    app_state.sim_results_by_type = {}
    # Make the load path deterministic: apply_session_to_state returns a summary.
    monkeypatch.setattr(
        "backend.main.parse_session_zip",
        lambda stream, max_bytes=None: SimpleNamespace(_root="ignored"),
    )
    monkeypatch.setattr(
        "backend.main.apply_session_to_state",
        lambda parsed, state: {"has_voxcity": True, "rectangle_vertices": None,
                               "land_cover_source": "OpenStreetMap",
                               "frontend_state": '{"zones":[]}',
                               "has_sim_results": False, "last_sim_type": None,
                               "sim_result_types": [], "landmark_building_ids": []},
    )
    monkeypatch.setattr("backend.main.parsed_session_temp_root", lambda parsed: None)
    monkeypatch.setattr("backend.main.shutil.rmtree", lambda *a, **k: None)
    return TestClient(app)


def test_post_share_returns_token_and_path(client):
    res = client.post("/api/share", data={"frontend_state": '{"zones":[]}'})
    assert res.status_code == 200
    body = res.json()
    assert share.is_valid_token(body["token"])
    assert body["path"] == f"/share/{body['token']}"


def test_load_unknown_share_is_404(client):
    res = client.post("/api/share/" + "a" * 20 + "/load")
    assert res.status_code == 404


def test_share_roundtrip_creates_then_loads(client):
    token = client.post("/api/share").json()["token"]
    res = client.post(f"/api/share/{token}/load")
    assert res.status_code == 200
    assert res.json()["has_voxcity"] is True
