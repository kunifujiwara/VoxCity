"""SPA serving is active only when a built frontend dist/ is available."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_spa_served_when_dist_present(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>VoxCity SPA</title>")
    (dist / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    import backend.main as m
    monkeypatch.setattr(m.config, "FRONTEND_DIST", str(dist))
    client = TestClient(m.app)

    # root serves index.html
    r = client.get("/")
    assert r.status_code == 200
    assert "VoxCity SPA" in r.text

    # deep client-route falls back to index.html
    r2 = client.get("/some/tab")
    assert r2.status_code == 200
    assert "VoxCity SPA" in r2.text

    # a real static file is served directly
    r3 = client.get("/logo.png")
    assert r3.status_code == 200
    assert r3.content.startswith(b"\x89PNG")

    # a registered API route still works (not shadowed by the catch-all)
    r4 = client.get("/api/health")
    assert r4.status_code == 200
    assert "VoxCity SPA" not in r4.text

    # an unknown API path 404s (does NOT fall through to index.html)
    r5 = client.get("/api/does-not-exist")
    assert r5.status_code == 404
    assert "VoxCity SPA" not in r5.text


def test_spa_404_when_dist_absent(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.config, "FRONTEND_DIST", None)
    client = TestClient(m.app)
    r = client.get("/some/tab")
    assert r.status_code == 404
