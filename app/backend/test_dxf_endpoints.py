"""Tests for the DXF import endpoints (upload / commit / delete)."""
from __future__ import annotations

import io

import ezdxf
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, import_dxf_store
from backend.state import app_state
from tests.importer.conftest import make_flat_voxcity


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _model_loaded():
    app_state.voxcity = make_flat_voxcity(nx=30, ny=30, nz=12, meshsize=1.0)
    app_state.rectangle_vertices = app_state.voxcity.extras["rectangle_vertices"]
    app_state.land_cover_source = "OpenStreetMap"
    app_state.auxiliary_lines = []
    import_dxf_store.clear()
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    app_state.auxiliary_lines = []
    import_dxf_store.clear()


def _dxf_bytes() -> bytes:
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (4, 0), (4, 3)], close=True, dxfattribs={"layer": "window"})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def test_dxf_upload_returns_layers_and_center(client):
    files = {"file": ("test.dxf", _dxf_bytes(), "application/dxf")}
    r = client.post("/api/model/import_dxf/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [l["name"] for l in body["layers"]] == ["window"]
    assert body["model_center"] == [2.0, 1.5]
    assert body["import_id"]


def test_dxf_commit_populates_auxiliary_lines_and_geo(client):
    up = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("test.dxf", _dxf_bytes(), "application/dxf")},
    ).json()
    geo0 = client.get("/api/model/geo").json()
    anchor = geo0["center"][::-1]  # geo center is [lat, lon]; placement wants [lon, lat]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {"window": True},
    })
    assert commit.status_code == 200, commit.text
    lines = commit.json()["auxiliary_lines"]
    assert len(lines) == 1
    assert lines[0]["layer"] == "window"
    assert lines[0]["file_name"] == "test.dxf"
    assert len(lines[0]["points"]) == 4  # closed ring

    geo = client.get("/api/model/geo").json()
    assert len(geo["auxiliary_lines"]) == 1


def test_dxf_upload_warns_when_geometry_only_in_blocks(client):
    doc = ezdxf.new()
    blk = doc.blocks.new(name="B")
    blk.add_line((0, 0), (1, 1))
    doc.modelspace().add_blockref("B", (0, 0))
    buf = io.StringIO(); doc.write(buf)
    r = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("b.dxf", buf.getvalue().encode(), "application/dxf")},
    )
    assert r.status_code == 200
    assert "block" in (r.json().get("warning") or "").lower()


def test_dxf_upload_rejects_malformed(client):
    r = client.post(
        "/api/model/import_dxf/upload",
        files={"file": ("bad.dxf", b"not a dxf", "application/dxf")},
    )
    assert r.status_code == 400


def test_delete_auxiliary_lines(client):
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    r = client.delete("/api/model/auxiliary_lines")
    assert r.status_code == 200
    assert client.get("/api/model/geo").json()["auxiliary_lines"] == []
    assert client.delete("/api/model/auxiliary_lines").status_code == 200


def test_upload_requires_model(client):
    app_state.voxcity = None
    r = client.post("/api/model/import_dxf/upload",
                    files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")})
    assert 400 <= r.status_code < 500


def test_commit_requires_model(client):
    app_state.voxcity = None
    r = client.post("/api/model/import_dxf/commit", json={
        "import_id": "nope",
        "placement": {"anchor_lonlat": [139.7, 35.69], "anchor_model_point": [0, 0],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    assert 400 <= r.status_code < 500


def test_commit_layer_visibility_filters(client):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (1, 0), (1, 1)], close=True, dxfattribs={"layer": "a"})
    msp.add_lwpolyline([(2, 2), (3, 2), (3, 3)], close=True, dxfattribs={"layer": "b"})
    buf = io.StringIO(); doc.write(buf)
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("m.dxf", buf.getvalue().encode(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {"a": True, "b": False}})
    assert commit.status_code == 200, commit.text
    layers = {ln["layer"] for ln in commit.json()["auxiliary_lines"]}
    assert layers == {"a"}


def test_delete_auxiliary_lines_by_file_and_id(client):
    up1 = client.post("/api/model/import_dxf/upload",
                      files={"file": ("f1.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    pl = {"anchor_lonlat": anchor, "anchor_model_point": up1["model_center"],
          "rotation": 0, "move": [0, 0], "units": "m"}
    client.post("/api/model/import_dxf/commit",
                json={"import_id": up1["import_id"], "placement": pl, "layer_visibility": {}})
    up2 = client.post("/api/model/import_dxf/upload",
                      files={"file": ("f2.dxf", _dxf_bytes(), "application/dxf")}).json()
    pl2 = {**pl, "anchor_model_point": up2["model_center"]}
    client.post("/api/model/import_dxf/commit",
                json={"import_id": up2["import_id"], "placement": pl2, "layer_visibility": {}})
    lines = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert {ln["file_name"] for ln in lines} == {"f1.dxf", "f2.dxf"}
    assert client.delete("/api/model/auxiliary_lines?file_name=f1.dxf").status_code == 200
    lines = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert {ln["file_name"] for ln in lines} == {"f2.dxf"}
    target_id = lines[0]["id"]
    assert client.delete(f"/api/model/auxiliary_lines?id={target_id}").status_code == 200
    remaining = client.get("/api/model/geo").json()["auxiliary_lines"]
    assert all(ln["id"] != target_id for ln in remaining)


def test_far_from_origin_lands_on_map(client):
    # A DXF drawn on a survey grid ~1,000 km from its own origin must still
    # bake to lon/lat within a few hundred metres of the chosen anchor
    # (anchor_model_point pins the DXF's own centre to the anchor).
    # NOTE: VoxCity's make_flat_voxcity rectangle sits near the EQUATOR
    # (lon0, lat0 = 0.0, 0.0), NOT Tokyo — bounds are center-relative.
    base = 1_000_000.0
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline(
        [(base, base), (base + 4, base), (base + 4, base + 3)],
        close=True, dxfattribs={"layer": "window"})
    buf = io.StringIO(); doc.write(buf)
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("far.dxf", buf.getvalue().encode(), "application/dxf")}).json()
    center_latlon = client.get("/api/model/geo").json()["center"]
    anchor = center_latlon[::-1]  # [lon, lat]
    commit = client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}}).json()
    pts = commit["auxiliary_lines"][0]["points"]
    lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
    # ~0.005 deg ≈ 550 m: generous but catches an un-anchored 1,000 km offset.
    tol = 0.005
    assert anchor[0] - tol < min(lons) and max(lons) < anchor[0] + tol
    assert anchor[1] - tol < min(lats) and max(lats) < anchor[1] + tol


def test_commit_does_not_mutate_voxels(client):
    before = int(np.asarray(app_state.voxcity.voxels.classes).sum())
    up = client.post("/api/model/import_dxf/upload",
                     files={"file": ("t.dxf", _dxf_bytes(), "application/dxf")}).json()
    anchor = client.get("/api/model/geo").json()["center"][::-1]
    client.post("/api/model/import_dxf/commit", json={
        "import_id": up["import_id"],
        "placement": {"anchor_lonlat": anchor, "anchor_model_point": up["model_center"],
                      "rotation": 0, "move": [0, 0], "units": "m"},
        "layer_visibility": {}})
    after = int(np.asarray(app_state.voxcity.voxels.classes).sum())
    assert before == after
