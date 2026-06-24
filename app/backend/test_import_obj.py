"""Tests for the OBJ import endpoints (upload / commit)."""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend.main import app, import_obj_store
from backend.state import app_state

# Reuse the importer's flat-model fixture builder.
from tests.importer.conftest import make_flat_voxcity


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _model_loaded():
    """Install a small flat model in app_state for the duration of each test."""
    app_state.voxcity = make_flat_voxcity(nx=30, ny=30, nz=12, meshsize=1.0)
    app_state.rectangle_vertices = app_state.voxcity.extras["rectangle_vertices"]
    app_state.land_cover_source = "OpenStreetMap"
    import_obj_store.clear()
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    import_obj_store.clear()


def _box_obj_bytes() -> bytes:
    """A single 3x3x4 box exported to OBJ, returned as bytes."""
    mesh = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    mesh.apply_translation((1.5, 1.5, 2.0))  # min corner at origin
    return mesh.export(file_type="obj").encode("utf-8")


def test_upload_returns_groups_and_preview(client):
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["import_id"]
    assert len(body["groups"]) >= 1
    assert body["groups"][0]["role"] == "building"
    assert body["groups"][0]["n_faces"] > 0
    assert len(body["preview"]["vertices"]) > 0
    assert len(body["preview"]["indices"]) > 0
    # import_id is registered server-side
    assert body["import_id"] in import_obj_store


def test_upload_rejects_non_obj_garbage(client):
    files = {"file": ("bad.obj", io.BytesIO(b"this is not an obj"), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400


def test_upload_rejects_structurally_malformed_obj(client):
    """A face referencing an out-of-range vertex index throws IndexError deep in
    trimesh's parser, not load_obj_groups's own ValueError/FileNotFoundError --
    this must still surface as a clean 400, not an uncaught 500."""
    bad_obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n"  # vertex index 99 doesn't exist
    files = {"file": ("malformed.obj", io.BytesIO(bad_obj), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400, r.text


def test_upload_requires_model(client):
    app_state.voxcity = None
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400


def _upload_box(client) -> str:
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()["import_id"]


def _domain_center_lonlat() -> list[float]:
    rect = app_state.rectangle_vertices
    lons = [p[0] for p in rect]
    lats = [p[1] for p in rect]
    return [sum(lons) / len(lons), sum(lats) / len(lats)]


def test_commit_imports_building(client):
    import_id = _upload_box(client)
    before = int(np.sum(app_state.voxcity.voxels.classes == -3))
    req = {
        "import_id": import_id,
        "placement": {
            "anchor_lonlat": _domain_center_lonlat(),
            "anchor_elevation": None,            # auto-sample DEM
            "anchor_model_point": [0.0, 0.0, 0.0],
            "rotation": 0.0,
            "move": [0.0, 0.0, 0.0],
            "units": "m",
            "z_up": True,
            "swap_yz": False,
        },
        "roles": {},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_building_voxels_added"] > 0
    assert len(body["imported_building_ids"]) >= 1
    assert body["figure_json"]
    after = int(np.sum(app_state.voxcity.voxels.classes == -3))
    assert after > before


def test_commit_skips_non_building_role(client):
    import_id = _upload_box(client)
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    name = client.post("/api/model/import_obj/upload", files=files).json()["groups"][0]["name"]
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat()},
        "roles": {name: "skip"},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["n_building_voxels_added"] == 0


def test_commit_unknown_import_id_404(client):
    req = {"import_id": "deadbeef", "placement": {"anchor_lonlat": _domain_center_lonlat()}}
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 404
