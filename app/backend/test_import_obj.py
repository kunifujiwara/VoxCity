"""Tests for the OBJ import endpoints (upload / commit)."""
from __future__ import annotations

import io
import json
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


def test_commit_rejects_wrong_length_move(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "move": [1.0, 2.0]},
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 400, r.text
    assert "move" in r.json()["detail"].lower()


def test_commit_rejects_nan_anchor(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": [float("nan"), 35.0]},
    }
    # httpx's TestClient.post(json=...) encodes with allow_nan=False and raises
    # ValueError before the request is ever sent, so NaN payloads must be built
    # manually with stdlib json.dumps (which allows NaN by default) and posted
    # as raw content with an explicit content-type.
    body = json.dumps(req).encode("utf-8")
    r = client.post(
        "/api/model/import_obj/commit",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400, r.text
    assert "anchor_lonlat" in r.json()["detail"].lower()


def test_commit_rejects_wrong_length_anchor_model_point(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "anchor_model_point": [0.0, 0.0]},
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 400, r.text
    assert "anchor_model_point" in r.json()["detail"].lower()


def test_commit_overlap_warning_distinct_from_offdomain(client):
    # First import lands a building at the centre.
    id1 = _upload_box(client)
    center = _domain_center_lonlat()
    r1 = client.post("/api/model/import_obj/commit", json={
        "import_id": id1, "placement": {"anchor_lonlat": center}, "roles": {}, "overwrite": True,
    })
    assert r1.status_code == 200 and r1.json()["n_building_voxels_added"] > 0

    # Second import of the SAME box at the SAME spot fully overlaps -> 0 net added,
    # but ids ARE assigned (in-domain). Warning must say "overlap", not "0 cells".
    id2 = _upload_box(client)
    r2 = client.post("/api/model/import_obj/commit", json={
        "import_id": id2, "placement": {"anchor_lonlat": center}, "roles": {}, "overwrite": True,
    })
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["n_building_voxels_added"] == 0
    assert body["warning"] is not None
    assert "overlap" in body["warning"].lower()
    assert "0 cells" not in body["warning"].lower()


def test_commit_rejects_nan_rotation(client):
    import_id = _upload_box(client)
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "rotation": float("nan")},
    }
    # See test_commit_rejects_nan_anchor: httpx's TestClient.post(json=...) encodes
    # with allow_nan=False and raises ValueError before the request is ever sent,
    # so NaN payloads must be built manually with stdlib json.dumps (which allows
    # NaN by default) and posted as raw content with an explicit content-type.
    body = json.dumps(req).encode("utf-8")
    r = client.post(
        "/api/model/import_obj/commit",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400, r.text
    assert "rotation" in r.json()["detail"].lower()


def test_commit_offdomain_anchor_warns(client):
    import_id = _upload_box(client)
    # Anchor far from the model rectangle (the flat fixture is near (0,0));
    # provide explicit elevation so the test doesn't depend on the auto-sample
    # path's own off-domain clamping behavior.
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": [10.0, 10.0], "anchor_elevation": 0.0},
        "roles": {}, "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    w = r.json()["warning"]
    assert w is not None and "outside the model domain" in w.lower()


def test_anchor_ground_returns_datum(client):
    lon, lat = _domain_center_lonlat()
    r = client.get("/api/model/anchor_ground", params={"lon": lon, "lat": lat})
    assert r.status_code == 200, r.text
    body = r.json()
    # Flat fixture (make_flat_voxcity(nx=30, ny=30, nz=12, meshsize=1.0)): DEM is
    # an all-zeros (30, 30) grid, so both the sampled elevation and the grid min
    # are 0.0; meshsize is the 1.0 passed to the fixture.
    assert body["dem_elevation"] == 0.0
    assert body["dem_min"] == 0.0
    assert body["meshsize_m"] == 1.0


def test_anchor_ground_requires_model(client):
    app_state.voxcity = None
    r = client.get("/api/model/anchor_ground", params={"lon": 0.0, "lat": 0.0})
    assert r.status_code == 400


def test_anchor_ground_offdomain_clamps_without_error(client):
    # Off-domain anchor must still return a datum (nearest in-bounds cell), not 500.
    r = client.get("/api/model/anchor_ground", params={"lon": 10.0, "lat": 10.0})
    assert r.status_code == 200, r.text
    assert "dem_elevation" in r.json()


def _box_with_window_obj_bytes() -> bytes:
    """A box plus a window-named planar pane, exported to OBJ bytes."""
    box = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    box.apply_translation((1.5, 1.5, 2.0))
    pane = trimesh.Trimesh(
        vertices=np.array(
            [[0.5, 0.0, 0.5], [2.5, 0.0, 0.5], [2.5, 0.0, 3.5], [0.5, 0.0, 3.5]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="BuildingA", geom_name="BuildingA")
    scene.add_geometry(pane, node_name="Windows", geom_name="Windows")
    return scene.export(file_type="obj").encode("utf-8")


def test_upload_reports_window_role(client):
    files = {"file": ("bw.obj", io.BytesIO(_box_with_window_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    roles = {g["name"]: g["role"] for g in r.json()["groups"]}
    assert roles.get("Windows") == "window"
    assert roles.get("BuildingA") == "building"


def test_commit_reports_window_voxels(client):
    files = {"file": ("bw.obj", io.BytesIO(_box_with_window_obj_bytes()), "text/plain")}
    import_id = client.post("/api/model/import_obj/upload", files=files).json()["import_id"]
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "anchor_elevation": 0.0},
        "roles": {},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["n_window_voxels_added"] > 0
