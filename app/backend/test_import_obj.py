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


def test_upload_requires_model(client):
    app_state.voxcity = None
    files = {"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 400
