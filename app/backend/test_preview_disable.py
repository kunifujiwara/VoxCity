# app/backend/test_preview_disable.py
"""Unit tests for the preview-disable threshold helpers."""
from __future__ import annotations

import numpy as np

from backend.main import (
    PREVIEW_MAX_CELLS,
    _preview_disabled_for_shape,
    _preview_figure_json,
)


def test_threshold_default_is_one_million():
    assert PREVIEW_MAX_CELLS == 1_000_000


def test_disabled_only_above_threshold():
    # 1000 x 1000 = 1_000_000 -> NOT disabled (strictly greater trips it)
    assert _preview_disabled_for_shape((1000, 1000, 20)) is False
    # 1001 x 1000 -> disabled
    assert _preview_disabled_for_shape((1001, 1000, 20)) is True
    # small grid -> not disabled
    assert _preview_disabled_for_shape((100, 100, 30)) is False
    # 2D shape supported
    assert _preview_disabled_for_shape((2000, 2000)) is True
    # degenerate -> not disabled
    assert _preview_disabled_for_shape(()) is False
    assert _preview_disabled_for_shape((5,)) is False


def test_preview_figure_json_skips_build_when_disabled(monkeypatch):
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return "{\"figure\": true}"

    monkeypatch.setattr("backend.main._make_plotly_json", spy)

    big = np.zeros((1200, 1200, 5), dtype=np.int8)
    assert _preview_figure_json(big, 5.0, {"title": "x"}) == ""
    assert called["n"] == 0  # expensive build skipped

    small = np.zeros((50, 50, 5), dtype=np.int8)
    assert _preview_figure_json(small, 5.0, {"title": "x"}) == "{\"figure\": true}"
    assert called["n"] == 1


# append to app/backend/test_preview_disable.py
from types import SimpleNamespace
from fastapi.testclient import TestClient
import pytest


def _fake_voxcity(shape):
    classes = np.zeros(shape, dtype=np.int8)
    meta = SimpleNamespace(meshsize=5.0)
    return SimpleNamespace(
        voxels=SimpleNamespace(classes=classes, meta=meta),
        extras={},
    )


@pytest.fixture
def client():
    from backend.main import app, app_state
    # Note: app_state.meshsize is a read-only property derived from
    # voxcity.voxels.meta.meshsize (see state.py); each test's fake voxcity
    # supplies meta.meshsize=5.0, so it does not need to be set here.
    app_state.rectangle_vertices = [[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]]
    app_state.land_cover_source = "OpenStreetMap"
    yield TestClient(app), app_state
    app_state.voxcity = None


def test_model_info_reports_preview_disabled(client):
    tc, app_state = client
    app_state.voxcity = _fake_voxcity((1200, 1200, 5))
    r = tc.get("/api/model/info")
    assert r.status_code == 200
    assert r.json()["preview_disabled"] is True


def test_model_info_preview_enabled_for_small_grid(client):
    tc, app_state = client
    app_state.voxcity = _fake_voxcity((100, 100, 5))
    r = tc.get("/api/model/info")
    assert r.status_code == 200
    assert r.json()["preview_disabled"] is False
