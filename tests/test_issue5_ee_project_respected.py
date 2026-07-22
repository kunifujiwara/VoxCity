"""Regression test for issue #5: initialize_earth_engine must respect a user's
already-initialized Earth Engine session and must not hardcode a project ID."""

import types

import pytest

from voxcity.downloader import gee


def test_skips_reinit_when_already_initialized(monkeypatch):
    pytest.importorskip("ee")
    calls = {"getAssetRoots": 0, "Initialize": 0}

    def fake_get_asset_roots():
        calls["getAssetRoots"] += 1
        return [{"id": "users/someone"}]  # success => already initialized

    def fake_initialize(*a, **k):
        calls["Initialize"] += 1

    monkeypatch.setattr(gee.ee, "data", types.SimpleNamespace(getAssetRoots=fake_get_asset_roots))
    monkeypatch.setattr(gee.ee, "Initialize", fake_initialize)

    gee.initialize_earth_engine()

    assert calls["getAssetRoots"] == 1
    assert calls["Initialize"] == 0  # must NOT reinit over the user's session


def test_no_hardcoded_project_id_in_source():
    import inspect

    src = inspect.getsource(gee.initialize_earth_engine)
    # The fixed implementation reads project from kwargs/env, never a literal.
    assert "GEE_PROJECT" in src or "project" in src
    # Guard against a reintroduced hardcoded project like project='some-ee-project'
    collapsed = src.replace(" ", "")
    assert "project='" not in collapsed
    assert 'project="' not in collapsed
