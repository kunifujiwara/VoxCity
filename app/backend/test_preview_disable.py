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
