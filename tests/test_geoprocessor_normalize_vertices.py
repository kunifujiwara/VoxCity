"""Tests for geoprocessor.utils.normalize_rectangle_vertices."""
import logging
import math

import pytest

from voxcity.geoprocessor.utils import normalize_rectangle_vertices

# Canonical [SW, NW, NE, SE] axis-aligned rectangle (lon, lat)
SW = (139.75, 35.65)
NW = (139.75, 35.66)
NE = (139.76, 35.66)
SE = (139.76, 35.65)
CANONICAL = [SW, NW, NE, SE]


def _rotations_and_windings(quad):
    """All 4 rotations of the ring, in both windings."""
    variants = []
    for start in range(4):
        rot = [quad[(start + k) % 4] for k in range(4)]
        variants.append(rot)
        variants.append(list(reversed(rot)))
    return variants


def test_canonical_input_is_unchanged():
    assert normalize_rectangle_vertices(CANONICAL) == CANONICAL


def test_all_rotations_and_windings_normalize_to_canonical():
    for variant in _rotations_and_windings(CANONICAL):
        assert normalize_rectangle_vertices(variant) == CANONICAL, variant


def test_idempotent():
    out1 = normalize_rectangle_vertices([NE, SE, SW, NW])
    out2 = normalize_rectangle_vertices(out1)
    assert out1 == out2 == CANONICAL


def test_closed_ring_accepted():
    ring = CANONICAL + [CANONICAL[0]]
    assert normalize_rectangle_vertices(ring) == CANONICAL


def test_wrong_count_raises():
    with pytest.raises(ValueError, match="4"):
        normalize_rectangle_vertices(CANONICAL[:3])


def test_latlon_swap_raises_with_hint():
    # (lat, lon)-swapped input: 139.x as latitude is out of range
    bad = [(35.65, 139.75), (35.66, 139.75), (35.66, 139.76), (35.65, 139.76)]
    with pytest.raises(ValueError, match=r"\(lat, lon\)"):
        normalize_rectangle_vertices(bad)


def test_rotated_rectangle_vertex_set_preserved():
    # ~30°-rotated rectangle around (0, 0); canonical order by construction:
    # v0 (origin), v1 = v0 + n*L (northward edge), v3 = v0 + e*W (eastward edge)
    ang = math.radians(30.0)
    n = (math.sin(ang) * 0.01, math.cos(ang) * 0.01)   # northward-ish edge
    e = (math.cos(ang) * 0.005, -math.sin(ang) * 0.005)  # eastward-ish edge
    v0 = (0.0, 0.0)
    v1 = (v0[0] + n[0], v0[1] + n[1])
    v2 = (v1[0] + e[0], v1[1] + e[1])
    v3 = (v0[0] + e[0], v0[1] + e[1])
    rotated_canonical = [v0, v1, v2, v3]
    for variant in _rotations_and_windings(rotated_canonical):
        out = normalize_rectangle_vertices(variant)
        assert sorted(out) == sorted(rotated_canonical)
        assert out == rotated_canonical, variant


def test_reorder_warns(caplog, propagate_voxcity_logs):
    with caplog.at_level(logging.WARNING, logger="voxcity"):
        normalize_rectangle_vertices([NE, SE, SW, NW])
    assert any("reordered" in r.message for r in caplog.records)


def test_canonical_input_does_not_warn(caplog, propagate_voxcity_logs):
    with caplog.at_level(logging.WARNING, logger="voxcity"):
        normalize_rectangle_vertices(CANONICAL)
    assert not [r for r in caplog.records if "reordered" in r.message]


def test_warn_false_suppresses_reorder_warning(caplog, propagate_voxcity_logs):
    with caplog.at_level(logging.WARNING, logger="voxcity"):
        out = normalize_rectangle_vertices([NE, SE, SW, NW], warn=False)
    assert out == CANONICAL
    assert not [r for r in caplog.records if "reordered" in r.message]


def test_non_parallelogram_warns(caplog, propagate_voxcity_logs):
    skewed = [SW, NW, (139.765, 35.663), SE]  # NE pushed well off the affine frame
    with caplog.at_level(logging.WARNING, logger="voxcity"):
        normalize_rectangle_vertices(skewed, warn=False)
    assert any("parallelogram" in r.message for r in caplog.records)
