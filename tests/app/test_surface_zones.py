"""Unit tests for app.backend.surface_zones helpers."""
import numpy as np
import pytest
from types import SimpleNamespace

from app.backend.models import SurfaceSelector
from app.backend.surface_zones import surface_zone_mask, stats_for_surface_zone


FACE_META = [
    {"face_key": "b1:r", "building_id": 1, "surface_kind": "roof", "orientation": None},
    {"face_key": "b1:e", "building_id": 1, "surface_kind": "wall", "orientation": "E"},
    {"face_key": "b1:n", "building_id": 1, "surface_kind": "wall", "orientation": "N"},
    {"face_key": "b1:bottom", "building_id": 1, "surface_kind": "bottom", "orientation": None},
    {"face_key": "b2:r", "building_id": 2, "surface_kind": "roof", "orientation": None},
]


def test_whole_selector_uses_only_roof_and_wall():
    mask = surface_zone_mask(FACE_META, [SurfaceSelector(building_id=1, mode="whole")])
    assert mask.tolist() == [True, True, True, False, False]


def test_roof_selector():
    mask = surface_zone_mask(FACE_META, [SurfaceSelector(building_id=1, mode="roof")])
    assert mask.tolist() == [True, False, False, False, False]


def test_all_walls_selector():
    mask = surface_zone_mask(FACE_META, [SurfaceSelector(building_id=1, mode="all_walls")])
    assert mask.tolist() == [False, True, True, False, False]


def test_wall_orientation_selector():
    mask = surface_zone_mask(FACE_META, [SurfaceSelector(building_id=1, mode="wall_orientation", orientation="E")])
    assert mask.tolist() == [False, True, False, False, False]


def test_faces_selector():
    mask = surface_zone_mask(FACE_META, [SurfaceSelector(building_id=1, mode="faces", face_keys=["b1:r", "b1:e"])])
    assert mask.tolist() == [True, True, False, False, False]


def test_exclude_faces_subtracts_from_bulk_selection():
    selectors = [
        SurfaceSelector(building_id=1, mode="whole"),
        SurfaceSelector(building_id=1, mode="exclude_faces", face_keys=["b1:e"]),
    ]
    mask = surface_zone_mask(FACE_META, selectors)
    assert mask.tolist() == [True, False, True, False, False]


def test_multiple_building_selectors():
    selectors = [
        SurfaceSelector(building_id=1, mode="roof"),
        SurfaceSelector(building_id=2, mode="roof"),
    ]
    mask = surface_zone_mask(FACE_META, selectors)
    assert mask.tolist() == [True, False, False, False, True]


def test_surface_stats_are_area_weighted():
    values = np.array([10.0, 100.0, 30.0, 999.0, 5.0])
    areas = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    selectors = [SurfaceSelector(building_id=1, mode="whole")]
    stat = stats_for_surface_zone("z1", FACE_META, selectors, values, areas)
    assert stat.cell_count == 3
    assert stat.mean == pytest.approx((10 * 1 + 100 * 2 + 30 * 1) / 4)


def test_surface_stats_empty_selection():
    values = np.array([10.0, 100.0, 30.0, 999.0, 5.0])
    areas = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    selectors = []
    stat = stats_for_surface_zone("z1", FACE_META, selectors, values, areas)
    assert stat.cell_count == 0
    assert stat.mean is None
