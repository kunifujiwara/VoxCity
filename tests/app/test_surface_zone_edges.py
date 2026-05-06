import numpy as np
import pytest

from app.backend.models import SurfaceSelector, ZoneSpec
from app.backend import surface_zone_edges as surface_zone_edges_module
from app.backend.surface_zone_edges import (
    AXIS_EPSILON_M,
    build_surface_zone_edge_payloads,
    extract_voxel_surface_records,
    is_axis_aligned_segment,
    normalize_segment_key,
    select_records_for_zone,
)
from app.backend.surface_zones import classify_surface_faces
from app.backend.scene_geometry import build_surface_selection_buffers
from voxcity.geoprocessor.mesh import create_voxel_mesh


def _single_building_voxels(shape=(3, 3, 3)):
    voxels = np.zeros(shape, dtype=int)
    ids = np.zeros(shape[:2], dtype=int)
    voxels[1, 1, 0] = -3
    ids[1, 1] = 7
    return voxels, ids


def _zone(zone_id, selectors):
    return ZoneSpec(
        id=zone_id,
        name=zone_id,
        type="building_surface",
        selectors=selectors,
    )


def test_record_face_keys_match_building_surfaces_mesh_for_single_voxel():
    voxels, ids = _single_building_voxels()
    records = extract_voxel_surface_records(voxels, ids, meshsize=1.0)

    mesh = create_voxel_mesh(
        voxels,
        -3,
        meshsize=1.0,
        building_id_grid=ids,
        mesh_type="open_air",
    )
    chunk, face_meta = build_surface_selection_buffers(mesh)

    record_keys = [key for record in records for key in record.face_keys]
    endpoint_keys = [meta.face_key for meta in face_meta]

    assert len(record_keys) == len(endpoint_keys)
    assert sorted(record_keys) == sorted(endpoint_keys)
    assert len(chunk.indices) // 3 == len(endpoint_keys)


def test_extract_records_uses_open_air_visibility_and_classification():
    voxels, ids = _single_building_voxels()
    records = extract_voxel_surface_records(voxels, ids, meshsize=2.0)

    planes = {record.plane for record in records}
    assert planes == {"+x", "-x", "+y", "-y", "+z", "-z"}

    selectable = [r for r in records if r.surface_kind in {"roof", "wall"}]
    assert len(selectable) == 5
    assert sum(r.surface_kind == "roof" for r in records) == 1
    assert sum(r.surface_kind == "wall" for r in records) == 4
    assert {r.orientation for r in records if r.surface_kind == "wall"} == {"N", "E", "S", "W"}


def test_extract_records_treats_tree_as_exposed_and_landcover_as_blocking():
    voxels = np.zeros((3, 4, 2), dtype=int)
    ids = np.zeros((3, 4), dtype=int)
    voxels[1, 1, 0] = -3
    ids[1, 1] = 9
    voxels[1, 2, 0] = -2  # +x neighbour, tree: exposed
    voxels[1, 0, 0] = 1   # -x neighbour, land cover: blocked

    records = extract_voxel_surface_records(voxels, ids, meshsize=1.0)
    planes = {record.plane for record in records}

    assert "+x" in planes
    assert "-x" not in planes


def test_select_records_for_whole_roof_walls_and_orientation():
    voxels, ids = _single_building_voxels()
    records = extract_voxel_surface_records(voxels, ids, meshsize=1.0)

    whole = select_records_for_zone(records, _zone("whole", [SurfaceSelector(building_id=7, mode="whole")]))
    roof = select_records_for_zone(records, _zone("roof", [SurfaceSelector(building_id=7, mode="roof")]))
    walls = select_records_for_zone(records, _zone("walls", [SurfaceSelector(building_id=7, mode="all_walls")]))
    east = select_records_for_zone(records, _zone("east", [SurfaceSelector(building_id=7, mode="wall_orientation", orientation="E")]))

    assert {record.surface_kind for record in whole} == {"roof", "wall"}
    assert [record.plane for record in roof] == ["+z"]
    assert {record.plane for record in walls} == {"+x", "-x", "+y", "-y"}
    assert [record.orientation for record in east] == ["E"]


def test_faces_and_exclude_faces_promote_triangle_keys_to_full_rectangle():
    voxels, ids = _single_building_voxels()
    records = extract_voxel_surface_records(voxels, ids, meshsize=1.0)
    roof_record = next(record for record in records if record.plane == "+z")

    selected = select_records_for_zone(records, _zone("faces", [
        SurfaceSelector(building_id=7, mode="faces", face_keys=[roof_record.face_keys[0]]),
    ]))
    assert selected == [roof_record]

    excluded = select_records_for_zone(records, _zone("exclude", [
        SurfaceSelector(building_id=7, mode="whole"),
        SurfaceSelector(building_id=7, mode="exclude_faces", face_keys=[roof_record.face_keys[1]]),
    ]))
    assert roof_record not in excluded


def test_filtered_extraction_preserves_face_keys_for_selected_building():
    voxels = np.zeros((2, 3, 1), dtype=int)
    ids = np.zeros((2, 3), dtype=int)
    voxels[0, 0, 0] = -3
    voxels[0, 2, 0] = -3
    ids[0, 0] = 4
    ids[0, 2] = 8

    full = extract_voxel_surface_records(voxels, ids, meshsize=1.0)
    filtered = extract_voxel_surface_records(voxels, ids, meshsize=1.0, building_ids={8})

    assert filtered
    assert {record.building_id for record in filtered} == {8}
    assert [record.face_keys for record in filtered] == [record.face_keys for record in full if record.building_id == 8]


def test_bulk_zone_edge_payload_extracts_only_selected_building_ids(monkeypatch):
    calls = []

    def fake_extract(voxels, building_id_grid, meshsize, *, building_ids=None, preserve_global_face_index=True):
        calls.append((building_ids, preserve_global_face_index))
        return []

    monkeypatch.setattr(surface_zone_edges_module, "extract_voxel_surface_records", fake_extract)

    voxels = np.zeros((2, 2, 1), dtype=int)
    ids = np.zeros((2, 2), dtype=int)
    build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("roof", [SurfaceSelector(building_id=4, mode="roof")]),
    ])

    assert calls == [({4}, False)]


def test_face_key_zone_edge_payload_preserves_global_face_indices(monkeypatch):
    calls = []

    def fake_extract(voxels, building_id_grid, meshsize, *, building_ids=None, preserve_global_face_index=True):
        calls.append((building_ids, preserve_global_face_index))
        return []

    monkeypatch.setattr(surface_zone_edges_module, "extract_voxel_surface_records", fake_extract)

    voxels = np.zeros((2, 2, 1), dtype=int)
    ids = np.zeros((2, 2), dtype=int)
    build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("faces", [SurfaceSelector(building_id=4, mode="faces", face_keys=["face-a"])]),
    ])

    assert calls == [({4}, True)]


def test_two_voxel_east_wall_merges_to_one_rectangle_with_four_edges():
    voxels = np.zeros((3, 3, 1), dtype=int)
    ids = np.zeros((3, 3), dtype=int)
    voxels[0:2, 1, 0] = -3
    ids[0:2, 1] = 4

    payloads = build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("east", [SurfaceSelector(building_id=4, mode="wall_orientation", orientation="E")]),
    ])

    assert len(payloads) == 1
    assert payloads[0].id == "east"
    assert len(payloads[0].segments) == 4
    assert all(is_axis_aligned_segment(segment) for segment in payloads[0].segments)

    expected = {
        normalize_segment_key((2, 0, 0, 2, 2, 0)),
        normalize_segment_key((2, 2, 0, 2, 2, 1)),
        normalize_segment_key((2, 2, 1, 2, 0, 1)),
        normalize_segment_key((2, 0, 1, 2, 0, 0)),
    }
    assert {normalize_segment_key(segment) for segment in payloads[0].segments} == expected


def test_greedy_merge_is_deterministic():
    voxels = np.zeros((3, 3, 1), dtype=int)
    ids = np.zeros((3, 3), dtype=int)
    voxels[0:2, 1, 0] = -3
    ids[0:2, 1] = 4
    zones = [_zone("east", [SurfaceSelector(building_id=4, mode="wall_orientation", orientation="E")])]

    first = build_surface_zone_edge_payloads(voxels, ids, 1.0, zones)[0].segments
    second = build_surface_zone_edge_payloads(voxels, ids, 1.0, zones)[0].segments

    assert first == second


def test_merge_does_not_cross_building_ids_or_zones():
    voxels = np.zeros((2, 2, 1), dtype=int)
    ids = np.zeros((2, 2), dtype=int)
    voxels[0, 1, 0] = -3
    voxels[1, 1, 0] = -3
    ids[0, 1] = 4
    ids[1, 1] = 5

    payloads = build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("z4", [SurfaceSelector(building_id=4, mode="wall_orientation", orientation="E")]),
        _zone("z5", [SurfaceSelector(building_id=5, mode="wall_orientation", orientation="E")]),
    ])

    assert [payload.id for payload in payloads] == ["z4", "z5"]
    assert [len(payload.segments) for payload in payloads] == [4, 4]


def test_axis_alignment_and_segment_normalization_drop_diagonals():
    assert is_axis_aligned_segment((0, 0, 0, 1, 0, 0))
    assert not is_axis_aligned_segment((0, 0, 0, 1, 1, 0))
    assert not is_axis_aligned_segment((0, 0, 0, 0, 0, 0))
    assert normalize_segment_key((1, 0, 0, 0, 0, 0)) == normalize_segment_key((0, 0, 0, 1, 0, 0))


def test_two_by_two_east_wall_returns_only_outer_boundary_four_edges():
    voxels = np.zeros((3, 3, 2), dtype=int)
    ids = np.zeros((3, 3), dtype=int)
    voxels[0:2, 1, 0:2] = -3
    ids[0:2, 1] = 7

    payloads = build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("east", [SurfaceSelector(building_id=7, mode="wall_orientation", orientation="E")]),
    ])

    assert len(payloads) == 1
    expected = {
        normalize_segment_key((2, 0, 0, 2, 2, 0)),
        normalize_segment_key((2, 2, 0, 2, 2, 2)),
        normalize_segment_key((2, 2, 2, 2, 0, 2)),
        normalize_segment_key((2, 0, 2, 2, 0, 0)),
    }
    assert {normalize_segment_key(segment) for segment in payloads[0].segments} == expected


def test_l_shaped_roof_returns_merged_outer_boundary_only():
    voxels = np.zeros((3, 3, 1), dtype=int)
    ids = np.zeros((3, 3), dtype=int)
    voxels[0, 0, 0] = -3
    voxels[0, 1, 0] = -3
    voxels[1, 0, 0] = -3
    ids[0, 0] = 11
    ids[0, 1] = 11
    ids[1, 0] = 11

    payloads = build_surface_zone_edge_payloads(voxels, ids, 1.0, [
        _zone("roof", [SurfaceSelector(building_id=11, mode="roof")]),
    ])

    assert len(payloads) == 1
    expected = {
        normalize_segment_key((0, 0, 1, 2, 0, 1)),
        normalize_segment_key((2, 0, 1, 2, 1, 1)),
        normalize_segment_key((2, 1, 1, 1, 1, 1)),
        normalize_segment_key((1, 1, 1, 1, 2, 1)),
        normalize_segment_key((1, 2, 1, 0, 2, 1)),
        normalize_segment_key((0, 2, 1, 0, 0, 1)),
    }
    assert {normalize_segment_key(segment) for segment in payloads[0].segments} == expected
