import numpy as np
import pytest

from app.backend.models import SurfaceSelector, ZoneSpec
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
