"""Unit tests for app.backend.scene_geometry buffer builders."""
from __future__ import annotations

import numpy as np
import pytest

from app.backend.scene_geometry import (
    build_building_overlay_buffers,
    build_ground_overlay_buffers,
    build_voxel_buffers,
)


# ---- Voxel buffers ---------------------------------------------------------

def test_voxel_buffers_emits_chunks_for_each_class():
    """A simple grid with two land cover classes + one building voxel
    should emit ``MeshChunk`` entries for each class with non-empty buffers."""
    grid = np.zeros((4, 4, 3), dtype=np.int32)
    grid[:, :, 0] = 1            # ground
    grid[1:3, 1:3, 1] = -3       # 2x2 building cluster on layer 1

    resp = build_voxel_buffers(grid, meshsize=2.0)

    classes = {chunk.metadata["class"] for chunk in resp.chunks}
    assert 1 in classes and -3 in classes

    for chunk in resp.chunks:
        assert len(chunk.positions) > 0
        assert len(chunk.indices) > 0
        # positions are flat XYZ -> divisible by 3
        assert len(chunk.positions) % 3 == 0
        # indices are flat triangles -> divisible by 3
        assert len(chunk.indices) % 3 == 0
        # uniform color in [0,1]
        assert chunk.color is not None
        assert all(0.0 <= c <= 1.0 for c in chunk.color)

    # Bounding box should reflect the grid extent in metres.
    assert resp.meshsize_m == pytest.approx(2.0)
    assert resp.bbox_max == [8.0, 8.0, 6.0]
    assert resp.bbox_min == [0.0, 0.0, 0.0]


def test_voxel_buffers_topz_face_count_for_ground_layer():
    """A flat single-layer ground produces exactly one +z face per cell."""
    grid = np.zeros((3, 3, 1), dtype=np.int32)
    grid[:, :, 0] = 1

    resp = build_voxel_buffers(grid, meshsize=1.0)
    plus_z = [c for c in resp.chunks if c.metadata == {"class": 1, "plane": "+z"}]
    assert len(plus_z) == 1
    chunk = plus_z[0]
    # 9 cells * 4 verts/cell * 3 floats/vert = 108
    assert len(chunk.positions) == 108
    # 9 cells * 2 tris/cell * 3 indices/tri = 54
    assert len(chunk.indices) == 54


# ---- Ground overlay --------------------------------------------------------

def test_ground_overlay_face_count_matches_finite_cells():
    sim = np.full((4, 4), np.nan, dtype=float)
    sim[0, 0] = 1.0
    sim[1, 2] = 5.0
    sim[3, 3] = 9.0

    grid3d = np.zeros((4, 4, 2), dtype=np.int32)
    grid3d[:, :, 0] = 1

    resp = build_ground_overlay_buffers(
        sim,
        grid3d,
        meshsize=2.0,
        view_point_height=1.5,
        sim_type="solar",
        colormap="viridis",
    )

    # 3 valid cells -> 3 quads -> 6 tris -> 18 indices, 12 verts (36 floats)
    chunk = resp.chunk
    assert len(chunk.indices) == 18
    assert len(chunk.positions) == 36
    # Per-vertex colors -> 12 verts * 3 = 36 floats
    assert len(chunk.colors) == 36

    # face_to_cell length matches triangle count, one (i,j) per tri
    assert resp.face_to_cell is not None
    assert len(resp.face_to_cell) == 6
    for ij in resp.face_to_cell:
        assert len(ij) == 2
        i, j = ij
        assert 0 <= i < 4 and 0 <= j < 4

    # value range covers the supplied finite values
    assert resp.value_min == pytest.approx(1.0)
    assert resp.value_max == pytest.approx(9.0)


def test_ground_overlay_no_valid_cells_returns_empty_chunk():
    sim = np.full((3, 3), np.nan, dtype=float)
    grid3d = np.zeros((3, 3, 1), dtype=np.int32)

    resp = build_ground_overlay_buffers(
        sim,
        grid3d,
        meshsize=1.0,
        view_point_height=1.5,
        sim_type="view",
    )
    assert resp.chunk.positions == []
    assert resp.chunk.indices == []
    assert resp.face_to_cell == []


# ---- Building overlay ------------------------------------------------------

class _FakeMesh:
    def __init__(self, vertices, faces, metadata):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.metadata = metadata


def test_building_overlay_solar_face_values_to_per_vertex_colors():
    # Two triangles sharing an edge.
    verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 2, 3]]
    mesh = _FakeMesh(verts, faces, metadata={
        "global": [100.0, 200.0],
        "building_face_ids": [7, 7],
    })

    resp = build_building_overlay_buffers(
        mesh, sim_type="solar", colormap="viridis", unit_label="W/m^2"
    )

    chunk = resp.chunk
    # Triangle soup: 2 faces * 3 verts * 3 floats = 18
    assert len(chunk.positions) == 18
    assert len(chunk.indices) == 6
    assert len(chunk.colors) == 18
    assert resp.face_to_building == [7, 7]
    assert resp.value_min == pytest.approx(100.0)
    assert resp.value_max == pytest.approx(200.0)


def test_building_overlay_view_uses_view_factor_values():
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2]]
    mesh = _FakeMesh(verts, faces, metadata={
        "view_factor_values": [0.5],
    })

    resp = build_building_overlay_buffers(
        mesh, sim_type="view", colormap="viridis"
    )
    assert resp.target == "building"
    assert resp.sim_type == "view"
    # No building IDs -> face_to_building is None
    assert resp.face_to_building is None
    assert resp.value_min == pytest.approx(0.5)
