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


# ---- Scene-plane mapping correctness ----------------------------------------

def _tri_normal(positions_flat, indices_flat, tri_idx: int) -> np.ndarray:
    """Return the (unnormalised) normal for triangle ``tri_idx``."""
    i0, i1, i2 = (
        indices_flat[tri_idx * 3],
        indices_flat[tri_idx * 3 + 1],
        indices_flat[tri_idx * 3 + 2],
    )
    pts = positions_flat.reshape(-1, 3)
    v0, v1, v2 = pts[i0], pts[i1], pts[i2]
    return np.cross(v1 - v0, v2 - v0)


def test_voxel_side_planes_are_at_correct_scene_boundary():
    """Scene convention: X=east=v, Y=north=u.
    For a single voxel at (u=2, v=1, k=0) with meshsize=1:
      +x face  must sit at x = (v+1)*ms = 2.0
      -x face  must sit at x = v*ms     = 1.0
      +y face  must sit at y = (u+1)*ms = 3.0
      -y face  must sit at y = u*ms     = 2.0
    """
    ms = 1.0
    # non-square grid (nx=5, ny=4) so u and v produce distinct values
    grid = np.zeros((5, 4, 2), dtype=np.int32)
    grid[2, 1, 0] = -3          # isolated building voxel at u=2, v=1, k=0

    resp = build_voxel_buffers(grid, meshsize=ms)

    for plane, expected_coord, axis, label in [
        ("+x", (1 + 1) * ms, 0, "max x"),
        ("-x", 1 * ms,       0, "min x"),
        ("+y", (2 + 1) * ms, 1, "max y"),
        ("-y", 2 * ms,       1, "min y"),
    ]:
        chunks = [c for c in resp.chunks if c.metadata.get("plane") == plane]
        assert chunks, f"Expected a chunk for plane {plane}"
        pos = np.asarray(chunks[0].positions, dtype=float).reshape(-1, 3)
        if "max" in label:
            actual = pos[:, axis].max()
        else:
            actual = pos[:, axis].min()
        assert actual == pytest.approx(expected_coord), (
            f"Plane {plane}: {label} = {actual}, expected {expected_coord}"
        )


def test_voxel_side_plane_normals_are_outward():
    """Each side-plane chunk must have a first-triangle normal pointing outward.
    +x -> positive x component, -x -> negative x, +y -> positive y, -y -> negative y,
    +z -> positive z, -z -> negative z.
    """
    ms = 1.0
    grid = np.zeros((5, 4, 3), dtype=np.int32)
    grid[2, 1, 1] = -3          # isolated building voxel at u=2, v=1, k=1

    resp = build_voxel_buffers(grid, meshsize=ms)

    sign_expectations = {
        "+x": (0, +1), "-x": (0, -1),
        "+y": (1, +1), "-y": (1, -1),
        "+z": (2, +1), "-z": (2, -1),
    }
    for plane, (axis, sign) in sign_expectations.items():
        chunks = [c for c in resp.chunks if c.metadata.get("plane") == plane]
        assert chunks, f"Expected a chunk for plane {plane}"
        pos = np.asarray(chunks[0].positions, dtype=float)
        idx = np.asarray(chunks[0].indices, dtype=np.int32)
        normal = _tri_normal(pos, idx, 0)
        assert (normal[axis] * sign) > 0, (
            f"Plane {plane}: normal[{axis}]={normal[axis]:.4f}, expected sign {sign:+d}"
        )


def test_adjacent_voxels_along_u_axis_emit_correct_side_planes():
    """Two voxels at (u=1, v=2) and (u=2, v=2) are adjacent along the u/north axis.
    The shared face is a +y boundary of the first and -y boundary of the second.
    Their outer end faces must both be in the y-direction (not x-direction).
    """
    ms = 1.0
    grid = np.zeros((5, 5, 2), dtype=np.int32)
    grid[1, 2, 0] = -3
    grid[2, 2, 0] = -3

    resp = build_voxel_buffers(grid, meshsize=ms)

    # Outer y-boundary faces (the two free ends along north axis):
    # u=1 voxel's -y face at y=1.0, u=2 voxel's +y face at y=3.0.
    py_chunks = [c for c in resp.chunks if c.metadata.get("class") == -3
                 and c.metadata.get("plane") == "+y"]
    ny_chunks = [c for c in resp.chunks if c.metadata.get("class") == -3
                 and c.metadata.get("plane") == "-y"]
    assert py_chunks, "Expected +y chunk for adjacent u-axis voxels"
    assert ny_chunks, "Expected -y chunk for adjacent u-axis voxels"

    # Each end chunk should have exactly 1 quad (4 verts, 2 tris → 6 indices)
    # because the shared face is culled.
    py_pos = np.asarray(py_chunks[0].positions, dtype=float).reshape(-1, 3)
    ny_pos = np.asarray(ny_chunks[0].positions, dtype=float).reshape(-1, 3)
    assert py_pos.shape[0] == 4, f"+y chunk has {py_pos.shape[0]} verts, expected 4"
    assert ny_pos.shape[0] == 4, f"-y chunk has {ny_pos.shape[0]} verts, expected 4"

    # +y face of (u=2) voxel must sit at y=3.0.
    assert py_pos[:, 1].max() == pytest.approx(3.0 * ms)
    # -y face of (u=1) voxel must sit at y=1.0.
    assert ny_pos[:, 1].min() == pytest.approx(1.0 * ms)
