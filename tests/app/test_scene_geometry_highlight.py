"""Coordinate-frame regression tests for backend overlay/highlight chunks.

Phase 3 contract: ``voxcity_grid``, ``bid_grid`` and 2-D ``sim_grid`` arrays
are uv layout (axis 0 = u = north). All scene-buffer builders place voxel
``(u, v, k)`` at scene ``(u*ms, v*ms, k*ms)``. Stale ``np.flipud`` /
``ensure_orientation(NORTH_UP, SOUTH_UP)`` calls on these arrays misalign the
chunks with respect to ``build_voxel_buffers`` — visible N-S inversion in the
Three.js scene.
"""
import numpy as np
import pytest

from app.backend.scene_geometry import (
    build_building_highlight_buffers,
    build_ground_overlay_buffers,
    build_voxel_buffers,
)


def test_highlight_chunks_align_with_uv_layout_voxcity_grid():
    nx, ny, nz = 5, 4, 3
    meshsize = 1.0

    # Single building voxel at uv(u=3, v=1, k=1). Use u=3 so flipud lands at
    # u=N-1-3=1, an obviously different row.
    voxcity_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    voxcity_grid[3, 1, 1] = -3

    bid_grid = np.zeros((nx, ny), dtype=np.int32)
    bid_grid[3, 1] = 99

    chunks = build_building_highlight_buffers(
        voxcity_grid, bid_grid, [99], meshsize,
    )
    assert len(chunks) > 0, "uv-aligned inputs should produce highlight chunks"

    # All chunk vertex positions must sit inside the voxel cell at uv(3, 1, 1):
    # scene x in [3*ms, 4*ms], scene y in [1*ms, 2*ms], scene z in [1*ms, 2*ms].
    for chunk in chunks:
        positions = np.asarray(chunk.positions, dtype=float).reshape(-1, 3)
        assert positions.size > 0
        assert positions[:, 0].min() >= 3.0 * meshsize - 1e-9
        assert positions[:, 0].max() <= 4.0 * meshsize + 1e-9
        assert positions[:, 1].min() >= 1.0 * meshsize - 1e-9
        assert positions[:, 1].max() <= 2.0 * meshsize + 1e-9


def test_highlight_chunks_match_voxel_buffers_for_same_cell():
    """The +x face of a building voxel produced by ``build_building_highlight_buffers``
    must occupy the same scene rectangle as the +x face produced by
    ``build_voxel_buffers`` for that voxel — they describe the same cell."""
    from app.backend.scene_geometry import build_voxel_buffers

    nx, ny, nz = 5, 4, 3
    meshsize = 1.0

    voxcity_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    voxcity_grid[3, 1, 1] = -3
    bid_grid = np.zeros((nx, ny), dtype=np.int32)
    bid_grid[3, 1] = 99

    voxel_resp = build_voxel_buffers(voxcity_grid, meshsize)
    voxel_chunks = [
        c for c in voxel_resp.chunks if c.metadata.get("class") == -3
    ]
    assert voxel_chunks, "Expected at least one -3 voxel chunk"

    voxel_positions = np.concatenate(
        [np.asarray(c.positions, dtype=float).reshape(-1, 3) for c in voxel_chunks],
        axis=0,
    )

    highlight_chunks = build_building_highlight_buffers(
        voxcity_grid, bid_grid, [99], meshsize,
    )
    highlight_positions = np.concatenate(
        [np.asarray(c.positions, dtype=float).reshape(-1, 3) for c in highlight_chunks],
        axis=0,
    )

    # The highlight covers exactly the same voxel; bbox must match.
    np.testing.assert_allclose(
        highlight_positions.min(axis=0), voxel_positions.min(axis=0)
    )
    np.testing.assert_allclose(
        highlight_positions.max(axis=0), voxel_positions.max(axis=0)
    )


def test_ground_overlay_quad_aligns_with_uv_cell():
    """A single non-NaN sim value at uv(u=3, v=1) must produce a ground-overlay
    quad whose vertices sit inside scene rectangle x in [3*ms, 4*ms],
    y in [1*ms, 2*ms]. A stale flipud places it at scene x in [0, 1*ms].
    """
    nx, ny, nz = 4, 4, 2
    meshsize = 1.0
    sim = np.full((nx, ny), np.nan, dtype=float)
    sim[3, 1] = 5.0

    voxel_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    voxel_grid[:, :, 0] = 1
    voxel_grid[3, 1, 1] = -3

    resp = build_ground_overlay_buffers(
        sim, voxel_grid,
        meshsize=meshsize, view_point_height=1.5,
        sim_type="solar", colormap="viridis",
    )

    pos = np.asarray(resp.chunk.positions, dtype=float).reshape(-1, 3)
    assert pos.size > 0
    assert pos[:, 0].min() == pytest.approx(3.0 * meshsize)
    assert pos[:, 0].max() == pytest.approx(4.0 * meshsize)
    assert pos[:, 1].min() == pytest.approx(1.0 * meshsize)
    assert pos[:, 1].max() == pytest.approx(2.0 * meshsize)

    # face_to_cell entries point back to original uv (3, 1).
    for i, j in resp.face_to_cell:
        assert (i, j) == (3, 1)


def test_ground_overlay_aligns_with_voxel_buffer_for_same_uv_cell():
    """Ground-overlay quad and the +z face of the underlying ground voxel must
    occupy the same scene rectangle (modulo the small z-offset for visibility).
    """
    nx, ny, nz = 4, 4, 1
    meshsize = 2.0
    sim = np.full((nx, ny), np.nan, dtype=float)
    sim[2, 3] = 1.0

    voxel_grid = np.zeros((nx, ny, nz), dtype=np.int32)
    voxel_grid[:, :, 0] = 1            # ground everywhere

    ground_resp = build_ground_overlay_buffers(
        sim, voxel_grid, meshsize=meshsize,
        view_point_height=1.5, sim_type="view",
    )
    voxel_resp = build_voxel_buffers(voxel_grid, meshsize=meshsize)

    ground_pos = np.asarray(ground_resp.chunk.positions, dtype=float).reshape(-1, 3)

    # +z faces of the ground class for the (2, 3) cell only.
    plus_z = next(
        c for c in voxel_resp.chunks
        if c.metadata.get("class") == 1 and c.metadata.get("plane") == "+z"
    )
    voxel_pos_all = np.asarray(plus_z.positions, dtype=float).reshape(-1, 3)
    cell_pos = voxel_pos_all[
        (voxel_pos_all[:, 0] >= 2.0 * meshsize - 1e-6)
        & (voxel_pos_all[:, 0] <= 3.0 * meshsize + 1e-6)
        & (voxel_pos_all[:, 1] >= 3.0 * meshsize - 1e-6)
        & (voxel_pos_all[:, 1] <= 4.0 * meshsize + 1e-6)
    ]
    assert cell_pos.shape[0] >= 4

    np.testing.assert_allclose(ground_pos[:, 0].min(), cell_pos[:, 0].min())
    np.testing.assert_allclose(ground_pos[:, 0].max(), cell_pos[:, 0].max())
    np.testing.assert_allclose(ground_pos[:, 1].min(), cell_pos[:, 1].min())
    np.testing.assert_allclose(ground_pos[:, 1].max(), cell_pos[:, 1].max())
