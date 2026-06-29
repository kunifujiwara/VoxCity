"""Stamp imported window geometry as a glass skin (code -16) on building voxels.

Window groups are surface-voxelized (not volume-filled) so thin/planar panes
rasterize reliably, then the building (-3) cells they coincide with -- within a
small Chebyshev radius -- are recolored to the glass code. Windows never create
new occupancy; they only reclassify existing building cells, so building
footprint/height metadata is unaffected.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from trimesh.voxel import creation as _vox_creation

from ..utils.logging import get_logger

_logger = get_logger(__name__)

BUILDING_CODE = -3
GLASS_CODE = -16


def _surface_cells(mesh, transform, grid_shape):
    """Return unique in-bounds (i, j, k) cells the mesh surface passes through.

    The mesh is mapped into voxel-index space by *transform*, surface-voxelized
    at unit pitch, and each occupied cell center (``VoxelGrid.points``, which is
    absolute -- unlike ``sparse_indices``) is floored to an index. Cells outside
    *grid_shape* are dropped.
    """
    nx, ny, nz = grid_shape
    m = mesh.copy()
    m.apply_transform(np.asarray(transform, dtype=float))
    if len(m.faces) == 0 or len(m.vertices) == 0:
        return np.empty((0, 3), dtype=np.int64)

    vg = _vox_creation.voxelize_subdivide(m, pitch=1.0)
    pts = np.asarray(vg.points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    ijk = np.floor(pts).astype(np.int64)
    in_bounds = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
    )
    ijk = ijk[in_bounds]
    if ijk.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(ijk, axis=0)


def _surface_normal_axis(cells):
    """Axis (0/1/2) of least spatial variance for a planar window's cell cloud.

    A window pane is flat along its normal, so the surface cells barely vary on
    that axis. Returns the least-variance axis, or ``None`` when the cloud is too
    degenerate to define a plane (the second-smallest variance is ~0, i.e. the
    cells form a line/point rather than a sheet) so the caller can fall back to
    isotropic matching.
    """
    if cells.shape[0] < 3:
        return None
    var = np.var(cells.astype(np.float64), axis=0)
    order = np.argsort(var)  # ascending; order[0] = least-variance (normal) axis
    if var[order[1]] < 1e-9:
        return None
    return int(order[0])


def _axis_line_structure(axis):
    """3x3x3 boolean structuring element: a 3-cell line along *axis* only.

    Dilating with this bridges the sub-voxel depth gap between a pane and the
    wall along the window normal, without growing the window laterally in the
    facade plane (which an isotropic structure would).
    """
    struct = np.zeros((3, 3, 3), dtype=bool)
    idx = [1, 1, 1]
    for d in (-1, 0, 1):
        idx[axis] = 1 + d
        struct[idx[0], idx[1], idx[2]] = True
    return struct


def stamp_windows(
    voxcity,
    window_groups,
    transform,
    *,
    window_value=GLASS_CODE,
    building_value=BUILDING_CODE,
    skin_radius=1,
):
    """Recolor facade building cells touched by window meshes to *window_value*.

    Args:
        voxcity: VoxCity object; ``voxels.classes`` is modified in place.
        window_groups: list of ``(name, trimesh.Trimesh)`` window groups.
        transform: 4x4 affine mapping model coords -> voxel-index space (the
            same matrix used to voxelize the buildings).
        window_value: code written for window cells (default -16, glass).
        building_value: code identifying building cells eligible for recolor.
        skin_radius: depth-direction radius (in voxels) for matching window
            surface cells to nearby building cells. ``1`` absorbs sub-voxel
            offsets between a pane plane and the wall surface. Dilation is along
            each window's normal axis only, so it does not inflate the window in
            the facade plane.

    Returns:
        int: number of building cells recolored to *window_value*.
    """
    classes = voxcity.voxels.classes
    grid_shape = classes.shape
    building_mask = classes == building_value

    cells_list = [
        _surface_cells(mesh, transform, grid_shape) for _name, mesh in window_groups
    ]
    cells_list = [c for c in cells_list if len(c)]
    if not cells_list:
        return 0

    # Window cells of all groups (for the unmatched-cell log below).
    win_cells = np.unique(np.concatenate(cells_list, axis=0), axis=0)
    win_mask = np.zeros(grid_shape, dtype=bool)
    win_mask[win_cells[:, 0], win_cells[:, 1], win_cells[:, 2]] = True

    iso_structure = ndimage.generate_binary_structure(3, 3)

    if skin_radius > 0:
        # Bucket each group's cells by its normal axis, then dilate each bucket
        # along that axis ONLY. This bridges the sub-voxel pane<->wall depth gap
        # without inflating the window in the facade plane (an isotropic
        # dilation grows it ~1 voxel per side, ~doubling small windows).
        # Groups whose plane is undefined fall into an isotropic bucket.
        axis_masks = {0: None, 1: None, 2: None}
        iso_mask = None
        for cells in cells_list:
            m = np.zeros(grid_shape, dtype=bool)
            m[cells[:, 0], cells[:, 1], cells[:, 2]] = True
            axis = _surface_normal_axis(cells)
            if axis is None:
                iso_mask = m if iso_mask is None else (iso_mask | m)
            else:
                axis_masks[axis] = m if axis_masks[axis] is None else (axis_masks[axis] | m)

        win_dilated = np.zeros(grid_shape, dtype=bool)
        for axis, m in axis_masks.items():
            if m is None:
                continue
            win_dilated |= ndimage.binary_dilation(
                m, structure=_axis_line_structure(axis), iterations=skin_radius
            )
        if iso_mask is not None:
            win_dilated |= ndimage.binary_dilation(
                iso_mask, structure=iso_structure, iterations=skin_radius
            )

        # Independent isotropic building dilation answers the distinct question
        # "which window cells are near a building cell" (used only for the
        # unmatched-cell count/log, so a generous radius is fine here).
        bld_dilated = ndimage.binary_dilation(
            building_mask, structure=iso_structure, iterations=skin_radius
        )
    else:
        win_dilated = win_mask
        bld_dilated = building_mask

    recolor = building_mask & win_dilated
    n = int(recolor.sum())
    if n:
        classes[recolor] = window_value

    n_unmatched = int(win_mask.sum() - (win_mask & bld_dilated).sum())
    if n_unmatched:
        _logger.info(
            "stamp_windows: %d window surface cell(s) had no building cell "
            "within radius %d; skipped (no floating glass).",
            n_unmatched,
            skin_radius,
        )
    return n
