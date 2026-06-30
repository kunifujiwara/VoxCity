"""Stamp imported window geometry as a glass skin (code -16) on building voxels.

Window groups are surface-voxelized (not volume-filled) so thin/planar panes
rasterize reliably. Each physically-distinct window (a connected component of the
surface cells) has its opening filled in the facade plane -- so a mullioned frame
becomes a solid pane rather than thin bars. The glass is then stamped on the
building's OUTWARD exterior skin only: the single most-outward building (-3)
voxel in each filled footprint column (1 voxel deep), so glass never sinks into a
thick wall. The outward direction is inferred from which side of the wall has
more exposed air. A proximity gate keeps a window that sits far from any wall
from recoloring a distant face. Windows never create new occupancy; they only
reclassify existing building cells, so building footprint/height metadata is
unaffected.
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


def _broadcast_along_axis(filled2d, axis, lo, hi, shape):
    """Place a filled 2D in-plane footprint back into the 3D grid across a
    component's (thin) extent along *axis* (indices ``lo..hi`` inclusive).

    ``filled2d`` has the grid shape with *axis* removed; it is re-inserted and
    broadcast across the ``lo..hi`` slab.
    """
    slab = np.zeros(shape, dtype=bool)
    idx = [slice(None), slice(None), slice(None)]
    idx[axis] = slice(lo, hi + 1)
    slab[tuple(idx)] = np.expand_dims(filled2d, axis)
    return slab


def _outward_direction(comp, building_mask, axis, window_cells):
    """Return +1 or -1: which way along *axis* faces 'outside' for this window.

    Primary signal: of the per-column most-+axis and most--axis building voxels
    under the component footprint, whichever side has more air (non-building)
    neighbors along *axis* is outward. Tie-break by sign(mean window - mean
    building) along *axis*; final fallback +1.
    """
    n_axis = building_mask.shape[axis]
    proj = comp.any(axis=axis)
    perp = [d for d in (0, 1, 2) if d != axis]

    pos_air = 0
    neg_air = 0
    for col in np.argwhere(proj):
        idx = [slice(None), slice(None), slice(None)]
        idx[perp[0]] = col[0]
        idx[perp[1]] = col[1]
        line = building_mask[tuple(idx)]
        occ = np.flatnonzero(line)
        if occ.size == 0:
            continue
        hi = int(occ.max())
        lo = int(occ.min())
        if hi + 1 >= n_axis or not line[hi + 1]:
            pos_air += 1
        if lo - 1 < 0 or not line[lo - 1]:
            neg_air += 1

    if pos_air != neg_air:
        return 1 if pos_air > neg_air else -1

    # Assumes windows sit on the building's OUTER facade; deeply-inset or
    # inner-courtyard panes within skin_radius of the outer face may pick the
    # outer voxel.
    bld = np.argwhere(building_mask)
    if window_cells.size and bld.size:
        if window_cells[:, axis].mean() >= bld[:, axis].mean():
            return 1
        return -1
    return 1


def _exterior_skin_cells(comp, building_mask, axis, direction):
    """Boolean grid: the single most-*direction* building voxel in each footprint
    column of *comp* (1 voxel deep, the outward skin).

    *direction* is +1 or -1 along *axis*. A column with no building voxel
    contributes nothing.
    """
    out = np.zeros(building_mask.shape, dtype=bool)
    proj = comp.any(axis=axis)
    perp = [d for d in (0, 1, 2) if d != axis]
    for col in np.argwhere(proj):
        idx = [slice(None), slice(None), slice(None)]
        idx[perp[0]] = col[0]
        idx[perp[1]] = col[1]
        line = building_mask[tuple(idx)]
        occ = np.flatnonzero(line)
        if occ.size == 0:
            continue
        a = int(occ.max()) if direction > 0 else int(occ.min())
        cell = [0, 0, 0]
        cell[perp[0]] = int(col[0])
        cell[perp[1]] = int(col[1])
        cell[axis] = a
        out[cell[0], cell[1], cell[2]] = True
    return out


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
        skin_radius: proximity gate only -- the glass skin is always 1 voxel
            deep regardless of this value. It controls how far (in voxels, along
            the window normal) a window may sit from the wall and still recolor
            it, absorbing sub-voxel offsets between a pane plane and the wall.

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

    if skin_radius <= 0:
        recolor = building_mask & win_mask
    else:
        # Each physically-distinct window is a connected component of the surface
        # cells. Fill its opening in the facade plane (mullion frame -> solid
        # pane), then recolor ONLY the outward 1-voxel skin under that footprint,
        # so glass never sinks into a thick wall.
        recolor = np.zeros(grid_shape, dtype=bool)
        labels, n_lab = ndimage.label(win_mask, structure=iso_structure)
        for lab in range(1, n_lab + 1):
            comp = labels == lab
            cells = np.argwhere(comp)
            axis = _surface_normal_axis(cells)
            if axis is None:
                # Degenerate (near-1D) cloud: no plane to fill. Isotropic bridge,
                # but clamp to the building EXTERIOR SHELL so it cannot go deep.
                bridged = ndimage.binary_dilation(
                    comp, structure=iso_structure, iterations=skin_radius
                )
                shell = building_mask & ~ndimage.binary_erosion(building_mask)
                recolor |= shell & bridged
                continue
            # Fill the opening in the facade plane, re-expand across the thin
            # normal span, then keep only the outward skin column-by-column.
            proj = comp.any(axis=axis)
            filled2d = ndimage.binary_fill_holes(proj)
            slab = _broadcast_along_axis(
                filled2d, axis,
                int(cells[:, axis].min()), int(cells[:, axis].max()), grid_shape,
            )
            direction = _outward_direction(slab, building_mask, axis, cells)
            # Restrict candidate wall voxels to the window's own facade: only
            # building cells within skin_radius of the window plane along the
            # normal. Gating BEFORE the outermost-per-column pick prevents
            # selecting a DIFFERENT building's face that lies along the same
            # column line (which would then be dropped, losing the window), and
            # also drops windows far from any wall.
            near = ndimage.binary_dilation(
                slab, structure=_axis_line_structure(axis), iterations=skin_radius
            )
            recolor |= _exterior_skin_cells(slab, building_mask & near, axis, direction)

    recolor &= building_mask
    n = int(recolor.sum())
    if n:
        classes[recolor] = window_value

    # Independent isotropic building dilation answers the distinct question
    # "which window cells are near a building cell" (used only for the
    # unmatched-cell count/log below, so a generous radius is fine here).
    if skin_radius > 0:
        bld_dilated = ndimage.binary_dilation(
            building_mask, structure=iso_structure, iterations=skin_radius
        )
    else:
        bld_dilated = building_mask

    n_unmatched = int(win_mask.sum() - (win_mask & bld_dilated).sum())
    if n_unmatched:
        _logger.info(
            "stamp_windows: %d window surface cell(s) had no building cell "
            "within radius %d; skipped (no floating glass).",
            n_unmatched,
            skin_radius,
        )
    return n
