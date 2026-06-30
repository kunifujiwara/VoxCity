"""Stamp imported window geometry as a glass skin (code -16) on building voxels.

Window groups are surface-voxelized (not volume-filled) so thin/planar panes
rasterize reliably. Each physically-distinct window (a connected component of the
surface cells) has its opening filled in the facade plane -- so a mullioned frame
becomes a solid pane rather than thin bars. Each filled footprint cell is then
snapped to its nearest building EXTERIOR-SHELL voxel (within ``skin_radius``),
recoloring that surface voxel to glass. Snapping to the shell keeps the glass on
the building's outer surface, exactly one voxel deep, at the window's true
footprint -- with no lateral halo and no sinking into thick walls -- and it
follows facades at any angle to the grid (a rotated wall voxelizes to a
staircase; the nearest-shell snap tracks that staircase instead of an
axis-aligned guess). A window that sits farther than ``skin_radius`` from any
wall snaps to nothing and is dropped (no floating glass). Windows never create
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


def _component_fill_cells(comp):
    """Cells of one window component with its facade-plane opening filled.

    For a planar component (a well-defined normal axis), the opening is filled in
    the plane perpendicular to the normal -- so a mullioned frame becomes a solid
    pane -- and re-expanded across the component's thin normal span. A degenerate
    (near-1D) component has no plane to fill and is returned as-is.

    Returns an ``(n, 3)`` int array of cell indices.
    """
    cells = np.argwhere(comp)
    axis = _surface_normal_axis(cells)
    if axis is None:
        return cells
    proj = comp.any(axis=axis)
    filled2d = ndimage.binary_fill_holes(proj)
    slab = _broadcast_along_axis(
        filled2d, axis,
        int(cells[:, axis].min()), int(cells[:, axis].max()), comp.shape,
    )
    return np.argwhere(slab)


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
        skin_radius: how far (in voxels) a window may sit from the wall and still
            snap to it. Each filled footprint cell is mapped to its nearest
            building exterior-shell voxel if that voxel is within
            ``skin_radius`` (measured as a full diagonal, ``skin_radius*sqrt(3)``)
            -- absorbing sub-voxel offsets between a pane plane and the wall. The
            recolored glass is always one voxel deep (the shell) regardless of
            this value; windows farther than the radius snap to nothing.

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
        # Snap each window's filled footprint onto the building's exterior shell.
        # The shell is the 1-voxel-thick outer surface; mapping every filled cell
        # to its nearest shell voxel keeps glass on the surface (1 deep, true
        # footprint, no halo) and follows the wall at any angle -- a rotated wall
        # voxelizes to a staircase that an axis-aligned guess would miss.
        shell = building_mask & ~ndimage.binary_erosion(building_mask)
        recolor = np.zeros(grid_shape, dtype=bool)
        if shell.any():
            # Nearest shell voxel (and its index) for every grid cell.
            dist, nearest = ndimage.distance_transform_edt(
                ~shell, return_indices=True
            )
            # A window cell offset from the wall by up to skin_radius voxels in
            # each axis (a full diagonal) still snaps; farther windows do not.
            max_snap = skin_radius * np.sqrt(3.0) + 1e-6
            labels, n_lab = ndimage.label(win_mask, structure=iso_structure)
            for lab in range(1, n_lab + 1):
                fill_cells = _component_fill_cells(labels == lab)
                for ci, cj, ck in fill_cells:
                    if dist[ci, cj, ck] <= max_snap:
                        recolor[
                            nearest[0, ci, cj, ck],
                            nearest[1, ci, cj, ck],
                            nearest[2, ci, cj, ck],
                        ] = True

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
