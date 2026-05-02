"""Pure builders that turn VoxCity arrays into raw Three.js BufferGeometry payloads.

These helpers replace the Plotly-figure-based transport used by the original
``visualize_voxcity_plotly``/``_build_sim_overlay_traces`` path. They emit the
flat ``positions``/``indices``/``colors`` arrays consumed by the React-Three-Fiber
viewer in ``app/frontend/src/three``.

Functions
---------
build_voxel_buffers
    Static city geometry: one ``MeshChunk`` per (class, plane) of exposed voxel faces.
build_ground_overlay_buffers
    Coloured ground simulation surface (one quad per non-NaN cell) with
    ``face_to_cell`` pick-back metadata.
build_building_overlay_buffers
    Coloured per-face building mesh with ``face_to_building`` pick-back metadata.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.cm as mcm
import matplotlib.colors as mcolors

from voxcity.visualizer.palette import get_voxel_color_map

from .models import MeshChunk, OverlayGeometryResponse, SceneGeometryResponse


# ---------------------------------------------------------------------------
# Voxel face extraction (refactored from visualizer.renderer.add_faces)
# ---------------------------------------------------------------------------

_PLANE_OFFSETS = {
    # (vx0,vx1) selectors, (vy0,vy1), (vz0,vz1) for each of the 4 quad corners
    "+x": ("x1x1x1x1", "y0y1y1y0", "z0z0z1z1"),
    "-x": ("x0x0x0x0", "y0y1y1y0", "z1z1z0z0"),
    "+y": ("x0x1x1x0", "y1y1y1y1", "z0z0z1z1"),
    "-y": ("x0x1x1x0", "y0y0y0y0", "z1z1z0z0"),
    "+z": ("x0x1x1x0", "y0y0y1y1", "z1z1z1z1"),
    "-z": ("x0x1x1x0", "y1y1y0y0", "z0z0z0z0"),
}


def _exposed_face_masks(occ: np.ndarray, occ_any: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return ``(posx, negx, posy, negy, posz, negz)`` boolean masks of voxel
    faces that touch empty space (any-class occluder)."""
    p = np.pad(occ_any, ((0, 1), (0, 0), (0, 0)), constant_values=False)
    posx = occ & (~p[1:, :, :])
    p = np.pad(occ_any, ((1, 0), (0, 0), (0, 0)), constant_values=False)
    negx = occ & (~p[:-1, :, :])
    p = np.pad(occ_any, ((0, 0), (0, 1), (0, 0)), constant_values=False)
    posy = occ & (~p[:, 1:, :])
    p = np.pad(occ_any, ((0, 0), (1, 0), (0, 0)), constant_values=False)
    negy = occ & (~p[:, :-1, :])
    p = np.pad(occ_any, ((0, 0), (0, 0), (0, 1)), constant_values=False)
    posz = occ & (~p[:, :, 1:])
    p = np.pad(occ_any, ((0, 0), (0, 0), (1, 0)), constant_values=False)
    negz = occ & (~p[:, :, :-1])
    return posx, negx, posy, negy, posz, negz


def _voxel_face_arrays(
    mask: np.ndarray,
    plane: str,
    meshsize: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions, indices, cell_idx)`` for one face direction.

    Each True cell in ``mask`` becomes a quad (4 vertices, 2 triangles).
    ``cell_idx`` is ``int32[(n,3)]`` of the source ``(i,j,k)`` voxel coords.
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32),
        )

    xi, yi, zi = idx[:, 0], idx[:, 1], idx[:, 2]
    xc = x[xi]
    yc = y[yi]
    zc = z[zi]
    x0, x1 = xc - dx / 2.0, xc + dx / 2.0
    y0, y1 = yc - dy / 2.0, yc + dy / 2.0
    z0, z1 = zc - dz / 2.0, zc + dz / 2.0

    if plane == "+x":
        vx = np.stack([x1, x1, x1, x1], axis=1)
        vy = np.stack([y0, y1, y1, y0], axis=1)
        vz = np.stack([z0, z0, z1, z1], axis=1)
    elif plane == "-x":
        vx = np.stack([x0, x0, x0, x0], axis=1)
        vy = np.stack([y0, y1, y1, y0], axis=1)
        vz = np.stack([z1, z1, z0, z0], axis=1)
    elif plane == "+y":
        vx = np.stack([x0, x1, x1, x0], axis=1)
        vy = np.stack([y1, y1, y1, y1], axis=1)
        vz = np.stack([z0, z0, z1, z1], axis=1)
    elif plane == "-y":
        vx = np.stack([x0, x1, x1, x0], axis=1)
        vy = np.stack([y0, y0, y0, y0], axis=1)
        vz = np.stack([z1, z1, z0, z0], axis=1)
    elif plane == "+z":
        vx = np.stack([x0, x1, x1, x0], axis=1)
        vy = np.stack([y0, y0, y1, y1], axis=1)
        vz = np.stack([z1, z1, z1, z1], axis=1)
    elif plane == "-z":
        vx = np.stack([x0, x1, x1, x0], axis=1)
        vy = np.stack([y1, y1, y0, y0], axis=1)
        vz = np.stack([z0, z0, z0, z0], axis=1)
    else:
        raise ValueError(f"Unknown plane: {plane}")

    positions = np.column_stack(
        [vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)]
    ).astype(np.float32, copy=False)

    n = idx.shape[0]
    starts = np.arange(0, 4 * n, 4, dtype=np.int32)
    tris = np.concatenate(
        [
            np.stack([starts, starts + 1, starts + 2], axis=1),
            np.stack([starts, starts + 2, starts + 3], axis=1),
        ],
        axis=0,
    )
    indices = tris.reshape(-1).astype(np.int32, copy=False)

    return positions.reshape(-1), indices, idx.astype(np.int32)


def _surface_aware_downsample(orig: np.ndarray, stride: int) -> np.ndarray:
    """Stride X/Y, pick topmost non-zero along Z within each window."""
    nx0, ny0, nz0 = orig.shape
    xs = orig[::stride, ::stride, :]
    nx_ds, ny_ds, _ = xs.shape
    nz_ds = int(np.ceil(nz0 / float(stride)))
    out = np.zeros((nx_ds, ny_ds, nz_ds), dtype=orig.dtype)
    for k in range(nz_ds):
        z0w = k * stride
        z1w = min(z0w + stride, nz0)
        W = xs[:, :, z0w:z1w]
        if W.size == 0:
            continue
        nz_mask = W != 0
        has_any = nz_mask.any(axis=2)
        rev_mask = nz_mask[:, :, ::-1]
        idx_rev = rev_mask.argmax(axis=2)
        real_idx = (W.shape[2] - 1) - idx_rev
        gathered = np.take_along_axis(W, real_idx[..., None], axis=2).squeeze(-1)
        out[:, :, k] = np.where(has_any, gathered, 0)
    return out


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_voxel_buffers(
    voxcity_grid: np.ndarray,
    meshsize: float,
    *,
    downsample: int = 1,
    classes: Optional[Sequence[int]] = None,
    color_scheme: str = "default",
    opacity: float = 1.0,
) -> SceneGeometryResponse:
    """Convert a 3D voxel class grid into one ``MeshChunk`` per (class, plane).

    Parameters
    ----------
    voxcity_grid : ``int`` array, shape ``(nx, ny, nz)``.
        ``0`` = empty, ``-3`` = building, ``-2`` = tree, ``>=1`` = land cover.
    meshsize : metres per voxel.
    downsample : surface-aware stride (``1`` = no downsample).
    classes : restrict to these class ids; default = all non-zero.
    color_scheme : passed to :func:`voxcity.visualizer.palette.get_voxel_color_map`.
    """
    if voxcity_grid is None or getattr(voxcity_grid, "ndim", 0) != 3:
        raise ValueError("voxcity_grid must be a 3D ndarray")

    stride = max(1, int(downsample))
    vox = (
        _surface_aware_downsample(voxcity_grid, stride)
        if stride > 1
        else voxcity_grid
    )

    nx, ny, nz = vox.shape
    dx = meshsize * stride
    dy = meshsize * stride
    dz = meshsize * stride
    x = np.arange(nx, dtype=float) * dx + dx / 2.0
    y = np.arange(ny, dtype=float) * dy + dy / 2.0
    z = np.arange(nz, dtype=float) * dz + dz / 2.0

    if classes is None:
        classes_list = [int(c) for c in np.unique(vox[vox != 0]).tolist()]
    else:
        classes_list = [int(c) for c in classes]

    palette = get_voxel_color_map(color_scheme)
    occluder = vox != 0

    chunks: List[MeshChunk] = []
    for cls in classes_list:
        if not np.any(vox == cls):
            continue
        occ = vox == cls
        masks = _exposed_face_masks(occ, occluder)
        rgb = palette.get(cls, [128, 128, 128])
        color01 = [c / 255.0 for c in rgb[:3]]

        for mask, plane in zip(masks, ("+x", "-x", "+y", "-y", "+z", "-z")):
            if not mask.any():
                continue
            positions, indices, _cell_idx = _voxel_face_arrays(
                mask, plane, meshsize, x, y, z, dx, dy, dz
            )
            if indices.size == 0:
                continue
            chunks.append(
                MeshChunk(
                    name=f"class{cls}{plane}",
                    positions=positions.tolist(),
                    indices=indices.tolist(),
                    color=color01,
                    opacity=float(opacity),
                    flat_shading=False,
                    metadata={"class": int(cls), "plane": plane},
                )
            )

    bbox_min = [0.0, 0.0, 0.0]
    bbox_max = [float(nx * dx), float(ny * dy), float(nz * dz)]

    # Topmost ground level: max k-index of any land-cover voxel (class >= 1)
    # converted to metres. Buildings (-3) and trees (-2) sit ON the ground so
    # we exclude them; if no land-cover is present we fall back to 0.
    ground_mask = vox >= 1
    if ground_mask.any():
        ks = np.where(ground_mask.any(axis=(0, 1)))[0]
        ground_top_m = float((ks.max() + 1) * dz)
    else:
        ground_top_m = 0.0

    return SceneGeometryResponse(
        chunks=chunks,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        meshsize_m=float(meshsize),
        ground_top_m=ground_top_m,
    )


def build_building_highlight_buffers(
    voxcity_grid: np.ndarray,
    bid_grid_aligned: np.ndarray,
    building_ids: Sequence[int],
    meshsize: float,
    *,
    color_rgb: Sequence[float] = (0.81, 0.96, 0.15),  # #CFF527, matches legacy
    opacity: float = 0.85,
    colormap: Optional[str] = None,
    emissive: bool = False,
) -> List[MeshChunk]:
    """Return one ``MeshChunk`` per face plane covering only the voxels of the
    given ``building_ids``.

    ``bid_grid_aligned`` must already be in the same orientation as
    ``voxcity_grid`` (i.e. ``ensure_orientation(bid_grid, NORTH_UP, SOUTH_UP)``).

    If ``colormap`` is given, the highlight colour is taken from the maximum
    value of that colormap (so it visually matches the top of a sim overlay).
    Set ``emissive=True`` to tag the chunks for self-illuminated rendering on
    the frontend.
    """
    if voxcity_grid is None or voxcity_grid.ndim != 3:
        return []
    if bid_grid_aligned is None or bid_grid_aligned.ndim != 2:
        return []
    ids = [int(b) for b in building_ids if int(b) != 0]
    if not ids:
        return []

    # Building voxels for the requested IDs.
    bid_mask_2d = np.isin(bid_grid_aligned, np.asarray(ids, dtype=bid_grid_aligned.dtype))
    if not bid_mask_2d.any():
        return []
    occ = (voxcity_grid == -3) & bid_mask_2d[:, :, None]
    if not occ.any():
        return []

    # Highlight voxels are exposed against *all* non-empty voxels in the city,
    # so faces hidden inside neighbouring buildings/trees aren't emitted.
    occluder = voxcity_grid != 0
    masks = _exposed_face_masks(occ, occluder)

    nx, ny, nz = voxcity_grid.shape
    dx = dy = dz = float(meshsize)
    x = np.arange(nx, dtype=float) * dx + dx / 2.0
    y = np.arange(ny, dtype=float) * dy + dy / 2.0
    z = np.arange(nz, dtype=float) * dz + dz / 2.0

    if colormap:
        try:
            rgba_max = mcm.get_cmap(colormap)(1.0)
            color01 = [float(c) for c in rgba_max[:3]]
        except Exception:
            color01 = [float(c) for c in color_rgb[:3]]
    else:
        color01 = [float(c) for c in color_rgb[:3]]

    chunks: List[MeshChunk] = []
    for mask, plane in zip(masks, ("+x", "-x", "+y", "-y", "+z", "-z")):
        if not mask.any():
            continue
        positions, indices, _ = _voxel_face_arrays(
            mask, plane, meshsize, x, y, z, dx, dy, dz
        )
        if indices.size == 0:
            continue
        chunks.append(
            MeshChunk(
                name=f"highlight{plane}",
                positions=positions.tolist(),
                indices=indices.tolist(),
                color=color01,
                opacity=float(opacity),
                flat_shading=False,
                metadata={
                    "highlight": True,
                    "plane": plane,
                    "emissive": bool(emissive),
                },
            )
        )
    return chunks


def _derive_dem_norm(voxcity_grid: np.ndarray, meshsize: float, ref_shape) -> np.ndarray:
    """Return ground-level elevation in metres, SOUTH_UP (row 0 = south).

    voxcity_grid is NORTH_UP; np.flipud converts to SOUTH_UP to match the
    row iteration order used by build_ground_overlay_buffers (which also
    flips sim_grid to SOUTH_UP before building quads).
    """
    if voxcity_grid is None or voxcity_grid.ndim != 3:
        return np.zeros(ref_shape, dtype=float)
    lc_mask = voxcity_grid >= 1
    k_indices = np.arange(voxcity_grid.shape[2])
    masked_k = np.where(lc_mask, k_indices[None, None, :], -1)
    k_top = np.max(masked_k, axis=2)
    k_top = np.maximum(k_top, 0)
    return np.flipud(k_top.astype(float) * float(meshsize))


def build_ground_overlay_buffers(
    sim_grid: np.ndarray,
    voxcity_grid: np.ndarray,
    meshsize: float,
    view_point_height: float,
    *,
    sim_type: str,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    unit_label: str = "",
) -> OverlayGeometryResponse:
    """Build a coloured ground-surface overlay (one quad per non-NaN cell).

    Returns per-vertex colours (4 verts per quad, all the same colour) plus a
    ``face_to_cell`` array of length ``= triangle_count`` mapping each triangle
    back to its source ``(i, j)`` cell in the *original* (north-up) grid.
    """
    sim = np.asarray(sim_grid, dtype=float)
    if sim.ndim != 2:
        raise ValueError("sim_grid must be 2-D")

    # Match create_sim_surface_mesh: flip N-up -> S-up so iteration order
    # corresponds to the on-screen mesh layout.
    from voxcity.utils.orientation import (
        ORIENTATION_NORTH_UP,
        ORIENTATION_SOUTH_UP,
        ensure_orientation,
    )

    sim_flipped = ensure_orientation(sim, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    dem_flipped = _derive_dem_norm(voxcity_grid, meshsize, sim.shape)

    finite = np.isfinite(sim_flipped) & ~np.isnan(sim_flipped)
    if not np.any(finite):
        empty = MeshChunk(
            name="ground_overlay",
            positions=[],
            indices=[],
            colors=[],
            opacity=0.95,
        )
        return OverlayGeometryResponse(
            target="ground",
            sim_type=sim_type,
            chunk=empty,
            face_to_cell=[],
            value_min=0.0,
            value_max=1.0,
            colormap=colormap,
            unit_label=unit_label,
        )

    finite_vals = sim_flipped[finite]
    vmin_f = float(np.nanmin(finite_vals)) if vmin is None else float(vmin)
    vmax_f = float(np.nanmax(finite_vals)) if vmax is None else float(vmax)
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-9

    z_off = float(meshsize) + max(float(view_point_height), float(meshsize))

    nrows, ncols = sim_flipped.shape
    cell_x, cell_y = np.where(finite)  # row=x in flipped frame
    n_cells = cell_x.size

    # z per cell, matching create_sim_surface_mesh formula
    z_base = (
        meshsize * (dem_flipped[cell_x, cell_y] / meshsize + 1.5).astype(int)
        + z_off
        - meshsize
    )

    fx0 = cell_x.astype(np.float32) * meshsize
    fx1 = (cell_x + 1).astype(np.float32) * meshsize
    fy0 = cell_y.astype(np.float32) * meshsize
    fy1 = (cell_y + 1).astype(np.float32) * meshsize
    z32 = z_base.astype(np.float32)

    # 4 vertices per cell: v0 v1 v2 v3
    vx = np.stack([fx0, fx1, fx1, fx0], axis=1)
    vy = np.stack([fy0, fy0, fy1, fy1], axis=1)
    vz = np.stack([z32, z32, z32, z32], axis=1)
    positions = np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])

    starts = np.arange(0, 4 * n_cells, 4, dtype=np.int32)
    tris = np.concatenate(
        [
            np.stack([starts, starts + 1, starts + 2], axis=1),
            np.stack([starts, starts + 2, starts + 3], axis=1),
        ],
        axis=0,
    )
    # Order: [tri1_of_cell0, tri1_of_cell1, ..., tri2_of_cell0, tri2_of_cell1, ...]
    indices = tris.reshape(-1).astype(np.int32)

    # Per-vertex colors
    norm = mcolors.Normalize(vmin=vmin_f, vmax=vmax_f)
    cmap = mcm.get_cmap(colormap)
    rgba = cmap(norm(sim_flipped[cell_x, cell_y]))  # (n_cells, 4)
    rgb = rgba[:, :3].astype(np.float32)
    # Replicate per cell to 4 vertices
    per_vert_colors = np.repeat(rgb, 4, axis=0).reshape(-1)

    # Map each triangle back to source (i_orig, j_orig) in the *original*
    # north-up grid. The orientation flip swaps row 0 <-> row (nrows-1).
    i_orig = (nrows - 1 - cell_x).astype(int)
    j_orig = cell_y.astype(int)
    cell_pairs = np.stack([i_orig, j_orig], axis=1)  # (n_cells, 2)
    # Triangle order is [first-tri-of-each-cell, second-tri-of-each-cell].
    face_to_cell = np.concatenate([cell_pairs, cell_pairs], axis=0)

    chunk = MeshChunk(
        name="ground_overlay",
        positions=positions.reshape(-1).tolist(),
        indices=indices.tolist(),
        colors=per_vert_colors.tolist(),
        opacity=0.95,
        flat_shading=False,
        metadata={"sim_type": sim_type},
    )
    return OverlayGeometryResponse(
        target="ground",
        sim_type=sim_type,
        chunk=chunk,
        face_to_cell=face_to_cell.tolist(),
        face_to_building=None,
        value_min=vmin_f,
        value_max=vmax_f,
        colormap=colormap,
        unit_label=unit_label,
    )


def build_building_overlay_buffers(
    sim_mesh,
    *,
    sim_type: str,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    unit_label: str = "",
    nan_color: str = "gray",
    zero_as_nan: bool = False,
) -> OverlayGeometryResponse:
    """Build a coloured per-face overlay from a trimesh-like building sim mesh.

    The mesh must expose ``vertices`` (V, 3), ``faces`` (F, 3), and
    ``metadata`` containing either ``"global"`` (solar) or
    ``"view_factor_values"`` (view/landmark) face- or vertex-aligned values.
    """
    if sim_mesh is None or getattr(sim_mesh, "vertices", None) is None:
        raise ValueError("sim_mesh missing vertices")

    Vb = np.asarray(sim_mesh.vertices, dtype=np.float32)
    Fb = np.asarray(sim_mesh.faces, dtype=np.int32)
    n_faces = Fb.shape[0]

    value_name = "global" if sim_type == "solar" else "view_factor_values"
    values = None
    if hasattr(sim_mesh, "metadata") and isinstance(sim_mesh.metadata, dict):
        raw = sim_mesh.metadata.get(value_name)
        if raw is not None:
            values = np.asarray(raw, dtype=float)

    face_vals: Optional[np.ndarray] = None
    if values is not None:
        if len(values) == n_faces:
            face_vals = values
        elif len(values) == len(Vb):
            face_vals = np.nanmean(values[Fb], axis=1)

    if face_vals is None:
        face_vals = np.zeros(n_faces, dtype=float)

    if zero_as_nan:
        face_vals = np.where(face_vals == 0, np.nan, face_vals)

    finite = np.isfinite(face_vals)
    if np.any(finite):
        finite_vals = face_vals[finite]
        vmin_f = float(np.nanmin(finite_vals)) if vmin is None else float(vmin)
        vmax_f = float(np.nanmax(finite_vals)) if vmax is None else float(vmax)
    else:
        vmin_f = 0.0 if vmin is None else float(vmin)
        vmax_f = 1.0 if vmax is None else float(vmax)
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-9

    norm = mcolors.Normalize(vmin=vmin_f, vmax=vmax_f)
    cmap = mcm.get_cmap(colormap)
    nan_rgba = np.array(mcolors.to_rgba(nan_color), dtype=float)

    face_rgba = np.zeros((n_faces, 4), dtype=float)
    face_rgba[finite] = cmap(norm(face_vals[finite]))
    face_rgba[~finite] = nan_rgba
    face_rgb = face_rgba[:, :3].astype(np.float32)

    # Triangle soup: replicate each face's 3 vertices and assign per-vertex colors.
    tri_positions = Vb[Fb].reshape(-1, 3)  # (n_faces*3, 3)
    per_vert_colors = np.repeat(face_rgb, 3, axis=0)  # (n_faces*3, 3)
    indices = np.arange(tri_positions.shape[0], dtype=np.int32)

    # Per-face building IDs (if metadata supplies them)
    face_to_building: Optional[List[int]] = None
    if hasattr(sim_mesh, "metadata") and isinstance(sim_mesh.metadata, dict):
        bids = sim_mesh.metadata.get("building_face_ids")
        if bids is not None:
            arr = np.asarray(bids).reshape(-1)
            if arr.size == n_faces:
                face_to_building = [int(x) for x in arr.tolist()]

    chunk = MeshChunk(
        name="building_overlay",
        positions=tri_positions.reshape(-1).tolist(),
        indices=indices.tolist(),
        colors=per_vert_colors.reshape(-1).tolist(),
        opacity=1.0,
        flat_shading=False,
        metadata={"sim_type": sim_type, "value_name": value_name},
    )
    return OverlayGeometryResponse(
        target="building",
        sim_type=sim_type,
        chunk=chunk,
        face_to_cell=None,
        face_to_building=face_to_building,
        value_min=vmin_f,
        value_max=vmax_f,
        colormap=colormap,
        unit_label=unit_label,
    )
