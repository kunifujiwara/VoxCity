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

from .models import MeshChunk, OverlayGeometryResponse, SceneGeometryResponse, SurfaceFaceMeta


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
    """Return ``(posu, negu, posv, negv, posz, negz)`` boolean masks of voxel
    faces that touch empty space.

    Array layout: axis 0 = u = north, axis 1 = v = east.
    ``posu`` / ``negu`` = cells exposed at their +/- north boundary.
    ``posv`` / ``negv`` = cells exposed at their +/- east boundary.
    ``posz`` / ``negz`` = cells exposed at their top / bottom boundary.
    """
    p = np.pad(occ_any, ((0, 1), (0, 0), (0, 0)), constant_values=False)
    posu = occ & (~p[1:, :, :])
    p = np.pad(occ_any, ((1, 0), (0, 0), (0, 0)), constant_values=False)
    negu = occ & (~p[:-1, :, :])
    p = np.pad(occ_any, ((0, 0), (0, 1), (0, 0)), constant_values=False)
    posv = occ & (~p[:, 1:, :])
    p = np.pad(occ_any, ((0, 0), (1, 0), (0, 0)), constant_values=False)
    negv = occ & (~p[:, :-1, :])
    p = np.pad(occ_any, ((0, 0), (0, 0), (0, 1)), constant_values=False)
    posz = occ & (~p[:, :, 1:])
    p = np.pad(occ_any, ((0, 0), (0, 0), (1, 0)), constant_values=False)
    negz = occ & (~p[:, :, :-1])
    return posu, negu, posv, negv, posz, negz


def _scene_plane_masks(
    occ: np.ndarray, occluder: np.ndarray
) -> List[Tuple[np.ndarray, str]]:
    """Return ``(mask, plane_name)`` pairs ordered by scene plane.

    Scene convention: X = east = axis 1 (v), Y = north = axis 0 (u).
    Matches the legacy renderer which maps:
      posv / negv  →  '+x' / '-x'  (east-exposed cells = scene +X face)
      posu / negu  →  '+y' / '-y'  (north-exposed cells = scene +Y face)
    """
    posu, negu, posv, negv, posz, negz = _exposed_face_masks(occ, occluder)
    return [
        (posv, "+x"),
        (negv, "-x"),
        (posu, "+y"),
        (negu, "-y"),
        (posz, "+z"),
        (negz, "-z"),
    ]


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
    # Scene convention: X = east = axis 1 (v), Y = north = axis 0 (u).
    xc = x[yi]  # x array is sized for east (ny), indexed by v-axis (yi)
    yc = y[xi]  # y array is sized for north (nx), indexed by u-axis (xi)
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
        # North face: y = y1 (constant). Vertex order gives outward +y normal.
        vx = np.stack([x1, x0, x0, x1], axis=1)
        vy = np.stack([y1, y1, y1, y1], axis=1)
        vz = np.stack([z0, z0, z1, z1], axis=1)
    elif plane == "-y":
        # South face: y = y0 (constant). Vertex order gives outward -y normal.
        vx = np.stack([x0, x1, x1, x0], axis=1)
        vy = np.stack([y0, y0, y0, y0], axis=1)
        vz = np.stack([z0, z0, z1, z1], axis=1)
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
    # Scene convention: X = east = axis 1 (v, size ny), Y = north = axis 0 (u, size nx).
    x = np.arange(ny, dtype=float) * dy + dy / 2.0
    y = np.arange(nx, dtype=float) * dx + dx / 2.0
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
        rgb = palette.get(cls, [128, 128, 128])
        color01 = [c / 255.0 for c in rgb[:3]]

        for mask, plane in _scene_plane_masks(occ, occluder):
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
    # X = east (ny * dy), Y = north (nx * dx), Z = up (nz * dz).
    bbox_max = [float(ny * dy), float(nx * dx), float(nz * dz)]

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
    ``bid_grid_aligned`` must be uv layout (axis 0 = u = north), the same as
    ``voxcity_grid``, so the two arrays index 1:1 by ``(i, j)``.

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

    nx, ny, nz = voxcity_grid.shape
    dx = dy = dz = float(meshsize)
    # Scene convention: X = east = axis 1 (v, size ny), Y = north = axis 0 (u, size nx).
    x = np.arange(ny, dtype=float) * dy + dy / 2.0
    y = np.arange(nx, dtype=float) * dx + dx / 2.0
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
    for mask, plane in _scene_plane_masks(occ, occluder):
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


def _derive_dem_norm(voxcity_grid: np.ndarray, meshsize: float, ref_shape,
                     include_building_roofs: bool = False) -> np.ndarray:
    """Return surface elevation in metres, in uv layout (axis 0 = u = north).

    When include_building_roofs=False: height of the first contiguous solid run
    from the ground (pilotis → open floor, normal building → roof, terrain → top).
    When include_building_roofs=True: height of the topmost solid surface, so the
    overlay sits on the roof for all elevated structures.
    """
    if voxcity_grid is None or voxcity_grid.ndim != 3:
        return np.zeros(ref_shape, dtype=float)
    if include_building_roofs:
        return _topmost_solid_top(voxcity_grid, meshsize)
    return _first_contiguous_solid_top(voxcity_grid, meshsize)


def _topmost_solid_top(voxcity_grid: np.ndarray, meshsize: float) -> np.ndarray:
    """Per-cell height (m) of the topmost air-above-solid surface.

    Scans each column top-to-bottom; returns the k-index of the highest voxel
    that is solid and has air/tree directly above it.  Skips water (7/8/9) and
    non-building negatives.  Mirrors the top-to-bottom observer scan used when
    include_building_roofs=True: pilotis and elevated masses yield the roof.
    """
    solid = (voxcity_grid == -1) | (voxcity_grid >= 1) | (voxcity_grid == -3)
    water = np.isin(voxcity_grid, [7, 8, 9])
    bad_neg = (voxcity_grid < 0) & (voxcity_grid != -3) & (voxcity_grid != -1)
    invalid_surface = water | bad_neg
    air = ~solid
    nz = solid.shape[2]
    nx, ny = voxcity_grid.shape[:2]
    top_k = np.zeros((nx, ny), dtype=np.int32)
    for z in range(nz - 1, 0, -1):
        above_is_air = air[:, :, z]
        below_is_solid = solid[:, :, z - 1]
        below_is_valid = ~invalid_surface[:, :, z - 1]
        new_top = above_is_air & below_is_solid & below_is_valid
        # Only assign cells not yet assigned (top_k==0 and not already at floor)
        unset = (top_k == 0) & new_top
        top_k = np.where(unset, z - 1, top_k)
    return top_k.astype(float) * float(meshsize)


def _first_contiguous_solid_top(voxcity_grid: np.ndarray, meshsize: float) -> np.ndarray:
    """Per-cell height (m) of the top of the first contiguous solid run from k=0.

    Solid = ground (-1), land cover (>=1) or building (-3).  Scanning each column
    upward, the run ends at the first air voxel above a solid cell; the returned
    height is that run's top voxel index * meshsize.  Mirrors the simulator's
    observer search and the solar topo kernel so the overlay is always drawn at
    the surface the simulation was computed on.
    """
    solid = (voxcity_grid == -1) | (voxcity_grid >= 1) | (voxcity_grid == -3)
    nz = solid.shape[2]
    air = ~solid
    has_air = air.any(axis=2)
    first_air = np.argmax(air, axis=2)            # first air k (0 if column all solid)
    first_air = np.where(has_air, first_air, nz)  # all-solid column -> top is nz
    top_k = np.maximum(first_air - 1, 0)
    return top_k.astype(float) * float(meshsize)


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
    include_building_roofs: bool = False,
) -> OverlayGeometryResponse:
    """Build a coloured ground-surface overlay (one quad per non-NaN cell).

    Returns per-vertex colours (4 verts per quad, all the same colour) plus a
    ``face_to_cell`` array of length ``= triangle_count`` mapping each triangle
    back to its source ``(i, j)`` cell in the uv_m/SOUTH_UP grid (Phase 3).
    """
    sim = np.asarray(sim_grid, dtype=float)
    if sim.ndim != 2:
        raise ValueError("sim_grid must be 2-D")

    # Both sim_grid and voxcity_grid arrive in uv layout (Phase 3); the scene
    # mesh places voxel (u, v) at scene (u*ms, v*ms), so iterate (u, v) directly.
    sim_uv = sim
    dem_uv = _derive_dem_norm(voxcity_grid, meshsize, sim.shape, include_building_roofs)

    finite = np.isfinite(sim_uv) & ~np.isnan(sim_uv)
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

    finite_vals = sim_uv[finite]
    vmin_f = float(np.nanmin(finite_vals)) if vmin is None else float(vmin)
    vmax_f = float(np.nanmax(finite_vals)) if vmax is None else float(vmax)
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-9

    z_off = float(meshsize) + max(float(view_point_height), float(meshsize))

    cell_u, cell_v = np.where(finite)  # uv-axis indices
    n_cells = cell_u.size

    # z per cell, matching create_sim_surface_mesh formula
    z_base = (
        meshsize * (dem_uv[cell_u, cell_v] / meshsize + 1.5).astype(int)
        + z_off
        - meshsize
    )

    # Scene convention: X = east = v (axis 1), Y = north = u (axis 0).
    fx0 = cell_v.astype(np.float32) * meshsize
    fx1 = (cell_v + 1).astype(np.float32) * meshsize
    fy0 = cell_u.astype(np.float32) * meshsize
    fy1 = (cell_u + 1).astype(np.float32) * meshsize
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
    rgba = cmap(norm(sim_uv[cell_u, cell_v]))  # (n_cells, 4)
    rgb = rgba[:, :3].astype(np.float32)
    # Replicate per cell to 4 vertices
    per_vert_colors = np.repeat(rgb, 4, axis=0).reshape(-1)

    # face_to_cell maps each triangle back to source uv (u, v).
    cell_pairs = np.stack([cell_u.astype(int), cell_v.astype(int)], axis=1)
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

    # Per-face building IDs (if metadata supplies them).
    # Accept both 'building_face_ids' (Plotly-renderer path) and 'building_id'
    # (create_voxel_mesh() / geoprocessor path); prefer the former for compat.
    face_to_building: Optional[List[int]] = None
    if hasattr(sim_mesh, "metadata") and isinstance(sim_mesh.metadata, dict):
        bids = sim_mesh.metadata.get("building_face_ids") or sim_mesh.metadata.get("building_id")
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


def build_surface_selection_buffers(mesh: object) -> tuple:
    """Build a triangle-soup MeshChunk from a selectable building mesh.

    Returns (MeshChunk, list[SurfaceFaceMeta]) where the face_to_surface list
    has exactly one entry per triangle in the chunk.
    """
    from .surface_zones import classify_surface_faces

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    meta = classify_surface_faces(mesh)

    # Build non-indexed triangle soup: each triangle has 3 unique vertices
    positions = vertices[faces].reshape(-1, 3).astype(np.float32, copy=False)
    indices = np.arange(len(faces) * 3, dtype=np.int32)

    chunk = MeshChunk(
        name="building_surfaces",
        positions=positions.reshape(-1).tolist(),
        indices=indices.tolist(),
        color=[1.0, 1.0, 1.0],
        opacity=0.02,
        flat_shading=False,
        metadata={"kind": "building_surfaces"},
    )
    return chunk, meta
