"""Public entry point: add buildings from an OBJ file to a VoxCity model."""
from __future__ import annotations

import copy
import os

import numpy as np

from ..utils.logging import get_logger
from .units import validate_units
from .transform import build_placement_transform, grid_geom_from_voxcity
from .loader import load_obj_groups, select_building_groups
from .voxelize import voxelize_mesh, voxelize_mesh_meshlib
from .integrate import stamp_buildings

_logger = get_logger(__name__)


def add_buildings_from_obj(
    voxcity,
    obj_path,
    anchor_lonlat,
    anchor_elevation,
    anchor_model_point=(0.0, 0.0, 0.0),
    rotation=0.0,
    move=(0.0, 0.0, 0.0),
    units="m",
    roles=None,
    backend="trimesh",
    z_up=True,
    swap_yz=False,
    overwrite=True,
    gridvis=False,
):
    """Voxelize buildings from an OBJ file and stamp them into a VoxCity model.

    Loads geometry groups from *obj_path*, selects the groups whose resolved
    role is ``"building"``, places them into the VoxCity domain using an
    anchor lon/lat + elevation (plus optional rotation/translation/unit
    scaling), voxelizes each building group, and stamps the resulting cells
    into a copy of *voxcity*. The input *voxcity* is never mutated; this
    function always returns a new object. See docs/rhino_obj_import.md for
    the Rhino export guide and conventions.

    Args:
        voxcity: source VoxCity object. Read for its grid geometry
            (``extras['rectangle_vertices']``), voxel shape, DEM, and
            existing building grids; never mutated. A deep copy is made
            internally and returned (with buildings stamped in), or
            returned unmodified (still a deep copy) if no in-domain
            geometry is found.
        obj_path: path to the OBJ file to import. Must exist on disk;
            checked before any other validation.
        anchor_lonlat: ``(lon, lat)`` pair giving the geographic position
            that ``anchor_model_point`` is placed at. Must have exactly 2
            elements.
        anchor_elevation: elevation (metres) of ``anchor_model_point`` in
            the real world. Combined with the VoxCity DEM minimum to fix
            the vertical (Z) placement of the model.
        anchor_model_point: ``(x, y, z)`` point in OBJ model coordinates
            (in *units*, pre-scale) that is pinned to ``anchor_lonlat`` /
            ``anchor_elevation``. Defaults to the model origin
            ``(0.0, 0.0, 0.0)``.
        rotation: rotation in degrees applied to the model in the
            horizontal (east/north) plane before placement. Positive values
            rotate the model counter-clockwise; e.g. at ``rotation=90``,
            model +X ends up pointing north. Defaults to ``0.0`` (no
            rotation).
        move: ``(east, north, up)`` offset in metres applied after
            anchoring, e.g. for fine-tuning placement without changing the
            anchor. Defaults to ``(0.0, 0.0, 0.0)`` (no offset).
        units: model length unit, used to scale OBJ coordinates to metres.
            One of ``"m"``, ``"cm"``, ``"mm"``, ``"ft"``, ``"in"``
            (case-insensitive). Defaults to ``"m"``.
        roles: optional ``{group_name: role}`` mapping used to mark some
            OBJ groups as non-building (e.g. ``{"windows": "window"}``) so
            they are excluded from voxelization. Matching is exact-string
            only. Group names absent from this mapping (including all
            names, when ``roles=None``, the default) are treated as role
            ``"building"`` and voxelized.
        backend: voxelization backend. ``"trimesh"`` (default) uses
            :func:`~voxcity.importer.voxelize.voxelize_mesh` (column z-ray
            casting). ``"meshlib"`` uses the optional
            :func:`~voxcity.importer.voxelize.voxelize_mesh_meshlib` SDF
            backend, which requires the optional ``meshlib`` package to be
            installed (see Raises below); this backend is best-effort and
            not empirically verified against a real MeshLib install (see
            that function's docstring for details).
        z_up: whether the OBJ's vertical axis is already Z-up (Rhino's
            convention), the default (``True``). When ``False``, the
            loaded mesh is treated as Y-up and axes 1/2 (Y/Z) are swapped
            before placement, equivalent to setting ``swap_yz=True``.
        swap_yz: if ``True``, force an explicit Y/Z axis swap on the
            loaded geometry regardless of ``z_up``. Defaults to ``False``.
            The effective swap applied is ``swap_yz or (not z_up)``.
        overwrite: if ``True`` (default), newly stamped building voxels
            overwrite any existing non-empty voxel at that cell (and the
            collision is counted/logged). If ``False``, only cells that are
            currently empty are stamped; already-occupied cells are left
            untouched.
        gridvis: if ``True``, after stamping, display a quick-look 2D
            visualization of the post-import building height grid. Purely
            a debugging aid; failures in the visualization step (including
            a missing/broken plotting backend) are swallowed silently.
            Defaults to ``False``.

    Returns:
        A new VoxCity object with the imported buildings stamped in (or an
        unmodified deep copy of *voxcity* if no building-role geometry was
        found, or none of it voxelized to cells inside the domain). The
        *voxcity* argument passed in is never mutated.

    Raises:
        FileNotFoundError: if *obj_path* does not exist.
        ValueError: if *units* is not one of the recognized unit strings,
            if *backend* is not ``"trimesh"`` or ``"meshlib"``, or if
            *anchor_lonlat* does not have exactly 2 elements.
        ImportError: if ``backend="meshlib"`` is requested but the optional
            ``meshlib`` package is not installed.
    """
    # --- validation (fail fast) ---
    if not os.path.exists(os.fspath(obj_path)):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    validate_units(units)
    if backend not in ("trimesh", "meshlib"):
        raise ValueError(f"Unknown backend {backend!r}; expected 'trimesh' or 'meshlib'.")
    if len(anchor_lonlat) != 2:
        raise ValueError("anchor_lonlat must be (lon, lat).")

    if backend == "meshlib":
        try:
            import meshlib  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='meshlib' requires the optional 'meshlib' package "
                "(non-commercial license). Install it or use backend='trimesh'."
            ) from e

    _voxelize = voxelize_mesh_meshlib if backend == "meshlib" else voxelize_mesh

    apply_swap = swap_yz or (not z_up)

    # --- load + role routing ---
    groups = load_obj_groups(obj_path, swap_yz=apply_swap)
    building_groups = select_building_groups(groups, roles=roles)
    if not building_groups:
        _logger.warning("No building-role geometry found in %s; nothing imported.", obj_path)
        return copy.deepcopy(voxcity)

    # --- transform + voxelize ---
    out = copy.deepcopy(voxcity)
    M = build_placement_transform(
        out, anchor_lonlat=anchor_lonlat, anchor_elevation=anchor_elevation,
        anchor_model_point=anchor_model_point, rotation=rotation, move=move, units=units,
    )
    grid_shape = out.voxels.classes.shape

    occupied_by_name = {}
    for name, mesh in building_groups:
        occ = _voxelize(mesh, M, grid_shape)
        if len(occ):
            occupied_by_name[name] = occ

    if not occupied_by_name:
        _logger.warning(
            "Imported geometry voxelized to 0 cells inside the domain. Check "
            "anchor_lonlat/anchor_elevation/rotation/move/units."
        )
        return out

    # --- stamp + metadata ---
    out = stamp_buildings(
        out, occupied_by_name, overwrite=overwrite,
        source=os.fspath(obj_path),
        manifest_extra={
            "anchor_lonlat": list(anchor_lonlat),
            "anchor_elevation": float(anchor_elevation),
            "anchor_model_point": list(anchor_model_point),
            "rotation": float(rotation), "move": list(move),
            "units": units, "backend": backend,
        },
    )

    if gridvis:
        try:
            from ..visualizer.grids import visualize_numerical_grid
            h = out.buildings.heights.copy()
            h[h == 0] = np.nan
            visualize_numerical_grid(h, float(out.voxels.meta.meshsize),
                                     "building height (m) after import", cmap="viridis", label="Value")
        except Exception:
            pass

    return out
