"""Public entry point: add buildings from an OBJ file to a VoxCity model."""
from __future__ import annotations

import copy
import os

import numpy as np

from ..utils.logging import get_logger
from .units import validate_units
from .transform import build_placement_transform, grid_geom_from_voxcity
from .loader import load_obj_groups, select_building_groups
from .voxelize import voxelize_mesh
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

    Returns a new VoxCity object (the input is not mutated).
    See docs/rhino_obj_import.md for the Rhino export guide and conventions.
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
        raise NotImplementedError(
            "meshlib backend voxelization is not implemented yet; use backend='trimesh'."
        )

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
        occ = voxelize_mesh(mesh, M, grid_shape)
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
