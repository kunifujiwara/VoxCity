"""Load OBJ geometry groups and route them to import roles (e.g. building vs.
window/non-building) before voxelization.

An OBJ file may bundle multiple named objects/groups in a single export
(e.g. a Rhino layer-per-group export). ``load_obj_groups`` recovers those as
separate named meshes; ``classify_roles``/``select_building_groups`` let a
caller mark some groups as non-building (windows, context, site furniture,
...) so only the actual building geometry gets voxelized.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh

from ..utils.logging import get_logger

_logger = get_logger(__name__)

_FALLBACK_GROUP_NAME = "imported_building_1"


def _yz_swap_matrix() -> np.ndarray:
    """Return the 4x4 affine that swaps axes 1 (Y) and 2 (Z)."""
    swap = np.eye(4)
    swap[[1, 2]] = swap[[2, 1]]
    return swap


def load_obj_groups(obj_path, swap_yz: bool = False) -> List[Tuple[str, "trimesh.Trimesh"]]:
    """Load *obj_path* and return its geometry as a list of (name, mesh).

    Args:
        obj_path: path to an OBJ file.
        swap_yz: if True, return copies of each mesh with axes 1 and 2
            (Y/Z) swapped, to reconcile Rhino's Z-up convention with
            Y-up OBJ exporters. The originally loaded meshes are never
            mutated.

    Returns:
        List of ``(name, mesh)`` tuples, in the order the geometry was
        discovered. trimesh collapses an OBJ to a single, unnamed
        ``Trimesh`` (rather than a ``Scene``) whenever the file resolves
        to only one geometry group — this includes both OBJs with no
        named scene structure at all *and* OBJs containing exactly one
        named ``o <name>`` block, since trimesh discards that name in
        the single-group case. In either case the single group returned
        here is named ``"imported_building_1"``.

    Raises:
        FileNotFoundError: if *obj_path* does not exist or is not a file
            (e.g. a directory path).
        ValueError: if the file loads but contains no usable mesh geometry.
    """
    path_str = str(obj_path)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"OBJ file not found: {path_str}")

    # Note: trimesh's OBJ loader kwarg is `split_objects` (plural); passing
    # it ensures distinct `o <name>` groups in the OBJ load back as separate
    # named geometries in a Scene rather than being merged into one mesh.
    loaded = trimesh.load(path_str, process=False, split_objects=True, group_material=False)

    if isinstance(loaded, trimesh.Scene):
        groups = [(name, mesh) for name, mesh in loaded.geometry.items()]
    elif isinstance(loaded, trimesh.Trimesh):
        groups = [(_FALLBACK_GROUP_NAME, loaded)]
    else:
        groups = []

    groups = [(name, mesh) for name, mesh in groups if isinstance(mesh, trimesh.Trimesh)]

    if not groups:
        raise ValueError(f"OBJ file contains no usable mesh geometry: {path_str}")

    if not swap_yz:
        return groups

    swap = _yz_swap_matrix()
    swapped = []
    for name, mesh in groups:
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(swap)
        swapped.append((name, mesh_copy))
    return swapped


def classify_roles(names: Iterable[str], roles: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Map each group name to its import role, defaulting to ``"building"``.

    Matching against *roles* is exact string match only (no glob/regex
    pattern support). A typo or an intended pattern in a ``roles`` key
    that doesn't exactly equal a group name is silently ignored, and that
    group defaults to role ``"building"``.

    Args:
        names: iterable of group name strings.
        roles: optional mapping of name -> role string. Names absent from
            this mapping default to role ``"building"``.

    Returns:
        Dict mapping each name to its resolved role string.
    """
    roles = roles or {}
    return {name: roles.get(name, "building") for name in names}


def select_building_groups(
    groups: List[Tuple[str, "trimesh.Trimesh"]],
    roles: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, "trimesh.Trimesh"]]:
    """Filter *groups* down to those classified with role ``"building"``.

    Groups with a non-building role (e.g. ``"window"``) are skipped and
    logged at INFO level.

    Args:
        groups: list of ``(name, mesh)`` tuples, e.g. from ``load_obj_groups``.
        roles: optional mapping of name -> role string, passed to
            ``classify_roles``.

    Returns:
        List of ``(name, mesh)`` tuples whose role resolved to ``"building"``.
    """
    names = [name for name, _mesh in groups]
    resolved = classify_roles(names, roles=roles)

    selected = []
    for name, mesh in groups:
        role = resolved[name]
        if role == "building":
            selected.append((name, mesh))
        else:
            _logger.info("Skipping group '%s' (role=%s)", name, role)
    return selected
