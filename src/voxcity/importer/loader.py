"""Load OBJ geometry groups and route them to import roles (e.g. building vs.
window/non-building) before voxelization.

An OBJ file may bundle multiple named objects/groups in a single export
(e.g. a Rhino layer-per-group export). ``load_obj_groups`` recovers those as
separate named meshes; ``classify_roles``/``select_building_groups`` let a
caller mark some groups as non-building (windows, context, site furniture,
...) so only the actual building geometry gets voxelized.
"""
from __future__ import annotations

import io
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh

from ..utils.logging import get_logger

_logger = get_logger(__name__)

_FALLBACK_GROUP_NAME = "imported_building_1"

DEFAULT_WINDOW_KEYWORDS = ("window", "glass", "glazing")


def _yz_swap_matrix() -> np.ndarray:
    """Return the 4x4 affine that swaps axes 1 (Y) and 2 (Z)."""
    swap = np.eye(4)
    swap[[1, 2]] = swap[[2, 1]]
    return swap


def group_material_name(mesh) -> Optional[str]:
    """Return the group's assigned OBJ material name, or None.

    trimesh exposes the material name on TextureVisuals as
    ``mesh.visual.material.name``. ColorVisuals / untextured meshes (or any
    missing attribute) yield None so callers fall back to name-only matching.
    """
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    name = getattr(material, "name", None)
    return name if isinstance(name, str) else None


def _matches_window(text: Optional[str], keywords) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(kw in low for kw in keywords)


def _group_directive_split(path_str: str):
    """Split an OBJ by its ``g <name>`` / ``o <name>`` group directives.

    trimesh's ``split_objects`` only honors ``o`` (object) directives, so a file
    that separates parts with ``g`` (group) directives -- e.g. ``g building`` /
    ``g window`` -- collapses into one unnamed mesh and its meaningful names are
    lost. Here we rewrite leading ``g <name>`` lines as ``o <name>`` and re-load
    in-memory (with a path resolver so a companion ``.mtl`` still resolves), so
    those named groups survive and drive role detection without needing the
    ``.mtl``.

    Returns the ``(name, mesh)`` groups only when the rewrite yields **two or
    more** named groups; otherwise ``None`` so the caller falls back.
    """
    try:
        with open(path_str, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None

    rewritten_lines = []
    has_group_directive = False
    for line in text.splitlines():
        m = re.match(r"^\s*g\s+(.+?)\s*$", line)
        name = m.group(1).strip() if m else None
        if name and name != "default":
            has_group_directive = True
            rewritten_lines.append("o " + name)
        else:
            rewritten_lines.append(line)
    if not has_group_directive:
        return None

    try:
        resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(path_str) or ".")
        loaded = trimesh.load(
            file_obj=io.StringIO("\n".join(rewritten_lines)),
            file_type="obj",
            process=False,
            split_objects=True,
            group_material=False,
            resolver=resolver,
        )
    except Exception:
        return None

    if not isinstance(loaded, trimesh.Scene):
        return None
    groups = [
        (name, mesh)
        for name, mesh in loaded.geometry.items()
        if isinstance(mesh, trimesh.Trimesh)
    ]
    return groups if len(groups) >= 2 else None


def _material_split_groups(path_str: str):
    """Re-load *path_str* grouping faces by material.

    Returns a list of ``(material_name, mesh)`` groups, but only when the file
    actually splits into **two or more** materials; otherwise returns ``None``
    so the caller keeps the single fallback group (single-material files are
    unchanged). When the OBJ's ``.mtl`` is reachable the group names are the
    material names (enabling window auto-detection); without it trimesh still
    splits by material but the names are generic (geometry index), so callers
    fall back to manual role assignment.
    """
    loaded = trimesh.load(path_str, process=False, split_objects=True, group_material=True)
    if not isinstance(loaded, trimesh.Scene):
        return None
    groups = [
        (name, mesh)
        for name, mesh in loaded.geometry.items()
        if isinstance(mesh, trimesh.Trimesh)
    ]
    if len(groups) < 2:
        return None

    # Without a reachable .mtl, trimesh names the material-split groups after the
    # OBJ file (e.g. 'window_test.obj', 'window_test.obj_1'). Those names are not
    # meaningful and -- worse -- a keyword in the filename (e.g. "window") would
    # make every part auto-detect as a window. Replace such filename-derived
    # names with neutral ones so role detection falls back to the material name
    # (None here) rather than the filename.
    base = os.path.basename(path_str)
    generic = re.compile(re.escape(base) + r"(_\d+)?$")
    cleaned = []
    for i, (name, mesh) in enumerate(groups, start=1):
        if generic.fullmatch(name):
            name = f"{_FALLBACK_GROUP_NAME[:-1]}{i}"  # imported_building_1, _2, ...
        cleaned.append((name, mesh))
    return cleaned


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
        the single-group case. In that situation named parts are recovered
        first from ``g <name>`` group directives (which trimesh otherwise
        ignores, e.g. ``g building`` / ``g window``), then by splitting on
        material (so e.g. a ``Glass`` material becomes its own
        window-detectable group); if neither yields 2+ groups, a single
        group named ``"imported_building_1"`` is returned.

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
        # trimesh collapsed the file to one mesh: it has no `o` objects to split
        # on. Recover named parts two ways, in order of reliability:
        #   1. honor `g <name>` group directives (e.g. `g building`/`g window`),
        #      which trimesh ignores -- names work even without the .mtl;
        #   2. else split by material (a common Rhino export, e.g. usemtl Glass +
        #      usemtl Wall), so a `Glass` material becomes a window-detectable
        #      group (needs the .mtl for meaningful names).
        groups = (
            _group_directive_split(path_str)
            or _material_split_groups(path_str)
            or [(_FALLBACK_GROUP_NAME, loaded)]
        )
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


def classify_roles(
    names: Iterable[str],
    roles: Optional[Dict[str, str]] = None,
    *,
    auto_window: bool = True,
    window_keywords=DEFAULT_WINDOW_KEYWORDS,
    material_names: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """Map each group name to its import role, defaulting to ``"building"``.

    Resolution order per name:
      1. explicit ``roles[name]`` if present (overrides everything),
      2. else ``"window"`` if ``auto_window`` and the group name OR its
         material name (from ``material_names``) contains a window keyword
         (case-insensitive substring),
      3. else ``"building"``.

    Matching against *roles* is exact string match only. ``material_names`` is
    an optional ``{name: material_name}`` mapping; absent/None material names
    simply skip the material-based check.
    """
    roles = roles or {}
    material_names = material_names or {}
    keywords = tuple(k.lower() for k in window_keywords)

    resolved: Dict[str, str] = {}
    for name in names:
        if name in roles:
            resolved[name] = roles[name]
        elif auto_window and (
            _matches_window(name, keywords)
            or _matches_window(material_names.get(name), keywords)
        ):
            resolved[name] = "window"
        else:
            resolved[name] = "building"
    return resolved


def select_groups_by_role(
    groups: List[Tuple[str, "trimesh.Trimesh"]],
    roles: Optional[Dict[str, str]] = None,
    *,
    auto_window: bool = True,
    window_keywords=DEFAULT_WINDOW_KEYWORDS,
) -> Dict[str, List[Tuple[str, "trimesh.Trimesh"]]]:
    """Bucket ``(name, mesh)`` groups by resolved role.

    Returns ``{"building": [...], "window": [...]}``. Material names are read
    from each mesh via :func:`group_material_name`, so window detection by name
    or material both work. Any other/unknown role (e.g. ``"skip"``) is dropped
    and logged at INFO.
    """
    material_names = {name: group_material_name(mesh) for name, mesh in groups}
    resolved = classify_roles(
        [name for name, _mesh in groups],
        roles=roles,
        auto_window=auto_window,
        window_keywords=window_keywords,
        material_names=material_names,
    )
    buckets: Dict[str, List[Tuple[str, "trimesh.Trimesh"]]] = {"building": [], "window": []}
    for name, mesh in groups:
        role = resolved[name]
        if role in buckets:
            buckets[role].append((name, mesh))
        else:
            _logger.info("Skipping group '%s' (role=%s)", name, role)
    return buckets


def select_building_groups(
    groups: List[Tuple[str, "trimesh.Trimesh"]],
    roles: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, "trimesh.Trimesh"]]:
    """Return only the building-role groups (back-compat wrapper).

    With auto window detection on by default, groups whose name or material
    marks them as windows are excluded here (they belong to the ``"window"``
    bucket of :func:`select_groups_by_role`).
    """
    return select_groups_by_role(groups, roles=roles)["building"]
