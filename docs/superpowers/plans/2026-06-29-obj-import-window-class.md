# OBJ-Import Window Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Voxelize window-role OBJ groups as a glass skin (code `-16`) that recolors the building facade voxels they touch, with auto-detection by group name or material name, exposed through the Python importer and the web import UI.

**Architecture:** Buildings voxelize solid (`-3`) as today; window groups are then surface-voxelized (`trimesh.voxel.creation.voxelize_subdivide`) and the building cells within a 1-voxel Chebyshev radius are recolored to `-16`. Detection extends `classify_roles` to match a window keyword against the group name *or* its OBJ material name. Building footprint/height metadata is never touched — windows only reclassify already-solid cells.

**Tech Stack:** Python (numpy, trimesh, scipy.ndimage), pytest; FastAPI backend (Pydantic models); React/TypeScript frontend.

**Conventions:**
- Run tests with the full conda path (conda is not on PATH):
  `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest ...`
- Voxel codes: `-3` building, `-16` glass/window (existing palette/material code).

**Key facts verified during planning:**
- trimesh exposes the per-group material name as `mesh.visual.material.name` with the importer's load options (`split_objects=True, group_material=False`). Untextured/`ColorVisuals` meshes have no such attribute → guard and return `None`.
- `voxelize_subdivide(mesh, pitch=1.0).sparse_indices` are **bbox-relative**; use `.points` (world/index-space cell centers) and `np.floor(...)` to get absolute `(i,j,k)`.
- `scipy` (`>=1.10`) is a project dependency; use `scipy.ndimage` for dilation.

**Known limitation (document, do not block):** The web upload endpoint persists only the uploaded `.obj`, not its `.mtl`. So **material-name** detection works in the Python API (MTL present on disk) but is a no-op in the web app unless the MTL is reachable; **name-based** detection covers the app. MTL upload is a future follow-up.

---

## File Structure

- `src/voxcity/importer/loader.py` (modify) — add `DEFAULT_WINDOW_KEYWORDS`, `group_material_name`, window-aware `classify_roles`, `select_groups_by_role`; keep `select_building_groups` as a wrapper.
- `src/voxcity/importer/windows.py` (create) — `stamp_windows` glass-skin stamping + `_surface_cells` helper.
- `src/voxcity/importer/rhino_obj.py` (modify) — route window groups, call `stamp_windows`, record `n_window_voxels` in the manifest, add `auto_window`/`window_keywords`/`window_value` params.
- `app/backend/models.py` (modify) — add `n_window_voxels_added` to `ImportObjCommitResponse`.
- `app/backend/main.py` (modify) — upload passes `material_names` to `classify_roles`; commit counts `-16` before/after.
- `app/frontend/src/api.ts` (modify) — add `n_window_voxels_added` to `ImportObjCommitResult`.
- `app/frontend/src/tabs/ImportTab.tsx` (modify) — add `window` role option, report window voxels.
- `docs/rhino_obj_import.md` (modify) — replace the "Windows (current behavior)" section.
- Tests: `tests/importer/test_loader.py`, `tests/importer/test_windows.py` (create), `tests/importer/test_add_buildings_from_obj.py`, `app/backend/test_import_obj.py`.

---

## Task 1: Window detection in the loader

**Files:**
- Modify: `src/voxcity/importer/loader.py`
- Test: `tests/importer/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/importer/test_loader.py` (and add `select_groups_by_role`, `group_material_name`, `DEFAULT_WINDOW_KEYWORDS` to the existing import from `voxcity.importer.loader`):

```python
from voxcity.importer.loader import (
    DEFAULT_WINDOW_KEYWORDS,
    group_material_name,
    select_groups_by_role,
)


def test_classify_roles_auto_detects_window_by_name():
    roles = classify_roles(["WallA", "Windows_South", "Glazing_2"])
    assert roles == {"WallA": "building", "Windows_South": "window", "Glazing_2": "window"}


def test_classify_roles_auto_detects_window_by_material():
    # generic group name, but the assigned material is glass
    roles = classify_roles(
        ["panelA", "panelB"],
        material_names={"panelA": "Glass_Clear", "panelB": "Concrete"},
    )
    assert roles == {"panelA": "window", "panelB": "building"}


def test_classify_roles_explicit_override_beats_auto():
    # force a glass-named group to stay building, and a plain group to window
    roles = classify_roles(
        ["Glass_Wall", "plain"],
        roles={"Glass_Wall": "building", "plain": "window"},
    )
    assert roles == {"Glass_Wall": "building", "plain": "window"}


def test_classify_roles_auto_window_can_be_disabled():
    roles = classify_roles(["Windows_1"], auto_window=False)
    assert roles == {"Windows_1": "building"}


def test_group_material_name_reads_texture_material(tmp_path):
    mtl = tmp_path / "m.mtl"
    mtl.write_text("newmtl GlassMat\nKd 0.2 0.4 0.9\n")
    obj = tmp_path / "m.obj"
    obj.write_text(
        "mtllib m.mtl\no Pane\nv 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n"
        "usemtl GlassMat\nf 1 2 3\nf 1 3 4\n"
    )
    groups = load_obj_groups(obj)
    assert any(group_material_name(mesh) == "GlassMat" for _name, mesh in groups)


def test_select_groups_by_role_buckets_building_and_window():
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [("WallA", box), ("Windows_1", box), ("WallB", box)]
    buckets = select_groups_by_role(groups)
    assert [n for n, _ in buckets["building"]] == ["WallA", "WallB"]
    assert [n for n, _ in buckets["window"]] == ["Windows_1"]


def test_select_groups_by_role_drops_skip(caplog, propagate_voxcity_logs):
    box = trimesh.creation.box(extents=(1, 1, 1))
    groups = [("WallA", box), ("ctx", box)]
    with caplog.at_level(logging.INFO, logger="voxcity"):
        buckets = select_groups_by_role(groups, roles={"ctx": "skip"})
    assert [n for n, _ in buckets["building"]] == ["WallA"]
    assert buckets["window"] == []
    assert "ctx" in caplog.text


def test_default_window_keywords_are_english():
    assert DEFAULT_WINDOW_KEYWORDS == ("window", "glass", "glazing")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_loader.py -v`
Expected: FAIL/ERROR with `ImportError: cannot import name 'select_groups_by_role'` (and the new tests undefined).

- [ ] **Step 3: Implement the loader changes**

In `src/voxcity/importer/loader.py`, add the constant near the top (after `_FALLBACK_GROUP_NAME`):

```python
DEFAULT_WINDOW_KEYWORDS = ("window", "glass", "glazing")
```

Add these helpers (place after `_yz_swap_matrix`):

```python
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
```

Replace the existing `classify_roles` function body with:

```python
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
```

Replace `select_building_groups` with the role-bucketing selector plus a thin back-compat wrapper:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_loader.py -v`
Expected: PASS (all old and new tests).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/loader.py tests/importer/test_loader.py
git commit -m "feat(importer): auto-detect window groups by name or material in loader"
```

---

## Task 2: Window stamping module

**Files:**
- Create: `src/voxcity/importer/windows.py`
- Test: `tests/importer/test_windows.py`

- [ ] **Step 1: Write the failing test**

Create `tests/importer/test_windows.py`:

```python
"""Tests for window glass-skin stamping."""
import logging

import numpy as np
import trimesh

from voxcity.importer.windows import stamp_windows
from tests.importer.conftest import make_flat_voxcity

GLASS_CODE = -16
BUILDING_CODE = -3
IDENTITY = np.eye(4)


def _vertical_pane(x0, x1, y, z0, z1):
    """A planar quad in the plane y=const, spanning x in [x0,x1], z in [z0,z1]."""
    verts = np.array(
        [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _wall_voxcity():
    """Flat model with a solid building wall slab at j=4, i in [2,8), z in [1,9)."""
    vc = make_flat_voxcity(nx=12, ny=12, nz=14, meshsize=1.0)
    vc.voxels.classes[2:8, 4, 1:9] = BUILDING_CODE
    return vc


def test_window_recolors_coincident_building_cells():
    vc = _wall_voxcity()
    n_building_before = int(np.sum(vc.voxels.classes == BUILDING_CODE))
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)  # within the wall slab
    n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert n > 0
    glass = vc.voxels.classes == GLASS_CODE
    assert glass.any()
    # all recolored cells are at the wall plane j=4
    assert np.all(np.where(glass)[1] == 4)
    # glass cells came out of the building set; no NEW occupancy was created
    assert int(np.sum(vc.voxels.classes == BUILDING_CODE)) == n_building_before - n


def test_window_far_from_building_is_skipped(caplog, propagate_voxcity_logs):
    vc = _wall_voxcity()
    pane = _vertical_pane(3.0, 6.0, 10.5, 2.0, 7.0)  # j=10, far from the wall at j=4
    with caplog.at_level(logging.INFO, logger="voxcity"):
        n = stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert n == 0
    assert not (vc.voxels.classes == GLASS_CODE).any()
    assert "no building cell" in caplog.text


def test_window_does_not_change_building_metadata():
    vc = _wall_voxcity()
    vc.buildings.ids[2:8, 4] = 7
    vc.buildings.heights[2:8, 4] = 8.0
    ids_before = vc.buildings.ids.copy()
    heights_before = vc.buildings.heights.copy()
    pane = _vertical_pane(3.0, 6.5, 4.5, 2.0, 7.0)
    stamp_windows(vc, [("Windows", pane)], IDENTITY)
    assert np.array_equal(vc.buildings.ids, ids_before)
    assert np.array_equal(vc.buildings.heights, heights_before)


def test_no_window_groups_returns_zero():
    vc = _wall_voxcity()
    assert stamp_windows(vc, [], IDENTITY) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_windows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxcity.importer.windows'`.

- [ ] **Step 3: Implement `windows.py`**

Create `src/voxcity/importer/windows.py`:

```python
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
        skin_radius: Chebyshev radius (in voxels) for matching window surface
            cells to nearby building cells. ``1`` absorbs sub-voxel offsets
            between a pane plane and the wall surface.

    Returns:
        int: number of building cells recolored to *window_value*.
    """
    classes = voxcity.voxels.classes
    grid_shape = classes.shape

    cells_list = [
        _surface_cells(mesh, transform, grid_shape) for _name, mesh in window_groups
    ]
    cells_list = [c for c in cells_list if len(c)]
    if not cells_list:
        return 0
    win_cells = np.unique(np.concatenate(cells_list, axis=0), axis=0)

    win_mask = np.zeros(grid_shape, dtype=bool)
    win_mask[win_cells[:, 0], win_cells[:, 1], win_cells[:, 2]] = True
    building_mask = classes == building_value

    if skin_radius > 0:
        structure = ndimage.generate_binary_structure(3, 3)  # full 3x3x3 (Chebyshev)
        win_dilated = ndimage.binary_dilation(
            win_mask, structure=structure, iterations=skin_radius
        )
        bld_dilated = ndimage.binary_dilation(
            building_mask, structure=structure, iterations=skin_radius
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_windows.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/windows.py tests/importer/test_windows.py
git commit -m "feat(importer): add stamp_windows glass-skin stamping (-16)"
```

---

## Task 3: Wire windows into `add_buildings_from_obj`

**Files:**
- Modify: `src/voxcity/importer/rhino_obj.py`
- Test: `tests/importer/test_add_buildings_from_obj.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/importer/test_add_buildings_from_obj.py`:

```python
import trimesh

GLASS_CODE = -16


def _box_with_window_obj(tmp_path):
    """OBJ with a solid box 'BuildingA' and a planar window pane 'Windows' on
    the box's -Y face (model y=0), inset from the edges."""
    box = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    box.apply_translation((1.5, 1.5, 2.0))  # min corner at origin
    pane_v = np.array(
        [[0.5, 0.0, 0.5], [2.5, 0.0, 0.5], [2.5, 0.0, 3.5], [0.5, 0.0, 3.5]],
        dtype=float,
    )
    pane = trimesh.Trimesh(vertices=pane_v, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                           process=False)
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="BuildingA", geom_name="BuildingA")
    scene.add_geometry(pane, node_name="Windows", geom_name="Windows")
    path = tmp_path / "bld_with_window.obj"
    scene.export(str(path))
    return path


def test_import_window_group_produces_glass_voxels(tmp_path):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    obj = _box_with_window_obj(tmp_path)
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        move=(5.0, 5.0, 0.0), rotation=0.0, units="m",
    )
    assert np.any(out.voxels.classes == BUILDING_CODE)
    assert np.any(out.voxels.classes == GLASS_CODE)
    manifest = out.extras["imported_buildings"][-1]
    assert manifest["n_window_voxels"] > 0


def test_building_only_import_has_no_glass(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    obj = box_obj_factory(size=(3.0, 3.0, 4.0))
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, move=(5.0, 5.0, 0.0),
    )
    assert not np.any(out.voxels.classes == GLASS_CODE)
    assert out.extras["imported_buildings"][-1].get("n_window_voxels", 0) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_add_buildings_from_obj.py -v`
Expected: FAIL — `test_import_window_group_produces_glass_voxels` finds no `-16` cells / `KeyError: 'n_window_voxels'` (windows currently discarded).

- [ ] **Step 3: Implement the wiring**

In `src/voxcity/importer/rhino_obj.py`:

(a) Update imports:

```python
from .loader import load_obj_groups, select_groups_by_role, DEFAULT_WINDOW_KEYWORDS
from .windows import stamp_windows
```

(remove the old `select_building_groups` import).

(b) Add parameters to the `add_buildings_from_obj` signature (place after `overwrite=True,` and before `gridvis=False,`):

```python
    auto_window=True,
    window_keywords=None,
    window_value=-16,
```

(c) Replace the load + role-routing block:

```python
    # --- load + role routing ---
    groups = load_obj_groups(obj_path, swap_yz=apply_swap)
    building_groups = select_building_groups(groups, roles=roles)
    if not building_groups:
        _logger.warning("No building-role geometry found in %s; nothing imported.", obj_path)
        return copy.deepcopy(voxcity)
```

with:

```python
    # --- load + role routing ---
    kw = DEFAULT_WINDOW_KEYWORDS if window_keywords is None else tuple(window_keywords)
    groups = load_obj_groups(obj_path, swap_yz=apply_swap)
    buckets = select_groups_by_role(
        groups, roles=roles, auto_window=auto_window, window_keywords=kw
    )
    building_groups = buckets["building"]
    window_groups = buckets["window"]
    if not building_groups:
        _logger.warning("No building-role geometry found in %s; nothing imported.", obj_path)
        return copy.deepcopy(voxcity)
```

(d) After the `out = stamp_buildings(...)` call and before the `if gridvis:` block, insert:

```python
    # --- windows: glass skin on building facade cells ---
    if window_groups:
        n_window = stamp_windows(out, window_groups, M, window_value=window_value)
        manifests = out.extras.get("imported_buildings")
        if manifests:
            manifests[-1]["n_window_voxels"] = int(n_window)
```

(e) Update the `roles` arg docstring sentence to reflect the new behavior. Replace the existing `roles:` paragraph in the docstring with:

```
        roles: optional ``{group_name: role}`` mapping overriding the
            auto-detected role of a group. Recognized roles: ``"building"``
            (voxelized solid as ``-3``), ``"window"`` (surface-voxelized and
            stamped as glass ``-16`` on the building facade it touches), and
            ``"skip"`` (excluded). Matching is exact-string only. Groups absent
            from this mapping are auto-classified (see ``auto_window``).
```

(f) Add docstring entries for the new params after the `overwrite:` paragraph:

```
        auto_window: if ``True`` (default), groups whose name or assigned OBJ
            material name contains a window keyword (see ``window_keywords``)
            are auto-classified as ``"window"``. An explicit ``roles`` entry
            always overrides this.
        window_keywords: optional iterable of case-insensitive substrings used
            for window auto-detection. ``None`` (default) uses
            ``("window", "glass", "glazing")``.
        window_value: voxel code written for window cells. Defaults to ``-16``
            (the standard glass code).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_add_buildings_from_obj.py tests/importer -v`
Expected: PASS (new window tests + all existing importer tests).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/rhino_obj.py tests/importer/test_add_buildings_from_obj.py
git commit -m "feat(importer): stamp window groups as glass skin in add_buildings_from_obj"
```

---

## Task 4: Backend — material-aware upload and window count

**Files:**
- Modify: `app/backend/models.py`
- Modify: `app/backend/main.py`
- Test: `app/backend/test_import_obj.py`

- [ ] **Step 1: Write the failing tests**

Append to `app/backend/test_import_obj.py`:

```python
def _box_with_window_obj_bytes() -> bytes:
    """A box plus a window-named planar pane, exported to OBJ bytes."""
    box = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    box.apply_translation((1.5, 1.5, 2.0))
    import numpy as _np
    pane = trimesh.Trimesh(
        vertices=_np.array(
            [[0.5, 0.0, 0.5], [2.5, 0.0, 0.5], [2.5, 0.0, 3.5], [0.5, 0.0, 3.5]],
            dtype=float,
        ),
        faces=_np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="BuildingA", geom_name="BuildingA")
    scene.add_geometry(pane, node_name="Windows", geom_name="Windows")
    return scene.export(file_type="obj").encode("utf-8")


def test_upload_reports_window_role(client):
    files = {"file": ("bw.obj", io.BytesIO(_box_with_window_obj_bytes()), "text/plain")}
    r = client.post("/api/model/import_obj/upload", files=files)
    assert r.status_code == 200, r.text
    roles = {g["name"]: g["role"] for g in r.json()["groups"]}
    assert roles.get("Windows") == "window"
    assert roles.get("BuildingA") == "building"


def test_commit_reports_window_voxels(client):
    files = {"file": ("bw.obj", io.BytesIO(_box_with_window_obj_bytes()), "text/plain")}
    import_id = client.post("/api/model/import_obj/upload", files=files).json()["import_id"]
    req = {
        "import_id": import_id,
        "placement": {"anchor_lonlat": _domain_center_lonlat(), "anchor_elevation": 0.0},
        "roles": {},
        "overwrite": True,
    }
    r = client.post("/api/model/import_obj/commit", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["n_window_voxels_added"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: FAIL — `test_upload_reports_window_role` sees `role == "building"`; `test_commit_reports_window_voxels` gets `KeyError`/missing `n_window_voxels_added`.

- [ ] **Step 3a: Add the response field**

In `app/backend/models.py`, in `ImportObjCommitResponse`, add the field after `n_building_voxels_added`:

```python
    n_window_voxels_added: int = 0
```

- [ ] **Step 3b: Material-aware upload classification**

In `app/backend/main.py`, in `import_obj_upload`, update the loader import and the role-map construction:

Change:
```python
    from voxcity.importer.loader import load_obj_groups, classify_roles
```
to:
```python
    from voxcity.importer.loader import load_obj_groups, classify_roles, group_material_name
```

Change:
```python
    role_map = classify_roles([name for name, _ in groups])
```
to:
```python
    material_names = {name: group_material_name(mesh) for name, mesh in groups}
    role_map = classify_roles([name for name, _ in groups], material_names=material_names)
```

- [ ] **Step 3c: Count window voxels on commit**

In `app/backend/main.py`, in `import_obj_commit`, add a window baseline next to the building baseline:

Change:
```python
    before = int(np.sum(np.asarray(app_state.voxcity.voxels.classes) == -3))
```
to:
```python
    before = int(np.sum(np.asarray(app_state.voxcity.voxels.classes) == -3))
    before_w = int(np.sum(np.asarray(app_state.voxcity.voxels.classes) == -16))
```

After `app_state.voxcity = out` / `app_state.refresh_raw_cache()` and the existing `after = ...` line, add:

```python
    after_w = int(np.sum(np.asarray(out.voxels.classes) == -16))
    n_window_added = after_w - before_w
```

Update the final return to include the new field:

```python
    return ImportObjCommitResponse(
        figure_json=_render_edit_preview(out, title="Imported building"),
        imported_building_ids=ids,
        n_building_voxels_added=int(n_added),
        n_window_voxels_added=int(n_window_added),
        warning=warning,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest app/backend/test_import_obj.py -v`
Expected: PASS (new + existing import endpoint tests).

- [ ] **Step 5: Commit**

```bash
git add app/backend/models.py app/backend/main.py app/backend/test_import_obj.py
git commit -m "feat(app): material-aware window detection on upload, report window voxel count"
```

---

## Task 5: Frontend — window role option and reporting

**Files:**
- Modify: `app/frontend/src/api.ts`
- Modify: `app/frontend/src/tabs/ImportTab.tsx`

No new unit test: `ImportTab` has no existing test harness and the change is presentational wiring. Verify via TypeScript build.

- [ ] **Step 1: Add the response field to the DTO**

In `app/frontend/src/api.ts`, in `ImportObjCommitResult`, add after `n_building_voxels_added`:

```typescript
  n_window_voxels_added: number;
```

- [ ] **Step 2: Add the `window` role option**

In `app/frontend/src/tabs/ImportTab.tsx`, in the GROUPS / ROLES `<select>`, add the `window` option between `building` and `skip`:

```tsx
                        <select value={roles[g.name] ?? 'building'} disabled={busy}
                                onChange={(e) => setRoles((r) => ({ ...r, [g.name]: e.target.value }))}>
                          <option value="building">building</option>
                          <option value="window">window</option>
                          <option value="skip">skip</option>
                        </select>
```

- [ ] **Step 3: Report window voxels in the success message**

In `app/frontend/src/tabs/ImportTab.tsx`, in `handleImport`, update the success branch message:

```tsx
        setInfo(
          `Imported ${r.imported_building_ids.length} building(s); ` +
          `${r.n_building_voxels_added} voxel(s) added` +
          (r.n_window_voxels_added > 0 ? `, ${r.n_window_voxels_added} window voxel(s)` : '') +
          `.`,
        );
```

- [ ] **Step 4: Verify the frontend builds (typecheck)**

Run: `npm --prefix app/frontend run build`
Expected: build succeeds with no TypeScript errors.

(If the project uses a separate typecheck script, e.g. `npm --prefix app/frontend run typecheck`, run that too.)

- [ ] **Step 5: Commit**

```bash
git add app/frontend/src/api.ts app/frontend/src/tabs/ImportTab.tsx
git commit -m "feat(app): add window role option and window voxel reporting to ImportTab"
```

---

## Task 6: Documentation

**Files:**
- Modify: `docs/rhino_obj_import.md`

- [ ] **Step 1: Replace the windows section**

In `docs/rhino_obj_import.md`, replace the entire `## Windows / glazing (current behavior)` section (through its code block) with:

```markdown
## Windows / glazing

Model opaque mass as **closed solids**; model windows as **open planar surfaces**
flush with the wall (within ~1 `meshsize`). Window groups are surface-voxelized
and the building facade voxels they touch are recolored to the glass code
(`-16`); the wall behind stays solid. Windows only reclassify existing building
cells, so there must be a solid wall behind each pane (do **not** cut window
holes in the building solid).

**Auto-detection.** A group is treated as a window when its object/layer name
**or** its assigned OBJ material name contains `window`, `glass`, or `glazing`
(case-insensitive). Override per group with `roles`, e.g.
`roles={"Facade_North": "window"}` or force a glass-named group back to building
with `roles={"Glass_Wall": "building"}`. Customize the keywords with
`window_keywords=(...)` (e.g. add Japanese terms). Disable with
`auto_window=False`.

```python
vc = add_buildings_from_obj(
    vc, "design.obj",
    anchor_lonlat=(139.7536, 35.6841), anchor_elevation=12.0,
    rotation=0.0, units="m",
    # windows auto-detected by name/material; or be explicit:
    roles={"Windows_South": "window"},
)
```

**Web app note:** the import UI uploads only the `.obj`, so material-name
detection requires the `.mtl` to be reachable; name-based detection always
works. The per-group role dropdown lets you set **building / window / skip**.

For procedural (non-geometry) windows, the `set_building_material_by_id`
material utilities still apply to imported buildings.
```

- [ ] **Step 2: Commit**

```bash
git add docs/rhino_obj_import.md
git commit -m "docs: document geometry-driven window import"
```

---

## Final verification

- [ ] **Run the full importer + app import test suites**

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer app/backend/test_import_obj.py -v
```
Expected: all PASS.

- [ ] **Confirm no regressions in the broader importer/app area** (optional, broader)

Run:
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer tests/app -q
```
Expected: all PASS.

---

## Self-Review (completed during planning)

**Spec coverage:**
- §2 role routing (name + material, keywords, skip vocab) → Task 1.
- §3 window stamping (surface-voxelize → recolor `-3`→`-16`, skip unmatched, metadata untouched, manifest count) → Tasks 2 & 3.
- §4 public API (`auto_window`/`window_keywords`/`window_value`) → Task 3.
- §5 web app (models field, upload material map, commit count, frontend dropdown + message, palette renders `-16` unchanged) → Tasks 4 & 5.
- §6 out-of-scope (simulators) → not implemented, by design.
- §7 testing → Tasks 1–4 test steps + final verification.

**Type/name consistency:** `select_groups_by_role` returns `{"building", "window"}`; `stamp_windows(voxcity, window_groups, transform, *, window_value, building_value, skin_radius)` and `_surface_cells(mesh, transform, grid_shape)` are used consistently across Tasks 2–3. Backend field `n_window_voxels_added` matches between `models.py`, `main.py`, and `api.ts`. Manifest key `n_window_voxels` matches between Task 3 (write) and Task 3 test (read).

**Known limitation captured:** material detection in the web app requires the MTL (header note + doc note); name detection covers the app.
