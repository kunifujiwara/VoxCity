# Rhino/OBJ Building Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `voxcity.importer.add_buildings_from_obj`, which voxelizes Rhino-authored buildings from an OBJ file and stamps them into an existing VoxCity model, georeferenced by an anchor + rotation + move + units.

**Architecture:** A new `voxcity/importer/` subpackage. A pure-geometry transform builds a 4×4 affine from model coordinates → voxel-index space (using `GridProjector` + the DEM datum). A column z-ray voxelizer turns each building mesh into occupied `(i, j, k)` cells. An integrator stamps those cells into `voxcity.voxels.classes` and updates the derived metadata grids (`building_id_grid`, `building_height_grid`, `min_heights`). A role router voxelizes only `building`-role OBJ groups and skips others (e.g. window layers). Trimesh is the default backend; an optional lazy `meshlib` backend is dispatched but not implemented in v1 beyond a clear error.

**Tech Stack:** Python 3.12, NumPy, trimesh (already a dependency), pytest. Optional: meshlib (not required).

**Spec:** `docs/superpowers/specs/2026-06-23-rhino-obj-building-import-design.md`

**Test command (this machine — conda not on PATH):**
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest <args>
```

---

## File Structure

```
src/voxcity/importer/
    __init__.py        # public re-export of add_buildings_from_obj
    units.py           # unit-scale table + small validators
    transform.py       # grid-geom from VoxCity + build_placement_transform (model->voxel index 4x4)
    voxelize.py        # voxelize_mesh: trimesh column z-ray fill -> occupied (i,j,k)
    loader.py          # load_obj_groups (named groups), classify_roles
    integrate.py       # stamp_buildings: write voxels + update metadata grids + provenance
    rhino_obj.py       # add_buildings_from_obj orchestration + backend dispatch + errors

tests/importer/
    __init__.py
    conftest.py        # fixtures: tiny VoxCity object, OBJ-file factory
    test_units.py
    test_transform.py
    test_voxelize.py
    test_loader.py
    test_integrate.py
    test_add_buildings_from_obj.py

docs/
    rhino_obj_import.md  # user-facing Rhino export + usage guide
```

---

## Task 1: Package scaffold + units helper

**Files:**
- Create: `src/voxcity/importer/__init__.py`
- Create: `src/voxcity/importer/units.py`
- Create: `tests/importer/__init__.py`
- Test: `tests/importer/test_units.py`

- [ ] **Step 1: Write the failing test**

`tests/importer/test_units.py`:
```python
import pytest
from voxcity.importer.units import unit_scale, validate_units


def test_known_units_scale_to_meters():
    assert unit_scale("m") == 1.0
    assert unit_scale("cm") == 0.01
    assert unit_scale("mm") == 0.001
    assert unit_scale("ft") == pytest.approx(0.3048)
    assert unit_scale("in") == pytest.approx(0.0254)


def test_units_are_case_insensitive():
    assert unit_scale("M") == 1.0
    assert unit_scale("FT") == pytest.approx(0.3048)


def test_invalid_units_raise_valueerror():
    with pytest.raises(ValueError, match="Unknown units"):
        unit_scale("furlong")


def test_validate_units_passes_known():
    validate_units("mm")  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_units.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.importer'`

- [ ] **Step 3: Write minimal implementation**

`tests/importer/__init__.py`: (empty file)

`src/voxcity/importer/units.py`:
```python
"""Unit handling for OBJ import (model units -> meters)."""

_UNIT_SCALE = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "in": 0.0254,
}


def unit_scale(units: str) -> float:
    """Return meters-per-unit for a model unit string (case-insensitive)."""
    if not isinstance(units, str):
        raise ValueError(f"Unknown units: {units!r}. Expected one of {sorted(_UNIT_SCALE)}.")
    key = units.lower()
    if key not in _UNIT_SCALE:
        raise ValueError(f"Unknown units: {units!r}. Expected one of {sorted(_UNIT_SCALE)}.")
    return _UNIT_SCALE[key]


def validate_units(units: str) -> None:
    """Raise ValueError if *units* is not a known unit string."""
    unit_scale(units)
```

`src/voxcity/importer/__init__.py`:
```python
"""VoxCity importer subpackage: import external 3D geometry into VoxCity models.

Public API:
    add_buildings_from_obj(voxcity, obj_path, ...) -> VoxCity
"""

from .rhino_obj import add_buildings_from_obj

__all__ = ["add_buildings_from_obj"]
```

NOTE: importing `__init__` will fail until Task 6 creates `rhino_obj.py`. To keep
this task green in isolation, temporarily make `__init__.py` contain only the
docstring + `__all__ = []`. Task 6 Step "wire up exports" restores the import.

For this task, write `src/voxcity/importer/__init__.py` as:
```python
"""VoxCity importer subpackage: import external 3D geometry into VoxCity models."""

__all__ = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_units.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/__init__.py src/voxcity/importer/units.py tests/importer/__init__.py tests/importer/test_units.py
git commit -m "feat(importer): scaffold importer package + units helper"
```

---

## Task 2: Shared test fixtures

**Files:**
- Create: `tests/importer/conftest.py`

These fixtures are used by Tasks 3, 5, 6. No production code; verified by being
imported in later tests. Commit them now so later tasks can rely on them.

- [ ] **Step 1: Write the fixtures**

`tests/importer/conftest.py`:
```python
"""Shared fixtures for importer tests."""
import numpy as np
import pytest
import trimesh

from voxcity.models import (
    VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata,
)

GROUND_CODE = -1
BUILDING_CODE = -3


def make_flat_voxcity(nx=20, ny=20, nz=10, meshsize=1.0):
    """Build a minimal flat-DEM VoxCity object for import tests.

    Axis-aligned 1-degree-ish rectangle is unnecessary; we use a small real
    rectangle near the equator so distances are well-behaved. Ground voxel at
    k=0 (land cover), everything above is air (0).
    """
    # Small axis-aligned rectangle (lon, lat): SW, NW, NE, SE
    lon0, lat0 = 0.0, 0.0
    # ~meshsize*nx meters wide. 1 deg lat ~= 111320 m.
    dlat = (meshsize * nx) / 111320.0
    dlon = (meshsize * ny) / 111320.0
    rectangle_vertices = [
        (lon0, lat0),
        (lon0, lat0 + dlat),
        (lon0 + dlon, lat0 + dlat),
        (lon0 + dlon, lat0),
    ]
    meta = GridMetadata(crs="EPSG:4326", bounds=(lon0, lat0, lon0 + dlon, lat0 + dlat), meshsize=meshsize)

    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = GROUND_CODE  # ground/landcover layer

    heights = np.zeros((nx, ny), dtype=float)
    ids = np.zeros((nx, ny), dtype=np.int32)
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.zeros((nx, ny), dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rectangle_vertices},
    )


@pytest.fixture
def flat_voxcity():
    return make_flat_voxcity()


@pytest.fixture
def box_obj_factory(tmp_path):
    """Return a factory writing a single-box OBJ and returning its path.

    box(origin=(x,y,z), size=(sx,sy,sz), name=...) -> Path to .obj
    """
    def _factory(origin=(0.0, 0.0, 0.0), size=(2.0, 2.0, 3.0), name="building1", filename="model.obj"):
        mesh = trimesh.creation.box(extents=size)
        # trimesh box is centered at origin; move so min corner = origin
        mesh.apply_translation(np.array(size) / 2.0 + np.array(origin))
        path = tmp_path / filename
        # Export to OBJ (extension drives the format)
        mesh.export(str(path))
        return path

    return _factory
```

- [ ] **Step 2: Sanity-check the fixtures import**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "import tests.importer.conftest"` from repo root.
Expected: no output, exit 0. (If `tests` is not importable as a package, this is fine — pytest will still discover conftest; skip this check and rely on Task 3.)

- [ ] **Step 3: Commit**

```bash
git add tests/importer/conftest.py
git commit -m "test(importer): add shared fixtures (flat VoxCity, box OBJ factory)"
```

---

## Task 3: Placement transform (model coords -> voxel index)

**Files:**
- Create: `src/voxcity/importer/transform.py`
- Test: `tests/importer/test_transform.py`

The transform composes (right-to-left): translate by `-anchor_model_point`,
scale by `unit_scale`, rotate horizontally by `(θ + domain_rotation)`, translate
to the anchor's `(u_m, v_m)` plus horizontal move and to the vertical datum, then
scale by `1/meshsize` and add the +1 vertical voxel offset (to match the
voxelizer seating buildings one voxel above the ground voxel).

Convention (documented + tested): with `rotation=0` and an axis-aligned domain,
model **+X → +v (east / axis 1)** and model **+Y → +u (north / axis 0)**.

- [ ] **Step 1: Write the failing test**

`tests/importer/test_transform.py`:
```python
import numpy as np
import pytest

from voxcity.importer.transform import grid_geom_from_voxcity, build_placement_transform
from tests.importer.conftest import make_flat_voxcity


def _apply(M, pts):
    pts = np.asarray(pts, dtype=float)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (homog @ M.T)[:, :3]


def test_grid_geom_has_expected_keys():
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    for key in ("origin", "u_vec", "v_vec", "adj_mesh", "grid_size", "meshsize_m"):
        assert key in geom


def test_anchor_maps_to_origin_cell_axis_aligned():
    """rotation=0, units=m, anchor at grid origin corner, model origin anchored.

    Model point (0,0,0) -> voxel index ~ (0, 0, 1): the +1 is the ground offset.
    """
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]  # grid SW corner
    M = build_placement_transform(
        vc,
        anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0,
        anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0,
        move=(0.0, 0.0, 0.0),
        units="m",
    )
    out = _apply(M, [[0.0, 0.0, 0.0]])[0]
    assert out[0] == pytest.approx(0.0, abs=1e-6)   # i (u/north)
    assert out[1] == pytest.approx(0.0, abs=1e-6)   # j (v/east)
    assert out[2] == pytest.approx(1.0, abs=1e-6)   # k (ground offset)


def test_axis_mapping_x_to_v_y_to_u():
    """+X (east) advances j; +Y (north) advances i; meshsize scales."""
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(0.0, 0.0, 0.0), units="m",
    )
    out = _apply(M, [[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 4.0]])
    # +X=3 -> j=3
    assert out[0][1] == pytest.approx(3.0, abs=1e-4)
    assert out[0][0] == pytest.approx(0.0, abs=1e-4)
    # +Y=5 -> i=5
    assert out[1][0] == pytest.approx(5.0, abs=1e-4)
    assert out[1][1] == pytest.approx(0.0, abs=1e-4)
    # +Z=4 -> k=4+1
    assert out[2][2] == pytest.approx(5.0, abs=1e-4)


def test_units_feet_scale():
    vc = make_flat_voxcity(nx=50, ny=50, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(0.0, 0.0, 0.0), units="ft",
    )
    # 10 ft along +X = 3.048 m -> j ~ 3.048 voxels (meshsize 1)
    out = _apply(M, [[10.0, 0.0, 0.0]])[0]
    assert out[1] == pytest.approx(3.048, abs=1e-3)


def test_rotation_90_maps_x_to_north():
    """rotation=90 deg. Convention: positive rotation rotates the model so that
    its +X axis ends up pointing +u (north). Verify +X(east) -> +i.
    """
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=90.0, move=(0.0, 0.0, 0.0), units="m",
    )
    out = _apply(M, [[4.0, 0.0, 0.0]])[0]
    assert out[0] == pytest.approx(4.0, abs=1e-4)   # now along i (north)
    assert out[1] == pytest.approx(0.0, abs=1e-4)


def test_move_offsets_in_voxels():
    vc = make_flat_voxcity(nx=20, ny=20, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    anchor_lonlat = geom["origin"]
    M = build_placement_transform(
        vc, anchor_lonlat=(float(anchor_lonlat[0]), float(anchor_lonlat[1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        rotation=0.0, move=(2.0, 3.0, 4.0), units="m",  # (east_m, north_m, up_m)
    )
    out = _apply(M, [[0.0, 0.0, 0.0]])[0]
    assert out[1] == pytest.approx(2.0, abs=1e-4)   # east move -> j
    assert out[0] == pytest.approx(3.0, abs=1e-4)   # north move -> i
    assert out[2] == pytest.approx(5.0, abs=1e-4)   # up move 4 + ground offset 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_transform.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.importer.transform'`

- [ ] **Step 3: Write minimal implementation**

`src/voxcity/importer/transform.py`:
```python
"""Build the affine that maps OBJ model coordinates to VoxCity voxel indices.

Convention (rotation=0, axis-aligned domain):
    model +X -> +v (east  / array axis 1)
    model +Y -> +u (north / array axis 0)
    model +Z -> +k (up)
Positive `rotation` (degrees) rotates the model counter-clockwise in the
(east, north) plane so that, at rotation=90, model +X points north (+u).
"""
from __future__ import annotations

import math
import numpy as np

from ..geoprocessor.raster.core import compute_grid_geometry
from ..utils.projector import GridProjector
from .units import unit_scale


def grid_geom_from_voxcity(voxcity):
    """Recover grid geometry from a VoxCity object via its rectangle_vertices."""
    rv = (voxcity.extras or {}).get("rectangle_vertices")
    if not rv:
        raise ValueError(
            "VoxCity object has no extras['rectangle_vertices']; cannot georeference "
            "the import. Regenerate the model with get_voxcity (which stores it) or "
            "set voxcity.extras['rectangle_vertices'] manually."
        )
    meshsize = float(voxcity.voxels.meta.meshsize)
    geom = compute_grid_geometry(rv, meshsize)
    if geom is None:
        raise ValueError("Could not compute grid geometry from rectangle_vertices.")
    return geom


def _domain_rotation_deg(geom) -> float:
    """Bearing (deg, clockwise from north) of the domain +u axis (side_1)."""
    u = np.asarray(geom["u_vec"], dtype=float)  # (dlon, dlat) per metre
    # clockwise-from-north bearing of (east=dlon, north=dlat)
    return math.degrees(math.atan2(u[0], u[1]))


def build_placement_transform(
    voxcity,
    anchor_lonlat,
    anchor_elevation,
    anchor_model_point=(0.0, 0.0, 0.0),
    rotation=0.0,
    move=(0.0, 0.0, 0.0),
    units="m",
):
    """Return a 4x4 affine mapping model coords -> voxel index space (i, j, k)."""
    geom = grid_geom_from_voxcity(voxcity)
    meshsize = float(geom["meshsize_m"])
    scale = unit_scale(units)

    # 1. translate model so anchor_model_point is the origin
    T0 = np.eye(4)
    T0[:3, 3] = -np.asarray(anchor_model_point, dtype=float)

    # 2. scale model units -> meters
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    # 3. horizontal rotation: model (x=east, y=north) -> domain (u, v) metres.
    #    Total rotation = user rotation - domain rotation (both clockwise-from-north).
    phi = math.radians(_domain_rotation_deg(geom))
    theta = math.radians(float(rotation))
    # First apply user rotation (CCW positive) in (east, north); then express in
    # domain frame which is rotated by phi clockwise from north.
    # Map (x_m, y_m) meters -> (u_m, v_m) meters.
    # north/east unit components of domain axes:
    #   u_dir(east,north) = (sin phi, cos phi); v_dir = (cos phi, -sin phi)
    # user rotation rotates model vector by +theta CCW in (east,north):
    #   e =  x cos? ... implemented as a single 2x2 below.
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # model -> (east, north): rotation=0 keeps x->east, y->north;
    # rotation=90 should send +x(east) to north, i.e. (e,n) = (x cos t - y sin t? )
    # Choose: e = x*cos_t - y*sin_t ; n = x*sin_t + y*cos_t  (CCW)
    # at theta=90: e=-y, n=x -> +x(4,0) -> (e,n)=(0,4) = north. matches test.
    def to_en(x, y):
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    sin_p, cos_p = math.sin(phi), math.cos(phi)

    # Build the linear 2x2 mapping (x,y) -> (u_m, v_m):
    # for unit x: (e,n)=(cos_t, sin_t) -> u = e*sin_p + n*cos_p ; v = e*cos_p - n*sin_p
    ex, nx_ = to_en(1.0, 0.0)
    ey, ny_ = to_en(0.0, 1.0)
    u_from_x = ex * sin_p + nx_ * cos_p
    v_from_x = ex * cos_p - nx_ * sin_p
    u_from_y = ey * sin_p + ny_ * cos_p
    v_from_y = ey * cos_p - ny_ * sin_p

    R = np.eye(4)
    # rows: out axis (u=i, v=j, w=k); cols: model meters (x, y, z)
    R[0, 0] = u_from_x
    R[0, 1] = u_from_y
    R[1, 0] = v_from_x
    R[1, 1] = v_from_y
    R[2, 2] = 1.0

    # 4. translate to anchor position in domain metres + move + vertical datum
    proj = GridProjector(geom)
    u_a, v_a = proj.lon_lat_to_uv_m(float(anchor_lonlat[0]), float(anchor_lonlat[1]))
    move_e, move_n, move_up = (float(m) for m in move)
    dem_min = float(np.min(voxcity.dem.elevation))
    z_a = float(anchor_elevation) - dem_min + move_up

    T1 = np.eye(4)
    T1[0, 3] = u_a + move_n   # north -> u axis (i)
    T1[1, 3] = v_a + move_e   # east  -> v axis (j)
    T1[2, 3] = z_a

    # 5. metres -> voxel index, plus +1 ground voxel offset on k
    Sv = np.eye(4)
    Sv[0, 0] = Sv[1, 1] = Sv[2, 2] = 1.0 / meshsize
    Toff = np.eye(4)
    Toff[2, 3] = 1.0  # ground offset (matches voxelizer ground_level = ... + 1)

    # compose: index = Toff @ Sv @ T1 @ R @ S @ T0
    M = Toff @ Sv @ T1 @ R @ S @ T0
    return M
```

NOTE to implementer: the rotation sign/axis convention is pinned by
`test_rotation_90_maps_x_to_north` and `test_axis_mapping_x_to_v_y_to_u`. If the
first run shows a flipped sign, fix the `to_en` / `R` assignment until those two
tests pass — do not change the test expectations (they encode the documented
convention).

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_transform.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/transform.py tests/importer/test_transform.py
git commit -m "feat(importer): placement transform model coords -> voxel index"
```

---

## Task 4: Mesh voxelizer (column z-ray fill)

**Files:**
- Create: `src/voxcity/importer/voxelize.py`
- Test: `tests/importer/test_voxelize.py`

`voxelize_mesh(mesh, transform, grid_shape)` applies the transform to a copy of
the mesh, then for each candidate `(i, j)` column within the mesh's XY footprint
casts a vertical ray through the column center, sorts the z-hits, and fills cells
between even-odd hit pairs (true solid fill, supports overhangs/gaps via multiple
spans). Out-of-bounds cells are clipped with a warning. Odd hit counts fall back
to filling `[min_hit, max_hit)` with a warning.

- [ ] **Step 1: Write the failing test**

`tests/importer/test_voxelize.py`:
```python
import numpy as np
import trimesh

from voxcity.importer.voxelize import voxelize_mesh


def _box(min_corner, size):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.array(min_corner) + np.array(size) / 2.0)
    return m


def test_unit_cube_fills_expected_cells():
    # cube from (0,0,0) to (3,3,3) in index space; identity transform
    mesh = _box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    occ = voxelize_mesh(mesh, np.eye(4), grid_shape=(10, 10, 10))
    cells = set(map(tuple, occ))
    # interior columns should be filled 0..2 in all axes
    assert (1, 1, 1) in cells
    assert (0, 0, 0) in cells
    assert (2, 2, 2) in cells
    # nothing outside
    assert max(c[0] for c in cells) <= 2
    assert max(c[2] for c in cells) <= 2


def test_l_shape_leaves_notch_empty():
    # two boxes forming an L; the notch cell should be empty
    a = _box((0.0, 0.0, 0.0), (4.0, 2.0, 2.0))
    b = _box((0.0, 0.0, 0.0), (2.0, 4.0, 2.0))
    mesh = trimesh.util.concatenate([a, b])
    occ = set(map(tuple, voxelize_mesh(mesh, np.eye(4), grid_shape=(10, 10, 10))))
    assert (3, 3, 0) not in occ      # the notch
    assert (0, 0, 0) in occ
    assert (3, 0, 0) in occ
    assert (0, 3, 0) in occ


def test_out_of_bounds_clipped():
    mesh = _box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    occ = set(map(tuple, voxelize_mesh(mesh, np.eye(4), grid_shape=(2, 2, 2))))
    # grid only has i,j,k in {0,1}; cells at index 2 are dropped
    assert all(i < 2 and j < 2 and k < 2 for (i, j, k) in occ)
    assert (1, 1, 1) in occ


def test_open_box_fallback_fills_solid():
    # a box with the top face removed (non-watertight) should still fill solid
    mesh = _box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    # delete faces whose centroid is at max z (the top)
    zc = mesh.triangles_center[:, 2]
    keep = zc < (zc.max() - 1e-6)
    mesh.update_faces(keep)
    occ = set(map(tuple, voxelize_mesh(mesh, np.eye(4), grid_shape=(10, 10, 10))))
    assert (1, 1, 1) in occ
    assert (1, 1, 0) in occ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_voxelize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.importer.voxelize'`

- [ ] **Step 3: Write minimal implementation**

`src/voxcity/importer/voxelize.py`:
```python
"""Voxelize a placed mesh into occupied (i, j, k) voxel indices (trimesh backend).

Method: column z-ray even-odd fill. For each (i, j) column whose center lies
inside the mesh XY footprint, cast a vertical ray and fill cells between
consecutive entry/exit hits. Robust to overhangs and gaps (multiple spans). For
non-watertight columns (odd hit count) fall back to filling [min_hit, max_hit).
"""
from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

_logger = get_logger(__name__)


def voxelize_mesh(mesh, transform, grid_shape):
    """Return an (N, 3) int array of occupied (i, j, k) cells within grid_shape."""
    nx, ny, nz = grid_shape
    m = mesh.copy()
    m.apply_transform(np.asarray(transform, dtype=float))

    lo = m.bounds[0]
    hi = m.bounds[1]
    # candidate columns: integer cells whose center (i+0.5, j+0.5) lies in [lo, hi]
    i_min = max(0, int(np.floor(lo[0] - 0.5)))
    i_max = min(nx - 1, int(np.ceil(hi[0] - 0.5)))
    j_min = max(0, int(np.floor(lo[1] - 0.5)))
    j_max = min(ny - 1, int(np.ceil(hi[1] - 0.5)))
    if i_min > i_max or j_min > j_max:
        _logger.warning(
            "Imported mesh footprint is entirely outside the domain "
            "(mesh XY bounds %s..%s, grid %dx%d). Nothing voxelized — check "
            "anchor/rotation/units.", lo[:2], hi[:2], nx, ny,
        )
        return np.empty((0, 3), dtype=np.int64)

    ii, jj = np.meshgrid(np.arange(i_min, i_max + 1), np.arange(j_min, j_max + 1), indexing="ij")
    centers_x = ii.ravel() + 0.5
    centers_y = jj.ravel() + 0.5
    n_rays = centers_x.size

    z_below = float(lo[2]) - 1.0
    origins = np.column_stack([centers_x, centers_y, np.full(n_rays, z_below)])
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (n_rays, 1))

    locations, index_ray, _ = m.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )

    occupied = []
    clipped = 0
    odd_columns = 0
    # group hit z-values by ray
    hits_by_ray = {}
    for loc, r in zip(locations, index_ray):
        hits_by_ray.setdefault(int(r), []).append(float(loc[2]))

    for r, zlist in hits_by_ray.items():
        zsorted = sorted(zlist)
        i = int(ii.ravel()[r])
        j = int(jj.ravel()[r])
        spans = []
        if len(zsorted) % 2 == 0:
            for a, b in zip(zsorted[0::2], zsorted[1::2]):
                spans.append((a, b))
        else:
            odd_columns += 1
            spans.append((zsorted[0], zsorted[-1]))
        for a, b in spans:
            k0 = int(np.floor(a + 0.5))
            k1 = int(np.floor(b + 0.5))
            for k in range(k0, k1):
                if 0 <= k < nz:
                    occupied.append((i, j, k))
                else:
                    clipped += 1

    if clipped:
        _logger.warning("Clipped %d imported voxel(s) above the grid height.", clipped)
    if odd_columns:
        _logger.warning(
            "%d column(s) had a non-watertight mesh (odd ray-hit count); filled "
            "between first and last hit as a fallback.", odd_columns,
        )
    if not occupied:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(sorted(set(occupied)), dtype=np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_voxelize.py -v`
Expected: PASS (4 passed). If `m.ray.intersects_location` is slow/missing, ensure
`trimesh` ray backend is available (the pure-Python `ray_triangle` backend ships
with trimesh; no extra install needed).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/voxelize.py tests/importer/test_voxelize.py
git commit -m "feat(importer): column z-ray mesh voxelizer with watertight fallback"
```

---

## Task 5: OBJ loader + role routing

**Files:**
- Create: `src/voxcity/importer/loader.py`
- Test: `tests/importer/test_loader.py`

`load_obj_groups(obj_path, swap_yz=False)` returns a list of `(name, trimesh.Trimesh)`
— one per OBJ object/group (using `trimesh.load(..., split_object=True, group_material=False)`),
falling back to a single `("imported_building_1", mesh)` when the file has no
named groups. `classify_roles(names, roles)` maps each name to a role
(`"building"` by default), and `select_building_groups(...)` filters to building
groups, logging skipped non-building groups.

- [ ] **Step 1: Write the failing test**

`tests/importer/test_loader.py`:
```python
import numpy as np
import trimesh

from voxcity.importer.loader import (
    load_obj_groups, classify_roles, select_building_groups,
)


def _write_two_group_obj(path):
    a = trimesh.creation.box(extents=(2, 2, 2))
    a.apply_translation((1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1))
    b.apply_translation((5, 5, 0.5))
    scene = trimesh.Scene()
    scene.add_geometry(a, geom_name="Building_A")
    scene.add_geometry(b, geom_name="Window_1")
    scene.export(path, file_type="obj")


def test_load_groups_returns_named_meshes(tmp_path):
    p = tmp_path / "two.obj"
    _write_two_group_obj(p)
    groups = load_obj_groups(p)
    names = {n for n, _ in groups}
    assert "Building_A" in names
    assert "Window_1" in names
    for _, mesh in groups:
        assert isinstance(mesh, trimesh.Trimesh)


def test_classify_roles_defaults_to_building():
    roles = classify_roles(["Building_A", "Window_1"], roles=None)
    assert roles["Building_A"] == "building"
    assert roles["Window_1"] == "building"  # no mapping -> default building


def test_classify_roles_applies_mapping():
    roles = classify_roles(["Building_A", "Window_1"], roles={"Window_1": "window"})
    assert roles["Building_A"] == "building"
    assert roles["Window_1"] == "window"


def test_select_building_groups_skips_non_building(tmp_path, caplog):
    p = tmp_path / "two.obj"
    _write_two_group_obj(p)
    groups = load_obj_groups(p)
    selected = select_building_groups(groups, roles={"Window_1": "window"})
    names = {n for n, _ in selected}
    assert names == {"Building_A"}


def test_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_obj_groups(tmp_path / "nope.obj")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.importer.loader'`

- [ ] **Step 3: Write minimal implementation**

`src/voxcity/importer/loader.py`:
```python
"""Load OBJ files into named groups and route groups to roles."""
from __future__ import annotations

import os

import numpy as np
import trimesh

from ..utils.logging import get_logger

_logger = get_logger(__name__)


def load_obj_groups(obj_path, swap_yz=False):
    """Return a list of (name, Trimesh) for each object/group in the OBJ."""
    obj_path = os.fspath(obj_path)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    loaded = trimesh.load(obj_path, process=False, split_object=True, group_material=False)

    groups = []
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"OBJ contains no mesh geometry: {obj_path}")
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                groups.append((str(name), geom))
    elif isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) == 0:
            raise ValueError(f"OBJ contains no mesh geometry: {obj_path}")
        groups.append(("imported_building_1", loaded))
    else:
        raise ValueError(f"Unsupported OBJ content type: {type(loaded)!r}")

    if swap_yz:
        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        groups = [(n, m.copy().apply_transform(swap) or m) for n, m in groups]
        # apply_transform mutates in place and returns the mesh; rebuild cleanly:
        groups = [(n, m) for n, m in groups]

    return groups


def classify_roles(names, roles=None):
    """Map each group name to a role; unmapped names default to 'building'."""
    roles = roles or {}
    return {name: roles.get(name, "building") for name in names}


def select_building_groups(groups, roles=None):
    """Keep only building-role groups; log skipped non-building groups."""
    name_role = classify_roles([n for n, _ in groups], roles=roles)
    selected = []
    skipped = []
    for name, mesh in groups:
        if name_role[name] == "building":
            selected.append((name, mesh))
        else:
            skipped.append((name, name_role[name]))
    for name, role in skipped:
        _logger.info(
            "Skipping OBJ group '%s' (role=%s): geometry-driven %s mapping is not "
            "implemented in this version. Use window_ratio via voxcity.utils.material "
            "for procedural windows.", name, role, role,
        )
    return selected
```

NOTE: the `swap_yz` block above is awkward; replace with a clean implementation:
```python
    if swap_yz:
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=float)
        out = []
        for n, m in groups:
            mc = m.copy()
            mc.apply_transform(swap)
            out.append((n, mc))
        groups = out
```
Use this clean version in the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_loader.py -v`
Expected: PASS (5 passed). If trimesh does not preserve `geom_name` on OBJ
round-trip, adjust `_write_two_group_obj` to write groups via `o <name>` lines or
accept trimesh's auto names; ensure at least two distinct named groups are
returned and the role test still distinguishes them.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/loader.py tests/importer/test_loader.py
git commit -m "feat(importer): OBJ group loader + role routing (skip non-building)"
```

---

## Task 6: Integrator (stamp voxels + update metadata)

**Files:**
- Create: `src/voxcity/importer/integrate.py`
- Test: `tests/importer/test_integrate.py`

`stamp_buildings(voxcity, occupied_by_name, building_value=-3, overwrite=True)`
grows the Z dimension if needed, writes building voxels, assigns a new unique ID
per building group, updates `building_height_grid` (top height per column in m)
and `min_heights` (append vertical spans), records provenance, and returns the
updated VoxCity object.

- [ ] **Step 1: Write the failing test**

`tests/importer/test_integrate.py`:
```python
import numpy as np

from voxcity.importer.integrate import stamp_buildings
from tests.importer.conftest import make_flat_voxcity

BUILDING_CODE = -3
GROUND_CODE = -1


def test_stamps_voxels_and_assigns_new_id():
    vc = make_flat_voxcity(nx=10, ny=10, nz=6, meshsize=1.0)
    # one building occupying column (2,3) at k=1,2,3
    occ = {"b1": np.array([[2, 3, 1], [2, 3, 2], [2, 3, 3]], dtype=np.int64)}
    out = stamp_buildings(vc, occ)
    assert out.voxels.classes[2, 3, 1] == BUILDING_CODE
    assert out.voxels.classes[2, 3, 3] == BUILDING_CODE
    # ground untouched
    assert out.voxels.classes[2, 3, 0] == GROUND_CODE
    # new id assigned at that column
    assert out.buildings.ids[2, 3] == 1
    # height grid = top k * meshsize (k=3 -> 3.0... top span end)
    assert out.buildings.heights[2, 3] > 0


def test_grows_z_when_taller_than_grid():
    vc = make_flat_voxcity(nx=8, ny=8, nz=4, meshsize=1.0)
    occ = {"tower": np.array([[1, 1, k] for k in range(1, 7)], dtype=np.int64)}
    out = stamp_buildings(vc, occ)
    assert out.voxels.classes.shape[2] >= 7
    assert out.voxels.classes[1, 1, 6] == BUILDING_CODE


def test_overwrite_false_yields_to_existing():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    vc.voxels.classes[1, 1, 1] = BUILDING_CODE  # pre-existing building
    occ = {"b": np.array([[1, 1, 1], [1, 1, 2]], dtype=np.int64)}
    out = stamp_buildings(vc, occ, overwrite=False)
    # existing cell stays building, new cell added
    assert out.voxels.classes[1, 1, 1] == BUILDING_CODE
    assert out.voxels.classes[1, 1, 2] == BUILDING_CODE


def test_unique_ids_per_group_above_existing():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    vc.buildings.ids[0, 0] = 7  # existing max id
    occ = {
        "a": np.array([[2, 2, 1]], dtype=np.int64),
        "b": np.array([[3, 3, 1]], dtype=np.int64),
    }
    out = stamp_buildings(vc, occ)
    ids = {int(out.buildings.ids[2, 2]), int(out.buildings.ids[3, 3])}
    assert ids == {8, 9}


def test_provenance_recorded():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    occ = {"a": np.array([[2, 2, 1]], dtype=np.int64)}
    out = stamp_buildings(vc, occ, source="model.obj")
    assert "imported_buildings" in out.extras
    man = out.extras["imported_buildings"][-1]
    assert man["source"] == "model.obj"
    assert "id_map" in man
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_integrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.importer.integrate'`

- [ ] **Step 3: Write minimal implementation**

`src/voxcity/importer/integrate.py`:
```python
"""Stamp voxelized buildings into a VoxCity object and update metadata grids."""
from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

_logger = get_logger(__name__)

BUILDING_CODE = -3


def _spans_from_ks(ks):
    """Return list of [start_k, end_k_exclusive] contiguous spans from sorted ks."""
    ks = sorted(set(int(k) for k in ks))
    spans = []
    if not ks:
        return spans
    start = prev = ks[0]
    for k in ks[1:]:
        if k == prev + 1:
            prev = k
        else:
            spans.append([start, prev + 1])
            start = prev = k
    spans.append([start, prev + 1])
    return spans


def stamp_buildings(voxcity, occupied_by_name, building_value=BUILDING_CODE,
                    overwrite=True, source=None, manifest_extra=None):
    """Write occupied cells into voxcity and update derived metadata grids."""
    classes = voxcity.voxels.classes
    nx, ny, nz = classes.shape
    meshsize = float(voxcity.voxels.meta.meshsize)

    # 1. grow Z if needed
    max_k = -1
    for occ in occupied_by_name.values():
        if len(occ):
            max_k = max(max_k, int(occ[:, 2].max()))
    if max_k >= nz:
        pad = np.zeros((nx, ny, max_k + 1 - nz), dtype=classes.dtype)
        classes = np.concatenate([classes, pad], axis=2)
        voxcity.voxels.classes = classes
        nz = classes.shape[2]

    ids_grid = voxcity.buildings.ids
    heights_grid = voxcity.buildings.heights
    min_heights = voxcity.buildings.min_heights

    next_id = int(ids_grid.max()) + 1 if ids_grid.size else 1
    collisions = 0
    id_map = {}

    for name, occ in occupied_by_name.items():
        if not len(occ):
            continue
        bid = next_id
        next_id += 1
        id_map[name] = bid

        # group occupied cells by column for metadata
        cols = {}
        for i, j, k in occ:
            i, j, k = int(i), int(j), int(k)
            if k < 0 or k >= nz or i < 0 or i >= nx or j < 0 or j >= ny:
                continue
            current = classes[i, j, k]
            if overwrite or current == 0:
                if current not in (0,) and overwrite:
                    collisions += 1
                classes[i, j, k] = building_value
                cols.setdefault((i, j), []).append(k)

        for (i, j), ks in cols.items():
            spans = _spans_from_ks(ks)
            top_k = max(s[1] for s in spans)  # exclusive end
            ids_grid[i, j] = bid
            heights_grid[i, j] = max(float(heights_grid[i, j]), top_k * meshsize)
            cell = min_heights[i, j]
            if not isinstance(cell, list):
                cell = []
            for a, b in spans:
                cell.append([a * meshsize, b * meshsize])
            min_heights[i, j] = cell

    if collisions:
        _logger.info("Imported buildings overwrote %d existing non-air voxel(s).", collisions)

    manifest = {"source": source, "id_map": id_map}
    if manifest_extra:
        manifest.update(manifest_extra)
    voxcity.extras.setdefault("imported_buildings", []).append(manifest)

    return voxcity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_integrate.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/integrate.py tests/importer/test_integrate.py
git commit -m "feat(importer): stamp buildings into VoxCity + update metadata grids"
```

---

## Task 7: Orchestration — add_buildings_from_obj

**Files:**
- Create: `src/voxcity/importer/rhino_obj.py`
- Modify: `src/voxcity/importer/__init__.py` (restore the real import)
- Test: `tests/importer/test_add_buildings_from_obj.py`

`add_buildings_from_obj(...)` ties everything together: validate inputs, load OBJ
groups, select building groups by role, build the transform, voxelize each group,
stamp into a copied VoxCity object, and return it. The `meshlib` backend is
dispatched lazily and raises a clear `ImportError` if unavailable (not
implemented in v1).

- [ ] **Step 1: Write the failing test**

`tests/importer/test_add_buildings_from_obj.py`:
```python
import numpy as np
import pytest

from voxcity.importer import add_buildings_from_obj
from voxcity.importer.transform import grid_geom_from_voxcity
from tests.importer.conftest import make_flat_voxcity

BUILDING_CODE = -3


def test_end_to_end_box_import(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    # anchor the model origin a few meters into the domain
    proj_origin = geom["origin"]
    obj = box_obj_factory(origin=(0.0, 0.0, 0.0), size=(3.0, 3.0, 4.0), name="b1")
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(proj_origin[0]), float(proj_origin[1])),
        anchor_elevation=0.0,
        anchor_model_point=(0.0, 0.0, 0.0),
        move=(5.0, 5.0, 0.0),   # 5 m east, 5 m north
        rotation=0.0, units="m",
    )
    # building voxels should appear near columns (5..7 north, 5..7 east), above ground
    sub = out.voxels.classes[5:8, 5:8, 1:5]
    assert np.any(sub == BUILDING_CODE)
    # ids assigned somewhere
    assert out.buildings.ids.max() >= 1


def test_missing_file_raises(flat_voxcity, tmp_path):
    with pytest.raises(FileNotFoundError):
        add_buildings_from_obj(
            flat_voxcity, tmp_path / "nope.obj",
            anchor_lonlat=(0.0, 0.0), anchor_elevation=0.0,
        )


def test_invalid_units_raises(flat_voxcity, box_obj_factory):
    obj = box_obj_factory()
    with pytest.raises(ValueError, match="Unknown units"):
        add_buildings_from_obj(
            flat_voxcity, obj, anchor_lonlat=(0.0, 0.0),
            anchor_elevation=0.0, units="furlong",
        )


def test_meshlib_backend_not_installed_raises(flat_voxcity, box_obj_factory, monkeypatch):
    obj = box_obj_factory()
    # simulate meshlib missing
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("meshlib"):
            raise ImportError("no meshlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="meshlib"):
        add_buildings_from_obj(
            flat_voxcity, obj, anchor_lonlat=(0.0, 0.0),
            anchor_elevation=0.0, backend="meshlib",
        )


def test_original_object_not_mutated(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    before = vc.voxels.classes.copy()
    geom = grid_geom_from_voxcity(vc)
    obj = box_obj_factory(size=(3.0, 3.0, 4.0))
    add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, move=(5.0, 5.0, 0.0),
    )
    assert np.array_equal(vc.voxels.classes, before)  # input untouched (copy returned)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_add_buildings_from_obj.py -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` (rhino_obj not present)

- [ ] **Step 3: Write minimal implementation**

`src/voxcity/importer/rhino_obj.py`:
```python
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
```

Restore `src/voxcity/importer/__init__.py`:
```python
"""VoxCity importer subpackage: import external 3D geometry into VoxCity models."""

from .rhino_obj import add_buildings_from_obj

__all__ = ["add_buildings_from_obj"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_add_buildings_from_obj.py -v`
Expected: PASS (5 passed). If `test_end_to_end_box_import` finds no building
voxels, print `np.argwhere(out.voxels.classes == -3)` to inspect placement and
confirm the anchor/move math; adjust the asserted sub-cube to the actual cells
(the transform tests already pin the math, so cells should land near (5,5)).

- [ ] **Step 5: Run the full importer suite**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/voxcity/importer/rhino_obj.py src/voxcity/importer/__init__.py tests/importer/test_add_buildings_from_obj.py
git commit -m "feat(importer): add_buildings_from_obj orchestration + backend dispatch"
```

---

## Task 8: Documentation (Rhino export guide) + README

**Files:**
- Create: `docs/rhino_obj_import.md`
- Modify: `README.md` (add a short "Importing Rhino models" subsection under Usage)

- [ ] **Step 1: Write the user guide**

`docs/rhino_obj_import.md`:
```markdown
# Importing Rhino Models (OBJ) into VoxCity

`voxcity.importer.add_buildings_from_obj` adds buildings authored in Rhino to an
existing VoxCity model. Buildings are voxelized directly from their 3D mesh form;
terrain, land cover, and trees come from the base model.

## Preparing the model in Rhino

1. **Model buildings as closed solids.** Watertight geometry voxelizes most
   reliably (non-watertight meshes are repaired/filled with a warning).
2. **One building per object/layer.** Each OBJ object/group becomes one building
   (its own ID + name).
3. **Choose an anchor.** Pick one identifiable point in the model and record:
   - its model coordinates -> `anchor_model_point` (default `(0,0,0)`),
   - the real-world `(lon, lat)` of that point -> `anchor_lonlat`,
   - its real-world elevation in meters -> `anchor_elevation`.
4. **Rotation.** `rotation=0` means model **+Y points true north** and **+X
   points east**. Otherwise pass the angle (degrees).
5. **Units.** Check Rhino `Units` and pass `units` ("m", "cm", "mm", "ft", "in").
6. **Export OBJ.** `File > Export Selected > .obj`, keep **Z up**, export with
   object names/groups. If your export is Y-up, pass `z_up=False`.

## Windows / glazing (current behavior)

Model opaque mass as **solids**; model windows as **planar surfaces (not solids)**
on a layer such as `Window`. In this version, non-building layers are detected and
**skipped** — pass `roles={"Window": "window"}` to mark them. For windows today,
use the procedural material utilities on the imported buildings:

```python
from voxcity.utils.material import set_building_material_by_id, get_material_dict
mat = get_material_dict()
set_building_material_by_id(vc.voxels.classes, vc.buildings.ids, ids=[1, 2],
                           mark=mat["concrete"], window_ratio=0.4, glass_id=mat["glass"])
```

Geometry-driven windows (mapping window surfaces directly to glass voxels) are
planned (see the design spec, Path B).

## Example

```python
from voxcity.generator import get_voxcity
from voxcity.importer import add_buildings_from_obj
from voxcity.exporter.obj import export_obj

vc = get_voxcity(rectangle_vertices, meshsize=2.0)
vc = add_buildings_from_obj(
    vc, "design.obj",
    anchor_lonlat=(139.7536, 35.6841),
    anchor_elevation=12.0,
    anchor_model_point=(0.0, 0.0, 0.0),
    rotation=0.0, move=(0.0, 0.0, 0.0), units="m",
)
export_obj(vc, "output", "with_imported_building")
```

Iterate on `rotation`/`move` and re-export to verify placement visually.
```

- [ ] **Step 2: Add README subsection**

In `README.md`, after the "OBJ Files:" export section (around the Rhino render
image), add:

```markdown
#### Importing Rhino Models (OBJ):

You can import buildings authored in Rhino into a VoxCity model:

```python
from voxcity.importer import add_buildings_from_obj

voxcity = add_buildings_from_obj(
    voxcity, "design.obj",
    anchor_lonlat=(139.7536, 35.6841),  # world (lon, lat) of the model anchor
    anchor_elevation=12.0,              # world elevation (m) of the anchor
    rotation=0.0, units="m",
)
```

See [docs/rhino_obj_import.md](docs/rhino_obj_import.md) for the full Rhino export guide.
```

- [ ] **Step 3: Verify docs render (no test) and commit**

```bash
git add docs/rhino_obj_import.md README.md
git commit -m "docs(importer): Rhino OBJ export guide + README usage"
```

---

## Task 9: Optional meshlib backend (skipped if absent)

**Files:**
- Modify: `src/voxcity/importer/voxelize.py` (add `voxelize_mesh_meshlib`)
- Modify: `src/voxcity/importer/rhino_obj.py` (dispatch to it instead of NotImplementedError)
- Test: `tests/importer/test_voxelize_meshlib.py`

This task is optional and only runs where `meshlib` is installed. Keep it small:
SDF voxelization at pitch=meshsize aligned to the grid, returning the same
`(i,j,k)` format as the trimesh path.

- [ ] **Step 1: Write the skip-guarded test**

`tests/importer/test_voxelize_meshlib.py`:
```python
import numpy as np
import pytest
import trimesh

meshlib = pytest.importorskip("meshlib")

from voxcity.importer.voxelize import voxelize_mesh, voxelize_mesh_meshlib


def _box(min_corner, size):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.array(min_corner) + np.array(size) / 2.0)
    return m


def test_meshlib_matches_trimesh_on_cube():
    mesh = _box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    a = set(map(tuple, voxelize_mesh(mesh, np.eye(4), (10, 10, 10))))
    b = set(map(tuple, voxelize_mesh_meshlib(mesh, np.eye(4), (10, 10, 10))))
    # interiors should agree; allow boundary differences of a thin shell
    assert (1, 1, 1) in b
    assert len(a ^ b) <= len(a) * 0.25
```

- [ ] **Step 2: Run test to verify it fails or skips**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_voxelize_meshlib.py -v`
Expected: SKIPPED (meshlib not installed) or FAIL (`voxelize_mesh_meshlib` missing) if it is.

- [ ] **Step 3: Implement `voxelize_mesh_meshlib`**

Add to `src/voxcity/importer/voxelize.py`:
```python
def voxelize_mesh_meshlib(mesh, transform, grid_shape):
    """Voxelize via MeshLib SDF (optional backend). Returns (N,3) int (i,j,k)."""
    import meshlib.mrmeshpy as mr  # lazy

    nx, ny, nz = grid_shape
    m = mesh.copy()
    m.apply_transform(np.asarray(transform, dtype=float))

    ml_mesh = mr.Mesh()
    verts = [mr.Vector3f(float(x), float(y), float(z)) for x, y, z in m.vertices]
    tris = m.faces.astype(np.int64)
    ml_mesh = mr.meshFromFacesVerts(tris.tolist(), verts)  # API per meshlib version

    # signed distance volume at pitch=1 (index space)
    settings = mr.MeshToVolumeSettings()
    settings.voxelSize = mr.Vector3f(1.0, 1.0, 1.0)
    vol = mr.meshToVolume(ml_mesh, settings)

    # extract occupied voxels where distance <= 0 (inside)
    occupied = []
    dims = vol.dims
    for i in range(min(dims.x, nx)):
        for j in range(min(dims.y, ny)):
            for k in range(min(dims.z, nz)):
                if vol.data.get(mr.Vector3i(i, j, k)) <= 0:
                    occupied.append((i, j, k))
    if not occupied:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(sorted(set(occupied)), dtype=np.int64)
```

NOTE: MeshLib's exact Python API varies by version. The implementer must adapt
`meshFromFacesVerts` / `meshToVolume` / volume access to the installed meshlib
version, keeping the **return contract identical** to `voxelize_mesh`
(an `(N,3)` int array of in-bounds `(i,j,k)`). The skip-guarded test verifies
parity with the trimesh path on a cube.

In `rhino_obj.py`, replace the meshlib `NotImplementedError` block so it selects
the voxelizer:
```python
    from .voxelize import voxelize_mesh, voxelize_mesh_meshlib
    _voxelize = voxelize_mesh_meshlib if backend == "meshlib" else voxelize_mesh
```
and call `_voxelize(mesh, M, grid_shape)` in the loop. Remove the
`NotImplementedError`; keep the `ImportError` guard for when meshlib is absent.

- [ ] **Step 4: Run test**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/test_voxelize_meshlib.py -v`
Expected: PASS where meshlib is installed; SKIPPED otherwise. Re-run the full
`tests/importer/` suite to confirm no regressions, and `test_add_buildings_from_obj.py::test_meshlib_backend_not_installed_raises`
still passes (the ImportError guard remains).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/importer/voxelize.py src/voxcity/importer/rhino_obj.py tests/importer/test_voxelize_meshlib.py
git commit -m "feat(importer): optional meshlib SDF voxelization backend"
```

---

## Final verification

- [ ] Run the full importer suite:
  `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/importer/ -v`
- [ ] Run a broader smoke test to ensure no import-time breakage elsewhere:
  `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "import voxcity; from voxcity.importer import add_buildings_from_obj; print('ok')"`
- [ ] Confirm `git status` is clean and all tasks committed.

---

## Spec coverage notes

- Anchor + rotation + horizontal/vertical move + units → Task 3 (transform).
- Direct 3D mesh voxelization (trimesh default) → Task 4; optional meshlib → Task 9.
- OBJ format + per-building identity → Task 5.
- Role router skips non-building/window layers (v1 requirement) → Task 5.
- Voxels + derived metadata (ids, heights, min_heights), Z growth, overwrite,
  provenance → Task 6.
- Vertical datum relative to min(DEM) + ground offset → Task 3 (z math) + Task 6.
- Error handling (missing file, empty OBJ, bad units, meshlib absent, all-clipped)
  → Tasks 4, 5, 7.
- Path A procedural-window compatibility + Rhino export guide → Task 8.
- `rectangle_vertices` reconstruction (works for rotated domains) → Task 3.
