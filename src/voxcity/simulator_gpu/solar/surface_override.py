"""Externally supplied per-surface normals and patch identity.

The solar solver derives its surfaces from the occupancy grid and gives each one
of six axis-aligned normals. A caller that knows the real geometry -- because it
built the voxel grid from polygons -- can supply a table instead, and the solver
will use the true normal and ignore occlusion by the surface's own patch.

The table is duck-typed: any object exposing ``cell``, ``face``, ``origin``,
``normal``, ``patch``, ``area`` arrays and a ``convention`` string is accepted.
That is deliberate -- the caller (voxcitywrapper) defines its own class and never
imports this module, so released voxcity and the wrapper stay import-decoupled
in both directions. The dataclass below exists for voxcity's own tests.

Frame: voxel-array axis order. cell[:, 0] indexes voxels.classes axis 0,
cell[:, 1] axis 1, cell[:, 2] is z; origin is index * meshsize along the same
axes. This equals the solver's world frame (Domain nx = shape[0]) and is NOT the
wrapper's scene frame, which has x = col -- the wrapper swaps components before
building the table. The convention string names this frame and is checked on
entry: voxcity has changed which voxel-array axis becomes scene x between
releases, and a bare (S, 3) of integers would let a table built against the
other convention through silently.

This module deliberately holds no Taichi. It is a data contract plus two pure
NumPy derivations.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SURFACE_TABLE_CONVENTION = "voxel-array[row,col,z]*meshsize"

NO_PATCH = -1

# How deep a patch id is carried into solid. The fill exists to cover the
# one-or-two-cell staircase a slanted surface leaves behind, not to label a
# building's whole core: a ray leaving a front face outward never enters the
# interior, and one grazing along the surface crosses at most a step or two.
# Bounding it keeps this O(N) instead of O(depth x N) -- unbounded, a solid
# terrain column costs seconds on a city-sized grid (measured: ~5.5s for a
# 59-cell column, ~10.6s for a ~100-cell block, both on grids this solver
# actually runs at).
MAX_PATCH_FILL_DEPTH = 3

# Signature sentinel for "no override supplied" -- kept next to NO_PATCH
# because both mean "behave as today", one at table granularity and one at
# per-surface granularity.
NO_SURFACE_OVERRIDE = "none"

_FIELDS = ("cell", "face", "origin", "normal", "patch", "area")

# Which fields are (S, 3) vectors rather than (S,) scalars -- the single
# source of truth the shape-check loop reads from, so adding a 7th field to
# _FIELDS can't be done without also deciding its shape here.
_VECTOR3_FIELDS = frozenset({"cell", "origin", "normal"})

# Sentinel distinct from any real attribute value, including None, so
# `getattr(table, name, _MISSING) is _MISSING` unambiguously means "absent".
_MISSING = object()


@dataclass
class SurfaceOverride:
    """Reference table type: one row per (voxel cell, owning polygon patch).

    cell    (S, 3) int32    the voxel this surface belongs to
    face    (S,)   int8     IUP/IDOWN/INORTH/... the exposed face the ray leaves
    origin  (S, 3) float32  ray origin, array-frame metres
    normal  (S, 3) float32  unit surface normal, replaces the axis normal
    patch   (S,)   int32    coplanar patch id; NO_PATCH means "behave as today"
    area    (S,)   float32  true surface area crossing this cell (m^2)
    """

    cell: np.ndarray
    face: np.ndarray
    origin: np.ndarray
    normal: np.ndarray
    patch: np.ndarray
    area: np.ndarray
    convention: str = SURFACE_TABLE_CONVENTION

    def __len__(self) -> int:
        return len(self.cell)


def _missing_fields(table) -> list:
    """Names of required attributes ``table`` does not have at all.

    Distinct from "present but wrong shape", which the caller checks next --
    this only answers "can I even call getattr(table, name) and get
    something back".
    """
    return [name for name in _FIELDS if getattr(table, name, _MISSING) is _MISSING]


def _length(table) -> Optional[int]:
    """Row count from a table, or ``None`` if it lacks ``cell`` entirely.

    ``None`` (attribute absent -- a malformed table) is kept distinct from
    ``0`` (attribute present, but an empty array -- a legitimately empty
    override) specifically so ``surface_override_signature`` cannot conflate
    "malformed" with "none supplied" and collapse a broken table onto the
    ``NO_SURFACE_OVERRIDE`` sentinel. Called before validation may have run,
    so it must not assume ``cell`` exists or has a sane shape.
    """
    cell = getattr(table, "cell", _MISSING)
    if cell is _MISSING:
        return None
    return len(np.asarray(cell))


def validate_surface_override(table, grid_shape: Tuple[int, int, int]) -> None:
    """Reject a table this solver cannot safely index.

    Ordering rule: call this before trusting ``surface_override_signature``
    on anything other than the ``table is None`` case. A table that passes
    here is guaranteed to have every required attribute and a well-defined
    length, which is exactly what the signature needs to be content-sensitive
    instead of raising or silently degrading.
    """
    conv = getattr(table, "convention", None)
    if conv != SURFACE_TABLE_CONVENTION:
        raise ValueError(
            f"surface table convention is {conv!r}, but this voxcity expects "
            f"{SURFACE_TABLE_CONVENTION!r}. The table was built against a "
            f"different voxel-axis convention and its cell indices would "
            f"silently address the wrong voxels.")

    missing = _missing_fields(table)
    if missing:
        raise ValueError(
            f"surface table is missing required attribute(s) {missing}; "
            f"expected all of {list(_FIELDS)}")

    n = _length(table)
    for name in _FIELDS:
        expected = (n, 3) if name in _VECTOR3_FIELDS else (n,)
        got = np.asarray(getattr(table, name)).shape
        if got != expected:
            raise ValueError(
                f"surface table arrays must all have the same length; "
                f"{name} has shape {got}, expected {expected}")

    if n == 0:
        return

    cell = np.asarray(table.cell)
    nx, ny, nz = grid_shape
    lo, hi = cell.min(axis=0), cell.max(axis=0)
    if (lo < 0).any() or hi[0] >= nx or hi[1] >= ny or hi[2] >= nz:
        raise ValueError(
            f"surface table cell indices {lo.tolist()}..{hi.tolist()} fall "
            f"outside the grid {grid_shape}")

    lens = np.linalg.norm(np.asarray(table.normal, dtype=np.float64), axis=1)
    if not np.allclose(lens, 1.0, atol=1e-3):
        raise ValueError(
            f"surface table normals must be unit vectors; worst deviation is "
            f"{float(np.abs(lens - 1.0).max()):.4f}")


def surface_override_signature(table) -> str:
    """Cache-key contribution. ``None`` gets its own constant,
    ``NO_SURFACE_OVERRIDE``, so cache entries written before this feature
    existed stay valid for the no-override case.

    Ordering rule: call ``validate_surface_override`` before trusting this
    signature. A malformed table -- present but missing a required attribute
    -- raises ``ValueError`` here rather than returning
    ``NO_SURFACE_OVERRIDE``: collapsing the two would let a broken table
    masquerade as "none supplied" and reuse another run's cached result.

    The derived cell-patch grid is NOT hashed: it is a pure function of the
    table and the voxel content, both already in the key.
    """
    if table is None:
        return NO_SURFACE_OVERRIDE

    missing = _missing_fields(table)
    if missing:
        raise ValueError(
            f"surface table is missing required attribute(s) {missing}; "
            f"cannot compute a signature for a malformed table -- call "
            f"validate_surface_override first.")

    if _length(table) == 0:
        return NO_SURFACE_OVERRIDE

    h = hashlib.blake2b(digest_size=16)
    h.update(getattr(table, "convention", "").encode("utf-8"))
    for name in _FIELDS:
        a = np.ascontiguousarray(np.asarray(getattr(table, name)))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def build_cell_patch_grid(table, is_solid: np.ndarray) -> np.ndarray:
    """Per-cell patch id for the skip DDA, interior cells filled up to a
    bounded depth.

    Surface rows are scattered onto their cells first (a corner cell touched
    by two patches keeps the last row's id -- either wall is a defensible
    owner and the mock behaved the same way). Interior solid cells then
    inherit the patch of the nearest patched cell by breadth-first dilation,
    capped at ``MAX_PATCH_FILL_DEPTH`` rounds. Without any interior fill, a
    ray skipping its own staircase cell immediately hits the patchless
    interior cell behind it and blocks anyway; the mock's headline numbers
    were measured with interior fill (its build_cell_patch). Cells deeper
    than the bound simply stay ``NO_PATCH``, which is the safe direction: an
    un-patched cell is opaque, so the worst case is a ray blocking where it
    would otherwise have passed, not the reverse.
    """
    solid = np.asarray(is_solid, dtype=bool)
    grid = np.full(solid.shape, NO_PATCH, dtype=np.int32)
    n = _length(table)
    if not n:
        return grid

    cell = np.asarray(table.cell)
    patch = np.asarray(table.patch)
    # Any negative patch id counts as "no patch" here, not just NO_PATCH
    # specifically -- a caller need not use exactly this constant.
    has_patch = patch >= 0
    grid[cell[has_patch, 0], cell[has_patch, 1], cell[has_patch, 2]] = patch[has_patch]

    # Bounded BFS dilation: each round hands ids one cell deeper along a
    # face-adjacent neighbour, for at most MAX_PATCH_FILL_DEPTH rounds -- see
    # the constant's comment for why that bound is enough. A round that makes
    # no progress means the remaining unfilled solid cells have no patched
    # neighbour anywhere yet (an isolated interior island with no surface
    # table entry touching it), so stop early rather than spend the rest of
    # the budget doing nothing.
    todo = solid & (grid == NO_PATCH)
    for _round in range(MAX_PATCH_FILL_DEPTH):
        if not todo.any():
            break
        progressed = False
        for axis in range(3):
            for shift in (1, -1):
                src = np.roll(grid, shift, axis=axis)
                # roll wraps around; the wrapped-in slice is not a real
                # neighbour, so blank it out before using it as a source.
                edge = [slice(None)] * 3
                edge[axis] = slice(0, 1) if shift == 1 else slice(-1, None)
                src[tuple(edge)] = NO_PATCH
                take = todo & (src != NO_PATCH)
                if take.any():
                    grid[take] = src[take]
                    todo &= grid == NO_PATCH
                    progressed = True
        if not progressed:
            break
    return grid
