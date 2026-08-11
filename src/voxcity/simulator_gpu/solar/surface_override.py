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
from typing import Tuple

import numpy as np

SURFACE_TABLE_CONVENTION = "voxel-array[row,col,z]*meshsize"

NO_PATCH = -1

_FIELDS = ("cell", "face", "origin", "normal", "patch", "area")


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


def _length(table) -> int:
    """Row count from a possibly-empty, possibly-duck-typed table.

    Called before validation has run, so this must not assume ``cell`` exists
    or has a sane shape -- ``surface_override_signature`` calls it on tables
    that have not been (and may never be) validated.
    """
    cell = getattr(table, "cell", None)
    if cell is None:
        return 0
    return len(np.asarray(cell))


def validate_surface_override(table, grid_shape: Tuple[int, int, int]) -> None:
    """Reject a table this solver cannot safely index."""
    conv = getattr(table, "convention", None)
    if conv != SURFACE_TABLE_CONVENTION:
        raise ValueError(
            f"surface table convention is {conv!r}, but this voxcity expects "
            f"{SURFACE_TABLE_CONVENTION!r}. The table was built against a "
            f"different voxel-axis convention and its cell indices would "
            f"silently address the wrong voxels.")

    n = _length(table)
    for name, shape in (("cell", (n, 3)), ("face", (n,)), ("origin", (n, 3)),
                        ("normal", (n, 3)), ("patch", (n,)), ("area", (n,))):
        got = np.asarray(getattr(table, name)).shape
        if got != shape:
            raise ValueError(
                f"surface table arrays must all have the same length; "
                f"{name} has shape {got}, expected {shape}")

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


NO_SURFACE_OVERRIDE = "none"


def surface_override_signature(table) -> str:
    """Cache-key contribution. ``None`` gets its own constant so cache entries
    written before this feature existed stay valid for the no-override case.

    The derived cell-patch grid is NOT hashed: it is a pure function of the
    table and the voxel content, both already in the key.
    """
    if table is None or _length(table) == 0:
        return NO_SURFACE_OVERRIDE
    h = hashlib.blake2b(digest_size=16)
    h.update(getattr(table, "convention", "").encode("utf-8"))
    for name in _FIELDS:
        a = np.ascontiguousarray(np.asarray(getattr(table, name)))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def build_cell_patch_grid(table, is_solid: np.ndarray) -> np.ndarray:
    """Per-cell patch id for the skip DDA, interior cells filled.

    Surface rows are scattered onto their cells first (a corner cell touched by
    two patches keeps the last row's id -- either wall is a defensible owner and
    the mock behaved the same way). Interior solid cells then inherit the patch
    of the nearest patched cell by breadth-first dilation. Without the interior
    fill, a ray skipping its own staircase cell immediately hits the patchless
    interior cell behind it and blocks anyway; the mock's headline numbers were
    measured with interior fill (its build_cell_patch).
    """
    solid = np.asarray(is_solid, dtype=bool)
    grid = np.full(solid.shape, NO_PATCH, dtype=np.int32)
    n = _length(table)
    if n == 0:
        return grid

    cell = np.asarray(table.cell)
    patch = np.asarray(table.patch)
    keyed = patch >= 0
    grid[cell[keyed, 0], cell[keyed, 1], cell[keyed, 2]] = patch[keyed]

    # BFS dilation: each pass hands ids one cell deeper along a face-adjacent
    # neighbour. Depth is bounded by the grid's diameter, so this terminates;
    # a pass that makes no progress means the remaining unfilled solid cells
    # have no patched neighbour anywhere (an isolated interior with no
    # surface table entry touching it), so stop rather than spin.
    todo = solid & (grid == NO_PATCH)
    while todo.any():
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
