import time

import numpy as np
import pytest

from voxcity.simulator_gpu.solar.surface_override import (
    SURFACE_TABLE_CONVENTION, NO_PATCH, MAX_PATCH_FILL_DEPTH, SurfaceOverride,
    validate_surface_override, surface_override_signature,
    build_cell_patch_grid,
)


def _minimal(n=2):
    return SurfaceOverride(
        cell=np.array([[1, 2, 3], [1, 2, 4]][:n], dtype=np.int32),
        face=np.array([0, 2][:n], dtype=np.int8),
        origin=np.array([[1.5, 2.5, 4.0], [1.5, 3.0, 4.5]][:n], dtype=np.float32),
        normal=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]][:n], dtype=np.float32),
        patch=np.array([7, 7][:n], dtype=np.int32),
        area=np.array([1.0, 1.0][:n], dtype=np.float32),
    )


class _DuckTable:
    """The wrapper's table type is a different class. Only attributes matter."""

    def __init__(self, src):
        for k in ("cell", "face", "origin", "normal", "patch", "area",
                  "convention"):
            setattr(self, k, getattr(src, k))


class _PartialTable:
    """A table missing a required attribute entirely -- e.g. a caller bug or
    a table built against a stale schema. Distinct from a wrong-shape array:
    this one doesn't have `cell` at all."""

    def __init__(self, src):
        for k in ("face", "origin", "normal", "patch", "area", "convention"):
            setattr(self, k, getattr(src, k))


def test_convention_string_is_the_frozen_literal():
    # Both packages hard-code this literally; a change here is a breaking change.
    assert SURFACE_TABLE_CONVENTION == "voxel-array[row,col,z]*meshsize"


def test_valid_table_passes_validation():
    validate_surface_override(_minimal(), grid_shape=(10, 10, 10))


def test_duck_typed_table_passes_validation():
    validate_surface_override(_DuckTable(_minimal()), grid_shape=(10, 10, 10))


def test_rejects_wrong_convention():
    ov = _minimal()
    ov.convention = "row=x, col=y, z=up"
    with pytest.raises(ValueError, match="convention"):
        validate_surface_override(ov, grid_shape=(10, 10, 10))


def test_rejects_cell_outside_grid():
    ov = _minimal()
    ov.cell[0, 2] = 99
    with pytest.raises(ValueError, match="outside the grid"):
        validate_surface_override(ov, grid_shape=(10, 10, 10))


def test_rejects_non_unit_normal():
    ov = _minimal()
    ov.normal[0] = [0.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="unit"):
        validate_surface_override(ov, grid_shape=(10, 10, 10))


def test_rejects_mismatched_lengths():
    ov = _minimal()
    ov.patch = np.array([7], dtype=np.int32)
    with pytest.raises(ValueError, match="same length"):
        validate_surface_override(ov, grid_shape=(10, 10, 10))


def test_rejects_missing_attribute_with_value_error_not_attribute_error():
    # A default-less getattr() deep in the shape-check loop would raise a
    # bare AttributeError here, which unlike every other rejection in this
    # function doesn't explain the consequence.
    ov = _PartialTable(_minimal())
    with pytest.raises(ValueError, match="missing required attribute"):
        validate_surface_override(ov, grid_shape=(10, 10, 10))


def test_signature_is_stable_content_sensitive_and_duck_typed():
    a, b = _minimal(), _minimal()
    assert surface_override_signature(a) == surface_override_signature(b)
    assert surface_override_signature(_DuckTable(a)) == surface_override_signature(a)
    b.normal[0] = [1.0, 0.0, 0.0]
    assert surface_override_signature(a) != surface_override_signature(b)
    assert surface_override_signature(None) == surface_override_signature(None)
    assert surface_override_signature(None) != surface_override_signature(a)


def test_signature_raises_for_malformed_table_instead_of_masquerading_as_none():
    # A malformed table (missing an attribute) must not collapse onto the
    # same signature as "no override supplied" -- a caller that skips
    # validate_surface_override could otherwise reuse a stale "none" cache
    # entry for a table that is actually broken.
    ov = _PartialTable(_minimal())
    with pytest.raises(ValueError, match="missing required attribute"):
        surface_override_signature(ov)


def test_cell_patch_grid_scatters_and_fills_interior():
    """Interior solid cells inherit the nearest surface's patch. Without this,
    a skipped staircase cell just exposes the patchless interior cell behind
    it and the ray blocks anyway -- the mock's numbers were measured WITH
    interior fill (build_cell_patch in the mock)."""
    is_solid = np.zeros((5, 3, 3), dtype=bool)
    is_solid[1:4, 1, 1] = True                       # a 3-cell-thick slab
    ov = SurfaceOverride(
        cell=np.array([[1, 1, 1], [3, 1, 1]], dtype=np.int32),
        face=np.array([5, 4], dtype=np.int8),        # IWEST face, IEAST face
        origin=np.array([[1.0, 1.5, 1.5], [4.0, 1.5, 1.5]], dtype=np.float32),
        normal=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        patch=np.array([10, 20], dtype=np.int32),
        area=np.array([1.0, 1.0], dtype=np.float32),
    )
    grid = build_cell_patch_grid(ov, is_solid)
    assert grid.shape == is_solid.shape
    assert grid[1, 1, 1] == 10
    assert grid[3, 1, 1] == 20
    assert grid[2, 1, 1] in (10, 20)                 # interior cell got filled
    assert (grid[~is_solid] == NO_PATCH).all()       # air stays patchless


def test_patch_fill_depth_is_bounded():
    """Unbounded dilation measured ~5.5s on a 59-cell solid column and
    ~10.6s on a ~100-cell block at city-scale grid sizes. Depth must be
    capped at MAX_PATCH_FILL_DEPTH; cells beyond the cap stay NO_PATCH,
    which is the safe direction (an unpatched cell is opaque, so the ray
    just blocks where it might otherwise have passed)."""
    thickness = 40  # much thicker than MAX_PATCH_FILL_DEPTH
    is_solid = np.zeros((thickness + 2, 3, 3), dtype=bool)
    is_solid[1:thickness + 1, 1, 1] = True
    ov = SurfaceOverride(
        cell=np.array([[1, 1, 1]], dtype=np.int32),
        face=np.array([5], dtype=np.int8),           # IWEST face
        origin=np.array([[1.0, 1.5, 1.5]], dtype=np.float32),
        normal=np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
        patch=np.array([10], dtype=np.int32),
        area=np.array([1.0], dtype=np.float32),
    )

    start = time.perf_counter()
    grid = build_cell_patch_grid(ov, is_solid)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"dilation took {elapsed:.2f}s; the depth bound is not holding"
    for depth in range(1, MAX_PATCH_FILL_DEPTH + 1):
        assert grid[1 + depth, 1, 1] == 10           # within the bound: filled
    assert grid[1 + MAX_PATCH_FILL_DEPTH + 1, 1, 1] == NO_PATCH  # beyond it: not


def test_bfs_terminates_on_an_unreachable_island():
    """A solid region with no patched cell anywhere in it must not spin --
    this pins the `if not progressed: break` early exit, which a
    well-meaning simplification of the loop could delete without any other
    test noticing."""
    is_solid = np.zeros((10, 3, 3), dtype=bool)
    is_solid[1:4, 1, 1] = True   # patched slab, as in the interior-fill test
    is_solid[7:9, 1, 1] = True   # isolated island: no patch touches it, and
                                  # it is separated from the slab by air (4-6)
    ov = SurfaceOverride(
        cell=np.array([[1, 1, 1], [3, 1, 1]], dtype=np.int32),
        face=np.array([5, 4], dtype=np.int8),
        origin=np.array([[1.0, 1.5, 1.5], [4.0, 1.5, 1.5]], dtype=np.float32),
        normal=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        patch=np.array([10, 20], dtype=np.int32),
        area=np.array([1.0, 1.0], dtype=np.float32),
    )
    grid = build_cell_patch_grid(ov, is_solid)
    assert grid[7, 1, 1] == NO_PATCH
    assert grid[8, 1, 1] == NO_PATCH


def test_corner_cell_touched_by_two_patches_keeps_the_last_row():
    """build_cell_patch_grid's docstring claims a cell touched by two
    surface-table rows keeps the last row's patch id. Pin that so it's
    enforced rather than merely asserted in prose."""
    is_solid = np.zeros((3, 3, 3), dtype=bool)
    is_solid[1, 1, 1] = True
    ov = SurfaceOverride(
        cell=np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32),
        face=np.array([4, 2], dtype=np.int8),
        origin=np.array([[1.5, 1.0, 1.5], [1.5, 1.5, 1.0]], dtype=np.float32),
        normal=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        patch=np.array([10, 20], dtype=np.int32),
        area=np.array([1.0, 1.0], dtype=np.float32),
    )
    grid = build_cell_patch_grid(ov, is_solid)
    assert grid[1, 1, 1] == 20
