import numpy as np
import pytest

from voxcity.simulator_gpu.solar.surface_override import (
    SURFACE_TABLE_CONVENTION, NO_PATCH, SurfaceOverride,
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


def test_signature_is_stable_content_sensitive_and_duck_typed():
    a, b = _minimal(), _minimal()
    assert surface_override_signature(a) == surface_override_signature(b)
    assert surface_override_signature(_DuckTable(a)) == surface_override_signature(a)
    b.normal[0] = [1.0, 0.0, 0.0]
    assert surface_override_signature(a) != surface_override_signature(b)
    assert surface_override_signature(None) == surface_override_signature(None)
    assert surface_override_signature(None) != surface_override_signature(a)


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
