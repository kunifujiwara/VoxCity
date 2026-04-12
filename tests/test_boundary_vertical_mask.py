"""Tests for compute_boundary_vertical_mask with non-square domains.

Regression test for the bug where grid_bounds had swapped X/Y extents,
causing interior south-facing surfaces to be falsely flagged as boundary
in non-square voxel grids.
"""
import numpy as np
import pytest

from voxcity.simulator_gpu.solar.integration.utils import compute_boundary_vertical_mask


class TestBoundaryVerticalMaskNonSquare:
    """Verify boundary detection on a non-square domain (ny_vc != nx_vc)."""

    def _make_grid(self, ny_vc=8, nx_vc=5, nz=6, meshsize=2.0):
        """Return grid parameters and correct grid bounds.

        Mesh coordinates follow VoxCity convention:
        - Mesh X = array axis 0 → range [0, ny_vc * meshsize]
        - Mesh Y = array axis 1 → range [0, nx_vc * meshsize]
        """
        grid_bounds = np.array([
            [0.0, 0.0, 0.0],
            [ny_vc * meshsize, nx_vc * meshsize, nz * meshsize],
        ], dtype=np.float64)
        boundary_epsilon = meshsize * 0.05
        return ny_vc, nx_vc, nz, meshsize, grid_bounds, boundary_epsilon

    def test_interior_face_not_masked(self):
        """An interior vertical face must NOT be flagged as boundary."""
        ny_vc, nx_vc, nz, meshsize, bounds, eps = self._make_grid()
        # Place a vertical face in the interior of the domain
        centers = np.array([[8.0, 5.0, 4.0]], dtype=np.float64)  # well inside
        normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # vertical
        mask = compute_boundary_vertical_mask(centers, normals, bounds, eps)
        assert not mask[0], "Interior face should NOT be flagged as boundary"

    def test_true_boundary_face_masked(self):
        """Vertical faces at domain edges must be flagged."""
        ny_vc, nx_vc, nz, meshsize, bounds, eps = self._make_grid()
        x_max = ny_vc * meshsize  # 16.0
        y_max = nx_vc * meshsize  # 10.0
        centers = np.array([
            [0.0, 5.0, 4.0],   # on x_min boundary
            [x_max, 5.0, 4.0], # on x_max boundary
            [8.0, 0.0, 4.0],   # on y_min boundary
            [8.0, y_max, 4.0],  # on y_max boundary
        ], dtype=np.float64)
        normals = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        mask = compute_boundary_vertical_mask(centers, normals, bounds, eps)
        assert np.all(mask), "All domain-edge vertical faces must be flagged"

    def test_horizontal_face_never_masked(self):
        """A horizontal face at the boundary should NOT be flagged."""
        ny_vc, nx_vc, nz, meshsize, bounds, eps = self._make_grid()
        centers = np.array([[0.0, 5.0, 4.0]], dtype=np.float64)  # on x_min boundary
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)  # upward (horizontal)
        mask = compute_boundary_vertical_mask(centers, normals, bounds, eps)
        assert not mask[0], "Horizontal faces should never be flagged"

    def test_swapped_bounds_false_positive(self):
        """With WRONG (swapped) bounds, an interior face gets falsely masked.

        This is the exact scenario that caused the cumulative solar bug:
        nx_vc * meshsize < ny_vc * meshsize, and a face at x = nx_vc * meshsize
        is interior but gets flagged as boundary when bounds are swapped.
        """
        ny_vc, nx_vc, nz, meshsize, correct_bounds, eps = self._make_grid()

        # Face at x = nx_vc * meshsize = 10.0, which is interior
        # (x_max = ny_vc * meshsize = 16.0)
        centers = np.array([[nx_vc * meshsize, 5.0, 4.0]], dtype=np.float64)
        normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        # With CORRECT bounds: should NOT be flagged
        mask_correct = compute_boundary_vertical_mask(centers, normals, correct_bounds, eps)
        assert not mask_correct[0], "Interior face should not be flagged with correct bounds"

        # With WRONG (swapped) bounds: would be falsely flagged
        wrong_bounds = np.array([
            [0.0, 0.0, 0.0],
            [nx_vc * meshsize, ny_vc * meshsize, nz * meshsize],
        ], dtype=np.float64)
        mask_wrong = compute_boundary_vertical_mask(centers, normals, wrong_bounds, eps)
        assert mask_wrong[0], "Swapped bounds should falsely flag this interior face"
