"""Reusable GPU field workspace for VoxCity visibility calculations.

A single ``ViewWorkspace`` allocates all Taichi fields required by
``ViewCalculator.compute_view_index()`` once and reuses them across calls
that share the same grid and ray-direction configuration.  This prevents
GPU memory growth when the optimizer calls the view index hundreds of times
with the same scene dimensions.

Cache management is handled by :func:`_get_or_create_view_workspace` and
:func:`clear_visibility_cache` in ``integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import taichi as ti

from .geometry import generate_ray_directions_fibonacci, generate_ray_directions_grid


@dataclass(frozen=True)
class ViewWorkspaceKey:
    """Immutable key that identifies a unique workspace allocation shape.

    Mode (sky / green) is intentionally *not* part of the key: both modes
    share the same field shapes, so a single workspace can be reused across
    them.  The masks are reinitialized on every ``compute_view_index`` call.
    """
    shape: Tuple[int, int, int]
    meshsize: float
    n_azimuth: int
    n_elevation: int
    ray_sampling: str
    n_rays: Optional[int]
    elevation_min_degrees: float
    elevation_max_degrees: float


@dataclass(frozen=True)
class SurfaceViewWorkspaceKey:
    """Immutable key for reusable building-surface visibility fields."""
    shape: Tuple[int, int, int]
    meshsize: float
    n_faces: int
    n_azimuth: int
    n_elevation: int
    ray_sampling: str
    n_rays: Optional[int]


class ViewWorkspace:
    """Pre-allocated Taichi fields for one visibility-ray configuration.

    Parameters are identical to the ray-configuration subset of
    ``ViewCalculator.__init__`` plus the grid dimensions.  All fields are
    created once at construction time.  The caller is responsible for
    reinitialising mask fields before each call to the GPU kernel.
    """

    def __init__(
        self,
        *,
        key: ViewWorkspaceKey,
        nx: int,
        ny: int,
        nz: int,
        n_azimuth: int,
        n_elevation: int,
        ray_sampling: str,
        n_rays: Optional[int],
        elevation_min_degrees: float,
        elevation_max_degrees: float,
    ) -> None:
        self.key = key
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Build the ray-direction array once.
        if ray_sampling.lower() == "fibonacci":
            dirs_np = generate_ray_directions_fibonacci(
                n_rays if n_rays is not None else n_azimuth * n_elevation,
                elevation_min_degrees,
                elevation_max_degrees,
            )
        else:
            dirs_np = generate_ray_directions_grid(
                n_azimuth,
                n_elevation,
                elevation_min_degrees,
                elevation_max_degrees,
            )

        self.n_ray_dirs = int(dirs_np.shape[0])
        self.ray_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_ray_dirs,))
        self.ray_dirs.from_numpy(np.asarray(dirs_np, dtype=np.float32))

        # Output map (reused each call; overwritten by the GPU kernel).
        self.vi_map = ti.field(dtype=ti.f32, shape=(nx, ny))

        # Mask fields (reinitialized by the kernel on each call).
        self.is_tree    = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_solid   = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_target  = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_allowed = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_blocker = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_walkable = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.mask_field = ti.field(dtype=ti.i8, shape=(nx, ny))
        self._reset_mask_all_on()

    def _reset_mask_all_on(self) -> None:
        """Fill mask_field with 1 (all cells enabled)."""
        self.mask_field.fill(1)

    def set_mask(self, mask_np) -> None:
        """Set mask_field from a boolean numpy array, or reset to all-on."""
        if mask_np is None:
            self._reset_mask_all_on()
        else:
            self.mask_field.from_numpy(np.asarray(mask_np, dtype=np.int8))

    def validate_voxel_data(self, voxel_data: np.ndarray) -> None:
        """Raise ``ValueError`` if *voxel_data* shape does not match this workspace."""
        if tuple(voxel_data.shape) != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"voxel_data shape {voxel_data.shape} does not match visibility workspace "
                f"shape {(self.nx, self.ny, self.nz)}."
            )


class SurfaceViewWorkspace:
    """Pre-allocated Taichi fields for one surface visibility configuration."""

    def __init__(
        self,
        *,
        key: SurfaceViewWorkspaceKey,
        nx: int,
        ny: int,
        nz: int,
        n_faces: int,
        n_azimuth: int,
        n_elevation: int,
        ray_sampling: str,
        n_rays: Optional[int],
    ) -> None:
        self.key = key
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n_faces = int(n_faces)

        if ray_sampling.lower() == "fibonacci":
            dirs_np = generate_ray_directions_fibonacci(
                n_rays if n_rays is not None else n_azimuth * n_elevation,
                0.0,
                90.0,
            )
        else:
            dirs_np = generate_ray_directions_grid(
                n_azimuth,
                n_elevation,
                0.0,
                90.0,
            )

        self.n_hemisphere_dirs = int(dirs_np.shape[0])
        self.hemisphere_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_hemisphere_dirs,))
        self.hemisphere_dirs.from_numpy(np.asarray(dirs_np, dtype=np.float32))

        self.face_centers = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_faces,))
        self.face_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_faces,))
        self.face_vf_values = ti.field(dtype=ti.f32, shape=(self.n_faces,))

        self.is_tree = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_target = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.is_opaque = ti.field(dtype=ti.i32, shape=(nx, ny, nz))

    def validate_face_data(self, face_centers: np.ndarray, face_normals: np.ndarray) -> None:
        expected_shape = (self.n_faces, 3)
        if tuple(face_centers.shape) != expected_shape or tuple(face_normals.shape) != expected_shape:
            raise ValueError(
                f"surface workspace face data shapes must match {expected_shape}; "
                f"got face_centers={face_centers.shape} face_normals={face_normals.shape}."
            )

    def validate_voxel_data(self, voxel_data: np.ndarray) -> None:
        if tuple(voxel_data.shape) != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"voxel_data shape {voxel_data.shape} does not match surface workspace "
                f"shape {(self.nx, self.ny, self.nz)}."
            )

    def set_faces(self, face_centers: np.ndarray, face_normals: np.ndarray) -> None:
        self.validate_face_data(face_centers, face_normals)
        self.face_centers.from_numpy(np.asarray(face_centers, dtype=np.float32))
        self.face_normals.from_numpy(np.asarray(face_normals, dtype=np.float32))
