"""
Geometry utilities for view analysis.

Provides ray direction generation using various sampling strategies:
- Grid-based sampling (uniform azimuth x elevation grid)
- Fibonacci spiral sampling (more uniform hemisphere coverage)
"""

import numpy as np
import taichi as ti
from typing import Tuple


def generate_ray_directions_grid(
    n_azimuth: int = 120,
    n_elevation: int = 20,
    elevation_min_degrees: float = -30.0,
    elevation_max_degrees: float = 30.0
) -> np.ndarray:
    """
    Generate ray directions using a regular grid sampling.
    
    Args:
        n_azimuth: Number of azimuthal divisions
        n_elevation: Number of elevation divisions
        elevation_min_degrees: Minimum elevation angle in degrees
        elevation_max_degrees: Maximum elevation angle in degrees
    
    Returns:
        Array of shape (n_azimuth * n_elevation, 3) with unit direction vectors
    """
    azimuth_angles = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    elevation_angles = np.deg2rad(
        np.linspace(float(elevation_min_degrees), float(elevation_max_degrees), int(n_elevation))
    )
    
    ray_directions = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=np.float32)
    out_idx = 0
    
    for elevation in elevation_angles:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles:
            # x = east, y = north, z = up
            dx = cos_elev * np.sin(azimuth)
            dy = cos_elev * np.cos(azimuth)
            dz = sin_elev
            ray_directions[out_idx, 0] = dx
            ray_directions[out_idx, 1] = dy
            ray_directions[out_idx, 2] = dz
            out_idx += 1
    
    return ray_directions


def generate_ray_directions_fibonacci(
    n_rays: int = 2400,
    elevation_min_degrees: float = -30.0,
    elevation_max_degrees: float = 30.0
) -> np.ndarray:
    """
    Generate ray directions using Fibonacci spiral sampling.
    
    This provides more uniform coverage of the hemisphere compared to grid sampling.
    
    Args:
        n_rays: Total number of rays
        elevation_min_degrees: Minimum elevation angle in degrees
        elevation_max_degrees: Maximum elevation angle in degrees
    
    Returns:
        Array of shape (n_rays, 3) with unit direction vectors
    """
    N = int(max(1, n_rays))
    emin = np.deg2rad(float(elevation_min_degrees))
    emax = np.deg2rad(float(elevation_max_degrees))
    
    z_min = np.sin(min(emin, emax))
    z_max = np.sin(max(emin, emax))
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    i = np.arange(N, dtype=np.float64)
    z = z_min + (i + 0.5) * (z_max - z_min) / N
    phi = i * golden_angle
    
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    return np.stack((x, y, z), axis=1).astype(np.float32)


def generate_hemisphere_directions(
    n_azimuth: int = 80,
    n_elevation: int = 40
) -> np.ndarray:
    """
    Generate ray directions covering the upper hemisphere.
    
    Elevation goes from 0 (horizon) to 90 (zenith).
    
    Args:
        n_azimuth: Number of azimuthal divisions
        n_elevation: Number of elevation divisions
    
    Returns:
        Array of shape (n_azimuth * n_elevation, 3) with unit direction vectors
    """
    return generate_ray_directions_grid(
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0
    )


def rotate_vector_axis_angle(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a vector around an axis by a specified angle using Rodrigues' rotation formula.
    
    This is a CPU implementation matching voxcity.simulator.common.geometry.rotate_vector_axis_angle.
    
    Args:
        vec: Vector to rotate (3,)
        axis: Rotation axis (3,) - will be normalized
        angle: Rotation angle in radians
    
    Returns:
        Rotated vector (3,)
    """
    axis_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_len < 1e-12:
        return vec.copy()
    ux, uy, uz = axis / axis_len
    c = np.cos(angle)
    s = np.sin(angle)
    dot = vec[0]*ux + vec[1]*uy + vec[2]*uz
    cross_x = uy*vec[2] - uz*vec[1]
    cross_y = uz*vec[0] - ux*vec[2]
    cross_z = ux*vec[1] - uy*vec[0]
    v_rot = np.zeros(3, dtype=np.float64)
    v_rot[0] = vec[0] * c + cross_x * s + ux * dot * (1.0 - c)
    v_rot[1] = vec[1] * c + cross_y * s + uy * dot * (1.0 - c)
    v_rot[2] = vec[2] * c + cross_z * s + uz * dot * (1.0 - c)
    return v_rot


@ti.func
def rotate_vector_axis_angle_ti(vec: ti.template(), axis: ti.template(), angle: ti.f32) -> ti.template():
    """
    Taichi GPU kernel function for rotating a vector around an axis.
    
    Uses Rodrigues' rotation formula.
    
    Args:
        vec: Vector to rotate (Vector3)
        axis: Rotation axis (Vector3) - will be normalized
        angle: Rotation angle in radians
    
    Returns:
        Rotated Vector3
    """
    result = vec
    axis_len = axis.norm()
    if axis_len > 1e-12:
        u = axis / axis_len
        c = ti.cos(angle)
        s = ti.sin(angle)
        dot = vec.dot(u)
        cross = u.cross(vec)
        result = vec * c + cross * s + u * dot * (1.0 - c)
    return result


def build_face_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build orthonormal basis (u, v, n) for a surface normal.
    
    Matches voxcity.simulator.common.geometry._build_face_basis.
    
    Args:
        normal: Surface normal vector (3,)
    
    Returns:
        Tuple of (u, v, n) orthonormal vectors
    """
    nx, ny, nz = normal
    nrm = np.sqrt(nx*nx + ny*ny + nz*nz)
    if nrm < 1e-12:
        return (np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]))
    invn = 1.0 / nrm
    nx *= invn
    ny *= invn
    nz *= invn
    n = np.array([nx, ny, nz])
    
    # Choose helper vector to cross with normal
    if abs(nz) < 0.999:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])
    
    # u = helper x n (normalized)
    u = np.cross(helper, n)
    ul = np.linalg.norm(u)
    if ul < 1e-12:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / ul
    
    # v = n x u
    v = np.cross(n, u)
    
    return u, v, n


@ti.func
def build_face_basis_ti(normal: ti.template()) -> ti.template():
    """
    Taichi GPU kernel function to build orthonormal basis for a surface normal.
    
    Args:
        normal: Surface normal Vector3
    
    Returns:
        Tuple of (u, v, n) orthonormal Vector3s
    """
    nrm = normal.norm()
    n = normal
    u = ti.Vector([1.0, 0.0, 0.0])
    v = ti.Vector([0.0, 1.0, 0.0])
    
    if nrm > 1e-12:
        n = normal / nrm
        
        # Choose helper vector
        helper = ti.Vector([0.0, 0.0, 1.0])
        if ti.abs(n[2]) >= 0.999:
            helper = ti.Vector([1.0, 0.0, 0.0])
        
        # u = helper x n
        u = helper.cross(n)
        ul = u.norm()
        if ul > 1e-12:
            u = u / ul
        else:
            u = ti.Vector([1.0, 0.0, 0.0])
        
        # v = n x u
        v = n.cross(u)
    
    return u, v, n


@ti.data_oriented
class RayDirectionField:
    """
    GPU-accessible ray direction field for batch processing.
    """
    
    def __init__(self, directions: np.ndarray):
        """
        Initialize with numpy array of directions.
        
        Args:
            directions: Array of shape (n_rays, 3)
        """
        self.n_rays = directions.shape[0]
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_rays,))
        self._init_from_numpy(directions)
    
    @ti.kernel
    def _init_from_numpy(self, dirs: ti.types.ndarray()):
        for i in range(self.n_rays):
            self.directions[i] = ti.Vector([dirs[i, 0], dirs[i, 1], dirs[i, 2]])
