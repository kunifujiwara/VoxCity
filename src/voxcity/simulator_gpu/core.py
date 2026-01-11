"""
Shared core utilities for simulator_gpu.

Vector and ray utilities using Taichi for GPU acceleration.
Based on ray-tracing-one-weekend-taichi patterns.

GPU Optimization Notes:
- All functions use @ti.func for GPU inlining
- Branchless operations preferred where possible
- Memory coalescing friendly access patterns
- done-flag pattern for early termination (reduces warp divergence)
"""

import taichi as ti
import math

# Type aliases for clarity
Vector3 = ti.math.vec3
Point3 = ti.math.vec3
Color3 = ti.math.vec3

# Constants - using Taichi static for compile-time optimization
PI = math.pi
TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Solar constant (W/m^2) - matching PALM's solar_constant
SOLAR_CONSTANT = 1361.0

# Default extinction coefficient for vegetation
# PALM: ext_coef = 0.6_wp (radiation_model_mod.f90 line ~890)
EXT_COEF = 0.6

# Minimum stable cosine of zenith angle
MIN_STABLE_COSZEN = 0.0262

# GPU block size hint for optimal thread occupancy
GPU_BLOCK_SIZE = 256


@ti.func
def normalize(v: Vector3) -> Vector3:
    """Normalize a vector."""
    return v / v.norm()


@ti.func
def normalize_safe(v: Vector3) -> Vector3:
    """Normalize a vector with safety check for zero-length."""
    len_sq = v.dot(v)
    if len_sq > 1e-10:
        return v / ti.sqrt(len_sq)
    return Vector3(0.0, 0.0, 1.0)


@ti.func
def dot(v1: Vector3, v2: Vector3) -> ti.f32:
    """Dot product of two vectors."""
    return v1.dot(v2)


@ti.func
def cross(v1: Vector3, v2: Vector3) -> Vector3:
    """Cross product of two vectors."""
    return v1.cross(v2)


@ti.func
def reflect(v: Vector3, n: Vector3) -> Vector3:
    """Reflect vector v around normal n."""
    return v - 2.0 * v.dot(n) * n


@ti.func
def ray_at(origin: Point3, direction: Vector3, t: ti.f32) -> Point3:
    """Get point along ray at parameter t."""
    return origin + t * direction


@ti.func
def length_squared(v: Vector3) -> ti.f32:
    """Compute squared length of vector (avoids sqrt)."""
    return v.dot(v)


@ti.func
def distance_squared(p1: Point3, p2: Point3) -> ti.f32:
    """Compute squared distance between two points (avoids sqrt)."""
    diff = p2 - p1
    return diff.dot(diff)


@ti.func
def min3(a: ti.f32, b: ti.f32, c: ti.f32) -> ti.f32:
    """Branchless minimum of three values."""
    return ti.min(a, ti.min(b, c))


@ti.func
def max3(a: ti.f32, b: ti.f32, c: ti.f32) -> ti.f32:
    """Branchless maximum of three values."""
    return ti.max(a, ti.max(b, c))


@ti.func
def clamp(x: ti.f32, lo: ti.f32, hi: ti.f32) -> ti.f32:
    """Clamp value to range [lo, hi]."""
    return ti.max(lo, ti.min(hi, x))


@ti.func
def random_in_unit_sphere() -> Vector3:
    """Generate random point in unit sphere."""
    theta = ti.random() * TWO_PI
    v = ti.random()
    phi = ti.acos(2.0 * v - 1.0)
    r = ti.random() ** (1.0 / 3.0)
    return Vector3(
        r * ti.sin(phi) * ti.cos(theta),
        r * ti.sin(phi) * ti.sin(theta),
        r * ti.cos(phi)
    )


@ti.func
def random_in_hemisphere(normal: Vector3) -> Vector3:
    """Generate random vector in hemisphere around normal."""
    vec = random_in_unit_sphere()
    if vec.dot(normal) < 0.0:
        vec = -vec
    return vec


@ti.func
def random_cosine_hemisphere(normal: Vector3) -> Vector3:
    """
    Generate random vector with cosine-weighted distribution in hemisphere.
    Used for diffuse radiation sampling.
    """
    u1 = ti.random()
    u2 = ti.random()
    r = ti.sqrt(u1)
    theta = TWO_PI * u2
    
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    z = ti.sqrt(1.0 - u1)
    
    # Create orthonormal basis around normal
    up = Vector3(0.0, 1.0, 0.0)
    if ti.abs(normal.y) > 0.999:
        up = Vector3(1.0, 0.0, 0.0)
    
    tangent = normalize(cross(up, normal))
    bitangent = cross(normal, tangent)
    
    return normalize(x * tangent + y * bitangent + z * normal)


@ti.func 
def spherical_to_cartesian(azimuth: ti.f32, elevation: ti.f32) -> Vector3:
    """
    Convert spherical coordinates to Cartesian unit vector.
    
    Args:
        azimuth: Angle from north (y-axis), clockwise, in radians
        elevation: Angle from horizontal, in radians (0 = horizontal, pi/2 = zenith)
    
    Returns:
        Unit vector (x, y, z) where z is vertical (up)
    """
    cos_elev = ti.cos(elevation)
    sin_elev = ti.sin(elevation)
    cos_azim = ti.cos(azimuth)
    sin_azim = ti.sin(azimuth)
    
    # x = east, y = north, z = up
    x = cos_elev * sin_azim
    y = cos_elev * cos_azim
    z = sin_elev
    
    return Vector3(x, y, z)


@ti.func
def cartesian_to_spherical(v: Vector3) -> ti.math.vec2:
    """
    Convert Cartesian unit vector to spherical coordinates.
    
    Returns:
        vec2(azimuth, elevation) in radians
    """
    elevation = ti.asin(ti.math.clamp(v.z, -1.0, 1.0))
    azimuth = ti.atan2(v.x, v.y)
    if azimuth < 0.0:
        azimuth += TWO_PI
    return ti.math.vec2(azimuth, elevation)


@ti.func
def rotate_vector_axis_angle(vec: Vector3, axis: Vector3, angle: ti.f32) -> Vector3:
    """
    Rotate vector around an axis by a given angle using Rodrigues' rotation formula.
    
    Args:
        vec: Vector to rotate
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians
    
    Returns:
        Rotated vector
    """
    axis_len = axis.norm()
    if axis_len < 1e-12:
        return vec
    
    k = axis / axis_len
    c = ti.cos(angle)
    s = ti.sin(angle)
    
    # Rodrigues' rotation formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    v_rot = vec * c + cross(k, vec) * s + k * dot(k, vec) * (1.0 - c)
    
    return v_rot


@ti.func
def build_face_basis(normal: Vector3):
    """
    Build orthonormal basis for a face with given normal.
    
    Returns:
        Tuple of (tangent, bitangent, normal) vectors
    """
    n_len = normal.norm()
    if n_len < 1e-12:
        return Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)
    
    n = normal / n_len
    
    # Choose helper vector not parallel to normal
    helper = Vector3(0.0, 0.0, 1.0)
    if ti.abs(n.z) > 0.999:
        helper = Vector3(1.0, 0.0, 0.0)
    
    # Compute tangent via cross product
    u = cross(helper, n)
    u_len = u.norm()
    if u_len < 1e-12:
        u = Vector3(1.0, 0.0, 0.0)
    else:
        u = u / u_len
    
    # Bitangent
    v = cross(n, u)
    
    return u, v, n


@ti.data_oriented
class Rays:
    """
    Array of rays for batch processing.
    Similar to ray-tracing-one-weekend-taichi but adapted for view/solar tracing.
    """
    
    def __init__(self, n_rays: int):
        self.n_rays = n_rays
        self.origin = ti.Vector.field(3, dtype=ti.f32, shape=(n_rays,))
        self.direction = ti.Vector.field(3, dtype=ti.f32, shape=(n_rays,))
        self.transparency = ti.field(dtype=ti.f32, shape=(n_rays,))
        self.active = ti.field(dtype=ti.i32, shape=(n_rays,))
    
    @ti.func
    def set(self, idx: ti.i32, origin: Point3, direction: Vector3, transp: ti.f32):
        self.origin[idx] = origin
        self.direction[idx] = direction
        self.transparency[idx] = transp
        self.active[idx] = 1
    
    @ti.func
    def get(self, idx: ti.i32):
        return self.origin[idx], self.direction[idx], self.transparency[idx]
    
    @ti.func
    def deactivate(self, idx: ti.i32):
        self.active[idx] = 0
    
    @ti.func
    def is_active(self, idx: ti.i32) -> ti.i32:
        return self.active[idx]


@ti.data_oriented
class HitRecord:
    """
    Store ray-surface intersection results.
    """
    
    def __init__(self, n_rays: int):
        self.n_rays = n_rays
        self.hit = ti.field(dtype=ti.i32, shape=(n_rays,))
        self.t = ti.field(dtype=ti.f32, shape=(n_rays,))
        self.point = ti.Vector.field(3, dtype=ti.f32, shape=(n_rays,))
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=(n_rays,))
        self.surface_id = ti.field(dtype=ti.i32, shape=(n_rays,))
    
    @ti.func
    def set(self, idx: ti.i32, hit: ti.i32, t: ti.f32, point: Point3, 
            normal: Vector3, surface_id: ti.i32):
        self.hit[idx] = hit
        self.t[idx] = t
        self.point[idx] = point
        self.normal[idx] = normal
        self.surface_id[idx] = surface_id
    
    @ti.func
    def get(self, idx: ti.i32):
        return (self.hit[idx], self.t[idx], self.point[idx], 
                self.normal[idx], self.surface_id[idx])
