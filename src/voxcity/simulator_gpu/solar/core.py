"""
Vector and ray utilities for palm-solar using Taichi.
Based on ray-tracing-one-weekend-taichi patterns.

GPU Optimization Notes:
- All functions use @ti.func for GPU inlining
- Branchless operations preferred where possible
- Memory coalescing friendly access patterns
- done-flag pattern for early termination (reduces warp divergence)

This module re-exports shared core utilities from simulator_gpu.core
and adds solar-specific extensions.
"""

import taichi as ti
import math

# Import shared core utilities from parent package
from ..core import (
    Vector3,
    Point3,
    Color3,
    PI,
    TWO_PI,
    HALF_PI,
    DEG_TO_RAD,
    RAD_TO_DEG,
    SOLAR_CONSTANT,
    EXT_COEF,
    MIN_STABLE_COSZEN,
    GPU_BLOCK_SIZE,
    normalize,
    normalize_safe,
    dot,
    cross,
    reflect,
    ray_at,
    length_squared,
    distance_squared,
    min3,
    max3,
    clamp,
    random_in_unit_sphere,
    random_in_hemisphere,
    random_cosine_hemisphere,
    spherical_to_cartesian,
    cartesian_to_spherical,
    rotate_vector_axis_angle,
    build_face_basis,
    Rays,
    HitRecord,
)

# Re-export all symbols for backward compatibility
__all__ = [
    'Vector3', 'Point3', 'Color3',
    'PI', 'TWO_PI', 'HALF_PI', 'DEG_TO_RAD', 'RAD_TO_DEG',
    'SOLAR_CONSTANT', 'EXT_COEF', 'MIN_STABLE_COSZEN', 'GPU_BLOCK_SIZE',
    'normalize', 'normalize_safe', 'dot', 'cross', 'reflect',
    'ray_at', 'length_squared', 'distance_squared',
    'min3', 'max3', 'clamp',
    'random_in_unit_sphere', 'random_in_hemisphere', 'random_cosine_hemisphere',
    'spherical_to_cartesian', 'cartesian_to_spherical',
    'rotate_vector_axis_angle', 'build_face_basis',
    'Rays', 'HitRecord',
]
