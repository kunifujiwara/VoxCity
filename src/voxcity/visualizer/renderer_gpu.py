"""
GPU-accelerated renderer using Taichi for ray tracing.

This module provides GPU-accelerated rendering functionality for VoxCity
visualization, using Taichi for parallel ray tracing on GPU.

Based on ray-tracing-one-weekend-taichi implementation, extended for
triangle mesh support.
"""
from __future__ import annotations

import os
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..models import VoxCity, MeshCollection
from .builder import MeshBuilder
from .palette import get_voxel_color_map
from ..geoprocessor.mesh import create_sim_surface_mesh


# ============================================================================
# Lighting Helper Function
# ============================================================================

def light_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> Tuple[float, float, float]:
    """Convert azimuth and elevation angles to light direction vector (Z-up).
    
    Args:
        azimuth_deg: Angle in XY plane from X-axis (0°=East, 90°=North), in degrees
        elevation_deg: Angle above horizon (0°=horizontal, 90°=zenith), in degrees
    
    Returns:
        Normalized (x, y, z) direction vector pointing toward the light source
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return (x, y, z)


# ============================================================================
# Taichi Import and Initialization
# ============================================================================

_HAS_TAICHI = False
_ti = None

try:
    import taichi as ti
    _ti = ti
    # Initialize Taichi immediately so decorators work
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        try:
            ti.init(arch=ti.cpu)
        except Exception:
            pass
    _HAS_TAICHI = True
except ImportError:
    pass


# ============================================================================
# Fast BVH Construction (Iterative, Vectorized)
# ============================================================================

def build_bvh_fast(vertices: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build BVH using fast iterative construction with vectorized operations.
    
    Returns arrays for GPU consumption:
    - bvh_triangle_id: triangle ID at each node (-1 for internal)
    - bvh_left_id: left child ID (-1 for none)
    - bvh_right_id: right child ID (-1 for none)
    - bvh_next_id: next node in traversal (-1 for none)
    - bvh_min: bounding box min (N, 3)
    - bvh_max: bounding box max (N, 3)
    """
    n_tris = len(indices)
    if n_tris == 0:
        return (np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32))
    
    # Precompute all triangle bounding boxes and centers (vectorized)
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    
    tri_min = np.minimum(v0, np.minimum(v1, v2))
    tri_max = np.maximum(v0, np.maximum(v1, v2))
    tri_centers = (v0 + v1 + v2) / 3.0
    
    # Maximum BVH nodes = 2 * n_tris - 1
    max_nodes = 2 * n_tris
    
    # Allocate arrays
    bvh_tri_id = np.full(max_nodes, -1, dtype=np.int32)
    bvh_left = np.full(max_nodes, -1, dtype=np.int32)
    bvh_right = np.full(max_nodes, -1, dtype=np.int32)
    bvh_parent = np.full(max_nodes, -1, dtype=np.int32)
    bvh_min = np.zeros((max_nodes, 3), dtype=np.float32)
    bvh_max = np.zeros((max_nodes, 3), dtype=np.float32)
    
    # Stack for iterative construction: (node_id, triangle_indices, parent_id, is_left_child)
    stack = [(0, np.arange(n_tris, dtype=np.int32), -1, True)]
    next_node_id = 1
    
    while stack:
        node_id, tri_ids, parent_id, is_left = stack.pop()
        
        if parent_id >= 0:
            bvh_parent[node_id] = parent_id
            if is_left:
                bvh_left[parent_id] = node_id
            else:
                bvh_right[parent_id] = node_id
        
        if len(tri_ids) == 1:
            # Leaf node
            tid = tri_ids[0]
            bvh_tri_id[node_id] = tid
            bvh_min[node_id] = tri_min[tid]
            bvh_max[node_id] = tri_max[tid]
        else:
            # Internal node
            # Compute bounds
            bvh_min[node_id] = tri_min[tri_ids].min(axis=0)
            bvh_max[node_id] = tri_max[tri_ids].max(axis=0)
            
            # Find split axis (longest span)
            centers = tri_centers[tri_ids]
            span = centers.max(axis=0) - centers.min(axis=0)
            axis = np.argmax(span)
            
            # Sort and split
            sort_idx = np.argsort(centers[:, axis])
            sorted_tris = tri_ids[sort_idx]
            mid = len(sorted_tris) // 2
            
            left_tris = sorted_tris[:mid]
            right_tris = sorted_tris[mid:]
            
            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2
            
            # Push children (right first so left is processed first)
            stack.append((right_id, right_tris, node_id, False))
            stack.append((left_id, left_tris, node_id, True))
    
    # Trim to actual size
    actual_nodes = next_node_id
    bvh_tri_id = bvh_tri_id[:actual_nodes]
    bvh_left = bvh_left[:actual_nodes]
    bvh_right = bvh_right[:actual_nodes]
    bvh_parent = bvh_parent[:actual_nodes]
    bvh_min = bvh_min[:actual_nodes]
    bvh_max = bvh_max[:actual_nodes]
    
    # Compute next_id for BVH traversal
    bvh_next = np.full(actual_nodes, -1, dtype=np.int32)
    for i in range(actual_nodes):
        # Find next node in traversal
        node = i
        while True:
            parent = bvh_parent[node]
            if parent == -1:
                break
            if bvh_left[parent] == node and bvh_right[parent] != -1:
                bvh_next[i] = bvh_right[parent]
                break
            node = parent
    
    return bvh_tri_id, bvh_left, bvh_right, bvh_next, bvh_min, bvh_max


# ============================================================================
# Taichi-based GPU Components (only defined if Taichi is available)
# ============================================================================

if _HAS_TAICHI:
    
    # Vector/Color Types
    Vector3 = ti.types.vector(3, ti.f32)
    Color3 = ti.types.vector(3, ti.f32)

    WHITE = Vector3([1.0, 1.0, 1.0])
    BLUE = Vector3([0.5, 0.7, 1.0])
    BLACK = Vector3([0.0, 0.0, 0.0])
    GRAY = Vector3([0.5, 0.5, 0.5])

    # NOTE: Taichi @ti.func decorated functions should NOT have Python type annotations.
    # Taichi infers types at JIT compile time.

    @ti.func
    def random_in_unit_sphere():
        """Generate a random point inside a unit sphere."""
        theta = ti.random() * math.pi * 2.0
        v = ti.random()
        phi = ti.acos(2.0 * v - 1.0)
        r = ti.random() ** (1.0 / 3.0)
        return ti.Vector([
            r * ti.sin(phi) * ti.cos(theta),
            r * ti.sin(phi) * ti.sin(theta),
            r * ti.cos(phi)
        ])

    @ti.func
    def random_in_hemisphere(normal):
        """Generate a random direction in the hemisphere defined by normal."""
        vec = random_in_unit_sphere()
        if vec.dot(normal) < 0:
            vec = -vec
        return vec

    @ti.func
    def random_in_unit_disk():
        """Generate a random point in a unit disk (z=0)."""
        theta = ti.random() * math.pi * 2.0
        r = ti.random() ** 0.5
        return ti.Vector([r * ti.cos(theta), r * ti.sin(theta), 0.0])

    @ti.func
    def reflect(v, n):
        """Reflect vector v around normal n."""
        return v - 2.0 * v.dot(n) * n

    @ti.func
    def refract(v, n, etai_over_etat):
        """Refract vector v through surface with normal n using Snell's law."""
        cos_theta = ti.min(-v.dot(n), 1.0)
        r_out_perp = etai_over_etat * (v + cos_theta * n)
        r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.norm_sqr())) * n
        return r_out_perp + r_out_parallel

    @ti.func
    def reflectance(cosine, ref_idx):
        """Schlick's approximation for Fresnel reflectance."""
        r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)) ** 2
        return r0 + (1.0 - r0) * ((1.0 - cosine) ** 5)

    @ti.func
    def ray_at(origin, direction, t):
        """Get point along ray at parameter t."""
        return origin + direction * t
    
    # Material type constants
    MAT_LAMBERT = 0    # Diffuse (Lambertian)
    MAT_METAL = 1      # Specular reflection (metal)
    MAT_DIELECTRIC = 2 # Glass/water with refraction


    @ti.data_oriented
    class TriangleBVH:
        """GPU-accelerated BVH for triangle meshes using Taichi."""
        
        def __init__(self, vertices: np.ndarray, indices: np.ndarray, 
                     colors: Optional[np.ndarray] = None,
                     materials: Optional[np.ndarray] = None,
                     roughness: Optional[np.ndarray] = None,
                     ior: Optional[np.ndarray] = None):
            """
            Initialize BVH from triangle mesh.
            
            Parameters
            ----------
            vertices : np.ndarray
                (N, 3) array of vertex positions
            indices : np.ndarray
                (M, 3) array of triangle vertex indices
            colors : np.ndarray, optional
                (M, 3) or (M, 4) array of per-face colors (0-255 range)
            materials : np.ndarray, optional
                (M,) array of material types per face (0=Lambert, 1=Metal, 2=Dielectric)
            roughness : np.ndarray, optional
                (M,) array of roughness values per face (0-1, used for Metal)
            ior : np.ndarray, optional
                (M,) array of index of refraction per face (used for Dielectric, default 1.5)
            """
            self.n_vertices = len(vertices)
            self.n_triangles = len(indices)
            
            # Store vertices and indices in Taichi fields
            self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
            self.indices = ti.Vector.field(3, dtype=ti.i32, shape=self.n_triangles)
            
            # Store face colors (normalized to 0-1)
            self.face_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.n_triangles)
            
            # Store material properties per face
            self.face_materials = ti.field(dtype=ti.i32, shape=self.n_triangles)
            self.face_roughness = ti.field(dtype=ti.f32, shape=self.n_triangles)
            self.face_ior = ti.field(dtype=ti.f32, shape=self.n_triangles)
            
            # Copy data to fields
            self.vertices.from_numpy(vertices.astype(np.float32))
            self.indices.from_numpy(indices.astype(np.int32))
            
            if colors is not None:
                if colors.shape[1] == 4:
                    colors = colors[:, :3]
                colors_normalized = colors.astype(np.float32) / 255.0
                self.face_colors.from_numpy(colors_normalized)
            else:
                # Default gray color
                default_colors = np.full((self.n_triangles, 3), 0.7, dtype=np.float32)
                self.face_colors.from_numpy(default_colors)
            
            # Set material properties
            if materials is not None:
                self.face_materials.from_numpy(materials.astype(np.int32))
            else:
                # Default to Lambert (diffuse)
                default_mats = np.zeros(self.n_triangles, dtype=np.int32)
                self.face_materials.from_numpy(default_mats)
            
            if roughness is not None:
                self.face_roughness.from_numpy(roughness.astype(np.float32))
            else:
                # Default roughness for metals
                default_rough = np.full(self.n_triangles, 0.1, dtype=np.float32)
                self.face_roughness.from_numpy(default_rough)
            
            if ior is not None:
                self.face_ior.from_numpy(ior.astype(np.float32))
            else:
                # Default IOR for glass/water (water ~1.33, glass ~1.5)
                default_ior = np.full(self.n_triangles, 1.33, dtype=np.float32)
                self.face_ior.from_numpy(default_ior)
            
            # Build BVH using fast iterative method
            (bvh_tri_id, bvh_left, bvh_right, bvh_next, 
             bvh_min_arr, bvh_max_arr) = build_bvh_fast(vertices, indices)
            
            total_nodes = len(bvh_tri_id)
            
            # Create Taichi fields for BVH
            self.bvh_triangle_id = ti.field(ti.i32, shape=total_nodes)
            self.bvh_left_id = ti.field(ti.i32, shape=total_nodes)
            self.bvh_right_id = ti.field(ti.i32, shape=total_nodes)
            self.bvh_next_id = ti.field(ti.i32, shape=total_nodes)
            self.bvh_min = ti.Vector.field(3, dtype=ti.f32, shape=total_nodes)
            self.bvh_max = ti.Vector.field(3, dtype=ti.f32, shape=total_nodes)
            
            # Copy to Taichi fields
            self.bvh_triangle_id.from_numpy(bvh_tri_id)
            self.bvh_left_id.from_numpy(bvh_left)
            self.bvh_right_id.from_numpy(bvh_right)
            self.bvh_next_id.from_numpy(bvh_next)
            self.bvh_min.from_numpy(bvh_min_arr)
            self.bvh_max.from_numpy(bvh_max_arr)
            
            self.bvh_root = 0
        
        @ti.func
        def hit_aabb(self, bvh_id, ray_origin, ray_direction, t_min, t_max):
            """Test ray-AABB intersection using slab method."""
            intersect = 1
            min_aabb = self.bvh_min[bvh_id]
            max_aabb = self.bvh_max[bvh_id]
            
            for i in ti.static(range(3)):
                if ray_direction[i] == 0:
                    if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                        intersect = 0
                else:
                    i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
                    i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]
                    new_t_max = ti.max(i1, i2)
                    new_t_min = ti.min(i1, i2)
                    t_max = ti.min(new_t_max, t_max)
                    t_min = ti.max(new_t_min, t_min)
            
            if t_min > t_max:
                intersect = 0
            return intersect
        
        @ti.func
        def hit_triangle(self, tri_id, ray_origin, ray_direction, t_min, t_max):
            """
            Möller–Trumbore ray-triangle intersection.
            Returns t value if hit, -1.0 if miss.
            """
            EPSILON = 1e-8
            t_hit = -1.0
            
            idx = self.indices[tri_id]
            v0 = self.vertices[idx[0]]
            v1 = self.vertices[idx[1]]
            v2 = self.vertices[idx[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = ray_direction.cross(edge2)
            a = edge1.dot(h)
            
            if ti.abs(a) > EPSILON:
                f = 1.0 / a
                s = ray_origin - v0
                u = f * s.dot(h)
                
                if u >= 0.0 and u <= 1.0:
                    q = s.cross(edge1)
                    v = f * ray_direction.dot(q)
                    
                    if v >= 0.0 and u + v <= 1.0:
                        t = f * edge2.dot(q)
                        if t > t_min and t < t_max:
                            t_hit = t
            
            return t_hit
        
        @ti.func
        def get_triangle_normal(self, tri_id):
            """Compute triangle normal."""
            idx = self.indices[tri_id]
            v0 = self.vertices[idx[0]]
            v1 = self.vertices[idx[1]]
            v2 = self.vertices[idx[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2)
            return normal.normalized()
        
        @ti.func
        def hit_all(self, ray_origin, ray_direction, t_min, t_max):
            """
            Intersect ray against all triangles using BVH.
            
            Returns (hit, t, normal, triangle_id)
            """
            hit_anything = 0
            closest_t = t_max
            hit_normal = ti.Vector([0.0, 0.0, 1.0])
            hit_tri_id = -1
            
            curr = self.bvh_root
            
            while curr != -1:
                tri_id = self.bvh_triangle_id[curr]
                left_id = self.bvh_left_id[curr]
                right_id = self.bvh_right_id[curr]
                next_id = self.bvh_next_id[curr]
                
                if tri_id != -1:
                    # Leaf node - test triangle
                    t = self.hit_triangle(tri_id, ray_origin, ray_direction, t_min, closest_t)
                    if t > 0:
                        hit_anything = 1
                        closest_t = t
                        hit_tri_id = tri_id
                        hit_normal = self.get_triangle_normal(tri_id)
                    curr = next_id
                else:
                    # Internal node - test AABB
                    if self.hit_aabb(curr, ray_origin, ray_direction, t_min, closest_t):
                        if left_id != -1:
                            curr = left_id
                        elif right_id != -1:
                            curr = right_id
                        else:
                            curr = next_id
                    else:
                        curr = next_id
            
            # Ensure normal faces the ray
            if hit_anything and ray_direction.dot(hit_normal) > 0:
                hit_normal = -hit_normal
            
            return hit_anything, closest_t, hit_normal, hit_tri_id
        
        @ti.func
        def get_face_color(self, tri_id):
            """Get the color of a triangle face."""
            return self.face_colors[tri_id]
        
        @ti.func
        def get_face_material(self, tri_id):
            """Get the material type of a triangle face."""
            return self.face_materials[tri_id]
        
        @ti.func
        def get_face_roughness(self, tri_id):
            """Get the roughness of a triangle face."""
            return self.face_roughness[tri_id]
        
        @ti.func
        def get_face_ior(self, tri_id):
            """Get the index of refraction of a triangle face."""
            return self.face_ior[tri_id]


    @ti.data_oriented
    class GPUCamera:
        """GPU-accelerated camera for ray generation."""
        
        def __init__(self, position: Tuple[float, float, float],
                     look_at: Tuple[float, float, float],
                     up: Tuple[float, float, float] = (0, 0, 1),
                     fov: float = 45.0,
                     aspect_ratio: float = 1.0,
                     aperture: float = 0.0,
                     focus_dist: float = 10.0):
            """
            Initialize camera.
            
            Parameters
            ----------
            position : tuple
                Camera position (x, y, z)
            look_at : tuple
                Point the camera looks at
            up : tuple
                Up vector
            fov : float
                Vertical field of view in degrees
            aspect_ratio : float
                Width / height
            aperture : float
                Lens aperture for depth of field (0 = pinhole)
            focus_dist : float
                Focus distance
            """
            self.fov = float(fov)
            self.aspect_ratio = float(aspect_ratio)
            self.aperture = float(aperture)
            self.focus_dist = float(focus_dist)
            self._look_at = np.array(look_at, dtype=np.float32)
            self._up = np.array(up, dtype=np.float32)
            
            # Taichi fields for camera state
            self.origin = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.u = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.v = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.w = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.lens_radius = ti.field(dtype=ti.f32, shape=())
            
            self._update_camera(np.array(position, dtype=np.float32))
        
        def _update_camera(self, position: np.ndarray):
            """Update camera parameters."""
            theta = math.radians(self.fov)
            h = math.tan(theta / 2.0)
            viewport_height = 2.0 * h
            viewport_width = viewport_height * self.aspect_ratio
            
            position = np.array(position, dtype=np.float32)
            look_at = self._look_at
            up = self._up
            
            w = position - look_at
            w = w / np.linalg.norm(w)
            u = np.cross(up, w)
            u = u / np.linalg.norm(u)
            v = np.cross(w, u)
            
            horizontal = self.focus_dist * viewport_width * u
            vertical = self.focus_dist * viewport_height * v
            lower_left = position - horizontal / 2 - vertical / 2 - self.focus_dist * w
            
            self.origin[None] = position.tolist()
            self.u[None] = u.tolist()
            self.v[None] = v.tolist()
            self.w[None] = w.tolist()
            self.horizontal[None] = horizontal.tolist()
            self.vertical[None] = vertical.tolist()
            self.lower_left_corner[None] = lower_left.tolist()
            self.lens_radius[None] = self.aperture / 2.0
        
        def set_position(self, position: Tuple[float, float, float]):
            """Update camera position."""
            self._update_camera(np.array(position, dtype=np.float32))
        
        def set_look_at(self, look_at: Tuple[float, float, float]):
            """Update look-at point."""
            self._look_at = np.array(look_at, dtype=np.float32)
            # Re-update with current origin
            origin = np.array([
                self.origin[None][0],
                self.origin[None][1],
                self.origin[None][2]
            ], dtype=np.float32)
            self._update_camera(origin)
        
        @ti.func
        def get_ray(self, s, t):
            """Generate ray for normalized screen coordinates (s, t)."""
            rd = self.lens_radius[None] * random_in_unit_disk()
            offset = self.u[None] * rd.x + self.v[None] * rd.y
            origin = self.origin[None] + offset
            direction = (self.lower_left_corner[None] + 
                         s * self.horizontal[None] + 
                         t * self.vertical[None] - origin)
            return origin, direction.normalized()


    @ti.data_oriented
    class TaichiRenderer:
        """GPU-accelerated ray tracing renderer using Taichi."""
        
        def __init__(self, width: int = 800, height: int = 600,
                     samples_per_pixel: int = 64, max_depth: int = 8):
            """
            Initialize the renderer.
            
            Parameters
            ----------
            width : int
                Image width in pixels
            height : int
                Image height in pixels
            samples_per_pixel : int
                Number of samples per pixel for anti-aliasing
            max_depth : int
                Maximum ray bounce depth
            """
            self.width = width
            self.height = height
            self.samples_per_pixel = samples_per_pixel
            self.max_depth = max_depth
            
            # Pixel buffer
            self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
            self.sample_count = ti.field(dtype=ti.i32, shape=(width, height))
            
            # Samples per batch for efficient rendering
            self.samples_per_batch = ti.field(dtype=ti.i32, shape=())
            
            # Scene components (set during rendering)
            self.bvh: Optional[TriangleBVH] = None
            self.camera: Optional[GPUCamera] = None
            
            # Lighting
            self.light_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.light_color = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.light_noise = ti.field(dtype=ti.f32, shape=())  # Light direction noise
            self.ambient_color = ti.Vector.field(3, dtype=ti.f32, shape=())
            
            # Floor plane (like voxel-challenge - receives shadows)
            self.floor_height = ti.field(dtype=ti.f32, shape=())
            self.floor_color = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.floor_height[None] = -1e10  # Disabled by default (very low)
            self.floor_color[None] = [0.44, 0.47, 0.44]  # Gray-green #707970
            
            # Background color (like voxel-challenge)
            self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())
            self.background_color[None] = [0.8, 0.85, 0.95]  # Light bluish gray
            
            # Exposure and vignette fields (like voxel-challenge)
            self._exposure_field = ti.field(dtype=ti.f32, shape=())
            self._vignette_strength_field = ti.field(dtype=ti.f32, shape=())
            self._vignette_radius_field = ti.field(dtype=ti.f32, shape=())
            self._exposure_field[None] = 1.3
            self._vignette_strength_field[None] = 0.0  # Disabled by default
            self._vignette_radius_field[None] = 0.0
            
            # Python-side exposure properties
            self.exposure = 1.3
            self.vignette_strength = 0.0  # Disabled by default
            self.vignette_radius = 0.0
            
            # Default lighting (azimuth=220°, elevation=45°)
            self.set_lighting(
                direction=light_direction_from_angles(220, 45),  # Southwest light
                color=(0.9, 0.9, 0.85),  # Warm directional light
                ambient=(0.15, 0.15, 0.18),  # Fill light for shadows
                light_noise=0.3
            )
        
        def set_lighting(self, direction: Tuple[float, float, float] = (1.0, 1.0, -0.5),
                         color: Tuple[float, float, float] = (0.9, 0.9, 0.85),
                         ambient: Tuple[float, float, float] = (0.25, 0.25, 0.28),
                         light_noise: float = 0.2):
            """Set scene lighting parameters."""
            d = np.array(direction, dtype=np.float32)
            d = d / np.linalg.norm(d)
            self.light_dir[None] = d.tolist()
            self.light_color[None] = color
            self.ambient_color[None] = ambient
            self.light_noise[None] = light_noise
        
        def set_background_color(self, color: Tuple[float, float, float]):
            """Set background color."""
            self.background_color[None] = color
        
        def set_exposure(self, exposure: float = 3.0, vignette_strength: float = 0.7, vignette_radius: float = 0.2):
            """Set exposure and vignette parameters."""
            self.exposure = exposure
            self.vignette_strength = vignette_strength
            self.vignette_radius = vignette_radius
        
        def set_floor(self, height: float = -0.1, color: Tuple[float, float, float] = (0.3, 0.35, 0.3)):
            """
            Set floor plane for shadow casting (like voxel-challenge).
            
            Parameters
            ----------
            height : float
                Y-coordinate of the floor plane. Set very negative to disable.
            color : Tuple[float, float, float]
                RGB color of the floor (0-1 range).
            """
            self.floor_height[None] = height
            self.floor_color[None] = color
        
        def set_scene(self, vertices: np.ndarray, indices: np.ndarray,
                      colors: Optional[np.ndarray] = None,
                      materials: Optional[np.ndarray] = None,
                      roughness: Optional[np.ndarray] = None,
                      ior: Optional[np.ndarray] = None):
            """
            Set the scene geometry with optional material properties.
            
            Parameters
            ----------
            vertices : np.ndarray
                (N, 3) array of vertex positions
            indices : np.ndarray
                (M, 3) array of triangle indices
            colors : np.ndarray, optional
                (M, 3) or (M, 4) array of face colors (0-255)
            materials : np.ndarray, optional
                (M,) array of material types per face:
                - 0 (MAT_LAMBERT): Diffuse Lambertian
                - 1 (MAT_METAL): Specular metal with roughness
                - 2 (MAT_DIELECTRIC): Glass/water with refraction
            roughness : np.ndarray, optional
                (M,) array of roughness values (0-1) for Metal materials
            ior : np.ndarray, optional
                (M,) array of index of refraction for Dielectric materials
            """
            self.bvh = TriangleBVH(vertices, indices, colors, materials, roughness, ior)
        
        def set_camera(self, position: Tuple[float, float, float],
                       look_at: Tuple[float, float, float],
                       up: Tuple[float, float, float] = (0, 0, 1),
                       fov: float = 45.0):
            """Set camera parameters."""
            aspect = self.width / self.height
            if self.camera is None:
                # First time - create camera
                self.camera = GPUCamera(position, look_at, up, fov, aspect)
            else:
                # Update existing camera in-place (avoid recreating Taichi fields)
                self.camera.fov = float(fov)
                self.camera._look_at = np.array(look_at, dtype=np.float32)
                self.camera._up = np.array(up, dtype=np.float32)
                self.camera._update_camera(np.array(position, dtype=np.float32))
        
        @ti.kernel
        def _clear_pixels(self):
            """Clear pixel buffer."""
            for x, y in self.pixels:
                self.pixels[x, y] = ti.Vector([0.0, 0.0, 0.0])
                self.sample_count[x, y] = 0
        
        @ti.kernel
        def _render_pass(self):
            """Execute render pass with multiple samples per pixel for efficiency."""
            samples_this_pass = self.samples_per_batch[None]
            for x, y in self.pixels:
                for s in range(samples_this_pass):
                    if self.sample_count[x, y] < self.samples_per_pixel:
                        # Generate ray with jitter for anti-aliasing
                        u = (x + ti.random()) / (self.width - 1)
                        v = (y + ti.random()) / (self.height - 1)
                        
                        ray_origin, ray_direction = self.camera.get_ray(u, v)
                        
                        # Trace ray
                        color = self._trace_ray(ray_origin, ray_direction, self.max_depth)
                        
                        # Accumulate color
                        self.pixels[x, y] += color
                        self.sample_count[x, y] += 1
        
        @ti.func
        def _scatter_lambert(self, ray_dir, hit_point, normal, color):
            """Scatter ray for Lambertian (diffuse) material."""
            out_dir = normal + random_in_hemisphere(normal)
            # Normalize to avoid degenerate directions
            out_dir = out_dir.normalized()
            out_origin = hit_point + normal * 0.001
            attenuation = color
            return True, out_origin, out_dir, attenuation
        
        @ti.func
        def _scatter_metal(self, ray_dir, hit_point, normal, color, roughness):
            """Scatter ray for Metal (specular) material."""
            reflected = reflect(ray_dir.normalized(), normal)
            out_dir = reflected + roughness * random_in_unit_sphere()
            out_dir = out_dir.normalized()
            out_origin = hit_point + normal * 0.001
            attenuation = color
            # Only reflect if going in same hemisphere as normal
            reflected_valid = out_dir.dot(normal) > 0.0
            return reflected_valid, out_origin, out_dir, attenuation
        
        @ti.func
        def _scatter_dielectric(self, ray_dir, hit_point, normal, color, ior, front_facing):
            """Scatter ray for Dielectric (glass/water) material with refraction."""
            refraction_ratio = 1.0 / ior
            if not front_facing:
                refraction_ratio = ior
            
            unit_dir = ray_dir.normalized()
            cos_theta = ti.min(-unit_dir.dot(normal), 1.0)
            sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
            
            # Determine if we refract or reflect (total internal reflection)
            cannot_refract = refraction_ratio * sin_theta > 1.0
            
            out_dir = ti.Vector([0.0, 0.0, 0.0])
            if cannot_refract or reflectance(cos_theta, refraction_ratio) > ti.random():
                # Reflect
                out_dir = reflect(unit_dir, normal)
            else:
                # Refract
                out_dir = refract(unit_dir, normal, refraction_ratio)
            
            out_origin = hit_point + out_dir * 0.001
            attenuation = color
            return True, out_origin, out_dir.normalized(), attenuation
        
        @ti.func
        def _trace_ray(self, origin, direction, depth):
            """
            Trace a ray and compute color using multi-bounce path tracing.
            
            Follows voxel-challenge rendering approach:
            - Lambert (diffuse) surfaces: random hemisphere scattering with direct lighting
            - Metal/Water (mat=1,2): specular reflection with Fresnel
            - Floor plane at configurable height that receives shadows
            - Direct lighting checked on ALL bounces for better global illumination
            - Russian Roulette for early termination
            """
            color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])
            
            ray_origin = origin
            ray_direction = direction
            
            hit_background = 0
            first_bounce_depth = 0
            
            floor_height = self.floor_height[None]
            floor_color = self.floor_color[None]
            eps = 1e-5
            
            for bounce in range(depth):
                first_bounce_depth = bounce
                
                # Check BVH hit
                bvh_hit, bvh_t, bvh_normal, tri_id = self.bvh.hit_all(ray_origin, ray_direction, 0.001, 1e10)
                
                # Check floor hit (Z-up coordinate system - floor is at z=floor_height)
                floor_hit = 0
                floor_t = 1e10
                if ray_direction[2] < -eps:  # Ray pointing down (negative Z)
                    floor_t = (floor_height - ray_origin[2]) / ray_direction[2]
                    if floor_t > 0.001:
                        floor_hit = 1
                
                # Determine which hit is closer
                hit = bvh_hit or floor_hit
                hit_floor = 0
                t = 1e10
                normal = ti.Vector([0.0, 0.0, 1.0])  # Default (floor normal points up in Z)
                if floor_hit and (not bvh_hit or floor_t < bvh_t):
                    hit_floor = 1
                    t = floor_t
                    normal = ti.Vector([0.0, 0.0, 1.0])  # Floor normal points up (Z-up)
                elif bvh_hit:
                    t = bvh_t
                    normal = bvh_normal
                
                if hit:
                    # Compute hit point
                    hit_point = ray_at(ray_origin, ray_direction, t)
                    
                    # Initialize surface properties (Taichi requires all vars defined)
                    surface_color = floor_color
                    mat_type = MAT_LAMBERT
                    roughness = 0.0
                    
                    if hit_floor:
                        # Floor hit - diffuse surface that receives shadows
                        pass  # Already set above
                    else:
                        # Get surface properties from BVH
                        surface_color = self.bvh.get_face_color(tri_id)
                        mat_type = self.bvh.get_face_material(tri_id)
                        roughness = self.bvh.get_face_roughness(tri_id)
                    
                        # Ensure normal faces correct direction
                        front_facing = ray_direction.dot(normal) < 0.0
                        if not front_facing:
                            normal = -normal
                    
                    # Handle materials (voxel-challenge style)
                    if mat_type == MAT_DIELECTRIC or mat_type == MAT_METAL:
                        # Water/reflective: use specular reflection with Fresnel (like voxel-challenge mat=3)
                        reflect_dir = ray_direction - 2.0 * ray_direction.dot(normal) * normal
                        # Add slight roughness
                        water_roughness = 0.08
                        noise = ti.Vector([
                            (ti.random() - 0.5) * water_roughness,
                            (ti.random() - 0.5) * water_roughness,
                            (ti.random() - 0.5) * water_roughness
                        ])
                        ray_direction = (reflect_dir + noise).normalized()
                        # Fresnel-like effect: more reflection at grazing angles
                        fresnel = 0.4 + 0.6 * (1.0 - ti.abs(normal.dot(ray_direction)))
                        throughput *= surface_color * fresnel
                        ray_origin = hit_point + ray_direction * 0.001
                    else:
                        # Diffuse: scatter randomly in hemisphere
                        ray_direction = normal + random_in_hemisphere(normal)
                        ray_direction = ray_direction.normalized()
                        throughput *= surface_color
                        ray_origin = hit_point + normal * 0.001
                    
                    # Direct lighting check (on ALL bounces for ALL materials, like voxel-challenge)
                    # Use hit_point + normal offset for shadow ray (not ray_origin which is for next bounce)
                    shadow_origin = hit_point + normal * 0.001
                    
                    dir_noise = ti.Vector([
                        (ti.random() - 0.5),
                        (ti.random() - 0.5),
                        (ti.random() - 0.5)
                    ]) * self.light_noise[None]
                    light_dir = (self.light_dir[None] + dir_noise).normalized()
                    
                    n_dot_l = light_dir.dot(normal)
                    if n_dot_l > 0:
                        # Shadow ray - check BVH hit from surface toward light
                        shadow_hit, _, _, _ = self.bvh.hit_all(
                            shadow_origin, light_dir, 0.001, 1e10)
                        if not shadow_hit:
                            # Light contribution (like voxel-challenge)
                            color += throughput * self.light_color[None] * n_dot_l
                else:
                    # Hit background
                    hit_background = 1
                    break
                
                # Russian Roulette for early termination
                max_c = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
                if ti.random() > max_c:
                    throughput = ti.Vector([0.0, 0.0, 0.0])
                    break
                else:
                    throughput /= max_c
            
            # Background handling (like voxel-challenge)
            # IMPORTANT: Only add background on DIRECT hits (first bounce)
            # For indirect hits, don't add background - this preserves shadows!
            if hit_background:
                if first_bounce_depth == 0:
                    # Direct hit to background - use background color
                    color = self.background_color[None]
                # else: don't add anything - indirect rays hitting background contribute nothing
                # This is how voxel-challenge does it (lines 331-339 in renderer.py)
            
            return color
        
        @ti.kernel
        def _finalize_pixels_kernel(self):
            """Normalize accumulated pixel values with exposure and vignette."""
            samples = float(self.samples_per_pixel)
            exposure = self._exposure_field[None]
            vignette_strength = self._vignette_strength_field[None]
            vignette_radius = self._vignette_radius_field[None]
            
            for x, y in self.pixels:
                if self.sample_count[x, y] > 0:
                    # Normalize by sample count
                    c = self.pixels[x, y] / samples
                    
                    # Compute vignette (darkens corners like voxel-challenge)
                    u = float(x) / float(self.width)
                    v = float(y) / float(self.height)
                    dist = ti.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
                    darken = 1.0 - vignette_strength * ti.max(dist - vignette_radius, 0.0)
                    
                    # Apply exposure and vignette, then gamma correction
                    for i in ti.static(range(3)):
                        self.pixels[x, y][i] = ti.sqrt(c[i] * darken * exposure)
        
        def render(self, show_progress: bool = True) -> np.ndarray:
            """
            Render the scene.
            
            Returns
            -------
            np.ndarray
                (height, width, 3) RGB image in 0-255 range
            """
            if self.bvh is None:
                raise ValueError("No scene set. Call set_scene() first.")
            if self.camera is None:
                raise ValueError("No camera set. Call set_camera() first.")
            
            self._clear_pixels()
            
            # Batch samples to reduce Python-Taichi kernel call overhead
            # Higher batch = fewer kernel calls but less responsive progress
            samples_per_batch = min(8, self.samples_per_pixel)
            num_batches = (self.samples_per_pixel + samples_per_batch - 1) // samples_per_batch
            
            for batch in range(num_batches):
                remaining = self.samples_per_pixel - batch * samples_per_batch
                this_batch = min(samples_per_batch, remaining)
                self.samples_per_batch[None] = this_batch
                self._render_pass()
                if show_progress:
                    progress = min(100.0, (batch + 1) / num_batches * 100)
                    print(f"\rRendering: {progress:.1f}%", end="", flush=True)
            
            if show_progress:
                print("\rRendering: 100.0%")
            
            # Sync exposure fields before finalizing
            self._exposure_field[None] = self.exposure
            self._vignette_strength_field[None] = self.vignette_strength
            self._vignette_radius_field[None] = self.vignette_radius
            
            self._finalize_pixels_kernel()
            
            # Convert to numpy and format for image
            img = self.pixels.to_numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            # Transpose to (H, W, 3) and flip vertically
            img = np.transpose(img, (1, 0, 2))
            img = np.flipud(img)
            
            return img
        
        def render_to_file(self, filepath: str, show_progress: bool = True):
            """
            Render and save to file.
            
            Parameters
            ----------
            filepath : str
                Output file path (supports PNG, JPG, etc.)
            """
            img = self.render(show_progress)
            
            try:
                from PIL import Image
                Image.fromarray(img).save(filepath)
            except ImportError:
                try:
                    import cv2
                    img_bgr = img[:, :, ::-1]  # RGB to BGR
                    cv2.imwrite(filepath, img_bgr)
                except ImportError:
                    import imageio
                    imageio.imwrite(filepath, img)

else:
    # Placeholders when Taichi is not available
    TriangleBVH = None
    GPUCamera = None
    TaichiRenderer = None


# ============================================================================
# High-Level Rendering Functions (no Taichi dependency at definition time)
# ============================================================================

# Classes that should use reflective/dielectric materials (water, glass, etc.)
REFLECTIVE_CLASSES = {9}  # Water class

# Classes that should use metallic materials (optional, for special effects)
METALLIC_CLASSES = set()  # Can add building class or others if desired


def merge_meshes(mesh_collection: MeshCollection, 
                 exclude_classes: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge all meshes in a collection into single arrays with material assignments.
    
    Parameters
    ----------
    mesh_collection : MeshCollection
        Collection of meshes to merge
    exclude_classes : set, optional
        Set of class IDs to exclude from merging (e.g., {-3} to skip buildings)
    
    Returns
    -------
    vertices : np.ndarray
        (N, 3) combined vertices
    indices : np.ndarray
        (M, 3) combined triangle indices
    colors : np.ndarray
        (M, 3) combined face colors
    materials : np.ndarray
        (M,) material types per face (0=Lambert, 1=Metal, 2=Dielectric)
    """
    all_vertices = []
    all_indices = []
    all_colors = []
    all_materials = []
    vertex_offset = 0
    
    if exclude_classes is None:
        exclude_classes = set()
    
    for name, mesh in mesh_collection:
        if mesh.vertices is None or len(mesh.vertices) == 0:
            continue
        if mesh.faces is None or len(mesh.faces) == 0:
            continue
        
        # Check if this class should be excluded
        try:
            class_id = int(name)
            if class_id in exclude_classes:
                continue
        except (ValueError, TypeError):
            pass
            
        all_vertices.append(mesh.vertices)
        all_indices.append(mesh.faces + vertex_offset)
        
        if mesh.colors is not None:
            all_colors.append(mesh.colors[:, :3])
        else:
            # Default color
            default = np.full((len(mesh.faces), 3), 180, dtype=np.uint8)
            all_colors.append(default)
        
        # Determine material type based on class ID (mesh name)
        try:
            class_id = int(name)
        except (ValueError, TypeError):
            class_id = 0
        
        n_faces = len(mesh.faces)
        if class_id in REFLECTIVE_CLASSES:
            # Water/glass - use dielectric material
            mat_type = 2  # MAT_DIELECTRIC
        elif class_id in METALLIC_CLASSES:
            # Metallic - use metal material
            mat_type = 1  # MAT_METAL
        else:
            # Default to Lambertian (diffuse)
            mat_type = 0  # MAT_LAMBERT
        
        all_materials.append(np.full(n_faces, mat_type, dtype=np.int32))
        vertex_offset += len(mesh.vertices)
    
    if not all_vertices:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), 
                np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.int32))
    
    vertices = np.vstack(all_vertices).astype(np.float32)
    indices = np.vstack(all_indices).astype(np.int32)
    colors = np.vstack(all_colors).astype(np.uint8)
    materials = np.concatenate(all_materials)
    
    return vertices, indices, colors, materials


class GPURenderer:
    """High-level GPU renderer for VoxCity objects."""
    
    def __init__(self, width: int = 1920, height: int = 1080,
                 samples_per_pixel: int = 64, max_depth: int = 6,
                 arch: str = 'gpu'):
        """
        Initialize the GPU renderer.
        
        Parameters
        ----------
        width : int
            Output image width
        height : int
            Output image height
        samples_per_pixel : int
            Anti-aliasing quality (higher = better quality, slower)
        max_depth : int
            Ray bounce depth
        arch : str
            Compute architecture ('gpu', 'cpu', 'cuda', 'vulkan', 'metal')
        """
        if not _HAS_TAICHI:
            raise ImportError("Taichi is required for GPU rendering. Install with: pip install taichi")
        
        self.taichi_renderer = TaichiRenderer(
            width=width,
            height=height,
            samples_per_pixel=samples_per_pixel,
            max_depth=max_depth
        )
        self.width = width
        self.height = height
    
    def render_city(self, city: VoxCity,
                    voxel_color_map: str | dict = "default",
                    camera_position: Optional[Tuple[float, float, float]] = None,
                    camera_look_at: Optional[Tuple[float, float, float]] = None,
                    camera_up: Tuple[float, float, float] = (0, 0, 1),
                    fov: float = 25.0,
                    light_direction: Optional[Tuple[float, float, float]] = None,
                    ambient: Tuple[float, float, float] = (0.15, 0.15, 0.18),
                    output_path: Optional[str] = None,
                    show_progress: bool = True,
                    # Building simulation overlay
                    building_sim_mesh=None,
                    building_value_name: str = 'svf_values',
                    building_colormap: str = 'viridis',
                    building_vmin: Optional[float] = None,
                    building_vmax: Optional[float] = None,
                    building_nan_color: str = 'gray',
                    render_voxel_buildings: bool = False,
                    # Ground simulation surface overlay
                    ground_sim_grid: Optional[np.ndarray] = None,
                    ground_dem_grid: Optional[np.ndarray] = None,
                    ground_z_offset: Optional[float] = None,
                    ground_view_point_height: Optional[float] = None,
                    ground_colormap: str = 'viridis',
                    ground_vmin: Optional[float] = None,
                    ground_vmax: Optional[float] = None) -> np.ndarray:
        """
        Render a VoxCity to an image using GPU ray tracing.
        
        Parameters
        ----------
        city : VoxCity
            The VoxCity object to render
        voxel_color_map : str or dict
            Color map for voxel classes
        camera_position : tuple, optional
            Camera position. Auto-computed if not specified.
        camera_look_at : tuple, optional
            Camera look-at point. Auto-computed if not specified.
        camera_up : tuple
            Camera up vector
        fov : float
            Field of view in degrees
        light_direction : tuple
            Light direction vector
        ambient : tuple
            Ambient light color
        output_path : str, optional
            If provided, save the rendered image to this path
        show_progress : bool
            Whether to show rendering progress
        building_sim_mesh : trimesh, optional
            Building mesh with simulation results in metadata
        building_value_name : str
            Metadata key for building values (e.g., 'svf_values', 'global', 'direct')
        building_colormap : str
            Matplotlib colormap for building values
        building_vmin : float, optional
            Minimum value for building color scale
        building_vmax : float, optional
            Maximum value for building color scale
        building_nan_color : str
            Color for NaN/invalid building values
        render_voxel_buildings : bool
            Whether to render voxel buildings when building_sim_mesh is provided
        ground_sim_grid : np.ndarray, optional
            2D array of ground-level simulation values (e.g., GVI, solar)
        ground_dem_grid : np.ndarray, optional
            2D DEM array for ground surface positioning
        ground_z_offset : float, optional
            Height offset for ground surface above DEM
        ground_view_point_height : float, optional
            Alternative height parameter for ground surface
        ground_colormap : str
            Matplotlib colormap for ground values
        ground_vmin : float, optional
            Minimum value for ground color scale
        ground_vmax : float, optional
            Maximum value for ground color scale
            
        Returns
        -------
        np.ndarray
            Rendered image as (H, W, 3) RGB array
        """
        meshsize = city.voxels.meta.meshsize
        
        # Determine which classes to exclude from voxel rendering
        exclude_classes = set()
        if building_sim_mesh is not None and not render_voxel_buildings:
            exclude_classes.add(-3)  # Exclude building voxels
        
        # Build mesh collection from voxels
        collection = MeshBuilder.from_voxel_grid(
            city.voxels, meshsize=meshsize, voxel_color_map=voxel_color_map
        )
        
        # Merge all meshes with material assignments
        vertices, indices, colors, materials = merge_meshes(collection, exclude_classes)
        
        # Add building sim mesh if provided
        if building_sim_mesh is not None and hasattr(building_sim_mesh, 'vertices'):
            bv = np.asarray(building_sim_mesh.vertices)
            bf = np.asarray(building_sim_mesh.faces)
            
            # Get simulation values from metadata and apply colormap
            values = None
            if hasattr(building_sim_mesh, 'metadata') and isinstance(building_sim_mesh.metadata, dict):
                values = building_sim_mesh.metadata.get(building_value_name)
            
            if values is not None:
                values = np.asarray(values)
                
                # Determine if values are per-face or per-vertex
                face_vals = None
                if len(values) == len(bf):
                    face_vals = values.astype(float)
                elif len(values) == len(bv):
                    vals_v = values.astype(float)
                    face_vals = np.nanmean(vals_v[bf], axis=1)
                
                if face_vals is not None:
                    # Apply colormap
                    finite = np.isfinite(face_vals)
                    vmin_b = building_vmin if building_vmin is not None else (float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0)
                    vmax_b = building_vmax if building_vmax is not None else (float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0)
                    norm_b = mcolors.Normalize(vmin=vmin_b, vmax=vmax_b)
                    cmap_b = cm.get_cmap(building_colormap)
                    
                    bc = np.zeros((len(bf), 3), dtype=np.uint8)
                    if np.any(finite):
                        colors_float = cmap_b(norm_b(face_vals[finite]))
                        bc[finite] = (colors_float[:, :3] * 255).astype(np.uint8)
                    
                    # Handle NaN values
                    nan_rgba = np.array(mcolors.to_rgba(building_nan_color))
                    bc[~finite] = (nan_rgba[:3] * 255).astype(np.uint8)
                else:
                    # Fallback to face colors
                    bc = getattr(building_sim_mesh.visual, 'face_colors', None)
                    if bc is None:
                        bc = np.full((len(bf), 3), 200, dtype=np.uint8)
                    else:
                        bc = np.asarray(bc)[:, :3]
            else:
                # No values, use face colors from mesh
                bc = getattr(building_sim_mesh.visual, 'face_colors', None)
                if bc is None:
                    bc = np.full((len(bf), 3), 200, dtype=np.uint8)
                else:
                    bc = np.asarray(bc)[:, :3]
            
            # Building meshes use default Lambert material
            bm = np.zeros(len(bf), dtype=np.int32)
            
            # Append to existing arrays
            vertex_offset = len(vertices) if len(vertices) > 0 else 0
            if len(vertices) > 0:
                vertices = np.vstack([vertices, bv]).astype(np.float32)
                indices = np.vstack([indices, bf + vertex_offset]).astype(np.int32)
                colors = np.vstack([colors, bc]).astype(np.uint8)
                materials = np.concatenate([materials, bm])
            else:
                vertices = bv.astype(np.float32)
                indices = bf.astype(np.int32)
                colors = bc.astype(np.uint8)
                materials = bm
        
        # Add ground simulation surface overlay
        if ground_sim_grid is not None:
            # Auto-fill DEM from city if not provided
            if ground_dem_grid is None:
                ground_dem_grid = getattr(city.dem, "elevation", None)
            
            if ground_dem_grid is not None:
                # Determine z offset (height above ground level)
                z_off = ground_z_offset if ground_z_offset is not None else ground_view_point_height
                try:
                    z_off = float(z_off) if z_off is not None else 1.5
                except Exception:
                    z_off = 1.5
                
                # Position at ground level (one meshsize up) plus view point height
                # Ground level in voxel space is at Z=meshsize (top of first layer)
                # We add the view_point_height to place the overlay at the simulation height
                try:
                    z_off = meshsize + z_off
                except Exception:
                    pass
                
                # Normalize DEM
                try:
                    dem_norm = np.asarray(ground_dem_grid, dtype=float)
                    dem_norm = dem_norm - np.nanmin(dem_norm)
                except Exception:
                    dem_norm = ground_dem_grid
                
                # Determine color range
                sim_vals = np.asarray(ground_sim_grid, dtype=float)
                finite = np.isfinite(sim_vals)
                vmin_g = ground_vmin if ground_vmin is not None else (float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0)
                vmax_g = ground_vmax if ground_vmax is not None else (float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0)
                
                # Create ground simulation mesh
                sim_mesh = create_sim_surface_mesh(
                    ground_sim_grid,
                    dem_norm,
                    meshsize=meshsize,
                    z_offset=z_off,
                    cmap_name=ground_colormap,
                    vmin=vmin_g,
                    vmax=vmax_g,
                )
                
                if sim_mesh is not None and hasattr(sim_mesh, 'vertices') and len(sim_mesh.vertices) > 0:
                    gv = np.asarray(sim_mesh.vertices)
                    gf = np.asarray(sim_mesh.faces)
                    
                    # Get colors from ground mesh
                    gc = getattr(sim_mesh.visual, 'face_colors', None)
                    if gc is None:
                        gc = np.full((len(gf), 3), 180, dtype=np.uint8)
                    else:
                        gc = np.asarray(gc)[:, :3]
                    
                    # Ground uses Lambert material
                    gm = np.zeros(len(gf), dtype=np.int32)
                    
                    # Append to existing arrays
                    vertex_offset = len(vertices) if len(vertices) > 0 else 0
                    if len(vertices) > 0:
                        vertices = np.vstack([vertices, gv]).astype(np.float32)
                        indices = np.vstack([indices, gf + vertex_offset]).astype(np.int32)
                        colors = np.vstack([colors, gc]).astype(np.uint8)
                        materials = np.concatenate([materials, gm])
                    else:
                        vertices = gv.astype(np.float32)
                        indices = gf.astype(np.int32)
                        colors = gc.astype(np.uint8)
                        materials = gm
        
        if len(vertices) == 0:
            raise ValueError("No geometry to render")
        
        # Set scene with materials
        self.taichi_renderer.set_scene(vertices, indices, colors, materials)
        
        # Compute scene bounds for auto camera
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        diagonal = np.linalg.norm(bounds_max - bounds_min)
        
        # Auto camera position if not specified (matching render_rotation defaults)
        if camera_position is None:
            # Isometric-like view with distance_factor=1.5, height_factor=0.5
            camera_radius = diagonal * 1.5
            camera_position = (
                center[0] + camera_radius * 0.7,
                center[1] + camera_radius * 0.7,
                center[2] + diagonal * 0.5
            )
        
        if camera_look_at is None:
            # look_at_z_factor=-0.1 to move object higher in frame
            camera_look_at = (center[0], center[1], center[2] + diagonal * (-0.1))
        
        # Set camera
        self.taichi_renderer.set_camera(camera_position, camera_look_at, camera_up, fov)
        
        # Set lighting only if explicitly provided (otherwise use renderer defaults)
        if light_direction is not None:
            self.taichi_renderer.set_lighting(light_direction, (0.9, 0.9, 0.85), ambient)
        
        # Render
        if output_path:
            self.taichi_renderer.render_to_file(output_path, show_progress)
            img = self.taichi_renderer.pixels.to_numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            img = np.transpose(img, (1, 0, 2))
            img = np.flipud(img)
            return img
        else:
            return self.taichi_renderer.render(show_progress)
    
    def render_rotation(self, city: VoxCity,
                        voxel_color_map: str | dict = "default",
                        output_directory: str = "output",
                        file_prefix: str = "city_rotation",
                        num_frames: int = 240,
                        camera_height_factor: float = 0.5,
                        camera_distance_factor: float = 1.5,
                        look_at_z_factor: float = -0.1,
                        fov: float = 25.0,
                        floor_enabled: bool = True,
                        floor_color: Tuple[float, float, float] = (0.3, 0.35, 0.3),
                        show_progress: bool = True,
                        # Building simulation overlay
                        building_sim_mesh=None,
                        building_value_name: str = 'svf_values',
                        building_colormap: str = 'viridis',
                        building_vmin: Optional[float] = None,
                        building_vmax: Optional[float] = None,
                        building_nan_color: str = 'gray',
                        render_voxel_buildings: bool = False,
                        # Ground simulation surface overlay
                        ground_sim_grid: Optional[np.ndarray] = None,
                        ground_dem_grid: Optional[np.ndarray] = None,
                        ground_z_offset: Optional[float] = None,
                        ground_view_point_height: Optional[float] = None,
                        ground_colormap: str = 'viridis',
                        ground_vmin: Optional[float] = None,
                        ground_vmax: Optional[float] = None) -> List[str]:
        """
        Render a rotating view of the city.
        
        Parameters
        ----------
        city : VoxCity
            The VoxCity to render
        voxel_color_map : str or dict
            Color map for voxel classes
        output_directory : str
            Directory to save frames
        file_prefix : str
            Prefix for frame filenames
        num_frames : int
            Number of frames in rotation
        camera_height_factor : float
            Camera height relative to scene diagonal
        camera_distance_factor : float
            Camera distance relative to scene diagonal
        look_at_z_factor : float
            Look-at Z offset as fraction of diagonal (negative = object appears higher)
        fov : float
            Field of view
        floor_enabled : bool
            Whether to render a floor plane that receives shadows
        floor_color : Tuple[float, float, float]
            RGB color of the floor (0-1 range)
        show_progress : bool
            Whether to show progress
        building_sim_mesh : trimesh, optional
            Building mesh with simulation results in metadata
        building_value_name : str
            Metadata key for building values
        building_colormap : str
            Matplotlib colormap for building values
        building_vmin, building_vmax : float, optional
            Value range for building color scale
        building_nan_color : str
            Color for NaN values
        render_voxel_buildings : bool
            Whether to render voxel buildings with sim mesh
        ground_sim_grid : np.ndarray, optional
            2D array of ground simulation values
        ground_dem_grid : np.ndarray, optional
            2D DEM array
        ground_z_offset, ground_view_point_height : float, optional
            Height offset for ground surface
        ground_colormap : str
            Matplotlib colormap for ground values
        ground_vmin, ground_vmax : float, optional
            Value range for ground color scale
            
        Returns
        -------
        List[str]
            Paths to rendered frame files
        """
        os.makedirs(output_directory, exist_ok=True)
        
        meshsize = city.voxels.meta.meshsize
        
        # Determine which classes to exclude from voxel rendering
        exclude_classes = set()
        if building_sim_mesh is not None and not render_voxel_buildings:
            exclude_classes.add(-3)  # Exclude building voxels
        
        collection = MeshBuilder.from_voxel_grid(
            city.voxels, meshsize=meshsize, voxel_color_map=voxel_color_map
        )
        vertices, indices, colors, materials = merge_meshes(collection, exclude_classes)
        
        # Add building sim mesh if provided
        if building_sim_mesh is not None and hasattr(building_sim_mesh, 'vertices'):
            bv = np.asarray(building_sim_mesh.vertices)
            bf = np.asarray(building_sim_mesh.faces)
            
            # Get simulation values from metadata and apply colormap
            values = None
            if hasattr(building_sim_mesh, 'metadata') and isinstance(building_sim_mesh.metadata, dict):
                values = building_sim_mesh.metadata.get(building_value_name)
            
            if values is not None:
                values = np.asarray(values)
                
                # Determine if values are per-face or per-vertex
                face_vals = None
                if len(values) == len(bf):
                    face_vals = values.astype(float)
                elif len(values) == len(bv):
                    vals_v = values.astype(float)
                    face_vals = np.nanmean(vals_v[bf], axis=1)
                
                if face_vals is not None:
                    # Apply colormap
                    finite = np.isfinite(face_vals)
                    vmin_b = building_vmin if building_vmin is not None else (float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0)
                    vmax_b = building_vmax if building_vmax is not None else (float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0)
                    norm_b = mcolors.Normalize(vmin=vmin_b, vmax=vmax_b)
                    cmap_b = cm.get_cmap(building_colormap)
                    
                    bc = np.zeros((len(bf), 3), dtype=np.uint8)
                    if np.any(finite):
                        colors_float = cmap_b(norm_b(face_vals[finite]))
                        bc[finite] = (colors_float[:, :3] * 255).astype(np.uint8)
                    
                    # Handle NaN values
                    nan_rgba = np.array(mcolors.to_rgba(building_nan_color))
                    bc[~finite] = (nan_rgba[:3] * 255).astype(np.uint8)
                else:
                    bc = np.full((len(bf), 3), 200, dtype=np.uint8)
            else:
                bc = np.full((len(bf), 3), 200, dtype=np.uint8)
            
            # Building meshes use default Lambert material
            bm = np.zeros(len(bf), dtype=np.int32)
            
            # Append to existing arrays
            vertex_offset = len(vertices) if len(vertices) > 0 else 0
            if len(vertices) > 0:
                vertices = np.vstack([vertices, bv]).astype(np.float32)
                indices = np.vstack([indices, bf + vertex_offset]).astype(np.int32)
                colors = np.vstack([colors, bc]).astype(np.uint8)
                materials = np.concatenate([materials, bm])
            else:
                vertices = bv.astype(np.float32)
                indices = bf.astype(np.int32)
                colors = bc.astype(np.uint8)
                materials = bm
        
        # Add ground simulation surface overlay
        if ground_sim_grid is not None:
            if ground_dem_grid is None:
                ground_dem_grid = getattr(city.dem, "elevation", None)
            
            if ground_dem_grid is not None:
                z_off = ground_z_offset if ground_z_offset is not None else ground_view_point_height
                try:
                    z_off = float(z_off) if z_off is not None else 1.5
                except Exception:
                    z_off = 1.5
                
                try:
                    z_off = meshsize + z_off
                except Exception:
                    pass
                
                try:
                    dem_norm = np.asarray(ground_dem_grid, dtype=float)
                    dem_norm = dem_norm - np.nanmin(dem_norm)
                except Exception:
                    dem_norm = ground_dem_grid
                
                sim_vals = np.asarray(ground_sim_grid, dtype=float)
                finite = np.isfinite(sim_vals)
                vmin_g = ground_vmin if ground_vmin is not None else (float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0)
                vmax_g = ground_vmax if ground_vmax is not None else (float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0)
                
                sim_mesh = create_sim_surface_mesh(
                    ground_sim_grid,
                    dem_norm,
                    meshsize=meshsize,
                    z_offset=z_off,
                    cmap_name=ground_colormap,
                    vmin=vmin_g,
                    vmax=vmax_g,
                )
                
                if sim_mesh is not None and hasattr(sim_mesh, 'vertices') and len(sim_mesh.vertices) > 0:
                    gv = np.asarray(sim_mesh.vertices)
                    gf = np.asarray(sim_mesh.faces)
                    
                    gc = getattr(sim_mesh.visual, 'face_colors', None)
                    if gc is None:
                        gc = np.full((len(gf), 3), 180, dtype=np.uint8)
                    else:
                        gc = np.asarray(gc)[:, :3]
                    
                    gm = np.zeros(len(gf), dtype=np.int32)
                    
                    vertex_offset = len(vertices) if len(vertices) > 0 else 0
                    if len(vertices) > 0:
                        vertices = np.vstack([vertices, gv]).astype(np.float32)
                        indices = np.vstack([indices, gf + vertex_offset]).astype(np.int32)
                        colors = np.vstack([colors, gc]).astype(np.uint8)
                        materials = np.concatenate([materials, gm])
                    else:
                        vertices = gv.astype(np.float32)
                        indices = gf.astype(np.int32)
                        colors = gc.astype(np.uint8)
                        materials = gm
        
        if len(vertices) == 0:
            raise ValueError("No geometry to render")
        
        # Set scene once with materials
        self.taichi_renderer.set_scene(vertices, indices, colors, materials)
        
        # Compute scene bounds
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        diagonal = np.linalg.norm(bounds_max - bounds_min)
        
        # Set floor just below the model (like voxel-challenge uses -0.1)
        if floor_enabled:
            floor_height = bounds_min[2] - diagonal * 0.02  # Slightly below minimum z
            self.taichi_renderer.set_floor(floor_height, floor_color)
        else:
            self.taichi_renderer.set_floor(-1e10, floor_color)  # Disabled
        
        camera_radius = diagonal * camera_distance_factor
        camera_height = center[2] + diagonal * camera_height_factor
        look_at = (center[0], center[1], center[2] + diagonal * look_at_z_factor)
        
        filenames = []
        
        for frame in range(num_frames):
            angle = 2.0 * math.pi * frame / num_frames
            cam_x = center[0] + camera_radius * math.cos(angle)
            cam_y = center[1] + camera_radius * math.sin(angle)
            
            self.taichi_renderer.set_camera(
                (cam_x, cam_y, camera_height),
                look_at,
                (0, 0, 1),
                fov
            )
            
            filename = os.path.join(output_directory, f"{file_prefix}_{frame:04d}.png")
            
            if show_progress:
                print(f"\rRendering frame {frame + 1}/{num_frames}", end="", flush=True)
            
            self.taichi_renderer.render_to_file(filename, show_progress=False)
            filenames.append(filename)
        
        if show_progress:
            print(f"\rRendered {num_frames} frames to {output_directory}")
        
        return filenames


def visualize_voxcity_gpu(
    city: VoxCity,
    voxel_color_map: str | dict = "default",
    width: int = 1920,
    height: int = 1080,
    samples_per_pixel: int = 64,
    max_depth: int = 8,
    camera_position: Optional[Tuple[float, float, float]] = None,
    camera_look_at: Optional[Tuple[float, float, float]] = None,
    fov: float = 25.0,
    output_path: Optional[str] = None,
    show_progress: bool = True,
    arch: str = 'gpu',
    # Rotation options
    rotation: bool = False,
    output_directory: str = "output",
    rotation_frames: int = 240,
    rotation_file_prefix: str = "city_rotation",
    # Building simulation overlay
    building_sim_mesh=None,
    building_value_name: str = 'svf_values',
    building_colormap: str = 'viridis',
    building_vmin: Optional[float] = None,
    building_vmax: Optional[float] = None,
    building_nan_color: str = 'gray',
    render_voxel_buildings: bool = False,
    # Ground simulation surface overlay
    ground_sim_grid: Optional[np.ndarray] = None,
    ground_dem_grid: Optional[np.ndarray] = None,
    ground_z_offset: Optional[float] = None,
    ground_view_point_height: Optional[float] = None,
    ground_colormap: str = 'viridis',
    ground_vmin: Optional[float] = None,
    ground_vmax: Optional[float] = None,
) -> np.ndarray | List[str]:
    """
    Visualize a VoxCity using GPU-accelerated ray tracing.
    
    This is the main entry point for GPU rendering, providing a simple
    interface similar to visualize_voxcity() but with GPU acceleration.
    
    Parameters
    ----------
    city : VoxCity
        The VoxCity object to visualize
    voxel_color_map : str or dict
        Color mapping for voxel classes
    width : int
        Output image width
    height : int
        Output image height
    samples_per_pixel : int
        Anti-aliasing quality (higher = better, slower)
    max_depth : int
        Ray bounce depth
    camera_position : tuple, optional
        Camera position (auto-computed if None)
    camera_look_at : tuple, optional
        Camera look-at point (auto-computed if None)
    fov : float
        Field of view in degrees
    output_path : str, optional
        Path to save the rendered image
    show_progress : bool
        Whether to show rendering progress
    arch : str
        Compute architecture ('gpu', 'cpu', 'cuda', 'vulkan', 'metal')
    rotation : bool
        If True, render rotating frames instead of single image
    output_directory : str
        Directory for rotation frames
    rotation_frames : int
        Number of frames for rotation
    rotation_file_prefix : str
        Filename prefix for rotation frames
    building_sim_mesh : trimesh, optional
        Building mesh with simulation results in metadata
    building_value_name : str
        Metadata key for building values (e.g., 'svf_values', 'global', 'direct')
    building_colormap : str
        Matplotlib colormap for building values
    building_vmin : float, optional
        Minimum value for building color scale
    building_vmax : float, optional
        Maximum value for building color scale
    building_nan_color : str
        Color for NaN/invalid building values
    render_voxel_buildings : bool
        Whether to show voxel buildings with sim mesh
    ground_sim_grid : np.ndarray, optional
        2D array of ground-level simulation values (e.g., GVI, solar)
    ground_dem_grid : np.ndarray, optional
        2D DEM array for ground surface positioning
    ground_z_offset : float, optional
        Height offset for ground surface above DEM
    ground_view_point_height : float, optional
        Alternative height parameter for ground surface
    ground_colormap : str
        Matplotlib colormap for ground values
    ground_vmin : float, optional
        Minimum value for ground color scale
    ground_vmax : float, optional
        Maximum value for ground color scale
    
    Returns
    -------
    np.ndarray or List[str]
        If rotation=False: (H, W, 3) RGB image array
        If rotation=True: List of frame file paths
    
    Examples
    --------
    Simple rendering:
    >>> img = visualize_voxcity_gpu(city)
    
    Save to file:
    >>> img = visualize_voxcity_gpu(city, output_path="render.png")
    
    High quality render:
    >>> img = visualize_voxcity_gpu(city, samples_per_pixel=64, width=1920, height=1080)
    
    Rotation animation:
    >>> frames = visualize_voxcity_gpu(city, rotation=True, rotation_frames=120)
    
    With building solar irradiance:
    >>> building_mesh = get_building_solar_irradiance(city, ...)
    >>> img = visualize_voxcity_gpu(city, building_sim_mesh=building_mesh, 
    ...                             building_value_name='global', building_colormap='magma')
    
    With ground-level Green View Index:
    >>> img = visualize_voxcity_gpu(city, ground_sim_grid=gvi_array, ground_colormap='YlGn')
    """
    if not _HAS_TAICHI:
        raise ImportError("Taichi is required for GPU rendering. Install with: pip install taichi")
    
    renderer = GPURenderer(
        width=width,
        height=height,
        samples_per_pixel=samples_per_pixel,
        max_depth=max_depth,
        arch=arch
    )
    
    if rotation:
        return renderer.render_rotation(
            city,
            voxel_color_map=voxel_color_map,
            output_directory=output_directory,
            file_prefix=rotation_file_prefix,
            num_frames=rotation_frames,
            fov=fov,
            show_progress=show_progress,
            # Building overlay
            building_sim_mesh=building_sim_mesh,
            building_value_name=building_value_name,
            building_colormap=building_colormap,
            building_vmin=building_vmin,
            building_vmax=building_vmax,
            building_nan_color=building_nan_color,
            render_voxel_buildings=render_voxel_buildings,
            # Ground overlay
            ground_sim_grid=ground_sim_grid,
            ground_dem_grid=ground_dem_grid,
            ground_z_offset=ground_z_offset,
            ground_view_point_height=ground_view_point_height,
            ground_colormap=ground_colormap,
            ground_vmin=ground_vmin,
            ground_vmax=ground_vmax,
        )
    else:
        return renderer.render_city(
            city,
            voxel_color_map=voxel_color_map,
            camera_position=camera_position,
            camera_look_at=camera_look_at,
            fov=fov,
            output_path=output_path,
            show_progress=show_progress,
            # Building overlay
            building_sim_mesh=building_sim_mesh,
            building_value_name=building_value_name,
            building_colormap=building_colormap,
            building_vmin=building_vmin,
            building_vmax=building_vmax,
            building_nan_color=building_nan_color,
            render_voxel_buildings=render_voxel_buildings,
            # Ground overlay
            ground_sim_grid=ground_sim_grid,
            ground_dem_grid=ground_dem_grid,
            ground_z_offset=ground_z_offset,
            ground_view_point_height=ground_view_point_height,
            ground_colormap=ground_colormap,
            ground_vmin=ground_vmin,
            ground_vmax=ground_vmax,
        )
