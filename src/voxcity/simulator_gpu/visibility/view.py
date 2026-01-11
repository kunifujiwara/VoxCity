"""
View Index and Sky View Factor calculation using Taichi GPU acceleration.

This module emulates the functionality of voxcity.simulator.visibility.view
with GPU-accelerated ray tracing.
"""

import taichi as ti
import numpy as np
import math
from typing import Tuple, Optional, Union, List

from ..core import Vector3, Point3, PI, TWO_PI
from ..init_taichi import ensure_initialized
from ..raytracing import (
    ray_voxel_first_hit,
    ray_voxel_transmissivity,
    ray_aabb_intersect,
)
from .geometry import (
    generate_ray_directions_grid,
    generate_ray_directions_fibonacci,
    generate_hemisphere_directions,
)


@ti.data_oriented
class ViewCalculator:
    """
    GPU-accelerated View Index calculator.
    
    Computes view indices (green view, sky view, custom targets) by tracing rays
    from observer positions through the voxel domain.
    """
    
    def __init__(
        self,
        domain,
        n_azimuth: int = 120,
        n_elevation: int = 20,
        ray_sampling: str = "grid",
        n_rays: int = None
    ):
        """
        Initialize View Calculator.
        
        Args:
            domain: Domain object with grid geometry (simulator_gpu.domain.Domain)
            n_azimuth: Number of azimuthal divisions (for grid sampling)
            n_elevation: Number of elevation divisions (for grid sampling)
            ray_sampling: Sampling method ('grid' or 'fibonacci')
            n_rays: Total rays for fibonacci sampling (default: n_azimuth * n_elevation)
        """
        # Ensure Taichi is initialized before creating any fields
        ensure_initialized()
        
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        
        self.n_azimuth = n_azimuth
        self.n_elevation = n_elevation
        self.ray_sampling = ray_sampling
        self.n_rays = n_rays if n_rays else n_azimuth * n_elevation
        
        # Maximum ray distance
        self.max_dist = domain.get_max_dist()
        
        # Pre-computed ray directions (will be set based on mode)
        self._ray_dirs = None
        self._n_ray_dirs = 0
    
    def _setup_ray_directions(
        self,
        elevation_min: float = -30.0,
        elevation_max: float = 30.0
    ):
        """Setup ray directions based on sampling method."""
        if self.ray_sampling.lower() == "fibonacci":
            dirs_np = generate_ray_directions_fibonacci(
                self.n_rays, elevation_min, elevation_max
            )
        else:
            dirs_np = generate_ray_directions_grid(
                self.n_azimuth, self.n_elevation, elevation_min, elevation_max
            )
        
        self._n_ray_dirs = dirs_np.shape[0]
        self._ray_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(self._n_ray_dirs,))
        self._ray_dirs.from_numpy(dirs_np)
    
    def compute_view_index(
        self,
        voxel_data: np.ndarray = None,
        mode: str = None,
        hit_values: Tuple[int, ...] = None,
        inclusion_mode: bool = True,
        view_point_height: float = 1.5,
        elevation_min_degrees: float = -30.0,
        elevation_max_degrees: float = 30.0,
        tree_k: float = 0.5,
        tree_lad: float = 1.0
    ) -> np.ndarray:
        """
        Compute View Index map.
        
        Args:
            voxel_data: 3D voxel class array (optional if domain has voxel data)
            mode: Predefined mode ('green', 'sky', or None for custom)
            hit_values: Target voxel values to count as visible
            inclusion_mode: If True, count hits on targets; if False, count non-blocked rays
            view_point_height: Observer height above ground (meters)
            elevation_min_degrees: Minimum viewing angle
            elevation_max_degrees: Maximum viewing angle
            tree_k: Tree extinction coefficient
            tree_lad: Leaf area density for trees
        
        Returns:
            2D array of view index values (nx, ny)
        """
        # Set up mode-specific parameters
        if mode == 'green':
            hit_values = (-2, 2, 5, 6, 7, 8)
            inclusion_mode = True
        elif mode == 'sky':
            hit_values = (0,)
            inclusion_mode = False
        elif hit_values is None:
            raise ValueError("For custom mode, you must provide hit_values.")
        
        # Setup ray directions
        self._setup_ray_directions(elevation_min_degrees, elevation_max_degrees)
        
        # Convert view height to voxels
        view_height_voxel = int(view_point_height / self.dz)
        
        # Prepare output
        vi_map = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        
        # Prepare masks from voxel data
        if voxel_data is None:
            # Use domain's is_solid and is_tree
            is_tree = self.domain.is_tree
            is_solid = self.domain.is_solid
            is_target = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_allowed = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_blocker = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_walkable = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            
            # Initialize target/allowed based on mode
            self._init_target_masks_from_domain(
                is_tree, is_solid, is_target, is_allowed, is_blocker, is_walkable,
                inclusion_mode, mode == 'green'
            )
        else:
            # Create masks from voxel_data
            is_tree = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_solid = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_target = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_allowed = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_blocker = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_walkable = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            
            # Convert hit_values to array for kernel
            hit_values_arr = np.array(hit_values, dtype=np.int32)
            
            self._setup_masks_from_voxel_data(
                voxel_data, hit_values_arr, inclusion_mode,
                is_tree, is_solid, is_target, is_allowed, is_blocker, is_walkable
            )
        
        # Compute transmissivity per voxel for trees
        tree_att = float(math.exp(-tree_k * tree_lad * self.dz))
        
        # Run GPU computation
        self._compute_vi_map_kernel(
            vi_map, view_height_voxel,
            is_tree, is_solid, is_target, is_allowed, is_blocker, is_walkable,
            inclusion_mode, tree_att
        )
        
        # Flip Y-axis to match VoxCity coordinate system
        result = np.flipud(vi_map.to_numpy())
        return result
    
    @ti.kernel
    def _init_target_masks_from_domain(
        self,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_allowed: ti.template(),
        is_blocker: ti.template(),
        is_walkable: ti.template(),
        inclusion_mode: ti.i32,
        green_mode: ti.i32
    ):
        """Initialize target masks from domain's is_tree and is_solid."""
        for i, j, k in is_target:
            # Without voxel_data, assume all surfaces are walkable
            is_walkable[i, j, k] = 1
            
            if green_mode == 1:
                # Green mode: trees are targets
                is_target[i, j, k] = is_tree[i, j, k]
                is_allowed[i, j, k] = is_tree[i, j, k]
                # Blockers are solids (non-tree)
                blocker = 0
                if is_solid[i, j, k] == 1 and is_tree[i, j, k] == 0:
                    blocker = 1
                is_blocker[i, j, k] = blocker
            else:
                # Sky mode: no targets, just check for blockers
                is_target[i, j, k] = 0
                is_allowed[i, j, k] = 0
                is_blocker[i, j, k] = 0
    
    def _setup_masks_from_voxel_data(
        self,
        voxel_data: np.ndarray,
        hit_values: np.ndarray,
        inclusion_mode: bool,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_allowed: ti.template(),
        is_blocker: ti.template(),
        is_walkable: ti.template()
    ):
        """Setup masks from voxel data array."""
        n_hits = len(hit_values)
        self._setup_masks_kernel(
            voxel_data, hit_values, n_hits, inclusion_mode,
            is_tree, is_solid, is_target, is_allowed, is_blocker, is_walkable
        )
    
    @ti.kernel
    def _setup_masks_kernel(
        self,
        voxel_data: ti.types.ndarray(),
        hit_values: ti.types.ndarray(),
        n_hits: ti.i32,
        inclusion_mode: ti.i32,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_allowed: ti.template(),
        is_blocker: ti.template(),
        is_walkable: ti.template()
    ):
        for i, j, k in is_tree:
            val = voxel_data[i, j, k]
            
            # Tree check (code -2)
            tree = 0
            if val == -2:
                tree = 1
            is_tree[i, j, k] = tree
            
            # Solid check (non-zero, non-tree)
            solid = 0
            if val != 0 and val != -2:
                solid = 1
            is_solid[i, j, k] = solid
            
            # Target check
            target = 0
            for h in range(n_hits):
                if val == hit_values[h]:
                    target = 1
            is_target[i, j, k] = target
            
            # Walkable: surfaces that are valid observer positions
            # Exclude: water (7, 8, 9) and negative values (ground -1, tree -2, building -3, etc.)
            walkable = 1
            if val == 7 or val == 8 or val == 9:  # Water
                walkable = 0
            elif val < 0:  # Ground, trees, buildings, landmarks, etc.
                walkable = 0
            is_walkable[i, j, k] = walkable
            
            # Set up allowed and blocker based on mode
            if inclusion_mode == 1:
                is_allowed[i, j, k] = target
                # Blocker: non-tree, non-target, non-empty
                blocker = 0
                if val != 0 and tree == 0 and target == 0:
                    blocker = 1
                is_blocker[i, j, k] = blocker
            else:
                is_allowed[i, j, k] = target
                is_blocker[i, j, k] = 0
    
    @ti.kernel
    def _compute_vi_map_kernel(
        self,
        vi_map: ti.template(),
        view_height_voxel: ti.i32,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_allowed: ti.template(),
        is_blocker: ti.template(),
        is_walkable: ti.template(),
        inclusion_mode: ti.i32,
        tree_att: ti.f32
    ):
        """Compute View Index map using GPU parallel processing."""
        for x, y in vi_map:
            # Find observer position (first non-solid cell above ground)
            # Allow being inside tree canopy
            observer_z = -1
            surface_walkable = 0
            for z in range(1, self.nz):
                # Current cell is not solid (but can be inside tree)
                current_not_solid = 1 if is_solid[x, y, z] == 0 else 0
                # Below cell has something (ground, building, or tree)
                below_has_something = is_solid[x, y, z-1] + is_tree[x, y, z-1]
                
                if current_not_solid == 1 and below_has_something > 0:
                    # Found ground level - check if walkable
                    surface_walkable = is_walkable[x, y, z-1]
                    observer_z = z + view_height_voxel
                    break
            
            # Mark as invalid if no observer position found or surface not walkable
            # (water, building tops, etc. are not walkable)
            if observer_z < 0 or observer_z >= self.nz or surface_walkable == 0:
                vi_map[x, y] = ti.cast(float('nan'), ti.f32)
                continue
            
            # Trace rays and count visibility
            visibility_sum = 0.0
            valid_rays = 0
            
            for r in range(self._n_ray_dirs):
                ray_dir = self._ray_dirs[r]
                
                # Trace ray through voxels
                hit, trans = self._trace_ray_vi(
                    x, y, observer_z, ray_dir,
                    is_tree, is_solid, is_target, is_allowed, is_blocker,
                    inclusion_mode, tree_att
                )
                
                if inclusion_mode == 1:
                    if hit == 1:
                        visibility_sum += 1.0
                else:
                    if hit == 0:
                        visibility_sum += trans
                
                valid_rays += 1
            
            if valid_rays > 0:
                vi_map[x, y] = visibility_sum / valid_rays
            else:
                vi_map[x, y] = 0.0
    
    @ti.func
    def _trace_ray_vi(
        self,
        ox: ti.i32,
        oy: ti.i32,
        oz: ti.i32,
        ray_dir: Vector3,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_allowed: ti.template(),
        is_blocker: ti.template(),
        inclusion_mode: ti.i32,
        tree_att: ti.f32
    ):
        """
        Trace a ray for view index calculation.
        
        Returns:
            (hit, transmissivity) where hit=1 means target was hit (inclusion) 
            or ray was blocked (exclusion)
        """
        hit = 0
        trans = 1.0
        
        # Start from observer position
        x = ti.cast(ox, ti.f32) + 0.5
        y = ti.cast(oy, ti.f32) + 0.5
        z = ti.cast(oz, ti.f32) + 0.5
        
        i = ox
        j = oy
        k = oz
        
        # Track starting position to skip it
        start_i = ox
        start_j = oy
        start_k = oz
        
        step_x = 1 if ray_dir[0] >= 0 else -1
        step_y = 1 if ray_dir[1] >= 0 else -1
        step_z = 1 if ray_dir[2] >= 0 else -1
        
        BIG = 1e30
        t_max_x, t_max_y, t_max_z = BIG, BIG, BIG
        t_delta_x, t_delta_y, t_delta_z = BIG, BIG, BIG
        
        if ti.abs(ray_dir[0]) > 1e-10:
            t_max_x = ((i + (1 if step_x > 0 else 0)) - x) / ray_dir[0]
            t_delta_x = ti.abs(1.0 / ray_dir[0])
        if ti.abs(ray_dir[1]) > 1e-10:
            t_max_y = ((j + (1 if step_y > 0 else 0)) - y) / ray_dir[1]
            t_delta_y = ti.abs(1.0 / ray_dir[1])
        if ti.abs(ray_dir[2]) > 1e-10:
            t_max_z = ((k + (1 if step_z > 0 else 0)) - z) / ray_dir[2]
            t_delta_z = ti.abs(1.0 / ray_dir[2])
        
        max_steps = self.nx + self.ny + self.nz
        done = 0
        first_step = 1  # Skip the starting voxel
        
        for _ in range(max_steps):
            if done == 0:
                # Bounds check
                if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
                    done = 1
                else:
                    # Skip the starting voxel (observer's position)
                    at_start = 0
                    if i == start_i and j == start_j and k == start_k:
                        at_start = 1
                    
                    if at_start == 0:
                        # Check tree - accumulate transmissivity
                        if is_tree[i, j, k] == 1:
                            trans *= tree_att
                            if trans < 0.01:
                                if inclusion_mode == 0:
                                    hit = 1  # Blocked in exclusion mode
                                done = 1
                        
                        if done == 0:
                            if inclusion_mode == 1:
                                # Inclusion mode: looking for target (including trees as targets)
                                if is_target[i, j, k] == 1:
                                    hit = 1
                                    done = 1
                                elif is_blocker[i, j, k] == 1:
                                    hit = 0
                                    done = 1
                            else:
                                # Exclusion mode: looking for non-allowed
                                if is_tree[i, j, k] == 0 and is_allowed[i, j, k] == 0 and is_solid[i, j, k] == 1:
                                    hit = 1  # Blocked
                                    done = 1
                    
                    if done == 0:
                        # Step to next voxel
                        if t_max_x < t_max_y:
                            if t_max_x < t_max_z:
                                t_max_x += t_delta_x
                                i += step_x
                            else:
                                t_max_z += t_delta_z
                                k += step_z
                        else:
                            if t_max_y < t_max_z:
                                t_max_y += t_delta_y
                                j += step_y
                            else:
                                t_max_z += t_delta_z
                                k += step_z
        
        return hit, trans
    
    def compute_sky_view_factor(
        self,
        voxel_data: np.ndarray = None,
        view_point_height: float = 1.5,
        n_azimuth: int = 120,
        n_elevation: int = 20,
        tree_k: float = 0.6,
        tree_lad: float = 1.0
    ) -> np.ndarray:
        """
        Compute Sky View Factor map.
        
        Args:
            voxel_data: 3D voxel class array (optional)
            view_point_height: Observer height above ground (meters)
            n_azimuth: Number of azimuthal divisions
            n_elevation: Number of elevation divisions
            tree_k: Tree extinction coefficient
            tree_lad: Leaf area density
        
        Returns:
            2D array of SVF values (nx, ny)
        """
        # SVF uses upper hemisphere (0-90 degrees elevation)
        self.n_azimuth = n_azimuth
        self.n_elevation = n_elevation
        
        return self.compute_view_index(
            voxel_data=voxel_data,
            mode='sky',
            view_point_height=view_point_height,
            elevation_min_degrees=0.0,
            elevation_max_degrees=90.0,
            tree_k=tree_k,
            tree_lad=tree_lad
        )


@ti.data_oriented
class SurfaceViewFactorCalculator:
    """
    GPU-accelerated Surface View Factor calculator.
    
    Computes view factors for building surface faces by tracing rays from
    face centers through the voxel domain.
    
    This emulates voxcity.simulator.visibility.view.get_surface_view_factor
    using Taichi GPU acceleration.
    """
    
    def __init__(
        self,
        domain,
        n_azimuth: int = 120,
        n_elevation: int = 20,
        ray_sampling: str = "grid",
        n_rays: int = None
    ):
        """
        Initialize Surface View Factor Calculator.
        
        Args:
            domain: Domain object with grid geometry
            n_azimuth: Number of azimuthal divisions
            n_elevation: Number of elevation divisions
            ray_sampling: 'grid' or 'fibonacci'
            n_rays: Total rays for fibonacci sampling
        """
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        self.meshsize = domain.dx  # Assuming uniform grid
        
        self.n_azimuth = n_azimuth
        self.n_elevation = n_elevation
        self.ray_sampling = ray_sampling
        self.n_rays = n_rays if n_rays else n_azimuth * n_elevation
        
        # Pre-computed hemisphere directions (upper hemisphere only for surfaces)
        self._hemisphere_dirs = None
        self._n_hemisphere_dirs = 0
        self._setup_hemisphere_directions()
    
    def _setup_hemisphere_directions(self):
        """Setup hemisphere ray directions for surface view factor."""
        if self.ray_sampling.lower() == "fibonacci":
            dirs_np = generate_ray_directions_fibonacci(
                self.n_rays, 0.0, 90.0
            )
        else:
            dirs_np = generate_ray_directions_grid(
                self.n_azimuth, self.n_elevation, 0.0, 90.0
            )
        
        self._n_hemisphere_dirs = dirs_np.shape[0]
        self._hemisphere_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(self._n_hemisphere_dirs,))
        self._hemisphere_dirs.from_numpy(dirs_np.astype(np.float32))
    
    def compute_surface_view_factor(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        voxel_data: np.ndarray = None,
        target_values: Tuple[int, ...] = (0,),
        inclusion_mode: bool = False,
        tree_k: float = 0.6,
        tree_lad: float = 1.0,
        boundary_epsilon: float = None
    ) -> np.ndarray:
        """
        Compute view factors for building surface faces.
        
        Args:
            face_centers: Array of face center positions (n_faces, 3) in world coords
            face_normals: Array of face normal vectors (n_faces, 3)
            voxel_data: 3D voxel class array
            target_values: Target voxel values for visibility
            inclusion_mode: If True, count hits on targets; if False, count unblocked rays
            tree_k: Tree extinction coefficient
            tree_lad: Leaf area density
            boundary_epsilon: Epsilon for boundary detection
        
        Returns:
            1D array of view factor values for each face
        """
        n_faces = face_centers.shape[0]
        
        if boundary_epsilon is None:
            boundary_epsilon = self.meshsize * 0.05
        
        # Grid bounds in world coordinates
        grid_bounds_real = np.array([
            [0.0, 0.0, 0.0],
            [self.nx * self.meshsize, self.ny * self.meshsize, self.nz * self.meshsize]
        ], dtype=np.float32)
        
        # Prepare Taichi fields
        face_centers_ti = ti.Vector.field(3, dtype=ti.f32, shape=(n_faces,))
        face_normals_ti = ti.Vector.field(3, dtype=ti.f32, shape=(n_faces,))
        face_vf_values = ti.field(dtype=ti.f32, shape=(n_faces,))
        
        face_centers_ti.from_numpy(face_centers.astype(np.float32))
        face_normals_ti.from_numpy(face_normals.astype(np.float32))
        
        # Prepare masks
        if voxel_data is not None:
            is_tree = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_solid = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_target = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_opaque = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            
            target_values_arr = np.array(target_values, dtype=np.int32)
            self._setup_surface_masks(
                voxel_data, target_values_arr, len(target_values_arr), inclusion_mode,
                is_tree, is_solid, is_target, is_opaque
            )
        else:
            is_tree = self.domain.is_tree
            is_solid = self.domain.is_solid
            is_target = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_opaque = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
        
        # Tree attenuation
        tree_att = float(math.exp(-tree_k * tree_lad * self.meshsize))
        att_cutoff = 0.01
        trees_are_targets = (-2 in target_values) and inclusion_mode
        
        # Run GPU computation
        self._compute_surface_vf_kernel(
            face_centers_ti, face_normals_ti, face_vf_values,
            is_tree, is_solid, is_target, is_opaque,
            tree_att, att_cutoff, inclusion_mode, int(trees_are_targets),
            grid_bounds_real, boundary_epsilon
        )
        
        return face_vf_values.to_numpy()
    
    @ti.kernel
    def _setup_surface_masks(
        self,
        voxel_data: ti.types.ndarray(),
        target_values: ti.types.ndarray(),
        n_targets: ti.i32,
        inclusion_mode: ti.i32,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_opaque: ti.template()
    ):
        for i, j, k in is_tree:
            val = voxel_data[i, j, k]
            
            # Tree check
            tree = 0
            if val == -2:
                tree = 1
            is_tree[i, j, k] = tree
            
            # Solid check
            solid = 0
            if val != 0:
                solid = 1
            is_solid[i, j, k] = solid
            
            # Target check
            target = 0
            for t in range(n_targets):
                if val == target_values[t]:
                    target = 1
            is_target[i, j, k] = target
            
            # Opaque: non-zero, non-tree, non-target (for inclusion mode)
            opaque = 0
            if inclusion_mode == 1:
                if val != 0 and tree == 0 and target == 0:
                    opaque = 1
            else:
                if val != 0 and tree == 0:
                    opaque = 1
            is_opaque[i, j, k] = opaque
    
    @ti.kernel
    def _compute_surface_vf_kernel(
        self,
        face_centers: ti.template(),
        face_normals: ti.template(),
        face_vf_values: ti.template(),
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_opaque: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32,
        inclusion_mode: ti.i32,
        trees_are_targets: ti.i32,
        grid_bounds: ti.types.ndarray(),
        boundary_epsilon: ti.f32
    ):
        """Compute surface view factors using GPU parallel processing."""
        for f in face_vf_values:
            center = face_centers[f]
            normal = face_normals[f]
            
            # Check if face is on domain boundary (vertical and on edge)
            is_vertical = ti.abs(normal[2]) < 0.01
            on_x_min = ti.abs(center[0] - grid_bounds[0, 0]) < boundary_epsilon
            on_y_min = ti.abs(center[1] - grid_bounds[0, 1]) < boundary_epsilon
            on_x_max = ti.abs(center[0] - grid_bounds[1, 0]) < boundary_epsilon
            on_y_max = ti.abs(center[1] - grid_bounds[1, 1]) < boundary_epsilon
            
            is_boundary = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
            
            if is_boundary:
                face_vf_values[f] = ti.cast(float('nan'), ti.f32)
            else:
                # Build local coordinate system for face
                u, v, n = self._build_face_basis(normal)
                
                # Origin: face center offset by normal (in voxel coordinates)
                meshsize = self.dx
                ox = center[0] / meshsize + n[0] * 0.51
                oy = center[1] / meshsize + n[1] * 0.51
                oz = center[2] / meshsize + n[2] * 0.51
                
                vis_sum = 0.0
                valid_count = 0
                
                # Trace rays in hemisphere
                for r in range(self._n_hemisphere_dirs):
                    local_dir = self._hemisphere_dirs[r]
                    
                    # Transform to world direction
                    dx = u[0]*local_dir[0] + v[0]*local_dir[1] + n[0]*local_dir[2]
                    dy = u[1]*local_dir[0] + v[1]*local_dir[1] + n[1]*local_dir[2]
                    dz = u[2]*local_dir[0] + v[2]*local_dir[1] + n[2]*local_dir[2]
                    
                    world_dir = ti.Vector([dx, dy, dz])
                    
                    # Only trace rays going outward from surface
                    dot = world_dir.dot(n)
                    if dot > 0.0:
                        contrib = self._trace_surface_ray(
                            ox, oy, oz, world_dir,
                            is_tree, is_solid, is_target, is_opaque,
                            tree_att, att_cutoff, inclusion_mode, trees_are_targets
                        )
                        vis_sum += contrib
                        valid_count += 1
                
                if valid_count > 0:
                    face_vf_values[f] = vis_sum / valid_count
                else:
                    face_vf_values[f] = 0.0
    
    @ti.func
    def _build_face_basis(self, normal: ti.template()) -> ti.template():
        """Build orthonormal basis (u, v, n) for a surface normal."""
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
    
    @ti.func
    def _trace_surface_ray(
        self,
        ox: ti.f32,
        oy: ti.f32,
        oz: ti.f32,
        ray_dir: ti.template(),
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_target: ti.template(),
        is_opaque: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32,
        inclusion_mode: ti.i32,
        trees_are_targets: ti.i32
    ) -> ti.f32:
        """Trace ray from surface for view factor calculation."""
        T = 1.0
        result = 0.0
        
        x = ox
        y = oy
        z = oz
        
        i = ti.cast(ti.floor(ox), ti.i32)
        j = ti.cast(ti.floor(oy), ti.i32)
        k = ti.cast(ti.floor(oz), ti.i32)
        
        step_x = 1 if ray_dir[0] >= 0 else -1
        step_y = 1 if ray_dir[1] >= 0 else -1
        step_z = 1 if ray_dir[2] >= 0 else -1
        
        BIG = 1e30
        t_max_x, t_max_y, t_max_z = BIG, BIG, BIG
        t_delta_x, t_delta_y, t_delta_z = BIG, BIG, BIG
        
        if ti.abs(ray_dir[0]) > 1e-10:
            t_max_x = ((i + (1 if step_x > 0 else 0)) - x) / ray_dir[0]
            t_delta_x = ti.abs(1.0 / ray_dir[0])
        if ti.abs(ray_dir[1]) > 1e-10:
            t_max_y = ((j + (1 if step_y > 0 else 0)) - y) / ray_dir[1]
            t_delta_y = ti.abs(1.0 / ray_dir[1])
        if ti.abs(ray_dir[2]) > 1e-10:
            t_max_z = ((k + (1 if step_z > 0 else 0)) - z) / ray_dir[2]
            t_delta_z = ti.abs(1.0 / ray_dir[2])
        
        max_steps = self.nx + self.ny + self.nz
        done = 0
        
        for _ in range(max_steps):
            if done == 0:
                # Bounds check - ray escaped
                if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
                    if inclusion_mode == 1:
                        result = 0.0
                    else:
                        result = T
                    done = 1
                else:
                    # Check opaque (blocker)
                    if is_opaque[i, j, k] == 1:
                        result = 0.0
                        done = 1
                    elif is_tree[i, j, k] == 1:
                        # Tree attenuation
                        T *= tree_att
                        if T < att_cutoff:
                            result = 0.0
                            done = 1
                        elif trees_are_targets == 1:
                            # Trees count as partial visibility
                            result = 1.0 - T
                            done = 1
                    elif inclusion_mode == 1 and is_target[i, j, k] == 1:
                        # Hit target in inclusion mode
                        result = 1.0
                        done = 1
                    
                    if done == 0:
                        # Step to next voxel
                        if t_max_x < t_max_y:
                            if t_max_x < t_max_z:
                                t_max_x += t_delta_x
                                i += step_x
                            else:
                                t_max_z += t_delta_z
                                k += step_z
                        else:
                            if t_max_y < t_max_z:
                                t_max_y += t_delta_y
                                j += step_y
                            else:
                                t_max_z += t_delta_z
                                k += step_z
        
        return result


# Convenience functions
def compute_view_index_map(
    domain,
    voxel_data: np.ndarray = None,
    mode: str = 'green',
    **kwargs
) -> np.ndarray:
    """
    Compute View Index map.
    
    Args:
        domain: Domain object
        voxel_data: 3D voxel class array
        mode: 'green', 'sky', or custom
        **kwargs: Additional parameters for ViewCalculator
    
    Returns:
        2D view index map
    """
    calc = ViewCalculator(domain)
    return calc.compute_view_index(voxel_data=voxel_data, mode=mode, **kwargs)


def compute_sky_view_factor_map(
    domain,
    voxel_data: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Compute Sky View Factor map.
    
    Args:
        domain: Domain object
        voxel_data: 3D voxel class array
        **kwargs: Additional parameters
    
    Returns:
        2D SVF map
    """
    calc = ViewCalculator(domain)
    return calc.compute_sky_view_factor(voxel_data=voxel_data, **kwargs)
