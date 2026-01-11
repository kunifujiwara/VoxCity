"""
Landmark visibility calculation using Taichi GPU acceleration.

This module emulates the functionality of voxcity.simulator.visibility.landmark
with GPU-accelerated ray tracing.
"""

import taichi as ti
import numpy as np
import math
from typing import Tuple, Optional, List

from ..core import Vector3, Point3
from ..init_taichi import ensure_initialized
from ..raytracing import ray_trace_to_target


@ti.data_oriented
class LandmarkVisibilityCalculator:
    """
    GPU-accelerated Landmark Visibility calculator.
    
    Computes visibility of landmark buildings from observation points
    throughout the domain.
    """
    
    def __init__(self, domain):
        """
        Initialize Landmark Visibility Calculator.
        
        Args:
            domain: Domain object with grid geometry
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
        
        # Landmark positions (will be set later)
        self._landmark_positions = None
        self._n_landmarks = 0
    
    def set_landmarks_from_positions(self, positions: np.ndarray):
        """
        Set landmark positions directly.
        
        Args:
            positions: Array of shape (n_landmarks, 3) with (x, y, z) coordinates
        """
        self._n_landmarks = positions.shape[0]
        self._landmark_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self._n_landmarks,))
        self._landmark_positions.from_numpy(positions.astype(np.float32))
    
    def set_landmarks_from_voxel_value(self, voxel_data: np.ndarray, landmark_value: int = -30):
        """
        Set landmark positions from voxel data based on a marker value.
        
        Args:
            voxel_data: 3D voxel class array
            landmark_value: Voxel value marking landmarks
        """
        positions = np.argwhere(voxel_data == landmark_value).astype(np.float32)
        if positions.shape[0] == 0:
            raise ValueError(f"No landmark with value {landmark_value} found in voxel data.")
        self.set_landmarks_from_positions(positions)
    
    def compute_visibility_map(
        self,
        voxel_data: np.ndarray = None,
        view_height_voxel: int = 0,
        tree_k: float = 0.6,
        tree_lad: float = 1.0
    ) -> np.ndarray:
        """
        Compute landmark visibility map.
        
        Args:
            voxel_data: 3D voxel class array (optional if domain has masks)
            view_height_voxel: Observer height in voxels above ground
            tree_k: Tree extinction coefficient
            tree_lad: Leaf area density
        
        Returns:
            2D array with 1 where landmark is visible, 0 otherwise, nan for invalid
        """
        if self._landmark_positions is None or self._n_landmarks == 0:
            raise ValueError("No landmarks set. Call set_landmarks_* first.")
        
        # Prepare output
        visibility_map = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        
        # Prepare masks
        if voxel_data is not None:
            is_tree = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_solid = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_walkable = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            self._setup_masks_from_voxel(voxel_data, is_tree, is_solid, is_walkable)
        else:
            is_tree = self.domain.is_tree
            is_solid = self.domain.is_solid
            # Create walkable mask - assume all non-solid, non-tree surfaces are walkable
            is_walkable = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            self._init_walkable_from_domain(is_tree, is_solid, is_walkable)
        
        # Tree attenuation
        tree_att = float(math.exp(-tree_k * tree_lad * self.dz))
        att_cutoff = 0.01
        
        # Run GPU computation
        self._compute_visibility_map_kernel(
            visibility_map, view_height_voxel,
            is_tree, is_solid, is_walkable, tree_att, att_cutoff
        )
        
        # Return flipped result
        result = visibility_map.to_numpy()
        return np.flipud(result)
    
    def _setup_masks_from_voxel(
        self,
        voxel_data: np.ndarray,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_walkable: ti.template()
    ):
        """Setup tree, solid, and walkable masks from voxel data."""
        self._setup_masks_kernel(voxel_data, is_tree, is_solid, is_walkable)
    
    @ti.kernel
    def _init_walkable_from_domain(
        self,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_walkable: ti.template()
    ):
        """Initialize walkable mask from domain masks (assume all walkable)."""
        for i, j, k in is_walkable:
            # Without voxel_data, assume surfaces are walkable if not tree/solid
            is_walkable[i, j, k] = 1
    
    @ti.kernel
    def _setup_masks_kernel(
        self,
        voxel_data: ti.types.ndarray(),
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_walkable: ti.template()
    ):
        for i, j, k in is_tree:
            val = voxel_data[i, j, k]
            
            tree = 0
            if val == -2:
                tree = 1
            is_tree[i, j, k] = tree
            
            # Solid blocks rays, but NOT landmark voxels (val == -30)
            solid = 0
            if val != 0 and val != -2 and val != -30:
                solid = 1
            is_solid[i, j, k] = solid
            
            # Walkable: surfaces that are valid observer positions
            # Exclude: water (7, 8, 9) and negative values (ground -1, tree -2, building -3, etc.)
            # A surface is walkable if the voxel value is positive and not water
            walkable = 1
            if val == 7 or val == 8 or val == 9:  # Water
                walkable = 0
            elif val < 0:  # Ground, trees, buildings, landmarks, etc.
                walkable = 0
            is_walkable[i, j, k] = walkable
    
    @ti.kernel
    def _compute_visibility_map_kernel(
        self,
        visibility_map: ti.template(),
        view_height_voxel: ti.i32,
        is_tree: ti.template(),
        is_solid: ti.template(),
        is_walkable: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32
    ):
        """Compute landmark visibility map using GPU parallel processing."""
        for x, y in visibility_map:
            # Find observer position (first air voxel above a solid surface)
            observer_z = -1
            surface_walkable = 0
            for z in range(1, self.nz):
                val_above = is_solid[x, y, z] + is_tree[x, y, z]
                val_below = is_solid[x, y, z-1] + is_tree[x, y, z-1]
                
                if val_above == 0 and val_below > 0:
                    # Found ground level - check if walkable
                    surface_walkable = is_walkable[x, y, z-1]
                    observer_z = z + view_height_voxel
                    break
            
            # Mark as invalid if no observer position found or surface not walkable
            # (water, building tops, etc. are not walkable)
            if observer_z < 0 or observer_z >= self.nz or surface_walkable == 0:
                visibility_map[x, y] = ti.cast(float('nan'), ti.f32)
                continue
            
            # Check visibility to any landmark
            visible = 0
            origin = Vector3(ti.cast(x, ti.f32), ti.cast(y, ti.f32), ti.cast(observer_z, ti.f32))
            
            for lm in range(self._n_landmarks):
                if visible == 0:
                    target = self._landmark_positions[lm]
                    
                    vis = self._trace_to_landmark(
                        origin, target,
                        is_tree, is_solid,
                        tree_att, att_cutoff
                    )
                    
                    if vis == 1:
                        visible = 1
            
            visibility_map[x, y] = ti.cast(visible, ti.f32)
    
    @ti.func
    def _trace_to_landmark(
        self,
        origin: Vector3,
        target: Vector3,
        is_tree: ti.template(),
        is_solid: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32
    ) -> ti.i32:
        """Trace ray from origin to target landmark."""
        diff = target - origin
        dist = diff.norm()
        
        visible = 1
        
        if dist < 0.01:
            visible = 1
        else:
            ray_dir = diff / dist
            
            ox, oy, oz = origin[0], origin[1], origin[2]
            x = ox + 0.5
            y = oy + 0.5
            z = oz + 0.5
            
            i = ti.cast(ti.floor(ox), ti.i32)
            j = ti.cast(ti.floor(oy), ti.i32)
            k = ti.cast(ti.floor(oz), ti.i32)
            
            ti_x = ti.cast(ti.floor(target[0]), ti.i32)
            tj_y = ti.cast(ti.floor(target[1]), ti.i32)
            tk_z = ti.cast(ti.floor(target[2]), ti.i32)
            
            step_x = 1 if ray_dir[0] >= 0 else -1
            step_y = 1 if ray_dir[1] >= 0 else -1
            step_z = 1 if ray_dir[2] >= 0 else -1
            
            BIG = 1e30
            t_max_x, t_max_y, t_max_z = BIG, BIG, BIG
            t_delta_x, t_delta_y, t_delta_z = BIG, BIG, BIG
            
            if ray_dir[0] != 0.0:
                t_max_x = ((i + (1 if step_x > 0 else 0)) - x) / ray_dir[0]
                t_delta_x = ti.abs(1.0 / ray_dir[0])
            if ray_dir[1] != 0.0:
                t_max_y = ((j + (1 if step_y > 0 else 0)) - y) / ray_dir[1]
                t_delta_y = ti.abs(1.0 / ray_dir[1])
            if ray_dir[2] != 0.0:
                t_max_z = ((k + (1 if step_z > 0 else 0)) - z) / ray_dir[2]
                t_delta_z = ti.abs(1.0 / ray_dir[2])
            
            T = 1.0
            max_steps = self.nx + self.ny + self.nz
            done = 0
            
            for _ in range(max_steps):
                if done == 0:
                    # Check bounds
                    if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
                        visible = 0
                        done = 1
                    # Check if reached target
                    elif i == ti_x and j == tj_y and k == tk_z:
                        visible = 1
                        done = 1
                    # Check for solid blocker (not the target)
                    elif is_solid[i, j, k] == 1:
                        visible = 0
                        done = 1
                    # Check for tree attenuation
                    elif is_tree[i, j, k] == 1:
                        T *= tree_att
                        if T < att_cutoff:
                            visible = 0
                            done = 1
                
                # Move to next voxel using 3D DDA
                if done == 0:
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
        
        return visible


def mark_building_by_id(
    voxcity_grid_ori: np.ndarray,
    building_id_grid_ori: np.ndarray,
    ids: List[int],
    mark: int = -30
) -> np.ndarray:
    """
    Mark specific buildings in voxel data with a marker value.
    
    Args:
        voxcity_grid_ori: 3D voxel class array
        building_id_grid_ori: 2D array of building IDs (VoxCity format - needs flipud to match voxel_data)
        ids: List of building IDs to mark
        mark: Marker value to use
    
    Returns:
        Modified voxel_data copy
    """
    voxel_data = voxcity_grid_ori.copy()
    
    # VoxCity building_id_grid is flipped relative to voxel_data coordinate system
    # We need to flip it to align with voxel_data
    building_id_grid_aligned = np.flipud(building_id_grid_ori)
    
    # Find positions where building IDs match
    positions = np.where(np.isin(building_id_grid_aligned, ids))
    for i in range(len(positions[0])):
        x, y = positions[0][i], positions[1][i]
        z_mask = voxel_data[x, y, :] == -3  # Building class
        voxel_data[x, y, z_mask] = mark
    
    return voxel_data


def compute_landmark_visibility(
    voxel_data: np.ndarray,
    target_value: int = -30,
    view_height_voxel: int = 0,
    colormap: str = 'viridis'
) -> np.ndarray:
    """VoxCity-compatible landmark visibility on raw voxel data.

    Matches `voxcity.simulator.visibility.landmark.compute_landmark_visibility`.

    Notes:
        - Uses Taichi GPU ray tracing underneath.
        - Returns a 2D map flipped with `np.flipud`, consistent with VoxCity.
    """
    from ..domain import Domain

    if voxel_data.ndim != 3:
        raise ValueError("voxel_data must be a 3D array")

    nx, ny, nz = voxel_data.shape
    domain = Domain(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
    calc = LandmarkVisibilityCalculator(domain)
    calc.set_landmarks_from_voxel_value(voxel_data, landmark_value=int(target_value))
    visibility_map = calc.compute_visibility_map(
        voxel_data=voxel_data,
        view_height_voxel=int(view_height_voxel),
    )

    # Plot (VoxCity function always plots)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        cmap = plt.cm.get_cmap(colormap, 2).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        plt.imshow(visibility_map, origin='lower', cmap=cmap, vmin=0, vmax=1)
        visible_patch = mpatches.Patch(color=cmap(1.0), label='Visible (1)')
        not_visible_patch = mpatches.Patch(color=cmap(0.0), label='Not Visible (0)')
        plt.legend(handles=[visible_patch, not_visible_patch], loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.axis('off')
        plt.show()
    except Exception:
        pass

    return visibility_map


@ti.data_oriented
class SurfaceLandmarkVisibilityCalculator:
    """
    GPU-accelerated Surface Landmark Visibility calculator.
    
    Computes visibility of landmarks from building surface faces
    using Taichi GPU acceleration.
    
    This emulates voxcity.simulator.visibility.landmark.get_surface_landmark_visibility.
    """
    
    def __init__(self, domain):
        """
        Initialize Surface Landmark Visibility Calculator.
        
        Args:
            domain: Domain object with grid geometry
        """
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        self.meshsize = domain.dx
        
        # Landmark positions
        self._landmark_positions = None
        self._n_landmarks = 0
    
    def set_landmarks_from_positions(self, positions: np.ndarray):
        """
        Set landmark positions directly.
        
        Args:
            positions: Array of shape (n_landmarks, 3) with (x, y, z) coordinates in voxels
        """
        self._n_landmarks = positions.shape[0]
        self._landmark_positions = ti.Vector.field(3, dtype=ti.f32, shape=(self._n_landmarks,))
        self._landmark_positions.from_numpy(positions.astype(np.float32))
    
    def set_landmarks_from_voxel_value(self, voxel_data: np.ndarray, landmark_value: int = -30):
        """
        Set landmark positions from voxel data based on a marker value.
        
        Args:
            voxel_data: 3D voxel class array
            landmark_value: Voxel value marking landmarks
        """
        positions = np.argwhere(voxel_data == landmark_value).astype(np.float32)
        if positions.shape[0] == 0:
            raise ValueError(f"No landmark with value {landmark_value} found in voxel data.")
        self.set_landmarks_from_positions(positions)
    
    def compute_surface_landmark_visibility(
        self,
        face_centers: np.ndarray,
        face_normals: np.ndarray,
        voxel_data: np.ndarray = None,
        landmark_value: int = -30,
        tree_k: float = 0.6,
        tree_lad: float = 1.0,
        boundary_epsilon: float = None
    ) -> np.ndarray:
        """
        Compute landmark visibility for building surface faces.
        
        Args:
            face_centers: Array of face center positions (n_faces, 3) in world coords
            face_normals: Array of face normal vectors (n_faces, 3)
            voxel_data: 3D voxel class array with landmarks marked
            landmark_value: Voxel value marking landmarks
            tree_k: Tree extinction coefficient
            tree_lad: Leaf area density
            boundary_epsilon: Epsilon for boundary detection
        
        Returns:
            1D array with 1.0 where any landmark is visible, 0.0 otherwise, nan for boundary
        """
        if self._landmark_positions is None or self._n_landmarks == 0:
            raise ValueError("No landmarks set. Call set_landmarks_* first.")
        
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
        visibility_values = ti.field(dtype=ti.f32, shape=(n_faces,))
        
        face_centers_ti.from_numpy(face_centers.astype(np.float32))
        face_normals_ti.from_numpy(face_normals.astype(np.float32))
        
        # Prepare masks
        if voxel_data is not None:
            is_tree = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            is_opaque = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
            self._setup_surface_masks(voxel_data, landmark_value, is_tree, is_opaque)
        else:
            is_tree = self.domain.is_tree
            is_opaque = ti.field(dtype=ti.i32, shape=(self.nx, self.ny, self.nz))
        
        # Tree attenuation
        tree_att = float(math.exp(-tree_k * tree_lad * self.meshsize))
        att_cutoff = 0.01
        
        # Run GPU computation
        self._compute_surface_landmark_kernel(
            face_centers_ti, face_normals_ti, visibility_values,
            is_tree, is_opaque, tree_att, att_cutoff,
            grid_bounds_real, boundary_epsilon
        )
        
        return visibility_values.to_numpy()
    
    @ti.kernel
    def _setup_surface_masks(
        self,
        voxel_data: ti.types.ndarray(),
        landmark_value: ti.i32,
        is_tree: ti.template(),
        is_opaque: ti.template()
    ):
        for i, j, k in is_tree:
            val = voxel_data[i, j, k]
            
            # Tree check
            tree = 0
            if val == -2:
                tree = 1
            is_tree[i, j, k] = tree
            
            # Opaque: non-zero, non-tree, non-landmark
            opaque = 0
            if val != 0 and val != -2 and val != landmark_value:
                opaque = 1
            is_opaque[i, j, k] = opaque
    
    @ti.kernel
    def _compute_surface_landmark_kernel(
        self,
        face_centers: ti.template(),
        face_normals: ti.template(),
        visibility_values: ti.template(),
        is_tree: ti.template(),
        is_opaque: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32,
        grid_bounds: ti.types.ndarray(),
        boundary_epsilon: ti.f32
    ):
        """Compute surface landmark visibility using GPU parallel processing."""
        for f in visibility_values:
            center = face_centers[f]
            normal = face_normals[f]
            
            # Check if face is on domain boundary
            is_vertical = ti.abs(normal[2]) < 0.01
            on_x_min = ti.abs(center[0] - grid_bounds[0, 0]) < boundary_epsilon
            on_y_min = ti.abs(center[1] - grid_bounds[0, 1]) < boundary_epsilon
            on_x_max = ti.abs(center[0] - grid_bounds[1, 0]) < boundary_epsilon
            on_y_max = ti.abs(center[1] - grid_bounds[1, 1]) < boundary_epsilon
            
            is_boundary = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
            
            if is_boundary:
                visibility_values[f] = ti.cast(float('nan'), ti.f32)
            else:
                # Normalize normal
                nrm = normal.norm()
                n = normal
                if nrm > 1e-12:
                    n = normal / nrm
                
                # Origin: face center offset by normal (in voxel coordinates)
                meshsize = self.dx
                ox = center[0] / meshsize + n[0] * 0.1
                oy = center[1] / meshsize + n[1] * 0.1
                oz = center[2] / meshsize + n[2] * 0.1
                
                visible = 0
                
                # Check visibility to each landmark
                for lm in range(self._n_landmarks):
                    if visible == 0:
                        target = self._landmark_positions[lm]
                        
                        # Direction to landmark
                        rx = target[0] - ox
                        ry = target[1] - oy
                        rz = target[2] - oz
                        rlen = ti.sqrt(rx*rx + ry*ry + rz*rz)
                        
                        if rlen > 0.0:
                            # Check if landmark is in front of face
                            rdx = rx / rlen
                            rdy = ry / rlen
                            rdz = rz / rlen
                            
                            dot = rdx*n[0] + rdy*n[1] + rdz*n[2]
                            if dot > 0.0:
                                # Trace ray to landmark
                                vis = self._trace_to_landmark(
                                    ox, oy, oz, target,
                                    is_tree, is_opaque,
                                    tree_att, att_cutoff
                                )
                                if vis == 1:
                                    visible = 1
                
                visibility_values[f] = ti.cast(visible, ti.f32)
    
    @ti.func
    def _trace_to_landmark(
        self,
        ox: ti.f32,
        oy: ti.f32,
        oz: ti.f32,
        target: ti.template(),
        is_tree: ti.template(),
        is_opaque: ti.template(),
        tree_att: ti.f32,
        att_cutoff: ti.f32
    ) -> ti.i32:
        """Trace ray from surface to landmark."""
        diff_x = target[0] - ox
        diff_y = target[1] - oy
        diff_z = target[2] - oz
        dist = ti.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        
        visible = 1
        
        if dist < 0.01:
            visible = 1
        else:
            ray_dir = ti.Vector([diff_x/dist, diff_y/dist, diff_z/dist])
            
            x = ox + 0.5
            y = oy + 0.5
            z = oz + 0.5
            
            i = ti.cast(ti.floor(ox), ti.i32)
            j = ti.cast(ti.floor(oy), ti.i32)
            k = ti.cast(ti.floor(oz), ti.i32)
            
            ti_x = ti.cast(ti.floor(target[0]), ti.i32)
            tj_y = ti.cast(ti.floor(target[1]), ti.i32)
            tk_z = ti.cast(ti.floor(target[2]), ti.i32)
            
            step_x = 1 if ray_dir[0] >= 0 else -1
            step_y = 1 if ray_dir[1] >= 0 else -1
            step_z = 1 if ray_dir[2] >= 0 else -1
            
            BIG = 1e30
            t_max_x, t_max_y, t_max_z = BIG, BIG, BIG
            t_delta_x, t_delta_y, t_delta_z = BIG, BIG, BIG
            
            if ray_dir[0] != 0.0:
                t_max_x = ((i + (1 if step_x > 0 else 0)) - x) / ray_dir[0]
                t_delta_x = ti.abs(1.0 / ray_dir[0])
            if ray_dir[1] != 0.0:
                t_max_y = ((j + (1 if step_y > 0 else 0)) - y) / ray_dir[1]
                t_delta_y = ti.abs(1.0 / ray_dir[1])
            if ray_dir[2] != 0.0:
                t_max_z = ((k + (1 if step_z > 0 else 0)) - z) / ray_dir[2]
                t_delta_z = ti.abs(1.0 / ray_dir[2])
            
            T = 1.0
            max_steps = self.nx + self.ny + self.nz
            done = 0
            
            for _ in range(max_steps):
                if done == 0:
                    if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
                        visible = 0
                        done = 1
                    elif is_opaque[i, j, k] == 1:
                        # Check if we're at the target
                        if not (i == ti_x and j == tj_y and k == tk_z):
                            visible = 0
                            done = 1
                    elif is_tree[i, j, k] == 1:
                        T *= tree_att
                        if T < att_cutoff:
                            visible = 0
                            done = 1
                    
                    if done == 0:
                        if i == ti_x and j == tj_y and k == tk_z:
                            done = 1
                        else:
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
        
        return visible


def compute_landmark_visibility_map(
    domain,
    voxel_data: np.ndarray,
    landmark_value: int = -30,
    view_height_voxel: int = 0,
    **kwargs
) -> np.ndarray:
    """
    Compute landmark visibility map.
    
    Args:
        domain: Domain object
        voxel_data: 3D voxel class array with landmarks marked
        landmark_value: Voxel value marking landmarks
        view_height_voxel: Observer height in voxels
        **kwargs: Additional parameters
    
    Returns:
        2D visibility map
    """
    calc = LandmarkVisibilityCalculator(domain)
    calc.set_landmarks_from_voxel_value(voxel_data, landmark_value)
    return calc.compute_visibility_map(
        voxel_data=voxel_data,
        view_height_voxel=view_height_voxel,
        **kwargs
    )
