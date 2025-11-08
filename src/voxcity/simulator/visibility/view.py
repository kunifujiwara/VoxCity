import numpy as np

from .geometry import (
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
)
from .raytracing import (
    compute_vi_map_generic,
    _prepare_masks_for_vi,
    _compute_vi_map_generic_fast,
)

from ...exporter.obj import grid_to_obj
import matplotlib.pyplot as plt


def get_view_index(voxcity, mode=None, hit_values=None, inclusion_mode=True, fast_path=True, **kwargs):
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize

    if mode == 'green':
        hit_values = (-2, 2, 5, 6, 7, 8)
        inclusion_mode = True
    elif mode == 'sky':
        hit_values = (0,)
        inclusion_mode = False
    else:
        if hit_values is None:
            raise ValueError("For custom mode, you must provide hit_values.")

    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)

    N_azimuth = kwargs.get("N_azimuth", 120)
    N_elevation = kwargs.get("N_elevation", 20)
    elevation_min_degrees = kwargs.get("elevation_min_degrees", -30)
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 30)
    ray_sampling = kwargs.get("ray_sampling", "grid")
    N_rays = kwargs.get("N_rays", N_azimuth * N_elevation)

    tree_k = kwargs.get("tree_k", 0.5)
    tree_lad = kwargs.get("tree_lad", 1.0)

    if str(ray_sampling).lower() == "fibonacci":
        ray_directions = _generate_ray_directions_fibonacci(int(N_rays), elevation_min_degrees, elevation_max_degrees)
    else:
        ray_directions = _generate_ray_directions_grid(int(N_azimuth), int(N_elevation), elevation_min_degrees, elevation_max_degrees)

    num_threads = kwargs.get("num_threads", None)
    if num_threads is not None:
        try:
            from numba import set_num_threads
            set_num_threads(int(num_threads))
        except Exception:
            pass

    if fast_path:
        try:
            is_tree, is_target, is_allowed, is_blocker_inc = _prepare_masks_for_vi(voxel_data, hit_values, inclusion_mode)
            trees_in_targets = bool(inclusion_mode and (-2 in hit_values))
            vi_map = _compute_vi_map_generic_fast(
                voxel_data, ray_directions, view_height_voxel,
                meshsize, tree_k, tree_lad,
                is_tree, is_target if is_target is not None else np.zeros(1, dtype=np.bool_),
                is_allowed if is_allowed is not None else np.zeros(1, dtype=np.bool_),
                is_blocker_inc if is_blocker_inc is not None else np.zeros(1, dtype=np.bool_),
                inclusion_mode, trees_in_targets
            )
        except Exception:
            vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
    else:
        vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, meshsize, tree_k, tree_lad, inclusion_mode)

    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='View Index')
    plt.axis('off')
    plt.show()

    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(vi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "view_index")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            vi_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )
    return vi_map


def get_sky_view_factor_map(voxcity, show_plot=False, **kwargs):
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'BuPu_r')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    N_azimuth = kwargs.get("N_azimuth", 120)
    N_elevation = kwargs.get("N_elevation", 20)
    elevation_min_degrees = kwargs.get("elevation_min_degrees", 0)
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 90)
    ray_sampling = kwargs.get("ray_sampling", "grid")
    N_rays = kwargs.get("N_rays", N_azimuth * N_elevation)
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    hit_values = (0,)
    inclusion_mode = False
    if str(ray_sampling).lower() == "fibonacci":
        ray_directions = _generate_ray_directions_fibonacci(int(N_rays), elevation_min_degrees, elevation_max_degrees)
    else:
        ray_directions = _generate_ray_directions_grid(int(N_azimuth), int(N_elevation), elevation_min_degrees, elevation_max_degrees)
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
    if show_plot:
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')
        plt.figure(figsize=(10, 8))
        plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Sky View Factor')
        plt.axis('off')
        plt.show()
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(vi_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "sky_view_factor")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            vi_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )
    return vi_map


# Surface view-factor (kept here for API; implementation uses local fast path if available)
import math
from .geometry import _build_face_basis, rotate_vector_axis_angle
from numba import njit, prange


def _prepare_masks_for_view(voxel_data, target_values, inclusion_mode):
    is_tree = (voxel_data == -2)
    target_mask = np.zeros(voxel_data.shape, dtype=np.bool_)
    for tv in target_values:
        target_mask |= (voxel_data == tv)
    if inclusion_mode:
        is_opaque = (voxel_data != 0) & (~is_tree) & (~target_mask)
        is_allowed = target_mask.copy()
    else:
        is_allowed = target_mask
        is_opaque = (~is_tree) & (~is_allowed)
    return is_tree, target_mask, is_allowed, is_opaque


@njit(cache=True, fastmath=True, nogil=True)
def _ray_visibility_contrib(origin, direction, vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque, att, att_cutoff, inclusion_mode, trees_are_targets):
    nx, ny, nz = vox_is_opaque.shape
    x0 = origin[0]; y0 = origin[1]; z0 = origin[2]
    dx = direction[0]; dy = direction[1]; dz = direction[2]
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    if L == 0.0:
        return 0.0
    invL = 1.0 / L
    dx *= invL; dy *= invL; dz *= invL
    x = x0 + 0.5
    y = y0 + 0.5
    z = z0 + 0.5
    i = int(x0); j = int(y0); k = int(z0)
    step_x = 1 if dx >= 0.0 else -1
    step_y = 1 if dy >= 0.0 else -1
    step_z = 1 if dz >= 0.0 else -1
    BIG = 1e30
    if dx != 0.0:
        t_max_x = (((i + (1 if step_x > 0 else 0)) - x) / dx)
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = BIG; t_delta_x = BIG
    if dy != 0.0:
        t_max_y = (((j + (1 if step_y > 0 else 0)) - y) / dy)
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = BIG; t_delta_y = BIG
    if dz != 0.0:
        t_max_z = (((k + (1 if step_z > 0 else 0)) - z) / dz)
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = BIG; t_delta_z = BIG
    T = 1.0
    while True:
        if (i < 0) or (i >= nx) or (j < 0) or (j >= ny) or (k < 0) or (k >= nz):
            if inclusion_mode:
                return 0.0
            else:
                return T
        if vox_is_opaque[i, j, k]:
            return 0.0
        if vox_is_tree[i, j, k]:
            T *= att
            if T < att_cutoff:
                return 0.0
            if inclusion_mode and trees_are_targets:
                return 1.0 - (T if T < 1.0 else 1.0)
        if inclusion_mode:
            if (not vox_is_tree[i, j, k]) and vox_is_target[i, j, k]:
                return 1.0
        else:
            if (not vox_is_tree[i, j, k]) and (not vox_is_allowed[i, j, k]):
                return 0.0
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max_x += t_delta_x; i += step_x
            else:
                t_max_z += t_delta_z; k += step_z
        else:
            if t_max_y < t_max_z:
                t_max_y += t_delta_y; j += step_y
            else:
                t_max_z += t_delta_z; k += step_z


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _compute_view_factor_faces_chunk(face_centers, face_normals, hemisphere_dirs, vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque, meshsize, att, att_cutoff, grid_bounds_real, boundary_epsilon, inclusion_mode, trees_are_targets):
    n_faces = face_centers.shape[0]
    out = np.empty(n_faces, dtype=np.float64)
    for f in prange(n_faces):
        center = face_centers[f]
        normal = face_normals[f]
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
        on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
        on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
        on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            out[f] = np.nan
            continue
        u, v, n = _build_face_basis(normal)
        ox = center[0] / meshsize + n[0] * 0.51
        oy = center[1] / meshsize + n[1] * 0.51
        oz = center[2] / meshsize + n[2] * 0.51
        origin = np.array((ox, oy, oz))
        vis_sum = 0.0
        valid = 0
        for i in range(hemisphere_dirs.shape[0]):
            lx = hemisphere_dirs[i,0]; ly = hemisphere_dirs[i,1]; lz = hemisphere_dirs[i,2]
            dx = u[0]*lx + v[0]*ly + n[0]*lz
            dy = u[1]*lx + v[1]*ly + n[1]*lz
            dz = u[2]*lx + v[2]*ly + n[2]*lz
            if (dx*n[0] + dy*n[1] + dz*n[2]) <= 0.0:
                continue
            contrib = _ray_visibility_contrib(origin, np.array((dx, dy, dz)), vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque, att, att_cutoff, inclusion_mode, trees_are_targets)
            vis_sum += contrib
            valid += 1
        out[f] = 0.0 if valid == 0 else (vis_sum / valid)
    return out


def _compute_view_factor_faces_progress(face_centers, face_normals, hemisphere_dirs, vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque, meshsize, att, att_cutoff, grid_bounds_real, boundary_epsilon, inclusion_mode, trees_are_targets, progress_report=False, chunks=10):
    n_faces = face_centers.shape[0]
    results = np.empty(n_faces, dtype=np.float64)
    step = math.ceil(n_faces / chunks) if n_faces > 0 else 1
    for start in range(0, n_faces, step):
        end = min(start + step, n_faces)
        results[start:end] = _compute_view_factor_faces_chunk(
            face_centers[start:end], face_normals[start:end], hemisphere_dirs,
            vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
            float(meshsize), float(att), float(att_cutoff),
            grid_bounds_real, float(boundary_epsilon),
            inclusion_mode, trees_are_targets
        )
        if progress_report:
            pct = (end / n_faces) * 100 if n_faces > 0 else 100.0
            print(f"  Processed {end}/{n_faces} faces ({pct:.1f}%)")
    return results


def compute_view_factor_for_all_faces(face_centers, face_normals, hemisphere_dirs, voxel_data, meshsize, tree_k, tree_lad, target_values, inclusion_mode, grid_bounds_real, boundary_epsilon, offset_vox=0.51):
    n_faces = face_centers.shape[0]
    face_vf_values = np.zeros(n_faces, dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0])
    for fidx in range(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
        on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
        on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
        on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
        if is_boundary_vertical:
            face_vf_values[fidx] = np.nan
            continue
        norm_n = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm_n < 1e-12:
            face_vf_values[fidx] = 0.0
            continue
        dot_zn = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2]
        cos_angle = dot_zn / (norm_n)
        if cos_angle >  1.0: cos_angle =  1.0
        if cos_angle < -1.0: cos_angle = -1.0
        angle = np.arccos(cos_angle)
        if abs(cos_angle - 1.0) < 1e-9:
            local_dirs = hemisphere_dirs
        elif abs(cos_angle + 1.0) < 1e-9:
            axis_180 = np.array([1.0, 0.0, 0.0])
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(hemisphere_dirs[i], axis_180, np.pi)
        else:
            axis_x = z_axis[1]*normal[2] - z_axis[2]*normal[1]
            axis_y = z_axis[2]*normal[0] - z_axis[0]*normal[2]
            axis_z = z_axis[0]*normal[1] - z_axis[1]*normal[0]
            rot_axis = np.array([axis_x, axis_y, axis_z], dtype=np.float64)
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(hemisphere_dirs[i], rot_axis, angle)
        total_outward = 0
        num_valid = 0
        for i in range(local_dirs.shape[0]):
            dvec = local_dirs[i]
            dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
            if dp > 0.0:
                total_outward += 1
                num_valid += 1
        if total_outward == 0:
            face_vf_values[fidx] = 0.0
            continue
        if num_valid == 0:
            face_vf_values[fidx] = 0.0
            continue
        valid_dirs_arr = np.empty((num_valid, 3), dtype=np.float64)
        out_idx = 0
        for i in range(local_dirs.shape[0]):
            dvec = local_dirs[i]
            dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
            if dp > 0.0:
                valid_dirs_arr[out_idx, 0] = dvec[0]
                valid_dirs_arr[out_idx, 1] = dvec[1]
                valid_dirs_arr[out_idx, 2] = dvec[2]
                out_idx += 1
        ray_origin = (center / meshsize) + (normal / norm_n) * offset_vox
        from .raytracing import compute_vi_generic  # local import for numba friendliness
        vf = compute_vi_generic(
            ray_origin,
            voxel_data,
            valid_dirs_arr,
            target_values,
            meshsize,
            tree_k,
            tree_lad,
            inclusion_mode
        )
        fraction_valid = num_valid / total_outward
        face_vf_values[fidx] = vf * fraction_valid
    return face_vf_values


def get_surface_view_factor(voxcity, **kwargs):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os
    from ...geoprocessor.mesh import create_voxel_mesh
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    value_name = kwargs.get("value_name", 'view_factor_values')
    colormap = kwargs.get("colormap", 'BuPu_r')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    N_azimuth = kwargs.get("N_azimuth", 120)
    N_elevation = kwargs.get("N_elevation", 20)
    ray_sampling = kwargs.get("ray_sampling", "grid")
    N_rays = kwargs.get("N_rays", N_azimuth * N_elevation)
    debug = kwargs.get("debug", False)
    progress_report = kwargs.get("progress_report", False)
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    target_values = kwargs.get("target_values", (0,))
    inclusion_mode = kwargs.get("inclusion_mode", False)
    building_class_id = kwargs.get("building_class_id", -3)
    try:
        building_mesh = create_voxel_mesh(
            voxel_data,
            building_class_id,
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No surfaces found in voxel data for the specified class.")
            return None
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None
    if progress_report:
        print(f"Processing view factor for {len(building_mesh.faces)} faces...")
    face_centers = building_mesh.triangles_center
    face_normals = building_mesh.face_normals
    if str(ray_sampling).lower() == "fibonacci":
        hemisphere_dirs = _generate_ray_directions_fibonacci(int(N_rays), 0.0, 90.0)
    else:
        hemisphere_dirs = _generate_ray_directions_grid(int(N_azimuth), int(N_elevation), 0.0, 90.0)
    nx, ny, nz = voxel_data.shape
    grid_bounds_voxel = np.array([[0,0,0],[nx, ny, nz]], dtype=np.float64)
    grid_bounds_real = grid_bounds_voxel * meshsize
    boundary_epsilon = meshsize * 0.05
    fast_path = kwargs.get("fast_path", True)
    face_vf_values = None
    if fast_path:
        try:
            vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque = _prepare_masks_for_view(voxel_data, target_values, inclusion_mode)
            att = float(np.exp(-tree_k * tree_lad * meshsize))
            att_cutoff = 0.01
            trees_are_targets = bool((-2 in target_values) and inclusion_mode)
            face_vf_values = _compute_view_factor_faces_progress(
                face_centers.astype(np.float64),
                face_normals.astype(np.float64),
                hemisphere_dirs.astype(np.float64),
                vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
                float(meshsize), float(att), float(att_cutoff),
                grid_bounds_real.astype(np.float64), float(boundary_epsilon),
                inclusion_mode, trees_are_targets,
                progress_report=progress_report
            )
        except Exception as e:
            if debug:
                print(f"Fast view-factor path failed: {e}. Falling back to standard path.")
            face_vf_values = None
    if face_vf_values is None:
        face_vf_values = compute_view_factor_for_all_faces(
            face_centers,
            face_normals,
            hemisphere_dirs,
            voxel_data,
            meshsize,
            tree_k,
            tree_lad,
            target_values,
            inclusion_mode,
            grid_bounds_real,
            boundary_epsilon
        )
    if not hasattr(building_mesh, 'metadata'):
        building_mesh.metadata = {}
    building_mesh.metadata[value_name] = face_vf_values
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "surface_view_factor")
        import os
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    return building_mesh
"""Visibility API aggregator.

This module re-exports selected public APIs:
- raytracing: low-level VI computation helpers
- landmark: landmark visibility utilities
"""

from .raytracing import (
    compute_vi_generic,
    compute_vi_map_generic,
)

# get_view_index, get_sky_view_factor_map, get_surface_view_factor, and
# compute_view_factor_for_all_faces are defined in this module above.

__all__ = [
    'get_view_index',
    'get_surface_view_factor',
    'get_sky_view_factor_map',
    'compute_view_factor_for_all_faces',
    'compute_vi_generic',
    'compute_vi_map_generic',
]


