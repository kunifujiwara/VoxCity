"""
Stage 2: Physics - convert geometry to irradiance.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

from ...models import VoxCity
from ...exporter.obj import grid_to_obj
from ..visibility import get_sky_view_factor_map
from ..common.raytracing import trace_ray_generic
from .kernels import compute_direct_solar_irradiance_map_binary


def get_direct_solar_irradiance_map(
    voxcity: VoxCity,
    azimuth_degrees_ori,
    elevation_degrees,
    direct_normal_irradiance,
    show_plot=False,
    **kwargs,
):
    """
    Compute horizontal direct irradiance map (W/m²) with tree transmittance.
    """
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize

    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", "magma")
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", direct_normal_irradiance)
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
    dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
    dz = np.sin(elevation_radians)
    sun_direction = (dx, dy, dz)

    hit_values = (0,)
    inclusion_mode = False
    transmittance_map = compute_direct_solar_irradiance_map_binary(
        voxel_data,
        sun_direction,
        view_point_height,
        hit_values,
        meshsize,
        tree_k,
        tree_lad,
        inclusion_mode,
    )

    sin_elev = dz
    direct_map = transmittance_map * direct_normal_irradiance * sin_elev

    if show_plot:
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color="lightgray")
        plt.figure(figsize=(10, 8))
        plt.imshow(direct_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Direct Solar Irradiance (W/m²)")
        plt.axis("off")
        plt.show()

    if kwargs.get("obj_export", False):
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(direct_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "direct_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            direct_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )

    return direct_map


def get_diffuse_solar_irradiance_map(
    voxcity: VoxCity,
    diffuse_irradiance=1.0,
    show_plot=False,
    **kwargs,
):
    """
    Compute diffuse horizontal irradiance map (W/m²) using SVF.
    """
    meshsize = voxcity.voxels.meta.meshsize
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", "magma")
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", diffuse_irradiance)

    svf_kwargs = kwargs.copy()
    svf_kwargs["colormap"] = "BuPu_r"
    svf_kwargs["vmin"] = 0
    svf_kwargs["vmax"] = 1

    SVF_map = get_sky_view_factor_map(voxcity, **svf_kwargs)
    diffuse_map = SVF_map * diffuse_irradiance

    if show_plot:
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color="lightgray")
        plt.figure(figsize=(10, 8))
        plt.imshow(diffuse_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Diffuse Solar Irradiance (W/m²)")
        plt.axis("off")
        plt.show()

    if kwargs.get("obj_export", False):
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(diffuse_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "diffuse_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            diffuse_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )

    return diffuse_map


def get_global_solar_irradiance_map(
    voxcity: VoxCity,
    azimuth_degrees_ori,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    show_plot=False,
    **kwargs,
):
    """
    Combine direct and diffuse horizontal irradiance (W/m²).
    """
    direct_map = get_direct_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori,
        elevation_degrees,
        direct_normal_irradiance,
        show_plot=False,
        **kwargs,
    )
    diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=diffuse_irradiance,
        show_plot=False,
        **kwargs,
    )
    global_map = np.where(np.isnan(direct_map), diffuse_map, direct_map + diffuse_map)

    if show_plot:
        colormap = kwargs.get("colormap", "magma")
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", max(float(np.nanmax(global_map)), 1.0))
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color="lightgray")
        plt.figure(figsize=(10, 8))
        plt.imshow(global_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Global Solar Irradiance (W/m²)")
        plt.axis("off")
        plt.show()

    if kwargs.get("obj_export", False):
        meshsize = voxcity.voxels.meta.meshsize
        view_point_height = kwargs.get("view_point_height", 1.5)
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(global_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "global_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        grid_to_obj(
            global_map,
            dem_grid,
            output_dir,
            output_file_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=kwargs.get("vmin", 0.0),
            vmax=kwargs.get("vmax", vmax if "vmax" in kwargs else None),
        )

    return global_map


# --------------------------
# Building-surface irradiance
# --------------------------

@njit(parallel=True)
def compute_solar_irradiance_for_all_faces(
    face_centers,
    face_normals,
    face_svf,
    sun_direction,
    direct_normal_irradiance,
    diffuse_irradiance,
    voxel_data,
    meshsize,
    tree_k,
    tree_lad,
    hit_values,
    inclusion_mode,
    grid_bounds_real,
    boundary_epsilon
):
    """
    Numba kernel: compute per-face direct/diffuse/global (W/m²) using generic ray tracer.
    """
    n_faces = face_centers.shape[0]
    face_direct = np.zeros(n_faces, dtype=np.float64)
    face_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_global = np.zeros(n_faces, dtype=np.float64)

    x_min, y_min, z_min = grid_bounds_real[0, 0], grid_bounds_real[0, 1], grid_bounds_real[0, 2]
    x_max, y_max, z_max = grid_bounds_real[1, 0], grid_bounds_real[1, 1], grid_bounds_real[1, 2]

    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf    = face_svf[fidx]

        # Exclude vertical boundary faces
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        if svf != svf:
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        # Direct term
        cos_incidence = normal[0]*sun_direction[0] + normal[1]*sun_direction[1] + normal[2]*sun_direction[2]
        direct_val = 0.0
        if cos_incidence > 0.0 and direct_normal_irradiance > 0.0:
            offset_vox = 0.1
            ox = center[0]/meshsize + normal[0]*offset_vox
            oy = center[1]/meshsize + normal[1]*offset_vox
            oz = center[2]/meshsize + normal[2]*offset_vox
            hit_detected, transmittance = trace_ray_generic(
                voxel_data,
                np.array([ox, oy, oz], dtype=np.float64),
                sun_direction,
                hit_values,
                meshsize,
                tree_k,
                tree_lad,
                inclusion_mode
            )
            if not hit_detected:
                direct_val = direct_normal_irradiance * cos_incidence * transmittance

        # Diffuse via SVF
        diffuse_val = svf * diffuse_irradiance
        if diffuse_val > diffuse_irradiance:
            diffuse_val = diffuse_irradiance

        face_direct[fidx]  = direct_val
        face_diffuse[fidx] = diffuse_val
        face_global[fidx]  = direct_val + diffuse_val

    return face_direct, face_diffuse, face_global


@njit(cache=True, fastmath=True, nogil=True)
def _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, att, att_cutoff=0.01):
    nx, ny, nz = vox_is_opaque.shape
    x0 = origin[0]; y0 = origin[1]; z0 = origin[2]
    dx = direction[0]; dy = direction[1]; dz = direction[2]

    # Normalize
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    if L == 0.0:
        return False, 1.0
    invL = 1.0 / L
    dx *= invL; dy *= invL; dz *= invL

    # Start at voxel centers
    x = x0 + 0.5; y = y0 + 0.5; z = z0 + 0.5
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
            return False, T

        if vox_is_opaque[i, j, k]:
            return True, T

        if vox_is_tree[i, j, k]:
            T *= att
            if T < att_cutoff:
                return True, T

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
def compute_solar_irradiance_for_all_faces_masked(
    face_centers,
    face_normals,
    face_svf,
    sun_direction,
    direct_normal_irradiance,
    diffuse_irradiance,
    vox_is_tree,
    vox_is_opaque,
    meshsize,
    att,
    x_min, y_min, z_min,
    x_max, y_max, z_max,
    boundary_epsilon
):
    n_faces = face_centers.shape[0]
    face_direct = np.zeros(n_faces, dtype=np.float64)
    face_diffuse = np.zeros(n_faces, dtype=np.float64)
    face_global = np.zeros(n_faces, dtype=np.float64)

    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf = face_svf[fidx]

        # Boundary vertical exclusion
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        if svf != svf:
            face_direct[fidx] = np.nan
            face_diffuse[fidx] = np.nan
            face_global[fidx] = np.nan
            continue

        # Direct component
        cos_incidence = normal[0]*sun_direction[0] + normal[1]*sun_direction[1] + normal[2]*sun_direction[2]
        direct_val = 0.0
        if cos_incidence > 0.0 and direct_normal_irradiance > 0.0:
            offset_vox = 0.1
            ox = center[0]/meshsize + normal[0]*offset_vox
            oy = center[1]/meshsize + normal[1]*offset_vox
            oz = center[2]/meshsize + normal[2]*offset_vox
            blocked, T = _trace_direct_masked(
                vox_is_tree,
                vox_is_opaque,
                np.array((ox, oy, oz), dtype=np.float64),
                sun_direction,
                att
            )
            if not blocked:
                direct_val = direct_normal_irradiance * cos_incidence * T

        # Diffuse component
        diffuse_val = svf * diffuse_irradiance
        if diffuse_val > diffuse_irradiance:
            diffuse_val = diffuse_irradiance

        face_direct[fidx] = direct_val
        face_diffuse[fidx] = diffuse_val
        face_global[fidx] = direct_val + diffuse_val

    return face_direct, face_diffuse, face_global


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def compute_cumulative_solar_irradiance_faces_masked_timeseries(
    face_centers,
    face_normals,
    face_svf,
    sun_dirs_arr,      # shape (T, 3)
    DNI_arr,           # shape (T,)
    DHI_arr,           # shape (T,)
    vox_is_tree,
    vox_is_opaque,
    meshsize,
    att,
    x_min, y_min, z_min,
    x_max, y_max, z_max,
    boundary_epsilon,
    t_start, t_end,              # [start, end) indices
    time_step_hours
):
    n_faces = face_centers.shape[0]
    out_dir  = np.zeros(n_faces, dtype=np.float64)
    out_diff = np.zeros(n_faces, dtype=np.float64)
    out_glob = np.zeros(n_faces, dtype=np.float64)

    for fidx in prange(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        svf = face_svf[fidx]

        # Boundary vertical exclusion
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - x_min) < boundary_epsilon)
        on_y_min = (abs(center[1] - y_min) < boundary_epsilon)
        on_x_max = (abs(center[0] - x_max) < boundary_epsilon)
        on_y_max = (abs(center[1] - y_max) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            out_dir[fidx]  = np.nan
            out_diff[fidx] = np.nan
            out_glob[fidx] = np.nan
            continue

        if svf != svf:
            out_dir[fidx]  = np.nan
            out_diff[fidx] = np.nan
            out_glob[fidx] = np.nan
            continue

        accum_dir = 0.0
        accum_diff = 0.0
        accum_glob = 0.0

        # Precompute ray origin (voxel coords) once per face
        offset_vox = 0.1
        ox = center[0]/meshsize + normal[0]*offset_vox
        oy = center[1]/meshsize + normal[1]*offset_vox
        oz = center[2]/meshsize + normal[2]*offset_vox
        origin = np.array((ox, oy, oz), dtype=np.float64)

        for t in range(t_start, t_end):
            dni = DNI_arr[t]
            dhi = DHI_arr[t]
            sd0 = sun_dirs_arr[t, 0]
            sd1 = sun_dirs_arr[t, 1]
            sd2 = sun_dirs_arr[t, 2]
            # Below horizon -> diffuse only
            if sd2 <= 0.0:
                diff_val = svf * dhi
                if diff_val > dhi:
                    diff_val = dhi
                accum_diff += diff_val * time_step_hours
                accum_glob += diff_val * time_step_hours
                continue

            # Direct
            cos_inc = normal[0]*sd0 + normal[1]*sd1 + normal[2]*sd2
            direct_val = 0.0
            if (dni > 0.0) and (cos_inc > 0.0):
                blocked, T = _trace_direct_masked(
                    vox_is_tree,
                    vox_is_opaque,
                    origin,
                    np.array((sd0, sd1, sd2), dtype=np.float64),
                    att
                )
                if not blocked:
                    direct_val = dni * cos_inc * T

            diff_val = svf * dhi
            if diff_val > dhi:
                diff_val = dhi

            accum_dir  += direct_val * time_step_hours
            accum_diff += diff_val   * time_step_hours
            accum_glob += (direct_val + diff_val) * time_step_hours

        out_dir[fidx]  = accum_dir
        out_diff[fidx] = accum_diff
        out_glob[fidx] = accum_glob

    return out_dir, out_diff, out_glob


def get_building_solar_irradiance(
    voxcity: VoxCity,
    building_svf_mesh,
    azimuth_degrees,
    elevation_degrees,
    direct_normal_irradiance,
    diffuse_irradiance,
    **kwargs
):
    """
    Compute per-face direct/diffuse/global (W/m²) on a building mesh with SVF.
    """
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    progress_report = kwargs.get("progress_report", False)
    fast_path = kwargs.get("fast_path", True)

    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize

    # Sun vector
    az_rad = np.deg2rad(180 - azimuth_degrees)
    el_rad = np.deg2rad(elevation_degrees)
    sun_dx = np.cos(el_rad) * np.cos(az_rad)
    sun_dy = np.cos(el_rad) * np.sin(az_rad)
    sun_dz = np.sin(el_rad)
    sun_direction = np.array([sun_dx, sun_dy, sun_dz], dtype=np.float64)

    # SVF
    if hasattr(building_svf_mesh, 'metadata') and ('svf' in building_svf_mesh.metadata):
        face_svf = building_svf_mesh.metadata['svf']
    else:
        face_svf = np.zeros(len(building_svf_mesh.faces), dtype=np.float64)

    # Geometry caches
    precomputed_geometry = kwargs.get("precomputed_geometry", None)
    if precomputed_geometry is not None:
        face_centers = precomputed_geometry.get("face_centers", building_svf_mesh.triangles_center)
        face_normals = precomputed_geometry.get("face_normals", building_svf_mesh.face_normals)
        grid_bounds_real = precomputed_geometry.get("grid_bounds_real", None)
        boundary_epsilon = precomputed_geometry.get("boundary_epsilon", None)
    else:
        face_centers = building_svf_mesh.triangles_center
        face_normals = building_svf_mesh.face_normals
        grid_bounds_real = None
        boundary_epsilon = None

    if grid_bounds_real is None or boundary_epsilon is None:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = np.array([[0, 0, 0], [grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05

    if fast_path:
        precomputed_masks = kwargs.get("precomputed_masks", None)
        if precomputed_masks is not None:
            vox_is_tree = precomputed_masks.get("vox_is_tree", (voxel_data == -2))
            vox_is_opaque = precomputed_masks.get("vox_is_opaque", (voxel_data != 0) & (voxel_data != -2))
            att = float(precomputed_masks.get("att", np.exp(-tree_k * tree_lad * meshsize)))
        else:
            vox_is_tree = (voxel_data == -2)
            vox_is_opaque = (voxel_data != 0) & (~vox_is_tree)
            att = float(np.exp(-tree_k * tree_lad * meshsize))

        face_direct, face_diffuse, face_global = compute_solar_irradiance_for_all_faces_masked(
            face_centers.astype(np.float64),
            face_normals.astype(np.float64),
            face_svf.astype(np.float64),
            sun_direction.astype(np.float64),
            float(direct_normal_irradiance),
            float(diffuse_irradiance),
            vox_is_tree,
            vox_is_opaque,
            float(meshsize),
            att,
            float(grid_bounds_real[0,0]), float(grid_bounds_real[0,1]), float(grid_bounds_real[0,2]),
            float(grid_bounds_real[1,0]), float(grid_bounds_real[1,1]), float(grid_bounds_real[1,2]),
            float(boundary_epsilon)
        )
    else:
        hit_values = (0,)
        inclusion_mode = False
        face_direct, face_diffuse, face_global = compute_solar_irradiance_for_all_faces(
            face_centers.astype(np.float64),
            face_normals.astype(np.float64),
            face_svf.astype(np.float64),
            sun_direction.astype(np.float64),
            float(direct_normal_irradiance),
            float(diffuse_irradiance),
            voxel_data,
            float(meshsize),
            float(tree_k),
            float(tree_lad),
            hit_values,
            inclusion_mode,
            grid_bounds_real.astype(np.float64),
            float(boundary_epsilon)
        )

    irradiance_mesh = building_svf_mesh.copy()
    if not hasattr(irradiance_mesh, 'metadata'):
        irradiance_mesh.metadata = {}
    irradiance_mesh.metadata['svf'] = face_svf
    irradiance_mesh.metadata['direct'] = face_direct
    irradiance_mesh.metadata['diffuse'] = face_diffuse
    irradiance_mesh.metadata['global'] = face_global
    irradiance_mesh.name = "Solar Irradiance (W/m²)"
    return irradiance_mesh


