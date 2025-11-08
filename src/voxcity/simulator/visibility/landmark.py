import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange

from ...geoprocessor.selection import find_building_containing_point, get_buildings_in_drawn_polygon
from ...geoprocessor.mesh import create_voxel_mesh
from ...exporter.obj import grid_to_obj, export_obj


def mark_building_by_id(voxcity_grid_ori, building_id_grid_ori, ids, mark):
    voxcity_grid = voxcity_grid_ori.copy()
    building_id_grid = np.flipud(building_id_grid_ori.copy())
    positions = np.where(np.isin(building_id_grid, ids))
    for i in range(len(positions[0])):
        x, y = positions[0][i], positions[1][i]
        z_mask = voxcity_grid[x, y, :] == -3
        voxcity_grid[x, y, z_mask] = mark
    return voxcity_grid


@njit
def trace_ray_to_target(voxel_data, origin, target, opaque_values):
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    x1, y1, z1 = target
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return True
    dx /= length
    dy /= length
    dz /= length
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1
    if dx != 0:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1 / dx)
    else:
        t_max_x = np.inf
        t_delta_x = np.inf
    if dy != 0:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1 / dy)
    else:
        t_max_y = np.inf
        t_delta_y = np.inf
    if dz != 0:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1 / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf
    while True:
        if (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
            voxel_value = voxel_data[i, j, k]
            if voxel_value in opaque_values:
                return False
        else:
            return False
        if i == int(x1) and j == int(y1) and k == int(z1):
            return True
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max = t_max_x
                t_max_x += t_delta_x
                i += step_x
            else:
                t_max = t_max_z
                t_max_z += t_delta_z
                k += step_z
        else:
            if t_max_y < t_max_z:
                t_max = t_max_y
                t_max_y += t_delta_y
                j += step_y
            else:
                t_max = t_max_z
                t_max_z += t_delta_z
                k += step_z


@njit
def compute_visibility_to_all_landmarks(observer_location, landmark_positions, voxel_data, opaque_values):
    for idx in range(landmark_positions.shape[0]):
        target = landmark_positions[idx].astype(np.float64)
        is_visible = trace_ray_to_target(voxel_data, observer_location, target, opaque_values)
        if is_visible:
            return 1
    return 0


@njit(parallel=True)
def compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel):
    nx, ny, nz = voxel_data.shape
    visibility_map = np.full((nx, ny), np.nan)
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] == 0 and voxel_data[x, y, z - 1] != 0:
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        visibility_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        observer_location = np.array([x, y, z+view_height_voxel], dtype=np.float64)
                        visible = compute_visibility_to_all_landmarks(observer_location, landmark_positions, voxel_data, opaque_values)
                        visibility_map[x, y] = visible
                        found_observer = True
                        break
            if not found_observer:
                visibility_map[x, y] = np.nan
    return visibility_map


def compute_landmark_visibility(voxel_data, target_value=-30, view_height_voxel=0, colormap='viridis'):
    landmark_positions = np.argwhere(voxel_data == target_value)
    if landmark_positions.shape[0] == 0:
        raise ValueError(f"No landmark with value {target_value} found in the voxel data.")
    unique_values = np.unique(voxel_data)
    opaque_values = np.array([v for v in unique_values if v != 0 and v != target_value], dtype=np.int32)
    visibility_map = compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel)
    cmap = plt.cm.get_cmap(colormap, 2).copy()
    cmap.set_bad(color='lightgray')
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(visibility_map), origin='lower', cmap=cmap, vmin=0, vmax=1)
    visible_patch = mpatches.Patch(color=cmap(1.0), label='Visible (1)')
    not_visible_patch = mpatches.Patch(color=cmap(0.0), label='Not Visible (0)')
    plt.legend(handles=[visible_patch, not_visible_patch], 
            loc='center left',
            bbox_to_anchor=(1.0, 0.5))
    plt.axis('off')
    plt.show()
    return np.flipud(visibility_map)


def get_landmark_visibility_map(voxcity, building_gdf=None, **kwargs):
    if building_gdf is None:
        building_gdf = voxcity.extras.get('building_gdf', None)
        if building_gdf is None:
            raise ValueError("building_gdf not provided and not found in voxcity.extras['building_gdf']")
    voxcity_grid_ori = voxcity.voxels.classes
    building_id_grid = voxcity.buildings.ids
    meshsize = voxcity.voxels.meta.meshsize
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_gdf, landmark_polygon, operation='within')
        else:
            rectangle_vertices = kwargs.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_ids.")
                return None
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            target_point = (center_lon, center_lat)
            landmark_ids = find_building_containing_point(building_gdf, target_point)
    target_value = -30
    voxcity_grid = mark_building_by_id(voxcity_grid_ori, building_id_grid, landmark_ids, target_value)
    landmark_vis_map = compute_landmark_visibility(voxcity_grid, target_value=target_value, view_height_voxel=view_height_voxel, colormap=colormap)
    obj_export = kwargs.get("obj_export")
    if obj_export == True:
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(landmark_vis_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "landmark_visibility")        
        num_colors = 2
        alpha = kwargs.get("alpha", 1.0)
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", 1.0)
        grid_to_obj(
            landmark_vis_map,
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
        output_file_name_vox = 'voxcity_' + output_file_name
        export_obj(voxcity_grid, output_dir, output_file_name_vox, meshsize)
    return landmark_vis_map, voxcity_grid


# Surface landmark visibility (fast, chunked)
import math
from ..common.raytracing import _trace_ray


def _prepare_voxel_classes(voxel_data, landmark_value=-30):
    is_tree = (voxel_data == -2)
    is_opaque = (voxel_data != 0) & (voxel_data != landmark_value) & (~is_tree)
    return is_tree, is_opaque


def _compute_all_faces_progress(face_centers, face_normals, landmark_positions_vox, vox_is_tree, vox_is_opaque, meshsize, att, att_cutoff, grid_bounds_real, boundary_epsilon, progress_report=False, chunks=10):
    n_faces = face_centers.shape[0]
    results = np.empty(n_faces, dtype=np.float64)
    step = math.ceil(n_faces / chunks)
    for start in range(0, n_faces, step):
        end = min(start + step, n_faces)
        results[start:end] = _compute_faces_chunk(
            face_centers[start:end],
            face_normals[start:end],
            landmark_positions_vox,
            vox_is_tree, vox_is_opaque,
            meshsize, att, att_cutoff,
            grid_bounds_real, boundary_epsilon
        )
        if progress_report:
            pct = (end / n_faces) * 100
            print(f"  Processed {end}/{n_faces} faces ({pct:.1f}%)")
    return results


@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def _compute_faces_chunk(face_centers, face_normals, landmark_positions_vox, vox_is_tree, vox_is_opaque, meshsize, att, att_cutoff, grid_bounds_real, boundary_epsilon):
    n_faces = face_centers.shape[0]
    out = np.empty(n_faces, dtype=np.float64)
    for f in prange(n_faces):
        out[f] = _compute_face_visibility(
            face_centers[f], face_normals[f],
            landmark_positions_vox,
            vox_is_tree, vox_is_opaque,
            meshsize, att, att_cutoff,
            grid_bounds_real, boundary_epsilon
        )
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _compute_face_visibility(face_center, face_normal, landmark_positions_vox, vox_is_tree, vox_is_opaque, meshsize, att, att_cutoff, grid_bounds_real, boundary_epsilon):
    is_vertical = (abs(face_normal[2]) < 0.01)
    on_x_min = (abs(face_center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
    on_y_min = (abs(face_center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
    on_x_max = (abs(face_center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
    on_y_max = (abs(face_center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
    if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
        return np.nan
    nx = face_normal[0]; ny = face_normal[1]; nz = face_normal[2]
    nrm = (nx*nx + ny*ny + nz*nz) ** 0.5
    if nrm < 1e-12:
        return 0.0
    invn = 1.0 / nrm
    nx *= invn; ny *= invn; nz *= invn
    offset_vox = 0.1
    ox = face_center[0] / meshsize + nx * offset_vox
    oy = face_center[1] / meshsize + ny * offset_vox
    oz = face_center[2] / meshsize + nz * offset_vox
    for idx in range(landmark_positions_vox.shape[0]):
        tx = landmark_positions_vox[idx, 0]
        ty = landmark_positions_vox[idx, 1]
        tz = landmark_positions_vox[idx, 2]
        rx = tx - ox; ry = ty - oy; rz = tz - oz
        rlen2 = rx*rx + ry*ry + rz*rz
        if rlen2 == 0.0:
            return 1.0
        invr = 1.0 / (rlen2 ** 0.5)
        rdx = rx * invr; rdy = ry * invr; rdz = rz * invr
        if (rdx*nx + rdy*ny + rdz*nz) <= 0.0:
            continue
        if _trace_ray(vox_is_tree, vox_is_opaque, np.array((ox, oy, oz)), np.array((tx, ty, tz)), att, att_cutoff):
            return 1.0
    return 0.0


def get_surface_landmark_visibility(voxcity, building_gdf=None, **kwargs):
    import os
    if building_gdf is None:
        building_gdf = voxcity.extras.get('building_gdf', None)
        if building_gdf is None:
            raise ValueError("building_gdf not provided and not found in voxcity.extras['building_gdf']")
    voxel_data = voxcity.voxels.classes
    building_id_grid = voxcity.buildings.ids
    meshsize = voxcity.voxels.meta.meshsize
    progress_report = kwargs.get("progress_report", False)
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_gdf, landmark_polygon, operation='within')
        else:
            rectangle_vertices = kwargs.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_ids.")
                return None, None
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            target_point = (center_lon, center_lat)
            landmark_ids = find_building_containing_point(building_gdf, target_point)
    building_class_id = kwargs.get("building_class_id", -3)
    landmark_value = -30
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    colormap = kwargs.get("colormap", 'RdYlGn')
    voxel_data_for_mesh = voxel_data.copy()
    voxel_data_modified = voxel_data.copy()
    voxel_data_modified = mark_building_by_id(voxel_data_modified, building_id_grid, landmark_ids, landmark_value)
    voxel_data_for_mesh = mark_building_by_id(voxel_data_for_mesh, building_id_grid, landmark_ids, 0)
    landmark_positions = np.argwhere(voxel_data_modified == landmark_value).astype(np.float64)
    if landmark_positions.shape[0] == 0:
        print(f"No landmarks found after marking buildings with IDs: {landmark_ids}")
        return None, None
    if progress_report:
        print(f"Found {landmark_positions.shape[0]} landmark voxels")
        print(f"Landmark building IDs: {landmark_ids}")
    try:
        building_mesh = create_voxel_mesh(
            voxel_data_for_mesh,
            building_class_id,
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No non-landmark building surfaces found in voxel data.")
            return None, None
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None, None
    if progress_report:
        print(f"Processing landmark visibility for {len(building_mesh.faces)} faces...")
    face_centers = building_mesh.triangles_center.astype(np.float64)
    face_normals = building_mesh.face_normals.astype(np.float64)
    nx, ny, nz = voxel_data_modified.shape
    grid_bounds_voxel = np.array([[0,0,0],[nx, ny, nz]], dtype=np.float64)
    grid_bounds_real = grid_bounds_voxel * meshsize
    boundary_epsilon = meshsize * 0.05
    vox_is_tree, vox_is_opaque = _prepare_voxel_classes(voxel_data_modified, landmark_value)
    att = float(np.exp(-tree_k * tree_lad * meshsize))
    att_cutoff = 0.01
    visibility_values = _compute_all_faces_progress(
        face_centers,
        face_normals,
        landmark_positions,
        vox_is_tree, vox_is_opaque,
        float(meshsize), att, att_cutoff,
        grid_bounds_real.astype(np.float64),
        float(boundary_epsilon),
        progress_report=progress_report
    )
    building_mesh.metadata = getattr(building_mesh, 'metadata', {})
    building_mesh.metadata['landmark_visibility'] = visibility_values
    valid_mask = ~np.isnan(visibility_values)
    n_valid = np.sum(valid_mask)
    n_visible = np.sum(visibility_values[valid_mask] > 0.5)
    if progress_report:
        print(f"Landmark visibility statistics:")
        print(f"  Total faces: {len(visibility_values)}")
        print(f"  Valid faces: {n_valid}")
        print(f"  Faces with landmark visibility: {n_visible} ({n_visible/n_valid*100:.1f}%)")
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "surface_landmark_visibility")
        os.makedirs(output_dir, exist_ok=True)
        try:
            cmap = plt.cm.get_cmap(colormap)
            face_colors = np.zeros((len(visibility_values), 4))
            for i, val in enumerate(visibility_values):
                if np.isnan(val):
                    face_colors[i] = [0.7, 0.7, 0.7, 1.0]
                else:
                    face_colors[i] = cmap(val)
            building_mesh.visual.face_colors = face_colors
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    return building_mesh, voxel_data_modified

