"""
Stage 3: Time-series integration.
"""

import os
from datetime import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import numba

from ...models import VoxCity
from ...exporter.obj import grid_to_obj
from .radiation import (
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    compute_cumulative_solar_irradiance_faces_masked_timeseries,
    get_building_solar_irradiance,
)


def get_solar_positions_astral(times, lon, lat):
    """
    Compute solar azimuth and elevation for given times and location using Astral.
    Returns a DataFrame indexed by times with columns ['azimuth', 'elevation'] (degrees).
    """
    import pandas as pd
    from astral import Observer
    from astral.sun import elevation, azimuth

    observer = Observer(latitude=lat, longitude=lon)
    df_pos = pd.DataFrame(index=times, columns=['azimuth', 'elevation'], dtype=float)
    for t in times:
        el = elevation(observer=observer, dateandtime=t)
        az = azimuth(observer=observer, dateandtime=t)
        df_pos.at[t, 'elevation'] = el
        df_pos.at[t, 'azimuth'] = az
    return df_pos

def _configure_num_threads(desired_threads=None, progress=False):
    try:
        cores = os.cpu_count() or 4
    except Exception:
        cores = 4
    used = desired_threads if desired_threads is not None else cores
    try:
        numba.set_num_threads(int(used))
    except Exception:
        pass
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(int(used))
    if progress:
        try:
            print(f"Numba threads: {numba.get_num_threads()} (requested {used})")
        except Exception:
            print(f"Numba threads set to {used}")
    return used


def _auto_time_batch_size(n_faces, total_steps, user_value=None):
    if user_value is not None:
        return max(1, int(user_value))
    if total_steps <= 0:
        return 1
    if n_faces <= 5_000:
        batches = 2
    elif n_faces <= 50_000:
        batches = 8
    elif n_faces <= 200_000:
        batches = 16
    else:
        batches = 32
    batches = min(batches, total_steps)
    return max(1, total_steps // batches)


def get_cumulative_global_solar_irradiance(
    voxcity: VoxCity,
    df,
    lon,
    lat,
    tz,
    direct_normal_irradiance_scaling=1.0,
    diffuse_irradiance_scaling=1.0,
    **kwargs,
):
    """
    Integrate global horizontal irradiance over a period using EPW data.
    Returns W/m²·hour accumulation on the ground plane.
    """
    view_point_height = kwargs.get("view_point_height", 1.5)
    colormap = kwargs.get("colormap", "magma")
    start_time = kwargs.get("start_time", "01-01 05:00:00")
    end_time = kwargs.get("end_time", "01-01 20:00:00")
    desired_threads = kwargs.get("numba_num_threads", None)
    progress_report = kwargs.get("progress_report", False)
    _configure_num_threads(desired_threads, progress=progress_report)

    if df.empty:
        raise ValueError("No data in EPW dataframe.")

    try:
        start_dt = datetime.strptime(start_time, "%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve

    df = df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1

    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]

    df_period = df_period[
        ((df_period.index.hour != start_dt.hour) | (df_period.index.minute >= start_dt.minute)) &
        ((df_period.index.hour != end_dt.hour) | (df_period.index.minute <= end_dt.minute))
    ]

    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")

    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)

    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)

    diffuse_kwargs = kwargs.copy()
    diffuse_kwargs.update({'show_plot': False, 'obj_export': False})
    base_diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=1.0,
        **diffuse_kwargs
    )

    nx, ny, _ = voxcity.voxels.classes.shape
    cumulative_map = np.zeros((nx, ny))
    mask_map = np.ones((nx, ny), dtype=bool)

    direct_kwargs = kwargs.copy()
    direct_kwargs.update({'show_plot': False, 'view_point_height': view_point_height, 'obj_export': False})

    for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
        DNI = float(row['DNI']) * direct_normal_irradiance_scaling
        DHI = float(row['DHI']) * diffuse_irradiance_scaling

        solpos = solar_positions.loc[time_utc]
        azimuth_degrees = float(solpos['azimuth'])
        elevation_degrees = float(solpos['elevation'])

        direct_map = get_direct_solar_irradiance_map(
            voxcity,
            azimuth_degrees,
            elevation_degrees,
            direct_normal_irradiance=DNI,
            **direct_kwargs,
        )

        diffuse_map = base_diffuse_map * DHI
        global_map = direct_map + diffuse_map
        mask_map &= ~np.isnan(global_map)
        cumulative_map += np.nan_to_num(global_map, nan=0.0)

        if kwargs.get("show_each_timestep", False):
            vmin = kwargs.get("vmin", 0.0)
            vmax = kwargs.get("vmax", max(direct_normal_irradiance_scaling, diffuse_irradiance_scaling) * 1000)
            cmap = plt.cm.get_cmap(kwargs.get("colormap", "viridis")).copy()
            cmap.set_bad(color="lightgray")
            plt.figure(figsize=(10, 8))
            plt.imshow(global_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.colorbar(label="Global Solar Irradiance (W/m²)")
            plt.show()

    cumulative_map[~mask_map] = np.nan

    if kwargs.get("show_plot", True):
        vmin = kwargs.get("vmin", float(np.nanmin(cumulative_map)))
        vmax = kwargs.get("vmax", float(np.nanmax(cumulative_map)))
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color="lightgray")
        plt.figure(figsize=(10, 8))
        plt.imshow(cumulative_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Cumulative Global Solar Irradiance (W/m²·hour)")
        plt.axis("off")
        plt.show()

    if kwargs.get("obj_export", False):
        vmin = kwargs.get("vmin", float(np.nanmin(cumulative_map)))
        vmax = kwargs.get("vmax", float(np.nanmax(cumulative_map)))
        dem_grid = kwargs.get("dem_grid", voxcity.dem.elevation if voxcity.dem else np.zeros_like(cumulative_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "cumulative_global_solar_irradiance")
        num_colors = kwargs.get("num_colors", 10)
        alpha = kwargs.get("alpha", 1.0)
        meshsize = voxcity.voxels.meta.meshsize
        grid_to_obj(
            cumulative_map,
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

    return cumulative_map


def get_cumulative_building_solar_irradiance(
    voxcity: VoxCity,
    building_svf_mesh,
    weather_df,
    lon,
    lat,
    tz,
    **kwargs
):
    """
    Cumulative Wh/m² on building faces over a period from weather dataframe.
    """
    import numpy as _np

    period_start = kwargs.get("period_start", "01-01 00:00:00")
    period_end = kwargs.get("period_end", "12-31 23:59:59")
    time_step_hours = float(kwargs.get("time_step_hours", 1.0))
    direct_normal_irradiance_scaling = float(kwargs.get("direct_normal_irradiance_scaling", 1.0))
    diffuse_irradiance_scaling = float(kwargs.get("diffuse_irradiance_scaling", 1.0))
    progress_report = kwargs.get("progress_report", False)
    fast_path = kwargs.get("fast_path", True)

    try:
        start_dt = datetime.strptime(period_start, "%m-%d %H:%M:%S")
        end_dt = datetime.strptime(period_end, "%m-%d %H:%M:%S")
    except ValueError as ve:
        raise ValueError("Time must be in format 'MM-DD HH:MM:SS'") from ve

    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)

    df_period = weather_df[
        ((weather_df.index.month > start_dt.month) |
         ((weather_df.index.month == start_dt.month) &
          (weather_df.index.day >= start_dt.day) &
          (weather_df.index.hour >= start_dt.hour))) &
        ((weather_df.index.month < end_dt.month) |
         ((weather_df.index.month == end_dt.month) &
          (weather_df.index.day <= end_dt.day) &
          (weather_df.index.hour <= end_dt.hour)))
    ]
    if df_period.empty:
        raise ValueError("No weather data in specified period.")

    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)

    precomputed_solar_positions = kwargs.get("precomputed_solar_positions", None)
    if precomputed_solar_positions is not None and len(precomputed_solar_positions) == len(df_period_utc.index):
        solar_positions = precomputed_solar_positions
    else:
        solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)

    times_len = len(df_period_utc.index)
    azimuth_deg_arr = solar_positions['azimuth'].to_numpy()
    elev_deg_arr = solar_positions['elevation'].to_numpy()
    az_rad_arr = _np.deg2rad(180.0 - azimuth_deg_arr)
    el_rad_arr = _np.deg2rad(elev_deg_arr)
    sun_dx_arr = _np.cos(el_rad_arr) * _np.cos(az_rad_arr)
    sun_dy_arr = _np.cos(el_rad_arr) * _np.sin(az_rad_arr)
    sun_dz_arr = _np.sin(el_rad_arr)
    sun_dirs_arr = _np.stack([sun_dx_arr, sun_dy_arr, sun_dz_arr], axis=1).astype(_np.float64)
    DNI_arr = (df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling).astype(_np.float64)
    DHI_arr = (df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling).astype(_np.float64)
    sun_above_mask = elev_deg_arr > 0.0

    n_faces = len(building_svf_mesh.faces)
    face_cum_direct = _np.zeros(n_faces, dtype=_np.float64)
    face_cum_diffuse = _np.zeros(n_faces, dtype=_np.float64)
    face_cum_global = _np.zeros(n_faces, dtype=_np.float64)

    voxel_data = voxcity.voxels.classes
    meshsize = float(voxcity.voxels.meta.meshsize)

    precomputed_geometry = kwargs.get("precomputed_geometry", None)
    if precomputed_geometry is not None:
        face_centers = precomputed_geometry.get("face_centers", building_svf_mesh.triangles_center)
        face_normals = precomputed_geometry.get("face_normals", building_svf_mesh.face_normals)
        face_svf = precomputed_geometry.get(
            "face_svf",
            building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else _np.zeros(n_faces, dtype=_np.float64)
        )
        grid_bounds_real = precomputed_geometry.get("grid_bounds_real", None)
        boundary_epsilon = precomputed_geometry.get("boundary_epsilon", None)
    else:
        face_centers = building_svf_mesh.triangles_center
        face_normals = building_svf_mesh.face_normals
        face_svf = building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else _np.zeros(n_faces, dtype=_np.float64)
        grid_bounds_real = None
        boundary_epsilon = None

    if grid_bounds_real is None or boundary_epsilon is None:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = _np.array([[0, 0, 0], [grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=_np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05

    hit_values = (0,)
    inclusion_mode = False
    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)

    boundary_mask = None
    instant_kwargs = kwargs.copy()
    instant_kwargs['obj_export'] = False

    total_steps = times_len
    progress_every = max(1, total_steps // 20)

    face_centers64 = (face_centers if isinstance(face_centers, _np.ndarray) else building_svf_mesh.triangles_center).astype(_np.float64)
    face_normals64 = (face_normals if isinstance(face_normals, _np.ndarray) else building_svf_mesh.face_normals).astype(_np.float64)
    face_svf64 = face_svf.astype(_np.float64)
    x_min, y_min, z_min = grid_bounds_real[0, 0], grid_bounds_real[0, 1], grid_bounds_real[0, 2]
    x_max, y_max, z_max = grid_bounds_real[1, 0], grid_bounds_real[1, 1], grid_bounds_real[1, 2]

    if fast_path:
        precomputed_masks = kwargs.get("precomputed_masks", None)
        if precomputed_masks is not None:
            vox_is_tree = precomputed_masks.get("vox_is_tree", (voxel_data == -2))
            vox_is_opaque = precomputed_masks.get("vox_is_opaque", (voxel_data != 0) & (voxel_data != -2))
            att = float(precomputed_masks.get("att", _np.exp(-tree_k * tree_lad * meshsize)))
        else:
            vox_is_tree = (voxel_data == -2)
            vox_is_opaque = (voxel_data != 0) & (~vox_is_tree)
            att = float(_np.exp(-tree_k * tree_lad * meshsize))

        time_batch_size = _auto_time_batch_size(n_faces, total_steps, kwargs.get("time_batch_size", None))
        if progress_report:
            print(f"Faces: {n_faces:,}, Timesteps: {total_steps:,}, Batch size: {time_batch_size}")

        for start in range(0, total_steps, time_batch_size):
            end = min(start + time_batch_size, total_steps)
            ch_dir, ch_diff, ch_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
                face_centers64,
                face_normals64,
                face_svf64,
                sun_dirs_arr.astype(_np.float64),
                DNI_arr.astype(_np.float64),
                DHI_arr.astype(_np.float64),
                vox_is_tree,
                vox_is_opaque,
                float(meshsize),
                float(att),
                float(x_min), float(y_min), float(z_min),
                float(x_max), float(y_max), float(z_max),
                float(boundary_epsilon),
                int(start), int(end),
                float(time_step_hours)
            )
            face_cum_direct  += ch_dir
            face_cum_diffuse += ch_diff
            face_cum_global  += ch_glob
            if progress_report:
                pct = (end * 100.0) / total_steps
                print(f"Cumulative irradiance: {end}/{total_steps} ({pct:.1f}%)")
    else:
        for idx in range(total_steps):
            DNI = float(DNI_arr[idx])
            DHI = float(DHI_arr[idx])
            if not sun_above_mask[idx]:
                if boundary_mask is None:
                    boundary_mask = _np.isnan(face_svf)
                face_cum_diffuse += _np.nan_to_num(face_svf * DHI) * time_step_hours
                face_cum_global  += _np.nan_to_num(face_svf * DHI) * time_step_hours
                if progress_report and (((idx + 1) % progress_every == 0) or (idx == total_steps - 1)):
                    pct = (idx + 1) * 100.0 / total_steps
                    print(f"Cumulative irradiance: {idx+1}/{total_steps} ({pct:.1f}%)")
                continue

            irr_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh,
                float(azimuth_deg_arr[idx]),
                float(elev_deg_arr[idx]),
                DNI,
                DHI,
                show_plot=False,
                **instant_kwargs
            )
            face_direct = irr_mesh.metadata['direct']
            face_diffuse = irr_mesh.metadata['diffuse']
            face_global = irr_mesh.metadata['global']

            if boundary_mask is None:
                boundary_mask = _np.isnan(face_global)

            face_cum_direct  += _np.nan_to_num(face_direct)  * time_step_hours
            face_cum_diffuse += _np.nan_to_num(face_diffuse) * time_step_hours
            face_cum_global  += _np.nan_to_num(face_global)  * time_step_hours

            if progress_report and (((idx + 1) % progress_every == 0) or (idx == total_steps - 1)):
                pct = (idx + 1) * 100.0 / total_steps
                print(f"Cumulative irradiance: {idx+1}/{total_steps} ({pct:.1f}%)")

    if boundary_mask is not None:
        face_cum_direct[boundary_mask]  = _np.nan
        face_cum_diffuse[boundary_mask] = _np.nan
        face_cum_global[boundary_mask]  = _np.nan

    cumulative_mesh = building_svf_mesh.copy()
    if not hasattr(cumulative_mesh, 'metadata'):
        cumulative_mesh.metadata = {}
    if 'svf' in building_svf_mesh.metadata:
        cumulative_mesh.metadata['svf'] = building_svf_mesh.metadata['svf']
    cumulative_mesh.metadata['direct']  = face_cum_direct
    cumulative_mesh.metadata['diffuse'] = face_cum_diffuse
    cumulative_mesh.metadata['global']  = face_cum_global
    cumulative_mesh.name = "Cumulative Solar Irradiance (Wh/m²)"
    return cumulative_mesh


