"""
Stage 4: High-level workflows & I/O.
"""

from datetime import datetime
import pytz

from ...models import VoxCity
from ...utils.weather import (
    get_nearest_epw_from_climate_onebuilding,
    read_epw_for_solar_simulation,
)
from .radiation import get_global_solar_irradiance_map, get_building_solar_irradiance
from .temporal import get_cumulative_global_solar_irradiance, get_cumulative_building_solar_irradiance
from ..visibility import get_surface_view_factor


def get_global_solar_irradiance_using_epw(
    voxcity: VoxCity,
    calc_type: str = "instantaneous",
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    **kwargs,
):
    """
    Compute global irradiance from EPW, either instantaneous or cumulative.
    """
    # EPW acquisition
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    epw_file_path = kwargs.get("epw_file_path", None)
    # Extract rectangle_vertices with fallback to voxcity.extras
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    if rectangle_vertices is None:
        extras = getattr(voxcity, "extras", None)
        if isinstance(extras, dict):
            rectangle_vertices = extras.get("rectangle_vertices", None)
    if download_nearest_epw:
        if rectangle_vertices is None:
            print("rectangle_vertices is required to download nearest EPW file")
            return None
        lons = [coord[0] for coord in rectangle_vertices]
        lats = [coord[1] for coord in rectangle_vertices]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
        output_dir = kwargs.get("output_dir", "output")
        max_distance = kwargs.get("max_distance", kwargs.get("max_distance_km", 100))
        epw_file_path, weather_data, metadata = get_nearest_epw_from_climate_onebuilding(
            longitude=center_lon,
            latitude=center_lat,
            output_dir=output_dir,
            max_distance=max_distance,
            extract_zip=True,
            load_data=True,
            allow_insecure_ssl=kwargs.get("allow_insecure_ssl", False),
            allow_http_fallback=kwargs.get("allow_http_fallback", False),
            ssl_verify=kwargs.get("ssl_verify", True),
        )
    if not download_nearest_epw and not epw_file_path:
        raise ValueError("epw_file_path must be provided when download_nearest_epw is False")

    # Read EPW
    df, lon, lat, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")

    if calc_type == "instantaneous":
        calc_time = kwargs.get("calc_time", "01-01 12:00:00")
        try:
            calc_dt = datetime.strptime(calc_time, "%m-%d %H:%M:%S")
        except ValueError as ve:
            raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'") from ve

        df_period = df[
            (df.index.month == calc_dt.month)
            & (df.index.day == calc_dt.day)
            & (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")

        # Localize and convert to UTC
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)

        from .geometry import get_solar_positions_astral

        solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]["DNI"]) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]["DHI"]) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]["azimuth"])
        elevation_degrees = float(solar_positions.iloc[0]["elevation"])

        solar_map = get_global_solar_irradiance_map(
            voxcity,
            azimuth_degrees,
            elevation_degrees,
            DNI,
            DHI,
            show_plot=True,
            **kwargs,
        )
        return solar_map

    if calc_type == "cumulative":
        start_hour = kwargs.get("start_hour", 0)
        end_hour = kwargs.get("end_hour", 23)
        df_filtered = df[(df.index.hour >= start_hour) & (df.index.hour <= end_hour)]
        solar_map = get_cumulative_global_solar_irradiance(
            voxcity,
            df_filtered,
            lon,
            lat,
            tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            **kwargs,
        )
        return solar_map

    raise ValueError("calc_type must be 'instantaneous' or 'cumulative'")
def get_building_global_solar_irradiance_using_epw(*args, **kwargs):
    """
    Compute building-surface irradiance using EPW (instantaneous or cumulative).
    """
    voxcity: VoxCity = kwargs.get("voxcity")
    if voxcity is None and len(args) > 0 and isinstance(args[0], VoxCity):
        voxcity = args[0]
    if voxcity is None:
        raise ValueError("voxcity (VoxCity) must be provided as first arg or kwarg")

    calc_type = kwargs.get("calc_type", "instantaneous")
    direct_normal_irradiance_scaling = float(kwargs.get("direct_normal_irradiance_scaling", 1.0))
    diffuse_irradiance_scaling = float(kwargs.get("diffuse_irradiance_scaling", 1.0))
    building_svf_mesh = kwargs.get("building_svf_mesh", None)
    building_id_grid = kwargs.get("building_id_grid", None)
    progress_report = kwargs.get("progress_report", False)
    fast_path = kwargs.get("fast_path", True)

    # Thread configuration
    from .temporal import _configure_num_threads
    desired_threads = kwargs.get("numba_num_threads", None)
    _configure_num_threads(desired_threads, progress=progress_report)

    # EPW acquisition
    download_nearest_epw = kwargs.get("download_nearest_epw", False)
    epw_file_path = kwargs.get("epw_file_path", None)
    # Extract rectangle_vertices with fallback to voxcity.extras
    rectangle_vertices = kwargs.get("rectangle_vertices", None)
    if rectangle_vertices is None:
        extras = getattr(voxcity, "extras", None)
        if isinstance(extras, dict):
            rectangle_vertices = extras.get("rectangle_vertices", None)
    if download_nearest_epw:
        if rectangle_vertices is None:
            print("rectangle_vertices is required to download nearest EPW file")
            return None
        lons = [coord[0] for coord in rectangle_vertices]
        lats = [coord[1] for coord in rectangle_vertices]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
        output_dir = kwargs.get("output_dir", "output")
        max_distance = kwargs.get("max_distance", kwargs.get("max_distance_km", 100))
        epw_file_path, _weather_data, _metadata = get_nearest_epw_from_climate_onebuilding(
            longitude=center_lon,
            latitude=center_lat,
            output_dir=output_dir,
            max_distance=max_distance,
            extract_zip=True,
            load_data=True,
            allow_insecure_ssl=kwargs.get("allow_insecure_ssl", False),
            allow_http_fallback=kwargs.get("allow_http_fallback", False),
            ssl_verify=kwargs.get("ssl_verify", True),
        )
    if not download_nearest_epw and not epw_file_path:
        raise ValueError("epw_file_path must be provided when download_nearest_epw is False")

    # Read EPW
    df, lon, lat, tz, _elevation_m = read_epw_for_solar_simulation(epw_file_path)
    if df.empty:
        raise ValueError("No data in EPW file.")

    # SVF for building faces (compute if not provided)
    if building_svf_mesh is None:
        if progress_report:
            print("Processing Sky View Factor for building surfaces...")
        svf_kwargs = {
            'value_name': 'svf',
            'target_values': (0,),
            'inclusion_mode': False,
            'building_id_grid': building_id_grid,
            'progress_report': progress_report,
            'fast_path': fast_path,
        }
        for k in ("N_azimuth", "N_elevation", "tree_k", "tree_lad", "debug"):
            if k in kwargs:
                svf_kwargs[k] = kwargs[k]
        building_svf_mesh = get_surface_view_factor(voxcity, **svf_kwargs)

    # Precompute geometry/masks
    import numpy as _np
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    precomputed_geometry = {}
    try:
        grid_shape = voxel_data.shape
        grid_bounds_voxel = _np.array([[0, 0, 0], [grid_shape[0], grid_shape[1], grid_shape[2]]], dtype=_np.float64)
        grid_bounds_real = grid_bounds_voxel * meshsize
        boundary_epsilon = meshsize * 0.05
        precomputed_geometry = {
            'face_centers': building_svf_mesh.triangles_center,
            'face_normals': building_svf_mesh.face_normals,
            'face_svf': building_svf_mesh.metadata['svf'] if ('svf' in building_svf_mesh.metadata) else None,
            'grid_bounds_real': grid_bounds_real,
            'boundary_epsilon': boundary_epsilon,
        }
    except Exception:
        precomputed_geometry = {}

    tree_k = kwargs.get("tree_k", 0.6)
    tree_lad = kwargs.get("tree_lad", 1.0)
    precomputed_masks = {
        'vox_is_tree': (voxel_data == -2),
        'vox_is_opaque': (voxel_data != 0) & (voxel_data != -2),
        'att': float(_np.exp(-tree_k * tree_lad * meshsize)),
    }

    if progress_report:
        t_cnt = int(_np.count_nonzero(precomputed_masks['vox_is_tree']))
        print(f"Precomputed caches: trees={t_cnt:,}, tree_att_per_voxel={precomputed_masks['att']:.4f}")

    result_mesh = None
    if calc_type == "instantaneous":
        calc_time = kwargs.get("calc_time", "01-01 12:00:00")
        try:
            calc_dt = datetime.strptime(calc_time, "%m-%d %H:%M:%S")
        except ValueError as ve:
            raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'") from ve

        df_period = df[
            (df.index.month == calc_dt.month) & (df.index.day == calc_dt.day) & (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")

        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)

        from .geometry import get_solar_positions_astral
        solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
        elevation_degrees = float(solar_positions.iloc[0]['elevation'])

        _call_kwargs = kwargs.copy()
        _call_kwargs.update({
            'progress_report': progress_report,
            'fast_path': fast_path,
            'precomputed_geometry': precomputed_geometry,
            'precomputed_masks': precomputed_masks,
        })
        result_mesh = get_building_solar_irradiance(
            voxcity,
            building_svf_mesh,
            azimuth_degrees,
            elevation_degrees,
            DNI,
            DHI,
            **_call_kwargs
        )
    elif calc_type == "cumulative":
        period_start = kwargs.get("period_start", "01-01 00:00:00")
        period_end = kwargs.get("period_end", "12-31 23:59:59")
        time_step_hours = float(kwargs.get("time_step_hours", 1.0))

        result_mesh = get_cumulative_building_solar_irradiance(
            voxcity,
            building_svf_mesh,
            df,
            lon,
            lat,
            tz,
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            progress_report=progress_report,
            fast_path=fast_path,
            precomputed_geometry=precomputed_geometry,
            precomputed_masks=precomputed_masks,
        )
    else:
        raise ValueError("calc_type must be 'instantaneous' or 'cumulative'")

    # Optional persist
    if kwargs.get("save_mesh", False):
        mesh_output_path = kwargs.get("mesh_output_path")
        if not mesh_output_path:
            output_directory = kwargs.get("output_directory", "output")
            output_file_name = kwargs.get("output_file_name", f"{calc_type}_solar_irradiance")
            mesh_output_path = f"{output_directory}/{output_file_name}.pkl"
        save_irradiance_mesh(result_mesh, mesh_output_path)
        if progress_report:
            print(f"Saved irradiance mesh data to: {mesh_output_path}")

    return result_mesh


def save_irradiance_mesh(irradiance_mesh, output_file_path):
    """
    Persist irradiance mesh to pickle file.
    """
    import pickle
    import os
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'wb') as f:
        pickle.dump(irradiance_mesh, f)


def load_irradiance_mesh(input_file_path):
    """
    Load irradiance mesh from pickle file.
    """
    import pickle
    with open(input_file_path, 'rb') as f:
        irradiance_mesh = pickle.load(f)
    return irradiance_mesh


