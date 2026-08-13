"""
Building surface solar irradiance functions for VoxCity.

This module provides GPU-accelerated building surface solar irradiance calculations:
- Building solar irradiance (instantaneous)
- Cumulative building solar irradiance
- Building sunlight hours (PSH and DSH modes)
- EPW-based building irradiance wrapper

These functions match the voxcity.simulator.solar API signatures for 
drop-in replacement with GPU acceleration.
"""

import logging
from functools import lru_cache

import numpy as np
from typing import Optional, Tuple, Dict
from scipy.spatial import cKDTree

from voxcity.geoprocessor.surface_meta import resolve_target_face_mask
from voxcity.simulator.common.coordinates import uv_domain_points_to_scene

from .utils import (
    VOXCITY_BUILDING_CODE,
    get_location_from_voxcity,
    compute_sun_direction,
    filter_df_to_period,
    parse_time_period,
    load_epw_data,
    get_solar_positions_astral,
    compute_boundary_vertical_mask,
    apply_computation_mask_to_faces,
    get_timezone_offset_from_location,
    generate_annual_hourly_dataframe,
)

from .caching import (
    get_building_radiation_model_cache,
    get_or_create_building_radiation_model,
    CachedBuildingRadiationModel,
    BUILDING_SURFACE_CLASSES,
)


logger = logging.getLogger(__name__)


def _scene_grid_bounds(shape, meshsize):
    ny_vc, nx_vc, nz = shape
    return np.array([
        [0.0, 0.0, 0.0],
        [nx_vc * meshsize, ny_vc * meshsize, nz * meshsize],
    ], dtype=np.float64)


def _mesh_geometry_signature(mesh):
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        bounds = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    else:
        bounds_arr = np.round(np.asarray(mesh.bounds, dtype=np.float64), 9)
        bounds = (tuple(bounds_arr[0]), tuple(bounds_arr[1]))
    return (id(mesh), len(mesh.faces), bounds)


@lru_cache(maxsize=1)
def _direction_to_scene_normal_key():
    """Map each surface ``direction`` index (0-5, see ``domain.DIR_NORMALS``)
    to the unit-axis normal key it corresponds to in scene coordinates.

    Mesh faces from ``create_voxel_mesh`` are always axis-aligned, so their
    face normals round exactly to one of these six keys. A surface's
    ``direction`` is the voxel-face axis it was built from -- including for
    override surfaces, where it is the exposed face the ray leaves from, not
    the (possibly non-axis-aligned) true polygon normal -- so ``direction``
    is the correct join key between mesh faces and surfaces, independent of
    whether the surface's own normal happens to round to a unit axis.
    """
    from ..domain import DIR_NORMALS

    mapping = {}
    for direction, normal in DIR_NORMALS.items():
        scene_normal = uv_domain_points_to_scene(np.array(normal, dtype=np.float64))
        key = tuple(int(v) for v in np.rint(scene_normal).astype(np.int8))
        mapping[direction] = key
    return mapping


def _map_mesh_faces_to_surfaces(
    mesh_face_centers,
    mesh_face_normals,
    surface_centers_scene,
    surface_directions,
    bldg_indices,
    max_match_distance,
):
    """Match voxel-mesh faces to solver surfaces.

    Both sides are bucketed by the voxel-face axis they occupy: mesh faces
    by their (always axis-aligned) rounded face normal, surfaces by their
    ``direction`` index translated to the same scene-axis key. Keying
    surfaces by ``direction`` rather than by rounding their own normal keeps
    this correct even when override surfaces carry a true, non-axis-aligned
    normal (e.g. a sloped facade whose normal rounds to (1, 1, 0)) -- such a
    surface's ``direction`` is still one of the six voxel-face axes.

    Within a bucket, each face is matched to its nearest surface centroid,
    but only if that centroid is within ``max_match_distance``; otherwise the
    face gets -1 (no match), which callers already render as NaN rather than
    silently inheriting a distant, implausible surface.
    """
    mesh_normals_key = np.rint(mesh_face_normals).astype(np.int8)
    direction_key_map = _direction_to_scene_normal_key()
    key_to_direction = {key: direction for direction, key in direction_key_map.items()}

    result = np.full(len(mesh_face_centers), -1, dtype=np.int64)
    bldg_directions = surface_directions[bldg_indices]

    for normal_key in np.unique(mesh_normals_key, axis=0):
        direction = key_to_direction.get(tuple(int(v) for v in normal_key))
        face_mask = np.all(mesh_normals_key == normal_key, axis=1)
        if direction is None:
            # Not one of the six voxel-face axes -- shouldn't happen for an
            # axis-aligned voxel mesh, but leave these faces unmatched (-1)
            # rather than guessing.
            continue

        surface_mask = bldg_directions == direction
        candidate_indices = bldg_indices[surface_mask]

        if candidate_indices.size == 0:
            continue

        tree = cKDTree(surface_centers_scene[candidate_indices])
        dist, nearest_idx = tree.query(mesh_face_centers[face_mask], k=1)
        matched = candidate_indices[nearest_idx]
        within_cap = dist <= max_match_distance

        face_indices = np.flatnonzero(face_mask)
        result[face_indices[within_cap]] = matched[within_cap]

    return result


_OVERRIDE_EXPORT_KEYS = ('surface_override_normals', 'surface_override_index')


def _capture_override_export(irradiance_mesh, n_faces):
    """Lift get_building_solar_irradiance's surface_override export off one
    per-timestep result, for an accumulating caller to re-attach to the mesh
    it actually returns.

    get_cumulative_building_solar_irradiance and get_building_sunlight_hours
    both return a copy of the input mesh taken *before* their loop and
    accumulate scalars into it, so the export -- which
    get_building_solar_irradiance writes onto the per-timestep meshes --
    never reaches the caller unless it is carried across explicitly. Without
    it a consumer of those two functions is back to guessing which normal
    produced a value, which is the defect the export exists to fix.

    Carrying one timestep's copy is exact, not an approximation: the export
    is a function of the mesh<->surface join and the surface normals, neither
    of which depends on sun position, so every timestep produces the same
    arrays. The first result that carries it is therefore authoritative.

    Returns None when there is nothing to carry: no override table was
    active, or the result's face count doesn't line up with the mesh being
    accumulated into -- the same length guard the value accumulation applies
    around it.

    The arrays are handed on by reference, which is safe because
    get_building_solar_irradiance already builds each result's pair as fresh
    copies detached from the model cache; nothing here aliases a cache
    buffer.
    """
    metadata = getattr(irradiance_mesh, 'metadata', None) or {}
    if any(key not in metadata for key in _OVERRIDE_EXPORT_KEYS):
        return None
    if any(len(metadata[key]) != n_faces for key in _OVERRIDE_EXPORT_KEYS):
        return None
    return {key: metadata[key] for key in _OVERRIDE_EXPORT_KEYS}


def _drop_override_export(metadata):
    """Clear an inherited export from a metadata dict when the run that
    produced it had no override table of its own.

    Every entry point here starts from the caller's mesh metadata -- the
    single-timestep function copies the input mesh's dict, the two
    accumulating ones start from ``building_svf_mesh.copy()``, which copies
    metadata too. A mesh that has already been through an override run
    carries that run's export, so without this it would ride through a run
    made *without* a table. Presence of these keys has to keep meaning "this
    run had an override" or a consumer keying off it reads stale normals with
    no way to notice.
    """
    for key in _OVERRIDE_EXPORT_KEYS:
        metadata.pop(key, None)


# =============================================================================
# Public API Functions
# =============================================================================

def get_building_solar_irradiance(
    voxcity,
    building_svf_mesh=None,
    azimuth_degrees_ori: float = None,
    elevation_degrees: float = None,
    direct_normal_irradiance: float = None,
    diffuse_irradiance: float = None,
    **kwargs
):
    """
    GPU-accelerated building surface solar irradiance computation.
    
    Uses cached RadiationModel to avoid recomputing SVF/CSF matrices for each timestep.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Pre-computed mesh with SVF values (optional)
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        **kwargs: Additional parameters including:
            - with_reflections (bool): Enable multi-bounce reflections (default: False)
            - n_reflection_steps (int): Number of reflection bounces (default: 2)
            - building_class_id (int or iterable): Building-surface voxel
              class code(s); default (-3, -16) includes window/glass cells.
            - computation_mask (np.ndarray): Optional 2D boolean mask
            - target_selectors (list): Optional surface selectors limiting returned faces
            - reference_mesh: Optional reference mesh for target selector metadata fast path
            - progress_report (bool): Print progress (default: False)
            - surface_override: Optional table of true polygon normals (see
              ``solar.surface_override.SurfaceOverride``). Also gates the
              surface_override_* metadata described below.

    Returns:
        Trimesh object with irradiance values in metadata.

        When ``surface_override`` is supplied AND the model has building
        surfaces, the metadata also carries the join back to the override
        table. These key names are a wire protocol for consumers projecting
        these values onto a polygon model; nothing in this package reads
        them, and there is no import relationship in either direction, so
        renaming a key is a breaking change.

          surface_override_normals : float64, (n_faces, 3), scene coords
              The normal the solver actually used for that face -- the true
              polygon normal, not the staircase mesh's axis normal -- as a
              unit vector. NaN row where the face matched no surface.
              Scene coords as defined in ``simulator/common/coordinates.py``:
              x = v/east, y = u/north, z = up.
          surface_override_index : int64, (n_faces,)
              Index into the model's surface set; -1 where unmatched. Under
              an override table this is the table's own row index --
              ``surfaces_from_override`` emits one surface per row, in order.

        Read both with ``.get()``: they are ABSENT (not empty) when there
        are no building surfaces, so "override supplied" does not imply
        "keys present". An empty override table does write them, carrying
        the occupancy-derived axis normals. They are deliberately NOT
        masked by the boundary / computation / target masks that NaN out
        the irradiance arrays, so the two sets of arrays have different
        validity patterns: a face whose irradiance is NaN still reports the
        real normal that produced it.
    """
    # Handle positional argument order from VoxCity API
    if isinstance(building_svf_mesh, (int, float)):
        diffuse_irradiance = direct_normal_irradiance
        direct_normal_irradiance = elevation_degrees
        elevation_degrees = azimuth_degrees_ori
        azimuth_degrees_ori = building_svf_mesh
        building_svf_mesh = None
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    ny_vc, nx_vc, nz = voxel_data.shape
    
    # Extract parameters
    progress_report = kwargs.pop('progress_report', False)
    building_class_id = kwargs.pop('building_class_id', BUILDING_SURFACE_CLASSES)
    n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
    with_reflections = kwargs.pop('with_reflections', False)
    computation_mask = kwargs.pop('computation_mask', None)
    target_selectors = kwargs.pop('target_selectors', None)
    # Popped rather than left in kwargs: get_or_create_building_radiation_model
    # is called below as (..., surface_override=surface_override, **kwargs);
    # if 'surface_override' were still in kwargs that call would raise
    # TypeError: got multiple values for argument 'surface_override'.
    surface_override = kwargs.pop('surface_override', None)
    if target_selectors is not None and with_reflections:
        logger.warning(
            "target_selectors for building solar irradiance currently use output-only restriction; "
            "disabling reflections because reflected exchange is computed over the full PALM surface set."
        )
        with_reflections = False
        n_reflection_steps = 0
    
    if not with_reflections:
        n_reflection_steps = 0
    
    # Get cached or create new RadiationModel
    model, is_building_surf = get_or_create_building_radiation_model(
        voxcity,
        n_reflection_steps=n_reflection_steps,
        progress_report=progress_report,
        building_class_id=building_class_id,
        surface_override=surface_override,
        **kwargs
    )
    
    # Set solar position
    rotation_angle = 0
    extras = getattr(voxcity, 'extras', None)
    if isinstance(extras, dict):
        rotation_angle = extras.get('rotation_angle', 0)

    sun_dir_x, sun_dir_y, sun_dir_z, cos_zenith = compute_sun_direction(
        azimuth_degrees_ori, elevation_degrees, rotation_angle
    )
    
    model.solar_calc.sun_direction[None] = (sun_dir_x, sun_dir_y, sun_dir_z)
    model.solar_calc.cos_zenith[None] = cos_zenith
    model.solar_calc.sun_up[None] = 1 if elevation_degrees > 0 else 0
    
    # Compute radiation
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance
    n_surfaces = model.surfaces.count
    sw_in_direct_all = model.surfaces.sw_in_direct.to_numpy()
    sw_in_diffuse_all = model.surfaces.sw_in_diffuse.to_numpy()
    
    if hasattr(model.surfaces, 'sw_in_reflected'):
        sw_in_reflected_all = model.surfaces.sw_in_reflected.to_numpy()
    else:
        sw_in_reflected_all = np.zeros_like(sw_in_direct_all)
    
    total_sw_all = sw_in_direct_all + sw_in_diffuse_all + sw_in_reflected_all
    
    # Get building indices from cache
    cache = get_building_radiation_model_cache()
    bldg_indices = cache.bldg_indices if cache else np.where(is_building_surf)[0]
    
    # Get or create building mesh
    if building_svf_mesh is not None:
        building_mesh = building_svf_mesh
        face_svf = building_mesh.metadata.get('svf') if hasattr(building_mesh, 'metadata') else None
    elif cache is not None and cache.cached_building_mesh is not None:
        building_mesh = cache.cached_building_mesh
        face_svf = None
    else:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            if progress_report:
                print("  Creating building mesh (first call, will be cached)...")
            building_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
            if building_mesh is None or len(building_mesh.faces) == 0:
                print("No building surfaces found.")
                return None
            if cache is not None:
                cache.cached_building_mesh = building_mesh
        except ImportError:
            print("VoxCity geoprocessor.mesh required for mesh creation")
            return None
        face_svf = None
    
    n_mesh_faces = len(building_mesh.faces)
    target_face_mask = None
    if target_selectors is not None:
        reference_mesh = kwargs.get("reference_mesh", None)
        target_face_mask = resolve_target_face_mask(
            building_mesh, target_selectors, reference_mesh=reference_mesh,
        )
        if target_face_mask.shape != (n_mesh_faces,):
            raise ValueError("target_selectors resolved to a mask with the wrong face count")

    mesh_signature = _mesh_geometry_signature(building_mesh)
    cache_matches_mesh = (
        cache is not None and
        cache.mesh_geometry_signature == mesh_signature
    )
    
    # Filled in below only when an override table is active; stays None on the
    # no-buildings path so that branch attaches nothing either.
    used_normals = None
    used_index = None

    # Map palm_solar values to mesh faces
    if len(bldg_indices) > 0:
        if (cache is not None and 
            cache.mesh_to_surface_idx is not None and
            len(cache.mesh_to_surface_idx) == n_mesh_faces and
            cache_matches_mesh):
            mesh_to_surface_idx = cache.mesh_to_surface_idx
        else:
            surf_centers_all = model.surfaces.center.to_numpy()[:n_surfaces]
            surf_centers_scene = uv_domain_points_to_scene(surf_centers_all)
            surf_directions_all = model.surfaces.direction.to_numpy()[:n_surfaces]

            if cache_matches_mesh and cache.mesh_face_centers is not None:
                mesh_face_centers = cache.mesh_face_centers
            else:
                mesh_face_centers = building_mesh.triangles_center
                if cache is not None:
                    cache.mesh_face_centers = mesh_face_centers.copy()
                    cache.mesh_face_normals = building_mesh.face_normals.copy()
            
            mesh_to_surface_idx = _map_mesh_faces_to_surfaces(
                mesh_face_centers,
                building_mesh.face_normals,
                surf_centers_scene,
                surf_directions_all,
                bldg_indices,
                # A correct match sits at ~0.47 m (half a voxel-face
                # diagonal-ish offset) on real cities; 2x meshsize leaves
                # generous headroom over that while still confidently
                # rejecting the 2-28 m cross-bucket mismatches the old,
                # uncapped rint-keyed matching produced (see
                # tests/simulator_gpu/test_mesh_surface_mapping.py).
                max_match_distance=2.0 * meshsize,
            )
            
            if cache is not None:
                cache.mesh_to_surface_idx = mesh_to_surface_idx
                cache.mesh_geometry_signature = mesh_signature
        
        valid_surface_mask = mesh_to_surface_idx >= 0

        if surface_override is not None:
            # The returned mesh is the axis-aligned voxel staircase, but with
            # an override table the value on a face was computed against the
            # true polygon normal instead. Export that normal per face so a
            # consumer projecting these values onto a polygon model can route
            # each one to the surface that produced it; matching by axis
            # bucket alone puts a value computed for a tilted facade on a
            # horizontal roof polygon, which then reports more than a
            # horizontal surface can physically receive.
            #
            # Cached beside mesh_to_surface_idx, and needs no key of its own:
            # it is a function of that mapping and of the model's surface
            # normals, and both already live in this cache object, which
            # get_or_create_building_radiation_model replaces outright (with
            # mesh_to_surface_idx=None) whenever the override signature
            # changes. The surface normals cannot drift under a surviving
            # cache object either: radiation.py:166-174 makes it a rule that
            # model.surfaces is assigned only in __init__ ("Injection must
            # happen HERE and not by assignment after construction"), so a
            # different surface set means a different model, hence a
            # different cache object.
            #
            # Worth caching because the callers loop over this function and
            # an uncached export would re-download surfaces.normal from the
            # device on every iteration -- on the very branch that exists to
            # touch no surface arrays at all. The loop length spans ~60x:
            # get_cumulative_building_solar_irradiance's per-timestep mode
            # can reach 8760 steps (~38 s of redundant downloads), while its
            # Tregenza-patch mode loops over ~145 active patches (~0.6 s) and
            # get_building_sunlight_hours over sunshine timesteps only.
            if (cache is not None and cache.mesh_used_normals is not None and
                    len(cache.mesh_used_normals) == n_mesh_faces and
                    cache_matches_mesh):
                used_normals = cache.mesh_used_normals
            else:
                # uv_domain_points_to_scene swaps the x/y columns of a copy
                # and translates nothing (see coordinates.py, and its
                # scene_vectors_to_uv_domain alias), so it is as correct for
                # normals as it is for the centres above.
                surf_normals_scene = uv_domain_points_to_scene(
                    model.surfaces.normal.to_numpy()[:n_surfaces].astype(np.float64))
                used_normals = np.full((n_mesh_faces, 3), np.nan, dtype=np.float64)
                used_normals[valid_surface_mask] = (
                    surf_normals_scene[mesh_to_surface_idx[valid_surface_mask]])
                if cache is not None:
                    cache.mesh_used_normals = used_normals
                    cache.mesh_geometry_signature = mesh_signature

            # Hand out copies, never the cached arrays themselves: on the
            # cache-hit path both of these ARE the cache's own buffers, and a
            # consumer that writes into what it was handed would silently
            # corrupt every later timestep. Keep the .copy() / .astype()
            # here -- np.asarray(..., dtype=np.int64) or astype(copy=False)
            # look like equivalent tidy-ups and are not.
            used_normals = used_normals.copy()
            # -1 marks "this face matched no surface", pairing with the NaN
            # rows above.
            used_index = mesh_to_surface_idx.astype(np.int64)

        sw_in_direct = np.full(n_mesh_faces, np.nan, dtype=np.float64)
        sw_in_diffuse = np.full(n_mesh_faces, np.nan, dtype=np.float64)
        sw_in_reflected = np.full(n_mesh_faces, np.nan, dtype=np.float64)
        total_sw = np.full(n_mesh_faces, np.nan, dtype=np.float64)
        sw_in_direct[valid_surface_mask] = sw_in_direct_all[mesh_to_surface_idx[valid_surface_mask]]
        sw_in_diffuse[valid_surface_mask] = sw_in_diffuse_all[mesh_to_surface_idx[valid_surface_mask]]
        sw_in_reflected[valid_surface_mask] = sw_in_reflected_all[mesh_to_surface_idx[valid_surface_mask]]
        total_sw[valid_surface_mask] = total_sw_all[mesh_to_surface_idx[valid_surface_mask]]
    else:
        sw_in_direct = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_diffuse = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_reflected = np.zeros(n_mesh_faces, dtype=np.float32)
        total_sw = np.zeros(n_mesh_faces, dtype=np.float32)
    
    # Handle boundary faces
    if (cache is not None and cache.boundary_mask is not None and
            len(cache.boundary_mask) == n_mesh_faces and cache_matches_mesh):
        is_boundary_vertical = cache.boundary_mask
    else:
        grid_bounds_real = _scene_grid_bounds(voxel_data.shape, meshsize)
        boundary_epsilon = meshsize * 0.05
        
        if cache_matches_mesh and cache.mesh_face_centers is not None:
            mesh_face_centers = cache.mesh_face_centers
            mesh_face_normals = cache.mesh_face_normals
        else:
            mesh_face_centers = building_mesh.triangles_center
            mesh_face_normals = building_mesh.face_normals
        
        is_boundary_vertical = compute_boundary_vertical_mask(
            mesh_face_centers, mesh_face_normals, grid_bounds_real, boundary_epsilon
        )
        
        if cache is not None:
            cache.boundary_mask = is_boundary_vertical
            cache.mesh_geometry_signature = mesh_signature
    
    sw_in_direct = np.where(is_boundary_vertical, np.nan, sw_in_direct)
    sw_in_diffuse = np.where(is_boundary_vertical, np.nan, sw_in_diffuse)
    sw_in_reflected = np.where(is_boundary_vertical, np.nan, sw_in_reflected)
    total_sw = np.where(is_boundary_vertical, np.nan, total_sw)
    
    # Apply computation mask
    if computation_mask is not None:
        if cache_matches_mesh and cache.mesh_face_centers is not None:
            mesh_face_centers = cache.mesh_face_centers
        else:
            mesh_face_centers = building_mesh.triangles_center
        
        sw_in_direct = apply_computation_mask_to_faces(
            sw_in_direct, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        sw_in_diffuse = apply_computation_mask_to_faces(
            sw_in_diffuse, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        sw_in_reflected = apply_computation_mask_to_faces(
            sw_in_reflected, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        total_sw = apply_computation_mask_to_faces(
            total_sw, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )

    if target_face_mask is not None:
        sw_in_direct = np.where(target_face_mask, sw_in_direct, np.nan)
        sw_in_diffuse = np.where(target_face_mask, sw_in_diffuse, np.nan)
        sw_in_reflected = np.where(target_face_mask, sw_in_reflected, np.nan)
        total_sw = np.where(target_face_mask, total_sw, np.nan)

    metadata = dict(getattr(building_mesh, 'metadata', {}) or {})
    metadata.update({
        'irradiance_direct': sw_in_direct,
        'irradiance_diffuse': sw_in_diffuse,
        'irradiance_reflected': sw_in_reflected,
        'irradiance_total': total_sw,
        'direct': sw_in_direct,
        'diffuse': sw_in_diffuse,
        'global': total_sw,
    })
    if used_normals is not None:
        metadata['surface_override_normals'] = used_normals
        metadata['surface_override_index'] = used_index
    else:
        # `metadata` starts as a copy of the input mesh's own dict, and this
        # function writes its result back onto that same mesh object -- so a
        # caller reusing a mesh across runs would otherwise carry the earlier
        # override run's export into this one. See _drop_override_export.
        _drop_override_export(metadata)
    building_mesh.metadata = metadata
    if face_svf is not None:
        building_mesh.metadata['svf'] = face_svf
    
    if kwargs.get('obj_export', False):
        import os
        output_dir = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'building_solar_irradiance')
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh


def get_cumulative_building_solar_irradiance(
    voxcity,
    building_svf_mesh,
    weather_df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    **kwargs
):
    """
    GPU-accelerated cumulative solar irradiance on building surfaces.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Trimesh object with SVF in metadata
        weather_df: pandas DataFrame with 'DNI' and 'DHI' columns
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with cumulative irradiance (Wh/m²) in metadata
    """
    from datetime import datetime
    import pytz
    
    kwargs = dict(kwargs)
    period_start = kwargs.pop('period_start', '01-01 00:00:00')
    period_end = kwargs.pop('period_end', '12-31 23:59:59')
    daily_start_hour = kwargs.pop('daily_start_hour', None)
    daily_end_hour = kwargs.pop('daily_end_hour', None)
    time_step_hours = float(kwargs.pop('time_step_hours', 1.0))
    progress_report = kwargs.pop('progress_report', False)
    use_sky_patches = kwargs.pop('use_sky_patches', False)
    computation_mask = kwargs.pop('computation_mask', None)
    target_selectors = kwargs.get('target_selectors', None)
    
    if weather_df.empty:
        raise ValueError("No data in weather dataframe.")
    
    # Filter dataframe
    df_period_utc = filter_df_to_period(weather_df, period_start, period_end, tz,
                                        daily_start_hour=daily_start_hour,
                                        daily_end_hour=daily_end_hour)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Initialize
    result_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
    n_faces = len(result_mesh.faces) if hasattr(result_mesh, 'faces') else 0
    
    if n_faces == 0:
        raise ValueError("Building mesh has no faces")

    target_face_mask = None
    if target_selectors is not None:
        reference_mesh = kwargs.get("reference_mesh", None)
        target_face_mask = resolve_target_face_mask(
            result_mesh, target_selectors, reference_mesh=reference_mesh,
        )
        if target_face_mask.shape != (n_faces,):
            raise ValueError("target_selectors resolved to a mask with the wrong face count")
    
    cumulative_direct = np.zeros(n_faces, dtype=np.float64)
    cumulative_diffuse = np.zeros(n_faces, dtype=np.float64)
    cumulative_global = np.zeros(n_faces, dtype=np.float64)
    # Filled from the first per-timestep result that carries it, and attached
    # to result_mesh at the end -- see _capture_override_export.
    override_export = None

    face_svf = result_mesh.metadata.get('svf') if hasattr(result_mesh, 'metadata') else None
    
    # Extract arrays
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    elevation_arr = solar_positions['elevation'].to_numpy()
    dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
    dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
    n_timesteps = len(azimuth_arr)
    
    if use_sky_patches:
        from ..sky import generate_sky_patches, get_tregenza_patch_index
        
        sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
        sky_patches = generate_sky_patches(sky_discretization)
        patches = sky_patches.patches
        n_patches = sky_patches.n_patches
        cumulative_dni_per_patch = np.zeros(n_patches, dtype=np.float64)
        total_cumulative_dhi = 0.0
        
        # Bin sun positions
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            dhi = dhi_arr[i]
            
            if dhi > 0:
                total_cumulative_dhi += dhi * time_step_hours
            
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            if dni <= 0:
                continue
            
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if 0 <= patch_idx < n_patches:
                cumulative_dni_per_patch[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni_per_patch > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            print(f"  Sky patch optimization: {n_timesteps} -> {n_active} active patches")
        
        # Diffuse component
        if face_svf is not None and len(face_svf) == n_faces:
            cumulative_diffuse = face_svf * total_cumulative_dhi
        else:
            diffuse_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=180.0,
                elevation_degrees=45.0,
                direct_normal_irradiance=0.0,
                diffuse_irradiance=1.0,
                progress_report=False,
                **kwargs
            )
            if diffuse_mesh is not None and 'diffuse' in diffuse_mesh.metadata:
                base_diffuse = diffuse_mesh.metadata['diffuse']
                cumulative_diffuse = np.nan_to_num(base_diffuse, nan=0.0) * total_cumulative_dhi
                # Also a capture site, and the only one an overcast period
                # reaches: with no direct beam anywhere in the period the
                # patch loop below has no active patches to iterate.
                if override_export is None:
                    override_export = _capture_override_export(diffuse_mesh, n_faces)
        
        # Direct component
        active_indices = np.where(active_mask)[0]
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            cumulative_dni_patch = cumulative_dni_per_patch[patch_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    cumulative_direct += np.nan_to_num(direct_vals, nan=0.0) * cumulative_dni_patch
                if override_export is None:
                    override_export = _capture_override_export(irradiance_mesh, n_faces)

            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0):
                print(f"  Patch {i+1}/{len(active_indices)} ({100*(i+1)/len(active_indices):.1f}%)")
        
        cumulative_global = cumulative_direct + cumulative_diffuse
    
    else:
        # Per-timestep
        for t_idx, (timestamp, row) in enumerate(df_period_utc.iterrows()):
            dni = float(row['DNI']) * direct_normal_irradiance_scaling
            dhi = float(row['DHI']) * diffuse_irradiance_scaling
            
            elevation = float(solar_positions.loc[timestamp, 'elevation'])
            azimuth = float(solar_positions.loc[timestamp, 'azimuth'])
            
            if elevation <= 0 or (dni <= 0 and dhi <= 0):
                continue
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=azimuth,
                elevation_degrees=elevation,
                direct_normal_irradiance=dni,
                diffuse_irradiance=dhi,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and hasattr(irradiance_mesh, 'metadata'):
                if 'direct' in irradiance_mesh.metadata:
                    cumulative_direct += np.nan_to_num(irradiance_mesh.metadata['direct'], nan=0.0) * time_step_hours
                if 'diffuse' in irradiance_mesh.metadata:
                    cumulative_diffuse += np.nan_to_num(irradiance_mesh.metadata['diffuse'], nan=0.0) * time_step_hours
                if 'global' in irradiance_mesh.metadata:
                    cumulative_global += np.nan_to_num(irradiance_mesh.metadata['global'], nan=0.0) * time_step_hours
                if override_export is None:
                    override_export = _capture_override_export(irradiance_mesh, n_faces)

            if progress_report and (t_idx + 1) % max(1, n_timesteps // 10) == 0:
                print(f"  Processed {t_idx + 1}/{n_timesteps} ({100*(t_idx+1)/n_timesteps:.1f}%)")
    
    # Apply boundary handling
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = _scene_grid_bounds(voxel_data.shape, meshsize)
    boundary_epsilon = meshsize * 0.05
    
    mesh_face_centers = result_mesh.triangles_center
    mesh_face_normals = result_mesh.face_normals
    
    is_boundary_vertical = compute_boundary_vertical_mask(
        mesh_face_centers, mesh_face_normals, grid_bounds_real, boundary_epsilon
    )
    
    cumulative_direct[is_boundary_vertical] = np.nan
    cumulative_diffuse[is_boundary_vertical] = np.nan
    cumulative_global[is_boundary_vertical] = np.nan
    
    # Apply computation mask
    if computation_mask is not None:
        cumulative_direct = apply_computation_mask_to_faces(
            cumulative_direct, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        cumulative_diffuse = apply_computation_mask_to_faces(
            cumulative_diffuse, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
        cumulative_global = apply_computation_mask_to_faces(
            cumulative_global, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )

    if target_face_mask is not None:
        cumulative_direct = np.where(target_face_mask, cumulative_direct, np.nan)
        cumulative_diffuse = np.where(target_face_mask, cumulative_diffuse, np.nan)
        cumulative_global = np.where(target_face_mask, cumulative_global, np.nan)
    
    # Store results
    result_mesh.metadata = getattr(result_mesh, 'metadata', {})
    result_mesh.metadata['cumulative_direct'] = cumulative_direct
    result_mesh.metadata['cumulative_diffuse'] = cumulative_diffuse
    result_mesh.metadata['cumulative_global'] = cumulative_global
    result_mesh.metadata['direct'] = cumulative_direct
    result_mesh.metadata['diffuse'] = cumulative_diffuse
    result_mesh.metadata['global'] = cumulative_global
    if face_svf is not None:
        result_mesh.metadata['svf'] = face_svf
    if override_export is not None:
        result_mesh.metadata.update(override_export)
    else:
        _drop_override_export(result_mesh.metadata)

    return result_mesh


def get_building_sunlight_hours(
    voxcity,
    building_svf_mesh=None,
    mode: str = 'PSH',
    epw_file_path: str = None,
    download_nearest_epw: bool = False,
    dni_threshold: float = 120.0,
    lon: float = None,
    lat: float = None,
    tz: float = None,
    **kwargs
):
    """
    GPU-accelerated sunlight hours computation for building surfaces.
    
    Supports PSH (Probable Sunlight Hours) and DSH (Direct Sun Hours) modes.
    
    **DSH mode** does NOT require an EPW file. Location is automatically
    extracted from the VoxCity object and timezone is inferred.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Trimesh object with building surfaces (optional)
        mode: 'PSH' or 'DSH'
        epw_file_path: Path to EPW file (required for PSH, optional for DSH)
        download_nearest_epw: If True, download nearest EPW
        dni_threshold: DNI threshold for PSH mode (default: 120.0 W/m²)
        lon: Longitude in degrees (optional, extracted from voxcity if not provided)
        lat: Latitude in degrees (optional, extracted from voxcity if not provided)
        tz: Timezone offset in hours (optional, inferred from location if not provided)
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with sunlight hours in metadata
    """
    from datetime import datetime
    import pytz
    
    mode = mode.upper()
    if mode not in ('PSH', 'DSH'):
        raise ValueError(f"mode must be 'PSH' or 'DSH', got '{mode}'")
    
    kwargs = dict(kwargs)
    period_start = kwargs.pop('period_start', '01-01 00:00:00')
    period_end = kwargs.pop('period_end', '12-31 23:59:59')
    daily_start_hour = kwargs.pop('daily_start_hour', None)
    daily_end_hour = kwargs.pop('daily_end_hour', None)
    time_step_hours = float(kwargs.pop('time_step_hours', 1.0))
    progress_report = kwargs.pop('progress_report', False)
    computation_mask = kwargs.pop('computation_mask', None)
    min_elevation = float(kwargs.pop('min_elevation', 0.0))
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    
    # Load data depending on mode
    if mode == 'DSH' and epw_file_path is None and not download_nearest_epw:
        # DSH mode without EPW: derive location and timezone from voxcity
        if lat is None or lon is None:
            _lat, _lon = get_location_from_voxcity(voxcity)
            if lat is None:
                lat = _lat
            if lon is None:
                lon = _lon
        if tz is None:
            tz = get_timezone_offset_from_location(lon, lat)
        
        # Generate synthetic annual hourly timestamps
        weather_df = generate_annual_hourly_dataframe()
    else:
        # PSH mode or DSH with EPW provided
        weather_df, lon_epw, lat_epw, tz_epw = load_epw_data(
            epw_file_path=epw_file_path,
            download_nearest_epw=download_nearest_epw,
            voxcity=voxcity,
            **kwargs
        )
        if lon is None:
            lon = lon_epw
        if lat is None:
            lat = lat_epw
        if tz is None:
            tz = tz_epw
        
        if mode == 'PSH' and 'DNI' not in weather_df.columns:
            raise ValueError("Weather dataframe must have 'DNI' column for PSH mode.")
    
    # Filter dataframe
    df_period_utc = filter_df_to_period(weather_df, period_start, period_end, tz,
                                        daily_start_hour=daily_start_hour,
                                        daily_end_hour=daily_end_hour)
    
    # Get solar positions
    solar_positions = get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Create building mesh if needed
    if building_svf_mesh is None:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            voxel_data = voxcity.voxels.classes
            meshsize = voxcity.voxels.meta.meshsize
            building_id_grid = voxcity.buildings.ids
            building_class_id = kwargs.pop('building_class_id', BUILDING_SURFACE_CLASSES)
            building_svf_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
        except ImportError:
            raise ImportError("VoxCity geoprocessor.mesh required for mesh creation")
    
    result_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
    n_faces = len(result_mesh.faces) if hasattr(result_mesh, 'faces') else 0
    
    if n_faces == 0:
        raise ValueError("Building mesh has no faces")
    
    sunlight_hours = np.zeros(n_faces, dtype=np.float64)
    potential_hours = 0.0
    # Filled from the first per-timestep result that carries it, and attached
    # to result_mesh at the end -- see _capture_override_export. Stays None on
    # the no-sunshine early return below, which runs no timestep at all.
    override_export = None

    elevation_arr = solar_positions['elevation'].to_numpy()
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    n_timesteps = len(elevation_arr)
    
    if mode == 'PSH':
        dni_arr = df_period_utc['DNI'].to_numpy()
    
    # Select sunshine timesteps
    sunshine_timesteps = []
    for t_idx in range(n_timesteps):
        elev = elevation_arr[t_idx]
        
        if mode == 'PSH':
            dni = dni_arr[t_idx]
            if elev > 0 and dni >= dni_threshold:
                sunshine_timesteps.append(t_idx)
                potential_hours += time_step_hours
        else:
            if elev > min_elevation:
                sunshine_timesteps.append(t_idx)
                potential_hours += time_step_hours
    
    n_sunshine = len(sunshine_timesteps)
    
    if progress_report:
        print(f"  Mode: {mode}, Sunshine timesteps: {n_sunshine}, Potential hours: {potential_hours:.1f}")
    
    if n_sunshine == 0:
        result_mesh.metadata = getattr(result_mesh, 'metadata', {})
        result_mesh.metadata['sunlight_hours'] = sunlight_hours
        result_mesh.metadata['potential_sunlight_hours'] = potential_hours
        result_mesh.metadata['sunlight_fraction'] = np.zeros(n_faces, dtype=np.float64)
        result_mesh.metadata['mode'] = mode
        return result_mesh
    
    if use_sky_patches:
        from ..sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', 4)
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', 36)
            n_el = kwargs.get('sky_n_elevation', 9)
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        else:
            n_patches_fib = kwargs.get('sky_n_patches', 145)
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches_fib)
        
        n_patches_sky = len(patches)
        hours_per_patch = np.zeros(n_patches_sky, dtype=np.float64)
        
        for t_idx in sunshine_timesteps:
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if 0 <= patch_idx < n_patches_sky:
                hours_per_patch[patch_idx] += time_step_hours
        
        active_mask = hours_per_patch > 0
        active_indices = np.where(active_mask)[0]
        
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            patch_hours = hours_per_patch[patch_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    receives_sun = np.nan_to_num(direct_vals, nan=0.0) > 0.0
                    sunlight_hours += receives_sun.astype(np.float64) * patch_hours
                if override_export is None:
                    override_export = _capture_override_export(irradiance_mesh, n_faces)

            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0):
                print(f"  Patch {i+1}/{len(active_indices)} ({100*(i+1)/len(active_indices):.1f}%)")
    else:
        for i, t_idx in enumerate(sunshine_timesteps):
            elev = elevation_arr[t_idx]
            az = azimuth_arr[t_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az,
                elevation_degrees=elev,
                direct_normal_irradiance=1.0,
                diffuse_irradiance=0.0,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and 'direct' in irradiance_mesh.metadata:
                direct_vals = irradiance_mesh.metadata['direct']
                if len(direct_vals) == n_faces:
                    receives_sun = np.nan_to_num(direct_vals, nan=0.0) > 0.0
                    sunlight_hours += receives_sun.astype(np.float64) * time_step_hours
                if override_export is None:
                    override_export = _capture_override_export(irradiance_mesh, n_faces)

            if progress_report and ((i + 1) % max(1, n_sunshine // 10) == 0):
                print(f"  Processed {i+1}/{n_sunshine} ({100*(i+1)/n_sunshine:.1f}%)")
    
    # Apply boundary handling
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = _scene_grid_bounds(voxel_data.shape, meshsize)
    
    mesh_face_centers = result_mesh.triangles_center
    mesh_face_normals = result_mesh.face_normals
    
    is_boundary_vertical = compute_boundary_vertical_mask(
        mesh_face_centers, mesh_face_normals, grid_bounds_real, meshsize * 0.05
    )
    sunlight_hours[is_boundary_vertical] = np.nan
    
    if computation_mask is not None:
        sunlight_hours = apply_computation_mask_to_faces(
            sunlight_hours, mesh_face_centers, computation_mask, meshsize, (ny_vc, nx_vc)
        )
    
    sunlight_fraction = sunlight_hours / potential_hours if potential_hours > 0 else np.zeros(n_faces)
    
    result_mesh.metadata = getattr(result_mesh, 'metadata', {})
    result_mesh.metadata['sunlight_hours'] = sunlight_hours
    result_mesh.metadata['potential_sunlight_hours'] = potential_hours
    result_mesh.metadata['sunlight_fraction'] = sunlight_fraction
    result_mesh.metadata['mode'] = mode
    if mode == 'PSH':
        result_mesh.metadata['dni_threshold'] = dni_threshold
    else:
        result_mesh.metadata['min_elevation'] = min_elevation
    if override_export is not None:
        result_mesh.metadata.update(override_export)
    else:
        _drop_override_export(result_mesh.metadata)

    return result_mesh


def get_building_global_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    building_svf_mesh=None,
    **kwargs
):
    """
    GPU-accelerated building surface irradiance using EPW weather data.
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        building_svf_mesh: Pre-computed building mesh (optional)
        **kwargs: Additional parameters
    
    Returns:
        Trimesh object with irradiance values in metadata
    """
    from datetime import datetime
    import pytz
    
    progress_report = kwargs.get('progress_report', False)
    kwargs = dict(kwargs)
    kwargs.pop('progress_report', None)
    
    # Load EPW data
    df, lon, lat, tz = load_epw_data(
        epw_file_path=kwargs.pop('epw_file_path', None),
        download_nearest_epw=kwargs.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs
    )
    
    # Create building mesh if needed
    if building_svf_mesh is None:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            building_class_id = kwargs.get('building_class_id', BUILDING_SURFACE_CLASSES)
            voxel_data = voxcity.voxels.classes
            meshsize = voxcity.voxels.meta.meshsize
            building_id_grid = voxcity.buildings.ids

            building_svf_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
        except ImportError:
            pass
    
    if calc_type == 'instantaneous':
        calc_time = kwargs.get('calc_time', '01-01 12:00:00')
        try:
            calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
        
        df_period = df[
            (df.index.month == calc_dt.month) &
            (df.index.day == calc_dt.day) &
            (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")
        
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)
        
        solar_positions = get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
        elevation_degrees = float(solar_positions.iloc[0]['elevation'])
        
        return get_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            azimuth_degrees_ori=azimuth_degrees,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=DNI,
            diffuse_irradiance=DHI,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
        period_start = kwargs.get('period_start', '01-01 00:00:00')
        period_end = kwargs.get('period_end', '12-31 23:59:59')
        time_step_hours = float(kwargs.get('time_step_hours', 1.0))
        
        kwargs.pop('period_start', None)
        kwargs.pop('period_end', None)
        kwargs.pop('time_step_hours', None)
        
        return get_cumulative_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            weather_df=df,
            lon=lon,
            lat=lat,
            tz=tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")
