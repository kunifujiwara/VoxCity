"""Functions for computing and visualizing various view indices in a voxel city model.

This module provides functionality to compute and visualize:
- Green View Index (GVI): Measures visibility of green elements like trees and vegetation
- Sky View Index (SVI): Measures visibility of open sky from street level 
- Sky View Factor (SVF): Measures the ratio of visible sky hemisphere to total hemisphere
- Landmark Visibility: Measures visibility of specified landmark buildings from different locations

The module uses optimized ray tracing techniques with Numba JIT compilation for efficient computation.
Key features:
- Generic ray tracing framework that can be customized for different view indices
- Parallel processing for fast computation of view maps
- Tree transmittance modeling using Beer-Lambert law
- Visualization tools including matplotlib plots and OBJ exports
- Support for both inclusion and exclusion based visibility checks

The module provides several key functions:
- trace_ray_generic(): Core ray tracing function that handles tree transmittance
- compute_vi_generic(): Computes view indices by casting rays in specified directions
- compute_vi_map_generic(): Generates 2D maps of view indices
- get_view_index(): High-level function to compute various view indices
- compute_landmark_visibility(): Computes visibility of landmark buildings
- get_sky_view_factor_map(): Computes sky view factor maps

The module uses a voxel-based representation where:
- Empty space is represented by 0
- Trees are represented by -2 
- Buildings are represented by -3
- Other values can be used for different features

Tree transmittance is modeled using the Beer-Lambert law with configurable parameters:
- tree_k: Static extinction coefficient (default 0.6)
- tree_lad: Leaf area density in m^-1 (default 1.0)

Additional implementation details:
- Uses DDA (Digital Differential Analyzer) algorithm for efficient ray traversal
- Handles edge cases like zero-length rays and division by zero
- Supports early exit optimizations for performance
- Provides flexible observer placement rules
- Includes comprehensive error checking and validation
- Allows customization of visualization parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange
import time
import trimesh
import math

from ..geoprocessor.polygon import find_building_containing_point, get_buildings_in_drawn_polygon
from ..geoprocessor.mesh import create_voxel_mesh
from ..exporter.obj import grid_to_obj, export_obj


def _generate_ray_directions_grid(N_azimuth: int,
                                  N_elevation: int,
                                  elevation_min_degrees: float,
                                  elevation_max_degrees: float) -> np.ndarray:
    """Generate ray directions using azimuth/elevation grid sampling.

    Elevation is measured from the horizontal plane: 0 deg at horizon, +90 at zenith.
    """
    azimuth_angles = np.linspace(0.0, 2.0 * np.pi, int(N_azimuth), endpoint=False)
    elevation_angles = np.deg2rad(
        np.linspace(float(elevation_min_degrees), float(elevation_max_degrees), int(N_elevation))
    )

    ray_directions = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=np.float64)
    out_idx = 0
    for elevation in elevation_angles:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles:
            dx = cos_elev * np.cos(azimuth)
            dy = cos_elev * np.sin(azimuth)
            dz = sin_elev
            ray_directions[out_idx, 0] = dx
            ray_directions[out_idx, 1] = dy
            ray_directions[out_idx, 2] = dz
            out_idx += 1
    return ray_directions


def _generate_ray_directions_fibonacci(N_rays: int,
                                       elevation_min_degrees: float,
                                       elevation_max_degrees: float) -> np.ndarray:
    """Generate ray directions with near-uniform solid-angle spacing using a Fibonacci lattice.

    Elevation is measured from the horizontal plane. Uniform solid-angle sampling over an
    elevation band [emin, emax] is achieved by sampling z = sin(elev) uniformly over
    [sin(emin), sin(emax)] and using a golden-angle azimuth sequence.
    """
    N = int(max(1, N_rays))
    emin = np.deg2rad(float(elevation_min_degrees))
    emax = np.deg2rad(float(elevation_max_degrees))
    # Map to z-range where z = sin(elevation)
    z_min = np.sin(min(emin, emax))
    z_max = np.sin(max(emin, emax))
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(N, dtype=np.float64)
    # Uniform in z over the band (equal solid angle within the band)
    z = z_min + (i + 0.5) * (z_max - z_min) / N
    # Wrap azimuth via golden-angle progression
    phi = i * golden_angle
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack((x, y, z), axis=1).astype(np.float64)

@njit
def calculate_transmittance(length, tree_k=0.6, tree_lad=1.0):
    """Calculate tree transmittance using the Beer-Lambert law.
    
    Uses the Beer-Lambert law to model light attenuation through tree canopy:
    transmittance = exp(-k * LAD * L)
    where:
    - k is the extinction coefficient
    - LAD is the leaf area density
    - L is the path length through the canopy
    
    Args:
        length (float): Path length through tree voxel in meters
        tree_k (float): Static extinction coefficient (default: 0.6)
            Controls overall light attenuation strength
        tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            Higher values = denser foliage = more attenuation
    
    Returns:
        float: Transmittance value between 0 and 1
            1.0 = fully transparent
            0.0 = fully opaque
    """
    return np.exp(-tree_k * tree_lad * length)

@njit
def trace_ray_generic(voxel_data, origin, direction, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Trace a ray through a voxel grid and check for hits with specified values.
    
    Uses DDA algorithm to efficiently traverse voxels along ray path.
    Handles tree transmittance using Beer-Lambert law.
    
    The DDA algorithm:
    1. Initializes ray at origin voxel
    2. Calculates distances to next voxel boundaries in each direction
    3. Steps to next voxel by choosing smallest distance
    4. Repeats until hit or out of bounds
    
    Tree transmittance:
    - When ray passes through tree voxels (-2), transmittance is accumulated
    - Uses Beer-Lambert law with configurable extinction coefficient and leaf area density
    - Ray is considered blocked if cumulative transmittance falls below 0.01
    
    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (ndarray): Starting point (x,y,z) of ray in voxel coordinates
        direction (ndarray): Direction vector of ray (will be normalized)
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        tuple: (hit_detected, transmittance_value)
            hit_detected (bool): Whether ray hit a target voxel
            transmittance_value (float): Cumulative transmittance through trees
    """
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction

    # Normalize direction vector to ensure consistent step sizes
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return False, 1.0
    dx /= length
    dy /= length
    dz /= length

    # Initialize ray position at center of starting voxel
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    # Determine step direction for each axis (-1 or +1)
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Calculate DDA parameters with safety checks to prevent division by zero
    EPSILON = 1e-10  # Small value to prevent division by zero
    
    # Calculate distances to next voxel boundaries and step sizes for X-axis
    if abs(dx) > EPSILON:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1 / dx)
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    # Calculate distances to next voxel boundaries and step sizes for Y-axis
    if abs(dy) > EPSILON:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1 / dy)
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    # Calculate distances to next voxel boundaries and step sizes for Z-axis
    if abs(dz) > EPSILON:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1 / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf

    # Track cumulative values for tree transmittance calculation
    cumulative_transmittance = 1.0
    cumulative_hit_contribution = 0.0
    last_t = 0.0

    # Main ray traversal loop using DDA algorithm
    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        voxel_value = voxel_data[i, j, k]
        
        # Find next intersection point along the ray
        t_next = min(t_max_x, t_max_y, t_max_z)
        
        # Calculate segment length in current voxel (in real world units)
        segment_length = (t_next - last_t) * meshsize
        segment_length = max(0.0, segment_length) 
        
        # Handle tree voxels (value -2) with Beer-Lambert law transmittance
        if voxel_value == -2:
            transmittance = calculate_transmittance(segment_length, tree_k, tree_lad)
            cumulative_transmittance *= transmittance

            # If transmittance becomes too low, consider the ray blocked.
            # In exclusion mode (e.g., sky view), a blocked ray counts as a hit (obstruction).
            # In inclusion mode (e.g., building view), trees should NOT count as a target hit;
            # we terminate traversal early but report no hit so callers can treat it as 0 visibility.
            if cumulative_transmittance < 0.01:
                if inclusion_mode:
                    return False, cumulative_transmittance
                else:
                    return True, cumulative_transmittance

        # Check for hits with target objects based on inclusion/exclusion mode
        if inclusion_mode:
            # Inclusion mode: hit if voxel value is in the target set
            for hv in hit_values:
                if voxel_value == hv:
                    return True, cumulative_transmittance
            # Opaque blockers (anything non-air, non-tree, and not a target) stop visibility
            if voxel_value != 0 and voxel_value != -2:
                return False, cumulative_transmittance
        else:
            # Exclusion mode: hit if voxel value is NOT in the allowed set
            in_set = False
            for hv in hit_values:
                if voxel_value == hv:
                    in_set = True
                    break
            if not in_set and voxel_value != -2:  # Exclude trees from regular hits
                return True, cumulative_transmittance

        # Update for next iteration
        last_t = t_next

        # Tie-aware DDA stepping to reduce corner leaks
        TIE_EPS = 1e-12
        eq_x = abs(t_max_x - t_next) <= TIE_EPS
        eq_y = abs(t_max_y - t_next) <= TIE_EPS
        eq_z = abs(t_max_z - t_next) <= TIE_EPS

        # Conservative occlusion at exact grid corner crossings in inclusion mode
        if inclusion_mode and ((eq_x and eq_y) or (eq_x and eq_z) or (eq_y and eq_z)):
            # Probe neighbor cells we are about to enter on tied axes; if any is opaque non-target, block
            # Note: bounds checks guard against out-of-grid probes
            if eq_x:
                ii = i + step_x
                if 0 <= ii < nx:
                    val = voxel_data[ii, j, k]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance
            if eq_y:
                jj = j + step_y
                if 0 <= jj < ny:
                    val = voxel_data[i, jj, k]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance
            if eq_z:
                kk = k + step_z
                if 0 <= kk < nz:
                    val = voxel_data[i, j, kk]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance

        # Step along all axes that hit at t_next (handles ties robustly)
        stepped = False
        if eq_x:
            t_max_x += t_delta_x
            i += step_x
            stepped = True
        if eq_y:
            t_max_y += t_delta_y
            j += step_y
            stepped = True
        if eq_z:
            t_max_z += t_delta_z
            k += step_z
            stepped = True

        if not stepped:
            # Fallback: should not happen, but keep classic ordering
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

    # Ray exited the grid without hitting a target
    return False, cumulative_transmittance

@njit
def compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Compute view index accounting for tree transmittance.
    
    Casts rays in specified directions and computes visibility index based on hits and transmittance.
    The view index is the ratio of visible rays to total rays cast, where:
    - For inclusion mode: Counts hits with target values
    - For exclusion mode: Counts rays that don't hit obstacles
    Tree transmittance is handled specially:
    - In inclusion mode with trees as targets: Uses (1 - transmittance) as contribution
    - In exclusion mode: Uses transmittance value directly
    
    Args:
        observer_location (ndarray): Observer position (x,y,z) in voxel coordinates
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        float: View index value between 0 and 1
            0.0 = no visibility in any direction
            1.0 = full visibility in all directions
    """
    total_rays = ray_directions.shape[0]
    visibility_sum = 0.0

    # Cast rays in all specified directions
    for idx in range(total_rays):
        direction = ray_directions[idx]
        hit, value = trace_ray_generic(voxel_data, observer_location, direction, 
                                     hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
        
        # Accumulate visibility contributions based on mode
        if inclusion_mode:
            if hit:
                # For trees in hit_values, use partial visibility based on transmittance (Beer-Lambert)
                if -2 in hit_values:
                    # value is cumulative transmittance (0..1).
                    # Contribution should be 1 - transmittance.
                    contrib = 1.0 - max(0.0, min(1.0, value))
                    visibility_sum += contrib
                else:
                    # Full visibility for non-tree targets
                    visibility_sum += 1.0
        else:
            if not hit:
                # For exclusion mode, use transmittance value directly as visibility
                visibility_sum += value

    # Return average visibility across all rays
    return visibility_sum / total_rays

@njit(parallel=True)
def compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, 
                          meshsize, tree_k, tree_lad, inclusion_mode=True):
    """Compute view index map incorporating tree transmittance.
    
    Places observers at valid locations and computes view index for each position.
    Valid observer locations are:
    - Empty voxels (0) or tree voxels (-2)
    - Above non-empty, non-tree voxels
    - Not above water (7,8,9) or negative values
    
    The function processes each x,y position in parallel for efficiency.
    
    Args:
        voxel_data (ndarray): 3D array of voxel values
        ray_directions (ndarray): Array of direction vectors for rays
        view_height_voxel (int): Observer height in voxel units
        hit_values (tuple): Values to check for hits
        meshsize (float): Size of each voxel in meters
        tree_k (float): Tree extinction coefficient
        tree_lad (float): Leaf area density in m^-1
        inclusion_mode (bool): If True, hit_values are hits. If False, hit_values are allowed values.
    
    Returns:
        ndarray: 2D array of view index values
            NaN = invalid observer location
            0.0-1.0 = view index value
    """
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)

    # Process each horizontal position in parallel for efficiency
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Search from bottom to top for valid observer placement
            for z in range(1, nz):
                # Check for valid observer location: empty space above solid ground
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    # Skip invalid ground types (water or negative values)
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        vi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer at specified height above ground level
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        # Compute view index for this location
                        vi_value = compute_vi_generic(observer_location, voxel_data, ray_directions, 
                                                    hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
                        vi_map[x, y] = vi_value
                        found_observer = True
                        break
            # Mark locations where no valid observer position was found
            if not found_observer:
                vi_map[x, y] = np.nan

    # Flip vertically to match display orientation
    return np.flipud(vi_map)

# ==========================
# Fast-path helpers (mask-based)
# ==========================

def _prepare_masks_for_vi(voxel_data: np.ndarray, hit_values, inclusion_mode: bool):
    """Precompute boolean masks to avoid expensive value checks inside Numba loops.

    Returns a tuple (is_tree, is_target, is_allowed, is_blocker_inc), where some entries
    may be None depending on mode.
    """
    is_tree = (voxel_data == -2)
    if inclusion_mode:
        is_target = np.isin(voxel_data, hit_values)
        is_blocker_inc = (voxel_data != 0) & (~is_tree) & (~is_target)
        return is_tree, is_target, None, is_blocker_inc
    else:
        is_allowed = np.isin(voxel_data, hit_values)
        return is_tree, None, is_allowed, None


@njit(cache=True, fastmath=True)
def _trace_ray_inclusion_masks(is_tree, is_target, is_blocker_inc,
                               origin, direction,
                               meshsize, tree_k, tree_lad):
    """DDA traversal using precomputed masks for inclusion mode.

    Returns (hit, cumulative_transmittance).
    Tree transmittance uses Beer-Lambert with LAD and segment length in meters.
    """
    nx, ny, nz = is_tree.shape

    x0, y0, z0 = origin
    dx, dy, dz = direction

    # Normalize
    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return False, 1.0
    dx /= length; dy /= length; dz /= length

    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    EPS = 1e-10
    if abs(dx) > EPS:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = np.inf; t_delta_x = np.inf
    if abs(dy) > EPS:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = np.inf; t_delta_y = np.inf
    if abs(dz) > EPS:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = np.inf; t_delta_z = np.inf

    cumulative_transmittance = 1.0
    last_t = 0.0

    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        t_next = t_max_x
        axis = 0
        if t_max_y < t_next:
            t_next = t_max_y; axis = 1
        if t_max_z < t_next:
            t_next = t_max_z; axis = 2

        segment_length = (t_next - last_t) * meshsize
        if segment_length < 0.0:
            segment_length = 0.0

        # Tree attenuation
        if is_tree[i, j, k]:
            # Beer-Lambert law over segment length
            trans = np.exp(-tree_k * tree_lad * segment_length)
            cumulative_transmittance *= trans
            if cumulative_transmittance < 1e-2:
                # Trees do not count as target here; early exit as blocked but no hit for inclusion mode
                return False, cumulative_transmittance

        # Inclusion: hit if voxel is in target set
        if is_target[i, j, k]:
            return True, cumulative_transmittance

        # Opaque blockers stop visibility
        if is_blocker_inc[i, j, k]:
            return False, cumulative_transmittance

        # advance
        last_t = t_next
        if axis == 0:
            t_max_x += t_delta_x; i += step_x
        elif axis == 1:
            t_max_y += t_delta_y; j += step_y
        else:
            t_max_z += t_delta_z; k += step_z

    return False, cumulative_transmittance


@njit(cache=True, fastmath=True)
def _trace_ray_exclusion_masks(is_tree, is_allowed,
                               origin, direction,
                               meshsize, tree_k, tree_lad):
    """DDA traversal using precomputed masks for exclusion mode.

    Returns (hit_blocker, cumulative_transmittance).
    For exclusion, a hit means obstruction (voxel not in allowed set and not a tree).
    """
    nx, ny, nz = is_tree.shape

    x0, y0, z0 = origin
    dx, dy, dz = direction

    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return False, 1.0
    dx /= length; dy /= length; dz /= length

    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    EPS = 1e-10
    if abs(dx) > EPS:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = np.inf; t_delta_x = np.inf
    if abs(dy) > EPS:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = np.inf; t_delta_y = np.inf
    if abs(dz) > EPS:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = np.inf; t_delta_z = np.inf

    cumulative_transmittance = 1.0
    last_t = 0.0

    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        t_next = t_max_x
        axis = 0
        if t_max_y < t_next:
            t_next = t_max_y; axis = 1
        if t_max_z < t_next:
            t_next = t_max_z; axis = 2

        segment_length = (t_next - last_t) * meshsize
        if segment_length < 0.0:
            segment_length = 0.0

        # Tree attenuation
        if is_tree[i, j, k]:
            trans = np.exp(-tree_k * tree_lad * segment_length)
            cumulative_transmittance *= trans
            # In exclusion, a tree alone never counts as obstruction; but we can early exit
            if cumulative_transmittance < 1e-2:
                return True, cumulative_transmittance

        # Obstruction if voxel is not allowed and not a tree
        if (not is_allowed[i, j, k]) and (not is_tree[i, j, k]):
            return True, cumulative_transmittance

        last_t = t_next
        if axis == 0:
            t_max_x += t_delta_x; i += step_x
        elif axis == 1:
            t_max_y += t_delta_y; j += step_y
        else:
            t_max_z += t_delta_z; k += step_z

    return False, cumulative_transmittance


@njit(parallel=True, cache=True, fastmath=True)
def _compute_vi_map_generic_fast(voxel_data, ray_directions, view_height_voxel,
                                 meshsize, tree_k, tree_lad,
                                 is_tree, is_target, is_allowed, is_blocker_inc,
                                 inclusion_mode, trees_in_targets):
    """Fast mask-based computation of VI map.

    trees_in_targets indicates whether to use partial contribution 1 - T for inclusion mode.
    """
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)

    # Precompute observer z for each (x,y): returns -1 if invalid, else z index base
    obs_base_z = _precompute_observer_base_z(voxel_data)

    for x in prange(nx):
        for y in range(ny):
            base_z = obs_base_z[x, y]
            if base_z < 0:
                vi_map[x, y] = np.nan
                continue

            # Skip invalid ground: water or negative
            below = voxel_data[x, y, base_z]
            if (below == 7) or (below == 8) or (below == 9) or (below < 0):
                vi_map[x, y] = np.nan
                continue

            oz = base_z + 1 + view_height_voxel
            obs = np.array([x, y, oz], dtype=np.float64)

            visibility_sum = 0.0
            n_rays = ray_directions.shape[0]
            for r in range(n_rays):
                direction = ray_directions[r]
                if inclusion_mode:
                    hit, value = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker_inc,
                                                            obs, direction,
                                                            meshsize, tree_k, tree_lad)
                    if hit:
                        if trees_in_targets:
                            contrib = 1.0 - max(0.0, min(1.0, value))
                            visibility_sum += contrib
                        else:
                            visibility_sum += 1.0
                else:
                    hit, value = _trace_ray_exclusion_masks(is_tree, is_allowed,
                                                            obs, direction,
                                                            meshsize, tree_k, tree_lad)
                    if not hit:
                        visibility_sum += value

            vi_map[x, y] = visibility_sum / n_rays

    return np.flipud(vi_map)


@njit(cache=True, fastmath=True)
def _precompute_observer_base_z(voxel_data):
    """For each (x,y), find the highest z such that z+1 is empty/tree and z is solid (non-empty & non-tree).
    Returns int32 array of shape (nx,ny) with z or -1 if none.
    """
    nx, ny, nz = voxel_data.shape
    out = np.empty((nx, ny), dtype=np.int32)
    for x in range(nx):
        for y in range(ny):
            found = False
            for z in range(1, nz):
                v_above = voxel_data[x, y, z]
                v_base = voxel_data[x, y, z - 1]
                if (v_above == 0 or v_above == -2) and not (v_base == 0 or v_base == -2):
                    out[x, y] = z - 1
                    found = True
                    break
            if not found:
                out[x, y] = -1
    return out


def get_view_index(voxel_data, meshsize, mode=None, hit_values=None, inclusion_mode=True, fast_path=True, **kwargs):
    """Calculate and visualize a generic view index for a voxel city model.

    This is a high-level function that provides a flexible interface for computing
    various view indices. It handles:
    - Mode presets for common indices (green, sky)
    - Ray direction generation
    - Tree transmittance parameters
    - Visualization
    - Optional OBJ export

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        mode (str): Predefined mode. Options: 'green', 'sky', or None.
            If 'green': GVI mode - measures visibility of vegetation
            If 'sky': SVI mode - measures visibility of open sky
            If None: Custom mode requiring hit_values parameter
        hit_values (tuple): Voxel values considered as hits (if inclusion_mode=True)
                            or allowed values (if inclusion_mode=False), if mode is None.
        inclusion_mode (bool): 
            True = voxel_value in hit_values is success.
            False = voxel_value not in hit_values is success.
        **kwargs: Additional arguments:
            - view_point_height (float): Observer height in meters (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'viridis')
            - obj_export (bool): Export as OBJ (default: False)
            - output_directory (str): Directory for OBJ output
            - output_file_name (str): Base filename for OBJ output
            - num_colors (int): Number of discrete colors for OBJ export
            - alpha (float): Transparency value for OBJ export
            - vmin (float): Minimum value for color mapping
            - vmax (float): Maximum value for color mapping
            - N_azimuth (int): Number of azimuth angles for ray directions
            - N_elevation (int): Number of elevation angles for ray directions
            - elevation_min_degrees (float): Minimum elevation angle in degrees
            - elevation_max_degrees (float): Maximum elevation angle in degrees
            - tree_k (float): Tree extinction coefficient (default: 0.5)
            - tree_lad (float): Leaf area density in m^-1 (default: 1.0)

    Returns:
        ndarray: 2D array of computed view index values.
    """
    # Handle predefined mode presets for common view indices
    if mode == 'green':
        # GVI defaults - detect vegetation and trees
        hit_values = (-2, 2, 5, 6, 7, 8)
        inclusion_mode = True
    elif mode == 'sky':
        # SVI defaults - detect open sky
        hit_values = (0,)
        inclusion_mode = False
    else:
        # For custom mode, user must specify hit_values
        if hit_values is None:
            raise ValueError("For custom mode, you must provide hit_values.")

    # Extract parameters from kwargs with sensible defaults
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'viridis')
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    
    # Ray casting parameters for hemisphere sampling
    N_azimuth = kwargs.get("N_azimuth", 60)
    N_elevation = kwargs.get("N_elevation", 10)
    elevation_min_degrees = kwargs.get("elevation_min_degrees", -30)
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 30)
    ray_sampling = kwargs.get("ray_sampling", "grid")  # 'grid' or 'fibonacci'
    N_rays = kwargs.get("N_rays", N_azimuth * N_elevation)
    
    # Tree transmittance parameters for Beer-Lambert law
    tree_k = kwargs.get("tree_k", 0.5)
    tree_lad = kwargs.get("tree_lad", 1.0)

    # Generate ray directions
    if str(ray_sampling).lower() == "fibonacci":
        ray_directions = _generate_ray_directions_fibonacci(
            int(N_rays), elevation_min_degrees, elevation_max_degrees
        )
    else:
        ray_directions = _generate_ray_directions_grid(
            int(N_azimuth), int(N_elevation), elevation_min_degrees, elevation_max_degrees
        )

    # Optional: configure numba threads
    num_threads = kwargs.get("num_threads", None)
    if num_threads is not None:
        try:
            from numba import set_num_threads
            set_num_threads(int(num_threads))
        except Exception:
            pass

    # Compute the view index map with transmittance parameters
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
            vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, 
                                           hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
    else:
        vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, 
                                       hit_values, meshsize, tree_k, tree_lad, inclusion_mode)

    # Create visualization with custom colormap handling
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgray')  # Color for NaN values (invalid locations)
    plt.figure(figsize=(10, 8))
    plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='View Index')
    plt.axis('off')
    plt.show()

    # Optional OBJ export for 3D visualization
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(vi_map))
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

def mark_building_by_id(voxcity_grid_ori, building_id_grid_ori, ids, mark):
    """Mark specific buildings in the voxel grid with a given value.

    This function is used to identify landmark buildings for visibility analysis
    by replacing their voxel values with a special marker value. It handles
    coordinate system alignment between the building ID grid and voxel grid.

    Args:
        voxcity_grid_ori (ndarray): 3D array of voxel values (original, will be copied)
        building_id_grid_ori (ndarray): 2D array of building IDs (original, will be copied)
        ids (list): List of building IDs to mark as landmarks
        mark (int): Value to mark the landmark buildings with (typically negative)

    Returns:
        ndarray: Modified 3D voxel grid with landmark buildings marked
    """
    # Create working copies to avoid modifying original data
    voxcity_grid = voxcity_grid_ori.copy()

    # Flip building ID grid vertically to match voxel grid orientation
    # This accounts for different coordinate system conventions
    building_id_grid = np.flipud(building_id_grid_ori.copy())

    # Find x,y positions where target building IDs are located
    positions = np.where(np.isin(building_id_grid, ids))

    # Process each location containing a target building
    for i in range(len(positions[0])):
        x, y = positions[0][i], positions[1][i]
        # Find all building voxels (-3) at this x,y location and mark them
        z_mask = voxcity_grid[x, y, :] == -3
        voxcity_grid[x, y, z_mask] = mark
    
    return voxcity_grid

@njit
def trace_ray_to_target(voxel_data, origin, target, opaque_values):
    """Trace a ray from origin to target through voxel data.

    Uses DDA algorithm to efficiently traverse voxels along ray path.
    Checks for any opaque voxels blocking the line of sight.

    Args:
        voxel_data (ndarray): 3D array of voxel values
        origin (tuple): Starting point (x,y,z) in voxel coordinates
        target (tuple): End point (x,y,z) in voxel coordinates
        opaque_values (ndarray): Array of voxel values that block the ray

    Returns:
        bool: True if target is visible from origin, False otherwise
    """
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    x1, y1, z1 = target
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    # Normalize direction vector for consistent traversal
    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return True  # Origin and target are at the same location
    dx /= length
    dy /= length
    dz /= length

    # Initialize ray position at center of starting voxel
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    # Determine step direction for each axis
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    # Calculate distances to next voxel boundaries and step sizes
    # Handle cases where direction components are zero to avoid division by zero
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

    # Main ray traversal loop using DDA algorithm
    while True:
        # Check if current voxel is within bounds and contains opaque material
        if (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
            voxel_value = voxel_data[i, j, k]
            if voxel_value in opaque_values:
                return False  # Ray is blocked by opaque voxel
        else:
            return False  # Ray went out of bounds before reaching target

        # Check if we've reached the target voxel
        if i == int(x1) and j == int(y1) and k == int(z1):
            return True  # Ray successfully reached the target

        # Move to next voxel using DDA algorithm
        # Choose the axis with the smallest distance to next boundary
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
    """Check if any landmark is visible from the observer location.

    Traces rays to each landmark position until finding one that's visible.
    Uses optimized ray tracing with early exit on first visible landmark.

    Args:
        observer_location (ndarray): Observer position (x,y,z) in voxel coordinates
        landmark_positions (ndarray): Array of landmark positions (n_landmarks, 3)
        voxel_data (ndarray): 3D array of voxel values
        opaque_values (ndarray): Array of voxel values that block visibility

    Returns:
        int: 1 if any landmark is visible, 0 if none are visible
    """
    # Check visibility to each landmark sequentially
    # Early exit strategy: return 1 as soon as any landmark is visible
    for idx in range(landmark_positions.shape[0]):
        target = landmark_positions[idx].astype(np.float64)
        is_visible = trace_ray_to_target(voxel_data, observer_location, target, opaque_values)
        if is_visible:
            return 1  # Return immediately when first visible landmark is found
    return 0  # No landmarks were visible from this location

@njit(parallel=True)
def compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel):
    """Compute visibility map for landmarks in the voxel grid.

    Places observers at valid locations (empty voxels above ground, excluding building
    roofs and vegetation) and checks visibility to any landmark.

    The function processes each x,y position in parallel for efficiency.
    Valid observer locations are:
    - Empty voxels (0) or tree voxels (-2)
    - Above non-empty, non-tree voxels
    - Not above water (7,8,9) or negative values

    Args:
        voxel_data (ndarray): 3D array of voxel values
        landmark_positions (ndarray): Array of landmark positions (n_landmarks, 3)
        opaque_values (ndarray): Array of voxel values that block visibility
        view_height_voxel (int): Height offset for observer in voxels

    Returns:
        ndarray: 2D array of visibility values
            NaN = invalid observer location
            0 = no landmarks visible
            1 = at least one landmark visible
    """
    nx, ny, nz = voxel_data.shape
    visibility_map = np.full((nx, ny), np.nan)

    # Process each x,y position in parallel for computational efficiency
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            # Find the lowest valid observer location by searching from bottom up
            for z in range(1, nz):
                # Valid observer location: empty voxel above non-empty ground
                if voxel_data[x, y, z] == 0 and voxel_data[x, y, z - 1] != 0:
                    # Skip locations above building roofs or vegetation
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        visibility_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        # Place observer at specified height above ground level
                        observer_location = np.array([x, y, z+view_height_voxel], dtype=np.float64)
                        # Check visibility to any landmark from this location
                        visible = compute_visibility_to_all_landmarks(observer_location, landmark_positions, voxel_data, opaque_values)
                        visibility_map[x, y] = visible
                        found_observer = True
                        break
            # Mark locations where no valid observer position exists
            if not found_observer:
                visibility_map[x, y] = np.nan

    return visibility_map

def compute_landmark_visibility(voxel_data, target_value=-30, view_height_voxel=0, colormap='viridis'):
    """Compute and visualize landmark visibility in a voxel grid.

    Places observers at valid locations and checks visibility to any landmark voxel.
    Generates a binary visibility map and visualization.

    The function:
    1. Identifies all landmark voxels (target_value)
    2. Determines which voxel values block visibility
    3. Computes visibility from each valid observer location
    4. Generates visualization with legend

    Args:
        voxel_data (ndarray): 3D array of voxel values
        target_value (int, optional): Value used to identify landmark voxels. Defaults to -30.
        view_height_voxel (int, optional): Height offset for observer in voxels. Defaults to 0.
        colormap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.

    Returns:
        ndarray: 2D array of visibility values (0 or 1) with y-axis flipped
            NaN = invalid observer location
            0 = no landmarks visible
            1 = at least one landmark visible

    Raises:
        ValueError: If no landmark voxels are found with the specified target_value
    """
    # Find positions of all landmark voxels
    landmark_positions = np.argwhere(voxel_data == target_value)

    if landmark_positions.shape[0] == 0:
        raise ValueError(f"No landmark with value {target_value} found in the voxel data.")

    # Define which voxel values block visibility
    unique_values = np.unique(voxel_data)
    opaque_values = np.array([v for v in unique_values if v != 0 and v != target_value], dtype=np.int32)

    # Compute visibility map
    visibility_map = compute_visibility_map(voxel_data, landmark_positions, opaque_values, view_height_voxel)

    # Set up visualization
    cmap = plt.cm.get_cmap(colormap, 2).copy()
    cmap.set_bad(color='lightgray')

    # Create main plot
    plt.figure(figsize=(10, 8))
    plt.imshow(np.flipud(visibility_map), origin='lower', cmap=cmap, vmin=0, vmax=1)

    # Create and add legend
    visible_patch = mpatches.Patch(color=cmap(1.0), label='Visible (1)')
    not_visible_patch = mpatches.Patch(color=cmap(0.0), label='Not Visible (0)')
    plt.legend(handles=[visible_patch, not_visible_patch], 
            loc='center left',
            bbox_to_anchor=(1.0, 0.5))
    plt.axis('off')
    plt.show()

    return np.flipud(visibility_map)

def get_landmark_visibility_map(voxcity_grid_ori, building_id_grid, building_gdf, meshsize, **kwargs):
    """Generate a visibility map for landmark buildings in a voxel city.

    Places observers at valid locations and checks visibility to any part of the
    specified landmark buildings. Can identify landmarks either by ID or by finding
    buildings within a specified rectangle.

    Args:
        voxcity_grid (ndarray): 3D array representing the voxel city
        building_id_grid (ndarray): 3D array mapping voxels to building IDs
        building_gdf (GeoDataFrame): GeoDataFrame containing building features
        meshsize (float): Size of each voxel in meters
        **kwargs: Additional keyword arguments
            view_point_height (float): Height of observer viewpoint in meters
            colormap (str): Matplotlib colormap name
            landmark_building_ids (list): List of building IDs to mark as landmarks
            rectangle_vertices (list): List of (lat,lon) coordinates defining rectangle
            obj_export (bool): Whether to export visibility map as OBJ file
            dem_grid (ndarray): Digital elevation model grid for OBJ export
            output_directory (str): Directory for OBJ file output
            output_file_name (str): Base filename for OBJ output
            alpha (float): Alpha transparency value for OBJ export
            vmin (float): Minimum value for color mapping
            vmax (float): Maximum value for color mapping

    Returns:
        ndarray: 2D array of visibility values for landmark buildings
    """
    # Convert observer height from meters to voxel units
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)

    colormap = kwargs.get("colormap", 'viridis')

    # Get landmark building IDs either directly or by finding buildings in rectangle
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_gdf, landmark_polygon, operation='within')
        else:
            rectangle_vertices = kwargs.get("rectangle_vertices", None)
            if rectangle_vertices is None:
                print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_ids.")
                return None
                
            # Calculate center point of rectangle
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            target_point = (center_lon, center_lat)
            
            # Find buildings at center point
            landmark_ids = find_building_containing_point(building_gdf, target_point)

    # Mark landmark buildings in voxel grid with special value
    target_value = -30
    voxcity_grid = mark_building_by_id(voxcity_grid_ori, building_id_grid, landmark_ids, target_value)
    
    # Compute visibility map
    landmark_vis_map = compute_landmark_visibility(voxcity_grid, target_value=target_value, view_height_voxel=view_height_voxel, colormap=colormap)

    # Handle optional OBJ export
    obj_export = kwargs.get("obj_export")
    if obj_export == True:
        dem_grid = kwargs.get("dem_grid", np.zeros_like(landmark_vis_map))
        output_dir = kwargs.get("output_directory", "output")
        output_file_name = kwargs.get("output_file_name", "landmark_visibility")        
        num_colors = 2
        alpha = kwargs.get("alpha", 1.0)
        vmin = kwargs.get("vmin", 0.0)
        vmax = kwargs.get("vmax", 1.0)
        
        # Export visibility map and voxel city as OBJ files
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

def get_sky_view_factor_map(voxel_data, meshsize, show_plot=False, **kwargs):
    """
    Compute and visualize the Sky View Factor (SVF) for each valid observer cell in the voxel grid.

    Sky View Factor measures the proportion of the sky hemisphere that is visible from a given point.
    It ranges from 0 (completely obstructed) to 1 (completely open sky). This implementation:
    - Uses hemisphere ray casting to sample sky visibility
    - Accounts for tree transmittance using Beer-Lambert law
    - Places observers at valid street-level locations
    - Provides optional visualization and OBJ export

    Args:
        voxel_data (ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        show_plot (bool): Whether to display the SVF visualization plot.
        **kwargs: Additional parameters including:
            view_point_height (float): Observer height in meters (default: 1.5)
            colormap (str): Matplotlib colormap name (default: 'BuPu_r')
            vmin, vmax (float): Color scale limits (default: 0.0, 1.0)
            N_azimuth (int): Number of azimuth angles for ray sampling (default: 60)
            N_elevation (int): Number of elevation angles for ray sampling (default: 10)
            elevation_min_degrees (float): Minimum elevation angle (default: 0)
            elevation_max_degrees (float): Maximum elevation angle (default: 90)
            tree_k (float): Tree extinction coefficient (default: 0.6)
            tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            obj_export (bool): Whether to export as OBJ file (default: False)

    Returns:
        ndarray: 2D array of SVF values at each valid observer location (x, y).
                 NaN values indicate invalid observer positions.
    """
    # Extract default parameters with sky-specific settings
    view_point_height = kwargs.get("view_point_height", 1.5)
    view_height_voxel = int(view_point_height / meshsize)
    colormap = kwargs.get("colormap", 'BuPu_r')  # Blue-purple colormap suitable for sky
    vmin = kwargs.get("vmin", 0.0)
    vmax = kwargs.get("vmax", 1.0)
    
    # Ray sampling parameters optimized for sky view factor
    N_azimuth = kwargs.get("N_azimuth", 60)      # Full 360-degree azimuth sampling
    N_elevation = kwargs.get("N_elevation", 10)   # Hemisphere elevation sampling
    elevation_min_degrees = kwargs.get("elevation_min_degrees", 0)   # Horizon
    elevation_max_degrees = kwargs.get("elevation_max_degrees", 90)  # Zenith
    ray_sampling = kwargs.get("ray_sampling", "grid")  # 'grid' or 'fibonacci'
    N_rays = kwargs.get("N_rays", N_azimuth * N_elevation)

    # Tree transmittance parameters for Beer-Lambert law
    tree_k = kwargs.get("tree_k", 0.6)    # Static extinction coefficient
    tree_lad = kwargs.get("tree_lad", 1.0) # Leaf area density in m^-1

    # Sky view factor configuration: detect open sky (value 0)
    hit_values = (0,)        # Sky voxels have value 0
    inclusion_mode = False   # Count rays that DON'T hit obstacles (exclusion mode)

    # Generate ray directions over the sky hemisphere (0 to 90 degrees elevation)
    if str(ray_sampling).lower() == "fibonacci":
        ray_directions = _generate_ray_directions_fibonacci(
            int(N_rays), elevation_min_degrees, elevation_max_degrees
        )
    else:
        ray_directions = _generate_ray_directions_grid(
            int(N_azimuth), int(N_elevation), elevation_min_degrees, elevation_max_degrees
        )

    # Compute the SVF map using the generic view index computation
    vi_map = compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, 
                                  hit_values, meshsize, tree_k, tree_lad, inclusion_mode)

    # Display visualization if requested
    if show_plot:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(colormap).copy()
        cmap.set_bad(color='lightgray')  # Gray for invalid observer locations
        plt.figure(figsize=(10, 8))
        plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Sky View Factor')
        plt.axis('off')
        plt.show()

    # Optional OBJ export for 3D visualization
    obj_export = kwargs.get("obj_export", False)
    if obj_export:        
        dem_grid = kwargs.get("dem_grid", np.zeros_like(vi_map))
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

@njit
def rotate_vector_axis_angle(vec, axis, angle):
    """
    Rotate a 3D vector around an arbitrary axis using Rodrigues' rotation formula.
    
    This function implements the Rodrigues rotation formula:
    v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    where k is the unit rotation axis, θ is the rotation angle, and v is the input vector.
    
    Args:
        vec (ndarray): 3D vector to rotate [x, y, z]
        axis (ndarray): 3D rotation axis vector [x, y, z] (will be normalized)
        angle (float): Rotation angle in radians
        
    Returns:
        ndarray: Rotated 3D vector [x, y, z]
    """
    # Normalize rotation axis to unit length
    axis_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_len < 1e-12:
        # Degenerate axis case: return original vector unchanged
        return vec
    
    ux, uy, uz = axis / axis_len
    c = np.cos(angle)
    s = np.sin(angle)
    
    # Calculate dot product: k·v
    dot = vec[0]*ux + vec[1]*uy + vec[2]*uz
    
    # Calculate cross product: k × v
    cross_x = uy*vec[2] - uz*vec[1]
    cross_y = uz*vec[0] - ux*vec[2]
    cross_z = ux*vec[1] - uy*vec[0]
    
    # Apply Rodrigues formula: v_rot = v*c + (k × v)*s + k*(k·v)*(1-c)
    v_rot = np.zeros(3, dtype=np.float64)
    
    # First term: v*cos(θ)
    v_rot[0] = vec[0] * c
    v_rot[1] = vec[1] * c
    v_rot[2] = vec[2] * c
    
    # Second term: (k × v)*sin(θ)
    v_rot[0] += cross_x * s
    v_rot[1] += cross_y * s
    v_rot[2] += cross_z * s
    
    # Third term: k*(k·v)*(1-cos(θ))
    tmp = dot * (1.0 - c)
    v_rot[0] += ux * tmp
    v_rot[1] += uy * tmp
    v_rot[2] += uz * tmp
    
    return v_rot

@njit
def compute_view_factor_for_all_faces(
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
    boundary_epsilon,
    offset_vox=0.51
):
    """
    Compute a per-face "view factor" for a specified set of target voxel classes.

    This function computes view factors from building surface faces to target voxel types
    (e.g., sky, trees, other buildings). It uses hemisphere ray casting with rotation
    to align rays with each face's normal direction.

    Typical usage examples:
    - Sky View Factor: target_values=(0,), inclusion_mode=False (sky voxels)
    - Tree View Factor: target_values=(-2,), inclusion_mode=True (tree voxels)  
    - Building View Factor: target_values=(-3,), inclusion_mode=True (building voxels)

    Args:
        face_centers (np.ndarray): (n_faces, 3) face centroid positions in real coordinates.
        face_normals (np.ndarray): (n_faces, 3) face normal vectors (outward pointing).
        hemisphere_dirs (np.ndarray): (N, 3) set of direction vectors in the upper hemisphere.
        voxel_data (np.ndarray): 3D array of voxel values.
        meshsize (float): Size of each voxel in meters.
        tree_k (float): Tree extinction coefficient for Beer-Lambert law.
        tree_lad (float): Leaf area density in m^-1 for tree transmittance.
        target_values (tuple[int]): Voxel classes that define a 'hit' or target.
        inclusion_mode (bool): If True, hitting target_values counts as visibility.
                               If False, hitting anything NOT in target_values blocks the ray.
        grid_bounds_real (np.ndarray): [[x_min,y_min,z_min],[x_max,y_max,z_max]] in real coords.
        boundary_epsilon (float): Tolerance for identifying boundary vertical faces.

    Returns:
        np.ndarray of shape (n_faces,): Computed view factor for each face.
            NaN values indicate boundary vertical faces that should be excluded.
    """
    n_faces = face_centers.shape[0]
    face_vf_values = np.zeros(n_faces, dtype=np.float64)
    
    # Reference vector pointing upward (+Z direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Process each face individually
    for fidx in range(n_faces):
        center = face_centers[fidx]
        normal = face_normals[fidx]
        
        # Check for boundary vertical faces and mark as NaN
        # This excludes faces on domain edges that may have artificial visibility
        is_vertical = (abs(normal[2]) < 0.01)  # Face normal is nearly horizontal
        
        # Check if face is near domain boundaries
        on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
        on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
        on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
        on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        
        is_boundary_vertical = is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max)
        if is_boundary_vertical:
            face_vf_values[fidx] = np.nan
            continue
        
        # Compute rotation to align face normal with +Z axis
        # This allows us to use the same hemisphere directions for all faces
        norm_n = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm_n < 1e-12:
            # Degenerate normal vector
            face_vf_values[fidx] = 0.0
            continue
        
        # Calculate angle between face normal and +Z axis
        dot_zn = z_axis[0]*normal[0] + z_axis[1]*normal[1] + z_axis[2]*normal[2]
        cos_angle = dot_zn / (norm_n)
        if cos_angle >  1.0: cos_angle =  1.0
        if cos_angle < -1.0: cos_angle = -1.0
        angle = np.arccos(cos_angle)
        
        # Handle special cases and general rotation
        if abs(cos_angle - 1.0) < 1e-9:
            # Face normal is already aligned with +Z => no rotation needed
            local_dirs = hemisphere_dirs
        elif abs(cos_angle + 1.0) < 1e-9:
            # Face normal points in -Z direction => rotate 180 degrees around X axis
            axis_180 = np.array([1.0, 0.0, 0.0])
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(hemisphere_dirs[i], axis_180, np.pi)
        else:
            # General case: rotate around axis perpendicular to both +Z and face normal
            axis_x = z_axis[1]*normal[2] - z_axis[2]*normal[1]
            axis_y = z_axis[2]*normal[0] - z_axis[0]*normal[2]
            axis_z = z_axis[0]*normal[1] - z_axis[1]*normal[0]
            rot_axis = np.array([axis_x, axis_y, axis_z], dtype=np.float64)
            
            local_dirs = np.empty_like(hemisphere_dirs)
            for i in range(hemisphere_dirs.shape[0]):
                local_dirs[i] = rotate_vector_axis_angle(
                    hemisphere_dirs[i],
                    rot_axis,
                    angle
                )
        
        # Count valid ray directions based on face orientation (outward only)
        total_outward = 0  # Rays pointing away from face surface
        num_valid = 0      # Rays that meet all criteria (outward)
        
        for i in range(local_dirs.shape[0]):
            dvec = local_dirs[i]
            # Check if ray points outward from face surface (positive dot product with normal)
            dp = dvec[0]*normal[0] + dvec[1]*normal[1] + dvec[2]*normal[2]
            if dp > 0.0:
                total_outward += 1
                num_valid += 1
        
        # Handle cases with no valid directions
        if total_outward == 0:
            face_vf_values[fidx] = 0.0
            continue
        
        if num_valid == 0:
            face_vf_values[fidx] = 0.0
            continue
        
        # Create array containing only the valid ray directions
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
        
        # Set ray origin slightly offset from face surface to avoid self-intersection
        # Use configurable offset to reduce self-hit artifacts.
        ray_origin = (center / meshsize) + (normal / norm_n) * offset_vox
        
        # Compute fraction of valid rays that "see" the target using generic ray tracing
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
        
        # Scale result by fraction of directions that were valid
        # This normalizes for the hemisphere portion that the face can actually "see"
        fraction_valid = num_valid / total_outward
        face_vf_values[fidx] = vf * fraction_valid
    
    return face_vf_values

def get_surface_view_factor(voxel_data, meshsize, **kwargs):
    """
    Compute and optionally visualize view factors for surface meshes with respect to target voxel classes.
    
    This function provides a flexible framework for computing various surface-based view factors:
    - Sky View Factor: Fraction of sky hemisphere visible from building surfaces
    - Tree View Factor: Fraction of directions that intersect vegetation
    - Building View Factor: Fraction of directions that intersect other buildings
    - Custom View Factors: User-defined target voxel classes
    
    The function extracts surface meshes from the voxel data, then computes view factors
    for each face using hemisphere ray casting with proper geometric transformations.

    Args:
        voxel_data (ndarray): 3D array of voxel values representing the urban environment.
        meshsize (float): Size of each voxel in meters for coordinate scaling.
        **kwargs: Extensive configuration options including:
            # Target specification:
            target_values (tuple[int]): Voxel classes to measure visibility to (default: (0,) for sky)
            inclusion_mode (bool): Interpretation of target_values (default: False for sky)
            
            # Surface extraction:
            building_class_id (int): Voxel class to extract surfaces from (default: -3 for buildings)
            building_id_grid (ndarray): Optional grid mapping voxels to building IDs
            
            # Ray sampling:
            N_azimuth (int): Number of azimuth angles for hemisphere sampling (default: 60)
            N_elevation (int): Number of elevation angles for hemisphere sampling (default: 10)
            
            # Tree transmittance (Beer-Lambert law):
            tree_k (float): Tree extinction coefficient (default: 0.6)
            tree_lad (float): Leaf area density in m^-1 (default: 1.0)
            
            # Visualization and export:
            colormap (str): Matplotlib colormap for visualization (default: 'BuPu_r')
            vmin, vmax (float): Color scale limits (default: 0.0, 1.0)
            obj_export (bool): Whether to export mesh as OBJ file (default: False)
            output_directory (str): Directory for OBJ export (default: "output")
            output_file_name (str): Base filename for OBJ export (default: "surface_view_factor")
            
            # Other options:
            progress_report (bool): Whether to print computation progress (default: False)
            debug (bool): Enable debug output (default: False)

    Returns:
        trimesh.Trimesh: Surface mesh with per-face view factor values stored in metadata.
                        The view factor values can be accessed via mesh.metadata[value_name].
                        Returns None if no surfaces are found or extraction fails.
                        
    Example Usage:
        # Sky View Factor for building surfaces
        mesh = get_surface_view_factor(voxel_data, meshsize, 
                                     target_values=(0,), inclusion_mode=False)
        
        # Tree View Factor for building surfaces  
        mesh = get_surface_view_factor(voxel_data, meshsize,
                                     target_values=(-2,), inclusion_mode=True)
        
        # Custom view factor with OBJ export
        mesh = get_surface_view_factor(voxel_data, meshsize,
                                     target_values=(-3,), inclusion_mode=True,
                                     obj_export=True, output_file_name="building_view_factor")
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os
    
    # Extract configuration parameters with appropriate defaults
    value_name     = kwargs.get("value_name", 'view_factor_values')
    colormap       = kwargs.get("colormap", 'BuPu_r')
    vmin           = kwargs.get("vmin", 0.0)
    vmax           = kwargs.get("vmax", 1.0)
    N_azimuth      = kwargs.get("N_azimuth", 60)
    N_elevation    = kwargs.get("N_elevation", 10)
    ray_sampling   = kwargs.get("ray_sampling", "grid")  # 'grid' or 'fibonacci'
    N_rays         = kwargs.get("N_rays", N_azimuth * N_elevation)
    debug          = kwargs.get("debug", False)
    progress_report= kwargs.get("progress_report", False)
    building_id_grid = kwargs.get("building_id_grid", None)
    
    # Tree transmittance parameters for Beer-Lambert law
    tree_k         = kwargs.get("tree_k", 0.6)
    tree_lad       = kwargs.get("tree_lad", 1.0)
    
    # Target specification - defaults to sky view factor configuration
    target_values  = kwargs.get("target_values", (0,))     # Sky voxels by default
    inclusion_mode = kwargs.get("inclusion_mode", False)   # Exclusion mode for sky
    
    # Surface extraction parameters
    building_class_id = kwargs.get("building_class_id", -3)  # Building voxel class
    
    # Extract surface mesh from the specified voxel class
    try:
        building_mesh = create_voxel_mesh(
            voxel_data, 
            building_class_id, 
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'  # Extract surfaces exposed to air
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No surfaces found in voxel data for the specified class.")
            return None
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None
    
    if progress_report:
        print(f"Processing view factor for {len(building_mesh.faces)} faces...")

    # Extract geometric properties from the mesh
    face_centers = building_mesh.triangles_center  # Centroid of each face
    face_normals = building_mesh.face_normals      # Outward normal of each face
    
    # Generate hemisphere ray directions (local +Z hemisphere)
    if str(ray_sampling).lower() == "fibonacci":
        hemisphere_dirs = _generate_ray_directions_fibonacci(
            int(N_rays), 0.0, 90.0
        )
    else:
        hemisphere_dirs = _generate_ray_directions_grid(
            int(N_azimuth), int(N_elevation), 0.0, 90.0
        )
    
    # Calculate domain bounds for boundary face detection
    nx, ny, nz = voxel_data.shape
    grid_bounds_voxel = np.array([[0,0,0],[nx, ny, nz]], dtype=np.float64)
    grid_bounds_real = grid_bounds_voxel * meshsize
    boundary_epsilon = meshsize * 0.05  # Tolerance for boundary detection
    
    # Attempt fast path using boolean masks + orthonormal basis + parallel Numba
    fast_path = kwargs.get("fast_path", True)
    face_vf_values = None
    if fast_path:
        try:
            vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque = _prepare_masks_for_view(
                voxel_data, target_values, inclusion_mode
            )
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

    # Fallback to original implementation if fast path unavailable/failed
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
    
    # Store computed view factor values in mesh metadata for later access
    if not hasattr(building_mesh, 'metadata'):
        building_mesh.metadata = {}
    building_mesh.metadata[value_name] = face_vf_values
       
    # Optional OBJ file export for external visualization/analysis
    obj_export = kwargs.get("obj_export", False)
    if obj_export:
        output_dir      = kwargs.get("output_directory", "output")
        output_file_name= kwargs.get("output_file_name", "surface_view_factor")
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh

# ==========================
# Fast per-face view factor (parallel)
# ==========================
def _prepare_masks_for_view(voxel_data, target_values, inclusion_mode):
    is_tree = (voxel_data == -2)
    # Targets mask (for inclusion mode)
    target_mask = np.zeros(voxel_data.shape, dtype=np.bool_)
    for tv in target_values:
        target_mask |= (voxel_data == tv)
    if inclusion_mode:
        # Opaque: anything non-air, non-tree, and not target
        is_opaque = (voxel_data != 0) & (~is_tree) & (~target_mask)
        # Allowed mask is unused in inclusion mode but keep shape compatibility
        is_allowed = target_mask.copy()
    else:
        # Exclusion mode: allowed voxels are target_values (e.g., sky=0)
        is_allowed = target_mask
        # Opaque: anything not tree and not allowed
        is_opaque = (~is_tree) & (~is_allowed)
    return is_tree, target_mask, is_allowed, is_opaque

@njit(cache=True, fastmath=True, nogil=True)
def _build_face_basis(normal):
    nx = normal[0]; ny = normal[1]; nz = normal[2]
    nrm = (nx*nx + ny*ny + nz*nz) ** 0.5
    if nrm < 1e-12:
        # Default to +Z if degenerate
        return (np.array((1.0, 0.0, 0.0)),
                np.array((0.0, 1.0, 0.0)),
                np.array((0.0, 0.0, 1.0)))
    invn = 1.0 / nrm
    nx *= invn; ny *= invn; nz *= invn
    n = np.array((nx, ny, nz))
    # Choose helper to avoid near-parallel cross
    if abs(nz) < 0.999:
        helper = np.array((0.0, 0.0, 1.0))
    else:
        helper = np.array((1.0, 0.0, 0.0))
    # u = normalize(helper x n)
    ux = helper[1]*n[2] - helper[2]*n[1]
    uy = helper[2]*n[0] - helper[0]*n[2]
    uz = helper[0]*n[1] - helper[1]*n[0]
    ul = (ux*ux + uy*uy + uz*uz) ** 0.5
    if ul < 1e-12:
        u = np.array((1.0, 0.0, 0.0))
    else:
        invul = 1.0 / ul
        u = np.array((ux*invul, uy*invul, uz*invul))
    # v = n x u
    vx = n[1]*u[2] - n[2]*u[1]
    vy = n[2]*u[0] - n[0]*u[2]
    vz = n[0]*u[1] - n[1]*u[0]
    v = np.array((vx, vy, vz))
    return u, v, n

@njit(cache=True, fastmath=True, nogil=True)
def _ray_visibility_contrib(origin, direction,
                            vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
                            att, att_cutoff,
                            inclusion_mode, trees_are_targets):
    nx, ny, nz = vox_is_opaque.shape
    x0 = origin[0]; y0 = origin[1]; z0 = origin[2]
    dx = direction[0]; dy = direction[1]; dz = direction[2]

    # Normalize
    L = (dx*dx + dy*dy + dz*dz) ** 0.5
    if L == 0.0:
        return 0.0
    invL = 1.0 / L
    dx *= invL; dy *= invL; dz *= invL

    # Starting point and indices
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
            # Out of bounds: for exclusion mode return transmittance, else no hit
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
                # First tree encountered; contribution is partial visibility
                return 1.0 - (T if T < 1.0 else 1.0)

        if inclusion_mode:
            if (not vox_is_tree[i, j, k]) and vox_is_target[i, j, k]:
                return 1.0
        else:
            # Exclusion: allow only allowed or tree; any other value blocks
            if (not vox_is_tree[i, j, k]) and (not vox_is_allowed[i, j, k]):
                return 0.0

        # Step DDA
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
def _compute_view_factor_faces_chunk(face_centers, face_normals, hemisphere_dirs,
                                     vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
                                     meshsize, att, att_cutoff,
                                     grid_bounds_real, boundary_epsilon,
                                     inclusion_mode, trees_are_targets):
    n_faces = face_centers.shape[0]
    out = np.empty(n_faces, dtype=np.float64)
    for f in prange(n_faces):
        center = face_centers[f]
        normal = face_normals[f]

        # Boundary vertical exclusion
        is_vertical = (abs(normal[2]) < 0.01)
        on_x_min = (abs(center[0] - grid_bounds_real[0,0]) < boundary_epsilon)
        on_y_min = (abs(center[1] - grid_bounds_real[0,1]) < boundary_epsilon)
        on_x_max = (abs(center[0] - grid_bounds_real[1,0]) < boundary_epsilon)
        on_y_max = (abs(center[1] - grid_bounds_real[1,1]) < boundary_epsilon)
        if is_vertical and (on_x_min or on_y_min or on_x_max or on_y_max):
            out[f] = np.nan
            continue

        u, v, n = _build_face_basis(normal)

        # Origin slightly outside face
        ox = center[0] / meshsize + n[0] * 0.51
        oy = center[1] / meshsize + n[1] * 0.51
        oz = center[2] / meshsize + n[2] * 0.51
        origin = np.array((ox, oy, oz))

        vis_sum = 0.0
        valid = 0
        for i in range(hemisphere_dirs.shape[0]):
            lx = hemisphere_dirs[i,0]; ly = hemisphere_dirs[i,1]; lz = hemisphere_dirs[i,2]
            # Transform local hemisphere (+Z up) into world; outward is +n
            dx = u[0]*lx + v[0]*ly + n[0]*lz
            dy = u[1]*lx + v[1]*ly + n[1]*lz
            dz = u[2]*lx + v[2]*ly + n[2]*lz
            # Only outward directions
            if (dx*n[0] + dy*n[1] + dz*n[2]) <= 0.0:
                continue
            contrib = _ray_visibility_contrib(origin, np.array((dx, dy, dz)),
                                              vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
                                              att, att_cutoff,
                                              inclusion_mode, trees_are_targets)
            vis_sum += contrib
            valid += 1
        out[f] = 0.0 if valid == 0 else (vis_sum / valid)
    return out

def _compute_view_factor_faces_progress(face_centers, face_normals, hemisphere_dirs,
                                        vox_is_tree, vox_is_target, vox_is_allowed, vox_is_opaque,
                                        meshsize, att, att_cutoff,
                                        grid_bounds_real, boundary_epsilon,
                                        inclusion_mode, trees_are_targets,
                                        progress_report=False, chunks=10):
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

# ==========================
# DDA ray traversal (fast)
# ==========================
@njit(cache=True, fastmath=True, nogil=True)
def _trace_ray(vox_is_tree, vox_is_opaque, origin, target, att, att_cutoff):
    nx, ny, nz = vox_is_opaque.shape
    x0, y0, z0 = origin[0], origin[1], origin[2]
    x1, y1, z1 = target[0], target[1], target[2]

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return True
    inv_len = 1.0 / length
    dx *= inv_len; dy *= inv_len; dz *= inv_len

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
    ti = int(x1); tj = int(y1); tk = int(z1)

    while True:
        if (i < 0) or (i >= nx) or (j < 0) or (j >= ny) or (k < 0) or (k >= nz):
            return False

        if vox_is_opaque[i, j, k]:
            return False
        if vox_is_tree[i, j, k]:
            T *= att
            if T < att_cutoff:
                return False

        if (i == ti) and (j == tj) and (k == tk):
            return True

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


# ==========================
# Per-face landmark visibility
# ==========================
@njit(cache=True, fastmath=True, nogil=True)
def _compute_face_visibility(face_center, face_normal,
                             landmark_positions_vox,
                             vox_is_tree, vox_is_opaque,
                             meshsize, att, att_cutoff,
                             grid_bounds_real, boundary_epsilon):
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

        if _trace_ray(vox_is_tree, vox_is_opaque,
                      np.array((ox, oy, oz)), np.array((tx, ty, tz)),
                      att, att_cutoff):
            return 1.0

    return 0.0

# ==========================
# Precompute voxel class masks
# ==========================
def _prepare_voxel_classes(voxel_data, landmark_value=-30):
    is_tree = (voxel_data == -2)
    is_opaque = (voxel_data != 0) & (voxel_data != landmark_value) & (~is_tree)
    return is_tree, is_opaque

# ==========================
# Chunked parallel loop for progress
# ==========================
def _compute_all_faces_progress(face_centers, face_normals, landmark_positions_vox,
                                vox_is_tree, vox_is_opaque,
                                meshsize, att, att_cutoff,
                                grid_bounds_real, boundary_epsilon,
                                progress_report=False, chunks=10):
    n_faces = face_centers.shape[0]
    results = np.empty(n_faces, dtype=np.float64)

    # Determine chunk size
    step = math.ceil(n_faces / chunks)
    for start in range(0, n_faces, step):
        end = min(start + step, n_faces)
        # Run parallel compute on this chunk
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
def _compute_faces_chunk(face_centers, face_normals, landmark_positions_vox,
                         vox_is_tree, vox_is_opaque,
                         meshsize, att, att_cutoff,
                         grid_bounds_real, boundary_epsilon):
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


# ==========================
# Main function
# ==========================
def get_surface_landmark_visibility(voxel_data, building_id_grid, building_gdf, meshsize, **kwargs):
    import matplotlib.pyplot as plt
    import os

    progress_report = kwargs.get("progress_report", False)

    # --- Landmark selection logic (unchanged) ---
    landmark_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    if landmark_ids is None:
        if landmark_polygon is not None:
            landmark_ids = get_buildings_in_drawn_polygon(building_gdf, landmark_polygon, operation='within')
        else:
            rectangle_vertices = kwargs.get("rectangle_vertices", None)
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

    # Precompute masks + attenuation
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