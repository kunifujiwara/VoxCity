"""
Sky Discretization Methods for Solar Simulation.

This module provides various methods for dividing the sky hemisphere into patches
to improve efficiency of cumulative solar irradiance calculations. Instead of
tracing rays for each hourly sun position, sun positions can be binned into sky
patches and rays traced once per patch.

Supported methods:
- Tregenza: 145 patches (standard in Radiance, EnergyPlus, DAYSIM)
- Reinhart: Tregenza × MF² patches (high-resolution, used in DAYSIM/Honeybee)
- Uniform Grid: Regular azimuth × elevation grid
- Fibonacci: Quasi-uniform distribution using golden angle spiral
"""

import numpy as np
from numba import njit


# =============================================================================
# Tregenza Sky Subdivision (145 patches)
# =============================================================================

# Tregenza band definitions: (elevation_center, num_patches_in_band)
TREGENZA_BANDS = [
    (6.0, 30),    # Band 1: 0°-12°, center at 6°
    (18.0, 30),   # Band 2: 12°-24°, center at 18°
    (30.0, 24),   # Band 3: 24°-36°, center at 24°
    (42.0, 24),   # Band 4: 36°-48°, center at 42°
    (54.0, 18),   # Band 5: 48°-60°, center at 54°
    (66.0, 12),   # Band 6: 60°-72°, center at 66°
    (78.0, 6),    # Band 7: 72°-84°, center at 78°
    (90.0, 1),    # Band 8: 84°-90°, zenith patch
]

# Tregenza band elevation boundaries (for binning)
TREGENZA_BAND_BOUNDARIES = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 90.0]


def generate_tregenza_patches():
    """
    Generate the 145 Tregenza sky patch center directions.
    
    The Tregenza subdivision divides the sky hemisphere into 145 patches
    arranged in 8 altitude bands. This is the standard sky discretization
    used in Radiance (genskyvec), EnergyPlus, DAYSIM, and Ladybug Tools.
    
    Returns
    -------
    patches : np.ndarray, shape (145, 2)
        Array of (azimuth_degrees, elevation_degrees) for each patch center.
    directions : np.ndarray, shape (145, 3)
        Unit direction vectors (dx, dy, dz) pointing to each patch center.
    solid_angles : np.ndarray, shape (145,)
        Solid angle (steradians) of each patch.
    
    References
    ----------
    Tregenza, P.R. (1987). "Subdivision of the sky hemisphere for luminance
    measurements." Lighting Research & Technology, 19(1), 13-14.
    """
    patches = []
    directions = []
    solid_angles = []
    
    for band_idx, (elev_center, n_patches) in enumerate(TREGENZA_BANDS):
        elev_rad = np.deg2rad(elev_center)
        cos_elev = np.cos(elev_rad)
        sin_elev = np.sin(elev_rad)
        
        # Solid angle calculation for band
        if band_idx == 0:
            elev_low = 0.0
        else:
            elev_low = TREGENZA_BAND_BOUNDARIES[band_idx]
        elev_high = TREGENZA_BAND_BOUNDARIES[band_idx + 1]
        
        # Solid angle of band = 2π × (sin(θ_high) - sin(θ_low))
        band_solid_angle = 2 * np.pi * (
            np.sin(np.deg2rad(elev_high)) - np.sin(np.deg2rad(elev_low))
        )
        patch_solid_angle = band_solid_angle / n_patches
        
        for i in range(n_patches):
            # Azimuth at patch center
            az_deg = (i + 0.5) * 360.0 / n_patches
            az_rad = np.deg2rad(az_deg)
            
            # Direction vector
            dx = cos_elev * np.cos(az_rad)
            dy = cos_elev * np.sin(az_rad)
            dz = sin_elev
            
            patches.append((az_deg, elev_center))
            directions.append((dx, dy, dz))
            solid_angles.append(patch_solid_angle)
    
    return (
        np.array(patches, dtype=np.float64),
        np.array(directions, dtype=np.float64),
        np.array(solid_angles, dtype=np.float64),
    )


def get_tregenza_patch_index(azimuth_deg, elevation_deg):
    """
    Get the Tregenza patch index for a given sun position.
    
    Parameters
    ----------
    azimuth_deg : float
        Solar azimuth in degrees (0-360, measured clockwise from north).
    elevation_deg : float
        Solar elevation in degrees (0-90).
    
    Returns
    -------
    int
        Patch index (0-144), or -1 if below horizon.
    """
    if elevation_deg < 0:
        return -1
    
    # Find altitude band
    band_idx = 0
    patch_offset = 0
    for i, boundary in enumerate(TREGENZA_BAND_BOUNDARIES[1:]):
        if elevation_deg < boundary:
            band_idx = i
            break
        patch_offset += TREGENZA_BANDS[i][1]
    else:
        # Zenith patch
        return 144
    
    # Find azimuth patch within band
    n_patches = TREGENZA_BANDS[band_idx][1]
    az_normalized = azimuth_deg % 360.0
    patch_in_band = int(az_normalized / (360.0 / n_patches))
    patch_in_band = min(patch_in_band, n_patches - 1)
    
    return patch_offset + patch_in_band


@njit(cache=True)
def get_tregenza_patch_index_fast(azimuth_deg, elevation_deg):
    """
    Numba-accelerated version of get_tregenza_patch_index.
    
    Parameters
    ----------
    azimuth_deg : float
        Solar azimuth in degrees (0-360).
    elevation_deg : float
        Solar elevation in degrees (0-90).
    
    Returns
    -------
    int
        Patch index (0-144), or -1 if below horizon.
    """
    if elevation_deg < 0.0:
        return -1
    
    # Band boundaries and patch counts (hardcoded for Numba)
    boundaries = np.array([0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 90.0])
    patch_counts = np.array([30, 30, 24, 24, 18, 12, 6, 1])
    
    # Find band
    band_idx = 7  # Default to zenith band
    for i in range(7):
        if elevation_deg < boundaries[i + 1]:
            band_idx = i
            break
    
    # Calculate offset to this band
    patch_offset = 0
    for i in range(band_idx):
        patch_offset += patch_counts[i]
    
    # Find patch within band
    n_patches = patch_counts[band_idx]
    if n_patches == 1:
        return patch_offset  # Zenith
    
    az_normalized = azimuth_deg % 360.0
    patch_in_band = int(az_normalized / (360.0 / n_patches))
    if patch_in_band >= n_patches:
        patch_in_band = n_patches - 1
    
    return patch_offset + patch_in_band


# =============================================================================
# Reinhart Sky Subdivision (Tregenza × MF²)
# =============================================================================

def generate_reinhart_patches(mf=4):
    """
    Generate Reinhart sky patches (subdivided Tregenza).
    
    The Reinhart subdivision increases resolution by subdividing each Tregenza
    patch by a multiplication factor (MF). With MF=4, this yields 2305 patches.
    
    Parameters
    ----------
    mf : int
        Multiplication factor. Common values:
        - MF=1: 145 patches (same as Tregenza)
        - MF=2: 577 patches
        - MF=4: 2305 patches (common for annual daylight simulation)
        - MF=6: 5185 patches
    
    Returns
    -------
    patches : np.ndarray, shape (N, 2)
        Array of (azimuth_degrees, elevation_degrees) for each patch center.
    directions : np.ndarray, shape (N, 3)
        Unit direction vectors (dx, dy, dz) for each patch center.
    solid_angles : np.ndarray, shape (N,)
        Solid angle (steradians) of each patch.
    
    References
    ----------
    Reinhart, C.F. & Walkenhorst, O. (2001). "Validation of dynamic RADIANCE-based
    daylight simulations for a test office with external blinds." Energy and
    Buildings, 33(7), 683-697.
    """
    mf = max(1, int(mf))
    patches = []
    directions = []
    solid_angles = []
    
    for band_idx, (elev_center_base, n_patches_base) in enumerate(TREGENZA_BANDS):
        # Subdivide elevation bands
        if band_idx == len(TREGENZA_BANDS) - 1:
            # Zenith: subdivide into MF² patches arranged in concentric rings
            n_sub_bands = mf
        else:
            n_sub_bands = mf
        
        elev_low = TREGENZA_BAND_BOUNDARIES[band_idx]
        elev_high = TREGENZA_BAND_BOUNDARIES[band_idx + 1]
        elev_range = elev_high - elev_low
        
        for sub_band in range(n_sub_bands):
            # Sub-band elevation center
            sub_elev_low = elev_low + sub_band * elev_range / n_sub_bands
            sub_elev_high = elev_low + (sub_band + 1) * elev_range / n_sub_bands
            sub_elev_center = (sub_elev_low + sub_elev_high) / 2.0
            
            elev_rad = np.deg2rad(sub_elev_center)
            cos_elev = np.cos(elev_rad)
            sin_elev = np.sin(elev_rad)
            
            # Solid angle of sub-band
            sub_band_solid_angle = 2 * np.pi * (
                np.sin(np.deg2rad(sub_elev_high)) - np.sin(np.deg2rad(sub_elev_low))
            )
            
            # Number of azimuth patches in sub-band
            if band_idx == len(TREGENZA_BANDS) - 1:
                # Zenith: for innermost ring, use fewer patches
                n_az = max(1, n_patches_base * mf * (sub_band + 1) // n_sub_bands)
            else:
                n_az = n_patches_base * mf
            
            patch_solid_angle = sub_band_solid_angle / n_az
            
            for i in range(n_az):
                az_deg = (i + 0.5) * 360.0 / n_az
                az_rad = np.deg2rad(az_deg)
                
                dx = cos_elev * np.cos(az_rad)
                dy = cos_elev * np.sin(az_rad)
                dz = sin_elev
                
                patches.append((az_deg, sub_elev_center))
                directions.append((dx, dy, dz))
                solid_angles.append(patch_solid_angle)
    
    return (
        np.array(patches, dtype=np.float64),
        np.array(directions, dtype=np.float64),
        np.array(solid_angles, dtype=np.float64),
    )


# =============================================================================
# Uniform Grid Subdivision
# =============================================================================

def generate_uniform_grid_patches(n_azimuth=36, n_elevation=9):
    """
    Generate uniform grid sky patches.
    
    Simple subdivision with equal azimuth and elevation spacing.
    Note: This creates non-equal solid angle patches (smaller near zenith).
    
    Parameters
    ----------
    n_azimuth : int
        Number of azimuth divisions (default: 36 = 10° spacing).
    n_elevation : int
        Number of elevation divisions (default: 9 = 10° spacing).
    
    Returns
    -------
    patches : np.ndarray, shape (N, 2)
        Array of (azimuth_degrees, elevation_degrees) for each patch center.
    directions : np.ndarray, shape (N, 3)
        Unit direction vectors for each patch center.
    solid_angles : np.ndarray, shape (N,)
        Solid angle (steradians) of each patch.
    """
    patches = []
    directions = []
    solid_angles = []
    
    elev_step = 90.0 / n_elevation
    az_step = 360.0 / n_azimuth
    
    for j in range(n_elevation):
        elev_low = j * elev_step
        elev_high = (j + 1) * elev_step
        elev_center = (elev_low + elev_high) / 2.0
        
        elev_rad = np.deg2rad(elev_center)
        cos_elev = np.cos(elev_rad)
        sin_elev = np.sin(elev_rad)
        
        # Solid angle for this elevation band
        band_solid_angle = 2 * np.pi * (
            np.sin(np.deg2rad(elev_high)) - np.sin(np.deg2rad(elev_low))
        )
        patch_solid_angle = band_solid_angle / n_azimuth
        
        for i in range(n_azimuth):
            az_center = (i + 0.5) * az_step
            az_rad = np.deg2rad(az_center)
            
            dx = cos_elev * np.cos(az_rad)
            dy = cos_elev * np.sin(az_rad)
            dz = sin_elev
            
            patches.append((az_center, elev_center))
            directions.append((dx, dy, dz))
            solid_angles.append(patch_solid_angle)
    
    return (
        np.array(patches, dtype=np.float64),
        np.array(directions, dtype=np.float64),
        np.array(solid_angles, dtype=np.float64),
    )


# =============================================================================
# Fibonacci Spiral (Quasi-Uniform)
# =============================================================================

def generate_fibonacci_patches(n_patches=145):
    """
    Generate quasi-uniform sky patches using Fibonacci spiral.
    
    Uses the golden angle spiral to distribute points nearly uniformly
    on the hemisphere. This provides more uniform patch areas than
    regular grids with fewer total patches.
    
    Parameters
    ----------
    n_patches : int
        Number of patches to generate (default: 145 to match Tregenza).
    
    Returns
    -------
    patches : np.ndarray, shape (N, 2)
        Array of (azimuth_degrees, elevation_degrees) for each patch center.
    directions : np.ndarray, shape (N, 3)
        Unit direction vectors for each patch center.
    solid_angles : np.ndarray, shape (N,)
        Approximate solid angle per patch (uniform for Fibonacci).
    """
    n = max(1, int(n_patches))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Hemisphere solid angle = 2π steradians
    patch_solid_angle = 2.0 * np.pi / n
    
    patches = []
    directions = []
    solid_angles = []
    
    for i in range(n):
        # z ranges from 0 (horizon) to 1 (zenith)
        z = (i + 0.5) / n
        elevation_rad = np.arcsin(z)
        elevation_deg = np.rad2deg(elevation_rad)
        
        # Azimuth from golden angle
        azimuth_rad = i * golden_angle
        azimuth_deg = np.rad2deg(azimuth_rad) % 360.0
        
        # Direction vector
        r = np.sqrt(1.0 - z * z)
        dx = r * np.cos(azimuth_rad)
        dy = r * np.sin(azimuth_rad)
        dz = z
        
        patches.append((azimuth_deg, elevation_deg))
        directions.append((dx, dy, dz))
        solid_angles.append(patch_solid_angle)
    
    return (
        np.array(patches, dtype=np.float64),
        np.array(directions, dtype=np.float64),
        np.array(solid_angles, dtype=np.float64),
    )


# =============================================================================
# Sun Position Binning
# =============================================================================

def bin_sun_positions_to_patches(
    azimuth_arr,
    elevation_arr,
    dni_arr,
    method="tregenza",
    **kwargs
):
    """
    Bin hourly sun positions into sky patches and aggregate DNI.
    
    This is the key optimization for cumulative solar irradiance: instead of
    tracing rays for every hourly sun position, aggregate DNI values for each
    sky patch and trace rays once per patch.
    
    Parameters
    ----------
    azimuth_arr : np.ndarray
        Array of solar azimuth values in degrees.
    elevation_arr : np.ndarray
        Array of solar elevation values in degrees.
    dni_arr : np.ndarray
        Array of Direct Normal Irradiance values (W/m²).
    method : str
        Sky discretization method: "tregenza", "reinhart", "uniform", "fibonacci".
    **kwargs : dict
        Additional parameters for patch generation (e.g., mf for Reinhart).
    
    Returns
    -------
    patch_directions : np.ndarray, shape (N, 3)
        Unit direction vectors for each patch.
    patch_cumulative_dni : np.ndarray, shape (N,)
        Cumulative DNI (W·h/m²) for each patch.
    patch_solid_angles : np.ndarray, shape (N,)
        Solid angle of each patch.
    patch_hours : np.ndarray, shape (N,)
        Number of hours with sun in each patch.
    """
    # Generate patches based on method
    if method.lower() == "tregenza":
        patches, directions, solid_angles = generate_tregenza_patches()
    elif method.lower() == "reinhart":
        mf = kwargs.get("mf", 4)
        patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
    elif method.lower() == "uniform":
        n_az = kwargs.get("n_azimuth", 36)
        n_el = kwargs.get("n_elevation", 9)
        patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
    elif method.lower() == "fibonacci":
        n = kwargs.get("n_patches", 145)
        patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n)
    else:
        raise ValueError(f"Unknown sky discretization method: {method}")
    
    n_patches = len(patches)
    cumulative_dni = np.zeros(n_patches, dtype=np.float64)
    hours_count = np.zeros(n_patches, dtype=np.int32)
    
    # Bin each sun position
    for i in range(len(azimuth_arr)):
        elev = elevation_arr[i]
        if elev <= 0:
            continue  # Below horizon
        
        az = azimuth_arr[i]
        dni = dni_arr[i]
        
        # Find nearest patch
        if method.lower() == "tregenza":
            patch_idx = get_tregenza_patch_index(az, elev)
        else:
            # For other methods, find nearest patch by direction
            elev_rad = np.deg2rad(elev)
            az_rad = np.deg2rad(az)
            sun_dir = np.array([
                np.cos(elev_rad) * np.cos(az_rad),
                np.cos(elev_rad) * np.sin(az_rad),
                np.sin(elev_rad)
            ])
            # Dot product with all patch directions
            dots = np.sum(directions * sun_dir, axis=1)
            patch_idx = np.argmax(dots)
        
        if patch_idx >= 0:
            cumulative_dni[patch_idx] += dni
            hours_count[patch_idx] += 1
    
    return directions, cumulative_dni, solid_angles, hours_count


@njit(cache=True, parallel=True)
def bin_sun_positions_to_tregenza_fast(azimuth_arr, elevation_arr, dni_arr):
    """
    Numba-accelerated binning of sun positions to Tregenza patches.
    
    Parameters
    ----------
    azimuth_arr : np.ndarray
        Array of solar azimuth values in degrees.
    elevation_arr : np.ndarray
        Array of solar elevation values in degrees.
    dni_arr : np.ndarray
        Array of Direct Normal Irradiance values (W/m²).
    
    Returns
    -------
    cumulative_dni : np.ndarray, shape (145,)
        Cumulative DNI (W·h/m²) for each Tregenza patch.
    hours_count : np.ndarray, shape (145,)
        Number of hours with sun in each patch.
    """
    cumulative_dni = np.zeros(145, dtype=np.float64)
    hours_count = np.zeros(145, dtype=np.int32)
    
    n = len(azimuth_arr)
    for i in range(n):
        elev = elevation_arr[i]
        if elev <= 0.0:
            continue
        
        az = azimuth_arr[i]
        dni = dni_arr[i]
        
        patch_idx = get_tregenza_patch_index_fast(az, elev)
        if patch_idx >= 0:
            cumulative_dni[patch_idx] += dni
            hours_count[patch_idx] += 1
    
    return cumulative_dni, hours_count


# =============================================================================
# Utility Functions
# =============================================================================

def get_patch_info(method="tregenza", **kwargs):
    """
    Get information about a sky discretization method.
    
    Parameters
    ----------
    method : str
        Sky discretization method.
    **kwargs : dict
        Additional parameters for the method.
    
    Returns
    -------
    dict
        Dictionary with patch count, method name, and parameters.
    """
    if method.lower() == "tregenza":
        patches, _, _ = generate_tregenza_patches()
        return {
            "method": "Tregenza",
            "n_patches": len(patches),
            "description": "Standard 145-patch subdivision (Radiance, DAYSIM)",
            "reference": "Tregenza (1987)"
        }
    elif method.lower() == "reinhart":
        mf = kwargs.get("mf", 4)
        patches, _, _ = generate_reinhart_patches(mf=mf)
        return {
            "method": "Reinhart",
            "n_patches": len(patches),
            "mf": mf,
            "description": f"High-resolution subdivision with MF={mf}",
            "reference": "Reinhart & Walkenhorst (2001)"
        }
    elif method.lower() == "uniform":
        n_az = kwargs.get("n_azimuth", 36)
        n_el = kwargs.get("n_elevation", 9)
        patches, _, _ = generate_uniform_grid_patches(n_az, n_el)
        return {
            "method": "Uniform Grid",
            "n_patches": len(patches),
            "n_azimuth": n_az,
            "n_elevation": n_el,
            "description": f"Regular grid with {n_az}×{n_el} patches"
        }
    elif method.lower() == "fibonacci":
        n = kwargs.get("n_patches", 145)
        patches, _, _ = generate_fibonacci_patches(n_patches=n)
        return {
            "method": "Fibonacci Spiral",
            "n_patches": len(patches),
            "description": "Quasi-uniform distribution using golden angle"
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_sky_patches(method="tregenza", ax=None, show=True, **kwargs):
    """
    Visualize sky patches on a polar plot.
    
    Parameters
    ----------
    method : str
        Sky discretization method.
    ax : matplotlib axis, optional
        Existing polar axis to plot on.
    show : bool
        Whether to call plt.show().
    **kwargs : dict
        Additional parameters for patch generation.
    
    Returns
    -------
    ax : matplotlib axis
        The plot axis.
    """
    import matplotlib.pyplot as plt
    
    # Generate patches
    if method.lower() == "tregenza":
        patches, _, _ = generate_tregenza_patches()
    elif method.lower() == "reinhart":
        patches, _, _ = generate_reinhart_patches(**kwargs)
    elif method.lower() == "uniform":
        patches, _, _ = generate_uniform_grid_patches(**kwargs)
    elif method.lower() == "fibonacci":
        patches, _, _ = generate_fibonacci_patches(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convert to polar coordinates (theta=azimuth, r=90-elevation)
    theta = np.deg2rad(patches[:, 0])
    r = 90.0 - patches[:, 1]  # Zenith at center
    
    ax.scatter(theta, r, s=10, alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax.set_title(f"{method.capitalize()} Sky Patches (n={len(patches)})")
    
    if show:
        plt.show()
    
    return ax
