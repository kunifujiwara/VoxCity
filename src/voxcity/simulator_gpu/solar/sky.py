"""
Sky Discretization Methods for Cumulative Solar Irradiance Calculation.

This module provides various methods for dividing the sky hemisphere into patches
to improve efficiency of cumulative solar irradiance calculations. Instead of
tracing rays for each hourly sun position, sun positions can be binned into sky
patches and rays traced once per patch.

Supported methods:
- Tregenza: 145 patches (standard in Radiance, EnergyPlus, DAYSIM)
- Reinhart: Tregenza × MF² patches (high-resolution, used in DAYSIM/Honeybee)
- Uniform Grid: Regular azimuth × elevation grid
- Fibonacci: Quasi-uniform distribution using golden angle spiral

This approach significantly reduces computation time for annual simulations:
- 8760 hourly timesteps → ~145-2305 ray traces (30-60× speedup)
- Each patch accumulates radiation from multiple sun positions
- Patch solid angles weight the contributions correctly

References:
- Tregenza, P.R. (1987). "Subdivision of the sky hemisphere for luminance
  measurements." Lighting Research & Technology, 19(1), 13-14.
- Reinhart, C.F. & Walkenhorst, O. (2001). "Validation of dynamic RADIANCE-based
  daylight simulations for a test office with external blinds." Energy and
  Buildings, 33(7), 683-697.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from numba import njit


# =============================================================================
# Constants and Data
# =============================================================================

# Tregenza band definitions: (elevation_center, num_patches_in_band)
TREGENZA_BANDS = [
    (6.0, 30),    # Band 1: 0°-12°, center at 6°
    (18.0, 30),   # Band 2: 12°-24°, center at 18°
    (30.0, 24),   # Band 3: 24°-36°, center at 30°
    (42.0, 24),   # Band 4: 36°-48°, center at 42°
    (54.0, 18),   # Band 5: 48°-60°, center at 54°
    (66.0, 12),   # Band 6: 60°-72°, center at 66°
    (78.0, 6),    # Band 7: 72°-84°, center at 78°
    (90.0, 1),    # Band 8: 84°-90°, zenith patch
]

# Tregenza band elevation boundaries (for binning)
TREGENZA_BAND_BOUNDARIES = np.array([0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 90.0])

# Number of patches per Tregenza band (for fast lookup)
TREGENZA_PATCH_COUNTS = np.array([30, 30, 24, 24, 18, 12, 6, 1], dtype=np.int32)


# =============================================================================
# Sky Patch Data Structure
# =============================================================================

@dataclass
class SkyPatches:
    """
    Container for sky patch discretization data.
    
    Attributes:
        method: Discretization method name
        n_patches: Total number of patches
        patches: Array of (azimuth, elevation) in degrees, shape (N, 2)
        directions: Unit direction vectors (dx, dy, dz), shape (N, 3)
        solid_angles: Solid angle per patch in steradians, shape (N,)
        metadata: Additional method-specific parameters
    """
    method: str
    n_patches: int
    patches: np.ndarray      # (N, 2) - azimuth, elevation in degrees
    directions: np.ndarray   # (N, 3) - unit vectors pointing to sky
    solid_angles: np.ndarray # (N,) - steradians
    metadata: Dict[str, Any]

    # VoxCity compatibility: allow tuple-unpacking
    # VoxCity returns (patches, directions, solid_angles)
    def __iter__(self):
        yield self.patches
        yield self.directions
        yield self.solid_angles

    # VoxCity compatibility aliases
    @property
    def patch_directions(self):
        return self.directions

    @property
    def patch_solid_angles(self):
        return self.solid_angles


@dataclass  
class BinnedSolarData:
    """
    Solar data binned into sky patches for cumulative simulation.
    
    Attributes:
        sky_patches: SkyPatches object with patch geometry
        cumulative_dni: Cumulative DNI (Wh/m²) per patch, shape (N,)
        cumulative_dhi: Cumulative DHI (Wh/m²) distributed by patch solid angle
        hours_per_patch: Number of sun hours in each patch, shape (N,)
        total_daytime_hours: Total hours with sun above horizon
    """
    sky_patches: SkyPatches
    cumulative_dni: np.ndarray      # (N,) Wh/m²
    cumulative_dhi: np.ndarray      # Total DHI for isotropic distribution
    hours_per_patch: np.ndarray     # (N,)
    total_daytime_hours: int

    # VoxCity compatibility: allow tuple-unpacking
    # VoxCity returns (directions, cumulative_dni, solid_angles, hours_count)
    def __iter__(self):
        yield self.sky_patches.directions
        yield self.cumulative_dni
        yield self.sky_patches.solid_angles
        yield self.hours_per_patch

    # VoxCity compatibility aliases
    @property
    def directions(self):
        return self.sky_patches.directions

    @property
    def solid_angles(self):
        return self.sky_patches.solid_angles

    @property
    def hours_count(self):
        return self.hours_per_patch

    @property
    def patch_directions(self):
        return self.sky_patches.directions

    @property
    def patch_cumulative_dni(self):
        return self.cumulative_dni

    @property
    def patch_solid_angles(self):
        return self.sky_patches.solid_angles

    @property
    def patch_hours(self):
        return self.hours_per_patch


# =============================================================================
# Tregenza Sky Subdivision (145 patches)
# =============================================================================

def generate_tregenza_patches() -> SkyPatches:
    """
    Generate the 145 Tregenza sky patch center directions.
    
    The Tregenza subdivision divides the sky hemisphere into 145 patches
    arranged in 8 altitude bands. This is the standard sky discretization
    used in Radiance (genskyvec), EnergyPlus, DAYSIM, and Ladybug Tools.
    
    Returns:
        SkyPatches object with patch data
    
    Example:
        >>> patches = generate_tregenza_patches()
        >>> print(f"Number of patches: {patches.n_patches}")  # 145
        >>> print(f"Total solid angle: {patches.solid_angles.sum():.4f}")  # ~2π
    """
    patches = []
    directions = []
    solid_angles = []
    
    for band_idx, (elev_center, n_patches) in enumerate(TREGENZA_BANDS):
        elev_rad = np.deg2rad(elev_center)
        cos_elev = np.cos(elev_rad)
        sin_elev = np.sin(elev_rad)
        
        # Solid angle calculation for band
        elev_low = TREGENZA_BAND_BOUNDARIES[band_idx]
        elev_high = TREGENZA_BAND_BOUNDARIES[band_idx + 1]
        
        # Solid angle of band = 2π × (sin(θ_high) - sin(θ_low))
        band_solid_angle = 2 * np.pi * (
            np.sin(np.deg2rad(elev_high)) - np.sin(np.deg2rad(elev_low))
        )
        patch_solid_angle = band_solid_angle / n_patches
        
        for i in range(n_patches):
            # Azimuth at patch center (0° = North, clockwise)
            az_deg = (i + 0.5) * 360.0 / n_patches
            az_rad = np.deg2rad(az_deg)
            
            # Direction vector (x=East, y=North, z=Up)
            dx = cos_elev * np.sin(az_rad)  # East component
            dy = cos_elev * np.cos(az_rad)  # North component
            dz = sin_elev                    # Up component
            
            patches.append((az_deg, elev_center))
            directions.append((dx, dy, dz))
            solid_angles.append(patch_solid_angle)
    
    return SkyPatches(
        method="tregenza",
        n_patches=145,
        patches=np.array(patches, dtype=np.float64),
        directions=np.array(directions, dtype=np.float64),
        solid_angles=np.array(solid_angles, dtype=np.float64),
        metadata={"bands": 8}
    )


@njit(cache=True)
def get_tregenza_patch_index(azimuth_deg: float, elevation_deg: float) -> int:
    """
    Get the Tregenza patch index for a given sun position.
    
    Numba-accelerated for fast binning of many sun positions.
    
    Args:
        azimuth_deg: Solar azimuth in degrees (0-360, 0=North, clockwise)
        elevation_deg: Solar elevation in degrees (0-90)
    
    Returns:
        Patch index (0-144), or -1 if below horizon
    """
    if elevation_deg < 0.0:
        return -1
    
    # Band boundaries and patch counts
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

def generate_reinhart_patches(mf: int = 4) -> SkyPatches:
    """
    Generate Reinhart sky patches (subdivided Tregenza).
    
    The Reinhart subdivision increases resolution by subdividing each Tregenza
    band by a multiplication factor (MF). This allows higher accuracy for
    detailed solar studies.
    
    Args:
        mf: Multiplication factor. Common values:
            - MF=1: 145 patches (same as Tregenza)
            - MF=2: 577 patches
            - MF=4: 2305 patches (common for annual daylight simulation)
            - MF=6: 5185 patches
    
    Returns:
        SkyPatches object with patch data
    
    Example:
        >>> patches = generate_reinhart_patches(mf=4)
        >>> print(f"Number of patches: {patches.n_patches}")  # ~2305
    
    References:
        Reinhart, C.F. & Walkenhorst, O. (2001). Energy and Buildings.
    """
    mf = max(1, int(mf))
    patches = []
    directions = []
    solid_angles = []
    
    for band_idx, (elev_center_base, n_patches_base) in enumerate(TREGENZA_BANDS):
        # Subdivide elevation bands
        n_sub_bands = mf
        
        elev_low = TREGENZA_BAND_BOUNDARIES[band_idx]
        elev_high = TREGENZA_BAND_BOUNDARIES[band_idx + 1]
        elev_range = elev_high - elev_low
        
        for sub_band in range(n_sub_bands):
            # Sub-band elevation bounds
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
                # Zenith: reduce patches for inner rings
                n_az = max(1, n_patches_base * mf * (sub_band + 1) // n_sub_bands)
            else:
                n_az = n_patches_base * mf
            
            patch_solid_angle = sub_band_solid_angle / n_az
            
            for i in range(n_az):
                az_deg = (i + 0.5) * 360.0 / n_az
                az_rad = np.deg2rad(az_deg)
                
                dx = cos_elev * np.sin(az_rad)
                dy = cos_elev * np.cos(az_rad)
                dz = sin_elev
                
                patches.append((az_deg, sub_elev_center))
                directions.append((dx, dy, dz))
                solid_angles.append(patch_solid_angle)
    
    return SkyPatches(
        method="reinhart",
        n_patches=len(patches),
        patches=np.array(patches, dtype=np.float64),
        directions=np.array(directions, dtype=np.float64),
        solid_angles=np.array(solid_angles, dtype=np.float64),
        metadata={"mf": mf}
    )


# =============================================================================
# Uniform Grid Subdivision
# =============================================================================

def generate_uniform_grid_patches(
    n_azimuth: int = 36,
    n_elevation: int = 9
) -> SkyPatches:
    """
    Generate uniform grid sky patches.
    
    Simple subdivision with equal azimuth and elevation spacing.
    Note: This creates non-equal solid angle patches (smaller near zenith).
    Useful when uniform angular sampling is preferred over uniform area.
    
    Args:
        n_azimuth: Number of azimuth divisions (default: 36 = 10° spacing)
        n_elevation: Number of elevation divisions (default: 9 = 10° spacing)
    
    Returns:
        SkyPatches object with patch data
    
    Example:
        >>> patches = generate_uniform_grid_patches(36, 9)
        >>> print(f"Number of patches: {patches.n_patches}")  # 324
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
            
            dx = cos_elev * np.sin(az_rad)
            dy = cos_elev * np.cos(az_rad)
            dz = sin_elev
            
            patches.append((az_center, elev_center))
            directions.append((dx, dy, dz))
            solid_angles.append(patch_solid_angle)
    
    return SkyPatches(
        method="uniform",
        n_patches=len(patches),
        patches=np.array(patches, dtype=np.float64),
        directions=np.array(directions, dtype=np.float64),
        solid_angles=np.array(solid_angles, dtype=np.float64),
        metadata={"n_azimuth": n_azimuth, "n_elevation": n_elevation}
    )


# =============================================================================
# Fibonacci Spiral (Quasi-Uniform)
# =============================================================================

def generate_fibonacci_patches(n_patches: int = 145) -> SkyPatches:
    """
    Generate quasi-uniform sky patches using Fibonacci spiral.
    
    Uses the golden angle spiral to distribute points nearly uniformly
    on the hemisphere. This provides more uniform patch areas than
    regular grids with fewer total patches.
    
    Args:
        n_patches: Number of patches to generate (default: 145 to match Tregenza)
    
    Returns:
        SkyPatches object with patch data
    
    Example:
        >>> patches = generate_fibonacci_patches(200)
        >>> # Check uniformity: solid angles should be equal
        >>> print(f"Solid angle std: {patches.solid_angles.std():.6f}")  # ~0
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
        
        # Direction vector (x=East, y=North, z=Up)
        r = np.sqrt(1.0 - z * z)
        dx = r * np.sin(azimuth_rad)  # Note: using sin for East, cos for North
        dy = r * np.cos(azimuth_rad)
        dz = z
        
        patches.append((azimuth_deg, elevation_deg))
        directions.append((dx, dy, dz))
        solid_angles.append(patch_solid_angle)
    
    return SkyPatches(
        method="fibonacci",
        n_patches=n,
        patches=np.array(patches, dtype=np.float64),
        directions=np.array(directions, dtype=np.float64),
        solid_angles=np.array(solid_angles, dtype=np.float64),
        metadata={"golden_angle": golden_angle}
    )


# =============================================================================
# Sky Patch Generation (Unified Interface)
# =============================================================================

def generate_sky_patches(
    method: str = "tregenza",
    **kwargs
) -> SkyPatches:
    """
    Generate sky patches using specified discretization method.
    
    This is the main entry point for sky discretization. It dispatches
    to the appropriate method-specific function.
    
    Args:
        method: Discretization method:
            - "tregenza": 145 patches (standard, fast)
            - "reinhart": Tregenza × MF² (high-resolution)
            - "uniform": Regular grid (simple)
            - "fibonacci": Quasi-uniform spiral (balanced)
        **kwargs: Method-specific parameters:
            - mf: Multiplication factor for Reinhart (default: 4)
            - n_azimuth, n_elevation: Grid size for uniform
            - n_patches: Number of patches for Fibonacci
    
    Returns:
        SkyPatches object with patch data
    
    Example:
        >>> # Standard Tregenza
        >>> patches = generate_sky_patches("tregenza")
        
        >>> # High-resolution Reinhart
        >>> patches = generate_sky_patches("reinhart", mf=4)
        
        >>> # Custom uniform grid
        >>> patches = generate_sky_patches("uniform", n_azimuth=72, n_elevation=18)
    """
    method = method.lower()
    
    if method == "tregenza":
        return generate_tregenza_patches()
    elif method == "reinhart":
        mf = kwargs.get("mf", 4)
        return generate_reinhart_patches(mf=mf)
    elif method == "uniform":
        n_az = kwargs.get("n_azimuth", 36)
        n_el = kwargs.get("n_elevation", 9)
        return generate_uniform_grid_patches(n_az, n_el)
    elif method == "fibonacci":
        n = kwargs.get("n_patches", 145)
        return generate_fibonacci_patches(n_patches=n)
    else:
        raise ValueError(
            f"Unknown sky discretization method: {method}. "
            f"Supported: tregenza, reinhart, uniform, fibonacci"
        )


# =============================================================================
# Sun Position Binning
# =============================================================================

def bin_sun_positions_to_patches(
    azimuth_arr: np.ndarray,
    elevation_arr: np.ndarray,
    dni_arr: np.ndarray,
    dhi_arr: Optional[np.ndarray] = None,
    method: str = "tregenza",
    **kwargs
) -> BinnedSolarData:
    """
    Bin hourly sun positions into sky patches and aggregate radiation.
    
    This is the key optimization for cumulative solar irradiance: instead of
    tracing rays for every hourly sun position, aggregate radiation values
    for each sky patch and trace rays once per patch.
    
    The DNI values are summed for each patch where the sun appears.
    The DHI values are distributed isotropically across all patches.
    
    Args:
        azimuth_arr: Array of solar azimuth values in degrees (0=North)
        elevation_arr: Array of solar elevation values in degrees
        dni_arr: Array of Direct Normal Irradiance values (W/m² or Wh/m²)
        dhi_arr: Array of Diffuse Horizontal Irradiance values (optional)
        method: Sky discretization method
        **kwargs: Additional parameters for patch generation
    
    Returns:
        BinnedSolarData with accumulated radiation per patch
    
    Example:
        >>> from palm_solar.epw import prepare_cumulative_simulation_input
        >>> az, el, dni, dhi, loc = prepare_cumulative_simulation_input("weather.epw")
        >>> binned = bin_sun_positions_to_patches(az, el, dni, dhi)
        >>> print(f"Active patches: {(binned.hours_per_patch > 0).sum()}")
    """
    # VoxCity compatibility: allow positional method as the 4th argument
    # (VoxCity signature is (azimuth_arr, elevation_arr, dni_arr, method='tregenza', **kwargs))
    if isinstance(dhi_arr, str):
        method = dhi_arr
        dhi_arr = None

    method = str(method).lower()

    # Generate sky patches
    sky_patches = generate_sky_patches(method, **kwargs)
    
    n_patches = sky_patches.n_patches
    cumulative_dni = np.zeros(n_patches, dtype=np.float64)
    hours_count = np.zeros(n_patches, dtype=np.int32)
    
    # Bin each sun position
    n_hours = len(azimuth_arr)
    
    if method == "tregenza":
        # Use fast Numba-accelerated binning for Tregenza
        for i in range(n_hours):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            patch_idx = get_tregenza_patch_index(az, elev)
            if patch_idx >= 0:
                cumulative_dni[patch_idx] += dni
                hours_count[patch_idx] += 1
    else:
        # For other methods, find nearest patch by direction
        directions = sky_patches.directions
        
        for i in range(n_hours):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            # Convert sun position to direction vector
            elev_rad = np.deg2rad(elev)
            az_rad = np.deg2rad(az)
            sun_dir = np.array([
                np.cos(elev_rad) * np.sin(az_rad),  # East
                np.cos(elev_rad) * np.cos(az_rad),  # North
                np.sin(elev_rad)                     # Up
            ])
            
            # Find nearest patch by dot product
            dots = np.sum(directions * sun_dir, axis=1)
            patch_idx = np.argmax(dots)
            
            cumulative_dni[patch_idx] += dni
            hours_count[patch_idx] += 1
    
    # Sum total DHI if provided
    total_dhi = np.sum(dhi_arr) if dhi_arr is not None else 0.0
    
    # Total daytime hours
    total_daytime = np.sum(elevation_arr > 0)
    
    return BinnedSolarData(
        sky_patches=sky_patches,
        cumulative_dni=cumulative_dni,
        cumulative_dhi=total_dhi,
        hours_per_patch=hours_count,
        total_daytime_hours=int(total_daytime)
    )


@njit(cache=True, parallel=False)
def _bin_tregenza_fast(
    azimuth_arr: np.ndarray,
    elevation_arr: np.ndarray,
    dni_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast binning of sun positions to Tregenza patches using Numba.
    
    Args:
        azimuth_arr: Solar azimuth array (degrees)
        elevation_arr: Solar elevation array (degrees)
        dni_arr: DNI values array
    
    Returns:
        Tuple of (cumulative_dni, hours_count) arrays
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
        
        patch_idx = get_tregenza_patch_index(az, elev)
        if patch_idx >= 0:
            cumulative_dni[patch_idx] += dni
            hours_count[patch_idx] += 1
    
    return cumulative_dni, hours_count


# =============================================================================
# Utility Functions
# =============================================================================

def get_patch_info(method: str = "tregenza", **kwargs) -> dict:
    """
    Get information about a sky discretization method.
    
    Args:
        method: Sky discretization method
        **kwargs: Method-specific parameters
    
    Returns:
        Dictionary with method details
    
    Example:
        >>> info = get_patch_info("reinhart", mf=4)
        >>> print(f"{info['method']}: {info['n_patches']} patches")
    """
    patches = generate_sky_patches(method, **kwargs)
    
    info = {
        "method": patches.method,
        "n_patches": patches.n_patches,
        "total_solid_angle": patches.solid_angles.sum(),
        "metadata": patches.metadata
    }
    
    # Add method-specific descriptions
    if method.lower() == "tregenza":
        info["description"] = "Standard 145-patch subdivision (Radiance, DAYSIM)"
        info["reference"] = "Tregenza (1987)"
    elif method.lower() == "reinhart":
        info["description"] = f"High-resolution subdivision with MF={kwargs.get('mf', 4)}"
        info["reference"] = "Reinhart & Walkenhorst (2001)"
    elif method.lower() == "uniform":
        info["description"] = f"Regular grid with {kwargs.get('n_azimuth', 36)}×{kwargs.get('n_elevation', 9)} patches"
    elif method.lower() == "fibonacci":
        info["description"] = "Quasi-uniform distribution using golden angle"
    
    return info


def calculate_cumulative_irradiance_weights(
    binned_data: BinnedSolarData,
    include_diffuse: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate patch weights for cumulative irradiance simulation.
    
    Returns weights that can be used with ray tracing results to compute
    cumulative irradiance. The direct component uses binned DNI values,
    and the diffuse component is distributed isotropically.
    
    Args:
        binned_data: BinnedSolarData from bin_sun_positions_to_patches
        include_diffuse: Whether to include diffuse component
    
    Returns:
        Tuple of:
        - direct_weights: DNI weight per patch (Wh/m²)
        - diffuse_weights: DHI weight per patch (Wh/m²)
    
    Example:
        >>> binned = bin_sun_positions_to_patches(az, el, dni, dhi)
        >>> direct_w, diffuse_w = calculate_cumulative_irradiance_weights(binned)
        >>> # Use with ray tracing:
        >>> # cumulative_irradiance = sum(visibility * direct_w + svf * diffuse_w)
    """
    sky_patches = binned_data.sky_patches
    
    # Direct weights are simply the cumulative DNI per patch
    direct_weights = binned_data.cumulative_dni.copy()
    
    # Diffuse weights: distribute total DHI by solid angle fraction
    if include_diffuse and binned_data.cumulative_dhi > 0:
        # Each patch receives DHI proportional to its solid angle
        # The solid angle fraction represents how much of the hemisphere it covers
        solid_angle_fraction = sky_patches.solid_angles / sky_patches.solid_angles.sum()
        diffuse_weights = binned_data.cumulative_dhi * solid_angle_fraction
    else:
        diffuse_weights = np.zeros(sky_patches.n_patches, dtype=np.float64)
    
    return direct_weights, diffuse_weights


def visualize_sky_patches(
    method: str = "tregenza",
    ax=None,
    show: bool = True,
    **kwargs
):
    """
    Visualize sky patches on a polar plot.
    
    Args:
        method: Sky discretization method
        show: Whether to call plt.show()
        **kwargs: Method-specific parameters
    
    Returns:
        matplotlib axis object
    """
    import matplotlib.pyplot as plt
    
    patches = generate_sky_patches(method, **kwargs)
    
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convert to polar coordinates (theta=azimuth, r=90-elevation)
    theta = np.deg2rad(patches.patches[:, 0])
    r = 90.0 - patches.patches[:, 1]  # Zenith at center
    
    # Color by solid angle
    colors = patches.solid_angles / patches.solid_angles.max()
    
    scatter = ax.scatter(theta, r, c=colors, s=10, alpha=0.7, cmap='viridis')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax.set_title(f"{method.capitalize()} Sky Patches (n={patches.n_patches})")
    
    plt.colorbar(scatter, ax=ax, label='Relative Solid Angle')
    
    if show:
        plt.show()
    
    return ax


# =============================================================================
# VoxCity API Compatibility Aliases
# =============================================================================

# Alias for VoxCity compatibility - the function is already Numba-accelerated
get_tregenza_patch_index_fast = get_tregenza_patch_index


def bin_sun_positions_to_tregenza_fast(
    azimuth_arr: np.ndarray,
    elevation_arr: np.ndarray,
    dni_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated binning of sun positions to Tregenza patches.
    
    This function matches the signature of voxcity.simulator.solar.sky.bin_sun_positions_to_tregenza_fast.
    
    Args:
        azimuth_arr: Array of solar azimuth values in degrees
        elevation_arr: Array of solar elevation values in degrees
        dni_arr: Array of Direct Normal Irradiance values (W/m²)
    
    Returns:
        Tuple of (cumulative_dni, hours_count) arrays:
        - cumulative_dni: shape (145,) - Cumulative DNI (W·h/m²) for each Tregenza patch
        - hours_count: shape (145,) - Number of hours with sun in each patch
    """
    return _bin_tregenza_fast(azimuth_arr, elevation_arr, dni_arr)


def visualize_binned_radiation(
    binned_data: BinnedSolarData,
    show: bool = True
):
    """
    Visualize binned solar radiation on a polar plot.
    
    Args:
        binned_data: BinnedSolarData from bin_sun_positions_to_patches
        show: Whether to call plt.show()
    
    Returns:
        matplotlib axis object
    """
    import matplotlib.pyplot as plt
    
    patches = binned_data.sky_patches
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Convert to polar coordinates
    theta = np.deg2rad(patches.patches[:, 0])
    r = 90.0 - patches.patches[:, 1]
    
    # Color by cumulative DNI (log scale for better visualization)
    dni = binned_data.cumulative_dni
    dni_log = np.log10(dni + 1)  # Add 1 to avoid log(0)
    
    # Size by hours
    sizes = 10 + binned_data.hours_per_patch * 2
    
    scatter = ax.scatter(theta, r, c=dni_log, s=sizes, alpha=0.7, cmap='hot')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax.set_title(
        f"Binned Solar Radiation ({patches.method.capitalize()})\n"
        f"Total Hours: {binned_data.total_daytime_hours}"
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='log₁₀(DNI + 1) [Wh/m²]')
    
    if show:
        plt.show()
    
    return ax
