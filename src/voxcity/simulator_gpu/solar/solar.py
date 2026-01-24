"""
Solar position calculation for palm-solar.

Based on PALM's calc_zenith subroutine (radiation_model_mod.f90 lines 7965-8012).
Computes solar declination, hour angle, zenith angle, and direction vector.

PALM Alignment:
- Declination formula: ASIN(decl_1 * SIN(decl_2 * day_of_year - decl_3))
- Hour angle formula: 2π * (second_of_day / 86400) + longitude - π
- cos_zenith: sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(hour_angle)
- Sun direction: Computed from declination and hour angle

All constants match PALM exactly:
- decl_1 = sin(23.45°) = 0.39794968147687266
- decl_2 = 2π/365 = 0.017214206321039962
- decl_3 = decl_2 * 81 = 1.3943507120042368 (vernal equinox offset)
"""

import taichi as ti
import math
from datetime import datetime, timezone
from typing import Tuple, Optional
from dataclasses import dataclass

from .core import Vector3, Point3, DEG_TO_RAD, RAD_TO_DEG, PI, TWO_PI


# Constants for solar declination calculation (matching PALM exactly)
# PALM: decl_1 = SIN( 23.45_wp * pi / 180.0_wp )
# PALM: decl_2 = 2.0_wp * pi / 365.0_wp
# PALM: decl_3 = decl_2 * 81.0_wp  (offset for vernal equinox ~March 21)
DECL_1 = 0.39794968147687266  # sin(23.45 * pi / 180)
DECL_2 = 0.017214206321039962  # 2 * pi / 365
DECL_3 = 1.3943507120042368    # DECL_2 * 81

# Seconds per day
SECONDS_PER_DAY = 86400.0


@dataclass
class SolarPosition:
    """
    Solar position data.
    
    Attributes:
        cos_zenith: Cosine of solar zenith angle (0 at horizon, 1 at zenith)
        zenith_angle: Solar zenith angle in degrees
        azimuth_angle: Solar azimuth angle in degrees (0 = North, 90 = East)
        elevation_angle: Solar elevation angle in degrees (0 = horizon, 90 = zenith)
        direction: Unit vector pointing towards the sun in VoxCity grid coordinates
                   (x = row/South direction, y = col/East direction, z = up)
        sun_up: True if sun is above horizon
    """
    cos_zenith: float
    zenith_angle: float
    azimuth_angle: float
    elevation_angle: float
    direction: Tuple[float, float, float]
    sun_up: bool


def calc_zenith(
    day_of_year: int,
    second_of_day: float,
    latitude: float,
    longitude: float
) -> SolarPosition:
    """
    Calculate solar position.
    
    Based on PALM's calc_zenith subroutine.
    
    Args:
        day_of_year: Day number (1-365)
        second_of_day: Seconds since midnight UTC
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
    
    Returns:
        SolarPosition with all solar geometry data
    """
    # Convert to radians
    lat = latitude * DEG_TO_RAD
    lon = longitude * DEG_TO_RAD
    
    # Solar declination angle
    declination = math.asin(DECL_1 * math.sin(DECL_2 * day_of_year - DECL_3))
    
    # Hour angle (solar noon at lon=0 is at 12:00 UTC)
    hour_angle = TWO_PI * (second_of_day / SECONDS_PER_DAY) + lon - PI
    
    # Cosine of zenith angle
    cos_zenith = (math.sin(lat) * math.sin(declination) + 
                  math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
    cos_zenith = max(0.0, cos_zenith)
    
    # Zenith and elevation angles
    zenith_angle = math.acos(min(1.0, cos_zenith)) * RAD_TO_DEG
    elevation_angle = 90.0 - zenith_angle
    
    # Solar direction vector in ENU coordinates (intermediate calculation)
    # Direction in longitudes (East component) = -sin(hour_angle) * cos(declination)
    sun_dir_lon = -math.sin(hour_angle) * math.cos(declination)
    
    # Direction in latitudes (North component)
    sun_dir_lat = (math.sin(declination) * math.cos(lat) - 
                   math.cos(hour_angle) * math.cos(declination) * math.sin(lat))
    
    # Normalize to get unit vector pointing toward sun
    sin_zenith = math.sqrt(1.0 - cos_zenith**2) if cos_zenith < 1.0 else 0.0
    
    if sin_zenith > 1e-10:
        # Horizontal components in ENU (x=East, y=North, z=Up)
        sun_x_enu = sun_dir_lon  # East component
        sun_y_enu = sun_dir_lat  # North component
        sun_z = cos_zenith       # Up component
        
        # Normalize
        length = math.sqrt(sun_x_enu**2 + sun_y_enu**2 + sun_z**2)
        if length > 1e-10:
            sun_x_enu /= length
            sun_y_enu /= length
            sun_z /= length
    else:
        # Sun at zenith
        sun_x_enu = 0.0
        sun_y_enu = 0.0
        sun_z = 1.0
    
    # Azimuth angle (0 = North, 90 = East) - computed from ENU coordinates
    azimuth_angle = math.atan2(sun_x_enu, sun_y_enu) * RAD_TO_DEG
    if azimuth_angle < 0:
        azimuth_angle += 360.0
    
    # Convert direction from ENU to VoxCity grid-index coordinates:
    # VoxCity grid: i (row) increases North->South, j (col) increases West->East
    # Grid-index: x = i direction = South = -North, y = j direction = East
    # Conversion: grid_x = -enu_y (North to South), grid_y = enu_x (East)
    sun_x = -sun_y_enu  # Grid x = -North = South direction
    sun_y = sun_x_enu   # Grid y = East direction
    
    sun_up = cos_zenith > 0.0
    
    return SolarPosition(
        cos_zenith=cos_zenith,
        zenith_angle=zenith_angle,
        azimuth_angle=azimuth_angle,
        elevation_angle=elevation_angle,
        direction=(sun_x, sun_y, sun_z),
        sun_up=sun_up
    )


def calc_solar_position_datetime(
    dt: datetime,
    latitude: float,
    longitude: float
) -> SolarPosition:
    """
    Calculate solar position from datetime.
    
    Args:
        dt: Datetime (should be in UTC or timezone-aware)
        latitude: Latitude in degrees
        longitude: Longitude in degrees
    
    Returns:
        SolarPosition
    """
    # Convert to UTC if timezone-aware
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    
    # Day of year (1-365)
    day_of_year = dt.timetuple().tm_yday
    
    # Seconds since midnight UTC
    second_of_day = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
    
    return calc_zenith(day_of_year, second_of_day, latitude, longitude)


def get_day_of_year(year: int, month: int, day: int) -> int:
    """Get day of year from date."""
    return datetime(year, month, day).timetuple().tm_yday


@ti.func
def calc_zenith_ti(
    day_of_year: ti.i32,
    second_of_day: ti.f32,
    lat_rad: ti.f32,
    lon_rad: ti.f32
) -> ti.math.vec4:
    """
    Taichi function to calculate solar position.
    
    Args:
        day_of_year: Day number (1-365)
        second_of_day: Seconds since midnight UTC
        lat_rad: Latitude in radians
        lon_rad: Longitude in radians
    
    Returns:
        vec4(cos_zenith, sun_x, sun_y, sun_z)
    """
    # Solar declination
    declination = ti.asin(0.409093 * ti.sin(0.0172028 * day_of_year - 1.39012))
    
    # Hour angle
    hour_angle = 6.283185 * (second_of_day / 86400.0) + lon_rad - 3.141593
    
    # Cosine of zenith
    cos_zenith = (ti.sin(lat_rad) * ti.sin(declination) + 
                  ti.cos(lat_rad) * ti.cos(declination) * ti.cos(hour_angle))
    cos_zenith = ti.max(0.0, cos_zenith)
    
    # Direction components in ENU (x=East, y=North)
    sun_dir_lon = -ti.sin(hour_angle) * ti.cos(declination)
    sun_dir_lat = (ti.sin(declination) * ti.cos(lat_rad) - 
                   ti.cos(hour_angle) * ti.cos(declination) * ti.sin(lat_rad))
    
    # Normalize ENU components
    sun_x_enu = sun_dir_lon
    sun_y_enu = sun_dir_lat
    sun_z = cos_zenith
    length = ti.sqrt(sun_x_enu**2 + sun_y_enu**2 + sun_z**2)
    
    if length > 1e-10:
        sun_x_enu /= length
        sun_y_enu /= length
        sun_z /= length
    else:
        sun_x_enu = 0.0
        sun_y_enu = 0.0
        sun_z = 1.0
    
    # Convert from ENU to VoxCity grid-index coordinates:
    # Grid x = -North = South direction, Grid y = East direction
    sun_x = -sun_y_enu
    sun_y = sun_x_enu
    
    return ti.math.vec4(cos_zenith, sun_x, sun_y, sun_z)


@ti.data_oriented
class SolarCalculator:
    """
    GPU-accelerated solar position calculator.
    
    Pre-computes solar positions for all time steps.
    """
    
    def __init__(self, latitude: float, longitude: float):
        """
        Initialize solar calculator.
        
        Args:
            latitude: Site latitude in degrees
            longitude: Site longitude in degrees
        """
        self.latitude = latitude
        self.longitude = longitude
        self.lat_rad = latitude * DEG_TO_RAD
        self.lon_rad = longitude * DEG_TO_RAD
        
        # Current solar position (stored as fields for GPU access)
        self.cos_zenith = ti.field(dtype=ti.f32, shape=())
        self.sun_direction = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.sun_up = ti.field(dtype=ti.i32, shape=())
        
        # Initialize
        self.cos_zenith[None] = 0.0
        self.sun_direction[None] = Vector3(0.0, 0.0, 1.0)
        self.sun_up[None] = 0
    
    def update(self, day_of_year: int, second_of_day: float):
        """Update solar position for given time."""
        pos = calc_zenith(day_of_year, second_of_day, self.latitude, self.longitude)
        self.cos_zenith[None] = pos.cos_zenith
        self.sun_direction[None] = Vector3(*pos.direction)
        self.sun_up[None] = 1 if pos.sun_up else 0
    
    def update_datetime(self, dt: datetime):
        """Update solar position for given datetime."""
        pos = calc_solar_position_datetime(dt, self.latitude, self.longitude)
        self.cos_zenith[None] = pos.cos_zenith
        self.sun_direction[None] = Vector3(*pos.direction)
        self.sun_up[None] = 1 if pos.sun_up else 0
    
    @ti.func
    def get_cos_zenith(self) -> ti.f32:
        """Get current cosine of zenith angle."""
        return self.cos_zenith[None]
    
    @ti.func
    def get_sun_direction(self) -> Vector3:
        """Get current sun direction unit vector."""
        return self.sun_direction[None]
    
    @ti.func
    def is_sun_up(self) -> ti.i32:
        """Check if sun is above horizon."""
        return self.sun_up[None]


def discretize_sky_directions(
    n_azimuth: int = 80,
    n_elevation: int = 40
) -> Tuple[list, list]:
    """
    Generate discretized sky directions for SVF/ray tracing.
    
    Args:
        n_azimuth: Number of azimuthal divisions
        n_elevation: Number of elevation divisions (full hemisphere)
    
    Returns:
        Tuple of (directions, solid_angles) where:
        - directions: List of (x, y, z) unit vectors
        - solid_angles: List of solid angle weights
    """
    directions = []
    solid_angles = []
    
    # Only upper hemisphere (elevation from 0 to 90)
    for i_elev in range(n_elevation // 2):
        # Center of elevation band
        elev_low = (i_elev / n_elevation) * PI
        elev_high = ((i_elev + 1) / n_elevation) * PI
        elevation = (elev_low + elev_high) / 2
        
        # Solid angle for this band
        d_omega = (2 * PI / n_azimuth) * (math.cos(elev_low) - math.cos(elev_high))
        
        for i_azim in range(n_azimuth):
            # Center of azimuth band
            azimuth = (i_azim + 0.5) * (2 * PI / n_azimuth)
            
            # Convert to Cartesian
            cos_elev = math.cos(elevation)
            sin_elev = math.sin(elevation)
            x = sin_elev * math.sin(azimuth)
            y = sin_elev * math.cos(azimuth)
            z = cos_elev
            
            directions.append((x, y, z))
            solid_angles.append(d_omega)
    
    return directions, solid_angles
