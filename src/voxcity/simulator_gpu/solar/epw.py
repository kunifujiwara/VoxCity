"""
EPW (EnergyPlus Weather) File Processing for palm_solar.

This module provides utilities for reading EPW weather files and extracting
solar radiation data for cumulative irradiance simulations.

EPW files contain hourly weather data including:
- Direct Normal Irradiance (DNI)
- Diffuse Horizontal Irradiance (DHI)
- Global Horizontal Irradiance (GHI)
- Location metadata (latitude, longitude, timezone, elevation)

References:
- EnergyPlus Weather File Format: https://energyplus.net/weather
- EPW Data Dictionary: https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class EPWLocation:
    """
    Location metadata from EPW file header.
    
    Attributes:
        city: City name
        state_province: State or province
        country: Country code
        data_source: Weather data source
        wmo_station: WMO station identifier
        latitude: Latitude in degrees (positive = North)
        longitude: Longitude in degrees (positive = East)
        timezone: UTC offset in hours (e.g., -5 for EST)
        elevation: Elevation above sea level in meters
    """
    city: str
    state_province: str
    country: str
    data_source: str
    wmo_station: str
    latitude: float
    longitude: float
    timezone: float
    elevation: float


@dataclass
class EPWSolarData:
    """
    Solar radiation data extracted from EPW file for simulation.
    
    Attributes:
        timestamps: Array of datetime-like indices (hourly)
        dni: Direct Normal Irradiance (W/m²)
        dhi: Diffuse Horizontal Irradiance (W/m²)
        ghi: Global Horizontal Irradiance (W/m²), optional
        location: EPWLocation with site metadata
        day_of_year: Array of day numbers (1-365)
        hour: Array of hour values (0-23)
        year: Year from EPW file
    """
    timestamps: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    ghi: Optional[np.ndarray]
    location: EPWLocation
    day_of_year: np.ndarray
    hour: np.ndarray
    year: int


def read_epw_header(epw_path: Union[str, Path]) -> EPWLocation:
    """
    Read location metadata from EPW file header.
    
    Args:
        epw_path: Path to EPW file
    
    Returns:
        EPWLocation with site metadata
    
    Raises:
        FileNotFoundError: If EPW file doesn't exist
        ValueError: If LOCATION line cannot be parsed
    """
    epw_path_obj = Path(epw_path)
    if not epw_path_obj.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    
    with open(epw_path_obj, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith("LOCATION"):
                parts = line.strip().split(',')
                if len(parts) < 10:
                    raise ValueError(f"Invalid LOCATION line in EPW file: {line}")
                
                return EPWLocation(
                    city=parts[1].strip(),
                    state_province=parts[2].strip(),
                    country=parts[3].strip(),
                    data_source=parts[4].strip(),
                    wmo_station=parts[5].strip(),
                    latitude=float(parts[6]),
                    longitude=float(parts[7]),
                    timezone=float(parts[8]),
                    elevation=float(parts[9])
                )
    
    raise ValueError("Could not find LOCATION line in EPW file")


def read_epw_solar_data(
    epw_path: Union[str, Path],
    start_month: Optional[int] = None,
    end_month: Optional[int] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None
) -> EPWSolarData:
    """
    Read solar radiation data from EPW file.
    
    This function extracts DNI, DHI, and GHI values along with temporal
    information needed for solar position calculations.
    
    Args:
        epw_path: Path to EPW file
        start_month: Filter to start from this month (1-12)
        end_month: Filter to end at this month (1-12)
        start_day: Filter to start from this day (1-31)
        end_day: Filter to end at this day (1-31)
    
    Returns:
        EPWSolarData containing radiation data and location metadata
    
    Example:
        >>> solar_data = read_epw_solar_data("weather.epw", start_month=6, end_month=6)
        >>> print(f"June DNI max: {solar_data.dni.max():.1f} W/m²")
    """
    epw_path_obj = Path(epw_path)
    if not epw_path_obj.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    
    # Read location header
    location = read_epw_header(epw_path)
    
    # Read weather data (starts at line 9, 0-indexed line 8)
    with open(epw_path_obj, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Find data start (after 8 header lines)
    data_start_index = 8
    
    # Parse weather data
    data_rows = []
    for line in lines[data_start_index:]:
        parts = line.strip().split(',')
        if len(parts) < 22:  # Need at least 22 columns for radiation data
            continue
        
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3]) - 1  # EPW hours are 1-24, convert to 0-23
            
            # Apply date filters
            if start_month is not None and month < start_month:
                continue
            if end_month is not None and month > end_month:
                continue
            if start_month is not None and end_month is not None:
                if start_month == end_month:
                    if start_day is not None and day < start_day:
                        continue
                    if end_day is not None and day > end_day:
                        continue
            
            # Radiation columns (from EPW specification):
            # Column 11 (index 10): Global Horizontal Radiation (Wh/m²)
            # Column 14 (index 13): Direct Normal Radiation (Wh/m²)
            # Column 15 (index 14): Diffuse Horizontal Radiation (Wh/m²)
            ghi = float(parts[13])  # Global Horizontal Radiation
            dni = float(parts[14])  # Direct Normal Radiation
            dhi = float(parts[15])  # Diffuse Horizontal Radiation
            
            data_rows.append({
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'dni': dni,
                'dhi': dhi,
                'ghi': ghi
            })
        except (ValueError, IndexError):
            continue
    
    if not data_rows:
        raise ValueError("No valid weather data found in EPW file")
    
    # Convert to arrays
    n = len(data_rows)
    timestamps = np.empty(n, dtype='datetime64[h]')
    dni = np.zeros(n, dtype=np.float64)
    dhi = np.zeros(n, dtype=np.float64)
    ghi = np.zeros(n, dtype=np.float64)
    day_of_year = np.zeros(n, dtype=np.int32)
    hour = np.zeros(n, dtype=np.int32)
    
    for i, row in enumerate(data_rows):
        # Create timestamp
        timestamps[i] = np.datetime64(
            f"{row['year']:04d}-{row['month']:02d}-{row['day']:02d}T{row['hour']:02d}"
        )
        dni[i] = row['dni']
        dhi[i] = row['dhi']
        ghi[i] = row['ghi']
        
        # Calculate day of year
        import datetime
        dt = datetime.date(row['year'], row['month'], row['day'])
        day_of_year[i] = dt.timetuple().tm_yday
        hour[i] = row['hour']
    
    return EPWSolarData(
        timestamps=timestamps,
        dni=dni,
        dhi=dhi,
        ghi=ghi,
        location=location,
        day_of_year=day_of_year,
        hour=hour,
        year=data_rows[0]['year'] if data_rows else 2020
    )


def prepare_cumulative_simulation_input(
    epw_path: Union[str, Path],
    start_month: Optional[int] = None,
    end_month: Optional[int] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
    filter_daytime: bool = True,
    min_elevation_deg: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, EPWLocation]:
    """
    Prepare solar simulation input data from EPW file.
    
    This function reads EPW data and calculates solar positions for each
    timestep, filtering out nighttime hours. Returns data ready for sky
    patch binning and cumulative irradiance calculation.
    
    Args:
        epw_path: Path to EPW file
        start_month: Filter start month (1-12)
        end_month: Filter end month (1-12)
        start_day: Filter start day (1-31)
        end_day: Filter end day (1-31)
        filter_daytime: If True, filter out hours with sun below horizon
        min_elevation_deg: Minimum solar elevation to include (degrees)
    
    Returns:
        Tuple of:
        - azimuth: Solar azimuth angles (degrees, 0=North, clockwise)
        - elevation: Solar elevation angles (degrees, 0=horizon, 90=zenith)
        - dni: Direct Normal Irradiance (W/m²)
        - dhi: Diffuse Horizontal Irradiance (W/m²)
        - location: EPWLocation with site metadata
    
    Example:
        >>> az, el, dni, dhi, loc = prepare_cumulative_simulation_input("weather.epw")
        >>> print(f"Location: {loc.city}, {loc.country}")
        >>> print(f"Daytime hours: {len(az)}")
    """
    from .solar import calc_zenith
    
    # Read EPW data
    solar_data = read_epw_solar_data(
        epw_path, start_month, end_month, start_day, end_day
    )
    
    location = solar_data.location
    
    # Calculate solar positions for each timestep
    n = len(solar_data.dni)
    azimuth = np.zeros(n, dtype=np.float64)
    elevation = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        doy = solar_data.day_of_year[i]
        hr = solar_data.hour[i]
        
        # Convert local hour to UTC seconds
        # EPW data is in local standard time, need to convert to UTC
        utc_hour = hr - location.timezone
        second_of_day = utc_hour * 3600.0
        
        # Handle day wraparound
        if second_of_day < 0:
            second_of_day += 86400
        elif second_of_day >= 86400:
            second_of_day -= 86400
        
        # Calculate solar position
        pos = calc_zenith(doy, second_of_day, location.latitude, location.longitude)
        azimuth[i] = pos.azimuth_angle
        elevation[i] = pos.elevation_angle
    
    # Filter daytime hours if requested
    if filter_daytime:
        mask = elevation >= min_elevation_deg
        azimuth = azimuth[mask]
        elevation = elevation[mask]
        dni = solar_data.dni[mask]
        dhi = solar_data.dhi[mask]
    else:
        dni = solar_data.dni
        dhi = solar_data.dhi
    
    return azimuth, elevation, dni, dhi, location


def get_typical_days(
    epw_path: Union[str, Path],
    months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Extract typical day profiles from EPW file for each month.
    
    Calculates average hourly DNI and DHI for each month, useful for
    quick annual simulations using representative days.
    
    Args:
        epw_path: Path to EPW file
        months: List of months to process (default: all 12)
    
    Returns:
        DataFrame with columns: month, hour, dni_avg, dhi_avg, ghi_avg
    
    Example:
        >>> typical = get_typical_days("weather.epw", months=[6, 12])
        >>> june = typical[typical.month == 6]
    """
    solar_data = read_epw_solar_data(epw_path)
    
    if months is None:
        months = list(range(1, 13))
    
    results = []
    
    for month in months:
        for hour in range(24):
            # Find matching hours
            mask = (
                (np.array([int(str(t)[5:7]) for t in solar_data.timestamps]) == month) &
                (solar_data.hour == hour)
            )
            
            if np.any(mask):
                results.append({
                    'month': month,
                    'hour': hour,
                    'dni_avg': np.mean(solar_data.dni[mask]),
                    'dhi_avg': np.mean(solar_data.dhi[mask]),
                    'ghi_avg': np.mean(solar_data.ghi[mask]) if solar_data.ghi is not None else np.nan
                })
    
    return pd.DataFrame(results)


def estimate_annual_irradiance(epw_path: Union[str, Path]) -> dict:
    """
    Estimate annual solar irradiance statistics from EPW file.
    
    Provides quick overview of solar resource without full simulation.
    
    Args:
        epw_path: Path to EPW file
    
    Returns:
        Dictionary with annual statistics:
        - total_dni_kwh_m2: Annual cumulative DNI (kWh/m²)
        - total_dhi_kwh_m2: Annual cumulative DHI (kWh/m²)
        - total_ghi_kwh_m2: Annual cumulative GHI (kWh/m²)
        - peak_dni: Maximum hourly DNI (W/m²)
        - peak_ghi: Maximum hourly GHI (W/m²)
        - sunshine_hours: Hours with DNI > 120 W/m²
        - location: EPWLocation metadata
    
    Example:
        >>> stats = estimate_annual_irradiance("weather.epw")
        >>> print(f"Annual GHI: {stats['total_ghi_kwh_m2']:.0f} kWh/m²")
    """
    solar_data = read_epw_solar_data(epw_path)
    
    # Sum up Wh values and convert to kWh
    total_dni = np.sum(solar_data.dni) / 1000.0
    total_dhi = np.sum(solar_data.dhi) / 1000.0
    total_ghi = np.sum(solar_data.ghi) / 1000.0 if solar_data.ghi is not None else np.nan
    
    # Peak values
    peak_dni = np.max(solar_data.dni)
    peak_ghi = np.max(solar_data.ghi) if solar_data.ghi is not None else np.nan
    
    # Sunshine hours (typical threshold: 120 W/m² DNI)
    sunshine_hours = np.sum(solar_data.dni > 120)
    
    return {
        'total_dni_kwh_m2': total_dni,
        'total_dhi_kwh_m2': total_dhi,
        'total_ghi_kwh_m2': total_ghi,
        'peak_dni': peak_dni,
        'peak_ghi': peak_ghi,
        'sunshine_hours': int(sunshine_hours),
        'location': solar_data.location
    }
