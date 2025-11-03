from pathlib import Path
from typing import Tuple, Union
import pandas as pd


def process_epw(epw_path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Process an EPW file into a pandas DataFrame and header metadata.
    """
    columns = [
        'Year', 'Month', 'Day', 'Hour', 'Minute',
        'Data Source and Uncertainty Flags',
        'Dry Bulb Temperature', 'Dew Point Temperature',
        'Relative Humidity', 'Atmospheric Station Pressure',
        'Extraterrestrial Horizontal Radiation',
        'Extraterrestrial Direct Normal Radiation',
        'Horizontal Infrared Radiation Intensity',
        'Global Horizontal Radiation',
        'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
        'Global Horizontal Illuminance',
        'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance',
        'Zenith Luminance', 'Wind Direction', 'Wind Speed',
        'Total Sky Cover', 'Opaque Sky Cover', 'Visibility',
        'Ceiling Height', 'Present Weather Observation',
        'Present Weather Codes', 'Precipitable Water',
        'Aerosol Optical Depth', 'Snow Depth',
        'Days Since Last Snowfall', 'Albedo',
        'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'
    ]

    with open(epw_path, 'r') as f:
        lines = f.readlines()

    headers = {
        'LOCATION': lines[0].strip(),
        'DESIGN_CONDITIONS': lines[1].strip(),
        'TYPICAL_EXTREME_PERIODS': lines[2].strip(),
        'GROUND_TEMPERATURES': lines[3].strip(),
        'HOLIDAYS_DAYLIGHT_SAVINGS': lines[4].strip(),
        'COMMENTS_1': lines[5].strip(),
        'COMMENTS_2': lines[6].strip(),
        'DATA_PERIODS': lines[7].strip()
    }

    location = headers['LOCATION'].split(',')
    if len(location) >= 10:
        headers['LOCATION'] = {
            'City': location[1].strip(),
            'State': location[2].strip(),
            'Country': location[3].strip(),
            'Data Source': location[4].strip(),
            'WMO': location[5].strip(),
            'Latitude': float(location[6]),
            'Longitude': float(location[7]),
            'Time Zone': float(location[8]),
            'Elevation': float(location[9])
        }

    data = [line.strip().split(',') for line in lines[8:]]
    df = pd.DataFrame(data, columns=columns)

    numeric_columns = [
        'Year', 'Month', 'Day', 'Hour', 'Minute',
        'Dry Bulb Temperature', 'Dew Point Temperature',
        'Relative Humidity', 'Atmospheric Station Pressure',
        'Extraterrestrial Horizontal Radiation',
        'Extraterrestrial Direct Normal Radiation',
        'Horizontal Infrared Radiation Intensity',
        'Global Horizontal Radiation',
        'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
        'Global Horizontal Illuminance',
        'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance',
        'Zenith Luminance', 'Wind Direction', 'Wind Speed',
        'Total Sky Cover', 'Opaque Sky Cover', 'Visibility',
        'Ceiling Height', 'Precipitable Water',
        'Aerosol Optical Depth', 'Snow Depth',
        'Days Since Last Snowfall', 'Albedo',
        'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['datetime'] = pd.to_datetime({
        'year': df['Year'],
        'month': df['Month'],
        'day': df['Day'],
        'hour': df['Hour'] - 1,
        'minute': df['Minute']
    })
    df.set_index('datetime', inplace=True)
    return df, headers


def read_epw_for_solar_simulation(epw_file_path):
    """
    Read EPW file specifically for solar simulation purposes.
    Returns (df[DNI,DHI], lon, lat, tz, elevation_m).
    """
    epw_path_obj = Path(epw_file_path)
    if not epw_path_obj.exists() or not epw_path_obj.is_file():
        raise FileNotFoundError(f"EPW file not found: {epw_file_path}")

    with open(epw_path_obj, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    location_line = None
    for line in lines:
        if line.startswith("LOCATION"):
            location_line = line.strip().split(',')
            break
    if location_line is None:
        raise ValueError("Could not find LOCATION line in EPW file.")

    lat = float(location_line[6])
    lon = float(location_line[7])
    tz = float(location_line[8])
    elevation_m = float(location_line[9])

    data_start_index = None
    for i, line in enumerate(lines):
        vals = line.strip().split(',')
        if i >= 8 and len(vals) > 30:
            data_start_index = i
            break
    if data_start_index is None:
        raise ValueError("Could not find start of weather data lines in EPW file.")

    data = []
    for l in lines[data_start_index:]:
        vals = l.strip().split(',')
        if len(vals) < 15:
            continue
        year = int(vals[0])
        month = int(vals[1])
        day = int(vals[2])
        hour = int(vals[3]) - 1
        dni = float(vals[14])
        dhi = float(vals[15])
        timestamp = pd.Timestamp(year, month, day, hour)
        data.append([timestamp, dni, dhi])

    df = pd.DataFrame(data, columns=['time', 'DNI', 'DHI']).set_index('time')
    df = df.sort_index()
    return df, lon, lat, tz, elevation_m


