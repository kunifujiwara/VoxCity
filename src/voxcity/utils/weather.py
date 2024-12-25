import requests
import xml.etree.ElementTree as ET
import re
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import json
import zipfile
import pandas as pd
import io
import os
import numpy as np
from datetime import datetime

def safe_rename(src: Path, dst: Path) -> Path:
    """
    Safely rename a file, handling existing files by adding a number suffix.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        Path: Final destination path used
    """
    if not dst.exists():
        src.rename(dst)
        return dst
        
    # If file exists, add number suffix
    base = dst.stem
    ext = dst.suffix
    counter = 1
    while True:
        new_dst = dst.with_name(f"{base}_{counter}{ext}")
        if not new_dst.exists():
            src.rename(new_dst)
            return new_dst
        counter += 1

def safe_extract(zip_ref: zipfile.ZipFile, filename: str, extract_dir: Path) -> Path:
    """
    Safely extract a file from zip, handling existing files.
    
    Args:
        zip_ref: Open ZipFile reference
        filename: Name of file to extract
        extract_dir: Directory to extract to
    
    Returns:
        Path: Path to extracted file
    """
    try:
        zip_ref.extract(filename, extract_dir)
        return extract_dir / filename
    except FileExistsError:
        # If file exists, extract to temporary name and return path
        temp_name = f"temp_{os.urandom(4).hex()}_{filename}"
        zip_ref.extract(filename, extract_dir, temp_name)
        return extract_dir / temp_name

def process_epw(epw_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Process an EPW file into a pandas DataFrame.
    
    Args:
        epw_path: Path to the EPW file
        
    Returns:
        Tuple containing:
        - DataFrame with hourly weather data
        - Dictionary with EPW header metadata
    """
    # EPW column names (these are standardized)
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
    
    # Read EPW file
    with open(epw_path, 'r') as f:
        lines = f.readlines()
    
    # Extract header metadata (first 8 lines)
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
    
    # Parse location data
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
    
    # Create DataFrame from weather data (skipping headers)
    data = [line.strip().split(',') for line in lines[8:]]
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns
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
    
    # Create datetime index
    df['datetime'] = pd.to_datetime({
        'year': df['Year'],
        'month': df['Month'],
        'day': df['Day'],
        'hour': df['Hour'] - 1,  # EPW hours are 1-24, pandas expects 0-23
        'minute': df['Minute']
    })
    df.set_index('datetime', inplace=True)
    
    return df, headers

def get_nearest_epw_from_climate_onebuilding(latitude: float, longitude: float, output_dir: str = "./", max_distance: Optional[float] = None, 
                extract_zip: bool = True, load_data: bool = True) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Download and process EPW weather file from Climate.OneBuilding.Org based on coordinates.
    
    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        output_dir (str): Directory to save the EPW file (defaults to current directory)
        max_distance (float, optional): Maximum distance in kilometers to search for stations
        extract_zip (bool): Whether to extract the ZIP file (default True)
        load_data (bool): Whether to load the EPW data into a DataFrame (default True)
    
    Returns:
        Tuple containing:
        - Path to the EPW file (or None if download fails)
        - DataFrame with hourly weather data (if load_data=True)
        - Dictionary with EPW header metadata (if load_data=True)
    """
    
    # List of primary KML sources
    KML_SOURCES = {
        "Africa": "https://climate.onebuilding.org/WMO_Region_1_Africa/Region1_Africa_EPW_Processing_locations.kml",
        "Asia": "https://climate.onebuilding.org/WMO_Region_2_Asia/Region2_Asia_EPW_Processing_locations.kml",
        "Japan": "https://climate.onebuilding.org/sources/JGMY_EPW_Processing_locations.kml",
        "India": "https://climate.onebuilding.org/sources/ITMY_EPW_Processing_locations.kml",
        "Argentina": "https://climate.onebuilding.org/sources/ArgTMY_EPW_Processing_locations.kml",
        "Canada": "https://climate.onebuilding.org/sources/Region4_Canada_TMYx_EPW_Processing_locations.kml",
        "USA": "https://climate.onebuilding.org/sources/Region4_USA_TMYx_EPW_Processing_locations.kml",
        "Caribbean": "https://climate.onebuilding.org/sources/Region4_NA_CA_Caribbean_TMYx_EPW_Processing_locations.kml",
        "Southwest_Pacific": "https://climate.onebuilding.org/sources/Region5_Southwest_Pacific_TMYx_EPW_Processing_locations.kml",
        "Europe": "https://climate.onebuilding.org/sources/Region6_Europe_TMYx_EPW_Processing_locations.kml",
        "Antarctica": "https://climate.onebuilding.org/sources/Region7_Antarctica_TMYx_EPW_Processing_locations.kml"
    }

    def try_decode(content: bytes) -> str:
        """Try different encodings to decode content."""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all else fails, try to decode with replacement
        return content.decode('utf-8', errors='replace')

    def clean_xml(content: str) -> str:
        """Clean XML content of invalid characters."""
        # Replace problematic characters
        content = content.replace('&ntilde;', 'n')
        content = content.replace('&Ntilde;', 'N')
        content = content.replace('ñ', 'n')
        content = content.replace('Ñ', 'N')
        
        # Remove other invalid XML characters
        content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\xFF]', '', content)
        return content

    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on Earth."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def parse_coordinates(point_text: str) -> Tuple[float, float, float]:
        """Parse coordinates from KML Point text."""
        try:
            coords = point_text.strip().split(',')
            if len(coords) >= 2:
                lon, lat = map(float, coords[:2])
                elevation = float(coords[2]) if len(coords) > 2 else 0
                return lat, lon, elevation
        except (ValueError, IndexError):
            pass
        return None

    def parse_station_from_description(desc: str, point_coords: Optional[Tuple[float, float, float]] = None) -> Dict:
        """Parse station metadata from KML description."""
        if not desc:
            return None
            
        # Extract URL
        url_match = re.search(r'URL (https://.*?\.zip)', desc)
        if not url_match:
            return None
            
        url = url_match.group(1)
        
        # First try to parse coordinates in deg/min format
        coord_match = re.search(r'([NS]) (\d+)&deg;\s*(\d+\.\d+)\'.*?([EW]) (\d+)&deg;\s*(\d+\.\d+)\'', desc)
        
        if coord_match:
            ns, lat_deg, lat_min, ew, lon_deg, lon_min = coord_match.groups()
            lat = float(lat_deg) + float(lat_min)/60
            if ns == 'S':
                lat = -lat
            lon = float(lon_deg) + float(lon_min)/60
            if ew == 'W':
                lon = -lon
        elif point_coords:
            lat, lon, _ = point_coords
        else:
            return None
            
        # Extract metadata with error handling
        def extract_value(pattern: str, default: str = None) -> str:
            match = re.search(pattern, desc)
            return match.group(1) if match else default

        metadata = {
            'url': url,
            'latitude': lat,
            'longitude': lon,
            'elevation': int(extract_value(r'Elevation <b>(-?\d+)</b>', '0')),
            'name': extract_value(r'<b>(.*?)</b>'),
            'wmo': extract_value(r'WMO <b>(\d+)</b>'),
            'climate_zone': extract_value(r'Climate Zone <b>(.*?)</b>'),
            'period': extract_value(r'Period of Record=(\d{4}-\d{4})'),
            'heating_db': extract_value(r'99% Heating DB <b>(.*?)</b>'),
            'cooling_db': extract_value(r'1% Cooling DB <b>(.*?)</b>'),
            'hdd18': extract_value(r'HDD18 <b>(\d+)</b>'),
            'cdd10': extract_value(r'CDD10 <b>(\d+)</b>'),
            'time_zone': extract_value(r'Time Zone {GMT <b>([-+]?\d+\.\d+)</b>')
        }
        
        return metadata

    def get_stations_from_kml(kml_url: str) -> List[Dict]:
        """Get weather stations from a KML file."""
        try:
            response = requests.get(kml_url, timeout=30)
            response.raise_for_status()
            
            # Try to decode content
            content = try_decode(response.content)
            content = clean_xml(content)
            
            try:
                root = ET.fromstring(content.encode('utf-8'))
            except ET.ParseError as e:
                print(f"Error parsing KML file {kml_url}: {e}")
                return []

            # Define namespace
            ns = {'kml': 'http://earth.google.com/kml/2.1'}
            
            stations = []
            
            # Find all Placemarks
            for placemark in root.findall('.//kml:Placemark', ns):
                name = placemark.find('kml:name', ns)
                desc = placemark.find('kml:description', ns)
                point = placemark.find('.//kml:Point/kml:coordinates', ns)
                
                # Skip placemarks without description
                if desc is None or not desc.text or "Data Source" not in desc.text:
                    continue

                # Get coordinates from Point element if available
                point_coords = None
                if point is not None and point.text:
                    point_coords = parse_coordinates(point.text)
                
                # Parse station data
                station_data = parse_station_from_description(desc.text, point_coords)
                if station_data:
                    station_data['name'] = name.text if name is not None else "Unknown"
                    station_data['kml_source'] = kml_url
                    stations.append(station_data)
            
            return stations
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing KML file {kml_url}: {e}")
            return []
        except Exception as e:
            print(f"Error processing KML file {kml_url}: {e}")
            return []
    
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get stations from all KML sources
        print("Fetching weather station data from Climate.OneBuilding.Org...")
        all_stations = []
        
        for region, url in KML_SOURCES.items():
            print(f"Scanning {region}...")
            stations = get_stations_from_kml(url)
            all_stations.extend(stations)
            print(f"Found {len(stations)} stations in {region}")
        
        print(f"\nTotal stations found: {len(all_stations)}")
        
        if not all_stations:
            raise ValueError("No weather stations found")
            
        # Calculate distances and find nearest station
        stations_with_distances = [
            (station, haversine_distance(latitude, longitude, station['latitude'], station['longitude']))
            for station in all_stations
        ]
        
        # Filter by max distance if specified
        if max_distance is not None:
            close_stations = [
                (station, distance) 
                for station, distance in stations_with_distances 
                if distance <= max_distance
            ]
            if not close_stations:
                # If no stations within max_distance, find the closest one anyway
                closest_station, min_distance = min(stations_with_distances, key=lambda x: x[1])
                print(f"\nNo stations found within {max_distance} km. Closest station is {min_distance:.1f} km away.")
                print("Using closest available station.")
                stations_with_distances = [(closest_station, min_distance)]
            else:
                stations_with_distances = close_stations
        
        # Find nearest station
        nearest_station, distance = min(stations_with_distances, key=lambda x: x[1])
        
        # Download the EPW file
        print(f"\nDownloading EPW file for {nearest_station['name']}...")
        epw_response = requests.get(nearest_station['url'])
        epw_response.raise_for_status()
        
        # Create a temporary directory for zip extraction
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the zip file
        zip_file = temp_dir / "weather_data.zip"
        with open(zip_file, 'wb') as f:
            f.write(epw_response.content)
        
        final_epw = None
        try:
            # Extract the zip file
            if extract_zip:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Find the EPW file in the archive
                    epw_files = [f for f in zip_ref.namelist() if f.lower().endswith('.epw')]
                    if not epw_files:
                        raise ValueError("No EPW file found in the downloaded archive")
                    
                    # Extract the EPW file
                    epw_filename = epw_files[0]
                    extracted_epw = safe_extract(zip_ref, epw_filename, temp_dir)
                    
                    # Move the EPW file to the final location
                    final_epw = Path(output_dir) / f"{nearest_station['name'].replace(' ', '_').replace(',', '').lower()}.epw"
                    final_epw = safe_rename(extracted_epw, final_epw)
        finally:
            # Clean up temporary files regardless of success or failure
            try:
                if zip_file.exists():
                    zip_file.unlink()
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")
        
        if final_epw is None:
            raise ValueError("Failed to extract EPW file")

        # Save metadata
        metadata_file = final_epw.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(nearest_station, f, indent=2)
        
        # Print station information
        print(f"\nDownloaded EPW file for {nearest_station['name']}")
        print(f"Distance: {distance:.2f} km")
        print(f"Station coordinates: {nearest_station['latitude']}, {nearest_station['longitude']}")
        if nearest_station['wmo']:
            print(f"WMO: {nearest_station['wmo']}")
        if nearest_station['climate_zone']:
            print(f"Climate zone: {nearest_station['climate_zone']}")
        if nearest_station['period']:
            print(f"Data period: {nearest_station['period']}")
        print(f"Files saved:")
        print(f"- EPW: {final_epw}")
        print(f"- Metadata: {metadata_file}")
        
        # Load the EPW data if requested
        df = None
        headers = None
        if load_data:
            print("\nLoading EPW data...")
            df, headers = process_epw(final_epw)
            print(f"Loaded {len(df)} hourly records")
        
        return str(final_epw), df, headers
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None, None

def read_epw_for_solar_simulation(epw_file_path):
    with open(epw_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the LOCATION line
    location_line = None
    for line in lines:
        if line.startswith("LOCATION"):
            location_line = line.strip().split(',')
            break

    if location_line is None:
        raise ValueError("Could not find LOCATION line in EPW file.")

    # LOCATION line format:
    # LOCATION,City,State/Country,Country,DataSource,WMO,Latitude,Longitude,Time Zone,Elevation
    # Example: LOCATION,Marina.Muni.AP,CA,USA,SRC-TMYx,690070,36.68300,-121.7670,-8.0,43.0
    lat = float(location_line[6])
    lon = float(location_line[7])
    tz = float(location_line[8])  # local standard time offset
    elevation_m = float(location_line[9])

    # Find start of data
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
        dni = float(vals[13])
        dhi = float(vals[14])
        timestamp = pd.Timestamp(year, month, day, hour)
        data.append([timestamp, dni, dhi])

    df = pd.DataFrame(data, columns=['time', 'DNI', 'DHI']).set_index('time')
    df = df.sort_index()

    return df, lat, lon, tz, elevation_m