"""
Weather data utilities for VoxelCity.

This module provides functionality to download and process Energyplus Weather (EPW) files
from Climate.OneBuilding.Org based on geographical coordinates. It includes utilities for:

- Automatically finding the nearest weather station to given coordinates
- Downloading EPW files from various global regions
- Processing EPW files into pandas DataFrames for analysis
- Extracting solar radiation data for solar simulations

The main function get_nearest_epw_from_climate_onebuilding() provides a comprehensive
solution for obtaining weather data for any global location by automatically detecting
the appropriate region and finding the closest available weather station.
"""

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

# =============================================================================
# FILE HANDLING UTILITIES
# =============================================================================

def safe_rename(src: Path, dst: Path) -> Path:
    """
    Safely rename a file, handling existing files by adding a number suffix.
    
    This function prevents file conflicts by automatically generating unique filenames
    when the target destination already exists. It appends incremental numbers to
    the base filename until a unique name is found.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        Path: Final destination path used
    """
    # If destination doesn't exist, simple rename
    if not dst.exists():
        src.rename(dst)
        return dst
        
    # If file exists, add number suffix
    base = dst.stem
    ext = dst.suffix
    counter = 1
    # Keep incrementing counter until we find a name that doesn't exist
    while True:
        new_dst = dst.with_name(f"{base}_{counter}{ext}")
        if not new_dst.exists():
            src.rename(new_dst)
            return new_dst
        counter += 1

def safe_extract(zip_ref: zipfile.ZipFile, filename: str, extract_dir: Path) -> Path:
    """
    Safely extract a file from zip, handling existing files.
    
    This function handles the case where a file with the same name already exists
    in the extraction directory by using a temporary filename with random suffix.
    
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

# =============================================================================
# EPW FILE PROCESSING
# =============================================================================

def process_epw(epw_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Process an EPW file into a pandas DataFrame.
    
    EPW (EnergyPlus Weather) files contain standardized weather data in a specific format.
    The first 8 lines contain metadata, followed by 8760 lines of hourly weather data
    for a typical meteorological year.
    
    Args:
        epw_path: Path to the EPW file
        
    Returns:
        Tuple containing:
        - DataFrame with hourly weather data indexed by datetime
        - Dictionary with EPW header metadata including location information
    """
    # EPW column names (these are standardized across all EPW files)
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
    
    # Read EPW file - EPW files are always in comma-separated format
    with open(epw_path, 'r') as f:
        lines = f.readlines()
    
    # Extract header metadata (first 8 lines contain standardized metadata)
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
    
    # Parse location data from first header line
    # Format: LOCATION,City,State,Country,Source,WMO,Latitude,Longitude,TimeZone,Elevation
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
    
    # Create DataFrame from weather data (skipping 8 header lines)
    data = [line.strip().split(',') for line in lines[8:]]
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns to appropriate data types
    # All weather parameters should be numeric except uncertainty flags and weather codes
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
    
    # Convert to numeric, handling any parsing errors gracefully
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create datetime index for time series analysis
    # EPW hours are 1-24, but pandas expects 0-23 for proper datetime handling
    df['datetime'] = pd.to_datetime({
        'year': df['Year'],
        'month': df['Month'],
        'day': df['Day'],
        'hour': df['Hour'] - 1,  # EPW hours are 1-24, pandas expects 0-23
        'minute': df['Minute']
    })
    df.set_index('datetime', inplace=True)
    
    return df, headers

# =============================================================================
# MAIN WEATHER DATA DOWNLOAD FUNCTION
# =============================================================================

def get_nearest_epw_from_climate_onebuilding(longitude: float, latitude: float, output_dir: str = "./", max_distance: Optional[float] = None, 
                extract_zip: bool = True, load_data: bool = True, region: Optional[Union[str, List[str]]] = None) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Download and process EPW weather file from Climate.OneBuilding.Org based on coordinates.
    
    This function automatically finds and downloads the nearest available weather station
    data from Climate.OneBuilding.Org's global database. It supports region-based searching
    for improved performance and can automatically detect the appropriate region based on
    coordinates.
    
    The function performs the following steps:
    1. Determines which regional KML files to scan based on coordinates or user input
    2. Downloads and parses KML files to extract weather station metadata
    3. Calculates distances to find the nearest station
    4. Downloads the EPW file from the nearest station
    5. Optionally processes the EPW data into a pandas DataFrame
    
    Args:
        longitude (float): Longitude of the location (-180 to 180)
        latitude (float): Latitude of the location (-90 to 90)
        output_dir (str): Directory to save the EPW file (defaults to current directory)
        max_distance (float, optional): Maximum distance in kilometers to search for stations.
                                       If no stations within this distance, uses closest available.
        extract_zip (bool): Whether to extract the ZIP file (default True)
        load_data (bool): Whether to load the EPW data into a DataFrame (default True)
        region (str or List[str], optional): Specific region(s) to scan for stations.
                                            Options: "Africa", "Asia", "Japan", "India", "Argentina",
                                            "Canada", "USA", "Caribbean", "Southwest_Pacific", 
                                            "Europe", "Antarctica", or "all".
                                            If None, will auto-detect region based on coordinates.
    
    Returns:
        Tuple containing:
        - Path to the EPW file (or None if download fails)
        - DataFrame with hourly weather data (if load_data=True, else None)
        - Dictionary with EPW header metadata (if load_data=True, else None)
    
    Raises:
        ValueError: If invalid region specified or no weather stations found
        requests.exceptions.RequestException: If network requests fail
    """
    
    # Regional KML sources from Climate.OneBuilding.Org
    # Each region maintains its own KML file with weather station locations and metadata
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
    
    # Define approximate geographical boundaries for automatic region detection
    # These bounds help determine which regional KML files to scan based on coordinates
    REGION_BOUNDS = {
        "Africa": {"lon_min": -20, "lon_max": 55, "lat_min": -35, "lat_max": 40},
        "Asia": {"lon_min": 25, "lon_max": 150, "lat_min": 0, "lat_max": 55},
        "Japan": {"lon_min": 127, "lon_max": 146, "lat_min": 24, "lat_max": 46},
        "India": {"lon_min": 68, "lon_max": 97, "lat_min": 6, "lat_max": 36},
        "Argentina": {"lon_min": -75, "lon_max": -53, "lat_min": -55, "lat_max": -22},
        "Canada": {"lon_min": -141, "lon_max": -52, "lat_min": 42, "lat_max": 83},
        "USA": {"lon_min": -170, "lon_max": -65, "lat_min": 20, "lat_max": 72},
        "Caribbean": {"lon_min": -90, "lon_max": -59, "lat_min": 10, "lat_max": 27},
        "Southwest_Pacific": {"lon_min": 110, "lon_max": 180, "lat_min": -50, "lat_max": 0},
        "Europe": {"lon_min": -25, "lon_max": 40, "lat_min": 35, "lat_max": 72},
        "Antarctica": {"lon_min": -180, "lon_max": 180, "lat_min": -90, "lat_max": -60}
    }

    def detect_regions(lon: float, lat: float) -> List[str]:
        """
        Detect which region(s) the coordinates belong to.
        
        Uses the REGION_BOUNDS to determine appropriate regions to search.
        If coordinates don't fall within any region, returns the 3 closest regions.
        
        Args:
            lon: Longitude coordinate
            lat: Latitude coordinate
            
        Returns:
            List of region names to search
        """
        matching_regions = []
        
        # Handle special case of longitude wrap around 180/-180
        # Normalize longitude to standard -180 to 180 range
        lon_adjusted = lon
        if lon < -180:
            lon_adjusted = lon + 360
        elif lon > 180:
            lon_adjusted = lon - 360
            
        # Check if coordinates fall within any region bounds
        for region_name, bounds in REGION_BOUNDS.items():
            # Check if point is within region bounds
            if (bounds["lon_min"] <= lon_adjusted <= bounds["lon_max"] and 
                bounds["lat_min"] <= lat <= bounds["lat_max"]):
                matching_regions.append(region_name)
        
        # If no regions matched, find the closest regions by boundary distance
        if not matching_regions:
            # Calculate "distance" to each region's boundary (simplified metric)
            region_distances = []
            for region_name, bounds in REGION_BOUNDS.items():
                # Calculate distance to closest edge of region bounds
                lon_dist = 0
                if lon_adjusted < bounds["lon_min"]:
                    lon_dist = bounds["lon_min"] - lon_adjusted
                elif lon_adjusted > bounds["lon_max"]:
                    lon_dist = lon_adjusted - bounds["lon_max"]
                
                lat_dist = 0
                if lat < bounds["lat_min"]:
                    lat_dist = bounds["lat_min"] - lat
                elif lat > bounds["lat_max"]:
                    lat_dist = lat - bounds["lat_max"]
                
                # Simple Euclidean distance metric (not actual geographic distance)
                distance = (lon_dist**2 + lat_dist**2)**0.5
                region_distances.append((region_name, distance))
            
            # Get 3 closest regions to ensure we find stations
            closest_regions = sorted(region_distances, key=lambda x: x[1])[:3]
            matching_regions = [r[0] for r in closest_regions]
        
        return matching_regions

    def try_decode(content: bytes) -> str:
        """
        Try different encodings to decode content.
        
        KML files from different regions may use various text encodings.
        This function tries common encodings to successfully decode the content.
        
        Args:
            content: Raw bytes content
            
        Returns:
            Decoded string content
        """
        # Try common encodings in order of preference
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all else fails, try to decode with replacement characters
        return content.decode('utf-8', errors='replace')

    def clean_xml(content: str) -> str:
        """
        Clean XML content of invalid characters.
        
        Some KML files contain characters that cause XML parsing issues.
        This function replaces or removes problematic characters to ensure
        successful XML parsing.
        
        Args:
            content: Raw XML content string
            
        Returns:
            Cleaned XML content string
        """
        # Replace problematic Spanish characters that cause XML parsing issues
        content = content.replace('&ntilde;', 'n')
        content = content.replace('&Ntilde;', 'N')
        content = content.replace('ñ', 'n')
        content = content.replace('Ñ', 'N')
        
        # Remove other invalid XML characters using regex
        # Keep only valid XML characters: tab, newline, carriage return, printable ASCII, and extended Latin
        content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\xFF]', '', content)
        return content

    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Uses the Haversine formula to calculate the shortest distance between
        two points on a sphere (Earth) given their latitude and longitude.
        
        Args:
            lon1, lat1: Coordinates of first point
            lon2, lat2: Coordinates of second point
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula calculation
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def parse_coordinates(point_text: str) -> Tuple[float, float, float]:
        """
        Parse coordinates from KML Point text.
        
        KML Point elements contain coordinates in the format "longitude,latitude,elevation".
        This function extracts and converts these values to float.
        
        Args:
            point_text: Raw coordinate text from KML Point element
            
        Returns:
            Tuple of (latitude, longitude, elevation) or None if parsing fails
        """
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
        """
        Parse station metadata from KML description.
        
        KML description fields contain detailed station information including:
        - Download URL for EPW file
        - Coordinates in degrees/minutes format
        - Station metadata (WMO code, climate zone, etc.)
        - Design conditions and climate statistics
        
        Args:
            desc: KML description text containing station metadata
            point_coords: Fallback coordinates from KML Point element
            
        Returns:
            Dictionary with parsed station metadata or None if parsing fails
        """
        if not desc:
            return None
            
        # Extract download URL - this is required for station to be valid
        url_match = re.search(r'URL (https://.*?\.zip)', desc)
        if not url_match:
            return None
            
        url = url_match.group(1)
        
        # First try to parse coordinates in degrees/minutes format from description
        # Format: N XX°YY.YY' W ZZ°AA.AA'
        coord_match = re.search(r'([NS]) (\d+)&deg;\s*(\d+\.\d+)\'.*?([EW]) (\d+)&deg;\s*(\d+\.\d+)\'', desc)
        
        if coord_match:
            # Convert degrees/minutes to decimal degrees
            ns, lat_deg, lat_min, ew, lon_deg, lon_min = coord_match.groups()
            lat = float(lat_deg) + float(lat_min)/60
            if ns == 'S':
                lat = -lat
            lon = float(lon_deg) + float(lon_min)/60
            if ew == 'W':
                lon = -lon
        elif point_coords:
            # Fall back to coordinates from KML Point element
            lat, lon, _ = point_coords
        else:
            # No coordinates available - station is not usable
            return None
            
        # Extract metadata with error handling using helper function
        def extract_value(pattern: str, default: str = None) -> str:
            """Extract value using regex pattern, return default if not found."""
            match = re.search(pattern, desc)
            return match.group(1) if match else default

        # Build comprehensive station metadata dictionary
        metadata = {
            'url': url,
            'longitude': lon,
            'latitude': lat,
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
        """
        Get weather stations from a KML file.
        
        Downloads and parses a KML file containing weather station information.
        Each Placemark in the KML represents a weather station with metadata
        in the description field and coordinates in Point elements.
        
        Args:
            kml_url: URL to the KML file
            
        Returns:
            List of dictionaries containing station metadata
        """
        try:
            # Download KML file with timeout
            response = requests.get(kml_url, timeout=30)
            response.raise_for_status()
            
            # Try to decode content with multiple encodings
            content = try_decode(response.content)
            content = clean_xml(content)
            
            # Parse XML content
            try:
                root = ET.fromstring(content.encode('utf-8'))
            except ET.ParseError as e:
                print(f"Error parsing KML file {kml_url}: {e}")
                return []

            # Define KML namespace for element searching
            ns = {'kml': 'http://earth.google.com/kml/2.1'}
            
            stations = []
            
            # Find all Placemark elements (each represents a weather station)
            for placemark in root.findall('.//kml:Placemark', ns):
                name = placemark.find('kml:name', ns)
                desc = placemark.find('kml:description', ns)
                point = placemark.find('.//kml:Point/kml:coordinates', ns)
                
                # Skip placemarks without description or that don't contain weather data
                if desc is None or not desc.text or "Data Source" not in desc.text:
                    continue

                # Get coordinates from Point element if available
                point_coords = None
                if point is not None and point.text:
                    point_coords = parse_coordinates(point.text)
                
                # Parse comprehensive station data from description
                station_data = parse_station_from_description(desc.text, point_coords)
                if station_data:
                    # Add station name and source information
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
        
        # Determine which regions to scan based on user input or auto-detection
        regions_to_scan = {}
        if region is None:
            # Auto-detect regions based on coordinates
            detected_regions = detect_regions(longitude, latitude)
            
            if detected_regions:
                print(f"Auto-detected regions: {', '.join(detected_regions)}")
                for r in detected_regions:
                    regions_to_scan[r] = KML_SOURCES[r]
            else:
                # Fallback to all regions if detection fails
                print("Could not determine region from coordinates. Scanning all regions.")
                regions_to_scan = KML_SOURCES
        elif isinstance(region, str):
            # Handle string input for region selection
            if region.lower() == "all":
                regions_to_scan = KML_SOURCES
            elif region in KML_SOURCES:
                regions_to_scan[region] = KML_SOURCES[region]
            else:
                valid_regions = ", ".join(KML_SOURCES.keys())
                raise ValueError(f"Invalid region: '{region}'. Valid regions are: {valid_regions}")
        else:
            # Handle list input for multiple regions
            for r in region:
                if r not in KML_SOURCES:
                    valid_regions = ", ".join(KML_SOURCES.keys())
                    raise ValueError(f"Invalid region: '{r}'. Valid regions are: {valid_regions}")
                regions_to_scan[r] = KML_SOURCES[r]
        
        # Get stations from selected KML sources
        print("Fetching weather station data from Climate.OneBuilding.Org...")
        all_stations = []
        
        # Process each selected region
        for region_name, url in regions_to_scan.items():
            print(f"Scanning {region_name}...")
            stations = get_stations_from_kml(url)
            all_stations.extend(stations)
            print(f"Found {len(stations)} stations in {region_name}")
        
        print(f"\nTotal stations found: {len(all_stations)}")
        
        if not all_stations:
            raise ValueError("No weather stations found")
            
        # Calculate distances from target coordinates to all stations
        stations_with_distances = [
            (station, haversine_distance(longitude, latitude, station['longitude'], station['latitude']))
            for station in all_stations
        ]
        
        # Filter by maximum distance if specified
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
        
        # Find the nearest weather station
        nearest_station, distance = min(stations_with_distances, key=lambda x: x[1])
        
        # Download the EPW file from the nearest station
        print(f"\nDownloading EPW file for {nearest_station['name']}...")
        epw_response = requests.get(nearest_station['url'])
        epw_response.raise_for_status()
        
        # Create a temporary directory for zip extraction
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the downloaded zip file temporarily
        zip_file = temp_dir / "weather_data.zip"
        with open(zip_file, 'wb') as f:
            f.write(epw_response.content)
        
        final_epw = None
        try:
            # Extract the EPW file from the zip archive
            if extract_zip:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Find the EPW file in the archive (should be exactly one)
                    epw_files = [f for f in zip_ref.namelist() if f.lower().endswith('.epw')]
                    if not epw_files:
                        raise ValueError("No EPW file found in the downloaded archive")
                    
                    # Extract the EPW file
                    epw_filename = epw_files[0]
                    extracted_epw = safe_extract(zip_ref, epw_filename, temp_dir)
                    
                    # Move the EPW file to the final location with cleaned filename
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

        # Save station metadata alongside the EPW file
        metadata_file = final_epw.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(nearest_station, f, indent=2)
        
        # Print comprehensive station information
        print(f"\nDownloaded EPW file for {nearest_station['name']}")
        print(f"Distance: {distance:.2f} km")
        print(f"Station coordinates: {nearest_station['longitude']}, {nearest_station['latitude']}")
        if nearest_station['wmo']:
            print(f"WMO: {nearest_station['wmo']}")
        if nearest_station['climate_zone']:
            print(f"Climate zone: {nearest_station['climate_zone']}")
        if nearest_station['period']:
            print(f"Data period: {nearest_station['period']}")
        print(f"Files saved:")
        print(f"- EPW: {final_epw}")
        print(f"- Metadata: {metadata_file}")
        
        # Load the EPW data into DataFrame if requested
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

# =============================================================================
# SOLAR SIMULATION UTILITIES
# =============================================================================

def read_epw_for_solar_simulation(epw_file_path):
    """
    Read EPW file specifically for solar simulation purposes.
    
    This function extracts essential solar radiation data and location metadata
    from an EPW file for use in solar energy calculations. It focuses on the
    Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI)
    which are the primary inputs for solar simulation models.
    
    Args:
        epw_file_path: Path to the EPW weather file
        
    Returns:
        Tuple containing:
        - DataFrame with time-indexed DNI and DHI data
        - Longitude (degrees)
        - Latitude (degrees) 
        - Time zone offset (hours from UTC)
        - Elevation (meters above sea level)
        
    Raises:
        ValueError: If LOCATION line not found or data parsing fails
    """
    # Read the entire EPW file
    with open(epw_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the LOCATION line (first line in EPW format)
    location_line = None
    for line in lines:
        if line.startswith("LOCATION"):
            location_line = line.strip().split(',')
            break

    if location_line is None:
        raise ValueError("Could not find LOCATION line in EPW file.")

    # Parse LOCATION line format:
    # LOCATION,City,State/Country,Country,DataSource,WMO,Latitude,Longitude,Time Zone,Elevation
    # Example: LOCATION,Marina.Muni.AP,CA,USA,SRC-TMYx,690070,36.68300,-121.7670,-8.0,43.0
    lat = float(location_line[6])
    lon = float(location_line[7])
    tz = float(location_line[8])  # local standard time offset from UTC
    elevation_m = float(location_line[9])

    # Find start of weather data (after 8 header lines)
    data_start_index = None
    for i, line in enumerate(lines):
        vals = line.strip().split(',')
        # Weather data lines have more than 30 columns and start after line 8
        if i >= 8 and len(vals) > 30:
            data_start_index = i
            break

    if data_start_index is None:
        raise ValueError("Could not find start of weather data lines in EPW file.")

    # Parse weather data focusing on solar radiation components
    data = []
    for l in lines[data_start_index:]:
        vals = l.strip().split(',')
        if len(vals) < 15:  # Skip malformed lines
            continue
        # Extract time components and solar radiation data
        year = int(vals[0])
        month = int(vals[1])
        day = int(vals[2])
        hour = int(vals[3]) - 1  # Convert EPW 1-24 hours to 0-23
        dni = float(vals[14])    # Direct Normal Irradiance (Wh/m²)
        dhi = float(vals[15])    # Diffuse Horizontal Irradiance (Wh/m²)
        
        # Create pandas timestamp for time series indexing
        timestamp = pd.Timestamp(year, month, day, hour)
        data.append([timestamp, dni, dhi])

    # Create DataFrame with time index for efficient time series operations
    df = pd.DataFrame(data, columns=['time', 'DNI', 'DHI']).set_index('time')
    df = df.sort_index()

    return df, lon, lat, tz, elevation_m