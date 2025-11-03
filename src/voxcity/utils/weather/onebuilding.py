from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import requests
import re
import xml.etree.ElementTree as ET
import json
import zipfile

from .files import safe_extract, safe_rename
from .epw import process_epw


def get_nearest_epw_from_climate_onebuilding(longitude: float, latitude: float, output_dir: str = "./", max_distance: Optional[float] = None,
                extract_zip: bool = True, load_data: bool = True, region: Optional[Union[str, List[str]]] = None,
                allow_insecure_ssl: bool = False, allow_http_fallback: bool = False,
                ssl_verify: Union[bool, str] = True) -> Tuple[Optional[str], Optional["pd.DataFrame"], Optional[Dict]]:
    """
    Download and process EPW weather file from Climate.OneBuilding.Org based on coordinates.
    """
    import numpy as np
    import pandas as pd

    # --- KML sources and region helpers (unchanged from monolith) ---
    KML_SOURCES = {
        "Africa": "https://climate.onebuilding.org/sources/Region1_Africa_TMYx_EPW_Processing_locations.kml",
        "Asia": "https://climate.onebuilding.org/sources/Region2_Asia_TMYx_EPW_Processing_locations.kml",
        "Japan": "https://climate.onebuilding.org/sources/JGMY_EPW_Processing_locations.kml",
        "India": "https://climate.onebuilding.org/sources/ITMY_EPW_Processing_locations.kml",
        "CSWD": "https://climate.onebuilding.org/sources/CSWD_EPW_Processing_locations.kml",
        "CityUHK": "https://climate.onebuilding.org/sources/CityUHK_EPW_Processing_locations.kml",
        "PHIKO": "https://climate.onebuilding.org/sources/PHIKO_EPW_Processing_locations.kml",
        "South_America": "https://climate.onebuilding.org/sources/Region3_South_America_TMYx_EPW_Processing_locations.kml",
        "Argentina": "https://climate.onebuilding.org/sources/ArgTMY_EPW_Processing_locations.kml",
        "INMET_TRY": "https://climate.onebuilding.org/sources/INMET_TRY_EPW_Processing_locations.kml",
        "AMTUes": "https://climate.onebuilding.org/sources/AMTUes_EPW_Processing_locations.kml",
        "BrazFuture": "https://climate.onebuilding.org/sources/BrazFuture_EPW_Processing_locations.kml",
        "Canada": "https://climate.onebuilding.org/sources/Region4_Canada_TMYx_EPW_Processing_locations.kml",
        "USA": "https://climate.onebuilding.org/sources/Region4_USA_TMYx_EPW_Processing_locations.kml",
        "Caribbean": "https://climate.onebuilding.org/sources/Region4_NA_CA_Caribbean_TMYx_EPW_Processing_locations.kml",
        "Southwest_Pacific": "https://climate.onebuilding.org/sources/Region5_Southwest_Pacific_TMYx_EPW_Processing_locations.kml",
        "Europe": "https://climate.onebuilding.org/sources/Region6_Europe_TMYx_EPW_Processing_locations.kml",
        "Antarctica": "https://climate.onebuilding.org/sources/Region7_Antarctica_TMYx_EPW_Processing_locations.kml",
    }

    REGION_DATASET_GROUPS = {
        "Africa": ["Africa"],
        "Asia": ["Asia", "Japan", "India", "CSWD", "CityUHK", "PHIKO"],
        "South_America": ["South_America", "Argentina", "INMET_TRY", "AMTUes", "BrazFuture"],
        "North_and_Central_America": ["North_and_Central_America", "Canada", "USA", "Caribbean"],
        "Southwest_Pacific": ["Southwest_Pacific"],
        "Europe": ["Europe"],
        "Antarctica": ["Antarctica"],
    }

    REGION_BOUNDS = {
        "Africa": {"lon_min": -25, "lon_max": 80, "lat_min": -55, "lat_max": 45},
        "Asia": {"lon_min": 20, "lon_max": 180, "lat_min": -10, "lat_max": 80},
        "Japan": {"lon_min": 127, "lon_max": 146, "lat_min": 24, "lat_max": 46},
        "India": {"lon_min": 68, "lon_max": 97, "lat_min": 6, "lat_max": 36},
        "South_America": {"lon_min": -92, "lon_max": -20, "lat_min": -60, "lat_max": 15},
        "Argentina": {"lon_min": -75, "lon_max": -53, "lat_min": -55, "lat_max": -22},
        "North_and_Central_America": {"lon_min": -180, "lon_max": 20, "lat_min": -10, "lat_max": 85},
        "Canada": {"lon_min": -141, "lon_max": -52, "lat_min": 42, "lat_max": 83},
        "USA": {"lon_min": -170, "lon_max": -65, "lat_min": 20, "lat_max": 72},
        "Caribbean": {"lon_min": -90, "lon_max": -59, "lat_min": 10, "lat_max": 27},
        "Southwest_Pacific": {"boxes": [
            {"lon_min": 90, "lon_max": 180, "lat_min": -50, "lat_max": 25},
            {"lon_min": -180, "lon_max": -140, "lat_min": -50, "lat_max": 25},
        ]},
        "Europe": {"lon_min": -75, "lon_max": 60, "lat_min": 25, "lat_max": 85},
        "Antarctica": {"lon_min": -180, "lon_max": 180, "lat_min": -90, "lat_max": -60}
    }

    def detect_regions(lon: float, lat: float) -> List[str]:
        matching_regions = []

        lon_adjusted = lon
        if lon < -180:
            lon_adjusted = lon + 360
        elif lon > 180:
            lon_adjusted = lon - 360

        def _in_box(bx: Dict[str, float], lon_v: float, lat_v: float) -> bool:
            return (bx["lon_min"] <= lon_v <= bx["lon_max"] and bx["lat_min"] <= lat_v <= bx["lat_max"]) 

        for region_name, bounds in REGION_BOUNDS.items():
            if "boxes" in bounds:
                for bx in bounds["boxes"]:
                    if _in_box(bx, lon_adjusted, lat):
                        matching_regions.append(region_name)
                        break
            else:
                if _in_box(bounds, lon_adjusted, lat):
                    matching_regions.append(region_name)

        if not matching_regions:
            region_distances = []
            def _box_distance(bx: Dict[str, float]) -> float:
                lon_dist = 0
                if lon_adjusted < bx["lon_min"]:
                    lon_dist = bx["lon_min"] - lon_adjusted
                elif lon_adjusted > bx["lon_max"]:
                    lon_dist = lon_adjusted - bx["lon_max"]
                lat_dist = 0
                if lat < bx["lat_min"]:
                    lat_dist = bx["lat_min"] - lat
                elif lat > bx["lat_max"]:
                    lat_dist = lat - bx["lat_max"]
                return (lon_dist**2 + lat_dist**2)**0.5
            for region_name, bounds in REGION_BOUNDS.items():
                if "boxes" in bounds:
                    d = min(_box_distance(bx) for bx in bounds["boxes"])
                else:
                    d = _box_distance(bounds)
                region_distances.append((region_name, d))
            closest_regions = sorted(region_distances, key=lambda x: x[1])[:3]
            matching_regions = [r[0] for r in closest_regions]
        return matching_regions

    def try_decode(content: bytes) -> str:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode('utf-8', errors='replace')

    def clean_xml(content: str) -> str:
        content = content.replace('&ntilde;', 'n').replace('&Ntilde;', 'N').replace('ñ', 'n').replace('Ñ', 'N')
        content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\xFF]', '', content)
        return content

    def parse_coordinates(point_text: str) -> Tuple[float, float, float]:
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
        if not desc:
            return None
        url_match = re.search(r'URL (https://.*?\.zip)', desc)
        if not url_match:
            return None
        url = url_match.group(1)
        coord_match = re.search(r'([NS]) (\d+)&deg;\s*(\d+\.\d+)'.encode('utf-8').decode('utf-8') + r"'.*?([EW]) (\d+)&deg;\s*(\d+\.\d+)'", desc)
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
        def extract_value(pattern: str, default: str = None) -> str:
            match = re.search(pattern, desc)
            return match.group(1) if match else default
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

    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        from math import radians, sin, cos, sqrt, atan2
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def try_download_station_zip(original_url: str, timeout_s: int = 30) -> Optional[bytes]:
        def candidate_urls(url: str) -> List[str]:
            urls = [url]
            if "/TUR_Turkey/" in url:
                urls.append(url.replace("/TUR_Turkey/", "/TUR_Turkiye/"))
            if "/TUR_Turkiye/" in url:
                urls.append(url.replace("/TUR_Turkiye/", "/TUR_Turkey/"))
            m = re.search(r"(.*_TMYx)(?:\.(\d{4}-\d{4}))?\.zip$", url)
            if m:
                base = m.group(1)
                variants = [
                    f"{base}.2009-2023.zip",
                    f"{base}.2007-2021.zip",
                    f"{base}.zip",
                    f"{base}.2004-2018.zip",
                ]
                for v in variants:
                    if v not in urls:
                        urls.append(v)
                extra = []
                for v in variants:
                    if "/TUR_Turkey/" in url:
                        extra.append(v.replace("/TUR_Turkey/", "/TUR_Turkiye/"))
                    if "/TUR_Turkiye/" in url:
                        extra.append(v.replace("/TUR_Turkiye/", "/TUR_Turkey/"))
                for v in extra:
                    if v not in urls:
                        urls.append(v)
            return urls

        tried = set()
        for u in candidate_urls(original_url):
            if u in tried:
                continue
            tried.add(u)
            try:
                resp = requests.get(u, timeout=timeout_s, verify=ssl_verify)
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.SSLError:
                if allow_insecure_ssl:
                    try:
                        resp = requests.get(u, timeout=timeout_s, verify=False)
                        resp.raise_for_status()
                        return resp.content
                    except requests.exceptions.RequestException:
                        if allow_http_fallback and u.lower().startswith("https://"):
                            insecure_url = "http://" + u.split("://", 1)[1]
                            try:
                                resp = requests.get(insecure_url, timeout=timeout_s)
                                resp.raise_for_status()
                                return resp.content
                            except requests.exceptions.RequestException:
                                pass
                        continue
                else:
                    if allow_http_fallback and u.lower().startswith("https://"):
                        insecure_url = "http://" + u.split("://", 1)[1]
                        try:
                            resp = requests.get(insecure_url, timeout=timeout_s)
                            resp.raise_for_status()
                            return resp.content
                        except requests.exceptions.RequestException:
                            pass
                    continue
            except requests.exceptions.HTTPError as he:
                if getattr(he.response, "status_code", None) == 404:
                    continue
                else:
                    raise
            except requests.exceptions.RequestException:
                continue
        return None

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        regions_to_scan = {}
        def _add_selection(selection_name: str, mapping: Dict[str, str], out: Dict[str, str]):
            if selection_name in REGION_DATASET_GROUPS:
                for key in REGION_DATASET_GROUPS[selection_name]:
                    if key in KML_SOURCES:
                        out[key] = KML_SOURCES[key]
            elif selection_name in KML_SOURCES:
                out[selection_name] = KML_SOURCES[selection_name]
            else:
                valid = sorted(list(REGION_DATASET_GROUPS.keys()) + list(KML_SOURCES.keys()))
                raise ValueError(f"Invalid region/dataset: '{selection_name}'. Valid options include: {', '.join(valid)}")

        if region is None:
            detected_regions = detect_regions(longitude, latitude)
            if detected_regions:
                print(f"Auto-detected regions: {', '.join(detected_regions)}")
                for r in detected_regions:
                    _add_selection(r, KML_SOURCES, regions_to_scan)
            else:
                print("Could not determine region from coordinates. Scanning all regions.")
                regions_to_scan = dict(KML_SOURCES)
        elif isinstance(region, str):
            if region.lower() == "all":
                regions_to_scan = dict(KML_SOURCES)
            else:
                _add_selection(region, KML_SOURCES, regions_to_scan)
        else:
            for r in region:
                _add_selection(r, KML_SOURCES, regions_to_scan)

        print("Fetching weather station data from Climate.OneBuilding.Org...")
        all_stations = []
        scanned_urls = set()
        for region_name, url in regions_to_scan.items():
            if url in scanned_urls:
                continue
            scanned_urls.add(url)
            print(f"Scanning {region_name}...")
            stations = []
            try:
                try:
                    response = requests.get(url, timeout=30, verify=ssl_verify)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    if allow_insecure_ssl:
                        try:
                            response = requests.get(url, timeout=30, verify=False)
                            response.raise_for_status()
                        except requests.exceptions.RequestException:
                            if allow_http_fallback and url.lower().startswith("https://"):
                                insecure_url = "http://" + url.split("://", 1)[1]
                                response = requests.get(insecure_url, timeout=30)
                                response.raise_for_status()
                            else:
                                raise
                    else:
                        if allow_http_fallback and url.lower().startswith("https://"):
                            insecure_url = "http://" + url.split("://", 1)[1]
                            response = requests.get(insecure_url, timeout=30)
                            response.raise_for_status()
                        else:
                            raise
                content = try_decode(response.content)
                content = clean_xml(content)
                try:
                    root = ET.fromstring(content.encode('utf-8'))
                except ET.ParseError as e:
                    print(f"Error parsing KML file {url}: {e}")
                    root = None
                if root is not None:
                    ns = {'kml': 'http://earth.google.com/kml/2.1'}
                    for placemark in root.findall('.//kml:Placemark', ns):
                        name = placemark.find('kml:name', ns)
                        desc = placemark.find('kml:description', ns)
                        point = placemark.find('.//kml:Point/kml:coordinates', ns)
                        if desc is None or not desc.text or "Data Source" not in desc.text:
                            continue
                        point_coords = None
                        if point is not None and point.text:
                            point_coords = parse_coordinates(point.text)
                        station_data = parse_station_from_description(desc.text, point_coords)
                        if station_data:
                            station_data['name'] = name.text if name is not None else "Unknown"
                            station_data['kml_source'] = url
                            stations.append(station_data)
            except requests.exceptions.RequestException as e:
                print(f"Error accessing KML file {url}: {e}")
            except Exception as e:
                print(f"Error processing KML file {url}: {e}")

            all_stations.extend(stations)
            print(f"Found {len(stations)} stations in {region_name}")

        print(f"\nTotal stations found: {len(all_stations)}")
        if not all_stations:
            if not (isinstance(region, str) and region.lower() == "all"):
                print("No stations found from detected/selected regions. Falling back to global scan...")
                regions_to_scan = dict(KML_SOURCES)
                all_stations = []
                scanned_urls = set()
                for region_name, url in regions_to_scan.items():
                    if url in scanned_urls:
                        continue
                    scanned_urls.add(url)
                    print(f"Scanning {region_name}...")
                    # re-use logic above
                    try:
                        response = requests.get(url, timeout=30, verify=ssl_verify)
                        response.raise_for_status()
                        content = try_decode(response.content)
                        content = clean_xml(content)
                        root = ET.fromstring(content.encode('utf-8'))
                        ns = {'kml': 'http://earth.google.com/kml/2.1'}
                        for placemark in root.findall('.//kml:Placemark', ns):
                            name = placemark.find('kml:name', ns)
                            desc = placemark.find('kml:description', ns)
                            point = placemark.find('.//kml:Point/kml:coordinates', ns)
                            if desc is None or not desc.text or "Data Source" not in desc.text:
                                continue
                            point_coords = None
                            if point is not None and point.text:
                                point_coords = parse_coordinates(point.text)
                            station_data = parse_station_from_description(desc.text, point_coords)
                            if station_data:
                                station_data['name'] = name.text if name is not None else "Unknown"
                                station_data['kml_source'] = url
                                all_stations.append(station_data)
                        print(f"Found {len(all_stations)} stations in {region_name}")
                    except Exception:
                        pass
                print(f"\nTotal stations found after global scan: {len(all_stations)}")
            if not all_stations:
                raise ValueError("No weather stations found")

        stations_with_distances = [
            (station, haversine_distance(longitude, latitude, station['longitude'], station['latitude']))
            for station in all_stations
        ]
        if max_distance is not None:
            close_stations = [
                (station, distance) for station, distance in stations_with_distances if distance <= max_distance
            ]
            if not close_stations:
                closest_station, min_distance = min(stations_with_distances, key=lambda x: x[1])
                print(f"\nNo stations found within {max_distance} km. Closest station is {min_distance:.1f} km away.")
                print("Using closest available station.")
                stations_with_distances = [(closest_station, min_distance)]
            else:
                stations_with_distances = close_stations

        nearest_station, distance = min(stations_with_distances, key=lambda x: x[1])
        print(f"\nDownloading EPW file for {nearest_station['name']}...")
        archive_bytes = try_download_station_zip(nearest_station['url'], timeout_s=30)
        if archive_bytes is None:
            raise ValueError(f"Failed to download EPW archive from station URL and fallbacks: {nearest_station['url']}")

        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        zip_file = temp_dir / "weather_data.zip"
        with open(zip_file, 'wb') as f:
            f.write(archive_bytes)

        final_epw = None
        try:
            if extract_zip:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    epw_files = [f for f in zip_ref.namelist() if f.lower().endswith('.epw')]
                    if not epw_files:
                        raise ValueError("No EPW file found in the downloaded archive")
                    epw_filename = epw_files[0]
                    extracted_epw = safe_extract(zip_ref, epw_filename, temp_dir)
                    final_epw = Path(output_dir) / f"{nearest_station['name'].replace(' ', '_').replace(',', '').lower()}.epw"
                    final_epw = safe_rename(extracted_epw, final_epw)
        finally:
            try:
                if zip_file.exists():
                    zip_file.unlink()
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")

        if final_epw is None:
            raise ValueError("Failed to extract EPW file")

        metadata_file = final_epw.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(nearest_station, f, indent=2)

        print(f"\nDownloaded EPW file for {nearest_station['name']}")
        print(f"Distance: {distance:.2f} km")
        print(f"Station coordinates: {nearest_station['longitude']}, {nearest_station['latitude']}")
        if nearest_station.get('wmo'):
            print(f"WMO: {nearest_station['wmo']}")
        if nearest_station.get('climate_zone'):
            print(f"Climate zone: {nearest_station['climate_zone']}")
        if nearest_station.get('period'):
            print(f"Data period: {nearest_station['period']}")
        print(f"Files saved:")
        print(f"- EPW: {final_epw}")
        print(f"- Metadata: {metadata_file}")

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


