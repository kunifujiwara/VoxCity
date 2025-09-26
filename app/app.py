# app.py (updated version with /tmp directory for outputs)
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, Geocoder
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
import os
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import zipfile
import tempfile
import glob
import requests
from geopy import distance

# Import VoxCity modules
try:
    from voxcity.generator import get_voxcity
    from voxcity.geoprocessor.draw import draw_rectangle_map_cityname, center_location_map_cityname
    from voxcity.simulator.solar import get_building_global_solar_irradiance_using_epw, get_global_solar_irradiance_using_epw
    from voxcity.simulator.view import get_view_index, get_surface_view_factor, get_landmark_visibility_map
    from voxcity.exporter.envimet import export_inx, generate_edb_file
    from voxcity.exporter.magicavoxel import export_magicavoxel_vox
    from voxcity.exporter.obj import export_obj
    from voxcity.utils.visualization import visualize_voxcity_plotly, visualize_building_sim_results, visualize_numerical_gdf_on_basemap
    from voxcity.geoprocessor.network import get_network_values
except ImportError:
    st.error("VoxCity package not installed. Please install it using: pip install voxcity")
    st.stop()

# Helper to show a visible toast on rerender
def _notify_success(message: str) -> None:
    try:
        st.toast(message)
    except Exception:
        st.success(message)

# Helpers for downloadable exports
def _zip_directory_to_bytes(dir_path: str) -> BytesIO:
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, start=dir_path)
                try:
                    zf.write(full_path, arcname)
                except Exception:
                    pass
    memory_file.seek(0)
    return memory_file

def _zip_files_to_bytes(file_paths) -> BytesIO:
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            if os.path.isfile(p):
                try:
                    zf.write(p, arcname=os.path.basename(p))
                except Exception:
                    pass
    memory_file.seek(0)
    return memory_file

# Try to import and initialize Earth Engine only if explicitly requested to avoid startup hangs
EE_AUTHENTICATED = False
if os.environ.get("VOXCITY_INIT_EE", "0") == "1":
    try:
        import ee
        try:
            ee.Initialize()
            EE_AUTHENTICATED = True
        except Exception:
            try:
                ee.Initialize(project='earthengine-legacy')
                EE_AUTHENTICATED = True
            except Exception:
                EE_AUTHENTICATED = False
    except ImportError:
        EE_AUTHENTICATED = False

# Use /tmp for all outputs - this is always writable
BASE_OUTPUT_DIR = "/tmp/voxcity_output"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Page configuration
# Use project logo if available; fall back to default iconless style
APP_DIR = os.path.dirname(__file__)
_icon_candidate = os.path.join(APP_DIR, '..', 'images', 'logo.png')
PAGE_ICON = _icon_candidate if os.path.exists(_icon_candidate) else None
st.set_page_config(
    page_title="VoxCity Web App",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Subtle, clean styling inspired by professional budgeting dashboards
st.markdown(
    """
    <style>
      :root {
        --vc-primary: #5B7C65; /* muted green */
        --vc-bg: #F7F8F6;
        --vc-surface: #FFFFFF;
        --vc-muted: #6B7280;
        --vc-ring: #E5E7EB;
      }
      header[data-testid="stHeader"] { background: var(--vc-bg); }
      /* Add extra top padding to avoid overlap with Streamlit top bar */
      .block-container { padding-top: 3.5rem; }
      div[data-testid="stSidebar"] { background: #EFF2EE; }
      .stButton>button {
        border-radius: 8px;
        border: 1px solid var(--vc-ring);
        background: var(--vc-primary);
        color: #ffffff;
      }
      .stButton>button:hover { background: #4a6854; }
      .stAlert { border-radius: 8px; }
      div[role="tablist"] button { gap: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'rectangle_vertices' not in st.session_state:
    st.session_state.rectangle_vertices = None
if 'voxcity_data' not in st.session_state:
    st.session_state.voxcity_data = None

# (Header removed per request)

# Sidebar for configuration
with st.sidebar:
    # (Configuration title removed)
    
    # Show Earth Engine status
    if EE_AUTHENTICATED:
        st.success("Google Earth Engine: Connected")
    else:
        st.warning("Google Earth Engine: Not authenticated")
        st.info("Some data sources requiring Earth Engine may not be available.")
    
    # Show output directory
    st.info(f"Output directory: {BASE_OUTPUT_DIR}")

# Main content area - Remove authentication requirement
# Create tabs for different steps
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Set Target Area", "Configure & Generate", "Notes", "Simulations", "Export"])

# Tab 1: Set Target Area
with tab1:
    # (Section title removed)
    
    area_method = st.radio("Select method to define target area:", 
                          ["Search by city name", "Draw on map", "Enter coordinates", "Set center and dimensions"],
                          index=0)

    # Simple geocoder for city names
    @st.cache_data(show_spinner=False, ttl=3600)
    def geocode_city(name: str):
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": name, "format": "json", "limit": 1}
            headers = {"User-Agent": "VoxCityApp/1.0 (streamlit)"}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            item = data[0]
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            bbox = item.get("boundingbox")
            if bbox and len(bbox) == 4:
                south, north, west, east = map(float, bbox)
                return {"lat": lat, "lon": lon, "bbox": (west, south, east, north)}
            return {"lat": lat, "lon": lon, "bbox": None}
        except Exception:
            return None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if area_method == "Enter coordinates":
            st.subheader("Enter Rectangle Vertices")
            st.info("Enter coordinates in the format: (longitude, latitude)")
            
            sw_lon = st.number_input("Southwest Longitude", value=-74.02034270713835, format="%.8f")
            sw_lat = st.number_input("Southwest Latitude", value=40.69992881162822, format="%.8f")
            nw_lon = st.number_input("Northwest Longitude", value=-74.02034270713835, format="%.8f")
            nw_lat = st.number_input("Northwest Latitude", value=40.7111851828668, format="%.8f")
            ne_lon = st.number_input("Northeast Longitude", value=-74.00555129286164, format="%.8f")
            ne_lat = st.number_input("Northeast Latitude", value=40.7111851828668, format="%.8f")
            se_lon = st.number_input("Southeast Longitude", value=-74.00555129286164, format="%.8f")
            se_lat = st.number_input("Southeast Latitude", value=40.69992881162822, format="%.8f")
            
            if st.button("Set Rectangle"):
                st.session_state.rectangle_vertices = [
                    (sw_lon, sw_lat),
                    (nw_lon, nw_lat),
                    (ne_lon, ne_lat),
                    (se_lon, se_lat)
                ]
                st.success("Rectangle vertices set successfully!")
        
        if area_method == "Search by city name":
            st.subheader("Search by City Name")
            city_name = st.text_input("Enter city name", value="New York")
            zoom_level = st.slider("Zoom level", 10, 18, 13)
            selection_mode = st.radio(
                "Selection mode",
                ["Free hand (draw)", "Set dimensions"],
                index=0,
                horizontal=True,
                key="search_select_mode",
            )
            # Show success toast if a rectangle was just applied in Set dimensions
            if st.session_state.get("show_dims_success"):
                _notify_success("Rectangle captured from map.")
                st.session_state["show_dims_success"] = False
            if selection_mode == "Set dimensions":
                col_w, col_h = st.columns(2)
                with col_w:
                    width_m = st.number_input("Width (m)", min_value=50, max_value=20000, value=1250, step=50, key="search_width_m")
                with col_h:
                    height_m = st.number_input("Height (m)", min_value=50, max_value=20000, value=1250, step=50, key="search_height_m")
            
            if st.button("Load Map"):
                with st.spinner("Loading map..."):
                    try:
                        geo = geocode_city(city_name.strip()) if city_name.strip() else None
                        if geo is None:
                            st.warning("Could not find the city. Showing default center (New York).")
                            center_lon, center_lat = -74.0129, 40.7056
                            city_bbox = None
                        else:
                            center_lon, center_lat = geo["lon"], geo["lat"]
                            city_bbox = geo.get("bbox")
                        # Persist center/zoom to keep map visible after interactions
                        st.session_state["search_map_center"] = (center_lat, center_lon)
                        st.session_state["search_map_zoom"] = zoom_level
                        st.session_state["search_map_bbox"] = city_bbox
                        if st.session_state.get("show_dims_success"):
                            _notify_success("Rectangle captured from map.")
                            st.session_state["show_dims_success"] = False
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
                        # Simple geocoder control (browser-side) to refine center
                        try:
                            Geocoder(collapsed=True, add_marker=False).add_to(m)
                        except Exception:
                            pass
                        if selection_mode == "Free hand (draw)":
                            Draw(
                                export=False,
                                draw_options={
                                    'polyline': False,
                                    'polygon': False,
                                    'circle': False,
                                    'circlemarker': False,
                                    'marker': False,
                                    'rectangle': {
                                        'shapeOptions': {
                                            'color': '#3388ff',
                                            'fillColor': '#3388ff',
                                            'fillOpacity': 0.2
                                        }
                                    }
                                },
                                edit_options={'edit': True}
                            ).add_to(m)
                        else:
                            st.info("Click the map to set the rectangle center, then press 'Apply dimensions'.")
                            try:
                                m.get_root().html.add_child(folium.Element("<style>.leaflet-container{cursor:crosshair !important;}</style>"))
                            except Exception:
                                pass
                        returned_objs = ["last_active_drawing", "all_drawings", "last_drawn", "last_clicked"]
                        if selection_mode == "Set dimensions":
                            returned_objs += ["zoom", "center"]
                        out = st_folium(m, height=500, width=700, key="search_map", returned_objects=returned_objs)            
                        feature = None
                        if isinstance(out, dict):
                            feature = out.get('last_active_drawing') or out.get('last_drawn')
                            if (feature is None) and out.get('all_drawings'):
                                drawings = out.get('all_drawings') or []
                                feature = drawings[-1] if len(drawings) > 0 else None
                        if feature and (selection_mode == "Free hand (draw)"):
                            geometry = feature.get('geometry', feature)
                            if geometry and geometry.get('type') == 'Polygon':
                                coords = geometry['coordinates'][0]
                                lons = [c[0] for c in coords]
                                lats = [c[1] for c in coords]
                                lon_min, lon_max = min(lons), max(lons)
                                lat_min, lat_max = min(lats), max(lats)
                                st.session_state.rectangle_vertices = [
                                    (lon_min, lat_min),
                                    (lon_min, lat_max),
                                    (lon_max, lat_max),
                                    (lon_max, lat_min)
                                ]
                                st.success("Rectangle captured from map.")
                        if selection_mode == "Set dimensions":
                            last_click = None
                            if isinstance(out, dict):
                                lc = out.get('last_clicked')
                                if lc and isinstance(lc, dict) and ('lat' in lc) and ('lng' in lc):
                                    last_click = (float(lc['lng']), float(lc['lat']))
                            prev_click = st.session_state.get("search_last_click")
                            if last_click is not None:
                                # Only update/apply if the click changed
                                if (prev_click is None) or (prev_click != last_click):
                                    st.session_state["search_last_click"] = last_click
                                    center_used = last_click
                                    # Use current dimension inputs from widgets (persisted by keys)
                                    w_m = st.session_state.get("search_width_m", 1250)
                                    h_m = st.session_state.get("search_height_m", 1250)
                                    # Use geodesic bearings to compute accurate rectangle corners around the clicked center
                                    lon_c, lat_c = center_used
                                    north = distance.distance(meters=h_m / 2.0).destination((lat_c, lon_c), bearing=0)
                                    south = distance.distance(meters=h_m / 2.0).destination((lat_c, lon_c), bearing=180)
                                    east = distance.distance(meters=w_m / 2.0).destination((lat_c, lon_c), bearing=90)
                                    west = distance.distance(meters=w_m / 2.0).destination((lat_c, lon_c), bearing=270)
                                    st.session_state.rectangle_vertices = [
                                        (west.longitude, south.latitude),
                                        (west.longitude, north.latitude),
                                        (east.longitude, north.latitude),
                                        (east.longitude, south.latitude)
                                    ]
                                    # Persist current map view (zoom/center) to avoid jumps after rerun
                                    if isinstance(out, dict):
                                        z = out.get('zoom')
                                        c = out.get('center')
                                        if isinstance(z, (int, float)):
                                            st.session_state["search_map_zoom"] = int(z)
                                        if isinstance(c, dict) and ('lat' in c) and ('lng' in c):
                                            st.session_state["search_map_center"] = (float(c['lat']), float(c['lng']))
                                    # Show toast before rerun by storing a flag
                                    st.session_state["show_dims_success"] = True
                                    st.rerun()
                                else:
                                    st.caption(f"Center set at lon {last_click[0]:.6f}, lat {last_click[1]:.6f}")
                                st.session_state["show_dims_success"] = True
                                st.rerun()
                        # Offer to use city bounding box directly
                        if (st.session_state.rectangle_vertices is None) and city_bbox:
                            if st.button("Use city bounding box as rectangle"):
                                west, south, east, north = city_bbox
                                st.session_state.rectangle_vertices = [
                                    (west, south), (west, north), (east, north), (east, south)
                                ]
                                st.success("Rectangle set from city bounding box.")
                    except Exception as e:
                        st.error(f"Error loading map: {str(e)}")
            # Keep map visible after draw/app reruns using persisted center/zoom
            elif st.session_state.get("search_map_center"):
                center_lat, center_lon = st.session_state["search_map_center"]
                zoom = st.session_state.get("search_map_zoom", 13)
                if st.session_state.get("show_dims_success"):
                    _notify_success("Rectangle captured from map.")
                    st.session_state["show_dims_success"] = False
                m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
                try:
                    Geocoder(collapsed=True, add_marker=False).add_to(m)
                except Exception:
                    pass
                if selection_mode == "Free hand (draw)":
                    Draw(
                        export=False,
                        draw_options={
                            'polyline': False,
                            'polygon': False,
                            'circle': False,
                            'circlemarker': False,
                            'marker': False,
                            'rectangle': {
                                'shapeOptions': {
                                    'color': '#3388ff',
                                    'fillColor': '#3388ff',
                                    'fillOpacity': 0.2
                                }
                            }
                        },
                        edit_options={'edit': True}
                    ).add_to(m)
                # Overlay existing rectangle if available (with consistent draw color)
                if st.session_state.rectangle_vertices:
                    folium.Rectangle(
                        bounds=[
                            [st.session_state.rectangle_vertices[0][1], st.session_state.rectangle_vertices[0][0]],
                            [st.session_state.rectangle_vertices[2][1], st.session_state.rectangle_vertices[2][0]]
                        ],
                        color='#3388ff',
                        fill=True,
                        fillColor='#3388ff',
                        fillOpacity=0.2
                    ).add_to(m)
                # Always render map with a stable key and capture clicks/draws
                returned_objs = ["last_active_drawing", "all_drawings", "last_drawn", "last_clicked"]
                if selection_mode == "Set dimensions":
                    returned_objs += ["zoom", "center"]
                out = st_folium(m, height=500, width=700, key="search_map", returned_objects=returned_objs)            
                if selection_mode == "Free hand (draw)":
                    feature = None
                    if isinstance(out, dict):
                        feature = out.get('last_active_drawing') or out.get('last_drawn')
                        if (feature is None) and out.get('all_drawings'):
                            drawings = out.get('all_drawings') or []
                            feature = drawings[-1] if len(drawings) > 0 else None
                    if feature:
                        geometry = feature.get('geometry', feature)
                        if geometry and geometry.get('type') == 'Polygon':
                            coords = geometry['coordinates'][0]
                            lons = [c[0] for c in coords]
                            lats = [c[1] for c in coords]
                            lon_min, lon_max = min(lons), max(lons)
                            lat_min, lat_max = min(lats), max(lats)
                            st.session_state.rectangle_vertices = [
                                (lon_min, lat_min),
                                (lon_min, lat_max),
                                (lon_max, lat_max),
                                (lon_max, lat_min)
                            ]
                            st.success("Rectangle captured from map.")
                else:
                    # Set-dimensions: capture center click and immediately apply current dimensions
                    last_click = None
                    if isinstance(out, dict):
                        lc = out.get('last_clicked')
                        if lc and isinstance(lc, dict) and ('lat' in lc) and ('lng' in lc):
                            last_click = (float(lc['lng']), float(lc['lat']))
                    prev_click = st.session_state.get("search_last_click")
                    if last_click is not None:
                        if (prev_click is None) or (prev_click != last_click):
                            st.session_state["search_last_click"] = last_click
                            center_used = last_click
                            # Pull current dimension inputs stored by keys
                            w_m = st.session_state.get("search_width_m", 1250)
                            h_m = st.session_state.get("search_height_m", 1250)
                            # Use geodesic bearings to compute accurate rectangle corners around the clicked center
                            lon_c, lat_c = center_used
                            north = distance.distance(meters=h_m / 2.0).destination((lat_c, lon_c), bearing=0)
                            south = distance.distance(meters=h_m / 2.0).destination((lat_c, lon_c), bearing=180)
                            east = distance.distance(meters=w_m / 2.0).destination((lat_c, lon_c), bearing=90)
                            west = distance.distance(meters=w_m / 2.0).destination((lat_c, lon_c), bearing=270)
                            st.session_state.rectangle_vertices = [
                                (west.longitude, south.latitude),
                                (west.longitude, north.latitude),
                                (east.longitude, north.latitude),
                                (east.longitude, south.latitude)
                            ]
                            # Persist current map view (zoom/center) to avoid jumps after rerun
                            z = out.get('zoom') if isinstance(out, dict) else None
                            c = out.get('center') if isinstance(out, dict) else None
                            if isinstance(z, (int, float)):
                                st.session_state["search_map_zoom"] = int(z)
                            if isinstance(c, dict) and ('lat' in c) and ('lng' in c):
                                st.session_state["search_map_center"] = (float(c['lat']), float(c['lng']))
                            st.rerun()
                        else:
                            st.caption(f"Center set at lon {last_click[0]:.6f}, lat {last_click[1]:.6f}")
                        # Immediately draw rectangle on the same map
                        try:
                            folium.Rectangle(
                                bounds=[
                                    [st.session_state.rectangle_vertices[0][1], st.session_state.rectangle_vertices[0][0]],
                                    [st.session_state.rectangle_vertices[2][1], st.session_state.rectangle_vertices[2][0]]
                                ],
                                color='#3388ff',
                                fill=True,
                                fillColor='#3388ff',
                                fillOpacity=0.2
                            ).add_to(m)
                        except Exception:
                            pass
                        # Force a re-render to keep map with overlay
                        st.session_state["show_dims_success"] = True
                        st.rerun()
        elif area_method == "Draw on map":
            st.subheader("Draw Rectangle on Interactive Map")
            center_lon = st.number_input("Map Center Longitude", value=-74.0129, format="%.6f")
            center_lat = st.number_input("Map Center Latitude", value=40.7056, format="%.6f")
            zoom_level = st.slider("Zoom level", 10, 18, 13, key="draw_zoom")
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
            try:
                Geocoder(collapsed=True, add_marker=False).add_to(m)
            except Exception:
                pass
            Draw(
                export=False,
                draw_options={
                    'polyline': False,
                    'polygon': False,
                    'circle': False,
                    'circlemarker': False,
                    'marker': False,
                    'rectangle': {
                        'shapeOptions': {
                            'color': '#3388ff',
                            'fillColor': '#3388ff',
                            'fillOpacity': 0.2
                        }
                    }
                },
                edit_options={'edit': True}
            ).add_to(m)
            out = st_folium(m, height=500, width=700, returned_objects=["last_active_drawing", "all_drawings", "last_drawn"])            
            feature = None
            if isinstance(out, dict):
                feature = out.get('last_active_drawing') or out.get('last_drawn')
                if (feature is None) and out.get('all_drawings'):
                    drawings = out.get('all_drawings') or []
                    feature = drawings[-1] if len(drawings) > 0 else None
            if feature:
                geometry = feature.get('geometry', feature)
                if geometry and geometry.get('type') == 'Polygon':
                    coords = geometry['coordinates'][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    lon_min, lon_max = min(lons), max(lons)
                    lat_min, lat_max = min(lats), max(lats)
                    st.session_state.rectangle_vertices = [
                        (lon_min, lat_min),
                        (lon_min, lat_max),
                        (lon_max, lat_max),
                        (lon_max, lat_min)
                    ]
                    st.success("Rectangle captured from map.")
        
        elif area_method == "Set center and dimensions":
            st.subheader("Set Center Location and Dimensions")
            center_lon = st.number_input("Center Longitude", value=-74.0129, format="%.6f")
            center_lat = st.number_input("Center Latitude", value=40.7056, format="%.6f")
            width = st.number_input("Width (meters)", value=1250, min_value=100, max_value=10000, step=50)
            height = st.number_input("Height (meters)", value=1250, min_value=100, max_value=10000, step=50)
            
            if st.button("Calculate Rectangle"):
                # Simple calculation for rectangle from center and dimensions
                # In reality, you would use proper geographic calculations
                degree_per_meter_lat = 1 / 111000
                degree_per_meter_lon = 1 / (111000 * np.cos(np.radians(center_lat)))
                
                half_width_deg = (width / 2) * degree_per_meter_lon
                half_height_deg = (height / 2) * degree_per_meter_lat
                
                st.session_state.rectangle_vertices = [
                    (center_lon - half_width_deg, center_lat - half_height_deg),
                    (center_lon - half_width_deg, center_lat + half_height_deg),
                    (center_lon + half_width_deg, center_lat + half_height_deg),
                    (center_lon + half_width_deg, center_lat - half_height_deg)
                ]
                st.success("Rectangle calculated successfully!")
        else:  # Draw on map
            st.subheader("Draw Rectangle on Interactive Map")
            center_lon = st.number_input("Map Center Longitude", value=-74.0129, format="%.6f")
            center_lat = st.number_input("Map Center Latitude", value=40.7056, format="%.6f")
            zoom_level = st.slider("Zoom level", 10, 18, 15, key="draw_zoom")
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
            Draw(
                export=False,
                draw_options={
                    'polyline': False,
                    'polygon': False,
                    'circle': False,
                    'circlemarker': False,
                    'marker': False,
                    'rectangle': {
                        'shapeOptions': {
                            'color': '#3388ff',
                            'fillColor': '#3388ff',
                            'fillOpacity': 0.2
                        }
                    }
                },
                edit_options={'edit': True}
            ).add_to(m)
            out = st_folium(m, height=500, width=700, returned_objects=["last_active_drawing", "all_drawings", "last_drawn"])            
            feature = None
            if isinstance(out, dict):
                feature = out.get('last_active_drawing') or out.get('last_drawn')
                if (feature is None) and out.get('all_drawings'):
                    drawings = out.get('all_drawings') or []
                    feature = drawings[-1] if len(drawings) > 0 else None
            if feature:
                geometry = feature.get('geometry', feature)
                if geometry and geometry.get('type') == 'Polygon':
                    coords = geometry['coordinates'][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    lon_min, lon_max = min(lons), max(lons)
                    lat_min, lat_max = min(lats), max(lats)
                    st.session_state.rectangle_vertices = [
                        (lon_min, lat_min),
                        (lon_min, lat_max),
                        (lon_max, lat_max),
                        (lon_max, lat_min)
                    ]
                    st.success("Rectangle captured from map.")
    
    with col2:
        if st.session_state.rectangle_vertices and (area_method in ["Enter coordinates", "Set center and dimensions"]):
            st.subheader("Selected Area Preview")
            # Create a simple folium map to show the selected area
            m = folium.Map(location=[
                (st.session_state.rectangle_vertices[0][1] + st.session_state.rectangle_vertices[2][1]) / 2,
                (st.session_state.rectangle_vertices[0][0] + st.session_state.rectangle_vertices[2][0]) / 2
            ], zoom_start=15)
            
            # Add rectangle to map
            folium.Rectangle(
                bounds=[[v[1], v[0]] for v in st.session_state.rectangle_vertices[:3:2]],
                color='#3388ff',
                fill=True,
                fillColor='#3388ff',
                fillOpacity=0.2
            ).add_to(m)
            
            st_folium(m, height=400, width=400)

# Tab 2: Configure Parameters
with tab2:
    # (Section title removed)
    
    # Filter out data sources that require Earth Engine if not authenticated
    if EE_AUTHENTICATED:
        building_sources = ['OpenStreetMap', 'Overture', 'EUBUCCO v0.1', 'Open Building 2.5D Temporal', 
                          'Microsoft Building Footprints', 'Local file']
        land_cover_sources = ['OpenStreetMap', 'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover', 
                            'ESRI 10m Annual Land Cover', 'Dynamic World V1']
        canopy_height_sources = ['High Resolution 1m Global Canopy Height Maps', 
                               'ETH Global Sentinel-2 10m Canopy Height (2020)', 'Static']
        dem_sources = ['DeltaDTM', 'FABDEM', 'England 1m DTM', 'DEM France 1m', 
                      'Netherlands 0.5m DTM', 'AUSTRALIA 5M DEM', 'USGS 3DEP 1m', 
                      'NASA', 'COPERNICUS', 'Flat']
    else:
        # Limit to sources that don't require Earth Engine
        building_sources = ['OpenStreetMap', 'Local file']
        land_cover_sources = ['OpenStreetMap']
        canopy_height_sources = ['Static']
        dem_sources = ['Flat']
        
        # (Informational note removed by request)
    
    # Narrower controls column (half the previous width)
    left_col, right_col = st.columns([1, 5])

    with left_col:
        st.subheader("Data Sources")
        building_source = st.selectbox("Building Source", building_sources)
        building_complementary_source = st.selectbox(
            "Building Complementary Source",
            ['None'] + [s for s in building_sources if s != 'Local file']
        )
        land_cover_source = st.selectbox("Land Cover Source", land_cover_sources)
        canopy_height_source = st.selectbox("Canopy Height Source", canopy_height_sources)
        dem_source = st.selectbox("DEM Source", dem_sources)

        st.subheader("Parameters")
        meshsize = st.number_input("Mesh Size (meters)", value=5, min_value=1, max_value=50)
        
        with st.expander("Advanced Parameters"):
            building_complement_height = st.number_input(
                "Building Complement Height (m)", 
                value=10, 
                help="Default height for buildings when height data is missing"
            )
            
            overlapping_footprint = st.checkbox(
                "Use Overlapping Footprints", 
                value=False
            )
            
            dem_interpolation = st.checkbox(
                "DEM Interpolation", 
                value=True,
                help="Use interpolation when mesh size is finer than DEM resolution"
            )
            
            debug_voxel = st.checkbox(
                "Debug Mode", 
                value=True,
                help="Enable step logs"
            )

    with right_col:
        st.subheader("Generate Model & Preview")
        if st.session_state.rectangle_vertices is None:
            st.warning("Set the target area first in the 'Set Target Area' tab.")
        else:
            generate_clicked_cfg = st.button("Generate VoxCity Model", type="primary")
            if generate_clicked_cfg:
                with st.spinner("Generating 3D city model... This may take several minutes."):
                    try:
                        output_dir = os.path.join(BASE_OUTPUT_DIR, "test")
                        os.makedirs(output_dir, exist_ok=True)
                        kwargs = {
                            "building_complementary_source": building_complementary_source,
                            "building_complement_height": building_complement_height,
                            "overlapping_footprint": overlapping_footprint,
                            "output_dir": output_dir,
                            "dem_interpolation": dem_interpolation,
                            "debug_voxel": debug_voxel,
                        }
                        result = get_voxcity(
                            st.session_state.rectangle_vertices,
                            building_source,
                            land_cover_source,
                            canopy_height_source,
                            dem_source,
                            meshsize,
                            **kwargs
                        )
                        st.session_state.voxcity_data = {
                            'voxcity_grid': result[0],
                            'building_height_grid': result[1],
                            'building_min_height_grid': result[2],
                            'building_id_grid': result[3],
                            'canopy_height_grid': result[4],
                            'canopy_bottom_height_grid': result[5],
                            'land_cover_grid': result[6],
                            'dem_grid': result[7],
                            'building_gdf': result[8],
                            'meshsize': meshsize,
                            'rectangle_vertices': st.session_state.rectangle_vertices
                        }
                        # Show preview only (no extra messages/metrics)
                        data = st.session_state.voxcity_data
                        with st.spinner("Rendering 3D view..."):
                            try:
                                fig = visualize_voxcity_plotly(
                                    data['voxcity_grid'],
                                    data['meshsize'],
                                    downsample=1,
                                    show=False,
                                    return_fig=True,
                                    title="VoxCity 3D"
                                )
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Visualization error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error generating VoxCity model: {str(e)}")
                        st.exception(e)

            # Persistent preview if already generated
            if (st.session_state.voxcity_data is not None) and (not generate_clicked_cfg):
                data = st.session_state.voxcity_data
                try:
                    fig = visualize_voxcity_plotly(
                        data['voxcity_grid'],
                        data['meshsize'],
                        downsample=1,
                        show=False,
                        return_fig=True,
                        title="VoxCity 3D"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

# Tab 3: Generate Model
with tab3:
    # (Section title removed)
    st.info("Model generation has been integrated into the 'Configure & Generate' tab for a streamlined workflow.")

# Tab 4: Simulations
with tab4:
    # (Section title removed)
    
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generate Model' tab.")
    else:
        simulation_type = st.selectbox(
            "Select Simulation Type",
            ["Solar Radiation", "View Index", "Landmark Visibility"]
        )
        
        if simulation_type == "Solar Radiation":
            st.subheader("Solar Radiation Analysis")
            ctrl_col, vis_col = st.columns([1, 2])

            with ctrl_col:
                solar_calc_type = st.radio("Calculation Type", ["Instantaneous", "Cumulative"]) 
                if solar_calc_type == "Instantaneous":
                    default_inst_date = st.session_state.get('solar_inst_date', datetime.now().date())
                    default_inst_time = st.session_state.get('solar_inst_time', datetime.strptime("12:00:00", "%H:%M:%S").time())
                    calc_date = st.date_input("Date (MM-DD)", value=default_inst_date, key="solar_inst_date")
                    calc_time = st.time_input("Time (HH:MM:SS)", value=default_inst_time, key="solar_inst_time")
                    calc_datetime = f"{calc_date.strftime('%m-%d')} {calc_time.strftime('%H:%M:%S')}"
                else:
                    default_start_date = datetime(datetime.now().year, 1, 1).date()
                    default_end_date = datetime(datetime.now().year, 1, 31).date()
                    start_date = st.date_input("Start Date (MM-DD)", default_start_date)
                    start_time = st.time_input("Start Time", datetime.strptime("01:00:00", "%H:%M:%S").time())
                    end_date = st.date_input("End Date (MM-DD)", default_end_date)
                    end_time = st.time_input("End Time", datetime.strptime("23:00:00", "%H:%M:%S").time())
                analysis_target = st.radio("Analysis Target", ["Ground Level", "Building Surfaces"])
                download_epw = st.checkbox("Download nearest EPW file", value=True)
                if not download_epw:
                    epw_file = st.file_uploader("Upload EPW file", type=['epw'])
                run_solar = st.button("Run Solar Analysis")

            if run_solar:
                with st.spinner("Running solar radiation analysis..."):
                    try:
                        data = st.session_state.voxcity_data
                        
                        # Save uploaded EPW to a temp path if provided
                        epw_local_path = None
                        if (not download_epw) and ('epw_file' in locals()) and (epw_file is not None):
                            try:
                                epw_dir = os.path.join(BASE_OUTPUT_DIR, 'epw')
                                os.makedirs(epw_dir, exist_ok=True)
                                epw_local_path = os.path.join(epw_dir, epw_file.name)
                                with open(epw_local_path, 'wb') as _f:
                                    _f.write(epw_file.read())
                            except Exception as _e:
                                st.warning(f"Failed to persist EPW file: {_e}")

                        if analysis_target == "Ground Level":
                            solar_kwargs = {
                                "download_nearest_epw": download_epw,
                                "rectangle_vertices": data['rectangle_vertices'],
                                "view_point_height": 1.5,
                                "tree_k": 0.6,
                                "tree_lad": 0.5,
                                "dem_grid": data['dem_grid'],
                                "colormap": 'magma',
                                "obj_export": False,
                                "output_directory": os.path.join(BASE_OUTPUT_DIR, 'solar'),
                                "output_dir": os.path.join(BASE_OUTPUT_DIR, 'epw'),
                                "alpha": 1.0,
                                "vmin": 0,
                            }
                            if epw_local_path:
                                solar_kwargs["epw_file_path"] = epw_local_path
                            
                            # Create output directories (for EPW downloads and outputs)
                            os.makedirs(solar_kwargs["output_directory"], exist_ok=True)
                            try:
                                os.makedirs(solar_kwargs["output_dir"], exist_ok=True)
                            except Exception:
                                pass
                            
                            if solar_calc_type == "Instantaneous":
                                solar_kwargs["calc_time"] = calc_datetime
                                calc_type = 'instantaneous'
                            else:
                                solar_kwargs["start_time"] = f"{start_date.strftime('%m-%d')} {start_time.strftime('%H:%M:%S')}"
                                solar_kwargs["end_time"] = f"{end_date.strftime('%m-%d')} {end_time.strftime('%H:%M:%S')}"
                                calc_type = 'cumulative'
                            
                            solar_grid = get_global_solar_irradiance_using_epw(
                                data['voxcity_grid'],
                                data['meshsize'],
                                calc_type=calc_type,
                                direct_normal_irradiance_scaling=1.0,
                                diffuse_irradiance_scaling=1.0,
                                **solar_kwargs
                            )
                            
                            st.success("Solar analysis completed!")
                            
                            # 3D Plotly visualization of ground-level results (left panel)
                            with vis_col:
                                with st.spinner("Rendering 3D overlay..."):
                                    try:
                                        fig = visualize_voxcity_plotly(
                                            data['voxcity_grid'],
                                            data['meshsize'],
                                            downsample=1,
                                            voxel_color_map='grayscale',
                                            ground_sim_grid=solar_grid,
                                            ground_dem_grid=data['dem_grid'],
                                            ground_view_point_height=1.5,
                                            ground_colormap='magma',
                                            ground_vmin=0.0,
                                            sim_surface_opacity=0.95,
                                            show=False,
                                            return_fig=True,
                                            title="Solar overlay"
                                        )
                                        if fig is not None:
                                            st.plotly_chart(fig, use_container_width=True)
                                            try:
                                                st.caption(
                                                    f"Grid: {data['voxcity_grid'].shape} • Buildings: {len(data['building_gdf'])} • Meshsize: {data['meshsize']} m"
                                                )
                                            except Exception:
                                                pass
                                        else:
                                            st.info("No 3D overlay generated.")
                                    except Exception as ee:
                                        st.warning(f"3D overlay rendering failed: {ee}")
                            
                        else:  # Building Surfaces
                            irradiance_kwargs = {
                                "download_nearest_epw": download_epw,
                                "rectangle_vertices": data['rectangle_vertices'],
                                "building_id_grid": data['building_id_grid']
                            }
                            if epw_local_path:
                                irradiance_kwargs["epw_file_path"] = epw_local_path
                            else:
                                # Ensure EPW download directory exists when downloading
                                irradiance_kwargs["output_dir"] = os.path.join(BASE_OUTPUT_DIR, 'epw')
                                try:
                                    os.makedirs(irradiance_kwargs["output_dir"], exist_ok=True)
                                except Exception:
                                    pass
                            
                            if solar_calc_type == "Instantaneous":
                                irradiance_kwargs["calc_type"] = "instantaneous"
                                irradiance_kwargs["calc_time"] = calc_datetime
                            else:
                                irradiance_kwargs["calc_type"] = "cumulative"
                                irradiance_kwargs["period_start"] = f"{start_date.strftime('%m-%d')} {start_time.strftime('%H:%M:%S')}"
                                irradiance_kwargs["period_end"] = f"{end_date.strftime('%m-%d')} {end_time.strftime('%H:%M:%S')}"
                            
                            with st.spinner("Computing building-surface solar irradiance..."):
                                irradiance = get_building_global_solar_irradiance_using_epw(
                                    data['voxcity_grid'],
                                    data['meshsize'],
                                    **irradiance_kwargs
                                )
                            
                            st.success("Building solar analysis completed!")
                            # Visualize building-surface irradiance in 3D (left panel)
                            with vis_col:
                                try:
                                    fig_b = visualize_voxcity_plotly(
                                        data['voxcity_grid'],
                                        data['meshsize'],
                                        downsample=1,
                                        voxel_color_map='grayscale',
                                        building_sim_mesh=irradiance,
                                        building_value_name='global',
                                        building_colormap='magma',
                                        building_vmin=None,
                                        building_vmax=None,
                                        building_opacity=1.0,
                                        building_shaded=False,
                                        render_voxel_buildings=False,
                                        show=False,
                                        return_fig=True,
                                        title="Building Surface Solar (Global)"
                                    )
                                    if fig_b is not None:
                                        st.plotly_chart(fig_b, use_container_width=True)
                                        try:
                                            st.caption(
                                                f"Grid: {data['voxcity_grid'].shape} • Buildings: {len(data['building_gdf'])} • Meshsize: {data['meshsize']} m"
                                            )
                                        except Exception:
                                            pass
                                    else:
                                        st.info("No building-surface visualization generated.")
                                except Exception as ve:
                                    st.warning(f"3D building visualization failed: {ve}")
                            
                    except Exception as e:
                        st.error(f"Error in solar analysis: {str(e)}")
        
        elif simulation_type == "View Index":
            st.subheader("View Index Analysis")
            ctrl_col, vis_col = st.columns([1, 2])

            with ctrl_col:
                view_type = st.selectbox("View Type", ["Green View Index", "Sky View Index"])
                view_point_height = st.number_input("View Point Height (m)", value=1.5, min_value=0.0, max_value=10.0)
                tree_k = st.slider("Tree Extinction Coefficient", 0.0, 1.0, 0.6)
                tree_lad = st.slider("Tree Leaf Area Density", 0.0, 2.0, 1.0)
                colormap = st.selectbox("Colormap", ["viridis", "BuPu_r", "RdYlGn", "coolwarm"])
                export_obj = st.checkbox("Export as OBJ file", value=False)
                run_view = st.button("Calculate View Index")
            
            if run_view:
                with st.spinner(f"Calculating {view_type}..."):
                    try:
                        data = st.session_state.voxcity_data
                        
                        output_dir = os.path.join(BASE_OUTPUT_DIR, "view")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        view_kwargs = {
                            "view_point_height": view_point_height,
                            "tree_k": tree_k,
                            "tree_lad": tree_lad,
                            "dem_grid": data['dem_grid'],
                            "colormap": colormap,
                            "obj_export": export_obj,
                            "output_directory": output_dir,
                            "output_file_name": "gvi" if view_type == "Green View Index" else "svi"
                        }
                        
                        mode = 'green' if view_type == "Green View Index" else 'sky'
                        view_grid = get_view_index(
                            data['voxcity_grid'], 
                            data['meshsize'], 
                            mode=mode, 
                            **view_kwargs
                        )
                        
                        st.success(f"{view_type} calculated successfully!")
                        
                        # Display results (left panel)
                        with vis_col:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(view_grid, cmap=colormap, origin='lower', vmin=0, vmax=1)
                            plt.colorbar(im, ax=ax, label=view_type)
                            ax.set_title(view_type)
                            st.pyplot(fig)
                        
                        # Optional 3D multi-view overlay
                        if st.checkbox("Also render 3D views with overlay", key="view_overlay"):
                            with st.spinner("Rendering 3D overlay..."):
                                try:
                                    fig = visualize_voxcity_plotly(
                                        data['voxcity_grid'],
                                        data['meshsize'],
                                        downsample=1,
                                        voxel_color_map='grayscale',
                                        ground_sim_grid=view_grid,
                                        ground_dem_grid=data['dem_grid'],
                                        ground_view_point_height=view_point_height,
                                        ground_colormap=colormap,
                                        ground_vmin=0.0,
                                        ground_vmax=1.0,
                                        sim_surface_opacity=0.95,
                                        show=False,
                                        return_fig=True,
                                        title=view_type
                                    )
                                    if fig is not None:
                                        st.plotly_chart(fig, use_container_width=True)
                                        try:
                                            st.caption(
                                                f"Grid: {data['voxcity_grid'].shape} • Buildings: {len(data['building_gdf'])} • Meshsize: {data['meshsize']} m"
                                            )
                                        except Exception:
                                            pass
                                    else:
                                        st.info("No 3D overlay generated.")
                                except Exception as ee:
                                    st.warning(f"3D overlay rendering failed: {ee}")
                        
                    except Exception as e:
                        st.error(f"Error calculating view index: {str(e)}")
        
        else:  # Landmark Visibility
            st.subheader("Landmark Visibility Analysis")
            ctrl_col, vis_col = st.columns([1, 2])
            with ctrl_col:
                st.caption("Optionally enter landmark building IDs (comma-separated). If left blank, the center building of the rectangle is used.")
                ids_text = st.text_input("Landmark Building IDs (optional)", value="")
                export_obj_landmark = st.checkbox("Export landmark visibility OBJ (surfaces)", value=False)
                run_landmark = st.button("Run Landmark Visibility")
            if run_landmark:
                with st.spinner("Computing landmark visibility..."):
                    try:
                        data = st.session_state.voxcity_data
                        kwargs_vis = {
                            "rectangle_vertices": data['rectangle_vertices'],
                            "obj_export": export_obj_landmark,
                            "output_directory": os.path.join(BASE_OUTPUT_DIR, 'landmark'),
                        }
                        os.makedirs(kwargs_vis["output_directory"], exist_ok=True)
                        if ids_text.strip():
                            try:
                                ids = [int(x.strip()) for x in ids_text.split(',') if x.strip()]
                                kwargs_vis["landmark_building_ids"] = ids
                            except Exception:
                                st.warning("Could not parse IDs; falling back to center building detection.")
                        vis_map, vox_marked = get_landmark_visibility_map(
                            data['voxcity_grid'],
                            data['building_id_grid'],
                            data['building_gdf'],
                            data['meshsize'],
                            **kwargs_vis
                        )
                        if vis_map is None:
                            st.warning("No landmarks found or visible.")
                        else:
                            with vis_col:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(vis_map, origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
                                plt.colorbar(im, ax=ax, label='Landmark Visibility (0/1)')
                                ax.set_title('Landmark Visibility')
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error computing landmark visibility: {e}")

# Tab 5: Export
with tab5:
    # (Section title removed)
    
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generate Model' tab.")
    else:
        export_format = st.selectbox(
            "Select Export Format",
            ["ENVI-MET INX", "MagicaVoxel VOX", "OBJ File"]
        )
        
        if export_format == "ENVI-MET INX":
            st.subheader("Export to ENVI-MET")
            
            col1, col2 = st.columns(2)
            
            with col1:
                author_name = st.text_input("Author Name", value="VoxCity User")
                model_description = st.text_area("Model Description", value="Generated using VoxCity Web App")
                file_basename = st.text_input("File Base Name", value="voxcity")
            
            with col2:
                domain_ratio = st.number_input("Domain/Building Height Ratio", value=2.0, min_value=1.0, max_value=5.0)
                use_telescoping = st.checkbox("Use Telescoping Grid", value=True)
                vertical_stretch = st.number_input("Vertical Stretch (%)", value=20, min_value=0, max_value=100)
                min_grids_z = st.number_input("Minimum Vertical Grids", value=20, min_value=10, max_value=100)
                lad = st.number_input("Leaf Area Density", value=1.0, min_value=0.1, max_value=5.0)
            
            if st.button("Export ENVI-MET Files"):
                with st.spinner("Exporting ENVI-MET files..."):
                    try:
                        data = st.session_state.voxcity_data
                        
                        output_dir = os.path.join(BASE_OUTPUT_DIR, 'envimet')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        envimet_kwargs = {
                            "output_directory": output_dir,
                            "file_basename": file_basename,
                            "author_name": author_name,
                            "model_desctiption": model_description,
                            "domain_building_max_height_ratio": domain_ratio,
                            "useTelescoping_grid": use_telescoping,
                            "verticalStretch": vertical_stretch,
                            "min_grids_Z": min_grids_z,
                            "lad": lad
                        }
                        
                        export_inx(
                            data['building_height_grid'],
                            data['building_id_grid'],
                            data['canopy_height_grid'],
                            data['land_cover_grid'],
                            data['dem_grid'],
                            data['meshsize'],
                            land_cover_source,
                            data['rectangle_vertices'],
                            **envimet_kwargs
                        )
                        
                        generate_edb_file(**envimet_kwargs)
                        
                        st.success("ENVI-MET files exported successfully!")
                        st.info(f"Files saved to {output_dir}")
                        try:
                            zip_buf = _zip_directory_to_bytes(output_dir)
                            st.download_button(
                                label="Download ENVI-MET outputs (ZIP)",
                                data=zip_buf,
                                file_name="envimet_outputs.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Could not prepare ZIP for download: {e}")
                    except Exception as e:
                        st.error(f"Error exporting ENVI-MET files: {str(e)}")
        
        elif export_format == "MagicaVoxel VOX":
            st.subheader("Export to MagicaVoxel")
            
            if st.button("Export VOX File"):
                with st.spinner("Exporting MagicaVoxel file..."):
                    try:
                        data = st.session_state.voxcity_data
                        
                        output_path = os.path.join(BASE_OUTPUT_DIR, "magicavoxel")
                        os.makedirs(output_path, exist_ok=True)
                        
                        export_magicavoxel_vox(data['voxcity_grid'], output_path)
                        
                        st.success("MagicaVoxel file exported successfully!")
                        st.info(f"File saved to {output_path}")
                        try:
                            zip_buf = _zip_directory_to_bytes(output_path)
                            st.download_button(
                                label="Download MagicaVoxel outputs (ZIP)",
                                data=zip_buf,
                                file_name="magicavoxel_outputs.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Could not prepare ZIP for download: {e}")
                    except Exception as e:
                        st.error(f"Error exporting MagicaVoxel file: {str(e)}")
        
        else:  # OBJ File
            st.subheader("Export to OBJ / NetCDF")
            
            output_filename = st.text_input("Output Filename", value="voxcity")
            
            export_netcdf = st.checkbox("Also export NetCDF (voxels)", value=False)

            if st.button("Export OBJ File"):
                with st.spinner("Exporting OBJ file..."):
                    try:
                        data = st.session_state.voxcity_data
                        
                        output_directory = os.path.join(BASE_OUTPUT_DIR, "obj")
                        os.makedirs(output_directory, exist_ok=True)
                        
                        export_obj(
                            data['voxcity_grid'], 
                            output_directory, 
                            output_filename, 
                            data['meshsize']
                        )
                        
                        if export_netcdf:
                            try:
                                from voxcity.exporter.netcdf import save_voxel_netcdf
                                nc_dir = os.path.join(BASE_OUTPUT_DIR, "netcdf")
                                os.makedirs(nc_dir, exist_ok=True)
                                nc_path = os.path.join(nc_dir, f"{output_filename}.nc")
                                save_voxel_netcdf(
                                    data['voxcity_grid'],
                                    nc_path,
                                    data['meshsize'],
                                    rectangle_vertices=data['rectangle_vertices'],
                                )
                                st.info(f"NetCDF saved to {nc_path}")
                                try:
                                    with open(nc_path, 'rb') as f:
                                        st.download_button(
                                            label="Download NetCDF",
                                            data=f.read(),
                                            file_name=os.path.basename(nc_path),
                                            mime="application/netcdf"
                                        )
                                except Exception as e:
                                    st.warning(f"Could not prepare NetCDF for download: {e}")
                            except Exception as ne:
                                st.warning(f"NetCDF export failed: {ne}")

                        st.success("OBJ file exported successfully!")
                        st.info(f"File saved to {output_directory}/{output_filename}.obj")
                        try:
                            files_to_zip = [
                                os.path.join(output_directory, f"{output_filename}.obj"),
                                os.path.join(output_directory, f"{output_filename}.mtl")
                            ]
                            zip_buf = _zip_files_to_bytes(files_to_zip)
                            st.download_button(
                                label="Download OBJ + MTL (ZIP)",
                                data=zip_buf,
                                file_name=f"{output_filename}.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Could not prepare ZIP for download: {e}")
                    except Exception as e:
                        st.error(f"Error exporting OBJ/NetCDF: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>VoxCity Web App | Based on <a href='https://github.com/kunifujiwara/VoxCity'>VoxCity</a> by Kunihiko Fujiwara</p>
</div>
""", unsafe_allow_html=True)