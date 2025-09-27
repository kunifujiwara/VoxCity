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
    from voxcity.simulator.view import get_view_index, get_surface_view_factor, get_landmark_visibility_map, mark_building_by_id
    from voxcity.exporter.cityles import export_cityles
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
# Prefer docs/_static/favicon.ico; fall back to images/logo.png
APP_DIR = os.path.dirname(__file__)
_fav_candidate = os.path.normpath(os.path.join(APP_DIR, '..', 'docs', '_static', 'favicon.ico'))
_logo_candidate = os.path.normpath(os.path.join(APP_DIR, '..', 'images', 'logo.png'))
_logo_blue_candidate = os.path.normpath(os.path.join(APP_DIR, '..', 'images', 'logo_blue.png'))
PAGE_ICON = (
    _fav_candidate if os.path.exists(_fav_candidate)
    else (_logo_candidate if os.path.exists(_logo_candidate) else None)
)
st.set_page_config(
    page_title="VoxCity Web App",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Try to display project logo in the top header bar
try:
    if os.path.exists(_logo_candidate):
        st.logo(_logo_candidate)
    elif os.path.exists(_logo_blue_candidate):
        st.logo(_logo_blue_candidate)
except Exception:
    pass

# Subtle, clean styling inspired by professional budgeting dashboards
st.markdown(
    """
    <style>
      :root {
        /* Neutral gray palette */
        --vc-primary: #4B5563; /* gray-600 */
        --vc-bg: #F5F5F5;      /* gray-100 */
        --vc-surface: #FFFFFF; /* white */
        --vc-muted: #6B7280;   /* gray-500 */
        --vc-ring: #E5E7EB;    /* gray-200 */
      }
      header[data-testid="stHeader"] {
        background: var(--vc-bg);
        color: inherit;
        position: relative !important;
        z-index: 10 !important; /* keep header below fixed tabs */
      }
      /* Base top padding under header (overridden below when tabs are fixed) */
      .block-container { padding-top: 1.5rem; }
      /* Hide sidebar entirely */
      div[data-testid="stSidebar"],
      section[data-testid="stSidebar"] { display: none !important; }
      /* Ensure header branding/logo always visible */
      header [data-testid="stHeaderBranding"],
      header .stLogo,
      header [data-testid="stLogo"] { display: flex !important; opacity: 1 !important; }
      .stButton>button {
        border-radius: 8px;
        border: 1px solid var(--vc-ring);
        background: var(--vc-primary);
        color: #ffffff;
      }
      .stButton>button:hover { background: #374151; }
      .stAlert { border-radius: 8px; }
      /* Increase tab label font size and weight with strong specificity */
      /* Streamlit tabs can render labels in different wrappers; cover all */
      .stTabs [data-baseweb="tab"],
      .stTabs [data-baseweb="tab"] p,
      .stTabs [data-baseweb="tab"] span,
      .stTabs [role="tab"],
      div[role="tablist"] button,
      div[role="tablist"] [role="tab"],
      div[role="tablist"] button p,
      div[role="tablist"] button span,
      div[data-baseweb="tab-list"] button,
      div[data-baseweb="tab"] {
        gap: .5rem;
        font-size: 1.0rem !important; /* a bit smaller */
        font-weight: 400 !important;
        line-height: 1.2 !important;
        margin-right: 0.5rem !important; /* tighter space between tabs */
      }
      /* Move tabs into the top bar next to the logo */
      .stTabs [role="tablist"],
      div[role="tablist"] {
        position: fixed !important;
        top: 8px !important;              /* align vertically with header */
        left: 230px !important;           /* extra space to the right of the logo */
        z-index: 2147483647 !important;    /* ensure above header */
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important; /* tighter spacing between tabs */
        padding: 0 .5rem !important;
        height: 44px !important;
        pointer-events: auto !important;
      }
      /* Tighter top padding so content sits closer to the top bar */
      .block-container { padding-top: 1.0rem; }
      /* Enlarge header logo image */
      header [data-testid="stLogo"] img,
      header .stLogo img,
      header img[alt="Logo"],
      header img[alt="logo"] {
        height: 35px !important; /* slightly bigger logo */
        width: auto !important;
      }
      /* Reduce overall font size slightly */
      html, body, [data-testid="stAppViewContainer"], .block-container {
        font-size: 0.90rem !important;
      }
      /* Make headings more compact and smaller */
      h1, h2, h3, h4, h5, h6 { margin: 0.35rem 0 !important; }
      h1 { font-size: 1.35rem !important; }
      h2 { font-size: 1.15rem !important; }
      h3 { font-size: 1.00rem !important; }
      /* Compact common widget text */
      label,
      .stCheckbox,
      .stRadio,
      .stSelectbox,
      .stNumberInput,
      .stTextInput,
      .stDateInput,
      .stTimeInput,
      .stSlider,
      .stButton > button {
        font-size: 0.90rem !important;
      }
      /* Explicit small subheader utility for specific sections */
      .vc-subheader-small {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        margin: 0.25rem 0 0.75rem 0 !important;
      }
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

# Sidebar removed per design request

# Main content area - Remove authentication requirement
# Create tabs for different steps
tab1, tab2, tab_solar, tab_view, tab_landmark, tab5 = st.tabs(["Target Area", "Generation", "Solar", "View", "Landmark", "Export"])

# Tab 1: Set Target Area
with tab1:
    # (Section title removed)
    
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
    
    # Controls (left 25%) and Map/Preview (right 75%)
    col1, col2 = st.columns([1, 3])
    # Persistent map container aligned with the radio title row
    MAP_HEIGHT = 640
    map_container = col2.container()
    
    with col1:
        area_method = st.radio(
            "Select method to define target area:",
            ["Draw on map", "Enter coordinates"],
            index=0,
        )
        if area_method == "Enter coordinates":
            st.subheader("Enter Rectangle Vertices")
            
            col_sw_lon, col_sw_lat = st.columns(2)
            with col_sw_lon:
                sw_lon = st.number_input("Southwest Longitude", value=-74.02034270713835, format="%.8f")
            with col_sw_lat:
                sw_lat = st.number_input("Southwest Latitude", value=40.69992881162822, format="%.8f")

            col_nw_lon, col_nw_lat = st.columns(2)
            with col_nw_lon:
                nw_lon = st.number_input("Northwest Longitude", value=-74.02034270713835, format="%.8f")
            with col_nw_lat:
                nw_lat = st.number_input("Northwest Latitude", value=40.7111851828668, format="%.8f")

            col_ne_lon, col_ne_lat = st.columns(2)
            with col_ne_lon:
                ne_lon = st.number_input("Northeast Longitude", value=-74.00555129286164, format="%.8f")
            with col_ne_lat:
                ne_lat = st.number_input("Northeast Latitude", value=40.7111851828668, format="%.8f")

            col_se_lon, col_se_lat = st.columns(2)
            with col_se_lon:
                se_lon = st.number_input("Southeast Longitude", value=-74.00555129286164, format="%.8f")
            with col_se_lat:
                se_lat = st.number_input("Southeast Latitude", value=40.69992881162822, format="%.8f")
            
            if st.button("Set Rectangle"):
                st.session_state.rectangle_vertices = [
                    (sw_lon, sw_lat),
                    (nw_lon, nw_lat),
                    (ne_lon, ne_lat),
                    (se_lon, se_lat)
                ]
                st.success("Rectangle vertices set successfully!")
        
        if area_method == "Draw on map":
            city_name = st.text_input("Enter city name", value="Tokyo")
            zoom_level = 14
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
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, tiles="CartoDB positron")
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
                        with map_container:
                            out = st_folium(m, height=MAP_HEIGHT, width=1100, key="search_map", returned_objects=returned_objs)
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
                zoom = st.session_state.get("search_map_zoom", 14)
                if st.session_state.get("show_dims_success"):
                    _notify_success("Rectangle captured from map.")
                    st.session_state["show_dims_success"] = False
                m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")
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
                with map_container:
                    out = st_folium(m, height=MAP_HEIGHT, width=1100, key="search_map", returned_objects=returned_objs)
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
        # 'Draw on map' option removed; drawing is available within 'Search by city name'
        
        # 'Set center and dimensions' option removed; this mode is available inside 'Search by city name'
    
    with map_container:
        if st.session_state.rectangle_vertices and (area_method in ["Enter coordinates"]):
            m = folium.Map(location=[
                (st.session_state.rectangle_vertices[0][1] + st.session_state.rectangle_vertices[2][1]) / 2,
                (st.session_state.rectangle_vertices[0][0] + st.session_state.rectangle_vertices[2][0]) / 2
            ], zoom_start=15, tiles="CartoDB positron")
            folium.Rectangle(
                bounds=[[v[1], v[0]] for v in st.session_state.rectangle_vertices[:3:2]],
                color='#3388ff',
                fill=True,
                fillColor='#3388ff',
                fillOpacity=0.2
            ).add_to(m)
            st_folium(m, height=MAP_HEIGHT, width=1100)

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
        # OpenEarthMapJapan is available without Earth Engine
        land_cover_sources = ['OpenStreetMap', 'OpenEarthMapJapan']
        canopy_height_sources = ['Static']
        dem_sources = ['Flat']
        
        # (Informational note removed by request)
    
    # Narrower controls column (half the previous width)
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.markdown("<h3 class='vc-subheader-small'>Data Sources</h3>", unsafe_allow_html=True)
        # Auto-default data sources based on location (Japan vs Other)
        rect_vertices = st.session_state.get('rectangle_vertices')
        loc_signature = tuple(rect_vertices) if rect_vertices else None
        # Apply defaults only when the target area changes (so user overrides persist)
        if st.session_state.get('ds_loc_sig') != loc_signature:
            # Determine if center of rectangle is within Japan bounding box
            is_japan = False
            if rect_vertices and len(rect_vertices) >= 3:
                center_lon = (rect_vertices[0][0] + rect_vertices[2][0]) / 2.0
                center_lat = (rect_vertices[0][1] + rect_vertices[2][1]) / 2.0
                if (122.0 <= center_lon <= 154.0) and (24.0 <= center_lat <= 46.5):
                    is_japan = True
            # Desired defaults
            desired_building_source = 'OpenStreetMap'
            desired_building_complementary = 'None'
            desired_land_cover = 'OpenEarthMapJapan' if is_japan else 'OpenStreetMap'
            desired_canopy = 'Static'
            desired_dem = 'Flat'
            # Resolve to available options with sensible fallbacks
            def resolve(desired_value, options, fallback=None):
                if desired_value in options:
                    return desired_value
                if fallback and (fallback in options):
                    return fallback
                return options[0] if options else desired_value
            # Compute lists used by widgets
            comp_options = ['None'] + [s for s in building_sources if s != 'Local file']
            st.session_state['ds_building_source'] = resolve(desired_building_source, building_sources)
            st.session_state['ds_building_complementary_source'] = resolve(desired_building_complementary, comp_options, 'None')
            st.session_state['ds_land_cover_source'] = resolve(desired_land_cover, land_cover_sources, 'OpenStreetMap')
            st.session_state['ds_canopy_height_source'] = resolve(desired_canopy, canopy_height_sources)
            st.session_state['ds_dem_source'] = resolve(desired_dem, dem_sources)
            st.session_state['ds_loc_sig'] = loc_signature

        # Widgets bound to session state keys, preserving user changes across reruns
        building_source = st.selectbox("Building Source", building_sources, key='ds_building_source')
        building_complementary_source = st.selectbox(
            "Building Complementary Source",
            ['None'] + [s for s in building_sources if s != 'Local file'],
            key='ds_building_complementary_source'
        )
        land_cover_source = st.selectbox("Land Cover Source", land_cover_sources, key='ds_land_cover_source')
        canopy_height_source = st.selectbox("Canopy Height Source", canopy_height_sources, key='ds_canopy_height_source')
        dem_source = st.selectbox("DEM Source", dem_sources, key='ds_dem_source')

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
        if st.session_state.rectangle_vertices is None:
            st.warning("Set the target area first in the 'Target Area' tab.")
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
# Notes tab removed per request

# Tab 4a: Solar
with tab_solar:
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generation' tab.")
    else:
        hdr_l, hdr_r = st.columns([1, 3])
        with hdr_l:
            st.subheader("Solar Radiation Analysis")
        with hdr_r:
            run_solar = st.button("Run Simulation", key="run_solar_btn")
        ctrl_col, vis_col = st.columns([1, 3])
        solar_status = vis_col.empty()

        with ctrl_col:
            solar_calc_type = st.radio("Calculation Type", ["Instantaneous", "Cumulative"], horizontal=True)
            # Place Analysis Target immediately below Calculation Type
            analysis_target = st.radio("Analysis Target", ["Ground Level", "Building Surfaces"], horizontal=True, key="solar_analysis_target")
            if solar_calc_type == "Instantaneous":
                default_inst_date = st.session_state.get('solar_inst_date', datetime.now().date())
                default_inst_time = st.session_state.get('solar_inst_time', datetime.strptime("12:00:00", "%H:%M:%S").time())
                col_id, col_it = st.columns(2)
                with col_id:
                    calc_date = st.date_input("Date (MM-DD)", value=default_inst_date, key="solar_inst_date")
                with col_it:
                    calc_time = st.time_input("Time (HH:MM:SS)", value=default_inst_time, key="solar_inst_time")
                calc_datetime = f"{calc_date.strftime('%m-%d')} {calc_time.strftime('%H:%M:%S')}"
            else:
                default_start_date = datetime(datetime.now().year, 1, 1).date()
                default_end_date = datetime(datetime.now().year, 1, 31).date()
                col_sd, col_st = st.columns(2)
                with col_sd:
                    start_date = st.date_input("Start Date (MM-DD)", value=default_start_date)
                with col_st:
                    start_time = st.time_input("Start Time", value=datetime.strptime("01:00:00", "%H:%M:%S").time())
                col_ed, col_et = st.columns(2)
                with col_ed:
                    end_date = st.date_input("End Date (MM-DD)", value=default_end_date)
                with col_et:
                    end_time = st.time_input("End Time", value=datetime.strptime("23:00:00", "%H:%M:%S").time())
            download_epw = st.checkbox("Download nearest EPW file", value=True)
            if not download_epw:
                epw_file = st.file_uploader("Upload EPW file", type=['epw'])

        if run_solar:
            with solar_status.container():
                with st.spinner("Running solar radiation analysis..."):
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
                        
                        irradiance = get_building_global_solar_irradiance_using_epw(
                            data['voxcity_grid'],
                            data['meshsize'],
                            **irradiance_kwargs
                        )
                        
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
                        
                    

# Tab 4b: View
with tab_view:
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generation' tab.")
    else:
        hdr_l, hdr_r = st.columns([1, 3])
        with hdr_l:
            st.subheader("View Index Analysis")
        with hdr_r:
            run_view = st.button("Run Simulation", key="run_view_btn")
        ctrl_col, vis_col = st.columns([1, 3])
        view_status = vis_col.empty()

        with ctrl_col:
            view_type = st.selectbox("View Type", ["Green View Index", "Sky View Index", "Custom (Select Classes)"])
            analysis_target_view = st.radio("Analysis Target", ["Ground Level", "Building Surfaces"], horizontal=True, key="view_analysis_target")
            view_point_height = st.number_input("View Point Height (m)", value=1.5, min_value=0.0, max_value=10.0)
            export_obj = st.checkbox("Export as OBJ file", value=False)
            # Defaults for custom selection to ensure variables exist even if not used
            selected_custom_values = []
            inclusion_mode_custom = True

            # Custom class selection UI
            if view_type == "Custom (Select Classes)":
                col_mode, col_sel = st.columns([1, 2])
                with col_mode:
                    inc_exc = st.radio(
                        "Mode",
                        ["Inclusion (count selected classes)", "Exclusion (allow only selected classes)"],
                        horizontal=False,
                        key="view_custom_mode"
                    )
                inclusion_mode_custom = inc_exc.startswith("Inclusion")

                # Fixed list of selectable classes
                base_class_options = [
                    (-3, "Building"),
                    (-2, "Tree"),
                ]
                land_cover_options = [
                    (1, "Bareland"),
                    (2, "Rangeland"),
                    (3, "Shrub"),
                    (4, "Agriculture land"),
                    (6, "Moss and lichen"),
                    (7, "Wet land"),
                    (8, "Mangrove"),
                    (9, "Water"),
                    (10, "Snow and ice"),
                    (11, "Developed space"),
                    (12, "Road"),
                ]

                selected_custom_values = []
                with col_sel:
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("Select all", key="vc_cls_btn_all"):
                            for code, _ in base_class_options + land_cover_options:
                                st.session_state[f"vc_cls_{code}"] = True
                    with bcol2:
                        if st.button("Clear all", key="vc_cls_btn_none"):
                            for code, _ in base_class_options + land_cover_options:
                                st.session_state[f"vc_cls_{code}"] = False

                    # Above ground section
                    st.markdown("Above ground")
                    b1, b2 = st.columns(2)
                    for (code, name), col in zip(base_class_options, [b1, b2]):
                        with col:
                            checked = st.checkbox(name, key=f"vc_cls_{code}")
                            if checked:
                                selected_custom_values.append(int(code))

                    # Land cover section
                    st.markdown("Land cover")
                    cols = st.columns(3)
                    for idx, (code, name) in enumerate(land_cover_options):
                        with cols[idx % 3]:
                            checked = st.checkbox(name, key=f"vc_cls_{code}")
                            if checked:
                                selected_custom_values.append(int(code))

        # run_view is defined in header; no duplicate button here
        
        if run_view:
            with view_status.container():
                with st.spinner(f"Calculating {view_type}..."):
                    data = st.session_state.voxcity_data
                    
                    output_dir = os.path.join(BASE_OUTPUT_DIR, "view")
                    os.makedirs(output_dir, exist_ok=True)

                    if analysis_target_view == "Ground Level":
                        view_kwargs = {
                            "view_point_height": view_point_height,
                            "dem_grid": data['dem_grid'],
                            "obj_export": export_obj,
                            "output_directory": output_dir,
                            "output_file_name": (
                                "gvi" if view_type == "Green View Index" else ("svi" if view_type == "Sky View Index" else "custom_vi")
                            )
                        }
                        if view_type == "Custom (Select Classes)":
                            if len(selected_custom_values) == 0:
                                st.error("Please select at least one class for the custom view.")
                                view_grid = None
                            else:
                                view_grid = get_view_index(
                                    data['voxcity_grid'],
                                    data['meshsize'],
                                    mode=None,
                                    hit_values=tuple(selected_custom_values),
                                    inclusion_mode=inclusion_mode_custom,
                                    **view_kwargs
                                )
                        else:
                            mode = 'green' if view_type == "Green View Index" else 'sky'
                            view_grid = get_view_index(
                                data['voxcity_grid'], 
                                data['meshsize'], 
                                mode=mode, 
                                **view_kwargs
                            )
                        with vis_col:
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
                                        ground_colormap='viridis',
                                        ground_vmin=0.0,
                                        ground_vmax=1.0,
                                        sim_surface_opacity=0.95,
                                        show=False,
                                        return_fig=True,
                                        title=(view_type if view_type != "Custom (Select Classes)" else "Custom View Index")
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
                    else:
                        # Building surfaces: compute surface view factor on building meshes
                        try:
                            if view_type == "Custom (Select Classes)":
                                if len(selected_custom_values) == 0:
                                    st.error("Please select at least one class for the custom view.")
                                    mesh = None
                                else:
                                    target_values = tuple(selected_custom_values)
                                    inclusion_mode = inclusion_mode_custom
                            elif view_type == "Green View Index":
                                target_values = (-2,)  # trees
                                inclusion_mode = True
                            else:
                                target_values = (0,)   # sky
                                inclusion_mode = False
                            mesh = get_surface_view_factor(
                                data['voxcity_grid'],
                                data['meshsize'],
                                target_values=target_values,
                                inclusion_mode=inclusion_mode,
                                building_id_grid=data.get('building_id_grid'),
                                colormap='viridis',
                                vmin=0.0,
                                vmax=1.0,
                                obj_export=export_obj,
                                output_directory=output_dir,
                                output_file_name=(
                                    'surface_' + (
                                        'gvi' if view_type == 'Green View Index' else (
                                            'svi' if view_type == 'Sky View Index' else 'custom_vi'
                                        )
                                    )
                                )
                            )
                        except Exception as e:
                            mesh = None
                            st.warning(f"Surface view computation failed: {e}")
                        if mesh is None:
                            st.info("No building surfaces generated; showing ground-level overlay instead.")
                            # Fallback to ground view
                            mode = 'green' if view_type == "Green View Index" else 'sky'
                            view_grid = get_view_index(
                                data['voxcity_grid'], 
                                data['meshsize'], 
                                mode=mode, 
                                view_point_height=view_point_height,
                                dem_grid=data['dem_grid'],
                            )
                            with vis_col:
                                fig = visualize_voxcity_plotly(
                                    data['voxcity_grid'],
                                    data['meshsize'],
                                    downsample=1,
                                    voxel_color_map='grayscale',
                                    ground_sim_grid=view_grid,
                                    ground_dem_grid=data['dem_grid'],
                                    ground_view_point_height=view_point_height,
                                    ground_colormap='viridis',
                                    ground_vmin=0.0,
                                    ground_vmax=1.0,
                                    sim_surface_opacity=0.95,
                                    show=False,
                                    return_fig=True,
                                    title=view_type
                                )
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            with vis_col:
                                try:
                                    fig_b = visualize_voxcity_plotly(
                                        data['voxcity_grid'],
                                        data['meshsize'],
                                        downsample=1,
                                        voxel_color_map='grayscale',
                                        building_sim_mesh=mesh,
                                        building_value_name='view_factor_values',
                                        building_colormap='viridis',
                                        building_vmin=0.0,
                                        building_vmax=1.0,
                                        render_voxel_buildings=False,
                                        show=False,
                                        return_fig=True,
                                        title=(
                                            "Surface " + (
                                                "GVI" if view_type=="Green View Index" else (
                                                    "SVI" if view_type=="Sky View Index" else "Custom View"
                                                )
                                            )
                                        )
                                    )
                                    if fig_b is not None:
                                        st.plotly_chart(fig_b, use_container_width=True)
                                except Exception as ve:
                                    st.warning(f"Surface view rendering failed: {ve}")
                    
                    

# Tab 4c: Landmark
with tab_landmark:
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generation' tab.")
    else:
        hdr_l, hdr_r = st.columns([1, 3])
        with hdr_l:
            st.subheader("Landmark Visibility Analysis")
        with hdr_r:
            run_landmark = st.button("Run Simulation", key="run_landmark_btn")
        ctrl_col, vis_col = st.columns([1, 3])
        landmark_status = vis_col.empty()
        with ctrl_col:
            analysis_target_lm = st.radio("Analysis Target", ["Ground Level", "Building Surfaces"], horizontal=True, key="landmark_analysis_target")
            if 'landmark_ids_text' not in st.session_state:
                st.session_state['landmark_ids_text'] = ""
            select_on_map = st.checkbox("Select landmarks on map", value=False)
            # Optional map-based landmark selection (rendered directly below the checkbox)
            if select_on_map and (st.session_state.voxcity_data is not None):
                data = st.session_state.voxcity_data
                building_gdf = data.get('building_gdf')
                # Center map on rectangle or building bounds
                if data.get('rectangle_vertices') is not None:
                    center_lat = (data['rectangle_vertices'][0][1] + data['rectangle_vertices'][2][1]) / 2
                    center_lon = (data['rectangle_vertices'][0][0] + data['rectangle_vertices'][2][0]) / 2
                    zoom_here = 15
                else:
                    try:
                        minx, miny, maxx, maxy = building_gdf.total_bounds
                        center_lon, center_lat = (minx + maxx) / 2, (miny + maxy) / 2
                        zoom_here = 15
                    except Exception:
                        center_lat, center_lon, zoom_here = 40.0, -100.0, 4
                m_land = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_here, tiles="CartoDB positron")
                # Draw building outlines (limit to avoid heavy rendering)
                try:
                    max_draw = 1500
                    cnt = 0
                    for _, row in building_gdf.iterrows():
                        if cnt >= max_draw:
                            break
                        geom = row.geometry
                        if geom is not None and hasattr(geom, "exterior"):
                            coords = list(geom.exterior.coords)
                            latlon = [(c[1], c[0]) for c in coords[:-1]]
                            folium.Polygon(locations=latlon, color="#1f77b4", weight=1, fill=True, fill_opacity=0.15).add_to(m_land)
                            cnt += 1
                except Exception:
                    pass
                # Enable polygon drawing
                Draw(
                    export=False,
                    draw_options={
                        'polyline': False,
                        'polygon': True,
                        'circle': False,
                        'circlemarker': False,
                        'marker': False,
                        'rectangle': False
                    },
                    edit_options={'edit': True}
                ).add_to(m_land)
                out_lm = st_folium(m_land, height=380, width=450, key="landmark_map", returned_objects=["last_active_drawing", "all_drawings", "last_drawn"])    
                # Capture last polygon and compute intersecting buildings
                feature = None
                if isinstance(out_lm, dict):
                    feature = out_lm.get('last_active_drawing') or out_lm.get('last_drawn')
                    if (feature is None) and out_lm.get('all_drawings'):
                        drawings = out_lm.get('all_drawings') or []
                        feature = drawings[-1] if len(drawings) > 0 else None
                if feature:
                    geometry = feature.get('geometry', feature)
                    if geometry and geometry.get('type') == 'Polygon':
                        coords = geometry['coordinates'][0]
                        try:
                            poly_sel = Polygon([(c[0], c[1]) for c in coords])
                            mask = building_gdf.geometry.intersects(poly_sel)
                            sel_ids = []
                            try:
                                sel_ids = list(building_gdf.loc[mask, 'building_id'].astype(int).values)
                            except Exception:
                                # Fallback to 'id' column
                                try:
                                    sel_ids = list(building_gdf.loc[mask, 'id'].astype(int).values)
                                except Exception:
                                    sel_ids = []
                            if len(sel_ids) > 0:
                                st.session_state['landmark_ids_text'] = ",".join(str(i) for i in sel_ids)
                                st.success(f"Selected {len(sel_ids)} buildings from map. IDs populated.")
                        except Exception as _:
                            pass
            # Use IDs populated from map selection if available (no manual input field)
            ids_text = st.session_state.get('landmark_ids_text', '')
        # run_landmark is defined in header; no duplicate here
        if run_landmark:
            with landmark_status.container():
                with st.spinner("Computing landmark visibility..."):
                    data = st.session_state.voxcity_data
                    output_dir_lm = os.path.join(BASE_OUTPUT_DIR, 'landmark')
                    os.makedirs(output_dir_lm, exist_ok=True)
                    # Determine landmark IDs
                    landmark_ids = []
                    if ids_text.strip():
                        try:
                            landmark_ids = [int(x.strip()) for x in ids_text.split(',') if x.strip()]
                        except Exception:
                            st.warning("Could not parse IDs; falling back to map/center selection if available.")
                    # If map selection stored, merge
                    map_ids_text = st.session_state.get('landmark_ids_text', '')
                    if map_ids_text.strip():
                        try:
                            landmark_ids += [int(x.strip()) for x in map_ids_text.split(',') if x.strip()]
                        except Exception:
                            pass
                    landmark_ids = sorted(list(set(landmark_ids)))

                    # Mark landmark voxels in a copy of the grid
                    voxcity_marked = data['voxcity_grid'].copy()
                    if len(landmark_ids) == 0:
                        # Fall back to center building via existing helper
                        vis_map, vox_marked_tmp = get_landmark_visibility_map(
                            data['voxcity_grid'],
                            data['building_id_grid'],
                            data['building_gdf'],
                            data['meshsize'],
                            rectangle_vertices=data['rectangle_vertices'],
                            output_directory=output_dir_lm,
                        )
                        if vox_marked_tmp is not None:
                            voxcity_marked = vox_marked_tmp
                    else:
                        try:
                            voxcity_marked = mark_building_by_id(
                                voxcity_marked,
                                data['building_id_grid'],
                                landmark_ids,
                                -30,
                            )
                        except Exception as e:
                            st.warning(f"Failed to mark landmark buildings: {e}")

                    if analysis_target_lm == "Ground Level":
                        view_kwargs = {
                            "view_point_height": 1.5,
                            "dem_grid": data['dem_grid'],
                            "obj_export": False,
                            "output_directory": output_dir_lm,
                            "output_file_name": "landmark"
                        }
                        landmark_grid = get_view_index(
                            voxcity_marked,
                            data['meshsize'],
                            hit_values=(-30,),
                            inclusion_mode=True,
                            **view_kwargs
                        )
                        # Set zeros to NaN for visualization emphasis
                        try:
                            lg = np.asarray(landmark_grid, dtype=float)
                            lg[lg == 0.0] = np.nan
                            landmark_grid = lg
                        except Exception:
                            pass
                        with vis_col:
                            fig = visualize_voxcity_plotly(
                                voxcity_marked,
                                data['meshsize'],
                                downsample=1,
                                voxel_color_map='grayscale',
                                ground_sim_grid=landmark_grid,
                                ground_dem_grid=data['dem_grid'],
                                ground_view_point_height=1.5,
                                ground_colormap='viridis',
                                ground_vmin=0.0,
                                ground_vmax=1.0,
                                sim_surface_opacity=0.95,
                                show=False,
                                return_fig=True,
                                title='Landmark Visibility (Ground)'
                            )
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        try:
                            landmark_mesh = get_surface_view_factor(
                                voxcity_marked,
                                data['meshsize'],
                                target_values=(-30,),
                                inclusion_mode=True,
                                progress_report=True,
                                building_id_grid=data.get('building_id_grid'),
                                colormap='viridis',
                                vmin=0.0,
                                vmax=1.0,
                                obj_export=False,
                                output_directory=output_dir_lm,
                                output_file_name='landmark_surface'
                            )
                        except Exception as e:
                            landmark_mesh = None
                            st.warning(f"Surface landmark computation failed: {e}")
                        if landmark_mesh is not None:
                            try:
                                vf = np.asarray(landmark_mesh.metadata.get('view_factor_values'), dtype=float).copy()
                                vf[vf == 0.0] = np.nan
                                landmark_mesh.metadata['view_factor_values'] = vf
                            except Exception:
                                pass
                            with vis_col:
                                try:
                                    fig_b = visualize_voxcity_plotly(
                                        voxcity_marked,
                                        data['meshsize'],
                                        downsample=1,
                                        voxel_color_map='grayscale',
                                        building_sim_mesh=landmark_mesh,
                                        building_value_name='view_factor_values',
                                        building_colormap='viridis',
                                        building_vmin=0.0,
                                        building_vmax=1.0,
                                        render_voxel_buildings=False,
                                        show=False,
                                        return_fig=True,
                                        title='Landmark Visibility (Surface)'
                                    )
                                    if fig_b is not None:
                                        st.plotly_chart(fig_b, use_container_width=True)
                                except Exception as ve:
                                    st.warning(f"3D surface rendering failed: {ve}")
                        else:
                            st.info("Falling back to ground-level landmark visibility.")
                            # Fallback ground visualization
                            landmark_grid = get_view_index(
                                voxcity_marked,
                                data['meshsize'],
                                hit_values=(-30,),
                                inclusion_mode=True,
                            )
                            with vis_col:
                                fig = visualize_voxcity_plotly(
                                    voxcity_marked,
                                    data['meshsize'],
                                    downsample=1,
                                    voxel_color_map='grayscale',
                                    ground_sim_grid=landmark_grid,
                                    ground_dem_grid=data['dem_grid'],
                                    ground_view_point_height=1.5,
                                    ground_colormap='viridis',
                                    ground_vmin=0.0,
                                    ground_vmax=1.0,
                                    sim_surface_opacity=0.95,
                                    show=False,
                                    return_fig=True,
                                    title='Landmark Visibility (Ground)'
                                )
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                        # (Removed duplicate rendering block that referenced undefined vis_map/vox_marked)
                    

# Tab 5: Export
with tab5:
    # (Section title removed)
    
    if st.session_state.voxcity_data is None:
        st.warning("Please generate a VoxCity model first in the 'Generation' tab.")
    else:
        export_format = st.selectbox(
            "Select Export Format",
            ["CityLES", "OBJ File"]
        )
        
        if export_format == "CityLES":
            st.subheader("Export to CityLES")
            cityles_dir = os.path.join(BASE_OUTPUT_DIR, 'cityles')
            os.makedirs(cityles_dir, exist_ok=True)
            building_material = st.selectbox("Building Material", ["default", "concrete", "brick"], index=0)
            tree_type = st.selectbox("Tree Type", ["default", "deciduous", "conifer"], index=0)
            tree_base_ratio = st.number_input("Tree Base Ratio", value=0.3, min_value=0.0, max_value=1.0)
            if st.button("Export CityLES Files"):
                with st.spinner("Exporting CityLES files..."):
                    try:
                        data = st.session_state.voxcity_data
                        export_cityles(
                            data['building_height_grid'],
                            data['building_id_grid'],
                            data['canopy_height_grid'],
                            data['land_cover_grid'],
                            data['dem_grid'],
                            data['meshsize'],
                            land_cover_source,
                            data['rectangle_vertices'],
                            output_directory=cityles_dir,
                            building_material=building_material,
                            tree_type=tree_type,
                            tree_base_ratio=float(tree_base_ratio),
                            canopy_bottom_height_grid=data.get('canopy_bottom_height_grid')
                        )
                        st.success("CityLES files exported successfully!")
                        st.info(f"Files saved to {cityles_dir}")
                        try:
                            zip_buf = _zip_directory_to_bytes(cityles_dir)
                            st.download_button(
                                label="Download CityLES outputs (ZIP)",
                                data=zip_buf,
                                file_name="cityles_outputs.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.warning(f"Could not prepare ZIP for download: {e}")
                    except Exception as e:
                        st.error(f"Error exporting CityLES files: {str(e)}")
        
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