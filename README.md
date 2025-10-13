[![PyPi version](https://img.shields.io/pypi/v/voxcity.svg)](https://pypi.python.org/pypi/voxcity)
[![Python versions](https://img.shields.io/pypi/pyversions/voxcity.svg)](https://pypi.org/project/voxcity/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lofd3RawKMr6QuUsamGaF48u2MN0hfrP?usp=sharing)
[![License](https://img.shields.io/pypi/l/voxcity.svg)](https://pypi.org/project/voxcity/)
[![Downloads](https://pepy.tech/badge/voxcity)](https://pepy.tech/project/voxcity)
[![Documentation Status](https://readthedocs.org/projects/voxcity/badge/?version=latest)](https://voxcity.readthedocs.io/en/latest/?badge=latest)
<!-- [![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/) -->

<p align="center">
  Tutorial preview: <a href="https://colab.research.google.com/drive/1Lofd3RawKMr6QuUsamGaF48u2MN0hfrP?usp=sharing">[Google Colab]</a> | Documentation: <a href="https://voxcity.readthedocs.io/en/latest">[Read the Docs]</a> | Video tutorial: <a href="https://youtu.be/qHusvKB07qk">[Watch on YouTube]</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/logo.png" alt="Voxcity logo" width="550">
</p>

 

# VoxCity

**voxcity** is a Python package that provides a seamless solution for grid-based 3D city model generation and urban simulation for cities worldwide. VoxCity's generator module automatically downloads building heights, tree canopy heights, land cover, and terrain elevation within a specified target area, and voxelizes buildings, trees, land cover, and terrain to generate an integrated voxel city model. The simulator module enables users to conduct environmental simulations, including solar radiation and view index analyses. Users can export the generated models using several file formats compatible with external software, such as ENVI-met (INX), Blender, and Rhino (OBJ). Try it out using the [Google Colab Demo](https://colab.research.google.com/drive/1Lofd3RawKMr6QuUsamGaF48u2MN0hfrP?usp=sharing) or your local environment. For detailed documentation, API reference, and tutorials, visit our [Read the Docs](https://voxcity.readthedocs.io/en/latest) page.

<!-- <p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/kunifujiwara/VoxCity/blob/main/images/concept.png">
    <img src="https://github.com/kunifujiwara/VoxCity/blob/main/images/concept.png" alt="Conceptual Diagram of voxcity" width="800">
  </picture>
</p> -->
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/concept.png" alt="Conceptual Diagram of voxcity" width="800">
</p>

## Tutorial

- **Google Colab (interactive notebook)**: <a href="https://colab.research.google.com/drive/1Lofd3RawKMr6QuUsamGaF48u2MN0hfrP?usp=sharing">Open tutorial in Colab</a>
- **YouTube video (walkthrough)**: <a href="https://youtu.be/qHusvKB07qk">Watch on YouTube</a>

<p align="center">
  <a href="https://youtu.be/qHusvKB07qk" title="Click to watch the VoxCity tutorial on YouTube">
    <img src="images/youtube_thumbnail_play.png" alt="VoxCity Tutorial — Click to watch on YouTube" width="480">
  </a>
</p>

<p align="center">
  <em>Tutorial video by <a href="https://ual.sg/author/liang-xiucheng/">Xiucheng Liang</a></em>
</p>


## Interactive 3D Demo (Plotly)

- **Open interactive demo**: <a href="https://voxcity.readthedocs.io/en/latest/_static/plotly/voxcity_demo.html">Launch the Plotly 3D viewer</a>

<p align="center">
  <a href="https://voxcity.readthedocs.io/en/latest/_static/plotly/voxcity_demo.html" title="Open the interactive 3D VoxCity demo">
    <img src="images/visualization.png" alt="Interactive VoxCity Plotly demo" width="600">
  </a>
</p>


## Key Features

- **Integration of Multiple Data Sources:**  
  Combines building footprints, land cover data, canopy height maps, and DEMs to generate a consistent 3D voxel representation of an urban scene.
  
- **Flexible Input Sources:**  
  Supports various building and terrain data sources including:
  - Building Footprints: OpenStreetMap, Overture, EUBUCCO, Microsoft Building Footprints, Open Building 2.5D
  - Land Cover: UrbanWatch, OpenEarthMap Japan, ESA WorldCover, ESRI Land Cover, Dynamic World, OpenStreetMap
  - Canopy Height: High Resolution 1m Global Canopy Height Maps, ETH Global Sentinel-2 10m
  - DEM: DeltaDTM, FABDEM, NASA, COPERNICUS, and more

  *Detailed information about each data source can be found in the [References of Data Sources](#references-of-data-sources) section.*
  
- **Customizable Domain and Resolution:**  
  Easily define a target area by drawing a rectangle on a map or specifying center coordinates and dimensions. Adjust the mesh size to meet resolution needs.
  
- **Integration with Earth Engine:**  
  Leverages Google Earth Engine for large-scale geospatial data processing (authentication and project setup required).
  
- **Output Formats:**
  - **ENVI-MET**: Export INX and EDB files suitable for ENVI-MET microclimate simulations.
  - **MagicaVoxel**: Export vox files for 3D editing and visualization in MagicaVoxel.
  - **OBJ**: Export wavefront OBJ for rendering and integration into other workflows.

- **Analytical Tools:**
  - **View Index Simulations**: Compute sky view index (SVI) and green view index (GVI) from a specified viewpoint.
  - **Landmark Visibility Maps**: Assess the visibility of selected landmarks within the voxelized environment.

## Installation

Make sure you have Python 3.12 installed. Install voxcity with:

### For Local Environment

```bash
conda create --name voxcity python=3.12
conda activate voxcity
conda install -c conda-forge gdal timezonefinder
pip install voxcity
```

### For Google Colab

```python
!pip install voxcity
```

## Setup for Earth Engine

To use Earth Engine data, set up your Earth Engine enabled Cloud Project by following the instructions here:
https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup

After setting up, authenticate and initialize Earth Engine:

### For Local Environment

```bash
earthengine authenticate
```

### For Google Colab

```python
# Click displayed link, generate token, copy and paste the token
!earthengine authenticate --auth_mode=notebook
```

## Usage Overview

### 1. Authenticate Earth Engine

```python
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')
```

### 2. Define Target Area

You can define your target area in three ways:

#### Option 1: Direct Coordinate Input
Define the target area by directly specifying the coordinates of the rectangle vertices.

```python
rectangle_vertices = [
    (-122.33587348582083, 47.59830044521263),  # Southwest corner (longitude, latitude)
    (-122.33587348582083, 47.60279755390168),  # Northwest corner (longitude, latitude) 
    (-122.32922451417917, 47.60279755390168),  # Northeast corner (longitude, latitude)
    (-122.32922451417917, 47.59830044521263)   # Southeast corner (longitude, latitude)
]
```

#### Option 2: Draw a Rectangle (for Jupyter Notebook)
Use the GUI map interface to draw a rectangular domain of interest.

```python
from voxcity.geoprocessor.draw import draw_rectangle_map_cityname

cityname = "tokyo"
m, rectangle_vertices = draw_rectangle_map_cityname(cityname, zoom=15)
m
```

#### Option 3: Specify Center and Dimensions (for Jupyter Notebook)
Choose the width and height in meters and select the center point on the map.

```python
from voxcity.geoprocessor.draw import center_location_map_cityname

width = 500
height = 500
m, rectangle_vertices = center_location_map_cityname(cityname, width, height, zoom=15)
m
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/draw_rect.png" alt="Draw Rectangle on Map GUI" width="400">
</p>

### 3. Set Parameters

Define data sources and mesh size (m):

```python
building_source = 'OpenStreetMap'                                     # Building footprint and height data source
land_cover_source = 'OpenStreetMap'                                   # Land cover classification data source
canopy_height_source = 'High Resolution 1m Global Canopy Height Maps' # Tree canopy height data source
dem_source = 'DeltaDTM'                                               # Digital elevation model data source
meshsize = 5                                                          # Grid cell size in meters

kwargs = {
    "output_dir": "output",   # Directory to save output files
    "dem_interpolation": True # Enable DEM interpolation
}
```

### 4. Get voxcity Output

Generate voxel data grids and a corresponding building GeoDataFrame:

```python
from voxcity.generator import get_voxcity

voxcity_grid, building_height_grid, building_min_height_grid, \
    building_id_grid, canopy_height_grid, canopy_bottom_height_grid, land_cover_grid, dem_grid, \
    building_gdf = get_voxcity(
    rectangle_vertices,
    building_source,
    land_cover_source,
    canopy_height_source,
    dem_source,
    meshsize,
    **kwargs
)
```

### 5. Exporting Files

#### ENVI-MET INX/EDB Files:
[ENVI-MET](https://www.envi-met.com/) is an advanced microclimate simulation software specialized in modeling urban environments. It simulates the interactions between buildings, vegetation, and various climate parameters like temperature, wind flow, humidity, and radiation. The software is used widely in urban planning, architecture, and environmental studies (Commercial, offers educational licenses).

```python
from voxcity.exporter.envimet import export_inx, generate_edb_file

envimet_kwargs = {
    "output_directory": "output",                     # Directory where output files will be saved
    "author_name": "your name",                       # Name of the model author
    "model_description": "generated with voxcity",  # Description text for the model
    "domain_building_max_height_ratio": 2,            # Maximum ratio between domain height and tallest building height
    "useTelescoping_grid": True,                      # Enable telescoping grid for better computational efficiency
    "verticalStretch": 20,                            # Vertical grid stretching factor (%)
    "min_grids_Z": 20,                                # Minimum number of vertical grid cells
    "lad": 1.0                                        # Leaf Area Density (m2/m3) for vegetation modeling 
}

export_inx(building_height_grid, building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, meshsize, land_cover_source, rectangle_vertices, **envimet_kwargs)
generate_edb_file(**envimet_kwargs)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/envimet.png" alt="Generated 3D City Model on Envi-MET GUI" width="600">
</p>
<p align="center">
  <em>Example Output Exported in INX and Inported in ENVI-met</em>
</p>

#### OBJ Files:

```python
from voxcity.exporter.obj import export_obj

output_directory = "output"  # Directory where output files will be saved
output_file_name = "voxcity" # Base name for the output OBJ file
# export_obj signature: export_obj(array, output_dir, file_name, voxel_size, voxel_color_map=None)
export_obj(voxcity_grid, output_directory, output_file_name, meshsize)
```
The generated OBJ files can be opened and rendered in the following 3D visualization software:

- [Twinmotion](https://www.twinmotion.com/): Real-time visualization tool (Free for personal use)
- [Blender](https://www.blender.org/): Professional-grade 3D creation suite (Free)
- [Rhino](https://www.rhino3d.com/): Professional 3D modeling software (Commercial, offers educational licenses)

<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/obj.png" alt="OBJ 3D City Model Rendered in Rhino" width="600">
</p>
<p align="center">
  <em>Example Output Exported in OBJ and Rendered in Rhino</em>
</p>

#### MagicaVoxel VOX Files:

[MagicaVoxel](https://ephtracy.github.io/) is a lightweight and user-friendly voxel art editor. It allows users to create, edit, and render voxel-based 3D models with an intuitive interface, making it perfect for modifying and visualizing voxelized city models. The software is free and available for Windows and Mac.

```python
from voxcity.exporter.magicavoxel import export_magicavoxel_vox

output_path = "output"
base_filename = "voxcity"
export_magicavoxel_vox(voxcity_grid, output_path, base_filename=base_filename)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/vox.png" alt="Generated 3D City Model on MagicaVoxel GUI" width="600">
</p>
<p align="center">
  <em>Example Output Exported in VOX and Rendered in MagicaVoxel</em>
</p>

### 6. Additional Use Cases

#### Compute Solar Irradiance:

```python
from voxcity.simulator.solar import get_global_solar_irradiance_using_epw

solar_kwargs = {
    "download_nearest_epw": True,  # Whether to automatically download nearest EPW weather file based on location from Climate.OneBuilding.Org
    "rectangle_vertices": rectangle_vertices,  # Coordinates defining the area of interest for calculation
    # "epw_file_path": "./output/new.york-downtown.manhattan.heli_ny_usa_1.epw",  # Path to EnergyPlus Weather (EPW) file containing climate data. Set if you already have an EPW file.
    "calc_time": "01-01 12:00:00",  # Time for instantaneous calculation in format "MM-DD HH:MM:SS"
    "view_point_height": 1.5,  # Height of view point in meters for calculating solar access. Default: 1.5 m
    "tree_k": 0.6,    # Static extinction coefficient - controls how much sunlight is blocked by trees (higher = more blocking)
    "tree_lad": 1.0,    # Leaf area density of trees - density of leaves/branches that affect shading (higher = denser foliage)
    "dem_grid": dem_grid,      # Digital elevation model grid for terrain heights
    "colormap": 'magma',       # Matplotlib colormap for visualization. Default: 'viridis'
    "obj_export": True,        # Whether to export results as 3D OBJ file
    "output_directory": 'output/test',  # Directory for saving output files
    "output_file_name": 'instantaneous_solar_irradiance',  # Base filename for outputs (without extension)
    "alpha": 1.0,             # Transparency of visualization (0.0-1.0)
    "vmin": 0,               # Minimum value for colormap scaling in visualization
    # "vmax": 900,             # Maximum value for colormap scaling in visualization
}

# Compute global solar irradiance map (direct + diffuse radiation)
solar_grid = get_global_solar_irradiance_using_epw(    
    voxcity_grid,                        # 3D voxel grid representing the urban environment
    meshsize,                            # Size of each voxel in meters
    calc_type='instantaneous',           # Calculate instantaneous irradiance at specified time
    direct_normal_irradiance_scaling=1.0, # Scaling factor for direct solar radiation (1.0 = no scaling)
    diffuse_irradiance_scaling=1.0,      # Scaling factor for diffuse solar radiation (1.0 = no scaling)
    **solar_kwargs                       # Pass all the parameters defined above
)

# Adjust parameters for cumulative calculation
solar_kwargs["start_time"] = "01-01 01:00:00" # Start time for cumulative calculation
solar_kwargs["end_time"] = "01-31 23:00:00" # End time for cumulative calculation
solar_kwargs["output_file_name"] = 'cumulative_solar_irradiance'  # Base filename for outputs (without extension)

# Calculate cumulative solar irradiance over the specified time period
cum_solar_grid = get_global_solar_irradiance_using_epw(    
    voxcity_grid,                        # 3D voxel grid representing the urban environment
    meshsize,                            # Size of each voxel in meters
    calc_type='cumulative',              # Calculate cumulative irradiance over time period instead of instantaneous
    direct_normal_irradiance_scaling=1.0, # Scaling factor for direct solar radiation (1.0 = no scaling)
    diffuse_irradiance_scaling=1.0,      # Scaling factor for diffuse solar radiation (1.0 = no scaling)
    **solar_kwargs                       # Pass all the parameters defined above
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/solar.png" alt="Solar Irradiance Maps Rendered in Rhino" width="800">
</p>
<p align="center">
  <em>Example Results Saved as OBJ and Rendered in Rhino</em>
</p>

#### Compute Green View Index (GVI) and Sky View Index (SVI):

```python
from voxcity.simulator.view import get_view_index

view_kwargs = {
    "view_point_height": 1.5,      # Height of observer viewpoint in meters
    "dem_grid": dem_grid,          # Digital elevation model grid
    "colormap": "viridis",         # Colormap for visualization
    "obj_export": True,            # Whether to export as OBJ file
    "output_directory": "output",  # Directory to save output files
    "output_file_name": "gvi"      # Base filename for outputs
}

# Compute Green View Index using mode='green'
gvi_grid = get_view_index(voxcity_grid, meshsize, mode='green', **view_kwargs)

# Adjust parameters for Sky View Index
view_kwargs["colormap"] = "BuPu_r"
view_kwargs["output_file_name"] = "svi"
view_kwargs["elevation_min_degrees"] = 0 # Start ray-tracing from the horizon

# Compute Sky View Index using mode='sky'
svi_grid = get_view_index(voxcity_grid, meshsize, mode='sky', **view_kwargs)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/view_index.png" alt="View Index Maps Rendered in Rhino" width="800">
</p>
<p align="center">
  <em>Example Results Saved as OBJ and Rendered in Rhino</em>
</p>

#### Landmark Visibility Map:

```python
from voxcity.simulator.view import get_landmark_visibility_map

# Dictionary of parameters for landmark visibility analysis
landmark_kwargs = {
    "view_point_height": 1.5,                 # Height of observer viewpoint in meters
    "rectangle_vertices": rectangle_vertices, # Vertices defining simulation domain boundary
    "dem_grid": dem_grid,                     # Digital elevation model grid
    "colormap": "cool",                       # Colormap for visualization
    "obj_export": True,                       # Whether to export as OBJ file
    "output_directory": "output",             # Directory to save output files
    "output_file_name": "landmark_visibility" # Base filename for outputs
}
landmark_vis_map = get_landmark_visibility_map(voxcity_grid, building_id_grid, building_gdf, meshsize, **landmark_kwargs)
```
<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/landmark.png" alt="Landmark Visibility Map Rendered in Rhino" width="500">
</p>
<p align="center">
  <em>Example Result Saved as OBJ and Rendered in Rhino</em>
</p>

#### Network Analysis:

```python
from voxcity.geoprocessor.network import get_network_values

network_kwargs = {
    "network_type": "walk",        # Type of network to download from OSM (walk, drive, all, etc.)
    "colormap": "magma",          # Matplotlib colormap for visualization
    "vis_graph": True,            # Whether to display the network visualization
    "vmin": 0.0,                  # Minimum value for color scaling
    "vmax": 600000,               # Maximum value for color scaling
    "edge_width": 2,              # Width of network edges in visualization
    "alpha": 0.8,                 # Transparency of network edges
    "zoom": 16                    # Zoom level for basemap
}

G, edge_gdf = get_network_values(
    cum_solar_grid,               # Grid of cumulative solar irradiance values
    rectangle_vertices,           # Coordinates defining simulation domain boundary
    meshsize,                     # Size of each grid cell in meters
    value_name='Cumulative Global Solar Irradiance (W/m²·hour)',  # Label for values in visualization
    **network_kwargs              # Additional visualization and network parameters
)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/kunifujiwara/VoxCity/main/images/network.png" alt="Example of Graph Output" width="500">
</p>
<p align="center">
  <em>Cumulative Global Solar Irradiance (kW/m²·hour) on Road Network</em>
</p>

## References of Data Sources

### Building 

| Dataset | Spatial Coverage | Source/Data Acquisition |
|---------|------------------|------------------------|
| [OpenStreetMap](https://www.openstreetmap.org) | Worldwide (24% completeness in city centers) | Volunteered / updated continuously |
| [Microsoft Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | North America, Europe, Australia | Prediction from satellite or aerial imagery / 2018-2019 for majority of the input imagery |
| [Open Buildings 2.5D Temporal Dataset](https://sites.research.google/gr/open-buildings/temporal/) | Africa, Latin America, and South and Southeast Asia | Prediction from satellite imagery / 2016-2023 |
| [EUBUCCO v0.1](https://eubucco.com/) | 27 EU countries and Switzerland (378 regions and 40,829 cities) | OpenStreetMap, government datasets / 2003-2021 (majority is after 2019) |
| [UT-GLOBUS](https://zenodo.org/records/11156602) | Worldwide (more than 1200 cities or locales) | Prediction from building footprints, population, spaceborne nDSM / not provided |
| [Overture Maps](https://overturemaps.org/) | Worldwide | OpenStreetMap, Esri Community Maps Program, Google Open Buildings, etc. / updated continuously |

### Tree Canopy Height

| Dataset | Coverage | Resolution | Source/Data Acquisition |
|---------|-----------|------------|------------------------|
| [High Resolution 1m Global Canopy Height Maps](https://sustainability.atmeta.com/blog/2024/04/22/using-artificial-intelligence-to-map-the-earths-forests/) | Worldwide | 1 m | Prediction from satellite imagery / 2009 and 2020 (80% are 2018-2020) |
| [ETH Global Sentinel-2 10m Canopy Height (2020)](https://langnico.github.io/globalcanopyheight/) | Worldwide | 10 m | Prediction from satellite imagery / 2020 |

### Land Cover

| Dataset | Spatial Coverage | Resolution | Source/Data Acquisition |
|---------|------------------|------------|----------------------|
| [ESA World Cover 10m 2021 V200](https://zenodo.org/records/7254221) | Worldwide | 10 m | Prediction from satellite imagery / 2021 |
| [ESRI 10m Annual Land Cover (2017-2023)](https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d) | Worldwide | 10 m | Prediction from satellite imagery / 2017-2023 |
| [Dynamic World V1](https://dynamicworld.app) | Worldwide | 10 m | Prediction from satellite imagery / updated continuously |
| [OpenStreetMap](https://www.openstreetmap.org) | Worldwide | - (Vector) | Volunteered / updated continuously |
| [OpenEarthMap Japan](https://www.open-earth-map.org/demo/Japan/leaflet.html) | Japan | ~1 m | Prediction from aerial imagery / 1974-2022 (mostly after 2018 in major cities) |
| [UrbanWatch](https://urbanwatch.charlotte.edu/) | 22 major cities in the US | 1 m | Prediction from aerial imagery / 2014–2017 |

### Terrain Elevation

| Dataset | Coverage | Resolution | Source/Data Acquisition |
|---------|-----------|------------|------------------------|
| [FABDEM](https://doi.org/10.5523/bris.25wfy0f9ukoge2gs7a5mqpq2j7) | Worldwide | 30 m | Correction of Copernicus DEM using canopy height and building footprints data / 2011-2015 (Copernicus DEM) |
| [DeltaDTM](https://gee-community-catalog.org/projects/delta_dtm/) | Worldwide (Only for coastal areas below 10m + mean sea level) | 30 m | Copernicus DEM, spaceborne LiDAR / 2011-2015 (Copernicus DEM) |
| [USGS 3DEP 1m DEM](https://www.usgs.gov/3d-elevation-program) | United States | 1 m | Aerial LiDAR / 2004-2024 (mostly after 2015) |
| [England 1m Composite DTM](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) | England | 1 m | Aerial LiDAR / 2000-2022 |
| [Australian 5M DEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/89644) | Australia | 5 m | Aerial LiDAR / 2001-2015 |
| [RGE Alti](https://geoservices.ign.fr/rgealti) | France | 1 m | Aerial LiDAR |


## Citation

Please cite the [paper](https://doi.org/10.48550/arXiv.2504.13934) if you use `voxcity` in a scientific publication:

Fujiwara K, Tsurumi R, Kiyono T, Fan Z, Liang X, Lei B, Yap W, Ito K, Biljecki F. VoxCity: A Seamless Framework for Open Geospatial Data Integration, Grid-Based Semantic 3D City Model Generation, and Urban Environment Simulation. arXiv preprint arXiv:2504.13934. 2025.

```bibtex
@article{fujiwara2025voxcity,
  title={VoxCity: A Seamless Framework for Open Geospatial Data Integration, Grid-Based Semantic 3D City Model Generation, and Urban Environment Simulation},
  author={Fujiwara, Kunihiko and Tsurumi, Ryuta and Kiyono, Tomoki and Fan, Zicheng and Liang, Xiucheng and Lei, Binyu and Yap, Winston and Ito, Koichi and Biljecki, Filip},
  journal={arXiv preprint arXiv:2504.13934},
  year={2025},
  doi = {10.48550/arXiv.2504.13934},
}
```

## Credit

 - Tutorial video by <a href="https://ual.sg/author/liang-xiucheng/">Xiucheng Liang</a>

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.

--------------------------------------------------------------------------------
<br>
<br>
<p align="center">
  <a href="https://ual.sg/">
    <img src="https://raw.githubusercontent.com/winstonyym/urbanity/main/images/ualsg.jpeg" width = 55% alt="Logo">
  </a>
</p>

