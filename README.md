# VoxelCity

VoxelCity is a Python package for creating voxel-based 3D city models by integrating various geospatial data sources including building footprints, land cover, canopy height, and digital elevation models.

## Installation

```bash
pip install voxelcity
```

## Prerequisites

- A Google Earth Engine enabled Cloud Project (Setup instructions: [Earth Engine Cloud Project Setup](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup))

## Basic Usage

```python
from voxelcity import get_voxelcity

# Define target area coordinates (rectangle vertices)
rectangle_vertices = [
    (latitude1, longitude1),
    (latitude2, longitude2),
    (latitude3, longitude3),
    (latitude4, longitude4)
]

# Get voxel data
voxelcity_grid, building_height_grid, building_min_height_grid, building_id_grid, \
canopy_height_grid, land_cover_grid, dem_grid, building_geojson = get_voxelcity(
    rectangle_vertices,
    building_source='OpenStreetMap',
    land_cover_source='OpenStreetMap',
    canopy_height_source='High Resolution 1m Global Canopy Height Maps',
    dem_source='DeltaDTM',
    meshsize=5
)
```

## Available Data Sources

### Building Sources
- OpenStreetMap
- Overture
- EUBUCCO v0.1
- Open Building 2.5D Temporal
- Microsoft Building Footprints
- OpenMapTiles
- Local file

### Land Cover Sources
- Urbanwatch
- OpenEarthMapJapan
- ESA WorldCover
- ESRI 10m Annual Land Cover
- Dynamic World V1
- OpenStreetMap

### Canopy Height Sources
- High Resolution 1m Global Canopy Height Maps
- ETH Global Sentinel-2 10m Canopy Height (2020)

### Digital Elevation Model Sources
- DeltaDTM
- FABDEM
- England 1m DTM
- DEM France 1m
- AUSTRALIA 5M DEM
- USGS 3DEP 1m
- NASA
- COPERNICUS
- Flat

## Export Options

### ENVI-MET Export
```python
from voxelcity.file.envimet import export_inx, generate_edb_file

# Export INX file
export_inx(
    building_height_grid,
    building_id_grid,
    canopy_height_grid,
    land_cover_grid,
    dem_grid,
    meshsize,
    land_cover_source,
    rectangle_vertices,
    output_directory='output/test',
    author_name="Your Name",
    model_description="Model description",
    domain_building_max_height_ratio=2,
    useTelescoping_grid=True,
    verticalStretch=20,
    min_grids_Z=20,
    lad=1.0
)

# Generate EDB file
generate_edb_file(output_directory='output/test')
```

### MagicaVoxel Export
```python
from voxelcity.file.magicavoxel import export_magicavoxel_vox

export_magicavoxel_vox(
    voxelcity_grid,
    output_path="output/magicavoxel",
    base_filename="project_name"
)
```

### OBJ Export
```python
from voxelcity.file.obj import export_obj

export_obj(
    voxelcity_grid,
    output_directory='./output/test',
    output_file_name='voxcity_test',
    meshsize=5
)
```

## Additional Features

### View Index Simulation
```python
from voxelcity.sim.view import get_green_view_index, get_sky_view_index

# Calculate Green View Index
gvi_grid = get_green_view_index(
    voxelcity_grid,
    meshsize,
    view_point_height=1.5,
    dem_grid=dem_grid,
    colormap='viridis',
    obj_export=True,
    output_directory='output/test',
    output_file_name='gvi_test'
)

# Calculate Sky View Index
svi_grid = get_sky_view_index(
    voxelcity_grid,
    meshsize,
    view_point_height=1.5,
    dem_grid=dem_grid,
    colormap='BuPu_r',
    obj_export=True,
    output_directory='output/test',
    output_file_name='svi_test'
)
```

### Landmark Visibility Analysis
```python
from voxelcity.sim.view import get_landmark_visibility_map

landmark_vis_map = get_landmark_visibility_map(
    voxelcity_grid,
    building_id_grid,
    building_geojson,
    meshsize,
    view_point_height=1.5,
    rectangle_vertices=rectangle_vertices,
    dem_grid=dem_grid,
    colormap='cool',
    obj_export=True,
    output_directory='output/test',
    output_file_name='landmark_visibility_test'
)
```

## Optional Parameters

The following optional parameters can be passed to `get_voxelcity()`:

```python
kwargs = {
    "building_path": 'path_to_building_source_file',  # For local building source files
    "building_complementary_path": 'path_to_building_complementary_source_file',
    "building_complementary_source": 'None',
    "complement_polygon": True,
    "output_dir": 'output/test',
    "remove_perimeter_object": 0.1,
    "gridvis": True,
    "mapvis": False,
    "voxelvis": False,
    "voxelvis_img_save_path": None,
    "maptiler_API_key": 'your_API_key',
    "trunk_height_ratio": None,
    "min_canopy_height": None,
    "dem_interpolation": True,
    "dynamic_world_date": '2021-04-02',
    "esri_landcover_year": '2023'
}
```

## Data Source References

### Building Data
- OpenStreetMap: [OSM Building Key](https://wiki.openstreetmap.org/wiki/Key:building)
- Microsoft Building Footprints: [Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints)

### Land Cover Data
- Urbanwatch: Coverage includes 22 major US cities. [More info](https://urbanwatch.charlotte.edu/)
- OpenEarthMapJapan: High-resolution land cover data for Japan
- ESA WorldCover: Global 10m resolution land cover data

### Digital Elevation Models
- DeltaDTM: Global coastal area DTM
- NASA DEM: Global coverage, ≤30m resolution
- COPERNICUS DEM: Global coverage, ≤30m resolution

For detailed citation information and additional data sources, please refer to the documentation.