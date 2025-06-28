# References

VoxCity is a comprehensive Python package for grid-based 3D city model generation and urban simulation. The following references are provided to give credit to the original authors of the tools and datasets used in VoxCity. Please cite them when using VoxCity in your research.

## Main Classes and Functions

### Generator Module
- `get_voxcity`: Main function for generating voxel city models
- Building data sources: OpenStreetMap {cite}`openstreetmap_2023`, EUBUCCO {cite}`brussee_2023_eubucco`, Overture Maps {cite}`li_2023_overture`, Microsoft Building Footprints, OpenBuilding 2.5D {cite}`wang_2023_openbuilding`
- Land cover data sources: UrbanWatch {cite}`liu_2023_urbanwatch`, ESA WorldCover {cite}`esa_2021_worldcover`, ESRI Land Cover {cite}`lang_2023_esri_landcover`, Dynamic World {cite}`potapov_2022_dynamic_world`, OpenStreetMap {cite}`openstreetmap_2023`
- Canopy height data sources: High Resolution 1m Global Canopy Height Maps {cite}`lang_2023_global_canopy_height`, ETH Global Sentinel-2 10m {cite}`schug_2023_eth_canopy_height`
- DEM data sources: DeltaDTM {cite}`hawker_2022_deltadtm`, FABDEM {cite}`hawker_2022_fabdem`, NASA {cite}`nasadem_2019`, COPERNICUS

### Downloader Module
- `OSMDownloader`: Downloader for OpenStreetMap building data {cite}`openstreetmap_2023`
- `EUBUCCODownloader`: Downloader for EUBUCCO building data {cite}`brussee_2023_eubucco`
- `OvertureDownloader`: Downloader for Overture Maps building data {cite}`li_2023_overture`
- `GEEDownloader`: Downloader for Google Earth Engine data {cite}`google_earth_engine_2023`

### Exporter Module
- `ENVIMETExporter`: Exporter for ENVI-met simulation files {cite}`envi_met_2020`
- `MagicaVoxelExporter`: Exporter for MagicaVoxel voxel files {cite}`magicavoxel_2020`
- `OBJExporter`: Exporter for OBJ 3D model files

### Simulator Module
- `SolarSimulator`: Class for solar radiation analysis
- `ViewSimulator`: Class for view index and visibility analysis

### Geoprocessor Module
- `GridProcessor`: Class for grid-based data processing
- `MeshProcessor`: Class for mesh generation and processing
- `PolygonProcessor`: Class for polygon operations

## Bibliography

```{bibliography}