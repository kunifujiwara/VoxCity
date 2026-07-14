# Data sources

VoxCity integrates many open geospatial datasets. This reference lists the
datasets available for each layer of a city model — building footprints and
heights, tree canopy height, land cover, and terrain elevation — together with
their coverage, resolution, and provenance.

For guidance on choosing between these sources, see the
{doc}`Choosing data sources guide <../guides/data_sources>`.

## Building

| Dataset | Spatial Coverage | Source/Data Acquisition |
|---------|------------------|------------------------|
| [OpenStreetMap](https://www.openstreetmap.org) | Worldwide (24% completeness in city centers) | Volunteered / updated continuously |
| [Microsoft Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | North America, Europe, Australia | Prediction from satellite or aerial imagery / 2018-2019 for majority of the input imagery |
| [Open Buildings 2.5D Temporal Dataset](https://sites.research.google/gr/open-buildings/temporal/) | Africa, Latin America, and South and Southeast Asia | Prediction from satellite imagery / 2016-2023 |
| [EUBUCCO v0.1](https://eubucco.com/) | 27 EU countries and Switzerland (378 regions and 40,829 cities) | OpenStreetMap, government datasets / 2003-2021 (majority is after 2019) |
| [UT-GLOBUS](https://zenodo.org/records/11156602) | Worldwide (more than 1200 cities or locales) | Prediction from building footprints, population, spaceborne nDSM / not provided |
| [Overture Maps](https://overturemaps.org/) | Worldwide | OpenStreetMap, Esri Community Maps Program, Google Open Buildings, etc. / updated continuously |

## Tree canopy height

| Dataset | Coverage | Resolution | Source/Data Acquisition |
|---------|-----------|------------|------------------------|
| [High Resolution 1m Global Canopy Height Maps](https://sustainability.atmeta.com/blog/2024/04/22/using-artificial-intelligence-to-map-the-earths-forests/) | Worldwide | 1 m | Prediction from satellite imagery / 2009 and 2020 (80% are 2018-2020) |
| [ETH Global Sentinel-2 10m Canopy Height (2020)](https://langnico.github.io/globalcanopyheight/) | Worldwide | 10 m | Prediction from satellite imagery / 2020 |

## Land cover

| Dataset | Spatial Coverage | Resolution | Source/Data Acquisition |
|---------|------------------|------------|----------------------|
| [ESA World Cover 10m 2021 V200](https://zenodo.org/records/7254221) | Worldwide | 10 m | Prediction from satellite imagery / 2021 |
| [ESRI 10m Annual Land Cover (2017-2023)](https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d) | Worldwide | 10 m | Prediction from satellite imagery / 2017-2023 |
| [Dynamic World V1](https://dynamicworld.app) | Worldwide | 10 m | Prediction from satellite imagery / updated continuously |
| [OpenStreetMap](https://www.openstreetmap.org) | Worldwide | - (Vector) | Volunteered / updated continuously |
| [OpenEarthMap Japan](https://www.open-earth-map.org/demo/Japan/leaflet.html) | Japan | ~1 m | Prediction from aerial imagery / 1974-2022 (mostly after 2018 in major cities) |
| [UrbanWatch](https://urbanwatch.charlotte.edu/) | 22 major cities in the US | 1 m | Prediction from aerial imagery / 2014–2017 |

## Terrain elevation

| Dataset | Coverage | Resolution | Source/Data Acquisition |
|---------|-----------|------------|------------------------|
| [FABDEM](https://doi.org/10.5523/bris.25wfy0f9ukoge2gs7a5mqpq2j7) | Worldwide | 30 m | Correction of Copernicus DEM using canopy height and building footprints data / 2011-2015 (Copernicus DEM) |
| [DeltaDTM](https://gee-community-catalog.org/projects/delta_dtm/) | Worldwide (Only for coastal areas below 10m + mean sea level) | 30 m | Copernicus DEM, spaceborne LiDAR / 2011-2015 (Copernicus DEM) |
| [USGS 3DEP 1m DEM](https://www.usgs.gov/3d-elevation-program) | United States | 1 m | Aerial LiDAR / 2004-2024 (mostly after 2015) |
| [England 1m Composite DTM](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) | England | 1 m | Aerial LiDAR / 2000-2022 |
| [Australian 5M DEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/89644) | Australia | 5 m | Aerial LiDAR / 2001-2015 |
| [RGE Alti](https://geoservices.ign.fr/rgealti) | France | 1 m | Aerial LiDAR |

## Citing data sources

Please credit the original authors of any dataset you use. Full citations are
collected in the {doc}`Bibliography <../bibliography>`.
