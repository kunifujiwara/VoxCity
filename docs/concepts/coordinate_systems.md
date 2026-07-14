# Coordinate systems and geodesy

This page summarizes how VoxCity handles coordinate reference systems (CRS) and
geodesy. A more detailed Japanese write-up is available in
{doc}`coordinate_systems_ja`.

## Internally everything is WGS84 (EPSG:4326)

VoxCity keeps all data in **WGS84 longitude/latitude** (`lon, lat`). Every
downloader normalizes its output to EPSG:4326, and the target area
(`rectangle_vertices`) is expected in `(lon, lat)`.

The one notable exception is PLATEAU/CityGML data, which is kept in EPSG:6697
(JGD2011) with only an axis-order swap — no reprojection to 4326. At
neighborhood scale the two are within sub-meter agreement, so this is safe in
practice.

## Distances use ellipsoidal geodesics

Distance and scale calculations are performed on the **WGS84 ellipsoid** using
`pyproj.Geod`, not a spherical approximation. A shared `Geod(ellps='WGS84')`
singleton computes geodesic (inverse) distances, and a helper converts between
meters and degrees locally. A spherical `haversine_distance` also exists as a
fallback.

## Local grid coordinates (uv_m)

Simulation grids do not use longitude/latitude directly. Instead a `GridProjector`
defines two systems:

- `lon_lat` — geographic coordinates (WGS84).
- `uv_m` — meters from the grid origin (`rectangle_vertices[0]`).

Conversion between them is a precomputed 2×2 affine matrix, giving O(1)
transforms in both directions. Mesh dimensions are derived from accurate
ellipsoidal geodesic distances, while the grid itself is an affine parallelogram
in lon/lat space — accurate locally, with some distortion over very large areas.

## CRS conversion utilities

General-purpose conversions live in the geoprocessor utilities. Transformers are
built with `always_xy=True` (to avoid axis-order confusion) and cached to avoid
rebuild cost. GeoDataFrames are normalized to EPSG:4326, assuming 4326 with a
warning when a CRS is missing.

## Dynamic projection for raster/DEM work

When reading GeoTIFFs, VoxCity respects the file's own CRS and only reprojects
when necessary. For DEM and canopy interpolation, it dynamically computes the
local **UTM zone** from the area center and interpolates in meters to avoid
distortion.

## Native CRS by source

| Source | Native CRS | Handling |
|--------|-----------|----------|
| GSI DEM (Japan) | EPSG:3857 (Web Mercator) | Reprojected downstream during DEM grid generation |
| OEMJ | EPSG:3857 | Reprojected downstream |
| PLATEAU/CityGML | EPSG:6697 (JGD2011 lat/lon + elevation) | Kept as 6697; axis-order swap only |
| GEE | EPSG:4326 | Specified on export |
| OSM buffering | Temporary Albers equal-area (AEA) | Projected for buffer operations |

For the full details and source-code references, see
{doc}`coordinate_systems_ja`.
