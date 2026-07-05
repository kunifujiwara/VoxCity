from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from voxcity.exporter.geotiff import _north_up_affine_and_array
from voxcity.geoprocessor.raster.core import compute_cell_center_coords

# Axis-aligned rectangle [SW, SE, NE, NW] near Tokyo
RECT = [(139.70, 35.60), (139.71, 35.60), (139.71, 35.61), (139.70, 35.61)]
MESH = 30.0


def _voxcity_index_grid(rect, mesh):
    """Return (grid, cc) where grid[i, j] == i*1000 + j for shape (nx, ny)."""
    cc = compute_cell_center_coords(rect, mesh)
    nx, ny = cc["grid_size"]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    grid = (ii * 1000 + jj).astype(np.float64)
    return grid, cc


def test_north_up_affine_and_array_orientation():
    grid, cc = _voxcity_index_grid(RECT, MESH)
    nx, ny = cc["grid_size"]
    array, transform = _north_up_affine_and_array(grid, RECT, MESH)

    # Raster shape is (rows=ny, cols=nx); north-up => negative row step
    assert array.shape == (ny, nx)
    assert isinstance(transform, Affine)
    assert transform.e < 0

    # Cell (i, j) lives at raster (row = ny-1-j, col = i)
    for i in (0, nx - 1):
        for j in (0, ny - 1):
            r, c = ny - 1 - j, i
            assert array[r, c] == grid[i, j]
            # affine pixel-center maps back to the true cell center lon/lat
            lon, lat = transform * (c + 0.5, r + 0.5)
            np.testing.assert_allclose(lon, cc["lons"][i, j], atol=1e-9)
            np.testing.assert_allclose(lat, cc["lats"][i, j], atol=1e-9)


def test_north_up_affine_and_array_shape_mismatch():
    with pytest.raises(ValueError):
        _north_up_affine_and_array(np.zeros((3, 3)), RECT, MESH)


def test_north_up_affine_and_array_insufficient_vertices():
    with pytest.raises(ValueError):
        _north_up_affine_and_array(np.zeros((1, 1)), RECT[:2], MESH)


import rasterio
from voxcity.exporter.geotiff import export_grid_geotiff


def test_export_grid_geotiff_float_roundtrip(tmp_path):
    grid, cc = _voxcity_index_grid(RECT, MESH)
    nx, ny = cc["grid_size"]
    out = tmp_path / "layer.tif"

    path = export_grid_geotiff(
        grid, RECT, MESH, out, dtype="float32", nodata=float("nan")
    )

    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:4326"
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        assert src.transform.e < 0                      # north-up
        assert np.isnan(src.nodata)
        data = src.read(1)
        assert data.shape == (ny, nx)
        # A zero value must survive as real data, not nodata
        z = np.zeros((nx, ny), dtype=np.float32)
        z_path = export_grid_geotiff(z, RECT, MESH, tmp_path / "z.tif",
                                     dtype="float32", nodata=float("nan"))
        with rasterio.open(z_path) as zsrc:
            assert np.all(zsrc.read(1) == 0)
            assert not np.isnan(zsrc.read(1)).any()


def test_export_grid_geotiff_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError):
        export_grid_geotiff(np.zeros((2, 2, 2)), RECT, MESH, tmp_path / "x.tif")


def test_export_grid_geotiff_color_table_and_names(tmp_path):
    grid, cc = _voxcity_index_grid(RECT, MESH)
    # Use small integer class indices for a categorical layer
    nx, ny = cc["grid_size"]
    classes = (np.arange(nx * ny).reshape(nx, ny) % 3).astype(np.uint8)

    color_table = {0: (10, 20, 30), 1: (40, 50, 60), 2: (70, 80, 90)}
    names = {0: "Water", 1: "Tree", 2: "Building"}
    out = tmp_path / "lc.tif"

    export_grid_geotiff(
        classes, RECT, MESH, out, dtype="uint8",
        color_table=color_table, category_names=names,
    )

    with rasterio.open(out) as src:
        assert src.dtypes[0] == "uint8"
        cmap = src.colormap(1)
        assert cmap[0][:3] == (10, 20, 30)
        assert cmap[2][:3] == (70, 80, 90)
        tags = src.tags(1)
        assert tags["0"] == "Water"
        assert tags["2"] == "Building"


def test_export_grid_geotiff_color_table_no_gdal_write_error(tmp_path, caplog):
    """Regression test: colormap/tags must be set before dst.write(), or
    libtiff logs a 'Cannot modify tag ... while writing' GDAL error.

    rasterio surfaces GDAL error/warning messages via the 'rasterio._env'
    logger (see rasterio._err), so we capture at INFO level and assert the
    known bad-ordering message never appears.
    """
    grid, cc = _voxcity_index_grid(RECT, MESH)
    nx, ny = cc["grid_size"]
    classes = (np.arange(nx * ny).reshape(nx, ny) % 3).astype(np.uint8)
    color_table = {0: (10, 20, 30), 1: (40, 50, 60), 2: (70, 80, 90)}
    names = {0: "Water", 1: "Tree", 2: "Building"}
    out = tmp_path / "lc_gdal_check.tif"

    with caplog.at_level("INFO", logger="rasterio._env"):
        export_grid_geotiff(
            classes, RECT, MESH, out, dtype="uint8",
            color_table=color_table, category_names=names,
        )

    for record in caplog.records:
        msg = record.getMessage()
        assert "Cannot modify tag" not in msg
        assert "TIFFSetField" not in msg


from voxcity.exporter.geotiff import export_geotiffs
from voxcity.models import (
    VoxCity, GridMetadata, LandCoverGrid, BuildingGrid, DemGrid, CanopyGrid, VoxelGrid,
)


def _make_voxcity(rect, mesh, source="Standard"):
    cc = compute_cell_center_coords(rect, mesh)
    nx, ny = cc["grid_size"]
    meta = GridMetadata(crs="EPSG:4326",
                        bounds=(rect[0][0], rect[0][1], rect[2][0], rect[2][1]),
                        meshsize=mesh)
    classes = (np.arange(nx * ny).reshape(nx, ny) % 3).astype(np.int32)
    heights = np.zeros((nx, ny), dtype=np.float64)
    heights[0, 0] = 12.0
    dem = np.full((nx, ny), 5.0, dtype=np.float64)
    canopy = np.zeros((nx, ny), dtype=np.float64)
    return VoxCity(
        voxels=VoxelGrid(classes=np.zeros((nx, ny, 1)), meta=meta),
        buildings=BuildingGrid(heights=heights,
                               min_heights=np.empty((nx, ny), dtype=object),
                               ids=np.zeros((nx, ny)), meta=meta),
        land_cover=LandCoverGrid(classes=classes, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, meta=meta),
        extras={"rectangle_vertices": rect, "land_cover_source": source},
    )


def test_export_geotiffs_writes_all_layers(tmp_path):
    city = _make_voxcity(RECT, MESH)
    written = export_geotiffs(city, tmp_path, base_filename="city")

    assert set(written) == {"land_cover", "building_height", "dem", "canopy_height"}
    for layer, path in written.items():
        assert Path(path).exists()
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:4326"
    # land cover is uint8 with a color table; heights are float32
    with rasterio.open(written["land_cover"]) as src:
        assert src.dtypes[0] == "uint8"
        assert src.colormap(1)  # non-empty
    with rasterio.open(written["building_height"]) as src:
        assert src.dtypes[0] == "float32"
        assert src.read(1)[src.height - 1, 0] == 12.0  # (i=0,j=0) -> row ny-1


def test_export_geotiffs_requires_rectangle_vertices(tmp_path):
    city = _make_voxcity(RECT, MESH)
    city.extras.pop("rectangle_vertices")
    with pytest.raises(ValueError):
        export_geotiffs(city, tmp_path)


def test_export_geotiffs_skips_missing_layer(tmp_path):
    city = _make_voxcity(RECT, MESH)
    city.dem.elevation = None
    with pytest.warns(UserWarning):
        written = export_geotiffs(city, tmp_path)
    assert set(written) == {"land_cover", "building_height", "canopy_height"}


def test_export_geotiffs_land_cover_missing_source_warns(tmp_path):
    city = _make_voxcity(RECT, MESH)
    city.extras.pop("land_cover_source")
    with pytest.warns(UserWarning):
        written = export_geotiffs(city, tmp_path)
    assert "land_cover" in written
    with rasterio.open(written["land_cover"]) as src:
        assert src.dtypes[0] == "uint8"
        # no color table was available, so no colormap was ever set
        with pytest.raises(ValueError):
            src.colormap(1)


def test_export_geotiffs_land_cover_unknown_source_warns(tmp_path):
    city = _make_voxcity(RECT, MESH, source="NotARealSource")
    with pytest.warns(UserWarning):
        written = export_geotiffs(city, tmp_path)
    assert "land_cover" in written


def test_export_geotiffs_land_cover_out_of_range_warns(tmp_path):
    """Regression test: a -1 (unmatched-class) value must not silently
    wrap around to 255 when cast to uint8 -- it should raise a warning."""
    city = _make_voxcity(RECT, MESH)
    city.land_cover.classes[0, 0] = -1
    with pytest.warns(UserWarning, match="uint8 range"):
        written = export_geotiffs(city, tmp_path)
    assert "land_cover" in written
    with rasterio.open(written["land_cover"]) as src:
        assert src.dtypes[0] == "uint8"
        # confirm the documented (if undesirable) wraparound actually happens,
        # i.e. the warning corresponds to real behavior, not a phantom check
        assert src.read(1)[src.height - 1, 0] == 255


def test_geotiff_exporter_class(tmp_path):
    from voxcity.exporter import GeoTIFFExporter  # must be re-exported
    city = _make_voxcity(RECT, MESH)
    written = GeoTIFFExporter().export(city, str(tmp_path), "city")
    assert set(written) == {"land_cover", "building_height", "dem", "canopy_height"}
    for path in written.values():
        assert Path(path).exists()


def test_geotiff_exporter_rejects_non_voxcity(tmp_path):
    from voxcity.exporter import GeoTIFFExporter
    with pytest.raises(TypeError):
        GeoTIFFExporter().export(object(), str(tmp_path), "x")


# Canonical VoxCity vertex order [SW, NW, NE, SE] -> u_vec=north, v_vec=east.
# This is what real generated models use (e.g. demo/output/tokyo/voxcity.h5).
RECT_CANON = [
    (139.75664, 35.67358),   # SW (origin)
    (139.75664, 35.67809),   # NW  -> side_1 = north
    (139.76216, 35.67809),   # NE
    (139.76216, 35.67358),   # SE  -> side_2 = east
]


def test_north_up_affine_is_diagonal_for_canonical_order():
    """For an axis-aligned AOI the affine must be north-up (diagonal), not rotated,
    regardless of whether u_vec points east or north."""
    grid, cc = _voxcity_index_grid(RECT_CANON, MESH)
    array, transform = _north_up_affine_and_array(grid, RECT_CANON, MESH)
    # North-up diagonal affine: zero rotation/shear terms, +x east, -y south.
    assert abs(transform.b) < 1e-12, f"expected no shear, got b={transform.b}"
    assert abs(transform.d) < 1e-12, f"expected no rotation, got d={transform.d}"
    assert transform.a > 0
    assert transform.e < 0


def test_export_places_corner_marker_north_up_canonical(tmp_path):
    """A marker at the geographic NE corner must land at the top-right of the
    written raster (row 0 = north, last col = east) for the canonical order."""
    cc = compute_cell_center_coords(RECT_CANON, MESH)
    nx, ny = cc["grid_size"]
    grid = np.zeros((nx, ny), dtype=np.float64)
    # NE corner in cell-index space = (max east, max north). With u=north/v=east,
    # east is axis j and north is axis i, so NE = (i=nx-1, j=ny-1) as always
    # (cc.lons/lats are indexed [i,j] identically to grid).
    grid[nx - 1, ny - 1] = 99.0
    ne_lon, ne_lat = cc["lons"][nx - 1, ny - 1], cc["lats"][nx - 1, ny - 1]
    sw_lon, sw_lat = cc["lons"][0, 0], cc["lats"][0, 0]

    out = tmp_path / "canon.tif"
    export_grid_geotiff(grid, RECT_CANON, MESH, out, dtype="float32", nodata=float("nan"))
    with rasterio.open(out) as src:
        assert src.transform.e < 0  # north-up
        arr = src.read(1)
        # Marker sampled at its TRUE lon/lat comes back as 99.
        r, c = src.index(ne_lon, ne_lat)
        assert arr[r, c] == 99.0
        # NE geographic corner is at top-right of a north-up raster.
        assert r <= 1, f"NE marker should be in the top (north) rows, got row {r}"
        assert c >= arr.shape[1] - 2, f"NE marker should be in the right (east) cols, got col {c}"
        # SW corner holds background (0), and sits bottom-left.
        r0, c0 = src.index(sw_lon, sw_lat)
        assert arr[r0, c0] == 0.0
        assert r0 >= arr.shape[0] - 2 and c0 <= 1


def test_export_geotiff_georef_roundtrip_canonical(tmp_path):
    """Sampling the written raster at every cell's true lon/lat returns the
    original grid value (georeferencing is exact for the canonical order)."""
    grid, cc = _voxcity_index_grid(RECT_CANON, MESH)
    nx, ny = cc["grid_size"]
    lons, lats = cc["lons"], cc["lats"]
    out = tmp_path / "canon_rt.tif"
    export_grid_geotiff(grid, RECT_CANON, MESH, out, dtype="float64")
    with rasterio.open(out) as src:
        arr = src.read(1)
        for i in range(0, nx, max(1, nx // 7)):
            for j in range(0, ny, max(1, ny // 7)):
                r, c = src.index(lons[i, j], lats[i, j])
                assert arr[r, c] == grid[i, j], f"cell ({i},{j}) misplaced"


def test_export_non_square_marker_placement(tmp_path):
    """Non-square AOI: a marker near the NE corner lands at the correct
    north-up raster corner, and the output raster shape spans (lat, lon)."""
    # Non-square: ~2.7 km E-W by ~1.1 km N-S. Canonical [SW, NW, NE, SE] order.
    lon0, lat0, dlon_deg, dlat_deg = 139.70, 35.60, 0.030, 0.010
    rect = [
        (lon0, lat0),                       # SW
        (lon0, lat0 + dlat_deg),            # NW  -> side_1 = north
        (lon0 + dlon_deg, lat0 + dlat_deg), # NE
        (lon0 + dlon_deg, lat0),            # SE  -> side_2 = east
    ]
    cc = compute_cell_center_coords(rect, MESH)
    nx, ny = cc["grid_size"]
    assert nx != ny  # sanity: genuinely non-square
    grid = np.zeros((nx, ny), dtype=np.float64)
    grid[nx - 1, ny - 1] = 42.0  # NE corner cell

    out = tmp_path / "nonsquare.tif"
    export_grid_geotiff(grid, rect, MESH, out, dtype="float32", nodata=float("nan"))
    with rasterio.open(out) as src:
        assert src.transform.e < 0  # north-up
        arr = src.read(1)
        # north-up raster: rows span latitude, cols span longitude.
        n_lat = len(np.unique(np.round(cc["lats"], 9)))
        n_lon = len(np.unique(np.round(cc["lons"], 9)))
        assert arr.shape == (n_lat, n_lon)
        r, c = src.index(cc["lons"][nx - 1, ny - 1], cc["lats"][nx - 1, ny - 1])
        assert arr[r, c] == 42.0
        assert r <= 1 and c >= arr.shape[1] - 2  # top-right = NE


import os

from voxcity.exporter.geotiff import export_geotiffs

_TOKYO_H5 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "demo", "output", "tokyo", "voxcity.h5",
)

@pytest.mark.skipif(not os.path.exists(_TOKYO_H5), reason="demo tokyo model not present")
def test_export_geotiffs_real_model_is_north_up(tmp_path):
    from voxcity.io import load_voxcity

    city = load_voxcity(_TOKYO_H5)
    written = export_geotiffs(city, tmp_path, base_filename="voxcity")
    assert set(written) >= {"building_height", "dem", "canopy_height", "land_cover"}

    rect = city.extras["rectangle_vertices"]
    ms = city.buildings.meta.meshsize
    cc = compute_cell_center_coords(rect, ms)
    lons, lats = cc["lons"], cc["lats"]
    nx, ny = cc["grid_size"]

    layer_grid = {
        "building_height": np.asarray(city.buildings.heights, dtype=float),
        "dem": np.asarray(city.dem.elevation, dtype=float),
        "canopy_height": np.asarray(city.tree_canopy.top, dtype=float),
    }
    for layer, path in written.items():
        with rasterio.open(path) as src:
            # every written layer must be north-up (diagonal, -y south)
            assert abs(src.transform.b) < 1e-12 and abs(src.transform.d) < 1e-12, layer
            assert src.transform.e < 0, layer
            # georef round-trip for the float layers we can compare exactly
            if layer in layer_grid:
                arr = src.read(1)
                g = layer_grid[layer]
                for i in range(0, nx, max(1, nx // 6)):
                    for j in range(0, ny, max(1, ny // 6)):
                        r, c = src.index(lons[i, j], lats[i, j])
                        got, exp = arr[r, c], g[i, j]
                        assert (np.isnan(got) and np.isnan(exp)) or np.isclose(got, exp), \
                            f"{layer} cell ({i},{j}) misplaced"
