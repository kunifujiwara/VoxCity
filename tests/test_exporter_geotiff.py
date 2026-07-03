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
