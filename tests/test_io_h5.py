"""Tests for voxcity.io – HDF5 save/load (save_results_h5, load_results_h5).

Covers round-trip fidelity, edge cases, error handling, and helper functions.
"""

import os
import json
import tempfile

import numpy as np
import pytest

from voxcity.io import (
    save_results_h5,
    load_results_h5,
    save_voxcity,
    load_voxcity,
    _serialize_min_heights,
    _deserialize_min_heights,
    _is_attr_serializable,
    _store_attr,
    _store_scalar_attrs,
    _decode_attr,
)
from voxcity.models import (
    GridMetadata,
    VoxelGrid,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    CanopyGrid,
    VoxCity,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_metadata(crs="EPSG:4326", bounds=(0, 0, 10, 10), meshsize=2.0):
    return GridMetadata(crs=crs, bounds=bounds, meshsize=meshsize)


def _make_voxcity(shape=(4, 4, 6), meshsize=2.0, with_canopy=True, with_extras=True):
    """Create a minimal but realistic VoxCity for testing."""
    ny, nx, nz = shape
    meta = _make_metadata(meshsize=meshsize)

    voxels = VoxelGrid(
        classes=np.random.randint(0, 5, shape, dtype=np.int8),
        meta=meta,
    )

    heights = np.random.rand(ny, nx) * 30
    min_heights = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            # Mix of empty, scalar‐like, tuple lists
            if (i + j) % 3 == 0:
                min_heights[i, j] = []
            elif (i + j) % 3 == 1:
                min_heights[i, j] = [(0.0, float(heights[i, j]))]
            else:
                min_heights[i, j] = [float(heights[i, j])]

    buildings = BuildingGrid(
        heights=heights,
        min_heights=min_heights,
        ids=np.arange(ny * nx).reshape(ny, nx).astype(float),
        meta=meta,
    )

    land = LandCoverGrid(classes=np.ones((ny, nx), dtype=int), meta=meta)
    dem = DemGrid(elevation=np.random.rand(ny, nx) * 100, meta=meta)

    canopy = None
    if with_canopy:
        canopy = CanopyGrid(
            top=np.random.rand(ny, nx) * 15,
            bottom=np.random.rand(ny, nx) * 5,
            meta=meta,
        )

    extras = {}
    if with_extras:
        extras = {
            "rectangle_vertices": [(0, 0), (0, 10), (10, 10), (10, 0)],
            "source": "test",
            "count": 42,
        }

    return VoxCity(
        voxels=voxels,
        buildings=buildings,
        land_cover=land,
        dem=dem,
        tree_canopy=canopy,
        extras=extras,
    )


class _FakeMesh:
    """Lightweight stand-in for trimesh.Trimesh (avoids trimesh dependency in tests)."""

    def __init__(self, n_verts=20, n_faces=10, metadata=None):
        self.vertices = np.random.rand(n_verts, 3).astype(np.float64)
        self.faces = np.random.randint(0, n_verts, (n_faces, 3)).astype(np.int32)
        self.face_normals = np.random.rand(n_faces, 3).astype(np.float64)
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# save_results_h5 / load_results_h5 round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadResultsH5:
    """Round-trip tests for HDF5 save/load."""

    def test_model_only(self, tmp_path):
        """Save and load with no simulation results."""
        city = _make_voxcity()
        path = str(tmp_path / "model_only.h5")
        save_results_h5(path, city)
        data = load_results_h5(path)

        assert isinstance(data["voxcity"], VoxCity)
        np.testing.assert_array_equal(data["voxcity"].voxels.classes, city.voxels.classes)
        assert "ground" not in data
        assert "building" not in data

    def test_ground_results_round_trip(self, tmp_path):
        """Ground-level arrays survive the round-trip."""
        city = _make_voxcity()
        ny, nx = city.voxels.classes.shape[:2]
        ground = {
            "cumulative_global": np.random.rand(ny, nx),
            "sunlight_hours": np.random.rand(ny, nx),
            "potential_sunlight_hours": 12.5,
            "mode": "annual",
        }
        path = str(tmp_path / "ground.h5")
        save_results_h5(path, city, ground_results=ground)
        data = load_results_h5(path)

        assert "ground" in data
        np.testing.assert_array_almost_equal(
            data["ground"]["cumulative_global"], ground["cumulative_global"]
        )
        np.testing.assert_array_almost_equal(
            data["ground"]["sunlight_hours"], ground["sunlight_hours"]
        )
        assert data["ground"]["potential_sunlight_hours"] == pytest.approx(12.5)
        assert data["ground"]["mode"] == "annual"

    def test_building_results_round_trip_dict(self, tmp_path):
        """Building results passed as dict with 'mesh' key."""
        city = _make_voxcity()
        n_faces = 30
        mesh = _FakeMesh(n_verts=40, n_faces=n_faces)
        meta = {
            "direct": np.random.rand(n_faces),
            "diffuse": np.random.rand(n_faces),
            "global": np.random.rand(n_faces),
            "sunlight_hours": np.random.rand(n_faces),
            "building_id": np.arange(n_faces, dtype=np.int32),
            "potential_sunlight_hours": 10.0,
        }
        building = {"mesh": mesh, "metadata": meta}
        path = str(tmp_path / "building.h5")
        save_results_h5(path, city, building_results=building)
        data = load_results_h5(path)

        b = data["building"]
        np.testing.assert_array_almost_equal(b["mesh_vertices"], mesh.vertices)
        np.testing.assert_array_equal(b["mesh_faces"], mesh.faces)
        assert "mesh_face_normals" in b
        np.testing.assert_array_almost_equal(b["direct"], meta["direct"])
        np.testing.assert_array_almost_equal(b["diffuse"], meta["diffuse"])
        assert b["potential_sunlight_hours"] == pytest.approx(10.0)

    def test_building_results_bare_mesh(self, tmp_path):
        """Building results passed as a bare mesh object (not a dict)."""
        city = _make_voxcity()
        n_faces = 15
        mesh = _FakeMesh(
            n_verts=25,
            n_faces=n_faces,
            metadata={"direct": np.random.rand(n_faces)},
        )
        path = str(tmp_path / "bare_mesh.h5")
        save_results_h5(path, city, building_results=mesh)
        data = load_results_h5(path)

        b = data["building"]
        np.testing.assert_array_almost_equal(b["mesh_vertices"], mesh.vertices)
        np.testing.assert_array_almost_equal(b["direct"], mesh.metadata["direct"])

    def test_building_results_mesh_metadata_fallback(self, tmp_path):
        """When dict has no 'metadata' key, falls back to mesh.metadata."""
        city = _make_voxcity()
        n_faces = 10
        mesh = _FakeMesh(
            n_verts=20,
            n_faces=n_faces,
            metadata={"irr": np.random.rand(n_faces)},
        )
        building = {"mesh": mesh}  # no 'metadata' key
        path = str(tmp_path / "fallback.h5")
        save_results_h5(path, city, building_results=building)
        data = load_results_h5(path)

        np.testing.assert_array_almost_equal(data["building"]["irr"], mesh.metadata["irr"])

    def test_full_round_trip(self, tmp_path):
        """Full save/load with model + ground + building results."""
        city = _make_voxcity()
        ny, nx = city.voxels.classes.shape[:2]
        ground = {"cumulative_global": np.random.rand(ny, nx)}
        n_faces = 20
        mesh = _FakeMesh(n_verts=30, n_faces=n_faces)
        building = {
            "mesh": mesh,
            "metadata": {"global": np.random.rand(n_faces)},
        }
        path = str(tmp_path / "full.h5")
        save_results_h5(path, city, ground_results=ground, building_results=building)
        data = load_results_h5(path)

        # VoxCity model
        assert isinstance(data["voxcity"], VoxCity)
        np.testing.assert_array_equal(data["voxcity"].voxels.classes, city.voxels.classes)
        # Ground
        np.testing.assert_array_almost_equal(
            data["ground"]["cumulative_global"], ground["cumulative_global"]
        )
        # Building
        np.testing.assert_array_almost_equal(
            data["building"]["global"], building["metadata"]["global"]
        )

    def test_metadata_preserved(self, tmp_path):
        """Root metadata (CRS, meshsize, bounds) is preserved."""
        city = _make_voxcity(meshsize=3.5)
        path = str(tmp_path / "meta.h5")
        save_results_h5(path, city)
        data = load_results_h5(path)

        assert data["meta"]["crs"] == "EPSG:4326"
        assert data["meta"]["meshsize"] == pytest.approx(3.5)
        assert data["meta"]["bounds"] == pytest.approx((0, 0, 10, 10))

    def test_extras_preserved(self, tmp_path):
        """JSON-serializable extras survive the round-trip."""
        city = _make_voxcity(with_extras=True)
        path = str(tmp_path / "extras.h5")
        save_results_h5(path, city)
        data = load_results_h5(path)

        loaded_extras = data["voxcity"].extras
        assert loaded_extras["source"] == "test"
        assert loaded_extras["count"] == 42
        # Tuples become lists via JSON
        assert loaded_extras["rectangle_vertices"] == [
            [0, 0], [0, 10], [10, 10], [10, 0]
        ]


# ---------------------------------------------------------------------------
# VoxCity model fields
# ---------------------------------------------------------------------------

class TestVoxCityFields:
    """Verify all VoxCity sub-grids survive the round-trip."""

    def test_voxel_grid(self, tmp_path):
        city = _make_voxcity()
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_equal(loaded.voxels.classes, city.voxels.classes)

    def test_building_heights(self, tmp_path):
        city = _make_voxcity()
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_almost_equal(loaded.buildings.heights, city.buildings.heights)

    def test_building_ids(self, tmp_path):
        city = _make_voxcity()
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_almost_equal(loaded.buildings.ids, city.buildings.ids)

    def test_dem(self, tmp_path):
        city = _make_voxcity()
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_almost_equal(loaded.dem.elevation, city.dem.elevation)

    def test_land_cover(self, tmp_path):
        city = _make_voxcity()
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_equal(loaded.land_cover.classes, city.land_cover.classes)

    def test_canopy_top_bottom(self, tmp_path):
        city = _make_voxcity(with_canopy=True)
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        np.testing.assert_array_almost_equal(loaded.tree_canopy.top, city.tree_canopy.top)
        np.testing.assert_array_almost_equal(loaded.tree_canopy.bottom, city.tree_canopy.bottom)

    def test_no_canopy(self, tmp_path):
        city = _make_voxcity(with_canopy=False)
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        assert loaded.tree_canopy.top is None

    def test_no_extras(self, tmp_path):
        city = _make_voxcity(with_extras=False)
        path = str(tmp_path / "fields.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]
        assert loaded.extras == {}

    def test_geodataframe_extras_round_trip(self, tmp_path):
        """GeoDataFrame extras survive the HDF5 round-trip via GeoParquet."""
        geopandas = pytest.importorskip("geopandas")
        from shapely.geometry import box

        gdf = geopandas.GeoDataFrame(
            {"building_id": [1, 2, 3], "height": [10.0, 20.0, 30.0]},
            geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
            crs="EPSG:4326",
        )
        city = _make_voxcity(with_extras=False)
        city.extras["building_gdf"] = gdf
        city.extras["source"] = "test"

        path = str(tmp_path / "gdf.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]

        assert "building_gdf" in loaded.extras
        loaded_gdf = loaded.extras["building_gdf"]
        assert isinstance(loaded_gdf, geopandas.GeoDataFrame)
        assert list(loaded_gdf.columns) == list(gdf.columns)
        assert len(loaded_gdf) == 3
        np.testing.assert_array_equal(loaded_gdf["building_id"].values, [1, 2, 3])
        np.testing.assert_array_almost_equal(loaded_gdf["height"].values, [10.0, 20.0, 30.0])
        assert loaded_gdf.crs.to_epsg() == 4326
        # JSON extras should also survive
        assert loaded.extras["source"] == "test"

    def test_numpy_array_extras_round_trip(self, tmp_path):
        """Numpy array extras survive the HDF5 round-trip."""
        arr = np.random.rand(5, 5)
        city = _make_voxcity(with_extras=False)
        city.extras["custom_grid"] = arr
        city.extras["label"] = "custom"

        path = str(tmp_path / "np_extras.h5")
        save_results_h5(path, city)
        loaded = load_results_h5(path)["voxcity"]

        assert "custom_grid" in loaded.extras
        np.testing.assert_array_almost_equal(loaded.extras["custom_grid"], arr)
        assert loaded.extras["label"] == "custom"


# ---------------------------------------------------------------------------
# min_heights serialization
# ---------------------------------------------------------------------------

class TestMinHeightsSerialization:
    """Tests for _serialize_min_heights / _deserialize_min_heights."""

    def test_empty_cells(self):
        arr = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(arr.shape):
            arr[idx] = []
        offsets, values, n_cols = _serialize_min_heights(arr)
        assert len(values) == 0
        assert n_cols == 0
        restored = _deserialize_min_heights(offsets, values, n_cols, arr.shape)
        for idx in np.ndindex(arr.shape):
            assert restored[idx] == []

    def test_scalar_cells(self):
        arr = np.empty((2, 2), dtype=object)
        arr[0, 0] = [5.0]
        arr[0, 1] = [10.0, 20.0]
        arr[1, 0] = []
        arr[1, 1] = [3.0]
        offsets, values, n_cols = _serialize_min_heights(arr)
        restored = _deserialize_min_heights(offsets, values, n_cols, arr.shape)
        assert restored[0, 0] == [5.0]
        assert restored[0, 1] == [10.0, 20.0]
        assert restored[1, 0] == []
        assert restored[1, 1] == [3.0]

    def test_tuple_cells(self):
        arr = np.empty((2, 1), dtype=object)
        arr[0, 0] = [(0.0, 5.0), (5.0, 10.0)]
        arr[1, 0] = [(2.0, 8.0)]
        offsets, values, n_cols = _serialize_min_heights(arr)
        assert n_cols == 2
        restored = _deserialize_min_heights(offsets, values, n_cols, arr.shape)
        assert restored[0, 0] == [(0.0, 5.0), (5.0, 10.0)]
        assert restored[1, 0] == [(2.0, 8.0)]

    def test_none_cells(self):
        arr = np.empty((1, 2), dtype=object)
        arr[0, 0] = None
        arr[0, 1] = [1.0]
        offsets, values, n_cols = _serialize_min_heights(arr)
        restored = _deserialize_min_heights(offsets, values, n_cols, arr.shape)
        assert restored[0, 0] == []
        assert restored[0, 1] == [1.0]

    def test_bare_scalar_cell(self):
        """A cell containing a bare numeric value (not in a list)."""
        arr = np.empty((1, 1), dtype=object)
        arr[0, 0] = 7.5
        offsets, values, n_cols = _serialize_min_heights(arr)
        restored = _deserialize_min_heights(offsets, values, n_cols, arr.shape)
        assert restored[0, 0] == [7.5]


# ---------------------------------------------------------------------------
# HDF5 attribute helpers
# ---------------------------------------------------------------------------

class TestAttrHelpers:
    """Tests for _is_attr_serializable, _store_attr, _decode_attr, etc."""

    def test_is_attr_serializable(self):
        assert _is_attr_serializable(42) is True
        assert _is_attr_serializable(3.14) is True
        assert _is_attr_serializable("hello") is True
        assert _is_attr_serializable(True) is True
        assert _is_attr_serializable([1, 2, 3]) is True
        assert _is_attr_serializable((1, 2)) is True
        assert _is_attr_serializable(np.array([1])) is False
        assert _is_attr_serializable({"a": 1}) is False
        assert _is_attr_serializable(None) is False

    def test_decode_attr_bytes(self):
        assert _decode_attr(b"hello") == "hello"

    def test_decode_attr_np_generic(self):
        assert _decode_attr(np.float64(3.14)) == pytest.approx(3.14)
        assert _decode_attr(np.int32(7)) == 7

    def test_decode_attr_np_array(self):
        result = _decode_attr(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_decode_attr_plain(self):
        assert _decode_attr("hello") == "hello"
        assert _decode_attr(42) == 42

    def test_store_attr_in_h5(self, tmp_path):
        """Integration test: store and read back various attr types via HDF5."""
        import h5py

        path = str(tmp_path / "attrs.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("test")
            _store_attr(g.attrs, "int_val", 42)
            _store_attr(g.attrs, "float_val", 3.14)
            _store_attr(g.attrs, "str_val", "hello")
            _store_attr(g.attrs, "list_val", [1, 2, 3])
            _store_attr(g.attrs, "np_int", np.int64(99))

        with h5py.File(path, "r") as f:
            g = f["test"]
            assert _decode_attr(g.attrs["int_val"]) == 42
            assert _decode_attr(g.attrs["float_val"]) == pytest.approx(3.14)
            assert _decode_attr(g.attrs["str_val"]) == "hello"
            assert _decode_attr(g.attrs["np_int"]) == 99

    def test_store_scalar_attrs(self, tmp_path):
        """_store_scalar_attrs stores only scalars, skipping arrays."""
        import h5py

        meta = {
            "potential_sunlight_hours": 12.5,
            "mode": "annual",
            "big_array": np.zeros(100),
        }
        path = str(tmp_path / "scalar.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("test")
            _store_scalar_attrs(g.attrs, meta)

        with h5py.File(path, "r") as f:
            g = f["test"]
            assert "potential_sunlight_hours" in g.attrs
            assert "mode" in g.attrs
            assert "big_array" not in g.attrs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error conditions."""

    def test_save_non_voxcity_raises_type_error(self, tmp_path):
        path = str(tmp_path / "bad.h5")
        with pytest.raises(TypeError, match="expects a VoxCity instance"):
            save_results_h5(path, {"not": "VoxCity"})

    def test_save_creates_directories(self, tmp_path):
        """Parent directories are created automatically."""
        city = _make_voxcity()
        path = str(tmp_path / "a" / "b" / "c" / "result.h5")
        save_results_h5(path, city)
        assert os.path.exists(path)

    def test_load_nonexistent_file_raises(self, tmp_path):
        path = str(tmp_path / "does_not_exist.h5")
        with pytest.raises((FileNotFoundError, OSError)):
            load_results_h5(path)


# ---------------------------------------------------------------------------
# Backward-compatible import from voxcity.generator.io
# ---------------------------------------------------------------------------

class TestBackwardCompatImport:
    """Ensure the re-export shim in voxcity.generator.io still works."""

    def test_import_save_results_h5(self):
        from voxcity.generator.io import save_results_h5 as fn
        assert callable(fn)

    def test_import_load_results_h5(self):
        from voxcity.generator.io import load_results_h5 as fn
        assert callable(fn)

    def test_import_save_voxcity(self):
        from voxcity.generator.io import save_voxcity as fn
        assert callable(fn)

    def test_import_load_voxcity(self):
        from voxcity.generator.io import load_voxcity as fn
        assert callable(fn)


# ---------------------------------------------------------------------------
# Canonical import from voxcity.io
# ---------------------------------------------------------------------------

class TestCanonicalImport:
    """Ensure voxcity.io exports the expected symbols."""

    def test_save_voxcity_importable(self):
        from voxcity.io import save_voxcity as fn
        assert callable(fn)

    def test_load_voxcity_importable(self):
        from voxcity.io import load_voxcity as fn
        assert callable(fn)

    def test_save_results_h5_importable(self):
        from voxcity.io import save_results_h5 as fn
        assert callable(fn)

    def test_load_results_h5_importable(self):
        from voxcity.io import load_results_h5 as fn
        assert callable(fn)
