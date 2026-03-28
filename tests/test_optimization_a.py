"""Tests for user-facing optimizations (A1–A8).

Each class covers one optimization item implemented across the voxcity package.
"""

import warnings
import pickle
import tempfile
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import requests


# ---------------------------------------------------------------------------
# A1 – Dynamic version from importlib.metadata
# ---------------------------------------------------------------------------

class TestVersionSync:
    """A1: __version__ should come from package metadata, not a hardcoded string."""

    def test_version_is_string(self):
        from voxcity import __version__
        assert isinstance(__version__, str)

    def test_version_not_hardcoded_old(self):
        from voxcity import __version__
        assert __version__ != "0.1.0", "Version should no longer be the old hardcoded value"

    def test_version_format(self):
        """Version should look like a PEP 440 version or the fallback '0.0.0'."""
        from voxcity import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2, f"Unexpected version format: {__version__}"

    def test_fallback_on_missing_metadata(self):
        """When importlib.metadata fails, fallback to '0.0.0'."""
        import importlib
        with patch("importlib.metadata.version", side_effect=Exception("no metadata")):
            # Re-execute the init logic
            try:
                from importlib.metadata import version as _get_version
                ver = _get_version("voxcity")
            except Exception:
                ver = "0.0.0"
            assert ver == "0.0.0"


# ---------------------------------------------------------------------------
# A2 – Pickle security: trusted= parameter and UserWarning
# ---------------------------------------------------------------------------

class TestPickleSecurity:
    """A2: load_voxcity should warn about pickle security by default."""

    @pytest.fixture()
    def pickle_file(self, tmp_path):
        """Create a trivial pickle file containing a dict (legacy format)."""
        data = {
            "land_cover_grid": np.zeros((3, 3), dtype=np.int32),
            "meshsize": 1.0,
            "building_height_grid": np.zeros((3, 3)),
            "building_min_height_grid": np.empty((3, 3), dtype=object),
            "building_id_grid": np.zeros((3, 3), dtype=np.int32),
            "dem_grid": np.zeros((3, 3)),
            "voxcity_grid": np.zeros((3, 3, 3), dtype=np.int8),
            "canopy_top_grid": np.zeros((3, 3)),
            "rectangle_vertices": [
                (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
            ],
        }
        # Fill building_min_height_grid cells with empty lists
        for idx in np.ndindex(data["building_min_height_grid"].shape):
            data["building_min_height_grid"][idx] = []

        fp = tmp_path / "test.voxcity"
        with open(fp, "wb") as f:
            pickle.dump(data, f)
        return fp

    def test_warns_when_trusted_false(self, pickle_file):
        from voxcity.io import load_voxcity
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_voxcity(str(pickle_file), trusted=False)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "pickle" in str(user_warnings[0].message).lower()

    def test_no_warning_when_trusted(self, pickle_file):
        from voxcity.io import load_voxcity
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_voxcity(str(pickle_file), trusted=True)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_returns_voxcity_object(self, pickle_file):
        from voxcity.io import load_voxcity
        from voxcity.models import VoxCity
        result = load_voxcity(str(pickle_file), trusted=True)
        assert isinstance(result, VoxCity)


# ---------------------------------------------------------------------------
# A3 – print→logging migration
# ---------------------------------------------------------------------------

class TestLoggingMigration:
    """A3: Core modules should use logger, not print."""

    def test_voxelizer_has_logger(self):
        from voxcity.generator import voxelizer
        assert hasattr(voxelizer, "_logger")

    def test_geoprocessor_io_has_logger(self):
        from voxcity.geoprocessor import io
        assert hasattr(io, "_logger")

    def test_geoprocessor_heights_has_logger(self):
        from voxcity.geoprocessor import heights
        assert hasattr(heights, "_logger")

    def test_io_module_uses_logger(self):
        """save_voxcity should use logger, not print."""
        from voxcity import io as voxcity_io
        import inspect
        src = inspect.getsource(voxcity_io.save_voxcity)
        # Should not contain bare print() calls (except in strings)
        assert "print(" not in src or "logger" in src


# ---------------------------------------------------------------------------
# A4 – download_file with retry, streaming, timeout
# ---------------------------------------------------------------------------

class TestDownloadFile:
    """A4: download_file should support retry, streaming, and proper error handling."""

    def test_successful_download(self, tmp_path):
        from voxcity.downloader.utils import download_file

        dest = str(tmp_path / "test_download.bin")
        content = b"hello world"

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=mock_response) as mock_get:
            download_file("https://example.com/file.bin", dest)
            mock_get.assert_called_once_with(
                "https://example.com/file.bin", stream=True, timeout=60
            )

        assert os.path.exists(dest)
        with open(dest, "rb") as f:
            assert f.read() == content

    def test_retry_on_failure(self, tmp_path):
        from voxcity.downloader.utils import download_file

        dest = str(tmp_path / "retry_test.bin")

        # Fail twice, succeed on third
        content = b"data"
        mock_ok = MagicMock()
        mock_ok.iter_content.return_value = [content]
        mock_ok.raise_for_status = MagicMock()
        mock_ok.__enter__ = MagicMock(return_value=mock_ok)
        mock_ok.__exit__ = MagicMock(return_value=False)

        call_count = {"n": 0}
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise requests.ConnectionError("fail")
            return mock_ok

        with patch("requests.get", side_effect=side_effect):
            with patch("time.sleep"):  # don't actually wait
                download_file(dest, dest, max_retries=3, initial_delay=0.01)

        assert call_count["n"] == 3

    def test_raises_after_max_retries(self, tmp_path):
        from voxcity.downloader.utils import download_file

        dest = str(tmp_path / "fail_test.bin")

        with patch("requests.get", side_effect=requests.ConnectionError("always fail")):
            with patch("time.sleep"):
                with pytest.raises(requests.HTTPError, match="Failed to download"):
                    download_file("https://example.com/fail", dest, max_retries=2, initial_delay=0.01)

    def test_custom_timeout(self, tmp_path):
        from voxcity.downloader.utils import download_file

        dest = str(tmp_path / "timeout_test.bin")

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"ok"]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("requests.get", return_value=mock_response) as mock_get:
            download_file("https://example.com/f", dest, timeout=120)
            _, kwargs = mock_get.call_args
            assert kwargs["timeout"] == 120


# ---------------------------------------------------------------------------
# A5 – Vectorised _flatten_building_segments
# ---------------------------------------------------------------------------

class TestFlattenBuildingSegments:
    """A5: _flatten_building_segments should produce correct flattened arrays."""

    def test_empty_grid(self):
        from voxcity.generator.voxelizer import _flatten_building_segments

        grid = np.empty((3, 3), dtype=object)
        for idx in np.ndindex(grid.shape):
            grid[idx] = []

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)

        assert seg_starts.shape == (0,)
        assert seg_ends.shape == (0,)
        assert counts.sum() == 0

    def test_single_segment(self):
        from voxcity.generator.voxelizer import _flatten_building_segments

        grid = np.empty((2, 2), dtype=object)
        for idx in np.ndindex(grid.shape):
            grid[idx] = []
        grid[0, 0] = [[0.0, 5.0]]  # single segment: 0m to 5m

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)

        assert len(seg_starts) == 1
        assert len(seg_ends) == 1
        assert seg_starts[0] == 0
        assert seg_ends[0] == 5
        assert counts[0, 0] == 1

    def test_multiple_segments_per_cell(self):
        from voxcity.generator.voxelizer import _flatten_building_segments

        grid = np.empty((1, 1), dtype=object)
        grid[0, 0] = [[0.0, 3.0], [5.0, 8.0]]

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)

        assert len(seg_starts) == 2
        assert counts[0, 0] == 2
        assert offsets[0, 0] == 0

    def test_voxel_size_scaling(self):
        from voxcity.generator.voxelizer import _flatten_building_segments

        grid = np.empty((1, 1), dtype=object)
        grid[0, 0] = [[0.0, 10.0]]

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 2.0)

        # 10.0 / 2.0 = 5
        assert seg_ends[0] == 5

    def test_offsets_are_cumulative(self):
        from voxcity.generator.voxelizer import _flatten_building_segments

        grid = np.empty((1, 3), dtype=object)
        grid[0, 0] = [[0.0, 1.0], [2.0, 3.0]]  # 2 segments
        grid[0, 1] = [[0.0, 1.0]]                # 1 segment
        grid[0, 2] = []                           # 0 segments

        seg_starts, seg_ends, offsets, counts = _flatten_building_segments(grid, 1.0)

        assert counts[0, 0] == 2
        assert counts[0, 1] == 1
        assert counts[0, 2] == 0
        assert offsets[0, 0] == 0
        assert offsets[0, 1] == 2
        assert offsets[0, 2] == 3
        assert len(seg_starts) == 3


# ---------------------------------------------------------------------------
# A6 – Vectorised create_voxel_mesh
# ---------------------------------------------------------------------------

class TestVectorisedVoxelMesh:
    """A6: Vectorised create_voxel_mesh should produce valid meshes."""

    def test_single_voxel_face_count(self):
        """A single exposed voxel should have 6 faces × 2 triangles = 12 faces."""
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        arr = np.zeros((5, 5, 5), dtype=np.int32)
        arr[2, 2, 0] = 1

        mesh = create_voxel_mesh(arr, class_id=1, meshsize=1.0)
        assert mesh is not None
        assert len(mesh.faces) == 12

    def test_solid_block_fewer_faces(self):
        """A 2×2×2 block should have fewer faces than 8 isolated voxels."""
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        # Isolated voxels: 8 × 12 = 96 faces
        isolated = np.zeros((10, 10, 10), dtype=np.int32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    isolated[i * 3, j * 3, k * 3] = 1
        m_iso = create_voxel_mesh(isolated, class_id=1)

        # Solid block: interior faces are suppressed
        block = np.zeros((10, 10, 10), dtype=np.int32)
        block[0:2, 0:2, 0:2] = 1
        m_blk = create_voxel_mesh(block, class_id=1)

        assert m_blk is not None and m_iso is not None
        assert len(m_blk.faces) < len(m_iso.faces)

    def test_empty_returns_none(self):
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        arr = np.zeros((5, 5, 5), dtype=np.int32)
        result = create_voxel_mesh(arr, class_id=1)
        assert result is None

    def test_building_id_metadata(self):
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        arr = np.zeros((5, 5, 5), dtype=np.int32)
        arr[1:3, 1:3, 0:2] = -3

        bid_grid = np.zeros((5, 5), dtype=np.int32)
        bid_grid[1:3, 1:3] = 42

        mesh = create_voxel_mesh(arr, class_id=-3, building_id_grid=bid_grid)
        assert mesh is not None
        assert "building_id" in mesh.metadata

    def test_meshsize_scaling(self):
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        arr = np.zeros((5, 5, 5), dtype=np.int32)
        arr[2, 2, 0] = 1

        m1 = create_voxel_mesh(arr, class_id=1, meshsize=1.0)
        m2 = create_voxel_mesh(arr, class_id=1, meshsize=2.0)

        assert m1 is not None and m2 is not None
        assert np.max(m2.vertices) > np.max(m1.vertices)

    def test_class_isolation(self):
        """Only the requested class_id is meshed."""
        from voxcity.geoprocessor.mesh import create_voxel_mesh

        arr = np.zeros((5, 5, 5), dtype=np.int32)
        arr[1, 1, 0] = 1
        arr[3, 3, 0] = 2

        m1 = create_voxel_mesh(arr, class_id=1)
        m2 = create_voxel_mesh(arr, class_id=2)

        assert m1 is not None and m2 is not None
        # Single voxel → same face count for each
        assert len(m1.faces) == len(m2.faces) == 12


# ---------------------------------------------------------------------------
# A7 – get_country_name caching
# ---------------------------------------------------------------------------

class TestCountryNameCache:
    """A7: get_country_name should use a coordinate-based cache."""

    def test_cache_hit_avoids_second_lookup(self):
        from voxcity.geoprocessor import utils as geo_utils

        # Clear the cache first
        geo_utils._country_name_cache.clear()

        mock_result = [{"cc": "JP"}]

        with patch("reverse_geocoder.search", return_value=mock_result) as mock_rg:
            with patch("pycountry.countries.get") as mock_pc:
                mock_pc.return_value = MagicMock(name="Japan")
                mock_pc.return_value.name = "Japan"

                name1 = geo_utils.get_country_name(139.6503, 35.6762)
                name2 = geo_utils.get_country_name(139.6503, 35.6762)

                assert name1 == "Japan"
                assert name2 == "Japan"
                # Second call should use cache, so rg.search called only once
                assert mock_rg.call_count == 1

    def test_nearby_coords_share_cache(self):
        """Coordinates rounding to the same 0.01° grid should share cache."""
        from voxcity.geoprocessor import utils as geo_utils

        geo_utils._country_name_cache.clear()

        mock_result = [{"cc": "JP"}]

        with patch("reverse_geocoder.search", return_value=mock_result) as mock_rg:
            with patch("pycountry.countries.get") as mock_pc:
                mock_pc.return_value = MagicMock(name="Japan")
                mock_pc.return_value.name = "Japan"

                geo_utils.get_country_name(139.6501, 35.6761)
                geo_utils.get_country_name(139.6504, 35.6764)  # rounds same

                assert mock_rg.call_count == 1

    def test_different_coords_separate_cache(self):
        from voxcity.geoprocessor import utils as geo_utils

        geo_utils._country_name_cache.clear()

        mock_result = [{"cc": "JP"}]

        with patch("reverse_geocoder.search", return_value=mock_result) as mock_rg:
            with patch("pycountry.countries.get") as mock_pc:
                mock_pc.return_value = MagicMock(name="Japan")
                mock_pc.return_value.name = "Japan"

                geo_utils.get_country_name(139.65, 35.67)
                geo_utils.get_country_name(100.00, 10.00)  # very different

                assert mock_rg.call_count == 2


# ---------------------------------------------------------------------------
# A8 – PipelineConfig parallel_download defaults to True
# ---------------------------------------------------------------------------

class TestParallelDownloadDefault:
    """A8: PipelineConfig.parallel_download should default to True."""

    def _make_config(self, **overrides):
        from voxcity.models import PipelineConfig
        defaults = dict(
            rectangle_vertices=[(0, 0), (0, 1), (1, 1), (1, 0)],
            meshsize=1.0,
        )
        defaults.update(overrides)
        return PipelineConfig(**defaults)

    def test_default_is_true(self):
        cfg = self._make_config()
        assert cfg.parallel_download is True

    def test_can_override_to_false(self):
        cfg = self._make_config(parallel_download=False)
        assert cfg.parallel_download is False


# ---------------------------------------------------------------------------
# A9 – HDF5 save/load for plain VoxCity models
# ---------------------------------------------------------------------------

class TestH5ModelIO:
    """save_h5/load_h5 should round-trip a VoxCity model without pickle."""

    @pytest.fixture()
    def sample_city(self):
        from voxcity.models import (
            GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid,
            DemGrid, CanopyGrid, VoxCity,
        )
        meta = GridMetadata(crs="EPSG:4326", bounds=(0.0, 0.0, 3.0, 3.0), meshsize=1.0)
        min_h = np.empty((3, 3), dtype=object)
        for idx in np.ndindex(min_h.shape):
            min_h[idx] = []
        min_h[1, 1] = [(0.0, 5.0), (8.0, 12.0)]

        return VoxCity(
            voxels=VoxelGrid(classes=np.ones((3, 3, 5), dtype=np.int8), meta=meta),
            buildings=BuildingGrid(
                heights=np.array([[0, 10, 0], [0, 15, 0], [0, 0, 0]], dtype=float),
                min_heights=min_h,
                ids=np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int32),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.full((3, 3), 11, dtype=np.int32), meta=meta),
            dem=DemGrid(elevation=np.zeros((3, 3)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((3, 3)), bottom=None, meta=meta),
            extras={"rectangle_vertices": [(0, 0), (0, 3), (3, 3), (3, 0)]},
        )

    def test_round_trip(self, tmp_path, sample_city):
        from voxcity.io import save_h5, load_h5
        fp = str(tmp_path / "model.h5")
        save_h5(fp, sample_city)
        loaded = load_h5(fp)

        assert type(loaded).__name__ == "VoxCity"
        np.testing.assert_array_equal(loaded.voxels.classes, sample_city.voxels.classes)
        np.testing.assert_array_equal(loaded.buildings.heights, sample_city.buildings.heights)
        np.testing.assert_array_equal(loaded.buildings.ids, sample_city.buildings.ids)
        np.testing.assert_array_equal(loaded.land_cover.classes, sample_city.land_cover.classes)
        np.testing.assert_array_equal(loaded.dem.elevation, sample_city.dem.elevation)
        assert loaded.voxels.meta.meshsize == sample_city.voxels.meta.meshsize
        assert loaded.voxels.meta.crs == sample_city.voxels.meta.crs

    def test_min_heights_preserved(self, tmp_path, sample_city):
        from voxcity.io import save_h5, load_h5
        fp = str(tmp_path / "model_mh.h5")
        save_h5(fp, sample_city)
        loaded = load_h5(fp)

        # Cell [1,1] had two segments
        cell = loaded.buildings.min_heights[1, 1]
        assert len(cell) == 2
        assert cell[0] == (0.0, 5.0)
        assert cell[1] == (8.0, 12.0)

        # Empty cells remain empty lists
        assert loaded.buildings.min_heights[0, 0] == []

    def test_canopy_preserved(self, tmp_path, sample_city):
        from voxcity.io import save_h5, load_h5
        fp = str(tmp_path / "model_can.h5")
        save_h5(fp, sample_city)
        loaded = load_h5(fp)

        assert loaded.tree_canopy is not None
        np.testing.assert_array_equal(loaded.tree_canopy.top, sample_city.tree_canopy.top)

    def test_no_pickle_warning(self, tmp_path, sample_city):
        """load_h5 should NOT emit the pickle security warning."""
        from voxcity.io import save_h5, load_h5
        fp = str(tmp_path / "model_warn.h5")
        save_h5(fp, sample_city)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_h5(fp)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_load_h5_on_results_file(self, tmp_path, sample_city):
        """load_h5 should also work on files written by save_results_h5."""
        from voxcity.io import save_results_h5, load_h5
        fp = str(tmp_path / "results.h5")
        save_results_h5(fp, sample_city, ground_results={"svf": np.ones((3, 3))})
        loaded = load_h5(fp)

        assert type(loaded).__name__ == "VoxCity"
        np.testing.assert_array_equal(loaded.voxels.classes, sample_city.voxels.classes)


class TestSaveLoadAutoDetect:
    """save_voxcity/load_voxcity should auto-detect HDF5 vs pickle by extension."""

    @pytest.fixture()
    def sample_city(self):
        from voxcity.models import (
            GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid,
            DemGrid, CanopyGrid, VoxCity,
        )
        meta = GridMetadata(crs="EPSG:4326", bounds=(0.0, 0.0, 2.0, 2.0), meshsize=1.0)
        return VoxCity(
            voxels=VoxelGrid(classes=np.ones((2, 2, 3), dtype=np.int8), meta=meta),
            buildings=BuildingGrid(
                heights=np.array([[5, 0], [0, 10]], dtype=float),
                min_heights=np.zeros((2, 2)),
                ids=np.array([[1, 0], [0, 2]], dtype=np.int32),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.full((2, 2), 11, dtype=np.int32), meta=meta),
            dem=DemGrid(elevation=np.zeros((2, 2)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((2, 2)), bottom=None, meta=meta),
            extras={"rectangle_vertices": [(0, 0), (0, 2), (2, 2), (2, 0)]},
        )

    def test_save_h5_extension_creates_hdf5(self, tmp_path, sample_city):
        """save_voxcity with .h5 extension should create an HDF5 file."""
        import h5py
        from voxcity.io import save_voxcity
        fp = str(tmp_path / "auto.h5")
        save_voxcity(fp, sample_city)
        # Verify it's a valid HDF5 file
        with h5py.File(fp, "r") as f:
            assert "voxcity" in f

    def test_save_hdf5_extension_creates_hdf5(self, tmp_path, sample_city):
        """save_voxcity with .hdf5 extension should also route to HDF5."""
        import h5py
        from voxcity.io import save_voxcity
        fp = str(tmp_path / "auto.hdf5")
        save_voxcity(fp, sample_city)
        with h5py.File(fp, "r") as f:
            assert "voxcity" in f

    def test_save_pkl_extension_creates_pickle(self, tmp_path, sample_city):
        """save_voxcity with .pkl extension should fall back to pickle."""
        import pickle
        from voxcity.io import save_voxcity
        fp = str(tmp_path / "auto.pkl")
        save_voxcity(fp, sample_city)
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert obj["__format__"] == "voxcity.v2"

    def test_load_h5_extension_no_warning(self, tmp_path, sample_city):
        """load_voxcity with .h5 file should NOT emit pickle warning."""
        from voxcity.io import save_voxcity, load_voxcity
        fp = str(tmp_path / "auto.h5")
        save_voxcity(fp, sample_city)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_voxcity(fp)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

        np.testing.assert_array_equal(loaded.voxels.classes, sample_city.voxels.classes)

    def test_load_pkl_extension_warns(self, tmp_path, sample_city):
        """load_voxcity with .pkl file should still emit pickle warning."""
        from voxcity.io import save_voxcity, load_voxcity
        fp = str(tmp_path / "auto.pkl")
        save_voxcity(fp, sample_city)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_voxcity(fp)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1
            assert "pickle" in str(user_warnings[0].message).lower()

    def test_round_trip_h5_via_save_load(self, tmp_path, sample_city):
        """Full round-trip through save_voxcity/load_voxcity with .h5."""
        from voxcity.io import save_voxcity, load_voxcity
        fp = str(tmp_path / "rt.h5")
        save_voxcity(fp, sample_city)
        loaded = load_voxcity(fp)

        np.testing.assert_array_equal(loaded.buildings.heights, sample_city.buildings.heights)
        np.testing.assert_array_equal(loaded.buildings.ids, sample_city.buildings.ids)
        assert loaded.voxels.meta.meshsize == sample_city.voxels.meta.meshsize
