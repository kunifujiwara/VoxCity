"""
Tests for voxcity.exporter.magicavoxel helper functions.
"""

import numpy as np
import pytest
import tempfile
import os

from voxcity.exporter.magicavoxel import (
    convert_colormap_and_array,
    create_custom_palette,
    create_mapping,
    split_array,
    numpy_to_vox,
)


class TestConvertColormapAndArray:
    def test_basic_conversion(self):
        color_map = {5: [255, 0, 0], 10: [0, 255, 0]}
        array = np.array([[[5, 10], [10, 5]]])
        new_map, new_array = convert_colormap_and_array(color_map, array)

        assert 0 in new_map
        assert 1 in new_map
        assert new_map[0] == [255, 0, 0]
        assert new_map[1] == [0, 255, 0]
        assert new_array[0, 0, 0] == 0
        assert new_array[0, 0, 1] == 1

    def test_sequential_keys(self):
        color_map = {0: [100, 100, 100], 1: [200, 200, 200]}
        array = np.array([[[0, 1]]])
        new_map, new_array = convert_colormap_and_array(color_map, array)

        assert new_map == color_map
        np.testing.assert_array_equal(new_array, array)

    def test_single_color(self):
        color_map = {7: [128, 128, 128]}
        array = np.array([[[7, 7, 7]]])
        new_map, new_array = convert_colormap_and_array(color_map, array)

        assert 0 in new_map
        assert np.all(new_array == 0)

    def test_preserves_original(self):
        color_map = {3: [255, 0, 0]}
        array = np.array([[[3]]])
        original = array.copy()
        new_map, new_array = convert_colormap_and_array(color_map, array)
        np.testing.assert_array_equal(array, original)


class TestCreateCustomPalette:
    def test_shape(self):
        color_map = {0: [255, 0, 0], 1: [0, 255, 0]}
        palette = create_custom_palette(color_map)
        assert palette.shape == (256, 4)
        assert palette.dtype == np.uint8

    def test_transparent_first(self):
        color_map = {0: [100, 100, 100]}
        palette = create_custom_palette(color_map)
        np.testing.assert_array_equal(palette[0], [0, 0, 0, 0])

    def test_colors_start_at_1(self):
        color_map = {0: [255, 0, 0], 1: [0, 255, 0]}
        palette = create_custom_palette(color_map)
        assert palette[1, 0] == 255  # R
        assert palette[1, 1] == 0    # G
        assert palette[2, 1] == 255  # G

    def test_alpha_255(self):
        color_map = {0: [128, 128, 128]}
        palette = create_custom_palette(color_map)
        # All non-zero indices should have alpha 255
        assert palette[1, 3] == 255
        assert palette[100, 3] == 255

    def test_empty_colormap(self):
        palette = create_custom_palette({})
        assert palette.shape == (256, 4)
        np.testing.assert_array_equal(palette[0], [0, 0, 0, 0])


class TestCreateMapping:
    def test_basic(self):
        color_map = {5: [255, 0, 0], 10: [0, 255, 0], 15: [0, 0, 255]}
        mapping = create_mapping(color_map)
        assert mapping[5] == 2
        assert mapping[10] == 3
        assert mapping[15] == 4

    def test_single_entry(self):
        color_map = {0: [128, 128, 128]}
        mapping = create_mapping(color_map)
        assert mapping[0] == 2

    def test_starts_at_2(self):
        color_map = {1: [0, 0, 0], 2: [255, 255, 255]}
        mapping = create_mapping(color_map)
        values = sorted(mapping.values())
        assert values[0] == 2


class TestSplitArray:
    def test_small_array_no_split(self):
        array = np.ones((100, 100, 100))
        chunks = list(split_array(array, max_size=255))
        assert len(chunks) == 1
        assert chunks[0][1] == (0, 0, 0)
        assert chunks[0][0].shape == (100, 100, 100)

    def test_exact_split(self):
        array = np.ones((510, 255, 255))
        chunks = list(split_array(array, max_size=255))
        assert len(chunks) == 2

    def test_multiple_splits(self):
        array = np.ones((300, 300, 300))
        chunks = list(split_array(array, max_size=255))
        assert len(chunks) == 8  # 2*2*2

    def test_chunk_positions(self):
        array = np.ones((510, 510, 255))
        chunks = list(split_array(array, max_size=255))
        positions = [c[1] for c in chunks]
        assert (0, 0, 0) in positions
        assert (1, 0, 0) in positions
        assert (0, 1, 0) in positions
        assert (1, 1, 0) in positions

    def test_small_max_size(self):
        array = np.ones((10, 10, 10))
        chunks = list(split_array(array, max_size=5))
        assert len(chunks) == 8  # 2*2*2

    def test_single_element(self):
        array = np.ones((1, 1, 1))
        chunks = list(split_array(array, max_size=255))
        assert len(chunks) == 1

    def test_all_chunks_within_size(self):
        array = np.ones((300, 300, 300))
        for chunk, pos in split_array(array, max_size=255):
            assert chunk.shape[0] <= 255
            assert chunk.shape[1] <= 255
            assert chunk.shape[2] <= 255


class TestNumpyToVoxAxisOrder:
    """Verify VoxCity (north, east, height) maps to pyvox dense (y=north, z=height, x=east)."""

    def test_voxcity_axes_map_to_pyvox_dense_y_z_x(self, monkeypatch, tmp_path):
        from pyvox.models import Vox

        captured = {}
        original_from_dense = Vox.from_dense

        def capture_from_dense(dense):
            captured["dense"] = dense.copy()
            return original_from_dense(dense)

        monkeypatch.setattr(Vox, "from_dense", staticmethod(capture_from_dense))

        array = np.zeros((3, 4, 5), dtype=np.uint8)  # (north=3, east=4, height=5)
        array[2, 1, 3] = 1

        numpy_to_vox(array, {1: [255, 0, 0]}, str(tmp_path / "axis.vox"))

        dense = captured["dense"]
        assert dense.shape == (3, 5, 4)  # pyvox dense: (y=north, z=height, x=east)
        # Marker at north=2, flipped_height=5-3-1=1, east=1
        assert dense[2, 1, 1] == 2  # create_mapping maps value 1 -> palette index 2

    def test_returns_magicavoxel_model_size_x_y_z(self, tmp_path):
        array = np.zeros((3, 4, 5), dtype=np.uint8)  # (north, east, height)
        array[2, 1, 3] = 1

        _, _, shape = numpy_to_vox(array, {1: [255, 0, 0]}, str(tmp_path / "axis.vox"))

        assert shape == (4, 3, 5)  # MagicaVoxel model size: (x=east, y=north, z=height)
