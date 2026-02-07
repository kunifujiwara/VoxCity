"""
Tests for voxcity.exporter.cityles helper functions and mappings.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from voxcity.exporter.cityles import (
    VOXCITY_STANDARD_CLASSES,
    OSM_CLASS_TO_CITYLES,
    URBANWATCH_CLASS_TO_CITYLES,
    OEMJ_CLASS_TO_CITYLES,
    ESA_CLASS_TO_CITYLES,
    ESRI_CLASS_TO_CITYLES,
    DYNAMIC_WORLD_CLASS_TO_CITYLES,
    BUILDING_MATERIAL_MAPPING,
    TREE_TYPE_MAPPING,
    create_cityles_directories,
    _get_source_name_mapping,
    _build_index_to_cityles_map,
    _resolve_under_tree_code,
    export_topog,
    export_landuse,
)


class TestMappingDicts:
    def test_standard_classes_keys(self):
        assert 1 in VOXCITY_STANDARD_CLASSES
        assert 14 in VOXCITY_STANDARD_CLASSES
        assert len(VOXCITY_STANDARD_CLASSES) == 14

    def test_osm_mapping_all_values_int(self):
        for v in OSM_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_urbanwatch_mapping_all_values_int(self):
        for v in URBANWATCH_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_oemj_mapping_all_values_int(self):
        for v in OEMJ_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_esa_mapping_all_values_int(self):
        for v in ESA_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_esri_mapping_all_values_int(self):
        for v in ESRI_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_dynamic_world_mapping_all_values_int(self):
        for v in DYNAMIC_WORLD_CLASS_TO_CITYLES.values():
            assert isinstance(v, int)

    def test_building_material_default(self):
        assert 'default' in BUILDING_MATERIAL_MAPPING
        assert BUILDING_MATERIAL_MAPPING['default'] == 110

    def test_tree_type_default(self):
        assert 'default' in TREE_TYPE_MAPPING
        assert TREE_TYPE_MAPPING['default'] == 101


class TestGetSourceNameMapping:
    def test_osm(self):
        mapping = _get_source_name_mapping('OpenStreetMap')
        assert mapping is OSM_CLASS_TO_CITYLES

    def test_standard(self):
        mapping = _get_source_name_mapping('Standard')
        assert mapping is OSM_CLASS_TO_CITYLES

    def test_urbanwatch(self):
        mapping = _get_source_name_mapping('Urbanwatch')
        assert mapping is URBANWATCH_CLASS_TO_CITYLES

    def test_oemj(self):
        mapping = _get_source_name_mapping('OpenEarthMapJapan')
        assert mapping is OEMJ_CLASS_TO_CITYLES

    def test_esa(self):
        mapping = _get_source_name_mapping('ESA WorldCover')
        assert mapping is ESA_CLASS_TO_CITYLES

    def test_esri(self):
        mapping = _get_source_name_mapping('ESRI 10m Annual Land Cover')
        assert mapping is ESRI_CLASS_TO_CITYLES

    def test_dynamic_world(self):
        mapping = _get_source_name_mapping('Dynamic World V1')
        assert mapping is DYNAMIC_WORLD_CLASS_TO_CITYLES

    def test_unknown_defaults_to_osm(self):
        mapping = _get_source_name_mapping('SomeUnknownSource')
        assert mapping is OSM_CLASS_TO_CITYLES


class TestResolveUnderTreeCode:
    def test_explicit_code(self):
        code = _resolve_under_tree_code('Bareland', 42, 'OpenStreetMap')
        assert code == 42

    def test_from_class_name(self):
        code = _resolve_under_tree_code('Bareland', None, 'OpenStreetMap')
        assert code == OSM_CLASS_TO_CITYLES['Bareland']

    def test_fallback_to_osm(self):
        code = _resolve_under_tree_code('Water', None, 'OpenStreetMap')
        assert code == 1

    def test_unknown_class_defaults_to_9(self):
        code = _resolve_under_tree_code('NonExistentClass', None, 'OpenStreetMap')
        assert code == 9

    def test_invalid_explicit_code(self):
        code = _resolve_under_tree_code('Bareland', 'not_a_number', 'OpenStreetMap')
        assert isinstance(code, int)


class TestCreateCitylesDirectories:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'cityles_test')
            result = create_cityles_directories(out)
            assert result.exists()
            assert result.is_dir()

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_cityles_directories(tmpdir)
            assert result.exists()


class TestExportTopog:
    def test_basic_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            building_height = np.array([[0.0, 10.0], [5.0, 0.0]])
            building_id = np.array([[0, 1], [2, 0]])
            output_path = Path(tmpdir)

            export_topog(building_height, building_id, output_path)

            topog_file = output_path / 'topog.txt'
            assert topog_file.exists()

            content = topog_file.read_text()
            lines = content.strip().split('\n')
            # First line is count of building cells
            n_buildings = int(lines[0])
            assert n_buildings == 2  # Two cells with height > 0

    def test_all_buildings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            building_height = np.ones((3, 3)) * 15.0
            building_id = np.arange(9).reshape(3, 3)
            output_path = Path(tmpdir)

            export_topog(building_height, building_id, output_path)

            content = (output_path / 'topog.txt').read_text()
            lines = content.strip().split('\n')
            n_buildings = int(lines[0])
            assert n_buildings == 9

    def test_no_buildings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            building_height = np.zeros((2, 2))
            building_id = np.zeros((2, 2))
            output_path = Path(tmpdir)

            export_topog(building_height, building_id, output_path)

            content = (output_path / 'topog.txt').read_text()
            lines = content.strip().split('\n')
            n_buildings = int(lines[0])
            assert n_buildings == 0

    def test_custom_material(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            building_height = np.array([[10.0]])
            building_id = np.array([[1]])
            output_path = Path(tmpdir)

            export_topog(building_height, building_id, output_path,
                         building_material='residential')

            content = (output_path / 'topog.txt').read_text()
            assert '111' in content  # residential material code


class TestExportLanduse:
    def test_basic_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lc_grid = np.zeros((3, 3), dtype=int)
            output_path = Path(tmpdir)

            export_landuse(lc_grid, output_path, land_cover_source='OpenStreetMap')

            landuse_file = output_path / 'landuse.txt'
            assert landuse_file.exists()
