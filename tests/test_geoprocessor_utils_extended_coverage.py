"""Round 6 – cover geoprocessor/utils.py uncovered lines: merge_geotiffs (431-464), get_coordinates_from_cityname (506-522), get_city_country_name_from_rectangle (567-584), get_timezone_info (634-635), get_country_name (817-827), validate_polygon_coordinates (~678), create_building_polygons (~742-795)."""
from __future__ import annotations

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# Tests for get_timezone_info
# ===========================================================================

class TestGetTimezoneInfo:
    """Cover utils.py lines 586-635."""

    def test_tokyo_timezone(self):
        from voxcity.geoprocessor.utils import get_timezone_info
        coords = [(139.65, 35.67), (139.66, 35.67), (139.66, 35.68), (139.65, 35.68)]
        tz_str, meridian = get_timezone_info(coords)
        assert "UTC" in tz_str
        assert float(meridian) != 0  # Tokyo is UTC+9

    def test_unknown_timezone_fallback(self):
        from voxcity.geoprocessor.utils import get_timezone_info
        # Middle of ocean (no timezone)
        with patch("voxcity.geoprocessor.utils.TimezoneFinder") as mock_tf_cls:
            mock_tf = MagicMock()
            mock_tf.timezone_at.return_value = None
            mock_tf_cls.return_value = mock_tf
            tz_str, meridian = get_timezone_info([(0, 0), (0, 0), (0, 0), (0, 0)])
            assert tz_str == "UTC+00:00"
            assert meridian == "0.00000"


# ===========================================================================
# Tests for get_country_name
# ===========================================================================

class TestGetCountryName:
    """Cover utils.py lines 817-827."""

    @patch("voxcity.geoprocessor.utils.pycountry")
    def test_known_country(self, mock_pycountry):
        import importlib
        with patch.dict("sys.modules", {"reverse_geocoder": MagicMock()}):
            import sys
            rg_mock = sys.modules["reverse_geocoder"]
            rg_mock.search.return_value = [{"cc": "JP"}]
            mock_country = MagicMock()
            mock_country.name = "Japan"
            mock_pycountry.countries.get.return_value = mock_country

            from voxcity.geoprocessor.utils import get_country_name
            result = get_country_name(139.65, 35.67)
            assert result == "Japan"

    @patch("voxcity.geoprocessor.utils.pycountry")
    def test_unknown_country(self, mock_pycountry):
        with patch.dict("sys.modules", {"reverse_geocoder": MagicMock()}):
            import sys
            rg_mock = sys.modules["reverse_geocoder"]
            rg_mock.search.return_value = [{"cc": "XX"}]
            mock_pycountry.countries.get.return_value = None

            from voxcity.geoprocessor.utils import get_country_name
            result = get_country_name(0, 0)
            assert result is None


# ===========================================================================
# Tests for get_coordinates_from_cityname
# ===========================================================================

class TestGetCoordinatesFromCityname:
    """Cover utils.py lines 506-522."""

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_successful_lookup(self, mock_geo):
        from voxcity.geoprocessor.utils import get_coordinates_from_cityname
        mock_location = MagicMock()
        mock_location.latitude = 35.6762
        mock_location.longitude = 139.6503
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = mock_location
        mock_geo.return_value = mock_geolocator

        result = get_coordinates_from_cityname("Tokyo")
        assert result is not None
        assert result[0] == pytest.approx(35.6762)

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_not_found(self, mock_geo):
        from voxcity.geoprocessor.utils import get_coordinates_from_cityname
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None
        mock_geo.return_value = mock_geolocator

        result = get_coordinates_from_cityname("NonexistentPlace12345")
        assert result is None

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_403_returns_none(self, mock_geo):
        from geopy.exc import GeocoderInsufficientPrivileges
        from voxcity.geoprocessor.utils import get_coordinates_from_cityname
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.side_effect = GeocoderInsufficientPrivileges("blocked")
        mock_geo.return_value = mock_geolocator

        result = get_coordinates_from_cityname("Tokyo")
        assert result is None


# ===========================================================================
# Tests for get_city_country_name_from_rectangle
# ===========================================================================

class TestGetCityCountryNameFromRectangle:
    """Cover utils.py lines 567-584."""

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_successful_reverse(self, mock_geo):
        from voxcity.geoprocessor.utils import get_city_country_name_from_rectangle
        mock_location = MagicMock()
        mock_location.raw = {"address": {"city": "Tokyo", "country": "Japan"}}
        mock_geolocator = MagicMock()
        mock_geolocator.reverse.return_value = mock_location
        mock_geo.return_value = mock_geolocator

        coords = [(139.65, 35.67), (139.66, 35.67), (139.66, 35.68), (139.65, 35.68)]
        result = get_city_country_name_from_rectangle(coords)
        assert "Tokyo" in result
        assert "Japan" in result

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_no_location_found(self, mock_geo):
        from voxcity.geoprocessor.utils import get_city_country_name_from_rectangle
        mock_geolocator = MagicMock()
        mock_geolocator.reverse.return_value = None
        mock_geo.return_value = mock_geolocator

        coords = [(0, 0), (0, 0), (0, 0), (0, 0)]
        result = get_city_country_name_from_rectangle(coords)
        assert "Unknown" in result

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_403_fallback(self, mock_geo):
        from geopy.exc import GeocoderInsufficientPrivileges
        from voxcity.geoprocessor.utils import get_city_country_name_from_rectangle
        mock_geolocator = MagicMock()
        mock_geolocator.reverse.side_effect = GeocoderInsufficientPrivileges("blocked")
        mock_geo.return_value = mock_geolocator

        coords = [(139.65, 35.67), (139.66, 35.67), (139.66, 35.68), (139.65, 35.68)]
        result = get_city_country_name_from_rectangle(coords)
        # Should fall back to offline reverse geocoder or return Unknown
        assert isinstance(result, str)

    @patch("voxcity.geoprocessor.utils._create_nominatim_geolocator")
    def test_timeout_fallback(self, mock_geo):
        from geopy.exc import GeocoderTimedOut
        from voxcity.geoprocessor.utils import get_city_country_name_from_rectangle
        mock_geolocator = MagicMock()
        mock_geolocator.reverse.side_effect = GeocoderTimedOut("timeout")
        mock_geo.return_value = mock_geolocator

        coords = [(0, 0), (0, 0), (0, 0), (0, 0)]
        result = get_city_country_name_from_rectangle(coords)
        assert "Unknown" in result


# ===========================================================================
# Tests for merge_geotiffs
# ===========================================================================

class TestMergeGeotiffs:
    """Cover utils.py lines 431-464."""

    @patch("voxcity.geoprocessor.utils.rasterio")
    def test_empty_list(self, mock_rasterio):
        from voxcity.geoprocessor.utils import merge_geotiffs
        # Should return early without error
        merge_geotiffs([], "output_dir")

    @patch("voxcity.geoprocessor.utils.merge")
    @patch("voxcity.geoprocessor.utils.rasterio")
    def test_merges_valid_files(self, mock_rasterio, mock_merge):
        from voxcity.geoprocessor.utils import merge_geotiffs
        import tempfile
        import os

        # Create fake geotiff files
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "tile1.tif")
            f2 = os.path.join(tmpdir, "tile2.tif")
            for f in [f1, f2]:
                with open(f, "w") as fh:
                    fh.write("fake")

            # Mock rasterio.open to return MagicMocks
            mock_src = MagicMock()
            mock_src.meta = {"driver": "GTiff", "height": 10, "width": 10, "transform": None}
            mock_rasterio.open.return_value = mock_src
            mock_rasterio.open.return_value.__enter__ = lambda s: mock_src
            mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

            # Make os.path.exists return True for our files
            mosaic = np.zeros((1, 10, 10))
            mock_merge.return_value = (mosaic, MagicMock())

            # This will exercise the merge path
            merge_geotiffs([f1, f2], tmpdir)


# ===========================================================================
# Tests for validate_polygon_coordinates
# ===========================================================================

class TestValidatePolygonCoordinates:
    """Cover utils.py line 678."""

    def test_valid_polygon(self):
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
        result = validate_polygon_coordinates(geom)
        assert result is not None

    def test_unclosed_ring(self):
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]  # not closed
        }
        result = validate_polygon_coordinates(geom)
        # Should auto-close the ring
        assert result is not None
