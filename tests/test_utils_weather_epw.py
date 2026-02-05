"""Tests for voxcity.utils.weather.epw module."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from voxcity.utils.weather.epw import (
    process_epw,
    read_epw_for_solar_simulation,
)


class TestProcessEpw:
    """Tests for process_epw function."""

    @pytest.fixture
    def sample_epw_file(self, tmp_path):
        """Generate a minimal valid EPW file."""
        lines = [
            "LOCATION,TestCity,TX,USA,TMY3,722590,32.89,-97.02,-6.0,180.0\n",
            "DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-4.5,-2.6\n",
            "TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature,Extreme,7/ 1,7/ 7\n",
            "GROUND TEMPERATURES,3,.5,,,,18.0,18.5,19.8,21.5,26.6,30.1,31.6,30.7,27.8,23.8,19.7,17.2\n",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n",
            "COMMENTS 1,TMY3 formatted data\n",
            "COMMENTS 2,\n",
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n",
        ]
        # Add some data rows (8760 for full year, but just add a few for testing)
        for i in range(24):  # One day of data
            hour = i + 1
            line = f"2021,1,1,{hour},0,?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0*0A0A0A0A0A0A0E0E0,10.0,5.0,60,101300,0,0,300,100,50,30,0,0,0,0,180,3.0,0,0,10000,77777,0,0,0,0,0,0,0.5,0.1,0.2\n"
            lines.append(line)
        
        epw_file = tmp_path / "test.epw"
        epw_file.write_text("".join(lines))
        return str(epw_file)

    def test_returns_dataframe_and_headers(self, sample_epw_file):
        """Test that function returns DataFrame and headers dict."""
        df, headers = process_epw(sample_epw_file)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(headers, dict)

    def test_parses_location_info(self, sample_epw_file):
        """Test that location info is parsed correctly."""
        df, headers = process_epw(sample_epw_file)
        loc = headers['LOCATION']
        assert loc['City'] == 'TestCity'
        assert loc['Country'] == 'USA'
        assert loc['Latitude'] == pytest.approx(32.89)
        assert loc['Longitude'] == pytest.approx(-97.02)

    def test_datetime_index(self, sample_epw_file):
        """Test that DataFrame has datetime index."""
        df, headers = process_epw(sample_epw_file)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_numeric_columns_converted(self, sample_epw_file):
        """Test that numeric columns are converted to numeric types."""
        df, headers = process_epw(sample_epw_file)
        assert pd.api.types.is_numeric_dtype(df['Dry Bulb Temperature'])
        assert pd.api.types.is_numeric_dtype(df['Direct Normal Radiation'])


class TestReadEpwForSolarSimulation:
    """Tests for read_epw_for_solar_simulation function."""

    @pytest.fixture
    def sample_epw_file(self, tmp_path):
        """Generate a minimal valid EPW file for solar simulation."""
        lines = [
            "LOCATION,TestCity,TX,USA,TMY3,722590,32.89,-97.02,-6.0,180.0\n",
            "DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-4.5,-2.6\n",
            "TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature,Extreme,7/ 1,7/ 7\n",
            "GROUND TEMPERATURES,3,.5,,,,18.0,18.5,19.8,21.5,26.6,30.1,31.6,30.7,27.8,23.8,19.7,17.2\n",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n",
            "COMMENTS 1,TMY3 formatted data\n",
            "COMMENTS 2,\n",
            "DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n",
        ]
        # Add some data rows
        for i in range(24):
            hour = i + 1
            dni = 500 + i * 10  # Direct Normal Irradiance
            dhi = 100 + i * 5   # Diffuse Horizontal Irradiance
            line = f"2021,1,1,{hour},0,?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0?0*0A0A0A0A0A0A0E0E0,10.0,5.0,60,101300,0,0,300,100,{dni},{dhi},0,0,0,0,180,3.0,0,0,10000,77777,0,0,0,0,0,0,0.5,0.1,0.2\n"
            lines.append(line)
        
        epw_file = tmp_path / "test_solar.epw"
        epw_file.write_text("".join(lines))
        return str(epw_file)

    def test_returns_correct_structure(self, sample_epw_file):
        """Test that function returns 5-tuple."""
        result = read_epw_for_solar_simulation(sample_epw_file)
        assert len(result) == 5
        df, lon, lat, tz, elev = result
        assert isinstance(df, pd.DataFrame)

    def test_extracts_location_coordinates(self, sample_epw_file):
        """Test that location coordinates are extracted correctly."""
        df, lon, lat, tz, elev = read_epw_for_solar_simulation(sample_epw_file)
        assert lat == pytest.approx(32.89)
        assert lon == pytest.approx(-97.02)
        assert tz == pytest.approx(-6.0)
        assert elev == pytest.approx(180.0)

    def test_dataframe_has_dni_dhi_columns(self, sample_epw_file):
        """Test that DataFrame has DNI and DHI columns."""
        df, lon, lat, tz, elev = read_epw_for_solar_simulation(sample_epw_file)
        assert 'DNI' in df.columns
        assert 'DHI' in df.columns

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            read_epw_for_solar_simulation("nonexistent_file.epw")

    def test_dataframe_sorted_by_time(self, sample_epw_file):
        """Test that DataFrame is sorted by time index."""
        df, lon, lat, tz, elev = read_epw_for_solar_simulation(sample_epw_file)
        # Check that index is sorted
        assert df.index.is_monotonic_increasing
