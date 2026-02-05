"""Tests for voxcity.downloader.eubucco module."""
import pytest
import numpy as np
import os
import tempfile

from voxcity.downloader.eubucco import country_links


class TestEubuccoCountryLinks:
    """Tests for EUBUCCO country links dictionary."""
    
    def test_country_links_is_dict(self):
        """Test country_links is a dictionary."""
        assert isinstance(country_links, dict)
    
    def test_country_links_not_empty(self):
        """Test country_links is not empty."""
        assert len(country_links) > 0
    
    def test_major_countries_present(self):
        """Test that major European countries are in the links."""
        expected_countries = [
            "Germany", "France", "Italy", "Spain", "Netherlands",
            "Belgium", "Austria", "Poland", "Portugal", "Sweden"
        ]
        
        for country in expected_countries:
            assert country in country_links, f"{country} not found in country_links"
    
    def test_links_are_valid_urls(self):
        """Test that all links are valid URLs."""
        for country, link in country_links.items():
            assert link.startswith("https://"), f"Link for {country} is not https"
            assert "eubucco.com" in link, f"Link for {country} is not from eubucco.com"
    
    def test_links_are_strings(self):
        """Test that all links are strings."""
        for country, link in country_links.items():
            assert isinstance(link, str)
            assert isinstance(country, str)
    
    def test_unique_links(self):
        """Test that all links are unique (mostly)."""
        # Note: Some countries may share links (e.g., other-license versions)
        links = list(country_links.values())
        # Most links should be unique
        assert len(set(links)) > len(links) * 0.8


class TestEubuccoDataStructure:
    """Tests for expected data structure from EUBUCCO."""
    
    def test_country_links_keys_are_capitalized(self):
        """Test that country names are properly capitalized."""
        for country in country_links.keys():
            # First letter should be uppercase
            assert country[0].isupper(), f"{country} should be capitalized"
    
    def test_special_license_countries(self):
        """Test that some countries have other-license versions."""
        other_license_countries = [k for k in country_links.keys() if "Other-license" in k]
        
        # There should be some other-license entries
        assert len(other_license_countries) >= 1
    
    def test_link_format_consistency(self):
        """Test that all links follow consistent format."""
        for country, link in country_links.items():
            # All links should end with /download
            assert link.endswith("/download"), f"Link for {country} doesn't end with /download"
            
            # All links should contain files/
            assert "/files/" in link, f"Link for {country} doesn't contain /files/"


class TestEubuccoFunctionSignatures:
    """Tests for function availability and signatures."""
    
    def test_filter_function_exists(self):
        """Test that filter_and_convert_gdf_to_geojson_eubucco exists."""
        from voxcity.downloader.eubucco import filter_and_convert_gdf_to_geojson_eubucco
        assert callable(filter_and_convert_gdf_to_geojson_eubucco)
    
    def test_download_function_exists(self):
        """Test that download_extract_open_gpkg_from_eubucco exists."""
        from voxcity.downloader.eubucco import download_extract_open_gpkg_from_eubucco
        assert callable(download_extract_open_gpkg_from_eubucco)
    
    def test_get_gdf_function_exists(self):
        """Test that get_gdf_from_eubucco exists."""
        from voxcity.downloader.eubucco import get_gdf_from_eubucco
        assert callable(get_gdf_from_eubucco)
    
    def test_load_gdf_function_exists(self):
        """Test that load_gdf_from_eubucco exists."""
        from voxcity.downloader.eubucco import load_gdf_from_eubucco
        assert callable(load_gdf_from_eubucco)
