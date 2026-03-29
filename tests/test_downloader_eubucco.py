"""Tests for voxcity.downloader.eubucco module."""
import json
import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

import voxcity.downloader.eubucco as eubucco_mod
from voxcity.downloader.eubucco import populate_country_links

# Fake API payload returned by the EUBUCCO /countries endpoint.
_FAKE_API_RESPONSE = [
    {"name": c, "gpkg": {"download_link": f"https://data.eubucco.com/v0.1/files/{c.lower()}/download"}}
    for c in [
        "Germany", "France", "Italy", "Spain", "Netherlands",
        "Belgium", "Austria", "Poland", "Portugal", "Sweden",
        "Czech Republic", "Switzerland",
        "France (Other-license)",
    ]
]


@pytest.fixture(autouse=True)
def _populate_country_links():
    """Ensure country_links is populated with mocked data for every test."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _FAKE_API_RESPONSE
    with patch("voxcity.downloader.eubucco.requests.get", return_value=mock_resp):
        populate_country_links()
    yield
    # Reset after tests
    eubucco_mod.country_links = {}


class TestEubuccoCountryLinks:
    """Tests for EUBUCCO country links dictionary."""

    def test_country_links_is_dict(self):
        """Test country_links is a dictionary."""
        assert isinstance(eubucco_mod.country_links, dict)

    def test_country_links_not_empty(self):
        """Test country_links is not empty."""
        assert len(eubucco_mod.country_links) > 0

    def test_major_countries_present(self):
        """Test that major European countries are in the links."""
        expected_countries = [
            "Germany", "France", "Italy", "Spain", "Netherlands",
            "Belgium", "Austria", "Poland", "Portugal", "Sweden"
        ]

        for country in expected_countries:
            assert country in eubucco_mod.country_links, f"{country} not found in country_links"

    def test_links_are_valid_urls(self):
        """Test that all links are valid URLs."""
        for country, link in eubucco_mod.country_links.items():
            assert link.startswith("https://"), f"Link for {country} is not https"
            assert "eubucco.com" in link, f"Link for {country} is not from eubucco.com"

    def test_links_are_strings(self):
        """Test that all links are strings."""
        for country, link in eubucco_mod.country_links.items():
            assert isinstance(link, str)
            assert isinstance(country, str)

    def test_unique_links(self):
        """Test that all links are unique (mostly)."""
        links = list(eubucco_mod.country_links.values())
        assert len(set(links)) > len(links) * 0.8


class TestEubuccoDataStructure:
    """Tests for expected data structure from EUBUCCO."""

    def test_country_links_keys_are_capitalized(self):
        """Test that country names are properly capitalized."""
        for country in eubucco_mod.country_links.keys():
            assert country[0].isupper(), f"{country} should be capitalized"

    def test_special_license_countries(self):
        """Test that some countries have other-license versions."""
        other_license_countries = [k for k in eubucco_mod.country_links.keys() if "Other-license" in k]
        assert len(other_license_countries) >= 1

    def test_link_format_consistency(self):
        """Test that all links follow consistent format."""
        for country, link in eubucco_mod.country_links.items():
            assert link.endswith("/download"), f"Link for {country} doesn't end with /download"
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
