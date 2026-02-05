"""Tests for voxcity.errors module - exception classes."""
import pytest

from voxcity.errors import (
    VoxCityError,
    ConfigurationError,
    DownloaderError,
    ProcessingError,
    VisualizationError,
)


class TestVoxCityError:
    def test_is_exception(self):
        assert issubclass(VoxCityError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(VoxCityError):
            raise VoxCityError("Base error")

    def test_message(self):
        err = VoxCityError("Test message")
        assert str(err) == "Test message"


class TestConfigurationError:
    def test_is_voxcity_error(self):
        assert issubclass(ConfigurationError, VoxCityError)

    def test_can_raise_and_catch(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid config")

    def test_catch_as_base_class(self):
        with pytest.raises(VoxCityError):
            raise ConfigurationError("Config issue")


class TestDownloaderError:
    def test_is_voxcity_error(self):
        assert issubclass(DownloaderError, VoxCityError)

    def test_can_raise_and_catch(self):
        with pytest.raises(DownloaderError):
            raise DownloaderError("Download failed")


class TestProcessingError:
    def test_is_voxcity_error(self):
        assert issubclass(ProcessingError, VoxCityError)

    def test_can_raise_and_catch(self):
        with pytest.raises(ProcessingError):
            raise ProcessingError("Processing failed")


class TestVisualizationError:
    def test_is_voxcity_error(self):
        assert issubclass(VisualizationError, VoxCityError)

    def test_can_raise_and_catch(self):
        with pytest.raises(VisualizationError):
            raise VisualizationError("Visualization failed")


class TestExceptionHierarchy:
    """Test that all exceptions can be caught by VoxCityError."""
    
    @pytest.mark.parametrize("exc_class", [
        ConfigurationError,
        DownloaderError,
        ProcessingError,
        VisualizationError,
    ])
    def test_catchable_by_base(self, exc_class):
        with pytest.raises(VoxCityError):
            raise exc_class("test")

    @pytest.mark.parametrize("exc_class", [
        VoxCityError,
        ConfigurationError,
        DownloaderError,
        ProcessingError,
        VisualizationError,
    ])
    def test_catchable_by_exception(self, exc_class):
        with pytest.raises(Exception):
            raise exc_class("test")
