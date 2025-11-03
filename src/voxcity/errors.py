"""
Package-specific error hierarchy for voxcity.

This enables precise exception handling without leaking low-level
implementation details across boundaries.
"""

from __future__ import annotations


class VoxCityError(Exception):
    """Base exception for all voxcity errors."""


class ConfigurationError(VoxCityError):
    """Raised when configuration values are missing or invalid."""


class DownloaderError(VoxCityError):
    """Raised by downloader modules when remote data retrieval fails."""


class ProcessingError(VoxCityError):
    """Raised for failures during grid/voxel processing or geoprocessing."""


class VisualizationError(VoxCityError):
    """Raised for visualization/rendering failures."""


