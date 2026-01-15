"""
Shared domain definition for simulator_gpu.

This module re-exports the Domain class from solar.domain for backward compatibility.
The main implementation is in simulator_gpu.solar.domain which includes:
- Domain class with full grid, terrain, building, and vegetation support
- Surfaces class for radiation calculations
- Surface extraction utilities
"""

# Re-export from solar.domain (the main implementation)
from .solar.domain import (
    Domain,
    Surfaces,
    extract_surfaces_from_domain,
    IUP,
    IDOWN,
    INORTH,
    ISOUTH,
    IEAST,
    IWEST,
    DIR_NORMALS,
)

__all__ = [
    'Domain',
    'Surfaces',
    'extract_surfaces_from_domain',
    'IUP',
    'IDOWN',
    'INORTH',
    'ISOUTH',
    'IEAST',
    'IWEST',
    'DIR_NORMALS',
]
