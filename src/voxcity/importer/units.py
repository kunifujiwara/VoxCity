"""Unit handling for OBJ import (model units -> meters)."""

_UNIT_SCALE = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "in": 0.0254,
}


def unit_scale(units: str) -> float:
    """Return meters-per-unit for a model unit string (case-insensitive)."""
    if not isinstance(units, str):
        raise ValueError(f"Unknown units: {units!r}. Expected one of {sorted(_UNIT_SCALE)}.")
    key = units.lower()
    if key not in _UNIT_SCALE:
        raise ValueError(f"Unknown units: {units!r}. Expected one of {sorted(_UNIT_SCALE)}.")
    return _UNIT_SCALE[key]


def validate_units(units: str) -> None:
    """Raise ValueError if *units* is not a known unit string."""
    unit_scale(units)
