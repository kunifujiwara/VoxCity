import pytest
from voxcity.importer.units import unit_scale, validate_units


def test_known_units_scale_to_meters():
    assert unit_scale("m") == 1.0
    assert unit_scale("cm") == 0.01
    assert unit_scale("mm") == 0.001
    assert unit_scale("ft") == pytest.approx(0.3048)
    assert unit_scale("in") == pytest.approx(0.0254)


def test_units_are_case_insensitive():
    assert unit_scale("M") == 1.0
    assert unit_scale("FT") == pytest.approx(0.3048)


def test_invalid_units_raise_valueerror():
    with pytest.raises(ValueError, match="Unknown units"):
        unit_scale("furlong")


def test_validate_units_passes_known():
    validate_units("mm")  # should not raise
