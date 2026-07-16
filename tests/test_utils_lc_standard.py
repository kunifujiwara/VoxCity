import numpy as np

from voxcity.utils.lc import convert_land_cover, get_class_priority


def test_standard_priority_is_dict_with_standard_names():
    pri = get_class_priority("Standard")
    assert isinstance(pri, dict)
    for name in ("Building", "Road", "Developed space", "Rangeland"):
        assert name in pri
    # VoxCity draws the LOWEST priority number LAST => it wins. So built
    # environment must have the smallest numbers (mirrors the OSM dict: Road=1).
    assert pri["Road"] < pri["Developed space"] < pri["Rangeland"]


def test_standard_convert_is_identity():
    arr = np.array([[2, 11, 12], [13, 5, 2]], dtype=np.int32)
    out = convert_land_cover(arr, land_cover_source="Standard")
    assert np.array_equal(out, arr)
