"""Tests for voxcity.utils.projector — GridGeom and GridProjector (uv_m API)."""
import math
import pytest
import numpy as np

from voxcity.utils.projector import GridGeom, GridProjector
from voxcity.geoprocessor.raster.core import compute_grid_geometry


# Axis-aligned ~4 km × 4 km rectangle in Tokyo, vertex order SW/NW/NE/SE.
_RECT = [
    [139.680, 35.680],  # 0 SW
    [139.680, 35.716],  # 1 NW
    [139.716, 35.716],  # 2 NE
    [139.716, 35.680],  # 3 SE
]
_MESHSIZE = 50.0


@pytest.fixture(scope="module")
def geom() -> GridGeom:
    g = compute_grid_geometry(_RECT, _MESHSIZE)
    assert g is not None
    return g


@pytest.fixture(scope="module")
def proj(geom) -> GridProjector:
    return GridProjector(geom)


# ── Rotated-grid fixture ────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def rotated_proj():
    import math as _math
    cx, cy = 139.698, 35.698
    half = 0.018
    angle = _math.radians(45)
    cos_a, sin_a = _math.cos(angle), _math.sin(angle)

    def rot(dx, dy):
        return (cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy)

    rect = [list(rot(-half, -half)), list(rot(+half, -half)),
            list(rot(+half, +half)), list(rot(-half, +half))]
    g = compute_grid_geometry(rect, 50.0)
    assert g is not None
    return GridProjector(g)


# ── GridGeom fields ─────────────────────────────────────────────────────────

class TestGridGeomFields:
    def test_has_meshsize_m(self, geom):
        assert "meshsize_m" in geom
        assert geom["meshsize_m"] == pytest.approx(_MESHSIZE)

    def test_all_required_keys(self, geom):
        for key in ("origin", "side_1", "side_2", "u_vec", "v_vec",
                    "grid_size", "adj_mesh", "meshsize_m"):
            assert key in geom

    def test_grid_size_positive(self, geom):
        nx, ny = geom["grid_size"]
        assert nx > 0 and ny > 0


# ── lon_lat ↔ uv_m round-trip ───────────────────────────────────────────────

class TestUvMRoundTrip:
    _TOL = 1e-9

    def _rt(self, proj, lon, lat):
        u_m, v_m = proj.lon_lat_to_uv_m(lon, lat)
        lon2, lat2 = proj.uv_m_to_lon_lat(u_m, v_m)
        assert abs(lon2 - lon) < self._TOL
        assert abs(lat2 - lat) < self._TOL

    def test_origin(self, proj, geom):
        self._rt(proj, geom["origin"][0], geom["origin"][1])

    def test_ne_corner(self, proj):     self._rt(proj, 139.716, 35.716)
    def test_sw_corner(self, proj):     self._rt(proj, 139.680, 35.680)
    def test_centre(self, proj):        self._rt(proj, 139.698, 35.698)
    def test_interior(self, proj):      self._rt(proj, 139.700, 35.702)


class TestUvMOrigin:
    def test_origin_maps_to_zero(self, proj, geom):
        u_m, v_m = proj.lon_lat_to_uv_m(geom["origin"][0], geom["origin"][1])
        assert abs(u_m) < 1e-9
        assert abs(v_m) < 1e-9


class TestCellCentreConsistency:
    """Cell-centre lon_lat → uv_m = (0.5*du, 0.5*dv)."""

    def test_cell_zero_centre(self, proj, geom):
        du_m, dv_m = geom["adj_mesh"]
        u, v = geom["u_vec"], geom["v_vec"]
        o = geom["origin"]
        clon = o[0] + 0.5 * du_m * u[0] + 0.5 * dv_m * v[0]
        clat = o[1] + 0.5 * du_m * u[1] + 0.5 * dv_m * v[1]
        u_m, v_m = proj.lon_lat_to_uv_m(clon, clat)
        assert abs(u_m - 0.5 * du_m) < 2e-9
        assert abs(v_m - 0.5 * dv_m) < 2e-9


# ── Scene position invariant ─────────────────────────────────────────────────

class TestScenePositionInvariant:
    """After Phase 3, lon_lat_to_uv_m gives the direct scene position."""

    def test_origin_maps_to_scene_zero(self, proj, geom):
        u_m, v_m = proj.lon_lat_to_uv_m(geom["origin"][0], geom["origin"][1])
        assert u_m == pytest.approx(0.0, abs=1e-6)
        assert v_m == pytest.approx(0.0, abs=1e-6)

    def test_no_nx_flip(self, proj, geom):
        # Old formula: (nx - u) * ms → origin maps to nx*ms, not 0.
        # New formula: u * du → origin maps to 0.
        nx = geom["grid_size"][0]
        ms = geom["meshsize_m"]
        u_m, _ = proj.lon_lat_to_uv_m(geom["origin"][0], geom["origin"][1])
        assert abs(u_m) < 1e-6            # NEW: origin → 0
        assert abs(u_m - nx * ms) > 1.0   # OLD formula would give nx*ms


class TestPackageRectangleConvention:
    def test_u_axis_is_north_for_drawn_rectangles(self, proj, geom):
        u_m, v_m = proj.lon_lat_to_uv_m(_RECT[1][0], _RECT[1][1])
        width_u = geom["grid_size"][0] * geom["adj_mesh"][0]
        assert u_m == pytest.approx(width_u, abs=1.0)
        assert v_m == pytest.approx(0.0, abs=1e-6)

    def test_v_axis_is_east_for_drawn_rectangles(self, proj, geom):
        u_m, v_m = proj.lon_lat_to_uv_m(_RECT[3][0], _RECT[3][1])
        width_v = geom["grid_size"][1] * geom["adj_mesh"][1]
        assert u_m == pytest.approx(0.0, abs=1e-6)
        assert v_m == pytest.approx(width_v, abs=1.0)


# ── Cell index ───────────────────────────────────────────────────────────────

class TestCellIndex:
    def test_returns_ints_for_scalar(self, proj):
        i, j = proj.lon_lat_to_cell(139.698, 35.698)
        assert isinstance(i, (int, np.integer))
        assert isinstance(j, (int, np.integer))

    def test_cell_to_lon_lat_round_trip(self, proj):
        i, j = proj.lon_lat_to_cell(139.700, 35.702)
        lon, lat = proj.cell_to_lon_lat(i, j)
        i2, j2 = proj.lon_lat_to_cell(lon, lat)
        assert i2 == i and j2 == j

    def test_floor_division_at_boundary(self, proj, geom):
        du_m, _ = geom["adj_mesh"]
        u, v = geom["u_vec"], geom["v_vec"]
        o = geom["origin"]
        lon = o[0] + 1.0 * du_m * u[0]
        lat = o[1] + 1.0 * du_m * u[1]
        i, _ = proj.lon_lat_to_cell(lon, lat)
        assert i == 1


# ── Vectorisation ─────────────────────────────────────────────────────────────

class TestVectorised:
    def test_scalar_returns_float(self, proj):
        u_m, v_m = proj.lon_lat_to_uv_m(139.698, 35.698)
        assert isinstance(u_m, float)

    def test_array_returns_ndarray(self, proj):
        lons = np.array([139.690, 139.695, 139.700])
        lats = np.array([35.690, 35.695, 35.700])
        u_m, v_m = proj.lon_lat_to_uv_m(lons, lats)
        assert isinstance(u_m, np.ndarray) and u_m.shape == (3,)

    def test_cell_array_returns_int_dtype(self, proj):
        lons = np.array([139.690, 139.695, 139.700])
        lats = np.array([35.690, 35.695, 35.700])
        i_arr, j_arr = proj.lon_lat_to_cell(lons, lats)
        assert i_arr.dtype.kind == 'i'

    def test_round_trip_vectorised(self, proj):
        lons = np.array([139.690, 139.698, 139.710])
        lats = np.array([35.690, 35.698, 35.710])
        u_m, v_m = proj.lon_lat_to_uv_m(lons, lats)
        lons2, lats2 = proj.uv_m_to_lon_lat(u_m, v_m)
        np.testing.assert_allclose(lons2, lons, atol=1e-9)
        np.testing.assert_allclose(lats2, lats, atol=1e-9)


# ── Rotated grid ──────────────────────────────────────────────────────────────

class TestRotatedGrid:
    _TOL = 1e-9

    def test_centre(self, rotated_proj):
        u_m, v_m = rotated_proj.lon_lat_to_uv_m(139.698, 35.698)
        lon2, lat2 = rotated_proj.uv_m_to_lon_lat(u_m, v_m)
        assert abs(lon2 - 139.698) < self._TOL

    def test_offset_point(self, rotated_proj):
        u_m, v_m = rotated_proj.lon_lat_to_uv_m(139.706, 35.704)
        lon2, lat2 = rotated_proj.uv_m_to_lon_lat(u_m, v_m)
        assert abs(lon2 - 139.706) < self._TOL


# ── Degenerate grid ───────────────────────────────────────────────────────────

class TestDegenerateGrid:
    def test_raises(self):
        bad: GridGeom = {
            "origin": np.array([0.0, 0.0]),
            "side_1": np.array([0.0, 0.0]),
            "side_2": np.array([0.0, 0.0]),
            "u_vec": np.array([0.0, 0.0]),
            "v_vec": np.array([0.0, 0.0]),
            "grid_size": (1, 1),
            "adj_mesh": (0.0, 0.0),
            "meshsize_m": 50.0,
        }
        with pytest.raises(ValueError, match="degenerate"):
            GridProjector(bad)


# ── utils __init__ export ──────────────────────────────────────────────────────

def test_importable_from_utils():
    from voxcity.utils import GridGeom, GridProjector  # noqa: F401
