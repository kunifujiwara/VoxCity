"""Tests for voxcity.utils.projector — GridGeom and GridProjector."""
import pytest
import numpy as np

from voxcity.utils.projector import GridGeom, GridProjector
from voxcity.geoprocessor.raster.core import compute_grid_geometry


# Axis-aligned rectangle in Tokyo (~4 km × 4 km), vertex order SW/SE/NE/NW.
# v0=SW, v1=SE, v3=NW  →  side_1 points east, side_2 points north.
_RECT = [
    [139.680, 35.680],  # 0 SW
    [139.716, 35.680],  # 1 SE
    [139.716, 35.716],  # 2 NE (not used by compute_grid_geometry)
    [139.680, 35.716],  # 3 NW
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


class TestProjectorRoundTrip:
    """lon_lat → ij_north → lon_lat must be the identity."""

    _TOL = 1e-9

    def _round_trip(self, proj, lon, lat):
        i, j = proj.lon_lat_to_ij_north(lon, lat)
        lon2, lat2 = proj.ij_north_to_lon_lat(i, j)
        assert abs(lon2 - lon) < self._TOL, f"lon drift: {lon2 - lon}"
        assert abs(lat2 - lat) < self._TOL, f"lat drift: {lat2 - lat}"

    def test_origin_corner(self, proj, geom):
        self._round_trip(proj, geom["origin"][0], geom["origin"][1])

    def test_ne_corner(self, proj):
        self._round_trip(proj, 139.716, 35.716)

    def test_sw_corner(self, proj):
        self._round_trip(proj, 139.680, 35.680)

    def test_grid_centre(self, proj):
        self._round_trip(proj, 139.698, 35.698)

    def test_arbitrary_interior(self, proj):
        self._round_trip(proj, 139.700, 35.702)


class TestProjectorOriginMapsToZero:
    """The grid origin (v0) must map to cell coordinates (0, 0)."""

    def test_origin_is_zero(self, proj, geom):
        i, j = proj.lon_lat_to_ij_north(geom["origin"][0], geom["origin"][1])
        assert abs(i) < 1e-9
        assert abs(j) < 1e-9


class TestProjectorCellCentreConsistency:
    """Cell (0, 0) centre is at origin + 0.5*dx*u + 0.5*dy*v."""

    def test_cell_centre_maps_to_half(self, proj, geom):
        dx_m, dy_m = geom["adj_mesh"]
        u = geom["u_vec"]
        v = geom["v_vec"]
        o = geom["origin"]
        centre_lon = o[0] + 0.5 * dx_m * u[0] + 0.5 * dy_m * v[0]
        centre_lat = o[1] + 0.5 * dx_m * u[1] + 0.5 * dy_m * v[1]
        i, j = proj.lon_lat_to_ij_north(centre_lon, centre_lat)
        assert abs(i - 0.5) < 1e-9
        assert abs(j - 0.5) < 1e-9


class TestProjectorWorldXY:
    """lon_lat ↔ xy_world_m round-trip and consistency with build_voxel_buffers."""

    _TOL = 1e-9

    def test_round_trip(self, proj, geom):
        """xy_world_m_to_lon_lat(lon_lat_to_xy_world_m(lon, lat)) == (lon, lat)."""
        lon, lat = 139.698, 35.698
        x, y = proj.lon_lat_to_xy_world_m(lon, lat)
        lon2, lat2 = proj.xy_world_m_to_lon_lat(x, y)
        assert abs(lon2 - lon) < self._TOL
        assert abs(lat2 - lat) < self._TOL

    def test_origin_maps_to_nx_times_ms(self, proj, geom):
        """Grid origin maps to x = nx * meshsize_m, y = 0 in world coords.

        build_voxel_buffers places voxel (i, j) at (i*ms, j*ms).  The origin
        (ij_north u=0) corresponds to the far x-end of the mesh (x = nx*ms)
        because of the (nx - u) flip in lonLatToWorldXY.
        """
        nx = geom["grid_size"][0]
        ms = geom["meshsize_m"]
        x, y = proj.lon_lat_to_xy_world_m(geom["origin"][0], geom["origin"][1])
        assert abs(x - nx * ms) < 1e-6
        assert abs(y) < 1e-6

    def test_far_corner_maps_to_world_origin(self, proj, geom):
        """The SE corner (u=nx, v=0) maps to world (0, 0).

        build_voxel_buffers's x=i*ms axis starts at the SE end of the grid
        because the (nx - u) flip maps ij_north_i=nx to x=0.
        """
        nx, ny = geom["grid_size"]
        # SE corner = origin moved nx cells along u_vec
        dx_m, dy_m = geom["adj_mesh"]
        u = geom["u_vec"]
        v = geom["v_vec"]
        o = geom["origin"]
        se_lon = o[0] + nx * dx_m * u[0]
        se_lat = o[1] + nx * dx_m * u[1]
        x, y = proj.lon_lat_to_xy_world_m(se_lon, se_lat)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_xy_world_m_to_lon_lat_round_trip_corners(self, proj, geom):
        nx, ny = geom["grid_size"]
        ms = geom["meshsize_m"]
        for x_w, y_w in [(0.0, 0.0), (nx * ms, 0.0), (0.0, ny * ms), (nx * ms, ny * ms)]:
            lon, lat = proj.xy_world_m_to_lon_lat(x_w, y_w)
            x2, y2 = proj.lon_lat_to_xy_world_m(lon, lat)
            assert abs(x2 - x_w) < self._TOL
            assert abs(y2 - y_w) < self._TOL


class TestProjectorRotatedGrid:
    """Round-trip on a ~45° rotated rectangle exercises all four affine terms.

    Swapped a/d or sign-flipped b/c would produce wrong results on axis-aligned
    grids but be caught here.  The rotated rectangle is built by rotating the
    Tokyo axis-aligned corners 45° around their centre in lon/lat space.
    """

    _TOL = 1e-9

    @pytest.fixture(scope="class")
    def rotated_proj(self):
        import math
        cx, cy = 139.698, 35.698
        half = 0.018   # ~2 km in degrees (approx)
        angle = math.radians(45)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def rot(dx, dy):
            return (cx + cos_a * dx - sin_a * dy,
                    cy + sin_a * dx + cos_a * dy)

        v0 = rot(-half, -half)
        v1 = rot(+half, -half)
        v2 = rot(+half, +half)
        v3 = rot(-half, +half)
        rect = [list(v0), list(v1), list(v2), list(v3)]
        g = compute_grid_geometry(rect, 50.0)
        assert g is not None
        return GridProjector(g)

    def _round_trip(self, proj, lon, lat):
        i, j = proj.lon_lat_to_ij_north(lon, lat)
        lon2, lat2 = proj.ij_north_to_lon_lat(i, j)
        assert abs(lon2 - lon) < self._TOL
        assert abs(lat2 - lat) < self._TOL

    def test_centre(self, rotated_proj):
        self._round_trip(rotated_proj, 139.698, 35.698)

    def test_offset_point(self, rotated_proj):
        self._round_trip(rotated_proj, 139.706, 35.704)

    def test_world_xy_round_trip(self, rotated_proj):
        lon, lat = 139.700, 35.702
        x, y = rotated_proj.lon_lat_to_xy_world_m(lon, lat)
        lon2, lat2 = rotated_proj.xy_world_m_to_lon_lat(x, y)
        assert abs(lon2 - lon) < self._TOL
        assert abs(lat2 - lat) < self._TOL


class TestProjectorDegenerateGrid:
    """A zero-area rectangle must raise ValueError."""

    def test_degenerate_raises(self):
        bad_geom: GridGeom = {
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
            GridProjector(bad_geom)
