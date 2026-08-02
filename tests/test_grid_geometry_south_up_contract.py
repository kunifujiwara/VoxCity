"""South-up cell-centre contract of ``compute_grid_geometry``/``compute_cell_center_coords``.

The app's ``[SW, NW, NE, SE]`` rectangle convention plus this contract is what
makes "row 0 = the southern edge" true everywhere downstream: the overlays
drawn by ``voxcity.geoprocessor.geojson``, the ``GridProjector`` lookups behind
``/api/model/anchor_ground``, and the OBJ-import placement all assume that the
grid's row index walks the ``vertex[0]`` end of ``side_1`` toward ``vertex[1]``.
Until now that claim was pinned only through the app's own test suite
(``app/backend/test_frame_consumers.py``, ``test_model_geo_overlays.py``),
which a library-only checkout cannot run.

The claims, stated once:

* ``origin`` is ``rectangle_vertices[0]`` — not a bounding-box corner.
* Cell ``(i, j)`` centre = ``origin + (i+0.5)*du*u_vec + (j+0.5)*dv*v_vec``,
  so for the app's ``[SW, NW, NE, SE]`` order **latitude strictly increases
  with the row index** and longitude with the column index.
* ``grid_size[0]`` counts cells along ``side_1`` (v0→v1) and ``grid_size[1]``
  along ``side_2`` (v0→v3) — the fixture's unequal side lengths make a
  transposed result impossible to miss.
* On a rotated rectangle the corner cells pair off with the vertices
  ``(0,0)→v0, (nx-1,0)→v1, (nx-1,ny-1)→v2, (0,ny-1)→v3`` — the assertion a
  north<->south mirror cannot survive.

Fixture discipline: the axis-aligned rectangle is 400 m x 300 m (20 x 15
cells — deliberately non-square), expected values are computed in metres from
the geodesy that built the rectangle, and every mirror-sensitive claim is
checked against an absolute lon/lat, not against another output of the same
function.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from voxcity.geoprocessor.raster import (
    compute_cell_center_coords,
    compute_grid_geometry,
)
from voxcity.utils import GridProjector

# ── Axis-aligned fixture: 400 m north-south x 300 m east-west, 20 m cells ──
M_PER_DEG_LAT = 111320.0
MESHSIZE = 20.0
NS_METERS, EW_METERS = 400.0, 300.0
LON0, LAT0 = 139.7000, 35.6800
DLAT = NS_METERS / M_PER_DEG_LAT
DLON = EW_METERS / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
# The app's vertex order: [SW, NW, NE, SE].
RECT = [
    (LON0, LAT0),
    (LON0, LAT0 + DLAT),
    (LON0 + DLON, LAT0 + DLAT),
    (LON0 + DLON, LAT0),
]


@pytest.fixture(scope="module")
def gg():
    g = compute_grid_geometry(RECT, MESHSIZE)
    assert g is not None
    return g


@pytest.fixture(scope="module")
def cc():
    c = compute_cell_center_coords(RECT, MESHSIZE)
    assert c is not None
    return c


class TestAxisAlignedSouthUp:
    def test_the_fixture_is_not_square(self, gg):
        """A square grid hid a transpose once before in this project."""
        nx, ny = gg["grid_size"]
        assert nx != ny

    def test_origin_is_vertex_zero(self, gg):
        np.testing.assert_allclose(gg["origin"], RECT[0])

    def test_axis0_counts_side1_cells_axis1_counts_side2_cells(self, gg):
        # 400 m of side_1 at 20 m cells vs 300 m of side_2: (20, 15), not (15, 20).
        assert gg["grid_size"] == (20, 15)

    def test_latitude_strictly_increases_with_the_row_index(self, cc):
        lats = cc["lats"]
        assert np.all(np.diff(lats, axis=0) > 0), (
            "row index must walk from vertex[0] (south, for the app's "
            "[SW, NW, NE, SE] order) toward vertex[1] (north)")
        # Constant along a row for an axis-aligned rectangle.
        assert np.allclose(np.diff(lats, axis=1), 0.0, atol=1e-12)

    def test_longitude_strictly_increases_with_the_column_index(self, cc):
        lons = cc["lons"]
        assert np.all(np.diff(lons, axis=1) > 0)
        assert np.allclose(np.diff(lons, axis=0), 0.0, atol=1e-12)

    def test_row0_centre_is_half_a_cell_north_of_the_south_edge(self, cc):
        """Absolute check in metres against the geodesy that built RECT.

        On a mirrored grid row 0 would sit half a cell *south of the north
        edge*, ~390 m away from this expectation — the 1% tolerance only
        absorbs the adjusted-meshsize rounding.
        """
        north_of_south_edge_m = (cc["lats"][0, 0] - LAT0) * M_PER_DEG_LAT
        assert north_of_south_edge_m == pytest.approx(MESHSIZE / 2, rel=0.01)

    def test_last_row_centre_is_half_a_cell_south_of_the_north_edge(self, cc):
        south_of_north_edge_m = (LAT0 + DLAT - cc["lats"][-1, 0]) * M_PER_DEG_LAT
        assert south_of_north_edge_m == pytest.approx(MESHSIZE / 2, rel=0.01)

    def test_row_pitch_is_one_cell_northward(self, cc):
        """Each row step must be worth exactly one (adjusted) cell of latitude."""
        pitch_m = np.diff(cc["lats"][:, 0]) * M_PER_DEG_LAT
        assert pitch_m == pytest.approx(np.full(pitch_m.shape, MESHSIZE), rel=0.01)

    @pytest.mark.parametrize("cell", [(2, 12), (17, 1)], ids=["south-east", "north-west"])
    def test_cell_centres_round_trip_through_the_projector(self, gg, cc, cell):
        """The contract the app's anchor_ground lookups stand on: the centre
        coordinates and ``GridProjector.lon_lat_to_cell`` agree, probed at two
        off-centre cells near opposite corners so a constant offset or a
        mirror cannot pass."""
        i, j = cell
        proj = GridProjector(gg)
        assert proj.lon_lat_to_cell(cc["lons"][i, j], cc["lats"][i, j]) == (i, j)


# ── Rotated rectangle: corner cells must pair with their vertices ──────────

def _rotated_rect(angle_deg, ns_m=400.0, ew_m=300.0, origin=(0.0, 0.0)):
    """[v0, v1, v2, v3] with the v0→v1 edge bearing ``angle_deg`` from north.

    Built near the equator, where degrees are isotropic, so planar distances
    in degrees rank the same as geodesic ones (same construction as
    ``tests/test_pipeline_rotation_angle.py``).
    """
    a = math.radians(angle_deg)
    d1 = (ns_m / M_PER_DEG_LAT * math.sin(a), ns_m / M_PER_DEG_LAT * math.cos(a))
    d2 = (ew_m / M_PER_DEG_LAT * math.cos(a), -ew_m / M_PER_DEG_LAT * math.sin(a))
    ox, oy = origin
    return [
        (ox, oy),
        (ox + d1[0], oy + d1[1]),
        (ox + d1[0] + d2[0], oy + d1[1] + d2[1]),
        (ox + d2[0], oy + d2[1]),
    ]


class TestRotatedRectangle:
    ANGLE = 30.0

    @pytest.fixture(scope="class")
    def rot_cc(self):
        c = compute_cell_center_coords(_rotated_rect(self.ANGLE), MESHSIZE)
        assert c is not None
        return c

    def test_the_rotated_fixture_is_asymmetric(self, rot_cc):
        """All four vertices distinct and the sides unequal, or the corner
        pairing below could not tell a mirror from the identity."""
        rect = np.asarray(_rotated_rect(self.ANGLE))
        assert len({tuple(v) for v in rect.round(12).tolist()}) == 4
        nx, ny = rot_cc["grid_size"]
        assert nx != ny

    @pytest.mark.parametrize(
        "corner_of, vertex_idx",
        [
            ("first-row-first-col", 0),
            ("last-row-first-col", 1),
            ("last-row-last-col", 2),
            ("first-row-last-col", 3),
        ],
    )
    def test_each_corner_cell_hugs_its_own_vertex(self, rot_cc, corner_of, vertex_idx):
        """Cell (0,0) belongs to vertex[0]; (nx-1,0) to vertex[1]; and so on.

        A north<->south mirror pairs (0,0) with vertex[1] instead, a
        column mirror with vertex[3], a transpose with vertex[3]/[1] — every
        one of them moves the corner cell hundreds of metres, so nearest-vertex
        is an unambiguous verdict.
        """
        rect = np.asarray(_rotated_rect(self.ANGLE))
        nx, ny = rot_cc["grid_size"]
        i = 0 if corner_of.startswith("first-row") else nx - 1
        j = 0 if corner_of.endswith("first-col") else ny - 1
        centre = np.array([rot_cc["lons"][i, j], rot_cc["lats"][i, j]])
        dists = np.linalg.norm(rect - centre, axis=1)
        assert int(np.argmin(dists)) == vertex_idx, (
            f"cell ({i}, {j}) sits nearest vertex {int(np.argmin(dists))}, "
            f"not vertex {vertex_idx} — the grid is mirrored or transposed "
            "relative to the rectangle convention")

    def test_row_steps_walk_along_side1(self, rot_cc):
        """Moving one row must move parallel to v0→v1, positively — even when
        that direction is nowhere near north."""
        side_1 = rot_cc["side_1"]
        step = np.array([
            rot_cc["lons"][1, 0] - rot_cc["lons"][0, 0],
            rot_cc["lats"][1, 0] - rot_cc["lats"][0, 0],
        ])
        cross = side_1[0] * step[1] - side_1[1] * step[0]
        assert abs(cross) < 1e-12 * np.linalg.norm(side_1)
        assert float(np.dot(step, side_1)) > 0
