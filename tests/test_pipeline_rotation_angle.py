"""``VoxCityPipeline.assemble_voxcity`` must derive ``extras['rotation_angle']``.

``compute_rotation_angle`` documents itself as "the single producer of the
``rotation_angle`` stored in ``VoxCity.extras``", but ``assemble_voxcity`` is a
*second* constructor path — the one ``voxcitygml.generate_voxcity`` (PLATEAU
LOD2) assembles through — and it used to populate ``rectangle_vertices``,
``canopy_top`` and ``canopy_bottom`` while leaving ``rotation_angle`` out.

Every solar path reads the angle with a default of zero
(``simulator/solar/radiation.py``, ``simulator_gpu/.../volumetric.py``), and the
angle is what maps a geographic sun azimuth into the grid frame. Missing, it
defaults to 0 and shadows over a rotated AOI come out rotated by the AOI's own
angle — silently, and looking entirely plausible. ``save_voxcity`` re-derives
the angle from ``rectangle_vertices`` when writing HDF5, so only the in-memory
model was affected: a saved-and-reloaded model behaved differently from the one
still in memory.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from voxcity.generator.pipeline import VoxCityPipeline
from voxcity.geoprocessor.utils import compute_rotation_angle
from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map


def _rotated_rect(angle_deg, size_deg=0.01, origin=(0.0, 0.0)):
    """Rectangle whose v0->v1 (SW->NW) edge bears ``angle_deg`` clockwise from north.

    Built near the equator, where Web Mercator x/y are proportional to lon/lat,
    so the bearing in lon/lat equals the Mercator bearing. Same construction as
    ``tests/test_rotation_angle.py``.
    """
    a = math.radians(angle_deg)
    ox, oy = origin
    d1 = (size_deg * math.sin(a), size_deg * math.cos(a))
    d2 = (size_deg * math.cos(a), -size_deg * math.sin(a))
    v0 = (ox, oy)
    v1 = (ox + d1[0], oy + d1[1])
    v3 = (ox + d2[0], oy + d2[1])
    v2 = (ox + d1[0] + d2[0], oy + d1[1] + d2[1])
    return [v0, v1, v2, v3]


# --- Scene fixture ---------------------------------------------------------
#
# One isolated tall building on flat terrain. Odd grid so the building sits on
# the exact centre cell and a shadow can run a long way in any direction.
_N = 41
_NZ = 12
_C = _N // 2
_BUILDING_H = 6
# tan(30 deg) -> shadow reaches ~10.4 cells from the building edge; comfortably
# inside the grid in every direction, so no assertion depends on clipping.
_SUN_ELEVATION = 30.0


def _scene_voxels():
    vox = np.zeros((_N, _N, _NZ), dtype=np.int32)
    vox[:, :, 0] = 1  # terrain
    vox[_C - 1:_C + 2, _C - 1:_C + 2, 1:1 + _BUILDING_H] = -3  # building
    return vox


def _assemble(rect, extras=None):
    """A VoxCity built the way ``voxcitygml.generate_voxcity`` builds one."""
    pipeline = VoxCityPipeline(meshsize=1.0, rectangle_vertices=rect)
    flat = np.zeros((_N, _N), dtype=float)
    return pipeline.assemble_voxcity(
        voxcity_grid=_scene_voxels(),
        building_height_grid=flat.copy(),
        building_min_height_grid=flat.copy(),
        building_id_grid=np.zeros((_N, _N), dtype=np.int32),
        land_cover_grid=flat.copy(),
        dem_grid=flat.copy(),
        extras=extras,
    )


def _shadow_bearing(city, azimuth_deg):
    """Bearing (degrees, clockwise from grid axis 0) of the shadow's centroid.

    Axis 0 is grid-north and axis 1 is grid-east, so ``atan2(east, north)``
    over the centroid offset from the building gives a clockwise bearing in the
    *grid* frame. Returns ``(bearing, n_shadow_cells)``.
    """
    irradiance = get_direct_solar_irradiance_map(
        city,
        azimuth_degrees_ori=azimuth_deg,
        elevation_degrees=_SUN_ELEVATION,
        direct_normal_irradiance=1000.0,
        view_point_height=0.0,
    )
    # Building columns come back NaN (no walkable ground); lit ground is > 0.
    rows, cols = np.nonzero(irradiance == 0.0)
    return (
        math.degrees(math.atan2(cols.mean() - _C, rows.mean() - _C)),
        len(rows),
    )


class TestAssembleVoxcityDerivesRotationAngle:
    def test_axis_aligned_rectangle_gives_zero(self):
        city = _assemble(_rotated_rect(0.0))
        assert city.extras["rotation_angle"] == 0

    def test_rotated_rectangle_gives_the_computed_angle(self):
        rect = _rotated_rect(30.0)
        city = _assemble(rect)
        # Agrees with the documented single producer...
        assert city.extras["rotation_angle"] == compute_rotation_angle(rect)
        # ...and independently has the right sign and magnitude, so a swapped
        # argument or a negated convention cannot hide behind that agreement.
        assert city.extras["rotation_angle"] == pytest.approx(30.0, abs=1e-3)

    def test_counterclockwise_rotation_gives_a_negative_angle(self):
        city = _assemble(_rotated_rect(-25.0))
        assert city.extras["rotation_angle"] == pytest.approx(-25.0, abs=1e-3)

    def test_explicit_extras_rotation_angle_wins(self):
        # The ``extras`` argument is merged over the defaults; an explicitly
        # supplied angle must keep beating the derived one.
        city = _assemble(_rotated_rect(30.0), extras={"rotation_angle": 12.5})
        assert city.extras["rotation_angle"] == 12.5

    def test_derivation_does_not_disturb_the_other_extras(self):
        rect = _rotated_rect(30.0)
        city = _assemble(rect, extras={"land_cover_source": "OpenStreetMap"})
        assert city.extras["rectangle_vertices"] == rect
        assert city.extras["land_cover_source"] == "OpenStreetMap"
        assert "canopy_top" in city.extras and "canopy_bottom" in city.extras


class TestSunAzimuthConvention:
    """Pin the azimuth convention empirically before relying on it below.

    ``compute_sun_direction`` returns ``(cos_elev*cos az, cos_elev*sin az,
    sin_elev)`` on array axes 0/1/2 and the kernel marches *toward* the sun, so
    ``azimuth_degrees_ori`` is the sun's own compass bearing and needs no +180
    correction. These tests assert that from the outside rather than resting on
    the derivation: flip the azimuth and the shadow must swap sides.
    """

    @pytest.mark.parametrize(
        "azimuth, expected_shadow_bearing",
        [
            (0.0, 180.0),    # sun due north  -> shadow due south
            (90.0, -90.0),   # sun due east   -> shadow due west
            (180.0, 0.0),    # sun due south  -> shadow due north
            (270.0, 90.0),   # sun due west   -> shadow due east
        ],
    )
    def test_shadow_falls_opposite_the_sun(self, azimuth, expected_shadow_bearing):
        city = _assemble(_rotated_rect(0.0))
        bearing, n_cells = _shadow_bearing(city, azimuth)
        assert n_cells > 0, "no shadow at all — the fixture proves nothing"
        assert bearing == pytest.approx(expected_shadow_bearing, abs=1.0)

    def test_flipping_the_azimuth_swaps_the_shadow(self):
        city = _assemble(_rotated_rect(0.0))
        east_sun, _ = _shadow_bearing(city, 90.0)
        west_sun, _ = _shadow_bearing(city, 270.0)
        assert abs(abs(east_sun - west_sun) - 180.0) < 1.0


class TestRotatedAoiShadowLandsCorrectly:
    """The test that proves the bug is fixed where it mattered.

    Uses the **CPU** simulator (``voxcity.simulator.solar.radiation``), the twin
    of the GPU volumetric path; both read ``extras['rotation_angle']`` with the
    same ``.get(..., 0)`` default, so the defect and the fix are identical on
    each. The CPU path is chosen because it needs no Taichi backend.

    With the AOI rotated 30 degrees clockwise, a sun at geographic azimuth 90
    (due east) must cast its shadow due geographic west — which in the grid
    frame is bearing -90 - 30 = -120. Without the angle the simulator sees
    rotation 0 and puts the shadow at -90: the shadow is rotated by the AOI's
    own angle. Measured before the fix: -90.00. After: -119.93.
    """

    ROTATION = 30.0
    SUN_AZIMUTH = 90.0  # due east, geographic

    def _city(self):
        return _assemble(_rotated_rect(self.ROTATION))

    def test_shadow_bearing_is_rotated_into_the_grid_frame(self):
        bearing, n_cells = _shadow_bearing(self._city(), self.SUN_AZIMUTH)
        assert n_cells > 0
        # Geographic shadow bearing is azimuth + 180 == -90; expressed in a
        # frame rotated ROTATION degrees clockwise it is -90 - ROTATION.
        assert bearing == pytest.approx(-90.0 - self.ROTATION, abs=1.5)

    def test_a_cell_the_unrotated_sun_would_shade_is_lit(self):
        # Due grid-west of the building: shadowed iff the angle never arrived.
        irradiance = get_direct_solar_irradiance_map(
            self._city(),
            azimuth_degrees_ori=self.SUN_AZIMUTH,
            elevation_degrees=_SUN_ELEVATION,
            direct_normal_irradiance=1000.0,
            view_point_height=0.0,
        )
        assert irradiance[_C, _C - 6] > 0.0
        # ...and the cell the correctly-rotated shadow does reach is dark.
        assert irradiance[_C - 3, _C - 5] == 0.0

    def test_the_angle_actually_reached_the_simulator(self):
        """Same scene, angle supplied by hand — must agree with the derived one.

        Guards against a fix that stores the angle somewhere the simulator does
        not read.
        """
        by_hand = _assemble(
            _rotated_rect(self.ROTATION),
            extras={"rotation_angle": compute_rotation_angle(_rotated_rect(self.ROTATION))},
        )
        derived_bearing, _ = _shadow_bearing(self._city(), self.SUN_AZIMUTH)
        by_hand_bearing, _ = _shadow_bearing(by_hand, self.SUN_AZIMUTH)
        assert derived_bearing == pytest.approx(by_hand_bearing, abs=1e-6)


class TestEveryGeneratorConstructorSetsIt:
    """Regression guard on the "single producer" claim.

    ``compute_rotation_angle``'s docstring says it is the single producer of
    ``extras['rotation_angle']``. That claim was false for two years because a
    second constructor path existed and nothing noticed. This inventory makes a
    *third* one fail loudly instead.
    """

    # site -> why the angle is guaranteed there.
    KNOWN_SITES = {
        ("pipeline.py", "assemble_voxcity"):
            "derives it from self.rectangle_vertices (tested above)",
        ("update.py", "update_voxcity"):
            "copies extras from the input city (dict(city.extras)); the input "
            "came from one of the other constructors",
    }

    @staticmethod
    def _construction_sites():
        generator_dir = Path(__file__).resolve().parents[1] / "src" / "voxcity" / "generator"
        sites = set()
        for path in sorted(generator_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(func):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "VoxCity"
                    ):
                        sites.add((path.name, func.name))
        return sites

    def test_no_unaudited_voxcity_constructor(self):
        found = self._construction_sites()
        assert found == set(self.KNOWN_SITES), (
            "a VoxCity construction site in src/voxcity/generator/ changed. "
            "Every such path must populate extras['rotation_angle'] — a model "
            "without it silently rotates solar shadows by the AOI's angle. Add "
            "the site to KNOWN_SITES with a test that pins the angle."
        )

    def test_the_audited_pipeline_site_really_sets_it(self):
        # Belt and braces: the inventory above is static, this is not.
        for angle in (0.0, 17.0, -42.0):
            city = _assemble(_rotated_rect(angle))
            assert "rotation_angle" in city.extras
            assert city.extras["rotation_angle"] == pytest.approx(angle, abs=1e-3)

    def test_update_voxcity_preserves_a_supplied_angle(self):
        # Pins the mechanism KNOWN_SITES relies on for update.py, without
        # dragging in the whole update pipeline: extras are copied wholesale.
        import inspect

        from voxcity.generator import update

        source = inspect.getsource(update.update_voxcity)
        assert "new_extras = dict(city.extras)" in source, (
            "update_voxcity no longer copies the input extras wholesale; "
            "rotation_angle may no longer propagate."
        )
        assert "new_extras[\"rotation_angle\"]" not in source, (
            "update_voxcity now writes rotation_angle itself; audit it."
        )
