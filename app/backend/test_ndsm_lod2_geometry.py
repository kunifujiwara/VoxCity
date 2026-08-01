"""Dataset-gated regression: nDSM refinement must not flatten LOD2 geometry.

This claim has been re-derived by hand, with a different metric, every time
someone has touched this path -- and the metric quoted in the earlier comments
(``roof_slope_fraction``) lives in the sibling VoxCityGML repository, so it is
not available here to check anything against. This file pins the claim in-repo,
with the metric defined below, so the next change to ``ndsm_pipeline`` either
keeps it or turns red.

What is at stake: an LOD2 voxel grid is mesh-voxelized from real CityGML roof
and wall surfaces. The 2.5-D component grids (heights, min-heights, ids) cannot
describe that, so ``regenerate_voxels()`` -- which rebuilds the voxel grid from
those grids -- replaces true roofs with extruded footprints. The pipeline
therefore routes LOD2 through ``voxcitygml.reapply_canopy``, which overlays the
canopy onto the existing grid and touches nothing else.

Both halves are asserted, and the second is what makes the first mean anything:
run over the *same* model with ``building_lod`` retagged as 1, the refinement
takes the rebuild route and the roof geometry measurably collapses. Without that
control, a refinement that silently did nothing at all would pass.

Measured on the rectangle below at the time of writing: the LOD2 model carries
117,270 building voxels at a roof slope fraction of 0.2868; refining it through
``reapply_canopy`` leaves both untouched; refining the same model through
``regenerate_voxels`` gives 89,053 voxels at 0.2039 -- a quarter of the building
voxels and a third of the roof relief, gone.

Gated on the real Chuo-ku PLATEAU dataset and the real nDSM COG, both of which
are far too large to vendor; skipped when either is absent. Read-only: nothing
here writes to the dataset directory, its ``.voxcitygml_cache``, or the COG.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from backend import config
from backend.ndsm_pipeline import refine_canopy_with_ndsm

#: Voxel class of a building cell (``voxcity.generator.voxelizer.BUILDING_CODE``).
BUILDING_CODE = -3

# A ~350 x 390 m rectangle over Tsukiji, inside standard mesh tile 53393691
# (lat 35.65833-35.66667, lon 139.7625-139.775). Chosen because it is dense
# enough to carry both LOD2 roofs and land-cover trees, is covered by the nDSM
# COG, and lies in tiles the dataset's .voxcitygml_cache already holds -- so the
# fixture does not re-parse CityGML from scratch. [SW, NW, NE, SE], the app's
# vertex convention.
RECT = [
    [139.7680, 35.6620], [139.7680, 35.6655],
    [139.7730, 35.6655], [139.7730, 35.6620],
]
MESHSIZE = 2.0
LAND_COVER_SOURCE = "OpenEarthMapJapan"

#: Roof slope fraction a fixture must beat to count as LOD2 on this rectangle.
#: Measured here: ``building_lod=1`` scores 0.1490, ``building_lod=2`` scores
#: 0.2868. Extruded footprints are not flat as a population -- neighbouring
#: footprints of unequal height touch constantly in Chuo-ku -- so a threshold
#: near zero is satisfied by the very thing it is meant to exclude.
LOD1_SLOPE_CEILING = 0.22


def _voxcitygml_ready() -> bool:
    try:
        from voxcitygml import (  # noqa: F401
            VoxelizerConfig, generate_voxcity, reapply_canopy,
        )
    except ImportError:
        return False
    return True


_CITYGML_PATH = config.CITYGML_PATH
_HAVE_DATASET = bool(_CITYGML_PATH) and os.path.isdir(str(_CITYGML_PATH))
_HAVE_COG = bool(config.NDSM_COG_PATH) and os.path.exists(config.NDSM_COG_PATH)

pytestmark = pytest.mark.skipif(
    not (_HAVE_DATASET and _HAVE_COG and _voxcitygml_ready()),
    reason=(
        "needs the real PLATEAU CityGML dataset (CITYGML_PATH), the nDSM COG "
        "(NDSM_COG_PATH) and a voxcitygml with generate_voxcity + "
        "reapply_canopy"
    ),
)


# ---------------------------------------------------------------------------
# The roof-geometry metric
# ---------------------------------------------------------------------------
def roof_top_z(voxel_classes) -> np.ndarray:
    """Highest z index holding a building voxel, per column; -1 where none.

    ``voxel_classes`` is the ``(nx, ny, nz)`` grid from ``voxcity_obj.voxels``.
    """
    occ = np.asarray(voxel_classes) == BUILDING_CODE
    n_z = occ.shape[2]
    has_building = occ.any(axis=2)
    # argmax on the reversed z axis finds the topmost True; columns with no
    # building give argmax 0, which the where() below discards anyway.
    from_top = np.argmax(occ[:, :, ::-1], axis=2)
    return np.where(has_building, n_z - 1 - from_top, -1)


def roof_slope_fraction(voxel_classes) -> float:
    """Fraction of touching building-column pairs whose roofs sit at different z.

    Defined here, in this repository, rather than imported: the number quoted in
    this project's comments (0.1512 -> 0.0965 on Chuo-ku) came from a function
    in the sibling VoxCityGML repo that cannot be called from these tests, and a
    metric nobody can recompute is a metric nobody can check.

    Definition. Take every column that holds at least one building voxel and its
    topmost building z. Consider each 4-connected neighbouring pair of such
    columns. The metric is the share of those pairs whose two roof heights
    differ. Returns 0.0 when there are no pairs at all.

    Why it separates the two representations. Extruded footprints give one flat
    roof per building, so pairs differ only where two footprints of unequal
    height happen to touch -- a boundary effect, scaling with footprint
    perimeter. True LOD2 geometry has pitched, stepped and multi-level roofs, so
    pairs differ across the whole roof *area*. The measure is dimensionless and
    needs no reference model, which is what lets it be compared before and after
    a transformation of the same grid.

    It is deliberately not a proxy for "did anything change": two grids can
    share a slope fraction and differ everywhere. That is why the LOD2 assertion
    below compares the building voxels *bitwise* and uses this only as the
    named, human-readable summary the earlier comments were reaching for.
    """
    occ = np.asarray(voxel_classes) == BUILDING_CODE
    has_building = occ.any(axis=2)
    top = roof_top_z(voxel_classes)

    pairs = 0
    differing = 0
    for axis in (0, 1):
        lo = [slice(None), slice(None)]
        hi = [slice(None), slice(None)]
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        both = has_building[lo] & has_building[hi]
        pairs += int(both.sum())
        differing += int((both & (top[lo] != top[hi])).sum())
    return differing / pairs if pairs else 0.0


class TestTheMetricItself:
    """A metric that cannot fail cannot detect a flattening."""

    def test_a_flat_slab_has_no_slope(self):
        grid = np.zeros((5, 5, 8), dtype=np.int16)
        grid[1:4, 1:4, :4] = BUILDING_CODE
        assert roof_slope_fraction(grid) == 0.0

    def test_a_staircase_roof_is_all_slope(self):
        grid = np.zeros((4, 1, 8), dtype=np.int16)
        for i in range(4):
            grid[i, 0, :i + 2] = BUILDING_CODE
        assert roof_slope_fraction(grid) == 1.0

    def test_columns_without_buildings_are_not_pairs(self):
        grid = np.zeros((3, 1, 4), dtype=np.int16)
        grid[0, 0, :2] = BUILDING_CODE
        grid[2, 0, :3] = BUILDING_CODE        # not adjacent to the first
        assert roof_slope_fraction(grid) == 0.0

    def test_an_empty_grid_is_zero_not_a_zero_division(self):
        assert roof_slope_fraction(np.zeros((3, 3, 3), dtype=np.int16)) == 0.0

    def test_a_partly_stepped_roof_scores_strictly_between_zero_and_one(self):
        # Four columns in a row -> three pairs; only the last one steps.
        grid = np.zeros((4, 1, 8), dtype=np.int16)
        grid[0:3, 0, :3] = BUILDING_CODE
        grid[3, 0, :5] = BUILDING_CODE
        assert roof_slope_fraction(grid) == pytest.approx(1 / 3)

    def test_pairs_are_four_connected_not_eight(self):
        """Diagonal neighbours must not count.

        Two columns touching only at a corner: under 8-connectivity that is one
        pair and the metric would be 1.0; under 4-connectivity there are no
        pairs at all and it is 0.0. Nothing else in this file pins which one
        the definition means.
        """
        grid = np.zeros((2, 2, 6), dtype=np.int16)
        grid[0, 0, :2] = BUILDING_CODE
        grid[1, 1, :4] = BUILDING_CODE
        assert roof_slope_fraction(grid) == 0.0

    def test_both_axes_contribute_pairs(self):
        # A 2x2 block, one corner raised: two of the four 4-connected pairs
        # straddle the step -- one along each axis.
        grid = np.zeros((2, 2, 8), dtype=np.int16)
        grid[:, :, :3] = BUILDING_CODE
        grid[1, 1, :5] = BUILDING_CODE
        assert roof_slope_fraction(grid) == pytest.approx(0.5)

    def test_the_top_is_the_topmost_building_voxel(self):
        grid = np.zeros((1, 1, 6), dtype=np.int16)
        grid[0, 0, 1] = BUILDING_CODE
        grid[0, 0, 4] = BUILDING_CODE
        assert roof_top_z(grid)[0, 0] == 4


@pytest.fixture(scope="module")
def lod2_model():
    """One real LOD2 Chuo-ku model, generated once and handed out as copies.

    Module-scoped because generating it parses CityGML and voxelizes meshes;
    every test below deep-copies it before touching anything, so the fixture
    itself is never mutated.
    """
    from voxcitygml import VoxelizerConfig, generate_voxcity

    config_obj = VoxelizerConfig(
        citygml_path=str(_CITYGML_PATH),
        rectangle_vertices=[tuple(v) for v in RECT],
        meshsize=MESHSIZE,
        building_lod=2,
        land_cover_source=LAND_COVER_SOURCE,
        canopy_height_source="Static",
        static_tree_height=10.0,
        save_output=False,
        gridvis=False,
        include_bridges=False,
    )
    # CITYGML_PATH may legitimately point at some *other* city, or at a
    # directory of datasets rather than one dataset -- config.py's fallback is
    # ``<DATA_DIR>/plateau``, which on a dev box is often exactly that. Those
    # are "no dataset for this rectangle", i.e. the same condition the module
    # gate is testing for, so they skip rather than fail. Nothing else is
    # caught: a genuine voxcitygml fault must still surface.
    try:
        model = generate_voxcity(config_obj)
    except FileNotFoundError as exc:                    # no udx/bldg layout
        pytest.skip(f"CITYGML_PATH is not a single PLATEAU dataset: {exc}")
    except ValueError as exc:
        if "no citygml buildings" not in str(exc).lower():
            raise
        pytest.skip(
            f"the dataset at {_CITYGML_PATH} does not cover the Tsukiji test "
            f"rectangle: {exc}")
    if isinstance(getattr(model, "extras", None), dict):
        model.extras["building_lod"] = 2
    return model


def _refine(model) -> bool:
    return refine_canopy_with_ndsm(
        model, RECT, MESHSIZE, LAND_COVER_SOURCE, config.NDSM_COG_PATH,
        static_tree_height=10.0,
    )


class TestLod2GeometryIsPreserved:
    def test_the_fixture_actually_has_lod2_roofs(self, lod2_model):
        """Non-vacuity for everything below: a grid of flat-topped extrusions
        would make 'the roofs survived' true and meaningless.

        The threshold has to sit above what LOD1 actually scores, not above
        zero. Measured on this rectangle, ``building_lod=1`` gives 0.1490 --
        extruded footprints are far from flat as a *population*, because Chuo-ku
        packs buildings of differing heights against each other and every such
        contact is a differing pair. LOD2 gives 0.2868. An earlier version of
        this guard used 0.05 and passed happily on the LOD1 fixture, i.e. it
        could not catch the one failure it exists for: a voxcitygml that
        silently falls back to LOD1. 0.22 sits 48% above the LOD1 score and 23%
        below the LOD2 one.
        """
        vox = np.asarray(lod2_model.voxels.classes)
        assert (vox == BUILDING_CODE).sum() > 1000
        assert roof_slope_fraction(vox) > LOD1_SLOPE_CEILING, (
            "the fixture's roofs are no more varied than extruded footprints "
            "would be, so this file cannot detect a flattening -- has "
            "voxcitygml fallen back to LOD1?")

    def test_the_fixture_has_trees_to_refine(self, lod2_model):
        """And a refinement with nothing to refine proves nothing either."""
        from backend.ndsm_pipeline import _resolve_tree_id

        tree_id = _resolve_tree_id(LAND_COVER_SOURCE)
        n_tree = int((np.asarray(lod2_model.land_cover.classes) == tree_id).sum())
        assert n_tree > 0

    def test_building_voxels_survive_refinement_bitwise(self, lod2_model):
        model = copy.deepcopy(lod2_model)
        before = np.asarray(model.voxels.classes) == BUILDING_CODE
        before_slope = roof_slope_fraction(model.voxels.classes)

        assert _refine(model) is True, (
            "refinement was skipped, so this asserts nothing about the LOD2 "
            "route")

        after = np.asarray(model.voxels.classes) == BUILDING_CODE
        assert after.shape == before.shape
        assert np.array_equal(after, before), (
            f"{int((after != before).sum())} building voxels changed: the LOD2 "
            "grid was rebuilt from the 2.5-D component grids, which replaces "
            "real roof and wall geometry with extruded footprints")
        assert roof_slope_fraction(model.voxels.classes) == before_slope

    def test_the_canopy_really_was_applied(self, lod2_model):
        """Otherwise 'the buildings did not change' is satisfied by a no-op."""
        model = copy.deepcopy(lod2_model)
        before = np.asarray(model.tree_canopy.top).copy()
        assert _refine(model) is True
        after = np.asarray(model.tree_canopy.top)
        assert not np.array_equal(after, before), (
            "the canopy grid is unchanged, so the preservation assertions are "
            "trivially satisfied")


class TestLod1GenuinelyRebuilds:
    """The control. Same model, same refinement, the other write-back route."""

    def test_the_rebuild_route_changes_the_voxel_grid(self, lod2_model):
        model = copy.deepcopy(lod2_model)
        model.extras["building_lod"] = 1          # take the regenerate route
        before = np.asarray(model.voxels.classes).copy()

        assert _refine(model) is True

        after = np.asarray(model.voxels.classes)
        assert not np.array_equal(after, before), (
            "the LOD1 route must actually rebuild the voxel grid; if it does "
            "not, the LOD2 assertions above are comparing two no-ops")

    def test_the_rebuild_flattens_the_roofs(self, lod2_model):
        """The measurement the project's comments have been quoting: rebuilding
        from the 2.5-D grids replaces LOD2 roofs with extruded footprints, and
        the slope fraction falls."""
        model = copy.deepcopy(lod2_model)
        model.extras["building_lod"] = 1
        before = roof_slope_fraction(model.voxels.classes)

        assert _refine(model) is True

        after = roof_slope_fraction(model.voxels.classes)
        assert after < before, (
            f"roof slope fraction {before:.4f} -> {after:.4f}: the rebuild was "
            "expected to flatten the roofs")
