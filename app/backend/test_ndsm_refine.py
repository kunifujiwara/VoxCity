"""Unit tests for evidence pooling, groupwise percentile and local median.

These are the numerical foundation of the evidence-based nDSM canopy
refinement: per-cell pooling of LiDAR return statistics, exact per-cell
percentiles, and a neighbourhood median used to detect spikes.

Fixtures are deliberately asymmetric where orientation could matter, and the
pooling tests assert *exact* arithmetic rather than plausible ranges, because
the property the design rests on is that counts and sums are additive and
therefore pool without error at any runtime cell size.

Import form note: sibling modules in this package are imported as
``backend.<module>`` (see ``test_rectangle_from_dimensions.py`` etc.);
``app/backend/__init__.py`` makes ``backend`` a package, so pytest puts
``app/`` -- not ``app/backend/`` -- on ``sys.path``.
"""
import math
import warnings

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from backend.ndsm_refine import (
    PARTITION_KEYS,
    VERDICT_AMBIGUOUS_KEEP,
    VERDICT_AMBIGUOUS_REPLACE,
    VERDICT_CANOPY,
    VERDICT_NO_DATA,
    VERDICT_NONE,
    VERDICT_ROOF,
    RefineParams,
    _cell_labels,
    _pixel_lonlat,
    classify_and_refine,
    pool_evidence,
    groupwise_percentile,
    load_ndsm_evidence,
    local_tree_median,
    refine_from_evidence,
)
from voxcity.geoprocessor.raster.core import compute_cell_center_coords


# ---------------------------------------------------------------------------
# groupwise_percentile
# ---------------------------------------------------------------------------
class TestGroupwisePercentile:
    def test_exact_percentile_per_group(self):
        values = np.array([1.0, 9.0, 5.0, 3.0, 7.0, 100.0])
        groups = np.array([0, 0, 0, 1, 1, 2])
        out = groupwise_percentile(values, groups, n_groups=4, q=50)
        assert out[0] == 5.0            # median of {1,9,5}
        assert out[1] == 5.0            # median of {3,7}
        assert out[2] == 100.0          # single-element group
        assert np.isnan(out[3])         # empty group -> NaN

    def test_returns_float_array_of_n_groups(self):
        out = groupwise_percentile(np.array([1.0]), np.array([0]), n_groups=5, q=50)
        assert out.shape == (5,)
        assert out.dtype == np.float64

    def test_p90_matches_linear_interpolation(self):
        values = np.arange(10, dtype=float)   # 0..9 in one group
        groups = np.zeros(10, dtype=int)
        out = groupwise_percentile(values, groups, n_groups=1, q=90)
        # np.percentile's default 'linear' method: (10-1)*0.9 = 8.1
        assert out[0] == pytest.approx(8.1)

    def test_q0_and_q100_are_min_and_max(self):
        values = np.array([5.0, -2.0, 11.0, 0.0])
        groups = np.zeros(4, dtype=int)
        assert groupwise_percentile(values, groups, 1, q=0)[0] == -2.0
        assert groupwise_percentile(values, groups, 1, q=100)[0] == 11.0

    def test_nan_values_are_ignored(self):
        values = np.array([np.nan, 4.0, np.nan, 6.0])
        groups = np.array([0, 0, 1, 0])
        out = groupwise_percentile(values, groups, n_groups=2, q=50)
        assert out[0] == 5.0            # NaN dropped, median of {4,6}
        assert np.isnan(out[1])         # only-NaN group

    def test_infinities_are_treated_as_invalid(self):
        # +/-inf is not a physical height; it must not poison a group median.
        values = np.array([np.inf, 4.0, 6.0, -np.inf])
        groups = np.array([0, 0, 0, 1])
        out = groupwise_percentile(values, groups, n_groups=2, q=50)
        assert out[0] == 5.0
        assert np.isnan(out[1])

    def test_interleaved_unsorted_groups(self):
        # Input order is scrambled in both group and value: an implementation
        # that assumes contiguous or ascending group runs must fail here.
        values = np.array([9.0, 1.0, 30.0, 5.0, 10.0, 20.0])
        groups = np.array([1, 0, 2, 1, 2, 2])
        out = groupwise_percentile(values, groups, n_groups=3, q=50)
        assert out[0] == 1.0            # {1}
        assert out[1] == 7.0            # {9,5} -> 7
        assert out[2] == 20.0           # {30,10,20} -> 20

    def test_out_of_range_group_labels_are_ignored(self):
        # The raster reader labels pixels that fall outside the model grid with
        # a negative sentinel; labels >= n_groups can arise from clipping.
        values = np.array([1.0, 2.0, 999.0, 999.0])
        groups = np.array([0, 0, -1, 7])
        out = groupwise_percentile(values, groups, n_groups=2, q=50)
        assert out[0] == 1.5
        assert np.isnan(out[1])

    def test_empty_input_all_nan(self):
        out = groupwise_percentile(
            np.array([], dtype=float), np.array([], dtype=int), n_groups=3, q=50
        )
        assert out.shape == (3,)
        assert np.isnan(out).all()

    def test_integer_values_accepted(self):
        out = groupwise_percentile(
            np.array([1, 2, 3, 4]), np.array([0, 0, 0, 0]), n_groups=1, q=50
        )
        assert out[0] == 2.5

    def test_matches_numpy_percentile_on_random_data(self):
        # Reference parity: whatever vectorized slicing trick is used must agree
        # with np.percentile group by group.
        rng = np.random.default_rng(7)
        n_groups = 12
        values = rng.normal(10.0, 5.0, 500)
        values[rng.random(500) < 0.1] = np.nan
        groups = rng.integers(0, n_groups, 500)
        for q in (25, 50, 90, 99):
            out = groupwise_percentile(values, groups, n_groups, q)
            for g in range(n_groups):
                sel = values[(groups == g) & np.isfinite(values)]
                if sel.size:
                    assert out[g] == pytest.approx(np.percentile(sel, q))
                else:
                    assert np.isnan(out[g])

    @pytest.mark.parametrize("q", [150, -10, 100.5, np.nan])
    def test_q_outside_0_100_is_rejected(self, q):
        # q=150 would index past the end of a group's run; q=-10 is worse --
        # it extrapolates below the minimum and returns a plausible number
        # (1.1 for values [1, 2]) with no error at all.
        with pytest.raises(ValueError, match="q must be in"):
            groupwise_percentile(np.array([1.0, 2.0]), np.array([0, 0]), 1, q)

    def test_q_bounds_are_inclusive(self):
        values, groups = np.array([1.0, 2.0]), np.array([0, 0])
        assert groupwise_percentile(values, groups, 1, 0)[0] == 1.0
        assert groupwise_percentile(values, groups, 1, 100)[0] == 2.0

    def test_float_group_labels_are_rejected(self):
        # Task 2 derives labels from coordinates; a float array there means a
        # missing floor, and astype(intp) would silently truncate 2.7 -> cell 2.
        with pytest.raises(TypeError, match="integer array"):
            groupwise_percentile(np.array([1.0, 2.0]), np.array([0.0, 2.7]), 3, 50)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            groupwise_percentile(np.array([1.0, 2.0]), np.array([0]), 1, 50)

    def test_does_not_mutate_inputs(self):
        values = np.array([3.0, np.nan, 1.0])
        groups = np.array([0, 0, 0])
        groupwise_percentile(values, groups, 1, 50)
        assert np.isnan(values[1]) and values[0] == 3.0 and values[2] == 1.0
        assert groups.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# pool_evidence
# ---------------------------------------------------------------------------
def _empty(n=5):
    return [np.array([], dtype=float) for _ in range(n)]


class TestPoolEvidence:
    def test_ratios_and_roughness_pool_exactly(self):
        # two pixels in cell 0: (n_all, n_multi, n_ng, sz, sz2)
        #   px A: 4 pts, 1 multi, 3 non-ground, heights {2,2,2} -> sz=6,  sz2=12
        #   px B: 4 pts, 3 multi, 1 non-ground, height  {10}    -> sz=10, sz2=100
        n_all = np.array([4.0, 4.0])
        n_multi = np.array([1.0, 3.0])
        n_ng = np.array([3.0, 1.0])
        sz = np.array([6.0, 10.0])
        sz2 = np.array([12.0, 100.0])
        groups = np.array([0, 0])
        ev = pool_evidence(groups, 1, n_all, n_multi, n_ng, sz, sz2)
        assert ev["mrf"][0] == pytest.approx(4.0 / 8.0)
        # pooled non-ground: n=4, mean=4, E[z^2]=112/4=28 -> std = sqrt(28-16)
        assert ev["roughness"][0] == pytest.approx(np.sqrt(12.0))
        assert ev["n_all"][0] == 8

    def test_output_keys_and_shapes(self):
        ev = pool_evidence(np.array([0]), 4, *[np.array([1.0]) for _ in range(5)])
        assert set(ev) == {"mrf", "roughness", "n_all", "n_nonground"}
        for key in ev:
            assert ev[key].shape == (4,), key

    def test_pooled_nonground_count_is_returned(self):
        # The classifier weights roughness confidence by this count, so it has
        # to come back alongside the statistic it qualifies.
        ev = pool_evidence(
            np.array([0, 0]), 1,
            np.array([4.0, 4.0]), np.array([1.0, 3.0]), np.array([3.0, 1.0]),
            np.array([6.0, 10.0]), np.array([12.0, 100.0]),
        )
        assert ev["n_nonground"][0] == 4

    def test_roughness_needs_two_nonground_points(self):
        # n=1: the population std is *defined* (0.0) but says nothing about
        # dispersion -- and 0.0 is the maximally roof-like value, so reporting
        # it would let a single pulse clipping a thin tree top be capped as a
        # roof. That is the exact false-cap failure this module removes.
        def _roughness(n_nonground, sum_z, sum_z2):
            ev = pool_evidence(
                np.array([0]), 1,
                np.array([10.0]), np.array([5.0]),
                np.array([float(n_nonground)]),
                np.array([sum_z]), np.array([sum_z2]),
            )
            return ev["roughness"][0]

        assert np.isnan(_roughness(1, 20.0, 400.0))
        # n=2 at heights {8, 12}: mean 10, E[z^2] = 104 -> std = 2
        assert _roughness(2, 20.0, 208.0) == pytest.approx(2.0)

    def test_roughness_nan_when_nonground_split_across_pixels_totals_one(self):
        # The guard is on the *pooled* count, not the per-pixel count: two
        # pixels holding half a point each is not a thing, but two pixels where
        # only one holds the single non-ground point is.
        ev = pool_evidence(
            np.array([0, 0]), 1,
            np.array([5.0, 5.0]), np.array([1.0, 1.0]), np.array([1.0, 0.0]),
            np.array([9.0, 0.0]), np.array([81.0, 0.0]),
        )
        assert np.isnan(ev["roughness"][0])
        assert ev["n_nonground"][0] == 1

    def test_pooling_is_additive_split_pixels_match_single_pixel(self):
        # THE property the design rests on: the same points, split across any
        # number of pixels, must pool to bit-comparable evidence. (This is why
        # the COG carries counts and sums, not pre-computed ratios.)
        rng = np.random.default_rng(3)
        z = rng.uniform(0.0, 30.0, 40)
        multi = rng.random(40) < 0.4
        one = pool_evidence(
            np.array([0]), 1,
            np.array([40.0]), np.array([float(multi.sum())]),
            np.array([40.0]), np.array([z.sum()]), np.array([(z ** 2).sum()]),
        )
        # same points cut into four uneven pixels
        cuts = [0, 7, 7, 25, 40]      # note the empty pixel in the middle
        parts = [slice(cuts[i], cuts[i + 1]) for i in range(4)]
        split = pool_evidence(
            np.zeros(4, dtype=int), 1,
            np.array([float(z[s].size) for s in parts]),
            np.array([float(multi[s].sum()) for s in parts]),
            np.array([float(z[s].size) for s in parts]),
            np.array([z[s].sum() for s in parts]),
            np.array([(z[s] ** 2).sum() for s in parts]),
        )
        assert split["mrf"][0] == pytest.approx(one["mrf"][0])
        assert split["roughness"][0] == pytest.approx(one["roughness"][0])
        assert split["n_all"][0] == pytest.approx(one["n_all"][0])
        # and against the direct definition
        assert one["roughness"][0] == pytest.approx(float(np.std(z)))
        assert one["mrf"][0] == pytest.approx(multi.mean())

    def test_no_pixels_at_all_yields_nan_evidence(self):
        ev = pool_evidence(np.array([], dtype=int), 2, *_empty())
        assert np.isnan(ev["mrf"]).all()
        assert np.isnan(ev["roughness"]).all()
        assert (ev["n_all"] == 0).all()

    def test_present_pixels_with_zero_counts_yield_nan_evidence(self):
        # The realistic empty cell: pixels exist inside it but hold no returns
        # (water, occlusion, flight-line gap). Must be NaN, not 0/0 -> warning.
        groups = np.array([0, 0, 1])
        zeros = np.zeros(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # no divide-by-zero warnings allowed
            ev = pool_evidence(groups, 2, zeros, zeros, zeros, zeros, zeros)
        assert np.isnan(ev["mrf"]).all()
        assert np.isnan(ev["roughness"]).all()
        assert (ev["n_all"] == 0).all()

    def test_nonground_zero_but_returns_present(self):
        # All returns are ground: mrf is defined, roughness has no sample.
        ev = pool_evidence(
            np.array([0]), 1,
            np.array([10.0]), np.array([2.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]),
        )
        assert ev["mrf"][0] == pytest.approx(0.2)
        assert np.isnan(ev["roughness"][0])
        assert ev["n_all"][0] == 10

    def test_planar_surface_has_zero_roughness(self):
        # 5 returns all at exactly 12.0 -> variance 0, not NaN, not negative.
        ev = pool_evidence(
            np.array([0]), 1,
            np.array([5.0]), np.array([0.0]), np.array([5.0]),
            np.array([60.0]), np.array([720.0]),
        )
        assert ev["roughness"][0] == 0.0
        assert ev["mrf"][0] == 0.0

    def test_variance_clamped_when_cancellation_goes_negative(self):
        # Sums arrive pre-accumulated from the raster, so E[z^2] can land a hair
        # below mean^2. Without the clamp this is sqrt(negative) -> NaN + warning.
        n = 2.0
        sum_z = 20.0
        sum_z2 = (sum_z ** 2) / n - 1e-9    # deliberately too small
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # invalid value in sqrt would raise
            ev = pool_evidence(
                np.array([0]), 1,
                np.array([2.0]), np.array([0.0]), np.array([n]),
                np.array([sum_z]), np.array([sum_z2]),
            )
        assert ev["roughness"][0] == 0.0

    def test_interleaved_and_out_of_range_groups(self):
        # Scrambled order plus a sentinel pixel that belongs to no cell.
        groups = np.array([1, 0, -1, 1, 0])
        n_all = np.array([1.0, 2.0, 500.0, 3.0, 4.0])
        n_multi = np.array([1.0, 0.0, 500.0, 0.0, 0.0])
        n_ng = np.array([1.0, 2.0, 500.0, 3.0, 4.0])
        sz = np.array([1.0, 2.0, 5e5, 3.0, 4.0])
        sz2 = np.array([1.0, 2.0, 5e7, 3.0, 4.0])
        ev = pool_evidence(groups, 2, n_all, n_multi, n_ng, sz, sz2)
        assert ev["n_all"][0] == 6      # 2 + 4, sentinel excluded
        assert ev["n_all"][1] == 4      # 1 + 3
        assert ev["mrf"][0] == 0.0
        assert ev["mrf"][1] == pytest.approx(0.25)

    def test_float_group_labels_are_rejected(self):
        with pytest.raises(TypeError, match="integer array"):
            pool_evidence(np.array([0.0]), 1, *[np.array([1.0])] * 5)

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same length as groups"):
            pool_evidence(
                np.array([0, 0]), 1,
                np.array([1.0, 1.0]), np.array([1.0]), np.array([1.0, 1.0]),
                np.array([1.0, 1.0]), np.array([1.0, 1.0]),
            )

    def test_does_not_mutate_inputs(self):
        args = [np.array([4.0]), np.array([1.0]), np.array([4.0]),
                np.array([8.0]), np.array([20.0])]
        snapshot = [a.copy() for a in args]
        pool_evidence(np.array([0]), 1, *args)
        for a, s in zip(args, snapshot):
            assert np.array_equal(a, s)


# ---------------------------------------------------------------------------
# local_tree_median
# ---------------------------------------------------------------------------
def _fixture_grid():
    """11x11 grid, radius-2 window (5x5). Trees:

      (2,2)=10 and (2,3)=20  -- an adjacent pair
      (8,8)=99               -- isolated, far outside the pair's windows
    """
    can = np.zeros((11, 11))
    mask = np.zeros((11, 11), bool)
    can[2, 2], can[2, 3], can[8, 8] = 10.0, 20.0, 99.0
    mask[2, 2] = mask[2, 3] = mask[8, 8] = True
    return can, mask


class TestLocalTreeMedian:
    def test_median_over_window_masked(self):
        can, mask = _fixture_grid()
        med = local_tree_median(can, mask, radius=2)
        # window rows 0..4, cols 0..4 -> {10, 20}; the 99 at (8,8) is outside
        assert med[2, 2] == 15.0
        assert med[2, 3] == 15.0

    def test_window_with_exactly_one_tree(self):
        can, mask = _fixture_grid()
        med = local_tree_median(can, mask, radius=2)
        # rows 6..10, cols 6..10 contains only (8,8)
        assert med[8, 8] == 99.0
        # rows -2..2, cols -2..2 reaches (2,2) but not (2,3)
        assert med[0, 0] == 10.0

    def test_window_with_no_valid_tree_is_nan(self):
        can, mask = _fixture_grid()
        med = local_tree_median(can, mask, radius=2)
        # rows 3..7, cols 3..7 contains none of the three trees
        assert np.isnan(med[5, 5])
        assert np.isnan(med[10, 0])

    def test_unmasked_cells_do_not_contribute(self):
        # A huge non-tree value adjacent to a tree must not move the median.
        can, mask = _fixture_grid()
        can[3, 3] = 1000.0          # left unmasked
        med = local_tree_median(can, mask, radius=2)
        assert med[2, 2] == 15.0

    def test_padding_does_not_inject_zeros_at_edges(self):
        can = np.zeros((5, 5))
        mask = np.zeros((5, 5), bool)
        can[0, 0] = 12.0
        mask[0, 0] = True
        med = local_tree_median(can, mask, radius=1)
        # If the pad value were 0 instead of NaN, this would be 0.0.
        assert med[0, 0] == 12.0
        assert med[1, 1] == 12.0

    def test_nan_canopy_at_masked_cell_is_ignored(self):
        can = np.zeros((5, 5))
        mask = np.zeros((5, 5), bool)
        can[2, 2], can[2, 3] = np.nan, 8.0
        mask[2, 2] = mask[2, 3] = True
        med = local_tree_median(can, mask, radius=1)
        assert med[2, 2] == 8.0      # NaN member dropped, not propagated

    def test_radius_zero_is_the_cell_itself(self):
        can, mask = _fixture_grid()
        med = local_tree_median(can, mask, radius=0)
        assert med[2, 2] == 10.0
        assert med[2, 3] == 20.0
        assert np.isnan(med[0, 0])
        assert np.isnan(med).sum() == med.size - 3

    def test_orientation_symmetry(self):
        # Non-square on purpose: against a square grid a row/column mix-up in
        # the shift indices still produces a conformable array, so the fixture
        # has to be rectangular for this test to have teeth.
        can = np.zeros((9, 13))
        mask = np.zeros((9, 13), bool)
        for (row, col), value in {(1, 2): 10.0, (1, 4): 20.0,
                                  (7, 11): 99.0, (3, 0): 5.0}.items():
            can[row, col] = value
            mask[row, col] = True
        a = local_tree_median(can, mask, radius=2).T
        b = local_tree_median(can.T, mask.T, radius=2)
        assert a.shape == (13, 9)
        assert np.allclose(a, b, equal_nan=True)

    def test_shape_and_no_input_mutation(self):
        can, mask = _fixture_grid()
        can_snap, mask_snap = can.copy(), mask.copy()
        med = local_tree_median(can, mask, radius=2)
        assert med.shape == can.shape
        assert med.dtype == np.float64
        assert np.array_equal(can, can_snap)
        assert np.array_equal(mask, mask_snap)

    def test_all_nan_window_emits_no_runtime_warning(self):
        # Empty windows are expected and return NaN by design; the numpy
        # "All-NaN slice encountered" warning must be suppressed narrowly.
        can = np.zeros((9, 9))
        mask = np.zeros((9, 9), bool)
        mask[0, 0] = True
        can[0, 0] = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            med = local_tree_median(can, mask, radius=2)
        assert np.isnan(med[8, 8])

    def test_suppression_is_not_global(self):
        # The narrow suppression must not leak into the caller's warning filters.
        #
        # Asserting on a *later* warning cannot detect a leak: pytest.warns (and
        # any catch_warnings) installs its own filter stack, wiping the leaked
        # state before the assertion runs. Compare the filter list itself.
        can, mask = _fixture_grid()
        before = list(warnings.filters)
        local_tree_median(can, mask, radius=2)
        assert warnings.filters == before

    def test_is_vectorized_single_nanmedian_call(self, monkeypatch):
        # Structural check for the timing test below: a per-cell Python loop
        # would call nanmedian H*W times. The vectorized stack calls it once.
        calls = []
        real = np.nanmedian

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(np, "nanmedian", counting)
        can, mask = _fixture_grid()
        local_tree_median(can, mask, radius=2)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# load_ndsm_evidence
#
# The COG is north-up raster space (row 0 = north); the model grid is anchored
# at rectangle_vertices[0] with axis 0 along side_1, which for the [SW, NW, NE,
# SE] rectangles below puts row 0 on the SOUTH. A windowed read carries the
# raster's orientation, so the fixture is asymmetric along BOTH axes and the
# assertions pin the tall band to the SOUTH rows and the multi-return evidence
# to the SOUTH-WEST corner. A symmetric fixture would pass whether or not the
# reader flipped, which is exactly how the last two-frame bug survived four
# review passes.
#
# Load-level assertions alone are not enough: a rotation-blind labelling off the
# lat/lon bounding box satisfies them too. test_cell_centres_round_trip_to_their
# _own_cells pins the geometry directly, and the rotated fixtures paint their
# bands in the rectangle's own frame so a bbox mapping smears them.
# ---------------------------------------------------------------------------
EPSG = 6677                      # JGD2011 / Japan Plane Rectangular CS IX
PIXEL = 0.5                      # metres, as in the real nDSM COG
MESHSIZE = 2.0                   # -> 4x4 raster pixels per model cell
SIDE_M = 40.0                    # -> 20x20 model cells
NODATA = -9999.0

_W, _S = 139.7600, 35.6800


def _axis_aligned_rect(north_m, east_m):
    """``[SW, NW, NE, SE]``: compute_grid_geometry takes v0 as the origin,
    side_1 = v1 - v0 (northward, the row axis) and side_2 = v3 - v0 (eastward,
    the column axis)."""
    north = _S + north_m / 110946.0
    east = _W + east_m / (111320.0 * math.cos(math.radians(_S)))
    return [(_W, _S), (_W, north), (east, north), (east, _S)]


RECT = _axis_aligned_rect(SIDE_M, SIDE_M)
# Non-square on purpose: against a square grid, transposing n_rows and n_cols is
# a literal no-op, so every shape and reshape below would be unguarded.
RECT_TALL = _axis_aligned_rect(60.0, 40.0)          # -> 30 rows x 20 cols


def _rect_fractions(lon, lat, rect):
    """Position within *rect*'s own frame: (along side_1, along side_2) in [0, 1].

    Inverts the rectangle parallelogram with ``np.linalg.solve``, not the
    reader's Cramer expansion, so the fixture places its bands independently of
    the code under test -- and honours rotation, so a rotated rectangle gets its
    bands painted along its own edges rather than a lat/lon bounding box.

    For an axis-aligned rectangle the two fractions are simply northward and
    eastward.
    """
    v0 = np.asarray(rect[0], dtype=float)
    axes = np.column_stack([
        np.asarray(rect[1], dtype=float) - v0,      # side_1
        np.asarray(rect[3], dtype=float) - v0,      # side_2
    ])
    shape = np.shape(lon)
    delta = np.stack([
        np.asarray(lon, dtype=float).ravel() - v0[0],
        np.asarray(lat, dtype=float).ravel() - v0[1],
    ])
    frac_1, frac_2 = np.linalg.solve(axes, delta)
    return frac_1.reshape(shape), frac_2.reshape(shape)


def _rotated(rect, degrees):
    """Rotate *rect* about its centroid by *degrees* in a locally metric frame."""
    if not degrees:
        return list(rect)
    points = np.asarray(rect, dtype=float)
    cx, cy = points.mean(axis=0)
    scale = math.cos(math.radians(cy))
    theta = math.radians(degrees)
    dx, dy = (points[:, 0] - cx) * scale, points[:, 1] - cy
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)
    return [(cx + x / scale, cy + y) for x, y in zip(rx, ry)]


def _write_synthetic_cog(
    path,
    *,
    bands=6,
    tall_south=True,
    dtype="float32",
    uniform=False,
    spike=False,
    nodata_ne=False,
    rect=RECT,
):
    """Write an evidence COG covering *rect* with a margin, 0.5 m pixels.

    Content is placed in *rect*'s own frame (see :func:`_rect_fractions`), so
    "the first quarter along side_1" is the southern quarter for the axis-aligned
    rectangles and the corresponding rotated strip otherwise. Named below by the
    axis-aligned reading, which is what the positional assertions use.

    Band 1 (nDSM) is 20 m in the SOUTHERN quarter of the rectangle and 5 m
    elsewhere -- asymmetric north<->south, so a flipped reader cannot pass.

    Band 3 (n_multi) is high only in the SOUTH-WEST quarter: a second,
    *independent* asymmetry, so the evidence bands' orientation is checked in
    its own right and not merely inherited from the height band.

    The NORTH-EAST quarter carries n_nonground = 0, so its pooled count is
    below the two-point minimum and roughness must come back NaN there.

    ``tall_south=False`` mirrors band 1 to the north -- the teeth-check for the
    frame assertions. ``spike=True`` makes band 1 flat 5 m with a single 30 m
    pixel at the rectangle centre: the contaminated-pixel mechanism the whole
    aggregation exists to reject.
    """
    to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)

    xs, ys = to_xy.transform([v[0] for v in rect], [v[1] for v in rect])
    margin = 5.0
    left = math.floor((min(xs) - margin) / PIXEL) * PIXEL
    bottom = math.floor((min(ys) - margin) / PIXEL) * PIXEL
    right = math.ceil((max(xs) + margin) / PIXEL) * PIXEL
    top = math.ceil((max(ys) + margin) / PIXEL) * PIXEL
    width = int(round((right - left) / PIXEL))
    height = int(round((top - bottom) / PIXEL))
    transform = from_origin(left, top, PIXEL, PIXEL)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    px = left + (cols + 0.5) * PIXEL
    py = top - (rows + 0.5) * PIXEL
    lon, lat = to_ll.transform(px, py)
    f_north, f_east = _rect_fractions(lon, lat, rect)

    if uniform or spike:
        ndsm = np.full((height, width), 5.0)
    else:
        tall = f_north < 0.25 if tall_south else f_north > 0.75
        ndsm = np.where(tall, 20.0, 5.0)
    if spike:
        centre = (f_north - 0.5) ** 2 + (f_east - 0.5) ** 2
        ndsm.flat[int(np.argmin(centre))] = 30.0
    if nodata_ne:
        ndsm = np.where((f_north > 0.75) & (f_east > 0.75), NODATA, ndsm)

    n_all = np.full((height, width), 10.0)
    n_multi = np.where((f_north < 0.25) & (f_east < 0.25), 8.0, 1.0)
    sparse = (f_north > 0.75) & (f_east > 0.75)
    n_ng = np.where(sparse, 0.0, 10.0)
    # 10 non-ground returns at h-1 and h+1 => population std exactly 1.0
    sum_z = n_ng * ndsm
    sum_z2 = np.where(sparse, 0.0, 10.0 * ndsm * ndsm + 10.0)

    layers = [ndsm, n_all, n_multi, n_ng, sum_z, sum_z2][:bands]
    profile = dict(
        driver="GTiff", height=height, width=width, count=bands,
        dtype=dtype, crs=f"EPSG:{EPSG}", transform=transform,
    )
    if np.dtype(dtype).kind == "f":
        profile["nodata"] = NODATA
    with rasterio.open(path, "w", **profile) as dst:
        for index, layer in enumerate(layers, start=1):
            dst.write(np.asarray(layer, dtype=dtype), index)
    return str(path)


def _load(path, *, rect=RECT, meshsize=MESHSIZE, height_q=90):
    return load_ndsm_evidence(rect, meshsize, str(path), height_q=height_q)


class TestSyntheticCogFixture:
    """Guard the guard: a flip-invariant fixture makes every check vacuous."""

    def test_height_band_is_asymmetric_north_south(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        with rasterio.open(path) as src:
            band1 = src.read(1)
        assert not np.array_equal(band1, np.flipud(band1))
        tall = band1 > 15.0
        assert tall.any()
        # The band and its own mirror must not overlap at all, otherwise a
        # flipped reader would still land partly on the tall cells.
        assert not (tall & np.flipud(tall)).any()

    def test_multi_return_band_is_asymmetric_on_both_axes(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        with rasterio.open(path) as src:
            band3 = src.read(3)
        hot = band3 > 4.0
        assert hot.any()
        assert not (hot & np.flipud(hot)).any()
        assert not (hot & np.fliplr(hot)).any()

    def test_grid_is_the_expected_20x20(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        assert out["shape"] == (20, 20)

    def test_broadcast_affine_matches_rasterio_transform_xy(self, tmp_path):
        # _pixel_lonlat applies the window affine by broadcasting rather than
        # materializing two meshgrids, to halve peak allocation on large
        # windows. Pin it to the library implementation it replaced.
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        with rasterio.open(path) as src:
            transform = src.window_transform(rasterio.windows.Window(3, 5, 7, 11))
            crs = src.crs
        lon, lat = _pixel_lonlat(transform, 11, 7, crs)
        rows, cols = np.meshgrid(np.arange(11), np.arange(7), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        ref_lon, ref_lat = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        ).transform(np.asarray(xs), np.asarray(ys))
        assert np.allclose(lon, ref_lon, rtol=0, atol=0)
        assert np.allclose(lat, ref_lat, rtol=0, atol=0)


class TestLoadNdsmEvidenceFrame:
    def test_tall_band_lands_on_the_south_rows(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        h = out["height"]
        n = h.shape[0]
        assert np.nanmedian(h[:n // 4]) > 15.0, (
            "the 20 m band was written in the southern quarter of the "
            "rectangle but did not come back on the southern rows -- the "
            "north-up raster frame leaked into the display frame")
        assert np.nanmedian(h[-(n // 4):]) < 8.0

    def test_multi_return_evidence_lands_on_the_south_west_corner(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        mrf = out["mrf"]
        n_rows, n_cols = mrf.shape
        r, c = n_rows // 4, n_cols // 4
        assert np.nanmedian(mrf[:r, :c]) > 0.5          # SW: hot
        assert np.nanmedian(mrf[:r, -c:]) < 0.3         # SE
        assert np.nanmedian(mrf[-r:, :c]) < 0.3         # NW
        assert np.nanmedian(mrf[-r:, -c:]) < 0.3        # NE

    def test_non_square_grid_keeps_rows_and_columns_apart(self, tmp_path):
        # 60 m north-south x 40 m east-west at 2 m -> 30 rows, 20 cols. On the
        # square fixture, transposing n_rows and n_cols is a no-op.
        path = _write_synthetic_cog(tmp_path / "ev.tif", rect=RECT_TALL)
        out = _load(path, rect=RECT_TALL)
        assert out["shape"] == (30, 20)
        assert out["height"].shape == (30, 20)
        assert np.nanmedian(out["height"][:7]) > 15.0        # south quarter
        assert np.nanmedian(out["height"][-7:]) < 8.0        # north quarter

    @pytest.mark.parametrize("degrees", [0.0, 30.0, 137.0])
    def test_rotated_rectangle_puts_the_band_on_its_own_south_edge(
        self, tmp_path, degrees
    ):
        # Coordinate-based assignment must not care about rotation. The fixture
        # paints the band in the *rectangle's* frame, so the band is a rotated
        # strip that no longer aligns with any lat/lon bounding box -- a reader
        # that labelled pixels off the bbox instead of the parallelogram would
        # smear it across the grid and fail here. (137 degrees also puts the
        # v0->v1 edge on a negative delta-lat, where "row 0" is compass-north.)
        rect = _rotated(RECT, degrees)
        path = _write_synthetic_cog(tmp_path / "ev.tif", rect=rect)
        out = _load(path, rect=rect)
        h = out["height"]
        n = h.shape[0]
        assert np.isfinite(h).all()
        assert np.nanmedian(h[:n // 4]) > 15.0
        assert np.nanmedian(h[-(n // 4):]) < 8.0

    @pytest.mark.parametrize("degrees", [0.0, 30.0, 137.0])
    @pytest.mark.parametrize("rect_base", [RECT, RECT_TALL], ids=["square", "tall"])
    def test_cell_centres_round_trip_to_their_own_cells(self, degrees, rect_base):
        # The load-level assertions are satisfied by any labelling that is
        # roughly right, including a rotation-blind bounding-box mapping. This
        # pins the geometry itself: every centre compute_cell_center_coords
        # produces must label as exactly its own cell, at every rotation and on
        # a non-square grid.
        rect = _rotated(rect_base, degrees)
        centres = compute_cell_center_coords(rect, MESHSIZE)
        n_rows, n_cols = centres["grid_size"]
        origin = np.asarray(centres["origin"], dtype=float)
        u_full = n_rows * centres["adj_mesh"][0] * np.asarray(centres["u_vec"], float)
        v_full = n_cols * centres["adj_mesh"][1] * np.asarray(centres["v_vec"], float)

        labels = _cell_labels(
            centres["lons"].ravel(), centres["lats"].ravel(),
            origin, u_full, v_full, n_rows, n_cols,
        )
        expected = (
            np.arange(n_rows)[:, None] * n_cols + np.arange(n_cols)[None, :]
        ).ravel()
        assert np.array_equal(labels, expected), (
            f"{int((labels != expected).sum())} of {labels.size} cell centres "
            f"landed outside their own cell at {degrees} degrees")


class TestLoadNdsmEvidenceAggregation:
    def test_single_contaminated_pixel_does_not_become_a_tree(self, tmp_path):
        # THE spike mechanism: one roof-edge pixel under a cell used to be the
        # whole cell, because the loader sampled a single point per centre.
        path = _write_synthetic_cog(tmp_path / "ev.tif", spike=True)
        p50 = _load(path, height_q=50)["height"]
        p90 = _load(path, height_q=90)["height"]
        p100 = _load(path, height_q=100)["height"]
        assert np.nanmax(p50) == pytest.approx(5.0)
        assert np.nanmax(p90) < 8.0
        assert np.nanmax(p100) == pytest.approx(30.0), (
            "the 30 m pixel is in the window but never reaches any cell -- "
            "the reader is not aggregating all pixels under the cell")

    def test_all_pixels_under_a_cell_are_pooled(self, tmp_path):
        # 2 m cells over 0.5 m pixels: 16 pixels each, 10 returns per pixel = 160.
        #
        # The floor is exact, not a tolerance. Rounding the read window's far
        # edge down instead of up starves the boundary rows and columns of a
        # fraction of a pixel, which showed up here as 120 -- a 25% deficit on
        # exactly the counts the classifier gates on, comfortably inside a
        # "> 100" tolerance. Assert the true full-coverage count.
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        assert out["n_all"].min() >= 160.0
        assert out["n_nonground"][:5].min() >= 160.0
        assert np.nanmedian(out["n_all"]) == pytest.approx(160.0, abs=1e-9)

    def test_roughness_matches_the_written_dispersion(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        n = out["roughness"].shape[0]
        # Southern half is a uniform 20 m band with +/-1 m returns.
        assert np.nanmedian(out["roughness"][:n // 4]) == pytest.approx(1.0, abs=1e-6)

    def test_roughness_is_nan_where_pooled_nonground_is_below_two(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        n_rows, n_cols = out["roughness"].shape
        r, c = n_rows // 4, n_cols // 4
        ne = out["roughness"][-r:, -c:]
        assert np.isnan(ne).all(), (
            "the NE quarter has zero non-ground returns; roughness 0.0 there "
            "would read as maximally roof-like")
        assert (out["n_nonground"][-r:, -c:] == 0).all()
        # ... and the rest is unaffected.
        assert np.isfinite(out["roughness"][:r, :c]).all()

    def test_integer_evidence_bands_are_read(self, tmp_path):
        # The real rebuild writes counts as uint16; a boundless read with
        # fill_value=nan silently casts on an integer band, so the reader must
        # not rely on it.
        path = _write_synthetic_cog(tmp_path / "ev.tif", dtype="uint16")
        out = _load(path)
        assert out["degraded"] is False
        assert np.isfinite(out["mrf"]).all()
        n = out["height"].shape[0]
        assert np.nanmedian(out["height"][:n // 4]) > 15.0

    def test_nodata_pixels_do_not_reach_the_height(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif", nodata_ne=True)
        out = _load(path)
        n_rows, n_cols = out["height"].shape
        r, c = n_rows // 4, n_cols // 4
        assert np.isnan(out["height"][-r + 1:, -c + 1:]).all()
        assert np.isfinite(out["height"][:r, :c]).all()


class TestLoadNdsmEvidenceDegraded:
    def test_single_band_cog_returns_height_only(self, tmp_path):
        # The COG on disk today is single-band; the rebuild ships separately.
        path = _write_synthetic_cog(tmp_path / "ndsm.tif", bands=1)
        out = _load(path)
        assert out["degraded"] is True
        assert out["mrf"] is None
        assert out["roughness"] is None
        assert out["n_all"] is None
        assert out["n_nonground"] is None
        assert np.isfinite(out["height"]).all()
        n = out["height"].shape[0]
        assert np.nanmedian(out["height"][:n // 4]) > 15.0

    def test_six_band_cog_is_not_degraded(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        assert _load(path)["degraded"] is False

    def test_missing_file_returns_none(self, tmp_path):
        assert _load(tmp_path / "does_not_exist.tif") is None

    def test_no_overlap_returns_none(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        far = [(lon + 1.0, lat + 1.0) for lon, lat in RECT]
        assert _load(path, rect=far) is None

    def test_shape_matches_every_returned_array(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        for key in ("height", "mrf", "roughness", "n_all", "n_nonground"):
            assert out[key].shape == out["shape"], key


# ---------------------------------------------------------------------------
# classify_and_refine
#
# The decision table under test, per land-cover Tree cell with a finite height:
#
#   canopy    mrf >= mrf_hi and roughness >= rough_hi_m      -> keep the nDSM height
#   roof      mrf <= mrf_lo and roughness <= rough_lo_m
#             and adjacent to a footprint                    -> local tree median
#   ambiguous everything else                                -> replace only when
#             adjacent AND |h - nearby building h| <= tol; otherwise keep
#
# The canopy verdict overrides the coincidence test. That is the whole point of
# the refactor, and test_tall_tree_beside_matching_building_survives is its
# acceptance test: the superseded _sanitize_ndsm_canopy flattens that cell to a
# flat 10 m because its height happens to match the neighbouring roof.
#
# Fixture discipline: the grid is 17x13, so rows and columns are never
# interchangeable; features are placed at least 2*median_radius+2 rows apart so
# one test region's trees can never leak into another's median window; and the
# expected replacement values are computed by hand and asserted exactly, so a
# regression to a flat constant (the old behaviour) fails rather than passing on
# a loose inequality. Every test that asserts "kept" carries a control cell that
# differs only in the evidence and must be replaced, so a pass cannot be vacuous.
# ---------------------------------------------------------------------------
SHAPE = (17, 13)                 # non-square on purpose
STATIC = 10.0                    # static_tree_height in every test below


def _scene(shape=SHAPE):
    """Neutral scene: no trees, no buildings, well-sampled canopy-like evidence.

    Evidence defaults sit clear of both thresholds on the canopy side, so any
    tree added without an explicit override is a canopy-verdict cell. Tests that
    need ambiguity or roof evidence say so at the cell.
    """
    return {
        "height": np.zeros(shape),
        "tree_mask": np.zeros(shape, dtype=bool),
        "building_heights": np.zeros(shape),
        "mrf": np.full(shape, 0.50),
        "roughness": np.full(shape, 2.0),
        "n_all": np.full(shape, 100.0),
        "n_nonground": np.full(shape, 60.0),
    }


def _tree(scene, rc, height, **evidence):
    """Mark *rc* a tree of *height*, overriding any named evidence band."""
    scene["tree_mask"][rc] = True
    scene["height"][rc] = height
    for key, value in evidence.items():
        scene[key][rc] = value
    return scene


AMBIGUOUS = dict(mrf=0.25, roughness=1.2)      # between lo and hi on both axes
ROOFLIKE = dict(mrf=0.05, roughness=0.3)       # single-echo, planar


def _refine(scene, *, static_tree_height=STATIC, params=None, **kwargs):
    return classify_and_refine(
        scene["height"],
        scene["tree_mask"],
        scene["building_heights"],
        static_tree_height,
        mrf=scene["mrf"],
        roughness=scene["roughness"],
        n_all=scene["n_all"],
        n_nonground=scene["n_nonground"],
        params=params or RefineParams(),
        **kwargs,
    )


class TestAcceptance:
    def test_tall_tree_beside_matching_building_survives(self):
        """THE regression this refactor exists to fix.

        A genuine 25 m tree in the cell next to a 24 m building. The heights
        coincide within the 5 m tolerance, so the superseded coincidence test
        calls it roof leakage and flattens it to a flat 10 m. With canopy
        evidence -- multi-return, rough -- the height must survive untouched.

        The control pair at rows 13/15 is the same geometry with mid-range
        evidence: it *is* replaced, which proves the fixture can trigger the
        replacement path and that the assertion above is not vacuous.
        """
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)                       # canopy evidence

        sc["building_heights"][13, 6] = 24.0
        _tree(sc, (13, 5), 25.0, **AMBIGUOUS)         # control
        _tree(sc, (15, 2), 8.0)                       # median source for the
        _tree(sc, (15, 3), 14.0)                      # control: median 11.0

        out = _refine(sc)

        assert out["canopy"][2, 5] == 25.0, (
            "a 25 m tree beside a 24 m building was capped -- the canopy "
            "verdict must override the height-coincidence test")
        assert out["verdict"][2, 5] == VERDICT_CANOPY
        # Control: identical geometry, no canopy evidence -> replaced with the
        # local tree median of {8, 14}, not a flat 10 m constant.
        assert out["verdict"][13, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][13, 5] == 11.0

    def test_canopy_verdict_survives_above_the_old_35m_cap(self):
        # The superseded sanitiser clamped every tree to 35 m. A 40 m tree with
        # canopy evidence beside a 39 m building must keep all 40 m.
        sc = _scene()
        sc["building_heights"][2, 6] = 39.0
        _tree(sc, (2, 5), 40.0)
        _tree(sc, (13, 5), 40.0, **AMBIGUOUS)
        sc["building_heights"][13, 6] = 39.0
        _tree(sc, (15, 2), 8.0)
        _tree(sc, (15, 3), 14.0)

        out = _refine(sc)
        assert out["canopy"][2, 5] == 40.0
        assert out["canopy"][13, 5] == 11.0          # control still replaced


class TestDecisionTable:
    def test_roof_spike_becomes_the_local_median_not_a_flat_constant(self):
        # Roof evidence next to a footprint, at the roof's height: replaced.
        # The replacement is the median of the credible trees around it --
        # {6, 12, 18} -> 12.0 -- and emphatically not the 10 m constant the
        # superseded sanitiser wrote.
        sc = _scene()
        sc["building_heights"][2, 6] = 30.0
        _tree(sc, (2, 5), 29.5, **ROOFLIKE)
        _tree(sc, (4, 3), 6.0)
        _tree(sc, (4, 4), 12.0)
        _tree(sc, (5, 3), 18.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_ROOF
        assert out["canopy"][2, 5] == 12.0, (
            "the roof spike must be replaced by the local tree median, not by "
            f"static_tree_height ({STATIC})")

    def test_ambiguous_adjacent_and_coincident_is_replaced(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 22.0
        _tree(sc, (2, 5), 24.0, **AMBIGUOUS)          # |24 - 22| = 2 <= 5
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0            # median of {7, 15}

    def test_ambiguous_far_from_any_building_is_kept(self):
        # A tall tree in open ground is never touched, whatever its height.
        sc = _scene()
        sc["building_heights"][2, 0] = 30.0           # four columns away
        _tree(sc, (2, 5), 31.0, **AMBIGUOUS)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 31.0

    def test_ambiguous_adjacent_but_height_does_not_coincide_is_kept(self):
        # Adjacency alone is not evidence: a 25 m tree beside a 10 m shed is
        # not roof leakage under any reading.
        sc = _scene()
        sc["building_heights"][2, 6] = 10.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)          # |25 - 10| = 15 > 5
        _tree(sc, (13, 5), 25.0, **AMBIGUOUS)         # control: coincident
        sc["building_heights"][13, 6] = 24.0
        _tree(sc, (15, 2), 8.0)
        _tree(sc, (15, 3), 14.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 25.0
        assert out["canopy"][13, 5] == 11.0           # control replaced

    def test_canopy_verdict_needs_both_axes_not_just_the_multi_return(self):
        # High multi-return over a *planar* surface is not canopy -- a glazed
        # or cluttered roof scatters echoes without being rough. Neither
        # verdict applies, so the cell falls to the ambiguous path and, being
        # adjacent and coincident, is replaced.
        #
        # This fixture is the only one where the two evidence axes disagree.
        # Without it, "mrf alone decides" and "roughness alone decides" are
        # both indistinguishable from the real rule.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, mrf=0.80, roughness=0.2)
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0

    def test_canopy_verdict_needs_both_axes_not_just_the_roughness(self):
        # The mirror case: rough but single-echo. A sparse crown seen once per
        # pulse is not proven canopy either.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, mrf=0.05, roughness=3.0)
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0

    def test_roof_evidence_without_a_footprint_is_only_ambiguous(self):
        # Pins a deliberate gap in the spec: the roof verdict requires an
        # adjacent footprint, so roof evidence over unmapped built-up ground
        # falls through to ambiguous -- and, having no neighbour to coincide
        # with, is kept. Change this only on purpose.
        sc = _scene()
        _tree(sc, (2, 5), 30.0, **ROOFLIKE)           # no building anywhere

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 30.0

    def test_a_tree_cell_on_a_footprint_counts_as_adjacent(self):
        # The adjacency window includes the cell itself: a tree cell sitting on
        # a building footprint is the strongest leakage case there is, so it
        # must not need a *neighbouring* footprint to qualify.
        sc = _scene()
        sc["building_heights"][2, 5] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        # Control: the same tree two cells from the footprint is out of range.
        sc["building_heights"][13, 7] = 24.0
        _tree(sc, (13, 5), 25.0, **AMBIGUOUS)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0            # median of {7, 15}
        assert out["verdict"][13, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][13, 5] == 25.0


class TestEvidenceTrust:
    def test_low_n_all_distrusts_the_evidence(self):
        # Canopy-looking numbers from four returns are noise, not evidence.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, n_all=4.0)            # below min_points = 8
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        sc["building_heights"][13, 6] = 24.0
        _tree(sc, (13, 5), 25.0)                      # control: n_all = 100

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0
        assert out["verdict"][13, 5] == VERDICT_CANOPY
        assert out["canopy"][13, 5] == 25.0

    def test_low_n_nonground_distrusts_the_evidence(self):
        # Roughness is a dispersion over the *non-ground* returns; three of
        # them do not make a trustworthy standard deviation in either
        # direction. pool_evidence already NaNs it below two -- this is the
        # confidence band above that hard floor, and it is why the reader
        # returns n_nonground at all.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, n_nonground=3.0)      # below min_nonground = 4
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        sc["building_heights"][13, 6] = 24.0
        _tree(sc, (13, 5), 25.0, n_nonground=60.0)    # control

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0
        assert out["verdict"][13, 5] == VERDICT_CANOPY

    def test_low_n_nonground_also_blocks_the_roof_verdict(self):
        # Symmetry matters: a barely-sampled cell reads planar, and treating
        # that as roof evidence is exactly the false cap this module removes.
        sc = _scene()
        sc["building_heights"][2, 6] = 30.0
        _tree(sc, (2, 5), 12.0, n_nonground=3.0, **ROOFLIKE)
        # |12 - 30| = 18 > 5, so with the roof verdict blocked the ambiguous
        # fallback keeps it.
        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 12.0

    def test_nan_roughness_is_not_canopy_evidence(self):
        # The sparse cell pool_evidence flags with NaN. It must satisfy
        # *neither* verdict -- and in particular must not be read as 0.0,
        # which is the maximally roof-like value.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, roughness=np.nan)
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0

    def test_nan_roughness_is_not_roof_evidence(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 30.0
        _tree(sc, (2, 5), 12.0, mrf=0.05, roughness=np.nan)
        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 12.0

    def test_nan_mrf_satisfies_neither_verdict(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 30.0
        _tree(sc, (2, 5), 12.0, mrf=np.nan, roughness=0.3)     # roof-like rough
        _tree(sc, (13, 5), 25.0, mrf=np.nan)                   # canopy-like rough
        sc["building_heights"][13, 6] = 24.0
        _tree(sc, (15, 2), 8.0)
        _tree(sc, (15, 3), 14.0)

        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP     # not roof
        assert out["verdict"][13, 5] == VERDICT_AMBIGUOUS_REPLACE  # not canopy
        assert out["canopy"][13, 5] == 11.0

    def test_thresholds_are_inclusive_at_the_boundary(self):
        # mrf == mrf_hi and roughness == rough_hi_m is a canopy cell; the
        # table's comparisons are >= and <=, not > and <.
        p = RefineParams()
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, mrf=p.mrf_hi, roughness=p.rough_hi_m)
        sc["building_heights"][13, 6] = 30.0
        _tree(sc, (13, 5), 29.0, mrf=p.mrf_lo, roughness=p.rough_lo_m)
        _tree(sc, (15, 2), 8.0)
        _tree(sc, (15, 3), 14.0)

        out = _refine(sc, params=p)
        assert out["verdict"][2, 5] == VERDICT_CANOPY
        assert out["verdict"][13, 5] == VERDICT_ROOF


class TestDegradedMode:
    """The live path today: the COG on disk is still single band."""

    def _degraded(self, scene, **kwargs):
        return classify_and_refine(
            scene["height"], scene["tree_mask"], scene["building_heights"],
            STATIC, mrf=None, roughness=None, n_all=None, n_nonground=None,
            **kwargs,
        )

    def test_every_tree_cell_is_ambiguous(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)                       # canopy evidence ignored
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)
        _tree(sc, (10, 10), 31.0)

        out = self._degraded(sc)
        assert out["degraded"] is True
        assert out["counts"]["canopy"] == 0
        assert out["counts"]["roof"] == 0
        assert set(np.unique(out["verdict"][sc["tree_mask"]])) <= {
            VERDICT_AMBIGUOUS_KEEP, VERDICT_AMBIGUOUS_REPLACE, VERDICT_NO_DATA
        }

    def test_tall_cell_far_from_a_building_is_still_kept(self):
        # Degraded mode must not become a blanket cap: without a footprint
        # nearby there is nothing to suspect.
        sc = _scene()
        _tree(sc, (10, 10), 31.0)
        out = self._degraded(sc)
        assert out["canopy"][10, 10] == 31.0

    def test_adjacent_coincident_cell_is_replaced_with_the_median(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = self._degraded(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_REPLACE
        assert out["canopy"][2, 5] == 11.0            # median, not the constant

    def test_explicit_degraded_flag_overrides_present_evidence(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)                       # canopy evidence present
        _tree(sc, (4, 3), 7.0)
        _tree(sc, (4, 4), 15.0)

        out = _refine(sc, degraded=True)
        assert out["degraded"] is True
        assert out["counts"]["canopy"] == 0
        assert out["canopy"][2, 5] == 11.0

    def test_degraded_false_without_evidence_is_rejected(self):
        sc = _scene()
        with pytest.raises(ValueError, match="degraded"):
            classify_and_refine(
                sc["height"], sc["tree_mask"], sc["building_heights"], STATIC,
                degraded=False,
            )


class TestGuards:
    def test_below_minimum_height_becomes_the_local_median(self):
        sc = _scene()
        _tree(sc, (2, 5), 0.5)                        # canopy verdict, absurd h
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)

        out = _refine(sc)
        assert out["canopy"][2, 5] == 12.0            # median of {8, 16}
        assert out["counts"]["guard_low"] == 1
        # The guard is orthogonal to the verdict: the cell is still classified.
        assert out["verdict"][2, 5] == VERDICT_CANOPY

    def test_below_minimum_with_no_credible_neighbour_falls_back_to_static(self):
        sc = _scene()
        _tree(sc, (2, 5), 0.5)                        # isolated
        out = _refine(sc)
        assert out["canopy"][2, 5] == STATIC

    def test_above_maximum_height_is_clamped(self):
        sc = _scene()
        _tree(sc, (2, 5), 60.0)
        out = _refine(sc)
        assert out["canopy"][2, 5] == RefineParams().max_tree_height_m
        assert out["counts"]["guard_high"] == 1
        assert out["verdict"][2, 5] == VERDICT_CANOPY

    def test_the_median_source_uses_clamped_heights(self):
        # A 60 m neighbour contributes its clamped 45 m, not 60, so a single
        # implausible cell cannot drag a whole neighbourhood upward.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        _tree(sc, (4, 3), 60.0)
        _tree(sc, (4, 4), 20.0)

        out = _refine(sc)
        assert out["canopy"][2, 5] == 32.5            # median of {45, 20}


class TestReplacementValue:
    def test_replaced_cells_are_not_their_own_median_source(self):
        # A run of contaminated cells along a roof edge would otherwise vouch
        # for itself: the median of five cells three of which read 30 m is
        # 30 m, and the replacement would be a no-op. Only kept cells are
        # credible sources.
        sc = _scene()
        sc["building_heights"][2, 6] = 30.0
        for row in (1, 2, 3):
            _tree(sc, (row, 5), 30.0, **AMBIGUOUS)
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)

        out = _refine(sc)
        for row in (1, 2, 3):
            assert out["verdict"][row, 5] == VERDICT_AMBIGUOUS_REPLACE
            assert out["canopy"][row, 5] == 12.0      # median of {8, 16}

    def test_isolated_replacement_falls_back_to_static(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)          # no tree neighbours
        out = _refine(sc)
        assert out["canopy"][2, 5] == STATIC

    def test_median_radius_bounds_the_neighbourhood(self):
        # A credible tree outside the window must not be reachable.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        _tree(sc, (9, 5), 18.0)                       # 7 rows away
        out = _refine(sc, params=RefineParams(median_radius=3))
        assert out["canopy"][2, 5] == STATIC
        out_wide = _refine(sc, params=RefineParams(median_radius=7))
        assert out_wide["canopy"][2, 5] == 18.0


class TestAccounting:
    def _mixed_scene(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)                              # canopy
        sc["building_heights"][6, 6] = 30.0
        _tree(sc, (6, 5), 29.5, **ROOFLIKE)                  # roof
        _tree(sc, (10, 1), 31.0, **AMBIGUOUS)                # ambiguous keep
        sc["building_heights"][14, 6] = 24.0
        _tree(sc, (14, 5), 25.0, **AMBIGUOUS)                # ambiguous replace
        _tree(sc, (16, 11), np.nan)                          # no nDSM data
        _tree(sc, (8, 2), 9.0)                               # plain kept tree
        return sc

    def test_verdict_counts_partition_the_tree_cells(self):
        sc = self._mixed_scene()
        out = _refine(sc)
        counts = out["counts"]
        assert counts["tree"] == int(sc["tree_mask"].sum())
        assert sum(counts[key] for key in PARTITION_KEYS) == counts["tree"]
        # every bucket is actually exercised, so the sum is not trivially right
        for key in PARTITION_KEYS:
            assert counts[key] > 0, key

    def test_verdict_array_agrees_with_the_counts(self):
        sc = self._mixed_scene()
        out = _refine(sc)
        codes = {
            "canopy": VERDICT_CANOPY,
            "roof": VERDICT_ROOF,
            "ambiguous_keep": VERDICT_AMBIGUOUS_KEEP,
            "ambiguous_replace": VERDICT_AMBIGUOUS_REPLACE,
            "no_data": VERDICT_NO_DATA,
        }
        for key, code in codes.items():
            assert int((out["verdict"] == code).sum()) == out["counts"][key], key
        assert (out["verdict"][~sc["tree_mask"]] == VERDICT_NONE).all()

    def test_guard_counts_are_reported_outside_the_partition(self):
        # Guards can fire on a cell in any bucket, so they overlap it and must
        # not be summed into it.
        sc = _scene()
        _tree(sc, (2, 5), 60.0)                       # canopy verdict, clamped
        _tree(sc, (10, 5), 0.5)                       # canopy verdict, lifted
        out = _refine(sc)
        counts = out["counts"]
        assert counts["canopy"] == 2
        assert counts["guard_high"] == 1
        assert counts["guard_low"] == 1
        assert sum(counts[key] for key in PARTITION_KEYS) == counts["tree"] == 2


class TestInvariants:
    def test_non_tree_cells_are_zero(self):
        sc = _scene()
        sc["height"][:] = 7.0
        sc["building_heights"][:, 6] = 20.0
        _tree(sc, (2, 5), 25.0)
        out = _refine(sc)
        assert (out["canopy"][~sc["tree_mask"]] == 0.0).all()

    def test_tree_cell_without_an_ndsm_height_gets_the_median_fallback(self):
        # Today's pipeline fills these with a flat static height before
        # sanitising; the local median is strictly better and keeps the tree.
        sc = _scene()
        _tree(sc, (2, 5), np.nan)
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)
        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_NO_DATA
        assert out["canopy"][2, 5] == 12.0
        # Filled deliberately, not rescued by the below-minimum guard: leaving
        # the cell at 0.0 and letting the guard pick it up produces the same
        # height but misreports it, and would hide the missing fill entirely.
        assert out["counts"]["guard_low"] == 0

    def test_isolated_tree_cell_without_a_height_falls_back_to_static(self):
        sc = _scene()
        _tree(sc, (2, 5), np.nan)
        out = _refine(sc)
        assert out["canopy"][2, 5] == STATIC

    def test_infinite_height_is_treated_as_missing(self):
        sc = _scene()
        _tree(sc, (2, 5), np.inf)
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)
        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_NO_DATA
        assert out["canopy"][2, 5] == 12.0

    def test_nan_building_height_is_not_a_footprint(self):
        sc = _scene()
        sc["building_heights"][2, 6] = np.nan
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        out = _refine(sc)
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP
        assert out["canopy"][2, 5] == 25.0

    def test_negative_building_height_is_not_a_footprint(self):
        sc = _scene()
        sc["building_heights"][2, 6] = -3.0
        _tree(sc, (2, 5), 1.0, **AMBIGUOUS)
        out = _refine(sc)
        # |1 - (-3)| = 4 <= 5 would "coincide" if negatives counted.
        assert out["verdict"][2, 5] == VERDICT_AMBIGUOUS_KEEP

    def test_rows_and_columns_are_not_interchanged(self):
        # Every operator here is isotropic, so transposing the whole scene must
        # transpose the result exactly. On the 17x13 grid a row/column mix-up
        # inside the neighbourhood filters cannot survive this.
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)
        _tree(sc, (11, 9), 31.0)

        direct = _refine(sc)
        flipped = _refine({k: v.T for k, v in sc.items()})
        assert flipped["canopy"].shape == (SHAPE[1], SHAPE[0])
        assert np.array_equal(flipped["canopy"], direct["canopy"].T)
        assert np.array_equal(flipped["verdict"], direct["verdict"].T)

    def test_inputs_are_not_mutated(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0, **AMBIGUOUS)
        _tree(sc, (4, 3), 8.0)
        snapshot = {key: value.copy() for key, value in sc.items()}
        _refine(sc)
        for key, value in sc.items():
            assert np.array_equal(value, snapshot[key], equal_nan=True), key

    @pytest.mark.parametrize(
        "key", ["tree_mask", "building_heights", "mrf", "roughness",
                "n_all", "n_nonground"]
    )
    def test_shape_mismatch_is_rejected(self, key):
        sc = _scene()
        sc[key] = sc[key][:-1]
        with pytest.raises(ValueError, match="shape"):
            _refine(sc)

    def test_one_dimensional_height_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            classify_and_refine(
                np.zeros(5), np.zeros(5, bool), np.zeros(5), STATIC,
                degraded=True,
            )


class TestRefineParams:
    @pytest.mark.parametrize("kwargs", [
        dict(mrf_lo=0.5, mrf_hi=0.3),
        dict(rough_lo_m=2.0, rough_hi_m=1.0),
        dict(min_tree_height_m=50.0, max_tree_height_m=45.0),
        dict(min_nonground=1),
        dict(median_radius=-1),
        dict(adjacency_radius=-1),
        dict(min_points=0),
    ])
    def test_incoherent_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RefineParams(**kwargs)


class TestStaticTreeHeight:
    """``static_tree_height`` is user-supplied, so it can contradict the guards.

    A fallback below ``min_tree_height_m`` never converges: the below-minimum
    guard fires on the cell, substitutes the fallback, and the fallback is
    itself below the minimum. The cell is left under the documented floor with
    ``guard_low`` incremented -- the guard reporting that it fixed something it
    did not. The output invariant is a promise, so the contradiction is refused
    at the door rather than half-applied.
    """

    def test_a_static_height_below_the_minimum_is_rejected(self):
        sc = _scene()
        _tree(sc, (2, 5), np.nan)
        with pytest.raises(ValueError, match="min_tree_height_m"):
            _refine(sc, static_tree_height=0.5)

    def test_a_non_finite_static_height_is_rejected(self):
        sc = _scene()
        with pytest.raises(ValueError):
            _refine(sc, static_tree_height=float("nan"))

    def test_the_minimum_itself_is_accepted(self):
        # The boundary is inclusive: at exactly the minimum the guard does not
        # fire, so the fallback is a fixed point and the invariant holds.
        sc = _scene()
        _tree(sc, (2, 5), np.nan)
        out = _refine(sc, static_tree_height=RefineParams().min_tree_height_m)
        assert out["canopy"][2, 5] == RefineParams().min_tree_height_m
        assert out["counts"]["guard_low"] == 0


class TestRefineFromEvidence:
    """The calling convention Task 4 wires up: reader dict in, canopy out."""

    def _evidence(self, scene, degraded=False):
        return {
            "height": scene["height"],
            "mrf": None if degraded else scene["mrf"],
            "roughness": None if degraded else scene["roughness"],
            "n_all": None if degraded else scene["n_all"],
            "n_nonground": None if degraded else scene["n_nonground"],
            "degraded": degraded,
            "shape": scene["height"].shape,
        }

    def test_unpacks_the_reader_dict(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)
        out = refine_from_evidence(
            self._evidence(sc), sc["tree_mask"], sc["building_heights"], STATIC
        )
        assert out["canopy"][2, 5] == 25.0
        assert out["degraded"] is False

    def test_degraded_dict_is_honoured(self):
        sc = _scene()
        sc["building_heights"][2, 6] = 24.0
        _tree(sc, (2, 5), 25.0)
        _tree(sc, (4, 3), 8.0)
        _tree(sc, (4, 4), 16.0)
        out = refine_from_evidence(
            self._evidence(sc, degraded=True), sc["tree_mask"],
            sc["building_heights"], STATIC,
        )
        assert out["degraded"] is True
        assert out["canopy"][2, 5] == 12.0

    def test_a_none_reader_result_is_rejected(self):
        # load_ndsm_evidence returns None for a missing or non-overlapping COG.
        # Refining nothing is not a sensible degradation; the caller keeps
        # static tree heights instead.
        sc = _scene()
        with pytest.raises(ValueError, match="fall back to static"):
            refine_from_evidence(
                None, sc["tree_mask"], sc["building_heights"], STATIC
            )

    def test_end_to_end_from_a_synthetic_cog(self, tmp_path):
        # Contract check between the two halves of the module: whatever
        # load_ndsm_evidence returns must feed the classifier unmodified.
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        evidence = _load(path)
        shape = evidence["shape"]
        tree_mask = np.ones(shape, dtype=bool)
        buildings = np.zeros(shape)
        # The fixture writes roughness == 1.0 exactly, so the seeds have to be
        # narrowed for either evidence verdict to be reachable here.
        params = RefineParams(mrf_hi=0.5, rough_hi_m=0.9, rough_lo_m=0.5)
        out = refine_from_evidence(
            evidence, tree_mask, buildings, STATIC, params=params
        )
        assert out["canopy"].shape == shape
        counts = out["counts"]
        assert sum(counts[key] for key in PARTITION_KEYS) == counts["tree"]
        r, c = shape[0] // 4, shape[1] // 4
        # SW quarter: mrf 0.8, roughness 1.0 -> canopy verdict.
        assert (out["verdict"][:r, :c] == VERDICT_CANOPY).all()
        # NE quarter: zero non-ground returns -> NaN roughness -> distrusted.
        assert (out["verdict"][-r:, -c:] == VERDICT_AMBIGUOUS_KEEP).all()

    def test_end_to_end_single_band_cog_is_degraded(self, tmp_path):
        path = _write_synthetic_cog(tmp_path / "ndsm.tif", bands=1)
        evidence = _load(path)
        shape = evidence["shape"]
        out = refine_from_evidence(
            evidence, np.ones(shape, bool), np.zeros(shape), STATIC
        )
        assert out["degraded"] is True
        assert out["counts"]["canopy"] == 0
        # No buildings anywhere -> nothing is suspect -> heights survive.
        assert np.nanmedian(out["canopy"][:shape[0] // 4]) > 15.0
