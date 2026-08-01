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
    pool_evidence,
    groupwise_percentile,
    load_ndsm_evidence,
    local_tree_median,
)


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
# The COG is north-up raster space (row 0 = north); the model grid is the
# display frame (row 0 = south, because compute_grid_geometry anchors at
# rectangle_vertices[0] = SW and side_1 runs SW->NW). A windowed read carries
# the raster's orientation, so the fixture below is asymmetric along BOTH axes
# and the assertions pin the tall band to the SOUTH rows and the multi-return
# evidence to the SOUTH-WEST corner. A symmetric fixture would pass whether or
# not the reader flipped, which is exactly how the last two-frame bug survived
# four review passes.
# ---------------------------------------------------------------------------
EPSG = 6677                      # JGD2011 / Japan Plane Rectangular CS IX
PIXEL = 0.5                      # metres, as in the real nDSM COG
MESHSIZE = 2.0                   # -> 4x4 raster pixels per model cell
SIDE_M = 40.0                    # -> 20x20 model cells
NODATA = -9999.0

_W, _S = 139.7600, 35.6800
_DLAT = SIDE_M / 110946.0
_DLON = SIDE_M / (111320.0 * math.cos(math.radians(_S)))
_E, _N = _W + _DLON, _S + _DLAT

# [SW, NW, NE, SE]: compute_grid_geometry takes v0 as the origin, side_1 =
# v1 - v0 (northward, the row axis) and side_2 = v3 - v0 (eastward, the col axis).
RECT = [(_W, _S), (_W, _N), (_E, _N), (_E, _S)]


def _rect_fractions(lon, lat):
    """Position within the target rectangle: (northward, eastward) in [0, 1]."""
    return (lat - _S) / (_N - _S), (lon - _W) / (_E - _W)


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
    f_north, f_east = _rect_fractions(lon, lat)

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

    def test_rotated_rectangle_still_resolves(self, tmp_path):
        # Coordinate-based assignment must not care about rotation. The band
        # geometry no longer aligns with the grid axes, so this only asserts
        # that every cell is filled -- a shape/indexing regression would leave
        # NaN holes or raise.
        cx, cy = (_W + _E) / 2.0, (_S + _N) / 2.0
        theta = math.radians(30.0)
        scale = math.cos(math.radians(_S))

        def _rot(lon, lat):
            dx, dy = (lon - cx) * scale, lat - cy
            rx = dx * math.cos(theta) - dy * math.sin(theta)
            ry = dx * math.sin(theta) + dy * math.cos(theta)
            return cx + rx / scale, cy + ry

        rect = [_rot(lon, lat) for lon, lat in RECT]
        path = _write_synthetic_cog(tmp_path / "ev.tif", rect=rect)
        out = _load(path, rect=rect)
        assert np.isfinite(out["height"]).all()
        assert np.isfinite(out["mrf"]).all()


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
        # 2 m cells over 0.5 m pixels: ~16 pixels each, 10 returns per pixel.
        path = _write_synthetic_cog(tmp_path / "ev.tif")
        out = _load(path)
        assert np.nanmedian(out["n_all"]) == pytest.approx(160.0, abs=20.0)
        assert out["n_all"].min() > 100.0

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
