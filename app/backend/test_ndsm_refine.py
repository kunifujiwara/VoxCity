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
import warnings

import numpy as np
import pytest

from backend.ndsm_refine import (
    pool_evidence,
    groupwise_percentile,
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
