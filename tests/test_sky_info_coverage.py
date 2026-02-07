"""Round 6 – cover sky.py uncovered lines: get_patch_info (636-668), visualize_sky_patches (lines 162-191), accumulate_to_tregenza_patches (535-552)."""
from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# Tests for get_patch_info
# ===========================================================================

class TestGetPatchInfo:
    """Cover sky.py lines 636-668."""

    def test_tregenza_info(self):
        from voxcity.simulator.solar.sky import get_patch_info
        info = get_patch_info("tregenza")
        assert info["method"] == "Tregenza"
        assert info["n_patches"] == 145
        assert "reference" in info

    def test_reinhart_info(self):
        from voxcity.simulator.solar.sky import get_patch_info
        info = get_patch_info("reinhart", mf=4)
        assert info["method"] == "Reinhart"
        assert info["mf"] == 4
        assert info["n_patches"] > 145

    def test_uniform_info(self):
        from voxcity.simulator.solar.sky import get_patch_info
        info = get_patch_info("uniform", n_azimuth=36, n_elevation=9)
        assert info["method"] == "Uniform Grid"
        assert info["n_azimuth"] == 36
        assert info["n_elevation"] == 9

    def test_fibonacci_info(self):
        from voxcity.simulator.solar.sky import get_patch_info
        info = get_patch_info("fibonacci", n_patches=100)
        assert info["method"] == "Fibonacci Spiral"
        assert info["n_patches"] == 100

    def test_unknown_raises(self):
        from voxcity.simulator.solar.sky import get_patch_info
        with pytest.raises(ValueError, match="Unknown method"):
            get_patch_info("nonexistent")


# ===========================================================================
# Tests for visualize_sky_patches
# ===========================================================================

class TestVisualizeSkyPatches:
    """Cover sky.py lines 162-191 (visualize_sky_patches helper)."""

    def test_tregenza_visualization(self):
        import matplotlib
        matplotlib.use("Agg")
        from voxcity.simulator.solar.sky import visualize_sky_patches
        ax = visualize_sky_patches("tregenza", show=False)
        assert ax is not None

    def test_reinhart_visualization(self):
        import matplotlib
        matplotlib.use("Agg")
        from voxcity.simulator.solar.sky import visualize_sky_patches
        ax = visualize_sky_patches("reinhart", show=False, mf=2)
        assert ax is not None

    def test_uniform_visualization(self):
        import matplotlib
        matplotlib.use("Agg")
        from voxcity.simulator.solar.sky import visualize_sky_patches
        ax = visualize_sky_patches("uniform", show=False, n_azimuth=18, n_elevation=5)
        assert ax is not None

    def test_fibonacci_visualization(self):
        import matplotlib
        matplotlib.use("Agg")
        from voxcity.simulator.solar.sky import visualize_sky_patches
        ax = visualize_sky_patches("fibonacci", show=False, n_patches=50)
        assert ax is not None

    def test_existing_axis(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from voxcity.simulator.solar.sky import visualize_sky_patches
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax_out = visualize_sky_patches("tregenza", ax=ax, show=False)
        assert ax_out is ax
        plt.close(fig)

    def test_unknown_raises(self):
        from voxcity.simulator.solar.sky import visualize_sky_patches
        with pytest.raises(ValueError, match="Unknown method"):
            visualize_sky_patches("bad_method", show=False)


# ===========================================================================
# Tests for accumulate_to_tregenza_patches
# ===========================================================================

class TestBinSunPositionsToTregenzaFast:
    """Cover sky.py lines 535-552 (bin_sun_positions_to_tregenza_fast)."""

    def test_basic_accumulation(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_tregenza_fast
        # One patch at azimuth=0, elevation=6 (first Tregenza band)
        az = np.array([0.0, 180.0, 0.0])
        el = np.array([6.0, 6.0, -5.0])   # third below horizon => skipped
        dni = np.array([100.0, 200.0, 500.0])
        cum, counts = bin_sun_positions_to_tregenza_fast(az, el, dni)
        assert cum.shape == (145,)
        assert counts.shape == (145,)
        # Third entry (elev <= 0) should be skipped
        assert cum.sum() == pytest.approx(300.0)
        assert counts.sum() == 2

    def test_zenith_patch(self):
        from voxcity.simulator.solar.sky import bin_sun_positions_to_tregenza_fast
        az = np.array([0.0])
        el = np.array([89.0])  # near-zenith => patch 144 (last band, single patch)
        dni = np.array([500.0])
        cum, counts = bin_sun_positions_to_tregenza_fast(az, el, dni)
        assert cum[144] == pytest.approx(500.0)
        assert counts[144] == 1


# ===========================================================================
# Tests for get_tregenza_patch_index_fast
# ===========================================================================

class TestGetTregenzaPatchIndexFast:
    """Cover additional branches of the fast Tregenza lookup."""

    def test_below_horizon_returns_neg1(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index_fast
        assert get_tregenza_patch_index_fast(45.0, -1.0) == -1

    def test_horizon_band(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index_fast
        idx = get_tregenza_patch_index_fast(0.0, 6.0)
        assert 0 <= idx < 30  # first band has 30 patches

    def test_zenith_single_patch(self):
        from voxcity.simulator.solar.sky import get_tregenza_patch_index_fast
        idx = get_tregenza_patch_index_fast(0.0, 88.0)
        assert idx == 144  # zenith patch
