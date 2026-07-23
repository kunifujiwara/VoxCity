"""Regression test: parallel downloads must repair a canopy grid that fell
back without a usable land-cover mask.

The parallel path hands the canopy worker a 1x1 placeholder (land cover isn't
downloaded yet). If the canopy source degrades to the Static land-cover-mask
fallback (e.g. Earth Engine unavailable / 403), it can only return a 1x1 grid.
Before the fix, that mismatched grid flowed into the voxelizer and crashed
with "Input grids must have the same shape". The pipeline must detect the
mismatch after the join and recompute static canopy from the real land cover
grid, matching sequential-mode behavior.
"""

import numpy as np

from voxcity.generator.pipeline import VoxCityPipeline
from voxcity.models import PipelineConfig

RECT = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
SHAPE = (8, 8)
# 'Tree' position in get_land_cover_classes('OpenStreetMap') under the
# 0-based enumerate used by the Static canopy mask (see StaticCanopyStrategy).
TREE_CLASS_INDEX = 5


class _LandCoverStub:
    def build_grid(self, *a, **k):
        grid = np.zeros(SHAPE, dtype=int)
        grid[2:4, 2:4] = TREE_CLASS_INDEX
        return grid


class _BuildingStub:
    def build_grids(self, *a, **k):
        import geopandas as gpd

        return (
            np.zeros(SHAPE),
            np.zeros(SHAPE),
            np.zeros(SHAPE, dtype=int),
            gpd.GeoDataFrame(),
        )


class _FallbackCanopyStub:
    """Mimics a GEE canopy source degrading to Static with a 1x1 placeholder."""

    def build_grids(self, rectangle_vertices, meshsize, land_cover_grid, output_dir, **kwargs):
        # get_canopy_height_grid's Static fallback shapes itself from the
        # placeholder it was given — reproduce that faithfully.
        top = np.zeros_like(land_cover_grid, dtype=float)
        return top, top * 0.5


class _DemStub:
    def build_grid(self, *a, **k):
        return np.zeros(SHAPE)


def test_parallel_repairs_mismatched_canopy(tmp_path):
    cfg = PipelineConfig(
        rectangle_vertices=RECT,
        meshsize=5.0,
        building_source="OpenStreetMap",
        land_cover_source="OpenStreetMap",
        canopy_height_source="High Resolution 1m Global Canopy Height Maps",
        dem_source="Flat",
        output_dir=str(tmp_path),
        static_tree_height=10.0,
    )
    pipeline = VoxCityPipeline(meshsize=5.0, rectangle_vertices=RECT)

    (
        land_cover_grid,
        bh,
        bmin,
        bid,
        gdf_out,
        canopy_top,
        canopy_bottom,
        dem,
        lc_src,
    ) = pipeline._run_parallel_downloads(
        cfg,
        _LandCoverStub(),
        _BuildingStub(),
        _FallbackCanopyStub(),
        _DemStub(),
        building_gdf=None,
        terrain_gdf=None,
        kwargs={},
    )

    # The repaired canopy must match the land cover shape...
    assert canopy_top.shape == land_cover_grid.shape == SHAPE
    assert canopy_bottom.shape == SHAPE
    # ...and carry static tree heights where land cover has trees.
    tree_mask = land_cover_grid == TREE_CLASS_INDEX
    assert (canopy_top[tree_mask] > 0).all()
    assert (canopy_top[~tree_mask] == 0).all()
