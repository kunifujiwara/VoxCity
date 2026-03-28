import numpy as np
from typing import Optional
from ..utils.logging import get_logger

_logger = get_logger(__name__)

try:
    from numba import jit, prange
    import numba  # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional accel
    NUMBA_AVAILABLE = False
    _logger.info("Numba not available. Using optimized version without JIT compilation.")

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    prange = range

from ..geoprocessor.raster import (
    group_and_label_cells,
    process_grid,
)
from ..utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
from ..utils.lc import convert_land_cover
from ..utils.classes import VOXEL_CODE_DESCRIPTIONS, LAND_COVER_DESCRIPTIONS


# -----------------------------
# Voxel class codes (semantics)
# -----------------------------
GROUND_CODE = -1
TREE_CODE = -2
BUILDING_CODE = -3


@jit(nopython=True, parallel=True)
def _voxelize_kernel(
    voxel_grid,
    land_cover_grid,
    dem_grid,
    tree_grid,
    canopy_bottom_grid,
    has_canopy_bottom,
    seg_starts,
    seg_ends,
    seg_offsets,
    seg_counts,
    trunk_height_ratio,
    voxel_size,
):
    rows, cols = land_cover_grid.shape
    for i in prange(rows):
        for j in range(cols):
            ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1

            # ground and land cover layer
            if ground_level > 0:
                voxel_grid[i, j, :ground_level] = GROUND_CODE
                voxel_grid[i, j, ground_level - 1] = land_cover_grid[i, j]

            # trees
            tree_height = tree_grid[i, j]
            if tree_height > 0.0:
                if has_canopy_bottom:
                    crown_base_height = canopy_bottom_grid[i, j]
                else:
                    crown_base_height = tree_height * trunk_height_ratio
                crown_base_level = int(crown_base_height / voxel_size + 0.5)
                crown_top_level = int(tree_height / voxel_size + 0.5)
                if (crown_top_level == crown_base_level) and (crown_base_level > 0):
                    crown_base_level -= 1
                tree_start = ground_level + crown_base_level
                tree_end = ground_level + crown_top_level
                if tree_end > tree_start:
                    voxel_grid[i, j, tree_start:tree_end] = TREE_CODE

            # buildings (packed segments)
            base = seg_offsets[i, j]
            count = seg_counts[i, j]
            for k in range(count):
                s = seg_starts[base + k]
                e = seg_ends[base + k]
                start = ground_level + s
                end = ground_level + e
                if end > start:
                    voxel_grid[i, j, start:end] = BUILDING_CODE


def _flatten_building_segments(building_min_height_grid: np.ndarray, voxel_size: float):
    rows, cols = building_min_height_grid.shape
    flat = building_min_height_grid.ravel()

    # Vectorised count: number of segments per cell
    counts_flat = np.array(
        [len(c) if isinstance(c, list) else 0 for c in flat],
        dtype=np.int32,
    )
    counts = counts_flat.reshape(rows, cols)
    total = int(counts_flat.sum())

    # Compute offsets via cumulative sum (no nested loop)
    cum = np.empty(len(counts_flat) + 1, dtype=np.int32)
    cum[0] = 0
    np.cumsum(counts_flat, out=cum[1:])
    offsets = cum[:-1].reshape(rows, cols)

    seg_starts = np.empty(total, dtype=np.int32)
    seg_ends = np.empty(total, dtype=np.int32)

    # Single pass: fill flattened arrays (only visit non-empty cells)
    inv_vs = 1.0 / voxel_size
    nz_indices = np.nonzero(counts_flat)[0]
    for idx in nz_indices:
        cell = flat[idx]
        base = cum[idx]
        for k, seg in enumerate(cell):
            seg_starts[base + k] = int(seg[0] * inv_vs + 0.5)
            seg_ends[base + k] = int(seg[1] * inv_vs + 0.5)

    return seg_starts, seg_ends, offsets, counts


class Voxelizer:
    def __init__(
        self,
        voxel_size: float,
        land_cover_source: str,
        trunk_height_ratio: Optional[float] = None,
        voxel_dtype=np.int8,
        max_voxel_ram_mb: Optional[float] = None,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.land_cover_source = land_cover_source
        self.trunk_height_ratio = float(trunk_height_ratio) if trunk_height_ratio is not None else (11.76 / 19.98)
        self.voxel_dtype = voxel_dtype
        self.max_voxel_ram_mb = max_voxel_ram_mb

    def _estimate_and_allocate(self, rows: int, cols: int, max_height: int) -> np.ndarray:
        try:
            bytes_per_elem = np.dtype(self.voxel_dtype).itemsize
            est_mb = rows * cols * max_height * bytes_per_elem / (1024 ** 2)
            _logger.info("Voxel grid shape: (%d, %d, %d), dtype: %s, ~%.1f MB", rows, cols, max_height, self.voxel_dtype, est_mb)
            if (self.max_voxel_ram_mb is not None) and (est_mb > self.max_voxel_ram_mb):
                raise MemoryError(
                    f"Estimated voxel grid memory {est_mb:.1f} MB exceeds limit {self.max_voxel_ram_mb} MB. Increase mesh size or restrict ROI."
                )
        except Exception:
            pass
        return np.zeros((rows, cols, max_height), dtype=self.voxel_dtype)

    def _convert_land_cover(self, land_cover_grid_ori: np.ndarray) -> np.ndarray:
        if self.land_cover_source == 'OpenStreetMap':
            return land_cover_grid_ori + 1  # Shift to 1-based indices
        return convert_land_cover(land_cover_grid_ori, land_cover_source=self.land_cover_source)

    def generate_combined(
        self,
        building_height_grid_ori: np.ndarray,
        building_min_height_grid_ori: np.ndarray,
        building_id_grid_ori: np.ndarray,
        land_cover_grid_ori: np.ndarray,
        dem_grid_ori: np.ndarray,
        tree_grid_ori: np.ndarray,
        canopy_bottom_height_grid_ori: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        _logger.info("Generating 3D voxel data")
        
        # Log class definitions if requested
        if kwargs.get("print_class_info", True):
            _logger.info(VOXEL_CODE_DESCRIPTIONS)
            _logger.info(LAND_COVER_DESCRIPTIONS)

        land_cover_grid_converted = self._convert_land_cover(land_cover_grid_ori)

        building_height_grid = ensure_orientation(
            np.nan_to_num(building_height_grid_ori, nan=10.0),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        building_min_height_grid = ensure_orientation(
            replace_nan_in_nested(building_min_height_grid_ori),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        building_id_grid = ensure_orientation(
            building_id_grid_ori,
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        land_cover_grid = ensure_orientation(
            land_cover_grid_converted.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        dem_grid = ensure_orientation(
            dem_grid_ori.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        ) - np.min(dem_grid_ori)
        dem_grid = process_grid(building_id_grid, dem_grid)
        tree_grid = ensure_orientation(
            tree_grid_ori.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        canopy_bottom_grid = None
        if canopy_bottom_height_grid_ori is not None:
            canopy_bottom_grid = ensure_orientation(
                canopy_bottom_height_grid_ori.copy(),
                ORIENTATION_NORTH_UP,
                ORIENTATION_SOUTH_UP,
            )

        assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"
        rows, cols = building_height_grid.shape
        max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / self.voxel_size)) + 1

        voxel_grid = self._estimate_and_allocate(rows, cols, max_height)

        trunk_height_ratio = float(kwargs.get("trunk_height_ratio", self.trunk_height_ratio))

        if NUMBA_AVAILABLE:
            has_canopy = canopy_bottom_grid is not None
            canopy_in = canopy_bottom_grid if has_canopy else np.zeros_like(tree_grid)
            seg_starts, seg_ends, seg_offsets, seg_counts = _flatten_building_segments(
                building_min_height_grid, self.voxel_size
            )
            _voxelize_kernel(
                voxel_grid,
                land_cover_grid.astype(np.int32, copy=False),
                dem_grid.astype(np.float32, copy=False),
                tree_grid.astype(np.float32, copy=False),
                canopy_in.astype(np.float32, copy=False),
                has_canopy,
                seg_starts,
                seg_ends,
                seg_offsets,
                seg_counts,
                float(trunk_height_ratio),
                float(self.voxel_size),
            )
            return voxel_grid

        for i in range(rows):
            for j in range(cols):
                ground_level = int(dem_grid[i, j] / self.voxel_size + 0.5) + 1
                tree_height = tree_grid[i, j]
                land_cover = land_cover_grid[i, j]

                voxel_grid[i, j, :ground_level] = GROUND_CODE
                voxel_grid[i, j, ground_level - 1] = land_cover

                if tree_height > 0:
                    if canopy_bottom_grid is not None:
                        crown_base_height = canopy_bottom_grid[i, j]
                    else:
                        crown_base_height = (tree_height * trunk_height_ratio)
                    crown_base_height_level = int(crown_base_height / self.voxel_size + 0.5)
                    crown_top_height_level = int(tree_height / self.voxel_size + 0.5)
                    if (crown_top_height_level == crown_base_height_level) and (crown_base_height_level > 0):
                        crown_base_height_level -= 1
                    tree_start = ground_level + crown_base_height_level
                    tree_end = ground_level + crown_top_height_level
                    voxel_grid[i, j, tree_start:tree_end] = TREE_CODE

                for k in building_min_height_grid[i, j]:
                    building_min_height = int(k[0] / self.voxel_size + 0.5)
                    building_height = int(k[1] / self.voxel_size + 0.5)
                    voxel_grid[i, j, ground_level + building_min_height:ground_level + building_height] = BUILDING_CODE

        return voxel_grid

    def generate_components(
        self,
        building_height_grid_ori: np.ndarray,
        land_cover_grid_ori: np.ndarray,
        dem_grid_ori: np.ndarray,
        tree_grid_ori: np.ndarray,
        layered_interval: Optional[int] = None,
        print_class_info: bool = True,
    ):
        _logger.info("Generating 3D voxel data")
        if print_class_info:
            _logger.info(VOXEL_CODE_DESCRIPTIONS)
            _logger.info(LAND_COVER_DESCRIPTIONS)

        if self.land_cover_source == 'OpenStreetMap':
            # OpenStreetMap uses Standard classification, just shift to 1-based
            land_cover_grid_converted = land_cover_grid_ori + 1
        else:
            # All other sources need remapping to standard indices
            land_cover_grid_converted = convert_land_cover(land_cover_grid_ori, land_cover_source=self.land_cover_source)

        building_height_grid = ensure_orientation(
            building_height_grid_ori.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        land_cover_grid = ensure_orientation(
            land_cover_grid_converted.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )
        dem_grid = ensure_orientation(
            dem_grid_ori.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        ) - np.min(dem_grid_ori)
        building_nr_grid = group_and_label_cells(
            ensure_orientation(
                building_height_grid_ori.copy(),
                ORIENTATION_NORTH_UP,
                ORIENTATION_SOUTH_UP,
            )
        )
        dem_grid = process_grid(building_nr_grid, dem_grid)
        tree_grid = ensure_orientation(
            tree_grid_ori.copy(),
            ORIENTATION_NORTH_UP,
            ORIENTATION_SOUTH_UP,
        )

        assert building_height_grid.shape == land_cover_grid.shape == dem_grid.shape == tree_grid.shape, "Input grids must have the same shape"
        rows, cols = building_height_grid.shape
        max_height = int(np.ceil(np.max(building_height_grid + dem_grid + tree_grid) / self.voxel_size))

        land_cover_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
        building_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
        tree_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)
        dem_voxel_grid = np.zeros((rows, cols, max_height), dtype=np.int32)

        for i in range(rows):
            for j in range(cols):
                ground_level = int(dem_grid[i, j] / self.voxel_size + 0.5)
                building_height = int(building_height_grid[i, j] / self.voxel_size + 0.5)
                tree_height = int(tree_grid[i, j] / self.voxel_size + 0.5)
                land_cover = land_cover_grid[i, j]

                dem_voxel_grid[i, j, :ground_level + 1] = -1
                land_cover_voxel_grid[i, j, 0] = land_cover
                if tree_height > 0:
                    tree_voxel_grid[i, j, :tree_height] = -2
                if building_height > 0:
                    building_voxel_grid[i, j, :building_height] = -3

        if not layered_interval:
            layered_interval = max(max_height, int(dem_grid.shape[0] / 4 + 0.5))

        extract_height = min(layered_interval, max_height)
        layered_voxel_grid = np.zeros((rows, cols, layered_interval * 4), dtype=np.int32)
        layered_voxel_grid[:, :, :extract_height] = dem_voxel_grid[:, :, :extract_height]
        layered_voxel_grid[:, :, layered_interval:layered_interval + extract_height] = land_cover_voxel_grid[:, :, :extract_height]
        layered_voxel_grid[:, :, 2 * layered_interval:2 * layered_interval + extract_height] = building_voxel_grid[:, :, :extract_height]
        layered_voxel_grid[:, :, 3 * layered_interval:3 * layered_interval + extract_height] = tree_voxel_grid[:, :, :extract_height]

        return land_cover_voxel_grid, building_voxel_grid, tree_voxel_grid, dem_voxel_grid, layered_voxel_grid


def replace_nan_in_nested(arr, replace_value=10.0):
    if not isinstance(arr, np.ndarray):
        return arr

    result = np.empty_like(arr, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            cell = arr[i, j]
            if cell is None or (isinstance(cell, list) and len(cell) == 0):
                result[i, j] = []
            elif isinstance(cell, list):
                new_cell = []
                for segment in cell:
                    if isinstance(segment, (list, np.ndarray)):
                        if isinstance(segment, np.ndarray):
                            new_segment = np.where(np.isnan(segment), replace_value, segment).tolist()
                        else:
                            new_segment = [replace_value if (isinstance(v, float) and np.isnan(v)) else v for v in segment]
                        new_cell.append(new_segment)
                    else:
                        new_cell.append(segment)
                result[i, j] = new_cell
            else:
                result[i, j] = cell
    return result


