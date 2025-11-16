from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, Callable

import numpy as np

from ..models import VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid


def _compute_center_crop_indices(size: int, target: int) -> Tuple[int, int]:
    if size <= target:
        return 0, size
    start = max(0, (size - target) // 2)
    end = start + target
    return start, end


def _pad_split(total_pad: int) -> Tuple[int, int]:
    # Split padding for centering; put extra on the bottom/right side
    a = total_pad // 2
    b = total_pad - a
    return a, b


def _pad_crop_2d(
    arr: np.ndarray,
    target_xy: Tuple[int, int],
    pad_value: Any,
    align_xy: str = "center",
    allow_crop_xy: bool = True,
) -> np.ndarray:
    x, y = arr.shape[:2]
    tx, ty = int(target_xy[0]), int(target_xy[1])

    # Crop (center) if needed and allowed
    if allow_crop_xy and (x > tx or y > ty):
        if align_xy == "center":
            xs, xe = _compute_center_crop_indices(x, tx) if x > tx else (0, x)
            ys, ye = _compute_center_crop_indices(y, ty) if y > ty else (0, y)
        else:
            # top-left alignment: crop from bottom/right only
            xs, xe = (0, tx) if x > tx else (0, x)
            ys, ye = (0, ty) if y > ty else (0, y)
        arr = arr[xs:xe, ys:ye]
        x, y = arr.shape[:2]

    # Pad to target
    px = max(0, tx - x)
    py = max(0, ty - y)

    if px == 0 and py == 0:
        return arr

    if align_xy == "center":
        px0, px1 = _pad_split(px)
        py0, py1 = _pad_split(py)
    else:
        # top-left: pad only on bottom/right
        px0, px1 = 0, px
        py0, py1 = 0, py

    if arr.ndim == 2:
        pad_width = ((px0, px1), (py0, py1))
    else:
        # Preserve trailing dims (e.g., channels)
        pad_width = ((px0, px1), (py0, py1)) + tuple((0, 0) for _ in range(arr.ndim - 2))

    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def _pad_crop_3d_zbottom(
    arr: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: Any,
    align_xy: str = "center",
    allow_crop_xy: bool = True,
    allow_crop_z: bool = False,
) -> np.ndarray:
    nx, ny, nz = arr.shape
    tx, ty, tz = int(target_shape[0]), int(target_shape[1]), int(target_shape[2])

    # XY crop/pad
    arr_xy = _pad_crop_2d(arr, (tx, ty), pad_value, align_xy=align_xy, allow_crop_xy=allow_crop_xy)
    nx, ny, nz = arr_xy.shape

    # Z handling: keep ground at z=0; pad only at the top by default
    if nz > tz:
        if allow_crop_z:
            arr_xy = arr_xy[:, :, :tz]
        else:
            tz = nz  # expand target to avoid cropping
    elif nz < tz:
        pad_top = tz - nz  # add empty air above
        arr_xy = np.pad(arr_xy, ((0, 0), (0, 0), (0, pad_top)), mode="constant", constant_values=pad_value)

    return arr_xy


def normalize_voxcity_shape(
    city: VoxCity,
    target_shape: Tuple[int, int, int],
    *,
    align_xy: str = "center",
    pad_values: Optional[Dict[str, Any]] = None,
    allow_crop_xy: bool = True,
    allow_crop_z: bool = False,
) -> VoxCity:
    """
    Return a new VoxCity with arrays padded/cropped to target (x, y, z).

    - XY alignment can be 'center' (default) or 'top-left'.
    - Z padding is added at the TOP to preserve ground level at z=0.
    - By default, Z is never cropped (allow_crop_z=False). If target z is smaller than current,
      target z is expanded to current to avoid losing data.
    """
    if pad_values is None:
        pad_values = {}

    # Resolve pad values for each layer
    pv_vox = pad_values.get("voxels", 0)
    pv_lc = pad_values.get("land_cover", 0)
    pv_dem = pad_values.get("dem", 0.0)
    pv_bh = pad_values.get("building_heights", 0.0)
    pv_bid = pad_values.get("building_ids", 0)
    pv_canopy = pad_values.get("canopy", 0.0)
    pv_bmin = pad_values.get("building_min_heights_factory", None)  # callable creating empty cell, default []
    if pv_bmin is None:
        def _empty_list() -> list:
            return []
        pv_bmin = _empty_list
    elif not callable(pv_bmin):
        const_val = pv_bmin
        pv_bmin = (lambda v=const_val: v)

    # Source arrays
    vox = city.voxels.classes
    bh = city.buildings.heights
    bmin = city.buildings.min_heights
    bid = city.buildings.ids
    lc = city.land_cover.classes
    dem = city.dem.elevation
    can_top = city.tree_canopy.top
    can_bot = city.tree_canopy.bottom if city.tree_canopy.bottom is not None else None

    # Normalize shapes
    vox_n = _pad_crop_3d_zbottom(
        vox.astype(vox.dtype, copy=False),
        target_shape,
        pad_value=np.array(pv_vox, dtype=vox.dtype),
        align_xy=align_xy,
        allow_crop_xy=allow_crop_xy,
        allow_crop_z=allow_crop_z,
    )
    tx, ty, tz = vox_n.shape

    def _pad2d(a: np.ndarray, pad_val: Any) -> np.ndarray:
        return _pad_crop_2d(
            a.astype(a.dtype, copy=False),
            (tx, ty),
            pad_value=np.array(pad_val, dtype=a.dtype) if not isinstance(pad_val, (list, tuple, dict)) else pad_val,
            align_xy=align_xy,
            allow_crop_xy=allow_crop_xy,
        )

    bh_n = _pad2d(bh, pv_bh)
    bid_n = _pad2d(bid, pv_bid)
    lc_n = _pad2d(lc, pv_lc)
    dem_n = _pad2d(dem, pv_dem)
    can_top_n = _pad2d(can_top, pv_canopy)
    can_bot_n = _pad2d(can_bot, pv_canopy) if can_bot is not None else None  # type: ignore

    # Object-dtype 2D padding/cropping for min_heights
    # Center-crop if needed, then pad with empty lists
    bx, by = bmin.shape
    if bx > tx or by > ty:
        if align_xy == "center":
            xs, xe = _compute_center_crop_indices(bx, tx) if bx > tx else (0, bx)
            ys, ye = _compute_center_crop_indices(by, ty) if by > ty else (0, by)
        else:
            xs, xe = (0, tx) if bx > tx else (0, bx)
            ys, ye = (0, ty) if by > ty else (0, by)
        bmin_c = bmin[xs:xe, ys:ye]
    else:
        bmin_c = bmin
    bx, by = bmin_c.shape
    px = max(0, tx - bx)
    py = max(0, ty - by)
    if px or py:
        out = np.empty((tx, ty), dtype=object)
        # Fill with empty factory values
        for i in range(tx):
            for j in range(ty):
                out[i, j] = pv_bmin()
        if align_xy == "center":
            px0, px1 = _pad_split(px)
            py0, py1 = _pad_split(py)
        else:
            px0, py0 = 0, 0
            px1, py1 = px, py
        out[px0:px0 + bx, py0:py0 + by] = bmin_c
        bmin_n = out
    else:
        bmin_n = bmin_c

    # Rebuild VoxCity with normalized arrays and same metadata (meshsize/bounds)
    meta = city.voxels.meta
    voxels_new = VoxelGrid(classes=vox_n, meta=meta)
    buildings_new = BuildingGrid(heights=bh_n, min_heights=bmin_n, ids=bid_n, meta=meta)
    land_new = LandCoverGrid(classes=lc_n, meta=meta)
    dem_new = DemGrid(elevation=dem_n, meta=meta)
    canopy_new = CanopyGrid(top=can_top_n, bottom=can_bot_n, meta=meta)

    city_new = VoxCity(
        voxels=voxels_new,
        buildings=buildings_new,
        land_cover=land_new,
        dem=dem_new,
        tree_canopy=canopy_new,
        extras=dict(city.extras) if city.extras is not None else {},
    )
    # Keep extras canopy mirrors in sync if present
    try:
        city_new.extras["canopy_top"] = can_top_n
        if can_bot_n is not None:
            city_new.extras["canopy_bottom"] = can_bot_n
    except Exception:
        pass
    return city_new


