import numpy as np
from numba import njit, prange


@njit
def calculate_transmittance(length, tree_k=0.6, tree_lad=1.0):
    return np.exp(-tree_k * tree_lad * length)


@njit
def trace_ray_generic(voxel_data, origin, direction, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    nx, ny, nz = voxel_data.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction

    length = np.sqrt(dx*dx + dy*dy + dz*dz)
    if length == 0.0:
        return False, 1.0
    dx /= length
    dy /= length
    dz /= length

    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)

    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1

    EPSILON = 1e-10

    if abs(dx) > EPSILON:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1 / dx)
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if abs(dy) > EPSILON:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1 / dy)
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    if abs(dz) > EPSILON:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1 / dz)
    else:
        t_max_z = np.inf
        t_delta_z = np.inf

    cumulative_transmittance = 1.0
    last_t = 0.0

    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        voxel_value = voxel_data[i, j, k]

        t_next = min(t_max_x, t_max_y, t_max_z)
        segment_length = (t_next - last_t) * meshsize
        if segment_length < 0.0:
            segment_length = 0.0

        if voxel_value == -2:
            transmittance = calculate_transmittance(segment_length, tree_k, tree_lad)
            cumulative_transmittance *= transmittance
            if cumulative_transmittance < 0.01:
                if inclusion_mode:
                    return False, cumulative_transmittance
                else:
                    return True, cumulative_transmittance

        if inclusion_mode:
            for hv in hit_values:
                if voxel_value == hv:
                    return True, cumulative_transmittance
            if voxel_value != 0 and voxel_value != -2:
                return False, cumulative_transmittance
        else:
            in_set = False
            for hv in hit_values:
                if voxel_value == hv:
                    in_set = True
                    break
            if not in_set and voxel_value != -2:
                return True, cumulative_transmittance

        last_t = t_next

        TIE_EPS = 1e-12
        eq_x = abs(t_max_x - t_next) <= TIE_EPS
        eq_y = abs(t_max_y - t_next) <= TIE_EPS
        eq_z = abs(t_max_z - t_next) <= TIE_EPS

        if inclusion_mode and ((eq_x and eq_y) or (eq_x and eq_z) or (eq_y and eq_z)):
            if eq_x:
                ii = i + step_x
                if 0 <= ii < nx:
                    val = voxel_data[ii, j, k]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance
            if eq_y:
                jj = j + step_y
                if 0 <= jj < ny:
                    val = voxel_data[i, jj, k]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance
            if eq_z:
                kk = k + step_z
                if 0 <= kk < nz:
                    val = voxel_data[i, j, kk]
                    is_target = False
                    for hv in hit_values:
                        if val == hv:
                            is_target = True
                            break
                    if (val != 0) and (val != -2) and (not is_target):
                        return False, cumulative_transmittance

        stepped = False
        if eq_x:
            t_max_x += t_delta_x
            i += step_x
            stepped = True
        if eq_y:
            t_max_y += t_delta_y
            j += step_y
            stepped = True
        if eq_z:
            t_max_z += t_delta_z
            k += step_z
            stepped = True

        if not stepped:
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    t_max_x += t_delta_x; i += step_x
                else:
                    t_max_z += t_delta_z; k += step_z
            else:
                if t_max_y < t_max_z:
                    t_max_y += t_delta_y; j += step_y
                else:
                    t_max_z += t_delta_z; k += step_z

    return False, cumulative_transmittance


@njit
def compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    total_rays = ray_directions.shape[0]
    visibility_sum = 0.0
    for idx in range(total_rays):
        direction = ray_directions[idx]
        hit, value = trace_ray_generic(voxel_data, observer_location, direction, hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
        if inclusion_mode:
            if hit:
                if -2 in hit_values:
                    contrib = 1.0 - max(0.0, min(1.0, value))
                    visibility_sum += contrib
                else:
                    visibility_sum += 1.0
        else:
            if not hit:
                visibility_sum += value
    return visibility_sum / total_rays


@njit(parallel=True)
def compute_vi_map_generic(voxel_data, ray_directions, view_height_voxel, hit_values, meshsize, tree_k, tree_lad, inclusion_mode=True):
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)
    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        vi_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        vi_value = compute_vi_generic(observer_location, voxel_data, ray_directions, hit_values, meshsize, tree_k, tree_lad, inclusion_mode)
                        vi_map[x, y] = vi_value
                        found_observer = True
                        break
            if not found_observer:
                vi_map[x, y] = np.nan
    return np.flipud(vi_map)


def _prepare_masks_for_vi(voxel_data: np.ndarray, hit_values, inclusion_mode: bool):
    is_tree = (voxel_data == -2)
    if inclusion_mode:
        is_target = np.isin(voxel_data, hit_values)
        is_blocker_inc = (voxel_data != 0) & (~is_tree) & (~is_target)
        return is_tree, is_target, None, is_blocker_inc
    else:
        is_allowed = np.isin(voxel_data, hit_values)
        return is_tree, None, is_allowed, None


@njit(cache=True, fastmath=True)
def _trace_ray_inclusion_masks(is_tree, is_target, is_blocker_inc, origin, direction, meshsize, tree_k, tree_lad):
    nx, ny, nz = is_tree.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction
    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return False, 1.0
    dx /= length; dy /= length; dz /= length
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1
    EPS = 1e-10
    if abs(dx) > EPS:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = np.inf; t_delta_x = np.inf
    if abs(dy) > EPS:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = np.inf; t_delta_y = np.inf
    if abs(dz) > EPS:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = np.inf; t_delta_z = np.inf
    cumulative_transmittance = 1.0
    last_t = 0.0
    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        t_next = t_max_x
        axis = 0
        if t_max_y < t_next:
            t_next = t_max_y; axis = 1
        if t_max_z < t_next:
            t_next = t_max_z; axis = 2
        segment_length = (t_next - last_t) * meshsize
        if segment_length < 0.0:
            segment_length = 0.0
        if is_tree[i, j, k]:
            trans = np.exp(-tree_k * tree_lad * segment_length)
            cumulative_transmittance *= trans
            if cumulative_transmittance < 1e-2:
                return False, cumulative_transmittance
        if is_target[i, j, k]:
            return True, cumulative_transmittance
        if is_blocker_inc[i, j, k]:
            return False, cumulative_transmittance
        last_t = t_next
        if axis == 0:
            t_max_x += t_delta_x; i += step_x
        elif axis == 1:
            t_max_y += t_delta_y; j += step_y
        else:
            t_max_z += t_delta_z; k += step_z
    return False, cumulative_transmittance


@njit(cache=True, fastmath=True)
def _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, meshsize, tree_k, tree_lad):
    nx, ny, nz = is_tree.shape
    x0, y0, z0 = origin
    dx, dy, dz = direction
    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return False, 1.0
    dx /= length; dy /= length; dz /= length
    x, y, z = x0 + 0.5, y0 + 0.5, z0 + 0.5
    i, j, k = int(x0), int(y0), int(z0)
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1
    EPS = 1e-10
    if abs(dx) > EPS:
        t_max_x = ((i + (step_x > 0)) - x) / dx
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = np.inf; t_delta_x = np.inf
    if abs(dy) > EPS:
        t_max_y = ((j + (step_y > 0)) - y) / dy
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = np.inf; t_delta_y = np.inf
    if abs(dz) > EPS:
        t_max_z = ((k + (step_z > 0)) - z) / dz
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = np.inf; t_delta_z = np.inf
    cumulative_transmittance = 1.0
    last_t = 0.0
    while (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz):
        t_next = t_max_x
        axis = 0
        if t_max_y < t_next:
            t_next = t_max_y; axis = 1
        if t_max_z < t_next:
            t_next = t_max_z; axis = 2
        segment_length = (t_next - last_t) * meshsize
        if segment_length < 0.0:
            segment_length = 0.0
        if is_tree[i, j, k]:
            trans = np.exp(-tree_k * tree_lad * segment_length)
            cumulative_transmittance *= trans
            if cumulative_transmittance < 1e-2:
                return True, cumulative_transmittance
        if (not is_allowed[i, j, k]) and (not is_tree[i, j, k]):
            return True, cumulative_transmittance
        last_t = t_next
        if axis == 0:
            t_max_x += t_delta_x; i += step_x
        elif axis == 1:
            t_max_y += t_delta_y; j += step_y
        else:
            t_max_z += t_delta_z; k += step_z
    return False, cumulative_transmittance


@njit(parallel=True, cache=True, fastmath=True)
def _compute_vi_map_generic_fast(voxel_data, ray_directions, view_height_voxel, meshsize, tree_k, tree_lad, is_tree, is_target, is_allowed, is_blocker_inc, inclusion_mode, trees_in_targets):
    nx, ny, nz = voxel_data.shape
    vi_map = np.full((nx, ny), np.nan)
    obs_base_z = _precompute_observer_base_z(voxel_data)
    for x in prange(nx):
        for y in range(ny):
            base_z = obs_base_z[x, y]
            if base_z < 0:
                vi_map[x, y] = np.nan
                continue
            below = voxel_data[x, y, base_z]
            if (below == 7) or (below == 8) or (below == 9) or (below < 0):
                vi_map[x, y] = np.nan
                continue
            oz = base_z + 1 + view_height_voxel
            obs = np.array([x, y, oz], dtype=np.float64)
            visibility_sum = 0.0
            n_rays = ray_directions.shape[0]
            for r in range(n_rays):
                direction = ray_directions[r]
                if inclusion_mode:
                    hit, value = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker_inc, obs, direction, meshsize, tree_k, tree_lad)
                    if hit:
                        if trees_in_targets:
                            contrib = 1.0 - max(0.0, min(1.0, value))
                            visibility_sum += contrib
                        else:
                            visibility_sum += 1.0
                else:
                    hit, value = _trace_ray_exclusion_masks(is_tree, is_allowed, obs, direction, meshsize, tree_k, tree_lad)
                    if not hit:
                        visibility_sum += value
            vi_map[x, y] = visibility_sum / n_rays
    return np.flipud(vi_map)


@njit(cache=True, fastmath=True)
def _precompute_observer_base_z(voxel_data):
    nx, ny, nz = voxel_data.shape
    out = np.empty((nx, ny), dtype=np.int32)
    for x in range(nx):
        for y in range(ny):
            found = False
            for z in range(1, nz):
                v_above = voxel_data[x, y, z]
                v_base = voxel_data[x, y, z - 1]
                if (v_above == 0 or v_above == -2) and not (v_base == 0 or v_base == -2):
                    out[x, y] = z - 1
                    found = True
                    break
            if not found:
                out[x, y] = -1
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _trace_ray(vox_is_tree, vox_is_opaque, origin, target, att, att_cutoff):
    nx, ny, nz = vox_is_opaque.shape
    x0, y0, z0 = origin[0], origin[1], origin[2]
    x1, y1, z1 = target[0], target[1], target[2]
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    if length == 0.0:
        return True
    inv_len = 1.0 / length
    dx *= inv_len; dy *= inv_len; dz *= inv_len
    x = x0 + 0.5
    y = y0 + 0.5
    z = z0 + 0.5
    i = int(x0); j = int(y0); k = int(z0)
    step_x = 1 if dx >= 0.0 else -1
    step_y = 1 if dy >= 0.0 else -1
    step_z = 1 if dz >= 0.0 else -1
    BIG = 1e30
    if dx != 0.0:
        t_max_x = (((i + (1 if step_x > 0 else 0)) - x) / dx)
        t_delta_x = abs(1.0 / dx)
    else:
        t_max_x = BIG; t_delta_x = BIG
    if dy != 0.0:
        t_max_y = (((j + (1 if step_y > 0 else 0)) - y) / dy)
        t_delta_y = abs(1.0 / dy)
    else:
        t_max_y = BIG; t_delta_y = BIG
    if dz != 0.0:
        t_max_z = (((k + (1 if step_z > 0 else 0)) - z) / dz)
        t_delta_z = abs(1.0 / dz)
    else:
        t_max_z = BIG; t_delta_z = BIG
    T = 1.0
    ti = int(x1); tj = int(y1); tk = int(z1)
    while True:
        if (i < 0) or (i >= nx) or (j < 0) or (j >= ny) or (k < 0) or (k >= nz):
            return False
        if vox_is_opaque[i, j, k]:
            return False
        if vox_is_tree[i, j, k]:
            T *= att
            if T < att_cutoff:
                return False
        if (i == ti) and (j == tj) and (k == tk):
            return True
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                t_max_x += t_delta_x; i += step_x
            else:
                t_max_z += t_delta_z; k += step_z
        else:
            if t_max_y < t_max_z:
                t_max_y += t_delta_y; j += step_y
            else:
                t_max_z += t_delta_z; k += step_z

