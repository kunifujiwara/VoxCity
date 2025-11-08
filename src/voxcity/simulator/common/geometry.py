import numpy as np
from numba import njit


def _generate_ray_directions_grid(N_azimuth: int, N_elevation: int, elevation_min_degrees: float, elevation_max_degrees: float) -> np.ndarray:
    azimuth_angles = np.linspace(0.0, 2.0 * np.pi, int(N_azimuth), endpoint=False)
    elevation_angles = np.deg2rad(
        np.linspace(float(elevation_min_degrees), float(elevation_max_degrees), int(N_elevation))
    )
    ray_directions = np.empty((len(azimuth_angles) * len(elevation_angles), 3), dtype=np.float64)
    out_idx = 0
    for elevation in elevation_angles:
        cos_elev = np.cos(elevation)
        sin_elev = np.sin(elevation)
        for azimuth in azimuth_angles:
            dx = cos_elev * np.cos(azimuth)
            dy = cos_elev * np.sin(azimuth)
            dz = sin_elev
            ray_directions[out_idx, 0] = dx
            ray_directions[out_idx, 1] = dy
            ray_directions[out_idx, 2] = dz
            out_idx += 1
    return ray_directions


def _generate_ray_directions_fibonacci(N_rays: int, elevation_min_degrees: float, elevation_max_degrees: float) -> np.ndarray:
    N = int(max(1, N_rays))
    emin = np.deg2rad(float(elevation_min_degrees))
    emax = np.deg2rad(float(elevation_max_degrees))
    z_min = np.sin(min(emin, emax))
    z_max = np.sin(max(emin, emax))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(N, dtype=np.float64)
    z = z_min + (i + 0.5) * (z_max - z_min) / N
    phi = i * golden_angle
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack((x, y, z), axis=1).astype(np.float64)


@njit
def rotate_vector_axis_angle(vec, axis, angle):
    axis_len = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if axis_len < 1e-12:
        return vec
    ux, uy, uz = axis / axis_len
    c = np.cos(angle)
    s = np.sin(angle)
    dot = vec[0]*ux + vec[1]*uy + vec[2]*uz
    cross_x = uy*vec[2] - uz*vec[1]
    cross_y = uz*vec[0] - ux*vec[2]
    cross_z = ux*vec[1] - uy*vec[0]
    v_rot = np.zeros(3, dtype=np.float64)
    v_rot[0] = vec[0] * c
    v_rot[1] = vec[1] * c
    v_rot[2] = vec[2] * c
    v_rot[0] += cross_x * s
    v_rot[1] += cross_y * s
    v_rot[2] += cross_z * s
    tmp = dot * (1.0 - c)
    v_rot[0] += ux * tmp
    v_rot[1] += uy * tmp
    v_rot[2] += uz * tmp
    return v_rot


@njit(cache=True, fastmath=True, nogil=True)
def _build_face_basis(normal):
    nx = normal[0]; ny = normal[1]; nz = normal[2]
    nrm = (nx*nx + ny*ny + nz*nz) ** 0.5
    if nrm < 1e-12:
        return (np.array((1.0, 0.0, 0.0)),
                np.array((0.0, 1.0, 0.0)),
                np.array((0.0, 0.0, 1.0)))
    invn = 1.0 / nrm
    nx *= invn; ny *= invn; nz *= invn
    n = np.array((nx, ny, nz))
    if abs(nz) < 0.999:
        helper = np.array((0.0, 0.0, 1.0))
    else:
        helper = np.array((1.0, 0.0, 0.0))
    ux = helper[1]*n[2] - helper[2]*n[1]
    uy = helper[2]*n[0] - helper[0]*n[2]
    uz = helper[0]*n[1] - helper[1]*n[0]
    ul = (ux*ux + uy*uy + uz*uz) ** 0.5
    if ul < 1e-12:
        u = np.array((1.0, 0.0, 0.0))
    else:
        invul = 1.0 / ul
        u = np.array((ux*invul, uy*invul, uz*invul))
    vx = n[1]*u[2] - n[2]*u[1]
    vy = n[2]*u[0] - n[0]*u[2]
    vz = n[0]*u[1] - n[1]*u[0]
    v = np.array((vx, vy, vz))
    return u, v, n


