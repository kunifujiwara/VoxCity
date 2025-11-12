from __future__ import annotations

import os
import numpy as np
import trimesh
import matplotlib.cm as cm
import matplotlib.colors as mcolors
try:
    import plotly.graph_objects as go
except ImportError:  # optional dependency
    go = None  # type: ignore

from ..models import VoxCity
from .builder import MeshBuilder
from .palette import get_voxel_color_map
from ..geoprocessor.mesh import create_sim_surface_mesh
try:
    import pyvista as pv
except ImportError:  # optional dependency
    pv = None  # type: ignore


def _rgb_tuple_to_plotly_color(rgb_tuple):
    """
    Convert [R, G, B] or (R, G, B) with 0-255 range to plotly 'rgb(r,g,b)' string.
    """
    try:
        r, g, b = rgb_tuple
        r = int(max(0, min(255, r)))
        g = int(max(0, min(255, g)))
        b = int(max(0, min(255, b)))
        return f"rgb({r},{g},{b})"
    except Exception:
        return "rgb(128,128,128)"


def _mpl_cmap_to_plotly_colorscale(cmap_name, n=256):
    """
    Convert a matplotlib colormap name to a Plotly colorscale list.
    """
    try:
        cmap = cm.get_cmap(cmap_name)
    except Exception:
        cmap = cm.get_cmap('viridis')
    if n < 2:
        n = 2
    scale = []
    for i in range(n):
        x = i / (n - 1)
        r, g, b, _ = cmap(x)
        scale.append([x, f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"])
    return scale


def visualize_voxcity_plotly(
    voxel_array,
    meshsize,
    classes=None,
    voxel_color_map='default',
    opacity=1.0,
    max_dimension=160,
    downsample=None,
    title=None,
    width=1000,
    height=800,
    show=True,
    return_fig=False,
    # Building simulation overlay
    building_sim_mesh=None,
    building_value_name='svf_values',
    building_colormap='viridis',
    building_vmin=None,
    building_vmax=None,
    building_nan_color='gray',
    building_opacity=1.0,
    building_shaded=False,
    render_voxel_buildings=False,
    # Ground simulation surface overlay
    ground_sim_grid=None,
    ground_dem_grid=None,
    ground_z_offset=None,
    ground_view_point_height=None,
    ground_colormap='viridis',
    ground_vmin=None,
    ground_vmax=None,
    sim_surface_opacity=0.95,
    ground_shaded=False,
):
    """
    Interactive 3D visualization using Plotly Mesh3d of voxel faces and optional overlays.
    """
    # Validate optional dependency
    if go is None:
        raise ImportError("Plotly is required for interactive visualization. Install with: pip install plotly")
    # Validate/prepare voxels
    if voxel_array is None or getattr(voxel_array, 'ndim', 0) != 3:
        if building_sim_mesh is None and (ground_sim_grid is None or ground_dem_grid is None):
            raise ValueError("voxel_array must be a 3D numpy array when no overlays are provided")
        vox = None
    else:
        vox = voxel_array

    # Downsample strategy
    stride = 1
    if vox is not None:
        if downsample is not None:
            stride = max(1, int(downsample))
        else:
            nx_tmp, ny_tmp, nz_tmp = vox.shape
            max_dim = max(nx_tmp, ny_tmp, nz_tmp)
            if max_dim > max_dimension:
                stride = int(np.ceil(max_dim / max_dimension))
        if stride > 1:
            # Surface-aware downsampling: stride X/Y, pick topmost non-zero along Z in each window
            orig = voxel_array
            nx0, ny0, nz0 = orig.shape
            xs = orig[::stride, ::stride, :]
            nx_ds, ny_ds, _ = xs.shape
            nz_ds = int(np.ceil(nz0 / float(stride)))
            vox = np.zeros((nx_ds, ny_ds, nz_ds), dtype=orig.dtype)
            for k in range(nz_ds):
                z0w = k * stride
                z1w = min(z0w + stride, nz0)
                W = xs[:, :, z0w:z1w]
                if W.size == 0:
                    continue
                nz_mask = (W != 0)
                has_any = nz_mask.any(axis=2)
                rev_mask = nz_mask[:, :, ::-1]
                idx_rev = rev_mask.argmax(axis=2)
                real_idx = (W.shape[2] - 1) - idx_rev
                gathered = np.take_along_axis(W, real_idx[..., None], axis=2).squeeze(-1)
                vox[:, :, k] = np.where(has_any, gathered, 0)

        nx, ny, nz = vox.shape
        dx = meshsize * stride
        dy = meshsize * stride
        dz = meshsize * stride
        x = np.arange(nx, dtype=float) * dx
        y = np.arange(ny, dtype=float) * dy
        z = np.arange(nz, dtype=float) * dz

        # Choose classes
        if classes is None:
            classes_all = np.unique(vox[vox != 0]).tolist()
        else:
            classes_all = list(classes)
        if building_sim_mesh is not None and getattr(building_sim_mesh, 'vertices', None) is not None:
            classes_to_draw = classes_all if render_voxel_buildings else [c for c in classes_all if int(c) != -3]
        else:
            classes_to_draw = classes_all

        # Resolve colors
        if isinstance(voxel_color_map, dict):
            vox_dict = voxel_color_map
        else:
            vox_dict = get_voxel_color_map(voxel_color_map)

        # Occluder mask (any occupancy)
        if stride > 1:
            def _bool_max_pool_3d(arr_bool, sx):
                if isinstance(sx, (tuple, list, np.ndarray)):
                    sx, sy, sz = int(sx[0]), int(sx[1]), int(sx[2])
                else:
                    sy = sz = int(sx)
                    sx = int(sx)
                a = np.asarray(arr_bool, dtype=bool)
                nx_, ny_, nz_ = a.shape
                px = (sx - (nx_ % sx)) % sx
                py = (sy - (ny_ % sy)) % sy
                pz = (sz - (nz_ % sz)) % sz
                if px or py or pz:
                    a = np.pad(a, ((0, px), (0, py), (0, pz)), constant_values=False)
                nxp, nyp, nzp = a.shape
                a = a.reshape(nxp // sx, sx, nyp // sy, sy, nzp // sz, sz)
                a = a.max(axis=1).max(axis=2).max(axis=4)
                return a
            occluder = _bool_max_pool_3d((voxel_array != 0), stride)
        else:
            occluder = (vox != 0)

        def exposed_face_masks(occ, occ_any):
            p = np.pad(occ_any, ((0,1),(0,0),(0,0)), constant_values=False)
            posx = occ & (~p[1:,:,:])
            p = np.pad(occ_any, ((1,0),(0,0),(0,0)), constant_values=False)
            negx = occ & (~p[:-1,:,:])
            p = np.pad(occ_any, ((0,0),(0,1),(0,0)), constant_values=False)
            posy = occ & (~p[:,1:,:])
            p = np.pad(occ_any, ((0,0),(1,0),(0,0)), constant_values=False)
            negy = occ & (~p[:,:-1,:])
            p = np.pad(occ_any, ((0,0),(0,0),(0,1)), constant_values=False)
            posz = occ & (~p[:,:,1:])
            p = np.pad(occ_any, ((0,0),(0,0),(1,0)), constant_values=False)
            negz = occ & (~p[:,:,:-1])
            return posx, negx, posy, negy, posz, negz

    fig = go.Figure()

    def add_faces(mask, plane, color_rgb):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return
        xi, yi, zi = idx[:,0], idx[:,1], idx[:,2]
        xc = x[xi]; yc = y[yi]; zc = z[zi]
        x0, x1 = xc - dx/2.0, xc + dx/2.0
        y0, y1 = yc - dy/2.0, yc + dy/2.0
        z0, z1 = zc - dz/2.0, zc + dz/2.0

        if plane == '+x':
            vx = np.stack([x1, x1, x1, x1], axis=1)
            vy = np.stack([y0, y1, y1, y0], axis=1)
            vz = np.stack([z0, z0, z1, z1], axis=1)
        elif plane == '-x':
            vx = np.stack([x0, x0, x0, x0], axis=1)
            vy = np.stack([y0, y1, y1, y0], axis=1)
            vz = np.stack([z1, z1, z0, z0], axis=1)
        elif plane == '+y':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y1, y1, y1, y1], axis=1)
            vz = np.stack([z0, z0, z1, z1], axis=1)
        elif plane == '-y':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y0, y0, y0, y0], axis=1)
            vz = np.stack([z1, z1, z0, z0], axis=1)
        elif plane == '+z':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y0, y0, y1, y1], axis=1)
            vz = np.stack([z1, z1, z1, z1], axis=1)
        elif plane == '-z':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y1, y1, y0, y0], axis=1)
            vz = np.stack([z0, z0, z0, z0], axis=1)
        else:
            return

        V = np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])
        n = idx.shape[0]
        starts = np.arange(0, 4*n, 4, dtype=np.int32)
        tris = np.vstack([
            np.stack([starts, starts+1, starts+2], axis=1),
            np.stack([starts, starts+2, starts+3], axis=1)
        ])

        lighting = dict(ambient=0.35, diffuse=1.0, specular=0.4, roughness=0.5, fresnel=0.1)
        cx = (x.min() + x.max()) * 0.5 if len(x) > 0 else 0.0
        cy = (y.min() + y.max()) * 0.5 if len(y) > 0 else 0.0
        cz = (z.min() + z.max()) * 0.5 if len(z) > 0 else 0.0
        lx = cx + (x.max() - x.min() + dx) * 0.9
        ly = cy + (y.max() - y.min() + dy) * 0.6
        lz = cz + (z.max() - z.min() + dz) * 1.4

        fig.add_trace(
            go.Mesh3d(
                x=V[:,0], y=V[:,1], z=V[:,2],
                i=tris[:,0], j=tris[:,1], k=tris[:,2],
                color=_rgb_tuple_to_plotly_color(color_rgb),
                opacity=float(opacity),
                flatshading=False,
                lighting=lighting,
                lightposition=dict(x=lx, y=ly, z=lx),
                name=f"{plane}"
            )
        )

    # Draw voxel faces
    if vox is not None and classes_to_draw:
        for cls in classes_to_draw:
            if not np.any(vox == cls):
                continue
            occ = (vox == cls)
            p = np.pad(occluder, ((0,1),(0,0),(0,0)), constant_values=False); posx = occ & (~p[1:,:,:])
            p = np.pad(occluder, ((1,0),(0,0),(0,0)), constant_values=False); negx = occ & (~p[:-1,:,:])
            p = np.pad(occluder, ((0,0),(0,1),(0,0)), constant_values=False); posy = occ & (~p[:,1:,:])
            p = np.pad(occluder, ((0,0),(1,0),(0,0)), constant_values=False); negy = occ & (~p[:,:-1,:])
            p = np.pad(occluder, ((0,0),(0,0),(0,1)), constant_values=False); posz = occ & (~p[:,:,1:])
            p = np.pad(occluder, ((0,0),(0,0),(1,0)), constant_values=False); negz = occ & (~p[:,:,:-1])
            color_rgb = vox_dict.get(int(cls), [128,128,128])
            add_faces(posx, '+x', color_rgb)
            add_faces(negx, '-x', color_rgb)
            add_faces(posy, '+y', color_rgb)
            add_faces(negy, '-y', color_rgb)
            add_faces(posz, '+z', color_rgb)
            add_faces(negz, '-z', color_rgb)

    # Building overlay
    if building_sim_mesh is not None and getattr(building_sim_mesh, 'vertices', None) is not None:
        Vb = np.asarray(building_sim_mesh.vertices)
        Fb = np.asarray(building_sim_mesh.faces)
        values = None
        if hasattr(building_sim_mesh, 'metadata') and isinstance(building_sim_mesh.metadata, dict):
            values = building_sim_mesh.metadata.get(building_value_name)
        if values is not None:
            values = np.asarray(values)

        face_vals = None
        if values is not None and len(values) == len(Fb):
            face_vals = values.astype(float)
        elif values is not None and len(values) == len(Vb):
            vals_v = values.astype(float)
            face_vals = np.nanmean(vals_v[Fb], axis=1)

        facecolor = None
        if face_vals is not None:
            finite = np.isfinite(face_vals)
            vmin_b = building_vmin if building_vmin is not None else (float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0)
            vmax_b = building_vmax if building_vmax is not None else (float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0)
            norm_b = mcolors.Normalize(vmin=vmin_b, vmax=vmax_b)
            cmap_b = cm.get_cmap(building_colormap)
            colors_rgba = np.zeros((len(Fb), 4), dtype=float)
            colors_rgba[finite] = cmap_b(norm_b(face_vals[finite]))
            nan_rgba = np.array(mcolors.to_rgba(building_nan_color))
            colors_rgba[~finite] = nan_rgba
            facecolor = [f"rgb({int(255*c[0])},{int(255*c[1])},{int(255*c[2])})" for c in colors_rgba]

        lighting_b = (dict(ambient=0.35, diffuse=1.0, specular=0.4, roughness=0.5, fresnel=0.1)
                      if building_shaded else dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0))

        cx = float((Vb[:,0].min() + Vb[:,0].max()) * 0.5)
        cy = float((Vb[:,1].min() + Vb[:,1].max()) * 0.5)
        lx = cx + (Vb[:,0].max() - Vb[:,0].min() + meshsize) * 0.9
        ly = cy + (Vb[:,1].max() - Vb[:,1].min() + meshsize) * 0.6
        lz = float((Vb[:,2].min() + Vb[:,2].max()) * 0.5) + (Vb[:,2].max() - Vb[:,2].min() + meshsize) * 1.4

        fig.add_trace(
            go.Mesh3d(
                x=Vb[:,0], y=Vb[:,1], z=Vb[:,2],
                i=Fb[:,0], j=Fb[:,1], k=Fb[:,2],
                facecolor=facecolor if facecolor is not None else None,
                color=None if facecolor is not None else 'rgb(200,200,200)',
                opacity=float(building_opacity),
                flatshading=False,
                lighting=lighting_b,
                lightposition=dict(x=lx, y=ly, z=lz),
                name=building_value_name if facecolor is not None else 'building_mesh'
            )
        )

        if face_vals is not None:
            colorscale_b = _mpl_cmap_to_plotly_colorscale(building_colormap)
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=0.1, color=[vmin_b, vmax_b], colorscale=colorscale_b, cmin=vmin_b, cmax=vmax_b,
                                colorbar=dict(title=building_value_name, len=0.5, y=0.8), showscale=True),
                    showlegend=False, hoverinfo='skip')
            )

    # Ground simulation surface overlay
    if ground_sim_grid is not None and ground_dem_grid is not None:
        sim_vals = np.asarray(ground_sim_grid, dtype=float)
        finite = np.isfinite(sim_vals)
        vmin_g = ground_vmin if ground_vmin is not None else (float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0)
        vmax_g = ground_vmax if ground_vmax is not None else (float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0)
        z_off = ground_z_offset if ground_z_offset is not None else ground_view_point_height
        try:
            z_off = float(z_off) if z_off is not None else 1.5
        except Exception:
            z_off = 1.5
        try:
            ms = float(meshsize)
            z_off = (z_off // ms + 1.0) * ms
        except Exception:
            pass
        try:
            dem_norm = np.asarray(ground_dem_grid, dtype=float)
            dem_norm = dem_norm - np.nanmin(dem_norm)
        except Exception:
            dem_norm = ground_dem_grid

        sim_mesh = create_sim_surface_mesh(
            ground_sim_grid,
            dem_norm,
            meshsize=meshsize,
            z_offset=z_off,
            cmap_name=ground_colormap,
            vmin=vmin_g,
            vmax=vmax_g,
        )

        if sim_mesh is not None and getattr(sim_mesh, 'vertices', None) is not None:
            V = np.asarray(sim_mesh.vertices)
            F = np.asarray(sim_mesh.faces)
            facecolor = None
            try:
                colors_rgba = np.asarray(sim_mesh.visual.face_colors)
                if colors_rgba.ndim == 2 and colors_rgba.shape[0] == len(F):
                    facecolor = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_rgba]
            except Exception:
                facecolor = None

            lighting = (dict(ambient=0.35, diffuse=1.0, specular=0.4, roughness=0.5, fresnel=0.1)
                        if ground_shaded else dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0))

            cx = float((V[:,0].min() + V[:,0].max()) * 0.5)
            cy = float((V[:,1].min() + V[:,1].max()) * 0.5)
            lx = cx + (V[:,0].max() - V[:,0].min() + meshsize) * 0.9
            ly = cy + (V[:,1].max() - V[:,1].min() + meshsize) * 0.6
            lz = float((V[:,2].min() + V[:,2].max()) * 0.5) + (V[:,2].max() - V[:,2].min() + meshsize) * 1.4

            fig.add_trace(
                go.Mesh3d(
                    x=V[:,0], y=V[:,1], z=V[:,2],
                    i=F[:,0], j=F[:,1], k=F[:,2],
                    facecolor=facecolor,
                    color=None if facecolor is not None else 'rgb(200,200,200)',
                    opacity=float(sim_surface_opacity),
                    flatshading=False,
                    lighting=lighting,
                    lightposition=dict(x=lx, y=ly, z=lz),
                    name='sim_surface'
                )
            )

            colorscale_g = _mpl_cmap_to_plotly_colorscale(ground_colormap)
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=0.1, color=[vmin_g, vmax_g], colorscale=colorscale_g, cmin=vmin_g, cmax=vmax_g,
                                colorbar=dict(title='ground', len=0.5, y=0.2), showscale=True),
                    showlegend=False, hoverinfo='skip')
            )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        )
    )

    if show:
        fig.show()
    if return_fig:
        return fig
    return None


def create_multi_view_scene(meshes, output_directory="output", projection_type="perspective", distance_factor=1.0,
                            image_size: "tuple[int, int] | None" = None):
    """
    Creates multiple rendered views of 3D city meshes from different camera angles.
    """
    if pv is None:
        raise ImportError("PyVista is required for static rendering. Install with: pip install pyvista")
    # NOTE: image_size is now supported via Plotter.window_size when invoked from renderer
    pv_meshes = {}
    for class_id, mesh in meshes.items():
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        faces = np.hstack([[3, *face] for face in mesh.faces])
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        colors = getattr(mesh.visual, 'face_colors', None)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.size and colors.max() > 1:
                colors = colors / 255.0
            pv_mesh.cell_data['colors'] = colors
        pv_meshes[class_id] = pv_mesh

    min_xyz = np.array([np.inf, np.inf, np.inf], dtype=float)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for mesh in meshes.values():
        if mesh is None or len(mesh.vertices) == 0:
            continue
        v = mesh.vertices
        min_xyz = np.minimum(min_xyz, v.min(axis=0))
        max_xyz = np.maximum(max_xyz, v.max(axis=0))
    bbox = np.vstack([min_xyz, max_xyz])

    center = (bbox[1] + bbox[0]) / 2
    diagonal = np.linalg.norm(bbox[1] - bbox[0])

    if projection_type.lower() == "orthographic":
        distance = diagonal * 5
    else:
        distance = diagonal * 1.8 * distance_factor

    iso_angles = {
        'iso_front_right': (1, 1, 0.7),
        'iso_front_left': (-1, 1, 0.7),
        'iso_back_right': (1, -1, 0.7),
        'iso_back_left': (-1, -1, 0.7)
    }

    camera_positions = {}
    for name, direction in iso_angles.items():
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        camera_pos = center + direction * distance
        camera_positions[name] = [camera_pos, center, (0, 0, 1)]

    ortho_views = {
        'xy_top': [center + np.array([0, 0, distance]), center, (-1, 0, 0)],
        'yz_right': [center + np.array([distance, 0, 0]), center, (0, 0, 1)],
        'xz_front': [center + np.array([0, distance, 0]), center, (0, 0, 1)],
        'yz_left': [center + np.array([-distance, 0, 0]), center, (0, 0, 1)],
        'xz_back': [center + np.array([0, -distance, 0]), center, (0, 0, 1)]
    }
    camera_positions.update(ortho_views)

    images = []
    for view_name, camera_pos in camera_positions.items():
        plotter = pv.Plotter(off_screen=True, window_size=image_size if image_size is not None else None)
        if projection_type.lower() == "orthographic":
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = diagonal * 0.4 * distance_factor
        elif projection_type.lower() != "perspective":
            print(f"Warning: Unknown projection_type '{projection_type}'. Using perspective projection.")
        for class_id, pv_mesh in pv_meshes.items():
            has_colors = 'colors' in pv_mesh.cell_data
            plotter.add_mesh(pv_mesh, rgb=True, scalars='colors' if has_colors else None)
        plotter.camera_position = camera_pos
        filename = f'{output_directory}/city_view_{view_name}.png'
        plotter.screenshot(filename)
        images.append((view_name, filename))
        plotter.close()
    return images


def create_rotation_view_scene(
    meshes,
    output_directory: str = "output",
    projection_type: str = "perspective",
    distance_factor: float = 1.0,
    frames_per_segment: int = 60,
    close_loop: bool = False,
    file_prefix: str = "city_rotation",
    image_size: "tuple[int, int] | None" = None,
):
    """
    Creates a sequence of rendered frames forming a smooth isometric rotation that
    passes through: iso_front_right -> iso_front_left -> iso_back_left -> iso_back_right.

    Parameters
    ----------
    meshes : dict[Any, trimesh.Trimesh]
        Dictionary of trimesh meshes keyed by class/label.
    output_directory : str
        Directory to save frames.
    projection_type : str
        "perspective" or "orthographic".
    distance_factor : float
        Camera distance multiplier.
    frames_per_segment : int
        Number of frames between each consecutive isometric anchor.
    close_loop : bool
        If True, also generates frames to return from iso_back_right to iso_front_right.
    file_prefix : str
        Prefix for saved frame filenames.

    Returns
    -------
    list[str]
        List of saved frame file paths in order.
    """
    if pv is None:
        raise ImportError("PyVista is required for static rendering. Install with: pip install pyvista")

    os.makedirs(output_directory, exist_ok=True)

    # Prepare PyVista meshes
    pv_meshes = {}
    for class_id, mesh in meshes.items():
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        faces = np.hstack([[3, *face] for face in mesh.faces])
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        colors = getattr(mesh.visual, 'face_colors', None)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.size and colors.max() > 1:
                colors = colors / 255.0
            pv_mesh.cell_data['colors'] = colors
        pv_meshes[class_id] = pv_mesh

    # Compute scene bounds
    min_xyz = np.array([np.inf, np.inf, np.inf], dtype=float)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for mesh in meshes.values():
        if mesh is None or len(mesh.vertices) == 0:
            continue
        v = mesh.vertices
        min_xyz = np.minimum(min_xyz, v.min(axis=0))
        max_xyz = np.maximum(max_xyz, v.max(axis=0))
    bbox = np.vstack([min_xyz, max_xyz])

    center = (bbox[1] + bbox[0]) / 2
    diagonal = np.linalg.norm(bbox[1] - bbox[0])

    # Camera distance
    if projection_type.lower() == "orthographic":
        distance = diagonal * 5
    else:
        distance = diagonal * 1.8 * distance_factor

    # Define isometric anchor directions and derive constant elevation
    # Anchors correspond to azimuths: 45°, 135°, 225°, 315°
    anchor_azimuths = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    if close_loop:
        anchor_azimuths.append(anchor_azimuths[0] + 2 * np.pi)

    # Use the canonical iso direction (1,1,0.7) to compute elevation angle
    iso_dir = np.array([1.0, 1.0, 0.7], dtype=float)
    iso_dir = iso_dir / np.linalg.norm(iso_dir)
    horiz_len = np.sqrt(iso_dir[0] ** 2 + iso_dir[1] ** 2)
    elevation = np.arctan2(iso_dir[2], horiz_len)  # radians
    cos_elev = np.cos(elevation)
    sin_elev = np.sin(elevation)

    # Generate frames along segments between anchors
    filenames = []
    frame_idx = 0
    num_segments = len(anchor_azimuths) - 1
    for i in range(num_segments):
        a0 = anchor_azimuths[i]
        a1 = anchor_azimuths[i + 1]
        for k in range(frames_per_segment):
            t = k / float(frames_per_segment)
            az = (1.0 - t) * a0 + t * a1
            direction = np.array([
                cos_elev * np.cos(az),
                cos_elev * np.sin(az),
                sin_elev
            ], dtype=float)
            direction = direction / np.linalg.norm(direction)
            camera_pos = center + direction * distance
            camera_tuple = [camera_pos, center, (0, 0, 1)]

            plotter = pv.Plotter(off_screen=True, window_size=image_size if image_size is not None else None)
            if projection_type.lower() == "orthographic":
                plotter.enable_parallel_projection()
                plotter.camera.parallel_scale = diagonal * 0.4 * distance_factor
            elif projection_type.lower() != "perspective":
                print(f"Warning: Unknown projection_type '{projection_type}'. Using perspective projection.")

            for _, pv_mesh in pv_meshes.items():
                has_colors = 'colors' in pv_mesh.cell_data
                plotter.add_mesh(pv_mesh, rgb=True, scalars='colors' if has_colors else None)

            plotter.camera_position = camera_tuple
            filename = os.path.join(output_directory, f"{file_prefix}_{frame_idx:04d}.png")
            plotter.screenshot(filename)
            filenames.append(filename)
            plotter.close()
            frame_idx += 1

    return filenames

class PyVistaRenderer:
    """Renderer that uses PyVista to produce multi-view images from meshes or VoxCity."""

    def render_city(self, city: VoxCity, projection_type: str = "perspective", distance_factor: float = 1.0,
                    output_directory: str = "output", voxel_color_map: "str|dict" = "default",
                    *,  # static rendering specific toggles
                    rotation: bool = False,
                    rotation_frames_per_segment: int = 60,
                    rotation_close_loop: bool = False,
                    rotation_file_prefix: str = "city_rotation",
                    image_size: "tuple[int, int] | None" = None,
                    building_sim_mesh=None, building_value_name: str = 'svf_values',
                    building_colormap: str = 'viridis', building_vmin=None, building_vmax=None,
                    building_nan_color: str = 'gray', building_opacity: float = 1.0,
                    render_voxel_buildings: bool = False,
                    ground_sim_grid=None, ground_dem_grid=None,
                    ground_z_offset: float | None = None, ground_view_point_height: float | None = None,
                    ground_colormap: str = 'viridis', ground_vmin=None, ground_vmax=None):
        """
        Render city to static images with optional simulation overlays.
        
        Parameters
        ----------
        city : VoxCity
            VoxCity object to render
        projection_type : str
            "perspective" or "orthographic"
        distance_factor : float
            Camera distance multiplier
        output_directory : str
            Directory to save rendered images
        voxel_color_map : str or dict
            Color mapping for voxel classes
        rotation : bool
            If True, generate rotating isometric frames instead of multi-view snapshots.
        rotation_frames_per_segment : int
            Number of frames between each isometric anchor when rotation=True.
        rotation_close_loop : bool
            If True, returns smoothly to the starting anchor when rotation=True.
        rotation_file_prefix : str
            Filename prefix for rotation frames when rotation=True.
        image_size : (int, int) or None
            Static rendering output image size (width, height). If None, uses default.
        building_sim_mesh : trimesh.Trimesh, optional
            Building mesh with simulation results
        building_value_name : str
            Metadata key for building values
        building_colormap : str
            Colormap for building values
        building_vmin, building_vmax : float, optional
            Color scale limits for buildings
        building_nan_color : str
            Color for NaN values
        building_opacity : float
            Building mesh opacity
        render_voxel_buildings : bool
            Whether to render voxel buildings when building_sim_mesh is provided
        ground_sim_grid : np.ndarray, optional
            Ground-level simulation grid
        ground_dem_grid : np.ndarray, optional
            DEM grid for ground surface positioning
        ground_z_offset : float, optional
            Height offset for ground surface
        ground_view_point_height : float, optional
            Alternative height parameter
        ground_colormap : str
            Colormap for ground values
        ground_vmin, ground_vmax : float, optional
            Color scale limits for ground
        """
        if pv is None:
            raise ImportError("PyVista is required for static rendering. Install with: pip install pyvista")
        
        meshsize = city.voxels.meta.meshsize
        trimesh_dict = {}
        
        # Build voxel meshes (always generate to show ground, trees, etc.)
        collection = MeshBuilder.from_voxel_grid(city.voxels, meshsize=meshsize, voxel_color_map=voxel_color_map)
        for key, mm in collection.items.items():
            if mm.vertices.size == 0 or mm.faces.size == 0:
                continue
            # Skip building voxels if we have building_sim_mesh and don't want to render both
            if not render_voxel_buildings and building_sim_mesh is not None and int(key) == -3:
                continue
            tri = trimesh.Trimesh(vertices=mm.vertices, faces=mm.faces, process=False)
            if mm.colors is not None:
                tri.visual.face_colors = mm.colors
            trimesh_dict[key] = tri
        
        # Add building simulation mesh overlay
        if building_sim_mesh is not None and getattr(building_sim_mesh, 'vertices', None) is not None:
            Vb = np.asarray(building_sim_mesh.vertices)
            Fb = np.asarray(building_sim_mesh.faces)
            
            # Get simulation values from metadata
            values = None
            if hasattr(building_sim_mesh, 'metadata') and isinstance(building_sim_mesh.metadata, dict):
                values = building_sim_mesh.metadata.get(building_value_name)
            
            if values is not None:
                values = np.asarray(values)
                
                # Determine if values are per-face or per-vertex
                face_vals = None
                if len(values) == len(Fb):
                    face_vals = values.astype(float)
                elif len(values) == len(Vb):
                    vals_v = values.astype(float)
                    face_vals = np.nanmean(vals_v[Fb], axis=1)
                
                if face_vals is not None:
                    # Apply colormap
                    finite = np.isfinite(face_vals)
                    vmin_b = building_vmin if building_vmin is not None else (float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0)
                    vmax_b = building_vmax if building_vmax is not None else (float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0)
                    norm_b = mcolors.Normalize(vmin=vmin_b, vmax=vmax_b)
                    cmap_b = cm.get_cmap(building_colormap)
                    
                    colors_rgba = np.zeros((len(Fb), 4), dtype=np.uint8)
                    if np.any(finite):
                        colors_float = cmap_b(norm_b(face_vals[finite]))
                        colors_rgba[finite] = (colors_float * 255).astype(np.uint8)
                    
                    # Handle NaN values
                    nan_rgba = np.array(mcolors.to_rgba(building_nan_color))
                    colors_rgba[~finite] = (nan_rgba * 255).astype(np.uint8)
                    
                    # Create trimesh with colors
                    building_tri = trimesh.Trimesh(vertices=Vb, faces=Fb, process=False)
                    building_tri.visual.face_colors = colors_rgba
                    trimesh_dict['building_sim'] = building_tri
            else:
                # No values, just add the mesh with default color
                building_tri = trimesh.Trimesh(vertices=Vb, faces=Fb, process=False)
                trimesh_dict['building_sim'] = building_tri
        
        # Add ground simulation surface overlay
        if ground_sim_grid is not None and ground_dem_grid is not None:
            z_off = ground_z_offset if ground_z_offset is not None else ground_view_point_height
            try:
                z_off = float(z_off) if z_off is not None else 1.5
            except Exception:
                z_off = 1.5
            
            # Snap to grid
            try:
                z_off = (z_off // meshsize + 1.0) * meshsize
            except Exception:
                pass
            
            # Normalize DEM
            try:
                dem_norm = np.asarray(ground_dem_grid, dtype=float)
                dem_norm = dem_norm - np.nanmin(dem_norm)
            except Exception:
                dem_norm = ground_dem_grid
            
            # Determine color range
            sim_vals = np.asarray(ground_sim_grid, dtype=float)
            finite = np.isfinite(sim_vals)
            vmin_g = ground_vmin if ground_vmin is not None else (float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0)
            vmax_g = ground_vmax if ground_vmax is not None else (float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0)
            
            # Create ground simulation mesh
            sim_mesh = create_sim_surface_mesh(
                ground_sim_grid,
                dem_norm,
                meshsize=meshsize,
                z_offset=z_off,
                cmap_name=ground_colormap,
                vmin=vmin_g,
                vmax=vmax_g,
            )
            
            if sim_mesh is not None and getattr(sim_mesh, 'vertices', None) is not None:
                trimesh_dict['ground_sim'] = sim_mesh
        
        os.makedirs(output_directory, exist_ok=True)
        if rotation:
            return create_rotation_view_scene(
                trimesh_dict,
                output_directory=output_directory,
                projection_type=projection_type,
                distance_factor=distance_factor,
                frames_per_segment=rotation_frames_per_segment,
                close_loop=rotation_close_loop,
                file_prefix=rotation_file_prefix,
                image_size=image_size,
            )
        else:
            return create_multi_view_scene(
                trimesh_dict,
                output_directory=output_directory,
                projection_type=projection_type,
                distance_factor=distance_factor,
                image_size=image_size,
            )



def visualize_voxcity(
    city: VoxCity,
    mode: str = "interactive",
    *,
    # Common options
    voxel_color_map: "str|dict" = "default",
    classes=None,
    title: str | None = None,
    # Interactive (Plotly) options
    opacity: float = 1.0,
    max_dimension: int = 160,
    downsample: int | None = None,
    show: bool = True,
    return_fig: bool = False,
    # Static (PyVista) options
    output_directory: str = "output",
    projection_type: str = "perspective",
    distance_factor: float = 1.0,
    rotation: bool = False,
    rotation_frames_per_segment: int = 60,
    rotation_close_loop: bool = False,
    rotation_file_prefix: str = "city_rotation",
    image_size: "tuple[int, int] | None" = None,
    # Building simulation overlay options
    building_sim_mesh=None,
    building_value_name: str = 'svf_values',
    building_colormap: str = 'viridis',
    building_vmin: float | None = None,
    building_vmax: float | None = None,
    building_nan_color: str = 'gray',
    building_opacity: float = 1.0,
    building_shaded: bool = False,
    render_voxel_buildings: bool = False,
    # Ground simulation surface overlay options
    ground_sim_grid=None,
    ground_dem_grid=None,
    ground_z_offset: float | None = None,
    ground_view_point_height: float | None = None,
    ground_colormap: str = 'viridis',
    ground_vmin: float | None = None,
    ground_vmax: float | None = None,
    sim_surface_opacity: float = 0.95,
    ground_shaded: bool = False,
):
    """
    Visualize a VoxCity object with optional simulation result overlays.

    Parameters
    ----------
    city : VoxCity
        VoxCity object to visualize
    mode : str, default="interactive"
        Visualization mode: "interactive" (Plotly) or "static" (PyVista)
    
    Common Options
    --------------
    voxel_color_map : str or dict, default="default"
        Color mapping for voxel classes
    classes : list, optional
        Specific voxel classes to render
    title : str, optional
        Plot title
    image_size : (int, int) or None, default=None
        Unified image size (width, height) applied across modes.
        - Interactive: overrides width/height below when provided.
        - Static (including rotation): sets PyVista window size for screenshots.
    
    Interactive Mode Options (Plotly)
    ----------------------------------
    opacity : float, default=1.0
        Voxel opacity (0-1)
    max_dimension : int, default=160
        Maximum grid dimension before downsampling
    downsample : int, optional
        Manual downsampling stride
    show : bool, default=True
        Whether to display the plot
    return_fig : bool, default=False
        Whether to return the figure object
    
    Static Mode Options (PyVista)
    ------------------------------
    output_directory : str, default="output"
        Directory for saving rendered images
    projection_type : str, default="perspective"
        Camera projection: "perspective" or "orthographic"
    distance_factor : float, default=1.0
        Camera distance multiplier
    rotation : bool, default=False
        If True, generate rotating isometric frames instead of multi-view snapshots
    rotation_frames_per_segment : int, default=60
        Frames between each isometric anchor when rotation=True
    rotation_close_loop : bool, default=False
        If True, continue frames to return to start when rotation=True
    rotation_file_prefix : str, default="city_rotation"
        Filename prefix for rotation frames when rotation=True
    image_size : (int, int) or None, default=None
        Static rendering output image size (width, height). If None, uses default.
    
    Building Simulation Overlay Options
    ------------------------------------
    building_sim_mesh : trimesh.Trimesh, optional
        Building mesh with simulation results in metadata.
        Typically created by get_surface_view_factor() or get_building_solar_irradiance().
    building_value_name : str, default='svf_values'
        Metadata key to use for coloring (e.g., 'svf_values', 'global', 'direct', 'diffuse')
    building_colormap : str, default='viridis'
        Matplotlib colormap for building values
    building_vmin : float, optional
        Minimum value for color scale
    building_vmax : float, optional
        Maximum value for color scale
    building_nan_color : str, default='gray'
        Color for NaN/invalid values
    building_opacity : float, default=1.0
        Building mesh opacity (0-1)
    building_shaded : bool, default=False
        Whether to apply shading to building mesh
    render_voxel_buildings : bool, default=False
        Whether to render voxel buildings when building_sim_mesh is provided
    
    Ground Simulation Surface Overlay Options
    ------------------------------------------
    ground_sim_grid : np.ndarray, optional
        2D array of ground-level simulation values (e.g., Green View Index, solar radiation).
        Should have the same shape as the city's 2D grids.
    ground_dem_grid : np.ndarray, optional
        2D DEM array for positioning the ground simulation surface.
        If None, uses city.dem.elevation when ground_sim_grid is provided.
    ground_z_offset : float, optional
        Height offset for ground simulation surface above DEM
    ground_view_point_height : float, optional
        Alternative parameter for ground surface height (used if ground_z_offset is None)
    ground_colormap : str, default='viridis'
        Matplotlib colormap for ground values
    ground_vmin : float, optional
        Minimum value for color scale
    ground_vmax : float, optional
        Maximum value for color scale
    sim_surface_opacity : float, default=0.95
        Ground simulation surface opacity (0-1)
    ground_shaded : bool, default=False
        Whether to apply shading to ground surface
    
    Returns
    -------
    For mode="interactive":
        plotly.graph_objects.Figure or None
        Returns Figure if return_fig=True, otherwise None
    
    For mode="static":
        list of (view_name, filepath) tuples
        List of rendered view names and their file paths
    
    Examples
    --------
    Basic visualization:
    >>> visualize_voxcity(city, mode="interactive")
    
    With building solar irradiance results:
    >>> building_mesh = get_building_solar_irradiance(city, ...)
    >>> visualize_voxcity(city, mode="interactive",
    ...                   building_sim_mesh=building_mesh,
    ...                   building_value_name='global')
    
    With ground-level Green View Index:
    >>> visualize_voxcity(city, mode="interactive",
    ...                   ground_sim_grid=gvi_array,
    ...                   ground_colormap='YlGn')
    
    Static rendering with simulation overlays:
    >>> visualize_voxcity(city, mode="static",
    ...                   building_sim_mesh=svf_mesh,
    ...                   output_directory="renders")
    """
    if not isinstance(mode, str):
        raise ValueError("mode must be a string: 'interactive' or 'static'")

    mode_l = mode.lower().strip()
    meshsize = getattr(city.voxels.meta, "meshsize", None)
    
    # Auto-fill ground_dem_grid from city if ground_sim_grid is provided but ground_dem_grid is not
    if ground_sim_grid is not None and ground_dem_grid is None:
        ground_dem_grid = getattr(city.dem, "elevation", None)
    
    if mode_l == "interactive":
        voxel_array = getattr(city.voxels, "classes", None)
        # Build kwargs to optionally pass width/height when image_size is provided
        size_kwargs = {}
        if image_size is not None:
            try:
                size_kwargs = {"width": int(image_size[0]), "height": int(image_size[1])}
            except Exception:
                size_kwargs = {}
        return visualize_voxcity_plotly(
            voxel_array=voxel_array,
            meshsize=meshsize,
            classes=classes,
            voxel_color_map=voxel_color_map,
            opacity=opacity,
            max_dimension=max_dimension,
            downsample=downsample,
            title=title,
            show=show,
            return_fig=return_fig,
            **size_kwargs,
            # Building simulation overlay
            building_sim_mesh=building_sim_mesh,
            building_value_name=building_value_name,
            building_colormap=building_colormap,
            building_vmin=building_vmin,
            building_vmax=building_vmax,
            building_nan_color=building_nan_color,
            building_opacity=building_opacity,
            building_shaded=building_shaded,
            render_voxel_buildings=render_voxel_buildings,
            # Ground simulation surface overlay
            ground_sim_grid=ground_sim_grid,
            ground_dem_grid=ground_dem_grid,
            ground_z_offset=ground_z_offset,
            ground_view_point_height=ground_view_point_height,
            ground_colormap=ground_colormap,
            ground_vmin=ground_vmin,
            ground_vmax=ground_vmax,
            sim_surface_opacity=sim_surface_opacity,
            ground_shaded=ground_shaded,
        )

    if mode_l == "static":
        renderer = PyVistaRenderer()
        return renderer.render_city(
            city,
            projection_type=projection_type,
            distance_factor=distance_factor,
            output_directory=output_directory,
            voxel_color_map=voxel_color_map,
            rotation=rotation,
            rotation_frames_per_segment=rotation_frames_per_segment,
            rotation_close_loop=rotation_close_loop,
            rotation_file_prefix=rotation_file_prefix,
            image_size=image_size,
            # Pass simulation overlay parameters
            building_sim_mesh=building_sim_mesh,
            building_value_name=building_value_name,
            building_colormap=building_colormap,
            building_vmin=building_vmin,
            building_vmax=building_vmax,
            building_nan_color=building_nan_color,
            building_opacity=building_opacity,
            render_voxel_buildings=render_voxel_buildings,
            ground_sim_grid=ground_sim_grid,
            ground_dem_grid=ground_dem_grid,
            ground_z_offset=ground_z_offset,
            ground_view_point_height=ground_view_point_height,
            ground_colormap=ground_colormap,
            ground_vmin=ground_vmin,
            ground_vmax=ground_vmax,
        )

    raise ValueError("Unknown mode. Use 'interactive' or 'static'.")

