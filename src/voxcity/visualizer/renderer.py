from __future__ import annotations

import os
import numpy as np
import trimesh
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from ..models import VoxCity
from .builder import MeshBuilder
from .palette import get_voxel_color_map
from ..geoprocessor.mesh import create_sim_surface_mesh
import pyvista as pv


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


def create_multi_view_scene(meshes, output_directory="output", projection_type="perspective", distance_factor=1.0):
    """
    Creates multiple rendered views of 3D city meshes from different camera angles.
    """
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
        plotter = pv.Plotter(off_screen=True)
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


class PyVistaRenderer:
    """Renderer that uses PyVista to produce multi-view images from meshes or VoxCity."""

    def render_city(self, city: VoxCity, projection_type: str = "perspective", distance_factor: float = 1.0,
                    output_directory: str = "output", voxel_color_map: "str|dict" = "default"):
        collection = MeshBuilder.from_voxel_grid(city.voxels, meshsize=city.voxels.meta.meshsize, voxel_color_map=voxel_color_map)
        trimesh_dict = {}
        for key, mm in collection.items.items():
            if mm.vertices.size == 0 or mm.faces.size == 0:
                continue
            tri = trimesh.Trimesh(vertices=mm.vertices, faces=mm.faces, process=False)
            if mm.colors is not None:
                tri.visual.face_colors = mm.colors
            trimesh_dict[key] = tri
        os.makedirs(output_directory, exist_ok=True)
        return create_multi_view_scene(trimesh_dict, output_directory=output_directory,
                                       projection_type=projection_type, distance_factor=distance_factor)


