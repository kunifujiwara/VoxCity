import numpy as np
import trimesh
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def create_voxel_mesh(voxel_array, class_id, meshsize=1.0, building_id_grid=None):
    """
    Create a mesh from voxels preserving sharp edges, scaled by meshsize.

    Parameters
    ----------
    voxel_array : np.ndarray (3D)
        The voxel array of shape (X, Y, Z).
    class_id : int
        The ID of the class to extract.
    meshsize : float
        The real-world size of each voxel in meters, for x, y, and z.
    building_id_grid : np.ndarray (2D), optional
        2D grid of building IDs, shape (X, Y). Used when class_id=-3 (buildings).

    Returns
    -------
    mesh : trimesh.Trimesh or None
        The resulting mesh for the given class_id (or None if no voxels).
        If class_id=-3, mesh.metadata['building_id'] contains building IDs.
    """
    # Find voxels of the current class
    voxel_coords = np.argwhere(voxel_array == class_id)

    if building_id_grid is not None:
        building_id_grid_flipud = np.flipud(building_id_grid)

    if len(voxel_coords) == 0:
        return None

    # Define the 6 faces of a unit cube (local coordinates 0..1)
    unit_faces = np.array([
        # Front
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        # Back
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        # Right
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
        # Left
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        # Top
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        # Bottom
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]
    ])

    # Define face normals
    face_normals = np.array([
        [0, 0, 1],   # Front
        [0, 0, -1],  # Back
        [1, 0, 0],   # Right
        [-1, 0, 0],  # Left
        [0, 1, 0],   # Top
        [0, -1, 0]   # Bottom
    ])

    vertices = []
    faces = []
    face_normals_list = []
    building_ids = []  # List to store building IDs for each face

    for x, y, z in voxel_coords:
        # For buildings, get the building ID from the grid
        building_id = None
        if class_id == -3 and building_id_grid is not None:
            building_id = building_id_grid_flipud[x, y]
            
        # Check each face of the current voxel
        adjacent_coords = [
            (x,   y,   z+1),  # Front
            (x,   y,   z-1),  # Back
            (x+1, y,   z),    # Right
            (x-1, y,   z),    # Left
            (x,   y+1, z),    # Top
            (x,   y-1, z)     # Bottom
        ]

        # Only create faces where there's a transition between this class and "not this class"
        for face_idx, adj_coord in enumerate(adjacent_coords):
            try:
                # If adj_coord is outside array bounds, it's a boundary => face is visible
                if adj_coord[0] < 0 or adj_coord[1] < 0 or adj_coord[2] < 0:
                    is_boundary = True
                else:
                    adj_value = voxel_array[adj_coord]
                    is_boundary = (adj_value == 0 or adj_value != class_id)
            except IndexError:
                # Out of range => boundary
                is_boundary = True

            if is_boundary:
                # Local face in (0..1) for x,y,z, then shift by voxel coords
                face_verts = (unit_faces[face_idx] + np.array([x, y, z])) * meshsize
                current_vert_count = len(vertices)

                vertices.extend(face_verts)
                # Convert quad to two triangles
                faces.extend([
                    [current_vert_count, current_vert_count + 1, current_vert_count + 2],
                    [current_vert_count, current_vert_count + 2, current_vert_count + 3]
                ])
                # Add face normals for both triangles
                face_normals_list.extend([face_normals[face_idx], face_normals[face_idx]])
                
                # Store building ID for both triangles if this is a building
                if class_id == -3 and building_id_grid is not None:
                    building_ids.extend([building_id, building_id])

    if not vertices:
        return None

    vertices = np.array(vertices)
    faces = np.array(faces)
    face_normals_list = np.array(face_normals_list)

    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals_list
    )

    # Merge vertices that are at the same position
    mesh.merge_vertices()
    
    # Add building IDs as metadata for buildings
    if class_id == -3 and building_id_grid is not None and building_ids:
        mesh.metadata = {'building_id': np.array(building_ids)}

    return mesh

def create_sim_surface_mesh(sim_grid, dem_grid,
                            meshsize=1.0, z_offset=1.5,
                            cmap_name='viridis',
                            vmin=None, vmax=None):
    """
    Create a planar surface mesh from sim_grid located at dem_grid + z_offset.
    Skips any cells in sim_grid that are NaN, and flips both sim_grid and dem_grid
    (up-down) to match voxel_array orientation. Applies meshsize scaling in x,y.

    Parameters
    ----------
    sim_grid : 2D np.ndarray
        2D array of simulation values (e.g., Green View Index).
    dem_grid : 2D np.ndarray
        2D array of ground elevations (same shape as sim_grid).
    meshsize : float
        Size of each cell in meters (same in x and y).
    z_offset : float
        Additional offset added to dem_grid for placing the mesh.
    cmap_name : str
        Matplotlib colormap name. Default is 'viridis'.
    vmin : float or None
        Minimum value for color mapping. If None, use min of sim_grid (non-NaN).
    vmax : float or None
        Maximum value for color mapping. If None, use max of sim_grid (non-NaN).

    Returns
    -------
    trimesh.Trimesh or None
        A single mesh containing one square face per non-NaN cell.
        Returns None if there are no valid cells.
    """
    # Flip arrays vertically
    sim_grid_flipped = np.flipud(sim_grid)
    dem_grid_flipped = np.flipud(dem_grid)

    # Identify valid (non-NaN) values
    valid_mask = ~np.isnan(sim_grid_flipped)
    valid_values = sim_grid_flipped[valid_mask]
    if valid_values.size == 0:
        return None

    # If vmin/vmax not provided, use actual min/max of the valid sim data
    if vmin is None:
        vmin = np.nanmin(valid_values) 
    if vmax is None:
        vmax = np.nanmax(valid_values)
        
    # Prepare the colormap and create colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    
    # Create a figure just for the colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    plt.colorbar(scalar_map, cax=ax, orientation='horizontal')
    plt.tight_layout()
    plt.close()

    vertices = []
    faces = []
    face_colors = []

    vert_index = 0
    nrows, ncols = sim_grid_flipped.shape

    # Build a quad (two triangles) for each valid cell
    for x in range(nrows):
        for y in range(ncols):
            val = sim_grid_flipped[x, y]
            if np.isnan(val):
                continue

            z_base = meshsize * int(dem_grid_flipped[x, y] / meshsize + 1.5) + z_offset            

            # 4 corners in (x,y)*meshsize
            v0 = [ x      * meshsize,  y      * meshsize, z_base ]
            v1 = [(x + 1) * meshsize,  y      * meshsize, z_base ]
            v2 = [(x + 1) * meshsize, (y + 1) * meshsize, z_base ]
            v3 = [ x      * meshsize, (y + 1) * meshsize, z_base ]

            vertices.extend([v0, v1, v2, v3])
            faces.extend([
                [vert_index,     vert_index + 1, vert_index + 2],
                [vert_index,     vert_index + 2, vert_index + 3]
            ])

            # Get color from colormap
            color_rgba = np.array(scalar_map.to_rgba(val))  # shape (4,)

            # Each cell has 2 faces => add the color twice
            face_colors.append(color_rgba)
            face_colors.append(color_rgba)

            vert_index += 4

    if len(vertices) == 0:
        return None

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)
    face_colors = np.array(face_colors, dtype=float)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors,
        process=False  # skip auto merge if you want to preserve quads
    )

    return mesh

def create_city_meshes(voxel_array, vox_dict, meshsize=1.0):
    """
    Create meshes from voxel data with sharp edges preserved.
    Applies meshsize for voxel scaling.
    """
    meshes = {}

    # Convert RGB colors to hex for material properties
    color_dict = {k: mcolors.rgb2hex([v[0]/255, v[1]/255, v[2]/255])
                 for k, v in vox_dict.items() if k != 0}  # Exclude air

    # Create vertices and faces for each object class
    for class_id in np.unique(voxel_array):
        if class_id == 0:  # Skip air
            continue

        try:
            mesh = create_voxel_mesh(voxel_array, class_id, meshsize=meshsize)

            if mesh is None:
                continue

            # Convert hex color to RGBA
            rgb_color = np.array(mcolors.hex2color(color_dict[class_id]))
            rgba_color = np.concatenate([rgb_color, [1.0]])

            # Assign color to all faces
            mesh.visual.face_colors = np.tile(rgba_color, (len(mesh.faces), 1))

            meshes[class_id] = mesh

        except ValueError as e:
            print(f"Skipping class {class_id}: {e}")

    return meshes

def export_meshes(meshes, output_directory, base_filename):
    """
    Export meshes to OBJ (with MTL) and STL formats.
    """
    # Export combined mesh as OBJ with materials
    combined_mesh = trimesh.util.concatenate(list(meshes.values()))
    combined_mesh.export(f"{output_directory}/{base_filename}.obj")

    # Export individual meshes as STL
    for class_id, mesh in meshes.items():
        # Convert class_id to a string for filename
        mesh.export(f"{output_directory}/{base_filename}_{class_id}.stl")