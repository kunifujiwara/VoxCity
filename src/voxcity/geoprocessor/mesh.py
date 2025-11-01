"""Mesh generation utilities for voxel and 2D grid visualization.

Orientation contract:
- Mesh builders expect 2D inputs (e.g., simulation grids, building_id grids)
  to be provided in north_up orientation (row 0 = north/top) with columns
  increasing eastward (col 0 = west/left). Any internal flips are
  implementation details to match mesh coordinates.
"""

import numpy as np
import os
import trimesh
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def create_voxel_mesh(voxel_array, class_id, meshsize=1.0, building_id_grid=None, mesh_type=None):
    """
    Create a 3D mesh from voxels preserving sharp edges, scaled by meshsize.
    
    This function converts a 3D voxel array into a triangulated mesh, where each voxel
    face is converted into two triangles. The function preserves sharp edges between
    different classes and handles special cases for buildings.
    
    Parameters
    ----------
    voxel_array : np.ndarray (3D)
        The voxel array of shape (X, Y, Z) where each cell contains a class ID.
        - 0: typically represents void/air
        - -2: typically represents trees
        - -3: typically represents buildings
        Other values can represent different classes as defined by the application.
    
    class_id : int
        The ID of the class to extract. Only voxels with this ID will be included
        in the output mesh.
    
    meshsize : float, default=1.0
        The real-world size of each voxel in meters, applied uniformly to x, y, and z
        dimensions. Used to scale the output mesh to real-world coordinates.
    
    building_id_grid : np.ndarray (2D), optional
        2D grid of building IDs with shape (X, Y). Only used when class_id=-3 (buildings).
        Each cell contains a unique identifier for the building at that location.
        This allows tracking which faces belong to which building.
    
    mesh_type : str, optional
        Type of mesh to create, controlling which faces are included:
        - None (default): create faces at boundaries between different classes
        - 'building_solar' or 'open_air': only create faces at boundaries between
                          buildings (-3) and either void (0) or trees (-2). Useful for
                          solar analysis where only exposed surfaces matter.

    Returns
    -------
    mesh : trimesh.Trimesh or None
        The resulting triangulated mesh for the given class_id. Returns None if no
        voxels of the specified class are found.
        
        The mesh includes:
        - vertices: 3D coordinates of each vertex
        - faces: triangles defined by vertex indices
        - face_normals: normal vectors for each face
        - metadata: If class_id=-3, includes 'building_id' mapping faces to buildings
    
    Examples
    --------
    Basic usage for a simple voxel array:
    >>> voxels = np.zeros((10, 10, 10))
    >>> voxels[4:7, 4:7, 0:5] = 1  # Create a simple column
    >>> mesh = create_voxel_mesh(voxels, class_id=1, meshsize=0.5)
    
    Creating a building mesh with IDs:
    >>> building_ids = np.zeros((10, 10))
    >>> building_ids[4:7, 4:7] = 1  # Mark building #1
    >>> mesh = create_voxel_mesh(voxels, class_id=-3, 
    ...                         building_id_grid=building_ids,
    ...                         meshsize=1.0)
    
    Notes
    -----
    - The function creates faces only at boundaries between different classes or at
      the edges of the voxel array.
    - Each face is split into two triangles for compatibility with graphics engines.
    - Face normals are computed to ensure correct lighting and rendering.
    - For buildings (class_id=-3), building IDs are tracked to maintain building identity.
    - The mesh preserves sharp edges, which is important for architectural visualization.
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

        # Only create faces where there's a transition based on mesh_type
        for face_idx, adj_coord in enumerate(adjacent_coords):
            try:
                # If adj_coord is outside array bounds, it's a boundary => face is visible
                if adj_coord[0] < 0 or adj_coord[1] < 0 or adj_coord[2] < 0:
                    is_boundary = True
                else:
                    adj_value = voxel_array[adj_coord]
                    
                    if class_id == -3 and mesh_type in ('building_solar', 'open_air'):
                        # Only create faces at boundaries with void (0) or trees (-2)
                        is_boundary = (adj_value == 0 or adj_value == -2)
                    else:
                        # Default behavior - create faces at any class change
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
    
    # Ensure metadata dict exists
    if not hasattr(mesh, 'metadata') or mesh.metadata is None:
        mesh.metadata = {}

    # Store intended per-triangle normals to avoid reliance on auto-computed normals
    mesh.metadata['provided_face_normals'] = face_normals_list

    # Add building IDs as metadata for buildings
    if class_id == -3 and building_id_grid is not None and building_ids:
        mesh.metadata['building_id'] = np.array(building_ids)

    return mesh

def create_sim_surface_mesh(sim_grid, dem_grid,
                            meshsize=1.0, z_offset=1.5,
                            cmap_name='viridis',
                            vmin=None, vmax=None):
    """
    Create a colored planar surface mesh from simulation data, positioned above a DEM.
    
    This function generates a 3D visualization mesh for 2D simulation results (like
    Green View Index, solar radiation, etc.). The mesh is positioned above the Digital
    Elevation Model (DEM) by a specified offset, and colored according to the simulation
    values using a matplotlib colormap.
    
    Parameters
    ----------
    sim_grid : 2D np.ndarray
        2D array of simulation values (e.g., Green View Index, solar radiation).
        NaN values in this grid will be skipped in the output mesh.
        The grid should be oriented with north at the top.
    
    dem_grid : 2D np.ndarray
        2D array of ground elevations in meters. Must have the same shape as sim_grid.
        Used to position the visualization mesh at the correct height above terrain.
    
    meshsize : float, default=1.0
        Size of each cell in meters. Applied uniformly to x and y dimensions.
        Determines the resolution of the output mesh.
    
    z_offset : float, default=1.5
        Additional height offset in meters added to dem_grid elevations.
        Used to position the visualization above ground level for better visibility.
    
    cmap_name : str, default='viridis'
        Matplotlib colormap name used for coloring the mesh based on sim_grid values.
        Common options:
        - 'viridis': Default, perceptually uniform, colorblind-friendly
        - 'RdYlBu': Red-Yellow-Blue, good for diverging data
        - 'jet': Rainbow colormap (not recommended for scientific visualization)
    
    vmin : float, optional
        Minimum value for color mapping. If None, uses min of sim_grid (excluding NaN).
        Used to control the range of the colormap.
    
    vmax : float, optional
        Maximum value for color mapping. If None, uses max of sim_grid (excluding NaN).
        Used to control the range of the colormap.
    
    Returns
    -------
    mesh : trimesh.Trimesh or None
        A single mesh containing one colored square face (two triangles) per non-NaN cell.
        Returns None if there are no valid (non-NaN) cells in sim_grid.
        
        The mesh includes:
        - vertices: 3D coordinates of each vertex
        - faces: triangles defined by vertex indices
        - face_colors: RGBA colors for each face based on sim_grid values
        - visual: trimesh.visual.ColorVisuals object storing the face colors
    
    Examples
    --------
    Basic usage with Green View Index data:
    >>> gvi = np.array([[0.5, 0.6], [0.4, 0.8]])  # GVI values
    >>> dem = np.array([[10.0, 10.2], [9.8, 10.1]])  # Ground heights
    >>> mesh = create_sim_surface_mesh(gvi, dem, meshsize=1.0, z_offset=1.5)
    
    Custom color range and colormap:
    >>> mesh = create_sim_surface_mesh(gvi, dem,
    ...                               cmap_name='RdYlBu',
    ...                               vmin=0.0, vmax=1.0)
    
    Notes
    -----
    - The function automatically creates a matplotlib colorbar figure for visualization
    - Both input grids are flipped vertically to match the voxel_array orientation
    - Each grid cell is converted to two triangles for compatibility with 3D engines
    - The mesh is positioned at dem_grid + z_offset to float above the terrain
    - Face colors are interpolated from the colormap based on sim_grid values
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

            # Match voxel ground rounding: int(dem/mesh + 0.5) + 1 == int(dem/mesh + 1.5)
            # Then lower the plane by one mesh layer as requested
            z_base = meshsize * int(dem_grid_flipped[x, y] / meshsize + 1.5) + z_offset - meshsize            

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

def create_city_meshes(
    voxel_array,
    vox_dict,
    meshsize=1.0,
    include_classes=None,
    exclude_classes=None,
):
    """
    Create a collection of colored 3D meshes representing different city elements.
    
    This function processes a voxelized city model and creates separate meshes for
    different urban elements (buildings, trees, etc.), each with its own color.
    The function preserves sharp edges and applies appropriate colors based on the
    provided color dictionary.
    
    Parameters
    ----------
    voxel_array : np.ndarray (3D)
        3D array representing the voxelized city model. Each voxel contains a class ID
        that maps to an urban element type:
        - 0: Void/air (automatically skipped)
        - -2: Trees
        - -3: Buildings
        Other values can represent different urban elements as defined in vox_dict.
    
    vox_dict : dict
        Dictionary mapping class IDs to RGB colors. Each entry should be:
        {class_id: [R, G, B]} where R, G, B are 0-255 integer values.
        Example: {-3: [200, 200, 200], -2: [0, 255, 0]} for grey buildings and
        green trees. The key 0 (air) is automatically excluded.
    
    meshsize : float, default=1.0
        Size of each voxel in meters, applied uniformly to x, y, and z dimensions.
        Used to scale the output meshes to real-world coordinates.
    
    Returns
    -------
    meshes : dict
        Dictionary mapping class IDs to their corresponding trimesh.Trimesh objects.
        Each mesh includes:
        - vertices: 3D coordinates scaled by meshsize
        - faces: triangulated faces preserving sharp edges
        - face_colors: RGBA colors from vox_dict
        - visual: trimesh.visual.ColorVisuals object storing the face colors
        
        Classes with no voxels are automatically excluded from the output.
    
    Examples
    --------
    Basic usage with buildings and trees:
    >>> voxels = np.zeros((10, 10, 10))
    >>> voxels[4:7, 4:7, 0:5] = -3  # Add a building
    >>> voxels[2:4, 2:4, 0:3] = -2  # Add some trees
    >>> colors = {
    ...     -3: [200, 200, 200],  # Grey buildings
    ...     -2: [0, 255, 0]       # Green trees
    ... }
    >>> meshes = create_city_meshes(voxels, colors, meshsize=1.0)
    
    Notes
    -----
    - The function automatically skips class_id=0 (typically air/void)
    - Each urban element type gets its own separate mesh for efficient rendering
    - Colors are converted from RGB [0-255] to RGBA [0-1] format
    - Sharp edges are preserved to maintain architectural features
    - Empty classes (no voxels) are automatically excluded from the output
    - Errors during mesh creation for a class are caught and reported
    """
    meshes = {}

    # Convert RGB colors to hex for material properties
    color_dict = {k: mcolors.rgb2hex([v[0]/255, v[1]/255, v[2]/255])
                 for k, v in vox_dict.items() if k != 0}  # Exclude air

    # Determine which classes to process
    unique_classes = np.unique(voxel_array)

    if include_classes is not None:
        # Only keep classes explicitly requested (and present in the data)
        class_iterable = [c for c in include_classes if c in unique_classes]
    else:
        class_iterable = list(unique_classes)

    exclude_set = set(exclude_classes) if exclude_classes is not None else set()

    # Create vertices and faces for each object class
    for class_id in class_iterable:
        if class_id == 0:  # Skip air
            continue

        if class_id in exclude_set:
            # Explicitly skipped (e.g., will be replaced with custom mesh)
            continue

        try:
            mesh = create_voxel_mesh(voxel_array, class_id, meshsize=meshsize)

            if mesh is None:
                continue

            # Convert hex color to RGBA
            if class_id not in color_dict:
                # Color not provided; skip silently for robustness
                continue
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
    Export a collection of meshes to both OBJ (with MTL) and STL formats.
    
    This function exports meshes in two ways:
    1. A single combined OBJ file with materials (and associated MTL file)
    2. Separate STL files for each mesh, named with their class IDs
    
    Parameters
    ----------
    meshes : dict
        Dictionary mapping class IDs to trimesh.Trimesh objects.
        Each mesh should have:
        - vertices: 3D coordinates
        - faces: triangulated faces
        - face_colors: RGBA colors (if using materials)
    
    output_directory : str
        Directory path where the output files will be saved.
        Will be created if it doesn't exist.
    
    base_filename : str
        Base name for the output files (without extension).
        Will be used to create:
        - {base_filename}.obj : Combined mesh with materials
        - {base_filename}.mtl : Material definitions for OBJ
        - {base_filename}_{class_id}.stl : Individual STL files
    
    Returns
    -------
    None
        Files are written directly to the specified output directory.
    
    Examples
    --------
    >>> meshes = {
    ...     -3: building_mesh,  # Building mesh with grey color
    ...     -2: tree_mesh      # Tree mesh with green color
    ... }
    >>> export_meshes(meshes, 'output/models', 'city_model')
    
    This will create:
    - output/models/city_model.obj
    - output/models/city_model.mtl
    - output/models/city_model_-3.stl
    - output/models/city_model_-2.stl
    
    Notes
    -----
    - OBJ/MTL format preserves colors and materials but is more complex
    - STL format is simpler but doesn't support colors
    - STL files are exported separately for each class for easier processing
    - The OBJ file combines all meshes while preserving their materials
    - File extensions are automatically added to the base filename
    """
    # Export combined mesh as OBJ with materials
    combined_mesh = trimesh.util.concatenate(list(meshes.values()))
    combined_mesh.export(f"{output_directory}/{base_filename}.obj")

    # Export individual meshes as STL
    for class_id, mesh in meshes.items():
        # Convert class_id to a string for filename
        mesh.export(f"{output_directory}/{base_filename}_{class_id}.stl")

def split_vertices_manual(mesh):
    """
    Split a mesh into independent faces by duplicating shared vertices.
    
    This function imitates trimesh's split_vertices() functionality but ensures
    complete face independence by giving each face its own copy of vertices.
    This is particularly useful for rendering applications where smooth shading
    between faces is undesirable, such as architectural visualization in Rhino.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to split. Should have:
        - vertices: array of vertex coordinates
        - faces: array of vertex indices forming triangles
        - visual: Optional ColorVisuals object with face colors
    
    Returns
    -------
    out_mesh : trimesh.Trimesh
        New mesh where each face is completely independent, with:
        - Duplicated vertices for each face
        - No vertex sharing between faces
        - Preserved face colors if present in input
        - Each face as a separate component
    
    Examples
    --------
    Basic usage:
    >>> vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    >>> faces = np.array([[0,1,2], [0,2,3]])  # Two triangles sharing vertices
    >>> mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    >>> split_mesh = split_vertices_manual(mesh)
    >>> print(f"Original vertices: {len(mesh.vertices)}")  # 4 vertices
    >>> print(f"Split vertices: {len(split_mesh.vertices)}")  # 6 vertices
    
    With face colors:
    >>> colors = np.array([[255,0,0,255], [0,255,0,255]])  # Red and green faces
    >>> mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=colors)
    >>> split_mesh = split_vertices_manual(mesh)  # Colors are preserved
    
    Notes
    -----
    - Each output face has exactly 3 unique vertices
    - Face colors are preserved in the output mesh
    - Useful for:
        - Preventing smooth shading artifacts
        - Ensuring face color independence
        - Preparing meshes for CAD software
        - Creating sharp edges in architectural models
    - Memory usage increases as vertices are duplicated
    """
    new_meshes = []
    
    # For each face, build a small, one-face mesh
    for face_idx, face in enumerate(mesh.faces):
        face_coords = mesh.vertices[face]
        
        # Create mini-mesh without colors first
        mini_mesh = trimesh.Trimesh(
            vertices=face_coords,
            faces=[[0, 1, 2]],
            process=False  # skip merging/cleaning
        )
        
        # If the mesh has per-face colors, set the face color properly
        if (mesh.visual.face_colors is not None 
            and len(mesh.visual.face_colors) == len(mesh.faces)):
            # Create a visual object with the face color (for one face)
            face_color = mesh.visual.face_colors[face_idx]
            color_visual = trimesh.visual.ColorVisuals(
                mesh=mini_mesh,
                face_colors=np.array([face_color]),  # One face, one color
                vertex_colors=None
            )
            mini_mesh.visual = color_visual
        
        new_meshes.append(mini_mesh)
    
    # Concatenate all the single-face meshes
    out_mesh = trimesh.util.concatenate(new_meshes)
    return out_mesh

def save_obj_from_colored_mesh(meshes, output_path, base_filename, max_materials=None):
    """
    Memory-safe OBJ/MTL exporter.
    - Streams vertices/faces to disk (no concatenate, no per-face mini-meshes).
    - Uses face colors -> materials (no vertex splitting).
    - Optional color quantization to reduce material count.
    """
    import os
    import numpy as np

    os.makedirs(output_path, exist_ok=True)
    obj_path = os.path.join(output_path, f"{base_filename}.obj")
    mtl_path = os.path.join(output_path, f"{base_filename}.mtl")

    # --------------- helpers ---------------
    def to_uint8_rgba(arr):
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            # Handle float [0..1] or int [0..255]
            if arr.dtype.kind == 'f':
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.shape[1] == 3:
            alpha = np.full((arr.shape[0], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=1)
        return arr

    # First pass: build material palette
    # We avoid collecting all colors at once—scan per mesh and update a dict.
    color_to_id = {}
    ordered_colors = []  # list of RGBA uint8 tuples in material order

    # Optional quantizer (lazy-init)
    quantizer = None
    if max_materials is not None:
        try:
            from sklearn.cluster import MiniBatchKMeans
            quantizer = MiniBatchKMeans(n_clusters=max_materials, random_state=42, batch_size=8192)
            # Partial-fit streaming pass over colors
            for m in meshes.values():
                fc = getattr(m.visual, "face_colors", None)
                if fc is None:
                    continue
                fc = to_uint8_rgba(fc)
                if fc.size == 0: 
                    continue
                # Use only RGB for clustering
                quantizer.partial_fit(fc[:, :3].astype(np.float32))
        except ImportError:
            raise ImportError("scikit-learn is required for color quantization. Install it with: pip install scikit-learn")

    # Assign material ids during a second scan, but still streaming to avoid big unions
    def get_material_id(rgba):
        key = (int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
        mid = color_to_id.get(key)
        if mid is None:
            mid = len(ordered_colors)
            color_to_id[key] = mid
            ordered_colors.append(key)
        return mid

    # 2nd pass if quantizing: we need color centroids
    centers_u8 = None
    if quantizer is not None:
        centers = quantizer.cluster_centers_.astype(np.float32)  # RGB float
        centers = np.clip(centers, 0.0, 255.0).astype(np.uint8)
        # Build a quick LUT fun
        def quantize_rgb(rgb_u8):
            # rgb_u8: (N,3) uint8 -> labels -> centers
            labels = quantizer.predict(rgb_u8.astype(np.float32))
            return centers[labels]
        # We'll convert each mesh's face colors to quantized RGB on the fly
        centers_u8 = centers

    # Build materials palette by scanning once (still O(total faces) but tiny memory)
    for m in meshes.values():
        fc = getattr(m.visual, "face_colors", None)
        if fc is None:
            # No colors: assign default grey
            rgba = np.array([[200,200,200,255]], dtype=np.uint8)
            get_material_id(rgba[0])
            continue
        fc = to_uint8_rgba(fc)
        if quantizer is not None:
            q_rgb = quantize_rgb(fc[:, :3])
            fc = np.concatenate([q_rgb, fc[:, 3:4]], axis=1)
        # Iterate unique colors in this mesh to limit get_material_id calls
        # but don't materialize huge sets; unique per mesh is fine.
        uniq = np.unique(fc, axis=0)
        for rgba in uniq:
            get_material_id(rgba)

    # Write MTL
    with open(mtl_path, "w") as mtl:
        for i, (r, g, b, a) in enumerate(ordered_colors):
            mtl.write(f"newmtl material_{i}\n")
            # Match viewport look: diffuse only, no specular. Many viewers assume sRGB.
            kd_r, kd_g, kd_b = r/255.0, g/255.0, b/255.0
            mtl.write(f"Kd {kd_r:.6f} {kd_g:.6f} {kd_b:.6f}\n")
            # Ambient same as diffuse to avoid darkening in some viewers
            mtl.write(f"Ka {kd_r:.6f} {kd_g:.6f} {kd_b:.6f}\n")
            # No specular highlight
            mtl.write("Ks 0.000000 0.000000 0.000000\n")
            # Disable lighting model with specular; keep simple shading
            mtl.write("illum 1\n")
            # Alpha
            mtl.write(f"d {a/255.0:.6f}\n\n")

    # Stream OBJ
    with open(obj_path, "w") as obj:
        obj.write(f"mtllib {os.path.basename(mtl_path)}\n")

        v_offset = 0  # running vertex index offset
        # Reusable cache so we don't keep writing 'usemtl' for the same block unnecessarily
        current_material = None

        for class_id, m in meshes.items():
            verts = np.asarray(m.vertices, dtype=np.float64)
            faces = np.asarray(m.faces, dtype=np.int64)
            if verts.size == 0 or faces.size == 0:
                continue

            # Write vertices
            # (We do a single pass; writing text is the bottleneck, but memory-safe.)
            for v in verts:
                obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Prepare face colors (face-level)
            fc = getattr(m.visual, "face_colors", None)
            if fc is None or len(fc) != len(faces):
                # default grey if missing or mismatched
                fc = np.tile(np.array([200,200,200,255], dtype=np.uint8), (len(faces), 1))
            else:
                fc = to_uint8_rgba(fc)

            if quantizer is not None:
                q_rgb = quantize_rgb(fc[:, :3])
                fc = np.concatenate([q_rgb, fc[:, 3:4]], axis=1)

            # Group faces by material id and stream in order
            # Build material id per face quickly
            # Convert face colors to material ids
            # (Avoid Python loops over faces more than once)
            # Map unique colors in this mesh to material ids first:
            uniq_colors, inv_idx = np.unique(fc, axis=0, return_inverse=True)
            color_to_mid_local = {tuple(c.tolist()): get_material_id(c) for c in uniq_colors}
            mids = np.fromiter(
                (color_to_mid_local[tuple(c.tolist())] for c in uniq_colors[inv_idx]),
                dtype=np.int64,
                count=len(inv_idx)
            )

            # Write faces grouped by material, but preserve simple ordering
            # Cheap approach: emit runs; switching material only when necessary
            current_material = None
            for i_face, face in enumerate(faces):
                mid = int(mids[i_face])
                if current_material != mid:
                    obj.write(f"usemtl material_{mid}\n")
                    current_material = mid
                a, b, c = face + 1 + v_offset  # OBJ is 1-based
                obj.write(f"f {a} {b} {c}\n")

            v_offset += len(verts)

    return obj_path, mtl_path