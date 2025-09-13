"""
Module for exporting voxel data to OBJ format.

This module provides functionality for converting voxel arrays and grid data to OBJ files,
including color mapping, material generation, and mesh optimization.

Key Features:
- Exports voxel data to industry-standard OBJ format with MTL materials
- Supports color mapping for visualization
- Performs greedy meshing for optimized face generation
- Handles proper face orientation and winding order
- Supports both regular voxel grids and terrain/elevation data
- Generates complete OBJ files with materials and textures

Main Functions:
- convert_colormap_indices: Converts arbitrary color indices to sequential ones
- create_face_vertices: Creates properly oriented face vertices
- mesh_faces: Performs greedy meshing on voxel layers
- export_obj: Main function to export voxel data to OBJ
- grid_to_obj: Converts 2D grid data to OBJ with elevation

Dependencies:
- numpy: For array operations
- matplotlib: For colormap handling
- trimesh: For mesh operations
"""

import numpy as np
import os
from numba import njit, prange
import matplotlib.pyplot as plt
import trimesh
import numpy as np
from ..utils.visualization import get_voxel_color_map

def convert_colormap_indices(original_map):
    """
    Convert a color map with arbitrary indices to sequential indices starting from 0.
    
    This function takes a color map with arbitrary integer keys and creates a new map
    with sequential indices starting from 0, maintaining the original color values.
    This is useful for ensuring consistent material indexing in OBJ files.
    
    Args:
        original_map (dict): Dictionary with integer keys and RGB color value lists.
            Each value should be a list of 3 integers (0-255) representing RGB colors.
        
    Returns:
        dict: New color map with sequential indices starting from 0.
            The values maintain their original RGB color assignments.
            
    Example:
        >>> original = {5: [255, 0, 0], 10: [0, 255, 0], 15: [0, 0, 255]}
        >>> new_map = convert_colormap_indices(original)
        >>> print(new_map)
        {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}
    """
    # Sort the original keys to maintain consistent ordering
    keys = sorted(original_map.keys())
    new_map = {}
    
    # Create new map with sequential indices
    for new_idx, old_idx in enumerate(keys):
        new_map[new_idx] = original_map[old_idx]
    
    # Print the new colormap for debugging/reference
    print("new_colormap = {")
    for key, value in new_map.items():
        original_key = keys[key]
        original_line = str(original_map[original_key])
        comment = ""
        if "#" in original_line:
            comment = "#" + original_line.split("#")[1].strip()
        print(f"    {key}: {value},  {comment}")
    print("}")
    
    return new_map

def create_face_vertices(coords, positive_direction, axis):
    """
    Helper function to create properly oriented face vertices for OBJ export.
    
    This function handles the creation of face vertices with correct winding order
    based on the face direction and axis. It accounts for OpenGL coordinate system
    conventions and ensures proper face orientation for rendering.
    
    Args:
        coords (list): List of 4 vertex coordinates defining the face corners.
            Each coordinate should be a tuple of (x, y, z) values.
        positive_direction (bool): Whether face points in positive axis direction.
            True = face normal points in positive direction along the axis
            False = face normal points in negative direction along the axis
        axis (str): Axis the face is perpendicular to ('x', 'y', or 'z').
            This determines how vertices are ordered for proper face orientation.
        
    Returns:
        list: Ordered vertex coordinates for the face, arranged to create proper
            face orientation and winding order for rendering.
            
    Notes:
        - Y-axis faces need special handling due to OpenGL coordinate system
        - Winding order determines which side of the face is visible
        - Consistent winding order is maintained for X and Z faces
    """
    # Y-axis faces need special handling due to OpenGL coordinate system
    if axis == 'y':
        if positive_direction:  # +Y face
            return [coords[3], coords[2], coords[1], coords[0]]  # Reverse order for +Y
        else:  # -Y face
            return [coords[0], coords[1], coords[2], coords[3]]  # Standard order for -Y
    else:
        # For X and Z faces, use consistent winding order
        if positive_direction:
            return [coords[0], coords[3], coords[2], coords[1]]
        else:
            return [coords[0], coords[1], coords[2], coords[3]]

def mesh_faces(mask, layer_index, axis, positive_direction, normal_idx, voxel_size_m, 
              vertex_dict, vertex_list, faces_per_material, voxel_value_to_material):
    """
    Performs greedy meshing on a 2D mask layer and adds optimized faces to the mesh.
    
    This function implements a greedy meshing algorithm to combine adjacent voxels
    into larger faces, reducing the total number of faces in the final mesh while
    maintaining visual accuracy. It processes each layer of voxels and generates
    optimized faces with proper materials and orientations.
    
    Args:
        mask (ndarray): 2D boolean array indicating voxel presence.
            Non-zero values indicate voxel presence, zero indicates empty space.
        layer_index (int): Index of current layer being processed.
            Used to position faces in 3D space.
        axis (str): Axis perpendicular to faces being generated ('x', 'y', or 'z').
            Determines how coordinates are generated for the faces.
        positive_direction (bool): Whether faces point in positive axis direction.
            Affects face normal orientation.
        normal_idx (int): Index of normal vector to use for faces.
            References pre-defined normal vectors in the OBJ file.
        voxel_size_m (float): Size of each voxel in meters.
            Used to scale coordinates to real-world units.
        vertex_dict (dict): Dictionary mapping vertex coordinates to indices.
            Used to avoid duplicate vertices in the mesh.
        vertex_list (list): List of unique vertex coordinates.
            Stores all vertices used in the mesh.
        faces_per_material (dict): Dictionary collecting faces by material.
            Keys are material names, values are lists of face definitions.
        voxel_value_to_material (dict): Mapping from voxel values to material names.
            Used to assign materials to faces based on voxel values.
            
    Notes:
        - Uses greedy meshing to combine adjacent same-value voxels
        - Handles coordinate system conversion for proper orientation
        - Maintains consistent face winding order for rendering
        - Optimizes mesh by reusing vertices and combining faces
        - Supports different coordinate systems for each axis
    """

    voxel_size = voxel_size_m

    # Create copy to avoid modifying original mask
    mask = mask.copy()
    h, w = mask.shape
    
    # Track which voxels have been processed
    visited = np.zeros_like(mask, dtype=bool)

    # Iterate through each position in the mask
    for u in range(h):
        v = 0
        while v < w:
            # Skip if already visited or empty voxel
            if visited[u, v] or mask[u, v] == 0:
                v += 1
                continue

            voxel_value = mask[u, v]
            material_name = voxel_value_to_material[voxel_value]

            # Greedy meshing: Find maximum width of consecutive same-value voxels
            width = 1
            while v + width < w and mask[u, v + width] == voxel_value and not visited[u, v + width]:
                width += 1

            # Find maximum height of same-value voxels
            height = 1
            done = False
            while u + height < h and not done:
                for k in range(width):
                    if mask[u + height, v + k] != voxel_value or visited[u + height, v + k]:
                        done = True
                        break
                if not done:
                    height += 1

            # Mark processed voxels as visited
            visited[u:u + height, v:v + width] = True

            # Generate vertex coordinates based on axis orientation
            if axis == 'x':
                i = float(layer_index) * voxel_size
                y0 = float(u) * voxel_size
                y1 = float(u + height) * voxel_size
                z0 = float(v) * voxel_size
                z1 = float(v + width) * voxel_size
                coords = [
                    (i, y0, z0),
                    (i, y1, z0),
                    (i, y1, z1),
                    (i, y0, z1),
                ]
            elif axis == 'y':
                i = float(layer_index) * voxel_size
                x0 = float(u) * voxel_size
                x1 = float(u + height) * voxel_size
                z0 = float(v) * voxel_size
                z1 = float(v + width) * voxel_size
                coords = [
                    (x0, i, z0),
                    (x1, i, z0),
                    (x1, i, z1),
                    (x0, i, z1),
                ]
            elif axis == 'z':
                i = float(layer_index) * voxel_size
                x0 = float(u) * voxel_size
                x1 = float(u + height) * voxel_size
                y0 = float(v) * voxel_size
                y1 = float(v + width) * voxel_size
                coords = [
                    (x0, y0, i),
                    (x1, y0, i),
                    (x1, y1, i),
                    (x0, y1, i),
                ]
            else:
                continue

            # Convert to right-handed coordinate system
            coords = [(c[2], c[1], c[0]) for c in coords]
            face_vertices = create_face_vertices(coords, positive_direction, axis)

            # Convert vertices to indices, adding new vertices as needed
            indices = []
            for coord in face_vertices:
                if coord not in vertex_dict:
                    vertex_list.append(coord)
                    vertex_dict[coord] = len(vertex_list)
                indices.append(vertex_dict[coord])

            # Create triangulated faces with proper winding order
            if axis == 'y':
                faces = [
                    {'vertices': [indices[2], indices[1], indices[0]], 'normal_idx': normal_idx},
                    {'vertices': [indices[3], indices[2], indices[0]], 'normal_idx': normal_idx}
                ]
            else:
                faces = [
                    {'vertices': [indices[0], indices[1], indices[2]], 'normal_idx': normal_idx},
                    {'vertices': [indices[0], indices[2], indices[3]], 'normal_idx': normal_idx}
                ]

            # Store faces by material
            if material_name not in faces_per_material:
                faces_per_material[material_name] = []
            faces_per_material[material_name].extend(faces)

            v += width

def export_obj(array, output_dir, file_name, voxel_size, voxel_color_map=None):
    """
    Export a voxel array to OBJ format with materials and proper face orientations.
    
    This function converts a 3D voxel array into a complete OBJ file with materials,
    performing mesh optimization and ensuring proper face orientations. It generates
    both OBJ and MTL files with all necessary components for rendering.
    
    Args:
        array (ndarray): 3D numpy array containing voxel values.
            Non-zero values indicate voxel presence and material type.
        output_dir (str): Directory to save the OBJ and MTL files.
            Will be created if it doesn't exist.
        file_name (str): Base name for the output files.
            Will be used for both .obj and .mtl files.
        voxel_size (float): Size of each voxel in meters.
            Used to scale the model to real-world units.
        voxel_color_map (dict, optional): Dictionary mapping voxel values to RGB colors.
            If None, uses default color map. Colors should be RGB lists (0-255).
            
    Notes:
        - Generates optimized mesh using greedy meshing
        - Creates complete OBJ file with vertices, normals, and faces
        - Generates MTL file with material definitions
        - Handles proper face orientation and winding order
        - Supports color mapping for visualization
        - Uses consistent coordinate system throughout
        
    File Format Details:
        OBJ file contains:
        - Vertex coordinates (v)
        - Normal vectors (vn)
        - Material references (usemtl)
        - Face definitions (f)
        
        MTL file contains:
        - Material names and colors
        - Ambient, diffuse, and specular properties
        - Transparency settings
        - Illumination model definitions
    """
    if voxel_color_map is None:
        voxel_color_map = get_voxel_color_map()

    # Extract unique voxel values (excluding zero)
    unique_voxel_values = np.unique(array)
    unique_voxel_values = unique_voxel_values[unique_voxel_values != 0]

    # Map voxel values to material names
    voxel_value_to_material = {val: f'material_{val}' for val in unique_voxel_values}

    # Define normal vectors for each face direction
    normals = [
        (1.0, 0.0, 0.0),   # 1: +X Right face
        (-1.0, 0.0, 0.0),  # 2: -X Left face
        (0.0, 1.0, 0.0),   # 3: +Y Top face
        (0.0, -1.0, 0.0),  # 4: -Y Bottom face
        (0.0, 0.0, 1.0),   # 5: +Z Front face
        (0.0, 0.0, -1.0),  # 6: -Z Back face
    ]

    # Map direction names to normal indices
    normal_indices = {
        'nx': 2,
        'px': 1,
        'ny': 4,
        'py': 3,
        'nz': 6,
        'pz': 5,
    }

    # Initialize data structures
    vertex_list = []
    vertex_dict = {}
    faces_per_material = {}

    # Transpose array for correct orientation in output
    array = array.transpose(2, 1, 0)  # Now array[x, y, z]
    size_x, size_y, size_z = array.shape

    # Define processing directions and their normals
    directions = [
        ('nx', (-1, 0, 0)),
        ('px', (1, 0, 0)),
        ('ny', (0, -1, 0)),
        ('py', (0, 1, 0)),
        ('nz', (0, 0, -1)),
        ('pz', (0, 0, 1)),
    ]

    # Process each face direction
    for direction, normal in directions:
        normal_idx = normal_indices[direction]
        
        # Process X-axis aligned faces
        if direction in ('nx', 'px'):
            for x in range(size_x):
                voxel_slice = array[x, :, :]
                if direction == 'nx':
                    neighbor_slice = array[x - 1, :, :] if x > 0 else np.zeros_like(voxel_slice)
                    layer = x
                else:
                    neighbor_slice = array[x + 1, :, :] if x + 1 < size_x else np.zeros_like(voxel_slice)
                    layer = x + 1

                # Create mask for faces that need to be generated
                mask = np.where((voxel_slice != neighbor_slice) & (voxel_slice != 0), voxel_slice, 0)
                mesh_faces(mask, layer, 'x', direction == 'px', normal_idx, voxel_size,
                         vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

        # Process Y-axis aligned faces
        elif direction in ('ny', 'py'):
            for y in range(size_y):
                voxel_slice = array[:, y, :]
                if direction == 'ny':
                    neighbor_slice = array[:, y - 1, :] if y > 0 else np.zeros_like(voxel_slice)
                    layer = y
                else:
                    neighbor_slice = array[:, y + 1, :] if y + 1 < size_y else np.zeros_like(voxel_slice)
                    layer = y + 1

                mask = np.where((voxel_slice != neighbor_slice) & (voxel_slice != 0), voxel_slice, 0)
                mesh_faces(mask, layer, 'y', direction == 'py', normal_idx, voxel_size,
                         vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

        # Process Z-axis aligned faces
        elif direction in ('nz', 'pz'):
            for z in range(size_z):
                voxel_slice = array[:, :, z]
                if direction == 'nz':
                    neighbor_slice = array[:, :, z - 1] if z > 0 else np.zeros_like(voxel_slice)
                    layer = z
                else:
                    neighbor_slice = array[:, :, z + 1] if z + 1 < size_z else np.zeros_like(voxel_slice)
                    layer = z + 1

                mask = np.where((voxel_slice != neighbor_slice) & (voxel_slice != 0), voxel_slice, 0)
                mesh_faces(mask, layer, 'z', direction == 'pz', normal_idx, voxel_size,
                         vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
    mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

    # Write OBJ file
    with open(obj_file_path, 'w') as f:
        f.write('# Generated OBJ file\n\n')
        f.write('# group\no \n\n')
        f.write(f'# material\nmtllib {file_name}.mtl\n\n')
        
        # Write normal vectors
        f.write('# normals\n')
        for nx, ny, nz in normals:
            f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        f.write('\n')
        
        # Write vertex coordinates
        f.write('# verts\n')
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')
        
        # Write faces grouped by material
        f.write('# faces\n')
        for material_name, faces in faces_per_material.items():
            f.write(f'usemtl {material_name}\n')
            for face in faces:
                v_indices = [str(vi) for vi in face['vertices']]
                normal_idx = face['normal_idx']
                face_str = ' '.join([f'{vi}//{normal_idx}' for vi in face['vertices']])
                f.write(f'f {face_str}\n')
            f.write('\n')

    # Write MTL file with material definitions
    with open(mtl_file_path, 'w') as f:
        f.write('# Material file\n\n')
        for voxel_value in unique_voxel_values:
            material_name = voxel_value_to_material[voxel_value]
            color = voxel_color_map.get(voxel_value, [0, 0, 0])
            r, g, b = [c / 255.0 for c in color]
            f.write(f'newmtl {material_name}\n')
            f.write(f'Ka {r:.6f} {g:.6f} {b:.6f}\n')  # Ambient color
            f.write(f'Kd {r:.6f} {g:.6f} {b:.6f}\n')  # Diffuse color
            f.write(f'Ke {r:.6f} {g:.6f} {b:.6f}\n')  # Emissive color
            f.write('Ks 0.500000 0.500000 0.500000\n')  # Specular reflection
            f.write('Ns 50.000000\n')                   # Specular exponent
            f.write('illum 2\n\n')                      # Illumination model

    print(f'OBJ and MTL files have been generated in {output_dir} with the base name "{file_name}".')

def grid_to_obj(value_array_ori, dem_array_ori, output_dir, file_name, cell_size, offset,
                 colormap_name='viridis', num_colors=256, alpha=1.0, vmin=None, vmax=None):
    """
    Converts a 2D array of values and a corresponding DEM array to an OBJ file
    with specified colormap, transparency, and value range.
    
    This function creates a 3D visualization of 2D grid data by using elevation
    data and color mapping. It's particularly useful for visualizing terrain data,
    analysis results, or any 2D data that should be displayed with elevation.
    
    Args:
        value_array_ori (ndarray): 2D array of values to visualize.
            These values will be mapped to colors using the specified colormap.
        dem_array_ori (ndarray): 2D array of DEM values corresponding to value_array.
            Provides elevation data for the 3D visualization.
        output_dir (str): Directory to save the OBJ and MTL files.
            Will be created if it doesn't exist.
        file_name (str): Base name for the output files.
            Used for both .obj and .mtl files.
        cell_size (float): Size of each cell in the grid (e.g., in meters).
            Used to scale the model to real-world units.
        offset (float): Elevation offset added after quantization.
            Useful for adjusting the base height of the model.
        colormap_name (str, optional): Name of the Matplotlib colormap to use.
            Defaults to 'viridis'. Must be a valid Matplotlib colormap name.
        num_colors (int, optional): Number of discrete colors to use from the colormap.
            Defaults to 256. Higher values give smoother color transitions.
        alpha (float, optional): Transparency value between 0.0 (transparent) and 1.0 (opaque).
            Defaults to 1.0 (fully opaque).
        vmin (float, optional): Minimum value for colormap normalization.
            If None, uses data minimum. Used to control color mapping range.
        vmax (float, optional): Maximum value for colormap normalization.
            If None, uses data maximum. Used to control color mapping range.
            
    Notes:
        - Automatically handles NaN values in input arrays
        - Creates triangulated mesh for proper rendering
        - Supports transparency and color mapping
        - Generates complete OBJ and MTL files
        - Maintains consistent coordinate system
        - Optimizes mesh generation for large grids
        
    Raises:
        ValueError: If vmin equals vmax or if colormap_name is invalid
    """
    # Validate input arrays
    if value_array_ori.shape != dem_array_ori.shape:
        raise ValueError("The value array and DEM array must have the same shape.")
    
    # Get the dimensions
    rows, cols = value_array_ori.shape

    # Flip arrays vertically and normalize DEM values
    value_array = np.flipud(value_array_ori.copy())
    dem_array = np.flipud(dem_array_ori.copy()) - np.min(dem_array_ori)

    # Get valid indices (non-NaN)
    valid_indices = np.argwhere(~np.isnan(value_array))

    # Set vmin and vmax if not provided
    if vmin is None:
        vmin = np.nanmin(value_array)
    if vmax is None:
        vmax = np.nanmax(value_array)
    
    # Handle case where vmin equals vmax
    if vmin == vmax:
        raise ValueError("vmin and vmax cannot be the same value.")
    
    # Normalize values to [0, 1] based on vmin and vmax
    normalized_values = (value_array - vmin) / (vmax - vmin)
    # Clip normalized values to [0, 1]
    normalized_values = np.clip(normalized_values, 0.0, 1.0)
    
    # Prepare the colormap
    if colormap_name not in plt.colormaps():
        raise ValueError(f"Colormap '{colormap_name}' is not recognized. Please choose a valid Matplotlib colormap.")
    colormap = plt.get_cmap(colormap_name, num_colors)  # Discrete colors

    # Create a mapping from quantized colors to material names
    color_to_material = {}
    materials = []
    material_index = 1  # Start indexing materials from 1

    # Initialize vertex tracking
    vertex_list = []
    vertex_dict = {}  # To avoid duplicate vertices
    vertex_index = 1  # OBJ indices start at 1

    faces_per_material = {}

    # Process each valid cell in the grid
    for idx in valid_indices:
        i, j = idx  # i is the row index, j is the column index
        value = value_array[i, j]
        normalized_value = normalized_values[i, j]

        # Get the color from the colormap
        rgba = colormap(normalized_value)
        rgb = rgba[:3]  # Ignore alpha channel
        r, g, b = [int(c * 255) for c in rgb]

        # Create unique material name for this color
        color_key = (r, g, b)
        material_name = f'material_{r}_{g}_{b}'

        # Add new material if not seen before
        if material_name not in color_to_material:
            color_to_material[material_name] = {
                'r': r / 255.0,
                'g': g / 255.0,
                'b': b / 255.0,
                'alpha': alpha
            }
            materials.append(material_name)

        # Calculate cell vertices
        x0 = i * cell_size
        x1 = (i + 1) * cell_size
        y0 = j * cell_size
        y1 = (j + 1) * cell_size

        # Calculate elevation with quantization and offset
        z = cell_size * int(dem_array[i, j] / cell_size + 1.5) + offset

        # Define quad vertices
        vertices = [
            (x0, y0, z),
            (x1, y0, z),
            (x1, y1, z),
            (x0, y1, z),
        ]

        # Convert vertices to indices
        indices = []
        for v in vertices:
            if v not in vertex_dict:
                vertex_list.append(v)
                vertex_dict[v] = vertex_index
                vertex_index += 1
            indices.append(vertex_dict[v])

        # Create triangulated faces
        faces = [
            {'vertices': [indices[0], indices[1], indices[2]]},
            {'vertices': [indices[0], indices[2], indices[3]]},
        ]

        # Store faces by material
        if material_name not in faces_per_material:
            faces_per_material[material_name] = []
        faces_per_material[material_name].extend(faces)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
    mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

    # Write OBJ file
    with open(obj_file_path, 'w') as f:
        f.write('# Generated OBJ file\n\n')
        f.write(f'mtllib {file_name}.mtl\n\n')
        # Write vertices
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')
        # Write faces grouped by material
        for material_name in materials:
            f.write(f'usemtl {material_name}\n')
            faces = faces_per_material[material_name]
            for face in faces:
                v_indices = face['vertices']
                face_str = ' '.join([f'{vi}' for vi in v_indices])
                f.write(f'f {face_str}\n')
            f.write('\n')

    # Write MTL file with material properties
    with open(mtl_file_path, 'w') as f:
        for material_name in materials:
            color = color_to_material[material_name]
            r, g, b = color['r'], color['g'], color['b']
            a = color['alpha']
            f.write(f'newmtl {material_name}\n')
            f.write(f'Ka {r:.6f} {g:.6f} {b:.6f}\n')  # Ambient color
            f.write(f'Kd {r:.6f} {g:.6f} {b:.6f}\n')  # Diffuse color
            f.write(f'Ks 0.000000 0.000000 0.000000\n')  # Specular reflection
            f.write('Ns 10.000000\n')                   # Specular exponent
            f.write('illum 1\n')                        # Illumination model
            f.write(f'd {a:.6f}\n')                     # Transparency (alpha)
            f.write('\n')

    print(f'OBJ and MTL files have been generated in {output_dir} with the base name "{file_name}".')


def export_netcdf_to_obj(
    voxcity_nc,
    scalar_nc,
    lonlat_txt,
    output_dir,
    vox_base_filename="voxcity_objects",
    tm_base_filename="tm_isosurfaces",
    scalar_var="tm",
    scalar_building_value=-999.99,
    scalar_building_tol=1e-4,
    stride_vox=(1, 1, 1),
    stride_scalar=(1, 1, 1),
    contour_levels=24,
    cmap_name="magma",
    vmin=None,
    vmax=None,
    opacity_points=None,
    max_opacity=0.10,
    classes_to_show=None,
    voxel_color_scheme="default",
    max_faces_warn=1_000_000,
):
    """
    Export two OBJ/MTL files using the same local meter frame:
    - VoxCity voxels: opaque, per-class color, fixed face winding and normals
    - Scalar iso-surfaces: colormap colors with variable transparency

    The two outputs share the same XY origin and axes (X east, Y north, Z up),
    anchored at the minimum lon/lat of the VoxCity bounding rectangle.

    Args:
        voxcity_nc (str): Path to VoxCity NetCDF (must include variable 'voxels' and coords 'x','y','z').
        scalar_nc (str): Path to scalar NetCDF containing variable specified by scalar_var.
        lonlat_txt (str): Text file with columns: i j lon lat (1-based indices) describing the scalar grid georef.
        output_dir (str): Directory to write results.
        vox_base_filename (str): Base filename for VoxCity OBJ/MTL.
        tm_base_filename (str): Base filename for scalar iso-surfaces OBJ/MTL.
        scalar_var (str): Name of scalar variable in scalar_nc.
        scalar_building_value (float): Value used in scalar field to mark buildings (to be masked).
        scalar_building_tol (float): Tolerance for building masking (isclose).
        stride_vox (tuple[int,int,int]): Downsampling strides for VoxCity (z,y,x) in voxels.
        stride_scalar (tuple[int,int,int]): Downsampling strides for scalar (k,j,i).
        contour_levels (int): Number of iso-surface levels between vmin and vmax.
        cmap_name (str): Matplotlib colormap name for iso-surfaces.
        vmin (float|None): Minimum scalar value for color mapping and iso range. If None, inferred.
        vmax (float|None): Maximum scalar value for color mapping and iso range. If None, inferred.
        opacity_points (list[tuple[float,float]]|None): Transfer function control points (value, alpha in [0..1]).
        max_opacity (float): Global max opacity multiplier for iso-surfaces (0..1).
        classes_to_show (set[int]|None): Optional subset of voxel classes to export; None -> all present (except 0).
        voxel_color_scheme (str): Color scheme name passed to get_voxel_color_map.
        max_faces_warn (int): Warn if a single class exceeds this many faces.

    Returns:
        dict: Paths of written files: keys 'vox_obj','vox_mtl','tm_obj','tm_mtl' (values may be None).
    """
    import json
    import numpy as np
    import os
    import xarray as xr
    import trimesh

    try:
        from skimage import measure as skim
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-image is required for iso-surface generation. Install 'scikit-image'."
        ) from e

    from matplotlib import cm

    if opacity_points is None:
        opacity_points = [(-0.2, 0.00), (2.0, 1.00)]

    def find_dims(ds):
        lvl = ["k", "level", "lev", "z", "height", "alt", "plev"]
        yy = ["j", "y", "south_north", "lat", "latitude"]
        xx = ["i", "x", "west_east", "lon", "longitude"]
        tt = ["time", "Times"]

        def pick(cands):
            for c in cands:
                if c in ds.dims:
                    return c
            return None

        t = pick(tt)
        k = pick(lvl)
        j = pick(yy)
        i = pick(xx)
        if (k is None or j is None or i is None) and len(ds.dims) >= 3:
            dims = list(ds.dims)
            k = k or dims[0]
            j = j or dims[-2]
            i = i or dims[-1]
        return t, k, j, i

    def squeeze_to_kji(da, tname, kname, jname, iname, time_index=0):
        if tname and tname in da.dims:
            da = da.isel({tname: time_index})
        for d in list(da.dims):
            if d not in (kname, jname, iname):
                da = da.isel({d: 0})
        return da.transpose(*(d for d in (kname, jname, iname) if d in da.dims))

    def downsample3(a, sk, sj, si):
        return a[:: max(1, sk), :: max(1, sj), :: max(1, si)]

    def clip_minmax(arr, frac):
        v = np.asarray(arr)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 1.0
        if frac <= 0:
            return float(np.nanmin(v)), float(np.nanmax(v))
        vmin_ = float(np.nanpercentile(v, 100 * frac))
        vmax_ = float(np.nanpercentile(v, 100 * (1 - frac)))
        if vmin_ >= vmax_:
            vmin_, vmax_ = float(np.nanmin(v)), float(np.nanmax(v))
        return vmin_, vmax_

    def meters_per_degree(lat_rad):
        m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
        m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
        return m_per_deg_lat, m_per_deg_lon

    def opacity_at(v, points):
        if not points:
            return 0.0 if np.isscalar(v) else np.zeros_like(v)
        pts = sorted((float(x), float(a)) for x, a in points)
        xs = np.array([p[0] for p in pts], dtype=float)
        as_ = np.array([p[1] for p in pts], dtype=float)
        v_arr = np.asarray(v, dtype=float)
        out = np.empty_like(v_arr, dtype=float)
        out[v_arr <= xs[0]] = as_[0]
        out[v_arr >= xs[-1]] = as_[-1]
        idx = np.searchsorted(xs, v_arr, side="right") - 1
        idx = np.clip(idx, 0, len(xs) - 2)
        x0, x1 = xs[idx], xs[idx + 1]
        a0, a1 = as_[idx], as_[idx + 1]
        t = np.where(x1 > x0, (v_arr - x0) / (x1 - x0), 0.0)
        mid = (v_arr > xs[0]) & (v_arr < xs[-1])
        out[mid] = a0[mid] + t[mid] * (a1[mid] - a0[mid])
        return out.item() if np.isscalar(v) else out

    def _exposed_face_masks(occ):
        K, J, I = occ.shape
        p = np.pad(occ, ((0, 0), (0, 0), (0, 1)), constant_values=False)
        posx = occ & (~p[..., 1:])
        p = np.pad(occ, ((0, 0), (0, 0), (1, 0)), constant_values=False)
        negx = occ & (~p[..., :-1])
        p = np.pad(occ, ((0, 0), (0, 1), (0, 0)), constant_values=False)
        posy = occ & (~p[:, 1:, :])
        p = np.pad(occ, ((0, 0), (1, 0), (0, 0)), constant_values=False)
        negy = occ & (~p[:, :-1, :])
        p = np.pad(occ, ((0, 1), (0, 0), (0, 0)), constant_values=False)
        posz = occ & (~p[1:, :, :])
        p = np.pad(occ, ((1, 0), (0, 0), (0, 0)), constant_values=False)
        negz = occ & (~p[:-1, :, :])
        return posx, negx, posy, negy, posz, negz

    def _emit_faces_trimesh(k, j, i, plane, X, Y, Z, start_idx):
        N = k.size
        if N == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), start_idx

        dx = (X[1] - X[0]) if len(X) > 1 else 1.0
        dy = (Y[1] - Y[0]) if len(Y) > 1 else 1.0
        dz = (Z[1] - Z[0]) if len(Z) > 1 else 1.0

        x = X[i].astype(np.float64)
        y = Y[j].astype(np.float64)
        z = Z[k].astype(np.float64)
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

        if plane == "+x":
            P = np.column_stack([x + hx, y - hy, z - hz])
            Q = np.column_stack([x + hx, y + hy, z - hz])
            R = np.column_stack([x + hx, y + hy, z + hz])
            S = np.column_stack([x + hx, y - hy, z + hz])
            order = "default"
        elif plane == "-x":
            P = np.column_stack([x - hx, y - hy, z + hz])
            Q = np.column_stack([x - hx, y + hy, z + hz])
            R = np.column_stack([x - hx, y + hy, z - hz])
            S = np.column_stack([x - hx, y - hy, z - hz])
            order = "default"
        elif plane == "+y":
            P = np.column_stack([x - hx, y + hy, z - hz])
            Q = np.column_stack([x + hx, y + hy, z - hz])
            R = np.column_stack([x + hx, y + hy, z + hz])
            S = np.column_stack([x - hx, y + hy, z + hz])
            order = "flip"  # enforce outward normals
        elif plane == "-y":
            P = np.column_stack([x - hx, y - hy, z + hz])
            Q = np.column_stack([x + hx, y - hy, z + hz])
            R = np.column_stack([x + hx, y - hy, z - hz])
            S = np.column_stack([x - hx, y - hy, z - hz])
            order = "flip"
        elif plane == "+z":
            P = np.column_stack([x - hx, y - hy, z + hz])
            Q = np.column_stack([x + hx, y - hy, z + hz])
            R = np.column_stack([x + hx, y + hy, z + hz])
            S = np.column_stack([x - hx, y + hy, z + hz])
            order = "default"
        else:  # "-z"
            P = np.column_stack([x - hx, y + hy, z - hz])
            Q = np.column_stack([x + hx, y + hy, z - hz])
            R = np.column_stack([x + hx, y - hy, z - hz])
            S = np.column_stack([x - hx, y - hy, z - hz])
            order = "default"

        verts = np.vstack([P, Q, R, S])
        a = np.arange(N, dtype=np.int64) + start_idx
        b = a + N
        c = a + 2 * N
        d = a + 3 * N

        if order == "default":
            tris = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        else:
            tris = np.vstack([np.column_stack([a, c, b]), np.column_stack([a, d, c])])

        return verts, tris, start_idx + 4 * N

    def make_voxel_mesh_uniform_color(occ_mask, X, Y, Z, rgb, name="class"):
        posx, negx, posy, negy, posz, negz = _exposed_face_masks(occ_mask.astype(bool))
        total_faces = int(posx.sum() + negx.sum() + posy.sum() + negy.sum() + posz.sum() + negz.sum())
        if total_faces == 0:
            return None, 0
        if total_faces > max_faces_warn:
            print(f"  Warning: {name} faces={total_faces:,} (> {max_faces_warn:,}). Consider increasing stride.")

        verts_all, tris_all, start_idx = [], [], 0
        for plane, mask in (("+x", posx), ("-x", negx), ("+y", posy), ("-y", negy), ("+z", posz), ("-z", negz)):
            idx = np.argwhere(mask)
            if idx.size == 0:
                continue
            k, j, i = idx[:, 0], idx[:, 1], idx[:, 2]
            Vp, Tp, start_idx = _emit_faces_trimesh(k, j, i, plane, X, Y, Z, start_idx)
            verts_all.append(Vp)
            tris_all.append(Tp)

        V = np.vstack(verts_all)
        F = np.vstack(tris_all)
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        rgba = np.array([int(rgb[0]), int(rgb[1]), int(rgb[2]), 255], dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(F), 1))
        return mesh, len(F)

    def build_tm_isosurfaces_regular_grid(A_scalar, vmin, vmax, levels, dx, dy, dz, origin_xyz, cmap_name, opacity_points, max_opacity):
        cmap = cm.get_cmap(cmap_name)
        meshes = []
        if levels <= 0:
            return meshes
        iso_vals = np.linspace(vmin, vmax, int(levels))
        for iso in iso_vals:
            a_base = float(opacity_at(iso, opacity_points or []))
            a_base = min(max(a_base, 0.0), 1.0)
            alpha = a_base * max_opacity
            if alpha <= 0.0:
                continue
            try:
                verts, faces, _, _ = skim.marching_cubes(A_scalar, level=iso, spacing=(dz, dy, dx))
            except Exception:
                continue
            if len(verts) == 0 or len(faces) == 0:
                continue
            V = verts[:, [2, 1, 0]].astype(np.float64)
            V += np.array(origin_xyz, dtype=np.float64)[None, :]
            m = trimesh.Trimesh(vertices=V, faces=faces.astype(np.int64), process=False)
            t = 0.0 if vmax <= vmin else (iso - vmin) / (vmax - vmin)
            r, g, b, _ = cmap(np.clip(t, 0.0, 1.0))
            rgba = (
                int(round(255 * r)),
                int(round(255 * g)),
                int(round(255 * b)),
                int(round(255 * alpha)),
            )
            m.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(m.faces), 1))
            meshes.append((iso, m, rgba))
            print(f"Iso {iso:.4f}: faces={len(m.faces):,}, alpha={alpha:.4f}")
        return meshes

    def save_obj_with_mtl_and_normals(meshes_dict, output_path, base_filename):
        os.makedirs(output_path, exist_ok=True)
        obj_path = os.path.join(output_path, f"{base_filename}.obj")
        mtl_path = os.path.join(output_path, f"{base_filename}.mtl")

        def to_uint8_rgba(arr):
            arr = np.asarray(arr)
            if arr.dtype != np.uint8:
                if arr.dtype.kind == "f":
                    arr = np.clip(arr, 0.0, 1.0)
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.shape[1] == 3:
                arr = np.concatenate([arr, np.full((arr.shape[0], 1), 255, np.uint8)], axis=1)
            return arr

        color_to_id, ordered = {}, []

        def mid_of(rgba):
            if rgba not in color_to_id:
                color_to_id[rgba] = len(ordered)
                ordered.append(rgba)
            return color_to_id[rgba]

        for m in meshes_dict.values():
            fc = getattr(m.visual, "face_colors", None)
            if fc is None or len(fc) == 0:
                mid_of((200, 200, 200, 255))
                continue
            for rgba in np.unique(to_uint8_rgba(fc), axis=0):
                mid_of(tuple(int(x) for x in rgba.tolist()))

        with open(mtl_path, "w") as mtl:
            for i, (r, g, b, a) in enumerate(ordered):
                kd = (r / 255.0, g / 255.0, b / 255.0)
                ka = kd
                dval = a / 255.0
                tr = max(0.0, min(1.0, 1.0 - dval))
                mtl.write(f"newmtl material_{i}\n")
                mtl.write(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")
                mtl.write(f"Ka {ka[0]:.6f} {ka[1]:.6f} {ka[2]:.6f}\n")
                mtl.write("Ks 0.000000 0.000000 0.000000\n")
                mtl.write("Ns 0.000000\n")
                mtl.write("illum 1\n")
                mtl.write(f"d {dval:.6f}\n")
                mtl.write(f"Tr {tr:.6f}\n\n")

        def face_normals(V, F):
            v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            L = np.linalg.norm(n, axis=1)
            mask = L > 0
            n[mask] /= L[mask][:, None]
            if (~mask).any():
                n[~mask] = np.array([0.0, 0.0, 1.0])
            return n

        with open(obj_path, "w") as obj:
            obj.write(f"mtllib {os.path.basename(mtl_path)}\n")
            v_offset = 0
            n_offset = 0
            for name, m in meshes_dict.items():
                V = np.asarray(m.vertices, dtype=np.float64)
                F = np.asarray(m.faces, dtype=np.int64)
                if len(V) == 0 or len(F) == 0:
                    continue
                obj.write(f"o {name}\n")
                obj.write("s off\n")
                for vx, vy, vz in V:
                    obj.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

                fc = getattr(m.visual, "face_colors", None)
                if fc is None or len(fc) != len(F):
                    fc = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (len(F), 1))
                else:
                    fc = to_uint8_rgba(fc)
                uniq, inv = np.unique(fc, axis=0, return_inverse=True)
                color2mid = {tuple(int(x) for x in c.tolist()): mid_of(tuple(int(x) for x in c.tolist())) for c in uniq}

                FN = face_normals(V, F)
                for nx, ny, nz in FN:
                    obj.write(f"vn {float(nx):.6f} {float(ny):.6f} {float(nz):.6f}\n")

                current_mid = None
                for i_face, face in enumerate(F):
                    key = tuple(int(x) for x in uniq[inv[i_face]].tolist())
                    mid = color2mid[key]
                    if current_mid != mid:
                        obj.write(f"usemtl material_{mid}\n")
                        current_mid = mid
                    a, b, c = face + 1 + v_offset
                    ni = n_offset + i_face + 1
                    obj.write(f"f {a}//{ni} {b}//{ni} {c}//{ni}\n")

                v_offset += len(V)
                n_offset += len(F)

        return obj_path, mtl_path

    # Load VoxCity
    dsv = xr.open_dataset(voxcity_nc)
    if "voxels" not in dsv:
        raise KeyError("'voxels' not found in VoxCity dataset.")
    dav = dsv["voxels"]
    if tuple(dav.dims) != ("y", "x", "z") and all(d in dav.dims for d in ("y", "x", "z")):
        dav = dav.transpose("y", "x", "z")

    Yv = dsv["y"].values.astype(float)
    Xv = dsv["x"].values.astype(float)
    Zv = dsv["z"].values.astype(float)

    Av = dav.values  # (y,x,z)
    Av_kji = np.transpose(Av, (2, 0, 1))  # (K=z, J=y, I=x)
    svz, svy, svx = stride_vox
    Av_kji = downsample3(Av_kji, svz, svy, svx)
    # Y flip (north-up)
    Av_kji = Av_kji[:, ::-1, :]

    Zv_s = Zv[:: max(1, svz)].astype(float)
    Yv_s = (Yv[:: max(1, svy)] - Yv.min()).astype(float)
    Xv_s = (Xv[:: max(1, svx)] - Xv.min()).astype(float)

    # Load scalar and georeference using lon/lat table
    dss = xr.open_dataset(scalar_nc, decode_coords="all", decode_times=True)
    tname, kname, jname, iname = find_dims(dss)
    if scalar_var not in dss:
        raise KeyError(f"{scalar_var} not found in scalar dataset")

    A = squeeze_to_kji(dss[scalar_var], tname, kname, jname, iname).values  # (K,J,I)
    K0, J0, I0 = map(int, A.shape)

    ll = np.loadtxt(lonlat_txt, comments="#")
    ii = ll[:, 0].astype(int) - 1
    jj = ll[:, 1].astype(int) - 1
    lon = ll[:, 2].astype(float)
    lat = ll[:, 3].astype(float)
    I_ll = int(ii.max() + 1)
    J_ll = int(jj.max() + 1)
    lon_grid = np.full((J_ll, I_ll), np.nan, float)
    lat_grid = np.full((J_ll, I_ll), np.nan, float)
    lon_grid[jj, ii] = lon
    lat_grid[jj, ii] = lat

    Jc = min(J0, J_ll)
    Ic = min(I0, I_ll)
    if (Jc != J0) or (Ic != I0):
        print(
            f"Warning: scalar (J,I)=({J0},{I0}) vs lonlat ({J_ll},{I_ll}); using common ({Jc},{Ic})."
        )
    A = A[:, :Jc, :Ic]
    lon_grid = lon_grid[:Jc, :Ic]
    lat_grid = lat_grid[:Jc, :Ic]

    ssk, ssj, ssi = stride_scalar
    A_s = downsample3(A, ssk, ssj, ssi)
    lon_s = lon_grid[:: max(1, ssj), :: max(1, ssi)]
    lat_s = lat_grid[:: max(1, ssj), :: max(1, ssi)]
    Ks, Js, Is = A_s.shape

    rect = np.array(json.loads(dsv.attrs.get("rectangle_vertices_lonlat_json", "[]")), float)
    if rect.size == 0:
        raise RuntimeError("VoxCity attribute 'rectangle_vertices_lonlat_json' missing.")
    lon0 = float(np.min(rect[:, 0]))
    lat0 = float(np.min(rect[:, 1]))
    lat_c = float(np.mean(rect[:, 1]))
    m_per_deg_lat, m_per_deg_lon = meters_per_degree(np.deg2rad(lat_c))
    Xs_m = (lon_s - lon0) * m_per_deg_lon
    Ys_m = (lat_s - lat0) * m_per_deg_lat

    if (kname is not None) and (kname in dss.coords):
        zc = dss.coords[kname].values
        if np.issubdtype(zc.dtype, np.number) and zc.ndim == 1 and len(zc) >= Ks:
            Zk = zc.astype(float)[:: max(1, ssk)][:Ks]
        else:
            Zk = np.arange(Ks, dtype=float) * float(dsv.attrs.get("meshsize_m", 1.0))
    else:
        Zk = np.arange(Ks, dtype=float) * float(dsv.attrs.get("meshsize_m", 1.0))

    # Mask scalar buildings
    bmask_scalar = downsample3(
        np.isclose(A, scalar_building_value, atol=scalar_building_tol), ssk, ssj, ssi
    )
    A_s = A_s.astype(float)
    A_s[bmask_scalar] = np.nan

    finite_vals = A_s[np.isfinite(A_s)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite scalar values after masking.")
    if (vmin is None) or (vmax is None):
        auto_vmin, auto_vmax = clip_minmax(finite_vals, 0.0)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
    if not (vmin < vmax):
        raise ValueError("vmin must be less than vmax.")
    A_s[np.isnan(A_s)] = vmin - 1e6

    Xmin, Xmax = np.nanmin(Xs_m), np.nanmax(Xs_m)
    Ymin, Ymax = np.nanmin(Ys_m), np.nanmax(Ys_m)
    dx_s = (Xmax - Xmin) / max(1, Is - 1)
    dy_s = (Ymax - Ymin) / max(1, Js - 1)
    dz_s = (Zk[-1] - Zk[0]) / max(1, Ks - 1) if Ks > 1 else 1.0
    origin_xyz = (float(Xmin), float(Ymin), float(Zk[0]))

    vox_meshes = {}
    tm_meshes = {}

    present = set(np.unique(Av_kji))
    present.discard(0)
    if classes_to_show is not None:
        present &= set(classes_to_show)
    present = sorted(present)

    faces_total = 0
    voxel_color_map = get_voxel_color_map(color_scheme=voxel_color_scheme)
    for cls in present:
        mask = Av_kji == cls
        if not np.any(mask):
            continue
        rgb = voxel_color_map.get(int(cls), [200, 200, 200])
        m_cls, faces = make_voxel_mesh_uniform_color(mask, Xv_s, Yv_s, Zv_s, rgb=rgb, name=f"class_{int(cls)}")
        if m_cls is not None:
            vox_meshes[f"voxclass_{int(cls)}"] = m_cls
            faces_total += faces
    print(f"[VoxCity] total voxel faces: {faces_total:,}")

    iso_meshes = build_tm_isosurfaces_regular_grid(
        A_scalar=A_s,
        vmin=vmin,
        vmax=vmax,
        levels=contour_levels,
        dx=dx_s,
        dy=dy_s,
        dz=dz_s,
        origin_xyz=origin_xyz,
        cmap_name=cmap_name,
        opacity_points=opacity_points,
        max_opacity=max_opacity,
    )
    for iso, m, rgba in iso_meshes:
        tm_meshes[f"iso_{iso:.6f}"] = m

    if not vox_meshes and not tm_meshes:
        raise RuntimeError("Nothing to export.")

    os.makedirs(output_dir, exist_ok=True)
    obj_vox = mtl_vox = obj_tm = mtl_tm = None
    if vox_meshes:
        obj_vox, mtl_vox = save_obj_with_mtl_and_normals(vox_meshes, output_dir, vox_base_filename)
    if tm_meshes:
        obj_tm, mtl_tm = save_obj_with_mtl_and_normals(tm_meshes, output_dir, tm_base_filename)

    print("Export finished.")
    if obj_vox:
        print(f"VoxCity OBJ: {obj_vox}")
        print(f"VoxCity MTL: {mtl_vox}")
    if obj_tm:
        print(f"Scalar Iso OBJ: {obj_tm}")
        print(f"Scalar Iso MTL: {mtl_tm}")

    return {"vox_obj": obj_vox, "vox_mtl": mtl_vox, "tm_obj": obj_tm, "tm_mtl": mtl_tm}