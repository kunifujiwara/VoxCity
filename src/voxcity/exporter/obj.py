"""
Module for exporting voxel data to OBJ format.

This module provides functionality for converting voxel arrays and grid data to OBJ files,
including color mapping, material generation, and mesh optimization.
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
    
    Args:
        original_map (dict): Dictionary with integer keys and RGB color value lists
        
    Returns:
        dict: New color map with sequential indices starting from 0
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
    Helper function to create properly oriented face vertices.
    
    Args:
        coords (list): List of vertex coordinates
        positive_direction (bool): Whether face points in positive axis direction
        axis (str): Axis the face is perpendicular to ('x', 'y', or 'z')
        
    Returns:
        list: Ordered vertex coordinates for the face
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
    Performs greedy meshing on the given mask and adds faces to the faces_per_material dictionary.
    
    Args:
        mask (ndarray): 2D boolean array indicating voxel presence
        layer_index (int): Index of current layer being processed
        axis (str): Axis perpendicular to faces being generated ('x', 'y', or 'z')
        positive_direction (bool): Whether faces point in positive axis direction
        normal_idx (int): Index of normal vector to use for faces
        voxel_size_m (float): Size of each voxel in meters
        vertex_dict (dict): Dictionary mapping vertex coordinates to indices
        vertex_list (list): List of unique vertex coordinates
        faces_per_material (dict): Dictionary collecting faces by material
        voxel_value_to_material (dict): Mapping from voxel values to material names
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
    Export a voxel array to OBJ format with corrected face orientations.
    
    Args:
        array (ndarray): 3D numpy array containing voxel values
        output_dir (str): Directory to save the OBJ and MTL files
        file_name (str): Base name for the output files
        voxel_size (float): Size of each voxel in meters
        voxel_color_map (dict, optional): Dictionary mapping voxel values to RGB colors.
            If None, uses default color map.
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

    Args:
        value_array_ori (ndarray): 2D array of values to visualize
        dem_array_ori (ndarray): 2D array of DEM values corresponding to value_array
        output_dir (str): Directory to save the OBJ and MTL files
        file_name (str): Base name for the output files
        cell_size (float): Size of each cell in the grid (e.g., in meters)
        offset (float): Elevation offset added after quantization
        colormap_name (str, optional): Name of the Matplotlib colormap to use. Defaults to 'viridis'
        num_colors (int, optional): Number of discrete colors to use from the colormap. Defaults to 256
        alpha (float, optional): Transparency value between 0.0 (transparent) and 1.0 (opaque). Defaults to 1.0
        vmin (float, optional): Minimum value for colormap normalization. If None, uses data minimum
        vmax (float, optional): Maximum value for colormap normalization. If None, uses data maximum
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