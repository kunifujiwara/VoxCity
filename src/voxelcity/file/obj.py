import numpy as np
import os
from numba import njit, prange

import numpy as np
import os
import matplotlib.pyplot as plt
from ..utils.visualization import get_default_voxel_color_map

def convert_colormap_indices(original_map):
    """
    Convert a color map with arbitrary indices to sequential indices starting from 0.
    """
    keys = sorted(original_map.keys())
    new_map = {}
    for new_idx, old_idx in enumerate(keys):
        new_map[new_idx] = original_map[old_idx]
    
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
    """Helper function to create properly oriented face vertices"""
    if axis == 'y':
        if positive_direction:  # +Y face
            return [coords[3], coords[2], coords[1], coords[0]]  # Reverse order for +Y
        else:  # -Y face
            return [coords[0], coords[1], coords[2], coords[3]]  # Standard order for -Y
    else:
        if positive_direction:
            return [coords[0], coords[3], coords[2], coords[1]]
        else:
            return [coords[0], coords[1], coords[2], coords[3]]

def mesh_faces(mask, layer_index, axis, positive_direction, normal_idx, voxel_size_m, 
              vertex_dict, vertex_list, faces_per_material, voxel_value_to_material):
    """
    Performs greedy meshing on the given mask and adds faces to the faces_per_material dictionary.
    """

    voxel_size = voxel_size_m #convet meter to centimeter

    mask = mask.copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for u in range(h):
        v = 0
        while v < w:
            if visited[u, v] or mask[u, v] == 0:
                v += 1
                continue

            voxel_value = mask[u, v]
            material_name = voxel_value_to_material[voxel_value]

            # Find the maximum width
            width = 1
            while v + width < w and mask[u, v + width] == voxel_value and not visited[u, v + width]:
                width += 1

            # Find the maximum height
            height = 1
            done = False
            while u + height < h and not done:
                for k in range(width):
                    if mask[u + height, v + k] != voxel_value or visited[u + height, v + k]:
                        done = True
                        break
                if not done:
                    height += 1

            # Mark visited
            visited[u:u + height, v:v + width] = True

            # Create face coordinates based on axis
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

            # Swap coordinates and apply correct winding order
            coords = [(c[2], c[1], c[0]) for c in coords]
            face_vertices = create_face_vertices(coords, positive_direction, axis)

            # Map vertices to indices
            indices = []
            for coord in face_vertices:
                if coord not in vertex_dict:
                    vertex_list.append(coord)
                    vertex_dict[coord] = len(vertex_list)
                indices.append(vertex_dict[coord])

            # Create triangulated faces with consistent winding
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

            if material_name not in faces_per_material:
                faces_per_material[material_name] = []
            faces_per_material[material_name].extend(faces)

            v += width

def export_obj(array, output_dir, file_name, voxel_size, voxel_color_map=None):
    """
    Export a voxel array to OBJ format with corrected face orientations.
    
    Args:
        array: 3D numpy array containing voxel values
        output_dir: Directory to save the OBJ and MTL files
        file_name: Base name for the output files
        voxel_size: Size of each voxel
        voxel_color_map: Dictionary mapping voxel values to RGB colors
    """
    if voxel_color_map is None:
        voxel_color_map = get_default_voxel_color_map()

    # Extract unique voxel values (excluding zero)
    unique_voxel_values = np.unique(array)
    unique_voxel_values = unique_voxel_values[unique_voxel_values != 0]

    # Map voxel values to material names
    voxel_value_to_material = {val: f'material_{val}' for val in unique_voxel_values}

    # Modified normals to ensure consistent orientation
    normals = [
        (1.0, 0.0, 0.0),   # 1: +X Right face
        (-1.0, 0.0, 0.0),  # 2: -X Left face
        (0.0, 1.0, 0.0),   # 3: +Y Top face
        (0.0, -1.0, 0.0),  # 4: -Y Bottom face
        (0.0, 0.0, 1.0),   # 5: +Z Front face
        (0.0, 0.0, -1.0),  # 6: -Z Back face
    ]

    normal_indices = {
        'nx': 2,
        'px': 1,
        'ny': 4,
        'py': 3,
        'nz': 6,
        'pz': 5,
    }

    # Initialize lists and dictionaries
    vertex_list = []
    vertex_dict = {}
    faces_per_material = {}

    # Swap axes for correct orientation
    array = array.transpose(2, 1, 0)  # Now array[x, y, z]
    size_x, size_y, size_z = array.shape

    # Process each direction
    directions = [
        ('nx', (-1, 0, 0)),
        ('px', (1, 0, 0)),
        ('ny', (0, -1, 0)),
        ('py', (0, 1, 0)),
        ('nz', (0, 0, -1)),
        ('pz', (0, 0, 1)),
    ]

    for direction, normal in directions:
        normal_idx = normal_indices[direction]
        
        if direction in ('nx', 'px'):
            for x in range(size_x):
                voxel_slice = array[x, :, :]
                if direction == 'nx':
                    neighbor_slice = array[x - 1, :, :] if x > 0 else np.zeros_like(voxel_slice)
                    layer = x
                else:
                    neighbor_slice = array[x + 1, :, :] if x + 1 < size_x else np.zeros_like(voxel_slice)
                    layer = x + 1

                mask = np.where((voxel_slice != neighbor_slice) & (voxel_slice != 0), voxel_slice, 0)
                mesh_faces(mask, layer, 'x', direction == 'px', normal_idx, voxel_size,
                         vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

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

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write OBJ file
    obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
    mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

    with open(obj_file_path, 'w') as f:
        f.write('# Generated OBJ file\n\n')
        f.write('# group\no \n\n')
        f.write(f'# material\nmtllib {file_name}.mtl\n\n')
        
        # Write normals
        f.write('# normals\n')
        for nx, ny, nz in normals:
            f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        f.write('\n')
        
        # Write vertices
        f.write('# verts\n')
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')
        
        # Write faces
        f.write('# faces\n')
        for material_name, faces in faces_per_material.items():
            f.write(f'usemtl {material_name}\n')
            for face in faces:
                v_indices = [str(vi) for vi in face['vertices']]
                normal_idx = face['normal_idx']
                face_str = ' '.join([f'{vi}//{normal_idx}' for vi in face['vertices']])
                f.write(f'f {face_str}\n')
            f.write('\n')

    # Write MTL file
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

# def convert_colormap_indices(original_map):
#     """
#     Convert a color map with arbitrary indices to sequential indices starting from 0.
    
#     Args:
#         original_map (dict): Dictionary with integer keys and RGB color value lists
        
#     Returns:
#         dict: New color map with sequential indices starting from 0
#     """
#     # Get all the keys and sort them
#     keys = sorted(original_map.keys())
    
#     # Create new map with sequential indices starting from 0
#     new_map = {}
#     for new_idx, old_idx in enumerate(keys):
#         new_map[new_idx] = original_map[old_idx]
    
#     # Print the new map in a formatted way
#     print("new_colormap = {")
#     for key, value in new_map.items():
#         # Get the comment from the original map if it exists
#         original_key = keys[key]
#         original_line = str(original_map[original_key])  # Convert to string to get the format
#         comment = ""
#         if "#" in original_line:
#             comment = "#" + original_line.split("#")[1].strip()
        
#         print(f"    {key}: {value},  {comment}")
#     print("}")
    
#     return new_map

# def export_obj(array, output_dir, file_name, voxel_size, voxel_color_map=None):
#     # Voxel color mapping (same as before)
#     if voxel_color_map is None:
#         voxel_color_map = get_default_voxel_color_map()

#     # Extract unique voxel values (excluding zero)
#     unique_voxel_values = np.unique(array)
#     unique_voxel_values = unique_voxel_values[unique_voxel_values != 0]

#     # Map voxel values to material names
#     voxel_value_to_material = {}
#     for voxel_value in unique_voxel_values:
#         material_name = f'material_{voxel_value}'
#         voxel_value_to_material[voxel_value] = material_name

#     # Normals for face directions
#     normals = [
#         (-1.0, 0.0, 0.0),  # 1: -X Left face
#         (1.0, 0.0, 0.0),   # 2: +X Right face
#         (0.0, -1.0, 0.0),  # 3: -Y Bottom face
#         (0.0, 1.0, 0.0),   # 4: +Y Top face
#         (0.0, 0.0, -1.0),  # 5: -Z Back face
#         (0.0, 0.0, 1.0),   # 6: +Z Front face
#     ]

#     normal_indices = {
#         'nx': 1,
#         'px': 2,
#         'ny': 3,
#         'py': 4,
#         'nz': 5,
#         'pz': 6,
#     }

#     # Initialize lists
#     vertex_list = []
#     vertex_dict = {}  # To avoid duplicate vertices

#     # Collect faces per material
#     faces_per_material = {}

#     # Dimensions
#     size_z, size_y, size_x = array.shape  # Original shape (z, y, x)

#     # Swap axes: Since we need to swap x and z, we transpose the array
#     array = array.transpose(2, 1, 0)  # Now array[x, y, z]
#     size_x, size_y, size_z = array.shape

#     # Generate masks and perform greedy meshing for each face direction
#     directions = [
#         ('nx', (-1, 0, 0)),  # -X Left face
#         ('px', (1, 0, 0)),   # +X Right face
#         ('ny', (0, -1, 0)),  # -Y Bottom face
#         ('py', (0, 1, 0)),   # +Y Top face
#         ('nz', (0, 0, -1)),  # -Z Back face
#         ('pz', (0, 0, 1)),   # +Z Front face
#     ]

#     for direction, normal in directions:
#         normal_idx = normal_indices[direction]
#         # Loop over the axis perpendicular to the face
#         if direction in ('nx', 'px'):
#             for x in range(size_x):
#                 # Vectorized mask generation
#                 voxel_slice = array[x, :, :]
#                 if direction == 'nx':
#                     if x > 0:
#                         neighbor_slice = array[x - 1, :, :]
#                     else:
#                         neighbor_slice = np.zeros((size_y, size_z), dtype=array.dtype)
#                     layer = x
#                 else:  # 'px'
#                     if x + 1 < size_x:
#                         neighbor_slice = array[x + 1, :, :]
#                     else:
#                         neighbor_slice = np.zeros((size_y, size_z), dtype=array.dtype)
#                     layer = x + 1  # Adjust layer index for 'px'

#                 mask = np.where(
#                     (voxel_slice != neighbor_slice) & (voxel_slice != 0),
#                     voxel_slice,
#                     0
#                 )

#                 # Greedy meshing on the mask
#                 mesh_faces(mask, layer, 'x', direction == 'px', normal_idx, voxel_size,
#                            vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
#         elif direction in ('ny', 'py'):
#             for y in range(size_y):
#                 # Vectorized mask generation
#                 voxel_slice = array[:, y, :]
#                 if direction == 'ny':
#                     if y > 0:
#                         neighbor_slice = array[:, y - 1, :]
#                     else:
#                         neighbor_slice = np.zeros((size_x, size_z), dtype=array.dtype)
#                     layer = y
#                 else:  # 'py'
#                     if y + 1 < size_y:
#                         neighbor_slice = array[:, y + 1, :]
#                     else:
#                         neighbor_slice = np.zeros((size_x, size_z), dtype=array.dtype)
#                     layer = y + 1  # Adjust layer index for 'py'

#                 mask = np.where(
#                     (voxel_slice != neighbor_slice) & (voxel_slice != 0),
#                     voxel_slice,
#                     0
#                 )

#                 # Greedy meshing on the mask
#                 mesh_faces(mask, layer, 'y', direction == 'py', normal_idx, voxel_size,
#                            vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
#         elif direction in ('nz', 'pz'):
#             for z in range(size_z):
#                 # Vectorized mask generation
#                 voxel_slice = array[:, :, z]
#                 if direction == 'nz':
#                     if z > 0:
#                         neighbor_slice = array[:, :, z - 1]
#                     else:
#                         neighbor_slice = np.zeros((size_x, size_y), dtype=array.dtype)
#                     layer = z
#                 else:  # 'pz'
#                     if z + 1 < size_z:
#                         neighbor_slice = array[:, :, z + 1]
#                     else:
#                         neighbor_slice = np.zeros((size_x, size_y), dtype=array.dtype)
#                     layer = z + 1  # Adjust layer index for 'pz'

#                 mask = np.where(
#                     (voxel_slice != neighbor_slice) & (voxel_slice != 0),
#                     voxel_slice,
#                     0
#                 )

#                 # Greedy meshing on the mask
#                 mesh_faces(mask, layer, 'z', direction == 'pz', normal_idx, voxel_size,
#                            vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # File paths
#     obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
#     mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

#     # Write OBJ file
#     with open(obj_file_path, 'w') as f:
#         f.write('# Generated OBJ file\n\n')
#         f.write('# group\no \n\n')
#         f.write(f'# material\nmtllib {file_name}.mtl\n\n')
#         # Normals
#         f.write('# normals\n')
#         for nx, ny, nz in normals:
#             f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
#         f.write('\n')
#         # Vertices
#         f.write('# verts\n')
#         for vx, vy, vz in vertex_list:
#             f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
#         f.write('\n')
#         # Faces per material
#         f.write('# faces\n')
#         for material_name, faces in faces_per_material.items():
#             f.write(f'usemtl {material_name}\n')
#             for face in faces:
#                 v_indices = face['vertices']
#                 normal_idx = face['normal_idx']
#                 face_str = ' '.join([f'{vi}//{normal_idx}' for vi in face['vertices']])
#                 f.write(f'f {face_str}\n')
#             f.write('\n')

#     # Write MTL file with adjusted properties (same as before)
#     with open(mtl_file_path, 'w') as f:
#         f.write('# Material file\n\n')
#         for voxel_value in unique_voxel_values:
#             material_name = voxel_value_to_material[voxel_value]
#             color = voxel_color_map.get(voxel_value, [0, 0, 0])
#             r, g, b = [c / 255.0 for c in color]
#             f.write(f'newmtl {material_name}\n')
#             f.write(f'Ka {r:.6f} {g:.6f} {b:.6f}\n')  # Ambient color
#             f.write(f'Kd {r:.6f} {g:.6f} {b:.6f}\n')  # Diffuse color
#             f.write(f'Ke {r:.6f} {g:.6f} {b:.6f}\n')  # Emissive color
#             f.write('Ks 0.500000 0.500000 0.500000\n')  # Specular reflection
#             f.write('Ns 50.000000\n')                   # Specular exponent
#             f.write('illum 2\n\n')                      # Illumination model

#     print(f'OBJ and MTL files have been generated in {output_dir} with the base name "{file_name}".')

# def mesh_faces(mask, layer_index, axis, positive_direction, normal_idx, voxel_size, vertex_dict, vertex_list, faces_per_material, voxel_value_to_material):
#     """
#     Performs greedy meshing on the given mask and adds faces to the faces_per_material dictionary.
#     """
#     mask = mask.copy()
#     h, w = mask.shape
#     visited = np.zeros_like(mask, dtype=bool)

#     for u in range(h):
#         v = 0
#         while v < w:
#             if visited[u, v] or mask[u, v] == 0:
#                 v += 1
#                 continue

#             voxel_value = mask[u, v]
#             material_name = voxel_value_to_material[voxel_value]

#             # Find the maximum width
#             width = 1
#             while v + width < w and mask[u, v + width] == voxel_value and not visited[u, v + width]:
#                 width += 1

#             # Find the maximum height
#             height = 1
#             done = False
#             while u + height < h and not done:
#                 for k in range(width):
#                     if mask[u + height, v + k] != voxel_value or visited[u + height, v + k]:
#                         done = True
#                         break
#                 if not done:
#                     height += 1

#             # Mark visited
#             visited[u:u + height, v:v + width] = True

#             # Create face
#             # Determine the coordinates based on the axis and direction
#             if axis == 'x':
#                 i = float(layer_index) * voxel_size
#                 y0 = float(u) * voxel_size
#                 y1 = float(u + height) * voxel_size
#                 z0 = float(v) * voxel_size
#                 z1 = float(v + width) * voxel_size
#                 coords = [
#                     (i, y0, z0),
#                     (i, y1, z0),
#                     (i, y1, z1),
#                     (i, y0, z1),
#                 ]
#             elif axis == 'y':
#                 i = float(layer_index) * voxel_size
#                 x0 = float(u) * voxel_size
#                 x1 = float(u + height) * voxel_size
#                 z0 = float(v) * voxel_size
#                 z1 = float(v + width) * voxel_size
#                 coords = [
#                     (x0, i, z0),
#                     (x1, i, z0),
#                     (x1, i, z1),
#                     (x0, i, z1),
#                 ]
#             elif axis == 'z':
#                 i = float(layer_index) * voxel_size
#                 x0 = float(u) * voxel_size
#                 x1 = float(u + height) * voxel_size
#                 y0 = float(v) * voxel_size
#                 y1 = float(v + width) * voxel_size
#                 coords = [
#                     (x0, y0, i),
#                     (x1, y0, i),
#                     (x1, y1, i),
#                     (x0, y1, i),
#                 ]
#             else:
#                 continue  # Invalid axis

#             # Swap x and z coordinates in coords
#             coords = [(c[2], c[1], c[0]) for c in coords]

#             # Map vertices to indices
#             indices = []
#             for coord in coords:
#                 if coord not in vertex_dict:
#                     vertex_list.append(coord)
#                     vertex_dict[coord] = len(vertex_list)
#                 indices.append(vertex_dict[coord])

#             # Create face with correct winding order (CCW)
#             if positive_direction:
#                 face_indices = [indices[0], indices[1], indices[2], indices[3]]
#             else:
#                 face_indices = [indices[0], indices[3], indices[2], indices[1]]

#             # Triangulate quad face
#             faces = [
#                 {'vertices': [face_indices[0], face_indices[1], face_indices[2]], 'normal_idx': normal_idx},
#                 {'vertices': [face_indices[0], face_indices[2], face_indices[3]], 'normal_idx': normal_idx},
#             ]

#             if material_name not in faces_per_material:
#                 faces_per_material[material_name] = []
#             faces_per_material[material_name].extend(faces)

#             v += width  # Move to the next segment


def grid_to_obj(value_array_ori, dem_array_ori, output_dir, file_name, cell_size, offset,
                 colormap_name='viridis', num_colors=256, alpha=1.0, vmin=None, vmax=None):
    """
    Converts a 2D array of values and a corresponding DEM array to an OBJ file
    with specified colormap, transparency, and value range.

    Parameters:
    - value_array: 2D NumPy array of values to visualize.
    - dem_array: 2D NumPy array of DEM values corresponding to value_array.
    - output_dir: Directory to save the OBJ and MTL files.
    - file_name: Base name for the output files.
    - cell_size: Size of each cell in the grid (e.g., in meters).
    - offset: Elevation offset added after quantization.
    - colormap_name: Name of the Matplotlib colormap to use.
    - num_colors: Number of discrete colors to use from the colormap.
    - alpha: Transparency value between 0.0 (transparent) and 1.0 (opaque).
    - vmin: Minimum value for colormap normalization.
    - vmax: Maximum value for colormap normalization.
    """
    # Validate input arrays
    if value_array_ori.shape != dem_array_ori.shape:
        raise ValueError("The value array and DEM array must have the same shape.")
    
    # Get the dimensions
    rows, cols = value_array_ori.shape

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

    # Initialize lists
    vertex_list = []
    vertex_dict = {}  # To avoid duplicate vertices
    vertex_index = 1  # OBJ indices start at 1

    faces_per_material = {}

    # For each valid cell
    for idx in valid_indices:
        i, j = idx  # i is the row index, j is the column index
        value = value_array[i, j]
        normalized_value = normalized_values[i, j]

        # Get the color from the colormap
        rgba = colormap(normalized_value)
        rgb = rgba[:3]  # Ignore alpha channel
        r, g, b = [int(c * 255) for c in rgb]

        # Quantize the color
        color_key = (r, g, b)
        material_name = f'material_{r}_{g}_{b}'

        if material_name not in color_to_material:
            color_to_material[material_name] = {
                'r': r / 255.0,
                'g': g / 255.0,
                'b': b / 255.0,
                'alpha': alpha
            }
            materials.append(material_name)

        # Calculate the vertices of the quad
        x0 = i * cell_size
        x1 = (i + 1) * cell_size
        y0 = j * cell_size
        y1 = (j + 1) * cell_size

        # Calculate the z-coordinate
        z = cell_size * int(dem_array[i, j] / cell_size + 1.5) + offset

        # Define the four corners of the cell (quad)
        vertices = [
            (x0, y0, z),
            (x1, y0, z),
            (x1, y1, z),
            (x0, y1, z),
        ]

        # Map vertices to indices
        indices = []
        for v in vertices:
            if v not in vertex_dict:
                vertex_list.append(v)
                vertex_dict[v] = vertex_index
                vertex_index += 1
            indices.append(vertex_dict[v])

        # Create face (quad split into two triangles)
        faces = [
            {'vertices': [indices[0], indices[1], indices[2]]},
            {'vertices': [indices[0], indices[2], indices[3]]},
        ]

        # Add faces to faces_per_material
        if material_name not in faces_per_material:
            faces_per_material[material_name] = []
        faces_per_material[material_name].extend(faces)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
    mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

    # Write OBJ file
    with open(obj_file_path, 'w') as f:
        f.write('# Generated OBJ file\n\n')
        f.write(f'mtllib {file_name}.mtl\n\n')
        # Vertices
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')
        # Faces per material
        for material_name in materials:
            f.write(f'usemtl {material_name}\n')
            faces = faces_per_material[material_name]
            for face in faces:
                v_indices = face['vertices']
                face_str = ' '.join([f'{vi}' for vi in v_indices])
                f.write(f'f {face_str}\n')
            f.write('\n')

    # Write MTL file with transparency
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













# import numpy as np
# import os

# def array_to_obj(array, output_dir, file_name, voxel_size):
#     # Voxel color mapping
#     default_voxel_color_map = {
#         -3: [180, 187, 216],  # Building
#         -2: [78, 99, 63],     # Tree
#         -1: [188, 143, 143],  # Underground
#         # 0: 'Air (Void)',     # Ignored
#         1: [239, 228, 176],   # Bareland
#         2: [123, 130, 59],    # Rangeland
#         3: [108, 119, 129],   # Developed space
#         4: [59, 62, 87],      # Road
#         5: [116, 150, 66],    # Tree ground
#         6: [44, 66, 133],     # Water
#         7: [112, 120, 56],    # Agriculture land
#         8: [150, 166, 190],   # Building ground
#     }

#     # Extract unique voxel values (excluding zero)
#     unique_voxel_values = np.unique(array)
#     unique_voxel_values = unique_voxel_values[unique_voxel_values != 0]

#     # Map voxel values to material names
#     voxel_value_to_material = {}
#     for voxel_value in unique_voxel_values:
#         material_name = f'material_{voxel_value}'
#         voxel_value_to_material[voxel_value] = material_name

#     # Normals
#     normals = [
#         (-1, 0, 0),  # Left
#         (1, 0, 0),   # Right
#         (0, 0, 1),   # Front
#         (0, 0, -1),  # Back
#         (0, -1, 0),  # Bottom
#         (0, 1, 0)    # Top
#     ]

#     # Initialize lists
#     vertex_list = []
#     vertex_dict = {}  # To avoid duplicate vertices
#     vertex_index = 1  # OBJ indices start at 1

#     # Collect faces per material
#     faces_per_material = {}

#     # Generate vertices and faces
#     for z in range(array.shape[0]):
#         for y in range(array.shape[1]):
#             for x in range(array.shape[2]):
#                 voxel_value = array[z, y, x]
#                 if voxel_value != 0:
#                     # Swap x and z coordinates
#                     xx, yy, zz = z, y, x  # Swap x and z

#                     # Scale coordinates by voxel_size
#                     xx *= voxel_size
#                     yy *= voxel_size
#                     zz *= voxel_size

#                     # Cube vertices
#                     cube_vertices = [
#                         (xx, yy, zz),
#                         (xx+voxel_size, yy, zz),
#                         (xx+voxel_size, yy+voxel_size, zz),
#                         (xx, yy+voxel_size, zz),
#                         (xx, yy, zz+voxel_size),
#                         (xx+voxel_size, yy, zz+voxel_size),
#                         (xx+voxel_size, yy+voxel_size, zz+voxel_size),
#                         (xx, yy+voxel_size, zz+voxel_size),
#                     ]
#                     # Map vertices to indices to avoid duplicates
#                     indices = []
#                     for v in cube_vertices:
#                         if v not in vertex_dict:
#                             vertex_list.append(v)
#                             vertex_dict[v] = vertex_index
#                             vertex_index += 1
#                         indices.append(vertex_dict[v])

#                     idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7 = indices

#                     # Faces with counter-clockwise vertex order
#                     faces = []
#                     # Left face (-x)
#                     faces.append({'vertices': [idx0, idx4, idx7], 'normal_idx': 1})
#                     faces.append({'vertices': [idx0, idx7, idx3], 'normal_idx': 1})
#                     # Right face (+x)
#                     faces.append({'vertices': [idx1, idx2, idx6], 'normal_idx': 2})
#                     faces.append({'vertices': [idx1, idx6, idx5], 'normal_idx': 2})
#                     # Front face (+z)
#                     faces.append({'vertices': [idx4, idx5, idx6], 'normal_idx': 3})
#                     faces.append({'vertices': [idx4, idx6, idx7], 'normal_idx': 3})
#                     # Back face (-z)
#                     faces.append({'vertices': [idx0, idx3, idx2], 'normal_idx': 4})
#                     faces.append({'vertices': [idx0, idx2, idx1], 'normal_idx': 4})
#                     # Bottom face (-y)
#                     faces.append({'vertices': [idx0, idx1, idx5], 'normal_idx': 5})
#                     faces.append({'vertices': [idx0, idx5, idx4], 'normal_idx': 5})
#                     # Top face (+y)
#                     faces.append({'vertices': [idx3, idx7, idx6], 'normal_idx': 6})
#                     faces.append({'vertices': [idx3, idx6, idx2], 'normal_idx': 6})

#                     # Add faces to faces_per_material
#                     material_name = voxel_value_to_material[voxel_value]
#                     if material_name not in faces_per_material:
#                         faces_per_material[material_name] = []
#                     faces_per_material[material_name].extend(faces)

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # File paths
#     obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
#     mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

#     # Write OBJ file
#     with open(obj_file_path, 'w') as f:
#         f.write('# MagicaVoxel @ Ephtracy\n\n')
#         f.write('# group\no \n\n')
#         f.write(f'# material\nmtllib {file_name}.mtl\n\n')
#         # Normals
#         f.write('# normals\n')
#         for nx, ny, nz in normals:
#             f.write(f'vn {nx} {ny} {nz}\n')
#         f.write('\n')
#         # Vertices
#         f.write('# verts\n')
#         for vx, vy, vz in vertex_list:
#             f.write(f'v {vx} {vy} {vz}\n')
#         f.write('\n')
#         # Faces per material
#         f.write('# faces\n')
#         for material_name, faces in faces_per_material.items():
#             f.write(f'usemtl {material_name}\n')
#             for face in faces:
#                 v_indices = face['vertices']
#                 normal_idx = face['normal_idx']
#                 face_str = ' '.join([f'{vi}//{normal_idx}' for vi in v_indices])  # Use '//' when no texture coordinates
#                 f.write(f'f {face_str}\n')

#     # Write MTL file with adjusted properties
#     with open(mtl_file_path, 'w') as f:
#         f.write('# MagicaVoxel @ Ephtracy\n\n')
#         for voxel_value in unique_voxel_values:
#             material_name = voxel_value_to_material[voxel_value]
#             color = default_voxel_color_map.get(voxel_value, [0, 0, 0])
#             r, g, b = [c / 255.0 for c in color]
#             f.write(f'newmtl {material_name}\n')
#             f.write(f'Ka {r:.3f} {g:.3f} {b:.3f}\n')  # Ambient color
#             f.write(f'Kd {r:.3f} {g:.3f} {b:.3f}\n')  # Diffuse color
#             f.write(f'Ke {r:.3f} {g:.3f} {b:.3f}\n')  # Emissive color
#             f.write('Ks 0.5 0.5 0.5\n')                # Specular reflection
#             f.write('Ns 50.0\n')                       # Specular exponent
#             f.write('illum 2\n\n')                     # Illumination model

#     print(f'OBJ and MTL files have been generated in {output_dir} with the base name "{file_name}".')

# output_directory = './output'
# output_file_name = 'sample_model'
# voxel_size_in_meters = 5  # Adjust voxel size as needed

# array_to_obj(voxelcity_grid, output_directory, output_file_name, voxel_size_in_meters)