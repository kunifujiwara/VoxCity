import numpy as np
import os

def export_obj(array, output_dir, file_name, voxel_size):
    # Voxel color mapping (same as before)
    default_voxel_color_map = {
        -3: [180, 187, 216],  # Building
        -2: [78, 99, 63],     # Tree
        -1: [188, 143, 143],  # Underground
        1: [239, 228, 176],   # Bareland
        2: [123, 130, 59],    # Rangeland
        3: [108, 119, 129],   # Developed space
        4: [59, 62, 87],      # Road
        5: [116, 150, 66],    # Tree ground
        6: [44, 66, 133],     # Water
        7: [112, 120, 56],    # Agriculture land
        8: [150, 166, 190],   # Building ground
    }

    # Extract unique voxel values (excluding zero)
    unique_voxel_values = np.unique(array)
    unique_voxel_values = unique_voxel_values[unique_voxel_values != 0]

    # Map voxel values to material names
    voxel_value_to_material = {}
    for voxel_value in unique_voxel_values:
        material_name = f'material_{voxel_value}'
        voxel_value_to_material[voxel_value] = material_name

    # Normals for face directions
    normals = [
        (-1.0, 0.0, 0.0),  # 1: -X Left face
        (1.0, 0.0, 0.0),   # 2: +X Right face
        (0.0, -1.0, 0.0),  # 3: -Y Bottom face
        (0.0, 1.0, 0.0),   # 4: +Y Top face
        (0.0, 0.0, -1.0),  # 5: -Z Back face
        (0.0, 0.0, 1.0),   # 6: +Z Front face
    ]

    normal_indices = {
        'nx': 1,
        'px': 2,
        'ny': 3,
        'py': 4,
        'nz': 5,
        'pz': 6,
    }

    # Initialize lists
    vertex_list = []
    vertex_dict = {}  # To avoid duplicate vertices

    # Collect faces per material
    faces_per_material = {}

    # Dimensions
    size_z, size_y, size_x = array.shape  # Original shape (z, y, x)

    # Swap axes: Since we need to swap x and z, we transpose the array
    array = array.transpose(2, 1, 0)  # Now array[x, y, z]
    size_x, size_y, size_z = array.shape

    # Generate masks and perform greedy meshing for each face direction
    directions = [
        ('nx', (-1, 0, 0)),  # -X Left face
        ('px', (1, 0, 0)),   # +X Right face
        ('ny', (0, -1, 0)),  # -Y Bottom face
        ('py', (0, 1, 0)),   # +Y Top face
        ('nz', (0, 0, -1)),  # -Z Back face
        ('pz', (0, 0, 1)),   # +Z Front face
    ]

    for direction, normal in directions:
        normal_idx = normal_indices[direction]
        # Loop over the axis perpendicular to the face
        if direction in ('nx', 'px'):
            for x in range(size_x):
                mask = np.zeros((size_y, size_z), dtype=np.int32)
                for y in range(size_y):
                    for z in range(size_z):
                        voxel = array[x, y, z]
                        if direction == 'nx':
                            neighbor = array[x - 1, y, z] if x > 0 else 0
                        else:  # 'px'
                            neighbor = array[x + 1, y, z] if x + 1 < size_x else 0

                        if voxel != neighbor:
                            if voxel != 0:
                                mask[y, z] = voxel

                # Greedy meshing on the mask
                layer = x if direction == 'nx' else x + 1  # Adjust layer index for 'px'
                mesh_faces(mask, layer, 'x', direction == 'px', normal_idx, voxel_size, vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        elif direction in ('ny', 'py'):
            for y in range(size_y):
                mask = np.zeros((size_x, size_z), dtype=np.int32)
                for x in range(size_x):
                    for z in range(size_z):
                        voxel = array[x, y, z]
                        if direction == 'ny':
                            neighbor = array[x, y - 1, z] if y > 0 else 0
                        else:  # 'py'
                            neighbor = array[x, y + 1, z] if y + 1 < size_y else 0

                        if voxel != neighbor:
                            if voxel != 0:
                                mask[x, z] = voxel

                # Greedy meshing on the mask
                layer = y if direction == 'ny' else y + 1  # Adjust layer index for 'py'
                mesh_faces(mask, layer, 'y', direction == 'py', normal_idx, voxel_size, vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)
        elif direction in ('nz', 'pz'):
            for z in range(size_z):
                mask = np.zeros((size_x, size_y), dtype=np.int32)
                for x in range(size_x):
                    for y in range(size_y):
                        voxel = array[x, y, z]
                        if direction == 'nz':
                            neighbor = array[x, y, z - 1] if z > 0 else 0
                        else:  # 'pz'
                            neighbor = array[x, y, z + 1] if z + 1 < size_z else 0

                        if voxel != neighbor:
                            if voxel != 0:
                                mask[x, y] = voxel

                # Greedy meshing on the mask
                layer = z if direction == 'nz' else z + 1  # Adjust layer index for 'pz'
                mesh_faces(mask, layer, 'z', direction == 'pz', normal_idx, voxel_size, vertex_dict, vertex_list, faces_per_material, voxel_value_to_material)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    obj_file_path = os.path.join(output_dir, f'{file_name}.obj')
    mtl_file_path = os.path.join(output_dir, f'{file_name}.mtl')

    # Write OBJ file
    with open(obj_file_path, 'w') as f:
        f.write('# Generated OBJ file\n\n')
        f.write('# group\no \n\n')
        f.write(f'# material\nmtllib {file_name}.mtl\n\n')
        # Normals
        f.write('# normals\n')
        for nx, ny, nz in normals:
            f.write(f'vn {nx:.6f} {ny:.6f} {nz:.6f}\n')
        f.write('\n')
        # Vertices
        f.write('# verts\n')
        for vx, vy, vz in vertex_list:
            f.write(f'v {vx:.6f} {vy:.6f} {vz:.6f}\n')
        f.write('\n')
        # Faces per material
        f.write('# faces\n')
        for material_name, faces in faces_per_material.items():
            f.write(f'usemtl {material_name}\n')
            for face in faces:
                v_indices = face['vertices']
                normal_idx = face['normal_idx']
                face_str = ' '.join([f'{vi}//{normal_idx}' for vi in face['vertices']])
                f.write(f'f {face_str}\n')
            f.write('\n')

    # Write MTL file with adjusted properties (same as before)
    with open(mtl_file_path, 'w') as f:
        f.write('# Material file\n\n')
        for voxel_value in unique_voxel_values:
            material_name = voxel_value_to_material[voxel_value]
            color = default_voxel_color_map.get(voxel_value, [0, 0, 0])
            r, g, b = [c / 255.0 for c in color]
            f.write(f'newmtl {material_name}\n')
            f.write(f'Ka {r:.6f} {g:.6f} {b:.6f}\n')  # Ambient color
            f.write(f'Kd {r:.6f} {g:.6f} {b:.6f}\n')  # Diffuse color
            f.write(f'Ke {r:.6f} {g:.6f} {b:.6f}\n')  # Emissive color
            f.write('Ks 0.500000 0.500000 0.500000\n')  # Specular reflection
            f.write('Ns 50.000000\n')                   # Specular exponent
            f.write('illum 2\n\n')                      # Illumination model

    print(f'OBJ and MTL files have been generated in {output_dir} with the base name "{file_name}".')

def mesh_faces(mask, layer_index, axis, positive_direction, normal_idx, voxel_size, vertex_dict, vertex_list, faces_per_material, voxel_value_to_material):
    """
    Performs greedy meshing on the given mask and adds faces to the faces_per_material dictionary.
    """
    mask = mask.copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for u in range(h):
        for v in range(w):
            if visited[u, v] or mask[u, v] == 0:
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
            for du in range(height):
                for dv in range(width):
                    visited[u + du, v + dv] = True

            # Create face
            # Determine the coordinates based on the axis and direction
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
                continue  # Invalid axis

            # Swap x and z coordinates in coords
            coords = [(c[2], c[1], c[0]) for c in coords]

            # Map vertices to indices
            indices = []
            for coord in coords:
                if coord not in vertex_dict:
                    vertex_list.append(coord)
                    vertex_dict[coord] = len(vertex_list)
                indices.append(vertex_dict[coord])

            # Create face with correct winding order (CCW)
            if positive_direction:
                face_indices = [indices[0], indices[1], indices[2], indices[3]]
            else:
                face_indices = [indices[0], indices[3], indices[2], indices[1]]

            # Triangulate quad face
            faces = [
                {'vertices': [face_indices[0], face_indices[1], face_indices[2]], 'normal_idx': normal_idx},
                {'vertices': [face_indices[0], face_indices[2], face_indices[3]], 'normal_idx': normal_idx},
            ]

            if material_name not in faces_per_material:
                faces_per_material[material_name] = []
            faces_per_material[material_name].extend(faces)

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