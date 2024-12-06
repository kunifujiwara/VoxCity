import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
import os
from ..utils.visualization import get_default_voxel_color_map

# # Define the color map (using the provided color map)
# default_voxel_color_map = {
#     -12: [238, 242, 234],  # (light gray) 'plaster',
#     -11: [56, 78, 84],  # (Dark blue) 'glass',
#     -10: [206, 196, 160],  # (Light brown) 'stone',
#     -9: [139, 149, 159],  # (Gray) 'metal',
#     -8: [186, 187, 181],  # (Gray) 'concrete',
#     -7: [248, 166, 2],  # (Orange) 'wood',
#     -6: [138, 83, 74],  # (Pink) 'brick',
#     -5: [255, 0, 102],  # (Pink) 'Landmark',
#     -4: [180, 187, 216],  # (lightgray) 'Building',
#     -3: [78, 99, 63],   # (forestgreen) 'Tree',
#     -2: [188, 143, 143],  # (saddle brown) 'Underground',
#     -1: [188, 143, 143],  # (saddle brown) 'Underground',
#     0: [239, 228, 176],   # 'Bareland (ground surface)',
#     1: [123, 130, 59],   # 'Rangeland (ground surface)',
#     2: [97, 140, 86],   # 'Shrub (ground surface)',
#     3: [112, 120, 56],   #  'Agriculture land (ground surface)',
#     4: [116, 150, 66],   #  'Tree (ground surface)',
#     5: [187, 204, 40],   #  'Moss and lichen (ground surface)',
#     6: [77, 118, 99],    #  'Wet land (ground surface)',
#     7: [22, 61, 51],    #  'Mangrove (ground surface)',
#     8: [44, 66, 133],    #  'Water (ground surface)',
#     9: [205, 215, 224],    #  'Snow and ice (ground surface)',
#     10: [108, 119, 129],   #  'Developed space (ground surface)',
#     11: [59, 62, 87],      # 'Road (ground surface)',
#     12: [150, 166, 190],    #  'Building (ground surface)'
#     13: [239, 228, 176],    #  'No Data (ground surface)'
# }

def convert_colormap_and_array(original_map, original_array):
    """
    Convert a color map with arbitrary indices to sequential indices starting from 0
    and update the corresponding 3D numpy array.
    
    Args:
        original_map (dict): Dictionary with integer keys and RGB color value lists
        original_array (numpy.ndarray): 3D array with integer values corresponding to color map keys
        
    Returns:
        tuple: (new_color_map, new_array)
            - new_color_map (dict): Color map with sequential indices starting from 0
            - new_array (numpy.ndarray): Updated array with new indices
    """
    # Get all the keys and sort them
    keys = sorted(original_map.keys())
    
    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keys)}
    
    # Create new color map with sequential indices
    new_map = {}
    for new_idx, old_idx in enumerate(keys):
        new_map[new_idx] = original_map[old_idx]
    
    # Create a copy of the original array
    new_array = original_array.copy()
    
    # Replace old indices with new ones in the array
    for old_idx, new_idx in old_to_new.items():
        new_array[original_array == old_idx] = new_idx
    
    # # Print the new map in a formatted way
    # print("new_colormap = {")
    # for key, value in new_map.items():
    #     # Get the comment from the original map if it exists
    #     original_key = keys[key]
    #     original_line = str(original_map[original_key])
    #     comment = ""
    #     if "#" in original_line:
    #         comment = "#" + original_line.split("#")[1].strip()
        
    #     print(f"    {key}: {value},  {comment}")
    # print("}")
    
    return new_map, new_array

def create_custom_palette(color_map):
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, 3] = 255  # Set alpha to 255 for all colors
    palette[0] = [0, 0, 0, 0]  # Set the first color to black with alpha 0
    for i, color in enumerate(color_map.values(), start=1):
        palette[i, :3] = color
    return palette

def create_mapping(color_map):
    return {value: i+2 for i, value in enumerate(color_map.keys())}

def split_array(array, max_size=255):
    x, y, z = array.shape
    x_splits = (x + max_size - 1) // max_size
    y_splits = (y + max_size - 1) // max_size
    z_splits = (z + max_size - 1) // max_size

    for i in range(x_splits):
        for j in range(y_splits):
            for k in range(z_splits):
                x_start, x_end = i * max_size, min((i + 1) * max_size, x)
                y_start, y_end = j * max_size, min((j + 1) * max_size, y)
                z_start, z_end = k * max_size, min((k + 1) * max_size, z)
                yield (
                    array[x_start:x_end, y_start:y_end, z_start:z_end],
                    (i, j, k)
                )

def numpy_to_vox(array, color_map, output_file):
    palette = create_custom_palette(color_map)
    value_mapping = create_mapping(color_map)
    value_mapping[0] = 0  # Ensure 0 maps to 0 (void)

    array_flipped = np.flip(array, axis=2)
    array_transposed = np.transpose(array_flipped, (1, 2, 0))
    mapped_array = np.vectorize(value_mapping.get)(array_transposed, 0)

    vox = Vox.from_dense(mapped_array.astype(np.uint8))
    vox.palette = palette
    VoxWriter(output_file, vox).write()

    return value_mapping, palette, array_transposed.shape

def export_large_voxel_model(array, color_map, output_prefix, max_size=255, base_filename='chunk'):
    os.makedirs(output_prefix, exist_ok=True)

    for sub_array, (i, j, k) in split_array(array, max_size):
        output_file = f"{output_prefix}/{base_filename}_{i}_{j}_{k}.vox"
        value_mapping, palette, shape = numpy_to_vox(sub_array, color_map, output_file)
        print(f"Chunk {i}_{j}_{k} saved as {output_file}")
        print(f"Shape: {shape}")

    return value_mapping, palette

def export_magicavoxel_vox(array, output_dir, base_filename='chunk', voxel_color_map=None):
    # Voxel color mapping (same as before)
    if voxel_color_map is None:
        voxel_color_map = get_default_voxel_color_map()
    converted_voxel_color_map, converted_array = convert_colormap_and_array(voxel_color_map, array)

    value_mapping, palette = export_large_voxel_model(converted_array, converted_voxel_color_map, output_dir, base_filename=base_filename)
    print(f"\tvox files was successfully exported in {output_dir}")
    # print(f"Original shape: {array.shape}")
    # print(f"Shape in VOX file: {new_shape}")

    # # Print the value mapping for reference
    # for original, new in value_mapping.items():
    #     print(f"Original value {original} mapped to palette index {new}")
    #     if new == 0:
    #         print("  Color: Void (transparent)")
    #     else:
    #         print(f"  Color: {palette[new, :3]}")
    #     print()