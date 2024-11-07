import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
import os

# Define the color map (using the provided color map)
default_voxel_color_map = {
    -4: [180, 187, 216],  # (lightgray) 'Building',
    -3: [78, 99, 63],   # (forestgreen) 'Tree',
    -2: [188, 143, 143],  # (saddle brown) 'Underground',
    -1: [188, 143, 143],  # (saddle brown) 'Underground',
    0: [235, 202, 178],   # 'Bareland (ground surface)',
    1: [123, 130, 59],   # (greenyellow) 'Rangeland (ground surface)',
    2: [108, 119, 129],   # (darkgray) 'Developed space (ground surface)',
    3: [59, 62, 87],      # (dimgray) 'Road (ground surface)',
    4: [116, 150, 66],   # (greenyellow) 'Tree (ground surface)',
    # 5: [16, 24, 48],    # (blue) 'Water (ground surface)',
    5: [44, 66, 133],    # (blue) 'Water (ground surface)',
    6: [112, 120, 56],   # (lightgreen) 'Agriculture land (ground surface)',
    7: [150, 166, 190],    # (lightgray) 'Building (ground surface)'
    8: [150, 166, 190],    # (lightgray) 'Building (ground surface)'
}

def create_custom_palette(color_map):
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, 3] = 255  # Set alpha to 255 for all colors
    palette[0] = [0, 0, 0, 0]  # Set the first color to black with alpha 0
    for i, color in enumerate(color_map.values(), start=1):
        palette[i, :3] = color
    return palette

def create_mapping(color_map):
    return {value: i+1 for i, value in enumerate(color_map.keys())}

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

def export_large_voxel_model(array, color_map, output_prefix, max_size=255):
    os.makedirs(output_prefix, exist_ok=True)

    for sub_array, (i, j, k) in split_array(array, max_size):
        output_file = f"{output_prefix}/chunk_{i}_{j}_{k}.vox"
        value_mapping, palette, shape = numpy_to_vox(sub_array, color_map, output_file)
        print(f"Chunk {i}_{j}_{k} saved as {output_file}")
        print(f"Shape: {shape}")

    return value_mapping, palette

def export_magicavoxel_vox(array, output_dir):

    value_mapping, palette = export_large_voxel_model(array, default_voxel_color_map, output_dir)
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