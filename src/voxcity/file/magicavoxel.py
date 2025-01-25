"""
Module for handling MagicaVoxel .vox files.

This module provides functionality for converting 3D numpy arrays to MagicaVoxel .vox files,
including color mapping and splitting large models into smaller chunks.
"""

# Required imports for voxel file handling and array manipulation
import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
import os
from ..utils.visualization import get_default_voxel_color_map

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
    
    return new_map, new_array

def create_custom_palette(color_map):
    """
    Create a palette array from a color map dictionary.
    
    Args:
        color_map (dict): Dictionary mapping indices to RGB color values
        
    Returns:
        numpy.ndarray: 256x4 array containing RGBA color values
    """
    # Initialize empty palette with alpha channel
    palette = np.zeros((256, 4), dtype=np.uint8)
    palette[:, 3] = 255  # Set alpha to 255 for all colors
    palette[0] = [0, 0, 0, 0]  # Set the first color to transparent black
    
    # Fill palette with RGB values from color map
    for i, color in enumerate(color_map.values(), start=1):
        palette[i, :3] = color
    return palette

def create_mapping(color_map):
    """
    Create a mapping from color map keys to sequential indices.
    
    Args:
        color_map (dict): Dictionary mapping indices to RGB color values
        
    Returns:
        dict: Mapping from original indices to sequential indices starting at 2
    """
    # Create mapping starting at index 2 (0 is void, 1 is reserved)
    return {value: i+2 for i, value in enumerate(color_map.keys())}

def split_array(array, max_size=255):
    """
    Split a 3D array into smaller chunks that fit within MagicaVoxel size limits.
    
    Args:
        array (numpy.ndarray): 3D array to split
        max_size (int): Maximum size allowed for each dimension
        
    Yields:
        tuple: (sub_array, (i,j,k)) where sub_array is the chunk and (i,j,k) are the chunk indices
    """
    # Calculate number of splits needed in each dimension
    x, y, z = array.shape
    x_splits = (x + max_size - 1) // max_size
    y_splits = (y + max_size - 1) // max_size
    z_splits = (z + max_size - 1) // max_size

    # Iterate through all possible chunk positions
    for i in range(x_splits):
        for j in range(y_splits):
            for k in range(z_splits):
                # Calculate chunk boundaries
                x_start, x_end = i * max_size, min((i + 1) * max_size, x)
                y_start, y_end = j * max_size, min((j + 1) * max_size, y)
                z_start, z_end = k * max_size, min((k + 1) * max_size, z)
                yield (
                    array[x_start:x_end, y_start:y_end, z_start:z_end],
                    (i, j, k)
                )

def numpy_to_vox(array, color_map, output_file):
    """
    Convert a numpy array to a MagicaVoxel .vox file.
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data
        color_map (dict): Dictionary mapping indices to RGB color values
        output_file (str): Path to save the .vox file
        
    Returns:
        tuple: (value_mapping, palette, shape) containing the index mapping, color palette and output shape
    """
    # Create color palette and value mapping
    palette = create_custom_palette(color_map)
    value_mapping = create_mapping(color_map)
    value_mapping[0] = 0  # Ensure 0 maps to 0 (void)

    # Transform array to match MagicaVoxel coordinate system
    array_flipped = np.flip(array, axis=2)  # Flip Z axis
    array_transposed = np.transpose(array_flipped, (1, 2, 0))  # Reorder axes
    mapped_array = np.vectorize(value_mapping.get)(array_transposed, 0)

    # Create and save vox file
    vox = Vox.from_dense(mapped_array.astype(np.uint8))
    vox.palette = palette
    VoxWriter(output_file, vox).write()

    return value_mapping, palette, array_transposed.shape

def export_large_voxel_model(array, color_map, output_prefix, max_size=255, base_filename='chunk'):
    """
    Export a large voxel model by splitting it into multiple .vox files.
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data
        color_map (dict): Dictionary mapping indices to RGB color values
        output_prefix (str): Directory to save the .vox files
        max_size (int): Maximum size allowed for each dimension
        base_filename (str): Base name for the output files
        
    Returns:
        tuple: (value_mapping, palette) containing the index mapping and color palette
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_prefix, exist_ok=True)

    # Process each chunk of the model
    for sub_array, (i, j, k) in split_array(array, max_size):
        output_file = f"{output_prefix}/{base_filename}_{i}_{j}_{k}.vox"
        value_mapping, palette, shape = numpy_to_vox(sub_array, color_map, output_file)
        print(f"Chunk {i}_{j}_{k} saved as {output_file}")
        print(f"Shape: {shape}")

    return value_mapping, palette

def export_magicavoxel_vox(array, output_dir, base_filename='chunk', voxel_color_map=None):
    """
    Export a voxel model to MagicaVoxel .vox format.
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data
        output_dir (str): Directory to save the .vox files
        base_filename (str): Base name for the output files
        voxel_color_map (dict, optional): Dictionary mapping indices to RGB color values.
            If None, uses default color map.
    """
    # Use default color map if none provided
    if voxel_color_map is None:
        voxel_color_map = get_default_voxel_color_map()
    
    # Convert color map and array to sequential indices
    converted_voxel_color_map, converted_array = convert_colormap_and_array(voxel_color_map, array)

    # Export the model and print confirmation
    value_mapping, palette = export_large_voxel_model(converted_array, converted_voxel_color_map, output_dir, base_filename=base_filename)
    print(f"\tvox files was successfully exported in {output_dir}")