"""
Module for handling MagicaVoxel .vox files.

This module provides functionality for converting 3D numpy arrays to MagicaVoxel .vox files,
including color mapping and splitting large models into smaller chunks.

The module handles:
- Color map conversion and optimization
- Large model splitting into MagicaVoxel-compatible chunks
- Custom palette creation
- Coordinate system transformation
- Batch export of multiple .vox files

Key Features:
- Supports models larger than MagicaVoxel's 256³ size limit
- Automatic color palette optimization
- Preserves color mapping across chunks
- Handles coordinate system differences between numpy and MagicaVoxel
"""

# Required imports for voxel file handling and array manipulation
import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter
import os
from ..visualizer import get_voxel_color_map

def convert_colormap_and_array(original_map, original_array):
    """
    Convert a color map with arbitrary indices to sequential indices starting from 0
    and update the corresponding 3D numpy array.
    
    This function optimizes the color mapping by:
    1. Converting arbitrary color indices to sequential ones
    2. Creating a new mapping that preserves color relationships
    3. Updating the voxel array to use the new sequential indices
    
    Args:
        original_map (dict): Dictionary with integer keys and RGB color value lists.
            Each key is a color index, and each value is a list of [R,G,B] values.
        original_array (numpy.ndarray): 3D array with integer values corresponding to color map keys.
            The array contains indices that match the keys in original_map.
        
    Returns:
        tuple: (new_color_map, new_array)
            - new_color_map (dict): Color map with sequential indices starting from 0
            - new_array (numpy.ndarray): Updated array with new sequential indices
    
    Example:
        >>> color_map = {5: [255,0,0], 10: [0,255,0]}
        >>> array = np.array([[[5,10],[10,5]]])
        >>> new_map, new_array = convert_colormap_and_array(color_map, array)
        >>> print(new_map)
        {0: [255,0,0], 1: [0,255,0]}
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
    Create a palette array from a color map dictionary suitable for MagicaVoxel format.
    
    This function:
    1. Creates a 256x4 RGBA palette array
    2. Sets full opacity (alpha=255) for all colors by default
    3. Reserves index 0 for transparent black (void)
    4. Maps colors sequentially starting from index 1
    
    Args:
        color_map (dict): Dictionary mapping indices to RGB color values.
            Each value should be a list of 3 integers [R,G,B] in range 0-255.
        
    Returns:
        numpy.ndarray: 256x4 array containing RGBA color values.
            - Shape: (256, 4)
            - Type: uint8
            - Format: [R,G,B,A] for each color
            - Index 0: [0,0,0,0] (transparent)
            - Indices 1-255: Colors from color_map with alpha=255
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
    Create a mapping from color map keys to sequential indices for MagicaVoxel compatibility.
    
    Creates a mapping that:
    - Reserves index 0 for void space
    - Reserves index 1 (typically for special use)
    - Maps colors sequentially starting from index 2
    
    Args:
        color_map (dict): Dictionary mapping indices to RGB color values.
            The keys can be any integers, they will be remapped sequentially.
        
    Returns:
        dict: Mapping from original indices to sequential indices starting at 2.
            Example: {original_index1: 2, original_index2: 3, ...}
    """
    # Create mapping starting at index 2 (0 is void, 1 is reserved)
    return {value: i+2 for i, value in enumerate(color_map.keys())}

def split_array(array, max_size=255):
    """
    Split a 3D array into smaller chunks that fit within MagicaVoxel size limits.
    
    This function handles large voxel models by:
    1. Calculating required splits in each dimension
    2. Dividing the model into chunks of max_size or smaller
    3. Yielding each chunk with its position information
    
    Args:
        array (numpy.ndarray): 3D array to split.
            Can be any size, will be split into chunks of max_size or smaller.
        max_size (int, optional): Maximum size allowed for each dimension.
            Defaults to 255 (MagicaVoxel's limit is 256).
        
    Yields:
        tuple: (sub_array, (i,j,k))
            - sub_array: numpy.ndarray of size <= max_size in each dimension
            - (i,j,k): tuple of indices indicating chunk position in the original model
    
    Example:
        >>> array = np.ones((300, 300, 300))
        >>> for chunk, (i,j,k) in split_array(array):
        ...     print(f"Chunk at position {i},{j},{k} has shape {chunk.shape}")
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
    
    This function handles the complete conversion process:
    1. Creates a custom color palette from the color map
    2. Generates value mapping for voxel indices
    3. Transforms coordinates to match MagicaVoxel's system
    4. Saves the model in .vox format
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data.
            Values should correspond to keys in color_map.
        color_map (dict): Dictionary mapping indices to RGB color values.
            Each value should be a list of [R,G,B] values (0-255).
        output_file (str): Path to save the .vox file.
            Will overwrite if file exists.
        
    Returns:
        tuple: (value_mapping, palette, shape)
            - value_mapping: dict mapping original indices to MagicaVoxel indices
            - palette: numpy.ndarray of shape (256,4) containing RGBA values
            - shape: tuple of (width, height, depth) of the output model
    
    Note:
        - Coordinates are transformed to match MagicaVoxel's coordinate system
        - Z-axis is flipped and axes are reordered in the process
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
    
    This function handles models of any size by:
    1. Creating the output directory if needed
    2. Splitting the model into manageable chunks
    3. Saving each chunk as a separate .vox file
    4. Maintaining consistent color mapping across all chunks
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data.
            Can be any size, will be split into chunks if needed.
        color_map (dict): Dictionary mapping indices to RGB color values.
            Each value should be a list of [R,G,B] values (0-255).
        output_prefix (str): Directory to save the .vox files.
            Will be created if it doesn't exist.
        max_size (int, optional): Maximum size allowed for each dimension.
            Defaults to 255 (MagicaVoxel's limit is 256).
        base_filename (str, optional): Base name for the output files.
            Defaults to 'chunk'. Final filenames will be {base_filename}_{i}_{j}_{k}.vox
        
    Returns:
        tuple: (value_mapping, palette)
            - value_mapping: dict mapping original indices to MagicaVoxel indices
            - palette: numpy.ndarray of shape (256,4) containing RGBA values
    
    Example:
        >>> array = np.ones((500,500,500))
        >>> color_map = {1: [255,0,0]}
        >>> export_large_voxel_model(array, color_map, "output/model")
        # Creates files like: output/model/chunk_0_0_0.vox, chunk_0_0_1.vox, etc.
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
    
    This is the main entry point for voxel model export. It handles:
    1. Color map management (using default if none provided)
    2. Color index optimization
    3. Large model splitting and export
    4. Progress reporting
    
    Args:
        array (numpy.ndarray): 3D array containing voxel data.
            Values should correspond to keys in voxel_color_map.
        output_dir (str): Directory to save the .vox files.
            Will be created if it doesn't exist.
        base_filename (str, optional): Base name for the output files.
            Defaults to 'chunk'. Used when model is split into multiple files.
        voxel_color_map (dict, optional): Dictionary mapping indices to RGB color values.
            If None, uses default color map from visualizer.
            Each value should be a list of [R,G,B] values (0-255).
    
    Note:
        - Large models are automatically split into multiple files
        - Color mapping is optimized and made sequential
        - Progress information is printed to stdout
    """
    # Use default color map if none provided
    if voxel_color_map is None:
        voxel_color_map = get_voxel_color_map()
    
    # Convert color map and array to sequential indices
    converted_voxel_color_map, converted_array = convert_colormap_and_array(voxel_color_map, array)

    # Export the model and print confirmation
    value_mapping, palette = export_large_voxel_model(converted_array, converted_voxel_color_map, output_dir, base_filename=base_filename)
    print(f"\tvox files was successfully exported in {output_dir}")


class MagicaVoxelExporter:
    """Exporter adapter to write VoxCity voxels as MagicaVoxel .vox chunks.

    Accepts either a VoxCity instance (uses `voxels.classes`) or a raw 3D numpy array.
    """

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        import numpy as _np
        os.makedirs(output_directory, exist_ok=True)
        try:
            from ..models import VoxCity as _VoxCity
            if isinstance(obj, _VoxCity):
                export_magicavoxel_vox(
                    obj.voxels.classes,
                    output_directory,
                    base_filename,
                    voxel_color_map=kwargs.get("voxel_color_map"),
                )
                return output_directory
        except Exception:
            pass

        if isinstance(obj, _np.ndarray) and obj.ndim == 3:
            export_magicavoxel_vox(obj, output_directory, base_filename, voxel_color_map=kwargs.get("voxel_color_map"))
            return output_directory

        raise TypeError("MagicaVoxelExporter.export expects VoxCity or a 3D numpy array")