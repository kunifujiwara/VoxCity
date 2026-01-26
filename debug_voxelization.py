"""Debug script to understand voxelization issue."""

import numpy as np
import pickle

# Load the voxcity data
import sys
data_source = sys.argv[1] if len(sys.argv) > 1 else "german"
filepath = f"output/lod2_test_{data_source}/voxcity.pkl"
print(f"Loading: {filepath}")

with open(filepath, "rb") as f:
    data = pickle.load(f)

# Check what type of object we have
print(f"Data type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys: {data.keys()}")
    voxcity_obj = data.get('voxcity')
    print(f"VoxCity type: {type(voxcity_obj)}")
    voxel_obj = voxcity_obj.voxels
    print(f"Voxels type: {type(voxel_obj)}")
    print(f"Voxels.classes type: {type(voxel_obj.classes)}")
    voxel_grid = voxel_obj.classes
else:
    voxel_grid = data.voxels if hasattr(data, 'voxels') else data
print(f"Voxel grid shape: {voxel_grid.shape}")
print(f"Unique values: {np.unique(voxel_grid)}")

# Find where buildings and ground are
building_mask = voxel_grid == -3
ground_mask = voxel_grid == -1

print(f"\nBuilding voxels: {np.sum(building_mask)}")
print(f"Ground voxels: {np.sum(ground_mask)}")

# For each column (i, j), find the lowest and highest building voxel
# and the highest ground voxel
print("\n=== Analyzing building/ground vertical distribution ===")

nx, ny, nz = voxel_grid.shape
building_lowest_k = []
building_highest_k = []
ground_highest_k = []
gap_sizes = []

for i in range(nx):
    for j in range(ny):
        column = voxel_grid[i, j, :]
        
        building_indices = np.where(column == -3)[0]
        ground_indices = np.where(column == -1)[0]
        
        if len(building_indices) > 0:
            building_lowest_k.append(building_indices.min())
            building_highest_k.append(building_indices.max())
            
            if len(ground_indices) > 0:
                max_ground = ground_indices.max()
                min_building = building_indices.min()
                gap = min_building - max_ground - 1  # -1 because they should be adjacent
                gap_sizes.append(gap)
                ground_highest_k.append(max_ground)

if building_lowest_k:
    print(f"\nBuilding lowest k: min={min(building_lowest_k)}, max={max(building_lowest_k)}, mean={np.mean(building_lowest_k):.1f}")
    print(f"Building highest k: min={min(building_highest_k)}, max={max(building_highest_k)}, mean={np.mean(building_highest_k):.1f}")
    
if ground_highest_k:
    print(f"Ground highest k: min={min(ground_highest_k)}, max={max(ground_highest_k)}, mean={np.mean(ground_highest_k):.1f}")

if gap_sizes:
    print(f"\nGap between ground and building: min={min(gap_sizes)}, max={max(gap_sizes)}, mean={np.mean(gap_sizes):.1f}")
    print(f"Gap distribution: 0={gap_sizes.count(0)}, 1={gap_sizes.count(1)}, 2={gap_sizes.count(2)}, >2={sum(1 for g in gap_sizes if g > 2)}")
    print(f"Negative gaps (building inside ground): {sum(1 for g in gap_sizes if g < 0)}")

# Show a sample column with building
print("\n=== Sample column with building ===")
sample_found = False
for i in range(nx):
    for j in range(ny):
        column = voxel_grid[i, j, :]
        building_indices = np.where(column == -3)[0]
        ground_indices = np.where(column == -1)[0]
        
        if len(building_indices) > 0 and len(ground_indices) > 0:
            print(f"Column ({i}, {j}):")
            print(f"  Ground at k: {list(ground_indices[:5])}... to {ground_indices.max()}")
            print(f"  Building at k: {building_indices.min()} to {building_indices.max()}")
            print(f"  Gap: {building_indices.min() - ground_indices.max() - 1}")
            
            # Show the values around the transition
            transition = ground_indices.max()
            print(f"  Values around transition (k={transition-1} to k={transition+3}):")
            for k in range(max(0, transition-1), min(nz, transition+4)):
                print(f"    k={k}: {column[k]} ({'ground' if column[k]==-1 else 'building' if column[k]==-3 else 'empty'})")
            sample_found = True
            break
    if sample_found:
        break
