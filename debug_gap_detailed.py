"""Detailed debug script to understand the 83 columns with gap=1."""

import numpy as np
import pickle
import sys

# Load the voxcity data
data_source = sys.argv[1] if len(sys.argv) > 1 else "german"
filepath = f"output/lod2_test_{data_source}/voxcity.pkl"
print(f"Loading: {filepath}")

with open(filepath, "rb") as f:
    data = pickle.load(f)

voxcity_obj = data.get('voxcity')
voxel_obj = voxcity_obj.voxels
voxel_grid = voxel_obj.classes

print(f"Voxel grid shape: {voxel_grid.shape}")

nx, ny, nz = voxel_grid.shape

# Find all columns with gap=1
columns_with_gap_1 = []
columns_with_gap_0 = []

for i in range(nx):
    for j in range(ny):
        column = voxel_grid[i, j, :]
        
        building_indices = np.where(column == -3)[0]
        ground_indices = np.where(column == -1)[0]
        
        if len(building_indices) > 0 and len(ground_indices) > 0:
            max_ground = ground_indices.max()
            min_building = building_indices.min()
            gap = min_building - max_ground - 1
            
            if gap == 1:
                columns_with_gap_1.append((i, j, max_ground, min_building))
            elif gap == 0:
                columns_with_gap_0.append((i, j, max_ground, min_building))

print(f"\nColumns with gap=0: {len(columns_with_gap_0)}")
print(f"Columns with gap=1: {len(columns_with_gap_1)}")

# Analyze the gap=1 columns
print("\n=== Analyzing gap=1 columns ===")

# Check if they're clustered in certain areas
if columns_with_gap_1:
    i_coords = [c[0] for c in columns_with_gap_1]
    j_coords = [c[1] for c in columns_with_gap_1]
    max_ground_values = [c[2] for c in columns_with_gap_1]
    min_building_values = [c[3] for c in columns_with_gap_1]
    
    print(f"i range: {min(i_coords)} to {max(i_coords)}")
    print(f"j range: {min(j_coords)} to {max(j_coords)}")
    print(f"max_ground range: {min(max_ground_values)} to {max(max_ground_values)}")
    print(f"min_building range: {min(min_building_values)} to {max(min_building_values)}")

# Check if the gap=0 columns have different ground levels
if columns_with_gap_0:
    max_ground_0 = [c[2] for c in columns_with_gap_0]
    min_building_0 = [c[3] for c in columns_with_gap_0]
    
    print(f"\nFor gap=0 columns:")
    print(f"max_ground range: {min(max_ground_0)} to {max(max_ground_0)}")
    print(f"min_building range: {min(min_building_0)} to {max(min_building_0)}")

# Check what's in the "gap" voxel for gap=1 columns
print("\n=== Content of gap voxel ===")
gap_content_counts = {}
for i, j, max_ground, min_building in columns_with_gap_1:
    gap_k = max_ground + 1  # The voxel that should be building but isn't
    value = voxel_grid[i, j, gap_k]
    gap_content_counts[value] = gap_content_counts.get(value, 0) + 1

print(f"Values in gap voxel: {gap_content_counts}")

# Check if gap=1 columns are at building edges
print("\n=== Checking if gap=1 columns are at building edges ===")
edge_count = 0
interior_count = 0

for i, j, max_ground, min_building in columns_with_gap_1:
    # Check neighbors for building voxels at the gap level
    gap_k = max_ground + 1
    has_building_neighbor = False
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                if voxel_grid[ni, nj, gap_k] == -3:  # Building
                    has_building_neighbor = True
                    break
        if has_building_neighbor:
            break
    
    if has_building_neighbor:
        edge_count += 1
    else:
        interior_count += 1

print(f"Gap=1 columns with building neighbor at gap level: {edge_count}")
print(f"Gap=1 columns without building neighbor at gap level: {interior_count}")

# Compare with gap=0 columns at edges
print("\n=== Sample gap=1 columns ===")
for idx, (i, j, max_ground, min_building) in enumerate(columns_with_gap_1[:5]):
    print(f"\nColumn ({i}, {j}):")
    print(f"  max_ground = {max_ground}, min_building = {min_building}, gap = 1")
    
    # Show the values around the transition
    for k in range(max(0, max_ground-1), min(nz, min_building+3)):
        value = voxel_grid[i, j, k]
        label = 'ground' if value == -1 else 'building' if value == -3 else 'empty'
        print(f"    k={k}: {value} ({label})")
