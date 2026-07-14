# Land cover classes

VoxCity normalizes land cover from every supported source into a single set of
standard classes. Each cell in the land cover grid — and each corresponding
voxel — is assigned one of the following integer class indices.

| Index | Class | Index | Class |
|:-----:|-------|:-----:|-------|
| 1 | Bareland | 8 | Mangrove |
| 2 | Rangeland | 9 | Water |
| 3 | Shrub | 10 | Snow and ice |
| 4 | Agriculture land | 11 | Developed space |
| 5 | Tree | 12 | Road |
| 6 | Moss and lichen | 13 | Building |
| 7 | Wet land | 14 | No Data |

The mapping from each source's native classes to these standard classes is
applied automatically during model generation. For the list of land cover
datasets and their coverage, see the {doc}`Data sources reference <data_sources>`.
