/** Shared constants used across multiple tabs */

export const COLORMAPS = [
  'magma', 'viridis', 'plasma', 'inferno', 'cividis', 'turbo',
  'Greens', 'Blues', 'BuPu_r', 'coolwarm', 'RdYlBu_r', 'Spectral', 'gray',
] as const;

export const BUILDING_SOURCES = [
  'OpenStreetMap',
  'Microsoft Building Footprints',
  'Open Building 2.5D Temporal',
  'EUBUCCO v0.1',
  'Overture',
  'GBA',
] as const;

export const BUILDING_COMPLEMENTARY_SOURCES = [
  'None',
  'OpenStreetMap',
  'Microsoft Building Footprints',
  'Open Building 2.5D Temporal',
  'EUBUCCO v0.1',
  'Overture',
  'GBA',
  'England 1m DSM - DTM',
  'Netherlands 0.5m DSM - DTM',
] as const;

export const LAND_COVER_SOURCES = [
  'OpenStreetMap',
  'OpenEarthMapJapan',
  'Urbanwatch',
  'ESA WorldCover',
  'ESRI 10m Annual Land Cover',
  'Dynamic World V1',
] as const;

export const CANOPY_HEIGHT_SOURCES = [
  'Static',
  'OpenStreetMap',
  'High Resolution 1m Global Canopy Height Maps',
  'ETH Global Sentinel-2 10m Canopy Height (2020)',
] as const;

export const DEM_SOURCES = [
  'Flat',
  'FABDEM',
  'DeltaDTM',
  'USGS 3DEP 1m',
  'England 1m DTM',
  'DEM France 1m',
  'DEM France 5m',
  'AUSTRALIA 5M DEM',
  'Netherlands 0.5m DTM',
] as const;

export const CUSTOM_CLASSES = [
  { id: -3, label: 'Building' },
  { id: -2, label: 'Tree' },
  { id: 1, label: 'Bareland' },
  { id: 2, label: 'Rangeland' },
  { id: 3, label: 'Shrub' },
  { id: 4, label: 'Agriculture land' },
  { id: 6, label: 'Moss and lichen' },
  { id: 7, label: 'Wet land' },
  { id: 8, label: 'Mangrove' },
  { id: 9, label: 'Water' },
  { id: 10, label: 'Snow and ice' },
  { id: 11, label: 'Developed space' },
  { id: 12, label: 'Road' },
] as const;

/** Voxel element classes available for visibility toggling in simulation tabs */
export const VOXEL_CLASSES = [
  { id: -3, label: 'Buildings' },
  { id: -2, label: 'Trees (canopy)' },
  { id: -1, label: 'Underground' },
  { id: -30, label: 'Landmark' },
  { id: 1, label: 'Bareland' },
  { id: 2, label: 'Rangeland' },
  { id: 3, label: 'Shrub' },
  { id: 4, label: 'Agriculture land' },
  { id: 5, label: 'Tree (ground cover)' },
  { id: 6, label: 'Moss and lichen' },
  { id: 7, label: 'Wet land' },
  { id: 8, label: 'Mangrove' },
  { id: 9, label: 'Water' },
  { id: 10, label: 'Snow and ice' },
  { id: 11, label: 'Developed space' },
  { id: 12, label: 'Road' },
  { id: 13, label: 'Building (ground)' },
] as const;
