/**
 * API client for the VoxCity FastAPI backend.
 */

import type { Zone, SurfaceSelector } from './types/zones';

const BASE = '/api';

/** Delay helper */
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Fetch wrapper with automatic retry for connection errors (backend may still
 * be booting when the frontend starts).
 */
async function request<T>(
  path: string,
  options?: RequestInit,
  retries = 3,
  delayMs = 2000,
): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(`${BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      return res.json();
    } catch (err: any) {
      // Retry on network-level errors (backend not ready yet)
      const isNetworkError =
        err instanceof TypeError || // fetch throws TypeError on network failure
        err?.message?.includes('Failed to fetch') ||
        err?.message?.includes('ECONNREFUSED');

      if (isNetworkError && attempt < retries) {
        console.warn(
          `[api] ${path} – connection failed, retrying in ${delayMs}ms (${attempt + 1}/${retries})`,
        );
        await sleep(delayMs);
        continue;
      }
      throw err;
    }
  }
}

// ── Types ────────────────────────────────────────────────────
export interface GeocodeResult {
  lat: number;
  lon: number;
  bbox: [number, number, number, number] | null;
}

export interface RectangleResult {
  vertices: number[][];
}

export interface GenerateResult {
  status: string;
  grid_shape: number[];
  meshsize: number;
  figure_json: string;
  preview_disabled?: boolean;
}

export interface SimulationResult {
  status: string;
  figure_json: string;
}

export interface ModelInfo {
  grid_shape: number[];
  meshsize: number;
  n_buildings: number;
  rectangle_vertices: number[][] | null;
  land_cover_source: string;
  preview_disabled?: boolean;
}

export interface AutoDetectResult {
  building_source: string;
  building_complementary_source: string;
  land_cover_source: string;
  canopy_height_source: string;
  dem_source: string;
}

// ── API functions ────────────────────────────────────────────

export async function healthCheck() {
  return request<{ status: string; has_model: boolean }>('/health');
}

export async function resetSession() {
  return request<{ status: string }>('/reset', { method: 'POST' });
}

export async function autoDetectSources(rectangleVertices: number[][]) {
  return request<AutoDetectResult>('/auto-detect-sources', {
    method: 'POST',
    body: JSON.stringify({ rectangle_vertices: rectangleVertices }),
  });
}

export async function geocodeCity(cityName: string) {
  return request<GeocodeResult>('/geocode', {
    method: 'POST',
    body: JSON.stringify({ city_name: cityName }),
  });
}

export async function rectangleFromDimensions(
  centerLon: number,
  centerLat: number,
  widthM: number,
  heightM: number,
  rotationDeg: number = 0,
) {
  return request<RectangleResult>('/rectangle-from-dimensions', {
    method: 'POST',
    body: JSON.stringify({
      center_lon: centerLon,
      center_lat: centerLat,
      width_m: widthM,
      height_m: heightM,
      rotation_deg: rotationDeg,
    }),
  });
}

export async function generateModel(params: {
  rectangle_vertices: number[][];
  meshsize: number;
  mode: string;
  building_source?: string | null;
  land_cover_source?: string | null;
  canopy_height_source?: string | null;
  dem_source?: string | null;
  building_complementary_source?: string | null;
  building_complement_height?: number;
  static_tree_height?: number;
  overlapping_footprint?: string;
  dem_interpolation?: boolean;
  use_citygml_cache?: boolean;
  use_ndsm_canopy?: boolean;
}) {
  return request<GenerateResult>('/generate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function runSolar(params: {
  calc_type: string;
  analysis_target: string;
  calc_time?: string;
  start_time?: string;
  end_time?: string;
  epw_source?: string;
  view_point_height?: number;
  include_building_roofs?: boolean;
  colormap?: string;
  vmin?: number | null;
  vmax?: number | null;
  hidden_classes?: number[];
}) {
  return request<SimulationResult>('/solar', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function runView(params: {
  view_type: string;
  analysis_target: string;
  view_point_height?: number;
  include_building_roofs?: boolean;
  custom_classes?: number[];
  inclusion_mode?: boolean;
  n_azimuth?: number;
  n_elevation?: number;
  elevation_min_degrees?: number;
  elevation_max_degrees?: number;
  colormap?: string;
  vmin?: number | null;
  vmax?: number | null;
  hidden_classes?: number[];
  export_obj?: boolean;
}) {
  return request<SimulationResult>('/view', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function runLandmark(params: {
  analysis_target: string;
  landmark_ids?: number[];
  view_point_height?: number;
  include_building_roofs?: boolean;
  n_azimuth?: number;
  n_elevation?: number;
  elevation_min_degrees?: number;
  elevation_max_degrees?: number;
  colormap?: string;
  vmin?: number | null;
  vmax?: number | null;
  hidden_classes?: number[];
}) {
  return request<SimulationResult>('/landmark', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function exportCityles(params: {
  building_material?: string;
  tree_type?: string;
  trunk_height_ratio?: number;
}) {
  const res = await fetch(`${BASE}/export/cityles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

export async function exportObj(params: {
  filename?: string;
  export_netcdf?: boolean;
}) {
  const res = await fetch(`${BASE}/export/obj`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

export async function exportGeotiff(params: { filename?: string }) {
  const res = await fetch(`${BASE}/export/geotiff`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

export async function getModelPreview() {
  return request<{ figure_json: string }>('/model/preview');
}

export async function getModelInfo() {
  return request<ModelInfo>('/model/info');
}

export async function rerenderSimulation(params: {
  colormap: string;
  vmin: number | null;
  vmax: number | null;
  hidden_classes: number[];
}) {
  return request<SimulationResult>('/rerender', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export interface BuildingInfo {
  id: number;
  cx: number;
  cy: number;
  cz: number;
  top_z: number;
}

export interface BuildingsListResult {
  buildings: BuildingInfo[];
}

export async function getBuildingsList() {
  return request<BuildingsListResult>('/buildings/list');
}

export async function getBuildingAt(x: number, y: number) {
  const params = new URLSearchParams({ x: String(x), y: String(y) });
  return request<{ building_id: number | null }>(`/buildings/at?${params.toString()}`);
}

export interface BuildingHighlightResult {
  chunks: MeshChunkDto[];
}

export async function getBuildingHighlight(
  ids: number[],
  options?: { colormap?: string; emissive?: boolean },
) {
  const params = new URLSearchParams();
  if (ids.length) params.set('ids', ids.join(','));
  if (options?.colormap) params.set('colormap', options.colormap);
  if (options?.emissive) params.set('emissive', 'true');
  const q = params.toString();
  return request<BuildingHighlightResult>(`/buildings/highlight${q ? `?${q}` : ''}`);
}

export interface LandmarkPreviewResult {
  figure_json: string;
}

export async function getLandmarkPreview() {
  return request<LandmarkPreviewResult>('/landmark/preview');
}

// ── Edit tab ──────────────────────────────────────────────────

export interface ModelGeoResult {
  rectangle_vertices: [number, number][];
  meshsize_m: number;
  grid_shape: [number, number];
  center: [number, number];
  grid_geom: {
    origin: [number, number];
    side_1: [number, number];
    side_2: [number, number];
    u_vec: [number, number];
    v_vec: [number, number];
    adj_mesh: [number, number];
    grid_size: [number, number];
  };
  land_cover_source: string | null;
  building_geojson: any;
  canopy_geojson: any;
  land_cover_geojson: any;
}

export async function getModelGeo() {
  return request<ModelGeoResult>('/model/geo');
}

export interface AnchorGroundResult {
  dem_elevation: number; // absolute DEM elevation at the anchor cell (metres)
  dem_min: number;       // DEM grid minimum (metres)
  meshsize_m: number;    // voxel size (metres)
}

/**
 * Vertical datum info for seating an imported OBJ at a lon/lat anchor. Used by
 * the Import tab's 3D preview so `move_up = 0` lands the building on the ground
 * at the same height the commit transform uses: scene-Z `(dem_elevation -
 * dem_min) + meshsize_m`.
 */
export async function getAnchorGround(lon: number, lat: number) {
  return request<AnchorGroundResult>(
    `/model/anchor_ground?lon=${encodeURIComponent(lon)}&lat=${encodeURIComponent(lat)}`,
  );
}

export interface LandCoverClass {
  index: number;
  name: string;
  color: string;
  editable: boolean;
}

export interface LandCoverClassesResult {
  classes: LandCoverClass[];
}

export async function listLandCoverClasses() {
  return request<LandCoverClassesResult>('/land-cover/classes');
}

export type PendingEditDto =
  | { kind: 'add_building';   cells: [number, number][]; height_m: number; min_height_m: number; ring?: [number, number][] }
  | { kind: 'delete_building'; building_ids: number[] }
  | { kind: 'set_building_height'; building_ids: number[]; height_m: number; min_height_m?: number }
  | { kind: 'add_trees';      cells: [number, number][]; height_m: number; bottom_m: number; tops?: number[]; bottoms?: number[] }
  | { kind: 'delete_trees';   cells: [number, number][] }
  | { kind: 'paint_lc';       cells: [number, number][]; class_index: number };

export interface ApplyEditsResult {
  figure_json: string;
  n_edits: number;
  n_changed_total: number;
  building_ids: number[];
}

export async function applyEdits(edits: PendingEditDto[]) {
  return request<ApplyEditsResult>('/model/apply_edits', {
    method: 'POST',
    body: JSON.stringify({ edits }),
  });
}

// ── Zones ────────────────────────────────────────────────────
export interface ZoneStat {
  zone_id: string;
  cell_count: number;
  valid_count: number;
  mean: number | null;
  min:  number | null;
  max:  number | null;
  std:  number | null;
}

export interface ZoneStatsResponse {
  target: 'ground' | 'building';
  sim_type: 'solar' | 'view' | 'landmark' | null;
  unit_label: string | null;
  stats: ZoneStat[];
}

/** Backend shape for a surface selector (snake_case). */
export interface SurfaceSelectorDto {
  building_id: number;
  mode: string;
  orientation: string | null;
  face_keys: string[] | null;
}

/** Backend DTO for a zone spec (snake_case). */
export interface ZoneSpecDto {
  id: string;
  name: string;
  type: 'horizontal' | 'building_surface';
  ring_lonlat: [number, number][] | null;
  selectors: SurfaceSelectorDto[];
  group_id: string | undefined;
}

/**
 * Convert a frontend Zone (camelCase) to the backend ZoneSpecDto (snake_case).
 */
export function toZoneSpecDto(zone: Zone): ZoneSpecDto {
  if (zone.type === 'horizontal') {
    return {
      id: zone.id,
      name: zone.name,
      type: 'horizontal',
      ring_lonlat: zone.ring_lonlat,
      selectors: [],
      group_id: zone.groupId,
    };
  }
  // building_surface
  const selectors: SurfaceSelectorDto[] = zone.selectors.map((s: SurfaceSelector) => ({
    building_id: s.buildingId,
    mode: s.mode,
    orientation: 'orientation' in s ? (s.orientation ?? null) : null,
    face_keys: 'faceKeys' in s ? (s.faceKeys ?? null) : null,
  }));
  return {
    id: zone.id,
    name: zone.name,
    type: 'building_surface',
    ring_lonlat: null,
    selectors,
    group_id: zone.groupId,
  };
}

export async function getZoneStats(zones: Zone[], simType?: string) {
  return request<ZoneStatsResponse>('/zones/stats', {
    method: 'POST',
    body: JSON.stringify({ zones: zones.map(toZoneSpecDto), sim_type: simType }),
  });
}

export type SurfaceZoneEdgeSegmentDto = [number, number, number, number, number, number];

export interface SurfaceZoneEdgePayloadDto {
  id: string;
  segments: SurfaceZoneEdgeSegmentDto[];
}

export interface SurfaceZoneEdgesResponse {
  zones: SurfaceZoneEdgePayloadDto[];
}

export async function getSurfaceZoneEdges(zones: Zone[], signal?: AbortSignal) {
  return request<SurfaceZoneEdgesResponse>('/buildings/surface-zone-edges', {
    method: 'POST',
    signal,
    body: JSON.stringify({ zones: zones.map(toZoneSpecDto) }),
  });
}

// ── Three.js raw geometry (R3F migration) ──────────────────────

export interface MeshChunkDto {
  name: string;
  positions: number[];          // flat XYZ
  indices:   number[];          // flat tris
  color?:    [number, number, number] | null;
  colors?:   number[] | null;   // per-vertex RGB
  opacity:   number;
  flat_shading: boolean;
  metadata:  Record<string, any>;
}

export interface SceneGeometryResponse {
  chunks:        MeshChunkDto[];
  bbox_min:      [number, number, number];
  bbox_max:      [number, number, number];
  meshsize_m:    number;
  ground_top_m?: number;
}

export interface OverlayGeometryResponse {
  target:    'ground' | 'building';
  sim_type:  'solar' | 'view' | 'landmark';
  chunk:     MeshChunkDto;
  face_to_cell?:     [number, number][] | null;
  face_to_building?: number[] | null;
  value_min: number;
  value_max: number;
  colormap:  string;
  unit_label: string;
}

export async function getSceneGeometry(downsample = 1, colorScheme = 'default') {
  const params = new URLSearchParams({
    downsample: String(downsample),
    color_scheme: colorScheme,
  });
  return request<SceneGeometryResponse>(`/scene/geometry?${params.toString()}`);
}

export async function getSimGeometry(
  kind: 'solar' | 'view' | 'landmark',
  body: { colormap?: string; vmin?: number | null; vmax?: number | null } = {},
) {
  return request<OverlayGeometryResponse>(`/sim/${kind}/geometry`, {
    method: 'POST',
    body: JSON.stringify({
      colormap: body.colormap ?? 'viridis',
      vmin: body.vmin ?? null,
      vmax: body.vmax ?? null,
    }),
  });
}

/** Response from GET /api/buildings/surfaces */
export interface SurfaceFaceMetaDto {
  face_key: string;
  building_id: number;
  surface_kind: string;
  orientation: string | null;
  is_window?: boolean;
}

export interface BuildingSurfacesResponse {
  chunk: MeshChunkDto;
  face_to_surface: SurfaceFaceMetaDto[];
  buildings: BuildingInfo[];
}

export async function getBuildingSurfaces() {
  return request<BuildingSurfacesResponse>('/buildings/surfaces');
}

// ── Session save / load ───────────────────────────────────────

export async function saveSession(
  frontendState?: string,
  includeSimResults: boolean = false,
): Promise<Blob> {
  const params = new URLSearchParams();
  if (includeSimResults) params.set('include_sim_results', '1');
  const query = params.toString() ? `?${params.toString()}` : '';
  const options: RequestInit = { method: 'POST' };
  if (frontendState !== undefined) {
    const form = new FormData();
    form.append('frontend_state', frontendState);
    options.body = form;
  }
  const res = await fetch(`${BASE}/session/save${query}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

export interface SessionLoadSummary {
  has_voxcity: boolean;
  rectangle_vertices: number[][] | null;
  land_cover_source: string;
  frontend_state: string | null;
  has_sim_results: boolean;
  last_sim_type: string | null;
  sim_result_types: string[];
  landmark_building_ids: number[];
}

export async function loadSession(file: File): Promise<SessionLoadSummary> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/session/load`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── OBJ import tab ────────────────────────────────────────────

export interface ImportObjGroupDto {
  name: string;
  role: string;
  n_faces: number;
  bbox_model: [number, number, number][]; // [min, max]
}

export interface ImportObjPreviewDto {
  footprints: [number, number][][];
  vertices: [number, number, number][];
  indices: [number, number, number][];
}

export interface ImportObjUploadResult {
  import_id: string;
  groups: ImportObjGroupDto[];
  model_bounds: [number, number, number][];
  preview: ImportObjPreviewDto;
}

export interface ImportPlacementDto {
  anchor_lonlat: [number, number];
  anchor_elevation: number | null;
  anchor_model_point: [number, number, number];
  rotation: number;
  move: [number, number, number];
  units: string;
  z_up: boolean;
  swap_yz: boolean;
}

export interface ImportObjCommitRequestDto {
  import_id: string;
  placement: ImportPlacementDto;
  roles: Record<string, string>;
  overwrite: boolean;
}

export interface ImportObjCommitResult {
  figure_json: string;
  imported_building_ids: number[];
  n_building_voxels_added: number;
  n_window_voxels_added: number;
  warning: string | null;
}

export async function uploadImportObj(
  file: File,
  sidecars: File[] = [],
): Promise<ImportObjUploadResult> {
  const form = new FormData();
  form.append('file', file);
  // Companion files (e.g. the .mtl + textures) ride along so the server can
  // resolve material names, which drive window auto-detection.
  for (const sc of sidecars) form.append('sidecars', sc);
  const res = await fetch(`${BASE}/model/import_obj/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function commitImportObj(req: ImportObjCommitRequestDto) {
  return request<ImportObjCommitResult>('/model/import_obj/commit', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

