/**
 * API client for the VoxCity FastAPI backend.
 */

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
) {
  return request<RectangleResult>('/rectangle-from-dimensions', {
    method: 'POST',
    body: JSON.stringify({
      center_lon: centerLon,
      center_lat: centerLat,
      width_m: widthM,
      height_m: heightM,
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

export async function getModelPreview() {
  return request<{ figure_json: string }>('/model/preview');
}

export async function getModelInfo() {
  return request<ModelInfo>('/model/info');
}
