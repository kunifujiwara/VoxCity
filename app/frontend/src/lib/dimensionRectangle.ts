export interface DimensionRectangleParams {
  centerLon: number;
  centerLat: number;
  widthM: number;
  heightM: number;
  rotationDeg?: number;
}

interface DimensionRectangleResponse {
  vertices?: number[][];
}

export async function fetchDimensionRectangle(
  params: DimensionRectangleParams,
  fetchImpl: typeof fetch = fetch,
): Promise<number[][]> {
  const res = await fetchImpl('/api/rectangle-from-dimensions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      center_lon: params.centerLon,
      center_lat: params.centerLat,
      width_m: params.widthM,
      height_m: params.heightM,
      rotation_deg: params.rotationDeg ?? 0,
    }),
  });
  const data = (await res.json()) as DimensionRectangleResponse;
  return data.vertices ?? [];
}