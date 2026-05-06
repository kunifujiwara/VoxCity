import { describe, expect, it } from 'vitest';
import { fetchDimensionRectangle } from './dimensionRectangle';

describe('fetchDimensionRectangle', () => {
  it('requests a fixed-dimension rectangle with the current rotation angle', async () => {
    const vertices = [
      [139.0, 35.0],
      [139.0, 35.1],
      [139.1, 35.1],
      [139.1, 35.0],
    ];
    const calls: Array<[string, RequestInit | undefined]> = [];
    const fetchMock = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(input), init]);
      return { json: async () => ({ vertices }) } as Response;
    }) as typeof fetch;

    const result = await fetchDimensionRectangle(
      {
        centerLon: 139.767125,
        centerLat: 35.681236,
        widthM: 1250,
        heightM: 900,
        rotationDeg: 37,
      },
      fetchMock,
    );

    expect(result).toEqual(vertices);
    expect(calls).toHaveLength(1);
    expect(calls[0][0]).toBe('/api/rectangle-from-dimensions');
    expect(calls[0][1]?.method).toBe('POST');
    expect(calls[0][1]?.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(calls[0][1]?.body as string)).toEqual({
      center_lon: 139.767125,
      center_lat: 35.681236,
      width_m: 1250,
      height_m: 900,
      rotation_deg: 37,
    });
  });

  it('defaults rotation to zero for backward-compatible dimension requests', async () => {
    const calls: Array<[string, RequestInit | undefined]> = [];
    const fetchMock = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(input), init]);
      return { json: async () => ({ vertices: [] }) } as Response;
    }) as typeof fetch;

    await fetchDimensionRectangle(
      {
        centerLon: 139.767125,
        centerLat: 35.681236,
        widthM: 1250,
        heightM: 900,
      },
      fetchMock,
    );

    expect(JSON.parse(calls[0][1]?.body as string).rotation_deg).toBe(0);
  });
});