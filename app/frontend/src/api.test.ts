import { describe, expect, it, vi } from 'vitest';
import { toZoneSpecDto } from './api';
import type { HorizontalZone, BuildingSurfaceZone } from './types/zones';

describe('toZoneSpecDto', () => {
  it('converts horizontal zone to DTO', () => {
    const zone: HorizontalZone = {
      id: 'z1',
      name: 'Zone 1',
      color: '#ff0000',
      type: 'horizontal',
      shape: 'polygon',
      ring_lonlat: [[1, 2], [3, 4], [5, 6]],
    };
    const dto = toZoneSpecDto(zone);
    expect(dto).toEqual({
      id: 'z1',
      name: 'Zone 1',
      type: 'horizontal',
      ring_lonlat: [[1, 2], [3, 4], [5, 6]],
      selectors: [],
      group_id: undefined,
    });
  });

  it('converts surface zone to DTO with snake_case selectors', () => {
    const zone: BuildingSurfaceZone = {
      id: 'z2',
      name: 'Roof',
      color: '#00ff00',
      type: 'building_surface',
      groupId: 'g1',
      selectors: [
        { buildingId: 42, mode: 'whole' },
        { buildingId: 42, mode: 'exclude_faces', faceKeys: ['f1', 'f2'] },
        { buildingId: 5, mode: 'wall_orientation', orientation: 'N' },
      ],
    };
    const dto = toZoneSpecDto(zone);
    expect(dto).toEqual({
      id: 'z2',
      name: 'Roof',
      type: 'building_surface',
      ring_lonlat: null,
      selectors: [
        { building_id: 42, mode: 'whole', orientation: null, face_keys: null },
        { building_id: 42, mode: 'exclude_faces', orientation: null, face_keys: ['f1', 'f2'] },
        { building_id: 5, mode: 'wall_orientation', orientation: 'N', face_keys: null },
      ],
      group_id: 'g1',
    });
  });
});

describe('saveSession / loadSession', () => {
  it('saveSession posts to /api/session/save with include_sim_results query and frontend_state form-field', async () => {
    const fakeBlob = new Blob([new Uint8Array([1, 2, 3])], { type: 'application/zip' });
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      expect(String(input)).toBe('/api/session/save?include_sim_results=1');
      expect(init?.method).toBe('POST');
      const body = init?.body as FormData;
      expect(body).toBeInstanceOf(FormData);
      expect(body.get('frontend_state')).toBe('{"zones":[]}');
      return new Response(fakeBlob, { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);

    const { saveSession } = await import('./api');
    const blob = await saveSession('{"zones":[]}', true);
    expect(blob.size).toBeGreaterThan(0);
    expect(fetchMock).toHaveBeenCalledOnce();

    vi.unstubAllGlobals();
  });

  it('loadSession posts the file as multipart and parses the summary', async () => {
    const summary = {
      has_voxcity: true,
      rectangle_vertices: [[1, 2]],
      land_cover_source: 'OpenStreetMap',
      frontend_state: '{"zones":[]}',
    };
    const fetchMock = vi.fn(async (input: RequestInfo, init?: RequestInit) => {
      expect(String(input)).toBe('/api/session/load');
      expect(init?.method).toBe('POST');
      const body = init?.body as FormData;
      expect(body.get('file')).toBeInstanceOf(File);
      return new Response(JSON.stringify(summary), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const { loadSession } = await import('./api');
    const file = new File([new Uint8Array([1, 2, 3])], 'session.zip', { type: 'application/zip' });
    const got = await loadSession(file);
    expect(got).toEqual(summary);

    vi.unstubAllGlobals();
  });

  it('loadSession throws Error with detail message on non-2xx', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'bad zip' }), {
        status: 400,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const { loadSession } = await import('./api');
    const file = new File([new Uint8Array([1])], 'session.zip');
    await expect(loadSession(file)).rejects.toThrow('bad zip');

    vi.unstubAllGlobals();
  });
});
