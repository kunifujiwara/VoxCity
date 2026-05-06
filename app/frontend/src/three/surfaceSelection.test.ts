import { describe, expect, it } from 'vitest';
import type { MeshChunkDto, SurfaceFaceMetaDto } from '../api';
import type { SurfaceFaceMeta } from './types';
import {
  buildBoundaryLinePoints,
  buildSceneSurfaceSelectionSpec,
  buildZoneFillGeometry,
  emptySurfaceGeometryState,
  getSurfaceZones,
  isSurfaceFaceSelected,
  shouldFetchSurfaceSelection,
  shouldEnableZoningSurfaceSelection,
  shouldMountPickableSurface,
  surfaceLoadErrorResult,
  surfaceTriangleCount,
  toSurfaceFaceMetaArray,
  toSurfaceSelectionLayerSpecs,
} from './surfaceSelection';

const metas: SurfaceFaceMeta[] = [
  { faceKey: 'roof-1', buildingId: 1, surfaceKind: 'roof', orientation: null },
  { faceKey: 'wall-n', buildingId: 1, surfaceKind: 'wall', orientation: 'N' },
  { faceKey: 'wall-e', buildingId: 1, surfaceKind: 'wall', orientation: 'E' },
  { faceKey: 'roof-2', buildingId: 2, surfaceKind: 'roof', orientation: null },
];

describe('isSurfaceFaceSelected', () => {
  it('supports whole, roof, all_walls, wall_orientation, and faces selectors', () => {
    expect(isSurfaceFaceSelected(metas[0], [{ buildingId: 1, mode: 'whole' }])).toBe(true);
    expect(isSurfaceFaceSelected(metas[0], [{ buildingId: 1, mode: 'roof' }])).toBe(true);
    expect(isSurfaceFaceSelected(metas[1], [{ buildingId: 1, mode: 'all_walls' }])).toBe(true);
    expect(isSurfaceFaceSelected(metas[1], [{ buildingId: 1, mode: 'wall_orientation', orientation: 'N' }])).toBe(true);
    expect(isSurfaceFaceSelected(metas[2], [{ buildingId: 1, mode: 'wall_orientation', orientation: 'N' }])).toBe(false);
    expect(isSurfaceFaceSelected(metas[2], [{ buildingId: 1, mode: 'faces', faceKeys: ['wall-e'] }])).toBe(true);
  });

  it('lets exclude_faces override positive selectors', () => {
    expect(isSurfaceFaceSelected(metas[0], [
      { buildingId: 1, mode: 'whole' },
      { buildingId: 1, mode: 'exclude_faces', faceKeys: ['roof-1'] },
    ])).toBe(false);
  });

  it('ignores selectors for other buildings', () => {
    expect(isSurfaceFaceSelected(metas[0], [{ buildingId: 2, mode: 'whole' }])).toBe(false);
  });
});

describe('surface geometry validity', () => {
  it('counts indexed triangles when indices are present', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };

    expect(surfaceTriangleCount(chunk)).toBe(1);
  });

  it('counts non-indexed triangles when indices are absent', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };

    expect(surfaceTriangleCount(chunk)).toBe(1);
  });
});

describe('surface selection payload decisions', () => {
  const surfaceZone = {
    type: 'building_surface' as const,
    id: 'z1',
    name: 'Surface zone',
    color: '#ff0000',
    selectors: [{ buildingId: 1, mode: 'whole' as const }],
  };

  it('does not fetch when no model exists or the layer is disabled', () => {
    expect(shouldFetchSurfaceSelection({ hasModel: false, enabled: true, surfaceZoneCount: 1, requireSurfaceZones: true })).toBe(false);
    expect(shouldFetchSurfaceSelection({ hasModel: true, enabled: false, surfaceZoneCount: 1, requireSurfaceZones: true })).toBe(false);
  });

  it('does not fetch for simulation tabs with no building-surface zones', () => {
    expect(shouldFetchSurfaceSelection({ hasModel: true, enabled: true, surfaceZoneCount: 0, requireSurfaceZones: true })).toBe(false);
  });

  it('keeps Zoning tab surface highlights enabled outside building-surface edit mode when surface zones exist', () => {
    expect(shouldEnableZoningSurfaceSelection({ zoneType: 'horizontal', surfaceZoneCount: 1 })).toBe(true);
    expect(shouldEnableZoningSurfaceSelection({ zoneType: 'building_surface', surfaceZoneCount: 0 })).toBe(true);
    expect(shouldEnableZoningSurfaceSelection({ zoneType: 'horizontal', surfaceZoneCount: 0 })).toBe(false);
  });

  it('builds no enabled payload when geometry metadata length does not match triangle count', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };

    expect(buildSceneSurfaceSelectionSpec({
      surfaceChunk: chunk,
      faceToSurface: [],
      zones: [surfaceZone],
      enabled: true,
      displayMode: 'boundary',
      requireSelectors: true,
    })).toBeNull();
  });

  it('mounts the pickable surface only when picking and enabled geometry are present', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const spec = buildSceneSurfaceSelectionSpec({
      surfaceChunk: chunk,
      faceToSurface: [{ faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null }],
      zones: [surfaceZone],
      enabled: true,
      displayMode: 'boundary',
      requireSelectors: true,
    });

    expect(shouldMountPickableSurface(undefined, spec)).toBe(false);
    expect(shouldMountPickableSurface(() => {}, spec)).toBe(true);
    expect(shouldMountPickableSurface(() => {}, null)).toBe(false);
  });

  it('returns empty geometry and reports errors only for non-silent callers', () => {
    const reports: string[] = [];

    expect(emptySurfaceGeometryState()).toEqual({ surfaceChunk: null, faceToSurface: [] });

    expect(surfaceLoadErrorResult(new Error('boom'), true, (message) => reports.push(message))).toEqual({
      surfaceChunk: null,
      faceToSurface: [],
    });
    expect(reports).toEqual([]);

    expect(surfaceLoadErrorResult(new Error('boom'), false, (message) => reports.push(message))).toEqual({
      surfaceChunk: null,
      faceToSurface: [],
    });
    expect(reports).toEqual(['boom']);
  });

  it('returns null when disabled even if geometry is present', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    expect(buildSceneSurfaceSelectionSpec({
      surfaceChunk: chunk,
      faceToSurface: [{ faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null }],
      zones: [surfaceZone],
      enabled: false,
      displayMode: 'boundary',
      requireSelectors: true,
    })).toBeNull();
  });

  it('returns null when requireSelectors is true and zones have no selectors', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const emptySelectorsZone = {
      type: 'building_surface' as const,
      id: 'z2',
      name: 'Empty zone',
      color: '#0000ff',
      selectors: [] as typeof surfaceZone['selectors'],
    };
    expect(buildSceneSurfaceSelectionSpec({
      surfaceChunk: chunk,
      faceToSurface: [{ faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null }],
      zones: [emptySelectorsZone],
      enabled: true,
      displayMode: 'boundary',
      requireSelectors: true,
    })).toBeNull();
  });
});

describe('toSurfaceFaceMetaArray', () => {
  it('converts backend list metadata to frontend metadata', () => {
    const input: SurfaceFaceMetaDto[] = [
      { face_key: 'f1', building_id: 7, surface_kind: 'roof', orientation: null },
    ];

    expect(toSurfaceFaceMetaArray(input)).toEqual([
      { faceKey: 'f1', buildingId: 7, surfaceKind: 'roof', orientation: null },
    ]);
  });

  it('also accepts legacy record metadata defensively', () => {
    const input = {
      '0': { face_key: 'f1', building_id: 7, surface_kind: 'wall', orientation: 'W' },
    } satisfies Record<string, SurfaceFaceMetaDto>;

    expect(toSurfaceFaceMetaArray(input)).toEqual([
      { faceKey: 'f1', buildingId: 7, surfaceKind: 'wall', orientation: 'W' },
    ]);
  });
});

describe('buildBoundaryLinePoints', () => {
  it('omits the internal diagonal shared by selected adjacent triangles', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1, 0,
        0, 1, 0,
      ],
      indices: [0, 1, 2, 3, 4, 5],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const faceToSurface: SurfaceFaceMeta[] = [
      { faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null },
      { faceKey: 'b', buildingId: 1, surfaceKind: 'roof', orientation: null },
    ];

    const points = buildBoundaryLinePoints(chunk, faceToSurface, [{ buildingId: 1, mode: 'whole' }]);

    expect(points).toHaveLength(8);
    const edges = new Set<string>();
    for (let i = 0; i < points.length; i += 2) {
      const a = points[i].join(',');
      const b = points[i + 1].join(',');
      edges.add([a, b].sort().join('|'));
    }
    expect(edges).toEqual(new Set([
      '0,0,0|1,0,0',
      '1,0,0|1,1,0',
      '0,1,0|1,1,0',
      '0,0,0|0,1,0',
    ]));
  });

  it('returns no points when selectors match no faces', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };

    const points = buildBoundaryLinePoints(chunk, [metas[0]], [{ buildingId: 9, mode: 'whole' }]);

    expect(points).toEqual([]);
  });

  it('does not merge edges whose coordinates differ beyond the documented precision', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 0, 0,
        1, 1.00002, 0,
        0, 1, 0,
      ],
      indices: [0, 1, 2, 3, 4, 5],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const faceToSurface: SurfaceFaceMeta[] = [
      { faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null },
      { faceKey: 'b', buildingId: 1, surfaceKind: 'roof', orientation: null },
    ];

    const points = buildBoundaryLinePoints(chunk, faceToSurface, [{ buildingId: 1, mode: 'whole' }]);

    expect(points.length).toBeGreaterThan(8);
  });

  it('extracts only the perimeter for a larger selected grid', () => {
    const size = 20;
    const positions: number[] = [];
    const indices: number[] = [];
    const faceToSurface: SurfaceFaceMeta[] = [];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const start = positions.length / 3;
        positions.push(
          x, y, 0,
          x + 1, y, 0,
          x + 1, y + 1, 0,
          x, y, 0,
          x + 1, y + 1, 0,
          x, y + 1, 0,
        );
        indices.push(start, start + 1, start + 2, start + 3, start + 4, start + 5);
        faceToSurface.push(
          { faceKey: `${x}-${y}-a`, buildingId: 1, surfaceKind: 'roof', orientation: null },
          { faceKey: `${x}-${y}-b`, buildingId: 1, surfaceKind: 'roof', orientation: null },
        );
      }
    }
    const chunk: MeshChunkDto = { name: 'grid', positions, indices, opacity: 1, flat_shading: false, metadata: {} };

    const points = buildBoundaryLinePoints(chunk, faceToSurface, [{ buildingId: 1, mode: 'whole' }]);

    expect(points).toHaveLength(size * 4 * 2);
  });
});

describe('getSurfaceZones', () => {
  it('filters Zone[] to only building_surface zones', () => {
    const zones = [
      { type: 'horizontal' as const, id: 'h1', name: 'H', color: '#fff', shape: 'rect' as const, ring_lonlat: [] as [number, number][] },
      { type: 'building_surface' as const, id: 'b1', name: 'B', color: '#f00', selectors: [] },
    ];
    const result = getSurfaceZones(zones);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('b1');
  });
});

describe('buildZoneFillGeometry', () => {
  it('builds geometry with correct positions for a selected triangle', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const faceToSurface: SurfaceFaceMeta[] = [
      { faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null },
    ];

    const geometry = buildZoneFillGeometry(chunk, faceToSurface, [{ buildingId: 1, mode: 'whole' }]);

    expect(geometry).not.toBeNull();
    const positions = geometry!.getAttribute('position');
    expect(positions.count).toBe(3);
    expect(positions.getX(0)).toBeCloseTo(0);
    expect(positions.getY(0)).toBeCloseTo(0);
    expect(positions.getX(1)).toBeCloseTo(1);
    expect(positions.getX(2)).toBeCloseTo(0);
    expect(positions.getY(2)).toBeCloseTo(1);
  });

  it('returns null when no faces match the selectors', () => {
    const chunk: MeshChunkDto = {
      name: 'surface',
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
      indices: [0, 1, 2],
      opacity: 1,
      flat_shading: false,
      metadata: {},
    };
    const faceToSurface: SurfaceFaceMeta[] = [
      { faceKey: 'a', buildingId: 1, surfaceKind: 'roof', orientation: null },
    ];

    const geometry = buildZoneFillGeometry(chunk, faceToSurface, [{ buildingId: 9, mode: 'whole' }]);

    expect(geometry).toBeNull();
  });
});

describe('toSurfaceSelectionLayerSpecs', () => {
  it('marks a zone active by groupId when present', () => {
    const zones = [
      { type: 'building_surface' as const, id: 'z1', name: 'A', color: '#f00', selectors: [], groupId: 'g1' },
      { type: 'building_surface' as const, id: 'z2', name: 'B', color: '#0f0', selectors: [] },
    ];

    const specs = toSurfaceSelectionLayerSpecs(zones, 'g1');
    expect(specs[0].active).toBe(true);
    expect(specs[1].active).toBe(false);
  });

  it('falls back to zone.id for active match when groupId is absent', () => {
    const zones = [
      { type: 'building_surface' as const, id: 'z1', name: 'A', color: '#f00', selectors: [] },
    ];

    const specs = toSurfaceSelectionLayerSpecs(zones, 'z1');
    expect(specs[0].active).toBe(true);
  });
});
