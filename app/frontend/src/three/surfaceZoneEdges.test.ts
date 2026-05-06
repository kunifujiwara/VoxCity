import { describe, expect, it } from 'vitest';
import type { Zone } from '../types/zones';
import {
  getSurfaceZonesWithSelectors,
  shouldFetchSurfaceZoneEdges,
  surfaceZoneEdgeRequestKey,
  toSurfaceZoneEdgeRenderSpecs,
  segmentsToLinePoints,
  buildSurfaceZoneEdgeLineSpecs,
} from './surfaceZoneEdges';

const surfaceZone: Zone = {
  id: 's1',
  name: 'Roof',
  type: 'building_surface',
  color: '#ff0080',
  selectors: [{ buildingId: 7, mode: 'roof' }],
};

const horizontalZone: Zone = {
  id: 'h1',
  name: 'Horizontal',
  type: 'horizontal',
  color: '#00ff00',
  shape: 'polygon',
  ring_lonlat: [[0, 0], [1, 0], [1, 1]],
};

describe('shouldFetchSurfaceZoneEdges', () => {
  it('requires a model, enabled overlay, and selected surface zones', () => {
    expect(shouldFetchSurfaceZoneEdges({ hasModel: false, enabled: true, zones: [surfaceZone] })).toBe(false);
    expect(shouldFetchSurfaceZoneEdges({ hasModel: true, enabled: false, zones: [surfaceZone] })).toBe(false);
    expect(shouldFetchSurfaceZoneEdges({ hasModel: true, enabled: true, zones: [horizontalZone] })).toBe(false);
    expect(shouldFetchSurfaceZoneEdges({ hasModel: true, enabled: true, zones: [surfaceZone] })).toBe(true);
  });
});

describe('surface zone edge request helpers', () => {
  it('keeps only building surface zones with selectors', () => {
    const emptySurfaceZone: Zone = { ...surfaceZone, id: 'empty', selectors: [] };

    expect(getSurfaceZonesWithSelectors([horizontalZone, emptySurfaceZone, surfaceZone])).toEqual([surfaceZone]);
  });

  it('does not change request key for unrelated horizontal zones or color edits', () => {
    const renamedAndRecolored: Zone = { ...surfaceZone, name: 'Recolored', color: '#ffff00' };
    const withHorizontal = [surfaceZone, horizontalZone];

    expect(surfaceZoneEdgeRequestKey([surfaceZone])).toBe(surfaceZoneEdgeRequestKey(withHorizontal));
    expect(surfaceZoneEdgeRequestKey([surfaceZone])).toBe(surfaceZoneEdgeRequestKey([renamedAndRecolored]));
  });

  it('changes request key when selected surface selectors change', () => {
    const walls: Zone = { ...surfaceZone, selectors: [{ buildingId: 7, mode: 'all_walls' }] };

    expect(surfaceZoneEdgeRequestKey([surfaceZone])).not.toBe(surfaceZoneEdgeRequestKey([walls]));
  });
});

describe('toSurfaceZoneEdgeRenderSpecs', () => {
  it('attaches frontend zone colors and skips empty or unknown payloads', () => {
    const specs = toSurfaceZoneEdgeRenderSpecs([surfaceZone], {
      zones: [
        { id: 's1', segments: [[0, 0, 0, 1, 0, 0]] },
        { id: 'missing', segments: [[0, 0, 0, 0, 1, 0]] },
        { id: 'empty', segments: [] },
      ],
    });
    expect(specs).toEqual([{ id: 's1', color: '#ff0080', segments: [[0, 0, 0, 1, 0, 0]] }]);
  });
});

describe('line conversion', () => {
  it('converts segments to drei Line points and builds colored line specs', () => {
    const points = segmentsToLinePoints([[0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0]]);
    expect(points).toEqual([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]]);

    const lineSpecs = buildSurfaceZoneEdgeLineSpecs([{ id: 's1', color: '#ff0080', segments: [[0, 0, 0, 1, 0, 0]] }]);
    expect(lineSpecs).toEqual([
      { id: 's1', color: '#ff0080', lineWidth: 2, opacity: 1, points: [[0, 0, 0], [1, 0, 0]] },
    ]);
  });
});
