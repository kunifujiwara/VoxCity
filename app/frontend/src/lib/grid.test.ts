import { describe, expect, it } from 'vitest';
import { buildingsFullyContainedInPolygon, buildingsInPolygon } from './grid';

function polygonFeature(id: number, ring: [number, number][]) {
  return {
    type: 'Feature',
    properties: { idx: id },
    geometry: { type: 'Polygon', coordinates: [ring] },
  };
}

const squareSelection: [number, number][] = [[0, 0], [4, 0], [4, 4], [0, 4]];

describe('buildingsFullyContainedInPolygon', () => {
  it('selects fully contained footprints and excludes partial intersections', () => {
    const fc = {
      type: 'FeatureCollection',
      features: [
        polygonFeature(1, [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]),
        polygonFeature(2, [[3, 1], [5, 1], [5, 2], [3, 2], [3, 1]]),
        polygonFeature(3, [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
      ],
    };

    expect(buildingsFullyContainedInPolygon(fc, squareSelection)).toEqual([1, 3]);
    expect(buildingsInPolygon(fc, squareSelection)).toContain(2);
  });

  it('returns an empty list for empty and degenerate inputs', () => {
    expect(buildingsFullyContainedInPolygon(null, squareSelection)).toEqual([]);
    expect(buildingsFullyContainedInPolygon({ type: 'FeatureCollection', features: [] }, squareSelection)).toEqual([]);
    expect(buildingsFullyContainedInPolygon({ type: 'FeatureCollection', features: [] }, [[0, 0], [1, 1]])).toEqual([]);
  });

  it('requires every MultiPolygon exterior ring to be contained', () => {
    const fc = {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: { idx: 4 },
          geometry: {
            type: 'MultiPolygon',
            coordinates: [
              [[[0.5, 0.5], [0.8, 0.5], [0.8, 0.8], [0.5, 0.8], [0.5, 0.5]]],
              [[[2.5, 2.5], [2.8, 2.5], [2.8, 2.8], [2.5, 2.8], [2.5, 2.5]]],
            ],
          },
        },
        {
          type: 'Feature',
          properties: { idx: 5 },
          geometry: {
            type: 'MultiPolygon',
            coordinates: [
              [[[0.5, 0.5], [0.8, 0.5], [0.8, 0.8], [0.5, 0.8], [0.5, 0.5]]],
              [[[4.5, 2.5], [4.8, 2.5], [4.8, 2.8], [4.5, 2.8], [4.5, 2.5]]],
            ],
          },
        },
      ],
    };

    expect(buildingsFullyContainedInPolygon(fc, squareSelection)).toEqual([4]);
  });

  it('excludes a footprint edge that crosses outside a concave selection', () => {
    const concaveSelection: [number, number][] = [
      [0, 0], [4, 0], [4, 4], [3, 4], [3, 1], [1, 1], [1, 4], [0, 4],
    ];
    const fc = {
      type: 'FeatureCollection',
      features: [
        polygonFeature(6, [[0.25, 0.25], [3.75, 0.25], [3.75, 0.75], [0.25, 0.75], [0.25, 0.25]]),
        polygonFeature(7, [[0.5, 3.2], [3.5, 3.2], [3.5, 3.5], [0.5, 3.5], [0.5, 3.2]]),
      ],
    };

    expect(buildingsFullyContainedInPolygon(fc, concaveSelection)).toEqual([6]);
  });

  it('excludes scaled lon/lat edges crossing narrow concave cut-outs', () => {
    const scale = 0.0001;
    const p = (lon: number, lat: number): [number, number] => [lon * scale, lat * scale];
    const concaveSelection: [number, number][] = [
      p(0, 0), p(10, 0), p(10, 10), p(8, 10), p(8, 4), p(7, 4),
      p(7, 10), p(3, 10), p(3, 4), p(2, 4), p(2, 10), p(0, 10),
    ];
    const fc = {
      type: 'FeatureCollection',
      features: [
        polygonFeature(8, [p(1, 5), p(9, 5), p(9, 5.5), p(1, 5.5), p(1, 5)]),
      ],
    };

    expect(buildingsFullyContainedInPolygon(fc, concaveSelection)).toEqual([]);
  });
});