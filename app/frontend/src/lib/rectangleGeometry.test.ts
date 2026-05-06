import { describe, it, expect } from 'vitest';
import {
  buildAlignedRectangle,
  buildRotatedRectangleFromClicks,
  normalizeRectangleVertices,
  rotateVertices,
  buildDimensionRectangleApprox,
} from './rectangleGeometry';

/** Euclidean distance between two lon/lat points (flat-Earth, good enough for tests). */
function dist([ax, ay]: [number, number], [bx, by]: [number, number]) {
  return Math.hypot(ax - bx, ay - by);
}

const D2R = Math.PI / 180;
const R = 6378137;

/** Web Mercator metres between two lon/lat points (same projection as the lib). */
function mercDist([aLon, aLat]: [number, number], [bLon, bLat]: [number, number]) {
  const ax = R * aLon * D2R;
  const ay = R * Math.log(Math.tan(Math.PI / 4 + (aLat * D2R) / 2));
  const bx = R * bLon * D2R;
  const by = R * Math.log(Math.tan(Math.PI / 4 + (bLat * D2R) / 2));
  return Math.hypot(ax - bx, ay - by);
}

/** Haversine great-circle distance in metres — true ground distance. */
function haversineDist([aLon, aLat]: [number, number], [bLon, bLat]: [number, number]) {
  const lat1 = aLat * D2R;
  const lat2 = bLat * D2R;
  const dLat = (bLat - aLat) * D2R;
  const dLon = (bLon - aLon) * D2R;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// ─────────────────────────────────────────────────────────────
describe('buildAlignedRectangle', () => {
  it('returns four vertices from two diagonal corners', () => {
    const v = buildAlignedRectangle([139.76, 35.67], [139.78, 35.69]) as [number,number][];
    expect(v).toHaveLength(4);
  });

  it('orders vertices SW → NW → NE → SE', () => {
    const [[swLon, swLat], [nwLon, nwLat], [neLon, neLat], [seLon, seLat]] =
      buildAlignedRectangle([139.76, 35.67], [139.78, 35.69]) as [number,number][];
    expect(swLon).toBe(nwLon);   // west side shares lon
    expect(neLon).toBe(seLon);   // east side shares lon
    expect(swLat).toBe(seLat);   // south side shares lat
    expect(nwLat).toBe(neLat);   // north side shares lat
    expect(swLon).toBeLessThan(neLon); // west < east
    expect(swLat).toBeLessThan(nwLat); // south < north
  });

  it('is commutative — swapping corners produces the same result', () => {
    const a = buildAlignedRectangle([139.76, 35.67], [139.78, 35.69]);
    const b = buildAlignedRectangle([139.78, 35.69], [139.76, 35.67]);
    a.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(b[i][0], 8);
      expect(lat).toBeCloseTo(b[i][1], 8);
    });
  });
});

// ─────────────────────────────────────────────────────────────
describe('buildRotatedRectangleFromClicks', () => {
  // Three points forming a 45° rotated rectangle around Tokyo Station
  const p1: [number, number] = [139.764, 35.681];
  const p2: [number, number] = [139.770, 35.687]; // ~850 m NE
  const p3: [number, number] = [139.770, 35.681]; // offset perpendicular

  it('returns four vertices', () => {
    const v = buildRotatedRectangleFromClicks(p1, p2, p3)!;
    expect(v).toHaveLength(4);
  });

  it('is NOT axis-aligned (opposite sides parallel but not lat/lon aligned)', () => {
    const [v0, v1, , v3] = buildRotatedRectangleFromClicks(p1, p2, p3)!;
    // If it were axis-aligned, v0 and v3 would share lon AND v0 and v1 would share lon
    // At least one side must NOT be iso-lon or iso-lat
    const side1IsAligned = Math.abs(v0[0] - v1[0]) < 1e-8 || Math.abs(v0[1] - v1[1]) < 1e-8;
    const side2IsAligned = Math.abs(v0[0] - v3[0]) < 1e-8 || Math.abs(v0[1] - v3[1]) < 1e-8;
    expect(side1IsAligned && side2IsAligned).toBe(false);
  });

  it('opposite sides have equal length (is a parallelogram)', () => {
    const [v0, v1, v2, v3] = buildRotatedRectangleFromClicks(p1, p2, p3)!;
    const side01 = mercDist(v0, v1);
    const side32 = mercDist(v3, v2);
    const side03 = mercDist(v0, v3);
    const side12 = mercDist(v1, v2);
    expect(side01).toBeCloseTo(side32, 0); // tolerance 1 m
    expect(side03).toBeCloseTo(side12, 0);
  });

  it('returns null when p1 and p2 are the same point', () => {
    expect(buildRotatedRectangleFromClicks([139.764, 35.681], [139.764, 35.681], [139.770, 35.681])).toBeNull();
  });

  it('returns null when p3 produces zero extrusion', () => {
    // p3 is exactly on the p1→p2 line → no extrusion
    const mid: [number, number] = [
      (p1[0] + p2[0]) / 2,
      (p1[1] + p2[1]) / 2,
    ];
    expect(buildRotatedRectangleFromClicks(p1, p2, mid)).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────
describe('normalizeRectangleVertices', () => {
  // Create a known-good aligned rectangle and then deliberately shuffle vertices
  const aligned = buildAlignedRectangle([139.76, 35.67], [139.78, 35.69]) as [number,number][];
  // SW=0, NW=1, NE=2, SE=3

  it('keeps correctly-ordered vertices unchanged', () => {
    const v = normalizeRectangleVertices(aligned as [number, number][]);
    aligned.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(v[i][0], 8);
      expect(lat).toBeCloseTo(v[i][1], 8);
    });
  });

  it('reorders reversed input to SW → NW → NE → SE', () => {
    // Reverse: SE, NE, NW, SW → should normalize back to SW, NW, NE, SE
    const shuffled: [number, number][] = [aligned[3], aligned[2], aligned[1], aligned[0]];
    const v = normalizeRectangleVertices(shuffled);
    aligned.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(v[i][0], 6);
      expect(lat).toBeCloseTo(v[i][1], 6);
    });
  });
});

// ─────────────────────────────────────────────────────────────
describe('rotateVertices', () => {
  const aligned = buildAlignedRectangle([139.76, 35.67], [139.78, 35.69]) as [number, number][];

  it('rotation by 0° returns same vertices', () => {
    const v = rotateVertices(aligned as [number, number][], 0);
    aligned.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(v[i][0], 8);
      expect(lat).toBeCloseTo(v[i][1], 8);
    });
  });

  it('preserves centroid under rotation', () => {
    const cx = aligned.reduce((s, [x]) => s + x, 0) / 4;
    const cy = aligned.reduce((s, [, y]) => s + y, 0) / 4;
    const v = rotateVertices(aligned as [number, number][], 30);
    const rcx = v.reduce((s, [x]) => s + x, 0) / 4;
    const rcy = v.reduce((s, [, y]) => s + y, 0) / 4;
    expect(rcx).toBeCloseTo(cx, 5);
    expect(rcy).toBeCloseTo(cy, 5);
  });

  it('preserves side lengths under rotation', () => {
    const v = rotateVertices(aligned as [number, number][], 30);
    expect(mercDist(v[0], v[1])).toBeCloseTo(mercDist(aligned[0] as [number,number], aligned[1] as [number,number]), 0);
    expect(mercDist(v[0], v[3])).toBeCloseTo(mercDist(aligned[0] as [number,number], aligned[3] as [number,number]), 0);
  });

  it('rotation by 360° returns to near-original position', () => {
    const v = rotateVertices(aligned as [number, number][], 360);
    aligned.forEach(([lon, lat], i) => {
      expect(lon).toBeCloseTo(v[i][0], 5);
      expect(lat).toBeCloseTo(v[i][1], 5);
    });
  });
});

// ─────────────────────────────────────────────────────────────
describe('buildDimensionRectangleApprox', () => {
  it('returns four vertices', () => {
    expect(buildDimensionRectangleApprox(139.77, 35.68, 1000, 800)).toHaveLength(4);
  });

  it('zero rotation is approximately axis-aligned', () => {
    const v = buildDimensionRectangleApprox(139.77, 35.68, 1000, 800, 0);
    // SW and NW share lon
    expect(v[0][0]).toBeCloseTo(v[1][0], 6);
    // SW and SE share lat
    expect(v[0][1]).toBeCloseTo(v[3][1], 6);
  });

  it('nonzero rotation produces non-axis-aligned rectangle', () => {
    const v = buildDimensionRectangleApprox(139.77, 35.68, 1000, 800, 30);
    const isAxisAligned =
      Math.abs(v[0][0] - v[1][0]) < 1e-6 && Math.abs(v[0][1] - v[3][1]) < 1e-6;
    expect(isAxisAligned).toBe(false);
  });

  it('preserves approximate width and height in metres', () => {
    const widthM = 1000;
    const heightM = 800;
    const v = buildDimensionRectangleApprox(139.77, 35.68, widthM, heightM, 0);
    const measuredWidth  = haversineDist(v[0] as [number,number], v[3] as [number,number]); // SW → SE ≈ width
    const measuredHeight = haversineDist(v[0] as [number,number], v[1] as [number,number]); // SW → NW ≈ height
    // Allow ±1 % tolerance for the flat-Earth approximation
    expect(Math.abs(measuredWidth  - widthM )).toBeLessThan(widthM  * 0.01);
    expect(Math.abs(measuredHeight - heightM)).toBeLessThan(heightM * 0.01);
  });
});
