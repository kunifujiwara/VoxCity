import { describe, it, expect } from 'vitest';
import { transformModelPoint, defaultPlacement, unitScale, type Placement } from '../lib/objPlacement';

// PlacementGizmo's mesh transform must place model point `amp` (anchorModelPoint)
// such that the gizmo's mesh.position (= anchorScene + move, per the component's
// derivation) plus a Z-rotation by (rotation + domainRotationDeg) reproduces
// transformModelPoint's output exactly. This test pins the combined-rotation
// invariant the component relies on, independent of any three.js/R3F rendering.
describe('PlacementGizmo rotation invariant', () => {
  it('rotation + domainRotationDeg matches the combined transformModelPoint rotation', () => {
    const p: Placement = { ...defaultPlacement(), rotation: 30, move: [0, 0, 0], units: 'm' };
    const phiDeg = 20;
    const out = transformModelPoint([1, 0, 0], p, phiDeg);
    const psiRad = ((p.rotation + phiDeg) * Math.PI) / 180;
    // Mesh-equivalent computation: Rot(psi) applied to local point [1,0] (amp=[0,0,0]).
    const meshEast = Math.cos(psiRad);
    const meshNorth = Math.sin(psiRad);
    expect(out[0]).toBeCloseTo(meshEast, 9);
    expect(out[1]).toBeCloseTo(meshNorth, 9);
  });

  it('amp != [0,0,0]: transformModelPoint(amp, ...) returns pure move (zero local offset)', () => {
    const p: Placement = {
      ...defaultPlacement(), rotation: 15, anchorModelPoint: [4, -2, 1],
      move: [3, 7, -1], units: 'm',
    };
    const out = transformModelPoint(p.anchorModelPoint, p, 25);
    expect(out[0]).toBeCloseTo(p.move[0], 9);
    expect(out[1]).toBeCloseTo(p.move[1], 9);
    expect(out[2]).toBeCloseTo(p.move[2], 9);
  });

  it('matches the mesh-equivalent formula with ALL of pt, anchorModelPoint, rotation, domainRotationDeg, and move simultaneously nonzero', () => {
    const p: Placement = {
      ...defaultPlacement(),
      rotation: 35,
      anchorModelPoint: [3, -4, 2],
      move: [10, -6, 1.5],
      units: 'm',
    };
    const phiDeg = 50;
    const pt: [number, number, number] = [7, 1, -2]; // != anchorModelPoint, nonzero offset
    const anchorScene: [number, number, number] = [500, -300, 12];

    // Real function under test.
    const offset = transformModelPoint(pt, p, phiDeg);
    const realWorld: [number, number, number] = [
      anchorScene[0] + offset[0],
      anchorScene[1] + offset[1],
      anchorScene[2] + offset[2],
    ];

    // Mesh-equivalent formula PlacementGizmo.tsx actually implements:
    // scale -> pre-translate geometry by -amp -> rotate by (rotation+domainRotationDeg)
    // about Z -> position = anchorScene + move.
    const s = unitScale(p.units);
    const amp = p.anchorModelPoint;
    const localX = (pt[0] - amp[0]) * s;
    const localY = (pt[1] - amp[1]) * s;
    const localZ = (pt[2] - amp[2]) * s;
    const psiRad = ((p.rotation + phiDeg) * Math.PI) / 180;
    const cosPsi = Math.cos(psiRad), sinPsi = Math.sin(psiRad);
    const meshPos: [number, number, number] = [
      anchorScene[0] + p.move[0],
      anchorScene[1] + p.move[1],
      anchorScene[2] + p.move[2],
    ];
    const meshWorld: [number, number, number] = [
      meshPos[0] + (localX * cosPsi - localY * sinPsi),
      meshPos[1] + (localX * sinPsi + localY * cosPsi),
      meshPos[2] + localZ,
    ];

    expect(meshWorld[0]).toBeCloseTo(realWorld[0], 9);
    expect(meshWorld[1]).toBeCloseTo(realWorld[1], 9);
    expect(meshWorld[2]).toBeCloseTo(realWorld[2], 9);
  });
});
