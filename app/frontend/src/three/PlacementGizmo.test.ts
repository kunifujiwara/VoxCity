import { describe, it, expect } from 'vitest';
import { transformModelPoint, defaultPlacement, type Placement } from '../lib/objPlacement';

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
});
