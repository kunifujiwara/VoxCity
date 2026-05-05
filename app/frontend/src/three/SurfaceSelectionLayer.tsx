/**
 * Renders selected building surface triangles as a colored overlay.
 *
 * Uses the same selector semantics as the backend:
 * - 'whole': all faces of a building
 * - 'roof': roof faces only
 * - 'all_walls': wall faces only
 * - 'wall_orientation': walls of specific orientation
 * - 'faces': specific face keys
 * - 'exclude_faces': excluded from the above
 */
import { useMemo } from 'react';
import * as THREE from 'three';

import type { MeshChunk } from './types';
import type { SurfaceFaceMeta } from './types';
import type { SurfaceSelector } from '../types/zones';

export interface SurfaceSelectionLayerProps {
  surfaceChunk: MeshChunk | null;
  faceToSurface: SurfaceFaceMeta[];
  activeZoneColor: string | null;
  selectedSelectors: SurfaceSelector[];
  enabled: boolean;
}

/** Determine if a face is selected given the active selectors. */
function isFaceSelected(meta: SurfaceFaceMeta, selectors: SurfaceSelector[]): boolean {
  const buildingSelectors = selectors.filter((s) => s.buildingId === meta.buildingId);
  if (buildingSelectors.length === 0) return false;

  // Check excludes first
  const excluded = buildingSelectors.some(
    (s) =>
      s.mode === 'exclude_faces' &&
      'faceKeys' in s &&
      s.faceKeys.includes(meta.faceKey),
  );
  if (excluded) return false;

  // Check positive selectors
  return buildingSelectors.some((s) => {
    if (s.mode === 'whole') return true;
    if (s.mode === 'roof') return meta.surfaceKind === 'roof';
    if (s.mode === 'all_walls') return meta.surfaceKind === 'wall';
    if (s.mode === 'wall_orientation')
      return meta.surfaceKind === 'wall' && 'orientation' in s && meta.orientation === s.orientation;
    if (s.mode === 'faces') return 'faceKeys' in s && s.faceKeys.includes(meta.faceKey);
    return false;
  });
}

export function SurfaceSelectionLayer({
  surfaceChunk,
  faceToSurface,
  activeZoneColor,
  selectedSelectors,
  enabled,
}: SurfaceSelectionLayerProps) {
  const geometry = useMemo(() => {
    if (!enabled || !surfaceChunk || !activeZoneColor || selectedSelectors.length === 0) {
      return null;
    }

    const srcPositions = surfaceChunk.positions;
    if (!srcPositions || srcPositions.length === 0) return null;

    // Non-indexed triangle soup: each face is 3 vertices * 3 floats = 9 floats
    // faceToSurface[i] maps triangle i to its surface metadata
    const selectedTriangleIndices: number[] = [];
    for (let i = 0; i < faceToSurface.length; i++) {
      const meta = faceToSurface[i];
      if (meta && isFaceSelected(meta, selectedSelectors)) {
        selectedTriangleIndices.push(i);
      }
    }

    if (selectedTriangleIndices.length === 0) return null;

    const floatsPerTriangle = 9; // 3 verts * 3 floats
    const out = new Float32Array(selectedTriangleIndices.length * floatsPerTriangle);
    let outOffset = 0;
    for (const triIdx of selectedTriangleIndices) {
      const srcOffset = triIdx * floatsPerTriangle;
      out[outOffset++] = srcPositions[srcOffset];
      out[outOffset++] = srcPositions[srcOffset + 1];
      out[outOffset++] = srcPositions[srcOffset + 2];
      out[outOffset++] = srcPositions[srcOffset + 3];
      out[outOffset++] = srcPositions[srcOffset + 4];
      out[outOffset++] = srcPositions[srcOffset + 5];
      out[outOffset++] = srcPositions[srcOffset + 6];
      out[outOffset++] = srcPositions[srcOffset + 7];
      out[outOffset++] = srcPositions[srcOffset + 8];
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(out, 3));
    geom.computeVertexNormals();
    return geom;
  }, [surfaceChunk, faceToSurface, activeZoneColor, selectedSelectors, enabled]);

  if (!geometry) return null;

  const color = new THREE.Color(activeZoneColor!);

  return (
    <mesh geometry={geometry} renderOrder={15}>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.5}
        side={THREE.DoubleSide}
        depthTest={false}
        depthWrite={false}
      />
    </mesh>
  );
}
