/**
 * Renders one MeshChunk as a `<mesh/>` with a non-indexed/indexed
 * BufferGeometry built from the flat `positions`/`indices`/`colors`
 * arrays produced by `app/backend/scene_geometry.py`.
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import type { MeshChunk } from './types';

export interface MeshLayerProps {
  chunk: MeshChunk;
  /** Optional picking userData to attach to the mesh (used by `<Picker/>`). */
  userData?: Record<string, unknown>;
  /** Optional renderOrder override (default 0). */
  renderOrder?: number;
}

/** Build a `THREE.BufferGeometry` from a flat MeshChunk payload. */
function buildGeometry(chunk: MeshChunk): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(chunk.positions);
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  if (chunk.colors && chunk.colors.length > 0) {
    const colors = new Float32Array(chunk.colors);
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  if (chunk.indices && chunk.indices.length > 0) {
    const indices =
      chunk.positions.length / 3 > 65535
        ? new Uint32Array(chunk.indices)
        : new Uint16Array(chunk.indices);
    geom.setIndex(new THREE.BufferAttribute(indices, 1));
  }
  geom.computeVertexNormals();
  return geom;
}

export function MeshLayer({ chunk, userData, renderOrder = 0 }: MeshLayerProps) {
  const geometry = useMemo(() => buildGeometry(chunk), [chunk]);
  const hasVertexColors = !!(chunk.colors && chunk.colors.length > 0);

  // Build material per-chunk; React-Three-Fiber will dispose for us.
  const material = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      color: hasVertexColors
        ? 0xffffff
        : new THREE.Color(...(chunk.color ?? [0.7, 0.7, 0.7])),
      vertexColors: hasVertexColors,
      flatShading: !!chunk.flat_shading,
      transparent: chunk.opacity < 1,
      opacity: chunk.opacity,
      metalness: 0.0,
      roughness: 0.85,
      side: THREE.DoubleSide,
    });
    return mat;
  }, [chunk, hasVertexColors]);

  return (
    <mesh
      geometry={geometry}
      material={material}
      userData={{ chunkName: chunk.name, ...(userData ?? {}) }}
      renderOrder={renderOrder}
      castShadow={false}
      receiveShadow={false}
    />
  );
}
