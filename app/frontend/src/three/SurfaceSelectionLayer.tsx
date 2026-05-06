/**
 * Renders selected building surface zones as either:
 * - 'fill': translucent colored mesh faces (Zoning tab)
 * - 'boundary': outer boundary lines with dark halo + dashed color line (Simulation tabs)
 */
import { Line } from '@react-three/drei';
import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

import type { MeshChunkDto } from '../api';
import type { SurfaceFaceMeta } from './types';
import {
  buildBoundaryLinePoints,
  buildZoneFillGeometry,
  type BoundaryPoint,
  type SurfaceSelectionDisplayMode,
  type SurfaceSelectionLayerSpec,
} from './surfaceSelection';

export interface SurfaceSelectionLayerProps {
  surfaceChunk: MeshChunkDto | null;
  faceToSurface: SurfaceFaceMeta[];
  zones: SurfaceSelectionLayerSpec[];
  enabled: boolean;
  displayMode: SurfaceSelectionDisplayMode;
}

function applyBoundaryLineMaterialFlags(line: unknown) {
  const material = (line as { material?: THREE.Material | THREE.Material[] }).material;
  const materials = Array.isArray(material) ? material : material ? [material] : [];
  for (const mat of materials) {
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.needsUpdate = true;
  }
}

export function SurfaceSelectionLayer({
  surfaceChunk,
  faceToSurface,
  zones,
  enabled,
  displayMode,
}: SurfaceSelectionLayerProps) {
  const zoneGeometries = useMemo(() => {
    if (!enabled || !surfaceChunk || displayMode !== 'fill') return [];
    return zones.map((zone) => ({
      id: zone.id,
      color: zone.color,
      active: zone.active,
      geometry: buildZoneFillGeometry(surfaceChunk, faceToSurface, zone.selectors),
    }));
  }, [surfaceChunk, faceToSurface, zones, enabled, displayMode]);

  useEffect(() => {
    return () => {
      for (const zone of zoneGeometries) {
        zone.geometry?.dispose();
      }
    };
  }, [zoneGeometries]);

  const zoneBoundaries = useMemo(() => {
    if (!enabled || !surfaceChunk || displayMode !== 'boundary') return [];
    return zones.map((zone) => ({
      id: zone.id,
      color: zone.color,
      points: buildBoundaryLinePoints(surfaceChunk, faceToSurface, zone.selectors),
    }));
  }, [surfaceChunk, faceToSurface, zones, enabled, displayMode]);

  if (!enabled || !surfaceChunk) return null;

  if (displayMode === 'fill') {
    return (
      <group>
        {zoneGeometries.map(
          (zone) =>
            zone.geometry && (
              <mesh key={zone.id} geometry={zone.geometry} renderOrder={15}>
                <meshBasicMaterial
                  color={zone.color}
                  transparent
                  opacity={zone.active ? 0.5 : 0.32}
                  side={THREE.DoubleSide}
                  depthTest={false}
                  depthWrite={false}
                />
              </mesh>
            ),
        )}
      </group>
    );
  }

  if (displayMode !== 'boundary') return null;

  return (
    <group>
      {zoneBoundaries.map((zone) =>
        zone.points.length > 0 ? (
          <group key={zone.id}>
            <Line
              points={zone.points as BoundaryPoint[]}
              segments
              color="#000000"
              lineWidth={4}
              transparent
              opacity={0.55}
              onUpdate={applyBoundaryLineMaterialFlags}
              renderOrder={16}
            />
            <Line
              points={zone.points as BoundaryPoint[]}
              segments
              color={zone.color}
              lineWidth={2}
              dashed
              dashSize={0.8}
              gapSize={0.45}
              dashScale={1}
              onUpdate={applyBoundaryLineMaterialFlags}
              renderOrder={17}
            />
          </group>
        ) : null,
      )}
    </group>
  );
}
