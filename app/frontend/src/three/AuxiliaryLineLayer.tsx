/**
 * Renders imported DXF auxiliary lines as flat 3D reference polylines draped at
 * ground height. Non-voxelized overlay; lon/lat is projected to scene metres via
 * `lonLatToXY` (typically lib/grid.ts's lonLatToUvM), matching ZoneOutlines.
 */
import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

import type { AuxiliaryLineDto } from '../api';
import { isAuxLayerVisible } from '../lib/auxiliaryLines';

export interface AuxiliaryLineLayerProps {
  lines: AuxiliaryLineDto[];
  /** Projection lon/lat -> world-XY metres (lib/grid.ts lonLatToUvM result). */
  lonLatToXY?: (lon: number, lat: number) => [number, number];
  /** Per-file/per-layer visibility; a layer shows unless explicitly false. */
  visibility?: Record<string, Record<string, boolean>>;
  /** Ground height (metres) to drape the lines at. */
  zHeight?: number;
  lineWidth?: number;
}

/** VoxCity indigo fallback when a DXF entity carries no usable color. */
const DEFAULT_COLOR = '#6666FF';

export function AuxiliaryLineLayer({
  lines,
  lonLatToXY,
  visibility = {},
  zHeight = 0.5,
  lineWidth = 2,
}: AuxiliaryLineLayerProps) {
  const entries = useMemo(() => {
    const out: { id: string; color: string; points: [number, number, number][] }[] = [];
    for (const ln of lines) {
      if (!isAuxLayerVisible(visibility, ln.file_name, ln.layer)) continue;
      if (!ln.points || ln.points.length < 2) continue;
      const pts: [number, number, number][] = ln.points.map(([lon, lat]) => {
        const [x, y] = lonLatToXY ? lonLatToXY(lon, lat) : [lon, lat];
        return [x, y, zHeight];
      });
      out.push({ id: ln.id, color: ln.color || DEFAULT_COLOR, points: pts });
    }
    return out;
  }, [lines, lonLatToXY, visibility, zHeight]);

  if (entries.length === 0) return null;

  return (
    <group renderOrder={998}>
      {entries.map((e) => (
        <Line
          key={e.id}
          points={e.points}
          color={new THREE.Color(e.color)}
          lineWidth={lineWidth}
          depthTest={false}
          depthWrite={false}
          renderOrder={998}
          transparent
        />
      ))}
    </group>
  );
}
