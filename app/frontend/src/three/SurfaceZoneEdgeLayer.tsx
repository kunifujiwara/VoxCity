import { Line } from '@react-three/drei';
import type { SurfaceZoneEdgeRenderSpec } from './surfaceZoneEdges';
import { buildSurfaceZoneEdgeLineSpecs } from './surfaceZoneEdges';

interface SurfaceZoneEdgeLayerProps {
  zones: SurfaceZoneEdgeRenderSpec[];
}

export function SurfaceZoneEdgeLayer({ zones }: SurfaceZoneEdgeLayerProps) {
  const lineSpecs = buildSurfaceZoneEdgeLineSpecs(zones);

  if (lineSpecs.length === 0) return null;

  return (
    <>
      {lineSpecs.map((spec) => (
        <Line
          key={spec.id}
          points={spec.points}
          color={spec.color}
          lineWidth={spec.lineWidth}
          transparent
          opacity={spec.opacity}
          segments
        />
      ))}
    </>
  );
}
