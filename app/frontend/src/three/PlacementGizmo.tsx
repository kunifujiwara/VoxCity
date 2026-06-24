/**
 * R3F preview of the imported OBJ mesh + a TransformControls gizmo.
 *
 * Translate X/Y/Z maps to placement.move = [east, north, up]; rotate about the
 * up axis maps to placement.rotation. The mesh is built from the upload's
 * decimated preview geometry (model coords); object position/rotation/scale
 * are derived from the current Placement (units scale, rotation, move).
 */
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { TransformControls } from '@react-three/drei';
import { Placement, unitScale } from '../lib/objPlacement';

export interface PlacementGizmoProps {
  vertices: [number, number, number][]; // model coords, from upload.preview.vertices
  indices: [number, number, number][]; // from upload.preview.indices
  placement: Placement;
  mode: 'translate' | 'rotate';
  onChange: (next: Partial<Placement>) => void;
}

export function PlacementGizmo({ vertices, indices, placement, mode, onChange }: PlacementGizmoProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const flatPositions = new Float32Array(vertices.flat());
    geom.setAttribute('position', new THREE.BufferAttribute(flatPositions, 3));
    const flatIndices = indices.flat();
    const indexArray =
      vertices.length > 65535 ? new Uint32Array(flatIndices) : new Uint16Array(flatIndices);
    geom.setIndex(new THREE.BufferAttribute(indexArray, 1));
    geom.computeVertexNormals();
    return geom;
  }, [vertices, indices]);

  // Apply units scale + rotation about up (Z) + position from move whenever
  // placement changes from OUTSIDE this component (e.g. the numeric form or
  // the 2D map's anchor click) -- the gizmo's own drags update placement via
  // onChange and don't need this effect to "round-trip" back to the mesh,
  // since React state flows down through these same props on next render.
  useEffect(() => {
    const m = meshRef.current;
    if (!m) return;
    const s = unitScale(placement.units);
    m.scale.set(s, s, s);
    m.rotation.set(0, 0, (placement.rotation * Math.PI) / 180);
    m.position.set(placement.move[0], placement.move[1], placement.move[2]);
  }, [placement]);

  const handleObjectChange = () => {
    const m = meshRef.current;
    if (!m) return;
    if (mode === 'translate') {
      onChange({ move: [m.position.x, m.position.y, m.position.z] });
    } else {
      onChange({ rotation: (m.rotation.z * 180) / Math.PI });
    }
  };

  // Constrain the gizmo to a single rotation axis (Z / up): Placement.rotation
  // is one scalar degree value, not a full 3D orientation, so the rotate-mode
  // gizmo must only ever show/allow the Z ring. Translate mode shows all
  // three arrows since `move` is a full [east, north, up] vector.
  const showX = mode === 'translate';
  const showY = mode === 'translate';
  const showZ = true;

  return (
    <TransformControls
      mode={mode}
      showX={showX}
      showY={showY}
      showZ={showZ}
      onObjectChange={handleObjectChange}
    >
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial color="#e8590c" transparent opacity={0.65} side={THREE.DoubleSide} flatShading />
      </mesh>
    </TransformControls>
  );
}
