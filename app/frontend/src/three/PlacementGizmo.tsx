/**
 * R3F preview of the imported OBJ mesh + a TransformControls gizmo.
 *
 * Translate X/Y/Z maps to placement.move = [east, north, up]; rotate about the
 * up axis maps to placement.rotation. The mesh is built from the upload's
 * decimated preview geometry (model coords); object position/rotation/scale
 * are derived from the current Placement (units scale, rotation, move).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { TransformControls } from '@react-three/drei';
import { Placement, unitScale } from '../lib/objPlacement';

export interface PlacementGizmoProps {
  vertices: [number, number, number][]; // model coords, from upload.preview.vertices
  indices: [number, number, number][]; // from upload.preview.indices
  placement: Placement;
  /**
   * Scene-metre offset of the placement anchor: [east, north, up]. The 2D map
   * draws each footprint at `anchorScene + transformModelPoint(...)`, so the 3D
   * mesh must live in the same frame to stay in sync -- i.e. its world position
   * is `anchorScene + move`. Defaults to [0,0,0] (legacy origin-relative) when
   * the anchor/grid geometry isn't available yet.
   */
  anchorScene: [number, number, number];
  mode: 'translate' | 'rotate';
  onChange: (next: Partial<Placement>) => void;
}

export function PlacementGizmo({ vertices, indices, placement, anchorScene, mode, onChange }: PlacementGizmoProps) {
  // Hold the mesh in state (not just a ref) so TransformControls receives a
  // concrete `object` to attach to once the mesh has mounted. Attaching to the
  // mesh itself (rather than nesting it as TransformControls' children) keeps
  // drei's attach effect from re-running on every render -- nesting passes a
  // brand-new `children` element each render, which made drei detach/re-attach
  // the controls mid-drag and abort the gesture after ~1m. See PR notes.
  const [mesh, setMesh] = useState<THREE.Mesh | null>(null);
  // True while the user is actively dragging a gizmo handle. Used to suppress
  // the placement->mesh sync effect below so external state round-trips don't
  // fight the in-progress drag.
  const draggingRef = useRef(false);

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

  // Apply units scale + rotation about up (Z) + position whenever placement (or
  // the anchor's scene position) changes from OUTSIDE this component (e.g. the
  // numeric form, the units dropdown, or a 2D-map anchor click). The mesh's
  // world position is `anchorScene + move` so the 3D object sits at the same
  // place as the 2D map's footprint, which is drawn at
  // `anchorScene + transformModelPoint(...)`. The gizmo's own drags update
  // placement.move via onChange and flow back down through these props on the
  // next render.
  //
  // While a drag is in progress we skip this entirely: the gizmo already owns
  // the mesh transform, and re-writing position/rotation here mid-gesture would
  // snap the object back and cancel free movement.
  //
  // NOTE: unlike lib/objPlacement.ts's transformModelPoint (used by the 2D map
  // in ObjPlacementMap.tsx), this effect does not subtract placement.anchorModelPoint
  // before applying scale/rotation. This is harmless today because no UI control
  // in this app ever sets anchorModelPoint away from its [0,0,0] default -- but if
  // one is ever added, this gizmo will silently diverge from the 2D map and the
  // server-side commit transform. Apply the same (pt - anchorModelPoint) offset
  // here if/when anchorModelPoint becomes user-settable.
  useEffect(() => {
    if (!mesh || draggingRef.current) return;
    const s = unitScale(placement.units);
    mesh.scale.set(s, s, s);
    mesh.rotation.set(0, 0, (placement.rotation * Math.PI) / 180);
    mesh.position.set(
      anchorScene[0] + placement.move[0],
      anchorScene[1] + placement.move[1],
      anchorScene[2] + placement.move[2],
    );
  }, [mesh, placement, anchorScene]);

  const handleObjectChange = () => {
    if (!mesh) return;
    if (mode === 'translate') {
      // Mesh world position is anchorScene + move; recover move for placement.
      onChange({
        move: [
          mesh.position.x - anchorScene[0],
          mesh.position.y - anchorScene[1],
          mesh.position.z - anchorScene[2],
        ],
      });
    } else {
      onChange({ rotation: (mesh.rotation.z * 180) / Math.PI });
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
    <>
      <mesh ref={setMesh} geometry={geometry}>
        <meshStandardMaterial
          color="#e8590c"
          transparent
          opacity={0.4}
          side={THREE.DoubleSide}
          flatShading
        />
      </mesh>
      {mesh && (
        <TransformControls
          object={mesh}
          mode={mode}
          // Larger handles are easier to grab; the arrows draw with depthTest
          // off so they stay readable through the translucent preview mesh.
          size={1.1}
          // Continuous movement -- explicitly no snapping/quantization.
          translationSnap={null}
          rotationSnap={null}
          showX={showX}
          showY={showY}
          showZ={showZ}
          // Drag start/end: while dragging, suppress the placement->mesh sync
          // effect so external state round-trips don't cancel the gesture.
          onMouseDown={() => {
            draggingRef.current = true;
          }}
          onMouseUp={() => {
            draggingRef.current = false;
          }}
          onObjectChange={handleObjectChange}
        />
      )}
    </>
  );
}
