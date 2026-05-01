/**
 * Camera + OrbitControls preset for VoxCity scenes.
 *
 * The viewer uses Z-up world space (matching backend metres). We keep the
 * controls' `up` vector aligned to +Z and orbit around the scene centre.
 */
import { useEffect } from 'react';
import { useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

export interface CameraControlsProps {
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
  /** When true, fits the camera to the bbox once on mount. */
  autoFit?: boolean;
}

export function CameraControls({
  bboxMin,
  bboxMax,
  autoFit = true,
}: CameraControlsProps) {
  const { camera } = useThree();

  useEffect(() => {
    camera.up.set(0, 0, 1);
    if (autoFit) {
      const cx = (bboxMin[0] + bboxMax[0]) / 2;
      const cy = (bboxMin[1] + bboxMax[1]) / 2;
      const cz = (bboxMin[2] + bboxMax[2]) / 2;
      const dx = bboxMax[0] - bboxMin[0];
      const dy = bboxMax[1] - bboxMin[1];
      const dz = bboxMax[2] - bboxMin[2];
      // Bounding-sphere fit, matching the legacy <ThreeViewer/> camera used
      // by the Generation / Edit tabs.
      const radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const isPersp = camera instanceof THREE.PerspectiveCamera;
      const fovRad = isPersp ? (camera as THREE.PerspectiveCamera).fov * Math.PI / 180 : 50 * Math.PI / 180;
      const fitDist = radius / Math.sin(fovRad / 2);
      // Direction: legacy ThreeViewer uses (0.65, 0.65, 0.5).normalize() in
      // its Y-up local space, where (x,y,z) world is mapped to (x, z, -y).
      // Inverting that mapping gives the equivalent Z-up world direction
      // (0.65, -0.5, 0.65).normalize() — east, south, up.
      const dir = new THREE.Vector3(0.65, -0.5, 0.65).normalize();
      camera.position.set(
        cx + dir.x * fitDist,
        cy + dir.y * fitDist,
        cz + dir.z * fitDist,
      );
      camera.lookAt(new THREE.Vector3(cx, cy, cz));
      if (isPersp) {
        const persp = camera as THREE.PerspectiveCamera;
        const maxDim = Math.max(dx, dy, dz) || 1;
        persp.near = Math.max(0.1, maxDim * 0.001);
        persp.far = maxDim * 20;
        persp.updateProjectionMatrix();
      }
    }
    // run once when bbox changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bboxMin[0], bboxMin[1], bboxMin[2], bboxMax[0], bboxMax[1], bboxMax[2]]);

  const target: [number, number, number] = [
    (bboxMin[0] + bboxMax[0]) / 2,
    (bboxMin[1] + bboxMax[1]) / 2,
    (bboxMin[2] + bboxMax[2]) / 2,
  ];

  return (
    <OrbitControls
      makeDefault
      target={target}
      enableDamping
      dampingFactor={0.08}
      // Z-up: keep gimbal sane.
      minPolarAngle={0.05}
      maxPolarAngle={Math.PI - 0.05}
    />
  );
}
