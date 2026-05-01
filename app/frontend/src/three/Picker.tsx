/**
 * Pointer-event picker. Wraps an arbitrary group of meshes; on click,
 * looks up the picked face's `face_to_cell` / `face_to_building` metadata
 * (attached as `mesh.userData.faceTo*`) and fires `onPick`.
 */
import { useCallback } from 'react';
import type { ThreeEvent } from '@react-three/fiber';
import type { PickResult } from './types';

export interface PickerProps {
  enabled?: boolean;
  onPick?: (hit: PickResult | null) => void;
  children: React.ReactNode;
}

export function Picker({ enabled = true, onPick, children }: PickerProps) {
  const handle = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      if (!enabled || !onPick) return;
      e.stopPropagation();
      const obj = e.object as any;
      const faceIdx = e.faceIndex ?? -1;
      const ud = obj?.userData ?? {};

      const faceToCell: number[][] | undefined = ud.faceToCell;
      const faceToBuilding: number[] | undefined = ud.faceToBuilding;
      const target: 'ground' | 'building' | undefined = ud.target;

      let cell: [number, number] | null = null;
      let buildingId: number | null = null;
      if (faceIdx >= 0 && target === 'ground' && faceToCell && faceToCell[faceIdx]) {
        const [i, j] = faceToCell[faceIdx];
        cell = [i, j];
      } else if (faceIdx >= 0 && target === 'building' && faceToBuilding) {
        buildingId = faceToBuilding[faceIdx] ?? null;
      }

      // Always emit a hit so consumers (e.g. LandmarkTab's nearest-centroid
      // lookup) can use the world-space click point even when the picked mesh
      // has no target/face metadata. Default `target` to 'ground' for typing.
      onPick({
        target: target ?? 'ground',
        cell,
        buildingId,
        point: [e.point.x, e.point.y, e.point.z],
      });
    },
    [enabled, onPick],
  );

  return (
    <group onClick={enabled ? handle : undefined}>{children}</group>
  );
}
