/**
 * Pointer-event picker. Wraps an arbitrary group of meshes; on click,
 * looks up the picked face's `face_to_cell` / `face_to_building` metadata
 * (attached as `mesh.userData.faceTo*`) and fires `onPick`.
 *
 * Drag filtering: pointer events are only forwarded to onPick when the
 * pointer moved less than DRAG_THRESHOLD_PX pixels between down and up,
 * so that camera orbit drags do not accidentally trigger selection.
 */
import { useCallback, useRef } from 'react';
import type { ThreeEvent } from '@react-three/fiber';
import type { PickResult } from './types';

/** Maximum pointer travel (in CSS pixels) that is still treated as a click. */
const DRAG_THRESHOLD_PX = 5;

export interface PickerProps {
  enabled?: boolean;
  onPick?: (hit: PickResult | null) => void;
  children: React.ReactNode;
}

export function Picker({ enabled = true, onPick, children }: PickerProps) {
  const downRef = useRef<{ x: number; y: number } | null>(null);

  const handlePointerDown = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!enabled) return;
      // Record pointer position on primary button down only.
      if (e.nativeEvent.button === 0) {
        downRef.current = { x: e.nativeEvent.clientX, y: e.nativeEvent.clientY };
      }
    },
    [enabled],
  );

  const handlePointerUp = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!enabled || !onPick) return;
      if (e.nativeEvent.button !== 0) return;

      const down = downRef.current;
      downRef.current = null;
      if (!down) return;

      const dx = e.nativeEvent.clientX - down.x;
      const dy = e.nativeEvent.clientY - down.y;
      if (Math.abs(dx) > DRAG_THRESHOLD_PX || Math.abs(dy) > DRAG_THRESHOLD_PX) {
        // Pointer moved too far — this is a drag (orbit), not a click. Ignore.
        return;
      }

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
    <group
      onPointerDown={enabled ? handlePointerDown : undefined}
      onPointerUp={enabled ? handlePointerUp : undefined}
    >
      {children}
    </group>
  );
}
