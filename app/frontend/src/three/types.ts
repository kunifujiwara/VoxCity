/**
 * Shared types for the R3F scene viewer.
 *
 * Mirrors the backend Pydantic ``MeshChunk`` / ``SceneGeometryResponse`` /
 * ``OverlayGeometryResponse`` payloads from `app/backend/scene_geometry.py`.
 */
import type {
  MeshChunkDto,
  OverlayGeometryResponse,
  SceneGeometryResponse,
} from '../api';
import type { WallOrientation } from '../types/zones';

export type SceneGeometry = SceneGeometryResponse;
export type OverlayGeometry = OverlayGeometryResponse;
export type MeshChunk = MeshChunkDto;

/** Metadata for a single picked building surface face. */
export interface SurfaceFaceMeta {
  faceKey: string;
  buildingId: number;
  surfaceKind: 'roof' | 'wall' | 'bottom' | 'other';
  orientation?: WallOrientation | null;
  isWindow?: boolean;
}

/** A picked element returned by the `<Picker/>` overlay. */
export interface PickResult {
  /** Source target the chunk came from. */
  target: 'ground' | 'building';
  /** Cell `(i, j)` for ground sims, ``null`` otherwise. */
  cell: [number, number] | null;
  /** Building id for building sims, ``null`` otherwise. */
  buildingId: number | null;
  /** World-space hit point in metres (Z-up). */
  point: [number, number, number];
  /** Surface face metadata when picking from a surface-tagged mesh. */
  surface?: SurfaceFaceMeta | null;
}
