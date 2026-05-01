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

export type SceneGeometry = SceneGeometryResponse;
export type OverlayGeometry = OverlayGeometryResponse;
export type MeshChunk = MeshChunkDto;

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
}
