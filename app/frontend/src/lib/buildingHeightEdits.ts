import type { PendingEditDto } from '../api';

export interface BuildingHeightValue {
  buildingId: number;
  height: number;
  minHeight: number;
}

export interface BuildingHeightOverride {
  height_m: number;
  min_height_m?: number;
}

function finiteNumber(value: unknown): number | null {
  if (value == null || typeof value === 'boolean') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function readBuildingHeightValue(feature: any): BuildingHeightValue | null {
  const props = feature?.properties;
  const buildingId = finiteNumber(props?.idx);
  const height = finiteNumber(props?.height);
  if (buildingId === null || height === null) return null;
  const minHeight = finiteNumber(props?.min_height) ?? 0;
  return { buildingId, height, minHeight };
}

function isHeightEdit(value: unknown): value is {
  kind: 'set_building_height';
  building_ids: unknown[];
  height_m: unknown;
  min_height_m?: unknown;
} {
  const edit = value as any;
  return edit?.kind === 'set_building_height' && Array.isArray(edit.building_ids);
}

export function pendingBuildingHeightOverrides(
  pendingEdits: readonly unknown[],
): Map<number, BuildingHeightOverride> {
  const overrides = new Map<number, BuildingHeightOverride>();
  for (const edit of pendingEdits) {
    if (!isHeightEdit(edit)) continue;
    const height = finiteNumber(edit.height_m);
    if (height === null) continue;
    const minHeight = edit.min_height_m === undefined ? null : finiteNumber(edit.min_height_m);
    const override: BuildingHeightOverride = { height_m: height };
    if (minHeight !== null) override.min_height_m = minHeight;
    for (const rawId of edit.building_ids) {
      const buildingId = finiteNumber(rawId);
      if (buildingId !== null) overrides.set(buildingId, override);
    }
  }
  return overrides;
}

export function buildingHeightLabelValue(
  feature: any,
  overrides: Map<number, BuildingHeightOverride>,
): BuildingHeightValue | null {
  const committed = readBuildingHeightValue(feature);
  if (!committed) return null;
  const override = overrides.get(committed.buildingId);
  if (!override) return committed;
  const height = finiteNumber(override.height_m);
  if (height === null) return committed;
  const minHeight = override.min_height_m === undefined
    ? committed.minHeight
    : finiteNumber(override.min_height_m) ?? committed.minHeight;
  return { buildingId: committed.buildingId, height, minHeight };
}

export function validateBuildingHeightInput(
  height: number,
  useMinHeight: boolean,
  minHeight: number,
): { ok: true; height: number; minHeight?: number } | { ok: false; error: string } {
  if (!Number.isFinite(height) || height <= 0) {
    return { ok: false, error: 'Height must be greater than 0.' };
  }
  if (!useMinHeight) return { ok: true, height };
  if (!Number.isFinite(minHeight) || minHeight < 0 || minHeight >= height) {
    return { ok: false, error: 'Min height / base must be at least 0 and less than height.' };
  }
  return { ok: true, height, minHeight };
}

export function toSetBuildingHeightDto(
  buildingIds: number[],
  height: number,
  minHeight?: number,
): PendingEditDto {
  const dto: PendingEditDto = {
    kind: 'set_building_height',
    building_ids: buildingIds,
    height_m: height,
  };
  if (minHeight !== undefined) dto.min_height_m = minHeight;
  return dto;
}