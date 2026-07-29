import type { AuxiliaryLineDto } from '../api';

/** A DXF layer is visible unless explicitly toggled off in the visibility map. */
export function isAuxLayerVisible(
  visibility: Record<string, Record<string, boolean>>,
  fileName: string,
  layer: string,
): boolean {
  return visibility[fileName]?.[layer] !== false;
}

/**
 * Group auxiliary lines by file, then by unique layer (first-seen color),
 * preserving encounter order for stable UI rendering.
 */
export function groupAuxLineLayers(
  lines: AuxiliaryLineDto[],
): { fileName: string; layers: { layer: string; color: string }[] }[] {
  const byFile = new Map<string, Map<string, string>>();
  for (const ln of lines) {
    if (!byFile.has(ln.file_name)) byFile.set(ln.file_name, new Map());
    const layers = byFile.get(ln.file_name)!;
    if (!layers.has(ln.layer)) layers.set(ln.layer, ln.color);
  }
  return [...byFile.entries()].map(([fileName, layers]) => ({
    fileName,
    layers: [...layers.entries()].map(([layer, color]) => ({ layer, color })),
  }));
}
