/**
 * Compact per-layer visibility toggles for imported DXF auxiliary lines.
 * Reads the (backend-authoritative) layers from `geo.auxiliary_lines` and
 * toggles the shared visibility map. Renders nothing when there are no lines.
 */
import React from 'react';
import type { ModelGeoResult } from '../api';
import { groupAuxLineLayers, isAuxLayerVisible } from '../lib/auxiliaryLines';

interface Props {
  geo: ModelGeoResult | null;
  visibility: Record<string, Record<string, boolean>>;
  onToggle: (fileName: string, layer: string, visible: boolean) => void;
  /** Remove all auxiliary lines from a file (backend delete + geo refresh). */
  onRemoveFile?: (fileName: string) => void;
  /** Extra styles merged into the row container (e.g. marginLeft: 'auto'). */
  style?: React.CSSProperties;
}

const AuxiliaryLinesControl: React.FC<Props> = ({ geo, visibility, onToggle, onRemoveFile, style }) => {
  const lines = geo?.auxiliary_lines ?? [];
  if (lines.length === 0) return null;

  const grouped = groupAuxLineLayers(lines);

  return (
    <div
      className="aux-lines-control"
      style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4, fontSize: '0.72rem', ...style }}
    >
      <span style={{ opacity: 0.6, marginRight: 2 }} title="Imported DXF auxiliary lines">DXF</span>
      {grouped.map(({ fileName, layers }) => (
        <React.Fragment key={fileName}>
          {layers.map(({ layer, color }) => {
            const visible = isAuxLayerVisible(visibility, fileName, layer);
            return (
              <button
                key={layer}
                type="button"
                title={`${fileName} · ${layer}`}
                className={`btn btn-xs${visible ? ' btn-primary' : ' btn-ghost'}`}
                onClick={() => onToggle(fileName, layer, !visible)}
              >
                <span
                  style={{
                    width: 8, height: 8, background: color, borderRadius: 2,
                    display: 'inline-block', border: '1px solid #0003', marginRight: 4,
                  }}
                />
                {layer}
              </button>
            );
          })}
          {onRemoveFile && (
            <button
              type="button"
              className="btn btn-xs btn-ghost"
              title={`Remove ${fileName}`}
              aria-label={`Remove ${fileName}`}
              onClick={() => onRemoveFile(fileName)}
            >
              ×
            </button>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default AuxiliaryLinesControl;
