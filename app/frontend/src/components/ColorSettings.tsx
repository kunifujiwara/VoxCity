import React, { useState } from 'react';
import { COLORMAPS } from '../constants';

interface ColorSettingsProps {
  colormap: string;
  onColormapChange: (value: string) => void;
  vmin: number;
  onVminChange: (value: number) => void;
  vmax: number | string;
  onVmaxChange: (value: number | string) => void;
  /** Whether vmax is a free-text field (e.g. "empty = auto"). Defaults to false. */
  vmaxAsText?: boolean;
}

const ColorSettings: React.FC<ColorSettingsProps> = ({
  colormap,
  onColormapChange,
  vmin,
  onVminChange,
  vmax,
  onVmaxChange,
  vmaxAsText = false,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="expander">
      <div className="expander-header" onClick={() => setOpen(!open)}>
        Color Settings <span>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div className="expander-body">
          <div className="form-group">
            <label>Colormap</label>
            <select value={colormap} onChange={(e) => onColormapChange(e.target.value)}>
              {COLORMAPS.map((cm) => (
                <option key={cm} value={cm}>{cm}</option>
              ))}
            </select>
          </div>
          <div className="form-row">
            <div>
              <label>vmin</label>
              <input
                type="number"
                value={vmin}
                step={0.1}
                onChange={(e) => onVminChange(Number(e.target.value))}
              />
            </div>
            <div>
              <label>{vmaxAsText ? 'vmax (empty = auto)' : 'vmax'}</label>
              {vmaxAsText ? (
                <input
                  type="text"
                  value={vmax}
                  onChange={(e) => onVmaxChange(e.target.value)}
                />
              ) : (
                <input
                  type="number"
                  value={vmax as number}
                  step={0.1}
                  onChange={(e) => onVmaxChange(Number(e.target.value))}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ColorSettings;
