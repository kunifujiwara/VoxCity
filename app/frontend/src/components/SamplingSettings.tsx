import React, { useState } from 'react';

interface SamplingSettingsProps {
  nAzimuth: number;
  onNAzimuthChange: (value: number) => void;
  nElevation: number;
  onNElevationChange: (value: number) => void;
  elevMin: number;
  onElevMinChange: (value: number) => void;
  elevMax: number;
  onElevMaxChange: (value: number) => void;
  /** Hide elevation min/max when analysis target is "building" */
  showElevationRange?: boolean;
}

const SamplingSettings: React.FC<SamplingSettingsProps> = ({
  nAzimuth,
  onNAzimuthChange,
  nElevation,
  onNElevationChange,
  elevMin,
  onElevMinChange,
  elevMax,
  onElevMaxChange,
  showElevationRange = true,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="expander">
      <div className="expander-header" onClick={() => setOpen(!open)}>
        Sampling Settings <span>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div className="expander-body">
          <div className="form-row">
            <div>
              <label>N_azimuth</label>
              <input
                type="number"
                value={nAzimuth}
                min={1}
                max={360}
                onChange={(e) => onNAzimuthChange(Number(e.target.value))}
              />
            </div>
            <div>
              <label>N_elevation</label>
              <input
                type="number"
                value={nElevation}
                min={1}
                max={180}
                onChange={(e) => onNElevationChange(Number(e.target.value))}
              />
            </div>
          </div>
          {showElevationRange && (
            <div className="form-row">
              <div>
                <label>Elev min (°)</label>
                <input
                  type="number"
                  value={elevMin}
                  onChange={(e) => onElevMinChange(Number(e.target.value))}
                />
              </div>
              <div>
                <label>Elev max (°)</label>
                <input
                  type="number"
                  value={elevMax}
                  onChange={(e) => onElevMaxChange(Number(e.target.value))}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SamplingSettings;
