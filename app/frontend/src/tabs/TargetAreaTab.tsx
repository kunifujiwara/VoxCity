import React, { useState, useCallback } from 'react';
import MapPicker from '../components/MapPicker';
import { geocodeCity } from '../api';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { targetAreaActionLabel } from './guidedTabState';

interface TargetAreaTabProps {
  rectangle: number[][] | null;
  onRectangleChange: (vertices: number[][]) => void;
}

const TargetAreaTab: React.FC<TargetAreaTabProps> = ({ rectangle, onRectangleChange }) => {
  const [areaMethod, setAreaMethod] = useState<'draw' | 'coordinates'>('draw');
  const [selectionMode, setSelectionMode] = useState<'draw' | 'dimensions' | 'rotated'>('draw');
  const [cityName, setCityName] = useState('Tokyo');
  const [mapCenter, setMapCenter] = useState<[number, number]>([35.681236, 139.767125]);
  const [mapZoom, setMapZoom] = useState(14);
  const [widthM, setWidthM] = useState(1250);
  const [heightM, setHeightM] = useState(1250);
  const [rotationDeg, setRotationDeg] = useState(0);
  const [loading, setLoading] = useState(false);
  const [coords, setCoords] = useState({
    sw_lon: 139.761, sw_lat: 35.676,
    nw_lon: 139.761, nw_lat: 35.686,
    ne_lon: 139.773, ne_lat: 35.686,
    se_lon: 139.773, se_lat: 35.676,
  });

  const handleLoadMap = async () => {
    if (!cityName.trim()) return;
    setLoading(true);
    try {
      const geo = await geocodeCity(cityName.trim());
      setMapCenter([geo.lat, geo.lon]);
      setMapZoom(14);
    } catch (err: any) {
      console.error(err);
    }
    setLoading(false);
  };

  const handleSetCoords = () => {
    onRectangleChange([
      [coords.sw_lon, coords.sw_lat],
      [coords.nw_lon, coords.nw_lat],
      [coords.ne_lon, coords.ne_lat],
      [coords.se_lon, coords.se_lat],
    ]);
  };

  const handleRectangleChange = useCallback((vertices: number[][]) => {
    onRectangleChange(vertices);
  }, [onRectangleChange]);

  return (
    <div className="two-col">
      {/* Left panel – controls */}
      <GuidedPanel
        title="Target Area"
        subtitle="Choose the city area used by model generation."
        status={rectangle ? <GuidedStatus tone="success">Target area is ready.</GuidedStatus> : undefined}
        footer={(
          <GuidedFooter>
            <button
              className="btn btn-primary"
              onClick={areaMethod === 'draw' ? handleLoadMap : handleSetCoords}
              disabled={areaMethod === 'draw' && loading}
              type="button"
            >
              {loading && areaMethod === 'draw' && <span className="spinner" />}
              {targetAreaActionLabel(areaMethod, loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection label="Define area by">
          <ChoiceGroup
            ariaLabel="Target area input method"
            value={areaMethod}
            onChange={setAreaMethod}
            options={[
              { id: 'draw', label: 'Draw on map' },
              { id: 'coordinates', label: 'Enter coordinates' },
            ]}
          />
        </GuidedSection>

        {areaMethod === 'draw' && (
          <>
            <GuidedSection>
              <div className="form-group">
                <label>City name</label>
                <input
                  type="text"
                  value={cityName}
                  onChange={(e) => setCityName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleLoadMap()}
                />
              </div>
            </GuidedSection>

            <GuidedSection label="Drawing mode">
              <ChoiceGroup
                ariaLabel="Target area drawing mode"
                value={selectionMode}
                onChange={setSelectionMode}
                columns={1}
                options={[
                  { id: 'draw', label: 'Free hand' },
                  { id: 'rotated', label: 'Rotated free hand' },
                  { id: 'dimensions', label: 'Set dimensions' },
                ]}
              />
            </GuidedSection>

            {selectionMode === 'dimensions' && (
              <GuidedSection>
                <div className="form-row">
                  <div>
                    <label>Width (m)</label>
                    <input
                      type="number"
                      value={widthM}
                      min={50}
                      max={20000}
                      step={50}
                      onChange={(e) => setWidthM(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label>Height (m)</label>
                    <input
                      type="number"
                      value={heightM}
                      min={50}
                      max={20000}
                      step={50}
                      onChange={(e) => setHeightM(Number(e.target.value))}
                    />
                  </div>
                  <div>
                    <label>Rotation (°)</label>
                    <input
                      type="number"
                      value={rotationDeg}
                      min={-90}
                      max={90}
                      step={1}
                      onChange={(e) => setRotationDeg(Number(e.target.value))}
                    />
                  </div>
                </div>
              </GuidedSection>
            )}
          </>
        )}

        {areaMethod === 'coordinates' && (
          <GuidedSection label="Rectangle vertices">
            {(['sw', 'nw', 'ne', 'se'] as const).map((corner) => (
              <div className="form-row" key={corner}>
                <div>
                  <label>{corner.toUpperCase()} Lon</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={(coords as any)[`${corner}_lon`]}
                    onChange={(e) =>
                      setCoords({ ...coords, [`${corner}_lon`]: parseFloat(e.target.value) })
                    }
                  />
                </div>
                <div>
                  <label>{corner.toUpperCase()} Lat</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={(coords as any)[`${corner}_lat`]}
                    onChange={(e) =>
                      setCoords({ ...coords, [`${corner}_lat`]: parseFloat(e.target.value) })
                    }
                  />
                </div>
              </div>
            ))}
          </GuidedSection>
        )}
      </GuidedPanel>

      {/* Right panel – map */}
      <div className="panel" style={{ padding: 0, position: 'relative' }}>
        {rectangle && <div className="panel-overlay-status">Target area set</div>}
        <div className="map-container" style={{ height: '100%', minHeight: 500 }}>
          <MapPicker
            center={mapCenter}
            zoom={mapZoom}
            rectangle={rectangle}
            selectionMode={selectionMode}
            widthM={widthM}
            heightM={heightM}
            rotationDeg={rotationDeg}
            onRectangleChange={handleRectangleChange}
          />
        </div>
      </div>
    </div>
  );
};

export default TargetAreaTab;
