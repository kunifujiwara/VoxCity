import React, { useState, useCallback, useMemo } from 'react';
import { PenTool, Hash, RotateCw, Ruler, Check } from 'lucide-react';
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

  const summaryText = useMemo(() => {
    if (!rectangle || rectangle.length !== 4) return null;
    const lons = rectangle.map(([lon]) => lon);
    const lats = rectangle.map(([, lat]) => lat);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const midLat = (minLat + maxLat) / 2;
    const width = (maxLon - minLon) * 111000 * Math.cos((midLat * Math.PI) / 180);
    const height = (maxLat - minLat) * 111000;
    const areaKm2 = Math.abs(width * height) / 1e6;
    const widthKm = Math.abs(width) / 1000;
    const heightKm = Math.abs(height) / 1000;
    return `${widthKm.toFixed(2)} × ${heightKm.toFixed(2)} km · ${areaKm2.toFixed(2)} km²`;
  }, [rectangle]);

  let idx = 0;

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
              <Check size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {targetAreaActionLabel(areaMethod, loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection label="LOCATION" index={++idx}>
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

        <GuidedSection label="DEFINE TARGET AREA" index={++idx}>
          <ChoiceGroup
            variant="checks"
            ariaLabel="Target area input method"
            value={areaMethod}
            onChange={setAreaMethod}
            options={[
              { id: 'draw', label: 'Map draw', icon: PenTool },
              { id: 'coordinates', label: 'Coordinates', icon: Hash },
            ]}
          />
        </GuidedSection>

        {areaMethod === 'draw' && (
          <GuidedSection label="DRAWING MODE" index={++idx}>
            <ChoiceGroup
              variant="checks"
              ariaLabel="Target area drawing mode"
              value={selectionMode}
              onChange={setSelectionMode}
              columns={1}
              options={[
                { id: 'draw', label: 'Free hand', icon: PenTool },
                { id: 'rotated', label: 'Rotated', icon: RotateCw },
                { id: 'dimensions', label: 'Set dimensions', icon: Ruler },
              ]}
            />
          </GuidedSection>
        )}

        {areaMethod === 'draw' && selectionMode === 'dimensions' && (
          <GuidedSection label="DIMENSIONS" index={++idx}>
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

        {areaMethod === 'coordinates' && (
          <GuidedSection label="RECTANGLE VERTICES" index={++idx}>
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

        {rectangle && summaryText && (
          <GuidedSection label="SUMMARY" index={++idx}>
            <div style={{ color: 'var(--vc-muted)', fontSize: '0.85rem' }}>{summaryText}</div>
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
