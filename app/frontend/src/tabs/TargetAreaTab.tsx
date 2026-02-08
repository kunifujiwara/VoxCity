import React, { useState, useCallback } from 'react';
import MapPicker from '../components/MapPicker';
import { geocodeCity } from '../api';

interface TargetAreaTabProps {
  rectangle: number[][] | null;
  onRectangleChange: (vertices: number[][]) => void;
}

const TargetAreaTab: React.FC<TargetAreaTabProps> = ({ rectangle, onRectangleChange }) => {
  const [areaMethod, setAreaMethod] = useState<'draw' | 'coordinates'>('draw');
  const [selectionMode, setSelectionMode] = useState<'draw' | 'dimensions'>('draw');
  const [cityName, setCityName] = useState('Tokyo');
  const [mapCenter, setMapCenter] = useState<[number, number]>([35.681236, 139.767125]);
  const [mapZoom, setMapZoom] = useState(14);
  const [widthM, setWidthM] = useState(1250);
  const [heightM, setHeightM] = useState(1250);
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
      <div className="panel">
        <div className="form-group">
          <div className="radio-group">
            <label>
              <input
                type="radio"
                checked={areaMethod === 'draw'}
                onChange={() => setAreaMethod('draw')}
              />
              Draw on map
            </label>
            <label>
              <input
                type="radio"
                checked={areaMethod === 'coordinates'}
                onChange={() => setAreaMethod('coordinates')}
              />
              Enter coordinates
            </label>
          </div>
        </div>

        {areaMethod === 'draw' && (
          <>
            <div className="form-group">
              <label>City name</label>
              <input
                type="text"
                value={cityName}
                onChange={(e) => setCityName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleLoadMap()}
              />
            </div>

            <div className="form-group">
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    checked={selectionMode === 'draw'}
                    onChange={() => setSelectionMode('draw')}
                  />
                  Free hand
                </label>
                <label>
                  <input
                    type="radio"
                    checked={selectionMode === 'dimensions'}
                    onChange={() => setSelectionMode('dimensions')}
                  />
                  Set dimensions
                </label>
              </div>
            </div>

            {selectionMode === 'dimensions' && (
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
              </div>
            )}

            <button className="btn btn-primary" onClick={handleLoadMap} disabled={loading}>
              {loading && <span className="spinner" />}
              Load Map
            </button>
          </>
        )}

        {areaMethod === 'coordinates' && (
          <>
            <h2>Rectangle Vertices</h2>
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
            <button className="btn btn-primary" onClick={handleSetCoords}>
              Set Rectangle
            </button>
          </>
        )}

        {rectangle && (
          <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>
            Rectangle set ✓
          </div>
        )}
      </div>

      {/* Right panel – map */}
      <div className="panel" style={{ padding: 0 }}>
        <div className="map-container" style={{ height: '100%', minHeight: 500 }}>
          <MapPicker
            center={mapCenter}
            zoom={mapZoom}
            rectangle={rectangle}
            selectionMode={selectionMode}
            widthM={widthM}
            heightM={heightM}
            onRectangleChange={handleRectangleChange}
          />
        </div>
      </div>
    </div>
  );
};

export default TargetAreaTab;
