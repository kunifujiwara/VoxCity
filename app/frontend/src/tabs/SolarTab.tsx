/**
 * Solar tab — runs a solar-radiation simulation and previews the result on
 * the new R3F `<SceneViewer>` (overlay + zone outlines + colorbar).
 *
 * The legacy `figureJson` / `useZoneOverlay` / `<ThreeViewer>` path was
 * removed during the Three.js migration (Chunk 4). The tab still accepts
 * `figureJson` / `onFigureChange` props for App-level state compatibility,
 * but they are intentionally unused.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getModelGeo, ModelGeoResult, runSolar } from '../api';
import { SceneViewer } from '../three';
import ColorSettings from '../components/ColorSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import ZoneStatsTable from '../components/ZoneStatsTable';
import { lonLatToWorldXY } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { Zone } from '../types/zones';

interface SolarTabProps {
  hasModel: boolean;
  /** @deprecated kept for App-level state compatibility, unused here. */
  figureJson: string;
  /** @deprecated kept for App-level state compatibility, unused here. */
  onFigureChange: (json: string) => void;
  zones: Zone[];
  simRunNonce: number;
  onSimRun: () => void;
}

const SolarTab: React.FC<SolarTabProps> = ({
  hasModel,
  zones,
  simRunNonce,
  onSimRun,
}) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, simRunNonce);
  const [calcType, setCalcType] = useState<'instantaneous' | 'cumulative'>('instantaneous');
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [calcDate, setCalcDate] = useState('01-01');
  const [calcTime, setCalcTime] = useState('12:00:00');
  const [startTime, setStartTime] = useState('01-01 01:00:00');
  const [endTime, setEndTime] = useState('01-31 23:00:00');
  const [colormap, setColormap] = useState('magma');
  const [vmin, setVmin] = useState<number>(0);
  const [vmax, setVmax] = useState<string>('');
  const [hiddenClasses, setHiddenClasses] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSimResult, setHasSimResult] = useState(false);

  // Geometry for the lon/lat -> world projection used by zone outlines.
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => { if (!cancelled) setGeo(g); })
      .catch(() => { /* ignore — zones just won't project */ });
    return () => { cancelled = true; };
  }, [hasModel]);
  const lonLatToXY = useMemo(() => lonLatToWorldXY(geo), [geo]);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: any = {
        calc_type: calcType,
        analysis_target: analysisTarget,
        epw_source: 'default',
        colormap,
        vmin,
        vmax: vmax ? parseFloat(vmax) : null,
        hidden_classes: Array.from(hiddenClasses),
      };
      if (calcType === 'instantaneous') {
        params.calc_time = `${calcDate} ${calcTime}`;
      } else {
        params.start_time = startTime;
        params.end_time = endTime;
      }
      await runSolar(params);
      setHasSimResult(true);
      onSimRun();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const vmaxNum = vmax ? parseFloat(vmax) : null;

  return (
    <div className="two-col">
      <div className="panel">
        <h2>Solar Radiation Analysis</h2>

        <div className="form-group">
          <label>Calculation Type</label>
          <div className="radio-group">
            <label>
              <input type="radio" checked={calcType === 'instantaneous'} onChange={() => setCalcType('instantaneous')} />
              Instantaneous
            </label>
            <label>
              <input type="radio" checked={calcType === 'cumulative'} onChange={() => setCalcType('cumulative')} />
              Cumulative
            </label>
          </div>
        </div>

        <div className="form-group">
          <label>Analysis Target</label>
          <div className="radio-group">
            <label>
              <input type="radio" checked={analysisTarget === 'ground'} onChange={() => setAnalysisTarget('ground')} />
              Ground Level
            </label>
            <label>
              <input type="radio" checked={analysisTarget === 'building'} onChange={() => setAnalysisTarget('building')} />
              Building Surfaces
            </label>
          </div>
        </div>

        {calcType === 'instantaneous' ? (
          <div className="form-row">
            <div>
              <label>Date (MM-DD)</label>
              <input type="text" value={calcDate} onChange={(e) => setCalcDate(e.target.value)} />
            </div>
            <div>
              <label>Time (HH:MM:SS)</label>
              <input type="text" value={calcTime} onChange={(e) => setCalcTime(e.target.value)} />
            </div>
          </div>
        ) : (
          <>
            <div className="form-group">
              <label>Start (MM-DD HH:MM:SS)</label>
              <input type="text" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
            </div>
            <div className="form-group">
              <label>End (MM-DD HH:MM:SS)</label>
              <input type="text" value={endTime} onChange={(e) => setEndTime(e.target.value)} />
            </div>
          </>
        )}

        <ColorSettings
          colormap={colormap}
          onColormapChange={setColormap}
          vmin={vmin}
          onVminChange={setVmin}
          vmax={vmax}
          onVmaxChange={(v) => setVmax(String(v))}
          vmaxAsText
        />

        <VoxelClassVisibility
          hiddenClasses={hiddenClasses}
          onHiddenClassesChange={setHiddenClasses}
        />

        <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Running...' : 'Run Simulation'}
        </button>

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}

        {zones.length > 0 && (
          <>
            <div className="form-group" style={{ marginTop: '0.75rem' }}>
              <label>
                <input
                  type="checkbox"
                  checked={showZones3D}
                  onChange={(e) => setShowZones3D(e.target.checked)}
                />{' '}
                Show zones in 3D
              </label>
            </div>
            <ZoneStatsTable zones={zones} stats={zoneStats} loading={zoneStatsLoading} />
          </>
        )}
      </div>

      <div className="panel" style={{ position: 'relative', minHeight: 400 }}>
        <SceneViewer
          geometryToken={hasModel ? 'loaded' : 'none'}
          downsample={1}
          colorScheme="grayscale"
          simKind={hasSimResult ? 'solar' : null}
          simToken={simRunNonce}
          colormap={colormap}
          vmin={vmin}
          vmax={vmaxNum}
          zones={zones}
          lonLatToXY={lonLatToXY}
          showZones={showZones3D}
          hiddenClasses={hiddenClasses}
        />
      </div>
    </div>
  );
};

export default SolarTab;
