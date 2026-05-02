/**
 * View tab — runs a view-index simulation (sky/green/custom) and previews
 * the result on the new R3F `<SceneViewer>`.
 *
 * Migrated from the legacy `figureJson` + `<ThreeViewer>` flow during
 * Chunk 4 of the Three.js migration.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { getModelGeo, ModelGeoResult, runView } from '../api';
import { SceneViewer } from '../three';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import ZoneStatsTable from '../components/ZoneStatsTable';
import { CUSTOM_CLASSES } from '../constants';
import { lonLatToUvM } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { Zone } from '../types/zones';

interface ViewTabProps {
  hasModel: boolean;
  /** @deprecated kept for App-level state compatibility, unused here. */
  figureJson: string;
  /** @deprecated kept for App-level state compatibility, unused here. */
  onFigureChange: (json: string) => void;
  zones: Zone[];
  simRunNonce: number;
  onSimRun: () => void;
}

const ViewTab: React.FC<ViewTabProps> = ({ hasModel, zones, simRunNonce, onSimRun }) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, simRunNonce);
  const [viewType, setViewType] = useState('green');
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [viewPointHeight, setViewPointHeight] = useState(1.5);
  const [customClasses, setCustomClasses] = useState<Set<number>>(new Set());
  const [inclusionMode, setInclusionMode] = useState(true);
  const [nAzimuth, setNAzimuth] = useState(60);
  const [nElevation, setNElevation] = useState(10);
  const [elevMin, setElevMin] = useState(-30);
  const [elevMax, setElevMax] = useState(30);
  const [colormap, setColormap] = useState('viridis');
  const [vmin, setVmin] = useState(0);
  const [vmax, setVmax] = useState(1);
  const [hiddenClasses, setHiddenClasses] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSimResult, setHasSimResult] = useState(false);

  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => { if (!cancelled) setGeo(g); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [hasModel]);
  const lonLatToXY = useMemo(() => lonLatToUvM(geo), [geo]);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const toggleClass = (id: number) => {
    const next = new Set(customClasses);
    next.has(id) ? next.delete(id) : next.add(id);
    setCustomClasses(next);
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      await runView({
        view_type: viewType,
        analysis_target: analysisTarget,
        view_point_height: viewPointHeight,
        custom_classes: Array.from(customClasses),
        inclusion_mode: inclusionMode,
        n_azimuth: nAzimuth,
        n_elevation: nElevation,
        elevation_min_degrees: elevMin,
        elevation_max_degrees: elevMax,
        colormap,
        vmin,
        vmax,
        hidden_classes: Array.from(hiddenClasses),
      });
      setHasSimResult(true);
      onSimRun();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <div className="panel">
        <h2>View Index Analysis</h2>

        <div className="form-group">
          <label>View Type</label>
          <select value={viewType} onChange={(e) => setViewType(e.target.value)}>
            <option value="green">Green View Index</option>
            <option value="sky">Sky View Index</option>
            <option value="custom">Custom (Select Classes)</option>
          </select>
        </div>

        {viewType === 'custom' && (
          <div className="form-group" style={{ maxHeight: 200, overflowY: 'auto', padding: '0.5rem', border: '1px solid var(--vc-ring)', borderRadius: 6 }}>
            <div className="radio-group" style={{ marginBottom: '0.5rem' }}>
              <label>
                <input type="radio" checked={inclusionMode} onChange={() => setInclusionMode(true)} />
                Inclusion
              </label>
              <label>
                <input type="radio" checked={!inclusionMode} onChange={() => setInclusionMode(false)} />
                Exclusion
              </label>
            </div>
            {CUSTOM_CLASSES.map((cls) => (
              <div className="checkbox-row" key={cls.id}>
                <input
                  type="checkbox"
                  checked={customClasses.has(cls.id)}
                  onChange={() => toggleClass(cls.id)}
                />
                <span>{cls.label}</span>
              </div>
            ))}
          </div>
        )}

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

        <div className="form-group">
          <label>View Point Height (m)</label>
          <input type="number" value={viewPointHeight} min={0} max={10} step={0.5} onChange={(e) => setViewPointHeight(Number(e.target.value))} />
        </div>

        <SamplingSettings
          nAzimuth={nAzimuth}
          onNAzimuthChange={setNAzimuth}
          nElevation={nElevation}
          onNElevationChange={setNElevation}
          elevMin={elevMin}
          onElevMinChange={setElevMin}
          elevMax={elevMax}
          onElevMaxChange={setElevMax}
          showElevationRange={analysisTarget === 'ground'}
        />

        <ColorSettings
          colormap={colormap}
          onColormapChange={setColormap}
          vmin={vmin}
          onVminChange={setVmin}
          vmax={vmax}
          onVmaxChange={(v) => setVmax(Number(v))}
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
          simKind={hasSimResult ? 'view' : null}
          simToken={simRunNonce}
          colormap={colormap}
          vmin={vmin}
          vmax={vmax}
          zones={zones}
          lonLatToXY={lonLatToXY}
          showZones={showZones3D}
          hiddenClasses={hiddenClasses}
        />
      </div>
    </div>
  );
};

export default ViewTab;
