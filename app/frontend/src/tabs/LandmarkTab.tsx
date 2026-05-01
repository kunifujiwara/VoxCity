/**
 * Landmark tab — runs a landmark-visibility simulation and previews the
 * result on the new R3F `<SceneViewer>`.
 *
 * Migrated from the legacy `figureJson` + `<ThreeViewer>` flow during
 * Chunk 4 of the Three.js migration.
 *
 * Building selection: while no sim result is shown the user can click
 * anywhere in the scene; the click hit point is matched (by 2D XY
 * proximity) against the cached building centroid list to toggle a
 * landmark id. Box selection from the legacy viewer is intentionally
 * out of scope for this chunk.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BuildingInfo,
  getBuildingAt,
  getBuildingsList,
  getBuildingHighlight,
  getModelGeo,
  ModelGeoResult,
  runLandmark,
  type MeshChunkDto,
} from '../api';
import { SceneViewer } from '../three';
import type { PickResult } from '../three/types';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import ZoneStatsTable from '../components/ZoneStatsTable';
import { lonLatToWorldXY } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { Zone } from '../types/zones';

interface LandmarkTabProps {
  hasModel: boolean;
  /** @deprecated kept for App-level state compatibility, unused here. */
  figureJson: string;
  /** @deprecated kept for App-level state compatibility, unused here. */
  onFigureChange: (json: string) => void;
  zones: Zone[];
  simRunNonce: number;
  onSimRun: () => void;
}

const LandmarkTab: React.FC<LandmarkTabProps> = ({
  hasModel,
  zones,
  simRunNonce,
  onSimRun,
}) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, simRunNonce);
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [landmarkIdsText, setLandmarkIdsText] = useState('');
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
  const [showingSimResult, setShowingSimResult] = useState(false);

  // Selection state
  const [selectedBuildingIds, setSelectedBuildingIds] = useState<number[]>([]);
  const [buildings, setBuildings] = useState<BuildingInfo[]>([]);

  // Geometry for zone projection.
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      setBuildings([]);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => { if (!cancelled) setGeo(g); })
      .catch(() => {});
    getBuildingsList()
      .then((r) => { if (!cancelled) setBuildings(r.buildings); })
      .catch((err) => { if (!cancelled) setError(`Failed to load buildings: ${err.message}`); });
    return () => { cancelled = true; };
  }, [hasModel]);
  const lonLatToXY = useMemo(() => lonLatToWorldXY(geo), [geo]);

  // Bidirectional sync: text -> selection
  const handleIdsTextChange = useCallback((text: string) => {
    setLandmarkIdsText(text);
    const ids = text
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n) && n > 0);
    setSelectedBuildingIds(ids);
  }, []);

  // Bidirectional sync: selection -> text
  const setSelection = useCallback((ids: number[]) => {
    setSelectedBuildingIds(ids);
    setLandmarkIdsText(ids.join(', '));
  }, []);

  const handleClearSelection = useCallback(() => setSelection([]), [setSelection]);

  const handleBackToSelection = useCallback(() => {
    setShowingSimResult(false);
  }, []);

  // Click pick: ask the backend for the building id at the clicked world XY.
  // This works for any face (roof OR walls), unlike a 2-D centroid lookup
  // which fails when the wall is closer to a neighbour's centroid.
  const handlePick = useCallback((hit: PickResult | null) => {
    if (showingSimResult) return;
    if (!hit) return;
    const [px, py] = hit.point;
    getBuildingAt(px, py)
      .then((r) => {
        const bid = r.building_id;
        if (bid == null) return;
        setSelectedBuildingIds((prev) => {
          const next = prev.includes(bid)
            ? prev.filter((i) => i !== bid)
            : [...prev, bid];
          setLandmarkIdsText(next.join(', '));
          return next;
        });
      })
      .catch(() => {});
  }, [showingSimResult]);

  // Fetch highlight geometry whenever the selection changes.
  const [highlightChunks, setHighlightChunks] = useState<MeshChunkDto[] | null>(null);
  useEffect(() => {
    if (!hasModel || selectedBuildingIds.length === 0) {
      setHighlightChunks(null);
      return;
    }
    let cancelled = false;
    getBuildingHighlight(selectedBuildingIds)
      .then((r) => { if (!cancelled) setHighlightChunks(r.chunks); })
      .catch(() => { if (!cancelled) setHighlightChunks(null); });
    return () => { cancelled = true; };
  }, [hasModel, selectedBuildingIds]);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const ids = landmarkIdsText
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n));

      await runLandmark({
        analysis_target: analysisTarget,
        landmark_ids: ids,
        view_point_height: 1.5,
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
      setShowingSimResult(true);
      onSimRun();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <div className="panel">
        <h2>Landmark Visibility Analysis</h2>

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
          <label>Select Landmark Buildings</label>
          {!showingSimResult && (
            <div className="selection-toolbar">
              <span className="hint">Click a building in the 3D viewer to toggle it.</span>
              <button
                className="selection-toolbar-btn clear-btn"
                onClick={handleClearSelection}
                title="Clear all selections"
                disabled={selectedBuildingIds.length === 0}
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {selectedBuildingIds.length > 0 && (
          <div className="form-group">
            <label>Selected Buildings ({selectedBuildingIds.length})</label>
            <div className="building-chips">
              {selectedBuildingIds.map((id) => (
                <span key={id} className="building-chip">
                  {id}
                  <button
                    className="chip-remove"
                    onClick={() => setSelection(selectedBuildingIds.filter((i) => i !== id))}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="form-group">
          <label>Landmark Building IDs (comma-separated, empty = center building)</label>
          <input
            type="text"
            value={landmarkIdsText}
            onChange={(e) => handleIdsTextChange(e.target.value)}
            placeholder="e.g. 12, 34, 56"
          />
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

        {showingSimResult && hasSimResult && (
          <button className="btn btn-secondary" onClick={handleBackToSelection} disabled={loading} style={{ marginBottom: '0.5rem' }}>
            Back to Selection
          </button>
        )}

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
          simKind={showingSimResult ? 'landmark' : null}
          simToken={simRunNonce}
          colormap={colormap}
          vmin={vmin}
          vmax={vmax}
          zones={zones}
          lonLatToXY={lonLatToXY}
          showZones={showZones3D}
          hiddenClasses={hiddenClasses}
          onPick={!showingSimResult ? handlePick : undefined}
          highlightChunks={!showingSimResult ? highlightChunks : null}
        />
      </div>
    </div>
  );
};

export default LandmarkTab;
