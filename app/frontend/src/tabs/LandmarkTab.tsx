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
import { Layers, Box, Flag, ArrowLeft } from 'lucide-react';
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
import { lonLatToUvM } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { useSurfaceZoneEdges } from '../hooks/useSurfaceZoneEdges';
import { Zone } from '../types/zones';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { prerequisiteMessageForTab, simulationActionLabel } from './guidedTabState';

interface LandmarkTabProps {
  hasModel: boolean;
  /** @deprecated kept for App-level state compatibility, unused here. */
  figureJson: string;
  /** @deprecated kept for App-level state compatibility, unused here. */
  onFigureChange: (json: string) => void;
  zones: Zone[];
  simRunNonce: number;
  onSimRun: () => void;
  geometryToken?: string | number;
}

const LandmarkTab: React.FC<LandmarkTabProps> = ({
  hasModel,
  zones,
  simRunNonce,
  onSimRun,
  geometryToken,
}) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, 'landmark', simRunNonce);
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

  const { surfaceZoneEdges } = useSurfaceZoneEdges({
    hasModel,
    enabled: showZones3D,
    zones,
  });

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
  const lonLatToXY = useMemo(() => lonLatToUvM(geo), [geo]);

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

  // Click pick: select the building that was clicked.
  // Prefer the direct buildingId from face metadata (set when a building-surface
  // overlay is active), which avoids the backend coordinate lookup entirely.
  // Fall back to getBuildingAt() for static scene geometry (base chunks) where
  // per-face building IDs are not available.
  const handlePick = useCallback((hit: PickResult | null) => {
    if (showingSimResult) return;
    if (!hit) return;

    if (hit.buildingId != null) {
      // Direct face metadata path — always correct, no backend round-trip.
      const bid = hit.buildingId;
      setSelectedBuildingIds((prev) => {
        const next = prev.includes(bid)
          ? prev.filter((i) => i !== bid)
          : [...prev, bid];
        setLandmarkIdsText(next.join(', '));
        return next;
      });
      return;
    }

    // Fallback: ask the backend for the building id at the clicked world XY.
    // Needed for static voxel geometry (roof/wall faces) which do not carry
    // per-face building IDs in their userData.
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

  // Fetch highlight geometry whenever the selection (or rendering style) changes.
  // Once a sim has been run we colour the highlights with the *max* value of
  // the active colormap and ask the renderer to make them emissive, so the
  // selected landmarks stand out alongside the simulation overlay (both for
  // ground-level and building-surface analyses).
  const highlightAsSimResult = showingSimResult || hasSimResult;
  const [highlightChunks, setHighlightChunks] = useState<MeshChunkDto[] | null>(null);
  useEffect(() => {
    if (!hasModel || selectedBuildingIds.length === 0) {
      setHighlightChunks(null);
      return;
    }
    let cancelled = false;
    const opts = highlightAsSimResult
      ? { colormap, emissive: true }
      : undefined;
    getBuildingHighlight(selectedBuildingIds, opts)
      .then((r) => { if (!cancelled) setHighlightChunks(r.chunks); })
      .catch(() => { if (!cancelled) setHighlightChunks(null); });
    return () => { cancelled = true; };
  }, [hasModel, selectedBuildingIds, highlightAsSimResult, colormap]);

  if (!hasModel) {
    const message = prerequisiteMessageForTab('landmark');
    return (
      <div className="two-col">
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
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
      <GuidedPanel
        title="Landmark Visibility"
        subtitle="Select buildings as landmarks and analyse how visible they are."
        status={
          error ? (
            <GuidedStatus tone="error">{error}</GuidedStatus>
          ) : hasSimResult ? (
            <GuidedStatus tone="success">Simulation complete.</GuidedStatus>
          ) : undefined
        }
        footer={(
          <GuidedFooter>
            {showingSimResult && hasSimResult && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleBackToSelection}
                disabled={loading}
              >
                <ArrowLeft size={14} aria-hidden="true" style={{ marginRight: 6 }} />
                Back to selection
              </button>
            )}
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleRun}
              disabled={loading}
            >
              {loading && <span className="spinner" />}
              <Flag size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {simulationActionLabel(loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="ANALYSIS TARGET">
          <ChoiceGroup
            variant="checks"
            ariaLabel="Analysis target"
            value={analysisTarget}
            onChange={setAnalysisTarget}
            options={[
              { id: 'ground', label: 'Ground level', icon: Layers },
              { id: 'building', label: 'Building surfaces', icon: Box },
            ]}
          />
        </GuidedSection>

        <GuidedSection index={2} label="LANDMARK BUILDINGS">
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
        </GuidedSection>

        <GuidedSection index={3} label="SAMPLING">
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
        </GuidedSection>

        <GuidedSection index={4} label="DISPLAY">
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
        </GuidedSection>

        {zones.length > 0 && (
          <GuidedSection index={5} label="ZONES AND RESULTS">
            <div className="form-group">
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
          </GuidedSection>
        )}
      </GuidedPanel>

      <div className="panel visual-panel">
        <div className="visual-frame">
          <SceneViewer
            geometryToken={hasModel ? (geometryToken ?? 'loaded') : 'none'}
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
            highlightChunks={highlightChunks}
            surfaceZoneEdges={surfaceZoneEdges}
          />
        </div>
      </div>
    </div>
  );
};

export default LandmarkTab;
