import React, { useRef, useState } from 'react';
import { Package, Box, Download, Upload, Map } from 'lucide-react';
import { exportCityles, exportObj, exportGeotiff, loadSession, saveSession } from '../api';
import type { SessionLoadSummary } from '../api';
import {
  buildRestoredFrontendState,
  parsePersistedFrontendState,
  type RestoredFrontendState,
} from '../lib/sessionRestore';
import type { Zone } from '../types/zones';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { ExportFormat, exportActionLabel, prerequisiteMessageForTab } from './guidedTabState';

interface ExportTabProps {
  hasModel: boolean;
  zones: Zone[];
  onSessionLoaded?: (summary: SessionLoadSummary, restored?: RestoredFrontendState) => void;
}

const ExportTab: React.FC<ExportTabProps> = ({ hasModel, zones, onSessionLoaded }) => {
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cityles');
  const [buildingMaterial, setBuildingMaterial] = useState('default');
  const [treeType, setTreeType] = useState('default');
  const [trunkHeightRatio, setTrunkHeightRatio] = useState(0.3);
  const [objFilename, setObjFilename] = useState('voxcity');
  const [exportNetcdf, setExportNetcdf] = useState(false);
  const [geotiffFilename, setGeotiffFilename] = useState('voxcity');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [sessionIncludeSim, setSessionIncludeSim] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [sessionSuccess, setSessionSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!hasModel) {
    const message = prerequisiteMessageForTab('export');
    return (
      <div style={{ maxWidth: 600 }}>
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
  }

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveSession = async () => {
    setSessionLoading(true);
    setSessionError(null);
    setSessionSuccess(null);
    try {
      const frontendStateJson = JSON.stringify({ zones });
      const blob = await saveSession(frontendStateJson, sessionIncludeSim);
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      downloadBlob(blob, `voxcity-session-${ts}.zip`);
      setSessionSuccess('Session saved.');
    } catch (err: any) {
      setSessionError(err.message);
    } finally {
      setSessionLoading(false);
    }
  };

  const handleLoadSession = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSessionLoading(true);
    setSessionError(null);
    setSessionSuccess(null);
    try {
      const summary = await loadSession(file);
      const persisted = parsePersistedFrontendState(summary.frontend_state);
      const malformed = Boolean(summary.frontend_state) && !persisted;
      const { restored, skippedFrontendState } = buildRestoredFrontendState(persisted);
      onSessionLoaded?.(summary, restored);
      setSessionSuccess(
        malformed || skippedFrontendState
          ? 'Session loaded; some frontend state was not restored.'
          : 'Session loaded.',
      );
    } catch (err: any) {
      setSessionError(err.message);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
      setSessionLoading(false);
    }
  };

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      if (exportFormat === 'cityles') {
        const blob = await exportCityles({
          building_material: buildingMaterial,
          tree_type: treeType,
          trunk_height_ratio: trunkHeightRatio,
        });
        downloadBlob(blob, 'cityles_outputs.zip');
        setSuccess('CityLES exported successfully!');
      } else if (exportFormat === 'geotiff') {
        const blob = await exportGeotiff({ filename: geotiffFilename });
        downloadBlob(blob, `${geotiffFilename}_geotiff.zip`);
        setSuccess('GeoTIFF exported successfully!');
      } else {
        const blob = await exportObj({
          filename: objFilename,
          export_netcdf: exportNetcdf,
        });
        downloadBlob(blob, `${objFilename}.zip`);
        setSuccess('OBJ exported successfully!');
      }
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600 }}>
      <GuidedPanel
        title="Save / Load Session"
        subtitle="Move the current scene and zones between browser sessions."
        status={
          sessionError ? <GuidedStatus tone="error">{sessionError}</GuidedStatus>
            : sessionSuccess ? <GuidedStatus tone="success">{sessionSuccess}</GuidedStatus>
            : undefined
        }
        footer={(
          <GuidedFooter>
            <button
              className="btn btn-primary"
              type="button"
              disabled={!hasModel || sessionLoading}
              onClick={handleSaveSession}
            >
              {sessionLoading && <span className="spinner" />}
              <Download size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              Save Session
            </button>
            <button
              className="btn"
              type="button"
              disabled={sessionLoading}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              Load Session
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip,application/zip"
              style={{ display: 'none' }}
              onChange={handleLoadSession}
            />
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="SESSION OPTIONS">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={sessionIncludeSim}
              disabled={!hasModel || sessionLoading}
              onChange={(e) => setSessionIncludeSim(e.target.checked)}
            />
            <span>Include simulation results (larger file, lets overlays render without re-running)</span>
          </label>
        </GuidedSection>
      </GuidedPanel>

      <GuidedPanel
        title="Export"
        subtitle="Download the VoxCity model in your preferred format."
        status={
          error ? (
            <GuidedStatus tone="error">{error}</GuidedStatus>
          ) : success ? (
            <GuidedStatus tone="success">{success}</GuidedStatus>
          ) : undefined
        }
        footer={(
          <GuidedFooter>
            <button className="btn btn-primary" onClick={handleExport} disabled={loading} type="button">
              {loading && <span className="spinner" />}
              <Download size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {exportActionLabel(exportFormat, loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="EXPORT FORMAT">
          <ChoiceGroup
            ariaLabel="Export format"
            value={exportFormat}
            onChange={setExportFormat}
            options={[
              { id: 'cityles', label: 'CityLES', description: 'CityLES output archive', icon: Package },
              { id: 'obj', label: 'OBJ', description: 'Mesh export archive', icon: Box },
              { id: 'geotiff', label: 'GeoTIFF', description: 'Georeferenced raster layers', icon: Map },
            ]}
          />
        </GuidedSection>

        {exportFormat === 'cityles' && (
          <GuidedSection index={2} label="CITYLES OPTIONS">
            <div className="form-group">
              <label>Building Material</label>
              <select value={buildingMaterial} onChange={(e) => setBuildingMaterial(e.target.value)}>
                <option value="default">Default</option>
                <option value="concrete">Concrete</option>
                <option value="brick">Brick</option>
              </select>
            </div>
            <div className="form-group">
              <label>Tree Type</label>
              <select value={treeType} onChange={(e) => setTreeType(e.target.value)}>
                <option value="default">Default</option>
                <option value="deciduous">Deciduous</option>
                <option value="conifer">Conifer</option>
              </select>
            </div>
            <div className="form-group">
              <label>Trunk Height Ratio</label>
              <input
                type="number"
                value={trunkHeightRatio}
                min={0}
                max={1}
                step={0.05}
                onChange={(e) => setTrunkHeightRatio(Number(e.target.value))}
              />
            </div>
          </GuidedSection>
        )}

        {exportFormat === 'obj' && (
          <GuidedSection index={2} label="OBJ OPTIONS">
            <div className="form-group">
              <label>Output Filename</label>
              <input
                type="text"
                value={objFilename}
                onChange={(e) => setObjFilename(e.target.value)}
              />
            </div>
            <div className="checkbox-row">
              <input
                type="checkbox"
                checked={exportNetcdf}
                onChange={(e) => setExportNetcdf(e.target.checked)}
              />
              <span>Also export NetCDF</span>
            </div>
          </GuidedSection>
        )}

        {exportFormat === 'geotiff' && (
          <GuidedSection index={2} label="GEOTIFF OPTIONS">
            <div className="form-group">
              <label>Output Filename</label>
              <input
                type="text"
                value={geotiffFilename}
                onChange={(e) => setGeotiffFilename(e.target.value)}
              />
            </div>
            <p style={{ fontSize: '0.78rem', opacity: 0.8, margin: '0.25rem 0 0' }}>
              Exports land cover, building height, DEM, and canopy height as four
              georeferenced GeoTIFFs (EPSG:4326), plus a README.md with layer
              details and usage instructions.
            </p>
          </GuidedSection>
        )}
      </GuidedPanel>
    </div>
  );
};

export default ExportTab;
