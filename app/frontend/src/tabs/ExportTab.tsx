import React, { useState } from 'react';
import { Package, Box, Download } from 'lucide-react';
import { exportCityles, exportObj } from '../api';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { ExportFormat, exportActionLabel, prerequisiteMessageForTab } from './guidedTabState';

interface ExportTabProps {
  hasModel: boolean;
}

const ExportTab: React.FC<ExportTabProps> = ({ hasModel }) => {
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cityles');
  const [buildingMaterial, setBuildingMaterial] = useState('default');
  const [treeType, setTreeType] = useState('default');
  const [trunkHeightRatio, setTrunkHeightRatio] = useState(0.3);
  const [objFilename, setObjFilename] = useState('voxcity');
  const [exportNetcdf, setExportNetcdf] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

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
      </GuidedPanel>
    </div>
  );
};

export default ExportTab;
