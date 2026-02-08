import React, { useState } from 'react';
import { exportCityles, exportObj } from '../api';

interface ExportTabProps {
  hasModel: boolean;
}

const ExportTab: React.FC<ExportTabProps> = ({ hasModel }) => {
  const [exportFormat, setExportFormat] = useState('cityles');
  const [buildingMaterial, setBuildingMaterial] = useState('default');
  const [treeType, setTreeType] = useState('default');
  const [trunkHeightRatio, setTrunkHeightRatio] = useState(0.3);
  const [objFilename, setObjFilename] = useState('voxcity');
  const [exportNetcdf, setExportNetcdf] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
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
      <div className="panel">
        <h2>Export</h2>

        <div className="form-group">
          <label>Export Format</label>
          <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
            <option value="cityles">CityLES</option>
            <option value="obj">OBJ File</option>
          </select>
        </div>

        {exportFormat === 'cityles' ? (
          <>
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
          </>
        ) : (
          <>
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
          </>
        )}

        <button className="btn btn-primary" onClick={handleExport} disabled={loading} style={{ marginTop: '0.5rem' }}>
          {loading && <span className="spinner" />}
          {loading ? 'Exporting...' : `Export ${exportFormat === 'cityles' ? 'CityLES' : 'OBJ'}`}
        </button>

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}
        {success && <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>{success}</div>}
      </div>
    </div>
  );
};

export default ExportTab;
