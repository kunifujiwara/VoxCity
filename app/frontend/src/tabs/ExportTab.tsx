import React, { useRef, useState } from 'react';
import { Package, Box, Download, Upload, Map, Link2, Copy } from 'lucide-react';
import { createShare, exportCityles, exportObj, exportGeotiff, loadSession, saveSession } from '../api';
import type { SessionLoadSummary } from '../api';
import {
  buildRestoredFrontendState,
  parsePersistedFrontendState,
  type RestoredFrontendState,
} from '../lib/sessionRestore';
import type { Zone } from '../types/zones';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { useT } from '../i18n';
import { ExportFormat, Translate, exportActionLabel, prerequisiteMessageForTab } from './guidedTabState';

interface ExportTabProps {
  hasModel: boolean;
  zones: Zone[];
  onSessionLoaded?: (summary: SessionLoadSummary, restored?: RestoredFrontendState) => void;
}

const ExportTab: React.FC<ExportTabProps> = ({ hasModel, zones, onSessionLoaded }) => {
  const t = useT() as Translate;
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

  const [shareLoading, setShareLoading] = useState(false);
  const [shareError, setShareError] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareCopied, setShareCopied] = useState(false);
  const shareUrlInputRef = useRef<HTMLInputElement | null>(null);

  const copyToClipboard = async (url: string): Promise<boolean> => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(url);
        return true;
      }
    } catch {
      // The async write can fail on insecure origins, or after the click's
      // user-activation expires (a network await ran between click and write).
      // Fall through to the legacy path — note it may ALSO be activation-gated
      // in some engines, so callers must keep the manual Copy button reachable.
    }
    const textarea = document.createElement('textarea');
    try {
      textarea.value = url;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.top = '0';
      textarea.style.opacity = '0';
      textarea.style.pointerEvents = 'none';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      if (textarea.parentNode) document.body.removeChild(textarea);
    }
  };

  const handleCreateShare = async () => {
    setShareLoading(true);
    setShareError(null);
    setShareUrl(null);
    setShareCopied(false);
    try {
      const result = await createShare(JSON.stringify({ zones }));
      const url = `${window.location.origin}${result.path}`;
      setShareUrl(url);
      setShareCopied(await copyToClipboard(url));
    } catch (err: any) {
      setShareError(err.message);
    } finally {
      setShareLoading(false);
    }
  };

  const handleCopyShareUrl = async () => {
    if (!shareUrl) return;
    const copied = await copyToClipboard(shareUrl);
    setShareCopied(copied);
    // On plain-http origins (common for a remote Docker host) the Clipboard API
    // is unavailable, so writeText fails silently. Select the URL text instead.
    if (!copied) shareUrlInputRef.current?.select();
  };

  if (!hasModel) {
    const message = prerequisiteMessageForTab(t, 'export');
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
      setSessionSuccess(t('export.sessionSaved'));
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
          ? t('export.sessionLoadedPartial')
          : t('export.sessionLoaded'),
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
        setSuccess(t('export.citylesExported'));
      } else if (exportFormat === 'geotiff') {
        const blob = await exportGeotiff({ filename: geotiffFilename });
        downloadBlob(blob, `${geotiffFilename}_geotiff.zip`);
        setSuccess(t('export.geotiffExported'));
      } else {
        const blob = await exportObj({
          filename: objFilename,
          export_netcdf: exportNetcdf,
        });
        downloadBlob(blob, `${objFilename}.zip`);
        setSuccess(t('export.objExported'));
      }
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600 }}>
      <GuidedPanel
        title={t('export.sessionTitle')}
        subtitle={t('export.sessionSubtitle')}
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
              disabled={!hasModel || sessionLoading || shareLoading}
              onClick={handleSaveSession}
            >
              {sessionLoading && <span className="spinner" />}
              <Download size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {t('export.saveSession')}
            </button>
            <button
              className="btn"
              type="button"
              disabled={sessionLoading || shareLoading}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {t('export.loadSession')}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip,application/zip"
              style={{ display: 'none' }}
              onChange={handleLoadSession}
            />
            <button
              className="btn"
              type="button"
              disabled={!hasModel || sessionLoading || shareLoading}
              onClick={handleCreateShare}
            >
              {shareLoading && <span className="spinner" />}
              <Link2 size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {t('export.shareLink')}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label={t('export.sessionOptions')}>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={sessionIncludeSim}
              disabled={!hasModel || sessionLoading}
              onChange={(e) => setSessionIncludeSim(e.target.checked)}
            />
            <span>{t('export.includeSim')}</span>
          </label>
        </GuidedSection>

        {(shareUrl || shareError) && (
          <GuidedSection index={2} label={t('export.shareSectionLabel')}>
            {shareError ? (
              <GuidedStatus tone="error">{shareError}</GuidedStatus>
            ) : (
              <>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    ref={shareUrlInputRef}
                    readOnly
                    value={shareUrl ?? ''}
                    aria-label={t('export.shareUrlLabel')}
                    onFocus={(e) => e.currentTarget.select()}
                    style={{ flex: 1, minWidth: 0 }}
                  />
                  <button className="btn" type="button" onClick={handleCopyShareUrl}>
                    <Copy size={14} aria-hidden="true" style={{ marginRight: 6 }} />
                    {t('export.shareCopy')}
                  </button>
                </div>
                <GuidedStatus tone="success">
                  {shareCopied
                    ? t('export.shareCopied')
                    : t('export.shareCreated')}
                </GuidedStatus>
              </>
            )}
          </GuidedSection>
        )}
      </GuidedPanel>

      <GuidedPanel
        title={t('export.exportTitle')}
        subtitle={t('export.exportSubtitle')}
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
              {exportActionLabel(t, exportFormat, loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label={t('export.formatHeading')}>
          <ChoiceGroup
            ariaLabel={t('export.formatAria')}
            value={exportFormat}
            onChange={setExportFormat}
            options={[
              { id: 'cityles', label: t('export.optCitylesLabel'), description: t('export.optCitylesDesc'), icon: Package },
              { id: 'obj', label: t('export.optObjLabel'), description: t('export.optObjDesc'), icon: Box },
              { id: 'geotiff', label: t('export.optGeotiffLabel'), description: t('export.optGeotiffDesc'), icon: Map },
            ]}
          />
        </GuidedSection>

        {exportFormat === 'cityles' && (
          <GuidedSection index={2} label={t('export.citylesOptions')}>
            <div className="form-group">
              <label>{t('export.buildingMaterial')}</label>
              <select value={buildingMaterial} onChange={(e) => setBuildingMaterial(e.target.value)}>
                <option value="default">{t('export.matDefault')}</option>
                <option value="concrete">{t('export.matConcrete')}</option>
                <option value="brick">{t('export.matBrick')}</option>
              </select>
            </div>
            <div className="form-group">
              <label>{t('export.treeType')}</label>
              <select value={treeType} onChange={(e) => setTreeType(e.target.value)}>
                <option value="default">{t('export.treeDefault')}</option>
                <option value="deciduous">{t('export.treeDeciduous')}</option>
                <option value="conifer">{t('export.treeConifer')}</option>
              </select>
            </div>
            <div className="form-group">
              <label>{t('export.trunkHeightRatio')}</label>
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
          <GuidedSection index={2} label={t('export.objOptions')}>
            <div className="form-group">
              <label>{t('export.outputFilename')}</label>
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
              <span>{t('export.alsoNetcdf')}</span>
            </div>
          </GuidedSection>
        )}

        {exportFormat === 'geotiff' && (
          <GuidedSection index={2} label={t('export.geotiffOptions')}>
            <div className="form-group">
              <label>{t('export.outputFilename')}</label>
              <input
                type="text"
                value={geotiffFilename}
                onChange={(e) => setGeotiffFilename(e.target.value)}
              />
            </div>
            <p style={{ fontSize: '0.78rem', opacity: 0.8, margin: '0.25rem 0 0' }}>
              {t('export.geotiffHint')}
            </p>
          </GuidedSection>
        )}
      </GuidedPanel>
    </div>
  );
};

export default ExportTab;
