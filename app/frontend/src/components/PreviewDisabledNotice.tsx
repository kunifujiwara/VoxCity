import React from 'react';
import { EyeOff } from 'lucide-react';
import { PREVIEW_MAX_CELLS } from '../constants';
import { useT } from '../i18n';

interface PreviewDisabledNoticeProps {
  gridShape?: number[] | null;
}

/**
 * Placeholder shown in a tab's 3D panel when the grid is large enough that the
 * 3D preview is auto-disabled. Generation, editing, simulation results, and
 * export still work.
 */
const PreviewDisabledNotice: React.FC<PreviewDisabledNoticeProps> = ({ gridShape }) => {
  const t = useT();
  const dims =
    gridShape && gridShape.length >= 2 ? `${gridShape[0]}×${gridShape[1]}` : null;
  return (
    <div
      className="preview-disabled-notice"
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100%', textAlign: 'center',
        gap: '0.75rem', padding: '2rem', opacity: 0.85,
      }}
    >
      <EyeOff size={32} aria-hidden="true" />
      <strong>{t('previewNotice.heading')}</strong>
      <p style={{ maxWidth: 360, fontSize: '0.85rem', margin: 0 }}>
        {dims
          ? t('previewNotice.bodyWithDims', { dims, cells: PREVIEW_MAX_CELLS.toLocaleString() })
          : t('previewNotice.bodyNoDims', { cells: PREVIEW_MAX_CELLS.toLocaleString() })}
      </p>
    </div>
  );
};

export default PreviewDisabledNotice;
