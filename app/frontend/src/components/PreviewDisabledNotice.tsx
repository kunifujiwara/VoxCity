import React from 'react';
import { EyeOff } from 'lucide-react';
import { PREVIEW_MAX_CELLS } from '../constants';

interface PreviewDisabledNoticeProps {
  gridShape?: number[] | null;
}

/**
 * Placeholder shown in a tab's 3D panel when the grid is large enough that the
 * 3D preview is auto-disabled. Generation, editing, simulation results, and
 * export still work.
 */
const PreviewDisabledNotice: React.FC<PreviewDisabledNoticeProps> = ({ gridShape }) => {
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
      <strong>3D preview disabled</strong>
      <p style={{ maxWidth: 360, fontSize: '0.85rem', margin: 0 }}>
        {dims ? <>The grid ({dims}) exceeds the preview limit of{' '}</>
              : <>This grid exceeds the preview limit of{' '}</>}
        {PREVIEW_MAX_CELLS.toLocaleString()} cells. Generation, editing,
        simulation results, and export still work.
      </p>
    </div>
  );
};

export default PreviewDisabledNotice;
