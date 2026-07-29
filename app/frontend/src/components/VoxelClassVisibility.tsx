import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { VOXEL_CLASSES } from '../constants';
import { useT } from '../i18n';

interface VoxelClassVisibilityProps {
  hiddenClasses: Set<number>;
  onHiddenClassesChange: (next: Set<number>) => void;
}

const VoxelClassVisibility: React.FC<VoxelClassVisibilityProps> = ({
  hiddenClasses,
  onHiddenClassesChange,
}) => {
  const t = useT();
  const [open, setOpen] = useState(false);

  const toggle = (id: number) => {
    const next = new Set(hiddenClasses);
    next.has(id) ? next.delete(id) : next.add(id);
    onHiddenClassesChange(next);
  };

  return (
    <div className="expander">
      <div className="expander-header" onClick={() => setOpen(!open)}>
        {t('voxelVisibility.heading')} <span className={`expander-chevron ${open ? 'open' : ''}`}><ChevronDown size={16} /></span>
      </div>
      {open && (
        <div className="expander-body">
          <label style={{ fontSize: '0.85rem', color: 'var(--vc-muted)' }}>
            {t('voxelVisibility.hideClasses')}
          </label>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '0.25rem 0.5rem',
              marginTop: '0.4rem',
            }}
          >
            {VOXEL_CLASSES.map((cls) => (
              <div className="checkbox-row" key={cls.id}>
                <input
                  type="checkbox"
                  checked={hiddenClasses.has(cls.id)}
                  onChange={() => toggle(cls.id)}
                />
                <span>{cls.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VoxelClassVisibility;
