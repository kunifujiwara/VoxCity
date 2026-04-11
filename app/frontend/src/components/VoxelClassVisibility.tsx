import React, { useState } from 'react';
import { VOXEL_CLASSES } from '../constants';

interface VoxelClassVisibilityProps {
  hiddenClasses: Set<number>;
  onHiddenClassesChange: (next: Set<number>) => void;
}

const VoxelClassVisibility: React.FC<VoxelClassVisibilityProps> = ({
  hiddenClasses,
  onHiddenClassesChange,
}) => {
  const [open, setOpen] = useState(false);

  const toggle = (id: number) => {
    const next = new Set(hiddenClasses);
    next.has(id) ? next.delete(id) : next.add(id);
    onHiddenClassesChange(next);
  };

  return (
    <div className="expander">
      <div className="expander-header" onClick={() => setOpen(!open)}>
        Visualization Settings <span>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div className="expander-body">
          <label style={{ fontSize: '0.85rem', color: 'var(--vc-muted)' }}>
            Hide element classes
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
