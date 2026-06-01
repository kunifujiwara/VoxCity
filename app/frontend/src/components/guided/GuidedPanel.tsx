import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';

export type GuidedTone = 'info' | 'success' | 'warning' | 'error';

interface GuidedPanelProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
  status?: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
}

export const GuidedPanel: React.FC<GuidedPanelProps> = ({
  title,
  subtitle,
  children,
  status,
  footer,
  className,
}) => (
  <div className={`panel guided-panel${className ? ` ${className}` : ''}`}>
    <div className="guided-panel-body">
      {(title || subtitle) && (
        <div className="guided-panel-heading">
          {title && <h2>{title}</h2>}
          {subtitle && <div className="guided-panel-subtitle">{subtitle}</div>}
        </div>
      )}
      {children}
    </div>
    {(status || footer) && (
      <div className="guided-panel-bottom">
        {status && <div className="guided-panel-status-slot">{status}</div>}
        {footer}
      </div>
    )}
  </div>
);

interface GuidedSectionProps {
  label?: React.ReactNode;
  title?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  collapsible?: boolean;
  defaultOpen?: boolean;
  index?: number;
  tone?: 'default' | 'danger';
}

export const GuidedSection: React.FC<GuidedSectionProps> = ({
  label,
  title,
  action,
  children,
  className,
  collapsible = false,
  defaultOpen = true,
  index,
  tone = 'default',
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const showBody = !collapsible || open;

  const rootClass = `guided-section${tone === 'danger' ? ' tone-danger' : ''}${className ? ` ${className}` : ''}`;

  const labelContent = (
    <>
      {index !== undefined && <span className="guided-section-index">{index}</span>}
      <span className="guided-section-label-text">{label}</span>
    </>
  );

  if (collapsible && action !== undefined) {
    throw new Error('GuidedSection: `action` and `collapsible` cannot be used together');
  }

  if (collapsible) {
    return (
      <section className={rootClass}>
        <button
          type="button"
          className="guided-section-header guided-section-header-collapsible"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          {label && <span className="guided-section-label">{labelContent}</span>}
          <span className="guided-section-action guided-section-collapse-link">
            {open ? 'Hide' : 'Show'}
            <ChevronDown
              size={12}
              className={`guided-section-collapse-chevron${open ? ' open' : ''}`}
              aria-hidden="true"
            />
          </span>
        </button>
        {title && showBody && <h3 className="guided-section-title">{title}</h3>}
        {showBody && children}
      </section>
    );
  }

  return (
    <section className={rootClass}>
      {(label || action) && (
        <div className="guided-section-header">
          {label && <div className="guided-section-label">{labelContent}</div>}
          {action && <div className="guided-section-action">{action}</div>}
        </div>
      )}
      {title && <h3 className="guided-section-title">{title}</h3>}
      {children}
    </section>
  );
};

export const GuidedFooter: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className,
}) => <div className={`guided-footer${className ? ` ${className}` : ''}`}>{children}</div>;

export const GuidedStatus: React.FC<{
  children: React.ReactNode;
  tone?: GuidedTone;
  className?: string;
}> = ({ children, tone = 'info', className }) => (
  <div className={`guided-status guided-status-${tone}${className ? ` ${className}` : ''}`}>
    {children}
  </div>
);
