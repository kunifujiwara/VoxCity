import React from 'react';

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
  children: React.ReactNode;
  className?: string;
}

export const GuidedSection: React.FC<GuidedSectionProps> = ({ label, title, children, className }) => (
  <section className={`guided-section${className ? ` ${className}` : ''}`}>
    {label && <div className="guided-section-label">{label}</div>}
    {title && <h3 className="guided-section-title">{title}</h3>}
    {children}
  </section>
);

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
