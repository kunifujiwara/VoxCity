import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

interface PlotlyViewerProps {
  figureJson: string;
}

const PlotlyViewer: React.FC<PlotlyViewerProps> = ({ figureJson }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!figureJson || figureJson === '{}' || !containerRef.current) return;

    try {
      const parsed = JSON.parse(figureJson);
      const data = parsed.data || [];
      const layout = {
        ...(parsed.layout || {}),
        autosize: true,
        margin: { l: 0, r: 0, t: 40, b: 0 },
      };
      (Plotly as any).react(containerRef.current!, data, layout, {
        responsive: true,
      });
    } catch (err) {
      console.error('Failed to render Plotly figure:', err);
    }
  }, [figureJson]);

  if (!figureJson || figureJson === '{}') {
    return <div className="alert alert-info">No 3D visualization available.</div>;
  }

  return <div ref={containerRef} className="plotly-container" />;
};

export default PlotlyViewer;
