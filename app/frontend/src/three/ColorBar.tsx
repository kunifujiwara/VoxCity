/**
 * Horizontal HTML colorbar overlay (drawn outside the WebGL canvas).
 * Matches the reference style: dark, semi-transparent glass panel anchored
 * to the bottom-right of the canvas, with a horizontal gradient bar and
 * tick labels below.
 *
 * Designed to be placed inside the same `position: relative` wrapper as
 * `<SceneViewer/>`.
 */
import { useMemo } from 'react';

export interface ColorBarProps {
  vmin: number;
  vmax: number;
  colormap: string;
  unitLabel?: string;
  /** Pixel offset from the canvas edge (bottom-right). */
  offset?: { right?: number; bottom?: number };
  /** Pixel size of the gradient bar itself. */
  size?: { width?: number; height?: number };
}

/** A small palette of common matplotlib colormap stops (3 stops each). */
const PALETTES: Record<string, string[]> = {
  viridis: ['#440154', '#21918c', '#fde725'],
  plasma:  ['#0d0887', '#cc4778', '#f0f921'],
  inferno: ['#000004', '#bb3754', '#fcffa4'],
  magma:   ['#000004', '#b73779', '#fcfdbf'],
  cividis: ['#00204c', '#7c7b78', '#ffea46'],
  jet:     ['#00007f', '#7fff7f', '#7f0000'],
  hot:     ['#000000', '#ff0000', '#ffff00'],
  cool:    ['#00ffff', '#7f7fff', '#ff00ff'],
  RdYlBu:  ['#a50026', '#ffffbf', '#313695'],
  Greens:  ['#f7fcf5', '#74c476', '#00441b'],
  Blues:   ['#f7fbff', '#6baed6', '#08306b'],
  BuPu_r:  ['#4d004b', '#88419d', '#e0ecf4'],
  Spectral: ['#9e0142', '#ffffbf', '#5e4fa2'],
  gray:    ['#000000', '#888888', '#ffffff'],
  Greys:   ['#ffffff', '#888888', '#000000'],
  coolwarm: ['#3b4cc0', '#dddddd', '#b40426'],
  RdYlBu_r: ['#313695', '#ffffbf', '#a50026'],
};

function gradientCss(name: string): string {
  const stops = PALETTES[name] ?? PALETTES.viridis;
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/** Compact tick formatter (mirrors the reference colorbar). */
function formatTick(v: number): string {
  if (!Number.isFinite(v)) return '-';
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v / 1e9).toPrecision(3) + 'G';
  if (abs >= 1e6) return (v / 1e6).toPrecision(3) + 'M';
  if (abs >= 1e4) return (v / 1e3).toPrecision(3) + 'k';
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toPrecision(3);
  if (abs === 0) return '0';
  return v.toPrecision(2);
}

const TICK_COUNT = 5;

export function ColorBar({
  vmin,
  vmax,
  colormap,
  unitLabel,
  offset = { right: 16, bottom: 16 },
  size = { width: 280, height: 12 },
}: ColorBarProps) {
  const gradient = useMemo(() => gradientCss(colormap), [colormap]);
  const ticks = useMemo(
    () =>
      Array.from({ length: TICK_COUNT }, (_, k) => {
        const t = k / (TICK_COUNT - 1);
        return vmin + t * (vmax - vmin);
      }),
    [vmin, vmax],
  );

  const barW = size.width ?? 280;
  const barH = size.height ?? 12;

  return (
    <div
      style={{
        position: 'absolute',
        right: offset.right,
        bottom: offset.bottom,
        background: 'rgba(0, 0, 0, 0.55)',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: 8,
        padding: '10px',
        fontFamily: '"Segoe UI", system-ui, sans-serif',
        color: 'rgba(255, 255, 255, 0.95)',
        pointerEvents: 'none',
        zIndex: 10,
        boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
      }}
    >
      {unitLabel && (
        <div
          style={{
            fontSize: 11,
            fontWeight: 600,
            textAlign: 'center',
            marginBottom: 6,
          }}
        >
          {unitLabel}
        </div>
      )}
      <div
        style={{
          width: barW,
          height: barH,
          background: gradient,
          borderRadius: 4,
        }}
      />
      <div
        style={{
          position: 'relative',
          width: barW,
          height: 16,
          marginTop: 4,
          fontSize: 11,
        }}
      >
        {ticks.map((v, i) => {
          const frac = i / (TICK_COUNT - 1);
          const align: 'left' | 'right' | 'center' =
            i === 0 ? 'left' : i === TICK_COUNT - 1 ? 'right' : 'center';
          return (
            <span
              key={i}
              style={{
                position: 'absolute',
                left: `${frac * 100}%`,
                transform:
                  align === 'left'
                    ? 'translateX(0)'
                    : align === 'right'
                    ? 'translateX(-100%)'
                    : 'translateX(-50%)',
                color: 'rgba(255, 255, 255, 0.9)',
                whiteSpace: 'nowrap',
              }}
            >
              {formatTick(v)}
            </span>
          );
        })}
      </div>
    </div>
  );
}
