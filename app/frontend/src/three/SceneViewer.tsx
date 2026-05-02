/**
 * Top-level R3F scene viewer. Renders:
 *   - the static city geometry (one `<MeshLayer/>` per chunk),
 *   - an optional simulation overlay (ground or building),
 *   - optional zone outlines drawn through occluders,
 *   - a colorbar legend (HTML overlay).
 *
 * Designed as a drop-in replacement for the legacy
 * `app/frontend/src/components/ThreeViewer.tsx`.
 */
import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

import {
  getSceneGeometry,
  getSimGeometry,
  type MeshChunkDto,
  type OverlayGeometryResponse,
  type SceneGeometryResponse,
} from '../api';
import type { Zone } from '../types/zones';

import { CameraControls } from './CameraControls';
import { ColorBar } from './ColorBar';
import { MeshLayer } from './MeshLayer';
import { Picker } from './Picker';
import { ZoneOutlines } from './ZoneOutlines';
import type { PickResult } from './types';

export interface SceneViewerProps {
  /** Triggers re-fetch of city geometry whenever this changes. */
  geometryToken?: string | number;
  /** Server-side downsample stride (>=1). */
  downsample?: number;
  /** Voxel palette name forwarded to the backend. The simulation tabs use
   *  ``"grayscale"`` so per-class colours don't compete with the overlay. */
  colorScheme?: string;

  /** When set, fetches and renders the per-tab simulation overlay. */
  simKind?: 'solar' | 'view' | 'landmark' | null;
  /** Triggers re-fetch of the sim overlay when it changes. */
  simToken?: string | number;
  /** Sim overlay colormap (matplotlib name). */
  colormap?: string;
  vmin?: number | null;
  vmax?: number | null;

  /** Zones drawn as through-occlusion outlines. */
  zones?: Zone[];
  /**
   * Project zone lon/lat to world XY metres for ZoneOutlines.
   * Typically the function returned by lonLatToWorldXY() in lib/grid.ts,
   * which maps cell (i,j) → world (i*meshsize, j*meshsize) with the
   * (nx-u) x-axis flip that aligns zone polygons with the voxel mesh.
   */
  lonLatToXY?: (lon: number, lat: number) => [number, number];
  /** Hide zone outlines without unmounting the scene. */
  showZones?: boolean;
  /** Optional override colour per zone id (e.g. to highlight a selection). */
  colorOverride?: Record<string, string>;

  /**
   * DEBUG: render the same zones via multiple candidate projections at once,
   * each in its own colour. Used to disambiguate the lon/lat -> world axis
   * convention. When provided, takes precedence over `lonLatToXY`/`zones`.
   */
  debugProjections?: Array<{
    label: string;
    color: string;
    project: (lon: number, lat: number) => [number, number];
  }>;

  /** Hide voxel chunks whose `metadata.class` is in this set. */
  hiddenClasses?: Set<number>;

  /** Click pick callback. Set to ``undefined`` to disable picking. */
  onPick?: (hit: PickResult | null) => void;

  /** Extra mesh chunks rendered on top of the scene (e.g. building highlights). */
  highlightChunks?: MeshChunkDto[] | null;

  /** Inline canvas style overrides. */
  style?: React.CSSProperties;
  /** Background colour for the canvas (CSS string). */
  background?: string;
}

const DEFAULT_BG = '#1a1a2e';

export function SceneViewer({
  geometryToken,
  downsample = 1,
  colorScheme = 'default',
  simKind = null,
  simToken,
  colormap = 'viridis',
  vmin = null,
  vmax = null,
  zones,
  lonLatToXY,
  showZones = true,
  colorOverride,
  debugProjections,
  hiddenClasses,
  onPick,
  highlightChunks,
  style,
  background = DEFAULT_BG,
}: SceneViewerProps) {
  const [scene, setScene] = useState<SceneGeometryResponse | null>(null);
  const [overlay, setOverlay] = useState<OverlayGeometryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch static city geometry
  useEffect(() => {
    let cancelled = false;
    setError(null);
    getSceneGeometry(downsample, colorScheme)
      .then((g) => {
        if (!cancelled) setScene(g);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      });
    return () => {
      cancelled = true;
    };
  }, [geometryToken, downsample, colorScheme]);

  // Fetch sim overlay (when requested)
  useEffect(() => {
    if (!simKind) {
      setOverlay(null);
      return;
    }
    let cancelled = false;
    getSimGeometry(simKind, { colormap, vmin, vmax })
      .then((o) => {
        if (!cancelled) setOverlay(o);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err?.message ?? err));
      });
    return () => {
      cancelled = true;
    };
  }, [simKind, simToken, colormap, vmin, vmax]);

  const overlayUserData = useMemo(() => {
    if (!overlay) return undefined;
    return {
      target: overlay.target,
      faceToCell: overlay.face_to_cell ?? undefined,
      faceToBuilding: overlay.face_to_building ?? undefined,
    };
  }, [overlay]);

  const bboxMin = scene?.bbox_min ?? [0, 0, 0];
  const bboxMax = scene?.bbox_max ?? [100, 100, 50];

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        ...style,
      }}
    >
      <Canvas
        camera={{ position: [200, -200, 200], up: [0, 0, 1], fov: 50 }}
        gl={{ antialias: true, preserveDrawingBuffer: false }}
        style={{ background }}
        onCreated={({ gl }) => {
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.55} />
          <directionalLight
            position={[bboxMax[0], bboxMin[1] - 100, bboxMax[2] + 200]}
            intensity={0.75}
          />
          <hemisphereLight args={[0xffffff, 0x444466, 0.35]} />

          {scene && (
            <Picker enabled={!!onPick} onPick={onPick}>
              {scene.chunks
                .filter((chunk) => {
                  // Hide background building meshes when a building-target sim
                  // overlay is active so the colored sim faces don't z-fight
                  // with the underlying voxel building chunks.
                  if (
                    overlay?.target === 'building' &&
                    chunk.metadata?.class === -3
                  ) {
                    return false;
                  }
                  if (!hiddenClasses || hiddenClasses.size === 0) return true;
                  const cls = chunk.metadata?.class;
                  return typeof cls !== 'number' || !hiddenClasses.has(cls);
                })
                .map((chunk, idx) => (
                  <MeshLayer key={`${chunk.name}-${idx}`} chunk={chunk} />
                ))}
              {overlay && (
                <MeshLayer
                  key={`overlay-${simKind}-${simToken ?? ''}`}
                  chunk={overlay.chunk}
                  userData={overlayUserData}
                  renderOrder={10}
                />
              )}
              {highlightChunks && highlightChunks.map((c, i) => (
                <MeshLayer key={`highlight-${c.name}-${i}`} chunk={c} renderOrder={20} />
              ))}
            </Picker>
          )}

          {showZones && zones && zones.length > 0 && (
            debugProjections && debugProjections.length > 0 ? (
              <>
                {debugProjections.map((proj) => (
                  <ZoneOutlines
                    key={proj.label}
                    zones={zones}
                    lonLatToXY={proj.project}
                    colorOverride={Object.fromEntries(
                      zones.map((z) => [z.id, proj.color]),
                    )}
                    zHeight={
                      scene
                        ? (scene.ground_top_m ?? 0) + scene.meshsize_m
                        : undefined
                    }
                    overlay={overlay}
                    meshsize={scene?.meshsize_m}
                  />
                ))}
              </>
            ) : (
              <ZoneOutlines
                zones={zones}
                lonLatToXY={lonLatToXY}
                colorOverride={colorOverride}
                zHeight={
                  scene
                    ? (scene.ground_top_m ?? 0) + scene.meshsize_m
                    : undefined
                }
                overlay={overlay}
                meshsize={scene?.meshsize_m}
              />
            )
          )}

          <CameraControls bboxMin={bboxMin} bboxMax={bboxMax} />
        </Suspense>
      </Canvas>

      {overlay && (
        <ColorBar
          vmin={overlay.value_min}
          vmax={overlay.value_max}
          colormap={overlay.colormap}
          unitLabel={overlay.unit_label}
        />
      )}

      {error && (
        <div
          style={{
            position: 'absolute',
            left: 12,
            bottom: 12,
            background: 'rgba(220, 50, 50, 0.92)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: 4,
            fontSize: 12,
            fontFamily: 'system-ui, sans-serif',
            maxWidth: '60%',
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
