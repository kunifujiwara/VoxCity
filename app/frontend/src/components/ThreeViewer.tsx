import React, { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface ThreeViewerProps {
  figureJson: string;
}

/* ──────────────────────────────────────────────────────────────
   Color helpers
   ────────────────────────────────────────────────────────────── */

/** Parse "rgb(R,G,B)" or "#RRGGBB" or "rgba(R,G,B,A)" → THREE.Color */
function parseColor(s: string): THREE.Color {
  if (!s) return new THREE.Color(0x888888);
  const rgb = s.match(/rgb[a]?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (rgb) {
    return new THREE.Color(+rgb[1] / 255, +rgb[2] / 255, +rgb[3] / 255);
  }
  return new THREE.Color(s);
}

/** Parse opacity from rgba string or return the passed default */
function parseOpacity(s: string | undefined, fallback: number): number {
  if (!s) return fallback;
  const m = s.match(/rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*([0-9.]+)/);
  return m ? parseFloat(m[1]) : fallback;
}

/* ──────────────────────────────────────────────────────────────
   Plotly binary-encoded array decoder
   ────────────────────────────────────────────────────────────── */

/**
 * Plotly serializes numpy arrays as { dtype, bdata } objects where
 * bdata is a base64-encoded binary buffer. This function decodes
 * them back to plain number arrays.
 */
function decodePlotlyArray(val: any): number[] | null {
  if (!val) return null;
  // Already a plain array
  if (Array.isArray(val)) return val;
  // Plotly binary format: { dtype: "f8"|"f4"|"i4"|"i2"|"u1"|..., bdata: "<base64>" }
  if (typeof val === 'object' && val.bdata && val.dtype) {
    const binary = atob(val.bdata);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const buffer = bytes.buffer;
    const dtype: string = val.dtype;
    let typedArray: ArrayLike<number>;
    if (dtype === 'f8') {
      typedArray = new Float64Array(buffer);
    } else if (dtype === 'f4') {
      typedArray = new Float32Array(buffer);
    } else if (dtype === 'i4') {
      typedArray = new Int32Array(buffer);
    } else if (dtype === 'i2') {
      typedArray = new Int16Array(buffer);
    } else if (dtype === 'i1') {
      typedArray = new Int8Array(buffer);
    } else if (dtype === 'u4') {
      typedArray = new Uint32Array(buffer);
    } else if (dtype === 'u2') {
      typedArray = new Uint16Array(buffer);
    } else if (dtype === 'u1') {
      typedArray = new Uint8Array(buffer);
    } else {
      console.warn(`[ThreeViewer] Unknown Plotly dtype: ${dtype}`);
      return null;
    }
    return Array.from(typedArray);
  }
  return null;
}

/* ──────────────────────────────────────────────────────────────
   Build Three.js meshes from a single Plotly Mesh3d trace
   ────────────────────────────────────────────────────────────── */

function buildMesh3d(trace: any): THREE.Mesh | null {
  const x = decodePlotlyArray(trace.x);
  const y = decodePlotlyArray(trace.y);
  const z = decodePlotlyArray(trace.z);
  const ti = decodePlotlyArray(trace.i);
  const tj = decodePlotlyArray(trace.j);
  const tk = decodePlotlyArray(trace.k);

  if (!x || !y || !z || !ti || !tj || !tk || x.length === 0 || ti.length === 0) return null;

  const nVerts = x.length;
  const nFaces = ti.length;

  // --- Build geometry (swap Y↔Z: Plotly is Z-up, Three.js is Y-up) ---
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(nVerts * 3);
  for (let v = 0; v < nVerts; v++) {
    positions[v * 3] = x[v];
    positions[v * 3 + 1] = z[v];
    positions[v * 3 + 2] = y[v];
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const indices = new Uint32Array(nFaces * 3);
  for (let f = 0; f < nFaces; f++) {
    indices[f * 3] = ti[f];
    indices[f * 3 + 1] = tj[f];
    indices[f * 3 + 2] = tk[f];
  }
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  // --- Per-face coloring (facecolor) vs uniform color ---
  const facecolorArr = Array.isArray(trace.facecolor) ? trace.facecolor : null;
  const hasFaceColor = facecolorArr !== null && facecolorArr.length === nFaces;
  const opacity = trace.opacity != null ? trace.opacity : 1.0;
  const transparent = opacity < 1.0;

  let material: THREE.Material;

  if (hasFaceColor) {
    // Non-indexed geometry so we can set per-face vertex colors
    const nonIndexed = geometry.toNonIndexed();
    const faceColors = new Float32Array(nFaces * 3 * 3); // 3 vertices * rgb
    for (let f = 0; f < nFaces; f++) {
      const c = parseColor(facecolorArr![f]);
      for (let vi = 0; vi < 3; vi++) {
        faceColors[(f * 3 + vi) * 3] = c.r;
        faceColors[(f * 3 + vi) * 3 + 1] = c.g;
        faceColors[(f * 3 + vi) * 3 + 2] = c.b;
      }
    }
    nonIndexed.setAttribute('color', new THREE.BufferAttribute(faceColors, 3));
    nonIndexed.computeVertexNormals();

    material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      opacity,
      transparent,
      side: THREE.DoubleSide,
      flatShading: true,
      roughness: 0.75,
      metalness: 0.0,
    });

    const mesh = new THREE.Mesh(nonIndexed, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  } else {
    // Uniform color
    const color = parseColor(trace.color);
    geometry.computeVertexNormals();

    material = new THREE.MeshStandardMaterial({
      color,
      opacity,
      transparent,
      side: THREE.DoubleSide,
      flatShading: true,
      roughness: 0.75,
      metalness: 0.0,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }
}

/* ──────────────────────────────────────────────────────────────
   Build a colorbar sprite (2D overlay) from Scatter3d marker data
   ────────────────────────────────────────────────────────────── */

function buildColorbar(trace: any): {
  canvas: HTMLCanvasElement;
  title: string;
  vmin: number;
  vmax: number;
} | null {
  const marker = trace?.marker;
  if (!marker || !marker.colorscale || !marker.showscale) return null;

  const vmin: number = marker.cmin ?? 0;
  const vmax: number = marker.cmax ?? 1;
  const stops: [number, string][] = marker.colorscale; // [[0, 'rgb(...)'], ...]
  const title: string = marker.colorbar?.title ?? '';

  const W = 40;
  const H = 256;
  const canvas = document.createElement('canvas');
  canvas.width = W + 80; // extra room for labels
  canvas.height = H + 40;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw gradient bar
  const grad = ctx.createLinearGradient(0, H, 0, 0); // bottom→top
  for (const [t, col] of stops) {
    grad.addColorStop(t, col);
  }
  ctx.fillStyle = grad;
  ctx.fillRect(8, 16, W, H);

  // Labels
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(vmax.toFixed(2), W + 14, 26);
  ctx.fillText(((vmin + vmax) / 2).toFixed(2), W + 14, H / 2 + 16);
  ctx.fillText(vmin.toFixed(2), W + 14, H + 14);

  // Title
  if (title) {
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, (W + 80) / 2, 12);
  }

  return { canvas, title, vmin, vmax };
}

/* ──────────────────────────────────────────────────────────────
   Main component
   ────────────────────────────────────────────────────────────── */

const ThreeViewer: React.FC<ThreeViewerProps> = ({ figureJson }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameRef = useRef<number>(0);
  const colorbarOverlayRef = useRef<HTMLDivElement | null>(null);

  /** Dispose everything */
  const cleanup = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    controlsRef.current?.dispose();
    if (sceneRef.current) {
      sceneRef.current.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
    }
    rendererRef.current?.dispose();
    rendererRef.current = null;
    sceneRef.current = null;
    cameraRef.current = null;
    controlsRef.current = null;

    // Remove colorbar overlay
    if (colorbarOverlayRef.current && containerRef.current) {
      try { containerRef.current.removeChild(colorbarOverlayRef.current); } catch { /* */ }
      colorbarOverlayRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!figureJson || figureJson === '{}' || !containerRef.current) return;

    let parsed: any;
    try {
      parsed = JSON.parse(figureJson);
    } catch (err) {
      console.error('Failed to parse figure JSON:', err);
      return;
    }

    const data: any[] = parsed.data || [];
    if (data.length === 0) return;

    // Clean up any previous scene
    cleanup();

    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // --- Renderer ---
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x1a1a2e, 1);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // --- Scene ---
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // --- Camera ---
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100000);
    cameraRef.current = camera;

    // --- Lights (realistic outdoor sun + sky) ---
    // Hemisphere light: sky blue above, warm ground bounce below
    const hemiLight = new THREE.HemisphereLight(0x87ceeb, 0x8b7355, 0.4);
    scene.add(hemiLight);

    // Ambient fill for shadow areas
    const ambientLight = new THREE.AmbientLight(0xd0d8e8, 0.35);
    scene.add(ambientLight);

    // Main sun light (warm directional with shadows)
    const sunLight = new THREE.DirectionalLight(0xfff4e0, 1.2);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.bias = -0.0005;
    sunLight.shadow.normalBias = 0.02;
    scene.add(sunLight);

    // Fill light from opposite side (cool sky bounce)
    const fillLight = new THREE.DirectionalLight(0xb0c4de, 0.3);
    scene.add(fillLight);

    // Subtle rim/back light for depth
    const rimLight = new THREE.DirectionalLight(0xffeedd, 0.15);
    scene.add(rimLight);

    // --- Parse traces and add meshes ---
    const meshGroup = new THREE.Group();
    let colorbarTrace: any = null;

    for (const trace of data) {
      if (trace.type === 'mesh3d') {
        const mesh = buildMesh3d(trace);
        if (mesh) meshGroup.add(mesh);
      } else if (trace.type === 'scatter3d') {
        // Colorbar-only trace
        if (trace.marker?.showscale) {
          colorbarTrace = trace;
        }
      }
    }

    scene.add(meshGroup);

    // --- Compute bounding box & center the scene ---
    const bbox = new THREE.Box3().setFromObject(meshGroup);
    const center = bbox.getCenter(new THREE.Vector3());
    const size = bbox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    // Center the group at origin
    meshGroup.position.sub(center);

    // Camera position
    const cameraDistance = maxDim * 1.6;
    camera.position.set(cameraDistance * 0.8, cameraDistance * 0.8, cameraDistance * 0.6);
    camera.lookAt(0, 0, 0);
    camera.near = maxDim * 0.001;
    camera.far = maxDim * 20;
    camera.updateProjectionMatrix();

    // Update light positions & shadow camera based on scene scale
    sunLight.position.set(maxDim * 0.8, maxDim * 1.5, maxDim * 0.6);
    sunLight.target.position.set(0, 0, 0);
    scene.add(sunLight.target);
    const shadowSize = maxDim * 1.2;
    sunLight.shadow.camera.left = -shadowSize;
    sunLight.shadow.camera.right = shadowSize;
    sunLight.shadow.camera.top = shadowSize;
    sunLight.shadow.camera.bottom = -shadowSize;
    sunLight.shadow.camera.near = maxDim * 0.01;
    sunLight.shadow.camera.far = maxDim * 5;
    sunLight.shadow.camera.updateProjectionMatrix();

    fillLight.position.set(-maxDim * 0.6, maxDim * 0.8, -maxDim * 0.4);
    rimLight.position.set(-maxDim * 0.3, maxDim * 0.2, maxDim * 0.9);

    // --- Controls ---
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.minDistance = maxDim * 0.1;
    controls.maxDistance = maxDim * 10;
    controls.update();
    controlsRef.current = controls;

    // --- Colorbar overlay (HTML canvas on top) ---
    if (colorbarTrace) {
      const cb = buildColorbar(colorbarTrace);
      if (cb) {
        const overlay = document.createElement('div');
        overlay.style.position = 'absolute';
        overlay.style.right = '16px';
        overlay.style.top = '16px';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = '10';
        overlay.appendChild(cb.canvas);
        container.appendChild(overlay);
        colorbarOverlayRef.current = overlay;
      }
    }

    // --- Axes helper ---
    const axesSize = maxDim * 0.2;
    const axes = new THREE.AxesHelper(axesSize);
    // Move axes to the bottom-left-back corner of the bounding box (after centering)
    axes.position.set(-size.x / 2, -size.y / 2, -size.z / 2);
    scene.add(axes);

    // --- Render loop ---
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // --- Resize handler ---
    const onResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      cleanup();
      // Remove the WebGL canvas
      if (renderer.domElement && renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [figureJson, cleanup]);

  if (!figureJson || figureJson === '{}') {
    return <div className="alert alert-info">No 3D visualization available.</div>;
  }

  return <div ref={containerRef} className="three-container" />;
};

export default ThreeViewer;
