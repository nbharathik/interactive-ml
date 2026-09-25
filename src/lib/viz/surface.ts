/** A height field drawn in three dimensions: facets in painter's order, a floor, and the projector for anything placed on it. */

import { MONO_STACK, drawMarker } from './canvas';
import type { MarkerShape } from './canvas';
import { isoSegments } from './contours';
import { mix, rgba } from './palette';
import type { Palette } from './palette';

export interface SurfaceView {
  /** Turn about the vertical axis, in radians. */
  azimuth: number;
  /** Tilt down from the horizon, in radians. */
  elevation: number;
}

export interface SurfaceRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Projected {
  x: number;
  y: number;
  /** Larger is farther from the viewer. */
  depth: number;
}

/** Maps floor coordinates u, v in [-1, 1] and a height z in [0, 1] to the canvas. */
export type Projector = (u: number, v: number, z: number) => Projected;

/** Height of the box relative to half the floor's side. */
const BOX_HEIGHT = 0.8;

/* ---------------- the view ---------------- */

export const DEFAULT_SURFACE_VIEW: SurfaceView = { azimuth: -0.65, elevation: 0.6 };
export const ELEVATION_MIN = 0.15;
export const ELEVATION_MAX = 1.45;

export function clampView(view: SurfaceView): SurfaceView {
  return { azimuth: view.azimuth, elevation: Math.max(ELEVATION_MIN, Math.min(ELEVATION_MAX, view.elevation)) };
}

/** The view after a drag of dx, dy pixels. */
export function dragView(from: SurfaceView, dx: number, dy: number): SurfaceView {
  return clampView({ azimuth: from.azimuth + dx * 0.012, elevation: from.elevation + dy * 0.008 });
}

export function turnView(from: SurfaceView, dAzimuth: number, dElevation: number): SurfaceView {
  return clampView({ azimuth: from.azimuth + dAzimuth, elevation: from.elevation + dElevation });
}

export function viewKey(view: SurfaceView): string {
  return view.azimuth.toFixed(3) + ',' + view.elevation.toFixed(3);
}

/* ---------------- projection ---------------- */

/** Screen height of the box across every allowed tilt, so the scale never depends on the view. */
function worstVerticalExtent(): number {
  let worst = 0;
  for (let e = ELEVATION_MIN; e <= ELEVATION_MAX + 1e-9; e += 0.01) {
    worst = Math.max(worst, 2 * Math.SQRT2 * Math.sin(e) + BOX_HEIGHT * Math.cos(e));
  }
  return worst;
}

const VERTICAL_EXTENT = worstVerticalExtent();

/** Orthographic, scaled once for the whole turn: dragging never shrinks or grows the box. */
export function makeProjector(rect: SurfaceRect, view: SurfaceView): Projector {
  const cosA = Math.cos(view.azimuth);
  const sinA = Math.sin(view.azimuth);
  const cosE = Math.cos(view.elevation);
  const sinE = Math.sin(view.elevation);
  const scale = Math.min(rect.w / (2 * Math.SQRT2), rect.h / VERTICAL_EXTENT) * 0.94;
  const cx = rect.x + rect.w / 2;
  // The box's own vertical middle sits on the frame's middle.
  const cy = rect.y + rect.h / 2 + (BOX_HEIGHT * cosE * scale) / 2;
  return (u, v, z) => {
    const across = u * cosA - v * sinA;
    const away = u * sinA + v * cosA;
    return { x: cx + across * scale, y: cy - away * scale * sinE - z * BOX_HEIGHT * scale * cosE, depth: away };
  };
}

/* ---------------- drawing ---------------- */

export interface SurfaceMark {
  u: number;
  v: number;
  /** Height in [0, 1]. */
  z: number;
  colour: string;
  radius: number;
  shape?: MarkerShape;
  /** Draw a ring around it, for the hovered one. */
  ring?: boolean;
}

export interface SurfaceOptions {
  /** Height in [0, 1] of a value. */
  zOf: (value: number) => number;
  /** Fill of a facet from the mean value of its corners and the floor point under its centre. */
  colourAt: (value: number, u: number, v: number) => string;
  /** A level to trace on the surface, in value units. */
  level?: number;
  marks?: readonly SurfaceMark[];
  labels?: { u: string; v: string; z: string };
  /** Numbers at the floor's ends and up the back edge: the limits of the box, and where zero sits as a height in [0, 1]. */
  ticks?: { u?: [string, string]; v?: [string, string]; z?: { lo: string; hi: string; zero?: number } };
  /** A dashed outline of the floor lifted to this height in [0, 1], such as where zero sits. */
  zeroPlane?: number;
  /** Lines drawn on the floor before the sheet, such as a unit's crease. */
  floorLines?: readonly { from: [number, number]; to: [number, number]; colour: string; width?: number; dash?: number[] }[];
  wireAlpha?: number;
  /** Fade everything, for a surface that is not the focus. */
  alpha?: number;
}

export interface SurfaceHandle {
  project: Projector;
  /** The vertex nearest a canvas point within `radius` pixels, or null. */
  nearest: (at: { x: number; y: number }, radius?: number) => { col: number; row: number; x: number; y: number } | null;
}

/** Draw `values` (row-major, `cols` by `rows` vertices over the floor) as a shaded mesh. */
export function drawSurface(
  ctx: CanvasRenderingContext2D,
  rect: SurfaceRect,
  palette: Palette,
  values: ArrayLike<number>,
  cols: number,
  rows: number,
  view: SurfaceView,
  options: SurfaceOptions,
): SurfaceHandle {
  const { zOf, colourAt, level, marks = [], labels, ticks, zeroPlane, floorLines = [], wireAlpha = 0.16, alpha = 1 } = options;
  const project = makeProjector(rect, view);
  const uOf = (col: number) => -1 + (2 * col) / (cols - 1);
  const vOf = (row: number) => -1 + (2 * row) / (rows - 1);
  const shadow = palette.isDark ? palette.bg : palette.text;

  const px = new Float64Array(cols * rows);
  const py = new Float64Array(cols * rows);
  const pz = new Float64Array(cols * rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const z = Math.max(0, Math.min(1, zOf(values[i])));
      const p = project(uOf(c), vOf(r), z);
      px[i] = p.x;
      py[i] = p.y;
      pz[i] = z;
    }
  }

  ctx.save();
  ctx.globalAlpha = alpha;

  // The floor and the box edges, always behind the surface.
  const corners = [project(-1, -1, 0), project(1, -1, 0), project(1, 1, 0), project(-1, 1, 0)];
  ctx.beginPath();
  corners.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  ctx.closePath();
  ctx.fillStyle = rgba(palette.textFaint, 0.07);
  ctx.fill();
  ctx.strokeStyle = rgba(palette.axis, 0.7);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = rgba(palette.axis, 0.5);
  for (const [u, v] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ]) {
    const a = project(u, v, 0);
    const b = project(u, v, 1);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  if (zeroPlane !== undefined && zeroPlane > 0.02) {
    const ring = [project(-1, -1, zeroPlane), project(1, -1, zeroPlane), project(1, 1, zeroPlane), project(-1, 1, zeroPlane)];
    ctx.beginPath();
    ring.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = rgba(palette.axis, 0.55);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const line of floorLines) {
    const a = project(line.from[0], line.from[1], 0);
    const b = project(line.to[0], line.to[1], 0);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = line.colour;
    ctx.lineWidth = line.width ?? 1.5;
    ctx.setLineDash(line.dash ?? []);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Light from the viewer's upper left, in the rotated frame.
  const light = [-0.3, -0.5, 0.8];
  const lightNorm = Math.hypot(light[0], light[1], light[2]);
  const cosA = Math.cos(view.azimuth);
  const sinA = Math.sin(view.azimuth);
  const du = 2 / (cols - 1);
  const dv = 2 / (rows - 1);

  // Facets, then marks, sorted back to front by index rather than by allocating a closure each.
  const facetCount = (rows - 1) * (cols - 1);
  const itemCount = facetCount + marks.length;
  const depth = new Float64Array(itemCount);
  const fills: string[] = new Array(facetCount);
  const markPoints: Projected[] = [];
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const k = r * (cols - 1) + c;
      const i00 = r * cols + c;
      const i10 = i00 + 1;
      const i01 = i00 + cols;
      const i11 = i01 + 1;
      const mean = (values[i00] + values[i10] + values[i01] + values[i11]) / 4;
      const gu = ((pz[i10] - pz[i00] + pz[i11] - pz[i01]) / 2 / du) * BOX_HEIGHT;
      const gv = ((pz[i01] - pz[i00] + pz[i11] - pz[i10]) / 2 / dv) * BOX_HEIGHT;
      const nu = -(gu * cosA - gv * sinA);
      const nv = -(gu * sinA + gv * cosA);
      const nNorm = Math.hypot(nu, nv, 1);
      const lit = Math.max(0, (nu * light[0] + nv * light[1] + light[2]) / (nNorm * lightNorm));
      const shade = 0.72 + 0.28 * lit;
      const cu = uOf(c) + du / 2;
      const cv = vOf(r) + dv / 2;
      fills[k] = mix(colourAt(mean, cu, cv), shadow, (1 - shade) * 0.7);
      depth[k] = cu * sinA + cv * cosA;
    }
  }
  marks.forEach((mark, m) => {
    const p = project(mark.u, mark.v, Math.max(0, Math.min(1, mark.z)));
    markPoints.push(p);
    depth[facetCount + m] = p.depth - 0.03;
  });
  const order = Array.from({ length: itemCount }, (_, i) => i);
  order.sort((a, b) => depth[b] - depth[a]);

  ctx.lineWidth = 0.6;
  const wire = rgba(palette.text, wireAlpha);
  for (const k of order) {
    if (k < facetCount) {
      const r = Math.floor(k / (cols - 1));
      const c = k % (cols - 1);
      const i00 = r * cols + c;
      const i10 = i00 + 1;
      const i01 = i00 + cols;
      const i11 = i01 + 1;
      ctx.beginPath();
      ctx.moveTo(px[i00], py[i00]);
      ctx.lineTo(px[i10], py[i10]);
      ctx.lineTo(px[i11], py[i11]);
      ctx.lineTo(px[i01], py[i01]);
      ctx.closePath();
      ctx.fillStyle = fills[k];
      ctx.fill();
      ctx.strokeStyle = wire;
      ctx.stroke();
    } else {
      const mark = marks[k - facetCount];
      const p = markPoints[k - facetCount];
      drawMarker(ctx, mark.shape ?? 'circle', p.x, p.y, mark.radius, mark.colour, rgba(palette.surface, 0.9));
      if (mark.ring) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, mark.radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = palette.accent;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.lineWidth = 0.6;
      }
    }
  }

  // The level line sits on top of the sheet, so it is drawn over it rather than sorted into it.
  if (level !== undefined) {
    const grid = { values: Float32Array.from(values as ArrayLike<number>), cols, rows, min: 0, max: 0 };
    const segments = isoSegments(grid, level);
    const z = Math.max(0, Math.min(1, zOf(level)));
    ctx.strokeStyle = rgba(palette.text, 0.9);
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let s = 0; s < segments.length; s += 4) {
      const a = project(uOf(segments[s]), vOf(segments[s + 1]), z);
      const b = project(uOf(segments[s + 2]), vOf(segments[s + 3]), z);
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
    ctx.stroke();
  }

  if (labels) {
    ctx.font = '11px ' + MONO_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const lu = project(0, -1.18, 0);
    const lv = project(1.18, 0, 0);
    const lz = project(-1, -1, 1.1);
    ctx.fillText(labels.u, lu.x, lu.y);
    ctx.fillText(labels.v, lv.x, lv.y);
    ctx.fillText(labels.z, lz.x, lz.y);
  }
  if (ticks) {
    ctx.font = '9px ' + MONO_STACK;
    ctx.fillStyle = palette.textFaint;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (ticks.u) {
      const a = project(-0.92, -1.14, 0);
      const b = project(0.92, -1.14, 0);
      ctx.fillText(ticks.u[0], a.x, a.y);
      ctx.fillText(ticks.u[1], b.x, b.y);
    }
    if (ticks.v) {
      const a = project(1.14, -0.92, 0);
      const b = project(1.14, 0.92, 0);
      ctx.fillText(ticks.v[0], a.x, a.y);
      ctx.fillText(ticks.v[1], b.x, b.y);
    }
    if (ticks.z) {
      // Up the farthest edge, so the numbers never cross the sheet.
      const far = corners.reduce((best, p, i) => (p.depth > corners[best].depth ? i : best), 0);
      const [cu, cv] = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ][far];
      const side = cu * Math.cos(view.azimuth) - cv * Math.sin(view.azimuth) < 0 ? 'right' : 'left';
      ctx.textAlign = side;
      const dx = side === 'right' ? -4 : 4;
      const lo = project(cu, cv, 0);
      const hi = project(cu, cv, 1);
      ctx.fillText(ticks.z.lo, lo.x + dx, lo.y);
      ctx.fillText(ticks.z.hi, hi.x + dx, hi.y);
      if (ticks.z.zero !== undefined && ticks.z.zero > 0.08 && ticks.z.zero < 0.92) {
        const zero = project(cu, cv, ticks.z.zero);
        ctx.strokeStyle = rgba(palette.axis, 0.7);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(zero.x - 3, zero.y);
        ctx.lineTo(zero.x + 3, zero.y);
        ctx.stroke();
        ctx.fillText('0', zero.x + dx, zero.y);
      }
    }
  }
  ctx.restore();

  const nearest = (at: { x: number; y: number }, radius = 12) => {
    let best = -1;
    let bestD = radius * radius;
    for (let i = 0; i < cols * rows; i++) {
      const d = (px[i] - at.x) ** 2 + (py[i] - at.y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0) return null;
    return { col: best % cols, row: Math.floor(best / cols), x: px[best], y: py[best] };
  };
  return { project, nearest };
}
