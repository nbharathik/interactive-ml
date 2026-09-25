/** A small 3D renderer for layer volumes: an orthographic turntable, textured sheets, painter's order, hit testing. */

import { rgba } from './palette';
import type { Palette } from './palette';

export interface VolumeView {
  /** Turn about the vertical axis, in radians. */
  azimuth: number;
  /** Tilt down from the front, in radians. */
  elevation: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Projected {
  x: number;
  y: number;
  /** Larger is farther from the viewer. */
  depth: number;
}

export type VolumeProjector = (p: Vec3) => Projected;

export const DEFAULT_VOLUME_VIEW: VolumeView = { azimuth: -0.55, elevation: 0.3 };
const AZIMUTH_MAX = 1.25;
const ELEVATION_MIN = -0.2;
const ELEVATION_MAX = 0.95;

export function clampVolumeView(view: VolumeView): VolumeView {
  return {
    azimuth: Math.max(-AZIMUTH_MAX, Math.min(AZIMUTH_MAX, view.azimuth)),
    elevation: Math.max(ELEVATION_MIN, Math.min(ELEVATION_MAX, view.elevation)),
  };
}

/** The view after a drag of dx, dy pixels. */
export function dragVolumeView(from: VolumeView, dx: number, dy: number): VolumeView {
  return clampVolumeView({ azimuth: from.azimuth + dx * 0.01, elevation: from.elevation + dy * 0.007 });
}

export function volumeViewKey(view: VolumeView): string {
  return view.azimuth.toFixed(3) + ',' + view.elevation.toFixed(3);
}

/** World to camera: turn about y, then tilt about x. z points at the viewer. */
export function rotateVolume(view: VolumeView, p: Vec3): Vec3 {
  const cosA = Math.cos(view.azimuth);
  const sinA = Math.sin(view.azimuth);
  const cosE = Math.cos(view.elevation);
  const sinE = Math.sin(view.elevation);
  const x = p.x * cosA + p.z * sinA;
  const z1 = -p.x * sinA + p.z * cosA;
  const y = p.y * cosE - z1 * sinE;
  const z = p.y * sinE + z1 * cosE;
  return { x, y, z };
}

export interface VolumeFit {
  scale: number;
  centre: Vec3;
}

/** One scale for every allowed view, so turning never grows or shrinks the scene. */
export function fitVolume(points: readonly Vec3[], width: number, height: number, margin = 0.06): VolumeFit {
  if (points.length === 0) return { scale: 1, centre: { x: 0, y: 0, z: 0 } };
  let lo = { x: Infinity, y: Infinity, z: Infinity };
  let hi = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (const p of points) {
    lo = { x: Math.min(lo.x, p.x), y: Math.min(lo.y, p.y), z: Math.min(lo.z, p.z) };
    hi = { x: Math.max(hi.x, p.x), y: Math.max(hi.y, p.y), z: Math.max(hi.z, p.z) };
  }
  const centre = { x: (lo.x + hi.x) / 2, y: (lo.y + hi.y) / 2, z: (lo.z + hi.z) / 2 };
  let halfW = 1e-6;
  let halfH = 1e-6;
  // The usual turns only; an extreme angle may spill a little rather than shrink every view.
  for (let a = -1; a <= 1 + 1e-9; a += 0.25) {
    for (let e = -0.15; e <= 0.75 + 1e-9; e += 0.3) {
      const view = { azimuth: a, elevation: e };
      for (const p of points) {
        const r = rotateVolume(view, { x: p.x - centre.x, y: p.y - centre.y, z: p.z - centre.z });
        halfW = Math.max(halfW, Math.abs(r.x));
        halfH = Math.max(halfH, Math.abs(r.y));
      }
    }
  }
  const scale = Math.min((width * (1 - 2 * margin)) / (2 * halfW), (height * (1 - 2 * margin)) / (2 * halfH));
  return { scale, centre };
}

export function makeVolumeProjector(view: VolumeView, fit: VolumeFit, cx: number, cy: number): VolumeProjector {
  return (p) => {
    const r = rotateVolume(view, { x: p.x - fit.centre.x, y: p.y - fit.centre.y, z: p.z - fit.centre.z });
    return { x: cx + r.x * fit.scale, y: cy - r.y * fit.scale, depth: -r.z };
  };
}

/* ---------------- scene items ---------------- */

export interface SheetItem {
  kind: 'sheet';
  id: string;
  /** Top-left, top-right, bottom-right, bottom-left. */
  corners: [Vec3, Vec3, Vec3, Vec3];
  texture: CanvasImageSource | null;
  fill: string;
  stroke: string;
  alpha?: number;
  /** Pushes the item back or forward in the sort, in world units. */
  bias?: number;
}

export interface DiscItem {
  kind: 'disc';
  id: string;
  at: Vec3;
  /** Radius in pixels. */
  r: number;
  fill: string;
  stroke: string;
  label?: string;
  labelColour?: string;
  bias?: number;
}

export interface LineItem {
  kind: 'line';
  from: Vec3;
  to: Vec3;
  colour: string;
  width: number;
  alpha: number;
  dash?: number[];
  bias?: number;
}

export interface PolyItem {
  kind: 'poly';
  points: Vec3[];
  stroke: string;
  width: number;
  alpha: number;
  fill?: string;
  bias?: number;
}

export type VolumeItem = SheetItem | DiscItem | LineItem | PolyItem;

/** A drawn sheet or disc, with its screen shape for hit tests and outlines. */
export interface PlacedItem {
  id: string;
  kind: 'sheet' | 'disc';
  /** Screen polygon of a sheet, or the four points around a disc. */
  polygon: Array<{ x: number; y: number }>;
  centre: { x: number; y: number };
  r: number;
  depth: number;
}

function itemDepth(item: VolumeItem, project: VolumeProjector): number {
  const bias = item.bias ?? 0;
  switch (item.kind) {
    case 'sheet': {
      let d = 0;
      for (const c of item.corners) d += project(c).depth;
      return d / 4 + bias;
    }
    case 'disc':
      return project(item.at).depth + bias;
    case 'line':
      return (project(item.from).depth + project(item.to).depth) / 2 + bias;
    case 'poly': {
      let d = 0;
      for (const p of item.points) d += project(p).depth;
      return d / Math.max(1, item.points.length) + bias;
    }
    default:
      return 0;
  }
}

/** Draw every item far to near. Returns the sheets and discs with their screen shapes, near first. */
export function drawVolume(ctx: CanvasRenderingContext2D, items: readonly VolumeItem[], project: VolumeProjector): PlacedItem[] {
  const depths = items.map((item) => itemDepth(item, project));
  const order = items.map((_, i) => i).sort((a, b) => depths[b] - depths[a]);
  const placed: PlacedItem[] = [];
  for (const i of order) {
    const item = items[i];
    switch (item.kind) {
      case 'sheet': {
        const p = item.corners.map(project);
        ctx.save();
        ctx.globalAlpha = item.alpha ?? 1;
        ctx.beginPath();
        p.forEach((q, k) => (k === 0 ? ctx.moveTo(q.x, q.y) : ctx.lineTo(q.x, q.y)));
        ctx.closePath();
        ctx.fillStyle = item.fill;
        ctx.fill();
        if (item.texture) {
          ctx.save();
          ctx.clip();
          // Orthographic projection keeps a rectangle affine, so one transform maps the whole texture.
          ctx.transform(p[1].x - p[0].x, p[1].y - p[0].y, p[3].x - p[0].x, p[3].y - p[0].y, p[0].x, p[0].y);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(item.texture, 0, 0, 1, 1);
          ctx.restore();
        }
        ctx.strokeStyle = item.stroke;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
        const cx = (p[0].x + p[2].x) / 2;
        const cy = (p[0].y + p[2].y) / 2;
        placed.push({ id: item.id, kind: 'sheet', polygon: p, centre: { x: cx, y: cy }, r: 0, depth: depths[i] });
        break;
      }
      case 'disc': {
        const q = project(item.at);
        ctx.save();
        ctx.beginPath();
        ctx.arc(q.x, q.y, item.r, 0, Math.PI * 2);
        ctx.fillStyle = item.fill;
        ctx.fill();
        ctx.strokeStyle = item.stroke;
        ctx.lineWidth = 1.2;
        ctx.stroke();
        if (item.label) {
          ctx.fillStyle = item.labelColour ?? item.stroke;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.font = '500 ' + Math.max(9, Math.round(item.r * 1.1)) + 'px system-ui, sans-serif';
          ctx.fillText(item.label, q.x, q.y + 0.5);
        }
        ctx.restore();
        const r = item.r;
        placed.push({
          id: item.id,
          kind: 'disc',
          polygon: [
            { x: q.x - r, y: q.y - r },
            { x: q.x + r, y: q.y - r },
            { x: q.x + r, y: q.y + r },
            { x: q.x - r, y: q.y + r },
          ],
          centre: { x: q.x, y: q.y },
          r,
          depth: depths[i],
        });
        break;
      }
      case 'line': {
        const a = project(item.from);
        const b = project(item.to);
        ctx.save();
        ctx.globalAlpha = item.alpha;
        ctx.strokeStyle = item.colour;
        ctx.lineWidth = item.width;
        ctx.lineCap = 'round';
        if (item.dash) ctx.setLineDash(item.dash);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
        break;
      }
      case 'poly': {
        const p = item.points.map(project);
        ctx.save();
        ctx.globalAlpha = item.alpha;
        ctx.beginPath();
        p.forEach((q, k) => (k === 0 ? ctx.moveTo(q.x, q.y) : ctx.lineTo(q.x, q.y)));
        ctx.closePath();
        if (item.fill) {
          ctx.fillStyle = item.fill;
          ctx.fill();
        }
        ctx.strokeStyle = item.stroke;
        ctx.lineWidth = item.width;
        ctx.stroke();
        ctx.restore();
        break;
      }
      default:
        break;
    }
  }
  placed.sort((a, b) => a.depth - b.depth);
  return placed;
}

function inPolygon(polygon: ReadonlyArray<{ x: number; y: number }>, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** The nearest sheet or disc under a point, or null. */
export function hitVolume(placed: readonly PlacedItem[], x: number, y: number, pad = 3): PlacedItem | null {
  for (const item of placed) {
    if (item.kind === 'disc') {
      if (Math.hypot(item.centre.x - x, item.centre.y - y) <= item.r + pad) return item;
    } else if (inPolygon(item.polygon, x, y)) return item;
  }
  return null;
}

/** Bounding box of a screen polygon, for plates and outlines. */
export function polygonBox(polygon: ReadonlyArray<{ x: number; y: number }>): { x: number; y: number; w: number; h: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of polygon) {
    x0 = Math.min(x0, p.x);
    y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x);
    y1 = Math.max(y1, p.y);
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** The floor grid under a scene, drawn first. */
export function drawVolumeFloor(
  ctx: CanvasRenderingContext2D,
  project: VolumeProjector,
  palette: Palette,
  x0: number,
  x1: number,
  z0: number,
  z1: number,
  y: number,
  step: number,
): void {
  ctx.save();
  ctx.strokeStyle = rgba(palette.axis, 0.28);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = x0; x <= x1 + 1e-9; x += step) {
    const a = project({ x, y, z: z0 });
    const b = project({ x, y, z: z1 });
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  for (let z = z0; z <= z1 + 1e-9; z += step) {
    const a = project({ x: x0, y, z });
    const b = project({ x: x1, y, z });
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
  ctx.restore();
}
