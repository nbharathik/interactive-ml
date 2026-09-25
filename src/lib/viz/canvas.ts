/** Canvas primitives: DPR sizing, scales, plot frames and drawing helpers. */

import type { Palette } from './palette';
import { rgba } from './palette';

/* ---------------- sizing ---------------- */

export interface CanvasContext {
  ctx: CanvasRenderingContext2D;
  /** CSS pixel width (what your drawing code should use) */
  width: number;
  /** CSS pixel height */
  height: number;
  dpr: number;
}

/** Size a canvas for the device pixel ratio and return a context in CSS pixels. */
export function setupCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasContext | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);

  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, width: w, height: h, dpr };
}

let scratch: HTMLCanvasElement | null = null;

/** One offscreen bitmap, reused by every draw that scales a small image up; draw it before the next call. */
export function scratchBitmap(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null {
  if (typeof document === 'undefined') return null;
  scratch ??= document.createElement('canvas');
  if (scratch.width !== width) scratch.width = width;
  if (scratch.height !== height) scratch.height = height;
  const ctx = scratch.getContext('2d');
  return ctx ? { canvas: scratch, ctx } : null;
}

/** Convert a pointer event to canvas-local CSS pixel coordinates. */
export function pointerPos(
  canvas: HTMLCanvasElement,
  event: { clientX: number; clientY: number },
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width === 0 ? 1 : canvas.clientWidth / rect.width;
  const scaleY = rect.height === 0 ? 1 : canvas.clientHeight / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

/* ---------------- scales ---------------- */

export interface Scale {
  (value: number): number;
  invert(pixel: number): number;
  domain: [number, number];
  range: [number, number];
}

/** A linear scale mapping a data domain onto a pixel range. */
export function linearScale(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = ((value: number) => r0 + ((value - d0) / span) * (r1 - r0)) as Scale;
  fn.invert = (pixel: number) => d0 + ((pixel - r0) / (r1 - r0 || 1)) * span;
  fn.domain = domain;
  fn.range = range;
  return fn;
}

/** Expand a data extent by a fraction so points never touch the frame edge. */
export function padExtent(
  min: number,
  max: number,
  fraction = 0.08,
  minSpan = 1e-6,
): [number, number] {
  let lo = min;
  let hi = max;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi - lo < minSpan) {
    const mid = (hi + lo) / 2;
    lo = mid - 0.5;
    hi = mid + 0.5;
  }
  const pad = (hi - lo) * fraction;
  return [lo - pad, hi + pad];
}

export function extentOf(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) return [0, 1];
  return [lo, hi];
}

/** "Nice" round tick values covering a domain, at most `count`-ish of them. */
export function ticks(domain: [number, number], count = 6): number[] {
  const [d0, d1] = domain;
  const span = d1 - d0;
  if (!Number.isFinite(span) || span === 0) return [d0];
  const rawStep = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(rawStep))));
  const norm = rawStep / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(d0 / step) * step;
  const out: number[] = [];
  // Bounded: a step below the float spacing at `start` would never move `v`.
  for (let i = 0, v = start; v <= d1 + step * 1e-6 && i <= 4 * count + 4; i++, v = start + i * step) {
    out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  return out;
}

/** Format a number for an axis label without trailing noise. */
export function formatTick(value: number, step?: number): string {
  const magnitude = Math.abs(step ?? value);
  if (value === 0) return '0';
  if (magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)) {
    return value.toExponential(1).replace('e+', 'e');
  }
  const decimals = magnitude >= 100 ? 0 : magnitude >= 10 ? 0 : magnitude >= 1 ? 1 : 2;
  return value.toFixed(decimals).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

/* ---------------- plot frame ---------------- */

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const DEFAULT_MARGIN: Margin = { top: 12, right: 14, bottom: 30, left: 40 };

export interface Frame {
  x: Scale;
  y: Scale;
  /** inner plot rectangle in CSS pixels */
  left: number;
  top: number;
  right: number;
  bottom: number;
  innerWidth: number;
  innerHeight: number;
}

export function makeFrame(
  width: number,
  height: number,
  xDomain: [number, number],
  yDomain: [number, number],
  margin: Partial<Margin> & { equal?: boolean } = {},
): Frame {
  const m = { ...DEFAULT_MARGIN, ...margin };
  const left = m.left;
  const top = m.top;
  const right = Math.max(left + 1, width - m.right);
  const bottom = Math.max(top + 1, height - m.bottom);
  if (m.equal) {
    // Same pixels per unit on both axes: widen the roomier domain about its centre.
    const ppu = Math.min(
      (right - left) / (xDomain[1] - xDomain[0] || 1),
      (bottom - top) / (yDomain[1] - yDomain[0] || 1),
    );
    const xMid = (xDomain[0] + xDomain[1]) / 2;
    const yMid = (yDomain[0] + yDomain[1]) / 2;
    const xHalf = (right - left) / ppu / 2;
    const yHalf = (bottom - top) / ppu / 2;
    xDomain = [xMid - xHalf, xMid + xHalf];
    yDomain = [yMid - yHalf, yMid + yHalf];
  }
  return {
    x: linearScale(xDomain, [left, right]),
    // y is inverted: data max at the top of the frame
    y: linearScale(yDomain, [bottom, top]),
    left,
    top,
    right,
    bottom,
    innerWidth: right - left,
    innerHeight: bottom - top,
  };
}

export interface AxesOptions {
  xLabel?: string;
  yLabel?: string;
  xTicks?: number;
  yTicks?: number;
  showGrid?: boolean;
  showZeroLines?: boolean;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  yLabelColour?: string;
}

/** Type scale for a chart: bigger frames get slightly bigger labels. */
export function chartScale(frame: Frame): number {
  return clamp(Math.min(frame.innerWidth, frame.innerHeight) / 300, 1, 1.4);
}

/** Draw grid lines, axis lines, ticks and labels for a frame. */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  options: AxesOptions = {},
): void {
  const {
    xLabel,
    yLabel,
    xTicks = 6,
    yTicks = 5,
    showGrid = true,
    showZeroLines = false,
    formatX = formatTick,
    formatY = formatTick,
    yLabelColour,
  } = options;

  const xs = ticks(frame.x.domain, Math.max(2, Math.min(xTicks, Math.floor(frame.innerWidth / 60))));
  const ys = ticks(frame.y.domain, Math.max(2, Math.min(yTicks, Math.floor(frame.innerHeight / 28))));

  const s = chartScale(frame);
  const fontSize = Math.round(10 * s);
  const gap = Math.round(6 * s);
  const lineGap = Math.round(8 * s);

  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = fontSize + 'px ' + FONT_STACK;
  ctx.textBaseline = 'middle';

  if (showGrid) {
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    for (const t of xs) {
      const px = Math.round(frame.x(t)) + 0.5;
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
    }
    for (const t of ys) {
      const py = Math.round(frame.y(t)) + 0.5;
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
  }

  if (showZeroLines) {
    ctx.strokeStyle = palette.axis;
    ctx.beginPath();
    if (frame.x.domain[0] < 0 && frame.x.domain[1] > 0) {
      const px = Math.round(frame.x(0)) + 0.5;
      ctx.moveTo(px, frame.top);
      ctx.lineTo(px, frame.bottom);
    }
    if (frame.y.domain[0] < 0 && frame.y.domain[1] > 0) {
      const py = Math.round(frame.y(0)) + 0.5;
      ctx.moveTo(frame.left, py);
      ctx.lineTo(frame.right, py);
    }
    ctx.stroke();
  }

  // Axis lines
  ctx.strokeStyle = palette.border;
  ctx.beginPath();
  ctx.moveTo(frame.left + 0.5, frame.top);
  ctx.lineTo(frame.left + 0.5, frame.bottom + 0.5);
  ctx.lineTo(frame.right, frame.bottom + 0.5);
  ctx.stroke();

  // Tick labels, skipping any that would collide with the previous one.
  ctx.fillStyle = palette.textFaint;
  ctx.textAlign = 'center';
  const xStep = xs.length > 1 ? xs[1] - xs[0] : undefined;
  let lastRight = -Infinity;
  for (const t of xs) {
    const text = formatX(t, xStep as never);
    const centre = frame.x(t);
    const halfWidth = ctx.measureText(text).width / 2;
    if (centre - halfWidth < lastRight + gap) continue;
    ctx.fillText(text, centre, frame.bottom + 12);
    lastRight = centre + halfWidth;
  }

  ctx.textAlign = 'right';
  const yStep = ys.length > 1 ? ys[1] - ys[0] : undefined;
  let lastTop = Infinity;
  let maxTickW = 0;
  for (const t of ys) {
    const centre = frame.y(t);
    if (centre + lineGap > lastTop) continue;
    const label = formatY(t, yStep as never);
    maxTickW = Math.max(maxTickW, ctx.measureText(label).width);
    ctx.fillText(label, frame.left - 6, centre);
    lastTop = centre - lineGap;
  }

  // Axis titles
  ctx.fillStyle = palette.textMuted;
  ctx.font = '600 ' + fontSize + 'px ' + FONT_STACK;
  if (xLabel) {
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, (frame.left + frame.right) / 2, frame.bottom + 25);
  }
  // The title sits left of the tick labels; skip it when there is no room.
  const titleX = Math.min(11, frame.left - 6 - maxTickW - 9);
  if (yLabel && titleX >= 6) {
    // No room beside the frame: centre it on the canvas, condensed if needed.
    const canvasHeight = ctx.canvas.clientHeight || frame.bottom;
    const labelWidth = ctx.measureText(yLabel).width;
    const inFrame = labelWidth <= frame.innerHeight;
    ctx.fillStyle = yLabelColour ?? palette.textMuted;
    ctx.save();
    ctx.translate(titleX, inFrame ? (frame.top + frame.bottom) / 2 : canvasHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yLabel, 0, 0, Math.max(20, canvasHeight - 6));
    ctx.restore();
  }
  ctx.restore();
}

export const FONT_STACK =
  "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
export const MONO_STACK = "'JetBrains Mono', ui-monospace, Consolas, monospace";

/* ---------------- drawing helpers ---------------- */

export function clipFrame(ctx: CanvasRenderingContext2D, frame: Frame): void {
  ctx.beginPath();
  ctx.rect(frame.left, frame.top, frame.innerWidth, frame.innerHeight);
  ctx.clip();
}

export function drawPoint(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fill: string,
  stroke?: string,
  lineWidth = 1.5,
): void {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

/** A cross marker. */
export function drawCross(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  stroke: string,
  lineWidth = 2,
): void {
  const r = radius * 0.85;
  ctx.beginPath();
  ctx.moveTo(x - r, y - r);
  ctx.lineTo(x + r, y + r);
  ctx.moveTo(x + r, y - r);
  ctx.lineTo(x - r, y + r);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.stroke();
  ctx.lineCap = 'butt';
}

export function drawSquare(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fill: string,
  stroke?: string,
): void {
  const s = radius * 1.7;
  ctx.beginPath();
  ctx.rect(x - s / 2, y - s / 2, s, s);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

export function drawTriangle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fill: string,
  stroke?: string,
): void {
  const r = radius * 1.2;
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r * 0.87, y + r * 0.5);
  ctx.lineTo(x - r * 0.87, y + r * 0.5);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

export type MarkerShape = 'circle' | 'cross' | 'square' | 'triangle';

export function drawMarker(
  ctx: CanvasRenderingContext2D,
  shape: MarkerShape,
  x: number,
  y: number,
  radius: number,
  colour: string,
  strokeColour?: string,
): void {
  switch (shape) {
    case 'cross':
      drawCross(ctx, x, y, radius, colour);
      break;
    case 'square':
      drawSquare(ctx, x, y, radius, colour, strokeColour);
      break;
    case 'triangle':
      drawTriangle(ctx, x, y, radius, colour, strokeColour);
      break;
    default:
      drawPoint(ctx, x, y, radius, colour, strokeColour);
  }
}

/** Polyline through pre-projected pixel points. */
export function drawPath(
  ctx: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number }>,
  stroke: string,
  lineWidth = 2,
  dash?: number[],
): void {
  if (points.length < 2) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (dash) ctx.setLineDash(dash);
  ctx.stroke();
  ctx.restore();
}

/** A rounded rectangle path without depending on roundRect. */
export function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/** A floating label on a translucent plate. */
export function drawLabelPlate(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  palette: Palette,
  options: { align?: CanvasTextAlign; colour?: string; bold?: boolean; scale?: number } = {},
): void {
  const { align = 'left', colour = palette.text, bold = false, scale = 1 } = options;
  ctx.save();
  ctx.font = (bold ? '600 ' : '') + Math.round(11 * scale) + 'px ' + FONT_STACK;
  const metrics = ctx.measureText(text);
  const padX = 5;
  const padY = 3;
  const w = metrics.width + padX * 2;
  const h = Math.round(16 * scale);
  const left = align === 'right' ? x - w : align === 'center' ? x - w / 2 : x;
  roundRect(ctx, left, y - h / 2, w, h, 4);
  ctx.fillStyle = rgba(palette.surface, 0.88);
  ctx.fill();
  ctx.strokeStyle = rgba(palette.border, 0.9);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = colour;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, left + padX, y + padY - 3 + 0.5);
  ctx.restore();
}

/** Clamp. */
export function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/** Linear interpolation. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
