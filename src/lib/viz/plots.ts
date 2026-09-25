/** Reusable plot routines over a canvas context and a `Frame`. */

import type { Frame } from './canvas';

import {
  chartScale,
  clipFrame,
  drawAxes,
  drawLabelPlate,
  drawMarker,
  drawPath,
  FONT_STACK,
  makeFrame,
  scratchBitmap,
  type MarkerShape,
} from './canvas';
import type { Palette } from './palette';
import { isoSegments, sampleGrid } from './contours';
import type { SampledGrid } from './contours';

// Re-exported so a visualization needs one import.
export { drawPath } from './canvas';
import { categorical, mix, rgba, toRgb } from './palette';

/* ---------------- class styling ---------------- */

/** Marker shape per class. */
export const CLASS_SHAPES: MarkerShape[] = ['circle', 'cross', 'triangle', 'square'];

export function classColour(palette: Palette, label: number): string {
  return categorical(palette, label);
}

export function classShape(label: number): MarkerShape {
  return CLASS_SHAPES[((label % CLASS_SHAPES.length) + CLASS_SHAPES.length) % CLASS_SHAPES.length];
}

/* ---------------- scatter ---------------- */

export interface ScatterPoint {
  x: number;
  y: number;
  label?: number;
}

export interface ScatterOptions {
  radius?: number;
  /** Override the colour per point, used to show predictions vs truth. */
  colourOf?: (point: ScatterPoint, index: number) => string;
  /** Draw an outline so points read against a coloured field. */
  outline?: boolean;
  /** Indices to draw larger and ringed. */
  highlight?: Set<number> | number[];
  /** Indices to draw faded. */
  dim?: Set<number> | number[];
  /** Use per-class marker shapes rather than all circles. */
  shapes?: boolean;
  opacity?: number;
}

function asSet(value: Set<number> | number[] | undefined): Set<number> | null {
  if (!value) return null;
  return value instanceof Set ? value : new Set(value);
}

export function drawScatter(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  points: readonly ScatterPoint[],
  options: ScatterOptions = {},
): void {
  const { colourOf, outline = true, shapes = false, opacity = 1 } = options;
  // Marks grow with the frame, at half the rate of the labels.
  const radius = (options.radius ?? 4) * (1 + (chartScale(frame) - 1) / 2);
  const highlight = asSet(options.highlight);
  const dim = asSet(options.dim);

  ctx.save();
  clipFrame(ctx, frame);
  const stroke = rgba(palette.surface, palette.isDark ? 0.75 : 0.9);

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const px = frame.x(p.x);
    const py = frame.y(p.y);
    if (px < frame.left - 12 || px > frame.right + 12) continue;
    if (py < frame.top - 12 || py > frame.bottom + 12) continue;

    const isDim = dim?.has(i) ?? false;
    const isHot = highlight?.has(i) ?? false;
    const base = colourOf ? colourOf(p, i) : classColour(palette, p.label ?? 0);
    const alpha = (isDim ? 0.22 : 1) * opacity;
    const colour = alpha >= 1 ? base : rgba(base, alpha);

    if (isHot) {
      ctx.beginPath();
      ctx.arc(px, py, radius + 5, 0, Math.PI * 2);
      ctx.strokeStyle = rgba(base, 0.55);
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const shape = shapes ? classShape(p.label ?? 0) : 'circle';
    drawMarker(
      ctx,
      shape,
      px,
      py,
      isHot ? radius + 1.2 : radius,
      colour,
      outline && !isDim ? stroke : undefined,
    );
  }
  ctx.restore();
}

/* ---------------- decision field ---------------- */

export interface FieldOptions {
  /** Pixel size of each sampled cell. Smaller is sharper and slower. */
  cellSize?: number;
  /** Maximum fill opacity. */
  alpha?: number;
  /** Draw the p = 0.5 contour on top. */
  contour?: boolean;
  contourColour?: string;
  contourWidth?: number;
  /** Render as discrete bands rather than a smooth ramp. */
  bands?: number;
}

/** Paint a scalar field by sampling `valueAt` (in [-1, 1]) on a coarse grid. */
export function drawSignedField(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  valueAt: (x: number, y: number) => number,
  options: FieldOptions = {},
): void {
  const { cellSize = 6, alpha = 0.45, contour = true, bands } = options;

  const cols = Math.max(2, Math.ceil(frame.innerWidth / cellSize));
  const rows = Math.max(2, Math.ceil(frame.innerHeight / cellSize));

  // Sample into an offscreen bitmap and let the canvas scale it up smoothly.
  const image = ctx.createImageData(cols, rows);
  const data = image.data;

  const negRgb = toRgb(palette.negative);
  const posRgb = toRgb(palette.positive);
  const values = new Float32Array(cols * rows);

  for (let row = 0; row < rows; row++) {
    const py = frame.top + ((row + 0.5) / rows) * frame.innerHeight;
    const yValue = frame.y.invert(py);
    for (let col = 0; col < cols; col++) {
      const px = frame.left + ((col + 0.5) / cols) * frame.innerWidth;
      const xValue = frame.x.invert(px);
      let v = valueAt(xValue, yValue);
      if (!Number.isFinite(v)) v = 0;
      v = v < -1 ? -1 : v > 1 ? 1 : v;
      if (bands && bands > 1) {
        const stepSize = 2 / bands;
        v = Math.round(v / stepSize) * stepSize;
      }
      values[row * cols + col] = v;

      const magnitude = Math.abs(v);
      const rgb = v < 0 ? negRgb : posRgb;
      const offset = (row * cols + col) * 4;
      data[offset] = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = Math.round(magnitude * alpha * 255);
    }
  }

  const bitmap = scratchBitmap(cols, rows);
  if (!bitmap) return;
  bitmap.ctx.putImageData(image, 0, 0);

  ctx.save();
  clipFrame(ctx, frame);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap.canvas, frame.left, frame.top, frame.innerWidth, frame.innerHeight);

  if (contour) {
    drawZeroContour(ctx, frame, values, cols, rows, {
      colour: options.contourColour ?? palette.text,
      width: options.contourWidth ?? 1.8,
    });
  }
  ctx.restore();
}

/** Paint a categorical field, one colour per predicted class. */
export function drawCategoryField(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  classAt: (x: number, y: number) => number,
  options: { cellSize?: number; alpha?: number; colourOf?: (label: number) => string } = {},
): void {
  const { cellSize = 6, alpha = 0.3 } = options;
  const colourOf = options.colourOf ?? ((label: number) => classColour(palette, label));

  const cols = Math.max(2, Math.ceil(frame.innerWidth / cellSize));
  const rows = Math.max(2, Math.ceil(frame.innerHeight / cellSize));
  const image = ctx.createImageData(cols, rows);
  const data = image.data;
  const cache = new Map<number, [number, number, number]>();

  for (let row = 0; row < rows; row++) {
    const py = frame.top + ((row + 0.5) / rows) * frame.innerHeight;
    const yValue = frame.y.invert(py);
    for (let col = 0; col < cols; col++) {
      const px = frame.left + ((col + 0.5) / cols) * frame.innerWidth;
      const label = classAt(frame.x.invert(px), yValue);
      let rgb = cache.get(label);
      if (!rgb) {
        rgb = toRgb(colourOf(label));
        cache.set(label, rgb);
      }
      const offset = (row * cols + col) * 4;
      data[offset] = rgb[0];
      data[offset + 1] = rgb[1];
      data[offset + 2] = rgb[2];
      data[offset + 3] = Math.round(alpha * 255);
    }
  }

  const bitmap = scratchBitmap(cols, rows);
  if (!bitmap) return;
  bitmap.ctx.putImageData(image, 0, 0);

  ctx.save();
  clipFrame(ctx, frame);
  // Nearest-neighbour scaling keeps class borders crisp.
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bitmap.canvas, frame.left, frame.top, frame.innerWidth, frame.innerHeight);
  ctx.restore();
}

/** Marching-squares zero crossing over a sampled grid. */
function drawZeroContour(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  values: Float32Array,
  cols: number,
  rows: number,
  style: { colour: string; width: number },
): void {
  const grid: SampledGrid = { values, cols, rows, min: 0, max: 0 };
  strokeSegments(ctx, frame, grid, isoSegments(grid, 0), style.colour, style.width, 0.75);
}

/** Stroke iso-line segments given in grid units. */
function strokeSegments(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  grid: SampledGrid,
  segments: readonly number[],
  colour: string,
  width: number,
  alpha: number,
): void {
  if (segments.length === 0) return;
  const cellW = frame.innerWidth / grid.cols;
  const cellH = frame.innerHeight / grid.rows;
  const px = (col: number) => frame.left + (col + 0.5) * cellW;
  const py = (row: number) => frame.top + (row + 0.5) * cellH;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.globalAlpha = alpha;
  for (let i = 0; i < segments.length; i += 4) {
    ctx.moveTo(px(segments[i]), py(segments[i + 1]));
    ctx.lineTo(px(segments[i + 2]), py(segments[i + 3]));
  }
  ctx.stroke();
  ctx.restore();
}

/** Iso-lines of a scalar field at the given levels. Pass `grid` to reuse a sample. */
export function drawIsoLines(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  valueAt: (x: number, y: number) => number,
  levels: readonly number[],
  options: { cellSize?: number; colour?: string; width?: number; alpha?: number; grid?: SampledGrid } = {},
): void {
  const grid = options.grid ?? sampleGrid(frame, valueAt, options.cellSize ?? 4);
  const colour = options.colour ?? palette.text;
  ctx.save();
  clipFrame(ctx, frame);
  for (const level of levels) {
    strokeSegments(ctx, frame, grid, isoSegments(grid, level), colour, options.width ?? 1, options.alpha ?? 0.35);
  }
  ctx.restore();
}

/* ---------------- line series ---------------- */

export interface Series {
  values: readonly number[];
  /** The x of the first value, for a history that keeps only its tail. Default 0. */
  x0?: number;
  colour: string;
  label?: string;
  width?: number;
  dash?: number[];
  /** Fill the area under the line. */
  fill?: boolean;
}

/** Whether a frame is tall enough for annotation plates. */
export function plateRoom(frame: Frame): boolean {
  return frame.innerHeight >= 120;
}

/** Plot one or more series against their index, the standard loss curve. */
export function drawSeries(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  series: readonly Series[],
  options: { markLast?: boolean; labelLast?: boolean } = {},
): void {
  ctx.save();
  clipFrame(ctx, frame);

  for (const s of series) {
    const points: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < s.values.length; i++) {
      const v = s.values[i];
      if (!Number.isFinite(v)) continue;
      points.push({ x: frame.x((s.x0 ?? 0) + i), y: frame.y(v) });
    }
    if (points.length === 0) continue;

    if (s.fill && points.length > 1) {
      const base = frame.y.domain[0] < 0 && frame.y.domain[1] > 0 ? frame.y(0) : frame.bottom;
      ctx.beginPath();
      ctx.moveTo(points[0].x, base);
      for (const p of points) ctx.lineTo(p.x, p.y);
      ctx.lineTo(points[points.length - 1].x, base);
      ctx.closePath();
      ctx.fillStyle = rgba(s.colour, palette.isDark ? 0.16 : 0.12);
      ctx.fill();
    }

    drawPath(ctx, points, s.colour, s.width ?? 2, s.dash);

    if (options.markLast !== false && points.length > 0) {
      const last = points[points.length - 1];
      ctx.beginPath();
      ctx.arc(last.x, last.y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = s.colour;
      ctx.fill();
      ctx.strokeStyle = rgba(palette.surface, 0.9);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  ctx.restore();

  if (options.labelLast && plateRoom(frame)) {
    // Push apart end labels closer than one label height.
    const placed: Array<{ label: string; colour: string; y: number }> = [];
    for (const s of series) {
      if (!s.label || s.values.length === 0) continue;
      const last = s.values[s.values.length - 1];
      if (!Number.isFinite(last)) continue;
      placed.push({
        label: s.label,
        colour: s.colour,
        y: Math.max(frame.top + 9, Math.min(frame.bottom - 9, frame.y(last))),
      });
    }

    placed.sort((a, b) => a.y - b.y);
    const minGap = 17;
    for (let i = 1; i < placed.length; i++) {
      if (placed[i].y - placed[i - 1].y < minGap) placed[i].y = placed[i - 1].y + minGap;
    }
    // Shift the stack back up if it ran past the bottom.
    const overshoot = placed.length > 0 ? placed[placed.length - 1].y - (frame.bottom - 9) : 0;
    if (overshoot > 0) {
      for (const item of placed) item.y -= overshoot;
    }

    for (const item of placed) {
      drawLabelPlate(ctx, item.label, frame.right - 6, item.y, palette, {
        align: 'right',
        colour: item.colour,
        bold: true,
        scale: chartScale(frame),
      });
    }
  }
}

/** Plot y = f(x) sampled across the frame's x-domain. */
export function drawFunction(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  fn: (x: number) => number,
  colour: string,
  options: { width?: number; dash?: number[]; samples?: number } = {},
): void {
  const { width = 2.4, dash, samples = 160 } = options;
  const points: Array<{ x: number; y: number }> = [];
  const [x0, x1] = frame.x.domain;
  for (let i = 0; i <= samples; i++) {
    const x = x0 + ((x1 - x0) * i) / samples;
    const y = fn(x);
    if (!Number.isFinite(y)) continue;
    points.push({ x: frame.x(x), y: frame.y(y) });
  }
  ctx.save();
  clipFrame(ctx, frame);
  drawPath(ctx, points, colour, width, dash);
  ctx.restore();
}

/** Vertical bars from each point to the fitted curve, the residuals. */
export function drawResiduals(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  points: readonly { x: number; y: number }[],
  predict: (x: number) => number,
  options: { showSquares?: boolean; colour?: string; alpha?: number } = {},
): void {
  const colour = options.colour ?? palette.red;
  const alpha = options.alpha ?? 0.4;
  ctx.save();
  clipFrame(ctx, frame);

  // One path so overlapping squares do not stack into an opaque block.
  const squares = new Path2D();
  for (const p of points) {
    const yhat = predict(p.x);
    if (!Number.isFinite(yhat)) continue;
    const px = frame.x(p.x);
    const py = frame.y(p.y);
    const pyHat = frame.y(yhat);

    if (options.showSquares) {
      const size = Math.abs(py - pyHat);
      squares.rect(px, Math.min(py, pyHat), size, size);
    }

    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, pyHat);
    ctx.strokeStyle = rgba(colour, alpha);
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  if (options.showSquares) {
    ctx.fillStyle = rgba(colour, 0.12);
    ctx.fill(squares);
    ctx.strokeStyle = rgba(colour, 0.3);
    ctx.lineWidth = 1;
    ctx.stroke(squares);
  }
  ctx.restore();
}

/* ---------------- contours ---------------- */

/** Filled contour map of a scalar function, log-compressed. */
export function drawContourMap(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  palette: Palette,
  valueAt: (x: number, y: number) => number,
  options: { cellSize?: number; levels?: number; grid?: SampledGrid } = {},
): void {
  const { cellSize = 4, levels = 12 } = options;
  const grid = options.grid ?? sampleGrid(frame, valueAt, cellSize);
  const { values, cols, rows, min: lo, max: hi } = grid;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-12) return;

  const image = ctx.createImageData(cols, rows);
  const data = image.data;
  const cool = toRgb(palette.basin);
  const warm = toRgb(palette.orange);

  for (let i = 0; i < values.length; i++) {
    // Log compression so the basin is not one flat colour.
    const t = Math.log1p((values[i] - lo) / (hi - lo) * 24) / Math.log(25);
    const banded = Math.floor(t * levels) / levels;
    const offset = i * 4;
    data[offset] = Math.round(cool[0] + (warm[0] - cool[0]) * banded);
    data[offset + 1] = Math.round(cool[1] + (warm[1] - cool[1]) * banded);
    data[offset + 2] = Math.round(cool[2] + (warm[2] - cool[2]) * banded);
    data[offset + 3] = 235;
  }

  const bitmap = scratchBitmap(cols, rows);
  if (!bitmap) return;
  bitmap.ctx.putImageData(image, 0, 0);

  ctx.save();
  clipFrame(ctx, frame);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(bitmap.canvas, frame.left, frame.top, frame.innerWidth, frame.innerHeight);
  ctx.restore();
}

/* ---------------- pixel grids ---------------- */

export interface PixelGridOptions {
  /** Signed blends negative and positive over the surface; magnitude runs surface to text. */
  ramp: 'signed' | 'magnitude';
  /** Value that maps to a full cell; defaults to the largest magnitude present. */
  scale?: number;
  /** Pixels between cells when they are large enough to separate. */
  gap?: number;
}

const LUT_STEPS = 32;
const lutCache = new Map<string, string[]>();

/** Quantised colour ramp so a grid of cells does not mix strings per cell. */
function pixelLut(palette: Palette, ramp: 'signed' | 'magnitude', pole: 'neg' | 'pos'): string[] {
  const key = palette.surfaceAlt + palette.text + ramp + pole;
  const cached = lutCache.get(key);
  if (cached) return cached;
  const target = ramp === 'magnitude' ? palette.text : pole === 'neg' ? palette.negative : palette.positive;
  const lut: string[] = [];
  for (let i = 0; i <= LUT_STEPS; i++) lut.push(mix(palette.surfaceAlt, target, i / LUT_STEPS));
  lutCache.set(key, lut);
  return lut;
}

/** A small image or feature map as crisp cells, centred in `rect`. */
export function drawPixelGrid(
  ctx: CanvasRenderingContext2D,
  rect: { x: number; y: number; width: number; height: number },
  values: ArrayLike<number>,
  cols: number,
  rows: number,
  palette: Palette,
  options: PixelGridOptions,
): void {
  if (cols <= 0 || rows <= 0) return;
  let scale = options.scale ?? 0;
  if (!scale) {
    for (let i = 0; i < values.length; i++) if (Math.abs(values[i]) > scale) scale = Math.abs(values[i]);
  }
  if (!(scale > 0)) scale = 1;
  const cell = Math.min(rect.width / cols, rect.height / rows);
  const gap = cell >= 6 ? (options.gap ?? 1) : 0;
  const left = rect.x + (rect.width - cell * cols) / 2;
  const top = rect.y + (rect.height - cell * rows) / 2;
  const neg = pixelLut(palette, options.ramp, 'neg');
  const pos = pixelLut(palette, options.ramp, 'pos');

  ctx.save();
  if (cell >= 2) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = values[r * cols + c];
        let t = Math.abs(v) / scale;
        if (!Number.isFinite(t)) t = 0;
        if (options.ramp === 'magnitude' && v < 0) t = 0;
        const idx = Math.round(Math.min(1, t) * LUT_STEPS);
        ctx.fillStyle = (v < 0 ? neg : pos)[idx];
        ctx.fillRect(left + c * cell, top + r * cell, cell - gap, cell - gap);
      }
    }
  } else {
    const image = ctx.createImageData(cols, rows);
    const data = image.data;
    const base = toRgb(palette.surfaceAlt);
    const negRgb = toRgb(options.ramp === 'magnitude' ? palette.text : palette.negative);
    const posRgb = toRgb(options.ramp === 'magnitude' ? palette.text : palette.positive);
    for (let i = 0; i < cols * rows; i++) {
      const v = values[i];
      let t = Math.abs(v) / scale;
      if (!Number.isFinite(t)) t = 0;
      if (options.ramp === 'magnitude' && v < 0) t = 0;
      t = Math.min(1, t);
      const rgb = v < 0 ? negRgb : posRgb;
      const o = i * 4;
      data[o] = Math.round(base[0] + (rgb[0] - base[0]) * t);
      data[o + 1] = Math.round(base[1] + (rgb[1] - base[1]) * t);
      data[o + 2] = Math.round(base[2] + (rgb[2] - base[2]) * t);
      data[o + 3] = 255;
    }
    const bitmap = scratchBitmap(cols, rows);
    if (bitmap) {
      bitmap.ctx.putImageData(image, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(bitmap.canvas, left, top, cell * cols, cell * rows);
    }
  }
  ctx.restore();
}

/* ---------------- bars ---------------- */

export interface BarDatum {
  label: string;
  value: number;
  colour?: string;
}

/** Horizontal bars centred on zero, weight and coefficient displays. */
export function drawSignedBars(
  ctx: CanvasRenderingContext2D,
  palette: Palette,
  bars: readonly BarDatum[],
  rect: { x: number; y: number; width: number; height: number },
  options: { labelWidth?: number; maxAbs?: number; valueFormat?: (v: number) => string } = {},
): void {
  if (bars.length === 0) return;
  const labelWidth = options.labelWidth ?? 46;
  const format = options.valueFormat ?? ((v: number) => v.toFixed(2));
  const maxAbs =
    options.maxAbs ?? Math.max(1e-6, ...bars.map((b) => Math.abs(b.value)));

  const rowHeight = rect.height / bars.length;
  const barHeight = Math.min(16, rowHeight * 0.62);
  const trackLeft = rect.x + labelWidth;
  const trackWidth = rect.width - labelWidth - 42;
  const centre = trackLeft + trackWidth / 2;

  ctx.save();
  ctx.font = '11px ' + FONT_STACK;
  ctx.textBaseline = 'middle';

  // Zero line
  ctx.strokeStyle = rgba(palette.border, 0.9);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(Math.round(centre) + 0.5, rect.y);
  ctx.lineTo(Math.round(centre) + 0.5, rect.y + rect.height);
  ctx.stroke();

  bars.forEach((bar, index) => {
    const cy = rect.y + rowHeight * (index + 0.5);
    const width = (Math.abs(bar.value) / maxAbs) * (trackWidth / 2);
    const colour = bar.colour ?? (bar.value < 0 ? palette.negative : palette.positive);

    ctx.fillStyle = palette.textMuted;
    ctx.textAlign = 'right';
    ctx.fillText(bar.label, trackLeft - 8, cy);

    ctx.fillStyle = rgba(colour, 0.85);
    if (bar.value < 0) ctx.fillRect(centre - width, cy - barHeight / 2, width, barHeight);
    else ctx.fillRect(centre, cy - barHeight / 2, width, barHeight);

    ctx.fillStyle = palette.text;
    ctx.textAlign = 'left';
    ctx.fillText(format(bar.value), trackLeft + trackWidth + 8, cy);
  });
  ctx.restore();
}

/** A soft glow behind an element. */
export function drawGlow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  colour: string,
): void {
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, rgba(colour, 0.35));
  gradient.addColorStop(1, rgba(colour, 0));
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

/** Blend a colour towards the panel surface, for "inactive" variants. */
export function fade(palette: Palette, colour: string, amount: number): string {
  return mix(colour, palette.surface, amount);
}

/** The axes alone, for a training chart at step 0: the frame is there before the first point is. */
export function drawEmptyAxes(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  palette: Palette,
  xLabel: string,
  yLabel: string,
  margins: { left: number; bottom: number; right: number; top: number } = { left: 54, bottom: 30, right: 14, top: 12 },
): void {
  const frame = makeFrame(width, height, [0, 1], [0, 1], margins);
  drawAxes(ctx, frame, palette, { xLabel, yLabel, xTicks: 2, yTicks: 2 });
}

/** A centred, word-wrapped message for a chart with nothing to draw yet. */
export function drawEmptyState(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  colour: string,
  message: string,
): void {
  ctx.save();
  ctx.font = '12px ' + FONT_STACK;
  ctx.fillStyle = colour;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const words = message.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const candidate = current ? current + ' ' + word : word;
    if (ctx.measureText(candidate).width > width - 40 && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  lines.forEach((line, i) => {
    ctx.fillText(line, width / 2, height / 2 + (i - (lines.length - 1) / 2) * 16);
  });
  ctx.restore();
}
