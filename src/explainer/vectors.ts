/** Vectors over the dataset, drawn as strips of coloured cells in the diagram and the detail card. */

import { scratchBitmap } from '../lib/viz/canvas';
import { toRgb } from '../lib/viz/palette';
import type { Palette } from '../lib/viz/palette';

/** How a value becomes a colour. */
export type VectorRamp = 'signed' | 'magnitude' | 'class' | 'probability';

export interface VectorColumn {
  /** 'x', 'ŷ', 'y'. */
  name: string;
  values: readonly number[];
  ramp?: VectorRamp;
  /** Fixed colour scale, so two columns can share one. Defaults to the largest magnitude. */
  scale?: number;
  decimals?: number;
}

export function maxAbs(values: readonly number[]): number {
  let m = 0;
  for (const v of values) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m;
}

export function columnScale(column: VectorColumn): number {
  if (column.scale !== undefined) return column.scale;
  if (column.ramp === 'class' || column.ramp === 'probability') return 1;
  return maxAbs(column.values) || 1;
}

/** The colour of one value, opaque, blended over the panel surface. */
export function vectorRgb(
  palette: Palette,
  ramp: VectorRamp | undefined,
  value: number,
  scale: number,
): [number, number, number] {
  const base = toRgb(palette.surfaceAlt);
  if (!Number.isFinite(value)) return blend(base, toRgb(palette.textFaint), 0.4);
  if (ramp === 'class') return blend(base, toRgb(value >= 0.5 ? palette.classB : palette.classA), 0.85);
  if (ramp === 'probability') {
    return blend(toRgb(palette.classA), toRgb(palette.classB), Math.max(0, Math.min(1, value)));
  }
  const t = scale > 0 ? Math.max(-1, Math.min(1, value / scale)) : 0;
  const hue = ramp === 'magnitude' || t >= 0 ? palette.positive : palette.negative;
  return blend(base, toRgb(hue), 0.1 + 0.8 * Math.abs(t));
}

export function vectorColour(
  palette: Palette,
  ramp: VectorRamp | undefined,
  value: number,
  scale: number,
): string {
  const [r, g, b] = vectorRgb(palette, ramp, value, scale);
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function blend(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

/** Row order shared by every strip and table on a page: by x, or by class then x. */
export function displayOrder(
  n: number,
  key: (index: number) => number,
  group?: (index: number) => number,
): number[] {
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => {
    if (group) {
      const g = group(a) - group(b);
      if (g !== 0) return g;
    }
    return key(a) - key(b);
  });
  return order;
}

/** Paint columns side by side into a box, one row per point in `order`. */
export function paintVectorThumbnail(
  ctx: CanvasRenderingContext2D,
  rect: { x: number; y: number; w: number; h: number },
  palette: Palette,
  columns: readonly VectorColumn[],
  order: readonly number[],
  /** Rows the last step trained on; everything else is drawn faded. */
  active?: ReadonlySet<number> | null,
): void {
  const n = order.length;
  if (n === 0 || columns.length === 0) return;
  // Pad each column with copies of itself so scaling never bleeds a neighbour in.
  const stride = 3;
  const width = columns.length * stride;
  const scratch = scratchBitmap(width, n);
  if (!scratch) return;

  const image = scratch.ctx.createImageData(width, n);
  const data = image.data;
  columns.forEach((column, c) => {
    const scale = columnScale(column);
    for (let r = 0; r < n; r++) {
      const index = order[r];
      const [red, green, blue] = vectorRgb(palette, column.ramp, column.values[index] ?? NaN, scale);
      const alpha = active && !active.has(index) ? 80 : 255;
      for (let k = 0; k < stride; k++) {
        const o = (r * width + c * stride + k) * 4;
        data[o] = red;
        data[o + 1] = green;
        data[o + 2] = blue;
        data[o + 3] = alpha;
      }
    }
  });
  scratch.ctx.putImageData(image, 0, 0);

  const gap = columns.length > 1 ? Math.max(1, rect.w * 0.04) : 0;
  const colW = (rect.w - gap * (columns.length - 1)) / columns.length;
  ctx.save();
  ctx.fillStyle = palette.surfaceAlt;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  columns.forEach((_, c) => {
    ctx.drawImage(scratch.canvas, c * stride + 1, 0, 1, n, rect.x + c * (colW + gap), rect.y, colW, rect.h);
  });
  ctx.restore();
}
