/** Sampling a scalar field on a grid and tracing its iso-lines. No canvas here, so it runs in tests. */

import type { Frame } from './canvas';

export interface SampledGrid {
  /** Row-major, sampled at cell centres. */
  values: Float32Array;
  cols: number;
  rows: number;
  min: number;
  max: number;
}

/** Sample `valueAt` at the centre of every cell of `cellSize` pixels across the frame. */
export function sampleGrid(
  frame: Frame,
  valueAt: (x: number, y: number) => number,
  cellSize: number,
): SampledGrid {
  const cols = Math.max(2, Math.ceil(frame.innerWidth / cellSize));
  const rows = Math.max(2, Math.ceil(frame.innerHeight / cellSize));
  const values = new Float32Array(cols * rows);
  let min = Infinity;
  let max = -Infinity;
  for (let row = 0; row < rows; row++) {
    const y = frame.y.invert(frame.top + ((row + 0.5) / rows) * frame.innerHeight);
    for (let col = 0; col < cols; col++) {
      const x = frame.x.invert(frame.left + ((col + 0.5) / cols) * frame.innerWidth);
      const v = valueAt(x, y);
      const safe = Number.isFinite(v) ? v : 0;
      values[row * cols + col] = safe;
      if (safe < min) min = safe;
      if (safe > max) max = safe;
    }
  }
  return { values, cols, rows, min, max };
}

/** Where the crossing sits between two samples, as a fraction from the first. */
function crossing(a: number, b: number): number {
  const denom = a - b;
  if (Math.abs(denom) < 1e-12) return 0.5;
  const t = a / denom;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * Marching squares at one level. Returns flat segments [c0, r0, c1, r1, ...] in grid units,
 * where integer (col, row) is a cell centre.
 */
export function isoSegments(grid: SampledGrid, level: number): number[] {
  const { values, cols, rows } = grid;
  const out: number[] = [];
  for (let row = 0; row < rows - 1; row++) {
    for (let col = 0; col < cols - 1; col++) {
      const tl = values[row * cols + col] - level;
      const tr = values[row * cols + col + 1] - level;
      const bl = values[(row + 1) * cols + col] - level;
      const br = values[(row + 1) * cols + col + 1] - level;
      const index = (tl > 0 ? 8 : 0) + (tr > 0 ? 4 : 0) + (br > 0 ? 2 : 0) + (bl > 0 ? 1 : 0);
      if (index === 0 || index === 15) continue;

      const top = [col + crossing(tl, tr), row];
      const bottom = [col + crossing(bl, br), row + 1];
      const left = [col, row + crossing(tl, bl)];
      const right = [col + 1, row + crossing(tr, br)];

      const push = (a: number[], b: number[]) => out.push(a[0], a[1], b[0], b[1]);
      switch (index) {
        case 1:
        case 14:
          push(left, bottom);
          break;
        case 2:
        case 13:
          push(bottom, right);
          break;
        case 3:
        case 12:
          push(left, right);
          break;
        case 4:
        case 11:
          push(top, right);
          break;
        case 6:
        case 9:
          push(top, bottom);
          break;
        case 7:
        case 8:
          push(left, top);
          break;
        case 5:
          push(left, top);
          push(bottom, right);
          break;
        case 10:
          push(left, bottom);
          push(top, right);
          break;
        default:
          break;
      }
    }
  }
  return out;
}

/**
 * `count` levels strictly inside [min, max]. Log mode follows the band edges of `drawContourMap`,
 * so lines and fills line up when count is its levels minus one.
 */
export function isoLevels(min: number, max: number, count: number, mode: 'linear' | 'log' = 'linear'): number[] {
  const out: number[] = [];
  if (!(max > min) || count <= 0) return out;
  for (let i = 1; i <= count; i++) {
    const u = i / (count + 1);
    const t = mode === 'log' ? (Math.pow(25, u) - 1) / 24 : u;
    out.push(min + (max - min) * t);
  }
  return out;
}
