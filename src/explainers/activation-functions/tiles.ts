/** Every hidden unit over the input plane, as a flat tile: its weighted sum z or its activation a, coloured by sign, with the crease where z = 0. */

import type { ActivationFn } from '../../lib/ml/activations';
import { forward } from '../../lib/ml/mlp';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { MONO_STACK, clamp, drawMarker, linearScale } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { isoSegments } from '../../lib/viz/contours';
import { categorical, mix, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawIsoLines, drawSignedField } from '../../lib/viz/plots';

/** Samples per side of the input plane. */
export const TILE_N = 33;

export interface UnitField {
  z: Float32Array;
  a: Float32Array;
}

/** One hidden layer's units, with the colour scales the whole layer shares. */
export interface LayerFields {
  units: UnitField[];
  /** Largest |z| in the layer, and the extents of z and a. */
  zMax: number;
  zLo: number;
  zHi: number;
  aMin: number;
  aMax: number;
}

const uOf = (i: number) => -1 + (2 * i) / (TILE_N - 1);

/** Every hidden layer over [-1, 1]², one forward pass per vertex; empty when the input is not a plane. */
export function sampleUnits(state: TrainerState): LayerFields[] {
  const hidden = state.net.layers.length - 1;
  if (hidden <= 0 || state.net.layers[0].inSize !== 2) return [];
  const layers: LayerFields[] = state.net.layers.slice(0, hidden).map((l) => ({
    units: Array.from({ length: l.outSize }, () => ({ z: new Float32Array(TILE_N * TILE_N), a: new Float32Array(TILE_N * TILE_N) })),
    zMax: 1e-6,
    zLo: Infinity,
    zHi: -Infinity,
    aMin: Infinity,
    aMax: -Infinity,
  }));
  for (let r = 0; r < TILE_N; r++) {
    for (let c = 0; c < TILE_N; c++) {
      const caches = forward(state.net, [uOf(c), uOf(r)], state.spec);
      const i = r * TILE_N + c;
      for (let k = 0; k < hidden; k++) {
        const layer = layers[k];
        const { z, a } = caches[k];
        for (let j = 0; j < layer.units.length; j++) {
          layer.units[j].z[i] = z[j];
          layer.units[j].a[i] = a[j];
          if (Math.abs(z[j]) > layer.zMax) layer.zMax = Math.abs(z[j]);
          if (z[j] < layer.zLo) layer.zLo = z[j];
          if (z[j] > layer.zHi) layer.zHi = z[j];
          if (a[j] < layer.aMin) layer.aMin = a[j];
          if (a[j] > layer.aMax) layer.aMax = a[j];
        }
      }
    }
  }
  return layers;
}

/** One unit alone, for its card. */
export function sampleUnit(state: TrainerState, j: number, k: number): { field: UnitField; layer: LayerFields } | null {
  const layers = sampleUnits(state);
  const layer = layers[k - 1];
  const field = layer?.units[j];
  return field ? { field, layer } : null;
}

/** Maps a value to [-1, 1] for the colour: z against the layer's largest |z|; a against the range of f, or the layer's extent when f is unbounded. */
export function tileNorm(act: ActivationFn, layer: LayerFields, after: boolean): (v: number) => number {
  if (!after) return (z) => clamp(z / layer.zMax, -1, 1);
  const range = act.range;
  if (range && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
    const mid = (range[0] + range[1]) / 2;
    const half = (range[1] - range[0]) / 2;
    return (a) => clamp((a - mid) / half, -1, 1);
  }
  const half = Math.max(Math.abs(layer.aMin), Math.abs(layer.aMax), 1e-6);
  return (a) => clamp(a / half, -1, 1);
}

/** A sample of the plane read between the vertices. */
function bilinear(values: Float32Array): (x: number, y: number) => number {
  const last = TILE_N - 1;
  return (x, y) => {
    const fx = clamp(((x + 1) / 2) * last, 0, last);
    const fy = clamp(((y + 1) / 2) * last, 0, last);
    const c0 = Math.floor(fx);
    const r0 = Math.floor(fy);
    const c1 = Math.min(last, c0 + 1);
    const r1 = Math.min(last, r0 + 1);
    const tx = fx - c0;
    const ty = fy - r0;
    const top = values[r0 * TILE_N + c0] * (1 - tx) + values[r0 * TILE_N + c1] * tx;
    const bottom = values[r1 * TILE_N + c0] * (1 - tx) + values[r1 * TILE_N + c1] * tx;
    return top * (1 - ty) + bottom * ty;
  };
}

export interface TileOptions {
  /** 'z₁' or 'a₁', drawn in the corner when there is room. */
  label?: string;
  /** Training inputs in network units and their labels, drawn as small dots. */
  points?: readonly (readonly number[])[];
  labels?: readonly number[] | null;
  alpha?: number;
  /** Colour of the crease z = 0. */
  creaseColour?: string;
  /** Axis names outside the tile, for a tile that stands alone. */
  axes?: boolean;
  cellSize?: number;
}

export function tileFrame(rect: { x: number; y: number; w: number; h: number }): Frame {
  return {
    x: linearScale([-1, 1], [rect.x, rect.x + rect.w]),
    y: linearScale([-1, 1], [rect.y + rect.h, rect.y]),
    left: rect.x,
    top: rect.y,
    right: rect.x + rect.w,
    bottom: rect.y + rect.h,
    innerWidth: rect.w,
    innerHeight: rect.h,
  };
}

/** The tile: the field of one unit's values in the sign's colour, white at 0, the crease on top, the points over it. */
export function drawUnitTile(
  ctx: CanvasRenderingContext2D,
  rect: { x: number; y: number; w: number; h: number },
  palette: Palette,
  values: Float32Array,
  norm: (v: number) => number,
  z: Float32Array,
  options: TileOptions = {},
): Frame {
  const { label, points, labels, alpha = 1, creaseColour, axes = false, cellSize } = options;
  const frame = tileFrame(rect);
  const value = bilinear(values);
  const zAt = bilinear(z);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = palette.surface;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  drawSignedField(ctx, frame, palette, (x, y) => norm(value(x, y)), { cellSize: cellSize ?? (rect.w > 120 ? 4 : 3), alpha: 0.65, contour: false });
  drawIsoLines(ctx, frame, palette, zAt, [0], { cellSize: rect.w > 120 ? 4 : 3, colour: creaseColour ?? palette.text, width: rect.w > 120 ? 1.4 : 1, alpha: 0.7 });
  if (points && rect.w >= 40) {
    const radius = rect.w > 120 ? 2.2 : 1.4;
    points.forEach((p, i) => {
      const colour = labels ? categorical(palette, labels[i]) : palette.blue;
      drawMarker(ctx, 'circle', frame.x(p[0]), frame.y(p[1] ?? 0), radius, colour, rect.w > 120 ? rgba(palette.surface, 0.9) : undefined);
    });
  }
  ctx.strokeStyle = palette.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  if (label && rect.w >= 30) {
    ctx.font = '600 9px ' + MONO_STACK;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = palette.text;
    ctx.fillText(label, rect.x + 3, rect.y + 10);
  }
  if (axes) {
    ctx.font = '9px ' + MONO_STACK;
    ctx.fillStyle = palette.textFaint;
    ctx.textAlign = 'right';
    ctx.fillText('x₁ →', rect.x + rect.w - 4, rect.y + rect.h - 4);
    ctx.textAlign = 'left';
    ctx.fillText('x₂ ↑', rect.x + 4, rect.y + 10);
  }
  ctx.restore();
  return frame;
}

/* ---------------- the sheets ---------------- */

export type Segment = { from: [number, number]; to: [number, number]; colour: string; width?: number; dash?: number[] };

/** Where a unit's z crosses zero, as floor segments in [-1, 1]². */
export function creaseSegments(z: Float32Array, colour: string, width: number, dash: number[]): Segment[] {
  const grid = { values: z, cols: TILE_N, rows: TILE_N, min: 0, max: 0 };
  const raw = isoSegments(grid, 0);
  const at = (i: number) => -1 + (2 * i) / (TILE_N - 1);
  const out: Segment[] = [];
  for (let s = 0; s < raw.length; s += 4) out.push({ from: [at(raw[s]), at(raw[s + 1])], to: [at(raw[s + 2]), at(raw[s + 3])], colour, width, dash });
  return out;
}

/** Σ wⱼ·aⱼ + b over the plane, from a layer's sampled sheets. */
export function weightedSum(layer: LayerFields, w: readonly number[], b: number): Float32Array {
  const out = new Float32Array(TILE_N * TILE_N).fill(b);
  layer.units.forEach((unit, j) => {
    const wj = w[j] ?? 0;
    for (let i = 0; i < out.length; i++) out[i] += wj * unit.a[i];
  });
  return out;
}

/** The box a layer's sheets share: f's range when bounded; else what a reaches, and at t < 1 of the replay, still part of what z reached below, so the fold is seen rising to the floor. */
export function rowRange(act: ActivationFn, layer: LayerFields, t = 1): [number, number] {
  const range = act.range;
  if (range && Number.isFinite(range[0]) && Number.isFinite(range[1])) return [range[0], range[1]];
  const lo0 = Math.min(layer.zLo, layer.aMin, 0);
  const hi0 = Math.max(layer.zHi, layer.aMax, lo0 + 1e-6);
  const lo1 = Math.min(layer.aMin, 0);
  const hi1 = Math.max(layer.aMax, lo1 + 1e-6);
  return [lo0 + (lo1 - lo0) * t, hi0 + (hi1 - hi0) * t];
}

export function signedColour(palette: Palette, v: number, half: number, lit: boolean): string {
  const strength = 0.1 + 0.7 * Math.min(1, Math.abs(v) / half);
  return mix(palette.surface, lit ? palette.accent : v < 0 ? palette.negative : palette.positive, strength);
}
