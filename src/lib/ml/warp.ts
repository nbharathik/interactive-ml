/** How a network bends its input plane: grid lines traced through every layer, and a two-axis view of wider layers. */

import { forward, layerForward } from './mlp';
import type { Mlp, MlpSpec } from './mlp';

export interface GridLine {
  /** 0 for lines of constant x₂ (running along x₁), 1 for lines of constant x₁. */
  family: 0 | 1;
  /** The line through the origin. */
  axis: boolean;
  points: number[][];
}

/** Evenly spaced lines over [-1, 1]², or one segment for a one-dimensional input. */
export function gridLines(inputDim: number, lines = 11, samples = 40): GridLine[] {
  const coord = (i: number, n: number) => -1 + (2 * i) / n;
  if (inputDim === 1) {
    return [{ family: 0, axis: true, points: Array.from({ length: samples + 1 }, (_, s) => [coord(s, samples)]) }];
  }
  const out: GridLine[] = [];
  const mid = (lines - 1) / 2;
  for (let l = 0; l < lines; l++) {
    const c = coord(l, lines - 1);
    out.push({ family: 0, axis: l === mid, points: Array.from({ length: samples + 1 }, (_, s) => [coord(s, samples), c]) });
    out.push({ family: 1, axis: l === mid, points: Array.from({ length: samples + 1 }, (_, s) => [c, coord(s, samples)]) });
  }
  return out;
}

export interface LayerTrace {
  /** Wh + b for every input, the linear part. */
  z: number[][];
  /** f(z) for every input, what the next layer sees. */
  a: number[][];
}

/** Every layer's z and a for a list of inputs. */
export function traceLayers(net: Mlp, spec: MlpSpec, inputs: readonly (readonly number[])[]): LayerTrace[] {
  const traces: LayerTrace[] = net.layers.map(() => ({ z: [], a: [] }));
  for (const input of inputs) {
    const caches = forward(net, input, spec);
    caches.forEach((c, l) => {
      traces[l].z.push(c.z);
      traces[l].a.push(c.a);
    });
  }
  return traces;
}

/** The output layer alone, applied to a vector of the last hidden layer. */
export function outputFromHidden(net: Mlp, spec: MlpSpec, hidden: readonly number[]): number[] {
  return layerForward(net.layers[net.layers.length - 1], hidden, spec.activationParams).a;
}

export interface Basis {
  mean: number[];
  axes: [number[], number[]];
}

function dotOf(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Flip a direction so it points the same way from frame to frame. */
function oriented(v: number[]): number[] {
  const sum = v.reduce((a, b) => a + b, 0);
  const sign = Math.abs(sum) > 1e-9 ? Math.sign(sum) : Math.sign(v.find((x) => Math.abs(x) > 1e-9) ?? 1);
  return v.map((x) => x * sign);
}

function powerIteration(cov: number[][], start: number[], iterations = 60): number[] {
  let v = start.slice();
  for (let it = 0; it < iterations; it++) {
    const next = cov.map((row) => dotOf(row, v));
    const norm = Math.sqrt(dotOf(next, next));
    if (norm < 1e-12) return start;
    v = next.map((x) => x / norm);
  }
  return v;
}

/** The identity for two dimensions, else the two directions of largest spread. */
export function principalBasis(points: readonly (readonly number[])[]): Basis {
  const dim = points[0]?.length ?? 2;
  if (dim <= 2) return { mean: new Array<number>(dim).fill(0), axes: [[1, 0], [0, 1]] };
  const n = Math.max(1, points.length);
  const mean = new Array<number>(dim).fill(0);
  for (const p of points) for (let i = 0; i < dim; i++) mean[i] += p[i] / n;
  const cov = Array.from({ length: dim }, () => new Array<number>(dim).fill(0));
  for (const p of points) {
    for (let i = 0; i < dim; i++) {
      const di = p[i] - mean[i];
      for (let j = 0; j < dim; j++) cov[i][j] += (di * (p[j] - mean[j])) / n;
    }
  }
  const seed = (k: number) => Array.from({ length: dim }, (_, i) => 1 + 0.1 * Math.sin(i + 1 + k * 3));
  const first = oriented(powerIteration(cov, seed(0)));
  const lambda = dotOf(first, cov.map((row) => dotOf(row, first)));
  const deflated = cov.map((row, i) => row.map((v, j) => v - lambda * first[i] * first[j]));
  let second = powerIteration(deflated, seed(1));
  const along = dotOf(second, first);
  second = second.map((x, i) => x - along * first[i]);
  const norm = Math.sqrt(dotOf(second, second));
  second = norm > 1e-9 ? oriented(second.map((x) => x / norm)) : seed(1).map((_, i) => (i === 1 ? 1 : 0));
  return { mean, axes: [first, second] };
}

export function projectOnto(basis: Basis, point: readonly number[]): [number, number] {
  const centred = point.map((v, i) => v - basis.mean[i]);
  return [dotOf(centred, basis.axes[0]), dotOf(centred, basis.axes[1])];
}

/** Extent of a set of 2D points, padded by a fraction of the larger side and never degenerate. */
export function extent2(points: readonly (readonly [number, number])[], pad = 0.08): { x: [number, number]; y: [number, number] } {
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const [x, y] of points) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) return { x: [-1, 1], y: [-1, 1] };
  const side = Math.max(x1 - x0, y1 - y0, 1e-3);
  const margin = side * pad;
  return { x: [x0 - margin, x1 + margin], y: [y0 - margin, y1 + margin] };
}
