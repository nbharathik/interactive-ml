/** The two-weight picture: the squared error over two weights as ellipses, the penalty region, and where they touch. */

import type { Prepared } from './elasticNet';

export type Point2 = [number, number];

/** Half a quadratic form over two weights: value = base + ½ (v - c)ᵀ G (v - c). */
export interface Quadratic2 {
  g11: number;
  g12: number;
  g22: number;
  /** The unconstrained minimiser. */
  centre: Point2;
  /** The value at the centre. */
  base: number;
}

/** The smooth loss restricted to weights i and j, every other weight held where it is. */
export function sliceQuadratic(prep: Prepared, w: readonly number[], i: number, j: number): Quadratic2 {
  const { gram, xty } = prep;
  let hi = xty[i];
  let hj = xty[j];
  for (let m = 0; m < prep.p; m++) {
    if (m === i || m === j) continue;
    const v = Number.isFinite(w[m]) ? w[m] : 0;
    hi -= gram[i][m] * v;
    hj -= gram[j][m] * v;
  }
  let g11 = gram[i][i];
  let g22 = gram[j][j];
  const g12 = gram[i][j];
  let det = g11 * g22 - g12 * g12;
  if (det < 1e-9) {
    // Near-perfect correlation: a whisper on the diagonal keeps the centre finite.
    g11 += 1e-4;
    g22 += 1e-4;
    det = g11 * g22 - g12 * g12;
  }
  const centre: Point2 = [(g22 * hi - g12 * hj) / det, (g11 * hj - g12 * hi) / det];
  // The full loss also carries the fixed weights' share; only differences matter here.
  const base = -0.5 * (centre[0] * hi + centre[1] * hj);
  return { g11, g12, g22, centre, base };
}

export function quadraticAt(q: Quadratic2, v: Point2): number {
  const a = v[0] - q.centre[0];
  const b = v[1] - q.centre[1];
  return q.base + 0.5 * (q.g11 * a * a + 2 * q.g12 * a * b + q.g22 * b * b);
}

/** Points on the ellipse where the quadratic equals `value`, or none if it is below the minimum. */
export function ellipse(q: Quadratic2, value: number, samples = 96): Point2[] {
  const rise = 2 * (value - q.base);
  if (rise <= 0) return [];
  // Eigen-decomposition of the symmetric 2 x 2 form.
  const tr = q.g11 + q.g22;
  const disc = Math.sqrt(Math.max(0, ((q.g11 - q.g22) / 2) ** 2 + q.g12 * q.g12));
  const e1 = tr / 2 + disc;
  const e2 = Math.max(1e-9, tr / 2 - disc);
  const angle = Math.abs(q.g12) < 1e-12 && q.g11 >= q.g22 ? 0 : Math.atan2(e1 - q.g11, q.g12);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const r1 = Math.sqrt(rise / e1);
  const r2 = Math.sqrt(rise / e2);
  const out: Point2[] = [];
  for (let k = 0; k <= samples; k++) {
    const t = (2 * Math.PI * k) / samples;
    const u = r1 * Math.cos(t);
    const v = r2 * Math.sin(t);
    out.push([q.centre[0] + u * cos - v * sin, q.centre[1] + u * sin + v * cos]);
  }
  return out;
}

/* ---------------- the penalty region ---------------- */

/** The elastic net penalty over two weights: α (|a| + |b|) + ½ (1 - α)(a² + b²). */
export function shapeValue(alpha: number, v: Point2): number {
  const [a, b] = v;
  return alpha * (Math.abs(a) + Math.abs(b)) + ((1 - alpha) / 2) * (a * a + b * b);
}

/** Where the region's boundary crosses an axis, for a given level. */
export function axisRadius(alpha: number, level: number): number {
  if (level <= 0) return 0;
  if (alpha >= 1) return level;
  return (-alpha + Math.sqrt(alpha * alpha + 2 * (1 - alpha) * level)) / (1 - alpha);
}

export function levelOf(alpha: number, radius: number): number {
  return alpha * radius + ((1 - alpha) / 2) * radius * radius;
}

/** The boundary point in direction θ. */
export function boundaryPoint(alpha: number, level: number, theta: number): Point2 {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const s = Math.abs(cos) + Math.abs(sin);
  let rho: number;
  if (alpha >= 1) rho = level / s;
  else rho = (-alpha * s + Math.sqrt(alpha * alpha * s * s + 2 * (1 - alpha) * level)) / (1 - alpha);
  return [rho * cos, rho * sin];
}

export function boundary(alpha: number, level: number, samples = 180): Point2[] {
  const out: Point2[] = [];
  for (let k = 0; k <= samples; k++) out.push(boundaryPoint(alpha, level, (2 * Math.PI * k) / samples));
  return out;
}

const SCAN = 720;
const SNAP = 1e-3;

/** The point of the region with the lowest quadratic: the centre if it is inside, else on the boundary. */
export function constrainedMinimum(q: Quadratic2, alpha: number, level: number): Point2 {
  if (level <= 0) return [0, 0];
  if (shapeValue(alpha, q.centre) <= level) return q.centre;
  const at = (theta: number) => quadraticAt(q, boundaryPoint(alpha, level, theta));
  let best = 0;
  let bestValue = Infinity;
  for (let k = 0; k < SCAN; k++) {
    const theta = (2 * Math.PI * k) / SCAN;
    const value = at(theta);
    if (value < bestValue) {
      bestValue = value;
      best = theta;
    }
  }
  let lo = best - (2 * Math.PI) / SCAN;
  let hi = best + (2 * Math.PI) / SCAN;
  for (let iter = 0; iter < 50; iter++) {
    const m1 = lo + (hi - lo) / 3;
    const m2 = hi - (hi - lo) / 3;
    if (at(m1) < at(m2)) hi = m2;
    else lo = m1;
  }
  let theta = (lo + hi) / 2;
  // A corner of the diamond is exactly on an axis; land on it rather than a hair off.
  const quarter = Math.PI / 2;
  const nearest = Math.round(theta / quarter) * quarter;
  if (Math.abs(theta - nearest) < SNAP && at(nearest) <= at(theta) + 1e-12) theta = nearest;
  const point = boundaryPoint(alpha, level, theta);
  return [Math.abs(point[0]) < 1e-12 ? 0 : point[0], Math.abs(point[1]) < 1e-12 ? 0 : point[1]];
}
