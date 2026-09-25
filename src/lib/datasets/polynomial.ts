/** One input, polynomial features: the wave from the blog post, so overfitting can be seen as a curve. */

import { gauss, makeRng, shuffled, uniform } from '../math/rng';
import { sup } from '../math/stats';
import type { TabularData } from './tabular';

export interface PolynomialConfig {
  count: number;
  /** Highest power: features x, x², ... up to x^degree. */
  degree: number;
  /** Noise as a fraction of the signal's spread. */
  noise: number;
  seed: number;
  trainFraction: number;
}

export interface PolynomialData extends TabularData {
  /** The raw input of every row. */
  x: number[];
  xRange: [number, number];
  yRange: [number, number];
  truth: (x: number) => number;
}

/** Centred on zero so the powers of x stay as uncorrelated as raw powers can be. */
export const X_RANGE: [number, number] = [-2.8, 2.8];

export function powerName(power: number): string {
  return power === 1 ? 'x' : 'x' + sup(power);
}

export function truthCurve(x: number): number {
  return Math.sin(1.2 * x + 3.6) + 0.4 * x + 0.2;
}

export function powersOf(x: number, degree: number): number[] {
  const row: number[] = [];
  let v = 1;
  for (let k = 1; k <= degree; k++) {
    v *= x;
    row.push(v);
  }
  return row;
}

export function generatePolynomial(config: PolynomialConfig): PolynomialData {
  const degree = Math.max(1, Math.round(config.degree));
  const n = Math.max(4, Math.round(config.count));
  const rng = makeRng(config.seed);
  const x: number[] = [];
  for (let i = 0; i < n; i++) x.push(uniform(rng, X_RANGE[0], X_RANGE[1]));
  x.sort((a, b) => a - b);
  const signal = x.map(truthCurve);
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  const spread = Math.sqrt(signal.reduce((a, s) => a + (s - mean) * (s - mean), 0) / n) || 1;
  const y = signal.map((s) => s + gauss(rng, 0, config.noise * spread));

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of y) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  for (let i = 0; i <= 40; i++) {
    const v = truthCurve(X_RANGE[0] + ((X_RANGE[1] - X_RANGE[0]) * i) / 40);
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const pad = Math.max(0.5, (hi - lo) * 0.2);

  const indices = shuffled(rng, Array.from({ length: n }, (_, i) => i));
  const trainCount = Math.max(2, Math.min(n - 1, Math.round(n * config.trainFraction)));
  // The frame hugs the samples: outside them a high-degree curve says nothing worth a picture.
  const edge = 0.15;
  return {
    kind: 'tabular',
    X: x.map((v) => powersOf(v, degree)),
    y,
    featureNames: Array.from({ length: degree }, (_, k) => powerName(k + 1)),
    trueWeights: new Array<number>(degree).fill(0),
    trueBias: 0,
    groups: [Array.from({ length: degree }, (_, k) => k)],
    scales: new Array<number>(degree).fill(1),
    trainIndex: indices.slice(0, trainCount).sort((a, b) => a - b),
    testIndex: indices.slice(trainCount).sort((a, b) => a - b),
    x,
    xRange: [x[0] - edge, x[n - 1] + edge],
    yRange: [lo - pad, hi + pad],
    truth: truthCurve,
  };
}

export function isPolynomial(data: TabularData): data is PolynomialData {
  return 'truth' in data;
}
