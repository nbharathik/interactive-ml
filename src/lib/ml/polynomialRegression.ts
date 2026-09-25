/** Polynomial regression by least squares. Fits are exact; the refit loop estimates bias and variance. */

import type { RegressionPoint } from '../datasets/types';
import { features, makeNormaliser, predict, solve } from './linearRegression';
import type { Normaliser } from './linearRegression';

export interface PolyConfig {
  /** Highest power of x. 0 is the mean, 1 a straight line. */
  degree: number;
  /** Ridge penalty on the non-bias weights. */
  l2: number;
  /** Standardise x before raising it to powers. */
  standardise: boolean;
}

export interface PolyFit {
  degree: number;
  /** [b, w1 … w_degree] in the fitting space. */
  weights: number[];
  norm: Normaliser;
  /** The normal equations were singular or overflowed: no usable curve. */
  failed: boolean;
}

/** (ΦᵀΦ + λnI) w = Φᵀy, with the bias left unpenalised. Null when the system is singular. */
export function normalEquations(
  points: readonly RegressionPoint[],
  degree: number,
  norm: Normaliser,
  l2: number,
): number[] | null {
  const size = degree + 1;
  if (points.length === 0) return null;
  // Without a penalty, fewer distinct x's than weights leaves many exact fits and no single answer.
  if (l2 <= 0 && new Set(points.map((p) => p.x)).size < size) return null;
  const xtx = Array.from({ length: size }, () => new Array<number>(size).fill(0));
  const xty = new Array<number>(size).fill(0);
  for (const p of points) {
    const f = features(p.x, degree, norm);
    for (let i = 0; i < size; i++) {
      xty[i] += f[i] * p.y;
      for (let j = 0; j < size; j++) xtx[i][j] += f[i] * f[j];
    }
  }
  if (l2 > 0) for (let i = 1; i < size; i++) xtx[i][i] += l2 * points.length;
  return solve(xtx, xty);
}

/** Exact least-squares fit through the normal equations. */
export function fitPolynomial(points: readonly RegressionPoint[], config: PolyConfig): PolyFit {
  const degree = Math.max(0, Math.round(config.degree));
  const norm = makeNormaliser(points, config.standardise);
  const weights = normalEquations(points, degree, norm, config.l2);
  const failed = weights === null || weights.some((w) => !Number.isFinite(w));
  return {
    degree,
    weights: failed ? new Array<number>(degree + 1).fill(0) : weights!,
    norm,
    failed,
  };
}

export function predictFit(fit: PolyFit, x: number): number {
  return predict(fit.weights, x, fit.degree, fit.norm);
}

/** The feature row [1, x̃, x̃², …] a point becomes. */
export function featureRow(fit: PolyFit, x: number): number[] {
  return features(x, fit.degree, fit.norm);
}

export function meanSquaredError(fit: PolyFit, points: readonly RegressionPoint[]): number {
  if (points.length === 0) return Number.NaN;
  let sum = 0;
  for (const p of points) {
    const err = predictFit(fit, p.x) - p.y;
    sum += err * err;
  }
  return sum / points.length;
}

/** Evenly spaced x's for curves and the bias-variance grid. */
export function gridOver(range: readonly [number, number], count = 61): number[] {
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) out[i] = range[0] + ((range[1] - range[0]) * i) / (count - 1);
  return out;
}

export function curveOn(fit: PolyFit, xs: readonly number[]): number[] {
  return xs.map((x) => predictFit(fit, x));
}

/* ---------------- the validation curve ---------------- */

export interface ValidationCurve {
  degrees: number[];
  train: number[];
  test: number[];
  /** Degree with the lowest test error. */
  best: number;
}

/** Train and test error at every degree up to `maxDegree`, the U-shaped picture. */
export function validationCurve(
  train: readonly RegressionPoint[],
  test: readonly RegressionPoint[],
  maxDegree: number,
  config: Omit<PolyConfig, 'degree'>,
): ValidationCurve {
  const degrees: number[] = [];
  const trainErr: number[] = [];
  const testErr: number[] = [];
  let best = 0;
  for (let d = 0; d <= maxDegree; d++) {
    const fit = fitPolynomial(train, { ...config, degree: d });
    degrees.push(d);
    trainErr.push(fit.failed ? Number.NaN : meanSquaredError(fit, train));
    testErr.push(fit.failed ? Number.NaN : meanSquaredError(fit, test));
    if (Number.isFinite(testErr[d]) && (!Number.isFinite(testErr[best]) || testErr[d] < testErr[best])) best = d;
  }
  return { degrees, train: trainErr, test: testErr, best };
}

/* ---------------- refits: the bias-variance loop ---------------- */

export interface RefitState {
  /** Refits so far. */
  epoch: number;
  /** Each refit's curve on the grid. */
  curves: number[][];
  /** Running mean of the curves on the grid. */
  mean: number[];
  /** Running sum of squared deviations, for the variance. */
  m2: number[];
  /** The sample the last refit trained on. */
  lastSample: RegressionPoint[];
  lastFit: PolyFit | null;
  /** Refits whose normal equations failed. */
  failures: number;
}

export function createRefitState(gridSize: number): RefitState {
  return {
    epoch: 0,
    curves: [],
    mean: new Array<number>(gridSize).fill(0),
    m2: new Array<number>(gridSize).fill(0),
    lastSample: [],
    lastFit: null,
    failures: 0,
  };
}

/** One refit: a fresh sample, an exact fit, its curve folded into the running mean and spread. Pure. */
export function refitStep(
  state: RefitState,
  sample: (k: number) => RegressionPoint[],
  config: PolyConfig,
  grid: readonly number[],
): RefitState {
  const points = sample(state.epoch);
  const fit = fitPolynomial(points, config);
  if (fit.failed) {
    return { ...state, epoch: state.epoch + 1, lastSample: points, lastFit: fit, failures: state.failures + 1 };
  }
  const curve = curveOn(fit, grid);
  const count = state.curves.length + 1;
  const mean = state.mean.slice();
  const m2 = state.m2.slice();
  for (let i = 0; i < grid.length; i++) {
    const delta = curve[i] - mean[i];
    mean[i] += delta / count;
    m2[i] += delta * (curve[i] - mean[i]);
  }
  return {
    epoch: state.epoch + 1,
    curves: state.curves.concat([curve]),
    mean,
    m2,
    lastSample: points,
    lastFit: fit,
    failures: state.failures,
  };
}

export interface BiasVariance {
  /** Mean over the grid of (mean refit − truth)². */
  bias2: number;
  /** Mean over the grid of the refits' variance. */
  variance: number;
}

/** Bias² and variance of the refits so far against the truth on the same grid. */
export function biasVariance(state: RefitState, truth: readonly number[]): BiasVariance {
  const count = state.curves.length;
  if (count < 2) return { bias2: Number.NaN, variance: Number.NaN };
  let bias2 = 0;
  let variance = 0;
  for (let i = 0; i < truth.length; i++) {
    bias2 += (state.mean[i] - truth[i]) ** 2;
    variance += state.m2[i] / (count - 1);
  }
  return { bias2: bias2 / truth.length, variance: variance / truth.length };
}

export interface BiasVarianceSweep {
  degrees: number[];
  bias2: number[];
  variance: number[];
  /** σ² of the noise, the floor no model gets under. */
  noise: number;
}

/** The decomposition at every degree, from the same refits the loop would take. */
export function biasVarianceSweep(
  sample: (k: number) => RegressionPoint[],
  truth: readonly number[],
  grid: readonly number[],
  maxDegree: number,
  refits: number,
  config: Omit<PolyConfig, 'degree'>,
  noiseSigma: number,
): BiasVarianceSweep {
  const samples = Array.from({ length: refits }, (_, k) => sample(k));
  const degrees: number[] = [];
  const bias2: number[] = [];
  const variance: number[] = [];
  for (let d = 0; d <= maxDegree; d++) {
    let state = createRefitState(grid.length);
    const full = { ...config, degree: d };
    for (const points of samples) state = refitStep(state, () => points, full, grid);
    const bv = biasVariance(state, truth);
    degrees.push(d);
    bias2.push(bv.bias2);
    variance.push(bv.variance);
  }
  return { degrees, bias2, variance, noise: noiseSigma * noiseSigma };
}

/** Largest |x̃ᵏ| over the points, the feature the solver has to balance against 1. */
export function largestFeature(fit: PolyFit, points: readonly RegressionPoint[]): number {
  let largest = 0;
  for (const p of points) {
    const row = features(p.x, fit.degree, fit.norm);
    for (const v of row) largest = Math.max(largest, Math.abs(v));
  }
  return largest;
}
