/** Logistic regression on two features: a linear score through a sigmoid. */

import type { Point2D } from '../datasets/types';
import { makeRng } from '../math/rng';

export type FeatureMap = 'linear' | 'quadratic' | 'interaction';

export interface LogRegConfig {
  learningRate: number;
  /** Extra features let a "linear" model draw a curve. */
  featureMap: FeatureMap;
  /** L2 penalty on the non-bias weights. */
  l2: number;
  batchSize: number;
  /** Probability above which a point is called class 1. */
  threshold: number;
  standardise: boolean;
  seed: number;
}

export interface LogRegState {
  weights: number[];
  epoch: number;
  loss: number;
  lossHistory: number[];
  accuracyHistory: number[];
  lastGradient: number[];
  gradientNorm: number;
  converged: boolean;
  diverged: boolean;
}

export interface Standardiser {
  meanX: number;
  meanY: number;
  stdX: number;
  stdY: number;
}

export function makeStandardiser(points: readonly Point2D[], enabled: boolean): Standardiser {
  if (!enabled || points.length === 0) return { meanX: 0, meanY: 0, stdX: 1, stdY: 1 };
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  const meanX = sx / points.length;
  const meanY = sy / points.length;
  let vx = 0;
  let vy = 0;
  for (const p of points) {
    vx += (p.x - meanX) ** 2;
    vy += (p.y - meanY) ** 2;
  }
  const stdX = Math.sqrt(vx / points.length);
  const stdY = Math.sqrt(vy / points.length);
  return {
    meanX,
    meanY,
    stdX: stdX < 1e-9 ? 1 : stdX,
    stdY: stdY < 1e-9 ? 1 : stdY,
  };
}

/** Feature names in the order `features()` produces them, used by the weight bars. */
export function featureNames(map: FeatureMap): string[] {
  switch (map) {
    case 'quadratic':
      return ['1', 'x₁', 'x₂', 'x₁²', 'x₂²', 'x₁x₂'];
    case 'interaction':
      return ['1', 'x₁', 'x₂', 'x₁x₂'];
    default:
      return ['1', 'x₁', 'x₂'];
  }
}

export function featureCount(map: FeatureMap): number {
  return featureNames(map).length;
}

export function features(x: number, y: number, map: FeatureMap, std: Standardiser): number[] {
  const a = (x - std.meanX) / std.stdX;
  const b = (y - std.meanY) / std.stdY;
  switch (map) {
    case 'quadratic':
      return [1, a, b, a * a, b * b, a * b];
    case 'interaction':
      return [1, a, b, a * b];
    default:
      return [1, a, b];
  }
}

/** Numerically stable logistic function. */
export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** The raw linear score before squashing, the signed distance-like quantity. */
export function score(
  weights: readonly number[],
  x: number,
  y: number,
  map: FeatureMap,
  std: Standardiser,
): number {
  const f = features(x, y, map, std);
  let sum = 0;
  for (let i = 0; i < f.length; i++) sum += weights[i] * f[i];
  return sum;
}

export function probability(
  weights: readonly number[],
  x: number,
  y: number,
  map: FeatureMap,
  std: Standardiser,
): number {
  return sigmoid(score(weights, x, y, map, std));
}

export function createState(config: LogRegConfig): LogRegState {
  const size = featureCount(config.featureMap);
  const rng = makeRng(config.seed);
  // Small random weights so the first boundary is visible rather than 50/50 everywhere.
  const weights = Array.from({ length: size }, (_, i) => (i === 0 ? 0 : (rng() - 0.5) * 0.4));
  return {
    weights,
    epoch: 0,
    loss: 0,
    lossHistory: [],
    accuracyHistory: [],
    lastGradient: new Array<number>(size).fill(0),
    gradientNorm: 0,
    converged: false,
    diverged: false,
  };
}

/** Average binary cross-entropy, plus the L2 penalty. */
export function computeLoss(
  weights: readonly number[],
  points: readonly Point2D[],
  config: LogRegConfig,
  std: Standardiser,
): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const p of points) {
    // −log σ(z) for y = 1 and −log(1 − σ(z)) for y = 0, as softplus(z) − y·z, exact at any z.
    const z = score(weights, p.x, p.y, config.featureMap, std);
    const softplus = Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
    sum += softplus - (p.label === 1 ? z : 0);
  }
  let loss = sum / points.length;
  if (config.l2 > 0) {
    for (let i = 1; i < weights.length; i++) loss += config.l2 * weights[i] * weights[i];
  }
  return loss;
}

/** Gradient of the cross-entropy: (prediction - target) x feature, the same shape as squared error. */
export function computeGradient(
  weights: readonly number[],
  points: readonly Point2D[],
  batch: readonly number[],
  config: LogRegConfig,
  std: Standardiser,
): number[] {
  const grad = new Array<number>(weights.length).fill(0);
  if (batch.length === 0) return grad;
  for (const index of batch) {
    const p = points[index];
    const f = features(p.x, p.y, config.featureMap, std);
    let z = 0;
    for (let i = 0; i < f.length; i++) z += weights[i] * f[i];
    const err = sigmoid(z) - p.label;
    for (let i = 0; i < f.length; i++) grad[i] += (err * f[i]) / batch.length;
  }
  if (config.l2 > 0) {
    for (let i = 1; i < grad.length; i++) grad[i] += 2 * config.l2 * weights[i];
  }
  return grad;
}

export function selectBatch(config: LogRegConfig, n: number, epoch: number): number[] {
  if (n === 0) return [];
  const size = Math.max(1, Math.min(n, config.batchSize));
  if (size >= n) return Array.from({ length: n }, (_, i) => i);
  const start = (epoch * size) % n;
  return Array.from({ length: size }, (_, i) => (start + i) % n);
}

export function step(
  state: LogRegState,
  points: readonly Point2D[],
  config: LogRegConfig,
  std: Standardiser,
): LogRegState {
  if (points.length === 0 || state.diverged) return state;

  const batch = selectBatch(config, points.length, state.epoch);
  const gradient = computeGradient(state.weights, points, batch, config, std);
  const weights = state.weights.map((w, i) => w - config.learningRate * gradient[i]);

  const diverged = weights.some((w) => !Number.isFinite(w) || Math.abs(w) > 1e10);
  const loss = diverged ? Number.POSITIVE_INFINITY : computeLoss(weights, points, config, std);

  let normSquared = 0;
  for (const g of gradient) normSquared += g * g;
  const gradientNorm = Math.sqrt(normSquared);

  const acc = diverged ? 0 : accuracyOf(weights, points, config, std);
  const lossHistory = state.lossHistory.concat(diverged ? Number.NaN : loss);
  const accuracyHistory = state.accuracyHistory.concat(acc);

  return {
    weights,
    epoch: state.epoch + 1,
    loss,
    lossHistory: lossHistory.length > 4000 ? lossHistory.slice(-4000) : lossHistory,
    accuracyHistory:
      accuracyHistory.length > 4000 ? accuracyHistory.slice(-4000) : accuracyHistory,
    lastGradient: gradient,
    gradientNorm,
    // Only a full-batch gradient can claim convergence.
    converged: !diverged && batch.length === points.length && gradientNorm < 1e-5,
    diverged,
  };
}

export function accuracyOf(
  weights: readonly number[],
  points: readonly Point2D[],
  config: LogRegConfig,
  std: Standardiser,
): number {
  if (points.length === 0) return 0;
  let hits = 0;
  for (const p of points) {
    const prob = probability(weights, p.x, p.y, config.featureMap, std);
    const predicted = prob >= config.threshold ? 1 : 0;
    if (predicted === p.label) hits += 1;
  }
  return hits / points.length;
}

export function predictAll(
  weights: readonly number[],
  points: readonly Point2D[],
  config: LogRegConfig,
  std: Standardiser,
): { labels: number[]; probabilities: number[] } {
  const labels: number[] = [];
  const probabilities: number[] = [];
  for (const p of points) {
    const prob = probability(weights, p.x, p.y, config.featureMap, std);
    probabilities.push(prob);
    labels.push(prob >= config.threshold ? 1 : 0);
  }
  return { labels, probabilities };
}
