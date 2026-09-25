/** Linear regression by gradient descent. Every step is pure and records what the page draws. */

import type { RegressionPoint } from '../datasets/types';

export type OptimiserName = 'gd' | 'momentum' | 'adam';
export type BatchMode = 'batch' | 'mini' | 'stochastic';

export interface LinRegConfig {
  learningRate: number;
  /** Polynomial degree: 1 is a straight line. */
  degree: number;
  optimiser: OptimiserName;
  /** Momentum coefficient, used by 'momentum' and as beta1 for Adam. */
  momentum: number;
  batchMode: BatchMode;
  batchSize: number;
  /** L2 penalty on the non-bias weights. */
  l2: number;
  /** Standardise x before fitting, the difference between diverging and not. */
  standardise: boolean;
  seed: number;
}

export interface LinRegState {
  /** [bias, w1, … w_degree] in the *fitting* space (standardised if enabled). */
  weights: number[];
  /** Optimiser accumulators. */
  velocity: number[];
  adamM: number[];
  adamV: number[];
  epoch: number;
  /** Loss measured on the full dataset after the most recent update. */
  loss: number;
  /** Full-dataset loss after each step, the last 4000 kept; the first kept is step epoch - length + 1. */
  lossHistory: number[];
  /** Weight trajectory, for the loss-surface path. Trimmed to the last 400. */
  weightHistory: number[][];
  /** The gradient used by the most recent step. */
  lastGradient: number[];
  /** The change the most recent step made to each weight, whatever the optimiser. */
  lastStep: number[];
  /** Indices of the points in the most recent batch. */
  lastBatch: number[];
  /** Gradient norm, the practical convergence signal. */
  gradientNorm: number;
  converged: boolean;
  /** Diverged to NaN/Infinity, a learning rate that was too large. */
  diverged: boolean;
}

/** Standardisation applied to x before fitting, so weights can be mapped back. */
export interface Normaliser {
  mean: number;
  std: number;
}

export function makeNormaliser(points: readonly RegressionPoint[], enabled: boolean): Normaliser {
  if (!enabled || points.length === 0) return { mean: 0, std: 1 };
  let sum = 0;
  for (const p of points) sum += p.x;
  const mean = sum / points.length;
  let sq = 0;
  for (const p of points) sq += (p.x - mean) * (p.x - mean);
  const std = Math.sqrt(sq / points.length);
  return { mean, std: std < 1e-9 ? 1 : std };
}

/** Feature vector [1, x, x², …] in the fitting space. */
export function features(x: number, degree: number, norm: Normaliser): number[] {
  const z = (x - norm.mean) / norm.std;
  const out = new Array<number>(degree + 1);
  let power = 1;
  for (let i = 0; i <= degree; i++) {
    out[i] = power;
    power *= z;
  }
  return out;
}

export function predict(
  weights: readonly number[],
  x: number,
  degree: number,
  norm: Normaliser,
): number {
  const f = features(x, degree, norm);
  let sum = 0;
  for (let i = 0; i < f.length; i++) sum += weights[i] * f[i];
  return sum;
}

export function createState(config: LinRegConfig): LinRegState {
  const size = config.degree + 1;
  const zeros = () => new Array<number>(size).fill(0);
  return {
    weights: zeros(),
    velocity: zeros(),
    adamM: zeros(),
    adamV: zeros(),
    epoch: 0,
    loss: 0,
    lossHistory: [],
    weightHistory: [],
    lastGradient: zeros(),
    lastStep: zeros(),
    lastBatch: [],
    gradientNorm: 0,
    converged: false,
    diverged: false,
  };
}

/** Mean squared error over `points`, plus the L2 penalty when one is set. */
export function computeLoss(
  weights: readonly number[],
  points: readonly RegressionPoint[],
  config: LinRegConfig,
  norm: Normaliser,
): number {
  if (points.length === 0) return 0;
  let sum = 0;
  for (const p of points) {
    const err = predict(weights, p.x, config.degree, norm) - p.y;
    sum += err * err;
  }
  let loss = sum / points.length;
  if (config.l2 > 0) {
    for (let i = 1; i < weights.length; i++) loss += config.l2 * weights[i] * weights[i];
  }
  return loss;
}

/** Gradient of the MSE (+ L2) with respect to the weights, over `batch`. */
export function computeGradient(
  weights: readonly number[],
  points: readonly RegressionPoint[],
  batch: readonly number[],
  config: LinRegConfig,
  norm: Normaliser,
): number[] {
  const grad = new Array<number>(weights.length).fill(0);
  if (batch.length === 0) return grad;
  for (const index of batch) {
    const p = points[index];
    const f = features(p.x, config.degree, norm);
    let yhat = 0;
    for (let i = 0; i < f.length; i++) yhat += weights[i] * f[i];
    const err = yhat - p.y;
    const scale = (2 * err) / batch.length;
    for (let i = 0; i < f.length; i++) grad[i] += scale * f[i];
  }
  if (config.l2 > 0) {
    for (let i = 1; i < grad.length; i++) grad[i] += 2 * config.l2 * weights[i];
  }
  return grad;
}

/** Which points this epoch's update sees. */
function selectBatch(config: LinRegConfig, n: number, epoch: number): number[] {
  if (config.batchMode === 'batch' || n === 0) {
    return Array.from({ length: n }, (_, i) => i);
  }
  const size = config.batchMode === 'stochastic' ? 1 : Math.max(1, Math.min(n, config.batchSize));
  // Deterministic walk; shuffling per epoch would only make the loss curve jitter.
  const start = (epoch * size) % n;
  const out: number[] = [];
  for (let i = 0; i < size; i++) out.push((start + i) % n);
  return out;
}

const ADAM_BETA2 = 0.999;
const ADAM_EPS = 1e-8;
const CONVERGE_GRAD = 1e-5;

/** One optimisation step. Pure: returns new state, never mutates the input. */
export function step(
  state: LinRegState,
  points: readonly RegressionPoint[],
  config: LinRegConfig,
  norm: Normaliser,
): LinRegState {
  if (points.length === 0 || state.diverged) return state;

  const batch = selectBatch(config, points.length, state.epoch);
  const gradient = computeGradient(state.weights, points, batch, config, norm);

  const weights = state.weights.slice();
  const velocity = state.velocity.slice();
  const adamM = state.adamM.slice();
  const adamV = state.adamV.slice();
  const t = state.epoch + 1;

  for (let i = 0; i < weights.length; i++) {
    const g = gradient[i];
    if (config.optimiser === 'momentum') {
      velocity[i] = config.momentum * velocity[i] - config.learningRate * g;
      weights[i] += velocity[i];
    } else if (config.optimiser === 'adam') {
      const beta1 = config.momentum;
      adamM[i] = beta1 * adamM[i] + (1 - beta1) * g;
      adamV[i] = ADAM_BETA2 * adamV[i] + (1 - ADAM_BETA2) * g * g;
      const mHat = adamM[i] / (1 - Math.pow(beta1, t));
      const vHat = adamV[i] / (1 - Math.pow(ADAM_BETA2, t));
      weights[i] -= (config.learningRate * mHat) / (Math.sqrt(vHat) + ADAM_EPS);
    } else {
      weights[i] -= config.learningRate * g;
    }
  }

  const diverged = weights.some((w) => !Number.isFinite(w) || Math.abs(w) > 1e12);
  const loss = diverged ? Number.POSITIVE_INFINITY : computeLoss(weights, points, config, norm);

  let normSquared = 0;
  for (const g of gradient) normSquared += g * g;
  const gradientNorm = Math.sqrt(normSquared);

  const lossHistory = state.lossHistory.concat(diverged ? Number.NaN : loss);
  const weightHistory = state.weightHistory.concat([weights.slice()]);

  return {
    weights,
    velocity,
    adamM,
    adamV,
    epoch: t,
    loss,
    lossHistory: lossHistory.length > 4000 ? lossHistory.slice(-4000) : lossHistory,
    weightHistory: weightHistory.length > 400 ? weightHistory.slice(-400) : weightHistory,
    lastGradient: gradient,
    lastStep: weights.map((w, i) => w - state.weights[i]),
    lastBatch: batch,
    gradientNorm,
    // Only a full-batch run can honestly claim convergence from one gradient.
    converged: !diverged && config.batchMode === 'batch' && gradientNorm < CONVERGE_GRAD,
    diverged,
  };
}

/** Exact least-squares solution via the normal equations. */
export function closedFormSolution(
  points: readonly RegressionPoint[],
  degree: number,
  norm: Normaliser,
  l2 = 0,
): number[] | null {
  const size = degree + 1;
  if (points.length < size) return null;

  // Build XᵀX and Xᵀy.
  const xtx = Array.from({ length: size }, () => new Array<number>(size).fill(0));
  const xty = new Array<number>(size).fill(0);
  for (const p of points) {
    const f = features(p.x, degree, norm);
    for (let i = 0; i < size; i++) {
      xty[i] += f[i] * p.y;
      for (let j = 0; j < size; j++) xtx[i][j] += f[i] * f[j];
    }
  }
  // Ridge term keeps the system solvable for high degrees.
  const ridge = l2 > 0 ? l2 * points.length : 1e-9;
  for (let i = 1; i < size; i++) xtx[i][i] += ridge;

  return solve(xtx, xty);
}

/** Gauss-Jordan with partial pivoting. Returns null for a singular system. */
export function solve(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat(rhs[i]));

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];

    const scale = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= scale;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      if (factor === 0) continue;
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map((row) => row[n]);
}

/** Weights in raw-x units. Only meaningful for degree 1. */
export function toOriginalUnits(
  weights: readonly number[],
  norm: Normaliser,
): { intercept: number; slope: number } {
  const slope = weights.length > 1 ? weights[1] / norm.std : 0;
  const intercept = weights[0] - (weights.length > 1 ? (weights[1] * norm.mean) / norm.std : 0);
  return { intercept, slope };
}
