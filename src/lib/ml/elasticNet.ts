/** Ridge, lasso and elastic net on a prepared design: coordinate descent, proximal gradient, the closed form, the path. */

import type { Design } from '../datasets/tabular';
import { solve } from './linearRegression';

/** `closed` is the ridge closed form, one step; the page maps it to `cd` when there is an L1 term. */
export type Solver = 'cd' | 'ista' | 'closed';

export interface ElasticNetConfig {
  lambda: number;
  /** Mix: 1 is lasso, 0 is ridge. */
  alpha: number;
  solver: Solver;
  /** Step size of the proximal gradient solver. */
  learningRate: number;
}

/** The centred (and optionally scaled) training design with its Gram form. */
export interface Prepared {
  Xs: number[][];
  yc: number[];
  yMean: number;
  means: number[];
  stds: number[];
  /** XsᵀXs over n. */
  gram: number[][];
  /** Xsᵀyc over n. */
  xty: number[];
  /** ycᵀyc over n. */
  yty: number;
  /** Largest eigenvalue of the Gram matrix, the curvature the step size must respect. */
  lipschitz: number;
  n: number;
  p: number;
}

export interface ElasticNetState {
  /** Weights in the prepared space. */
  w: number[];
  epoch: number;
  objective: number;
  objectiveHistory: number[];
  wHistory: number[][];
  activeCount: number;
  maxChange: number;
  converged: boolean;
  diverged: boolean;
}

const CONVERGE_CHANGE = 1e-6;

export function prepare(train: Design, standardise: boolean): Prepared {
  const n = train.X.length;
  const p = train.X[0]?.length ?? 0;
  const means = new Array<number>(p).fill(0);
  const stds = new Array<number>(p).fill(1);
  for (let j = 0; j < p; j++) {
    let sum = 0;
    for (const row of train.X) sum += row[j];
    means[j] = n ? sum / n : 0;
    if (standardise) {
      let sq = 0;
      for (const row of train.X) sq += (row[j] - means[j]) ** 2;
      const sd = Math.sqrt(n ? sq / n : 0);
      stds[j] = sd < 1e-9 ? 1 : sd;
    }
  }
  const yMean = n ? train.y.reduce((a, b) => a + b, 0) / n : 0;
  const Xs = train.X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
  const yc = train.y.map((v) => v - yMean);

  const gram = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const xty = new Array<number>(p).fill(0);
  let yty = 0;
  for (let i = 0; i < n; i++) {
    const row = Xs[i];
    yty += yc[i] * yc[i];
    for (let j = 0; j < p; j++) {
      xty[j] += row[j] * yc[i];
      for (let k = j; k < p; k++) gram[j][k] += row[j] * row[k];
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = j; k < p; k++) {
      gram[j][k] /= Math.max(1, n);
      gram[k][j] = gram[j][k];
    }
    xty[j] /= Math.max(1, n);
  }
  yty /= Math.max(1, n);
  return { Xs, yc, yMean, means, stds, gram, xty, yty, lipschitz: largestEigenvalue(gram), n, p };
}

/** Power iteration; enough for a step size bound. */
function largestEigenvalue(matrix: number[][]): number {
  const p = matrix.length;
  if (p === 0) return 1;
  let v = new Array<number>(p).fill(1 / Math.sqrt(p));
  let value = 1;
  for (let iter = 0; iter < 60; iter++) {
    const next = matrix.map((row) => row.reduce((s, m, j) => s + m * v[j], 0));
    const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0));
    if (norm < 1e-12) return 1e-12;
    value = norm;
    v = next.map((x) => x / norm);
  }
  return value;
}

export function softThreshold(z: number, t: number): number {
  if (z > t) return z - t;
  if (z < -t) return z + t;
  return 0;
}

export function penalty(w: readonly number[], lambda: number, alpha: number): number {
  let l1 = 0;
  let l2 = 0;
  for (const v of w) {
    l1 += Math.abs(v);
    l2 += v * v;
  }
  return lambda * (alpha * l1 + ((1 - alpha) / 2) * l2);
}

/** Half the mean squared residual in the prepared space, from the Gram form. */
export function smoothLoss(w: readonly number[], prep: Prepared): number {
  let quad = 0;
  let lin = 0;
  for (let j = 0; j < prep.p; j++) {
    lin += w[j] * prep.xty[j];
    let row = 0;
    for (let k = 0; k < prep.p; k++) row += prep.gram[j][k] * w[k];
    quad += w[j] * row;
  }
  return 0.5 * (prep.yty - 2 * lin + quad);
}

export function objective(w: readonly number[], prep: Prepared, config: ElasticNetConfig): number {
  return smoothLoss(w, prep) + penalty(w, config.lambda, config.alpha);
}

/** Gradient of the smooth part, the squared error plus the ridge term. */
export function smoothGradient(w: readonly number[], prep: Prepared, config: ElasticNetConfig): number[] {
  const ridge = config.lambda * (1 - config.alpha);
  return prep.gram.map((row, j) => row.reduce((s, m, k) => s + m * w[k], 0) - prep.xty[j] + ridge * w[j]);
}

/** One pass over the coordinates, each solved exactly with the others held fixed. */
export function coordinateSweep(
  w: readonly number[],
  prep: Prepared,
  config: ElasticNetConfig,
): { w: number[]; maxChange: number } {
  const out = w.slice();
  const l1 = config.lambda * config.alpha;
  const ridge = config.lambda * (1 - config.alpha);
  let maxChange = 0;
  for (let j = 0; j < prep.p; j++) {
    let rho = prep.xty[j];
    for (let k = 0; k < prep.p; k++) if (k !== j) rho -= prep.gram[j][k] * out[k];
    const next = softThreshold(rho, l1) / (prep.gram[j][j] + ridge);
    maxChange = Math.max(maxChange, Math.abs(next - out[j]));
    out[j] = next;
  }
  return { w: out, maxChange };
}

/** One proximal gradient step: a gradient step on the smooth part, then the soft threshold. */
export function istaEpoch(w: readonly number[], prep: Prepared, config: ElasticNetConfig): number[] {
  const grad = smoothGradient(w, prep, config);
  const eta = config.learningRate;
  return w.map((v, j) => softThreshold(v - eta * grad[j], eta * config.lambda * config.alpha));
}

export function ridgeClosedForm(prep: Prepared, lambda: number): number[] {
  const matrix = prep.gram.map((row, j) => row.map((m, k) => (j === k ? m + lambda : m)));
  return solve(matrix, prep.xty.slice()) ?? new Array<number>(prep.p).fill(0);
}

/** Least squares, with a whisper of ridge so a wide design still solves. */
export function olsSolution(prep: Prepared): number[] {
  return ridgeClosedForm(prep, 1e-9);
}

export function createState(prep: Prepared, config: ElasticNetConfig): ElasticNetState {
  const w = new Array<number>(prep.p).fill(0);
  const value = objective(w, prep, config);
  return {
    w,
    epoch: 0,
    objective: value,
    objectiveHistory: [value],
    wHistory: [w.slice()],
    activeCount: 0,
    maxChange: Infinity,
    converged: false,
    diverged: false,
  };
}

/** One sweep or one proximal epoch. Pure. */
export function step(state: ElasticNetState, prep: Prepared, config: ElasticNetConfig): ElasticNetState {
  if (state.converged || state.diverged || prep.p === 0) return state;
  let w: number[];
  let maxChange: number;
  if (config.solver === 'closed') {
    w = ridgeClosedForm(prep, config.lambda * (1 - config.alpha));
    maxChange = 0;
  } else if (config.solver === 'ista') {
    w = istaEpoch(state.w, prep, config);
    maxChange = w.reduce((m, v, j) => Math.max(m, Math.abs(v - state.w[j])), 0);
  } else {
    const sweep = coordinateSweep(state.w, prep, config);
    w = sweep.w;
    maxChange = sweep.maxChange;
  }
  const diverged = w.some((v) => !Number.isFinite(v) || Math.abs(v) > 1e8);
  const value = diverged ? Number.POSITIVE_INFINITY : objective(w, prep, config);
  const history = state.objectiveHistory.concat(diverged ? Number.NaN : value);
  const wHistory = state.wHistory.concat([w.slice()]);
  return {
    w,
    epoch: state.epoch + 1,
    objective: value,
    objectiveHistory: history.length > 4000 ? history.slice(-4000) : history,
    wHistory: wHistory.length > 400 ? wHistory.slice(-400) : wHistory,
    activeCount: w.filter((v) => v !== 0).length,
    maxChange,
    converged: !diverged && maxChange < CONVERGE_CHANGE,
    diverged,
  };
}

/** Weights and bias in the units of the raw features. */
export function toRawUnits(w: readonly number[], prep: Prepared): { weights: number[]; bias: number } {
  const weights = w.map((v, j) => v / prep.stds[j]);
  let bias = prep.yMean;
  for (let j = 0; j < prep.p; j++) bias -= weights[j] * prep.means[j];
  return { weights, bias };
}

export function predictRaw(w: readonly number[], prep: Prepared, row: readonly number[]): number {
  let sum = prep.yMean;
  for (let j = 0; j < prep.p; j++) sum += (w[j] * (row[j] - prep.means[j])) / prep.stds[j];
  return sum;
}

export function meanSquaredErrorOf(w: readonly number[], prep: Prepared, design: Design): number {
  if (design.X.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < design.X.length; i++) {
    const d = predictRaw(w, prep, design.X[i]) - design.y[i];
    sum += d * d;
  }
  return sum / design.X.length;
}

/** The smallest lambda at which every coefficient is zero (exact for lasso). */
export function lambdaMax(prep: Prepared, alpha: number): number {
  let top = 0;
  for (const v of prep.xty) top = Math.max(top, Math.abs(v));
  return top / Math.max(alpha, 0.1);
}

/** Log-spaced lambdas, descending from `top`. */
export function lambdaGrid(top: number, count = 60, ratio = 1e-3): number[] {
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push(top * Math.pow(ratio, i / (count - 1)));
  return out;
}

export interface PathResult {
  lambdas: number[];
  /** One weight vector per lambda, prepared space. */
  weights: number[][];
  trainMse: number[];
  testMse: number[];
  activeCounts: number[];
  /** Index of the lambda with the lowest test error. */
  bestIndex: number;
}

/** The whole regularisation path, warm-started down the grid. */
export function regularisationPath(
  train: Design,
  test: Design,
  prep: Prepared,
  alpha: number,
  lambdas: readonly number[],
  options: { maxSweeps?: number; tol?: number } = {},
): PathResult {
  const maxSweeps = options.maxSweeps ?? 200;
  const tol = options.tol ?? 1e-7;
  let w = new Array<number>(prep.p).fill(0);
  const weights: number[][] = [];
  const trainMse: number[] = [];
  const testMse: number[] = [];
  const activeCounts: number[] = [];
  for (const lambda of lambdas) {
    if (alpha === 0) {
      w = ridgeClosedForm(prep, lambda);
    } else {
      const config: ElasticNetConfig = { lambda, alpha, solver: 'cd', learningRate: 0 };
      for (let sweep = 0; sweep < maxSweeps; sweep++) {
        const next = coordinateSweep(w, prep, config);
        w = next.w;
        if (next.maxChange < tol) break;
      }
    }
    weights.push(w.slice());
    trainMse.push(meanSquaredErrorOf(w, prep, train));
    testMse.push(meanSquaredErrorOf(w, prep, test));
    activeCounts.push(w.filter((v) => Math.abs(v) > 1e-12).length);
  }
  let bestIndex = 0;
  for (let i = 1; i < testMse.length; i++) if (testMse[i] < testMse[bestIndex]) bestIndex = i;
  return { lambdas: lambdas.slice(), weights, trainMse, testMse, activeCounts, bestIndex };
}

/** Where a coefficient enters the path coming down from lambda max, or null if it never does. */
export function entryLambda(path: PathResult, feature: number): number | null {
  for (let i = 0; i < path.lambdas.length; i++) {
    if (Math.abs(path.weights[i][feature]) > 1e-12) return path.lambdas[i];
  }
  return null;
}

/* ---------------- the two-weight picture ---------------- */

/** The smooth loss over two chosen weights with the others held where they are. */
export function lossSlice(prep: Prepared, w: readonly number[], i: number, j: number): (a: number, b: number) => number {
  const base = w.slice();
  return (a, b) => {
    base[i] = a;
    base[j] = b;
    return smoothLoss(base, prep);
  };
}

export function penaltyAt(alpha: number): (a: number, b: number) => number {
  return (a, b) => alpha * (Math.abs(a) + Math.abs(b)) + ((1 - alpha) / 2) * (a * a + b * b);
}
