/** Synthetic multi-feature regression: a sparse true weight vector, optional correlated groups, a train/test split. */

import { gauss, makeRng, shuffled, uniform } from '../math/rng';
import { sub } from '../math/stats';

export interface TabularConfig {
  count: number;
  features: number;
  /** How many features carry a non-zero true weight. */
  informative: number;
  /** Correlation inside each group of consecutive features. */
  correlation: number;
  groupSize: number;
  /** Noise as a fraction of the signal's spread. */
  noise: number;
  /** Give every feature its own scale and offset, so standardising matters. */
  variedScales: boolean;
  seed: number;
  trainFraction: number;
}

export interface TabularData {
  kind: 'tabular';
  X: number[][];
  y: number[];
  featureNames: string[];
  /** In standardised units: the weight on (x minus mean) over std. */
  trueWeights: number[];
  trueBias: number;
  groups: number[][];
  scales: number[];
  trainIndex: number[];
  testIndex: number[];
}

export interface Design {
  X: number[][];
  y: number[];
}

export function featureName(index: number): string {
  return 'x' + sub(index + 1);
}

export const DEFAULT_TABULAR: TabularConfig = {
  count: 80,
  features: 10,
  informative: 3,
  correlation: 0,
  groupSize: 3,
  noise: 0.3,
  variedScales: false,
  seed: 11,
  trainFraction: 0.75,
};

/** Feature order for handing out true weights: first of every group, then second, so groups share signal. */
function informativeOrder(features: number, groupSize: number): number[] {
  const out: number[] = [];
  for (let offset = 0; offset < groupSize; offset++) {
    for (let start = 0; start < features; start += groupSize) {
      if (start + offset < features) out.push(start + offset);
    }
  }
  return out;
}

export function generateTabular(config: TabularConfig): TabularData {
  const p = Math.max(1, Math.round(config.features));
  const n = Math.max(4, Math.round(config.count));
  const groupSize = Math.max(1, Math.round(config.groupSize));
  const rng = makeRng(config.seed);
  const rho = Math.min(0.99, Math.max(0, config.correlation));

  const groups: number[][] = [];
  for (let start = 0; start < p; start += groupSize) {
    groups.push(Array.from({ length: Math.min(groupSize, p - start) }, (_, i) => start + i));
  }

  const trueWeights = new Array<number>(p).fill(0);
  const order = informativeOrder(p, groupSize);
  const k = Math.min(p, Math.max(0, Math.round(config.informative)));
  for (let i = 0; i < k; i++) {
    const sign = rng() < 0.5 ? -1 : 1;
    trueWeights[order[i]] = sign * uniform(rng, 1, 3);
  }
  const trueBias = 2;

  const scales = new Array<number>(p).fill(1);
  const offsets = new Array<number>(p).fill(0);
  if (config.variedScales) {
    for (let j = 0; j < p; j++) {
      scales[j] = Math.pow(10, uniform(rng, -1, 1));
      offsets[j] = uniform(rng, -2, 2) * scales[j];
    }
  }

  // Standard-normal features, correlated within each group through a shared latent.
  const Z: number[][] = [];
  const signal: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(p).fill(0);
    for (const group of groups) {
      const latent = gauss(rng);
      for (const j of group) row[j] = Math.sqrt(rho) * latent + Math.sqrt(1 - rho) * gauss(rng);
    }
    Z.push(row);
    let s = trueBias;
    for (let j = 0; j < p; j++) s += trueWeights[j] * row[j];
    signal.push(s);
  }
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  const spread = Math.sqrt(signal.reduce((a, s) => a + (s - mean) * (s - mean), 0) / n) || 1;

  const X = Z.map((row) => row.map((v, j) => v * scales[j] + offsets[j]));
  const y = signal.map((s) => s + gauss(rng, 0, config.noise * spread));

  const indices = shuffled(rng, Array.from({ length: n }, (_, i) => i));
  const trainCount = Math.max(2, Math.min(n - 1, Math.round(n * config.trainFraction)));
  return {
    kind: 'tabular',
    X,
    y,
    featureNames: Array.from({ length: p }, (_, j) => featureName(j)),
    trueWeights,
    trueBias,
    groups,
    scales,
    trainIndex: indices.slice(0, trainCount).sort((a, b) => a - b),
    testIndex: indices.slice(trainCount).sort((a, b) => a - b),
  };
}

export function trainRows(data: TabularData): Design {
  return { X: data.trainIndex.map((i) => data.X[i]), y: data.trainIndex.map((i) => data.y[i]) };
}

export function testRows(data: TabularData): Design {
  return { X: data.testIndex.map((i) => data.X[i]), y: data.testIndex.map((i) => data.y[i]) };
}
