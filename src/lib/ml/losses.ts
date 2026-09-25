/** The loss catalogue: value and gradient per sample, target encoding, and a one-dimensional curve to plot. */

import { sigmoid, softplus } from './activations';

export type LossName = 'mse' | 'mae' | 'huber' | 'bce' | 'hinge' | 'softmaxCE';

export interface LossParams {
  huberDelta?: number;
  classCount?: number;
}

/** What the loss expects the model to output for a sample. */
export type LossKind = 'regression' | 'binary' | 'multiclass';

export interface LossCurve {
  xLabel: string;
  xDomain: [number, number];
  /** Class labels (or 0 for a regression loss) worth drawing a curve for. */
  targets: number[];
  f(x: number, target: number): number;
  df(x: number, target: number): number;
}

export interface LossFn {
  name: LossName;
  label: string;
  tex: string;
  kind: LossKind;
  /** Width of the model output this loss reads. */
  outputWidth(classCount: number): number;
  f(output: readonly number[], target: readonly number[]): number;
  /** dL/d(output). Sigmoid and softmax are fused, so this is p minus y for bce and softmaxCE. */
  grad(output: readonly number[], target: readonly number[]): number[];
  encodeClass(label: number, classCount: number): number[];
  encodeValue(y: number): number[];
  decodeClass(output: readonly number[]): number;
  decodeValue(output: readonly number[]): number;
  curve: LossCurve;
}

export const LOSS_NAMES: readonly LossName[] = ['mse', 'mae', 'huber', 'bce', 'hinge', 'softmaxCE'];

export const LOSS_LABELS: Record<LossName, string> = {
  mse: 'Squared error',
  mae: 'Absolute error',
  huber: 'Huber',
  bce: 'Cross-entropy',
  hinge: 'Hinge',
  softmaxCE: 'Softmax cross-entropy',
};

export function logSumExp(logits: readonly number[]): number {
  let max = -Infinity;
  for (const z of logits) if (z > max) max = z;
  if (!Number.isFinite(max)) return max;
  let sum = 0;
  for (const z of logits) sum += Math.exp(z - max);
  return max + Math.log(sum);
}

export function softmax(logits: readonly number[]): number[] {
  const lse = logSumExp(logits);
  return logits.map((z) => Math.exp(z - lse));
}

function argmax(values: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

const sign = (x: number) => (x > 0 ? 1 : x < 0 ? -1 : 0);

export function loss(name: LossName, params: LossParams = {}): LossFn {
  const delta = params.huberDelta ?? 1;
  const k = Math.max(2, params.classCount ?? 2);
  const label = LOSS_LABELS[name];
  const one = () => 1;
  const regression = {
    kind: 'regression' as const,
    outputWidth: one,
    encodeClass: (c: number) => [c === 1 ? 1 : 0],
    encodeValue: (y: number) => [y],
    decodeClass: (o: readonly number[]) => (o[0] > 0.5 ? 1 : 0),
    decodeValue: (o: readonly number[]) => o[0],
  };
  const errorCurve = (f: (e: number) => number, df: (e: number) => number): LossCurve => ({
    xLabel: 'error',
    xDomain: [-4, 4],
    targets: [0],
    f: (x) => f(x),
    df: (x) => df(x),
  });

  switch (name) {
    case 'mae':
      return {
        name,
        label,
        tex: 'L = |\\hat{y} - y|',
        ...regression,
        f: (o, t) => Math.abs(o[0] - t[0]),
        grad: (o, t) => [sign(o[0] - t[0])],
        curve: errorCurve(Math.abs, sign),
      };
    case 'huber': {
      const h = (e: number) => (Math.abs(e) <= delta ? 0.5 * e * e : delta * (Math.abs(e) - 0.5 * delta));
      const dh = (e: number) => (Math.abs(e) <= delta ? e : delta * sign(e));
      return {
        name,
        label,
        tex: 'L = \\tfrac{1}{2}e^2 \\text{ if } |e| \\le \\delta \\text{ else } \\delta(|e| - \\tfrac{1}{2}\\delta)',
        ...regression,
        f: (o, t) => h(o[0] - t[0]),
        grad: (o, t) => [dh(o[0] - t[0])],
        curve: errorCurve(h, dh),
      };
    }
    case 'bce':
      return {
        name,
        label,
        tex: 'L = \\ln(1 + e^{z}) - y\\,z',
        kind: 'binary',
        outputWidth: one,
        encodeClass: (c) => [c === 1 ? 1 : 0],
        encodeValue: (y) => [y],
        decodeClass: (o) => (o[0] > 0 ? 1 : 0),
        decodeValue: (o) => sigmoid(o[0]),
        f: (o, t) => softplus(o[0]) - t[0] * o[0],
        grad: (o, t) => [sigmoid(o[0]) - t[0]],
        curve: {
          xLabel: 'logit z',
          xDomain: [-6, 6],
          targets: [1, 0],
          f: (z, y) => softplus(z) - y * z,
          df: (z, y) => sigmoid(z) - y,
        },
      };
    case 'hinge':
      return {
        name,
        label,
        tex: 'L = \\max(0, 1 - y\\,s)',
        kind: 'binary',
        outputWidth: one,
        encodeClass: (c) => [c === 1 ? 1 : -1],
        encodeValue: (y) => [y],
        decodeClass: (o) => (o[0] > 0 ? 1 : 0),
        decodeValue: (o) => o[0],
        f: (o, t) => Math.max(0, 1 - t[0] * o[0]),
        grad: (o, t) => [t[0] * o[0] < 1 ? -t[0] : 0],
        curve: {
          xLabel: 'score s',
          xDomain: [-3, 3],
          targets: [1, 0],
          f: (s, c) => Math.max(0, 1 - (c === 1 ? 1 : -1) * s),
          df: (s, c) => {
            const y = c === 1 ? 1 : -1;
            return y * s < 1 ? -y : 0;
          },
        },
      };
    case 'softmaxCE':
      return {
        name,
        label,
        tex: 'L = \\ln \\sum_j e^{z_j} - z_{y}',
        kind: 'multiclass',
        outputWidth: (c) => Math.max(2, c),
        encodeClass: (c, count) => {
          const out = new Array<number>(Math.max(2, count)).fill(0);
          if (c >= 0 && c < out.length) out[c] = 1;
          return out;
        },
        encodeValue: (y) => [y],
        decodeClass: (o) => argmax(o),
        decodeValue: (o) => argmax(o),
        f: (o, t) => {
          let picked = 0;
          for (let i = 0; i < o.length; i++) picked += t[i] * o[i];
          return logSumExp(o) - picked;
        },
        grad: (o, t) => softmax(o).map((p, i) => p - t[i]),
        curve: {
          xLabel: 'true-class logit',
          xDomain: [-6, 6],
          targets: [0],
          f: (z) => Math.log(Math.exp(z) + (k - 1)) - z,
          df: (z) => Math.exp(z) / (Math.exp(z) + (k - 1)) - 1,
        },
      };
    default:
      return {
        name: 'mse',
        label,
        tex: 'L = \\tfrac{1}{2}(\\hat{y} - y)^2',
        ...regression,
        f: (o, t) => 0.5 * (o[0] - t[0]) * (o[0] - t[0]),
        grad: (o, t) => [o[0] - t[0]],
        curve: errorCurve((e) => 0.5 * e * e, (e) => e),
      };
  }
}
