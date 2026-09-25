/** A small dense network with explicit per-layer primitives, so a page can run one phase at a time. */

import { makeRng, gauss, uniform } from '../math/rng';
import { activation } from './activations';
import type { ActivationName, ActivationParams } from './activations';
import { loss as lossOf } from './losses';
import type { LossFn, LossName, LossParams } from './losses';
import type { Initialiser } from './neuralNetwork';
import { createOptState, optStep } from './optim';
import type { OptState, OptimiserConfig } from './optim';

export interface MlpSpec {
  /** [inputs, hidden..., outputs]. */
  sizes: number[];
  /** One per weight layer; the last is the output activation. */
  activations: ActivationName[];
  activationParams?: ActivationParams;
  loss: LossName;
  lossParams?: LossParams;
  init: Initialiser;
  /** Multiplies the initial weights; 1 is the textbook scale. */
  initScale?: number;
  seed: number;
}

export interface DenseLayer {
  inSize: number;
  outSize: number;
  /** Row-major: W[out * inSize + in]. */
  W: number[];
  b: number[];
  activation: ActivationName;
}

export interface Mlp {
  layers: DenseLayer[];
}

export interface LayerCache {
  input: number[];
  z: number[];
  a: number[];
}

export interface LayerGrads {
  /** dL/dz for every unit. */
  delta: number[];
  dW: number[];
  db: number[];
}

export interface Gradients {
  dW: number[][];
  db: number[][];
}

function initialWeight(init: Initialiser, fanIn: number, fanOut: number, rng: () => number): number {
  switch (init) {
    case 'xavier':
      return gauss(rng, 0, Math.sqrt(2 / (fanIn + fanOut)));
    case 'he':
      return gauss(rng, 0, Math.sqrt(2 / fanIn));
    case 'small':
      return uniform(rng, -0.1, 0.1);
    case 'large':
      return gauss(rng, 0, 2.5);
    default:
      return 0;
  }
}

export function createMlp(spec: MlpSpec): Mlp {
  const rng = makeRng(spec.seed);
  const scale = spec.initScale ?? 1;
  const layers: DenseLayer[] = [];
  for (let l = 0; l < spec.sizes.length - 1; l++) {
    const inSize = spec.sizes[l];
    const outSize = spec.sizes[l + 1];
    const W = new Array<number>(inSize * outSize);
    for (let i = 0; i < W.length; i++) W[i] = scale * initialWeight(spec.init, inSize, outSize, rng);
    layers.push({ inSize, outSize, W, b: new Array<number>(outSize).fill(0), activation: spec.activations[l] ?? 'linear' });
  }
  return { layers };
}

export function lossFnOf(spec: MlpSpec): LossFn {
  return lossOf(spec.loss, spec.lossParams);
}

/* ---------------- per-layer primitives ---------------- */

export function layerForward(layer: DenseLayer, input: readonly number[], params?: ActivationParams): { z: number[]; a: number[] } {
  const act = activation(layer.activation, params);
  const z = new Array<number>(layer.outSize);
  const a = new Array<number>(layer.outSize);
  for (let o = 0; o < layer.outSize; o++) {
    let sum = layer.b[o];
    const base = o * layer.inSize;
    for (let i = 0; i < layer.inSize; i++) sum += layer.W[base + i] * input[i];
    z[o] = sum;
    a[o] = act.f(sum);
  }
  return { z, a };
}

/** dL/dz at the output: the loss gradient through the output activation. */
export function outputDelta(
  layer: DenseLayer,
  z: readonly number[],
  a: readonly number[],
  target: readonly number[],
  lossFn: LossFn,
  params?: ActivationParams,
): number[] {
  const act = activation(layer.activation, params);
  const grad = lossFn.grad(a, target);
  return grad.map((g, o) => g * act.df(z[o]));
}

/** dL/dz for a hidden layer from the layer after it. */
export function layerDelta(
  layer: DenseLayer,
  z: readonly number[],
  next: DenseLayer,
  nextDelta: readonly number[],
  params?: ActivationParams,
): number[] {
  const act = activation(layer.activation, params);
  const delta = new Array<number>(layer.outSize);
  for (let o = 0; o < layer.outSize; o++) {
    let sum = 0;
    for (let k = 0; k < next.outSize; k++) sum += next.W[k * next.inSize + o] * nextDelta[k];
    delta[o] = sum * act.df(z[o]);
  }
  return delta;
}

export function layerGradients(delta: readonly number[], input: readonly number[]): { dW: number[]; db: number[] } {
  const dW = new Array<number>(delta.length * input.length);
  for (let o = 0; o < delta.length; o++) {
    for (let i = 0; i < input.length; i++) dW[o * input.length + i] = delta[o] * input[i];
  }
  return { dW, db: delta.slice() };
}

/* ---------------- the whole net ---------------- */

export function forward(net: Mlp, input: readonly number[], spec: MlpSpec): LayerCache[] {
  const caches: LayerCache[] = [];
  let current = input.slice();
  for (const layer of net.layers) {
    const { z, a } = layerForward(layer, current, spec.activationParams);
    caches.push({ input: current, z, a });
    current = a;
  }
  return caches;
}

export function backward(net: Mlp, caches: readonly LayerCache[], target: readonly number[], spec: MlpSpec): LayerGrads[] {
  const lossFn = lossFnOf(spec);
  const last = net.layers.length - 1;
  const grads: LayerGrads[] = new Array(net.layers.length);
  let delta = outputDelta(net.layers[last], caches[last].z, caches[last].a, target, lossFn, spec.activationParams);
  for (let l = last; l >= 0; l--) {
    const { dW, db } = layerGradients(delta, caches[l].input);
    grads[l] = { delta, dW, db };
    if (l > 0) delta = layerDelta(net.layers[l - 1], caches[l - 1].z, net.layers[l], delta, spec.activationParams);
  }
  return grads;
}

export function predict(net: Mlp, input: readonly number[], spec: MlpSpec): number[] {
  const caches = forward(net, input, spec);
  return caches[caches.length - 1].a;
}

export function sampleLoss(net: Mlp, input: readonly number[], target: readonly number[], spec: MlpSpec): number {
  return lossFnOf(spec).f(predict(net, input, spec), target);
}

export function zeroGradients(net: Mlp): Gradients {
  return {
    dW: net.layers.map((l) => new Array<number>(l.W.length).fill(0)),
    db: net.layers.map((l) => new Array<number>(l.b.length).fill(0)),
  };
}

export function addGradients(into: Gradients, grads: readonly LayerGrads[], scale = 1): Gradients {
  return {
    dW: into.dW.map((row, l) => row.map((v, i) => v + scale * grads[l].dW[i])),
    db: into.db.map((row, l) => row.map((v, i) => v + scale * grads[l].db[i])),
  };
}

export function scaleGradients(grads: Gradients, scale: number): Gradients {
  return { dW: grads.dW.map((row) => row.map((v) => v * scale)), db: grads.db.map((row) => row.map((v) => v * scale)) };
}

export interface BatchResult {
  /** Mean gradient over the batch. */
  grads: Gradients;
  loss: number;
  /** Pre-activations per layer, flattened over the batch. */
  zs: number[][];
  /** Share of units per layer whose activation slope was below 1e-3 on every sample: dead or saturated. */
  smallGradFraction: number[];
  outputs: number[][];
}

export function batchGradients(net: Mlp, inputs: readonly number[][], targets: readonly number[][], spec: MlpSpec): BatchResult {
  const lossFn = lossFnOf(spec);
  let total = zeroGradients(net);
  let lossSum = 0;
  const zs = net.layers.map(() => [] as number[]);
  const alive = net.layers.map((l) => new Array<boolean>(l.outSize).fill(false));
  const outputs: number[][] = [];
  const acts = net.layers.map((l) => activation(l.activation, spec.activationParams));
  for (let s = 0; s < inputs.length; s++) {
    const caches = forward(net, inputs[s], spec);
    const grads = backward(net, caches, targets[s], spec);
    total = addGradients(total, grads);
    lossSum += lossFn.f(caches[caches.length - 1].a, targets[s]);
    outputs.push(caches[caches.length - 1].a);
    caches.forEach((cache, l) => {
      cache.z.forEach((z, o) => {
        zs[l].push(z);
        if (Math.abs(acts[l].df(z)) >= 1e-3) alive[l][o] = true;
      });
    });
  }
  const n = Math.max(1, inputs.length);
  return {
    grads: scaleGradients(total, 1 / n),
    loss: lossSum / n,
    zs,
    smallGradFraction: alive.map((units) => (inputs.length === 0 ? 0 : units.filter((a) => !a).length / Math.max(1, units.length))),
    outputs,
  };
}

export function applyGradients(net: Mlp, grads: Gradients, lr: number): Mlp {
  return {
    layers: net.layers.map((layer, l) => ({
      ...layer,
      W: layer.W.map((w, i) => w - lr * grads.dW[l][i]),
      b: layer.b.map((b, i) => b - lr * grads.db[l][i]),
    })),
  };
}

/** One accumulator per tensor: [W0, b0, W1, b1, ...]. */
export function createOptStates(net: Mlp, config: OptimiserConfig): OptState[] {
  return net.layers.flatMap((layer) => [createOptState(config, layer.W.length), createOptState(config, layer.b.length)]);
}

export function updateWithOptimiser(
  net: Mlp,
  grads: Gradients,
  opt: readonly OptState[],
  lr: number,
  config: OptimiserConfig,
): { net: Mlp; opt: OptState[] } {
  const nextOpt: OptState[] = [];
  const layers = net.layers.map((layer, l) => {
    const w = optStep(opt[2 * l], grads.dW[l], lr, config);
    const b = optStep(opt[2 * l + 1], grads.db[l], lr, config);
    nextOpt.push(w.state, b.state);
    return { ...layer, W: layer.W.map((v, i) => v + w.delta[i]), b: layer.b.map((v, i) => v + b.delta[i]) };
  });
  return { net: { layers }, opt: nextOpt };
}

export function parameterCount(net: Mlp): number {
  return net.layers.reduce((sum, l) => sum + l.W.length + l.b.length, 0);
}

export function gradientNorms(grads: Gradients): number[] {
  return grads.dW.map((row, l) => Math.sqrt(row.reduce((s, v) => s + v * v, 0) + grads.db[l].reduce((s, v) => s + v * v, 0)));
}

/** Central differences on every parameter, for checking the analytic gradient. */
export function numericalGradients(net: Mlp, input: readonly number[], target: readonly number[], spec: MlpSpec, h = 1e-5): Gradients {
  const out = zeroGradients(net);
  const probe = (l: number, which: 'W' | 'b', i: number): number => {
    const nudge = (sign: number): Mlp => ({
      layers: net.layers.map((layer, k) =>
        k !== l ? layer : { ...layer, [which]: layer[which].map((v, j) => (j === i ? v + sign * h : v)) },
      ),
    });
    return (sampleLoss(nudge(1), input, target, spec) - sampleLoss(nudge(-1), input, target, spec)) / (2 * h);
  };
  net.layers.forEach((layer, l) => {
    for (let i = 0; i < layer.W.length; i++) out.dW[l][i] = probe(l, 'W', i);
    for (let i = 0; i < layer.b.length; i++) out.db[l][i] = probe(l, 'b', i);
  });
  return out;
}

export interface GradientCheck {
  maxAbs: number;
  maxRel: number;
  analytic: Gradients;
  numeric: Gradients;
}

export function gradientCheck(net: Mlp, input: readonly number[], target: readonly number[], spec: MlpSpec): GradientCheck {
  const grads = backward(net, forward(net, input, spec), target, spec);
  const analytic: Gradients = { dW: grads.map((g) => g.dW), db: grads.map((g) => g.db) };
  const numeric = numericalGradients(net, input, target, spec);
  let maxAbs = 0;
  let maxRel = 0;
  const compare = (a: number, b: number) => {
    const diff = Math.abs(a - b);
    maxAbs = Math.max(maxAbs, diff);
    maxRel = Math.max(maxRel, diff / Math.max(1e-6, Math.abs(a) + Math.abs(b)));
  };
  analytic.dW.forEach((row, l) => row.forEach((v, i) => compare(v, numeric.dW[l][i])));
  analytic.db.forEach((row, l) => row.forEach((v, i) => compare(v, numeric.db[l][i])));
  return { maxAbs, maxRel, analytic, numeric };
}
