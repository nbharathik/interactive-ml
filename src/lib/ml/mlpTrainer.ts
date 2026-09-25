/** Trains a dense network of chosen depth, activation and loss, and reports what each layer saw. */

import type { Point2D, RegressionPoint } from '../datasets/types';
import { makeRng, shuffled } from '../math/rng';
import { activation } from './activations';
import type { ActivationName, ActivationParams } from './activations';
import type { LossFn, LossName, LossParams } from './losses';
import { batchGradients, createMlp, createOptStates, gradientNorms, lossFnOf, predict, updateWithOptimiser } from './mlp';
import type { Mlp, MlpSpec } from './mlp';
import type { Initialiser } from './neuralNetwork';
import type { OptState, OptimiserConfig } from './optim';
import { outputFromHidden } from './warp';

export type Task = 'classify' | 'regress';

export interface TrainerConfig {
  task: Task;
  depth: number;
  width: number;
  activation: ActivationName;
  activationParams?: ActivationParams;
  loss: LossName;
  lossParams?: LossParams;
  classCount: number;
  init: Initialiser;
  initScale: number;
  seed: number;
  optimiser: OptimiserConfig;
  learningRate: number;
  batchSize: number;
}

export interface TrainerSet {
  inputs: number[][];
  targets: number[][];
  /** Class labels, or null for regression. */
  labels: number[] | null;
}

export interface TrainerData {
  train: TrainerSet;
  test: TrainerSet;
}

/** Points scaled by a sixth, so the inputs sit in [-1, 1]. */
export function dataFromPoints(train: readonly Point2D[], test: readonly Point2D[], lossFn: LossFn, classCount: number): TrainerData {
  const make = (points: readonly Point2D[]): TrainerSet => ({
    inputs: points.map((p) => [p.x / 6, p.y / 6]),
    targets: points.map((p) => lossFn.encodeClass(p.label, classCount)),
    labels: points.map((p) => p.label),
  });
  return { train: make(train), test: make(test) };
}

export interface ValueScale {
  mean: number;
  std: number;
}

/** x mapped onto [-1, 1] and y standardised on the training set. */
export function dataFromRegression(
  train: readonly RegressionPoint[],
  test: readonly RegressionPoint[],
  xRange: [number, number],
  scale: ValueScale,
): TrainerData {
  const make = (points: readonly RegressionPoint[]): TrainerSet => ({
    inputs: points.map((p) => [scaleX(p.x, xRange)]),
    targets: points.map((p) => [(p.y - scale.mean) / scale.std]),
    labels: null,
  });
  return { train: make(train), test: make(test) };
}

export function scaleX(x: number, xRange: [number, number]): number {
  return ((x - xRange[0]) / (xRange[1] - xRange[0] || 1)) * 2 - 1;
}

export function valueScale(points: readonly RegressionPoint[]): ValueScale {
  if (points.length === 0) return { mean: 0, std: 1 };
  const mean = points.reduce((s, p) => s + p.y, 0) / points.length;
  const sd = Math.sqrt(points.reduce((s, p) => s + (p.y - mean) ** 2, 0) / points.length);
  return { mean, std: sd < 1e-9 ? 1 : sd };
}

export interface LayerDiagnostics {
  /** Norm of the mean gradient on the layer's weights in the last batch. */
  gradNorm: number;
  weightNorm: number;
  /** Pre-activations seen in the last batch, at most SAMPLE_CAP of them. */
  zSamples: number[];
  zStd: number;
  /** Standard deviation of the layer's activations, the signal that reaches the next layer. */
  aStd: number;
  /** Share of the layer's units whose activation slope was below 1e-3 on every sample of the batch. */
  smallGradFraction: number;
}

export interface TrainerState {
  net: Mlp;
  spec: MlpSpec;
  opt: OptState[];
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number | null;
  testAccuracy: number | null;
  lossHistory: Array<{ train: number; test: number }>;
  layers: LayerDiagnostics[];
  /** Outputs and targets of the last batch, for the loss chart's rug. */
  outputs: number[][];
  batchTargets: number[][];
  diverged: boolean;
}

const SAMPLE_CAP = 256;
const MAX_HISTORY = 3000;

export function specOf(config: TrainerConfig, inputSize: number): MlpSpec {
  const lossFn = lossFnOf({ loss: config.loss, lossParams: { ...config.lossParams, classCount: config.classCount } } as MlpSpec);
  const outputs = lossFn.outputWidth(config.classCount);
  // Depth zero is one output unit on the inputs: logistic regression for a two-class task.
  const depth = Math.max(0, Math.round(config.depth));
  const sizes = [inputSize, ...new Array<number>(depth).fill(Math.max(1, Math.round(config.width))), outputs];
  // Squared error on a class goes through a sigmoid; every other loss reads the raw output.
  const outputActivation: ActivationName = config.task === 'classify' && config.loss === 'mse' ? 'sigmoid' : 'linear';
  return {
    sizes,
    activations: [...new Array<ActivationName>(depth).fill(config.activation), outputActivation],
    activationParams: config.activationParams,
    loss: config.loss,
    lossParams: { ...config.lossParams, classCount: config.classCount },
    init: config.init,
    initScale: config.initScale,
    seed: config.seed,
  };
}

function emptyDiagnostics(net: Mlp): LayerDiagnostics[] {
  return net.layers.map((layer) => ({
    gradNorm: 0,
    weightNorm: Math.sqrt(layer.W.reduce((s, w) => s + w * w, 0)),
    zSamples: [],
    zStd: 0,
    aStd: 0,
    smallGradFraction: 0,
  }));
}

export function evaluateSet(net: Mlp, spec: MlpSpec, set: TrainerSet): { loss: number; accuracy: number | null } {
  const lossFn = lossFnOf(spec);
  if (set.inputs.length === 0) return { loss: 0, accuracy: null };
  let lossSum = 0;
  let correct = 0;
  for (let i = 0; i < set.inputs.length; i++) {
    const out = predict(net, set.inputs[i], spec);
    lossSum += lossFn.f(out, set.targets[i]);
    if (set.labels && lossFn.decodeClass(out) === set.labels[i]) correct++;
  }
  return { loss: lossSum / set.inputs.length, accuracy: set.labels ? correct / set.inputs.length : null };
}

export function createTrainer(config: TrainerConfig, data: TrainerData): TrainerState {
  const inputSize = data.train.inputs[0]?.length ?? 1;
  const spec = specOf(config, inputSize);
  const net = createMlp(spec);
  const train = evaluateSet(net, spec, data.train);
  const test = evaluateSet(net, spec, data.test);
  return {
    net,
    spec,
    opt: createOptStates(net, config.optimiser),
    epoch: 0,
    trainLoss: train.loss,
    testLoss: test.loss,
    trainAccuracy: train.accuracy,
    testAccuracy: test.accuracy,
    lossHistory: [{ train: train.loss, test: test.loss }],
    layers: emptyDiagnostics(net),
    outputs: [],
    batchTargets: [],
    diverged: false,
  };
}

function std(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
}

/** One epoch of mini-batches. Pure. */
export function stepTrainer(state: TrainerState, data: TrainerData, config: TrainerConfig): TrainerState {
  if (state.diverged || data.train.inputs.length === 0) return state;
  const rng = makeRng(config.seed + state.epoch * 7919);
  const order = shuffled(rng, Array.from({ length: data.train.inputs.length }, (_, i) => i));
  const size = Math.max(1, Math.min(order.length, Math.round(config.batchSize)));
  let net = state.net;
  let opt = state.opt;
  let last: ReturnType<typeof batchGradients> | null = null;
  let lastTargets: number[][] = [];
  for (let start = 0; start < order.length; start += size) {
    const batch = order.slice(start, start + size);
    const inputs = batch.map((i) => data.train.inputs[i]);
    const targets = batch.map((i) => data.train.targets[i]);
    const result = batchGradients(net, inputs, targets, state.spec);
    const update = updateWithOptimiser(net, result.grads, opt, config.learningRate, config.optimiser);
    net = update.net;
    opt = update.opt;
    last = result;
    lastTargets = targets;
  }
  const diverged = net.layers.some((l) => l.W.some((w) => !Number.isFinite(w) || Math.abs(w) > 1e6));
  const train = diverged ? { loss: Number.POSITIVE_INFINITY, accuracy: null } : evaluateSet(net, state.spec, data.train);
  const test = diverged ? { loss: Number.POSITIVE_INFINITY, accuracy: null } : evaluateSet(net, state.spec, data.test);
  const norms = last ? gradientNorms(last.grads) : [];
  const layers: LayerDiagnostics[] = net.layers.map((layer, l) => {
    const zs = last ? last.zs[l] : [];
    const sample = zs.length > SAMPLE_CAP ? zs.filter((_, i) => i % Math.ceil(zs.length / SAMPLE_CAP) === 0).slice(0, SAMPLE_CAP) : zs.slice();
    const act = activation(layer.activation, state.spec.activationParams);
    return {
      gradNorm: norms[l] ?? 0,
      weightNorm: Math.sqrt(layer.W.reduce((s, w) => s + w * w, 0)),
      zSamples: sample,
      zStd: std(zs),
      aStd: std(zs.map((z) => act.f(z))),
      smallGradFraction: last ? last.smallGradFraction[l] : 0,
    };
  });
  const history = state.lossHistory.concat({ train: train.loss, test: test.loss });
  return {
    net,
    spec: state.spec,
    opt,
    epoch: state.epoch + 1,
    trainLoss: train.loss,
    testLoss: test.loss,
    trainAccuracy: train.accuracy,
    testAccuracy: test.accuracy,
    lossHistory: history.length > MAX_HISTORY ? history.slice(-MAX_HISTORY) : history,
    layers,
    outputs: last ? last.outputs.slice(0, SAMPLE_CAP) : [],
    batchTargets: lastTargets.slice(0, SAMPLE_CAP),
    diverged,
  };
}

/** A signed class score in [-1, 1] for a two-class input, whatever the loss. */
export function signedScore(state: TrainerState, input: readonly number[]): number {
  return scoreFromOutput(state.spec.loss, predict(state.net, input, state.spec));
}

/** The same score from a vector of the last hidden layer, through the output layer alone. */
export function signedScoreFromHidden(state: TrainerState, hidden: readonly number[]): number {
  return scoreFromOutput(state.spec.loss, outputFromHidden(state.net, state.spec, hidden));
}

export function scoreFromOutput(lossName: LossName, out: readonly number[]): number {
  switch (lossName) {
    case 'softmaxCE': {
      const shifted = out.map((v) => v - Math.max(...out));
      const exps = shifted.map(Math.exp);
      const total = exps.reduce((a, b) => a + b, 0);
      return (2 * exps[1]) / total - 1;
    }
    case 'bce':
      return Math.tanh(out[0] / 2);
    case 'hinge':
      return Math.max(-1, Math.min(1, out[0]));
    default:
      return Math.max(-1, Math.min(1, 2 * out[0] - 1));
  }
}

/** The class the network calls for an input. */
export function predictedClass(state: TrainerState, input: readonly number[]): number {
  return lossFnOf(state.spec).decodeClass(predict(state.net, input, state.spec));
}

/** The class the output layer calls for a vector of the last hidden layer. */
export function predictedClassFromHidden(state: TrainerState, hidden: readonly number[]): number {
  return lossFnOf(state.spec).decodeClass(outputFromHidden(state.net, state.spec, hidden));
}

/** Where a sample sits on the loss curve's axis: the score, the error, or the true-class logit. */
export function lossAxisValue(lossName: LossName, output: readonly number[], target: readonly number[]): number {
  switch (lossName) {
    case 'softmaxCE': {
      const index = target.findIndex((v) => v === 1);
      return output[Math.max(0, index)] ?? 0;
    }
    case 'bce':
    case 'hinge':
      return output[0];
    default:
      return output[0] - target[0];
  }
}
