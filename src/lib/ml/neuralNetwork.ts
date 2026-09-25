/** Small feed-forward network with explicit forward and backward passes, no autograd. */

import type { Point2D } from '../datasets/types';
import { gauss, makeRng, shuffled } from '../math/rng';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';
export type InputFeature = 'x1' | 'x2' | 'x1sq' | 'x2sq' | 'x1x2' | 'sinx1' | 'sinx2';
export type Initialiser = 'xavier' | 'he' | 'small' | 'large' | 'zeros';
export type Regularisation = 'none' | 'l1' | 'l2';

export interface NetConfig {
  /** Hidden layer widths. An empty array is a bare logistic regression. */
  hiddenLayers: number[];
  activation: Activation;
  learningRate: number;
  batchSize: number;
  inputFeatures: InputFeature[];
  initialiser: Initialiser;
  regularisation: Regularisation;
  regRate: number;
  seed: number;
  /** Fraction of data held out to show the generalisation gap. */
  testFraction: number;
}

export interface Neuron {
  /** Incoming weights, one per neuron in the previous layer. */
  weights: number[];
  bias: number;
  /** Post-activation output for the most recent forward pass, per sample. */
  activation: number;
  /** dLoss/dInput of this neuron for the most recent backward pass. */
  delta: number;
  /** Accumulated gradient for each incoming weight, from the last batch. */
  weightGradients: number[];
  biasGradient: number;
}

export type Layer = Neuron[];

export interface NetState {
  /** layers[0] is the first hidden layer; the last is the single output unit. */
  layers: Layer[];
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy: number;
  testAccuracy: number;
  lossHistory: Array<{ train: number; test: number }>;
  diverged: boolean;
}

/* ---------------- input features ---------------- */

export const FEATURE_LABELS: Record<InputFeature, string> = {
  x1: 'x₁',
  x2: 'x₂',
  x1sq: 'x₁²',
  x2sq: 'x₂²',
  x1x2: 'x₁x₂',
  sinx1: 'sin x₁',
  sinx2: 'sin x₂',
};

/** Raw feature value before `inputVector` divides by 3. The presets depend on these scales. */
export function featureValue(feature: InputFeature, x: number, y: number): number {
  switch (feature) {
    case 'x1':
      return x;
    case 'x2':
      return y;
    case 'x1sq':
      return (x * x) / 3;
    case 'x2sq':
      return (y * y) / 3;
    case 'x1x2':
      return (x * y) / 3;
    case 'sinx1':
      return Math.sin(x) * 2;
    case 'sinx2':
      return Math.sin(y) * 2;
    default:
      return 0;
  }
}

export function inputVector(features: readonly InputFeature[], x: number, y: number): number[] {
  return features.map((f) => featureValue(f, x, y) / 3);
}

/* ---------------- activations ---------------- */

export function activate(name: Activation, z: number): number {
  switch (name) {
    case 'relu':
      return z > 0 ? z : 0;
    case 'sigmoid':
      return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
    case 'linear':
      return z;
    default:
      return Math.tanh(z);
  }
}

/** Derivative in terms of the cached output. */
export function activateDerivative(name: Activation, output: number): number {
  switch (name) {
    case 'relu':
      return output > 0 ? 1 : 0;
    case 'sigmoid':
      return output * (1 - output);
    case 'linear':
      return 1;
    default:
      return 1 - output * output;
  }
}

/* ---------------- construction ---------------- */

function initialWeight(
  initialiser: Initialiser,
  fanIn: number,
  fanOut: number,
  rng: () => number,
): number {
  switch (initialiser) {
    case 'he':
      return gauss(rng, 0, Math.sqrt(2 / Math.max(1, fanIn)));
    case 'small':
      return (rng() - 0.5) * 0.1;
    case 'large':
      return (rng() - 0.5) * 6;
    case 'zeros':
      return 0;
    default:
      // Xavier/Glorot
      return (rng() - 0.5) * 2 * Math.sqrt(6 / Math.max(1, fanIn + fanOut));
  }
}

export function createState(config: NetConfig): NetState {
  const rng = makeRng(config.seed);
  const widths = [config.inputFeatures.length, ...config.hiddenLayers, 1];
  const layers: Layer[] = [];

  for (let l = 1; l < widths.length; l++) {
    const fanIn = widths[l - 1];
    const fanOut = widths[l + 1] ?? 1;
    const layer: Layer = [];
    for (let n = 0; n < widths[l]; n++) {
      layer.push({
        weights: Array.from({ length: fanIn }, () =>
          initialWeight(config.initialiser, fanIn, fanOut, rng),
        ),
        bias: 0,
        activation: 0,
        delta: 0,
        weightGradients: new Array<number>(fanIn).fill(0),
        biasGradient: 0,
      });
    }
    layers.push(layer);
  }

  return {
    layers,
    epoch: 0,
    trainLoss: 0,
    testLoss: 0,
    trainAccuracy: 0,
    testAccuracy: 0,
    lossHistory: [],
    diverged: false,
  };
}

function cloneLayers(layers: readonly Layer[]): Layer[] {
  return layers.map((layer) =>
    layer.map((neuron) => ({
      weights: neuron.weights.slice(),
      bias: neuron.bias,
      activation: neuron.activation,
      delta: neuron.delta,
      weightGradients: neuron.weightGradients.slice(),
      biasGradient: neuron.biasGradient,
    })),
  );
}

/* ---------------- forward pass ---------------- */

/** Forward pass for one input, caching each activation. Returns the raw (linear) output. */
export function forward(layers: Layer[], input: readonly number[], activation: Activation): number {
  let previous = input as number[];
  for (let l = 0; l < layers.length; l++) {
    const isOutput = l === layers.length - 1;
    const current: number[] = [];
    for (const neuron of layers[l]) {
      let z = neuron.bias;
      for (let i = 0; i < neuron.weights.length; i++) z += neuron.weights[i] * previous[i];
      // The output neuron stays linear.
      neuron.activation = isOutput ? z : activate(activation, z);
      current.push(neuron.activation);
    }
    previous = current;
  }
  return previous[0];
}

/** Read-only forward pass, used for the decision-boundary grid. */
export function evaluate(layers: readonly Layer[], input: readonly number[], activation: Activation): number {
  return evaluateNeuron(layers, input, activation, layers.length - 1, 0);
}

/** Every unit's weighted sum z and its output, layer by layer, for one input. */
export function layerValues(
  layers: readonly Layer[],
  input: readonly number[],
  activation: Activation,
): { sums: number[][]; activations: number[][] } {
  const sums: number[][] = [];
  const activations: number[][] = [];
  let previous = input as number[];
  for (let l = 0; l < layers.length; l++) {
    const isOutput = l === layers.length - 1;
    const z: number[] = [];
    const a: number[] = [];
    for (const neuron of layers[l]) {
      let sum = neuron.bias;
      for (let i = 0; i < neuron.weights.length; i++) sum += neuron.weights[i] * previous[i];
      z.push(sum);
      a.push(isOutput ? sum : activate(activation, sum));
    }
    sums.push(z);
    activations.push(a);
    previous = a;
  }
  return { sums, activations };
}

/** The output of one specific hidden neuron, what each unit "sees". */
export function evaluateNeuron(
  layers: readonly Layer[],
  input: readonly number[],
  activation: Activation,
  targetLayer: number,
  targetNeuron: number,
): number {
  let previous = input as number[];
  for (let l = 0; l <= targetLayer && l < layers.length; l++) {
    const isOutput = l === layers.length - 1;
    const current: number[] = [];
    for (const neuron of layers[l]) {
      let z = neuron.bias;
      for (let i = 0; i < neuron.weights.length; i++) z += neuron.weights[i] * previous[i];
      current.push(isOutput ? z : activate(activation, z));
    }
    previous = current;
  }
  return previous[targetNeuron] ?? 0;
}

/* ---------------- loss ---------------- */

/** Squared error on the raw output against targets of -1 / +1. */
export function targetOf(label: number): number {
  return label === 1 ? 1 : -1;
}

export function sampleLoss(output: number, target: number): number {
  const d = output - target;
  return 0.5 * d * d;
}

/* ---------------- backward pass ---------------- */

/** Backpropagate one sample into the neurons' gradients. Call `forward` first. */
export function backward(
  layers: Layer[],
  input: readonly number[],
  target: number,
  activation: Activation,
): void {
  const lastIndex = layers.length - 1;

  // Output layer: dLoss/dz = (output − target), because the output is linear.
  const output = layers[lastIndex][0];
  output.delta = output.activation - target;

  // Hidden layers: delta = weighted sum of downstream deltas times the activation derivative.
  for (let l = lastIndex - 1; l >= 0; l--) {
    for (let n = 0; n < layers[l].length; n++) {
      let sum = 0;
      for (const downstream of layers[l + 1]) sum += downstream.weights[n] * downstream.delta;
      layers[l][n].delta = sum * activateDerivative(activation, layers[l][n].activation);
    }
  }

  // Turn deltas into weight gradients: delta × the input that weight carried.
  for (let l = 0; l < layers.length; l++) {
    const previous = l === 0 ? (input as number[]) : layers[l - 1].map((n) => n.activation);
    for (const neuron of layers[l]) {
      for (let i = 0; i < neuron.weights.length; i++) {
        neuron.weightGradients[i] += neuron.delta * previous[i];
      }
      neuron.biasGradient += neuron.delta;
    }
  }
}

function zeroGradients(layers: Layer[]): void {
  for (const layer of layers) {
    for (const neuron of layer) {
      neuron.weightGradients.fill(0);
      neuron.biasGradient = 0;
    }
  }
}

function applyGradients(layers: Layer[], config: NetConfig, batchSize: number): void {
  const scale = config.learningRate / Math.max(1, batchSize);
  for (const layer of layers) {
    for (const neuron of layer) {
      for (let i = 0; i < neuron.weights.length; i++) {
        let grad = neuron.weightGradients[i];
        if (config.regularisation === 'l2') grad += config.regRate * neuron.weights[i];
        else if (config.regularisation === 'l1') grad += config.regRate * Math.sign(neuron.weights[i]);
        neuron.weights[i] -= scale * grad;
      }
      neuron.bias -= scale * neuron.biasGradient;
    }
  }
}

/* ---------------- training ---------------- */

export interface TrainingData {
  train: Point2D[];
  test: Point2D[];
}

/** One epoch: a full sweep over the training set in mini-batches. */
export function step(
  state: NetState,
  data: TrainingData,
  config: NetConfig,
): NetState {
  if (state.diverged || data.train.length === 0) return state;

  const layers = cloneLayers(state.layers);
  const batchSize = Math.max(1, Math.min(data.train.length, config.batchSize));

  // Deterministic order so a seed replays exactly.
  const order = shuffled(makeRng(config.seed + state.epoch * 7919), data.train.map((_, i) => i));

  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize);
    zeroGradients(layers);
    for (const index of batch) {
      const point = data.train[index];
      const input = inputVector(config.inputFeatures, point.x, point.y);
      forward(layers, input, config.activation);
      backward(layers, input, targetOf(point.label), config.activation);
    }
    applyGradients(layers, config, batch.length);
  }

  const diverged = layers.some((layer) =>
    layer.some(
      (n) => !Number.isFinite(n.bias) || n.weights.some((w) => !Number.isFinite(w) || Math.abs(w) > 1e8),
    ),
  );

  const trainStats = diverged
    ? { loss: Number.NaN, accuracy: 0 }
    : evaluateSet(layers, data.train, config);
  const testStats = diverged
    ? { loss: Number.NaN, accuracy: 0 }
    : evaluateSet(layers, data.test, config);

  const history = state.lossHistory.concat({ train: trainStats.loss, test: testStats.loss });

  return {
    layers,
    epoch: state.epoch + 1,
    trainLoss: trainStats.loss,
    testLoss: testStats.loss,
    trainAccuracy: trainStats.accuracy,
    testAccuracy: testStats.accuracy,
    lossHistory: history.length > 3000 ? history.slice(-3000) : history,
    diverged,
  };
}

export function evaluateSet(
  layers: readonly Layer[],
  points: readonly Point2D[],
  config: NetConfig,
): { loss: number; accuracy: number } {
  if (points.length === 0) return { loss: 0, accuracy: 0 };
  let loss = 0;
  let hits = 0;
  for (const p of points) {
    const output = evaluate(layers, inputVector(config.inputFeatures, p.x, p.y), config.activation);
    const target = targetOf(p.label);
    loss += sampleLoss(output, target);
    if ((output >= 0 ? 1 : 0) === p.label) hits += 1;
  }
  return { loss: loss / points.length, accuracy: hits / points.length };
}

/* ---------------- introspection ---------------- */

/** Largest absolute weight in the network, used to scale the connection widths. */
export function maxAbsWeight(layers: readonly Layer[]): number {
  let max = 1e-6;
  for (const layer of layers) {
    for (const neuron of layer) {
      for (const w of neuron.weights) {
        const a = Math.abs(w);
        if (a > max) max = a;
      }
    }
  }
  return max;
}

/** Total parameter count, for the "capacity" metric. */
export function parameterCount(layers: readonly Layer[]): number {
  let total = 0;
  for (const layer of layers) {
    for (const neuron of layer) total += neuron.weights.length + 1;
  }
  return total;
}

/** Split a dataset deterministically into train and test halves. */
export function splitDataset(points: readonly Point2D[], testFraction: number, seed: number): TrainingData {
  const order = shuffled(makeRng(seed + 4211), points.map((_, i) => i));
  const testCount = Math.round(points.length * Math.min(0.8, Math.max(0, testFraction)));
  const test: Point2D[] = [];
  const train: Point2D[] = [];
  order.forEach((index, rank) => {
    if (rank < testCount) test.push(points[index]);
    else train.push(points[index]);
  });
  return { train, test };
}
