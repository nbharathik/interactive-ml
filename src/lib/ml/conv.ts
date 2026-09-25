/** A tiny convolutional network: conv, ReLU, pool, dense, softmax, with an explicit backward pass. */

import type { ImageDataset } from '../datasets/images';
import { makeRng, gauss, shuffled } from '../math/rng';
import { logSumExp, softmax } from './losses';
import { createOptState, optStep } from './optim';
import type { OptState, OptimiserConfig } from './optim';

export interface Tensor {
  c: number;
  h: number;
  w: number;
  /** Channel-major, then rows, then columns. */
  data: Float64Array;
}

export interface ConvLayer {
  kernel: number;
  inC: number;
  outC: number;
  pad: number;
  /** W[((o * inC + i) * kernel + ky) * kernel + kx]. */
  W: Float64Array;
  b: Float64Array;
}

export interface DenseLayer {
  inSize: number;
  outSize: number;
  /** W[o * inSize + i]. */
  W: Float64Array;
  b: Float64Array;
}

export type PoolMode = 'max' | 'avg' | 'none';

export interface CnnSpec {
  size: number;
  /** A plain dense network on the raw pixels, for comparison. */
  model: 'cnn' | 'dense';
  kernel: number;
  filters: number;
  padding: 'same' | 'valid';
  pool: PoolMode;
  secondConv: boolean;
  secondFilters: number;
  /** Hidden dense units before the output; 0 goes straight to the logits. */
  dense: number;
  classCount: number;
  seed: number;
  optimiser: OptimiserConfig;
  learningRate: number;
  batchSize: number;
  /** Evaluate the train and test sets every this many steps. */
  evalEvery: number;
}

export interface CnnParams {
  convs: ConvLayer[];
  hidden: DenseLayer | null;
  out: DenseLayer;
}

export interface StageCache {
  input: Tensor;
  z: Tensor;
  a: Tensor;
  pooled: Tensor;
  argmax: Int32Array | null;
}

export interface CnnCache {
  input: Tensor;
  stages: StageCache[];
  flat: Float64Array;
  hiddenZ: Float64Array | null;
  hiddenA: Float64Array | null;
  logits: Float64Array;
  probs: Float64Array;
  predicted: number;
}

export interface CnnGrads {
  convs: Array<{ dW: Float64Array; db: Float64Array }>;
  hidden: { dW: Float64Array; db: Float64Array } | null;
  out: { dW: Float64Array; db: Float64Array };
}

export function tensor(c: number, h: number, w: number, data?: Float64Array): Tensor {
  return { c, h, w, data: data ?? new Float64Array(c * h * w) };
}

/** One channel of a tensor, without copying. */
export function channel(t: Tensor, index: number): Float64Array {
  return t.data.subarray(index * t.h * t.w, (index + 1) * t.h * t.w);
}

/* ---------------- shapes ---------------- */

export interface StageShape {
  conv: [number, number, number];
  pooled: [number, number, number];
}

export function stageShapes(spec: CnnSpec): { stages: StageShape[]; flat: number } {
  if (spec.model === 'dense') return { stages: [], flat: spec.size * spec.size };
  const stages: StageShape[] = [];
  let c = 1;
  let h = spec.size;
  let w = spec.size;
  const count = spec.secondConv ? 2 : 1;
  for (let s = 0; s < count; s++) {
    const outC = s === 0 ? spec.filters : spec.secondFilters;
    const pad = spec.padding === 'same' ? Math.floor(spec.kernel / 2) : 0;
    const ch = h + 2 * pad - spec.kernel + 1;
    const cw = w + 2 * pad - spec.kernel + 1;
    const ph = spec.pool === 'none' ? ch : Math.floor(ch / 2);
    const pw = spec.pool === 'none' ? cw : Math.floor(cw / 2);
    stages.push({ conv: [outC, ch, cw], pooled: [outC, ph, pw] });
    c = outC;
    h = ph;
    w = pw;
  }
  return { stages, flat: c * h * w };
}

/* ---------------- parameters ---------------- */

export function createCnn(spec: CnnSpec): CnnParams {
  const rng = makeRng(spec.seed);
  const he = (fanIn: number) => gauss(rng, 0, Math.sqrt(2 / fanIn));
  const shapes = stageShapes(spec);
  const convs: ConvLayer[] = [];
  let inC = 1;
  shapes.stages.forEach((stage) => {
    const outC = stage.conv[0];
    const pad = spec.padding === 'same' ? Math.floor(spec.kernel / 2) : 0;
    const W = new Float64Array(outC * inC * spec.kernel * spec.kernel);
    for (let i = 0; i < W.length; i++) W[i] = he(inC * spec.kernel * spec.kernel);
    convs.push({ kernel: spec.kernel, inC, outC, pad, W, b: new Float64Array(outC) });
    inC = outC;
  });
  const dense = (inSize: number, outSize: number): DenseLayer => {
    const W = new Float64Array(inSize * outSize);
    for (let i = 0; i < W.length; i++) W[i] = he(inSize);
    return { inSize, outSize, W, b: new Float64Array(outSize) };
  };
  const hidden = spec.dense > 0 ? dense(shapes.flat, spec.dense) : null;
  const out = dense(hidden ? hidden.outSize : shapes.flat, Math.max(2, spec.classCount));
  return { convs, hidden, out };
}

export function parameterCount(params: CnnParams): number {
  let n = params.out.W.length + params.out.b.length;
  if (params.hidden) n += params.hidden.W.length + params.hidden.b.length;
  for (const c of params.convs) n += c.W.length + c.b.length;
  return n;
}

/* ---------------- layers ---------------- */

export function convForward(x: Tensor, layer: ConvLayer): Tensor {
  const { kernel: k, pad, inC, outC } = layer;
  const oh = x.h + 2 * pad - k + 1;
  const ow = x.w + 2 * pad - k + 1;
  const out = tensor(outC, oh, ow);
  for (let o = 0; o < outC; o++) {
    const base = o * oh * ow;
    for (let y = 0; y < oh; y++) {
      for (let xx = 0; xx < ow; xx++) {
        let sum = layer.b[o];
        for (let i = 0; i < inC; i++) {
          const inBase = i * x.h * x.w;
          const wBase = (o * inC + i) * k * k;
          for (let ky = 0; ky < k; ky++) {
            const yy = y + ky - pad;
            if (yy < 0 || yy >= x.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const xs = xx + kx - pad;
              if (xs < 0 || xs >= x.w) continue;
              sum += layer.W[wBase + ky * k + kx] * x.data[inBase + yy * x.w + xs];
            }
          }
        }
        out.data[base + y * ow + xx] = sum;
      }
    }
  }
  return out;
}

/** Gradients of a convolution: the kernel correlates input with dZ, the input gets dZ scattered through the kernel. */
export function convBackward(x: Tensor, layer: ConvLayer, dZ: Tensor): { dX: Tensor; dW: Float64Array; db: Float64Array } {
  const { kernel: k, pad, inC, outC } = layer;
  const dX = tensor(x.c, x.h, x.w);
  const dW = new Float64Array(layer.W.length);
  const db = new Float64Array(outC);
  for (let o = 0; o < outC; o++) {
    const base = o * dZ.h * dZ.w;
    for (let y = 0; y < dZ.h; y++) {
      for (let xx = 0; xx < dZ.w; xx++) {
        const g = dZ.data[base + y * dZ.w + xx];
        if (g === 0) continue;
        db[o] += g;
        for (let i = 0; i < inC; i++) {
          const inBase = i * x.h * x.w;
          const wBase = (o * inC + i) * k * k;
          for (let ky = 0; ky < k; ky++) {
            const yy = y + ky - pad;
            if (yy < 0 || yy >= x.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const xs = xx + kx - pad;
              if (xs < 0 || xs >= x.w) continue;
              dW[wBase + ky * k + kx] += g * x.data[inBase + yy * x.w + xs];
              dX.data[inBase + yy * x.w + xs] += g * layer.W[wBase + ky * k + kx];
            }
          }
        }
      }
    }
  }
  return { dX, dW, db };
}

export function reluForward(z: Tensor): Tensor {
  const out = tensor(z.c, z.h, z.w);
  for (let i = 0; i < z.data.length; i++) out.data[i] = z.data[i] > 0 ? z.data[i] : 0;
  return out;
}

export function reluBackward(dA: Tensor, z: Tensor): Tensor {
  const out = tensor(z.c, z.h, z.w);
  for (let i = 0; i < z.data.length; i++) out.data[i] = z.data[i] > 0 ? dA.data[i] : 0;
  return out;
}

export function poolForward(a: Tensor, mode: PoolMode): { out: Tensor; argmax: Int32Array | null } {
  if (mode === 'none') return { out: a, argmax: null };
  const oh = Math.floor(a.h / 2);
  const ow = Math.floor(a.w / 2);
  const out = tensor(a.c, oh, ow);
  const argmax = mode === 'max' ? new Int32Array(a.c * oh * ow) : null;
  for (let c = 0; c < a.c; c++) {
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        let best = -Infinity;
        let bestAt = 0;
        let sum = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const at = c * a.h * a.w + (2 * y + dy) * a.w + (2 * x + dx);
            const v = a.data[at];
            sum += v;
            if (v > best) {
              best = v;
              bestAt = at;
            }
          }
        }
        const o = c * oh * ow + y * ow + x;
        out.data[o] = mode === 'max' ? best : sum / 4;
        if (argmax) argmax[o] = bestAt;
      }
    }
  }
  return { out, argmax };
}

export function poolBackward(dOut: Tensor, a: Tensor, mode: PoolMode, argmax: Int32Array | null): Tensor {
  if (mode === 'none') return dOut;
  const dA = tensor(a.c, a.h, a.w);
  const oh = dOut.h;
  const ow = dOut.w;
  for (let c = 0; c < a.c; c++) {
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const o = c * oh * ow + y * ow + x;
        const g = dOut.data[o];
        if (mode === 'max' && argmax) {
          dA.data[argmax[o]] += g;
        } else {
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) dA.data[c * a.h * a.w + (2 * y + dy) * a.w + (2 * x + dx)] += g / 4;
          }
        }
      }
    }
  }
  return dA;
}

function denseForward(layer: DenseLayer, input: Float64Array): Float64Array {
  const out = new Float64Array(layer.outSize);
  for (let o = 0; o < layer.outSize; o++) {
    let sum = layer.b[o];
    const base = o * layer.inSize;
    for (let i = 0; i < layer.inSize; i++) sum += layer.W[base + i] * input[i];
    out[o] = sum;
  }
  return out;
}

function denseBackward(layer: DenseLayer, input: Float64Array, dOut: Float64Array): { dIn: Float64Array; dW: Float64Array; db: Float64Array } {
  const dIn = new Float64Array(layer.inSize);
  const dW = new Float64Array(layer.W.length);
  for (let o = 0; o < layer.outSize; o++) {
    const g = dOut[o];
    const base = o * layer.inSize;
    for (let i = 0; i < layer.inSize; i++) {
      dW[base + i] = g * input[i];
      dIn[i] += g * layer.W[base + i];
    }
  }
  return { dIn, dW, db: Float64Array.from(dOut) };
}

/* ---------------- the whole net ---------------- */

export function forwardProbe(params: CnnParams, image: Float32Array, spec: CnnSpec): CnnCache {
  const input = tensor(1, spec.size, spec.size, Float64Array.from(image));
  const stages: StageCache[] = [];
  let current = input;
  for (const layer of params.convs) {
    const z = convForward(current, layer);
    const a = reluForward(z);
    const { out, argmax } = poolForward(a, spec.pool);
    stages.push({ input: current, z, a, pooled: out, argmax });
    current = out;
  }
  const flat = current.data;
  let hiddenZ: Float64Array | null = null;
  let hiddenA: Float64Array | null = null;
  let head = flat;
  if (params.hidden) {
    hiddenZ = denseForward(params.hidden, flat);
    hiddenA = hiddenZ.map((v) => (v > 0 ? v : 0));
    head = hiddenA;
  }
  const logits = denseForward(params.out, head);
  const probs = Float64Array.from(softmax(Array.from(logits)));
  let predicted = 0;
  for (let c = 1; c < probs.length; c++) if (probs[c] > probs[predicted]) predicted = c;
  return { input, stages, flat, hiddenZ, hiddenA, logits, probs, predicted };
}

export function sampleLoss(cache: CnnCache, label: number): number {
  return logSumExp(Array.from(cache.logits)) - cache.logits[label];
}

export function backwardProbe(params: CnnParams, cache: CnnCache, label: number, spec: CnnSpec): CnnGrads {
  const dLogits = Float64Array.from(cache.probs);
  dLogits[label] -= 1;
  const head = cache.hiddenA ?? cache.flat;
  const out = denseBackward(params.out, head, dLogits);
  let dFlat = out.dIn;
  let hidden: CnnGrads['hidden'] = null;
  if (params.hidden && cache.hiddenZ) {
    const dHidden = dFlat.map((g, i) => (cache.hiddenZ![i] > 0 ? g : 0));
    const back = denseBackward(params.hidden, cache.flat, dHidden);
    hidden = { dW: back.dW, db: back.db };
    dFlat = back.dIn;
  }
  const convs: CnnGrads['convs'] = new Array(params.convs.length);
  let dPooled: Tensor | null = null;
  for (let s = params.convs.length - 1; s >= 0; s--) {
    const stage = cache.stages[s];
    const dOut = dPooled ?? tensor(stage.pooled.c, stage.pooled.h, stage.pooled.w, dFlat);
    const dA = poolBackward(dOut, stage.a, spec.pool, stage.argmax);
    const dZ = reluBackward(dA, stage.z);
    const back = convBackward(stage.input, params.convs[s], dZ);
    convs[s] = { dW: back.dW, db: back.db };
    dPooled = back.dX;
  }
  return { convs, hidden, out: { dW: out.dW, db: out.db } };
}

export function evaluate(
  params: CnnParams,
  images: readonly Float32Array[],
  labels: readonly number[],
  spec: CnnSpec,
): { loss: number; accuracy: number; predictions: number[] } {
  if (images.length === 0) return { loss: 0, accuracy: 0, predictions: [] };
  let lossSum = 0;
  let correct = 0;
  const predictions: number[] = [];
  images.forEach((image, i) => {
    const cache = forwardProbe(params, image, spec);
    lossSum += sampleLoss(cache, labels[i]);
    if (cache.predicted === labels[i]) correct++;
    predictions.push(cache.predicted);
  });
  return { loss: lossSum / images.length, accuracy: correct / images.length, predictions };
}

/** One kernel as a picture: its input channels averaged. */
export function filterTensor(layer: ConvLayer, index: number): Tensor {
  const k = layer.kernel;
  const out = tensor(1, k, k);
  for (let i = 0; i < layer.inC; i++) {
    const base = (index * layer.inC + i) * k * k;
    for (let p = 0; p < k * k; p++) out.data[p] += layer.W[base + p] / layer.inC;
  }
  return out;
}

/* ---------------- training ---------------- */

export interface CnnState {
  params: CnnParams;
  opt: OptState[];
  step: number;
  epoch: number;
  /** Position in the epoch's shuffled order. */
  cursor: number;
  order: number[];
  lastBatch: number[];
  batchLoss: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss: number;
  testAccuracy: number;
  testPredictions: number[];
  history: Array<{ step: number; train: number; test: number; trainAcc: number; testAcc: number }>;
  gradNorms: number[];
  diverged: boolean;
}

/** Every tensor in one flat list, so the optimiser can walk them: conv W, conv b, ..., hidden W, hidden b, out W, out b. */
function tensorsOf(params: CnnParams): Float64Array[] {
  const list: Float64Array[] = [];
  for (const c of params.convs) list.push(c.W, c.b);
  if (params.hidden) list.push(params.hidden.W, params.hidden.b);
  list.push(params.out.W, params.out.b);
  return list;
}

function gradTensorsOf(grads: CnnGrads): Float64Array[] {
  const list: Float64Array[] = [];
  for (const c of grads.convs) list.push(c.dW, c.db);
  if (grads.hidden) list.push(grads.hidden.dW, grads.hidden.db);
  list.push(grads.out.dW, grads.out.db);
  return list;
}

function withTensors(params: CnnParams, tensors: Float64Array[]): CnnParams {
  let at = 0;
  const convs = params.convs.map((c) => ({ ...c, W: tensors[at++], b: tensors[at++] }));
  const hidden = params.hidden ? { ...params.hidden, W: tensors[at++], b: tensors[at++] } : null;
  const out = { ...params.out, W: tensors[at++], b: tensors[at++] };
  return { convs, hidden, out };
}

function orderFor(seed: number, epoch: number, count: number): number[] {
  return shuffled(makeRng(seed + epoch * 7919), Array.from({ length: count }, (_, i) => i));
}

export function createState(spec: CnnSpec, data: ImageDataset): CnnState {
  const params = createCnn(spec);
  const train = evaluate(params, data.trainIndex.map((i) => data.images[i]), data.trainIndex.map((i) => data.labels[i]), spec);
  const test = evaluate(params, data.testIndex.map((i) => data.images[i]), data.testIndex.map((i) => data.labels[i]), spec);
  return {
    params,
    opt: tensorsOf(params).map((t) => createOptState(spec.optimiser, t.length)),
    step: 0,
    epoch: 0,
    cursor: 0,
    order: orderFor(spec.seed, 0, data.trainIndex.length),
    lastBatch: [],
    batchLoss: train.loss,
    trainLoss: train.loss,
    trainAccuracy: train.accuracy,
    testLoss: test.loss,
    testAccuracy: test.accuracy,
    testPredictions: test.predictions,
    history: [{ step: 0, train: train.loss, test: test.loss, trainAcc: train.accuracy, testAcc: test.accuracy }],
    gradNorms: [],
    diverged: false,
  };
}

/** One mini-batch update. Pure. */
export function step(state: CnnState, data: ImageDataset, spec: CnnSpec): CnnState {
  if (state.diverged || data.trainIndex.length === 0) return state;
  const size = Math.max(1, Math.min(state.order.length, Math.round(spec.batchSize)));
  let cursor = state.cursor;
  let epoch = state.epoch;
  let order = state.order;
  if (cursor >= order.length) {
    cursor = 0;
    epoch += 1;
    order = orderFor(spec.seed, epoch, data.trainIndex.length);
  }
  const batch = order.slice(cursor, cursor + size).map((i) => data.trainIndex[i]);
  cursor += size;

  const tensors = tensorsOf(state.params);
  const sums = tensors.map((t) => new Float64Array(t.length));
  let lossSum = 0;
  for (const index of batch) {
    const cache = forwardProbe(state.params, data.images[index], spec);
    lossSum += sampleLoss(cache, data.labels[index]);
    const grads = gradTensorsOf(backwardProbe(state.params, cache, data.labels[index], spec));
    grads.forEach((g, t) => {
      const sum = sums[t];
      for (let i = 0; i < g.length; i++) sum[i] += g[i];
    });
  }
  const scale = 1 / batch.length;
  const gradNorms: number[] = [];
  const nextOpt: OptState[] = [];
  const nextTensors = tensors.map((t, k) => {
    const g = sums[k];
    let norm = 0;
    for (let i = 0; i < g.length; i++) {
      g[i] *= scale;
      norm += g[i] * g[i];
    }
    gradNorms.push(Math.sqrt(norm));
    const { state: opt, delta } = optStep(state.opt[k], g, spec.learningRate, spec.optimiser);
    nextOpt.push(opt);
    const out = new Float64Array(t.length);
    for (let i = 0; i < t.length; i++) out[i] = t[i] + delta[i];
    return out;
  });
  const params = withTensors(state.params, nextTensors);
  const diverged = nextTensors.some((t) => t.some((v) => !Number.isFinite(v) || Math.abs(v) > 1e6));
  const stepCount = state.step + 1;
  const epochDone = cursor >= order.length;
  const evaluateNow = !diverged && (stepCount % Math.max(1, spec.evalEvery) === 0 || epochDone);

  let { trainLoss, trainAccuracy, testLoss, testAccuracy, testPredictions, history } = state;
  if (evaluateNow) {
    const train = evaluate(params, data.trainIndex.map((i) => data.images[i]), data.trainIndex.map((i) => data.labels[i]), spec);
    const test = evaluate(params, data.testIndex.map((i) => data.images[i]), data.testIndex.map((i) => data.labels[i]), spec);
    trainLoss = train.loss;
    trainAccuracy = train.accuracy;
    testLoss = test.loss;
    testAccuracy = test.accuracy;
    testPredictions = test.predictions;
    history = history.concat({ step: stepCount, train: train.loss, test: test.loss, trainAcc: train.accuracy, testAcc: test.accuracy });
    if (history.length > 2000) history = history.slice(-2000);
  }
  return {
    params,
    opt: nextOpt,
    step: stepCount,
    epoch,
    cursor,
    order,
    lastBatch: batch,
    batchLoss: lossSum / batch.length,
    trainLoss,
    trainAccuracy,
    testLoss,
    testAccuracy,
    testPredictions,
    history,
    gradNorms,
    diverged,
  };
}
