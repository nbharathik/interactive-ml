/** A tiny transformer: token and position embeddings, multi-head self-attention, a feed-forward block and layer norm, with an explicit backward pass. */

import type { SequenceDataset } from '../datasets/sequences';
import { gauss, makeRng, shuffled } from '../math/rng';
import { createOptState, optStep } from './optim';
import type { OptState, OptimiserConfig } from './optim';

export type PositionMode = 'sinusoidal' | 'learned' | 'none';

export interface TransformerSpec {
  vocab: number;
  outVocab: number;
  /** The longest sequence the model reads. */
  length: number;
  /** d, the width of every token's vector. */
  width: number;
  heads: number;
  layers: number;
  /** Hidden units of each feed-forward block; 0 leaves the block out. */
  ffn: number;
  positions: PositionMode;
  /** Each position sees only itself and the positions before it. */
  causal: boolean;
  /** Layer norm before attention, before the feed-forward block and before the output. */
  norm: boolean;
  seed: number;
  optimiser: OptimiserConfig;
  learningRate: number;
  batchSize: number;
  /** Evaluate the train and test sets every this many steps. */
  evalEvery: number;
}

export interface LayerWeights {
  ln1g: Float64Array;
  ln1b: Float64Array;
  /** Every d × d matrix is stored row by input: W[i * d + o]. */
  Wq: Float64Array;
  Wk: Float64Array;
  Wv: Float64Array;
  Wo: Float64Array;
  bo: Float64Array;
  ln2g: Float64Array;
  ln2b: Float64Array;
  /** d × f and f × d; empty without a feed-forward block. */
  W1: Float64Array;
  b1: Float64Array;
  W2: Float64Array;
  b2: Float64Array;
}

export interface TransformerWeights {
  /** vocab × d. */
  E: Float64Array;
  /** length × d, learned positions only. */
  P: Float64Array;
  layers: LayerWeights[];
  lnfg: Float64Array;
  lnfb: Float64Array;
  /** d × outVocab. */
  U: Float64Array;
  bu: Float64Array;
}

/* ---------------- weights ---------------- */

export function headWidth(spec: TransformerSpec): number {
  return Math.floor(spec.width / spec.heads);
}

export function createTransformer(spec: TransformerSpec): TransformerWeights {
  const rng = makeRng(spec.seed);
  const d = spec.width;
  const f = spec.ffn;
  const fill = (n: number, sd: number) => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = gauss(rng, 0, sd);
    return out;
  };
  const ones = (n: number) => new Float64Array(n).fill(1);
  // Small starting token vectors: training, not the random start, decides where they end up.
  const E = fill(spec.vocab * d, 0.3);
  const P = spec.positions === 'learned' ? fill(spec.length * d, 0.3) : new Float64Array(0);
  const layers: LayerWeights[] = [];
  for (let l = 0; l < spec.layers; l++) {
    layers.push({
      ln1g: ones(d),
      ln1b: new Float64Array(d),
      Wq: fill(d * d, 1 / Math.sqrt(d)),
      Wk: fill(d * d, 1 / Math.sqrt(d)),
      Wv: fill(d * d, 1 / Math.sqrt(d)),
      Wo: fill(d * d, 1 / Math.sqrt(d)),
      bo: new Float64Array(d),
      ln2g: ones(f > 0 ? d : 0),
      ln2b: new Float64Array(f > 0 ? d : 0),
      W1: fill(d * f, Math.sqrt(2 / d)),
      b1: new Float64Array(f),
      W2: fill(f * d, 1 / Math.sqrt(Math.max(1, f))),
      b2: new Float64Array(f > 0 ? d : 0),
    });
  }
  return { E, P, layers, lnfg: ones(d), lnfb: new Float64Array(d), U: fill(d * spec.outVocab, 1 / Math.sqrt(d)), bu: new Float64Array(spec.outVocab) };
}

const LAYER_KEYS = ['ln1g', 'ln1b', 'Wq', 'Wk', 'Wv', 'Wo', 'bo', 'ln2g', 'ln2b', 'W1', 'b1', 'W2', 'b2'] as const;

/** Every tensor in one fixed order, for the optimiser. */
export function tensorsOf(weights: TransformerWeights): Float64Array[] {
  const list = [weights.E, weights.P];
  for (const layer of weights.layers) for (const key of LAYER_KEYS) list.push(layer[key]);
  list.push(weights.lnfg, weights.lnfb, weights.U, weights.bu);
  return list;
}

export function withTensors(weights: TransformerWeights, tensors: Float64Array[]): TransformerWeights {
  let at = 0;
  const E = tensors[at++];
  const P = tensors[at++];
  const layers = weights.layers.map(() => {
    const layer = {} as LayerWeights;
    for (const key of LAYER_KEYS) layer[key] = tensors[at++];
    return layer;
  });
  return { E, P, layers, lnfg: tensors[at++], lnfb: tensors[at++], U: tensors[at++], bu: tensors[at++] };
}

export function zerosLike(weights: TransformerWeights): TransformerWeights {
  return withTensors(weights, tensorsOf(weights).map((t) => new Float64Array(t.length)));
}

export function parameterCount(weights: TransformerWeights): number {
  return tensorsOf(weights).reduce((n, t) => n + t.length, 0);
}

/** The position vectors added to the tokens: fixed waves, the learned table, or zeros. */
export function positionTable(weights: TransformerWeights, spec: TransformerSpec, length = spec.length): Float64Array {
  const d = spec.width;
  const out = new Float64Array(length * d);
  if (spec.positions === 'learned') {
    out.set(weights.P.subarray(0, length * d));
  } else if (spec.positions === 'sinusoidal') {
    for (let t = 0; t < length; t++) {
      for (let i = 0; i < d; i += 2) {
        const angle = t / Math.pow(10000, i / d);
        out[t * d + i] = Math.sin(angle);
        if (i + 1 < d) out[t * d + i + 1] = Math.cos(angle);
      }
    }
  }
  return out;
}

/* ---------------- building blocks ---------------- */

/** out[t, o] = b[o] + sum_i x[t, i] W[i, o]. */
function linear(x: Float64Array, W: Float64Array, b: Float64Array | null, rows: number, inSize: number, outSize: number): Float64Array {
  const out = new Float64Array(rows * outSize);
  for (let t = 0; t < rows; t++) {
    const o0 = t * outSize;
    if (b) for (let o = 0; o < outSize; o++) out[o0 + o] = b[o];
    for (let i = 0; i < inSize; i++) {
      const v = x[t * inSize + i];
      if (v === 0) continue;
      const w0 = i * outSize;
      for (let o = 0; o < outSize; o++) out[o0 + o] += v * W[w0 + o];
    }
  }
  return out;
}

/** Accumulates dW and db, returns dx. */
function linearBack(x: Float64Array, W: Float64Array, dOut: Float64Array, rows: number, inSize: number, outSize: number, dW: Float64Array, db: Float64Array | null): Float64Array {
  const dx = new Float64Array(rows * inSize);
  for (let t = 0; t < rows; t++) {
    const o0 = t * outSize;
    if (db) for (let o = 0; o < outSize; o++) db[o] += dOut[o0 + o];
    for (let i = 0; i < inSize; i++) {
      const v = x[t * inSize + i];
      const w0 = i * outSize;
      let sum = 0;
      for (let o = 0; o < outSize; o++) {
        const g = dOut[o0 + o];
        sum += g * W[w0 + o];
        dW[w0 + o] += v * g;
      }
      dx[t * inSize + i] = sum;
    }
  }
  return dx;
}

const EPS = 1e-5;

export interface NormCache {
  /** Each row centred and scaled to unit variance, before the gain. */
  n: Float64Array;
  rstd: Float64Array;
  mean: Float64Array;
  y: Float64Array;
}

function normForward(x: Float64Array, g: Float64Array, b: Float64Array, rows: number, d: number): NormCache {
  const n = new Float64Array(rows * d);
  const y = new Float64Array(rows * d);
  const rstd = new Float64Array(rows);
  const mean = new Float64Array(rows);
  for (let t = 0; t < rows; t++) {
    let mu = 0;
    for (let i = 0; i < d; i++) mu += x[t * d + i];
    mu /= d;
    let v = 0;
    for (let i = 0; i < d; i++) v += (x[t * d + i] - mu) ** 2;
    const r = 1 / Math.sqrt(v / d + EPS);
    mean[t] = mu;
    rstd[t] = r;
    for (let i = 0; i < d; i++) {
      const z = (x[t * d + i] - mu) * r;
      n[t * d + i] = z;
      y[t * d + i] = g[i] * z + b[i];
    }
  }
  return { n, rstd, mean, y };
}

function normBack(dy: Float64Array, cache: NormCache, g: Float64Array, rows: number, d: number, dg: Float64Array, db: Float64Array): Float64Array {
  const dx = new Float64Array(rows * d);
  for (let t = 0; t < rows; t++) {
    let s1 = 0;
    let s2 = 0;
    for (let i = 0; i < d; i++) {
      const k = t * d + i;
      dg[i] += dy[k] * cache.n[k];
      db[i] += dy[k];
      const dn = dy[k] * g[i];
      s1 += dn;
      s2 += dn * cache.n[k];
    }
    const r = cache.rstd[t];
    for (let i = 0; i < d; i++) {
      const k = t * d + i;
      dx[k] = r * (dy[k] * g[i] - s1 / d - (cache.n[k] * s2) / d);
    }
  }
  return dx;
}

/** Without layer norm the block reads its input as it is. */
function identityNorm(x: Float64Array, rows: number): NormCache {
  return { n: x, rstd: new Float64Array(rows).fill(1), mean: new Float64Array(rows), y: x };
}

/* ---------------- forward ---------------- */

export interface LayerCache {
  /** The residual stream entering the block, T × d. */
  x: Float64Array;
  norm1: NormCache;
  q: Float64Array;
  k: Float64Array;
  v: Float64Array;
  /** heads × T × T, scaled; masked entries are -Infinity. */
  scores: Float64Array;
  /** heads × T × T, each row sums to 1. */
  attn: Float64Array;
  /** The heads' outputs side by side, T × d. */
  z: Float64Array;
  /** z Wo + bo, what attention adds to the stream. */
  att: Float64Array;
  /** x + att. */
  mid: Float64Array;
  norm2: NormCache | null;
  /** T × f before and after ReLU. */
  pre: Float64Array | null;
  act: Float64Array | null;
  /** What the feed-forward block adds, T × d. */
  ff: Float64Array | null;
  /** The stream leaving the block. */
  out: Float64Array;
}

export interface TransformerCache {
  tokens: number[];
  length: number;
  /** The token vectors and the position vectors, T × d each, and their sum. */
  embed: Float64Array;
  pos: Float64Array;
  x0: Float64Array;
  layers: LayerCache[];
  normF: NormCache;
  /** T × outVocab. */
  logits: Float64Array;
  probs: Float64Array;
  predicted: number[];
}

export function forward(weights: TransformerWeights, tokens: readonly number[], spec: TransformerSpec): TransformerCache {
  const T = Math.min(tokens.length, spec.length);
  const d = spec.width;
  const H = spec.heads;
  const dh = headWidth(spec);
  const scale = 1 / Math.sqrt(dh);
  const embed = new Float64Array(T * d);
  for (let t = 0; t < T; t++) embed.set(weights.E.subarray(tokens[t] * d, tokens[t] * d + d), t * d);
  const pos = positionTable(weights, spec, T);
  const x0 = new Float64Array(T * d);
  for (let i = 0; i < x0.length; i++) x0[i] = embed[i] + pos[i];

  const layers: LayerCache[] = [];
  let x = x0;
  for (const layer of weights.layers) {
    const norm1 = spec.norm ? normForward(x, layer.ln1g, layer.ln1b, T, d) : identityNorm(x, T);
    const h = norm1.y;
    const q = linear(h, layer.Wq, null, T, d, d);
    const k = linear(h, layer.Wk, null, T, d, d);
    const v = linear(h, layer.Wv, null, T, d, d);
    const scores = new Float64Array(H * T * T);
    const attn = new Float64Array(H * T * T);
    const z = new Float64Array(T * d);
    for (let hd = 0; hd < H; hd++) {
      const c0 = hd * dh;
      for (let i = 0; i < T; i++) {
        const row = (hd * T + i) * T;
        let max = -Infinity;
        for (let j = 0; j < T; j++) {
          if (spec.causal && j > i) {
            scores[row + j] = -Infinity;
            continue;
          }
          let s = 0;
          for (let c = 0; c < dh; c++) s += q[i * d + c0 + c] * k[j * d + c0 + c];
          s *= scale;
          scores[row + j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j < T; j++) {
          const e = scores[row + j] === -Infinity ? 0 : Math.exp(scores[row + j] - max);
          attn[row + j] = e;
          sum += e;
        }
        for (let j = 0; j < T; j++) {
          const a = attn[row + j] / sum;
          attn[row + j] = a;
          if (a === 0) continue;
          for (let c = 0; c < dh; c++) z[i * d + c0 + c] += a * v[j * d + c0 + c];
        }
      }
    }
    const att = linear(z, layer.Wo, layer.bo, T, d, d);
    const mid = new Float64Array(T * d);
    for (let i = 0; i < mid.length; i++) mid[i] = x[i] + att[i];
    let norm2: NormCache | null = null;
    let pre: Float64Array | null = null;
    let act: Float64Array | null = null;
    let ff: Float64Array | null = null;
    let out = mid;
    if (spec.ffn > 0) {
      norm2 = spec.norm ? normForward(mid, layer.ln2g, layer.ln2b, T, d) : identityNorm(mid, T);
      pre = linear(norm2.y, layer.W1, layer.b1, T, d, spec.ffn);
      act = new Float64Array(pre.length);
      for (let i = 0; i < pre.length; i++) act[i] = pre[i] > 0 ? pre[i] : 0;
      ff = linear(act, layer.W2, layer.b2, T, spec.ffn, d);
      out = new Float64Array(T * d);
      for (let i = 0; i < out.length; i++) out[i] = mid[i] + ff[i];
    }
    layers.push({ x, norm1, q, k, v, scores, attn, z, att, mid, norm2, pre, act, ff, out });
    x = out;
  }

  const normF = spec.norm ? normForward(x, weights.lnfg, weights.lnfb, T, d) : identityNorm(x, T);
  const V = spec.outVocab;
  const logits = linear(normF.y, weights.U, weights.bu, T, d, V);
  const probs = new Float64Array(T * V);
  const predicted: number[] = [];
  for (let t = 0; t < T; t++) {
    let max = -Infinity;
    for (let c = 0; c < V; c++) max = Math.max(max, logits[t * V + c]);
    let sum = 0;
    for (let c = 0; c < V; c++) {
      const e = Math.exp(logits[t * V + c] - max);
      probs[t * V + c] = e;
      sum += e;
    }
    let best = 0;
    for (let c = 0; c < V; c++) {
      probs[t * V + c] /= sum;
      if (probs[t * V + c] > probs[t * V + best]) best = c;
    }
    predicted.push(best);
  }
  return { tokens: tokens.slice(0, T), length: T, embed, pos, x0, layers, normF, logits, probs, predicted };
}

/** Summed cross-entropy over the positions that have a target, and how many there were. */
export function sequenceLoss(cache: TransformerCache, targets: readonly number[], outVocab: number): { loss: number; count: number } {
  let loss = 0;
  let count = 0;
  for (let t = 0; t < cache.length; t++) {
    const y = targets[t];
    if (y === undefined || y < 0) continue;
    loss -= Math.log(Math.max(1e-12, cache.probs[t * outVocab + y]));
    count += 1;
  }
  return { loss, count };
}

/* ---------------- backward ---------------- */

/** Adds this sequence's gradient, scaled by `weight`, into `grads`. */
export function backward(weights: TransformerWeights, cache: TransformerCache, targets: readonly number[], spec: TransformerSpec, grads: TransformerWeights, weight: number): void {
  const T = cache.length;
  const d = spec.width;
  const H = spec.heads;
  const dh = headWidth(spec);
  const scale = 1 / Math.sqrt(dh);
  const V = spec.outVocab;

  const dLogits = new Float64Array(T * V);
  for (let t = 0; t < T; t++) {
    const y = targets[t];
    if (y === undefined || y < 0) continue;
    for (let c = 0; c < V; c++) dLogits[t * V + c] = weight * (cache.probs[t * V + c] - (c === y ? 1 : 0));
  }
  const dNormF = linearBack(cache.normF.y, weights.U, dLogits, T, d, V, grads.U, grads.bu);
  let dx = spec.norm ? normBack(dNormF, cache.normF, weights.lnfg, T, d, grads.lnfg, grads.lnfb) : dNormF;

  for (let l = weights.layers.length - 1; l >= 0; l--) {
    const layer = weights.layers[l];
    const g = grads.layers[l];
    const c = cache.layers[l];
    let dMid = dx;
    if (spec.ffn > 0 && c.norm2 && c.pre && c.act) {
      const dAct = linearBack(c.act, layer.W2, dx, T, spec.ffn, d, g.W2, g.b2);
      for (let i = 0; i < dAct.length; i++) if (c.pre[i] <= 0) dAct[i] = 0;
      const dN2 = linearBack(c.norm2.y, layer.W1, dAct, T, d, spec.ffn, g.W1, g.b1);
      const dFromFf = spec.norm ? normBack(dN2, c.norm2, layer.ln2g, T, d, g.ln2g, g.ln2b) : dN2;
      dMid = new Float64Array(T * d);
      for (let i = 0; i < dMid.length; i++) dMid[i] = dx[i] + dFromFf[i];
    }
    const dz = linearBack(c.z, layer.Wo, dMid, T, d, d, g.Wo, g.bo);
    const dq = new Float64Array(T * d);
    const dk = new Float64Array(T * d);
    const dv = new Float64Array(T * d);
    const dA = new Float64Array(T);
    for (let hd = 0; hd < H; hd++) {
      const c0 = hd * dh;
      for (let i = 0; i < T; i++) {
        const row = (hd * T + i) * T;
        let dot = 0;
        for (let j = 0; j < T; j++) {
          const a = c.attn[row + j];
          let s = 0;
          if (a !== 0) {
            for (let e = 0; e < dh; e++) {
              const gz = dz[i * d + c0 + e];
              s += gz * c.v[j * d + c0 + e];
              dv[j * d + c0 + e] += a * gz;
            }
          }
          dA[j] = s;
          dot += a * s;
        }
        for (let j = 0; j < T; j++) {
          const a = c.attn[row + j];
          if (a === 0) continue;
          const dS = a * (dA[j] - dot) * scale;
          for (let e = 0; e < dh; e++) {
            dq[i * d + c0 + e] += dS * c.k[j * d + c0 + e];
            dk[j * d + c0 + e] += dS * c.q[i * d + c0 + e];
          }
        }
      }
    }
    const h = c.norm1.y;
    const dh1 = linearBack(h, layer.Wq, dq, T, d, d, g.Wq, null);
    const dh2 = linearBack(h, layer.Wk, dk, T, d, d, g.Wk, null);
    const dh3 = linearBack(h, layer.Wv, dv, T, d, d, g.Wv, null);
    for (let i = 0; i < dh1.length; i++) dh1[i] += dh2[i] + dh3[i];
    const dFromAtt = spec.norm ? normBack(dh1, c.norm1, layer.ln1g, T, d, g.ln1g, g.ln1b) : dh1;
    const next = new Float64Array(T * d);
    for (let i = 0; i < next.length; i++) next[i] = dMid[i] + dFromAtt[i];
    dx = next;
  }

  for (let t = 0; t < T; t++) {
    const tok = cache.tokens[t];
    for (let i = 0; i < d; i++) {
      grads.E[tok * d + i] += dx[t * d + i];
      if (spec.positions === 'learned') grads.P[t * d + i] += dx[t * d + i];
    }
  }
}

/* ---------------- evaluation ---------------- */

export interface Evaluation {
  loss: number;
  /** Share of scored positions called right. */
  accuracy: number;
  /** Share of sequences with every scored position right; for a next-word task, sentences written in full from their prompt. */
  exact: number;
  /** Per sequence, the call at each scored position. */
  predictions: number[][];
}

/** The call at a scored position: next-word tasks read only the prefix, as they would when writing; under a causal mask the full pass already does. */
export function callAt(weights: TransformerWeights, data: SequenceDataset, tokens: readonly number[], position: number, spec: TransformerSpec, full?: TransformerCache): { call: number; probs: Float64Array } {
  const V = spec.outVocab;
  const cache = data.nextToken && !(spec.causal && full) ? forward(weights, tokens.slice(0, position + 1), spec) : full ?? forward(weights, tokens, spec);
  return { call: cache.predicted[position], probs: cache.probs.subarray(position * V, position * V + V) };
}

/**
 * Whether the model, given a sentence's prompt, writes a sentence of the language: any sentence the data holds.
 * With the language's prefixes it stops at the first token no sentence continues with.
 */
export function writesSentence(weights: TransformerWeights, data: SequenceDataset, index: number, spec: TransformerSpec, language: ReadonlySet<string>, prefixes?: ReadonlySet<string>): boolean {
  const tokens = data.inputs[index];
  const stop = tokens[tokens.length - 1];
  const written = tokens.slice(0, Math.min(data.prompt[index], spec.length));
  while (written.length < spec.length) {
    const next = forward(weights, written, spec).predicted[written.length - 1];
    written.push(next);
    if (next === stop) break;
    if (prefixes && !prefixes.has(written.join(','))) return false;
  }
  return language.has(written.join(','));
}

/** Every start of every sentence, so writing can be cut short once it leaves the language. */
function prefixesOf(inputs: readonly number[][]): Set<string> {
  const out = new Set<string>();
  for (const tokens of inputs) for (let k = 1; k <= tokens.length; k++) out.add(tokens.slice(0, k).join(','));
  return out;
}

/** Loss, accuracy and the calls; `write` also has a next-word model write each sentence out. */
export function evaluate(weights: TransformerWeights, data: SequenceDataset, indices: readonly number[], spec: TransformerSpec, write = false): Evaluation {
  if (indices.length === 0) return { loss: 0, accuracy: 0, exact: 0, predictions: [] };
  const language = data.nextToken && write ? new Set(data.inputs.map((t) => t.join(','))) : null;
  const prefixes = language ? prefixesOf(data.inputs) : undefined;
  let loss = 0;
  let count = 0;
  let right = 0;
  let scored = 0;
  let exact = 0;
  const predictions: number[][] = [];
  for (const index of indices) {
    const tokens = data.inputs[index];
    const cache = forward(weights, tokens, spec);
    const l = sequenceLoss(cache, data.targets[index], spec.outVocab);
    loss += l.loss;
    count += l.count;
    const calls: number[] = [];
    let all = true;
    for (const position of data.scored[index]) {
      const { call } = callAt(weights, data, tokens, position, spec, cache);
      calls.push(call);
      scored += 1;
      if (call === data.targets[index][position]) right += 1;
      else all = false;
    }
    if (data.nextToken ? language !== null && writesSentence(weights, data, index, spec, language, prefixes) : all) exact += 1;
    predictions.push(calls);
  }
  return { loss: loss / Math.max(1, count), accuracy: right / Math.max(1, scored), exact: exact / indices.length, predictions };
}

/* ---------------- training ---------------- */

export interface TransformerState {
  weights: TransformerWeights;
  opt: OptState[];
  step: number;
  epoch: number;
  cursor: number;
  order: number[];
  lastBatch: number[];
  batchLoss: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss: number;
  testAccuracy: number;
  testExact: number;
  testPredictions: number[][];
  history: Array<{ step: number; train: number; test: number; trainAcc: number; testAcc: number }>;
  diverged: boolean;
}

function orderFor(seed: number, epoch: number, count: number): number[] {
  return shuffled(makeRng(seed + epoch * 7919), Array.from({ length: count }, (_, i) => i));
}

/** The training score reads this many training sequences, so an evaluation stays cheap. */
export const TRAIN_EVAL = 160;

function trainSample(data: SequenceDataset): number[] {
  return data.trainIndex.slice(0, TRAIN_EVAL);
}

export function createState(spec: TransformerSpec, data: SequenceDataset, weights = createTransformer(spec)): TransformerState {
  const train = evaluate(weights, data, trainSample(data), spec);
  const test = evaluate(weights, data, data.testIndex, spec, true);
  return {
    weights,
    opt: tensorsOf(weights).map((t) => createOptState(spec.optimiser, t.length)),
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
    testExact: test.exact,
    testPredictions: test.predictions,
    history: [{ step: 0, train: train.loss, test: test.loss, trainAcc: train.accuracy, testAcc: test.accuracy }],
    diverged: false,
  };
}

/** One mini-batch update. Pure. */
export function step(state: TransformerState, data: SequenceDataset, spec: TransformerSpec): TransformerState {
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

  let targetCount = 0;
  for (const index of batch) for (const y of data.targets[index]) if (y >= 0) targetCount += 1;
  const weight = 1 / Math.max(1, targetCount);
  const grads = zerosLike(state.weights);
  let lossSum = 0;
  for (const index of batch) {
    const cache = forward(state.weights, data.inputs[index], spec);
    lossSum += sequenceLoss(cache, data.targets[index], spec.outVocab).loss;
    backward(state.weights, cache, data.targets[index], spec, grads, weight);
  }

  const tensors = tensorsOf(state.weights);
  const gradList = tensorsOf(grads);
  const nextOpt: OptState[] = [];
  const nextTensors = tensors.map((t, k) => {
    if (t.length === 0) {
      nextOpt.push(state.opt[k]);
      return t;
    }
    const { state: opt, delta } = optStep(state.opt[k], gradList[k], spec.learningRate, spec.optimiser);
    nextOpt.push(opt);
    const out = new Float64Array(t.length);
    for (let i = 0; i < t.length; i++) out[i] = t[i] + delta[i];
    return out;
  });
  const weights = withTensors(state.weights, nextTensors);
  const diverged = nextTensors.some((t) => t.some((v) => !Number.isFinite(v) || Math.abs(v) > 1e6));
  const stepCount = state.step + 1;
  const evaluateNow = !diverged && stepCount % Math.max(1, spec.evalEvery) === 0;

  let { trainLoss, trainAccuracy, testLoss, testAccuracy, testExact, testPredictions, history } = state;
  if (evaluateNow) {
    const train = evaluate(weights, data, trainSample(data), spec);
    const test = evaluate(weights, data, data.testIndex, spec, true);
    trainLoss = train.loss;
    trainAccuracy = train.accuracy;
    testLoss = test.loss;
    testAccuracy = test.accuracy;
    testExact = test.exact;
    testPredictions = test.predictions;
    history = history.concat({ step: stepCount, train: train.loss, test: test.loss, trainAcc: train.accuracy, testAcc: test.accuracy });
    if (history.length > 2000) history = history.slice(-2000);
  }
  return {
    weights,
    opt: nextOpt,
    step: stepCount,
    epoch,
    cursor,
    order,
    lastBatch: batch,
    batchLoss: lossSum * weight,
    trainLoss,
    trainAccuracy,
    testLoss,
    testAccuracy,
    testExact,
    testPredictions,
    history,
    diverged,
  };
}

/** Greedy continuation of a next-word model: append the likeliest token until `stop` or the model's length. */
export function generate(weights: TransformerWeights, prompt: readonly number[], spec: TransformerSpec, stop: number): number[] {
  const tokens = prompt.slice(0, spec.length);
  while (tokens.length < spec.length) {
    const cache = forward(weights, tokens, spec);
    const next = cache.predicted[tokens.length - 1];
    tokens.push(next);
    if (next === stop) break;
  }
  return tokens;
}
