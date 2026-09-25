/** The page's model plumbing: data and spec from the knobs, the probe, and the readings the views and lessons share. */

import { ADJECTIVES, ADVERBS, ANIMALS, generateSequences, labelTokens, parseTokens, promptLength } from '../../lib/datasets/sequences';
import type { SequenceDataset, SequenceTask } from '../../lib/datasets/sequences';
import { createState, createTransformer, forward, generate, headWidth, tensorsOf, withTensors } from '../../lib/ml/transformer';
import type { PositionMode, TransformerCache, TransformerSpec, TransformerState, TransformerWeights } from '../../lib/ml/transformer';
import type { OptimiserName } from '../../lib/ml/optim';
import type { TfParams } from './config';
import { LETTER_WEIGHTS } from './letterWeights';

export function taskOf(task: string): SequenceTask {
  return task === 'count' || task === 'animals' || task === 'letters' ? task : 'reverse';
}

export type DataKnobs = Pick<TfParams, 'task' | 'size' | 'digits' | 'count' | 'seed'>;
export type ModelKnobs = Pick<TfParams, 'width' | 'heads' | 'layers' | 'ffn' | 'positions' | 'mask' | 'norm' | 'seed' | 'optimiser' | 'learningRate' | 'batchSize'>;

export function buildData(p: DataKnobs): SequenceDataset {
  return generateSequences({ task: taskOf(p.task), size: p.size, digits: p.digits, count: p.count, seed: p.seed, trainFraction: 0.75 });
}

export function buildSpec(p: ModelKnobs, data: SequenceDataset): TransformerSpec {
  const width = [8, 16, 32].includes(Number(p.width)) ? Number(p.width) : 16;
  const heads = [1, 2, 4].includes(Number(p.heads)) ? Number(p.heads) : 2;
  return {
    vocab: data.vocab.length,
    outVocab: data.outVocab.length,
    length: data.length,
    width,
    heads: Math.min(heads, width),
    layers: Math.max(1, Math.min(2, Math.round(p.layers))),
    ffn: p.ffn ? 4 * width : 0,
    positions: (['sinusoidal', 'learned', 'none'].includes(p.positions) ? p.positions : 'sinusoidal') as PositionMode,
    causal: p.mask === 'causal',
    norm: p.norm,
    seed: p.seed,
    optimiser: { name: p.optimiser as OptimiserName },
    learningRate: p.learningRate,
    batchSize: p.batchSize,
    // Scoring the letter task writes every held-out sentence, so it runs less often.
    evalEvery: data.task === 'letters' ? 25 : 10,
  };
}

/** What a set of weights was built for: a checkpoint fits only the model and the split it was trained on. */
export function specSignature(spec: TransformerSpec): string {
  return [spec.vocab, spec.outVocab, spec.length, spec.width, spec.heads, spec.layers, spec.ffn, spec.positions, spec.causal, spec.norm, spec.seed].join('|');
}

/** The shipped trained letter model, when the spec is the one it was trained for. */
export function trainedLetters(spec: TransformerSpec): { weights: TransformerWeights; steps: number } | null {
  if (LETTER_WEIGHTS.signature !== specSignature(spec)) return null;
  const bytes = Uint8Array.from(atob(LETTER_WEIGHTS.data), (c) => c.charCodeAt(0));
  const flat = new Float32Array(bytes.buffer);
  const shape = createTransformer(spec);
  let at = 0;
  const tensors = tensorsOf(shape).map((t) => {
    const out = Float64Array.from(flat.subarray(at, at + t.length));
    at += t.length;
    return out;
  });
  return at === flat.length ? { weights: withTensors(shape, tensors), steps: LETTER_WEIGHTS.steps } : null;
}

// The trained start is scored once: training never changes a state in place, so every run may begin from the same one.
let trainedStart: { key: string; state: TransformerState } | null = null;

/** A fresh run: random weights, or the trained letter model when asked for and it fits. */
export function startState(spec: TransformerSpec, data: SequenceDataset, start: string): TransformerState {
  if (start !== 'trained' || data.task !== 'letters') return createState(spec, data);
  const key = specSignature(spec) + '|' + data.testIndex.join(',');
  if (trainedStart?.key === key) return trainedStart.state;
  const trained = trainedLetters(spec);
  if (!trained) return createState(spec, data);
  const state = createState(spec, data, trained.weights);
  trainedStart = { key, state: { ...state, step: trained.steps, history: state.history.map((h) => ({ ...h, step: trained.steps })) } };
  return trainedStart.state;
}

/** The sequence the diagram follows: a held-out one, or what the reader typed. */
export interface Probe {
  tokens: number[];
  targets: number[];
  scored: number[];
  /** Position in the held-out list, or null for a typed sequence. */
  index: number | null;
  /** Why the typed text was not used, or null. */
  error: string | null;
}

export function probeOf(input: string, probe: number, data: SequenceDataset): Probe {
  if (input.trim() !== '') {
    const parsed = parseTokens(data, input);
    if ('tokens' in parsed) return { tokens: parsed.tokens, ...labelTokens(data.task, data.vocab, parsed.tokens), index: null, error: null };
    return { ...heldOut(probe, data), error: parsed.error };
  }
  return { ...heldOut(probe, data), error: null };
}

function heldOut(probe: number, data: SequenceDataset): Omit<Probe, 'error'> {
  const index = Math.max(0, Math.min(probe, data.testIndex.length - 1));
  const at = data.testIndex[index] ?? 0;
  const first = data.scored[at][0];
  // A letter sentence is followed up to "says ", so its last row is the letter to guess.
  if (data.task === 'letters' && first !== undefined) {
    return { tokens: data.inputs[at].slice(0, first + 1), targets: data.targets[at].slice(0, first + 1), scored: [first], index };
  }
  return { tokens: data.inputs[at], targets: data.targets[at], scored: data.scored[at], index };
}

/** The call at every position the way the model would make it while writing: a next-word task reads only the prefix. */
export function honestCalls(weights: TransformerWeights, spec: TransformerSpec, data: SequenceDataset, tokens: readonly number[], full: TransformerCache): number[] {
  if (!data.nextToken || spec.causal) return full.predicted;
  return tokens.map((_, t) => forward(weights, tokens.slice(0, t + 1), spec).predicted[t]);
}

/** What the model writes, greedily to the full stop: from the words up to the animal, or from all of a typed start. */
export function writeFrom(weights: TransformerWeights, spec: TransformerSpec, data: SequenceDataset, tokens: readonly number[], whole = false): { prompt: number[]; tokens: number[] } | null {
  if (!data.nextToken) return null;
  const prompt = whole ? tokens.slice() : tokens.slice(0, promptLength(data.task, data.vocab, tokens));
  return { prompt, tokens: generate(weights, prompt, spec, data.vocab.indexOf('.')) };
}

/* ---------------- readings ---------------- */

/** Attention weight from query i to key j in one head. */
export function weightAt(cache: TransformerCache, layer: number, head: number, i: number, j: number): number {
  const T = cache.length;
  return cache.layers[layer]?.attn[(head * T + i) * T + j] ?? 0;
}

/** For the reverse task: the mean weight each head puts on the mirror position, and the head that does it most. */
export function mirrorShare(cache: TransformerCache, spec: TransformerSpec): { head: number; share: number; shares: number[] } {
  const T = cache.length;
  const shares: number[] = [];
  for (let h = 0; h < spec.heads; h++) {
    let sum = 0;
    for (let i = 1; i < T; i++) sum += weightAt(cache, 0, h, i, T - i);
    shares.push(sum / Math.max(1, T - 1));
  }
  let head = 0;
  shares.forEach((s, h) => {
    if (s > shares[head]) head = h;
  });
  return { head, share: shares[head] ?? 0, shares };
}

/** For the language task: where says is, where the animal is, and each head's weight from one to the other. */
export function animalShare(cache: TransformerCache, spec: TransformerSpec, data: SequenceDataset): { says: number; animal: number; head: number; share: number; shares: number[] } | null {
  const says = cache.tokens.indexOf(data.vocab.indexOf('says'));
  const animal = cache.tokens.findIndex((t) => ANIMALS.some((a) => a.name === data.vocab[t]));
  if (says < 0 || animal < 0 || animal > says) return null;
  const shares: number[] = [];
  for (let h = 0; h < spec.heads; h++) shares.push(weightAt(cache, 0, h, says, animal));
  let head = 0;
  shares.forEach((s, h) => {
    if (s > shares[head]) head = h;
  });
  return { says, animal, head, share: shares[head] ?? 0, shares };
}

/** The largest weight in a query's row, averaged over the rows: 1 / T for a blur, 1 for a pointer. */
export function sharpness(cache: TransformerCache, spec: TransformerSpec, layer = 0): number {
  const T = cache.length;
  let sum = 0;
  let rows = 0;
  for (let h = 0; h < spec.heads; h++) {
    for (let i = 0; i < T; i++) {
      let top = 0;
      for (let j = 0; j < T; j++) top = Math.max(top, weightAt(cache, layer, h, i, j));
      sum += top;
      rows += 1;
    }
  }
  return rows > 0 ? sum / rows : 0;
}

/** The mean share of each row's attention that lands on later positions: 0 under a causal mask. */
export function aheadShare(cache: TransformerCache, spec: TransformerSpec, layer = 0): number {
  const T = cache.length;
  let sum = 0;
  let rows = 0;
  for (let h = 0; h < spec.heads; h++) {
    for (let i = 0; i < T - 1; i++) {
      for (let j = i + 1; j < T; j++) sum += weightAt(cache, layer, h, i, j);
      rows += 1;
    }
  }
  return rows > 0 ? sum / rows : 0;
}

/** Euclidean length of row t of a T × d matrix. */
export function rowNorm(values: Float64Array, t: number, d: number): number {
  let s = 0;
  for (let i = 0; i < d; i++) s += values[t * d + i] ** 2;
  return Math.sqrt(s);
}

/** One row of a T × d matrix. */
export function rowOf(values: Float64Array, t: number, d: number): Float64Array {
  return values.subarray(t * d, t * d + d);
}

/** One head's slice of a query, key or value row. */
export function headRow(values: Float64Array, t: number, head: number, spec: TransformerSpec): Float64Array {
  const dh = headWidth(spec);
  const d = spec.width;
  return values.subarray(t * d + head * dh, t * d + head * dh + dh);
}

/** The token vectors on their two main directions: principal components of the embedding table. */
export function embeddingPlane(weights: TransformerWeights, spec: TransformerSpec): { points: Array<{ x: number; y: number }>; explained: number } {
  const V = spec.vocab;
  const d = spec.width;
  const centred = new Float64Array(V * d);
  for (let i = 0; i < d; i++) {
    let mean = 0;
    for (let a = 0; a < V; a++) mean += weights.E[a * d + i];
    mean /= V;
    for (let a = 0; a < V; a++) centred[a * d + i] = weights.E[a * d + i] - mean;
  }
  const cov = new Float64Array(d * d);
  for (let a = 0; a < V; a++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i * d + j] += centred[a * d + i] * centred[a * d + j];
  let total = 0;
  for (let i = 0; i < d; i++) total += cov[i * d + i];
  const components: Float64Array[] = [];
  let explained = 0;
  for (let k = 0; k < 2; k++) {
    // Power iteration from a fixed start, then deflate.
    let v = new Float64Array(d).map((_, i) => 1 + i * 0.1);
    let lambda = 0;
    for (let it = 0; it < 200; it++) {
      const next = new Float64Array(d);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) next[i] += cov[i * d + j] * v[j];
      const norm = Math.hypot(...next) || 1;
      lambda = norm;
      v = next.map((x) => x / norm);
    }
    components.push(v);
    explained += lambda;
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i * d + j] -= lambda * v[i] * v[j];
  }
  const points = Array.from({ length: V }, (_, a) => {
    let x = 0;
    let y = 0;
    for (let i = 0; i < d; i++) {
      x += centred[a * d + i] * components[0][i];
      y += centred[a * d + i] * components[1][i];
    }
    return { x, y };
  });
  return { points, explained: total > 0 ? explained / total : 0 };
}

/** Which kind of word a token is, for the embedding map's colours. */
export function wordGroup(data: SequenceDataset, token: number): number {
  const word = data.vocab[token] ?? '';
  if (data.task === 'letters') return token === 0 ? 0 : 'aeiou'.includes(word) ? 1 : word === ' ' || word === '.' ? 3 : 2;
  if (data.task !== 'animals') return token === 0 ? 0 : 1;
  if (ANIMALS.some((a) => a.name === word)) return 1;
  if (ANIMALS.some((a) => a.sound === word)) return 2;
  if (ADJECTIVES.includes(word)) return 3;
  if (ADVERBS.includes(word)) return 4;
  return 0;
}

/* ---------------- diagram ids ---------------- */

/** The components a card can open: a token, its input vector, one query's attention, the add, the feed-forward block, the prediction. */
export const NODE = {
  token: (t: number) => 'tok:' + t,
  embed: (t: number) => 'emb:' + t,
  attn: (layer: number, t: number) => 'attn:' + layer + ':' + t,
  add: (layer: number, t: number) => 'add:' + layer + ':' + t,
  ffn: (layer: number, t: number) => 'ffn:' + layer + ':' + t,
  pred: (t: number) => 'pred:' + t,
} as const;

export type NodeKind = 'tok' | 'emb' | 'attn' | 'add' | 'ffn' | 'pred';

export function parseNode(id: string): { kind: NodeKind; layer: number; position: number } {
  const [kind, a, b] = id.split(':');
  if (b === undefined) return { kind: kind as NodeKind, layer: 0, position: Number(a ?? 0) };
  return { kind: kind as NodeKind, layer: Number(a), position: Number(b) };
}

export const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

export function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => SUB[Number(d)] ?? d)
    .join('');
}

/** How a token reads in the diagrams: the start token, a digit, a word, or a letter with the space made visible. */
export function tokenName(data: SequenceDataset, token: number): string {
  const name = data.vocab[token] ?? '?';
  return name === ' ' ? '␣' : name;
}
