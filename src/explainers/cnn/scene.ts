/** The network as a scene: every layer, kernel, map, unit and class, with the values they hold for the probe image. Shared by the 2D and 3D views. */

import { fmt } from '../../lib/math/stats';
import { channel, filterTensor, parameterCount } from '../../lib/ml/conv';
import type { CnnCache, CnnParams, CnnSpec } from '../../lib/ml/conv';

export type CnnNodeKind = 'image' | 'filter' | 'map' | 'pool' | 'flat' | 'unit' | 'out';

export interface Picture {
  values: ArrayLike<number>;
  cols: number;
  rows: number;
  ramp: 'signed' | 'magnitude';
}

export interface CnnNode {
  id: string;
  kind: CnnNodeKind;
  stage: number;
  index: number;
  /** The symbol: 'x', 'k₁', 'm₁', 'p₁', 'f', 'h₁', '─'. */
  label: string;
  /** The number or shape it holds now, for the hover plate. */
  readout: string;
  picture?: Picture;
  /** A unit's activation or a class's probability, in [0, 1], for its tint. */
  value?: number;
  /** The predicted class. */
  emphasis?: boolean;
}

export interface CnnEdge {
  id: string;
  from: string;
  to: string;
  weight?: number;
  symbol?: string;
  readout?: string;
}

export type CnnLayerKind = 'image' | 'filters' | 'maps' | 'pooled' | 'flat' | 'dense' | 'scores';

export interface CnnLayer {
  id: string;
  kind: CnnLayerKind;
  stage: number;
  title: string;
  subtitle: string;
  nodes: string[];
}

export interface CnnScene {
  layers: CnnLayer[];
  nodes: CnnNode[];
  byId: Map<string, CnnNode>;
  edges: CnnEdge[];
  summary: string;
  maxWeight: number;
  /** The probe's true class and the network's call. */
  label: number;
  predicted: number;
  /** False before the first batch: the call is a random guess. */
  trained: boolean;
}

const SUB = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

export function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => SUB[Number(d)] ?? d)
    .join('');
}

export const NODE = {
  image: 'image',
  filter: (stage: number, index: number) => 'filter:' + stage + ':' + index,
  map: (stage: number, index: number) => 'map:' + stage + ':' + index,
  pool: (stage: number, index: number) => 'pool:' + stage + ':' + index,
  flat: 'flat',
  unit: (index: number) => 'unit:' + index,
  out: (index: number) => 'out:' + index,
} as const;

/** The weight from dense unit `u` (or flat value `u` without a dense layer) to class `c`. */
export function weightEdge(u: number, c: number): string {
  return 'w:' + u + ':' + c;
}

export function parseNode(id: string): { kind: string; stage: number; index: number } {
  const [kind, a, b] = id.split(':');
  if (b === undefined) return { kind, stage: 0, index: Number(a ?? 0) };
  return { kind, stage: Number(a), index: Number(b) };
}

export function parseEdge(id: string): { unit: number; cls: number } | null {
  const [kind, u, c] = id.split(':');
  if (kind !== 'w') return null;
  return { unit: Number(u), cls: Number(c) };
}

export function maxOf(values: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
}

/** The brightest cell of a map. */
export function brightest(values: ArrayLike<number>, width: number): { x: number; y: number; value: number } {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return { x: best % width, y: Math.floor(best / width), value: values[best] };
}

export function architectureLabel(spec: CnnSpec, params: CnnParams, withCount = true): string {
  const parts: string[] = [];
  if (spec.model === 'dense') parts.push('flatten ' + spec.size * spec.size);
  else {
    params.convs.forEach((c) => parts.push('conv ' + c.kernel + '×' + c.kernel + ' ×' + c.outC + (spec.pool === 'none' ? '' : ' → ' + spec.pool + ' 2')));
  }
  if (params.hidden) parts.push('dense ' + params.hidden.outSize);
  parts.push('softmax ' + params.out.outSize);
  return parts.join(' → ') + (withCount ? ' · ' + parameterCount(params).toLocaleString('en-US') + ' params' : '');
}

export interface SceneInput {
  params: CnnParams;
  cache: CnnCache;
  spec: CnnSpec;
  classSymbols: readonly string[];
  classNames: readonly string[];
  label: number;
  trained: boolean;
}

export function buildScene({ params, cache, spec, classSymbols, classNames, label, trained }: SceneInput): CnnScene {
  const layers: CnnLayer[] = [];
  const nodes: CnnNode[] = [];
  const edges: CnnEdge[] = [];
  const size = spec.size;

  const add = (layer: CnnLayer, node: CnnNode) => {
    nodes.push(node);
    layer.nodes.push(node.id);
  };

  const image: CnnLayer = { id: 'image', kind: 'image', stage: 0, title: 'Image', subtitle: size + ' × ' + size, nodes: [] };
  layers.push(image);
  add(image, {
    id: NODE.image,
    kind: 'image',
    stage: 0,
    index: 0,
    label: 'x',
    readout: size + ' × ' + size + ' · ' + (classSymbols[label] ?? ''),
    picture: { values: cache.input.data, cols: size, rows: size, ramp: 'magnitude' },
  });

  let previous: string[] = [NODE.image];
  cache.stages.forEach((stage, s) => {
    const layer = params.convs[s];
    const k = layer.kernel;
    const prime = s > 0 ? '′' : '';
    const filters: CnnLayer = {
      id: 'filters:' + s,
      kind: 'filters',
      stage: s,
      title: 'Kernels' + (s > 0 ? ' ' + (s + 1) : ''),
      subtitle: k + ' × ' + k + (layer.inC > 1 ? ' × ' + layer.inC : ''),
      nodes: [],
    };
    const maps: CnnLayer = {
      id: 'maps:' + s,
      kind: 'maps',
      stage: s,
      title: 'Maps' + (s > 0 ? ' ' + (s + 1) : ''),
      subtitle: stage.a.h + ' × ' + stage.a.w + ' · relu',
      nodes: [],
    };
    layers.push(filters, maps);
    const pooled = spec.pool !== 'none';
    const pools: CnnLayer | null = pooled
      ? { id: 'pooled:' + s, kind: 'pooled', stage: s, title: 'Pooled' + (s > 0 ? ' ' + (s + 1) : ''), subtitle: spec.pool + ' 2 × 2 · ' + stage.pooled.h + ' × ' + stage.pooled.w, nodes: [] }
      : null;
    if (pools) layers.push(pools);
    const next: string[] = [];
    for (let i = 0; i < layer.outC; i++) {
      const kernel = filterTensor(layer, i);
      const map = channel(stage.a, i);
      const filterId = NODE.filter(s, i);
      const mapId = NODE.map(s, i);
      add(filters, {
        id: filterId,
        kind: 'filter',
        stage: s,
        index: i,
        label: 'k' + sub(i + 1) + prime,
        readout: k + ' × ' + k + (layer.inC > 1 ? ' × ' + layer.inC : '') + ' · b = ' + fmt(layer.b[i], 2),
        picture: { values: kernel.data, cols: k, rows: k, ramp: 'signed' },
      });
      for (const from of previous) edges.push({ id: from + '->' + filterId, from, to: filterId, symbol: '∗' });
      add(maps, {
        id: mapId,
        kind: 'map',
        stage: s,
        index: i,
        label: 'm' + sub(i + 1) + prime,
        readout: stage.a.h + ' × ' + stage.a.w + ' · max ' + fmt(maxOf(map), 2),
        picture: { values: map, cols: stage.a.w, rows: stage.a.h, ramp: 'magnitude' },
      });
      edges.push({ id: filterId + '->' + mapId, from: filterId, to: mapId, symbol: 'relu' });
      if (pools) {
        const poolId = NODE.pool(s, i);
        const pool = channel(stage.pooled, i);
        add(pools, {
          id: poolId,
          kind: 'pool',
          stage: s,
          index: i,
          label: 'p' + sub(i + 1) + prime,
          readout: stage.pooled.h + ' × ' + stage.pooled.w + ' · max ' + fmt(maxOf(pool), 2),
          picture: { values: pool, cols: stage.pooled.w, rows: stage.pooled.h, ramp: 'magnitude' },
        });
        edges.push({ id: mapId + '->' + poolId, from: mapId, to: poolId, symbol: '↓2' });
        next.push(poolId);
      } else next.push(mapId);
    }
    previous = next;
  });

  const flat: CnnLayer = { id: 'flat', kind: 'flat', stage: 0, title: 'Flatten', subtitle: cache.flat.length + ' values', nodes: [] };
  layers.push(flat);
  let active = 0;
  for (let i = 0; i < cache.flat.length; i++) if (cache.flat[i] > 0) active++;
  add(flat, {
    id: NODE.flat,
    kind: 'flat',
    stage: 0,
    index: 0,
    label: 'f',
    readout: cache.flat.length + ' × 1 · ' + active + ' > 0',
    picture: { values: cache.flat, cols: 1, rows: cache.flat.length, ramp: 'magnitude' },
  });
  for (const from of previous) edges.push({ id: from + '->' + NODE.flat, from, to: NODE.flat, symbol: 'flatten' });

  let head: string[] = [NODE.flat];
  let headValues: ArrayLike<number> = cache.flat;
  if (params.hidden && cache.hiddenA) {
    const dense: CnnLayer = { id: 'dense', kind: 'dense', stage: 0, title: 'Dense', subtitle: params.hidden.outSize + ' relu units', nodes: [] };
    layers.push(dense);
    const scale = maxOf(cache.hiddenA) || 1;
    head = [];
    for (let u = 0; u < params.hidden.outSize; u++) {
      const id = NODE.unit(u);
      add(dense, {
        id,
        kind: 'unit',
        stage: 0,
        index: u,
        label: 'h' + sub(u + 1),
        readout: '= ' + fmt(cache.hiddenA[u], 2),
        value: cache.hiddenA[u] / scale,
      });
      edges.push({ id: NODE.flat + '->' + id, from: NODE.flat, to: id });
      head.push(id);
    }
    headValues = cache.hiddenA;
  }

  const scores: CnnLayer = { id: 'scores', kind: 'scores', stage: 0, title: 'Scores', subtitle: 'softmax ' + params.out.outSize, nodes: [] };
  layers.push(scores);
  let maxWeight = 0;
  for (let c = 0; c < params.out.outSize; c++) {
    const p = cache.probs[c] ?? 0;
    const id = NODE.out(c);
    add(scores, {
      id,
      kind: 'out',
      stage: 0,
      index: c,
      label: classSymbols[c] ?? String(c),
      readout: 'p = ' + fmt(p, 2) + (cache.predicted === c ? (trained ? ' · called' : ' · guess') : '') + (c === label ? ' · true' : ''),
      value: p,
      emphasis: cache.predicted === c,
    });
    if (params.hidden) {
      head.forEach((from, u) => {
        const w = params.out.W[c * params.out.inSize + u];
        maxWeight = Math.max(maxWeight, Math.abs(w));
        edges.push({ id: weightEdge(u, c), from, to: id, weight: w, symbol: 'w', readout: '= ' + fmt(w, 2) + ' · ' + fmt(headValues[u], 2) });
      });
    } else {
      edges.push({ id: NODE.flat + '->' + id, from: NODE.flat, to: id, symbol: 'W', readout: 'z = ' + fmt(cache.logits[c], 2) });
    }
  }

  const byId = new Map(nodes.map((n) => [n.id, n]));
  return {
    layers,
    nodes,
    byId,
    edges,
    summary: architectureLabel(spec, params) + ' · ' + (classNames[label] ?? ''),
    maxWeight: maxWeight || 1,
    label,
    predicted: cache.predicted,
    trained,
  };
}

/* ---------------- the receptive field ---------------- */

export interface Window {
  /** A map pixel and the patch it reads, or a pooled cell and its block. */
  kind: 'map' | 'pool';
  stage: number;
  index: number;
  /** The output cell. */
  y: number;
  x: number;
}

/** The input patch a map pixel reads, in the coordinates of the stage's input. */
export function patchOf(params: CnnParams, cache: CnnCache, window: Window): { y: number; x: number; k: number } | null {
  const layer = params.convs[window.stage];
  if (!layer || !cache.stages[window.stage]) return null;
  return { y: window.y - layer.pad, x: window.x - layer.pad, k: layer.kernel };
}

/** Every term of one map pixel: pixel, weight and product for each kernel cell of one input channel, plus the sum over the other channels. */
export function windowTerms(
  params: CnnParams,
  cache: CnnCache,
  window: Window,
  inputChannel: number,
): { terms: Array<{ pixel: number; weight: number }>; others: number; bias: number; z: number; a: number } | null {
  const layer = params.convs[window.stage];
  const stage = cache.stages[window.stage];
  if (!layer || !stage) return null;
  const k = layer.kernel;
  const input = stage.input;
  const terms: Array<{ pixel: number; weight: number }> = [];
  let others = 0;
  for (let i = 0; i < layer.inC; i++) {
    for (let ky = 0; ky < k; ky++) {
      for (let kx = 0; kx < k; kx++) {
        const yy = window.y + ky - layer.pad;
        const xx = window.x + kx - layer.pad;
        const inside = yy >= 0 && yy < input.h && xx >= 0 && xx < input.w;
        const pixel = inside ? input.data[i * input.h * input.w + yy * input.w + xx] : 0;
        const weight = layer.W[((window.index * layer.inC + i) * k + ky) * k + kx];
        if (i === inputChannel) terms.push({ pixel, weight });
        else others += pixel * weight;
      }
    }
  }
  const at = window.y * stage.z.w + window.x;
  const z = stage.z.data[window.index * stage.z.h * stage.z.w + at];
  return { terms, others, bias: layer.b[window.index], z, a: Math.max(0, z) };
}

/* ---------------- the field on the image ---------------- */

export interface Field {
  y0: number;
  x0: number;
  /** Exclusive. */
  y1: number;
  x1: number;
}

/** Cells of a stage's input that a map pixel (or pooled cell) at stage `stage` reads, before clamping. */
function stageField(params: CnnParams, window: Window): Field {
  const layer = params.convs[window.stage];
  const k = layer?.kernel ?? 1;
  const pad = layer?.pad ?? 0;
  if (window.kind === 'pool') {
    // A pooled cell covers a 2 x 2 block of the map, which reads k + 1 rows of the input.
    return { y0: 2 * window.y - pad, x0: 2 * window.x - pad, y1: 2 * window.y - pad + k + 1, x1: 2 * window.x - pad + k + 1 };
  }
  return { y0: window.y - pad, x0: window.x - pad, y1: window.y - pad + k, x1: window.x - pad + k };
}

/** The patch of the image a cell depends on, through every earlier layer, clamped to the image. */
export function imageFieldOf(params: CnnParams, spec: CnnSpec, window: Window): Field {
  let field = stageField(params, window);
  for (let s = window.stage - 1; s >= 0; s--) {
    const layer = params.convs[s];
    const k = layer.kernel;
    const pad = layer.pad;
    // Pooled cells back to map pixels, then map pixels back to their input.
    if (spec.pool !== 'none') field = { y0: 2 * field.y0, x0: 2 * field.x0, y1: 2 * field.y1, x1: 2 * field.x1 };
    field = { y0: field.y0 - pad, x0: field.x0 - pad, y1: field.y1 - pad + k - 1, x1: field.x1 - pad + k - 1 };
  }
  const size = spec.size;
  return { y0: Math.max(0, field.y0), x0: Math.max(0, field.x0), y1: Math.min(size, field.y1), x1: Math.min(size, field.x1) };
}

/** Rows and columns of the map a window slides over. */
export function windowDims(cache: CnnCache, window: Window): { cols: number; rows: number } {
  const stage = cache.stages[window.stage];
  if (!stage) return { cols: 1, rows: 1 };
  const t = window.kind === 'pool' ? stage.pooled : stage.a;
  return { cols: t.w, rows: t.h };
}
