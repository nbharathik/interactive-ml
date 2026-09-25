/** The plane through the network: the input grid, then every layer's weighted sums and activations, then the output, as one chain. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { prefersReducedMotion } from '../../explainer/useSimulation';
import { fmt, sub } from '../../lib/math/stats';
import type { ActivationFn } from '../../lib/ml/activations';
import { softmax } from '../../lib/ml/losses';
import { lossFnOf } from '../../lib/ml/mlp';
import { predictedClassFromHidden, scoreFromOutput } from '../../lib/ml/mlpTrainer';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { extent2, gridLines, outputFromHidden, principalBasis, projectOnto, traceLayers } from '../../lib/ml/warp';
import type { Basis, GridLine } from '../../lib/ml/warp';
import { FONT_STACK, MONO_STACK, clamp, drawMarker, linearScale, roundRect } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { categorical, mix, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawCategoryField, drawSignedField } from '../../lib/viz/plots';
import { sameTarget, targetKey, unitLayer } from './targets';
import type { DiagramTarget, Spot } from './targets';
import { drawUnitTile, sampleUnits, tileNorm } from './tiles';
import type { LayerFields } from './tiles';

/** Square panes laid out in a grid, as large as the canvas allows. */
export function paneRects(width: number, height: number, count: number, gap = 10): Rect[] {
  let best = { cols: 1, size: 0 };
  for (let cols = 1; cols <= count; cols++) {
    const rows = Math.ceil(count / cols);
    const size = Math.min((width - gap * (cols - 1)) / cols, (height - gap * (rows - 1)) / rows);
    if (size > best.size) best = { cols, size };
  }
  const rows = Math.ceil(count / best.cols);
  const size = Math.max(40, Math.floor(best.size));
  const left = (width - (best.cols * size + gap * (best.cols - 1))) / 2;
  const top = (height - (rows * size + gap * (rows - 1))) / 2;
  return Array.from({ length: count }, (_, i) => ({
    x: left + (i % best.cols) * (size + gap),
    y: top + Math.floor(i / best.cols) * (size + gap),
    w: size,
    h: size,
  }));
}

/* ---------------- the stages ---------------- */

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
type XY = [number, number];
type Domain = { x: [number, number]; y: [number, number] };

/** The two numbers written at each axis, and the box a bounded f fills. */
interface Ticks {
  lo: number;
  hi: number;
  scale: number;
  box: boolean;
  /** Numbers for the y axis when they differ from x. */
  y?: [number, number];
}

interface Stage {
  /** 'x' or 'h₁'. */
  symbol: string;
  /** 'z₁ = W₁x + b₁'; null for the input. */
  linearTitle: string | null;
  /** 'h₁ = tanh(z₁)'. */
  title: string;
  axes: [string, string];
  projected: boolean;
  /** Units in this layer; 2 for the input. */
  width: number;
  /** Full vectors of the grid samples, then the points, in this stage's space. */
  vectors: number[][];
  /** The same as two coordinates, after f and before it. */
  after: XY[];
  before: XY[] | null;
  domain: Domain;
  beforeDomain: Domain | null;
}

function squared(domain: Domain): Domain {
  const span = Math.max(domain.x[1] - domain.x[0], domain.y[1] - domain.y[0]);
  const cx = (domain.x[0] + domain.x[1]) / 2;
  const cy = (domain.y[0] + domain.y[1]) / 2;
  return { x: [cx - span / 2, cx + span / 2], y: [cy - span / 2, cy + span / 2] };
}

function to2D(basis: Basis, vectors: readonly (readonly number[])[]): XY[] {
  return vectors.map((v) => (v.length === 1 ? [v[0], 0] : projectOnto(basis, v)));
}

/** The box a bounded activation fills, so the pane never rescales; else the extent of what is drawn. */
function domainOf(act: ActivationFn, coords: XY[]): Domain {
  const range = act.range;
  if (range && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
    const pad = (range[1] - range[0]) * 0.1;
    return { x: [range[0] - pad, range[1] + pad], y: [range[0] - pad, range[1] + pad] };
  }
  return squared(extent2(coords));
}

export interface StageSet {
  lines: GridLine[];
  gridCount: number;
  stages: Stage[];
  /** Raw output vector of every training point. */
  outputs: number[][];
}

export function buildStages(state: TrainerState, act: ActivationFn, inputs: readonly (readonly number[])[]): StageSet {
  const inputDim = inputs[0]?.length ?? 2;
  const lines = gridLines(inputDim);
  const gridPoints = lines.flatMap((l) => l.points);
  const all = gridPoints.concat(inputs.map((p) => p.slice()));
  const traces = traceLayers(state.net, state.spec, all);
  const hiddenCount = state.net.layers.length - 1;
  const f = act.label.toLowerCase();
  const identity: Basis = { mean: [0, 0], axes: [[1, 0], [0, 1]] };
  const stages: Stage[] = [
    {
      symbol: 'x',
      linearTitle: null,
      title: inputDim === 1 ? 'x' : 'x = (x₁, x₂)',
      axes: inputDim === 1 ? ['x', ''] : ['x₁', 'x₂'],
      projected: false,
      width: inputDim,
      vectors: all,
      after: to2D(identity, all),
      before: null,
      domain: { x: [-1.1, 1.1], y: [-1.1, 1.1] },
      beforeDomain: null,
    },
  ];
  for (let l = 0; l < hiddenCount; l++) {
    const trace = traces[l];
    const width = state.net.layers[l].outSize;
    const projected = width > 2;
    const basis = projected ? principalBasis(trace.a.slice(0, gridPoints.length)) : identity;
    const after = to2D(basis, trace.a);
    const before = to2D(basis, trace.z);
    const k = l + 1;
    const z = 'z' + sub(k);
    stages.push({
      symbol: 'h' + sub(k),
      linearTitle: z + ' = W' + sub(k) + (l === 0 ? 'x' : 'h' + sub(l)) + ' + b' + sub(k),
      title: 'h' + sub(k) + ' = ' + (act.name === 'linear' ? z : f + '(' + z + ')'),
      axes: projected ? ['pc 1 of ' + width + ' units', 'pc 2'] : ['unit 1', 'unit 2'],
      projected,
      width,
      vectors: trace.a,
      after,
      before,
      domain: projected ? squared(extent2(after)) : domainOf(act, after),
      beforeDomain: squared(extent2(before)),
    });
  }
  const last = traces[traces.length - 1];
  return { lines, gridCount: gridPoints.length, stages, outputs: last ? last.a.slice(gridPoints.length) : [] };
}

/** ŷ as the number the reader expects: a probability, a margin, or the standardised value. */
export function outputReadout(state: TrainerState, out: readonly number[], task: 'classify' | 'regress'): { label: string; value: string } {
  if (task === 'regress') return { label: 'ŷ', value: fmt(out[0], 2) };
  if (state.spec.loss === 'hinge') return { label: 'score', value: fmt(out[0], 2) };
  const score = scoreFromOutput(state.spec.loss, out);
  return { label: 'ŷ', value: fmt((score + 1) / 2, 2) };
}

/* ---------------- the output stage: every point at its ŷ, against its y ---------------- */

interface OutputStage {
  title: string;
  axes: [string, string];
  coords: XY[];
  domain: Domain;
  /** Where the call flips, drawn as a line; null when there is none. */
  flip: number | null;
  /** Class rows, or null for a value fit. */
  rows: number | null;
  ticks: Ticks | null;
}

function buildOutput(
  state: TrainerState,
  outputs: readonly (readonly number[])[],
  labels: readonly number[] | null,
  targets: readonly (readonly number[])[],
  classCount: number,
): OutputStage {
  if (!labels) {
    const coords: XY[] = outputs.map((o, i) => [o[0], targets[i]?.[0] ?? 0]);
    return { title: 'ŷ = z', axes: ['ŷ', 'y'], coords, domain: squared(extent2(coords, 0.12)), flip: null, rows: null, ticks: null };
  }
  // Points of one class share a row; a little spread keeps them apart.
  const row = (i: number) => labels[i] + (((i * 7919) % 97) / 96 - 0.5) * 0.5;
  const y: [number, number] = [-0.7, classCount - 0.3];
  const rows: [number, number] = [0, classCount - 1];
  const loss = state.spec.loss;
  if (loss === 'softmaxCE') {
    const coords: XY[] = outputs.map((o, i) => [softmax(o)[labels[i]] ?? 0, row(i)]);
    return { title: 'ŷ = softmax(z)', axes: ['p of own class', 'y'], coords, domain: { x: [-0.08, 1.08], y }, flip: null, rows: classCount, ticks: { lo: 0, hi: 1, scale: 1, box: false, y: rows } };
  }
  if (loss === 'hinge') {
    const coords: XY[] = outputs.map((o, i) => [clamp(o[0], -3, 3), row(i)]);
    return { title: 'score = z', axes: ['score', 'y'], coords, domain: { x: [-3.3, 3.3], y }, flip: 0, rows: classCount, ticks: { lo: -3, hi: 3, scale: 1, box: false, y: rows } };
  }
  const decode = lossFnOf(state.spec).decodeValue;
  const coords: XY[] = outputs.map((o, i) => [decode(o), row(i)]);
  return { title: 'ŷ = σ(z)', axes: ['ŷ', 'y'], coords, domain: { x: [-0.08, 1.08], y }, flip: 0.5, rows: classCount, ticks: { lo: 0, hi: 1, scale: 1, box: false, y: rows } };
}

/* ---------------- what f does, in a few words ---------------- */

const BEND: Record<string, string> = {
  sigmoid: 'squashed into (0, 1)',
  tanh: 'squashed into (−1, 1)',
  relu: 'negatives clipped to 0',
  leakyRelu: 'negatives shrunk by α',
  elu: 'negatives bent to −α',
  gelu: 'negatives bent to 0',
  softplus: 'negatives bent to 0',
  swish: 'negatives bent to 0',
  linear: 'unchanged: still a grid',
};

/* ---------------- layout: rows of panes, read left to right and down ---------------- */

interface Pane {
  kind: 'input' | 'linear' | 'activation' | 'output';
  /** 0 for the input, k for z_k and h_k, hidden count + 1 for the output. */
  layer: number;
  rect: Rect;
  /** Alone in its row: the title and caption go beside it instead of above and below. */
  beside: boolean;
}

interface Link {
  /** 'both' is a folded layer: W and f on one arrow. */
  kind: 'linear' | 'activation' | 'both' | 'output';
  layer: number;
  /** Two points for a straight arrow, four for the return to the next row. */
  path: XY[];
  /** 'across' sits over and under a horizontal arrow; 'line' is one line anchored at (x, y). */
  label: { kind: 'across' | 'line'; x: number; y: number; align: 'left' | 'center' | 'right' };
  zone: Rect;
}

interface Layout {
  folded: boolean;
  size: number;
  width: number;
  /** Lines under a pane: a folded caption tells of W and f both. */
  captionLines: number;
  nodes: Pane[];
  links: Link[];
}

const PAD = 8;
const TITLE_H = 20;
/** Room for the numbers left of and under a pane. */
const TICK_W = 24;
const TICK_H = 12;
const CAPTION_LINE = 12;
/** Between rows: the arrow down, or the return to the next row with its label. */
const GAP_ROW = 40;
/** Between two panes that have their rows to themselves: only the ticks and a short arrow. */
const GAP_BESIDE = 26;
/** The replay button sits in the top-left corner, above a first title. */
const TOOL_H = 30;
/** Below this pane size the weighted sums fold into the arrow before f. */
const FOLD_MIN = 88;
const MIN_SIZE = 34;

/** Between columns; a folded arrow carries W and f, so it needs more. */
function gapX(width: number, folded: boolean): number {
  return folded ? 112 : clamp(width * 0.07, 52, 76);
}

/** The panes in chain order, grouped so a layer's z and h never split across rows. */
function atomsFor(hiddenCount: number, folded: boolean): Pane[][] {
  const pane = (kind: Pane['kind'], layer: number): Pane => ({ kind, layer, rect: { x: 0, y: 0, w: 0, h: 0 }, beside: false });
  const atoms: Pane[][] = [[pane('input', 0)]];
  for (let k = 1; k <= hiddenCount; k++) atoms.push(folded ? [pane('activation', k)] : [pane('linear', k), pane('activation', k)]);
  atoms.push([pane('output', hiddenCount + 1)]);
  return atoms;
}

function pack(atoms: Pane[][], cols: number): Pane[][] {
  const rows: Pane[][] = [];
  let row: Pane[] = [];
  for (const atom of atoms) {
    if (row.length > 0 && row.length + atom.length > cols) {
      rows.push(row);
      row = [];
    }
    row.push(...atom);
  }
  rows.push(row);
  return rows;
}

const captionLines = (folded: boolean) => (folded ? 4 : 3);
const above = (row: Pane[]) => (row.length === 1 ? 0 : TITLE_H);
const below = (row: Pane[], folded: boolean) => TICK_H + (row.length === 1 ? 0 : 2 + CAPTION_LINE * captionLines(folded));
const gapAfter = (rows: Pane[][], i: number) => (i === rows.length - 1 ? 0 : rows[i].length === 1 && rows[i + 1].length === 1 ? GAP_BESIDE : GAP_ROW);

/** The pane size the rows allow; one column keeps 45% of the width for the text beside it. */
function sizeFor(rows: Pane[][], cols: number, width: number, height: number, folded: boolean): number {
  const w = cols === 1 ? (width - PAD * 2 - TICK_W) * 0.55 : (width - PAD * 2 - cols * TICK_W - (cols - 1) * gapX(width, folded)) / cols;
  const top = rows[0].length === 1 ? PAD : TOOL_H;
  const used = rows.reduce((sum, row, i) => sum + above(row) + below(row, folded) + gapAfter(rows, i), 0);
  return Math.min(w, (height - top - PAD - used) / rows.length);
}

/** The column count with the biggest panes; more columns only when clearly bigger, since their captions get a few lines under the pane. */
function pick(hiddenCount: number, folded: boolean, width: number, height: number): { rows: Pane[][]; cols: number; size: number } {
  const atoms = atomsFor(hiddenCount, folded);
  const count = atoms.reduce((sum, atom) => sum + atom.length, 0);
  let best = { rows: [] as Pane[][], cols: 1, size: -1 };
  for (let cols = Math.max(...atoms.map((atom) => atom.length)); cols <= count; cols++) {
    const rows = pack(atoms, cols);
    const size = sizeFor(rows, cols, width, height, folded);
    const floor = cols === 1 ? -1 : Math.max(FOLD_MIN, best.size * 1.1);
    if (size > best.size && size >= floor) best = { rows, cols, size };
  }
  return best;
}

/** Every stage as a pane if they can be FOLD_MIN wide, else one pane per layer. */
function layout(width: number, height: number, hiddenCount: number): Layout {
  const full = pick(hiddenCount, false, width, height);
  const folded = hiddenCount > 0 && full.size < FOLD_MIN;
  const { rows, cols, size: raw } = folded ? pick(hiddenCount, true, width, height) : full;
  const size = Math.max(MIN_SIZE, Math.floor(raw));
  const gap = gapX(width, folded);
  const slot = TICK_W + size;
  const left = cols === 1 ? PAD + TICK_W : (width - (cols * slot + (cols - 1) * gap)) / 2 + TICK_W;
  const used = rows.reduce((sum, row, i) => sum + above(row) + size + below(row, folded) + gapAfter(rows, i), 0);
  let y = rows[0].length === 1 ? PAD : TOOL_H;
  y += Math.max(0, (height - y - PAD - used) / 2);
  const nodes: Pane[] = [];
  rows.forEach((row, i) => {
    y += above(row);
    row.forEach((n, col) => {
      n.rect = { x: left + col * (slot + gap), y, w: size, h: size };
      n.beside = row.length === 1;
      nodes.push(n);
    });
    y += size + below(row, folded) + gapAfter(rows, i);
  });

  const links: Link[] = [];
  for (let i = 1; i < nodes.length; i++) {
    const from = nodes[i - 1];
    const to = nodes[i];
    const a = from.rect;
    const b = to.rect;
    const kind: Link['kind'] = to.kind === 'output' ? 'output' : to.kind === 'linear' ? 'linear' : folded ? 'both' : 'activation';
    if (a.y === b.y) {
      const yy = a.y + a.h / 2;
      const x0 = a.x + a.w + 5;
      const x1 = b.x - TICK_W - 4;
      links.push({ kind, layer: to.layer, path: [[x0, yy], [x1, yy]], label: { kind: 'across', x: (x0 + x1) / 2, y: yy, align: 'center' }, zone: { x: x0 - 2, y: yy - 40, w: x1 - x0 + 4, h: 76 } });
      continue;
    }
    const xa = a.x + a.w / 2;
    const xb = b.x + b.w / 2;
    const y0 = a.y + a.h + (from.beside ? TICK_H + 2 : TICK_H + 2 + CAPTION_LINE * captionLines(folded));
    const y1 = b.y - (to.beside ? 3 : TITLE_H + 2);
    if (xa === xb) {
      // Straight down; the label beside the arrow, in the text column when there is one.
      const mid = (y0 + y1) / 2;
      const label: Link['label'] = cols === 1 ? { kind: 'line', x: a.x + a.w + 14, y: mid, align: 'left' } : xa < width / 2 ? { kind: 'line', x: xa + 10, y: mid, align: 'left' } : { kind: 'line', x: xa - 10, y: mid, align: 'right' };
      links.push({ kind, layer: to.layer, path: [[xa, y0], [xa, y1]], label, zone: { x: xa - 12, y: y0 - 2, w: 24, h: y1 - y0 + 4 } });
      continue;
    }
    // The return to the next row: down, across, down, with the label over the crossing.
    const ym = y0 + (y1 - y0) * 0.62;
    links.push({ kind, layer: to.layer, path: [[xa, y0], [xa, ym], [xb, ym], [xb, y1]], label: { kind: 'line', x: (xa + xb) / 2, y: ym - 11, align: 'center' }, zone: { x: Math.min(xa, xb) - 12, y: y0 - 2, w: Math.abs(xa - xb) + 24, h: y1 - y0 + 4 } });
  }
  return { folded, size, width, captionLines: captionLines(folded), nodes, links };
}

/* ---------------- drawing pieces ---------------- */

function paneFrame(rect: Rect, domain: Domain): Frame {
  const left = rect.x;
  const top = rect.y;
  const right = rect.x + rect.w;
  const bottom = rect.y + rect.h;
  return { x: linearScale(domain.x, [left, right]), y: linearScale(domain.y, [bottom, top]), left, top, right, bottom, innerWidth: rect.w, innerHeight: rect.h };
}

function union(a: Rect, b: Rect): Rect {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
}

export function ease(t: number): number {
  const u = clamp(t, 0, 1);
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

function between(a: XY[], b: XY[], t: number): XY[] {
  if (t <= 0) return a;
  if (t >= 1) return b;
  return a.map(([x0, y0], i) => [x0 + (b[i][0] - x0) * t, y0 + (b[i][1] - y0) * t]);
}

function tick(v: number): string {
  return String(Number(v.toFixed(Math.abs(v) >= 10 ? 0 : 1)));
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  lines: readonly GridLine[],
  coords: readonly XY[],
  palette: Palette,
  options: { alpha?: number; width?: number; segmentColour?: (index: number) => string },
): void {
  const { alpha = 1, width = 1, segmentColour } = options;
  ctx.save();
  let offset = 0;
  for (const line of lines) {
    const family = line.family === 0 ? palette.violet : palette.cyan;
    ctx.lineWidth = line.axis ? width * 1.7 : width;
    if (segmentColour && !line.axis) {
      for (let s = 1; s < line.points.length; s++) {
        const [x0, y0] = coords[offset + s - 1];
        const [x1, y1] = coords[offset + s];
        ctx.strokeStyle = segmentColour(offset + s);
        ctx.beginPath();
        ctx.moveTo(frame.x(x0), frame.y(y0));
        ctx.lineTo(frame.x(x1), frame.y(y1));
        ctx.stroke();
      }
    } else {
      ctx.strokeStyle = rgba(family, (line.axis ? 0.9 : 0.4) * alpha);
      ctx.beginPath();
      for (let s = 0; s < line.points.length; s++) {
        const [x, y] = coords[offset + s];
        if (s === 0) ctx.moveTo(frame.x(x), frame.y(y));
        else ctx.lineTo(frame.x(x), frame.y(y));
      }
      ctx.stroke();
    }
    offset += line.points.length;
  }
  ctx.restore();
}

function drawPoints(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  coords: readonly XY[],
  labels: readonly number[] | null,
  palette: Palette,
  radius: number,
  hot = -1,
  alpha = 1,
): void {
  ctx.save();
  ctx.globalAlpha = alpha;
  coords.forEach(([x, y], i) => {
    const colour = labels ? categorical(palette, labels[i]) : palette.blue;
    drawMarker(ctx, 'circle', frame.x(x), frame.y(y), i === hot ? radius + 1.5 : radius, colour, rgba(palette.surface, 0.9));
  });
  ctx.restore();
  if (hot >= 0 && coords[hot]) {
    const [x, y] = coords[hot];
    ctx.beginPath();
    ctx.arc(frame.x(x), frame.y(y), radius + 5, 0, Math.PI * 2);
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function drawZeroLines(ctx: CanvasRenderingContext2D, frame: Frame, domain: Domain, palette: Palette): void {
  ctx.save();
  ctx.setLineDash([2, 3]);
  ctx.strokeStyle = rgba(palette.axis, 0.7);
  ctx.lineWidth = 1;
  if (domain.x[0] < 0 && domain.x[1] > 0) {
    ctx.beginPath();
    ctx.moveTo(frame.x(0), frame.top);
    ctx.lineTo(frame.x(0), frame.bottom);
    ctx.stroke();
  }
  if (domain.y[0] < 0 && domain.y[1] > 0) {
    ctx.beginPath();
    ctx.moveTo(frame.left, frame.y(0));
    ctx.lineTo(frame.right, frame.y(0));
    ctx.stroke();
  }
  ctx.restore();
}

/** The pane's square, the axis names inside, the numbers outside at the box a bounded f fills or at the edges; bare for a pane of tiles. */
function drawPaneBox(ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette, axes: [string, string], domain: Domain, ticks: Ticks | null, bare = false): void {
  ctx.fillStyle = rgba(palette.surfaceAlt, 0.6);
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = palette.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  if (bare) return;
  ctx.font = '9px ' + MONO_STACK;
  ctx.fillStyle = palette.textFaint;
  ctx.textBaseline = 'alphabetic';
  if (rect.w >= 90) {
    ctx.textAlign = 'right';
    ctx.fillText(axes[0] + ' →', rect.x + rect.w - 4, rect.y + rect.h - 4);
    ctx.textAlign = 'left';
    ctx.fillText(axes[1] + ' ↑', rect.x + 4, rect.y + 10);
  }
  const frame = paneFrame(rect, domain);
  const lo = ticks ? ticks.lo : domain.x[0];
  const hi = ticks ? ticks.hi : domain.x[1];
  const [ylo, yhi] = ticks?.y ?? [ticks ? ticks.lo : domain.y[0], ticks ? ticks.hi : domain.y[1]];
  const scale = ticks ? ticks.scale : 1;
  if (ticks && ticks.box) {
    ctx.strokeStyle = rgba(palette.textFaint, 0.5);
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(frame.x(lo) + 0.5, frame.y(hi) + 0.5, frame.x(hi) - frame.x(lo), frame.y(lo) - frame.y(hi));
    ctx.setLineDash([]);
  }
  if (rect.w < 48) return;
  ctx.fillStyle = palette.textFaint;
  ctx.textAlign = 'center';
  ctx.fillText(tick(lo * scale), clamp(frame.x(lo), rect.x + 8, rect.x + rect.w - 8), rect.y + rect.h + 10);
  ctx.fillText(tick(hi * scale), clamp(frame.x(hi), rect.x + 8, rect.x + rect.w - 8), rect.y + rect.h + 10);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(tick(yhi * scale), rect.x - 4, clamp(frame.y(yhi), rect.y + 5, rect.y + rect.h - 5));
  ctx.fillText(tick(ylo * scale), rect.x - 4, clamp(frame.y(ylo), rect.y + 5, rect.y + rect.h - 5));
  ctx.textBaseline = 'alphabetic';
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxW: number, maxLines: number): string[] {
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? line + ' ' + word : word;
    if (ctx.measureText(next).width > maxW && line) {
      lines.push(line);
      line = word;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines.slice(0, maxLines);
}

/** Title over the pane and caption under it; both beside a pane that has its row to itself. */
function drawPaneText(ctx: CanvasRenderingContext2D, L: Layout, pane: Pane, palette: Palette, title: string, caption: string, hot: boolean): void {
  const rect = pane.rect;
  ctx.textBaseline = 'alphabetic';
  ctx.font = '600 11px ' + MONO_STACK;
  ctx.fillStyle = hot ? palette.accent : palette.text;
  ctx.textAlign = 'left';
  if (!pane.beside) {
    ctx.fillText(title, rect.x - TICK_W + 2, rect.y - 6);
    ctx.font = '10px ' + FONT_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.textAlign = 'center';
    wrap(ctx, caption, rect.w + TICK_W, L.captionLines).forEach((l, i) => ctx.fillText(l, rect.x + rect.w / 2 - TICK_W / 2, rect.y + rect.h + TICK_H + 13 + i * CAPTION_LINE));
    return;
  }
  const textX = rect.x + rect.w + 14;
  ctx.fillText(title, textX, rect.y + 10);
  ctx.font = '10px ' + FONT_STACK;
  ctx.fillStyle = palette.textMuted;
  const maxLines = Math.max(1, Math.min(5, Math.floor((rect.h - 18) / 12)));
  wrap(ctx, caption, L.width - PAD - textX, maxLines).forEach((l, i) => ctx.fillText(l, textX, rect.y + 27 + i * 12));
}

/** Room kept at the top of a pane for its chip. */
const CHIP_H = 20;

/** The fold that shows a layer unit by unit: a small pill in the pane's top-right corner. */
function drawUnitsChip(ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette, label: string, hot: boolean, on: boolean): Rect {
  ctx.font = '600 9px ' + MONO_STACK;
  const w = Math.ceil(ctx.measureText(label).width) + 12;
  const h = 16;
  const left = rect.x + rect.w - w - 3;
  const top = rect.y + 3;
  roundRect(ctx, left + 0.5, top + 0.5, w, h, 4);
  ctx.fillStyle = on ? rgba(palette.accent, 0.12) : rgba(palette.surface, 0.9);
  ctx.fill();
  ctx.strokeStyle = on || hot ? palette.accent : palette.border;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = on || hot ? palette.accent : palette.textMuted;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, left + 6, top + h / 2 + 1);
  ctx.textBaseline = 'alphabetic';
  return { x: left, y: top, w: w + 1, h: h + 1 };
}

/** Square tiles inside a pane, one per unit, under the chip row. */
function tileRects(rect: Rect, count: number): Rect[] {
  const inset = 4;
  const top = rect.y + CHIP_H;
  return paneRects(rect.w - inset * 2, rect.h - CHIP_H - inset, count, 3).map((r) => ({ ...r, x: r.x + rect.x + inset, y: r.y + top }));
}

/** A polyline with the head on its last segment. */
export function drawArrow(ctx: CanvasRenderingContext2D, path: XY[], palette: Palette): void {
  const [x0, y0] = path[path.length - 2];
  const [x1, y1] = path[path.length - 1];
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  ctx.save();
  ctx.strokeStyle = rgba(palette.textMuted, 0.9);
  ctx.fillStyle = rgba(palette.textMuted, 0.9);
  ctx.lineWidth = 1.2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  path.forEach(([x, y], i) => {
    const tip = i === path.length - 1;
    const px = tip ? x - ux * 5 : x;
    const py = tip ? y - uy * 5 : y;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * 6 - uy * 3.5, y1 - uy * 6 + ux * 3.5);
  ctx.lineTo(x1 - ux * 6 + uy * 3.5, y1 - uy * 6 - ux * 3.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/** The shape of f: what every coordinate goes through. */
function drawCurveIcon(ctx: CanvasRenderingContext2D, box: Rect, act: ActivationFn, palette: Palette): void {
  const N = 40;
  const values: number[] = [];
  for (let i = 0; i <= N; i++) values.push(act.f(-4 + (8 * i) / N));
  const lo = Math.min(0, ...values);
  const hi = Math.max(1, ...values);
  const y = (v: number) => box.y + box.h - ((v - lo) / (hi - lo)) * box.h;
  ctx.save();
  ctx.strokeStyle = rgba(palette.axis, 0.6);
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(box.x, y(0));
  ctx.lineTo(box.x + box.w, y(0));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = palette.blue;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  values.forEach((v, i) => {
    const px = box.x + (box.w * i) / N;
    if (i === 0) ctx.moveTo(px, y(v));
    else ctx.lineTo(px, y(v));
  });
  ctx.stroke();
  ctx.restore();
}

/** The arrow's label: the symbols in bold, what they do in small print, the curve of f beside an activation. Returns the box of a one-line label. */
function drawLinkLabel(ctx: CanvasRenderingContext2D, link: Link, act: ActivationFn | null, symbols: string, note: string, palette: Palette, hot: boolean): Rect | null {
  const curve = act && link.kind !== 'linear' && link.kind !== 'output';
  const { x, y, align } = link.label;
  ctx.textBaseline = 'alphabetic';
  if (link.label.kind === 'across') {
    const x0 = link.path[0][0];
    const x1 = link.path[1][0];
    ctx.textAlign = 'center';
    if (curve) {
      drawCurveIcon(ctx, { x: x0 + 6, y: y - 34, w: x1 - x0 - 12, h: 24 }, act, palette);
    } else {
      ctx.font = '600 11px ' + MONO_STACK;
      ctx.fillStyle = hot ? palette.accent : palette.text;
      ctx.fillText(symbols, x, y - 9);
    }
    ctx.font = curve ? '600 11px ' + MONO_STACK : '9px ' + FONT_STACK;
    ctx.fillStyle = curve ? (hot ? palette.accent : palette.text) : palette.textMuted;
    ctx.fillText(curve ? symbols : note, x, y + 15);
    if (curve) {
      ctx.font = '9px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.fillText(note, x, y + 26);
    }
    return null;
  }
  ctx.font = '600 11px ' + MONO_STACK;
  const symbolsW = ctx.measureText(symbols).width;
  ctx.font = '9px ' + FONT_STACK;
  const noteW = ctx.measureText(note).width;
  const place = (withNote: boolean) => {
    const width = (curve ? 30 : 0) + symbolsW + (withNote ? 8 + noteW : 0);
    return { width, left: align === 'left' ? x : align === 'right' ? x - width : x - width / 2 };
  };
  // The pane caption says the same thing, so a narrow column drops the note rather than run off the canvas.
  const room = ctx.canvas.clientWidth || Infinity;
  let { width: total, left } = place(true);
  const withNote = left + total <= room;
  if (!withNote) ({ width: total, left } = place(false));
  let cursor = left;
  if (curve) {
    drawCurveIcon(ctx, { x: cursor, y: y - 7, w: 24, h: 13 }, act, palette);
    cursor += 30;
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.font = '600 11px ' + MONO_STACK;
  ctx.fillStyle = hot ? palette.accent : palette.text;
  ctx.fillText(symbols, cursor, y);
  cursor += symbolsW + 8;
  if (withNote) {
    ctx.font = '9px ' + FONT_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.fillText(note, cursor, y + 1);
  }
  ctx.textBaseline = 'alphabetic';
  return { x: left - 4, y: y - 10, w: total + 8, h: 20 };
}

/* ---------------- the diagram ---------------- */

export interface SpaceDiagramProps {
  state: TrainerState;
  act: ActivationFn;
  task: 'classify' | 'regress';
  /** Training inputs in network units, their targets, and their class labels or null for regression. */
  inputs: readonly (readonly number[])[];
  targets: readonly (readonly number[])[];
  labels: readonly number[] | null;
  classCount: number;
  /** Multiplies the input pane's numbers back to the raw axes. */
  inputScale: number;
  hover: DiagramTarget | null;
  open: DiagramTarget | null;
  spot: Spot | null;
  onHover: (target: DiagramTarget | null) => void;
  onOpen: (target: DiagramTarget | null) => void;
  /** Hidden layers shown unit by unit, from 1. */
  unitsShown: readonly number[];
  onToggleUnits: (layer: number) => void;
  description: string;
  redrawKey: string;
}

const POINT_HIT = 7;

/** The replay: t runs 0 to units, one unit per turn or bend down the chain, the last for the output. Settled (t = units) until play is called. */
export function useReplay(units: number): { t: number; play: () => void } {
  const [t, setT] = useState(units);
  const frame = useRef(0);
  useEffect(() => {
    cancelAnimationFrame(frame.current);
    setT(units);
  }, [units]);
  const play = useCallback(() => {
    cancelAnimationFrame(frame.current);
    if (prefersReducedMotion()) {
      setT(units);
      return;
    }
    const duration = clamp(units * 640, 1200, 4400);
    const start = performance.now();
    const step = (now: number) => {
      const u = Math.min(1, (now - start) / duration);
      setT(u * units);
      if (u < 1) frame.current = requestAnimationFrame(step);
    };
    frame.current = requestAnimationFrame(step);
  }, [units]);
  useEffect(() => () => cancelAnimationFrame(frame.current), []);
  return { t, play };
}

export function SpaceDiagram(props: SpaceDiagramProps) {
  const { state, act, task, inputs, targets, labels, classCount, inputScale, hover, open, spot, onHover, onOpen, unitsShown, onToggleUnits, description, redrawKey } = props;
  const set = useMemo(() => buildStages(state, act, inputs), [state, act, inputs]);
  const out = useMemo(() => buildOutput(state, set.outputs, labels, targets, classCount), [state, set.outputs, labels, targets, classCount]);
  const hiddenCount = set.stages.length - 1;
  const units = 2 * hiddenCount + 1;
  const anyUnits = unitsShown.length > 0;
  const fields = useMemo(() => (anyUnits ? sampleUnits(state) : []), [state, anyUnits]);
  const planeInput = inputs[0]?.length === 2;
  const layoutRef = useRef<Layout | null>(null);
  const framesRef = useRef<Frame[]>([]);
  const tilesRef = useRef<Map<number, { layer: number; rects: Rect[] }>>(new Map());
  const chipsRef = useRef<{ layer: number; rect: Rect }[]>([]);
  const focus = open ?? hover;
  const binary = task === 'classify' && classCount <= 2;
  const replay = useReplay(units);

  const scoreOf = useCallback(
    (vector: readonly number[]) => {
      const o = outputFromHidden(state.net, state.spec, vector);
      if (task === 'regress') return clamp(o[0] / 2, -1, 1);
      return scoreFromOutput(state.spec.loss, o);
    },
    [state, task],
  );

  // The arrows walk the panes, each hidden layer's units chip after its pane; Enter on the chip folds it.
  const targetsList = useMemo<DiagramTarget[]>(
    () => [...set.stages.flatMap((_, index): DiagramTarget[] => (index > 0 && planeInput ? [{ kind: 'layer', index }, { kind: 'units', layer: index }] : [{ kind: 'layer', index }])), { kind: 'output' }],
    [set.stages, planeInput],
  );
  const handleKey = useDiagramKeys<DiagramTarget>({
    targets: targetsList,
    cursor: focus && (focus.kind === 'layer' || focus.kind === 'output' || focus.kind === 'units') ? focus : null,
    same: sameTarget,
    onCursor: onHover,
    onOpen: (target) => (target.kind === 'units' ? onToggleUnits(target.layer) : onOpen(target)),
    onClose: () => {
      onOpen(null);
      onHover(null);
    },
  });

  const outSize = state.net.layers[state.net.layers.length - 1]?.outSize ?? 1;
  const outputSymbols = (outSize === 1 ? 'v, b' : 'W' + sub(hiddenCount + 1) + ', b' + sub(hiddenCount + 1)) + (state.spec.loss === 'bce' || (task === 'classify' && state.spec.loss === 'mse') ? ' · σ' : state.spec.loss === 'softmaxCE' ? ' · softmax' : '');

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const L = layout(width, height, hiddenCount);
      layoutRef.current = L;
      framesRef.current = [];
      tilesRef.current = new Map();
      chipsRef.current = [];
      const hotPoint = focus && focus.kind === 'point' ? focus.index : -1;
      const hotLayer = focus && focus.kind === 'layer' ? focus.index : -1;
      const hotUnit = focus && focus.kind === 'unit' ? { layer: unitLayer(focus), index: focus.index } : null;
      const hotChip = focus && focus.kind === 'units' ? focus.layer : -1;
      const spotUnit = spot && spot.target.kind === 'unit' ? { layer: unitLayer(spot.target), index: spot.target.index } : null;
      const fieldsOf = (k: number): LayerFields | null => (unitsShown.includes(k) && fields[k - 1] && fields[k - 1].units.length > 0 ? fields[k - 1] : null);
      const hotFunction = focus !== null && focus.kind === 'function' && focus.which === 'activation';
      const hotOutput = focus !== null && focus.kind === 'output';
            const spotLayer = spot && spot.target.kind === 'layer' ? spot.target.index : -1;
      const spotFunction = spot !== null && spot.target.kind === 'function' && spot.target.which === 'activation';
      const spotOutput = spot !== null && spot.target.kind === 'output';
      const range = act.range && Number.isFinite(act.range[0]) && Number.isFinite(act.range[1]) ? (act.range as [number, number]) : null;
      const t = replay.t;
      const radius = clamp(L.size / 44, 1.5, 3.2);
      const gridWidth = L.size < 80 ? 0.7 : 1;
      const linear = act.name === 'linear';
      const f = linear ? 'f(z) = z' : act.label.toLowerCase();
      const bend = BEND[act.name] ?? 'bent by ' + f;

      const drawField = (frame: Frame, stage: Stage) => {
        if (stage.projected || stage.vectors[0]?.length !== 2) return;
        if (binary || task === 'regress') drawSignedField(ctx, frame, palette, (x, y) => scoreOf([x, y]), { cellSize: 5, alpha: 0.3 });
        else drawCategoryField(ctx, frame, palette, (x, y) => predictedClassFromHidden(state, [x, y]), { cellSize: 5, alpha: 0.22 });
      };
      const bySegment = (stage: Stage) => (index: number) => {
        const v = stage.vectors[index];
        return classCount > 2 && task === 'classify' ? rgba(categorical(palette, predictedClassFromHidden(state, v)), 0.7) : rgba(mix(palette.muted, scoreOf(v) < 0 ? palette.negative : palette.positive, Math.min(1, Math.abs(scoreOf(v)))), 0.8);
      };
      const boxTicks = (stage: Stage): Ticks | null => (range && !stage.projected ? { lo: range[0], hi: range[1], scale: 1, box: true } : null);
      const clipTo = (rect: Rect) => {
        ctx.save();
        ctx.beginPath();
        ctx.rect(rect.x, rect.y, rect.w, rect.h);
        ctx.clip();
      };
      const outlines: Array<() => void> = [];
      const outlineRect = (rect: Rect, state_: 'hover' | 'open' | 'spot', label: string | null) => {
        outlines.push(() => {
          strokeOutline(
            ctx,
            () => {
              ctx.beginPath();
              ctx.rect(rect.x - 1, rect.y - 1, rect.w + 2, rect.h + 2);
            },
            state_,
            palette,
          );
          if (label) {
            const size = measurePlate(ctx, [label]);
            const place = placePlate(size, { x: rect.x, y: rect.y + rect.h - 26, w: rect.w, h: 0 }, width, height);
            drawHoverPlate(ctx, [label], place.left, place.top, palette);
          }
        });
      };

      /* A layer unit by unit: one tile per unit over the input plane, the hot one lit and the rest dimmed. */
      const drawTiles = (nodeIndex: number, rect: Rect, k: number, layerFields: LayerFields, after: boolean) => {
        const rects = tileRects(rect, layerFields.units.length);
        tilesRef.current.set(nodeIndex, { layer: k, rects });
        const norm = tileNorm(act, layerFields, after);
        layerFields.units.forEach((unit, j) => {
          const tile = rects[j];
          const lit = hotUnit !== null && hotUnit.layer === k && hotUnit.index === j;
          const dimmed = hotUnit !== null && hotUnit.layer === k && !lit;
          drawUnitTile(ctx, tile, palette, after ? unit.a : unit.z, norm, unit.z, {
            label: (after ? 'a' : 'z') + sub(j + 1),
            points: inputs,
            labels,
            alpha: dimmed ? 0.45 : 1,
            creaseColour: lit ? palette.accent : undefined,
          });
          const tileState = lit ? (open && open.kind === 'unit' && open.index === j && unitLayer(open) === k ? 'open' : 'hover') : spotUnit && spotUnit.layer === k && spotUnit.index === j ? 'spot' : null;
          if (tileState) outlineRect(tile, tileState, tileState === 'spot' ? spot!.label : null);
        });
      };
      const drawChip = (rect: Rect, k: number) => {
        if (!planeInput) return;
        const on = unitsShown.includes(k);
        const chip = drawUnitsChip(ctx, rect, palette, set.stages[k].width + ' units ' + (on ? '▴' : '▾'), hotChip === k, on);
        chipsRef.current.push({ layer: k, rect: chip });
      };

      /* The panes, in chain order. */
      L.nodes.forEach((node, nodeIndex) => {
        const rect = node.rect;
        const k = node.layer;
        if (node.kind === 'input') {
          const stage = set.stages[0];
          const frame = paneFrame(rect, stage.domain);
          framesRef.current.push(frame);
          drawPaneBox(ctx, rect, palette, stage.axes, stage.domain, { lo: -1, hi: 1, scale: inputScale, box: false });
          clipTo(rect);
          if (hiddenCount === 0) drawField(frame, stage);
          drawZeroLines(ctx, frame, stage.domain, palette);
          const colourByScore = hiddenCount === 0 && stage.vectors[0]?.length !== 2;
          drawGrid(ctx, frame, set.lines, stage.after, palette, { width: gridWidth, ...(colourByScore ? { segmentColour: bySegment(stage) } : {}) });
          drawPoints(ctx, frame, stage.after.slice(set.gridCount), labels, palette, radius, hotPoint);
          ctx.restore();
          const caption =
            hiddenCount === 0
              ? 'no hidden layer: the output unit cuts the input plane with one straight line'
              : stage.vectors[0]?.length === 1
                ? 'the input line, with the training points on it'
                : 'the input plane: a grid, with the training points on it';
          drawPaneText(ctx, L, node, palette, stage.title, caption, hotLayer === 0);
          if (hotLayer === 0) outlineRect(rect, open && open.kind === 'layer' && open.index === 0 ? 'open' : 'hover', null);
          else if (spotLayer === 0) outlineRect(rect, 'spot', spot!.label);
          return;
        }
        if (node.kind === 'output') {
          const frame = paneFrame(rect, out.domain);
          framesRef.current.push(frame);
          drawPaneBox(ctx, rect, palette, out.axes, out.domain, out.ticks);
          clipTo(rect);
          if (out.rows !== null) {
            for (let c = 0; c < out.rows; c++) {
              ctx.strokeStyle = rgba(categorical(palette, c), 0.35);
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(frame.left, frame.y(c));
              ctx.lineTo(frame.right, frame.y(c));
              ctx.stroke();
            }
          } else {
            ctx.save();
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = rgba(palette.axis, 0.8);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(frame.x(out.domain.x[0]), frame.y(out.domain.y[0]));
            ctx.lineTo(frame.x(out.domain.x[1]), frame.y(out.domain.y[1]));
            ctx.stroke();
            ctx.restore();
          }
          if (out.flip !== null) {
            ctx.strokeStyle = rgba(palette.text, 0.7);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(frame.x(out.flip), frame.top);
            ctx.lineTo(frame.x(out.flip), frame.bottom);
            ctx.stroke();
          }
          const fade = ease(t - (units - 1));
          if (fade > 0) drawPoints(ctx, frame, out.coords, labels, palette, radius, hotPoint, fade);
          ctx.restore();
          const caption =
            out.rows === null
              ? 'every point at its output, against its target: a perfect fit lies on the diagonal'
              : state.spec.loss === 'softmaxCE'
                ? 'every point at the probability of its own class, in its class row: fitted when they all sit near 1'
                : 'every point at its output, in its class row: fitted when the classes split at ' + (out.flip === 0 ? '0' : '½');
          drawPaneText(ctx, L, node, palette, out.title, caption, hotOutput);
          if (hotOutput) outlineRect(rect, open && open.kind === 'output' ? 'open' : 'hover', null);
          else if (spotOutput) outlineRect(rect, 'spot', spot!.label);
          return;
        }
        const stage = set.stages[k];
        const previous = set.stages[k - 1];
        if (!stage.before || !stage.beforeDomain) return;
        const turn = ease(t - (2 * k - 2));
        const bendT = ease(t - (2 * k - 1));
        const isLast = k === hiddenCount;
        const hot = hotLayer === k;
        const outlineState = hot ? (open && open.kind === 'layer' && open.index === k ? 'open' : 'hover') : spotLayer === k ? 'spot' : null;
        const layerFields = fieldsOf(k);
        if (node.kind === 'linear') {
          const frame = paneFrame(rect, stage.beforeDomain);
          framesRef.current.push(frame);
          if (layerFields) {
            drawPaneBox(ctx, rect, palette, stage.axes, stage.beforeDomain, null, true);
            drawTiles(nodeIndex, rect, k, layerFields, false);
            drawPaneText(ctx, L, node, palette, stage.linearTitle ?? '', 'each unit’s weighted sum z over the input plane, x₁ across and x₂ up: orange below 0, blue above, the line where z = 0', hot);
            if (outlineState) outlineRect(rect, outlineState, null);
            return;
          }
          drawPaneBox(ctx, rect, palette, stage.axes, stage.beforeDomain, null);
          clipTo(rect);
          drawZeroLines(ctx, frame, stage.beforeDomain, palette);
          if (t > 2 * k - 2) {
            const coords = between(previous.after, stage.before, turn);
            drawGrid(ctx, frame, set.lines, coords, palette, { width: gridWidth });
            drawPoints(ctx, frame, coords.slice(set.gridCount), labels, palette, radius, hotPoint);
          }
          ctx.restore();
          const turned = 'turned and stretched by W' + sub(k) + ', slid by b' + sub(k);
          drawPaneText(ctx, L, node, palette, stage.linearTitle ?? '', stage.projected ? turned + '; ' + stage.width + ' units, seen along the 2 widest directions' : 'still a grid: ' + turned, hot);
          if (outlineState) outlineRect(rect, outlineState, null);
          return;
        }
        const frame = paneFrame(rect, stage.domain);
        framesRef.current.push(frame);
        const then = isLast ? (task === 'classify' ? '; the output unit cuts it with one straight line' : '; the output unit reads it off with one flat ramp') : '; on to layer ' + (k + 1);
        if (layerFields) {
          drawPaneBox(ctx, rect, palette, stage.axes, stage.domain, null, true);
          drawTiles(nodeIndex, rect, k, layerFields, true);
          const bent = linear ? 'unchanged' : act.name === 'relu' ? 'negatives clipped to 0, the white zone' : bend;
          const caption = (L.folded ? 'W' + sub(k) + ', b' + sub(k) + ' then ' + f + ', unit by unit over the input plane: ' : 'each unit after ' + f + ', over the input plane: ') + bent + (isLast ? '; the output adds them up' : then);
          drawPaneText(ctx, L, node, palette, stage.title, caption, hot);
          drawChip(rect, k);
          if (outlineState) outlineRect(rect, outlineState, outlineState === 'spot' ? spot!.label : null);
          return;
        }
        drawPaneBox(ctx, rect, palette, stage.axes, stage.domain, boxTicks(stage));
        clipTo(rect);
        if (isLast) drawField(frame, stage);
        drawZeroLines(ctx, frame, stage.domain, palette);
        const settled = t >= 2 * k;
        if (t > 2 * k - 2 && (L.folded || t >= 2 * k - 1)) {
          // A folded layer turns, then bends, in the one pane.
          const coords = L.folded && t < 2 * k - 1 ? between(previous.after, stage.before, turn) : between(stage.before, stage.after, bendT);
          const colourByScore = isLast && settled && (stage.projected || stage.vectors[0]?.length !== 2);
          drawGrid(ctx, frame, set.lines, coords, palette, { width: gridWidth, ...(colourByScore ? { segmentColour: bySegment(stage) } : {}) });
          drawPoints(ctx, frame, coords.slice(set.gridCount), labels, palette, radius, hotPoint);
        }
        ctx.restore();
        const caption = L.folded ? 'W' + sub(k) + ', b' + sub(k) + ': turned, stretched and slid' + (linear ? '; f(z) = z: still a grid' : ', still a grid; ' + f + ': ' + bend) + then : bend + then;
        drawPaneText(ctx, L, node, palette, stage.title, caption, hot);
        drawChip(rect, k);
        if (outlineState) outlineRect(rect, outlineState, outlineState === 'spot' ? spot!.label : null);
      });

      /* The arrows: the matrix, the curve every coordinate goes through, the output unit. */
      let firstActivation = true;
      for (const link of L.links) {
        drawArrow(ctx, link.path, palette);
        const k = link.layer;
        const drawLabel = (fn: ActivationFn | null, symbols: string, note: string, hot: boolean) => {
          const box = drawLinkLabel(ctx, link, fn, symbols, note, palette, hot);
          if (box) link.zone = union(link.zone, box);
        };
        const outlineZone = (state_: 'hover' | 'open' | 'spot', label: string | null) => {
          const zone = link.zone;
          outlines.push(() => {
            strokeOutline(
              ctx,
              () => {
                ctx.beginPath();
                ctx.rect(zone.x, zone.y, zone.w, zone.h);
              },
              state_,
              palette,
            );
            if (label) {
              const size = measurePlate(ctx, [label]);
              const place = placePlate(size, zone, width, height);
              drawHoverPlate(ctx, [label], place.left, place.top, palette);
            }
          });
        };
        if (link.kind === 'linear') {
          drawLabel(null, 'W' + sub(k) + ', b' + sub(k), 'weighted sums', hotLayer === k);
        } else if (link.kind === 'activation') {
          drawLabel(act, f, linear ? 'nothing bends' : 'on every value', hotFunction);
          if (hotFunction) outlineZone(open && open.kind === 'function' ? 'open' : 'hover', null);
          else if (spotFunction) outlineZone('spot', firstActivation ? spot!.label : null);
          firstActivation = false;
        } else if (link.kind === 'both') {
          drawLabel(act, 'W' + sub(k) + ', b' + sub(k) + (linear ? '' : ', ' + f), linear ? 'weighted sums, nothing bends' : 'weighted sums, bent', hotLayer === k || hotFunction);
          if (hotFunction) outlineZone(open && open.kind === 'function' ? 'open' : 'hover', null);
          else if (spotFunction) outlineZone('spot', firstActivation ? spot!.label : null);
          firstActivation = false;
        } else {
          drawLabel(null, outputSymbols, outSize === 1 ? 'the output unit' : 'the output units', hotOutput);
        }
      }
      outlines.forEach((fn) => fn());
    },
    [set, out, units, hiddenCount, focus, open, spot, act, replay.t, inputScale, inputs, labels, binary, task, classCount, state, scoreOf, outputSymbols, outSize, unitsShown, fields, planeInput],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): DiagramTarget | null => {
      const L = layoutRef.current;
      if (!L) return null;
      const inside = (r: Rect) => pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h;
      const coordsOf = (node: Pane): XY[] | null => {
        if (node.kind === 'input') return set.stages[0].after.slice(set.gridCount);
        if (node.kind === 'output') return out.coords;
        const stage = set.stages[node.layer];
        return (node.kind === 'linear' ? stage.before : stage.after)?.slice(set.gridCount) ?? null;
      };
      for (const chip of chipsRef.current) if (inside(chip.rect)) return { kind: 'units', layer: chip.layer };
      for (let i = 0; i < L.nodes.length; i++) {
        const node = L.nodes[i];
        if (!inside(node.rect)) continue;
        const tiles = tilesRef.current.get(i);
        if (tiles) {
          const j = tiles.rects.findIndex(inside);
          return j >= 0 ? { kind: 'unit', index: j, layer: tiles.layer } : { kind: 'layer', index: node.layer };
        }
        const frame = framesRef.current[i];
        const coords = coordsOf(node);
        if (frame && coords) {
          let best = -1;
          let bestD = POINT_HIT * POINT_HIT;
          coords.forEach(([x, y], p) => {
            const d = (frame.x(x) - pos.x) ** 2 + (frame.y(y) - pos.y) ** 2;
            if (d < bestD) {
              bestD = d;
              best = p;
            }
          });
          if (best >= 0) return { kind: 'point', index: best };
        }
        return node.kind === 'output' ? { kind: 'output' } : { kind: 'layer', index: node.layer };
      }
      for (const link of L.links) {
        if (!inside(link.zone)) continue;
        if (link.kind === 'activation') return { kind: 'function', which: 'activation' };
        if (link.kind === 'output') return { kind: 'output' };
        return { kind: 'layer', index: link.layer };
      }
      return null;
    },
    [set, out],
  );

  return (
    <div
      className="mlx-arch__stage"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys move along the chain, Enter opens a stage or folds a layer open unit by unit.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onHover(null);
      }}
    >
      <Chart
        draw={draw}
        height="fill"
        description={description}
        cursor={hover ? 'pointer' : 'default'}
        onPointerDown={(pos) => {
          const found = locate(pos);
          if (found && found.kind === 'units') {
            onToggleUnits(found.layer);
            return;
          }
          onOpen(found && sameTarget(found, open) ? null : found);
        }}
        onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
        onPointerLeave={() => onHover(null)}
        redrawKey={redrawKey + '|' + replay.t.toFixed(3) + '|' + unitsShown.join(',') + '|' + targetKey(hover) + '|' + targetKey(open) + '|' + (spot ? targetKey(spot.target) + spot.label : '')}
      />
      <button type="button" className="mlx-arch__tool" onClick={replay.play} title="Replay: the grid flows down the chain" aria-label="Replay the flow through the network">
        ▶
      </button>
    </div>
  );
}
