/** The whole network, live: each hidden unit as its own activation curve with the z it sees piled on it, the output as the loss with the batch on it, weights as wires, the gradient each layer receives at its end. */

import { useCallback, useMemo, useRef } from 'react';

import type { EdgePath } from '../../explainer/architecture';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline, strokeWire } from '../../explainer/diagramStyle';
import type { WireEmphasis } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { sigmoid, softplus } from '../../lib/ml/activations';
import type { ActivationFn } from '../../lib/ml/activations';
import { logSumExp } from '../../lib/ml/losses';
import type { LossFn } from '../../lib/ml/losses';
import { forward } from '../../lib/ml/mlp';
import { fmt, fmtPercent, sub } from '../../lib/math/stats';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { MONO_STACK, clamp, roundRect } from '../../lib/viz/canvas';
import { categorical, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { layerColour } from './panels';
import { sameTarget, targetKey, unitLayer } from './targets';
import type { DiagramTarget, Spot } from './targets';

/* ---------------- what every unit sees ---------------- */

export interface UnitStat {
  /** Pre-activations over the training set. */
  z: number[];
  /** Share of inputs where |f′| is at least 1e-3: the unit can still learn from them. */
  on: number;
  /** Mean |f′| over the inputs. */
  slope: number;
}

/** The loss as one curve every sample can sit on: a y = 0 sample is mirrored onto the y = 1 curve. */
export interface OutputCurve {
  label: string;
  domain: [number, number];
  f: (x: number) => number;
  df: (x: number) => number;
}

export interface NetStats {
  /** units[l][u] for hidden layer l, from 0. */
  units: UnitStat[][];
  /** Where every training sample sits on the output curve, per output unit. */
  outputs: { x: number; target: number }[][];
  curve: OutputCurve;
}

/** Whether a hidden unit has gone flat for every input: dead, or saturated. */
export function isOff(stat: UnitStat): boolean {
  return stat.z.length > 0 && stat.on < 0.01;
}

function outputCurve(lossFn: LossFn, sigmoidOut: boolean): OutputCurve {
  switch (lossFn.name) {
    case 'bce':
    case 'softmaxCE':
      return { label: 'margin', domain: [-6, 6], f: (x) => softplus(-x), df: (x) => sigmoid(x) - 1 };
    case 'hinge':
      return { label: 'margin', domain: [-3, 3], f: (s) => Math.max(0, 1 - s), df: (s) => (s < 1 ? -1 : 0) };
    case 'mse':
      if (sigmoidOut) {
        return {
          label: 'margin',
          domain: [-6, 6],
          f: (z) => 0.5 * (sigmoid(z) - 1) ** 2,
          df: (z) => (sigmoid(z) - 1) * sigmoid(z) * (1 - sigmoid(z)),
        };
      }
      break;
    default:
      break;
  }
  return { label: 'error', domain: lossFn.curve.xDomain, f: (e) => lossFn.curve.f(e, 0), df: (e) => lossFn.curve.df(e, 0) };
}

export function netStats(state: TrainerState, act: ActivationFn, lossFn: LossFn, inputs: readonly (readonly number[])[], targets: readonly (readonly number[])[]): NetStats {
  const layers = state.net.layers;
  const hidden = layers.length - 1;
  const last = layers[hidden];
  const units: UnitStat[][] = layers.slice(0, hidden).map((l) => Array.from({ length: l.outSize }, () => ({ z: [], on: 0, slope: 0 })));
  const outputs: { x: number; target: number }[][] = Array.from({ length: last.outSize }, () => []);
  const sigmoidOut = last.activation === 'sigmoid';
  const curve = outputCurve(lossFn, sigmoidOut);
  for (let i = 0; i < inputs.length; i++) {
    const caches = forward(state.net, inputs[i], state.spec);
    for (let l = 0; l < hidden; l++) {
      caches[l].z.forEach((z, u) => {
        const stat = units[l][u];
        const s = Math.abs(act.df(z));
        stat.z.push(z);
        if (s >= 1e-3) stat.on += 1;
        stat.slope += s;
      });
    }
    const out = caches[hidden];
    const target = targets[i] ?? [];
    if (lossFn.kind === 'regression') outputs[0].push({ x: out.a[0] - (target[0] ?? 0), target: 0 });
    else if (lossFn.kind === 'binary') {
      const y = target[0] === 1 ? 1 : 0;
      outputs[0].push({ x: y === 1 ? out.z[0] : -out.z[0], target: y });
    } else {
      const j = target.indexOf(1);
      if (j >= 0 && outputs[j]) outputs[j].push({ x: out.z[j] - logSumExp(out.z.filter((_, k) => k !== j)), target: j });
    }
  }
  const n = Math.max(1, inputs.length);
  for (const layer of units) for (const stat of layer) {
    stat.on /= n;
    stat.slope /= n;
  }
  return { units, outputs, curve };
}

const mean = (values: readonly number[]) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

/** The readout when the output is hovered: how hard the loss still pushes. */
export function outputRows(stats: NetStats): { label: string; value: string }[] {
  const all = stats.outputs.flat();
  if (all.length === 0) return [];
  const slopes = all.map((s) => Math.abs(stats.curve.df(s.x)));
  return [
    { label: 'mean L', value: fmt(mean(all.map((s) => stats.curve.f(s.x))), 3) },
    { label: 'mean |dL|', value: fmt(mean(slopes), 3) },
    { label: 'flat', value: fmtPercent(slopes.filter((d) => d < 0.05).length / slopes.length, 0) },
  ];
}

/** The readout when the activation's name is hovered: the hidden units as a whole. */
export function activationRows(stats: NetStats): { label: string; value: string }[] {
  const units = stats.units.flat();
  if (units.length === 0 || units[0].z.length === 0) return [];
  return [
    { label: 'off', value: units.filter(isOff).length + ' / ' + units.length },
    { label: 'mean |f′|', value: fmt(mean(units.map((u) => u.slope)), 2) },
  ];
}

/* ---------------- layout ---------------- */

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Slot {
  /** 0 is the input, sizes.length - 1 the output. */
  index: number;
  boxes: Box[];
  /** The whole column or row, for outlines. */
  bounds: Box;
  label: { x: number; y: number };
  /** Where the layer's gradient is drawn, null for the input. */
  grad: Box | null;
}

interface Layout {
  vertical: boolean;
  slots: Slot[];
  /** The activation's name, once beside the hidden layers. */
  act: Box | null;
  box: number;
}

const PAD = 10;
const MAX_BOX = 64;
const MIN_BOX = 16;
const WIRE_MIN = 26;
/** Vertical: the symbol column and the gradient column. Horizontal: the name row and the symbol + gradient rows. */
const SIDE_L = 52;
const SIDE_R = 76;
const TOP = 18;
const BOTTOM = 44;

function boxFor(flow: number, cross: number, count: number, maxUnits: number): number {
  const byFlow = flow / count - WIRE_MIN;
  const byCross = cross / maxUnits - 6;
  return clamp(Math.min(MAX_BOX, byFlow, byCross), MIN_BOX, MAX_BOX);
}

function layout(width: number, height: number, sizes: readonly number[]): Layout {
  const count = sizes.length;
  const maxUnits = Math.max(1, ...sizes);
  const vBox = boxFor(height - PAD * 2, width - SIDE_L - SIDE_R, count, maxUnits);
  const hBox = boxFor(width - PAD * 2, height - TOP - BOTTOM, count, maxUnits);
  const vertical = vBox > hBox;
  const box = vertical ? vBox : hBox;
  const gap = Math.min(14, box * 0.35);
  const slots: Slot[] = [];
  if (vertical) {
    const pitch = (height - PAD * 2) / count;
    const cx = SIDE_L + (width - SIDE_L - SIDE_R) / 2;
    sizes.forEach((n, k) => {
      const cy = PAD + pitch * (k + 0.5);
      const boxes = Array.from({ length: n }, (_, u) => ({ x: cx + (u - (n - 1) / 2) * (box + gap) - box / 2, y: cy - box / 2, w: box, h: box }));
      const left = boxes[0].x;
      const right = boxes[n - 1].x + box;
      slots.push({
        index: k,
        boxes,
        bounds: { x: left - 4, y: cy - box / 2 - 4, w: right - left + 8, h: box + 8 },
        label: { x: PAD + 20, y: cy },
        grad: k === 0 ? null : { x: width - SIDE_R + 8, y: cy - 10, w: SIDE_R - 12, h: 20 },
      });
    });
    const first = slots[1];
    const lastHidden = slots[count - 2];
    const act = count > 2 ? { x: PAD, y: first.bounds.y, w: 12, h: lastHidden.bounds.y + lastHidden.bounds.h - first.bounds.y } : null;
    return { vertical, slots, act, box };
  }
  // The block sits centred; the symbols and gradients hang just under it.
  const pitch = (width - PAD * 2) / count;
  const blockH = maxUnits * (box + gap) - gap;
  const cy = TOP + (height - TOP - BOTTOM) / 2;
  const labelY = cy + blockH / 2 + 18;
  sizes.forEach((n, k) => {
    const cx = PAD + pitch * (k + 0.5);
    const boxes = Array.from({ length: n }, (_, u) => ({ x: cx - box / 2, y: cy + (u - (n - 1) / 2) * (box + gap) - box / 2, w: box, h: box }));
    const top = boxes[0].y;
    const bottom = boxes[n - 1].y + box;
    const barW = Math.min(pitch - 10, 64);
    slots.push({
      index: k,
      boxes,
      bounds: { x: cx - box / 2 - 4, y: top - 4, w: box + 8, h: bottom - top + 8 },
      label: { x: cx, y: labelY },
      grad: k === 0 ? null : { x: cx - barW / 2, y: labelY + 6, w: barW, h: 22 },
    });
  });
  const first = slots[1];
  const lastHidden = slots[count - 2];
  const act = count > 2 ? { x: first.bounds.x, y: cy - blockH / 2 - 20, w: lastHidden.bounds.x + lastHidden.bounds.w - first.bounds.x, h: 14 } : null;
  return { vertical, slots, act, box };
}

/* ---------------- drawing pieces ---------------- */

const Z_WINDOW = 4;
const LOSS_CAP = 4;

function histogram(values: readonly number[], domain: [number, number], bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const span = domain[1] - domain[0];
  for (const v of values) {
    const t = clamp((v - domain[0]) / span, 0, 0.999999);
    counts[Math.floor(t * bins)] += 1;
  }
  return counts;
}

function drawRug(ctx: CanvasRenderingContext2D, inner: Box, counts: readonly number[], colour: string, rugH: number): void {
  const max = Math.max(1, ...counts);
  const barW = inner.w / counts.length;
  ctx.fillStyle = rgba(colour, 0.75);
  counts.forEach((c, i) => {
    if (c === 0) return;
    const h = Math.max(1, (c / max) * rugH);
    ctx.fillRect(inner.x + i * barW + 0.5, inner.y + inner.h - h, Math.max(1, barW - 1), h);
  });
}

function drawFrame(ctx: CanvasRenderingContext2D, box: Box, palette: Palette, off: boolean): void {
  roundRect(ctx, box.x, box.y, box.w, box.h, 3);
  ctx.fillStyle = off ? rgba(palette.red, 0.12) : rgba(palette.surfaceAlt, 0.8);
  ctx.fill();
  ctx.strokeStyle = off ? rgba(palette.red, 0.9) : palette.border;
  ctx.lineWidth = off ? 1.4 : 1;
  ctx.stroke();
}

/** A hidden unit: f over z in [-4, 4], flat zones tinted, the z it sees as bars along the floor. */
function drawUnit(ctx: CanvasRenderingContext2D, box: Box, act: ActivationFn, stat: UnitStat, colour: string, palette: Palette): void {
  const off = isOff(stat);
  drawFrame(ctx, box, palette, off);
  const inner = { x: box.x + 2, y: box.y + 3, w: box.w - 4, h: box.h - 6 };
  const rugH = inner.h * 0.36;
  const plotH = inner.h - rugH - 1;
  const range: [number, number] = act.range && Number.isFinite(act.range[0]) && Number.isFinite(act.range[1]) ? [Math.min(act.range[0], -1), Math.max(act.range[1], 1)] : [-1, Z_WINDOW];
  const xOf = (z: number) => inner.x + ((z + Z_WINDOW) / (2 * Z_WINDOW)) * inner.w;
  const yOf = (v: number) => inner.y + plotH - ((clamp(v, range[0], range[1]) - range[0]) / (range[1] - range[0])) * plotH;
  ctx.save();
  ctx.beginPath();
  ctx.rect(inner.x, inner.y, inner.w, inner.h);
  ctx.clip();
  if (box.w >= 24) {
    // Flat zones, then the axes through zero.
    ctx.fillStyle = rgba(palette.red, 0.1);
    const steps = 24;
    let start: number | null = null;
    for (let i = 0; i <= steps; i++) {
      const z = -Z_WINDOW + (2 * Z_WINDOW * i) / steps;
      const flat = Math.abs(act.df(z)) < 0.05;
      if (flat && start === null) start = z;
      if ((!flat || i === steps) && start !== null) {
        ctx.fillRect(xOf(start), inner.y, xOf(z) - xOf(start), inner.h);
        start = null;
      }
    }
    ctx.strokeStyle = rgba(palette.axis, 0.5);
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(xOf(0), inner.y);
    ctx.lineTo(xOf(0), inner.y + inner.h);
    ctx.moveTo(inner.x, yOf(0));
    ctx.lineTo(inner.x + inner.w, yOf(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = off ? rgba(palette.blue, 0.6) : palette.blue;
  ctx.lineWidth = box.w >= 32 ? 1.4 : 1.1;
  ctx.beginPath();
  const samples = 32;
  for (let i = 0; i <= samples; i++) {
    const z = -Z_WINDOW + (2 * Z_WINDOW * i) / samples;
    const px = xOf(z);
    const py = yOf(act.f(z));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  if (stat.z.length > 0) drawRug(ctx, inner, histogram(stat.z, [-Z_WINDOW, Z_WINDOW], Math.max(8, Math.floor(inner.w / 3))), off ? palette.red : colour, rugH);
  ctx.restore();
}

/** The output: the loss over its axis, the training samples as bars where they sit. */
function drawOutput(ctx: CanvasRenderingContext2D, box: Box, curve: OutputCurve, samples: readonly { x: number; target: number }[], palette: Palette, byClass: boolean): void {
  drawFrame(ctx, box, palette, false);
  const inner = { x: box.x + 2, y: box.y + 3, w: box.w - 4, h: box.h - 6 };
  const rugH = inner.h * 0.36;
  const plotH = inner.h - rugH - 1;
  const [lo, hi] = curve.domain;
  let top = 0;
  for (let i = 0; i <= 16; i++) top = Math.max(top, curve.f(lo + ((hi - lo) * i) / 16));
  top = Math.min(Math.max(top, 0.5), LOSS_CAP);
  const xOf = (x: number) => inner.x + ((x - lo) / (hi - lo)) * inner.w;
  const yOf = (v: number) => inner.y + plotH - (clamp(v, 0, top) / top) * plotH;
  ctx.save();
  ctx.beginPath();
  ctx.rect(inner.x, inner.y, inner.w, inner.h);
  ctx.clip();
  if (box.w >= 24) {
    ctx.fillStyle = rgba(palette.red, 0.1);
    let start: number | null = null;
    const steps = 24;
    for (let i = 0; i <= steps; i++) {
      const x = lo + ((hi - lo) * i) / steps;
      const flat = Math.abs(curve.df(x)) < 0.05;
      if (flat && start === null) start = x;
      if ((!flat || i === steps) && start !== null) {
        ctx.fillRect(xOf(start), inner.y, xOf(x) - xOf(start), inner.h);
        start = null;
      }
    }
    ctx.strokeStyle = rgba(palette.axis, 0.5);
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(xOf(0), inner.y);
    ctx.lineTo(xOf(0), inner.y + inner.h);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = palette.blue;
  ctx.lineWidth = box.w >= 32 ? 1.4 : 1.1;
  ctx.beginPath();
  const n = 32;
  for (let i = 0; i <= n; i++) {
    const x = lo + ((hi - lo) * i) / n;
    const px = xOf(x);
    const py = yOf(curve.f(x));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  if (samples.length > 0) {
    const bins = Math.max(8, Math.floor(inner.w / 3));
    if (byClass) {
      // One rug per class, stacked, so the class still short of the margin shows.
      const classes = Array.from(new Set(samples.map((s) => s.target))).sort();
      const stacks = classes.map((c) => histogram(samples.filter((s) => s.target === c).map((s) => s.x), curve.domain, bins));
      const totals = stacks[0].map((_, i) => stacks.reduce((sum, h) => sum + h[i], 0));
      const max = Math.max(1, ...totals);
      const barW = inner.w / bins;
      for (let i = 0; i < bins; i++) {
        let base = inner.y + inner.h;
        stacks.forEach((h, c) => {
          if (h[i] === 0) return;
          const height = Math.max(1, (h[i] / max) * rugH);
          ctx.fillStyle = rgba(categorical(palette, classes[c]), 0.8);
          ctx.fillRect(inner.x + i * barW + 0.5, base - height, Math.max(1, barW - 1), height);
          base -= height;
        });
      }
    } else drawRug(ctx, inner, histogram(samples.map((s) => s.x), curve.domain, bins), palette.orange, rugH);
  }
  ctx.restore();
}

function drawInput(ctx: CanvasRenderingContext2D, box: Box, label: string, palette: Palette): void {
  const r = Math.max(6, box.w * 0.3);
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = rgba(palette.surfaceAlt, 0.8);
  ctx.fill();
  ctx.strokeStyle = palette.border;
  ctx.lineWidth = 1;
  ctx.stroke();
  if (r >= 8) {
    ctx.font = '600 ' + Math.round(clamp(r * 0.95, 8, 12)) + 'px ' + MONO_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, cx, cy + 0.5);
  }
}

function gradText(g: number): string {
  if (!(g > 0)) return '0';
  return g >= 0.01 ? g.toFixed(2) : g.toExponential(0).replace('e-', 'e−');
}

/** The gradient a layer's weights received, on a fixed log scale so layers and epochs compare. */
function drawGradient(ctx: CanvasRenderingContext2D, zone: Box, g: number, colour: string, palette: Palette, vertical: boolean, lit: boolean): void {
  const t = clamp((Math.log10(Math.max(g, 1e-7)) + 6) / 7, 0, 1);
  ctx.font = (lit ? '600 ' : '') + '9px ' + MONO_STACK;
  ctx.fillStyle = lit ? palette.text : palette.textMuted;
  if (vertical) {
    const barW = Math.max(14, zone.w - 42);
    ctx.fillStyle = rgba(colour, 0.25);
    ctx.fillRect(zone.x, zone.y + zone.h / 2 - 3, barW, 6);
    ctx.fillStyle = rgba(colour, 0.95);
    ctx.fillRect(zone.x, zone.y + zone.h / 2 - 3, Math.max(1, t * barW), 6);
    ctx.fillStyle = lit ? palette.text : palette.textMuted;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('∇ ' + gradText(g), zone.x + barW + 4, zone.y + zone.h / 2 + 0.5);
    return;
  }
  ctx.fillStyle = rgba(colour, 0.25);
  ctx.fillRect(zone.x, zone.y + 2, zone.w, 5);
  ctx.fillStyle = rgba(colour, 0.95);
  ctx.fillRect(zone.x, zone.y + 2, Math.max(1, t * zone.w), 5);
  ctx.fillStyle = lit ? palette.text : palette.textMuted;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText('∇ ' + gradText(g), zone.x + zone.w / 2, zone.y + zone.h - 3);
}

function wirePath(from: Box, to: Box, vertical: boolean): EdgePath {
  const start = vertical ? { x: from.x + from.w / 2, y: from.y + from.h + 1 } : { x: from.x + from.w + 1, y: from.y + from.h / 2 };
  const end = vertical ? { x: to.x + to.w / 2, y: to.y - 1 } : { x: to.x - 1, y: to.y + to.h / 2 };
  const c1 = vertical ? { x: start.x, y: (start.y + end.y) / 2 } : { x: (start.x + end.x) / 2, y: start.y };
  const c2 = vertical ? { x: end.x, y: (start.y + end.y) / 2 } : { x: (start.x + end.x) / 2, y: end.y };
  return { start, c1, c2, end, angle: 0, backwards: false, label: { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 } };
}

/* ---------------- the diagram ---------------- */

export interface NetworkDiagramProps {
  state: TrainerState;
  act: ActivationFn;
  lossLabel: string;
  stats: NetStats;
  classify: boolean;
  hover: DiagramTarget | null;
  open: DiagramTarget | null;
  spot: Spot | null;
  onHover: (target: DiagramTarget | null) => void;
  onOpen: (target: DiagramTarget | null) => void;
  description: string;
  redrawKey: string;
}

interface Hit {
  box: Box;
  target: DiagramTarget;
}

const LOSS: DiagramTarget = { kind: 'function', which: 'loss' };
const ACT: DiagramTarget = { kind: 'function', which: 'activation' };

export function NetworkDiagram({ state, act, lossLabel, stats, classify, hover, open, spot, onHover, onOpen, description, redrawKey }: NetworkDiagramProps) {
  const hitsRef = useRef<Hit[]>([]);
  const sizes = state.spec.sizes;
  const outputIndex = sizes.length - 1;
  const focus = open ?? hover;

  const targets = useMemo<DiagramTarget[]>(() => {
    const list: DiagramTarget[] = [{ kind: 'layer', index: 0 }];
    for (let k = 1; k < outputIndex; k++) {
      list.push({ kind: 'layer', index: k });
      for (let u = 0; u < sizes[k]; u++) list.push({ kind: 'unit', index: u, layer: k });
    }
    if (outputIndex > 1) list.push(ACT);
    list.push(LOSS);
    return list;
  }, [sizes, outputIndex]);
  const handleKey = useDiagramKeys<DiagramTarget>({
    targets,
    cursor: focus,
    same: sameTarget,
    onCursor: onHover,
    onOpen,
    onClose: () => {
      onOpen(null);
      onHover(null);
    },
  });

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const L = layout(width, height, sizes);
      const hits: Hit[] = [];
      const layers = state.net.layers;
      let maxAbs = 1e-6;
      for (const layer of layers) for (const w of layer.W) maxAbs = Math.max(maxAbs, Math.abs(w));

      const focusUnit = focus && focus.kind === 'unit' ? { layer: unitLayer(focus), index: focus.index } : null;
      const focusLayer = focus && focus.kind === 'layer' ? focus.index : focus && focus.kind === 'function' && focus.which === 'loss' ? outputIndex : -1;
      const focusAct = focus !== null && focus.kind === 'function' && focus.which === 'activation';

      // Wires: every weight, thick and bright by size, coloured by sign; the focused unit's own stay lit.
      for (let k = 1; k < L.slots.length; k++) {
        const layer = layers[k - 1];
        const from = L.slots[k - 1];
        const to = L.slots[k];
        for (let o = 0; o < layer.outSize; o++) {
          for (let i = 0; i < layer.inSize; i++) {
            const w = layer.W[o * layer.inSize + i];
            const size = Math.abs(w) / maxAbs;
            let emphasis: WireEmphasis = 'plain';
            if (focusUnit) {
              const into = focusUnit.layer === k && focusUnit.index === o;
              const outOf = focusUnit.layer === k - 1 && focusUnit.index === i;
              emphasis = into || outOf ? 'lit' : 'dim';
            } else if (focusLayer >= 0) emphasis = focusLayer === k ? 'lit' : 'dim';
            strokeWire(ctx, wirePath(from.boxes[i], to.boxes[o], L.vertical), {
              colour: w >= 0 ? palette.positive : palette.negative,
              alpha: 0.15 + 0.55 * size,
              width: 0.6 + 2.2 * size,
              emphasis,
            });
          }
        }
      }

      // The units.
      L.slots.forEach((slot, k) => {
        const hidden = k > 0 && k < outputIndex;
        const colour = k === 0 ? palette.textMuted : k === outputIndex ? palette.text : layerColour(palette, k - 1);
        slot.boxes.forEach((box, u) => {
          if (k === 0) {
            drawInput(ctx, box, sizes[0] === 1 ? 'x' : 'x' + sub(u + 1), palette);
            hits.push({ box, target: { kind: 'layer', index: 0 } });
          } else if (hidden) {
            drawUnit(ctx, box, act, stats.units[k - 1][u], colour, palette);
            hits.push({ box, target: { kind: 'unit', index: u, layer: k } });
          } else {
            drawOutput(ctx, box, stats.curve, stats.outputs[u] ?? [], palette, classify);
            hits.push({ box, target: LOSS });
          }
        });
        // The symbol and, past the units, the gradient the layer received.
        const symbol = k === 0 ? 'x' : k === outputIndex ? 'ŷ' : 'h' + sub(k);
        const lit = focusLayer === k;
        ctx.font = (lit ? '600 ' : '') + '11px ' + MONO_STACK;
        ctx.fillStyle = lit ? palette.accent : palette.textMuted;
        ctx.textAlign = L.vertical ? 'left' : 'center';
        ctx.textBaseline = L.vertical ? 'middle' : 'alphabetic';
        ctx.fillText(symbol, slot.label.x, slot.label.y);
        const labelBox: Box = L.vertical ? { x: PAD, y: slot.label.y - 12, w: SIDE_L - PAD, h: 24 } : { x: slot.bounds.x, y: slot.label.y - 12, w: slot.bounds.w, h: 16 };
        hits.push({ box: labelBox, target: k === outputIndex ? LOSS : { kind: 'layer', index: k } });
        if (slot.grad && state.epoch > 0) {
          drawGradient(ctx, slot.grad, state.layers[k - 1]?.gradNorm ?? 0, colour, palette, L.vertical, lit);
          hits.push({ box: slot.grad, target: k === outputIndex ? LOSS : { kind: 'layer', index: k } });
        }
        if (k === outputIndex && !L.vertical) {
          ctx.font = '9px ' + MONO_STACK;
          ctx.fillStyle = palette.textFaint;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText(lossLabel, slot.bounds.x + slot.bounds.w / 2, slot.bounds.y - 6);
        }
      });

      // The activation's name, once, beside every hidden layer.
      if (L.act) {
        ctx.save();
        ctx.font = (focusAct ? '600 ' : '') + '10px ' + MONO_STACK;
        ctx.fillStyle = focusAct ? palette.accent : palette.textFaint;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (L.vertical) {
          ctx.translate(L.act.x + L.act.w / 2, L.act.y + L.act.h / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.fillText(act.label.toLowerCase(), 0, 0);
        } else ctx.fillText(act.label.toLowerCase(), L.act.x + L.act.w / 2, L.act.y + L.act.h / 2);
        ctx.restore();
        hits.push({ box: L.act, target: ACT });
      }
      if (L.vertical) {
        const out = L.slots[outputIndex];
        ctx.font = '9px ' + MONO_STACK;
        ctx.fillStyle = palette.textFaint;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(lossLabel, out.bounds.x + out.bounds.w + 8, out.label.y);
      }
      hitsRef.current = hits;

      // Outline: the hovered or open thing, else the lesson's spot with its label.
      const boxOf = (target: DiagramTarget): Box | null => {
        switch (target.kind) {
          case 'unit': {
            const slot = L.slots[unitLayer(target)];
            return slot && slot.index < outputIndex ? (slot.boxes[target.index] ?? null) : null;
          }
          case 'layer':
            return L.slots[target.index]?.bounds ?? null;
          case 'function':
            return target.which === 'loss' ? L.slots[outputIndex].bounds : L.act;
          default:
            return null;
        }
      };
      const outline = (box: Box, mode: 'hover' | 'open' | 'spot') => strokeOutline(ctx, () => roundRect(ctx, box.x - 2, box.y - 2, box.w + 4, box.h + 4, 4), mode, palette);
      const focusBox = focus ? boxOf(focus) : null;
      if (focusBox) outline(focusBox, open && sameTarget(open, focus) ? 'open' : 'hover');
      else if (spot) {
        const box = boxOf(spot.target);
        if (box) {
          outline(box, 'spot');
          const size = measurePlate(ctx, [spot.label]);
          const place = placePlate(size, box, width, height);
          drawHoverPlate(ctx, [spot.label], place.left, place.top, palette);
        }
      }
    },
    [sizes, outputIndex, state, act, lossLabel, stats, classify, focus, open, spot],
  );

  const locate = useCallback((pos: { x: number; y: number }): DiagramTarget | null => {
    for (const hit of hitsRef.current) {
      const b = hit.box;
      if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) return hit.target;
    }
    return null;
  }, []);

  return (
    <div
      className="mlx-arch__stage"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys move between the units and layers, Enter opens one.'}
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
          onOpen(found && sameTarget(found, open) ? null : found);
        }}
        onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
        onPointerLeave={() => onHover(null)}
        redrawKey={redrawKey + '|' + targetKey(hover) + '|' + targetKey(open) + '|' + (spot ? targetKey(spot.target) + spot.label : '')}
      />
    </div>
  );
}
