/**
 * The model in one picture, the way Transformer Explainer draws it: each token is a row that runs left to right through
 * the embedding, the attention, the feed-forward block and the output. Click a part and it opens in place, step by step.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt } from '../../lib/math/stats';
import { headWidth } from '../../lib/ml/transformer';
import type { LayerCache, NormCache, TransformerCache, TransformerSpec } from '../../lib/ml/transformer';
import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { mix, readableOn, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawChip, drawStrip, hatch, headColour, scaleOf, weightColour } from './draw';
import type { Probe } from './model';
import { tokenName } from './model';

type Rect = { x: number; y: number; w: number; h: number };

const inside = (p: { x: number; y: number }, r: Rect) => p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;

/* ---------------- parts ---------------- */

/** What a click opens; 'none' is the whole picture. */
export type Part = 'none' | 'embed' | 'norm1' | 'attention' | 'norm2' | 'ffn' | 'final' | 'output';

export const PART_NAMES: Record<Part, string> = {
  none: 'The whole model',
  embed: 'Embedding',
  norm1: 'Layer norm before attention',
  attention: 'Multi-head attention',
  norm2: 'Layer norm before the feed-forward block',
  ffn: 'Feed forward',
  final: 'Final layer norm',
  output: 'Output probabilities',
};

/** A box named by a lesson or an older link, as the part that shows it. */
export function partOf(id: string): Part {
  switch (id) {
    case 'embed':
    case 'position':
      return 'embed';
    case 'norm1':
    case 'norm2':
    case 'final':
    case 'attention':
    case 'ffn':
    case 'output':
      return id;
    case 'add1':
      return 'attention';
    case 'add2':
      return 'ffn';
    case 'linear':
    case 'softmax':
      return 'output';
    default:
      return 'none';
  }
}

function partsOf(spec: TransformerSpec): Part[] {
  const out: Part[] = ['embed'];
  if (spec.norm) out.push('norm1');
  out.push('attention');
  if (spec.ffn > 0) {
    if (spec.norm) out.push('norm2');
    out.push('ffn');
  }
  if (spec.norm) out.push('final');
  out.push('output');
  return out;
}

/* ---------------- the task and each part in words ---------------- */

interface TaskStory {
  title: string;
  what: string;
  attention: string;
  ffn: string;
}

/** The task in plain words, and what attention and the feed-forward block have to do for it. */
function taskStory(data: SequenceDataset, spec: TransformerSpec): TaskStory {
  const n = data.size;
  if (data.task === 'letters') {
    return {
      title: 'Next letter',
      what: 'Read a sentence of a tiny language one letter at a time, like “the big dog often says woof.”, and guess every next letter. Most letters are spelling; the sound after says depends on the animal many letters back. Guess, append, repeat, and the model writes whole sentences, the way a large language model writes.',
      attention: 'Here the letters after says look back to the animal’s letters.',
      ffn: 'Here it turns what attention found into the next letter.',
    };
  }
  if (data.task === 'count') {
    return {
      title: 'Count',
      what: 'Read ' + n + ' digits and, at each one, say how many times that digit appears in the whole string. Where a digit sits does not matter, only which digits match it.',
      attention: 'Here each digit gathers the digits that match it.',
      ffn: 'Here it turns what attention gathered into a count.',
    };
  }
  if (data.task === 'animals') {
    return {
      title: 'Animal sounds',
      what: 'Predict the next word of sentences like “the old dog often says woof”. Most next words are a guess, but after says the sound is certain once you know the animal a few words back.',
      attention: 'Here says looks back at the animal, the word that decides the sound.',
      ffn: 'Here it turns the animal it found into the right sound.',
    };
  }
  return {
    title: 'Reverse',
    what: 'Read ' + n + ' digits and write them backwards. The answer at each position is the digit opposite it, so every position has to find its mirror and copy it.',
    attention: spec.causal ? 'Each digit would look for the digit opposite it, but the mask only lets it look back.' : 'Here each digit finds the digit opposite it and copies it.',
    ffn: 'Here it tidies what was copied into a clean answer.',
  };
}

/** The line under the picture: the task, or what the open (or pointed at) part does. */
export function partNote(part: Part, data: SequenceDataset, spec: TransformerSpec): { title: string; text: string } {
  const story = taskStory(data, spec);
  const d = spec.width;
  switch (part) {
    case 'embed':
      return {
        title: 'Embedding.',
        text:
          'Each token looks up its own row of the table E: ' + d + ' numbers learned in training. ' +
          (spec.positions === 'sinusoidal'
            ? 'Then p is added, a vector that depends only on the place: each pair of numbers is a clock hand turning at a fixed speed, fast on the left, so every place gets its own pattern and equal tokens now differ.'
            : spec.positions === 'learned'
              ? 'Then p is added, one trained vector per place, so equal tokens at different places differ.'
              : 'Nothing is added for the place, so a token looks the same wherever it sits and order is invisible.'),
      };
    case 'norm1':
    case 'norm2':
    case 'final':
      return {
        title: 'Layer norm.',
        text: 'Every token on its own: its numbers are moved to centre 0 and scaled to spread 1 (the band), then stretched by γ and shifted by β. The next block always reads numbers of one size, however big the stream has grown.',
      };
    case 'attention':
      return {
        title: 'Attention.',
        text:
          'Each token makes a query Q (what it looks for), a key K (what it offers) and a value V (what it passes on). Every query is scored against every key' +
          (spec.causal ? ', later tokens hidden' : '') +
          '; softmax turns each row of scores into weights that add to 1, and the token takes that blend of values. ' + story.attention,
      };
    case 'ffn':
      return {
        title: 'Feed forward.',
        text: 'The same small network runs on every token alone: widen to ' + spec.ffn + ' units, ReLU sets the negative ones to 0, narrow back to ' + d + ' and add. Attention moved information between tokens; this computes with it. ' + story.ffn,
      };
    case 'output':
      return {
        title: 'Output.',
        text: 'One score per possible ' + (data.nextToken ? 'next word' : 'answer') + ', z = x̂ W_U. Softmax turns the scores into shares that add to 1; the tallest bar is the model’s call, and a green outline marks the right answer.',
      };
    default:
      return {
        title: 'The task: ' + story.title + '.',
        text: story.what + ' Each row follows one token from left to right. Click any part to open it.',
      };
  }
}

/* ---------------- the layout ---------------- */

interface Col {
  x: number;
  w: number;
}

interface Layout {
  width: number;
  height: number;
  top: number;
  bottom: number;
  pitch: number;
  strip: number;
  lab: Col;
  emb: Col;
  ln1: Col | null;
  qkv: Col;
  mat: Col;
  cell: number;
  att: Col;
  ln2: Col | null;
  hid: Col | null;
  ffo: Col | null;
  lnf: Col | null;
  out: Col;
  block: Rect;
  parts: Array<{ part: Part; rect: Rect }>;
}

/** The first token row; above it sit the titles, the residual arcs and the key labels. */
const TOP = 90;

function layoutArch(width: number, height: number, spec: TransformerSpec, T: number, words: boolean): Layout {
  const pitch = Math.max(10, Math.min(44, (height - TOP - 44) / T));
  const strip = Math.max(5, Math.min(14, pitch * 0.46));
  const vec = Math.max(28, Math.min(52, spec.width * 3));
  type Item = { key: string; w: number; fixed?: boolean; gap?: boolean };
  const items: Item[] = [
    { key: 'lab', w: words ? 50 : 26, fixed: true },
    { key: 'g', w: 12, gap: true },
    { key: 'emb', w: vec },
    { key: 'g', w: 16, gap: true },
  ];
  if (spec.norm) items.push({ key: 'ln1', w: 10, fixed: true }, { key: 'g', w: 14, gap: true });
  items.push(
    { key: 'qkv', w: Math.max(24, Math.min(40, headWidth(spec) * 4)) },
    { key: 'g', w: 26, gap: true },
    { key: 'mat', w: Math.min(220, T * Math.min(pitch, 18)) },
    { key: 'g', w: 22, gap: true },
    { key: 'att', w: vec },
  );
  if (spec.ffn > 0) {
    items.push({ key: 'g', w: 16, gap: true });
    if (spec.norm) items.push({ key: 'ln2', w: 10, fixed: true }, { key: 'g', w: 14, gap: true });
    items.push({ key: 'hid', w: Math.max(44, Math.min(84, spec.ffn * 1.2)) }, { key: 'g', w: 12, gap: true }, { key: 'ffo', w: vec });
  }
  items.push({ key: 'g', w: 16, gap: true });
  if (spec.norm) items.push({ key: 'lnf', w: 10, fixed: true }, { key: 'g', w: 14, gap: true });
  items.push({ key: 'out', w: words ? 100 : 76, fixed: true });

  // Room to spare lets the parts grow a quarter and the flows take the rest; too little shrinks everything but the labels.
  const margin = 6;
  const fixed = items.reduce((s, i) => s + (i.fixed ? i.w : 0), 0) + margin * 2;
  const partsW = items.reduce((s, i) => s + (i.fixed || i.gap ? 0 : i.w), 0);
  const gapsW = items.reduce((s, i) => s + (i.gap ? i.w : 0), 0);
  const scale = (width - fixed) / (partsW + gapsW);
  const grow = Math.min(scale, 1.25);
  const gapScale = scale >= 1 ? (width - fixed - partsW * grow) / gapsW : scale;
  const cols: Record<string, Col> = {};
  let x = margin;
  for (const item of items) {
    const w = item.fixed ? item.w : item.gap ? item.w * gapScale : item.w * (scale >= 1 ? grow : scale);
    if (!item.gap) cols[item.key] = { x, w };
    x += w;
  }
  const bottom = TOP + T * pitch;
  const first = cols.ln1 ?? cols.qkv;
  const last = cols.ffo ?? cols.att;
  const block = { x: first.x - 10, y: 24, w: last.x + last.w + 10 - (first.x - 10), h: bottom + 36 - 24 };
  const tall = (c: Col, pad = 6, y = 34): Rect => ({ x: c.x - pad, y, w: c.w + pad * 2, h: bottom + 6 - y });
  const parts: Layout['parts'] = [{ part: 'embed', rect: tall(cols.emb) }];
  if (cols.ln1) parts.push({ part: 'norm1', rect: tall(cols.ln1, 5, TOP - 22) });
  parts.push({ part: 'attention', rect: { x: cols.qkv.x - 6, y: 34, w: cols.att.x + cols.att.w - cols.qkv.x + 12, h: bottom + 6 - 34 } });
  if (cols.ln2) parts.push({ part: 'norm2', rect: tall(cols.ln2, 5, TOP - 22) });
  if (cols.hid && cols.ffo) parts.push({ part: 'ffn', rect: { x: cols.hid.x - 6, y: 34, w: cols.ffo.x + cols.ffo.w - cols.hid.x + 12, h: bottom + 6 - 34 } });
  if (cols.lnf) parts.push({ part: 'final', rect: tall(cols.lnf, 5, TOP - 22) });
  parts.push({ part: 'output', rect: tall(cols.out) });
  return {
    width,
    height,
    top: TOP,
    bottom,
    pitch,
    strip,
    lab: cols.lab,
    emb: cols.emb,
    ln1: cols.ln1 ?? null,
    qkv: cols.qkv,
    mat: cols.mat,
    cell: cols.mat.w / T,
    att: cols.att,
    ln2: cols.ln2 ?? null,
    hid: cols.hid ?? null,
    ffo: cols.ffo ?? null,
    lnf: cols.lnf ?? null,
    out: cols.out,
    block,
    parts,
  };
}

/* ---------------- drawing pieces ---------------- */

interface Scene {
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  probe: Probe;
  layer: number;
  head: number;
  tokens: string[];
}

/** A clickable glyph the canvas draws: layer and head arrows, the card's buttons. */
interface Button {
  rect: Rect;
  run: () => void;
}

const rowY = (L: Layout, t: number) => L.top + (t + 0.5) * L.pitch;

/** One head's slice of a T × d row. */
function headRow(values: Float64Array, t: number, d: number, dh: number, head: number): Float64Array {
  return values.subarray(t * d + head * dh, t * d + (head + 1) * dh);
}

/** A band between two columns at a row, thicker or thinner at each end. */
function band(ctx: CanvasRenderingContext2D, x0: number, y0: number, h0: number, x1: number, y1: number, h1: number, fill: string): void {
  const mx = (x0 + x1) / 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0 - h0 / 2);
  ctx.bezierCurveTo(mx, y0 - h0 / 2, mx, y1 - h1 / 2, x1, y1 - h1 / 2);
  ctx.lineTo(x1, y1 + h1 / 2);
  ctx.bezierCurveTo(mx, y1 + h1 / 2, mx, y0 + h0 / 2, x0, y0 + h0 / 2);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, colour: string, font: string, align: CanvasTextAlign = 'center', maxWidth?: number): void {
  ctx.font = font;
  ctx.fillStyle = colour;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  if (maxWidth) ctx.fillText(text, x, y, maxWidth);
  else ctx.fillText(text, x, y);
}

/** A small square button with a glyph; returns its rect. */
function glyphButton(ctx: CanvasRenderingContext2D, glyph: string, x: number, y: number, palette: Palette, size = 18): Rect {
  ctx.save();
  roundRect(ctx, x, y, size, size, 4);
  ctx.fillStyle = palette.surface;
  ctx.fill();
  ctx.strokeStyle = rgba(palette.textFaint, 0.7);
  ctx.lineWidth = 1;
  ctx.stroke();
  label(ctx, glyph, x + size / 2, y + size / 2 + 0.5, palette.text, '600 12px ' + FONT_STACK);
  ctx.restore();
  return { x, y, w: size, h: size };
}

/** A title with ⊕ after it: this part opens. */
function partTitle(ctx: CanvasRenderingContext2D, text: string, cx: number, y: number, palette: Palette, state: 'hover' | 'spot' | null): Rect {
  ctx.save();
  ctx.font = '600 11px ' + FONT_STACK;
  const w = ctx.measureText(text).width;
  const x0 = cx - (w + 14) / 2;
  const colour = state ? palette.accent : palette.text;
  label(ctx, text, x0, y, colour, '600 11px ' + FONT_STACK, 'left');
  const gx = x0 + w + 9;
  ctx.beginPath();
  ctx.arc(gx, y, 4.5, 0, Math.PI * 2);
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1.1;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(gx - 2.4, y);
  ctx.lineTo(gx + 2.4, y);
  ctx.moveTo(gx, y - 2.4);
  ctx.lineTo(gx, y + 2.4);
  ctx.stroke();
  ctx.restore();
  return { x: x0 - 4, y: y - 9, w: w + 22, h: 18 };
}

/** A token at the head of its row: a chip, or plain text when the rows are too tight for one. */
function drawToken(ctx: CanvasRenderingContext2D, L: Layout, text: string, t: number, tone: 'accent' | 'start' | 'plain', palette: Palette, words: boolean): void {
  const y = rowY(L, t);
  if (L.pitch >= 15) {
    drawChip(ctx, text, L.lab.x + L.lab.w / 2, y, L.lab.w - 2, Math.min(L.pitch - 4, 20), palette, tone, words ? 9 : 10);
    return;
  }
  label(ctx, text, L.lab.x + L.lab.w / 2, y, tone === 'accent' ? palette.accent : tone === 'start' ? palette.textMuted : palette.text, '600 ' + Math.max(8, Math.min(10, L.pitch - 1)) + 'px ' + MONO_STACK);
}

/** How far a pair's hand turns per place, in degrees. */
function turnOf(k: number, d: number): string {
  const deg = 180 / Math.PI / Math.pow(10000, (2 * k) / d);
  return (deg >= 10 ? Math.round(deg).toString() : deg >= 1 ? deg.toFixed(1) : deg.toFixed(2).replace(/^0/, '')) + '°';
}

/* ---------------- the whole picture ---------------- */

function drawBase(
  ctx: CanvasRenderingContext2D,
  L: Layout,
  s: Scene,
  palette: Palette,
  hover: { part: Part | null; row: number | null; spot: Part | null },
  buttons: Button[],
  onPick: (layer: number, head: number) => void,
): void {
  const { spec, cache, data, probe, layer, head, tokens } = s;
  const T = cache.length;
  const d = spec.width;
  const dh = headWidth(spec);
  const Lc: LayerCache = cache.layers[Math.min(layer, cache.layers.length - 1)];
  const hue = headColour(palette, head);
  const Q = palette.blue;
  const K = palette.red;
  const V = palette.green;
  const grey = rgba(palette.textFaint, 0.16);
  const sub = Math.max(3, Math.min(8, L.pitch * 0.25));
  ctx.save();

  if (hover.row !== null) {
    ctx.fillStyle = rgba(palette.accent, 0.07);
    ctx.fillRect(0, L.top + hover.row * L.pitch, L.width, L.pitch);
  }

  // The transformer block, stacked when it repeats.
  const b = L.block;
  for (let k = Math.min(2, spec.layers - 1); k >= 1; k--) {
    roundRect(ctx, b.x + 4 * k, b.y + 4 * k, b.w, b.h, 10);
    ctx.fillStyle = palette.surface;
    ctx.fill();
    ctx.strokeStyle = rgba(palette.textFaint, 0.35);
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  roundRect(ctx, b.x, b.y, b.w, b.h, 10);
  ctx.fillStyle = palette.surface;
  ctx.fill();
  ctx.fillStyle = rgba(palette.textFaint, 0.07);
  ctx.fill();
  ctx.strokeStyle = rgba(palette.textFaint, 0.5);
  ctx.lineWidth = 1;
  ctx.stroke();
  const blockTitle = 'Transformer block' + (spec.layers > 1 ? ' ' + (layer + 1) + ' of ' + spec.layers : '');
  label(ctx, blockTitle, b.x + b.w / 2, 12, palette.textMuted, '600 10px ' + FONT_STACK);
  if (spec.layers > 1) {
    ctx.font = '600 10px ' + FONT_STACK;
    const half = ctx.measureText(blockTitle).width / 2;
    const prev = glyphButton(ctx, '‹', b.x + b.w / 2 - half - 22, 3, palette, 16);
    const next = glyphButton(ctx, '›', b.x + b.w / 2 + half + 6, 3, palette, 16);
    buttons.push({ rect: prev, run: () => onPick((layer + spec.layers - 1) % spec.layers, head) }, { rect: next, run: () => onPick((layer + 1) % spec.layers, head) });
  }

  // The residual paths: over the top of each part, into its add.
  const residual = (x0: number, x1: number) => {
    const yTop = 62;
    ctx.strokeStyle = rgba(palette.textMuted, 0.8);
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(x0, L.top - 2);
    ctx.lineTo(x0, yTop + 6);
    ctx.quadraticCurveTo(x0, yTop, x0 + 6, yTop);
    ctx.lineTo(x1 - 6, yTop);
    ctx.quadraticCurveTo(x1, yTop, x1, yTop + 6);
    ctx.lineTo(x1, L.top - 19);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, L.top - 18);
    ctx.lineTo(x1 - 3.5, L.top - 24);
    ctx.lineTo(x1 + 3.5, L.top - 24);
    ctx.closePath();
    ctx.fillStyle = rgba(palette.textMuted, 0.8);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x1, L.top - 11, 6, 0, Math.PI * 2);
    ctx.fillStyle = palette.surface;
    ctx.fill();
    ctx.strokeStyle = palette.text;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1 - 3, L.top - 11);
    ctx.lineTo(x1 + 3, L.top - 11);
    ctx.moveTo(x1, L.top - 14);
    ctx.lineTo(x1, L.top - 8);
    ctx.stroke();
    label(ctx, 'residual', (x0 + x1) / 2, yTop - 7, palette.textFaint, '9px ' + FONT_STACK);
  };
  const embRight = L.emb.x + L.emb.w;
  residual(embRight + ((L.ln1 ?? L.qkv).x - embRight) / 2, L.att.x + L.att.w / 2);
  if (L.hid && L.ffo) {
    const attRight = L.att.x + L.att.w;
    residual(attRight + ((L.ln2 ?? L.hid).x - attRight) / 2, L.ffo.x + L.ffo.w / 2);
  }

  // The part titles.
  const state = (part: Part) => (hover.part === part ? 'hover' : hover.spot === part ? 'spot' : null);
  partTitle(ctx, 'Embedding', L.emb.x + L.emb.w / 2, 44, palette, state('embed'));
  partTitle(ctx, spec.causal ? 'Masked attention' : 'Attention', (L.qkv.x + L.att.x + L.att.w) / 2, 44, palette, state('attention'));
  if (L.hid && L.ffo) partTitle(ctx, 'Feed forward', (L.hid.x + L.ffo.x + L.ffo.w) / 2, 44, palette, state('ffn'));
  partTitle(ctx, 'Output', L.out.x + L.out.w / 2, 44, palette, state('output'));

  // The flows, one per token.
  const lnBar = (c: Col | null) => c;
  const scores = { x: scaleOf(cache.x0), q: 0, k: 0, v: 0, mid: scaleOf(Lc.mid), act: Lc.act ? scaleOf(Lc.act) : 1, out: scaleOf(Lc.out) };
  for (let t = 0; t < T; t++) {
    scores.q = Math.max(scores.q, scaleOf(headRow(Lc.q, t, d, dh, head)));
    scores.k = Math.max(scores.k, scaleOf(headRow(Lc.k, t, d, dh, head)));
    scores.v = Math.max(scores.v, scaleOf(headRow(Lc.v, t, d, dh, head)));
  }
  const matX = (k: number) => L.mat.x + (k + 0.5) * L.cell;
  const qkvRight = L.qkv.x + L.qkv.w;
  for (let t = 0; t < T; t++) {
    const y = rowY(L, t);
    const dim = hover.row !== null && hover.row !== t;
    const a = dim ? 0.45 : 1;
    ctx.globalAlpha = a;
    band(ctx, L.lab.x + L.lab.w, y, L.strip, L.emb.x, y, L.strip, grey);
    const into = lnBar(L.ln1);
    if (into) band(ctx, embRight, y, L.strip, into.x, y, L.strip, grey);
    const from = into ? into.x + into.w : embRight;
    const ys = [y - sub - 1, y, y + sub + 1];
    [Q, K, V].forEach((c, i) => band(ctx, from, y + (i - 1) * (L.strip / 3), L.strip / 3, L.qkv.x, ys[i], sub, rgba(c, 0.2)));
    band(ctx, qkvRight, ys[0], sub, L.mat.x, y, sub, rgba(Q, 0.22));
    band(ctx, L.mat.x + L.mat.w, y, L.strip * 0.8, L.att.x, y, L.strip, rgba(hue, 0.22));
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = rgba(K, 0.45);
    ctx.beginPath();
    ctx.moveTo(qkvRight, ys[1]);
    ctx.bezierCurveTo(qkvRight + (L.mat.x - qkvRight) * 0.9, ys[1], matX(t), L.top + 12, matX(t), L.top - 18);
    ctx.stroke();
    ctx.strokeStyle = rgba(V, 0.45);
    ctx.beginPath();
    ctx.moveTo(qkvRight, ys[2]);
    ctx.bezierCurveTo(qkvRight + (L.mat.x - qkvRight) * 0.9, ys[2], matX(t), L.bottom + 26, matX(t), L.bottom + 6);
    ctx.stroke();
    const attRight = L.att.x + L.att.w;
    if (L.hid && L.ffo) {
      if (L.ln2) {
        band(ctx, attRight, y, L.strip, L.ln2.x, y, L.strip, grey);
        band(ctx, L.ln2.x + L.ln2.w, y, L.strip, L.hid.x, y, L.strip, grey);
      } else band(ctx, attRight, y, L.strip, L.hid.x, y, L.strip, grey);
      band(ctx, L.hid.x + L.hid.w, y, L.strip, L.ffo.x, y, L.strip, grey);
    }
    const last = L.ffo ?? L.att;
    const lastRight = last.x + last.w;
    if (L.lnf) {
      band(ctx, lastRight, y, L.strip, L.lnf.x, y, L.strip, grey);
      band(ctx, L.lnf.x + L.lnf.w, y, L.strip, L.out.x, y, L.strip * 0.6, grey);
    } else band(ctx, lastRight, y, L.strip, L.out.x, y, L.strip * 0.6, grey);
    ctx.globalAlpha = 1;
  }

  // The layer norms as thin bars the flows pass through.
  for (const c of [L.ln1, L.ln2, L.lnf]) {
    if (!c) continue;
    roundRect(ctx, c.x, L.top - 4, c.w, L.bottom - L.top + 8, 4);
    ctx.fillStyle = palette.surface;
    ctx.fill();
    ctx.fillStyle = rgba(palette.yellow, 0.35);
    ctx.fill();
    ctx.strokeStyle = rgba(palette.text, 0.35);
    ctx.lineWidth = 1;
    ctx.stroke();
    const part: Part = c === L.ln1 ? 'norm1' : c === L.ln2 ? 'norm2' : 'final';
    label(ctx, 'LN', c.x + c.w / 2, L.top - 12, state(part) ? palette.accent : palette.textMuted, '600 9px ' + FONT_STACK);
  }

  // Captions over the columns.
  const faint = '9px ' + MONO_STACK;
  label(ctx, 'x = e + p', L.emb.x + L.emb.w / 2, L.top - 10, palette.textFaint, faint, 'center', L.emb.w + 12);
  ['Q', 'K', 'V'].forEach((letter, i) => label(ctx, letter, L.qkv.x + ((i + 0.5) * L.qkv.w) / 3, L.top - 10, [Q, K, V][i], '600 10px ' + MONO_STACK));
  if (L.hid) label(ctx, 'ReLU', L.hid.x + L.hid.w / 2, L.top - 10, palette.textFaint, faint);
  label(ctx, data.nextToken ? 'next word' : 'call', L.out.x + 16, L.top - 10, palette.textFaint, faint, 'left');

  // The vectors, one strip per token.
  for (let t = 0; t < T; t++) {
    const y = rowY(L, t);
    ctx.globalAlpha = hover.row !== null && hover.row !== t ? 0.5 : 1;
    drawStrip(ctx, cache.x0.subarray(t * d, (t + 1) * d), L.emb.x, y - L.strip / 2, L.emb.w, L.strip, palette, scores.x);
    const ys = [y - sub - 1, y, y + sub + 1];
    drawStrip(ctx, headRow(Lc.q, t, d, dh, head), L.qkv.x, ys[0] - sub / 2, L.qkv.w, sub, palette, scores.q, { magnitude: true, tint: Q });
    drawStrip(ctx, headRow(Lc.k, t, d, dh, head), L.qkv.x, ys[1] - sub / 2, L.qkv.w, sub, palette, scores.k, { magnitude: true, tint: K });
    drawStrip(ctx, headRow(Lc.v, t, d, dh, head), L.qkv.x, ys[2] - sub / 2, L.qkv.w, sub, palette, scores.v, { magnitude: true, tint: V });
    drawStrip(ctx, Lc.mid.subarray(t * d, (t + 1) * d), L.att.x, y - L.strip / 2, L.att.w, L.strip, palette, scores.mid);
    if (L.hid && L.ffo && Lc.act) {
      drawStrip(ctx, Lc.act.subarray(t * spec.ffn, (t + 1) * spec.ffn), L.hid.x, y - L.strip / 2, L.hid.w, L.strip, palette, scores.act, { magnitude: true });
      drawStrip(ctx, Lc.out.subarray(t * d, (t + 1) * d), L.ffo.x, y - L.strip / 2, L.ffo.w, L.strip, palette, scores.out);
    }
    ctx.globalAlpha = 1;
  }

  // Who reads whom: a dot per query and key, bigger for more weight. Keys above, values below.
  const r0 = Math.max(1.5, Math.min(L.cell, L.pitch) / 2 - 1);
  const weights = Lc.attn.subarray(head * T * T, (head + 1) * T * T);
  for (let q = 0; q < T; q++) {
    const y = rowY(L, q);
    for (let k = 0; k < T; k++) {
      const x = matX(k);
      if (spec.causal && k > q) {
        ctx.fillStyle = rgba(palette.textFaint, 0.18);
        ctx.fillRect(x - 1, y - 1, 2, 2);
        continue;
      }
      const w = weights[q * T + k];
      ctx.beginPath();
      ctx.arc(x, y, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = rgba(palette.textFaint, 0.35);
      ctx.fill();
      if (w > 0.02) {
        ctx.beginPath();
        ctx.arc(x, y, Math.max(1.4, Math.sqrt(w) * r0), 0, Math.PI * 2);
        ctx.fillStyle = weightColour(palette, w, hue);
        ctx.fill();
      }
    }
  }
  const words = data.task === 'animals';
  for (let k = 0; k < T; k++) {
    const x = matX(k);
    const name = tokens[k];
    if (L.cell >= 8) {
      if (words) {
        ctx.save();
        ctx.translate(x, L.top - 5);
        ctx.rotate(-Math.PI / 2.6);
        label(ctx, name, 0, 0, rgba(K, 0.9), '8px ' + MONO_STACK, 'left');
        ctx.restore();
      } else label(ctx, name, x, L.top - 8, K, '600 9px ' + MONO_STACK);
    }
    ctx.fillStyle = rgba(V, 0.8);
    ctx.fillRect(x - 1.5, L.bottom + 4, 3, 4);
  }
  const headText = spec.heads > 1 ? 'head ' + (head + 1) + ' of ' + spec.heads : 'one head';
  const hx = L.mat.x + L.mat.w / 2;
  label(ctx, headText, hx, L.bottom + 20, hue, '600 10px ' + FONT_STACK);
  if (spec.heads > 1) {
    ctx.font = '600 10px ' + FONT_STACK;
    const half = ctx.measureText(headText).width / 2;
    const prev = glyphButton(ctx, '‹', hx - half - 20, L.bottom + 12, palette, 16);
    const next = glyphButton(ctx, '›', hx + half + 4, L.bottom + 12, palette, 16);
    buttons.push({ rect: prev, run: () => onPick(layer, (head + spec.heads - 1) % spec.heads) }, { rect: next, run: () => onPick(layer, (head + 1) % spec.heads) });
  }

  // The output: the call per token and how sure it is.
  const V2 = spec.outVocab;
  const chipW = words ? 46 : 22;
  for (let t = 0; t < T; t++) {
    const y = rowY(L, t);
    const target = probe.targets[t] ?? -1;
    if (target < 0 && !data.nextToken) continue;
    const call = cache.predicted[t];
    const p = cache.probs[t * V2 + call];
    ctx.globalAlpha = hover.row !== null && hover.row !== t ? 0.5 : 1;
    const tone = target < 0 ? 'plain' : probe.scored.includes(t) ? (call === target ? 'right' : 'wrong') : 'plain';
    const name = tokenName(data, call);
    if (L.pitch >= 15) drawChip(ctx, name, L.out.x + chipW / 2 + 2, y, chipW, Math.min(L.pitch - 4, 18), palette, tone, words ? 9 : 10);
    else label(ctx, name, L.out.x + chipW / 2 + 2, y, tone === 'right' ? palette.green : tone === 'wrong' ? palette.orange : palette.text, '600 9px ' + MONO_STACK);
    const bx = L.out.x + chipW + 8;
    const bw = L.out.w - chipW - 34;
    ctx.fillStyle = rgba(palette.textFaint, 0.25);
    ctx.fillRect(bx, y - 2, bw, 4);
    ctx.fillStyle = palette.accent;
    ctx.fillRect(bx, y - 2, bw * p, 4);
    label(ctx, Math.round(p * 100) + '%', bx + bw + 3, y, palette.textMuted, '9px ' + MONO_STACK, 'left');
    ctx.globalAlpha = 1;
  }

  // The tokens themselves, on the left of every row.
  for (let t = 0; t < T; t++) drawToken(ctx, L, tokens[t], t, hover.row === t ? 'accent' : t === 0 ? 'start' : 'plain', palette, words);

  // The part under the pointer, and a lesson's part.
  for (const { part, rect } of L.parts) {
    const st = hover.part === part ? 'hover' : hover.spot === part ? 'spot' : null;
    if (st) strokeOutline(ctx, () => roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 8), st, palette);
  }
  ctx.restore();
}

/* ---------------- an open part ---------------- */

type ColKind = 'strip' | 'op' | 'text' | 'dials' | 'dots' | 'matrix' | 'relu' | 'bars' | 'chips';

interface CardCol {
  kind: ColKind;
  key: string;
  title?: string;
  sub?: string;
  ideal: number;
  min: number;
  values?: Float64Array;
  n?: number;
  scale?: number;
  tint?: string;
  mask?: boolean;
  labels?: string[];
  texts?: string[];
  norm?: NormCache;
  lim?: number;
  after?: boolean;
  /** Rows outlined: the equal tokens on the embedding. */
  marked?: number[];
  /** A cell's name for the hover card: the row's token and the cell index. */
  name?: (t: number, i: number) => string;
}

interface Placed {
  col: CardCol;
  x: number;
  w: number;
}

function cardCols(part: Part, s: Scene, L: Layout, palette: Palette): CardCol[] {
  const { spec, cache, data, layer, head, tokens } = s;
  const T = cache.length;
  const d = spec.width;
  const dh = headWidth(spec);
  const Lc = cache.layers[Math.min(layer, cache.layers.length - 1)];
  const vec = (key: string, title: string, sub: string, values: Float64Array, n = d, extra: Partial<CardCol> = {}): CardCol => ({
    kind: 'strip',
    key,
    title,
    sub,
    values,
    n,
    scale: scaleOf(values),
    ideal: Math.max(40, Math.min(110, n * 4)),
    min: Math.max(24, n * 1.2),
    name: (t, i) => title + ' · ' + tokens[t] + ' · dim ' + (i + 1),
    ...extra,
  });
  // Between steps only a glyph; the step's caption names what was done.
  const op = (glyph: string): CardCol => ({ kind: 'op', key: 'op', texts: [glyph], ideal: 16, min: 12 });
  const matrixW = T * Math.min(L.pitch - 2, 24);
  switch (part) {
    case 'embed': {
      // The first token that appears twice: its rows of e match, its rows of x do not.
      const twice = cache.tokens.find((tok, t) => cache.tokens.indexOf(tok) !== t);
      const marked = twice === undefined ? [] : cache.tokens.map((tok, t) => (tok === twice ? t : -1)).filter((t) => t >= 0);
      const cols: CardCol[] = [
        { kind: 'text', key: 'id', title: 'token', sub: 'row of E', texts: cache.tokens.map((tok) => '#' + tok), ideal: 34, min: 26 },
        op('→'),
        vec('e', 'token vector e', marked.length ? 'equal tokens alike' : 'E[token]', cache.embed, d, { marked }),
      ];
      if (spec.positions === 'none') return [...cols, op('='), vec('x', 'x = e', 'no place added', cache.x0, d, { marked })];
      if (spec.positions === 'sinusoidal') {
        const pairs = Math.floor(d / 2);
        cols.push(op('+'), {
          kind: 'dials',
          key: 'dials',
          title: 'place as clock hands',
          sub: 'fixed, fast on the left',
          n: pairs,
          ideal: Math.min(pairs, 8) * Math.min(24, L.pitch),
          min: Math.min(pairs, 8) * 12,
        });
        cols.push(op('='), vec('p', 'position p', 'the hands as numbers', cache.pos));
      } else cols.push(op('+'), vec('p', 'position p', 'learned, one per place', cache.pos));
      return [...cols, op('='), vec('x', 'x = e + p', marked.length ? 'now they differ' : 'into the block', cache.x0, d, { marked })];
    }
    case 'norm1':
    case 'norm2':
    case 'final': {
      const norm = part === 'norm1' ? Lc.norm1 : part === 'norm2' ? Lc.norm2 : cache.normF;
      const input = part === 'norm1' ? Lc.x : part === 'norm2' ? Lc.mid : (cache.layers[cache.layers.length - 1]?.out ?? cache.x0);
      if (!norm) return [vec('in', 'x', 'layer norm off', input)];
      const lim = Math.max(3, scaleOf(input));
      return [
        vec('in', 'x', 'the stream', input),
        op('→'),
        { kind: 'dots', key: 'before', title: 'before', sub: 'tick μ, band ±σ', norm, values: input, lim, ideal: 150, min: 90 },
        op('→'),
        { kind: 'dots', key: 'after', title: 'after', sub: '(x − μ) / σ', norm, values: norm.n, lim, after: true, ideal: 150, min: 90 },
        op('→'),
        vec('y', 'x̂', '× γ + β', norm.y),
      ];
    }
    case 'attention': {
      const slice = (values: Float64Array) => {
        const out = new Float64Array(T * dh);
        for (let t = 0; t < T; t++) out.set(headRow(values, t, d, dh, head), t * dh);
        return out;
      };
      const hue = headColour(palette, head);
      const scores = Lc.scores.subarray(head * T * T, (head + 1) * T * T);
      const weights = Lc.attn.subarray(head * T * T, (head + 1) * T * T);
      const pair = (title: string) => (t: number, i: number) => title + ' · ' + tokens[t] + ' → ' + tokens[i];
      return [
        vec('q', 'Q', 'looks for', slice(Lc.q), dh, { tint: palette.blue, ideal: 40 }),
        vec('k', 'K', 'offers', slice(Lc.k), dh, { tint: palette.red, ideal: 40 }),
        vec('v', 'V', 'passes on', slice(Lc.v), dh, { tint: palette.green, ideal: 40 }),
        op('→'),
        {
          kind: 'matrix',
          key: 'scores',
          title: 'scores Q · K',
          sub: '÷ √' + dh + (spec.causal ? ', later ones hidden' : ''),
          values: scores,
          n: T,
          scale: scaleOf(Array.from(scores).filter(Number.isFinite)),
          mask: spec.causal,
          labels: tokens,
          ideal: matrixW,
          min: T * 9,
          name: pair('score'),
        },
        op('→'),
        { kind: 'matrix', key: 'weights', title: 'softmax weights', sub: 'each row adds to 1', values: weights, n: T, tint: hue, mask: spec.causal, labels: tokens, ideal: matrixW, min: T * 9, name: pair('weight') },
        op('→'),
        vec('z', 'weights × V', 'the blend it takes', slice(Lc.z), dh, { tint: hue, ideal: 44 }),
      ];
    }
    case 'ffn': {
      if (!Lc.pre || !Lc.ff || !Lc.norm2) return [];
      const f = spec.ffn;
      return [
        vec('xhat', 'x̂', 'after layer norm', Lc.norm2.y, d, { ideal: 44 }),
        op('→'),
        {
          kind: 'relu',
          key: 'h',
          title: 'h = x̂ W₁: ' + f + ' units',
          sub: 'ReLU sets the faded ones to 0',
          values: Lc.pre,
          n: f,
          scale: scaleOf(Lc.pre),
          ideal: Math.min(260, f * 3.2 + 36),
          min: f + 36,
          name: (t, i) => 'unit ' + (i + 1) + ' · ' + tokens[t],
        },
        op('→'),
        vec('f', 'f = ReLU(h) W₂', 'back to ' + d, Lc.ff, d, { ideal: 48 }),
        op('+'),
        vec('out', 'x + f', 'the stream moves on', Lc.out, d, { ideal: 48 }),
      ];
    }
    case 'output': {
      const Vn = spec.outVocab;
      const classes = data.outVocab.slice();
      return [
        vec('xhat', 'x̂', 'final layer norm', cache.normF.y, d, { ideal: 44 }),
        op('→'),
        {
          kind: 'strip',
          key: 'z',
          title: 'scores z = x̂ W_U',
          sub: 'one per ' + (data.nextToken ? 'word' : 'answer'),
          values: cache.logits,
          n: Vn,
          scale: scaleOf(cache.logits),
          labels: classes,
          ideal: Math.min(200, Vn * 16),
          min: Vn * 5,
          name: (t, i) => 'score of ' + classes[i] + ' · ' + tokens[t],
        },
        op('→'),
        {
          kind: 'bars',
          key: 'p',
          title: 'softmax shares',
          sub: 'add to 1',
          values: cache.probs,
          n: Vn,
          labels: classes,
          ideal: Math.min(200, Vn * 16),
          min: Vn * 5,
          name: (t, i) => 'share of ' + classes[i] + ' · ' + tokens[t],
        },
        op('→'),
        { kind: 'chips', key: 'call', title: 'call', sub: 'vs answer', ideal: data.nextToken ? 110 : 64, min: data.nextToken ? 96 : 56 },
      ];
    }
    default:
      return [];
  }
}

function drawDials(ctx: CanvasRenderingContext2D, x: number, w: number, y: number, t: number, s: Scene, L: Layout, palette: Palette, hot: boolean): void {
  const d = s.spec.width;
  const pairs = Math.floor(d / 2);
  const every = Math.max(1, Math.ceil(pairs / 8));
  const shown: number[] = [];
  for (let k = 0; k < pairs; k += every) shown.push(k);
  const dw = w / shown.length;
  const r = Math.max(3, Math.min(dw, L.pitch) / 2 - 2);
  shown.forEach((k, i) => {
    const cx = x + (i + 0.5) * dw;
    const sn = s.cache.pos[t * d + 2 * k];
    const cs = 2 * k + 1 < d ? s.cache.pos[t * d + 2 * k + 1] : 1;
    const turned = t / Math.pow(10000, (2 * k) / d);
    ctx.beginPath();
    ctx.arc(cx, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = rgba(palette.textFaint, 0.5);
    ctx.lineWidth = 1;
    ctx.stroke();
    if (turned > 0.001) {
      ctx.beginPath();
      ctx.moveTo(cx, y);
      ctx.arc(cx, y, r, 0, -Math.min(turned, Math.PI * 2 - 0.001), true);
      ctx.closePath();
      ctx.fillStyle = rgba(hot ? palette.accent : palette.positive, 0.28);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(cx, y);
    ctx.lineTo(cx + cs * r, y - sn * r);
    ctx.strokeStyle = hot ? palette.accent : palette.text;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function drawCard(
  ctx: CanvasRenderingContext2D,
  L: Layout,
  s: Scene,
  palette: Palette,
  part: Part,
  lit: number[],
  hoverCell: { key: string; t: number; i: number } | null,
  numbers: boolean,
  buttons: Button[],
  actions: { onPart: (part: Part) => void; onPick: (layer: number, head: number) => void; onSweep: () => void },
): { card: Rect; placed: Placed[] } {
  const { spec, cache, data, probe, layer, head } = s;
  const T = cache.length;
  const x0 = L.lab.x + L.lab.w + 8;
  const card = { x: x0, y: 22, w: L.width - x0 - 4, h: Math.min(L.height - 24, L.bottom + 40 - 22) };
  ctx.save();
  ctx.shadowColor = 'rgba(0, 0, 0, 0.22)';
  ctx.shadowBlur = 14;
  ctx.shadowOffsetY = 3;
  roundRect(ctx, card.x, card.y, card.w, card.h, 10);
  ctx.fillStyle = palette.surface;
  ctx.fill();
  ctx.restore();
  ctx.save();
  roundRect(ctx, card.x, card.y, card.w, card.h, 10);
  ctx.strokeStyle = rgba(palette.textFaint, 0.6);
  ctx.lineWidth = 1;
  ctx.stroke();

  // The header: the part's name, its pickers, and the way to the next part or out.
  const title = PART_NAMES[part] + (spec.layers > 1 && part !== 'embed' && part !== 'output' && part !== 'final' ? ' · block ' + (layer + 1) : '');
  label(ctx, title, card.x + 12, card.y + 14, palette.text, '600 12px ' + FONT_STACK, 'left');
  ctx.font = '600 12px ' + FONT_STACK;
  let hx = card.x + 12 + ctx.measureText(title).width + 12;
  const picker = (text: string, colour: string, prev: () => void, next: () => void) => {
    const a = glyphButton(ctx, '‹', hx, card.y + 5, palette, 16);
    ctx.font = '600 10px ' + FONT_STACK;
    const w = ctx.measureText(text).width;
    label(ctx, text, hx + 20, card.y + 13.5, colour, '600 10px ' + FONT_STACK, 'left');
    const b = glyphButton(ctx, '›', hx + 24 + w, card.y + 5, palette, 16);
    buttons.push({ rect: a, run: prev }, { rect: b, run: next });
    hx += 48 + w;
  };
  if (part === 'attention' && spec.heads > 1) {
    picker('head ' + (head + 1) + ' of ' + spec.heads, headColour(palette, head), () => actions.onPick(layer, (head + spec.heads - 1) % spec.heads), () => actions.onPick(layer, (head + 1) % spec.heads));
  }
  if (spec.layers > 1 && (part === 'attention' || part === 'ffn' || part === 'norm1' || part === 'norm2')) {
    picker('block ' + (layer + 1) + ' of ' + spec.layers, palette.textMuted, () => actions.onPick((layer + spec.layers - 1) % spec.layers, head), () => actions.onPick((layer + 1) % spec.layers, head));
  }
  const order = partsOf(spec);
  const at = order.indexOf(part);
  const close = glyphButton(ctx, '×', card.x + card.w - 24, card.y + 5, palette);
  const next = glyphButton(ctx, '›', card.x + card.w - 46, card.y + 5, palette);
  const prev = glyphButton(ctx, '‹', card.x + card.w - 66, card.y + 5, palette);
  const play = glyphButton(ctx, '▸', card.x + card.w - 92, card.y + 5, palette);
  buttons.push(
    { rect: play, run: actions.onSweep },
    { rect: close, run: () => actions.onPart('none') },
    { rect: next, run: () => actions.onPart(order[(at + 1) % order.length]) },
    { rect: prev, run: () => actions.onPart(order[(at + order.length - 1) % order.length]) },
  );

  // The lit rows across the card.
  for (const t of lit) {
    ctx.fillStyle = rgba(palette.accent, 0.08);
    ctx.fillRect(card.x + 1, L.top + t * L.pitch, card.w - 2, L.pitch);
  }

  // Lay the steps out across the card.
  const cols = cardCols(part, s, L, palette);
  const left = card.x + 12;
  const room = card.w - 24;
  const gap = 8;
  const gaps = gap * Math.max(0, cols.length - 1);
  const ideal = cols.reduce((sum, c) => sum + c.ideal, 0);
  let widths: number[];
  if (ideal + gaps <= room) {
    const grow = Math.min(1.35, (room - gaps) / ideal);
    widths = cols.map((c) => (c.kind === 'op' ? c.ideal : c.ideal * grow));
  } else {
    const fixed = cols.reduce((sum, c) => sum + (c.kind === 'op' ? c.min : 0), 0);
    const flex = cols.reduce((sum, c) => sum + (c.kind === 'op' ? 0 : c.ideal), 0);
    const f = Math.max(0.2, (room - gaps - fixed) / flex);
    widths = cols.map((c) => (c.kind === 'op' ? c.min : Math.max(Math.min(c.min, c.ideal * f), c.ideal * f)));
  }
  const used = widths.reduce((a, b) => a + b, 0) + gaps;
  let x = left + Math.max(0, (room - used) / 2);
  const placed: Placed[] = [];
  const sh = Math.max(7, Math.min(18, L.pitch - 10));
  const hovered = hoverCell ? hoverCell.t : null;
  cols.forEach((col, ci) => {
    const w = widths[ci];
    placed.push({ col, x, w });
    const cx = x + w / 2;
    if (col.title) label(ctx, col.title, cx, L.top - 30, palette.text, '600 10px ' + FONT_STACK, 'center', w + gap);
    if (col.sub) label(ctx, col.sub, cx, L.top - 18, palette.textFaint, '9px ' + FONT_STACK, 'center', w + gap);
    // Column names over a matrix, a bar chart or a dial row.
    if ((col.kind === 'matrix' || col.kind === 'bars' || (col.kind === 'strip' && col.labels)) && col.labels && col.n) {
      const cw = w / col.n;
      if (cw >= 8) col.labels.forEach((name, i) => label(ctx, name, x + (i + 0.5) * cw, L.top - 6, palette.textMuted, (cw >= 12 ? '9px ' : '8px ') + MONO_STACK, 'center', cw + 4));
    }
    if (col.kind === 'dials') {
      // How far each hand turns per place; with no room for all, the fastest and the slowest.
      const pairs = col.n ?? 1;
      const every = Math.max(1, Math.ceil(pairs / 8));
      const count = Math.ceil(pairs / every);
      if (w / count >= 26) {
        for (let i = 0; i < count; i++) label(ctx, turnOf(i * every, spec.width), x + ((i + 0.5) * w) / count, L.top - 6, palette.textFaint, '8px ' + MONO_STACK);
      } else {
        label(ctx, turnOf(0, spec.width) + ' per place', x, L.top - 6, palette.textFaint, '8px ' + MONO_STACK, 'left');
        label(ctx, turnOf((count - 1) * every, spec.width), x + w, L.top - 6, palette.textFaint, '8px ' + MONO_STACK, 'right');
      }
    }
    for (let t = 0; t < T; t++) {
      const y = rowY(L, t);
      const hot = lit.includes(t) || hovered === t;
      switch (col.kind) {
        case 'strip':
          if (col.values && col.n) drawStrip(ctx, col.values.subarray(t * col.n, (t + 1) * col.n), x, y - sh / 2, w, sh, palette, col.scale ?? 1, col.tint ? { magnitude: true, tint: col.tint } : {});
          if (col.marked?.includes(t)) {
            ctx.strokeStyle = palette.accent;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x - 2, y - sh / 2 - 2, w + 4, sh + 4);
          }
          break;
        case 'op':
          label(ctx, col.texts?.[0] ?? '', cx, y, hot ? palette.text : palette.textMuted, '11px ' + MONO_STACK, 'center', w);
          break;
        case 'text':
          label(ctx, col.texts?.[t] ?? '', cx, y, hot ? palette.accent : palette.textMuted, '10px ' + MONO_STACK, 'center', w);
          break;
        case 'dials':
          drawDials(ctx, x, w, y, t, s, L, palette, lit.includes(t));
          break;
        case 'dots': {
          if (!col.norm || !col.values) break;
          const lim = col.lim ?? 3;
          const xAt = (v: number) => x + ((Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * w;
          const mu = col.after ? 0 : col.norm.mean[t];
          const sd = col.after ? 1 : 1 / col.norm.rstd[t];
          const bh = Math.min(L.pitch - 4, 16);
          ctx.fillStyle = rgba(palette.accent, 0.14);
          ctx.fillRect(xAt(mu - sd), y - bh / 2, xAt(mu + sd) - xAt(mu - sd), bh);
          ctx.strokeStyle = rgba(palette.textFaint, 0.35);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + w, y);
          ctx.stroke();
          ctx.strokeStyle = palette.accent;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(xAt(mu), y - bh / 2 - 1);
          ctx.lineTo(xAt(mu), y + bh / 2 + 1);
          ctx.stroke();
          const r = Math.max(1.5, Math.min(2.5, L.pitch * 0.09));
          for (let i = 0; i < spec.width; i++) {
            const v = col.values[t * spec.width + i];
            ctx.beginPath();
            ctx.arc(xAt(v), y + ((i % 3) - 1) * r, r, 0, Math.PI * 2);
            ctx.fillStyle = mix(palette.surfaceAlt, v < 0 ? palette.negative : palette.positive, Math.min(1, 0.35 + Math.abs(v) / lim));
            ctx.fill();
          }
          break;
        }
        case 'matrix': {
          if (!col.values || !col.n) break;
          const n = col.n;
          const cw = w / n;
          const sz = Math.max(3, Math.min(cw, L.pitch) - 2);
          for (let k = 0; k < n; k++) {
            const cxk = x + (k + 0.5) * cw;
            if (col.mask && k > t) {
              hatch(ctx, cxk - sz / 2, y - sz / 2, sz, sz, palette);
              continue;
            }
            const v = col.values[t * n + k];
            const fill = col.tint ? weightColour(palette, v, col.tint) : mix(palette.surfaceAlt, v < 0 ? palette.negative : palette.positive, Math.min(1, Math.abs(v) / (col.scale ?? 1)));
            ctx.fillStyle = fill;
            ctx.fillRect(cxk - sz / 2, y - sz / 2, sz, sz);
            if (numbers && sz >= 18) label(ctx, col.tint ? (v >= 0.995 ? '1' : v.toFixed(2).replace(/^0/, '')) : fmt(v, 1), cxk, y + 0.5, readableOn(fill), '8px ' + MONO_STACK, 'center', sz - 2);
          }
          break;
        }
        case 'relu': {
          if (!col.values || !col.n) break;
          const n = col.n;
          const cw = (w - 34) / n;
          let kept = 0;
          for (let i = 0; i < n; i++) {
            const v = col.values[t * n + i];
            if (v > 0) {
              kept++;
              ctx.fillStyle = mix(palette.surfaceAlt, palette.positive, Math.min(1, 0.3 + (0.7 * v) / (col.scale ?? 1)));
            } else ctx.fillStyle = rgba(palette.negative, 0.16);
            ctx.fillRect(x + i * cw, y - sh / 2, Math.max(0.6, cw - (cw >= 4 ? 0.6 : 0)), sh);
          }
          ctx.strokeStyle = rgba(palette.textFaint, 0.35);
          ctx.lineWidth = 1;
          ctx.strokeRect(Math.round(x) + 0.5, Math.round(y - sh / 2) + 0.5, Math.round(w - 34) - 1, Math.round(sh) - 1);
          label(ctx, kept + '/' + n, x + w - 2, y, hot ? palette.text : palette.textMuted, '9px ' + MONO_STACK, 'right');
          break;
        }
        case 'bars': {
          if (!col.values || !col.n) break;
          const n = col.n;
          const cw = w / n;
          const hmax = L.pitch - 6;
          const base = y + hmax / 2;
          let call = 0;
          for (let i = 1; i < n; i++) if (col.values[t * n + i] > col.values[t * n + call]) call = i;
          const target = probe.targets[t] ?? -1;
          for (let i = 0; i < n; i++) {
            const p = col.values[t * n + i];
            ctx.fillStyle = i === call ? palette.accent : mix(palette.surfaceAlt, palette.textMuted, 0.55);
            ctx.fillRect(x + i * cw + 0.5, base - Math.max(0.5, p * hmax), Math.max(1, cw - 1), Math.max(0.5, p * hmax));
            if (i === target) {
              ctx.strokeStyle = palette.green;
              ctx.lineWidth = 1.3;
              ctx.strokeRect(x + i * cw, base - hmax, cw, hmax);
            }
          }
          ctx.strokeStyle = rgba(palette.textFaint, 0.5);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, base + 0.5);
          ctx.lineTo(x + w, base + 0.5);
          ctx.stroke();
          break;
        }
        case 'chips': {
          const target = probe.targets[t] ?? -1;
          if (target < 0 && !data.nextToken) break;
          const call = cache.predicted[t];
          const cw = (w - 6) / 2;
          const words = data.task === 'animals';
          drawChip(ctx, data.outVocab[call] ?? '?', x + cw / 2, y, cw, Math.min(L.pitch - 4, 18), palette, target < 0 ? 'plain' : probe.scored.includes(t) ? (call === target ? 'right' : 'wrong') : 'plain', words ? 9 : 10);
          if (target >= 0) drawChip(ctx, data.outVocab[target] ?? '?', x + cw * 1.5 + 6, y, cw, Math.min(L.pitch - 4, 18), palette, 'ghost', words ? 9 : 10);
          break;
        }
      }
    }
    // The scale under a pair of dot strips.
    if (col.kind === 'dots') {
      const lim = col.lim ?? 3;
      label(ctx, fmt(-lim, 1), x, L.bottom + 10, palette.textFaint, '9px ' + MONO_STACK, 'left');
      label(ctx, '0', x + w / 2, L.bottom + 10, palette.textFaint, '9px ' + MONO_STACK);
      label(ctx, fmt(lim, 1), x + w, L.bottom + 10, palette.textFaint, '9px ' + MONO_STACK, 'right');
    }
    x += w + gap;
  });
  ctx.restore();
  return { card, placed };
}

/** The cell under a point on an open card, and what it holds. */
function cellAt(placed: Placed[], L: Layout, s: Scene, pos: { x: number; y: number }): { key: string; t: number; i: number; lines: string[]; rect: Rect } | null {
  if (pos.y < L.top || pos.y >= L.bottom) return null;
  const t = Math.floor((pos.y - L.top) / L.pitch);
  const hit = placed.find((p) => pos.x >= p.x && pos.x < p.x + p.w);
  if (!hit) return null;
  const { col, x, w } = hit;
  const y = L.top + t * L.pitch;
  if (col.kind === 'dials') {
    const d = s.spec.width;
    const pairs = Math.floor(d / 2);
    const every = Math.max(1, Math.ceil(pairs / 8));
    const count = Math.ceil(pairs / every);
    const i = Math.min(count - 1, Math.floor(((pos.x - x) / w) * count));
    const k = i * every;
    const deg = ((t / Math.pow(10000, (2 * k) / d)) * 180) / Math.PI;
    return {
      key: col.key,
      t,
      i,
      lines: ['place ' + t + ' · dims ' + (2 * k + 1) + ',' + (2 * k + 2), 'turned ' + fmt(deg % 360, 1) + '°, sin ' + fmt(s.cache.pos[t * d + 2 * k], 2) + ', cos ' + fmt(s.cache.pos[t * d + 2 * k + 1] ?? 1, 2)],
      rect: { x: x + (i * w) / count, y, w: w / count, h: L.pitch },
    };
  }
  if (!col.values || !col.n || !col.name) return { key: col.key, t, i: 0, lines: [], rect: { x, y, w, h: L.pitch } };
  const n = col.n;
  const span = col.kind === 'relu' ? w - 34 : w;
  const i = Math.floor(((pos.x - x) / span) * n);
  if (i < 0 || i >= n) return null;
  const v = col.values[t * n + i];
  const masked = col.mask && i > t;
  const value = masked ? 'masked' : col.kind === 'relu' ? fmt(v, 3) + (v > 0 ? ', kept' : ', set to 0') : fmt(v, 3);
  return { key: col.key, t, i, lines: [col.name(t, i), value], rect: { x: x + (i * span) / n, y, w: span / n, h: L.pitch } };
}

/* ---------------- the view ---------------- */

export interface ArchitectureProps {
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  probe: Probe;
  part: Part;
  onPart: (part: Part) => void;
  layer: number;
  head: number;
  onPick: (layer: number, head: number) => void;
  /** A lesson's part. */
  spot?: Part | null;
  numbers?: boolean;
  /** Rows a lesson lights on an open part. */
  places?: number[];
  /** Counts a lesson's sweeps: each new value walks a highlight down the rows once. */
  sweep?: number;
  description: string;
  redrawKey?: unknown;
}

/** Milliseconds a sweep rests on each row. */
const SWEEP_BEAT = 700;

const NO_PLACES: number[] = [];

export function ArchitectureView({ data, spec, cache, probe, part, onPart, layer, head, onPick, spot = null, numbers = false, places = NO_PLACES, sweep = 0, description, redrawKey }: ArchitectureProps) {
  const [hoverPart, setHoverPart] = useState<Part | null>(null);
  const [pointerRow, setHoverRow] = useState<number | null>(null);
  // A sweep walks one row at a time, from the top, and then lets go; the card's play button starts one too.
  const [ownSweep, setOwnSweep] = useState(0);
  const [sweepRow, setSweepRow] = useState<number | null>(null);
  const T = cache.length;
  useEffect(() => {
    if (sweep + ownSweep === 0) return;
    let row = 0;
    let id = 0;
    const tick = () => {
      setSweepRow(row < T ? row : null);
      row += 1;
      if (row <= T) id = window.setTimeout(tick, SWEEP_BEAT);
    };
    id = window.setTimeout(tick, 0);
    return () => window.clearTimeout(id);
  }, [sweep, ownSweep, T]);
  const hoverRow = pointerRow ?? sweepRow;
  const [hoverCell, setHoverCell] = useState<{ key: string; t: number; i: number; at: { x: number; y: number } } | null>(null);
  const [onButton, setOnButton] = useState(false);
  const layoutRef = useRef<Layout | null>(null);
  const buttonsRef = useRef<Button[]>([]);
  const cardRef = useRef<{ card: Rect; placed: Placed[] } | null>(null);
  const tokens = useMemo(() => cache.tokens.map((t) => tokenName(data, t)), [cache.tokens, data]);
  const words = data.task === 'animals';
  const open = partsOf(spec).includes(part) ? part : 'none';

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const L = layoutArch(width, height, spec, cache.length, words);
      layoutRef.current = L;
      const scene: Scene = { data, spec, cache, probe, layer: Math.min(layer, spec.layers - 1), head: Math.min(head, spec.heads - 1), tokens };
      const buttons: Button[] = [];
      if (open === 'none') {
        drawBase(ctx, L, scene, palette, { part: hoverPart, row: hoverRow, spot }, buttons, onPick);
        cardRef.current = null;
      } else {
        // The whole model stays behind the open part, faded, with its tokens still readable.
        ctx.save();
        ctx.globalAlpha = 0.14;
        drawBase(ctx, L, scene, palette, { part: null, row: null, spot: null }, [], onPick);
        ctx.restore();
        const rows = hoverRow === null ? places : [hoverRow, ...places];
        for (let t = 0; t < cache.length; t++) drawToken(ctx, L, tokens[t], t, rows.includes(t) ? 'accent' : t === 0 ? 'start' : 'plain', palette, words);
        cardRef.current = drawCard(ctx, L, scene, palette, open, rows, hoverCell, numbers, buttons, { onPart, onPick, onSweep: () => setOwnSweep((n) => n + 1) });
        // The hovered cell, read afresh so its number follows training.
        const cell = hoverCell ? cellAt(cardRef.current.placed, L, scene, { x: hoverCell.at.x, y: hoverCell.at.y }) : null;
        if (cell && cell.lines.length) {
          const size = measurePlate(ctx, cell.lines);
          const where = placePlate(size, cell.rect, width, height);
          drawHoverPlate(ctx, cell.lines, where.left, where.top, palette);
        }
      }
      buttonsRef.current = buttons;
    },
    [data, spec, cache, probe, layer, head, tokens, words, open, hoverPart, hoverRow, hoverCell, spot, numbers, places, onPart, onPick],
  );

  const onDown = (pos: { x: number; y: number }) => {
    const button = buttonsRef.current.find((b) => inside(pos, b.rect));
    if (button) {
      button.run();
      return;
    }
    const L = layoutRef.current;
    if (!L) return;
    if (open !== 'none') {
      const card = cardRef.current?.card;
      if (card && !inside(pos, card) && pos.x > L.lab.x + L.lab.w) onPart('none');
      return;
    }
    const hit = L.parts.find((p) => inside(pos, p.rect));
    if (hit) onPart(hit.part);
  };

  const onMove = (pos: { x: number; y: number } | null) => {
    const L = layoutRef.current;
    if (!pos || !L) {
      setHoverPart(null);
      setHoverRow(null);
      setHoverCell(null);
      setOnButton(false);
      return;
    }
    setOnButton(buttonsRef.current.some((b) => inside(pos, b.rect)));
    const row = pos.y >= L.top && pos.y < L.bottom ? Math.floor((pos.y - L.top) / L.pitch) : null;
    setHoverRow((prev) => (prev === row ? prev : row));
    if (open === 'none') {
      const hit = L.parts.find((p) => inside(pos, p.rect))?.part ?? null;
      setHoverPart((prev) => (prev === hit ? prev : hit));
      setHoverCell(null);
      return;
    }
    const placed = cardRef.current?.placed ?? [];
    const scene: Scene = { data, spec, cache, probe, layer, head, tokens };
    const cell = cellAt(placed, L, scene, pos);
    const next = cell && cell.lines.length ? { key: cell.key, t: cell.t, i: cell.i, at: { x: cell.rect.x + cell.rect.w / 2, y: cell.rect.y + cell.rect.h / 2 } } : null;
    setHoverCell((prev) => (prev && next && prev.key === next.key && prev.t === next.t && prev.i === next.i ? prev : next));
  };

  const targets = useMemo(() => partsOf(spec), [spec]);
  // The keys walk on from the open part.
  const handleKey = useDiagramKeys<Part>({
    targets,
    cursor: hoverPart ?? (open !== 'none' ? open : null),
    same: (a, b) => a === b,
    onCursor: (target) => setHoverPart(target),
    onOpen: (target) => onPart(target),
    onClose: () => onPart('none'),
  });

  const noted = open !== 'none' ? open : (hoverPart ?? 'none');
  const note = partNote(noted, data, spec);
  const label =
    description +
    (open === 'none' ? ' Click a part to open it.' : ' Open: ' + PART_NAMES[open].toLowerCase() + (open === 'attention' ? ', head ' + (head + 1) : '') + '.');

  return (
    <div className="mlx-tfa">
      <div className="mlx-arch mlx-tfa__stage" tabIndex={0} role="group" aria-label={description + ' Arrow keys move between the parts, Enter opens one, Escape closes it.'} onKeyDown={handleKey} data-arch-detail="">
        <Chart
          draw={draw}
          height="fill"
          description={label}
          cursor={onButton || (open === 'none' && hoverPart) ? 'pointer' : open !== 'none' && hoverCell ? 'crosshair' : 'default'}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerLeave={() => onMove(null)}
          redrawKey={redrawKey}
        />
      </div>
      <p className="mlx-tfa__note" aria-live="polite">
        <strong>{note.title}</strong> {note.text}
      </p>
    </div>
  );
}
