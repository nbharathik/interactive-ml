/** The output column: the answer for the probe, a sentence written word by word, the held-out gallery, the curves, the embedding and position maps. */

import { useCallback, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate } from '../../explainer/diagramStyle';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { spellTokens } from '../../lib/datasets/sequences';
import { fmt, fmtPercent } from '../../lib/math/stats';
import type { TransformerCache, TransformerSpec, TransformerState } from '../../lib/ml/transformer';
import { FONT_STACK, MONO_STACK, drawAxes, extentOf, makeFrame, padExtent } from '../../lib/viz/canvas';
import { categorical, mix, rgba } from '../../lib/viz/palette';
import { drawEmptyAxes } from '../../lib/viz/plots';
import { caption, drawChip } from './draw';
import type { ChipTone } from './draw';
import type { Probe } from './model';
import { tokenName, wordGroup } from './model';

/* ---------------- the answer ---------------- */

export interface Written {
  /** The prompt the model was given. */
  prompt: number[];
  /** Prompt and continuation. */
  tokens: number[];
}

/** The letter task's answer: what the model reads, its five likeliest next letters, and the sentence it writes. */
function LetterAnswer({ data, spec, cache, probe, written }: { data: SequenceDataset; spec: TransformerSpec; cache: TransformerCache; probe: Probe; written: Written | null }) {
  const T = cache.length;
  const V = spec.outVocab;
  const last = T - 1;
  const ranked = Array.from({ length: V }, (_, c) => c)
    .filter((c) => c !== 0)
    .sort((a, b) => cache.probs[last * V + b] - cache.probs[last * V + a])
    .slice(0, 5);
  const target = probe.targets[last] ?? -1;
  const draw = useCallback(
    ({ ctx, width, palette }: DrawArgs) => {
      const gutter = 64;
      const text = (tokens: readonly number[]) => tokens.slice(1).map((t) => tokenName(data, t)).join('');
      caption(ctx, 'reads', gutter - 8, 11, palette);
      caption(ctx, 'next letter', gutter - 8, 38, palette);
      ctx.save();
      ctx.font = '12px ' + MONO_STACK;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      // The prompt, cut from the left if it does not fit, its last letter marked.
      let shown = text(cache.tokens);
      while (shown.length > 1 && ctx.measureText(shown).width > width - gutter - 12) shown = shown.slice(1);
      ctx.fillStyle = palette.text;
      ctx.fillText(shown.slice(0, -1), gutter, 11);
      const head = ctx.measureText(shown.slice(0, -1)).width;
      ctx.fillStyle = palette.accent;
      ctx.fillText(shown.slice(-1), gutter + head, 11);
      ctx.fillRect(gutter + head, 19, ctx.measureText(shown.slice(-1)).width, 1.5);
      // The five likeliest next letters, each with its share.
      const slot = Math.min(76, (width - gutter - 4) / ranked.length);
      ranked.forEach((c, i) => {
        const x = gutter + i * slot;
        const p = cache.probs[last * V + c];
        const tone: ChipTone = c === target ? 'right' : i === 0 ? 'accent' : 'plain';
        drawChip(ctx, tokenName(data, c), x + 11, 38, 20, 18, palette, tone, 11);
        ctx.fillStyle = palette.surfaceAlt;
        ctx.fillRect(x + 25, 36, slot - 32, 4);
        ctx.fillStyle = i === 0 ? palette.accent : rgba(palette.textMuted, 0.8);
        ctx.fillRect(x + 25, 36, (slot - 32) * p, 4);
        ctx.font = '9px ' + MONO_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.fillText(fmtPercent(p, 0), x + 25, 46);
      });
      if (written) {
        caption(ctx, 'writes', gutter - 8, 72, palette);
        ctx.font = '12px ' + MONO_STACK;
        const given = text(written.tokens.slice(0, written.prompt.length));
        const rest = written.tokens
          .slice(written.prompt.length)
          .map((t) => data.vocab[t] ?? '')
          .join('');
        ctx.fillStyle = palette.textMuted;
        ctx.fillText(given, gutter, 72, width - gutter - 8);
        ctx.fillStyle = palette.accent;
        ctx.fillText(rest, gutter + ctx.measureText(given).width, 72, Math.max(20, width - gutter - 8 - ctx.measureText(given).width));
      }
      ctx.restore();
    },
    [cache, data, ranked, V, last, target, written],
  );
  return (
    <Chart
      draw={draw}
      height={written ? 86 : 58}
      description={
        'The model reads ' + JSON.stringify(spellTokens(data, cache.tokens)) + ' and ranks the next letter: ' + ranked.map((c) => tokenName(data, c) + ' ' + fmtPercent(cache.probs[last * V + c], 0)).join(', ') + '.' +
        (written ? ' From the prompt it writes, one letter at a time: ' + JSON.stringify(spellTokens(data, written.tokens)) + '.' : '')
      }
      redrawKey={cache.predicted.join(',') + '|' + (written ? written.tokens.join(',') : '')}
    />
  );
}

export function AnswerView({ data, spec, cache, probe, honest, written }: { data: SequenceDataset; spec: TransformerSpec; cache: TransformerCache; probe: Probe; honest: number[]; written: Written | null }) {
  if (data.task === 'letters') return <LetterAnswer data={data} spec={spec} cache={cache} probe={probe} written={written} />;
  return <TokenAnswer data={data} spec={spec} cache={cache} probe={probe} honest={honest} written={written} />;
}

function TokenAnswer({ data, spec, cache, probe, honest, written }: { data: SequenceDataset; spec: TransformerSpec; cache: TransformerCache; probe: Probe; honest: number[]; written: Written | null }) {
  const T = cache.length;
  const draw = useCallback(
    ({ ctx, width, palette }: DrawArgs) => {
      const gutter = 64;
      const laneW = Math.min(64, (width - gutter - 6) / T);
      const cx = (t: number) => gutter + (t + 0.5) * laneW;
      const chipW = Math.min(laneW - 4, 58);
      const rows = data.nextToken ? ['input', 'next word', 'model'] : ['input', 'answer', 'model'];
      const y = [11, 34, 57];
      rows.forEach((label, r) => caption(ctx, label, gutter - 8, y[r], palette));
      for (let t = 0; t < T; t++) {
        drawChip(ctx, tokenName(data, cache.tokens[t]), cx(t), y[0], chipW, 18, palette, t === 0 ? 'start' : 'plain', 11);
        const target = probe.targets[t] ?? -1;
        if (target < 0) continue;
        const scored = probe.scored.includes(t);
        drawChip(ctx, data.outVocab[target] ?? '?', cx(t), y[1], chipW, 18, palette, 'ghost', 11);
        const call = honest[t] ?? cache.predicted[t];
        const tone: ChipTone = scored ? (call === target ? 'right' : 'wrong') : 'plain';
        drawChip(ctx, data.outVocab[call] ?? '?', cx(t), y[2], chipW, 18, palette, tone, 11);
        const p = cache.probs[t * spec.outVocab + call];
        ctx.fillStyle = palette.surfaceAlt;
        ctx.fillRect(cx(t) - chipW / 2, y[2] + 11, chipW, 3);
        ctx.fillStyle = rgba(palette.accent, 0.85);
        ctx.fillRect(cx(t) - chipW / 2, y[2] + 11, chipW * p, 3);
      }
      if (written) {
        const wy = 88;
        caption(ctx, 'writes', gutter - 8, wy, palette);
        const words = written.tokens.map((tok) => tokenName(data, tok));
        ctx.save();
        ctx.font = '12px ' + MONO_STACK;
        let x = gutter;
        words.forEach((word, i) => {
          if (i === 0) return;
          const generated = i >= written.prompt.length;
          const w = ctx.measureText(word).width + 12;
          if (x + w > width - 4) return;
          drawChip(ctx, word, x + w / 2, wy, w, 18, palette, generated ? 'accent' : 'plain', 11);
          x += w + 4;
        });
        ctx.restore();
      }
    },
    [T, cache, data, honest, probe, spec, written],
  );
  const right = probe.scored.filter((t) => (honest[t] ?? cache.predicted[t]) === probe.targets[t]).length;
  return (
    <Chart
      draw={draw}
      height={written ? 100 : 74}
      description={
        'The probe, its answers and the model’s: ' + right + ' of ' + probe.scored.length + ' scored positions right.' +
        (written ? ' From the prompt it writes, one word at a time, each read from the words before: ' + written.tokens.slice(1).map((t) => tokenName(data, t)).join(' ') + '.' : '')
      }
      redrawKey={cache.predicted.join(',') + '|' + (written ? written.tokens.join(',') : '')}
    />
  );
}

/* ---------------- held-out gallery ---------------- */

const GALLERY_ROWS = 8;
const ROW_H = 20;

/** Columns that fit the width: a sentence needs more room than six digits. */
function galleryColumns(data: SequenceDataset, width: number): number {
  return Math.max(1, Math.min(4, Math.floor(width / (data.nextToken ? 200 : 128))));
}

/** The first held-out sequences, as many as the panel has rows for. */
export function Gallery({ data, state, probe, onProbe }: { data: SequenceDataset; state: TransformerState; probe: number | null; onProbe: (index: number) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  const grid = useRef({ cols: 1, width: 300, shown: 0 });
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const rowH = ROW_H;
      const words = data.nextToken;
      const n = galleryColumns(data, width);
      const shown = Math.min(data.testIndex.length, n * Math.max(1, Math.floor((height - 2) / rowH)));
      grid.current = { cols: n, width, shown };
      const w = width / n;
      ctx.save();
      ctx.textBaseline = 'middle';
      for (let i = 0; i < shown; i++) {
        const col = i % n;
        const row = Math.floor(i / n);
        const x = col * w;
        const y = row * rowH;
        const index = data.testIndex[i];
        const calls = state.testPredictions[i] ?? [];
        const scored = data.scored[index];
        const right = calls.length > 0 && scored.every((pos, k) => calls[k] === data.targets[index][pos]);
        if (i === probe || i === hover) {
          ctx.fillStyle = rgba(palette.accent, i === probe ? 0.14 : 0.07);
          ctx.fillRect(x + 1, y + 1, w - 2, rowH - 2);
        }
        ctx.font = '600 11px ' + MONO_STACK;
        ctx.fillStyle = state.step === 0 ? palette.textFaint : right ? palette.green : palette.orange;
        ctx.textAlign = 'left';
        ctx.fillText(state.step === 0 ? '·' : right ? '✓' : '✗', x + 6, y + rowH / 2);
        ctx.font = (words ? '10.5' : '11') + 'px ' + MONO_STACK;
        ctx.fillStyle = palette.text;
        const input = spellTokens(data, data.inputs[index]);
        const answer = words ? '' : ' → ' + scored.map((_, k) => data.outVocab[calls[k]] ?? '?').join('');
        ctx.fillText(input + answer, x + 20, y + rowH / 2, w - 26);
      }
      ctx.restore();
    },
    [data, hover, probe, state],
  );
  const locate = (pos: { x: number; y: number }) => {
    const { cols, width, shown } = grid.current;
    const i = Math.floor(pos.y / ROW_H) * cols + Math.floor(pos.x / (width / cols));
    return i >= 0 && i < shown ? i : null;
  };
  return (
    <Chart
      draw={draw}
      height={(w) => Math.ceil(Math.min(data.testIndex.length, galleryColumns(data, w) * GALLERY_ROWS) / galleryColumns(data, w)) * ROW_H + 2}
      description={'The first held-out sequences, each ticked when the model gets every scored position right.'}
      cursor={hover !== null ? 'pointer' : 'default'}
      onPointerDown={(pos) => {
        const i = locate(pos);
        if (i !== null) onProbe(i);
      }}
      onPointerMove={(pos) => setHover(pos ? locate(pos) : null)}
      onPointerLeave={() => setHover(null)}
      redrawKey={state.step + ':' + probe + ':' + hover}
    />
  );
}

/* ---------------- curves ---------------- */

export function Curves({ history, view }: { history: TransformerState['history']; view: 'loss' | 'accuracy' }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (history.length < 2) {
        drawEmptyAxes(ctx, width, height, palette, 'step', view === 'loss' ? 'log₁₀ loss' : 'accuracy');
        return;
      }
      const xs = history.map((h) => h.step);
      const train = history.map((h) => (view === 'loss' ? Math.log10(Math.max(h.train, 1e-4)) : h.trainAcc));
      const test = history.map((h) => (view === 'loss' ? Math.log10(Math.max(h.test, 1e-4)) : h.testAcc));
      const [lo, hi] = extentOf(train.concat(test));
      const frame = makeFrame(width, height, [0, xs[xs.length - 1] || 1], view === 'loss' ? padExtent(lo, hi, 0.12) : [0, 1.02], { left: 54, bottom: 30, right: 14, top: 12 });
      drawAxes(ctx, frame, palette, { xLabel: 'step', yLabel: view === 'loss' ? 'log₁₀ loss' : 'accuracy', yTicks: 4 });
      ctx.save();
      ctx.lineWidth = 2;
      for (const [ys, colour] of [
        [train, palette.blue],
        [test, palette.orange],
      ] as const) {
        ctx.strokeStyle = colour;
        ctx.beginPath();
        xs.forEach((x, i) => (i === 0 ? ctx.moveTo(frame.x(x), frame.y(ys[i])) : ctx.lineTo(frame.x(x), frame.y(ys[i]))));
        ctx.stroke();
      }
      ctx.restore();
    },
    [history, view],
  );
  return (
    <Chart
      draw={draw}
      height={190}
      description={history.length < 2 ? 'Training curves, empty until training starts.' : (view === 'loss' ? 'Training and held-out loss' : 'Training and held-out accuracy') + ' over ' + history[history.length - 1].step + ' steps.'}
      redrawKey={history.length + ':' + view}
    />
  );
}

/* ---------------- a labelled heat map ---------------- */

/** A signed grid with a label per row, and the value under the pointer. */
function HeatMap({ values, rows, cols, rowLabels, description, height, plate }: {
  values: ArrayLike<number>;
  rows: number;
  cols: number;
  rowLabels: string[];
  description: string;
  height: number;
  plate: (r: number, c: number) => string[];
}) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const geometry = useRef<{ x: number; y: number; cw: number; ch: number } | null>(null);
  const draw = useCallback(
    ({ ctx, width, height: h, palette }: DrawArgs) => {
      const left = Math.min(60, 8 + Math.max(...rowLabels.map((l) => l.length)) * 7);
      const top = 6;
      const cw = (width - left - 8) / cols;
      const ch = Math.min(22, (h - top - 6) / rows);
      geometry.current = { x: left, y: top, cw, ch };
      let scale = 0;
      for (let i = 0; i < values.length; i++) scale = Math.max(scale, Math.abs(values[i]));
      scale = scale || 1;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const v = values[r * cols + c];
          ctx.fillStyle = mix(palette.surfaceAlt, v < 0 ? palette.negative : palette.positive, Math.min(1, Math.abs(v) / scale));
          ctx.fillRect(left + c * cw, top + r * ch, cw - (cw > 6 ? 1 : 0), ch - 1);
        }
      }
      ctx.save();
      ctx.font = '10px ' + MONO_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      rowLabels.forEach((label, r) => ctx.fillText(label, left - 4, top + (r + 0.5) * ch));
      ctx.restore();
      if (hover) {
        const lines = plate(hover.r, hover.c);
        const size = measurePlate(ctx, lines);
        const at = placePlate(size, { x: left + hover.c * cw, y: top + hover.r * ch, w: cw, h: ch }, width, h);
        drawHoverPlate(ctx, lines, at.left, at.top, palette);
      }
    },
    [values, rows, cols, rowLabels, hover, plate],
  );
  return (
    <Chart
      draw={draw}
      height={height}
      description={description}
      onPointerMove={(pos) => {
        const g = geometry.current;
        if (!g || !pos) {
          setHover(null);
          return;
        }
        const c = Math.floor((pos.x - g.x) / g.cw);
        const r = Math.floor((pos.y - g.y) / g.ch);
        setHover(r >= 0 && c >= 0 && r < rows && c < cols ? { r, c } : null);
      }}
      onPointerLeave={() => setHover(null)}
      redrawKey={hover ? hover.r + ':' + hover.c : ''}
    />
  );
}

const GROUPS = { animals: ['other', 'animals', 'sounds', 'sizes', 'adverbs'], letters: ['start', 'vowels', 'consonants', 'space and stop'], digits: ['start', 'digits'] };

/** Every token's vector on the table's two main directions: tokens the model treats alike land close together. */
export function EmbeddingPlane({ data, plane }: { data: SequenceDataset; plane: { points: Array<{ x: number; y: number }>; explained: number } }) {
  const [hover, setHover] = useState<number | null>(null);
  const placed = useRef<Array<{ x: number; y: number }>>([]);
  const names = data.task === 'animals' ? GROUPS.animals : data.task === 'letters' ? GROUPS.letters : GROUPS.digits;
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const xs = plane.points.map((p) => p.x);
      const ys = plane.points.map((p) => p.y);
      const frame = makeFrame(width, height, padExtent(...extentOf(xs), 0.12), padExtent(...extentOf(ys), 0.12), { left: 14, right: 40, top: 26, bottom: 24, equal: true });
      ctx.save();
      ctx.strokeStyle = palette.grid;
      ctx.beginPath();
      ctx.moveTo(frame.x(0), frame.top);
      ctx.lineTo(frame.x(0), frame.bottom);
      ctx.moveTo(frame.left, frame.y(0));
      ctx.lineTo(frame.right, frame.y(0));
      ctx.stroke();
      const at: Array<{ x: number; y: number }> = [];
      plane.points.forEach((p, token) => {
        const x = frame.x(p.x);
        const y = frame.y(p.y);
        at.push({ x, y });
        const colour = categorical(palette, wordGroup(data, token));
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.arc(x, y, token === hover ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = (token === hover ? '600 ' : '') + '11px ' + MONO_STACK;
        ctx.fillStyle = token === hover ? palette.text : palette.textMuted;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(tokenName(data, token), x + 6, y);
      });
      placed.current = at;
      // The key, top left.
      let lx = 10;
      ctx.font = '10px ' + FONT_STACK;
      names.forEach((name, g) => {
        ctx.fillStyle = categorical(palette, g);
        ctx.beginPath();
        ctx.arc(lx + 4, 12, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = palette.textMuted;
        ctx.fillText(name, lx + 11, 12.5);
        lx += ctx.measureText(name).width + 24;
      });
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'center';
      ctx.fillText('two main directions of the ' + data.vocab.length + ' token vectors · ' + fmtPercent(plane.explained, 0) + ' of their spread', width / 2, height - 9, width - 16);
      ctx.restore();
    },
    [data, hover, names, plane],
  );
  return (
    <Chart
      draw={draw}
      height={260}
      description={'The ' + data.vocab.length + ' token embeddings projected onto their two main directions; tokens the model treats alike sit close together.'}
      onPointerMove={(pos) => {
        if (!pos) return setHover(null);
        let best: number | null = null;
        let dist = 12;
        placed.current.forEach((p, i) => {
          const d = Math.hypot(p.x - pos.x, p.y - pos.y);
          if (d < dist) {
            dist = d;
            best = i;
          }
        });
        setHover(best);
      }}
      onPointerLeave={() => setHover(null)}
      redrawKey={String(hover)}
    />
  );
}

/** The position vectors, one row per position. */
export function PositionMap({ pos, length, width, mode }: { pos: Float64Array; length: number; width: number; mode: string }) {
  return (
    <HeatMap
      values={pos}
      rows={length}
      cols={width}
      rowLabels={Array.from({ length }, (_, t) => 'p' + t)}
      height={Math.min(260, 20 + length * 22)}
      description={mode === 'none' ? 'No position vectors: every row is zero.' : 'The ' + length + ' position vectors, ' + width + ' numbers each' + (mode === 'sinusoidal' ? ': fixed waves, fast on the left, slow on the right.' : ', learned.')}
      plate={(r, c) => ['position ' + r + ' · dim ' + (c + 1), fmt(pos[r * width + c], 2)]}
    />
  );
}
