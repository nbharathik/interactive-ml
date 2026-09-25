/** The sheets: the network over the input plane. Units mode walks the pipeline: the input, every hidden layer's units as sheets (the picked layer big, the rest as flat strips) and the sum the next stage makes of them; Output mode is ŷ alone as one surface. */

import { useCallback, useEffect, useMemo, useRef } from 'react';
import type { KeyboardEvent } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import { useTurntable } from '../../explainer/useTurntable';
import { fmt, sub } from '../../lib/math/stats';
import type { ActivationFn } from '../../lib/ml/activations';
import { forward } from '../../lib/ml/mlp';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { FONT_STACK, MONO_STACK, clamp, drawMarker, linearScale } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { categorical, mix, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawCategoryField, drawSignedField } from '../../lib/viz/plots';
import { drawSurface } from '../../lib/viz/surface';
import type { SurfaceHandle, SurfaceMark, SurfaceOptions } from '../../lib/viz/surface';
import { drawArrow, ease, paneRects, useReplay } from './space';
import { sameTarget, targetKey, unitLayer } from './targets';
import type { DiagramTarget, Spot } from './targets';
import { TILE_N, creaseSegments, drawUnitTile, rowRange, sampleUnits, signedColour, tileNorm, weightedSum } from './tiles';

export type SurfaceMode = 'output' | 'units';

export interface SurfaceDiagramProps {
  state: TrainerState;
  act: ActivationFn;
  mode: SurfaceMode;
  /** The hidden layer drawn big, from 1. */
  layer: number;
  onPick: (layer: number) => void;
  /** Training inputs in network units and their class labels. */
  inputs: readonly (readonly number[])[];
  labels: readonly number[];
  classCount: number;
  lossName: string;
  /** Multiplies network units back to the raw axes. */
  inputScale: number;
  /** Signed score in [-1, 1] at an input in network units. */
  scoreAt: (u: number, v: number) => number;
  classAt: (u: number, v: number) => number;
  hover: DiagramTarget | null;
  open: DiagramTarget | null;
  spot: Spot | null;
  onHover: (target: DiagramTarget | null) => void;
  onOpen: (target: DiagramTarget | null) => void;
  description: string;
  redrawKey: string;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Vertices per side of the output mesh. */
export const SURFACE_N = 29;

export function uOf(i: number): number {
  return -1 + (2 * i) / (SURFACE_N - 1);
}

/** Unit j of hidden layer k as its weight row and bias. */
export function unitOf(state: TrainerState, j: number, k = 1): { w: number[]; b: number } {
  const layer = state.net.layers[Math.max(0, Math.min(k, state.net.layers.length) - 1)];
  return { w: Array.from({ length: layer.inSize }, (_, i) => layer.W[j * layer.inSize + i]), b: layer.b[j] };
}

/** The output unit's weight on every unit of the last hidden layer, and its bias. */
export function outputWeights(state: TrainerState): { v: number[]; b: number } {
  const last = state.net.layers[state.net.layers.length - 1];
  return { v: Array.from({ length: last.inSize }, (_, i) => last.W[i]), b: last.b[0] };
}

/** True when the output is one unit reading the shown layer, so the sum of its sheets is the output. */
export function sumShown(state: TrainerState, layer: number): boolean {
  const hiddenCount = state.net.layers.length - 1;
  return hiddenCount > 0 && layer === hiddenCount && state.net.layers[hiddenCount].outSize === 1;
}

/* ---------------- pieces ---------------- */

/** The highest screen point of a drawn box, so a label can sit just above it. */
function boxTop(handle: SurfaceHandle): number {
  return Math.min(handle.project(-1, -1, 1).y, handle.project(1, -1, 1).y, handle.project(1, 1, 1).y, handle.project(-1, 1, 1).y);
}

function frameOf(rect: Rect): Frame {
  return { x: linearScale([-1, 1], [rect.x, rect.x + rect.w]), y: linearScale([-1, 1], [rect.y + rect.h, rect.y]), left: rect.x, top: rect.y, right: rect.x + rect.w, bottom: rect.y + rect.h, innerWidth: rect.w, innerHeight: rect.h };
}

/** The input plane as a small square: the points, the axes and their limits. */
function drawInputTile(ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette, inputs: readonly (readonly number[])[], labels: readonly number[], scale: number): void {
  const frame = frameOf(rect);
  ctx.fillStyle = rgba(palette.surfaceAlt, 0.6);
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.save();
  ctx.setLineDash([2, 3]);
  ctx.strokeStyle = rgba(palette.axis, 0.7);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(frame.x(0), rect.y);
  ctx.lineTo(frame.x(0), rect.y + rect.h);
  ctx.moveTo(rect.x, frame.y(0));
  ctx.lineTo(rect.x + rect.w, frame.y(0));
  ctx.stroke();
  ctx.restore();
  inputs.forEach((p, i) => drawMarker(ctx, 'circle', frame.x(p[0]), frame.y(p[1] ?? 0), rect.w > 60 ? 1.8 : 1.4, categorical(palette, labels[i])));
  ctx.strokeStyle = palette.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  ctx.font = '9px ' + MONO_STACK;
  ctx.fillStyle = palette.textFaint;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'right';
  ctx.fillText('x₁ →', rect.x + rect.w - 3, rect.y + rect.h - 3);
  ctx.textAlign = 'left';
  ctx.fillText('x₂ ↑', rect.x + 3, rect.y + 9);
  ctx.textAlign = 'center';
  ctx.fillText(String(-scale), rect.x, rect.y + rect.h + 10);
  ctx.fillText(String(scale), rect.x + rect.w, rect.y + rect.h + 10);
}

/** The network's answer over the plane as a small square: the score field and its boundary. */
function drawOutputTile(
  ctx: CanvasRenderingContext2D,
  rect: Rect,
  palette: Palette,
  inputs: readonly (readonly number[])[],
  labels: readonly number[],
  multiclass: boolean,
  scoreAt: (u: number, v: number) => number,
  classAt: (u: number, v: number) => number,
): void {
  const frame = frameOf(rect);
  ctx.fillStyle = palette.surface;
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  if (multiclass) drawCategoryField(ctx, frame, palette, classAt, { cellSize: 3, alpha: 0.3 });
  else drawSignedField(ctx, frame, palette, scoreAt, { cellSize: 3, alpha: 0.4, contourWidth: 1.2 });
  inputs.forEach((p, i) => drawMarker(ctx, 'circle', frame.x(p[0]), frame.y(p[1] ?? 0), rect.w > 60 ? 1.8 : 1.4, categorical(palette, labels[i])));
  ctx.strokeStyle = palette.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
}

/* ---------------- layout ---------------- */

const PAD = 8;
const LABEL_W = 30;
const ARROW_H = 30;
const TITLE_H = 18;
const EQ_H = 18;
const GAP = 8;

interface Row {
  kind: 'input' | 'strip' | 'big' | 'output';
  layer: number;
  y: number;
  h: number;
}

/** Rows top to bottom with an arrow between each pair; the big row takes what the small ones leave, up to what it wants, and the stack sits in the middle. */
function layoutRows(height: number, hidden: number, picked: number, withOutputRow: boolean, stripSide: number, inputSide: number, bigWanted: number): Row[] {
  const rows: Row[] = [{ kind: 'input', layer: 0, y: 0, h: inputSide + 12 }];
  for (let k = 1; k <= hidden; k++) rows.push({ kind: k === picked ? 'big' : 'strip', layer: k, y: 0, h: TITLE_H + stripSide });
  if (withOutputRow) rows.push({ kind: 'output', layer: hidden + 1, y: 0, h: inputSide + 4 });
  const fixed = rows.reduce((sum, row) => (row.kind === 'big' ? sum : sum + row.h), 0) + ARROW_H * (rows.length - 1);
  const bigH = Math.min(bigWanted, Math.max(120, height - PAD * 2 - fixed));
  let y = PAD + Math.max(0, (height - PAD * 2 - fixed - bigH) / 2);
  for (const row of rows) {
    if (row.kind === 'big') row.h = bigH;
    row.y = y;
    y += row.h + ARROW_H;
  }
  return rows;
}

/* ---------------- the diagram ---------------- */

export function SurfaceDiagram(props: SurfaceDiagramProps) {
  const { state, act, mode, layer, onPick, inputs, labels, classCount, lossName, inputScale, scoreAt, classAt, hover, open, spot, onHover, onOpen, description, redrawKey } = props;
  const turntable = useTurntable();
  const { view } = turntable;
  const zones = useRef<{ rect: Rect; target: DiagramTarget; pick?: number }[]>([]);
  const sumRef = useRef<{ rect: Rect; handle: SurfaceHandle; target: DiagramTarget } | null>(null);
  const focus = open ?? hover;
  const multiclass = classCount > 2;
  const hiddenCount = state.net.layers.length - 1;
  const k = Math.max(1, Math.min(layer, hiddenCount));
  const replay = useReplay(1);
  const { play } = replay;
  useEffect(() => {
    play();
  }, [k, act.name, mode, play]);

  const outputGrid = useMemo(() => {
    const values = new Float32Array(SURFACE_N * SURFACE_N);
    for (let r = 0; r < SURFACE_N; r++) for (let c = 0; c < SURFACE_N; c++) values[r * SURFACE_N + c] = scoreAt(uOf(c), uOf(r));
    return values;
  }, [scoreAt]);

  // Every hidden unit over the plane, and every training point through the network, once per state.
  const fields = useMemo(() => (mode === 'units' ? sampleUnits(state) : []), [mode, state]);
  const pointCaches = useMemo(() => (mode === 'units' ? inputs.map((p) => forward(state.net, p, state.spec)) : []), [mode, inputs, state]);
  const picked = fields[k - 1] ?? null;

  // The sum on the right: the output when this is the last layer, else one unit of the next layer (the hot one, else the first).
  const hotUnitOf = useCallback((kk: number, t: DiagramTarget | null) => (t && t.kind === 'unit' && unitLayer(t) === kk ? t.index : -1), []);
  const isLast = k === hiddenCount;
  const nextUnit = isLast ? 0 : Math.max(0, hotUnitOf(k + 1, hover) >= 0 ? hotUnitOf(k + 1, hover) : hotUnitOf(k + 1, open));
  const sumWeights = useMemo(() => {
    if (!picked) return null;
    if (isLast) return sumShown(state, k) ? { ...outputWeights(state), label: 'z, then ŷ = σ(z)', symbol: 'z', target: { kind: 'output' } as DiagramTarget } : null;
    const { w, b } = unitOf(state, nextUnit, k + 1);
    return { v: w, b, label: 'z' + sub(nextUnit + 1) + ' of layer ' + (k + 1) + ' = Σ w·a + b', symbol: 'z' + sub(nextUnit + 1), target: { kind: 'unit', index: nextUnit, layer: k + 1 } as DiagramTarget };
  }, [picked, isLast, state, k, nextUnit]);
  const sum = useMemo(() => {
    if (!picked || !sumWeights) return null;
    const values = weightedSum(picked, sumWeights.v, sumWeights.b);
    let max = 1e-6;
    for (let i = 0; i < values.length; i++) max = Math.max(max, Math.abs(values[i]));
    const points = pointCaches.map((caches) => caches[k - 1].a.reduce((s, a, j) => s + (sumWeights.v[j] ?? 0) * a, sumWeights.b));
    return { values, max, points };
  }, [picked, sumWeights, pointCaches, k]);
  const creases = useMemo(() => (picked ? picked.units.map((u) => creaseSegments(u.z, '', 1, [3, 3])) : []), [picked]);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      zones.current = [];
      sumRef.current = null;
      const hot = focus;
      if (mode === 'output' || !picked) {
        const rect = { x: 8, y: 8, w: width - 16, h: height - 16 };
        const marks: SurfaceMark[] = inputs.map((p, i) => ({ u: p[0], v: p[1] ?? 0, z: labels[i] === 1 ? 1 : 0, colour: categorical(palette, labels[i]), radius: 3 }));
        const handle = drawSurface(ctx, rect, palette, outputGrid, SURFACE_N, SURFACE_N, view, {
          zOf: (s) => (s + 1) / 2,
          colourAt: (s, u, v) => (multiclass ? mix(palette.surface, categorical(palette, classAt(u, v)), 0.55) : mix(palette.surface, s < 0 ? palette.negative : palette.positive, 0.12 + 0.68 * Math.min(1, Math.abs(s)))),
          level: multiclass ? undefined : 0,
          marks,
          labels: { u: 'x₁', v: 'x₂', z: lossName === 'hinge' ? 'score' : 'ŷ' },
          ticks: { u: [String(-inputScale), String(inputScale)], v: [String(-inputScale), String(inputScale)], z: lossName === 'hinge' ? { lo: '−', hi: '+', zero: 0.5 } : { lo: '0', hi: '1', zero: 0.5 } },
        });
        sumRef.current = { rect, handle, target: { kind: 'output' } };
        drawVertexMark(ctx, handle, hot, open, palette, (u, v) => (scoreAt(u, v) + 1) / 2);
        return;
      }

      const t = ease(replay.t);
      const f = act.name === 'linear' ? 'z' : act.label.toLowerCase();
      const n = picked.units.length;
      const inputSide = clamp(width * 0.11, 52, 76);
      const stripSide = clamp((width - LABEL_W - PAD * 2 - GAP * (n - 1)) / n, 34, 58);
      const left = PAD + LABEL_W;
      const spanW = width - left - PAD;
      // The sheets in a grid beside the sum when they can be 64px, else a grid over the sum with all the height.
      const withSum = sum !== null && sumWeights !== null;
      const sumW = withSum ? clamp(spanW * 0.3, 110, 300) : 0;
      const sheetsW = spanW - (withSum ? sumW + GAP * 2 : 0);
      const fixedH = inputSide + 12 + (hiddenCount - 1) * (TITLE_H + stripSide) + (!isLast || !sumWeights ? inputSide + 4 : 0) + ARROW_H * (hiddenCount + (!isLast || !sumWeights ? 1 : 0));
      const zoneAvail = Math.max(120, height - PAD * 2 - fixedH) - TITLE_H - (withSum ? EQ_H : 0);
      const grid = paneRects(sheetsW, Math.min(zoneAvail, 2 * 240 + GAP), n, GAP);
      const gridCols = new Set(grid.map((r) => r.x)).size;
      const wide = !withSum || (grid[0].w >= 64 && (gridCols >= 2 || n === 1));
      const gridRows = wide ? Math.ceil(n / gridCols) : 0;
      const bigWanted = wide ? TITLE_H + Math.max(gridRows * grid[0].h + (gridRows - 1) * GAP, sumW) + (withSum ? EQ_H : 0) : Infinity;
      const rows = layoutRows(height, hiddenCount, k, !isLast || !sumWeights, stripSide, inputSide, bigWanted);
      const spotUnit = spot && spot.target.kind === 'unit' ? { layer: unitLayer(spot.target), index: spot.target.index } : null;

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
            const place = placePlate(size, { x: rect.x, y: rect.y + rect.h - 24, w: rect.w, h: 0 }, width, height);
            drawHoverPlate(ctx, [label], place.left, place.top, palette);
          }
        });
      };
      const title = (x: number, y: number, text: string, hotText: boolean) => {
        ctx.font = '600 11px ' + MONO_STACK;
        ctx.fillStyle = hotText ? palette.accent : palette.text;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(text, x, y);
      };
      // A note fits or gives way to its short form, or to nothing.
      const note = (x: number, y: number, text: string, short = '') => {
        ctx.font = '10px ' + FONT_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        const room = width - PAD - x;
        const shown = ctx.measureText(text).width <= room ? text : ctx.measureText(short).width <= room ? short : '';
        if (shown) ctx.fillText(shown, x, y);
      };

      /* Arrows between the rows, each with what happens on the way. */
      for (let i = 1; i < rows.length; i++) {
        const above = rows[i - 1];
        const below = rows[i];
        const kk = below.layer;
        const x = left + inputSide / 2;
        const y0 = above.y + above.h + 4;
        const y1 = below.y - 4;
        drawArrow(ctx, [[x, y0], [x, y1]], palette);
        const mid = (y0 + y1) / 2;
        const outSize = state.net.layers[state.net.layers.length - 1].outSize;
        const read = lossName === 'softmaxCE' ? ' · softmax' : lossName === 'hinge' ? '' : ' · σ';
        const symbols = below.kind === 'output' ? (outSize === 1 ? 'v, b' : 'W' + sub(kk) + ', b' + sub(kk)) + read : 'W' + sub(kk) + ', b' + sub(kk) + (act.name === 'linear' ? '' : ' then ' + f);
        const what = below.kind === 'output' ? (outSize === 1 ? 'the output unit' : 'the output units') : act.name === 'linear' ? 'weighted sums, nothing bends' : 'weighted sums, then each one bent';
        title(x + 12, mid + 4, symbols, false);
        ctx.font = '600 11px ' + MONO_STACK;
        note(x + 12 + ctx.measureText(symbols).width + 8, mid + 4, what, below.kind === 'output' ? '' : 'sums, then bent');
      }

      /* The rows. */
      for (const row of rows) {
        if (row.kind === 'input') {
          const rect = { x: left, y: row.y, w: inputSide, h: inputSide };
          drawInputTile(ctx, rect, palette, inputs, labels, inputScale);
          title(rect.x + rect.w + 12, rect.y + 11, 'x = (x₁, x₂)', hot !== null && hot.kind === 'layer' && hot.index === 0);
          note(rect.x + rect.w + 12, rect.y + 25, 'the input plane; every sheet below stands on it', 'the input plane');
          zones.current.push({ rect, target: { kind: 'layer', index: 0 } });
          if (hot && hot.kind === 'layer' && hot.index === 0) outlineRect(rect, open && open.kind === 'layer' && open.index === 0 ? 'open' : 'hover', null);
          else if (spot && spot.target.kind === 'layer' && spot.target.index === 0) outlineRect(rect, 'spot', spot.label);
          continue;
        }
        if (row.kind === 'output') {
          const rect = { x: left, y: row.y, w: inputSide, h: inputSide };
          drawOutputTile(ctx, rect, palette, inputs, labels, multiclass, scoreAt, classAt);
          const hotOut = hot !== null && hot.kind === 'output';
          title(rect.x + rect.w + 12, rect.y + 11, lossName === 'hinge' ? 'score = z' : lossName === 'softmaxCE' ? 'ŷ = softmax(z)' : 'ŷ = σ(z)', hotOut);
          note(rect.x + rect.w + 12, rect.y + 25, 'the answer over the plane, with its boundary', 'the answer');
          zones.current.push({ rect, target: { kind: 'output' } });
          if (hotOut) outlineRect(rect, open && open.kind === 'output' ? 'open' : 'hover', null);
          else if (spot && spot.target.kind === 'output') outlineRect(rect, 'spot', spot.label);
          continue;
        }
        const kk = row.layer;
        const layerFields = fields[kk - 1];
        if (!layerFields) continue;
        const hotUnit = hotUnitOf(kk, hot);
        if (row.kind === 'strip') {
          const count = layerFields.units.length;
          title(left, row.y + 11, 'h' + sub(kk) + ' = ' + (act.name === 'linear' ? 'z' + sub(kk) : f + '(z' + sub(kk) + ')'), false);
          ctx.font = '600 11px ' + MONO_STACK;
          const tw = ctx.measureText('h' + sub(kk) + ' = ' + f + '(z' + sub(kk) + ')').width;
          note(left + tw + 10, row.y + 11, count + ' units from above; click to see them as sheets', 'click for sheets');
          const norm = tileNorm(act, layerFields, true);
          layerFields.units.forEach((unit, j) => {
            const rect = { x: left + j * (stripSide + GAP), y: row.y + TITLE_H, w: stripSide, h: stripSide };
            const lit = hotUnit === j;
            drawUnitTile(ctx, rect, palette, unit.a, norm, unit.z, { label: 'a' + sub(j + 1), alpha: hotUnit >= 0 && !lit ? 0.45 : 1, creaseColour: lit ? palette.accent : undefined });
            zones.current.push({ rect, target: { kind: 'unit', index: j, layer: kk }, pick: kk });
            if (lit) outlineRect(rect, open && open.kind === 'unit' && open.index === j && unitLayer(open) === kk ? 'open' : 'hover', null);
            else if (spotUnit && spotUnit.layer === kk && spotUnit.index === j) outlineRect(rect, 'spot', spot!.label);
          });
          continue;
        }

        /* The big row: the sheets, then the sum the next stage makes of them. */
        const [lo, hi] = rowRange(act, layerFields, t);
        const half = Math.max(Math.abs(layerFields.aMin), Math.abs(layerFields.aMax), 1e-6);
        const zOf = (v: number) => (v - lo) / (hi - lo);
        const zeroPlane = lo < 0 && hi > 0 ? zOf(0) : undefined;
        const zoneTop = row.y + TITLE_H;
        const zoneH = row.h - TITLE_H - (withSum ? EQ_H : 0);
        title(left, row.y + 11, 'h' + sub(kk) + ' = ' + (act.name === 'linear' ? 'z' + sub(kk) : f + '(z' + sub(kk) + ')'), false);
        ctx.font = '600 11px ' + MONO_STACK;
        const tw = ctx.measureText('h' + sub(kk) + ' = ' + f + '(z' + sub(kk) + ')').width;
        note(left + tw + 10, row.y + 11, 'one sheet per unit; ▶ replays the bend from the flat sum z', 'one sheet per unit');
        let rects: Rect[];
        let sumRect: Rect | null = null;
        if (wide) {
          rects = paneRects(sheetsW, zoneH, n, GAP).map((r) => ({ ...r, x: r.x + left, y: r.y + zoneTop }));
          if (withSum) {
            const sw = Math.min(sumW, zoneH);
            sumRect = { x: left + spanW - sw, y: zoneTop + (zoneH - sw) / 2, w: sw, h: sw };
          }
        } else {
          const gridH = Math.floor(zoneH * 0.5);
          rects = paneRects(spanW, gridH, n, 6).map((r) => ({ ...r, x: r.x + left, y: r.y + zoneTop }));
          sumRect = { x: left, y: zoneTop + gridH + 6, w: spanW, h: zoneH - gridH - 6 };
        }

        layerFields.units.forEach((unit, j) => {
          const rect = rects[j];
          const lit = hotUnit === j;
          const dimmed = hotUnit >= 0 && !lit;
          const values = new Float32Array(TILE_N * TILE_N);
          for (let i = 0; i < values.length; i++) values[i] = unit.z[i] + (unit.a[i] - unit.z[i]) * t;
          const marks: SurfaceMark[] = inputs.map((p, i) => ({ u: p[0], v: p[1] ?? 0, z: zOf(pointCaches[i]?.[kk - 1].a[j] ?? 0), colour: categorical(palette, labels[i]), radius: rect.w > 120 ? 2.4 : 1.8 }));
          const crease = creases[j].map((seg) => ({ ...seg, colour: rgba(lit ? palette.accent : palette.text, 0.6), width: lit ? 1.6 : 1 }));
          const options: SurfaceOptions = {
            zOf,
            colourAt: (v) => signedColour(palette, v, half, lit),
            floorLines: crease,
            zeroPlane,
            alpha: dimmed ? 0.45 : 1,
            marks: t >= 1 ? marks : [],
          };
          if (j === 0) {
            options.labels = { u: 'x₁', v: 'x₂', z: 'a' };
            options.ticks = { u: [String(-inputScale), String(inputScale)], z: { lo: fmt(lo, 1), hi: fmt(hi, 1), zero: zeroPlane } };
          }
          const handle = drawSurface(ctx, { x: rect.x, y: rect.y + 12, w: rect.w, h: rect.h - 12 }, palette, values, TILE_N, TILE_N, view, options);
          const labelY = boxTop(handle) - 5;
          ctx.font = '600 11px ' + MONO_STACK;
          ctx.fillStyle = lit ? palette.accent : palette.textMuted;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText('a' + sub(j + 1), rect.x + 4, labelY);
          if (withSum && sumWeights) {
            const v = sumWeights.v[j] ?? 0;
            ctx.font = (lit ? '600 ' : '') + '10px ' + MONO_STACK;
            ctx.fillStyle = v < 0 ? palette.negative : palette.positive;
            ctx.textAlign = 'right';
            ctx.fillText('× ' + fmt(v, 2), rect.x + rect.w - 3, labelY);
          }
          zones.current.push({ rect, target: { kind: 'unit', index: j, layer: kk } });
          if (lit) outlineRect(rect, open && open.kind === 'unit' && open.index === j && unitLayer(open) === kk ? 'open' : 'hover', null);
          else if (spotUnit && spotUnit.layer === kk && spotUnit.index === j) outlineRect(rect, 'spot', spot!.label);
        });

        if (!withSum || !sumRect || !sum || !sumWeights) continue;
        const sumMax = sum.max;
        // The equation under the row, the hot unit's term lit.
        const eqY = row.y + row.h - 4;
        const terms = sumWeights.v.map((v, j) => (j === 0 ? (v < 0 ? '−' : '') : v < 0 ? ' − ' : ' + ') + fmt(Math.abs(v), 2) + '·a' + sub(j + 1));
        const tail = (sumWeights.b < 0 ? ' − ' : ' + ') + fmt(Math.abs(sumWeights.b), 2);
        ctx.font = '11px ' + MONO_STACK;
        const full = sumWeights.symbol + ' = ' + terms.join('') + tail;
        const compact = ctx.measureText(full).width > spanW;
        const pieces = compact ? [sumWeights.symbol + ' = Σ wⱼ·aⱼ + b', ...(hotUnit >= 0 ? ['   w' + sub(hotUnit + 1) + ' = ' + fmt(sumWeights.v[hotUnit], 2)] : [])] : [sumWeights.symbol + ' = ', ...terms, tail];
        const hotPiece = compact ? 1 : hotUnit + 1;
        let x = left + Math.max(0, (sheetsW - pieces.reduce((w, piece, i) => (ctx.font = (i === hotPiece && hotUnit >= 0 ? '600 ' : '') + '11px ' + MONO_STACK, w + ctx.measureText(piece).width), 0)) / 2);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        pieces.forEach((piece, i) => {
          const litPiece = i === hotPiece && hotUnit >= 0;
          ctx.font = (litPiece ? '600 ' : '') + '11px ' + MONO_STACK;
          ctx.fillStyle = litPiece ? palette.accent : palette.text;
          ctx.fillText(piece, x, eqY);
          x += ctx.measureText(piece).width;
        });

        // The sum: over the plane with its zero line, the points at their own height; a hot unit tints it by its share.
        let shareMax = 1e-6;
        const share = hotUnit >= 0 ? layerFields.units[hotUnit].a : null;
        const hotV = hotUnit >= 0 ? (sumWeights.v[hotUnit] ?? 0) : 0;
        if (share) for (let i = 0; i < share.length; i++) shareMax = Math.max(shareMax, Math.abs(hotV * share[i]));
        const marks: SurfaceMark[] = inputs.map((p, i) => ({ u: p[0], v: p[1] ?? 0, z: clamp((sum.points[i] / sumMax + 1) / 2, 0, 1), colour: categorical(palette, labels[i]), radius: 2.6 }));
        const hotCrease = hotUnit >= 0 ? creases[hotUnit].map((seg) => ({ ...seg, colour: palette.accent, width: 2, dash: [4, 3] })) : [];
        const handle = drawSurface(ctx, { x: sumRect.x, y: sumRect.y + 12, w: sumRect.w, h: sumRect.h - 12 }, palette, sum.values, TILE_N, TILE_N, view, {
          zOf: (z) => clamp((z / sumMax + 1) / 2, 0, 1),
          colourAt: (z, cu, cv) => {
            if (share) {
              const col = Math.round(((cu + 1) / 2) * (TILE_N - 1));
              const row_ = Math.round(((cv + 1) / 2) * (TILE_N - 1));
              const c = hotV * share[row_ * TILE_N + col];
              return mix(palette.surface, c < 0 ? palette.negative : palette.positive, 0.08 + 0.72 * Math.min(1, Math.abs(c) / shareMax));
            }
            return mix(palette.surface, z < 0 ? palette.negative : palette.positive, 0.12 + 0.68 * Math.min(1, Math.abs(z) / sumMax));
          },
          level: 0,
          marks,
          labels: { u: 'x₁', v: 'x₂', z: sumWeights.symbol },
          ticks: { z: { lo: fmt(-sumMax, 1), hi: fmt(sumMax, 1), zero: 0.5 } },
          floorLines: hotCrease,
        });
        sumRef.current = { rect: sumRect, handle, target: sumWeights.target };
        const hotSum = hot !== null && sameTarget(hot, sumWeights.target);
        title(sumRect.x + 4, Math.max(sumRect.y + 10, boxTop(handle) - 6), sumWeights.label, hotSum);
        if (hotSum) outlineRect(sumRect, open && sameTarget(open, sumWeights.target) ? 'open' : 'hover', null);
        else if (spot && sameTarget(spot.target, sumWeights.target)) outlineRect(sumRect, 'spot', spot.label);
        drawVertexMark(ctx, handle, hot, open, palette, (u, v) => {
          const col = Math.round(((u + 1) / 2) * (TILE_N - 1));
          const row_ = Math.round(((v + 1) / 2) * (TILE_N - 1));
          return clamp((sum.values[row_ * TILE_N + col] / sumMax + 1) / 2, 0, 1);
        });
      }
      outlines.forEach((fn) => fn());
    },
    [mode, picked, fields, inputs, labels, outputGrid, view, multiclass, classAt, lossName, focus, scoreAt, hiddenCount, k, isLast, sumWeights, sum, pointCaches, creases, open, spot, act, replay.t, inputScale, hotUnitOf, state.net.layers],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): { target: DiagramTarget; pick?: number } | null => {
      const inside = (r: Rect) => pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h;
      for (const zone of zones.current) if (inside(zone.rect)) return zone;
      const s = sumRef.current;
      if (!s) return null;
      const found = s.handle.nearest(pos, 14);
      if (found) return { target: { kind: 'input', x: -1 + (2 * found.col) / (mode === 'output' ? SURFACE_N - 1 : TILE_N - 1), y: -1 + (2 * found.row) / (mode === 'output' ? SURFACE_N - 1 : TILE_N - 1) } };
      return inside(s.rect) && mode === 'units' ? { target: s.target } : null;
    },
    [mode],
  );

  const handleKey = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.target !== event.currentTarget) return;
      if (turntable.onKey(event)) {
        event.preventDefault();
        return;
      }
      switch (event.key) {
        case 'Home':
          turntable.reset();
          break;
        case 'Enter':
        case ' ':
          if (!hover) return;
          if (hover.kind === 'unit' && unitLayer(hover) !== k) onPick(unitLayer(hover));
          onOpen(hover);
          break;
        case 'Escape':
          event.stopPropagation();
          onOpen(null);
          onHover(null);
          return;
        default:
          return;
      }
      event.preventDefault();
    },
    [turntable, hover, onOpen, onHover, onPick, k],
  );

  return (
    <div
      className="mlx-arch__stage"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys turn the sheets, Home resets the view, Enter opens what is under the pointer.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onHover(null);
      }}
    >
      <Chart
        draw={draw}
        height="fill"
        description={description}
        cursor={turntable.dragging ? 'grabbing' : hover ? 'pointer' : 'grab'}
        drag
        onPointerDown={(pos) => turntable.down(pos)}
        onPointerMove={(pos) => {
          if (turntable.move(pos)) {
            onHover(null);
            return;
          }
          onHover(pos ? (locate(pos)?.target ?? null) : null);
        }}
        onPointerUp={(pos) => {
          if (!turntable.up()) return;
          const found = locate(pos);
          if (found?.pick !== undefined) onPick(found.pick);
          onOpen(found && sameTarget(found.target, open) ? null : (found?.target ?? null));
        }}
        onPointerLeave={() => onHover(null)}
        redrawKey={redrawKey + '|' + mode + '|' + k + '|' + turntable.key + '|' + replay.t.toFixed(3) + '|' + targetKey(hover) + '|' + targetKey(open) + '|' + (spot ? targetKey(spot.target) + spot.label : '')}
      />
      {mode === 'units' ? (
        <button type="button" className="mlx-arch__tool" onClick={replay.play} title="Replay: each sheet bends from its flat weighted sum" aria-label="Replay the bend">
          ▶
        </button>
      ) : null}
      <button type="button" className="mlx-arch__tool mlx-arch__tool--foot" onClick={turntable.reset} title="Drag to turn; this resets the view (Home)" aria-label="Reset the view">
        ↺
      </button>
    </div>
  );
}

/** The vertex under the pointer, or the open one; its numbers show in the readout. */
function drawVertexMark(
  ctx: CanvasRenderingContext2D,
  handle: SurfaceHandle,
  hot: DiagramTarget | null,
  open: DiagramTarget | null,
  palette: Palette,
  zAt: (u: number, v: number) => number,
): void {
  if (!hot || hot.kind !== 'input') return;
  const p = handle.project(hot.x, hot.y, zAt(hot.x, hot.y));
  ctx.beginPath();
  ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
  ctx.strokeStyle = palette.accent;
  ctx.lineWidth = open && open.kind === 'input' ? 2.4 : 1.6;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = palette.accent;
  ctx.fill();
}
