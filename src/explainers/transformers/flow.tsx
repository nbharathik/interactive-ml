/** The flow view: every token is a lane running down through the network; attention is the only place lanes meet. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailOverlay } from '../../explainer/components/Detail';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt } from '../../lib/math/stats';
import type { TransformerCache, TransformerSpec } from '../../lib/ml/transformer';
import { MONO_STACK, roundRect } from '../../lib/viz/canvas';
import type { Rect } from '../../explainer/architecture';
import { rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { caption, drawChip, drawStrip, headColour, scaleOf } from './draw';
import type { ChipTone } from './draw';
import { NODE, parseNode, rowNorm, tokenName, weightAt } from './model';
import type { NodeKind, Probe } from './model';

type RowKind = 'tokens' | 'strip' | 'band' | 'grid' | 'pred' | 'answer';

interface Row {
  key: string;
  label: string;
  kind: RowKind;
  group: NodeKind;
  layer: number;
  y: number;
  h: number;
}

interface FlowLayout {
  gutter: number;
  laneW: number;
  stripW: number;
  cx: number[];
  rows: Row[];
  footerY: number;
}

const FOOTER = 20;

/** Rows from the tokens at the top to the prediction at the bottom, sized to the frame. */
function layoutFlow(width: number, height: number, T: number, spec: TransformerSpec): FlowLayout {
  const wide = width >= 520;
  const gutter = wide ? 96 : 58;
  const laneW = (width - gutter - 8) / Math.max(1, T);
  const stripW = Math.max(18, Math.min(laneW - 10, spec.width * 7));
  const cx = Array.from({ length: T }, (_, t) => gutter + (t + 0.5) * laneW);

  // A tall frame gets taller strips and a deeper attention band.
  const blocks = spec.layers * (spec.ffn > 0 ? 6 : 3) + 3;
  const strip = Math.max(10, Math.min(16, (height - 260 * spec.layers) / (blocks * 2.2)));
  const inner = 3;
  const between = strip;
  const name = (long: string, short: string) => (wide ? long : short);
  type Spec = Omit<Row, 'y'> & { gapAfter: number };
  const specs: Spec[] = [];
  specs.push({ key: 'tokens', label: 'input', kind: 'tokens', group: 'tok', layer: 0, h: 22, gapAfter: 8 });
  specs.push({ key: 'e', label: name('token vector', 'token'), kind: 'strip', group: 'emb', layer: 0, h: strip, gapAfter: inner });
  specs.push({ key: 'p', label: name('+ position', '+ pos'), kind: 'strip', group: 'emb', layer: 0, h: strip, gapAfter: inner });
  specs.push({ key: 'x0', label: '= stream', kind: 'strip', group: 'emb', layer: 0, h: strip, gapAfter: 0 });
  for (let l = 0; l < spec.layers; l++) {
    specs.push({ key: 'band' + l, label: spec.layers > 1 ? name('attention ' + (l + 1), 'attn ' + (l + 1)) : name('attention', 'attn'), kind: 'band', group: 'attn', layer: l, h: 0, gapAfter: 0 });
    specs.push({ key: 'att' + l, label: name('+ attention', '+ attn'), kind: 'strip', group: 'add', layer: l, h: strip, gapAfter: inner });
    specs.push({ key: 'mid' + l, label: '= stream', kind: 'strip', group: 'add', layer: l, h: strip, gapAfter: spec.ffn > 0 ? between : 0 });
    if (spec.ffn > 0) {
      specs.push({ key: 'units' + l, label: name('feed-forward', 'ffn'), kind: 'grid', group: 'ffn', layer: l, h: Math.round(strip * 1.9), gapAfter: inner });
      specs.push({ key: 'ff' + l, label: name('+ feed-forward', '+ ffn'), kind: 'strip', group: 'ffn', layer: l, h: strip, gapAfter: inner });
      specs.push({ key: 'out' + l, label: '= stream', kind: 'strip', group: 'ffn', layer: l, h: strip, gapAfter: 0 });
    }
  }
  specs.push({ key: 'pred', label: name('prediction', 'output'), kind: 'pred', group: 'pred', layer: 0, h: 30, gapAfter: 4 });
  specs.push({ key: 'answer', label: 'answer', kind: 'answer', group: 'pred', layer: 0, h: 18, gapAfter: 0 });

  // The top strip stays clear for a lesson's tag.
  const clear = 28;
  const fixed = specs.reduce((sum, s) => sum + s.h + s.gapAfter, 0) + between * 2;
  const room = height - FOOTER - clear - 4 - fixed;
  const band = Math.max(44, Math.min(230, room / spec.layers));
  let used = fixed + band * spec.layers;
  const top = Math.max(clear, (height - FOOTER - used) / 2);
  let y = top;
  const rows: Row[] = [];
  specs.forEach((s, i) => {
    const h = s.kind === 'band' ? band : s.h;
    rows.push({ key: s.key, label: s.label, kind: s.kind, group: s.group, layer: s.layer, y, h });
    y += h + s.gapAfter;
    // The prediction sits a little apart from the last block.
    if (specs[i + 1]?.kind === 'pred') y += between * 2;
  });
  used = y;
  return { gutter, laneW, stripW, cx, rows, footerY: Math.min(height - FOOTER / 2, used + FOOTER) };
}

/** The rectangle a component covers in one lane. */
function groupRect(layout: FlowLayout, kind: NodeKind, layer: number, t: number): Rect | null {
  const rows = layout.rows.filter((r) => r.group === kind && (kind === 'tok' || kind === 'emb' || kind === 'pred' || r.layer === layer));
  if (rows.length === 0) return null;
  const y0 = rows[0].y;
  const last = rows[rows.length - 1];
  const w = kind === 'attn' ? layout.laneW - 2 : layout.stripW + 6;
  return { x: layout.cx[t] - w / 2, y: y0 - 3, w, h: last.y + last.h - y0 + 6 };
}

function values(cache: TransformerCache, key: string): { data: Float64Array; cols: number } | null {
  if (key === 'e') return { data: cache.embed, cols: cache.embed.length / cache.length };
  if (key === 'p') return { data: cache.pos, cols: cache.pos.length / cache.length };
  if (key === 'x0') return { data: cache.x0, cols: cache.x0.length / cache.length };
  const layer = cache.layers[Number(key.replace(/\D+/g, ''))];
  if (!layer) return null;
  const d = layer.x.length / cache.length;
  if (key.startsWith('att')) return { data: layer.att, cols: d };
  if (key.startsWith('mid')) return { data: layer.mid, cols: d };
  if (key.startsWith('ff') && layer.ff) return { data: layer.ff, cols: d };
  if (key.startsWith('out')) return { data: layer.out, cols: d };
  if (key.startsWith('units') && layer.act) return { data: layer.act, cols: layer.act.length / cache.length };
  return null;
}

function same(a: ArchSelection | null, b: ArchSelection | null): boolean {
  if (!a || !b) return a === b;
  return a.kind === b.kind && a.id === b.id;
}

export interface FlowProps {
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  probe: Probe;
  selection: ArchSelection | null;
  onSelect: (next: ArchSelection | null) => void;
  spot?: ArchSelection | null;
  renderDetail?: (selection: ArchSelection, open: (next: ArchSelection) => void) => ReactNode;
  summary: string;
  /** Write the focused token's weights over the keys. */
  numbers?: boolean;
  description: string;
  redrawKey?: unknown;
}

export function FlowView({ data, spec, cache, probe, selection, onSelect, spot = null, renderDetail, summary, numbers = false, description, redrawKey }: FlowProps) {
  const [hover, setHover] = useState<ArchSelection | null>(null);
  const layoutRef = useRef<FlowLayout | null>(null);
  const shown = hover ?? spot;
  const spotlit = hover === null && spot !== null;
  const T = cache.length;

  const plateLines = useCallback(
    (target: ArchSelection): string[] => {
      const { kind, layer, position: t } = parseNode(target.id);
      const name = tokenName(data, cache.tokens[t]);
      const d = spec.width;
      switch (kind) {
        case 'tok':
          return [name, 'position ' + t];
        case 'emb':
          return ['x = e + p', '|x| = ' + fmt(rowNorm(cache.x0, t, d), 2)];
        case 'attn': {
          let best = { h: 0, j: 0, w: -1 };
          for (let h = 0; h < spec.heads; h++) for (let j = 0; j < T; j++) {
            const w = weightAt(cache, layer, h, t, j);
            if (w > best.w) best = { h, j, w };
          }
          return [name + ' looks at', tokenName(data, cache.tokens[best.j]) + ' (' + best.j + ') · ' + fmt(best.w, 2) + (spec.heads > 1 ? ' · head ' + (best.h + 1) : '')];
        }
        case 'add':
          return ['+ attention', '|Δ| = ' + fmt(rowNorm(cache.layers[layer].att, t, d), 2)];
        case 'ffn': {
          const act = cache.layers[layer].act;
          let on = 0;
          if (act) for (let u = 0; u < spec.ffn; u++) if (act[t * spec.ffn + u] > 0) on++;
          return ['feed-forward', on + ' of ' + spec.ffn + ' units on'];
        }
        case 'pred': {
          const c = cache.predicted[t];
          return [data.outVocab[c] ?? '?', 'p = ' + fmt(cache.probs[t * spec.outVocab + c], 2)];
        }
        default:
          return [];
      }
    },
    [cache, data, spec, T],
  );

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const layout = layoutFlow(width, height, T, spec);
      layoutRef.current = layout;
      const focus = selection ?? shown;
      const focusLane = focus && focus.kind === 'node' ? parseNode(focus.id) : null;

      // Gutter labels, one per row.
      for (const row of layout.rows) {
        if (row.kind === 'band') {
          caption(ctx, row.label, layout.gutter - 10, row.y + row.h / 2 - 6, palette);
          if (spec.causal) caption(ctx, 'causal', layout.gutter - 10, row.y + row.h / 2 + 7, palette, 'right', true);
          continue;
        }
        caption(ctx, row.label, layout.gutter - 10, row.y + row.h / 2, palette, 'right', row.label.startsWith('=') || row.label.startsWith('+'));
      }

      // One colour scale per row, so the lanes compare.
      const scales = new Map<string, number>();
      for (const row of layout.rows) {
        const v = values(cache, row.key);
        if (v) scales.set(row.key, scaleOf(v.data));
      }

      for (const row of layout.rows) {
        if (row.kind === 'band') {
          drawBand(ctx, layout, row, cache, spec, palette, focusLane, numbers);
          continue;
        }
        for (let t = 0; t < T; t++) {
          const x = layout.cx[t] - layout.stripW / 2;
          if (row.kind === 'tokens') {
            ctx.save();
            ctx.font = '9px ' + MONO_STACK;
            ctx.fillStyle = palette.textFaint;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            if (row.y > 12) ctx.fillText(String(t), layout.cx[t], row.y - 1);
            ctx.restore();
            drawChip(ctx, tokenName(data, cache.tokens[t]), layout.cx[t], row.y + row.h / 2, Math.min(layout.stripW, 56), row.h, palette, t === 0 ? 'start' : 'plain');
          } else if (row.kind === 'strip' || row.kind === 'grid') {
            const v = values(cache, row.key);
            if (!v) continue;
            const slice = v.data.subarray(t * v.cols, t * v.cols + v.cols);
            drawStrip(ctx, slice, x, row.y, layout.stripW, row.h, palette, scales.get(row.key) ?? 1, row.kind === 'grid' ? { magnitude: true, rows: 4, tint: palette.positive } : {});
          } else if (row.kind === 'pred') {
            const c = cache.predicted[t];
            const target = probe.targets[t] ?? -1;
            const scored = probe.scored.includes(t);
            const tone: ChipTone = target < 0 ? 'ghost' : scored ? (c === target ? 'right' : 'wrong') : 'plain';
            const p = cache.probs[t * spec.outVocab + c];
            const chipW = Math.min(layout.stripW + 4, 64);
            drawChip(ctx, data.outVocab[c] ?? '?', layout.cx[t], row.y + 11, chipW, 22, palette, tone);
            ctx.fillStyle = palette.surfaceAlt;
            ctx.fillRect(layout.cx[t] - chipW / 2, row.y + 25, chipW, 3);
            ctx.fillStyle = target < 0 ? rgba(palette.textFaint, 0.6) : rgba(palette.accent, 0.85);
            ctx.fillRect(layout.cx[t] - chipW / 2, row.y + 25, chipW * p, 3);
          } else if (row.kind === 'answer') {
            const target = probe.targets[t] ?? -1;
            if (target < 0) continue;
            ctx.save();
            ctx.font = (probe.scored.includes(t) ? '600 ' : '') + '11px ' + MONO_STACK;
            ctx.fillStyle = probe.scored.includes(t) ? palette.textMuted : palette.textFaint;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(data.outVocab[target] ?? '?', layout.cx[t], row.y + row.h / 2, layout.laneW - 4);
            ctx.restore();
          }
        }
      }

      // The open card and the hovered component.
      const outline = (target: ArchSelection, state: 'hover' | 'open' | 'spot') => {
        const { kind, layer, position } = parseNode(target.id);
        const rect = groupRect(layout, kind, layer, position);
        if (!rect) return null;
        strokeOutline(ctx, () => roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 4), state, palette);
        return rect;
      };
      if (selection) outline(selection, 'open');
      if (shown && !(selection && same(shown, selection))) {
        const rect = outline(shown, spotlit ? 'spot' : 'hover');
        const lines = plateLines(shown);
        if (rect && lines.length > 0) {
          const plate = measurePlate(ctx, lines);
          const at = placePlate(plate, rect, width, height);
          drawHoverPlate(ctx, lines, at.left, at.top, palette);
        }
      }

      ctx.save();
      ctx.font = '11px ' + MONO_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(summary, width / 2, Math.min(height - 9, layout.footerY), width - 24);
      ctx.restore();
    },
    [T, cache, data, probe, selection, shown, spotlit, spec, summary, plateLines, numbers],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): ArchSelection | null => {
      const layout = layoutRef.current;
      if (!layout || pos.x < layout.gutter - 4) return null;
      const t = Math.max(0, Math.min(T - 1, Math.floor((pos.x - layout.gutter) / layout.laneW)));
      for (const row of layout.rows) {
        if (pos.y < row.y - 3 || pos.y > row.y + row.h + 3) continue;
        if (row.kind !== 'band' && Math.abs(pos.x - layout.cx[t]) > layout.stripW / 2 + 6) return null;
        const id =
          row.group === 'tok' ? NODE.token(t)
          : row.group === 'emb' ? NODE.embed(t)
          : row.group === 'attn' ? NODE.attn(row.layer, t)
          : row.group === 'add' ? NODE.add(row.layer, t)
          : row.group === 'ffn' ? NODE.ffn(row.layer, t)
          : NODE.pred(t);
        return { kind: 'node', id };
      }
      return null;
    },
    [T],
  );

  const targets = useMemo<ArchSelection[]>(() => {
    const list: string[] = [];
    for (let t = 0; t < T; t++) list.push(NODE.token(t));
    for (let t = 0; t < T; t++) list.push(NODE.embed(t));
    for (let l = 0; l < spec.layers; l++) {
      for (let t = 0; t < T; t++) list.push(NODE.attn(l, t));
      for (let t = 0; t < T; t++) list.push(NODE.add(l, t));
      if (spec.ffn > 0) for (let t = 0; t < T; t++) list.push(NODE.ffn(l, t));
    }
    for (let t = 0; t < T; t++) list.push(NODE.pred(t));
    return list.map((id) => ({ kind: 'node' as const, id }));
  }, [T, spec.layers, spec.ffn]);

  const handleKey = useDiagramKeys<ArchSelection>({
    targets,
    cursor: hover ?? selection,
    same,
    onCursor: setHover,
    onOpen: (target) => onSelect(target),
    onClose: () => {
      onSelect(null);
      setHover(null);
    },
  });

  // A card whose component has gone (a shorter sequence, one layer fewer) closes with it.
  useEffect(() => {
    if (selection && !targets.some((t) => same(t, selection))) onSelect(null);
  }, [targets, selection, onSelect]);

  const cursorText = useMemo(() => {
    const cursor = hover ?? selection;
    return cursor ? plateLines(cursor).join(', ') : '';
  }, [hover, selection, plateLines]);
  const detail = selection && renderDetail ? renderDetail(selection, onSelect) : null;
  const key = (shown ? shown.id : '') + '|' + (selection ? selection.id : '');

  return (
    <div
      className="mlx-arch"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys move between components, Enter opens one.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHover(null);
      }}
    >
      <span className="mlx-visually-hidden" aria-live="polite">
        {cursorText}
      </span>
      <div className="mlx-arch__stage">
        <Chart
          draw={draw}
          height="fill"
          description={description}
          cursor={hover ? 'pointer' : 'default'}
          onPointerDown={(pos) => {
            const found = locate(pos);
            onSelect(found && same(found, selection) ? null : found);
          }}
          onPointerMove={(pos) => {
            const found = pos ? locate(pos) : null;
            setHover((prev) => (same(prev, found) ? prev : found));
          }}
          onPointerLeave={() => setHover(null)}
          redrawKey={String(redrawKey) + '|' + key}
        />
      </div>
      {detail ? (
        <DetailOverlay dock onClose={() => onSelect(null)}>
          {detail}
        </DetailOverlay>
      ) : null}
    </div>
  );
}

/** The attention band: a line from every key above to every query below, as strong as the weight. */
function drawBand(
  ctx: CanvasRenderingContext2D,
  layout: FlowLayout,
  row: Row,
  cache: TransformerCache,
  spec: TransformerSpec,
  palette: Palette,
  focus: { kind: NodeKind; layer: number; position: number } | null,
  numbers: boolean,
): void {
  const T = cache.length;
  const y0 = row.y + 2;
  const y1 = row.y + row.h - 2;
  const spread = Math.min(10, layout.stripW / (spec.heads + 1));
  // A focused lane keeps its own lines; the rest recede.
  const lane = focus && (focus.kind !== 'attn' || focus.layer === row.layer) ? focus.position : null;
  ctx.save();
  ctx.lineCap = 'round';
  // Receding lines first, the focused lane's on top.
  const passes = lane === null ? [null] : [false, true];
  for (const wanted of passes) {
    for (let h = 0; h < spec.heads; h++) {
      const hue = headColour(palette, h);
      const dx = (h - (spec.heads - 1) / 2) * spread;
      for (let i = 0; i < T; i++) {
        const mine = lane === null || lane === i;
        if (wanted !== null && mine !== wanted) continue;
        for (let j = 0; j < T; j++) {
          const w = weightAt(cache, row.layer, h, i, j);
          if (w < 0.02) continue;
          const alpha = (mine ? 0.9 : 0.12) * Math.pow(w, 0.75);
          const xa = layout.cx[j] + dx;
          const xb = layout.cx[i] + dx;
          ctx.strokeStyle = rgba(hue, alpha);
          ctx.lineWidth = 0.8 + 2.6 * w;
          ctx.beginPath();
          ctx.moveTo(xa, y0);
          ctx.bezierCurveTo(xa, y0 + (y1 - y0) * 0.5, xb, y1 - (y1 - y0) * 0.5, xb, y1);
          ctx.stroke();
        }
      }
    }
  }
  // The focused query's weights, written where its lines leave the keys.
  if (numbers && lane !== null && row.h >= 60) {
    ctx.font = '9px ' + MONO_STACK;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let j = 0; j < T; j++) {
      let top = 0;
      let head = 0;
      for (let h = 0; h < spec.heads; h++) {
        const w = weightAt(cache, row.layer, h, lane, j);
        if (w > top) {
          top = w;
          head = h;
        }
      }
      if (top < 0.05) continue;
      ctx.fillStyle = headColour(palette, head);
      ctx.fillText(top.toFixed(2).replace(/^0/, ''), layout.cx[j], y0 + 2);
    }
  }
  ctx.restore();
}
