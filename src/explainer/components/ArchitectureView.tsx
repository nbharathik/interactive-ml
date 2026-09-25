/** Draws an `ArchGraph` on canvas. Hover shows a readout, click docks the page's detail card. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { Chart } from './Chart';
import type { DrawArgs } from './Chart';
import { DetailOverlay } from './Detail';
import { FONT_STACK, MONO_STACK, roundRect, scratchBitmap } from '../../lib/viz/canvas';
import { rgba, toRgb } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import {
  edgePath,
  hitTestEdge,
  hitTestNode,
  labelInside,
  layoutArchitecture,
} from '../architecture';
import type { ArchEdge, ArchGraph, ArchLayout, ArchNodeKind, LayoutOptions } from '../architecture';
import { DIM, drawHoverPlate, measurePlate, placePlate, strokeOutline, strokeWire, underWire } from '../diagramStyle';
import type { WireEmphasis } from '../diagramStyle';
import { useDiagramKeys } from '../useDiagramKeys';
import { useElementSize } from '../useElementSize';

export interface ArchSelection {
  kind: 'node' | 'edge';
  id: string;
}

export interface ArchitectureViewProps {
  graph: ArchGraph;
  /** The open component, if any. */
  selection: ArchSelection | null;
  onSelect: (selection: ArchSelection | null) => void;
  /** The detail card for a component, or null. `open` lets the card jump to another component. */
  renderDetail?: (selection: ArchSelection, open: (next: ArchSelection) => void) => ReactNode;
  /** Describes the current architecture for screen readers. */
  description: string;
  /** Redraw trigger for values that change without the graph identity changing. */
  redrawKey?: unknown;
  /** Advances the dash on the active (gradient) edges, usually the iteration count. */
  phase?: number;
  /** A lesson's spotlight, drawn like a hover until the pointer takes over. */
  spot?: ArchSelection | null;
}

function sameSelection(a: ArchSelection | null, b: ArchSelection | null): boolean {
  if (!a || !b) return a === b;
  return a.kind === b.kind && a.id === b.id;
}

/** A lesson target id, `node:<id>` or `edge:<id>`, as a selection. */
export function parseArchTarget(id: string): ArchSelection | null {
  const at = id.indexOf(':');
  const kind = id.slice(0, at);
  if (at < 0 || (kind !== 'node' && kind !== 'edge')) return null;
  return { kind, id: id.slice(at + 1) };
}

/** The two hover lines of a component: its symbol, then the number it holds. */
function plateLines(graph: ArchGraph, target: ArchSelection): string[] {
  if (target.kind === 'node') {
    const node = graph.nodes.find((n) => n.id === target.id);
    return node ? [node.label, node.readout].filter((line): line is string => Boolean(line)) : [];
  }
  const edge = graph.edges.find((e) => e.id === target.id);
  return edge ? [edge.symbol, edge.readout].filter((line): line is string => Boolean(line)) : [];
}

export function ArchitectureView({
  graph,
  selection,
  onSelect,
  renderDetail,
  description,
  redrawKey,
  phase = 0,
  spot = null,
}: ArchitectureViewProps) {
  const [setFrame, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<ArchSelection | null>(null);
  const shown = hover ?? spot;

  // A return path needs a clear band under the nodes.
  const options = useMemo<LayoutOptions>(() => {
    const column = new Map(graph.nodes.map((n) => [n.id, n.column]));
    const hasBackward = graph.edges.some(
      (e) => (column.get(e.to) ?? 0) < (column.get(e.from) ?? 0),
    );
    // Only boxes stacked in a column need room for the labels under them.
    const stacked = graph.nodes.filter((n) => n.rows > 1);
    const thumbs = stacked.some((n) => n.thumbnail);
    return {
      padBottom: hasBackward ? 84 : 34,
      rowGap: thumbs ? (stacked.some((n) => n.thumbnail && n.value) ? 36 : 24) : 10,
    };
  }, [graph]);

  const layout = useMemo(
    () => (size.width > 0 ? layoutArchitecture(graph, size.width, size.height, options) : null),
    [graph, size.width, size.height, options],
  );
  const layoutRef = useRef<ArchLayout | null>(null);
  layoutRef.current = layout;

  const active = selection;

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const current =
        layout && size.width === width && size.height === height
          ? layout
          : layoutArchitecture(graph, width, height, options);
      drawColumnHeaders(ctx, graph, current, palette);
      drawEdges(ctx, graph, current, palette, shown, selection, phase);
      drawNodes(ctx, current, palette, shown, selection, hover === null && spot !== null);
      drawFooter(ctx, graph, width, height, palette);
      // No plate for the open component; its card says everything.
      if (shown && !(selection && sameSelection(shown, selection))) {
        drawArchPlate(ctx, graph, current, shown, width, height, palette);
      }
    },
    [graph, layout, size.width, size.height, options, shown, selection, phase, hover, spot],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): ArchSelection | null => {
      const current = layoutRef.current;
      if (!current) return null;
      const node = hitTestNode(current, pos.x, pos.y);
      if (node) return { kind: 'node', id: node.id };
      const edge = hitTestEdge(graph, current, pos.x, pos.y, 8);
      if (edge) return { kind: 'edge', id: edge.id };
      return null;
    },
    [graph],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const found = pos ? locate(pos) : null;
      setHover((prev) => (sameSelection(prev, found) ? prev : found));
    },
    [locate],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      const found = locate(pos);
      // Clicking the open component again closes it.
      onSelect(found && sameSelection(found, selection) ? null : found);
    },
    [locate, onSelect, selection],
  );

  const close = useCallback(() => onSelect(null), [onSelect]);

  // Arrows walk nodes then wires; the hover halo is the cursor.
  const targets = useMemo<ArchSelection[]>(
    () => [
      ...graph.nodes.map((n) => ({ kind: 'node' as const, id: n.id })),
      ...graph.edges.filter((e) => e.symbol).map((e) => ({ kind: 'edge' as const, id: e.id })),
    ],
    [graph],
  );
  const handleKey = useDiagramKeys<ArchSelection>({
    targets,
    cursor: hover ?? selection,
    same: sameSelection,
    onCursor: setHover,
    onOpen: (target) => onSelect(target),
    onClose: () => {
      onSelect(null);
      setHover(null);
    },
  });
  const cursorText = useMemo(() => {
    const cursor = hover ?? selection;
    return cursor ? plateLines(graph, cursor).join(', ') : '';
  }, [graph, hover, selection]);

  // A card whose component has left the graph closes with it.
  useEffect(() => {
    if (!selection) return undefined;
    const exists =
      selection.kind === 'node'
        ? graph.nodes.some((n) => n.id === selection.id)
        : graph.edges.some((e) => e.id === selection.id);
    if (!exists) onSelect(null);
    return undefined;
  }, [graph, selection, onSelect]);

  const detail = active && renderDetail ? renderDetail(active, onSelect) : null;

  const hoverKey = shown ? shown.kind + ':' + shown.id : '';
  const selectionKey = selection ? selection.kind + ':' + selection.id : '';

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
      <div className="mlx-arch__stage" ref={setFrame}>
        <Chart
          draw={draw}
          height="fill"
          description={description}
          cursor={hover ? 'pointer' : 'default'}
          onPointerDown={handleDown}
          onPointerMove={handleMove}
          onPointerLeave={() => setHover(null)}
          redrawKey={String(redrawKey) + '|' + hoverKey + '|' + selectionKey + '|' + phase}
        />
      </div>
      {detail ? (
        <DetailOverlay dock onClose={close}>
          {detail}
        </DetailOverlay>
      ) : null}
    </div>
  );
}

/* ---------------- drawing ---------------- */

/** Node colour by kind. */
function kindAlpha(kind: ArchNodeKind): number {
  if (kind === 'data') return 0.04;
  return kind === 'op' || kind === 'unit' ? 0.1 : 0.16;
}

function drawColumnHeaders(
  ctx: CanvasRenderingContext2D,
  graph: ArchGraph,
  layout: ArchLayout,
  palette: Palette,
): void {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  graph.columns.forEach((column, index) => {
    const x = layout.columnX[index];
    if (x === undefined) return;
    ctx.font = '700 10px ' + FONT_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.fillText(column.title.toUpperCase(), x, layout.headerY, layout.columnWidth - 8);
    ctx.font = '10px ' + MONO_STACK;
    ctx.fillStyle = palette.textFaint;
    const maxW = layout.columnWidth - 8;
    if (column.subtitle && ctx.measureText(column.subtitle).width <= maxW) {
      ctx.fillText(column.subtitle, x, layout.headerY + 13);
    }
  });
  ctx.restore();
}

function drawEdges(
  ctx: CanvasRenderingContext2D,
  graph: ArchGraph,
  layout: ArchLayout,
  palette: Palette,
  hover: ArchSelection | null,
  selection: ArchSelection | null,
  phase: number,
): void {
  const maxWeight = graph.maxWeight || 1;

  // The open card decides which wires recede; a hover only adds its own halo.
  const focus = selection ?? hover;
  const focusNode = focus?.kind === 'node' ? focus.id : null;
  const isHot = (edge: ArchEdge) =>
    [hover, selection].some((s) => s?.kind === 'edge' && s.id === edge.id);

  for (const edge of graph.edges) {
    const path = edgePath(edge, layout);
    if (!path) continue;

    let emphasis: WireEmphasis = 'plain';
    if (isHot(edge)) emphasis = 'hot';
    else if (focusNode !== null) {
      emphasis = edge.from === focusNode || edge.to === focusNode ? 'lit' : 'dim';
    } else if (focus) emphasis = 'dim';

    const weight = edge.weight ?? 0;
    const magnitude = Math.min(1, Math.abs(weight) / maxWeight);
    // The gradient shares the violet of the parameters it updates.
    const colour = edge.active
      ? palette.violet
      : edge.weight === undefined
        ? palette.textMuted
        : weight < 0
          ? palette.negative
          : palette.positive;
    const alpha = edge.weight === undefined ? 0.6 : 0.45 + 0.5 * magnitude;
    const width = edge.weight === undefined ? 1.6 : 1.2 + 3.4 * magnitude;

    strokeWire(ctx, path, {
      colour,
      alpha,
      width,
      emphasis,
      dash: edge.active ? [6, 7] : undefined,
      dashOffset: -((phase * 3) % 13),
    });
    ctx.save();
    ctx.fillStyle = rgba(colour, emphasis === 'dim' ? alpha * DIM : Math.min(1, alpha + 0.2));
    drawArrowHead(ctx, path.end.x, path.end.y, path.angle, 5.5 + width);
    ctx.restore();
  }
}

/** A filled triangle pointing along `angle`, its tip at (x, y). */
function drawArrowHead(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  angle: number,
  size: number,
): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(1, 0);
  ctx.lineTo(-size, -size * 0.55);
  ctx.lineTo(-size * 0.7, 0);
  ctx.lineTo(-size, size * 0.55);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawNodes(
  ctx: CanvasRenderingContext2D,
  layout: ArchLayout,
  palette: Palette,
  hover: ArchSelection | null,
  selection: ArchSelection | null,
  /** The hover is a lesson's spotlight rather than the pointer. */
  spotlit = false,
): void {
  for (const node of layout.nodes) {
    const isHovered = hover?.kind === 'node' && hover.id === node.id;
    const isSelected = selection?.kind === 'node' && selection.id === node.id;
    const { rect } = node;
    const round = node.kind === 'op' || node.kind === 'unit';
    const radius = Math.max(3, rect.w * 0.06);
    const hue = palette.accent;

    ctx.save();
    if (node.muted) ctx.globalAlpha = 0.4;

    const outline = () => {
      ctx.beginPath();
      if (round) ctx.arc(node.cx, node.cy, rect.w / 2, 0, Math.PI * 2);
      else roundRect(ctx, rect.x, rect.y, rect.w, rect.h, radius);
    };

    outline();
    if (node.thumbnail) {
      ctx.save();
      ctx.clip();
      node.thumbnail(ctx, rect, palette);
      ctx.restore();
    } else {
      const tint = node.tint ?? 0;
      ctx.fillStyle = palette.surface;
      ctx.fill();
      ctx.fillStyle =
        tint !== 0
          ? rgba(tint < 0 ? palette.negative : palette.positive, 0.22 + 0.5 * Math.abs(tint))
          : rgba(hue, kindAlpha(node.kind));
      ctx.fill();
    }

    outline();
    ctx.strokeStyle = rgba(hue, 0.55);
    ctx.lineWidth = 1.2;
    if (node.kind === 'data') ctx.setLineDash([4, 3]);
    ctx.stroke();
    if (isSelected || isHovered) strokeOutline(ctx, outline, isSelected ? 'open' : spotlit ? 'spot' : 'hover', palette);
    ctx.restore();

    // The symbol: inside a circle or plain box, under a picture.
    ctx.save();
    ctx.textAlign = 'center';
    if (labelInside(node)) {
      ctx.textBaseline = 'middle';
      const size = round ? Math.max(11, Math.min(20, rect.h * 0.42)) : fontSize(rect, node.label);
      ctx.font = (round ? '500 ' : '650 ') + size + 'px ' + FONT_STACK;
      ctx.fillStyle = palette.text;
      ctx.fillText(node.label, node.cx, node.cy + (round ? 1 : 0), rect.w - 8);
    } else {
      ctx.textBaseline = 'top';
      ctx.font = '600 10px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.fillText(node.label, node.cx, rect.y + rect.h + 5, Math.max(48, rect.w + 26));
      ctx.font = '9px ' + MONO_STACK;
      ctx.fillStyle = palette.textFaint;
      const room = Math.min(Math.max(48, rect.w + 26), layout.columnWidth - 4);
      if (node.value && ctx.measureText(node.value).width <= room) {
        ctx.fillText(node.value, node.cx, rect.y + rect.h + 17);
      }
    }
    ctx.restore();
  }
}

function fontSize(rect: { w: number; h: number }, label: string): number {
  const byWidth = (rect.w - 12) / Math.max(1.6, label.length * 0.58);
  return Math.max(9, Math.min(17, byWidth, rect.h * 0.3));
}

/** The architecture in symbols, along the bottom. */
function drawFooter(
  ctx: CanvasRenderingContext2D,
  graph: ArchGraph,
  width: number,
  height: number,
  palette: Palette,
): void {
  if (!graph.summary) return;
  ctx.save();
  ctx.font = '11px ' + MONO_STACK;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillStyle = palette.textFaint;
  const fits = ctx.measureText(graph.summary).width <= width - 28;
  const lines = fits ? [graph.summary] : graph.summary.split(',   ');
  lines.forEach((line, i) => {
    ctx.fillText(line.trim(), width / 2, height - 9 - (lines.length - 1 - i) * 14);
  });
  ctx.restore();
}

/** The readout for a graph component, just under it. */
function drawArchPlate(
  ctx: CanvasRenderingContext2D,
  graph: ArchGraph,
  layout: ArchLayout,
  hover: ArchSelection,
  width: number,
  height: number,
  palette: Palette,
): void {
  const lines = plateLines(graph, hover);
  if (lines.length === 0) return;
  const size = measurePlate(ctx, lines);
  let spot: { left: number; top: number } | null = null;
  if (hover.kind === 'node') {
    const node = layout.byId.get(hover.id);
    if (!node) return;
    spot = placePlate(size, node.rect, width, height);
  } else {
    const edge = graph.edges.find((e) => e.id === hover.id);
    const path = edge ? edgePath(edge, layout) : null;
    if (!path) return;
    spot = placePlate(size, underWire(path.label), width, height);
  }
  drawHoverPlate(ctx, lines, spot.left, spot.top, palette);
}

/* ---------------- thumbnail helpers ---------------- */

/** Paint a scalar field into a node box from a row-major grid of samples in [-1, 1]. */
export function paintFieldThumbnail(
  ctx: CanvasRenderingContext2D,
  rect: { x: number; y: number; w: number; h: number },
  palette: Palette,
  values: Float32Array | number[],
  size: number,
): void {
  const cell = ctx.createImageData(size, size);
  const data = cell.data;
  const neg = toRgb(palette.negative);
  const pos = toRgb(palette.positive);
  const surface = toRgb(palette.surfaceAlt);

  for (let i = 0; i < size * size; i++) {
    const v = Math.max(-1, Math.min(1, values[i] ?? 0));
    const target = v < 0 ? neg : pos;
    const t = Math.abs(v);
    const o = i * 4;
    data[o] = Math.round(surface[0] + (target[0] - surface[0]) * t);
    data[o + 1] = Math.round(surface[1] + (target[1] - surface[1]) * t);
    data[o + 2] = Math.round(surface[2] + (target[2] - surface[2]) * t);
    data[o + 3] = 255;
  }

  const bitmap = scratchBitmap(size, size);
  if (!bitmap) return;
  bitmap.ctx.putImageData(cell, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap.canvas, rect.x, rect.y, rect.w, rect.h);
}
