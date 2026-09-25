/** Neural network charts: the diagram, the boundary, train vs test loss, the atlas, the vote. */

import { useCallback, useMemo, useRef } from 'react';
import type { ReactNode } from 'react';

import { MAX_BOX, distanceToSegment, sampleEdge } from '../../explainer/architecture';
import type { EdgePath, Rect } from '../../explainer/architecture';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailOverlay } from '../../explainer/components/Detail';
import {
  drawHoverPlate,
  measurePlate,
  placePlate,
  strokeOutline,
  strokeWire,
  underWire,
} from '../../explainer/diagramStyle';
import type { WireEmphasis } from '../../explainer/diagramStyle';
import { useLessonTarget } from '../../explainer/lessonFocus';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { PHONE_QUERY, useMediaQuery } from '../../explainer/useMediaQuery';
import { paintVectorThumbnail } from '../../explainer/vectors';
import type { VectorColumn } from '../../explainer/vectors';
import {
  FONT_STACK,
  MONO_STACK,
  chartScale,
  clamp,
  clipFrame,
  drawAxes,
  drawLabelPlate,
  extentOf,
  makeFrame,
  padExtent,
  roundRect,
  scratchBitmap,
} from '../../lib/viz/canvas';
import {
  classColour,
  drawEmptyAxes,
  drawEmptyState,
  drawScatter,
  drawSeries,
  drawSignedBars,
  drawSignedField,
  plateRoom,
} from '../../lib/viz/plots';
import { diverging, rgba, toRgb } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { POINT_RANGE } from '../../lib/datasets/points';
import { sub } from '../../lib/math/stats';
import type { Point2D } from '../../lib/datasets/types';
import {
  FEATURE_LABELS,
  evaluate,
  evaluateNeuron,
  inputVector,
  maxAbsWeight,
} from '../../lib/ml/neuralNetwork';
import type { Activation, InputFeature, Layer } from '../../lib/ml/neuralNetwork';
import { FLAT_THRESHOLD } from './grids';
import type { NeuronGrid, NeuronGrids, NeuronRef } from './grids';
import { unitTarget } from './lessons';

/* ---------------- painting a grid ---------------- */

/** Paint a unit's activation grid into a rectangle, scaled by its own range with a floor. */
function paintGrid(
  ctx: CanvasRenderingContext2D,
  grid: NeuronGrid,
  x: number,
  y: number,
  w: number,
  h: number,
  palette: Palette,
  maxAlpha = 0.95,
): void {
  const n = grid.size;
  const scratch = scratchBitmap(n, n);
  if (!scratch) return;

  const image = scratch.ctx.createImageData(n, n);
  const data = image.data;
  const neg = toRgb(palette.negative);
  const pos = toRgb(palette.positive);
  const scale = Math.max(0.3, Math.abs(grid.min), Math.abs(grid.max));

  for (let i = 0; i < n * n; i++) {
    const v = clamp(grid.values[i] / scale, -1, 1);
    const rgb = v < 0 ? neg : pos;
    const offset = i * 4;
    data[offset] = rgb[0];
    data[offset + 1] = rgb[1];
    data[offset + 2] = rgb[2];
    data[offset + 3] = Math.round(Math.abs(v) * maxAlpha * 255);
  }
  scratch.ctx.putImageData(image, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(scratch.canvas, 0, 0, n, n, x, y, w, h);
  ctx.restore();
}

/* ---------------- the network diagram ---------------- */

/** Every vector the diagram shows over the training set, one value per point. */
export interface NetworkVectors {
  x1: number[];
  x2: number[];
  /** 0 or 1. */
  label: number[];
  /** inputs[i] is feature i for every point. */
  inputs: number[][];
  /** activations[l][u] is unit u of layer l for every point; the last layer is ŷ. */
  activations: number[][][];
  /** sums[l][u] is the weighted sum z of unit u before its activation. */
  sums: number[][][];
  /** Row order shared with the detail card. */
  order: number[];
}

export type DiagramHover =
  | { kind: 'point' }
  | { kind: 'neuron'; layer: number; neuron: number }
  | { kind: 'input'; index: number }
  | { kind: 'weight'; layer: number; neuron: number; input: number }
  | null;

export function sameHover(a: DiagramHover, b: DiagramHover): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  if (a.kind === 'point') return true;
  if (a.kind === 'input') return b.kind === 'input' && a.index === b.index;
  if (a.kind === 'neuron') return b.kind === 'neuron' && a.layer === b.layer && a.neuron === b.neuron;
  return (
    b.kind === 'weight' && a.layer === b.layer && a.neuron === b.neuron && a.input === b.input
  );
}

/** The lesson target id of a component: `point`, `input:<i>`, `unit:<l>:<n>`, `weight:<l>:<n>:<i>`, `output`. */
export function encodeTarget(hover: NonNullable<DiagramHover>, hiddenCount: number): string {
  switch (hover.kind) {
    case 'point':
      return 'point';
    case 'input':
      return 'input:' + hover.index;
    case 'weight':
      return 'weight:' + hover.layer + ':' + hover.neuron + ':' + hover.input;
    default:
      return hover.layer >= hiddenCount ? 'output' : unitTarget(hover.layer, hover.neuron);
  }
}

export function decodeTarget(id: string, hiddenCount: number): DiagramHover {
  const [kind, ...rest] = id.split(':');
  const nums = rest.map(Number);
  if (nums.some((n) => !Number.isInteger(n) || n < 0)) return null;
  if (kind === 'point') return { kind: 'point' };
  if (kind === 'output') return { kind: 'neuron', layer: hiddenCount, neuron: 0 };
  if (kind === 'input' && nums.length === 1) return { kind: 'input', index: nums[0] };
  if (kind === 'unit' && nums.length === 2) return { kind: 'neuron', layer: nums[0], neuron: nums[1] };
  if (kind === 'weight' && nums.length === 3) return { kind: 'weight', layer: nums[0], neuron: nums[1], input: nums[2] };
  return null;
}

/** The symbol of hidden unit `index` in its layer: a₁, a₂, ... */
export function unitSymbol(index: number): string {
  return 'a' + sub(index + 1);
}

interface NodeBox {
  x: number;
  y: number;
  size: number;
}

interface DiagramLayout {
  /** columns[0] is the example point, columns[1] the inputs; columns[c] for c ≥ 2 is layers[c − 2]. */
  columns: NodeBox[][];
}

/** Column index of the inputs; layers start one column later. */
const INPUT_COLUMN = 1;

/** Side margin shared with `NetworkHeader`, so its cells sit over the columns. */
export const DIAGRAM_MARGIN_X = 16;

function layoutDiagram(width: number, height: number, counts: readonly number[]): DiagramLayout {
  const marginTop = 12;
  const marginBottom = 26;
  const innerWidth = Math.max(40, width - DIAGRAM_MARGIN_X * 2);
  const innerHeight = Math.max(40, height - marginTop - marginBottom);
  const colPitch = innerWidth / Math.max(1, counts.length);
  const maxRows = Math.max(1, ...counts);
  const size = Math.max(12, Math.min(MAX_BOX, colPitch * 0.5, (innerHeight / maxRows) * 0.78));
  const pitch = Math.min(size + 18, innerHeight / maxRows);
  const blockH = size + (maxRows - 1) * pitch;
  // Hang the block just under the headings.
  const centreY = marginTop + Math.min(innerHeight * 0.46, blockH / 2 + 28);

  const columns = counts.map((count, index) => {
    const cx = DIAGRAM_MARGIN_X + colPitch * (index + 0.5);
    return Array.from({ length: count }, (_, row) => ({
      x: cx,
      y: centreY + (row - (count - 1) / 2) * pitch,
      size,
    }));
  });

  return { columns };
}

/** A wire from the right edge of one box to the left edge of the next. */
function wirePath(from: NodeBox, to: NodeBox): EdgePath {
  const start = { x: from.x + from.size / 2 + 1, y: from.y };
  const end = { x: to.x - to.size / 2 - 1, y: to.y };
  const midX = (start.x + end.x) / 2;
  return {
    start,
    c1: { x: midX, y: start.y },
    c2: { x: midX, y: end.y },
    end,
    angle: 0,
    backwards: false,
    label: { x: midX, y: (start.y + end.y) / 2 },
  };
}

function wireDistance(p: { x: number; y: number }, path: EdgePath): number {
  const points = sampleEdge(path, 10);
  let best = Infinity;
  for (let i = 1; i < points.length; i++) {
    const d = distanceToSegment(
      p.x,
      p.y,
      points[i - 1].x,
      points[i - 1].y,
      points[i].x,
      points[i].y,
    );
    if (d < best) best = d;
  }
  return best;
}

export interface NetworkDiagramProps {
  layers: readonly Layer[];
  features: readonly InputFeature[];
  grids: NeuronGrids;
  /** The training set and what every input carries for it. */
  vectors: NetworkVectors;
  /** What the hover plate says for a component: what it is, then its number now. */
  plate: (target: NonNullable<DiagramHover>) => readonly string[] | null;
  hover: DiagramHover;
  onHover: (hover: DiagramHover) => void;
  /** The lesson's spotlight, drawn like a hover while nothing is hovered. */
  spot?: DiagramHover;
  /** The component whose card is open, set by clicking it. */
  open: DiagramHover;
  onOpen: (target: DiagramHover) => void;
  showThumbnails: boolean;
  epoch: number;
  /** Take the container height instead of an intrinsic one (the studio hero). */
  fill?: boolean;
  /** The column headings. */
  header?: ReactNode;
  /** The detail card for the open component. `jump` lets it open another one. */
  renderDetail?: (target: NonNullable<DiagramHover>, jump: (next: DiagramHover) => void) => ReactNode;
}

export function NetworkDiagram({
  layers,
  features,
  grids,
  vectors,
  plate,
  hover,
  onHover,
  spot = null,
  open,
  onOpen,
  showThumbnails,
  epoch,
  fill,
  header,
  renderDetail,
}: NetworkDiagramProps) {
  const layoutRef = useRef<DiagramLayout | null>(null);
  const maxRows = Math.max(1, features.length, ...layers.map((l) => l.length));

  // The card opens for the clicked component; hovering only highlights.
  const active = open;

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const counts = [1, features.length, ...layers.map((layer) => layer.length)];
      const layout = layoutDiagram(width, height, counts);
      layoutRef.current = layout;
      const lastColumn = layout.columns.length - 1;
      const maxAbs = maxAbsWeight(layers);
      const shown = hover ?? spot;
      const shownStyle = hover === null && spot !== null ? 'spot' : 'hover';
      // The open card decides which wires recede; a hover only adds its own halo.
      const focus = open ?? shown;
      const hotWire = (l: number, n: number, i: number) =>
        [shown, open].some(
          (t) => t?.kind === 'weight' && t.layer === l && t.neuron === n && t.input === i,
        );

      /* The data feeds every input --------------------------------------- */
      const pointBox = layout.columns[0][0];
      for (const inputBox of layout.columns[INPUT_COLUMN]) {
        strokeWire(ctx, wirePath(pointBox, inputBox), {
          colour: palette.textMuted,
          alpha: 0.7,
          width: 1.2,
          emphasis: focus && focus.kind !== 'point' ? 'dim' : 'plain',
          dash: [3, 4],
        });
      }

      /* Connections ----------------------------------------------------- */
      // Two passes so emphasised wires land on top.
      for (const pass of ['back', 'front'] as const) {
        for (let c = INPUT_COLUMN; c < lastColumn; c++) {
          const from = layout.columns[c];
          const to = layout.columns[c + 1];
          const l = c - INPUT_COLUMN;
          const layer = layers[l];
          for (let n = 0; n < to.length; n++) {
            const neuron = layer[n];
            for (let i = 0; i < from.length; i++) {
              const emphasis = hotWire(l, n, i) ? 'hot' : connectionEmphasis(focus, l, n, i);
              const wanted = emphasis === 'lit' || emphasis === 'hot' ? 'front' : 'back';
              if (wanted !== pass) continue;

              const w = neuron.weights[i] ?? 0;
              const t = clamp(w / maxAbs, -1, 1);
              const magnitude = Math.abs(t);
              // Keep a small weight faintly visible.
              const ramped = Math.sign(t) * (0.3 + 0.7 * magnitude);
              strokeWire(ctx, wirePath(from[i], to[n]), {
                colour: diverging(palette, ramped, 1),
                alpha: 0.9,
                width: 0.8 + magnitude * 4.2,
                emphasis,
              });
            }
          }
        }
      }

      /* The data: every training point as a row --------------------------- */
      {
        const half = pointBox.size / 2;
        const x = pointBox.x - half;
        const y = pointBox.y - half;
        const columns: VectorColumn[] = [
          { name: 'x₁', values: vectors.x1 },
          { name: 'x₂', values: vectors.x2 },
          { name: 'y', values: vectors.label, ramp: 'class' },
        ];
        const outline = () => roundRect(ctx, x, y, pointBox.size, pointBox.size, 4);
        ctx.save();
        outline();
        ctx.clip();
        paintVectorThumbnail(ctx, { x, y, w: pointBox.size, h: pointBox.size }, palette, columns, vectors.order);
        ctx.restore();
        ctx.save();
        outline();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = rgba(palette.accent, 0.55);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.setLineDash([]);
        if (open?.kind === 'point') strokeOutline(ctx, outline, 'open', palette);
        else if (shown?.kind === 'point') strokeOutline(ctx, outline, shownStyle, palette);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.font = '600 10px ' + FONT_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.fillText('X, y', pointBox.x, pointBox.y + half + 4);
        ctx.font = '9px ' + MONO_STACK;
        ctx.fillStyle = palette.textFaint;
        ctx.fillText(vectors.label.length + ' × 3', pointBox.x, pointBox.y + half + 16);
        ctx.restore();
      }

      /* Nodes ------------------------------------------------------------ */
      layout.columns.forEach((column, c) => {
        if (c === 0) return;
        column.forEach((box, i) => {
          const isInput = c === INPUT_COLUMN;
          const isOutput = c === lastColumn;
          const layerIndex = c - INPUT_COLUMN - 1;
          const half = box.size / 2;
          const x = box.x - half;
          const y = box.y - half;
          const radius = Math.max(3, box.size * 0.08);
          const grid = isInput ? null : isOutput ? grids.output : grids.hidden[layerIndex]?.[i];

          const matches = (t: DiagramHover) =>
            (t?.kind === 'input' && isInput && t.index === i) ||
            (t?.kind === 'neuron' && !isInput && t.layer === layerIndex && t.neuron === i);
          const outline = () => roundRect(ctx, x, y, box.size, box.size, radius);

          ctx.save();
          outline();
          ctx.fillStyle = palette.surface;
          ctx.fill();

          if (isInput) {
            ctx.save();
            roundRect(ctx, x, y, box.size, box.size, radius);
            ctx.clip();
            paintVectorThumbnail(
              ctx,
              { x, y, w: box.size, h: box.size },
              palette,
              [{ name: FEATURE_LABELS[features[i]], values: vectors.inputs[i] ?? [] }],
              vectors.order,
            );
            ctx.restore();
          } else if (showThumbnails && grid) {
            ctx.save();
            roundRect(ctx, x, y, box.size, box.size, radius);
            ctx.clip();
            paintGrid(ctx, grid, x, y, box.size, box.size, palette);
            ctx.restore();
          }

          outline();
          ctx.strokeStyle = rgba(palette.textFaint, 0.55);
          ctx.lineWidth = 1;
          ctx.stroke();
          if (matches(open)) strokeOutline(ctx, outline, 'open', palette);
          else if (matches(shown)) strokeOutline(ctx, outline, shownStyle, palette);
          ctx.restore();

          // Every node carries its symbol.
          ctx.save();
          ctx.font = '600 10px ' + FONT_STACK;
          ctx.fillStyle = palette.textMuted;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(
            isInput ? FEATURE_LABELS[features[i]] : isOutput ? 'ŷ' : unitSymbol(i),
            box.x,
            box.y + half + 4,
          );
          ctx.restore();
        });
      });

      if (features.length === 0) {
        ctx.save();
        ctx.font = '600 11px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText('No input features selected, every unit sees only its bias.', width / 2, height - 4);
        ctx.restore();
      }

      /* The readout, just under the component ----------------------------- */
      if (shown && !(open && sameHover(shown, open))) {
        const lines = plate(shown);
        if (lines && lines.length > 0) {
          const size = measurePlate(ctx, lines);
          let spot: { left: number; top: number } | null = null;
          if (shown.kind === 'weight') {
            const from = layout.columns[shown.layer + INPUT_COLUMN]?.[shown.input];
            const to = layout.columns[shown.layer + INPUT_COLUMN + 1]?.[shown.neuron];
            if (from && to) {
              const mid = wirePath(from, to).label;
              spot = placePlate(size, underWire(mid), width, height);
            }
          } else {
            const c =
              shown.kind === 'point' ? 0 : shown.kind === 'input' ? INPUT_COLUMN : shown.layer + INPUT_COLUMN + 1;
            const row = shown.kind === 'point' ? 0 : shown.kind === 'input' ? shown.index : shown.neuron;
            const box = layout.columns[c]?.[row];
            if (box) {
              const rect: Rect = { x: box.x - box.size / 2, y: box.y - box.size / 2, w: box.size, h: box.size };
              spot = placePlate(size, rect, width, height);
            }
          }
          if (spot) drawHoverPlate(ctx, lines, spot.left, spot.top, palette);
        }
      }
    },
    [layers, features, grids, vectors, plate, hover, spot, open, showThumbnails],
  );

  const hitTest = useCallback((pos: { x: number; y: number }): DiagramHover => {
    const layout = layoutRef.current;
    if (!layout) return null;

    for (let c = 0; c < layout.columns.length; c++) {
      const column = layout.columns[c];
      for (let i = 0; i < column.length; i++) {
        const box = column[i];
        const reach = box.size / 2 + 3;
        if (Math.abs(pos.x - box.x) <= reach && Math.abs(pos.y - box.y) <= reach) {
          if (c === 0) return { kind: 'point' };
          return c === INPUT_COLUMN
            ? { kind: 'input', index: i }
            : { kind: 'neuron', layer: c - INPUT_COLUMN - 1, neuron: i };
        }
      }
    }

    let best: DiagramHover = null;
    let bestDistance = 6;
    for (let c = INPUT_COLUMN; c < layout.columns.length - 1; c++) {
      const from = layout.columns[c];
      const to = layout.columns[c + 1];
      for (let n = 0; n < to.length; n++) {
        for (let i = 0; i < from.length; i++) {
          const distance = wireDistance(pos, wirePath(from[i], to[n]));
          if (distance < bestDistance) {
            bestDistance = distance;
            best = { kind: 'weight', layer: c - INPUT_COLUMN, neuron: n, input: i };
          }
        }
      }
    }
    return best;
  }, []);

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const next = pos ? hitTest(pos) : null;
      if (!sameHover(next, hover)) onHover(next);
    },
    [hitTest, hover, onHover],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      const hit = hitTest(pos);
      // Clicking the open component again closes it.
      onOpen(hit && sameHover(hit, open) ? null : hit);
    },
    [hitTest, onOpen, open],
  );

  // Keyboard order: the point, the inputs, then every unit and its weights.
  const targets = useMemo<NonNullable<DiagramHover>[]>(() => {
    const out: NonNullable<DiagramHover>[] = [{ kind: 'point' }];
    features.forEach((_, index) => out.push({ kind: 'input', index }));
    layers.forEach((layer, l) => {
      layer.forEach((unit, n) => {
        out.push({ kind: 'neuron', layer: l, neuron: n });
        unit.weights.forEach((_, i) => out.push({ kind: 'weight', layer: l, neuron: n, input: i }));
      });
    });
    return out;
  }, [features, layers]);
  const handleKey = useDiagramKeys<NonNullable<DiagramHover>>({
    targets,
    cursor: hover ?? open,
    same: sameHover,
    onCursor: onHover,
    onOpen: (target) => onOpen(target),
    onClose: () => {
      onOpen(null);
      onHover(null);
    },
  });
  const cursorText = useMemo(() => {
    const cursor = hover ?? open;
    return cursor ? (plate(cursor) ?? []).join(', ') : '';
  }, [hover, open, plate]);

  const hiddenWidths = layers.slice(0, -1).map((layer) => layer.length);
  const description =
    'Network diagram: ' +
    features.length +
    ' input features, ' +
    (hiddenWidths.length === 0
      ? 'no hidden layers'
      : hiddenWidths.length + ' hidden layer(s) of ' + hiddenWidths.join(' and ') + ' units') +
    ', one output. Line thickness is the size of each weight and colour is its sign, after ' +
    epoch +
    ' epochs. Each hidden node shows that unit’s own output across the input space.';

  const detail = active && renderDetail ? renderDetail(active, onOpen) : null;

  return (
    <div
      className="mlx-arch"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys move between components, Enter opens one.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onHover(null);
      }}
    >
      <span className="mlx-visually-hidden" aria-live="polite">
        {cursorText}
      </span>
      <div className="mlx-arch__stage">
        {header}
        <Chart
          draw={draw}
          height={fill ? 'fill' : () => Math.round(clamp(maxRows * 58 + 60, 130, 470))}
          description={description}
          cursor={hover ? 'pointer' : 'default'}
          onPointerMove={handleMove}
          onPointerDown={handleDown}
          onPointerLeave={() => onHover(null)}
          redrawKey={epoch}
        />
      </div>
      {detail ? (
        <DetailOverlay dock onClose={() => onOpen(null)}>
          {detail}
        </DetailOverlay>
      ) : null}
    </div>
  );
}

/** A wire's emphasis relative to the focused component. */
function connectionEmphasis(
  focus: DiagramHover,
  column: number,
  neuron: number,
  input: number,
): WireEmphasis {
  if (!focus) return 'plain';
  if (focus.kind === 'weight') {
    return focus.layer === column && focus.neuron === neuron && focus.input === input ? 'hot' : 'dim';
  }
  if (focus.kind === 'neuron') {
    // Both the wires feeding the unit and the wires leaving it.
    if (focus.layer === column && focus.neuron === neuron) return 'lit';
    if (focus.layer === column - 1 && focus.neuron === input) return 'lit';
    return 'dim';
  }
  if (focus.kind === 'point') return 'plain';
  return column === 0 && focus.index === input ? 'lit' : 'dim';
}

/* ---------------- the decision boundary ---------------- */

export interface BoundaryPlotProps {
  train: readonly Point2D[];
  test: readonly Point2D[];
  layers: readonly Layer[];
  features: readonly InputFeature[];
  activation: Activation;
  /** When set, the chart shows this one unit's output instead of the network's. */
  focus: NeuronRef | null;
  focusGrid: NeuronGrid | null;
  focusLabel: string | null;
  showTest: boolean;
  bands: boolean;
  diverged: boolean;
  epoch: number;
  /** The example the cards are traced through, ringed on the plot. */
}

export function BoundaryPlot({
  train,
  test,
  layers,
  features,
  activation,
  focus,
  focusGrid,
  focusLabel,
  showTest,
  bands,
  diverged,
  epoch,
}: BoundaryPlotProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, POINT_RANGE, POINT_RANGE, {
        left: 40,
        bottom: 32,
        right: 14,
        top: 14,
      });

      if (diverged) {
        drawAxes(ctx, frame, palette, { xLabel: 'x₁', yLabel: 'x₂', showGrid: false });
        drawEmptyState(
          ctx,
          width,
          height,
          palette.red,
          'The weights overflowed. Lower the learning rate and reset.',
        );
        return;
      }

      const scale = focusGrid
        ? Math.max(0.3, Math.abs(focusGrid.min), Math.abs(focusGrid.max))
        : 1;
      const valueAt = focus
        ? (x: number, y: number) =>
            evaluateNeuron(layers, inputVector(features, x, y), activation, focus.layer, focus.neuron) /
            scale
        : (x: number, y: number) => evaluate(layers, inputVector(features, x, y), activation);

      drawSignedField(ctx, frame, palette, valueAt, {
        cellSize: 6,
        alpha: 0.5,
        contour: true,
        contourColour: palette.text,
        contourWidth: 1.6,
        bands: bands ? 6 : undefined,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'x₁',
        yLabel: 'x₂',
        showGrid: false,
        xTicks: 5,
        yTicks: 5,
      });

      if (showTest && test.length > 0) {
        // A ring under every held-out point.
        ctx.save();
        clipFrame(ctx, frame);
        ctx.lineWidth = 1.3;
        for (const point of test) {
          ctx.beginPath();
          ctx.arc(frame.x(point.x), frame.y(point.y), 6.4, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(classColour(palette, point.label), 0.85);
          ctx.stroke();
        }
        ctx.restore();
        drawScatter(ctx, frame, palette, test, { radius: 3.2, shapes: true, opacity: 0.95 });
      }

      drawScatter(ctx, frame, palette, train, { radius: 4, shapes: true });

      if (focusLabel) {
        drawLabelPlate(ctx, focusLabel, frame.left + 8, frame.top + 12, palette, { bold: true });
      }
    },
    [train, test, layers, features, activation, focus, focusGrid, focusLabel, showTest, bands, diverged],
  );

  return (
    <Chart
      draw={draw}
      height={(w) => Math.round(clamp(w * 0.92, 240, 400))}
      description={
        (focusLabel
          ? 'Output of ' + focusLabel + ' across the input space'
          : 'Decision boundary of the whole network across the input space') +
        ' after ' +
        epoch +
        ' epochs, with ' +
        train.length +
        ' training points and ' +
        test.length +
        ' held-out points drawn ringed.'
      }
      redrawKey={epoch + ':' + (focusLabel ?? '')}
    />
  );
}

/* ---------------- train vs test loss ---------------- */

export function LossCurves({
  history,
  logScale,
  hasTest,
}: {
  history: ReadonlyArray<{ train: number; test: number }>;
  logScale: boolean;
  hasTest: boolean;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const finite = history.filter((h) => Number.isFinite(h.train));
      if (finite.length === 0) {
        drawEmptyAxes(ctx, width, height, palette, 'epoch', logScale ? 'log₁₀ loss' : 'loss');
        return;
      }

      const squash = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-6)) : v);
      const trainValues = history.map((h) => squash(h.train));
      const testValues = hasTest ? history.map((h) => squash(h.test)) : [];
      const all = trainValues.concat(testValues).filter((v) => Number.isFinite(v));
      const [lo, hi] = extentOf(all);
      const yDomain: [number, number] = logScale ? padExtent(lo, hi, 0.14) : [0, hi * 1.14 || 1];

      // Entry i is the loss after epoch i + 1.
      const frame = makeFrame(width, height, [1, Math.max(2, history.length)], yDomain, {
        left: 48,
        bottom: 30,
        right: 18,
        top: 12,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'epoch',
        yLabel: logScale ? 'log₁₀ loss' : 'loss',
        yTicks: 4,
      });

      // The band between the curves is the generalisation gap.
      if (hasTest && trainValues.length > 1) {
        ctx.save();
        clipFrame(ctx, frame);
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < trainValues.length; i++) {
          if (!Number.isFinite(trainValues[i])) continue;
          const px = frame.x(i + 1);
          const py = frame.y(trainValues[i]);
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else ctx.lineTo(px, py);
        }
        for (let i = testValues.length - 1; i >= 0; i--) {
          if (!Number.isFinite(testValues[i])) continue;
          ctx.lineTo(frame.x(i + 1), frame.y(testValues[i]));
        }
        ctx.closePath();
        ctx.fillStyle = rgba(palette.orange, palette.isDark ? 0.2 : 0.14);
        ctx.fill();
        ctx.restore();
      }

      drawSeries(
        ctx,
        frame,
        palette,
        hasTest
          ? [
              { values: trainValues, x0: 1, colour: palette.blue, width: 2 },
              { values: testValues, x0: 1, colour: palette.orange, width: 2 },
            ]
          : [{ values: trainValues, x0: 1, colour: palette.blue, width: 2, fill: true }],
      );

      // Labels placed by hand, since the curves usually end on top of each other.
      const labels: Array<{ text: string; colour: string; y: number }> = [];
      const addLabel = (text: string, colour: string, values: number[]) => {
        if (!plateRoom(frame)) return;
        for (let i = values.length - 1; i >= 0; i--) {
          if (!Number.isFinite(values[i])) continue;
          labels.push({ text, colour, y: clamp(frame.y(values[i]), frame.top + 9, frame.bottom - 9) });
          return;
        }
      };
      addLabel('train', palette.blue, trainValues);
      if (hasTest) addLabel('held out', palette.orange, testValues);
      if (labels.length === 2 && Math.abs(labels[0].y - labels[1].y) < 17) {
        const middle = (labels[0].y + labels[1].y) / 2;
        const upper = labels[0].y <= labels[1].y ? labels[0] : labels[1];
        const lower = upper === labels[0] ? labels[1] : labels[0];
        upper.y = clamp(middle - 9, frame.top + 9, frame.bottom - 9);
        lower.y = clamp(middle + 9, frame.top + 9, frame.bottom - 9);
      }
      for (const label of labels) {
        drawLabelPlate(ctx, label.text, frame.right - 4, label.y, palette, {
          align: 'right',
          colour: label.colour,
          bold: true,
          scale: chartScale(frame),
        });
      }
    },
    [history, logScale, hasTest],
  );

  return (
    <Chart
      draw={draw}
      height={(w) => Math.round(clamp(w * 0.7, 200, 320))}
      description={
        history.length === 0
          ? 'Loss curves, empty until training starts.'
          : 'Training loss' +
            (hasTest ? ' and held-out loss' : '') +
            ' over ' +
            history.length +
            ' epochs.'
      }
      redrawKey={history.length + ':' + logScale + ':' + hasTest}
    />
  );
}

/* ---------------- the atlas of hidden units ---------------- */

function atlasCell(width: number, maxCount: number): number {
  return clamp(Math.floor((width - 24) / Math.max(1, maxCount)) - 10, 26, 84);
}

export function NeuronAtlas({
  grids,
  focus,
  onFocus,
  activationLabel,
}: {
  grids: NeuronGrids;
  focus: NeuronRef | null;
  onFocus: (ref: NeuronRef | null) => void;
  activationLabel: string;
}) {
  const layoutRef = useRef<Array<{ x: number; y: number; size: number; ref: NeuronRef }>>([]);
  const rows = grids.hidden.length;
  const maxCount = Math.max(1, ...grids.hidden.map((layer) => layer.length));

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      layoutRef.current = [];
      if (rows === 0) {
        drawEmptyState(
          ctx,
          width,
          height,
          palette.textFaint,
          'No hidden layers. Add one above the diagram and every unit will appear here.',
        );
        return;
      }

      const cell = atlasCell(width, maxCount);
      let y = 10;

      grids.hidden.forEach((layer, l) => {
        ctx.save();
        ctx.font = '600 9px ' + FONT_STACK;
        ctx.fillStyle = palette.textFaint;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('HIDDEN ' + (l + 1) + ' · ' + activationLabel, 12, y + 5);
        ctx.restore();

        const top = y + 16;
        layer.forEach((grid, n) => {
          const x = 12 + n * (cell + 10);
          const isFocused = focus !== null && focus.layer === l && focus.neuron === n;
          layoutRef.current.push({ x, y: top, size: cell, ref: { layer: l, neuron: n } });

          ctx.save();
          roundRect(ctx, x, top, cell, cell, 3);
          ctx.fillStyle = palette.surface;
          ctx.fill();
          ctx.save();
          roundRect(ctx, x, top, cell, cell, 3);
          ctx.clip();
          paintGrid(ctx, grid, x, top, cell, cell, palette);
          ctx.restore();
          roundRect(ctx, x, top, cell, cell, 3);
          ctx.strokeStyle = isFocused ? palette.accent : rgba(palette.border, 1);
          ctx.lineWidth = isFocused ? 2.2 : 1;
          ctx.stroke();
          ctx.restore();

          ctx.save();
          ctx.font = '9px ' + FONT_STACK;
          ctx.fillStyle = grid.spread < FLAT_THRESHOLD ? palette.red : palette.textFaint;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(
            'unit ' + (n + 1) + (grid.spread < FLAT_THRESHOLD ? ' · flat' : ''),
            x + cell / 2,
            top + cell + 3,
          );
          ctx.restore();
        });

        y = top + cell + 18;
      });
    },
    [grids, rows, maxCount, focus, activationLabel],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      if (!pos) {
        onFocus(null);
        return;
      }
      const hit = layoutRef.current.find(
        (cell) =>
          pos.x >= cell.x && pos.x <= cell.x + cell.size && pos.y >= cell.y && pos.y <= cell.y + cell.size,
      );
      const next = hit ? hit.ref : null;
      const same =
        (next === null && focus === null) ||
        (next !== null && focus !== null && next.layer === focus.layer && next.neuron === focus.neuron);
      if (!same) onFocus(next);
    },
    [focus, onFocus],
  );

  const total = grids.hidden.reduce((sum, layer) => sum + layer.length, 0);

  return (
    <Chart
      draw={draw}
      height={(w) => (rows === 0 ? 120 : 10 + rows * (atlasCell(w, maxCount) + 34))}
      description={
        rows === 0
          ? 'No hidden units to show.'
          : 'Small multiples: the output of each of the ' +
            total +
            ' hidden units across the input space, arranged one row per layer.'
      }
      cursor="pointer"
      onPointerMove={handleMove}
      onPointerLeave={() => onFocus(null)}
      redrawKey={total + ':' + (focus ? focus.layer + '-' + focus.neuron : 'none')}
    />
  );
}

/* ---------------- the output layer's vote ---------------- */

export function OutputWeightBars({
  layers,
  features,
  learningRate,
  batchSize,
}: {
  layers: readonly Layer[];
  features: readonly InputFeature[];
  learningRate: number;
  batchSize: number;
}) {
  const output = layers[layers.length - 1]?.[0];

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const unit = layers[layers.length - 1]?.[0];
      if (!unit) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Nothing to show yet.');
        return;
      }
      // With no hidden layer the output votes directly on the input features.
      const sourceLabels =
        layers.length <= 1
          ? features.map((f) => FEATURE_LABELS[f])
          : layers[layers.length - 2].map((_, i) => 'unit ' + (i + 1));

      const bars = unit.weights.map((w, i) => ({ label: sourceLabels[i] ?? 'in ' + (i + 1), value: w }));
      bars.push({ label: 'bias', value: unit.bias });

      const scale = learningRate / Math.max(1, batchSize);
      const updates = unit.weightGradients.map((g, i) => ({
        label: sourceLabels[i] ?? 'in ' + (i + 1),
        value: -scale * g,
      }));
      updates.push({ label: 'bias', value: -scale * unit.biasGradient });

      // Stack the two blocks when every row keeps 14px; otherwise sit them side by side.
      const stacked = height >= 56 + bars.length * 2 * 14;
      const blockH = stacked ? height / 2 - 28 : height - 28;
      const blockW = stacked ? width - 16 : width / 2 - 12;
      const secondX = stacked ? 8 : width / 2 + 4;
      const secondY = stacked ? height / 2 + 20 : 20;
      const labelWidth = stacked ? 54 : 44;
      ctx.save();
      ctx.font = '600 10px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(stacked ? 'THE VOTE, output weights' : 'THE VOTE', 8, 12);
      ctx.fillText(
        stacked ? 'THE LAST UPDATE, −η·g / batch' : 'LAST UPDATE',
        secondX,
        secondY - 8,
      );
      ctx.restore();

      drawSignedBars(ctx, palette, bars, { x: 8, y: 20, width: blockW, height: blockH }, {
        labelWidth,
        valueFormat: (v) => v.toFixed(2),
      });
      drawSignedBars(
        ctx,
        palette,
        updates,
        { x: secondX, y: secondY, width: blockW, height: blockH },
        {
          labelWidth,
          valueFormat: (v) => (Math.abs(v) < 0.001 ? v.toExponential(1) : v.toFixed(4)),
        },
      );
    },
    [layers, features, learningRate, batchSize],
  );

  const count = (output?.weights.length ?? 0) + 1;
  return (
    <Chart
      draw={draw}
      height={Math.max(190, Math.min(330, 70 + count * 32))}
      description={
        'Bar chart of the ' +
        count +
        ' weights feeding the output unit and the update each of them last received.'
      }
      redrawKey={(output?.weights.join(',') ?? '') + ':' + (output?.bias ?? 0)}
    />
  );
}

/* ---------------- the column headings and the architecture editor ---------------- */

/** One heading cell per diagram column. Layers and widths are edited here, above the nodes. */
export function NetworkHeader({
  featureCount,
  pointCount,
  layers,
  onChange,
  maxLayers,
  maxUnits,
}: {
  featureCount: number;
  /** How many examples the data column can step through. */
  pointCount: number;
  layers: readonly number[];
  onChange: (next: number[]) => void;
  maxLayers: number;
  maxUnits: number;
}) {
  const columns = layers.length + 3;
  // Weight matrix shapes: each layer takes the width of the one before it.
  const widths = [featureCount, ...layers, 1];
  // On a phone the unit counts join the layers row.
  const phone = useMediaQuery(PHONE_QUERY);
  const target = useLessonTarget('custom', 'layers');
  const setCount = (index: number, value: number) => {
    const next = layers.slice();
    next[index] = clamp(Math.round(value), 1, maxUnits);
    onChange(next);
  };
  const unitStepper = (count: number, index: number, label: string) => (
    <Stepper
      key={index}
      value={count}
      label={label}
      canDecrease={count > 1}
      canIncrease={count < maxUnits}
      onDecrease={() => setCount(index, count - 1)}
      onIncrease={() => setCount(index, count + 1)}
      decreaseLabel={'Remove a unit from hidden layer ' + (index + 1)}
      increaseLabel={'Add a unit to hidden layer ' + (index + 1)}
    />
  );

  return (
    <div
      className="mlx-net-head"
      style={{
        ['--mlx-net-columns' as string]: String(columns),
        ['--mlx-net-margin' as string]: DIAGRAM_MARGIN_X + 'px',
      }}
    >
      <div
        className="mlx-net-head__layers"
        style={{ gridColumn: phone ? '1 / -1' : layers.length > 0 ? '3 / -2' : '2 / -1' }}
        {...target.attrs}
      >
        <Stepper
          value={layers.length}
          label={
            layers.length === 0
              ? 'no hidden layer'
              : layers.length + (layers.length === 1 ? ' hidden layer' : ' hidden layers')
          }
          canDecrease={layers.length > 0}
          canIncrease={layers.length < maxLayers}
          onDecrease={() => onChange(layers.slice(0, -1))}
          onIncrease={() =>
            onChange([...layers, layers.length > 0 ? layers[layers.length - 1] : 4])
          }
          decreaseLabel="Remove the last hidden layer"
          increaseLabel="Add a hidden layer"
        />
        {phone
          ? layers.map((count, index) =>
              unitStepper(count, index, 'H' + (index + 1) + ': ' + count + (count === 1 ? ' unit' : ' units')),
            )
          : null}
      </div>

      <div className="mlx-net-head__col">
        <span className="mlx-net-head__title">Data</span>
        <span className="mlx-net-head__sub">{pointCount} points</span>
      </div>
      <div className="mlx-net-head__col">
        <span className="mlx-net-head__title">Inputs</span>
        <span className="mlx-net-head__sub">
          {featureCount === 0 ? 'none' : featureCount + (featureCount === 1 ? ' feature' : ' features')}
        </span>
      </div>
      {layers.map((count, index) => (
        <div className="mlx-net-head__col" key={index}>
          <span className="mlx-net-head__title">Hidden {index + 1}</span>
          {phone ? (
            <span className="mlx-net-head__sub">{count} units</span>
          ) : (
            unitStepper(count, index, count + (count === 1 ? ' unit' : ' units'))
          )}
          <span className="mlx-net-head__sub">
            W {widths[index]}×{count} · b {count}
          </span>
        </div>
      ))}
      <div className="mlx-net-head__col">
        <span className="mlx-net-head__title">Output</span>
        <span className="mlx-net-head__sub">
          ŷ · W {widths[widths.length - 2]}×1 · b 1
        </span>
      </div>
    </div>
  );
}

function Stepper({
  value,
  label,
  canDecrease,
  canIncrease,
  onDecrease,
  onIncrease,
  decreaseLabel,
  increaseLabel,
}: {
  value: number;
  label: string;
  canDecrease: boolean;
  canIncrease: boolean;
  onDecrease: () => void;
  onIncrease: () => void;
  decreaseLabel: string;
  increaseLabel: string;
}) {
  return (
    <span className="mlx-stepper" data-value={value}>
      <button
        type="button"
        className="mlx-stepper__button"
        disabled={!canDecrease}
        onClick={onDecrease}
        aria-label={decreaseLabel}
        title={decreaseLabel}
      >
        −
      </button>
      <span className="mlx-stepper__label">{label}</span>
      <button
        type="button"
        className="mlx-stepper__button"
        disabled={!canIncrease}
        onClick={onIncrease}
        aria-label={increaseLabel}
        title={increaseLabel}
      >
        +
      </button>
    </span>
  );
}

/* ---------------- shared ---------------- */
