/** The whole network in one small picture: one column per layer, the wires in the sign and weight of what they carry, lit where the view is looking. */

import { useCallback, useRef } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DIM, strokeOutline } from '../../explainer/diagramStyle';
import { useLessonTarget } from '../../explainer/lessonFocus';
import { sub } from '../../lib/math/stats';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { rgba } from '../../lib/viz/palette';
import { layerColour } from './panels';
import { sameTarget, targetKey, unitLayer } from './targets';
import type { DiagramTarget, Spot } from './targets';

export interface NetworkMapProps {
  state: TrainerState;
  /** Short names over the hidden columns and the output. */
  activationLabel: string;
  lossLabel: string;
  /** The hidden layer the picture follows, marked under its column. */
  shown?: number | null;
  hover: DiagramTarget | null;
  open: DiagramTarget | null;
  spot: Spot | null;
  onHover: (target: DiagramTarget | null) => void;
  onOpen: (target: DiagramTarget | null) => void;
}

const HEIGHT = 156;
const PAD_X = 22;
const TOP = 26;
const LABEL_H = 20;

interface Column {
  x: number;
  ys: number[];
  /** Dot radius, the reach of a click on one unit. */
  radius: number;
  /** The layer index a click opens: 0 is the input plane. */
  layer: number;
  label: string;
  colour: string;
}

/** What the pointer is on, by column. */
function targetOf(column: Column, outputIndex: number): DiagramTarget {
  return column.layer === outputIndex ? { kind: 'function', which: 'loss' } : { kind: 'layer', index: column.layer };
}

export function NetworkMap({ state, activationLabel, lossLabel, shown = null, hover, open, spot, onHover, onOpen }: NetworkMapProps) {
  const columnsRef = useRef<Column[]>([]);
  const sizes = state.spec.sizes;
  const outputIndex = sizes.length - 1;
  const focus = open ?? hover;
  const target = useLessonTarget('custom', 'network');

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const cols = sizes.length;
      const gap = cols > 1 ? (width - PAD_X * 2) / (cols - 1) : 0;
      const bodyH = height - TOP - LABEL_H;
      const radius = Math.max(3, Math.min(6, (bodyH / Math.max(...sizes)) * 0.34));
      const columns: Column[] = sizes.map((n, k) => {
        const x = PAD_X + gap * k;
        const step = Math.min(18, bodyH / Math.max(1, n));
        const y0 = TOP + (bodyH - step * (n - 1)) / 2;
        return {
          x,
          ys: Array.from({ length: n }, (_, i) => y0 + step * i),
          radius,
          layer: k,
          label: k === 0 ? 'x' : k === outputIndex ? 'ŷ' : 'h' + sub(k),
          colour: k === 0 ? palette.textMuted : k === outputIndex ? palette.text : layerColour(palette, k - 1),
        };
      });
      columnsRef.current = columns;

      const litLayer = focus && focus.kind === 'layer' ? focus.index : -1;
      const litUnit = focus && focus.kind === 'unit' ? focus.index : -1;
      const litUnitLayer = focus && focus.kind === 'unit' ? unitLayer(focus) : -1;
      const litFunction = focus && focus.kind === 'function' ? focus.which : focus && focus.kind === 'output' ? 'loss' : null;
      const spotLayer = spot && spot.target.kind === 'layer' ? spot.target.index : -1;
      const spotUnit = spot && spot.target.kind === 'unit' ? spot.target.index : -1;
      const spotUnitLayer = spot && spot.target.kind === 'unit' ? unitLayer(spot.target) : -1;
      const spotFunction = spot && spot.target.kind === 'function' ? spot.target.which : spot && spot.target.kind === 'output' ? 'loss' : null;
      const litColumn = (k: number) => k === litLayer || (litFunction === 'activation' && k > 0 && k < outputIndex) || (litFunction === 'loss' && k === outputIndex);
      const anyLit = focus !== null;

      // Wires in the sign of their weight, as strong as the weight; the lit column's wires stay full while the rest dim.
      let maxW = 1e-6;
      for (const layer of state.net.layers) for (const w of layer.W) maxW = Math.max(maxW, Math.abs(w));
      ctx.lineCap = 'round';
      for (let k = 1; k < columns.length; k++) {
        const a = columns[k - 1];
        const b = columns[k];
        const layer = state.net.layers[k - 1];
        const columnTouched = litColumn(k) || litColumn(k - 1);
        for (let o = 0; o < b.ys.length; o++) {
          // A lit unit keeps the wires it reads; the rest dim.
          const touched = columnTouched || (k === litUnitLayer && o === litUnit);
          for (let i = 0; i < a.ys.length; i++) {
            const w = layer.W[o * layer.inSize + i];
            const t = Math.min(1, Math.abs(w) / maxW);
            const alpha = (0.1 + 0.55 * t) * (anyLit && !touched ? DIM : 1);
            ctx.strokeStyle = rgba(w < 0 ? palette.negative : palette.positive, alpha);
            ctx.lineWidth = 0.8 + 1.4 * t;
            ctx.beginPath();
            ctx.moveTo(a.x, a.ys[i]);
            ctx.lineTo(b.x, b.ys[o]);
            ctx.stroke();
          }
        }
      }

      columns.forEach((column, k) => {
        const lit = litColumn(k);
        const spotted = !lit && (k === spotLayer || (spotFunction === 'activation' && k > 0 && k < outputIndex) || (spotFunction === 'loss' && k === outputIndex));
        column.ys.forEach((y, i) => {
          const unitLit = (k === litUnitLayer && i === litUnit) || (litUnit < 0 && k === spotUnitLayer && i === spotUnit);
          ctx.beginPath();
          ctx.arc(column.x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = column.colour;
          ctx.fill();
          ctx.strokeStyle = palette.surface;
          ctx.lineWidth = 1.2;
          ctx.stroke();
          if (unitLit) {
            ctx.beginPath();
            ctx.arc(column.x, y, radius + 3.5, 0, Math.PI * 2);
            ctx.strokeStyle = palette.accent;
            ctx.lineWidth = 1.6;
            ctx.stroke();
          }
        });
        if (lit || spotted) {
          const top = column.ys[0] - radius - 5;
          const bottom = column.ys[column.ys.length - 1] + radius + 5;
          strokeOutline(
            ctx,
            () => roundRect(ctx, column.x - radius - 6, top, (radius + 6) * 2, bottom - top, 5),
            lit ? (open && sameTarget(open, targetOf(column, outputIndex)) ? 'open' : 'hover') : 'spot',
            palette,
          );
        }
        const marked = shown !== null && k === shown;
        ctx.font = (lit || marked ? '600 ' : '') + '11px ' + MONO_STACK;
        ctx.fillStyle = lit ? palette.text : marked ? palette.accent : palette.textMuted;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(column.label, column.x, height - 6);
      });

      // Over the columns: the activation across the hidden layers, the loss over the output.
      ctx.font = '10px ' + MONO_STACK;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      if (outputIndex > 1) {
        const first = columns[1];
        const last = columns[outputIndex - 1];
        ctx.fillStyle = litFunction === 'activation' ? palette.text : palette.textMuted;
        ctx.fillText(activationLabel, (first.x + last.x) / 2, 13);
        ctx.strokeStyle = rgba(palette.textFaint, 0.6);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(first.x - radius, 18);
        ctx.lineTo(last.x + radius, 18);
        ctx.stroke();
      }
      ctx.fillStyle = litFunction === 'loss' ? palette.text : palette.textMuted;
      ctx.textAlign = outputIndex > 1 ? 'center' : 'right';
      ctx.fillText(lossLabel, outputIndex > 1 ? columns[outputIndex].x : width - 4, 13);
    },
    [sizes, outputIndex, state.net.layers, activationLabel, lossLabel, shown, focus, open, spot],
  );

  // A hidden unit's dot opens that unit; anywhere else in a column opens the layer.
  const locate = useCallback(
    (pos: { x: number; y: number }): DiagramTarget | null => {
      let best: Column | null = null;
      let bestD = 16;
      for (const column of columnsRef.current) {
        const d = Math.abs(column.x - pos.x);
        if (d < bestD) {
          bestD = d;
          best = column;
        }
      }
      if (!best) return null;
      if (best.layer > 0 && best.layer < outputIndex) {
        const reach = best.radius + 4;
        const i = best.ys.findIndex((y) => Math.abs(y - pos.y) <= reach);
        if (i >= 0) return { kind: 'unit', index: i, layer: best.layer };
      }
      return targetOf(best, outputIndex);
    },
    [outputIndex],
  );

  const shape = sizes.join(' → ');
  return (
    <div className="mlx-inspector__map" {...target.attrs}>
      <Chart
        draw={draw}
        height={HEIGHT}
        description={'The network as columns of units, ' + shape + ', the wires coloured by the sign and size of their weights. Hover a column or a unit for its numbers, click to open it.'}
        cursor="pointer"
        onPointerDown={(pos) => {
          const found = locate(pos);
          onOpen(found && sameTarget(found, open) ? null : found);
        }}
        onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
        onPointerLeave={() => onHover(null)}
        redrawKey={shape + '|' + state.epoch + '|' + activationLabel + '|' + lossLabel + '|' + shown + '|' + targetKey(hover) + '|' + targetKey(open) + '|' + (spot ? targetKey(spot.target) : '')}
      />
    </div>
  );
}
