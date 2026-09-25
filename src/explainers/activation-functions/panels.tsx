/** Activations and losses charts: one function at a time with the live units on it, the fit, the per-layer bars. */

import { useCallback, useMemo, useRef } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import type { Point2D, RegressionData } from '../../lib/datasets/types';
import { POINT_RANGE } from '../../lib/datasets/points';
import { clamp, drawAxes, extentOf, makeFrame, padExtent, MONO_STACK } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { categorical, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawCategoryField, drawEmptyAxes, drawEmptyState, drawFunction, drawScatter, drawSeries, drawSignedField } from '../../lib/viz/plots';
import { fmt } from '../../lib/math/stats';
import type { ActivationFn } from '../../lib/ml/activations';
import type { LossFn } from '../../lib/ml/losses';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import type { Spot } from './targets';

export type FunctionTarget = 'activation' | 'loss';

/** One colour per hidden layer, the same on every chart. */
export function layerColour(palette: Palette, index: number): string {
  return categorical(palette, index);
}

/* ---------------- one function ---------------- */

export interface FunctionsDiagramProps {
  /** Which of the two functions is on show. */
  which: FunctionTarget;
  act: ActivationFn;
  lossFn: LossFn;
  state: TrainerState;
  /** Where each hidden layer's units sit, one list of z per layer. */
  zByLayer: readonly (readonly number[])[];
  /** Where the outputs sit on the loss axis, with the class they belong to. */
  outputs: readonly { x: number; target: number }[];
  showDerivative: boolean;
  /** Every other activation, drawn faintly behind the chosen one. */
  others: readonly ActivationFn[];
  showRug: boolean;
  hover: FunctionTarget | null;
  open: FunctionTarget | null;
  spot: Spot | null;
  /** The chart under the pointer and where along its axis, or null. */
  onHover: (target: FunctionTarget | null, x?: number) => void;
  onOpen: (target: FunctionTarget | null) => void;
  description: string;
  redrawKey: string;
}

export function FunctionsDiagram(props: FunctionsDiagramProps) {
  const { which, hover, open, onHover, onOpen, description } = props;
  const targets = useMemo(() => [which], [which]);
  const handleKey = useDiagramKeys<FunctionTarget>({
    targets,
    cursor: hover ?? open,
    same: (a, b) => a === b,
    onCursor: onHover,
    onOpen: (target) => onOpen(target),
    onClose: () => {
      onOpen(null);
      onHover(null);
    },
  });
  return (
    <div
      className="mlx-arch__stage mlx-functions"
      tabIndex={0}
      role="group"
      aria-label={description + ' Enter opens the card.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onHover(null);
      }}
    >
      <FunctionChart {...props} />
    </div>
  );
}

function FunctionChart({
  which,
  act,
  lossFn,
  zByLayer,
  outputs,
  showDerivative,
  others,
  showRug,
  hover,
  open,
  spot,
  onHover,
  onOpen,
  redrawKey,
}: FunctionsDiagramProps) {
  const frameRef = useRef<Frame | null>(null);
  const lit = (open ?? hover) === which;
  const dimmed = (open ?? hover) !== null && !lit;
  const spotted = spot !== null && spot.target.kind === 'function' && spot.target.which === which;
  const pointer = useRef<{ x: number; y: number } | null>(null);

  const curve = lossFn.curve;
  const domain = useMemo<[number, number]>(() => (which === 'activation' ? [-6, 6] : curve.xDomain), [which, curve.xDomain]);
  const fns = useMemo(() => {
    if (which === 'activation') return [{ f: act.f, df: act.df, label: act.label, dash: undefined as number[] | undefined }];
    const t = curve.targets[0] ?? 0;
    return [{ f: (x: number) => curve.f(x, t), df: (x: number) => curve.df(x, t), label: lossFn.label, dash: undefined as number[] | undefined }];
  }, [which, act, curve, lossFn.label]);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const samples: number[] = [];
      for (const fn of fns) {
        for (let i = 0; i <= 80; i++) {
          const x = domain[0] + ((domain[1] - domain[0]) * i) / 80;
          samples.push(fn.f(x));
          if (showDerivative) samples.push(fn.df(x));
        }
      }
      const [lo, hi] = extentOf(samples);
      const yDomain = padExtent(Math.min(lo, -0.2), Math.min(Math.max(hi, 1), which === 'activation' ? 6 : 8), 0.1);
      const frame = makeFrame(width, height, domain, yDomain, { left: 42, bottom: 26, right: 12, top: 12 });
      frameRef.current = frame;
      ctx.save();
      if (dimmed) ctx.globalAlpha = 0.45;
      drawAxes(ctx, frame, palette, {
        xLabel: which === 'activation' ? 'z' : curve.xLabel,
        yLabel: which === 'activation' ? 'f(z)' : 'L',
        xTicks: 6,
        yTicks: 4,
        showZeroLines: true,
      });

      // Flat zones: where the slope is too small to learn from.
      if (which === 'activation') {
        ctx.fillStyle = rgba(palette.red, 0.08);
        let start: number | null = null;
        const steps = 120;
        for (let i = 0; i <= steps; i++) {
          const x = domain[0] + ((domain[1] - domain[0]) * i) / steps;
          const flat = Math.abs(act.df(x)) < 0.05;
          if (flat && start === null) start = x;
          if ((!flat || i === steps) && start !== null) {
            ctx.fillRect(frame.x(start), frame.top, frame.x(x) - frame.x(start), frame.innerHeight);
            start = null;
          }
        }
      }

      // The live units or outputs, as a rug along the floor.
      if (showRug) {
        if (which === 'activation') {
          zByLayer.forEach((zs, l) => {
            ctx.strokeStyle = rgba(layerColour(palette, l), 0.5);
            ctx.lineWidth = 1;
            ctx.beginPath();
            const y0 = frame.bottom - 2 - l * 3;
            for (const z of zs) {
              const x = frame.x(clamp(z, domain[0], domain[1]));
              ctx.moveTo(x, y0);
              ctx.lineTo(x, y0 - 6);
            }
            ctx.stroke();
          });
        } else {
          for (const o of outputs) {
            ctx.strokeStyle = rgba(categorical(palette, o.target), 0.55);
            ctx.lineWidth = 1;
            const x = frame.x(clamp(o.x, domain[0], domain[1]));
            ctx.beginPath();
            ctx.moveTo(x, frame.bottom - 2);
            ctx.lineTo(x, frame.bottom - 9);
            ctx.stroke();
          }
        }
      }

      // The other activations, faint, each labelled once along its curve.
      if (which === 'activation') {
        ctx.font = '9px ' + MONO_STACK;
        ctx.textAlign = 'left';
        others.forEach((other, i) => {
          const colour = categorical(palette, i + 1);
          drawFunction(ctx, frame, other.f, rgba(colour, 0.5), { width: 1.1, samples: 200 });
          const z = -5 + (i * 10) / Math.max(1, others.length);
          const y = clamp(other.f(z), yDomain[0], yDomain[1]);
          ctx.fillStyle = rgba(colour, 0.9);
          ctx.fillText(other.label.toLowerCase(), frame.x(z) + 3, frame.y(y) - 4);
        });
      }

      fns.forEach((fn) => {
        const colour = palette.blue;
        if (showDerivative) drawFunction(ctx, frame, fn.df, rgba(palette.orange, 0.9), { width: 1.4, dash: fn.dash ?? [2, 3], samples: 200 });
        drawFunction(ctx, frame, fn.f, colour, { width: lit ? 2.8 : 2.2, dash: fn.dash, samples: 200 });
      });

      ctx.font = '10px ' + MONO_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'left';
      ctx.fillText(
        which === 'activation'
          ? act.label + (showDerivative ? '  ·  f′ dotted' : '')
          : lossFn.label + (curve.targets.length > 1 ? ' for y = ' + curve.targets[0] : '') + (showDerivative ? '  ·  dL dotted' : ''),
        frame.left + 4,
        frame.top + 10,
      );
      ctx.restore();

      const outline = () => {
        ctx.beginPath();
        ctx.rect(frame.left - 1, frame.top - 1, frame.innerWidth + 2, frame.innerHeight + 2);
      };
      if (lit) {
        strokeOutline(ctx, outline, open === which ? 'open' : 'hover', palette);
        // The dot on the curve under the pointer; its numbers show in the readout.
        const at = pointer.current;
        if (at && at.x >= frame.left && at.x <= frame.right) {
          const x = frame.x.invert(at.x);
          const fn = fns[0];
          ctx.beginPath();
          ctx.arc(frame.x(x), frame.y(clamp(fn.f(x), yDomain[0], yDomain[1])), 4, 0, Math.PI * 2);
          ctx.fillStyle = palette.accent;
          ctx.fill();
        }
      } else if (spotted) {
        strokeOutline(ctx, outline, 'spot', palette);
        const size = measurePlate(ctx, [spot!.label]);
        const place = placePlate(size, { x: frame.left, y: frame.top - 4, w: frame.innerWidth, h: 0 }, width, height);
        drawHoverPlate(ctx, [spot!.label], place.left, place.top, palette);
      }
    },
    [which, act, lossFn, curve, fns, domain, zByLayer, outputs, showDerivative, others, showRug, lit, dimmed, open, spotted, spot],
  );

  return (
    <Chart
      draw={draw}
      height="fill"
      description={
        which === 'activation'
          ? act.label + ' and its derivative over z, with a mark for every hidden unit of the network.'
          : lossFn.label + ' against the ' + curve.xLabel + ', with a mark for every output of the last batch.'
      }
      cursor="pointer"
      onPointerDown={() => onOpen(open === which ? null : which)}
      onPointerMove={(pos) => {
        pointer.current = pos;
        const frame = frameRef.current;
        const x = pos && frame && pos.x >= frame.left && pos.x <= frame.right ? frame.x.invert(pos.x) : undefined;
        onHover(pos ? which : null, x);
      }}
      onPointerLeave={() => {
        pointer.current = null;
        onHover(null);
      }}
      redrawKey={redrawKey + '|' + which + '|' + (hover ?? '') + '|' + (open ?? '') + '|' + (spotted ? spot!.label : '') + '|' + (pointer.current ? Math.round(pointer.current.x) : '')}
    />
  );
}

/* ---------------- the fit ---------------- */

export function ClassFit({
  points,
  score,
  classAt,
  classCount,
  epoch,
}: {
  points: readonly Point2D[];
  score: (x: number, y: number) => number;
  classAt: (x: number, y: number) => number;
  classCount: number;
  epoch: number;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, POINT_RANGE, POINT_RANGE, { left: 34, bottom: 26, right: 12, top: 10, equal: true });
      if (classCount > 2) drawCategoryField(ctx, frame, palette, classAt, { cellSize: 7, alpha: 0.32 });
      else drawSignedField(ctx, frame, palette, score, { cellSize: 7, alpha: 0.45 });
      drawAxes(ctx, frame, palette, { xLabel: 'x₁', yLabel: 'x₂', showGrid: false, xTicks: 4, yTicks: 4 });
      drawScatter(ctx, frame, palette, points, { shapes: true, radius: 3.5 });
    },
    [points, score, classAt, classCount],
  );
  return (
    <Chart
      draw={draw}
      height={230}
      description={'The class the network gives every point of the plane after ' + epoch + ' epochs, with the ' + points.length + ' points.'}
      redrawKey={epoch + ':' + points.length + ':' + classCount}
    />
  );
}

export function CurveFit({ data, predict, epoch }: { data: RegressionData; predict: (x: number) => number; epoch: number }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, data.xRange, data.yRange, { left: 46, bottom: 30, right: 14, top: 12 });
      drawAxes(ctx, frame, palette, { xLabel: data.xLabel, yLabel: data.yLabel, xTicks: 5, yTicks: 4 });
      if (data.truth) drawFunction(ctx, frame, data.truth, palette.muted, { width: 1.6, dash: [5, 4] });
      drawScatter(ctx, frame, palette, data.points, { radius: 3.5, colourOf: () => palette.blue });
      drawFunction(ctx, frame, predict, palette.orange, { width: 2.6 });
    },
    [data, predict],
  );
  return (
    <Chart
      draw={draw}
      height={230}
      description={'The curve the network draws through ' + data.points.length + ' points after ' + epoch + ' epochs.'}
      redrawKey={epoch + ':' + data.points.length}
    />
  );
}

/* ---------------- per-layer bars ---------------- */

export function LayerBars({ state, logScale }: { state: TrainerState; logScale: boolean }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const layers = state.layers;
      if (state.epoch === 0 || layers.length === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Press play to see what each layer receives.');
        return;
      }
      const values = layers.map((l) => l.gradNorm);
      const labels = layers.map((_, l) => (l === layers.length - 1 ? 'out' : 'layer ' + (l + 1)));
      const useLog = logScale;
      const shown = values.map((v) => (useLog ? Math.log10(Math.max(v, 1e-9)) : v));
      const [lo, hi] = extentOf(shown);
      const floor = useLog ? Math.min(lo, -6) - 0.5 : 0;
      const top = useLog ? Math.max(hi, 0) + 0.5 : Math.max(hi, 1e-6) * 1.15;
      const labelWidth = 52;
      const valueWidth = 62;
      const left = 8 + labelWidth;
      const trackWidth = Math.max(40, width - left - valueWidth - 8);
      const rowHeight = height / layers.length;
      const barHeight = Math.min(16, rowHeight * 0.6);
      ctx.save();
      ctx.textBaseline = 'middle';
      layers.forEach((_, l) => {
        const cy = rowHeight * (l + 0.5);
        const t = (shown[l] - floor) / (top - floor || 1);
        ctx.font = '11px ' + MONO_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.textAlign = 'right';
        ctx.fillText(labels[l], left - 8, cy);
        ctx.fillStyle = rgba(layerColour(palette, l), 0.9);
        ctx.fillRect(left, cy - barHeight / 2, Math.max(1, t * trackWidth), barHeight);
        ctx.fillStyle = palette.text;
        ctx.textAlign = 'left';
        ctx.fillText(values[l] < 0.001 && values[l] > 0 ? values[l].toExponential(1) : fmt(values[l], 3), left + trackWidth + 8, cy);
      });
      ctx.restore();
    },
    [state, logScale],
  );
  return (
    <Chart
      draw={draw}
      height={Math.max(160, 26 * state.layers.length + 16)}
      description="Gradient norm on the weights of each layer in the last batch, first layer at the top."
      redrawKey={state.epoch + ':' + logScale}
    />
  );
}

/* ---------------- loss curves ---------------- */

export function LossCurves({ history, logScale }: { history: readonly { train: number; test: number }[]; logScale: boolean }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (history.length < 2) {
        drawEmptyAxes(ctx, width, height, palette, 'epoch', logScale ? 'log₁₀ loss' : 'loss');
        return;
      }
      const tr = history.map((h) => (logScale ? Math.log10(Math.max(h.train, 1e-9)) : h.train));
      const te = history.map((h) => (logScale ? Math.log10(Math.max(h.test, 1e-9)) : h.test));
      const [lo, hi] = extentOf(tr.concat(te).filter(Number.isFinite));
      const frame = makeFrame(width, height, [0, history.length - 1], logScale ? padExtent(lo, hi, 0.12) : [0, hi * 1.1 || 1], { left: 54, bottom: 30, right: 14, top: 12 });
      drawAxes(ctx, frame, palette, { xLabel: 'epoch', yLabel: logScale ? 'log₁₀ loss' : 'loss', yTicks: 4 });
      drawSeries(
        ctx,
        frame,
        palette,
        [
          { values: tr, colour: palette.blue, width: 2, label: 'train' },
          { values: te, colour: palette.orange, width: 2, label: 'held out' },
        ],
        { labelLast: true },
      );
    },
    [history, logScale],
  );
  return (
    <Chart
      draw={draw}
      height={200}
      description={history.length < 2 ? 'Loss curves, empty until training starts.' : 'Training and test loss over ' + (history.length - 1) + ' epochs.'}
      redrawKey={history.length + ':' + logScale}
    />
  );
}
