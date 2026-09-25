/** Linear regression charts: the fit, the loss curve, the loss surface. */

import { useCallback } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { clipFrame, drawAxes, extentOf, makeFrame, padExtent, drawPoint, FONT_STACK } from '../../lib/viz/canvas';
import {
  drawContourMap,
  drawEmptyState,
  drawFunction,
  drawPath,
  drawResiduals,
  drawScatter,
  drawSeries,
} from '../../lib/viz/plots';
import { rgba } from '../../lib/viz/palette';
import type { RegressionData } from '../../lib/datasets/types';
import type { LinRegState, Normaliser } from '../../lib/ml/linearRegression';
import { computeLoss, predict } from '../../lib/ml/linearRegression';
import type { LinRegConfig } from '../../lib/ml/linearRegression';

/* ---------------- the fit ---------------- */

export interface FitPlotProps {
  data: RegressionData;
  state: LinRegState;
  config: LinRegConfig;
  norm: Normaliser;
  closedForm: number[] | null;
  showResiduals: boolean;
  showSquares: boolean;
  showTruth: boolean;
  showClosedForm: boolean;
  /** Index of a point the reader is dragging, if any. */
  activeIndex: number | null;
  /** Index of the point under the pointer, for the cursor. */
  hoverIndex?: number | null;
  dragging?: boolean;
  onPointerDown?: (pos: { x: number; y: number }) => void;
  onPointerMove?: (pos: { x: number; y: number } | null) => void;
  onPointerUp?: () => void;
  /** Convert canvas pixels to data coordinates for the drag handlers. */
  onFrame?: (frame: ReturnType<typeof makeFrame>) => void;
}

export function FitPlot({
  data,
  state,
  config,
  norm,
  closedForm,
  showResiduals,
  showSquares,
  showTruth,
  showClosedForm,
  activeIndex,
  hoverIndex = null,
  dragging = false,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onFrame,
}: FitPlotProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, data.xRange, data.yRange, {
        left: 48,
        bottom: 34,
        right: 16,
        top: 14,
      });
      onFrame?.(frame);

      drawAxes(ctx, frame, palette, {
        xLabel: data.xLabel,
        yLabel: data.yLabel,
        xTicks: 6,
        yTicks: 5,
      });

      const fitted = (x: number) => predict(state.weights, x, config.degree, norm);

      if (showResiduals && !state.diverged) {
        drawResiduals(ctx, frame, palette, data.points, fitted, {
          showSquares,
          colour: palette.red,
        });
        if (activeIndex !== null) {
          drawResiduals(ctx, frame, palette, [data.points[activeIndex]], fitted, {
            colour: palette.red,
            alpha: 0.9,
          });
        }
      }

      if (showTruth && data.truth) {
        drawFunction(ctx, frame, data.truth, palette.muted, { width: 2, dash: [5, 4] });
      }

      if (showClosedForm && closedForm) {
        drawFunction(
          ctx,
          frame,
          (x) => predict(closedForm, x, config.degree, norm),
          palette.green,
          { width: 1.8, dash: [2, 4] },
        );
      }

      drawScatter(ctx, frame, palette, data.points, {
        radius: 4,
        colourOf: () => palette.blue,
        highlight: activeIndex !== null ? [activeIndex] : undefined,
        dim:
          config.batchMode !== 'batch' && state.epoch > 0 && state.lastBatch.length < data.points.length
            ? invert(data.points.length, state.lastBatch)
            : undefined,
      });

      if (!state.diverged) {
        drawFunction(ctx, frame, fitted, palette.orange, { width: 2.8 });
      } else {
        ctx.save();
        ctx.font = '600 13px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.fillText(
          !config.standardise && config.degree > 1
            ? 'The weights overflowed. Turn on Standardise x.'
            : 'The weights overflowed. Lower the learning rate.',
          (frame.left + frame.right) / 2,
          (frame.top + frame.bottom) / 2,
        );
        ctx.restore();
      }
    },
    [
      data,
      state,
      config.degree,
      config.batchMode,
      config.standardise,
      norm,
      closedForm,
      showResiduals,
      showSquares,
      showTruth,
      showClosedForm,
      activeIndex,
      onFrame,
    ],
  );

  return (
    <Chart
      draw={draw}
      height={(w) => Math.round(Math.min(430, Math.max(260, w * 0.52)))}
      description={
        'Scatter plot of ' +
        data.points.length +
        ' points with the fitted ' +
        (config.degree === 1 ? 'line' : 'degree-' + config.degree + ' curve') +
        ' after ' +
        state.epoch +
        ' steps.'
      }
      cursor={dragging ? 'grabbing' : hoverIndex !== null ? 'grab' : 'default'}
      drag={Boolean(onPointerDown)}
      onPointerDown={onPointerDown ? (pos) => onPointerDown(pos) : undefined}
      onPointerMove={onPointerMove ? (pos) => onPointerMove(pos) : undefined}
      onPointerUp={onPointerUp ? () => onPointerUp() : undefined}
      redrawKey={state.epoch + ':' + activeIndex}
    />
  );
}

function invert(total: number, subset: readonly number[]): number[] {
  const inSubset = new Set(subset);
  const out: number[] = [];
  for (let i = 0; i < total; i++) if (!inSubset.has(i)) out.push(i);
  return out;
}

/* ---------------- loss curve ---------------- */

export function LossCurve({
  history,
  epoch,
  initialLoss,
  closedFormLoss,
  logScale,
  penalised,
}: {
  history: readonly number[];
  /** Steps taken; the history keeps only its tail, which ends here. */
  epoch: number;
  /** The loss before any step, so the chart has axes and a starting point at step 0. */
  initialLoss: number;
  closedFormLoss: number | null;
  logScale: boolean;
  /** The loss carries an L2 term. */
  penalised: boolean;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      // Step 0 is on the chart only while the whole history is kept.
      const first = Math.max(0, epoch - history.length);
      const series = first === 0 ? [initialLoss, ...history] : history;
      const x0 = first === 0 ? 0 : first + 1;
      if (!series.some((v) => Number.isFinite(v))) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'The loss overflowed.');
        return;
      }

      const values = series.map((v) => (logScale && Number.isFinite(v) ? Math.log10(Math.max(v, 1e-12)) : v));
      const [lo, hi] = extentOf(values.filter((v) => Number.isFinite(v)));
      const yDomain: [number, number] = logScale ? padExtent(lo, hi, 0.12) : [0, hi * 1.12 || 1];
      const frame = makeFrame(width, height, [x0, Math.max(x0 + 1, epoch)], yDomain, {
        left: 54,
        bottom: 30,
        right: 14,
        top: 12,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'step',
        yLabel: (logScale ? 'log₁₀ ' : '') + (penalised ? 'MSE + λΣw²' : logScale ? 'loss' : 'MSE'),
        yTicks: 4,
      });

      if (closedFormLoss !== null && Number.isFinite(closedFormLoss)) {
        const target = logScale ? Math.log10(Math.max(closedFormLoss, 1e-12)) : closedFormLoss;
        if (target >= yDomain[0] && target <= yDomain[1]) {
          const y = frame.y(target);
          drawPath(
            ctx,
            [
              { x: frame.left, y },
              { x: frame.right, y },
            ],
            palette.green,
            1.5,
            [4, 4],
          );
        }
      }

      drawSeries(ctx, frame, palette, [{ values, x0, colour: palette.orange, width: 2, fill: true }]);
    },
    [history, epoch, initialLoss, closedFormLoss, logScale, penalised],
  );

  return (
    <Chart
      draw={draw}
      height={200}
      description={
        history.length === 0
          ? 'Loss curve at step 0, the starting loss only.'
          : 'Loss over ' + history.length + ' training steps.'
      }
      redrawKey={epoch + ':' + history.length + ':' + logScale + ':' + initialLoss}
    />
  );
}

/* ---------------- loss surface ---------------- */

/** The loss over the two weights with the descent path. Degree 1 only. */
export function LossSurface({
  data,
  state,
  config,
  norm,
  closedForm,
}: {
  data: RegressionData;
  state: LinRegState;
  config: LinRegConfig;
  norm: Normaliser;
  closedForm: number[] | null;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (config.degree !== 1) {
        drawEmptyState(
          ctx,
          width,
          height,
          palette.textFaint,
          'The surface is only drawable with two weights (degree 1).',
        );
        return;
      }

      // Centre the view on the optimum.
      const centreB = closedForm ? closedForm[0] : 0;
      const centreW = closedForm ? closedForm[1] : 0;
      const spanB = Math.max(6, Math.abs(centreB) * 0.9 + 6);
      const spanW = Math.max(6, Math.abs(centreW) * 0.9 + 6);

      const frame = makeFrame(
        width,
        height,
        [centreB - spanB, centreB + spanB],
        [centreW - spanW, centreW + spanW],
        { left: 46, bottom: 32, right: 14, top: 12 },
      );

      drawContourMap(
        ctx,
        frame,
        palette,
        (b, w) => computeLoss([b, w], data.points, config, norm),
        { cellSize: 5, levels: 14 },
      );

      drawAxes(ctx, frame, palette, {
        xLabel: 'bias b',
        yLabel: 'weight w',
        showGrid: false,
        yTicks: 4,
      });

      // The descent path, kept inside the plot when a large α throws it off the map.
      ctx.save();
      clipFrame(ctx, frame);
      const path = state.weightHistory
        .filter((w) => Number.isFinite(w[0]) && Number.isFinite(w[1]))
        .map((w) => ({ x: frame.x(w[0]), y: frame.y(w[1]) }));
      if (path.length > 1) {
        drawPath(ctx, path, rgba(palette.text, 0.75), 1.6);
      }

      if (closedForm) {
        const cx = frame.x(closedForm[0]);
        const cy = frame.y(closedForm[1]);
        ctx.save();
        ctx.strokeStyle = palette.green;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy);
        ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5);
        ctx.lineTo(cx, cy + 5);
        ctx.stroke();
        ctx.restore();
      }

      if (Number.isFinite(state.weights[0]) && Number.isFinite(state.weights[1])) {
        drawPoint(
          ctx,
          frame.x(state.weights[0]),
          frame.y(state.weights[1]),
          5,
          palette.orange,
          rgba(palette.surface, 0.95),
          2,
        );
      }
      ctx.restore();
    },
    [data, state, config, norm, closedForm],
  );

  return (
    <Chart
      draw={draw}
      height={230}
      description="Contour map of the loss over the two weights, with the path gradient descent has taken."
      redrawKey={state.epoch + ':' + config.degree}
    />
  );
}
