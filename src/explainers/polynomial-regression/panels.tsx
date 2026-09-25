/** Polynomial regression charts: the fit with its refits, the validation curve, the bias-variance decomposition. */

import { useCallback } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawAxes, drawLabelPlate, drawMarker, formatTick, makeFrame, FONT_STACK } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { drawEmptyState, drawFunction, drawPath, drawResiduals, drawScatter, plateRoom } from '../../lib/viz/plots';
import { rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import type { RegressionData, RegressionPoint } from '../../lib/datasets/types';
import { predictFit } from '../../lib/ml/polynomialRegression';
import type { BiasVarianceSweep, PolyFit, RefitState, ValidationCurve } from '../../lib/ml/polynomialRegression';

/* ---------------- the fit ---------------- */

export interface FitPlotProps {
  data: RegressionData;
  test: readonly RegressionPoint[];
  fit: PolyFit;
  refits: RefitState;
  /** x's the refit curves were sampled on. */
  grid: readonly number[];
  showResiduals: boolean;
  showTruth: boolean;
  showTest: boolean;
  showRefits: boolean;
  activeIndex: number | null;
  hoverIndex?: number | null;
  dragging?: boolean;
  onPointerDown?: (pos: { x: number; y: number }) => void;
  onPointerMove?: (pos: { x: number; y: number } | null) => void;
  onPointerUp?: () => void;
  onFrame?: (frame: Frame) => void;
}

export function FitPlot({
  data,
  test,
  fit,
  refits,
  grid,
  showResiduals,
  showTruth,
  showTest,
  showRefits,
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
      const frame = makeFrame(width, height, data.xRange, data.yRange, { left: 44, bottom: 34, right: 16, top: 14 });
      onFrame?.(frame);
      drawAxes(ctx, frame, palette, { xLabel: data.xLabel, yLabel: data.yLabel, xTicks: 6, yTicks: 5 });

      const fitted = (x: number) => predictFit(fit, x);

      if (showRefits && refits.curves.length > 0) {
        const alpha = Math.max(0.05, Math.min(0.2, 3 / refits.curves.length));
        for (const curve of refits.curves) drawGridCurve(ctx, frame, grid, curve, rgba(palette.orange, alpha), 1);
        if (refits.curves.length >= 2) {
          drawGridCurve(ctx, frame, grid, refits.mean, rgba(palette.orange, 0.9), 1.6, [5, 4]);
        }
      }

      if (showTruth && data.truth) {
        drawFunction(ctx, frame, data.truth, palette.muted, { width: 2, dash: [5, 4] });
      }

      if (showResiduals && !fit.failed) {
        drawResiduals(ctx, frame, palette, data.points, fitted, { colour: palette.red });
        if (activeIndex !== null) {
          drawResiduals(ctx, frame, palette, [data.points[activeIndex]], fitted, { colour: palette.red, alpha: 0.9 });
        }
      }

      if (showTest) {
        ctx.save();
        for (const p of test) {
          const px = frame.x(p.x);
          const py = frame.y(p.y);
          if (py < frame.top - 6 || py > frame.bottom + 6) continue;
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, Math.PI * 2);
          ctx.strokeStyle = palette.green;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        }
        ctx.restore();
      }

      drawScatter(ctx, frame, palette, data.points, {
        radius: 4,
        colourOf: () => palette.blue,
        highlight: activeIndex !== null ? [activeIndex] : undefined,
      });

      if (!fit.failed) {
        drawFunction(ctx, frame, fitted, palette.orange, { width: 2.8, samples: 240 });
      } else {
        ctx.save();
        ctx.font = '600 13px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.fillText('No solution: the normal equations are singular.', (frame.left + frame.right) / 2, (frame.top + frame.bottom) / 2);
        ctx.restore();
      }
    },
    [data, test, fit, refits, grid, showResiduals, showTruth, showTest, showRefits, activeIndex, onFrame],
  );

  return (
    <Chart
      draw={draw}
      height={(w) => Math.round(Math.min(430, Math.max(260, w * 0.52)))}
      description={
        'Scatter plot of ' + data.points.length + ' training points and ' + test.length +
        ' test points with the fitted degree-' + fit.degree + ' polynomial' +
        (refits.curves.length > 0 ? ' and ' + refits.curves.length + ' refits on fresh noise' : '') + '.'
      }
      cursor={dragging ? 'grabbing' : hoverIndex !== null ? 'grab' : 'default'}
      drag={Boolean(onPointerDown)}
      onPointerDown={onPointerDown ? (pos) => onPointerDown(pos) : undefined}
      onPointerMove={onPointerMove ? (pos) => onPointerMove(pos) : undefined}
      onPointerUp={onPointerUp ? () => onPointerUp() : undefined}
      redrawKey={refits.epoch + ':' + activeIndex + ':' + fit.weights.join(',')}
    />
  );
}

function drawGridCurve(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  grid: readonly number[],
  values: readonly number[],
  colour: string,
  width: number,
  dash?: number[],
) {
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < grid.length; i++) {
    if (!Number.isFinite(values[i])) continue;
    points.push({ x: frame.x(grid[i]), y: frame.y(values[i]) });
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.left, frame.top, frame.innerWidth, frame.innerHeight);
  ctx.clip();
  drawPath(ctx, points, colour, width, dash);
  ctx.restore();
}

/* ---------------- charts over degree ---------------- */

interface DegreeSeries {
  values: readonly number[];
  colour: string;
  label: string;
  dash?: number[];
}

const LOG_FLOOR = 1e-4;

/** Lines against degree on a log-10 axis, with the current degree marked. */
function drawDegreeChart(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  palette: Palette,
  degrees: readonly number[],
  series: readonly DegreeSeries[],
  degree: number,
  logScale: boolean,
  yLabel: string,
  marker?: { degree: number; label: string },
) {
  const transform = (v: number) => (logScale ? Math.log10(Math.max(v, LOG_FLOOR)) : v);
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (!Number.isFinite(v)) continue;
      const t = transform(v);
      lo = Math.min(lo, t);
      hi = Math.max(hi, t);
    }
  }
  if (!Number.isFinite(lo)) {
    drawEmptyState(ctx, width, height, palette.textFaint, 'Nothing to draw.');
    return;
  }
  const yDomain: [number, number] = logScale
    ? [Math.floor(lo - 0.05), Math.ceil(hi + 0.05)]
    : [0, hi * 1.12 || 1];
  const frame = makeFrame(width, height, [degrees[0], degrees[degrees.length - 1]], yDomain, {
    left: 52,
    bottom: 30,
    right: 14,
    top: 12,
  });
  drawAxes(ctx, frame, palette, {
    xLabel: 'degree',
    yLabel,
    xTicks: 8,
    yTicks: 4,
    formatX: (v) => String(Math.round(v)),
    formatY: logScale ? (v) => formatTick(10 ** v) : undefined,
  });

  // The current degree.
  const px = Math.round(frame.x(degree)) + 0.5;
  ctx.save();
  ctx.strokeStyle = rgba(palette.text, 0.35);
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(px, frame.top);
  ctx.lineTo(px, frame.bottom);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.left, frame.top, frame.innerWidth, frame.innerHeight);
  ctx.clip();
  for (const s of series) {
    const points: Array<{ x: number; y: number }> = [];
    degrees.forEach((d, i) => {
      const v = s.values[i];
      if (!Number.isFinite(v)) return;
      points.push({ x: frame.x(d), y: frame.y(transform(v)) });
    });
    drawPath(ctx, points, s.colour, 2, s.dash);
    const at = degrees.indexOf(degree);
    const v = at >= 0 ? s.values[at] : Number.NaN;
    if (Number.isFinite(v)) {
      drawMarker(ctx, 'circle', frame.x(degree), frame.y(transform(v)), 4, s.colour, rgba(palette.surface, 0.95));
    }
  }
  ctx.restore();

  if (marker) {
    const mx = frame.x(marker.degree);
    ctx.save();
    ctx.strokeStyle = palette.green;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(mx, frame.bottom);
    ctx.lineTo(mx, frame.bottom - 8);
    ctx.stroke();
    ctx.restore();
    if (plateRoom(frame)) drawLabelPlate(ctx, marker.label, mx, frame.bottom - 16, palette, { align: 'center', colour: palette.green });
  }

  if (plateRoom(frame)) {
    const placed = series
      .map((s) => {
        const last = s.values[s.values.length - 1];
        return Number.isFinite(last)
          ? { label: s.label, colour: s.colour, y: Math.max(frame.top + 9, Math.min(frame.bottom - 9, frame.y(transform(last)))) }
          : null;
      })
      .filter((p): p is { label: string; colour: string; y: number } => p !== null)
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < placed.length; i++) {
      if (placed[i].y - placed[i - 1].y < 17) placed[i].y = placed[i - 1].y + 17;
    }
    const overshoot = placed.length > 0 ? placed[placed.length - 1].y - (frame.bottom - 9) : 0;
    if (overshoot > 0) for (const item of placed) item.y -= overshoot;
    for (const item of placed) {
      drawLabelPlate(ctx, item.label, frame.right - 6, item.y, palette, { align: 'right', colour: item.colour, bold: true });
    }
  }
}

export function ValidationCurveChart({
  curve,
  degree,
  logScale,
}: {
  curve: ValidationCurve;
  degree: number;
  logScale: boolean;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      drawDegreeChart(
        ctx,
        width,
        height,
        palette,
        curve.degrees,
        [
          { values: curve.train, colour: palette.orange, label: 'train' },
          { values: curve.test, colour: palette.green, label: 'test' },
        ],
        degree,
        logScale,
        'MSE',
        { degree: curve.best, label: 'best ' + curve.best },
      );
    },
    [curve, degree, logScale],
  );
  return (
    <Chart
      draw={draw}
      height={220}
      description={
        'Training and test error at every degree from 0 to ' + (curve.degrees.length - 1) + '; the test error is lowest at degree ' + curve.best + '.'
      }
      redrawKey={degree + ':' + logScale + ':' + curve.test.join(',')}
    />
  );
}

export function BiasVarianceChart({ sweep, degree }: { sweep: BiasVarianceSweep; degree: number }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const total = sweep.bias2.map((b, i) => b + sweep.variance[i] + sweep.noise);
      const series: DegreeSeries[] = [
        { values: total, colour: palette.text, label: 'total' },
        { values: sweep.bias2, colour: palette.violet, label: 'bias²' },
        { values: sweep.variance, colour: palette.pink, label: 'variance' },
      ];
      if (sweep.noise > 0) {
        series.push({ values: sweep.degrees.map(() => sweep.noise), colour: palette.muted, label: 'σ²', dash: [4, 4] });
      }
      drawDegreeChart(
        ctx,
        width,
        height,
        palette,
        sweep.degrees,
        series,
        degree,
        true,
        'expected error',
      );
    },
    [sweep, degree],
  );
  return (
    <Chart
      draw={draw}
      height={220}
      description={'Bias squared, variance and the noise floor at every degree, with their total, an estimate of the test error.'}
      redrawKey={degree + ':' + sweep.variance.join(',')}
    />
  );
}
