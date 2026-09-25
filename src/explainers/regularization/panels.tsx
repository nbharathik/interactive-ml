/** Ridge and lasso charts: the fitted curve and the two-weight plane side by side, coefficient bars, the error curve. */

import { useCallback, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { usePalette } from '../../components/ThemeProvider';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { Legend } from '../../explainer/components/Panels';
import type { LegendItem } from '../../explainer/components/Panels';
import { DIM, strokeOutline } from '../../explainer/diagramStyle';
import { useLessonTarget } from '../../explainer/lessonFocus';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { clamp, clipFrame, drawAxes, drawCross, drawLabelPlate, drawPath, drawPoint, extentOf, makeFrame, FONT_STACK, MONO_STACK } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { categorical, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawEmptyState, drawFunction, drawResiduals } from '../../lib/viz/plots';
import { fmt, fmtKnob } from '../../lib/math/stats';
import { powersOf } from '../../lib/datasets/polynomial';
import type { PolynomialData } from '../../lib/datasets/polynomial';
import type { TabularData } from '../../lib/datasets/tabular';
import { predictRaw } from '../../lib/ml/elasticNet';
import type { ElasticNetState, PathResult, Prepared } from '../../lib/ml/elasticNet';
import {
  axisRadius,
  boundary,
  constrainedMinimum,
  ellipse,
  levelOf,
  quadraticAt,
  shapeValue,
  sliceQuadratic,
} from '../../lib/ml/penaltyGeometry';
import type { Point2, Quadratic2 } from '../../lib/ml/penaltyGeometry';
import { featureOf, nodeOf, sameTarget, targetKey } from './targets';
import type { RegTarget } from './targets';

export function featureColour(palette: Palette, index: number): string {
  return categorical(palette, index);
}

const LOG_FLOOR = 1e-6;
const log = (v: number) => Math.log10(Math.max(v, LOG_FLOOR));

/** The axis name of a weight: x₂ gets w₂, x² gets w(x²). */
export function weightName(feature: string): string {
  return /^x[₀-₉]+$/.test(feature) ? 'w' + feature.slice(1) : 'w(' + feature + ')';
}

/** One term of the model on its own: the mean of y plus wⱼ times the standardised feature. */
export function termCurve(w: readonly number[], prep: Prepared, j: number): (x: number) => number {
  const wj = Number.isFinite(w[j]) ? w[j] : 0;
  return (x) => prep.yMean + (wj * (Math.pow(x, j + 1) - prep.means[j])) / prep.stds[j];
}

/* ---------------- the diagram ---------------- */

export interface RegDiagramProps {
  state: ElasticNetState;
  prep: Prepared;
  alpha: number;
  featureNames: readonly string[];
  /** True weights in the prepared space, or null when not comparable. */
  truth: readonly number[] | null;
  ols: readonly number[];
  showOls: boolean;
  /** Draw the wave the curve came from, or the true weights of the table. */
  showTruth: boolean;
  /** Every row, fitted and held out. */
  data: TabularData;
  /** The one-input data, when the curve is on show. */
  curve: PolynomialData | null;
  /** The two features the plane draws. */
  pair: [number, number];
  /** A least-squares point the reader dragged, or null for the real one. */
  centre: Point2 | null;
  onDragCentre: (centre: Point2 | null) => void;
  hover: RegTarget | null;
  open: RegTarget | null;
  onHover: (target: RegTarget | null) => void;
  onOpen: (target: RegTarget | null) => void;
  /** The inspector column: the map, the readout and the open card. */
  inspector: ReactNode;
  /** The axis picker for the plane, when there is a choice. */
  picker?: ReactNode;
  /** The solver's step name, for the descriptions. */
  unit: string;
  redrawKey: string;
}

export function RegDiagram(props: RegDiagramProps) {
  const { state, prep, alpha, featureNames, curve, pair, hover, open, onHover, onOpen, inspector, picker, unit, showOls, showTruth } = props;
  const targets = useMemo<RegTarget[]>(() => Array.from({ length: prep.p }, (_, index) => ({ kind: 'feature', index })), [prep.p]);
  const cursor = hover && hover.kind === 'feature' ? hover : open && open.kind === 'feature' ? open : null;
  const handleKey = useDiagramKeys<RegTarget>({
    targets,
    cursor,
    same: sameTarget,
    onCursor: onHover,
    onOpen,
    onClose: () => {
      onOpen(null);
      onHover(null);
    },
  });
  const palette = usePalette();
  const fitLegend = useMemo<LegendItem[]>(
    () => [
      { label: 'fitted', colour: palette.blue, shape: 'dot' },
      { label: 'held out', colour: palette.orange, shape: 'ring' },
      { label: curve ? 'fit' : 'ŷ = y', colour: featureColour(palette, 0), shape: 'line' },
      ...(curve && showOls ? [{ label: 'least squares', colour: palette.textMuted, shape: 'dashed' as const }] : []),
      ...(curve && showTruth ? [{ label: 'wave', colour: palette.muted, shape: 'dashed' as const }] : []),
    ],
    [palette, curve, showOls, showTruth],
  );
  const planeLegend = useMemo<LegendItem[]>(
    () => [
      { label: 'equal error', colour: palette.textMuted, shape: 'line' },
      { label: alpha === 1 ? 'diamond' : alpha === 0 ? 'disc' : 'region', colour: palette.accent, shape: 'line' },
      { label: alpha === 1 ? 'disc' : alpha === 0 ? 'diamond' : 'pure shapes', colour: palette.textMuted, shape: 'dashed' },
      ...(showOls ? [{ label: 'least squares', colour: palette.green, shape: 'cross' as const }] : []),
      { label: 'answer', colour: palette.orange, shape: 'dot' },
      // The hollow dot: where the other pure penalty would land at the same budget.
      ...(alpha === 0 || alpha === 1 ? [{ label: alpha === 1 ? 'ridge answer' : 'lasso answer', colour: palette.textMuted, shape: 'ring' as const }] : []),
    ],
    [palette, alpha, showOls],
  );
  const after = ' after ' + state.epoch + ' ' + unit + (state.epoch === 1 ? '' : 's') + '.';
  const fitDescription = curve
    ? 'The fitted curve over ' + curve.x.length + ' points, with the wave the data came from,' + after
    : 'Every row’s prediction against its y' + after;
  const planeDescription =
    'Error rings over ' + weightName(featureNames[pair[0]]) + ' and ' + weightName(featureNames[pair[1]]) + ' with the ' +
    (alpha === 1 ? 'diamond' : alpha === 0 ? 'disc' : 'elastic net') + ' penalty region and the answer where they touch.';
  return (
    <div
      className="mlx-arch"
      tabIndex={0}
      role="group"
      aria-label={fitDescription + ' ' + planeDescription + ' Arrow keys move between features, Enter opens one.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) onHover(null);
      }}
    >
      <div className="mlx-arch__stage mlx-reg-stage">
        <Pane id="fit" title={curve ? 'The fit' : 'Prediction against y'} legend={fitLegend}>
          {curve ? <FitChart {...props} curve={curve} description={fitDescription} /> : <TableFitChart {...props} description={fitDescription} />}
        </Pane>
        <Pane id="geometry" title="Two weights" legend={planeLegend} tool={picker}>
          <GeometryChart {...props} description={planeDescription} />
        </Pane>
      </div>
      {inspector}
    </div>
  );
}

/** One chart of the stage with its name, an optional tool and its legend over it. */
function Pane({ id, title, legend, tool, children }: { id: 'fit' | 'geometry'; title: string; legend: LegendItem[]; tool?: ReactNode; children: ReactNode }) {
  const target = useLessonTarget('custom', 'pane:' + id);
  return (
    <section className="mlx-reg-pane" {...target.attrs}>
      <header className="mlx-reg-pane__head">
        <h4 className="mlx-reg-pane__title">{title}</h4>
        {tool}
        <Legend dense items={legend} />
      </header>
      {children}
    </section>
  );
}

/* ---------------- the fitted curve ---------------- */

type ChartProps = RegDiagramProps & { description: string };

function FitChart({ state, prep, ols, showOls, showTruth, curve, hover, open, onHover, onOpen, description, redrawKey }: ChartProps & { curve: PolynomialData }) {
  const frameRef = useRef<Frame | null>(null);
  const litPoint = open && open.kind === 'point' ? open.index : hover && hover.kind === 'point' ? hover.index : null;
  const litFeature = featureOf(open) ?? featureOf(hover);
  const litNode = nodeOf(open) ?? nodeOf(hover);
  const curveHover = hover && hover.kind === 'curve' ? hover.x : null;
  // The frame hugs the samples and the wave over them, not the whole domain the data could span.
  const yRange = useMemo<[number, number]>(() => {
    const ys = curve.y.slice();
    const [x0, x1] = curve.xRange;
    for (let k = 0; k <= 40; k++) ys.push(curve.truth(x0 + ((x1 - x0) * k) / 40));
    const [lo, hi] = extentOf(ys);
    const pad = (hi - lo || 1) * 0.12;
    return [lo - pad, hi + pad];
  }, [curve]);
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const degree = prep.p;
      const frame = makeFrame(width, height, curve.xRange, yRange, { left: 46, bottom: 30, right: 14, top: 8 });
      frameRef.current = frame;
      drawAxes(ctx, frame, palette, { xLabel: 'x', yLabel: 'y', xTicks: 6, yTicks: 5 });
      const fitted = (x: number) => (state.diverged ? NaN : predictRaw(state.w, prep, powersOf(x, degree)));
      const dimCurve = litNode === 'data' ? DIM : 1;
      const dimPoints = litNode === 'prediction' || litNode === 'sum' ? DIM : 1;
      if (showTruth) drawFunction(ctx, frame, curve.truth, rgba(palette.muted, dimCurve), { width: 2, dash: [5, 4] });
      if (showOls) {
        drawFunction(ctx, frame, (x) => predictRaw(ols, prep, powersOf(x, degree)), rgba(palette.textMuted, dimCurve), { width: 1.6, dash: [2, 4], samples: 400 });
      }
      // Hovering a feature shows its term on its own, around the mean of y.
      if (litFeature !== null && litFeature < degree && !state.diverged) {
        const yMean = frame.y(prep.yMean);
        drawPath(ctx, [{ x: frame.left, y: yMean }, { x: frame.right, y: yMean }], rgba(palette.textFaint, 0.8), 1, [2, 3]);
        drawFunction(ctx, frame, termCurve(state.w, prep, litFeature), featureColour(palette, litFeature), { width: 2, dash: [6, 4], samples: 300 });
      }
      // The error station shows every residual; a hovered point shows its own.
      const points = curve.x.map((x, k) => ({ x, y: curve.y[k] }));
      const test = new Set(curve.testIndex);
      if ((litNode === 'error' || litNode === 'objective') && !state.diverged) {
        drawResiduals(ctx, frame, palette, curve.trainIndex.map((k) => points[k]), fitted, { colour: palette.red, alpha: 0.55 });
        drawResiduals(ctx, frame, palette, curve.testIndex.map((k) => points[k]), fitted, { colour: palette.orange, alpha: 0.45 });
      }
      for (let k = 0; k < curve.x.length; k++) {
        const px = frame.x(curve.x[k]);
        const py = frame.y(curve.y[k]);
        const alpha = (litPoint === null || litPoint === k ? 1 : 0.55) * dimPoints;
        if (test.has(k)) drawPoint(ctx, px, py, 4, rgba(palette.surface, alpha), rgba(palette.orange, alpha), 1.6);
        else drawPoint(ctx, px, py, 4, rgba(palette.blue, alpha), rgba(palette.surface, 0.9 * alpha), 1);
      }
      if (state.diverged) {
        ctx.save();
        ctx.font = '600 12px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.fillText('The weights went to infinity, lower the step size.', (frame.left + frame.right) / 2, (frame.top + frame.bottom) / 2);
        ctx.restore();
      } else {
        const strong = litNode === 'prediction' || litNode === 'sum' || curveHover !== null;
        drawFunction(ctx, frame, fitted, rgba(featureColour(palette, 0), dimCurve), { width: strong ? 3.4 : 2.6, samples: 400 });
      }
      if (curveHover !== null && !state.diverged) {
        const yHat = fitted(curveHover);
        if (Number.isFinite(yHat)) drawPoint(ctx, frame.x(curveHover), frame.y(yHat), 4.5, palette.surface, featureColour(palette, 0), 2);
      }
      if (litPoint !== null && litPoint < curve.x.length) {
        const px = frame.x(curve.x[litPoint]);
        const py = frame.y(curve.y[litPoint]);
        const yHat = fitted(curve.x[litPoint]);
        if (Number.isFinite(yHat)) drawPath(ctx, [{ x: px, y: py }, { x: px, y: frame.y(yHat) }], rgba(palette.red, 0.8), 1.5);
        strokeOutline(
          ctx,
          () => {
            ctx.beginPath();
            ctx.arc(px, py, 8, 0, Math.PI * 2);
          },
          open && open.kind === 'point' && open.index === litPoint ? 'open' : 'hover',
          palette,
        );
      }
    },
    [state, prep, ols, showOls, showTruth, curve, yRange, litPoint, litFeature, litNode, curveHover, open],
  );

  /** The point nearest the pointer, else the curve under it. */
  const locate = useCallback(
    (pos: { x: number; y: number }): RegTarget | null => {
      const frame = frameRef.current;
      if (!frame) return null;
      let best: number | null = null;
      let bestDist = 11;
      for (let k = 0; k < curve.x.length; k++) {
        const d = Math.hypot(frame.x(curve.x[k]) - pos.x, frame.y(curve.y[k]) - pos.y);
        if (d < bestDist) {
          bestDist = d;
          best = k;
        }
      }
      if (best !== null) return { kind: 'point', index: best };
      if (state.diverged || pos.x < frame.left || pos.x > frame.right) return null;
      const x = frame.x.invert(pos.x);
      const yHat = predictRaw(state.w, prep, powersOf(x, prep.p));
      return Number.isFinite(yHat) && Math.abs(frame.y(yHat) - pos.y) < 8 ? { kind: 'curve', x } : null;
    },
    [curve, state, prep],
  );

  const hot = hover && (hover.kind === 'point' || hover.kind === 'curve');
  return (
    <Chart
      draw={draw}
      height="fill"
      description={description}
      cursor={hot ? 'pointer' : 'default'}
      onPointerDown={(pos) => {
        const found = locate(pos);
        if (!found) {
          onOpen(null);
          return;
        }
        const target: RegTarget = found.kind === 'curve' ? { kind: 'node', id: 'prediction' } : found;
        onOpen(sameTarget(target, open) ? null : target);
      }}
      onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
      onPointerLeave={() => onHover(null)}
      redrawKey={redrawKey + '|' + targetKey(hover) + '|' + targetKey(open)}
    />
  );
}

/* ---------------- prediction against y, for the table ---------------- */

function TableFitChart({ state, prep, data, hover, open, onHover, onOpen, description, redrawKey }: ChartProps) {
  const frameRef = useRef<Frame | null>(null);
  const litPoint = open && open.kind === 'point' ? open.index : hover && hover.kind === 'point' ? hover.index : null;
  const litNode = nodeOf(open) ?? nodeOf(hover);
  const yHat = useMemo(() => data.X.map((row) => (state.diverged ? NaN : predictRaw(state.w, prep, row))), [data, state, prep]);
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const finite = yHat.filter(Number.isFinite);
      const [lo, hi] = extentOf(data.y.concat(finite));
      const pad = (hi - lo || 1) * 0.08;
      const domain: [number, number] = [lo - pad, hi + pad];
      const frame = makeFrame(width, height, domain, domain, { left: 46, bottom: 30, right: 14, top: 8, equal: true });
      frameRef.current = frame;
      drawAxes(ctx, frame, palette, { xLabel: 'y', yLabel: 'ŷ', xTicks: 5, yTicks: 5 });
      const test = new Set(data.testIndex);
      const dimPoints = litNode === 'prediction' || litNode === 'sum' ? DIM : 1;
      const strong = litNode === 'prediction' || litNode === 'sum';
      ctx.save();
      clipFrame(ctx, frame);
      // A perfect prediction sits on the diagonal.
      drawPath(ctx, [{ x: frame.x(domain[0]), y: frame.y(domain[0]) }, { x: frame.x(domain[1]), y: frame.y(domain[1]) }], rgba(featureColour(palette, 0), litNode === 'data' ? DIM : 1), strong ? 3 : 2);
      // The error station shows every residual as the drop onto the diagonal.
      if (litNode === 'error' || litNode === 'objective') {
        for (let k = 0; k < data.y.length; k++) {
          if (!Number.isFinite(yHat[k])) continue;
          const px = frame.x(data.y[k]);
          drawPath(ctx, [{ x: px, y: frame.y(yHat[k]) }, { x: px, y: frame.y(data.y[k]) }], rgba(test.has(k) ? palette.orange : palette.red, 0.55), 1);
        }
      }
      ctx.restore();
      for (let k = 0; k < data.y.length; k++) {
        if (!Number.isFinite(yHat[k])) continue;
        const px = frame.x(data.y[k]);
        const py = frame.y(yHat[k]);
        const alpha = (litPoint === null || litPoint === k ? 1 : 0.55) * dimPoints;
        if (test.has(k)) drawPoint(ctx, px, py, 4, rgba(palette.surface, alpha), rgba(palette.orange, alpha), 1.6);
        else drawPoint(ctx, px, py, 4, rgba(palette.blue, alpha), rgba(palette.surface, 0.9 * alpha), 1);
      }
      if (state.diverged) {
        ctx.save();
        ctx.font = '600 12px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.fillText('The weights went to infinity, lower the step size.', (frame.left + frame.right) / 2, (frame.top + frame.bottom) / 2);
        ctx.restore();
      }
      if (litPoint !== null && litPoint < data.y.length && Number.isFinite(yHat[litPoint])) {
        const px = frame.x(data.y[litPoint]);
        const py = frame.y(yHat[litPoint]);
        drawPath(ctx, [{ x: px, y: py }, { x: px, y: frame.y(data.y[litPoint]) }], rgba(palette.red, 0.8), 1.5);
        strokeOutline(
          ctx,
          () => {
            ctx.beginPath();
            ctx.arc(px, py, 8, 0, Math.PI * 2);
          },
          open && open.kind === 'point' && open.index === litPoint ? 'open' : 'hover',
          palette,
        );
      }
    },
    [state.diverged, data, yHat, litPoint, litNode, open],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): RegTarget | null => {
      const frame = frameRef.current;
      if (!frame) return null;
      let best: number | null = null;
      let bestDist = 11;
      for (let k = 0; k < data.y.length; k++) {
        if (!Number.isFinite(yHat[k])) continue;
        const d = Math.hypot(frame.x(data.y[k]) - pos.x, frame.y(yHat[k]) - pos.y);
        if (d < bestDist) {
          bestDist = d;
          best = k;
        }
      }
      return best === null ? null : { kind: 'point', index: best };
    },
    [data, yHat],
  );

  return (
    <Chart
      draw={draw}
      height="fill"
      description={description}
      cursor={hover && hover.kind === 'point' ? 'pointer' : 'default'}
      onPointerDown={(pos) => {
        const found = locate(pos);
        onOpen(found && sameTarget(found, open) ? null : found);
      }}
      onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
      onPointerLeave={() => onHover(null)}
      redrawKey={redrawKey + '|' + targetKey(hover) + '|' + targetKey(open)}
    />
  );
}

/* ---------------- the two-weight geometry ---------------- */

/** What the picture shows for one penalty shape at one budget. */
export interface Touch {
  alpha: number;
  level: number;
  point: Point2;
}

export interface GeometryModel {
  quad: Quadratic2;
  /** The shape of the penalty in use, at the level the current weights reach. */
  active: Touch;
  /** The pure shape the reader is not using, at the same axis radius; none for an elastic mix. */
  other: Touch | null;
  /** Where the boundaries cross the axes. */
  radius: number;
  dragged: boolean;
}

/** The geometry of two weights: the quadratic, the region at the current budget, and both touching points. */
export function geometryOf(
  prep: Prepared,
  w: readonly number[],
  pair: [number, number],
  alpha: number,
  centre: Point2 | null,
): GeometryModel | null {
  const [i, j] = pair;
  if (prep.p < 2 || i === j) return null;
  const real = sliceQuadratic(prep, w, i, j);
  const wi = Number.isFinite(w[i]) ? w[i] : 0;
  const wj = Number.isFinite(w[j]) ? w[j] : 0;
  const level = shapeValue(alpha, [wi, wj]);
  const radius = axisRadius(alpha, level);
  const quad = centre ? { ...real, centre } : real;
  const active: Touch = { alpha, level, point: centre ? constrainedMinimum(quad, alpha, level) : [wi, wj] };
  let other: Touch | null = null;
  if (alpha === 0 || alpha === 1) {
    const otherAlpha = 1 - alpha;
    const otherLevel = levelOf(otherAlpha, radius);
    other = { alpha: otherAlpha, level: otherLevel, point: constrainedMinimum(quad, otherAlpha, otherLevel) };
  }
  return { quad, active, other, radius, dragged: centre !== null };
}

type Mark = 'centre' | 'now' | 'other';

function GeometryChart({ state, prep, alpha, featureNames, ols, showOls, pair, centre, onDragCentre, hover, open, onHover, onOpen, description, redrawKey }: ChartProps) {
  const frameRef = useRef<Frame | null>(null);
  const [dragging, setDragging] = useState(false);
  const model = useMemo(() => geometryOf(prep, state.w, pair, alpha, centre), [prep, state.w, pair, alpha, centre]);
  const [i, j] = pair;
  const mark: Mark | null = hover && hover.kind === 'mark' ? hover.which : null;
  const litFeature = featureOf(open) ?? featureOf(hover);
  const litNode = nodeOf(open) ?? nodeOf(hover);
  const ringsLit = litNode === 'error' || litNode === 'objective';
  const regionLit = litNode === 'penalty' || litNode === 'objective';

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (!model) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'The plane needs two features.');
        return;
      }
      const { quad, active, other, radius } = model;
      // The frame follows the real picture so a drag never moves it.
      const span = Math.max(1, Math.abs(ols[i]), Math.abs(ols[j]), radius, Math.abs(active.point[0]), Math.abs(active.point[1])) * 1.35 + 0.4;
      const frame = makeFrame(width, height, [-span, span], [-span, span], { left: 46, bottom: 30, right: 14, top: 8, equal: true });
      frameRef.current = frame;
      const xName = weightName(featureNames[i]);
      const yName = weightName(featureNames[j]);
      drawAxes(ctx, frame, palette, { xLabel: xName, yLabel: yName, xTicks: 5, yTicks: 5, showZeroLines: true });
      const px = (p: Point2) => ({ x: frame.x(p[0]), y: frame.y(p[1]) });

      ctx.save();
      clipFrame(ctx, frame);
      // Rings of equal squared error, evenly spaced in radius; the one through the answer is stronger.
      const touch = quadraticAt(quad, active.point) - quad.base;
      const e2 = (quad.g11 + quad.g22) / 2 - Math.sqrt(((quad.g11 - quad.g22) / 2) ** 2 + quad.g12 * quad.g12);
      const reach = (Math.max(e2, 1e-6) * (2 * span) ** 2) / 2;
      const ringAlpha = ringsLit ? 0.62 : litNode !== null || litFeature !== null ? 0.22 : 0.32;
      for (let k = 1; k <= 8; k++) {
        const rise = reach * (k / 8) ** 2;
        drawPath(ctx, ellipse(quad, quad.base + rise).map(px), rgba(palette.textMuted, ringAlpha), ringsLit ? 1.3 : 1);
      }
      if (touch > 1e-9) drawPath(ctx, ellipse(quad, quad.base + touch).map(px), rgba(palette.text, ringsLit ? 0.95 : 0.75), ringsLit ? 2 : 1.6);

      // The penalty region in use, then the other shape at the same axis radius.
      const shapeAlpha = regionLit ? 0.95 : 0.7;
      if (other) drawPath(ctx, boundary(other.alpha, other.level).map(px), rgba(palette.textMuted, shapeAlpha), 1.4, [5, 4]);
      if (alpha > 0 && alpha < 1 && radius > 0) {
        drawPath(ctx, boundary(1, levelOf(1, radius)).map(px), rgba(palette.textMuted, 0.4), 1, [5, 4]);
        drawPath(ctx, boundary(0, levelOf(0, radius)).map(px), rgba(palette.textMuted, 0.4), 1, [5, 4]);
      }
      if (active.level > 0) {
        const region = boundary(alpha, active.level).map(px);
        ctx.beginPath();
        region.forEach((p, k) => (k === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.closePath();
        ctx.fillStyle = rgba(palette.accent, regionLit ? 0.22 : 0.1);
        ctx.fill();
        drawPath(ctx, region, palette.accent, regionLit ? 2.8 : 2);
      }

      if (!model.dragged) {
        const trail = state.wHistory.filter((w) => Number.isFinite(w[i]) && Number.isFinite(w[j])).map((w) => px([w[i], w[j]]));
        if (trail.length > 1) drawPath(ctx, trail, rgba(palette.text, 0.5), 1.2);
      }

      // A hovered weight of the pair: its axis lights, and a guide drops from the answer onto it.
      const now = px(active.point);
      if (litFeature === i || litFeature === j) {
        const onX = litFeature === i;
        const foot = onX ? { x: now.x, y: frame.y(0) } : { x: frame.x(0), y: now.y };
        drawPath(ctx, [now, foot], rgba(palette.accent, 0.9), 1.2, [3, 3]);
        drawPoint(ctx, foot.x, foot.y, 3, palette.accent);
      }
      ctx.restore();

      const c = px(quad.centre);
      if (showOls || model.dragged) {
        if (mark === 'centre' || dragging) drawPoint(ctx, c.x, c.y, 11, rgba(palette.green, 0.15));
        drawCross(ctx, c.x, c.y, 6, palette.green, 2.2);
      }
      if (other) {
        const o = px(other.point);
        drawPoint(ctx, o.x, o.y, mark === 'other' ? 5.5 : 4.5, palette.surface, palette.textMuted, 1.6);
      }
      drawPoint(ctx, now.x, now.y, mark === 'now' ? 7 : 6, palette.orange, rgba(palette.surface, 0.95), 1.6);
      // The guide's value, on a plate over the marks.
      if (litFeature === i || litFeature === j) {
        const onX = litFeature === i;
        const value = onX ? active.point[0] : active.point[1];
        const text = (onX ? xName : yName) + ' = ' + fmt(value, 2);
        if (onX) drawLabelPlate(ctx, text, now.x, frame.y(0) + 12, palette, { align: 'center', colour: palette.accent, scale: 0.9 });
        else drawLabelPlate(ctx, text, frame.x(0) - 6, now.y, palette, { align: 'right', colour: palette.accent, scale: 0.9 });
      }
      if (litNode === 'features' || litNode === 'sum') {
        strokeOutline(
          ctx,
          () => {
            ctx.beginPath();
            ctx.arc(now.x, now.y, 10, 0, Math.PI * 2);
          },
          'hover',
          palette,
        );
      }

      ctx.save();
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'right';
      ctx.fillText(prep.p > 2 ? 'the other ' + (prep.p - 2) + ' weights held where they are' : 'rings: equal error', frame.right - 4, frame.top + 11);
      if (model.dragged) {
        ctx.textAlign = 'left';
        ctx.fillText('× moved, double-click to return', frame.left + 4, frame.top + 11);
      }
      ctx.restore();
    },
    [model, state.wHistory, prep.p, alpha, featureNames, ols, showOls, i, j, mark, dragging, litFeature, litNode, ringsLit, regionLit],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): Mark | null => {
      const frame = frameRef.current;
      if (!frame || !model) return null;
      const near = (p: Point2, r: number) => Math.hypot(frame.x(p[0]) - pos.x, frame.y(p[1]) - pos.y) < r;
      if (near(model.active.point, 10)) return 'now';
      if (model.other && near(model.other.point, 9)) return 'other';
      if ((showOls || model.dragged) && near(model.quad.centre, 12)) return 'centre';
      return null;
    },
    [model, showOls],
  );

  const toData = useCallback((pos: { x: number; y: number }): Point2 | null => {
    const frame = frameRef.current;
    if (!frame) return null;
    const [x0, x1] = frame.x.domain;
    const [y0, y1] = frame.y.domain;
    return [clamp(frame.x.invert(pos.x), x0, x1), clamp(frame.y.invert(pos.y), y0, y1)];
  }, []);

  return (
    <Chart
      draw={draw}
      height="fill"
      description={description}
      drag
      cursor={dragging ? 'grabbing' : mark === 'centre' ? 'grab' : mark ? 'pointer' : 'default'}
      onPointerDown={(pos, event) => {
        if (event.detail >= 2 && model?.dragged) {
          onDragCentre(null);
          return;
        }
        const found = locate(pos);
        if (found === 'centre') setDragging(true);
        // The answer opens the penalty station, the error rings the error station.
        else if (found === 'now') onOpen(sameTarget(open, { kind: 'node', id: 'penalty' }) ? null : { kind: 'node', id: 'penalty' });
        else if (found === null) onOpen(null);
      }}
      onPointerMove={(pos) => {
        if (!pos) {
          onHover(null);
          return;
        }
        if (dragging) {
          const p = toData(pos);
          if (p) onDragCentre(p);
          return;
        }
        const found = locate(pos);
        onHover(found ? { kind: 'mark', which: found } : null);
      }}
      onPointerUp={() => setDragging(false)}
      onPointerLeave={() => {
        setDragging(false);
        onHover(null);
      }}
      redrawKey={redrawKey + '|' + (centre ? centre.join(',') : '') + '|' + targetKey(hover) + '|' + targetKey(open) + '|' + dragging}
    />
  );
}

/* ---------------- coefficient bars ---------------- */

export function CoefficientBars({
  weights,
  ols,
  truth,
  names,
  showOls,
  lit,
  pair,
  onHover,
  onOpen,
}: {
  weights: readonly number[];
  ols: readonly number[] | null;
  truth: readonly number[] | null;
  names: readonly string[];
  showOls: boolean;
  /** The feature under the pointer or open anywhere on the page. */
  lit: number | null;
  /** The two weights the plane draws, ticked beside their names. */
  pair: [number, number];
  onHover: (index: number | null) => void;
  onOpen: (index: number) => void;
}) {
  const rowsRef = useRef<{ top: number; height: number }>({ top: 0, height: 1 });
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const p = weights.length;
      if (p === 0) return;
      const labelWidth = 34;
      const valueWidth = 46;
      const left = 8 + labelWidth;
      const trackWidth = Math.max(40, width - left - valueWidth - 8);
      const centre = left + trackWidth / 2;
      const rowHeight = height / p;
      rowsRef.current = { top: 0, height: rowHeight };
      const barHeight = Math.min(14, rowHeight * 0.6);
      // The fit and the truth set the scale; a wild least-squares ghost is clamped to the track's edge.
      let maxAbs = 1e-6;
      for (const v of weights) if (Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v));
      if (truth) for (const v of truth) maxAbs = Math.max(maxAbs, Math.abs(v));
      const px = (v: number) => centre + (clamp(v, -maxAbs, maxAbs) / maxAbs) * (trackWidth / 2);

      ctx.save();
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = rgba(palette.border, 0.9);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(centre) + 0.5, 0);
      ctx.lineTo(Math.round(centre) + 0.5, height);
      ctx.stroke();
      for (let j = 0; j < p; j++) {
        const cy = rowHeight * (j + 0.5);
        const colour = featureColour(palette, j);
        const alpha = lit === null || lit === j ? 1 : DIM;
        if (lit === j) {
          ctx.fillStyle = rgba(palette.accent, 0.08);
          ctx.fillRect(0, cy - rowHeight / 2, width, rowHeight);
        }
        ctx.font = (lit === j ? '600 ' : '') + '11px ' + MONO_STACK;
        ctx.fillStyle = rgba(lit === j ? palette.text : palette.textMuted, alpha);
        ctx.textAlign = 'right';
        ctx.fillText(names[j] ?? String(j + 1), left - 8, cy);
        if (j === pair[0] || j === pair[1]) {
          ctx.fillStyle = rgba(palette.accent, 0.9 * alpha);
          ctx.fillRect(2, cy - 4, 2, 8);
        }
        if (ols && showOls) {
          const x0 = Math.min(centre, px(ols[j]));
          ctx.strokeStyle = rgba(palette.textMuted, 0.5 * alpha);
          ctx.setLineDash([2, 2]);
          ctx.strokeRect(x0 + 0.5, cy - barHeight / 2 - 2.5, Math.abs(px(ols[j]) - centre), barHeight + 5);
          ctx.setLineDash([]);
        }
        const v = Number.isFinite(weights[j]) ? weights[j] : 0;
        ctx.fillStyle = rgba(colour, 0.9 * alpha);
        ctx.fillRect(Math.min(centre, px(v)), cy - barHeight / 2, Math.abs(px(v) - centre), barHeight);
        if (truth) {
          const tx = Math.round(px(truth[j])) + 0.5;
          ctx.strokeStyle = rgba(palette.text, alpha);
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(tx, cy - barHeight / 2 - 4);
          ctx.lineTo(tx, cy + barHeight / 2 + 4);
          ctx.stroke();
        }
        ctx.font = '11px ' + MONO_STACK;
        ctx.fillStyle = rgba(v === 0 ? palette.textFaint : palette.text, alpha);
        ctx.textAlign = 'left';
        ctx.fillText(v === 0 ? '0' : fmt(v, 2), left + trackWidth + 8, cy);
      }
      ctx.restore();
    },
    [weights, ols, truth, names, showOls, lit, pair],
  );
  const rowAt = useCallback(
    (pos: { x: number; y: number }): number | null => {
      const { height } = rowsRef.current;
      const j = Math.floor(pos.y / height);
      return j >= 0 && j < weights.length ? j : null;
    },
    [weights.length],
  );
  return (
    <Chart
      draw={draw}
      height={Math.max(170, 24 * weights.length + 16)}
      description={'Bar chart of the ' + weights.length + ' fitted weights, ' + weights.filter((v) => v !== 0).length + ' non-zero. Hover a bar for its numbers, click to open it.'}
      cursor="pointer"
      onPointerDown={(pos) => {
        const j = rowAt(pos);
        if (j !== null) onOpen(j);
      }}
      onPointerMove={(pos) => onHover(pos ? rowAt(pos) : null)}
      onPointerLeave={() => onHover(null)}
      redrawKey={weights.map((v) => v.toFixed(3)).join(',') + '|' + showOls + '|' + lit + '|' + pair.join()}
    />
  );
}

/* ---------------- error against lambda ---------------- */

export function ErrorCurve({ path, lambda, showTest }: { path: PathResult; lambda: number; showTest: boolean }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (path.lambdas.length < 2) return;
      const xDomain: [number, number] = [log(path.lambdas[path.lambdas.length - 1]), log(path.lambdas[0])];
      const values = showTest ? path.trainMse.concat(path.testMse) : path.trainMse;
      const [, hi] = extentOf(values);
      const frame = makeFrame(width, height, xDomain, [0, hi * 1.12 || 1], { left: 50, bottom: 30, right: 14, top: 12 });
      drawAxes(ctx, frame, palette, { xLabel: 'λ', yLabel: 'MSE', yTicks: 4, xTicks: 5, formatX: (v) => fmtKnob(Math.pow(10, v)) });
      const toPoints = (ys: number[]) => path.lambdas.map((l, k) => ({ x: frame.x(log(l)), y: frame.y(ys[k]) }));
      ctx.save();
      drawPath(ctx, toPoints(path.trainMse), palette.blue, 2);
      if (showTest) drawPath(ctx, toPoints(path.testMse), palette.orange, 2);
      ctx.restore();
      const xNow = frame.x(clamp(log(lambda), xDomain[0], xDomain[1]));
      drawPath(ctx, [{ x: xNow, y: frame.top }, { x: xNow, y: frame.bottom }], rgba(palette.text, 0.45), 1, [4, 4]);
      if (showTest) {
        const bx = frame.x(log(path.lambdas[path.bestIndex]));
        drawCross(ctx, bx, frame.y(path.testMse[path.bestIndex]), 5, palette.green, 2);
      }
    },
    [path, lambda, showTest],
  );
  return (
    <Chart
      draw={draw}
      height={200}
      description={'Training and test error across the whole range of λ; the best test error is at λ = ' + fmtKnob(path.lambdas[path.bestIndex] ?? 0) + '.'}
      redrawKey={path.lambdas.length + ':' + lambda + ':' + showTest + ':' + path.bestIndex}
    />
  );
}
