/** K-means charts: the cluster plot, inertia over rounds, elbow and silhouette, cluster sizes. */

import { useCallback, useRef } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import {
  clamp,
  clipFrame,
  drawAxes,
  drawLabelPlate,
  drawPoint,
  extentOf,
  makeFrame,
  roundRect,
  FONT_STACK,
} from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import {
  classColour,
  drawCategoryField,
  drawEmptyState,
  drawPath,
  drawScatter,
} from '../../lib/viz/plots';
import { readableOn, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import type { PointData } from '../../lib/datasets/types';
import { fmt } from '../../lib/math/stats';
import type { KMeansState } from '../../lib/ml/kmeans';
import { nearestCentroid } from '../../lib/ml/kmeans';

/* ---------------- the cluster plot ---------------- */

export interface ClusterPlotProps {
  data: PointData;
  state: KMeansState;
  showVoronoi: boolean;
  showTrails: boolean;
  showLinks: boolean;
  showTruth: boolean;
  /** Centroid currently being dragged, if any. */
  dragIndex: number | null;
  /** Centroid under the pointer. */
  hoverIndex: number | null;
  /** Points per cluster, for the hover plate. */
  sizes?: readonly number[];
  /** Centroid whose card is open. */
  cardIndex?: number | null;
  /** Reports the pixel to data mapping for hit testing. */
  onFrame?: (frame: Frame) => void;
  onPointerDown?: (pos: { x: number; y: number }) => void;
  onPointerMove?: (pos: { x: number; y: number } | null) => void;
  onPointerUp?: (pos: { x: number; y: number }) => void;
  /** Take the container height instead of an intrinsic one (the studio hero). */
  fill?: boolean;
}

export function ClusterPlot({
  data,
  state,
  showVoronoi,
  showTrails,
  showLinks,
  showTruth,
  dragIndex,
  hoverIndex,
  sizes,
  cardIndex,
  onFrame,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  fill,
}: ClusterPlotProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, data.xRange, data.yRange, {
        left: 44,
        bottom: 32,
        right: 16,
        top: 14,
        equal: true,
      });
      onFrame?.(frame);

      const { centroids, assignments, phase } = state;
      const k = centroids.length;

      // 1. Voronoi territory.
      if (showVoronoi && k > 0) {
        drawCategoryField(ctx, frame, palette, (x, y) => nearestCentroid(centroids, x, y), {
          cellSize: 3,
          alpha: palette.isDark ? 0.24 : 0.17,
        });
      }

      drawAxes(ctx, frame, palette, {
        xLabel: data.xLabel,
        yLabel: data.yLabel,
        xTicks: 6,
        yTicks: 5,
        showGrid: !showVoronoi,
      });

      // 2. Assignment links, brighter during the assign phase.
      if (showLinks && phase !== 'seeded' && k > 0) {
        ctx.save();
        clipFrame(ctx, frame);
        ctx.lineWidth = 1;
        const strength = phase === 'assign' ? 0.55 : 0.22;
        for (let i = 0; i < data.points.length; i++) {
          const c = assignments[i];
          if (c === undefined || c < 0 || c >= k) continue;
          ctx.beginPath();
          ctx.moveTo(frame.x(data.points[i].x), frame.y(data.points[i].y));
          ctx.lineTo(frame.x(centroids[c].x), frame.y(centroids[c].y));
          ctx.strokeStyle = rgba(classColour(palette, c), strength);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 3. Where each centroid has been.
      if (showTrails) {
        ctx.save();
        clipFrame(ctx, frame);
        for (let c = 0; c < k; c++) {
          const trail = state.trails[c];
          if (!trail || trail.length < 2) continue;
          drawPath(
            ctx,
            trail.map((t) => ({ x: frame.x(t.x), y: frame.y(t.y) })),
            rgba(classColour(palette, c), 0.6),
            1.4,
            [4, 3],
          );
        }
        ctx.restore();
      }

      // 4. The true groups, on request.
      if (showTruth) {
        ctx.save();
        clipFrame(ctx, frame);
        ctx.lineWidth = 1.5;
        for (const p of data.points) {
          if (p.label < 0) continue;
          ctx.beginPath();
          ctx.arc(frame.x(p.x), frame.y(p.y), 6.5, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(classColour(palette, p.label), 0.7);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 5. The points, coloured and shaped by cluster.
      const display = data.points.map((p, i) => {
        const a = assignments[i];
        return { x: p.x, y: p.y, label: a === undefined || a < 0 ? 0 : a };
      });
      drawScatter(ctx, frame, palette, display, {
        radius: 3.7,
        shapes: true,
        colourOf: (_p, i) => {
          const a = assignments[i];
          return a === undefined || a < 0 ? palette.muted : classColour(palette, a);
        },
      });

      // 6. The update move: ghost of where the centroid was, arrow to where it is.
      if (phase === 'update') {
        for (let c = 0; c < k; c++) {
          const prev = state.previousCentroids[c];
          if (!prev) continue;
          const colour = classColour(palette, c);
          const x0 = frame.x(prev.x);
          const y0 = frame.y(prev.y);
          const x1 = frame.x(centroids[c].x);
          const y1 = frame.y(centroids[c].y);
          drawGhost(ctx, x0, y0, colour);
          if (Math.hypot(x1 - x0, y1 - y0) > 14) drawArrow(ctx, x0, y0, x1, y1, colour);
        }
      }

      // 7. The centroids themselves.
      for (let c = 0; c < k; c++) {
        const colour = classColour(palette, c);
        const cx = frame.x(centroids[c].x);
        const cy = frame.y(centroids[c].y);
        drawCentroid(ctx, cx, cy, colour, palette, dragIndex === c || hoverIndex === c);
        if (state.emptyClusters.includes(c)) {
          // Keep the flag inside the frame.
          drawLabelPlate(
            ctx,
            'empty',
            Math.min(cx + 14, frame.right - 44),
            Math.max(cy - 15, frame.top + 9),
            palette,
            { colour: palette.red, bold: true },
          );
        }
      }

      // 8. Which move produced this picture.
      drawLabelPlate(ctx, phaseSentence(state), frame.left + 6, frame.top + 11, palette, {
        bold: true,
        colour: palette.textMuted,
      });

      // 9. The hovered centroid's readout.
      if (hoverIndex !== null && dragIndex === null && hoverIndex !== cardIndex && centroids[hoverIndex]) {
        const c = centroids[hoverIndex];
        const previous = state.previousCentroids[hoverIndex];
        const moved = previous ? Math.hypot(c.x - previous.x, c.y - previous.y) : null;
        const text =
          'Centroid ' + (hoverIndex + 1) + ' · ' + (sizes?.[hoverIndex] ?? 0) + ' points' +
          (moved !== null && Number.isFinite(moved) ? ' · moved ' + fmt(moved, 2) : '') +
          ' · click for its numbers';
        const px = frame.x(c.x);
        const py = frame.y(c.y);
        // Centred under the centroid, slid just enough to keep the whole plate inside the plot.
        ctx.save();
        ctx.font = '600 11px ' + FONT_STACK;
        const half = ctx.measureText(text).width / 2 + 5;
        ctx.restore();
        const cx = Math.min(Math.max(px, frame.left + half), frame.right - half);
        drawLabelPlate(ctx, text, cx, Math.min(py + 24, frame.bottom - 10), palette, {
          align: 'center',
          colour: classColour(palette, hoverIndex),
          bold: true,
        });
      }
    },
    [data, state, showVoronoi, showTrails, showLinks, showTruth, dragIndex, hoverIndex, sizes, cardIndex, onFrame],
  );

  return (
    <Chart
      draw={draw}
      height={fill ? 'fill' : (w) => Math.round(Math.min(470, Math.max(300, w * 0.6)))}
      description={
        data.points.length +
        ' points in two dimensions with ' +
        state.centroids.length +
        ' centroids. ' +
        phaseSentence(state) +
        '. Inertia ' +
        state.inertia.toFixed(1) +
        ' after ' +
        state.iteration +
        ' full rounds.'
      }
      cursor={dragIndex !== null ? 'grabbing' : hoverIndex !== null ? 'grab' : 'crosshair'}
      drag={Boolean(onPointerDown)}
      onPointerDown={onPointerDown ? (pos) => onPointerDown(pos) : undefined}
      onPointerMove={onPointerMove ? (pos) => onPointerMove(pos) : undefined}
      onPointerUp={onPointerUp ? (pos) => onPointerUp(pos) : undefined}
      redrawKey={
        state.phase + ':' + state.iteration + ':' + state.inertia + ':' + dragIndex + ':' + hoverIndex + ':' + cardIndex
      }
    />
  );
}

function phaseSentence(state: KMeansState): string {
  if (state.phase === 'seeded') return 'Seeded, no point belongs to anything yet';
  if (state.phase === 'assign') {
    return state.changedCount === 0
      ? 'Assigned, nobody changed cluster, so this is the answer'
      : 'Assigned, ' + state.changedCount + ' point' + (state.changedCount === 1 ? '' : 's') + ' changed cluster';
  }
  // A hand-dragged centroid did not move to a mean.
  return Number.isFinite(state.shift)
    ? 'Moved, each centroid jumped to the mean of its own points'
    : 'Placed by hand, nothing has been reassigned yet';
}

/** The centroid marker: a ringed cross. */
function drawCentroid(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  colour: string,
  palette: Palette,
  active: boolean,
): void {
  const r = active ? 9.5 : 8;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r + 4, 0, Math.PI * 2);
  ctx.fillStyle = rgba(palette.surface, 0.82);
  ctx.fill();
  ctx.strokeStyle = rgba(colour, active ? 1 : 0.85);
  ctx.lineWidth = active ? 2.2 : 1.6;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();

  const arm = r * 0.62;
  ctx.strokeStyle = readableOn(colour);
  ctx.lineWidth = 1.8;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x - arm, y);
  ctx.lineTo(x + arm, y);
  ctx.moveTo(x, y - arm);
  ctx.lineTo(x, y + arm);
  ctx.stroke();
  ctx.restore();
}

/** Where the centroid stood before the update that just ran. */
function drawGhost(ctx: CanvasRenderingContext2D, x: number, y: number, colour: string): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.strokeStyle = rgba(colour, 0.55);
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.restore();
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  colour: string,
): void {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  // Stop short of the marker.
  const tipX = x1 - Math.cos(angle) * 12;
  const tipY = y1 - Math.sin(angle) * 12;
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();
  const head = 6.5;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - head * Math.cos(angle - 0.45), tipY - head * Math.sin(angle - 0.45));
  ctx.lineTo(tipX - head * Math.cos(angle + 0.45), tipY - head * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* ---------------- inertia over iterations ---------------- */

/** Inertia as a step chart: it holds for a round, then drops. */
export function InertiaCurve({
  history,
  current,
  phase,
  iteration,
}: {
  history: readonly number[];
  current: number;
  phase: KMeansState['phase'];
  iteration: number;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const values = history.filter((v) => Number.isFinite(v));
      const live = phase === 'assign' && Number.isFinite(current) ? current : null;
      const all = live !== null ? values.concat(live) : values.slice();
      if (all.length === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Take a step to record the first inertia.');
        return;
      }

      const [, hi] = extentOf(all);
      const yDomain: [number, number] = [0, hi > 0 ? hi * 1.12 : 1];
      const xMax = Math.max(1.7, values.length + 0.7, live !== null ? iteration + 1.2 : 0);
      const frame = makeFrame(width, height, [0.15, xMax], yDomain, {
        left: 52,
        bottom: 30,
        right: 16,
        top: 14,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'full rounds completed',
        yLabel: 'inertia',
        xTicks: Math.min(8, Math.max(2, Math.round(xMax))),
        yTicks: 4,
        formatX: (v) => (Number.isInteger(v) ? String(v) : ''),
      });

      ctx.save();
      clipFrame(ctx, frame);

      // step-after: hold the value across the round, then drop.
      const steps: Array<{ x: number; y: number }> = [];
      for (let i = 0; i < values.length; i++) {
        const py = frame.y(values[i]);
        steps.push({ x: frame.x(i + 0.5), y: py });
        steps.push({ x: frame.x(i + 1.5), y: py });
      }

      if (steps.length > 1) {
        ctx.beginPath();
        ctx.moveTo(steps[0].x, frame.bottom);
        for (const p of steps) ctx.lineTo(p.x, p.y);
        ctx.lineTo(steps[steps.length - 1].x, frame.bottom);
        ctx.closePath();
        ctx.fillStyle = rgba(palette.orange, palette.isDark ? 0.16 : 0.12);
        ctx.fill();
        drawPath(ctx, steps, palette.orange, 2.2);
      }

      for (let i = 0; i < values.length; i++) {
        drawPoint(ctx, frame.x(i + 1), frame.y(values[i]), 3.2, palette.orange, rgba(palette.surface, 0.9), 1.4);
      }

      if (live !== null) {
        const lx = frame.x(iteration + 0.5);
        const ly = frame.y(live);
        if (iteration > 0 && values.length >= iteration) {
          drawPath(
            ctx,
            [{ x: frame.x(iteration), y: frame.y(values[iteration - 1]) }, { x: lx, y: ly }],
            rgba(palette.cyan, 0.9),
            1.6,
            [3, 3],
          );
        }
        drawPoint(ctx, lx, ly, 3.4, palette.cyan, rgba(palette.surface, 0.9), 1.4);
      }
      ctx.restore();
    },
    [history, current, phase, iteration],
  );

  return (
    <Chart
      draw={draw}
      height={200}
      description={
        history.length === 0
          ? 'Inertia chart, empty until the first update.'
          : 'Inertia over ' +
            history.length +
            ' completed rounds, falling from ' +
            history[0].toFixed(1) +
            ' to ' +
            history[history.length - 1].toFixed(1) +
            '.'
      }
      redrawKey={history.length + ':' + phase + ':' + current}
    />
  );
}

/* ---------------- elbow + silhouette ---------------- */

export interface ElbowPoint {
  k: number;
  inertia: number;
  silhouette: number;
}

const SIL_LO = -0.15;
const SIL_HI = 1;

/** Inertia (left axis) and mean silhouette (right axis) against k. */
export function ElbowChart({
  curve,
  currentK,
  onSelectK,
}: {
  curve: readonly ElbowPoint[];
  currentK: number;
  onSelectK?: (k: number) => void;
}) {
  const frameRef = useRef<Frame | null>(null);
  const maxK = curve.length > 0 ? curve[curve.length - 1].k : 1;

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (curve.length === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Not enough points to sweep k.');
        return;
      }

      const maxInertia = Math.max(...curve.map((p) => p.inertia), 1e-6);
      const frame = makeFrame(width, height, [0.5, maxK + 0.5], [0, maxInertia * 1.08], {
        left: 52,
        bottom: 32,
        right: 46,
        top: 16,
      });
      frameRef.current = frame;

      drawAxes(ctx, frame, palette, {
        xLabel: 'k',
        yLabel: 'inertia',
        xTicks: maxK,
        yTicks: 4,
        formatX: (v) => (Number.isInteger(v) ? String(v) : ''),
        yLabelColour: palette.orange,
      });

      const silY = (s: number) =>
        frame.bottom - ((clamp(s, SIL_LO, SIL_HI) - SIL_LO) / (SIL_HI - SIL_LO)) * frame.innerHeight;

      // Right-hand axis for the silhouette series.
      ctx.save();
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      for (const s of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(s.toFixed(2), frame.right + 6, silY(s));
      ctx.translate(width - 8, (frame.top + frame.bottom) / 2);
      ctx.rotate(Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.font = '600 10px ' + FONT_STACK;
      ctx.fillStyle = palette.cyan;
      ctx.fillText('silhouette', 0, 0);
      ctx.restore();

      // The chosen k, behind the curves.
      const kx = frame.x(clamp(currentK, 1, maxK));
      drawPath(
        ctx,
        [{ x: kx, y: frame.top }, { x: kx, y: frame.bottom }],
        rgba(palette.text, 0.35),
        1.4,
        [4, 4],
      );

      ctx.save();
      clipFrame(ctx, frame);

      const silPoints = curve
        .filter((p) => Number.isFinite(p.silhouette))
        .map((p) => ({ x: frame.x(p.k), y: silY(p.silhouette) }));
      if (silPoints.length > 1) drawPath(ctx, silPoints, palette.cyan, 1.8, [5, 3]);
      for (const p of silPoints) {
        drawPoint(ctx, p.x, p.y, 2.8, palette.cyan, rgba(palette.surface, 0.9), 1.2);
      }

      drawPath(
        ctx,
        curve.map((p) => ({ x: frame.x(p.k), y: frame.y(p.inertia) })),
        palette.orange,
        2.2,
      );
      for (const p of curve) {
        const isCurrent = p.k === currentK;
        drawPoint(
          ctx,
          frame.x(p.k),
          frame.y(p.inertia),
          isCurrent ? 5.5 : 3.2,
          palette.orange,
          rgba(palette.surface, 0.95),
          isCurrent ? 2.2 : 1.4,
        );
      }
      ctx.restore();

      // The silhouette peak.
      const scored = curve.filter((p) => Number.isFinite(p.silhouette));
      if (scored.length > 0) {
        const best = scored.reduce((a, b) => (b.silhouette > a.silhouette ? b : a));
        drawLabelPlate(
          ctx,
          'best silhouette · k = ' + best.k,
          clamp(frame.x(best.k), frame.left + 4, frame.right - 4),
          Math.max(frame.top + 9, silY(best.silhouette) - 14),
          palette,
          { align: 'center', colour: palette.cyan, bold: true },
        );
      }
    },
    [curve, currentK, maxK],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      const frame = frameRef.current;
      if (!frame || !onSelectK) return;
      onSelectK(clamp(Math.round(frame.x.invert(pos.x)), 1, maxK));
    },
    [maxK, onSelectK],
  );

  return (
    <Chart
      draw={draw}
      height={236}
      description={
        curve.length === 0
          ? 'Elbow chart, unavailable for this data.'
          : 'Inertia and mean silhouette for k = 1 to ' +
            maxK +
            ', with k = ' +
            currentK +
            ' marked. Inertia falls from ' +
            curve[0].inertia.toFixed(0) +
            ' at k = 1 to ' +
            curve[curve.length - 1].inertia.toFixed(0) +
            ' at k = ' +
            maxK +
            '.'
      }
      cursor={onSelectK ? 'pointer' : 'default'}
      onPointerDown={onSelectK ? handleDown : undefined}
      redrawKey={curve.length + ':' + currentK + ':' + (curve[0]?.inertia ?? 0)}
    />
  );
}

/* ---------------- cluster sizes ---------------- */

export function ClusterSizeBars({
  sizes,
  emptyClusters,
  total,
}: {
  sizes: readonly number[];
  emptyClusters: readonly number[];
  total: number;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const k = sizes.length;
      // Before the first assignment every cluster is empty; that is not a failure.
      const assigned = sizes.reduce((a, b) => a + b, 0);
      if (k === 0 || total === 0 || assigned === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Take a step to assign the points.');
        return;
      }

      const left = 58;
      const right = width - 14;
      const top = 20;
      const bottom = height - 26;
      const innerW = right - left;
      const innerH = bottom - top;
      const maxSize = Math.max(1, ...sizes);
      const ceiling = maxSize * 1.2;
      const yOf = (v: number) => bottom - (v / ceiling) * innerH;
      const slot = innerW / k;
      const barW = Math.min(slot * 0.6, Math.max(48, Math.round(innerH * 0.18)));

      ctx.save();
      ctx.font = '10px ' + FONT_STACK;

      // Baseline and two reference ticks.
      ctx.strokeStyle = palette.border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, bottom + 0.5);
      ctx.lineTo(right, bottom + 0.5);
      ctx.stroke();
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText('0', left - 6, bottom);
      // Skip the tick where the even-split label would overprint it.
      const even = assigned / k;
      if (Math.abs(yOf(maxSize) - yOf(even)) > 11) ctx.fillText(String(maxSize), left - 6, yOf(maxSize));

      for (let i = 0; i < k; i++) {
        const cx = left + slot * (i + 0.5);
        const isEmpty = sizes[i] === 0 || emptyClusters.includes(i);

        if (isEmpty) {
          ctx.save();
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = palette.red;
          ctx.lineWidth = 1.4;
          ctx.strokeRect(cx - barW / 2, bottom - 12, barW, 12);
          ctx.restore();
          ctx.fillStyle = palette.red;
          ctx.font = '600 10px ' + FONT_STACK;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText('empty', cx, bottom - 18);
        } else {
          const y = yOf(sizes[i]);
          roundRect(ctx, cx - barW / 2, y, barW, Math.max(2, bottom - y), 4);
          ctx.fillStyle = rgba(classColour(palette, i), 0.85);
          ctx.fill();
          ctx.fillStyle = palette.text;
          ctx.font = '600 ' + (innerH > 300 ? 13 : 11) + 'px ' + FONT_STACK;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText(String(sizes[i]), cx, y - 5);
        }

        ctx.fillStyle = palette.textMuted;
        ctx.font = '10px ' + FONT_STACK;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText('C' + (i + 1), cx, bottom + 15);
      }

      // The even-split line, labelled at the axis.
      drawPath(
        ctx,
        [{ x: left, y: yOf(even) }, { x: right, y: yOf(even) }],
        rgba(palette.muted, 0.8),
        1.2,
        [4, 4],
      );
      ctx.fillStyle = palette.textMuted;
      ctx.font = '10px ' + FONT_STACK;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText('even ' + even.toFixed(0), left - 6, yOf(even));
      ctx.restore();
    },
    [sizes, emptyClusters, total],
  );

  return (
    <Chart
      draw={draw}
      height={190}
      description={
        sizes.length === 0
          ? 'Cluster size chart, empty until the points are assigned.'
          : 'Bar chart of points per cluster: ' +
            sizes.map((s, i) => 'cluster ' + (i + 1) + ' has ' + s).join(', ') +
            (emptyClusters.length > 0 ? '. ' + emptyClusters.length + ' cluster(s) are empty.' : '.')
      }
      redrawKey={sizes.join(',') + ':' + emptyClusters.join(',')}
    />
  );
}

/* ---------------- shared ---------------- */
