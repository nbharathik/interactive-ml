/** k-NN charts: the neighbourhood, the vote, accuracy against k, the neighbour list. */

import { useCallback, useMemo, useRef } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { useThemeVersion } from '../../components/ThemeProvider';
import {
  FONT_STACK,
  MONO_STACK,
  clipFrame,
  drawAxes,
  drawLabelPlate,
  drawPath,
  drawPoint,
  makeFrame,
  roundRect,
} from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import {
  classColour,
  classShape,
  drawCategoryField,
  drawEmptyState,
  drawGlow,
  drawScatter,
} from '../../lib/viz/plots';
import { rgba, readPalette } from '../../lib/viz/palette';
import { fmt, fmtPercent } from '../../lib/math/stats';
import type { PointData } from '../../lib/datasets/types';
import type { DistanceMetric, Neighbour, Scaling, Vote } from '../../lib/ml/knn';

/* ---------------- metric geometry ---------------- */

/** What the set of points at a fixed distance from the query looks like. */
export const NEIGHBOURHOOD_SHAPE: Record<DistanceMetric, string> = {
  euclidean: 'circle',
  manhattan: 'diamond',
  chebyshev: 'square',
  minkowski3: 'rounded square',
};

/** Length of a direction vector under the chosen metric, for drawing its unit ball. */
function metricNorm(x: number, y: number, metric: DistanceMetric): number {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  switch (metric) {
    case 'manhattan':
      return ax + ay;
    case 'chebyshev':
      return Math.max(ax, ay);
    case 'minkowski3':
      return Math.cbrt(ax ** 3 + ay ** 3);
    default:
      return Math.sqrt(ax * ax + ay * ay);
  }
}

/* ---------------- the neighbourhood ---------------- */

export interface NeighbourhoodPlotProps {
  /** Points in plot coordinates, already stretched, if the x unit was changed. */
  data: PointData;
  /** Query position in the same plot coordinates. */
  query: { x: number; y: number };
  /** Every stored point, ranked by distance. Only the first `shown` are voting. */
  neighbours: readonly Neighbour[];
  shown: number;
  /** Rank of the neighbour revealed by the last step, or -1 when at rest. */
  newest: number;
  vote: Vote;
  classNames: string[];
  metric: DistanceMetric;
  scaling: Scaling;
  weighted: boolean;
  /** What k-NN would predict at an arbitrary point, the decision regions. */
  classAt: (x: number, y: number) => number;
  /** Changes whenever the regions need recomputing; nothing else invalidates them. */
  fieldKey: string;
  showRegions: boolean;
  showLinks: boolean;
  showBall: boolean;
  dragging: boolean;
  /** Index of the stored point under the pointer, for its plate. */
  hoverIndex?: number | null;
  /** The pointer is over the query. */
  queryHover?: boolean;
  /** The query's card is open. */
  cardOpen?: boolean;
  onFrame?: (frame: Frame) => void;
  onPointerDown?: (pos: { x: number; y: number }, event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerMove?: (pos: { x: number; y: number } | null) => void;
  onPointerUp?: () => void;
  /** Take the container height instead of an intrinsic one (the studio hero). */
  fill?: boolean;
}

export function NeighbourhoodPlot({
  data,
  query,
  neighbours,
  shown,
  newest,
  vote,
  classNames,
  metric,
  scaling,
  weighted,
  classAt,
  fieldKey,
  showRegions,
  showLinks,
  showBall,
  dragging,
  hoverIndex,
  queryHover,
  cardOpen,
  onFrame,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  fill,
}: NeighbourhoodPlotProps) {
  // The regions never depend on the query, so they are cached offscreen while dragging.
  const fieldRef = useRef<{ key: string; canvas: HTMLCanvasElement } | null>(null);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      // A stretched x axis should look stretched; only near-square domains keep equal aspect.
      const xSpan = data.xRange[1] - data.xRange[0];
      const ySpan = data.yRange[1] - data.yRange[0];
      const frame = makeFrame(width, height, data.xRange, data.yRange, {
        left: 48,
        bottom: 34,
        right: 16,
        top: 16,
        equal: xSpan <= ySpan * 2 && ySpan <= xSpan * 2,
      });
      onFrame?.(frame);

      if (showRegions) {
        const w = Math.max(1, Math.round(frame.innerWidth));
        const h = Math.max(1, Math.round(frame.innerHeight));
        const key = fieldKey + '|' + w + 'x' + h + '|' + (palette.isDark ? 'dark' : 'light');
        if (!fieldRef.current || fieldRef.current.key !== key) {
          const offscreen = document.createElement('canvas');
          offscreen.width = w;
          offscreen.height = h;
          const offCtx = offscreen.getContext('2d');
          if (offCtx) {
            const offFrame = makeFrame(w, h, frame.x.domain, frame.y.domain, {
              left: 0,
              top: 0,
              right: 0,
              bottom: 0,
            });
            drawCategoryField(offCtx, offFrame, palette, classAt, { cellSize: 4, alpha: 0.3 });
            fieldRef.current = { key, canvas: offscreen };
          }
        }
        const cached = fieldRef.current;
        if (cached) {
          ctx.save();
          clipFrame(ctx, frame);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(cached.canvas, frame.left, frame.top, frame.innerWidth, frame.innerHeight);
          ctx.restore();
        }
      }

      drawAxes(ctx, frame, palette, {
        xLabel: data.xLabel,
        yLabel: data.yLabel,
        xTicks: 6,
        yTicks: 5,
        showGrid: !showRegions,
      });

      const qx = frame.x(query.x);
      const qy = frame.y(query.y);
      const counted = neighbours.slice(0, Math.max(0, shown));

      // Plates go on whichever side of the query has fewer ringed neighbours.
      const above = counted.filter((n) => frame.y(n.point.y) < qy).length;
      const flip = above > counted.length / 2;

      // The metric's unit ball, scaled to reach the last counted neighbour.
      if (showBall && counted.length > 0) {
        const radius = counted[counted.length - 1].distance;
        const outline: Array<{ x: number; y: number }> = [];
        const samples = 240;
        for (let i = 0; i <= samples; i++) {
          const angle = (i / samples) * Math.PI * 2;
          const ux = Math.cos(angle);
          const uy = Math.sin(angle);
          const norm = metricNorm(ux, uy, metric) || 1;
          outline.push({
            x: frame.x(query.x + (ux / norm) * radius * scaling.sx),
            y: frame.y(query.y + (uy / norm) * radius * scaling.sy),
          });
        }
        ctx.save();
        clipFrame(ctx, frame);
        drawPath(ctx, outline, rgba(palette.text, palette.isDark ? 0.6 : 0.5), 1.4, [5, 4]);
        ctx.restore();

        // Outside the outline, opposite the verdict plate, so the two never collide.
        if (!cardOpen) {
          const edgeY = flip
            ? Math.max(frame.top + 9, frame.y(query.y + radius * scaling.sy) - 12)
            : Math.min(frame.bottom - 9, frame.y(query.y - radius * scaling.sy) + 12);
          drawLabelPlate(
            ctx,
            NEIGHBOURHOOD_SHAPE[metric] + ', r = ' + fmt(radius, 2),
            qx,
            edgeY,
            palette,
            { align: 'center', colour: palette.textMuted },
          );
        }
      }

      // A line to every neighbour that is voting, thickened by its share.
      if (showLinks && counted.length > 0) {
        let maxWeight = 0;
        for (const n of counted) maxWeight = Math.max(maxWeight, n.weight);
        ctx.save();
        clipFrame(ctx, frame);
        for (const n of counted) {
          const share = maxWeight > 0 ? Math.sqrt(n.weight / maxWeight) : 1;
          drawPath(
            ctx,
            [
              { x: qx, y: qy },
              { x: frame.x(n.point.x), y: frame.y(n.point.y) },
            ],
            rgba(classColour(palette, n.point.label), 0.7),
            weighted ? 0.6 + 3.4 * share : 1.6,
          );
        }
        ctx.restore();
      }

      drawScatter(ctx, frame, palette, data.points, {
        radius: 4.2,
        shapes: true,
        highlight: counted.map((n) => n.index),
      });

      // Call out the neighbour that just joined.
      if (newest >= 0 && newest < counted.length) {
        const n = counted[newest];
        const colour = classColour(palette, n.point.label);
        const px = frame.x(n.point.x);
        const py = frame.y(n.point.y);
        ctx.save();
        clipFrame(ctx, frame);
        drawGlow(ctx, px, py, 24, colour);
        ctx.restore();
        // Label on the side away from the radius plate.
        const below = flip && py < qy;
        drawLabelPlate(
          ctx,
          '#' + (newest + 1) + ' ' + (classNames[n.point.label] ?? 'class ' + n.point.label),
          Math.min(frame.right - 4, px + 10),
          below ? Math.min(frame.bottom - 9, py + 16) : Math.max(frame.top + 9, py - 12),
          palette,
          { colour, bold: true },
        );
      }

      // The query itself.
      const winnerColour =
        counted.length === 0 ? palette.muted : classColour(palette, vote.winner);
      ctx.save();
      clipFrame(ctx, frame);
      drawGlow(ctx, qx, qy, 28, winnerColour);
      drawPoint(ctx, qx, qy, 7, winnerColour, rgba(palette.surface, 0.95), 2.5);
      ctx.beginPath();
      ctx.arc(qx, qy, 11, 0, Math.PI * 2);
      ctx.strokeStyle = rgba(palette.text, 0.45);
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();

      const verdict =
        counted.length === 0
          ? 'no stored points'
          : vote.tied
            ? 'tied, no winner'
            : 'predicts ' + (classNames[vote.winner] ?? 'class ' + vote.winner);
      if (!cardOpen) {
        drawLabelPlate(
          ctx,
          verdict,
          qx,
          flip ? Math.min(frame.bottom - 9, qy + 30) : Math.max(frame.top + 9, qy - 26),
          palette,
          { align: 'center', colour: vote.tied ? palette.orange : winnerColour, bold: true },
        );
      }

      // The hovered point's readout.
      if (hoverIndex !== null && hoverIndex !== undefined && !dragging && data.points[hoverIndex]) {
        const point = data.points[hoverIndex];
        const rank = neighbours.findIndex((n) => n.index === hoverIndex);
        const colour = classColour(palette, point.label);
        const text =
          (rank >= 0 ? '#' + (rank + 1) + ' · ' : '') +
          (classNames[point.label] ?? 'class ' + point.label) +
          (rank >= 0 ? ' · d = ' + fmt(neighbours[rank].distance, 2) : '') +
          (rank >= 0 && rank < shown ? ' · voting' : rank >= 0 ? ' · not counted' : '');
        const px = frame.x(point.x);
        const py = frame.y(point.y);
        drawLabelPlate(ctx, text, Math.min(frame.right - 4, px + 10), Math.min(frame.bottom - 9, py + 16), palette, {
          colour,
          bold: true,
        });
      }
    },
    [
      data,
      query,
      neighbours,
      shown,
      newest,
      vote,
      classNames,
      metric,
      scaling,
      weighted,
      classAt,
      fieldKey,
      showRegions,
      showLinks,
      showBall,
      dragging,
      hoverIndex,
      cardOpen,
      onFrame,
    ],
  );

  const description =
    'Scatter plot of ' +
    data.points.length +
    ' stored points in ' +
    data.classCount +
    ' classes, with a query point at x ' +
    fmt(query.x, 1) +
    ', y ' +
    fmt(query.y, 1) +
    '. ' +
    (shown > 0
      ? shown +
        ' nearest neighbours are counted, reaching a ' +
        NEIGHBOURHOOD_SHAPE[metric] +
        ' of radius ' +
        fmt(neighbours[Math.min(shown, neighbours.length) - 1]?.distance ?? 0, 2) +
        '. '
      : '') +
    (vote.tied
      ? 'The vote is tied.'
      : 'The predicted class is ' + (classNames[vote.winner] ?? 'unknown') + '.');

  return (
    <Chart
      draw={draw}
      height={fill ? 'fill' : (w) => Math.round(Math.min(460, Math.max(300, w * 0.62)))}
      description={description}
      cursor={
        dragging
          ? 'grabbing'
          : queryHover
            ? 'grab'
            : hoverIndex !== null && hoverIndex !== undefined
              ? 'pointer'
              : 'crosshair'
      }
      drag={Boolean(onPointerDown)}
      onPointerDown={onPointerDown ? (pos, event) => onPointerDown(pos, event) : undefined}
      onPointerMove={onPointerMove ? (pos) => onPointerMove(pos) : undefined}
      onPointerUp={onPointerUp ? () => onPointerUp() : undefined}
      redrawKey={fieldKey + '|' + shown + '|' + fmt(query.x, 3) + ',' + fmt(query.y, 3)}
    />
  );
}

/* ---------------- the vote ---------------- */

export interface VoteBarsProps {
  classNames: string[];
  /** Vote weight per class over the neighbours counted so far. */
  scores: readonly number[];
  /** Vote weight per class once all k neighbours are in, drawn as a ghost. */
  finalScores: readonly number[];
  winner: number;
  tied: boolean;
  weighted: boolean;
  shown: number;
}

export function VoteBars({
  classNames,
  scores,
  finalScores,
  winner,
  tied,
  weighted,
  shown,
}: VoteBarsProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const rows = scores.length;
      if (rows === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'No classes to count.');
        return;
      }

      const top = 22;
      const rowHeight = Math.min(Math.max(38, height * 0.16), (height - top - 10) / rows);
      const labelWidth = 76;
      const trackLeft = labelWidth + 14;
      const trackRight = width - 54;
      const trackWidth = Math.max(10, trackRight - trackLeft);
      const scale = Math.max(1e-9, ...finalScores, ...scores);

      ctx.save();
      ctx.font = '600 10px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(
        (weighted ? 'WEIGHTED VOTE' : 'VOTES') + ' AFTER ' + shown + (shown === 1 ? ' NEIGHBOUR' : ' NEIGHBOURS'),
        8,
        13,
      );
      ctx.restore();

      ctx.save();
      ctx.textBaseline = 'middle';

      for (let c = 0; c < rows; c++) {
        const cy = top + rowHeight * (c + 0.5);
        const colour = classColour(palette, c);
        const isWinner = !tied && c === winner && scores[c] > 0;

        ctx.font = (isWinner ? '650 ' : '') + '11px ' + FONT_STACK;
        ctx.fillStyle = isWinner ? palette.text : palette.textMuted;
        ctx.textAlign = 'right';
        ctx.fillText(truncate(ctx, classNames[c] ?? 'Class ' + c, labelWidth - 14), labelWidth, cy);

        // The winner gets a mark as well as a colour.
        if (isWinner) {
          ctx.beginPath();
          ctx.moveTo(labelWidth + 4, cy - 4);
          ctx.lineTo(labelWidth + 9, cy);
          ctx.lineTo(labelWidth + 4, cy + 4);
          ctx.closePath();
          ctx.fillStyle = palette.text;
          ctx.fill();
        }

        roundRect(ctx, trackLeft, cy - 9, trackWidth, 18, 4);
        ctx.fillStyle = rgba(palette.border, 0.5);
        ctx.fill();

        const finalWidth = (finalScores[c] / scale) * trackWidth;
        if (finalWidth > 1) {
          roundRect(ctx, trackLeft, cy - 9, Math.max(3, finalWidth), 18, 4);
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = rgba(colour, 0.55);
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.setLineDash([]);
        }

        const currentWidth = (scores[c] / scale) * trackWidth;
        if (currentWidth > 0.5) {
          roundRect(ctx, trackLeft, cy - 9, Math.max(3, currentWidth), 18, 4);
          ctx.fillStyle = rgba(colour, 0.85);
          ctx.fill();
        }

        ctx.font = '11px ' + MONO_STACK;
        ctx.fillStyle = palette.text;
        ctx.textAlign = 'left';
        ctx.fillText(weighted ? fmt(scores[c], 2) : String(Math.round(scores[c])), trackRight + 8, cy);
      }

      // Mark a tie.
      if (tied && scale > 0) {
        const best = Math.max(...scores);
        const x = trackLeft + (best / scale) * trackWidth;
        drawPath(
          ctx,
          [
            { x, y: top - 4 },
            { x, y: top + rowHeight * rows + 2 },
          ],
          palette.orange,
          1.6,
          [4, 3],
        );
        ctx.font = '650 10px ' + FONT_STACK;
        ctx.fillStyle = palette.orange;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText('TIE', width - 8, 13);
      }
      ctx.restore();
    },
    [classNames, scores, finalScores, winner, tied, weighted, shown],
  );

  return (
    <Chart
      draw={draw}
      height={Math.max(120, 34 + scores.length * 38)}
      description={
        'Bar chart of accumulated vote weight per class after ' +
        shown +
        ' neighbours. ' +
        scores
          .map((s, c) => (classNames[c] ?? 'Class ' + c) + ': ' + fmt(s, 2))
          .join(', ') +
        '. ' +
        (tied ? 'The leaders are tied.' : 'Leading class: ' + (classNames[winner] ?? 'none') + '.')
      }
      redrawKey={scores.join(',') + '|' + shown + '|' + String(tied)}
    />
  );
}

/* ---------------- accuracy against k ---------------- */

export interface AccuracyVsKProps {
  /** Leave-one-out accuracy for every k from 1 upwards. */
  curve: ReadonlyArray<{ k: number; accuracy: number }>;
  currentK: number;
  bestK: number;
  /** Accuracy from always predicting the most common class. */
  baseline: number;
}

export function AccuracyVsK({ curve, currentK, bestK, baseline }: AccuracyVsKProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (curve.length === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'Store at least two points to score any k.');
        return;
      }

      const lowest = Math.min(baseline, ...curve.map((c) => c.accuracy));
      const frame = makeFrame(
        width,
        height,
        [curve[0].k, Math.max(curve[curve.length - 1].k, curve[0].k + 1)],
        [Math.max(0, lowest - 0.06), 1.0],
        { left: 46, bottom: 32, right: 16, top: 14 },
      );

      drawAxes(ctx, frame, palette, {
        xLabel: 'k',
        yLabel: 'leave-one-out accuracy',
        yTicks: 4,
        xTicks: 6,
        formatY: (v) => Math.round(v * 100) + '%',
      });

      // Majority-class accuracy.
      const baseY = frame.y(baseline);
      if (baseY >= frame.top && baseY <= frame.bottom) {
        drawPath(
          ctx,
          [
            { x: frame.left, y: baseY },
            { x: frame.right, y: baseY },
          ],
          rgba(palette.muted, 0.85),
          1.3,
          [5, 4],
        );
        drawLabelPlate(ctx, 'always guess the biggest class', frame.right - 4, baseY - 11, palette, {
          align: 'right',
          colour: palette.textMuted,
        });
      }

      const points = curve.map((c) => ({ x: frame.x(c.k), y: frame.y(c.accuracy) }));
      ctx.save();
      clipFrame(ctx, frame);
      if (points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, frame.bottom);
        for (const p of points) ctx.lineTo(p.x, p.y);
        ctx.lineTo(points[points.length - 1].x, frame.bottom);
        ctx.closePath();
        ctx.fillStyle = rgba(palette.accent, palette.isDark ? 0.18 : 0.12);
        ctx.fill();
      }
      drawPath(ctx, points, palette.accent, 2.2);
      ctx.restore();

      const best = curve.find((c) => c.k === bestK);
      if (best) {
        const bestY = frame.y(best.accuracy);
        drawPoint(ctx, frame.x(best.k), bestY, 4.6, palette.green, rgba(palette.surface, 0.95), 2);
        // Below the point when it sits high, clear of the "k = n" plate.
        drawLabelPlate(
          ctx,
          'best k = ' + best.k + ' at ' + fmtPercent(best.accuracy, 0),
          frame.x(best.k),
          bestY < frame.top + 42 ? bestY + 17 : bestY - 14,
          palette,
          { align: 'center', colour: palette.green, bold: true },
        );
      }

      const markK = Math.min(Math.max(currentK, frame.x.domain[0]), frame.x.domain[1]);
      const cx = frame.x(markK);
      drawPath(
        ctx,
        [
          { x: cx, y: frame.top },
          { x: cx, y: frame.bottom },
        ],
        rgba(palette.orange, 0.9),
        1.5,
        [3, 3],
      );
      const current = curve.find((c) => c.k === currentK);
      if (current) {
        drawPoint(ctx, cx, frame.y(current.accuracy), 4.6, palette.orange, rgba(palette.surface, 0.95), 2);
      }
      drawLabelPlate(ctx, 'k = ' + currentK, cx, frame.top + 9, palette, {
        align: 'center',
        colour: palette.orange,
        bold: true,
      });
    },
    [curve, currentK, bestK, baseline],
  );

  return (
    <Chart
      draw={draw}
      height={230}
      description={
        curve.length === 0
          ? 'Accuracy against k, empty until there are points to score.'
          : 'Line chart of leave-one-out accuracy for k from 1 to ' +
            curve[curve.length - 1].k +
            '. It peaks at k = ' +
            bestK +
            '. The current k is ' +
            currentK +
            '.'
      }
      redrawKey={curve.length + '|' + currentK + '|' + bestK + '|' + fmt(baseline, 3)}
    />
  );
}

/* ---------------- the neighbour list ---------------- */

const SHAPE_GLYPH: Record<string, string> = {
  circle: '●',
  cross: '✕',
  triangle: '▲',
  square: '■',
};

export interface NeighbourTableProps {
  /** Ranked neighbours; only the first k are listed. */
  neighbours: readonly Neighbour[];
  k: number;
  shown: number;
  classNames: string[];
  weighted: boolean;
}

/** The neighbour list as text. */
export function NeighbourTable({ neighbours, k, shown, classNames, weighted }: NeighbourTableProps) {
  // themeVersion invalidates the cached palette.
  const themeVersion = useThemeVersion();
  const palette = useMemo(() => {
    void themeVersion;
    return readPalette();
  }, [themeVersion]);
  const rows = neighbours.slice(0, Math.max(1, k));

  if (rows.length === 0) {
    return <p className="mlx-note">There are no stored points, so there is nothing to rank.</p>;
  }

  return (
    <div className="mlx-table__scroll">
      <table className="mlx-table mlx-table--compact">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Class</th>
            <th scope="col">Distance</th>
            <th scope="col">Weight</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((n, index) => {
            const counted = index < shown;
            const label = n.point.label;
            return (
              <tr key={n.index} style={counted ? undefined : { opacity: 0.45 }}>
                <td className="mlx-num">{index + 1}</td>
                <td>
                  <span style={{ whiteSpace: 'nowrap' }}>
                    <span
                      aria-hidden="true"
                      style={{ color: classColour(palette, label), marginRight: 6 }}
                    >
                      {SHAPE_GLYPH[classShape(label)] ?? '●'}
                    </span>
                    {classNames[label] ?? 'Class ' + label}
                  </span>
                </td>
                <td className="mlx-num">{fmt(n.distance, 3)}</td>
                <td className="mlx-num">{weighted ? fmt(n.weight, 3) : '1'}</td>
                <td>
                  {counted ? (
                    <span className="mlx-badge mlx-badge--good">counted</span>
                  ) : (
                    <span className="mlx-badge">waiting</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- shared ---------------- */

/** Trim a label to fit, with an ellipsis. */
function truncate(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let clipped = text;
  while (clipped.length > 1 && ctx.measureText(clipped + '…').width > maxWidth) {
    clipped = clipped.slice(0, -1);
  }
  return clipped + '…';
}
