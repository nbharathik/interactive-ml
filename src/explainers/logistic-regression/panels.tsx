/** Logistic regression charts: the field, the sigmoid, loss and accuracy, confusion, ROC. */

import { useCallback } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { MetricGrid } from '../../explainer/components/Panels';
import {
  FONT_STACK,
  clipFrame,
  drawAxes,
  drawLabelPlate,
  drawPoint,
  extentOf,
  makeFrame,
} from '../../lib/viz/canvas';
import {
  drawEmptyAxes,
  drawFunction,
  drawPath,
  drawScatter,
  drawSeries,
  drawSignedField,
  plateRoom,
} from '../../lib/viz/plots';
import { rgba } from '../../lib/viz/palette';
import type { PointData } from '../../lib/datasets/types';
import type { LogRegConfig, LogRegState, Standardiser } from '../../lib/ml/logisticRegression';
import { probability, score, sigmoid } from '../../lib/ml/logisticRegression';
import { fmt, fmtPercent } from '../../lib/math/stats';

/** Canvas pointer facts the decision field needs. A React pointer event supplies both. */
export interface PointerModifiers {
  shiftKey: boolean;
  button: number;
}

/* ---------------- decision field ---------------- */

export interface DecisionFieldProps {
  data: PointData;
  state: LogRegState;
  config: LogRegConfig;
  std: Standardiser;
  /** Probability assigned to each point, in the same order as `data.points`. */
  probabilities: readonly number[];
  /** Indices of the points currently on the wrong side of the threshold. */
  misclassified: readonly number[];
  showField: boolean;
  showMisclassified: boolean;
  hoverIndex: number | null;
  onFrame?: (frame: ReturnType<typeof makeFrame>) => void;
  onPointerDown?: (pos: { x: number; y: number }, event: PointerModifiers) => void;
  onPointerMove?: (pos: { x: number; y: number } | null) => void;
}

export function DecisionField({
  data,
  state,
  config,
  std,
  probabilities,
  misclassified,
  showField,
  showMisclassified,
  hoverIndex,
  onFrame,
  onPointerDown,
  onPointerMove,
}: DecisionFieldProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, data.xRange, data.yRange, {
        left: 44,
        bottom: 34,
        right: 16,
        top: 14,
      });
      onFrame?.(frame);

      const weights = state.weights;
      const probAt = (x: number, y: number) =>
        probability(weights, x, y, config.featureMap, std);

      if (!state.diverged) {
        if (showField) {
          // Shading is 2p - 1, transparent where the model is unsure.
          drawSignedField(ctx, frame, palette, (x, y) => 2 * probAt(x, y) - 1, {
            cellSize: 5,
            alpha: 0.5,
            contour: false,
          });
        }
        // A transparent pass whose only job is the contour at the threshold.
        drawSignedField(ctx, frame, palette, (x, y) => 2 * (probAt(x, y) - config.threshold), {
          cellSize: 5,
          alpha: 0,
          contour: true,
          contourColour: palette.text,
          contourWidth: 2,
        });
      }

      drawAxes(ctx, frame, palette, {
        xLabel: data.xLabel,
        yLabel: data.yLabel,
        xTicks: 6,
        yTicks: 6,
        showGrid: !showField || state.diverged,
      });

      drawScatter(ctx, frame, palette, data.points, {
        radius: 4.4,
        shapes: true,
        highlight: showMisclassified ? (misclassified as number[]) : undefined,
      });

      if (state.diverged) {
        ctx.save();
        ctx.font = '600 13px ' + FONT_STACK;
        ctx.fillStyle = palette.red;
        ctx.textAlign = 'center';
        ctx.fillText(
          'The weights overflowed, lower the learning rate and reset.',
          (frame.left + frame.right) / 2,
          frame.top + 24,
        );
        ctx.restore();
      }


      if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < data.points.length) {
        const point = data.points[hoverIndex];
        const px = frame.x(point.x);
        const py = frame.y(point.y);

        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 9.5, 0, Math.PI * 2);
        ctx.strokeStyle = rgba(palette.text, 0.8);
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.restore();

        const value = probabilities[hoverIndex] ?? probAt(point.x, point.y);
        const name = data.classNames[point.label] ?? 'class ' + point.label;
        const text = 'p = ' + fmt(value, 3) + '  ·  actually ' + name;
        const align: CanvasTextAlign = px > frame.right - 150 ? 'right' : 'left';
        drawLabelPlate(
          ctx,
          text,
          align === 'right' ? px - 13 : px + 13,
          Math.max(frame.top + 10, py - 14),
          palette,
          { align, bold: true },
        );
      }
    },
    [
      data,
      state,
      config.featureMap,
      config.threshold,
      std,
      probabilities,
      misclassified,
      showField,
      showMisclassified,
      hoverIndex,
      onFrame,
    ],
  );

  const wrong = misclassified.length;

  return (
    // Right-click adds a point of the other class.
    <div className="mlx-chart-host" onContextMenu={(event) => event.preventDefault()}>
      <Chart
        draw={draw}
        height={(w) => Math.round(Math.min(460, Math.max(280, w * 0.62)))}
        description={
          'Scatter plot of ' +
          data.points.length +
          ' points in two classes, shaded by the probability the model assigns each position. ' +
          wrong +
          ' points are on the wrong side of the p = ' +
          config.threshold.toFixed(2) +
          ' boundary after ' +
          state.epoch +
          ' training steps.'
        }
        cursor={onPointerDown ? 'crosshair' : 'default'}
        onPointerDown={onPointerDown ? (pos, event) => onPointerDown(pos, event) : undefined}
        onPointerMove={onPointerMove ? (pos) => onPointerMove(pos) : undefined}
        redrawKey={state.epoch + ':' + hoverIndex + ':' + data.points.length}
      />
    </div>
  );
}

/* ---------------- the sigmoid ---------------- */

/** The sigmoid with every point placed on it at its own score. */
export function SigmoidCurve({
  data,
  state,
  config,
  std,
  misclassified,
  showProjection,
}: {
  data: PointData;
  state: LogRegState;
  config: LogRegConfig;
  std: Standardiser;
  misclassified: readonly number[];
  showProjection: boolean;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const scores = data.points.map((p) =>
        score(state.weights, p.x, p.y, config.featureMap, std),
      );
      const [lo, hi] = extentOf(scores);
      const reach = Math.max(Math.abs(lo), Math.abs(hi));
      const limit = Math.min(60, Math.max(6, reach * 1.15));

      const frame = makeFrame(width, height, [-limit, limit], [-0.06, 1.06], {
        left: 44,
        bottom: 32,
        right: 16,
        top: 14,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'linear score  z',
        yLabel: 'p = σ(z)',
        xTicks: 6,
        yTicks: 5,
      });

      // The threshold, and the single z at which the sigmoid crosses it.
      const ty = frame.y(config.threshold);
      drawPath(
        ctx,
        [
          { x: frame.left, y: ty },
          { x: frame.right, y: ty },
        ],
        rgba(palette.text, 0.5),
        1.4,
        [4, 4],
      );
      const zt = Math.log(config.threshold / (1 - config.threshold));
      if (zt > -limit && zt < limit) {
        const tx = frame.x(zt);
        drawPath(
          ctx,
          [
            { x: tx, y: frame.top },
            { x: tx, y: frame.bottom },
          ],
          rgba(palette.text, 0.5),
          1.4,
          [4, 4],
        );
      }

      drawFunction(ctx, frame, (z) => sigmoid(z), palette.pink, { width: 2.6, samples: 260 });

      if (showProjection && !state.diverged) {
        // Each point drops to its label; the drop is |p - y|.
        ctx.save();
        clipFrame(ctx, frame);
        ctx.strokeStyle = rgba(palette.red, 0.3);
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < data.points.length; i++) {
          const px = frame.x(scores[i]);
          ctx.moveTo(px, frame.y(data.points[i].label));
          ctx.lineTo(px, frame.y(sigmoid(scores[i])));
        }
        ctx.stroke();
        ctx.restore();

        drawScatter(
          ctx,
          frame,
          palette,
          data.points.map((p, i) => ({ x: scores[i], y: sigmoid(scores[i]), label: p.label })),
          { radius: 3.8, shapes: true, highlight: misclassified as number[] },
        );
      }

      if (plateRoom(frame)) {
        drawLabelPlate(
          ctx,
          'threshold ' + config.threshold.toFixed(2),
          frame.right - 6,
          Math.max(frame.top + 10, ty - 11),
          palette,
          { align: 'right', colour: palette.textMuted },
        );
      }
    },
    [data, state, config.featureMap, config.threshold, std, misclassified, showProjection],
  );

  return (
    <Chart
      draw={draw}
      height={250}
      description={
        'The sigmoid curve from 0 to 1, with ' +
        data.points.length +
        ' points placed on it at their linear scores, and a dashed line marking the ' +
        config.threshold.toFixed(2) +
        ' threshold.'
      }
      redrawKey={state.epoch + ':' + config.threshold}
    />
  );
}

/* ---------------- loss and accuracy over time ---------------- */

export function TrainingCurves({
  lossHistory,
  accuracyHistory,
  epoch,
  baseline,
  penalised,
}: {
  lossHistory: readonly number[];
  accuracyHistory: readonly number[];
  /** Steps taken; the histories keep only their tail, which ends here. */
  epoch: number;
  /** Accuracy you get by always predicting the majority class. */
  baseline: number;
  /** The loss carries an L2 term. */
  penalised: boolean;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const yLabel = penalised ? 'log loss + λΣw²' : 'log loss';
      const clean = lossHistory.filter((v) => Number.isFinite(v));
      if (clean.length === 0) {
        drawEmptyAxes(ctx, width, height, palette, 'step', yLabel, { left: 46, bottom: 30, right: 50, top: 14 });
        return;
      }

      const [, hi] = extentOf(clean);
      // Leave room for the ln 2 line.
      const top = Math.max(hi * 1.1, 0.8);
      // The first value kept is the loss after this step.
      const x0 = Math.max(1, epoch - lossHistory.length + 1);
      const frame = makeFrame(
        width,
        height,
        [x0, Math.max(x0 + 1, epoch)],
        [0, top],
        { left: 46, bottom: 30, right: 50, top: 14 },
      );

      // Four x-ticks so long runs do not crowd the labels.
      drawAxes(ctx, frame, palette, {
        xLabel: 'step',
        yLabel,
        xTicks: 4,
        yTicks: 4,
        yLabelColour: palette.orange,
      });

      // A 50/50 guess scores exactly ln 2.
      const coinFlip = frame.y(Math.LN2);
      drawPath(
        ctx,
        [
          { x: frame.left, y: coinFlip },
          { x: frame.right, y: coinFlip },
        ],
        rgba(palette.muted, 0.75),
        1.3,
        [4, 4],
      );

      // Majority-class accuracy.
      const baseY = frame.y(baseline * top);
      drawPath(
        ctx,
        [
          { x: frame.left, y: baseY },
          { x: frame.right, y: baseY },
        ],
        rgba(palette.green, 0.45),
        1.3,
        [2, 4],
      );

      drawSeries(
        ctx,
        frame,
        palette,
        [
          { values: lossHistory, x0, colour: palette.orange, width: 2, fill: true, label: 'log loss' },
          {
            values: accuracyHistory.map((a) => a * top),
            x0,
            colour: palette.green,
            width: 1.8,
            label: 'accuracy',
          },
        ],
        { labelLast: true },
      );

      // Accuracy rides a second, invisible axis on the right.
      ctx.save();
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.green;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('100%', frame.right + 6, frame.y(top));
      ctx.fillText('50%', frame.right + 6, frame.y(top / 2));
      ctx.fillText('0%', frame.right + 6, frame.y(0));
      ctx.restore();

      if (plateRoom(frame)) {
        drawLabelPlate(ctx, 'ln 2, coin flip', frame.left + 8, coinFlip - 11, palette, {
          colour: palette.textMuted,
        });
      }
    },
    [lossHistory, accuracyHistory, epoch, baseline, penalised],
  );

  return (
    <Chart
      draw={draw}
      height={250}
      description={
        lossHistory.length === 0
          ? 'Training chart, empty until the first step is taken.'
          : 'Cross-entropy loss and accuracy over ' + epoch + ' training steps.'
      }
      redrawKey={epoch + ':' + lossHistory.length + ':' + baseline + ':' + penalised}
    />
  );
}

/* ---------------- ROC curve ---------------- */

export function RocPlot({
  curve,
  auc,
  operating,
  threshold,
}: {
  curve: ReadonlyArray<{ fpr: number; tpr: number }>;
  auc: number;
  operating: { fpr: number; tpr: number };
  threshold: number;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, [0, 1], [0, 1], {
        left: 44,
        bottom: 34,
        right: 16,
        top: 14,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'false positive rate',
        yLabel: 'true positive rate',
        xTicks: 5,
        yTicks: 5,
      });

      // Chance line.
      drawPath(
        ctx,
        [
          { x: frame.x(0), y: frame.y(0) },
          { x: frame.x(1), y: frame.y(1) },
        ],
        rgba(palette.muted, 0.8),
        1.3,
        [4, 4],
      );

      const points = curve.map((p) => ({ x: frame.x(p.fpr), y: frame.y(p.tpr) }));
      if (points.length > 1) {
        ctx.save();
        clipFrame(ctx, frame);
        ctx.beginPath();
        ctx.moveTo(points[0].x, frame.bottom);
        for (const p of points) ctx.lineTo(p.x, p.y);
        ctx.lineTo(points[points.length - 1].x, frame.bottom);
        ctx.closePath();
        ctx.fillStyle = rgba(palette.violet, palette.isDark ? 0.18 : 0.12);
        ctx.fill();
        ctx.restore();
        drawPath(ctx, points, palette.violet, 2.2);
      }

      const ox = frame.x(operating.fpr);
      const oy = frame.y(operating.tpr);
      drawPoint(ctx, ox, oy, 5, palette.orange, rgba(palette.surface, 0.95), 2);
      drawLabelPlate(
        ctx,
        't = ' + threshold.toFixed(2),
        ox + 10 > frame.right - 60 ? ox - 10 : ox + 10,
        Math.max(frame.top + 10, oy - 12),
        palette,
        { align: ox + 10 > frame.right - 60 ? 'right' : 'left', colour: palette.orange, bold: true },
      );

      drawLabelPlate(ctx, 'AUC ' + fmt(auc, 3), frame.right - 6, frame.bottom - 14, palette, {
        align: 'right',
        bold: true,
      });
    },
    [curve, auc, operating, threshold],
  );

  return (
    <Chart
      draw={draw}
      height={250}
      description={
        'ROC curve with area ' +
        fmt(auc, 3) +
        ', and the operating point at threshold ' +
        threshold.toFixed(2) +
        ' marked at a false positive rate of ' +
        fmtPercent(operating.fpr, 0) +
        ' and a true positive rate of ' +
        fmtPercent(operating.tpr, 0) +
        '.'
      }
      redrawKey={curve.length + ':' + threshold + ':' + auc}
    />
  );
}

/* ---------------- confusion matrix ---------------- */

export function ConfusionTable({
  counts,
  classNames,
}: {
  /** counts[actual][predicted] for the two classes. */
  counts: number[][];
  classNames: string[];
}) {
  const negative = classNames[0] ?? 'Negative';
  const positive = classNames[1] ?? 'Positive';
  const tn = counts[0]?.[0] ?? 0;
  const fp = counts[0]?.[1] ?? 0;
  const fn = counts[1]?.[0] ?? 0;
  const tp = counts[1]?.[1] ?? 0;

  return (
    <MetricGrid
      columns={2}
      metrics={[
        { key: 'tn', label: 'True ' + negative, value: String(tn), caption: 'called ' + negative + ', was ' + negative, tone: 'good' },
        { key: 'fp', label: 'False alarm', value: String(fp), caption: 'called ' + positive + ', was ' + negative, tone: fp > 0 ? 'bad' : 'neutral' },
        { key: 'fn', label: 'Missed', value: String(fn), caption: 'called ' + negative + ', was ' + positive, tone: fn > 0 ? 'bad' : 'neutral' },
        { key: 'tp', label: 'True ' + positive, value: String(tp), caption: 'called ' + positive + ', was ' + positive, tone: 'good' },
      ]}
    />
  );
}
