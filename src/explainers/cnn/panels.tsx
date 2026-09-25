/** CNN charts: the test gallery, the training curves, the confusion matrix, the kernels up close. */

import { useCallback, useMemo, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import type { ImageDataset } from '../../lib/datasets/images';
import { drawAxes, extentOf, makeFrame, padExtent, roundRect, MONO_STACK } from '../../lib/viz/canvas';
import { drawEmptyAxes, drawEmptyState, drawPixelGrid } from '../../lib/viz/plots';
import { fmt } from '../../lib/math/stats';
import { filterTensor } from '../../lib/ml/conv';
import type { CnnParams, CnnState } from '../../lib/ml/conv';
import { sub } from './scene';

/* ---------------- the probe ---------------- */

/** Steps through the held-out images in the architecture panel's head. */
export function ProbeStepper({ index, count, onChange }: { index: number; count: number; onChange: (next: number) => void }) {
  const last = Math.max(0, count - 1);
  return (
    <span className="mlx-stepper mlx-probe">
      <button type="button" className="mlx-stepper__button" disabled={index <= 0} onClick={() => onChange(index - 1)} aria-label="Previous held-out image" title="Previous held-out image">
        ‹
      </button>
      <span className="mlx-stepper__label">{'image ' + (index + 1) + ' / ' + count}</span>
      <button type="button" className="mlx-stepper__button" disabled={index >= last} onClick={() => onChange(index + 1)} aria-label="Next held-out image" title="Next held-out image">
        ›
      </button>
    </span>
  );
}

/* ---------------- the test gallery ---------------- */

export interface TestGalleryProps {
  data: ImageDataset;
  state: CnnState;
  /** Index into the test set. */
  probe: number;
  onProbe: (index: number) => void;
  /** Probability of the true class for a test image, computed on demand. */
  confidence: (index: number) => number;
  /** How many images to show. */
  limit?: number;
}

export function TestGallery({ data, state, probe, onProbe, confidence, limit = 40 }: TestGalleryProps) {
  const cellsRef = useRef<Array<{ x: number; y: number; w: number; h: number }>>([]);
  const [hover, setHover] = useState<number | null>(null);
  const shown = Math.min(limit, data.testIndex.length);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (shown === 0) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'No test images.');
        return;
      }
      const cols = Math.max(1, Math.min(shown, Math.floor(width / 54)));
      const rows = Math.ceil(shown / cols);
      const cell = Math.min(Math.floor((width - 8) / cols), Math.floor((height - 8) / Math.max(1, rows)), 72);
      const left = (width - cell * cols) / 2;
      const top = 4;
      const cells: Array<{ x: number; y: number; w: number; h: number }> = [];
      for (let i = 0; i < shown; i++) {
        const x = left + (i % cols) * cell;
        const y = top + Math.floor(i / cols) * cell;
        const inner = { x: x + 3, y: y + 3, width: cell - 6, height: cell - 6 };
        cells.push({ x: inner.x, y: inner.y, w: inner.width, h: inner.height });
        const imageIndex = data.testIndex[i];
        const predicted = state.testPredictions[i];
        const right = predicted === data.labels[imageIndex];
        ctx.fillStyle = palette.surfaceAlt;
        roundRect(ctx, inner.x, inner.y, inner.width, inner.height, 3);
        ctx.fill();
        drawPixelGrid(ctx, { x: inner.x + 2, y: inner.y + 2, width: inner.width - 4, height: inner.height - 4 }, data.images[imageIndex], data.size, data.size, palette, { ramp: 'magnitude', gap: 0, scale: 1 });
        if (state.step > 0 && predicted !== undefined) {
          roundRect(ctx, inner.x + 0.5, inner.y + 0.5, inner.width - 1, inner.height - 1, 3);
          ctx.strokeStyle = right ? palette.green : palette.orange;
          ctx.lineWidth = right ? 1.2 : 2;
          ctx.stroke();
          ctx.font = '600 11px ' + MONO_STACK;
          ctx.fillStyle = right ? palette.green : palette.orange;
          ctx.textAlign = 'right';
          ctx.textBaseline = 'top';
          ctx.fillText(data.classSymbols[predicted] ?? '?', inner.x + inner.width - 3, inner.y + 2);
        }
        if (i === probe) {
          strokeOutline(
            ctx,
            () => roundRect(ctx, inner.x - 1, inner.y - 1, inner.width + 2, inner.height + 2, 4),
            'open',
            palette,
          );
        } else if (i === hover) {
          strokeOutline(ctx, () => roundRect(ctx, inner.x - 1, inner.y - 1, inner.width + 2, inner.height + 2, 4), 'hover', palette);
        }
      }
      cellsRef.current = cells;
      if (hover !== null && hover < shown) {
        const c = cells[hover];
        const imageIndex = data.testIndex[hover];
        const lines = [data.classSymbols[data.labels[imageIndex]] + ' ' + data.classNames[data.labels[imageIndex]], state.step > 0 ? 'p ' + fmt(confidence(hover), 2) : 'untrained'];
        const size = measurePlate(ctx, lines);
        const at = placePlate(size, c, width, height);
        drawHoverPlate(ctx, lines, at.left, at.top, palette);
      }
    },
    [data, state, shown, probe, hover, confidence],
  );

  const locate = (pos: { x: number; y: number }) => {
    const cells = cellsRef.current;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      if (pos.x >= c.x && pos.x <= c.x + c.w && pos.y >= c.y && pos.y <= c.y + c.h) return i;
    }
    return null;
  };

  const rows = Math.ceil(shown / Math.max(1, Math.min(shown, 6)));
  return (
    <Chart
      draw={draw}
      height={Math.max(150, 54 * rows + 8)}
      description={shown + ' held-out glyphs, each ringed green when the network gets it right and orange when it does not; the probe is outlined.'}
      cursor={hover !== null ? 'pointer' : 'default'}
      onPointerDown={(pos) => {
        const found = locate(pos);
        if (found !== null) onProbe(found);
      }}
      onPointerMove={(pos) => setHover(pos ? locate(pos) : null)}
      onPointerLeave={() => setHover(null)}
      redrawKey={state.step + ':' + probe + ':' + hover + ':' + data.images.length}
    />
  );
}

/* ---------------- curves ---------------- */

export function TrainingCurves({ history, view }: { history: CnnState['history']; view: 'loss' | 'accuracy' }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (history.length < 2) {
        drawEmptyAxes(ctx, width, height, palette, 'batch', view === 'loss' ? 'log₁₀ loss' : 'accuracy');
        return;
      }
      const xs = history.map((h) => h.step);
      const train = history.map((h) => (view === 'loss' ? Math.log10(Math.max(h.train, 1e-6)) : h.trainAcc));
      const test = history.map((h) => (view === 'loss' ? Math.log10(Math.max(h.test, 1e-6)) : h.testAcc));
      const [lo, hi] = extentOf(train.concat(test));
      const yDomain: [number, number] = view === 'loss' ? padExtent(lo, hi, 0.12) : [0, 1.02];
      const frame = makeFrame(width, height, [0, xs[xs.length - 1] || 1], yDomain, { left: 54, bottom: 30, right: 14, top: 12 });
      drawAxes(ctx, frame, palette, { xLabel: 'batch', yLabel: view === 'loss' ? 'log₁₀ loss' : 'accuracy', yTicks: 4 });
      const toPoints = (ys: number[]) => xs.map((x, i) => ({ x: frame.x(x), y: frame.y(ys[i]) }));
      ctx.save();
      ctx.lineWidth = 2;
      for (const [ys, colour] of [
        [train, palette.blue],
        [test, palette.orange],
      ] as const) {
        const pts = toPoints(ys);
        ctx.strokeStyle = colour;
        ctx.beginPath();
        pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.stroke();
      }
      ctx.restore();
    },
    [history, view],
  );
  return (
    <Chart
      draw={draw}
      height={200}
      description={history.length < 2 ? 'Training curves, empty until training starts.' : (view === 'loss' ? 'Training and test loss' : 'Training and test accuracy') + ' over ' + history[history.length - 1].step + ' batches.'}
      redrawKey={history.length + ':' + view}
    />
  );
}

/* ---------------- confusion ---------------- */

export function ConfusionGrid({ data, predictions }: { data: ImageDataset; predictions: readonly number[] }) {
  const k = data.classNames.length;
  const counts = useMemo(() => {
    const table = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    predictions.forEach((p, i) => {
      const actual = data.labels[data.testIndex[i]];
      if (actual !== undefined && p !== undefined && p < k) table[actual][p]++;
    });
    return table;
  }, [data, predictions, k]);
  const total = predictions.length || 1;
  return (
    <div className="mlx-table__scroll">
      <table className="mlx-table mlx-table--compact mlx-confusion">
        <thead>
          <tr>
            <th scope="col">true ↓ called →</th>
            {data.classSymbols.map((s, c) => (
              <th key={c} scope="col">
                {s}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {counts.map((row, actual) => (
            <tr key={actual}>
              <th scope="row">{data.classSymbols[actual] + ' ' + data.classNames[actual]}</th>
              {row.map((n, called) => (
                <td key={called} className="mlx-num" data-tone={n === 0 ? undefined : called === actual ? 'good' : 'bad'} style={{ opacity: n === 0 ? 0.4 : 0.55 + (0.45 * n) / total }}>
                  {n}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- kernels up close ---------------- */

export function FilterGallery({ params }: { params: CnnParams }) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const layer = params.convs[0];
      if (!layer) {
        drawEmptyState(ctx, width, height, palette.textFaint, 'The dense model has no kernels.');
        return;
      }
      const n = layer.outC;
      const cols = Math.min(n, Math.max(1, Math.floor(width / 84)));
      const rows = Math.ceil(n / cols);
      const cell = Math.min(84, Math.floor(width / cols), Math.floor((height - 4) / rows));
      const left = (width - cell * cols) / 2;
      for (let i = 0; i < n; i++) {
        const kernel = filterTensor(layer, i);
        const x = left + (i % cols) * cell;
        const y = 2 + Math.floor(i / cols) * cell;
        drawPixelGrid(ctx, { x: x + 6, y: y + 4, width: cell - 12, height: cell - 24 }, kernel.data, layer.kernel, layer.kernel, palette, { ramp: 'signed', gap: 1 });
        ctx.font = '10px ' + MONO_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('k' + sub(i + 1), x + cell / 2, y + cell - 18);
      }
    },
    [params],
  );
  const n = params.convs[0]?.outC ?? 0;
  return (
    <Chart
      draw={draw}
      height={Math.max(120, 84 * Math.ceil(Math.max(1, n) / 4) + 8)}
      description={n ? 'The ' + n + ' first-layer kernels drawn large: blue cells add the pixel under them, orange cells subtract it.' : 'No kernels: the dense model reads pixels directly.'}
      redrawKey={params.convs[0]?.W.slice(0, 4).join(',') ?? 'dense'}
    />
  );
}
