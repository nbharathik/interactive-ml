/** The window view: one cell of a map, the patch it reads, the kernel between them, all drawn large. */

import { useCallback, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { strokeOutline } from '../../explainer/diagramStyle';
import { channel } from '../../lib/ml/conv';
import type { CnnCache, CnnParams, CnnSpec } from '../../lib/ml/conv';
import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { fmt } from '../../lib/math/stats';
import { rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawPixelGrid } from '../../lib/viz/plots';
import { brightest, imageFieldOf, sub, windowTerms } from './scene';
import type { Window } from './scene';
import type { Slide } from './useSlide';

export interface WindowViewProps {
  params: CnnParams;
  cache: CnnCache;
  spec: CnnSpec;
  window: Window | null;
  onWindow: (window: Window) => void;
  slide: Slide;
  description: string;
  redrawKey?: unknown;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Grid {
  /** Origin of the picture's first cell. */
  x: number;
  y: number;
  cell: number;
  cols: number;
  rows: number;
}

interface Hit {
  kind: 'output' | 'thumb' | 'channel';
  box: Box;
  grid?: Grid;
  index?: number;
}

/** Fit a cols by rows picture into a box, leaving `pad` cells of border on every side. */
function fitGrid(box: Box, cols: number, rows: number, pad: number): Grid {
  const cell = Math.min(box.w / (cols + 2 * pad), box.h / (rows + 2 * pad));
  const x = box.x + (box.w - cell * (cols + 2 * pad)) / 2 + pad * cell;
  const y = box.y + (box.h - cell * (rows + 2 * pad)) / 2 + pad * cell;
  return { x, y, cell, cols, rows };
}

function drawPicture(ctx: CanvasRenderingContext2D, grid: Grid, values: ArrayLike<number>, ramp: 'signed' | 'magnitude', palette: Palette, pad = 0): void {
  const { x, y, cell, cols, rows } = grid;
  if (pad > 0) {
    // The zero border same padding adds, as faint empty cells.
    ctx.save();
    ctx.strokeStyle = rgba(palette.textFaint, 0.35);
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1;
    ctx.strokeRect(x - pad * cell, y - pad * cell, (cols + 2 * pad) * cell, (rows + 2 * pad) * cell);
    ctx.restore();
  }
  ctx.fillStyle = palette.surfaceAlt;
  ctx.fillRect(x, y, cell * cols, cell * rows);
  drawPixelGrid(ctx, { x, y, width: cell * cols, height: cell * rows }, values, cols, rows, palette, { ramp, gap: cell >= 8 ? 1 : 0 });
  ctx.strokeStyle = rgba(palette.accent, 0.35);
  ctx.lineWidth = 1;
  ctx.strokeRect(x - 0.5, y - 0.5, cell * cols + 1, cell * rows + 1);
}

/** A block of cells outlined on a picture, kept inside it and its `pad` border. Returns the drawn rectangle. */
function outlineCells(ctx: CanvasRenderingContext2D, grid: Grid, x0: number, y0: number, w: number, h: number, palette: Palette, width = 2, pad = 0): Box {
  const cx0 = Math.max(-pad, x0);
  const cy0 = Math.max(-pad, y0);
  const cx1 = Math.min(grid.cols + pad, x0 + w);
  const cy1 = Math.min(grid.rows + pad, y0 + h);
  const box = { x: grid.x + cx0 * grid.cell, y: grid.y + cy0 * grid.cell, w: (cx1 - cx0) * grid.cell, h: (cy1 - cy0) * grid.cell };
  ctx.save();
  ctx.strokeStyle = palette.accent;
  ctx.lineWidth = width;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.restore();
  return box;
}

/** Small numbers in tinted cells. */
function drawNumbers(ctx: CanvasRenderingContext2D, x: number, y: number, cell: number, values: readonly number[], cols: number, signed: boolean, palette: Palette, scale?: number): void {
  let top = scale ?? 0;
  if (!top) for (const v of values) top = Math.max(top, Math.abs(v));
  if (!top) top = 1;
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = '500 ' + Math.max(8, Math.min(11, cell * 0.42)) + 'px ' + MONO_STACK;
  values.forEach((v, i) => {
    const cx = x + (i % cols) * cell;
    const cy = y + Math.floor(i / cols) * cell;
    const t = Math.min(1, Math.abs(v) / top);
    const colour = signed && v < 0 ? palette.negative : signed ? palette.positive : palette.text;
    ctx.fillStyle = rgba(colour, 0.08 + 0.5 * t);
    roundRect(ctx, cx + 1, cy + 1, cell - 2, cell - 2, 2);
    ctx.fill();
    if (cell >= 16) {
      ctx.fillStyle = palette.text;
      ctx.fillText(fmt(v, cell >= 36 ? 2 : 1), cx + cell / 2, cy + cell / 2 + 0.5);
    }
  });
  ctx.restore();
}

function caption(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, palette: Palette, align: CanvasTextAlign = 'center', strong = false): void {
  ctx.save();
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.font = (strong ? '700 10px ' + FONT_STACK : '10px ' + MONO_STACK);
  ctx.fillStyle = strong ? palette.textMuted : palette.textFaint;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function thumbRow(ctx: CanvasRenderingContext2D, pictures: Array<{ values: ArrayLike<number>; cols: number; rows: number; label: string }>, cx: number, y: number, current: number, palette: Palette): Box[] {
  const size = 30;
  const gap = 8;
  const total = pictures.length * size + (pictures.length - 1) * gap;
  const boxes: Box[] = [];
  pictures.forEach((p, i) => {
    const x = cx - total / 2 + i * (size + gap);
    const box = { x, y, w: size, h: size };
    boxes.push(box);
    ctx.fillStyle = palette.surfaceAlt;
    ctx.fillRect(x, y, size, size);
    drawPixelGrid(ctx, { x: x + 1, y: y + 1, width: size - 2, height: size - 2 }, p.values, p.cols, p.rows, palette, { ramp: 'magnitude', gap: 0 });
    ctx.strokeStyle = rgba(palette.accent, i === current ? 1 : 0.3);
    ctx.lineWidth = i === current ? 2 : 1;
    ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    caption(ctx, p.label, x + size / 2, y + size + 12, palette);
  });
  return boxes;
}

export function WindowView({ params, cache, spec, window, onWindow, slide, description, redrawKey }: WindowViewProps) {
  const hitsRef = useRef<Hit[]>([]);
  const [channelAt, setChannelAt] = useState(0);
  const [hover, setHover] = useState<Hit | null>(null);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const hits: Hit[] = [];
      hitsRef.current = hits;
      const layer = window ? params.convs[window.stage] : undefined;
      const stage = window ? cache.stages[window.stage] : undefined;
      if (!window || !layer || !stage) {
        caption(ctx, 'The dense model has no window: every unit reads every pixel.', width / 2, height / 2, palette);
        return;
      }
      const margin = 16;
      const k = layer.kernel;
      const pad = layer.pad;
      const inC = layer.inC;
      const ch = Math.min(channelAt, inC - 1);
      const prime = window.stage > 0 ? '′' : '';
      const name = (i: number) => (window.kind === 'pool' ? 'p' : 'm') + sub(i + 1) + prime;

      // Two pictures side by side, the numbers under them.
      const cell = Math.max(14, Math.min(36, (width - 300) / (3 * (window.kind === 'pool' ? 2 : k) + 4)));
      const mathH = Math.max(window.kind === 'pool' ? 2 * cell : k * cell, 66) + 26;
      const thumbsH = 54;
      const titleH = 44;
      const midGap = Math.max(56, width * 0.09);
      const side = Math.max(80, Math.min(height - titleH - thumbsH - mathH - 24, (width - 2 * margin - midGap) / 2));
      const total = titleH + side + thumbsH + mathH;
      const top = Math.max(titleH, (height - total) / 2 + titleH);
      const leftBox: Box = { x: margin + ((width - 2 * margin - midGap) / 2 - side) / 2, y: top, w: side, h: side };
      const rightBox: Box = { x: width - margin - ((width - 2 * margin - midGap) / 2 - side) / 2 - side, y: top, w: side, h: side };
      const mathY = top + side + thumbsH + 14;
      const centreX = width / 2;

      const cone = (from: Box, to: Box) => {
        ctx.save();
        ctx.strokeStyle = rgba(palette.accent, 0.5);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(from.x + from.w, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.moveTo(from.x + from.w, from.y + from.h);
        ctx.lineTo(to.x, to.y + to.h);
        ctx.stroke();
        ctx.restore();
      };
      const sumLines = (lines: string[], x: number, y: number) => {
        ctx.save();
        ctx.font = '500 12px ' + MONO_STACK;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        lines.forEach((line, i) => {
          ctx.fillStyle = i === lines.length - 1 ? palette.accent : palette.text;
          ctx.fillText(line, x, y + i * 18);
        });
        ctx.restore();
      };

      if (window.kind === 'map') {
        // Left: what the kernel reads, with the patch. Right: the map, with the pixel.
        const input = stage.input;
        const inputValues = channel(input, ch);
        const inGrid = fitGrid(leftBox, input.w, input.h, pad);
        drawPicture(ctx, inGrid, inputValues, 'magnitude', palette, pad);
        const outGrid = fitGrid(rightBox, stage.a.w, stage.a.h, 0);
        const map = channel(stage.a, window.index);
        drawPicture(ctx, outGrid, map, 'magnitude', palette);
        const patch = outlineCells(ctx, inGrid, window.x - pad, window.y - pad, k, k, palette, 2, pad);
        const pixel = outlineCells(ctx, outGrid, window.x, window.y, 1, 1, palette, 2);
        hits.push({ kind: 'output', box: rightBox, grid: outGrid });
        cone(patch, pixel);

        const inName = window.stage === 0 ? 'x' : 'p' + sub(ch + 1);
        caption(ctx, (window.stage === 0 ? 'IMAGE' : 'INPUT') + ' ' + inName + ' · ' + input.h + ' × ' + input.w, leftBox.x + side / 2, top - 22, palette, 'center', true);
        caption(ctx, 'receptive field · ' + k + ' × ' + k + ' patch at (' + (window.y - pad) + ', ' + (window.x - pad) + ')', leftBox.x + side / 2, top - 9, palette);
        caption(ctx, 'MAP ' + name(window.index) + ' · ' + stage.a.h + ' × ' + stage.a.w, rightBox.x + side / 2, top - 22, palette, 'center', true);
        caption(ctx, 'one pixel · (' + window.y + ', ' + window.x + ') · hover to move', rightBox.x + side / 2, top - 9, palette);

        // Under the input: the patch back on the image for a later layer, and the input channels.
        if (window.stage > 0) {
          const field = imageFieldOf(params, spec, window);
          const inset = 40;
          const ix = leftBox.x;
          const iy = leftBox.y + side + 8;
          const g = fitGrid({ x: ix, y: iy, w: inset, h: inset }, cache.input.w, cache.input.h, 0);
          drawPicture(ctx, g, cache.input.data, 'magnitude', palette);
          outlineCells(ctx, g, field.x0, field.y0, field.x1 - field.x0, field.y1 - field.y0, palette, 1.5);
          caption(ctx, 'image ' + (field.y1 - field.y0) + ' × ' + (field.x1 - field.x0), ix + inset / 2, iy + inset + 12, palette);
          if (inC > 1) {
            const thumbs = Array.from({ length: inC }, (_, i) => ({ values: channel(input, i), cols: input.w, rows: input.h, label: 'p' + sub(i + 1) }));
            const boxes = thumbRow(ctx, thumbs, leftBox.x + side / 2 + 36, leftBox.y + side + 8, ch, palette);
            boxes.forEach((box, i) => hits.push({ kind: 'channel', box, index: i }));
          }
        }
        const maps = Array.from({ length: layer.outC }, (_, i) => ({ values: channel(stage.a, i), cols: stage.a.w, rows: stage.a.h, label: 'm' + sub(i + 1) + prime }));
        const boxes = thumbRow(ctx, maps, rightBox.x + side / 2, rightBox.y + side + 8, window.index, palette);
        boxes.forEach((box, i) => hits.push({ kind: 'thumb', box, index: i }));

        // The numbers: patch, kernel, products, then the sum.
        const terms = windowTerms(params, cache, window, ch);
        if (terms) {
          const gridW = cell * k;
          const textW = 250;
          const blockW = 3 * gridW + 56 + 24 + textW;
          const gx = centreX - blockW / 2;
          const gy = mathY + 14;
          const pixels = terms.terms.map((t) => t.pixel);
          const weights = terms.terms.map((t) => t.weight);
          const products = terms.terms.map((t) => t.pixel * t.weight);
          drawNumbers(ctx, gx, gy, cell, pixels, k, false, palette, 1);
          drawNumbers(ctx, gx + gridW + 28, gy, cell, weights, k, true, palette);
          drawNumbers(ctx, gx + 2 * gridW + 56, gy, cell, products, k, true, palette);
          caption(ctx, 'patch', gx + gridW / 2, gy - 6, palette);
          caption(ctx, 'kernel k' + sub(window.index + 1) + prime, gx + gridW + 28 + gridW / 2, gy - 6, palette);
          caption(ctx, 'products', gx + 2 * gridW + 56 + gridW / 2, gy - 6, palette);
          ctx.save();
          ctx.font = '500 13px ' + MONO_STACK;
          ctx.fillStyle = palette.textMuted;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('⊙', gx + gridW + 14, gy + (k * cell) / 2);
          ctx.fillText('=', gx + 2 * gridW + 42, gy + (k * cell) / 2);
          ctx.restore();
          const sum = products.reduce((a, b) => a + b, 0);
          const lines: string[] = ['Σ products = ' + fmt(sum, 2)];
          if (inC > 1) lines.push('+ other slices ' + fmt(terms.others, 2));
          lines.push((terms.bias < 0 ? '− ' : '+ ') + 'bias ' + fmt(Math.abs(terms.bias), 2) + '  →  z = ' + fmt(terms.z, 2));
          lines.push('relu(z) = ' + fmt(terms.a, 2) + '  →  ' + name(window.index) + '[' + window.y + ', ' + window.x + ']');
          sumLines(lines, gx + 3 * gridW + 80, gy + Math.max(0, (k * cell - lines.length * 18) / 2));
        }
      } else {
        // Pooling: a 2 x 2 block of the map becomes one cell.
        const map = channel(stage.a, window.index);
        const pooled = channel(stage.pooled, window.index);
        const inGrid = fitGrid(leftBox, stage.a.w, stage.a.h, 0);
        drawPicture(ctx, inGrid, map, 'magnitude', palette);
        const outGrid = fitGrid(rightBox, stage.pooled.w, stage.pooled.h, 0);
        drawPicture(ctx, outGrid, pooled, 'magnitude', palette);
        const block = outlineCells(ctx, inGrid, 2 * window.x, 2 * window.y, 2, 2, palette, 2);
        const cellBox = outlineCells(ctx, outGrid, window.x, window.y, 1, 1, palette, 2);
        hits.push({ kind: 'output', box: rightBox, grid: outGrid });
        cone(block, cellBox);

        caption(ctx, 'MAP m' + sub(window.index + 1) + prime + ' · ' + stage.a.h + ' × ' + stage.a.w, leftBox.x + side / 2, top - 22, palette, 'center', true);
        caption(ctx, 'block 2 × 2 at (' + 2 * window.y + ', ' + 2 * window.x + ')', leftBox.x + side / 2, top - 9, palette);
        caption(ctx, 'POOLED ' + name(window.index) + ' · ' + stage.pooled.h + ' × ' + stage.pooled.w, rightBox.x + side / 2, top - 22, palette, 'center', true);
        caption(ctx, 'one cell · (' + window.y + ', ' + window.x + ') · hover to move', rightBox.x + side / 2, top - 9, palette);
        const pools = Array.from({ length: stage.pooled.c }, (_, i) => ({ values: channel(stage.pooled, i), cols: stage.pooled.w, rows: stage.pooled.h, label: 'p' + sub(i + 1) + prime }));
        const boxes = thumbRow(ctx, pools, rightBox.x + side / 2, rightBox.y + side + 8, window.index, palette);
        boxes.forEach((box, i) => hits.push({ kind: 'thumb', box, index: i }));

        const values: number[] = [];
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) values.push(map[(2 * window.y + dy) * stage.a.w + (2 * window.x + dx)] ?? 0);
        const result = spec.pool === 'max' ? Math.max(...values) : values.reduce((a, b) => a + b, 0) / 4;
        const gridW = 2 * cell;
        const blockW = gridW + 24 + 250;
        const gx = centreX - blockW / 2;
        const gy = mathY + 14;
        drawNumbers(ctx, gx, gy, cell, values, 2, false, palette);
        caption(ctx, '2 × 2 block', gx + gridW / 2, gy - 6, palette);
        sumLines(
          [(spec.pool === 'max' ? 'max' : 'mean') + ' of the four', '= ' + fmt(result, 2) + '  →  ' + name(window.index) + '[' + window.y + ', ' + window.x + ']'],
          gx + gridW + 24,
          gy + Math.max(0, (2 * cell - 36) / 2),
        );
      }

      if (hover && hover.kind !== 'output') strokeOutline(ctx, () => { ctx.beginPath(); ctx.rect(hover.box.x - 1, hover.box.y - 1, hover.box.w + 2, hover.box.h + 2); }, 'hover', palette);
    },
    [params, cache, spec, window, channelAt, hover],
  );

  const locate = (pos: { x: number; y: number }): Hit | null => {
    for (const hit of hitsRef.current) {
      const b = hit.box;
      if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) return hit;
    }
    return null;
  };

  const cellAt = (hit: Hit, pos: { x: number; y: number }) => {
    const g = hit.grid!;
    const x = Math.floor((pos.x - g.x) / g.cell);
    const y = Math.floor((pos.y - g.y) / g.cell);
    return x >= 0 && y >= 0 && x < g.cols && y < g.rows ? { x, y } : null;
  };

  const handleMove = (pos: { x: number; y: number } | null) => {
    const hit = pos ? locate(pos) : null;
    if (hit?.kind === 'output' && pos && window) {
      const at = cellAt(hit, pos);
      slide.setHovering(at !== null);
      if (at && (at.x !== window.x || at.y !== window.y)) onWindow({ ...window, x: at.x, y: at.y });
    } else slide.setHovering(false);
    setHover((prev) => (prev?.kind === hit?.kind && prev?.index === hit?.index ? prev : hit));
  };

  const handleDown = (pos: { x: number; y: number }) => {
    const hit = locate(pos);
    if (!hit || !window) return;
    if (hit.kind === 'channel' && hit.index !== undefined) setChannelAt(hit.index);
    if (hit.kind === 'thumb' && hit.index !== undefined && hit.index !== window.index) {
      const stage = cache.stages[window.stage];
      const t = window.kind === 'pool' ? stage.pooled : stage.a;
      const spot = brightest(channel(t, hit.index), t.w);
      onWindow({ ...window, index: hit.index, x: spot.x, y: spot.y });
    }
  };

  // Every layer the window can sit in.
  const layers = cache.stages.flatMap((_, s) => {
    const suffix = s > 0 ? ' ' + (s + 1) : '';
    const list: Array<{ kind: Window['kind']; stage: number; label: string }> = [{ kind: 'map', stage: s, label: 'Maps' + suffix }];
    if (spec.pool !== 'none') list.push({ kind: 'pool', stage: s, label: 'Pooled' + suffix });
    return list;
  });
  const goTo = (kind: Window['kind'], stage: number) => {
    const st = cache.stages[stage];
    if (!st) return;
    const t = kind === 'pool' ? st.pooled : st.a;
    const spot = brightest(channel(t, 0), t.w);
    onWindow({ kind, stage, index: 0, x: spot.x, y: spot.y });
  };

  return (
    <div className="mlx-arch">
      <div className="mlx-arch__stage">
        <Chart
          draw={draw}
          height="fill"
          description={description}
          cursor={hover ? (hover.kind === 'output' ? 'crosshair' : 'pointer') : 'default'}
          onPointerDown={handleDown}
          onPointerMove={handleMove}
          onPointerLeave={() => {
            slide.setHovering(false);
            setHover(null);
          }}
          redrawKey={String(redrawKey) + '|' + (window ? window.kind + window.stage + ':' + window.index + ':' + window.y + ',' + window.x : '') + '|' + channelAt + '|' + (hover ? hover.kind + hover.index : '')}
        />
        {window?.kind === 'map' ? (
          <button type="button" className="mlx-arch__tool mlx-arch__tool--wide mlx-arch__tool--foot" onClick={slide.toggle} title={slide.paused ? 'Slide the window again' : 'Hold the window still'}>
            {slide.paused ? '▶ slide' : '❚❚ hold'}
          </button>
        ) : null}
        {layers.length > 1 ? (
          <div className="mlx-segmented mlx-arch__tools" role="group" aria-label="Layer">
            {layers.map((l) => (
              <button
                key={l.kind + l.stage}
                type="button"
                className="mlx-segmented__item"
                data-active={(window?.kind === l.kind && window.stage === l.stage) || undefined}
                aria-pressed={window?.kind === l.kind && window.stage === l.stage}
                onClick={() => goTo(l.kind, l.stage)}
              >
                {l.label}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
