/** The attention view: every head's weights as a grid, rows looking at columns; a click opens that row inside its head. */

import { useCallback, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate, strokeOutline } from '../../explainer/diagramStyle';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt } from '../../lib/math/stats';
import type { TransformerCache, TransformerSpec } from '../../lib/ml/transformer';
import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { readableOn, rgba } from '../../lib/viz/palette';
import { hatch, headColour, weightColour } from './draw';
import { tokenName, weightAt } from './model';

/** Which row of which head is being looked at. */
export interface HeadPick {
  layer: number;
  head: number;
  query: number;
}

interface Box {
  layer: number;
  head: number;
  x: number;
  y: number;
  cell: number;
}

interface Hit {
  layer: number;
  head: number;
  i: number;
  j: number;
}

export interface AttentionGridProps {
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  /** The row the head view shows, outlined here too. */
  pick: HeadPick | null;
  onPick: (pick: HeadPick) => void;
  /** A lesson's spotlit row, drawn like a hover. */
  spot?: HeadPick | null;
  /** Write each weight in its cell; otherwise the colour carries it. */
  numbers?: boolean;
  description: string;
  redrawKey?: unknown;
}

const LABEL = 11;

function labelOf(data: SequenceDataset, token: number, room: number): string {
  const name = tokenName(data, token);
  return room < 34 && name.length > 3 ? name.slice(0, 3) : name;
}

export function AttentionGrid({ data, spec, cache, pick, onPick, spot = null, numbers = false, description, redrawKey }: AttentionGridProps) {
  const [hover, setHover] = useState<Hit | null>(null);
  const boxes = useRef<Box[]>([]);
  const T = cache.length;

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const words = data.task === 'animals';
      const side = words ? 50 : 22;
      const top = words ? 50 : 26;
      const gapX = 22;
      const gapY = 34;
      const footer = 22;
      const count = spec.heads * spec.layers;
      // Lay the grids out in whichever arrangement gives the largest cells.
      const cellFor = (c: number, r: number) =>
        Math.min(46, ((width - 16 - c * side - (c - 1) * gapX) / c) / T, ((height - footer - 12 - r * (top + 16) - (r - 1) * gapY) / r) / T);
      let cols = count;
      for (let c = 1; c <= count; c++) if (cellFor(c, Math.ceil(count / c)) > cellFor(cols, Math.ceil(count / cols))) cols = c;
      const rows = Math.ceil(count / cols);
      const cell = Math.max(8, cellFor(cols, rows));
      const blockW = side + cell * T;
      const blockH = top + 16 + cell * T;
      const x0 = Math.max(8, (width - (cols * blockW + (cols - 1) * gapX)) / 2);
      const y0 = Math.max(8, (height - footer - (rows * blockH + (rows - 1) * gapY)) / 2);
      const placed: Box[] = [];
      const shownRow = hover ? { layer: hover.layer, head: hover.head, query: hover.i } : spot;

      for (let l = 0; l < spec.layers; l++) {
        for (let h = 0; h < spec.heads; h++) {
          const k = l * spec.heads + h;
          const bx = x0 + (k % cols) * (blockW + gapX);
          const by = y0 + Math.floor(k / cols) * (blockH + gapY);
          const gx = bx + side;
          const gy = by + top + 16;
          placed.push({ layer: l, head: h, x: gx, y: gy, cell });
          const hue = headColour(palette, h);

          ctx.save();
          ctx.fillStyle = hue;
          ctx.beginPath();
          ctx.arc(gx + 5, by + 6, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.font = '700 10px ' + FONT_STACK;
          ctx.fillStyle = palette.textMuted;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(('head ' + (h + 1) + (spec.layers > 1 ? ' · layer ' + (l + 1) : '')).toUpperCase(), gx + 14, by + 6.5);
          ctx.restore();

          // Column labels: the keys, the tokens being looked at.
          ctx.save();
          ctx.font = (cell < 20 ? 9 : 10) + 'px ' + MONO_STACK;
          ctx.fillStyle = palette.textMuted;
          for (let j = 0; j < T; j++) {
            // Turned labels have the room for whole words.
            const text = words ? tokenName(data, cache.tokens[j]) : labelOf(data, cache.tokens[j], cell);
            const cx = gx + (j + 0.5) * cell;
            if (words) {
              ctx.save();
              ctx.translate(cx, gy - 4);
              ctx.rotate(-Math.PI / 4);
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              ctx.fillText(text, 0, 0);
              ctx.restore();
            } else {
              ctx.textAlign = 'center';
              ctx.textBaseline = 'bottom';
              ctx.fillText(text, cx, gy - 3);
            }
          }
          // Row labels: the queries, the tokens doing the looking.
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          for (let i = 0; i < T; i++) ctx.fillText(labelOf(data, cache.tokens[i], side), gx - 5, gy + (i + 0.5) * cell);
          ctx.restore();

          for (let i = 0; i < T; i++) {
            for (let j = 0; j < T; j++) {
              const x = gx + j * cell;
              const y = gy + i * cell;
              if (spec.causal && j > i) {
                hatch(ctx, x, y, cell - 1, cell - 1, palette);
                continue;
              }
              const w = weightAt(cache, l, h, i, j);
              const fill = weightColour(palette, w, hue);
              ctx.fillStyle = fill;
              ctx.fillRect(x, y, cell - 1, cell - 1);
              if (numbers && cell >= 20) {
                ctx.font = (cell >= 34 ? 11 : cell >= 26 ? 9 : 8) + 'px ' + MONO_STACK;
                ctx.fillStyle = readableOn(fill);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(w >= 0.995 ? '1' : w.toFixed(2).replace(/^0/, ''), x + cell / 2 - 0.5, y + cell / 2);
              }
            }
          }
          ctx.strokeStyle = rgba(palette.textFaint, 0.5);
          ctx.lineWidth = 1;
          ctx.strokeRect(gx - 0.5, gy - 0.5, cell * T, cell * T);

          const outlineRow = (query: number, state: 'hover' | 'open' | 'spot') =>
            strokeOutline(ctx, () => roundRect(ctx, gx - 2, gy + query * cell - 2, cell * T + 3, cell + 3, 3), state, palette);
          if (pick && pick.layer === l && pick.head === h) outlineRow(pick.query, 'open');
          if (shownRow && shownRow.layer === l && shownRow.head === h && !(pick && pick.layer === l && pick.head === h && pick.query === shownRow.query)) {
            outlineRow(shownRow.query, hover ? 'hover' : 'spot');
          }
        }
      }
      boxes.current = placed;

      if (hover) {
        const box = placed.find((b) => b.layer === hover.layer && b.head === hover.head);
        if (box) {
          const masked = spec.causal && hover.j > hover.i;
          const w = weightAt(cache, hover.layer, hover.head, hover.i, hover.j);
          const s = cache.layers[hover.layer].scores[(hover.head * T + hover.i) * T + hover.j];
          const lines = [
            tokenName(data, cache.tokens[hover.i]) + ' (' + hover.i + ') → ' + tokenName(data, cache.tokens[hover.j]) + ' (' + hover.j + ')',
            masked ? 'masked: in the future' : 'a = ' + fmt(w, 2) + ' · score ' + fmt(s, 2),
          ];
          const plate = measurePlate(ctx, lines);
          const at = placePlate(plate, { x: box.x + hover.j * box.cell, y: box.y + hover.i * box.cell, w: box.cell, h: box.cell }, width, height);
          drawHoverPlate(ctx, lines, at.left, at.top, palette);
        }
      }

      ctx.save();
      ctx.font = LABEL + 'px ' + FONT_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Each row is a token looking; each column a token looked at. A row sums to 1.' + (spec.causal ? ' Hatched: masked, in the future.' : ''), width / 2, height - 11, width - 16);
      ctx.restore();
    },
    [T, cache, data, hover, pick, spec, spot, numbers],
  );

  const locate = (pos: { x: number; y: number }): Hit | null => {
    for (const b of boxes.current) {
      const j = Math.floor((pos.x - b.x) / b.cell);
      const i = Math.floor((pos.y - b.y) / b.cell);
      if (i >= 0 && j >= 0 && i < T && j < T) return { layer: b.layer, head: b.head, i, j };
      // The row label picks the row too.
      if (pos.x < b.x && pos.x > b.x - 52 && i >= 0 && i < T) return { layer: b.layer, head: b.head, i, j: 0 };
    }
    return null;
  };

  return (
    <div className="mlx-arch">
      <div className="mlx-arch__stage">
        <Chart
          draw={draw}
          height="fill"
          description={description}
          cursor={hover ? 'pointer' : 'default'}
          onPointerMove={(pos) => {
            const found = pos ? locate(pos) : null;
            setHover((prev) => (prev && found && prev.layer === found.layer && prev.head === found.head && prev.i === found.i && prev.j === found.j ? prev : found));
          }}
          onPointerDown={(pos) => {
            const found = locate(pos);
            if (found) onPick({ layer: found.layer, head: found.head, query: found.i });
          }}
          onPointerLeave={() => setHover(null)}
          redrawKey={String(redrawKey) + '|' + (hover ? hover.layer + ':' + hover.head + ':' + hover.i + ':' + hover.j : '') + '|' + (pick ? pick.layer + ':' + pick.head + ':' + pick.query : '')}
        />
      </div>
    </div>
  );
}
