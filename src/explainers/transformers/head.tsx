/** Inside one head, for one query: its q against every key, the scaled scores, the softmax, and the weighted sum of the values. */

import { useCallback, useRef, useState } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, placePlate } from '../../explainer/diagramStyle';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt } from '../../lib/math/stats';
import { headWidth } from '../../lib/ml/transformer';
import type { TransformerCache, TransformerSpec } from '../../lib/ml/transformer';
import { FONT_STACK, MONO_STACK } from '../../lib/viz/canvas';
import { mix, rgba } from '../../lib/viz/palette';
import type { HeadPick } from './attention';
import { caption, drawChip, drawStrip, headColour, scaleOf } from './draw';
import { headRow, tokenName, weightAt } from './model';

export interface HeadTerms {
  dot: number;
  score: number;
  weight: number;
  masked: boolean;
}

/** Every term of one query row: the raw dot product, the scaled score and the weight, per key. */
export function headTerms(cache: TransformerCache, spec: TransformerSpec, pick: HeadPick): HeadTerms[] {
  const layer = cache.layers[pick.layer];
  if (!layer) return [];
  const T = cache.length;
  const q = headRow(layer.q, pick.query, pick.head, spec);
  const out: HeadTerms[] = [];
  for (let j = 0; j < T; j++) {
    const k = headRow(layer.k, j, pick.head, spec);
    let dot = 0;
    for (let c = 0; c < q.length; c++) dot += q[c] * k[c];
    const masked = spec.causal && j > pick.query;
    out.push({ dot, score: masked ? -Infinity : dot / Math.sqrt(q.length), weight: weightAt(cache, pick.layer, pick.head, pick.query, j), masked });
  }
  return out;
}

export interface HeadViewProps {
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  pick: HeadPick;
  onPick: (pick: HeadPick) => void;
  /** Write the scores and weights beside their bars. */
  numbers?: boolean;
  description: string;
  redrawKey?: unknown;
}

export function HeadView({ data, spec, cache, pick, onPick, numbers = false, description, redrawKey }: HeadViewProps) {
  const T = cache.length;
  const [hover, setHover] = useState<number | null>(null);
  const rowsRef = useRef<{ top: number; rowH: number } | null>(null);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const layer = cache.layers[pick.layer];
      if (!layer) return;
      const colour = headColour(palette, pick.head);
      const dh = headWidth(spec);
      const terms = headTerms(cache, spec, pick);
      const words = data.task === 'animals';

      // Column geometry: token | k | q·k | ÷√d | softmax | v | a·v.
      const chipW = words ? 56 : 34;
      const gap = 14;
      const fixed = chipW + 64 + 48 + 92 + gap * 6 + 16;
      const stripW = Math.max(40, Math.min(dh * 12, (width - fixed) / 3));
      const cols = {
        chip: 8 + chipW / 2,
        k: 8 + chipW + gap,
        dot: 8 + chipW + gap + stripW + gap,
        score: 8 + chipW + gap + stripW + gap + 64 + gap,
        weight: 8 + chipW + gap + stripW + gap + 64 + gap + 48 + gap,
        v: 8 + chipW + gap + stripW + gap + 64 + gap + 48 + gap + 92 + gap,
        av: 8 + chipW + gap + stripW * 2 + gap * 2 + 64 + gap + 48 + gap + 92 + gap,
      };
      const header = 92;
      const footer = 56;
      const rowH = Math.max(18, Math.min(42, (height - header - footer) / T));
      const top = header;
      rowsRef.current = { top, rowH };

      const q = headRow(layer.q, pick.query, pick.head, spec);
      const kScale = scaleOf(layer.k.subarray(0));
      const vScale = scaleOf(layer.v.subarray(0));

      // The query on top, then the column heads over the keys.
      drawChip(ctx, tokenName(data, cache.tokens[pick.query]), cols.chip, 20, chipW, 22, palette, 'accent');
      drawStrip(ctx, q, cols.k, 14, stripW, 12, palette, kScale);
      ctx.save();
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('the query q, from ' + tokenName(data, cache.tokens[pick.query]) + ' at position ' + pick.query + ': what it is looking for', cols.k + stripW + gap, 20, width - cols.k - stripW - gap - 8);
      const headY = top - 30;
      ctx.font = '600 10px ' + FONT_STACK;
      ctx.fillText('key k', cols.k, headY);
      ctx.fillText('q · k', cols.dot, headY);
      ctx.fillText('÷ √' + dh, cols.score, headY);
      ctx.fillText('softmax', cols.weight, headY);
      ctx.fillText('value v', cols.v, headY);
      ctx.fillText('a · v', cols.av, headY);
      ctx.textAlign = 'center';
      ctx.fillText('token', cols.chip, headY);
      ctx.textAlign = 'left';
      ctx.fillStyle = palette.textFaint;
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillText('what it offers', cols.k, headY + 14);
      ctx.fillText('match', cols.dot, headY + 14);
      ctx.fillText('scaled', cols.score, headY + 14);
      ctx.fillText('weight a', cols.weight, headY + 14);
      ctx.fillText('what it carries', cols.v, headY + 14);
      ctx.fillText('what it gives', cols.av, headY + 14);
      ctx.restore();

      let dotScale = 0;
      for (const t of terms) dotScale = Math.max(dotScale, Math.abs(t.dot));
      dotScale = dotScale || 1;

      for (let j = 0; j < T; j++) {
        const term = terms[j];
        const y = top + j * rowH;
        const cy = y + rowH / 2;
        const faded = term.masked;
        if (hover === j) {
          ctx.fillStyle = rgba(palette.accent, 0.07);
          ctx.fillRect(4, y, width - 8, rowH);
        }
        ctx.save();
        if (faded) ctx.globalAlpha = 0.35;
        drawChip(ctx, tokenName(data, cache.tokens[j]), cols.chip, cy, chipW, Math.min(22, rowH - 4), palette, j === 0 ? 'start' : 'plain', 11);
        drawStrip(ctx, headRow(layer.k, j, pick.head, spec), cols.k, cy - 5, stripW, 10, palette, kScale);
        ctx.restore();

        ctx.save();
        ctx.font = '11px ' + MONO_STACK;
        ctx.textBaseline = 'middle';
        if (faded) {
          ctx.fillStyle = palette.textFaint;
          ctx.textAlign = 'left';
          ctx.fillText('masked', cols.dot, cy);
          ctx.fillText('−∞', cols.score, cy);
          ctx.fillText('0', cols.weight, cy);
          ctx.restore();
          continue;
        }
        // q·k as a signed bar about a zero line.
        const mid = cols.dot + 32;
        const len = (Math.abs(term.dot) / dotScale) * 30;
        ctx.fillStyle = rgba(term.dot < 0 ? palette.negative : palette.positive, 0.75);
        ctx.fillRect(term.dot < 0 ? mid - len : mid, cy - 4, len, 8);
        ctx.fillStyle = palette.textFaint;
        ctx.fillRect(mid, cy - 7, 1, 14);
        if (numbers) {
          ctx.fillStyle = palette.textMuted;
          ctx.textAlign = 'left';
          ctx.fillText(fmt(term.score, 2), cols.score, cy);
        } else {
          // The scaled score as a cell: the same sign and size, smaller by √d.
          ctx.fillStyle = mix(palette.surfaceAlt, term.score < 0 ? palette.negative : palette.positive, Math.min(1, Math.abs(term.dot) / dotScale));
          ctx.fillRect(cols.score, cy - 7, 22, 14);
        }
        // The weight as a bar out of 1.
        ctx.fillStyle = palette.surfaceAlt;
        ctx.fillRect(cols.weight, cy - 5, 56, 10);
        ctx.fillStyle = colour;
        ctx.fillRect(cols.weight, cy - 5, 56 * term.weight, 10);
        if (numbers) {
          ctx.fillStyle = palette.text;
          ctx.fillText(term.weight >= 0.995 ? '1.00' : term.weight.toFixed(2), cols.weight + 62, cy);
        }
        ctx.restore();

        const v = headRow(layer.v, j, pick.head, spec);
        drawStrip(ctx, v, cols.v, cy - 5, stripW, 10, palette, vScale);
        const av = new Float64Array(v.length);
        for (let c = 0; c < v.length; c++) av[c] = term.weight * v[c];
        drawStrip(ctx, av, cols.av, cy - 5, stripW, 10, palette, vScale);
      }

      // The sum: what this head hands back for the query.
      const zy = top + T * rowH + 12;
      const z = headRow(layer.z, pick.query, pick.head, spec);
      ctx.save();
      ctx.strokeStyle = rgba(palette.textFaint, 0.6);
      ctx.beginPath();
      ctx.moveTo(cols.av, zy - 6);
      ctx.lineTo(cols.av + stripW, zy - 6);
      ctx.stroke();
      ctx.restore();
      caption(ctx, 'z = Σ a · v', cols.av - 8, zy + 7, palette, 'right');
      drawStrip(ctx, z, cols.av, zy + 1, stripW, 12, palette, vScale);
      ctx.save();
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('The weights sum to 1, so z is a blend of the values, most of it from the keys that matched best.', width / 2, height - 10, width - 16);
      ctx.restore();

      if (hover !== null && terms[hover] && !terms[hover].masked) {
        const t = terms[hover];
        const lines = [tokenName(data, cache.tokens[pick.query]) + ' · ' + tokenName(data, cache.tokens[hover]), 'q·k ' + fmt(t.dot, 2) + ' → ' + fmt(t.score, 2) + ' → a ' + fmt(t.weight, 2)];
        const plate = measurePlate(ctx, lines);
        const at = placePlate(plate, { x: cols.dot, y: top + hover * rowH, w: 200, h: rowH }, width, height);
        drawHoverPlate(ctx, lines, at.left, at.top, palette);
      }
    },
    [T, cache, data, hover, pick, spec, numbers],
  );

  const locate = (pos: { x: number; y: number }): number | null => {
    const r = rowsRef.current;
    if (!r) return null;
    const j = Math.floor((pos.y - r.top) / r.rowH);
    return j >= 0 && j < T ? j : null;
  };

  const layers = Array.from({ length: spec.layers }, (_, l) => l);
  const heads = Array.from({ length: spec.heads }, (_, h) => h);

  return (
    <div className="mlx-tf-head">
      <div className="mlx-tf-head__picks" role="group" aria-label="Which head and which query">
        {spec.layers > 1
          ? layers.map((l) => (
              <button key={'l' + l} type="button" className="mlx-scalar mlx-scalar--jump" data-tone={l === pick.layer ? 'accent' : undefined} aria-pressed={l === pick.layer} onClick={() => onPick({ ...pick, layer: l })}>
                <span className="mlx-scalar__label">{'layer ' + (l + 1)}</span>
              </button>
            ))
          : null}
        {heads.map((h) => (
          <button key={'h' + h} type="button" className="mlx-scalar mlx-scalar--jump mlx-tf-head__pick" data-tone={h === pick.head ? 'accent' : undefined} aria-pressed={h === pick.head} onClick={() => onPick({ ...pick, head: h })}>
            <span className="mlx-tf-dot" data-head={h} aria-hidden="true" />
            <span className="mlx-scalar__label">{'head ' + (h + 1)}</span>
          </button>
        ))}
        <span className="mlx-tf-head__sep" aria-hidden="true" />
        {cache.tokens.map((token, t) => (
          <button key={'q' + t} type="button" className="mlx-scalar mlx-scalar--jump" data-tone={t === pick.query ? 'accent' : undefined} aria-pressed={t === pick.query} title={'Query at position ' + t} onClick={() => onPick({ ...pick, query: t })}>
            <span className="mlx-scalar__label">{tokenName(data, token)}</span>
          </button>
        ))}
      </div>
      <div className="mlx-arch">
        <div className="mlx-arch__stage">
          <Chart
            draw={draw}
            height="fill"
            description={description}
            cursor="default"
            onPointerMove={(pos) => setHover(pos ? locate(pos) : null)}
            onPointerLeave={() => setHover(null)}
            redrawKey={String(redrawKey) + '|' + hover + '|' + pick.layer + ':' + pick.head + ':' + pick.query}
          />
        </div>
      </div>
    </div>
  );
}
