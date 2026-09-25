/** Drawing pieces the transformer views share: head colours, vector strips, token chips, attention cells. */

import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { mix, rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';

/** One hue per head, clear of the orange and blue that mean negative and positive. */
export function headColour(palette: Palette, head: number): string {
  const ramp = [palette.violet, palette.green, palette.pink, palette.yellow];
  return ramp[head % ramp.length];
}

/** The largest magnitude in a list, for a shared colour scale. */
export function scaleOf(values: ArrayLike<number>): number {
  let top = 0;
  for (let i = 0; i < values.length; i++) if (Math.abs(values[i]) > top) top = Math.abs(values[i]);
  return top || 1;
}

/** A vector as a row of cells: blue for positive, orange for negative, depth for size. */
export function drawStrip(
  ctx: CanvasRenderingContext2D,
  values: ArrayLike<number>,
  x: number,
  y: number,
  w: number,
  h: number,
  palette: Palette,
  scale: number,
  options: { magnitude?: boolean; rows?: number; tint?: string } = {},
): void {
  const n = values.length;
  if (n === 0 || w <= 0 || h <= 0) return;
  const rows = Math.max(1, options.rows ?? 1);
  const cols = Math.ceil(n / rows);
  const cw = w / cols;
  const ch = h / rows;
  const gap = cw >= 5 && ch >= 5 ? 0.6 : 0;
  ctx.save();
  ctx.fillStyle = palette.surfaceAlt;
  ctx.fillRect(x, y, w, h);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const t = Math.min(1, Math.abs(v) / (scale || 1));
    if (!(t > 0.01)) continue;
    const colour = options.magnitude ? (options.tint ?? palette.text) : v < 0 ? palette.negative : palette.positive;
    ctx.fillStyle = mix(palette.surfaceAlt, colour, options.magnitude ? t * 0.9 : t);
    const r = Math.floor(i / cols);
    const c = i % cols;
    ctx.fillRect(x + c * cw, y + r * ch, cw - gap, ch - gap);
  }
  ctx.strokeStyle = rgba(palette.textFaint, 0.35);
  ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w) - 1, Math.round(h) - 1);
  ctx.restore();
}

export type ChipTone = 'plain' | 'start' | 'right' | 'wrong' | 'ghost' | 'accent';

/** A token in a rounded box. */
export function drawChip(ctx: CanvasRenderingContext2D, text: string, cx: number, cy: number, w: number, h: number, palette: Palette, tone: ChipTone = 'plain', size = 12): void {
  ctx.save();
  roundRect(ctx, cx - w / 2, cy - h / 2, w, h, Math.min(5, h / 3));
  const fill = tone === 'start' ? palette.surfaceAlt : tone === 'accent' ? rgba(palette.accent, 0.12) : tone === 'ghost' ? 'transparent' : palette.surface;
  ctx.fillStyle = fill;
  ctx.fill();
  const stroke =
    tone === 'right' ? palette.green : tone === 'wrong' ? palette.orange : tone === 'accent' ? palette.accent : tone === 'ghost' ? rgba(palette.textFaint, 0.5) : rgba(palette.textFaint, 0.7);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = tone === 'right' || tone === 'wrong' ? 1.6 : 1;
  if (tone === 'ghost') ctx.setLineDash([3, 2]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = tone === 'start' || tone === 'ghost' ? palette.textMuted : palette.text;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  let font = size;
  ctx.font = '600 ' + font + 'px ' + (text.length <= 2 ? MONO_STACK : FONT_STACK);
  while (font > 8 && ctx.measureText(text).width > w - 6) {
    font -= 1;
    ctx.font = '600 ' + font + 'px ' + (text.length <= 2 ? MONO_STACK : FONT_STACK);
  }
  ctx.fillText(text, cx, cy + 0.5);
  ctx.restore();
}

/** The colour of an attention weight: the head's hue, deeper for more. */
export function weightColour(palette: Palette, weight: number, hue: string): string {
  return mix(palette.surfaceAlt, hue, Math.min(1, Math.sqrt(Math.max(0, weight))));
}

/** A diagonal hatch for masked cells: a weight that cannot exist. */
export function hatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, palette: Palette): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.fillStyle = palette.surfaceAlt;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = rgba(palette.textFaint, 0.35);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = -h; s < w; s += 5) {
    ctx.moveTo(x + s, y + h);
    ctx.lineTo(x + s + h, y);
  }
  ctx.stroke();
  ctx.restore();
}

/** Small caps text for a gutter or a header. */
export function caption(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, palette: Palette, align: CanvasTextAlign = 'right', faint = false): void {
  ctx.save();
  ctx.font = '600 10px ' + FONT_STACK;
  ctx.fillStyle = faint ? palette.textFaint : palette.textMuted;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y);
  ctx.restore();
}
