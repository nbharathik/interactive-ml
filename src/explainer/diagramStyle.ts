/** One hover language for every diagram: outline, wire emphasis, one readout line. */

import { FONT_STACK, MONO_STACK, roundRect } from '../lib/viz/canvas';
import { rgba } from '../lib/viz/palette';
import type { Palette } from '../lib/viz/palette';
import type { EdgePath, Rect } from './architecture';

/** How a wire is drawn relative to the focused component. */
export type WireEmphasis = 'plain' | 'lit' | 'hot' | 'dim';

/** Alpha multiplier for wires unrelated to the focused component. */
export const DIM = 0.3;

const PLATE_H = 20;
const PLATE_PAD = 7;
const PLATE_GAP = 7;

function plateFont(index: number): string {
  return index === 0 ? '600 11px ' + FONT_STACK : '500 11px ' + MONO_STACK;
}

/** Size of the one-line readout: the symbol, then the number it holds. */
export function measurePlate(
  ctx: CanvasRenderingContext2D,
  lines: readonly string[],
): { w: number; h: number } {
  let w = 0;
  lines.forEach((line, index) => {
    ctx.font = plateFont(index);
    w += ctx.measureText(line).width + (index > 0 ? PLATE_GAP : 0);
  });
  return { w: w + PLATE_PAD * 2, h: PLATE_H };
}

/** Where the readout goes: centred under the component, kept inside the frame. */
export function placePlate(
  size: { w: number; h: number },
  around: Rect,
  width: number,
  height: number,
): { left: number; top: number } {
  const cx = around.x + around.w / 2;
  return {
    left: Math.max(4, Math.min(width - size.w - 4, cx - size.w / 2)),
    top: Math.max(4, Math.min(height - size.h - 4, around.y + around.h + 4)),
  };
}

/** The spot under a wire's midpoint, for `placePlate`. */
export function underWire(mid: { x: number; y: number }): Rect {
  return { x: mid.x, y: mid.y + 2, w: 0, h: 0 };
}

/** The readout on one line: the symbol, then the number in the accent. */
export function drawHoverPlate(
  ctx: CanvasRenderingContext2D,
  lines: readonly string[],
  left: number,
  top: number,
  palette: Palette,
): void {
  if (lines.length === 0) return;
  ctx.save();
  const { w, h } = measurePlate(ctx, lines);
  roundRect(ctx, left, top, w, h, 4);
  ctx.fillStyle = rgba(palette.surface, 0.97);
  ctx.fill();
  ctx.strokeStyle = rgba(palette.accent, 0.6);
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  let x = left + PLATE_PAD;
  lines.forEach((line, index) => {
    ctx.font = plateFont(index);
    ctx.fillStyle = index === 0 ? palette.text : palette.accent;
    ctx.fillText(line, x, top + h / 2 + 0.5);
    x += ctx.measureText(line).width + PLATE_GAP;
  });
  ctx.restore();
}

/** A node's outline while hovered or open. A lesson spot adds a halo. */
export function strokeOutline(
  ctx: CanvasRenderingContext2D,
  outline: () => void,
  state: 'hover' | 'open' | 'spot',
  palette: Palette,
): void {
  ctx.save();
  ctx.setLineDash([]);
  if (state === 'spot') {
    outline();
    ctx.strokeStyle = rgba(palette.accent, 0.28);
    ctx.lineWidth = 12;
    ctx.stroke();
  }
  outline();
  ctx.strokeStyle = palette.accent;
  ctx.lineWidth = state === 'hover' ? 1.8 : 2.4;
  ctx.stroke();
  ctx.restore();
}

export interface WireStyle {
  colour: string;
  alpha: number;
  width: number;
  emphasis: WireEmphasis;
  dash?: number[];
  dashOffset?: number;
}

/** A wire in its own colour and weight. Hot adds a halo rather than a thicker stroke. */
export function strokeWire(
  ctx: CanvasRenderingContext2D,
  path: EdgePath,
  { colour, alpha, width, emphasis, dash, dashOffset }: WireStyle,
): void {
  const trace = () => {
    ctx.beginPath();
    ctx.moveTo(path.start.x, path.start.y);
    ctx.bezierCurveTo(path.c1.x, path.c1.y, path.c2.x, path.c2.y, path.end.x, path.end.y);
  };
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (emphasis === 'hot') {
    trace();
    ctx.strokeStyle = rgba(colour, 0.22);
    ctx.lineWidth = width + 7;
    ctx.stroke();
  }
  const shown =
    emphasis === 'dim' ? alpha * DIM : emphasis === 'plain' ? alpha : Math.min(1, alpha + 0.25);
  trace();
  ctx.strokeStyle = rgba(colour, shown);
  ctx.lineWidth = width;
  if (dash && emphasis !== 'dim') {
    ctx.setLineDash(dash);
    ctx.lineDashOffset = dashOffset ?? 0;
  }
  ctx.stroke();
  ctx.restore();
}
