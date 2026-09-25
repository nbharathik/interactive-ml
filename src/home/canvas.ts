/** Drawing helpers for the home page's demo and thumbnails. */

import type { Frame } from '../lib/viz/canvas';
import { makeFrame, ticks } from '../lib/viz/canvas';
import type { Palette } from '../lib/viz/palette';
import { rgba } from '../lib/viz/palette';

/** A frame with a thin margin and, for point data, one scale on both axes. */
export function plainFrame(
  width: number,
  height: number,
  xRange: [number, number],
  yRange: [number, number],
  equal = true,
): Frame {
  const m = 10;
  return makeFrame(width, height, xRange, yRange, { top: m, right: m, bottom: m, left: m, equal });
}

export function drawGrid(ctx: CanvasRenderingContext2D, frame: Frame, palette: Palette): void {
  ctx.save();
  ctx.strokeStyle = palette.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const t of ticks(frame.x.domain, 6)) {
    const px = Math.round(frame.x(t)) + 0.5;
    ctx.moveTo(px, frame.top);
    ctx.lineTo(px, frame.bottom);
  }
  for (const t of ticks(frame.y.domain, 5)) {
    const py = Math.round(frame.y(t)) + 0.5;
    ctx.moveTo(frame.left, py);
    ctx.lineTo(frame.right, py);
  }
  ctx.stroke();
  ctx.restore();
}

/** A cluster centre: a filled disc, a surface ring, and a small cross. */
export function drawCentroid(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  colour: string,
  palette: Palette,
  radius = 7,
): void {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = palette.surface;
  ctx.stroke();
  ctx.strokeStyle = rgba(palette.surface, 0.95);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(x - radius * 0.5, y);
  ctx.lineTo(x + radius * 0.5, y);
  ctx.moveTo(x, y - radius * 0.5);
  ctx.lineTo(x, y + radius * 0.5);
  ctx.stroke();
  ctx.restore();
}
