/** Canvas host: sizing, DPR, theme redraws and accessibility. A visualization supplies `draw`. */

import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useRef } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';

import { setupCanvas, pointerPos } from '../../lib/viz/canvas';
import type { Palette } from '../../lib/viz/palette';
import { readPalette } from '../../lib/viz/palette';
import { useElementSize } from '../useElementSize';
import { useThemeVersion } from '../../components/ThemeProvider';

/** True inside a panel shown full screen, where every chart fills the space it is given. */
export const ChartFillContext = createContext(false);

export interface DrawArgs {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  palette: Palette;
}

export interface ChartProps {
  /** Everything the picture needs; called on every resize, theme flip and data change. */
  draw: (args: DrawArgs) => void;
  /** Height in CSS pixels, a function of the width, or 'fill' to take the parent's height. */
  height: number | 'fill' | ((width: number) => number);
  /** Alternative text describing what the chart currently shows. Required. */
  description: string;
  /** Pointer handlers receive canvas-local CSS pixel coordinates. */
  onPointerDown?: (pos: { x: number; y: number }, event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerMove?: (pos: { x: number; y: number } | null, event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerUp?: (pos: { x: number; y: number }, event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerLeave?: (event: ReactPointerEvent<HTMLCanvasElement>) => void;
  /** Cursor to show over the canvas, signals whether it is interactive. */
  cursor?: CSSProperties['cursor'];
  /** The chart drags things, so a touch on it must not scroll the page. */
  drag?: boolean;
  className?: string;
  /** Extra values that should force a redraw. */
  redrawKey?: unknown;
}

export function Chart({
  draw,
  height,
  description,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerLeave,
  cursor,
  drag,
  className,
  redrawKey,
}: ChartProps) {
  const [setContainer, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const themeVersion = useThemeVersion();

  const drawRef = useRef(draw);
  drawRef.current = draw;

  const inFullPanel = useContext(ChartFillContext);
  const fills = height === 'fill' || inFullPanel;
  const resolvedHeight = useMemo(() => {
    if (fills) return Math.round(size.height);
    if (typeof height === 'function') return size.width > 0 ? Math.round(height(size.width)) : 240;
    return height as number;
  }, [fills, height, size.width, size.height]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width <= 0 || resolvedHeight <= 0) return;
    const setup = setupCanvas(canvas, size.width, resolvedHeight);
    if (!setup) return;
    drawRef.current({
      ctx: setup.ctx,
      width: setup.width,
      height: setup.height,
      palette: readPalette(),
    });
  }, [resolvedHeight, size.width]);

  // Drawn before paint, so a chart never shows a blank frame.
  useLayoutEffect(() => {
    render();
  }, [render, themeVersion, redrawKey, draw]);

  const toLocal = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    return pointerPos(canvas, event);
  }, []);

  const interactive = Boolean(onPointerDown || onPointerMove || onPointerUp);

  return (
    <figure className={'mlx-chart' + (className ? ' ' + className : '')}>
      <div
        className={
          'mlx-chart__canvas' +
          (fills ? ' mlx-chart__canvas--fill' : '') +
          (drag ? ' mlx-chart__canvas--drag' : '')
        }
        ref={setContainer}
        style={fills ? undefined : { height: resolvedHeight }}
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={description}
          style={{ cursor: cursor ?? (interactive ? 'crosshair' : 'default') }}
          onPointerDown={
            onPointerDown
              ? (event) => {
                  // Capture so a drag that leaves the canvas keeps delivering moves.
                  event.currentTarget.setPointerCapture?.(event.pointerId);
                  onPointerDown(toLocal(event), event);
                }
              : undefined
          }
          onPointerMove={onPointerMove ? (event) => onPointerMove(toLocal(event), event) : undefined}
          onPointerUp={(event) => {
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
              event.currentTarget.releasePointerCapture?.(event.pointerId);
            }
            onPointerUp?.(toLocal(event), event);
          }}
          onPointerCancel={(event) => {
            if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
              event.currentTarget.releasePointerCapture?.(event.pointerId);
            }
            onPointerLeave?.(event);
          }}
          onPointerLeave={(event) => {
            // Clear hover state when the pointer leaves.
            onPointerMove?.(null, event);
            onPointerLeave?.(event);
          }}
        />
      </div>
    </figure>
  );
}
