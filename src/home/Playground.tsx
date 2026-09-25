/** The hero's live canvas: click to add a point, Train to run the model. */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { Link } from 'react-router-dom';

import { useThemeVersion } from '../components/ThemeProvider';
import { IconArrow, IconPlay, IconStop } from '../explainer/components/Icons';
import { useElementSize } from '../explainer/useElementSize';
import { useMediaQuery } from '../explainer/useMediaQuery';
import { pointerPos, setupCanvas } from '../lib/viz/canvas';
import { readPalette } from '../lib/viz/palette';

import { DEMOS, getDemo } from './demos';
import type { DemoKey, DemoRun } from './demos';

const ASPECT = 0.64;
const MAX_INSTANT_TICKS = 2000;
// Keep painting this long after the model settles.
const TAIL_MS = 500;

function finish(run: DemoRun) {
  for (let i = 0; i < MAX_INSTANT_TICKS && !run.tick(i, true); i++) {
    /* run to completion */
  }
}

export function Playground() {
  const [mode, setMode] = useState<DemoKey>('boundary');
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('');
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
  const themeVersion = useThemeVersion();
  const [setHost, size, hostRef] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runRef = useRef<DemoRun | null>(null);
  const visibleRef = useRef(true);
  const demo = getDemo(mode);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const run = runRef.current;
    if (!canvas || !run || size.width <= 0) return;
    const setup = setupCanvas(canvas, size.width, Math.round(size.width * ASPECT));
    if (!setup) return;
    run.draw({ ctx: setup.ctx, width: setup.width, height: setup.height, palette: readPalette() });
  }, [size.width]);
  const paintRef = useRef(paint);
  paintRef.current = paint;

  useEffect(() => {
    const run = demo.create(1);
    finish(run);
    runRef.current = run;
    setRunning(false);
    setStatus(run.status());
    paintRef.current();
  }, [demo]);

  useEffect(() => {
    paint();
  }, [paint, themeVersion]);

  useEffect(() => {
    const el = hostRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const observer = new IntersectionObserver(([entry]) => {
      visibleRef.current = entry.isIntersecting;
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [hostRef]);

  // Frames only run while the reader has pressed Train.
  useEffect(() => {
    if (!running) return undefined;
    let id = 0;
    let doneAt = 0;
    let frames = 0;
    const loop = (now: number) => {
      id = requestAnimationFrame(loop);
      const run = runRef.current;
      if (!run || !visibleRef.current) return;
      const done = run.tick(now);
      if (!done) doneAt = 0;
      else if (!doneAt) doneAt = now;
      paint();
      frames += 1;
      if (done || frames % 6 === 0) setStatus(run.status());
      if (doneAt && now - doneAt > TAIL_MS) setRunning(false);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [running, paint]);

  const train = () => {
    const run = runRef.current;
    if (!run) return;
    // Once settled, Train replays from scratch.
    if (run.settled()) run.restart();
    if (reducedMotion) {
      finish(run);
      setStatus(run.status());
      paint();
      return;
    }
    setRunning(true);
  };

  const addPoint = (event: MouseEvent<HTMLCanvasElement>, label: number) => {
    const run = runRef.current;
    if (!run) return;
    const pos = pointerPos(event.currentTarget, event);
    run.addPoint(pos.x, pos.y, label);
    setStatus(run.status());
    paint();
  };

  return (
    <div className="mlx-play">
      <div className="mlx-play__bar">
        <div className="mlx-pills" role="group" aria-label="Choose a demo">
          {DEMOS.map((d) => (
            <button
              key={d.key}
              type="button"
              className="mlx-pill"
              aria-pressed={d.key === mode}
              onClick={() => setMode(d.key)}
            >
              {d.title}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="mlx-button mlx-button--secondary mlx-button--sm"
          onClick={running ? () => setRunning(false) : train}
        >
          {running ? <IconStop size={13} /> : <IconPlay size={13} />}
          {running ? 'Stop' : 'Train'}
        </button>
      </div>

      <div className="mlx-play__stage" ref={setHost}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={demo.title + '. ' + demo.hint}
          onClick={(event) => addPoint(event, 0)}
          onContextMenu={(event) => {
            event.preventDefault();
            addPoint(event, 1);
          }}
        />
      </div>

      <p className="mlx-play__hint">{demo.hint}</p>

      <div className="mlx-play__foot">
        <p className="mlx-caps mlx-play__status">{status}</p>
        <Link className="mlx-caps mlx-play__link" to={'/explainer/' + demo.slug}>
          Open the explainer
          <IconArrow size={12} />
        </Link>
      </div>
    </div>
  );
}
