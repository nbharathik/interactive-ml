/** The shared training loop: play, pause, step, reset, speed. `create` and `step` must be pure. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export interface SimulationOptions<S> {
  /** Build iteration-zero state. Re-created whenever `deps` change. */
  create: () => S;
  /** Advance one logical step. Pure. */
  step: (state: S) => S;
  /** Stop automatically when this returns true. */
  isComplete?: (state: S) => boolean;
  /** Values that invalidate the run, changing any of them resets to step 0. */
  deps: readonly unknown[];
  /** Logical steps per second at speed 1×. Default 12. */
  baseStepsPerSecond?: number;
  /** Hard ceiling on steps. Default 100000. */
  maxSteps?: number;
  /** Begin playing as soon as the simulation is (re)created. */
  autoPlay?: boolean;
}

export type SpeedName = 'slow' | 'normal' | 'fast' | 'turbo';

/** Multipliers applied to `baseStepsPerSecond`. Turbo also batches per frame. */
export const SPEED_MULTIPLIERS: Record<SpeedName, number> = {
  slow: 0.35,
  normal: 1,
  fast: 4,
  turbo: 20,
};

export const SPEED_LABELS: Record<SpeedName, string> = {
  slow: 'Slow',
  normal: 'Normal',
  fast: 'Fast',
  turbo: 'Turbo',
};

/** Default step limit for an unbounded "run ahead" without animation. */
export const RUN_AHEAD_STEPS = 2000;

export interface SimulationApi<S> {
  state: S;
  /** How many logical steps have been taken since the last reset. */
  iteration: number;
  isRunning: boolean;
  isComplete: boolean;
  speed: SpeedName;
  play(): void;
  pause(): void;
  toggle(): void;
  /** Take exactly `count` steps while paused. */
  stepOnce(count?: number): void;
  /** Run without animation until complete or `limit` steps have passed. */
  runToCompletion(limit?: number): void;
  reset(): void;
  /** Reset and start again in one move, for a run that has finished. */
  replay(): void;
  setSpeed(speed: SpeedName): void;
  /** Replace the state directly, for user edits to the data mid-run. */
  patch(updater: (state: S) => S): void;
}

export const prefersReducedMotion = (): boolean => {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

export function useSimulation<S>(options: SimulationOptions<S>): SimulationApi<S> {
  const {
    create,
    step,
    isComplete: isCompleteFn,
    deps,
    baseStepsPerSecond = 12,
    maxSteps = 100_000,
    autoPlay = false,
  } = options;

  // Latest callbacks in refs so the animation loop never goes stale.
  const createRef = useRef(create);
  const stepRef = useRef(step);
  const completeRef = useRef(isCompleteFn);
  createRef.current = create;
  stepRef.current = step;
  completeRef.current = isCompleteFn;

  const [state, setState] = useState<S>(() => create());
  const [iteration, setIteration] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState<SpeedName>('normal');

  const stateRef = useRef(state);
  stateRef.current = state;
  const iterationRef = useRef(0);
  iterationRef.current = iteration;
  const runningRef = useRef(false);
  const speedRef = useRef<SpeedName>('normal');
  speedRef.current = speed;

  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);
  const accumulatorRef = useRef(0);

  // The predicate may read the page's config, so it is re-asked whenever either changes.
  const complete = useMemo(() => (isCompleteFn ? isCompleteFn(state) : false), [state, isCompleteFn]);
  const completeRefValue = useRef(complete);
  completeRefValue.current = complete;

  const stopLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    accumulatorRef.current = 0;
  }, []);

  /** Apply `count` steps synchronously, honouring completion and the step cap. */
  const advance = useCallback(
    (count: number) => {
      if (count <= 0) return;
      let next = stateRef.current;
      let taken = 0;
      for (let i = 0; i < count; i++) {
        if (iterationRef.current + taken >= maxSteps) break;
        if (completeRef.current && completeRef.current(next)) break;
        next = stepRef.current(next);
        taken += 1;
      }
      if (taken === 0) {
        // Complete or capped: stop.
        runningRef.current = false;
        setIsRunning(false);
        stopLoop();
        return;
      }
      stateRef.current = next;
      iterationRef.current += taken;
      setState(next);
      setIteration(iterationRef.current);
      if (completeRef.current && completeRef.current(next)) {
        runningRef.current = false;
        setIsRunning(false);
        stopLoop();
      }
    },
    [maxSteps, stopLoop],
  );

  const frame = useCallback(
    (time: number) => {
      if (!runningRef.current) return;
      const last = lastTimeRef.current || time;
      // Clamp so a backgrounded tab does not fast-forward on return.
      const delta = Math.min(120, time - last);
      lastTimeRef.current = time;

      const stepsPerSecond = baseStepsPerSecond * SPEED_MULTIPLIERS[speedRef.current];
      accumulatorRef.current += (delta / 1000) * stepsPerSecond;

      // Cap work per frame.
      const whole = Math.floor(accumulatorRef.current);
      if (whole >= 1) {
        accumulatorRef.current -= whole;
        advance(Math.min(whole, 400));
      }

      if (runningRef.current) rafRef.current = requestAnimationFrame(frame);
    },
    [advance, baseStepsPerSecond],
  );

  const play = useCallback(() => {
    if (runningRef.current) return;
    if (completeRefValue.current) return;
    runningRef.current = true;
    setIsRunning(true);
    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    rafRef.current = requestAnimationFrame((t) => {
      lastTimeRef.current = t;
      rafRef.current = requestAnimationFrame(frame);
    });
  }, [frame]);

  const pause = useCallback(() => {
    runningRef.current = false;
    setIsRunning(false);
    stopLoop();
  }, [stopLoop]);

  const toggle = useCallback(() => {
    if (runningRef.current) pause();
    else play();
  }, [pause, play]);

  const stepOnce = useCallback(
    (count = 1) => {
      pause();
      advance(count);
    },
    [advance, pause],
  );

  const runToCompletion = useCallback(
    (limit = RUN_AHEAD_STEPS) => {
      pause();
      advance(limit);
    },
    [advance, pause],
  );

  const reset = useCallback(() => {
    runningRef.current = false;
    setIsRunning(false);
    stopLoop();
    const fresh = createRef.current();
    stateRef.current = fresh;
    iterationRef.current = 0;
    setState(fresh);
    setIteration(0);
  }, [stopLoop]);

  // Not via play(): the completion memo still reads the finished state this tick.
  const replay = useCallback(() => {
    reset();
    runningRef.current = true;
    setIsRunning(true);
    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    rafRef.current = requestAnimationFrame(frame);
  }, [reset, frame]);

  const patch = useCallback((updater: (s: S) => S) => {
    const next = updater(stateRef.current);
    stateRef.current = next;
    setState(next);
  }, []);

  // Rebuild whenever the declared dependencies change.
  useEffect(() => {
    runningRef.current = false;
    setIsRunning(false);
    stopLoop();
    const fresh = createRef.current();
    stateRef.current = fresh;
    iterationRef.current = 0;
    setState(fresh);
    setIteration(0);
    if (autoPlay && !prefersReducedMotion()) {
      // Defer so the fresh state is committed before the loop reads it.
      const id = window.setTimeout(() => {
        runningRef.current = true;
        setIsRunning(true);
        lastTimeRef.current = 0;
        accumulatorRef.current = 0;
        rafRef.current = requestAnimationFrame(frame);
      }, 0);
      return () => window.clearTimeout(id);
    }
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  // Pause when the tab is hidden.
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && runningRef.current) pause();
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [pause]);

  // A play queued just before leaving must not start a loop on the unmounted page.
  useEffect(
    () => () => {
      runningRef.current = false;
      stopLoop();
    },
    [stopLoop],
  );

  return {
    state,
    iteration,
    isRunning,
    isComplete: complete,
    speed,
    play,
    pause,
    toggle,
    stepOnce,
    runToCompletion,
    reset,
    replay,
    setSpeed,
    patch,
  };
}
