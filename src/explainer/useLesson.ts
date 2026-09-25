/** Runs a page's lessons: the open step, its entry actions, its runs and the spotlight. */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

import type {
  FocusTarget,
  Lesson,
  LessonAction,
  LessonContextBase,
  LessonEvent,
  LessonSection,
  LessonState,
  LessonStep,
  LiveNumber,
  StepGauge,
} from './lessons';
import {
  FREE_PLAY,
  courseGauge,
  findLesson,
  focusKey,
  focusLabel,
  lessonQuery,
  lessonReducer,
  presetBefore,
  readLessonQuery,
  resolveSay,
  runList,
  stepAt,
} from './lessons';
import { readList, writeList } from './storage';
import { COMPACT_QUERY } from './useMediaQuery';
import type { ExplainerParamsApi, ParamsRecord } from './useExplainerParams';
import type { SimulationApi, SpeedName } from './useSimulation';
import { prefersReducedMotion } from './useSimulation';

const DONE_KEY = 'mlx-lesson-done:';
const SKIP_CHUNK = 25;
const SKIP_CAP = 20_000;

export interface LessonActions {
  /** Switch a named view, e.g. the second chart. */
  view?: (id: string, value: string) => void;
  /** Open a detail card by target id, or close it with null. */
  open?: (target: string | null) => void;
}

export interface LessonHost<P extends ParamsRecord, S, Ctx extends LessonContextBase> {
  /** The page, so its progress marks have their own key. */
  id: string;
  lessons: readonly Lesson<Ctx, P>[];
  params: ExplainerParamsApi<P>;
  sim: SimulationApi<S>;
  actions?: LessonActions;
  /** The page's own context fields; the hook adds `sim` and `lesson`. */
  context: Omit<Ctx, keyof LessonContextBase>;
}

export interface ExperimentView {
  label: string;
  /** The outcome, once tried: `then` with live numbers for a run that has met its condition. */
  say: string;
  done: boolean;
  /** The one whose outcome the panel is narrating. */
  active: boolean;
  /** A live run still playing towards its condition. */
  running: boolean;
}

/** A run in flight: armed once its setup has committed, met once its condition holds, stopped if the reader halts it first. */
interface RunState {
  index: number;
  armed: boolean;
  met: boolean;
  stopped: boolean;
  enteredAt: number;
  metAt: number | null;
  /** Reduced motion: step in chunks to the condition instead of playing. */
  skip: boolean;
}

/** One row of the outline. */
export interface OutlineItem {
  id: string;
  title: string;
  hook: string;
  section: LessonSection;
  done: boolean;
  current: boolean;
}

/** What the panel renders. Generic-free on purpose. */
export interface LessonView {
  active: boolean;
  lessonIndex: number;
  lessonCount: number;
  title: string;
  hook: string;
  section: LessonSection | null;
  outline: OutlineItem[];
  /** Every step of the course, for the card to measure its fixed height by. */
  course: StepGauge[];
  /** The first lesson not yet done, for the Start button. */
  resumeId: string | null;
  stepIndex: number;
  /** The narration. */
  text: string;
  /** The one idea to keep, or empty. */
  takeaway: string;
  experiments: ExperimentView[];
  /** What the step points at, for the "look here" pill. */
  focusLabel: string | null;
  /** The numbers the narration quotes, as chips. */
  numbers: LiveNumber[];
  /** The knobs the open lesson names, or null in free play. */
  knobs: string[] | null;
  /** The step wanted to play but could not (a deep link, reduced motion); offer a button. */
  needsRun: boolean;
  exit(): void;
  /** Open a lesson at a step; the outline calls this. */
  start(lessonId: string, stepIndex?: number): void;
  prevLesson(): void;
  nextLesson(): void;
  run(): void;
  tryExperiment(index: number): void;
}

export interface LessonApi<Ctx, P> {
  view: LessonView;
  lesson: Lesson<Ctx, P> | null;
  step: LessonStep<Ctx, P> | null;
  focus: FocusTarget | null;
  /** A lesson is open or one has been marked done: the chapter has something to reset. */
  dirty: boolean;
  /** Back to free play with every done mark cleared. */
  reset(): void;
}

/** The knobs a lesson touches: its runs' patches, its entry params and its control targets. */
function namedKnobs<Ctx, P>(lesson: Lesson<Ctx, P>): string[] | null {
  const keys = new Set<string>();
  const add = (action: LessonAction<P>) => {
    if (action.type === 'params') Object.keys(action.patch).forEach((k) => keys.add(k));
  };
  for (const step of lesson.steps) {
    if (step.focus?.kind === 'control') keys.add(step.focus.key);
    step.enter?.forEach(add);
    for (const e of step.experiments) {
      if (e.patch) Object.keys(e.patch).forEach((k) => keys.add(k));
      if (e.focus?.kind === 'control') keys.add(e.focus.key);
      e.enter?.forEach(add);
    }
  }
  return keys.size > 0 ? [...keys] : null;
}

export function useLesson<P extends ParamsRecord, S, Ctx extends LessonContextBase>(
  host: LessonHost<P, S, Ctx>,
): LessonApi<Ctx, P> {
  const { lessons, params, sim, actions, context } = host;

  const [state, dispatch] = useReducer(
    (prev: LessonState, event: LessonEvent) => lessonReducer(prev, event, lessons),
    undefined,
    (): LessonState => {
      // Free play unless the URL names a lesson.
      const position = readLessonQuery(params.extras, lessons);
      return position ? { ...FREE_PLAY, position, entry: 1 } : FREE_PLAY;
    },
  );

  const lesson = state.position ? findLesson(lessons, state.position.lessonId) ?? null : null;
  const step = stepAt(lessons, state.position) ?? null;
  const lessonIndex = lesson ? lessons.indexOf(lesson) : -1;

  const [needsRun, setNeedsRun] = useState(false);
  const [tried, setTried] = useState<number[]>([]);
  // The run last clicked; its outcome is the one the panel narrates.
  const [experiment, setExperiment] = useState<number | null>(null);
  const [focusOverride, setFocusOverride] = useState<FocusTarget | null>(null);
  const [live, setLive] = useState<RunState | null>(null);
  // A lesson is done once one of its runs has been tried on its last step.
  const doneKey = DONE_KEY + host.id;
  const [done, setDone] = useState<string[]>(() => readList(doneKey));
  useEffect(() => {
    if (!lesson || tried.length === 0 || !state.position || state.position.stepIndex < lesson.steps.length - 1) return;
    setDone((prev) => {
      if (prev.includes(lesson.id)) return prev;
      const next = [...prev, lesson.id];
      writeList(doneKey, next);
      return next;
    });
  }, [lesson, tried, state.position, doneKey]);

  // Effects read the latest apis through refs so they never re-run on a render.
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const simRef = useRef(sim);
  simRef.current = sim;
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  const savedSpeed = useRef<SpeedName | null>(null);
  // Play actions a step entry did not perform (a deep link, reduced motion); Run performs them.
  const pendingDrive = useRef<LessonAction<P>[]>([]);

  const ctx = useMemo(
    () =>
      ({
        ...context,
        sim: { iteration: sim.iteration, isRunning: sim.isRunning, isComplete: sim.isComplete },
        // A run in flight reads its own start and finish.
        lesson: live ? { metAt: live.metAt, enteredAt: live.enteredAt } : { metAt: null, enteredAt: state.enteredAt },
      }) as Ctx,
    [context, sim.iteration, sim.isRunning, sim.isComplete, state.enteredAt, live],
  );

  // The gauge reads live numbers once; the course itself never changes.
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;
  const course = useMemo(() => courseGauge(lessons, ctxRef.current), [lessons]);

  // Mirror the position into the URL through the params hook's single writer.
  useEffect(() => {
    paramsRef.current.setExtras(lessonQuery(state.position));
  }, [state.position]);

  const performDrive = useCallback((drive: readonly LessonAction<P>[]) => {
    for (const action of drive) {
      if (action.type === 'play') simRef.current.play();
      else if (action.type === 'step') simRef.current.stepOnce(action.count);
      else if (action.type === 'runTo') simRef.current.runToCompletion(action.steps);
    }
  }, []);

  // A drive waits for its setup to commit, so the page has rebuilt its data and model first.
  const commits = useRef(0);
  const awaiting = useRef<{ at: number; before: P | null; go: () => void } | null>(null);
  // Declared before the entry effect and after the simulation's, so it runs after the rebuild in the same commit.
  useEffect(() => {
    commits.current += 1;
    const pending = awaiting.current;
    if (!pending || commits.current <= pending.at) return;
    if (pending.before !== null && paramsRef.current.params === pending.before) return;
    awaiting.current = null;
    pending.go();
  });

  /** Perform a list of actions; driving the simulation waits for any setup to commit. */
  const runActions = useCallback(
    (
      list: readonly LessonAction<P>[] | undefined,
      mode: { skipSetup: boolean; skipPlay: boolean; onArmed?: () => void },
    ) => {
      if (!list || list.length === 0) {
        mode.onArmed?.();
        return () => {};
      }
      const before = paramsRef.current.params;
      let changed = false;
      let wait = false;
      const drive: LessonAction<P>[] = [];
      const skipped: LessonAction<P>[] = [];
      for (const action of list) {
        switch (action.type) {
          case 'preset':
            if (!mode.skipSetup) {
              paramsRef.current.applyPreset(action.id);
              changed = true;
            }
            break;
          case 'params':
            if (!mode.skipSetup) {
              paramsRef.current.merge(action.patch);
              changed = true;
            }
            break;
          case 'reset':
            if (!mode.skipSetup) {
              simRef.current.reset();
              wait = true;
            }
            break;
          case 'play':
          case 'step':
          case 'runTo':
            // A deep link never autoplays, but a runTo is instant: the picture opens solved.
            if (mode.skipPlay && action.type !== 'runTo') skipped.push(action);
            else drive.push(action);
            break;
          case 'pause':
            simRef.current.pause();
            break;
          case 'speed':
            if (savedSpeed.current === null) savedSpeed.current = simRef.current.speed;
            simRef.current.setSpeed(action.speed);
            break;
          case 'view':
            actionsRef.current?.view?.(action.id, action.value);
            break;
          case 'open':
            actionsRef.current?.open?.(action.target);
            break;
          default:
            break;
        }
      }
      if (skipped.length > 0) {
        pendingDrive.current = skipped;
        setNeedsRun(true);
      }
      const go = () => {
        mode.onArmed?.();
        if (drive.length === 0) return;
        if (drive.some((a) => a.type === 'play') && prefersReducedMotion()) {
          pendingDrive.current = drive;
          setNeedsRun(true);
          return;
        }
        performDrive(drive);
      };
      if (!changed && !wait) {
        go();
        return () => {};
      }
      const pending = { at: commits.current, before: changed ? before : null, go };
      awaiting.current = pending;
      return () => {
        if (awaiting.current === pending) awaiting.current = null;
      };
    },
    [performDrive],
  );

  // Step entry: close any card, run the step's actions, reset per-step state.
  const fromUrl = params.fromUrl;
  useEffect(() => {
    pendingDrive.current = [];
    setNeedsRun(false);
    setTried([]);
    setExperiment(null);
    setLive(null);
    setFocusOverride(null);
    actionsRef.current?.open?.(null);
    if (!step) {
      if (savedSpeed.current !== null) {
        simRef.current.setSpeed(savedSpeed.current);
        savedSpeed.current = null;
      }
      return undefined;
    }
    // A deep link keeps its params and never autoplays; a menu pick already applied its setup.
    const init = state.via === 'init';
    const picked = state.via === 'start';
    let list: readonly LessonAction<P>[] = step.enter ?? [];
    if (init && !fromUrl && lesson && state.position && !list.some((a) => a.type === 'preset')) {
      list = [{ type: 'preset', id: presetBefore(lesson, state.position.stepIndex) }, ...list];
    }
    return runActions(list, { skipSetup: (init && fromUrl) || picked, skipPlay: init });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.entry]);

  // A live run's condition, once its setup has committed; it pauses so its numbers hold still.
  useEffect(() => {
    if (!live || !live.armed || live.met || live.stopped || !step) return;
    const chosen = step.experiments[live.index];
    if (!chosen?.until) return;
    if (chosen.until(ctx)) {
      setLive({ ...live, met: true, metAt: sim.iteration });
      simRef.current.pause();
      return;
    }
    if (live.skip) {
      if (!sim.isComplete && sim.iteration - live.enteredAt < SKIP_CAP) simRef.current.stepOnce(SKIP_CHUNK);
      else setLive({ ...live, stopped: true });
      return;
    }
    // Paused, reset or finished short of its condition: the run is over.
    if (!sim.isRunning) setLive({ ...live, stopped: true });
  }, [ctx, live, step, sim.iteration, sim.isRunning, sim.isComplete]);

  const focus = focusOverride ?? step?.focus ?? null;

  // On narrow screens the target may be off-screen: bring it into view.
  useEffect(() => {
    if (!focus || typeof window === 'undefined' || !window.matchMedia(COMPACT_QUERY).matches) return undefined;
    const raf = window.requestAnimationFrame(() => {
      const selector =
        focus.kind === 'node' || focus.kind === 'edge'
          ? '.mlx-studio__arch'
          : '[data-lesson-target="' + focusKey(focus) + '"]';
      document
        .querySelector(selector)
        ?.scrollIntoView({ block: 'nearest', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    });
    return () => window.cancelAnimationFrame(raf);
  }, [focus]);

  // The setup runs in the click itself so the picture never lags the menu.
  const applySetup = useCallback(
    (lessonId: string, stepIndex: number) => {
      const target = findLesson(lessons, lessonId);
      const setup = target?.steps[stepIndex]?.enter ?? [];
      // A step entered directly starts from the last preset its lesson loaded before it.
      if (target && !setup.some((a) => a.type === 'preset')) paramsRef.current.applyPreset(presetBefore(target, stepIndex));
      for (const action of setup) {
        if (action.type === 'preset') paramsRef.current.applyPreset(action.id);
        else if (action.type === 'params') paramsRef.current.merge(action.patch);
        else if (action.type === 'reset') simRef.current.reset();
      }
      paramsRef.current.setExtras(lessonQuery({ lessonId, stepIndex }));
    },
    [lessons],
  );
  const start = useCallback(
    (lessonId: string, stepIndex = 0) => {
      applySetup(lessonId, stepIndex);
      dispatch({ type: 'start', lessonId, stepIndex, iteration: simRef.current.iteration });
    },
    [applySetup],
  );
  const run = useCallback(() => {
    setNeedsRun(false);
    const drive = pendingDrive.current.length > 0 ? pendingDrive.current : [{ type: 'play' as const }];
    pendingDrive.current = [];
    performDrive(drive);
  }, [performDrive]);
  const exit = useCallback(() => dispatch({ type: 'exit' }), []);
  const reset = useCallback(() => {
    dispatch({ type: 'exit' });
    setDone([]);
    writeList(doneKey, []);
  }, [doneKey]);
  const jump = useCallback(
    (delta: 1 | -1) => {
      const target = lessons[lessonIndex < 0 && delta < 0 ? 0 : lessonIndex + delta];
      if (!target) return;
      applySetup(target.id, 0);
      dispatch({ type: 'jump', delta, iteration: simRef.current.iteration });
    },
    [lessons, lessonIndex, applySetup],
  );
  const prevLesson = useCallback(() => jump(-1), [jump]);
  const nextLesson = useCallback(() => jump(1), [jump]);
  // A run restates its step's setup, applies its knobs, replays from the start, and narrates.
  const tryExperiment = useCallback(
    (index: number) => {
      if (!step || !lesson || !state.position) return;
      const chosen = step.experiments[index];
      if (!chosen) return;
      setExperiment(index);
      setTried((prev) => (prev.includes(index) ? prev : [...prev, index]));
      const list = runList(lesson, state.position.stepIndex, chosen);
      // A run without a speed of its own keeps the one its step set.
      const stepSpeed = step.enter?.filter((a) => a.type === 'speed').pop();
      if (chosen.speed) list.unshift({ type: 'speed', speed: chosen.speed });
      else if (stepSpeed) list.unshift(stepSpeed);
      else if (savedSpeed.current !== null) {
        simRef.current.setSpeed(savedSpeed.current);
        savedSpeed.current = null;
      }
      const isLive = Boolean(chosen.until);
      // Reduced motion: a live run steps to its condition without animating.
      const skip = isLive && prefersReducedMotion() && list.some((a) => a.type === 'play');
      const actions = skip ? list.filter((a) => a.type !== 'play') : list;
      setLive(isLive ? { index, armed: false, met: false, stopped: false, enteredAt: 0, metAt: null, skip } : null);
      runActions(actions, {
        skipSetup: false,
        skipPlay: false,
        // The start is read before the drive and outside the updater: an updater may run again after the drive.
        onArmed: isLive
          ? () => {
              const enteredAt = simRef.current.iteration;
              setLive((r) => (r && r.index === index ? { ...r, armed: true, enteredAt } : r));
            }
          : undefined,
      });
      if (chosen.focus) {
        setFocusOverride(chosen.focus);
        window.setTimeout(() => setFocusOverride((current) => (current === chosen.focus ? null : current)), 1800);
      }
    },
    [step, lesson, state.position, runActions],
  );

  const view = useMemo<LessonView>(() => {
    const experiments = step
      ? step.experiments.map((e, i) => {
          const running = live?.index === i && !live.met && !live.stopped;
          const met = live?.index === i && live.met;
          return {
            label: e.label,
            say: met ? resolveSay(e.then, ctx) || e.say : e.say,
            done: tried.includes(i),
            active: i === experiment,
            running,
          };
        })
      : [];
    let numbers: LiveNumber[] = [];
    if (step?.numbers) {
      try {
        numbers = step.numbers(ctx);
      } catch {
        numbers = [];
      }
    }
    const outline = lessons.map((l) => ({
      id: l.id,
      title: l.title,
      hook: l.hook,
      section: l.section,
      done: done.includes(l.id),
      current: l === lesson,
    }));
    return {
      active: step !== null,
      lessonIndex,
      lessonCount: lessons.length,
      title: lesson ? lesson.title : 'Free play',
      hook: lesson ? lesson.hook : '',
      section: lesson ? lesson.section : null,
      outline,
      course,
      resumeId: outline.find((l) => !l.done)?.id ?? lessons[0]?.id ?? null,
      stepIndex: state.position?.stepIndex ?? 0,
      text: step ? resolveSay(step.say, ctx) : '',
      takeaway: step ? resolveSay(step.takeaway, ctx) : '',
      experiments,
      focusLabel: focus ? focusLabel(focus) : null,
      numbers,
      knobs: lesson ? (lesson.knobs ?? namedKnobs(lesson)) : null,
      needsRun,
      exit,
      start,
      prevLesson,
      nextLesson,
      run,
      tryExperiment,
    };
  }, [step, lesson, lessons, lessonIndex, ctx, course, state.position, tried, experiment, live, done, focus, needsRun, exit, start, prevLesson, nextLesson, run, tryExperiment]);

  return { view, lesson, step, focus, dirty: step !== null || done.length > 0, reset };
}
