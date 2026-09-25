/** Guided lessons: a preset, one screen of narration and a few one-click runs. Pure data and functions, no React. */

import type { SpeedName } from './useSimulation';

/* ---------------- content types ---------------- */

/** What every lesson can read, whatever the explainer. Pages extend it. */
export interface LessonContextBase {
  sim: { iteration: number; isRunning: boolean; isComplete: boolean };
  lesson: {
    /** Iteration when the run's condition first held, or null. */
    metAt: number | null;
    /** Iteration when the run was entered. */
    enteredAt: number;
  };
}

/** Static prose, or prose with live numbers in it. */
export type Say<Ctx> = string | ((ctx: Ctx) => string);

export type FocusTarget =
  | { kind: 'node' | 'edge'; id: string; label?: string }
  | { kind: 'metric'; key: string; label?: string }
  | { kind: 'panel'; id: string; label?: string }
  | { kind: 'control'; key: string; label?: string }
  | { kind: 'transport'; part: 'play' | 'step' | 'run' | 'reset' | 'speed'; label?: string }
  | { kind: 'view'; id: string; value?: string; label?: string }
  | { kind: 'custom'; id: string; label?: string };

export type LessonAction<P> =
  | { type: 'preset'; id: string }
  | { type: 'params'; patch: Partial<P> }
  | { type: 'reset' }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step'; count: number }
  | { type: 'runTo'; steps: number }
  | { type: 'speed'; speed: SpeedName }
  | { type: 'view'; id: string; value: string }
  | { type: 'open'; target: string | null };

export type Condition<Ctx> = (ctx: Ctx) => boolean;

/** A live number shown as a chip under the narration, so the words and the page agree. */
export interface LiveNumber {
  label: string;
  value: string;
}

/** A one-click run under the narration. */
export interface Experiment<Ctx, P> {
  label: string;
  /** The outcome line; static, or replaced by `then` once `until` holds. */
  say: string;
  patch?: Partial<P>;
  focus?: FocusTarget;
  /** What the run does; default: the step's setup, the patch, a fresh start, play. */
  enter?: LessonAction<P>[];
  /** A live run: pauses when this holds and shows `then` with the numbers. */
  until?: Condition<Ctx>;
  then?: Say<Ctx>;
  speed?: SpeedName;
}

/** One screen: the picture opens solved, the text names what is on it, the runs sit under it. */
export interface LessonStep<Ctx, P> {
  id: string;
  kind: 'sandbox';
  /** A short name for the step, shown in the outline when a lesson has several. */
  title?: string;
  say?: Say<Ctx>;
  /** The one idea to keep, shown under the runs. */
  takeaway?: Say<Ctx>;
  focus?: FocusTarget;
  /** Run once, in order, when the step is entered. */
  enter?: LessonAction<P>[];
  /** The numbers the narration quotes, read live. */
  numbers?: (ctx: Ctx) => LiveNumber[];
  experiments: Experiment<Ctx, P>[];
}

/** The interest curve, in order. Also the outline's sections. */
export type LessonSection = 'hook' | 'core' | 'failure' | 'sandbox';

export const SECTION_ORDER: LessonSection[] = ['hook', 'core', 'failure', 'sandbox'];

export const SECTION_TITLES: Record<LessonSection, string> = {
  hook: 'Introduction',
  core: 'Core concepts',
  failure: 'Failure modes',
  sandbox: 'Practice',
};

export interface Lesson<Ctx, P> {
  id: string;
  title: string;
  /** One line for the outline: the question the lesson answers. */
  hook: string;
  section: LessonSection;
  /** The preset the first step loads. Later steps may load others. */
  presetId: string;
  steps: LessonStep<Ctx, P>[];
  /** The knobs the lesson names; the others dim. Default: the keys its runs and focus touch. */
  knobs?: string[];
}

export function resolveSay<Ctx>(say: Say<Ctx> | undefined, ctx: Ctx): string {
  if (say === undefined) return '';
  return typeof say === 'function' ? say(ctx) : say;
}

/* ---------------- gauge ---------------- */

/** A step's static content, rendered unseen so the card can take the height of the tallest step. */
export interface StepGauge {
  key: string;
  title: string;
  text: string;
  takeaway: string;
  focusLabel: string | null;
  /** The labels of the step's runs, one row each. */
  experiments: string[];
  /** The longest outcome a run can open, so the card makes room for one. */
  experimentOpen: string;
  /** How many live-number chips the step shows. */
  numbers: number;
}

function gaugeSay<Ctx>(say: Say<Ctx> | undefined, ctx: Ctx): string {
  try {
    return resolveSay(say, ctx);
  } catch {
    return '';
  }
}

function gaugeNumbers<Ctx>(numbers: (ctx: Ctx) => LiveNumber[], ctx: Ctx): number {
  try {
    return numbers(ctx).length;
  } catch {
    return 0;
  }
}

const longest = (lines: readonly string[]): string => lines.reduce((best, line) => (line.length > best.length ? line : best), '');

/** Every step of the course, with its live numbers read once. */
export function courseGauge<Ctx, P>(lessons: readonly Lesson<Ctx, P>[], ctx: Ctx): StepGauge[] {
  const out: StepGauge[] = [];
  for (const lesson of lessons) {
    lesson.steps.forEach((step, i) => {
      out.push({
        key: lesson.id + ':' + i,
        title: step.title ?? '',
        text: gaugeSay(step.say, ctx),
        takeaway: gaugeSay(step.takeaway, ctx),
        focusLabel: step.focus ? focusLabel(step.focus) : null,
        experiments: step.experiments.map((e) => e.label),
        experimentOpen: longest(step.experiments.map((e) => longest([e.say, gaugeSay(e.then, ctx)]))),
        numbers: step.numbers ? gaugeNumbers(step.numbers, ctx) : 0,
      });
    });
  }
  return out;
}

/* ---------------- runtime state ---------------- */

export const LESSON_QUERY_KEYS = ['lesson', 'step'] as const;

export interface LessonPosition {
  lessonId: string;
  stepIndex: number;
}

export interface LessonState {
  position: LessonPosition | null;
  /** Iteration when the step was entered. */
  enteredAt: number;
  /** Bumps on every step entry; the hook runs the step's actions on it. */
  entry: number;
  /** How the step was entered; the hook skips autoplay on init. */
  via: 'init' | 'start';
}

export type LessonEvent =
  | { type: 'start'; lessonId: string; stepIndex?: number; iteration: number }
  | { type: 'jump'; delta: 1 | -1; iteration: number }
  | { type: 'exit' };

export function findLesson<Ctx, P>(
  lessons: readonly Lesson<Ctx, P>[],
  id: string | null | undefined,
): Lesson<Ctx, P> | undefined {
  return id ? lessons.find((l) => l.id === id) : undefined;
}

/** The preset a step starts from when entered directly: the last one loaded before it in its lesson. */
export function presetBefore<Ctx, P>(lesson: Lesson<Ctx, P>, stepIndex: number): string {
  for (let i = Math.min(stepIndex, lesson.steps.length - 1); i >= 0; i--) {
    for (const action of lesson.steps[i].enter ?? []) {
      if (action.type === 'preset') return action.id;
    }
  }
  return lesson.presetId;
}

/** A step's own setup, its preset and params, so a run can restate it and runs never stack. */
export function stepSetup<Ctx, P>(lesson: Lesson<Ctx, P>, stepIndex: number): LessonAction<P>[] {
  const own = (lesson.steps[stepIndex]?.enter ?? []).filter((a) => a.type === 'preset' || a.type === 'params');
  return own.some((a) => a.type === 'preset') ? own : [{ type: 'preset', id: presetBefore(lesson, stepIndex) }, ...own];
}

/** What a run does: its own actions, else the step's setup, its patch, a fresh start and play. */
export function runList<Ctx, P>(lesson: Lesson<Ctx, P>, stepIndex: number, run: Experiment<Ctx, P>): LessonAction<P>[] {
  if (run.enter) return [...run.enter];
  return [
    ...stepSetup(lesson, stepIndex),
    ...(run.patch ? [{ type: 'params' as const, patch: run.patch }] : []),
    { type: 'reset' },
    { type: 'play' },
  ];
}

export function stepAt<Ctx, P>(
  lessons: readonly Lesson<Ctx, P>[],
  position: LessonPosition | null,
): LessonStep<Ctx, P> | undefined {
  const lesson = findLesson(lessons, position?.lessonId);
  return lesson && position ? lesson.steps[position.stepIndex] : undefined;
}

function enter(state: LessonState, position: LessonPosition, via: LessonState['via'], iteration: number): LessonState {
  return { position, enteredAt: iteration, entry: state.entry + 1, via };
}

export const FREE_PLAY: LessonState = { position: null, enteredAt: 0, entry: 0, via: 'init' };

export function lessonReducer<Ctx, P>(
  state: LessonState,
  event: LessonEvent,
  lessons: readonly Lesson<Ctx, P>[],
): LessonState {
  const current = state.position ? findLesson(lessons, state.position.lessonId) : undefined;
  const index = current ? lessons.indexOf(current) : -1;
  switch (event.type) {
    case 'start': {
      const lesson = findLesson(lessons, event.lessonId);
      if (!lesson) return state;
      const stepIndex = Math.min(Math.max(0, event.stepIndex ?? 0), lesson.steps.length - 1);
      return enter(state, { lessonId: lesson.id, stepIndex }, 'start', event.iteration);
    }
    case 'jump': {
      const target = lessons[index < 0 && event.delta < 0 ? 0 : index + event.delta];
      if (!target) return state;
      return enter(state, { lessonId: target.id, stepIndex: 0 }, 'start', event.iteration);
    }
    case 'exit':
      return { ...FREE_PLAY, entry: state.entry + 1 };
    default:
      return state;
  }
}

/* ---------------- url ---------------- */

export function readLessonQuery<Ctx, P>(
  extras: Record<string, string>,
  lessons: readonly Lesson<Ctx, P>[],
): LessonPosition | null {
  const lesson = findLesson(lessons, extras.lesson);
  if (!lesson) return null;
  const raw = extras.step === undefined ? 0 : Number(extras.step);
  if (!Number.isInteger(raw)) return null;
  return { lessonId: lesson.id, stepIndex: Math.min(Math.max(0, raw), lesson.steps.length - 1) };
}

/** Free play is the default, so only an open lesson travels; the step only past the first. */
export function lessonQuery(position: LessonPosition | null): Record<string, string | undefined> {
  if (!position) return { lesson: undefined, step: undefined };
  return { lesson: position.lessonId, step: position.stepIndex > 0 ? String(position.stepIndex) : undefined };
}

/** What to call a target: its label, else its key or id. */
export function focusLabel(target: FocusTarget): string {
  if (target.label) return target.label;
  switch (target.kind) {
    case 'metric':
    case 'control':
      return target.key;
    case 'transport':
      return target.part;
    default:
      return target.id;
  }
}

/** The key a target renders as data-lesson-target, and the focus matches on. */
export function focusKey(target: FocusTarget): string {
  switch (target.kind) {
    case 'metric':
      return 'metric:' + target.key;
    case 'control':
      return 'control:' + target.key;
    case 'transport':
      // The buttons share one target.
      return 'transport:' + (target.part === 'speed' ? 'speed' : 'buttons');
    case 'view':
      return 'view:' + target.id + (target.value ? ':' + target.value : '');
    default:
      return target.kind + ':' + target.id;
  }
}
