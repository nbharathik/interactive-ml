/** A headless copy of a page's lesson wiring, shared by the lesson suites. */

import { strict as assert } from 'node:assert';

import type { Experiment, Lesson, LessonAction, LessonContextBase } from '../src/explainer/lessons.ts';
import { resolveSay } from '../src/explainer/lessons.ts';
import type { Preset } from '../src/explainer/types.ts';

type Params = Record<string, number | string | boolean>;

export const BUDGET = { takeaway: 250, label: 30, title: 28 };
/** One-screen courses: a shorter say, and an outcome line per run. */
export const SCREEN = { say: 260, then: 220, runs: [2, 6] as const };
const DASH = new RegExp('[' + String.fromCharCode(0x2013, 0x2014) + ']');
const RUN_CAP = 30_000;

export interface RigModel<P extends Params, S, M, U> {
  defaults: P;
  presets: readonly Preset<P>[];
  /** Everything derived from the params: the data, the config. */
  build(params: P): M;
  create(model: M): S;
  step(state: S, model: M): S;
  complete(state: S, model: M): boolean;
  /** The page's ui state a lesson can read, as it starts. */
  ui(): U;
  /** Whether changing this param restarts the run, mirroring the page's `deps`. Default: every param does. */
  resets?(key: string): boolean;
}

export class LessonRig<P extends Params, S, M, U> {
  params: P;
  state: S;
  iteration = 0;
  enteredAt = 0;
  ui: U;
  private cache: M | null = null;

  constructor(private readonly def: RigModel<P, S, M, U>) {
    this.params = { ...def.defaults };
    this.ui = def.ui();
    this.state = def.create(this.model());
  }

  model(): M {
    if (!this.cache) this.cache = this.def.build(this.params);
    return this.cache;
  }

  applyPreset(id: string) {
    const preset = this.def.presets.find((p) => p.id === id);
    assert.ok(preset, 'unknown preset ' + id);
    this.params = { ...this.def.defaults, ...preset.params };
    this.cache = null;
    this.reset();
  }

  /** Change params the way the page's controls do: the run restarts only for the knobs that rebuild it. */
  merge(patch: Partial<P>) {
    this.params = { ...this.params, ...patch };
    this.cache = null;
    const resets = this.def.resets ?? (() => true);
    if (Object.keys(patch).some((key) => resets(key))) this.reset();
  }

  reset() {
    this.state = this.def.create(this.model());
    this.iteration = 0;
  }

  get complete() {
    return this.def.complete(this.state, this.model());
  }

  stepOnce(count = 1) {
    for (let i = 0; i < count && !this.complete; i++) {
      this.state = this.def.step(this.state, this.model());
      this.iteration += 1;
    }
  }

  /** The hook's share of the context. */
  base(): LessonContextBase {
    return {
      sim: { iteration: this.iteration, isRunning: false, isComplete: this.complete },
      lesson: { metAt: null, enteredAt: this.enteredAt },
    };
  }

  /** Setup actions only; play, step and runTo are driven by `drive`. The entry is counted once the setup has applied, as the page arms it. */
  enter(actions: readonly LessonAction<P>[] | undefined) {
    for (const action of actions ?? []) {
      if (action.type === 'preset') this.applyPreset(action.id);
      else if (action.type === 'params') this.merge(action.patch);
      else if (action.type === 'reset') this.reset();
      else if (action.type === 'open' && this.ui && typeof this.ui === 'object' && 'open' in this.ui) (this.ui as { open: string | null }).open = action.target;
    }
    this.enteredAt = this.iteration;
  }

  /** Drive the transport the way the runtime would until `until` holds. */
  drive<Ctx>(actions: readonly LessonAction<P>[] | undefined, until: (c: Ctx) => boolean, context: () => Ctx) {
    for (const action of actions ?? []) {
      if (action.type === 'step') this.stepOnce(action.count);
      else if (action.type === 'runTo') this.stepOnce(action.steps);
      else if (action.type === 'play') {
        let taken = 0;
        while (!until(context()) && !this.complete && taken < RUN_CAP) {
          this.stepOnce();
          taken += 1;
        }
      }
    }
    assert.ok(until(context()), 'condition never held');
  }
}

/* ---------------- copy checks ---------------- */

export function clean(text: string, what: string) {
  assert.ok(text.length > 0, what + ' is empty');
  for (const bad of ['NaN', 'undefined', 'n/a', 'null', 'Infinity']) {
    assert.ok(!text.includes(bad), what + ' reads ' + bad + ': ' + text);
  }
  assert.ok(!DASH.test(text), what + ' has a dash: ' + text);
  assert.ok(!text.includes('!'), what + ' has an exclamation mark: ' + text);
}

export function budget(text: string, max: number, what: string) {
  assert.ok(text.length <= max, what + ' is ' + text.length + ' > ' + max + ': ' + text);
}

/** The static copy quotes a number the way the page formats it. */
export function reads(where: string, text: string, value: string) {
  assert.ok(text.includes(value), where + ' should read ' + value + ': ' + text);
}

/** The structural rules every authored lesson set follows; `oneScreen` adds the one-lesson-one-screen rules. */
export function checkStructure<Ctx, P extends Params>(
  lessons: readonly Lesson<Ctx, P>[],
  presets: readonly Preset<P>[],
  defaults: P,
  options: { oneScreen?: boolean; instant?: boolean } = {},
) {
  const knownPreset = (id: string, where: string) => assert.ok(presets.some((p) => p.id === id), where + ': unknown preset ' + id);
  const knownKeys = (patch: object, where: string) => {
    for (const key of Object.keys(patch)) assert.ok(key in defaults, where + ': unknown param ' + key);
  };
  const checkActions = (actions: readonly LessonAction<P>[] | undefined, where: string) => {
    for (const action of actions ?? []) {
      if (action.type === 'preset') knownPreset(action.id, where);
      if (action.type === 'params') knownKeys(action.patch, where);
    }
  };
  const ids = new Set<string>();
  for (const lesson of lessons) {
    assert.ok(!ids.has(lesson.id), 'duplicate lesson id ' + lesson.id);
    ids.add(lesson.id);
    knownPreset(lesson.presetId, lesson.id);
    const first = lesson.steps[0].enter?.[0];
    assert.deepEqual(first, { type: 'preset', id: lesson.presetId }, lesson.id + ': first action must load its preset');
    clean(lesson.title, lesson.id + ' title');
    clean(lesson.hook, lesson.id + ' hook');
    const stepIds = new Set<string>();
    for (const s of lesson.steps) {
      const where = lesson.id + '/' + s.id;
      assert.ok(!stepIds.has(s.id), where + ': duplicate step id');
      stepIds.add(s.id);
      if (s.title !== undefined) {
        clean(s.title, where + ' title');
        budget(s.title, BUDGET.title, where + ' title');
      }
      if (s.focus) assert.ok(s.focus.label, where + ': focus without a label');
      checkActions(s.enter, where);
      for (const e of s.experiments) {
        assert.ok(e.focus?.label, where + ': experiment ' + e.label + ' without a focus label');
        knownKeys(e.patch ?? {}, where);
        checkActions(e.enter, where);
        if (e.until) assert.ok(e.then, where + ': run ' + e.label + ' has a condition but no outcome');
        clean(e.label, where + ' experiment label');
        budget(e.label, BUDGET.label, where + ' experiment label');
        clean(e.say, where + ' experiment say');
        budget(e.say, SCREEN.then, where + ' experiment say');
      }
    }
    if (options.oneScreen) {
      assert.equal(lesson.steps.length, 1, lesson.id + ' has ' + lesson.steps.length + ' steps');
      const s = lesson.steps[0];
      // A closed-form page has its picture at step 0; the others drive to it.
      const drives = (s.enter ?? []).some((a) => a.type === 'runTo' || a.type === 'step' || a.type === 'play');
      assert.ok(drives || options.instant || lesson.section === 'sandbox', lesson.id + ' must open on its picture');
      // The Experiments lesson is the sandbox and may list more.
      const most = lesson.section === 'sandbox' ? 12 : SCREEN.runs[1];
      assert.ok(s.experiments.length >= SCREEN.runs[0] && s.experiments.length <= most, lesson.id + ' has ' + s.experiments.length + ' runs');
      for (const key of lesson.knobs ?? []) assert.ok(key in defaults, lesson.id + ': unknown knob ' + key);
    }
  }
  assert.equal(lessons[0].section, 'hook');
  assert.equal(lessons[lessons.length - 1].section, 'sandbox');
  for (let i = 1; i < lessons.length; i++) {
    const order = ['hook', 'core', 'failure', 'sandbox'];
    assert.ok(order.indexOf(lessons[i].section) >= order.indexOf(lessons[i - 1].section), 'sections out of order at ' + lessons[i].id);
  }
}

/** Enter a one-screen lesson the way the runtime does: its setup, then the drive that opens it on its picture. */
export function openLesson<P extends Params, S, M, U, Ctx extends LessonContextBase>(
  make: () => LessonRig<P, S, M, U>,
  lesson: Lesson<Ctx, P>,
  context: (rig: LessonRig<P, S, M, U>) => Ctx,
): LessonRig<P, S, M, U> {
  const rig = make();
  const s = lesson.steps[0];
  rig.enter(s.enter);
  rig.drive(s.enter, () => true, () => context(rig));
  return rig;
}

/** The lesson's own lines, read on its opening picture: say, key idea and the live chips. */
export function exerciseScreen<P extends Params, S, M, U, Ctx extends LessonContextBase>(
  rig: LessonRig<P, S, M, U>,
  lesson: Lesson<Ctx, P>,
  context: (rig: LessonRig<P, S, M, U>) => Ctx,
  reveals: string[],
) {
  const s = lesson.steps[0];
  const ctx = context(rig);
  const text = resolveSay(s.say, ctx);
  clean(text, lesson.id + ' say');
  budget(text, SCREEN.say, lesson.id + ' say');
  if (s.takeaway !== undefined) {
    const idea = resolveSay(s.takeaway, ctx);
    clean(idea, lesson.id + ' takeaway');
    budget(idea, BUDGET.takeaway, lesson.id + ' takeaway');
  }
  const numbers = s.numbers?.(ctx) ?? [];
  for (const n of numbers) {
    clean(n.label, lesson.id + ' number label');
    clean(n.value, lesson.id + ' number ' + n.label);
  }
  reveals.push(lesson.id + ' opens: ' + (numbers.length ? numbers.map((n) => n.label + ' ' + n.value).join(', ') : text.slice(0, 60)));
}

/** Run one experiment from the lesson's opening picture and check its outcome line. */
export function exerciseRun<P extends Params, S, M, U, Ctx extends LessonContextBase>(
  rig: LessonRig<P, S, M, U>,
  e: Experiment<Ctx, P>,
  where: string,
  context: (rig: LessonRig<P, S, M, U>) => Ctx,
  reveals: string[],
) {
  const ctx = () => context(rig);
  const list: LessonAction<P>[] = e.enter ?? [
    ...(e.patch ? [{ type: 'params' as const, patch: e.patch }] : []),
    { type: 'reset' as const },
    { type: 'play' as const },
  ];
  rig.enter(list);
  if (!e.until) {
    rig.drive(list, () => true, ctx);
    reveals.push(where + ': ' + e.say);
    return;
  }
  rig.drive(list, e.until, ctx);
  const then = resolveSay(e.then, ctx());
  clean(then, where + ' then');
  budget(then, SCREEN.then, where + ' then');
  reveals.push(where + ': ' + then);
}
