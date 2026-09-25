/** The lesson framework: reducer transitions, the URL round trip, focus keys, the preset copy. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  FREE_PLAY,
  courseGauge,
  focusKey,
  focusLabel,
  lessonQuery,
  lessonReducer,
  presetBefore,
  readLessonQuery,
  runList,
  stepSetup,
} from '../src/explainer/lessons.ts';
import type { Lesson, LessonAction, LessonContextBase, LessonState } from '../src/explainer/lessons.ts';
import type { Preset } from '../src/explainer/types.ts';
import { DEFAULT_PARAMS as LR_DEFAULTS, PRESETS as LR_PRESETS } from '../src/explainers/linear-regression/config.ts';
import { DEFAULT_PARAMS as PR_DEFAULTS, PRESETS as PR_PRESETS } from '../src/explainers/polynomial-regression/config.ts';
import { DEFAULT_PARAMS as NN_DEFAULTS, PRESETS as NN_PRESETS } from '../src/explainers/neural-network/config.ts';
import { DEFAULT_PARAMS as KM_DEFAULTS, PRESETS as KM_PRESETS } from '../src/explainers/k-means/config.ts';
import { DEFAULT_PARAMS as DT_DEFAULTS, PRESETS as DT_PRESETS } from '../src/explainers/decision-tree/config.ts';
import { DEFAULT_PARAMS as LG_DEFAULTS, PRESETS as LG_PRESETS } from '../src/explainers/logistic-regression/config.ts';
import { DEFAULT_PARAMS as KNN_DEFAULTS, PRESETS as KNN_PRESETS } from '../src/explainers/knn/config.ts';
import { DEFAULT_PARAMS as REG_DEFAULTS, PRESETS as REG_PRESETS } from '../src/explainers/regularization/config.ts';
import { DEFAULT_PARAMS as ACT_DEFAULTS, PRESETS as ACT_PRESETS } from '../src/explainers/activation-functions/config.ts';
import { DEFAULT_PARAMS as CNN_DEFAULTS, PRESETS as CNN_PRESETS } from '../src/explainers/cnn/config.ts';
import { DEFAULT_PARAMS as TF_DEFAULTS, PRESETS as TF_PRESETS } from '../src/explainers/transformers/config.ts';
import { LESSONS as LR_LESSONS } from '../src/explainers/linear-regression/lessons.ts';
import { LESSONS as PR_LESSONS } from '../src/explainers/polynomial-regression/lessons.ts';
import { LESSONS as NN_LESSONS } from '../src/explainers/neural-network/lessons.ts';
import { LESSONS as KM_LESSONS } from '../src/explainers/k-means/lessons.ts';
import { LESSONS as DT_LESSONS } from '../src/explainers/decision-tree/lessons.ts';
import { LESSONS as LG_LESSONS } from '../src/explainers/logistic-regression/lessons.ts';
import { LESSONS as KNN_LESSONS } from '../src/explainers/knn/lessons.ts';
import { LESSONS as REG_LESSONS } from '../src/explainers/regularization/lessons.ts';
import { LESSONS as ACT_LESSONS } from '../src/explainers/activation-functions/lessons.ts';
import { LESSONS as CNN_LESSONS } from '../src/explainers/cnn/lessons.ts';
import { LESSONS as TF_LESSONS } from '../src/explainers/transformers/lessons.ts';
import { clean } from './lessonRig.ts';

type Params = Record<string, number | string | boolean>;

const ctx: LessonContextBase = {
  sim: { iteration: 0, isRunning: false, isComplete: false },
  lesson: { metAt: null, enteredAt: 0 },
};

const lessons: Lesson<LessonContextBase, Params>[] = [
  {
    id: 'a',
    title: 'A',
    hook: '',
    section: 'hook',
    presetId: 'a',
    steps: [
      { id: 'one', kind: 'sandbox', say: 'one', experiments: [] },
      {
        id: 'two',
        kind: 'sandbox',
        say: (c) => 'Step ' + c.sim.iteration,
        takeaway: 'Because.',
        experiments: [{ label: 'Run', say: 'Short.', then: 'A longer outcome line.' }],
      },
    ],
  },
  { id: 'b', title: 'B', hook: '', section: 'core', presetId: 'b', steps: [{ id: 'one', kind: 'sandbox', experiments: [] }] },
];

const reduce = (state: LessonState, event: Parameters<typeof lessonReducer>[1]) =>
  lessonReducer(state, event, lessons);

describe('lesson reducer', () => {
  it('starts a lesson at a step, counting entries', () => {
    const s = reduce(FREE_PLAY, { type: 'start', lessonId: 'a', stepIndex: 1, iteration: 3 });
    assert.deepEqual(s.position, { lessonId: 'a', stepIndex: 1 });
    assert.equal(s.enteredAt, 3);
    assert.equal(s.via, 'start');
    assert.ok(s.entry > FREE_PLAY.entry);
  });

  it('exit clears the position and still counts as an entry', () => {
    const s = reduce(FREE_PLAY, { type: 'start', lessonId: 'a', iteration: 0 });
    const free = reduce(s, { type: 'exit' });
    assert.equal(free.position, null);
    assert.ok(free.entry > s.entry);
  });

  it('ignores unknown lessons and clamps the step', () => {
    assert.equal(reduce(FREE_PLAY, { type: 'start', lessonId: 'nope', iteration: 0 }), FREE_PLAY);
    const s = reduce(FREE_PLAY, { type: 'start', lessonId: 'a', stepIndex: 99, iteration: 0 });
    assert.equal(s.position?.stepIndex, 1);
  });

  it('jump walks whole lessons', () => {
    let s = reduce(FREE_PLAY, { type: 'jump', delta: 1, iteration: 0 });
    assert.equal(s.position?.lessonId, 'a', 'from free play the first jump lands on lesson one');
    s = reduce(s, { type: 'jump', delta: 1, iteration: 0 });
    assert.equal(s.position?.lessonId, 'b');
    assert.equal(reduce(s, { type: 'jump', delta: 1, iteration: 0 }), s);
    assert.equal(reduce(s, { type: 'jump', delta: -1, iteration: 0 }).position?.lessonId, 'a');
  });
});

describe('lesson URL', () => {
  it('reads a position back, rejecting unknown ids and clamping', () => {
    assert.deepEqual(readLessonQuery({ lesson: 'a', step: '1' }, lessons), { lessonId: 'a', stepIndex: 1 });
    assert.deepEqual(readLessonQuery({ lesson: 'a' }, lessons), { lessonId: 'a', stepIndex: 0 });
    assert.deepEqual(readLessonQuery({ lesson: 'a', step: '7' }, lessons), { lessonId: 'a', stepIndex: 1 });
    assert.equal(readLessonQuery({ lesson: 'zzz' }, lessons), null);
    assert.equal(readLessonQuery({ lesson: 'a', step: 'x' }, lessons), null);
    assert.equal(readLessonQuery({}, lessons), null);
  });

  it('writes the open lesson, the step only past the first, and nothing in free play', () => {
    assert.deepEqual(lessonQuery({ lessonId: 'a', stepIndex: 0 }), { lesson: 'a', step: undefined });
    assert.deepEqual(lessonQuery({ lessonId: 'a', stepIndex: 1 }), { lesson: 'a', step: '1' });
    assert.deepEqual(lessonQuery(null), { lesson: undefined, step: undefined });
  });
});

describe('focus targets', () => {
  it('names a target by its label, else its key', () => {
    assert.equal(focusLabel({ kind: 'metric', key: 'loss', label: 'Loss' }), 'Loss');
    assert.equal(focusLabel({ kind: 'control', key: 'learningRate' }), 'learningRate');
    assert.equal(focusLabel({ kind: 'transport', part: 'step' }), 'step');
    assert.equal(focusLabel({ kind: 'node', id: 'sum' }), 'sum');
  });

  it('keys targets the way elements register them', () => {
    assert.equal(focusKey({ kind: 'metric', key: 'loss' }), 'metric:loss');
    assert.equal(focusKey({ kind: 'transport', part: 'play' }), 'transport:buttons');
    assert.equal(focusKey({ kind: 'transport', part: 'speed' }), 'transport:speed');
    assert.equal(focusKey({ kind: 'view', id: 'second', value: 'log' }), 'view:second:log');
    assert.equal(focusKey({ kind: 'node', id: 'sum' }), 'node:sum');
  });
});

describe('the gauge', () => {
  it('lays out every step with its longest outcome, reading live text once', () => {
    const gauge = courseGauge(lessons, ctx);
    assert.equal(gauge.length, 3);
    assert.equal(gauge[1].text, 'Step 0');
    assert.equal(gauge[1].takeaway, 'Because.');
    assert.deepEqual(gauge[1].experiments, ['Run']);
    assert.equal(gauge[1].experimentOpen, 'A longer outcome line.');
    assert.equal(gauge[2].text, '');
  });
});

describe('a step entered directly', () => {
  it('starts from the last preset its lesson loaded before it', () => {
    const lesson: Lesson<LessonContextBase, Params> = {
      id: 'x',
      title: 'X',
      hook: '',
      section: 'core',
      presetId: 'first',
      steps: [
        { id: 'a', kind: 'sandbox', enter: [{ type: 'preset', id: 'first' }], experiments: [] },
        { id: 'b', kind: 'sandbox', enter: [{ type: 'preset', id: 'second' }, { type: 'runTo', steps: 5 }], experiments: [] },
        { id: 'c', kind: 'sandbox', experiments: [] },
      ],
    };
    assert.equal(presetBefore(lesson, 0), 'first');
    assert.equal(presetBefore(lesson, 1), 'second');
    assert.equal(presetBefore(lesson, 2), 'second');
    assert.equal(presetBefore({ ...lesson, steps: [{ id: 'c', kind: 'sandbox', experiments: [] }] }, 0), 'first');
  });

  it('a run restates its step setup before its patch; its own actions stand as written', () => {
    const lesson: Lesson<LessonContextBase, Params> = {
      id: 'x',
      title: 'X',
      hook: '',
      section: 'core',
      presetId: 'first',
      steps: [
        {
          id: 'a',
          kind: 'sandbox',
          enter: [{ type: 'preset', id: 'first' }, { type: 'params', patch: { k: 1 } }, { type: 'runTo', steps: 5 }],
          experiments: [
            { label: 'Patch', say: 'p', patch: { k: 2 } },
            { label: 'Own', say: 'o', enter: [{ type: 'play' }] },
          ],
        },
        { id: 'b', kind: 'sandbox', experiments: [] },
      ],
    };
    assert.deepEqual(stepSetup(lesson, 0), [{ type: 'preset', id: 'first' }, { type: 'params', patch: { k: 1 } }]);
    assert.deepEqual(stepSetup(lesson, 1), [{ type: 'preset', id: 'first' }]);
    const [patch, own] = lesson.steps[0].experiments;
    assert.deepEqual(runList(lesson, 0, patch), [
      { type: 'preset', id: 'first' },
      { type: 'params', patch: { k: 1 } },
      { type: 'params', patch: { k: 2 } },
      { type: 'reset' },
      { type: 'play' },
    ]);
    assert.deepEqual(runList(lesson, 0, own), [{ type: 'play' }]);
  });
});

describe('runs never stack', () => {
  const pages = [
    ['linear-regression', LR_LESSONS, LR_PRESETS, LR_DEFAULTS],
    ['polynomial-regression', PR_LESSONS, PR_PRESETS, PR_DEFAULTS],
    ['logistic-regression', LG_LESSONS, LG_PRESETS, LG_DEFAULTS],
    ['k-means', KM_LESSONS, KM_PRESETS, KM_DEFAULTS],
    ['decision-tree', DT_LESSONS, DT_PRESETS, DT_DEFAULTS],
    ['neural-network', NN_LESSONS, NN_PRESETS, NN_DEFAULTS],
    ['knn', KNN_LESSONS, KNN_PRESETS, KNN_DEFAULTS],
    ['regularization', REG_LESSONS, REG_PRESETS, REG_DEFAULTS],
    ['activation-functions', ACT_LESSONS, ACT_PRESETS, ACT_DEFAULTS],
    ['cnn', CNN_LESSONS, CNN_PRESETS, CNN_DEFAULTS],
    ['transformers', TF_LESSONS, TF_PRESETS, TF_DEFAULTS],
  ] as const;

  for (const [slug, lessonList, presets, defaults] of pages) {
    it(slug + ': a run that sets params lands on the same params whichever run came before it', () => {
      const all = lessonList as readonly Lesson<unknown, Params>[];
      const apply = (from: Params, list: readonly LessonAction<Params>[]): Params => {
        let params = from;
        for (const action of list) {
          if (action.type === 'preset') {
            const preset = (presets as readonly Preset<Params>[]).find((p) => p.id === action.id);
            assert.ok(preset, slug + ': unknown preset ' + action.id);
            params = { ...(defaults as Params), ...preset.params };
          } else if (action.type === 'params') params = { ...params, ...action.patch };
        }
        return params;
      };
      for (const lesson of all) {
        lesson.steps.forEach((step, i) => {
          const opened = apply(defaults as Params, stepSetup(lesson, i));
          for (const second of step.experiments) {
            const list = runList(lesson, i, second);
            if (!list.some((a) => a.type === 'preset' || a.type === 'params')) continue;
            const alone = apply(opened, list);
            for (const first of step.experiments) {
              const after = apply(apply(opened, runList(lesson, i, first)), list);
              assert.deepEqual(after, alone, slug + '/' + lesson.id + ': ' + second.label + ' after ' + first.label);
            }
          }
        });
      }
    });
  }
});

describe('preset copy', () => {
  const pages = [
    ['linear-regression', LR_PRESETS],
    ['polynomial-regression', PR_PRESETS],
    ['logistic-regression', LG_PRESETS],
    ['k-means', KM_PRESETS],
    ['decision-tree', DT_PRESETS],
    ['neural-network', NN_PRESETS],
    ['knn', KNN_PRESETS],
    ['regularization', REG_PRESETS],
    ['activation-functions', ACT_PRESETS],
    ['cnn', CNN_PRESETS],
    ['transformers', TF_PRESETS],
  ] as const;

  for (const [slug, presets] of pages) {
    it(slug + ': unique ids, a name and a clean blurb on every preset', () => {
      const ids = new Set<string>();
      for (const preset of presets) {
        assert.ok(!ids.has(preset.id), slug + ': duplicate preset id ' + preset.id);
        ids.add(preset.id);
        clean(preset.name, slug + ' / ' + preset.id + ' name');
        clean(preset.blurb, slug + ' / ' + preset.id + ' blurb');
      }
    });
  }
});
