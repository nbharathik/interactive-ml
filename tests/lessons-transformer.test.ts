/** The transformer lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { resolveSay } from '../src/explainer/lessons.ts';
import type { LessonAction } from '../src/explainer/lessons.ts';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from '../src/explainers/transformers/config.ts';
import type { TfParams } from '../src/explainers/transformers/config.ts';
import { LETTER_WEIGHTS } from '../src/explainers/transformers/letterWeights.ts';
import { LESSONS, REFERENCE, isFitted, makeTfContext, placeGap } from '../src/explainers/transformers/lessons.ts';
import type { TfLessonContext } from '../src/explainers/transformers/lessons.ts';
import { buildData, buildSpec, probeOf, startState, writeFrom } from '../src/explainers/transformers/model.ts';
import { letterSentences } from '../src/lib/datasets/sequences.ts';
import type { SequenceDataset } from '../src/lib/datasets/sequences.ts';
import { forward, parameterCount, positionTable, step } from '../src/lib/ml/transformer.ts';
import type { TransformerSpec, TransformerState } from '../src/lib/ml/transformer.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: SequenceDataset;
  spec: TransformerSpec;
  start: string;
}

type Ui = TfLessonContext['ui'];

const METRIC_KEYS = ['test-acc', 'train-loss', 'train-acc', 'exact', 'test-loss', 'params', 'epoch'];
const PANEL_IDS = ['architecture', 'answer', 'gallery', 'second'];
const VIEW_IDS = ['arch', 'second'];
const CONTROL_KEYS = CONTROL_GROUPS.flatMap((g) => g.controls.map((c) => c.key as string));

/** The model built the way Page.tsx builds it. */
function build(p: TfParams): Model {
  const data = buildData(p);
  return { data, spec: buildSpec(p, data), start: p.start };
}

/** The page's rig, plus the head view's pick, which the lessons set with a view action. */
class TfRig extends LessonRig<TfParams, TransformerState, Model, Ui> {
  enter(actions: readonly LessonAction<TfParams>[] | undefined) {
    for (const action of actions ?? []) {
      if (action.type !== 'view') continue;
      if (action.id === 'pick') {
        const [layer, head, query] = action.value.split(':').map(Number);
        this.ui.pick = { layer, head, query };
      } else if (action.id === 'arch') this.ui.view = action.value as Ui['view'];
      else if (action.id === 'second') this.ui.secondView = action.value as Ui['secondView'];
    }
    super.enter(actions);
    // The page writes on a letter a beat; the outcome line reads the finished sentence, so the rig sets the start at once.
    for (const action of actions ?? []) if (action.type === 'view' && action.id === 'write') this.merge({ input: action.value });
  }
}

type Rig = TfRig;

const make = (): Rig =>
  new TfRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build,
    create: (m) => startState(m.spec, m.data, m.start),
    step: (s, m) => step(s, m.data, m.spec),
    complete: (s, m) => s.diverged || isFitted(s, m.data.task),
    ui: () => ({ view: 'model', secondView: 'loss', open: null, pick: { layer: 0, head: -1, query: 1 } }),
    // The probe and a typed sequence change what is shown, not the run.
    resets: (key) => key !== 'probe' && key !== 'input',
  });

const context = (rig: Rig): TfLessonContext => {
  const { data, spec } = rig.model();
  const probe = probeOf(rig.params.input, rig.params.probe, data);
  const cache = forward(rig.state.weights, probe.tokens, spec);
  const written = writeFrom(rig.state.weights, spec, data, probe.tokens, probe.index === null && data.task === 'letters');
  return { ...makeTfContext({ params: rig.params, state: rig.state, data, spec, cache, written, ui: rig.ui }), ...rig.base() };
};

describe('transformer lessons: structure', () => {
  // The letter lessons open on the shipped trained model, so their picture is there without a run.
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true, instant: true }));

  it('every focus points at something the page renders', () => {
    for (const lesson of LESSONS) {
      for (const s of lesson.steps) {
        for (const target of [s.focus, ...s.experiments.map((e) => e.focus)]) {
          if (!target) continue;
          const where = lesson.id + '/' + s.id;
          switch (target.kind) {
            case 'metric':
              assert.ok(METRIC_KEYS.includes(target.key), where + ': metric ' + target.key);
              break;
            case 'control':
              assert.ok(CONTROL_KEYS.includes(target.key), where + ': control ' + target.key);
              break;
            case 'panel':
              assert.ok(PANEL_IDS.includes(target.id), where + ': panel ' + target.id);
              break;
            case 'view':
              assert.ok(VIEW_IDS.includes(target.id), where + ': view ' + target.id);
              break;
            case 'node':
              // A lane component of the flow view, or a box of the model view.
              assert.match(target.id, /^(tok|emb|pred):\d+$|^(attn|add|ffn):\d+:\d+$|^(embed|position|norm1|attention|add1|norm2|ffn|add2|final|linear|softmax)$/, where + ': target ' + target.id);
              break;
            default:
              break;
          }
        }
      }
    }
  });
});

const reveals: string[] = [];

describe('transformer lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    for (const e of lesson.steps[0].experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

const pc = (v: number) => Math.round(v * 100) + '%';
const n = (v: number) => v.toLocaleString('en-US');

describe('transformer lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    const c = context(openLesson(make, lesson, context));
    return { text: resolveSay(lesson.steps[0].say, c), c };
  };
  /** A run's setup applied on its lesson's opening picture, and the context it leaves. */
  const ran = (id: string, label: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    const run = lesson.steps[0].experiments.find((e) => e.label === label)!;
    const rig = openLesson(make, lesson, context);
    rig.enter(run.enter);
    rig.drive(run.enter, run.until ?? (() => true), () => context(rig));
    const c = context(rig);
    return { c, then: run.then ? resolveSay(run.then, c) : '' };
  };

  it('the letter lessons open on the shipped trained model', () => {
    const { c } = opened('next-letter');
    assert.equal(c.state.step, LETTER_WEIGHTS.steps);
    assert.ok(c.derived.fitted, 'the shipped model is fitted');
  });

  it('writing: the trained model writes sentences of the language, the untrained one does not', () => {
    const language = new Set(letterSentences());
    for (const label of ['From “the h”', 'From “the dog”']) assert.ok(language.has(ran('next-letter', label).c.derived.writes), label);
    assert.ok(!language.has(ran('next-letter', 'Before training').c.derived.writes), 'untrained');
  });

  it('self-attention: from the space after says, block 2 head 2 reads the cat', () => {
    const { text, c } = opened('self-attention');
    assert.deepEqual([c.derived.focus.layer, c.derived.focus.head, c.derived.focus.animal], [1, 1, 'cat']);
    assert.ok(c.derived.focus.share > 0.9, 'share ' + c.derived.focus.share);
    reads('self-attention', text, pc(c.derived.focus.share) + ' on cat');
    // The run names the head it shows.
    const owl = ran('self-attention', 'Another animal').c;
    assert.deepEqual([owl.derived.focus.layer, owl.derived.focus.head, owl.derived.focus.animal], [1, 0, 'owl']);
  });

  it('queries: the lit row is the space after says, and its top key is the c of cat', () => {
    const { c } = opened('queries-keys-values');
    assert.equal(c.derived.pick.position, 'the cat often says '.length);
    assert.deepEqual([c.derived.pick.key, c.derived.pick.keyPosition], ['c', 5]);
  });

  it('softmax: the bets the copy reads', () => {
    const { text, c } = opened('softmax');
    reads('softmax', text, c.derived.next.map((x) => x.letter + ' ' + pc(x.p)).join(', '));
    assert.deepEqual(
      c.derived.next.map((x) => x.letter),
      ['s', 'b', 'h'],
    );
    assert.deepEqual(
      ran('softmax', 'After “the c”').c.derived.next.slice(0, 2).map((x) => x.letter),
      ['a', 'o'],
    );
    const sure = ran('softmax', 'After “the cat says ”').c.derived.next[0];
    assert.ok(sure.letter === 'm' && sure.p > 0.9, 'sure ' + JSON.stringify(sure));
  });

  it('positional encoding: the place gaps are the position table’s own', () => {
    const rig = make();
    rig.applyPreset('reverse');
    const { spec } = rig.model();
    const P = positionTable(rig.state.weights, spec);
    const d = spec.width;
    const row = (t: number) => Array.from(P.subarray(t * d, (t + 1) * d));
    const cosine = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0) / Math.hypot(...a) / Math.hypot(...b);
    for (const [a, b] of [[0, 1], [0, 6], [2, 5]]) {
      assert.ok(Math.abs(placeGap(d, a, b).similarity - cosine(row(a), row(b))) < 1e-9, a + ',' + b);
    }
    const far = placeGap(d, 0, 6);
    const near = placeGap(d, 0, 1);
    assert.deepEqual([far.fast, far.next], [344, 109]);
    assert.ok(far.similarity < near.similarity, 'near places look more alike');
    const text = ran('embeddings', 'Places 0 and 6').then;
    reads('embeddings', text, 'turned 344°');
    reads('embeddings', text, 'at 109°');
    reads('embeddings', text, 'similarity 0.81, against 0.94');
  });

  it('the mask and the learning rate quote their opening numbers', () => {
    const cheat = opened('no-mask');
    reads('no-mask', cheat.text, 'falls to ' + cheat.c.state.trainLoss.toFixed(2));
    reads('no-mask', cheat.text, 'only ' + pc(cheat.c.state.testExact));
    const hot = opened('learning-rate');
    reads('learning-rate', hot.text, 'held-out accuracy ' + pc(hot.c.state.testAccuracy) + ' after ' + hot.c.state.step + ' steps');
  });
});

describe('transformer lessons: the claims hold', () => {
  const fresh = (preset: string, patch: Partial<TfParams> = {}) => {
    const rig = make();
    rig.applyPreset(preset);
    rig.merge(patch);
    return rig;
  };
  const runTo = (rig: Rig, steps: number) => {
    while (rig.state.step < steps && !rig.complete) rig.stepOnce();
    return rig.state;
  };
  const fitAt = (preset: string, patch: Partial<TfParams> = {}, cap = 1500) => {
    const rig = fresh(preset, patch);
    runTo(rig, cap);
    return rig.complete && !rig.state.diverged ? rig.state.step : null;
  };

  it('reverse fits at its reference; no positions stalls', () => {
    assert.equal(fitAt('reverse'), REFERENCE.reverseFit);
    const bag = fresh('reverse-no-positions');
    runTo(bag, 400);
    assert.equal(pc(bag.state.testAccuracy), pc(REFERENCE.noPositions));
  });

  it('count needs the feed-forward block', () => {
    const full = fresh('count');
    runTo(full, 400);
    assert.equal(pc(full.state.testAccuracy), pc(REFERENCE.count400));
    const bare = fresh('count-no-ffn');
    runTo(bare, 800);
    assert.equal(pc(bare.state.testAccuracy), pc(REFERENCE.countNoFfn));
    assert.ok(bare.state.testAccuracy < full.state.testAccuracy - 0.1);
  });

  it('without the mask the language task copies: loss near zero, few sentences written right', () => {
    const cheat = fresh('animals-no-mask');
    runTo(cheat, 300);
    assert.ok(cheat.state.trainLoss < 0.01, 'loss ' + cheat.state.trainLoss);
    assert.equal(pc(cheat.state.testExact), pc(REFERENCE.noMask));
    const honest = fresh('animals');
    runTo(honest, 300);
    assert.equal(honest.state.testAccuracy, 1);
    assert.equal(honest.state.testExact, 1, 'with the mask every held-out sentence is written right');
    assert.ok(honest.state.trainLoss > 0.3, 'an honest model cannot know the words before the animal');
  });

  it('η = 0.1 stalls at its reference', () => {
    const hot = fresh('reverse-hot');
    runTo(hot, 400);
    assert.equal(pc(hot.state.testAccuracy), pc(REFERENCE.hot));
  });

  describe('the experiments', () => {
    const sandbox = LESSONS[LESSONS.length - 1].steps[0];
    const run = (label: string) => {
      const e = sandbox.experiments.find((x) => x.label === label);
      assert.ok(e, 'experiment ' + label);
      return { patch: e.patch ?? {}, say: e.say };
    };

    it('causal mask on reverse stalls at the quoted share', () => {
      const { patch, say } = run('Causal mask');
      const rig = fresh('reverse', patch);
      runTo(rig, 400);
      reads('Causal mask', say, pc(rig.state.testAccuracy));
    });

    it('one head, two layers, the widths, length and fewer sequences fit where the copy says', () => {
      for (const label of ['One head', 'Two layers', 'Width 8', 'Width 32', 'No layer norm', 'Eight digits long', '200 sequences']) {
        const { patch, say } = run(label);
        const at = fitAt('reverse', patch);
        assert.ok(at !== null, label + ' never fits');
        reads(label, say, 'step ' + at);
        if (label === 'Two layers' || label.startsWith('Width')) reads(label, say, n(parameterCount(fresh('reverse', patch).state.weights)) + ' parameters');
      }
    });

    it('plain SGD crawls', () => {
      const { patch, say } = run('Plain SGD');
      const rig = fresh('reverse', patch);
      runTo(rig, 800);
      reads('Plain SGD', say, pc(rig.state.testAccuracy) + ' after 800 steps');
    });
  });
});
