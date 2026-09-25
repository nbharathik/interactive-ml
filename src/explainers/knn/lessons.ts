/** k-nearest neighbours lessons, one screen each. Pure data, driven against the real model in tests/lessons-knn.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { Point2D, PointData } from '../../lib/datasets/types';
import { fmt, fmtPercent } from '../../lib/math/stats';
import { makeScaling, rankNeighbours, tally } from '../../lib/ml/knn';
import type { DistanceMetric, KnnConfig, Neighbour, Scaling, Vote } from '../../lib/ml/knn';
import type { KnnParams } from './config';

/* ---------------- context ---------------- */

export type SecondView = 'table' | 'k';

export interface KnnLessonContext extends LessonContextBase {
  params: KnnParams;
  state: { revealed: number };
  derived: {
    n: number;
    k: number;
    /** k, capped by the number of stored points. */
    countedK: number;
    /** Neighbours counted so far; the final k at rest. */
    shown: number;
    vote: Vote;
    finalVote: Vote;
    classNames: string[];
    /** Distance to the furthest counted neighbour. */
    radius: number;
    looAccuracy: number;
    trainAccuracy: number;
    bestK: number;
    bestAccuracy: number;
    /** Leave-one-out accuracy at another k on the same points. */
    looAt: (k: number) => number;
    /** What always calling the biggest class scores. */
    baseline: number;
    /** Share of the vote the winner leads the runner-up by. */
    margin: number;
    /** How many of the current k nearest are also among the k nearest under another metric. */
    sharedWith: (metric: DistanceMetric) => number;
    /** Leave-one-out accuracy under other settings, on the same stored points. */
    looUnder: (over: Partial<KnnConfig>) => number;
    /** Leave-one-out recall of one class under the current settings. */
    looRecall: (label: number) => number;
  };
  ui: {
    secondView: SecondView;
    queryOpen: boolean;
    /** Times the query was dragged since the step was entered. */
    queryMoves: number;
    /** Stored points added or removed since the step was entered. */
    pointEdits: number;
  };
}

export interface KnnLessonInput {
  params: KnnParams;
  state: { revealed: number };
  /** The stored points as plotted, with any unit stretch applied. */
  data: PointData;
  config: KnnConfig;
  scaling: Scaling;
  query: { x: number; y: number };
  /** The page's leave-one-out curve over k. */
  curve: ReadonlyArray<{ k: number; accuracy: number }>;
  ui: KnnLessonContext['ui'];
}

/** Leave-one-out predictions for every stored point. */
function looPredictions(points: readonly Point2D[], config: KnnConfig, scaling: Scaling): number[] {
  return points.map((p, i) => {
    const ranked = rankNeighbours(points, p.x, p.y, config, scaling)
      .filter((n) => n.index !== i)
      .slice(0, Math.max(1, config.k));
    return tally(ranked, config.classCount).winner;
  });
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeKnnContext(input: KnnLessonInput): Omit<KnnLessonContext, keyof LessonContextBase> {
  const { params, state, data, config, scaling, query, curve, ui } = input;
  const points = data.points;
  const ranked: Neighbour[] = rankNeighbours(points, query.x, query.y, config, scaling);
  const countedK = Math.min(config.k, ranked.length);
  const shown = state.revealed === 0 ? countedK : Math.min(state.revealed, countedK);
  const counted = ranked.slice(0, shown);
  const vote = tally(counted, config.classCount);
  const finalVote = tally(ranked.slice(0, countedK), config.classCount);
  const sorted = [...vote.scores].sort((a, b) => b - a);
  const total = sorted.reduce((a, b) => a + b, 0);
  const counts = new Array<number>(Math.max(1, config.classCount)).fill(0);
  for (const p of points) if (p.label >= 0 && p.label < counts.length) counts[p.label] += 1;
  const classNames = Array.from(
    { length: config.classCount },
    (_, i) => data.classNames[i] ?? 'Class ' + String.fromCharCode(65 + i),
  );
  const loo = curve.length > 0 ? curve[Math.min(config.k, curve.length) - 1].accuracy : 0;
  const best = curve.reduce((b, e) => (e.accuracy > b.accuracy ? e : b), { k: 1, accuracy: 0 });
  const nearestIds = new Set(ranked.slice(0, countedK).map((n) => n.index));
  return {
    params,
    state,
    derived: {
      n: points.length,
      k: config.k,
      countedK,
      shown,
      vote,
      finalVote,
      classNames,
      radius: counted.length > 0 ? counted[counted.length - 1].distance : 0,
      looAccuracy: loo,
      trainAccuracy:
        points.length === 0
          ? 0
          : points.filter((p) => tally(rankNeighbours(points, p.x, p.y, config, scaling).slice(0, config.k), config.classCount).winner === p.label).length /
            points.length,
      bestK: best.k,
      bestAccuracy: best.accuracy,
      looAt: (k) => (curve.length > 0 ? curve[Math.min(Math.max(1, k), curve.length) - 1].accuracy : 0),
      baseline: points.length === 0 ? 0 : Math.max(...counts) / points.length,
      margin: total > 0 ? (sorted[0] - (sorted[1] ?? 0)) / total : 0,
      sharedWith: (metric) =>
        rankNeighbours(points, query.x, query.y, { ...config, metric }, scaling)
          .slice(0, countedK)
          .filter((n) => nearestIds.has(n.index)).length,
      looUnder: (over) => {
        const cfg = { ...config, ...over };
        const scale = makeScaling(points, cfg.standardise);
        const predicted = looPredictions(points, cfg, scale);
        return points.length === 0 ? 0 : predicted.filter((label, i) => label === points[i].label).length / points.length;
      },
      looRecall: (label) => {
        const predicted = looPredictions(points, config, scaling);
        const own = points.map((p, i) => [p.label, predicted[i]] as const).filter(([truth]) => truth === label);
        return own.length === 0 ? 0 : own.filter(([, guess]) => guess === label).length / own.length;
      },
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = KnnLessonContext;
type L = Lesson<Ctx, KnnParams>;

const pc = (v: number) => fmtPercent(v, 0);
const name = (c: Ctx, label: number) => c.derived.classNames[label] ?? 'Class ' + (label + 1);
const tallyText = (c: Ctx, v: Vote) =>
  v.scores.map((score, label) => name(c, label) + ' ' + (c.params.weighting === 'distance' ? fmt(score, 2) : String(score))).join(', ');
const winner = (c: Ctx, v: Vote) => (v.tied ? 'a tie' : name(c, v.winner) + ' with ' + pc(v.confidence) + ' of the vote');

/* ---------------- the lessons ---------------- */

type Act = LessonAction<KnnParams>;

const TABLE: Act = { type: 'view', id: 'second', value: 'table' };
const WHICH_K: Act = { type: 'view', id: 'second', value: 'k' };
/** Counts every neighbour at once, so the lesson opens on the finished vote. */
const COUNT: Act = { type: 'runTo', steps: 100 };
const counted = (c: Ctx) => c.sim.isComplete;

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh count. */
const on = (preset: string, patch: Partial<KnnParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'k', value: String(c.derived.k) },
  { label: 'prediction', value: c.derived.finalVote.tied ? 'tie' : name(c, c.derived.finalVote.winner) },
  { label: 'leave-one-out', value: pc(c.derived.looAccuracy) },
];

const majorityVote: L = {
  id: 'majority-vote',
  title: 'Majority vote',
  hook: 'The k nearest stored points vote.',
  section: 'hook',
  presetId: 'walkthrough',
  knobs: ['k'],
  steps: [
    {
      id: 'vote',
      kind: 'sandbox',
      say: '90 stored points in 3 classes and one query. Nothing is fitted: the stored points are the whole model. The 7 nearest, joined by lines, vote: Class C 7 of 7, so the query is called Class C. The circle reaches the 7th nearest point. Try it: Watch the count.',
      takeaway: 'k-NN spends nothing on training and everything on prediction: every query is compared with every stored point, and the k nearest vote.',
      focus: { kind: 'panel', id: 'architecture', label: 'The neighbourhood' },
      enter: [{ type: 'preset', id: 'walkthrough' }, { type: 'reset' }, TABLE, COUNT],
      numbers,
      experiments: [
        {
          label: 'Watch the count',
          say: 'The nearest join the tally first; the table is the ranking itself.',
          enter: on('walkthrough', null, TABLE, { type: 'play' }),
          speed: 'normal',
          until: counted,
          then: (c) => 'All ' + c.derived.countedK + ' counted: ' + tallyText(c, c.derived.vote) + '. The prediction is ' + winner(c, c.derived.vote) + '.',
          focus: { kind: 'panel', id: 'vote', label: 'The vote' },
        },
        {
          label: 'k = 3',
          say: 'Only the three nearest vote.',
          enter: on('walkthrough', { k: 3 }, TABLE, COUNT),
          until: counted,
          then: (c) => 'k = 3: ' + tallyText(c, c.derived.vote) + '. The prediction is ' + winner(c, c.derived.vote) + ', and the circle shrinks to the 3rd nearest point.',
          focus: { kind: 'control', key: 'k', label: 'Neighbours (k)' },
        },
      ],
    },
  ],
};

const choosingK: L = {
  id: 'choosing-k',
  title: 'Choosing k',
  hook: 'k = 1 memorises, k = 45 over-smooths.',
  section: 'core',
  presetId: 'k-one',
  knobs: ['k', 'weighting'],
  steps: [
    {
      id: 'k',
      kind: 'sandbox',
      say: 'k = 1: every stored point owns the patch around it, stray points included. Training accuracy reads 100%, because each point is its own nearest neighbour; leave-one-out, which hides each point from itself, 98%. The chart plots it for every k. Try it: k = 45.',
      takeaway: 'Choose k by held-out accuracy, never by training accuracy. A large k averages so many neighbours that the local shape is voted away; distance weighting lets it keep local detail.',
      focus: { kind: 'panel', id: 'second', label: 'Which k?' },
      enter: [{ type: 'preset', id: 'k-one' }, { type: 'reset' }, WHICH_K, COUNT],
      numbers: (c) => [{ label: 'k', value: String(c.derived.k) }, { label: 'training', value: pc(c.derived.trainAccuracy) }, { label: 'leave-one-out', value: pc(c.derived.looAccuracy) }],
      experiments: [
        {
          label: 'k = 45',
          say: 'The two moons blur into a nearly straight split.',
          enter: on('k-large', null, WHICH_K, COUNT),
          until: counted,
          then: (c) => 'k = ' + c.derived.k + ': leave-one-out ' + pc(c.derived.looAccuracy) + ', and at k = 59 it is ' + pc(c.derived.looAt(59)) + '. The neighbourhood spans both moons.',
          focus: { kind: 'control', key: 'k', label: 'Neighbours (k)' },
        },
        {
          label: 'k = 45, votes worth 1/d²',
          say: 'The same k with distance weighting.',
          enter: on('weighted-rescue', null, WHICH_K, COUNT),
          until: counted,
          then: (c) => 'k = ' + c.derived.k + ' with each vote worth 1/d²: leave-one-out ' + pc(c.derived.looAccuracy) + '. The near neighbours keep their say.',
          focus: { kind: 'control', key: 'weighting', label: 'Vote weighting' },
        },
        {
          label: 'k = 1 again',
          say: 'Every point its own patch.',
          enter: on('k-one', null, WHICH_K, COUNT),
          until: counted,
          then: (c) => 'k = 1: training accuracy ' + pc(c.derived.trainAccuracy) + ', leave-one-out ' + pc(c.derived.looAccuracy) + '. Stray points get patches of their own.',
          focus: { kind: 'panel', id: 'second', label: 'Which k?' },
        },
      ],
    },
  ],
};

const distanceAndScaling: L = {
  id: 'distance',
  title: 'Distance metric and scaling',
  hook: 'Metric and units decide who is near.',
  section: 'core',
  presetId: 'manhattan',
  knobs: ['metric', 'xStretch', 'standardise'],
  steps: [
    {
      id: 'metric',
      kind: 'sandbox',
      say: 'Manhattan distance adds the gaps along each axis, so the neighbourhood is a diamond: 3 of these 5 neighbours are also among the 5 nearest by Euclidean distance. The metric decides who counts as a neighbour, not just the boundary. Try it: Unequal units.',
      takeaway: 'Distance adds units together, so a feature on a larger scale gets a larger say for no reason. Scale the axes before measuring.',
      focus: { kind: 'control', key: 'metric', label: 'Distance metric' },
      enter: [{ type: 'preset', id: 'manhattan' }, { type: 'reset' }, TABLE, COUNT],
      numbers: (c) => [{ label: 'metric', value: c.params.metric }, { label: 'same as euclidean', value: c.derived.sharedWith('euclidean') + ' / ' + c.derived.countedK }, { label: 'leave-one-out', value: pc(c.derived.looAccuracy) }],
      experiments: [
        {
          label: 'x in units 20 times larger',
          say: 'Euclidean distance, as if kilometres against metres.',
          enter: on('bad-units', null, TABLE, COUNT),
          until: counted,
          then: (c) => 'x stretched ' + c.params.xStretch + ' times: the regions collapse into vertical stripes and leave-one-out falls to ' + pc(c.derived.looAccuracy) + '.',
          focus: { kind: 'control', key: 'xStretch', label: 'Stretch x (×)' },
        },
        {
          label: 'Standardise the axes',
          say: 'Each axis divided by its spread before measuring.',
          enter: on('bad-units', { standardise: true }, TABLE, COUNT),
          until: counted,
          then: (c) => 'Standardised: leave-one-out back to ' + pc(c.derived.looAccuracy) + '. The stretch no longer matters.',
          focus: { kind: 'control', key: 'standardise', label: 'Standardise axes' },
        },
        {
          label: 'Chebyshev',
          say: 'Only the largest single-axis gap counts: a square neighbourhood.',
          enter: on('chebyshev', null, TABLE, COUNT),
          until: counted,
          then: (c) => 'Chebyshev: ' + c.derived.sharedWith('euclidean') + ' of these ' + c.derived.countedK + ' neighbours are also among the Euclidean ' + c.derived.countedK + '. A different set of points gets counted.',
          focus: { kind: 'control', key: 'metric', label: 'Distance metric' },
        },
      ],
    },
  ],
};

const imbalanceAndTies: L = {
  id: 'imbalance-ties',
  title: 'Class imbalance and ties',
  hook: 'A rare class outvoted, an even k tied.',
  section: 'failure',
  presetId: 'rare-class',
  knobs: ['k'],
  steps: [
    {
      id: 'rare',
      kind: 'sandbox',
      say: 'Only 12% of the points are positive. At k = 35 leave-one-out recall for them is 0%: the majority outvotes them everywhere, and accuracy still looks fine. A small k lets a small class win where it is locally dense. Try it: k = 5.',
      takeaway: 'A large k on imbalanced data predicts the majority class almost everywhere. With two classes use an odd k, so a vote can never deadlock.',
      focus: { kind: 'metric', key: 'loo', label: 'Leave-one-out' },
      enter: [{ type: 'preset', id: 'rare-class' }, { type: 'reset' }, WHICH_K, COUNT],
      numbers: (c) => [{ label: 'k', value: String(c.derived.k) }, { label: 'rare-class recall', value: pc(c.derived.looRecall(1)) }, { label: 'leave-one-out', value: pc(c.derived.looAccuracy) }],
      experiments: [
        {
          label: 'k = 5',
          say: 'A neighbourhood small enough for the rare class.',
          enter: on('rare-class', { k: 5 }, WHICH_K, COUNT),
          until: counted,
          then: (c) => 'k = ' + c.derived.k + ': recall for the rare class ' + pc(c.derived.looRecall(1)) + ', leave-one-out ' + pc(c.derived.looAccuracy) + '.',
          focus: { kind: 'control', key: 'k', label: 'Neighbours (k)' },
        },
        {
          label: 'Tied vote, k = 4',
          say: 'Two classes, an even k, the query on the border.',
          enter: on('tie', null, TABLE, COUNT),
          until: counted,
          then: (c) =>
            'k = ' + c.derived.k + ': ' + (c.derived.finalVote.tied ? 'tied, ' + tallyText(c, c.derived.finalVote) + '. An implementation has to break this arbitrarily.' : tallyText(c, c.derived.finalVote) + '.'),
          focus: { kind: 'panel', id: 'vote', label: 'The vote' },
        },
        {
          label: 'k = 35 again',
          say: 'The rare class outvoted.',
          enter: on('rare-class', null, WHICH_K, COUNT),
          until: counted,
          then: (c) => 'k = ' + c.derived.k + ': recall for the rare class ' + pc(c.derived.looRecall(1)) + ', leave-one-out ' + pc(c.derived.looAccuracy) + '. Accuracy hides the miss.',
          focus: { kind: 'metric', key: 'loo', label: 'Leave-one-out' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Seven experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'walkthrough',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change, the count replays, and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'walkthrough' }, { type: 'reset' }, WHICH_K],
      experiments: [
        {
          label: 'Spiral at k = 3',
          say: 'A small k follows the arms and every stray point too. Leave-one-out stays high because the arms are dense.',
          patch: { dataset: 'spiral', sampleCount: 200, noise: 0.12, k: 3 },
          focus: { kind: 'control', key: 'dataset', label: 'Dataset' },
        },
        {
          label: 'Spiral at k = 45',
          say: 'The same spiral at k = 45: the neighbourhood spans both arms, the boundary smears, and leave-one-out collapses to about 40%.',
          patch: { dataset: 'spiral', sampleCount: 200, noise: 0.12, k: 45 },
          focus: { kind: 'control', key: 'k', label: 'Neighbours (k)' },
        },
        {
          label: '300 stored points',
          say: 'Three times the points: every query now costs 300 distances, and the regions grow smoother edges.',
          patch: { sampleCount: 300 },
          focus: { kind: 'control', key: 'sampleCount', label: 'Stored points' },
        },
        {
          label: 'Noise 1.0',
          say: 'Heavy noise: the classes interleave and the best k moves up, because averaging over more neighbours cancels more noise.',
          patch: { noise: 1 },
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'Chebyshev',
          say: 'Chebyshev counts only the largest single-axis gap, so the neighbourhood is a square and a different set of points gets counted.',
          patch: { dataset: 'gaussians', sampleCount: 80, k: 5, metric: 'chebyshev', queryX: 2.6, queryY: -2 },
          focus: { kind: 'control', key: 'metric', label: 'Distance metric' },
        },
        {
          label: 'Minkowski, p = 3',
          say: 'Between Euclidean and Chebyshev: a square with rounded corners, and a neighbour list between the two.',
          patch: { metric: 'minkowski3' },
          focus: { kind: 'control', key: 'metric', label: 'Distance metric' },
        },
        {
          label: 'Four classes',
          say: 'Four classes, same vote. The count runs over four buckets and nothing else changes.',
          patch: { dataset: 'three-class', classCount: 4 },
          focus: { kind: 'control', key: 'classCount', label: 'Classes' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [majorityVote, choosingK, distanceAndScaling, imbalanceAndTies, experiments];
