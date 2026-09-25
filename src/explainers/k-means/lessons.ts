/** k-means lessons, one screen each. Pure data, driven against the real model in tests/lessons-kmeans.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { PointData } from '../../lib/datasets/types';
import { fmt, fmtCompact, fmtPercent } from '../../lib/math/stats';
import {
  clusterSizes,
  createState,
  isComplete as kmeansIsComplete,
  silhouetteScore,
  step as kmeansStep,
} from '../../lib/ml/kmeans';
import type { KMeansConfig, KMeansState } from '../../lib/ml/kmeans';
import type { KMeansParams } from './config';

/* ---------------- context ---------------- */

export type SecondView = 'sizes' | 'elbow';

export interface Settled {
  inertia: number;
  rounds: number;
  sizes: number[];
  silhouette: number;
  purity: number;
}

export interface KMeansLessonContext extends LessonContextBase {
  params: KMeansParams;
  state: KMeansState;
  derived: {
    n: number;
    k: number;
    /** Full assign-and-update rounds so far. */
    rounds: number;
    sizes: number[];
    emptyCount: number;
    inertia: number;
    silhouette: number;
    /** Share of points sitting in a cluster whose majority true group is their own. */
    purity: number;
    /** The generator gave every point a true group. */
    hasTruth: boolean;
    trueGroups: number;
    /** Inertia around the overall mean: the k = 1 answer every other k is measured against. */
    k1Inertia: number;
    /** The same points run to convergence with a different k or seeding. */
    settle: (over: Partial<Pick<KMeansConfig, 'k' | 'init' | 'seed'>>) => Settled;
  };
  ui: {
    secondView: SecondView;
    /** The centroid whose card is open, or null. */
    open: number | null;
  };
}

export interface KMeansLessonInput {
  params: KMeansParams;
  state: KMeansState;
  data: PointData;
  config: KMeansConfig;
  bounds: { xRange: [number, number]; yRange: [number, number] };
  ui: KMeansLessonContext['ui'];
}

/** Share of points whose cluster mostly holds their own true group; NaN without truth. */
export function purityOf(data: PointData, assignments: readonly number[], k: number): number {
  const n = Math.min(data.points.length, assignments.length);
  if (n === 0 || data.classCount < 2) return Number.NaN;
  const table: number[][] = Array.from({ length: k }, () => new Array<number>(data.classCount).fill(0));
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const c = assignments[i];
    const label = data.points[i].label;
    if (c < 0 || c >= k || label < 0 || label >= data.classCount) continue;
    table[c][label] += 1;
    counted += 1;
  }
  if (counted === 0) return Number.NaN;
  let agree = 0;
  for (const row of table) agree += Math.max(...row);
  return agree / counted;
}

/** Run the whole algorithm to convergence, the way the page's transport would. */
export function settleRun(
  data: PointData,
  config: KMeansConfig,
  bounds: KMeansLessonInput['bounds'],
): Settled {
  let state = createState(data.points, config, bounds);
  for (let i = 0; i < 400; i++) {
    if (kmeansIsComplete(state, config.tolerance)) break;
    if (state.phase === 'update' && state.iteration > 0 && state.shift < config.tolerance) break;
    const next = kmeansStep(state, data.points);
    if (next === state) break;
    state = next;
  }
  const k = state.centroids.length;
  return {
    inertia: state.inertia,
    rounds: state.iteration,
    sizes: clusterSizes(state.assignments, k),
    silhouette: silhouetteScore(data.points, state.assignments, k),
    purity: purityOf(data, state.assignments, k),
  };
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeKMeansContext(input: KMeansLessonInput): Omit<KMeansLessonContext, keyof LessonContextBase> {
  const { params, state, data, config, bounds, ui } = input;
  const points = data.points;
  const k = state.centroids.length;
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x;
    my += p.y;
  }
  mx /= Math.max(1, points.length);
  my /= Math.max(1, points.length);
  let k1 = 0;
  for (const p of points) k1 += (p.x - mx) ** 2 + (p.y - my) ** 2;
  return {
    params,
    state,
    derived: {
      n: points.length,
      k,
      rounds: state.iteration,
      sizes: clusterSizes(state.assignments, k),
      emptyCount: state.emptyClusters.length,
      inertia: state.inertia,
      silhouette: silhouetteScore(points, state.assignments, k),
      purity: purityOf(data, state.assignments, k),
      hasTruth: data.classCount >= 2,
      trueGroups: data.classCount,
      k1Inertia: k1,
      settle: (over) => settleRun(data, { ...config, ...over }, bounds),
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = KMeansLessonContext;
type L = Lesson<Ctx, KMeansParams>;

const num = (v: number) => fmtCompact(v);
const pc = (v: number) => fmtPercent(v, 0);
const rounds = (n: number) => n + ' round' + (n === 1 ? '' : 's');
const sizes = (s: readonly number[]) => s.join(' / ');
const done = (c: Ctx) => c.sim.isComplete;

/** "converged after 4 rounds" or "stopped after 3 rounds", from the flag that ended the run. */
const ending = (c: Ctx) => (c.state.converged ? 'Converged after ' : 'Stopped after ') + rounds(c.derived.rounds);

/* ---------------- the lessons ---------------- */

type Act = LessonAction<KMeansParams>;

const SIZES: Act = { type: 'view', id: 'second', value: 'sizes' };
const ELBOW: Act = { type: 'view', id: 'second', value: 'elbow' };
const PLAY: Act = { type: 'play' };
/** Opens the lesson on its settled clustering. */
const SETTLE: Act = { type: 'runTo', steps: 400 };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<KMeansParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'k', value: String(c.derived.k) },
  { label: 'rounds', value: String(c.derived.rounds) },
  { label: 'inertia', value: num(c.derived.inertia) },
  { label: 'silhouette', value: fmt(c.derived.silhouette, 2) },
];

const withOwn = (c: Ctx) => (Number.isFinite(c.derived.purity) ? ', ' + pc(c.derived.purity) + ' of the points with their own group' : '');

const assignmentAndUpdate: L = {
  id: 'assignment-update',
  title: 'Assignment and update',
  hook: 'Assign points, move centroids, repeat.',
  section: 'hook',
  presetId: 'textbook',
  steps: [
    {
      id: 'rounds',
      kind: 'sandbox',
      say: '120 points, k = 3, seeded by k-means++ and settled: 4 rounds, inertia 147, sizes 39 / 40 / 41. Each round assigns every point to its nearest centroid, then moves each centroid to the mean of its points. The rings are the true groups. Try it: Assignment step.',
      takeaway: 'Inertia is the total squared distance from every point to its own centroid. Neither step can raise it, so the alternation settles within a handful of rounds.',
      focus: { kind: 'panel', id: 'architecture', label: 'The clustering' },
      enter: [{ type: 'preset', id: 'textbook' }, { type: 'params', patch: { showTruth: true } }, { type: 'reset' }, SIZES, SETTLE],
      numbers,
      experiments: [
        {
          label: 'Assignment step',
          say: 'From the seeds, one step.',
          enter: on('textbook', { showTruth: true }, SIZES, { type: 'step', count: 1 }),
          until: (c) => c.sim.iteration >= c.lesson.enteredAt + 1,
          then: (c) => 'Assignment: every point joins its nearest centroid, ' + sizes(c.derived.sizes) + '. Inertia ' + num(c.derived.inertia) + '.',
          focus: { kind: 'panel', id: 'architecture', label: 'The clustering' },
        },
        {
          label: 'Update step',
          say: 'The step after the first assignment.',
          enter: on('textbook', { showTruth: true }, SIZES, { type: 'step', count: 2 }),
          until: (c) => c.sim.iteration >= c.lesson.enteredAt + 2,
          then: (c) =>
            'Update: each centroid moves to the mean of its points, the position that minimises the squared distances to them. The furthest moved ' +
            fmt(c.state.shift, 2) + '; inertia ' + num(c.derived.inertia) + '.',
          focus: { kind: 'metric', key: 'shift', label: 'Centroid shift' },
        },
        {
          label: 'Run to convergence',
          say: 'Alternate until no point changes cluster.',
          enter: on('textbook', { showTruth: true }, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => ending(c) + ': no point changed cluster. Inertia ' + num(c.derived.inertia) + ', sizes ' + sizes(c.derived.sizes) + withOwn(c) + '.',
          focus: { kind: 'metric', key: 'inertia', label: 'Inertia' },
        },
      ],
    },
  ],
};

const initialisation: L = {
  id: 'initialisation',
  title: 'Initialisation',
  hook: 'Same points, started badly and well.',
  section: 'core',
  presetId: 'bad-seed',
  knobs: ['init', 'k'],
  steps: [
    {
      id: 'seeds',
      kind: 'sandbox',
      say: 'Three round groups, but every centroid started crowded into one corner of the data. Each step only goes downhill, so it settled on the nearest local minimum: 5 rounds, sizes 78 / 25 / 17 over three equal groups, inertia 706. Try it: k-means++.',
      takeaway: 'Where k-means lands depends on where it starts. Seed from the data, and in practice start several times and keep the lowest inertia.',
      focus: { kind: 'metric', key: 'inertia', label: 'Inertia' },
      enter: [{ type: 'preset', id: 'bad-seed' }, { type: 'reset' }, SIZES, SETTLE],
      numbers,
      experiments: [
        {
          label: 'k-means++',
          say: 'Each new seed is drawn with probability proportional to its squared distance from the seeds so far.',
          enter: on('kmeanspp-fix', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => {
            const bad = c.derived.settle({ init: 'forgy-far' });
            return 'Seeded by k-means++: ' + ending(c).toLowerCase() + ', sizes ' + sizes(c.derived.sizes) + ', inertia ' + num(c.derived.inertia) + ' against ' + num(bad.inertia) + ' from the corner start.';
          },
          focus: { kind: 'control', key: 'init', label: 'Seeding' },
        },
        {
          label: 'Corner start again',
          say: 'The same points from the crowded corner, live.',
          enter: on('bad-seed', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => 'From the corner: ' + ending(c).toLowerCase() + ', sizes ' + sizes(c.derived.sizes) + ', inertia ' + num(c.derived.inertia) + '. The algorithm has no way to know it is stuck.',
          focus: { kind: 'metric', key: 'inertia', label: 'Inertia' },
        },
        {
          label: 'Random coordinates, k = 6',
          say: 'Six centroids dropped anywhere in the box, some where there is no data.',
          enter: on('empty-cluster', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) =>
            ending(c) + ': ' + c.derived.emptyCount + ' of ' + c.derived.k + ' centroids caught nothing and stay where they were seeded, sizes ' + sizes(c.derived.sizes) +
            '. An empty cluster has no mean to move to.',
          focus: { kind: 'metric', key: 'smallest', label: 'Smallest cluster' },
        },
      ],
    },
  ],
};

const choosingK: L = {
  id: 'choosing-k',
  title: 'Choosing k',
  hook: 'Inertia cannot pick k; silhouette can.',
  section: 'core',
  presetId: 'too-many-k',
  knobs: ['k'],
  steps: [
    {
      id: 'k',
      kind: 'sandbox',
      say: 'Six centroids over three real groups: inertia 129, below the 171 that k = 3 reaches, because each real group was cut in two. Silhouette 0.38 against 0.71 at k = 3. Inertia falls with every k; silhouette does not. Try it: k = 3.',
      takeaway: 'Silhouette measures how much closer each point is to its own cluster than to the next. Pick k where the elbow bends and the silhouette peaks, not where inertia is lowest.',
      focus: { kind: 'metric', key: 'silhouette', label: 'Silhouette' },
      enter: [{ type: 'preset', id: 'too-many-k' }, { type: 'reset' }, SIZES, SETTLE],
      numbers,
      experiments: [
        {
          label: 'k = 3',
          say: 'One centroid per real group; the elbow chart runs every k from 1 to 10.',
          enter: on('too-many-k', { k: 3 }, ELBOW, PLAY),
          speed: 'normal',
          until: done,
          then: (c) =>
            'k = 3: inertia ' + num(c.derived.inertia) + ', silhouette ' + fmt(c.derived.silhouette, 2) + withOwn(c) +
            '. The elbow chart bends where another centroid stops paying for itself.',
          focus: { kind: 'panel', id: 'second', label: 'Choosing k' },
        },
        {
          label: 'k = 6 again',
          say: 'Two centroids per group, live.',
          enter: on('too-many-k', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => {
            const three = c.derived.settle({ k: 3 });
            return 'k = 6: inertia ' + num(c.derived.inertia) + ' against ' + num(three.inertia) + ' at k = 3, silhouette ' + fmt(c.derived.silhouette, 2) + ' against ' + fmt(three.silhouette, 2) + '. Lower inertia, worse clustering.';
          },
          focus: { kind: 'metric', key: 'silhouette', label: 'Silhouette' },
        },
        {
          label: 'Uniform noise, k = 4',
          say: 'Points with no groups at all.',
          enter: on('no-structure', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => ending(c) + ': four clusters of ' + sizes(c.derived.sizes) + ' points, silhouette ' + fmt(c.derived.silhouette, 2) + '. k-means always returns k clusters; a low silhouette is the only hint that the groups were never there.',
          focus: { kind: 'metric', key: 'silhouette', label: 'Silhouette' },
        },
      ],
    },
  ],
};

const clusterShape: L = {
  id: 'shapes',
  title: 'Cluster shape and spread',
  hook: 'Rings, stretched and unequal groups.',
  section: 'failure',
  presetId: 'rings',
  knobs: ['dataset'],
  steps: [
    {
      id: 'rings',
      kind: 'sandbox',
      say: 'Two concentric rings and k = 2. The boundary between two centroids is always a straight line, so each cluster holds about half of each ring: 50% of the points with their own group, whatever the start. Try it: Elongated groups.',
      takeaway: 'k-means puts the boundary halfway between two centroids, so it assumes round groups of similar size and spread.',
      focus: { kind: 'panel', id: 'architecture', label: 'The clustering' },
      enter: [{ type: 'preset', id: 'rings' }, { type: 'reset' }, SIZES, SETTLE],
      numbers: (c) => [...numbers(c), ...(Number.isFinite(c.derived.purity) ? [{ label: 'own group', value: pc(c.derived.purity) }] : [])],
      experiments: [
        {
          label: 'Elongated groups',
          say: 'Three long, tilted groups.',
          enter: on('stretched', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => ending(c) + withOwn(c) + '. The round regions slice across the long ones.',
          focus: { kind: 'panel', id: 'architecture', label: 'The clustering' },
        },
        {
          label: 'Unequal spread',
          say: 'One tight group beside two loose ones.',
          enter: on('varied', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => ending(c) + ': sizes ' + sizes(c.derived.sizes) + withOwn(c) + '. The tight group annexes part of a loose one.',
          focus: { kind: 'control', key: 'showTruth', label: 'True groups' },
        },
        {
          label: 'Rings again',
          say: 'The two rings, live.',
          enter: on('rings', null, SIZES, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => ending(c) + ': each cluster holds about half of each ring' + withOwn(c) + '. A ring inside a ring needs a curve.',
          focus: { kind: 'panel', id: 'architecture', label: 'The clustering' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Six experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'textbook',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'textbook' }, { type: 'reset' }, SIZES],
      experiments: [
        {
          label: 'k = 1',
          say: 'One centroid at the overall mean, done in one round. Its inertia is the baseline every other k is measured against.',
          patch: { k: 1 },
          focus: { kind: 'control', key: 'k', label: 'Clusters (k)' },
        },
        {
          label: 'Random coordinates',
          say: 'Seeds anywhere in the box. On these blobs it still finds the three groups; it takes a round or two longer.',
          patch: { init: 'random-coords' },
          focus: { kind: 'control', key: 'init', label: 'Seeding' },
        },
        {
          label: 'Spread 0.8',
          say: 'The groups overlap. Inertia stays high and about a tenth of the points end up with another group.',
          patch: { noise: 0.8, showTruth: true },
          focus: { kind: 'control', key: 'noise', label: 'Spread' },
        },
        {
          label: 'Five groups, k = 3',
          say: 'Five true groups and three centroids: neighbouring groups are merged, and the picture looks perfectly reasonable.',
          patch: { trueClusters: 5, k: 3, showTruth: true },
          focus: { kind: 'control', key: 'trueClusters', label: 'True groups' },
        },
        {
          label: 'Tolerance 1',
          say: 'A loose tolerance stops the run a round early, while the last centroid is still visibly moving and before any assignment has confirmed the answer.',
          patch: { tolerance: 1 },
          focus: { kind: 'control', key: 'tolerance', label: 'Convergence tolerance' },
        },
        {
          label: '400 points',
          say: 'Three times the data, the same handful of rounds. Each round costs n times k distances, which is why k-means scales.',
          patch: { sampleCount: 400 },
          focus: { kind: 'control', key: 'sampleCount', label: 'Points' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [assignmentAndUpdate, initialisation, choosingK, clusterShape, experiments];
