/** K-means: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLUSTERING_DATASETS } from '../../lib/datasets/points';
import { fmtKnob } from '../../lib/math/stats';

/** Datasets whose generator honours a requested number of true groups. */
const TUNABLE_TRUTH = new Set(['blobs', 'varied', 'anisotropic']);

export interface KMeansParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  trueClusters: number;
  seed: number;

  k: number;
  init: string;
  tolerance: number;

  showVoronoi: boolean;
  showTrails: boolean;
  showLinks: boolean;
  showTruth: boolean;
}

export const DEFAULT_PARAMS: KMeansParams = {
  dataset: 'blobs',
  sampleCount: 120,
  noise: 0.25,
  trueClusters: 3,
  seed: 27,

  k: 3,
  init: 'kmeans++',
  tolerance: 0.01,

  showVoronoi: true,
  showTrails: true,
  showLinks: true,
  showTruth: false,
};

export const CONTROL_GROUPS: ControlGroup<keyof KMeansParams & string>[] = [
  {
    title: 'Data',
    blurb: 'The algorithm never sees the true groups. You can, with the display toggles.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'Round blobs is the shape k-means assumes. Every other option breaks one of its assumptions on purpose.',
        options: CLUSTERING_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 20,
        max: 400,
        step: 10,
        help: 'How many observations to cluster. The cost of a round is linear in this number, which is why k-means scales to data other methods choke on.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Spread',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'This page\'s noise knob: how widely each true group scatters. Push it up and the groups overlap, so there stops being a single right answer.',
      },
      {
        kind: 'segmented',
        key: 'trueClusters',
        label: 'True groups',
        options: [2, 3, 4, 5].map((n) => ({ value: String(n), label: String(n) })),
        visibleWhen: (p) => TUNABLE_TRUTH.has(String(p.dataset)),
        help: 'How many groups the generator actually makes. Set k to something else and watch what the algorithm does with the mismatch.',
      },
      {
        kind: 'seed',
        key: 'seed',
        label: 'Seed',
        help: 'Fixes both the data and the starting centroids. Re-roll it to get a different run on the same kind of data.',
      },
    ],
  },
  {
    title: 'Algorithm',
    zone: 'top',
    controls: [
      {
        kind: 'stepper',
        key: 'k',
        label: 'Clusters (k)',
        min: 1,
        max: 10,
        step: 1,
        help: 'You pick this before running. The algorithm returns exactly this many clusters, whether or not that many exist in the data.',
      },
      {
        kind: 'select',
        key: 'init',
        label: 'Seeding',
        help: 'Where the centroids start. This one choice decides which answer the run lands on more often than anything else here.',
        options: [
          {
            value: 'kmeans++',
            label: 'k-means++',
            hint: 'Each new seed is drawn with probability proportional to its squared distance from the seeds already chosen.',
          },
          {
            value: 'random-points',
            label: 'Random data points',
            hint: 'Forgy seeding: k of the observations, picked at random. Two seeds can land inside the same group.',
          },
          {
            value: 'random-coords',
            label: 'Random coordinates',
            hint: 'Uniform inside the bounding box, so a centroid can start where there is no data at all.',
          },
          {
            value: 'forgy-far',
            label: 'All in one corner',
            hint: 'Deliberately bad: every seed is crowded around the point furthest from the overall mean.',
          },
        ],
      },
      {
        kind: 'stepper',
        key: 'tolerance',
        advanced: true,
        label: 'Convergence tolerance',
        min: 0.0001,
        max: 1,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'Stop once no centroid moves further than this in one update. Loose values stop while the picture is still visibly changing; tight ones run extra rounds that change nothing you can see.',
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showVoronoi',
        label: 'Voronoi regions',
        help: 'Tints every part of the plane by which centroid is nearest. These regions are what the assignment step reads off.',
      },
      {
        kind: 'toggle',
        key: 'showTrails',
        label: 'Centroid trails',
        help: 'The path each centroid has taken since seeding. A long trail means the seed started far from where it ended up.',
      },
      {
        kind: 'toggle',
        key: 'showLinks',
        label: 'Assignment lines',
        help: 'A thin line from each point to the centroid it belongs to. The total squared length of these lines is the inertia.',
      },
      {
        kind: 'toggle',
        key: 'showTruth',
        label: 'True groups',
        help: 'Draws a ring around each point in the colour of the group it was generated from. The algorithm never gets to see this.',
      },
    ],
  },
];

export const PRESETS: Preset<KMeansParams>[] = [
  {
    id: 'textbook',
    name: 'The case it was built for',
    blurb: 'Three round, well-separated groups. Four rounds, and nearly every point ends up where it was generated.',
    params: { dataset: 'blobs', trueClusters: 3, k: 3, init: 'kmeans++', seed: 27 },
    autoRun: true,
  },
  {
    id: 'bad-seed',
    name: 'A start that ruins the run',
    blurb: 'Every seed crowded into one corner. It converges to a split of 78 / 25 / 17 over three equal groups.',
    params: { dataset: 'blobs', trueClusters: 3, k: 3, init: 'forgy-far', seed: 21 },
    autoRun: true,
  },
  {
    id: 'kmeanspp-fix',
    name: 'The same data, seeded well',
    blurb: 'The same points as above, seeded by k-means++. Two rounds, and a fifth of the inertia.',
    params: { dataset: 'blobs', trueClusters: 3, k: 3, init: 'kmeans++', seed: 21 },
    autoRun: true,
  },
  {
    id: 'too-many-k',
    name: 'k set too high',
    blurb: 'Six clusters over three real groups. Inertia is lower than at k = 3, and the answer is worse.',
    params: { dataset: 'blobs', trueClusters: 3, k: 6, init: 'kmeans++', seed: 7, showTruth: true },
    autoRun: true,
  },
  {
    id: 'empty-cluster',
    name: 'A cluster with nothing in it',
    blurb: 'Six seeds dropped anywhere in the box. Two of them catch nothing at all and are left stranded.',
    params: { dataset: 'blobs', trueClusters: 3, k: 6, init: 'random-coords', seed: 12 },
    autoRun: true,
  },
  {
    id: 'rings',
    name: 'Rings it cannot see',
    blurb: 'Two concentric rings, k = 2. Every boundary is a straight line, so each cluster ends up with half of each ring.',
    params: { dataset: 'rings', k: 2, init: 'kmeans++', sampleCount: 200, seed: 42, showTruth: true },
    autoRun: true,
  },
  {
    id: 'stretched',
    name: 'Stretched clusters',
    blurb: 'Long, rotated groups. The round regions slice across them and about a fifth of the points end up in the wrong cluster.',
    params: { dataset: 'anisotropic', trueClusters: 3, k: 3, init: 'kmeans++', seed: 42, showTruth: true },
    autoRun: true,
  },
  {
    id: 'no-structure',
    name: 'Nothing to find',
    blurb: 'Uniform noise, k = 4. It returns four confident, evenly-sized clusters anyway.',
    params: { dataset: 'uniform', k: 4, init: 'kmeans++', sampleCount: 200, seed: 42 },
    autoRun: true,
  },
  {
    id: 'varied',
    name: 'One tight group, one loose',
    blurb: 'Three groups with very different spreads. The tight one annexes a chunk of the loose one.',
    params: { dataset: 'varied', trueClusters: 3, noise: 0.5, k: 3, init: 'kmeans++', seed: 42, showTruth: true },
    autoRun: true,
  },
];
