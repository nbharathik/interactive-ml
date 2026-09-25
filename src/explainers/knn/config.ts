/** k-nearest neighbours: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLASSIFICATION_DATASETS, POINT_RANGE } from '../../lib/datasets/points';
import { METRIC_LABELS } from '../../lib/ml/knn';
import type { DistanceMetric } from '../../lib/ml/knn';

export interface KnnParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  classCount: number;
  seed: number;

  k: number;
  metric: string;
  weighting: string;
  standardise: boolean;
  /** Multiplies every x, simulating a feature measured in a larger unit. */
  xStretch: number;

  /** The query point, in the dataset's own units before any stretch is applied. */
  queryX: number;
  queryY: number;

  /** Class assigned to points added by clicking. */
  paintClass: string;
  clickAdds: boolean;

  showRegions: boolean;
  showLinks: boolean;
  showBall: boolean;
}

export const DEFAULT_PARAMS: KnnParams = {
  dataset: 'three-class',
  sampleCount: 90,
  noise: 0.4,
  classCount: 3,
  seed: 42,

  k: 7,
  metric: 'euclidean',
  weighting: 'uniform',
  standardise: false,
  xStretch: 1,

  queryX: 0.6,
  queryY: 0.4,

  paintClass: '0',
  clickAdds: true,

  showRegions: true,
  showLinks: true,
  showBall: true,
};

const METRIC_HINTS: Record<DistanceMetric, string> = {
  euclidean: 'Straight-line distance. Everything within a fixed distance of the query forms a circle.',
  manhattan: 'Add up the gap along each axis separately. The neighbourhood becomes a diamond.',
  chebyshev: 'Only the largest single-axis gap counts. The neighbourhood becomes a square.',
  minkowski3: 'Between Euclidean and Chebyshev. The neighbourhood is a square with rounded corners.',
};

const METRIC_ORDER: DistanceMetric[] = ['euclidean', 'manhattan', 'chebyshev', 'minkowski3'];

export const CONTROL_GROUPS: ControlGroup<keyof KnnParams & string>[] = [
  {
    title: 'Data',
    blurb: 'These points are the model. Nothing else is stored.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'k-NN decides locally, so ring and checkerboard shapes cost it nothing. Only the spiral is genuinely hard.',
        options: CLASSIFICATION_DATASETS.map((d) => ({
          value: d.id,
          label: d.name,
          hint: d.blurb,
        })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Stored points',
        min: 20,
        max: 300,
        step: 10,
        help: 'Every one of these is compared against the query on every prediction, so each stored point costs one more distance per query.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.01,
        format: (v) => v.toFixed(2),
        help: 'How far the classes bleed into each other. Noise is what a small k memorises.',
      },
      {
        kind: 'segmented',
        key: 'classCount',
        label: 'Classes',
        options: [
          { value: '3', label: '3' },
          { value: '4', label: '4' },
        ],
        visibleWhen: (p) => p.dataset === 'three-class',
        help: 'k-NN handles more than two classes without any change: the vote is a count over more buckets.',
      },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Same seed, same points, every time.' },
    ],
  },
  {
    title: 'Neighbours',
    zone: 'top',
    blurb: 'The two decisions that make a prediction.',
    controls: [
      {
        kind: 'stepper',
        key: 'k',
        label: 'Neighbours (k)',
        min: 1,
        max: 60,
        step: 1,
        help: 'At k = 1 the prediction copies the single closest point. As k grows the neighbourhood widens and the boundary smooths, until small classes are outvoted everywhere.',
      },
      {
        kind: 'select',
        key: 'metric',
        label: 'Distance metric',
        help: 'This is the definition of "close". Changing it changes which points are even in the neighbourhood, not only how the boundary looks.',
        options: METRIC_ORDER.map((id) => ({
          value: id,
          label: METRIC_LABELS[id],
          hint: METRIC_HINTS[id],
        })),
      },
      {
        kind: 'segmented',
        key: 'weighting',
        label: 'Vote weighting',
        help: 'Whether every neighbour counts the same, or the nearer ones count for more.',
        options: [
          { value: 'uniform', label: 'Uniform', hint: 'One vote each. A neighbour at the edge of the circle counts as much as one touching the query.' },
          { value: 'distance', label: 'By 1/d²', hint: 'Closer neighbours count for more, so a large k stops blurring the boundary.' },
        ],
      },
      {
        kind: 'toggle',
        key: 'standardise',
        advanced: true,
        label: 'Standardise axes',
        help: 'Divides each axis by its standard deviation before measuring distance. With both axes on the same range this changes nothing, stretch x first, then turn it on and watch it undo the damage.',
      },
      {
        kind: 'stepper',
        key: 'xStretch',
        advanced: true,
        label: 'Stretch x (×)',
        min: 1,
        max: 30,
        step: 1,
        format: (v) => '×' + v.toFixed(0),
        help: 'Multiplies every x by this factor, as if the first feature were recorded in kilometres and the second in metres. Distance then notices almost nothing but x.',
      },
    ],
  },
  {
    title: 'Query point',
    blurb: 'Drag it on the chart, or move it here.',
    controls: [
      {
        kind: 'slider',
        key: 'queryX',
        label: 'Query x',
        min: POINT_RANGE[0],
        max: POINT_RANGE[1],
        step: 0.1,
        format: (v) => v.toFixed(1),
        help: 'Horizontal position of the point being classified, before the unit stretch is applied.',
      },
      {
        kind: 'slider',
        key: 'queryY',
        label: 'Query y',
        min: POINT_RANGE[0],
        max: POINT_RANGE[1],
        step: 0.1,
        format: (v) => v.toFixed(1),
        help: 'Vertical position of the point being classified.',
      },
    ],
  },
  {
    title: 'Editing',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'clickAdds',
        label: 'Click adds a point',
        help: 'With this on, clicking empty space stores a new training point there. Shift-click always deletes the nearest one.',
      },
      {
        kind: 'segmented',
        key: 'paintClass',
        label: 'New points join',
        help: 'The class a point added by clicking is stored under.',
        visibleWhen: (p) => p.clickAdds === true,
        options: [
          { value: '0', label: 'A', hint: 'Drop one deep inside another class to carve an island out of the boundary.' },
          { value: '1', label: 'B', hint: 'Drop one deep inside another class to carve an island out of the boundary.' },
          { value: '2', label: 'C', hint: 'On a two-class dataset this creates a third class from nothing, k-NN needs no retraining for it.' },
        ],
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showRegions',
        label: 'Decision regions',
        help: 'Colours every point of the plane by what k-NN would predict there. Recomputed from scratch whenever k, the metric or the data changes.',
      },
      {
        kind: 'toggle',
        key: 'showLinks',
        label: 'Neighbour lines',
        help: 'Draws a line from the query to each neighbour that is voting. With 1/d² weighting the line thickness is that neighbour’s share of the vote.',
      },
      {
        kind: 'toggle',
        key: 'showBall',
        label: 'Neighbourhood outline',
        help: 'The set of points exactly as far from the query as the last counted neighbour. Its shape is the metric made visible.',
      },
    ],
  },
];

export const PRESETS: Preset<KnnParams>[] = [
  {
    id: 'walkthrough',
    name: 'Count the votes one by one',
    blurb: 'Press play: neighbours join the tally nearest-first, and the outline grows to reach each one.',
    params: { dataset: 'three-class', sampleCount: 90, k: 7, queryX: 0.6, queryY: 0.4 },
    autoRun: true,
  },
  {
    id: 'k-one',
    name: 'k = 1 memorises',
    blurb: 'Every stored point owns a patch of the plane, noise included. Training accuracy reads 100% and means nothing.',
    params: { dataset: 'moons', sampleCount: 140, noise: 0.35, k: 1 },
  },
  {
    id: 'k-large',
    name: 'k = 45 over-smooths',
    blurb: 'Same points, one dial moved. The two moons blur into a straight-ish split.',
    params: { dataset: 'moons', sampleCount: 140, noise: 0.35, k: 45 },
  },
  {
    id: 'weighted-rescue',
    name: 'Large k, distance weighted',
    blurb: 'k = 45 again, but each vote is worth 1/d². Close neighbours keep their say and the curve comes back.',
    params: { dataset: 'moons', sampleCount: 140, noise: 0.35, k: 45, weighting: 'distance' },
  },
  {
    id: 'rare-class',
    name: 'The rare class disappears',
    blurb: '12% of the points are positive. At k = 35 no region of the plane predicts them at all.',
    params: { dataset: 'imbalanced', sampleCount: 150, k: 35, queryX: 2.6, queryY: 2.2 },
  },
  {
    id: 'tie',
    name: 'An even k that can tie',
    blurb: 'k = 4 on two classes, with the query on the border where the vote deadlocks two-all.',
    params: { dataset: 'gaussians', sampleCount: 80, noise: 0.45, k: 4, queryX: 0, queryY: 0.4 },
  },
  {
    id: 'manhattan',
    name: 'Manhattan makes diamonds',
    blurb: 'Same query, same k. Only the definition of close changed, and the neighbour list changed with it.',
    params: { dataset: 'gaussians', sampleCount: 80, k: 5, metric: 'manhattan', queryX: 2.6, queryY: -2 },
  },
  {
    id: 'chebyshev',
    name: 'Chebyshev makes squares',
    blurb: 'Only the largest single-axis gap counts, so the neighbourhood is a box and diagonal points are cheap.',
    params: { dataset: 'gaussians', sampleCount: 80, k: 5, metric: 'chebyshev', queryX: 2.6, queryY: -2 },
  },
  {
    id: 'bad-units',
    name: 'One feature swamps the other',
    blurb: 'x recorded in units 20× larger, standardisation off. The regions collapse into vertical stripes.',
    params: { dataset: 'gaussians', sampleCount: 100, k: 9, xStretch: 20, standardise: false },
  },
  {
    id: 'spiral',
    name: 'The spiral, at k = 3',
    blurb: 'Only a small k can follow the arms, and a small k also follows every stray point.',
    params: { dataset: 'spiral', sampleCount: 200, noise: 0.12, k: 3 },
  },
];
