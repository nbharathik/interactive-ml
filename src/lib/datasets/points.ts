/** 2-D point datasets for the classifiers and clustering. Coordinates stay in [-6, 6]. */

import { gauss, makeRng, shuffled, uniform } from '../math/rng';
import type { DatasetOption, Point2D, PointData } from './types';

const RANGE: [number, number] = [-6, 6];

function finish(
  points: Point2D[],
  classCount: number,
  classNames: string[],
  labels: { x: string; y: string } = { x: 'x₁', y: 'x₂' },
): PointData {
  return {
    kind: 'points',
    points,
    xRange: RANGE,
    yRange: RANGE,
    classCount,
    classNames,
    xLabel: labels.x,
    yLabel: labels.y,
  };
}

const BINARY_NAMES = ['Class A', 'Class B'];

function clampToRange(v: number): number {
  return Math.max(RANGE[0] + 0.15, Math.min(RANGE[1] - 0.15, v));
}

/* ---------------- classification shapes ---------------- */

export const CLASSIFICATION_DATASETS: DatasetOption<PointData>[] = [
  {
    id: 'gaussians',
    name: 'Two blobs',
    blurb: 'Linearly separable until you crank the noise. The friendliest starting point.',
    generate(config) {
      const rng = makeRng(config.seed);
      const spread = 0.9 + config.noise * 2.4;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % 2;
        const cx = label === 0 ? -2.4 : 2.4;
        const cy = label === 0 ? -1.8 : 1.8;
        points.push({
          x: clampToRange(gauss(rng, cx, spread)),
          y: clampToRange(gauss(rng, cy, spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, BINARY_NAMES);
    },
  },
  {
    id: 'circle',
    name: 'Ring',
    blurb: 'One class inside, one outside. No straight line can ever separate them.',
    generate(config) {
      const rng = makeRng(config.seed + 3);
      const jitter = config.noise * 1.4;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % 2;
        const radius = label === 0 ? uniform(rng, 0, 2.2) : uniform(rng, 3.4, 5.2);
        const angle = uniform(rng, 0, Math.PI * 2);
        points.push({
          x: clampToRange(radius * Math.cos(angle) + gauss(rng, 0, jitter)),
          y: clampToRange(radius * Math.sin(angle) + gauss(rng, 0, jitter)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, ['Inside', 'Outside']);
    },
  },
  {
    id: 'xor',
    name: 'XOR',
    blurb: 'Four quadrants, alternating labels. The problem that killed the single perceptron.',
    generate(config) {
      const rng = makeRng(config.seed + 5);
      const jitter = config.noise * 1.1;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        let x = uniform(rng, -5, 5);
        let y = uniform(rng, -5, 5);
        // Push points off the axes so the boundary is unambiguous.
        if (Math.abs(x) < 0.6) x += x >= 0 ? 0.6 : -0.6;
        if (Math.abs(y) < 0.6) y += y >= 0 ? 0.6 : -0.6;
        const label = x * y > 0 ? 1 : 0;
        points.push({
          x: clampToRange(x + gauss(rng, 0, jitter)),
          y: clampToRange(y + gauss(rng, 0, jitter)),
          label,
        });
      }
      return finish(points, 2, BINARY_NAMES);
    },
  },
  {
    id: 'spiral',
    name: 'Spiral',
    blurb: 'Two interleaved arms. The hardest of the classic toy sets, needs real capacity.',
    generate(config) {
      const rng = makeRng(config.seed + 7);
      const jitter = config.noise * 0.9;
      const points: Point2D[] = [];
      const perArm = Math.floor(config.count / 2);
      for (let arm = 0; arm < 2; arm++) {
        for (let i = 0; i < perArm; i++) {
          const t = (i / perArm) * 3.6;
          const radius = 0.4 + t * 1.35;
          const angle = t * 2.1 + arm * Math.PI;
          points.push({
            x: clampToRange(radius * Math.cos(angle) + gauss(rng, 0, jitter)),
            y: clampToRange(radius * Math.sin(angle) + gauss(rng, 0, jitter)),
            label: arm,
          });
        }
      }
      return finish(shuffled(rng, points), 2, ['Arm A', 'Arm B']);
    },
  },
  {
    id: 'moons',
    name: 'Two moons',
    blurb: 'Curved but nearly separable. A good test of how gracefully a model bends.',
    generate(config) {
      const rng = makeRng(config.seed + 11);
      const jitter = config.noise * 0.9;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % 2;
        const t = uniform(rng, 0, Math.PI);
        const radius = 3.2;
        const x = label === 0 ? radius * Math.cos(t) - 1.4 : radius * Math.cos(t + Math.PI) + 1.4;
        const y = label === 0 ? radius * Math.sin(t) - 1.1 : radius * Math.sin(t + Math.PI) + 1.1;
        points.push({
          x: clampToRange(x + gauss(rng, 0, jitter)),
          y: clampToRange(y + gauss(rng, 0, jitter)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, ['Upper', 'Lower']);
    },
  },
  {
    id: 'stripes',
    name: 'Diagonal stripes',
    blurb: 'A repeating pattern. Trees carve it into steps; networks find the diagonal.',
    generate(config) {
      const rng = makeRng(config.seed + 13);
      const jitter = config.noise * 0.6;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const x = uniform(rng, -5.4, 5.4);
        const y = uniform(rng, -5.4, 5.4);
        const band = Math.floor((x + y + 12) / 3.2);
        points.push({
          x: clampToRange(x + gauss(rng, 0, jitter)),
          y: clampToRange(y + gauss(rng, 0, jitter)),
          label: band % 2 === 0 ? 0 : 1,
        });
      }
      return finish(points, 2, BINARY_NAMES);
    },
  },
  {
    id: 'imbalanced',
    name: 'Rare positives',
    blurb: 'Only ~12% positives. Accuracy looks great while the model learns nothing.',
    generate(config) {
      const rng = makeRng(config.seed + 17);
      const spread = 0.8 + config.noise * 1.8;
      const points: Point2D[] = [];
      const positives = Math.max(4, Math.round(config.count * 0.12));
      for (let i = 0; i < config.count; i++) {
        const label = i < positives ? 1 : 0;
        const cx = label === 1 ? 2.6 : -1.4;
        const cy = label === 1 ? 2.2 : -1.0;
        points.push({
          x: clampToRange(gauss(rng, cx, spread)),
          y: clampToRange(gauss(rng, cy, spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, ['Common', 'Rare']);
    },
  },
  {
    id: 'three-class',
    name: 'Three classes',
    blurb: 'For models that go beyond yes/no, trees and nearest neighbours handle it directly.',
    generate(config) {
      const rng = makeRng(config.seed + 19);
      const k = Math.min(4, Math.max(3, config.classCount ?? 3));
      const spread = 0.75 + config.noise * 1.6;
      const centres = [
        [-2.8, -2.2],
        [2.9, -2.4],
        [0.1, 3.0],
        [3.4, 3.2],
      ];
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % k;
        points.push({
          x: clampToRange(gauss(rng, centres[label][0], spread)),
          y: clampToRange(gauss(rng, centres[label][1], spread)),
          label,
        });
      }
      return finish(
        shuffled(rng, points),
        k,
        ['Class A', 'Class B', 'Class C', 'Class D'].slice(0, k),
      );
    },
  },
];

/* ---------------- clustering shapes (labels are hidden ground truth) ---------------- */

export const CLUSTERING_DATASETS: DatasetOption<PointData>[] = [
  {
    id: 'blobs',
    name: 'Round blobs',
    blurb: 'Well-separated, equal size. The case k-means was designed for.',
    generate(config) {
      const rng = makeRng(config.seed);
      const k = Math.max(2, Math.min(6, config.classCount ?? 3));
      const spread = 0.5 + config.noise * 1.3;
      const centres = ringCentres(k, 3.4);
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % k;
        points.push({
          x: clampToRange(gauss(rng, centres[label][0], spread)),
          y: clampToRange(gauss(rng, centres[label][1], spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), k, clusterNames(k));
    },
  },
  {
    id: 'varied',
    name: 'Different sizes',
    blurb: 'One tight cluster, one loose. k-means splits the loose one, watch it happen.',
    generate(config) {
      const rng = makeRng(config.seed + 3);
      const k = Math.max(2, Math.min(6, config.classCount ?? 3));
      const centres = ringCentres(k, 3.3);
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % k;
        const spread = (0.35 + 0.55 * label) * (1 + config.noise);
        points.push({
          x: clampToRange(gauss(rng, centres[label][0], spread)),
          y: clampToRange(gauss(rng, centres[label][1], spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), k, clusterNames(k));
    },
  },
  {
    id: 'anisotropic',
    name: 'Stretched',
    blurb: 'Elongated, rotated clusters. Spherical centroids cut straight across them.',
    generate(config) {
      const rng = makeRng(config.seed + 5);
      const k = Math.max(2, Math.min(5, config.classCount ?? 3));
      const centres = ringCentres(k, 2.7);
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % k;
        const long = gauss(rng, 0, 1.9 * (1 + config.noise * 0.6));
        const short = gauss(rng, 0, 0.42 * (1 + config.noise));
        const angle = 0.7 + label * 0.55;
        points.push({
          x: clampToRange(centres[label][0] + long * Math.cos(angle) - short * Math.sin(angle)),
          y: clampToRange(centres[label][1] + long * Math.sin(angle) + short * Math.cos(angle)),
          label,
        });
      }
      return finish(shuffled(rng, points), k, clusterNames(k));
    },
  },
  {
    id: 'rings',
    name: 'Concentric rings',
    blurb: 'The honest failure case: no set of centroids can describe a ring.',
    generate(config) {
      const rng = makeRng(config.seed + 7);
      const jitter = 0.18 + config.noise * 0.7;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % 2;
        const radius = label === 0 ? 1.7 : 4.4;
        const angle = uniform(rng, 0, Math.PI * 2);
        points.push({
          x: clampToRange(radius * Math.cos(angle) + gauss(rng, 0, jitter)),
          y: clampToRange(radius * Math.sin(angle) + gauss(rng, 0, jitter)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, ['Inner ring', 'Outer ring']);
    },
  },
  {
    id: 'uniform',
    name: 'No structure',
    blurb: 'Pure noise. k-means still returns k clusters, a lesson in itself.',
    generate(config) {
      const rng = makeRng(config.seed + 11);
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        points.push({ x: uniform(rng, -5, 5), y: uniform(rng, -5, 5), label: 0 });
      }
      return finish(points, 1, ['Unlabelled']);
    },
  },
  {
    id: 'touching',
    name: 'Touching blobs',
    blurb: 'Two clusters that overlap. Where exactly should the border go?',
    generate(config) {
      const rng = makeRng(config.seed + 13);
      const spread = 1.0 + config.noise * 1.4;
      const points: Point2D[] = [];
      for (let i = 0; i < config.count; i++) {
        const label = i % 2;
        points.push({
          x: clampToRange(gauss(rng, label === 0 ? -1.5 : 1.5, spread)),
          y: clampToRange(gauss(rng, 0, spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), 2, clusterNames(2));
    },
  },
  {
    id: 'unequal-counts',
    name: 'Unequal counts',
    blurb: 'One cluster has five times the points. Centroids drift towards the crowd.',
    generate(config) {
      const rng = makeRng(config.seed + 17);
      const points: Point2D[] = [];
      const big = Math.round(config.count * 0.66);
      for (let i = 0; i < config.count; i++) {
        const label = i < big ? 0 : i < big + (config.count - big) / 2 ? 1 : 2;
        const centre = [
          [-2.6, -1.6],
          [2.8, -1.2],
          [0.4, 3.2],
        ][label];
        const spread = (label === 0 ? 1.25 : 0.6) * (1 + config.noise * 0.8);
        points.push({
          x: clampToRange(gauss(rng, centre[0], spread)),
          y: clampToRange(gauss(rng, centre[1], spread)),
          label,
        });
      }
      return finish(shuffled(rng, points), 3, clusterNames(3));
    },
  },
];

function ringCentres(k: number, radius: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < k; i++) {
    const angle = (i / k) * Math.PI * 2 - Math.PI / 2;
    out.push([radius * Math.cos(angle), radius * Math.sin(angle) * 0.85]);
  }
  return out;
}

function clusterNames(k: number): string[] {
  return Array.from({ length: k }, (_, i) => 'Cluster ' + (i + 1));
}

export function getClassificationDataset(id: string): DatasetOption<PointData> {
  return CLASSIFICATION_DATASETS.find((d) => d.id === id) ?? CLASSIFICATION_DATASETS[0];
}

export function getClusteringDataset(id: string): DatasetOption<PointData> {
  return CLUSTERING_DATASETS.find((d) => d.id === id) ?? CLUSTERING_DATASETS[0];
}

/* ---------------- editing helpers ---------------- */

/** Add a point, keeping it inside the shared coordinate range. */
export function addPoint(data: PointData, x: number, y: number, label: number): PointData {
  return {
    ...data,
    points: [...data.points, { x: clampToRange(x), y: clampToRange(y), label }],
    classCount: Math.max(data.classCount, label + 1),
  };
}

/** Remove the point nearest to (x, y) within `radius` data units. Returns the same object if none. */
export function removeNearestPoint(data: PointData, x: number, y: number, radius: number): PointData {
  let bestIndex = -1;
  let bestDist = radius * radius;
  data.points.forEach((p, i) => {
    const d = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    if (d < bestDist) {
      bestDist = d;
      bestIndex = i;
    }
  });
  if (bestIndex < 0) return data;
  const points = data.points.slice();
  points.splice(bestIndex, 1);
  return { ...data, points };
}

/** Index of the point nearest to (x, y), or -1 when nothing is within `radius`. */
export function nearestPointIndex(points: readonly Point2D[], x: number, y: number, radius: number): number {
  let bestIndex = -1;
  let bestDist = radius * radius;
  for (let i = 0; i < points.length; i++) {
    const d = (points[i].x - x) * (points[i].x - x) + (points[i].y - y) * (points[i].y - y);
    if (d < bestDist) {
      bestDist = d;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/** Split a dataset into train/test partitions with a deterministic shuffle. */
export function splitData<T>(items: readonly T[], trainFraction: number, seed: number): { train: T[]; test: T[] } {
  const rng = makeRng(seed);
  const order = shuffled(rng, items.map((_, i) => i));
  const cut = Math.round(items.length * Math.min(1, Math.max(0, trainFraction)));
  const train: T[] = [];
  const test: T[] = [];
  order.forEach((idx, rank) => {
    if (rank < cut) train.push(items[idx]);
    else test.push(items[idx]);
  });
  return { train, test };
}

export const POINT_RANGE = RANGE;
