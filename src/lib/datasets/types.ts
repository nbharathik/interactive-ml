/** Shared data shapes. Every explainer consumes one of these two. */

/** A single (x, y) observation for a one-input regression problem. */
export interface RegressionPoint {
  x: number;
  y: number;
}

export interface RegressionData {
  kind: 'regression';
  points: RegressionPoint[];
  xRange: [number, number];
  yRange: [number, number];
  /** The generating function, when there is one, drawn as the "truth" line. */
  truth?: (x: number) => number;
  /** Axis labels. */
  xLabel: string;
  yLabel: string;
}

/** A 2-D point with an integer class label. Used by every classifier and cluster demo. */
export interface Point2D {
  x: number;
  y: number;
  /** Class label. For clustering datasets this is the *hidden* ground truth. */
  label: number;
}

export interface PointData {
  kind: 'points';
  points: Point2D[];
  xRange: [number, number];
  yRange: [number, number];
  classCount: number;
  xLabel: string;
  yLabel: string;
  /** Names shown in legends, indexed by label. */
  classNames: string[];
}

/** A generator entry that the dataset picker renders. */
export interface DatasetOption<TData> {
  id: string;
  name: string;
  /** One line on what makes this dataset worth trying. */
  blurb: string;
  generate(config: DatasetConfig): TData;
}

export interface DatasetConfig {
  /** Number of samples to draw. */
  count: number;
  /** Noise magnitude, 0..1, interpreted per generator. */
  noise: number;
  /** Deterministic seed. */
  seed: number;
  /** Number of classes/clusters where the generator supports it. */
  classCount?: number;
}
