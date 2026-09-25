/** Logistic regression as a one-unit network plus the sigmoid and the decision. */

import { GraphBuilder } from '../../explainer/architecture';
import type { ArchGraph, ArchNode, Rect } from '../../explainer/architecture';
import { sub } from '../../lib/math/stats';
import { maxAbs, paintVectorThumbnail } from '../../explainer/vectors';
import type { VectorColumn } from '../../explainer/vectors';
import type { Palette } from '../../lib/viz/palette';
import { fmt, fmtCompact, fmtPercent } from '../../lib/math/stats';
import type { LogRegState } from '../../lib/ml/logisticRegression';

export type GraphView = 'model' | 'training';

/** Every vector the graph shows, one value per point. */
export interface LogRegVectors {
  x1: number[];
  x2: number[];
  /** 0 or 1. */
  label: number[];
  /** features[i] is feature i for every point, in `featureNames` order (0 is the constant 1). */
  features: number[][];
  z: number[];
  p: number[];
  /** The class called at the current threshold, 0 or 1. */
  called: number[];
  /** Cross-entropy per point. */
  loss: number[];
}

export interface LogRegGraphInput {
  state: LogRegState;
  featureNames: readonly string[];
  threshold: number;
  accuracy: number;
  logLoss: number;
  started: boolean;
  standardised: boolean;
  vectors: LogRegVectors;
  order: readonly number[];
  /** Rows the last step trained on, or null when every step sees them all. */
  batch: ReadonlySet<number> | null;
  /** `model` hides everything that only exists to train. */
  view: GraphView;
}

export const LOG_NODES = {
  data: 'data',
  score: 'z',
  sigmoid: 'sigma',
  prob: 'p',
  decision: 'decision',
  target: 'y',
  loss: 'loss',
} as const;

/** The input node for feature `index` in `featureNames` order; 0 is the bias. */
export function featureId(index: number): string {
  return 'f' + index;
}

export function weightEdgeId(index: number): string {
  return featureId(index) + '->' + LOG_NODES.score;
}

/** The wire from the data to feature `index`. */
export function featureEdgeId(index: number): string {
  return LOG_NODES.data + '->' + featureId(index);
}

export const LOG_EDGES = {
  toSigmoid: LOG_NODES.score + '->' + LOG_NODES.sigmoid,
  toProb: LOG_NODES.sigmoid + '->' + LOG_NODES.prob,
  toDecision: LOG_NODES.prob + '->' + LOG_NODES.decision,
  toLoss: LOG_NODES.prob + '->' + LOG_NODES.loss,
  targetToLoss: LOG_NODES.target + '->' + LOG_NODES.loss,
  gradient: LOG_NODES.loss + '->' + LOG_NODES.score,
} as const;

export function weightLabel(index: number): string {
  return index === 0 ? 'b' : 'w' + sub(index);
}

/** The symbol a unit is fed: x̃₁ for x₁ once the data is standardised. */
export function fedName(name: string, standardised: boolean): string {
  return standardised ? name.replace(/x/g, 'x̃') : name;
}

export function buildLogRegGraph({
  state,
  featureNames: names,
  threshold,
  accuracy,
  logLoss,
  started,
  standardised,
  vectors,
  order,
  batch,
  view,
}: LogRegGraphInput): ArchGraph {
  const g = new GraphBuilder();
  const training = view === 'training';
  const n = vectors.label.length;
  const shape = n + ' × 1';
  const rule = 'p ≥ ' + fmt(threshold, 2);

  const strip =
    (columns: VectorColumn[]): ArchNode['thumbnail'] =>
    (ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette) =>
      paintVectorThumbnail(ctx, rect, palette, columns, order, batch);

  const fed = names.map((name) => fedName(name, standardised));
  const features = fed.slice(1);
  const cData = g.column('Data', n + ' points');
  const cInput = g.column(
    'Inputs',
    standardised ? 'x̃ = (x − μ) / s' : features.length <= 3 ? features.join(', ') + ', 1' : names.length + ' inputs',
  );
  const cUnit = g.column('Unit', 'z = w·' + (standardised ? 'x̃' : 'x') + ' + b');
  const cSquash = g.column('Squash', 'σ(z)');
  const cOutput = g.column('Output', 'ŷ = [' + rule + ']');

  g.node(cData, {
    id: LOG_NODES.data,
    kind: 'data',
    label: 'X, y',
    value: n + ' × 3',
    thumbnail: strip([
      { name: 'x₁', values: vectors.x1 },
      { name: 'x₂', values: vectors.x2 },
      { name: 'y', values: vectors.label, ramp: 'class' },
    ]),
    readout: batch ? batch.size + ' of ' + n + ' rows' : n + ' × 3',
  });

  // One node per feature; the bias input comes last.
  fed.forEach((name, index) => {
    if (index === 0) return;
    g.node(cInput, {
      id: featureId(index),
      kind: 'input',
      label: name,
      thumbnail: strip([{ name, values: vectors.features[index] }]),
      readout: shape,
    });
    g.edge(LOG_NODES.data, featureId(index), { symbol: name, readout: shape });
  });
  g.node(cInput, {
    id: featureId(0),
    kind: 'input',
    label: '1',
    thumbnail: strip([{ name: '1', values: vectors.features[0], ramp: 'magnitude', scale: 2 }]),
    readout: shape,
  });

  g.node(cUnit, {
    id: LOG_NODES.score,
    kind: 'unit',
    label: 'Σ',
    readout: 'z = w·' + (standardised ? 'x̃' : 'x') + ' + b',
  });

  g.node(cSquash, {
    id: LOG_NODES.sigmoid,
    kind: 'op',
    label: 'σ',
    readout: '1 / (1 + e⁻ᶻ)',
  });

  // The probability takes the centre line; the decision hangs under it.
  g.node(cOutput, {
    id: LOG_NODES.prob,
    kind: 'output',
    label: 'p',
    anchor: true,
    thumbnail: strip([{ name: 'p', values: vectors.p, ramp: 'probability' }]),
    readout: shape,
  });
  g.node(cOutput, {
    id: LOG_NODES.decision,
    kind: 'output',
    label: 'ŷ',
    thumbnail: strip([{ name: 'ŷ', values: vectors.called, ramp: 'class' }]),
    readout: started ? fmtPercent(accuracy, 1) + ' correct' : shape,
  });

  names.forEach((_, index) => {
    const w = state.weights[index] ?? 0;
    g.edge(featureId(index), LOG_NODES.score, {
      weight: started ? w : 0,
      symbol: weightLabel(index),
      readout: '= ' + fmt(w, 2),
    });
  });

  g.edge(LOG_NODES.score, LOG_NODES.sigmoid, { symbol: 'z', readout: shape });
  g.edge(LOG_NODES.sigmoid, LOG_NODES.prob, { symbol: 'p', readout: shape });
  g.edge(LOG_NODES.prob, LOG_NODES.decision, { symbol: rule, readout: shape });

  const score = 'z = w·' + (standardised ? 'x̃' : 'x') + ' + b';
  if (!training) return g.build(score + ',   p = σ(z),   ŷ = [' + rule + ']');

  const cLoss = g.column('Loss', 'mean −log p(y)');
  g.node(cLoss, {
    id: LOG_NODES.target,
    kind: 'input',
    label: 'y',
    thumbnail: strip([{ name: 'y', values: vectors.label, ramp: 'class' }]),
    readout: shape,
  });
  g.node(cLoss, {
    id: LOG_NODES.loss,
    kind: 'loss',
    label: 'L',
    anchor: true,
    thumbnail: strip([{ name: '−log p(y)', values: vectors.loss, ramp: 'magnitude', scale: maxAbs(vectors.loss) }]),
    readout: started && Number.isFinite(logLoss) ? '= ' + fmtCompact(logLoss) : 'mean −log p(y)',
  });

  // The loss reads the probability, not the decision; the threshold never reaches training.
  g.edge(LOG_NODES.prob, LOG_NODES.loss, { symbol: 'p', readout: shape });
  g.edge(LOG_NODES.target, LOG_NODES.loss, { symbol: 'y', readout: shape });

  g.edge(LOG_NODES.loss, LOG_NODES.score, {
    active: true,
    symbol: '∂L/∂w, ∂L/∂b',
    readout: started ? '‖∇L‖ = ' + fmtCompact(state.gradientNorm) : 'w ← w − α ∂L/∂w,  b ← b − α ∂L/∂b',
  });

  return g.build(score + ',   p = σ(z),   L = mean −log p(y)');
}
