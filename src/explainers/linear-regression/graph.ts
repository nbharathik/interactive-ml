/** Linear regression as a one-unit network. `training` adds the target, the loss and the gradient. */

import { GraphBuilder } from '../../explainer/architecture';
import type { ArchGraph, ArchNode, Rect } from '../../explainer/architecture';
import { maxAbs, paintVectorThumbnail } from '../../explainer/vectors';
import type { VectorColumn } from '../../explainer/vectors';
import type { Palette } from '../../lib/viz/palette';
import { fmt, fmtCompact, sub, sup } from '../../lib/math/stats';
import type { LinRegState } from '../../lib/ml/linearRegression';

export type GraphView = 'model' | 'training';

/** Every vector the graph shows, one value per point. */
export interface LinRegVectors {
  x: number[];
  y: number[];
  /** features[k] is xᵏ for every point (0 is the constant 1). */
  features: number[][];
  prediction: number[];
  /** Shared colour scale for y, ŷ and the data's y column. */
  yScale: number;
}

export interface LinRegGraphInput {
  state: LinRegState;
  degree: number;
  started: boolean;
  standardised: boolean;
  vectors: LinRegVectors;
  /** Row order shared with the detail card. */
  order: readonly number[];
  /** Rows the last step trained on, or null when every step sees them all. */
  batch: ReadonlySet<number> | null;
  /** `model` hides everything that only exists to train. */
  view: GraphView;
}

/** Node and edge ids, exported so the detail card can switch on them. */
export const LR_NODES = {
  data: 'data',
  one: 'one',
  sum: 'sum',
  yhat: 'yhat',
  target: 'y',
  loss: 'loss',
} as const;

/** The input node carrying x to the given power (1..degree). */
export function inputId(power: number): string {
  return 'x' + power;
}

/** The wire carrying the weight of the input at the given power. */
export function weightEdgeId(power: number): string {
  return inputId(power) + '->' + LR_NODES.sum;
}

export function dataEdgeId(power: number): string {
  return LR_NODES.data + '->' + inputId(power);
}

export const LR_EDGES = {
  bias: LR_NODES.one + '->' + LR_NODES.sum,
  output: LR_NODES.sum + '->' + LR_NODES.yhat,
  toLoss: LR_NODES.yhat + '->' + LR_NODES.loss,
  targetToLoss: LR_NODES.target + '->' + LR_NODES.loss,
  gradient: LR_NODES.loss + '->' + LR_NODES.sum,
} as const;

/** The input at the given power: x², or x̃² once x is standardised. */
export function powerLabel(power: number, standardised = false): string {
  return (standardised ? 'x̃' : 'x') + (power === 1 ? '' : sup(power));
}

/** 'w' at degree 1, 'w₁', 'w₂', ... above it; b for the bias. */
export function weightLabel(index: number, degree: number): string {
  if (index === 0) return 'b';
  return degree === 1 ? 'w' : 'w' + sub(index);
}

/** Squared error per point, the terms the loss averages. */
export function squaredErrors(vectors: LinRegVectors): number[] {
  return vectors.prediction.map((p, i) => (p - vectors.y[i]) ** 2);
}

/** 'x, 1' at degree 1, 'x, x², 1' at 2, then 'x … x⁵, 1'. */
function inputsSubtitle(degree: number, standardised: boolean): string {
  if (degree <= 2) {
    return Array.from({ length: degree }, (_, k) => powerLabel(k + 1, standardised)).join(', ') + ', 1';
  }
  return powerLabel(1, standardised) + ' … ' + powerLabel(degree, standardised) + ', 1';
}

export function buildLinRegGraph({
  state,
  degree,
  started,
  standardised,
  vectors,
  order,
  batch,
  view,
}: LinRegGraphInput): ArchGraph {
  const g = new GraphBuilder();
  const training = view === 'training';
  const weights = state.weights;
  const n = vectors.x.length;
  const shape = n + ' × 1';
  const x = powerLabel(1, standardised);
  const equation = degree === 1 ? 'w' + x + ' + b' : 'Σ wₖ' + x + 'ᵏ + b';

  const strip =
    (columns: VectorColumn[]): ArchNode['thumbnail'] =>
    (ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette) =>
      paintVectorThumbnail(ctx, rect, palette, columns, order, batch);

  const cData = g.column('Data', n + ' points');
  const cInputs = g.column('Inputs', standardised ? 'x̃ = (x − μ) / s' : inputsSubtitle(degree, false));
  const cUnit = g.column('Unit', equation);
  const cOutput = g.column('Output');

  g.node(cData, {
    id: LR_NODES.data,
    kind: 'data',
    label: 'X, y',
    value: n + ' × 2',
    thumbnail: strip([
      { name: 'x', values: vectors.x },
      { name: 'y', values: vectors.y, scale: vectors.yScale },
    ]),
    readout: batch ? batch.size + ' of ' + n + ' rows' : n + ' × 2',
  });

  for (let k = 1; k <= degree; k++) {
    const label = powerLabel(k, standardised);
    g.node(cInputs, {
      id: inputId(k),
      kind: 'input',
      label,
      thumbnail: strip([{ name: label, values: vectors.features[k] }]),
      readout: shape,
    });
    g.edge(LR_NODES.data, inputId(k), { symbol: label, readout: shape });
  }
  g.node(cInputs, {
    id: LR_NODES.one,
    kind: 'input',
    label: '1',
    thumbnail: strip([{ name: '1', values: vectors.features[0], ramp: 'magnitude', scale: 2 }]),
    readout: shape,
  });

  g.node(cUnit, {
    id: LR_NODES.sum,
    kind: 'unit',
    label: 'Σ',
    readout: equation,
  });

  g.node(cOutput, {
    id: LR_NODES.yhat,
    kind: 'output',
    label: 'ŷ',
    thumbnail: strip([{ name: 'ŷ', values: vectors.prediction, scale: vectors.yScale }]),
    readout: shape,
  });

  for (let k = 1; k <= degree; k++) {
    const w = weights[k] ?? 0;
    g.edge(inputId(k), LR_NODES.sum, {
      weight: started ? w : 0,
      symbol: weightLabel(k, degree),
      readout: '= ' + fmt(w, 2),
    });
  }
  g.edge(LR_NODES.one, LR_NODES.sum, {
    weight: started ? (weights[0] ?? 0) : 0,
    symbol: 'b',
    readout: '= ' + fmt(weights[0] ?? 0, 2),
  });
  g.edge(LR_NODES.sum, LR_NODES.yhat, { symbol: 'ŷ', readout: shape });

  if (!training) return g.build('ŷ = ' + equation);

  const cLoss = g.column('Loss', 'mean (ŷ − y)²');
  const errors = squaredErrors(vectors);
  g.node(cLoss, {
    id: LR_NODES.target,
    kind: 'input',
    label: 'y',
    thumbnail: strip([{ name: 'y', values: vectors.y, scale: vectors.yScale }]),
    readout: shape,
  });
  // The loss takes the centre line.
  g.node(cLoss, {
    id: LR_NODES.loss,
    kind: 'loss',
    label: 'L',
    anchor: true,
    thumbnail: strip([{ name: '(ŷ − y)²', values: errors, ramp: 'magnitude', scale: maxAbs(errors) }]),
    readout: started && Number.isFinite(state.loss) ? '= ' + fmtCompact(state.loss) : 'mean (ŷ − y)²',
  });

  g.edge(LR_NODES.yhat, LR_NODES.loss, { symbol: 'ŷ', readout: shape });
  g.edge(LR_NODES.target, LR_NODES.loss, { symbol: 'y', readout: shape });

  // One dashed edge stands for the whole gradient step, bias included.
  g.edge(LR_NODES.loss, LR_NODES.sum, {
    active: true,
    symbol: '∂L/∂w, ∂L/∂b',
    readout: started ? '‖∇L‖ = ' + fmtCompact(state.gradientNorm) : 'w ← w − α ∂L/∂w,  b ← b − α ∂L/∂b',
  });

  return g.build('ŷ = ' + equation + ',   L = mean (ŷ − y)²');
}
