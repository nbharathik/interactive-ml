/** Polynomial regression as a one-unit network: the powers of x in, one weighted sum, the squared error out. */

import { GraphBuilder } from '../../explainer/architecture';
import type { ArchGraph, ArchNode, Rect } from '../../explainer/architecture';
import { maxAbs, paintVectorThumbnail } from '../../explainer/vectors';
import type { VectorColumn } from '../../explainer/vectors';
import type { Palette } from '../../lib/viz/palette';
import { fmt, fmtCompact, sub, sup } from '../../lib/math/stats';
import type { PolyFit } from '../../lib/ml/polynomialRegression';

/** Every vector the graph shows, one value per training point. */
export interface PolyVectors {
  x: number[];
  y: number[];
  /** features[k] is x̃ᵏ for every point (0 is the constant 1). */
  features: number[][];
  prediction: number[];
  /** Shared colour scale for y and ŷ. */
  yScale: number;
}

export interface PolyGraphInput {
  fit: PolyFit;
  standardised: boolean;
  vectors: PolyVectors;
  /** Row order shared with the detail card. */
  order: readonly number[];
  trainError: number;
}

export const PR_NODES = {
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

export function weightEdgeId(power: number): string {
  return inputId(power) + '->' + PR_NODES.sum;
}

export function dataEdgeId(power: number): string {
  return PR_NODES.data + '->' + inputId(power);
}

export const PR_EDGES = {
  bias: PR_NODES.one + '->' + PR_NODES.sum,
  output: PR_NODES.sum + '->' + PR_NODES.yhat,
  toLoss: PR_NODES.yhat + '->' + PR_NODES.loss,
  targetToLoss: PR_NODES.target + '->' + PR_NODES.loss,
} as const;

/** x², or x̃² once x is standardised; x¹² past the single digits. */
export function powerLabel(power: number, standardised = false): string {
  return (standardised ? 'x̃' : 'x') + (power === 1 ? '' : sup(power));
}

/** 'w' at degree 1, 'w₁', 'w₂', … above it; b for the bias. */
export function weightLabel(index: number, degree: number): string {
  if (index === 0) return 'b';
  return degree === 1 ? 'w' : 'w' + sub(index);
}

export function squaredErrors(vectors: PolyVectors): number[] {
  return vectors.prediction.map((p, i) => (p - vectors.y[i]) ** 2);
}

function inputsSubtitle(degree: number, standardised: boolean): string {
  if (degree === 0) return '1';
  if (degree <= 2) {
    return Array.from({ length: degree }, (_, k) => powerLabel(k + 1, standardised)).join(', ') + ', 1';
  }
  return powerLabel(1, standardised) + ' … ' + powerLabel(degree, standardised) + ', 1';
}

export function buildPolyGraph({ fit, standardised, vectors, order, trainError }: PolyGraphInput): ArchGraph {
  const g = new GraphBuilder();
  const degree = fit.degree;
  const weights = fit.weights;
  const n = vectors.x.length;
  const shape = n + ' × 1';
  const x = powerLabel(1, standardised);
  const equation = degree === 0 ? 'b' : degree === 1 ? 'w' + x + ' + b' : 'Σ wₖ' + x + 'ᵏ + b';

  const strip =
    (columns: VectorColumn[]): ArchNode['thumbnail'] =>
    (ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette) =>
      paintVectorThumbnail(ctx, rect, palette, columns, order);

  const cData = g.column('Data', n + ' points');
  const cInputs = g.column('Inputs', standardised ? 'x̃ = (x − μ) / s' : inputsSubtitle(degree, false));
  const cUnit = g.column('Unit', equation);
  const cOutput = g.column('Output');
  const cLoss = g.column('Error', 'mean (ŷ − y)²');

  g.node(cData, {
    id: PR_NODES.data,
    kind: 'data',
    label: 'X, y',
    value: n + ' × 2',
    thumbnail: strip([
      { name: 'x', values: vectors.x },
      { name: 'y', values: vectors.y, scale: vectors.yScale },
    ]),
    readout: n + ' × 2',
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
    g.edge(PR_NODES.data, inputId(k), { symbol: label, readout: shape });
  }
  g.node(cInputs, {
    id: PR_NODES.one,
    kind: 'input',
    label: '1',
    thumbnail: strip([{ name: '1', values: vectors.features[0], ramp: 'magnitude', scale: 2 }]),
    readout: shape,
  });

  g.node(cUnit, { id: PR_NODES.sum, kind: 'unit', label: 'Σ', readout: equation });

  g.node(cOutput, {
    id: PR_NODES.yhat,
    kind: 'output',
    label: 'ŷ',
    thumbnail: strip([{ name: 'ŷ', values: vectors.prediction, scale: vectors.yScale }]),
    readout: shape,
  });

  for (let k = 1; k <= degree; k++) {
    const w = weights[k] ?? 0;
    g.edge(inputId(k), PR_NODES.sum, { weight: w, symbol: weightLabel(k, degree), readout: '= ' + fmt(w, 2) });
  }
  g.edge(PR_NODES.one, PR_NODES.sum, { weight: weights[0] ?? 0, symbol: 'b', readout: '= ' + fmt(weights[0] ?? 0, 2) });
  g.edge(PR_NODES.sum, PR_NODES.yhat, { symbol: 'ŷ', readout: shape });

  const errors = squaredErrors(vectors);
  g.node(cLoss, {
    id: PR_NODES.target,
    kind: 'input',
    label: 'y',
    thumbnail: strip([{ name: 'y', values: vectors.y, scale: vectors.yScale }]),
    readout: shape,
  });
  g.node(cLoss, {
    id: PR_NODES.loss,
    kind: 'loss',
    label: 'L',
    anchor: true,
    thumbnail: strip([{ name: '(ŷ − y)²', values: errors, ramp: 'magnitude', scale: maxAbs(errors) }]),
    readout: Number.isFinite(trainError) ? '= ' + fmtCompact(trainError) : 'mean (ŷ − y)²',
  });
  g.edge(PR_NODES.yhat, PR_NODES.loss, { symbol: 'ŷ', readout: shape });
  g.edge(PR_NODES.target, PR_NODES.loss, { symbol: 'y', readout: shape });

  return g.build('ŷ = ' + equation + ',   L = mean (ŷ − y)²');
}
