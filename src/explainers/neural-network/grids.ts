/** Activation grids: every unit's output sampled over the input space. Pure, no canvas. */

import { POINT_RANGE } from '../../lib/datasets/points';
import { activate, evaluateNeuron, inputVector } from '../../lib/ml/neuralNetwork';
import type { Activation, InputFeature, Layer } from '../../lib/ml/neuralNetwork';

/* ---------------- activation grids ---------------- */

/** One unit's output sampled on a square grid over the input space. */
export interface NeuronGrid {
  size: number;
  /** Row-major, row 0 is the top of the plot (largest x₂). */
  values: Float32Array;
  min: number;
  max: number;
  /** max − min. A unit whose spread is ~0 says the same thing everywhere. */
  spread: number;
}

export interface NeuronGrids {
  size: number;
  /** hidden[layer][neuron]; the output layer is not included here. */
  hidden: NeuronGrid[][];
  /** The network's own output, used for the output node's thumbnail. */
  output: NeuronGrid;
}

/** Which unit the reader is currently looking at. `layer` indexes `state.layers`. */
export interface NeuronRef {
  layer: number;
  neuron: number;
}

function blankGrid(size: number): NeuronGrid {
  return { size, values: new Float32Array(size * size), min: 0, max: 0, spread: 0 };
}

function finaliseGrid(grid: NeuronGrid): void {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < grid.values.length; i++) {
    const v = grid.values[i];
    if (!Number.isFinite(v)) {
      grid.values[i] = 0;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
  }
  grid.min = Number.isFinite(min) ? min : 0;
  grid.max = Number.isFinite(max) ? max : 0;
  grid.spread = grid.max - grid.min;
}

/** Sample every unit across the input space in one sweep, one forward pass per grid point. */
export function computeNeuronGrids(
  layers: readonly Layer[],
  features: readonly InputFeature[],
  activation: Activation,
  size = 14,
): NeuronGrids {
  const hiddenCount = Math.max(0, layers.length - 1);
  const hidden: NeuronGrid[][] = [];
  for (let l = 0; l < hiddenCount; l++) hidden.push(layers[l].map(() => blankGrid(size)));
  const output = blankGrid(size);

  const [lo, hi] = POINT_RANGE;
  for (let row = 0; row < size; row++) {
    const y = hi - ((row + 0.5) / size) * (hi - lo);
    for (let col = 0; col < size; col++) {
      const x = lo + ((col + 0.5) / size) * (hi - lo);
      const index = row * size + col;

      let previous = inputVector(features, x, y);
      for (let l = 0; l < layers.length; l++) {
        const isOutput = l === layers.length - 1;
        const current: number[] = [];
        for (const neuron of layers[l]) {
          let z = neuron.bias;
          for (let i = 0; i < neuron.weights.length; i++) z += neuron.weights[i] * previous[i];
          current.push(isOutput ? z : activate(activation, z));
        }
        if (isOutput) output.values[index] = current[0] ?? 0;
        else for (let n = 0; n < current.length; n++) hidden[l][n].values[index] = current[n];
        previous = current;
      }
    }
  }

  for (const layer of hidden) for (const grid of layer) finaliseGrid(grid);
  finaliseGrid(output);
  return { size, hidden, output };
}

/** One unit at a finer grid, for the enlarged picture. */
export function computeUnitGrid(
  layers: readonly Layer[],
  features: readonly InputFeature[],
  activation: Activation,
  target: NeuronRef,
  size = 64,
): NeuronGrid {
  const grid = blankGrid(size);
  const [lo, hi] = POINT_RANGE;
  for (let row = 0; row < size; row++) {
    const y = hi - ((row + 0.5) / size) * (hi - lo);
    for (let col = 0; col < size; col++) {
      const x = lo + ((col + 0.5) / size) * (hi - lo);
      grid.values[row * size + col] = evaluateNeuron(
        layers,
        inputVector(features, x, y),
        activation,
        target.layer,
        target.neuron,
      );
    }
  }
  finaliseGrid(grid);
  return grid;
}

/** A spread under this says the unit paints the same value everywhere. */
export const FLAT_THRESHOLD = 0.02;

/** Units whose output barely varies across the whole input space. */
export function countFlatUnits(grids: NeuronGrids, threshold = FLAT_THRESHOLD): number {
  let total = 0;
  for (const layer of grids.hidden) for (const grid of layer) if (grid.spread < threshold) total += 1;
  return total;
}

/** The hidden units whose output barely varies, in diagram order. */
export function flatUnitRefs(grids: NeuronGrids, threshold = FLAT_THRESHOLD): NeuronRef[] {
  const refs: NeuronRef[] = [];
  grids.hidden.forEach((layer, l) => {
    layer.forEach((grid, n) => {
      if (grid.spread < threshold) refs.push({ layer: l, neuron: n });
    });
  });
  return refs;
}
