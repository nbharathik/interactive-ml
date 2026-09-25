/** Detail cards for the network diagram. */

import { useCallback, useState } from 'react';

import { paintFieldThumbnail } from '../../explainer/components/ArchitectureView';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailHead, Formula, Scalar, Scalars, VectorTable } from '../../explainer/components/Detail';
import { texLines, texNum, texSum, texSymbol } from '../../explainer/tex';
import type { VectorColumn } from '../../explainer/vectors';
import { maxAbs } from '../../explainer/vectors';
import { fmtCell, sub } from '../../lib/math/stats';
import { FEATURE_LABELS } from '../../lib/ml/neuralNetwork';
import type { Activation, InputFeature, Layer } from '../../lib/ml/neuralNetwork';
import { rgba } from '../../lib/viz/palette';
import type { NeuronGrid, NeuronGrids } from './grids';
import type { DiagramHover, NetworkVectors } from './panels';
import { unitSymbol } from './panels';

const ACTIVATION_NAMES: Record<Activation, string> = {
  tanh: 'tanh',
  relu: 'ReLU',
  sigmoid: 'sigmoid',
  linear: 'linear',
};

const ACTIVATION_TEX: Record<Activation, string> = {
  tanh: '\\tanh z',
  relu: '\\max(0, z)',
  sigmoid: '\\sigma(z)',
  linear: 'z',
};

/** What each input node carries, in terms of the raw coordinates. */
const FEATURE_TEX: Record<InputFeature, string> = {
  x1: 'x_1 / 3',
  x2: 'x_2 / 3',
  x1sq: '(x_1 / 3)^2',
  x2sq: '(x_2 / 3)^2',
  x1x2: '(x_1 / 3)(x_2 / 3)',
  sinx1: '\\tfrac{2}{3} \\sin x_1',
  sinx2: '\\tfrac{2}{3} \\sin x_2',
};

export interface NetworkDetailProps {
  target: NonNullable<DiagramHover>;
  layers: readonly Layer[];
  features: readonly InputFeature[];
  activation: Activation;
  grids: NeuronGrids;
  /** A sharper grid for the unit whose card is open, when one is ready. */
  focusGrid?: NeuronGrid | null;
  vectors: NetworkVectors;
  learningRate: number;
  batchSize: number;
  /** Opens another component's card. */
  jump: (next: DiagramHover) => void;
}

export function NetworkDetail({
  target,
  layers,
  features,
  activation,
  grids,
  focusGrid,
  vectors,
  learningRate,
  batchSize,
  jump,
}: NetworkDetailProps) {
  const n = vectors.label.length;
  const featureNames = features.map((f) => FEATURE_LABELS[f]);
  const lastLayer = layers.length - 1;
  const x1: VectorColumn = { name: 'x₁', values: vectors.x1 };
  const x2: VectorColumn = { name: 'x₂', values: vectors.x2 };
  const y: VectorColumn = { name: 'y', values: vectors.label, ramp: 'class' };
  const table = (columns: VectorColumn[], extra?: VectorColumn[]) => (
    <VectorTable
      key={columns.map((c) => c.name).join()}
      columns={columns}
      order={vectors.order}
      classes={vectors.label}
      extra={extra}
    />
  );

  /** The symbols feeding layer `l`: the inputs, or the units of the layer before. */
  const inputNames = (l: number) =>
    l === 0 ? featureNames : (layers[l - 1] ?? []).map((_, i) => unitSymbol(i));
  const inputColumns = (l: number): VectorColumn[] =>
    inputNames(l).map((name, i) => ({
      name,
      values: (l === 0 ? vectors.inputs[i] : vectors.activations[l - 1]?.[i]) ?? [],
    }));
  const inputTarget = (l: number, i: number): DiagramHover =>
    l === 0 ? { kind: 'input', index: i } : { kind: 'neuron', layer: l - 1, neuron: i };
  const unitTarget = (l: number, u: number): DiagramHover => ({ kind: 'neuron', layer: l, neuron: u });
  const unitName = (l: number, u: number) => (l === lastLayer ? 'ŷ' : unitSymbol(u));
  const layerName = (l: number) => (l === lastLayer ? 'output' : 'hidden ' + (l + 1));
  const link = (label: string, to: DiagramHover) => (
    <button type="button" className="mlx-wsum__jump" onClick={() => jump(to)} title="Open it">
      {label}
    </button>
  );

  if (target.kind === 'point') {
    const ones = vectors.label.reduce((sum, v) => sum + v, 0);
    return (
      <>
        <DetailHead eyebrow="Data" title="X, y" shape={n + ' × 3'} />
        <Scalars>
          <Scalar label="n₀" value={n - ones} decimals={0} />
          <Scalar label="n₁" value={ones} decimals={0} />
        </Scalars>
        {table([x1, x2, y])}
      </>
    );
  }

  if (target.kind === 'input') {
    const feature = features[target.index];
    const name = featureNames[target.index];
    const fed: VectorColumn = { name, values: vectors.inputs[target.index] ?? [] };
    const raw =
      feature === 'x1' || feature === 'x2' ? [] : feature === 'x1x2' ? [x1, x2] : feature.startsWith('x1') || feature === 'sinx1' ? [x1] : [x2];
    return (
      <>
        <DetailHead
          eyebrow={'Input ' + (target.index + 1) + ' / ' + features.length}
          title={name}
          shape={n + ' × 1 · → ' + layerName(0)}
        />
        <Formula tex={texSymbol(name) + ' \\leftarrow ' + FEATURE_TEX[feature]} />
        {table([...raw, fed, y])}
      </>
    );
  }

  if (target.kind === 'weight') {
    const neuron = layers[target.layer]?.[target.neuron];
    if (!neuron) return null;
    const w = neuron.weights[target.input] ?? 0;
    const from = inputNames(target.layer)[target.input] ?? 'in ' + (target.input + 1);
    const to = unitName(target.layer, target.neuron);
    const source = inputColumns(target.layer)[target.input] ?? { name: from, values: [] };
    const fromTex = texSymbol(from);
    return (
      <>
        <DetailHead
          eyebrow={'Weight · ' + layerName(target.layer)}
          title={
            <>
              {link(from, inputTarget(target.layer, target.input))} → {link(to, unitTarget(target.layer, target.neuron))}
            </>
          }
          shape={'W' + sub(target.layer + 1) + '[' + (target.input + 1) + ', ' + (target.neuron + 1) + ']'}
        />
        <Formula tex={'w\\,' + fromTex + ' = ' + texNum(w) + '\\,' + fromTex} />
        <Scalars>
          <Scalar label="w" value={w} tone="accent" />
          <Scalar
            label="Δw"
            value={(-learningRate * (neuron.weightGradients[target.input] ?? 0)) / Math.max(1, Math.min(batchSize, n))}
            decimals={4}
          />
        </Scalars>
        {table([source, { name: 'w·' + from, values: source.values.map((v) => v * w) }, y])}
      </>
    );
  }

  const neuron = layers[target.layer]?.[target.neuron];
  if (!neuron) return null;
  const isOutput = target.layer === lastLayer;
  const names = inputNames(target.layer);
  const grid = (isOutput ? null : focusGrid) ?? (isOutput ? grids.output : grids.hidden[target.layer]?.[target.neuron]);
  const width = layers[target.layer].length;
  const title = unitName(target.layer, target.neuron);
  const out = vectors.activations[target.layer]?.[target.neuron] ?? [];
  const sum = vectors.sums[target.layer]?.[target.neuron] ?? [];
  const terms = neuron.weights.map((coef, i) => ({ coef, symbol: texSymbol(names[i] ?? 'x_{' + (i + 1) + '}') }));
  const weighted = texSum(terms, neuron.bias, { perLine: 2 });
  const equation = isOutput
    ? texLines([['\\hat{y}', weighted]])
    : texLines([
        ['z', weighted],
        [texSymbol(title), ACTIVATION_TEX[activation]],
      ]);

  let columns: VectorColumn[];
  if (isOutput) {
    const t = vectors.label.map((v) => (v === 1 ? 1 : -1));
    const loss = out.map((v, i) => 0.5 * (v - t[i]) ** 2);
    columns = [
      { name: 'ŷ', values: out },
      { name: 't', values: t, ramp: 'class' },
      { name: '½(ŷ−t)²', values: loss, ramp: 'magnitude', scale: maxAbs(loss) },
    ];
  } else {
    columns = [{ name: 'z', values: sum }, { name: title, values: out }, y];
  }

  return (
    <>
      <DetailHead
        eyebrow={isOutput ? 'Output' : 'Hidden ' + (target.layer + 1) + ' · unit ' + (target.neuron + 1) + ' / ' + width}
        title={title}
        shape={
          neuron.weights.length + ' w · 1 b · ' + (isOutput ? 'linear' : ACTIVATION_NAMES[activation]) + ' · ' + n + ' × 1'
        }
      />
      <div className="mlx-detail__row">
        {grid ? <UnitPicture grid={grid} /> : null}
        <Formula tex={equation} />
      </div>
      <div className="mlx-scalars">
        {neuron.weights.map((w, i) => (
          <button
            key={i}
            type="button"
            className="mlx-scalar mlx-scalar--jump"
            title={(names[i] ?? 'input ' + (i + 1)) + ' → ' + title}
            onClick={() => jump({ kind: 'weight', layer: target.layer, neuron: target.neuron, input: i })}
          >
            <span className="mlx-scalar__label">w{sub(i + 1)}</span>
            <span className="mlx-scalar__value">{fmtCell(w, 3, true)}</span>
          </button>
        ))}
        <Scalar label="b" value={neuron.bias} />
        <Scalar label="Δb" value={(-learningRate * neuron.biasGradient) / Math.max(1, Math.min(batchSize, n))} decimals={4} />
      </div>
      {table(columns, inputColumns(target.layer))}
    </>
  );
}

/** The unit's own picture. Click it to see it large. */
function UnitPicture({ grid }: { grid: NeuronGrid }) {
  const [large, setLarge] = useState(false);
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const side = Math.min(width, height);
      const values = new Float32Array(grid.values.length);
      const scale = Math.max(Math.abs(grid.min), Math.abs(grid.max), 1e-6);
      for (let i = 0; i < grid.values.length; i++) values[i] = grid.values[i] / scale;
      paintFieldThumbnail(ctx, { x: 0, y: 0, w: side, h: side }, palette, values, grid.size);
      ctx.save();
      ctx.strokeStyle = rgba(palette.border, 1);
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, side - 1, side - 1);
      ctx.restore();
    },
    [grid],
  );

  return (
    <div
      className="mlx-detail__picture"
      data-large={large || undefined}
      role="button"
      tabIndex={0}
      title={large ? 'Shrink' : 'Enlarge'}
      onClick={() => setLarge((v) => !v)}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          setLarge((v) => !v);
        }
      }}
    >
      <Chart
        draw={draw}
        height={(width) => width}
        description="The unit's output across the input space."
        redrawKey={grid.spread + ':' + grid.size + ':' + large}
      />
    </div>
  );
}
