/** Detail cards for the linear-regression graph. */

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { DetailHead, Formula, Scalar, Scalars, VectorTable, WeightedSumTable } from '../../explainer/components/Detail';
import { texLines, texNum, texSum, texSymbol, texSymbolicSum } from '../../explainer/tex';
import type { VectorColumn } from '../../explainer/vectors';
import { maxAbs } from '../../explainer/vectors';
import { fmt, fmtCell, fmtCompact } from '../../lib/math/stats';
import type { LinRegConfig, LinRegState, Normaliser } from '../../lib/ml/linearRegression';
import {
  LR_EDGES,
  LR_NODES,
  dataEdgeId,
  inputId,
  powerLabel,
  squaredErrors,
  weightEdgeId,
  weightLabel,
} from './graph';
import type { LinRegVectors } from './graph';

/** The update each rule applies to every weight, the bias included. */
const STEP_TEX: Record<string, string> = {
  gd: '\\begin{aligned} w &\\leftarrow w - \\alpha\\,\\frac{\\partial L}{\\partial w} \\\\ b &\\leftarrow b - \\alpha\\,\\frac{\\partial L}{\\partial b} \\end{aligned}',
  momentum: '\\begin{aligned} v &\\leftarrow \\beta v - \\alpha\\,\\frac{\\partial L}{\\partial w} \\\\ w &\\leftarrow w + v \\end{aligned}',
  adam: '\\begin{aligned} m &\\leftarrow \\beta m + (1 - \\beta)\\,g, \\quad v \\leftarrow 0.999\\,v + 0.001\\,g^2 \\\\ w &\\leftarrow w - \\alpha\\,\\hat{m} / (\\sqrt{\\hat{v}} + \\epsilon) \\end{aligned}',
};

export interface LinRegDetailProps {
  selection: ArchSelection;
  state: LinRegState;
  config: LinRegConfig;
  norm: Normaliser;
  /** The inputs are x̃ = (x − μ) / s rather than x. */
  standardised: boolean;
  vectors: LinRegVectors;
  order: readonly number[];
  batch: ReadonlySet<number> | null;
  /** Opens another component's card. */
  jump: (next: ArchSelection) => void;
}

export function LinRegDetail({
  selection,
  state,
  config,
  norm,
  standardised,
  vectors,
  order,
  batch,
  jump,
}: LinRegDetailProps) {
  const degree = config.degree;
  const n = vectors.x.length;
  const weights = state.weights;
  const weightNames = weights.map((_, i) => weightLabel(i, degree));
  const powers = Array.from({ length: degree }, (_, k) => k + 1);
  const openWeight = (i: number) => jump({ kind: 'edge', id: i === 0 ? LR_EDGES.bias : weightEdgeId(i) });
  const plain = config.optimiser !== 'momentum' && config.optimiser !== 'adam';

  const xCol: VectorColumn = { name: 'x', values: vectors.x };
  const yCol: VectorColumn = { name: 'y', values: vectors.y, scale: vectors.yScale };
  const yhatCol: VectorColumn = { name: 'ŷ', values: vectors.prediction, scale: vectors.yScale };
  /** The input at power k, named as the node is. */
  const inputName = (k: number) => powerLabel(k, standardised);
  const inputCol = (k: number): VectorColumn => ({ name: inputName(k), values: vectors.features[k] });
  const inputTex = (k: number) => texSymbol(inputName(k));
  const table = (columns: VectorColumn[], extra?: VectorColumn[]) => (
    <VectorTable key={columns.map((c) => c.name).join()} columns={columns} order={order} active={batch} extra={extra} limit={10} />
  );

  /* The equation of the unit, symbols then numbers. */
  const terms = powers.map((k) => ({ coef: weights[k] ?? 0, symbol: inputTex(k) }));
  const symbolic =
    degree === 1 ? 'w' + inputTex(1) + ' + b' : texSymbolicSum(powers.map(inputTex), standardised ? '\\tilde{\\mathbf{x}}' : '\\mathbf{x}');
  const numeric = texSum(terms, weights[0] ?? 0, { perLine: 3 });
  const prediction = texLines([['\\hat{y}', symbolic], ['', numeric]]);
  const standardiseTex = '(x - \\mu) / s';
  const standardiseNumeric = '(x - ' + texNum(norm.mean) + ') / ' + texNum(norm.std);
  const normChips = standardised ? (
    <Scalars>
      <Scalar label="μ" value={norm.mean} decimals={2} />
      <Scalar label="s" value={norm.std} decimals={2} />
    </Scalars>
  ) : null;

  /** What input k is made of: x̃ = (x − μ)/s, x̃² = ((x − μ)/s)², x² = x·x. */
  const inputCard = (k: number, eyebrow: string, title: string) => {
    const formula = standardised
      ? k === 1
        ? inputTex(1) + ' = ' + standardiseTex + ' = ' + standardiseNumeric
        : inputTex(k) + ' = \\big(' + standardiseTex + '\\big)^{' + k + '}'
      : k === 1
        ? null
        : inputTex(k);
    return (
      <>
        <DetailHead eyebrow={eyebrow} title={title} shape={n + ' × 1 · → Σ'} />
        {formula ? <Formula tex={formula} /> : null}
        {normChips}
        {table(standardised || k > 1 ? [xCol, inputCol(k)] : [xCol])}
      </>
    );
  };

  if (selection.kind === 'edge') {
    if (selection.id === LR_EDGES.gradient) {
      return (
        <>
          <DetailHead eyebrow="Gradient" title="∂L/∂w, ∂L/∂b" shape={weights.length + ' × 1'} />
          <Formula tex={STEP_TEX[config.optimiser] ?? STEP_TEX.gd} />
          <Scalars>
            <Scalar label="‖∇L‖" value={state.epoch === 0 ? 'n/a' : fmtCompact(state.gradientNorm)} tone="accent" />
            <Scalar label="α" value={config.learningRate} />
          </Scalars>
          <WeightedSumTable
            rows={weightNames.map((name, i) => {
              const g = state.lastGradient[i] ?? 0;
              // Momentum and Adam scale each gradient their own way; the factor is what the step applied.
              const factor = plain || g === 0 ? -config.learningRate : (state.lastStep[i] ?? 0) / g;
              return { name, value: g, weight: factor };
            })}
            columns={['w, b', '∂L/∂w, ∂L/∂b', plain ? '× −α' : '× step', 'Δw, Δb']}
            noTotal
            decimals={3}
            onSelectInput={openWeight}
          />
        </>
      );
    }
    const power = powers.find((k) => weightEdgeId(k) === selection.id);
    if (selection.id === LR_EDGES.bias || power !== undefined) {
      const index = power ?? 0;
      const w = weights[index] ?? 0;
      const g = state.lastGradient[index] ?? 0;
      const symbol = index === 0 ? '1' : inputTex(index);
      return (
        <>
          <DetailHead
            eyebrow={'Weight ' + (index + 1) + ' / ' + weights.length}
            title={weightNames[index]}
            shape={(index === 0 ? '1' : inputName(index)) + ' → Σ'}
          />
          <Formula tex={texSymbol(weightNames[index]) + '\\,' + symbol + ' = ' + texNum(w) + '\\,' + symbol} />
          <Scalars>
            <Scalar label={weightNames[index]} value={w} decimals={2} tone="accent" />
            <Scalar label={'∂L/∂' + weightNames[index]} value={g} />
            <Scalar label={'Δ' + weightNames[index]} value={plain ? -config.learningRate * g : state.lastStep[index] ?? 0} decimals={4} />
          </Scalars>
          {index === 0
            ? null
            : table([inputCol(index), { name: weightNames[index] + '·' + inputName(index), values: vectors.features[index].map((v) => v * w) }])}
        </>
      );
    }
    if (selection.id === LR_EDGES.toLoss || selection.id === LR_EDGES.output) {
      return (
        <>
          <DetailHead eyebrow="Wire" title="ŷ" shape={n + ' × 1'} />
          {table([yhatCol])}
        </>
      );
    }
    if (selection.id === LR_EDGES.targetToLoss) {
      return (
        <>
          <DetailHead eyebrow="Wire" title="y" shape={n + ' × 1'} />
          {table([yCol])}
        </>
      );
    }
    const fed = powers.find((k) => dataEdgeId(k) === selection.id);
    if (fed !== undefined) return inputCard(fed, 'Wire', 'x → ' + inputName(fed));
    return null;
  }

  const power = powers.find((k) => inputId(k) === selection.id);
  if (power !== undefined) return inputCard(power, 'Input ' + power + ' / ' + (degree + 1), inputName(power));

  switch (selection.id) {
    case LR_NODES.data:
      return (
        <>
          <DetailHead eyebrow="Data" title="X, y" shape={n + ' × 2'} />
          {normChips}
          {table([xCol, yCol])}
        </>
      );
    case LR_NODES.one:
      return (
        <>
          <DetailHead eyebrow={'Input ' + (degree + 1) + ' / ' + (degree + 1)} title="1" shape={n + ' × 1 · → Σ'} />
          <Formula tex={'b \\cdot 1 = ' + texNum(weights[0] ?? 0)} />
          <Scalars>
            <Scalar label="b" value={weights[0] ?? 0} decimals={2} tone="accent" />
          </Scalars>
        </>
      );
    case LR_NODES.sum:
      return (
        <>
          <DetailHead eyebrow="Unit" title="Σ" shape={weights.length + ' w · ' + n + ' × 1'} />
          <Formula tex={prediction} />
          <div className="mlx-scalars">
            {weightNames.map((name, i) => (
              <button
                key={name}
                type="button"
                className="mlx-scalar mlx-scalar--jump"
                title={(i === 0 ? '1' : inputName(i)) + ' → Σ'}
                onClick={() => openWeight(i)}
              >
                <span className="mlx-scalar__label">{name}</span>
                <span className="mlx-scalar__value">{fmtCell(weights[i] ?? 0, 2, true)}</span>
              </button>
            ))}
          </div>
          {table([yhatCol], powers.map(inputCol))}
        </>
      );
    case LR_NODES.yhat:
      return (
        <>
          <DetailHead eyebrow="Output" title="ŷ" shape={n + ' × 1'} />
          <Formula tex={'\\hat{y} = ' + numeric} />
          {table([yhatCol, yCol, { name: 'ŷ − y', values: vectors.prediction.map((p, i) => p - vectors.y[i]) }])}
        </>
      );
    case LR_NODES.target:
      return (
        <>
          <DetailHead eyebrow="Target" title="y" shape={n + ' × 1'} />
          {table([yCol])}
        </>
      );
    case LR_NODES.loss: {
      const errors = squaredErrors(vectors);
      return (
        <>
          <DetailHead eyebrow="Loss" title="L" shape={'n = ' + n} />
          <Formula
            tex={
              'L = \\frac{1}{n} \\sum_i (\\hat{y}_i - y_i)^2' +
              (config.l2 > 0 ? ' + \\lambda \\sum_j w_j^2' : '') +
              (state.epoch === 0 || !Number.isFinite(state.loss) ? '' : ' = ' + texNum(state.loss, state.loss >= 100 ? 0 : 2))
            }
          />
          <Scalars>
            <Scalar label="L" value={state.epoch === 0 ? 'n/a' : fmtCompact(state.loss)} tone="accent" />
            <Scalar label="max" value={fmt(maxAbs(errors), 2)} />
          </Scalars>
          {table([yhatCol, yCol, { name: '(ŷ − y)²', values: errors, ramp: 'magnitude', scale: maxAbs(errors) }])}
        </>
      );
    }
    default:
      return null;
  }
}
