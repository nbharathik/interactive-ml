/** Detail cards for the polynomial-regression graph. */

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { DetailHead, Formula, Scalar, Scalars, VectorTable } from '../../explainer/components/Detail';
import { texLines, texNum, texSum, texSymbol, texSymbolicSum } from '../../explainer/tex';
import type { VectorColumn } from '../../explainer/vectors';
import { maxAbs } from '../../explainer/vectors';
import { fmt, fmtCell, fmtCompact } from '../../lib/math/stats';
import type { PolyFit } from '../../lib/ml/polynomialRegression';
import {
  PR_EDGES,
  PR_NODES,
  dataEdgeId,
  inputId,
  powerLabel,
  squaredErrors,
  weightEdgeId,
  weightLabel,
} from './graph';
import type { PolyVectors } from './graph';

/** The exact solution, with the bias left out of the penalty. */
/** The bias is never charged, so the ridge term skips its diagonal entry. */
const NORMAL_EQUATIONS_TEX =
  '\\mathbf{w} = (\\Phi^{\\mathsf{T}}\\Phi + \\lambda n\\,\\mathrm{diag}(0, 1, \\dots, 1))^{-1}\\,\\Phi^{\\mathsf{T}}\\mathbf{y}';

export interface PolyDetailProps {
  selection: ArchSelection;
  fit: PolyFit;
  l2: number;
  standardised: boolean;
  vectors: PolyVectors;
  order: readonly number[];
  trainError: number;
  testError: number;
  jump: (next: ArchSelection) => void;
}

export function PolyDetail({ selection, fit, l2, standardised, vectors, order, trainError, testError, jump }: PolyDetailProps) {
  const degree = fit.degree;
  const n = vectors.x.length;
  const weights = fit.weights;
  const weightNames = weights.map((_, i) => weightLabel(i, degree));
  const powers = Array.from({ length: degree }, (_, k) => k + 1);
  const openWeight = (i: number) => jump({ kind: 'edge', id: i === 0 ? PR_EDGES.bias : weightEdgeId(i) });

  const xCol: VectorColumn = { name: 'x', values: vectors.x };
  const yCol: VectorColumn = { name: 'y', values: vectors.y, scale: vectors.yScale };
  const yhatCol: VectorColumn = { name: 'ŷ', values: vectors.prediction, scale: vectors.yScale };
  const inputName = (k: number) => powerLabel(k, standardised);
  const inputCol = (k: number): VectorColumn => ({ name: inputName(k), values: vectors.features[k] });
  const inputTex = (k: number) => texSymbol(inputName(k));
  const table = (columns: VectorColumn[], extra?: VectorColumn[]) => (
    <VectorTable key={columns.map((c) => c.name).join()} columns={columns} order={order} extra={extra} />
  );

  const terms = powers.map((k) => ({ coef: weights[k] ?? 0, symbol: inputTex(k) }));
  const symbolic =
    degree === 0
      ? 'b'
      : degree === 1
        ? 'w' + inputTex(1) + ' + b'
        : texSymbolicSum(powers.map(inputTex), standardised ? '\\tilde{\\mathbf{x}}' : '\\mathbf{x}');
  const numeric = texSum(terms, weights[0] ?? 0, { perLine: 3 });
  const prediction = texLines([['\\hat{y}', symbolic], ['', numeric]]);
  const standardiseTex = '(x - \\mu) / s';
  const standardiseNumeric = '(x - ' + texNum(fit.norm.mean) + ') / ' + texNum(fit.norm.std);
  const normChips = standardised ? (
    <Scalars>
      <Scalar label="μ" value={fit.norm.mean} decimals={2} />
      <Scalar label="s" value={fit.norm.std} decimals={2} />
    </Scalars>
  ) : null;

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
        <Scalars>
          {standardised ? <Scalar label="μ" value={fit.norm.mean} decimals={2} /> : null}
          {standardised ? <Scalar label="s" value={fit.norm.std} decimals={2} /> : null}
          <Scalar label={'max |' + inputName(k) + '|'} value={fmtCell(maxAbs(vectors.features[k]), 2, true)} tone="accent" />
        </Scalars>
        {table(standardised || k > 1 ? [xCol, inputCol(k)] : [xCol])}
      </>
    );
  };

  if (selection.kind === 'edge') {
    const power = powers.find((k) => weightEdgeId(k) === selection.id);
    if (selection.id === PR_EDGES.bias || power !== undefined) {
      const index = power ?? 0;
      const w = weights[index] ?? 0;
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
            {index > 0 && l2 > 0 ? <Scalar label="λw²" value={l2 * w * w} decimals={4} /> : null}
          </Scalars>
          {index === 0
            ? null
            : table([inputCol(index), { name: weightNames[index] + '·' + inputName(index), values: vectors.features[index].map((v) => v * w) }])}
        </>
      );
    }
    if (selection.id === PR_EDGES.toLoss || selection.id === PR_EDGES.output) {
      return (
        <>
          <DetailHead eyebrow="Wire" title="ŷ" shape={n + ' × 1'} />
          {table([yhatCol])}
        </>
      );
    }
    if (selection.id === PR_EDGES.targetToLoss) {
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
    case PR_NODES.data:
      return (
        <>
          <DetailHead eyebrow="Data" title="X, y" shape={n + ' × 2'} />
          {normChips}
          {table([xCol, yCol])}
        </>
      );
    case PR_NODES.one:
      return (
        <>
          <DetailHead eyebrow={'Input ' + (degree + 1) + ' / ' + (degree + 1)} title="1" shape={n + ' × 1 · → Σ'} />
          <Formula tex={'b \\cdot 1 = ' + texNum(weights[0] ?? 0)} />
          <Scalars>
            <Scalar label="b" value={weights[0] ?? 0} decimals={2} tone="accent" />
          </Scalars>
        </>
      );
    case PR_NODES.sum:
      return (
        <>
          <DetailHead eyebrow="Unit" title="Σ" shape={weights.length + ' w · ' + n + ' × 1'} />
          <Formula tex={prediction} />
          <Formula tex={NORMAL_EQUATIONS_TEX + (l2 > 0 ? ',\\quad \\lambda = ' + texNum(l2, 3) : ',\\quad \\lambda = 0')} />
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
    case PR_NODES.yhat:
      return (
        <>
          <DetailHead eyebrow="Output" title="ŷ" shape={n + ' × 1'} />
          <Formula tex={'\\hat{y} = ' + numeric} />
          {table([yhatCol, yCol, { name: 'ŷ − y', values: vectors.prediction.map((p, i) => p - vectors.y[i]) }])}
        </>
      );
    case PR_NODES.target:
      return (
        <>
          <DetailHead eyebrow="Target" title="y" shape={n + ' × 1'} />
          {table([yCol])}
        </>
      );
    case PR_NODES.loss: {
      const errors = squaredErrors(vectors);
      return (
        <>
          <DetailHead eyebrow="Error" title="L" shape={'n = ' + n} />
          <Formula
            tex={
              'L = \\frac{1}{n} \\sum_i (\\hat{y}_i - y_i)^2' +
              (Number.isFinite(trainError) ? ' = ' + texNum(trainError, trainError >= 100 ? 0 : 3) : '')
            }
          />
          <Scalars>
            <Scalar label="train" value={Number.isFinite(trainError) ? fmtCompact(trainError) : 'n/a'} tone="accent" />
            <Scalar label="test" value={Number.isFinite(testError) ? fmtCompact(testError) : 'n/a'} />
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
