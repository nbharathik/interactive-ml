/** Detail cards for the logistic-regression graph. */

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { DetailHead, Formula, Scalar, Scalars, VectorTable, WeightedSumTable } from '../../explainer/components/Detail';
import { texLines, texNum, texSum, texSymbol, texSymbolicSum } from '../../explainer/tex';
import type { VectorColumn } from '../../explainer/vectors';
import { maxAbs } from '../../explainer/vectors';
import { fmt, fmtCell, fmtCompact, fmtPercent } from '../../lib/math/stats';
import type { LogRegConfig, LogRegState, Standardiser } from '../../lib/ml/logisticRegression';
import { LOG_EDGES, LOG_NODES, featureEdgeId, featureId, fedName, weightEdgeId, weightLabel } from './graph';
import type { LogRegVectors } from './graph';

/** Both update rules, since the bias trains with the weights. */
const GRADIENT_STEP_TEX =
  '\\begin{aligned} w &\\leftarrow w - \\alpha\\,\\frac{\\partial L}{\\partial w} \\\\ b &\\leftarrow b - \\alpha\\,\\frac{\\partial L}{\\partial b} \\end{aligned}';

export interface LogRegDetailProps {
  selection: ArchSelection;
  state: LogRegState;
  config: LogRegConfig;
  /** Raw feature names in weight order: '1', 'x₁', 'x₂', ... */
  featureNames: readonly string[];
  std: Standardiser;
  /** The inputs are x̃ = (x − μ) / s rather than x. */
  standardised: boolean;
  vectors: LogRegVectors;
  order: readonly number[];
  batch: ReadonlySet<number> | null;
  accuracy: number;
  logLoss: number;
  /** Opens another component's card. */
  jump: (next: ArchSelection) => void;
}

export function LogRegDetail({
  selection,
  state,
  config,
  featureNames: rawNames,
  std,
  standardised,
  vectors,
  order,
  batch,
  accuracy,
  logLoss,
  jump,
}: LogRegDetailProps) {
  const n = vectors.label.length;
  const weights = state.weights;
  const names = rawNames.map((name) => fedName(name, standardised));
  const weightNames = names.map((_, i) => weightLabel(i));
  const openWeight = (i: number) => jump({ kind: 'edge', id: weightEdgeId(i) });

  const x1: VectorColumn = { name: 'x₁', values: vectors.x1 };
  const x2: VectorColumn = { name: 'x₂', values: vectors.x2 };
  const y: VectorColumn = { name: 'y', values: vectors.label, ramp: 'class' };
  const z: VectorColumn = { name: 'z', values: vectors.z };
  const p: VectorColumn = { name: 'p', values: vectors.p, ramp: 'probability' };
  const called: VectorColumn = { name: 'ŷ', values: vectors.called, ramp: 'class' };
  const inputCol = (i: number): VectorColumn => ({ name: names[i], values: vectors.features[i] });
  const inputTex = (i: number) => texSymbol(names[i]);
  const inputs = names.map((_, i) => i).filter((i) => i > 0);
  const table = (columns: VectorColumn[], extra?: VectorColumn[]) => (
    <VectorTable
      key={columns.map((c) => c.name).join()}
      columns={columns}
      order={order}
      active={batch}
      classes={vectors.label}
      extra={extra}
    />
  );
  const counts = (
    <Scalars>
      <Scalar label="n₀" value={vectors.label.filter((v) => v === 0).length} decimals={0} />
      <Scalar label="n₁" value={vectors.label.filter((v) => v === 1).length} decimals={0} />
    </Scalars>
  );

  /* The equation of the unit, symbols then numbers. */
  const terms = inputs.map((i) => ({ coef: weights[i] ?? 0, symbol: inputTex(i) }));
  const symbolic = texSymbolicSum(inputs.map(inputTex), standardised ? '\\tilde{\\mathbf{x}}' : '\\mathbf{x}', 2);
  const numeric = texSum(terms, weights[0] ?? 0, { perLine: 3 });
  const score = texLines([['z', symbolic], ['', numeric]]);
  const rule = 'p \\ge ' + texNum(config.threshold);

  /** What input i is made of: x̃₁ = (x₁ − μ₁)/s₁, x̃₁² = (x̃₁)², x₁x₂ = x₁·x₂. */
  const inputCard = (i: number, eyebrow: string, title: string) => {
    const raw = rawNames[i];
    const usesX1 = raw.includes('x₁');
    const usesX2 = raw.includes('x₂');
    const rawCols = [...(usesX1 ? [x1] : []), ...(usesX2 ? [x2] : [])];
    const scaled = (axis: 1 | 2) =>
      '(x_' + axis + ' - ' + texNum(axis === 1 ? std.meanX : std.meanY) + ') / ' + texNum(axis === 1 ? std.stdX : std.stdY);
    const derived = raw !== 'x₁' && raw !== 'x₂';
    let formula: string | null = null;
    if (standardised && !derived) {
      const axis = raw === 'x₁' ? 1 : 2;
      formula = inputTex(i) + ' = (x_' + axis + ' - \\mu_' + axis + ') / s_' + axis + ' = ' + scaled(axis);
    } else if (standardised) {
      const expanded = texSymbol(raw)
        .replace(/x_\{1\}/g, '\\big(' + scaled(1) + '\\big)')
        .replace(/x_\{2\}/g, '\\big(' + scaled(2) + '\\big)');
      formula = inputTex(i) + ' = ' + expanded;
    }
    const chips = standardised ? (
      <Scalars>
        {usesX1 ? <Scalar label="μ₁" value={std.meanX} decimals={2} /> : null}
        {usesX1 ? <Scalar label="s₁" value={std.stdX} decimals={2} /> : null}
        {usesX2 ? <Scalar label="μ₂" value={std.meanY} decimals={2} /> : null}
        {usesX2 ? <Scalar label="s₂" value={std.stdY} decimals={2} /> : null}
      </Scalars>
    ) : null;
    return (
      <>
        <DetailHead eyebrow={eyebrow} title={title} shape={n + ' × 1 · → Σ'} />
        {formula ? <Formula tex={formula} /> : null}
        {chips}
        {table([...(standardised || derived ? rawCols : []), inputCol(i), y])}
      </>
    );
  };

  if (selection.kind === 'edge') {
    if (selection.id === LOG_EDGES.gradient) {
      return (
        <>
          <DetailHead eyebrow="Gradient" title="∂L/∂w, ∂L/∂b" shape={weights.length + ' × 1'} />
          <Formula tex={GRADIENT_STEP_TEX} />
          <Scalars>
            <Scalar label="‖∇L‖" value={state.epoch === 0 ? 'n/a' : fmtCompact(state.gradientNorm)} tone="accent" />
            <Scalar label="α" value={config.learningRate} />
          </Scalars>
          <WeightedSumTable
            rows={weightNames.map((name, i) => ({
              name,
              value: state.lastGradient[i] ?? 0,
              weight: -config.learningRate,
            }))}
            columns={['w, b', '∂L/∂w, ∂L/∂b', '× −α', 'Δw, Δb']}
            noTotal
            decimals={3}
            onSelectInput={openWeight}
          />
        </>
      );
    }
    const index = names.findIndex((_, i) => weightEdgeId(i) === selection.id);
    if (index >= 0) {
      const w = weights[index] ?? 0;
      const g = state.lastGradient[index] ?? 0;
      const symbol = index === 0 ? '1' : inputTex(index);
      return (
        <>
          <DetailHead
            eyebrow={'Weight ' + (index + 1) + ' / ' + weights.length}
            title={weightNames[index]}
            shape={names[index] + ' → Σ'}
          />
          <Formula tex={texSymbol(weightNames[index]) + '\\,' + symbol + ' = ' + texNum(w) + '\\,' + symbol} />
          <Scalars>
            <Scalar label={weightNames[index]} value={w} decimals={2} tone="accent" />
            <Scalar label={'∂L/∂' + weightNames[index]} value={g} />
            <Scalar label={'Δ' + weightNames[index]} value={-config.learningRate * g} decimals={4} />
          </Scalars>
          {index === 0
            ? null
            : table([inputCol(index), { name: weightNames[index] + '·' + names[index], values: vectors.features[index].map((v) => v * w) }, y])}
        </>
      );
    }
    if (selection.id === LOG_EDGES.toDecision) {
      return (
        <>
          <DetailHead eyebrow="Decision rule" title={'p ≥ ' + fmt(config.threshold, 2)} shape={n + ' × 1'} />
          <Formula tex={'\\hat{y} = [\\,' + rule + '\\,]'} />
          {table([p, called, y])}
        </>
      );
    }
    const fed = names.findIndex((_, i) => i > 0 && featureEdgeId(i) === selection.id);
    if (fed > 0) return inputCard(fed, 'Wire', 'X → ' + names[fed]);
    const carried: Record<string, { title: string; columns: VectorColumn[] }> = {
      [LOG_EDGES.toSigmoid]: { title: 'z', columns: [z, y] },
      [LOG_EDGES.toProb]: { title: 'p', columns: [p, y] },
      [LOG_EDGES.toLoss]: { title: 'p', columns: [p, y] },
      [LOG_EDGES.targetToLoss]: { title: 'y', columns: [y] },
    };
    const wire = carried[selection.id];
    if (wire) {
      return (
        <>
          <DetailHead eyebrow="Wire" title={wire.title} shape={n + ' × 1'} />
          {table(wire.columns)}
        </>
      );
    }
    return null;
  }

  const featureIndex = names.findIndex((_, i) => featureId(i) === selection.id);
  if (featureIndex > 0) {
    return inputCard(featureIndex, 'Input ' + featureIndex + ' / ' + names.length, names[featureIndex]);
  }

  switch (selection.id) {
    case LOG_NODES.data:
      return (
        <>
          <DetailHead eyebrow="Data" title="X, y" shape={n + ' × 3'} />
          {counts}
          {table([x1, x2, y])}
        </>
      );
    case featureId(0):
      return (
        <>
          <DetailHead eyebrow={'Input ' + names.length + ' / ' + names.length} title="1" shape={n + ' × 1 · → Σ'} />
          <Formula tex={'b \\cdot 1 = ' + texNum(weights[0] ?? 0)} />
          <Scalars>
            <Scalar label="b" value={weights[0] ?? 0} decimals={2} tone="accent" />
          </Scalars>
        </>
      );
    case LOG_NODES.score:
      return (
        <>
          <DetailHead eyebrow="Unit" title="Σ" shape={weights.length + ' w · ' + n + ' × 1'} />
          <Formula tex={score} />
          <div className="mlx-scalars">
            {weightNames.map((name, i) => (
              <button
                key={name}
                type="button"
                className="mlx-scalar mlx-scalar--jump"
                title={names[i] + ' → Σ'}
                onClick={() => openWeight(i)}
              >
                <span className="mlx-scalar__label">{name}</span>
                <span className="mlx-scalar__value">{fmtCell(weights[i] ?? 0, 2, true)}</span>
              </button>
            ))}
          </div>
          {table([z, y], inputs.map(inputCol))}
        </>
      );
    case LOG_NODES.sigmoid:
      return (
        <>
          <DetailHead eyebrow="Squash" title="σ" shape={n + ' × 1 → ' + n + ' × 1'} />
          <Formula tex={'p = \\sigma(z) = \\frac{1}{1 + e^{-z}}'} />
          {table([z, p, y])}
        </>
      );
    case LOG_NODES.prob:
      return (
        <>
          <DetailHead eyebrow="Output" title="p" shape={n + ' × 1 · p ∈ (0, 1)'} />
          <Formula tex={'p = \\sigma(z)'} />
          {table([p, y])}
        </>
      );
    case LOG_NODES.decision:
      return (
        <>
          <DetailHead eyebrow="Decision" title="ŷ" shape={n + ' × 1 · ŷ ∈ {0, 1}'} />
          <Formula tex={'\\hat{y} = [\\,' + rule + '\\,]'} />
          <Scalars>
            <Scalar label="ŷ = y" value={state.epoch === 0 ? 'n/a' : fmtPercent(accuracy, 1)} tone="accent" />
          </Scalars>
          {table([p, called, y])}
        </>
      );
    case LOG_NODES.target:
      return (
        <>
          <DetailHead eyebrow="Target" title="y" shape={n + ' × 1 · y ∈ {0, 1}'} />
          {counts}
          {table([y])}
        </>
      );
    case LOG_NODES.loss:
      return (
        <>
          <DetailHead eyebrow="Loss" title="L" shape={'n = ' + n} />
          <Formula
            tex={
              'L = \\frac{1}{n} \\sum_i -\\log p_i(y_i)' +
              (state.epoch === 0 || !Number.isFinite(logLoss) ? '' : ' = ' + texNum(logLoss, 3))
            }
          />
          <Scalars>
            <Scalar label="L" value={state.epoch === 0 ? 'n/a' : fmtCompact(logLoss)} tone="accent" />
            <Scalar label="max" value={fmt(maxAbs(vectors.loss), 2)} />
          </Scalars>
          {table([p, y, { name: '−log p(y)', values: vectors.loss, ramp: 'magnitude', scale: maxAbs(vectors.loss) }])}
        </>
      );
    default:
      return null;
  }
}
