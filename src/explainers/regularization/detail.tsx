/** The cards: a feature and how its weight is updated, a data point, and each station of the model map with its live numbers. */

import { Annotation, DetailHead, Formula, Scalar, Scalars } from '../../explainer/components/Detail';
import { fmt, fmtCompact, fmtKnob, pearson, sup } from '../../lib/math/stats';
import { entryLambda } from '../../lib/ml/elasticNet';
import type { PathResult, Prepared } from '../../lib/ml/elasticNet';
import { penaltyName } from './map';
import type { ModelNode } from './targets';

export interface FeatureDetailProps {
  index: number;
  name: string;
  weights: readonly number[];
  ols: readonly number[];
  truth: readonly number[] | null;
  prep: Prepared;
  path: PathResult;
  lambda: number;
  alpha: number;
}

export function FeatureDetail({ index, name, weights, ols, truth, prep, path, lambda, alpha }: FeatureDetailProps) {
  const w = weights[index] ?? 0;
  // The signal: what the feature explains once the other weights have had their say.
  let rho = prep.xty[index];
  for (let k = 0; k < prep.p; k++) if (k !== index) rho -= prep.gram[index][k] * (weights[k] ?? 0);
  const column = prep.Xs.map((row) => row[index]);
  const corr = pearson(column, prep.yc);
  const enters = entryLambda(path, index);
  // A standardised column has x·x/n = 1; a raw one keeps its own scale in the denominator.
  const own = prep.gram[index][index];
  const base = Math.abs(own - 1) < 1e-6 ? '1' : fmt(own, 3);
  const note =
    alpha === 0
      ? 'Ridge divides the signal ρ for this weight by ' + base + ' + λ: shrunk, never zero.'
      : alpha === 1
        ? 'Lasso soft-thresholds ρ by λ: a signal within λ of zero leaves the weight at exactly zero.'
        : 'Elastic net soft-thresholds ρ by λα and divides by ' + base + ' + λ(1 − α): zeroed if small, shrunk if not.';
  return (
    <>
      <DetailHead
        eyebrow="Feature"
        title={name}
        shape={'w = ' + fmt(w, 3) + (truth ? ' · true ' + fmt(truth[index], 2) : '') + (w === 0 ? ' · zeroed' : '')}
      />
      <Annotation>{note} The dashed curve in the fit is this term on its own, w × {name}, around the mean of y.</Annotation>
      <Formula tex={'w_j \\leftarrow \\frac{S(\\rho_j,\\ \\lambda\\alpha)}{\\tfrac{1}{n} x_j^{\\top} x_j + \\lambda(1 - \\alpha)},\\qquad \\rho_j = \\tfrac{1}{n}\\, x_j^{\\top}\\Big(y - \\sum_{k \\ne j} w_k x_k\\Big)'} />
      <Scalars>
        <Scalar label="w" value={w} decimals={3} tone="accent" />
        <Scalar label="ρ" value={rho} decimals={3} />
        <Scalar label="λα" value={lambda * alpha} decimals={3} />
        <Scalar label="least squares" value={ols[index] ?? 0} decimals={3} />
        {truth ? <Scalar label="true" value={truth[index]} decimals={3} /> : null}
        <Scalar label="corr(x, y)" value={corr} decimals={3} />
        <Scalar label="enters at λ" value={enters === null ? 'never' : fmtKnob(enters)} />
      </Scalars>
    </>
  );
}

/* ---------------- a data point ---------------- */

export interface PointDetailProps {
  index: number;
  /** The row's inputs: x for the curve, the columns for the table. */
  inputs: Array<{ name: string; value: number }>;
  y: number;
  prediction: number;
  /** Whether the row was fitted or held out. */
  train: boolean;
  trainCount: number;
  testCount: number;
}

/** The point card: where it sits, what the model says there, and what that costs. */
export function PointDetail({ index, inputs, y, prediction, train, trainCount, testCount }: PointDetailProps) {
  const residual = y - prediction;
  const one = inputs.length === 1;
  return (
    <>
      <DetailHead
        eyebrow={train ? 'Training point' : 'Test point'}
        title={one ? inputs[0].name + ' = ' + fmt(inputs[0].value, 2) : 'row ' + (index + 1)}
        shape={(train ? 'one of ' + trainCount + ' fitted' : 'one of ' + testCount + ' held out') + (one ? ' · row ' + (index + 1) : '')}
      />
      <Annotation>
        {train
          ? 'The solver sees this point: its squared error is part of what the weights minimise.'
          : 'The solver never sees this point. It only scores the fit, which is what the test error measures.'}
      </Annotation>
      <Scalars>
        {one ? null : inputs.map((input) => <Scalar key={input.name} label={input.name} value={input.value} decimals={2} />)}
        <Scalar label="y" value={y} decimals={3} />
        <Scalar label="ŷ" value={prediction} decimals={3} tone="accent" />
        <Scalar label="y - ŷ" value={residual} decimals={3} />
        <Scalar label="(y − ŷ)²" value={residual * residual} decimals={4} />
      </Scalars>
    </>
  );
}

/* ---------------- a station of the map ---------------- */

export interface StationDetailProps {
  id: ModelNode;
  curve: boolean;
  featureNames: readonly string[];
  weights: readonly number[];
  prep: Prepared;
  trainCount: number;
  testCount: number;
  lambda: number;
  alpha: number;
  error: number;
  penalty: number;
  objective: number;
  trainMse: number;
  testMse: number;
  lambdaMax: number;
  epoch: number;
  unit: string;
}

const STATION_TEX: Record<ModelNode, string> = {
  data: '(x_i, y_i),\\quad i = 1 \\ldots n',
  features: '\\tilde{x}_j = \\frac{x_j - \\bar{x}_j}{s_j}',
  sum: '\\hat{y} = \\bar{y} + \\sum_j w_j \\tilde{x}_j',
  prediction: '\\hat{y}_i = \\bar{y} + w^{\\top} \\tilde{x}_i',
  error: '\\tfrac{1}{2n} \\sum_i (y_i - \\hat{y}_i)^2',
  penalty: '\\lambda \\Big( \\alpha \\lVert w \\rVert_1 + \\tfrac{1 - \\alpha}{2} \\lVert w \\rVert_2^2 \\Big)',
  objective: '\\tfrac{1}{2n} \\lVert y - \\hat{y} \\rVert^2 + \\lambda R(w)',
};

/** One station of the map: what it does, its formula, and its numbers right now. */
export function StationDetail(props: StationDetailProps) {
  const { id, curve, featureNames, weights, prep, trainCount, testCount, lambda, alpha, error, penalty, objective, trainMse, testMse, lambdaMax, epoch, unit } = props;
  const w = weights.map((v) => (Number.isFinite(v) ? v : 0));
  const active = w.filter((v) => v !== 0).length;
  const l1 = w.reduce((s, v) => s + Math.abs(v), 0);
  const l2 = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
  const rawPenalty = penalty / Math.max(lambda, 1e-12);
  const name = (j: number) => featureNames[j] ?? String(j + 1);
  const share = (j: number) => alpha * Math.abs(w[j]) + ((1 - alpha) / 2) * w[j] * w[j];
  const raw = prep.stds.every((s) => s === 1);
  const tex = STATION_TEX[id];

  switch (id) {
    case 'data':
      return (
        <>
          <DetailHead eyebrow="Station" title="Data" shape={trainCount + testCount + ' rows · ' + trainCount + ' fitted · ' + testCount + ' held out'} />
          <Annotation>
            {curve
              ? 'Every row is one x and its noisy y. The solver fits the blue dots; the orange rings are held back, so their error is the one to watch.'
              : 'Every row is two columns and a y. The solver fits three quarters of the rows; the rest are held back to score the fit.'}
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="n" value={String(trainCount)} tone="accent" />
            <Scalar label="held out" value={String(testCount)} />
            <Scalar label="ȳ" value={prep.yMean} decimals={3} />
          </Scalars>
        </>
      );
    case 'features':
      return (
        <>
          <DetailHead eyebrow="Station" title="Features" shape={prep.p + (curve ? ' powers of x' : ' columns') + ' · ' + active + ' non-zero'} />
          <Annotation>
            {curve ? 'Each row becomes x, x², … up to x' + sup(prep.p) + ', ' : 'Each column is '}
            {raw
              ? 'used as it is: a large column needs a small weight, so one λ charges every weight differently.'
              : 'centred and divided by its spread, so one λ means the same for every weight.'}
            {' A greyed cell is a weight at exactly zero: that term is out of the model.'}
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            {w.map((v, j) => (
              <Scalar key={j} label={'w(' + name(j) + ')'} value={v === 0 ? '0' : fmt(v, 3)} tone={v === 0 ? undefined : 'accent'} />
            ))}
          </Scalars>
        </>
      );
    case 'sum':
      return (
        <>
          <DetailHead eyebrow="Station" title="Weighted sum" shape={prep.p + ' weights + bias'} />
          <Annotation>Each feature times its weight, added up, plus the mean of y as the bias. The weights are what the solver chooses, and the only thing the penalty ever looks at.</Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="b = ȳ" value={prep.yMean} decimals={3} />
            <Scalar label="‖w‖₁" value={l1} decimals={3} tone="accent" />
            <Scalar label="‖w‖₂" value={l2} decimals={3} />
            <Scalar label="max |w|" value={Math.max(0, ...w.map(Math.abs))} decimals={3} />
          </Scalars>
        </>
      );
    case 'prediction':
      return (
        <>
          <DetailHead eyebrow="Station" title="Prediction" shape={curve ? 'the curve, at every x' : 'one number per row'} />
          <Annotation>
            {curve ? 'The solid curve is ŷ at every x. ' : 'ŷ for every row. '}
            At each fitted row ŷ is compared with y; at each held-out row it is only scored.
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="MSE fitted" value={fmtCompact(trainMse)} />
            <Scalar label="MSE held out" value={fmtCompact(testMse)} tone="accent" />
          </Scalars>
        </>
      );
    case 'error':
      return (
        <>
          <DetailHead eyebrow="Station" title="Error" shape={'½ MSE on ' + trainCount + ' fitted rows'} />
          <Annotation>
            Half the mean squared error on the fitted rows: the smooth part of what the solver minimises, and what the penalty trades against. The held-out error says whether the trade was worth it.
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="error" value={fmtCompact(error)} tone="accent" />
            <Scalar label="MSE fitted" value={fmtCompact(trainMse)} />
            <Scalar label="MSE held out" value={fmtCompact(testMse)} />
          </Scalars>
        </>
      );
    case 'penalty':
      return (
        <>
          <DetailHead eyebrow="Station" title="Penalty" shape={'λ ' + fmtKnob(lambda) + ' · ' + penaltyName(alpha) + ' ' + fmtCompact(rawPenalty)} />
          <Annotation>
            {alpha === 1
              ? 'The lasso charges λ for every unit of |w|. A weight whose signal is worth less than λ lands on exactly zero.'
              : alpha === 0
                ? 'Ridge charges λ for every unit of ½w². Doubling a weight costs four times more, so the largest weights are cut hardest; none reaches zero.'
                : 'A mix: α of the charge is L1, which makes zeros, and the rest L2, which shrinks. λ scales the whole thing.'}
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="λ" value={fmtKnob(lambda)} />
            {alpha > 0 && alpha < 1 ? <Scalar label="α" value={alpha.toFixed(1)} /> : null}
            <Scalar label={penaltyName(alpha)} value={rawPenalty} decimals={3} />
            <Scalar label="λ · R(w)" value={fmtCompact(penalty)} tone="accent" />
            <Scalar label="λ max" value={fmtKnob(lambdaMax)} />
          </Scalars>
          <Annotation>Each weight’s share of R(w):</Annotation>
          <Scalars>
            {w.map((v, j) => (
              <Scalar key={j} label={name(j)} value={v === 0 ? '0' : fmt(share(j), 3)} />
            ))}
          </Scalars>
        </>
      );
    default:
      return (
        <>
          <DetailHead eyebrow="Station" title="Objective" shape={fmtCompact(error) + ' + ' + fmtCompact(penalty) + ' = ' + fmtCompact(objective)} />
          <Annotation>
            What the solver minimises: the error plus λ times the penalty.{' '}
            {unit === 'step'
              ? 'Ridge has a closed form, one step.'
              : unit === 'epoch'
                ? 'Proximal gradient steps down the error, then shrinks every weight by the penalty.'
                : 'Coordinate descent solves the weights one at a time, sweep after sweep.'}
            {epoch > 0 ? ' Reached after ' + epoch + ' ' + unit + (epoch === 1 ? '' : 's') + '.' : ''}
          </Annotation>
          <Formula tex={tex} />
          <Scalars>
            <Scalar label="error" value={fmtCompact(error)} />
            <Scalar label="penalty" value={fmtCompact(penalty)} />
            <Scalar label="objective" value={fmtCompact(objective)} tone="accent" />
          </Scalars>
        </>
      );
  }
}
