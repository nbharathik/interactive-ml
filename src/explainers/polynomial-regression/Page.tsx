/** Polynomial regression explainer: the degree knob, train against test error, bias against variance. */

import { useCallback, useMemo, useRef, useState } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import { ArchitectureView, parseArchTarget } from '../../explainer/components/ArchitectureView';
import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { Legend, Math as Tex, MetricStrip, Panel, ParamsReset, ViewSwitch } from '../../explainer/components/Panels';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { Transport } from '../../explainer/components/Transport';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { useSimulation } from '../../explainer/useSimulation';
import { useEditable } from '../../explainer/useEditable';
import type { GuideStatus, Metric } from '../../explainer/types';
import { displayOrder, maxAbs } from '../../explainer/vectors';
import { texNum } from '../../explainer/tex';
import { usePalette } from '../../components/ThemeProvider';
import { getCurveDataset, testSeed } from '../../lib/datasets/curves';
import { fmt, fmtCompact, rSquared } from '../../lib/math/stats';
import {
  biasVarianceSweep,
  createRefitState,
  featureRow,
  fitPolynomial,
  largestFeature,
  meanSquaredError,
  predictFit,
  refitStep,
  validationCurve,
} from '../../lib/ml/polynomialRegression';
import type { PolyConfig } from '../../lib/ml/polynomialRegression';
import type { Frame } from '../../lib/viz/canvas';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, MAX_DEGREE, PRESETS } from './config';
import type { PolyParams } from './config';
import { BiasVarianceChart, FitPlot, ValidationCurveChart } from './panels';
import { buildPolyGraph, powerLabel } from './graph';
import type { PolyVectors } from './graph';
import { PolyDetail } from './detail';
import { LESSONS, makePolyContext } from './lessons';
import type { SecondView } from './lessons';
import { BV_GRID, CURVE_GRID, innerBiasVariance, samplerFor } from './model';

const META = getExplainer('polynomial-regression')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);

export default function PolynomialRegressionPage() {
  const paramsApi = useExplainerParams<PolyParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const dataset = useMemo(() => getCurveDataset(params.dataset), [params.dataset]);
  const generated = useMemo(
    () => dataset.generate({ count: params.sampleCount, noise: params.noise, seed: params.seed }),
    [dataset, params.sampleCount, params.noise, params.seed],
  );
  const test = useMemo(
    () => dataset.generate({ count: params.testCount, noise: params.noise, seed: testSeed(params.seed) }).points,
    [dataset, params.testCount, params.noise, params.seed],
  );
  const truth = generated.truth!;

  // A local copy so dragging a point edits the data without regenerating it.
  const [data, setData] = useEditable(generated);

  const config = useMemo<PolyConfig>(
    () => ({ degree: params.degree, l2: params.l2, standardise: params.standardise }),
    [params.degree, params.l2, params.standardise],
  );
  const shared = useMemo(() => ({ l2: params.l2, standardise: params.standardise }), [params.l2, params.standardise]);

  /* ---------------- the fit and the curves over degree ---------------- */

  const fit = useMemo(() => fitPolynomial(data.points, config), [data.points, config]);
  const curve = useMemo(() => validationCurve(data.points, test, MAX_DEGREE, shared), [data.points, test, shared]);
  const truthOnGrid = useMemo(() => BV_GRID.map(truth), [truth]);

  // The refit loop draws fresh noise on the generated x's; a dragged point does not change them.
  const sampler = useMemo(
    () => samplerFor(generated.points, truth, params.noise, params.seed),
    [generated.points, truth, params.noise, params.seed],
  );
  const sweep = useMemo(
    () => biasVarianceSweep(sampler, truthOnGrid, BV_GRID, MAX_DEGREE, params.refits, shared, params.noise),
    [sampler, truthOnGrid, params.refits, shared, params.noise],
  );

  /* ---------------- refits ---------------- */

  const sim = useSimulation({
    create: () => createRefitState(CURVE_GRID.length),
    step: (state) => refitStep(state, sampler, config, CURVE_GRID),
    isComplete: (state) => state.epoch >= params.refits,
    deps: [config, generated, params.refits, sampler],
    baseStepsPerSecond: 8,
    maxSteps: params.refits,
  });
  const state = sim.state;

  const bv = useMemo(() => innerBiasVariance(state, truthOnGrid), [state, truthOnGrid]);

  /* ---------------- derived numbers ---------------- */

  const predictions = useMemo(() => data.points.map((p) => predictFit(fit, p.x)), [data.points, fit]);
  const actuals = useMemo(() => data.points.map((p) => p.y), [data.points]);
  const trainError = fit.failed ? Number.NaN : meanSquaredError(fit, data.points);
  const testError = fit.failed ? Number.NaN : meanSquaredError(fit, test);
  const r2Train = useMemo(() => rSquared(actuals, predictions), [actuals, predictions]);
  const r2Test = useMemo(
    () => rSquared(test.map((p) => p.y), test.map((p) => predictFit(fit, p.x))),
    [test, fit],
  );
  const maxWeight = fit.weights.slice(1).reduce((best, w) => Math.max(best, Math.abs(w)), 0);
  const maxFeature = useMemo(() => largestFeature(fit, data.points), [fit, data.points]);

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    if (fit.failed) return { text: 'No solution: the normal equations are singular. Add points or lower the degree.', tone: 'bad' };
    if (fit.degree + 1 >= data.points.length && params.l2 === 0) {
      return { text: fit.degree + 1 + ' weights for ' + data.points.length + ' points: the curve passes through every one.', tone: 'warn' };
    }
    if (sim.isComplete && state.curves.length >= 2) {
      return {
        text: state.curves.length + ' refits: bias² ' + fmtCompact(bv.bias2) + ', variance ' + fmtCompact(bv.variance) + '.',
        tone: 'good',
      };
    }
    if (Number.isFinite(testError) && Number.isFinite(trainError) && trainError > 0 && testError > 4 * trainError && fit.degree >= 4) {
      return { text: 'Overfitting: test error is ' + Math.round(testError / trainError) + ' times the training error.', tone: 'warn' };
    }
    return { text: '' };
  }, [fit, data.points.length, sim.isComplete, state.curves.length, bv, testError, trainError, params.l2]);

  const palette = usePalette();

  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'train',
        label: 'train',
        value: fmtCompact(trainError),
        tone: fit.failed ? 'bad' : 'neutral',
        colour: palette.orange,
        help: 'Mean squared error on the ' + data.points.length + ' training points, the ones the fit was solved on.',
      },
      {
        key: 'test',
        label: 'test',
        value: fmtCompact(testError),
        tone: fit.failed ? 'bad' : Number.isFinite(testError) && testError > 4 * trainError && fit.degree >= 4 ? 'warn' : 'neutral',
        colour: palette.green,
        help: 'Mean squared error on the ' + test.length + ' held-out points. Lowest at degree ' + curve.best + ' for this data.',
      },
    ],
    [trainError, testError, fit, palette, data.points.length, test.length, curve.best],
  );

  const more: Metric[] = useMemo(
    () => [
      {
        key: 'r2',
        label: 'R² train / test',
        value: fit.failed ? 'n/a' : fmt(r2Train, 3) + ' / ' + fmt(r2Test, 3),
        help: 'Share of the variance in y the curve explains, on the training points and on the test points.',
      },
      {
        key: 'weights',
        label: 'Largest |w|',
        value: fit.failed ? 'n/a' : fmtCompact(maxWeight),
        caption: fit.degree + 1 + ' weights',
        help: 'The biggest non-bias weight. Wiggly curves are made of large weights with opposite signs; ridge charges for them.',
      },
      {
        key: 'feature',
        label: 'Largest |' + powerLabel(fit.degree, params.standardise) + '|',
        value: fmtCompact(maxFeature),
        help: 'The largest value in the highest-power column, against the constant 1. Standardising keeps this in the thousands rather than the billions.',
      },
      {
        key: 'bias',
        label: 'Bias²',
        value: fmtCompact(bv.bias2),
        caption: state.curves.length >= 2 ? state.curves.length + ' refits' : 'needs 2 refits',
        help: 'Mean squared gap between the average refit and the true function, over the inner 90% of the x range.',
      },
      {
        key: 'variance',
        label: 'Variance',
        value: fmtCompact(bv.variance),
        caption: state.curves.length >= 2 ? state.curves.length + ' refits' : 'needs 2 refits',
        help: 'How much the refits disagree with each other, averaged over the same x range. The noise floor σ² is ' + fmtCompact(sweep.noise) + '.',
      },
    ],
    [fit, r2Train, r2Test, maxWeight, maxFeature, params.standardise, bv, state.curves.length, sweep.noise],
  );

  /* ---------------- point dragging ---------------- */

  const frameRef = useRef<Frame | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [pointEdits, setPointEdits] = useState(0);
  const editsAtEntry = useRef(0);
  const movedRef = useRef(false);

  const handleFrame = useCallback((frame: Frame) => {
    frameRef.current = frame;
  }, []);

  const findPoint = useCallback(
    (pos: { x: number; y: number }) => {
      const frame = frameRef.current;
      if (!frame) return -1;
      let best = -1;
      let bestDist = 14 * 14;
      data.points.forEach((p, i) => {
        const dx = frame.x(p.x) - pos.x;
        const dy = frame.y(p.y) - pos.y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      return best;
    },
    [data.points],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      const index = findPoint(pos);
      if (index >= 0) setDragIndex(index);
      movedRef.current = false;
    },
    [findPoint],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      if (dragIndex === null) {
        const index = pos ? findPoint(pos) : -1;
        const next = index >= 0 ? index : null;
        if (next !== hoverIndex) setHoverIndex(next);
        return;
      }
      if (!pos) return;
      const frame = frameRef.current;
      if (!frame) return;
      // A dragged point stays on the chart, where the fit can be seen chasing it.
      const [y0, y1] = data.yRange;
      const y = Math.min(y1, Math.max(y0, frame.y.invert(pos.y)));
      setData((prev) => {
        const points = prev.points.slice();
        points[dragIndex] = { ...points[dragIndex], y };
        return { ...prev, points };
      });
      if (!movedRef.current) {
        movedRef.current = true;
        setPointEdits((n) => n + 1);
      }
    },
    [dragIndex, findPoint, hoverIndex, setData, data.yRange],
  );

  const handleUp = useCallback(() => setDragIndex(null), []);

  /* ---------------- render ---------------- */

  const [logScale, setLogScale] = useState(true);
  const [secondView, setSecondView] = useState<SecondView>('curve');
  const [selection, setSelection] = useState<ArchSelection | null>(null);

  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'second') setSecondView(value as SecondView);
        else if (id === 'scale') setLogScale(value === 'log');
      },
      open: (target: string | null) => setSelection(target === null ? null : parseArchTarget(target)),
    }),
    [],
  );
  const lessonContext = useMemo(
    () =>
      makePolyContext({
        params,
        state,
        data,
        test,
        fit,
        curve,
        sweep,
        bv,
        grid: BV_GRID,
        truthOnGrid,
        ui: {
          view: secondView,
          logScale,
          open: selection ? selection.kind + ':' + selection.id : null,
          get pointEdits() {
            return pointEdits - editsAtEntry.current;
          },
        },
      }),
    [params, state, data, test, fit, curve, sweep, bv, truthOnGrid, secondView, logScale, selection, pointEdits],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });
  // The baseline moves in the same render the step does, before its condition is checked.
  const lastStep = useRef(lessonApi.step);
  if (lastStep.current !== lessonApi.step) {
    lastStep.current = lessonApi.step;
    editsAtEntry.current = pointEdits;
  }

  /* ---------------- the chapter reset ---------------- */

  // Free play with no lesson marks, default knobs and views, a fresh run.
  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  const resetAll = useCallback(() => {
    resetLessons();
    reset();
    resetRun();
    setData(generated);
    setLogScale(true);
    setSecondView('curve');
    setSelection(null);
  }, [resetLessons, reset, resetRun, generated, setData]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || data !== generated || !logScale || secondView !== 'curve' || selection !== null;

  const focus = lessonApi.focus;
  const spot = useMemo<ArchSelection | null>(
    () => (focus && (focus.kind === 'node' || focus.kind === 'edge') && !selection ? { kind: focus.kind, id: focus.id } : null),
    [focus, selection],
  );

  const fed = params.standardise ? '\\tilde{x}' : 'x';
  const term = (w: number, i: number) => texNum(Math.abs(w)) + (i === 0 ? '' : i === 1 ? fed : fed + '^{' + i + '}');
  const signed = (w: number, i: number) => (i === 0 ? (w < 0 ? '-' : '') : w < 0 ? ' - ' : ' + ') + term(w, i);
  // Five terms fit the column head; the Σ card has the rest.
  const equationText = fit.failed
    ? '\\hat{y} = \\text{?}'
    : '\\hat{y} = ' + fit.weights.slice(0, 5).map(signed).join('') + (fit.weights.length > 5 ? ' + \\cdots' : '');

  const vectors = useMemo<PolyVectors>(() => {
    const rows = data.points.map((p) => featureRow(fit, p.x));
    return {
      x: data.points.map((p) => p.x),
      y: actuals,
      features: Array.from({ length: fit.degree + 1 }, (_, k) => rows.map((r) => r[k] ?? 0)),
      prediction: predictions,
      yScale: Math.max(maxAbs(actuals), maxAbs(predictions)) || 1,
    };
  }, [data.points, fit, actuals, predictions]);
  const order = useMemo(() => displayOrder(data.points.length, (i) => data.points[i].x), [data.points]);
  const standardised = params.standardise && fit.norm.std !== 1;

  const graph = useMemo(
    () => buildPolyGraph({ fit, standardised, vectors, order, trainError }),
    [fit, standardised, vectors, order, trainError],
  );

  const renderDetail = useCallback(
    (active: ArchSelection, jump: (next: ArchSelection) => void) => (
      <PolyDetail
        selection={active}
        fit={fit}
        l2={params.l2}
        standardised={standardised}
        vectors={vectors}
        order={order}
        trainError={trainError}
        testError={testError}
        jump={jump}
      />
    ),
    [fit, params.l2, standardised, vectors, order, trainError, testError],
  );

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
      <Studio
        meta={META}
        shareUrl={shareUrl}
        lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
        transport={
          <Transport sim={sim} unit="refit" total={params.refits} completeLabel={state.curves.length + ' refits'} completeTone="good" />
        }
        topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={set} lesson={lastPreset} named={lessonApi.view.knobs} />}
        tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
        controls={<ControlPanel groups={RAIL_GROUPS} params={params} onChange={set} />}
        wide
        architecture={
          <Panel id="architecture" title="The model">
            <ArchitectureView
              graph={graph}
              selection={selection}
              onSelect={setSelection}
              renderDetail={renderDetail}
              description={
                'Computation graph: the powers of x and 1 in, a weighted sum, the prediction out, then the mean squared error. Degree ' +
                fit.degree + '.'
              }
              redrawKey={fit.degree + ':' + fmt(fit.weights[0] ?? 0, 4) + ':' + standardised}
              phase={state.epoch}
              spot={spot}
            />
          </Panel>
        }
        output={
          <>
            <ColumnHead title="Output" blurb={<Tex tex={equationText} />} />

            <Panel
              id="fit"
              fill
              title="The fit"
              subtitle="Drag a point."
              actions={
                <Legend
                  dense
                  items={[
                    { label: 'train', colour: palette.blue, shape: 'dot' },
                    ...(params.showTest ? [{ label: 'test', colour: palette.green, shape: 'ring' as const }] : []),
                    { label: 'model', colour: palette.orange, shape: 'line' },
                    ...(params.showTruth ? [{ label: 'truth', colour: palette.muted, shape: 'dashed' as const }] : []),
                    ...(params.showRefits && state.curves.length > 0
                      ? [{ label: 'refit mean', colour: palette.orange, shape: 'dashed' as const }]
                      : []),
                  ]}
                />
              }
            >
              <FitPlot
                data={data}
                test={test}
                fit={fit}
                refits={state}
                grid={CURVE_GRID}
                showResiduals={params.showResiduals}
                showTruth={params.showTruth}
                showTest={params.showTest}
                showRefits={params.showRefits}
                activeIndex={dragIndex}
                hoverIndex={hoverIndex}
                dragging={dragIndex !== null}
                onFrame={handleFrame}
                onPointerDown={handleDown}
                onPointerMove={handleMove}
                onPointerUp={handleUp}
              />
            </Panel>

            <Panel
              id="second"
              fill
              title={secondView === 'curve' ? 'Error against degree' : 'Bias and variance against degree'}
              subtitle={
                secondView === 'curve'
                  ? 'Every degree refitted on the same data; the marker is the current one.'
                  : 'From ' + params.refits + ' refits per degree; the total estimates the test error.'
              }
              actions={
                <>
                  {secondView === 'curve' ? (
                    <ViewSwitch
                      id="scale"
                      label="Scale"
                      value={logScale ? 'log' : 'linear'}
                      options={[
                        { value: 'linear', label: 'Linear' },
                        { value: 'log', label: 'Log' },
                      ]}
                      onChange={(next) => setLogScale(next === 'log')}
                    />
                  ) : null}
                  <ViewSwitch
                    id="second"
                    label="Second chart"
                    value={secondView}
                    options={[
                      { value: 'curve', label: 'Errors' },
                      { value: 'bv', label: 'Bias-variance' },
                    ]}
                    onChange={(next) => setSecondView(next as SecondView)}
                  />
                </>
              }
            >
              {secondView === 'curve' ? (
                <ValidationCurveChart curve={curve} degree={fit.degree} logScale={logScale} />
              ) : (
                <BiasVarianceChart sweep={sweep} degree={fit.degree} />
              )}
            </Panel>

            <MetricStrip headline={headline} more={more} />
          </>
        }
      />
    </LessonOpenContext.Provider>
    </LessonFocusContext.Provider>
  );
}
