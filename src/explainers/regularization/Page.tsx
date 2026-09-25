/** Ridge and lasso explainer: the fitted curve and the two-weight plane side by side, the model map and the inspector beside them, the weights and the error curve. Lessons drive it. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import { IconChevron } from '../../explainer/components/Icons';
import { Inspector } from '../../explainer/components/Inspector';
import type { Readout, Setup } from '../../explainer/components/Inspector';
import { Formula } from '../../explainer/components/Detail';
import { Legend, Math as Tex, MetricStrip, Panel, ParamsReset } from '../../explainer/components/Panels';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { PresetPicker } from '../../explainer/components/Presets';
import { Transport } from '../../explainer/components/Transport';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { usePresets } from '../../explainer/usePresets';
import { prefersReducedMotion, useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { generatePolynomial, isPolynomial, powersOf } from '../../lib/datasets/polynomial';
import { generateTabular, testRows, trainRows } from '../../lib/datasets/tabular';
import type { TabularData } from '../../lib/datasets/tabular';
import { fmt, fmtCompact, fmtKnob, sup } from '../../lib/math/stats';
import {
  createState,
  lambdaGrid,
  lambdaMax,
  meanSquaredErrorOf,
  olsSolution,
  penalty as penaltyOf,
  predictRaw,
  prepare,
  regularisationPath,
  smoothLoss,
  step as solverStep,
} from '../../lib/ml/elasticNet';
import type { ElasticNetConfig, Solver } from '../../lib/ml/elasticNet';
import type { Point2 } from '../../lib/ml/penaltyGeometry';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { RegParams } from './config';
import { FeatureDetail, PointDetail, StationDetail } from './detail';
import { LESSONS, makeRegContext, pairOf, truthOf } from './lessons';
import { ModelMap, penaltyName } from './map';
import { CoefficientBars, ErrorCurve, RegDiagram, geometryOf, weightName } from './panels';
import { featureOf, isModelNode, spotOf } from './targets';
import type { RegTarget, Spot } from './targets';

const META = getExplainer('regularization')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_STEPS = 5000;
const HINT = 'Hover the map, a point or a bar; click to open it';

/** The solver the params ask for: ridge gets its closed form unless a solver is named. */
function solverOf(params: RegParams, alpha: number): Solver {
  if (params.solver === 'ista') return 'ista';
  if (params.solver === 'cd') return 'cd';
  return alpha === 0 ? 'closed' : 'cd';
}

export default function RegularizationPage() {
  const paramsApi = useExplainerParams<RegParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, activePresetId, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  // Two plain columns with a true weight each, so the weight plane is exact; or the wave with its powers of x.
  const data = useMemo<TabularData>(
    () =>
      params.dataset === 'two'
        ? generateTabular({
            count: params.count,
            features: 2,
            informative: 2,
            correlation: 0,
            groupSize: 3,
            noise: params.noise,
            variedScales: false,
            seed: params.seed,
            trainFraction: 0.75,
          })
        : generatePolynomial({ count: params.count, degree: params.features, noise: params.noise, seed: params.seed, trainFraction: 0.75 }),
    [params.dataset, params.count, params.features, params.noise, params.seed],
  );
  const curve = isPolynomial(data) ? data : null;
  const train = useMemo(() => trainRows(data), [data]);
  const test = useMemo(() => testRows(data), [data]);
  const prep = useMemo(() => prepare(train, params.standardise), [train, params.standardise]);

  const alpha = params.penalty === 'ridge' ? 0 : params.penalty === 'lasso' ? 1 : params.alpha;
  const solver = solverOf(params, alpha);
  const config = useMemo<ElasticNetConfig>(
    () => ({ lambda: params.lambda, alpha, solver, learningRate: params.learningRate }),
    [params.lambda, alpha, solver, params.learningRate],
  );

  /* ---------------- simulation ---------------- */

  const sim = useSimulation({
    create: () => createState(prep, config),
    step: (state) => solverStep(state, prep, config),
    isComplete: (state) => state.converged || state.diverged,
    deps: [prep, config],
    baseStepsPerSecond: 20,
    maxSteps: MAX_STEPS,
  });
  const state = sim.state;
  const presets = usePresets(paramsApi, sim);

  // After the first render every change of the data or the penalty re-solves on its own, so a knob answers at once.
  const simRef = useRef(sim);
  simRef.current = sim;
  const initial = useRef<{ prep: typeof prep; config: ElasticNetConfig } | null>(null);
  // Set by the chapter reset for its own commit, which lands on the unsolved page like a fresh load.
  const resetting = useRef(false);
  useEffect(() => {
    if (initial.current === null || resetting.current) initial.current = { prep, config };
    if ((initial.current.prep === prep && initial.current.config === config) || prefersReducedMotion()) return undefined;
    const id = window.setTimeout(() => simRef.current.play(), 0);
    return () => window.clearTimeout(id);
  }, [prep, config]);
  useEffect(() => {
    resetting.current = false;
  });

  /* ---------------- derived ---------------- */

  const path = useMemo(
    () => regularisationPath(train, test, prep, alpha, lambdaGrid(lambdaMax(prep, alpha), 60)),
    [train, test, prep, alpha],
  );
  const ols = useMemo(() => olsSolution(prep), [prep]);
  const truth = useMemo(() => truthOf(data, prep), [data, prep]);
  const pair = useMemo(() => pairOf(prep.p, params.plane, truth), [prep.p, params.plane, truth]);
  const trainMse = useMemo(() => (state.diverged ? Infinity : meanSquaredErrorOf(state.w, prep, train)), [state, prep, train]);
  const testMse = useMemo(() => (state.diverged ? Infinity : meanSquaredErrorOf(state.w, prep, test)), [state, prep, test]);
  const l1 = state.w.reduce((s, v) => s + Math.abs(v), 0);
  const finiteW = useMemo(() => (state.diverged ? state.w.map(() => 0) : state.w), [state]);
  const error = useMemo(() => smoothLoss(finiteW, prep), [finiteW, prep]);
  const penalty = penaltyOf(finiteW, params.lambda, alpha);
  const objective = Number.isFinite(state.objective) ? state.objective : error + penalty;
  const l2 = Math.sqrt(state.w.reduce((s, v) => s + v * v, 0));
  // Ridge never reaches all zeros, so it quotes the lasso's λ max.
  const top = lambdaMax(prep, alpha > 0 ? alpha : 1);
  const unit = solver === 'ista' ? 'epoch' : solver === 'closed' ? 'step' : 'sweep';

  /* ---------------- hover, cards ---------------- */

  const [hover, setHover] = useState<RegTarget | null>(null);
  const [open, setOpenRaw] = useState<RegTarget | null>(null);
  // Only a feature, a point or a station opens a card.
  const setOpen = useCallback((target: RegTarget | null) => {
    setOpenRaw(target && (target.kind === 'feature' || target.kind === 'point' || target.kind === 'node') ? target : null);
  }, []);
  const hoverFeature = useCallback((index: number | null) => setHover(index === null ? null : { kind: 'feature', index }), []);
  const openFeature = useCallback((index: number) => setOpenRaw((prev) => (prev && prev.kind === 'feature' && prev.index === index ? null : { kind: 'feature', index })), []);
  useEffect(() => setOpenRaw((prev) => (prev && prev.kind === 'point' ? null : prev)), [data]);
  useEffect(() => setOpenRaw((prev) => (prev && prev.kind === 'feature' && prev.index >= prep.p ? null : prev)), [prep.p]);
  // The plane's least-squares point while the reader drags it; the real one otherwise.
  const [centre, setCentre] = useState<Point2 | null>(null);
  useEffect(() => setCentre(null), [prep, config]);

  /* ---------------- the lessons ---------------- */

  const lessonActions = useMemo(
    () => ({
      open: (target: string | null) => {
        if (target === null) {
          setOpenRaw(null);
          return;
        }
        const index = Number(target);
        if (Number.isInteger(index)) setOpenRaw({ kind: 'feature', index });
        else if (isModelNode(target)) setOpenRaw({ kind: 'node', id: target });
      },
    }),
    [],
  );
  const lessonContext = useMemo(
    () => makeRegContext({ params, state, data, train, test, prep, path, ols, alpha, pair, ui: { open: featureOf(open) } }),
    [params, state, data, train, test, prep, path, ols, alpha, pair, open],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });
  // Reset takes the whole chapter back: defaults, free play, no card, nothing dragged.
  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  const resetAll = useCallback(() => {
    resetting.current = true;
    resetLessons();
    reset();
    resetRun();
    setOpenRaw(null);
    setHover(null);
    setCentre(null);
  }, [resetLessons, reset, resetRun]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || open !== null || centre !== null;
  // A step pointing at a station of the map, a feature cell or a pane, decoded from a custom target.
  const spot = useMemo<Spot | null>(() => {
    const focus = lessonApi.focus;
    if (!focus || focus.kind !== 'custom') return null;
    return spotOf(focus.id, focus.label ?? focus.id);
  }, [lessonApi.focus]);

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    if (state.diverged) {
      return {
        text: 'Diverged: η = ' + fmtKnob(params.learningRate) + ' is above 2 over the largest curvature (' + fmtKnob(2 / (prep.lipschitz + params.lambda * (1 - alpha))) + '). Lower it.',
        tone: 'bad',
      };
    }
    if (params.lambda < 0.02 && prep.p > prep.n) {
      return { text: 'More features than training rows and almost no penalty: the fit is not unique, and the test error shows it.', tone: 'warn' };
    }
    if (state.converged) {
      if (state.activeCount === 0) {
        return { text: 'λ = ' + fmtKnob(params.lambda) + ' is above λ max = ' + fmtKnob(top) + ': every weight is zero, the model predicts the mean.', tone: 'warn' };
      }
      return {
        text: 'Converged after ' + state.epoch + ' ' + unit + (state.epoch === 1 ? '' : 's') + ': ' + state.activeCount + ' of ' + prep.p + ' weights non-zero.',
        tone: 'good',
      };
    }
    if (sim.iteration >= MAX_STEPS) {
      return { text: 'Stopped at ' + MAX_STEPS.toLocaleString('en-US') + ' ' + unit + 's. Reset to run again.', tone: 'warn' };
    }
    return { text: '' };
  }, [state, params.learningRate, params.lambda, prep, top, unit, sim.iteration, alpha]);

  /* ---------------- metrics ---------------- */

  const palette = usePalette();

  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'test',
        label: 'test MSE',
        value: state.epoch === 0 || !Number.isFinite(testMse) ? 'n/a' : fmtCompact(testMse),
        colour: palette.orange,
        tone: state.diverged ? 'bad' : 'neutral',
        help: 'Mean squared error on the held-out quarter of the rows. The number regularisation is meant to improve.',
      },
      {
        key: 'nonzero',
        label: 'non-zero',
        value: state.activeCount + ' / ' + prep.p,
        tone: state.activeCount === 0 && state.epoch > 0 ? 'warn' : 'neutral',
        help: 'How many weights are not exactly zero. Only the lasso and the elastic net can zero one.',
      },
    ],
    [state, testMse, prep.p, palette.orange],
  );

  const more: Metric[] = useMemo(
    () => [
      {
        key: 'train',
        label: 'Train MSE',
        value: state.epoch === 0 || !Number.isFinite(trainMse) ? 'n/a' : fmtCompact(trainMse),
        help: 'Mean squared error on the training rows. The penalty trades a little of this for less test error.',
      },
      { key: 'l1', label: '‖w‖₁', value: fmtCompact(l1), help: 'Sum of the absolute weights, what the lasso charges for.' },
      { key: 'l2', label: '‖w‖₂', value: fmtCompact(l2), help: 'Length of the weight vector, what ridge charges for.' },
      { key: 'best-lambda', label: 'Best λ', value: fmtKnob(path.lambdas[path.bestIndex] ?? 0), help: 'The λ on the error curve with the lowest test error.' },
      { key: 'lambda-max', label: 'λ max', value: fmtKnob(top), help: 'The smallest λ at which the lasso keeps nothing.' },
    ],
    [state.epoch, trainMse, l1, l2, path, top],
  );

  /* ---------------- the inspector ---------------- */

  const name = useCallback((j: number) => data.featureNames[j] ?? String(j + 1), [data.featureNames]);
  const geometry = useMemo(() => geometryOf(prep, state.w, pair, alpha, centre), [prep, state.w, pair, alpha, centre]);

  // The symbol and its number for whatever the pointer is on.
  const readout = useMemo<Readout | null>(() => {
    if (!hover) return null;
    const rows = (pairs: Array<[string, string]>) => pairs.map(([label, value]) => ({ label, value }));
    switch (hover.kind) {
      case 'feature': {
        const j = hover.index;
        const w = Number.isFinite(state.w[j]) ? state.w[j] : 0;
        return {
          title: name(j) + (w === 0 ? ', out of the model' : ''),
          rows: rows([
            [weightName(name(j)), w === 0 ? '0' : fmt(w, 3)],
            ['least squares', fmt(ols[j] ?? 0, 3)],
            ...(truth ? [['true', fmt(truth[j], 3)] as [string, string]] : []),
            ['share of ' + penaltyName(alpha), fmtCompact(alpha * Math.abs(w) + ((1 - alpha) / 2) * w * w)],
          ]),
        };
      }
      case 'point': {
        const k = hover.index;
        const yHat = state.diverged ? NaN : predictRaw(state.w, prep, data.X[k]);
        const held = data.testIndex.includes(k);
        const inputs: Array<[string, string]> = curve ? [['x', fmt(curve.x[k], 2)]] : data.X[k].map((v, j) => [name(j), fmt(v, 2)]);
        return {
          title: (held ? 'held-out point ' : 'fitted point ') + (k + 1),
          rows: rows([...inputs, ['y', fmt(data.y[k], 3)], ['ŷ', fmt(yHat, 3)], ['(y − ŷ)²', fmt((data.y[k] - yHat) ** 2, 4)]]),
        };
      }
      case 'curve': {
        if (!curve) return null;
        const yHat = state.diverged ? NaN : predictRaw(state.w, prep, powersOf(hover.x, prep.p));
        return { title: 'the curve, ŷ at x', rows: rows([['x', fmt(hover.x, 2)], ['ŷ', fmt(yHat, 3)], ['wave', fmt(curve.truth(hover.x), 3)]]) };
      }
      case 'mark': {
        if (!geometry) return null;
        const [i, j] = pair;
        const point = hover.which === 'now' ? geometry.active.point : hover.which === 'other' ? geometry.other?.point : geometry.quad.centre;
        if (!point) return null;
        const shape = (a: number) => (a === 1 ? 'lasso' : a === 0 ? 'ridge' : 'elastic net');
        const title =
          hover.which === 'now'
            ? 'the answer: ' + shape(alpha) + ', λ ' + fmtKnob(params.lambda)
            : hover.which === 'other'
              ? shape(geometry.other?.alpha ?? 0) + ' with the same budget'
              : geometry.dragged
                ? 'least squares, dragged'
                : 'least squares, no penalty';
        return {
          title,
          rows: rows([
            [weightName(name(i)), fmt(point[0], 3)],
            [weightName(name(j)), fmt(point[1], 3)],
            ...(hover.which === 'now' ? [['budget ' + penaltyName(alpha), fmt(geometry.active.level, 3)] as [string, string]] : []),
          ]),
        };
      }
      case 'node':
        switch (hover.id) {
          case 'data':
            return { title: 'data: ' + (train.X.length + test.X.length) + ' rows', rows: rows([['fitted', String(train.X.length)], ['held out', String(test.X.length)], ['ȳ', fmt(prep.yMean, 3)]]) };
          case 'features':
            return {
              title: 'features: ' + prep.p + (curve ? ' powers of x' : ' columns'),
              rows: rows([
                ['non-zero', state.activeCount + ' / ' + prep.p],
                ['max |w|', fmt(Math.max(0, ...finiteW.map(Math.abs)), 3)],
                ['scaled', params.standardise ? 'yes' : 'no'],
              ]),
            };
          case 'sum':
            return { title: 'sum: ŷ = ȳ + Σ wⱼ x̃ⱼ', rows: rows([['b = ȳ', fmt(prep.yMean, 3)], ['‖w‖₁', fmt(l1, 3)], ['‖w‖₂', fmt(l2, 3)]]) };
          case 'prediction':
            return { title: curve ? 'prediction: the curve' : 'prediction: one per row', rows: rows([['MSE fitted', fmtCompact(trainMse)], ['MSE held out', fmtCompact(testMse)]]) };
          case 'error':
            return { title: 'error: ½ MSE on the fitted rows', rows: rows([['error', fmtCompact(error)], ['MSE fitted', fmtCompact(trainMse)], ['MSE held out', fmtCompact(testMse)]]) };
          case 'penalty':
            return {
              title: 'penalty: λ · ' + penaltyName(alpha),
              rows: rows([['λ', fmtKnob(params.lambda)], [penaltyName(alpha), fmtCompact(penalty / Math.max(params.lambda, 1e-12))], ['λ · R(w)', fmtCompact(penalty)], ['λ max', fmtKnob(top)]]),
            };
          default:
            return { title: 'objective: error + penalty', rows: rows([['error', fmtCompact(error)], ['penalty', fmtCompact(penalty)], ['objective', fmtCompact(objective)]]) };
        }
    }
  }, [hover, state, name, ols, truth, alpha, curve, data, prep, geometry, pair, params.lambda, params.standardise, train, test, finiteW, l1, l2, trainMse, testMse, error, penalty, objective, top]);

  // The settings behind the picture, in symbols, for the fold under the readout.
  const setup = useMemo<Setup>(
    () => ({
      summary: (params.penalty === 'ridge' ? 'ridge' : params.penalty === 'lasso' ? 'lasso' : 'elastic α ' + params.alpha.toFixed(1)) + ' · λ ' + fmtKnob(params.lambda) + ' · ' + prep.p + (curve ? ' powers' : ' cols') + ' · n ' + prep.n,
      rows: [
        { label: 'data', value: curve ? 'curve, ' + params.count + ' pts, noise ' + params.noise.toFixed(2) : 'two columns, ' + params.count + ' rows, noise ' + params.noise.toFixed(2) },
        { label: 'features', value: curve ? 'x … x' + (prep.p > 1 ? sup(prep.p) : '') + (params.standardise ? ', standardised' : ', raw') : '2 columns' + (params.standardise ? ', standardised' : ', raw') },
        { label: 'penalty', value: params.penalty === 'ridge' ? 'ridge, ½Σw²' : params.penalty === 'lasso' ? 'lasso, Σ|w|' : 'elastic net, α ' + params.alpha.toFixed(1) },
        { label: 'λ', value: fmtKnob(params.lambda) + ' (max ' + fmtKnob(top) + ')' },
        { label: 'solver', value: solver === 'closed' ? 'closed form' : solver === 'cd' ? 'coordinate descent' : 'proximal gradient, η ' + fmtKnob(params.learningRate) },
        { label: 'seed', value: String(params.seed) },
      ],
    }),
    [params, prep.p, prep.n, curve, top, solver],
  );

  const renderDetail = useCallback(
    (target: RegTarget): ReactNode => {
      if (target.kind === 'feature') {
        return (
          <FeatureDetail
            index={target.index}
            name={name(target.index)}
            weights={state.w}
            ols={ols}
            truth={params.showTruth ? truth : null}
            prep={prep}
            path={path}
            lambda={params.lambda}
            alpha={alpha}
          />
        );
      }
      if (target.kind === 'point') {
        const k = target.index;
        return (
          <PointDetail
            index={k}
            inputs={curve ? [{ name: 'x', value: curve.x[k] }] : data.X[k].map((value, j) => ({ name: name(j), value }))}
            y={data.y[k]}
            prediction={state.diverged ? NaN : predictRaw(state.w, prep, data.X[k])}
            train={data.trainIndex.includes(k)}
            trainCount={data.trainIndex.length}
            testCount={data.testIndex.length}
          />
        );
      }
      if (target.kind === 'node') {
        return (
          <StationDetail
            id={target.id}
            curve={curve !== null}
            featureNames={data.featureNames}
            weights={state.w}
            prep={prep}
            trainCount={train.X.length}
            testCount={test.X.length}
            lambda={params.lambda}
            alpha={alpha}
            error={error}
            penalty={penalty}
            objective={objective}
            trainMse={trainMse}
            testMse={testMse}
            lambdaMax={top}
            epoch={state.epoch}
            unit={unit}
          />
        );
      }
      return null;
    },
    [name, state, ols, params.showTruth, params.lambda, truth, prep, path, alpha, curve, data, train, test, error, penalty, objective, trainMse, testMse, top, unit],
  );

  const objectiveTex =
    '\\tfrac{1}{2n}\\lVert y - Xw \\rVert^2 + \\lambda\\Big(\\alpha \\lVert w \\rVert_1 + \\tfrac{1 - \\alpha}{2} \\lVert w \\rVert_2^2\\Big)';
  const reference = (
    <>
      <Formula tex={objectiveTex} />
      <p>
        {fmtCompact(error)} + {fmtKnob(params.lambda)} × {fmtCompact(penalty / Math.max(params.lambda, 1e-12))} = {fmtCompact(objective)}
      </p>
    </>
  );

  const inspector = (
    <Inspector
      readout={readout}
      hint={HINT}
      setup={setup}
      card={open ? renderDetail(open) : null}
      onClose={() => setOpenRaw(null)}
      idle={reference}
      header={
        <ModelMap
          featureNames={data.featureNames}
          weights={state.w}
          curve={curve !== null}
          trainCount={train.X.length}
          testCount={test.X.length}
          lambda={params.lambda}
          alpha={alpha}
          error={error}
          penalty={penalty}
          objective={objective}
          pair={pair}
          hover={hover}
          open={open}
          spot={spot}
          onHover={setHover}
          onOpen={setOpen}
        />
      }
    />
  );

  /* ---------------- render ---------------- */

  const planeOptions = useMemo(
    () => (curve && prep.p >= 3 ? Array.from({ length: prep.p - 1 }, (_, k) => ({ value: String(k), label: weightName(name(k)) + ', ' + weightName(name(k + 1)) })) : null),
    [curve, prep.p, name],
  );

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
      <Studio
        meta={META}
        shareUrl={shareUrl}
        lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
        transport={
          <Transport
            sim={sim}
            unit={unit}
            completeLabel={state.diverged ? 'Diverged' : 'Converged'}
            completeTone={state.diverged ? 'bad' : 'good'}
          />
        }
        topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={set} lesson={lastPreset} named={lessonApi.view.knobs} />}
        tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
        controls={
          <ControlPanel
            groups={RAIL_GROUPS}
            params={params}
            onChange={set}
            header={
              <PresetPicker
                compact
                presets={PRESETS}
                activeId={activePresetId}
                lastPreset={lastPreset}
                onPick={presets.pick}
                status={guide.text}
                tone={guide.tone}
              />
            }
          />
        }
        wide="rail"
        architecture={
          <Panel id="architecture" title="The model">
            <RegDiagram
              state={state}
              prep={prep}
              alpha={alpha}
              featureNames={data.featureNames}
              truth={params.showTruth ? truth : null}
              ols={ols}
              showOls={params.showOls}
              showTruth={params.showTruth}
              data={data}
              curve={curve}
              pair={pair}
              centre={centre}
              onDragCentre={setCentre}
              hover={hover}
              open={open}
              onHover={setHover}
              onOpen={setOpen}
              inspector={inspector}
              picker={
                planeOptions ? (
                  <label className="mlx-reg-pane__pick" title="Which two weights the plane draws; the other weights stay where the solver left them">
                    axes
                    <span className="mlx-select mlx-select--head">
                      <select aria-label="Weights on the plane" value={String(pair[0])} onChange={(event) => set('plane', Number(event.target.value))}>
                        {planeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <IconChevron size={12} />
                    </span>
                  </label>
                ) : undefined
              }
              unit={unit}
              redrawKey={state.epoch + ':' + alpha + ':' + params.showTruth + ':' + params.showOls + ':' + pair.join()}
            />
          </Panel>
        }
        output={
          <>
            <ColumnHead title="Output" blurb={<Tex tex={objectiveTex} />} />

            <Panel
              id="coefficients"
              fill
              title="Weights"
              subtitle={truth ? 'Standardised units; the tick is the true weight.' : 'Standardised units, one per power of x; the marked pair is on the plane.'}
              actions={
                <Legend
                  dense
                  items={[
                    { label: 'fit', colour: palette.textMuted, shape: 'square' },
                    ...(params.showOls ? [{ label: 'least squares', colour: palette.textMuted, shape: 'dashed' as const }] : []),
                    ...(truth && params.showTruth ? [{ label: 'true', colour: palette.text, shape: 'line' as const }] : []),
                  ]}
                />
              }
            >
              <CoefficientBars
                weights={state.w}
                ols={ols}
                truth={params.showTruth ? truth : null}
                names={data.featureNames}
                showOls={params.showOls}
                lit={featureOf(open) ?? featureOf(hover)}
                pair={pair}
                onHover={hoverFeature}
                onOpen={openFeature}
              />
            </Panel>

            <Panel
              id="second"
              fill
              title="Error against λ"
              subtitle="Every strength, fitted points against held-out ones."
              actions={
                <Legend
                  dense
                  items={[
                    { label: 'fitted', colour: palette.blue, shape: 'line' },
                    ...(params.showTest ? [{ label: 'held out', colour: palette.orange, shape: 'line' as const }, { label: 'best', colour: palette.green, shape: 'cross' as const }] : []),
                  ]}
                />
              }
            >
              <ErrorCurve path={path} lambda={params.lambda} showTest={params.showTest} />
            </Panel>

            <MetricStrip headline={headline} more={more} />
          </>
        }
      />
    </LessonOpenContext.Provider>
    </LessonFocusContext.Provider>
  );
}
