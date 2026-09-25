/** Activations and losses explainer: the two functions, the plane they bend, the surface they draw, and a network of chosen depth living on them. */

import { useCallback, useEffect, useMemo, useState } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { Formula } from '../../explainer/components/Detail';
import { Inspector } from '../../explainer/components/Inspector';
import type { Readout, Setup } from '../../explainer/components/Inspector';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import { Legend, Math as Tex, MetricStrip, Panel, ParamsReset, ViewSwitch } from '../../explainer/components/Panels';
import { PresetPicker } from '../../explainer/components/Presets';
import { Transport } from '../../explainer/components/Transport';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { usePresets } from '../../explainer/usePresets';
import { useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { fmt, fmtCompact, fmtKnob, fmtPercent, sub } from '../../lib/math/stats';
import { getClassificationDataset } from '../../lib/datasets/points';
import { getRegressionDataset } from '../../lib/datasets/regression';
import { ACTIVATION_NAMES, activation } from '../../lib/ml/activations';
import { predict } from '../../lib/ml/mlp';
import { createTrainer, lossAxisValue, predictedClass, scaleX, signedScore, stepTrainer } from '../../lib/ml/mlpTrainer';
import { forward } from '../../lib/ml/mlp';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { ActParams } from './config';
import { ActivationDetail, LayerDetail, LossDetail, OutputDetail, PointDetail, UnitDetail, outputRead } from './detail';
import { LESSONS, deriveNumbers } from './lessons';
import type { ActLessonContext, GraphView, SecondView, SurfaceOf } from './lessons';
import { NetworkMap } from './minimap';
import { NetworkDiagram, activationRows, netStats, outputRows } from './network';
import { MODEL_KEYS, buildModel } from './model';
import { ClassFit, CurveFit, FunctionsDiagram, LayerBars, LossCurves, layerColour } from './panels';
import { SpaceDiagram, outputReadout } from './space';
import { SurfaceDiagram, outputWeights, sumShown, unitOf } from './surface';
import { decodeTarget, encodeTarget, unitLayer } from './targets';
import type { DiagramTarget, Spot } from './targets';

const META = getExplainer('activation-functions')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_EPOCHS = 3000;
/** Network units are a sixth of the raw axes. */
const RAW = 6;

const HINTS: Record<GraphView, string> = {
  network: 'Hover a unit, a layer or the output, click to open it',
  activation: 'Hover the curve, click to open it',
  loss: 'Hover the curve, click to open it',
  space: 'Hover a stage, an arrow or a point, click to open it; units ▾ shows a layer unit by unit',
  surface: 'Hover a sheet or the sum, click to open it; click a strip to see that layer as sheets',
};

/** What each view can show open, so a card survives a switch to a view that draws it. */
const KEEPS: Record<GraphView, ReadonlyArray<DiagramTarget['kind']>> = {
  network: ['unit', 'layer', 'function', 'output'],
  activation: ['function'],
  loss: ['function'],
  space: ['layer', 'point', 'function', 'output', 'unit'],
  surface: ['unit', 'input', 'output'],
};

const TITLES: Record<GraphView, string> = {
  network: 'The network',
  activation: 'The activation',
  loss: 'The loss',
  space: 'The plane through the network',
  surface: 'The network over the input plane',
};

export default function ActivationFunctionsPage() {
  const paramsApi = useExplainerParams<ActParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, activePresetId, isDirty, shareUrl } = paramsApi;

  /* ---------------- the model ---------------- */

  const modelKey = MODEL_KEYS.map((key) => String(params[key])).join('|');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const model = useMemo(() => buildModel(params), [modelKey]);
  const { classify, pointData, curveData, classCount, lossName, lossFn, act, scale, data } = model;
  // The surface needs a plane to stand on, so a curve fit falls back to the network.
  const view: GraphView = params.view === 'space' ? 'space' : params.view === 'surface' && classify ? 'surface' : params.view === 'loss' ? 'loss' : params.view === 'activation' ? 'activation' : 'network';
  const surfaceOf: SurfaceOf = params.surfaceOf === 'units' && params.depth >= 1 ? 'units' : 'output';
  const others = useMemo(
    () => (params.showAll ? ACTIVATION_NAMES.filter((name) => name !== params.activation).map((name) => activation(name, { leakSlope: params.leak })) : []),
    [params.showAll, params.activation, params.leak],
  );

  /* ---------------- simulation ---------------- */

  const sim = useSimulation({
    create: () => createTrainer(model.config, model.data),
    step: (state) => stepTrainer(state, model.data, model.config),
    isComplete: (state) => state.diverged || (state.epoch > 10 && state.trainLoss < model.fitted),
    deps: [model],
    baseStepsPerSecond: 8,
    maxSteps: MAX_EPOCHS,
  });
  const state = sim.state;
  const presets = usePresets(paramsApi, sim);

  /* ---------------- derived ---------------- */

  const numbers = useMemo(() => deriveNumbers(state, data, act, lossName, model.fitted), [state, data, act, lossName, model.fitted]);
  const zByLayer = useMemo(() => state.layers.slice(0, -1).map((l) => l.zSamples), [state.layers]);
  // What every unit sees over the training set, for the network view.
  const stats = useMemo(() => (view === 'network' ? netStats(state, act, lossFn, data.train.inputs, data.train.targets) : null), [view, state, act, lossFn, data]);
  const outputs = useMemo(
    () =>
      state.outputs.map((o, i) => {
        const target = state.batchTargets[i] ?? [];
        return { x: lossAxisValue(lossName, o, target), target: lossFn.kind === 'multiclass' ? target.indexOf(1) : target[0] === 1 ? 1 : 0 };
      }),
    [state.outputs, state.batchTargets, lossName, lossFn.kind],
  );

  /* ---------------- ui state ---------------- */

  const [hover, setHover] = useState<DiagramTarget | null>(null);
  const [open, setOpen] = useState<DiagramTarget | null>(null);
  const [secondView, setSecondView] = useState<SecondView>('loss');
  // The hidden layer the unit surfaces show: picked in the network map, else the lesson's, else the last.
  const [pickedLayer, setPickedLayer] = useState<number | null>(null);
  // Hidden layers the plane view shows unit by unit.
  const [unitsShown, setUnitsShown] = useState<number[]>([]);
  const toggleUnits = useCallback((layer: number) => setUnitsShown((prev) => (prev.includes(layer) ? prev.filter((k) => k !== layer) : [...prev, layer])), []);
  useEffect(() => {
    setHover(null);
    setOpen((prev) => (prev && KEEPS[view].includes(prev.kind) ? prev : null));
  }, [view, surfaceOf]);
  useEffect(() => {
    setPickedLayer(null);
    setUnitsShown([]);
  }, [params.depth, params.width]);
  // The knob shows the loss the model uses: more than two classes always take softmax.
  useEffect(() => {
    if (classify && classCount > 2 && params.classLoss !== 'softmaxCE') set('classLoss', 'softmaxCE');
  }, [classify, classCount, params.classLoss, set]);

  /* ---------------- the lessons ---------------- */

  const openId = open ? encodeTarget(open) : null;
  const lessonContext = useMemo<Omit<ActLessonContext, 'sim' | 'lesson'>>(
    () => ({ params, state, derived: numbers, ui: { view, surfaceOf, secondView, open: openId } }),
    [params, state, numbers, view, surfaceOf, secondView, openId],
  );
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'graph') set('view', value);
        else if (id === 'surface') set('surfaceOf', value);
        else if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => setOpen(target === null ? null : decodeTarget(target)),
    }),
    [set],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });

  /* ---------------- the chapter reset ---------------- */

  // Free play with no lesson marks, default knobs and views, a fresh run.
  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  const resetAll = useCallback(() => {
    resetLessons();
    reset();
    resetRun();
    setSecondView('loss');
    setOpen(null);
    setPickedLayer(null);
    setUnitsShown([]);
  }, [resetLessons, reset, resetRun]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || secondView !== 'loss' || open !== null || pickedLayer !== null || unitsShown.length > 0;
  // A canvas target the step points at, outlined like a hover while nothing is open.
  const lessonFocus = lessonApi.focus;
  const spot = useMemo<Spot | null>(() => {
    if (open || !lessonFocus || lessonFocus.kind !== 'custom') return null;
    const target = decodeTarget(lessonFocus.id);
    return target ? { target, label: lessonFocus.label ?? lessonFocus.id } : null;
  }, [open, lessonFocus]);
  const spotLayer = spot && spot.target.kind === 'layer' && spot.target.index >= 1 ? spot.target.index : spot && spot.target.kind === 'unit' ? unitLayer(spot.target) : null;
  const shownLayer = Math.min(params.depth, Math.max(1, pickedLayer ?? spotLayer ?? params.depth));
  // An open or spotlit unit shows its layer unit by unit on the plane, again after a resize clears the fold.
  const wantedUnit = open && open.kind === 'unit' ? open : spot && spot.target.kind === 'unit' ? spot.target : null;
  useEffect(() => {
    if (!wantedUnit) return;
    const k = unitLayer(wantedUnit);
    setUnitsShown((prev) => (prev.includes(k) ? prev : [...prev, k]));
  }, [wantedUnit, params.depth, params.width]);

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    if (state.diverged) return { text: 'The weights blew up. Lower the rate or the init scale and replay.', tone: 'bad' };
    if (state.epoch === 0) return { text: '' };
    if (sim.isComplete) return { text: 'Fitted: training loss below ' + model.fitted + ' after ' + state.epoch + ' epochs.', tone: 'good' };
    if (params.activation === 'linear' && params.depth >= 1 && classify) {
      return { text: 'No activation: the ' + params.depth + ' layer' + (params.depth === 1 ? '' : 's') + ' multiply out to one matrix, so this is logistic regression with a straight boundary.', tone: 'warn' };
    }
    if (numbers.gradRatio !== null && numbers.gradRatio < 1e-3 && params.depth >= 3) {
      return { text: 'Vanishing: the first layer’s gradient is ' + fmtCompact(numbers.gradRatio) + ' of the last layer’s. The early layers are barely learning.', tone: 'warn' };
    }
    if (numbers.deadShare > 0.3 && params.activation === 'relu') {
      return { text: fmtPercent(numbers.deadShare, 0) + ' of hidden units are off for every sample: no gradient reaches them, so they stay dead.', tone: 'warn' };
    }
    if (numbers.deadShare > 0.3 && (params.activation === 'sigmoid' || params.activation === 'tanh')) {
      return { text: fmtPercent(numbers.deadShare, 0) + ' of hidden units are saturated on every sample: their slope is near zero and they barely learn.', tone: 'warn' };
    }
    if (lossName === 'hinge' && outputs.length > 0 && numbers.pastMargin >= 1) {
      return { text: 'Every output clears the margin: the hinge is zero and nothing trains until a point slips back.', tone: 'warn' };
    }
    if (sim.iteration >= MAX_EPOCHS) return { text: 'Stopped at ' + MAX_EPOCHS.toLocaleString('en-US') + ' epochs. Reset to run again.', tone: 'warn' };
    return { text: '' };
  }, [state, sim.isComplete, sim.iteration, model.fitted, numbers, params.depth, params.activation, classify, lossName, outputs.length]);

  /* ---------------- metrics ---------------- */

  const palette = usePalette();
  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'loss',
        label: 'train loss',
        value: state.diverged ? '∞' : fmtCompact(state.trainLoss),
        colour: palette.blue,
        tone: state.diverged ? 'bad' : 'neutral',
        help: 'Mean loss over the training set after the last epoch.',
      },
      classify
        ? {
            key: 'accuracy',
            label: 'held out',
            value: state.testAccuracy === null ? 'n/a' : fmtPercent(state.testAccuracy, 0),
            tone: (state.testAccuracy ?? 0) > 0.9 ? 'good' : 'neutral',
            help: 'Share of held-out points the network classifies correctly.',
          }
        : {
            key: 'r2',
            label: 'held-out R²',
            value: numbers.r2 === null ? 'n/a' : fmt(numbers.r2, 3),
            tone: (numbers.r2 ?? 0) > 0.9 ? 'good' : 'neutral',
            help: 'How much of the held-out variance the curve explains.',
          },
    ],
    [state, classify, numbers.r2, palette.blue],
  );

  const more: Metric[] = useMemo(
    () => [
      {
        key: 'grad-ratio',
        label: 'First / last gradient',
        value: numbers.gradRatio === null ? 'n/a' : fmtCompact(numbers.gradRatio),
        tone: numbers.gradRatio !== null && numbers.gradRatio < 1e-3 ? 'warn' : 'neutral',
        help: 'Gradient norm of the first hidden layer over the output layer’s. Far below one means the gradient vanishes on its way back.',
      },
      {
        key: 'flat',
        label: 'Values in flat zones',
        value: state.epoch === 0 ? 'n/a' : fmtPercent(numbers.flatShare, 0),
        help: 'Share of hidden pre-activations in the last batch that sit where |f′| is below 0.05.',
      },
      {
        key: 'dead',
        label: 'Dead or saturated',
        value: state.epoch === 0 ? 'n/a' : fmtPercent(numbers.deadShare, 0),
        tone: numbers.deadShare > 0.3 ? 'warn' : 'neutral',
        help: 'Share of hidden units whose slope was below 0.001 on every sample of the last batch: nothing reaches them.',
      },
      { key: 'test-loss', label: 'Held-out loss', value: state.diverged ? '∞' : fmtCompact(state.testLoss), help: 'Mean loss on the held-out quarter.' },
    ],
    [numbers, state.epoch, state.diverged, state.testLoss],
  );

  /* ---------------- the inspector ---------------- */

  const renderDetail = useCallback(
    (target: DiagramTarget) => {
      switch (target.kind) {
        case 'function':
          return target.which === 'activation' ? (
            <ActivationDetail act={act} state={state} zByLayer={zByLayer} />
          ) : (
            <LossDetail lossFn={lossFn} state={state} axis={outputs.map((o) => o.x)} />
          );
        case 'layer':
          return <LayerDetail index={target.index} state={state} act={act} />;
        case 'point': {
          const input = data.train.inputs[target.index];
          const label = data.train.labels ? data.train.labels[target.index] : (data.train.targets[target.index]?.[0] ?? null);
          return input ? <PointDetail state={state} input={input} label={label} task={classify ? 'classify' : 'regress'} classCount={classCount} /> : null;
        }
        case 'input':
          return <PointDetail state={state} input={[target.x, target.y]} label={null} task="classify" classCount={classCount} eyebrow="Input" />;
        case 'unit':
          return <UnitDetail index={target.index} layer={unitLayer(target)} state={state} act={act} points={data.train.inputs} labels={data.train.labels} />;
        case 'output':
          return <OutputDetail state={state} lossName={lossName} classify={classify} />;
        default:
          return null;
      }
    },
    [act, lossFn, state, zByLayer, outputs, data, classify, classCount, lossName],
  );

  // The symbol and its number for whatever the pointer is on.
  const readout = useMemo<Readout | null>(() => {
    if (!hover) return null;
    switch (hover.kind) {
      case 'function': {
        if (hover.which === 'activation') {
          if (hover.x === undefined) return { title: act.label, rows: stats ? activationRows(stats) : [] };
          return { title: act.label, rows: [{ label: 'z', value: fmt(hover.x, 2) }, { label: 'f(z)', value: fmt(act.f(hover.x), 3) }, { label: 'f′(z)', value: fmt(act.df(hover.x), 3) }] };
        }
        const curve = lossFn.curve;
        if (hover.x === undefined) return { title: lossFn.label, rows: stats ? outputRows(stats) : [] };
        const rows = [{ label: curve.xLabel.split(' ').pop() ?? 'x', value: fmt(hover.x, 2) }];
        for (const t of curve.targets) {
          const suffix = curve.targets.length > 1 ? ' (y = ' + t + ')' : '';
          rows.push({ label: 'L' + suffix, value: fmt(curve.f(hover.x, t), 3) }, { label: 'dL' + suffix, value: fmt(curve.df(hover.x, t), 3) });
        }
        return { title: lossFn.label, rows };
      }
      case 'layer': {
        if (hover.index === 0) return { title: 'input', rows: [{ label: 'x', value: data.train.inputs[0]?.length === 1 ? 'x' : '(x₁, x₂)' }, { label: 'points', value: String(data.train.inputs.length) }] };
        const layer = state.net.layers[hover.index - 1];
        const diag = state.layers[hover.index - 1];
        const previous = hover.index === 1 ? 'x' : 'h' + sub(hover.index - 1);
        const axes = view !== 'space' ? [] : unitsShown.includes(hover.index) ? [{ label: 'axes', value: 'x₁, x₂ per unit' }] : layer.outSize > 2 ? [{ label: 'axes', value: 'pc 1, pc 2 of ' + layer.outSize + ' units' }] : [];
        return {
          title: 'layer ' + hover.index + ': h' + sub(hover.index) + ' = ' + act.label.toLowerCase() + '(W' + sub(hover.index) + previous + ' + b' + sub(hover.index) + ')',
          rows: [
            { label: 'W', value: layer.outSize + ' × ' + layer.inSize },
            { label: '‖W‖', value: fmt(diag.weightNorm, 2) },
            { label: '‖∇W‖', value: diag.gradNorm < 0.001 && diag.gradNorm > 0 ? diag.gradNorm.toExponential(1) : fmt(diag.gradNorm, 3) },
            ...axes,
          ],
        };
      }
      case 'units': {
        const n = state.net.layers[hover.layer - 1]?.outSize ?? 0;
        return unitsShown.includes(hover.layer)
          ? { title: 'layer ' + hover.layer + ': back to all ' + n + ' units in one pane', rows: [] }
          : { title: 'layer ' + hover.layer + ': its ' + n + ' units one by one, each over the input plane', rows: [] };
      }
      case 'point': {
        const input = data.train.inputs[hover.index];
        if (!input) return null;
        const out = predict(state.net, input, state.spec);
        const y = data.train.labels ? String(data.train.labels[hover.index]) : fmt(data.train.targets[hover.index]?.[0] ?? 0, 2);
        const rows = input.length === 1 ? [{ label: 'x', value: fmt(input[0], 2) }] : [{ label: 'x₁', value: fmt(input[0] * RAW, 1) }, { label: 'x₂', value: fmt(input[1] * RAW, 1) }];
        return { title: 'point ' + (hover.index + 1), rows: [...rows, { label: 'y', value: y }, outputReadout(state, out, classify ? 'classify' : 'regress')] };
      }
      case 'input': {
        const s = signedScore(state, [hover.x, hover.y]);
        const z = forward(state.net, [hover.x, hover.y], state.spec).slice(-1)[0].z[0];
        return {
          title: 'input',
          rows: [
            { label: 'x₁', value: fmt(hover.x * RAW, 1) },
            { label: 'x₂', value: fmt(hover.y * RAW, 1) },
            ...(classCount <= 2 ? [{ label: 'z', value: fmt(z, 2) }] : []),
            // Three classes have no single ŷ; the call is the class with the largest p.
            classCount > 2
              ? { label: 'class', value: String(predictedClass(state, [hover.x, hover.y])) }
              : lossName === 'hinge'
                ? { label: 'score', value: fmt(s, 2) }
                : { label: 'ŷ', value: fmt((s + 1) / 2, 2) },
          ],
        };
      }
      case 'output': {
        const { v, b } = outputWeights(state);
        const rows = v.length <= 4 ? v.map((value, j) => ({ label: 'v' + sub(j + 1), value: fmt(value, 2) })) : [{ label: '‖v‖', value: fmt(Math.hypot(...v), 2) }];
        return { title: 'output: z = v·a + b, ŷ = ' + outputRead(classify, lossName).text, rows: [...rows, { label: 'b', value: fmt(b, 2) }] };
      }
      case 'unit': {
        const k = unitLayer(hover);
        const { w, b } = unitOf(state, hover.index, k);
        const rows = w.length <= 2 ? w.map((v, i) => ({ label: 'w' + sub(i + 1), value: fmt(v, 2) })) : [];
        const stat = stats?.units[k - 1]?.[hover.index];
        const live = stat && state.epoch > 0 ? [{ label: 'on', value: fmtPercent(stat.on, 0) }, { label: 'mean |f′|', value: fmt(stat.slope, 2) }] : [];
        return {
          title: (k > 1 ? 'layer ' + k + ' ' : '') + 'unit ' + (hover.index + 1) + ': a = ' + act.label.toLowerCase() + '(w·' + (k === 1 ? 'x' : 'h' + sub(k - 1)) + ' + b)',
          rows: [...rows, { label: 'b', value: fmt(b, 2) }, { label: '‖w‖', value: fmt(Math.hypot(...w), 2) }, ...live],
        };
      }
      default:
        return null;
    }
  }, [hover, act, lossFn, state, data, classify, classCount, lossName, stats, view, unitsShown]);

  const reference = (
    <>
      <Formula tex={act.tex} />
      <Formula tex={act.dtex} />
      <Formula tex={lossFn.tex} />
      <p className="mlx-inspector__hint">{HINTS[view]}.</p>
    </>
  );

  // The settings behind the picture, in symbols, for the fold at the top of the inspector.
  const setup = useMemo<Setup>(() => {
    const dataset = classify ? getClassificationDataset(params.pointsDataset).name : getRegressionDataset(params.curveDataset).name;
    return {
      summary: act.label.toLowerCase() + ' · ' + lossFn.label.toLowerCase() + ' · ' + state.spec.sizes.join('→') + ' · η ' + fmtKnob(params.learningRate),
      rows: [
        { label: 'f', value: act.label + (params.activation === 'leakyRelu' ? ' (α ' + params.leak + ')' : '') },
        { label: 'L', value: lossFn.label + (lossName === 'huber' ? ' (δ ' + params.huberDelta + ')' : '') },
        { label: 'net', value: state.spec.sizes.join(' → ') },
        { label: 'η', value: fmtKnob(params.learningRate) + ' (Adam)' },
        { label: 'init', value: '×' + params.initScale.toFixed(1) },
        { label: 'batch', value: String(params.batchSize) },
        { label: 'data', value: dataset + ', ' + params.count + ' pts, noise ' + params.noise.toFixed(2) },
        { label: 'seed', value: String(params.seed) },
      ],
    };
  }, [classify, params, act.label, lossFn.label, lossName, state.spec.sizes]);

  /* ---------------- render ---------------- */

  const score = useCallback((x: number, y: number) => signedScore(state, [x / RAW, y / RAW]), [state]);
  const classAt = useCallback((x: number, y: number) => predictedClass(state, [x / RAW, y / RAW]), [state]);
  const scoreAtUnit = useCallback((u: number, v: number) => signedScore(state, [u, v]), [state]);
  const classAtUnit = useCallback((u: number, v: number) => predictedClass(state, [u, v]), [state]);
  const trainLabels = useMemo(() => data.train.labels ?? data.train.inputs.map(() => 0), [data]);
  const predictCurve = useCallback(
    (x: number) => predict(state.net, [scaleX(x, curveData.xRange)], state.spec)[0] * scale.std + scale.mean,
    [state, curveData.xRange, scale],
  );

  const architectureTex =
    state.spec.sizes.join(' \\to ') + ',\\ \\text{' + act.label.toLowerCase() + '},\\ \\text{' + lossFn.label.toLowerCase() + '}';

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
    <Studio
      meta={META}
      shareUrl={shareUrl}
      lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
      transport={<Transport sim={sim} unit="epoch" completeLabel={state.diverged ? 'Diverged' : 'Fitted'} completeTone={state.diverged ? 'bad' : 'good'} />}
      topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={set} lesson={lastPreset} named={lessonApi.view.knobs} />}
      tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
      controls={
        <ControlPanel
          groups={RAIL_GROUPS}
          params={params}
          onChange={set}
          header={<PresetPicker compact presets={PRESETS} activeId={activePresetId} lastPreset={lastPreset} onPick={presets.pick} status={guide.text} tone={guide.tone} />}
        />
      }
      wide="rail"
      architecture={
        <Panel
          id="architecture"
          title={view === 'surface' && surfaceOf === 'units' ? 'The network over the input plane, layer ' + shownLayer + ' as sheets' : view === 'surface' ? 'The output over the input plane' : TITLES[view]}
          actions={
            <>
              {view === 'network' ? (
                <Legend
                  dense
                  items={[
                    { label: 'w > 0', colour: palette.positive, shape: 'line' },
                    { label: 'w < 0', colour: palette.negative, shape: 'line' },
                    { label: 'off', colour: palette.red, shape: 'square' },
                  ]}
                />
              ) : view === 'activation' ? (
                <Legend
                  dense
                  items={[
                    { label: 'f', colour: palette.blue, shape: 'line' },
                    ...(params.showDerivative ? [{ label: 'f′', colour: palette.orange, shape: 'dashed' as const }] : []),
                    { label: 'flat zone', colour: palette.red, shape: 'square' },
                    ...(params.showRug ? [{ label: 'units by layer', colour: layerColour(palette, 0), shape: 'line' as const }] : []),
                  ]}
                />
              ) : view === 'loss' ? (
                <Legend
                  dense
                  items={[
                    { label: 'L', colour: palette.blue, shape: 'line' },
                    ...(params.showDerivative ? [{ label: 'dL', colour: palette.orange, shape: 'dashed' as const }] : []),
                    ...(params.showRug ? [{ label: 'outputs', colour: palette.classA, shape: 'line' as const }] : []),
                  ]}
                />
              ) : view === 'space' ? (
                <Legend
                  dense
                  items={[
                    { label: 'along x₁', colour: palette.violet, shape: 'line' },
                    { label: 'along x₂', colour: palette.cyan, shape: 'line' },
                    { label: 'ŷ = ½', colour: palette.text, shape: 'line' },
                    ...(unitsShown.length > 0
                      ? [
                          { label: '< 0', colour: palette.negative, shape: 'square' as const },
                          { label: '> 0', colour: palette.positive, shape: 'square' as const },
                        ]
                      : []),
                  ]}
                />
              ) : view === 'surface' && params.depth >= 1 ? (
                <>
                  <Legend
                    dense
                    items={[
                      { label: '< 0', colour: palette.negative, shape: 'square' },
                      { label: '> 0', colour: palette.positive, shape: 'square' },
                    ]}
                  />
                  <ViewSwitch
                    id="surface"
                    label="Surface of"
                    value={surfaceOf}
                    options={[
                      { value: 'output', label: 'Output' },
                      { value: 'units', label: 'Units' },
                    ]}
                    onChange={(next) => set('surfaceOf', next)}
                  />
                </>
              ) : null}
              <ViewSwitch
                id="graph"
                label="View"
                value={view}
                options={[
                  { value: 'network', label: 'Network' },
                  { value: 'activation', label: 'Activation' },
                  { value: 'loss', label: 'Loss' },
                  { value: 'space', label: 'Plane' },
                  ...(classify ? [{ value: 'surface', label: 'Sheets' }] : []),
                ]}
                onChange={(next) => set('view', next)}
              />
            </>
          }
        >
          <div className="mlx-arch">
            {view === 'network' && stats ? (
              <NetworkDiagram
                state={state}
                act={act}
                lossLabel={lossFn.name === 'softmaxCE' ? 'softmax' : lossFn.name}
                stats={stats}
                classify={classify}
                hover={hover}
                open={open}
                spot={spot}
                onHover={setHover}
                onOpen={setOpen}
                description={
                  'The network ' + state.spec.sizes.join(' to ') + ' after ' + state.epoch + ' epochs: every hidden unit as its ' + act.label.toLowerCase() + ' curve with the values it sees, the output as the ' + lossFn.label.toLowerCase() + ' with the training samples on it, and the gradient each layer received.'
                }
                redrawKey={state.epoch + ':' + act.name + ':' + lossFn.name}
              />
            ) : view === 'activation' || view === 'loss' ? (
              <FunctionsDiagram
                which={view}
                act={act}
                lossFn={lossFn}
                state={state}
                zByLayer={zByLayer}
                outputs={outputs}
                showDerivative={params.showDerivative}
                others={others}
                showRug={params.showRug}
                hover={hover && hover.kind === 'function' ? hover.which : null}
                open={open && open.kind === 'function' ? open.which : null}
                spot={spot}
                onHover={(which, x) => setHover(which ? { kind: 'function', which, x } : null)}
                onOpen={(which) => setOpen(which ? { kind: 'function', which } : null)}
                description={
                  view === 'activation'
                    ? act.label + ' and its derivative, with the network’s units marked on it after ' + state.epoch + ' epochs.'
                    : lossFn.label + ' and its slope, with the last batch’s outputs marked on it after ' + state.epoch + ' epochs.'
                }
                redrawKey={state.epoch + ':' + act.name + ':' + lossFn.name + ':' + params.showDerivative + params.showAll + params.showRug}
              />
            ) : view === 'space' ? (
              <SpaceDiagram
                state={state}
                act={act}
                task={classify ? 'classify' : 'regress'}
                inputs={data.train.inputs}
                targets={data.train.targets}
                labels={data.train.labels}
                classCount={classCount}
                inputScale={RAW}
                hover={hover}
                open={open}
                spot={spot}
                onHover={setHover}
                onOpen={setOpen}
                unitsShown={unitsShown}
                onToggleUnits={toggleUnits}
                description={
                  'The input grid and the training points carried through every layer: after each layer’s weighted sums, still a grid, then after ' + act.label.toLowerCase() + ', bent; last, every point at its output, after ' + state.epoch + ' epochs.'
                }
                redrawKey={state.epoch + ':' + act.name}
              />
            ) : (
              <SurfaceDiagram
                state={state}
                act={act}
                mode={surfaceOf}
                layer={shownLayer}
                onPick={setPickedLayer}
                inputs={data.train.inputs}
                labels={trainLabels}
                classCount={classCount}
                lossName={lossName}
                inputScale={RAW}
                scoreAt={scoreAtUnit}
                classAt={classAtUnit}
                hover={hover}
                open={open}
                spot={spot}
                onHover={setHover}
                onOpen={setOpen}
                description={
                  surfaceOf === 'units'
                    ? 'The network over the input plane after ' + state.epoch + ' epochs: the input, every hidden layer as one tile per unit, layer ' + shownLayer + ' big as sheets' + (sumShown(state, shownLayer) ? ' beside their weighted sum z, the output before the sigmoid' : ' beside the sum the next layer makes of them') + ', and the output.'
                    : 'The network’s output over the input plane as a surface, with the training points at their class height, after ' + state.epoch + ' epochs.'
                }
                redrawKey={state.epoch + ':' + act.name + ':' + lossName + ':' + shownLayer}
              />
            )}
            <Inspector
              readout={readout}
              hint={HINTS[view]}
              setup={setup}
              card={open ? renderDetail(open) : null}
              onClose={() => setOpen(null)}
              idle={reference}
              header={
                <NetworkMap
                  state={state}
                  activationLabel={act.label.toLowerCase()}
                  lossLabel={lossFn.name === 'softmaxCE' ? 'softmax' : lossFn.name}
                  shown={params.depth >= 1 && view === 'surface' && surfaceOf === 'units' ? shownLayer : null}
                  hover={hover}
                  open={open}
                  spot={spot}
                  onHover={setHover}
                  onOpen={(target) => {
                    if (target && target.kind === 'layer' && target.index >= 1) setPickedLayer(target.index);
                    if (target && target.kind === 'unit') setPickedLayer(unitLayer(target));
                    setOpen(target);
                  }}
                />
              }
            />
          </div>
        </Panel>
      }
      output={
        <>
          <ColumnHead title="Output" blurb={<Tex tex={architectureTex} />} />

          <Panel id="fit" fill title="The fit">
            {classify ? (
              <ClassFit points={pointData.points} score={score} classAt={classAt} classCount={classCount} epoch={state.epoch} />
            ) : (
              <CurveFit data={curveData} predict={predictCurve} epoch={state.epoch} />
            )}
          </Panel>

          <Panel
            id="second"
            fill
            title={secondView === 'gradients' ? 'Gradient per layer' : 'Loss over epochs'}
            actions={
              <ViewSwitch
                id="second"
                label="Second chart"
                value={secondView}
                options={[
                  { value: 'loss', label: 'Loss' },
                  { value: 'gradients', label: 'Gradients' },
                ]}
                onChange={(next) => setSecondView(next as SecondView)}
              />
            }
          >
            {secondView === 'loss' ? (
              <LossCurves history={state.lossHistory} logScale={params.logScale} />
            ) : (
              <LayerBars state={state} logScale={params.logScale} />
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
