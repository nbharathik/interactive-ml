/** Neural network explainer. Every hidden unit renders its own picture inside its node. */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import {
  ColumnHead,
  Studio,
} from '../../explainer/components/Studio';
import {
  Legend,
  Math as Tex,
  MetricStrip,
  Panel,
  ParamsReset,
  ViewSwitch,
} from '../../explainer/components/Panels';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useLesson } from '../../explainer/useLesson';
import { Transport } from '../../explainer/components/Transport';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { getClassificationDataset } from '../../lib/datasets/points';
import { fmt, fmtCompact, fmtPercent } from '../../lib/math/stats';
import {
  FEATURE_LABELS,
  createState,
  evaluateSet,
  inputVector,
  layerValues,
  maxAbsWeight,
  parameterCount,
  splitDataset,
  step as trainEpoch,
} from '../../lib/ml/neuralNetwork';
import type {
  Activation,
  Initialiser,
  InputFeature,
  NetConfig,
  Regularisation,
} from '../../lib/ml/neuralNetwork';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import {
  CONTROL_GROUPS,
  DEFAULT_PARAMS,
  MAX_LAYERS,
  MAX_UNITS,
  PRESETS,
  architectureLabel,
  formatLayers,
  parseLayers,
} from './config';
import type { NeuralNetParams } from './config';
import { computeNeuronGrids, computeUnitGrid, countFlatUnits } from './grids';
import type { NeuronRef } from './grids';
import { LESSONS, makeNetContext } from './lessons';
import type { SecondView } from './lessons';
import {
  BoundaryPlot,
  LossCurves,
  NetworkDiagram,
  NetworkHeader,
  NeuronAtlas,
  OutputWeightBars,
  decodeTarget,
  encodeTarget,
  unitSymbol,
} from './panels';
import type { DiagramHover, NetworkVectors } from './panels';
import { displayOrder } from '../../explainer/vectors';
import { NetworkDetail } from './detail';

const META = getExplainer('neural-network')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_EPOCHS = 3000;

const SECOND_VIEW_TITLES: Record<SecondView, string> = {
  loss: 'Train vs held-out loss',
  units: 'Every hidden unit',
  weights: 'How the output combines the units',
};

const SECOND_VIEW_BLURBS: Record<SecondView, string | undefined> = {
  loss: undefined,
  units: 'Hover a unit to send it to the chart.',
  weights: undefined,
};

export default function NeuralNetworkPage() {
  const paramsApi = useExplainerParams<NeuralNetParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const dataset = useMemo(
    () =>
      getClassificationDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed],
  );

  const split = useMemo(
    () => splitDataset(dataset.points, params.testFraction, params.seed),
    [dataset, params.testFraction, params.seed],
  );

  const hasTest = split.test.length > 0;

  /* ---------------- architecture ---------------- */

  const hidden = useMemo(() => parseLayers(params.hiddenLayers), [params.hiddenLayers]);

  // Built from the model knobs only, so a display switch does not reset the run.
  const features = useMemo<InputFeature[]>(() => {
    const chosen: InputFeature[] = [];
    if (params.featX1) chosen.push('x1');
    if (params.featX2) chosen.push('x2');
    if (params.featX1sq) chosen.push('x1sq');
    if (params.featX2sq) chosen.push('x2sq');
    if (params.featX1x2) chosen.push('x1x2');
    if (params.featSinX1) chosen.push('sinx1');
    if (params.featSinX2) chosen.push('sinx2');
    return chosen;
  }, [
    params.featX1,
    params.featX2,
    params.featX1sq,
    params.featX2sq,
    params.featX1x2,
    params.featSinX1,
    params.featSinX2,
  ]);

  const config = useMemo<NetConfig>(
    () => ({
      hiddenLayers: hidden,
      activation: params.activation as Activation,
      learningRate: params.learningRate,
      batchSize: params.batchSize,
      inputFeatures: features,
      initialiser: params.initialiser as Initialiser,
      regularisation: params.regularisation as Regularisation,
      regRate: params.regRate,
      seed: params.seed,
      testFraction: params.testFraction,
    }),
    [
      hidden,
      features,
      params.activation,
      params.learningRate,
      params.batchSize,
      params.initialiser,
      params.regularisation,
      params.regRate,
      params.seed,
      params.testFraction,
    ],
  );

  /* ---------------- simulation ---------------- */

  const sim = useSimulation({
    create: () => createState(config),
    step: (state) => trainEpoch(state, split, config),
    // Done means memorised or diverged.
    isComplete: (state) => state.diverged || (state.epoch > 20 && state.trainLoss < 0.0015),
    deps: [config, split],
    baseStepsPerSecond: 8,
    maxSteps: MAX_EPOCHS,
  });

  const state = sim.state;
  const hiddenLayerCount = Math.max(0, state.layers.length - 1);

  /* ---------------- what every unit is computing ---------------- */

  const grids = useMemo(
    () => computeNeuronGrids(state.layers, features, config.activation, 28),
    [state.layers, features, config.activation],
  );

  /* ---------------- focus: which unit is under the pointer ---------------- */

  const [diagramHover, setDiagramHover] = useState<DiagramHover>(null);
  const [atlasHover, setAtlasHover] = useState<NeuronRef | null>(null);
  const [open, setOpen] = useState<DiagramHover>(null);
  // A hidden unit with its card open is also the unit the output chart shows.
  const pinned = useMemo<NeuronRef | null>(
    () =>
      open?.kind === 'neuron' && open.layer < hiddenLayerCount
        ? { layer: open.layer, neuron: open.neuron }
        : null,
    [open, hiddenLayerCount],
  );
  const [logScale, setLogScale] = useState(false);
  const [secondView, setSecondView] = useState<SecondView>('loss');

  const focus = useMemo<NeuronRef | null>(() => {
    const hovered =
      diagramHover?.kind === 'neuron' && diagramHover.layer < hiddenLayerCount
        ? { layer: diagramHover.layer, neuron: diagramHover.neuron }
        : null;
    const candidate = hovered ?? atlasHover ?? pinned;
    if (!candidate) return null;
    // The architecture can shrink under a pinned unit; drop the reference then.
    const width = state.layers[candidate.layer]?.length ?? 0;
    return candidate.layer < hiddenLayerCount && candidate.neuron < width ? candidate : null;
  }, [diagramHover, atlasHover, pinned, hiddenLayerCount, state.layers]);

  // The unit on show is sampled finely; every other thumbnail stays coarse.
  const focusGrid = useMemo(
    () => (focus ? computeUnitGrid(state.layers, features, config.activation, focus, 64) : null),
    [focus, state.layers, features, config.activation],
  );
  const focusLabel = focus
    ? unitSymbol(focus.neuron) + (hiddenLayerCount > 1 ? ' · hidden ' + (focus.layer + 1) : '')
    : null;

  // Edits made through the header steppers since the lesson step was entered.
  const [layerEdits, setLayerEdits] = useState(0);
  const handleLayers = useCallback(
    (next: number[]) => {
      setOpen(null);
      setLayerEdits((n) => n + 1);
      set('hiddenLayers', formatLayers(next));
    },
    [set],
  );

  /* ---------------- derived numbers ---------------- */

  const gap = state.epoch > 0 && hasTest ? state.testLoss - state.trainLoss : null;
  const largestWeight = useMemo(() => maxAbsWeight(state.layers), [state.layers]);
  const paramCount = useMemo(() => parameterCount(state.layers), [state.layers]);
  const flatUnits = useMemo(() => countFlatUnits(grids), [grids]);
  const hiddenUnits = useMemo(() => hidden.reduce((a, b) => a + b, 0), [hidden]);

  /* ---------------- the lessons ---------------- */

  const openId = useMemo(() => (open ? encodeTarget(open, hiddenLayerCount) : null), [open, hiddenLayerCount]);
  const lessonContext = useMemo(
    () => makeNetContext({ params, state, split, features, grids, ui: { secondView, open: openId, layerEdits } }),
    [params, state, split, features, grids, secondView, openId, layerEdits],
  );
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'second') setSecondView(value as SecondView);
        else if (id === 'scale') setLogScale(value === 'log');
      },
      open: (target: string | null) => setOpen(target === null ? null : decodeTarget(target, hidden.length)),
    }),
    [hidden.length],
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
    setLogScale(false);
    setOpen(null);
  }, [resetLessons, reset, resetRun]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || secondView !== 'loss' || logScale || open !== null;
  const lessonStep = (lessonApi.lesson?.id ?? '') + ':' + lessonApi.view.stepIndex;
  useEffect(() => setLayerEdits(0), [lessonStep]);

  // A node or wire the step points at, drawn like a hover while no card is open.
  const lessonFocus = lessonApi.focus;
  const spot = useMemo<DiagramHover>(() => {
    if (open || !lessonFocus || (lessonFocus.kind !== 'node' && lessonFocus.kind !== 'edge')) return null;
    return decodeTarget(lessonFocus.id, hiddenLayerCount);
  }, [open, lessonFocus, hiddenLayerCount]);

  /* ---------------- the guide ---------------- */

  // Only events reach the card: the running commentary lives in the metrics.
  const guide = useMemo<GuideStatus>(() => {
    const history = state.lossHistory;
    const eta = fmt(config.learningRate, config.learningRate < 0.01 ? 4 : 3);
    if (state.diverged) {
      return { text: 'Diverged: a weight overflowed. η = ' + eta + ' is too large. Lower it and replay.', tone: 'bad' };
    }
    if (state.epoch === 0) return { text: '' };
    if (sim.isComplete) {
      return {
        text:
          'Fitted at epoch ' + state.epoch.toLocaleString('en-US') +
          (gap !== null && gap > 0.08 ? '. The test loss says it memorised rather than learned.' : ''),
        tone: 'good',
      };
    }
    if (sim.iteration >= MAX_EPOCHS) {
      return { text: 'Stopped at ' + MAX_EPOCHS.toLocaleString('en-US') + ' epochs. Reset to run again.', tone: 'warn' };
    }
    const warn = (text: string): GuideStatus => ({ text, tone: 'warn' });
    if (flatUnits > 0 && config.initialiser === 'zeros') {
      return warn('All ' + hiddenUnits + ' hidden units are identical: equal weights get equal gradients.');
    }
    if (flatUnits > 0 && config.activation === 'relu') {
      return warn(flatUnits + ' of ' + hiddenUnits + ' ReLU units are dead: output 0 everywhere, so no gradient reaches them.');
    }
    if (flatUnits > 0) return warn(flatUnits + ' of ' + hiddenUnits + ' units are flat and contribute nothing.');
    if (largestWeight > 20) return warn('Largest weight ' + fmt(largestWeight, 1) + ' and climbing: η = ' + eta + ' is about to overflow.');
    if (state.epoch >= 10) {
      let rises = 0;
      for (let i = Math.max(1, history.length - 10); i < history.length; i++) {
        if (history[i].train > history[i - 1].train) rises += 1;
      }
      const tenBack = history[history.length - 11];
      const twentyBack = history[history.length - 21];
      const hundredBack = history[history.length - 101];
      // Small batches wobble; an overshooting η also ends the window higher than it started.
      if (rises >= 5 && tenBack && state.trainLoss > tenBack.train * 1.02) {
        return warn('The loss rises as often as it falls: η = ' + eta + ' is too large.');
      }
      if (hasTest && state.epoch > 40 && gap !== null && gap > 0.08 && twentyBack && state.testLoss > twentyBack.test) {
        return warn('Test loss is climbing while train loss falls: the network is memorising.');
      }
      if (state.epoch >= 200 && state.trainAccuracy < 0.85 && hundredBack && Math.abs(hundredBack.train - state.trainLoss) < 0.003) {
        return warn('Plateaued: this architecture cannot draw the boundary the data needs.');
      }
    }
    return { text: '' };
  }, [state, config, hasTest, gap, flatUnits, hiddenUnits, largestWeight, sim.isComplete, sim.iteration]);

  const palette = usePalette();

  const headline: Metric[] = useMemo(() => {
    const history = state.lossHistory;
    const previous = history.length > 1 ? history[history.length - 2].train : Number.NaN;
    const trend: Metric['trend'] =
      Number.isFinite(previous) && Number.isFinite(state.trainLoss)
        ? state.trainLoss < previous
          ? 'down'
          : state.trainLoss > previous
            ? 'up'
            : 'flat'
        : 'flat';
    // At epoch 0 the chips read the untrained network, not a blank.
    const blank = state.epoch === 0;
    const trainLoss = blank ? evaluateSet(state.layers, split.train, config).loss : state.trainLoss;
    const testLoss = blank ? evaluateSet(state.layers, split.test, config).loss : state.testLoss;

    return [
      {
        key: 'train-loss',
        label: 'train',
        value: state.diverged ? '∞' : fmtCompact(trainLoss),
        tone: state.diverged ? 'bad' : 'neutral',
        trend,
        colour: palette.blue,
        help: 'The mean of ½(ŷ − t)² between the output and the ±1 target, over the points the network is allowed to learn from.',
      },
      {
        key: 'test-loss',
        label: 'held out',
        value: !hasTest ? 'n/a' : state.diverged ? '∞' : fmtCompact(testLoss),
        tone: gap !== null && gap > 0.08 ? 'warn' : 'neutral',
        colour: palette.orange,
        help: 'The same loss, measured on the ' + split.test.length + ' points that never appear in a gradient. This is the only number that estimates how the model behaves on data it has not seen.',
      },
    ];
  }, [state, hasTest, gap, split, config, palette.blue, palette.orange]);

  const more: Metric[] = useMemo(() => {
    const blank = state.epoch === 0;
    const trainAcc = blank ? evaluateSet(state.layers, split.train, config).accuracy : state.trainAccuracy;
    const testAcc = blank ? evaluateSet(state.layers, split.test, config).accuracy : state.testAccuracy;
    return [
      {
        key: 'train-acc',
        label: 'Train accuracy',
        value: fmtPercent(trainAcc, 1),
        tone: state.trainAccuracy > 0.95 ? 'good' : state.trainAccuracy > 0.8 ? 'neutral' : 'warn',
        help: 'Share of training points on the correct side of the boundary. A network can reach 100% here and still be useless.',
      },
      {
        key: 'test-acc',
        label: 'Held-out accuracy',
        value: !hasTest ? 'n/a' : fmtPercent(testAcc, 1),
        tone: state.testAccuracy > 0.92 ? 'good' : state.testAccuracy > 0.75 ? 'neutral' : 'warn',
        help: 'The honest score. When this stops improving while train accuracy keeps rising, the extra learning is memorisation.',
      },
      {
        key: 'gap',
        label: 'Generalisation gap',
        value: gap === null || !Number.isFinite(gap) ? 'n/a' : fmt(gap, 3),
        tone: gap !== null && gap > 0.08 ? 'bad' : gap !== null && gap > 0.03 ? 'warn' : 'good',
        help: 'The vertical distance between the two loss curves, as one number. Small is good; growing while train loss falls is the signature of overfitting.',
      },
      {
        key: 'max-weight',
        label: 'Largest weight',
        value: state.diverged ? '∞' : fmt(largestWeight, 2),
        tone: largestWeight > 20 ? 'warn' : 'neutral',
        help: 'The biggest |w| in the network. Connection thickness in the diagram is measured against it, and a runaway value here is the first sign of divergence.',
      },
    ];
  }, [state, hasTest, gap, largestWeight, split, config]);

  /* ---------------- render ---------------- */

  // Every vector the diagram shows, in a shared row order.
  const vectors = useMemo<NetworkVectors>(() => {
    const train = split.train;
    const inputs = train.map((p) => inputVector(features, p.x, p.y));
    const perPoint = inputs.map((input) => layerValues(state.layers, input, config.activation));
    return {
      x1: train.map((p) => p.x),
      x2: train.map((p) => p.y),
      label: train.map((p) => (p.label === 1 ? 1 : 0)),
      inputs: features.map((_, i) => inputs.map((row) => row[i] ?? 0)),
      activations: state.layers.map((layer, l) =>
        layer.map((_, u) => perPoint.map((v) => v.activations[l]?.[u] ?? 0)),
      ),
      sums: state.layers.map((layer, l) =>
        layer.map((_, u) => perPoint.map((v) => v.sums[l]?.[u] ?? 0)),
      ),
      order: displayOrder(train.length, (i) => train[i].x, (i) => train[i].label),
    };
  }, [split.train, features, state.layers, config.activation]);
  const n = split.train.length;

  // The hover plate: the symbol, then the number it holds now.
  const plate = useCallback(
    (target: NonNullable<DiagramHover>): readonly string[] | null => {
      const last = state.layers.length - 1;
      if (target.kind === 'point') return ['X, y', n + ' × 3'];
      if (target.kind === 'input') return [FEATURE_LABELS[features[target.index]], n + ' × 1'];
      if (target.kind === 'weight') {
        const w = state.layers[target.layer]?.[target.neuron]?.weights[target.input] ?? 0;
        const from = target.layer === 0 ? FEATURE_LABELS[features[target.input]] : unitSymbol(target.input);
        const to = target.layer === last ? 'ŷ' : unitSymbol(target.neuron);
        return [from + ' → ' + to, 'w = ' + fmt(w, 3)];
      }
      return [target.layer === last ? 'ŷ' : unitSymbol(target.neuron), n + ' × 1'];
    },
    [state.layers, features, n],
  );

  const renderDetail = useCallback(
    (target: NonNullable<DiagramHover>, jump: (next: DiagramHover) => void) =>
      n > 0 ? (
        <NetworkDetail
          target={target}
          layers={state.layers}
          features={features}
          activation={config.activation}
          grids={grids}
          focusGrid={focusGrid}
          vectors={vectors}
          learningRate={config.learningRate}
          batchSize={config.batchSize}
          jump={jump}
        />
      ) : null,
    [n, state.layers, features, config.activation, grids, focusGrid, vectors, config.learningRate, config.batchSize],
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
          unit="epoch"
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
        />
      }
      wide
      architecture={
        <Panel
          id="architecture"
          title="The network"
        >
          <NetworkDiagram
            header={
              <NetworkHeader
                featureCount={features.length}
                pointCount={split.train.length}
                layers={hidden}
                onChange={handleLayers}
                maxLayers={MAX_LAYERS}
                maxUnits={MAX_UNITS}
              />
            }
            layers={state.layers}
            features={features}
            grids={grids}
            vectors={vectors}
            plate={plate}
            hover={diagramHover}
            onHover={setDiagramHover}
            spot={spot}
            open={open}
            onOpen={setOpen}
            showThumbnails={params.showThumbnails}
            epoch={state.epoch}
            fill
            renderDetail={renderDetail}
          />
        </Panel>
      }
      output={
        <>
          <ColumnHead
            title="Output"
            blurb={
              <>
                <Tex tex={architectureLabel(features.length, hidden).replace(/→/g, '\\to')} /> ·{' '}
                {paramCount} parameters
              </>
            }
          />

          <Panel
            id="boundary"
            fill
            title={focusLabel ? 'One unit’s view' : 'What the network predicts'}
            actions={
              <Legend
                dense
                items={[
                  { label: dataset.classNames[0], colour: palette.classA, shape: 'dot' },
                  { label: dataset.classNames[1], colour: palette.classB, shape: 'cross' },
                  ...(hasTest && params.showTest
                    ? [{ label: 'held out', colour: palette.muted, shape: 'ring' as const }]
                    : []),
                ]}
              />
            }
          >
            <BoundaryPlot
              train={split.train}
              test={split.test}
              layers={state.layers}
              features={features}
              activation={config.activation}
              focus={focus}
              focusGrid={focusGrid}
              focusLabel={focusLabel}
              showTest={params.showTest}
              bands={params.bands}
              diverged={state.diverged}
              epoch={state.epoch}
            />
          </Panel>

          <Panel
            id="second"
            fill
            title={SECOND_VIEW_TITLES[secondView]}
            subtitle={SECOND_VIEW_BLURBS[secondView]}
            actions={
              <>
                {secondView === 'loss' ? (
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
                    { value: 'loss', label: 'Loss' },
                    { value: 'units', label: 'Units' },
                    { value: 'weights', label: 'Weights' },
                  ]}
                  onChange={(next) => setSecondView(next as SecondView)}
                />
              </>
            }
          >
            {secondView === 'loss' ? (
              <LossCurves history={state.lossHistory} logScale={logScale} hasTest={hasTest} />
            ) : secondView === 'units' ? (
              <NeuronAtlas
                grids={grids}
                focus={focus}
                onFocus={setAtlasHover}
                activationLabel={
                  params.activation === 'linear' ? 'no activation' : String(params.activation)
                }
              />
            ) : (
              <OutputWeightBars
                layers={state.layers}
                features={features}
                learningRate={config.learningRate}
                batchSize={config.batchSize}
              />
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
