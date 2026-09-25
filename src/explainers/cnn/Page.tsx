/** Convolutional networks explainer: one probe image through every layer, in 2D, 3D or one window at a time, while a glyph classifier trains. */

import { useCallback, useEffect, useMemo, useState } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import { parseArchTarget } from '../../explainer/components/ArchitectureView';
import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { Legend, MetricStrip, Panel, ParamsReset, ViewSwitch } from '../../explainer/components/Panels';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { Transport } from '../../explainer/components/Transport';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { generateGlyphs } from '../../lib/datasets/images';
import { fmt, fmtCompact, fmtKnob, fmtPercent } from '../../lib/math/stats';
import { channel, createState, forwardProbe, parameterCount, step as batchStep } from '../../lib/ml/conv';
import type { CnnSpec } from '../../lib/ml/conv';
import type { OptimiserName } from '../../lib/ml/optim';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { CnnParams } from './config';
import { CnnDetail } from './detail';
import { CnnDiagram } from './diagram';
import { useSlide } from './useSlide';
import { WindowView } from './window';
import { LESSONS, isFitted, makeCnnContext } from './lessons';
import type { ArchView, SecondView } from './lessons';
import { ConfusionGrid, FilterGallery, ProbeStepper, TestGallery, TrainingCurves } from './panels';
import { architectureLabel, brightest, buildScene, parseNode, windowDims } from './scene';
import type { Window } from './scene';

const META = getExplainer('cnn')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_STEPS = 4000;

const SECOND_TITLES: Record<SecondView, string> = {
  curves: 'Loss over batches',
  accuracy: 'Accuracy over batches',
  confusion: 'Confusion on held-out glyphs',
  filters: 'The kernels',
};

export default function CnnPage() {
  const paramsApi = useExplainerParams<CnnParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data and model ---------------- */

  const data = useMemo(
    () =>
      generateGlyphs({
        size: 16,
        classCount: params.classCount,
        count: params.count,
        jitter: params.jitter,
        thickness: params.thickness,
        noise: params.noise,
        seed: params.seed,
        trainFraction: 0.75,
      }),
    [params.classCount, params.count, params.jitter, params.thickness, params.noise, params.seed],
  );

  const spec = useMemo<CnnSpec>(
    () => ({
      size: 16,
      model: params.model === 'dense' ? 'dense' : 'cnn',
      kernel: Number(params.kernel) === 5 ? 5 : 3,
      filters: params.filters,
      padding: params.padding === 'same' ? 'same' : 'valid',
      pool: params.pool === 'avg' ? 'avg' : params.pool === 'none' ? 'none' : 'max',
      secondConv: params.secondConv,
      secondFilters: params.filters,
      dense: params.dense,
      classCount: data.classNames.length,
      seed: params.seed,
      optimiser: { name: params.optimiser as OptimiserName },
      learningRate: params.learningRate,
      batchSize: params.batchSize,
      evalEvery: 5,
    }),
    [params.model, params.kernel, params.filters, params.padding, params.pool, params.secondConv, params.dense, data.classNames.length, params.seed, params.optimiser, params.learningRate, params.batchSize],
  );

  /* ---------------- simulation ---------------- */

  const sim = useSimulation({
    create: () => createState(spec, data),
    step: (state) => batchStep(state, data, spec),
    isComplete: (state) => state.diverged || isFitted(state),
    deps: [spec, data],
    baseStepsPerSecond: 8,
    maxSteps: MAX_STEPS,
  });
  const state = sim.state;

  /* ---------------- the probe ---------------- */

  const probeIndex = Math.min(params.probe, Math.max(0, data.testIndex.length - 1));
  const probeImage = data.testIndex[probeIndex] ?? 0;
  const cache = useMemo(() => forwardProbe(state.params, data.images[probeImage], spec), [state.params, data.images, probeImage, spec]);
  const label = data.labels[probeImage] ?? 0;
  const paramCount = useMemo(() => parameterCount(state.params), [state.params]);

  const confidence = useCallback(
    (index: number) => {
      const image = data.testIndex[index];
      if (image === undefined) return 0;
      return forwardProbe(state.params, data.images[image], spec).probs[data.labels[image]] ?? 0;
    },
    [data, state.params, spec],
  );

  /* ---------------- views and the open card ---------------- */

  const [selection, setSelection] = useState<ArchSelection | null>(null);
  const [archView, setArchView] = useState<ArchView>('2d');
  const [secondView, setSecondView] = useState<SecondView>('curves');
  const [windowAt, setWindowAt] = useState<Window | null>(null);
  // Opening a map or a pooled map asks for its brightest cell (x = -1 until the cache has the stage); anything else clears it.
  const select = useCallback((next: ArchSelection | null) => {
    setSelection(next);
    const { kind, stage, index } = next && next.kind === 'node' ? parseNode(next.id) : { kind: '', stage: 0, index: 0 };
    setWindowAt(kind === 'map' || kind === 'pool' ? { kind, stage, index, x: -1, y: -1 } : null);
  }, []);
  const stageCount = spec.model === 'dense' ? 0 : spec.secondConv ? 2 : 1;
  useEffect(() => {
    if (!windowAt) {
      // The window view needs a window: the first map.
      if (archView === 'window' && stageCount > 0) setWindowAt({ kind: 'map', stage: 0, index: 0, x: -1, y: -1 });
      return;
    }
    if (windowAt.stage >= stageCount) {
      setWindowAt(null);
      return;
    }
    const st = cache.stages[windowAt.stage];
    if (!st) return;
    const t = windowAt.kind === 'pool' ? st.pooled : st.a;
    // Resolved and still inside the stack: nothing to do.
    if (windowAt.x >= 0 && windowAt.index < t.c && windowAt.x < t.w && windowAt.y < t.h) return;
    const index = Math.min(windowAt.index, t.c - 1);
    const spot = brightest(channel(t, index), t.w);
    setWindowAt({ ...windowAt, index, x: spot.x, y: spot.y });
  }, [archView, windowAt, stageCount, cache]);
  // Views only see a resolved window.
  const windowShown = useMemo(() => {
    if (!windowAt || windowAt.x < 0) return null;
    const st = cache.stages[windowAt.stage];
    if (!st) return null;
    const t = windowAt.kind === 'pool' ? st.pooled : st.a;
    return windowAt.index < t.c && windowAt.x < t.w && windowAt.y < t.h ? windowAt : null;
  }, [windowAt, cache]);
  const slideDims = useMemo(() => (windowShown ? windowDims(cache, windowShown) : { cols: 1, rows: 1 }), [cache, windowShown]);
  const mapOpen = selection?.kind === 'node' && selection.id.startsWith('map:');
  const slide = useSlide((archView === 'window' || mapOpen) && windowShown?.kind === 'map', windowShown, slideDims, setWindowAt);

  const scene = useMemo(
    () => buildScene({ params: state.params, cache, spec, classSymbols: data.classSymbols, classNames: data.classNames, label, trained: state.step > 0 }),
    [state.params, cache, spec, data.classSymbols, data.classNames, label, state.step],
  );

  /* ---------------- the lessons ---------------- */

  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'arch') setArchView(value === '3d' ? '3d' : value === 'window' ? 'window' : '2d');
        else if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => select(target === null ? null : parseArchTarget(target)),
    }),
    [select],
  );
  const openId = selection ? selection.kind + ':' + selection.id : null;
  const lessonContext = useMemo(
    () => makeCnnContext({ params, state, data, spec, cache, label, ui: { view: archView, secondView, open: openId } }),
    [params, state, data, spec, cache, label, archView, secondView, openId],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });
  const focus = lessonApi.focus;
  const spot = useMemo<ArchSelection | null>(
    () => (focus && (focus.kind === 'node' || focus.kind === 'edge') && !selection ? { kind: focus.kind, id: focus.id } : null),
    [focus, selection],
  );

  /* ---------------- the chapter reset ---------------- */

  // Free play with no lesson marks, default knobs and views, a fresh run.
  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  // Bumped by the chapter reset, so the diagram forgets how the 3D view was turned.
  const [resets, setResets] = useState(0);
  const resetAll = useCallback(() => {
    resetLessons();
    reset();
    resetRun();
    setArchView('2d');
    setSecondView('curves');
    select(null);
    slide.reset();
    setResets((n) => n + 1);
  }, [resetLessons, reset, resetRun, select, slide]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || archView !== '2d' || secondView !== 'curves' || selection !== null;

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    if (state.diverged) return { text: 'The weights blew up: η = ' + fmtKnob(params.learningRate) + ' is too large. Lower it and replay.', tone: 'bad' };
    if (state.step === 0) return { text: '' };
    if (sim.isComplete) return { text: 'Fitted: ' + fmtPercent(state.testAccuracy, 0) + ' of the held-out glyphs right after ' + state.step + ' batches.', tone: 'good' };
    const chance = 1 / data.classNames.length;
    if (state.step >= 30 && state.trainAccuracy < chance + 0.1) {
      return { text: 'Still at chance after ' + state.step + ' batches: every map may be dead. Lower the rate or add a kernel.', tone: 'warn' };
    }
    if (state.trainAccuracy - state.testAccuracy > 0.2) {
      return { text: 'Memorising: ' + fmtPercent(state.trainAccuracy, 0) + ' on training images but ' + fmtPercent(state.testAccuracy, 0) + ' on held-out ones.', tone: 'warn' };
    }
    if (sim.iteration >= MAX_STEPS) return { text: 'Stopped at ' + MAX_STEPS.toLocaleString('en-US') + ' batches. Reset to run again.', tone: 'warn' };
    return { text: '' };
  }, [state, params.learningRate, sim.isComplete, sim.iteration, data.classNames.length]);

  /* ---------------- metrics ---------------- */

  const palette = usePalette();
  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'test-acc',
        label: 'held out',
        value: fmtPercent(state.testAccuracy, 0),
        colour: palette.orange,
        tone: isFitted(state) ? 'good' : 'neutral',
        help: 'Share of held-out glyphs called correctly at the last evaluation. The one number to trust.',
      },
      {
        key: 'train-loss',
        label: 'train loss',
        value: state.diverged ? '∞' : fmtCompact(state.trainLoss),
        colour: palette.blue,
        tone: state.diverged ? 'bad' : 'neutral',
        help: 'Mean cross-entropy over the training images at the last evaluation.',
      },
    ],
    [state, palette],
  );

  const more: Metric[] = useMemo(
    () => [
      { key: 'train-acc', label: 'Train accuracy', value: fmtPercent(state.trainAccuracy, 0), help: 'Share of training glyphs called correctly. Far above the held-out score means memorising.' },
      { key: 'test-loss', label: 'Held-out loss', value: state.diverged ? '∞' : fmtCompact(state.testLoss), help: 'Mean cross-entropy on the held-out glyphs.' },
      { key: 'params', label: 'Parameters', value: paramCount.toLocaleString('en-US'), help: 'Every weight and bias in the model. Compare the dense baseline.' },
      { key: 'epoch', label: 'Epochs', value: String(state.epoch), help: 'Full passes over the training images.' },
    ],
    [state, paramCount],
  );

  /* ---------------- render ---------------- */

  const renderDetail = useCallback(
    (active: ArchSelection, jump: (next: ArchSelection) => void) => (
      <CnnDetail
        selection={active}
        scene={scene}
        params={state.params}
        cache={cache}
        spec={spec}
        classNames={data.classNames}
        classSymbols={data.classSymbols}
        window={windowShown}
        onWindow={setWindowAt}
        slide={slide}
        jump={jump}
      />
    ),
    [scene, state.params, cache, spec, data.classNames, data.classSymbols, windowShown, slide],
  );

  // What the network says about the probe, and whether that is a trained answer or a random guess.
  const call = (state.step > 0 ? ' · calls it a ' : ' · untrained guess: ') + (data.classNames[cache.predicted] ?? '') + ', p = ' + fmt(cache.probs[cache.predicted] ?? 0, 2);
  const trainCount = data.trainIndex.length;
  const testCount = data.testIndex.length;
  const batchesPerEpoch = Math.ceil(trainCount / Math.max(1, params.batchSize));
  const batchHelp =
    'One batch: ' + params.batchSize + ' training images go through the network, the loss is measured and every weight takes one step. ' +
    batchesPerEpoch + ' batches = 1 epoch, a full pass over the ' + trainCount + ' training images. The ' + testCount + ' held-out images are only ever scored.';

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
      <Studio
        meta={META}
        shareUrl={shareUrl}
        lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
        transport={<Transport sim={sim} unit="batch" unitHelp={batchHelp} completeLabel={state.diverged ? 'Diverged' : 'Fitted'} completeTone={state.diverged ? 'bad' : 'good'} />}
        topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={set} lesson={lastPreset} named={lessonApi.view.knobs} />}
        tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
        controls={<ControlPanel groups={RAIL_GROUPS} params={params} onChange={set} />}
        wide
        architecture={
          <Panel
            id="architecture"
            title="The network"
            subtitle={'Held-out image ' + (probeIndex + 1) + ', a ' + (data.classNames[label] ?? '') + call}
            actions={
              <>
                <ProbeStepper index={probeIndex} count={testCount} onChange={(next) => set('probe', next)} />
                <Legend
                  dense
                  items={[
                    { label: 'pixel', colour: palette.text, shape: 'square' },
                    { label: 'w +', colour: palette.positive, shape: 'square' },
                    { label: 'w −', colour: palette.negative, shape: 'square' },
                  ]}
                />
                <ViewSwitch
                  id="arch"
                  label="Architecture view"
                  value={archView}
                  options={[
                    { value: '2d', label: '2D' },
                    { value: '3d', label: '3D' },
                    { value: 'window', label: 'Window' },
                  ]}
                  onChange={(next) => setArchView(next === '3d' ? '3d' : next === 'window' ? 'window' : '2d')}
                />
              </>
            }
          >
            {archView === 'window' ? (
              <WindowView
                params={state.params}
                cache={cache}
                spec={spec}
                window={windowShown}
                onWindow={setWindowAt}
                slide={slide}
                description={
                  windowShown
                    ? 'One ' + (windowShown.kind === 'pool' ? 'pooled cell' : 'map pixel') + ' at row ' + windowShown.y + ', column ' + windowShown.x + ', the block it reads and the numbers between them.'
                    : 'The dense model has no window.'
                }
                redrawKey={state.step + ':' + probeImage}
              />
            ) : (
            <CnnDiagram
              key={resets}
              scene={scene}
              params={state.params}
              cache={cache}
              view={archView}
              selection={selection}
              onSelect={select}
              spot={spot}
              window={windowShown}
              renderDetail={renderDetail}
              description={
                'The network for one probe image: ' + architectureLabel(spec, state.params) + '. The probe is a ' + (data.classNames[label] ?? 'glyph') +
                (state.step > 0 ? ' and the network calls it a ' + (data.classNames[cache.predicted] ?? '') : '') + '.'
              }
              redrawKey={state.step + ':' + probeImage}
              phase={state.step}
            />
            )}
          </Panel>
        }
        output={
          <>
            <ColumnHead title="Output" blurb={architectureLabel(spec, state.params, false)} />

            <Panel
              id="gallery"
              fill
              title="Held-out glyphs"
              subtitle={
                testCount + ' of ' + data.images.length + ' glyphs held out, ' + trainCount + ' train' +
                (testCount > 40 ? '; the first 40 shown' : '') + '. Click one to make it the probe.'
              }
              actions={
                <Legend
                  dense
                  items={[
                    { label: 'right', colour: palette.green, shape: 'ring' },
                    { label: 'wrong', colour: palette.orange, shape: 'ring' },
                  ]}
                />
              }
            >
              <TestGallery data={data} state={state} probe={probeIndex} onProbe={(index) => set('probe', index)} confidence={confidence} />
            </Panel>

            <Panel
              id="second"
              fill
              title={SECOND_TITLES[secondView]}
              actions={
                <>
                  {secondView === 'curves' || secondView === 'accuracy' ? (
                    <Legend
                      dense
                      items={[
                        { label: 'train', colour: palette.blue, shape: 'line' },
                        { label: 'held out', colour: palette.orange, shape: 'line' },
                      ]}
                    />
                  ) : null}
                  <ViewSwitch
                    id="second"
                    label="Second chart"
                    value={secondView}
                    options={[
                      { value: 'curves', label: 'Loss' },
                      { value: 'accuracy', label: 'Accuracy' },
                      { value: 'confusion', label: 'Confusion' },
                      { value: 'filters', label: 'Kernels' },
                    ]}
                    onChange={(next) => setSecondView(next as SecondView)}
                  />
                </>
              }
            >
              {secondView === 'curves' ? (
                <TrainingCurves history={state.history} view="loss" />
              ) : secondView === 'accuracy' ? (
                <TrainingCurves history={state.history} view="accuracy" />
              ) : secondView === 'confusion' ? (
                <ConfusionGrid data={data} predictions={state.testPredictions} />
              ) : (
                <FilterGallery params={state.params} />
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
