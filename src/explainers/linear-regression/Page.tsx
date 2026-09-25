/** Linear regression explainer. The reference wiring every other page follows. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import {
  ColumnHead,
  Studio,
} from '../../explainer/components/Studio';
import { ArchitectureView, parseArchTarget } from '../../explainer/components/ArchitectureView';
import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import {
  Legend,
  Math as Tex,
  MetricStrip,
  Panel,
  ParamsReset,
  ViewSwitch,
} from '../../explainer/components/Panels';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { Transport } from '../../explainer/components/Transport';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { useSimulation } from '../../explainer/useSimulation';
import { useEditable } from '../../explainer/useEditable';
import type { GuideStatus, Metric } from '../../explainer/types';
import { getRegressionDataset } from '../../lib/datasets/regression';
import { fmt, fmtCompact, fmtKnob, rSquared, meanAbsoluteError, sup } from '../../lib/math/stats';
import {
  closedFormSolution,
  computeLoss,
  createState,
  features,
  makeNormaliser,
  predict,
  step as gradientStep,
  toOriginalUnits,
} from '../../lib/ml/linearRegression';
import type { BatchMode, LinRegConfig, OptimiserName } from '../../lib/ml/linearRegression';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import type { makeFrame } from '../../lib/viz/canvas';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { LinRegParams } from './config';
import { FitPlot, LossCurve, LossSurface } from './panels';
import { buildLinRegGraph } from './graph';
import { LinRegDetail } from './detail';
import { LESSONS, makeLinRegContext } from './lessons';
import type { GraphView, LinRegVectors } from './graph';
import { displayOrder, maxAbs } from '../../explainer/vectors';
import { texNum } from '../../explainer/tex';

const META = getExplainer('linear-regression')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_STEPS = 60_000;

export default function LinearRegressionPage() {
  const paramsApi = useExplainerParams<LinRegParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const generated = useMemo(
    () =>
      getRegressionDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed],
  );

  // A local copy so dragging a point edits the data without regenerating it.
  const [data, setData] = useEditable(generated);

  const config = useMemo<LinRegConfig>(
    () => ({
      learningRate: params.learningRate,
      degree: params.degree,
      optimiser: params.optimiser as OptimiserName,
      momentum: params.momentum,
      batchMode: params.batchMode as BatchMode,
      batchSize: params.batchSize,
      l2: params.l2,
      standardise: params.standardise,
      seed: params.seed,
    }),
    [
      params.learningRate,
      params.degree,
      params.optimiser,
      params.momentum,
      params.batchMode,
      params.batchSize,
      params.l2,
      params.standardise,
      params.seed,
    ],
  );

  const norm = useMemo(
    () => makeNormaliser(data.points, params.standardise),
    [data.points, params.standardise],
  );

  const closedForm = useMemo(
    () => closedFormSolution(data.points, config.degree, norm, config.l2),
    [data.points, config.degree, config.l2, norm],
  );

  const closedFormLoss = useMemo(
    () => (closedForm ? computeLoss(closedForm, data.points, config, norm) : null),
    [closedForm, data.points, config, norm],
  );

  /* ---------------- simulation ---------------- */

  // Resets on a new dataset or model, not on a dragged point: `data` is the editable copy.
  const sim = useSimulation({
    create: () => createState(config),
    step: (state) => gradientStep(state, data.points, config, norm),
    isComplete: (state) => state.converged || state.diverged,
    deps: [config, generated, params.standardise],
    baseStepsPerSecond: 14,
    maxSteps: MAX_STEPS,
  });

  const state = sim.state;

  /* ---------------- derived numbers ---------------- */

  const predictions = useMemo(
    () => data.points.map((p) => predict(state.weights, p.x, config.degree, norm)),
    [data.points, state.weights, config.degree, norm],
  );

  const actuals = useMemo(() => data.points.map((p) => p.y), [data.points]);

  const r2 = useMemo(() => rSquared(actuals, predictions), [actuals, predictions]);
  const mae = useMemo(() => meanAbsoluteError(actuals, predictions), [actuals, predictions]);
  const line = useMemo(() => toOriginalUnits(state.weights, norm), [state.weights, norm]);

  const gapToOptimum =
    closedFormLoss !== null && Number.isFinite(state.loss) && state.epoch > 0
      ? state.loss - closedFormLoss
      : null;

  // The loss at all-zero weights, so the first step has a "before" to report.
  const initialLoss = useMemo(
    () => computeLoss(new Array<number>(config.degree + 1).fill(0), data.points, config, norm),
    [config, data.points, norm],
  );

  /* ---------------- the guide ---------------- */

  // Only events reach the card.
  const guide = useMemo<GuideStatus>(() => {
    const alpha = fmtKnob(config.learningRate);
    if (state.diverged) {
      return {
        text:
          !config.standardise && config.degree > 1
            ? 'Diverged: raw x' + sup(config.degree) + ' overshoots. Turn Standardise x on and replay.'
            : 'Diverged: α = ' + alpha + ' is too large. Lower it and replay.',
        tone: 'bad',
      };
    }
    if (state.converged) {
      const exact = gapToOptimum !== null && gapToOptimum < 0.01;
      return {
        text: 'Converged at step ' + state.epoch.toLocaleString('en-US') + (exact ? ', the exact solution' : ''),
        tone: 'good',
      };
    }
    if (sim.iteration >= MAX_STEPS) {
      return { text: 'Stopped at ' + MAX_STEPS.toLocaleString('en-US') + ' steps. Reset to run again.', tone: 'warn' };
    }
    if (state.epoch >= 100 && gapToOptimum !== null && gapToOptimum > 0.5 * initialLoss) {
      return { text: 'Barely moving: α = ' + alpha + ' is too small.', tone: 'warn' };
    }
    return { text: '' };
  }, [state, config, initialLoss, gapToOptimum, sim.iteration]);

  const palette = usePalette();

  const headline: Metric[] = useMemo(() => {
    const previous = state.lossHistory[state.lossHistory.length - 2];
    const trend =
      Number.isFinite(previous) && Number.isFinite(state.loss)
        ? state.loss < previous
          ? 'down'
          : state.loss > previous
            ? 'up'
            : 'flat'
        : 'flat';
    return [
      {
        key: 'loss',
        label: 'loss',
        value: state.diverged ? '∞' : fmtCompact(state.epoch === 0 ? initialLoss : state.loss),
        tone: state.diverged ? 'bad' : 'neutral',
        trend: trend as Metric['trend'],
        colour: palette.orange,
        help:
          (config.l2 > 0 ? 'Mean squared error plus the L2 charge λΣw². ' : 'Mean squared error over every point. ') +
          'This is the number gradient descent is trying to make small' +
          (closedFormLoss !== null ? '; the exact solution scores ' + fmtCompact(closedFormLoss) : '') +
          '.',
      },
      {
        key: 'r2',
        label: 'R²',
        value: state.diverged ? 'n/a' : fmt(r2, 3),
        tone: r2 > 0.9 ? 'good' : r2 > 0.5 ? 'neutral' : 'warn',
        help: '1.0 means the model explains every wiggle in the data; 0 means it does no better than predicting the average.',
      },
    ];
  }, [state, r2, closedFormLoss, initialLoss, palette.orange, config.l2]);

  const more: Metric[] = useMemo(() => {
    return [
      {
        key: 'mae',
        label: 'Mean abs. error',
        value: state.diverged || state.epoch === 0 ? 'n/a' : fmtCompact(mae),
        help: 'The average distance between prediction and truth, in the units of the y-axis. Less punishing of outliers than MSE.',
      },
      {
        key: 'grad',
        label: 'Gradient norm',
        value: state.epoch === 0 ? 'n/a' : fmtCompact(state.gradientNorm),
        caption: state.converged ? 'converged' : undefined,
        tone: state.converged ? 'good' : 'neutral',
        help: 'How steep the loss is here. Descent stops when this reaches zero, because there is no downhill left.',
      },
      {
        key: 'gap',
        label: 'Gap to optimum',
        value:
          gapToOptimum === null || !Number.isFinite(gapToOptimum) ? 'n/a' : fmtCompact(Math.max(0, gapToOptimum)),
        tone: gapToOptimum !== null && gapToOptimum < 0.01 ? 'good' : 'neutral',
        help: 'How far the iterative answer still is from the one the normal equations give directly.',
      },
    ];
  }, [state, mae, gapToOptimum]);

  /* ---------------- point dragging ---------------- */

  const frameRef = useRef<ReturnType<typeof makeFrame> | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  // Drag count, read through a getter against a baseline taken on step entry.
  const [pointEdits, setPointEdits] = useState(0);
  const editsAtEntry = useRef(0);
  const movedRef = useRef(false);
  // A finished run refuses to play; a drag un-latches it, then resumes here.
  const [resume, setResume] = useState(false);
  useEffect(() => {
    if (!resume || sim.isComplete || sim.isRunning) return;
    setResume(false);
    sim.play();
  }, [resume, sim]);

  const handleFrame = useCallback((frame: ReturnType<typeof makeFrame>) => {
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
        if (state.converged) {
          sim.patch((s) => ({ ...s, converged: false }));
          setResume(true);
        }
      }
    },
    [dragIndex, findPoint, hoverIndex, state.converged, sim, setData, data.yRange],
  );

  const handleUp = useCallback(() => setDragIndex(null), []);

  /* ---------------- render ---------------- */

  const [logScale, setLogScale] = useState(false);
  const [lossView, setLossView] = useState<'curve' | 'surface'>('curve');
  const [selection, setSelection] = useState<ArchSelection | null>(null);

  // Model view until training starts, unless chosen explicitly.
  const [graphView, setGraphView] = useState<GraphView | null>(null);
  const view: GraphView = graphView ?? (sim.isRunning || state.epoch > 0 ? 'training' : 'model');

  // The lessons drive the presets, the transport, the views and the open card.
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'graph') setGraphView(value === 'auto' ? null : (value as GraphView));
        else if (id === 'second') setLossView(value as 'curve' | 'surface');
        else if (id === 'scale') setLogScale(value === 'log');
      },
      open: (target: string | null) => setSelection(target === null ? null : parseArchTarget(target)),
    }),
    [],
  );
  const lessonContext = useMemo(
    () =>
      makeLinRegContext({
        params,
        state,
        data,
        config,
        norm,
        closedFormLoss,
        initialLoss,
        ui: {
          view,
          lossView,
          logScale,
          open: selection ? selection.kind + ':' + selection.id : null,
          get pointEdits() {
            return pointEdits - editsAtEntry.current;
          },
        },
      }),
    [params, state, data, config, norm, closedFormLoss, initialLoss, view, lossView, logScale, selection, pointEdits],
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
    setLogScale(false);
    setLossView('curve');
    setGraphView(null);
    setSelection(null);
  }, [resetLessons, reset, resetRun, generated, setData]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || data !== generated || logScale || lossView !== 'curve' || graphView !== null || selection !== null;

  // A node or edge the step points at, drawn as a hover while no card is open.
  const focus = lessonApi.focus;
  const spot = useMemo<ArchSelection | null>(
    () => (focus && (focus.kind === 'node' || focus.kind === 'edge') && !selection ? { kind: focus.kind, id: focus.id } : null),
    [focus, selection],
  );

  const fed = config.standardise ? '\\tilde{x}' : 'x';
  const term = (w: number, i: number) => texNum(Math.abs(w)) + (i === 0 ? '' : i === 1 ? fed : fed + '^{' + i + '}');
  const signed = (w: number, i: number) => (i === 0 ? (w < 0 ? '-' : '') : w < 0 ? ' - ' : ' + ') + term(w, i);
  // Five terms fit the column head; the card has the rest.
  const equationText = state.diverged
    ? 'y = \\text{diverged}'
    : config.degree === 1
      ? 'y = ' + texNum(line.slope) + '\\,x' + (line.intercept < 0 ? ' - ' : ' + ') + texNum(Math.abs(line.intercept))
      : 'y = ' + state.weights.slice(0, 5).map(signed).join('') + (config.degree > 4 ? ' + \\cdots' : '');

  // Every vector the diagram shows, one value per point, in a shared row order.
  const vectors = useMemo<LinRegVectors>(() => {
    const rows = data.points.map((p) => features(p.x, config.degree, norm));
    return {
      x: data.points.map((p) => p.x),
      y: actuals,
      features: Array.from({ length: config.degree + 1 }, (_, k) => rows.map((r) => r[k] ?? 0)),
      prediction: predictions,
      yScale: Math.max(maxAbs(actuals), maxAbs(predictions)) || 1,
    };
  }, [data.points, config.degree, norm, actuals, predictions]);
  const order = useMemo(
    () => displayOrder(data.points.length, (i) => data.points[i].x),
    [data.points],
  );
  const batch = useMemo(
    () =>
      config.batchMode !== 'batch' && state.epoch > 0 && state.lastBatch.length < data.points.length
        ? new Set(state.lastBatch)
        : null,
    [config.batchMode, state.epoch, state.lastBatch, data.points.length],
  );

  const graph = useMemo(
    () =>
      buildLinRegGraph({
        state,
        degree: config.degree,
        started: state.epoch > 0,
        standardised: config.standardise && norm.std !== 1,
        vectors,
        order,
        batch,
        view,
      }),
    [state, config.degree, config.standardise, norm, vectors, order, batch, view],
  );

  const renderDetail = useCallback(
    (active: ArchSelection, jump: (next: ArchSelection) => void) => (
      <LinRegDetail
        selection={active}
        state={state}
        config={config}
        norm={norm}
        standardised={config.standardise && norm.std !== 1}
        vectors={vectors}
        order={order}
        batch={batch}
        jump={jump}
      />
    ),
    [state, config, norm, vectors, order, batch],
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
          unit="step"
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
          title={view === 'model' ? 'The model' : 'The model, and how it learns'}
          actions={
            <>
              <Legend
                dense
                items={[
                  { label: 'w > 0', colour: palette.positive, shape: 'line' },
                  { label: 'w < 0', colour: palette.negative, shape: 'line' },
                ]}
              />
              <ViewSwitch
                id="graph"
                label="Graph"
                value={view}
                options={[
                  { value: 'model', label: 'Model' },
                  { value: 'training', label: 'Training' },
                ]}
                onChange={(next) => setGraphView(next as GraphView)}
              />
            </>
          }
        >
          <ArchitectureView
            graph={graph}
            selection={selection}
            onSelect={setSelection}
            renderDetail={renderDetail}
            description={
              'Computation graph: x and 1 in, a weighted sum, the prediction out' +
              (view === 'training' ? ', then the mean squared loss and its gradient' : '') +
              '. Step ' +
              state.epoch +
              '.'
            }
            redrawKey={view + ':' + fmt(line.slope, 4)}
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
                  { label: 'data', colour: palette.blue, shape: 'dot' },
                  { label: 'model', colour: palette.orange, shape: 'line' },
                  ...(params.showClosedForm
                    ? [{ label: 'exact', colour: palette.green, shape: 'dashed' as const }]
                    : []),
                ]}
              />
            }
          >
            <FitPlot
              data={data}
              state={state}
              config={config}
              norm={norm}
              closedForm={closedForm}
              showResiduals={params.showResiduals}
              showSquares={params.showSquares}
              showTruth={params.showTruth}
              showClosedForm={params.showClosedForm}
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
            title={lossView === 'curve' ? 'Loss over time' : 'The loss landscape'}
            subtitle={
              lossView === 'surface'
                ? config.degree === 1
                  ? 'Loss at every (b, w); the trail is the path taken.'
                  : 'Needs exactly two weights to draw.'
                : undefined
            }
            actions={
              <>
                {lossView === 'curve' ? (
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
                  value={lossView}
                  options={[
                    { value: 'curve', label: 'Curve' },
                    { value: 'surface', label: 'Landscape' },
                  ]}
                  onChange={(next) => setLossView(next as 'curve' | 'surface')}
                />
              </>
            }
          >
            {lossView === 'curve' ? (
              <LossCurve
                history={state.lossHistory}
                epoch={state.epoch}
                initialLoss={initialLoss}
                closedFormLoss={closedFormLoss}
                logScale={logScale}
                penalised={config.l2 > 0}
              />
            ) : (
              <LossSurface
                data={data}
                state={state}
                config={config}
                norm={norm}
                closedForm={closedForm}
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
