/** Logistic regression explainer. Same wiring as linear regression, plus the sigmoid and a threshold. */

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
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useLesson } from '../../explainer/useLesson';
import { Transport } from '../../explainer/components/Transport';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useSimulation } from '../../explainer/useSimulation';
import { useEditable } from '../../explainer/useEditable';
import type { GuideStatus, Metric } from '../../explainer/types';
import {
  addPoint,
  getClassificationDataset,
  nearestPointIndex,
  removeNearestPoint,
} from '../../lib/datasets/points';
import {
  accuracy as accuracyOf,
  confusionMatrix,
  fmt,
  fmtPercent,
  logLoss,
  precisionRecallF1,
  rocAuc,
  rocCurve,
} from '../../lib/math/stats';
import {
  createState,
  featureNames,
  features,
  makeStandardiser,
  predictAll,
  selectBatch,
  step as gradientStep,
} from '../../lib/ml/logisticRegression';
import type { FeatureMap, LogRegConfig } from '../../lib/ml/logisticRegression';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { classColour } from '../../lib/viz/plots';
import type { makeFrame } from '../../lib/viz/canvas';
import { CONTROL_GROUPS, DEFAULT_PARAMS, FULL_BATCH, PRESETS } from './config';
import type { LogRegParams } from './config';
import {
  ConfusionTable,
  DecisionField,
  RocPlot,
  SigmoidCurve,
  TrainingCurves,
} from './panels';
import { buildLogRegGraph } from './graph';
import { LogRegDetail } from './detail';
import { LESSONS, makeLogRegContext } from './lessons';
import type { SecondView } from './lessons';
import type { GraphView, LogRegVectors } from './graph';
import { displayOrder } from '../../explainer/vectors';

const META = getExplainer('logistic-regression')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_STEPS = 60_000;

const SECOND_VIEW_TITLES: Record<SecondView, string> = {
  sigmoid: 'The sigmoid',
  curves: 'Loss and accuracy',
  confusion: 'Confusion matrix',
  roc: 'ROC curve',
};

const SECOND_VIEW_BLURBS: Record<SecondView, string | undefined> = {
  sigmoid: 'Every point at its own z.',
  curves: undefined,
  confusion: 'Rows are the truth, columns the call.',
  roc: 'Every threshold at once; the dot is t.',
};

export default function LogisticRegressionPage() {
  const paramsApi = useExplainerParams<LogRegParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const generated = useMemo(
    () =>
      getClassificationDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed],
  );

  // A local copy, so clicking the chart edits the data without regenerating it.
  const [data, setData] = useEditable(generated);

  const featureMap = params.featureMap as FeatureMap;

  const config = useMemo<LogRegConfig>(
    () => ({
      learningRate: params.learningRate,
      featureMap,
      l2: params.l2,
      // "Full batch" stays every point, even after clicks add more than the knob's top.
      batchSize: params.batchSize >= FULL_BATCH ? Infinity : params.batchSize,
      threshold: params.threshold,
      standardise: params.standardise,
      seed: params.seed,
    }),
    [
      params.learningRate,
      featureMap,
      params.l2,
      params.batchSize,
      params.threshold,
      params.standardise,
      params.seed,
    ],
  );

  // Fitted to the generated points, so a clicked point does not rescale the space the weights live in.
  const std = useMemo(
    () => makeStandardiser(generated.points, params.standardise),
    [generated.points, params.standardise],
  );

  /* ---------------- simulation ---------------- */

  // Neither the threshold nor a hand-edited `data` is in `deps`; only regenerating resets.
  const sim = useSimulation({
    create: () => createState(config),
    step: (state) => gradientStep(state, data.points, config, std),
    isComplete: (state) => state.converged || state.diverged,
    deps: [
      generated,
      params.standardise,
      params.learningRate,
      featureMap,
      params.l2,
      params.batchSize,
      params.seed,
    ],
    baseStepsPerSecond: 18,
    maxSteps: MAX_STEPS,
  });

  const state = sim.state;

  /* ---------------- derived numbers ---------------- */

  const prediction = useMemo(
    () => predictAll(state.weights, data.points, config, std),
    [state.weights, data.points, config, std],
  );

  const actual = useMemo(() => data.points.map((p) => p.label), [data.points]);

  const accuracy = useMemo(
    () => accuracyOf(actual, prediction.labels),
    [actual, prediction.labels],
  );

  const loss = useMemo(
    () => logLoss(actual, prediction.probabilities),
    [actual, prediction.probabilities],
  );

  const prf = useMemo(
    () => precisionRecallF1(actual, prediction.labels, 1),
    [actual, prediction.labels],
  );

  const matrix = useMemo(
    () => confusionMatrix(actual, prediction.labels, 2),
    [actual, prediction.labels],
  );

  const auc = useMemo(
    () => rocAuc(actual, prediction.probabilities),
    [actual, prediction.probabilities],
  );

  const roc = useMemo(
    () => rocCurve(actual, prediction.probabilities),
    [actual, prediction.probabilities],
  );

  const operating = useMemo(() => {
    const negatives = prf.fp + prf.tn;
    const positives = prf.tp + prf.fn;
    return {
      fpr: negatives === 0 ? 0 : prf.fp / negatives,
      tpr: positives === 0 ? 0 : prf.tp / positives,
    };
  }, [prf]);

  // Majority-class accuracy.
  const positiveCount = useMemo(() => actual.filter((v) => v === 1).length, [actual]);
  const baseline =
    actual.length === 0
      ? 0
      : Math.max(positiveCount, actual.length - positiveCount) / actual.length;

  const misclassified = useMemo(() => {
    const out: number[] = [];
    for (let i = 0; i < actual.length; i++) {
      if (prediction.labels[i] !== actual[i]) out.push(i);
    }
    return out;
  }, [actual, prediction.labels]);

  const palette = usePalette();

  const headline: Metric[] = useMemo(() => {
    const history = state.lossHistory;
    const previous = history[history.length - 2];
    const latest = history[history.length - 1];
    const trend: Metric['trend'] =
      Number.isFinite(previous) && Number.isFinite(latest)
        ? latest < previous
          ? 'down'
          : latest > previous
            ? 'up'
            : 'flat'
        : 'flat';

    return [
      {
        key: 'loss',
        label: 'log loss',
        value: state.diverged ? '∞' : fmt(loss, 4),
        tone: state.diverged ? 'bad' : loss < 0.35 ? 'good' : loss < Math.LN2 ? 'neutral' : 'warn',
        trend,
        colour: palette.orange,
        help:
          'Average cross-entropy over every point. This is the number training is shrinking, and it cares how confident you were, not only whether you were right. A coin flip scores ' +
          fmt(Math.LN2, 2) +
          '.',
      },
      {
        key: 'accuracy',
        label: 'accuracy',
        value: fmtPercent(accuracy, 0),
        tone: accuracy > baseline + 0.04 ? 'good' : 'warn',
        colour: palette.green,
        help:
          'The fraction of points on the correct side of the threshold. Always guessing the majority class scores ' +
          fmtPercent(baseline, 0) +
          ', so on skewed data a model that has learned nothing still looks good.',
      },
    ];
  }, [state, loss, accuracy, baseline, palette.orange, palette.green]);

  const more: Metric[] = useMemo(() => {
    return [
      {
        key: 'precision',
        label: 'Precision',
        value: fmt(prf.precision, 3),
        caption: prf.tp + ' / ' + (prf.tp + prf.fp),
        tone: 'neutral',
        help: 'Of the points the model calls positive, the fraction that really are. Raising the threshold raises this and costs you recall.',
      },
      {
        key: 'recall',
        label: 'Recall',
        value: fmt(prf.recall, 3),
        caption: prf.tp + ' / ' + (prf.tp + prf.fn),
        tone: 'neutral',
        help: 'Of the points that really are positive, the fraction the model finds. Lowering the threshold raises this and costs you precision.',
      },
      {
        key: 'f1',
        label: 'F1',
        value: fmt(prf.f1, 3),
        tone: prf.f1 > 0.8 ? 'good' : prf.f1 > 0.5 ? 'neutral' : 'warn',
        help: 'The harmonic mean of precision and recall. Unlike an average it stays low when either one collapses, which is what you want from a single summary number.',
      },
      {
        key: 'auc',
        label: 'ROC AUC',
        value: fmt(auc, 3),
        tone: auc > 0.9 ? 'good' : auc > 0.7 ? 'neutral' : 'warn',
        help: 'The chance that a random positive is scored above a random negative. It judges the ranking the weights produce, so moving the threshold cannot change it.',
      },
    ];
  }, [prf, auc]);

  /* ---------------- editing the data by hand ---------------- */

  const frameRef = useRef<ReturnType<typeof makeFrame> | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selection, setSelection] = useState<ArchSelection | null>(null);
  // Edit count, read against a baseline taken on step entry.
  const [pointEdits, setPointEdits] = useState(0);
  const editsAtEntry = useRef(0);
  // A finished run refuses to play; an edit un-latches it, then resumes here.
  const [resume, setResume] = useState(false);
  useEffect(() => {
    if (!resume || sim.isComplete || sim.isRunning) return;
    setResume(false);
    sim.play();
  }, [resume, sim]);
  const noteEdit = useCallback(() => {
    setPointEdits((n) => n + 1);
    if (state.converged) {
      sim.patch((s) => ({ ...s, converged: false }));
      setResume(true);
    }
  }, [state.converged, sim]);

  /* ---------------- the guide ---------------- */

  // Only events reach the card.
  const guide = useMemo<GuideStatus>(() => {
    const history = state.lossHistory;
    const previous = history.length >= 2 ? history[history.length - 2] : loss;
    const alpha = fmt(config.learningRate, config.learningRate < 0.01 ? 4 : 2);
    if (state.diverged) {
      return {
        text:
          config.l2 > 0
            ? 'Diverged: α = ' + alpha + ' times the penalty 2λw flips the weights every step. Lower α or λ and replay.'
            : 'Diverged: α = ' + alpha + ' is too large. Lower it and replay.',
        tone: 'bad',
      };
    }
    if (state.converged) {
      return {
        text:
          'Converged at step ' + state.epoch.toLocaleString('en-US') +
          (accuracy < 0.9 ? '. The classes overlap, so no boundary of this shape does better.' : ''),
        tone: 'good',
      };
    }
    if (sim.iteration >= MAX_STEPS) {
      return { text: 'Stopped at ' + MAX_STEPS.toLocaleString('en-US') + ' steps. Reset to run again.', tone: 'warn' };
    }
    const perfect = state.accuracyHistory[state.accuracyHistory.length - 1] === 1;
    if (config.l2 === 0 && state.epoch >= 40 && perfect && state.loss < previous) {
      return { text: 'Every point is on the right side, so the weights just keep growing. A penalty stops that.', tone: 'warn' };
    }
    if (state.epoch >= 100 && accuracy <= baseline + 0.02) {
      return { text: 'No better than always calling the bigger class. This boundary shape cannot separate them.', tone: 'warn' };
    }
    return { text: '' };
  }, [state, config, loss, accuracy, baseline, sim.iteration]);

  const handleFrame = useCallback((frame: ReturnType<typeof makeFrame>) => {
    frameRef.current = frame;
  }, []);

  const handleDown = useCallback(
    (pos: { x: number; y: number }, event: { shiftKey: boolean; button: number }) => {
      const frame = frameRef.current;
      if (!frame) return;
      const x = frame.x.invert(pos.x);
      const y = frame.y.invert(pos.y);

      if (event.shiftKey) {
        if (nearestPointIndex(data.points, x, y, 0.9) < 0) return;
        setData((prev) => removeNearestPoint(prev, x, y, 0.9));
        setHoverIndex(null);
        noteEdit();
        return;
      }
      // Clicking an existing point does nothing.
      if (nearestPointIndex(data.points, x, y, 0.4) >= 0) return;
      // The left button adds the brush class, the right button the other one.
      const brush = Number(params.brushClass) === 1 ? 1 : 0;
      setData((prev) => addPoint(prev, x, y, event.button === 2 ? 1 - brush : brush));
      noteEdit();
    },
    [data.points, params.brushClass, noteEdit, setData],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const frame = frameRef.current;
      if (!pos || !frame) {
        setHoverIndex(null);
        return;
      }
      const index = nearestPointIndex(
        data.points,
        frame.x.invert(pos.x),
        frame.y.invert(pos.y),
        0.45,
      );
      setHoverIndex(index >= 0 ? index : null);
    },
    [data.points],
  );

  /* ---------------- render ---------------- */

  const brush = Number(params.brushClass) === 1 ? 1 : 0;
  const brushName = data.classNames[brush];
  const otherName = data.classNames[1 - brush];
  const [secondView, setSecondView] = useState<SecondView>('sigmoid');

  // The graph shows the classifier alone until there is training to show.
  const [graphView, setGraphView] = useState<GraphView | null>(null);
  const view: GraphView = graphView ?? (sim.isRunning || state.epoch > 0 ? 'training' : 'model');

  // The lessons drive the presets, the transport, the views and the open card.
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'graph') setGraphView(value === 'auto' ? null : (value as GraphView));
        else if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => setSelection(target === null ? null : parseArchTarget(target)),
    }),
    [],
  );
  const lessonContext = useMemo(
    () =>
      makeLogRegContext({
        params,
        state,
        data,
        config,
        std,
        ui: {
          view,
          secondView,
          open: selection ? selection.kind + ':' + selection.id : null,
          get pointEdits() {
            return pointEdits - editsAtEntry.current;
          },
        },
      }),
    [params, state, data, config, std, view, secondView, selection, pointEdits],
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
    setSecondView('sigmoid');
    setGraphView(null);
    setSelection(null);
  }, [resetLessons, reset, resetRun, generated, setData]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || data !== generated || secondView !== 'sigmoid' || graphView !== null || selection !== null;

  // A node or edge the step points at, drawn as a hover while no card is open.
  const focus = lessonApi.focus;
  const spot = useMemo<ArchSelection | null>(
    () => (focus && (focus.kind === 'node' || focus.kind === 'edge') && !selection ? { kind: focus.kind, id: focus.id } : null),
    [focus, selection],
  );

  // Every vector the diagram shows, one value per point, in a shared row order.
  const names = useMemo(() => featureNames(featureMap), [featureMap]);
  const vectors = useMemo<LogRegVectors>(() => {
    const rows = data.points.map((pt) => features(pt.x, pt.y, featureMap, std));
    const z = rows.map((row) => row.reduce((sum, f, i) => sum + f * (state.weights[i] ?? 0), 0));
    const label = data.points.map((pt) => (pt.label === 1 ? 1 : 0));
    return {
      x1: data.points.map((pt) => pt.x),
      x2: data.points.map((pt) => pt.y),
      label,
      features: names.map((_, i) => rows.map((row) => row[i] ?? 0)),
      z,
      p: prediction.probabilities,
      called: prediction.labels,
      loss: prediction.probabilities.map((p, i) =>
        -(label[i] * Math.log(Math.max(p, 1e-9)) + (1 - label[i]) * Math.log(Math.max(1 - p, 1e-9))),
      ),
    };
  }, [data.points, featureMap, std, state.weights, names, prediction]);
  const order = useMemo(
    () => displayOrder(data.points.length, (i) => data.points[i].x, (i) => data.points[i].label),
    [data.points],
  );
  const batch = useMemo(() => {
    if (state.epoch === 0 || config.batchSize >= data.points.length) return null;
    return new Set(selectBatch(config, data.points.length, state.epoch - 1));
  }, [config, data.points.length, state.epoch]);

  const graph = useMemo(
    () =>
      buildLogRegGraph({
        state,
        featureNames: names,
        threshold: params.threshold,
        accuracy,
        logLoss: loss,
        started: state.epoch > 0,
        standardised: std.stdX !== 1 || std.stdY !== 1,
        vectors,
        order,
        batch,
        view,
      }),
    [state, names, params.threshold, accuracy, loss, std, vectors, order, batch, view],
  );

  const renderDetail = useCallback(
    (active: ArchSelection, jump: (next: ArchSelection) => void) =>
      data.points.length === 0 ? null : (
        <LogRegDetail
          selection={active}
          state={state}
          config={config}
          featureNames={names}
          std={std}
          standardised={std.stdX !== 1 || std.stdY !== 1}
          vectors={vectors}
          order={order}
          batch={batch}
          accuracy={accuracy}
          logLoss={loss}
          jump={jump}
        />
      ),
    [state, config, names, std, vectors, order, batch, data.points.length, accuracy, loss],
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
          }
        >
          <ArchitectureView
            graph={graph}
            selection={selection}
            onSelect={setSelection}
            renderDetail={renderDetail}
            description={
              'Computation graph: ' +
              featureNames(featureMap).length +
              ' weighted inputs into a score z, a sigmoid into a probability p, a threshold into a class' +
              (view === 'training' ? ', then the cross-entropy loss and its gradient' : '') +
              '. Step ' +
              state.epoch +
              '.'
            }
            redrawKey={view + ':' + params.threshold + ':' + featureMap}
            phase={state.epoch}
            spot={spot}
          />
        </Panel>
      }
      output={
        <>
          <ColumnHead
            title="Output"
            blurb={
              <>
                <Tex tex={modelTex(state.weights, featureMap, std.stdX !== 1 || std.stdY !== 1)} />
              </>
            }
          />

          <Panel
            id="field"
            fill
            title="Probability across the plane"
            subtitle={'Click: ' + brushName + ' · right-click: ' + otherName + ' · shift-click removes'}
            actions={
              <Legend
                dense
                items={[
                  { label: data.classNames[0], colour: classColour(palette, 0), shape: 'dot' },
                  { label: data.classNames[1], colour: classColour(palette, 1), shape: 'cross' },
                  { label: 'p = t', colour: palette.text, shape: 'line' },
                ]}
              />
            }
          >
            <DecisionField
              data={data}
              state={state}
              config={config}
              std={std}
              probabilities={prediction.probabilities}
              misclassified={misclassified}
              showField={params.showField}
              showMisclassified={params.showMisclassified}
              hoverIndex={hoverIndex}
              onFrame={handleFrame}
              onPointerDown={handleDown}
              onPointerMove={handleMove}
            />
          </Panel>

          <Panel
            id="second"
            fill
            title={SECOND_VIEW_TITLES[secondView]}
            subtitle={SECOND_VIEW_BLURBS[secondView]}
            actions={
              <ViewSwitch
                id="second"
                label="Second chart"
                value={secondView}
                options={[
                  { value: 'sigmoid', label: 'Sigmoid' },
                  { value: 'curves', label: 'Loss' },
                  { value: 'confusion', label: 'Matrix' },
                  { value: 'roc', label: 'ROC' },
                ]}
                onChange={(next) => setSecondView(next as SecondView)}
              />
            }
          >
            {secondView === 'sigmoid' ? (
              <SigmoidCurve
                data={data}
                state={state}
                config={config}
                std={std}
                misclassified={misclassified}
                showProjection={params.showProjection}
              />
            ) : secondView === 'curves' ? (
              <TrainingCurves
                lossHistory={state.lossHistory}
                accuracyHistory={state.accuracyHistory}
                epoch={state.epoch}
                baseline={baseline}
                penalised={config.l2 > 0}
              />
            ) : secondView === 'confusion' ? (
              <ConfusionTable counts={matrix.counts} classNames={data.classNames} />
            ) : (
              <RocPlot curve={roc} auc={auc} operating={operating} threshold={params.threshold} />
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
/** Feature names in KaTeX, in the order the model's weights arrive. */
const TEX_FEATURES: Record<FeatureMap, string[]> = {
  linear: ['', 'x_1', 'x_2'],
  interaction: ['', 'x_1', 'x_2', 'x_1x_2'],
  quadratic: ['', 'x_1', 'x_2', 'x_1^2', 'x_2^2', 'x_1x_2'],
};

/** The live model as a readable formula, for the note under the main chart. */
function modelTex(weights: readonly number[], map: FeatureMap, standardised: boolean): string {
  const names = TEX_FEATURES[map].map((name) => (standardised ? name.replace(/x_/g, '\\tilde{x}_') : name));
  const body = weights
    .map((w, i) => {
      const value = fmt(Math.abs(w), 2);
      const term = names[i] ? value + '\\,' + names[i] : value;
      if (i === 0) return (w < 0 ? '-' : '') + term;
      return (w < 0 ? ' - ' : ' + ') + term;
    })
    .join('');
  return 'p = \\sigma\\left(' + body + '\\right)';
}
