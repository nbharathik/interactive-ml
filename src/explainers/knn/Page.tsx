/** k-NN explainer. The transport reveals one query's neighbours, nearest first. */

import { useCallback, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import {
  Legend,
  MetricStrip,
  Panel,
  ParamsReset,
  ViewSwitch,
} from '../../explainer/components/Panels';
import {
  Annotation,
  DetailHead,
  DetailOverlay,
  Scalar,
  Scalars,
} from '../../explainer/components/Detail';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { useElementSize } from '../../explainer/useElementSize';
import { Button } from '../../explainer/components/Controls';
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
  POINT_RANGE,
  removeNearestPoint,
} from '../../lib/datasets/points';
import type { PointData } from '../../lib/datasets/types';
import { fmt, fmtPercent } from '../../lib/math/stats';
import {
  accuracyAcrossK,
  classify,
  makeScaling,
  METRIC_LABELS,
  rankNeighbours,
  tally,
  trainingAccuracy,
} from '../../lib/ml/knn';
import type { DistanceMetric, KnnConfig, WeightingScheme } from '../../lib/ml/knn';
import { clamp } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { classColour, classShape } from '../../lib/viz/plots';
import type { LegendItem } from '../../explainer/components/Panels';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { KnnParams } from './config';
import { LESSONS, makeKnnContext } from './lessons';
import type { SecondView } from './lessons';
import { AccuracyVsK, NeighbourTable, NeighbourhoodPlot, VoteBars } from './panels';

const META = getExplainer('knn')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);

/** Matches the k stepper's ceiling in config.ts. */
const MAX_K = 60;

/** Marker shapes, translated into the legend's vocabulary. */
const LEGEND_SHAPE: Record<string, LegendItem['shape']> = {
  circle: 'dot',
  cross: 'cross',
  triangle: 'triangle',
  square: 'square',
};

export default function KnnPage() {
  const paramsApi = useExplainerParams<KnnParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, merge, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const generated = useMemo(
    () =>
      getClassificationDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
        classCount: params.classCount,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed, params.classCount],
  );

  // A local copy, so clicking the chart edits the stored set without regenerating it.
  const [data, setData] = useEditable(generated);

  const stretch = params.xStretch;

  // Stretching x simulates a feature measured in bigger units.
  const plotData = useMemo<PointData>(() => {
    if (stretch === 1) return data;
    return {
      ...data,
      points: data.points.map((p) => ({ ...p, x: p.x * stretch })),
      xRange: [data.xRange[0] * stretch, data.xRange[1] * stretch],
      xLabel: data.xLabel + ' × ' + stretch,
    };
  }, [data, stretch]);

  const classNames = useMemo(
    () =>
      Array.from(
        { length: plotData.classCount },
        (_, i) => plotData.classNames[i] ?? 'Class ' + String.fromCharCode(65 + i),
      ),
    [plotData],
  );

  /* ---------------- configuration ---------------- */

  const config = useMemo<KnnConfig>(
    () => ({
      k: Math.max(1, params.k),
      metric: params.metric as DistanceMetric,
      weighting: params.weighting as WeightingScheme,
      classCount: plotData.classCount,
      standardise: params.standardise,
    }),
    [params.k, params.metric, params.weighting, params.standardise, plotData.classCount],
  );

  const scaling = useMemo(
    () => makeScaling(plotData.points, params.standardise),
    [plotData.points, params.standardise],
  );

  const weighted = config.weighting === 'distance';

  /* ---------------- the query ---------------- */

  // Local while dragging, committed to params on release; new params replace it in the same render.
  const paramsQuery = useMemo(() => ({ x: params.queryX, y: params.queryY }), [params.queryX, params.queryY]);
  const [query, setQuery] = useEditable(paramsQuery);

  const queryPoint = useMemo(() => ({ x: query.x * stretch, y: query.y }), [query.x, query.y, stretch]);

  const ranked = useMemo(
    () => rankNeighbours(plotData.points, queryPoint.x, queryPoint.y, config, scaling),
    [plotData.points, queryPoint, config, scaling],
  );

  const countedK = Math.min(config.k, ranked.length);

  /* ---------------- simulation: reveal the neighbours ---------------- */

  const sim = useSimulation<{ revealed: number }>({
    create: () => ({ revealed: 0 }),
    step: (state) => ({ revealed: state.revealed + 1 }),
    isComplete: (state) => state.revealed >= countedK,
    deps: [ranked, countedK],
    baseStepsPerSecond: 2.2,
    maxSteps: MAX_K + 1,
  });

  // At rest the chart shows the finished answer; stepping replays it one at a time.
  const revealed = sim.state.revealed;
  const shown = revealed === 0 ? countedK : Math.min(revealed, countedK);
  const newest = revealed === 0 ? -1 : Math.min(revealed, countedK) - 1;

  const counted = useMemo(() => ranked.slice(0, shown), [ranked, shown]);
  const vote = useMemo(() => tally(counted, config.classCount), [counted, config.classCount]);
  const finalVote = useMemo(
    () => tally(ranked.slice(0, countedK), config.classCount),
    [ranked, countedK, config.classCount],
  );

  const radius = counted.length > 0 ? counted[counted.length - 1].distance : 0;
  const spread = ranked.length > 0 ? ranked[ranked.length - 1].distance : 1;

  /* ---------------- scoring ---------------- */

  const curveConfig = useMemo(
    () => ({
      metric: config.metric,
      weighting: config.weighting,
      classCount: config.classCount,
      standardise: config.standardise,
    }),
    [config],
  );

  // n rankings reused across every k; must not rerun on a query drag.
  const curve = useMemo(
    () => accuracyAcrossK(plotData.points, MAX_K, curveConfig, scaling),
    [plotData.points, curveConfig, scaling],
  );

  const looAccuracy = curve.length > 0 ? curve[Math.min(config.k, curve.length) - 1].accuracy : 0;

  const best = useMemo(
    () =>
      curve.reduce<{ k: number; accuracy: number }>(
        (bestSoFar, entry) => (entry.accuracy > bestSoFar.accuracy ? entry : bestSoFar),
        { k: 1, accuracy: 0 },
      ),
    [curve],
  );

  const trainAccuracy = useMemo(
    () => trainingAccuracy(plotData.points, config, scaling),
    [plotData.points, config, scaling],
  );

  const baseline = useMemo(() => {
    if (plotData.points.length === 0) return 0;
    const counts = new Array<number>(Math.max(1, plotData.classCount)).fill(0);
    for (const p of plotData.points) {
      if (p.label >= 0 && p.label < counts.length) counts[p.label] += 1;
    }
    return Math.max(...counts) / plotData.points.length;
  }, [plotData]);

  const classAt = useCallback(
    (x: number, y: number) => classify(plotData.points, x, y, config, scaling).winner,
    [plotData.points, config, scaling],
  );

  /* ---------------- metrics ---------------- */

  const margin = useMemo(() => {
    const sorted = [...vote.scores].sort((a, b) => b - a);
    const total = sorted.reduce((sum, v) => sum + v, 0);
    if (total <= 0) return 0;
    return (sorted[0] - (sorted[1] ?? 0)) / total;
  }, [vote]);

  const palette = usePalette();
  const winnerColour = counted.length > 0 && !vote.tied ? classColour(palette, vote.winner) : undefined;

  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'prediction',
        label: 'prediction',
        value: counted.length === 0 ? 'n/a' : vote.tied ? 'Tie' : classNames[vote.winner] ?? 'n/a',
        tone: vote.tied ? 'warn' : 'neutral',
        colour: winnerColour,
        help:
          'The class holding the most vote weight among the neighbours counted so far' +
          (counted.length > 0 ? ', with ' + fmtPercent(vote.confidence, 0) + ' of the vote' : '') +
          '. With uniform weighting this is the most common label among the k.',
      },
      {
        key: 'loo',
        label: 'leave-one-out',
        // Leaving one point out leaves n − 1 to vote, so a k past that has no honest score.
        value: curve.length === 0 || config.k > curve.length ? 'n/a' : fmtPercent(looAccuracy, 0),
        tone: looAccuracy >= 0.9 ? 'good' : looAccuracy >= 0.75 ? 'neutral' : 'warn',
        help: 'Each stored point classified using every point except itself: the honest number. This is the one to choose k by; it is the cheapest form of held-out evaluation there is.',
      },
    ],
    [counted.length, vote, classNames, winnerColour, curve.length, looAccuracy, config.k],
  );

  const more: Metric[] = useMemo(
    () => [
      {
        key: 'margin',
        label: 'Vote margin',
        value: counted.length === 0 ? 'n/a' : fmtPercent(margin, 0),
        caption: 'over runner-up',
        tone: margin > 0.4 ? 'good' : margin > 0.1 ? 'neutral' : 'warn',
        help: 'How far ahead the winner is, as a share of the total weight. Near zero means the query sits on a boundary and a small nudge would flip the answer.',
      },
      {
        key: 'radius',
        label: 'Neighbourhood radius',
        value: counted.length === 0 ? 'n/a' : fmt(radius, 2),
        caption: 'to neighbour #' + Math.max(1, shown),
        tone: radius > spread * 0.45 ? 'warn' : 'neutral',
        help: 'How far out you must reach to collect that many neighbours, measured in the metric currently selected. A large radius means the query is out where there is no data.',
      },
      {
        key: 'train',
        label: 'Training accuracy',
        value: fmtPercent(trainAccuracy, 1),
        caption: config.k === 1 ? 'always 100% at k=1' : 'on stored points',
        tone: config.k === 1 ? 'warn' : 'neutral',
        help: 'Every stored point classified using the full stored set. At k = 1 each point is its own nearest neighbour, so this reads 100% no matter how bad the model is.',
      },
      {
        key: 'best-k',
        label: 'Best k',
        value: curve.length === 0 ? 'n/a' : String(best.k),
        caption: curve.length === 0 ? 'no data' : 'peaks at ' + fmtPercent(best.accuracy, 1),
        tone: curve.length > 0 && Math.abs(best.k - config.k) <= 1 ? 'good' : 'neutral',
        help: 'The k with the highest leave-one-out accuracy on this exact data. Step k onto it and the orange marker in the accuracy chart lands on the green one.',
      },
    ],
    [counted.length, margin, radius, spread, shown, trainAccuracy, config.k, curve.length, best],
  );

  /* ---------------- pointer interaction ---------------- */

  const frameRef = useRef<Frame | null>(null);
  const [dragging, setDragging] = useState(false);
  const [secondView, setSecondView] = useState<SecondView>('table');
  const [queryOpen, setQueryOpen] = useState(false);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [queryHover, setQueryHover] = useState(false);
  const [setPlotFrame, plotSize] = useElementSize<HTMLDivElement>();
  const dragMovedRef = useRef(false);
  // Drags so far; a lesson step asks for one since it was entered, so the
  // context reads the count against a baseline taken on entry.
  const [queryMoves, setQueryMoves] = useState(0);
  const movesAtEntry = useRef(0);
  const [pointEdits, setPointEdits] = useState(0);
  const editsAtEntry = useRef(0);

  // The lessons drive the presets, the count, the second chart and the query card.
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => setQueryOpen(target === 'query'),
    }),
    [],
  );
  const lessonContext = useMemo(
    () =>
      makeKnnContext({
        params,
        state: sim.state,
        data: plotData,
        config,
        scaling,
        query: queryPoint,
        curve,
        ui: {
          secondView,
          queryOpen,
          get queryMoves() {
            return queryMoves - movesAtEntry.current;
          },
          get pointEdits() {
            return pointEdits - editsAtEntry.current;
          },
        },
      }),
    [params, sim.state, plotData, config, scaling, queryPoint, curve, secondView, queryOpen, queryMoves, pointEdits],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });
  // The baseline moves in the same render the step does, before its condition is checked.
  const lastStep = useRef(lessonApi.step);
  if (lastStep.current !== lessonApi.step) {
    lastStep.current = lessonApi.step;
    movesAtEntry.current = queryMoves;
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
    setSecondView('table');
    setQueryOpen(false);
  }, [resetLessons, reset, resetRun, generated, setData]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || data !== generated || secondView !== 'table' || queryOpen;

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    const name = (c: number) => classNames[c] ?? 'Class ' + (c + 1);
    const tallyText = (v: typeof vote) =>
      v.scores.map((score, c) => name(c) + ' ' + (weighted ? fmt(score, 2) : String(score))).join(' · ');
    const far = radius > spread * 0.45 ? ' · the query is out where there is no data' : '';
    // Only two classes and an even k can split evenly for certain; otherwise the fix is another k or another spot.
    const tieFix = classNames.length === 2 && countedK % 2 === 0 ? 'pick an odd k or drag the query' : 'change k or drag the query';
    if (ranked.length === 0) {
      return { text: 'No stored points, so there is nothing to vote. Click the chart to add some.', tone: 'warn' };
    }
    if (revealed === 0) {
      if (finalVote.tied) {
        return {
          text:
            'All ' + countedK + ' neighbours voted and the vote is tied: ' + tallyText(finalVote) +
            '. A tie has no winner; ' + tieFix + '.' + far,
          tone: 'warn',
        };
      }
      return {
        text:
          'All ' + countedK + ' neighbours have voted: ' + name(finalVote.winner) + ' wins ' +
          fmtPercent(finalVote.confidence, 0) + ' of the vote. Press play to replay the count nearest-first, or drag the query point.' + far,
        tone: far ? 'warn' : 'neutral',
      };
    }
    if (sim.isComplete) {
      if (vote.tied) {
        return {
          text:
            'All ' + countedK + ' counted · tied: ' + tallyText(vote) +
            ' · implementations break a tie arbitrarily; ' + tieFix + '.' + far,
          tone: 'warn',
        };
      }
      return {
        text:
          'All ' + countedK + ' counted · ' + name(vote.winner) + ' wins with ' + fmtPercent(vote.confidence, 0) +
          ' (margin ' + fmtPercent(margin, 0) + ') · radius ' + fmt(radius, 2) + ' to the furthest neighbour' + far,
        tone: far ? 'warn' : 'good',
      };
    }
    const latest = counted[counted.length - 1];
    let text = 'Neighbour ' + shown + ' of ' + countedK;
    if (latest) {
      text += ' · a ' + name(latest.point.label) + ' point at distance ' + fmt(latest.distance, 2) +
        (weighted ? ', worth ' + fmt(latest.weight, 2) : '');
    }
    text += ' · tally: ' + tallyText(vote) + (vote.tied ? ' · tied so far' : ' · ' + name(vote.winner) + ' leads');
    return { text, tone: vote.tied ? 'warn' : 'neutral' };
  }, [classNames, weighted, radius, spread, ranked.length, revealed, finalVote, countedK, sim.isComplete, vote, margin, counted, shown]);

  const handleFrame = useCallback((frame: Frame) => {
    frameRef.current = frame;
  }, []);

  const handleDown = useCallback(
    (pos: { x: number; y: number }, event: ReactPointerEvent<HTMLCanvasElement>) => {
      const frame = frameRef.current;
      if (!frame || event.button !== 0) return;
      const qx = frame.x(query.x * stretch);
      const qy = frame.y(query.y);
      if ((qx - pos.x) ** 2 + (qy - pos.y) ** 2 <= 20 * 20) {
        dragMovedRef.current = false;
        setDragging(true);
        return;
      }
      const bx = frame.x.invert(pos.x) / stretch;
      const by = frame.y.invert(pos.y);
      if (event.shiftKey) {
        if (nearestPointIndex(data.points, bx, by, 0.8) < 0) return;
        setData((prev) => removeNearestPoint(prev, bx, by, 0.8));
        setPointEdits((n) => n + 1);
        return;
      }
      if (params.clickAdds) {
        setData((prev) => addPoint(prev, bx, by, Number(params.paintClass)));
        setPointEdits((n) => n + 1);
      }
    },
    [query.x, query.y, stretch, params.clickAdds, params.paintClass, data.points, setData],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const frame = frameRef.current;
      if (!frame) return;
      if (!dragging) {
        if (!pos) {
          setHoverIndex(null);
          setQueryHover(false);
          return;
        }
        const qx = frame.x(query.x * stretch);
        const qy = frame.y(query.y);
        setQueryHover(Math.hypot(qx - pos.x, qy - pos.y) <= 20);
        // In pixels, so a stretched x axis does not shrink the hover target to nothing.
        let index = -1;
        let nearest = 12 * 12;
        plotData.points.forEach((p, i) => {
          const d = (frame.x(p.x) - pos.x) ** 2 + (frame.y(p.y) - pos.y) ** 2;
          if (d < nearest) {
            nearest = d;
            index = i;
          }
        });
        setHoverIndex(index >= 0 ? index : null);
        return;
      }
      if (!pos) return;
      dragMovedRef.current = true;
      setQuery({
        x: clamp(frame.x.invert(pos.x) / stretch, POINT_RANGE[0], POINT_RANGE[1]),
        y: clamp(frame.y.invert(pos.y), POINT_RANGE[0], POINT_RANGE[1]),
      });
    },
    [dragging, stretch, plotData.points, query.x, query.y, setQuery],
  );

  const handleUp = useCallback(() => {
    if (!dragging) return;
    setDragging(false);
    // A click on the query that did not drag opens its card.
    if (!dragMovedRef.current) setQueryOpen((v) => !v);
    else setQueryMoves((n) => n + 1);
    merge({
      queryX: Math.round(query.x * 10) / 10,
      queryY: Math.round(query.y * 10) / 10,
    });
  }, [dragging, query.x, query.y, merge]);

  // The query's spot on the plot, read each render since the frame follows resizes.
  const queryFrame = frameRef.current;
  const queryAnchor =
    queryFrame && plotSize.width > 0
      ? { x: queryFrame.x(queryPoint.x) - 12, y: queryFrame.y(queryPoint.y) - 12, w: 24, h: 24 }
      : null;
  const closeQuery = useCallback(() => setQueryOpen(false), []);

  // The query is the one thing the keyboard can open.
  const queryTargets = useMemo(() => ['query'], []);
  const handlePlotKey = useDiagramKeys<string>({
    targets: queryTargets,
    cursor: queryOpen ? 'query' : null,
    same: (a, b) => a === b,
    onCursor: () => setQueryOpen(true),
    onOpen: () => setQueryOpen(true),
    onClose: closeQuery,
  });

  /* ---------------- render ---------------- */

  // Cache key for the regions: everything except the query.
  const dataSignature = useMemo(() => {
    let hash = 17;
    for (const p of data.points) {
      hash = (hash * 31 + Math.round(p.x * 100)) | 0;
      hash = (hash * 31 + Math.round(p.y * 100)) | 0;
      hash = (hash * 31 + p.label) | 0;
    }
    return data.points.length + ':' + hash;
  }, [data.points]);

  const fieldKey = [
    dataSignature,
    config.k,
    config.metric,
    config.weighting,
    config.standardise ? 'std' : 'raw',
    stretch,
    plotData.classCount,
  ].join('|');

  const legendItems: LegendItem[] = classNames.map((name, index) => ({
    label: name,
    colour: classColour(palette, index),
    shape: LEGEND_SHAPE[classShape(index)] ?? 'dot',
  }));

  const paintName = classNames[Number(params.paintClass)] ?? 'Class C';

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
          unit="neighbour"
          total={countedK}
          completeLabel={'All ' + countedK + ' counted'}
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
          title="The neighbourhood"
          actions={
            <>
              <Button size="sm" variant="secondary" onClick={() => setQueryOpen((v) => !v)}>
                {queryOpen ? 'Hide query' : 'Query point'}
              </Button>
              {data !== generated ? (
                <Button size="sm" variant="ghost" onClick={() => setData(generated)}>
                  Restore points
                </Button>
              ) : null}
              <Legend
                dense
                items={[
                  ...legendItems,
                  { label: 'neighbourhood', colour: palette.text, shape: 'dashed' as const },
                ]}
              />
            </>
          }
        >
          <div
            className="mlx-arch"
            ref={setPlotFrame}
            tabIndex={0}
            role="group"
            aria-label="The neighbourhood. Enter opens the query point's card."
            onKeyDown={handlePlotKey}
          >
            <NeighbourhoodPlot
              data={plotData}
              query={queryPoint}
              neighbours={ranked}
              shown={shown}
              newest={newest}
              vote={vote}
              classNames={classNames}
              metric={config.metric}
              scaling={scaling}
              weighted={weighted}
              classAt={classAt}
              fieldKey={fieldKey}
              showRegions={params.showRegions}
              showLinks={params.showLinks}
              showBall={params.showBall}
              dragging={dragging}
              hoverIndex={hoverIndex}
              queryHover={queryHover}
              cardOpen={queryOpen && !dragging}
              onFrame={handleFrame}
              onPointerDown={handleDown}
              onPointerMove={handleMove}
              onPointerUp={handleUp}
              fill
            />
            {queryOpen && queryAnchor && !dragging ? (
              <DetailOverlay anchor={queryAnchor} bounds={plotSize} onClose={closeQuery}>
                <DetailHead
                  eyebrow="Query"
                  title="The point being classified"
                  shape={'measured against all ' + plotData.points.length + ' stored points · k = ' + countedK}
                />
                <Annotation>
                  There is no model to open: the stored points are the model. Every prediction ranks them by distance and lets the nearest k vote.
                </Annotation>
                <Scalars>
                  <Scalar label="x₁" value={queryPoint.x} decimals={3} tone="accent" />
                  <Scalar label="x₂" value={queryPoint.y} decimals={3} tone="accent" />
                  <Scalar label="metric" value={METRIC_LABELS[config.metric]} />
                  <Scalar label="weighting" value={weighted ? 'by 1/d²' : 'uniform'} />
                  <Scalar
                    label="prediction"
                    value={vote.tied ? 'tied' : (classNames[vote.winner] ?? String(vote.winner))}
                    tone={vote.tied ? 'bad' : 'good'}
                  />
                  <Scalar label="confidence" value={fmtPercent(vote.confidence, 0)} />
                  <Scalar label="counted" value={shown + ' / ' + countedK} />
                  <Scalar
                    label="radius reached"
                    value={ranked[Math.max(0, shown - 1)] ? fmt(ranked[shown - 1].distance, 3) : 'n/a'}
                  />
                </Scalars>
              </DetailOverlay>
            ) : null}
          </div>
        </Panel>
      }
      output={
        <>
          <ColumnHead
            title="Output"
            blurb={
              plotData.points.length +
              ' stored points · ' +
              (params.clickAdds ? 'click adds a point to ' + paintName : 'shift-click deletes')
            }
          />

          <Panel
            id="vote"
            title="The vote"
            subtitle={
              weighted
                ? 'Each neighbour contributes 1/d². The dashed outline is the final total.'
                : 'One vote each. The dashed outline is the final total.'
            }
          >
            <VoteBars
              classNames={classNames}
              scores={vote.scores}
              finalScores={finalVote.scores}
              winner={vote.winner}
              tied={vote.tied}
              weighted={weighted}
              shown={shown}
            />
          </Panel>

          <Panel
            id="second"
            fill
            title={secondView === 'table' ? 'The k nearest' : 'Which k?'}
            subtitle={
              secondView === 'table'
                ? 'Ranked by distance, measured as ' + METRIC_LABELS[config.metric] + '.'
                : 'Leave-one-out accuracy for every k on this exact data.'
            }
            actions={
              <ViewSwitch
                id="second"
                label="Second chart"
                value={secondView}
                options={[
                  { value: 'table', label: 'Neighbours' },
                  { value: 'k', label: 'Which k' },
                ]}
                onChange={(next) => setSecondView(next as SecondView)}
              />
            }
          >
            {secondView === 'table' ? (
              <NeighbourTable
                neighbours={ranked}
                k={config.k}
                shown={shown}
                classNames={classNames}
                weighted={weighted}
              />
            ) : (
              <AccuracyVsK curve={curve} currentK={config.k} bestK={best.k} baseline={baseline} />
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
