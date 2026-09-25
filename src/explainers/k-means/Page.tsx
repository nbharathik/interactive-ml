/** K-means explainer. One transport step is half an iteration: assign, then update. */

import { useCallback, useMemo, useRef, useState } from 'react';

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
import { addPoint, getClusteringDataset, POINT_RANGE } from '../../lib/datasets/points';
import { fmt, fmtCompact, fmtPercent } from '../../lib/math/stats';
import {
  clusterSizes,
  computeInertia,
  createState,
  elbowCurve,
  isComplete as kmeansIsComplete,
  silhouetteScore,
  step as kmeansStep,
} from '../../lib/ml/kmeans';
import type { InitMethod, KMeansConfig, KMeansState } from '../../lib/ml/kmeans';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { clamp } from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { KMeansParams } from './config';
import { LESSONS, makeKMeansContext } from './lessons';
import type { SecondView } from './lessons';
import { ClusterPlot, ClusterSizeBars, ElbowChart, InertiaCurve } from './panels';

const META = getExplainer('k-means')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);

/** How many k values the elbow sweep covers. */
const MAX_K = 10;

const INIT_LABELS: Record<string, string> = {
  'kmeans++': 'k-means++',
  'random-points': 'random data points',
  'random-coords': 'random coordinates',
  'forgy-far': 'the all-in-one-corner seeding',
};

export default function KMeansPage() {
  const paramsApi = useExplainerParams<KMeansParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const generated = useMemo(
    () =>
      getClusteringDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
        classCount: params.trueClusters,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed, params.trueClusters],
  );

  // A local copy so clicking the chart can add points without regenerating.
  const [data, setData] = useEditable(generated);
  const addedCount = data.points.length - generated.points.length;

  const config = useMemo<KMeansConfig>(
    () => ({
      k: params.k,
      init: params.init as InitMethod,
      seed: params.seed,
      tolerance: params.tolerance,
    }),
    [params.k, params.init, params.seed, params.tolerance],
  );

  const bounds = useMemo(
    () => ({ xRange: data.xRange, yRange: data.yRange }),
    [data.xRange, data.yRange],
  );

  /* ---------------- simulation ---------------- */

  const sim = useSimulation<KMeansState>({
    create: () => createState(data.points, config, bounds),
    // One step is one half-iteration: assign, then update, then assign again.
    step: (state) => kmeansStep(state, data.points),
    // The shift test makes the tolerance knob visible on screen.
    isComplete: (state) =>
      kmeansIsComplete(state, config.tolerance) ||
      (state.phase === 'update' && state.iteration > 0 && state.shift < config.tolerance),
    deps: [data.points, config.k, config.init, config.seed, bounds],
    baseStepsPerSecond: 2.4,
    maxSteps: 400,
  });

  const state = sim.state;
  const k = state.centroids.length;

  /* ---------------- derived numbers ---------------- */

  const sizes = useMemo(() => clusterSizes(state.assignments, k), [state.assignments, k]);

  const silhouette = useMemo(
    () => silhouetteScore(data.points, state.assignments, k),
    [data.points, state.assignments, k],
  );

  // The k sweep is memoised on the data and the seeding only.
  const elbow = useMemo(() => {
    if (data.points.length < MAX_K + 2) return [];
    return elbowCurve(
      data.points,
      MAX_K,
      { init: params.init as InitMethod, seed: params.seed, tolerance: 1e-4 },
      bounds,
      3,
    );
  }, [data.points, params.init, params.seed, bounds]);

  const baseline = elbow.length > 0 ? elbow[0].inertia : null;

  const palette = usePalette();

  const headline: Metric[] = useMemo(() => {
    const history = state.inertiaHistory;
    const previous = history[history.length - 2];
    const trend: Metric['trend'] =
      history.length >= 2 && Number.isFinite(previous)
        ? state.inertia < previous
          ? 'down'
          : state.inertia > previous
            ? 'up'
            : 'flat'
        : 'flat';

    return [
      {
        key: 'inertia',
        label: 'inertia',
        value: state.phase === 'seeded' ? 'n/a' : fmtCompact(state.inertia),
        tone: 'neutral',
        trend,
        colour: palette.orange,
        help:
          'Total squared distance from every point to its own centroid. This is the only number k-means is trying to make small, and neither move can ever increase it' +
          (baseline !== null && baseline > 0 && state.phase !== 'seeded'
            ? '. Now ' + fmtPercent(1 - state.inertia / baseline, 0) + ' below k = 1.'
            : '.'),
      },
      {
        key: 'silhouette',
        label: 'silhouette',
        value: Number.isFinite(silhouette) ? fmt(silhouette, 2) : 'n/a',
        tone: silhouette > 0.5 ? 'good' : silhouette > 0.25 ? 'neutral' : 'warn',
        help: 'Per point: how much closer it sits to its own cluster than to the next-nearest one, averaged over the data. Higher is better, and unlike inertia it does not automatically improve as k grows.',
      },
    ];
  }, [state, silhouette, baseline, palette.orange]);

  const more: Metric[] = useMemo(() => {
    const smallest = sizes.length > 0 ? Math.min(...sizes) : 0;

    return [
      {
        key: 'changed',
        label: 'Points reassigned',
        value: state.phase === 'seeded' ? 'n/a' : String(state.changedCount),
        caption: state.changedCount === 0 && state.phase !== 'seeded' ? 'nothing to move' : 'last assignment',
        tone: state.changedCount === 0 && state.phase !== 'seeded' ? 'good' : 'neutral',
        help: 'How many points switched cluster in the most recent assignment. When this hits zero the run is finished; no future move can change anything.',
      },
      {
        key: 'shift',
        label: 'Centroid shift',
        value: Number.isFinite(state.shift) ? fmtCompact(state.shift) : 'n/a',
        caption: 'largest move',
        tone: Number.isFinite(state.shift) && state.shift < params.tolerance ? 'good' : 'neutral',
        help: 'The distance travelled by the centroid that moved most in the last update. The run stops once this drops below the convergence tolerance.',
      },
      {
        key: 'rounds',
        label: 'Full rounds',
        value: String(state.iteration),
        caption: sim.iteration + ' half-steps',
        help: 'One round is an assignment followed by an update. K-means usually needs single digits of these, which is why it feels instant next to gradient descent.',
      },
      {
        key: 'smallest',
        label: 'Smallest cluster',
        value: state.phase === 'seeded' ? 'n/a' : String(smallest),
        caption:
          state.emptyClusters.length > 0
            ? state.emptyClusters.length + ' cluster(s) empty'
            : 'points',
        tone: smallest === 0 && state.phase !== 'seeded' ? 'bad' : 'neutral',
        help: 'The size of the thinnest cluster. Zero means a centroid caught nothing at all: it has no mean to move to, so it stays stranded where it was seeded.',
      },
    ];
  }, [state, sizes, params.tolerance, sim.iteration]);

  /* ---------------- pointer interaction ---------------- */

  const frameRef = useRef<Frame | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [secondView, setSecondView] = useState<SecondView>('sizes');
  const [setPlotFrame, plotSize] = useElementSize<HTMLDivElement>();

  // The lessons drive the presets, the transport, the second chart and the open card.
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => {
        if (target === null) setSelectedCluster(null);
        else if (target.startsWith('cluster:')) setSelectedCluster(Number(target.slice(8)));
      },
    }),
    [],
  );
  const lessonContext = useMemo(
    () => makeKMeansContext({ params, state, data, config, bounds, ui: { secondView, open: selectedCluster } }),
    [params, state, data, config, bounds, secondView, selectedCluster],
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
    setData(generated);
    setSecondView('sizes');
    setSelectedCluster(null);
  }, [resetLessons, reset, resetRun, generated, setData]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || data !== generated || secondView !== 'sizes' || selectedCluster !== null;

  /* ---------------- the guide ---------------- */

  const guide = useMemo<GuideStatus>(() => {
    const n = data.points.length;
    const seeding = INIT_LABELS[params.init] ?? params.init;
    const empty = state.emptyClusters.length > 0;
    const emptyNote = empty
      ? ' · cluster ' + state.emptyClusters.map((c) => c + 1).join(', ') + ' caught nothing and stays where it was seeded'
      : '';
    const sizesText = sizes.length > 0 ? ' · sizes ' + sizes.join(' / ') : '';
    const inertiaText =
      ' · inertia ' + fmtCompact(state.inertia) +
      (baseline !== null && baseline > 0 ? ', ' + fmtPercent(1 - state.inertia / baseline, 0) + ' below k = 1' : '');
    if (state.phase === 'seeded') {
      return {
        text:
          'Seeded ' + k + ' centroid' + (k === 1 ? '' : 's') + ' by ' + seeding +
          '. Press play: the first move gives every point to its nearest centroid. Click a centroid for its numbers, or drag it.',
      };
    }
    if (sim.isComplete) {
      const rounds = state.iteration + ' round' + (state.iteration === 1 ? '' : 's') + ' (' + sim.iteration + ' half-steps)';
      return {
        text: state.converged
          ? 'Converged after ' + rounds + ': no point changed cluster' + inertiaText + sizesText + emptyNote
          : 'Stopped after ' + rounds + ': the furthest centroid moved ' + fmt(state.shift, 3) + ', under the ' +
            fmt(params.tolerance, 3) + ' tolerance' + inertiaText + sizesText + emptyNote,
        tone: empty ? 'warn' : 'good',
      };
    }
    if (!Number.isFinite(state.shift)) {
      return {
        text: 'You placed that centroid by hand · memberships are unchanged until the next move re-checks every point.' + emptyNote,
        tone: empty ? 'warn' : 'neutral',
      };
    }
    const round = state.iteration + 1;
    if (state.phase === 'assign') {
      const moved = state.iteration === 0
        ? 'every point joined its nearest centroid'
        : state.changedCount + ' of ' + n + ' points switched cluster';
      return {
        text: 'Round ' + round + ', assignment · ' + moved + ' · inertia ' + fmtCompact(state.inertia) +
          ' · next: each centroid jumps to the mean of its points' + emptyNote,
        tone: empty ? 'warn' : 'neutral',
      };
    }
    return {
      text: 'Round ' + state.iteration + ', update · furthest centroid moved ' + fmt(state.shift, 2) +
        ' · inertia ' + fmtCompact(state.inertia) + ' · next: re-check which centroid each point is nearest' + emptyNote,
      tone: empty ? 'warn' : 'neutral',
    };
  }, [state, data.points.length, params.init, params.tolerance, k, sizes, baseline, sim.isComplete, sim.iteration]);
  const dragMovedRef = useRef(false);

  const handleFrame = useCallback((frame: Frame) => {
    frameRef.current = frame;
  }, []);

  /** Index of the centroid under the pointer, or -1. */
  const centroidAt = useCallback(
    (pos: { x: number; y: number }) => {
      const frame = frameRef.current;
      if (!frame) return -1;
      let best = -1;
      let bestDist = 17 * 17;
      state.centroids.forEach((c, i) => {
        const dx = frame.x(c.x) - pos.x;
        const dy = frame.y(c.y) - pos.y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      return best;
    },
    [state.centroids],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      dragMovedRef.current = false;
      const index = centroidAt(pos);
      if (index >= 0) {
        sim.pause();
        setDragIndex(index);
      }
    },
    [centroidAt, sim],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const frame = frameRef.current;
      if (!pos || !frame) {
        setHoverIndex(null);
        return;
      }
      if (dragIndex === null) {
        const index = centroidAt(pos);
        setHoverIndex(index >= 0 ? index : null);
        return;
      }
      dragMovedRef.current = true;
      const x = clamp(frame.x.invert(pos.x), POINT_RANGE[0], POINT_RANGE[1]);
      const y = clamp(frame.y.invert(pos.y), POINT_RANGE[0], POINT_RANGE[1]);
      sim.patch((s) => {
        const centroids = s.centroids.map((c, i) => (i === dragIndex ? { x, y } : c));
        // Before the first assignment a dragged seed is still a seed; nothing has been measured yet.
        if (s.phase === 'seeded') {
          return {
            ...s,
            centroids,
            previousCentroids: centroids.map((c) => ({ ...c })),
            trails: centroids.map((c) => [{ ...c }]),
          };
        }
        return {
          ...s,
          centroids,
          previousCentroids: centroids.map((c) => ({ ...c })),
          // A hand-placed centroid counts as an update, so the next move is an assignment.
          phase: 'update',
          converged: false,
          changedCount: 0,
          shift: Number.POSITIVE_INFINITY,
          inertia: computeInertia(data.points, s.assignments, centroids),
          emptyClusters: [],
        };
      });
    },
    [centroidAt, data.points, dragIndex, sim],
  );

  const handleUp = useCallback(
    (pos: { x: number; y: number }) => {
      if (dragIndex !== null) {
        const index = dragIndex;
        setDragIndex(null);
        // A click without a drag toggles the centroid's card.
        if (!dragMovedRef.current) setSelectedCluster((prev) => (prev === index ? null : index));
        if (dragMovedRef.current) {
          sim.patch((s) => ({
            ...s,
            trails: s.trails.map((t, i) => (i === index ? t.concat({ ...s.centroids[i] }) : t)),
          }));
        }
        return;
      }
      const frame = frameRef.current;
      if (!frame) return;
      if (pos.x < frame.left || pos.x > frame.right || pos.y < frame.top || pos.y > frame.bottom) {
        return;
      }
      // A click on empty space adds a point and resets the run.
      setData((prev) => addPoint(prev, frame.x.invert(pos.x), frame.y.invert(pos.y), -1));
    },
    [dragIndex, sim, setData],
  );

  /* ---------------- render ---------------- */

  // The open centroid's spot on the plot, read each render since the frame follows resizes.
  const selected = selectedCluster !== null ? state.centroids[selectedCluster] : undefined;
  const plotFrame = frameRef.current;
  const cardAnchor =
    selected && plotFrame && plotSize.width > 0
      ? { x: plotFrame.x(selected.x) - 12, y: plotFrame.y(selected.y) - 12, w: 24, h: 24 }
      : null;
  const closeCard = useCallback(() => setSelectedCluster(null), []);

  // Arrows walk the centroids, Enter opens one.
  const centroidTargets = useMemo(() => state.centroids.map((_, i) => i), [state.centroids]);
  const handlePlotKey = useDiagramKeys<number>({
    targets: centroidTargets,
    cursor: hoverIndex ?? selectedCluster,
    same: (a, b) => a === b,
    onCursor: setHoverIndex,
    onOpen: (index) => setSelectedCluster(index),
    onClose: () => {
      setSelectedCluster(null);
      setHoverIndex(null);
    },
  });

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
    <Studio
      meta={META}
      shareUrl={shareUrl}
      lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
      transport={
        <Transport sim={sim} unit="half-step" completeLabel="Converged" />
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
          title="The clustering"
          actions={
            <>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setData(generated)}
                softDisabled={addedCount === 0}
              >
                Restore points
              </Button>
              <Legend
                dense
                items={[
                  { label: 'centroid', colour: palette.text, shape: 'target' },
                  ...(params.showVoronoi
                    ? [{ label: 'region', colour: palette.muted, shape: 'square' as const }]
                    : []),
                  ...(params.showTrails
                    ? [{ label: 'trail', colour: palette.muted, shape: 'dashed' as const }]
                    : []),
                  ...(params.showTruth
                    ? [{ label: 'true group', colour: palette.muted, shape: 'ring' as const }]
                    : []),
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
            aria-label="The clustering. Arrow keys move between centroids, Enter opens one."
            onKeyDown={handlePlotKey}
            onBlur={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHoverIndex(null);
            }}
          >
            <ClusterPlot
              data={data}
              state={state}
              showVoronoi={params.showVoronoi}
              showTrails={params.showTrails}
              showLinks={params.showLinks}
              showTruth={params.showTruth}
              dragIndex={dragIndex}
              hoverIndex={hoverIndex}
              sizes={sizes}
              cardIndex={selectedCluster}
              onFrame={handleFrame}
              onPointerDown={handleDown}
              onPointerMove={handleMove}
              onPointerUp={handleUp}
              fill
            />
            {selected && selectedCluster !== null && cardAnchor && dragIndex === null ? (
              <DetailOverlay anchor={cardAnchor} bounds={plotSize} onClose={closeCard}>
                <DetailHead
                  eyebrow={'Cluster ' + (selectedCluster + 1)}
                  title={'Centroid ' + (selectedCluster + 1)}
                  shape={
                    (sizes[selectedCluster] ?? 0) + ' of ' + data.points.length + ' points · ' +
                    fmtPercent((sizes[selectedCluster] ?? 0) / Math.max(1, data.points.length), 0) + ' of the data'
                  }
                />
                <Annotation>
                  {state.emptyClusters.includes(selectedCluster)
                    ? 'Empty: nothing was assigned to it, so it has no mean to move to. Re-seed with k-means++ or lower k.'
                    : 'Not a data point: the mean of everything assigned to it, which is why an update can slide it into empty space.'}
                </Annotation>
                <Scalars>
                  <Scalar label="x" value={selected.x} decimals={3} tone="accent" />
                  <Scalar label="y" value={selected.y} decimals={3} tone="accent" />
                  <Scalar
                    label="moved last round"
                    value={
                      state.previousCentroids[selectedCluster]
                        ? fmt(
                            Math.hypot(
                              selected.x - state.previousCentroids[selectedCluster].x,
                              selected.y - state.previousCentroids[selectedCluster].y,
                            ),
                            3,
                          )
                        : 'n/a'
                    }
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
              data.points.length +
              ' points' +
              (addedCount > 0 ? ', ' + addedCount + ' added by hand' : '')
            }
          />

          <Panel
            id="inertia"
            fill
            title="Inertia over rounds"
            subtitle="It can only fall. The dashed marker is the drop the last assignment made."
          >
            <InertiaCurve
              history={state.inertiaHistory}
              current={state.inertia}
              phase={state.phase}
              iteration={state.iteration}
            />
          </Panel>

          <Panel
            id="second"
            fill
            title={secondView === 'sizes' ? 'Points per cluster' : 'Choosing k'}
            subtitle={
              secondView === 'sizes'
                ? 'A cluster with no points has no mean to move to.'
                : 'Every k from 1 to 10, best of three restarts. Click to change k.'
            }
            actions={
              <ViewSwitch
                id="second"
                label="Second chart"
                value={secondView}
                options={[
                  { value: 'sizes', label: 'Sizes' },
                  { value: 'elbow', label: 'Choosing k' },
                ]}
                onChange={(next) => setSecondView(next as SecondView)}
              />
            }
          >
            {secondView === 'sizes' ? (
              <ClusterSizeBars
                sizes={sizes}
                emptyClusters={state.emptyClusters}
                total={data.points.length}
              />
            ) : (
              <ElbowChart curve={elbow} currentK={params.k} onSelectK={(value) => set('k', value)} />
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
