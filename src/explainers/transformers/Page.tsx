/** Attention and transformers: a tiny transformer trains live on a sequence task, and one sequence is followed through every part of it. */

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
import { readFlag, writeFlag } from '../../explainer/storage';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useLesson } from '../../explainer/useLesson';
import { useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { usePalette } from '../../components/ThemeProvider';
import { spellTokens } from '../../lib/datasets/sequences';
import { fmtCompact, fmtKnob, fmtPercent } from '../../lib/math/stats';
import { forward, headWidth, parameterCount, positionTable, step as trainStep } from '../../lib/ml/transformer';
import type { TransformerSpec, TransformerWeights } from '../../lib/ml/transformer';
import { ArchitectureView, partOf } from './architecture';
import type { Part } from './architecture';
import { AttentionGrid } from './attention';
import type { HeadPick } from './attention';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS, digitsFor, layersFor, maskFor } from './config';
import type { TfParams } from './config';
import { TfDetail } from './detail';
import { headColour } from './draw';
import { FlowView } from './flow';
import { HeadView } from './head';
import { LESSONS, isFitted, makeTfContext, resolveHead } from './lessons';
import type { ArchView, SecondView } from './lessons';
import { META } from './meta';
import { buildData, buildSpec, embeddingPlane, honestCalls, probeOf, startState, writeFrom } from './model';
import { AnswerView, Curves, EmbeddingPlane, Gallery, PositionMap } from './panels';
import { ProbeStepper, TokenInput } from './probe';
import './transformers.css';

const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);
const MAX_STEPS = 3000;
const DEFAULT_PICK: HeadPick = { layer: 0, head: -1, query: 1 };
const NUMBERS_KEY = 'mlx-tf-numbers';
/** Milliseconds between the letters the model writes. */
const WRITE_BEAT = 420;

function archViewOf(value: string): ArchView {
  return value === 'flow' || value === 'attention' || value === 'head' ? value : 'model';
}
const pc = (v: number) => fmtPercent(v, 0);

const SECOND_TITLES: Record<SecondView, string> = {
  loss: 'Loss over steps',
  accuracy: 'Accuracy over steps',
  embeddings: 'Token embeddings',
  positions: 'Position vectors',
};

/** The model in one line: blocks, heads, width, the feed-forward size and the parameter count. */
function architectureLabel(spec: TransformerSpec, weights: TransformerWeights): string {
  return (
    spec.layers + (spec.layers > 1 ? ' layers' : ' layer') + ' · ' + spec.heads + (spec.heads > 1 ? ' heads' : ' head') + ' × ' + headWidth(spec) +
    ' · d ' + spec.width + (spec.ffn > 0 ? ' · ffn ' + spec.ffn : '') + ' · ' + parameterCount(weights).toLocaleString('en-US') + ' params'
  );
}

export default function TransformersPage() {
  const paramsApi = useExplainerParams<TfParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, merge, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data and model ---------------- */

  const data = useMemo(
    () => buildData({ task: params.task, size: params.size, digits: params.digits, count: params.count, seed: params.seed }),
    [params.task, params.size, params.digits, params.count, params.seed],
  );
  const spec = useMemo(
    () =>
      buildSpec(
        {
          width: params.width,
          heads: params.heads,
          layers: params.layers,
          ffn: params.ffn,
          positions: params.positions,
          mask: params.mask,
          norm: params.norm,
          seed: params.seed,
          optimiser: params.optimiser,
          learningRate: params.learningRate,
          batchSize: params.batchSize,
        },
        data,
      ),
    [data, params.width, params.heads, params.layers, params.ffn, params.positions, params.mask, params.norm, params.seed, params.optimiser, params.learningRate, params.batchSize],
  );

  const sim = useSimulation({
    create: () => startState(spec, data, params.start),
    step: (state) => trainStep(state, data, spec),
    isComplete: (state) => state.diverged || isFitted(state, data.task),
    deps: [spec, data, params.start],
    baseStepsPerSecond: 10,
    maxSteps: MAX_STEPS,
  });
  const state = sim.state;
  // A trained start has its steps behind it; the counter shows the model's.
  const shownSim = useMemo(() => (state.step === sim.iteration ? sim : { ...sim, iteration: state.step }), [sim, state.step]);

  /* ---------------- the probe ---------------- */

  const probe = useMemo(() => probeOf(params.input, params.probe, data), [params.input, params.probe, data]);
  const cache = useMemo(() => forward(state.weights, probe.tokens, spec), [state.weights, probe.tokens, spec]);
  const honest = useMemo(() => honestCalls(state.weights, spec, data, probe.tokens, cache), [state.weights, spec, data, probe.tokens, cache]);
  // A typed start of a letter sentence is continued as typed; otherwise the model writes on from the animal.
  const whole = probe.index === null && data.task === 'letters';
  const written = useMemo(() => writeFrom(state.weights, spec, data, probe.tokens, whole), [state.weights, spec, data, probe.tokens, whole]);
  const plane = useMemo(() => embeddingPlane(state.weights, spec), [state.weights, spec]);
  const positions = useMemo(() => positionTable(state.weights, spec), [state.weights, spec]);

  /* ---------------- writing, one letter at a time ---------------- */

  // While on, the model's next letter joins the text every beat, and every view redraws on the longer text, to the full stop.
  // Any other change to the text, a lesson's or the reader's, stops it.
  const [writing, setWriting] = useState(false);
  const [wrote, setWrote] = useState<string | null>(null);
  const lastToken = probe.tokens[probe.tokens.length - 1];
  const canWrite = data.task === 'letters' && probe.index === null && probe.tokens.length < spec.length && data.vocab[lastToken] !== '.';
  const writingNow = writing && canWrite && params.input === wrote;
  useEffect(() => {
    if (!writingNow) return;
    const id = window.setTimeout(() => {
      const next = spellTokens(data, [...probe.tokens, cache.predicted[probe.tokens.length - 1]]);
      setWrote(next);
      set('input', next);
    }, WRITE_BEAT);
    return () => window.clearTimeout(id);
  }, [writingNow, probe.tokens, cache, data, set]);
  const write = useCallback(
    (text: string) => {
      setWrote(text);
      set('input', text);
      setWriting(true);
    },
    [set],
  );
  const typeInput = useCallback(
    (text: string) => {
      setWriting(false);
      set('input', text);
    },
    [set],
  );
  // A lesson's sweep: each press starts the highlight down the rows again.
  const [sweep, setSweep] = useState(0);

  /* ---------------- views and the open card ---------------- */

  const [archView, setArchView] = useState<ArchView>('model');
  const [secondView, setSecondView] = useState<SecondView>('loss');
  const [selection, setSelection] = useState<ArchSelection | null>(null);
  const [pick, setPick] = useState<HeadPick>(DEFAULT_PICK);
  const [part, setPart] = useState<Part>('none');
  // Places a lesson marks on the positional encoding.
  const [places, setPlaces] = useState<number[]>([]);
  // Colours by default; the numbers on request, remembered across visits.
  const [numbers, setNumbers] = useState(() => readFlag(NUMBERS_KEY));
  const toggleNumbers = useCallback(() => {
    setNumbers((on) => {
      writeFlag(NUMBERS_KEY, !on);
      return !on;
    });
  }, []);
  const shownPick = useMemo<HeadPick>(() => {
    const layer = Math.min(pick.layer, spec.layers - 1);
    const query = Math.min(pick.query, cache.length - 1);
    return { layer, query, head: resolveHead(cache, spec, { layer, head: pick.head, query }) };
  }, [pick, cache, spec]);
  const pickHead = useCallback((layer: number, head: number) => setPick((prev) => ({ ...prev, layer, head })), []);
  const inspect = useCallback((next: HeadPick) => {
    setPick(next);
    setSelection(null);
    setArchView('head');
  }, []);

  /* ---------------- the lessons ---------------- */

  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'arch') setArchView(archViewOf(value));
        else if (id === 'block') setPart(partOf(value));
        else if (id === 'second') setSecondView(value as SecondView);
        else if (id === 'place') setPlaces(value ? value.split(',').map(Number).filter(Number.isFinite) : []);
        else if (id === 'write') write(value);
        else if (id === 'sweep') setSweep((n) => n + 1);
        else if (id === 'pick') {
          const [layer, head, query] = value.split(':').map(Number);
          setPick({ layer: layer || 0, head: Number.isFinite(head) ? head : -1, query: query || 0 });
        }
      },
      open: (target: string | null) => setSelection(target === null ? null : parseArchTarget(target)),
    }),
    [write],
  );
  const openId = selection ? selection.kind + ':' + selection.id : null;
  const lessonContext = useMemo(
    () => makeTfContext({ params, state, data, spec, cache, written, ui: { view: archView, secondView, open: openId, pick } }),
    [params, state, data, spec, cache, written, archView, secondView, openId, pick],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });
  const focus = lessonApi.focus;
  const spot = useMemo<ArchSelection | null>(() => (focus && focus.kind === 'node' && !selection ? { kind: 'node', id: focus.id } : null), [focus, selection]);
  const partSpot = focus && focus.kind === 'node' && partOf(focus.id) !== 'none' ? partOf(focus.id) : null;

  /* ---------------- the chapter reset ---------------- */

  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  const resetAll = useCallback(() => {
    resetLessons();
    reset();
    resetRun();
    setArchView('model');
    setPart('none');
    setSecondView('loss');
    setSelection(null);
    setPick(DEFAULT_PICK);
    setPlaces([]);
    setWriting(false);
  }, [resetLessons, reset, resetRun]);
  const pickMoved = pick.layer !== DEFAULT_PICK.layer || pick.head !== DEFAULT_PICK.head || pick.query !== DEFAULT_PICK.query;
  const canReset =
    isDirty || lessonApi.dirty || sim.iteration > 0 || archView !== 'model' || part !== 'none' || secondView !== 'loss' || selection !== null || pickMoved || places.length > 0;

  // Picking a task also picks the mask and the alphabet it is written for.
  const change = useCallback(
    <K extends keyof TfParams>(key: K, value: TfParams[K]) => {
      if (key === 'task') merge({ task: String(value), mask: maskFor(String(value)), digits: digitsFor(String(value)), layers: layersFor(String(value)), input: '', probe: 0 });
      else set(key, value);
    },
    [merge, set],
  );

  /* ---------------- the guide ---------------- */

  const soundTask = data.nextToken;
  const letters = data.task === 'letters';
  const guide = useMemo<GuideStatus>(() => {
    if (state.diverged) return { text: 'The weights blew up: η = ' + fmtKnob(params.learningRate) + ' is too large. Lower it and replay.', tone: 'bad' };
    if (state.step === 0) return { text: '' };
    if (sim.isComplete && letters) return { text: 'Fitted: it writes every held-out sentence and gets ' + pc(state.testAccuracy) + ' of the sound letters, after ' + state.step + ' steps.', tone: 'good' };
    if (sim.isComplete) return { text: 'Fitted: every held-out ' + (soundTask ? 'sound' : 'answer') + ' right after ' + state.step + ' steps.', tone: 'good' };
    if (soundTask && !spec.causal && state.trainLoss < 0.05) {
      return { text: 'Training loss ' + fmtCompact(state.trainLoss) + ', yet only ' + pc(state.testAccuracy) + ' of sounds right when writing: with no mask every word sees the next one.', tone: 'warn' };
    }
    if (state.step >= 300 && state.testAccuracy < 0.6) return { text: 'Still at ' + pc(state.testAccuracy) + ' after ' + state.step + ' steps: something the task needs is missing.', tone: 'warn' };
    if (sim.iteration >= MAX_STEPS) return { text: 'Stopped at ' + MAX_STEPS.toLocaleString('en-US') + ' steps. Reset to run again.', tone: 'warn' };
    return { text: '' };
  }, [state, params.learningRate, sim.isComplete, sim.iteration, soundTask, letters, spec.causal]);

  /* ---------------- metrics ---------------- */

  const palette = usePalette();
  const headline: Metric[] = useMemo(
    () => [
      {
        key: 'test-acc',
        label: letters ? 'sound letters' : soundTask ? 'sounds right' : 'held out',
        value: fmtPercent(state.testAccuracy, 0),
        colour: palette.orange,
        tone: isFitted(state, data.task) ? 'good' : 'neutral',
        help: letters
          ? 'Share of the sound’s letters in held-out sentences that the model guesses right, reading only the letters before. The sound depends on the animal words back.'
          : soundTask
            ? 'Share of held-out sentences where the model, reading only the words before, names the right sound after says.'
            : 'Share of held-out digits answered right at the last evaluation. The one number to trust.',
      },
      {
        key: 'train-loss',
        label: 'train loss',
        value: state.diverged ? '∞' : fmtCompact(state.trainLoss),
        colour: palette.blue,
        tone: state.diverged ? 'bad' : 'neutral',
        help: 'Mean cross-entropy over the predictions the training sequences ask for.',
      },
    ],
    [state, palette, soundTask, letters, data.task],
  );
  const more: Metric[] = useMemo(
    () => [
      { key: 'train-acc', label: 'Train accuracy', value: fmtPercent(state.trainAccuracy, 0), help: 'The same score on training sequences. Far above the held-out score means memorising.' },
      soundTask
        ? { key: 'exact', label: 'Sentences written', value: fmtPercent(state.testExact, 0), help: 'Held-out sentences the model, given the start up to the animal, writes out as a sentence of the language.' }
        : { key: 'exact', label: 'Whole sequences', value: fmtPercent(state.testExact, 0), help: 'Held-out sequences with every scored position right.' },
      { key: 'test-loss', label: 'Held-out loss', value: state.diverged ? '∞' : fmtCompact(state.testLoss), help: 'Mean cross-entropy on the held-out sequences.' },
      { key: 'params', label: 'Parameters', value: parameterCount(state.weights).toLocaleString('en-US'), help: 'Every weight in the embeddings, the blocks and the output.' },
      { key: 'epoch', label: 'Epochs', value: String(state.epoch), help: 'Full passes over the training sequences.' },
    ],
    [state, soundTask],
  );

  /* ---------------- render ---------------- */

  const renderDetail = useCallback(
    (active: ArchSelection, jump: (next: ArchSelection) => void) => <TfDetail selection={active} data={data} spec={spec} cache={cache} probe={probe} jump={jump} inspect={inspect} numbers={numbers} />,
    [data, spec, cache, probe, inspect, numbers],
  );

  const input = spellTokens(data, probe.tokens);
  const right = probe.scored.filter((t) => honest[t] === probe.targets[t]).length;
  const subtitle =
    (probe.index === null ? 'Your sequence ' : 'Held-out ' + (probe.index + 1) + ': ') + input +
    (probe.scored.length > 0 ? ' · ' + right + ' of ' + probe.scored.length + ' right' : '') + (state.step === 0 ? ' · untrained' : '');
  const summary = architectureLabel(spec, state.weights);
  const trainCount = data.trainIndex.length;
  const testCount = data.testIndex.length;
  const stepHelp = 'One step: ' + params.batchSize + ' training sequences go through the model, the loss is measured and every weight moves once. The ' + testCount + ' held-out sequences are only ever scored.';
  const heads = Array.from({ length: spec.heads }, (_, h) => ({ label: 'head ' + (h + 1), colour: headColour(palette, h), shape: 'line' as const }));
  const viewDescription =
    archView === 'model'
      ? 'The whole model for ' + input + ', one row per token from the embedding on the left through attention and the feed-forward block to the output on the right.'
      : archView === 'head'
        ? 'Inside head ' + (shownPick.head + 1) + ' for the token at position ' + shownPick.query + ': its query against every key, the scores, the softmax weights and the blend of values.'
        : archView === 'attention'
          ? 'The attention weights of every head for ' + input + ': each row is a token, each cell how much of another token it takes.'
          : 'The sequence ' + input + ' flowing down through the model, one lane per token; lines in the attention band show who takes from whom.';

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
      <LessonOpenContext.Provider value={lessonApi.view.active}>
        <Studio
          meta={META}
          shareUrl={shareUrl}
          lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
          transport={<Transport sim={shownSim} unit="step" unitHelp={stepHelp} completeLabel={state.diverged ? 'Diverged' : 'Fitted'} completeTone={state.diverged ? 'bad' : 'good'} />}
          topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={change} lesson={lastPreset} named={lessonApi.view.knobs} />}
          tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
          controls={<ControlPanel groups={RAIL_GROUPS} params={params} onChange={change} />}
          wide
          architecture={
            <Panel
              id="architecture"
              title="The transformer"
              subtitle={subtitle}
              actions={
                <>
                  <ProbeStepper index={probe.index} count={testCount} onChange={(next) => merge({ probe: next, input: '' })} />
                  <Legend dense items={heads} />
                  <div className="mlx-viewswitch mlx-segmented" role="group" aria-label="Numbers">
                    <button
                      type="button"
                      className="mlx-segmented__item"
                      data-active={numbers || undefined}
                      aria-pressed={numbers}
                      onClick={toggleNumbers}
                      title={numbers ? 'Hide the numbers: colours only, more room for the picture' : 'Write the numbers in the cells'}
                    >
                      Numbers
                    </button>
                  </div>
                  <ViewSwitch
                    id="arch"
                    label="Architecture view"
                    value={archView}
                    options={[
                      { value: 'model', label: 'Model' },
                      { value: 'flow', label: 'Flow' },
                      { value: 'attention', label: 'Attention' },
                      { value: 'head', label: 'Head' },
                    ]}
                    onChange={(next) => setArchView(archViewOf(next))}
                  />
                </>
              }
            >
              {archView === 'model' ? (
                <ArchitectureView
                  data={data}
                  spec={spec}
                  cache={cache}
                  probe={probe}
                  part={part}
                  onPart={setPart}
                  layer={shownPick.layer}
                  head={shownPick.head}
                  onPick={pickHead}
                  spot={partSpot}
                  numbers={numbers}
                  places={places}
                  sweep={sweep}
                  description={viewDescription}
                  redrawKey={state.step}
                />
              ) : archView === 'attention' ? (
                <AttentionGrid data={data} spec={spec} cache={cache} pick={null} onPick={inspect} numbers={numbers} description={viewDescription} redrawKey={state.step} />
              ) : archView === 'head' ? (
                <HeadView data={data} spec={spec} cache={cache} pick={shownPick} onPick={setPick} numbers={numbers} description={viewDescription} redrawKey={state.step} />
              ) : (
                <FlowView
                  data={data}
                  spec={spec}
                  cache={cache}
                  probe={probe}
                  selection={selection}
                  onSelect={setSelection}
                  spot={spot}
                  renderDetail={renderDetail}
                  summary={summary}
                  numbers={numbers}
                  description={viewDescription}
                  redrawKey={state.step}
                />
              )}
              <TokenInput
                data={data}
                probe={probe}
                value={params.input}
                onChange={typeInput}
                onWrite={data.task === 'letters' ? write : undefined}
                writing={writingNow}
                onStop={() => setWriting(false)}
              />
            </Panel>
          }
          output={
            <>
              <ColumnHead title="Output" blurb={summary} />

              <Panel id="answer" title={letters ? 'Next letter' : soundTask ? 'Next words' : 'The answer'}>
                <AnswerView data={data} spec={spec} cache={cache} probe={probe} honest={honest} written={written} />
              </Panel>

              <Panel id="gallery" fill className="mlx-tf-gallery" title="Held-out sequences" subtitle={testCount + ' held out, ' + trainCount + ' train. Click one to follow it.'}>
                <Gallery data={data} state={state} probe={probe.index} onProbe={(index) => merge({ probe: index, input: '' })} />
              </Panel>

              <Panel
                id="second"
                fill
                className="mlx-tf-second"
                title={SECOND_TITLES[secondView]}
                actions={
                  <>
                    {secondView === 'loss' || secondView === 'accuracy' ? (
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
                        { value: 'loss', label: 'Loss' },
                        { value: 'accuracy', label: 'Accuracy' },
                        { value: 'embeddings', label: 'Embeddings' },
                        { value: 'positions', label: 'Positions' },
                      ]}
                      onChange={(next) => setSecondView(next as SecondView)}
                    />
                  </>
                }
              >
                {secondView === 'loss' || secondView === 'accuracy' ? (
                  <Curves history={state.history} view={secondView} />
                ) : secondView === 'embeddings' ? (
                  <EmbeddingPlane data={data} plane={plane} />
                ) : (
                  <PositionMap pos={positions} length={spec.length} width={spec.width} mode={spec.positions} />
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
