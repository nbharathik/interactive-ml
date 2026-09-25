/** Cards for the two functions, a layer of the bent plane, a point's forward pass, and a hidden unit over the input plane. */

import { useCallback, useMemo } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { Annotation, DetailHead, Formula, Scalar, Scalars, WeightedSumTable } from '../../explainer/components/Detail';
import { useTurntable } from '../../explainer/useTurntable';
import { fmt, fmtPercent, sub } from '../../lib/math/stats';
import { FONT_STACK, MONO_STACK } from '../../lib/viz/canvas';
import { categorical, rgba } from '../../lib/viz/palette';
import { drawIsoLines } from '../../lib/viz/plots';
import { drawSurface } from '../../lib/viz/surface';
import type { SurfaceMark } from '../../lib/viz/surface';
import type { ActivationFn, ActivationName } from '../../lib/ml/activations';
import type { LossFn } from '../../lib/ml/losses';
import { forward } from '../../lib/ml/mlp';
import { scoreFromOutput } from '../../lib/ml/mlpTrainer';
import type { TrainerState } from '../../lib/ml/mlpTrainer';
import { TILE_N, creaseSegments, drawUnitTile, rowRange, sampleUnit, signedColour, tileFrame, tileNorm } from './tiles';

/** The cheat sheet, one entry per activation: range, slope, what it is for. */
const ACTIVATION_NOTES: Record<ActivationName, string> = {
  sigmoid:
    'Range (0, 1), not centred on zero. Its slope peaks at 0.25 at z = 0, so d sigmoid layers scale a gradient back by at most 0.25 to the power d: five of them by about 0.001. Its place today is the output of a two-class model, where it turns a score into a probability.',
  tanh: 'Range (−1, 1), centred on zero, slope up to 1. It still flattens beyond |z| ≈ 3. The usual choice inside recurrent networks.',
  relu: 'Zero on the left, the identity on the right: the slope is exactly 0 or 1, so a gradient on an active path passes back untouched. A unit whose z is negative for every input gets no gradient and never recovers: a dead unit. The default for hidden layers.',
  leakyRelu: 'ReLU with slope α on the left, so a negative unit still passes α of the gradient back and cannot die. One more number to choose.',
  elu: 'The identity on the right, α(eᶻ − 1) on the left: smooth, and the negative outputs keep a layer’s mean near zero. Costs an exponential.',
  gelu: 'z times the chance a standard normal falls below z: a smooth gate that dips a little below zero near z = −0.75 and matches ReLU far out. The default inside transformers.',
  softplus: 'log(1 + eᶻ): ReLU with the corner rounded off. The slope is never exactly zero and the output never exactly zero either.',
  swish: 'z·σ(z): a smooth, self-gated ReLU with a dip below zero near z = −1.3. Found by an automated search; used in EfficientNet.',
  linear: 'f(z) = z. A stack of linear layers multiplies out to one matrix, so depth adds nothing: with a sigmoid on the output the whole network is logistic regression, whatever the depth.',
};

/** What f does to the grid it receives, in one line. */
const BEND_NOTES: Record<ActivationName, string> = {
  sigmoid: 'squashes every value into (0, 1); the far reaches pile up at the edges, where the slope is near zero.',
  tanh: 'squashes every value into (−1, 1); the middle keeps its shape, the far reaches pile up at ±1.',
  relu: 'sets every negative value to 0, so the sheet folds onto the axes; the creases are where the next line can bend.',
  leakyRelu: 'folds the negative side too, but keeps a thin copy of it instead of flattening it.',
  elu: 'folds the negative side smoothly and lets it dip a little below zero.',
  gelu: 'folds the negative side with a rounded crease and a slight dip below zero.',
  softplus: 'folds the negative side with a rounded crease, keeping everything above zero.',
  swish: 'folds the negative side with a rounded crease and a dip below zero.',
  linear: 'changes nothing: the grid stays a grid, and this layer could be multiplied into the next.',
};

export function ActivationDetail({ act, state, zByLayer }: { act: ActivationFn; state: TrainerState; zByLayer: readonly (readonly number[])[] }) {
  const all = zByLayer.flat();
  const slopes = all.map((z) => Math.abs(act.df(z)));
  const meanSlope = slopes.length ? slopes.reduce((a, b) => a + b, 0) / slopes.length : 0;
  const flat = slopes.length ? slopes.filter((s) => s < 0.05).length / slopes.length : 0;
  const hidden = state.layers.slice(0, -1);
  const dead = hidden.length ? hidden.reduce((s, l) => s + l.smallGradFraction, 0) / hidden.length : 0;
  const range = act.range ? '[' + (Number.isFinite(act.range[0]) ? fmt(act.range[0], 0) : '−∞') + ', ' + (Number.isFinite(act.range[1]) ? fmt(act.range[1], 0) : '∞') + ']' : 'unbounded';
  return (
    <>
      <DetailHead eyebrow="Activation" title={act.label} shape={'range ' + range + (act.saturatesLeft ? ' · flat on the left' : '') + (act.saturatesRight ? ' · flat on the right' : '')} />
      <Formula tex={act.tex} />
      <Formula tex={act.dtex} />
      <Annotation>{ACTIVATION_NOTES[act.name]}</Annotation>
      <Annotation>
        {act.saturatesLeft || act.saturatesRight
          ? 'Where the curve is flat the derivative is near zero, and a unit sitting there passes almost no gradient back.'
          : 'The slope never reaches zero, so every unit keeps passing gradient back wherever it sits.'}
      </Annotation>
      <Scalars>
        <Scalar label="units sampled" value={all.length} decimals={0} />
        <Scalar label="mean |f′|" value={meanSlope} decimals={3} tone="accent" />
        <Scalar label="in flat zones" value={fmtPercent(flat, 0)} />
        <Scalar label="slope below 1e-3" value={fmtPercent(dead, 0)} tone={dead > 0.3 ? 'bad' : undefined} />
      </Scalars>
    </>
  );
}

export function LossDetail({ lossFn, state, axis }: { lossFn: LossFn; state: TrainerState; axis: readonly number[] }) {
  const grads = state.outputs.map((o, i) => (state.batchTargets[i] ? lossFn.grad(o, state.batchTargets[i]) : [0]));
  const meanGrad = grads.length ? grads.reduce((s, g) => s + Math.hypot(...g), 0) / grads.length : 0;
  const zeroShare = grads.length ? grads.filter((g) => Math.hypot(...g) < 1e-9).length / grads.length : 0;
  const meanAxis = axis.length ? axis.reduce((a, b) => a + b, 0) / axis.length : 0;
  return (
    <>
      <DetailHead
        eyebrow="Loss"
        title={lossFn.label}
        shape={lossFn.kind === 'regression' ? 'scores a number' : lossFn.kind === 'binary' ? 'scores a two-class score' : 'scores one logit per class'}
      />
      <Formula tex={lossFn.tex} />
      <Annotation>
        {lossFn.name === 'bce'
          ? 'Its gradient on the logit is p − y: largest when the model is confident and wrong, never zero.'
          : lossFn.name === 'hinge'
            ? 'Zero, with zero gradient, once y·s exceeds 1: a point past the margin stops teaching.'
            : lossFn.name === 'mse' && lossFn.kind !== 'regression'
              ? 'Through a sigmoid its gradient carries ŷ(1 − ŷ), which vanishes at both confident ends.'
              : lossFn.name === 'mae'
                ? 'The gradient is ±1 whatever the error, so the pull never softens as the fit improves.'
                : lossFn.name === 'huber'
                  ? 'Squared inside δ, absolute outside: outliers pull with a capped force.'
                  : lossFn.name === 'softmaxCE'
                    ? 'The gradient on each logit is its probability minus its target.'
                    : 'The gradient grows with the error, so far points pull hardest.'}
      </Annotation>
      <Scalars>
        <Scalar label="batch outputs" value={state.outputs.length} decimals={0} />
        <Scalar label={'mean ' + lossFn.curve.xLabel} value={meanAxis} decimals={3} />
        <Scalar label="mean |dL|" value={meanGrad} decimals={4} tone="accent" />
        <Scalar label="zero gradient" value={fmtPercent(zeroShare, 0)} tone={zeroShare > 0.5 ? 'bad' : undefined} />
        <Scalar label="train loss" value={state.trainLoss} decimals={4} />
      </Scalars>
    </>
  );
}

/* ---------------- the bent plane ---------------- */

function matrixTex(W: readonly number[], rows: number, cols: number): string {
  const body = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => fmt(W[r * cols + c], 2)).join(' & '),
  ).join(' \\\\ ');
  return '\\begin{bmatrix} ' + body + ' \\end{bmatrix}';
}

function vectorTex(b: readonly number[]): string {
  return '\\begin{bmatrix} ' + b.map((v) => fmt(v, 2)).join(' \\\\ ') + ' \\end{bmatrix}';
}

export function LayerDetail({ index, state, act }: { index: number; state: TrainerState; act: ActivationFn }) {
  if (index === 0) {
    const inputDim = state.net.layers[0]?.inSize ?? 2;
    return (
      <>
        <DetailHead eyebrow="Input" title="x, the plane every point starts on" shape={inputDim === 1 ? 'x · one number per point' : 'x₁, x₂ · two numbers per point'} />
        <Formula tex={inputDim === 1 ? 'x \\in [-1, 1]' : '\\mathbf{x} = (x_1, x_2) \\in [-1, 1]^2'} />
        <Annotation>A grid drawn on the inputs, with the training points on it. Every layer that follows bends this grid; the points ride along.</Annotation>
      </>
    );
  }
  const layer = state.net.layers[index - 1];
  const projected = layer.outSize > 2;
  const diag = state.layers[index - 1];
  const f = act.name === 'linear' ? '' : '\\text{' + act.label.toLowerCase() + '}';
  const previous = index === 1 ? '\\mathbf{x}' : '\\mathbf{h}_{' + (index - 1) + '}';
  const small = layer.inSize * layer.outSize <= 9;
  const tex = small
    ? '\\mathbf{h}_{' + index + '} = ' + f + '\\left(' + matrixTex(layer.W, layer.outSize, layer.inSize) + previous + ' + ' + vectorTex(layer.b) + '\\right)'
    : '\\mathbf{h}_{' + index + '} = ' + f + '(W_{' + index + '} ' + previous + ' + \\mathbf{b}_{' + index + '}),\\quad W_{' + index + '} \\in \\mathbb{R}^{' + layer.outSize + ' \\times ' + layer.inSize + '}';
  const slopes = diag.zSamples.map((z) => Math.abs(act.df(z)));
  const flat = slopes.length ? slopes.filter((s) => s < 0.05).length / slopes.length : 0;
  const hiddenCount = state.net.layers.length - 1;
  const before = index === 1 ? 'x' : 'h' + sub(index - 1);
  const z = 'z' + sub(index);
  return (
    <>
      <DetailHead
        eyebrow={'Layer ' + index + ' of ' + hiddenCount}
        title={'h' + sub(index) + ' = ' + (act.name === 'linear' ? z : act.label.toLowerCase() + '(' + z + ')')}
        shape={layer.inSize + ' → ' + layer.outSize + ' · ' + (layer.W.length + layer.b.length) + ' parameters'}
      />
      <Formula tex={tex} />
      <ol className="mlx-detail__steps">
        <li>
          <code>{z + ' = W' + sub(index) + before + ' + b' + sub(index)}</code>
          {' the grid is turned and stretched by W' + sub(index) + ', then slid by b' + sub(index) + '. Still a grid.'}
        </li>
        <li>
          <code>{act.label.toLowerCase()}</code>
          {' ' + BEND_NOTES[act.name]}
        </li>
        <li>
          <code>{index === hiddenCount ? 'ŷ' : 'h' + sub(index)}</code>
          {index === hiddenCount ? ' the output unit draws one straight line through this bent grid.' : ' the bent grid is what layer ' + (index + 1) + ' receives.'}
        </li>
      </ol>
      {projected ? (
        <Annotation>
          {'Each point here is ' + layer.outSize + ' numbers, one per unit, too many to draw flat. The pane shows them along their two directions of largest spread (pc 1 and pc 2), so its axes are mixes of units, not units. Units ▾ on the pane shows each unit on its own, over the input plane.'}
        </Annotation>
      ) : null}
      <Scalars>
        <Scalar label="‖W‖" value={diag.weightNorm} decimals={2} />
        <Scalar label="flat" value={fmtPercent(flat, 0)} tone={flat > 0.5 ? 'bad' : undefined} />
        <Scalar label="‖∇W‖" value={diag.gradNorm < 0.001 && diag.gradNorm > 0 ? diag.gradNorm.toExponential(1) : fmt(diag.gradNorm, 3)} tone="accent" />
      </Scalars>
    </>
  );
}

/* ---------------- one point through the network ---------------- */

export function PointDetail({
  state,
  input,
  label,
  task,
  classCount,
  eyebrow = 'Point',
}: {
  state: TrainerState;
  /** In network units. */
  input: readonly number[];
  label: number | null;
  task: 'classify' | 'regress';
  classCount: number;
  eyebrow?: string;
}) {
  const caches = forward(state.net, input, state.spec);
  const last = state.net.layers[state.net.layers.length - 1];
  const hiddenIn = caches[caches.length - 1].input;
  const out = caches[caches.length - 1].a;
  const z = caches[caches.length - 1].z;
  const lossName = state.spec.loss;
  // The output row shown: the class unit for multiclass, else the single unit.
  const row = lossName === 'softmaxCE' ? out.indexOf(Math.max(...out)) : 0;
  const rows = hiddenIn.map((h, j) => ({ name: caches.length === 1 ? 'x' + sub(j + 1) : 'h' + sub(j + 1), value: h, weight: last.W[row * last.inSize + j], weightName: 'w' + sub(j + 1) }));
  const score = task === 'classify' ? scoreFromOutput(lossName, out) : 0;
  const top = Math.max(...out);
  const softmaxRow = Math.exp(out[row] - top) / out.reduce((s, v) => s + Math.exp(v - top), 0);
  const readout =
    task === 'regress'
      ? { label: 'ŷ = z', value: out[0] }
      : lossName === 'hinge'
        ? { label: 'score = z', value: z[0] }
        : lossName === 'softmaxCE'
          ? { label: 'softmax(z)' + sub(row + 1), value: softmaxRow }
          : lossName === 'mse'
            ? { label: 'ŷ = σ(z)', value: out[0] }
            : { label: 'ŷ = σ(z)', value: (score + 1) / 2 };
  const raw = input.map((v) => fmt(v * 6, 1)).join(', ');
  const layersText = state.net.layers.length - 1;
  return (
    <>
      <DetailHead
        eyebrow={eyebrow}
        title={'x = (' + raw + ')'}
        shape={(label !== null ? 'y ' + (task === 'regress' ? fmt(label, 2) : label) + ' · ' : '') + (layersText === 0 ? 'no hidden layer' : layersText + ' hidden layer' + (layersText === 1 ? '' : 's') + ' → ' + hiddenIn.length + ' values')}
      />
      <WeightedSumTable rows={rows} bias={last.b[row]} columns={['input', 'value', 'weight', 'product']} sumLabel="z" result={readout} decimals={2} />
      <Annotation>
        {task === 'classify'
          ? layersText === 0
            ? 'This is logistic regression: a weighted sum of the inputs and a sigmoid. Its boundary is the straight line z = 0.'
            : 'The last step is logistic regression on the last hidden layer: a weighted sum and a sigmoid, a straight line in that space. Everything before it only moves the point to where a straight line can do the job.'
          : 'The output is a weighted sum of the last hidden layer, so the bends come from the layers before it.'}
      </Annotation>
      {classCount > 2 && task === 'classify' ? <Annotation>{'One row per class; this is the row of the winning class, class ' + row + '.'}</Annotation> : null}
    </>
  );
}

/* ---------------- one hidden unit ---------------- */

/** The unit over the input plane: as a sheet you can turn (its bend), and from above as the ramp z with its contour lines and the direction of w. */
function UnitTiles({ state, act, index, layer, inputs, labels }: { state: TrainerState; act: ActivationFn; index: number; layer: number; inputs: readonly (readonly number[])[]; labels: readonly number[] | null }) {
  const sample = useMemo(() => sampleUnit(state, index, layer), [state, index, layer]);
  const turntable = useTurntable();
  const { view } = turntable;
  const f = act.name === 'linear' ? 'z' : act.label.toLowerCase() + '(z)';
  const { w, b } = useMemo(() => {
    const dense = state.net.layers[layer - 1];
    return { w: Array.from({ length: dense.inSize }, (_, i) => dense.W[index * dense.inSize + i]), b: dense.b[index] };
  }, [state, index, layer]);
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (!sample) return;
      const side = Math.max(40, Math.min(Math.floor((width - 10) / 2), height - 30));
      const left = Math.floor((width - (side * 2 + 10)) / 2);
      const sheet = { x: left, y: 16, w: side, h: side };
      const flat = { x: left + side + 10, y: 16, w: side, h: side };
      ctx.font = '600 10px ' + MONO_STACK;
      ctx.fillStyle = palette.text;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('a = ' + f + ', drag to turn', sheet.x, 11, side - 4);
      ctx.fillText('z = w·x + b, from above', flat.x, 11, side - 4);

      // The sheet on the layer's shared scale, its fold on the zero plane, the points at their height.
      const [lo, hi] = rowRange(act, sample.layer);
      const half = Math.max(Math.abs(sample.layer.aMin), Math.abs(sample.layer.aMax), 1e-6);
      const zOf = (v: number) => (v - lo) / (hi - lo);
      const zero = lo < 0 && hi > 0 ? zOf(0) : undefined;
      const at = (p: readonly number[]) => {
        const col = Math.round(((p[0] + 1) / 2) * (TILE_N - 1));
        const row = Math.round((((p[1] ?? 0) + 1) / 2) * (TILE_N - 1));
        return sample.field.a[Math.min(TILE_N - 1, Math.max(0, row)) * TILE_N + Math.min(TILE_N - 1, Math.max(0, col))];
      };
      const marks: SurfaceMark[] = inputs.map((p, i) => ({ u: p[0], v: p[1] ?? 0, z: zOf(at(p)), colour: labels ? categorical(palette, labels[i]) : palette.blue, radius: 2 }));
      drawSurface(ctx, sheet, palette, sample.field.a, TILE_N, TILE_N, view, {
        zOf,
        colourAt: (v) => signedColour(palette, v, half, false),
        floorLines: creaseSegments(sample.field.z, rgba(palette.text, 0.6), 1, [3, 3]),
        zeroPlane: zero,
        marks,
        labels: { u: 'x₁', v: 'x₂', z: 'a' },
        ticks: { u: ['−6', '6'], z: { lo: fmt(lo, 1), hi: fmt(hi, 1), zero } },
      });

      // From above: the ramp, its contour lines, and w pointing up the slope from the line z = 0.
      drawUnitTile(ctx, flat, palette, sample.field.z, tileNorm(act, sample.layer, false), sample.field.z, { points: inputs, labels, axes: true });
      const frame = tileFrame(flat);
      const zAt = (x: number, y: number) => {
        const col = Math.round(((x + 1) / 2) * (TILE_N - 1));
        const row = Math.round(((y + 1) / 2) * (TILE_N - 1));
        return sample.field.z[Math.min(TILE_N - 1, Math.max(0, row)) * TILE_N + Math.min(TILE_N - 1, Math.max(0, col))];
      };
      const step = sample.layer.zMax / 4;
      drawIsoLines(ctx, frame, palette, zAt, [-3, -2, -1, 1, 2, 3].map((m) => m * step), { cellSize: 3, colour: palette.text, width: 0.8, alpha: 0.3 });
      if (w.length === 2) {
        const norm = Math.hypot(w[0], w[1]);
        if (norm > 1e-9) {
          // The foot of w on the line, kept inside the square, then 0.35 of the way up the slope.
          const foot = [(-b * w[0]) / (norm * norm), (-b * w[1]) / (norm * norm)].map((v) => Math.max(-0.7, Math.min(0.7, v)));
          const tip = [foot[0] + (0.35 * w[0]) / norm, foot[1] + (0.35 * w[1]) / norm];
          const x0 = frame.x(foot[0]);
          const y0 = frame.y(foot[1]);
          const x1 = frame.x(tip[0]);
          const y1 = frame.y(tip[1]);
          const dx = x1 - x0;
          const dy = y1 - y0;
          const len = Math.hypot(dx, dy) || 1;
          const ux = dx / len;
          const uy = dy / len;
          ctx.save();
          ctx.strokeStyle = palette.accent;
          ctx.fillStyle = palette.accent;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1 - ux * 5, y1 - uy * 5);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ux * 7 - uy * 4, y1 - uy * 7 + ux * 4);
          ctx.lineTo(x1 - ux * 7 + uy * 4, y1 - uy * 7 - ux * 4);
          ctx.closePath();
          ctx.fill();
          ctx.font = '600 10px ' + MONO_STACK;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText('w', x1 + uy * 6 + 4, y1 - ux * 6);
          ctx.restore();
        }
      }

      // The scale, once for both.
      ctx.font = '10px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('x₁, x₂: −6 to 6 · a: ' + fmt(lo, 1) + ' to ' + fmt(hi, 1) + ' · orange < 0 < blue · line: z = 0', left, sheet.y + side + 12, side * 2 + 10);
    },
    [sample, act, f, inputs, labels, view, w, b],
  );
  if (!sample) return null;
  return (
    <Chart
      className="mlx-detail__tiles"
      draw={draw}
      height={(width) => Math.max(56, Math.floor((width - 10) / 2)) + 30}
      description={'Unit ' + (index + 1) + ' of layer ' + layer + ' over the input plane: as a sheet, flat at zero on one side of its line and rising on the other, and from above as the ramp z with its contour lines.'}
      cursor={turntable.dragging ? 'grabbing' : 'grab'}
      drag
      onPointerDown={(pos) => turntable.down(pos)}
      onPointerMove={(pos) => {
        turntable.move(pos);
      }}
      onPointerUp={() => {
        turntable.up();
      }}
      redrawKey={state.epoch + ':' + index + ':' + layer + ':' + act.name + ':' + turntable.key}
    />
  );
}

export function UnitDetail({
  index,
  layer = 1,
  state,
  act,
  points,
  labels,
}: {
  index: number;
  layer?: number;
  state: TrainerState;
  act: ActivationFn;
  /** Training inputs in network units, with their labels, for the tiles. */
  points?: readonly (readonly number[])[];
  labels?: readonly number[] | null;
}) {
  const hiddenCount = state.net.layers.length - 1;
  const k = Math.max(1, Math.min(layer, hiddenCount));
  const dense = state.net.layers[k - 1];
  const w = Array.from({ length: dense.inSize }, (_, i) => dense.W[index * dense.inSize + i]);
  const b = dense.b[index];
  const norm = Math.hypot(...w);
  const f = act.name === 'linear' ? '' : '\\text{' + act.label.toLowerCase() + '}';
  const inputs = k === 1 ? ['x_1', 'x_2'] : w.map((_, i) => 'h_{' + (k - 1) + ',' + (i + 1) + '}');
  const term = (v: number, name: string, first: boolean) => (first ? (v < 0 ? '-' : '') : v < 0 ? ' - ' : ' + ') + fmt(Math.abs(v), 2) + '\\,' + name;
  const sum = w.length <= 3 ? w.map((v, i) => term(v, inputs[i], i === 0)).join('') : '\\mathbf{w} \\cdot \\mathbf{h}_{' + (k - 1) + '}';
  const tex = 'a_{' + (index + 1) + '} = ' + f + '\\left(' + sum + (b < 0 ? ' - ' : ' + ') + fmt(Math.abs(b), 2) + '\\right)';
  const diag = state.layers[k - 1];
  const slopes = diag.zSamples.map((z) => Math.abs(act.df(z)));
  const meanSlope = slopes.length ? slopes.reduce((a, s) => a + s, 0) / slopes.length : 0;
  const shape = act.name === 'sigmoid' || act.name === 'tanh' ? 'a step' : act.name === 'linear' ? 'a tilted plane' : 'a crease';
  const over = k === 1 ? 'the input plane' : 'the space layer ' + (k - 1) + ' made';
  return (
    <>
      <DetailHead eyebrow={'Unit ' + (index + 1) + ' of layer ' + k} title={'a = ' + act.label.toLowerCase() + '(w·' + (k === 1 ? 'x' : 'h' + sub(k - 1)) + ' + b)'} shape={w.length + ' weights + bias · ' + act.label.toLowerCase()} />
      <Formula tex={tex} />
      {points ? <UnitTiles state={state} act={act} index={index} layer={k} inputs={points} labels={labels ?? null} /> : null}
      <Annotation>
        {'Over ' + over + ' this unit is ' + shape + ' along the line w·' + (k === 1 ? 'x' : 'h') + ' + b = 0; w sets which way it faces and ‖w‖ how sharp it is.' +
          (k === 1 ? ' The output adds such units up and reads the sum through a sigmoid: logistic regression over the ridges.' : ' Over the input plane it bends what the earlier layers bent.')}
      </Annotation>
      <Scalars>
        <Scalar label="‖w‖" value={norm} decimals={2} tone="accent" />
        {w.length === 2 ? <Scalar label="direction" value={fmt((Math.atan2(w[1], w[0]) * 180) / Math.PI, 0) + '°'} /> : null}
        <Scalar label="b/‖w‖" value={norm > 1e-9 ? b / norm : 0} decimals={2} />
        <Scalar label="layer mean |f′|" value={meanSlope} decimals={3} />
      </Scalars>
    </>
  );
}

/* ---------------- the output unit ---------------- */

/** How the output reads its sum z, per task and loss: a sigmoid, a raw score, the value itself, or a softmax. */
export function outputRead(classify: boolean, lossName: string): { tex: string; text: string; shape: string } {
  if (!classify) return { tex: 'z', text: 'z', shape: 'the value itself' };
  if (lossName === 'hinge') return { tex: 'z', text: 'z', shape: 'score' };
  if (lossName === 'softmaxCE') return { tex: '\\mathrm{softmax}(\\mathbf{z})', text: 'softmax(z)', shape: 'one z per class' };
  return { tex: '\\sigma(z)', text: 'σ(z)', shape: 'sigmoid' };
}

export function OutputDetail({ state, lossName, classify }: { state: TrainerState; lossName: string; classify: boolean }) {
  const last = state.net.layers[state.net.layers.length - 1];
  const v = Array.from({ length: last.inSize }, (_, i) => last.W[i]);
  const b = last.b[0];
  const hidden = state.net.layers.length - 1;
  const inputs = hidden === 0 ? ['x_1', 'x_2'] : v.map((_, j) => 'a_{' + (j + 1) + '}');
  const term = (value: number, name: string, first: boolean) => (first ? (value < 0 ? '-' : '') : value < 0 ? ' - ' : ' + ') + fmt(Math.abs(value), 2) + '\\,' + name;
  // Two terms per line, so a wide sum stays inside the card.
  const terms = v.length <= 6 ? v.map((value, j) => term(value, inputs[j], j === 0)) : ['\\mathbf{v} \\cdot \\mathbf{a}'];
  const lines: string[] = [];
  for (let i = 0; i < terms.length; i += 2) lines.push(terms.slice(i, i + 2).join(''));
  lines[lines.length - 1] += (b < 0 ? ' - ' : ' + ') + fmt(Math.abs(b), 2);
  const sum = lines.length === 1 ? 'z = ' + lines[0] : '\\begin{aligned} z &= ' + lines.join(' \\\\ &\\quad ') + ' \\end{aligned}';
  const read = outputRead(classify, lossName);
  const sigmoid = read.shape === 'sigmoid';
  return (
    <>
      <DetailHead eyebrow="Output" title={'ŷ = ' + read.text + ', z = v·a + b'} shape={v.length + ' weights + bias · ' + read.shape} />
      <Formula tex={sum} />
      <Formula tex={'\\hat{y} = ' + read.tex} />
      <Annotation>
        {(hidden === 0
          ? sigmoid
            ? 'Logistic regression: a weighted sum of the inputs read through a sigmoid; z = 0 is its straight boundary.'
            : 'A weighted sum of the inputs, a straight line or plane.'
          : 'Each sheet times its weight v, all added, plus b: the creases of the sheets become the bends of z.') +
          (sigmoid && hidden > 0 ? ' The sigmoid only squashes z into (0, 1), so ŷ = ½ exactly where z = 0.' : '') +
          (read.shape === 'one z per class' ? ' Each class has its own row of v and b; this is the first class’s.' : '')}
      </Annotation>
      <Scalars>
        <Scalar label="‖v‖" value={Math.hypot(...v)} decimals={2} tone="accent" />
        <Scalar label="b" value={b} decimals={2} />
        <Scalar label="max |v|" value={Math.max(...v.map(Math.abs))} decimals={2} />
      </Scalars>
    </>
  );
}
