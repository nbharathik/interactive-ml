/** Cards for the convolutional scene: the sliding window behind a map pixel, a pooling block, a dense unit's sum, the softmax. */

import { useCallback, useEffect, useState } from 'react';
import type { ReactNode } from 'react';

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailHead, Formula, Scalar, Scalars, WeightedSumTable } from '../../explainer/components/Detail';
import type { WeightedRow } from '../../explainer/components/Detail';
import { texNum } from '../../explainer/tex';
import { usePalette } from '../../components/ThemeProvider';
import { fmt, fmtCell, fmtPercent } from '../../lib/math/stats';
import { channel, filterTensor } from '../../lib/ml/conv';
import type { CnnCache, CnnParams, CnnSpec } from '../../lib/ml/conv';
import { rgba } from '../../lib/viz/palette';
import { drawPixelGrid } from '../../lib/viz/plots';
import { NODE, brightest, parseEdge, parseNode, sub, weightEdge, windowTerms } from './scene';
import type { CnnScene, Picture, Window } from './scene';
import type { Slide } from './useSlide';

export interface CnnDetailProps {
  selection: ArchSelection;
  scene: CnnScene;
  params: CnnParams;
  cache: CnnCache;
  spec: CnnSpec;
  classNames: readonly string[];
  classSymbols: readonly string[];
  /** The cell whose block is drawn on the diagram; the card moves it. */
  window: Window | null;
  onWindow: (window: Window | null) => void;
  /** The page's slide, shared with the window view. */
  slide: Slide;
  jump: (next: ArchSelection) => void;
}

/* ---------------- pieces ---------------- */

/** A small grid of numbers, tinted by sign or size. */
function NumberGrid({ values, cols, signed, scale }: { values: ArrayLike<number>; cols: number; signed: boolean; scale?: number }) {
  const palette = usePalette();
  let top = scale ?? 0;
  if (!top) for (let i = 0; i < values.length; i++) top = Math.max(top, Math.abs(values[i]));
  if (!top) top = 1;
  return (
    <div className="mlx-pixels" style={{ gridTemplateColumns: 'repeat(' + cols + ', 1fr)' }}>
      {Array.from({ length: values.length }, (_, i) => {
        const v = values[i];
        const t = Math.min(1, Math.abs(v) / top);
        const colour = signed && v < 0 ? palette.negative : signed ? palette.positive : palette.text;
        return (
          <span key={i} className="mlx-pixels__cell mlx-num" style={{ background: rgba(colour, 0.08 + 0.55 * t) }}>
            {fmt(v, 2)}
          </span>
        );
      })}
    </div>
  );
}

interface Highlight {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A picture with an optional highlighted block; hovering reports the cell under the pointer. */
function PictureView({
  picture,
  highlight,
  onCell,
  onLeave,
  size = 84,
  description,
}: {
  picture: Picture;
  highlight?: Highlight | null;
  onCell?: (x: number, y: number) => void;
  onLeave?: () => void;
  size?: number;
  description: string;
}) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const cell = Math.min(width / picture.cols, height / picture.rows);
      const left = (width - cell * picture.cols) / 2;
      const top = (height - cell * picture.rows) / 2;
      ctx.fillStyle = palette.surfaceAlt;
      ctx.fillRect(left, top, cell * picture.cols, cell * picture.rows);
      drawPixelGrid(ctx, { x: left, y: top, width: cell * picture.cols, height: cell * picture.rows }, picture.values, picture.cols, picture.rows, palette, { ramp: picture.ramp, gap: 0 });
      if (highlight) {
        const x0 = Math.max(0, highlight.x);
        const y0 = Math.max(0, highlight.y);
        const x1 = Math.min(picture.cols, highlight.x + highlight.w);
        const y1 = Math.min(picture.rows, highlight.y + highlight.h);
        ctx.strokeStyle = palette.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(left + x0 * cell, top + y0 * cell, (x1 - x0) * cell, (y1 - y0) * cell);
      }
    },
    [picture, highlight],
  );
  const locate = (pos: { x: number; y: number }) => {
    const cell = Math.min(size / picture.cols, size / picture.rows);
    const left = (size - cell * picture.cols) / 2;
    const top = (size - cell * picture.rows) / 2;
    const x = Math.floor((pos.x - left) / cell);
    const y = Math.floor((pos.y - top) / cell);
    return x >= 0 && y >= 0 && x < picture.cols && y < picture.rows ? { x, y } : null;
  };
  return (
    <div className="mlx-cnn-picture" style={{ width: size, height: size }}>
      <Chart
        draw={draw}
        height={size}
        description={description}
        cursor={onCell ? 'crosshair' : 'default'}
        onPointerMove={
          onCell
            ? (pos) => {
                const at = pos ? locate(pos) : null;
                if (at) onCell(at.x, at.y);
                else onLeave?.();
              }
            : undefined
        }
        onPointerLeave={onLeave}
        redrawKey={(highlight ? highlight.x + ',' + highlight.y : '') + ':' + picture.values.length}
      />
    </div>
  );
}

/** Pictures with operator symbols between them. */
function Flow({ children }: { children: ReactNode }) {
  return <div className="mlx-cnn-flow">{children}</div>;
}

function Op({ children }: { children: ReactNode }) {
  return (
    <span className="mlx-cnn-flow__op" aria-hidden="true">
      {children}
    </span>
  );
}

function Jump({ label, value, title, onClick }: { label: string; value?: string; title: string; onClick: () => void }) {
  return (
    <button type="button" className="mlx-scalar mlx-scalar--jump" title={title} onClick={onClick}>
      <span className="mlx-scalar__label">{label}</span>
      {value !== undefined ? <span className="mlx-scalar__value">{value}</span> : null}
    </button>
  );
}

/* ---------------- the cards ---------------- */

export function CnnDetail(props: CnnDetailProps) {
  const { scene, params, cache, spec, classNames, classSymbols, jump } = props;
  const open = (id: string) => jump({ kind: 'node', id });
  // An operation wire opens the card of what it produces.
  const selection = props.selection.kind === 'edge' && !parseEdge(props.selection.id) ? { kind: 'node' as const, id: props.selection.id.split('->')[1] ?? '' } : props.selection;

  if (selection.kind === 'edge') {
    const edge = parseEdge(selection.id);
    if (!edge || !params.hidden || !cache.hiddenA) return null;
    const w = params.out.W[edge.cls * params.out.inSize + edge.unit] ?? 0;
    const h = cache.hiddenA[edge.unit] ?? 0;
    return (
      <>
        <DetailHead eyebrow="Weight" title="w" shape={'h' + sub(edge.unit + 1) + ' → ' + (classSymbols[edge.cls] ?? '') + ' ' + (classNames[edge.cls] ?? '')} />
        <Formula tex={'w \\cdot h_{' + (edge.unit + 1) + '} = ' + texNum(w) + ' \\times ' + texNum(h) + ' = ' + texNum(w * h)} />
        <Scalars>
          <Scalar label="w" value={w} />
          <Scalar label={'h' + sub(edge.unit + 1)} value={h} />
          <Scalar label="product" value={w * h} tone="accent" />
          <Jump label={'h' + sub(edge.unit + 1)} title="Open the unit" onClick={() => open(NODE.unit(edge.unit))} />
          <Jump label={classSymbols[edge.cls] ?? ''} title="Open the class" onClick={() => open(NODE.out(edge.cls))} />
        </Scalars>
      </>
    );
  }

  const { kind, stage, index } = parseNode(selection.id);
  const node = scene.byId.get(selection.id);
  if (!node) return null;

  if (kind === 'image') {
    const values = cache.input.data;
    let sum = 0;
    let lit = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
      if (values[i] > 0.5) lit++;
    }
    const kernels = scene.layers.find((l) => l.kind === 'filters' && l.stage === 0)?.nodes ?? [];
    return (
      <>
        <DetailHead eyebrow="Input" title="x" shape={spec.size + ' × ' + spec.size + ' · 1 channel · [0, 1]'} />
        <div className="mlx-detail__row">
          <PictureView picture={node.picture!} size={120} description={'The probe image, a ' + (classNames[scene.label] ?? 'glyph') + '.'} />
          <Scalars>
            <Scalar label="class" value={(classSymbols[scene.label] ?? '') + ' ' + (classNames[scene.label] ?? '')} />
            <Scalar label="mean" value={sum / values.length} />
            <Scalar label="> 0.5" value={lit + ' px'} />
          </Scalars>
        </div>
        <Scalars>
          {kernels.map((id) => (
            <Jump key={id} label={scene.byId.get(id)?.label ?? ''} title="Open the kernel" onClick={() => open(id)} />
          ))}
          {kernels.length === 0 ? <Jump label="f" title="Open the flattened pixels" onClick={() => open(NODE.flat)} /> : null}
        </Scalars>
      </>
    );
  }

  if (kind === 'filter') return <KernelCard {...props} selection={selection} stage={stage} index={index} />;
  if (kind === 'map') return <MapCard {...props} selection={selection} stage={stage} index={index} />;
  if (kind === 'pool') return <PoolCard {...props} selection={selection} stage={stage} index={index} />;

  if (kind === 'flat') {
    const values = cache.flat;
    let active = 0;
    let top = 0;
    for (let i = 0; i < values.length; i++) {
      if (values[i] > 0) active++;
      top = Math.max(top, values[i]);
    }
    const last = cache.stages[cache.stages.length - 1];
    const shape = last ? last.pooled.c + ' × ' + last.pooled.h + ' × ' + last.pooled.w + ' → ' + values.length : spec.size + ' × ' + spec.size + ' → ' + values.length;
    const sources = scene.edges.filter((e) => e.to === NODE.flat).map((e) => e.from);
    return (
      <>
        <DetailHead eyebrow="Flatten" title="f" shape={shape} />
        <Formula tex={last ? 'f = \\operatorname{vec}(p_1, \\ldots, p_{' + last.pooled.c + '})' : 'f = \\operatorname{vec}(x)'} />
        <Scalars>
          <Scalar label="values" value={values.length} decimals={0} />
          <Scalar label="> 0" value={active} decimals={0} />
          <Scalar label="max" value={top} tone="accent" />
        </Scalars>
        <Scalars>
          {sources.map((id) => (
            <Jump key={id} label={scene.byId.get(id)?.label ?? ''} title="Open it" onClick={() => open(id)} />
          ))}
        </Scalars>
      </>
    );
  }

  if (kind === 'unit' && params.hidden && cache.hiddenZ && cache.hiddenA) {
    const hidden = params.hidden;
    const z = cache.hiddenZ[index] ?? 0;
    const h = cache.hiddenA[index] ?? 0;
    const rows: WeightedRow[] = [];
    for (let j = 0; j < hidden.inSize; j++) rows.push({ name: 'f' + sub(j + 1), value: cache.flat[j], weight: hidden.W[index * hidden.inSize + j] });
    rows.sort((a, b) => Math.abs(b.value * b.weight) - Math.abs(a.value * a.weight));
    const shown = rows.slice(0, 8);
    return (
      <>
        <DetailHead eyebrow={'Dense · unit ' + (index + 1) + ' / ' + hidden.outSize} title={'h' + sub(index + 1)} shape={hidden.inSize + ' w · 1 b · relu'} />
        <Formula tex={'h = \\max(0,\\; b + \\textstyle\\sum_j w_j f_j) = \\max(0,\\; ' + texNum(z) + ') = ' + texNum(h)} />
        <Scalars>
          <Scalar label="z" value={z} />
          <Scalar label="h" value={h} tone={h > 0 ? 'accent' : undefined} />
          <Scalar label="b" value={hidden.b[index]} />
        </Scalars>
        <WeightedSumTable rows={shown} columns={['f', 'value', 'w', 'product']} noTotal result={{ label: 'largest ' + shown.length + ' of ' + rows.length + ' terms', value: fmt(shown.reduce((a, r) => a + r.value * r.weight, 0), 3) }} />
        <Scalars>
          {Array.from({ length: params.out.outSize }, (_, c) => (
            <Jump
              key={c}
              label={'→ ' + (classSymbols[c] ?? c)}
              value={fmtCell(params.out.W[c * params.out.inSize + index], 2, true)}
              title={'Weight to ' + (classNames[c] ?? 'class ' + c)}
              onClick={() => jump({ kind: 'edge', id: weightEdge(index, c) })}
            />
          ))}
        </Scalars>
      </>
    );
  }

  if (kind === 'out') {
    const z = cache.logits[index] ?? 0;
    const p = cache.probs[index] ?? 0;
    let total = 0;
    for (let c = 0; c < cache.logits.length; c++) total += Math.exp(cache.logits[c]);
    const head = cache.hiddenA ?? cache.flat;
    const rows: WeightedRow[] = [];
    for (let u = 0; u < params.out.inSize; u++) rows.push({ name: (cache.hiddenA ? 'h' : 'f') + sub(u + 1), value: head[u], weight: params.out.W[index * params.out.inSize + u] });
    const all = rows.length <= 16;
    const shown = all ? rows : rows.slice().sort((a, b) => Math.abs(b.value * b.weight) - Math.abs(a.value * a.weight)).slice(0, 8);
    return (
      <>
        <DetailHead
          eyebrow={'Scores · class ' + (index + 1) + ' / ' + cache.probs.length}
          title={(classSymbols[index] ?? '') + ' ' + (classNames[index] ?? '')}
          shape={'p = ' + fmt(p, 3) + (index === scene.label ? ' · the true class' : '') + (index === cache.predicted ? ' · called' : '')}
        />
        <Formula tex={'p = \\frac{e^{z}}{\\sum_j e^{z_j}} = \\frac{e^{' + texNum(z) + '}}{' + texNum(total) + '} = ' + texNum(p, 3)} />
        <table className="mlx-wsum mlx-num">
          <thead>
            <tr>
              <th scope="col">class</th>
              <th scope="col">z</th>
              <th scope="col">eᶻ</th>
              <th scope="col">p</th>
            </tr>
          </thead>
          <tbody>
            {Array.from(cache.probs).map((q, c) => (
              <tr key={c} data-tone={c === index ? 'accent' : undefined}>
                <th scope="row">
                  <button type="button" className="mlx-wsum__jump" onClick={() => open(NODE.out(c))} title={classNames[c]}>
                    {classSymbols[c]}
                  </button>
                </th>
                <td>{fmt(cache.logits[c], 2)}</td>
                <td>{fmt(Math.exp(cache.logits[c]), 2)}</td>
                <td>{fmt(q, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <WeightedSumTable
          rows={shown}
          bias={all ? params.out.b[index] : undefined}
          columns={[cache.hiddenA ? 'h' : 'f', 'value', 'w', 'product']}
          sumLabel="z"
          noTotal={!all}
          result={all ? undefined : { label: 'largest ' + shown.length + ' of ' + rows.length + ' terms · z', value: z }}
          onSelectInput={cache.hiddenA ? (i) => open(NODE.unit(shown === rows ? i : rows.indexOf(shown[i]))) : undefined}
          onSelectWeight={cache.hiddenA ? (i) => jump({ kind: 'edge', id: weightEdge(shown === rows ? i : rows.indexOf(shown[i]), index) }) : undefined}
        />
      </>
    );
  }
  return null;
}

/* ---------------- convolution ---------------- */

function KernelCard({ scene, params, cache, stage, index, jump }: CnnDetailProps & { stage: number; index: number }) {
  const layer = params.convs[stage];
  const stageCache = cache.stages[stage];
  const [slice, setSlice] = useState(0);
  if (!layer || !stageCache) return null;
  const k = layer.kernel;
  const inC = layer.inC;
  const picture = inC > 1 ? layer.W.subarray(((index * inC + slice) * k) * k, ((index * inC + slice + 1) * k) * k) : filterTensor(layer, index).data;
  const positions = stageCache.a.h * stageCache.a.w;
  const prime = stage > 0 ? '′' : '';
  const inputs = scene.edges.filter((e) => e.to === NODE.filter(stage, index)).map((e) => e.from);
  return (
    <>
      <DetailHead
        eyebrow={'Kernels' + (stage > 0 ? ' ' + (stage + 1) : '') + ' · kernel ' + (index + 1) + ' / ' + layer.outC}
        title={'k' + sub(index + 1) + prime}
        shape={k + ' × ' + k + (inC > 1 ? ' × ' + inC : '') + ' · ' + k * k * inC + ' w · 1 b'}
      />
      <Formula tex={'z_{y,x} = b + \\sum_{i,j' + (inC > 1 ? ',c' : '') + '} k_{i,j' + (inC > 1 ? ',c' : '') + '}\\, x_{y+i,\\,x+j' + (inC > 1 ? ',c' : '') + '}'} />
      {inC > 1 ? (
        <Scalars>
          {Array.from({ length: inC }, (_, c) => (
            <button key={c} type="button" className="mlx-scalar mlx-scalar--jump" data-tone={c === slice ? 'accent' : undefined} onClick={() => setSlice(c)} title={'The slice that reads ' + (scene.byId.get(inputs[c])?.label ?? 'map ' + (c + 1))}>
              <span className="mlx-scalar__label">{'slice ' + (c + 1)}</span>
              <span className="mlx-scalar__value">{scene.byId.get(inputs[c])?.label ?? ''}</span>
            </button>
          ))}
        </Scalars>
      ) : null}
      <NumberGrid values={picture} cols={k} signed />
      <Scalars>
        <Scalar label="b" value={layer.b[index]} />
        <Scalar label="positions" value={stageCache.a.h + ' × ' + stageCache.a.w + ' = ' + positions} tone="accent" />
      </Scalars>
      <Scalars>
        <Jump label={'m' + sub(index + 1) + prime} title="Open its map" onClick={() => jump({ kind: 'node', id: NODE.map(stage, index) })} />
        {inputs.slice(0, 1).map((id) => (
          <Jump key={id} label={scene.byId.get(id)?.label ?? ''} title="Open what it reads" onClick={() => jump({ kind: 'node', id })} />
        ))}
      </Scalars>
    </>
  );
}

function MapCard({ scene, params, cache, spec, stage, index, window, onWindow, slide, jump }: CnnDetailProps & { stage: number; index: number }) {
  const layer = params.convs[stage];
  const stageCache = cache.stages[stage];
  const [inputChannel, setInputChannel] = useState(0);
  const mine = window !== null && window.kind === 'map' && window.stage === stage && window.index === index;
  const map = stageCache ? channel(stageCache.a, index) : null;
  const cols = stageCache?.a.w ?? 1;
  const rows = stageCache?.a.h ?? 1;
  // The window starts on the brightest pixel, then slides.
  useEffect(() => {
    if (mine || !map) return;
    const spot = brightest(map, cols);
    onWindow({ kind: 'map', stage, index, y: spot.y, x: spot.x });
  }, [mine, map, cols, stage, index, onWindow]);
  if (!layer || !stageCache || !map) return null;
  const at = mine ? window : brightest(map, cols);
  const k = layer.kernel;
  const prime = stage > 0 ? '′' : '';
  const inputs = scene.edges.filter((e) => e.to === NODE.filter(stage, index)).map((e) => scene.byId.get(e.from)!).filter(Boolean);
  const input = inputs[Math.min(inputChannel, inputs.length - 1)];
  const terms = windowTerms(params, cache, { kind: 'map', stage, index, y: at.y, x: at.x }, Math.min(inputChannel, inputs.length - 1));
  const kernelNode = scene.byId.get(NODE.filter(stage, index));
  let active = 0;
  let sum = 0;
  for (let i = 0; i < map.length; i++) {
    if (map[i] > 0) active++;
    sum += map[i];
  }
  const patch: Highlight = { x: at.x - layer.pad, y: at.y - layer.pad, w: k, h: k };
  const mapPicture: Picture = { values: map, cols, rows, ramp: 'magnitude' };
  const kernelPicture: Picture = terms ? { values: terms.terms.map((t) => t.weight), cols: k, rows: k, ramp: 'signed' } : kernelNode!.picture!;
  return (
    <>
      <DetailHead eyebrow={'Maps' + (stage > 0 ? ' ' + (stage + 1) : '') + ' · map ' + (index + 1) + ' / ' + layer.outC} title={'m' + sub(index + 1) + prime} shape={rows + ' × ' + cols + ' · relu(x ∗ k' + sub(index + 1) + prime + ' + b)'} />
      <Flow>
        {input?.picture ? (
          <PictureView picture={input.picture} highlight={patch} description={'What the kernel reads, with the block behind the current pixel outlined.'} />
        ) : null}
        <Op>∗</Op>
        <PictureView picture={kernelPicture} size={44} description="The kernel." />
        <Op>→</Op>
        <PictureView
          picture={mapPicture}
          highlight={{ x: at.x, y: at.y, w: 1, h: 1 }}
          onCell={(x, y) => {
            slide.setHovering(true);
            if (x !== at.x || y !== at.y) onWindow({ kind: 'map', stage, index, y, x });
          }}
          onLeave={() => slide.setHovering(false)}
          description={'The map, with the current pixel outlined. Hover to move it.'}
        />
      </Flow>
      <Scalars>
        <Scalar label="(y, x)" value={at.y + ', ' + at.x} />
        <button type="button" className="mlx-scalar mlx-scalar--jump" onClick={slide.toggle} title={slide.paused ? 'Slide the window' : 'Hold the window'}>
          <span className="mlx-scalar__label">{slide.paused ? 'slide' : 'hold'}</span>
        </button>
        {inputs.length > 1 ? (
          inputs.map((n, c) => (
            <button key={n.id} type="button" className="mlx-scalar mlx-scalar--jump" data-tone={c === inputChannel ? 'accent' : undefined} onClick={() => setInputChannel(c)} title={'Show the terms that read ' + n.label}>
              <span className="mlx-scalar__label">{n.label}</span>
            </button>
          ))
        ) : null}
      </Scalars>
      {terms ? (
        <>
          <div className="mlx-cnn-grids">
            <NumberGrid values={terms.terms.map((t) => t.pixel)} cols={k} signed={false} scale={1} />
            <Op>⊙</Op>
            <NumberGrid values={terms.terms.map((t) => t.weight)} cols={k} signed />
            <Op>=</Op>
            <NumberGrid values={terms.terms.map((t) => t.pixel * t.weight)} cols={k} signed />
          </div>
          <Formula
            tex={
              'z = \\underbrace{' + texNum(terms.terms.reduce((a, t) => a + t.pixel * t.weight, 0)) + '}_{\\sum}' +
              (inputs.length > 1 ? ' + \\underbrace{' + texNum(terms.others) + '}_{\\text{other slices}}' : '') +
              (terms.bias < 0 ? ' - ' : ' + ') + '\\underbrace{' + texNum(Math.abs(terms.bias)) + '}_{b} = ' + texNum(terms.z) +
              ', \\quad \\max(0, z) = ' + texNum(terms.a)
            }
          />
        </>
      ) : null}
      <Scalars>
        <Scalar label="max" value={brightest(map, cols).value} tone="accent" />
        <Scalar label="mean" value={sum / map.length} />
        <Scalar label="> 0" value={fmtPercent(active / map.length, 0)} />
        <Jump label={'k' + sub(index + 1) + prime} title="Open the kernel" onClick={() => jump({ kind: 'node', id: NODE.filter(stage, index) })} />
        {spec.pool !== 'none' ? <Jump label={'p' + sub(index + 1) + prime} title="Open the pooled map" onClick={() => jump({ kind: 'node', id: NODE.pool(stage, index) })} /> : null}
      </Scalars>
    </>
  );
}

function PoolCard({ cache, spec, stage, index, window, onWindow, jump }: CnnDetailProps & { stage: number; index: number }) {
  const stageCache = cache.stages[stage];
  const mine = window !== null && window.kind === 'pool' && window.stage === stage && window.index === index;
  const pool = stageCache ? channel(stageCache.pooled, index) : null;
  const cols = stageCache?.pooled.w ?? 1;
  useEffect(() => {
    if (mine || !pool) return;
    const spot = brightest(pool, cols);
    onWindow({ kind: 'pool', stage, index, y: spot.y, x: spot.x });
  }, [mine, pool, cols, stage, index, onWindow]);
  if (!stageCache || !pool) return null;
  const map = channel(stageCache.a, index);
  const at = mine ? window : brightest(pool, cols);
  const block: number[] = [];
  for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) block.push(map[(2 * at.y + dy) * stageCache.a.w + (2 * at.x + dx)] ?? 0);
  const value = spec.pool === 'max' ? Math.max(...block) : block.reduce((a, b) => a + b, 0) / 4;
  const prime = stage > 0 ? '′' : '';
  const m = 'm_{' + (index + 1) + '}';
  return (
    <>
      <DetailHead eyebrow={'Pooled' + (stage > 0 ? ' ' + (stage + 1) : '') + ' · map ' + (index + 1) + ' / ' + stageCache.pooled.c} title={'p' + sub(index + 1) + prime} shape={stageCache.pooled.h + ' × ' + stageCache.pooled.w + ' · ' + spec.pool + ' of 2 × 2'} />
      <Formula tex={'p_{y,x} = ' + (spec.pool === 'max' ? '\\max' : '\\tfrac{1}{4}\\textstyle\\sum') + '\\big(' + m + '[2y{:}2y{+}1,\\; 2x{:}2x{+}1]\\big)'} />
      <Flow>
        <PictureView picture={{ values: map, cols: stageCache.a.w, rows: stageCache.a.h, ramp: 'magnitude' }} highlight={{ x: 2 * at.x, y: 2 * at.y, w: 2, h: 2 }} description="The map with the current block outlined." />
        <Op>{spec.pool === 'max' ? 'max' : 'mean'}</Op>
        <PictureView
          picture={{ values: pool, cols, rows: stageCache.pooled.h, ramp: 'magnitude' }}
          highlight={{ x: at.x, y: at.y, w: 1, h: 1 }}
          onCell={(x, y) => {
            if (x !== at.x || y !== at.y) onWindow({ kind: 'pool', stage, index, y, x });
          }}
          description="The pooled map with the current cell outlined. Hover to move it."
        />
      </Flow>
      <div className="mlx-cnn-grids">
        <NumberGrid values={block} cols={2} signed={false} />
        <Op>→</Op>
        <Scalar label={spec.pool === 'max' ? 'max' : 'mean'} value={value} tone="accent" />
      </div>
      <Scalars>
        <Scalar label="(y, x)" value={at.y + ', ' + at.x} />
        <Scalar label="cells" value={stageCache.a.h * stageCache.a.w + ' → ' + stageCache.pooled.h * stageCache.pooled.w} />
        <Jump label={'m' + sub(index + 1) + prime} title="Open the map" onClick={() => jump({ kind: 'node', id: NODE.map(stage, index) })} />
      </Scalars>
    </>
  );
}
