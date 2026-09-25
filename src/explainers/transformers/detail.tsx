/** Cards for the flow view: a token's input vector, one query's attention, the residual add, the feed-forward block and the output softmax. */

import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { Annotation, DetailHead, Formula, Scalar, Scalars } from '../../explainer/components/Detail';
import { texNum } from '../../explainer/tex';
import { usePalette } from '../../components/ThemeProvider';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt, fmtPercent } from '../../lib/math/stats';
import { headWidth } from '../../lib/ml/transformer';
import type { TransformerCache, TransformerSpec } from '../../lib/ml/transformer';
import { rgba } from '../../lib/viz/palette';
import type { HeadPick } from './attention';
import { headColour } from './draw';
import { NODE, parseNode, rowNorm, rowOf, tokenName, weightAt } from './model';
import type { Probe } from './model';

export interface TfDetailProps {
  selection: ArchSelection;
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  probe: Probe;
  jump: (next: ArchSelection) => void;
  /** Open a query row in the head view. */
  inspect: (pick: HeadPick) => void;
  /** Tables of numbers; otherwise vectors are colour strips. */
  numbers: boolean;
}

/** A vector as a row of coloured cells, blue positive and orange negative. */
function VectorRow({ label, values, scale }: { label: string; values: ArrayLike<number>; scale: number }) {
  const palette = usePalette();
  return (
    <div className="mlx-tf-vec">
      <span className="mlx-tf-vec__label">{label}</span>
      <span className="mlx-tf-vec__cells" style={{ gridTemplateColumns: 'repeat(' + values.length + ', 1fr)' }}>
        {Array.from(values, (v, i) => (
          <span key={i} title={fmt(v, 3)} style={{ background: rgba(v < 0 ? palette.negative : palette.positive, 0.08 + 0.8 * Math.min(1, Math.abs(v) / (scale || 1))) }} />
        ))}
      </span>
    </div>
  );
}

function Jump({ label, title, onClick }: { label: string; title: string; onClick: () => void }) {
  return (
    <button type="button" className="mlx-scalar mlx-scalar--jump" title={title} onClick={onClick}>
      <span className="mlx-scalar__label">{label}</span>
    </button>
  );
}

/** A tinted number cell. */
function Cell({ value, scale, decimals = 2 }: { value: number; scale: number; decimals?: number }) {
  const palette = usePalette();
  const t = Math.min(1, Math.abs(value) / (scale || 1));
  return (
    <td>
      <span className="mlx-vtable__cell" style={{ background: rgba(value < 0 ? palette.negative : palette.positive, 0.08 + 0.5 * t) }}>
        {fmt(value, decimals)}
      </span>
    </td>
  );
}

function maxAbs(...rows: ArrayLike<number>[]): number {
  let top = 0;
  for (const row of rows) for (let i = 0; i < row.length; i++) top = Math.max(top, Math.abs(row[i]));
  return top || 1;
}

export function TfDetail({ selection, data, spec, cache, probe, jump, inspect, numbers }: TfDetailProps) {
  const palette = usePalette();
  const { kind, layer, position: t } = parseNode(selection.id);
  const d = spec.width;
  const name = tokenName(data, cache.tokens[t]);
  const open = (id: string) => jump({ kind: 'node', id });
  if (t >= cache.length) return null;

  if (kind === 'tok' || kind === 'emb') {
    const e = rowOf(cache.embed, t, d);
    const p = rowOf(cache.pos, t, d);
    const x = rowOf(cache.x0, t, d);
    const scale = maxAbs(e, p, x);
    return (
      <>
        <DetailHead eyebrow={'Input · position ' + t + ' of ' + cache.length} title={name} shape={'d = ' + d + ' · token vector + position vector'} />
        <Formula tex={'x_{' + t + '} = E[\\text{' + name.replace(/[▸]/g, 'start') + '}] + p_{' + t + '}'} />
        <Annotation>
          {spec.positions === 'none'
            ? 'No position is added: the same token looks the same wherever it sits.'
            : spec.positions === 'sinusoidal'
              ? 'The position vector is a set of fixed waves: the first numbers swing fast from position to position, the last ones hardly move.'
              : 'The position vector is learned, one row of a table per position.'}
        </Annotation>
        {numbers ? (
          <div className="mlx-table__scroll mlx-tf-table">
            <table className="mlx-wsum mlx-num">
              <thead>
                <tr>
                  <th scope="col">dim</th>
                  <th scope="col">e</th>
                  <th scope="col">p</th>
                  <th scope="col">x</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: d }, (_, i) => (
                  <tr key={i}>
                    <th scope="row">{i + 1}</th>
                    <Cell value={e[i]} scale={scale} />
                    <Cell value={p[i]} scale={scale} />
                    <Cell value={x[i]} scale={scale} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="mlx-tf-vecs">
            <VectorRow label="e" values={e} scale={scale} />
            <VectorRow label="+ p" values={p} scale={scale} />
            <VectorRow label="= x" values={x} scale={scale} />
          </div>
        )}
        <Scalars>
          <Scalar label="|e|" value={rowNorm(cache.embed, t, d)} decimals={2} />
          <Scalar label="|p|" value={rowNorm(cache.pos, t, d)} decimals={2} />
          <Scalar label="|x|" value={rowNorm(cache.x0, t, d)} decimals={2} tone="accent" />
          <Jump label="attention" title="What this token looks at" onClick={() => open(NODE.attn(0, t))} />
        </Scalars>
      </>
    );
  }

  if (kind === 'attn') {
    const dh = headWidth(spec);
    const T = cache.length;
    return (
      <>
        <DetailHead eyebrow={'Attention' + (spec.layers > 1 ? ' · layer ' + (layer + 1) : '') + ' · position ' + t} title={name + ' looks at'} shape={spec.heads + (spec.heads > 1 ? ' heads' : ' head') + ' of ' + dh + ' · ' + (spec.causal ? 'causal' : 'no mask')} />
        <Formula tex={'\\begin{aligned} a_{' + t + 'j} &= \\operatorname{softmax}_j\\Big(\\frac{q_{' + t + '} \\cdot k_j}{\\sqrt{' + dh + '}}\\Big) \\\\ z_{' + t + '} &= \\textstyle\\sum_j a_{' + t + 'j}\\, v_j \\end{aligned}'} />
        <div className="mlx-table__scroll mlx-tf-table">
          <table className="mlx-wsum mlx-num">
            <thead>
              <tr>
                <th scope="col">key</th>
                {Array.from({ length: spec.heads }, (_, h) => (
                  <th key={h} scope="col" style={{ color: headColour(palette, h) }}>
                    {'head ' + (h + 1)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: T }, (_, j) => (
                <tr key={j} data-dim={spec.causal && j > t ? '' : undefined}>
                  <th scope="row">{tokenName(data, cache.tokens[j]) + ' (' + j + ')'}</th>
                  {Array.from({ length: spec.heads }, (_, h) => {
                    const w = weightAt(cache, layer, h, t, j);
                    return (
                      <td key={h}>
                        {spec.causal && j > t ? (
                          <span className="mlx-tf-masked">masked</span>
                        ) : (
                          <span className="mlx-vtable__cell mlx-tf-swatch" title={w.toFixed(3)} style={{ background: rgba(headColour(palette, h), 0.06 + 0.6 * Math.sqrt(w)) }}>
                            {numbers ? w.toFixed(2) : ''}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Annotation>Each column sums to 1: a head spends one unit of attention across the tokens it may see, and hands back that blend of their values.</Annotation>
        <Scalars>
          {Array.from({ length: spec.heads }, (_, h) => (
            <Jump key={h} label={'inside head ' + (h + 1)} title="Every term of this row: q·k, the scale, the softmax and the values" onClick={() => inspect({ layer, head: h, query: t })} />
          ))}
          <Jump label="+ attention" title="What attention adds to the stream" onClick={() => open(NODE.add(layer, t))} />
        </Scalars>
      </>
    );
  }

  if (kind === 'add') {
    const c = cache.layers[layer];
    const before = rowNorm(c.x, t, d);
    const delta = rowNorm(c.att, t, d);
    const after = rowNorm(c.mid, t, d);
    return (
      <>
        <DetailHead eyebrow={'Residual' + (spec.layers > 1 ? ' · layer ' + (layer + 1) : '') + ' · position ' + t} title={name + ': stream + attention'} shape={'W_O mixes ' + spec.heads + ' × ' + headWidth(spec) + ' back to d = ' + d} />
        <Formula tex={"x' = x + W_O\\,[z^{(1)}" + (spec.heads > 1 ? ';\\ldots;z^{(' + spec.heads + ')}' : '') + '] + b_O'} />
        <Annotation>Attention adds to the stream instead of replacing it, so what the token was is still there for the layers after.</Annotation>
        <Scalars>
          <Scalar label="|x|" value={before} decimals={2} />
          <Scalar label="|Δ|" value={delta} decimals={2} tone="accent" />
          <Scalar label="|x′|" value={after} decimals={2} />
          <Scalar label="Δ / x" value={fmtPercent(delta / (before || 1), 0)} />
        </Scalars>
        <Scalars>
          <Jump label="attention" title="The weights behind the added vector" onClick={() => open(NODE.attn(layer, t))} />
          {spec.ffn > 0 ? <Jump label="feed-forward" title="The next block" onClick={() => open(NODE.ffn(layer, t))} /> : <Jump label="prediction" title="The output at this position" onClick={() => open(NODE.pred(t))} />}
        </Scalars>
      </>
    );
  }

  if (kind === 'ffn') {
    const c = cache.layers[layer];
    if (!c.pre || !c.act || !c.ff) return null;
    const f = spec.ffn;
    const units = Array.from({ length: f }, (_, u) => ({ u, pre: c.pre![t * f + u], act: c.act![t * f + u] }));
    const on = units.filter((x) => x.act > 0).length;
    const top = units.slice().sort((a, b) => b.act - a.act).slice(0, 6);
    return (
      <>
        <DetailHead eyebrow={'Feed-forward' + (spec.layers > 1 ? ' · layer ' + (layer + 1) : '') + ' · position ' + t} title={on + ' of ' + f + ' units on'} shape={d + ' → ' + f + ' relu → ' + d + ' · the same for every token'} />
        <Formula tex={'f = W_2 \\max(0,\\; W_1 \\hat{x} + b_1) + b_2'} />
        <Annotation>Attention moved information between tokens; this block works on each token alone, turning what it gathered into something the output can read.</Annotation>
        <table className="mlx-wsum mlx-num">
          <thead>
            <tr>
              <th scope="col">unit</th>
              <th scope="col">W₁x̂ + b₁</th>
              <th scope="col">relu</th>
            </tr>
          </thead>
          <tbody>
            {top.map((x) => (
              <tr key={x.u}>
                <th scope="row">{'u' + (x.u + 1)}</th>
                <td>{fmt(x.pre, 2)}</td>
                <td>{fmt(x.act, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Scalars>
          <Scalar label="|f|" value={rowNorm(c.ff, t, d)} decimals={2} tone="accent" />
          <Scalar label="|x′|" value={rowNorm(c.mid, t, d)} decimals={2} />
          <Jump label="prediction" title="The output at this position" onClick={() => open(NODE.pred(t))} />
        </Scalars>
      </>
    );
  }

  if (kind === 'pred') {
    const V = spec.outVocab;
    const probs = cache.probs.subarray(t * V, t * V + V);
    const logits = cache.logits.subarray(t * V, t * V + V);
    const called = cache.predicted[t];
    const target = probe.targets[t] ?? -1;
    const order = Array.from({ length: V }, (_, c) => c).sort((a, b) => probs[b] - probs[a]);
    const shown = order.slice(0, 6);
    if (target >= 0 && !shown.includes(target)) shown.push(target);
    let total = 0;
    for (let c = 0; c < V; c++) total += Math.exp(logits[c]);
    return (
      <>
        <DetailHead
          eyebrow={'Output · position ' + t}
          title={data.outVocab[called] ?? '?'}
          shape={'p = ' + fmt(probs[called], 2) + (target >= 0 ? ' · answer ' + (data.outVocab[target] ?? '?') + (called === target ? ' · right' : ' · wrong') : ' · nothing to predict here')}
        />
        <Formula tex={'p = \\frac{e^{z}}{\\sum_c e^{z_c}} = \\frac{e^{' + texNum(logits[called]) + '}}{' + texNum(total) + '} = ' + texNum(probs[called], 3)} />
        <table className="mlx-wsum mlx-num">
          <thead>
            <tr>
              <th scope="col">{data.nextToken ? 'next' : 'answer'}</th>
              <th scope="col">z</th>
              <th scope="col">p</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => (
              <tr key={c} data-tone={c === called ? 'accent' : undefined}>
                <th scope="row">{(data.outVocab[c] ?? '?') + (c === target ? ' ✓' : '')}</th>
                <td>{fmt(logits[c], 2)}</td>
                <td>{fmt(probs[c], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.nextToken && !spec.causal && t + 1 < cache.length ? <Annotation>No mask: this position can see the word after it, so it can copy the answer instead of predicting it.</Annotation> : null}
        <Scalars>
          <Jump label="attention" title="What this position looked at" onClick={() => open(NODE.attn(spec.layers - 1, t))} />
        </Scalars>
      </>
    );
  }
  return null;
}
