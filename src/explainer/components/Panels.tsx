/** Presentational blocks: panels, metrics, legends, maths. */

import { useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import katex from 'katex';

import type { Metric } from '../types';
import { LessonFocusContext, useLessonTarget } from '../lessonFocus';
import { useModalLayer } from '../useModalLayer';
import { ChartFillContext } from './Chart';
import { StudioLayoutContext } from './Studio';
import { Button, InfoTip } from './Controls';
import { IconChevron, IconCollapse, IconExpand } from './Icons';

/* ---------------- panel ---------------- */

export interface PanelProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Rendered on the right of the header, usually a legend or a toggle. */
  actions?: ReactNode;
  children: ReactNode;
  /** Removes the inner padding, for panels that are entirely a chart. */
  flush?: boolean;
  className?: string;
  /** Offer a button that gives the panel the whole screen. On by default. */
  expandable?: boolean;
  /** Share the column's height with sibling panels. Only while the studio is locked to the viewport. */
  fill?: boolean;
  /** Lets a lesson step point at this panel. */
  id?: string;
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  flush,
  className,
  expandable = true,
  fill,
  id,
}: PanelProps) {
  const [full, setFull] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const { locked } = useContext(StudioLayoutContext);
  const fills = Boolean(fill) && locked;
  useModalLayer(full, sectionRef);
  const target = useLessonTarget('panel', id ?? '');

  useEffect(() => {
    if (!full) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setFull(false);
    };
    window.addEventListener('keydown', onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = previous;
    };
  }, [full]);

  const canExpand = expandable;
  const headRef = useRef<HTMLElement>(null);
  useFittedHead(headRef);

  return (
    <section
      ref={sectionRef}
      tabIndex={full ? -1 : undefined}
      {...(id ? target.attrs : {})}
      className={
        'mlx-panel' +
        (full ? ' mlx-panel--full' : '') +
        (fills && !full ? ' mlx-panel--fill' : '') +
        (className ? ' ' + className : '')
      }
      role={full ? 'dialog' : undefined}
      aria-modal={full || undefined}
      aria-label={full ? (typeof title === 'string' ? title : 'Full screen panel') : undefined}
    >
      {title || actions || canExpand ? (
        <header ref={headRef} className="mlx-panel__head">
          <div className="mlx-panel__titles">
            {title ? <h3 className="mlx-panel__title">{title}</h3> : null}
            {subtitle ? (
              <div className="mlx-panel__subtitle" title={typeof subtitle === 'string' ? subtitle : undefined}>
                {subtitle}
              </div>
            ) : null}
          </div>
          {actions ? <div className="mlx-panel__actions">{actions}</div> : null}
          {canExpand ? (
            <button
              type="button"
              className="mlx-panel__expand"
              onClick={() => setFull((v) => !v)}
              aria-label={full ? 'Back to the page' : 'View full screen'}
              title={full ? 'Back to the page (Esc)' : 'View full screen'}
            >
              {full ? <IconCollapse size={14} /> : <IconExpand size={14} />}
            </button>
          ) : null}
        </header>
      ) : null}
      <ChartFillContext.Provider value={full || fills}>
        <div className={'mlx-panel__body' + (flush ? ' mlx-panel__body--flush' : '')}>{children}</div>
      </ChartFillContext.Provider>
    </section>
  );
}

/** Sets data-fit on the header while its actions fit beside a one-line title. */
function useFittedHead(ref: React.RefObject<HTMLElement>) {
  useLayoutEffect(() => {
    const head = ref.current;
    if (!head || typeof ResizeObserver === 'undefined') return undefined;
    liveHeads.add(head);
    scheduleFit(head, false);
    const resize = new ResizeObserver(() => scheduleFit(head, true));
    resize.observe(head);
    const mutate = new MutationObserver(() => scheduleFit(head, false));
    mutate.observe(head, { childList: true, characterData: true, subtree: true });
    return () => {
      liveHeads.delete(head);
      pendingFits.delete(head);
      resize.disconnect();
      mutate.disconnect();
    };
  }, [ref]);
}

// Every header measures in one batch: all the writes, then all the reads, so a page of panels costs one layout.
const liveHeads = new Set<HTMLElement>();
const pendingFits = new Set<HTMLElement>();
let fitFlushQueued = false;

// A web font landing after first paint changes what fits.
if (typeof document !== 'undefined') {
  document.fonts?.addEventListener('loadingdone', () => liveHeads.forEach((head) => scheduleFit(head, false)));
}

function scheduleFit(head: HTMLElement, defer: boolean) {
  pendingFits.add(head);
  if (fitFlushQueued) return;
  fitFlushQueued = true;
  // A resize callback must not relayout synchronously; a frame later is fine.
  if (defer) requestAnimationFrame(flushFits);
  else queueMicrotask(flushFits);
}

function flushFits() {
  fitFlushQueued = false;
  const heads = [...pendingFits];
  pendingFits.clear();
  const jobs = heads.flatMap((head) => {
    const title = head.querySelector<HTMLElement>('.mlx-panel__title');
    const actions = head.querySelector<HTMLElement>('.mlx-panel__actions');
    if (!title || !actions) {
      head.removeAttribute('data-fit');
      return [];
    }
    head.setAttribute('data-fit', '');
    return [{ head, title }];
  });
  const fits = jobs.map(({ title }) => title.scrollWidth <= title.clientWidth);
  jobs.forEach(({ head }, i) => head.toggleAttribute('data-fit', fits[i]));
}

/* ---------------- metrics ---------------- */

export function MetricGrid({ metrics, columns }: { metrics: Metric[]; columns?: number }) {
  return (
    <div
      className="mlx-metrics"
      style={columns ? { ['--mlx-metric-columns' as string]: String(columns) } : undefined}
    >
      {metrics.map((metric) => (
        <MetricTile key={metric.key} metric={metric} />
      ))}
    </div>
  );
}

export function MetricTile({ metric }: { metric: Metric }) {
  const target = useLessonTarget('metric', metric.key);
  return (
    <div className="mlx-metric" data-tone={metric.tone ?? 'neutral'} {...target.attrs}>
      <div className="mlx-metric__label">
        {metric.label}
        {metric.help ? <InfoTip text={metric.help} label={metric.label} /> : null}
      </div>
      <div
        className="mlx-metric__value mlx-num"
        data-empty={metric.value === 'n/a' || undefined}
      >
        {metric.value}
        {metric.trend && metric.trend !== 'flat' ? (
          <span className="mlx-metric__trend" data-dir={metric.trend} aria-hidden="true">
            {metric.trend === 'down' ? '↓' : '↑'}
          </span>
        ) : null}
      </div>
      {metric.caption ? <div className="mlx-metric__caption">{metric.caption}</div> : null}
    </div>
  );
}

/** The column's numbers: headline values on one line, the rest behind More. */
export function MetricStrip({ headline, more }: { headline: Metric[]; more: Metric[] }) {
  const [open, setOpen] = useState(false);
  const focus = useContext(LessonFocusContext);
  const id = useId();
  const folded = more.map((m) => m.key).join(',');

  useEffect(() => {
    if (focus?.kind === 'metric' && folded.split(',').includes(focus.key)) setOpen(true);
  }, [focus, folded]);

  return (
    <div className="mlx-strip">
      <div className="mlx-strip__row">
        <div className="mlx-strip__chips">
          {headline.map((metric) => (
            <MetricChip key={metric.key} metric={metric} />
          ))}
        </div>
        {more.length > 0 ? (
          <button
            type="button"
            className="mlx-strip__more"
            aria-expanded={open}
            aria-controls={id}
            onClick={() => setOpen((v) => !v)}
            title={open ? 'Hide the other numbers' : 'Show ' + more.length + ' more numbers'}
          >
            {open ? 'Less' : 'More'}
            <IconChevron size={11} />
          </button>
        ) : null}
      </div>
      {more.length > 0 ? (
        <div id={id} className="mlx-strip__drawer" hidden={!open}>
          <MetricGrid metrics={more} />
        </div>
      ) : null}
    </div>
  );
}

/** One metric on a line: swatch, name, live value. */
function MetricChip({ metric }: { metric: Metric }) {
  const target = useLessonTarget('metric', metric.key);
  return (
    <div className="mlx-chip" data-tone={metric.tone ?? 'neutral'} title={metric.help} {...target.attrs}>
      {metric.colour ? (
        <span className="mlx-chip__mark" style={{ background: metric.colour }} aria-hidden="true" />
      ) : null}
      <span className="mlx-chip__label">{metric.label}</span>
      <span className="mlx-chip__value mlx-num" data-empty={metric.value === 'n/a' || undefined}>
        {metric.value}
        {metric.trend && metric.trend !== 'flat' ? (
          <span className="mlx-metric__trend" data-dir={metric.trend} aria-hidden="true">
            {metric.trend === 'down' ? '↓' : '↑'}
          </span>
        ) : null}
      </span>
    </div>
  );
}

/* ---------------- legend ---------------- */

export interface LegendItem {
  label: string;
  colour: string;
  /** How the mark is drawn in the chart, so the legend matches it exactly. */
  shape?: 'dot' | 'square' | 'line' | 'dashed' | 'cross' | 'triangle' | 'ring' | 'target';
  hint?: string;
}

export function Legend({ items, dense }: { items: LegendItem[]; dense?: boolean }) {
  return (
    <ul className={'mlx-legend' + (dense ? ' mlx-legend--dense' : '')}>
      {items.map((item) => (
        <li key={item.label} className="mlx-legend__item" title={item.hint}>
          <span
            className="mlx-legend__mark"
            data-shape={item.shape ?? 'dot'}
            style={{ ['--mlx-legend-colour' as string]: item.colour }}
            aria-hidden="true"
          />
          <span>{item.label}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---------------- switches ---------------- */

/** A tiny segmented switch for panel headers: which of two views to draw. */
export function ViewSwitch({
  label,
  value,
  options,
  onChange,
  id,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  /** Lets a lesson step point at this switch. */
  id?: string;
}) {
  const target = useLessonTarget('view', id ?? '');
  return (
    <div className="mlx-viewswitch mlx-segmented" role="group" aria-label={label} {...(id ? target.attrs : {})}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="mlx-segmented__item"
          data-active={option.value === value || undefined}
          aria-pressed={option.value === value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** Back to the page's defaults, in the title strip. */
export function ParamsReset({ onReset, isDirty, title = 'Back to defaults' }: { onReset: () => void; isDirty: boolean; title?: string }) {
  return (
    <Button size="sm" variant="ghost" onClick={onReset} softDisabled={!isDirty} title={title}>
      Reset
    </Button>
  );
}

/* ---------------- maths ---------------- */

/** KaTeX rendering, typeset during render so the formula is in the first frame. Failures fall back to the raw source. */
export function Math({
  tex,
  display = false,
  label,
}: {
  tex: string;
  display?: boolean;
  label?: string;
}) {
  const html = useMemo(() => {
    try {
      // MathML rides along, hidden, so a screen reader reads the maths instead of the TeX source.
      return katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        strict: false,
        output: 'htmlAndMathml',
      });
    } catch {
      return null;
    }
  }, [tex, display]);

  const inner =
    html === null ? (
      <span aria-label={label ?? tex} role="math">
        {tex}
      </span>
    ) : label ? (
      <span aria-label={label} role="math" dangerouslySetInnerHTML={{ __html: html }} />
    ) : (
      <span dangerouslySetInnerHTML={{ __html: html }} />
    );

  if (display) return <span className="mlx-math mlx-math--display">{inner}</span>;
  return <span className="mlx-math">{inner}</span>;
}
