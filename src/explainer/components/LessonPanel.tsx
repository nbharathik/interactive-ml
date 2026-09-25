/** The Lesson tab: the open lesson in one card, then the course as an outline. */

import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';

import type { GuideTone } from '../types';
import { SECTION_ORDER, SECTION_TITLES } from '../lessons';
import type { LessonSection, LiveNumber, StepGauge } from '../lessons';
import type { LessonView, OutlineItem } from '../useLesson';
import { prefersReducedMotion } from '../useSimulation';
import { Button } from './Controls';
import { IconCheck, IconChevron, IconPlay, IconTarget } from './Icons';

const PHONE = '(max-width: 760px)';

export function LessonPanel({
  view,
  status,
  tone = 'neutral',
  exitLabel = 'Free play',
}: {
  view: LessonView;
  /** The page's event line, shown in free play only. */
  status: string;
  tone?: GuideTone;
  /** The button that leaves the lesson. */
  exitLabel?: string;
}) {
  const flagged = !view.active && tone !== 'neutral' && status !== '';

  // Screen readers hear each lesson once, on entry.
  const announcement = useMemo(
    () => (view.active ? view.title + '. ' + view.text : ''),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [view.active, view.lessonIndex, view.stepIndex],
  );

  return (
    <div className="mlx-lesson" data-free={view.active ? undefined : ''}>
      {view.active ? <LessonHead view={view} exitLabel={exitLabel} /> : <FreePlayHead view={view} status={status} tone={tone} flagged={flagged} />}
      {view.active ? <LessonStepCard view={view} /> : null}
      <Outline view={view} />
      <Announcer text={announcement} />
    </div>
  );
}

/* ---------------- heads ---------------- */

function LessonHead({ view, exitLabel }: { view: LessonView; exitLabel: string }) {
  return (
    <header className="mlx-lesson__head">
      <div className="mlx-lesson__where">
        <button
          type="button"
          className="mlx-lesson__arrow"
          onClick={view.prevLesson}
          aria-disabled={view.lessonIndex <= 0 || undefined}
          aria-label="Previous lesson"
          title="Previous lesson"
        >
          <IconChevron size={12} />
        </button>
        <span className="mlx-lesson__count">
          {'Lesson ' + (view.lessonIndex + 1) + ' / ' + view.lessonCount}
        </span>
        <button
          type="button"
          className="mlx-lesson__arrow mlx-lesson__arrow--next"
          onClick={view.nextLesson}
          aria-disabled={view.lessonIndex >= view.lessonCount - 1 || undefined}
          aria-label="Next lesson"
          title="Next lesson"
        >
          <IconChevron size={12} />
        </button>
        <button type="button" className="mlx-lesson__exit" onClick={view.exit} title="Leave the lesson; everything stays as it is">
          {exitLabel}
        </button>
      </div>
      <h3 className="mlx-lesson__title" title={view.title}>
        {view.title}
      </h3>
      <p className="mlx-lesson__hook mlx-lesson__hook--line" title={view.hook}>
        {view.hook}
      </p>
    </header>
  );
}

function FreePlayHead({
  view,
  status,
  tone,
  flagged,
}: {
  view: LessonView;
  status: string;
  tone: GuideTone;
  flagged: boolean;
}) {
  const doneCount = view.outline.filter((l) => l.done).length;
  const resumeIndex = view.outline.findIndex((l) => l.id === view.resumeId);
  const resume = resumeIndex >= 0 ? view.outline[resumeIndex] : null;
  const label =
    doneCount === 0 ? 'Start lesson 1' : doneCount >= view.lessonCount ? 'Start again' : 'Continue lesson ' + (resumeIndex + 1);
  return (
    <header className="mlx-lesson__head">
      <h3 className="mlx-lesson__title">Free play</h3>
      {flagged ? (
        <p className="mlx-lesson__status" data-tone={tone}>
          {status}
        </p>
      ) : (
        <p className="mlx-lesson__hook">Every control is live. Start a lesson for a guided walk through this chapter.</p>
      )}
      {resume ? (
        <div className="mlx-lesson__start">
          <Button size="sm" variant="primary" onClick={() => view.start(resume.id)} title={resume.title}>
            <IconPlay size={10} />
            {label}
          </Button>
        </div>
      ) : null}
    </header>
  );
}

/* ---------------- the step card ---------------- */

/** One lesson: the text, the live chips, the runs, the key idea. */
function LessonStepCard({ view }: { view: LessonView }) {
  // A new lesson reads from its top; the keyed body fades in.
  const stepKey = view.lessonIndex + ':' + view.stepIndex;
  const card = useRef<HTMLElement>(null);
  useEffect(() => {
    if (card.current) card.current.scrollTop = 0;
  }, [stepKey]);
  // The card keeps the height of the course's tallest lesson, measured off the unseen gauge.
  const gauge = useRef<HTMLDivElement>(null);
  const [bodyHeight, setBodyHeight] = useState(0);
  useLayoutEffect(() => {
    const el = gauge.current;
    if (!el) return;
    const measure = () => {
      let max = 0;
      for (const child of Array.from(el.children)) max = Math.max(max, (child as HTMLElement).offsetHeight);
      // A little slack: a live outcome can round a pixel taller than the one the gauge measured.
      setBodyHeight(max > 0 ? max + 2 : 0);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [view.course]);
  const style = bodyHeight > 0 ? ({ '--mlx-step-body': bodyHeight + 'px' } as CSSProperties) : undefined;

  return (
    <section ref={card} className="mlx-step" style={style} aria-label={'Lesson ' + (view.lessonIndex + 1)}>
      <div className="mlx-step__body" key={stepKey}>
        {view.text ? <p className="mlx-step__text">{view.text}</p> : null}

        {view.needsRun ? (
          <div className="mlx-step__run">
            <Button size="sm" variant="secondary" onClick={view.run} title="Play the run this lesson opens with">
              <IconPlay size={10} />
              Run
            </Button>
          </div>
        ) : null}

        {view.numbers.length > 0 ? <LiveNumbers numbers={view.numbers} /> : null}

        {view.focusLabel ? (
          <p className="mlx-step__focus" title="Highlighted on the page">
            <IconTarget size={12} />
            <span>{view.focusLabel}</span>
          </p>
        ) : null}

        {view.experiments.length > 0 ? <Experiments view={view} /> : null}

        {view.takeaway ? (
          <div className="mlx-step__takeaway">
            <span className="mlx-step__takeaway-label">Key idea</span>
            <p>{view.takeaway}</p>
          </div>
        ) : null}
      </div>

      <div ref={gauge} className="mlx-step__gauge" aria-hidden="true">
        <Gauge course={view.course} />
      </div>
    </section>
  );
}

/** The course's lessons, laid out unseen; the list only re-renders with the course. */
const Gauge = memo(function Gauge({ course }: { course: StepGauge[] }) {
  return (
    <>
      {course.map((step) => (
        <GaugeStep key={step.key} step={step} />
      ))}
    </>
  );
});

/** One lesson's static parts, laid out like the body, so the card can measure the tallest. */
function GaugeStep({ step }: { step: StepGauge }) {
  return (
    <div className="mlx-step__body mlx-step__body--gauge">
      {step.text ? <p className="mlx-step__text">{step.text}</p> : null}
      {step.numbers > 0 ? (
        <LiveNumbers numbers={Array.from({ length: step.numbers }, (_, i) => ({ label: 'n' + i, value: '0' }))} />
      ) : null}
      {step.focusLabel ? (
        <p className="mlx-step__focus">
          <IconTarget size={12} />
          <span>{step.focusLabel}</span>
        </p>
      ) : null}
      {step.experiments.length > 0 ? (
        <div className="mlx-step__experiments">
          {step.experiments.map((label, i) => (
            <div key={label} className="mlx-step__experiment">
              <button type="button" className="mlx-step__experiment-head" tabIndex={-1}>
                <span className="mlx-step__experiment-mark" />
                <span className="mlx-step__experiment-label">{label}</span>
              </button>
              {i === 0 && step.experimentOpen ? <p className="mlx-step__experiment-say">{step.experimentOpen}</p> : null}
            </div>
          ))}
        </div>
      ) : null}
      {step.takeaway ? (
        <div className="mlx-step__takeaway">
          <span className="mlx-step__takeaway-label">Key idea</span>
          <p>{step.takeaway}</p>
        </div>
      ) : null}
    </div>
  );
}

/** The runs: one row each; every one tried stays open to its outcome, so rows never jump. */
function Experiments({ view }: { view: LessonView }) {
  return (
    <div className="mlx-step__experiments" role="group" aria-label="Experiments">
      {view.experiments.map((experiment, i) => (
        <div
          key={experiment.label}
          className="mlx-step__experiment"
          data-done={experiment.done || undefined}
          data-active={experiment.active || undefined}
        >
          <button
            type="button"
            className="mlx-step__experiment-head"
            aria-expanded={experiment.done}
            onClick={() => view.tryExperiment(i)}
          >
            <span className="mlx-step__experiment-mark" aria-hidden="true">
              {experiment.done ? <IconCheck size={11} /> : <IconPlay size={10} />}
            </span>
            <span className="mlx-step__experiment-label">{experiment.label}</span>
          </button>
          {experiment.running ? (
            <p className="mlx-step__experiment-say mlx-step__experiment-say--running">
              <span className="mlx-step__spinner" aria-hidden="true" />
              Running
            </p>
          ) : experiment.done ? (
            <p className="mlx-step__experiment-say">{experiment.say}</p>
          ) : null}
        </div>
      ))}
    </div>
  );
}

/** The numbers the narration quotes, read live, so the words and the page agree. */
function LiveNumbers({ numbers }: { numbers: LiveNumber[] }) {
  return (
    <dl className="mlx-step__numbers">
      {numbers.map((n) => (
        <div key={n.label} className="mlx-step__number">
          <dt>{n.label}</dt>
          <dd>{n.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ---------------- the outline ---------------- */

/** The course by section; each section folds. */
function Outline({ view }: { view: LessonView }) {
  const sections = useMemo(
    () =>
      SECTION_ORDER.map((section) => ({
        section,
        items: view.outline.filter((l) => l.section === section),
      })).filter((group) => group.items.length > 0),
    [view.outline],
  );
  const doneCount = view.outline.filter((l) => l.done).length;
  // Phones stack the studio, so every section starts folded there.
  const [closed, setClosed] = useState<LessonSection[]>(() =>
    typeof window !== 'undefined' && window.matchMedia(PHONE).matches ? [...SECTION_ORDER] : [],
  );
  // The section holding the open lesson unfolds itself.
  const current = view.section;
  useEffect(() => {
    if (current) setClosed((prev) => (prev.includes(current) ? prev.filter((s) => s !== current) : prev));
  }, [current]);
  const toggle = (section: LessonSection) =>
    setClosed((prev) => (prev.includes(section) ? prev.filter((s) => s !== section) : [...prev, section]));
  // When the outline scrolls in its own band, the open lesson's row sits under the head.
  const nav = useRef<HTMLElement>(null);
  const lessonIndex = view.active ? view.lessonIndex : -1;
  useEffect(() => {
    const el = nav.current;
    if (!el || lessonIndex < 0 || el.scrollHeight <= el.clientHeight) return;
    const row = el.querySelector<HTMLElement>('.mlx-outline__entry[data-current]');
    const head = el.querySelector<HTMLElement>('.mlx-outline__head');
    if (!row) return;
    el.scrollTo({ top: row.offsetTop - (head?.offsetHeight ?? 0), behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }, [lessonIndex]);

  return (
    <nav ref={nav} className="mlx-outline" aria-label="All lessons">
      <div className="mlx-outline__head">
        <span className="mlx-outline__title">Lessons</span>
        <span className="mlx-outline__progress">{doneCount + ' / ' + view.lessonCount}</span>
      </div>
      {sections.map(({ section, items }) => {
        const open = !closed.includes(section);
        const sectionDone = items.filter((l) => l.done).length;
        return (
          <section key={section} className="mlx-outline__section" data-open={open || undefined}>
            <button type="button" className="mlx-outline__label" aria-expanded={open} onClick={() => toggle(section)}>
              <span className="mlx-outline__chevron">
                <IconChevron size={12} />
              </span>
              <span className="mlx-outline__label-text">{SECTION_TITLES[section]}</span>
              <span className="mlx-outline__label-count">{sectionDone + ' / ' + items.length}</span>
            </button>
            {open ? (
              <ol className="mlx-outline__items">
                {items.map((item) => (
                  <OutlineRow key={item.id} item={item} index={view.outline.indexOf(item)} onStart={view.start} />
                ))}
              </ol>
            ) : null}
          </section>
        );
      })}
    </nav>
  );
}

function OutlineRow({ item, index, onStart }: { item: OutlineItem; index: number; onStart: (id: string) => void }) {
  return (
    <li className="mlx-outline__entry" data-current={item.current || undefined}>
      <button
        type="button"
        className="mlx-outline__item"
        data-current={item.current || undefined}
        data-done={item.done || undefined}
        aria-current={item.current ? 'true' : undefined}
        title={item.hook}
        onClick={() => onStart(item.id)}
      >
        <span className="mlx-outline__num">{index + 1}</span>
        <span className="mlx-outline__name">{item.title}</span>
        {item.done ? (
          <span className="mlx-outline__tick">
            <IconCheck size={12} />
            <span className="mlx-visually-hidden">, done</span>
          </span>
        ) : null}
      </button>
    </li>
  );
}

/** A polite live region that only changes on entry. */
function Announcer({ text }: { text: string }) {
  const [spoken, setSpoken] = useState('');
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setSpoken(text), 150);
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, [text]);
  return (
    <span className="mlx-visually-hidden" role="status">
      {spoken}
    </span>
  );
}
