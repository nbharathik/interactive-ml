/** The studio shell: title strip, control row, then lesson/setup | architecture | output. */

import { createContext, useCallback, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { preloadOnIdle, warmOn } from '../../explainers/pages';
import { blogUrl, getNeighbours } from '../../explainers/registry';
import { useBandNav } from '../../lib/useBandNav';
import { useDocumentTitle } from '../../lib/useDocumentTitle';
import { LessonFocusContext, LessonOpenContext, SpotTag } from '../lessonFocus';
import { readFlag, writeFlag } from '../storage';
import { COMPACT_QUERY, PHONE_QUERY, useMediaQuery } from '../useMediaQuery';
import { useModalLayer } from '../useModalLayer';
import {
  IconArrow,
  IconArticle,
  IconCheck,
  IconCollapse,
  IconExitFullscreen,
  IconExpand,
  IconFullscreen,
  IconInfo,
  IconLink,
  IconSliders,
} from './Icons';
import { Popover } from './Popover';
import type { ExplainerMeta } from '../types';

const DIFFICULTY_LABELS: Record<ExplainerMeta['difficulty'], string> = {
  gentle: 'Gentle',
  core: 'Core',
  deep: 'Deep dive',
};

/** Whether the studio is locked to the viewport (desktop) or flows with the page. */
export const StudioLayoutContext = createContext<{ locked: boolean }>({ locked: false });

/** Remembers the full-page view across pages. */
const FULL_PAGE_KEY = 'mlx-full-page';

type RailTab = 'lesson' | 'setup';

export interface StudioProps {
  meta: ExplainerMeta;
  /** Play, step, reset and the counter, on the left of the top row. */
  transport: ReactNode;
  /** The model's main knobs, filling the rest of the top row. */
  topControls?: ReactNode;
  /** Small tools for the title strip, such as the reset button. */
  tools?: ReactNode;
  /** A link that reproduces the current configuration, behind the copy icon. */
  shareUrl?: string;
  /** The Setup tab: the data and everything else that sets the run up. */
  controls: ReactNode;
  /** Centre column: the architecture. */
  architecture: ReactNode;
  /** Right column: the output charts and numbers. */
  output: ReactNode;
  /** The Lesson tab, beside the setup in the left rail. */
  lesson?: ReactNode;
  /** The wide desk: the lesson column and the output column both get more room, the words at reading size. `'rail'` widens the lesson column only, for a stage that holds its own inspector. */
  wide?: boolean | 'rail';
}

export function Studio({
  meta,
  transport,
  topControls,
  tools,
  shareUrl,
  controls,
  architecture,
  output,
  lesson,
  wide,
}: StudioProps) {
  const { prev, next } = getNeighbours(meta.slug);
  const postUrl = meta.blogPath && meta.blogStatus === 'live' ? blogUrl(meta.blogPath) : null;
  // The neighbours in the reading order load while this page idles.
  useEffect(
    () => preloadOnIdle([next?.slug, prev?.slug].filter((slug): slug is string => Boolean(slug)), 8000),
    [next, prev],
  );

  // Below the tablet breakpoint the studio flows with the page.
  const compact = useMediaQuery(COMPACT_QUERY);
  const phone = useMediaQuery(PHONE_QUERY);
  const layout = useMemo(() => ({ locked: !compact }), [compact]);
  const [knobsOpen, setKnobsOpen] = useState(false);
  // A lesson step pointing at a knob behind Settings opens the sheet first.
  const focus = useContext(LessonFocusContext);
  useEffect(() => {
    if (phone && focus?.kind === 'control') setKnobsOpen(true);
  }, [phone, focus]);
  const knobsId = useId();
  const [tab, setTab] = useState<RailTab>('lesson');
  const tabsId = useId();
  // A step pointing at a knob in the Setup tab lights the tab while the lesson is showing.
  const [setupSpot, setSetupSpot] = useState(false);
  const hasLesson = Boolean(lesson);
  useEffect(() => {
    const next =
      hasLesson && tab === 'lesson' && focus?.kind === 'control'
        ? document.querySelector('.mlx-studio__setup [data-lesson-target="control:' + focus.key + '"]') !== null
        : false;
    setSetupSpot((prev) => (prev === next ? prev : next));
  }, [focus, tab, hasLesson]);

  // Expanded, the architecture takes the whole studio.
  const [expanded, setExpanded] = useState(false);
  const toggleExpanded = useCallback(() => setExpanded((v) => !v), []);
  const studioRef = useRef<HTMLElement>(null);
  useModalLayer(expanded, studioRef);

  // Full page: no site header or title strip; the title and links join the control row.
  const [fullPage, setFullPage] = useState(() => readFlag(FULL_PAGE_KEY));
  const toggleFullPage = useCallback(() => {
    setFullPage((v) => {
      writeFlag(FULL_PAGE_KEY, !v);
      return !v;
    });
  }, []);
  useEffect(() => {
    const root = document.documentElement;
    if (fullPage) root.setAttribute('data-mlx-full', '');
    else root.removeAttribute('data-mlx-full');
    return () => root.removeAttribute('data-mlx-full');
  }, [fullPage]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'f' && event.key !== 'F') return;
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) return;
      toggleFullPage();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggleFullPage]);

  useDocumentTitle(meta.title);
  // The header and the title strip read as one band.
  useBandNav();

  // The wide desk marks an open lesson so the rail can give the card the room.
  const lessonOpen = useContext(LessonOpenContext);

  useEffect(() => {
    if (!expanded) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setExpanded(false);
    };
    window.addEventListener('keydown', onKey);
    // The page behind a full-page view must not scroll.
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = previous;
    };
  }, [expanded]);

  const ident = (
    <div className="mlx-studio__ident">
      <div className="mlx-studio__titlerow">
        <h1 className="mlx-studio__title">{meta.title}</h1>
        <Popover
          button={<IconInfo size={14} />}
          label={'About ' + meta.title.toLowerCase()}
          className="mlx-studio__about"
          align={fullPage ? 'right' : 'left'}
        >
          <p className="mlx-studio__about-summary">{meta.summary}</p>
          <ul className="mlx-studio__about-concepts">
            {meta.concepts.map((concept) => (
              <li key={concept}>{concept}</li>
            ))}
          </ul>
          <p className="mlx-studio__about-meta">
            {DIFFICULTY_LABELS[meta.difficulty]}
          </p>
          {phone ? (
            <div className="mlx-studio__about-links">
              {postUrl ? (
                <a href={postUrl} target="_blank" rel="noreferrer">
                  Read the post
                </a>
              ) : null}
              {prev ? (
                <Link to={'/explainer/' + prev.slug} {...warmOn(prev.slug)}>
                  Previous: {prev.title}
                </Link>
              ) : null}
              {next ? (
                <Link to={'/explainer/' + next.slug} {...warmOn(next.slug)}>
                  Next: {next.title}
                </Link>
              ) : null}
            </div>
          ) : null}
        </Popover>
      </div>
    </div>
  );

  const toolbar = (
    <div className="mlx-studio__tools">
      {tools}
      <div className="mlx-studio__links">
        {shareUrl ? <ShareButton url={shareUrl} /> : null}
        {!phone && postUrl ? (
          <a
            className="mlx-studio__link"
            href={postUrl}
            target="_blank"
            rel="noreferrer"
            title="Read the post: the derivation and the code"
            aria-label="Read the post"
          >
            <IconArticle size={14} />
          </a>
        ) : null}
        {!phone && next ? (
          <Link
            className="mlx-studio__link"
            to={'/explainer/' + next.slug}
            title={'Next: ' + next.title}
            aria-label={'Next: ' + next.title}
            {...warmOn(next.slug)}
          >
            <IconArrow size={14} />
          </Link>
        ) : null}
        <button
          type="button"
          className="mlx-studio__link"
          onClick={toggleFullPage}
          aria-pressed={fullPage}
          title={fullPage ? 'Show the site header and title (F)' : 'Full page: hide the site header and title (F)'}
          aria-label={fullPage ? 'Leave full page' : 'Full page'}
        >
          {fullPage ? <IconExitFullscreen size={14} /> : <IconFullscreen size={14} />}
        </button>
      </div>
    </div>
  );

  return (
    <StudioLayoutContext.Provider value={layout}>
    <div className="mlx-studio-page">
      <section
        ref={studioRef}
        tabIndex={expanded ? -1 : undefined}
        className="mlx-studio"
        aria-label={meta.title + ' studio'}
        data-expanded={expanded || undefined}
        data-full={fullPage || undefined}
        data-wide={wide === 'rail' ? 'rail' : wide ? '' : undefined}
        data-lesson={wide && lessonOpen ? '' : undefined}
      >
        {fullPage ? null : (
          <header className="mlx-band mlx-studio__head">
            {ident}
            {toolbar}
          </header>
        )}

        <div className="mlx-studio__bar">
          {transport}
          {topControls && phone ? (
            <button
              type="button"
              className="mlx-studio__settings"
              aria-expanded={knobsOpen}
              aria-controls={knobsId}
              aria-label="Settings"
              onClick={() => setKnobsOpen((v) => !v)}
            >
              <IconSliders size={14} />
              <span>Settings</span>
            </button>
          ) : null}
          {topControls ? (
            <div id={knobsId} className="mlx-studio__knobs" hidden={phone && !knobsOpen}>
              {topControls}
            </div>
          ) : null}
          {fullPage ? (
            <div className="mlx-studio__barend">
              {ident}
              {toolbar}
            </div>
          ) : null}
        </div>

        <div className="mlx-studio__body">
          <aside className="mlx-studio__rail" aria-label={lesson ? 'Lesson and setup' : 'Setup'}>
            {lesson ? (
              <div className="mlx-tabs" role="tablist" aria-label="Left column">
                {(['lesson', 'setup'] as const).map((name) => (
                  <button
                    key={name}
                    type="button"
                    role="tab"
                    id={tabsId + '-' + name}
                    className="mlx-tabs__tab"
                    aria-selected={tab === name}
                    aria-controls={tabsId + '-' + name + '-panel'}
                    tabIndex={tab === name ? 0 : -1}
                    data-mlx-spot={setupSpot && name === 'setup' ? '' : undefined}
                    onClick={() => setTab(name)}
                    onKeyDown={(event) => {
                      const next =
                        event.key === 'ArrowLeft' || event.key === 'ArrowRight'
                          ? name === 'lesson' ? 'setup' : 'lesson'
                          : event.key === 'Home' ? 'lesson' : event.key === 'End' ? 'setup' : null;
                      if (!next) return;
                      event.preventDefault();
                      setTab(next);
                      document.getElementById(tabsId + '-' + next)?.focus();
                    }}
                  >
                    {name === 'lesson' ? 'Lesson' : 'Setup'}
                  </button>
                ))}
              </div>
            ) : null}
            {lesson ? (
              <div
                role="tabpanel"
                id={tabsId + '-lesson-panel'}
                aria-labelledby={tabsId + '-lesson'}
                className="mlx-studio__lesson"
                hidden={tab !== 'lesson'}
              >
                {lesson}
              </div>
            ) : null}
            <div
              role={lesson ? 'tabpanel' : undefined}
              id={tabsId + '-setup-panel'}
              aria-labelledby={lesson ? tabsId + '-setup' : undefined}
              className="mlx-studio__setup"
              hidden={lesson ? tab !== 'setup' : undefined}
            >
              {controls}
            </div>
          </aside>

          <div className="mlx-studio__stage">
            <button
              type="button"
              className="mlx-studio__expand"
              onClick={toggleExpanded}
              title={expanded ? 'Back to the three columns (Esc)' : 'Give the architecture the whole screen'}
            >
              {expanded ? (
                <IconCollapse size={14} />
              ) : (
                <IconExpand size={14} />
              )}
            </button>
            <div className="mlx-studio__arch">{architecture}</div>
          </div>

          <aside className="mlx-studio__out" aria-label="Output">
            {output}
          </aside>
        </div>
      </section>
      <SpotTag within={wide && tab === 'setup' ? '.mlx-studio__setup' : undefined} />
    </div>
    </StudioLayoutContext.Provider>
  );
}

/** Copies the share link; the icon turns into a tick for a moment. */
function ShareButton({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return undefined;
    const id = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(id);
  }, [copied]);

  return (
    <>
      <button
        type="button"
        className="mlx-studio__link"
        data-done={copied || undefined}
        title={copied ? 'Link copied' : 'Copy a link to this exact configuration'}
        aria-label="Copy link"
        onClick={() => {
          navigator.clipboard?.writeText(url).then(
            () => setCopied(true),
            () => setCopied(false),
          );
        }}
      >
        {copied ? <IconCheck size={14} /> : <IconLink size={14} />}
      </button>
      <span className="mlx-visually-hidden" role="status">
        {copied ? 'Link copied' : ''}
      </span>
    </>
  );
}

/* ---------------- column furniture ---------------- */

/** The small uppercase heading that names a studio column. */
export function ColumnHead({ title, blurb }: { title: string; blurb?: ReactNode }) {
  return (
    <div className="mlx-colhead">
      <h2 className="mlx-colhead__title">{title}</h2>
      {blurb ? <p className="mlx-colhead__blurb">{blurb}</p> : null}
    </div>
  );
}
