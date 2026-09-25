/**
 * Browser smoke tests: builds the app, serves it and drives headless Chrome.
 *
 * Usage:  node tests/browser.mjs [--headful] [--only=slug] [--preview]
 *
 * --preview builds with VITE_PREVIEW=1 and also tests the parked pages.
 */

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const shotDir = path.join(here, '.shots');

const args = process.argv.slice(2);
const headful = args.includes('--headful');
const only = args.find((a) => a.startsWith('--only='))?.slice(7);
const preview = args.includes('--preview');

const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];

const EXPLAINERS = [
  'linear-regression',
  'polynomial-regression',
  'logistic-regression',
  'k-means',
  'decision-tree',
  'neural-network',
  'knn',
  'regularization',
  'activation-functions',
  'cnn',
];

/** Parked pages: routed only in a VITE_PREVIEW=1 build, so tested only with --preview. */
const PARKED = ['transformers'];

/** Pages without authored lessons: the rail opens on Setup with a preset picker. */
const FREE_PLAY = new Set([]);

const VIEWPORTS = [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'tablet', width: 834, height: 1112 },
  { name: 'desktop', width: 1440, height: 900 },
];

/* ------------------------------------------------------------------ */

const results = [];
let failures = 0;

function check(name, ok, detail) {
  results.push({ name, ok, detail });
  if (!ok) failures += 1;
  const mark = ok ? 'PASS' : 'FAIL';
  console.log('  ' + mark + '  ' + name + (detail ? '  · ' + detail : ''));
}

function findChrome() {
  const fromEnv = process.env.CHROME_PATH;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  for (const candidate of CHROME_CANDIDATES) {
    if (existsSync(candidate)) return candidate;
  }
  throw new Error('No Chrome or Edge found. Set CHROME_PATH to a browser executable.');
}

async function waitForServer(url, timeoutMs = 90_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return false;
}

/** A canvas that painted has non-uniform pixels; a blank one does not. */
const CANVAS_PROBE = `(() => {
  const out = [];
  for (const canvas of document.querySelectorAll('canvas')) {
    const rect = canvas.getBoundingClientRect();
    let painted = false;
    let distinct = 0;
    try {
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      if (ctx && w > 0 && h > 0) {
        const data = ctx.getImageData(0, 0, w, h).data;
        const seen = new Set();
        const stride = Math.max(4, Math.floor(data.length / 4 / 4000) * 4);
        for (let i = 0; i < data.length; i += stride) {
          if (data[i + 3] !== 0) painted = true;
          seen.add((data[i] << 16) | (data[i+1] << 8) | data[i+2]);
          if (seen.size > 60) break;
        }
        distinct = seen.size;
      }
    } catch (e) { /* tainted or unavailable */ }
    out.push({
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      label: canvas.getAttribute('aria-label') || '',
      painted,
      distinct,
    });
  }
  return out;
})()`;

async function collectCanvases(page) {
  return page.evaluate(CANVAS_PROBE);
}

/** Start the nth lesson from the outline in the Lesson tab. */
async function startLesson(page, index) {
  return page.evaluate((n) => {
    const rows = document.querySelectorAll('.mlx-outline__item');
    const target = rows[Math.min(n, rows.length - 1)];
    if (!target) return false;
    target.click();
    return true;
  }, index);
}

async function run() {
  mkdirSync(shotDir, { recursive: true });

  console.log('Building for preview…');
  const build = spawn('npm', ['run', 'build'], { cwd: root, shell: true, stdio: 'pipe', env: preview ? { ...process.env, VITE_PREVIEW: '1' } : process.env });
  let buildOutput = '';
  build.stdout.on('data', (d) => (buildOutput += d));
  build.stderr.on('data', (d) => (buildOutput += d));
  const buildCode = await new Promise((resolve) => build.on('exit', resolve));
  if (buildCode !== 0) {
    console.error(buildOutput.slice(-4000));
    throw new Error('Build failed, cannot run browser tests.');
  }
  console.log('Build OK.\n');

  const port = 4180;
  const server = spawn(
    'npx',
    ['vite', 'preview', '--port', String(port), '--strictPort'],
    { cwd: root, shell: true, stdio: 'pipe' },
  );
  server.stdout.on('data', () => {});
  server.stderr.on('data', () => {});

  const base = 'http://localhost:' + port + '/interactive-ml/';
  const up = await waitForServer(base);
  if (!up) {
    server.kill();
    throw new Error('Preview server did not start.');
  }

  const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: !headful,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--force-device-scale-factor=1'],
    defaultViewport: { width: 1440, height: 900 },
  });

  try {
    await testHome(browser, base);
    for (const slug of preview ? EXPLAINERS.concat(PARKED) : EXPLAINERS) {
      if (only && slug !== only) continue;
      await testExplainer(browser, base, slug);
    }
    await testResponsive(browser, base);
    await testTheme(browser, base);
    await testDeepLink(browser, base);
    await testLessons(browser, base);
  } finally {
    await browser.close();
    server.kill();
  }

  console.log('\n' + '='.repeat(64));
  console.log(
    results.length + ' checks · ' + (results.length - failures) + ' passed · ' + failures + ' failed',
  );
  if (failures > 0) {
    console.log('\nFailures:');
    for (const r of results.filter((x) => !x.ok)) {
      console.log('  - ' + r.name + (r.detail ? ': ' + r.detail : ''));
    }
  }
  console.log('Screenshots: ' + shotDir);
  process.exit(failures > 0 ? 1 : 0);
}

/** Open a page, capturing console errors and uncaught exceptions. */
async function openPage(browser, url, viewport) {
  const page = await browser.newPage();
  if (viewport) await page.setViewport(viewport);
  const problems = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      // Ignore favicon and font noise that says nothing about our code.
      if (/favicon|net::ERR_|Failed to load resource/i.test(text)) return;
      problems.push('console: ' + text);
    }
  });
  page.on('pageerror', (err) => problems.push('uncaught: ' + err.message));
  await page.goto(url, { waitUntil: 'networkidle2', timeout: 60_000 });
  return { page, problems };
}

async function testHome(browser, base) {
  console.log('Home page');
  const { page, problems } = await openPage(browser, base);

  const title = await page.$eval('.mlx-home__title', (el) => el.textContent?.trim() ?? '');
  check('home renders its title', title.length > 10, title.slice(0, 40));

  const cardCount = await page.$$eval('a.mlx-card', (els) => els.length);
  check('catalog lists every finished explainer', cardCount === EXPLAINERS.length, cardCount + ' cards');

  const hrefs = await page.$$eval('a.mlx-card', (els) => els.map((e) => e.getAttribute('href')));
  const allLinked = hrefs.every((h) => h && h.includes('/explainer/'));
  check('every card links to an explainer', allLinked, hrefs.length + ' hrefs');

  const roadmap = await page.$$eval('.mlx-roadmap li', (els) => els.length);
  check('roadmap is visible', roadmap >= 5, roadmap + ' planned');

  // Parked pages open by their address only: the roadmap may name them, but nothing links them.
  const leaked = await page.evaluate((slugs) => slugs.filter((slug) => document.querySelector('a[href*="/explainer/' + slug + '"]') !== null), PARKED);
  check('parked pages are not linked from the home page', leaked.length === 0, leaked.join(', '));

  const taglines = await page.$$eval('.mlx-card__tagline', (els) =>
    els.map((e) => e.textContent?.trim()).filter(Boolean).length,
  );
  check('every card carries a tagline', taglines === cardCount, taglines + ' labelled');

  // A card actually navigates, and the explainer arrives in one frame: no spinner, charts and maths already drawn.
  await page.evaluate(() => {
    const seen = { spinner: false, shell: null, charts: null, math: null };
    window.__arrival = seen;
    const tick = () => {
      const now = performance.now();
      if (document.querySelector('.mlx-loading, .mlx-skeleton')) seen.spinner = true;
      if (seen.shell === null && document.querySelector('.mlx-studio-page:not(.mlx-skeleton) .mlx-studio')) seen.shell = now;
      if (seen.shell !== null && seen.charts === null) {
        const canvases = [...document.querySelectorAll('.mlx-studio canvas')];
        if (canvases.length && canvases.every((c) => c.width !== 300 || c.height !== 150)) seen.charts = now;
      }
      if (seen.shell !== null && seen.math === null && document.querySelector('.mlx-studio .katex')) seen.math = now;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    document.querySelector('a.mlx-card').click();
  });
  await page.waitForSelector('.mlx-studio-page:not(.mlx-skeleton) canvas', { timeout: 20_000 }).catch(() => null);
  await new Promise((r) => setTimeout(r, 400));
  const landed = page.url().includes('/explainer/');
  check('a card opens its explainer', landed, page.url().split('/').slice(-1)[0]);
  const arrival = await page.evaluate(() => window.__arrival);
  check('the old page stays up until the explainer is ready', !arrival.spinner);
  const oneFrame = arrival.shell !== null && arrival.charts === arrival.shell && (arrival.math === null || arrival.math === arrival.shell);
  check(
    'the explainer arrives complete, charts and maths in its first frame',
    oneFrame,
    'shell ' + Math.round(arrival.shell ?? -1) + ' charts ' + Math.round(arrival.charts ?? -1) + ' math ' + Math.round(arrival.math ?? -1),
  );

  check('home has no console errors', problems.length === 0, problems.slice(0, 2).join(' | '));
  await page.goto(base, { waitUntil: 'networkidle2', timeout: 30_000 });
  await page.screenshot({ path: path.join(shotDir, 'home.png'), fullPage: false });
  await page.close();
}

async function testExplainer(browser, base, slug) {
  console.log('\nExplainer: ' + slug);
  const url = base + 'explainer/' + slug;
  const { page, problems } = await openPage(browser, url);

  // Mounted?
  const heroText = await page
    .$eval('.mlx-studio__title', (el) => el.textContent?.trim() ?? '')
    .catch(() => '');
  check(slug + ': page mounts', heroText.length > 3, heroText.slice(0, 40));
  if (heroText.length <= 3) {
    check(slug + ': (skipping rest: page did not mount)', false, problems.slice(0, 3).join(' | '));
    await page.close();
    return;
  }

  // The studio layout: control row, then rail, architecture, output.
  const studioParts = await page.evaluate(() => ({
    studio: !!document.querySelector('.mlx-studio'),
    bar: !!document.querySelector('.mlx-studio__bar .mlx-transport'),
    knobs: document.querySelectorAll('.mlx-toprow__cell').length > 0,
    rail: !!document.querySelector('.mlx-studio__rail'),
    arch: !!document.querySelector('.mlx-studio__arch canvas'),
    out: !!document.querySelector('.mlx-studio__out'),
  }));
  check(
    slug + ': studio has a control row, rail, architecture and output column',
    Object.values(studioParts).every(Boolean),
    JSON.stringify(studioParts),
  );
  const g0 = await readGuide(page);

  // The studio is exactly one screen and the architecture gets a real share of it.
  const fit = await page.evaluate(() => {
    const studio = document.querySelector('.mlx-studio')?.getBoundingClientRect();
    // Every canvas in the stage counts: a page may stack two.
    const boxes = [...document.querySelectorAll('.mlx-studio__arch canvas')].map((c) => c.getBoundingClientRect());
    const canvasHeight = boxes.length ? Math.round(Math.max(...boxes.map((b) => b.bottom)) - Math.min(...boxes.map((b) => b.top))) : 0;
    return {
      bottom: studio ? Math.round(studio.bottom) : 0,
      viewport: window.innerHeight,
      canvasHeight,
    };
  });
  check(
    slug + ': the studio fills exactly one screen',
    Math.abs(fit.bottom - fit.viewport) <= 2,
    fit.bottom + 'px bottom vs ' + fit.viewport + 'px viewport',
  );
  check(
    slug + ': the architecture keeps most of the height',
    fit.canvasHeight >= fit.viewport * 0.35,
    fit.canvasHeight + 'px canvas',
  );

  // The algorithm is the page: nothing sits under the studio.
  const below = await page.evaluate(() => {
    const studio = document.querySelector('.mlx-studio');
    return studio ? studio.parentElement.children.length : 0;
  });
  check(slug + ': nothing below the studio', below === 1, below + ' blocks in the page');

  // The architecture can take the whole studio and give it back.
  const studioState = () =>
    page.evaluate(() => ({
      expanded: document.querySelector('.mlx-studio')?.hasAttribute('data-expanded') ?? false,
      railHidden:
        getComputedStyle(document.querySelector('.mlx-studio__rail')).display === 'none',
    }));
  const hasExpand = (await page.$('.mlx-studio__expand')) !== null;
  let expandable = 'missing';
  if (hasExpand) {
    await page.click('.mlx-studio__expand');
    await new Promise((r) => setTimeout(r, 250));
    const open = await studioState();
    await page.keyboard.press('Escape');
    await new Promise((r) => setTimeout(r, 250));
    const closed = await studioState();
    expandable =
      open.expanded && open.railHidden && !closed.expanded && !closed.railHidden
        ? 'ok'
        : JSON.stringify({ open, closed });
  }
  check(slug + ': the architecture expands and collapses', expandable === 'ok', expandable);

  // Clicking the architecture must open something. Real mouse input: the canvas uses pointer capture.
  const archRect = await page.evaluate(() => {
    const canvas = document.querySelector('.mlx-studio__arch canvas');
    if (!canvas) return null;
    const r = canvas.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
  });
  let inspectOpened = false;
  if (archRect) {
    const readInspect = () =>
      page.evaluate(() => ({
        strip: document.querySelector('.mlx-studio__inspect')?.textContent ?? '',
        card: document.querySelector('.mlx-detail')?.textContent ?? '',
        // A diagram that shows its detail in a pane of its own labels that pane.
        pane: document.querySelector('[data-arch-detail] canvas')?.getAttribute('aria-label') ?? '',
      }));
    const before = await readInspect();
    // Sweep finely and stop at the first thing that responds.
    for (let fx = 0.12; fx <= 0.92 && !inspectOpened; fx += 0.05) {
      for (let fy = 0.06; fy <= 0.9 && !inspectOpened; fy += 0.06) {
        await page.mouse.click(archRect.x + archRect.w * fx, archRect.y + archRect.h * fy);
        await new Promise((r) => setTimeout(r, 35));
        const after = await readInspect();
        if (after.card.length > 40) inspectOpened = true;
        if (after.strip !== before.strip && after.strip.length > 60) inspectOpened = true;
        if (after.pane !== before.pane && after.pane.length > 20) inspectOpened = true;
      }
    }
    // Leave the canvas so a preview card does not linger over the next checks.
    await page.mouse.move(2, 2);
  }
  check(slug + ': clicking the architecture opens its detail', inspectOpened);
  check(slug + ': no inspector strip changes the stage height', (await page.$('.mlx-studio__inspect')) === null);

  // Both output charts sit inside the viewport: nothing to scroll for.
  const outFit = await page.evaluate(() => {
    const out = document.querySelector('.mlx-studio__out');
    if (!out) return null;
    const canvases = Array.from(out.querySelectorAll('canvas')).map((c) => c.getBoundingClientRect().bottom);
    return { overflow: out.scrollHeight - out.clientHeight, lowest: Math.max(0, ...canvases), viewport: window.innerHeight };
  });
  check(
    slug + ': the output column fits without scrolling',
    outFit !== null && outFit.overflow <= 1 && outFit.lowest <= outFit.viewport + 1,
    JSON.stringify(outFit),
  );

  // A top-row tooltip opens clear of the header and inside the viewport.
  let tipRect = null;
  if ((await page.$('.mlx-toprow .mlx-tip__button')) !== null) {
    await page.hover('.mlx-toprow .mlx-tip__button');
    await new Promise((r) => setTimeout(r, 200));
    tipRect = await page.evaluate(() => {
      const bubble = document.querySelector('.mlx-tip__bubble');
      const header = document.querySelector('.mlx-header');
      if (!bubble || !header) return null;
      const r = bubble.getBoundingClientRect();
      return { top: r.top, right: r.right, headerBottom: header.getBoundingClientRect().bottom, vw: window.innerWidth };
    });
    await page.mouse.move(2, 2);
  }
  check(
    slug + ': a top-row tooltip clears the header',
    tipRect !== null && tipRect.top >= tipRect.headerBottom && tipRect.right <= tipRect.vw,
    JSON.stringify(tipRect),
  );

  // Keyboard: the diagram takes focus, the arrows move a cursor, Enter opens a card.
  const keyboard = await page.evaluate(() => {
    const diagram = document.querySelector('.mlx-arch[tabindex], .mlx-tree[tabindex]');
    if (!diagram) return { supported: false };
    diagram.focus();
    return { supported: true, focused: document.activeElement === diagram };
  });
  if (keyboard.supported) {
    const paneLabel = () => page.evaluate(() => document.querySelector('[data-arch-detail] canvas')?.getAttribute('aria-label') ?? '');
    const paneBefore = await paneLabel();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Enter');
    await new Promise((r) => setTimeout(r, 300));
    const opened = (await page.$('.mlx-detail')) !== null || (paneBefore !== '' && (await paneLabel()) !== paneBefore);
    await page.keyboard.press('Escape');
    await new Promise((r) => setTimeout(r, 200));
    const closed = (await page.$('.mlx-detail')) === null;
    const counter = await page.$eval('.mlx-counter__value', (el) => el.textContent?.trim() ?? '');
    check(slug + ': the diagram is keyboard accessible', keyboard.focused && opened && closed, JSON.stringify({ opened, closed, counter }));
  }

  // Canvases painted.
  await new Promise((r) => setTimeout(r, 900));
  const canvases = await collectCanvases(page);
  check(slug + ': has canvases', canvases.length >= 2, canvases.length + ' canvases');
  const blank = canvases.filter((c) => c.width > 20 && c.height > 20 && !c.painted);
  check(
    slug + ': every canvas paints',
    blank.length === 0,
    blank.length ? blank.map((b) => b.label || '(unlabelled)').join(' | ') : '',
  );
  const flat = canvases.filter((c) => c.width > 40 && c.height > 40 && c.distinct < 3);
  check(
    slug + ': canvases render more than a flat fill',
    flat.length === 0,
    flat.length ? flat.map((b) => b.label || '(unlabelled)').join(' | ') : '',
  );

  // Every canvas has an accessible label.
  const unlabelled = canvases.filter((c) => !c.label);
  check(slug + ': every canvas is labelled', unlabelled.length === 0, unlabelled.length + ' unlabelled');

  // Transport: does stepping advance the counter?
  const hasTransport = (await page.$('.mlx-transport')) !== null;
  if (hasTransport) {
    const readCount = () =>
      page.$eval('.mlx-counter__value', (el) => el.textContent?.trim() ?? '');
    const before = await readCount();
    // The panel just before the step, since earlier clicks may have edited the data it describes.
    const gPre = await readGuide(page);
    // Click the "Step" button by its visible label.
    const clicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('.mlx-transport button'));
      const step = buttons.find((b) => (b.getAttribute('aria-label') || '').startsWith('Advance one'));
      if (step) {
        step.click();
        return true;
      }
      return false;
    });
    await new Promise((r) => setTimeout(r, 400));
    const after = await readCount();
    check(slug + ': step advances the simulation', clicked && before !== after, before + ' -> ' + after);

    // The page opens in free play with the course on show, and stays quiet while the model trains.
    const g1 = await readGuide(page);
    if (FREE_PLAY.has(slug)) {
      check(slug + ': the page opens in free play with a preset picker', g0.name === 'Free play' && g0.presetOptions >= 6, g0.name + ' ' + g0.presetOptions + ' presets');
      check(slug + ': the first preset is the default setting', g0.preset !== '' && g0.preset === g1.preset, g0.preset || '(custom)');
    } else {
      check(slug + ': the page opens in free play', g0.selected === -1 && g0.free && g0.name === 'Free play', g0.name + ' ' + g0.selected);
      check(slug + ': free play offers Start lesson 1', g0.start === 'Start lesson 1', g0.start);
      check(slug + ': the outline groups the lessons in sections', g0.sections >= 3 && g0.sections <= 4, g0.sections + ' sections');
    }
    check(slug + ': a step leaves the panel alone', g1.text === gPre.text && g1.status === '', g1.text === gPre.text ? g1.status.slice(0, 60) || '(quiet)' : gPre.text.slice(0, 40) + ' -> ' + g1.text.slice(0, 40));

    // Play then pause.
    await page.click('.mlx-tbtn--play');
    await new Promise((r) => setTimeout(r, 900));
    const running = await readCount();
    await page.click('.mlx-tbtn--play');
    check(slug + ': play advances further', running !== after, after + ' -> ' + running);

    // Reset returns to zero.
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('.mlx-transport button'));
      const reset = buttons.find((b) => b.getAttribute('aria-label') === 'Reset');
      reset?.click();
    });
    await new Promise((r) => setTimeout(r, 300));
    const afterReset = await readCount();
    // The counter is "0" when unbounded and "0 / 11" when the run has a known length.
    const resetCount = Number(afterReset.split('/')[0].replace(/[^0-9]/g, ''));
    check(slug + ': reset returns to zero', resetCount === 0, afterReset);

    // Speed control, with words rather than bare multipliers.
    const speeds = await page.$$eval('.mlx-speed option', (els) => els.map((o) => o.textContent ?? ''));
    check(slug + ': speed control present', speeds.length === 4, speeds.length + ' options');
    check(slug + ': speed options are readable', speeds.every((t) => /[a-z]/i.test(t)), speeds.join(', '));

    // The play nudge is gone once anything has run.
    check(slug + ': the play nudge stops after the first run', (await page.$('.mlx-tbtn--play[data-nudge]')) === null);

    // A finished run offers Replay instead of a dead button.
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('.mlx-transport button'));
      buttons.find((b) => (b.getAttribute('aria-label') || '').startsWith('Run '))?.click();
    });
    await new Promise((r) => setTimeout(r, 500));
    const finished = (await page.$('.mlx-transport__label[data-done]')) !== null;
    if (finished) {
      const replay = (await page.$('.mlx-tbtn--play[data-replay]:not([disabled])')) !== null;
      check(slug + ': a finished run offers Replay', replay);
    }
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('.mlx-transport button'));
      buttons.find((b) => b.getAttribute('aria-label') === 'Reset')?.click();
    });
    await new Promise((r) => setTimeout(r, 300));
  } else {
    check(slug + ': has a transport', false, 'no .mlx-transport found');
  }

  // Lessons: the outline lists them all, Start opens the first, and picking one changes the URL query.
  const lessonCount = FREE_PLAY.has(slug) ? 0 : await page.$$eval('.mlx-outline__item', (els) => els.length);
  if (FREE_PLAY.has(slug)) {
    // A preset picker instead: choosing one rewrites the URL and the picker follows.
    const urlBefore = page.url();
    const picked = await page.evaluate(() => {
      const select = document.querySelector('.mlx-presets select');
      if (!select) return '';
      const next = Array.from(select.options).find((o) => o.value && o.value !== select.value);
      if (!next) return '';
      const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
      setter?.call(select, next.value);
      select.dispatchEvent(new Event('change', { bubbles: true }));
      return next.value;
    });
    await new Promise((r) => setTimeout(r, 500));
    const gPick = await readGuide(page);
    check(slug + ': picking a preset rewrites the URL', picked !== '' && page.url() !== urlBefore && page.url().includes('?'), page.url().split('?')[1]?.slice(0, 60) ?? '(no query)');
    check(slug + ': the picker shows the preset on show', gPick.preset === picked, gPick.preset + ' vs ' + picked);
    check(slug + ': the preset explains what to notice', gPick.presetNote.length > 20, gPick.presetNote.slice(0, 60));
    await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll('.mlx-transport button'));
      buttons.find((b) => b.getAttribute('aria-label') === 'Reset')?.click();
    });
    await new Promise((r) => setTimeout(r, 300));
  } else {
    check(slug + ': ships lessons', lessonCount >= 4, lessonCount + ' lessons');
  }
  if (lessonCount > 1) {
    await page.click('.mlx-lesson__start button');
    await new Promise((r) => setTimeout(r, 500));
    const gStart = await readGuide(page);
    check(slug + ': Start opens lesson 1 in its card', gStart.selected === 0 && gStart.card, gStart.name);
    check(slug + ': the lesson narrates its picture', gStart.text.length > 0, gStart.text.slice(0, 60));
    check(slug + ': the lesson offers runs', gStart.runs >= 2, gStart.runs + ' runs');
    await page.click('.mlx-lesson__exit');
    await new Promise((r) => setTimeout(r, 400));

    const urlBefore = page.url();
    await startLesson(page, 1);
    await new Promise((r) => setTimeout(r, 500));
    const urlAfter = page.url();
    check(slug + ': a lesson changes the shareable URL', urlBefore !== urlAfter && urlAfter.includes('lesson='), urlAfter.split('?')[1]?.slice(0, 60) ?? '(no query)');

    const g2 = await readGuide(page);
    check(slug + ': picking a lesson opens it', g2.selected === 1 && g2.card, g2.name);
    await page.click('.mlx-lesson__arrow--next');
    await new Promise((r) => setTimeout(r, 500));
    const g3 = await readGuide(page);
    check(slug + ': the lesson arrows walk the lessons', g3.selected === 2, g3.name.slice(0, 50));

    // The next arrow advances, the URL follows, the previous arrow returns, Free play leaves the lesson but keeps the setup.
    await page.click('.mlx-lesson__arrow--next');
    await new Promise((r) => setTimeout(r, 400));
    const g5 = await readGuide(page);
    check(slug + ': the next arrow advances', g5.selected === 3 && g5.card, g5.name);
    check(slug + ': the URL carries the position', page.url().includes('lesson='), page.url().split('?')[1]?.slice(0, 60) ?? '');
    const midFit = await page.evaluate(() => {
      const studio = document.querySelector('.mlx-studio')?.getBoundingClientRect();
      const boxes = [...document.querySelectorAll('.mlx-studio__arch canvas')].map((c) => c.getBoundingClientRect());
      const canvasHeight = boxes.length ? Math.round(Math.max(...boxes.map((b) => b.bottom)) - Math.min(...boxes.map((b) => b.top))) : 0;
      return { bottom: studio ? Math.round(studio.bottom) : 0, viewport: window.innerHeight, canvasHeight };
    });
    check(slug + ': the studio still fills one screen mid-lesson', Math.abs(midFit.bottom - midFit.viewport) <= 2 && midFit.canvasHeight >= midFit.viewport * 0.35, JSON.stringify(midFit));
    await page.click('.mlx-lesson__arrow:not(.mlx-lesson__arrow--next)');
    await new Promise((r) => setTimeout(r, 400));
    const g6 = await readGuide(page);
    check(slug + ': the previous arrow returns', g6.selected === 2, 'lesson ' + (g6.selected + 1));
    // The setup a lesson leaves behind is the URL minus the lesson keys, empty when the lesson used the defaults.
    const kept = await page.evaluate(() => {
      const query = new URLSearchParams(location.search);
      query.delete('lesson');
      query.delete('step');
      const rest = query.toString();
      return rest ? '?' + rest : '';
    });
    await page.click('.mlx-lesson__exit');
    await new Promise((r) => setTimeout(r, 400));
    const free = await page.evaluate(() => ({
      free: document.querySelector('.mlx-lesson')?.hasAttribute('data-free') ?? false,
      url: location.search,
      current: document.querySelector('.mlx-outline__item[data-current]') !== null,
    }));
    check(slug + ': Free play leaves the lesson but keeps the setup', free.free && !free.url.includes('lesson=') && !free.current && free.url === kept, JSON.stringify({ ...free, kept }));
    await startLesson(page, 2);
    await new Promise((r) => setTimeout(r, 500));
  }

  // The control row keeps one rhythm: every caption, label and control line up across the bar.
  const rhythm = await page.evaluate(() => {
    const tops = (selector) => [...document.querySelectorAll(selector)].map((el) => Math.round(el.getBoundingClientRect().top));
    const distinct = (list) => [...new Set(list)];
    return {
      captions: distinct(tops('.mlx-studio__bar .mlx-toprow__caption')),
      labels: distinct(tops('.mlx-studio__bar .mlx-transport__label, .mlx-studio__bar .mlx-toprow__cell .mlx-field__head')),
      controls: distinct([
        ...tops('.mlx-transport__buttons'),
        ...tops('.mlx-speed select'),
        ...tops('.mlx-keys-menu .mlx-popover__button'),
        ...tops('.mlx-toprow__cell .mlx-stepper--field, .mlx-toprow__cell .mlx-select select'),
        ...tops('.mlx-toprow__more'),
      ]),
    };
  });
  check(slug + ': the bar captions, labels and controls each sit on one line', rhythm.captions.length === 1 && rhythm.labels.length === 1 && rhythm.controls.length === 1, JSON.stringify(rhythm));

  // Full page hides the site header and title strip and moves the title into the bar.
  await page.click('.mlx-studio__link[aria-label="Full page"]');
  await new Promise((r) => setTimeout(r, 300));
  const full = await page.evaluate(() => ({
    attr: document.documentElement.hasAttribute('data-mlx-full'),
    header: getComputedStyle(document.querySelector('.mlx-header')).display,
    strip: document.querySelector('.mlx-studio__head') !== null,
    title: document.querySelector('.mlx-studio__barend .mlx-studio__title')?.textContent?.trim() ?? '',
    fits: Math.abs(Math.round(document.querySelector('.mlx-studio').getBoundingClientRect().bottom) - window.innerHeight) <= 2,
    top: Math.round(document.querySelector('.mlx-studio').getBoundingClientRect().top),
  }));
  check(slug + ': full page hides the chrome and keeps the title in the bar', full.attr && full.header === 'none' && !full.strip && full.title.length > 3 && full.fits && full.top === 0, JSON.stringify(full));
  await page.keyboard.press('f');
  await new Promise((r) => setTimeout(r, 300));
  const back = await page.evaluate(() => ({
    attr: document.documentElement.hasAttribute('data-mlx-full'),
    strip: document.querySelector('.mlx-studio__head') !== null,
  }));
  check(slug + ': F leaves full page', !back.attr && back.strip, JSON.stringify(back));

  // More reveals the advanced knobs, Less hides them, and the bar never reflows.
  const barHeight = () => page.$eval('.mlx-studio__bar', (el) => Math.round(el.getBoundingClientRect().height));
  const cellsBefore = await page.$$eval('.mlx-toprow__cell', (els) => els.length);
  if ((await page.$('.mlx-toprow__more')) !== null) {
    const barClosed = await barHeight();
    await page.click('.mlx-toprow__more');
    await new Promise((r) => setTimeout(r, 300));
    const cellsMore = await page.$$eval('.mlx-toprow__cell', (els) => els.length);
    const barOpen = await barHeight();
    // The free-play pages keep every knob on one line even with More open; the older pages are not held to it yet.
    if (FREE_PLAY.has(slug)) check(slug + ': the bar keeps one line whether More is open or not', barOpen === barClosed, barClosed + ' -> ' + barOpen);
    await page.click('.mlx-toprow__more');
    await new Promise((r) => setTimeout(r, 300));
    const cellsLess = await page.$$eval('.mlx-toprow__cell', (els) => els.length);
    // A preset picked earlier may have opened the set already; either way one click toggles and the next restores.
    check(slug + ': More reveals the advanced knobs and Less hides them', cellsMore !== cellsBefore && cellsLess === cellsBefore, cellsBefore + ' -> ' + cellsMore + ' -> ' + cellsLess);
  }
  const barBefore = await barHeight();
  const flipped = await page.evaluate(() => {
    const select = document.querySelector('.mlx-toprow .mlx-select select');
    if (!select) return false;
    const next = Array.from(select.options).find((o) => o.value !== select.value);
    if (!next) return false;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
    setter?.call(select, next.value);
    select.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  });
  await new Promise((r) => setTimeout(r, 300));
  const barAfter = await barHeight();
  if (flipped) check(slug + ': switching a top-row dropdown never reflows the bar', barBefore === barAfter, barBefore + ' -> ' + barAfter);

  // Sliders: dragging one must move its readout and the URL.
  const sliderCount = await page.$$eval('.mlx-slider', (els) => els.length);
  check(slug + ': has sliders in the rail', sliderCount >= 2, sliderCount + ' sliders');
  if (sliderCount > 0) {
    const readSlider = () =>
      page.$eval('.mlx-slider', (el) => el.closest('.mlx-field')?.querySelector('.mlx-field__value')?.textContent);
    const before = await readSlider();
    await page.evaluate(() => {
      const slider = document.querySelector('.mlx-slider');
      if (!slider) return;
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
      const next = Number(slider.value) < Number(slider.max) ? slider.max : slider.min;
      setter?.call(slider, next);
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await new Promise((r) => setTimeout(r, 500));
    check(slug + ': a slider accepts input', before !== (await readSlider()), before + ' -> ' + (await readSlider()));
  }

  // Steppers: pressing one must change something on screen.
  const stepperCount = await page.$$eval('.mlx-stepper--field', (els) => els.length);
  check(slug + ': has steppers', stepperCount >= 1, stepperCount + ' steppers');
  if (stepperCount > 0) {
    // The first stepper that is in use; a dimmed one is by design.
    const readStepper = () =>
      page.evaluate(() => {
        const live = [...document.querySelectorAll('.mlx-stepper--field')].find((el) => [...el.querySelectorAll('button')].some((b) => !b.disabled));
        return live?.querySelector('output')?.textContent ?? '';
      });
    const before = await readStepper();
    await page.evaluate(() => {
      const stepper = [...document.querySelectorAll('.mlx-stepper--field')].find((el) => [...el.querySelectorAll('button')].some((b) => !b.disabled));
      const target = [...(stepper?.querySelectorAll('button') ?? [])].find((b) => !b.disabled && b.getAttribute('aria-disabled') !== 'true');
      target?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0 }));
      target?.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, button: 0 }));
    });
    await new Promise((r) => setTimeout(r, 500));
    const changed = before !== (await readStepper());
    check(slug + ': a stepper accepts input', changed);
    const afterStepper = await collectCanvases(page);
    check(
      slug + ': canvases survive a parameter change',
      afterStepper.every((c) => !(c.width > 20 && c.height > 20) || c.painted),
      '',
    );
  }

  // Expand every collapsed control group first.
  await page.evaluate(() => {
    document.querySelectorAll('.mlx-control-group').forEach((group) => {
      if (!group.hasAttribute('data-open')) {
        group.querySelector('.mlx-control-group__head')?.click();
      }
    });
  });
  await new Promise((r) => setTimeout(r, 300));

  // Selects and toggles.
  const selectCount = await page.$$eval('.mlx-select select', (els) => els.length);
  const toggleCount = await page.$$eval('.mlx-switch', (els) => els.length);
  const segmentCount = await page.$$eval('.mlx-segmented__item', (els) => els.length);
  check(
    slug + ': has selects/toggles/segments',
    selectCount + toggleCount + segmentCount >= 3,
    selectCount + ' selects, ' + toggleCount + ' toggles, ' + segmentCount + ' segments',
  );

  // Change the dataset select if one exists, the most disruptive control.
  if (selectCount > 0) {
    const switched = await page.evaluate(() => {
      const select = document.querySelector('.mlx-select select');
      if (!select || select.options.length < 2) return false;
      const before = select.value;
      const next = Array.from(select.options).find((o) => o.value !== before);
      if (!next) return false;
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype,
        'value',
      )?.set;
      setter?.call(select, next.value);
      select.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    });
    await new Promise((r) => setTimeout(r, 700));
    const afterSelect = await collectCanvases(page);
    const stillPainted = afterSelect.every((c) => !(c.width > 20 && c.height > 20) || c.painted);
    check(slug + ': changing the dataset keeps everything drawn', switched && stillPainted);
  }

  // Toggle a display switch.
  if (toggleCount > 0) {
    await page.evaluate(() => {
      const sw = document.querySelector('.mlx-switch');
      sw?.click();
    });
    await new Promise((r) => setTimeout(r, 400));
    const afterToggle = await collectCanvases(page);
    check(
      slug + ': a display toggle does not break the view',
      afterToggle.every((c) => !(c.width > 20 && c.height > 20) || c.painted),
    );
  }

  // Metrics, insights, maths, code.
  const metricCount = await page.$$eval('.mlx-chip, .mlx-metric', (els) => els.length);
  check(slug + ': shows at least five metrics', metricCount >= 5, metricCount + ' metrics');

  // Every output panel can take the whole screen and give it back.
  const panelFull = await page.evaluate(() => {
    const button = document.querySelector('.mlx-studio__out .mlx-panel__expand');
    if (!button) return 'missing';
    button.click();
    return 'clicked';
  });
  await new Promise((r) => setTimeout(r, 250));
  const fullState = await page.evaluate(() => {
    const panel = document.querySelector('.mlx-panel--full');
    if (!panel) return null;
    const r = panel.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height) };
  });
  await page.keyboard.press('Escape');
  await new Promise((r) => setTimeout(r, 250));
  const fullClosed = await page.evaluate(() => document.querySelector('.mlx-panel--full') === null);
  check(
    slug + ': an output panel opens full screen and closes again',
    panelFull === 'clicked' && fullState !== null && fullState.w >= 1400 && fullClosed,
    JSON.stringify({ panelFull, fullState, fullClosed }),
  );

  // Canvas interaction: click in the middle of the biggest canvas.
  const biggest = await page.evaluate(() => {
    let best = null;
    let bestArea = 0;
    document.querySelectorAll('canvas').forEach((c, i) => {
      const r = c.getBoundingClientRect();
      if (r.width * r.height > bestArea) {
        bestArea = r.width * r.height;
        best = { index: i, x: r.left + r.width / 2, y: r.top + r.height / 2 };
      }
    });
    return best;
  });
  if (biggest) {
    await page.mouse.move(biggest.x, biggest.y);
    await page.mouse.down();
    await page.mouse.move(biggest.x + 30, biggest.y - 20, { steps: 6 });
    await page.mouse.up();
    await new Promise((r) => setTimeout(r, 400));
    const afterInteract = await collectCanvases(page);
    check(
      slug + ': pointer interaction on the main canvas is safe',
      afterInteract.every((c) => !(c.width > 20 && c.height > 20) || c.painted),
    );
  }

  check(slug + ': no console errors', problems.length === 0, problems.slice(0, 3).join(' | '));
  await page.screenshot({ path: path.join(shotDir, slug + '.png'), fullPage: false });
  await page.close();
}

/** The one-screen lessons on the two flagships: opens on its picture, a run plays to its outcome, the spotlight, a deep link, reduced motion. */
async function testLessons(browser, base) {
  console.log('\nLessons');
  const settle = (ms) => new Promise((r) => setTimeout(r, ms));
  const clickRun = (page, label) =>
    page.evaluate((t) => {
      const head = [...document.querySelectorAll('button.mlx-step__experiment-head')].find((b) => b.textContent.trim().startsWith(t));
      if (!head) return false;
      head.click();
      return true;
    }, label);
  const outcomeOf = (page, label) =>
    page.evaluate((t) => {
      const row = [...document.querySelectorAll('.mlx-step__experiment')].find((r) => r.textContent.trim().startsWith(t));
      if (!row || !row.hasAttribute('data-done')) return '';
      const say = row.querySelector('.mlx-step__experiment-say');
      return say && !say.classList.contains('mlx-step__experiment-say--running') ? say.textContent.trim() : '';
    }, label);
  const waitForOutcome = async (page, label, tries = 60) => {
    let text = '';
    for (let i = 0; i < tries && !text; i++) {
      await settle(250);
      text = await outcomeOf(page, label);
    }
    return text;
  };

  // Linear regression, learning rate: the lesson opens on its trained picture, the knob is spotlit, a run diverges live.
  const { page, problems } = await openPage(browser, base + 'explainer/linear-regression?lesson=learning-rate');
  await settle(1200);
  const opened = await readGuide(page);
  const counter = await page.$eval('.mlx-counter__value', (el) => el.textContent?.trim() ?? '');
  check('a lesson opens on its trained picture', opened.card && Number(counter.replace(/[^0-9]/g, '')) > 0, 'step ' + counter);
  check('the lesson explains the picture and names a run to try', /Try it:/.test(opened.text), opened.text.slice(0, 60));
  check('the numbers the text quotes sit under it as chips', (await page.$$('.mlx-step__numbers .mlx-step__number')).length >= 2);
  check('the lesson spotlights its knob', (await page.$('[data-mlx-spot][data-lesson-target="control:learningRate"]')) !== null);
  check('the spotlight carries its label on the page', (await page.$eval('.mlx-spot-tag', (el) => el.textContent?.trim() ?? '').catch(() => '')) === 'Learning rate (α)');
  const studioBefore = await page.evaluate(() => Math.round(document.querySelector('.mlx-studio')?.getBoundingClientRect().height ?? 0));
  const stageBefore = await page.evaluate(() => Math.round(document.querySelector('.mlx-studio__arch')?.getBoundingClientRect().width ?? 0));
  check('a run is one click', await clickRun(page, 'α = 2.2'));
  const running = await page.evaluate(() => document.querySelector('.mlx-step__experiment-say--running') !== null || document.querySelector('.mlx-transport__label')?.textContent?.trim() === 'Running');
  check('the run plays live', running);
  const outcome = await waitForOutcome(page, 'α = 2.2');
  check('the run ends and its outcome line quotes the numbers', /diverged/i.test(outcome) && /step \d+/i.test(outcome), outcome.slice(0, 70));
  check('the key idea stays on screen', (await page.$('.mlx-step__takeaway')) !== null);
  const fits = await page.evaluate(() => {
    const body = document.querySelector('.mlx-step__body');
    return body ? body.scrollHeight <= body.clientHeight + 2 : false;
  });
  check('the card fits without scrolling once a run is open', fits);
  const studioAfter = await page.evaluate(() => Math.round(document.querySelector('.mlx-studio')?.getBoundingClientRect().height ?? 0));
  const stageAfter = await page.evaluate(() => Math.round(document.querySelector('.mlx-studio__arch')?.getBoundingClientRect().width ?? 0));
  check('the outcome does not change the studio height', studioBefore === studioAfter, studioBefore + ' -> ' + studioAfter);
  check('the stage keeps its width through a run', stageBefore === stageAfter, stageBefore + ' -> ' + stageAfter);
  check('the lesson is announced to screen readers', (await page.$eval('.mlx-lesson [role="status"]', (el) => el.textContent?.trim().length ?? 0)) > 10);
  check('lesson page logs no errors', problems.length === 0, problems.slice(0, 2).join(' | '));
  await page.screenshot({ path: path.join(shotDir, 'lesson-outcome.png'), fullPage: false });
  await page.close();

  // Neural network: a deep link opens trained on the lesson's preset; a run makes its own change and names it.
  const { page: nn } = await openPage(browser, base + 'explainer/neural-network?lesson=no-hidden');
  await settle(1200);
  const linear = await readGuide(nn);
  check('a deep link restores the lesson', linear.selected === 1 && linear.card, linear.name);
  check('a bare deep link applies the lesson preset', (await nn.$eval('.mlx-outline__item[data-current] .mlx-outline__name', (el) => el.textContent?.trim())) === 'No hidden layer');
  const layersBefore = await nn.$eval('.mlx-net-head__layers', (el) => el.textContent?.trim() ?? '');
  check('the deep link opens on the trained network', Number((await nn.$eval('.mlx-counter__value', (el) => el.textContent?.trim() ?? '')).replace(/[^0-9]/g, '')) > 0 && /no hidden layer/.test(layersBefore), layersBefore.slice(0, 40));
  check('a run makes the change itself', await clickRun(nn, 'Add a hidden layer'));
  await settle(600);
  const layersAfter = await nn.$eval('.mlx-net-head__layers', (el) => el.textContent?.trim() ?? '');
  check('the run changed the network instead of asking', /1 hidden layer/.test(layersAfter), layersAfter.slice(0, 40));
  check('the run spotlights the layers row', (await nn.$('[data-mlx-spot][data-lesson-target="custom:layers"]')) !== null);
  const changed = await waitForOutcome(nn, 'Add a hidden layer', 120);
  check('the run completes and its outcome names the change', /hidden layer/i.test(changed), changed.slice(0, 60));
  await nn.close();

  // Reduced motion: a run steps to its outcome without animating.
  const rm = await browser.newPage();
  await rm.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  await rm.goto(base + 'explainer/neural-network?lesson=feature-learning', { waitUntil: 'networkidle2', timeout: 60_000 });
  await settle(1200);
  await clickRun(rm, 'Hand in x₁x₂');
  const quiet = await waitForOutcome(rm, 'Hand in x₁x₂', 40);
  const label = await rm.$eval('.mlx-transport__label', (el) => el.textContent?.trim() ?? '');
  check('a run respects reduced motion', quiet.length > 0 && label !== 'Running', label + ' · ' + quiet.slice(0, 40));
  await rm.close();
}

/** The lesson panel: the head, the step card and the outline. */
async function readGuide(page) {
  return page.evaluate(() => ({
    name: document.querySelector('.mlx-lesson__title, .mlx-lesson[data-free] .mlx-lesson__count, .mlx-presets .mlx-lesson__count')?.textContent?.trim() ?? '',
    free: document.querySelector('.mlx-lesson')?.hasAttribute('data-free') ?? false,
    start: document.querySelector('.mlx-lesson__start button')?.textContent?.trim() ?? '',
    sections: document.querySelectorAll('.mlx-outline__section').length,
    // Index of the open lesson in the outline, -1 in free play.
    selected: [...document.querySelectorAll('.mlx-outline__item')].findIndex((el) => el.hasAttribute('data-current')),
    card: document.querySelector('.mlx-step') !== null,
    runs: document.querySelectorAll('.mlx-step__body:not(.mlx-step__body--gauge) .mlx-step__experiment').length,
    text: document.querySelector('.mlx-step__text')?.textContent?.trim() ?? '',
    status: document.querySelector('.mlx-lesson__status')?.textContent?.trim() ?? '',
    preset: document.querySelector('.mlx-presets select')?.value ?? '',
    presetOptions: document.querySelectorAll('.mlx-presets select option').length,
    presetNote: document.querySelector('.mlx-presets .mlx-field__note')?.textContent?.trim() ?? '',
  }));
}

async function testResponsive(browser, base) {
  console.log('\nResponsive layout');
  for (const viewport of VIEWPORTS) {
    for (const slug of ['linear-regression', 'neural-network']) {
      const { page, problems } = await openPage(browser, base + 'explainer/' + slug, viewport);
      await new Promise((r) => setTimeout(r, 800));

      const overflow = await page.evaluate(() => {
        const doc = document.documentElement;
        return { scrollWidth: doc.scrollWidth, clientWidth: doc.clientWidth };
      });
      check(
        viewport.name + ' (' + viewport.width + 'px) ' + slug + ': no horizontal overflow',
        overflow.scrollWidth <= overflow.clientWidth + 2,
        overflow.scrollWidth + ' > ' + overflow.clientWidth,
      );

      // Anything wider than the viewport is a bug unless a container scrolls it.
      const wideElements = await page.evaluate((vw) => {
        const scrollsHorizontally = (el) => {
          const x = getComputedStyle(el).overflowX;
          return x === 'auto' || x === 'scroll' || x === 'hidden' || x === 'clip';
        };
        const insideScroller = (el) => {
          for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
            if (scrollsHorizontally(node)) return true;
          }
          return false;
        };
        const bad = [];
        document.querySelectorAll('body *').forEach((el) => {
          const r = el.getBoundingClientRect();
          if (r.width > vw + 4 && r.height > 4) {
            if (scrollsHorizontally(el) || insideScroller(el)) return;
            bad.push((el.className || el.tagName).toString().slice(0, 50) + ' @' + Math.round(r.width));
          }
        });
        return bad.slice(0, 4);
      }, viewport.width);
      check(
        viewport.name + ' ' + slug + ': nothing exceeds the viewport width',
        wideElements.length === 0,
        wideElements.join(' | '),
      );

      const canvases = await collectCanvases(page);
      const blank = canvases.filter((c) => c.width > 20 && c.height > 20 && !c.painted);
      check(viewport.name + ' ' + slug + ': canvases paint at this size', blank.length === 0);

      // On narrow screens the columns stack.
      if (viewport.width < 1100) {
        const order = await page.evaluate(() => {
          const top = (sel) => document.querySelector(sel)?.getBoundingClientRect().top ?? NaN;
          return { stage: top('.mlx-studio__stage'), out: top('.mlx-studio__out'), rail: top('.mlx-studio__rail') };
        });
        check(viewport.name + ' ' + slug + ': the output column stacks below the stage', order.out > order.stage, JSON.stringify(order));
        if (viewport.width <= 760) {
          check(viewport.name + ' ' + slug + ': the lesson comes first on a phone', order.rail < order.stage, JSON.stringify(order));
          const phone = await page.evaluate(() => {
            const canvas = document.querySelector('.mlx-studio__arch canvas')?.getBoundingClientRect();
            const knobs = document.querySelector('.mlx-studio__knobs');
            const bar = document.querySelector('.mlx-studio__bar');
            return {
              canvasTop: canvas ? canvas.top : NaN,
              canvasHeight: canvas ? canvas.height : 0,
              vh: window.innerHeight,
              knobsHidden: knobs ? knobs.offsetParent === null : false,
              sticky: bar ? getComputedStyle(bar).position : '',
            };
          });
          check(viewport.name + ' ' + slug + ': the picture follows the lesson on a phone', phone.canvasTop < phone.vh * 1.5 && phone.canvasHeight >= 200, JSON.stringify(phone));
          check(viewport.name + ' ' + slug + ': the knobs fold behind Settings on a phone', phone.knobsHidden && phone.sticky === 'sticky', JSON.stringify(phone));
        } else {
          // The folded tiles only get their tracks once More opens them.
          await page.click('.mlx-strip__more');
          await new Promise((r) => setTimeout(r, 150));
          const tablet = await page.evaluate(() => {
            const panels = Array.from(document.querySelectorAll('.mlx-studio__out .mlx-panel')).map((p) => p.getBoundingClientRect());
            const metrics = document.querySelector('.mlx-studio__out .mlx-metrics');
            return {
              sideBySide: panels.length >= 2 && Math.abs(panels[0].top - panels[1].top) < 2,
              metricColumns: metrics ? getComputedStyle(metrics).gridTemplateColumns.split(' ').length : 0,
            };
          });
          check(viewport.name + ' ' + slug + ': the two output panels sit side by side on a tablet', tablet.sideBySide && tablet.metricColumns === 4, JSON.stringify(tablet));
        }
      }

      check(viewport.name + ' ' + slug + ': no console errors', problems.length === 0, problems.slice(0, 2).join(' | '));
      await page.screenshot({
        path: path.join(shotDir, slug + '-' + viewport.name + '.png'),
        fullPage: false,
      });
      await page.close();
    }
  }
}

async function testTheme(browser, base) {
  console.log('\nTheme');
  const { page, problems } = await openPage(browser, base + 'explainer/k-means');
  await new Promise((r) => setTimeout(r, 700));

  const before = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  await page.click('.mlx-theme-toggle');
  await new Promise((r) => setTimeout(r, 700));
  const after = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  check('theme toggle flips the theme', before !== after, before + ' -> ' + after);

  const canvases = await collectCanvases(page);
  const blank = canvases.filter((c) => c.width > 20 && c.height > 20 && !c.painted);
  check('canvases repaint after a theme flip', blank.length === 0, blank.length + ' blank');

  // The body background must actually change.
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  check('dark theme paints a dark background', after !== 'dark' || bg !== 'rgb(255, 255, 255)', bg);

  check('theme flip logs no errors', problems.length === 0, problems.slice(0, 2).join(' | '));
  await page.screenshot({ path: path.join(shotDir, 'theme-' + after + '.png') });
  await page.close();
}

async function testDeepLink(browser, base) {
  console.log('\nShareable configuration');
  const { page, problems } = await openPage(browser, base + 'explainer/linear-regression');
  await new Promise((r) => setTimeout(r, 600));

  // A captured URL restores the params and the lesson in a fresh page. Next moves to the following lesson, which loads its own preset.
  await startLesson(page, 2);
  await new Promise((r) => setTimeout(r, 600));
  await page.click('.mlx-lesson__arrow--next');
  await new Promise((r) => setTimeout(r, 400));
  const shared = page.url();
  const lessonId = shared.match(/lesson=([a-z-]+)/)?.[1] ?? '';
  check('a preset writes parameters into the URL', shared.includes('?') && lessonId.length > 0 && shared.includes('='), shared.split('/').pop()?.slice(0, 70) ?? '');
  await page.close();

  if (shared.includes('?')) {
    const { page: fresh, problems: freshProblems } = await openPage(browser, shared);
    await new Promise((r) => setTimeout(r, 800));
    const activePreset = await fresh.evaluate(
      () => document.querySelector('.mlx-outline__item[data-current] .mlx-outline__name')?.textContent?.trim() ?? '',
    );
    check(
      'reloading a shared URL restores that configuration',
      activePreset.length > 0,
      activePreset || '(none)',
    );
    const freshGuide = await readGuide(fresh);
    check('reloading a shared URL restores the lesson', fresh.url().includes('lesson=' + lessonId) && freshGuide.card, freshGuide.name);
    const canvases = await collectCanvases(fresh);
    check('the shared configuration renders', canvases.some((c) => c.painted));
    check('shared URL logs no errors', freshProblems.length === 0, freshProblems.slice(0, 2).join(' | '));
    await fresh.close();
  }

  // The GitHub Pages bounce: 404.html stashes the path and loads the catalog, which must open the explainer straight away.
  const bounced = await browser.newPage();
  const stashed = new URL(base).pathname + 'explainer/knn';
  await bounced.evaluateOnNewDocument((target) => {
    sessionStorage.setItem('mlx-redirect', target);
    window.__sawHome = false;
    new MutationObserver(() => {
      if (document.querySelector('.mlx-home')) window.__sawHome = true;
    }).observe(document.documentElement, { childList: true, subtree: true });
  }, stashed);
  await bounced.goto(base, { waitUntil: 'networkidle2', timeout: 30_000 });
  await bounced.waitForSelector('.mlx-studio-page:not(.mlx-skeleton) canvas', { timeout: 20_000 }).catch(() => null);
  const bouncedTitle = await bounced.evaluate(() => document.querySelector('.mlx-studio__title')?.textContent?.trim() ?? '');
  const sawHome = await bounced.evaluate(() => window.__sawHome);
  check('a bounced deep link opens its explainer', bounced.url().includes('/explainer/knn') && bouncedTitle.length > 0, bouncedTitle);
  check('a bounced deep link never shows the catalog first', !sawHome);
  await bounced.close();

  // 404 handling
  const { page: missing } = await openPage(browser, base + 'explainer/does-not-exist');
  await new Promise((r) => setTimeout(r, 400));
  const redirected = missing.url();
  check('an unknown explainer redirects to the catalog', !redirected.includes('does-not-exist'), redirected);
  await missing.close();
}

run().catch((error) => {
  console.error('\nBrowser tests could not run: ' + error.message);
  process.exit(1);
});
