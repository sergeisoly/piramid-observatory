// Behaviour checks for V4: poster, playback, Slides, read-only Notes, rebuilds and device loss.
//   npm run check                      (the dev server must be running: npm run dev)
//   npm run check -- --url=http://127.0.0.1:5196/piramid-observatory/ --only=slides,modes
//
// Every check reads the app's state (engine fields, classes, numbers) and never the wording of a caption,
// so a change of text cannot fail it. A check that fails means the behaviour changed.
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';

const args = readFlags();
const base = (args.url ?? 'http://127.0.0.1:5199/').replace(/\/?$/, '/');
const only = args.only ? String(args.only).split(',') : null;
const groups = ['boot', 'modes', 'slides', 'pin', 'lost', 'entk'];
if (only?.some((id) => !groups.includes(id))) throw new Error(`Unknown check group; choose ${groups.join(', ')}`);
const want = (id) => !only || only.includes(id);

const browser = await launchChrome({ chrome: args.chrome, headful: args.headful, extraArgs: ['--autoplay-policy=no-user-gesture-required'] });
let failed = 0;
const check = (name, ok, info = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${!ok && info ? '  — ' + info : ''}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const open = async (query) => {
  const page = await browser.newPage();
  page.on('pageerror', (e) => { failed++; console.log('[pageerror]', e.message); });
  await page.goto(base + query, { waitUntil: 'load' });
  await page.waitForFunction(() => document.querySelector('.boot__go')?.classList.contains('is-armed') || document.getElementById('boot').classList.contains('is-done'), { timeout: 180000 });
  return page;
};
/** The app's modes, read from every place that stores them. */
const modes = (page) => page.evaluate(() => {
  const e = window.__piramid, x = window.__explore, hud = document.getElementById('hud');
  const act = [...document.querySelectorAll('.seg')].map((s, i) => (s.classList.contains('is-active') ? i : -1)).filter((i) => i >= 0);
  return {
    T: e.T, total: e.total, playing: e.playing, slides: e.slides, active: e.active, frames: e.frames, slide: e.slideInfo().index,
    explore: x.open, notesOpen: e.notesOpen, notesAvailable: x.inspector.hasNotes, inspector: x.inspector.open, exploreClass: hud.classList.contains('is-explore'),
    slidesClass: hud.classList.contains('is-slides'), minimal: hud.classList.contains('is-minimal'), barInert: !!document.querySelector('.hud__bar')?.inert,
    activeSegs: act, pin: x.inspector.pin?.kind ?? null, started: document.getElementById('boot').classList.contains('is-done'),
  };
});
/** What must hold in every reachable state. Returns the names of the rules that are broken. */
const broken = (m) => [
  ['T is a time inside the film', Number.isFinite(m.T) && m.T >= 0 && m.T < m.total],
  ['Notes session: panel, engine and inspector agree', m.explore === m.notesOpen && m.explore === m.inspector],
  ['Notes column appears only with authored pages', m.exploreClass === (m.explore && m.notesAvailable)],
  ['slides: engine and class agree', m.slides === m.slidesClass],
  ['hidden chrome is out of the Tab order', m.minimal === m.barInert],
  ['exactly the chapter on show is marked on the track', m.activeSegs.length === 1 && m.activeSegs[0] === m.active],
].filter((r) => !r[1]).map((r) => r[0]);

try {
// ── opening screen: Space starts Slides; fullscreen is also allowed ──
if (want('boot')) {
  const page = await open('?only=prologue,scales');
  for (const k of ['e', 'h', 's', '1', 'ArrowRight', 'm', 'Escape']) await page.keyboard.press(k);
  const before = await modes(page);
  check('opening screen: keys change nothing', !before.started && !before.explore && !before.minimal && !before.slides && before.T < 0.05, JSON.stringify(before));
  await page.keyboard.press(' ');
  await sleep(700);
  const m = await modes(page);
  check('opening screen: Space starts Slides from the beginning', m.started && m.playing && !m.explore && !m.minimal && m.slides && m.active === 0 && m.T < 2, JSON.stringify(m));
  await page.close();
}

// ── any key sequence leaves the modes consistent ──
if (want('modes')) {
  const page = await open('?autostart=1&only=prologue,scales,finale');
  const keys = [' ', 'ArrowRight', 'ArrowLeft', 's', 'e', 'h', 'Escape', '0', '1', '2', ',', '.'];
  let seed = 20261114;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const log = [], bad = [];
  for (let n = 0; n < 90 && !bad.length; n++) {
    const k = keys[Math.floor(rnd() * keys.length)];
    log.push(k);
    await page.keyboard.press(k);
    await sleep(60); // a few frames: the track mark follows the engine one frame later
    const b = broken(await modes(page));
    if (b.length) bad.push(`after [${log.join(' ')}]: ${b.join('; ')}`);
  }
  check('90 random key presses: the modes stay consistent after each', !bad.length, bad[0]);
  await page.close();
}

// ── slides ──
if (want('slides')) {
  const page = await open('?autostart=1&slides=1'); // the whole film
  const L = await page.evaluate(() => window.__piramid.slideList());
  // forward through every slide: each press ends on that slide's stop, paused
  await page.evaluate(() => { const e = window.__piramid; e.playing = false; e.seek(e.slideList()[0].a); e.setRate(1); });
  const bad = [];
  for (let k = 0; k < L.length; k++) {
    await page.evaluate((h) => { const e = window.__piramid; e.playing = true; e.T = Math.max(e.T, h - 0.3); }, L[k].h); // skip ahead: the stop is what is tested
    await page.waitForFunction(() => !window.__piramid.playing, { timeout: 5000 }).catch(() => {});
    const m = await modes(page);
    if (m.playing || Math.abs(m.T - L[k].h) > 1e-6 || m.slide !== k) bad.push(`slide ${k}: T ${m.T.toFixed(3)}, stop ${L[k].h.toFixed(3)}, playing ${m.playing}`);
    if (k + 1 < L.length) await page.keyboard.press('ArrowRight'); // starts the next slide
    await sleep(40);
  }
  check(`slides: playback stops on the stop of each of the ${L.length} slides`, !bad.length, bad.slice(0, 3).join(' | '));

  // a held slide stays held when playback is resumed on it (Explore opened while it played, then closed)
  await page.evaluate(() => { const e = window.__piramid; e.seek(e.slideList()[1].a + 0.2); e.playing = true; });
  await sleep(150);
  await page.keyboard.press('e');
  await page.keyboard.press(' ');
  await page.evaluate((h) => { const e = window.__piramid; e.T = Math.max(e.T, h - 0.3); }, L[1].h);
  await page.waitForFunction(() => !window.__piramid.playing, { timeout: 5000 }).catch(() => {});
  const held = await modes(page);
  await page.keyboard.press('e');
  await sleep(900);
  const after = await modes(page);
  check('slides: closing Explore on a held slide leaves it held', !after.playing && Math.abs(after.T - held.T) < 1e-6 && Math.abs(held.T - L[1].h) < 1e-6, JSON.stringify({ held: held.T, after: after.T, playing: after.playing }));

  // Film can be playing while held for reading. Switching to Slides must stop on that endpoint.
  const endpoint = await page.evaluate(() => {
    const e = window.__piramid, slide = e.slideList().find((s) => s.x > 0.1);
    e.setSlides(false); e.seek(slide.h - 0.04); e.playing = true;
    return slide.h;
  });
  await page.waitForFunction(() => window.__piramid.dwell > 0, { timeout: 5000 });
  await page.keyboard.press('s');
  await sleep(350);
  const stopped = await modes(page);
  check('Film reading hold switched to Slides stays at the held endpoint', stopped.slides && !stopped.playing && Math.abs(stopped.T - endpoint) < 1e-6, JSON.stringify({ endpoint, T: stopped.T, playing: stopped.playing }));

  // Back never moves forward
  await page.evaluate(() => { const e = window.__piramid; e.playing = false; e.seek(e.slideList()[0].a + 0.4); });
  await sleep(100);
  const a = await modes(page);
  await page.keyboard.press('ArrowLeft');
  await sleep(100);
  const b = await modes(page);
  await page.evaluate(() => { const e = window.__piramid; e.playing = false; e.seek(0.2); });
  await sleep(100);
  await page.keyboard.press('ArrowLeft');
  await sleep(100);
  const c = await modes(page);
  check('slides: Back on the first slide does not move forward, from inside it and from before its caption', b.T <= a.T && !b.playing && c.T <= 0.2 + 1e-9, JSON.stringify({ from: a.T, to: b.T, fromStart: c.T }));
  await page.close();
}

// ── read-only Notes survive an offscreen rebuild; pointerup has current touch coordinates ──
if (want('pin')) {
  const page = await open('?shot=1&only=prologue,scales&ch=scales&t=17&explore=1');
  await page.waitForFunction(() => [...document.querySelectorAll('.hot')].some((el) => el.style.display !== 'none'));
  const id = await page.evaluate(() => [...document.querySelectorAll('.hot')].find((el) => el.style.display !== 'none').dataset.id);
  await page.click(`.hot[data-id="${id}"]`);
  const before = await page.evaluate(() => ({ pin: window.__explore.inspector.pin, key: window.__piramid.hud.notePageKey }));
  await page.evaluate(async () => { const e = window.__piramid; await e.rebuild(e.factories.findIndex((factory, i) => factory && i !== e.active)); });
  await sleep(250);
  const after = await page.evaluate(() => ({ pin: window.__explore.inspector.pin, key: window.__piramid.hud.notePageKey }));
  const m = await modes(page);
  check('offscreen rebuild preserves the numbered Notes page and active track segment', before.pin?.kind === 'hot' && before.pin.id === after.pin?.id && before.key === after.key && !broken(m).length, JSON.stringify({ before, after, broken: broken(m) }));
  const touchX = 173, touchY = 281;
  await page.evaluate(() => {
    const e = window.__piramid, click = e.onStageClick;
    e.onStageClick = () => {
      const input = e.pointerInput();
      window.__touchInput = input && [input.x, input.y];
      click();
    };
  });
  const finger = await page.touchscreen.touchStart(touchX, touchY);
  await page.evaluate(() => {
    // Simulate a stale hover sample between press and release without a pointermove.
    window.__piramid.ptr.on = false;
  });
  await finger.end();
  const touch = await page.evaluate((x, y) => {
    const e = window.__piramid, box = document.getElementById('stage').getBoundingClientRect();
    return {
      actual: window.__touchInput, expected: [(x - box.left) / e.ui, (y - box.top) / e.ui],
      hover: e.pointerInput(), pin: window.__explore.inspector.pin,
    };
  }, touchX, touchY);
  const releasedAtTap = touch.actual?.every((v, i) => Math.abs(v - touch.expected[i]) < 1e-6);
  const sameNote = touch.pin?.kind === 'hot' && touch.pin.id === after.pin.id;
  check('touch release refreshes click coordinates and leaves no hover or new probe', releasedAtTap && touch.hover === null && sameNote, JSON.stringify(touch));
  await page.close();
}

// ── a lost graphics device is final ──
if (want('lost')) {
  const lose = (page) => page.evaluate(() => { const r = window.__piramid.renderer; r.onDeviceLost({ api: 'WebGPU', message: 'check', reason: 'unknown' }); r.onDeviceLost({ api: 'WebGPU', message: 'check', reason: 'unknown' }); });
  const page = await open('?autostart=1&only=prologue,scales');
  await sleep(400);
  await lose(page);
  await sleep(200);
  const a = await modes(page);
  for (const k of [' ', 'ArrowRight', '1', 's', 'e', ' ']) await page.keyboard.press(k);
  await sleep(600);
  const b = await modes(page), notices = await page.evaluate(() => document.querySelectorAll('.fatal').length);
  const same = b.frames === a.frames && b.T === a.T && !b.playing && b.slides === a.slides && b.explore === a.explore;
  check('device lost (renderer hook, not a real loss): no key changes anything, the frame loop has ended, one notice', same && notices === 1, JSON.stringify({ a, b, notices }));
  await page.close();
  const boot = await open('?only=prologue,scales');
  await lose(boot);
  await boot.keyboard.press(' ');
  await sleep(400);
  const m = await modes(boot);
  check('device lost on the opening screen: the film does not start', !m.started && !m.playing, JSON.stringify(m));
  await boot.close();
}

// Editable-Explore Defaults/cut-label tests are intentionally absent: V4 exposes frozen Notes.
// URL-parameter numerical checks remain applicable below.

// ── eNTK, second-layer kernel with few neurons: no ratio of two noise-level eigenvalues is reported ──
if (want('entk')) {
  const page = await open('?shot=1&only=entk&ch=entk&t=16.5&p.layer=K2&p.n=64');
  await page.waitForFunction('window.__shotReady === true', { timeout: 180000 });
  const out = await page.evaluate(() => {
    const e = window.__piramid, chapter = e.chapters[0], t = e.T - e.starts[0];
    const bad = [], values = (rows) => rows.map((row) => String(row.value));
    const multipliers = (rows) => values(rows).filter((value) => /\\times/.test(value));
    const panel = multipliers(chapter.readouts?.(t) ?? []), probes = new Set();
    for (const value of values(chapter.readouts?.(t) ?? [])) if (/NaN|Infinity/.test(value)) bad.push(value);
    for (let y = 100; y < 820; y += 8) for (let x = 40; x < 1560; x += 4) {
      const rows = chapter.probe?.(t, e.probeInput(x, y))?.rows ?? [];
      for (const value of multipliers(rows)) {
        probes.add(value);
        if (/NaN|Infinity/.test(value)) bad.push(value);
        for (const match of value.matchAll(/\\times\s*(-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?)/gi)) {
          const number = Number(match[1].replace(/,/g, ''));
          if (!Number.isFinite(number) || Math.abs(number) > 1e6) bad.push(value);
        }
      }
    }
    return { bad, panel, probeRatios: probes.size, matchingDrops: panel.slice(0, 2).filter((value) => probes.has(value)).length };
  });
  check('eNTK K2/n64: actual spectrum ratios are finite, bounded, and both reported drops agree with probes', !out.bad.length && out.panel.length >= 2 && out.probeRatios > 0 && out.matchingDrops === 2, JSON.stringify(out));
  await page.close();
}
} finally {
  await browser.close();
}
console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exitCode = failed ? 1 : 0;
