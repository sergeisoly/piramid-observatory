import 'katex/dist/katex.min.css';
import '@fontsource-variable/source-serif-4/opsz.css';
import '@fontsource-variable/source-serif-4/opsz-italic.css';
import '@fontsource-variable/inter/wght.css';
import './styles.css';
import { Engine } from './engine/engine';
import { ExplorePanel } from './engine/explore';
import { manifest, ORDER } from './scenes';
import { breathe } from './lib/math';
import { markSvg, CORNERS } from './engine/mark';
import { TEAM_ROLE, TEAM_CHAPTERS, TEAM_LABEL, TITLES, PILLAR_OF } from './engine/cards';
import { CSS as INK, MARK, PILLAR_HUE } from './engine/palette';
import { pretty } from './engine/hud';
import { AUTHOR, POSTER_CREDIT_HTML } from './engine/credits';

/**
 * URL switches (all optional):
 *   ?only=id1,id2   build just these chapters (fast iteration)
 *   ?ch=id&t=12.5   start at local time t of chapter id
 *   ?T=95           start at global time T
 *   ?shot=1         no boot button, paused at the requested time (used by tools/shot.mjs)
 *   ?autostart=1    skip the boot button and play
 *   ?rate=0.75      playback speed (0.5, 0.75, 1, 1.25, or 1.5)
 *   ?slides=1       start in slides mode: playback stops after every slide, the arrow keys step
 *   ?details=1      open this slide's authored details, when present
 *   ?explore=1      open the notes at start (the film is then paused)
 *   ?hot=id         with Explore: open the explanation of that hotspot
 *   ?hover=x,y      with Explore: probe the figure at that frame-pixel position (1600 × 900 frame) and pin the result
 *   ?p.key=value    override a chapter control (see ChapterFactory.controls), e.g. ?only=meanfield&p.beta=400
 */
const q = new URLSearchParams(location.search);

// Dev edits trigger a batched full reload. Carry the reading position through it, then forget it.
// Normal refreshes, new visits and production builds keep their usual poster/deep-link behavior.
const DEV_RELOAD_KEY = 'piramid:dev-reload:v1';
type DevReloadState = {
  url: string; savedAt: number; chapter: string; time: number; slides: boolean; notes: boolean; rate: number; page: string | null;
};
function takeDevReloadState(): DevReloadState | null {
  if (!import.meta.hot) return null;
  try {
    const raw = sessionStorage.getItem(DEV_RELOAD_KEY);
    sessionStorage.removeItem(DEV_RELOAD_KEY); // one use, including invalid or stale data
    if (!raw) return null;
    const state = JSON.parse(raw) as Partial<DevReloadState>;
    const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
    const age = Date.now() - (state.savedAt ?? NaN);
    if (nav?.type !== 'reload' || state.url !== location.href || !Number.isFinite(age) || age < 0 || age > 30_000
      || typeof state.chapter !== 'string' || typeof state.time !== 'number' || !Number.isFinite(state.time)
      || typeof state.slides !== 'boolean' || typeof state.notes !== 'boolean'
      || typeof state.rate !== 'number' || !Number.isFinite(state.rate)
      || !(state.page === null || typeof state.page === 'string')) return null;
    return state as DevReloadState;
  } catch { return null; } // blocked storage must never stop the film
}
const devReload = takeDevReloadState();
const bootEl = document.getElementById('boot')!;
// The opening screen is the film's poster. It has to work for a visitor who never presses Play: it
// says what PIRAMID is, shows how the film's chapters sit in the program, and names who made it.
// The figures take a few seconds to compute, and a browser lets sound start only after a click, so
// the poster shows the progress and then offers its buttons.
const tocRow = (id: string) => `<button class="toc__row" data-ch="${id}" disabled><i>${String(ORDER.indexOf(id as (typeof ORDER)[number])).padStart(2, '0')}</i><span>${TITLES[id] ?? id}</span></button>`;
// The blog mounts the built film under this path in both production and its local preview.
const blogMount = !import.meta.env.DEV && /^\/piramid-observatory\/(?:index\.html)?$/.test(location.pathname);
const blogUrl = blogMount ? '/' : AUTHOR.url;
bootEl.innerHTML = `
  <div class="poster">
    <div class="boot__warn" hidden></div>
    <h1 class="poster__title">PIRAMID<i>Observatory</i></h1>
    <div class="poster__body">
    <p class="poster__sub">PIRAMID uses ideas from statistical physics to explain how neural networks learn and use patterns in data, with the goal of making AI safer.</p>
    <p class="poster__by">${POSTER_CREDIT_HTML} <a href="${blogUrl}">Back to blog</a></p>
    <div class="poster__start">
      <button class="boot__go" disabled title="One slide at a time: the arrow keys step forward and back">Read as slides</button>
      <button class="boot__alt" disabled title="The same slides, advancing by themselves"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>Play as a film</button>
      <span class="boot__status" role="status">Computing the figures…</span>
    </div>
    <div class="boot__bar" aria-hidden="true"></div>
    <details class="boot__details" hidden><summary>What went wrong</summary><div class="boot__log"></div></details>
    </div>
    <div class="poster__map">
      ${markSvg({ size: 420 })}
      ${CORNERS.map((c) => `<div class="toc toc--${c}" style="--c:${INK[PILLAR_HUE[c]]}"><span class="toc__team">${TEAM_LABEL[c]}</span><span class="toc__q">${TEAM_ROLE[c]}</span>${TEAM_CHAPTERS[c].map(tocRow).join('')}</div>`).join('')}
      <div class="toc toc--frame"><span class="toc__team">Across the teams</span>${tocRow('scales')}${tocRow('finale')}</div>
    </div>
    <p class="poster__foot">PIRAMID describes its program at <a href="https://princint.ai/piramid/" target="_blank" rel="noopener noreferrer">princint.ai/piramid</a>.</p>
  </div>`;
if (q.has('card')) bootEl.classList.add('is-still'); // the poster as a picture (the link-preview image is a capture of it)
const logEl = bootEl.querySelector('.boot__log') as HTMLElement;
const warnEl = bootEl.querySelector('.boot__warn') as HTMLElement;
const barEl = bootEl.querySelector('.boot__bar') as HTMLElement;
const statusEl = bootEl.querySelector('.boot__status') as HTMLElement;
const detailsEl = bootEl.querySelector('.boot__details') as HTMLDetailsElement;
/** A notice above the button, for a browser that cannot run the film as intended. */
const warn = (text: string) => { warnEl.textContent = text; warnEl.hidden = false; };
const goBtn = bootEl.querySelector('.boot__go') as HTMLButtonElement;
const altBtn = bootEl.querySelector('.boot__alt') as HTMLButtonElement;

function row(name: string, status: string) {
  const el = document.createElement('div');
  el.className = 'boot__row';
  el.innerHTML = `<span class="n"></span><span class="d"></span><span class="s"></span>`;
  (el.children[0] as HTMLElement).textContent = name;
  const s = el.children[2] as HTMLElement;
  s.textContent = status;
  logEl.appendChild(el);
  return {
    name: (t: string) => ((el.children[0] as HTMLElement).textContent = t),
    set: (t: string) => (s.textContent = t),
    ok: (t: string) => { s.textContent = t; el.classList.add('is-ok'); },
    err: (t: string) => { s.textContent = t; el.classList.add('is-err'); detailsEl.hidden = false; detailsEl.open = true; },
  };
}

// p.* overrides apply to one chapter: the one named by ?ch=, or the only one being built.
const urlParams: Record<string, string> = {};
for (const [k, v] of q) if (k.startsWith('p.')) urlParams[k.slice(2)] = v;
const onlyIds = q.get('only')?.split(',');
const paramTarget = q.get('ch') ?? (onlyIds?.length === 1 ? onlyIds[0] : null);

const hudEl = document.getElementById('hud')!;
hudEl.inert = true; // the film's controls are behind the opening screen until the film starts
const engine = new Engine(document.getElementById('stage')!, hudEl);
(window as any).__piramid = engine;

/** The film's typefaces, loaded before anything is measured or photographed. */
const fontsReady = Promise.all([
  '600 40px "Source Serif 4 Variable"', '400 23px "Source Serif 4 Variable"', 'italic 400 20px "Source Serif 4 Variable"',
  '400 13px "Inter Variable"', '600 13px "Inter Variable"',
].map((f) => document.fonts.load(f))).then(() => document.fonts.ready).catch(() => undefined);

async function main() {
  const r0 = row('renderer', 'requesting GPU adapter…');
  await engine.init();
  r0.ok(`${engine.backend} · three r187 · TSL`);
  if (engine.backend !== 'WebGPU')
    warn('This browser has no WebGPU. This presentation is running on a WebGL2 fallback that has not been tested, and some figures may be wrong or missing. A current desktop Chrome, Edge or Safari has WebGPU.');
  await breathe();

  // The log lists the chapters in film order. A chapter with a long computation starts it now, off the
  // main thread (prepare), and is built last, so that the computation runs while the others build.
  const list = manifest(onlyIds);
  const rows = list.map((m) => row(m.id, 'loading…'));
  barEl.innerHTML = list.map((m) => `<i style="--c:${PILLAR_OF[m.id] && PILLAR_OF[m.id] !== 'frame' ? MARK[PILLAR_HUE[PILLAR_OF[m.id]]] : 'var(--ink)'}"></i>`).join('');
  let nDone = 0;
  const tick = (i: number, ok: boolean) => {
    barEl.children[i]?.classList.add(ok ? 'is-on' : 'is-err');
    statusEl.textContent = `Computing the figures · ${++nDone} of ${list.length}`;
  };
  await breathe();
  const facs = await Promise.all(list.map((m) => m.load().catch((e) => { console.error(`chapter ${m.id} failed to load`, e); return null; })));
  const paramsOf = (i: number) => (list[i].id === paramTarget ? urlParams : undefined);
  facs.forEach((f, i) => {
    if (!f) { tick(i, false); return rows[i].err('failed to load'); }
    rows[i].name(f.label);
    rows[i].set(f.prepare ? 'computing in the background…' : 'waiting…');
    try { f.prepare?.(paramsOf(i)); } catch (e) { console.warn('prepare failed for', list[i].id, e); }
  });
  const order = list.map((_, i) => i).sort((a, b) => +!!facs[a]?.prepare - +!!facs[b]?.prepare || a - b);
  const built: (Awaited<ReturnType<NonNullable<(typeof facs)[number]>['build']>> | null)[] = list.map(() => null);
  for (const i of order) {
    const f = facs[i], r = rows[i];
    if (!f) continue;
    r.set('computing…');
    await breathe();
    const t0 = performance.now();
    try {
      const ch = await f.build(
        {
          renderer: engine.renderer,
          log: async (msg) => { r.set(msg); await breathe(); },
          film: engine,
        },
        paramsOf(i),
      );
      await engine.warm(ch);
      built[i] = ch;
      r.ok(`${Math.round(performance.now() - t0)} ms ✓`);
      tick(i, true);
    } catch (e: any) {
      console.error(`chapter ${list[i].id} failed to build`, e);
      r.err(`failed: ${e?.message ?? e}`);
      tick(i, false);
    }
  }
  // film order, whatever the build order was
  built.forEach((ch, i) => { if (ch) engine.addChapter(ch, facs[i]!, paramsOf(i) ? { ...urlParams } : {}); });
  if (!engine.chapters.length) throw new Error('no chapters built');
  // A chapter that failed is left out. Say so next to the button, so that nobody takes the rest for the whole film.
  const missing = list.map((m, i) => (built[i] ? null : facs[i]?.label ?? m.id)).filter((x): x is string => x !== null);
  if (missing.length)
    warn(`This browser could not load or compute ${missing.length === 1 ? 'one chapter' : `${missing.length} chapters`}, and this presentation runs without ${missing.length === 1 ? 'it' : 'them'}: ${missing.join(', ')}. The log below has the error. Reloading tries again.`);

  // initial position
  if (q.has('slides')) engine.setSlides(true);
  const chId = q.get('ch');
  if (chId && engine.indexOf(chId) >= 0) {
    const i = engine.indexOf(chId);
    const t = parseFloat(q.get('t') ?? '');
    if (Number.isFinite(t)) {
      const local = Math.max(-engine.intros[i], Math.min(engine.chapters[i].duration - 0.001, t));
      engine.seek(engine.starts[i] + local);
    } else engine.goto(i);
  } else if (q.has('T')) engine.seek(parseFloat(q.get('T')!));

  const explore = new ExplorePanel(engine.hud.element, engine);
  engine.onActive = (i) => explore.onChapter(i);
  const toggleExplore = (force?: boolean) => explore.toggle(force);
  engine.hud.onExplore = () => toggleExplore();
  (window as any).__explore = explore;

  engine.ready();
  if (q.has('rate')) engine.setRate(parseFloat(q.get('rate')!));
  const shot = q.has('shot');
  await fontsReady;
  // the poster's table of contents starts the film at a chapter
  const tocRows = [...bootEl.querySelectorAll<HTMLButtonElement>('.toc__row')];
  tocRows.forEach((b) => {
    const i = engine.indexOf(b.dataset.ch ?? '');
    if (i < 0) return;
    (b.querySelector('span') as HTMLElement).textContent = pretty(engine.chapters[i].title);
    b.addEventListener('click', () => { if (started || engine.lost) return; engine.goto(i); void begin(); });
  });
  let started = false;
  let hasBegun = false;
  let launchEpoch = 0;
  engine.hud.onReplay = (slides) => {
    engine.setSlides(slides);
    engine.seek(0);
    engine.play();
    engine.hud.focusTransport();
  };
  engine.hud.onEndExplore = () => {
    engine.goto(1);
    toggleExplore(true);
    engine.hud.focusTransport();
  };
  /** Sound is off until the viewer asks for it (M): the score is loaded on that first request. */
  let scoreState: 'none' | 'loading' | 'on' = 'none';
  const startScore = async () => {
    if (scoreState !== 'none') return;
    scoreState = 'loading';
    try {
      const { Score } = await import('./engine/score');
      const score = new Score();
      score.attach(engine);
      (window as any).__score = score;
      let muted = false;
      engine.hud.setSound(muted);
      engine.hud.onSound = () => { score.toggleMute(); muted = !muted; engine.hud.setSound(muted); };
      scoreState = 'on';
    } catch (e) {
      console.warn('score unavailable', e);
      scoreState = 'none';
    }
  };
  const begin = async (restoring = false) => {
    if (started || engine.lost) return;
    const firstStart = !hasBegun, epoch = ++launchEpoch;
    hasBegun = true;
    started = true;
    bootEl.classList.add('is-done');
    bootEl.inert = true; // the faded card must not keep its buttons in the Tab order
    hudEl.inert = false;
    engine.hud.show();
    if ((!shot || !firstStart) && !restoring) engine.play();
    if (!shot && !restoring) engine.hud.focusTransport();
    if (firstStart && !restoring && q.has('details') && q.get('details') !== '0') {
      // Authored details open after the slide has posed itself and established its note stack.
      const f1 = engine.frames;
      const open = () => {
        if (!started || epoch !== launchEpoch) return;
        if (engine.frames - f1 < 3) requestAnimationFrame(open);
        else explore.showDetails();
      };
      open();
    }
    if (firstStart && !restoring && q.has('explore')) {
      toggleExplore(true); // after play(), so that Explore's pause is the last word
      // deep links into the inspector; they wait for the chapter to have posed itself once
      const hot = q.get('hot'), hover = q.get('hover')?.split(',').map(Number);
      const f0 = engine.frames;
      const pin = () => {
        if (!started || epoch !== launchEpoch) return;
        if (engine.frames - f0 < 3) return void requestAnimationFrame(pin);
        if (hot) explore.inspector.pinHot(hot);
        else if (hover && hover.length === 2 && hover.every(isFinite)) explore.inspector.pinAt(hover[0], hover[1]);
      };
      if (hot || hover) pin();
    }
  };
  // These actions stay available after returning to the overview; all figures are already built.
  const slideCount = engine.slideInfo().count, mins = Math.round(engine.clockTotal / 60);
  statusEl.textContent = `${slideCount} slides · about ${mins} min as a film`;
  goBtn.disabled = false;
  goBtn.classList.add('is-armed');
  altBtn.disabled = false;
  tocRows.forEach((b) => { if (engine.indexOf(b.dataset.ch ?? '') >= 0) b.disabled = false; });
  const startFromPoster = (slides: boolean) => {
    if (started || engine.lost) return;
    if (hasBegun) engine.seek(0);
    engine.setSlides(slides);
    void begin();
  };
  goBtn.addEventListener('click', () => startFromPoster(true));
  altBtn.addEventListener('click', () => startFromPoster(false));
  engine.hud.onHome = () => {
    if (!started || engine.lost) return;
    engine.pause();
    toggleExplore(false);
    engine.seek(0);
    engine.resetView();
    engine.hud.hide();
    hudEl.inert = true;
    started = false;
    launchEpoch++; // cancel deferred deep-link work from the prior opening
    bootEl.inert = false;
    bootEl.classList.remove('is-done', 'is-still');
    bootEl.scrollTop = 0;
    goBtn.focus({ preventScroll: true });
  };
  // Register before taking away the poster so even an edit immediately after Start is recoverable.
  if (import.meta.hot) {
    import.meta.hot.on('vite:beforeFullReload', (payload) => {
      // Vite broadcasts HTML changes too; only arm restoration when this page will really reload.
      if (payload.path?.endsWith('.html')) {
        const pagePath = decodeURI(location.pathname), payloadPath = import.meta.env.BASE_URL + payload.path.slice(1);
        if (pagePath !== payloadPath && payload.path !== '/index.html' && !(pagePath.endsWith('/') && pagePath + 'index.html' === payloadPath)) return;
      }
      if (!started || engine.lost || engine.active < 0) return;
      const state: DevReloadState = {
        url: location.href, savedAt: Date.now(), chapter: engine.chapters[engine.active].id,
        time: engine.T - engine.starts[engine.active], slides: engine.slides, notes: explore.open, rate: engine.rate,
        page: explore.inspector.selectedPage,
      };
      try { sessionStorage.setItem(DEV_RELOAD_KEY, JSON.stringify(state)); } catch { /* storage is optional */ }
    });
  }
  const restoreChapter = devReload ? engine.indexOf(devReload.chapter) : -1;
  if (devReload && restoreChapter >= 0) {
    const i = restoreChapter;
    const time = Math.max(-engine.intros[i], Math.min(engine.chapters[i].duration - 0.001, devReload.time));
    engine.seek(engine.starts[i] + time);
    engine.setSlides(devReload.slides);
    engine.setRate(devReload.rate);
    await begin(true);
    engine.playing = false; // edits never make a slide or film run away while its owner is reading
    if (devReload.notes) {
      // Wait for the restored chapter to pose itself, so opening Notes does not use the prior chapter.
      const f0 = engine.frames, epoch = launchEpoch;
      const open = () => {
        if (!started || epoch !== launchEpoch) return;
        if (engine.frames - f0 < 2) return void requestAnimationFrame(open);
        toggleExplore(true);
        if (devReload.page) explore.inspector.selectPage(devReload.page);
      };
      open();
    }
  } else if (shot || q.has('autostart')) {
    await begin();
    if (shot) {
      // Let the first frames settle before signaling the frame grabber.
      const f0 = engine.frames;
      const wait = () => (engine.frames - f0 > 20 ? ((window as any).__shotReady = true) : requestAnimationFrame(wait));
      wait();
    }
  }

  // A button or link pressed with the mouse gives the keyboard back, so that Space keeps driving the film.
  // A button reached with Tab keeps its focus (a click from the keyboard has detail 0).
  document.addEventListener('click', (e) => {
    if (e.detail > 0) ((e.target as HTMLElement | null)?.closest('button, a') as HTMLElement | null)?.blur();
  });

  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || engine.lost) return;
    const k = e.key.toLowerCase();
    if (!started && k !== ' ' && k !== 'f') return; // the poster: only Space (and fullscreen) act
    if (k === 'home') { e.preventDefault(); engine.hud.onHome(); return; }
    // Keys an Explore control uses itself stay with it: arrows move a slider, space ticks a box,
    // letters search a menu. Everything else (space on a slider, E, F, M, H) still drives the film.
    const el = e.target as HTMLElement | null;
    // Space and Enter on a focused button or disclosure press that element. A link takes Enter only.
    if ((k === ' ' || k === 'enter') && el?.closest('button, summary')) return;
    if (k === 'enter' && el?.closest('a')) return;
    if (el?.tagName === 'SELECT' && k !== 'escape') return;
    if (el?.tagName === 'INPUT' && (k.startsWith('arrow') || ((el as HTMLInputElement).type === 'checkbox' && k === ' '))) return;
    if (el?.closest('.note__body, .note__rows') && (k === ' ' || k === 'pageup' || k === 'pagedown')) return;
    const slides = engine.slides && bootEl.classList.contains('is-done');
    if (e.repeat && ([' ', 's', 'f', 'h', 'e', 'd', 'm', 'arrowleft', 'arrowright', 'pageup', 'pagedown'].includes(k) || /^[0-9]$/.test(k))) {
      e.preventDefault();
      return;
    }
    if (k === ' ') { e.preventDefault(); if (!bootEl.classList.contains('is-done')) goBtn.click(); else if (slides) engine.stepSlide(1); else engine.toggle(); }
    else if (k === 'arrowright' || k === 'pagedown') { e.preventDefault(); if (slides) engine.stepSlide(1); else engine.step(1); }
    else if (k === 'arrowleft' || k === 'pageup') { e.preventDefault(); if (slides) engine.stepSlide(-1); else engine.step(-1); }
    else if (k === 's') engine.setSlides(!engine.slides);
    else if (!slides && (k === '<' || k === ',')) engine.stepRate(-1);
    else if (!slides && (k === '>' || k === '.')) engine.stepRate(1);
    else if (k === 'f') { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.(); }
    else if (k === 'h') engine.hud.toggleChrome();
    else if (k === 'e') toggleExplore();
    else if (k === 'd') explore.showDetails();
    else if (k === '[' || k === ']') explore.inspector.step(k === '[' ? -1 : 1);
    else if (k === 'escape') {
      // the keys panel closes first, then a pinned card, then the notes
      const hud = document.querySelector('.hud'), kb = document.querySelector('.kbtn');
      if (hud?.classList.contains('is-keys')) { hud.classList.remove('is-keys'); kb?.setAttribute('aria-expanded', 'false'); }
      else explore.escape();
    }
    else if (k === 'm') { if (scoreState === 'on') engine.hud.onSound(); else void startScore(); }
    else if (/^[0-9]$/.test(k)) engine.goto(parseInt(k, 10));
  });
}

main().catch((e) => {
  console.error(e);
  if (!('gpu' in navigator)) warn('This presentation needs WebGPU, and this browser does not have it. Open it in a current desktop Chrome, Edge or Safari.');
  row('fatal', '').err(String(e?.message ?? e));
});
