// Interaction smoke test of the film's shell. It drives the real page in headless Chrome and prints one
// line per check. It tests behavior (state, counts, positions in time), never the wording on screen.
//   node tools/smoke.mjs --url=http://127.0.0.1:5199      (exit code 1 if a check fails)
//   options: --only=prologue,meanfield,finale   a subset of chapters (default: the whole film)
// What it covers: the poster's two ways in, a card, the film's wait for reading time, optional authored
// details, the Notes layer, every chapter boundary in film and in slides mode with Notes closed
// and open, the closing board and the end card.
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';

const args = readFlags();
const base = args.url ?? 'http://127.0.0.1:5199';
const only = args.only ? `only=${args.only}&` : '';
const browser = await launchChrome({ chrome: args.chrome });
let failed = 0;
const errors = [];
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 300)); });

  const state = () => page.evaluate(() => {
    const e = window.__piramid, h = document.getElementById('hud'), cl = (c) => h.classList.contains(c);
    const vis = (sel) => { const el = document.querySelector(sel); if (!el) return false; const s = getComputedStyle(el); return s.display !== 'none' && s.visibility !== 'hidden' && +s.opacity > 0.5; };
    const c = e.chapters[e.active];
    return {
      T: e.T, total: e.total, playing: e.playing, active: e.active, id: c?.id, slides: e.slides, ended: e.ended,
      local: e.active >= 0 ? e.T - e.starts[e.active] : 0, dur: c?.duration ?? 0,
      card: cl('is-card'), titleVisible: vis('.hud__card') && vis('.card__title'),
      col: cl('is-col'), explore: cl('is-explore'), notesSession: e.notesOpen, notesAvailable: window.__explore.inspector.hasNotes,
      end: cl('is-end'), board: cl('has-board'),
      head: document.querySelector('.head__title')?.textContent ?? '', cardTitle: document.querySelector('.card__title')?.textContent ?? '',
      caption: (document.querySelector('.hud__caption p')?.textContent ?? '').length,
      teleRows: [...document.querySelectorAll('.hud__tele .row')].filter((r) => r.style.display !== 'none').length,
      mainRows: (c?.readouts?.(Math.max(0, e.T - e.starts[e.active])) ?? []).filter((r) => !r.detail).length,
      hots: [...document.querySelectorAll('.hot')].filter((b) => b.style.display !== 'none').length,
      controls: document.querySelectorAll('.xp__row').length,
      // the notes column: which page is on show, k of N, and which marker is lit
      note: (() => {
        const n = document.querySelector('.hud__note'), on = [...document.querySelectorAll('.hot.is-on')].filter((b) => b.style.display !== 'none');
        const [k, of] = (n?.querySelector('.note__count')?.textContent ?? '').split('/').map((x) => parseInt(x, 10));
        const first = [...document.querySelectorAll('.hot')].find((b) => b.style.display !== 'none' && Number(b.textContent) === 1);
        return { in: !!n && n.classList.contains('is-in') && vis('.hud__note'), kind: e.hud.notePageKey.split(':')[1] ?? '', hasDetails: window.__explore.inspector.hasDetails, k: k || 0, of: of || 0, lit: on.map((b) => b.dataset.id), first: first?.dataset.id ?? '', rows: n?.querySelectorAll('.note__rows .row').length ?? 0 };
      })(),
      rateShown: vis('.rate'), rate: e.rate,
      slide: e.slideInfo().index, slideCount: e.slideInfo().count,
      overflow: [...document.querySelectorAll('.hud__head, .hud__caption, .hud__tele, .hud__bar')].filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && (r.right > innerWidth + 1 || r.bottom > innerHeight + 1 || r.left < -1 || r.top < -1); }).map((el) => el.className),
    };
  });
  /** Wait until `pred(state)` holds, for at most `ms`. Returns the last state. */
  const until = async (pred, ms = 8000) => { let s = await state(); for (let t = 0; t < ms && !pred(s); t += 150) { await sleep(150); s = await state(); } return s; };
  const load = async (query) => {
    await page.goto(`${base}/?${only}${query}`, { waitUntil: 'load' });
    await page.waitForFunction("document.querySelector('.boot__go')?.classList.contains('is-armed') || window.__piramid?.frames > 30", { timeout: 180000 });
  };
  const ids = () => page.evaluate(() => window.__piramid.chapters.map((c) => c.id));
  /** Put the film a little before the end of chapter i's last slide. */
  const nearEnd = (i, back = 1.6) => page.evaluate((i, back) => { const e = window.__piramid; e.seek(e.starts[i] + e.chapters[i].duration - back); }, i, back);

  // ── the poster ──
  await load('');
  const order = await ids();
  const N = order.length;
  const toc = await page.evaluate(() => [...document.querySelectorAll('.toc__row')].filter((b) => !b.disabled).length);
  check('the poster enables one row per built chapter', toc === order.filter((id) => id !== 'prologue').length, `rows=${toc} chapters=${N}`);
  const logShown = await page.evaluate(() => { const d = document.querySelector('.boot__details'); return !!d && !d.hidden; });
  check('the poster shows no computation log when nothing failed', !logShown);
  // keys on the poster must not change the film behind it
  for (const k of ['e', 's', '3', 'ArrowRight', 'd']) await page.keyboard.press(k);
  await page.click('.boot__go');
  let s = await until((x) => !x.playing && x.caption > 10, 20000);
  check('keys pressed on the poster leave the film alone', !s.explore && s.slides && s.id === order[0], `explore=${s.explore} slides=${s.slides} id=${s.id}`);
  check('the main button opens the slides and stops at the end of the first slide', s.slides && !s.playing && s.id === order[0] && s.caption > 10, `slide=${s.slide} T=${s.T.toFixed(2)}`);
  check('nothing runs out of the frame', s.overflow.length === 0, s.overflow.join(','));
  check('Notes are closed by default in slides', !s.col);

  // ── film mode from the poster: the film waits at the end of a slide for its text ──
  await load('');
  await page.click('.boot__alt');
  s = await until((x) => x.playing && x.T > 0.3, 8000);
  check('the second button plays the same slides as a film', s.playing && !s.slides && s.id === order[0], `T=${s.T.toFixed(2)}`);
  const wait = await page.evaluate(async () => {
    const e = window.__piramid;
    // run up to the end of the first slide, then watch film time stand still while the clock runs
    const t0 = performance.now();
    let held = 0, last = e.T, moved = false;
    while (performance.now() - t0 < 30000) {
      await new Promise((r) => setTimeout(r, 100));
      if (!e.playing) break;
      if (Math.abs(e.T - last) < 1e-6) held += 0.1; else if (held > 0.8) { moved = true; break; } else held = 0;
      last = e.T;
    }
    return { held, moved, slide: e.slideInfo().index };
  });
  check('the film holds a slide for its reading time and then moves on by itself', wait.held > 0.8 && wait.moved, `held=${wait.held.toFixed(1)}s moved=${wait.moved}`);
  const clock = await page.evaluate(() => { const e = window.__piramid; return { total: e.clockTotal, raw: e.total / e.tempo }; });
  check('the film\'s length counts the reading waits', clock.total > clock.raw + 5, `${clock.total.toFixed(0)}s vs ${clock.raw.toFixed(0)}s`);

  // ── a card, the head, the measured rows ──
  await page.evaluate(() => window.__piramid.goto(1));
  s = await until((x) => x.card, 4000);
  const title1 = await page.evaluate(() => document.querySelector('.card__title')?.textContent ?? '');
  check('a chapter opens with its card', s.card && s.id === order[1] && title1.length > 2, `id=${s.id}`);
  s = await until((x) => !x.card && x.caption > 10, 20000);
  check('the card gives way to the running head and a slide', !s.card && s.head === title1 && s.caption > 10);
  // Sample the measurements once a second of chapter time.
  const rows = await page.evaluate(() => {
    const e = window.__piramid; let worst = { n: 0, id: '', t: 0 };
    for (const c of e.chapters) { if (c.board) continue; for (let t = 0; t < c.duration; t += 1) { const n = (c.readouts?.(t) ?? []).filter((r) => !r.detail).length; if (n > worst.n) worst = { n, id: c.id, t }; } }
    return worst;
  });
  // the numbers now live on a page of the notes; keep that page short enough to read
  check('no slide\'s numbers page lists more than six rows', rows.n <= 6, `${rows.n} rows in ${rows.id} at t=${rows.t}`);

  // ── one Notes column with manual pages and optional authored context ──
  const { notesHold, lastKind } = await page.evaluate(() => {
    const e = window.__piramid, ch = e.chapters[1];
    const slide = e.slideList().find((slide) => slide.i === 1 && slide.a >= e.starts[1] && (ch.inspect?.(slide.h - e.starts[1]) ?? []).length >= 2);
    if (!slide) throw new Error('the smoke fixture needs two authored markers on one held picture');
    const t = slide.h - e.starts[1]; e.seek(slide.h); e.playing = false;
    return { notesHold: slide.h, lastKind: ch.details?.(t) ? 'details' : ch.readouts?.(t)?.length ? 'numbers' : 'hot' };
  });
  await sleep(500);
  s = await state();
  check('the mandatory methodology controls are absent', await page.$('.hud__setup, .runbtn, .dbtn') === null);
  check('the film shows its speed button', s.rateShown);
  await page.click('.rate');
  s = await until((x) => x.rate !== 1, 2000);
  check('the speed button changes the speed', s.rate !== 1, `rate=${s.rate}`);
  await page.keyboard.press('>'); await page.keyboard.press('>'); await page.keyboard.press('>'); await page.keyboard.press('>');
  s = await state();
  check('the speed goes above normal', s.rate > 1, `rate=${s.rate}`);
  await page.evaluate(() => window.__piramid.setRate(1));
  await page.keyboard.press('e');
  s = await until((x) => x.explore && x.note.in, 4000);
  check('E opens the notes, pauses, and shows note 1 with its marker lit', s.explore && !s.playing && s.note.in && s.note.k === 1 && s.note.lit.length === 1 && s.note.lit[0] === s.note.first, JSON.stringify(s.note));
  check('the notes offer no parameter controls', s.controls === 0, `controls=${s.controls}`);
  const of = s.note.of;
  await page.click('.note__next');
  s = await until((x) => x.note.k === 2, 2000);
  check('› shows the next note and lights its marker', s.note.k === 2 && (s.note.kind !== 'hot' ? s.note.lit.length === 0 : s.note.lit.length === 1 && s.note.lit[0] !== s.note.first), JSON.stringify(s.note));
  await page.keyboard.press('[');
  s = await until((x) => x.note.k === 1, 2000);
  check('[ steps back', s.note.k === 1, JSON.stringify(s.note));
  for (let k = 1; k < of + 2; k++) await page.keyboard.press(']');
  s = await until((x) => x.note.k === of, 2000);
  check('manual next reaches the last authored page', s.note.k === of && s.note.kind === lastKind, JSON.stringify(s.note));
  await page.click('.hot');
  s = await until((x) => x.note.kind === 'hot' && x.note.lit.length === 1, 2000);
  check('a marker click brings back its note', s.note.in && s.note.kind === 'hot' && s.note.lit.length === 1, JSON.stringify(s.note));
  await page.keyboard.press('Escape');
  s = await until((x) => !x.explore, 3000);
  check('Esc closes the notes', !s.explore && !s.col);
  if (s.note.hasDetails) {
    await page.keyboard.press('d');
    s = await until((x) => x.explore && x.note.kind === 'details', 3000);
    check('D opens authored context without a field table', s.explore && s.note.kind === 'details' && s.note.rows === 0, JSON.stringify(s.note));
    await page.keyboard.press('d');
    s = await until((x) => !x.explore, 3000);
    check('D again closes Notes', !s.explore);
  } else {
    await page.keyboard.press('d'); await sleep(200); s = await state();
    check('D does nothing on a slide without authored details', !s.explore);
  }
  // Clicking a visible non-first marker on the held figure opens that marker's page.
  const beforePose = await page.evaluate((t) => {
    const e = window.__piramid; e.pause(); e.seek(t);
    return e.frames;
  }, notesHold);
  await page.waitForFunction((frame) => window.__piramid.frames >= frame + 3, {}, beforePose);
  const markerHandle = await page.evaluateHandle(() => [...document.querySelectorAll('.hud__hots.is-peek .hot')]
    .find((b) => b.getBoundingClientRect().width > 0 && Number(b.textContent) > 1));
  const marker = markerHandle.asElement();
  if (!marker) throw new Error('the smoke fixture needs a visible non-first note marker');
  const expected = await marker.evaluate((b) => ({ id: b.dataset.id, page: Number(b.textContent) }));
  await marker.click();
  await marker.dispose();
  const selected = (x) => x.explore && x.note.in && x.note.lit.length === 1 && x.note.lit[0] === expected.id && x.note.k === expected.page;
  s = await until(selected, 3000);
  check('a faint marker opens the notes on its note', selected(s), JSON.stringify({ expected, actual: s.note }));
  // in slides the next slide opens on its first note
  await page.evaluate(() => { const e = window.__piramid; if (!e.slides) e.setSlides(true); });
  // from the middle of a slide the arrow first plays that slide to its end; press until the next slide holds
  const slide0 = (await state()).slide;
  for (let k = 0; k < 3; k++) {
    await page.keyboard.press('ArrowRight');
    s = await until((x) => !x.playing && x.note.in, 12000);
    if (s.slide > slide0) break;
  }
  check('the next slide opens on its first note', s.explore && s.note.in && s.note.k === 1, JSON.stringify(s.note));
  await page.keyboard.press('Escape');
  await until((x) => !x.explore, 3000);
  s = await state();
  check('slides hide the speed button', !s.rateShown);
  await page.evaluate(() => { const e = window.__piramid; e.setSlides(false); });

  // ── a numeric chapter jump lands directly on its divider and E leaves it in place ──
  await page.evaluate(() => window.__piramid.setSlides(true));
  await page.keyboard.press('2');
  s = await until((x) => x.id === order[Math.min(2, N - 1)] && x.titleVisible, 4000);
  const cardTime = s.T;
  await page.keyboard.press('e');
  await sleep(250);
  s = await state();
  check('an opening card is a held slide and E leaves it visible when Notes are closed',
    s.titleVisible && s.card && !s.playing && !s.notesAvailable && !s.notesSession && !s.col && s.T === cardTime);
  // the viewer's pause wins over the notes' resume
  await page.evaluate(() => { const e = window.__piramid; e.setSlides(false); e.seek(e.starts[2] + 3); e.play(); });
  await sleep(300);
  await page.keyboard.press('e');
  await until((x) => x.explore, 3000);
  await page.keyboard.press(' '); await page.keyboard.press(' ');
  await page.keyboard.press('e');
  s = await until((x) => !x.explore, 3000);
  await sleep(300);
  check('a pause made while the notes were open stays after closing them', !s.explore && !(await state()).playing);
  // a slide with measured rows keeps them in the notes, not on the slide
  const numbersPage = await page.evaluate(() => {
    const e = window.__piramid;
    const slide = e.slideList().find((slide) => {
      const ch = e.chapters[slide.i];
      return !ch.board && slide.a >= e.starts[slide.i] && (ch.readouts?.(slide.h - e.starts[slide.i]) ?? []).length > 0;
    });
    if (!slide) return false;
    e.playing = false; e.seek(slide.h); return true;
  });
  await sleep(400);
  s = await state();
  check('no measured panel on a slide', s.teleRows === 0 || s.board, `rows=${s.teleRows}`);
  if (numbersPage) {
    await page.keyboard.press('e');
    await until((x) => x.explore && x.note.in, 3000);
    for (let k = 0; k < 24; k++) {
      if ((await state()).note.kind === 'numbers') break;
      await page.keyboard.press(']'); await sleep(30);
    }
    s = await state();
    check('the slide’s measurements remain an accessible Notes page', s.note.kind === 'numbers' && s.note.rows >= 1, JSON.stringify(s.note));
    await page.keyboard.press('Escape');
    await until((x) => !x.explore, 3000);
  }

  // ── every chapter boundary, in Film and in Slides, with and without a Notes session ──
  const atDivider = (id) => (x) => x.id === id && x.card && x.titleVisible;
  const firstFigure = (id) => (x) => x.id === id && !x.card && x.caption > 10;
  const openNotesSession = async () => {
    await page.evaluate((h) => { const e = window.__piramid; e.pause(); e.seek(h); }, notesHold);
    await until((x) => x.id === order[1] && x.T === notesHold && x.notesAvailable);
    await page.keyboard.press('e');
    await until((x) => x.notesSession && x.note.in);
  };
  const atLastSlide = (i) => page.evaluate((i) => {
    const e = window.__piramid, last = e.slideList().findLast((s) => s.i === i);
    e.pause(); e.seek(last.h);
  }, i);
  for (let i = 0; i + 1 < N; i++) {
    const a = order[i], b = order[i + 1], tag = `${a} → ${b}`;
    // film, notes closed
    await page.evaluate(() => { const e = window.__piramid; e.pause(); e.setSlides(false); });
    await nearEnd(i, 0.6);
    await page.evaluate(() => window.__piramid.play());
    const filmDivider = await until(atDivider(b));
    s = await until(firstFigure(b), 15000);
    check(`film: ${tag}`, atDivider(b)(filmDivider) && firstFigure(b)(s) && s.playing, `id=${s.id} card=${filmDivider.titleVisible}`);

    // film, notes open: jump forward, then let the title card run into the first figure
    await openNotesSession();
    await page.evaluate((i) => { const e = window.__piramid; e.seek(e.starts[i] + e.chapters[i].duration * 0.5); }, i);
    await until((x) => x.id === a);
    await page.keyboard.press('ArrowRight');
    const filmWithNotes = await until(atDivider(b));
    await page.click('.play');
    s = await until(firstFigure(b), 15000);
    check(`film, notes open: ${tag}`, atDivider(b)(filmWithNotes) && filmWithNotes.notesSession && !filmWithNotes.col
      && firstFigure(b)(s) && s.notesSession && s.playing && s.col === s.notesAvailable, `id=${s.id} card=${filmWithNotes.titleVisible} notes=${s.notesSession}`);
    await page.keyboard.press('Escape');
    await until((x) => !x.notesSession, 3000);

    // slides: one step onto the card and one onto the first figure. Back visits the same card.
    await page.evaluate(() => window.__piramid.setSlides(true));
    for (const withNotes of [false, true]) {
      if (withNotes) await openNotesSession();
      await atLastSlide(i);
      const lastSlide = await until((x) => x.id === a && !x.playing);
      if (i % 2) await page.click('.play');
      else await page.keyboard.press('ArrowRight');
      const forward = await until(atDivider(b));
      await page.keyboard.press('ArrowRight');
      const figure = await until((x) => firstFigure(b)(x) && !x.playing && (!withNotes || !x.notesAvailable || x.note.in), 20000);
      if (i % 2) await page.click('.prev');
      else await page.keyboard.press('ArrowLeft');
      const back = await until(atDivider(b));
      await page.keyboard.press('ArrowLeft');
      const previous = await until((x) => x.id === a && x.slide === lastSlide.slide);
      const onCard = (x) => atDivider(b)(x) && !x.playing && x.notesSession === withNotes && !x.col && !x.notesAvailable;
      check(`slides${withNotes ? ', notes open' : ''}, both directions: ${tag}`,
        onCard(forward) && onCard(back) && forward.T === back.T && forward.slide === lastSlide.slide + 1
        && firstFigure(b)(figure) && figure.slide === forward.slide + 1 && figure.notesSession === withNotes
        && (!withNotes || !figure.notesAvailable || (figure.col && figure.note.k === 1))
        && previous.id === a && previous.T === lastSlide.T,
        `last=${lastSlide.slide} title=${forward.slide} figure=${figure.slide} back=${back.slide} prev=${previous.slide}`);
      if (withNotes) {
        await page.keyboard.press('Escape');
        await until((x) => !x.notesSession, 3000);
      }
    }
  }

  // ── stepping back, the board, the end card ──
  await page.evaluate(() => { const e = window.__piramid; if (!e.slides) e.setSlides(true); e.goto(1); });
  await sleep(300);
  await page.keyboard.press('ArrowRight');
  s = await until((x) => !x.playing && !x.card, 12000);
  const before = s.slide;
  await page.keyboard.press('ArrowLeft');
  s = await until((x) => x.slide < before, 3000);
  check('the left arrow steps one slide back', s.slide === before - 1 && !s.playing, `slide ${before} → ${s.slide}`);

  if (order.includes('finale')) {
    const fi = order.indexOf('finale');
    await page.evaluate((fi) => { const e = window.__piramid; if (e.slides) e.setSlides(false); e.seek(e.starts[fi] + e.chapters[fi].duration * 0.6); e.play(); }, fi);
    s = await until((x) => x.board && x.teleRows >= 1, 6000);
    check('the closing chapter shows its board', s.board && s.teleRows >= 1, `rows=${s.teleRows}`);
    // the board is staged: each slide shows only the rows of the question on screen, never the whole map
    check('the board shows only the rows of the slide on screen', s.teleRows >= 1 && s.teleRows <= 4, `rows=${s.teleRows}`);
  }
  await page.evaluate(() => { const e = window.__piramid; if (e.slides) e.setSlides(false); e.seek(e.total - 0.6); e.play(); });
  s = await until((x) => x.ended && x.end, 8000);
  check('the film stops on the end card', s.ended && !s.playing && s.end);
  check('no markers over the end card', s.hots === 0, `markers=${s.hots}`);
  await page.keyboard.press('e');
  await sleep(500);
  s = await state();
  check('the end card has no notes and stays up', s.end && !s.explore, `end=${s.end} explore=${s.explore}`);
  const sources = await page.evaluate(() => document.querySelectorAll('.end__sources .row a').length);
  if (order.includes('finale')) check('the end card lists the sources with their links', sources >= 8, `links=${sources}`);
  await page.click('.end__slides');
  s = await until((x) => !x.end && x.T < 3, 4000);
  check('"Read again" starts the slides over', s.slides && !s.end && s.T < 3 && s.id === order[0], `T=${s.T.toFixed(2)}`);

  // ── a deep link into a chapter skips its card ──
  const mid = order[Math.min(2, N - 1)];
  await page.goto(`${base}/?only=${mid}&ch=${mid}&t=3&autostart=1`, { waitUntil: 'load' });
  await page.waitForFunction('window.__piramid?.chapters?.length === 1 && window.__piramid.frames > 30', { timeout: 120000 });
  s = await state();
  check('a deep link opens the chapter at its moment', !s.card && s.id === mid && s.local >= 3, `local=${s.local.toFixed(1)}`);

  check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  failed++;
  console.log('FAIL smoke test threw:', e.message);
} finally {
  await browser.close();
}
console.log(failed ? `${failed} check(s) failed` : 'all checks passed');
process.exit(failed ? 1 : 0);
