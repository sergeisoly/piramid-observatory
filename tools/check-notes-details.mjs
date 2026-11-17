// Verify the authored details contract on every content slide, plus end-card sources and restore IDs.
// node tools/check-notes-details.mjs --url=http://127.0.0.1:5199
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { launchChrome } from './browser.mjs';
const { values } = parseArgs({ options: {
  url: { type: 'string', default: 'http://127.0.0.1:5199' },
  chrome: { type: 'string' },
} });
const url = new URL(values.url); url.search = 'shot=1&slides=1';
const browser = await launchChrome({ chrome: values.chrome, width: 1200, height: 675, windowArgs: [] });
const errors = [];
try {
  const page = await browser.newPage(); page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(url.href, { waitUntil: 'load' });
  await page.waitForFunction('window.__shotReady === true', { timeout: 240000 });
  const frames = () => page.evaluate(() => new Promise((resolve) => {
    const e = window.__piramid, start = e.frames;
    const wait = () => e.frames - start >= 3 ? resolve() : requestAnimationFrame(wait); wait();
  }));
  const slides = await page.evaluate(() => {
    const e = window.__piramid;
    return e.slideList().filter((slide) => slide.a >= e.starts[slide.i] && slide.a < e.total - 0.9).map((slide) => ({ ...slide, chapter: e.chapters[slide.i].id }));
  });
  let present = 0, absent = 0, restored = 0;
  const chapters = new Set();
  for (const slide of slides) {
    await page.evaluate((hold) => {
      const e = window.__piramid; window.__explore.toggle(false); e.seek(hold); e.playing = false;
    }, slide.h);
    await frames();
    const expected = await page.evaluate(() => {
      const e = window.__piramid, ch = e.chapters[e.active], t = e.T - e.starts[e.active];
      const details = ch.details?.(t), hot = ch.inspect?.(t) ?? [], numbers = ch.readouts?.(t) ?? [];
      return { details, count: hot.length + (numbers.length ? 1 : 0) + (details ? 1 : 0) };
    });
    await page.keyboard.press('d'); await frames();
    const actual = await page.evaluate(() => {
      const x = window.__explore, note = document.querySelector('.hud__note');
      return { open: x.open, page: x.inspector.selectedPage, title: note.querySelector('.note__title').textContent,
        text: note.querySelector('.note__body').textContent, rows: note.querySelector('.note__rows').childElementCount,
        count: Number(note.querySelector('.note__count').textContent.split('/')[1]) };
    });
    if (expected.details) {
      present++; chapters.add(slide.chapter);
      assert.equal(actual.open, true, `${slide.chapter}: D opens authored context`);
      assert.equal(actual.page, 'details');
      assert.equal(actual.title, expected.details.title);
      assert.ok(actual.text.trim().length > 0, 'the author supplied readable prose');
      assert.equal(actual.rows, 0, 'details impose no table');
      assert.equal(actual.count, expected.count, 'only authored pages and measurements appear');
      assert.ok(!/^(How this was run|Experiment details)$/i.test(actual.title), 'details use a contextual title');
      // The dev-reload integration uses these two public methods after posing the restored slide.
      const id = await page.evaluate(() => window.__explore.inspector.selectedPage);
      await page.keyboard.press('e'); await frames();
      await page.keyboard.press('e'); await frames();
      await page.evaluate((id) => window.__explore.inspector.selectPage(id), id); await frames();
      assert.equal(await page.evaluate(() => window.__explore.inspector.selectedPage), id);
      restored++;
    } else {
      absent++;
      assert.equal(actual.open, false, `${slide.chapter}: D leaves Notes closed when there are no authored details`);
    }
  }
  assert.ok(present > 0 && absent > 0, 'the film exercises both branches of the optional contract');
  const sources = await page.evaluate(() => {
    const e = window.__piramid;
    return e.chapters.flatMap((ch) => ch.endSources ?? []);
  });
  assert.ok(sources.length > 0, 'end-card sources have independent metadata');
  await page.evaluate(() => {
    const e = window.__piramid; window.__explore.toggle(false); e.seek(e.total - 0.01); e.playing = false;
  });
  await frames();
  const end = await page.evaluate(() => ({
    shown: document.getElementById('hud').classList.contains('is-end'),
    rows: document.querySelectorAll('.end__sources .row').length,
    links: [...document.querySelectorAll('.end__sources a')].map((a) => a.href),
    legacy: document.querySelector('.hud__setup, .runbtn, .dbtn') !== null,
  }));
  assert.equal(end.shown, true); assert.equal(end.rows, sources.length); assert.equal(end.legacy, false);
  for (const source of sources) {
    const expected = source.url ?? (source.value.match(/arXiv:(\d+\.\d+)/)?.[1] ? `https://arxiv.org/abs/${source.value.match(/arXiv:(\d+\.\d+)/)[1]}` : null);
    if (expected) assert.ok(end.links.includes(expected), `source link preserved: ${expected}`);
  }
  assert.deepEqual(errors, [], 'no uncaught page errors');
  console.log(`Details: ${slides.length} slides across all chapters; ${present} authored pages in ${chapters.size} chapters, ${absent} absent, ${restored} restore IDs, ${sources.length} end-card sources passed`);
} catch (error) { console.error(error.stack ?? error); process.exitCode = 1; }
finally { await browser.close(); }
