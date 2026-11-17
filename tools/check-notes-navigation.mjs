// Real-browser regression for manual Notes navigation, stable slide stacks and optional details.
// node tools/check-notes-navigation.mjs --url=http://127.0.0.1:5199
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { launchChrome } from './browser.mjs';

const { values } = parseArgs({ options: {
  url: { type: 'string', default: 'http://127.0.0.1:5199' },
  chrome: { type: 'string' },
} });
const url = new URL(values.url);
url.search = new URLSearchParams({ only: 'prologue,meanfield', shot: '1', slides: '1', ch: 'prologue', t: '0', explore: '1' }).toString();
const browser = await launchChrome({ chrome: values.chrome, width: 1200, height: 675, windowArgs: [] });
const errors = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  const load = async (target) => {
    await page.goto(target, { waitUntil: 'load' });
    await page.waitForFunction('window.__shotReady === true', { timeout: 180000 });
  };
  await load(url.href);
  const frames = async (count = 3) => page.evaluate((count) => new Promise((resolve) => {
    const engine = window.__piramid, start = engine.frames;
    const wait = () => engine.frames - start >= count ? resolve() : requestAnimationFrame(wait);
    wait();
  }), count);
  const pageId = () => page.evaluate(() => {
    const note = document.querySelector('.hud__note');
    return note.classList.contains('is-in') ? window.__piramid.hud.notePageKey.split(':').slice(1).join(':') : null;
  });
  await frames();
  // Chapter content supplies the expected endpoint pages; assertions do not depend on prose or
  // how many notes an editor keeps. Navigation still uses the real keyboard and rendered HUD.
  const slides = await page.evaluate(() => {
    const e = window.__piramid;
    return e.slideList().map((slide, index) => {
      const ch = e.chapters[slide.i], t = slide.h - e.starts[slide.i];
      const hot = ch.inspect?.(t)?.[0];
      const first = hot ? `hot:${hot.id}` : ch.readouts?.(t)?.length ? 'numbers' : ch.details?.(t) ? 'details' : null;
      return { ...slide, index, chapter: ch.id, first };
    }).filter((slide) => slide.chapter === 'prologue' && slide.a >= e.starts[slide.i]);
  });
  await page.evaluate(() => {
    const e = window.__piramid, show = e.hud.showNote.bind(e.hud);
    window.__noteFrames = [];
    e.hud.showNote = (content, ...args) => {
      window.__noteFrames.push({ slide: e.slideInfo().index, page: args[1].split(':').slice(1).join(':') });
      return show(content, ...args);
    };
  });
  assert.equal(await pageId(), slides[0].first);
  for (const slide of slides.slice(1)) {
    await page.evaluate(() => { const e = window.__piramid; e.seek(e.slideInfo().hold); e.playing = false; });
    await frames();
    await page.evaluate(() => { window.__noteFrames = []; });
    await page.keyboard.press('ArrowRight');
    await frames(5);
    assert.equal(await pageId(), slide.first, `slide ${slide.index} starts at its first page`);
    await page.evaluate(() => { const e = window.__piramid; e.seek(e.slideInfo().hold); e.playing = false; });
    await frames();
    const seen = await page.evaluate((index) => window.__noteFrames.filter((frame) => frame.slide === index).map((frame) => frame.page), slide.index);
    assert.ok(slide.first === null ? seen.length === 0 : seen.length > 0 && seen.every((entry) => entry === slide.first), `a transition must not flash another page: ${JSON.stringify(seen)}`);
  }

  // Add one optional authored page through the chapter contract. Legacy metadata is intentionally
  // present too: it must neither append another page nor create a mandatory methodology table.
  const target = slides.find((slide) => slide.first?.startsWith('hot:') && slide.index < slides.at(-1).index);
  assert.ok(target, 'a numbered note is available for the navigation fixture');
  const fixture = await page.evaluate((target) => {
    const e = window.__piramid, chapter = e.chapters[target.i];
    window.__originalChapter = chapter;
    e.chapters[target.i] = { ...chapter,
      setup: [{ label: 'Legacy metadata', value: 'Never shown automatically' }],
      details: (t) => t >= target.a - e.starts[target.i] && t <= target.h - e.starts[target.i]
        ? { title: 'A comparison worth explaining', body: '<p>The reference comparison changes what this result supports.</p>' } : undefined,
    };
    e.seek(target.h); e.playing = false;
    return { count: (chapter.inspect?.(target.h - e.starts[target.i]) ?? []).length + (chapter.readouts?.(target.h - e.starts[target.i])?.length ? 1 : 0) + 1 };
  }, target);
  await frames();
  assert.equal(await pageId(), target.first);
  assert.equal(await page.$eval('.note__count', (el) => Number(el.textContent.split('/')[1])), fixture.count);
  for (let k = 1; k < fixture.count; k++) await page.keyboard.press(']');
  await frames();
  assert.equal(await pageId(), 'details', 'manual next reaches optional authored details');
  assert.equal(await page.$eval('.note__title', (el) => el.textContent), 'A comparison worth explaining');
  assert.equal(await page.$eval('.note__rows', (el) => el.childElementCount), 0, 'prose has no imposed field schema');
  await page.evaluate(() => { const e = window.__piramid; e.seek(e.slideInfo().start + 0.05); e.playing = true; });
  await frames(8);
  assert.equal(await pageId(), 'details', 'animation preserves the manually selected page');
  await page.keyboard.press('[');
  await frames();
  assert.notEqual(await pageId(), 'details', 'manual previous leaves authored details');
  await page.keyboard.press('d');
  await frames();
  assert.equal(await pageId(), 'details', 'D explicitly selects relevant authored details');
  await page.keyboard.press('d');
  await frames();
  assert.equal(await page.evaluate(() => window.__explore.open), false, 'D again closes Notes');
  await page.keyboard.press('d');
  await frames();
  assert.equal(await pageId(), 'details');
  await page.evaluate(() => { const e = window.__piramid; e.seek(e.slideInfo().hold); e.playing = false; });
  await frames();
  await page.keyboard.press('ArrowRight');
  await frames(5);
  const next = slides.find((slide) => slide.index === target.index + 1);
  assert.equal(await pageId(), next.first, 'forward navigation starts at the new slide’s first page');
  await page.keyboard.press('d');
  await frames();
  assert.equal(await pageId(), next.first, 'D does nothing when the slide has no authored details');
  await page.keyboard.press('ArrowLeft');
  await frames(5);
  assert.equal(await pageId(), target.first, 'backward navigation starts at the previous slide’s first page');

  // Empty Notes stay empty, even with legacy metadata. D must not open an empty column.
  await page.evaluate(() => {
    const e = window.__piramid, chapter = e.chapters[e.active];
    e.chapters[e.active] = { ...chapter, inspect: () => [], readouts: () => [], details: undefined };
  });
  await frames();
  assert.equal(await pageId(), null);
  await page.keyboard.press('e');
  await frames();
  await page.keyboard.press('d');
  await frames();
  assert.equal(await page.evaluate(() => window.__explore.open), false);
  assert.equal(await page.$('.hud__setup, .runbtn, .dbtn'), null, 'the mandatory setup UI has been removed');

  // A real deep link without authored details must not open Notes. A real authored page, when
  // the selected chapters have one, must be directly reachable without a generic wrapper.
  await page.evaluate(() => { const e = window.__piramid; e.chapters[0] = window.__originalChapter; });
  const links = await page.evaluate(() => {
    const e = window.__piramid, links = {};
    for (const slide of e.slideList()) {
      const chapter = e.chapters[slide.i]; if (slide.a < e.starts[slide.i]) continue;
      const t = slide.h - e.starts[slide.i], details = chapter.details?.(t);
      if (!links[details ? 'present' : 'absent']) links[details ? 'present' : 'absent'] = { chapter: chapter.id, t, title: details?.title };
    }
    return links;
  });
  for (const [kind, target] of Object.entries(links)) {
    const link = new URL(url); link.searchParams.delete('explore'); link.searchParams.set('details', '1');
    link.searchParams.set('ch', target.chapter); link.searchParams.set('t', String(target.t));
    await load(link.href); await frames(8);
    if (kind === 'present') {
      assert.equal(await pageId(), 'details');
      assert.equal(await page.$eval('.note__title', (el) => el.textContent), target.title);
    } else assert.equal(await page.evaluate(() => window.__explore.open), false);
  }
  assert.deepEqual(errors, [], 'no uncaught browser errors');
  console.log(`Notes navigation: ${slides.length} prologue slides, transient frames, optional details, manual pages, forward/back, empty Notes and deep links passed`);
} catch (error) {
  console.error(error.stack ?? error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
