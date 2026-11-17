// Focused Notes scroll regression. Uses the real chapter navigation first, then exercises
// same-page HTML refreshes independently of scientific/editorial wording.
//   node tools/check-notes-scroll.mjs --url=http://127.0.0.1:5199
//   --chrome=/path/to/chrome overrides the browser executable. No files are written.
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { launchChrome } from './browser.mjs';

const { values } = parseArgs({ options: {
  url: { type: 'string', default: 'http://127.0.0.1:5199' },
  chrome: { type: 'string' },
} });
const url = new URL(values.url.endsWith('/') ? values.url : `${values.url}/`);
url.search = new URLSearchParams({ only: 'percolation,meanfield', shot: '1', slides: '1', ch: 'percolation', t: '55' }).toString();
const browser = await launchChrome({ chrome: values.chrome, width: 1200, height: 675, windowArgs: ['--hide-scrollbars'] });
const errors = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(url.href, { waitUntil: 'load' });
  await page.waitForFunction('window.__shotReady === true', { timeout: 180000 });
  // Long authored prose makes this independent of chapter wording and the presence of details.
  await page.evaluate(() => {
    const e = window.__piramid;
    for (const id of ['percolation', 'meanfield']) {
      const i = e.indexOf(id), chapter = e.chapters[i];
      e.chapters[i] = { ...chapter, details: () => ({
        title: `Reading context for ${id}`,
        body: `<p>${Array(160).fill('A useful comparison explains the result and its limitations.').join(' ')}</p>`,
      }) };
    }
  });
  const waitFrames = () => page.evaluate(() => new Promise((resolve) => {
    const e = window.__piramid, start = e.frames;
    const wait = () => e.frames - start >= 5 ? resolve() : requestAnimationFrame(wait);
    wait();
  }));
  const openDetails = async (chapter) => {
    await page.waitForFunction((chapter) => window.__piramid.chapters[window.__piramid.active]?.id === chapter, {}, chapter);
    await waitFrames();
    if (!await page.evaluate(() => window.__explore.inspector.onDetails)) await page.keyboard.press('d');
    await page.waitForFunction(() => window.__explore.inspector.onDetails && document.querySelector('.hud__note')?.classList.contains('is-in'));
  };
  await openDetails('percolation');
  await page.hover('.hud__note .note__body');
  await page.mouse.wheel({ deltaY: 1800 });
  await page.waitForFunction(() => document.querySelector('.note__body').scrollTop > 0);
  const before = await page.$eval('.note__body', (body) => body.scrollTop);
  const nextChapter = await page.evaluate(() => window.__piramid.indexOf('meanfield'));
  assert.ok(nextChapter >= 0 && nextChapter < 10, 'the target chapter must have a digit shortcut');
  await page.keyboard.press(String(nextChapter));
  // The chapter shortcut lands on its title. The reading session resumes on the first figure.
  await page.waitForFunction((i) => {
    const e = window.__piramid;
    return e.active === i && e.hud.root.classList.contains('is-card') && !e.hud.notesColumnVisible;
  }, {}, nextChapter);
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(() => window.__piramid.hud.notesColumnVisible);
  await openDetails('meanfield');
  await waitFrames();
  const after = await page.$eval('.note__body', (body) => body.scrollTop);
  console.log(`chapter navigation: scroll ${before} → ${after}`);
  assert.ok(before > 0, 'the first chapter must actually be scrolled');
  assert.equal(after, 0, 'a different authored details page starts at the beginning');

  // Controlled data makes both containers scrollable without depending on chapter prose.
  // This synchronous call sequence keeps the live Inspector from posing another page between calls.
  const refresh = await page.evaluate(() => {
    const hud = window.__piramid.hud;
    const rows = Array.from({ length: 24 }, (_, i) => ({ label: `Row ${i}`, value: String(i), note: 'An explanatory row long enough to make the content area scroll.' }));
    const body = `<p>${Array(100).fill('A long explanatory paragraph.').join(' ')}</p>`;
    hud.showNote({ title: 'Runtime regression', body, rows }, 1, 'scroll-test:one', 1, 2);
    const rowBox = document.querySelector('.note__rows'), bodyBox = document.querySelector('.note__body');
    rowBox.style.overflow = 'auto';
    rowBox.style.maxHeight = '180px';
    bodyBox.style.height = '100px';
    bodyBox.style.maxHeight = '100px';
    bodyBox.style.flex = 'none';
    rowBox.scrollTop = 120;
    bodyBox.scrollTop = 80;
    const offsets = () => ({ rows: rowBox.scrollTop, body: bodyBox.scrollTop });
    const before = offsets();
    hud.showNote({ title: 'Runtime regression', body: body.replace('paragraph', 'updated paragraph'), rows: rows.map((row) => ({ ...row, value: `updated ${row.value}`, note: `${row.note} Updated.` })) }, 2, 'scroll-test:one', 2, 4);
    const samePage = offsets();
    hud.showNote({ title: 'Runtime regression', body, rows }, 1, 'scroll-test:two', 1, 2);
    return { before, samePage, newPage: offsets() };
  });
  assert.ok(refresh.before.rows > 0 && refresh.before.body > 0, 'both controlled containers must actually scroll');
  assert.deepEqual(refresh.samePage, refresh.before, 'updated text, values and marker counts must preserve same-page scroll');
  assert.deepEqual(refresh.newPage, { rows: 0, body: 0 }, 'a different page must reset both containers');
  assert.deepEqual(errors, [], 'the browser must report no uncaught page errors');
  console.log(`same-page refresh: ${JSON.stringify(refresh)}`);
  console.log('all Notes scroll checks passed');
} catch (error) {
  console.error(error.stack ?? error);
  process.exitCode = 1;
} finally {
  await browser.close();
}
