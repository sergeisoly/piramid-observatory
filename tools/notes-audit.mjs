// Check whether figure labels or markers intersect the Notes column at each held slide.
// Samples the page that opens on each slide and saves frames where the overlap count increases.
//   node tools/notes-audit.mjs http://127.0.0.1:5199 ../review/notes-audit
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { launchChrome } from './browser.mjs';

const [base, out] = process.argv.slice(2);
if (!base || !out) throw new Error('Usage: node tools/notes-audit.mjs <url> <output-directory>');
const url = new URL(base);
url.search = 'shot=1&explore=1';
const browser = await launchChrome();
try {
  const page = await browser.newPage();
  await page.goto(url.href, { waitUntil: 'load' });
  await page.waitForFunction('window.__shotReady === true', { timeout: 180000 });
  if (await page.$('.boot__bar .is-err')) throw new Error('Some chapters failed to build');

  const slides = await page.evaluate(() => {
    const engine = window.__piramid;
    return engine.chapters.flatMap((chapter, i) => chapter.beats.map((beat, j) => {
      engine.seek(engine.starts[i] + beat.at);
      return { id: chapter.id, j, hold: engine.slideInfo().hold };
    }));
  });
  const worst = new Map();
  for (const slide of slides) {
    await page.evaluate((hold) => window.__piramid.seek(hold), slide.hold);
    await sleep(1600);
    const m = await page.evaluate(() => {
      const engine = window.__piramid;
      const col = engine.hud.notesColumnVisible ? document.querySelector('.hud__side').getBoundingClientRect() : null;
      const labels = [...document.querySelectorAll('.hud__labels .lbl, .hud__hots .hot')].flatMap((el) => {
        const style = getComputedStyle(el);
        if (style.display === 'none' || style.visibility !== 'visible' || Number(style.opacity) <= 0.3) return [];
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 ? [rect] : [];
      });
      const over = col ? labels.filter((r) => r.right > col.left + 2 && r.left < col.right && r.bottom > col.top && r.top < col.bottom).length : 0;
      const maxRight = labels.reduce((right, rect) => Math.max(right, rect.right), 0);
      const hots = [...document.querySelectorAll('.hud__hots .hot')].filter((button) => button.style.display !== 'none').length;
      return { colLeft: Math.round(col?.left ?? innerWidth), over, maxRight: Math.round(maxRight), hots,
        page: col ? window.__explore.inspector.selectedPage : null };
    });
    const label = `${slide.id} b${slide.j + 1}`;
    console.log(`${label}  markers=${m.hots} labelsUnderColumn=${m.over} maxRight=${m.maxRight} colLeft=${m.colLeft} page=${m.page ?? '-'}`);
    if (m.over > (worst.get(slide.id) ?? 0)) {
      worst.set(slide.id, m.over);
      mkdirSync(out, { recursive: true });
      await page.screenshot({ path: join(out, `notes-${slide.id}-b${slide.j + 1}.jpg`), type: 'jpeg', quality: 85 });
    }
  }
} finally {
  await browser.close();
}
