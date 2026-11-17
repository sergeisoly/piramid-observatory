// Re-runs one chapter on other seeds in the real page and prints its results, to see how far a number
// that the film shows for one run holds on others. Nothing here is shown in the film.
//   node tools/seeds.mjs --url=http://127.0.0.1:5199 --ch=percolation --keys=seed,seedE,seedV --seeds=1,2,3
//   node tools/seeds.mjs --url=… --ch=entk --keys=seed --seeds=1,2,3 --beat=7     (also the measured rows of that slide)
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';

const args = readFlags();
const base = args.url ?? 'http://127.0.0.1:5199', ch = args.ch, keys = String(args.keys ?? 'seed').split(','), seeds = String(args.seeds ?? '1,2,3').split(',');
const browser = await launchChrome({ chrome: args.chrome });
try {
  for (const s of seeds) {
    const page = await browser.newPage();
    const q = keys.map((k) => `p.${k}=${s}`).join('&');
    await page.goto(`${base}/?shot=1&only=${ch}&ch=${ch}&t=0&${q}`, { waitUntil: 'load' });
    await page.waitForFunction('window.__shotReady === true', { timeout: 240000 });
    const out = await page.evaluate((beat) => {
      const c = window.__piramid.chapters[0];
      const plain = (h) => { const d = document.createElement('div'); d.innerHTML = h; return d.textContent.trim(); };
      const res = (c.results ?? []).map((r) => `${plain(r.label)}: ${plain(r.measured)}`);
      const rows = beat ? (c.readouts?.(c.beats[beat - 1].at + (c.beats[beat] ? (c.beats[beat].at - c.beats[beat - 1].at) * 0.9 : 1)) ?? []).map((r) => `${plain(r.label)}: ${plain(r.value)}`) : [];
      const cap = beat ? plain(c.beats[beat - 1].caption).slice(0, 160) : '';
      return { res, rows, cap };
    }, +(args.beat ?? 0));
    console.log(`seed ${s}\n  ` + [...out.res, ...out.rows.map((r) => 'slide: ' + r), out.cap ? 'text: ' + out.cap : ''].filter(Boolean).join('\n  '));
    await page.close();
  }
} finally {
  await browser.close();
}
