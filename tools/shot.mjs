// Headless frame grabber: renders exact moments of a chapter with the real WebGPU pipeline.
//   npm run shot -- --ch=prologue --t=3,12,24            → shots/prologue-003.0.jpg …
//   options: --w=1600 --h=900 --out=shots --full=1 (build every chapter) --headful=1
//            --p=beta:400,N:128   override chapter controls (Explore-mode parameters)
//            --explore=1          capture with the Explore panel open
//            --hot=id             Explore, with that hotspot's explanation open
//            --hover=x,y          Explore, probing the figure at frame pixel (x, y) of the 1600 × 900 frame
//            --theme=dark         light-on-dark variant (default is ink on paper)
//            --slides=1           slides mode;  --details=1  open authored details when the slide has them
//            --poster=1           the opening screen (--card=1: without its controls);  --t=-1.5  a chapter's opening card
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const args = readFlags();
const ch = args.ch ?? 'prologue';
const times = String(args.t ?? '5').split(',').map(Number);
const out = resolve(args.out ?? 'shots');
const W = +(args.w ?? 1600), H = +(args.h ?? 900);
const base = args.url ?? 'http://127.0.0.1:5199';

const browser = await launchChrome({ chrome: args.chrome, headful: args.headful, width: W, height: H });
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warn' || m.type() === 'warning') console.log(`[console.${m.type()}]`, m.text().slice(0, 600));
  });
  page.on('pageerror', (e) => { failed = true; console.log('[pageerror]', e.message); });
  if (args.poster) {
    // the opening screen, once every figure is computed (--poster=1; with --full=0 only chapter --ch is built)
    await page.goto(`${base}/?${args.full === '0' ? `only=${ch}` : ''}${args.theme ? `&theme=${args.theme}` : ''}${args.card ? '&card=1' : ''}`, { waitUntil: 'load' });
    await page.waitForFunction("document.querySelector('.boot__go')?.classList.contains('is-armed')", { timeout: 120000 });
    await new Promise((r) => setTimeout(r, 700));
    mkdirSync(out, { recursive: true });
    const file = resolve(out, `poster${args.tag ? `-${args.tag}` : ''}.jpg`);
    await page.screenshot({ path: file, type: 'jpeg', quality: 90 });
    console.log('saved', file);
    await browser.close();
    process.exit(0);
  }
  const only = args.full ? '' : `&only=${ch}`;
  const pq = args.p ? String(args.p).split(',').map((kv) => { const [k, ...v] = kv.split(':'); return `&p.${encodeURIComponent(k)}=${encodeURIComponent(v.join(':'))}`; }).join('') : '';
  const insp = (args.hot ? `&hot=${encodeURIComponent(args.hot)}` : '') + (args.hover ? `&hover=${args.hover}` : '');
  const ex = (args.explore || insp ? '&explore=1' : '') + insp + (args.theme ? `&theme=${args.theme}` : '') + (args.slides ? '&slides=1' : '') + (args.details ? `&details=${args.details}` : '');
  await page.goto(`${base}/?shot=1${only}&ch=${ch}&t=${times[0]}${pq}${ex}`, { waitUntil: 'load' });
  await page.waitForFunction('window.__shotReady === true', { timeout: 90000 });
  const info = await page.evaluate(() => {
    const e = window.__piramid;
    return { backend: e.backend, chapters: e.chapters.map((c) => `${c.id}:${c.duration}s`), boot: document.querySelector('.boot__log')?.innerText.replace(/\n/g, ' | ') };
  });
  console.log('backend:', info.backend, '| chapters:', info.chapters.join(' '));
  console.log('boot:', info.boot);
  mkdirSync(out, { recursive: true });
  for (const t of times) {
    await page.evaluate((id, t) => {
      const e = window.__piramid;
      const i = e.indexOf(id);
      e.seek(e.starts[i] + t);
    }, ch, t);
    await page.evaluate(() => new Promise((r) => {
      const e = window.__piramid, f0 = e.frames;
      const w = () => (e.frames - f0 > 10 ? r() : requestAnimationFrame(w));
      w();
    }));
    await new Promise((r) => setTimeout(r, 1100)); // let the caption animation finish
    const tag = args.tag ? `-${args.tag}` : '';
    const file = resolve(out, `${ch}${tag}-${t.toFixed(1).padStart(5, '0')}.jpg`);
    await page.screenshot({ path: file, type: 'jpeg', quality: 88 });
    const fps = await page.evaluate(() => Math.round(window.__piramid.fps ?? 0));
    console.log('saved', file, fps ? `(${fps} fps)` : '');
  }
} catch (e) {
  failed = true;
  console.log('[shot failed]', e.message);
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
