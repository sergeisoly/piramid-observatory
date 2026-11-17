// Measures where each chapter's figure sits in the 1600 × 900 frame, over time.
//
// The engine re-frames every figure into the free area of the layout (see src/engine/framing.ts).
// For that it needs the figure's bounding box in the frame the scenes were composed for. This tool
// renders every chapter with the HUD hidden (labels stay: they are part of the figure), finds the
// box of everything that is not blank paper, includes the rendered bounds of labels and markers,
// and writes one box per caption beat.
//
//   node tools/boxes.mjs --url=http://127.0.0.1:5199            measure all chapters
//   options: --ch=id1,id2   only these chapters
//            --step=1       seconds between samples
//            --raw=shots/boxes-raw-all.json   sample file for all chapters (--ch uses the selected IDs)
//            --from=shots/boxes-raw-all.json  skip rendering, rebuild the table from saved samples
//            --out=src/engine/framing            the folder that gets one <chapter>.json per measured chapter
// Each chapter has its own file and its own raw-sample file, so that two chapters can be measured at the same time.
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';
import { inflateSync } from 'node:zlib';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const args = readFlags();
const base = args.url ?? 'http://127.0.0.1:5199';
const STEP = +(args.step ?? 1);
if (!Number.isFinite(STEP) || STEP <= 0) throw new Error('--step must be a positive number of seconds');
const OUT = resolve(args.out ?? 'src/engine/framing');
const W = 1600, H = 900;
const ORDER = ['prologue', 'percolation', 'generality', 'meanfield', 'frustration', 'entk', 'elicitation', 'parity', 'scales', 'finale'];
const ids = args.ch ? String(args.ch).split(',') : ORDER;
const RAW = resolve(args.raw ?? `shots/boxes-raw-${ids.length === ORDER.length ? 'all' : ids.join('-')}.json`);

/** Minimal PNG decoder: 8-bit RGB or RGBA, not interlaced (what Chrome writes). Returns { w, h, bpp, data }. */
function decodePng(buf) {
  let p = 8, w = 0, h = 0, type = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p), tag = buf.toString('ascii', p + 4, p + 8), body = buf.subarray(p + 8, p + 8 + len);
    if (tag === 'IHDR') { w = body.readUInt32BE(0); h = body.readUInt32BE(4); type = body[9]; if (body[8] !== 8 || body[12] !== 0) throw new Error('unsupported PNG'); }
    else if (tag === 'IDAT') idat.push(body);
    else if (tag === 'IEND') break;
    p += 12 + len;
  }
  const bpp = type === 6 ? 4 : type === 2 ? 3 : 0;
  if (!bpp) throw new Error('unsupported PNG color type ' + type);
  const raw = inflateSync(Buffer.concat(idat)), stride = w * bpp, data = Buffer.alloc(h * stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)], src = y * (stride + 1) + 1, dst = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? data[dst + x - bpp] : 0, b = y > 0 ? data[dst + x - stride] : 0, c = x >= bpp && y > 0 ? data[dst + x - stride - bpp] : 0;
      let v = raw[src + x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      data[dst + x] = v & 255;
    }
  }
  return { w, h, bpp, data };
}

/** Box of everything that differs from the paper color: [x0, y0, x1, y1], or null for a blank frame. */
function inkBox(png, paper = [250, 249, 245], thr = 30, minCount = 4) {
  const { w, h, bpp, data } = png;
  const cols = new Uint32Array(w), rows = new Uint32Array(h);
  for (let y = 0; y < h; y++)
    for (let x = 0, i = y * w * bpp; x < w; x++, i += bpp) {
      const d = Math.max(Math.abs(data[i] - paper[0]), Math.abs(data[i + 1] - paper[1]), Math.abs(data[i + 2] - paper[2]));
      if (d > thr) { cols[x]++; rows[y]++; }
    }
  let x0 = 0, x1 = w - 1, y0 = 0, y1 = h - 1;
  while (x0 < w && cols[x0] < minCount) x0++;
  while (x1 >= 0 && cols[x1] < minCount) x1--;
  while (y0 < h && rows[y0] < minCount) y0++;
  while (y1 >= 0 && rows[y1] < minCount) y1--;
  return x1 <= x0 || y1 <= y0 ? null : [x0, y0, x1 + 1, y1 + 1];
}

async function measure() {
  const browser = await launchChrome({ chrome: args.chrome, width: W, height: H });
  let raw = {};
  try { raw = JSON.parse(readFileSync(RAW, 'utf8')); } catch { /* first run */ }
  try {
    for (const id of ids) {
      const page = await browser.newPage();
      page.on('pageerror', (e) => console.log('[pageerror]', e.message));
      // frame=base: the engine draws the figure where the scene put it, with no re-framing
      await page.goto(`${base}/?shot=1&only=${id}&ch=${id}&t=0&frame=base`, { waitUntil: 'load' });
      await page.waitForFunction('window.__shotReady === true', { timeout: 120000 });
      await page.addStyleTag({ content: '.hud {visibility:hidden!important} .hud__labels {visibility:visible!important} #boot {display:none!important}' });
      const meta = await page.evaluate((id) => {
        const e = window.__piramid, c = e.chapters[e.indexOf(id)];
        const start = e.starts[e.indexOf(id)];
        const holds = c.beats.map(b => {
          e.seek(start + b.at);
          return +(e.slideInfo().hold - start).toFixed(5);
        });
        return { dur: c.duration, beats: c.beats.map((b) => b.at), holds };
      }, id);
      const samples = [];
      const times = new Set(meta.holds);
      for (let t = 0; t <= meta.dur - 0.5; t += STEP) times.add(+t.toFixed(5));
      for (const t of [...times].sort((a,b) => a - b)) {
        await page.evaluate((id, t) => { const e = window.__piramid; e.seek(e.starts[e.indexOf(id)] + t); }, id, t);
        await page.evaluate(() => new Promise((r) => { const e = window.__piramid, f0 = e.frames; const w = () => (e.frames - f0 > 3 ? r() : requestAnimationFrame(w)); w(); }));
        const box = inkBox(decodePng(Buffer.from(await page.screenshot({ type: 'png' }))));
        const labels = await page.evaluate(() => {
          const labels = [...document.querySelectorAll('.hud__labels .lbl, .hud__hots .hot')]
            .filter((el) => el.style.display !== 'none' && Number(el.style.opacity || 1) >= 0.12)
            .map((el) => el.getBoundingClientRect());
          if (!labels.length) return null;
          return [
            Math.floor(Math.min(...labels.map((r) => r.left))),
            Math.floor(Math.min(...labels.map((r) => r.top))),
            Math.ceil(Math.max(...labels.map((r) => r.right))),
            Math.ceil(Math.max(...labels.map((r) => r.bottom))),
          ];
        });
        if (box && labels) for (let k = 0; k < 4; k++) box[k] = k < 2 ? Math.min(box[k], labels[k]) : Math.max(box[k], labels[k]);
        samples.push([t, ...(box ?? labels ?? [0, 0, 0, 0])]);
      }
      raw[id] = { ...meta, samples };
      console.log(id, `${samples.length} samples`);
      await page.close();
      mkdirSync(dirname(RAW), { recursive: true });
      writeFileSync(RAW, JSON.stringify(raw));
    }
  } finally {
    await browser.close();
  }
  return raw;
}

/**
 * One box per caption beat: the union of the samples taken once the beat has settled (the scene's own
 * camera move at the start of a beat is left out). A beat with nothing on screen inherits its neighbor.
 */
function table(raw) {
  const out = {};
  for (const id of Object.keys(raw)) {
    const { dur, beats, samples, holds } = raw[id];
    const edges = [0, ...beats.slice(1), dur];
    const keys = [];
    for (let j = 0; j + 1 < edges.length; j++) {
      const a = edges[j], b = edges[j + 1], settle = Math.min(2.5, 0.45 * (b - a));
      let box = null;
      for (const [t, x0, y0, x1, y1] of samples) {
        if (x1 <= x0 || (t !== holds?.[j] && (t < a + settle || t > b - 0.3))) continue;
        box = box ? [Math.min(box[0], x0), Math.min(box[1], y0), Math.max(box[2], x1), Math.max(box[3], y1)] : [x0, y0, x1, y1];
      }
      keys.push([+a.toFixed(2), ...(box ?? [0, 0, 0, 0])]);
    }
    // fill empty beats from the nearest measured one
    for (let j = 0; j < keys.length; j++) if (keys[j][3] <= keys[j][1]) {
      const near = keys.slice(j + 1).find((k) => k[3] > k[1]) ?? keys.slice(0, j).reverse().find((k) => k[3] > k[1]);
      if (near) keys[j] = [keys[j][0], ...near.slice(1)];
    }
    out[id] = keys;
  }
  return out;
}

const raw = args.from ? JSON.parse(readFileSync(resolve(args.from), 'utf8')) : await measure();
const tab = table(raw);
mkdirSync(OUT, { recursive: true });
for (const id of Object.keys(tab)) if (ids.includes(id)) writeFileSync(resolve(OUT, `${id}.json`), '[\n' + tab[id].map((k) => '  ' + JSON.stringify(k)).join(',\n') + '\n]\n');
for (const id of Object.keys(tab)) console.log(id.padEnd(12), tab[id].map((k) => `${k[0]}s [${k.slice(1).join(',')}]`).join('  '));
