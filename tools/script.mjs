// The presentation script: everything a viewer reads, in playback order, with how long each caption stays.
//   node tools/script.mjs --url=http://127.0.0.1:5199 --out=../review/script.md [--frames=dir] [--budget=2.5]
// With --frames it also saves the storyboard: the opening screen, every card, every beat at the moment
// its caption is about to leave, and the end card, as dir/NN-chapter-*.jpg.
// It drives the real page (every chapter built), stops at each opening card and at the end of each
// beat, and writes down the card, the caption, the definition, the labels on the figure and the
// measured rows that are on screen there. Each caption gets its word count and its reading load
// (words per second of the time it stays up at 1× speed).
import { launchChrome } from './browser.mjs';
import { readFlags } from './flags.mjs';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const args = readFlags();
const base = args.url ?? 'http://127.0.0.1:5299';
const out = resolve(args.out ?? 'review/script.md');
const framesDir = args.frames ? resolve(args.frames) : null;
if (framesDir) mkdirSync(framesDir, { recursive: true });

const browser = await launchChrome({ chrome: args.chrome });

// Text of an element as a reader sees it: maths as its TeX source between dollar signs.
const TEXT_FN = `(el) => {
  if (!el) return '';
  const c = el.cloneNode(true);
  c.querySelectorAll('.katex').forEach((k) => { const a = k.querySelector('annotation'); k.replaceWith(document.createTextNode(a ? '$' + a.textContent + '$' : k.textContent)); });
  c.querySelectorAll('br').forEach((b) => b.replaceWith(document.createTextNode(' ')));
  c.querySelectorAll('.src').forEach((s) => s.replaceWith(document.createTextNode('[' + s.textContent.toUpperCase() + ']')));
  return c.textContent.replace(/\\u00a0/g, ' ').replace(/\\s+/g, ' ').trim();
}`;

const words = (s) => s.replace(/\$[^$]*\$/g, ' x ').split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
// What a viewer who reads at `budget` words per second gets of a text that stays up for `stay` seconds.
const budget = +(args.budget ?? 2.5);
const cut = (s, stay) => { const tok = s.match(/\$[^$]*\$\S*|\S+/g) ?? [], n = Math.floor(budget * stay); return tok.length <= n ? s : tok.slice(0, n).join(' ') + ' …[cut off]'; };
let md = '', x1 = '';
try {
  // the opening screen
  let page = await browser.newPage();
  await page.goto(`${base}/`, { waitUntil: 'load' });
  await page.waitForFunction("document.querySelector('.boot__go')?.classList.contains('is-armed')", { timeout: 180000 });
  if (framesDir) { await new Promise((r) => setTimeout(r, 700)); await page.screenshot({ path: resolve(framesDir, '00-poster.jpg'), type: 'jpeg', quality: 88 }); }
  const poster = await page.evaluate(`(${TEXT_FN})(document.querySelector('#boot'))`);
  await page.close();

  page = await browser.newPage();
  await page.goto(`${base}/?shot=1&ch=prologue&t=0`, { waitUntil: 'load' });
  await page.waitForFunction('window.__shotReady === true', { timeout: 180000 });
  const film = await page.evaluate(() => {
    const e = window.__piramid;
    const schedule = e.slideList();
    const chapters = e.chapters.map((c, i) => {
      const slides = schedule.filter(s => s.i === i);
      const [card] = e.intros[i] > 0 ? slides.splice(0, 1) : [];
      const timing = s => ({
        start: e.clockAt(s.a),
        hold: s.h,
        readingWindow: (s.h - s.a) / e.tempo + s.x,
      });
      return {
        id: c.id, kicker: c.kicker, title: c.title, source: c.source,
        card: card && timing(card),
        beats: c.beats.map((b, j) => ({ ...timing(slides[j]), eq: b.eq })),
        results: c.results ?? [],
      };
    });
    return { total: e.total, clockTotal: e.clockTotal, chapters };
  });
  const grab = async (T, name) => { const r = await grab0(T); if (framesDir && name) { await new Promise((q) => setTimeout(q, 700)); await page.screenshot({ path: resolve(framesDir, name + '.jpg'), type: 'jpeg', quality: 88 }); } return r; };
  const grab0 = (T) => page.evaluate(`(async () => {
    const T = ${TEXT_FN};
    const e = window.__piramid; e.seek(${T});
    await new Promise((r) => { const f0 = e.frames; const w = () => (e.frames - f0 > 8 ? r() : requestAnimationFrame(w)); w(); });
    await Promise.allSettled(document.getAnimations()
      .filter(a => a.playState === 'running' && Number.isFinite(a.effect?.getComputedTiming().endTime))
      .map(a => a.finished));
    const vis = (el) => {
      let alpha = 1;
      for (let p = el; p; p = p.parentElement) {
        const s = getComputedStyle(p);
        if (s.display === 'none' || s.visibility === 'hidden') return false;
        alpha *= Number(s.opacity);
      }
      return alpha > 0.25 && el.getBoundingClientRect().width > 0;
    };
    const card = document.querySelector('.hud__card');
    return {
      card: card && card.classList.contains('is-in') ? { n: T(card.querySelector('.card__n')), title: T(card.querySelector('.card__title')), line: T(card.querySelector('.card__line')), source: T(card.querySelector('.card__source')) } : null,
      q: T(document.querySelector('.head__q')),
      caption: T(document.querySelector('.hud__caption p')),
      labels: [...document.querySelectorAll('.hud__labels .lbl')].filter(vis).map(T).filter(Boolean),
      rows: [...document.querySelectorAll('.hud__tele .rows .row')].filter(vis).map((r) => ({ l: T(r.querySelector('.l')), v: T(r.querySelector('.v')), n: T(r.querySelector('.n')) })).filter((r) => r.l),
      note: T(document.querySelector('.hud__tele .note')),
      end: document.querySelector('.hud__end') && vis(document.querySelector('.hud__end')) ? T(document.querySelector('.hud__end .end__body')) : '',
    };
  })()`);

  const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  md += `# PIRAMID Observatory presentation script\n\nRuntime is ${mmss(film.clockTotal)} at 1× speed. Timestamps follow the player clock, including scheduled reading waits. Math is shown as TeX.\n`;
  md += `Reading load counts caption words from the start of the slide through its scheduled hold. A formula counts as one word. Any time the caption lingers during the transition is excluded.\n\n`;
  md += `## Opening screen\n\n${poster}\n\n`;
  x1 += `# What a viewer reads at 1× speed\n\nThe presentation’s captions and cards, each cut to the words a reader gets through at ${budget} words per second in the time the text stays on screen in film mode, reading waits included. Labels, numbers and the opening screen are left out. "[cut off]" marks where the text was still unread when it left the screen.\n\n`;
  const table = [];
  let n = 0;
  for (const c of film.chapters) {
    md += `## ${String(n).padStart(2, '0')} ${c.title}  (${c.id}; ${c.kicker})\n\nSource line: ${c.source}\n\n`;
    if (c.card) {
      const g = await grab(c.card.hold, `${String(n).padStart(2, '0')}-${c.id}-card`);
      if (g.card) md += `**Opening card.** ${g.card.title}. ${g.card.line}\n\n`;
      x1 += `## ${c.title}\n\n`;
      if (g.card) { const ct = `${g.card.title}. ${g.card.line}`, cs = c.card.readingWindow; x1 += `Card (${cs.toFixed(1)} s): ${cut(ct, cs)}\n\n`; }
    }
    for (let j = 0; j < c.beats.length; j++) {
      const b = c.beats[j];
      const g = await grab(b.hold, `${String(n).padStart(2, '0')}-${c.id}-b${j + 1}`);
      const w = words(g.caption), stay = b.readingWindow, load = w / stay;
      table.push({ ch: c.id, j: j + 1, at: b.start, stay, w, load });
      if (j === 0 && !c.card) x1 += `## ${c.title}\n\n`;
      x1 += `- ${cut(g.caption, stay)}\n`;
      md += `### ${c.id} beat ${j + 1}  (${mmss(b.start)}, reading window ${stay.toFixed(1)} s, ${w} words, load ${load.toFixed(2)} w/s)\n\n`;
      if (j === 0 && g.q) md += `Running head question: ${g.q}\n\n`;
      md += `**Caption.** ${g.caption}\n\n`;
      if (b.eq) md += `**Definition corner.** $${b.eq}$\n\n`;
      if (g.labels.length) md += `**Labels on the figure.** ${g.labels.map((l) => `"${l}"`).join(' · ')}\n\n`;
      if (g.rows.length) md += `**Measured** (${g.note}):\n\n${g.rows.map((r) => `- ${r.l}: ${r.v}${r.n ? ` (${r.n})` : ''}`).join('\n')}\n\n`;
    }
    if (c.results.length) md += `**Results the chapter reports to the closing board.**\n\n${c.results.map((r) => `- ${r.label}: measured ${r.measured}; reference ${r.reference}${r.claim ? `; claim: ${r.claim}` : ''}${r.headline ? ' (headline)' : ''}`).join('\n')}\n\n`;
    x1 += '\n';
    n++;
  }
  const g = await grab(film.total - 0.01, '99-end');
  if (g.end) md += `## End card\n\n${g.end}\n\n`;

  const W = table.reduce((s, r) => s + r.w, 0);
  const sorted = [...table].sort((a, b) => b.load - a.load);
  md += `## Reading load\n\n${table.length} captions, ${W} words, mean ${(W / table.length).toFixed(1)} words per caption, mean load ${(table.reduce((s, r) => s + r.load, 0) / table.length).toFixed(2)} w/s.\n`;
  md += `Captions above 2.5 w/s (150 words per minute): ${table.filter((r) => r.load > 2.5).length}. Above 3.0 w/s: ${table.filter((r) => r.load > 3).length}. Longest caption: ${Math.max(...table.map((r) => r.w))} words.\n\n`;
  md += `| chapter | beat | at | stays (s) | words | load (w/s) |\n| --- | --- | --- | --- | --- | --- |\n${table.map((r) => `| ${r.ch} | ${r.j} | ${mmss(r.at)} | ${r.stay.toFixed(1)} | ${r.w} | ${r.load.toFixed(2)} |`).join('\n')}\n\n`;
  md += `Heaviest five: ${sorted.slice(0, 5).map((r) => `${r.ch} ${r.j} (${r.load.toFixed(2)})`).join(', ')}.\n`;
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, md);
  writeFileSync(out.replace(/\.md$/, '') + '-1x.md', x1);
  console.log('saved', out, `${table.length} captions, ${W} words`);
} catch (e) {
  console.log('[script failed]', e.message);
  process.exitCode = 1;
} finally {
  await browser.close();
}
