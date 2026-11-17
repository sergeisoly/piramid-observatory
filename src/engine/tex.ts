import katex from 'katex';
import type { Link } from './types';

/**
 * Text helpers for everything the HUD prints.
 *
 * Any visible string may contain inline maths between dollar signs, `$\lVert\Delta\rVert_2$`.
 * It is typeset with KaTeX; the rest of the string is ordinary text. Inside the dollars write
 * plain TeX (a bare `<` or `&` is fine there: the maths is cut out before the browser sees it).
 */
const cache = new Map<string, string>();
const MAX_CACHE = 4000;

export const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function math(src: string): string {
  let h = cache.get(src);
  if (h === undefined) {
    h = katex.renderToString(src, { throwOnError: false, strict: false, displayMode: false });
    if (cache.size > MAX_CACHE) cache.clear(); // live numbers inside formulas would otherwise grow it without bound
    cache.set(src, h);
  }
  return h;
}

const MATH = /\$([^$]+)\$/g;

/** For strings that are already HTML (labels, captions, hotspot bodies): typeset the `$…$` spans, leave the rest. */
export function tex(html: string): string {
  return html.indexOf('$') < 0 ? html : html.replace(MATH, (_, m: string) => math(m));
}

/** For plain strings (panel rows, hints, titles): escape the text and typeset the `$…$` spans. */
export function texPlain(text: string): string {
  if (text.indexOf('$') < 0) return esc(text);
  let out = '', i = 0;
  for (const m of text.matchAll(MATH)) {
    out += esc(text.slice(i, m.index)) + math(m[1]);
    i = m.index! + m[0].length;
  }
  return out + esc(text.slice(i));
}

const anchor = (url: string, inner: string) => `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${inner}</a>`;

/** References the film prints in a fixed form become links without the chapter saying so. */
const AUTO: [RegExp, (m: RegExpMatchArray) => string][] = [
  [/arXiv:\s?(\d{4}\.\d{4,5})/g, (m) => `https://arxiv.org/abs/${m[1]}`],
  [/princint\.ai\/piramid/g, () => 'https://princint.ai/piramid/'],
];

/**
 * Turn references inside an HTML string into links: the phrases listed in `links` (first
 * occurrence each) and the automatic patterns above. Text inside tags is never touched.
 */
export function linkify(html: string, links: Link[] = []): string {
  const spans: { a: number; b: number; url: string }[] = [];
  const taken = (a: number, b: number) => spans.some((s) => a < s.b && b > s.a);
  const inTag = (a: number) => html.lastIndexOf('<', a) > html.lastIndexOf('>', a);
  for (const l of links) {
    // a caption's hyphens are set as non-breaking ones before it is linked: look for the phrase in both spellings
    for (const text of [l.text, l.text.replace(/(\w)-(?=\w)/g, '$1\u2011')]) {
      const a = html.indexOf(text);
      if (a >= 0 && !inTag(a) && !taken(a, a + text.length)) { spans.push({ a, b: a + text.length, url: l.url }); break; }
    }
  }
  for (const [re, url] of AUTO)
    for (const m of html.matchAll(re)) {
      const a = m.index!, b = a + m[0].length;
      if (!inTag(a) && !taken(a, b)) spans.push({ a, b, url: url(m) });
    }
  if (!spans.length) return html;
  spans.sort((p, q) => p.a - q.a);
  let out = '', i = 0;
  for (const s of spans) {
    out += html.slice(i, s.a) + anchor(s.url, html.slice(s.a, s.b));
    i = s.b;
  }
  return out + html.slice(i);
}
