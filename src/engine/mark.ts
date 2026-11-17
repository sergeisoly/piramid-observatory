import { CSS, PILLAR_HUE, printed, type Hue } from './palette';
import type { Pillar } from './types';

/**
 * The film's emblem: the program seen from above.
 *
 * Draw the three teams as the corners of a triangle and give each team the triangle scaled by 3/4
 * about its own corner, printed in the team's ink. The three prints overlap. Along an edge two inks
 * mix, which is where two teams work on the same thing; in the middle all three overlap and print
 * almost black. Read as a pyramid in plan, that middle is the apex, the account all three teams
 * are after. The emblem is this film's own drawing; PIRAMID describes its teams in words.
 *
 * Geometry (unit side): Data on top, Learning Theory bottom right, Interpretability Applications
 * bottom left, so that clockwise order is the order of the chapters and of the loop the teams form.
 */
export type Corner = 'data' | 'theory' | 'tools';
export const CORNERS: Corner[] = ['data', 'theory', 'tools'];

const H = Math.sqrt(3) / 2;
/** Corners of the unit triangle (y down, as in SVG). */
export const VERT: Record<Corner, [number, number]> = { data: [0.5, 0], theory: [1, H], tools: [0, H] };
export const CENTER: [number, number] = [0.5, (2 * H) / 3];
const K = 0.75; // each team's triangle: the whole one scaled by this about the team's corner
/** Density of each team's print. Light enough for the mixtures along the sides to stay apart from the center. */
export const INK = 0.54;

/** A team's triangle, as three points of the unit triangle. */
export function teamTriangle(c: Corner, k = K): [number, number][] {
  const v = VERT[c];
  return CORNERS.map((o) => [v[0] + (VERT[o][0] - v[0]) * k, v[1] + (VERT[o][1] - v[1]) * k]);
}

export interface MarkOpts {
  /** Side of the triangle in pixels. */
  size: number;
  /** Ink density of each team's print, 0..1. Each team defaults to INK. */
  ink?: Partial<Record<Corner, number>>;
  /** Draw the outline and the three edges that run to the center. */
  lines?: boolean;
  /**
   * Show one team: the emblem is printed as always, and the regions that team's print does not reach are
   * veiled with paper. The colors are the full emblem's, so a card never shows a hue the poster lacks.
   */
  focus?: Corner;
  /** With `focus`: light the team's own corner only and veil the mixtures and the center too (the chapter cards). */
  solo?: boolean;
  /** Chapter pips: for each team, its chapters' numbers, how many are done and which one (0-based) is on show, or −1. */
  pips?: Partial<Record<Corner, { n: number; done?: number; at?: number; labels?: string[] }>>;
  /** Extra class on the <svg>. */
  cls?: string;
}

const f = (n: number) => n.toFixed(2);

/**
 * The emblem as an SVG string. Each of its seven regions is filled with the color its inks print
 * (palette.printed), so the drawing matches the emblem the GPU prints in the prologue and the finale.
 */
export function markSvg(o: MarkOpts): string {
  const s = o.size, pad = Math.max(2, s * 0.02), w = s + 2 * pad, h = s * H + 2 * pad;
  const P = (p: [number, number]) => `${f(pad + p[0] * s)},${f(pad + p[1] * s)}`;
  const k = (c: Corner) => o.ink?.[c] ?? INK;
  // a point of the triangle from its weights on the three corners
  const at = (wd: number, wt: number, wl: number): [number, number] => [
    wd * VERT.data[0] + wt * VERT.theory[0] + wl * VERT.tools[0],
    wd * VERT.data[1] + wt * VERT.theory[1] + wl * VERT.tools[1],
  ];
  const wts = (c: Corner, v: number, rest: number): [number, number, number] => CORNERS.map((x) => (x === c ? v : rest)) as [number, number, number];
  // The three prints cut the triangle into seven regions: a corner with one ink, an edge with two, the center with three.
  const a = 1 - K, m = K - a; // a: how far a print stops short of the opposite side
  let tri = '', veil = '';
  const region = (pts: [number, number][], inks: Corner[]) => {
    const parts = inks.map((c) => [PILLAR_HUE[c as Pillar], k(c)] as [Hue, number]).filter((x) => x[1] > 0);
    if (parts.length) tri += `<polygon points="${pts.map(P).join(' ')}" fill="${printed(parts)}" stroke="${printed(parts)}" stroke-width="0.6"/>`;
    if (o.focus && (o.solo ? !(inks.length === 1 && inks[0] === o.focus) : !inks.includes(o.focus))) veil += `<polygon points="${pts.map(P).join(' ')}" fill="${printed([])}" fill-opacity="0.8" stroke="${printed([])}" stroke-opacity="0.8" stroke-width="0.6"/>`;
  };
  CORNERS.forEach((c, i) => {
    const d = CORNERS[(i + 1) % 3], e = CORNERS[(i + 2) % 3];
    const w = (vc: number, vd: number, ve: number) => { const o3 = { [c]: vc, [d]: vd, [e]: ve } as Record<Corner, number>; return at(o3.data, o3.theory, o3.tools); };
    region([w(1, 0, 0), w(K, a, 0), w(m, a, a), w(K, 0, a)], [c]); // the corner of c
    region([w(K, a, 0), w(a, K, 0), w(a, m, a), w(m, a, a)], [c, d]); // the edge between c and d
  });
  region([at(...wts('data', m, a)), at(...wts('theory', m, a)), at(...wts('tools', m, a))], ['data', 'theory', 'tools']);
  const sw = Math.max(1, s / 150);
  const ghosts = veil;
  const lines = o.lines === false ? '' :
    `<g fill="none" stroke="${CSS.white}" stroke-linejoin="round" stroke-linecap="round">` +
    `<polygon points="${CORNERS.map((c) => P(VERT[c])).join(' ')}" stroke-width="${f(sw * 1.25)}"/>` +
    CORNERS.map((c) => `<line x1="${f(pad + VERT[c][0] * s)}" y1="${f(pad + VERT[c][1] * s)}" x2="${f(pad + CENTER[0] * s)}" y2="${f(pad + CENTER[1] * s)}" stroke-width="${f(sw * 0.6)}" stroke-opacity="0.55"/>`).join('') +
    `</g>`;
  let pips = '';
  if (o.pips) {
    // numbered pips need room for two digits; without numbers a pip is a dot
    const r = Math.max(2.2, s * 0.017);
    for (const c of CORNERS) {
      const p = o.pips[c];
      if (!p) continue;
      const numbered = !!p.labels && s >= 240, rr = numbered ? Math.max(9.5, s * 0.028) : r;
      // the pips sit in the team's own corner, in a row across the line from the corner towards the center
      const v = VERT[c], dx = CENTER[0] - v[0], dy = CENTER[1] - v[1], len = Math.hypot(dx, dy);
      for (let i = 0; i < p.n; i++) {
        const along = numbered ? 0.4 : 0.36, nx = -dy / len, ny = dx / len, off = (i - (p.n - 1) / 2) * (numbered ? 0.072 : 0.085);
        const x = pad + (v[0] + dx * along + nx * off) * s, y = pad + (v[1] + dy * along + ny * off) * s;
        const on = p.at === i, done = (p.done ?? 0) > i;
        if (numbered) {
          const paper = printed([]);
          pips += `<circle cx="${f(x)}" cy="${f(y)}" r="${f(rr)}" fill="${on ? CSS.white : paper}" fill-opacity="${on ? 1 : done ? 0.92 : 0.7}" stroke="${CSS.white}" stroke-width="${f(on || done ? sw : sw * 0.7)}" stroke-opacity="${on || done ? 1 : 0.5}"/>` +
            `<text x="${f(x)}" y="${f(y + rr * 0.36)}" text-anchor="middle" font-family="Inter Variable, Inter, sans-serif" font-size="${f(rr * 1.02)}" font-weight="${on ? 700 : 600}" fill="${on ? paper : CSS.white}" fill-opacity="${on || done ? 1 : 0.6}">${p.labels![i]}</text>`;
          if (on) pips += `<circle cx="${f(x)}" cy="${f(y)}" r="${f(rr + 4)}" fill="none" stroke="${CSS.white}" stroke-width="${f(sw * 0.8)}"/>`;
        } else {
          pips += `<circle cx="${f(x)}" cy="${f(y)}" r="${f(on ? r * 1.45 : r)}" fill="${on || done ? CSS.white : 'none'}" stroke="${CSS.white}" stroke-width="${f(sw * 0.9)}" ${on ? '' : 'stroke-opacity="0.7"'}/>`;
          if (on) pips += `<circle cx="${f(x)}" cy="${f(y)}" r="${f(r * 2.7)}" fill="none" stroke="${CSS.white}" stroke-width="${f(sw * 0.7)}"/>`;
        }
      }
    }
  }
  return `<svg class="mark${o.cls ? ' ' + o.cls : ''}" width="${f(w)}" height="${f(h)}" viewBox="0 0 ${f(w)} ${f(h)}" aria-hidden="true">${tri}${ghosts}${lines}${pips}</svg>`;
}

/** The corner of the emblem a chapter belongs to, if it belongs to one team. */
export function cornerOf(pillar: Pillar): Corner | undefined {
  return pillar === 'frame' ? undefined : (pillar as Corner);
}
