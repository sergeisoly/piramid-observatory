import * as THREE from 'three/webgpu';
import type { Pillar } from './types';

/**
 * Colors. The film is a three-ink print on warm paper: a key ink (slate) and one ink per team
 * of the program. The paper and the key ink are the owner's blog's (sergeisoly.github.io).
 *
 * Two themes share one set of names:
 *  - 'paper' (default): marks are INK on paper. Each palette entry is an absorbance vector;
 *    the frame buffer accumulates ink and the output stage applies Beer–Lambert,
 *    color = paper · exp(−ink). Overlapping marks darken and mix like real spot inks.
 *  - 'dark' (?theme=dark): marks are LIGHT on a deep-ink background; entries are emitted color.
 * Either way a scene writes `setColor(i, C.ice, k)`: k is the amount (ink density or brightness),
 * 1 = the nominal color, below 1 a lighter tint, above 1 stronger.
 *
 * Every hue has two values on paper. `C` and `MARK` hold the pigment of a filled mark (a point, a
 * bar, a tile), chosen so that the three team inks sit at different lightness and stay apart from
 * the key ink. `CSS` holds a darker value of the same hue for text and hairlines, which need a
 * contrast of at least 4.5 : 1 against the paper.
 */
export type RGB = [number, number, number];
export type Theme = 'paper' | 'dark';

export const THEME: Theme =
  typeof location !== 'undefined' && new URLSearchParams(location.search).get('theme') === 'dark' ? 'dark' : 'paper';

export const PAPER_HEX = 0xfaf9f5;
export const DARK_BG_HEX = 0x0d1820;

/** Pigments of filled marks on paper. */
const PIGMENT = {
  ice: 0x2563ad, // Learning Theory
  cyan: 0x2c8a92, // the negative end of the signed ramp; a second cool category
  amber: 0xc0861b, // Interpretability Applications
  gold: 0xdcb560, // tint bands and fills
  violet: 0xa3669b, // Data Models and Validation
  magenta: 0xb9507e,
  white: 0x182d3b, // neutral = the key ink
  red: 0xd0582e, // failure, ablation, the wrong field
  green: 0x17654d, // verified
  slate: 0x8797a0, // baseline, control, comparator
} as const;

/** The same hues for text and hairlines on paper (each at least 4.5 : 1 against the paper). */
const TEXT = {
  ice: 0x2563ad,
  cyan: 0x1f6c73,
  amber: 0x8a5a0e,
  gold: 0x7d6118,
  violet: 0x7a4773,
  magenta: 0x99375f,
  white: 0x182d3b,
  red: 0xa43a25,
  green: 0x17654d,
  slate: 0x5a6b73,
} as const;

/** The hues lifted for the dark theme. */
const LIGHT = {
  ice: 0x7fb2e6,
  cyan: 0x74c0c2,
  amber: 0xe0a445,
  gold: 0xecc983,
  violet: 0xc79cc4,
  magenta: 0xd88bae,
  white: 0xf3f1ea,
  red: 0xe58070,
  green: 0x86c7a6,
  slate: 0x6f838c,
} as const;

export type Hue = keyof typeof PIGMENT;
const HUES = Object.keys(PIGMENT) as Hue[];

const lin = (hex: number): RGB => {
  const c = new THREE.Color(hex); // sRGB hex -> linear working space
  return [c.r, c.g, c.b];
};
export const PAPER_LIN = lin(PAPER_HEX);

/** Absorbance that turns paper into pigment `hex` at unit density. */
const absorb = (hex: number): RGB => {
  const p = lin(hex);
  return [0, 1, 2].map((i) => Math.max(0, -Math.log(Math.max(p[i], 1e-4) / PAPER_LIN[i]))) as RGB;
};
const css = (hex: number) => '#' + hex.toString(16).padStart(6, '0');

export const HEX: Record<Hue, number> = THEME === 'paper' ? { ...PIGMENT } : { ...LIGHT };

/** Palette entries for GPU marks (see the note at the top about what the numbers mean). */
export const C: Record<Hue, RGB> = Object.fromEntries(
  HUES.map((k) => [k, THEME === 'paper' ? absorb(PIGMENT[k]) : lin(LIGHT[k])]),
) as Record<Hue, RGB>;

/** The colors as CSS strings for labels and other DOM text. */
export const CSS: Record<Hue, string> = Object.fromEntries(
  HUES.map((k) => [k, css(THEME === 'paper' ? TEXT[k] : LIGHT[k])]),
) as Record<Hue, string>;

/** The colors of filled marks as CSS strings, for DOM drawings that imitate the GPU's ink (the emblem). */
export const MARK: Record<Hue, string> = Object.fromEntries(
  HUES.map((k) => [k, css(HEX[k])]),
) as Record<Hue, string>;

export const PILLAR_HUE: Record<Pillar, Hue> = {
  frame: 'white',
  theory: 'ice',
  tools: 'amber',
  data: 'violet',
};

export const PILLAR_NAME: Record<Pillar, string> = {
  frame: 'PIRAMID',
  theory: 'LEARNING THEORY',
  tools: 'INTERPRETABILITY APPLICATIONS',
  data: 'DATA MODELS AND VALIDATION',
};

/**
 * The color a stack of inks prints, as a CSS string: `parts` lists each ink and its density. It is the
 * arithmetic of the output stage (paper · exp(−ink) on paper, added light in the dark theme), so a DOM
 * drawing made with it matches what the GPU prints for the same inks.
 */
export function printed(parts: [Hue, number][]): string {
  const sum: RGB = [0, 0, 0];
  for (const [h, k] of parts) for (let i = 0; i < 3; i++) sum[i] += C[h][i] * k;
  const bg = THEME === 'paper' ? PAPER_LIN : lin(DARK_BG_HEX);
  const out = new THREE.Color().setRGB(
    ...(sum.map((v, i) => (THEME === 'paper' ? bg[i] * Math.exp(-v) : Math.min(1, bg[i] + v))) as RGB),
  );
  return '#' + out.getHexString();
}

export const scaleRGB = (c: RGB, k: number): RGB => [c[0] * k, c[1] * k, c[2] * k];
export const mixRGB = (a: RGB, b: RGB, t: number): RGB => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/**
 * The film's one diverging ramp for signed data: teal (−) → blank paper → ochre (+), t in [−1, 1].
 * The amount of ink is linear in |t|, and the two ends are matched in lightness, so that neither
 * sign looks heavier than the other.
 */
export const diverge = (t: number): RGB => {
  const a = Math.min(1, Math.abs(t));
  const base = t < 0 ? C.cyan : C.amber;
  const k = THEME === 'paper' ? 0.03 + a * (t < 0 ? 1.45 : 1.9) : 0.04 + 1.3 * a;
  return [base[0] * k, base[1] * k, base[2] * k];
};
