/** Small, dependency-free numerics shared by the chapters. Everything is deterministic. */

/** Seeded uniform RNG in [0, 1). */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sampler built on a uniform RNG (Box–Muller). */
export function gaussian(rnd: () => number) {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    v = rnd();
    const r = Math.sqrt(-2 * Math.log(u)), a = 2 * Math.PI * v;
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

export const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** 0 below a, 1 above b, smooth in between. Works for a > b too. */
export const smooth = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
/** Trapezoid window: fades in over [a, a+f], holds, fades out over [b−f, b]. */
export const windowed = (x: number, a: number, b: number, f = 0.6) => smooth(a, a + f, x) * (1 - smooth(b - f, b, x));
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Yield to the browser so the boot screen can repaint between heavy steps. */
export const breathe = () => new Promise<void>((r) => setTimeout(r, 0));

/** Compact number formatting for readouts. */
export const fmt = (x: number, digits = 3) => {
  if (!isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(digits);
};
