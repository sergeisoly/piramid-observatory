import type { Control, Params, ParamValue } from './types';

/** Coerce one raw value (possibly a URL string) to the control's type and clamp it to its range. */
export function coerce(c: Control, raw: ParamValue | undefined): ParamValue {
  if (raw === undefined) return c.value;
  if (c.kind === 'toggle') return raw === true || raw === 'true' || raw === '1' || raw === 1;
  if (c.kind === 'range') {
    let v = typeof raw === 'number' ? raw : parseFloat(String(raw));
    if (!isFinite(v)) return c.value;
    if (c.min !== undefined) v = Math.max(c.min, v);
    if (c.max !== undefined) v = Math.min(c.max, v);
    if (c.step) v = Math.round(v / c.step) * c.step;
    return +v.toFixed(8);
  }
  const hit = c.options?.find((o) => String(o.value) === String(raw));
  return hit ? hit.value : c.value;
}

/**
 * Fill in defaults and sanitize. Call this first thing in build():
 *   const P = resolveParams(factory.controls, params);  // P.beta, P.seed, …
 */
export function resolveParams(controls: Control[] | undefined, params?: Params): Record<string, any> {
  const out: Record<string, ParamValue> = {};
  for (const c of controls ?? []) out[c.key] = coerce(c, params?.[c.key]);
  return out;
}
