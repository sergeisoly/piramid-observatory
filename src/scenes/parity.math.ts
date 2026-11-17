/**
 * Numerics for the `parity` chapter (Deep Parity Bottleneck, arXiv:2607.20652). Callers supply the seeded
 * samplers. The code has no DOM dependencies and can also run under Node.
 *
 *   seed matrix   M ∈ F₂^{d×2t}, d = 2^t; row k = (coord(ξ_k), coord(ξ_k⁻¹)), ξ_k ∈ F_{2^t}   (paper eq. 22)
 *   hash          φ̃_i = R(M bits(i)) / √d,  R: 0 ↦ +1, 1 ↦ −1                               (paper eq. 1)
 *   coherence     ⟨φ̃_i, φ̃_j⟩ = Ŵ(i ⊕ j) / d, Ŵ = Walsh–Hadamard transform of the multiset of rows of M
 *   encoder       level 0 keeps the K0 largest |x_k|; children of parent i are i ⊕ g_j; z = Γ (φ̃_i ⊙ x)
 */

export const popcount = (x: number): number => {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
};

/** Irreducible polynomials over F₂ used for F_{2^t} (bit t is the leading term). `text` is TeX. */
export const FIELD: Record<number, { poly: number; text: string }> = {
  6: { poly: 0x43, text: 'z^6+z+1' },
  8: { poly: 0x11b, text: 'z^8+z^4+z^3+z+1' },
  10: { poly: 0x409, text: 'z^{10}+z^3+1' },
};

function gfMul(a: number, b: number, t: number, poly: number) {
  let r = 0;
  while (b) {
    if (b & 1) r ^= a;
    b >>>= 1;
    a <<= 1;
    if (a & (1 << t)) a ^= poly;
  }
  return r;
}

/**
 * Kloosterman seed matrix as d row masks: bit p of M[k] is M_{k,p}. The low t bits are the
 * coordinates of ξ_k (the binary expansion of k), the high t bits those of ξ_k⁻¹ (0⁻¹ := 0).
 */
export function seedMatrix(t: number): Int32Array {
  const d = 1 << t, poly = FIELD[t].poly;
  const M = new Int32Array(d);
  for (let x = 1; x < d; x++) {
    let r = 1, b = x, e = d - 2; // ξ⁻¹ = ξ^(2^t − 2)
    while (e) {
      if (e & 1) r = gfMul(r, b, t, poly);
      b = gfMul(b, b, t, poly);
      e >>>= 1;
    }
    if (gfMul(x, r, t, poly) !== 1) throw new Error('field inverse failed');
    M[x] = x | (r << t);
  }
  return M;
}

/** In-place fast Walsh–Hadamard transform (unnormalized). Length must be a power of two. */
export function wht<T extends Int32Array | Float64Array>(a: T): T {
  const n = a.length;
  for (let h = 1; h < n; h <<= 1)
    for (let i = 0; i < n; i += h << 1)
      for (let j = i; j < i + h; j++) {
        const x = a[j], y = a[j + h];
        a[j] = x + y;
        a[j + h] = x - y;
      }
  return a;
}

/** One attainable value of ⟨φ̃_i, φ̃_j⟩ and how many index differences i ⊕ j ≠ 0 produce it. */
export interface Atom {
  ip: number; // inner product
  count: number; // number of differences u = i ⊕ j ≠ 0 with this value
  mass: number; // fraction of all pairs
  density: number; // mass / spacing between atoms (4/d)
}

export interface Coherence {
  t: number;
  d: number;
  features: number; // d² hashed directions
  pairs: number; // unordered pairs i ≠ j
  bound: number; // (1 + 2√d) / d
  max: number; // measured max |⟨φ̃_i, φ̃_j⟩|
  min: number; // most negative inner product
  atoms: Atom[]; // ascending in ip
}

/**
 * The complete distribution of pairwise inner products of the d² hashed directions. Because
 * ⟨φ̃_i, φ̃_j⟩ depends only on u = i ⊕ j, one WHT of length d² covers every pair.
 */
export function coherence(t: number): Coherence {
  const d = 1 << t, n = d * d, M = seedMatrix(t);
  const f = new Int32Array(n);
  for (let k = 0; k < d; k++) f[M[k]]++;
  wht(f);
  const counts = new Float64Array(2 * d + 1);
  let mx = 0, mn = 0;
  for (let u = 1; u < n; u++) {
    const w = f[u];
    counts[w + d]++;
    if (w > mx) mx = w;
    if (w < mn) mn = w;
  }
  const atoms: Atom[] = [];
  for (let w = -d; w <= d; w++) {
    const c = counts[w + d];
    if (c > 0) atoms.push({ ip: w / d, count: c, mass: c / (n - 1), density: c / (n - 1) / (4 / d) });
  }
  return {
    t, d, features: n, pairs: (n * (n - 1)) / 2,
    bound: (1 + 2 * Math.sqrt(d)) / d,
    max: Math.max(mx, -mn) / d, min: mn / d, atoms,
  };
}

/** Sign pattern √d·φ̃_index restricted to the index bits in `mask` (running parity). */
export function hashSigns(M: Int32Array, index: number, out: Int8Array, mask = -1): Int8Array {
  const q = index & mask;
  for (let k = 0; k < M.length; k++) out[k] = popcount(M[k] & q) & 1 ? -1 : 1;
  return out;
}

/** ⟨φ̃_i, φ̃_j⟩ by direct summation over the d coordinates. */
export function innerProduct(M: Int32Array, i: number, j: number): number {
  const u = i ^ j;
  let s = 0;
  for (let k = 0; k < M.length; k++) s += popcount(M[k] & u) & 1 ? -1 : 1;
  return s / M.length;
}

/**
 * Check R(Mu) ⊙ R(Mv) = R(M(u ⊕ v)) coordinate by coordinate on `pairs` random index pairs.
 * Returns the number of coordinates (out of pairs·d) on which the two sides differ.
 */
export function xorIdentityMismatches(M: Int32Array, bits: number, pairs: number, rnd: () => number): number {
  const d = M.length, n = 2 ** bits;
  const a = new Int8Array(d), b = new Int8Array(d), c = new Int8Array(d);
  let bad = 0;
  for (let q = 0; q < pairs; q++) {
    const u = Math.floor(rnd() * n), v = Math.floor(rnd() * n);
    hashSigns(M, u, a); hashSigns(M, v, b); hashSigns(M, u ^ v, c);
    for (let k = 0; k < d; k++) if (a[k] * b[k] !== c[k]) bad++;
  }
  return bad;
}

/** `count` distinct XOR generators over lowBits + highBits index bits, each with a non-zero high part. */
export function makeGenerators(rnd: () => number, count: number, lowBits: number, highBits: number): number[] {
  const g: number[] = [];
  const space = 1 << (lowBits + highBits);
  for (let guard = 0; g.length < count && guard < 200000; guard++) {
    const c = Math.floor(rnd() * space);
    if (c >> lowBits === 0 || g.includes(c)) continue;
    g.push(c);
  }
  return g;
}

export interface Candidate {
  pi: number; // which kept parent (0 … K0−1)
  parent: number; // its coordinate / level-0 index
  j: number; // generator
  child: number; // parent ⊕ g_j
  z: number; // score from the matrix product
  slot: number; // index into `distinct`
}
export interface Scored {
  child: number;
  z: number;
  kept: boolean;
  rank: number; // by |z| among distinct children
  from: number[]; // candidate indices that proposed it
}
export interface Encoding {
  parents: number[]; // K0 coordinates, by decreasing |x_k|
  cands: Candidate[]; // K0·Δ scored (parent, generator) pairs
  distinct: Scored[]; // de-duplicated children, by decreasing |z|
  kept: Scored[]; // top K1
  maxErr: number; // max |z − ⟨φ̃_child, x⟩| over all candidates
}

/** Two-level hierarchical encoder: beam search with one Γ·(φ̃_i ⊙ x) product per kept parent. */
export function encode(M: Int32Array, gens: number[], x: Float64Array, K0: number, K1: number): Encoding {
  const d = M.length, inv = 1 / Math.sqrt(d), D = gens.length;
  const gamma = new Int8Array(D * d); // rows γ_j = R(M g_j)
  for (let j = 0; j < D; j++) hashSigns(M, gens[j], gamma.subarray(j * d, (j + 1) * d));
  const order = Array.from({ length: d }, (_, k) => k).sort((a, b) => Math.abs(x[b]) - Math.abs(x[a]) || a - b);
  const parents = order.slice(0, Math.max(1, Math.min(K0, d)));
  const cands: Candidate[] = [];
  const v = new Float64Array(d), sp = new Int8Array(d), sc = new Int8Array(d);
  let maxErr = 0;
  parents.forEach((p, pi) => {
    hashSigns(M, p, sp);
    for (let k = 0; k < d; k++) v[k] = sp[k] * inv * x[k]; // φ̃_p ⊙ x
    for (let j = 0; j < D; j++) {
      let z = 0;
      const row = j * d;
      for (let k = 0; k < d; k++) z += gamma[row + k] * v[k];
      const child = p ^ gens[j];
      hashSigns(M, child, sc);
      let direct = 0;
      for (let k = 0; k < d; k++) direct += sc[k] * inv * x[k];
      maxErr = Math.max(maxErr, Math.abs(z - direct));
      cands.push({ pi, parent: p, j, child, z, slot: -1 });
    }
  });
  const byChild = new Map<number, Scored>();
  cands.forEach((c, ci) => {
    let s = byChild.get(c.child);
    if (!s) byChild.set(c.child, (s = { child: c.child, z: c.z, kept: false, rank: 0, from: [] }));
    s.from.push(ci);
  });
  const distinct = [...byChild.values()].sort((a, b) => Math.abs(b.z) - Math.abs(a.z) || a.child - b.child);
  distinct.forEach((s, r) => {
    s.rank = r;
    s.kept = r < K1;
    for (const ci of s.from) cands[ci].slot = r;
  });
  return { parents, cands, distinct, kept: distinct.slice(0, K1), maxErr };
}

/** Scores ⟨φ̃_u, x⟩ of every index u < 2^bits in one WHT (the exhaustive encoder the beam search avoids). */
export function allScores(M: Int32Array, x: Float64Array, bits: number): Float64Array {
  const d = M.length, inv = 1 / Math.sqrt(d), mask = (1 << bits) - 1;
  const f = new Float64Array(1 << bits);
  for (let k = 0; k < d; k++) f[M[k] & mask] += x[k] * inv;
  return wht(f);
}

export interface Planted {
  x: Float64Array;
  children: { index: number; coef: number; parent: number; j: number }[];
  parents: { index: number; coef: number }[];
  coParent: boolean; // the 4th parent is a second parent of children[0]
}

/**
 * x = Σ coef_c φ̃_c (planted level-1 features) + Σ a_p e_p (their parents) + σ·noise.
 * Each planted child is p ⊕ g_j for a planted parent p. If two generators share the high bits of
 * the first child, its second parent is planted too (the hierarchy is a DAG, not a tree).
 */
export function plant(
  M: Int32Array, gens: number[], lowBits: number, coefs: number[], pcoefs: number[], sigma: number,
  rnd: () => number, gauss: () => number,
): Planted {
  const d = M.length, inv = 1 / Math.sqrt(d), D = gens.length;
  const ps: number[] = [], js: number[] = [], cs: number[] = [];
  for (let guard = 0; cs.length < coefs.length && guard < 10000; guard++) {
    const p = Math.floor(rnd() * d), j = Math.floor(rnd() * D), c = p ^ gens[j];
    if (ps.includes(p) || cs.includes(c)) continue;
    ps.push(p); js.push(j); cs.push(c);
  }
  // second parent of the first child, through another generator with the same high bits
  let coParent = false, p4 = -1;
  const same: number[] = [];
  for (let j = 0; j < D; j++) if (j !== js[0] && gens[j] >> lowBits === gens[js[0]] >> lowBits) same.push(j);
  if (same.length) {
    const q = cs[0] ^ gens[same[Math.floor(rnd() * same.length)]];
    if (q >> lowBits === 0 && !ps.includes(q)) { p4 = q; coParent = true; }
  }
  for (let guard = 0; p4 < 0 && guard < 10000; guard++) {
    const q = Math.floor(rnd() * d);
    if (!ps.includes(q)) p4 = q;
  }
  const x = new Float64Array(d), s = new Int8Array(d);
  cs.forEach((c, n) => {
    hashSigns(M, c, s);
    for (let k = 0; k < d; k++) x[k] += coefs[n] * s[k] * inv;
  });
  const parents = [...ps, p4].filter((p) => p >= 0).map((p, n) => ({ index: p, coef: pcoefs[n % pcoefs.length] }));
  for (const p of parents) x[p.index] += p.coef;
  for (let k = 0; k < d; k++) x[k] += sigma * gauss();
  return { x, children: cs.map((c, n) => ({ index: c, coef: coefs[n], parent: ps[n], j: js[n] })), parents, coParent };
}
